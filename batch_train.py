"""
batch_train.py — Unattended data generation and model training
==============================================================
Run this script while you are away.  It will:

  1. Generate TILES_2025 image/mask pairs from the 2025 WMS layer
       →  training_data/2025/images/  +  training_data/2025/masks/
  2. Train one U-Net on those pairs
       →  models/unet_2025_<timestamp>.pt

WHY ONLY 2025?
  Helsinki's WFS building-footprint service always returns current (2025)
  footprints.  Training on older imagery with today's masks produces
  corrupted labels — the model is penalised for correct predictions.
  Augmentation (blur, gamma, colour jitter) in the training loop makes
  the resulting model robust to the visual style of older photographs.

Everything is logged to the console AND to batch_train.log in this folder.
The script resumes safely if interrupted — existing complete tile pairs are
counted and skipped automatically.

Usage
-----
  py -3.12 batch_train.py

Optional tweaks (edit the CONFIG block near the top of this file):
  TILES_2025  — how many image/mask pairs to generate (default 6000)
  EPOCHS      — training epochs (default 20)
  TILE_SIZE   — U-Net tile size in pixels (default 256 = standard model)
  BATCH_SIZE  — mini-batch size during training (default 8)
"""

from __future__ import annotations

# ── Standard library ─────────────────────────────────────────────────────────
import json           # read/write JSON (for WFS responses)
import logging        # write timestamped log messages to file and console
import os             # environment variable access
import random         # reproducible random number generation
import sys            # sys.exit on fatal errors
import time           # throttle API requests
from datetime import datetime          # timestamp model filenames
from pathlib import Path               # cross-platform file paths
from urllib.parse import urlencode     # build URL query strings

# ── Third-party ───────────────────────────────────────────────────────────────
import cv2             # OpenCV — image I/O and processing
import numpy as np     # array maths
import requests        # HTTP requests to WMS / WFS
from PIL import Image, ImageDraw       # rasterise building polygons
from shapely.geometry import shape     # GeoJSON → Shapely geometry
from shapely.ops import unary_union    # merge many polygons into one
import torch                           # deep-learning framework
from torch.utils.data import DataLoader, Subset  # data loading for training

# ── Local module ──────────────────────────────────────────────────────────────
# pipeline.py lives in the same folder as this script.
# We add its directory to sys.path so Python can find it.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pipeline as P   # U-Net model, training loop, dataset class

# =============================================================================
# CONFIG — edit these values to change the run
# =============================================================================
TILES_2025    = 6000    # tiles fetched exclusively from the 2025 WMS layer.
                        # The WFS masks are current (2025), so image and mask
                        # are from the same year — perfectly accurate labels.
EPOCHS        = 20      # training epochs
TILE_SIZE     = 256     # U-Net sliding-window tile size (256 = standard model)
BATCH_SIZE    = 8       # mini-batch size during training
MIN_BUILDING_PCT = 3.0  # minimum % of tile that must be buildings to keep it
HARD_NEG_EDGE_THR = 0.04  # keep low-building tile if edge density exceeds this

# =============================================================================
# HELSINKI API CONSTANTS  (match app.py exactly)
# =============================================================================
WMS_BASE       = "https://kartta.hel.fi/ws/geoserver/avoindata/wms"
WFS_BASE       = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
BUILDING_LAYER = "avoindata:Rakennukset_alue"
CRS            = "EPSG:3879"
TILE_PX        = 1024      # output image size in pixels
GROUND_RES     = 0.5       # metres per pixel at WMS output

# Helsinki city centre sampling box (EPSG:3879 metres)
SAMPLE_X_MIN = 25494000
SAMPLE_X_MAX = 25502000
SAMPLE_Y_MIN = 6671000
SAMPLE_Y_MAX = 6677000

# Water-rejection thresholds
WATER_BLUE_LEAD = 8
WATER_MAX_STD   = 28
MAX_BUILDING_PCT = 95.0

# WMS layers grouped by era — same dict as in app.py
# Only the 2025 layer is used for training — it is the only year where the
# WFS building masks and the aerial imagery are from the same point in time.
TRAINING_LAYERS_BY_ERA: dict[str, list[str]] = {
    "2025": [
        "avoindata:Ortoilmakuva_2025_5cm",   # 5 cm/px — highest available resolution
        "avoindata:Ortoilmakuva",             # "latest" alias, also resolves to 2025
    ],
}

# =============================================================================
# LOGGING — writes to console AND to batch_train.log
# =============================================================================
LOG_PATH = HERE / "batch_train.log"

# Set up a logger named "batch".
log = logging.getLogger("batch")
log.setLevel(logging.DEBUG)

# Format: "2025-05-25 14:32:01  INFO  Generating modern tile 3/2000"
fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                         datefmt="%Y-%m-%d %H:%M:%S")

# Handler 1: write every message to the log file.
fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
fh.setFormatter(fmt)
log.addHandler(fh)

# Handler 2: also print to the console so you can watch progress live.
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(ch)

# =============================================================================
# WMS / WFS HELPER FUNCTIONS  (standalone copies of the app.py helpers)
# =============================================================================

def _wms_url(bbox, layer: str) -> str:
    """Build a Helsinki WMS GetMap URL for the given bounding box and layer."""
    minx, miny, maxx, maxy = bbox
    return f"{WMS_BASE}?" + urlencode({
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": layer, "STYLES": "", "SRS": CRS,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": str(TILE_PX), "HEIGHT": str(TILE_PX),
        "FORMAT": "image/png",
    })


def _fetch_image(bbox, path: Path, layer: str) -> bool:
    """Download one WMS tile to disk.  Returns True on success."""
    try:
        r = requests.get(_wms_url(bbox, layer), timeout=30)
        if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
            return False
        path.write_bytes(r.content)
        return True
    except requests.RequestException:
        return False


def _is_water(image_path: Path) -> bool:
    """Return True if the tile looks like open water (skip these)."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    # Blue channel dominant AND very uniform colour → water
    return (float(img[:, :, 0].mean()) > float(img[:, :, 2].mean()) + WATER_BLUE_LEAD
            and float(img.std()) < WATER_MAX_STD)


def _fetch_polygons(bbox):
    """Fetch building footprint polygons from Helsinki WFS.  Returns list or None."""
    minx, miny, maxx, maxy = bbox
    try:
        r = requests.get(f"{WFS_BASE}?" + urlencode({
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": BUILDING_LAYER, "SRSNAME": CRS,
            "BBOX": f"{minx},{miny},{maxx},{maxy},{CRS}",
            "OUTPUTFORMAT": "application/json",
        }), timeout=30)
        if r.status_code != 200:
            return None
        return json.loads(r.text).get("features", [])
    except (requests.RequestException, json.JSONDecodeError):
        return None


def _rasterize(features, bbox) -> np.ndarray:
    """Paint building polygons onto a binary mask image."""
    minx, miny, maxx, maxy = bbox
    dx, dy = maxx - minx, maxy - miny

    canvas = Image.new("L", (TILE_PX, TILE_PX), 0)
    draw   = ImageDraw.Draw(canvas)

    def to_px(x, y):
        return ((x - minx) / dx * TILE_PX,
                TILE_PX - (y - miny) / dy * TILE_PX)

    geoms = []
    for feat in features:
        try:
            geom = shape(feat["geometry"]).buffer(0)
            if geom.area >= 20:
                geoms.append(geom.buffer(0.15))
        except Exception:
            continue

    if not geoms:
        return np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)

    merged = unary_union(geoms)
    polys  = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)

    for poly in polys:
        draw.polygon([to_px(x, y) for x, y in poly.exterior.coords], fill=255)
        for hole in poly.interiors:
            draw.polygon([to_px(x, y) for x, y in hole.coords], fill=0)

    mask   = np.array(canvas)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_era_data(era: str, n_tiles: int, out_dir: Path) -> int:
    """
    Generate `n_tiles` image/mask pairs for the given era.
    Tiles are saved to out_dir/images/ and out_dir/masks/.
    Resumes automatically if out_dir already contains completed pairs.
    Returns the total number of complete pairs in out_dir after the run.
    """
    images_dir = out_dir / "images"
    masks_dir  = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    layers = TRAINING_LAYERS_BY_ERA[era]   # list of WMS layer names for this era
    tile_m = TILE_PX * GROUND_RES          # tile footprint in metres (512 m)

    # ── Resume logic ────────────────────────────────────────────────────────
    def _tile_nums(d: Path) -> set:
        """Return the set of tile index numbers found in a directory."""
        nums = set()
        for p in d.glob("tile_*.png"):
            parts = p.stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                nums.add(int(parts[1]))
        return nums

    # Only count pairs where BOTH image and mask are present.
    complete    = _tile_nums(images_dir) & _tile_nums(masks_dir)
    start_index = max(complete, default=0)

    # Clean up orphaned images (image downloaded but mask never written).
    orphans = _tile_nums(images_dir) - complete
    for idx in orphans:
        (images_dir / f"tile_{idx:04d}.png").unlink(missing_ok=True)

    if start_index >= n_tiles:
        log.info(f"[{era}] Already have {start_index} pairs — skipping generation.")
        return start_index

    if start_index:
        log.info(f"[{era}] Resuming from tile {start_index + 1} "
                 f"({start_index} complete pairs found).")
    else:
        log.info(f"[{era}] Starting fresh — target: {n_tiles} pairs.")

    # ── Random number generator ──────────────────────────────────────────────
    # Seeded with 42 + a per-era offset so each era gets different tile positions.
    # Each era gets a unique seed so tile positions don't repeat across runs.
    era_seed = {"2025": 42, "modern": 42, "auto": 142}.get(era, 42)
    rng = random.Random(era_seed)
    # Fast-forward past tiles that are already done.
    # Each tile consumes: 2 coordinate calls + 1 layer choice = 3 calls.
    for _ in range(start_index * 3):
        rng.random()

    # ── Rejection counters ───────────────────────────────────────────────────
    rej     = {"fetch": 0, "water": 0, "wfs": 0, "dense": 0, "empty": 0}
    count   = 0    # new pairs saved this run
    attempt = 0    # total positions tried this run

    while count < (n_tiles - start_index):
        attempt += 1

        # Random tile position inside Helsinki sampling area
        x    = rng.uniform(SAMPLE_X_MIN, SAMPLE_X_MAX - tile_m)
        y    = rng.uniform(SAMPLE_Y_MIN, SAMPLE_Y_MAX - tile_m)
        bbox = (x, y, x + tile_m, y + tile_m)

        # Pick a random layer from this era's pool
        layer      = layers[int(rng.random() * len(layers))]
        layer_year = layer.split("_")[-1] if "_" in layer else "latest"

        tile_idx  = start_index + count + 1
        name      = f"tile_{tile_idx:04d}"
        img_path  = images_dir / f"{name}.png"
        mask_path = masks_dir  / f"{name}.png"

        # Never overwrite a complete pair
        if img_path.exists() and mask_path.exists():
            count += 1
            continue

        total_done = start_index + count + 1
        prefix = f"[{era}] {total_done}/{n_tiles} {name} [{layer_year}]"

        # ── Fetch aerial image ──────────────────────────────────────────────
        if not _fetch_image(bbox, img_path, layer):
            log.warning(f"{prefix}  SKIP fetch failed")
            rej["fetch"] += 1
            continue

        # ── Reject water tiles ──────────────────────────────────────────────
        if _is_water(img_path):
            img_path.unlink(missing_ok=True)
            log.debug(f"{prefix}  SKIP water")
            rej["water"] += 1
            continue

        # ── Fetch building polygons ─────────────────────────────────────────
        features = _fetch_polygons(bbox)
        if features is None:
            img_path.unlink(missing_ok=True)
            log.warning(f"{prefix}  SKIP WFS failed")
            rej["wfs"] += 1
            continue

        # ── Build mask and check coverage ───────────────────────────────────
        mask = _rasterize(features, bbox)
        pct  = mask.mean() / 255 * 100

        if pct > MAX_BUILDING_PCT:
            img_path.unlink(missing_ok=True)
            log.debug(f"{prefix}  SKIP too dense ({pct:.1f}%)")
            rej["dense"] += 1
            continue

        if pct < MIN_BUILDING_PCT:
            # Keep as a hard-negative example only if it has useful edge structure
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            ed   = cv2.Canny(gray, 50, 150).mean() / 255.0
            if ed <= HARD_NEG_EDGE_THR:
                img_path.unlink(missing_ok=True)
                log.debug(f"{prefix}  SKIP empty (edges={ed:.3f})")
                rej["empty"] += 1
                continue
            log.info(f"{prefix}  ok hard-neg (edges={ed:.3f})")
        else:
            log.info(f"{prefix}  ok ({pct:.1f}% buildings)")

        # ── Save mask ───────────────────────────────────────────────────────
        cv2.imwrite(str(mask_path), mask)
        count += 1

        # Throttle to avoid hammering the server
        time.sleep(0.15)

    total = len(list(images_dir.glob("*.png")))
    rate  = count / max(attempt, 1) * 100
    log.info(f"[{era}] Generation done — {count} new, {total} total, "
             f"{rate:.1f}% accepted | "
             f"fetch:{rej['fetch']} water:{rej['water']} "
             f"wfs:{rej['wfs']} dense:{rej['dense']} empty:{rej['empty']}")
    return total

# =============================================================================
# TRAINING
# =============================================================================

def train_era_model(era: str, data_dir: Path, device_str: str) -> Path:
    """
    Train a U-Net on the tiles in data_dir/images + data_dir/masks.
    Saves the checkpoint to models/unet_{era}_{timestamp}.pt and returns its path.
    """
    images_dir = data_dir / "images"
    masks_dir  = data_dir / "masks"

    log.info(f"[{era}] Building dataset from {data_dir} ...")

    # Full dataset with augmentation (for training)
    full = P.BuildingDataset(images_dir, masks_dir,
                             tile_size=TILE_SIZE, augment=True)
    # Same dataset without augmentation (for validation)
    fval = P.BuildingDataset(images_dir, masks_dir,
                             tile_size=TILE_SIZE, augment=False)

    n = len(full)
    if n < 5:
        raise RuntimeError(f"[{era}] Need at least 5 pairs to train, found {n}.")

    log.info(f"[{era}] Dataset: {n} pairs — splitting 80/20 train/val ...")

    # Shuffle and split 80% train / 20% val
    idx   = list(range(n))
    random.Random(42).shuffle(idx)
    n_val    = max(1, int(n * 0.2))
    train_ds = Subset(full, sorted(idx[n_val:]))
    val_ds   = Subset(fval, sorted(idx[:n_val]))

    log.info(f"[{era}] Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Timestamped checkpoint filename, e.g. unet_modern_20250525_143200.pt
    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = HERE / "models"
    models_dir.mkdir(exist_ok=True)
    save_path  = models_dir / f"unet_{era}_{ts}.pt"

    log.info(f"[{era}] Training {EPOCHS} epochs  →  {save_path.name} ...")

    # Build a standard-width U-Net (32 → 64 → 128 → 256 → 512 channels)
    # These are the same "FULL_WIDTHS" used in app.py for the standard model.
    STANDARD_WIDTHS = (32, 64, 128, 256, 512)
    model = P.UNet(widths=STANDARD_WIDTHS)

    # Training configuration
    tc = P.TrainConfig(
        epochs          = EPOCHS,
        lr              = 1e-3,
        batch_size      = BATCH_SIZE,
        num_workers     = 0,           # 0 = main thread only (safe on Windows)
        device          = device_str,
        checkpoint_path = str(save_path),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    P.train(model, train_loader, val_loader, tc)

    log.info(f"[{era}] Training complete — saved to {save_path}")
    return save_path

# =============================================================================
# DEVICE SELECTION
# =============================================================================

def pick_device() -> str:
    """
    Choose the best available compute device.
    Tries DirectML (AMD GPU) first, then CUDA, then falls back to CPU.
    """
    try:
        import torch_directml          # AMD GPU via DirectML
        dev = torch_directml.device()
        log.info(f"Using AMD GPU via DirectML: {dev}")
        return str(dev)
    except ImportError:
        pass

    if torch.cuda.is_available():
        log.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"

    log.info("No GPU found — using CPU (training will be slower).")
    return "cpu"

# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  MTACD batch training — single robust model")
    log.info("")
    log.info("  WHY ONE MODEL, NOT THREE:")
    log.info("  Helsinki WFS only provides *current* building footprints.")
    log.info("  Training on 2001 imagery with 2025 masks gives corrupted")
    log.info("  labels — the model is penalised for correct predictions.")
    log.info("  Instead we train on modern imagery (accurate masks) with")
    log.info("  heavy augmentation (blur, gamma, colour jitter) to make")
    log.info("  the model robust to the visual style of older photos.")
    log.info("")
    log.info(f"  Tiles         : {TILES_2025} (2025 WMS layer only — perfect masks)")
    log.info(f"  Epochs        : {EPOCHS}")
    log.info(f"  Tile size     : {TILE_SIZE} px")
    log.info(f"  Log file      : {LOG_PATH}")
    log.info("=" * 65)

    device   = pick_device()
    era      = "2025"
    data_dir = HERE / "training_data" / era

    # ── Step 1: generate data ──────────────────────────────────────────────
    log.info("")
    log.info(f"Step 1/2 — generating {TILES_2025} tiles from 2025 imagery ...")
    total = generate_era_data(era, TILES_2025, data_dir)

    if total < 5:
        log.error(f"Only {total} pairs after generation — cannot train.")
        return

    # ── Step 2: train ──────────────────────────────────────────────────────
    log.info("")
    log.info(f"Step 2/2 — training U-Net on {total} pairs ...")
    try:
        ckpt = train_era_model(era, data_dir, device)
        log.info(f"Model saved: {ckpt.name}")
    except Exception as exc:
        log.error(f"Training failed: {exc}")
        return

    log.info("")
    log.info("=" * 65)
    log.info("  Done.  One robust model trained on modern imagery.")
    log.info("  It handles older photos via augmentation (blur/gamma).")
    log.info("  Open the app → Detect Changes → Load checkpoint.")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
