"""
fetch_pairs.py — Download before/after Helsinki aerial pairs for change detection.

Downloads 2001 and 2025 orthophotos for Kallio, Kruununhaka, and Pasila,
then normalises each pair so brightness differences between the two epochs
don't trigger false detections.

Normalisation pipeline:
  1. Histogram matching — 2001 is matched to the 2025 colour distribution
     so overall tone and contrast are aligned.
  2. CLAHE (per-channel) — boosts local contrast on both images equally,
     making building edges and rooftop textures stand out.

Run:  py -3.12 fetch_pairs.py
Output written to:  pairs/<name>_2001.png  and  pairs/<name>_2025.png
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
import requests
from PIL import Image
from skimage.exposure import match_histograms

HERE = Path(__file__).parent.resolve()
OUT  = HERE / "pairs"
OUT.mkdir(exist_ok=True)

WMS_BASE   = "https://kartta.hel.fi/ws/geoserver/avoindata/wms"
CRS        = "EPSG:3879"
LAYER_2001 = "avoindata:Ortoilmakuva_2001"
LAYER_2025 = "avoindata:Ortoilmakuva_2025_5cm"
TILE_PX    = 1024   # pixels — 1024 px × 0.5 m/px = 512 m footprint

# ---------------------------------------------------------------------------
# Bounding boxes in EPSG:3879  (minX, minY, maxX, maxY)
# Each box is 512 m × 512 m, centred on the neighbourhood.
# Centre coordinates were projected with pyproj from known WGS84 landmarks.
# ---------------------------------------------------------------------------
HALF = TILE_PX * 0.5 / 2   # half the ground footprint in metres

def _box(cx: float, cy: float) -> tuple:
    return (cx - HALF, cy - HALF, cx + HALF, cy + HALF)

NEIGHBORHOODS: dict[str, tuple] = {
    "kallio":       _box(25_497_281, 6_674_596),   # centred on Kallio church
    "kruununhaka":  _box(25_497_669, 6_673_125),   # historic old-town peninsula
    "pasila":       _box(25_496_262, 6_676_202),   # Pasila railway / development zone
}


# ---------------------------------------------------------------------------
# WMS helpers
# ---------------------------------------------------------------------------

def _wms_url(layer: str, bbox: tuple) -> str:
    minx, miny, maxx, maxy = bbox
    return f"{WMS_BASE}?" + urlencode({
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS":  layer, "STYLES": "", "SRS": CRS,
        "BBOX":    f"{minx},{miny},{maxx},{maxy}",
        "WIDTH":   str(TILE_PX), "HEIGHT": str(TILE_PX),
        "FORMAT":  "image/png",
    })


def _fetch(layer: str, bbox: tuple, path: Path) -> bool:
    url = _wms_url(layer, bbox)
    try:
        r = requests.get(url, timeout=60)
        ct = r.headers.get("Content-Type", "")
        if r.status_code != 200 or "image" not in ct:
            print(f"    HTTP {r.status_code}  Content-Type: {ct}")
            return False
        path.write_bytes(r.content)
        return True
    except requests.RequestException as exc:
        print(f"    Request error: {exc}")
        return False


def _load_rgb(path: Path) -> np.ndarray:
    """Load any image (including greyscale 2001 scans) as uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _clahe(arr: np.ndarray) -> np.ndarray:
    """Apply CLAHE independently to each RGB channel."""
    out = np.empty_like(arr)
    cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for c in range(arr.shape[2]):
        out[:, :, c] = cl.apply(arr[:, :, c])
    return out


def normalize_pair(before: np.ndarray, after: np.ndarray):
    """
    Return a normalised (before, after) pair.

    Step 1 — histogram match: warp the 2001 colour distribution to resemble
    the 2025 image so overall brightness / white balance are aligned.

    Step 2 — CLAHE: enhance local contrast on both images with the same
    settings, making rooftop textures and building outlines crisper without
    amplifying the seasonal / sensor difference between epochs.
    """
    matched = match_histograms(before, after, channel_axis=-1).astype(np.uint8)
    return _clahe(matched), _clahe(after)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Helsinki aerial pair downloader")
    print(f"  2001 layer : {LAYER_2001}")
    print(f"  2025 layer : {LAYER_2025}")
    print(f"  Tile size  : {TILE_PX} px  ({TILE_PX * 0.5:.0f} m footprint)")
    print(f"  Output     : {OUT}\n")

    for name, bbox in NEIGHBORHOODS.items():
        print(f"--- {name.upper()} ---")
        raw_2001 = OUT / f"{name}_2001_raw.png"
        raw_2025 = OUT / f"{name}_2025_raw.png"

        print("  Fetching 2001 …", end=" ", flush=True)
        if not _fetch(LAYER_2001, bbox, raw_2001):
            print("FAILED — skipping.")
            continue
        print("ok")

        print("  Fetching 2025 …", end=" ", flush=True)
        if not _fetch(LAYER_2025, bbox, raw_2025):
            print("FAILED — skipping.")
            raw_2001.unlink(missing_ok=True)
            continue
        print("ok")

        before = _load_rgb(raw_2001)
        after  = _load_rgb(raw_2025)

        is_grey = (before[:, :, 0] == before[:, :, 1]).all()
        print(f"  2001: {before.shape}  mean={before.mean():.0f}"
              f"{'  (greyscale scan)' if is_grey else ''}")
        print(f"  2025: {after.shape}  mean={after.mean():.0f}")

        norm_b, norm_a = normalize_pair(before, after)

        out_b = OUT / f"{name}_2001.png"
        out_a = OUT / f"{name}_2025.png"
        Image.fromarray(norm_b).save(out_b)
        Image.fromarray(norm_a).save(out_a)

        # Remove raw files; keep only the normalised pair
        raw_2001.unlink(missing_ok=True)
        raw_2025.unlink(missing_ok=True)

        diff = abs(float(norm_b.mean()) - float(norm_a.mean()))
        print(f"  Saved: {out_b.name}  {out_a.name}"
              f"  (brightness diff after normalisation: {diff:.1f})")
        print()

    print("Done — load the pairs in the 'Detect Changes' tab.")


if __name__ == "__main__":
    main()
