"""
Helsinki Urban Change Detection — Unified App
==============================================
Combines DataGenerator13 and helsinki_gui_app_v3 into a single dark-mode
desktop application with three tabs:

  Generate Data   — Pull real aerial tiles + building masks from Helsinki's
                    open WMS / WFS APIs (kartta.hel.fi).
  Train Model     — Train or retrain the U-Net on those tiles.
  Detect Changes  — Compare a before / after aerial pair and visualise changes.

Run:  python app.py   or double-click Run.bat
"""

# "from __future__ import annotations" tells Python to treat all type hints
# in this file as plain strings instead of evaluating them immediately.
# This lets us use newer typing syntax even on slightly older Python versions.
from __future__ import annotations

# "import" loads an external library (a collection of ready-made code) so we
# can use its functions and classes without writing them from scratch.

import json          # json lets us read and write JSON files (a common data format).
import os            # os gives access to operating system features like environment variables.
import random        # random lets us generate random numbers (used for picking tile locations).
import threading     # threading lets us run code in the background so the GUI stays responsive.
import time          # time lets us pause the program (e.g. time.sleep) to avoid hammering the server.
from datetime import datetime   # datetime lets us get the current date and time (used for file names).
from pathlib import Path        # Path makes it easy to work with file and folder paths on any OS.
from typing import Optional     # Optional[X] means a variable can hold a value of type X *or* None.
from urllib.parse import urlencode  # urlencode converts a dictionary into a URL query string like "a=1&b=2".

import matplotlib                  # matplotlib is the main plotting library; we configure its backend here.
matplotlib.use("TkAgg")            # Tell matplotlib to draw plots inside a Tkinter window (not a separate one).

import tkinter as tk               # tkinter is Python's built-in GUI (graphical user interface) library.
from tkinter import ttk, filedialog, messagebox
# ttk      — themed widgets (buttons, labels, etc.) that look nicer than plain tk widgets.
# filedialog — pop-up dialogs for picking files and folders.
# messagebox — pop-up dialogs for showing warnings and errors to the user.

import cv2           # OpenCV (cv2) is a computer-vision library used for image processing.
import numpy as np   # NumPy (np) lets us work with large arrays of numbers efficiently (images are arrays).
import requests      # requests makes it easy to download data from the internet (HTTP GET/POST).
from PIL import Image, ImageDraw, ImageTk
# PIL (Pillow) is an image library.
# Image     — open, convert, and save image files.
# ImageDraw — draw shapes (polygons) onto images.
# ImageTk   — convert PIL images into a format Tkinter can display.
from shapely.geometry import shape    # shape() converts a GeoJSON geometry dict into a Shapely geometry object.
from shapely.ops import unary_union   # unary_union merges a list of geometry shapes into a single shape.

import torch                          # PyTorch is the deep-learning framework used to run the neural network.
from torch.utils.data import DataLoader, Subset
# DataLoader  — feeds batches of images into the neural network during training.
# Subset      — picks a slice (subset) of a dataset by index (used for train/val split).
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# FigureCanvasTkAgg lets us embed a matplotlib figure directly inside a Tkinter window.
from pyproj import Transformer        # pyproj converts coordinates between different map projections.
from skimage.exposure import match_histograms
# match_histograms adjusts one image's brightness/colour distribution to match another image.

import pipeline as P  # Our own pipeline module (pipeline.py) containing the U-Net model and training code.


# ── Module-level constants ────────────────────────────────────────────────────

# HERE is the folder that contains this very file (app.py).
HERE          = Path(__file__).parent.resolve()

# DEFAULT_MODEL is the path where the trained neural-network weights will be saved.
DEFAULT_MODEL = HERE / "models" / "unet_buildings.pt"

# PREVIEW_W/H are the pixel dimensions used for thumbnail previews in the GUI.
PREVIEW_W, PREVIEW_H = 320, 220

# DEMO_WIDTHS are the channel counts for each layer of the small (demo) U-Net model.
DEMO_WIDTHS = (16,  32,  64,  128, 256)

# FULL_WIDTHS are the channel counts for the larger, more accurate U-Net model.
FULL_WIDTHS = (32,  64, 128,  256, 512)


# =============================================================================
# DARK THEME
# =============================================================================
# These are colour codes in hex format (e.g. "#1e1e1e" = very dark grey).
# They are used throughout the GUI so that changing one variable updates every widget.

BG     = "#1e1e1e"   # Main window background — near-black.
BG2    = "#252526"   # Slightly lighter background used for header and notebook.
BG3    = "#2d2d2d"   # Even lighter background for status bar and canvas.
FG     = "#d4d4d4"   # Main foreground (text) colour — light grey.
FG2    = "#9d9d9d"   # Secondary text colour — dimmer grey, used for hints.
ACCENT = "#0078d4"   # Accent colour — bright blue, used for highlights and buttons.
BORDER = "#3e3e42"   # Border / separator colour.
INPUT  = "#3c3c3c"   # Background colour for text-entry fields.
SELECT = "#264f78"   # Background colour when text is selected.
BTN    = "#3a3a3a"   # Normal button background.
BTN_A  = "#4a4a4a"   # Button background when the mouse hovers over it.
GREEN  = "#4ec94e"   # Used for "success" log messages.
RED    = "#f44747"   # Used for "error" log messages.


# "def" defines a new function — a reusable block of code with a name.
# "_apply_dark_theme" receives the main Tkinter window (root) and styles every widget.
def _apply_dark_theme(root: tk.Tk) -> None:
    # Set the window's background colour to our near-black BG constant.
    root.configure(bg=BG)

    # Create a ttk Style object attached to this window so we can customise widget appearance.
    s = ttk.Style(root)

    # Switch to the "clam" base theme, which is the most customisable built-in theme.
    s.theme_use("clam")

    # Configure the default style for ALL widgets ("." means the root/default).
    # Each keyword sets a colour for a specific part of the widget.
    s.configure(".",
        background=BG, foreground=FG,
        fieldbackground=INPUT, selectbackground=SELECT,
        selectforeground=FG, bordercolor=BORDER,
        troughcolor=BG3, insertcolor=FG,
        darkcolor=BG, lightcolor=BG2,
    )
    # Style plain frames (invisible containers used to arrange widgets).
    s.configure("TFrame",      background=BG)
    # Style text labels.
    s.configure("TLabel",      background=BG, foreground=FG)
    # Style labelled frames (boxes with a title in the border).
    s.configure("TLabelframe", background=BG, bordercolor=BORDER)
    # Style the title text of a labelled frame.
    s.configure("TLabelframe.Label", background=BG, foreground=FG2)
    # Style horizontal/vertical separator lines.
    s.configure("TSeparator",  background=BORDER)

    # Style normal buttons.
    s.configure("TButton",
        background=BTN, foreground=FG,
        bordercolor=BORDER, padding=(10, 4), relief="flat",
    )
    # "s.map" changes how a widget looks when it enters a specific state (e.g. mouse hover).
    s.map("TButton",
        background=[("active", BTN_A), ("pressed", ACCENT)],
        foreground=[("active", FG)],
    )
    # "Accent.TButton" is a custom style for primary action buttons (blue background).
    s.configure("Accent.TButton",
        background=ACCENT, foreground="white", padding=(10, 4), relief="flat",
    )
    s.map("Accent.TButton",
        background=[("active", "#006cbd"), ("pressed", "#005fa3")],
    )
    # "Danger.TButton" is a custom style for destructive buttons like "Stop" (dark red).
    s.configure("Danger.TButton",
        background="#6b1a1a", foreground="white", padding=(10, 4), relief="flat",
    )
    s.map("Danger.TButton",
        background=[("active", "#8b2020")],
    )

    # Style text entry fields where the user types.
    s.configure("TEntry",
        fieldbackground=INPUT, foreground=FG,
        insertcolor=FG, bordercolor=BORDER,
    )
    # Style numeric spin-boxes (up/down arrow number pickers).
    s.configure("TSpinbox",
        fieldbackground=INPUT, foreground=FG,
        bordercolor=BORDER, arrowcolor=FG, background=BTN,
    )
    # Style drop-down combo-boxes.
    s.configure("TCombobox",
        fieldbackground=INPUT, background=BTN,
        foreground=FG, bordercolor=BORDER, arrowcolor=FG,
    )
    s.map("TCombobox",
        fieldbackground=[("readonly", INPUT)],
        selectbackground=[("readonly", SELECT)],
        selectforeground=[("readonly", FG)],
    )

    # Style the notebook (the tabbed panel that holds Generate/Train/Detect tabs).
    s.configure("TNotebook",     background=BG2, bordercolor=BORDER)
    # Style each individual tab in the notebook.
    s.configure("TNotebook.Tab",
        background=BG3, foreground=FG2, padding=(18, 7),
    )
    s.map("TNotebook.Tab",
        background=[("selected", BG), ("active", BG2)],
        foreground=[("selected", FG), ("active", FG)],
        expand=[("selected", [1, 1, 1, 0])],
    )

    # Style the progress bar that shows while the app is busy.
    s.configure("TProgressbar",
        background=ACCENT, troughcolor=BG3, bordercolor=BORDER,
    )
    # Style tick-boxes.
    s.configure("TCheckbutton", background=BG, foreground=FG)
    s.map("TCheckbutton",
        background=[("active", BG)],
        indicatorcolor=[("selected", ACCENT), ("!selected", INPUT)],
    )
    # Style radio buttons (mutually exclusive options).
    s.configure("TRadiobutton", background=BG, foreground=FG)
    s.map("TRadiobutton",
        background=[("active", BG)],
        indicatorcolor=[("selected", ACCENT), ("!selected", INPUT)],
    )
    # Style scroll bars.
    s.configure("TScrollbar",
        background=BG3, troughcolor=BG2,
        bordercolor=BORDER, arrowcolor=FG2, relief="flat",
    )
    s.map("TScrollbar", background=[("active", BTN_A)])


# =============================================================================
# DATA GENERATION  (DataGenerator13 — updated)
# =============================================================================

# WMS_BASE is the base URL for Helsinki's Web Map Service (WMS) — a standard
# API that returns map images as PNG files for a given bounding box.
WMS_BASE       = "https://kartta.hel.fi/ws/geoserver/avoindata/wms"

# WFS_BASE is the base URL for Helsinki's Web Feature Service (WFS) — a standard
# API that returns geographic features (like building footprints) as GeoJSON.
WFS_BASE       = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"

# AERIAL_LAYER is the WMS layer name for the current (latest) aerial photo.
AERIAL_LAYER   = "avoindata:Ortoilmakuva"

# BUILDING_LAYER is the WFS layer name for the building footprint polygons.
BUILDING_LAYER = "avoindata:Rakennukset_alue"

# CRS is the coordinate reference system used by the Helsinki open-data APIs.
# EPSG:3879 is the Finnish national grid projection (GK25FIN).
CRS            = "EPSG:3879"

# These four constants define the bounding box (in EPSG:3879 metres) of the
# area we sample random tile positions from — Helsinki city centre.
SAMPLE_X_MIN = 25494000
SAMPLE_X_MAX = 25502000
SAMPLE_Y_MIN = 6671000
SAMPLE_Y_MAX = 6677000

# Each downloaded tile is 1024 pixels wide and tall.
TILE_PX    = 1024

# Each pixel represents 0.5 metres on the ground, so the tile covers 512 m x 512 m.
GROUND_RES = 0.5                  # metres / pixel  →  512 m × 512 m tiles

# If a tile is more than 95% buildings we consider it too dense and skip it.
MAX_BUILDING_PCT  = 95.0

# A tile with few buildings but lots of edges (roads, trees) is kept as a
# "hard negative" example so the model learns not to confuse edges with buildings.
HARD_NEG_EDGE_THR = 0.04

# Water detection thresholds: tiles where the blue channel dominates AND the
# image has low variance (uniform colour) are classified as water and skipped.
WATER_BLUE_LEAD   = 8
WATER_MAX_STD     = 28

# AERIAL_LAYERS maps human-readable year labels to the exact GeoServer layer names.
# This dictionary is used to populate the year drop-down menus in the GUI.
# All historical aerial photo layers available from the Helsinki open-data WMS.
# Keys are human-readable labels shown in the year dropdowns.
# Values are the exact GeoServer layer names returned by GetCapabilities.
AERIAL_LAYERS: dict[str, str] = {
    "1932":          "avoindata:Ortoilmakuva_1932",
    "1943":          "avoindata:Ortoilmakuva_1943",
    "1950":          "avoindata:Ortoilmakuva_1950",
    "1954":          "avoindata:Ortoilmakuva_1954",
    "1956":          "avoindata:Ortoilmakuva_1956",
    "1961":          "avoindata:Ortoilmakuva_1961",
    "1964":          "avoindata:Ortoilmakuva_1964",
    "1969":          "avoindata:Ortoilmakuva_1969",
    "1972":          "avoindata:Ortoilmakuva_1972",
    "1976":          "avoindata:Ortoilmakuva_1976",
    "1987":          "avoindata:Ortoilmakuva_1987",
    "1988":          "avoindata:Ortoilmakuva_1988",
    "1991":          "avoindata:Ortoilmakuva_1991",
    "1993":          "avoindata:Ortoilmakuva_1993",
    "1995":          "avoindata:Ortoilmakuva_1995",
    "1996":          "avoindata:Ortoilmakuva_1996",
    "1997":          "avoindata:Ortoilmakuva_1997",
    "1998":          "avoindata:Ortoilmakuva_1998",
    "2001":          "avoindata:Ortoilmakuva_2001",
    "2004":          "avoindata:Ortoilmakuva_2004_50cm",
    "2005":          "avoindata:Ortoilmakuva_2005",
    "2006":          "avoindata:Ortoilmakuva_2006_50cm",
    "2008":          "avoindata:Ortoilmakuva_2008_25cm",
    "2009":          "avoindata:Ortoilmakuva_2009_25cm",
    "2010":          "avoindata:Ortoilmakuva_2010",
    "2011":          "avoindata:Ortoilmakuva_2011",
    "2012":          "avoindata:Ortoilmakuva_2012",
    "2013":          "avoindata:Ortoilmakuva_2013",
    "2014":          "avoindata:Ortoilmakuva_2014",
    "2015":          "avoindata:Ortoilmakuva_2015_20cm",
    "2016":          "avoindata:Ortoilmakuva_2016",
    "2017":          "avoindata:Ortoilmakuva_2017_20cm",
    "2018":          "avoindata:Ortoilmakuva_2018",
    "2019":          "avoindata:Ortoilmakuva_2019_20cm",
    "2020":          "avoindata:Ortoilmakuva_2020",
    "2021":          "avoindata:Ortoilmakuva_2021_10cm",
    "2022":          "avoindata:Ortoilmakuva_2022_5cm",
    "2023":          "avoindata:Ortoilmakuva_2023_5cm",
    "2024":          "avoindata:Ortoilmakuva_2024_30cm",
    "2025 (5cm)":    "avoindata:Ortoilmakuva_2025_5cm",
    "Latest":        "avoindata:Ortoilmakuva",
}

# Geocoding transformer: WGS84 lat/lon  →  EPSG:3879 (Finnish GK25FIN)
# This transformer converts latitude/longitude (GPS coordinates) into the
# metric coordinate system that the Helsinki APIs expect.
_WGS84_TO_3879 = Transformer.from_crs("EPSG:4326", "EPSG:3879", always_xy=True)

# TRAINING_LAYERS_BY_ERA groups Helsinki WMS layers by training strategy.
#
# WHY ONLY TWO OPTIONS (not three):
#   Helsinki's WFS building registry only contains *current* footprints.
#   Training on mid/historic imagery with today's masks produces corrupted
#   labels — buildings appear in the mask that don't exist in the old photo,
#   and the model gets penalised for correct predictions.
#   The augmentation pipeline (blur, gamma, colour jitter) already teaches
#   the model to handle older image styles without needing bad training data.
#
#   "auto"   — every available year; maximum variety for a general model.
#   "modern" — 2015-2025 only; accurate masks, best ground truth quality.
#              Use this for the primary/production model.
TRAINING_LAYERS_BY_ERA: dict[str, list[str]] = {
    # "2025" — the only option with perfectly accurate masks.
    # The WFS building registry reflects 2025, and the imagery is also 2025,
    # so every pixel label is correct.  This is the only recommended choice
    # for primary training.  The augmentation pipeline (blur, gamma, colour
    # jitter) teaches the model to handle older image styles at inference time.
    "2025": [
        "avoindata:Ortoilmakuva_2025_5cm",   # 5 cm/px — best available
        "avoindata:Ortoilmakuva",             # "latest" alias (also 2025)
    ],

    # "auto" mixes every available year.  The masks will be wrong for all
    # pre-2025 layers, introducing noise.  Only use this if you specifically
    # want the model to see a wide variety of image styles and can tolerate
    # some label inaccuracy.
    "auto": list(AERIAL_LAYERS.values()),
}


def _wms_url(bbox, layer: str = None) -> str:
    # Unpack the bounding box tuple into four separate variables.
    minx, miny, maxx, maxy = bbox
    # Use the provided layer name, or fall back to the default latest-imagery layer.
    # This lets callers request a specific historical year's layer.
    active_layer = layer if layer else AERIAL_LAYER
    # Build and return the full WMS URL by appending URL-encoded query parameters
    # to the base URL. The server will return a PNG image for this bounding box.
    return f"{WMS_BASE}?" + urlencode({
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS": active_layer, "STYLES": "", "SRS": CRS,
        "BBOX": f"{minx},{miny},{maxx},{maxy}",
        "WIDTH": str(TILE_PX), "HEIGHT": str(TILE_PX),
        "FORMAT": "image/png",
    })


def _fetch_image(bbox, path, layer: str = None) -> bool:
    # "try" attempts the code inside; if an error occurs, execution jumps to "except".
    try:
        # Send an HTTP GET request to the WMS server and wait up to 30 seconds.
        # Pass the layer name so we can request imagery from a specific historical year.
        r = requests.get(_wms_url(bbox, layer=layer), timeout=30)
        # If the server returned an error code OR didn't send an image, report failure.
        if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
            return False  # "return False" exits the function and sends back the value False.
        # Write the raw image bytes directly to the file at the given path.
        path.write_bytes(r.content)
        return True  # Return True to signal that the download succeeded.
    except requests.RequestException:
        # If any network error occurred (timeout, DNS failure, etc.), return False.
        return False


def _is_water(image_path) -> bool:
    # Load the image from disk using OpenCV (returns a NumPy array, BGR colour order).
    img = cv2.imread(str(image_path))
    # If the file couldn't be read, it's not water — return False.
    if img is None:
        return False
    # OpenCV stores colours as Blue, Green, Red (index 0, 1, 2).
    # A water tile is bluish (blue channel average > red channel average + threshold)
    # AND has low variance (smooth, uniform surface — std < WATER_MAX_STD).
    return (float(img[:, :, 0].mean()) > float(img[:, :, 2].mean()) + WATER_BLUE_LEAD
            and float(img.std()) < WATER_MAX_STD)


def _fetch_polygons(bbox):
    # Unpack the bounding box coordinates.
    minx, miny, maxx, maxy = bbox
    try:
        # Request building footprint polygons from Helsinki's WFS in GeoJSON format.
        r = requests.get(f"{WFS_BASE}?" + urlencode({
            "SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
            "TYPENAMES": BUILDING_LAYER, "SRSNAME": CRS,
            "BBOX": f"{minx},{miny},{maxx},{maxy},{CRS}",
            "OUTPUTFORMAT": "application/json",
        }), timeout=30)
        # If the server returned an error, signal failure by returning None.
        if r.status_code != 200:
            return None
        # Parse the JSON response and extract the list of feature objects.
        return json.loads(r.text).get("features", [])
    except (requests.RequestException, json.JSONDecodeError):
        # If the network request or JSON parsing failed, return None.
        return None


def _rasterize(features, bbox) -> np.ndarray:
    # Unpack the bounding box and calculate its width (dx) and height (dy) in metres.
    minx, miny, maxx, maxy = bbox
    dx, dy = maxx - minx, maxy - miny

    # Create a blank greyscale image (mode "L") filled with 0 (black = no building).
    canvas = Image.new("L", (TILE_PX, TILE_PX), 0)
    # Create a drawing context so we can paint polygons onto the canvas.
    draw   = ImageDraw.Draw(canvas)

    # Helper function: convert real-world coordinates (metres) to pixel positions.
    def to_px(x, y):
        # x moves right → pixel column; y increases upwards, but pixels go down → flip.
        return ((x - minx) / dx * TILE_PX, TILE_PX - (y - miny) / dy * TILE_PX)

    # Build a list of valid building geometries from the WFS feature list.
    geoms = []
    # "for ... in ..." loops over each item in the list one by one.
    for feat in features:
        try:
            # Convert the GeoJSON geometry dictionary into a Shapely polygon object.
            geom = shape(feat["geometry"]).buffer(0)
            # Only keep buildings larger than 20 square metres to ignore tiny artefacts.
            if geom.area >= 20:
                # Slightly expand (buffer) the geometry to close thin gaps between tiles.
                geoms.append(geom.buffer(0.15))
        except Exception:
            # If this feature's geometry is invalid, skip it and continue the loop.
            continue

    # If no valid geometries were found, return an all-black (no-building) mask.
    if not geoms:
        return np.zeros((TILE_PX, TILE_PX), dtype=np.uint8)

    # Merge all building geometries into a single shape to avoid overlapping fills.
    merged = unary_union(geoms)
    # Handle both single Polygon and MultiPolygon results from the union.
    polys  = [merged] if merged.geom_type == "Polygon" else list(merged.geoms)

    # Paint each building polygon white (255) onto the black canvas.
    for poly in polys:
        draw.polygon([to_px(x, y) for x, y in poly.exterior.coords], fill=255)
        # Paint any interior holes (courtyards, etc.) back to black (0).
        for hole in poly.interiors:
            draw.polygon([to_px(x, y) for x, y in hole.coords], fill=0)

    # Convert the PIL image to a NumPy array for OpenCV processing.
    mask   = np.array(canvas)

    # Create an elliptical structuring element (a small "brush") for morphological ops.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # MORPH_OPEN removes tiny isolated white dots (noise) by eroding then dilating.
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    # MORPH_CLOSE fills tiny holes inside buildings by dilating then eroding.
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find all connected white regions (individual building blobs) and their statistics.
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    # Start with an empty output mask; we will only copy blobs that pass quality checks.
    cleaned = np.zeros_like(mask)

    # Loop over every detected region (label 0 is background, so start at 1).
    for i in range(1, stats.shape[0]):
        # Read the pixel area, bounding-box width, and bounding-box height of this blob.
        area = stats[i, cv2.CC_STAT_AREA]
        w    = stats[i, cv2.CC_STAT_WIDTH]
        h    = stats[i, cv2.CC_STAT_HEIGHT]

        # Skip blobs smaller than 120 pixels — too small to be a real building.
        if area < 120:
            continue

        # Skip very elongated blobs (aspect ratio > 6) — they look like roads, not buildings.
        if max(w / (h + 1e-6), h / (w + 1e-6)) > 6:
            continue

        # Skip blobs that barely fill their bounding box (fill ratio < 18%) — likely noise.
        if area / (w * h + 1e-6) < 0.18:
            continue

        # This blob passed all checks — copy it into the cleaned mask.
        cleaned[labels == i] = 255

    # Return the cleaned binary mask where 255 = building and 0 = not building.
    return cleaned


# =============================================================================
# PAIR DOWNLOAD HELPERS
# =============================================================================

def _fetch_pair_image(layer: str, bbox: tuple, path: Path, px: int = 1024) -> bool:
    """Download one WMS tile (any historical layer) and write it to *path*."""
    # Unpack the four bounding-box values from the tuple.
    minx, miny, maxx, maxy = bbox
    # Build the full WMS URL for the requested historical layer, bounding box, and size.
    url = f"{WMS_BASE}?" + urlencode({
        "SERVICE": "WMS", "VERSION": "1.1.1", "REQUEST": "GetMap",
        "LAYERS":  layer,  "STYLES": "", "SRS": CRS,
        "BBOX":    f"{minx},{miny},{maxx},{maxy}",
        "WIDTH":   str(px), "HEIGHT": str(px),
        "FORMAT":  "image/png",
    })
    try:
        # Download the image; allow up to 60 seconds for large tiles.
        r = requests.get(url, timeout=60)
        # Read the Content-Type header to confirm the server returned an image.
        ct = r.headers.get("Content-Type", "")
        # If the status code is not 200 (OK) or the content is not an image, fail.
        if r.status_code != 200 or "image" not in ct:
            return False
        # Save the raw bytes to the given file path.
        path.write_bytes(r.content)
        return True
    except requests.RequestException:
        # Any network error means the download failed.
        return False


def _load_rgb_pair(path: Path) -> np.ndarray:
    """Load any image — including older greyscale aerial scans — as uint8 RGB."""
    # Open the file with Pillow, convert it to 3-channel RGB (handles greyscale too),
    # and return it as a NumPy array of unsigned 8-bit integers (0–255 per channel).
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _normalize_pair(before: np.ndarray, after: np.ndarray):
    """
    Make a before/after pair visually comparable despite sensor or seasonal differences.

    Step 1 — histogram matching: shifts the 2001 (before) colour distribution to
    match the 2025 (after) distribution so overall brightness and contrast align.

    Step 2 — CLAHE: boosts local contrast on both images with identical settings,
    making building edges and roof textures stand out equally.

    Returns (normalised_before, normalised_after) as uint8 RGB numpy arrays.
    """
    # Histogram matching: remap the "before" image's pixel values so its colour
    # distribution looks like the "after" image (same brightness and contrast range).
    matched = match_histograms(before, after, channel_axis=-1).astype(np.uint8)

    # Define a nested helper function that applies CLAHE to a single image.
    # CLAHE (Contrast Limited Adaptive Histogram Equalisation) sharpens local contrast.
    def _clahe(arr: np.ndarray) -> np.ndarray:
        # Create an output array with the same shape and type as the input.
        out = np.empty_like(arr)
        # Create a CLAHE processor with mild clipping and an 8×8 tile grid.
        cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # Apply CLAHE independently to each colour channel (R, G, B).
        for c in range(arr.shape[2]):
            out[:, :, c] = cl.apply(arr[:, :, c])
        return out

    # Return the histogram-matched + CLAHE-enhanced before, and CLAHE-enhanced after.
    return _clahe(matched), _clahe(after)


# =============================================================================
# HELPERS
# =============================================================================

def _load_image(path: Path) -> np.ndarray:
    # Open any image file with Pillow, ensure it is RGB (3 channels), and return
    # it as a uint8 NumPy array so the rest of the code can process it uniformly.
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def _to_photo(arr: np.ndarray, max_w: int, max_h: int) -> ImageTk.PhotoImage:
    # Convert a NumPy image array into a PIL Image object.
    img = Image.fromarray(arr)
    # Resize the image so it fits within (max_w, max_h) while keeping its aspect ratio.
    img.thumbnail((max_w, max_h))
    # Convert to a Tkinter-compatible PhotoImage that can be displayed on a canvas.
    return ImageTk.PhotoImage(img)


# =============================================================================
# APPLICATION
# =============================================================================

# "class" defines a blueprint for creating objects that bundle together data and functions.
# "App" is our main application class — one instance of it runs the whole GUI.
class App:
    # "__init__" is the special method called automatically when a new App object is created.
    # "self" refers to the App object being created; "root" is the main Tkinter window.
    def __init__(self, root: tk.Tk):
        # Store the main window as an attribute so every method can access it.
        self.root = root

        # Set the text shown in the window's title bar.
        self.root.title("Helsinki Urban Change Detection")

        # Set the initial window size to 1000 × 740 pixels.
        self.root.geometry("1000x740")

        # Prevent the window from being resized smaller than 860 × 620 pixels.
        self.root.minsize(860, 620)

        # Apply all the dark-theme colours defined above.
        _apply_dark_theme(root)

        # Initialise instance variables that will hold data while the app runs.
        # "Optional[np.ndarray]" means the variable holds a NumPy array OR None (not loaded yet).
        self.before_arr:  Optional[np.ndarray] = None   # The "before" aerial image array.
        self.after_arr:   Optional[np.ndarray] = None   # The "after" aerial image array.
        self.model:       Optional[P.UNet]     = None   # The loaded neural network (UNet or SiameseUNet).
        self.model_type:  str                 = "segmentation"  # "segmentation" or "siamese"
        self.model_widths = DEMO_WIDTHS                 # Channel widths of the currently loaded model.
        self.device       = P.get_default_device()      # "cuda" if a GPU is available, else "cpu".
        self.device_label = P.device_label(self.device) # A human-readable string like "GPU" or "CPU".
        self._photos:      dict = {}          # Keeps references to preview images (prevents garbage collection).
        self._gen_stop     = threading.Event() # An event flag used to signal the generation thread to stop.
        self._before_path: Optional[Path] = None   # File path of the loaded "before" image.
        self._after_path:  Optional[Path] = None   # File path of the loaded "after" image.
        self._model_name:  str            = ""      # File name of the currently loaded model checkpoint.

        # Build all the GUI widgets (tabs, buttons, labels, etc.).
        self._build_ui()

        # Load the previously saved default image pair from config.json (if it exists).
        self._load_defaults()

        # Automatically load the latest saved model or train a demo model if none exists.
        self._auto_load_or_train()

    # =========================================================================
    # UI construction
    # =========================================================================

    def _build_ui(self) -> None:
        # ── Header ────────────────────────────────────────────────────────────
        # Create a Frame widget at the top of the window to act as a header bar.
        hdr = tk.Frame(self.root, bg=BG2, pady=9)
        # Pack the header so it stretches the full width of the window.
        hdr.pack(fill="x")

        # Add the application title text to the left side of the header.
        tk.Label(hdr, text="Helsinki Urban Change Detection",
                 bg=BG2, fg=FG, font=("Segoe UI", 13, "bold")).pack(side="left", padx=14)

        # Create a StringVar — a special Tkinter variable that automatically updates
        # any label linked to it whenever its value changes.
        self.model_status = tk.StringVar(value="Loading…")

        # Add a label that shows the model status (e.g. "Model loaded [GPU]").
        # "textvariable=self.model_status" links the label text to the StringVar.
        tk.Label(hdr, textvariable=self.model_status,
                 bg=BG2, fg=ACCENT, font=("Segoe UI", 9)).pack(side="left", padx=4)

        # Add a label showing which hardware device (GPU/CPU) is being used.
        tk.Label(hdr, text=f"[{self.device_label}]",
                 bg=BG2, fg=FG2, font=("Segoe UI", 9)).pack(side="left")

        # ── Notebook ──────────────────────────────────────────────────────────
        # A Notebook widget is a tabbed panel — clicking a tab shows a different page.
        nb = ttk.Notebook(self.root)
        # Pack the notebook so it fills all remaining space in the window.
        nb.pack(fill="both", expand=True)

        # Create one Frame (page) for each tab; padding=14 adds space inside the frame.
        gen_tab    = ttk.Frame(nb, padding=14)
        train_tab  = ttk.Frame(nb, padding=14)
        pairs_tab  = ttk.Frame(nb, padding=14)
        detect_tab = ttk.Frame(nb, padding=14)

        # Add each frame to the notebook with a visible tab label.
        nb.add(gen_tab,    text="  Generate Data  ")
        nb.add(train_tab,  text="  Train Model  ")
        nb.add(pairs_tab,  text="  Download Pairs  ")
        nb.add(detect_tab, text="  Detect Changes  ")

        # Call the helper methods that fill each tab with its specific widgets.
        self._build_gen_tab(gen_tab)
        self._build_train_tab(train_tab)
        self._build_pairs_tab(pairs_tab)
        self._build_detect_tab(detect_tab)

        # ── Status bar ────────────────────────────────────────────────────────
        # Create a thin bar at the very bottom of the window for status messages.
        bar = tk.Frame(self.root, bg=BG3, pady=5)
        bar.pack(fill="x", side="bottom")

        # A StringVar for the status message so any code can update it easily.
        self.message = tk.StringVar(value="Ready.")

        # Add the status text label, anchored to the left ("w" = west).
        tk.Label(bar, textvariable=self.message, bg=BG3, fg=FG2,
                 anchor="w", font=("Segoe UI", 9)).pack(
                     side="left", padx=10, fill="x", expand=True)

        # Add an animated progress bar on the right side of the status bar.
        # mode="indeterminate" means it bounces back and forth (used when duration is unknown).
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=10)

    # ── Generate Data tab ─────────────────────────────────────────────────────

    def _build_gen_tab(self, parent) -> None:
        # Make column 1 (the log panel) expand when the window is resized.
        parent.columnconfigure(1, weight=1)
        # Make row 0 expand vertically when the window is resized.
        parent.rowconfigure(0, weight=1)

        # Settings panel (left)
        # A LabelFrame is a box with a visible border and a title in the top edge.
        cfg = ttk.LabelFrame(parent, text="Settings", padding=14)
        # Place the settings panel in row 0, column 0, sticking to all four sides.
        cfg.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # Label asking the user where to save the generated tiles.
        ttk.Label(cfg, text="Output folder:").pack(anchor="w")

        # StringVar holds the current value of the output-folder text field.
        self._gen_out = tk.StringVar(value=str(HERE / "training_data"))

        # A text entry field pre-filled with the default output path.
        ttk.Entry(cfg, textvariable=self._gen_out, width=32).pack(fill="x", pady=(2, 2))

        # A button that opens a folder-picker dialog and updates _gen_out.
        ttk.Button(cfg, text="Browse…", command=self._gen_browse).pack(anchor="w")

        # A horizontal separator line for visual grouping.
        ttk.Separator(cfg).pack(fill="x", pady=10)

        # Label for the tile count spinner.
        ttk.Label(cfg, text="Tiles to generate:").pack(anchor="w")

        # IntVar holds an integer; Spinbox is a number picker with up/down arrows.
        self._gen_n = tk.IntVar(value=50)
        ttk.Spinbox(cfg, from_=1, to=10000, increment=10,
                    textvariable=self._gen_n, width=10).pack(anchor="w", pady=(2, 0))

        ttk.Separator(cfg).pack(fill="x", pady=10)

        # Label for the minimum building coverage threshold.
        ttk.Label(cfg, text="Min building coverage (%):").pack(anchor="w")

        # DoubleVar holds a floating-point number; format="%.1f" shows one decimal place.
        self._gen_minpct = tk.DoubleVar(value=3.0)
        ttk.Spinbox(cfg, from_=0.5, to=20.0, increment=0.5, format="%.1f",
                    textvariable=self._gen_minpct, width=10).pack(anchor="w", pady=(2, 0))

        ttk.Separator(cfg).pack(fill="x", pady=10)

        # ── Training layer selector ───────────────────────────────────────────
        # "modern" fetches tiles exclusively from 2015–2025 WMS layers where the
        # WFS building masks are accurate.  "auto" mixes all available years for
        # maximum image-style variety — useful if you want the model to handle
        # a wide range of lighting conditions but don't mind a little mask noise
        # from older layers.
        #
        # Mid and historic options are intentionally omitted: Helsinki's WFS only
        # provides current building footprints, so training on old imagery with
        # today's masks produces corrupted ground truth that harms accuracy.
        ttk.Label(cfg, text="Training layers:").pack(anchor="w")

        # StringVar holds the selected option; default is "2025" (perfect masks).
        self._gen_era = tk.StringVar(value="2025")
        # Readonly Combobox — only two valid choices.
        ttk.Combobox(cfg, values=["2025", "auto"],
                     textvariable=self._gen_era,
                     state="readonly", width=16).pack(anchor="w", pady=(2, 2))

        # Short hint labels explaining each choice.
        tk.Label(cfg,
                 text=(
                     "2025 = 2025 imagery only  (recommended)\n"
                     "       masks are guaranteed accurate\n"
                     "auto = all years mixed  (noisy masks)"
                 ),
                 bg=BG, fg=FG2, font=("Segoe UI", 8), justify="left").pack(anchor="w")

        ttk.Separator(cfg).pack(fill="x", pady=10)

        # Informational label explaining the coverage area and tile specifications.
        # "\n" inside the string creates a new line.
        tk.Label(cfg,
                 text=(
                     "Coverage area\n"
                     "Helsinki City Centre\n"
                     "EPSG:3879\n"
                     "X  25 494 000 – 25 502 000\n"
                     "Y   6 671 000 –  6 677 000\n"
                     "\n"
                     "Tile:  1024 px @ 0.5 m/px\n"
                     "Footprint:  512 m × 512 m"
                 ),
                 bg=BG, fg=FG2, font=("Segoe UI", 8), justify="left").pack(anchor="w")

        ttk.Separator(cfg).pack(fill="x", pady=10)

        # BooleanVar holds True or False; the checkbox sets it when clicked.
        self._auto_train_var = tk.BooleanVar(value=True)
        # Checkbutton is a tick-box; ticking it sets _auto_train_var to True.
        ttk.Checkbutton(cfg, text="Auto-train when done",
                        variable=self._auto_train_var).pack(anchor="w", pady=(0, 8))

        # A small frame to hold the Start and Stop buttons side by side.
        btns = ttk.Frame(cfg)
        btns.pack(fill="x")

        # "Start Generating" button uses the Accent (blue) style to look like the primary action.
        self._gen_btn = ttk.Button(btns, text="Start Generating",
                                   style="Accent.TButton", command=self._start_gen)
        self._gen_btn.pack(side="left", fill="x", expand=True)

        # "Stop" button uses the Danger (red) style; it starts as disabled (greyed out).
        self._stop_btn = ttk.Button(btns, text="Stop",
                                    style="Danger.TButton", command=self._stop_gen,
                                    state="disabled")
        self._stop_btn.pack(side="left", padx=(6, 0))

        # Log panel (right)
        # Another LabelFrame for the scrollable log output.
        log_frm = ttk.LabelFrame(parent, text="Progress Log", padding=4)
        log_frm.grid(row=0, column=1, sticky="nsew")

        # A Text widget is a multi-line text box; state="disabled" makes it read-only.
        self._gen_log = tk.Text(
            log_frm, bg=BG2, fg=FG, insertbackground=FG,
            selectbackground=SELECT, font=("Consolas", 9),
            state="disabled", wrap="word", relief="flat", borderwidth=0,
        )
        # A Scrollbar linked to the Text widget so the user can scroll through the log.
        sb = ttk.Scrollbar(log_frm, command=self._gen_log.yview)
        # Link the Text widget's scroll position to the scrollbar.
        self._gen_log.configure(yscrollcommand=sb.set)
        # Pack the text widget on the left (filling all space) and the scrollbar on the right.
        self._gen_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Define named colour tags so log messages can be highlighted differently.
        self._gen_log.tag_configure("ok",   foreground=GREEN)   # Green for successful tiles.
        self._gen_log.tag_configure("skip", foreground=FG2)     # Grey for skipped tiles.
        self._gen_log.tag_configure("err",  foreground=RED)     # Red for errors.
        self._gen_log.tag_configure("head", foreground=ACCENT,  # Blue bold for section headers.
                                    font=("Consolas", 9, "bold"))

    # ── Train Model tab ───────────────────────────────────────────────────────

    def _build_train_tab(self, parent) -> None:
        # ── Quick actions ─────────────────────────────────────────────────────
        quick = ttk.LabelFrame(parent, text="Quick Actions", padding=10)
        quick.pack(fill="x", pady=(0, 10))
        ttk.Button(quick, text="Load checkpoint (.pt)…",
                   command=self._load_model).pack(side="left")
        ttk.Button(quick, text="Retrain demo model",
                   command=self._retrain).pack(side="left", padx=(6, 0))
        tk.Label(quick, text="  Demo uses synthetic data",
                 bg=BG, fg=FG2, font=("Segoe UI", 9)).pack(side="left", padx=10)

        # ── Main training frame ───────────────────────────────────────────────
        frm = ttk.LabelFrame(parent, text="Train model", padding=14)
        frm.pack(fill="both", expand=True)
        frm.columnconfigure(1, weight=1)

        # ── Model type dropdown (row 0) ───────────────────────────────────────
        ttk.Label(frm, text="Model type:").grid(row=0, column=0, sticky="w")
        self._train_type = tk.StringVar(value="Segmentation")
        type_cb = ttk.Combobox(
            frm,
            values=["Segmentation", "Siamese (Change Detection)"],
            textvariable=self._train_type,
            state="readonly", width=28)
        type_cb.grid(row=0, column=1, columnspan=2, sticky="w", padx=8)
        type_cb.bind("<<ComboboxSelected>>", self._on_train_type_change)

        # ── Hint text (row 1, updated on type change) ─────────────────────────
        self._train_hint = tk.StringVar(
            value="Paired by sorted filename.  Non-zero mask pixel = building.")
        tk.Label(frm, textvariable=self._train_hint,
                 bg=BG, fg=FG2, font=("Segoe UI", 9),
                 justify="left").grid(row=1, column=0, columnspan=3,
                                      sticky="w", pady=(4, 8))

        # ── Folder row A: Training images / Before images (row 2) ─────────────
        self._ti    = tk.StringVar()
        self._lbl_a = ttk.Label(frm, text="Training images:")
        self._lbl_a.grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=self._ti, width=46).grid(
            row=2, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(frm, text="Browse…",
                   command=lambda: self._browse_to(
                       self._ti, self._lbl_a.cget("text"))
                   ).grid(row=2, column=2, pady=3)

        # ── Folder row B: Training masks / After images (row 3) ───────────────
        self._tm    = tk.StringVar()
        self._lbl_b = ttk.Label(frm, text="Training masks:")
        self._lbl_b.grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=self._tm, width=46).grid(
            row=3, column=1, sticky="ew", padx=8, pady=3)
        ttk.Button(frm, text="Browse…",
                   command=lambda: self._browse_to(
                       self._tm, self._lbl_b.cget("text"))
                   ).grid(row=3, column=2, pady=3)

        # ── Folder row C: Change masks — Siamese only (row 4) ─────────────────
        # Hidden initially; shown when Siamese is selected.
        self._siamese_change = tk.StringVar()
        self._lbl_c = ttk.Label(frm, text="Change masks:")
        ent_c       = ttk.Entry(frm, textvariable=self._siamese_change, width=36)
        c_btns      = ttk.Frame(frm)
        ttk.Button(c_btns, text="Browse…",
                   command=lambda: self._browse_to(
                       self._siamese_change, "Change masks")
                   ).pack(side="left")
        ttk.Button(c_btns, text="Derive from model",
                   command=self._derive_change_masks).pack(side="left", padx=(4, 0))
        self._lbl_c.grid(row=4, column=0, sticky="w", pady=3)
        ent_c.grid(row=4, column=1, sticky="ew", padx=8, pady=3)
        c_btns.grid(row=4, column=2, pady=3, sticky="w")
        # Track all three widgets so we can show/hide them together.
        self._row_c_widgets = [self._lbl_c, ent_c, c_btns]

        # ── Auto-split checkbox (row 5) ───────────────────────────────────────
        self._auto_split = tk.BooleanVar(value=True)
        self._auto_split_cb = ttk.Checkbutton(
            frm, text="Auto-split  80 % train / 20 % val",
            variable=self._auto_split, command=self._toggle_val)
        self._auto_split_cb.grid(row=5, column=0, columnspan=3, sticky="w", pady=8)

        # ── Validation rows (rows 6–7) — segmentation only ───────────────────
        self._vi          = tk.StringVar()
        self._vm          = tk.StringVar()
        self._val_widgets = []
        self._folder_row(frm, 6, "Validation images:", self._vi, val=True)
        self._folder_row(frm, 7, "Validation masks:",  self._vm, val=True)

        # ── Hyperparameters (rows 8–12) ───────────────────────────────────────
        ttk.Separator(frm).grid(row=8, column=0, columnspan=3, sticky="ew", pady=10)

        ttk.Label(frm, text="Epochs:").grid(row=9, column=0, sticky="w")
        self._epochs = tk.IntVar(value=60)
        ttk.Spinbox(frm, from_=1, to=500, increment=10,
                    textvariable=self._epochs, width=8).grid(
                        row=9, column=1, sticky="w", padx=8)

        ttk.Label(frm, text="Tile size:").grid(row=10, column=0, sticky="w", pady=4)
        self._tile_sz = tk.StringVar(value="256")
        ttk.Combobox(frm, values=("128", "256", "384", "512"),
                     textvariable=self._tile_sz, width=7,
                     state="readonly").grid(row=10, column=1, sticky="w", padx=8)

        ttk.Label(frm, text="Model size:").grid(row=11, column=0, sticky="w", pady=4)
        self._model_sz = tk.StringVar(value="standard")
        sz = ttk.Frame(frm)
        sz.grid(row=11, column=1, columnspan=2, sticky="w", padx=8)
        ttk.Radiobutton(sz, text="Standard  (32 → 512 channels)",
                        variable=self._model_sz, value="standard").pack(anchor="w")
        ttk.Radiobutton(sz, text="Small  (16 → 256 channels, faster)",
                        variable=self._model_sz, value="small").pack(anchor="w")

        tk.Label(frm, text=f"Device: {self.device_label}",
                 bg=BG, fg=FG2, font=("Segoe UI", 9)).grid(
                     row=12, column=0, columnspan=3, sticky="w", pady=(10, 4))

        # ── Single shared Start Training button ───────────────────────────────
        ttk.Button(frm, text="Start Training", style="Accent.TButton",
                   command=self._start_training).grid(
                       row=13, column=0, columnspan=3, sticky="w", pady=(8, 0))

        # Apply initial state (segmentation mode, auto-split on).
        self._on_train_type_change(None)
        self._toggle_val()

    def _folder_row(self, parent, row, label, var, val=False):
        # Add a label in the first column.
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=0, sticky="w", pady=3)

        # Add a text entry field in the second column, linked to the given StringVar.
        ent = ttk.Entry(parent, textvariable=var, width=46)
        ent.grid(row=row, column=1, sticky="ew", padx=8, pady=3)

        # Define a local function that opens a folder-picker dialog and updates the StringVar.
        def browse():
            p = filedialog.askdirectory(title=label.rstrip(":"))
            # Only update the variable if the user actually chose a folder.
            if p:
                var.set(p)

        # Add a "Browse…" button in the third column that calls the local function above.
        btn = ttk.Button(parent, text="Browse…", command=browse)
        btn.grid(row=row, column=2, pady=3)

        # If this row belongs to the validation section, add its widgets to the tracking list
        # so _toggle_val() can enable/disable them based on the auto-split checkbox.
        if val:
            self._val_widgets.extend([lbl, ent, btn])

    def _toggle_val(self):
        # Validation rows are always hidden for Siamese (it uses auto-split internally).
        # For Segmentation they follow the auto-split checkbox.
        siamese = getattr(self, "_train_type", None) and self._train_type.get().startswith("Siamese")
        if siamese:
            state = "disabled"
        else:
            state = "disabled" if self._auto_split.get() else "normal"

        for w in self._val_widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    # ── Train-tab helpers ─────────────────────────────────────────────────────

    def _browse_to(self, var: tk.StringVar, title: str) -> None:
        """Open a folder-picker dialog and store the chosen path in *var*."""
        p = filedialog.askdirectory(title=title.rstrip(":"))
        if p:
            var.set(p)

    def _on_train_type_change(self, event) -> None:
        """Called whenever the model-type dropdown changes.

        Updates hint text, row-A/B labels, shows/hides row C (change masks),
        and refreshes the validation-row visibility.
        """
        siamese = self._train_type.get().startswith("Siamese")

        if siamese:
            self._train_hint.set(
                "Needs triplets: before image, after image, change mask.\n"
                "Click 'Derive from model' to auto-generate masks from an "
                "existing segmentation checkpoint.")
            self._lbl_a.configure(text="Before images:")
            self._lbl_b.configure(text="After images:")
            # Show row C (change masks).
            for w in self._row_c_widgets:
                w.grid()
        else:
            self._train_hint.set(
                "Paired by sorted filename.  Non-zero mask pixel = building.")
            self._lbl_a.configure(text="Training images:")
            self._lbl_b.configure(text="Training masks:")
            # Hide row C (change masks).
            for w in self._row_c_widgets:
                w.grid_remove()

        # Re-evaluate validation-row visibility under the new type.
        self._toggle_val()

    def _start_training(self) -> None:
        """Dispatcher: routes to the right training path based on the dropdown."""
        if self._train_type.get().startswith("Siamese"):
            self._start_train_siamese()
        else:
            self._start_train_from_data()

    def _derive_change_masks(self) -> None:
        """Validate folders then start the mask-derivation worker thread."""
        before = self._ti.get()
        after  = self._tm.get()

        def err(msg):
            messagebox.showwarning("Check the form", msg)

        if not before or not Path(before).is_dir():
            err("Set the 'Before images' folder first."); return
        if not after or not Path(after).is_dir():
            err("Set the 'After images' folder first."); return
        if self.model is None:
            err("No model loaded.  Load a segmentation checkpoint first."); return
        if self.model_type != "segmentation":
            err("Derivation needs a segmentation model, not a Siamese one."); return

        # Default output folder sits next to the before/ folder.
        out_dir = Path(before).parent / "changes"
        self._siamese_change.set(str(out_dir))

        self._busy(True, f"Deriving change masks → {out_dir} …")
        threading.Thread(
            target=self._do_derive_change_masks,
            args=(Path(before), Path(after), out_dir),
            daemon=True,
        ).start()

    def _do_derive_change_masks(self, before_dir: Path,
                                after_dir: Path, out_dir: Path) -> None:
        """Background worker: run the segmentation model on every before/after pair
        and save the XOR of predictions as a binary change mask."""
        try:
            out_dir.mkdir(parents=True, exist_ok=True)

            # Collect matching filenames present in both folders.
            exts   = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
            before_files = sorted(
                f for f in before_dir.iterdir()
                if f.suffix.lower() in exts)

            if not before_files:
                raise ValueError(f"No images found in {before_dir}")

            n_saved = 0
            for bf in before_files:
                af = after_dir / bf.name
                if not af.exists():
                    continue   # Skip unpaired files silently.

                # Load images as RGB uint8 arrays.
                bimg = np.array(Image.open(bf).convert("RGB"))
                aimg = np.array(Image.open(af).convert("RGB"))

                # Run segmentation inference on each image separately.
                _, b_mask = P.predict_full_image(
                    self.model, bimg,
                    tile_size=256, overlap=64,
                    device=self.device,
                )
                _, a_mask = P.predict_full_image(
                    self.model, aimg,
                    tile_size=256, overlap=64,
                    device=self.device,
                )

                # Change mask = pixels that are building in one era but not the other.
                change = ((b_mask > 0) ^ (a_mask > 0)).astype(np.uint8) * 255

                Image.fromarray(change).save(out_dir / bf.name)
                n_saved += 1

            msg = f"Saved {n_saved} change masks to {out_dir}"
            self.root.after(0, lambda: self._busy(False, msg))
            self.root.after(0, lambda: self._set_message(msg))

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Derivation failed", err))
            self.root.after(0, lambda: self._busy(False, "Mask derivation failed."))

    # ── Download Pairs tab ────────────────────────────────────────────────────

    def _build_pairs_tab(self, parent) -> None:
        """Build the Download Pairs tab.

        The user types any Helsinki location name, clicks Geocode to resolve it
        to EPSG:3879 coordinates via the Nominatim geocoding service, then picks
        a before and after year from the full list of Helsinki aerial layers.
        Clicking Download fetches both images from the WMS, histogram-matches
        them for brightness consistency, applies CLAHE for local contrast, saves
        them, and shows a preview. The pair can then be sent directly to the
        Detect Changes tab with one click.
        """
        # Column 0 (settings) stays fixed; column 1 (preview) stretches.
        parent.columnconfigure(0, weight=0)
        parent.columnconfigure(1, weight=1)
        # The single row stretches vertically to fill the tab.
        parent.rowconfigure(0, weight=1)

        # ── Left settings panel ───────────────────────────────────────────────
        # LabelFrame for all location/year controls.
        cfg = ttk.LabelFrame(parent, text="Location & Years", padding=14)
        cfg.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        # --- Place name -------------------------------------------------
        # Bold label prompting the user to type a location name.
        tk.Label(cfg, text="Place name:", bg=BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        # Hint text explaining what kinds of names are accepted.
        tk.Label(cfg,
                 text="Type any Helsinki neighbourhood, street, or landmark.",
                 bg=BG, fg=FG2, font=("Segoe UI", 8)).pack(anchor="w")

        # A small frame to place the entry field and the Geocode button side by side.
        place_row = ttk.Frame(cfg)
        place_row.pack(fill="x", pady=(4, 2))

        # StringVar for the place name; pre-filled with a sensible default.
        self._pair_place = tk.StringVar(value="Kallio, Helsinki")
        # Entry field linked to _pair_place.
        ttk.Entry(place_row, textvariable=self._pair_place,
                  width=26).pack(side="left", fill="x", expand=True)
        # Geocode button calls _geocode_place() to look up the coordinates.
        ttk.Button(place_row, text="Geocode",
                   command=self._geocode_place).pack(side="left", padx=(6, 0))

        # Label that will show the resolved EPSG:3879 coordinates after geocoding.
        self._pair_coords = tk.StringVar(value="(click Geocode to resolve)")
        tk.Label(cfg, textvariable=self._pair_coords,
                 bg=BG, fg=ACCENT, font=("Consolas", 8),
                 wraplength=260, justify="left").pack(anchor="w", pady=(0, 8))

        ttk.Separator(cfg).pack(fill="x", pady=6)

        # --- Before year ------------------------------------------------
        tk.Label(cfg, text="'Before' year:", bg=BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        # StringVar for the selected "before" year.
        self._pair_before_year = tk.StringVar(value="2001")
        # Read-only Combobox populated with all keys from the AERIAL_LAYERS dictionary.
        ttk.Combobox(cfg, values=list(AERIAL_LAYERS.keys()),
                     textvariable=self._pair_before_year,
                     state="readonly", width=22).pack(anchor="w", pady=(2, 8))

        # --- After year -------------------------------------------------
        tk.Label(cfg, text="'After' year:", bg=BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        # StringVar for the selected "after" year; defaults to the 5 cm 2025 layer.
        self._pair_after_year = tk.StringVar(value="2025 (5cm)")
        ttk.Combobox(cfg, values=list(AERIAL_LAYERS.keys()),
                     textvariable=self._pair_after_year,
                     state="readonly", width=22).pack(anchor="w", pady=(2, 8))

        ttk.Separator(cfg).pack(fill="x", pady=6)

        # --- Coverage footprint -----------------------------------------
        tk.Label(cfg, text="Coverage footprint:", bg=BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        tk.Label(cfg, text="Ground area each image covers.",
                 bg=BG, fg=FG2, font=("Segoe UI", 8)).pack(anchor="w")
        # StringVar for the chosen footprint size (256 m, 512 m, 1 km, or 2 km).
        self._pair_size = tk.StringVar(value="512 m")
        ttk.Combobox(cfg, values=["256 m", "512 m", "1 km", "2 km"],
                     textvariable=self._pair_size,
                     state="readonly", width=10).pack(anchor="w", pady=(2, 8))

        ttk.Separator(cfg).pack(fill="x", pady=6)

        # --- Output folder ----------------------------------------------
        tk.Label(cfg, text="Save folder:", bg=BG, fg=FG,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w")
        # StringVar for the folder where downloaded pair images will be saved.
        self._pair_out = tk.StringVar(value=str(HERE / "pairs"))
        out_row = ttk.Frame(cfg)
        out_row.pack(fill="x", pady=(2, 12))
        ttk.Entry(out_row, textvariable=self._pair_out,
                  width=26).pack(side="left", fill="x", expand=True)
        # Small "..." button to browse for an output folder.
        ttk.Button(out_row, text="...", command=self._browse_pair_out,
                   width=3).pack(side="left", padx=(4, 0))

        # --- Action buttons ---------------------------------------------
        # Primary button to start the download and normalisation process.
        self._pair_dl_btn = ttk.Button(cfg, text="Download & Normalise",
                                       style="Accent.TButton",
                                       command=self._start_download_pair)
        self._pair_dl_btn.pack(fill="x", pady=(0, 6))

        # Secondary button to copy the downloaded pair to the Detect Changes tab.
        ttk.Button(cfg, text="Send pair to Detect Changes tab",
                   command=self._load_pair_to_detect).pack(fill="x")

        # ── Right preview panel ───────────────────────────────────────────────
        # LabelFrame that contains side-by-side thumbnail previews of the two images.
        prev = ttk.LabelFrame(parent, text="Preview", padding=8)
        prev.grid(row=0, column=1, sticky="nsew")
        # Both preview columns share the horizontal space equally.
        prev.columnconfigure(0, weight=1)
        prev.columnconfigure(1, weight=1)
        prev.rowconfigure(0, weight=1)

        # Create image panel widgets for the before and after thumbnails.
        # lambda: None is a dummy callback because no browse action is needed here.
        self._pair_before_panel = self._image_panel(prev, "BEFORE", 0, lambda: None)
        self._pair_after_panel  = self._image_panel(prev, "AFTER",  1, lambda: None)

        # ── Status bar ────────────────────────────────────────────────────────
        # A small status label below the preview panels with instructions/feedback.
        self._pair_status = tk.StringVar(
            value="Enter a place name, click Geocode, then Download & Normalise.")
        tk.Label(parent, textvariable=self._pair_status,
                 bg=BG, fg=FG2, font=("Segoe UI", 9),
                 anchor="w").grid(row=1, column=0, columnspan=2,
                                  sticky="ew", pady=(10, 0))

    # ── Detect Changes tab ────────────────────────────────────────────────────

    def _build_detect_tab(self, parent) -> None:
        # Both image columns share the horizontal space equally.
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        # Create image panels for the before and after images in the Detect tab.
        # _pick_before and _pick_after open file-picker dialogs when "Browse…" is clicked.
        self._before_panel = self._image_panel(parent, "BEFORE", 0, self._pick_before)
        self._after_panel  = self._image_panel(parent, "AFTER",  1, self._pick_after)

        # A row of buttons below the image panels.
        btn_row = ttk.Frame(parent)
        btn_row.grid(row=1, column=0, columnspan=2, pady=(12, 4), sticky="ew")

        # Button to load a synthetic demo pair for demonstration purposes.
        ttk.Button(btn_row, text="Use Demo Pair",
                   command=self._use_demo_pair).pack(side="left")

        # Button to save the current pair paths to config.json as the default.
        ttk.Button(btn_row, text="Set as default pair",
                   command=self._save_defaults).pack(side="left", padx=8)

        # The main "Detect Changes" button that runs the neural network on both images.
        ttk.Button(btn_row, text="Detect Changes",
                   style="Accent.TButton",
                   command=self._detect_changes).pack(side="right")

        # Threshold slider row — lets the user adjust detection sensitivity.
        thresh_row = ttk.Frame(parent)
        thresh_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))

        tk.Label(thresh_row, text="Detection threshold:",
                 bg=BG, fg=FG2, font=("Segoe UI", 9)).pack(side="left")

        # DoubleVar to hold the slider value (0.1 to 0.9).
        self._threshold = tk.DoubleVar(value=0.5)
        # StringVar to show the current threshold as a formatted number beside the slider.
        self._thresh_lbl = tk.StringVar(value="0.50")

        # Callback function that updates the displayed threshold label whenever the slider moves.
        def _on_thresh(*_):
            # Format the current threshold value to two decimal places and update the label.
            self._thresh_lbl.set(f"{self._threshold.get():.2f}")

        # trace_add("write", ...) calls _on_thresh every time _threshold is written to.
        self._threshold.trace_add("write", _on_thresh)

        # A horizontal slider widget linked to _threshold.
        ttk.Scale(thresh_row, from_=0.1, to=0.9, variable=self._threshold,
                  orient="horizontal", length=200).pack(side="left", padx=8)

        # Label that shows the current numeric value next to the slider.
        tk.Label(thresh_row, textvariable=self._thresh_lbl,
                 bg=BG, fg=FG, font=("Consolas", 9, "bold"), width=4).pack(side="left")

        # Hint explaining what lower vs higher threshold values mean.
        tk.Label(thresh_row, text="← more detections  |  fewer →",
                 bg=BG, fg=FG2, font=("Segoe UI", 8)).pack(side="left", padx=8)

    def _image_panel(self, parent, title, col, on_browse):
        # Create a labelled frame to hold one image preview (title is "BEFORE" or "AFTER").
        frame = ttk.LabelFrame(parent, text=title, padding=8)
        # Place the frame in column `col` of the parent grid, stretching to fill.
        frame.grid(row=0, column=col, padx=5, sticky="nsew")

        # A Canvas widget is a drawing surface where we display the thumbnail image.
        canvas = tk.Canvas(frame, width=PREVIEW_W, height=PREVIEW_H,
                           bg=BG3, highlightthickness=1,
                           highlightbackground=BORDER)
        canvas.pack(pady=4)

        # Draw placeholder text in the centre of the canvas before any image is loaded.
        canvas.create_text(PREVIEW_W // 2, PREVIEW_H // 2,
                           text="(no image)", fill=FG2,
                           font=("Segoe UI", 11))

        # StringVar to display the file name beneath the image.
        filename = tk.StringVar(value="")
        tk.Label(frame, textvariable=filename, bg=BG, fg=FG2).pack()

        # StringVar for the auto-detected era label shown below the filename.
        # Shows something like "Era: modern  •  sharpness 842  •  brightness 134"
        # so the user immediately knows which model era should be applied.
        era_info = tk.StringVar(value="")
        tk.Label(frame, textvariable=era_info,
                 bg=BG, fg=FG2,
                 font=("Segoe UI", 8, "italic")).pack()

        # Button that calls on_browse when clicked (opens a file-picker dialog).
        ttk.Button(frame, text="Browse…", command=on_browse).pack(pady=4)

        # Return a dictionary so callers can access the canvas, filename, and era labels.
        return {"canvas": canvas, "filename": filename, "era": era_info}

    # =========================================================================
    # Download Pairs logic
    # =========================================================================

    def _geocode_place(self) -> None:
        """Resolve the place-name field to EPSG:3879 coordinates via Nominatim."""
        # Get the text the user typed and strip leading/trailing whitespace.
        name = self._pair_place.get().strip()
        # If the field is empty, do nothing.
        if not name:
            return

        # Update the status label to show that geocoding has started.
        self._pair_status.set(f"Geocoding '{name}' ...")
        self._pair_coords.set("searching ...")

        # Define the background worker function that will run on a separate thread.
        def worker():
            try:
                # Send a GET request to OpenStreetMap's Nominatim geocoding service.
                r = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={"q": name, "format": "json", "limit": 1,
                            "countrycodes": "fi"},      # Limit results to Finland.
                    headers={"User-Agent": "Helsinki-Change-Detection/1.0"},
                    timeout=10,
                )
                # Parse the JSON response into a Python list.
                results = r.json()

                # If the list is empty, the location was not found.
                if not results:
                    # root.after(0, ...) schedules a GUI update on the main thread
                    # (because GUI changes must only happen on the main thread).
                    self.root.after(0, lambda: self._pair_coords.set("Not found."))
                    self.root.after(0, lambda: self._pair_status.set(
                        "Location not found — try adding ', Helsinki' to the name."))
                    return

                # Extract latitude and longitude from the first result.
                lat  = float(results[0]["lat"])
                lon  = float(results[0]["lon"])

                # Convert WGS84 lat/lon to EPSG:3879 metric coordinates.
                cx, cy = _WGS84_TO_3879.transform(lon, lat)

                # Store the resolved coordinates as instance attributes for later use.
                self._pair_cx = cx
                self._pair_cy = cy

                # Extract a short display name (just the first part before any comma).
                label = results[0].get("display_name", name).split(",")[0]

                # Update the GUI labels on the main thread.
                self.root.after(0, lambda: self._pair_coords.set(
                    f"{label}\n{lat:.5f} N   {lon:.5f} E\n"
                    f"EPSG:3879  X={cx:.0f}  Y={cy:.0f}"))
                self.root.after(0, lambda: self._pair_status.set(
                    f"Geocoded: {label}  ({lat:.4f} N, {lon:.4f} E)  "
                    f"— now click Download & Normalise."))
            except Exception as exc:
                # If any error occurred, show it in the status label.
                err = str(exc)
                self.root.after(0, lambda: self._pair_status.set(f"Geocode error: {err}"))

        # Start the worker function on a new background thread so the GUI stays responsive.
        # daemon=True means the thread will be killed automatically when the app closes.
        threading.Thread(target=worker, daemon=True).start()

    def _browse_pair_out(self) -> None:
        # Open a folder-picker dialog and update _pair_out if the user chose a folder.
        p = filedialog.askdirectory(title="Select folder to save pairs")
        if p:
            self._pair_out.set(p)

    def _start_download_pair(self) -> None:
        """Validate inputs and kick off the download thread."""
        # Check that geocoding has already been done (i.e. _pair_cx attribute exists).
        if not hasattr(self, "_pair_cx"):
            messagebox.showwarning("No location",
                "Click Geocode first to resolve the place name to coordinates.")
            return

        # Look up the WMS layer names for the selected before and after years.
        before_layer = AERIAL_LAYERS.get(self._pair_before_year.get())
        after_layer  = AERIAL_LAYERS.get(self._pair_after_year.get())

        # If either year is invalid (not in the dictionary), show a warning.
        if not before_layer or not after_layer:
            messagebox.showwarning("Invalid year", "Select valid before / after years.")
            return

        # Map the human-readable footprint size to metres and calculate the half-width.
        size_m = {"256 m": 256, "512 m": 512, "1 km": 1000, "2 km": 2000}
        half   = size_m.get(self._pair_size.get(), 512) / 2

        # Build the bounding box as (minx, miny, maxx, maxy) centred on the geocoded point.
        bbox   = (self._pair_cx - half, self._pair_cy - half,
                  self._pair_cx + half, self._pair_cy + half)

        # Create the output folder if it doesn't already exist.
        out_dir = Path(self._pair_out.get())
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build a slug (safe file-name fragment) from the place name and selected years.
        place_slug  = self._pair_place.get().split(",")[0].strip().lower().replace(" ", "_")
        before_slug = self._pair_before_year.get().split()[0]
        after_slug  = self._pair_after_year.get().split()[0]
        stem = f"{place_slug}_{before_slug}_{after_slug}"  # e.g. "kallio_2001_2025"

        # Disable the download button while downloading to prevent double-clicks.
        self._pair_dl_btn.configure(state="disabled")
        self._pair_status.set("Downloading ...")

        # Start the actual download on a background thread.
        threading.Thread(
            target=self._do_download_pair,
            args=(before_layer, after_layer, bbox, out_dir, stem),
            daemon=True,
        ).start()

    def _do_download_pair(self, before_layer: str, after_layer: str,
                          bbox: tuple, out_dir: Path, stem: str) -> None:
        """Background worker: fetch, normalise, preview, and save the pair."""
        try:
            # Define paths for the temporary raw (unnormalised) downloads.
            raw_b = out_dir / f"{stem}_before_raw.png"
            raw_a = out_dir / f"{stem}_after_raw.png"

            # Update status and download the "before" image from the WMS.
            self.root.after(0, lambda: self._pair_status.set("Fetching 'before' image ..."))
            if not _fetch_pair_image(before_layer, bbox, raw_b):
                raise RuntimeError("WMS returned no image for the 'before' layer.")

            # Update status and download the "after" image from the WMS.
            self.root.after(0, lambda: self._pair_status.set("Fetching 'after' image ..."))
            if not _fetch_pair_image(after_layer, bbox, raw_a):
                raise RuntimeError("WMS returned no image for the 'after' layer.")

            # Update status and normalise the pair for brightness consistency.
            self.root.after(0, lambda: self._pair_status.set("Normalising brightness ..."))
            before_arr = _load_rgb_pair(raw_b)   # Load the raw before image as a NumPy array.
            after_arr  = _load_rgb_pair(raw_a)   # Load the raw after image as a NumPy array.
            norm_b, norm_a = _normalize_pair(before_arr, after_arr)  # Histogram-match + CLAHE.

            # Define the final output paths (without "_raw" in the name).
            out_b = out_dir / f"{stem}_before.png"
            out_a = out_dir / f"{stem}_after.png"

            # Save the normalised images to disk.
            Image.fromarray(norm_b).save(out_b)
            Image.fromarray(norm_a).save(out_a)

            # Delete the temporary raw files (missing_ok=True won't crash if they're gone).
            raw_b.unlink(missing_ok=True)
            raw_a.unlink(missing_ok=True)

            # Store the paths and arrays as instance attributes so other methods can use them.
            self._downloaded_before      = out_b
            self._downloaded_after       = out_a
            self._downloaded_before_arr  = norm_b
            self._downloaded_after_arr   = norm_a

            # Compute the absolute difference in mean brightness as a quality check.
            diff = abs(float(norm_b.mean()) - float(norm_a.mean()))
            by, ay = self._pair_before_year.get(), self._pair_after_year.get()
            h, w = norm_b.shape[:2]   # shape[:2] gives (height, width) — ignoring channels.

            # Show a summary in the status bar.
            self.root.after(0, lambda: self._pair_status.set(
                f"Saved: {out_b.name}  &  {out_a.name}  |  "
                f"Size: {w}x{h} px  |  Brightness diff after norm: {diff:.1f}"))

            # Display the normalised thumbnails in the preview panels.
            self.root.after(0, lambda: self._set_preview(
                self._pair_before_panel, norm_b, f"{by}  ({w}x{h})"))
            self.root.after(0, lambda: self._set_preview(
                self._pair_after_panel,  norm_a, f"{ay}  ({w}x{h})"))

            # Re-enable the download button now that the download is finished.
            self.root.after(0, lambda: self._pair_dl_btn.configure(state="normal"))

        except Exception as exc:
            # If anything went wrong, show the error message and re-enable the button.
            err = str(exc)
            self.root.after(0, lambda: self._pair_status.set(f"Error: {err}"))
            self.root.after(0, lambda: self._pair_dl_btn.configure(state="normal"))

    def _load_pair_to_detect(self) -> None:
        """Copy the last-downloaded pair into the Detect Changes tab."""
        # Check that a pair has actually been downloaded first.
        if not hasattr(self, "_downloaded_before"):
            messagebox.showwarning("No pair downloaded",
                "Download a pair first using the Download & Normalise button.")
            return

        # Copy the downloaded arrays and paths into the Detect tab's instance variables.
        self.before_arr   = self._downloaded_before_arr
        self.after_arr    = self._downloaded_after_arr
        self._before_path = self._downloaded_before
        self._after_path  = self._downloaded_after

        # Update the Detect tab's preview panels with the downloaded images.
        self._set_preview(self._before_panel, self.before_arr,
                          self._downloaded_before.name)
        self._set_preview(self._after_panel,  self.after_arr,
                          self._downloaded_after.name)

        # Inform the user that the pair is ready for detection.
        self._set_message(
            "Pair loaded from Download tab — switch to 'Detect Changes' "
            "and click 'Detect Changes'.")

    # =========================================================================
    # Data generation logic
    # =========================================================================

    def _gen_browse(self):
        # Open a folder-picker dialog and update the output folder field.
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self._gen_out.set(p)

    def _log(self, msg: str, tag: str = "") -> None:
        # Define a nested function that performs the actual text insertion.
        # It must run on the main thread because Tkinter widgets are not thread-safe.
        def _insert():
            # Temporarily unlock the read-only Text widget so we can insert text.
            self._gen_log.configure(state="normal")
            # Append the message with a newline; `tag` sets the colour.
            self._gen_log.insert("end", msg + "\n", tag)
            # Scroll the text box to the very end so the latest message is visible.
            self._gen_log.see("end")
            # Lock the widget as read-only again.
            self._gen_log.configure(state="disabled")
        # Schedule _insert to run on the main thread (delay 0 ms = as soon as possible).
        self.root.after(0, _insert)

    def _start_gen(self):
        # Clear the stop flag so the generation loop knows it should run.
        self._gen_stop.clear()
        # Disable the Start button and enable the Stop button.
        self._gen_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        # Read the output folder path, tile count, minimum coverage, and training era.
        out     = Path(self._gen_out.get())
        n       = self._gen_n.get()
        minpct  = self._gen_minpct.get()
        era     = self._gen_era.get()    # e.g. "2025" or "auto"

        # Log a header message to the progress log, showing which era will be fetched.
        era_label = era if era != "auto" else "auto (all eras)"
        self._log(f"Starting — {n} tiles  →  {out}  [era: {era_label}]", "head")

        # Start the generation loop on a background thread so the GUI stays responsive.
        # Pass the era so the loop can pick WMS layers from the right decade.
        threading.Thread(target=self._do_generate,
                         args=(out, n, minpct, era), daemon=True).start()

    def _stop_gen(self):
        # Set the stop event flag so the generation loop will exit after the current tile.
        self._gen_stop.set()
        self._log("Stop requested — finishing current tile…", "err")
        # Disable the Stop button immediately to prevent repeated clicks.
        self._stop_btn.configure(state="disabled")

    def _do_generate(self, out_dir: Path, n_tiles: int, min_pct: float,
                     era: str = "auto"):
        # Create the images and masks subdirectories (parents=True creates any missing parent folders).
        images_dir = out_dir / "images"
        masks_dir  = out_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        # Helper function: return the set of tile numbers that exist in a directory.
        def _tile_nums(directory: Path) -> set:
            nums: set = set()
            # Glob for files matching the pattern "tile_XXXX.png".
            for p in directory.glob("tile_*.png"):
                parts = p.stem.split("_")  # Split "tile_0001" into ["tile", "0001"].
                # Check that the second part is a number before converting.
                if len(parts) >= 2 and parts[1].isdigit():
                    nums.add(int(parts[1]))
            return nums

        # Only count tiles where BOTH image and mask are present.
        # Tiles where the image was downloaded but the mask was never written
        # (e.g. the run was interrupted mid-tile) are excluded so they get
        # re-downloaded cleanly on the next run.
        # Intersection (&) gives only tile numbers present in BOTH directories.
        complete    = _tile_nums(images_dir) & _tile_nums(masks_dir)

        # The next tile number to write; max() returns 0 if complete is empty.
        start_index = max(complete, default=0)

        # Clean up any orphaned images that have no matching mask.
        orphans = _tile_nums(images_dir) - complete
        for n in orphans:
            # Delete the orphaned image; missing_ok=True won't crash if already gone.
            (images_dir / f"tile_{n:04d}.png").unlink(missing_ok=True)

        # Inform the user if we are resuming a previous run.
        if start_index:
            self._log(
                f"Resuming — {start_index} complete pair(s) found, "
                f"next tile: tile_{start_index + 1:04d}", "head")

        # Store the selected era so auto-train can include it in the model filename.
        # "self." makes it an instance variable accessible from other methods.
        self._current_gen_era = era

        # Build the pool of WMS layers to draw from.  If the era key is not found
        # in the dictionary (which shouldn't happen), fall back to the latest layer.
        era_layers: list[str] = TRAINING_LAYERS_BY_ERA.get(era, [AERIAL_LAYER])

        # Create a reproducible random number generator seeded with 42.
        rng = random.Random(42)

        # Fast-forward the random state to where it left off on a previous run.
        # Each tile originally used 3 coordinates × 2 calls each.
        # Each tile consumes exactly 3 rng.random() calls:
        #   1 × x coordinate, 1 × y coordinate, 1 × layer index pick.
        # Fast-forward past the tiles already completed so resumed runs produce
        # the same sequence of NEW positions as a fresh run would have.
        for _ in range(start_index * 3):
            rng.random()

        # Calculate the tile footprint size in metres.
        tile_m  = TILE_PX * GROUND_RES

        # Dictionary to count how many tiles were rejected for each reason.
        rej     = {"fetch": 0, "water": 0, "wfs": 0, "dense": 0, "empty": 0}

        count   = 0   # Number of tiles successfully saved in this run.
        attempt = 0   # Total number of tile positions tried.

        # "while" loops as long as the condition is True.
        # Keep generating until we have enough tiles OR the Stop button was pressed.
        while count < n_tiles and not self._gen_stop.is_set():
            attempt += 1

            # Pick a random top-left corner inside the Helsinki sampling area.
            x    = rng.uniform(SAMPLE_X_MIN, SAMPLE_X_MAX - tile_m)
            y    = rng.uniform(SAMPLE_Y_MIN, SAMPLE_Y_MAX - tile_m)
            bbox = (x, y, x + tile_m, y + tile_m)

            # Pick a random WMS layer from the selected era's layer pool.
            # rng.random() returns a float 0.0–1.0; multiplied by the list length
            # and converted to int it gives a valid index.
            layer_idx  = int(rng.random() * len(era_layers))
            wms_layer  = era_layers[layer_idx]
            # Extract the year portion from the layer name for the log message
            # (e.g. "avoindata:Ortoilmakuva_2012" → "2012").
            layer_year = wms_layer.split("_")[-1] if "_" in wms_layer else "latest"

            # Build the file name for this tile (e.g. "tile_0003").
            name = f"tile_{start_index + count + 1:04d}"
            img_path  = images_dir / f"{name}.png"
            mask_path = masks_dir  / f"{name}.png"
            prefix    = f"[{count + 1}/{n_tiles}] {name} [{layer_year}]"  # Log shows year.

            # Safety guard: never overwrite a complete pair that already exists.
            if img_path.exists() and mask_path.exists():
                self._log(f"{prefix}  skip — already exists", "skip")
                count += 1
                continue   # "continue" skips the rest of the loop body and goes to the next iteration.

            # Try to download the aerial image for this bounding box from the chosen layer.
            if not _fetch_image(bbox, img_path, layer=wms_layer):
                self._log(f"{prefix}  SKIP — image fetch failed", "skip")
                rej["fetch"] += 1
                continue

            # Skip tiles that are mostly water (not useful for building detection).
            if _is_water(img_path):
                self._log(f"{prefix}  SKIP — water tile", "skip")
                img_path.unlink(missing_ok=True)   # Delete the downloaded image.
                rej["water"] += 1
                continue

            # Fetch building polygon data for this tile from the WFS.
            features = _fetch_polygons(bbox)
            if features is None:
                self._log(f"{prefix}  SKIP — WFS failed", "skip")
                img_path.unlink(missing_ok=True)
                rej["wfs"] += 1
                continue

            # Rasterize the building polygons into a binary mask image.
            mask = _rasterize(features, bbox)

            # Calculate what percentage of the tile is covered by buildings.
            pct  = mask.mean() / 255 * 100

            # Skip tiles that are almost entirely buildings (unrealistic / data error).
            if pct > MAX_BUILDING_PCT:
                self._log(f"{prefix}  SKIP — too dense ({pct:.1f}%)", "skip")
                img_path.unlink(missing_ok=True)
                rej["dense"] += 1
                continue

            # If building coverage is below the minimum, check whether the tile has
            # useful edge structure (roads, trees) to serve as a "hard negative" example.
            if pct < min_pct:
                # Read the image in greyscale and run Canny edge detection.
                gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                ed   = cv2.Canny(gray, 50, 150).mean() / 255.0  # Average edge density 0–1.

                # Keep the tile if edge density is high enough (complex, non-trivial scene).
                if ed > HARD_NEG_EDGE_THR:
                    self._log(f"{prefix}  ok — hard neg  (edges={ed:.3f})", "ok")
                else:
                    # Discard featureless tiles — they provide little training value.
                    self._log(f"{prefix}  SKIP — empty  (edges={ed:.3f})", "skip")
                    img_path.unlink(missing_ok=True)
                    rej["empty"] += 1
                    continue
            else:
                # The tile has enough buildings — log it as accepted.
                self._log(f"{prefix}  ok  ({pct:.1f}%)", "ok")

            # Save the binary mask as a PNG file.
            cv2.imwrite(str(mask_path), mask)
            count += 1

            # Wait 0.2 seconds between requests to avoid overloading the server.
            time.sleep(0.2)

        # Calculate final statistics for the log summary.
        total = len(list(images_dir.glob("*.png")))      # Total tiles in the images folder.
        rate  = count / max(attempt, 1) * 100            # Acceptance rate as a percentage.

        # Log the summary to the progress log.
        self._log(f"\nDone.  {count} new  |  {total} total  |  {rate:.1f}% accepted", "head")
        self._log(
            f"Skipped — fetch:{rej['fetch']}  water:{rej['water']}  "
            f"WFS:{rej['wfs']}  dense:{rej['dense']}  empty:{rej['empty']}", "skip")

        # Re-enable the Start button and disable the Stop button on the main thread.
        self.root.after(0, lambda: self._gen_btn.configure(state="normal"))
        self.root.after(0, lambda: self._stop_btn.configure(state="disabled"))

        # If auto-train is enabled and at least one tile was generated, start training.
        if self._auto_train_var.get() and count > 0:
            self._log("\nAuto-train enabled — starting training on generated data…", "head")
            # Schedule the auto-train call 200 ms from now (lets the GUI settle first).
            self.root.after(200, lambda: self._auto_train_on_data(out_dir))

    def _auto_train_on_data(self, out_dir: Path):
        # Build paths to the images and masks subdirectories inside the output folder.
        imgs = out_dir / "images"
        msks = out_dir / "masks"

        # Safety check: both subdirectories must exist before we start training.
        if not imgs.is_dir() or not msks.is_dir():
            self._set_message("Auto-train skipped — output folders not found.")
            return

        # Pre-fill the training folder fields in the Train tab.
        self._ti.set(str(imgs))
        self._tm.set(str(msks))

        # Enable the auto-split option so no separate validation folders are needed.
        self._auto_split.set(True)
        self._toggle_val()

        # Show the progress bar and a status message.
        self._busy(True, "Auto-training on generated data…")

        # Start training on a background thread using the generated data.
        threading.Thread(target=lambda: self._do_train_from_data(
            train_imgs=imgs, train_msks=msks,
            val_imgs=None, val_msks=None,
            auto_split=True,
            epochs=self._epochs.get(),
            tile_size=int(self._tile_sz.get()),
            widths=FULL_WIDTHS if self._model_sz.get() == "standard" else DEMO_WIDTHS,
        ), daemon=True).start()

    # =========================================================================
    # Training logic
    # =========================================================================

    def _auto_load_or_train(self):
        # Show the progress bar while we look for an existing checkpoint.
        self._busy(True, "Looking for a trained model…")

        # Define the background worker that does the file-system search.
        def worker():
            # Ensure the models folder exists.
            models_dir = DEFAULT_MODEL.parent
            models_dir.mkdir(parents=True, exist_ok=True)

            # Find all .pt files in the models folder, sorted by modification time.
            candidates = sorted(models_dir.glob("*.pt"),
                                key=lambda p: p.stat().st_mtime)

            # The last element is the most recently modified (latest) model.
            latest = candidates[-1] if candidates else None

            # If a checkpoint was found, try to load it.
            if latest:
                try:
                    # Detect the model architecture and type from the checkpoint.
                    widths, mtype = self._detect_widths(latest)
                    # Build the correct model skeleton based on the detected type.
                    if mtype == "siamese":
                        model = P.SiameseUNet(widths=widths)
                    else:
                        model = P.UNet(widths=widths)
                    # Load the saved weights into the model skeleton.
                    model  = P.load_checkpoint(model, latest, device=self.device)

                    # Store the loaded model and metadata.
                    self.model        = model
                    self.model_type   = mtype
                    self.model_widths = widths
                    self._model_name  = latest.name
                    lname = latest.name

                    # Choose a status label that tells the user which model type was loaded.
                    type_tag = "Siamese" if mtype == "siamese" else "Segmentation"
                    # Update the GUI on the main thread.
                    self.root.after(0, lambda t=type_tag: self.model_status.set(
                        f"{t} model loaded  [{self.device_label}]"))
                    self.root.after(0, lambda: self._busy(False,
                        f"Model ready — {lname}"))
                    return   # Loading succeeded — no need to train.
                except Exception as e:
                    # If loading failed, warn the user and fall through to training.
                    self.root.after(0, lambda: messagebox.showwarning(
                        "Checkpoint failed",
                        f"{e}\n\nA demo model will be trained instead."))

            # No valid checkpoint found — train a demo model from scratch.
            self.root.after(0, lambda: self.model_status.set("Training demo model…"))
            self.root.after(0, lambda: self._busy(True,
                "Training demo model — this takes a few minutes, please wait."))
            self._do_train(epochs=6)

        # Start the worker on a background thread.
        threading.Thread(target=worker, daemon=True).start()

    def _retrain(self):
        # Ask the user to confirm before overwriting the existing checkpoint.
        if not messagebox.askyesno("Retrain demo model",
                "Train a fresh demo model on synthetic data?\n"
                "This will overwrite the existing checkpoint."):
            return   # User clicked "No" — do nothing.

        # Show the progress bar and start training on a background thread.
        self._busy(True, "Training…")
        threading.Thread(target=lambda: self._do_train(epochs=8), daemon=True).start()

    def _do_train(self, epochs: int):
        try:
            # Create a time-stamped file name for the new checkpoint (avoids overwriting).
            # Tag with "modern" if that layer set was used, otherwise plain timestamp.
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            era_tag = getattr(self, "_current_gen_era", "auto")
            if era_tag == "2025":
                save_path = DEFAULT_MODEL.parent / f"unet_2025_{ts}.pt"
            else:
                save_path = DEFAULT_MODEL.parent / f"unet_{ts}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Point to the folder where synthetic training data will be generated.
            data_root = HERE / "data" / "synthetic"

            # Create a configuration object for the synthetic dataset generator.
            cfg = P.SyntheticConfig(tile_size=128, n_train=80, n_val=20, seed=42)

            # Generate the synthetic images and masks on disk.
            P.generate_synthetic_dataset(data_root, cfg)

            # Create dataset objects that load and augment the training images.
            train_ds = P.BuildingDataset(data_root / "train" / "images",
                                         data_root / "train" / "masks",
                                         tile_size=128, augment=True)

            # Validation dataset — no augmentation so results are reproducible.
            val_ds   = P.BuildingDataset(data_root / "val" / "images",
                                         data_root / "val" / "masks",
                                         tile_size=128, augment=False)

            # Create the U-Net model with the demo (smaller) channel widths.
            model = P.UNet(widths=DEMO_WIDTHS)

            # Create a training configuration with learning rate, batch size, etc.
            tc = P.TrainConfig(epochs=epochs, lr=1e-3, batch_size=8, num_workers=0,
                               device=self.device, checkpoint_path=str(save_path))

            # Run the training loop; DataLoader batches the data and shuffles each epoch.
            P.train(model, DataLoader(train_ds, batch_size=8, shuffle=True),
                    DataLoader(val_ds,  batch_size=8, shuffle=False), tc)

            # Load the best-performing checkpoint saved during training.
            model = P.load_checkpoint(model, save_path, device=self.device)

            # Store the newly trained model and its metadata.
            self.model        = model
            self.model_widths = DEMO_WIDTHS
            self._model_name  = save_path.name
            mname = save_path.name

            # Update the GUI on the main thread.
            self.root.after(0, lambda: self.model_status.set(
                f"Demo model  [{self.device_label}]  —  {mname}"))
            self.root.after(0, lambda: self._busy(False, f"Training complete — {mname}"))

        except Exception as e:
            # If training crashed, show an error dialog and hide the progress bar.
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Training failed", err))
            self.root.after(0, lambda: self._busy(False, "Training failed."))

    # =========================================================================
    # Siamese training helpers
    # =========================================================================

    def _start_train_siamese(self):
        """Validate the shared folder fields for Siamese training and start the worker."""
        before = self._ti.get()
        after  = self._tm.get()
        change = self._siamese_change.get()

        def err(msg):
            messagebox.showwarning("Check the form", msg)

        if not before or not Path(before).is_dir():
            err("Pick a valid 'Before images' folder."); return
        if not after or not Path(after).is_dir():
            err("Pick a valid 'After images' folder."); return
        if not change or not Path(change).is_dir():
            err("Pick a valid 'Change masks' folder.\n"
                "Use 'Derive from model' to generate masks automatically."); return

        epochs = self._epochs.get()
        self._busy(True, f"Training Siamese model for {epochs} epochs…")

        threading.Thread(
            target=self._do_train_siamese,
            args=(Path(before), Path(after), Path(change), epochs),
            daemon=True,
        ).start()

    def _do_train_siamese(self, before_dir: Path, after_dir: Path,
                          change_dir: Path, epochs: int):
        """Background worker: build ChangeDataset, train SiameseUNet, load result."""
        try:
            # Load full dataset with augmentation (training) and without (validation).
            full = P.ChangeDataset(before_dir, after_dir, change_dir,
                                   tile_size=256, augment=True)
            fval = P.ChangeDataset(before_dir, after_dir, change_dir,
                                   tile_size=256, augment=False)

            n = len(full)
            if n < 5:
                raise ValueError(f"Need at least 5 triplets, found {n}.")

            # 80 / 20 split.
            idx   = list(range(n))
            random.Random(42).shuffle(idx)
            n_val = max(1, int(n * 0.2))
            from torch.utils.data import Subset
            train_ds = Subset(full, sorted(idx[n_val:]))
            val_ds   = Subset(fval, sorted(idx[:n_val]))

            self.root.after(0, lambda: self._set_message(
                f"Siamese auto-split: {n - n_val} train / {n_val} val"))

            # Timestamped checkpoint name so old checkpoints are never overwritten.
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = DEFAULT_MODEL.parent / f"unet_siamese_{ts}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            model = P.SiameseUNet(widths=FULL_WIDTHS)
            tc    = P.TrainConfig(
                epochs          = epochs,
                lr              = 1e-3,
                batch_size      = 8,
                num_workers     = 0,
                device          = self.device,
                checkpoint_path = str(save_path),
            )

            P.train_siamese(model,
                            DataLoader(train_ds, batch_size=8, shuffle=True),
                            DataLoader(val_ds,   batch_size=8, shuffle=False),
                            tc)

            # Load the best-performing checkpoint.
            model = P.load_checkpoint(model, save_path, device=self.device)

            # Make the newly trained Siamese model the active model.
            self.model        = model
            self.model_type   = "siamese"
            self.model_widths = FULL_WIDTHS
            self._model_name  = save_path.name
            mname = save_path.name

            self.root.after(0, lambda: self.model_status.set(
                f"Siamese model  [{self.device_label}]  —  {mname}"))
            self.root.after(0, lambda: self._busy(
                False, f"Siamese training complete — {mname}"))

        except Exception as e:
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Siamese training failed", err))
            self.root.after(0, lambda: self._busy(False, "Siamese training failed."))

    def _start_train_from_data(self):
        # Validate the form fields and get a dictionary of parameters.
        params = self._validate_train_form()
        # If validation failed, params is None — do nothing.
        if params is None:
            return
        # Show the progress bar and start training on a background thread.
        self._busy(True, "Training on your data — this may take a while…")
        # "**params" unpacks the dictionary as keyword arguments to the function.
        threading.Thread(target=lambda: self._do_train_from_data(**params),
                         daemon=True).start()

    def _validate_train_form(self):
        # Read the current values from the StringVar fields.
        ti, tm = self._ti.get(), self._tm.get()
        vi, vm = self._vi.get(), self._vm.get()
        split  = self._auto_split.get()

        # Local helper that shows a warning dialog with a message.
        def err(msg):
            messagebox.showwarning("Check the form", msg)

        # Check that the training images folder path is valid.
        if not ti or not Path(ti).is_dir():
            err("Pick a valid training-images folder."); return None

        # Check that the training masks folder path is valid.
        if not tm or not Path(tm).is_dir():
            err("Pick a valid training-masks folder."); return None

        # If auto-split is OFF, also validate the separate validation folders.
        if not split:
            if not vi or not Path(vi).is_dir():
                err("Pick a validation-images folder, or tick Auto-split."); return None
            if not vm or not Path(vm).is_dir():
                err("Pick a validation-masks folder, or tick Auto-split."); return None

        # Choose channel widths based on the model size radio button selection.
        widths = FULL_WIDTHS if self._model_sz.get() == "standard" else DEMO_WIDTHS

        # Return a dictionary of all validated parameters.
        return dict(
            train_imgs=Path(ti), train_msks=Path(tm),
            val_imgs=Path(vi) if not split else None,  # None means "use auto-split".
            val_msks=Path(vm) if not split else None,
            auto_split=split,
            epochs=self._epochs.get(),
            tile_size=int(self._tile_sz.get()),
            widths=widths,
        )

    def _do_train_from_data(self, train_imgs, train_msks, val_imgs, val_msks,
                            auto_split, epochs, tile_size, widths):
        try:
            # If auto-split is enabled, split the dataset into train and val subsets in-place.
            if auto_split:
                # Load the full dataset twice: once with augmentation, once without.
                # The augmented version is used for training; the non-augmented for validation.
                full = P.BuildingDataset(train_imgs, train_msks,
                                         tile_size=tile_size, augment=True)
                fval = P.BuildingDataset(train_imgs, train_msks,
                                         tile_size=tile_size, augment=False)
                n = len(full)   # Total number of image/mask pairs.

                # Require at least 5 pairs to make a meaningful split.
                if n < 5:
                    raise ValueError(f"Need at least 5 pairs to auto-split, found {n}.")

                # Create a list of indices [0, 1, 2, ..., n-1] and shuffle it randomly.
                idx = list(range(n))
                random.Random(42).shuffle(idx)

                # The first 20% of shuffled indices go to validation.
                n_val    = max(1, int(n * 0.2))
                # Subset wraps a dataset and only exposes a chosen list of indices.
                train_ds = Subset(full, sorted(idx[n_val:]))     # Remaining 80% for training.
                val_ds   = Subset(fval, sorted(idx[:n_val]))     # First 20% for validation.

                # Show the split sizes in the status bar.
                self.root.after(0, lambda: self._set_message(
                    f"Auto-split: {n - n_val} train / {n_val} val"))
            else:
                # Use the separately specified validation folders.
                train_ds = P.BuildingDataset(train_imgs, train_msks,
                                             tile_size=tile_size, augment=True)
                val_ds   = P.BuildingDataset(val_imgs, val_msks,
                                             tile_size=tile_size, augment=False)

            # Create a time-stamped checkpoint file name.
            # If an era was set during data generation (and it was "2025"),
            # include it in the checkpoint name for easy identification.
            # e.g. "unet_2025_20250525_143200.pt" or "unet_20250525_143200.pt"
            ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
            era_tag = getattr(self, "_current_gen_era", "auto")
            if era_tag == "2025":
                save_path = DEFAULT_MODEL.parent / f"unet_2025_{ts}.pt"
            else:
                save_path = DEFAULT_MODEL.parent / f"unet_{ts}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Build and train the model.
            model = P.UNet(widths=widths)
            tc = P.TrainConfig(epochs=epochs, lr=1e-3, batch_size=8, num_workers=0,
                               device=self.device, checkpoint_path=str(save_path))
            P.train(model, DataLoader(train_ds, batch_size=8, shuffle=True),
                    DataLoader(val_ds,  batch_size=8, shuffle=False), tc)

            # Load the best checkpoint saved during training.
            model = P.load_checkpoint(model, save_path, device=self.device)

            # Store the trained model and metadata.
            self.model        = model
            self.model_widths = widths
            self._model_name  = save_path.name
            mname = save_path.name

            # Update the GUI on the main thread.
            self.root.after(0, lambda: self.model_status.set(
                f"Model  [{self.device_label}]  —  {mname}"))
            self.root.after(0, lambda: self._busy(False, f"Training complete — {mname}"))

        except Exception as e:
            # If training failed, show an error and hide the progress bar.
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Training failed", err))
            self.root.after(0, lambda: self._busy(False, "Training failed."))

    def _load_model(self):
        # Open a file-picker restricted to .pt and .pth checkpoint files.
        path = filedialog.askopenfilename(
            title="Load model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("All files", "*.*")])

        # If the user cancelled the dialog (no file chosen), do nothing.
        if not path:
            return
        try:
            # Detect the model type and channel widths from the checkpoint.
            widths, mtype = self._detect_widths(Path(path))
            # Build the correct model skeleton and load the weights.
            if mtype == "siamese":
                model = P.SiameseUNet(widths=widths)
            else:
                model = P.UNet(widths=widths)
            model = P.load_checkpoint(model, path, device=self.device)

            # Store the model and update the GUI status.
            self.model        = model
            self.model_type   = mtype
            self.model_widths = widths
            self._model_name  = Path(path).name
            type_tag = "Siamese" if mtype == "siamese" else "Segmentation"
            self.model_status.set(f"{type_tag} model: {Path(path).name}")
            self._set_message(f"{type_tag} model loaded — {Path(path).name}")
        except Exception as e:
            # Show an error dialog if loading failed.
            messagebox.showerror("Load failed", str(e))

    def _detect_widths(self, path):
        """Return (widths_tuple, model_type) by inspecting a checkpoint file.

        Model type is "siamese" if the file's ``model_type`` key says so, OR
        if the state-dict keys are prefixed with "unet." (SiameseUNet layout).
        Otherwise it is "segmentation" (standard U-Net).
        """
        # Load the checkpoint file into CPU memory (avoids needing a GPU just to inspect it).
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)

        # If the checkpoint is a dictionary with a "model_state" key, extract just that.
        # Otherwise assume the checkpoint IS the state dictionary directly.
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

        # Auto-detect model type from the explicit key (written by train_siamese)
        # or by checking whether the state-dict uses "unet." prefixed keys.
        if isinstance(ckpt, dict) and ckpt.get("model_type") == "siamese":
            model_type = "siamese"
        elif "unet.inc.block.0.weight" in state:
            model_type = "siamese"
        else:
            model_type = "segmentation"

        # Read the first-layer output channel count to determine the width preset.
        if model_type == "siamese":
            # SiameseUNet wraps a UNet at self.unet, so keys are "unet.inc..."
            w1 = int(state["unet.inc.block.0.weight"].shape[0])
        else:
            w1 = int(state["inc.block.0.weight"].shape[0])

        # Map the first-layer width to the corresponding preset channel-width tuple.
        if w1 == 16:
            widths = DEMO_WIDTHS
        elif w1 == 32:
            widths = FULL_WIDTHS
        else:
            # For any other width, construct a custom tuple that doubles each time.
            widths = (w1, w1 * 2, w1 * 4, w1 * 8, w1 * 16)

        return widths, model_type

    # =========================================================================
    # Detection logic
    # =========================================================================

    def _pick_before(self):
        # Open a file-picker that filters for common image formats.
        path = filedialog.askopenfilename(
            title="Pick the 'before' aerial image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                       ("All files", "*.*")])

        # If the user cancelled the dialog, do nothing.
        if not path:
            return
        try:
            # Load the chosen image as a NumPy RGB array.
            self.before_arr = _load_image(Path(path))
        except Exception as e:
            # Show an error dialog if the image couldn't be loaded.
            messagebox.showerror("Load error", str(e)); return

        # Store the file path and update the preview thumbnail.
        self._before_path = Path(path)
        self._set_preview(self._before_panel, self.before_arr, Path(path).name)
        self._set_message("'Before' loaded. Now pick an 'After' image.")

    def _pick_after(self):
        # Open a file-picker for the "after" image.
        path = filedialog.askopenfilename(
            title="Pick the 'after' aerial image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.tif *.tiff"),
                       ("All files", "*.*")])

        # If the user cancelled the dialog, do nothing.
        if not path:
            return
        try:
            # Load the chosen image as a NumPy RGB array.
            self.after_arr = _load_image(Path(path))
        except Exception as e:
            messagebox.showerror("Load error", str(e)); return

        # Store the file path and update the preview thumbnail.
        self._after_path = Path(path)
        self._set_preview(self._after_panel, self.after_arr, Path(path).name)
        self._set_message("'After' loaded. Click 'Detect Changes'.")

    def _use_demo_pair(self):
        # Update the status bar to say we are generating the demo pair.
        self._set_message("Generating synthetic demo pair…")

        # Configuration for the synthetic scene: 384×384 pixels, 8–14 buildings.
        cfg  = P.SyntheticConfig(tile_size=384, min_buildings=8, max_buildings=14)

        # Generate a random seed from 2 bytes of OS randomness (range 0–9999).
        seed = int.from_bytes(os.urandom(2), "big") % 10000

        # Generate a before/after pair with 3 new buildings and 2 removed buildings.
        before, _, after, _ = P.generate_temporal_pair(
            cfg=cfg, seed=seed, n_added=3, n_removed=2)

        # Store the generated arrays as the current before/after images.
        self.before_arr = before
        self.after_arr  = after

        # Show thumbnails of both images in the Detect tab.
        self._set_preview(self._before_panel, before, "(demo)")
        self._set_preview(self._after_panel,  after,  "(demo)")
        self._set_message("Demo pair ready. Click 'Detect Changes'.")

    def _detect_changes(self):
        # Make sure a model is actually loaded before we try to use it.
        if self.model is None:
            messagebox.showwarning("No model", "Wait for the model to finish loading or training.")
            return

        # Make sure both images have been loaded.
        if self.before_arr is None or self.after_arr is None:
            messagebox.showwarning("Missing images",
                                   "Load both a 'Before' and an 'After' image first.")
            return

        # Both images must have the same width, height, and number of channels.
        if self.before_arr.shape != self.after_arr.shape:
            messagebox.showwarning("Size mismatch",
                f"Both images must have the same dimensions.\n"
                f"Before: {self.before_arr.shape}\n"
                f"After:  {self.after_arr.shape}")
            return

        # Calculate the mean pixel brightness of each image (0–255).
        before_mean = float(self.before_arr.mean())
        after_mean  = float(self.after_arr.mean())

        # If the brightness differs by more than 50, warn the user — this can cause
        # false change detections because the model was trained on consistent images.
        if abs(before_mean - after_mean) > 50 and self.model_type != "siamese":
            # Brightness warning only for segmentation models: the two-pass diff
            # approach is sensitive to per-image lighting differences. A Siamese
            # model was trained to handle this, so no warning is needed.
            if not messagebox.askyesno("Brightness difference detected",
                f"The images have significantly different brightness\n"
                f"(Before: {before_mean:.0f}, After: {after_mean:.0f}).\n\n"
                "This may indicate different seasons or sensors and could cause\n"
                "false change detections. Continue anyway?"):
                return   # User chose not to continue.

        # Choose the status message based on the loaded model type.
        is_siamese = (self.model_type == "siamese")
        busy_msg   = ("Running Siamese network on image pair…"
                      if is_siamese else "Running U-Net on both images…")
        self._busy(True, busy_msg)
        threading.Thread(target=self._do_detect, daemon=True).start()

    def _do_detect(self):
        try:
            # Choose the tile size based on the model size (demo uses 128, full uses 256).
            tile       = 128 if self.model_widths == DEMO_WIDTHS else 256
            thresh     = self._threshold.get()    # The current threshold slider value.
            model_name = self._model_name         # The checkpoint file name for the report title.

            # ── Suggest a newer model if one is available ─────────────────────
            try:
                models_dir  = DEFAULT_MODEL.parent
                all_models  = sorted(models_dir.glob("*.pt"),
                                     key=lambda p: p.stat().st_mtime)
                active_stem = Path(model_name).stem if model_name else ""
                if all_models and all_models[-1].stem != active_stem:
                    newest = all_models[-1].name
                    tip = (f"A newer model is available: {newest}. "
                           f"Load it via 'Load checkpoint' for best results.")
                    self.root.after(0, lambda t=tip: self._set_message(t))
            except Exception:
                pass

            # ── Branch: Siamese (direct change detection) vs Segmentation ─────
            if self.model_type == "siamese":
                # ── Siamese path ─────────────────────────────────────────────
                # One forward pass that sees both images simultaneously.
                # Returns a change-probability map and a binary change mask.
                # Unlike the two-pass approach, errors don't compound here:
                # the network learned to detect *differences*, not buildings.
                _, change_mask = P.predict_change(
                    self.model, self.before_arr, self.after_arr,
                    tile_size=tile, device=self.device, threshold=thresh)

                # Post-process: remove tiny noise blobs (< 400 px²).
                change_mask = P.clean_building_mask(
                    change_mask, image=self.after_arr,
                    min_area=400, max_aspect_ratio=8.0,
                    min_fill=0.20, min_edge_density=0.04)

                # For the result window, pass the change mask as the "after" mask
                # and zeros as the "before" mask so the visualiser colours all
                # changed pixels as "added" (orange).  The title clarifies this.
                mask_before = np.zeros_like(change_mask)
                report      = P.detect_changes(mask_before, change_mask, min_area=400)

                n_changed = int(change_mask.sum())
                self.root.after(0, lambda: self._show_result(
                    mask_before, change_mask, report, thresh, model_name,
                    siamese=True))
                self.root.after(0, lambda: self._busy(
                    False, f"Done — {n_changed} changed pixels detected"))

            else:
                # ── Segmentation path (original two-pass approach) ─────────────
                # Run the U-Net on the full "before" image, sliding a window across it.
                _, mb = P.predict_full_image(self.model, self.before_arr,
                                             tile_size=tile, device=self.device,
                                             threshold=thresh)
                # Run the U-Net on the full "after" image.
                _, ma = P.predict_full_image(self.model, self.after_arr,
                                             tile_size=tile, device=self.device,
                                             threshold=thresh)

                # Remove false-positive detections that are too small, elongated,
                # or featureless to be real buildings.
                mb = P.clean_building_mask(mb, image=self.before_arr,
                                           min_area=600, max_aspect_ratio=5.0,
                                           min_fill=0.30, min_edge_density=0.07)
                ma = P.clean_building_mask(ma, image=self.after_arr,
                                           min_area=600, max_aspect_ratio=5.0,
                                           min_fill=0.30, min_edge_density=0.07)

                # Compare the two cleaned masks to classify each pixel.
                report = P.detect_changes(mb, ma, min_area=600)

                self.root.after(0, lambda: self._show_result(
                    mb, ma, report, thresh, model_name))
                self.root.after(0, lambda: self._busy(
                    False,
                    f"Done — Added: {report.stats['n_added']}, "
                    f"Removed: {report.stats['n_removed']}"))

        except Exception as e:
            # If detection failed, show an error and stop the progress bar.
            err = str(e)
            self.root.after(0, lambda: messagebox.showerror("Detection failed", err))
            self.root.after(0, lambda: self._busy(False, "Detection failed."))

    def _show_result(self, mask_before, mask_after, report,
                     threshold: float, model_name: str,
                     siamese: bool = False):
        # Get the checkpoint stem (name without extension) for the window title.
        model_stem = Path(model_name).stem if model_name else "unknown"

        # Create a different title for Siamese results so the user knows which
        # inference path produced the output.
        mode_label = "Siamese direct" if siamese else "Segmentation diff"

        # Create a new pop-up window (Toplevel) separate from the main window.
        win = tk.Toplevel(self.root)
        win.title(
            f"Change Detection  |  {mode_label}  |  threshold {threshold:.2f}  |  {model_stem}")
        win.geometry("1180x800")
        win.configure(bg=BG)

        # Ask the pipeline module to create a matplotlib figure comparing the images.
        fig_title = (f"Helsinki Urban Change Detection  "
                     f"|  {mode_label}  |  threshold {threshold:.2f}  |  {model_stem}")
        fig = P.plot_comparison(
            self.before_arr, self.after_arr, mask_before, mask_after, report,
            title=fig_title)

        # Apply the dark theme colours to the matplotlib figure.
        fig.patch.set_facecolor(BG)
        for ax in fig.axes:
            ax.set_facecolor(BG2)
            ax.title.set_color(FG)
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        # Embed the matplotlib figure inside the Tkinter pop-up window.
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()   # Render the figure into the canvas widget.
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Add a label showing the model name and threshold below the figure.
        tk.Label(win,
                 text=f"Model: {model_stem}    Threshold: {threshold:.2f}",
                 bg=BG3, fg=ACCENT, font=("Segoe UI", 9), pady=4).pack(fill="x")

        # Unpack the statistics dictionary for easier access.
        s = report.stats

        # Add a label summarising the detected building changes in numbers.
        tk.Label(win,
                 text=(f"Added: {s['n_added']} buildings    "
                       f"Removed: {s['n_removed']} buildings    "
                       f"Added area: {s['added_pct']:.2f}%    "
                       f"Removed area: {s['removed_pct']:.2f}%    "
                       f"Net change: {s['net_change_px']:+d} px"),
                 bg=BG2, fg=FG, font=("Segoe UI", 10, "bold"),
                 pady=8).pack(fill="x")

        # Define a local function that saves the current figure to a PNG file.
        def save_png():
            # Open a save-as dialog pre-set to save a .png file.
            out = filedialog.asksaveasfilename(
                title="Save result as PNG", defaultextension=".png",
                filetypes=[("PNG image", "*.png")])
            if out:
                # Save the figure at 140 DPI with a tight bounding box.
                fig.savefig(out, dpi=140, bbox_inches="tight", facecolor=BG)
                messagebox.showinfo("Saved", f"Saved to:\n{out}")

        # A row of buttons at the bottom of the pop-up window.
        btn_row = tk.Frame(win, bg=BG, pady=8)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Save as PNG…", command=save_png).pack(side="right", padx=10)
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side="right")

    # =========================================================================
    # Config — default before/after pair
    # =========================================================================

    # Class-level constant for the configuration file path.
    _CONFIG = HERE / "config.json"

    def _load_defaults(self):
        """Load the saved default before/after paths and pre-populate the detect tab."""
        # If no config file exists yet, there is nothing to load.
        if not self._CONFIG.exists():
            return
        try:
            # Read and parse the JSON file into a Python dictionary.
            cfg = json.loads(self._CONFIG.read_text())
            before = cfg.get("before", "")   # Get the "before" path, or "" if missing.
            after  = cfg.get("after",  "")   # Get the "after" path, or "" if missing.

            # Load the before image if the path is non-empty and the file exists.
            if before and Path(before).exists():
                self.before_arr   = _load_image(Path(before))
                self._before_path = Path(before)
                self._set_preview(self._before_panel, self.before_arr, Path(before).name)

            # Load the after image if the path is non-empty and the file exists.
            if after and Path(after).exists():
                self.after_arr   = _load_image(Path(after))
                self._after_path = Path(after)
                self._set_preview(self._after_panel, self.after_arr, Path(after).name)

            # Show a ready message only if both images were loaded successfully.
            if before and after:
                self._set_message("Default pair loaded. Click 'Detect Changes' when ready.")
        except Exception:
            # If anything goes wrong (corrupted JSON, etc.), silently ignore it.
            pass

    def _save_defaults(self):
        """Save the currently loaded before/after paths as the default pair."""
        # Both images must be loaded before we can save them as defaults.
        if self._before_path is None or self._after_path is None:
            messagebox.showwarning("No pair loaded",
                                   "Load both a 'Before' and 'After' image first.")
            return
        try:
            # Write the two file paths to config.json with 2-space indentation.
            self._CONFIG.write_text(json.dumps({
                "before": str(self._before_path),
                "after":  str(self._after_path),
            }, indent=2))

            # Confirm to the user that the defaults were saved.
            messagebox.showinfo("Saved",
                f"Default pair saved.\nThese images will load automatically next time.\n\n"
                f"Before: {self._before_path.name}\n"
                f"After:  {self._after_path.name}")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    # =========================================================================
    # Utilities
    # =========================================================================

    def _set_preview(self, panel, arr, name):
        # Convert the NumPy image array to a Tkinter-compatible PhotoImage thumbnail.
        photo = _to_photo(arr, PREVIEW_W, PREVIEW_H)

        # Clear whatever was previously drawn on the canvas.
        panel["canvas"].delete("all")

        # Draw the new thumbnail image centred on the canvas.
        panel["canvas"].create_image(PREVIEW_W // 2, PREVIEW_H // 2,
                                     image=photo, anchor="center")

        # Extract image height and width from the array shape (height, width, channels).
        h, w = arr.shape[:2]

        # Update the filename label to show the name and pixel dimensions.
        panel["filename"].set(f"{name}  ({w}×{h})")

        # Auto-detect the era of the image and show the result below the filename.
        # "era" key was added to panels in _image_panel; pair panels also have it.
        # We check for the key so older call sites without it don't crash.
        if "era" in panel:
            try:
                # Pass the filename so detect_image_era can extract the year
                # from it (e.g. "kallio_2001.png" → year 2001 → historic).
                # This is more reliable than image statistics for Helsinki WMS.
                era, stats = P.detect_image_era(arr, filename=name)

                # Choose what to show after the era label depending on whether
                # the year was read from the filename or estimated from pixels.
                if stats.get("source") == "filename" and stats.get("year"):
                    # Exact year known — show it clearly.
                    detail = f"year {stats['year']} from filename"
                else:
                    # Estimated from image — show the raw numbers so the user
                    # can judge and report any misclassifications.
                    detail = (f"score {stats.get('era_score', 0):.0f}"
                              f"  sat {stats['saturation']:.0f}"
                              f"  sharp {stats['sharpness']:.0f}")

                panel["era"].set(f"Era: {era}  •  {detail}")
            except Exception:
                # Era detection is best-effort; never crash the preview.
                panel["era"].set("")

        # Keep a reference to the photo in a dictionary to prevent Python's garbage
        # collector from deleting it (Tkinter images are lost if not referenced).
        self._photos[id(panel)] = photo

    def _set_message(self, msg: str):
        # Update the status bar StringVar so the label at the bottom of the window changes.
        self.message.set(msg)
        try:
            # Force Tkinter to process any pending GUI events immediately (refresh the label).
            self.root.update_idletasks()
        except tk.TclError:
            # Ignore errors that can occur if the window is closing.
            pass

    def _busy(self, on: bool, msg: str = None):
        # Start or stop the animated progress bar depending on the `on` flag.
        if on:
            self.progress.start(10)   # 10 ms update interval for smooth animation.
        else:
            self.progress.stop()

        # If a message was provided, update the status bar text as well.
        if msg:
            self._set_message(msg)


# =============================================================================
# ENTRY POINT
# =============================================================================

# "def main" defines the function that launches the application.
def main():
    # Create the main Tkinter window (the root window).
    root = tk.Tk()

    # On systems without a display (e.g. a headless server) and if the backend
    # is not already TkAgg, switch matplotlib to the non-interactive Agg backend
    # so it does not crash when trying to open a window.
    if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "tkagg":
        try:
            matplotlib.use("Agg")
        except Exception:
            pass   # If even Agg fails, let matplotlib use whatever default it has.

    # Create the App object — this builds all the widgets and starts loading the model.
    App(root)

    # "mainloop" hands control to Tkinter, which processes button clicks and key presses
    # until the user closes the window.
    root.mainloop()


# "if __name__ == '__main__'" is True only when this file is run directly
# (not when it is imported as a module by another file).
# This is the standard Python way to define the entry point of a script.
if __name__ == "__main__":
    main()
