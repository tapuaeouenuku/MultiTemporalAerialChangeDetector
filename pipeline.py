"""
================================================================================
URBAN CHANGE DETECTION IN HELSINKI USING MULTI-TEMPORAL AERIAL PHOTOGRAPHS
================================================================================

A single-file Python implementation of the full pipeline:

    1. Train a U-Net from scratch to identify buildings in aerial images
    2. Detect changes between two aerial images of the same area
    3. Highlight the changes (added / removed / preserved) in a clean format

Author: George McHugh

Usage
-----
    pip install torch torchvision numpy pillow matplotlib scipy scikit-image tqdm
    python helsinki_change_detection_all_in_one.py

This produces:
    sample_outputs/training_curves.png
    sample_outputs/change_detection_summary.png
    models/unet_buildings.pt

To run on real Helsinki aerial photographs, replace the synthetic-data block
in main() with calls to BuildingDataset(images_dir, masks_dir) pointing at a
real building-segmentation dataset (Inria, Massachusetts Buildings, WHU).
"""

# "from __future__ import annotations" lets us use newer Python type-hint syntax
# (like writing "str | None") even on older Python versions — it is a compatibility tool.
from __future__ import annotations

# "import os" loads the built-in 'os' module so we can interact with the operating system
# (e.g., read environment variables like DISPLAY).
import os
# "from dataclasses import dataclass, field" gives us tools to create simple
# data-holding classes without writing a lot of repetitive code.
from dataclasses import dataclass, field
# "from pathlib import Path" imports the Path class, which makes working with
# file and folder locations much easier than using raw strings.
from pathlib import Path
# "from typing import ..." imports type-hint helpers (Dict, List, etc.) so we
# can document what kind of data each variable or function is expected to hold.
from typing import Dict, List, Optional, Sequence, Tuple

# "import numpy as np" loads NumPy, a library for fast numerical arrays.
# We give it the short name "np" so we can write "np.array(...)" instead of "numpy.array(...)".
import numpy as np
# "from PIL import Image" imports the Image class from Pillow, a library for
# opening, converting, and saving image files (PNG, JPEG, TIFF, etc.).
from PIL import Image
# "from scipy import ndimage" imports SciPy's image-processing tools, used for
# operations like labelling connected regions and applying morphological filters.
from scipy import ndimage

# "import torch" loads PyTorch, the core deep-learning library used to build and train the neural network.
import torch
# "import torch.nn as nn" imports the neural-network building blocks (layers, loss functions, etc.).
import torch.nn as nn
# "import torch.nn.functional as F" imports stateless neural-network functions
# (e.g., activation functions) that do not have trainable parameters.
import torch.nn.functional as F
# "from torch.utils.data import DataLoader, Dataset" imports two classes:
# Dataset defines how to load individual data samples,
# DataLoader wraps a Dataset and delivers batches of samples during training.
from torch.utils.data import DataLoader, Dataset

# "import matplotlib" imports the base Matplotlib library (for backend control).
import matplotlib
# "import matplotlib.pyplot as plt" imports Matplotlib's plotting interface,
# which we use to draw graphs and images.
import matplotlib.pyplot as plt
# "import matplotlib.patches as mpatches" imports shape primitives like colored
# rectangles, used to build the legend on our result figure.
import matplotlib.patches as mpatches

# "from tqdm.auto import tqdm" imports tqdm, a library that displays a
# progress bar in the terminal so we can see how far training has advanced.
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
# Preference order:
#   1) DirectML  - any DirectX 12 GPU on Windows (AMD, Intel, NVIDIA).
#                  Installed via `pip install torch-directml`.
#   2) CUDA      - NVIDIA GPUs.
#   3) CPU       - fallback.
#
# On AMD GPUs running Windows, DirectML is the practical path. ROCm only
# works on Linux and a small handful of Windows preview builds.
# ---------------------------------------------------------------------------
# "def" starts the definition of a function named get_default_device.
# This function picks the best available hardware (GPU or CPU) for computation.
def get_default_device():
    """Return the best available compute device for training / inference."""
    # "try" begins a block where Python will attempt to run the code inside.
    # If anything goes wrong (an error is raised) it jumps to "except".
    try:
        # Try importing the optional torch-directml package for AMD/Intel GPU support on Windows.
        import torch_directml  # type: ignore
        # Check whether a DirectML-capable GPU is actually present on this machine.
        if torch_directml.is_available():
            # "return" immediately exits the function and sends the DirectML device back to the caller.
            return torch_directml.device()
    # "except ImportError" handles the case where torch_directml is not installed —
    # in that case we simply ignore the error and continue.
    except ImportError:
        pass
    # If no DirectML GPU is available, check whether an NVIDIA CUDA GPU is present.
    if torch.cuda.is_available():
        # Return a CUDA (NVIDIA) GPU device.
        return torch.device("cuda")
    # If neither GPU type is available, fall back to the CPU.
    return torch.device("cpu")


# "def device_label(device) -> str" defines a function that takes a device object
# and returns a human-readable string ("-> str" is the return-type hint).
def device_label(device) -> str:
    """Human-readable name for a device, used in GUI status strings."""
    # Convert the device object to a plain string so we can inspect its name.
    s = str(device)
    # If the string matches the internal DirectML identifier, return a friendly label.
    if s.startswith("privateuseone") or s == "dml":
        return "GPU (DirectML)"
    # If the string starts with "cuda", this is an NVIDIA GPU.
    if s.startswith("cuda"):
        return "GPU (CUDA)"
    # Otherwise it is the CPU — return a simple label.
    return "CPU"


# =============================================================================
# SECTION 1 - DATA: dataset class, augmentation, synthetic generator
# =============================================================================

# A Python set containing the file extensions we treat as images.
# Sets use curly braces { } and store unique values.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# "def _list_images(folder: Path) -> List[Path]" defines a helper function.
# The leading underscore is a naming convention meaning "private / internal use".
# It expects a Path object and returns a List of Path objects.
def _list_images(folder: Path) -> List[Path]:
    """Return a sorted list of image files under ``folder``."""
    # "sorted(...)" sorts the result alphabetically so images and masks are always paired correctly.
    # "p for p in Path(folder).iterdir()" is a generator expression that loops over every
    # file and folder inside "folder".
    # "if p.suffix.lower() in IMAGE_EXTS" keeps only files whose extension is in our set.
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


# This helper opens an image file and converts it to a NumPy array of unsigned 8-bit integers.
def _load_rgb(path: Path) -> np.ndarray:
    # "Image.open(path)" opens the image file at the given path using Pillow.
    # ".convert('RGB')" ensures the image always has exactly 3 colour channels (red, green, blue).
    # "np.asarray(..., dtype=np.uint8)" converts the Pillow image to a NumPy array where
    # each pixel value is an integer from 0 to 255.
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


# This helper loads a greyscale mask image and converts it to a binary (0 or 1) array.
def _load_mask(path: Path) -> np.ndarray:
    """Load a single-channel mask. Any pixel above 127 becomes 1."""
    # ".convert('L')" converts the image to greyscale (one channel, "L" = luminance).
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    # "(arr > 127)" creates a boolean array: True where the pixel is brighter than half-white.
    # ".astype(np.uint8)" converts True -> 1, False -> 0, giving a clean binary mask.
    return (arr > 127).astype(np.uint8)


# "class BuildingDataset(Dataset):" defines a new class called BuildingDataset.
# "Dataset" inside the parentheses means BuildingDataset *inherits* from PyTorch's Dataset class,
# so PyTorch knows how to use it with DataLoader.
class BuildingDataset(Dataset):
    """A folder-based building-segmentation dataset.

    Parameters
    ----------
    images_dir, masks_dir
        Directories of matched image / mask files. Pairing is done by sorted
        filename, which matches the layout of Inria, Massachusetts Buildings,
        and WHU.
    tile_size
        Random crops of this size are taken from each tile during training.
        Set to ``None`` to return the full image unchanged.
    augment
        Whether to apply random flip / rotate / brightness augmentations.
    """

    # "__init__" is a special method that Python calls automatically when you create
    # a new instance of the class (e.g., BuildingDataset(...)).
    # "self" refers to the specific object being created.
    def __init__(self, images_dir, masks_dir, tile_size: Optional[int] = 256,
                 augment: bool = True):
        # List all image files in the images folder and store them on the object.
        self.images = _list_images(images_dir)
        # List all mask files in the masks folder and store them on the object.
        self.masks = _list_images(masks_dir)
        # Check that the number of images matches the number of masks.
        if len(self.images) != len(self.masks):
            # "raise ValueError" stops the program immediately and reports the mismatch.
            raise ValueError(
                f"Image / mask count mismatch: "
                f"{len(self.images)} images vs {len(self.masks)} masks"
            )
        # Store the desired crop size so other methods can access it.
        self.tile_size = tile_size
        # Store whether data augmentation should be applied during loading.
        self.augment = augment

    # "__len__" is a special method that Python calls when you write len(dataset).
    # It tells the DataLoader how many samples are in the dataset.
    def __len__(self) -> int:
        return len(self.images)

    # This internal method cuts a random square crop out of an image and its paired mask.
    def _random_crop(self, image, mask):
        # "h, w" are the height and width of the image; image.shape[:2] gives (height, width).
        h, w = image.shape[:2]
        ts = self.tile_size
        # If no crop size is set, or the image is already small enough, return it unchanged.
        if ts is None or (h <= ts and w <= ts):
            return image, mask
        # Pick a random top-left y coordinate so the crop fits within the image height.
        y = np.random.randint(0, h - ts + 1)
        # Pick a random top-left x coordinate so the crop fits within the image width.
        x = np.random.randint(0, w - ts + 1)
        # Slice the image and mask arrays to the crop region and return both.
        return image[y:y + ts, x:x + ts], mask[y:y + ts, x:x + ts]

    # "@staticmethod" means this method belongs to the class but does not need "self" —
    # it does not access any instance-specific data.
    @staticmethod
    def _augment(image, mask):
        # --- Geometric augmentations (applied to both image and mask) -----------

        # With 50% probability, flip the image and mask left-to-right (horizontal mirror).
        if np.random.rand() < 0.5:
            image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()
        # With 50% probability, flip the image and mask upside-down (vertical mirror).
        if np.random.rand() < 0.5:
            image, mask = image[::-1].copy(), mask[::-1].copy()
        # Choose a random multiple of 90 degrees to rotate (0, 90, 180, or 270).
        k = np.random.randint(0, 4)
        if k:
            image = np.rot90(image, k=k).copy()
            mask  = np.rot90(mask,  k=k).copy()

        # --- Photometric augmentations (image only, mask is unchanged) ----------
        # These simulate the large visual differences between aerial photos taken
        # in different years with different cameras and lighting conditions.
        # A model trained with these augmentations will generalise much better
        # when comparing e.g. a crisp 2025 image against a blurry 2001 scan.

        img = image.astype(np.float32)

        # Global brightness: multiply all channels by a factor in [0.60, 1.40].
        # This is wider than before (was ±15%) to cover the brightness gap between
        # summer and winter aerial surveys, and between old and new sensors.
        gain = np.random.uniform(0.60, 1.40)
        img  = img * gain

        # Gamma correction (40% chance): raises or lowers mid-tone contrast without
        # clipping the highlights. gamma < 1 brightens mid-tones; gamma > 1 darkens them.
        if np.random.rand() < 0.40:
            gamma = np.random.uniform(0.70, 1.40)
            img   = 255.0 * np.power(np.clip(img / 255.0, 1e-6, 1.0), gamma)

        # Per-channel colour jitter (30% chance): shifts each colour channel (R, G, B)
        # by a different random amount in [-30, +30]. This simulates different white-balance
        # settings between cameras and seasonal colour shifts (green summer vs brown autumn).
        if np.random.rand() < 0.30:
            for c in range(img.shape[2]):
                img[:, :, c] += np.random.uniform(-30, 30)

        # Gaussian blur (35% chance): smooths the image with a random kernel.
        # sigma in [0.5, 2.5] simulates the softness of older low-resolution aerial scans
        # so the model does not become over-fitted to the sharpness of modern 5 cm imagery.
        if np.random.rand() < 0.35:
            import cv2 as _cv2
            sigma = np.random.uniform(0.5, 2.5)
            # Kernel size must be an odd number; choose it proportional to sigma.
            ksize = max(3, int(sigma * 3) | 1)
            img   = _cv2.GaussianBlur(img, (ksize, ksize), sigma)

        # Clamp all pixel values back to the valid [0, 255] range and convert to uint8.
        image = np.clip(img, 0, 255).astype(np.uint8)
        return image, mask

    # "__getitem__" is a special method Python calls when you index a dataset (e.g., dataset[5]).
    # The DataLoader uses it to fetch individual samples.
    def __getitem__(self, idx):
        # Load the RGB image at position idx.
        image = _load_rgb(self.images[idx])
        # Load the corresponding binary mask at the same position.
        mask = _load_mask(self.masks[idx])
        # Apply the random crop to make both image and mask the same tile size.
        image, mask = self._random_crop(image, mask)
        # Apply random augmentations if augmentation is turned on.
        if self.augment:
            image, mask = self._augment(image, mask)
        # Rearrange the image from (Height, Width, Channels) to (Channels, Height, Width),
        # which is the format PyTorch's convolutional layers expect.
        # Then convert to a float tensor and scale pixel values from 0–255 down to 0.0–1.0.
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        # Convert the mask to a float tensor and add an extra dimension at position 0
        # so the shape is (1, Height, Width) — matching the model's single-channel output.
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)
        return image_t, mask_t


class ChangeDataset(Dataset):
    """Dataset of (before, after, change_mask) triplets for Siamese training.

    Expected directory layout (produced by generate_change_dataset):
        root/
          before/   ← 'before' aerial images  (RGB PNG)
          after/    ← 'after'  aerial images  (RGB PNG)
          changes/  ← binary change masks     (grayscale PNG, 255 = changed)

    Augmentation strategy
    ---------------------
    Geometric transforms (flip / rotate) are applied IDENTICALLY to all
    three arrays so their spatial correspondence is preserved.

    Photometric transforms (brightness / gamma / blur) are applied
    INDEPENDENTLY to 'before' and 'after' — this deliberately teaches the
    network to ignore sensor and lighting differences between image epochs
    and focus only on genuine structural changes.
    """

    def __init__(self, before_dir, after_dir, change_dir,
                 tile_size: Optional[int] = 256, augment: bool = True):
        self.befores = _list_images(Path(before_dir))
        self.afters  = _list_images(Path(after_dir))
        self.changes = _list_images(Path(change_dir))

        if not (len(self.befores) == len(self.afters) == len(self.changes)):
            raise ValueError(
                f"Triplet count mismatch: "
                f"{len(self.befores)} before / {len(self.afters)} after / "
                f"{len(self.changes)} change images"
            )
        self.tile_size = tile_size
        self.augment   = augment

    def __len__(self) -> int:
        return len(self.befores)

    def _random_crop(self, before, after, change):
        """Apply the same random crop to before, after, and change mask."""
        h, w = before.shape[:2]
        ts   = self.tile_size
        if ts is None or (h <= ts and w <= ts):
            return before, after, change
        y = np.random.randint(0, h - ts + 1)
        x = np.random.randint(0, w - ts + 1)
        return (before[y:y + ts, x:x + ts],
                after [y:y + ts, x:x + ts],
                change[y:y + ts, x:x + ts])

    @staticmethod
    def _geometric_augment(before, after, change):
        """Identical flip / rotate applied to all three arrays."""
        # Horizontal flip
        if np.random.rand() < 0.5:
            before = before[:, ::-1].copy()
            after  = after [:, ::-1].copy()
            change = change[:, ::-1].copy()
        # Vertical flip
        if np.random.rand() < 0.5:
            before = before[::-1].copy()
            after  = after [::-1].copy()
            change = change[::-1].copy()
        # Rotation in multiples of 90°
        k = np.random.randint(0, 4)
        if k:
            before = np.rot90(before, k=k).copy()
            after  = np.rot90(after,  k=k).copy()
            change = np.rot90(change, k=k).copy()
        return before, after, change

    @staticmethod
    def _photo_augment(image: np.ndarray) -> np.ndarray:
        """Independent photometric augmentation per image."""
        img = image.astype(np.float32)
        # Brightness gain
        img = img * np.random.uniform(0.60, 1.40)
        # Gamma correction (40% chance)
        if np.random.rand() < 0.40:
            gamma = np.random.uniform(0.70, 1.40)
            img   = 255.0 * np.power(np.clip(img / 255.0, 1e-6, 1.0), gamma)
        # Per-channel colour jitter (30% chance)
        if np.random.rand() < 0.30:
            for c in range(img.shape[2]):
                img[:, :, c] += np.random.uniform(-30, 30)
        # Gaussian blur (35% chance) — simulates older, lower-res imagery
        if np.random.rand() < 0.35:
            import cv2 as _cv2
            sigma = np.random.uniform(0.5, 2.5)
            ksize = max(3, int(sigma * 3) | 1)
            img   = _cv2.GaussianBlur(img, (ksize, ksize), sigma)
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, idx):
        before = _load_rgb(self.befores[idx])
        after  = _load_rgb(self.afters[idx])
        change = _load_mask(self.changes[idx])

        before, after, change = self._random_crop(before, after, change)

        if self.augment:
            before, after, change = self._geometric_augment(before, after, change)
            # Photometric aug is independent per image (same scene, different era)
            before = self._photo_augment(before)
            after  = self._photo_augment(after)

        # Convert to PyTorch tensors in [C, H, W] layout, values in [0, 1].
        before_t = torch.from_numpy(before.transpose(2, 0, 1)).float() / 255.0
        after_t  = torch.from_numpy(after.transpose(2, 0, 1)).float() / 255.0
        change_t = torch.from_numpy(change).float().unsqueeze(0)
        return before_t, after_t, change_t


# "@dataclass" is a decorator that automatically generates boilerplate methods
# like __init__ and __repr__ for a class, based on the fields we declare below.
@dataclass
class SyntheticConfig:
    """Parameters for the synthetic aerial generator."""
    # Size (in pixels) of each generated square tile.
    tile_size: int = 256
    # Number of training tiles to generate.
    n_train: int = 80
    # Number of validation tiles to generate.
    n_val: int = 20
    # Minimum number of buildings to place on a single tile.
    min_buildings: int = 4
    # Maximum number of buildings to place on a single tile.
    max_buildings: int = 18
    # Random seed so the generated data is reproducible across runs.
    seed: int = 42


# This function creates a noisy green-brown background to imitate grass and vegetation.
def _add_grass_texture(rng: np.random.Generator, size: int) -> np.ndarray:
    """Noisy green-brown background that mimics vegetation."""
    # "rng.normal(loc=[...], scale=12, size=...)" generates an array of random numbers
    # drawn from a normal distribution centred on the given RGB colour values.
    base = rng.normal(loc=[95, 135, 70], scale=12, size=(size, size, 3))
    # Generate a coarse low-resolution noise pattern (1/16 of the tile size).
    coarse = rng.normal(loc=0, scale=18, size=(size // 16, size // 16, 3))
    # "np.kron" tiles (repeats) the coarse pattern to fill the full tile size,
    # creating large blotchy variations that look like patches of light and shadow.
    coarse = np.kron(coarse, np.ones((16, 16, 1)))[:size, :size]
    # Add the fine and coarse noise together, clamp to 0–255, and return as uint8.
    return np.clip(base + coarse, 0, 255).astype(np.uint8)


# This function paints a grey road stripe (horizontal or vertical) across the tile.
def _draw_road(image: np.ndarray, rng: np.random.Generator) -> None:
    """Paint a horizontal or vertical asphalt strip across the tile."""
    # Get the height and width of the image (we ignore the channel count with "_").
    h, w, _ = image.shape
    # Choose a random road width between 10 and 21 pixels.
    width = int(rng.integers(10, 22))
    # With 50% probability, draw a horizontal road; otherwise draw a vertical one.
    if rng.random() < 0.5:
        # Pick a random vertical position that keeps the full road width inside the tile.
        y = int(rng.integers(width, h - width))
        # Paint the road strip an asphalt grey colour (values between 70 and 94).
        image[y - width // 2: y + width // 2] = rng.integers(70, 95)
    else:
        # Pick a random horizontal position for a vertical road strip.
        x = int(rng.integers(width, w - width))
        # Paint the vertical road strip with the same grey colour range.
        image[:, x - width // 2: x + width // 2] = rng.integers(70, 95)


# This function draws one rectangular building "roof" on the image and marks it in the mask.
def _draw_building(image, mask, rng):
    """Drop one axis-aligned rectangular roof onto the tile and its mask."""
    # Get image dimensions.
    h, w, _ = image.shape
    # Choose a random building height between 15 and 54 pixels.
    bh = int(rng.integers(15, 55))
    # Choose a random building width between 15 and 54 pixels.
    bw = int(rng.integers(15, 55))
    # Choose a random top-left y position so the building fits within the tile.
    y = int(rng.integers(0, max(1, h - bh)))
    # Choose a random top-left x position so the building fits within the tile.
    x = int(rng.integers(0, max(1, w - bw)))
    # A small collection of realistic roof colours (terracotta, concrete, bitumen, pale stone).
    palette = np.array([
        [165, 75, 55],   # terracotta
        [120, 120, 120], # concrete
        [80, 60, 50],    # bitumen
        [200, 190, 170], # pale stone
    ])
    # Pick one colour from the palette at random.
    colour = palette[rng.integers(0, len(palette))]
    # Add a small random colour variation (jitter) so identical buildings look slightly different.
    jitter = rng.normal(0, 8, size=3)
    # Paint the building rectangle onto the image, clamping colour values to 0–255.
    image[y:y + bh, x:x + bw] = np.clip(colour + jitter, 0, 255)
    # Mark the same rectangle as "building" (value = 1) in the ground-truth mask.
    mask[y:y + bh, x:x + bw] = 1


# This function creates one (image, mask) pair by combining a grass background,
# an optional road, and several randomly placed buildings.
def _make_pair(rng: np.random.Generator, cfg: SyntheticConfig):
    # Start with a grassy background.
    image = _add_grass_texture(rng, cfg.tile_size)
    # With 70% probability, add a road stripe across the tile.
    if rng.random() < 0.7:
        _draw_road(image, rng)
    # Create an all-zero mask (no buildings yet).
    mask = np.zeros((cfg.tile_size, cfg.tile_size), dtype=np.uint8)
    # Choose a random number of buildings to place on this tile.
    n = int(rng.integers(cfg.min_buildings, cfg.max_buildings + 1))
    # "for _ in range(n)" repeats the loop body n times; "_" means we don't need the loop counter.
    for _ in range(n):
        # Draw one building on the image and record it in the mask.
        _draw_building(image, mask, rng)
    return image, mask


# This function generates the entire synthetic dataset and saves it to disk.
def generate_synthetic_dataset(out_dir, cfg: Optional[SyntheticConfig] = None) -> Path:
    """Materialise a synthetic train/val dataset on disk."""
    # Use the provided config or create a default one if none was given.
    cfg = cfg or SyntheticConfig()
    # Create a reproducible random number generator using the configured seed.
    rng = np.random.default_rng(cfg.seed)
    # Turn the output directory string/path into a Path object.
    out = Path(out_dir)

    # Loop over both splits: "train" (80 samples) and "val" (20 samples).
    for split, count in [("train", cfg.n_train), ("val", cfg.n_val)]:
        # Build paths to the images and masks sub-folders for this split.
        img_dir = out / split / "images"
        msk_dir = out / split / "masks"
        # "mkdir(parents=True, exist_ok=True)" creates the folder (and any parent folders)
        # if they do not already exist; does nothing if they do.
        img_dir.mkdir(parents=True, exist_ok=True)
        msk_dir.mkdir(parents=True, exist_ok=True)
        # Generate the specified number of image/mask pairs for this split.
        for i in range(count):
            image, mask = _make_pair(rng, cfg)
            # Save the image as a PNG file; the filename is zero-padded to 4 digits (e.g., tile_0003.png).
            Image.fromarray(image).save(img_dir / f"tile_{i:04d}.png")
            # Save the mask; multiply by 255 so building pixels appear white (0 = black, 1*255 = white).
            Image.fromarray(mask * 255).save(msk_dir / f"tile_{i:04d}.png")
    return out


# This function creates a pair of synthetic "before" and "after" aerial images,
# simulating urban change over time (some buildings added, some demolished).
def generate_temporal_pair(cfg: Optional[SyntheticConfig] = None, seed: int = 7,
                           n_added: int = 3, n_removed: int = 2):
    """Build a 'before' and 'after' synthetic aerial pair plus ground-truth masks.

    The 'after' image keeps most buildings from 'before' but adds ``n_added``
    new ones and demolishes ``n_removed``.
    """
    # Default to a 384-pixel tile if no config is provided.
    cfg = cfg or SyntheticConfig(tile_size=384)
    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Create the "before" background grass texture.
    before_img = _add_grass_texture(rng, cfg.tile_size)
    # Add a road to the before image.
    _draw_road(before_img, rng)
    # Start with an empty mask for the before image.
    before_mask = np.zeros((cfg.tile_size, cfg.tile_size), dtype=np.uint8)
    # Choose how many buildings appear in the "before" scene.
    n_initial = int(rng.integers(cfg.min_buildings, cfg.max_buildings))
    # Place that many buildings on the before image and mask.
    for _ in range(n_initial):
        _draw_building(before_img, before_mask, rng)

    # The "after" image starts as an exact copy of "before".
    after_img = before_img.copy()
    # The "after" mask also starts as a copy of the before mask.
    after_mask = before_mask.copy()

    # Generate a fresh grass patch that will replace demolished building footprints.
    grass_patch = _add_grass_texture(np.random.default_rng(seed + 1), cfg.tile_size)
    # "ndimage.label" finds and labels each separate connected group of building pixels.
    # "labelled" is an array where each building region has a unique integer ID.
    # "n_components" is the total number of separate building regions found.
    labelled, n_components = ndimage.label(after_mask)
    if n_components > 0:
        # Build a list of all component IDs (1, 2, 3, ..., n_components).
        ids = list(range(1, n_components + 1))
        # Shuffle the list randomly so we pick arbitrary buildings to demolish.
        rng.shuffle(ids)
        # Iterate over the first n_removed IDs (but no more than actually exist).
        for comp_id in ids[: min(n_removed, n_components)]:
            # Create a boolean mask that is True only for pixels belonging to this building.
            region = labelled == comp_id
            # Replace the building pixels in the after image with grass texture.
            after_img[region] = grass_patch[region]
            # Remove the building from the after mask (set to 0 = background).
            after_mask[region] = 0

    # Add the requested number of new buildings to the "after" image and mask.
    for _ in range(n_added):
        _draw_building(after_img, after_mask, rng)

    # Return both images and both ground-truth masks.
    return before_img, before_mask, after_img, after_mask


def generate_change_dataset(out_dir, cfg: Optional[SyntheticConfig] = None,
                            n_pairs: int = 200, seed: int = 42) -> Path:
    """Generate synthetic (before, after, change_mask) triplets for Siamese training.

    For each pair the generator:
      1. Creates a 'before' scene with some buildings.
      2. Randomly adds and removes buildings to produce an 'after' scene.
      3. Computes the change mask as the XOR of the two building masks
         (pixels that changed = added OR removed building footprint).

    Written layout::

        out_dir/
          before/pair_NNNN.png    ← 'before' aerial image
          after/ pair_NNNN.png    ← 'after'  aerial image
          changes/pair_NNNN.png   ← binary change mask (255 = changed)

    Parameters
    ----------
    out_dir : str or Path
        Root folder to write triplets into (created if absent).
    cfg : SyntheticConfig, optional
        Tile size and building count parameters.  Defaults to 256 px tiles
        with 3–12 buildings per scene.
    n_pairs : int
        Number of triplets to generate (default 200).
    seed : int
        Master random seed for reproducibility (default 42).

    Returns
    -------
    Path
        Path to ``out_dir``.
    """
    cfg = cfg or SyntheticConfig(tile_size=256, min_buildings=3, max_buildings=12)

    before_dir  = Path(out_dir) / "before"
    after_dir   = Path(out_dir) / "after"
    changes_dir = Path(out_dir) / "changes"

    for d in (before_dir, after_dir, changes_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Master RNG that produces per-pair seeds and change counts deterministically.
    rng_meta = np.random.default_rng(seed)

    for i in range(n_pairs):
        pair_seed = int(rng_meta.integers(0, 1_000_000))
        n_added   = int(rng_meta.integers(1, 5))   # 1–4 new buildings per pair
        n_removed = int(rng_meta.integers(0, 4))   # 0–3 demolished buildings per pair

        before_img, before_mask, after_img, after_mask = generate_temporal_pair(
            cfg=cfg, seed=pair_seed, n_added=n_added, n_removed=n_removed)

        # Change mask = pixels that differ between before and after masks (XOR).
        # A pixel is "changed" if it was building before but not after (demolition)
        # OR not building before but building after (construction).
        before_bin  = before_mask > 127
        after_bin   = after_mask  > 127
        change_mask = (before_bin ^ after_bin).astype(np.uint8) * 255

        stem = f"pair_{i:04d}.png"
        Image.fromarray(before_img).save(before_dir  / stem)
        Image.fromarray(after_img ).save(after_dir   / stem)
        Image.fromarray(change_mask, mode="L").save(changes_dir / stem)

    return Path(out_dir)


# =============================================================================
# SECTION 2 - MODEL: U-Net architecture (from scratch)
# =============================================================================

# "class DoubleConv(nn.Module):" defines a neural-network block that inherits from
# PyTorch's base Module class.  All custom layers must extend nn.Module.
class DoubleConv(nn.Module):
    """Two 3x3 conv -> BatchNorm -> ReLU blocks in series."""

    # "__init__" sets up the layers of this block when it is created.
    def __init__(self, in_channels: int, out_channels: int):
        # "super().__init__()" calls the parent class (nn.Module) constructor —
        # this is required for PyTorch to track all the layers correctly.
        super().__init__()
        # "nn.Sequential" chains multiple layers so data passes through them one after another.
        self.block = nn.Sequential(
            # First 2D convolution: slides a 3x3 filter over the input to detect features.
            # "padding=1" adds a 1-pixel border so the output is the same size as the input.
            # "bias=False" because the following normalisation layer makes bias redundant.
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            # InstanceNorm2d normalises each feature map independently,
            # which stabilises training and helps the model converge faster.
            nn.InstanceNorm2d(out_channels, affine=True),
            # ReLU (Rectified Linear Unit) is the activation function: sets negative values to 0.
            # "inplace=True" saves memory by modifying the tensor directly instead of making a copy.
            nn.ReLU(inplace=True),

            # Second convolution block with the same number of filters (out_channels -> out_channels).
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            # Normalise the output of the second convolution.
            nn.InstanceNorm2d(out_channels, affine=True),
            # Another ReLU activation after the second convolution.
            nn.ReLU(inplace=True),
)

    # "forward" defines what happens when you pass data through this block.
    # PyTorch calls this automatically when you use the block like a function: block(x).
    def forward(self, x):
        # Pass the input x through all the layers defined in self.block and return the result.
        return self.block(x)


# This block reduces the spatial resolution of feature maps by a factor of 2 (downsampling).
class Down(nn.Module):
    """Max-pool downsample then DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # MaxPool2d takes the maximum value in each 2x2 window, halving the height and width.
        self.pool = nn.MaxPool2d(2)
        # After pooling, apply the double-convolution block to extract richer features.
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # First downsample x with max-pooling, then run through the double-conv block.
        return self.conv(self.pool(x))


# This block doubles the spatial resolution of feature maps (upsampling) and combines
# them with a matching feature map saved from the downsampling path (a "skip connection").
class Up(nn.Module):
    """Upsample, concatenate the skip connection, then DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # ConvTranspose2d is a "learnable upsampling" layer that doubles the spatial size.
        # kernel_size=2, stride=2 ensures the output is exactly twice the input size.
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # After concatenating the upsampled features with the skip connection,
        # apply a double-conv block to blend the combined information.
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        # Upsample x to double its spatial size.
        x = self.up(x)
        # Calculate any size difference between the skip connection and the upsampled x.
        # Slight mismatches can occur when the original size is not perfectly divisible by 2.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        # "F.pad" adds padding to x so its size matches the skip connection.
        x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2,
                      diff_y // 2, diff_y - diff_y // 2])
        # "torch.cat" concatenates the skip connection and the upsampled x along the channel axis (dim=1).
        x = torch.cat([skip, x], dim=1)
        # Pass the combined tensor through the double-conv block and return.
        return self.conv(x)


# The full U-Net model: a symmetric encoder-decoder with skip connections.
class UNet(nn.Module):
    """Binary-segmentation U-Net (Ronneberger et al. 2015)."""

    # "in_channels=3" because aerial photos are RGB (3 colours).
    # "out_channels=1" because we output a single probability map (building vs. not).
    # "widths" controls how many filters (feature maps) each level of the U-Net has.
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 widths: Sequence[int] = (32, 64, 128, 256, 512)):
        super().__init__()
        # We need exactly 5 width values for a 4-level deep U-Net.
        if len(widths) != 5:
            raise ValueError("Expected 5 channel widths for a depth-4 U-Net.")
        # Unpack the five width values into named variables for clarity.
        w1, w2, w3, w4, w5 = widths

        # "inc" (input convolution): the very first double-conv block applied to the raw image.
        self.inc = DoubleConv(in_channels, w1)
        # Four downsampling steps — each halves the spatial size and doubles the feature depth.
        self.down1 = Down(w1, w2)
        self.down2 = Down(w2, w3)
        self.down3 = Down(w3, w4)
        # The deepest level (the bottleneck) — smallest spatial size, most features.
        self.down4 = Down(w4, w5)

        # Four upsampling steps — each doubles the spatial size and reduces the feature depth,
        # while merging a skip connection from the corresponding downsampling level.
        self.up1 = Up(w5, w4, w4)
        self.up2 = Up(w4, w3, w3)
        self.up3 = Up(w3, w2, w2)
        self.up4 = Up(w2, w1, w1)

        # "outc" (output convolution): a 1x1 convolution that maps w1 feature channels
        # down to a single output channel (raw "logit" score per pixel).
        self.outc = nn.Conv2d(w1, out_channels, kernel_size=1)

    # "forward" defines the full forward pass: data flows down through the encoder,
    # across the bottleneck, then back up through the decoder.
    def forward(self, x):
        # Encoder path — save each feature map for use as a skip connection.
        x1 = self.inc(x)      # Full resolution.
        x2 = self.down1(x1)   # 1/2 resolution.
        x3 = self.down2(x2)   # 1/4 resolution.
        x4 = self.down3(x3)   # 1/8 resolution.
        x5 = self.down4(x4)   # 1/16 resolution (bottleneck).
        # Decoder path — upsample and merge skip connections in reverse order.
        x = self.up1(x5, x4)  # 1/8 resolution.
        x = self.up2(x, x3)   # 1/4 resolution.
        x = self.up3(x, x2)   # 1/2 resolution.
        x = self.up4(x, x1)   # Full resolution.
        # Apply the final 1x1 convolution to get one raw score per pixel.
        return self.outc(x)

    # "@torch.no_grad()" is a decorator that tells PyTorch not to track gradients
    # inside this method — used for quick utility calls that don't involve training.
    @torch.no_grad()
    def count_parameters(self) -> int:
        # "sum(...)" adds up the total number of trainable numbers (weights) in the model.
        # "p.numel()" returns the number of elements in a parameter tensor.
        # "if p.requires_grad" skips frozen (non-trainable) parameters.
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SiameseUNet(nn.Module):
    """Early-fusion change-detection network (FC-EF style).

    Concatenates the 'before' and 'after' images along the channel axis,
    producing a 6-channel input tensor, then passes it through a standard
    U-Net that outputs a single-channel change-probability map.

    Why this is better than running U-Net twice and diffing the masks
    ----------------------------------------------------------------
    Running U-Net independently on each image and subtracting the results
    compounds two separate sources of error: a building missed in one image
    appears as a false "change" even if it was always there. With early
    fusion the network sees BOTH images simultaneously and learns directly
    what *differences* look like — handling brightness, colour, and
    resolution gaps between years without any explicit normalisation.
    """

    def __init__(self, widths: Sequence[int] = (32, 64, 128, 256, 512)):
        super().__init__()
        # Store widths so we can write them into checkpoints for auto-detection.
        self.widths = tuple(widths)
        # Inner U-Net: identical to the segmentation one but with 6 input channels
        # (3 for 'before' + 3 for 'after').
        self.unet = UNet(in_channels=6, widths=widths)

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        before, after : [B, 3, H, W] float tensors, pixel values in [0, 1].

        Returns
        -------
        [B, 1, H, W] raw logits — apply sigmoid to get change probabilities.
        """
        # Concatenate along the channel axis:
        # [B, 3, H, W] cat [B, 3, H, W] → [B, 6, H, W]
        x = torch.cat([before, after], dim=1)
        return self.unet(x)

    @torch.no_grad()
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# SECTION 3 - TRAINING: losses, metrics, train loop, checkpoint I/O
# =============================================================================

# The Dice loss measures how much the predicted mask overlaps with the true mask.
# It works well for building segmentation because buildings are rare (class imbalance).
def dice_loss(logits, targets, eps: float = 1e-6):
    """Soft Dice loss for a single foreground class."""
    # "torch.sigmoid(logits)" converts raw model outputs (any real number) to probabilities in [0, 1].
    # ".view(logits.size(0), -1)" flattens all spatial dimensions into a single long vector per image.
    probs = torch.sigmoid(logits).view(logits.size(0), -1)
    # Flatten the targets the same way.
    targets = targets.view(targets.size(0), -1)
    # Intersection = sum of elementwise products — high when prediction and target agree.
    intersection = (probs * targets).sum(dim=1)
    # Union = sum of both — the total "mass" of predictions and targets.
    union = probs.sum(dim=1) + targets.sum(dim=1)
    # Dice coefficient: 2 * overlap / total (ranges from 0 = no overlap to 1 = perfect match).
    # "eps" prevents division by zero when both masks are empty.
    dice = (2.0 * intersection + eps) / (union + eps)
    # Loss = 1 - Dice, so a perfect prediction gives loss = 0.
    return 1.0 - dice.mean()


# This function combines two loss terms for more stable and accurate training.
def combined_loss(logits, targets):
    """0.5 * BCE (pos_weight=2) + 0.5 * Dice. The pos_weight penalises missed
    buildings 2x more than false positives, counteracting the large class
    imbalance between building and background pixels."""
    # Binary Cross-Entropy (BCE) loss measures prediction confidence.
    # "pos_weight=2.0" means missing a building pixel is penalised twice as heavily
    # as incorrectly predicting a building — this compensates for buildings being rare.
    bce = F.binary_cross_entropy_with_logits(
        logits, targets,
        pos_weight=torch.tensor(2.0, device=logits.device)
    )
    # Return the average of BCE loss and Dice loss.
    return 0.5 * bce + 0.5 * dice_loss(logits, targets)


# "@torch.no_grad()" tells PyTorch to skip gradient tracking — faster and uses less memory
# since we are only evaluating, not training.
@torch.no_grad()
def segmentation_metrics(logits, targets, threshold: float = 0.5) -> Dict[str, float]:
    """Compute IoU, F1, and pixel accuracy for a batch of predictions."""
    # Apply sigmoid to get probabilities, then threshold to get a binary 0/1 prediction.
    preds = (torch.sigmoid(logits) > threshold).float()
    # True Positive (TP): predicted building AND actually building.
    tp = ((preds == 1) & (targets == 1)).sum().item()
    # False Positive (FP): predicted building BUT actually background.
    fp = ((preds == 1) & (targets == 0)).sum().item()
    # False Negative (FN): predicted background BUT actually building.
    fn = ((preds == 0) & (targets == 1)).sum().item()
    # True Negative (TN): predicted background AND actually background.
    tn = ((preds == 0) & (targets == 0)).sum().item()
    # Small constant to avoid dividing by zero.
    eps = 1e-6
    # Return a dictionary of three common evaluation metrics.
    return {
        # IoU (Intersection over Union): how much the predicted and true regions overlap.
        "iou": tp / (tp + fp + fn + eps),
        # F1 score: the harmonic mean of precision and recall (balanced measure).
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        # Pixel accuracy: fraction of all pixels that were classified correctly.
        "pixel_acc": (tp + tn) / (tp + tn + fp + fn + eps),
    }


# Another dataclass — this one holds all the hyperparameters controlling how training runs.
@dataclass
class TrainConfig:
    """Hyperparameters and runtime settings for a training run."""
    # Number of times to loop through the entire training dataset.
    epochs: int = 20
    # Learning rate: controls how large each gradient update step is.
    lr: float = 1e-3
    # Weight decay: a regularisation term that penalises very large weights to prevent overfitting.
    weight_decay: float = 1e-5
    # Batch size: how many images are processed together in each training step.
    batch_size: int = 8
    # Number of parallel worker processes for loading data (0 = load on the main thread).
    num_workers: int = 2
    # NOTE: this field can hold either a string ("cuda" / "cpu") or a
    # torch.device object (DirectML returns one of these). Both work with
    # tensor.to(device) and model.to(device) the same way.
    # "field(default_factory=...)" calls get_default_device() each time a new TrainConfig
    # is created so the device is detected freshly rather than once at import time.
    device: object = field(default_factory=get_default_device)
    # File path where the best model weights will be saved.
    checkpoint_path: Optional[str] = "models/unet_buildings.pt"
    # How often (in batches) to print progress to the terminal during training.
    log_every: int = 10


# This internal function runs the model through the data in one "loader" (train or validation).
def _run_epoch(model, loader, optimizer, device, desc: str) -> Dict[str, float]:
    # If an optimizer is provided, this is a training epoch; otherwise it is a validation epoch.
    is_train = optimizer is not None
    # "model.train(True)" puts the model in training mode (activates dropout, batch norm updates).
    # "model.train(False)" switches to evaluation mode.
    model.train(is_train)
    # Dictionary to accumulate the sum of each metric across all batches.
    running = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "pixel_acc": 0.0}
    n_batches = 0

    # non_blocking=True is a CUDA pinned-memory optimization; a no-op on
    # DirectML or CPU, so only set it when we're actually on CUDA.
    is_cuda = str(device).startswith("cuda")
    # Wrap the loader in tqdm to display a progress bar; "leave=False" clears it when done.
    progress = tqdm(loader, desc=desc, leave=False)
    # Loop over each batch of (images, masks) from the data loader.
    for images, masks in progress:
        # Move data to the selected device (GPU or CPU).
        images = images.to(device, non_blocking=is_cuda)
        masks = masks.to(device, non_blocking=is_cuda)

        # "torch.set_grad_enabled(is_train)" enables or disables gradient tracking
        # depending on whether we are training (True) or validating (False).
        with torch.set_grad_enabled(is_train):
            # Forward pass: run the images through the model to get raw output (logits).
            logits = model(images)
            # Compute how wrong the model's predictions were.
            loss = combined_loss(logits, masks)
            # Only update model weights during training.
            if is_train:
                # Reset gradients from the previous batch (they accumulate by default).
                optimizer.zero_grad()
                # "loss.backward()" computes gradients via backpropagation.
                loss.backward()
                # "optimizer.step()" applies the gradients to update the model's weights.
                optimizer.step()

        # Compute accuracy metrics for this batch (detached from the computation graph).
        metrics = segmentation_metrics(logits.detach(), masks)
        # Accumulate the loss value (a plain Python float, not a tensor).
        running["loss"] += loss.item()
        # Accumulate each metric.
        for k in ("iou", "f1", "pixel_acc"):
            running[k] += metrics[k]
        n_batches += 1
        # Update the progress bar with the latest batch loss and IoU.
        progress.set_postfix(loss=f"{loss.item():.4f}", iou=f"{metrics['iou']:.3f}")

    # Return the average of each metric over all batches in this epoch.
    # "max(n_batches, 1)" prevents division by zero if the loader was empty.
    return {k: v / max(n_batches, 1) for k, v in running.items()}


# The main training function: runs all epochs and returns the history of metrics.
def train(model: UNet, train_loader, val_loader, config: TrainConfig) -> Dict[str, List[float]]:
    """Train ``model`` and return per-epoch training and validation metrics."""
    device = config.device
    # Move the model's weights to the selected device.
    model.to(device)
    # Adam is an adaptive gradient optimizer — it adjusts the learning rate per parameter.
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    # CosineAnnealingLR gradually reduces the learning rate following a cosine curve,
    # which often improves final performance compared to a constant learning rate.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs))

    # Dictionary to store the training and validation metrics for every epoch.
    history = {"train_loss": [], "train_iou": [], "train_f1": [],
               "val_loss": [], "val_iou": [], "val_f1": []}

    # Track the best validation IoU seen so far so we only save improved checkpoints.
    best_val_iou = -1.0
    if config.checkpoint_path:
        # Create the folder for the checkpoint file if it does not already exist.
        Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Loop from epoch 1 to config.epochs (inclusive).
    for epoch in range(1, config.epochs + 1):
        # Run one training epoch and get average metrics.
        train_metrics = _run_epoch(model, train_loader, optimizer, device,
                                   desc=f"Epoch {epoch}/{config.epochs} [train]")
        # Run one validation epoch (no optimizer = no weight updates).
        val_metrics = _run_epoch(model, val_loader, None, device,
                                 desc=f"Epoch {epoch}/{config.epochs} [val]  ")
        # Advance the learning rate scheduler by one step.
        scheduler.step()

        # Append this epoch's metrics to the history lists.
        history["train_loss"].append(train_metrics["loss"])
        history["train_iou"].append(train_metrics["iou"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_f1"].append(val_metrics["f1"])

        # Print a one-line summary for this epoch to the terminal.
        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_metrics['loss']:.4f}  IoU {train_metrics['iou']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}  IoU {val_metrics['iou']:.3f}"
        )

        # If validation IoU improved, save a checkpoint of the model weights.
        if config.checkpoint_path and val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            # "torch.save" writes a dictionary containing the epoch number, weights,
            # and best IoU value to disk as a binary file.
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_iou": best_val_iou}, config.checkpoint_path)

    # Return the full training history so the caller can plot it.
    return history


def _run_siamese_epoch(model: SiameseUNet, loader, optimizer, device,
                       desc: str) -> Dict[str, float]:
    """Like _run_epoch but unpacks (before, after, change_mask) batches."""
    is_train  = optimizer is not None
    model.train(is_train)
    running   = {"loss": 0.0, "iou": 0.0, "f1": 0.0, "pixel_acc": 0.0}
    n_batches = 0
    is_cuda   = str(device).startswith("cuda")

    progress = tqdm(loader, desc=desc, leave=False)
    for before, after, change in progress:
        before = before.to(device, non_blocking=is_cuda)
        after  = after.to(device,  non_blocking=is_cuda)
        change = change.to(device, non_blocking=is_cuda)

        with torch.set_grad_enabled(is_train):
            # Forward pass: the model receives both images and outputs change logits.
            logits = model(before, after)
            loss   = combined_loss(logits, change)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        metrics = segmentation_metrics(logits.detach(), change)
        running["loss"] += loss.item()
        for k in ("iou", "f1", "pixel_acc"):
            running[k] += metrics[k]
        n_batches += 1
        progress.set_postfix(loss=f"{loss.item():.4f}", iou=f"{metrics['iou']:.3f}")

    return {k: v / max(n_batches, 1) for k, v in running.items()}


def train_siamese(model: SiameseUNet, train_loader, val_loader,
                  config: TrainConfig) -> Dict[str, List[float]]:
    """Train a SiameseUNet on (before, after, change_mask) triplet batches.

    The checkpoint saved to ``config.checkpoint_path`` includes a
    ``"model_type": "siamese"`` key so the app can auto-detect model type
    on load.
    """
    device = config.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs))

    history = {"train_loss": [], "train_iou": [], "train_f1": [],
               "val_loss": [],   "val_iou": [],   "val_f1": []}

    best_val_iou = -1.0
    if config.checkpoint_path:
        Path(config.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        train_m = _run_siamese_epoch(model, train_loader, optimizer, device,
                                     desc=f"Epoch {epoch}/{config.epochs} [train]")
        val_m   = _run_siamese_epoch(model, val_loader,   None, device,
                                     desc=f"Epoch {epoch}/{config.epochs} [val]  ")
        scheduler.step()

        history["train_loss"].append(train_m["loss"])
        history["train_iou"].append(train_m["iou"])
        history["train_f1"].append(train_m["f1"])
        history["val_loss"].append(val_m["loss"])
        history["val_iou"].append(val_m["iou"])
        history["val_f1"].append(val_m["f1"])

        print(
            f"Epoch {epoch:3d} | "
            f"train loss {train_m['loss']:.4f}  IoU {train_m['iou']:.3f} | "
            f"val loss {val_m['loss']:.4f}  IoU {val_m['iou']:.3f}"
        )

        if config.checkpoint_path and val_m["iou"] > best_val_iou:
            best_val_iou = val_m["iou"]
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "model_type":  "siamese",       # ← used by app to auto-detect on load
                "widths":      list(model.widths),
                "val_iou":     best_val_iou,
            }, config.checkpoint_path)

    return history


# This function loads previously saved model weights from a checkpoint file.
def load_checkpoint(model: UNet, path, device=None) -> UNet:
    """Load weights from a checkpoint produced by ``train``.

    DirectML doesn't always accept its device as ``map_location``, so we
    always deserialize to CPU first and then move the model to ``device``.
    """
    # Use the best available device if none is specified.
    if device is None:
        device = get_default_device()
    # Load the saved file; "map_location='cpu'" ensures the data lands on CPU first,
    # avoiding compatibility issues with DirectML and other GPU backends.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # If the file is a dictionary with a "model_state" key, extract just the weights.
    # Otherwise assume the file itself is the state dictionary.
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    # "load_state_dict" copies the saved weights into the model.
    model.load_state_dict(state)
    # Move the model to the target device.
    model.to(device)
    # Put the model in evaluation mode (disables dropout, uses stored batch-norm statistics).
    model.eval()
    return model


# =============================================================================
# SECTION 4 - INFERENCE: sliding-window prediction on large aerial images
# =============================================================================

# This helper creates a smooth 1D taper that is high in the middle and low at the edges.
# It is used to blend overlapping tile predictions smoothly.
def _cosine_window(size: int) -> np.ndarray:
    """1D cosine taper, used to blend tile predictions in the overlap zone."""
    # "np.linspace(-pi/2, pi/2, size)" creates "size" evenly spaced values from -90° to +90°.
    x = np.linspace(-np.pi / 2, np.pi / 2, size)
    # cos²(x) is 0 at the edges and 1 in the centre — a smooth bell-shaped weight.
    return np.cos(x) ** 2


# This helper turns the 1D cosine window into a 2D blending weight map.
def _make_2d_window(tile: int) -> np.ndarray:
    # Create a 1D window of the given tile size.
    w = _cosine_window(tile)
    # "np.outer(w, w)" computes the outer product, giving a 2D array where
    # every point is the product of its row weight and column weight.
    return np.outer(w, w).astype(np.float32)


# "@torch.no_grad()" disables gradient tracking since we are only predicting, not training.
@torch.no_grad()
def predict_tile(model: UNet, tile: np.ndarray, device=None) -> np.ndarray:
    """Run the model on a single HWC uint8 RGB tile, returning a float prob map."""
    # Validate that the input is in the expected format.
    if tile.dtype != np.uint8:
        raise TypeError("Expected uint8 RGB tile")
    if device is None:
        device = get_default_device()
    # Convert the NumPy HWC array to a CHW tensor, add a batch dimension of 1
    # with ".unsqueeze(0)", and scale pixel values to 0.0–1.0.
    t = torch.from_numpy(tile.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    # Move the tensor to the target device.
    t = t.to(device)
    # Run the model forward pass to get raw logits.
    logits = model(t)
    # Apply sigmoid to convert logits to probabilities; remove the batch and channel
    # dimensions ([0, 0]) and move the result back to CPU as a NumPy float32 array.
    return torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)


# "@torch.no_grad()" again — no gradients needed for inference.
@torch.no_grad()
def predict_full_image(model: UNet, image: np.ndarray, tile_size: int = 256,
                       overlap: float = 0.25, device=None,
                       threshold: Optional[float] = 0.5):
    """Run sliding-window inference over an arbitrarily-sized aerial image.

    Returns
    -------
    prob_map : float32 H x W array in [0, 1]
    binary_mask : uint8 H x W array; empty if ``threshold`` is None.

    DirectML note
    -------------
    torch-directml has a long-standing bug where ``nn.BatchNorm2d`` in
    ``eval()`` mode does not use its stored running statistics correctly,
    causing the network to emit saturated (all-one-class) output. Training
    isn't affected because it uses per-batch statistics instead. We work
    around the inference issue by temporarily moving the model to CPU,
    running prediction there, and restoring its original device afterwards.
    Inference on CPU for a ~2000x1000 image is ~5-15 s - slower than the
    GPU would be in theory, but it actually gives correct masks.
    """
    if device is None:
        device = get_default_device()

    # --- DirectML workaround: do inference on CPU ---
    # We do NOT bounce the live model between DirectML and CPU - doing so
    # repeatedly has been observed to corrupt BatchNorm running statistics,
    # causing the network to silently degrade to "all-one-class" output. Make
    # a one-time CPU clone of the model state and use that for inference;
    # the original model is left untouched on DirectML and stays good for
    # subsequent training calls.
    # Check whether the selected device is a DirectML GPU.
    is_dml = str(device).startswith("privateuseone") or str(device) == "dml"
    if is_dml:
        # If using DirectML, create a CPU copy of the model for inference (see note above).
        infer_model = _cpu_clone(model)
        inference_device = torch.device("cpu")
    else:
        # Otherwise, use the model directly on its current device.
        infer_model = model
        inference_device = device

    # Put the inference model in evaluation mode.
    infer_model.eval()
    # Get the full image dimensions (height, width, channels).
    h, w, _ = image.shape
    # Compute the stride (step size) between tiles based on the desired overlap fraction.
    stride = max(1, int(tile_size * (1 - overlap)))

    # Compute how many extra rows/columns to pad so the image is evenly divisible by the stride.
    pad_h = (tile_size - (h - tile_size) % stride) % stride if h > tile_size else tile_size - h
    pad_w = (tile_size - (w - tile_size) % stride) % stride if w > tile_size else tile_size - w
    # "np.pad" extends the image with reflected (mirror) pixels to avoid black borders.
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    # Record the padded image dimensions.
    H, W, _ = padded.shape

    # Pre-compute the 2D cosine blending weight for a single tile.
    weight = _make_2d_window(tile_size)
    # Two arrays to accumulate weighted predictions and the total weight at each pixel.
    accumulated = np.zeros((H, W), dtype=np.float32)
    weights = np.zeros((H, W), dtype=np.float32)

    # Slide a tile-shaped window over the padded image, row by row, column by column.
    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            # Extract the current tile.
            tile = padded[y:y + tile_size, x:x + tile_size]
            # Run the model on this tile to get a probability map.
            probs = predict_tile(infer_model, tile, device=inference_device)
            # Add the weighted probabilities to the accumulator array.
            accumulated[y:y + tile_size, x:x + tile_size] += probs * weight
            # Add the tile's weight map so we can normalise later.
            weights[y:y + tile_size, x:x + tile_size] += weight

    # Normalise by dividing accumulated predictions by the total weight at each pixel.
    # "np.maximum(weights, 1e-6)" prevents division by zero in any unvisited pixel.
    # "[:h, :w]" removes the padding to restore the original image size.
    prob_map = (accumulated / np.maximum(weights, 1e-6))[:h, :w]
    # If no threshold is requested, return just the probability map and an empty mask.
    if threshold is None:
        return prob_map, np.empty((0, 0), dtype=np.uint8)
    # Otherwise, threshold the probability map to produce a binary building mask.
    return prob_map, (prob_map > threshold).astype(np.uint8)


@torch.no_grad()
def predict_change(model: SiameseUNet, before: np.ndarray, after: np.ndarray,
                   tile_size: int = 256, overlap: float = 0.25,
                   device=None, threshold: Optional[float] = 0.5):
    """Sliding-window change detection on a (before, after) image pair.

    Extracts matching tiles from both images simultaneously, concatenates
    them into a 6-channel input, and runs through a SiameseUNet.  Tile
    predictions are blended with a cosine window to avoid seam artefacts.

    Parameters
    ----------
    model : SiameseUNet
    before, after : np.ndarray
        HxWx3 uint8 RGB arrays of equal shape.
    tile_size : int
        Sliding-window tile size (should match the value used during training).
    overlap : float
        Fraction of each tile that overlaps with its neighbours (0.25 = 25%).
    device : str or torch.device, optional
    threshold : float or None
        Binarisation threshold.  Pass None to return only the probability map.

    Returns
    -------
    prob_map : float32 HxW array in [0, 1]
        Per-pixel change probability output by the network.
    binary_mask : uint8 HxW array
        Thresholded map (1 = changed, 0 = unchanged).
        Empty array if threshold is None.
    """
    if device is None:
        device = get_default_device()

    if before.shape != after.shape:
        raise ValueError(
            f"before/after shape mismatch: {before.shape} vs {after.shape}")

    # --- DirectML workaround: inference on CPU clone (see predict_full_image) ---
    is_dml = str(device).startswith("privateuseone") or str(device) == "dml"
    if is_dml:
        infer_model    = _cpu_clone(model)
        infer_device   = torch.device("cpu")
    else:
        infer_model    = model
        infer_device   = device

    infer_model.eval()

    h, w, _ = before.shape
    stride  = max(1, int(tile_size * (1 - overlap)))

    # Pad both images to a size evenly divisible by the stride.
    pad_h = (tile_size - (h - tile_size) % stride) % stride if h > tile_size else tile_size - h
    pad_w = (tile_size - (w - tile_size) % stride) % stride if w > tile_size else tile_size - w

    pb = np.pad(before, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    pa = np.pad(after,  ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    H, W, _ = pb.shape

    # Cosine blending window — high in the centre, tapered to zero at the edges.
    weight      = _make_2d_window(tile_size)
    accumulated = np.zeros((H, W), dtype=np.float32)
    weights     = np.zeros((H, W), dtype=np.float32)

    for y in range(0, H - tile_size + 1, stride):
        for x in range(0, W - tile_size + 1, stride):
            # Extract matching tiles from both images.
            tb = pb[y:y + tile_size, x:x + tile_size]
            ta = pa[y:y + tile_size, x:x + tile_size]

            # Convert HWC uint8 → [1, 3, H, W] float in [0, 1].
            tb_t = torch.from_numpy(tb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            ta_t = torch.from_numpy(ta.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            tb_t = tb_t.to(infer_device)
            ta_t = ta_t.to(infer_device)

            # Run forward pass; sigmoid converts logits → probabilities.
            logits = infer_model(tb_t, ta_t)
            probs  = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)

            # Accumulate weighted predictions.
            accumulated[y:y + tile_size, x:x + tile_size] += probs * weight
            weights    [y:y + tile_size, x:x + tile_size] += weight

    # Normalise and crop back to original image size.
    prob_map = (accumulated / np.maximum(weights, 1e-6))[:h, :w]

    if threshold is None:
        return prob_map, np.empty((0, 0), dtype=np.uint8)
    return prob_map, (prob_map > threshold).astype(np.uint8)


# This helper creates a CPU copy of a model without touching the original's weights.
def _cpu_clone(model: nn.Module) -> nn.Module:
    """Return a CPU-resident deep copy of ``model`` without touching the original.

    We copy via state_dict + a fresh module instance instead of copy.deepcopy
    because deepcopy on a DirectML-resident model has occasionally been seen
    to preserve the device assignment - which is exactly what we're trying
    to avoid.
    """
    # "import copy" loads Python's built-in deep-copy utility inside this function.
    import copy
    # Reconstruct on CPU. type(model)() doesn't know the original's
    # constructor args, so we deepcopy the structure then explicitly cpu() it.
    # "copy.deepcopy(model)" makes an independent copy of the entire model structure and weights.
    # ".to('cpu')" moves that copy to CPU memory.
    clone = copy.deepcopy(model).to("cpu")
    return clone


# =============================================================================
# SECTION 5 - CHANGE DETECTION: compare two masks, classify per-pixel events
# =============================================================================

# Integer codes used in the change map array to represent each pixel's category.
# BG = background (no building in either image).
BG = 0
# PRESERVED = building existed before AND still exists after.
PRESERVED = 1
# ADDED = new building appears in the "after" image.
ADDED = 2
# REMOVED = building that existed "before" has been demolished.
REMOVED = 3


# This function measures how "textured" a building region looks from above.
# Real roofs have edges (pipes, vents, seams); smooth surfaces like car parks do not.
def _edge_density(gray: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of pixels in ``mask`` (uint8 0/1) that are Sobel edge pixels.

    Real building roofs are visually busy (seams, vents, skylights, AC units,
    shadows of chimneys) and produce many strong edges. Car-park asphalt and
    bare construction-site soil are visually smooth and produce very few.
    """
    if gray.shape != mask.shape:
        # Crop or skip gracefully - just return a value that passes the check.
        return 1.0
    # If the mask is empty there are no pixels to check, so return 0.
    if mask.sum() == 0:
        return 0.0
    # Apply the Sobel filter horizontally (axis=1) to detect vertical edges.
    gx = ndimage.sobel(gray.astype(np.float32), axis=1)
    # Apply the Sobel filter vertically (axis=0) to detect horizontal edges.
    gy = ndimage.sobel(gray.astype(np.float32), axis=0)
    # "np.hypot" computes the overall edge magnitude as the hypotenuse: sqrt(gx² + gy²).
    mag = np.hypot(gx, gy)
    # Threshold: anything stronger than ~6% of full-scale Sobel magnitude.
    edge = mag > (255 * 0.06 * 4)   # 4 = max Sobel weight sum
    # Count how many edge pixels fall inside the masked region.
    inside = int((edge & mask.astype(bool)).sum())
    # Return the fraction of masked pixels that are edge pixels.
    return inside / max(1, int(mask.sum()))


# This function removes small or unrealistic regions from a predicted building mask.
def clean_building_mask(mask: np.ndarray,
                        image: Optional[np.ndarray] = None,
                        min_area: int = 200,
                        min_width: int = 8,
                        max_aspect_ratio: float = 8.0,
                        min_fill: float = 0.25,
                        min_edge_density: float = 0.04) -> np.ndarray:
    """Remove non-building clutter from a binary mask.

    Per-component filters, in order. A component must pass ALL of them:

      1. AREA      - drop if smaller than ``min_area`` pixels. Catches "small
                     dots": cars, debris, isolated mis-classified pixels.
      2. MIN WIDTH - drop if the SHORTER dimension of the bounding box is
                     below ``min_width``. Catches FOOTPATHS, which can be
                     long and well-filled but are fundamentally narrow.
      3. SNAKE     - drop if aspect ratio > ``max_aspect_ratio`` AND fill
                     < ``min_fill``. Catches fragmented road predictions.
      4. EDGE DENSITY (only if ``image`` is provided) - drop if the fraction
                     of Sobel-edge pixels inside the component is below
                     ``min_edge_density``. Catches CAR PARKS and CONSTRUCTION
                     SITES, which are large+blocky like buildings but
                     visually smooth from above.

    Returns a uint8 mask with values 0/1 (NOT 0/255 - matches detect_changes'
    expectations).
    """
    # Normalise the mask to strictly 0 and 1.
    m = (mask > 0).astype(np.uint8)
    # If the mask is already completely empty, return it immediately.
    if m.sum() == 0:
        return m

    # Start with no greyscale image; it will be set below if one is provided.
    gray = None
    if image is not None:
        # If the image has 3 colour channels, average them to get a greyscale version.
        if image.ndim == 3:
            gray = image.mean(axis=2)
        else:
            # If it is already greyscale, just convert it to float for later calculations.
            gray = image.astype(np.float32)

    # Label connected regions in the binary mask; each group of touching pixels gets a unique ID.
    labelled, n = ndimage.label(m)
    if n == 0:
        return m

    # "ndimage.find_objects" returns the bounding-box slice for each labelled region.
    objects = ndimage.find_objects(labelled)
    # Output mask (starts as all zeros; we will add back the regions that pass all filters).
    out = np.zeros_like(m)
    # Loop over each labelled region; "enumerate(..., start=1)" gives us (1, ...), (2, ...), etc.
    for i, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        # Extract the boolean sub-region for this component from the labelled array.
        region = (labelled[slc] == i).astype(np.uint8)
        area = int(region.sum())
        # Filter 1: discard regions that are too small (likely noise or tiny artefacts).
        if area < min_area:
            continue
        # Compute bounding-box height and width from the slice objects.
        h = slc[0].stop - slc[0].start
        w = slc[1].stop - slc[1].start
        # Filter 2: discard regions that are too narrow (e.g., footpaths, lane lines).
        if min(h, w) < min_width:
            continue  # too thin → footpath / curb / lane line
        # Aspect ratio: how elongated is the bounding box?
        aspect = max(h, w) / max(1, min(h, w))
        # Fill: what fraction of the bounding box does the region actually cover?
        fill = area / max(1, h * w)
        # Filter 3: discard snake-shaped (very elongated, sparsely filled) regions — likely roads.
        if aspect > max_aspect_ratio and fill < min_fill:
            continue  # snake-shaped → fragmented road / sidewalk
        # Filter 4: if we have a greyscale image, check visual texture.
        if gray is not None:
            ed = _edge_density(gray[slc], region)
            if ed < min_edge_density:
                continue  # smooth interior → car park / construction site
        # If the region passed all filters, copy it into the output mask.
        out[slc][region.astype(bool)] = 1
    return out


# A dataclass representing a single detected change event (one building added or removed).
@dataclass
class ChangeEvent:
    """A single connected change region (one added or one removed building)."""
    # Unique integer ID assigned to this event.
    label: int
    # Whether this change is a new building ("added") or a demolished one ("removed").
    category: str                       # "added" or "removed"
    # How many pixels this change event covers.
    area_px: int
    # Bounding box of this event: (top row, left col, bottom row, right col).
    bbox: Tuple[int, int, int, int]     # (ymin, xmin, ymax, xmax)
    # The average (row, col) position of the event — useful for placing labels on a map.
    centroid: Tuple[float, float]       # (y, x)


# A dataclass that bundles the complete result of a change-detection comparison.
@dataclass
class ChangeReport:
    """Aggregated result of comparing two masks."""
    # A 2D array where each pixel is coded as BG / PRESERVED / ADDED / REMOVED.
    change_map: np.ndarray              # uint8 HxW with values {0,1,2,3}
    # A list of all individual change events found.
    events: List[ChangeEvent]
    # Summary statistics (counts, areas, percentages).
    stats: Dict[str, float]

    # "@property" turns this method into an attribute — you access it as report.added_events
    # without parentheses, even though it is defined as a method.
    @property
    def added_events(self):
        # Return only the events that represent newly added buildings.
        return [e for e in self.events if e.category == "added"]

    @property
    def removed_events(self):
        # Return only the events that represent demolished buildings.
        return [e for e in self.events if e.category == "removed"]


# This helper finds all connected regions in a binary difference mask and
# converts them into a list of ChangeEvent objects.
def _component_events(diff_mask: np.ndarray, category: str, min_area: int,
                      max_aspect_ratio: float = 4.0,
                      min_fill: float = 0.25) -> List[ChangeEvent]:
    """Enumerate change-event components.

    A component is dropped if its area is below ``min_area`` OR if it looks
    like a road / linear feature (very elongated AND sparsely filled). Even
    cleaned input masks can leak narrow strips at building edges; filtering
    here is the second line of defense.
    """
    # Label all connected regions in the difference mask.
    labelled, n = ndimage.label(diff_mask)
    # If there are no regions at all, return an empty list immediately.
    if n == 0:
        return []
    # Find the bounding boxes for all labelled regions.
    objects = ndimage.find_objects(labelled)
    # Start with an empty list of events.
    events: List[ChangeEvent] = []
    # Counter for labelling accepted events (events that pass all filters get sequential IDs).
    keep_id = 0
    for i, slc in enumerate(objects, start=1):
        if slc is None:
            continue
        # Extract the boolean sub-region for this component.
        region = labelled[slc] == i
        area = int(region.sum())
        # Skip regions that are too small to be meaningful.
        if area < min_area:
            continue
        # Extract bounding-box corners from the slice objects.
        ymin, xmin = slc[0].start, slc[1].start
        ymax, xmax = slc[0].stop, slc[1].stop
        h = ymax - ymin
        w = xmax - xmin
        # Compute elongation and fill ratio for the road/footpath filter.
        aspect = max(h, w) / max(1, min(h, w))
        fill = area / max(1, h * w)
        # Skip snake-shaped (very elongated and sparse) regions — likely road artefacts.
        if aspect > max_aspect_ratio and fill < min_fill:
            continue  # snake-shaped → likely road / sidewalk
        keep_id += 1
        # "np.where(region)" returns the (row, col) coordinates of all True pixels in the region.
        ys, xs = np.where(region)
        # Compute the centroid (average position) of the event in full-image coordinates.
        centroid = (float(ys.mean()) + ymin, float(xs.mean()) + xmin)
        # Add a new ChangeEvent to the list.
        events.append(ChangeEvent(keep_id, category, area, (ymin, xmin, ymax, xmax), centroid))
    return events


# This is the main function for comparing two binary building masks and finding what changed.
def detect_changes(mask_before: np.ndarray, mask_after: np.ndarray,
                   min_area: int = 200, morph_radius: int = 2) -> ChangeReport:
    """Compare two binary masks and return a :class:`ChangeReport`."""
    # Both masks must cover the same area (same pixel dimensions).
    if mask_before.shape != mask_after.shape:
        raise ValueError("Masks must have the same shape")

    # Convert both masks to strict binary (0 or 1) uint8 arrays.
    a = (mask_before > 0).astype(np.uint8)
    b = (mask_after > 0).astype(np.uint8)

    # Added pixels: present in "after" (b=1) but not in "before" (a=0).
    added_raw = ((b == 1) & (a == 0)).astype(np.uint8)
    # Removed pixels: present in "before" (a=1) but not in "after" (b=0).
    removed_raw = ((a == 1) & (b == 0)).astype(np.uint8)
    # Preserved pixels: present in both images.
    preserved = ((a == 1) & (b == 1)).astype(np.uint8)

    # Apply morphological opening to remove tiny speckles and isolated noise pixels.
    if morph_radius > 0:
        # "generate_binary_structure(2, 1)" creates a simple cross-shaped structuring element.
        struct = ndimage.generate_binary_structure(2, 1)
        # Binary opening = erosion followed by dilation — removes small isolated spots.
        added_raw = ndimage.binary_opening(added_raw, structure=struct,
                                           iterations=morph_radius).astype(np.uint8)
        removed_raw = ndimage.binary_opening(removed_raw, structure=struct,
                                             iterations=morph_radius).astype(np.uint8)

    # Find meaningful connected regions in the added and removed pixel sets.
    added_events = _component_events(added_raw, "added", min_area)
    removed_events = _component_events(removed_raw, "removed", min_area)

    # Reconstruct clean binary masks that contain only the accepted events.
    added_clean = np.zeros_like(a)
    removed_clean = np.zeros_like(a)
    for ev in added_events:
        ymin, xmin, ymax, xmax = ev.bbox
        # Paste the raw added pixels from this event's bounding box into the clean mask.
        added_clean[ymin:ymax, xmin:xmax] |= added_raw[ymin:ymax, xmin:xmax]
    for ev in removed_events:
        ymin, xmin, ymax, xmax = ev.bbox
        # Paste the raw removed pixels from this event's bounding box into the clean mask.
        removed_clean[ymin:ymax, xmin:xmax] |= removed_raw[ymin:ymax, xmin:xmax]

    # Build the final integer change map by assigning each pixel its category code.
    change_map = np.zeros_like(a, dtype=np.uint8)
    change_map[preserved == 1] = PRESERVED
    change_map[added_clean == 1] = ADDED
    change_map[removed_clean == 1] = REMOVED

    # Compute summary statistics for the report.
    total_pixels = a.size
    stats = {
        "n_added": len(added_events),           # Number of distinct new-building events.
        "n_removed": len(removed_events),        # Number of distinct demolished-building events.
        "added_area_px": int(added_clean.sum()), # Total pixel area of newly added buildings.
        "removed_area_px": int(removed_clean.sum()), # Total pixel area of removed buildings.
        "preserved_area_px": int(preserved.sum()),   # Total pixel area of unchanged buildings.
        "added_pct": 100.0 * added_clean.sum() / total_pixels,    # Added area as % of image.
        "removed_pct": 100.0 * removed_clean.sum() / total_pixels, # Removed area as % of image.
        # Positive value means more building was added than removed (net growth).
        "net_change_px": int(added_clean.sum()) - int(removed_clean.sum()),
    }

    # Bundle everything into a ChangeReport and return it.
    return ChangeReport(change_map=change_map,
                        events=added_events + removed_events,
                        stats=stats)


# =============================================================================
# SECTION 5b - ERA DETECTION: classify aerial image by capture decade
# =============================================================================
# WHY IMAGE STATISTICS ALONE DON'T WORK FOR HELSINKI WMS
# -------------------------------------------------------
# We tried using Laplacian variance (sharpness) to tell eras apart, but
# Helsinki's WMS server breaks that assumption in two ways:
#
#   1. Downsampling paradox
#      The WMS always outputs a 1024×1024 px tile covering the same 512 m
#      area, regardless of the source resolution.  A 25 cm/px source (2010)
#      is downsampled 50× to reach the output.  A 5 cm/px source (2025) is
#      downsampled 10×.  Heavy downsampling applies anti-aliasing blur, so
#      the 2010 image can appear *sharper* than the 2025 image at output
#      resolution.  Measured scores confirm this:
#        2010 Kallio: score 203 (sharpest)
#        2014 Kallio: score 196
#        2025 Kallio: score 181
#        2001 Kallio: score 145 (softest)
#      The order is wrong — older images score higher than newer ones.
#
#   2. Colour normalisation
#      The WMS normalises colours server-side.  Saturation is ~37–41 for
#      every decade; it adds a near-constant to every score and cannot
#      separate eras.
#
# THE SOLUTION: read the year from the filename first
# ---------------------------------------------------
# Downloaded pairs are always named "<place>_<year>.png" (e.g. kallio_2001).
# Parsing a 4-digit year from the filename gives a perfectly reliable era
# with no image-statistics guesswork.
#
# Fallback for unnamed files:
#   If no year is found in the filename, we still run the image statistics
#   pipeline and pick the closest era — it won't be perfect, but it gives
#   the user a reasonable starting point and shows the raw numbers.
#
# ERA YEAR BOUNDARIES (calendar-based, not statistics-based):
#   modern   : year >= 2015   (high-res digital 5–20 cm)
#   mid      : 2005 <= year < 2015   (early digital 25–50 cm)
#   historic : year < 2005   (analogue film scanned to digital)

import re as _re   # "re" is Python's regular-expression library; used to search text.

def detect_image_era(image: np.ndarray, filename: str = "") -> tuple:
    """
    Classify an aerial image as "modern", "mid", or "historic".

    Strategy (in priority order)
    ----------------------------
    1. Year from filename
       Look for a 4-digit year (1900–2099) in the filename string.
       If found, map it directly to an era using calendar boundaries.
       This is always accurate for the project's downloaded pairs.

    2. Image statistics fallback
       If no year is found, measure post-blur Laplacian variance
       (sharpness) and HSV saturation as a rough guide.  This is
       imperfect for Helsinki WMS imagery (see section header above)
       but still gives a useful hint.

    Parameters
    ----------
    image : np.ndarray
        RGB image array of shape (H, W, 3), dtype uint8.
    filename : str, optional
        File name or full path string that may contain a 4-digit year,
        e.g. "kallio_2001.png" or "pairs/pasila_2025.png".
        Leave empty to skip the filename check and use statistics only.

    Returns
    -------
    era : str
        One of "modern" (2015+), "mid" (2005–2014), or "historic" (pre-2005).
    stats : dict
        Image statistics and the source of the classification decision.
        Keys: "sharpness", "brightness", "saturation", "era_score",
              "year" (int or None), "source" ("filename" or "image_stats").
    """
    import cv2 as _cv2  # OpenCV for image processing operations

    # =========================================================================
    # PATH 1: extract year from the filename
    # =========================================================================
    # r'\b(19|20)\d{2}\b' matches a standalone 4-digit number starting with
    # 19xx or 20xx — a plausible aerial-photo year (1900–2099).
    # \b is a "word boundary" anchor so "1024" inside a pixel dimension string
    # won't be mistaken for a year.
    year: int | None = None
    year_match = _re.search(r'\b((?:19|20)\d{2})\b', filename)
    if year_match:
        # int() converts the matched string (e.g. "2001") to an integer.
        year = int(year_match.group(1))

    # Always compute image statistics — they are shown in the GUI regardless
    # of which path is used for the actual classification decision.

    # Convert RGB → greyscale for the Laplacian.
    gray = _cv2.cvtColor(image, _cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Pre-blur (σ=1.5) to suppress film grain before measuring sharpness.
    blurred  = _cv2.GaussianBlur(gray, (0, 0), 1.5)
    laplacian = _cv2.Laplacian(blurred, _cv2.CV_32F)
    sharpness = float(laplacian.var())

    # Mean pixel brightness across all channels (0 = black, 255 = white).
    brightness = float(image.mean())

    # Mean HSV saturation (0 = greyscale, 255 = fully vivid colour).
    hsv        = _cv2.cvtColor(image, _cv2.COLOR_RGB2HSV)
    saturation = float(hsv[:, :, 1].mean())

    # Combined image-statistics score (kept for display even when filename wins).
    era_score  = sharpness * 3.0 + saturation * 1.0

    stats = {
        "sharpness":  round(sharpness,  1),
        "brightness": round(brightness, 1),
        "saturation": round(saturation, 1),
        "era_score":  round(era_score,  1),
        "year":       year,          # None if no year was found in the filename
        "source":     "",            # filled in below
    }

    # =========================================================================
    # CLASSIFY — filename year takes priority over image statistics
    # =========================================================================
    if year is not None:
        # We know the exact year — use calendar boundaries directly.
        # This is always correct for pairs downloaded through the app.
        stats["source"] = "filename"
        if year >= 2015:
            return "modern", stats    # 2015 onward: high-res digital
        elif year >= 2005:
            return "mid", stats       # 2005–2014: early digital
        else:
            return "historic", stats  # pre-2005: scanned analogue film

    # No year found — fall back to image statistics.
    # Note: for Helsinki WMS tiles the ordering is unreliable (see header),
    # so treat this result as a rough hint rather than a definitive answer.
    stats["source"] = "image_stats"
    if era_score >= 163:
        era = "modern"
    elif era_score >= 60:
        era = "mid"
    else:
        era = "historic"

    return era, stats


# =============================================================================
# SECTION 6 - VISUALIZATION: 6-panel summary plot
# =============================================================================

# RGB colour constants used to colour-code the change map in figures.
# Gold colour for preserved buildings.
COLOUR_PRESERVED = np.array([255, 215,   0], dtype=np.uint8)   # gold
# Bright green for newly added buildings.
COLOUR_ADDED     = np.array([ 60, 200,  60], dtype=np.uint8)   # green
# Red for demolished buildings.
COLOUR_REMOVED   = np.array([220,  40,  40], dtype=np.uint8)   # red


# This function converts the integer change map into a colour RGB image for display.
def colourise_change_map(change_map: np.ndarray) -> np.ndarray:
    """Render a {0,1,2,3} integer map as an HWC uint8 RGB image."""
    h, w = change_map.shape
    # Start with an all-black (background) RGB image.
    out = np.zeros((h, w, 3), dtype=np.uint8)
    # Paint each category with its designated colour.
    out[change_map == PRESERVED] = COLOUR_PRESERVED
    out[change_map == ADDED] = COLOUR_ADDED
    out[change_map == REMOVED] = COLOUR_REMOVED
    return out


# This function blends the change map over the aerial image so changes are easy to spot.
def overlay_on_aerial(aerial: np.ndarray, change_map: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Blend the coloured change map on top of an aerial image."""
    # Ensure the aerial photo and change map cover the same area.
    if aerial.shape[:2] != change_map.shape:
        raise ValueError("Aerial and change map must have matching H, W")
    # Convert the change map to a colour RGB image.
    overlay = colourise_change_map(change_map)
    # Create a boolean mask: True wherever a change occurred (not background).
    mask = (change_map != BG)
    # Work on a float copy of the aerial image so we can do blending arithmetic.
    blended = aerial.astype(np.float32).copy()
    # For changed pixels: blend the aerial colour with the change colour using alpha.
    # alpha=0.55 means the change colour contributes 55% and the aerial contributes 45%.
    blended[mask] = (1 - alpha) * blended[mask] + alpha * overlay[mask]
    # Clamp all values back to 0–255 and convert to uint8.
    return np.clip(blended, 0, 255).astype(np.uint8)


# This function creates the full 6-panel summary figure showing before/after/changes.
def plot_comparison(before_img, after_img, before_mask, after_mask, report: ChangeReport,
                    title: str = "Helsinki Urban Change Detection",
                    save_path: Optional[str] = None):
    """6-panel summary figure of the full change-detection result."""
    # Create the blended overlay image for the top-right panel.
    overlay = overlay_on_aerial(after_img, report.change_map)
    # Create a figure with 2 rows and 3 columns of sub-plots; figsize is in inches.
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Top row: the three aerial/overlay images.
    axes[0, 0].imshow(before_img); axes[0, 0].set_title("Aerial - Before")
    axes[0, 1].imshow(after_img);  axes[0, 1].set_title("Aerial - After")
    axes[0, 2].imshow(overlay);    axes[0, 2].set_title("Detected Changes")

    # Bottom row: the three binary masks / change map.
    axes[1, 0].imshow(before_mask, cmap="gray"); axes[1, 0].set_title("Predicted Buildings - Before")
    axes[1, 1].imshow(after_mask, cmap="gray");  axes[1, 1].set_title("Predicted Buildings - After")
    axes[1, 2].imshow(colourise_change_map(report.change_map)); axes[1, 2].set_title("Change Map")

    # "axes.ravel()" flattens the 2D array of axes into a 1D list.
    for ax in axes.ravel():
        # Remove x and y tick marks from all panels so the images look clean.
        ax.set_xticks([]); ax.set_yticks([])

    # Build legend patches (coloured rectangles) for the three change categories.
    legend_handles = [
        mpatches.Patch(color=COLOUR_ADDED / 255, label=f"Added  ({report.stats['n_added']})"),
        mpatches.Patch(color=COLOUR_REMOVED / 255, label=f"Removed ({report.stats['n_removed']})"),
        mpatches.Patch(color=COLOUR_PRESERVED / 255, label="Preserved"),
    ]
    # Add the legend below all panels at the bottom centre of the figure.
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.01))

    # Compose a one-line statistics summary string.
    stats_line = (
        f"Net building footprint change: {report.stats['net_change_px']:+d} px  |  "
        f"Added area: {report.stats['added_pct']:.2f}%  |  "
        f"Removed area: {report.stats['removed_pct']:.2f}%"
    )
    # Add the title and statistics as a "super title" spanning all panels.
    fig.suptitle(f"{title}\n{stats_line}", fontsize=13, y=0.995)
    # Adjust spacing between panels automatically to avoid overlap.
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    if save_path:
        # Save the figure to disk at 140 DPI (dots per inch), cropping whitespace.
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# This function plots the training and validation metrics over each epoch.
def plot_training_history(history: dict, save_path: Optional[str] = None):
    """Plot the loss / IoU / F1 curves produced by :func:`train`."""
    # Build a range of epoch numbers (1, 2, 3, ...) to use as the x-axis.
    epochs = range(1, len(history["train_loss"]) + 1)
    # Create a figure with 1 row and 3 columns of sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left panel: loss curves for training and validation.
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

    # Middle panel: IoU (Intersection over Union) curves.
    axes[1].plot(epochs, history["train_iou"], label="train")
    axes[1].plot(epochs, history["val_iou"], label="val")
    axes[1].set_title("IoU (buildings)"); axes[1].set_xlabel("Epoch"); axes[1].legend()

    # Right panel: F1/Dice score curves.
    axes[2].plot(epochs, history["train_f1"], label="train")
    axes[2].plot(epochs, history["val_f1"], label="val")
    axes[2].set_title("F1 / Dice (buildings)"); axes[2].set_xlabel("Epoch"); axes[2].legend()

    # Adjust spacing between panels.
    fig.tight_layout()
    if save_path:
        # Save the figure to disk.
        fig.savefig(save_path, dpi=140, bbox_inches="tight")
    return fig


# =============================================================================
# SECTION 7 - MAIN: end-to-end pipeline runner
# =============================================================================

# "def main() -> None" defines the top-level function that runs the whole pipeline.
# "-> None" means it returns nothing — it just runs a sequence of steps and saves files.
def main() -> None:
    """Generate data, train the U-Net, run change detection, write figures."""
    # "__file__" is a special Python variable that holds the path of the current script.
    # ".parent" gives the folder containing the script.
    here = Path(__file__).parent
    # Build the path where synthetic training data will be stored.
    data_root = here / "data" / "synthetic"
    # Build the path where the best model weights will be saved.
    ckpt_path = here / "models" / "unet_buildings.pt"
    # Build the path where output images (figures) will be written.
    out_dir = here / "sample_outputs"
    # Create the output folder if it does not already exist (including any parent folders).
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: synthetic training data ------------------------------------
    print("[1/5] Generating synthetic training data...")
    # Configure the synthetic data generator: 128-pixel tiles, 80 training + 20 validation.
    cfg = SyntheticConfig(tile_size=128, n_train=80, n_val=20, seed=42)
    # Generate and save the synthetic images and masks to disk.
    generate_synthetic_dataset(data_root, cfg)

    # Create a BuildingDataset for training (with augmentation enabled).
    train_ds = BuildingDataset(data_root / "train" / "images",
                               data_root / "train" / "masks",
                               tile_size=128, augment=True)
    # Create a BuildingDataset for validation (no augmentation — we want clean evaluation).
    val_ds = BuildingDataset(data_root / "val" / "images",
                             data_root / "val" / "masks",
                             tile_size=128, augment=False)

    # ---- Step 2: U-Net ------------------------------------------------------
    print("[2/5] Building U-Net...")
    # Create a smaller U-Net (fewer filters per layer) so training is fast for demo purposes.
    model = UNet(widths=(16, 32, 64, 128, 256))   # small for fast demo
    # Print how many learnable numbers (parameters) the model has.
    print(f"    parameters: {model.count_parameters():,}")

    # ---- Step 3: Train ------------------------------------------------------
    # Detect the best available hardware (GPU or CPU) for training.
    device = get_default_device()
    print(f"[3/5] Training (device={device_label(device)})...")
    # Create a training configuration with custom hyperparameters.
    config = TrainConfig(
        epochs=8,           # Train for 8 full passes over the dataset.
        lr=1e-3,            # Learning rate = 0.001.
        batch_size=8,       # Process 8 images per training step.
        num_workers=0,      # Load data on the main thread (safe for Windows).
        device=device,      # Use the detected GPU or CPU.
        checkpoint_path=str(ckpt_path),  # Where to save the best model.
    )
    # Wrap the training dataset in a DataLoader to handle batching and shuffling.
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    # Wrap the validation dataset; no shuffling needed for evaluation.
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    # Run the training loop and collect the history of metrics.
    history = train(model, train_loader, val_loader, config)
    # Save a figure of the training and validation curves to disk.
    plot_training_history(history, save_path=str(out_dir / "training_curves.png"))
    # Close all open Matplotlib figures to free memory.
    plt.close("all")

    # ---- Step 4: Inference on a multi-temporal pair -------------------------
    print("[4/5] Generating temporal aerial pair and running inference...")
    # Generate synthetic "before" and "after" aerial images (with 4 new and 3 removed buildings).
    # We only need the images here, not the ground-truth masks, so we discard them with "_".
    before_img, _, after_img, _ = generate_temporal_pair(
        cfg=SyntheticConfig(tile_size=256, min_buildings=8, max_buildings=14),
        seed=7, n_added=4, n_removed=3,
    )
    # Load the best checkpoint weights into the model.
    model = load_checkpoint(model, ckpt_path, device=device)
    # Run sliding-window inference on the "before" image to get a building mask.
    # The "_" discards the raw probability map; we only keep the binary mask.
    _, mask_before = predict_full_image(model, before_img, tile_size=128, device=device)
    # Run inference on the "after" image to get its building mask.
    _, mask_after = predict_full_image(model, after_img, tile_size=128, device=device)

    # ---- Step 5: Change detection + final visualisation ---------------------
    print("[5/5] Detecting changes and writing summary figure...")
    # Compare the two masks to find added and removed buildings.
    report = detect_changes(mask_before, mask_after, min_area=60)
    # Print the top-level statistics to the terminal.
    print(f"    added: {report.stats['n_added']}, removed: {report.stats['n_removed']}")
    print(f"    added area:   {report.stats['added_pct']:.2f}%")
    print(f"    removed area: {report.stats['removed_pct']:.2f}%")

    # Draw and save the 6-panel summary figure.
    plot_comparison(
        before_img, after_img,
        mask_before, mask_after,
        report,
        title="Helsinki Urban Change Detection",
        save_path=str(out_dir / "change_detection_summary.png"),
    )
    # Close all open figures.
    plt.close("all")
    print(f"\nDone. Figures saved in: {out_dir}")


# "if __name__ == '__main__':" is a standard Python guard.
# This block only runs when you execute the script directly (e.g., "python pipeline.py").
# It does NOT run if another script imports this file as a module.
if __name__ == "__main__":
    # Headless matplotlib backend if no display is available
    # Check whether a graphical display is attached to this process.
    if not os.environ.get("DISPLAY") and matplotlib.get_backend().lower() != "agg":
        try:
            # "Agg" is a non-interactive backend that renders images to files
            # without needing a screen — useful on servers or headless systems.
            matplotlib.use("Agg")
        except Exception:
            pass
    # Run the full pipeline.
    main()
