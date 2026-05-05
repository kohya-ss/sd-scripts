"""
Procedural mask generation for inpainting training.

Provides three mask types that can be combined:
  - cloud   : fractional Brownian motion turbulence (layered smooth noise)
  - polygon : random convex or irregular polygon
  - shape   : original rectangle / ellipse (kept for backwards compat)

All functions return a PIL Image in mode "L" (0 = keep, 255 = mask/inpaint).

| Function | Description |
|---|---|
| `cloud_mask(w, h, ...)` | Organic blob shapes via fractional Brownian motion (layered noise) |
| `polygon_mask(w, h, ...)` | Random convex polygons |
| `shape_mask(w, h, ...)` | Axis-aligned rectangles and ellipses |
| `combine_masks(*masks)` | Union or intersection of any of the above |
| `random_mask(w, h, ...)` | Randomly selects and optionally combines mask types |

To visually inspect mask output, run:

python3 tests/visualize_masks.py --data-dir /path/to/images --out-dir ./mask_viz

This produces gallery PNG files for each mask type, showing the raw mask alongside the mask applied to
source images. If no `--data-dir` is provided (or the directory is empty), synthetic test images
are used as the background.

"""

import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Cloud / turbulence masks
# ---------------------------------------------------------------------------

def _smooth_noise(w: int, h: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a single octave of smooth noise by upsampling small random values."""
    small_w = max(2, int(w / scale))
    small_h = max(2, int(h / scale))
    small = rng.random((small_h, small_w)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def cloud_mask(
    width: int,
    height: int,
    octaves: int = 4,
    persistence: float = 0.5,
    base_scale: float = 16.0,
    threshold: float = 0.2,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Fractional Brownian Motion cloud mask.

    Parameters
    ----------
    width, height : output size in pixels
    octaves       : number of noise layers; more = finer detail, more = grainier
    persistence   : amplitude falloff per octave (0..1); lower = smoother / fewer details
    base_scale    : spatial scale of the coarsest octave — higher = larger blobs
    threshold     : fraction of image area to mask (0..1); keep in ~0.1–0.3 for realistic patches
    seed          : optional RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    noise = np.zeros((height, width), dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    scale = base_scale

    for _ in range(octaves):
        noise += amplitude * _smooth_noise(width, height, scale, rng)
        total_amplitude += amplitude
        amplitude *= persistence
        scale *= 2.0

    noise /= total_amplitude          # normalise to [0, 1]
    cutoff = np.quantile(noise, 1.0 - threshold)
    binary = (noise >= cutoff).astype(np.uint8) * 255

    return Image.fromarray(binary, mode="L")


# ---------------------------------------------------------------------------
# Polygon masks
# ---------------------------------------------------------------------------

def _random_convex_polygon(
    cx: float, cy: float, radius: float, n_points: int, irregularity: float, rng: random.Random
) -> list:
    """Return vertices of a roughly convex polygon centred at (cx, cy)."""
    angles = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n_points))
    points = []
    for angle in angles:
        r = radius * rng.uniform(1.0 - irregularity, 1.0)
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def polygon_mask(
    width: int,
    height: int,
    n_points: int = 6,
    min_coverage: float = 0.1,
    max_coverage: float = 0.6,
    irregularity: float = 0.4,
    n_polygons: int = 1,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Random polygon mask.

    Parameters
    ----------
    width, height  : output size in pixels
    n_points       : number of vertices per polygon
    min_coverage   : minimum fraction of image area each polygon can cover
    max_coverage   : maximum fraction of image area each polygon can cover
    irregularity   : how irregular the polygon is (0 = circle, 1 = very jagged)
    n_polygons     : how many polygons to draw
    seed           : optional RNG seed
    """
    rng = random.Random(seed)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    min_r = math.sqrt(min_coverage * width * height / math.pi)
    max_r = math.sqrt(max_coverage * width * height / math.pi)

    for _ in range(n_polygons):
        radius = rng.uniform(min_r, max_r)
        cx = rng.uniform(radius * 0.2, width - radius * 0.2)
        cy = rng.uniform(radius * 0.2, height - radius * 0.2)
        pts = _random_convex_polygon(cx, cy, radius, n_points, irregularity, rng)
        draw.polygon(pts, fill=255)

    return mask


# ---------------------------------------------------------------------------
# Shape masks (rectangles / ellipses — original behaviour)
# ---------------------------------------------------------------------------

def shape_mask(
    width: int,
    height: int,
    min_coverage: float = 0.1,
    max_coverage: float = 0.4,
    shape: str = "random",
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Rectangle or ellipse mask (mirrors the original random_mask behaviour).

    Parameters
    ----------
    shape : "rectangle", "ellipse", or "random"
    """
    rng = random.Random(seed)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    area = width * height
    size_w = rng.randint(int(width * min_coverage ** 0.5), int(width * max_coverage ** 0.5))
    size_h = rng.randint(int(height * min_coverage ** 0.5), int(height * max_coverage ** 0.5))

    x1 = rng.randint(0, width - size_w)
    y1 = rng.randint(0, height - size_h)
    x2, y2 = x1 + size_w, y1 + size_h

    use_ellipse = (shape == "ellipse") or (shape == "random" and rng.random() < 0.5)
    if use_ellipse:
        draw.ellipse([x1, y1, x2, y2], fill=255)
    else:
        draw.rectangle([x1, y1, x2, y2], fill=255)

    return mask


# ---------------------------------------------------------------------------
# Wobbly ellipse mask (fBm radius variation — good default for sampling)
# ---------------------------------------------------------------------------

def wobbly_ellipse_mask(
    width: int,
    height: int,
    coverage: float = 0.2,
    wobble_scale: float = 0.5,
    octaves: int = 4,
    persistence: float = 0.5,
    n_points: int = 128,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Ellipse mask whose radius varies along the circumference using periodic
    fractional Brownian motion noise, producing organic blob-like shapes.

    Unlike cloud masks (which can produce large, fragmented regions), this
    always yields a single connected region, making it well-suited as a
    default mask for inpainting sampling and inference.

    Parameters
    ----------
    width, height  : output size in pixels
    coverage       : approximate fraction of the image area to mask (0..1)
    wobble_scale   : radius variation relative to the base radius.
                     Default 0.25 means radius varies ±25% around the base.
    octaves        : number of harmonic layers in the radius noise
    persistence    : amplitude falloff per octave (0..1)
    n_points       : number of polygon vertices used to approximate the shape
    seed           : optional RNG seed for reproducibility
    """
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(0, 2**31))

    # Center — biased toward image interior so the ellipse stays mostly inside
    cx = rng.uniform(0.25, 0.75) * width
    cy = rng.uniform(0.25, 0.75) * height

    # Ellipse aspect ratio and rotation
    aspect = rng.uniform(0.6, 1.4)   # ratio of semi-major to semi-minor
    rotation = rng.uniform(0, math.pi)

    # Semi-axes derived from target coverage area (≈ π·a·b)
    area_target = coverage * width * height
    r_base = math.sqrt(area_target / (math.pi * aspect))
    a = r_base * math.sqrt(aspect)   # semi-major axis
    b = r_base / math.sqrt(aspect)   # semi-minor axis

    # Periodic fBm using harmonic cosines so the noise wraps seamlessly at 0 = 2π
    thetas = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    noise = np.zeros(n_points, dtype=np.float32)
    amplitude = 1.0
    total_amplitude = 0.0
    for k in range(1, octaves + 1):
        phase = rng.uniform(0, 2 * math.pi)
        noise += amplitude * np.cos(k * thetas + phase)
        total_amplitude += amplitude
        amplitude *= persistence
    noise /= total_amplitude
    std = noise.std()
    if std > 1e-6:
        noise = noise / std * wobble_scale   # normalise so ±wobble_scale bounds hold approximately

    # Build polygon: for each sample angle, compute the ellipse radius at that
    # angle (accounting for rotation), then scale by the noise modulation.
    points = []
    for i, theta in enumerate(thetas):
        t_rot = theta - rotation
        r = (a * b) / math.sqrt((b * math.cos(t_rot)) ** 2 + (a * math.sin(t_rot)) ** 2)
        r_wobbled = max(r * (1.0 + noise[i]), 1.0)
        points.append((cx + r_wobbled * math.cos(theta), cy + r_wobbled * math.sin(theta)))

    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(points, fill=255)
    return img


# ---------------------------------------------------------------------------
# Combiner
# ---------------------------------------------------------------------------

def combine_masks(*masks: Image.Image, mode: str = "union") -> Image.Image:
    """
    Combine multiple "L" mode masks.

    mode : "union"        — pixel is masked if ANY mask covers it  (logical OR)
           "intersection" — pixel is masked if ALL masks cover it  (logical AND)
    """
    arrays = [np.array(m, dtype=np.uint8) for m in masks]
    if mode == "union":
        result = arrays[0].copy()
        for a in arrays[1:]:
            result = np.maximum(result, a)
    else:
        result = arrays[0].copy()
        for a in arrays[1:]:
            result = np.minimum(result, a)
    return Image.fromarray(result, mode="L")


# ---------------------------------------------------------------------------
# High-level entry point used by BaseDataset.random_mask
# ---------------------------------------------------------------------------

# Probability weights for mask type selection
_TYPE_WEIGHTS = {
    "cloud":   0.40,
    "polygon": 0.35,
    "shape":   0.25,
}

def random_mask(
    width: int,
    height: int,
    cloud_octaves: int = 4,
    cloud_persistence: float = 0.5,
    cloud_threshold: float = 0.2,
    polygon_n_points: int = 6,
    polygon_irregularity: float = 0.4,
    polygon_n_polygons: int = 1,
    min_coverage: float = 0.1,
    max_coverage: float = 0.6,
    combine_cloud_and_shape: bool = True,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Generate a random inpainting mask using a randomly chosen (or combined) strategy.

    With combine_cloud_and_shape=True (default) there is a 33% chance the result
    is the union of a cloud mask and a polygon/shape mask, which produces more
    varied and naturalistic training masks.
    """
    rng = random.Random(seed)
    np_seed = rng.randint(0, 2**31)
    poly_seed = rng.randint(0, 2**31)

    mask_type = rng.choices(
        list(_TYPE_WEIGHTS.keys()), weights=list(_TYPE_WEIGHTS.values())
    )[0]

    if mask_type == "cloud":
        threshold = rng.uniform(max(0.05, cloud_threshold - 0.1), min(0.4, cloud_threshold + 0.1))
        m = cloud_mask(width, height, octaves=cloud_octaves, persistence=cloud_persistence,
                       threshold=threshold, seed=np_seed)
        if combine_cloud_and_shape and rng.random() < 0.33:
            shape_type = rng.choice(["polygon", "shape"])
            if shape_type == "polygon":
                m2 = polygon_mask(width, height, n_points=polygon_n_points,
                                  min_coverage=min_coverage, max_coverage=max_coverage,
                                  irregularity=polygon_irregularity, seed=poly_seed)
            else:
                m2 = shape_mask(width, height, min_coverage=min_coverage,
                                max_coverage=max_coverage, seed=poly_seed)
            m = combine_masks(m, m2)

    elif mask_type == "polygon":
        n_poly = rng.randint(1, max(1, polygon_n_polygons * 2))
        m = polygon_mask(width, height, n_points=polygon_n_points,
                         min_coverage=min_coverage, max_coverage=max_coverage,
                         irregularity=polygon_irregularity, n_polygons=n_poly,
                         seed=poly_seed)

    else:  # shape
        m = shape_mask(width, height, min_coverage=min_coverage,
                       max_coverage=max_coverage, seed=poly_seed)

    return m
