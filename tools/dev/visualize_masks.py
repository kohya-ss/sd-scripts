#!/usr/bin/env python3
"""
Visualise inpainting mask generation.

Produces a grid PNG for each mask type (cloud, polygon, shape, combined)
showing both the raw mask and the mask applied to a source image.

Source images are loaded from a downloaded-data directory (DreamBooth folder
structure: <repeats>_<concept>/*.jpg|png) produced by download_training_data.py.
Falls back to synthetic images when no real images are found.

Usage:
    # Use downloaded images (default location)
    python3 tests/visualize_masks.py

    # Specify a custom data directory
    python3 tests/visualize_masks.py --data-dir tests/downloaded_data

    # Synthetic fallback (no data dir or empty)
    python3 tests/visualize_masks.py --data-dir /nonexistent

Output files (in --out-dir):
    cloud_masks.png
    polygon_masks.png
    shape_masks.png
    combined_masks.png
    all_random_masks.png
"""

import argparse
import sys
import os
import glob
import random

# Allow running from repo root or tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw

from library.mask_generator import (
    cloud_mask,
    polygon_mask,
    shape_mask,
    combine_masks,
    random_mask,
    wobbly_ellipse_mask,
)


# ---------------------------------------------------------------------------
# Image source
# ---------------------------------------------------------------------------

def load_image_pool(data_dir: str) -> list:
    """
    Walk a DreamBooth-style data directory and return all image paths found.

    Expected structure (produced by download_training_data.py):
        <data_dir>/<repeats>_<concept>/image_00000.jpg
        <data_dir>/<repeats>_<concept>/image_00001.png
        ...

    Returns a sorted list of absolute paths. Empty list if nothing is found.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return []
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        paths.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    return sorted(paths)


def _synthetic_image(width: int, height: int, seed: int) -> Image.Image:
    """Simple synthetic image used as fallback when no real images are available."""
    rng = random.Random(seed)
    bg = tuple(rng.randint(40, 200) for _ in range(3))
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(5, 12)):
        x1 = rng.randint(0, width - 60)
        y1 = rng.randint(0, height - 60)
        x2 = min(x1 + rng.randint(50, 200), width)
        y2 = min(y1 + rng.randint(50, 200), height)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)
    return img


class ImageSource:
    """
    Provides source images for the visualiser.

    If a pool of real images is available they are used (cycling with a fixed
    offset so each gallery sees a different slice of the pool). Otherwise falls
    back to synthetic images.
    """

    def __init__(self, pool: list):
        self._pool = pool
        if pool:
            print(f"  Using {len(pool)} real image(s) from data directory.")
        else:
            print("  No real images found — using synthetic images.")

    def get(self, index: int, width: int, height: int) -> Image.Image:
        if self._pool:
            path = self._pool[index % len(self._pool)]
            try:
                return Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
            except Exception as e:
                print(f"  Warning: could not open {path}: {e} — using synthetic fallback")
        return _synthetic_image(width, height, seed=index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_mask(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Show masked region as mid-grey, keep original elsewhere."""
    grey = Image.new("RGB", image.size, (128, 128, 128))
    mask_bin = mask.point(lambda p: 255 if p >= 128 else 0)
    return Image.composite(grey, image, mask_bin)


def _label(text: str, width: int, height: int = 20) -> Image.Image:
    """Render a small label bar."""
    bar = Image.new("RGB", (width, height), (30, 30, 30))
    draw = ImageDraw.Draw(bar)
    draw.text((4, 2), text, fill=(220, 220, 220))
    return bar


def make_grid(
    samples: list,          # list of (label, mask_img, source_img)
    cols: int,
    cell_size: int,
) -> Image.Image:
    """Arrange (mask, composite) pairs in a grid."""
    label_h = 20
    cell_h = cell_size * 2 + label_h   # mask row + composite row + label
    rows = (len(samples) + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * cell_h
    grid = Image.new("RGB", (grid_w, grid_h), (60, 60, 60))

    for i, (label, mask, source) in enumerate(samples):
        col = i % cols
        row = i // cols
        x = col * cell_size
        y = row * cell_h

        mask_r  = mask.resize((cell_size, cell_size), Image.NEAREST)
        comp_r  = _apply_mask(source.resize((cell_size, cell_size)), mask_r)
        lbl_bar = _label(label, cell_size, label_h)

        grid.paste(lbl_bar, (x, y))
        grid.paste(mask_r.convert("RGB"), (x, y + label_h))
        grid.paste(comp_r,               (x, y + label_h + cell_size))

    return grid


# ---------------------------------------------------------------------------
# Gallery generators
# ---------------------------------------------------------------------------

def gallery_cloud(size: int, n: int, cols: int, src: ImageSource) -> Image.Image:
    samples = []
    param_sets = [
        dict(octaves=2, persistence=0.5, base_scale=16.0, threshold=0.2),
        dict(octaves=4, persistence=0.5, base_scale=16.0, threshold=0.2),
        dict(octaves=6, persistence=0.5, base_scale=16.0, threshold=0.2),
        dict(octaves=4, persistence=0.3, base_scale=16.0, threshold=0.2),
        dict(octaves=4, persistence=0.7, base_scale=16.0, threshold=0.2),
        dict(octaves=4, persistence=0.5, base_scale= 8.0, threshold=0.2),
        dict(octaves=4, persistence=0.5, base_scale=32.0, threshold=0.2),
        dict(octaves=4, persistence=0.5, base_scale=16.0, threshold=0.1),
        dict(octaves=4, persistence=0.5, base_scale=16.0, threshold=0.15),
        dict(octaves=4, persistence=0.5, base_scale=16.0, threshold=0.25),
        dict(octaves=4, persistence=0.5, base_scale=16.0, threshold=0.3),
        dict(octaves=3, persistence=0.4, base_scale=24.0, threshold=0.15),
    ]
    for i in range(n):
        p = param_sets[i % len(param_sets)]
        mask = cloud_mask(size, size, seed=i * 7, **p)
        image = src.get(i, size, size)
        lbl = f"sc={p['base_scale']} oct={p['octaves']} per={p['persistence']} thr={p['threshold']}"
        samples.append((lbl, mask, image))
    return make_grid(samples, cols, size)


def gallery_polygon(size: int, n: int, cols: int, src: ImageSource) -> Image.Image:
    samples = []
    param_sets = [
        dict(n_points=4,  irregularity=0.1, n_polygons=1),
        dict(n_points=6,  irregularity=0.3, n_polygons=1),
        dict(n_points=8,  irregularity=0.5, n_polygons=1),
        dict(n_points=12, irregularity=0.7, n_polygons=1),
        dict(n_points=6,  irregularity=0.4, n_polygons=2),
        dict(n_points=6,  irregularity=0.4, n_polygons=3),
        dict(n_points=5,  irregularity=0.8, n_polygons=1),
        dict(n_points=3,  irregularity=0.2, n_polygons=4),
    ]
    for i in range(n):
        p = param_sets[i % len(param_sets)]
        mask = polygon_mask(size, size, seed=i * 13,
                            min_coverage=0.1, max_coverage=0.55, **p)
        image = src.get(i, size, size)
        lbl = f"pts={p['n_points']} irr={p['irregularity']} n={p['n_polygons']}"
        samples.append((lbl, mask, image))
    return make_grid(samples, cols, size)


def gallery_shape(size: int, n: int, cols: int, src: ImageSource) -> Image.Image:
    samples = []
    for i in range(n):
        s = ["rectangle", "ellipse", "random"][i % 3]
        mask = shape_mask(size, size, min_coverage=0.05, max_coverage=0.6,
                          shape=s, seed=i * 17)
        image = src.get(i, size, size)
        samples.append((f"shape={s}", mask, image))
    return make_grid(samples, cols, size)


def gallery_combined(size: int, n: int, cols: int, src: ImageSource) -> Image.Image:
    samples = []
    for i in range(n):
        c_mask = cloud_mask(size, size, octaves=6, persistence=0.5,
                            threshold=0.5, seed=i * 3)
        p_mask = polygon_mask(size, size, n_points=6, irregularity=0.4,
                              min_coverage=0.1, max_coverage=0.5, seed=i * 5)
        mask  = combine_masks(c_mask, p_mask)
        image = src.get(i, size, size)
        samples.append((f"cloud+polygon #{i}", mask, image))
    return make_grid(samples, cols, size)


def gallery_wobbly_ellipse(size: int, n: int, cols: int, src: ImageSource) -> Image.Image:
    samples = []
    param_sets = [
        dict(coverage=0.2,  wobble_scale=0.1),
        dict(coverage=0.2,  wobble_scale=0.25),
        dict(coverage=0.2,  wobble_scale=0.5),
        dict(coverage=0.3,  wobble_scale=0.25),
        dict(coverage=0.4,  wobble_scale=0.25),
        dict(coverage=0.5,  wobble_scale=0.25),
        dict(coverage=0.3,  wobble_scale=0.1),
        dict(coverage=0.3,  wobble_scale=0.5),
    ]
    for i in range(n):
        p = param_sets[i % len(param_sets)]
        mask = wobbly_ellipse_mask(size, size, seed=i * 19, **p)
        image = src.get(i, size, size)
        lbl = f"cov={p['coverage']} wob={p['wobble_scale']}"
        samples.append((lbl, mask, image))
    return make_grid(samples, cols, size)


def gallery_random(size: int, n: int, cols: int, src: ImageSource) -> Image.Image:
    samples = []
    for i in range(n):
        mask = random_mask(size, size, seed=i * 11)
        image = src.get(i, size, size)
        samples.append((f"random #{i}", mask, image))
    return make_grid(samples, cols, size)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise inpainting masks")
    parser.add_argument("--out-dir",  default=os.path.join(os.path.dirname(__file__), "mask_viz"))
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "downloaded_data"),
                        help="Directory of downloaded training images (DreamBooth folder structure). "
                             "Falls back to synthetic images if not found or empty.")
    parser.add_argument("--size", type=int, default=256, help="Image cell size (pixels)")
    parser.add_argument("--cols", type=int, default=4,  help="Grid columns")
    parser.add_argument("--n",    type=int, default=16, help="Samples per gallery")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pool = load_image_pool(args.data_dir)
    src = ImageSource(pool)

    galleries = [
        ("cloud_masks.png",          gallery_cloud),
        ("polygon_masks.png",        gallery_polygon),
        ("shape_masks.png",          gallery_shape),
        ("combined_masks.png",       gallery_combined),
        ("wobbly_ellipse_masks.png", gallery_wobbly_ellipse),
        ("all_random_masks.png",     gallery_random),
    ]

    for filename, fn in galleries:
        out_path = os.path.join(args.out_dir, filename)
        print(f"Generating {filename}...")
        img = fn(args.size, args.n, args.cols, src)
        img.save(out_path)
        print(f"  Saved {img.size[0]}x{img.size[1]} → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
