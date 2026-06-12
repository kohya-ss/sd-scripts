#!/usr/bin/env python3
# If "python" is not found, run with: python3 generate_inpainting_test_data.py
"""
Generate synthetic test images for the inpainting training test.

Creates simple geometric images that exercise the full inpainting data
pipeline without requiring real photographs or external downloads.

Output structure (DreamBooth-style):
  tests/test_data/1_testconcept/
      image_00.png  + image_00.caption
      ...
      image_09.png  + image_09.caption
"""

import os
import random
from PIL import Image, ImageDraw

CAPTIONS = [
    "a red circle on a blue background",
    "yellow rectangles arranged on green",
    "purple ellipses on an orange canvas",
    "cyan triangles on magenta",
    "white squares on a dark gray background",
    "overlapping colored shapes on beige",
    "concentric circles in warm colors",
    "diagonal stripes in cool tones",
    "random colored rectangles on black",
    "bright geometric pattern on white",
]


def make_image(width: int, height: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    bg = (rng.randint(30, 220), rng.randint(30, 220), rng.randint(30, 220))
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    for _ in range(rng.randint(4, 10)):
        x1 = rng.randint(0, width - 80)
        y1 = rng.randint(0, height - 80)
        x2 = min(x1 + rng.randint(60, 220), width)
        y2 = min(y1 + rng.randint(60, 220), height)
        color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
        if rng.random() < 0.5:
            draw.rectangle([x1, y1, x2, y2], fill=color)
        else:
            draw.ellipse([x1, y1, x2, y2], fill=color)

    return img


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "test_data", "1_testconcept")
    os.makedirs(out_dir, exist_ok=True)

    num_images = len(CAPTIONS)
    for i in range(num_images):
        img = make_image(512, 512, seed=i)
        img.save(os.path.join(out_dir, f"image_{i:02d}.png"))
        with open(os.path.join(out_dir, f"image_{i:02d}.caption"), "w") as f:
            f.write(CAPTIONS[i])

    print(f"Generated {num_images} images in: {out_dir}")


if __name__ == "__main__":
    main()
