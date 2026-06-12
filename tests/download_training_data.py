#!/usr/bin/env python3
"""
Download training images from common-canvas/commoncatalog-cc-by.

Images are saved in DreamBooth folder format so they can be used directly
with train_network.py / sdxl_train_network.py.

Output structure:
    <out_dir>/<repeats>_<concept>/
        image_00000.jpg
        image_00000.caption
        ...

Usage:
    python3 tests/download_training_data.py --out-dir tests/downloaded_data
    python3 tests/download_training_data.py --out-dir tests/downloaded_data --n 500 --repeats 5 --concept photography
    python3 tests/download_training_data.py --min-size 1024  # filter for 1024x1024+
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

# os._exit(0) is used at the end of main() to avoid a crash on exit caused by
# HuggingFace datasets' PyArrow background prefetch threads — they crash during
# normal Python interpreter teardown with "terminate called without an active
# exception". os._exit() bypasses teardown and kills them cleanly.


def build_caption(record: dict) -> str:
    """Combine available text fields into a training caption."""
    parts = []
    title = (record.get("title") or "").strip()
    description = (record.get("description") or "").strip()
    tags = (record.get("usertags") or "").strip()

    if title:
        parts.append(title)
    if description and description != title:
        parts.append(description)
    if tags:
        tag_list = [t.strip() for t in tags.split() if t.strip()]
        if tag_list:
            parts.append(", ".join(tag_list))

    return ", ".join(parts) if parts else "photograph"


def download_image(url: str, dest_path: str, timeout: int = 15, min_size: int = 0) -> bool:
    """
    Download a single image. Returns True on success.

    If min_size > 0, opens the saved file with PIL and verifies both dimensions
    are at least min_size pixels. Deletes the file and returns False if not.
    This catches cases where the dataset metadata dimensions don't match what
    the server actually serves (e.g. Flickr returning a scaled-down version).
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1024:
            return False
        with open(dest_path, "wb") as f:
            f.write(data)
    except (urllib.error.URLError, OSError, TimeoutError):
        return False

    if min_size > 0:
        try:
            from PIL import Image
            with Image.open(dest_path) as im:
                w, h = im.size
            if w < min_size or h < min_size:
                os.remove(dest_path)
                return False
        except Exception:
            os.remove(dest_path)
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Download commoncatalog training images")
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "downloaded_data"),
                        help="Root output directory")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of images to successfully download")
    parser.add_argument("--repeats", type=int, default=1,
                        help="DreamBooth repeat count (prefix of subfolder name)")
    parser.add_argument("--concept", type=str, default="photo",
                        help="Concept token (suffix of subfolder name)")
    parser.add_argument("--timeout", type=int, default=15,
                        help="Per-image download timeout in seconds")
    parser.add_argument("--max-attempts", type=int, default=0,
                        help="Stop after this many dataset rows regardless of success (0 = unlimited)")
    parser.add_argument("--min-size", type=int, default=1024,
                        help="Minimum width AND height in pixels (default: 1024). Use 0 to disable.")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    folder_name = f"{args.repeats}_{args.concept}"
    out_dir = Path(args.out_dir) / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming commoncatalog-cc-by dataset...")
    print(f"Target: {args.n} images >= {args.min_size}px → {out_dir}")
    print()

    dataset = load_dataset(
        "common-canvas/commoncatalog-cc-by",
        streaming=True,
    )

    saved = 0
    attempted = 0
    skipped_status = 0
    skipped_size = 0
    skipped_download = 0

    # Wrap iteration in try/finally so we always delete the iterator explicitly
    # before Python exit. The HuggingFace streaming backend uses PyArrow threads
    # that crash with "terminate called without an active exception" if they are
    # still running during interpreter teardown.
    stream_iter = iter(dataset["train"])
    try:
        for record in stream_iter:
            if saved >= args.n:
                break
            if args.max_attempts > 0 and attempted >= args.max_attempts:
                break

            attempted += 1

            if record.get("status") != "success":
                skipped_status += 1
                continue

            # Size filter: the dataset carries width/height of the original image
            if args.min_size > 0:
                w = record.get("original_width") or record.get("width") or 0
                h = record.get("original_height") or record.get("height") or 0
                if w < args.min_size or h < args.min_size:
                    skipped_size += 1
                    continue

            # Prefer the original-resolution URL over the scaled thumbnail
            url = record.get("url") or record.get("downloadurl") or ""
            if not url:
                skipped_download += 1
                continue

            ext = record.get("ext") or "jpg"
            img_name = f"image_{saved:05d}.{ext}"
            img_path = out_dir / img_name
            cap_path = out_dir / f"image_{saved:05d}.caption"

            if download_image(url, str(img_path), timeout=args.timeout, min_size=args.min_size):
                caption = build_caption(record)
                cap_path.write_text(caption, encoding="utf-8")
                saved += 1
                print(f"  [{saved:4d}/{args.n}] {img_name}  \"{caption[:72]}\"")
            else:
                skipped_download += 1

    finally:
        del stream_iter

    print()
    print(f"Done: {saved} saved, {attempted} rows examined")
    print(f"  skipped (status)  : {skipped_status}")
    print(f"  skipped (size)    : {skipped_size}")
    print(f"  skipped (download): {skipped_download}")
    print(f"Data directory: {out_dir}")

    # Flush before bypassing Python teardown — os._exit() skips atexit handlers
    # including the normal stdout flush, so we do it explicitly here.
    sys.stdout.flush()
    # Bypass Python interpreter teardown to prevent PyArrow background threads
    # from crashing with "terminate called without an active exception".
    os._exit(0)


if __name__ == "__main__":
    main()
