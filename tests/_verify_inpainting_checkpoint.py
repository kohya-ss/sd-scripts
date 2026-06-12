#!/usr/bin/env python3
"""
Verify a saved checkpoint produced by an inpainting smoke test.

Two modes:

  --expect-in-channels 9
      Treat the file as a full UNet checkpoint (ft / DreamBooth output) and
      assert that the UNet input conv has the expected number of input
      channels. Used by ft-mode smoke tests to catch regressions in the
      4ch -> 9ch conv_in expansion path.

  --expect-lora
      Treat the file as a LoRA-only checkpoint and assert that at least one
      LoRA-shaped key is present. Used by lora-mode smoke tests as a sanity
      check (LoRA outputs do not contain the UNet, so conv_in cannot be
      checked).

Exits 0 on success, 1 on failure with a descriptive message on stderr.

Usage:
    python tests/_verify_inpainting_checkpoint.py path/to/model.safetensors \\
        --expect-in-channels 9
    python tests/_verify_inpainting_checkpoint.py path/to/lora.safetensors \\
        --expect-lora
"""

import argparse
import sys
from pathlib import Path

# Saved SD/SDXL checkpoints store the UNet input conv at this key, regardless
# of whether the source was diffusers-style or the custom SDXL UNet — both
# get converted to "model.diffusion_model.*" on save (see library/model_util.py
# and library/sdxl_model_util.py).
UNET_CONV_IN_KEY = "model.diffusion_model.input_blocks.0.0.weight"


def _load_state_dict_keys(path: Path):
    """Return (keys, get_tensor) where get_tensor(key) lazily fetches a tensor."""
    from safetensors import safe_open

    f = safe_open(str(path), framework="pt")
    return list(f.keys()), f


def verify_in_channels(path: Path, expected: int) -> int:
    keys, f = _load_state_dict_keys(path)
    if UNET_CONV_IN_KEY not in keys:
        print(
            f"FAIL: expected UNet input-conv key not found in {path}\n"
            f"      looked for: {UNET_CONV_IN_KEY}\n"
            f"      first 5 keys present: {keys[:5]}",
            file=sys.stderr,
        )
        return 1
    tensor = f.get_tensor(UNET_CONV_IN_KEY)
    actual = tensor.shape[1]
    if actual != expected:
        print(
            f"FAIL: UNet conv_in in_channels = {actual}, expected {expected}\n"
            f"      key:   {UNET_CONV_IN_KEY}\n"
            f"      shape: {tuple(tensor.shape)}",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: {path.name} has UNet conv_in in_channels = {actual} "
        f"(shape {tuple(tensor.shape)})"
    )
    return 0


def verify_lora(path: Path) -> int:
    keys, _ = _load_state_dict_keys(path)
    lora_keys = [k for k in keys if k.startswith("lora_unet_")]
    if not lora_keys:
        print(
            f"FAIL: no lora_unet_* keys found in {path}\n"
            f"      first 5 keys present: {keys[:5]}",
            file=sys.stderr,
        )
        return 1
    print(
        f"PASS: {path.name} contains {len(lora_keys)} lora_unet_* keys "
        f"(total keys: {len(keys)})"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="Path to .safetensors checkpoint")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--expect-in-channels", type=int,
                       help="Assert UNet conv_in has this many input channels (ft mode)")
    group.add_argument("--expect-lora", action="store_true",
                       help="Assert the file contains LoRA-shaped keys (lora mode)")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"FAIL: file not found: {path}", file=sys.stderr)
        return 1

    if args.expect_lora:
        return verify_lora(path)
    return verify_in_channels(path, args.expect_in_channels)


if __name__ == "__main__":
    sys.exit(main())
