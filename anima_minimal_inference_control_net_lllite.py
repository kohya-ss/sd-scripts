"""Anima ControlNet-LLLite inference.

This script reuses ``anima_minimal_inference`` (single / batch / interactive
modes, latent-decode mode, prompt-line override syntax, etc.) and adds:

  * ``--lllite_weights``      ControlNet-LLLite weights (.safetensors)
  * ``--control_image``       Control image path (single / global)
  * ``--lllite_multiplier``   global LLLite output multiplier
  * Prompt-line overrides ``--cn <path>`` and ``--am <float>`` (per-prompt
    control image / multiplier in batch mode)

Implementation: monkey-patches ``parse_args``, ``parse_prompt_line``,
``load_dit_model`` and ``generate_body`` of ``anima_minimal_inference`` and
then delegates to ``anima_minimal_inference.main()``. All other behavior
(VAE loading, text encoding, save logic, batch/interactive flow, latent-only
decode mode) is inherited unchanged.

Usage examples:

  # single prompt
  python anima_minimal_inference_control_net_lllite.py \
    --dit ... --vae ... --text_encoder ... \
    --lllite_weights out/last.safetensors --control_image canny.png \
    --prompt "a cat" --image_size 1024 1024 --save_path out/

  # batch
  python anima_minimal_inference_control_net_lllite.py \
    --dit ... --vae ... --text_encoder ... \
    --lllite_weights out/last.safetensors --control_image default.png \
    --from_file prompts.txt --save_path out/
  # prompts.txt line:
  #   a cat sitting on a chair --w 1024 --h 1024 --d 42 --cn images/canny_a.png --am 0.8
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

import anima_minimal_inference as ami
from networks.control_net_lllite_anima import ControlNetLLLiteDiT, load_lllite_weights
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_lllite_metadata(weights_path: str) -> Dict[str, str]:
    with safe_open(weights_path, framework="pt") as f:
        meta = f.metadata()
    return meta or {}


def _load_control_image(
    path: str, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load and normalize a control image to a (1, 3, H, W) tensor in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    if img.size != (width, height):  # PIL size is (W, H)
        img = img.resize((width, height), Image.BICUBIC)
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
    return t.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# parse_args (replaces ami.parse_args)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anima ControlNet-LLLite inference")

    # --- mirror anima_minimal_inference.parse_args() ---
    parser.add_argument("--dit", type=str, default=None, help="DiT directory or path")
    parser.add_argument("--vae", type=str, default=None, help="VAE directory or path")
    parser.add_argument("--vae_chunk_size", type=int, default=None)
    parser.add_argument("--vae_disable_cache", action="store_true")
    parser.add_argument("--text_encoder", type=str, required=True, help="Qwen3 Text Encoder path")

    parser.add_argument("--lora_weight", type=str, nargs="*", default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")
    parser.add_argument("--include_patterns", type=str, nargs="*", default=None)
    parser.add_argument("--exclude_patterns", type=str, nargs="*", default=None)

    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="height width")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--flow_shift", type=float, default=5.0)

    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--fp8_scaled", action="store_true")
    parser.add_argument("--text_encoder_cpu", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--attn_mode", type=str, default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],
    )
    parser.add_argument(
        "--output_type", type=str, default="images",
        choices=["images", "latent", "latent_images"],
    )
    parser.add_argument("--no_metadata", action="store_true")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None)
    parser.add_argument(
        "--lycoris", action="store_true",
        help=f"use lycoris{'' if ami.lycoris_available else ' (not available)'}",
    )

    parser.add_argument("--from_file", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")

    # --- LLLite-specific ---
    parser.add_argument(
        "--lllite_weights", type=str, default=None,
        help="ControlNet-LLLite weights (.safetensors). Required unless --latent_path is given.",
    )
    parser.add_argument(
        "--control_image", type=str, default=None,
        help="Path to a control image. May be overridden per-prompt with --cn in --from_file mode.",
    )
    parser.add_argument(
        "--lllite_multiplier", type=float, default=1.0,
        help="LLLite output multiplier (default 1.0). Per-prompt override: --am <float>.",
    )
    parser.add_argument(
        "--lllite_cond_emb_dim", type=int, default=None,
        help="override cond_emb_dim from weights metadata",
    )
    parser.add_argument(
        "--lllite_mlp_dim", type=int, default=None,
        help="override mlp_dim from weights metadata",
    )
    parser.add_argument(
        "--lllite_target_layers", type=str, default=None,
        help="override target_layers from weights metadata (preset or comma-separated atomic specifiers)",
    )
    parser.add_argument(
        "--lllite_cond_dim", type=int, default=None,
        help="override conditioning1 trunk channel width from weights metadata",
    )
    parser.add_argument(
        "--lllite_cond_resblocks", type=int, default=None,
        help="override conditioning1 ResBlock count from weights metadata",
    )
    parser.add_argument(
        "--lllite_use_aspp", type=str, default=None, choices=["true", "false"],
        help="override use_aspp from weights metadata (true/false)",
    )

    args = parser.parse_args()

    # validation (mirrors ami.parse_args + LLLite checks)
    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")

    latents_mode = args.latent_path is not None and len(args.latent_path) > 0
    if not latents_mode:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")
        if args.lllite_weights is None:
            raise ValueError("--lllite_weights is required for inference (unless --latent_path is given)")
        if args.control_image is None and not args.from_file and not args.interactive:
            raise ValueError(
                "--control_image is required for single-prompt inference. "
                "In --from_file mode, you may instead specify --cn per prompt."
            )

    if args.lycoris and not ami.lycoris_available:
        raise ValueError("install lycoris: https://github.com/KohakuBlueleaf/LyCORIS")

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    return args


# ---------------------------------------------------------------------------
# parse_prompt_line (extends ami.parse_prompt_line with --cn / --am)
# ---------------------------------------------------------------------------

def parse_prompt_line(line: str) -> Dict[str, Any]:
    parts = line.split(" --")
    prompt = parts[0].strip()
    overrides: Dict[str, Any] = {"prompt": prompt}

    for part in parts[1:]:
        if not part.strip():
            continue
        option_parts = part.split(" ", 1)
        option = option_parts[0].strip()
        value = option_parts[1].strip() if len(option_parts) > 1 else ""

        if option == "w":
            overrides["image_size_width"] = int(value)
        elif option == "h":
            overrides["image_size_height"] = int(value)
        elif option == "d":
            overrides["seed"] = int(value)
        elif option == "s":
            overrides["infer_steps"] = int(value)
        elif option in ("g", "l"):
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value
        elif option == "cn":
            overrides["control_image"] = value
        elif option == "am":
            overrides["lllite_multiplier"] = float(value)

    return overrides


# ---------------------------------------------------------------------------
# load_dit_model (replaces ami.load_dit_model — also attaches LLLite)
# ---------------------------------------------------------------------------

_original_load_dit_model = ami.load_dit_model


def load_dit_model(args, device, dit_weight_dtype=None):
    dit = _original_load_dit_model(args, device, dit_weight_dtype)

    meta = _read_lllite_metadata(args.lllite_weights)
    cond_emb_dim = (
        args.lllite_cond_emb_dim
        if args.lllite_cond_emb_dim is not None
        else int(meta.get("lllite.cond_emb_dim", 32))
    )
    mlp_dim = (
        args.lllite_mlp_dim
        if args.lllite_mlp_dim is not None
        else int(meta.get("lllite.mlp_dim", 64))
    )
    # canonical atomic 形式 (lllite.target_atomics) を優先的に参照、なければ lllite.target_layers にフォールバック
    target_layers = (
        args.lllite_target_layers
        if args.lllite_target_layers is not None
        else meta.get("lllite.target_atomics", meta.get("lllite.target_layers", "self_attn_q"))
    )
    cond_dim = (
        args.lllite_cond_dim
        if args.lllite_cond_dim is not None
        else int(meta.get("lllite.cond_dim", 64))
    )
    cond_resblocks = (
        args.lllite_cond_resblocks
        if args.lllite_cond_resblocks is not None
        else int(meta.get("lllite.cond_resblocks", 1))
    )
    if args.lllite_use_aspp is not None:
        use_aspp = args.lllite_use_aspp == "true"
    else:
        use_aspp = meta.get("lllite.use_aspp", "false").lower() == "true"
    aspp_dilations_meta = meta.get("lllite.aspp_dilations")
    if use_aspp and aspp_dilations_meta:
        aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
    else:
        from networks.control_net_lllite_anima import ASPP_DEFAULT_DILATIONS as _ASPP_DD
        aspp_dilations = _ASPP_DD
    version = meta.get("lllite.version", "?")
    logger.info(
        f"LLLite config (v{version}): cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}, "
        f"target_layers={target_layers}, cond_dim={cond_dim}, cond_resblocks={cond_resblocks}, "
        f"use_aspp={use_aspp}{(' dilations=' + str(list(aspp_dilations))) if use_aspp else ''}, "
        f"multiplier={args.lllite_multiplier}"
    )

    lllite = ControlNetLLLiteDiT(
        dit,
        cond_emb_dim=cond_emb_dim,
        mlp_dim=mlp_dim,
        target_layers=target_layers,
        multiplier=args.lllite_multiplier,
        cond_dim=cond_dim,
        cond_resblocks=cond_resblocks,
        use_aspp=use_aspp,
        aspp_dilations=aspp_dilations,
    )
    load_lllite_weights(lllite, args.lllite_weights, strict=False)
    lllite.apply_to()
    lllite.to(device=device, dtype=torch.bfloat16)
    lllite.eval().requires_grad_(False)

    # Attach onto dit so generate_body can reach set_cond_image
    dit.lllite = lllite
    return dit


# ---------------------------------------------------------------------------
# generate_body (replaces ami.generate_body — sets cond image before loop)
# ---------------------------------------------------------------------------

_original_generate_body = ami.generate_body


def generate_body(
    args,
    anima,
    context: Dict[str, Any],
    context_null: Optional[Dict[str, Any]],
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    height, width = ami.check_inputs(args)

    ci_path = args.control_image
    if ci_path is None:
        raise ValueError(
            "control_image is not set. Specify --control_image globally, "
            "or --cn per prompt in --from_file mode."
        )
    cond_image = _load_control_image(ci_path, height, width, device, torch.bfloat16)
    logger.info(f"Loaded control image: {ci_path} -> {tuple(cond_image.shape)}")

    if not hasattr(anima, "lllite"):
        raise RuntimeError("DiT has no .lllite attribute; load_dit_model patch was not applied")

    # honor per-prompt override of multiplier
    anima.lllite.set_multiplier(args.lllite_multiplier)
    anima.lllite.set_cond_image(cond_image)

    try:
        return _original_generate_body(args, anima, context, context_null, device, seed)
    finally:
        anima.lllite.clear_cond_image()


# ---------------------------------------------------------------------------
# install patches and run ami.main
# ---------------------------------------------------------------------------

ami.parse_args = parse_args
ami.parse_prompt_line = parse_prompt_line
ami.load_dit_model = load_dit_model
ami.generate_body = generate_body


if __name__ == "__main__":
    ami.main()
