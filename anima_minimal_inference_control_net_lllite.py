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
import os
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

import anima_minimal_inference as ami
from library import anima_train_utils
from networks.control_net_lllite_anima import (
    COND_INPUT_SPACES,
    REF_CONTEXT_MODES,
    ControlNetLLLiteDiT,
    build_cond_tensors,
    build_uncond_ref_context,
    encode_reference_hidden_states,
    install_ref_context_dispatch,
    load_lllite_weights,
    parse_ref_blocks,
)
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


def _load_mask_image(
    path: str, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load and binarize a mask image to a (1, 1, H, W) tensor in {0, 1}.
    1.0 = inpaint area (穴), 0.0 = keep.
    """
    img = Image.open(path).convert("L")
    if img.size != (width, height):
        img = img.resize((width, height), Image.NEAREST)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).contiguous()
    return t.to(device=device, dtype=dtype)


# VAE used to encode the control image in latent cond mode. Loaded lazily and kept on CPU
# between prompts (mirrors how anima_minimal_inference handles the decode VAE).
_cond_vae = None


def _get_cond_vae(args):
    global _cond_vae
    if _cond_vae is None:
        logger.info("Loading VAE for LLLite cond encoding (latent cond input space)...")
        _cond_vae = anima_train_utils.load_qwen_image_vae(args, device="cpu", disable_mmap=True)
        _cond_vae.to(torch.bfloat16)
        _cond_vae.eval()
        _cond_vae.requires_grad_(False)
    return _cond_vae


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
    parser.add_argument("--qwen_image_vae_2d", action="store_true")
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
    parser.add_argument(
        "--mask_image", type=str, default=None,
        help=(
            "[inpaint] global mask image. Required for single-prompt inpainting (cond_in_channels=4). "
            "Per-prompt override: --mk <path>."
        ),
    )
    parser.add_argument(
        "--lllite_cond_in_channels", type=int, default=None,
        help="override cond_in_channels from weights metadata (3 or 4)",
    )
    parser.add_argument(
        "--lllite_inpaint_masked_input", type=str, default=None, choices=["true", "false"],
        help="override inpaint_masked_input from weights metadata (true/false)",
    )
    parser.add_argument(
        "--lllite_cond_input", type=str, default=None, choices=list(COND_INPUT_SPACES),
        help="override cond_input_space from weights metadata (pixel/latent)",
    )
    parser.add_argument(
        "--lllite_ref_block", type=str, default=None,
        help=(
            "[semantic trunk] override ref_block from weights metadata "
            "(comma-separated for the dual/multi concat trunk, e.g. '2,13')"
        ),
    )
    parser.add_argument(
        "--lllite_ref_timestep", type=float, default=None,
        help="[semantic trunk] override ref_timestep from weights metadata",
    )
    parser.add_argument(
        "--lllite_ref_context", type=str, default=None, choices=list(REF_CONTEXT_MODES),
        help=(
            "[semantic trunk] override ref_context from weights metadata (zero/uncond/caption). "
            "Use the mode the weights were trained with; overriding is for ablation only. "
            "'caption' precomputes one reference forward per CFG branch before the denoising loop"
        ),
    )
    parser.add_argument(
        "--save_gate_maps", type=str, default=None,
        help=(
            "[semantic trunk] directory to dump per-module gate maps (predicted change-region masks) "
            "as grayscale PNGs after each generation"
        ),
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
        elif option == "mk":
            overrides["mask_image"] = value
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
    cond_in_channels = (
        args.lllite_cond_in_channels
        if args.lllite_cond_in_channels is not None
        else int(meta.get("lllite.cond_in_channels", 3))
    )
    if args.lllite_inpaint_masked_input is not None:
        inpaint_masked_input = args.lllite_inpaint_masked_input == "true"
    else:
        inpaint_masked_input = (
            meta.get("lllite.inpaint_masked_input", "false").lower() == "true"
        )
    # cond 入力空間 (v2.1). メタデータ欠落時は pixel (旧重み互換)
    cond_input_space = (
        args.lllite_cond_input
        if args.lllite_cond_input is not None
        else meta.get("lllite.cond_input_space", "pixel")
    )
    # trunk (v3). メタデータ欠落時は stem (旧重み互換)
    trunk = meta.get("lllite.trunk", "stem")
    # single は "13"、dual/multi は "2,13" (カンマ区切り、v3 dual)
    ref_blocks = parse_ref_blocks(
        args.lllite_ref_block
        if args.lllite_ref_block is not None
        else meta.get("lllite.ref_block")
    )
    ref_timestep = (
        args.lllite_ref_timestep
        if args.lllite_ref_timestep is not None
        else float(meta.get("lllite.ref_timestep", 0.0))
    )
    # ref_context (v3). メタデータ欠落時は zero (旧重み互換)
    meta_ref_context = meta.get("lllite.ref_context", "zero")
    ref_context = (
        args.lllite_ref_context if args.lllite_ref_context is not None else meta_ref_context
    )
    if trunk == "semantic" and ref_context != meta_ref_context:
        logger.warning(
            f"ref_context override: weights were trained with '{meta_ref_context}' but running "
            f"with '{ref_context}' (train/inference mismatch; for ablation only)"
        )
    version = meta.get("lllite.version", "?")
    inpaint_log = (
        f", inpaint=on(masked_input={inpaint_masked_input})" if cond_in_channels == 4 else ""
    )
    trunk_log = (
        f", trunk=semantic(ref_blocks={list(ref_blocks) if ref_blocks else None}, "
        f"ref_timestep={ref_timestep}, ref_context={ref_context})"
        if trunk == "semantic"
        else ""
    )
    logger.info(
        f"LLLite config (v{version}): cond_emb_dim={cond_emb_dim}, mlp_dim={mlp_dim}, "
        f"target_layers={target_layers}, cond_dim={cond_dim}, cond_resblocks={cond_resblocks}, "
        f"use_aspp={use_aspp}{(' dilations=' + str(list(aspp_dilations))) if use_aspp else ''}, "
        f"cond_input={cond_input_space}{trunk_log}, "
        f"cond_in_channels={cond_in_channels}{inpaint_log}, multiplier={args.lllite_multiplier}"
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
        cond_in_channels=cond_in_channels,
        inpaint_masked_input=inpaint_masked_input,
        cond_input_space=cond_input_space,
        trunk=trunk,
        ref_block=ref_blocks,
        ref_timestep=ref_timestep,
        ref_context=ref_context if trunk == "semantic" else "zero",
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
    rgb = _load_control_image(ci_path, height, width, device, torch.bfloat16)
    logger.info(f"Loaded control image: {ci_path} -> {tuple(rgb.shape)}")

    if not hasattr(anima, "lllite"):
        raise RuntimeError("DiT has no .lllite attribute; load_dit_model patch was not applied")

    # inpainting (4ch): require a mask image
    mask = None
    if anima.lllite.cond_in_channels == 4:
        mk_path = getattr(args, "mask_image", None)
        if mk_path is None:
            raise ValueError(
                "mask_image is required for 4-channel (inpaint) LLLite. "
                "Specify --mask_image globally, or --mk per prompt in --from_file mode."
            )
        mask = _load_mask_image(mk_path, height, width, device, torch.bfloat16)
        logger.info(
            f"Loaded mask image: {mk_path} (masked_input={anima.lllite.inpaint_masked_input})"
        )

    # latent cond mode: VAE-encode the control image (kept on CPU between prompts)
    is_latent = anima.lllite.cond_input_space == "latent"
    cond_vae = _get_cond_vae(args) if is_latent else None
    if cond_vae is not None:
        cond_vae.to(device)
    try:
        cond_image, cond_mask = build_cond_tensors(
            rgb,
            mask,
            cond_input_space=anima.lllite.cond_input_space,
            cond_in_channels=anima.lllite.cond_in_channels,
            inpaint_masked_input=anima.lllite.inpaint_masked_input,
            vae=cond_vae,
        )
    finally:
        if cond_vae is not None:
            cond_vae.to("cpu")
    logger.info(
        f"LLLite cond ({anima.lllite.cond_input_space}): cond_image={tuple(cond_image.shape)}"
        + (f", cond_mask={tuple(cond_mask.shape)}" if cond_mask is not None else "")
    )

    # honor per-prompt override of multiplier
    anima.lllite.set_multiplier(args.lllite_multiplier)

    dispatch_handle = None
    if anima.lllite.trunk == "semantic":
        # v3: cond latent を凍結 DiT に通し hidden states を条件源にする (デノイズループ前に 1 回)。
        # per-step の t は dit.t_embedding_norm の forward hook が毎ステップ配る。
        # h_ref は t 不変なので、ref_context='caption' でも (画像, context) 毎の前計算で足りる
        anima.lllite.clear_cond_image()
        ref_context = anima.lllite.ref_context
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=args.fp8
        ):
            if ref_context == "caption":
                # 正プロンプトの context で h_ref を作り、CFG 有効時は negative 側も前計算して
                # forward pre-hook (install_ref_context_dispatch) でブランチごとに切り替える。
                # 学習の「本体と同じ context を参照にも渡す」配線 (caption dropout ↔ uncond) と整合
                ctx_pos = context["embed"][0].to(device, dtype=torch.bfloat16)
                entries = [
                    (
                        ctx_pos,
                        encode_reference_hidden_states(
                            anima, cond_image, anima.lllite.ref_blocks,
                            anima.lllite.ref_timestep, context=ctx_pos,
                        ),
                    )
                ]
                if args.guidance_scale != 1.0:
                    ctx_neg = (context_null if context_null is not None else context)["embed"][0].to(
                        device, dtype=torch.bfloat16
                    )
                    if not torch.equal(ctx_neg, ctx_pos):
                        entries.append(
                            (
                                ctx_neg,
                                encode_reference_hidden_states(
                                    anima, cond_image, anima.lllite.ref_blocks,
                                    anima.lllite.ref_timestep, context=ctx_neg,
                                ),
                            )
                        )
                if len(entries) == 1:
                    anima.lllite.set_cond_hidden_states(entries[0][1])
                else:
                    dispatch_handle = install_ref_context_dispatch(anima, anima.lllite, entries)
                h_ref = entries[0][1]
                branch_log = f"{len(entries)} CFG branch(es)"
            else:
                ref_ctx = (
                    build_uncond_ref_context(
                        anima, device, torch.bfloat16,
                        pad_to_length=context["embed"][0].shape[1],
                    )
                    if ref_context == "uncond"
                    else None
                )
                h_ref = encode_reference_hidden_states(
                    anima, cond_image, anima.lllite.ref_blocks, anima.lllite.ref_timestep,
                    context=ref_ctx,
                )
                anima.lllite.set_cond_hidden_states(h_ref)
                branch_log = "shared"
        logger.info(
            f"LLLite reference forward: h_ref={tuple(h_ref.shape)} "
            f"(ref_blocks={list(anima.lllite.ref_blocks)}, ref_timestep={anima.lllite.ref_timestep}, "
            f"ref_context={ref_context}, {branch_log})"
        )
    else:
        anima.lllite.set_cond_image(cond_image, cond_mask)

    capture_gates = args.save_gate_maps is not None and anima.lllite.trunk == "semantic"
    if args.save_gate_maps is not None and anima.lllite.trunk != "semantic":
        logger.warning("--save_gate_maps is only supported with the semantic trunk (v3); ignored")
    if capture_gates:
        for m in anima.lllite.lllite_modules:
            m.capture_gate = True

    try:
        return _original_generate_body(args, anima, context, context_null, device, seed)
    finally:
        if capture_gates:
            _dump_gate_maps(anima.lllite, args.save_gate_maps, f"seed{seed}")
            for m in anima.lllite.lllite_modules:
                m.capture_gate = False
                m.last_gate = None
        if dispatch_handle is not None:
            dispatch_handle.remove()
        anima.lllite.clear_cond_image()


def _dump_gate_maps(lllite, out_dir: str, prefix: str) -> None:
    """各 LLLite モジュールの最終ステップの gate マップ (= モデルが予測した変更領域マスクの逆:
    1=コピー / 0=書き換え) を token grid 解像度のグレースケール PNG として保存する。"""
    hw = lllite.last_cond_hw
    if hw is None:
        logger.warning("no cond token grid recorded; skipping gate map dump")
        return
    os.makedirs(out_dir, exist_ok=True)
    gates = []
    for m in lllite.lllite_modules:
        if m.last_gate is None:
            continue
        g = m.last_gate[0, :, 0].float().reshape(hw)  # (H, W)
        gates.append(g)
        arr = (g.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(
            os.path.join(out_dir, f"{prefix}_{m.lllite_name}.png")
        )
    if gates:
        mean_g = torch.stack(gates).mean(0)
        arr = (mean_g.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(os.path.join(out_dir, f"{prefix}_mean.png"))
        logger.info(
            f"saved {len(gates)} gate maps (+mean) to {out_dir} "
            f"(grid={hw}, mean gate={mean_g.mean().item():.3f})"
        )


# ---------------------------------------------------------------------------
# install patches and run ami.main
# ---------------------------------------------------------------------------

ami.parse_args = parse_args
ami.parse_prompt_line = parse_prompt_line
ami.load_dit_model = load_dit_model
ami.generate_body = generate_body


if __name__ == "__main__":
    ami.main()
