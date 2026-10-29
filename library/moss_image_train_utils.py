# MOSS-Image training / sampling utilities
# The sampler follows the official `MOSSImageScheduler` (pipeline_moss_image.py): the shifted inference grid is snapped
# to the nearest sigma of the training grid (shift 1, 1000 steps) and integrated with Euler steps in float32.

import argparse
import gc
import os
import time
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from accelerate import Accelerator, PartialState
from PIL import Image
from tqdm import tqdm

from library import device_utils, flux2_vae, flux_train_utils, moss_image_models, moss_image_ref_gate, moss_image_utils, sampling
from library.moss_image_ref_gate import ReferenceGate
from library.device_utils import clean_memory_on_device, synchronize_device
import library.compile_utils as compile_utils
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_UNCOND_SKIP_LAYERS = (22,)  # block skipped in the negative branch of CFG by the official pipeline

# defaults of the official pipeline, used for sample image generation
DEFAULT_SAMPLE_STEPS = 50
DEFAULT_SAMPLE_GUIDANCE_SCALE = 3.0
DEFAULT_SAMPLE_FLOW_SHIFT = 5.0
DEFAULT_SAMPLE_SIZE = 1024


# region arguments


def add_moss_image_model_arguments(parser: argparse.ArgumentParser):
    """Arguments to load the MOSS-Image models (shared by the trainer and the caching tools). `--vae` and
    `--tokenizer_cache_dir` are registered by `add_sd_models_arguments`."""
    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="path to the Qwen3.5-9B text encoder checkpoint: the first shard of the diffusers format (model-00001-of-00004.safetensors) or the ComfyUI single file"
        + " / Qwen3.5-9B Text Encoder のチェックポイント。diffusers 形式なら最初の分割ファイル（model-00001-of-00004.safetensors）、ComfyUI 形式なら単一ファイル",
    )
    parser.add_argument(
        "--moss_max_token_length",
        type=int,
        default=moss_image_utils.MAX_TOKEN_LENGTH,
        help=f"maximum number of text tokens (including the chat template), default and maximum: {moss_image_utils.MAX_TOKEN_LENGTH}"
        + f" / テキストトークンの最大数（chat template を含む）。デフォルトおよび最大値: {moss_image_utils.MAX_TOKEN_LENGTH}",
    )
    parser.add_argument(
        "--convrot_int8_text_encoder",
        action="store_true",
        help="quantize the text encoder (Qwen3.5-9B) weights to ConvRot INT8 at load time (about 16GB -> 9GB). The ComfyUI"
        " pre-quantized file (qwen3.5_9b_int8_convrot.safetensors) is detected automatically and does not need this option"
        + " / Text Encoder（Qwen3.5-9B）の重みをロード時に ConvRot INT8 に量子化する（約 16GB → 9GB）。ComfyUI の事前量子化済みファイル"
        + "（qwen3.5_9b_int8_convrot.safetensors）は自動判別されるのでこのオプションは不要",
    )


def add_convrot_int8_arguments(parser: argparse.ArgumentParser):
    """ConvRot INT8 options for the DiT (shared by the trainer and the inference script)."""
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="quantize the DiT base weights (Linear layers of the main blocks) to ConvRot INT8 at load time: Hadamard rotation"
        " + per-channel int8, fused Triton int8 GEMM in forward (requires triton / triton-windows, otherwise a slower"
        " dequantizing fallback with the same memory saving). An alternative to fp8. The ComfyUI pre-quantized files"
        " (*_int8_convrot.safetensors) are detected automatically and do not need this option"
        + " / DiT のベース重み（メインブロックの Linear 層）をロード時に ConvRot INT8 に量子化する。Hadamard 回転＋チャネル毎 int8、forward は"
        + " Triton の融合 int8 GEMM（triton / triton-windows が必要。ない場合はメモリ削減のみで遅い逆量子化フォールバック）。fp8 の代替。"
        + "ComfyUI の事前量子化済みファイル（*_int8_convrot.safetensors）は自動判別されるのでこのオプションは不要",
    )
    parser.add_argument(
        "--convrot_int8_bwd",
        type=str,
        default="bf16",
        choices=["bf16", "int8"],
        help="backward mode of the ConvRot INT8 layers (DiT and text encoder). bf16 (default): transient dequantized matmul,"
        " most accurate. int8: reuse the fused int8 GEMM for grad_x (faster, quantizes the gradients slightly, requires triton"
        " and CUDA)"
        + " / ConvRot INT8 層（DiT と Text Encoder）の backward の方式。bf16（デフォルト）: 一時的に逆量子化した matmul で最も高精度。"
        + "int8: grad_x にも融合 int8 GEMM を使う（高速だが勾配がわずかに量子化される。triton と CUDA が必要）",
    )


def add_moss_image_training_arguments(parser: argparse.ArgumentParser):
    """MOSS-Image specific training arguments (timestep sampling, attention, compile, sampling, checkpointing)."""
    add_convrot_int8_arguments(parser)
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=1.0,
        help="Timestep distribution shift for rectified flow training, used by --timestep_sampling sigma / shift / truncated_logit_normal"
        + " (default: 1.0, the training grid of MOSS-Image)"
        + " / rectified flow 学習のタイムステップ分布のシフト。--timestep_sampling sigma / shift / truncated_logit_normal で使われる"
        + "（デフォルト: 1.0、MOSS-Image の学習グリッド）",
    )
    parser.add_argument(
        "--timestep_sampling",
        type=str,
        default="sigmoid",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "truncated_logit_normal"],
        help="Timestep sampling method (default: sigmoid (logit normal)). The pre-training distribution of MOSS-Image is reproduced by"
        + " `--timestep_sampling truncated_logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0`"
        + " / タイムステップのサンプリング方法（デフォルト: sigmoid（logit normal））。MOSS-Image の事前学習の分布は"
        + " `--timestep_sampling truncated_logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0` で再現できる",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for sigmoid (logit_normal) timestep sampling (default: 1.0) / sigmoid（logit normal）タイムステップサンプリングのスケール（デフォルト: 1.0）",
    )
    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],  # "sdpa" is for backward compatibility
        default=None,
        help="Attention implementation to use. Default is None (torch). xformers requires --split_attn. sageattn does not support training (inference only). This option overrides --xformers or --sdpa."
        + " / 使用するAttentionの実装。デフォルトはNone（torch）です。xformersは--split_attnの指定が必要です。sageattnはトレーニングをサポートしていません（推論のみ）。このオプションは--xformersまたは--sdpaを上書きします。",
    )
    parser.add_argument(
        "--split_attn",
        action="store_true",
        help="split attention computation to reduce memory usage / メモリ使用量を減らすためにattention時にバッチを分割する",
    )
    parser.add_argument(
        "--sample_uncond_skip_layers",
        type=str,
        default="22",
        help="comma separated DiT block indices skipped in the negative (unconditional) branch of CFG in sample image generation. Default is 22 (official). Empty string disables skipping"
        + " / サンプル画像生成で、CFG の negative（unconditional）側で飛ばす DiT ブロックのインデックス（カンマ区切り）。デフォルトは 22（公式）。空文字列で無効",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap."
        + " / 非同期転送で activation を CPU RAM に退避する（--cpu_offload_checkpointing より高速）。--cpu_offload_checkpointing および --blocks_to_swap とは併用できない",
    )

    # torch.compile (per-block compilation, Triton required) and CUDA performance switches
    compile_utils.add_compile_arguments(parser)


# endregion


# region timestep visualization


def show_timesteps(args):
    """Visualize the sampled timestep / loss weighting distribution for the current settings, then return.

    MOSS-Image reuses ``flux_train_utils.get_noisy_model_input_and_timesteps`` for sampling and
    ``compute_loss_weighting_for_sd3`` for the weighting (1 for ``logit_normal``, which matches the official training).
    """
    from library import sd3_train_utils, timestep_visualization

    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=NUM_TRAIN_TIMESTEPS, shift=args.discrete_flow_shift
    )
    h, w = flux_train_utils.parse_show_timesteps_latent_size(args, vae_compression=16)  # 2x2 patchified latents
    device, dtype = device_utils.get_preferred_device(), torch.float32
    offset, offset_note = flux_train_utils.get_show_timesteps_offset(args)

    def sample_timesteps(bsz):
        latents = torch.zeros(bsz, moss_image_models.MOSS_IMAGE_CONFIG["latent_channels"], h, w, dtype=dtype, device=device)
        noise = torch.ones_like(latents)
        tso = None if offset is None else torch.full((bsz,), offset, dtype=dtype, device=device)
        _, timesteps, _ = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype, timestep_sampling_offset=tso
        )
        return timesteps

    def compute_weighting(timesteps):
        sigmas = timesteps / NUM_TRAIN_TIMESTEPS
        return flux_train_utils.compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)

    header = (
        "Timestep distribution / タイムステップ分布:\n  "
        + flux_train_utils.get_timestep_sampling_info(args)
        + offset_note
        + f", resolution={args.show_timesteps_resolution} (latent {h}x{w})"
    )
    timestep_visualization.show_timestep_distribution(
        args.show_timesteps, sample_timesteps, compute_weighting, num_train_timesteps=NUM_TRAIN_TIMESTEPS, header=header
    )


# endregion


# region sampler


def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count, dtype=torch.float32)
    return shift * values / (1 + (shift - 1) * values)


def get_sampling_sigmas(
    num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = NUM_TRAIN_TIMESTEPS
) -> torch.Tensor:
    """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

    The number of effective steps can be smaller than `num_inference_steps` because the shifted grid is snapped to the
    training grid and duplicates are removed (same as the official scheduler).
    """
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = num_train_timesteps
    sigmas = _sigma_grid(num_inference_steps, shift, total)
    training = _sigma_grid(total, training_shift, total).flip(0)  # ascending
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(latents: torch.Tensor, model_pred: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    """One Euler step in float32: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v"""
    dt = (sigmas[step_index + 1] - sigmas[step_index]).to(torch.float32)
    return latents.float() + dt * model_pred.float()


def parse_skip_layers(value: Optional[str]) -> tuple[int, ...]:
    """Parse a comma separated list of block indices. None or an empty string disables skipping."""
    if value is None:
        return ()
    value = value.strip()
    if not value:
        return ()
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())


def generate_latents(
    dit: moss_image_models.MossImage,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    negative_embeds: Optional[torch.Tensor],
    negative_mask: Optional[torch.Tensor],
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    flow_shift: float,
    uncond_skip_layers: tuple[int, ...],
    seed: Optional[int],
    device: torch.device,
    show_progress: bool = True,
    ref: Optional[torch.Tensor] = None,
    ref_rope_r: float = 1.0,
    ref_kv: Optional[list] = None,
    ref_gate: Optional["ReferenceGate"] = None,
) -> torch.Tensor:
    """Sample normalized, patchified latents [1, 128, H/16, W/16] (float32).

    The initial noise is drawn on a CPU generator in float32 and CFG / Euler updates are done in float32 like the official
    pipeline. CFG is applied when `guidance_scale != 1` and the negative embeddings are given. `ref` (normalized, patchified
    reference latents [1, 128, Hr/16, Wr/16], sequence concatenation) or `ref_kv` (cached reference K/V from
    `compute_reference_kv`, one-way reference) is given to both the conditional and the unconditional branch. With a
    `ref_gate` (learned reference gate table) the reference K/V of each block are gated per step according to the
    timestep band (binarized tables skip the closed blocks, continuous tables use the bias path of the attention).
    """
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)
    shape = (1, dit.latent_channels, height // 16, width // 16)
    latents = torch.randn(shape, generator=generator, dtype=torch.float32).to(device)

    positive = prompt_embeds.to(device, dtype=dit.dtype)
    positive_mask = prompt_mask.to(device)
    do_cfg = guidance_scale != 1.0 and negative_embeds is not None
    if do_cfg:
        negative = negative_embeds.to(device, dtype=dit.dtype)
        negative_mask = negative_mask.to(device)

    sigmas = get_sampling_sigmas(steps, flow_shift).to(device)
    num_steps = len(sigmas) - 1

    if ref is not None:
        ref = ref.to(device, dtype=dit.dtype)
    with torch.no_grad(), tqdm(total=num_steps, desc="Denoising steps", disable=not show_progress) as pbar:
        for i in range(num_steps):
            timestep = (sigmas[i] * NUM_TRAIN_TIMESTEPS).expand(latents.shape[0])
            x = latents.to(dit.dtype)
            step_kv, step_gate = ref_kv, None
            if ref_kv is not None and ref_gate is not None:
                step_kv, step_gate = moss_image_ref_gate.resolve_block_gates(ref_kv, ref_gate.log_gates(timestep))
            prediction = dit(
                x, timestep, positive, positive_mask, ref=ref, ref_rope_r=ref_rope_r, ref_kv=step_kv, ref_gate=step_gate
            ).float()
            if do_cfg:
                uncond = dit(
                    x,
                    timestep,
                    negative,
                    negative_mask,
                    skip_layers=uncond_skip_layers,
                    ref=ref,
                    ref_rope_r=ref_rope_r,
                    ref_kv=step_kv,
                    ref_gate=step_gate,
                ).float()
                prediction = uncond + guidance_scale * (prediction - uncond)
            latents = euler_step(latents, prediction, sigmas, i)
            pbar.update()

    return latents


# region reference image (editing)


def compute_reference_kv(
    dit: moss_image_models.MossImage,
    ref: torch.Tensor,
    ref_timestep: float,
    uncond_embeds: torch.Tensor,
    uncond_mask: Optional[torch.Tensor],
    ref_rope_r: float = 1.0,
    blocks: Optional[Sequence[int]] = None,
    enable_grad: bool = False,
) -> list:
    """Cached reference K/V for the one-way reference (editing, --ref_mode kv): one forward of the clean reference latents
    [B, 128, Hr/16, Wr/16] with the empty prompt embeddings and the fixed `ref_timestep` (sigma * 1000). `enable_grad`
    lets the gradients flow through the reference forward (training with --ref_kv_lora on). The result depends only on the
    reference image (and the LoRA), so it is computed once per image at inference and shared by all steps."""
    batch = ref.shape[0]
    timesteps = torch.full((batch,), float(ref_timestep), dtype=torch.float32, device=ref.device)
    if uncond_embeds.shape[0] == 1 and batch > 1:
        uncond_embeds = uncond_embeds.expand(batch, -1, -1)
        uncond_mask = None if uncond_mask is None else uncond_mask.expand(batch, -1)
    with torch.set_grad_enabled(enable_grad):
        return dit.forward_ref_kv(ref, timesteps, uncond_embeds.to(ref.device), uncond_mask, ref_rope_r=ref_rope_r, blocks=blocks)


def downsample_mask_to_latents(mask: torch.Tensor) -> torch.Tensor:
    """Pixel resolution hole mask [B, H, W] (1 = hole) -> smooth latent resolution mask [B, 1, H/8, W/8] (area average)"""
    return torch.nn.functional.avg_pool2d(mask[:, None].float(), flux2_vae.LATENT_SCALE_FACTOR)


def apply_reference_latent_mask(ref: torch.Tensor, mask_latents: torch.Tensor, fill: float) -> torch.Tensor:
    """Blend the normalized, patchified reference latents [B, 128, h, w] with the fill value using the latent resolution
    mask [B, 1, 2h, 2w]: `ref * (1 - m) + fill * m` per latent cell. The mask is patchified like the latents (the 4 cells of a
    token keep their own mask value: sub-token precision at the hole boundary)."""
    mask = flux2_vae.patchify_latents(mask_latents.to(ref.dtype).expand(-1, ref.shape[1] // 4, -1, -1))  # [B, 128, h, w]
    return ref * (1 - mask) + fill * mask


def load_reference_pixels(
    image_path: str, width: int, height: int, mask_path: Optional[str] = None, mask_from_alpha: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load a reference image resized to (width, height) as pixels [1, 3, H, W] in -1..1 (float32), and the hole mask
    [1, H, W] (1 = hole) from the grayscale `mask_path` image (255 = hole) or, with `mask_from_alpha`, from the alpha channel
    of the image (alpha 255 = hole). The mask is None if neither is given."""
    image = Image.open(image_path)
    rgb = np.array(image.convert("RGB").resize((width, height), Image.LANCZOS), dtype=np.float32)
    pixels = torch.from_numpy(rgb / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0)

    alpha = None
    if mask_path is not None:
        alpha = Image.open(mask_path).convert("L")
    elif mask_from_alpha:
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            alpha = image.convert("RGBA").getchannel("A")
        else:
            logger.warning(f"reference image has no alpha channel: no hole mask is applied: {image_path}")
    if alpha is None:
        return pixels, None
    mask = np.array(alpha.resize((width, height), Image.BILINEAR), dtype=np.float32) / 255.0
    return pixels, torch.from_numpy(mask).unsqueeze(0)


def encode_reference_image(
    vae: flux2_vae.AutoEncoder,
    image_path: str,
    width: int,
    height: int,
    device: torch.device,
    mask_path: Optional[str] = None,
    mask_from_alpha: bool = False,
    fill_latent: float = 0.0,
) -> torch.Tensor:
    """Load a reference image (see `load_reference_pixels`), resize it to (width, height) and encode it to normalized,
    patchified latents [1, 128, H/16, W/16] (float32). The hole mask (if any) is applied like in training: the latents are
    blended with `fill_latent` using the smooth downsampled mask. The VAE is moved to `device` for the encoding and back
    afterwards."""
    pixels, mask = load_reference_pixels(image_path, width, height, mask_path, mask_from_alpha)

    org_vae_device = vae.device
    vae.to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        ref = vae.encode_for_dit(pixels.to(device, dtype=vae.dtype)).float()
    vae.to(org_vae_device)

    if mask is not None:
        ref = apply_reference_latent_mask(ref, downsample_mask_to_latents(mask.to(device)), fill_latent)
    return ref


# endregion


# endregion


# region sample images during training


def should_sample_images(args: argparse.Namespace, epoch, steps) -> bool:
    """Whether sample images are generated at this point of the training (--sample_at_first, --sample_every_n_steps / epochs)"""
    if steps == 0:
        return bool(args.sample_at_first)
    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return False
    if args.sample_every_n_epochs is not None:
        return epoch is not None and epoch % args.sample_every_n_epochs == 0
    return steps % args.sample_every_n_steps == 0 and epoch is None


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: moss_image_models.MossImage,
    vae: flux2_vae.AutoEncoder,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
    ref_gate: Optional["ReferenceGate"] = None,
):
    """Generate sample images during training with the official sampler settings (steps 50, CFG 3.0, flow shift 5.0,
    uncond skip layer 22 by default). Per prompt options: `--w`, `--h`, `--s`, `--l`, `--fs`, `--n`, `--d`, and `--cn`
    (reference image for reference-conditioned editing, resized to the sample size) with `--mk` (hole mask, 255 = hole; with
    `--ref_mask_source alpha` the alpha channel of the reference image is used instead). `ref_gate` (optional) is the learned
    reference gate table of the KV cache editing (`--ref_mode kv --ref_gate`)."""
    if not should_sample_images(args, epoch, steps):
        return

    logger.info(f"Generating sample images at step {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    dit.switch_block_swap_for_inference()

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    prompts = sampling.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    def _run_one(prompt_dict):
        dit.prepare_block_swap_before_forward()
        _sample_image_inference(
            accelerator,
            args,
            dit,
            text_encoder,
            vae,
            tokenize_strategy,
            text_encoding_strategy,
            save_dir,
            prompt_dict,
            epoch,
            steps,
            sample_prompts_te_outputs,
            prompt_replacement,
            ref_gate,
        )

    if distributed_state.num_processes <= 1:
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                _run_one(prompt_dict)
    else:
        # Distribute prompts across processes. Each process handles prompts[i :: num_processes].
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad(), accelerator.autocast():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    _run_one(prompt_dict)

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def _sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    dit: moss_image_models.MossImage,
    text_encoder,
    vae: flux2_vae.AutoEncoder,
    tokenize_strategy,
    text_encoding_strategy,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
    ref_gate=None,
):
    """Generate a single sample image."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")  # "" is the official default negative prompt
    sample_steps = prompt_dict.get("sample_steps", DEFAULT_SAMPLE_STEPS)
    width = prompt_dict.get("width", DEFAULT_SAMPLE_SIZE)
    height = prompt_dict.get("height", DEFAULT_SAMPLE_SIZE)
    scale = prompt_dict.get("scale", DEFAULT_SAMPLE_GUIDANCE_SCALE)
    seed = prompt_dict.get("seed")
    flow_shift = float(prompt_dict.get("flow_shift", DEFAULT_SAMPLE_FLOW_SHIFT))
    uncond_skip_layers = parse_skip_layers(args.sample_uncond_skip_layers)
    ref_image_path = prompt_dict.get("controlnet_image")  # `--cn`: reference image (editing)
    ref_mask_path = prompt_dict.get("mask_image")  # `--mk`: optional hole mask (255 = hole)
    ref_rope_r = float(getattr(args, "ref_rope_r", 1.0))
    ref_mask_from_alpha = getattr(args, "ref_mask_source", "none") == "alpha"
    ref_mode = getattr(args, "ref_mode", "kv")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    height = max(16, height - height % 16)
    width = max(16, width - width % 16)

    logger.info(
        f"  prompt: {prompt}, negative: {negative_prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}, "
        f"flow_shift: {flow_shift}, uncond skip layers: {uncond_skip_layers}, seed: {seed}"
        + ("" if ref_image_path is None else f", reference image: {ref_image_path} (mode: {ref_mode}, rope r: {ref_rope_r})")
    )

    # Encode prompt: [prompt_embeds [1, L, C], attn_mask [1, L]]
    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            return text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return
    prompt_embeds, prompt_mask = encoded[0], encoded[1]

    negative_embeds = negative_mask = None
    if scale != 1.0:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is None:
            logger.warning("Cannot encode the negative prompt, CFG is disabled for this sample")
        else:
            negative_embeds, negative_mask = neg_encoded[0], neg_encoded[1]

    # Encode the reference image (resized to the sample size) with the VAE in float32 outside autocast
    ref = None
    if ref_image_path is not None:
        ref = encode_reference_image(
            vae,
            ref_image_path,
            width,
            height,
            accelerator.device,
            ref_mask_path,
            ref_mask_from_alpha,
            getattr(args, "ref_mask_fill_latent", 0.0),
        )

    # One-way reference: cache the reference K/V once (empty prompt, fixed timestep, LoRA off)
    ref_kv = None
    if ref is not None and ref_mode == "kv":
        uncond = encode_prompt("")
        if uncond is None:
            logger.warning(
                "Cannot encode the empty prompt for the reference forward (not in the cached sample prompt outputs and no text"
                " encoder), skipping sample"
            )
            return
        with accelerator.autocast():
            ref_kv = compute_reference_kv(
                dit,
                ref.to(dit.dtype),
                float(getattr(args, "ref_timestep", 0.0)),
                uncond[0].to(dit.dtype),
                uncond[1],
                ref_rope_r,
                parse_skip_layers(getattr(args, "ref_kv_blocks", None)) or None,
            )
        ref = None

    clean_memory_on_device(accelerator.device)
    latents = generate_latents(
        dit,
        prompt_embeds,
        prompt_mask,
        negative_embeds,
        negative_mask,
        height,
        width,
        sample_steps,
        scale,
        flow_shift,
        uncond_skip_layers,
        seed,
        accelerator.device,
        ref=ref,
        ref_rope_r=ref_rope_r,
        ref_kv=ref_kv,
        ref_gate=ref_gate if ref_kv is not None else None,
    )
    del ref_kv

    # Decode latents in float32 outside autocast (the VAE is in float32)
    gc.collect()
    synchronize_device(accelerator.device)
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device
    vae.to(accelerator.device)
    with torch.autocast(device_type=accelerator.device.type, enabled=False):
        pixels = vae.decode_from_dit(latents.to(accelerator.device, dtype=vae.dtype))
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    pixels = ((pixels[0].float().clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    image = Image.fromarray(pixels)

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i = prompt_dict.get("enum", 0)
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # Log to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)


# endregion
