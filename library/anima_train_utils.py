# Anima Training Utilities

import argparse
import math
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from accelerate import Accelerator, PartialState
from tqdm import tqdm
from PIL import Image

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import anima_models, anima_utils, strategy_base, train_util

from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler, get_sigmas


# Anima-specific training arguments

def add_anima_training_arguments(parser: argparse.ArgumentParser):
    """Add Anima-specific training arguments to the parser."""
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path to Anima DiT model safetensors file",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path to WanVAE safetensors/pth file",
    )
    parser.add_argument(
        "--qwen3_path",
        type=str,
        default=None,
        help="Path to Qwen3-0.6B model (safetensors file or directory)",
    )
    parser.add_argument(
        "--llm_adapter_path",
        type=str,
        default=None,
        help="Path to separate LLM adapter weights. If None, adapter is loaded from DiT file if present",
    )
    parser.add_argument(
        "--llm_adapter_lr",
        type=float,
        default=None,
        help="Learning rate for LLM adapter. None=same as base LR, 0=freeze adapter",
    )
    parser.add_argument(
        "--self_attn_lr",
        type=float,
        default=None,
        help="Learning rate for self-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--cross_attn_lr",
        type=float,
        default=None,
        help="Learning rate for cross-attention layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=None,
        help="Learning rate for MLP layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--mod_lr",
        type=float,
        default=None,
        help="Learning rate for AdaLN modulation layers. None=same as base LR, 0=freeze",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer directory. If None, uses default configs/t5_old/",
    )
    parser.add_argument(
        "--qwen3_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for Qwen3 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--t5_max_token_length",
        type=int,
        default=512,
        help="Maximum token length for T5 tokenizer (default: 512)",
    )
    parser.add_argument(
        "--apply_t5_attn_mask",
        action="store_true",
        help="Apply attention mask to T5 tokens in LLM adapter",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="Timestep distribution shift for rectified flow training (default: 3.0)",
    )
    parser.add_argument(
        "--timestep_sample_method",
        type=str,
        default="logit_normal",
        choices=["logit_normal", "uniform"],
        help="Timestep sampling method (default: logit_normal)",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale factor for logit_normal timestep sampling (default: 1.0)",
    )
    # Note: --caption_dropout_rate is defined by base add_dataset_arguments().
    # Anima uses embedding-level dropout (via AnimaTextEncodingStrategy.dropout_rate)
    # instead of dataset-level caption dropout, so the subset caption_dropout_rate
    # is zeroed out in the training scripts to allow caching.
    parser.add_argument(
        "--transformer_dtype",
        type=str,
        default=None,
        choices=["float16", "bfloat16", "float32", None],
        help="Separate dtype for transformer blocks. If None, uses same as mixed_precision",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="Use Flash Attention for DiT self/cross-attention (requires flash-attn package). "
        "Falls back to PyTorch SDPA if flash-attn is not installed.",
    )


# Noise & Timestep sampling (Rectified Flow)
def get_noisy_model_input_and_timesteps(
    args,
    latents: torch.Tensor,
    noise: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate noisy model input and timesteps for rectified flow training.

    Rectified flow: noisy_input = (1 - t) * latents + t * noise
    Target: noise - latents

    Args:
        args: Training arguments with timestep_sample_method, sigmoid_scale, discrete_flow_shift
        latents: Clean latent tensors
        noise: Random noise tensors
        device: Target device
        dtype: Target dtype

    Returns:
        (noisy_model_input, timesteps, sigmas)
    """
    bs = latents.shape[0]

    timestep_sample_method = getattr(args, 'timestep_sample_method', 'logit_normal')
    sigmoid_scale = getattr(args, 'sigmoid_scale', 1.0)
    shift = getattr(args, 'discrete_flow_shift', 1.0)

    if timestep_sample_method == 'logit_normal':
        dist = torch.distributions.normal.Normal(0, 1)
    elif timestep_sample_method == 'uniform':
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError(f"Unknown timestep_sample_method: {timestep_sample_method}")

    t = dist.sample((bs,)).to(device)

    if timestep_sample_method == 'logit_normal':
        t = t * sigmoid_scale
        t = torch.sigmoid(t)

    # Apply shift
    if shift is not None and shift != 1.0:
        t = (t * shift) / (1 + (shift - 1) * t)

    # Clamp to avoid exact 0 or 1
    t = t.clamp(1e-5, 1.0 - 1e-5)

    # Create noisy input: (1 - t) * latents + t * noise
    t_expanded = t.view(-1, *([1] * (latents.ndim - 1)))

    ip_noise_gamma = getattr(args, 'ip_noise_gamma', None)
    if ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        if getattr(args, 'ip_noise_gamma_random_strength', False):
            ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * ip_noise_gamma
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * (noise + ip_noise_gamma * xi)
    else:
        noisy_model_input = (1 - t_expanded) * latents + t_expanded * noise

    # Sigmas for potential loss weighting
    sigmas = t.view(-1, 1)

    return noisy_model_input.to(dtype), t.to(dtype), sigmas.to(dtype)


# Loss weighting

def compute_loss_weighting_for_anima(weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    """Compute loss weighting for Anima training.

    Same schemes as SD3 but can add Anima-specific ones.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    elif weighting_scheme == "none" or weighting_scheme is None:
        weighting = torch.ones_like(sigmas)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


# Parameter groups (6 groups with separate LRs)
def get_anima_param_groups(
    dit,
    base_lr: float,
    self_attn_lr: Optional[float] = None,
    cross_attn_lr: Optional[float] = None,
    mlp_lr: Optional[float] = None,
    mod_lr: Optional[float] = None,
    llm_adapter_lr: Optional[float] = None,
):
    """Create parameter groups for Anima training with separate learning rates.

    Args:
        dit: MiniTrainDIT model
        base_lr: Base learning rate
        self_attn_lr: LR for self-attention layers (None = base_lr, 0 = freeze)
        cross_attn_lr: LR for cross-attention layers
        mlp_lr: LR for MLP layers
        mod_lr: LR for AdaLN modulation layers
        llm_adapter_lr: LR for LLM adapter

    Returns:
        List of parameter group dicts for optimizer
    """
    if self_attn_lr is None:
        self_attn_lr = base_lr
    if cross_attn_lr is None:
        cross_attn_lr = base_lr
    if mlp_lr is None:
        mlp_lr = base_lr
    if mod_lr is None:
        mod_lr = base_lr
    if llm_adapter_lr is None:
        llm_adapter_lr = base_lr

    base_params = []
    self_attn_params = []
    cross_attn_params = []
    mlp_params = []
    mod_params = []
    llm_adapter_params = []

    for name, p in dit.named_parameters():
        # Store original name for debugging
        p.original_name = name

        if 'llm_adapter' in name:
            llm_adapter_params.append(p)
        elif '.self_attn' in name:
            self_attn_params.append(p)
        elif '.cross_attn' in name:
            cross_attn_params.append(p)
        elif '.mlp' in name:
            mlp_params.append(p)
        elif '.adaln_modulation' in name:
            mod_params.append(p)
        else:
            base_params.append(p)

    logger.info(f"Parameter groups:")
    logger.info(f"  base_params: {len(base_params)} (lr={base_lr})")
    logger.info(f"  self_attn_params: {len(self_attn_params)} (lr={self_attn_lr})")
    logger.info(f"  cross_attn_params: {len(cross_attn_params)} (lr={cross_attn_lr})")
    logger.info(f"  mlp_params: {len(mlp_params)} (lr={mlp_lr})")
    logger.info(f"  mod_params: {len(mod_params)} (lr={mod_lr})")
    logger.info(f"  llm_adapter_params: {len(llm_adapter_params)} (lr={llm_adapter_lr})")

    param_groups = []
    for lr, params, name in [
        (base_lr, base_params, "base"),
        (self_attn_lr, self_attn_params, "self_attn"),
        (cross_attn_lr, cross_attn_params, "cross_attn"),
        (mlp_lr, mlp_params, "mlp"),
        (mod_lr, mod_params, "mod"),
        (llm_adapter_lr, llm_adapter_params, "llm_adapter"),
    ]:
        if lr == 0:
            for p in params:
                p.requires_grad_(False)
            logger.info(f"  Frozen {name} params ({len(params)} parameters)")
        elif len(params) > 0:
            param_groups.append({'params': params, 'lr': lr})

    total_trainable = sum(p.numel() for group in param_groups for p in group['params'] if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_trainable:,}")

    return param_groups


# Save functions
def save_anima_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    dit: anima_models.MiniTrainDIT,
):
    """Save Anima model at the end of training."""
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True
        )
        dit_sd = dit.state_dict()
        # Save with 'net.' prefix for ComfyUI compatibility
        anima_utils.save_anima_model(ckpt_file, dit_sd, save_dtype)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


def save_anima_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    dit: anima_models.MiniTrainDIT,
):
    """Save Anima model at epoch end or specific steps."""
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True
        )
        dit_sd = dit.state_dict()
        anima_utils.save_anima_model(ckpt_file, dit_sd, save_dtype)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


# Sampling (Euler discrete for rectified flow)
def do_sample(
    height: int,
    width: int,
    seed: Optional[int],
    dit: anima_models.MiniTrainDIT,
    crossattn_emb: torch.Tensor,
    steps: int,
    dtype: torch.dtype,
    device: torch.device,
    guidance_scale: float = 1.0,
    neg_crossattn_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generate a sample using Euler discrete sampling for rectified flow.

    Args:
        height, width: Output image dimensions
        seed: Random seed (None for random)
        dit: MiniTrainDIT model
        crossattn_emb: Cross-attention embeddings (B, N, D)
        steps: Number of sampling steps
        dtype: Compute dtype
        device: Compute device
        guidance_scale: CFG scale (1.0 = no guidance)
        neg_crossattn_emb: Negative cross-attention embeddings for CFG

    Returns:
        Denoised latents
    """
    # Latent shape: (1, 16, 1, H/8, W/8) for single image
    latent_h = height // 8
    latent_w = width // 8
    latent = torch.zeros(1, 16, 1, latent_h, latent_w, device=device, dtype=dtype)

    # Generate noise
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    noise = torch.randn(
        latent.size(), dtype=torch.float32, generator=generator, device="cpu"
    ).to(dtype).to(device)

    # Timestep schedule: linear from 1.0 to 0.0
    sigmas = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)

    # Start from pure noise
    x = noise.clone()

    # Padding mask (zeros = no padding) — resized in prepare_embedded_sequence to match latent dims
    padding_mask = torch.zeros(1, 1, latent_h, latent_w, dtype=dtype, device=device)

    use_cfg = guidance_scale > 1.0 and neg_crossattn_emb is not None

    for i in tqdm(range(steps), desc="Sampling"):
        sigma = sigmas[i]
        t = sigma.unsqueeze(0)  # (1,)

        dit.prepare_block_swap_before_forward()

        if use_cfg:
            # CFG: concat positive and negative
            x_input = torch.cat([x, x], dim=0)
            t_input = torch.cat([t, t], dim=0)
            crossattn_input = torch.cat([crossattn_emb, neg_crossattn_emb], dim=0)
            padding_input = torch.cat([padding_mask, padding_mask], dim=0)

            model_output = dit(x_input, t_input, crossattn_input, padding_mask=padding_input)
            model_output = model_output.float()

            pos_out, neg_out = model_output.chunk(2)
            model_output = neg_out + guidance_scale * (pos_out - neg_out)
        else:
            model_output = dit(x, t, crossattn_emb, padding_mask=padding_mask)
            model_output = model_output.float()

        # Euler step: x_{t-1} = x_t - (sigma_t - sigma_{t-1}) * model_output
        dt = sigmas[i + 1] - sigma
        x = x + model_output * dt
        x = x.to(dtype)

    dit.prepare_block_swap_before_forward()
    return x


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit,
    vae,
    vae_scale,
    text_encoder,
    tokenize_strategy,
    text_encoding_strategy,
    sample_prompts_te_outputs=None,
    prompt_replacement=None,
):
    """Generate sample images during training.

    This is a simplified sampler for Anima - it generates images using the current model state.
    """
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info(f"Generating sample images at step {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file: {args.sample_prompts}")
        return

    # Unwrap models
    dit = accelerator.unwrap_model(dit)
    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    # Save RNG state
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    with torch.no_grad(), accelerator.autocast():
        for prompt_dict in prompts:
            _sample_image_inference(
                accelerator, args, dit, text_encoder, vae, vae_scale,
                tokenize_strategy, text_encoding_strategy,
                save_dir, prompt_dict, epoch, steps,
                sample_prompts_te_outputs, prompt_replacement,
            )

    # Restore RNG state
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


def _sample_image_inference(
    accelerator, args, dit, text_encoder, vae, vae_scale,
    tokenize_strategy, text_encoding_strategy,
    save_dir, prompt_dict, epoch, steps,
    sample_prompts_te_outputs, prompt_replacement,
):
    """Generate a single sample image."""
    prompt = prompt_dict.get("prompt", "")
    negative_prompt = prompt_dict.get("negative_prompt", "")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # seed all CUDA devices for multi-GPU

    height = max(64, height - height % 16)
    width = max(64, width - width % 16)

    logger.info(f"  prompt: {prompt}, size: {width}x{height}, steps: {sample_steps}, scale: {scale}")

    # Encode prompt
    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            return sample_prompts_te_outputs[prpt]
        if text_encoder is not None:
            tokens = tokenize_strategy.tokenize(prpt)
            encoded = text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)
            return encoded
        return None

    encoded = encode_prompt(prompt)
    if encoded is None:
        logger.warning("Cannot encode prompt, skipping sample")
        return

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = encoded

    # Convert to tensors if numpy
    if isinstance(prompt_embeds, np.ndarray):
        prompt_embeds = torch.from_numpy(prompt_embeds).unsqueeze(0)
        attn_mask = torch.from_numpy(attn_mask).unsqueeze(0)
        t5_input_ids = torch.from_numpy(t5_input_ids).unsqueeze(0)
        t5_attn_mask = torch.from_numpy(t5_attn_mask).unsqueeze(0)

    prompt_embeds = prompt_embeds.to(accelerator.device, dtype=dit.t_embedding_norm.weight.dtype)
    attn_mask = attn_mask.to(accelerator.device)
    t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
    t5_attn_mask = t5_attn_mask.to(accelerator.device)

    # Process through LLM adapter if available
    if dit.use_llm_adapter and hasattr(dit, 'llm_adapter'):
        crossattn_emb = dit.llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
        crossattn_emb[~t5_attn_mask.bool()] = 0
    else:
        crossattn_emb = prompt_embeds

    # Encode negative prompt for CFG
    neg_crossattn_emb = None
    if scale > 1.0 and negative_prompt is not None:
        neg_encoded = encode_prompt(negative_prompt)
        if neg_encoded is not None:
            neg_pe, neg_am, neg_t5_ids, neg_t5_am = neg_encoded
            if isinstance(neg_pe, np.ndarray):
                neg_pe = torch.from_numpy(neg_pe).unsqueeze(0)
                neg_am = torch.from_numpy(neg_am).unsqueeze(0)
                neg_t5_ids = torch.from_numpy(neg_t5_ids).unsqueeze(0)
                neg_t5_am = torch.from_numpy(neg_t5_am).unsqueeze(0)

            neg_pe = neg_pe.to(accelerator.device, dtype=dit.t_embedding_norm.weight.dtype)
            neg_am = neg_am.to(accelerator.device)
            neg_t5_ids = neg_t5_ids.to(accelerator.device, dtype=torch.long)
            neg_t5_am = neg_t5_am.to(accelerator.device)

            if dit.use_llm_adapter and hasattr(dit, 'llm_adapter'):
                neg_crossattn_emb = dit.llm_adapter(
                    source_hidden_states=neg_pe,
                    target_input_ids=neg_t5_ids,
                    target_attention_mask=neg_t5_am,
                    source_attention_mask=neg_am,
                )
                neg_crossattn_emb[~neg_t5_am.bool()] = 0
            else:
                neg_crossattn_emb = neg_pe

    # Generate sample
    clean_memory_on_device(accelerator.device)
    latents = do_sample(
        height, width, seed, dit, crossattn_emb,
        sample_steps, dit.t_embedding_norm.weight.dtype,
        accelerator.device, scale, neg_crossattn_emb,
    )

    # Decode latents
    clean_memory_on_device(accelerator.device)
    org_vae_device = next(vae.parameters()).device
    vae.to(accelerator.device)
    decoded = vae.decode(latents.to(next(vae.parameters()).device, dtype=next(vae.parameters()).dtype), vae_scale)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    # Convert to image
    image = decoded.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    # Remove temporal dim if present
    if image.ndim == 4:
        image = image[:, 0, :, :]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)

    image = Image.fromarray(decoded_np)

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
