# Minimal inpainting inference script for SD1.5 and SDXL inpainting models.
# インペインティングモデル（SD1.5およびSDXL）を使った最低限の推論スクリプト。
#
# Loads a 9-channel inpainting UNet checkpoint, encodes a source image and mask
# via VAE, then runs a standard denoising loop with the 9-channel UNet input:
#   [noisy_latents(4ch), mask(1ch), masked_image_latents(4ch)]
#
# Usage / 使い方:
#   # SD1.5 inpainting
#   python inpainting_minimal_inference.py \
#       --ckpt_path sd-v1-5-inpainting.ckpt \
#       --image input.png \
#       --mask mask.png \
#       --prompt "a yawning cat"
#
#   # SDXL inpainting
#   python inpainting_minimal_inference.py \
#       --ckpt_path sd_xl_base_1.0_inpainting.safetensors \
#       --sdxl \
#       --image input.png \
#       --mask mask.png \
#       --prompt "a yawning cat"
#
#   # Generate a random procedural mask instead of supplying one
#   python inpainting_minimal_inference.py \
#       --ckpt_path sd-v1-5-inpainting.ckpt \
#       --image input.png \
#       --prompt "a yawning cat"

import argparse
import datetime
import math
import os
import random

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from library.device_utils import get_preferred_device, init_ipex

init_ipex()

from library import model_util, sdxl_model_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.012
SCHEDULER_TIMESTEPS = 1000
SCHEDULER_SCHEDULE = "scaled_linear"

SD15_VAE_SCALE_FACTOR = 0.18215
SDXL_VAE_SCALE_FACTOR = sdxl_model_util.VAE_SCALE_FACTOR  # 0.13025


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def _get_timestep_embedding(x, outdim):
    b = x.shape[0]
    x = torch.flatten(x)
    emb = _timestep_embedding(x, outdim)
    return torch.reshape(emb, (b, -1))


def load_image(path: str, width: int, height: int) -> torch.Tensor:
    """Load an RGB image and return a [-1, 1] float32 tensor of shape (1, 3, H, W)."""
    img = Image.open(path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def load_mask(path: str, width: int, height: int) -> torch.Tensor:
    """
    Load a mask image and return a binary float32 tensor of shape (1, 1, H, W).
    White pixels (>= 128) indicate the region to regenerate.
    """
    mask = Image.open(path).convert("L").resize((width, height), Image.NEAREST)
    arr = np.array(mask).astype(np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def make_default_mask(width: int, height: int, seed: int = None) -> torch.Tensor:
    """
    Generate a wobbly-ellipse mask — a single connected organic region that is
    well-suited for inpainting sampling and inference.  (The full random_mask
    used during training produces fragmented cloud patterns that can confuse the
    sampler when large proportions of the image are masked.)
    """
    from library.mask_generator import wobbly_ellipse_mask
    pil_mask = wobbly_ellipse_mask(width, height, seed=seed)
    arr = (np.array(pil_mask).astype(np.float32) / 255.0 >= 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def encode_image(vae, image_t: torch.Tensor, vae_scale_factor: float, device, dtype):
    """Encode an image tensor to latents."""
    image_t = image_t.to(device=device, dtype=dtype)
    with torch.no_grad():
        latents = vae.encode(image_t).latent_dist.sample()
    return latents * vae_scale_factor


def downsample_mask(mask_t: torch.Tensor, latent_h: int, latent_w: int, device, dtype):
    """Downsample a (1,1,H,W) mask to latent spatial size."""
    mask_latent = torch.nn.functional.interpolate(
        mask_t, size=(latent_h, latent_w), mode="nearest"
    )
    return mask_latent.to(device=device, dtype=dtype)


# -------------------------------------------------------------------------
# SD1.5 text encoding
# -------------------------------------------------------------------------

def encode_text_sd15(tokenizer, text_encoder, text: str, device, dtype):
    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )["input_ids"].to(device)
    with torch.no_grad():
        return text_encoder(tokens)[0].to(dtype)


# -------------------------------------------------------------------------
# SDXL text encoding (mirrors sdxl_minimal_inference.py)
# -------------------------------------------------------------------------

def encode_text_sdxl(tokenizer1, text_model1, tokenizer2, text_model2, text: str, device, dtype,
                     original_size, crop_top_left, target_size):
    # encoder 1
    tokens1 = tokenizer1(
        text, truncation=True, padding="max_length",
        max_length=tokenizer1.model_max_length, return_tensors="pt",
    )["input_ids"].to(device)
    with torch.no_grad():
        enc1 = text_model1(tokens1, output_hidden_states=True, return_dict=True)
        emb1 = enc1["hidden_states"][11].to(dtype)

    # encoder 2
    tokens2 = tokenizer2(
        text, truncation=True, padding="max_length",
        max_length=tokenizer2.model_max_length, return_tensors="pt",
    )["input_ids"].to(device)
    with torch.no_grad():
        enc2 = text_model2(tokens2, output_hidden_states=True, return_dict=True)
        emb2 = enc2["hidden_states"][-2].to(dtype)
        pool2 = enc2["text_embeds"].to(dtype)

    text_emb = torch.cat([emb1, emb2], dim=2)

    # vector embedding
    oh, ow = original_size
    ct, cl = crop_top_left
    th, tw = target_size
    e1 = _get_timestep_embedding(torch.FloatTensor([[oh, ow]]), 256)
    e2 = _get_timestep_embedding(torch.FloatTensor([[ct, cl]]), 256)
    e3 = _get_timestep_embedding(torch.FloatTensor([[th, tw]]), 256)
    vec = torch.cat([pool2, e1.to(device, dtype), e2.to(device, dtype), e3.to(device, dtype)], dim=1)

    return text_emb, vec


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Minimal inpainting inference for SD1.5 and SDXL inpainting models")
    parser.add_argument("--ckpt_path",        type=str, required=True,  help="Path to inpainting model checkpoint (.ckpt or .safetensors)")
    parser.add_argument("--image",            type=str, required=True,  help="Source image path (PNG/JPG)")
    parser.add_argument("--mask",             type=str, default=None,   help="Mask image path — white = region to regenerate. Omit to use a random procedural mask.")
    parser.add_argument("--prompt",           type=str, default="",     help="Positive prompt")
    parser.add_argument("--negative_prompt",  type=str, default="",     help="Negative prompt")
    parser.add_argument("--width",            type=int, default=512,    help="Output width (must be multiple of 64)")
    parser.add_argument("--height",           type=int, default=512,    help="Output height (must be multiple of 64)")
    parser.add_argument("--steps",            type=int, default=30,     help="Number of denoising steps")
    parser.add_argument("--guidance_scale",   type=float, default=7.5,  help="Classifier-free guidance scale")
    parser.add_argument("--seed",             type=int, default=None,   help="Random seed")
    parser.add_argument("--sdxl",            action="store_true",       help="Use SDXL model (default: SD1.5)")
    parser.add_argument("--output_dir",       type=str, default=".",    help="Directory to save output images")
    parser.add_argument(
        "--lora_weights",
        type=str, nargs="*", default=[],
        help="LoRA weights to merge, each as 'path;multiplier'",
    )
    args = parser.parse_args()

    # Round dimensions to multiples of 64
    args.width  = max(64, args.width  - args.width  % 64)
    args.height = max(64, args.height - args.height % 64)

    DEVICE = get_preferred_device()
    DTYPE  = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    if args.sdxl:
        logger.info("Loading SDXL inpainting model...")
        text_model1, text_model2, vae, unet, _, _ = sdxl_model_util.load_models_from_sdxl_checkpoint(
            sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, args.ckpt_path, "cpu"
        )
        VAE_SCALE = SDXL_VAE_SCALE_FACTOR

        tokenizer1 = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        tokenizer2 = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

        text_model1.to(DEVICE, dtype=DTYPE).eval()
        text_model2.to(DEVICE, dtype=DTYPE).eval()
    else:
        logger.info("Loading SD1.5 inpainting model...")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
            False, args.ckpt_path, "cpu"
        )
        VAE_SCALE = SD15_VAE_SCALE_FACTOR

        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder.to(DEVICE, dtype=DTYPE).eval()

    # fp16 VAE is unstable; use float32
    vae_dtype = torch.float32 if DTYPE == torch.float16 else DTYPE
    vae.to(DEVICE, dtype=vae_dtype).eval()

    unet.to(DEVICE, dtype=DTYPE).eval()
    unet.set_use_memory_efficient_attention(True, False)

    # LoRA
    import networks.lora as lora
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0
        text_encoders = [text_model1, text_model2] if args.sdxl else [text_encoder]
        lora_model, weights_sd = lora.create_network_from_weights(
            multiplier, weights_file, vae, text_encoders, unet, None, True
        )
        lora_model.merge_to(text_encoders, unet, weights_sd, DTYPE, DEVICE)

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------
    scheduler = EulerDiscreteScheduler(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDULER_SCHEDULE,
    )

    # ------------------------------------------------------------------
    # Prepare image and mask
    # ------------------------------------------------------------------
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    image_t = load_image(args.image, args.width, args.height)

    if args.mask is not None:
        mask_t = load_mask(args.mask, args.width, args.height)
        logger.info(f"Using mask: {args.mask}")
    else:
        mask_t = make_default_mask(args.width, args.height, seed=args.seed)
        logger.info("Using wobbly-ellipse mask")

    # Encode masked image: source × (1 − mask)
    masked_image_t = image_t * (1.0 - mask_t)

    latent_h = args.height // 8
    latent_w = args.width  // 8

    image_latents  = encode_image(vae, image_t,        VAE_SCALE, DEVICE, vae_dtype)
    masked_latents = encode_image(vae, masked_image_t, VAE_SCALE, DEVICE, vae_dtype)
    mask_latent    = downsample_mask(mask_t, latent_h, latent_w, DEVICE, DTYPE)

    # Initial noise
    latents = torch.randn(
        (1, 4, latent_h, latent_w), device="cpu", dtype=torch.float32
    ).to(DEVICE, dtype=DTYPE) * scheduler.init_noise_sigma

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------
    if args.sdxl:
        size_args = dict(
            original_size=(args.height, args.width),
            crop_top_left=(0, 0),
            target_size=(args.height, args.width),
        )
        c_emb,  c_vec  = encode_text_sdxl(tokenizer1, text_model1, tokenizer2, text_model2,
                                           args.prompt,          DEVICE, DTYPE, **size_args)
        uc_emb, uc_vec = encode_text_sdxl(tokenizer1, text_model1, tokenizer2, text_model2,
                                           args.negative_prompt, DEVICE, DTYPE, **size_args)
        cond_args   = (c_emb,  c_vec)
        uncond_args = (uc_emb, uc_vec)

        def unet_call(inp, t, emb, vec):
            return unet(inp, t, emb, vec)
    else:
        c_emb  = encode_text_sd15(tokenizer, text_encoder, args.prompt,          DEVICE, DTYPE)
        uc_emb = encode_text_sd15(tokenizer, text_encoder, args.negative_prompt, DEVICE, DTYPE)
        cond_args   = (c_emb,)
        uncond_args = (uc_emb,)

        def unet_call(inp, t, emb):
            return unet(inp, t, emb).sample

    # ------------------------------------------------------------------
    # Denoising loop
    # ------------------------------------------------------------------
    logger.info(f"Denoising ({args.steps} steps, guidance={args.guidance_scale})...")

    scheduler.set_timesteps(args.steps, DEVICE)
    timesteps = scheduler.timesteps.to(DEVICE)

    # Pre-generate fixed noise used to re-noise the original image latents each
    # step so that unmasked regions stay anchored to the original image.
    # After each scheduler step we composite:
    #   latents = mask * denoised  +  (1 - mask) * (original + noise * sigma_next)
    # At the final step sigma_next = 0, so the unmasked area is exactly the original.
    compositing_noise = torch.randn_like(image_latents).to(DTYPE)
    image_latents_dtype = image_latents.to(DTYPE)
    mask_latent_dtype   = mask_latent.to(DTYPE)
    masked_latents_dtype = masked_latents.to(DTYPE)

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            latent_model_input = scheduler.scale_model_input(latents, t)

            unet_input = torch.cat([latent_model_input, mask_latent_dtype, masked_latents_dtype], dim=1)

            if args.sdxl:
                noise_pred_uncond = unet(unet_input, t, uc_emb, uc_vec)
                noise_pred_cond   = unet(unet_input, t, c_emb,  c_vec)
            else:
                noise_pred_uncond = unet(unet_input, t, uc_emb).sample
                noise_pred_cond   = unet(unet_input, t, c_emb).sample

            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_cond - noise_pred_uncond)
            latents = scheduler.step(noise_pred, t, latents).prev_sample

            # Composite: keep unmasked regions anchored to the original.
            # sigma_next is the noise level after this step (0.0 on the last step).
            sigma_next = scheduler.sigmas[i + 1].to(device=DEVICE, dtype=DTYPE)
            noised_original = image_latents_dtype + compositing_noise * sigma_next
            latents = mask_latent_dtype * latents + (1.0 - mask_latent_dtype) * noised_original

        # Decode
        latents = latents.to(vae_dtype) / VAE_SCALE
        image_out = vae.decode(latents).sample
        image_out = (image_out / 2 + 0.5).clamp(0, 1)

    image_np = image_out.cpu().permute(0, 2, 3, 1).float().numpy()
    image_np = (image_np * 255).round().astype("uint8")
    pil_images = [Image.fromarray(im) for im in image_np]

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    for i, img in enumerate(pil_images):
        out_path = os.path.join(args.output_dir, f"inpaint_{timestamp}_{i:03d}.png")
        img.save(out_path)
        logger.info(f"Saved: {out_path}")

    logger.info("Done!")
