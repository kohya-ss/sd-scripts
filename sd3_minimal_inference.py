# Minimum Inference Code for SD3

import argparse
import datetime
import math
import os
import random
from typing import Optional, Tuple
import numpy as np

import torch
from safetensors.torch import safe_open, load_file
from tqdm import tqdm
from PIL import Image

from library.device_utils import init_ipex, get_preferred_device

init_ipex()

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sd3_models, sd3_utils


def get_noise(seed, latent):
    generator = torch.manual_seed(seed)
    return torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu").to(latent.dtype)


def get_sigmas(sampling: sd3_utils.ModelSamplingDiscreteFlow, steps):
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(sampling.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def max_denoise(model_sampling, sigmas):
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def do_sample(
    height: int,
    width: int,
    initial_latent: Optional[torch.Tensor],
    seed: int,
    cond: Tuple[torch.Tensor, torch.Tensor],
    neg_cond: Tuple[torch.Tensor, torch.Tensor],
    mmdit: sd3_models.MMDiT,
    steps: int,
    guidance_scale: float,
    dtype: torch.dtype,
    device: str,
):
    if initial_latent is None:
        latent = torch.ones(1, 16, height // 8, width // 8, device=device) * 0.0609
    else:
        latent = initial_latent

    latent = latent.to(dtype).to(device)

    noise = get_noise(seed, latent).to(device)

    model_sampling = sd3_utils.ModelSamplingDiscreteFlow()

    sigmas = get_sigmas(model_sampling, steps).to(device)
    # sigmas = sigmas[int(steps * (1 - denoise)) :] # do not support i2i

    # conditioning = fix_cond(conditioning)
    # neg_cond = fix_cond(neg_cond)
    # extra_args = {"cond": cond, "uncond": neg_cond, "cond_scale": guidance_scale}

    noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(model_sampling, sigmas))

    c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device).to(dtype)
    y = torch.cat([cond[1], neg_cond[1]]).to(device).to(dtype)

    x = noise_scaled.to(device).to(dtype)
    # print(x.shape)

    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1)):
            sigma_hat = sigmas[i]

            timestep = model_sampling.timestep(sigma_hat).float()
            timestep = torch.FloatTensor([timestep, timestep]).to(device)

            x_c_nc = torch.cat([x, x], dim=0)
            # print(x_c_nc.shape, timestep.shape, c_crossattn.shape, y.shape)

            model_output = mmdit(x_c_nc, timestep, context=c_crossattn, y=y)
            model_output = model_output.float()
            batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)

            pos_out, neg_out = batched.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * guidance_scale
            # print(denoised.shape)

            # d = to_d(x, sigma_hat, denoised)
            dims_to_append = x.ndim - sigma_hat.ndim
            sigma_hat_dims = sigma_hat[(...,) + (None,) * dims_to_append]
            # print(dims_to_append, x.shape, sigma_hat.shape, denoised.shape, sigma_hat_dims.shape)
            """Converts a denoiser output to a Karras ODE derivative."""
            d = (x - denoised) / sigma_hat_dims

            dt = sigmas[i + 1] - sigma_hat

            # Euler method
            x = x + d * dt
            x = x.to(dtype)

    latent = x
    scale_factor = 1.5305
    shift_factor = 0.0609
    # def process_out(self, latent):
    #     return (latent / self.scale_factor) + self.shift_factor
    latent = (latent / scale_factor) + shift_factor
    return latent


if __name__ == "__main__":
    target_height = 1024
    target_width = 1024

    # steps = 50  # 28  # 50
    guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_g", type=str, required=False)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    # parser.add_argument("--prompt2", type=str, default=None)  # do not support different prompts for text encoders
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--do_not_use_t5xxl", action="store_true")
    parser.add_argument("--attn_mode", type=str, default="torch", help="torch (SDPA) or xformers. default: torch")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    # parser.add_argument(
    #     "--lora_weights",
    #     type=str,
    #     nargs="*",
    #     default=[],
    #     help="LoRA weights, only supports networks.lora, each argument is a `path;multiplier` (semi-colon separated)",
    # )
    # parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    steps = args.steps

    sd3_dtype = torch.float32
    if args.fp16:
        sd3_dtype = torch.float16
    elif args.bf16:
        sd3_dtype = torch.bfloat16

    # TODO test with separated safetenors files for each model

    # load state dict
    logger.info(f"Loading SD3 models from {args.ckpt_path}...")
    state_dict = load_file(args.ckpt_path)

    if "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
        # found clip_g: remove prefix "text_encoders.clip_g."
        logger.info("clip_g is included in the checkpoint")
        clip_g_sd = {}
        prefix = "text_encoders.clip_g."
        for k, v in list(state_dict.items()):
            if k.startswith(prefix):
                clip_g_sd[k[len(prefix) :]] = state_dict.pop(k)
    else:
        logger.info(f"Lodaing clip_g from {args.clip_g}...")
        clip_g_sd = load_file(args.clip_g)
        for key in list(clip_g_sd.keys()):
            clip_g_sd["transformer." + key] = clip_g_sd.pop(key)

    if "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
        # found clip_l: remove prefix "text_encoders.clip_l."
        logger.info("clip_l is included in the checkpoint")
        clip_l_sd = {}
        prefix = "text_encoders.clip_l."
        for k, v in list(state_dict.items()):
            if k.startswith(prefix):
                clip_l_sd[k[len(prefix) :]] = state_dict.pop(k)
    else:
        logger.info(f"Lodaing clip_l from {args.clip_l}...")
        clip_l_sd = load_file(args.clip_l)
        for key in list(clip_l_sd.keys()):
            clip_l_sd["transformer." + key] = clip_l_sd.pop(key)

    if "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight" in state_dict:
        # found t5xxl: remove prefix "text_encoders.t5xxl."
        logger.info("t5xxl is included in the checkpoint")
        if not args.do_not_use_t5xxl:
            t5xxl_sd = {}
            prefix = "text_encoders.t5xxl."
            for k, v in list(state_dict.items()):
                if k.startswith(prefix):
                    t5xxl_sd[k[len(prefix) :]] = state_dict.pop(k)
        else:
            logger.info("but not used")
            for key in list(state_dict.keys()):
                if key.startswith("text_encoders.t5xxl."):
                    state_dict.pop(key)
            t5xxl_sd = None
    elif args.t5xxl:
        assert not args.do_not_use_t5xxl, "t5xxl is not used but specified"
        logger.info(f"Lodaing t5xxl from {args.t5xxl}...")
        t5xxl_sd = load_file(args.t5xxl)
        for key in list(t5xxl_sd.keys()):
            t5xxl_sd["transformer." + key] = t5xxl_sd.pop(key)
    else:
        logger.info("t5xxl is not used")
        t5xxl_sd = None

    use_t5xxl = t5xxl_sd is not None

    # MMDiT and VAE
    vae_sd = {}
    vae_prefix = "first_stage_model."
    mmdit_prefix = "model.diffusion_model."
    for k, v in list(state_dict.items()):
        if k.startswith(vae_prefix):
            vae_sd[k[len(vae_prefix) :]] = state_dict.pop(k)
        elif k.startswith(mmdit_prefix):
            state_dict[k[len(mmdit_prefix) :]] = state_dict.pop(k)

    # load tokenizers
    logger.info("Loading tokenizers...")
    tokenizer = sd3_models.SD3Tokenizer(use_t5xxl)  # combined tokenizer

    # load models
    # logger.info("Create MMDiT from SD3 checkpoint...")
    # mmdit = sd3_utils.create_mmdit_from_sd3_checkpoint(state_dict)
    logger.info("Create MMDiT")
    mmdit = sd3_models.create_mmdit_sd3_medium_configs(args.attn_mode)

    logger.info("Loading state dict...")
    info = mmdit.load_state_dict(state_dict)
    logger.info(f"Loaded MMDiT: {info}")

    logger.info(f"Move MMDiT to {device} and {sd3_dtype}...")
    mmdit.to(device, dtype=sd3_dtype)
    mmdit.eval()

    # load VAE
    logger.info("Create VAE")
    vae = sd3_models.SDVAE()
    logger.info("Loading state dict...")
    info = vae.load_state_dict(vae_sd)
    logger.info(f"Loaded VAE: {info}")

    logger.info(f"Move VAE to {device} and {sd3_dtype}...")
    vae.to(device, dtype=sd3_dtype)
    vae.eval()

    # load text encoders
    logger.info("Create clip_l")
    clip_l = sd3_models.create_clip_l(device, sd3_dtype, clip_l_sd)

    logger.info("Loading state dict...")
    info = clip_l.load_state_dict(clip_l_sd)
    logger.info(f"Loaded clip_l: {info}")

    logger.info(f"Move clip_l to {device} and {sd3_dtype}...")
    clip_l.to(device, dtype=sd3_dtype)
    clip_l.eval()
    logger.info(f"Set attn_mode to {args.attn_mode}...")
    clip_l.set_attn_mode(args.attn_mode)

    logger.info("Create clip_g")
    clip_g = sd3_models.create_clip_g(device, sd3_dtype, clip_g_sd)

    logger.info("Loading state dict...")
    info = clip_g.load_state_dict(clip_g_sd)
    logger.info(f"Loaded clip_g: {info}")

    logger.info(f"Move clip_g to {device} and {sd3_dtype}...")
    clip_g.to(device, dtype=sd3_dtype)
    clip_g.eval()
    logger.info(f"Set attn_mode to {args.attn_mode}...")
    clip_g.set_attn_mode(args.attn_mode)

    if use_t5xxl:
        logger.info("Create t5xxl")
        t5xxl = sd3_models.create_t5xxl(device, sd3_dtype, t5xxl_sd)

        logger.info("Loading state dict...")
        info = t5xxl.load_state_dict(t5xxl_sd)
        logger.info(f"Loaded t5xxl: {info}")

        logger.info(f"Move t5xxl to {device} and {sd3_dtype}...")
        t5xxl.to(device, dtype=sd3_dtype)
        # t5xxl.to("cpu", dtype=torch.float32) # run on CPU
        t5xxl.eval()
        logger.info(f"Set attn_mode to {args.attn_mode}...")
        t5xxl.set_attn_mode(args.attn_mode)
    else:
        t5xxl = None

    # prepare embeddings
    logger.info("Encoding prompts...")
    # embeds, pooled_embed
    lg_out, t5_out, pooled = sd3_utils.get_cond(args.prompt, tokenizer, clip_l, clip_g, t5xxl)
    cond = torch.cat([lg_out, t5_out], dim=-2), pooled

    lg_out, t5_out, pooled = sd3_utils.get_cond(args.negative_prompt, tokenizer, clip_l, clip_g, t5xxl)
    neg_cond = torch.cat([lg_out, t5_out], dim=-2), pooled

    # generate image
    logger.info("Generating image...")
    latent_sampled = do_sample(
        target_height, target_width, None, seed, cond, neg_cond, mmdit, steps, guidance_scale, sd3_dtype, device
    )

    # latent to image
    with torch.no_grad():
        image = vae.decode(latent_sampled)
    image = image.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)
    out_image = Image.fromarray(decoded_np)

    # save image
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    out_image.save(output_path)

    logger.info(f"Saved image to {output_path}")
