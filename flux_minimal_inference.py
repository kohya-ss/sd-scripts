# Minimum Inference Code for FLUX

import argparse
import datetime
import math
import os
import random
from typing import Callable, Optional, Tuple
import einops
import numpy as np

import torch
from safetensors.torch import safe_open, load_file
from tqdm import tqdm
from PIL import Image
import accelerate

from library import device_utils
from library.device_utils import init_ipex, get_preferred_device

init_ipex()


from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

import networks.lora_flux as lora_flux
from library import flux_models, flux_utils, sd3_utils, strategy_flux


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    txt: torch.Tensor,
    txt_ids: torch.Tensor,
    vec: torch.Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec, timesteps=t_vec, guidance=guidance_vec)

        img = img + (t_prev - t_curr) * pred

    return img


def do_sample(
    accelerator: Optional[accelerate.Accelerator],
    model: flux_models.Flux,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    l_pooled: torch.Tensor,
    t5_out: torch.Tensor,
    txt_ids: torch.Tensor,
    num_steps: int,
    guidance: float,
    is_schnell: bool,
    device: torch.device,
    flux_dtype: torch.dtype,
):
    timesteps = get_schedule(num_steps, img.shape[1], shift=not is_schnell)

    # denoise initial noise
    if accelerator:
        with accelerator.autocast(), torch.no_grad():
            x = denoise(model, img, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=guidance)
    else:
        with torch.autocast(device_type=device.type, dtype=flux_dtype), torch.no_grad():
            x = denoise(model, img, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=guidance)

    return x


def generate_image(
    model,
    clip_l,
    t5xxl,
    ae,
    prompt: str,
    seed: Optional[int],
    image_width: int,
    image_height: int,
    steps: Optional[int],
    guidance: float,
):
    # make first noise with packed shape
    # original: b,16,2*h//16,2*w//16, packed: b,h//16*w//16,16*2*2
    packed_latent_height, packed_latent_width = math.ceil(image_height / 16), math.ceil(image_width / 16)
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )

    # prepare img and img ids

    # this is needed only for img2img
    # img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # if img.shape[0] == 1 and bs > 1:
    #     img = repeat(img, "1 ... -> bs ...", bs=bs)

    # txt2img only needs img_ids
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width)

    # prepare embeddings
    logger.info("Encoding prompts...")
    tokens_and_masks = tokenize_strategy.tokenize(prompt)
    clip_l = clip_l.to(device)
    t5xxl = t5xxl.to(device)
    with torch.no_grad():
        if is_fp8(clip_l_dtype) or is_fp8(t5xxl_dtype):
            clip_l.to(clip_l_dtype)
            t5xxl.to(t5xxl_dtype)
            with accelerator.autocast():
                _, t5_out, txt_ids = encoding_strategy.encode_tokens(
                    tokenize_strategy, [clip_l, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                )
        else:
            with torch.autocast(device_type=device.type, dtype=clip_l_dtype):
                l_pooled, _, _ = encoding_strategy.encode_tokens(tokenize_strategy, [clip_l, None], tokens_and_masks)
            with torch.autocast(device_type=device.type, dtype=t5xxl_dtype):
                _, t5_out, txt_ids = encoding_strategy.encode_tokens(
                    tokenize_strategy, [None, t5xxl], tokens_and_masks, args.apply_t5_attn_mask
                )

    # NaN check
    if torch.isnan(l_pooled).any():
        raise ValueError("NaN in l_pooled")
    if torch.isnan(t5_out).any():
        raise ValueError("NaN in t5_out")

    if args.offload:
        clip_l = clip_l.cpu()
        t5xxl = t5xxl.cpu()
    # del clip_l, t5xxl
    device_utils.clean_memory()

    # generate image
    logger.info("Generating image...")
    model = model.to(device)
    if steps is None:
        steps = 4 if is_schnell else 50

    img_ids = img_ids.to(device)
    x = do_sample(
        accelerator, model, noise, img_ids, l_pooled, t5_out, txt_ids, steps, guidance_scale, is_schnell, device, flux_dtype
    )
    if args.offload:
        model = model.cpu()
    # del model
    device_utils.clean_memory()

    # unpack
    x = x.float()
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)

    # decode
    logger.info("Decoding image...")
    ae = ae.to(device)
    with torch.no_grad():
        if is_fp8(ae_dtype):
            with accelerator.autocast():
                x = ae.decode(x)
        else:
            with torch.autocast(device_type=device.type, dtype=ae_dtype):
                x = ae.decode(x)
    if args.offload:
        ae = ae.cpu()

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    img = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # save image
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    img.save(output_path)

    logger.info(f"Saved image to {output_path}")


if __name__ == "__main__":
    target_height = 768  # 1024
    target_width = 1360  # 1024

    # steps = 50  # 28  # 50
    # guidance_scale = 5
    # seed = 1  # None  # 1

    device = get_preferred_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--clip_l", type=str, required=False)
    parser.add_argument("--t5xxl", type=str, required=False)
    parser.add_argument("--ae", type=str, required=False)
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="base dtype")
    parser.add_argument("--clip_l_dtype", type=str, default=None, help="dtype for clip_l")
    parser.add_argument("--ae_dtype", type=str, default=None, help="dtype for ae")
    parser.add_argument("--t5xxl_dtype", type=str, default=None, help="dtype for t5xxl")
    parser.add_argument("--flux_dtype", type=str, default=None, help="dtype for flux")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None, help="Number of steps. Default is 4 for schnell, 50 for dev")
    parser.add_argument("--guidance", type=float, default=3.5)
    parser.add_argument("--offload", action="store_true", help="Offload to CPU")
    parser.add_argument(
        "--lora_weights",
        type=str,
        nargs="*",
        default=[],
        help="LoRA weights, only supports networks.lora_flux, each argument is a `path;multiplier` (semi-colon separated)",
    )
    parser.add_argument("--width", type=int, default=target_width)
    parser.add_argument("--height", type=int, default=target_height)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    seed = args.seed
    steps = args.steps
    guidance_scale = args.guidance

    name = "schnell" if "schnell" in args.ckpt_path else "dev"  # TODO change this to a more robust way
    is_schnell = name == "schnell"

    def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
        if s is None:
            return default_dtype
        if s in ["bf16", "bfloat16"]:
            return torch.bfloat16
        elif s in ["fp16", "float16"]:
            return torch.float16
        elif s in ["fp32", "float32"]:
            return torch.float32
        elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
            return torch.float8_e4m3fn
        elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
            return torch.float8_e4m3fnuz
        elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
            return torch.float8_e5m2
        elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
            return torch.float8_e5m2fnuz
        elif s in ["fp8", "float8"]:
            return torch.float8_e4m3fn  # default fp8
        else:
            raise ValueError(f"Unsupported dtype: {s}")

    def is_fp8(dt):
        return dt in [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]

    dtype = str_to_dtype(args.dtype)
    clip_l_dtype = str_to_dtype(args.clip_l_dtype, dtype)
    t5xxl_dtype = str_to_dtype(args.t5xxl_dtype, dtype)
    ae_dtype = str_to_dtype(args.ae_dtype, dtype)
    flux_dtype = str_to_dtype(args.flux_dtype, dtype)

    logger.info(f"Dtypes for clip_l, t5xxl, ae, flux: {clip_l_dtype}, {t5xxl_dtype}, {ae_dtype}, {flux_dtype}")

    loading_device = "cpu" if args.offload else device

    use_fp8 = [is_fp8(d) for d in [dtype, clip_l_dtype, t5xxl_dtype, ae_dtype, flux_dtype]]
    if any(use_fp8):
        accelerator = accelerate.Accelerator(mixed_precision="bf16")
    else:
        accelerator = None

    # load clip_l
    logger.info(f"Loading clip_l from {args.clip_l}...")
    clip_l = flux_utils.load_clip_l(args.clip_l, clip_l_dtype, loading_device)
    clip_l.eval()

    logger.info(f"Loading t5xxl from {args.t5xxl}...")
    t5xxl = flux_utils.load_t5xxl(args.t5xxl, t5xxl_dtype, loading_device)
    t5xxl.eval()

    if is_fp8(clip_l_dtype):
        clip_l = accelerator.prepare(clip_l)
    if is_fp8(t5xxl_dtype):
        t5xxl = accelerator.prepare(t5xxl)

    t5xxl_max_length = 256 if is_schnell else 512
    tokenize_strategy = strategy_flux.FluxTokenizeStrategy(t5xxl_max_length)
    encoding_strategy = strategy_flux.FluxTextEncodingStrategy()

    # DiT
    model = flux_utils.load_flow_model(name, args.ckpt_path, flux_dtype, loading_device)
    model.eval()
    logger.info(f"Casting model to {flux_dtype}")
    model.to(flux_dtype)  # make sure model is dtype
    if is_fp8(flux_dtype):
        model = accelerator.prepare(model)

    # AE
    ae = flux_utils.load_ae(name, args.ae, ae_dtype, loading_device)
    ae.eval()
    if is_fp8(ae_dtype):
        ae = accelerator.prepare(ae)

    # LoRA
    for weights_file in args.lora_weights:
        if ";" in weights_file:
            weights_file, multiplier = weights_file.split(";")
            multiplier = float(multiplier)
        else:
            multiplier = 1.0

        lora_model, weights_sd = lora_flux.create_network_from_weights(
            multiplier, weights_file, ae, [clip_l, t5xxl], model, None, True
        )
        lora_model.merge_to([clip_l, t5xxl], model, weights_sd)

    if not args.interactive:
        generate_image(model, clip_l, t5xxl, ae, args.prompt, args.seed, args.width, args.height, args.steps, args.guidance)
    else:
        # loop for interactive
        width = target_width
        height = target_height
        steps = None
        guidance = args.guidance

        while True:
            print("Enter prompt (empty to exit). Options: --w <width> --h <height> --s <steps> --d <seed> --g <guidance>")
            prompt = input()
            if prompt == "":
                break

            # parse options
            options = prompt.split("--")
            prompt = options[0].strip()
            seed = None
            for opt in options[1:]:
                opt = opt.strip()
                if opt.startswith("w"):
                    width = int(opt[1:].strip())
                elif opt.startswith("h"):
                    height = int(opt[1:].strip())
                elif opt.startswith("s"):
                    steps = int(opt[1:].strip())
                elif opt.startswith("d"):
                    seed = int(opt[1:].strip())
                elif opt.startswith("g"):
                    guidance = float(opt[1:].strip())

            generate_image(model, clip_l, t5xxl, ae, prompt, seed, width, height, steps, guidance)

    logger.info("Done!")
