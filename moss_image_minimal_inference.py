# Minimal inference script for MOSS-Image
#
# Example:
#   python moss_image_minimal_inference.py --dit path/to/diffusion_pytorch_model-00001-of-00003.safetensors \
#       --text_encoder path/to/model-00001-of-00004.safetensors --vae path/to/flux2-vae.safetensors \
#       --prompt "A cat" --image_size 1024 1024 --seed 42 --save_path output

import argparse
import copy
import datetime
import gc
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library import flux2_vae, moss_image_models, moss_image_train_utils, moss_image_utils
from library.device_utils import clean_memory_on_device, synchronize_device
from library.custom_offloading_utils import BlockSwapConfig
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSS-Image inference script")

    parser.add_argument("--dit", type=str, required=True, help="DiT checkpoint (first shard of the diffusers format or ComfyUI single file)")
    parser.add_argument("--vae", type=str, required=True, help="FLUX.2 VAE checkpoint (BFL / ComfyUI or diffusers format)")
    parser.add_argument(
        "--text_encoder", type=str, required=True, help="Text Encoder (Qwen3.5-9B) checkpoint (first shard or ComfyUI single file)"
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="Hugging Face cache directory for the tokenizer, or a local directory containing the tokenizer files",
    )
    parser.add_argument("--max_token_length", type=int, default=moss_image_utils.MAX_TOKEN_LENGTH, help="maximum number of text tokens")

    # LoRA
    parser.add_argument("--lora_weight", type=str, nargs="*", required=False, default=None, help="LoRA weight path")
    parser.add_argument("--lora_multiplier", type=float, nargs="*", default=1.0, help="LoRA multiplier")

    # inference
    parser.add_argument("--prompt", type=str, default=None, help="prompt for generation. '\\n' is converted to a newline")
    parser.add_argument("--negative_prompt", type=str, default="", help="negative prompt for generation, default is empty string")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024], help="image size, height and width (multiples of 16)")
    parser.add_argument("--infer_steps", type=int, default=50, help="number of inference steps, default is 50")
    parser.add_argument("--guidance_scale", type=float, default=3.0, help="Guidance scale for classifier free guidance. Default is 3.0")
    parser.add_argument("--flow_shift", type=float, default=5.0, help="Shift factor for the flow matching scheduler. Default is 5.0")
    parser.add_argument(
        "--uncond_skip_layers",
        type=str,
        default="22",
        help="comma separated DiT block indices skipped in the negative (unconditional) branch of CFG. Default is 22 (official). Empty string disables",
    )
    parser.add_argument("--save_path", type=str, required=True, help="path to save generated images")
    parser.add_argument("--seed", type=int, default=None, help="Seed for generation")

    parser.add_argument("--text_encoder_cpu", action="store_true", help="Run the Text Encoder on CPU")
    parser.add_argument("--device", type=str, default=None, help="device to use for inference. If None, use CUDA if available, otherwise use CPU")
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=["flash", "torch", "sageattn", "xformers", "sdpa"],  # "sdpa" for backward compatibility
        help="attention mode",
    )
    parser.add_argument("--blocks_to_swap", type=int, default=0, help="number of DiT blocks to swap to CPU to reduce VRAM usage")
    parser.add_argument(
        "--convrot_int8",
        action="store_true",
        help="quantize the DiT base weights to ConvRot INT8 at load time (ComfyUI *_int8_convrot.safetensors files are detected automatically)",
    )
    parser.add_argument(
        "--convrot_int8_text_encoder",
        action="store_true",
        help="quantize the text encoder weights to ConvRot INT8 at load time (the ComfyUI int8 file is detected automatically)",
    )
    parser.add_argument("--output_type", type=str, default="images", choices=["images", "latent", "latent_images"], help="output type")
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata")
    parser.add_argument("--latent_path", type=str, nargs="*", default=None, help="path to latent for decode. no inference")

    # batch and interactive modes
    parser.add_argument("--from_file", type=str, default=None, help="Read prompts from a file")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode: read prompts from console")

    args = parser.parse_args()

    if args.from_file and args.interactive:
        raise ValueError("Cannot use both --from_file and --interactive at the same time")
    if args.latent_path is None or len(args.latent_path) == 0:
        if args.prompt is None and not args.from_file and not args.interactive:
            raise ValueError("Either --prompt, --from_file or --interactive must be specified")
    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    device = args.device if args.device is not None else "cuda" if torch.cuda.is_available() else "cpu"
    args.device = torch.device(device)
    return args


# region prompt handling


def parse_prompt_line(line: str) -> Dict[str, Any]:
    """Parse a prompt line (`prompt --w 1024 --h 1024 --s 50 --l 3.0 --fs 5.0 --d 42 --n negative`)"""
    parts = line.split(" --")
    overrides = {"prompt": parts[0].strip()}
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
        elif option == "g" or option == "l":
            overrides["guidance_scale"] = float(value)
        elif option == "fs":
            overrides["flow_shift"] = float(value)
        elif option == "n":
            overrides["negative_prompt"] = value
    return overrides


def apply_overrides(args: argparse.Namespace, overrides: Dict[str, Any]) -> argparse.Namespace:
    args_copy = copy.deepcopy(args)
    for key, value in overrides.items():
        if key == "image_size_width":
            args_copy.image_size[1] = value
        elif key == "image_size_height":
            args_copy.image_size[0] = value
        else:
            setattr(args_copy, key, value)
    return args_copy


def process_escape(text: str) -> str:
    """Convert the two characters '\\n' to a newline (multi-line prompts)."""
    return text.replace("\\n", "\n")


def check_inputs(args: argparse.Namespace) -> Tuple[int, int]:
    height, width = args.image_size
    if height <= 0 or width <= 0 or height % 16 != 0 or width % 16 != 0:
        raise ValueError(f"`height` and `width` must be positive multiples of 16 but are {height} and {width}.")
    return height, width


# endregion


# region models


def load_lora_weights(args: argparse.Namespace, prefix: str) -> Optional[List[Dict[str, torch.Tensor]]]:
    if args.lora_weight is None or len(args.lora_weight) == 0:
        return None
    lora_weights_list = []
    for lora_weight in args.lora_weight:
        logger.info(f"Loading LoRA weight from: {lora_weight}")
        lora_sd = load_file(lora_weight)  # load on CPU, dtype is as is
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith(prefix)}
        lora_weights_list.append(lora_sd)
    return lora_weights_list


def load_dit_model(args: argparse.Namespace, device: torch.device) -> moss_image_models.MossImage:
    lora_weights_list = load_lora_weights(args, "lora_unet_")
    loading_device = "cpu" if args.blocks_to_swap > 0 else device
    model = moss_image_utils.load_moss_image_model(
        device,
        args.dit,
        args.attn_mode,
        True,  # split_attn: the batch size is 1, so attention is run without masks
        loading_device,
        torch.bfloat16,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier,
        convrot_int8=args.convrot_int8,
    )
    model.eval().requires_grad_(False)

    if args.blocks_to_swap > 0:
        model.enable_block_swap(args.blocks_to_swap, BlockSwapConfig(device, supports_backward=False))
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        model.to(device)

    clean_memory_on_device(device)
    return model


def load_text_encoder(args: argparse.Namespace, device: torch.device):
    lora_weights_list = load_lora_weights(args, "lora_te_")
    if lora_weights_list is not None:
        # the loader matches keys without a prefix, so strip "lora_te_". Files without Text Encoder LoRA give empty dicts.
        lora_weights_list = [{k[len("lora_te_") :]: v for k, v in sd.items()} for sd in lora_weights_list]
        if not any(len(sd) > 0 for sd in lora_weights_list):
            lora_weights_list = None
    text_encoder = moss_image_utils.load_text_encoder(
        args.text_encoder,
        dtype=torch.bfloat16,
        device=device,
        lora_weights_list=lora_weights_list,
        lora_multipliers=args.lora_multiplier if lora_weights_list is not None else None,
        convrot_int8=args.convrot_int8_text_encoder,
        calc_device=torch.device("cuda") if device.type == "cpu" and torch.cuda.is_available() else None,
    )
    return text_encoder


def load_vae(args: argparse.Namespace, device: torch.device) -> flux2_vae.AutoEncoder:
    vae = flux2_vae.load_flux2_vae(args.vae, dtype=torch.float32, device=device)
    vae.eval().requires_grad_(False)
    return vae


# endregion


# region generation


def encode_prompts(
    tokenizer, text_encoder, prompts: List[str], max_token_length: int, device: torch.device
) -> List[torch.Tensor]:
    """Encode prompts one by one. Returns a list of [L_i, C] bf16 tensors on the CPU."""
    input_ids_list = moss_image_utils.tokenize_prompts(tokenizer, prompts, max_token_length)
    with torch.no_grad():
        embeds = moss_image_utils.encode_tokens(text_encoder, input_ids_list, device)
    return [e.cpu() for e in embeds]


def prepare_text_inputs(args: argparse.Namespace, tokenizer, text_encoder, device: torch.device, conds_cache: Dict[str, torch.Tensor]):
    """Return (prompt embeds, negative prompt embeds) [L, C] tensors on the CPU. The cache is keyed by the prompt string."""
    prompt = process_escape(args.prompt)
    negative_prompt = process_escape(args.negative_prompt)
    to_encode = [p for p in dict.fromkeys([prompt, negative_prompt]) if p not in conds_cache]
    if to_encode:
        logger.info(f"Encoding {len(to_encode)} prompt(s) with the Text Encoder")
        embeds = encode_prompts(tokenizer, text_encoder, to_encode, args.max_token_length, device)
        for p, e in zip(to_encode, embeds):
            conds_cache[p] = e
    return conds_cache[prompt], conds_cache[negative_prompt]


def generate_latents(
    args: argparse.Namespace,
    dit: moss_image_models.MossImage,
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Sample normalized, patchified latents [1, 128, H/16, W/16] (float32)."""
    height, width = check_inputs(args)
    uncond_skip_layers = moss_image_train_utils.parse_skip_layers(args.uncond_skip_layers)
    logger.info(
        f"Image size: {height}x{width} (HxW), steps: {args.infer_steps}, guidance: {args.guidance_scale}, "
        f"flow shift: {args.flow_shift}, uncond skip layers: {uncond_skip_layers}, seed: {seed}"
    )

    positive, positive_mask = moss_image_utils.pad_text_embeds([prompt_embeds.to(device)])
    negative, negative_mask = moss_image_utils.pad_text_embeds([negative_embeds.to(device)])

    # Initial noise is sampled in float32 on the CPU generator (same as the official pipeline)
    return moss_image_train_utils.generate_latents(
        dit,
        positive,
        positive_mask,
        negative,
        negative_mask,
        height,
        width,
        args.infer_steps,
        args.guidance_scale,
        args.flow_shift,
        uncond_skip_layers,
        seed,
        device,
    )


def decode_latents(vae: flux2_vae.AutoEncoder, latents: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Decode normalized, patchified latents to pixels [3, H, W] in -1..1 (float32, CPU)."""
    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_from_dit(latents.to(device, dtype=vae.dtype))
    pixels = pixels.clamp(-1, 1).to("cpu", dtype=torch.float32)
    return pixels[0]


# endregion


# region output


def get_time_flag() -> str:
    return datetime.datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S-%f")[:-3]


def save_latent(latent: torch.Tensor, args: argparse.Namespace, height: int, width: int) -> str:
    os.makedirs(args.save_path, exist_ok=True)
    latent_path = os.path.join(args.save_path, f"{get_time_flag()}_{args.seed}_latent.safetensors")
    metadata = None
    if not args.no_metadata:
        metadata = {
            "seeds": f"{args.seed}",
            "prompt": f"{args.prompt}",
            "negative_prompt": f"{args.negative_prompt}",
            "height": f"{height}",
            "width": f"{width}",
            "infer_steps": f"{args.infer_steps}",
            "guidance_scale": f"{args.guidance_scale}",
            "flow_shift": f"{args.flow_shift}",
            "uncond_skip_layers": f"{args.uncond_skip_layers}",
        }
    save_file({"latent": latent.contiguous()}, latent_path, metadata=metadata)
    logger.info(f"Latent saved to: {latent_path}")
    return latent_path


def save_image(pixels: torch.Tensor, args: argparse.Namespace, original_base_name: Optional[str] = None) -> str:
    os.makedirs(args.save_path, exist_ok=True)
    original_name = "" if original_base_name is None else f"_{original_base_name}"
    image_path = os.path.join(args.save_path, f"{get_time_flag()}_{args.seed}{original_name}.png")
    x = ((pixels.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
    Image.fromarray(x).save(image_path)
    logger.info(f"Image saved to: {image_path}")
    return image_path


def save_output(
    args: argparse.Namespace, vae: Optional[flux2_vae.AutoEncoder], latent: torch.Tensor, device: torch.device, original_base_name=None
):
    height, width = latent.shape[-2] * 16, latent.shape[-1] * 16
    if args.output_type in ["latent", "latent_images"]:
        save_latent(latent, args, height, width)
    if args.output_type == "latent":
        return
    if vae is None:
        logger.error("VAE is None, cannot decode latents")
        return
    pixels = decode_latents(vae, latent, device)
    save_image(pixels, args, original_base_name)


# endregion


# region modes


def generate_single(args: argparse.Namespace):
    device = args.device
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    args.seed = seed

    # 1. text encoder
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    tokenizer = moss_image_utils.load_tokenizer(args.tokenizer_cache_dir)
    text_encoder = load_text_encoder(args, text_encoder_device)
    prompt_embeds, negative_embeds = prepare_text_inputs(args, tokenizer, text_encoder, text_encoder_device, {})
    del text_encoder
    gc.collect()
    clean_memory_on_device(device)

    # 2. DiT
    dit = load_dit_model(args, device)
    latents = generate_latents(args, dit, prompt_embeds, negative_embeds, device, seed)
    del dit
    gc.collect()
    clean_memory_on_device(device)
    synchronize_device(device)

    # 3. VAE
    vae = load_vae(args, "cpu") if args.output_type != "latent" else None
    save_output(args, vae, latents, device)


def generate_batch(args: argparse.Namespace):
    with open(args.from_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    prompts_data = [parse_prompt_line(line.strip()) for line in lines if line.strip() and not line.strip().startswith("#")]
    if not prompts_data:
        logger.warning("No valid prompts found")
        return
    all_args = [apply_overrides(args, pd) for pd in prompts_data]
    for prompt_args in all_args:
        check_inputs(prompt_args)
        if prompt_args.seed is None:
            prompt_args.seed = random.randint(0, 2**32 - 1)

    device = args.device

    # 1. encode all prompts
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    tokenizer = moss_image_utils.load_tokenizer(args.tokenizer_cache_dir)
    text_encoder = load_text_encoder(args, text_encoder_device)
    conds_cache = {}
    all_embeds = [prepare_text_inputs(a, tokenizer, text_encoder, text_encoder_device, conds_cache) for a in all_args]
    del text_encoder, conds_cache
    gc.collect()
    clean_memory_on_device(device)

    # 2. generate all latents
    dit = load_dit_model(all_args[0], device)
    all_latents = []
    for i, (prompt_args, (prompt_embeds, negative_embeds)) in enumerate(zip(all_args, all_embeds)):
        logger.info(f"Generating {i + 1}/{len(all_args)}: {prompt_args.prompt}")
        all_latents.append(generate_latents(prompt_args, dit, prompt_embeds, negative_embeds, device, prompt_args.seed))
    del dit
    gc.collect()
    clean_memory_on_device(device)
    synchronize_device(device)

    # 3. decode
    vae = load_vae(args, "cpu") if args.output_type != "latent" else None
    for prompt_args, latents in zip(all_args, all_latents):
        save_output(prompt_args, vae, latents, device)


def generate_interactive(args: argparse.Namespace):
    device = args.device
    text_encoder_device = torch.device("cpu") if args.text_encoder_cpu else device
    tokenizer = moss_image_utils.load_tokenizer(args.tokenizer_cache_dir)
    text_encoder = load_text_encoder(args, text_encoder_device)
    dit = load_dit_model(args, device)
    vae = load_vae(args, "cpu") if args.output_type != "latent" else None
    conds_cache = {}

    print("Interactive mode. Enter prompts (Ctrl+D or Ctrl+Z (Windows) to exit):")
    try:
        import prompt_toolkit

        session = prompt_toolkit.PromptSession()
        input_line = lambda p: session.prompt(p)
    except ImportError:
        logger.warning("prompt_toolkit not found. Using basic input instead.")
        input_line = lambda p: input(p)

    try:
        while True:
            try:
                line = input_line("> ")
                if not line.strip():
                    continue
                if len(line.strip()) == 1 and line.strip() in ["\x04", "\x1a"]:
                    raise EOFError
                prompt_args = apply_overrides(args, parse_prompt_line(line))
                check_inputs(prompt_args)
                prompt_args.seed = prompt_args.seed if prompt_args.seed is not None else random.randint(0, 2**32 - 1)
                prompt_embeds, negative_embeds = prepare_text_inputs(prompt_args, tokenizer, text_encoder, text_encoder_device, conds_cache)
                latents = generate_latents(prompt_args, dit, prompt_embeds, negative_embeds, device, prompt_args.seed)
                save_output(prompt_args, vae, latents, device)
                if vae is not None:
                    vae.to("cpu")
                clean_memory_on_device(device)
            except KeyboardInterrupt:
                print("\nInterrupted. Continue (Ctrl+D or Ctrl+Z (Windows) to exit)")
                continue
    except EOFError:
        print("\nExiting interactive mode")


def decode_latent_files(args: argparse.Namespace):
    device = args.device
    vae = load_vae(args, device)
    for latent_path in args.latent_path:
        original_base_name = os.path.splitext(os.path.basename(latent_path))[0]
        if os.path.splitext(latent_path)[1] != ".safetensors":
            latents = torch.load(latent_path, map_location="cpu")
            metadata = {}
        else:
            latents = load_file(latent_path)["latent"]
            with safe_open(latent_path, framework="pt") as f:
                metadata = f.metadata() or {}
        logger.info(f"Loaded latent from {latent_path}. Shape: {latents.shape}, metadata: {metadata}")
        args.seed = int(metadata.get("seeds", 0))
        if latents.ndim == 3:
            latents = latents.unsqueeze(0)
        pixels = decode_latents(vae, latents, device)
        save_image(pixels, args, original_base_name)


# endregion


def main():
    args = parse_args()
    logger.info(f"Using device: {args.device}")

    if args.latent_path is not None and len(args.latent_path) > 0:
        decode_latent_files(args)
    elif args.from_file:
        generate_batch(args)
    elif args.interactive:
        generate_interactive(args)
    else:
        generate_single(args)

    logger.info("Done!")


if __name__ == "__main__":
    main()
