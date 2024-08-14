import argparse
import math
import os
import numpy as np
import toml
import json
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, PartialState
from transformers import CLIPTextModel
from tqdm import tqdm
from PIL import Image

from library import flux_models, flux_utils, strategy_base
from library.sd3_train_utils import load_prompts
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    flux,
    ae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap unet and text_encoder(s)
    flux = accelerator.unwrap_model(flux)
    text_encoders = [accelerator.unwrap_model(te) for te in text_encoders]
    # print([(te.parameters().__next__().device if te is not None else None) for te in text_encoders])

    prompts = load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    flux,
                    text_encoders,
                    ae,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator,
                        args,
                        flux,
                        text_encoders,
                        ae,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    flux: flux_models.Flux,
    text_encoders: List[CLIPTextModel],
    ae: flux_models.AutoEncoder,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    # negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 3.5)
    seed = prompt_dict.get("seed")
    # controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    # sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        # if negative_prompt is not None:
        #     negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    # if negative_prompt is None:
    #     negative_prompt = ""

    height = max(64, height - height % 16)  # round to divisible by 16
    width = max(64, width - width % 16)  # round to divisible by 16
    logger.info(f"prompt: {prompt}")
    # logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    # logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # encode prompts
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
        te_outputs = sample_prompts_te_outputs[prompt]
    else:
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        te_outputs = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens_and_masks)

    l_pooled, t5_out, txt_ids = te_outputs

    # sample image
    weight_dtype = ae.dtype  # TOFO give dtype as argument
    packed_latent_height = height // 16
    packed_latent_width = width // 16
    noise = torch.randn(
        1,
        packed_latent_height * packed_latent_width,
        16 * 2 * 2,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None,
    )
    timesteps = get_schedule(sample_steps, noise.shape[1], shift=True)  # FLUX.1 dev -> shift=True
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(accelerator.device, weight_dtype)

    with accelerator.autocast(), torch.no_grad():
        x = denoise(flux, noise, img_ids, t5_out, txt_ids, l_pooled, timesteps=timesteps, guidance=scale)

    x = x.float()
    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)

    # latent to image
    clean_memory_on_device(accelerator.device)
    org_vae_device = ae.device  # will be on cpu
    ae.to(accelerator.device)  # distributed_state.device is same as accelerator.device
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    ae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # wandb有効時のみログを送信
    try:
        wandb_tracker = accelerator.get_tracker("wandb")
        try:
            import wandb
        except ImportError:  # 事前に一度確認するのでここはエラー出ないはず
            raise ImportError("No wandb / wandb がインストールされていないようです")

        wandb_tracker.log({f"sample_{i}": wandb.Image(image)})
    except:  # wandb 無効時
        pass


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
