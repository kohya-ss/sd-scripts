# use Diffusers' pipeline to generate images

import argparse
import datetime
import math
import os
import random
import re
from einops import repeat
import numpy as np
import torch

try:
    import intel_extension_for_pytorch as ipex

    if torch.xpu.is_available():
        from library.ipex import ipex_init

        ipex_init()
except Exception:
    pass
from tqdm import tqdm
from PIL import Image
from transformers import CLIPTextModel, PreTrainedTokenizerFast
from diffusers.pipelines.wuerstchen.modeling_wuerstchen_prior import WuerstchenPrior
from diffusers import AutoPipelineForText2Image, DDPMWuerstchenScheduler

# from diffusers.pipelines.wuerstchen.pipeline_wuerstchen_prior import DEFAULT_STAGE_C_TIMESTEPS
from wuerstchen_train import EfficientNetEncoder


def generate(args):
    dtype = torch.float32
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16

    device = args.device

    os.makedirs(args.outdir, exist_ok=True)

    # load tokenizer
    print("load tokenizer")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.pretrained_prior_model_name_or_path, subfolder="tokenizer")

    # load text encoder
    print("load text encoder")
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="text_encoder", torch_dtype=dtype
    )

    # load prior model
    print("load prior model")
    prior: WuerstchenPrior = WuerstchenPrior.from_pretrained(
        args.pretrained_prior_model_name_or_path, subfolder="prior", torch_dtype=dtype
    )

    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        print("Use xformers by Diffusers")
        set_diffusers_xformers_flag(prior, True)

    # load pipeline
    print("load pipeline")
    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.pretrained_decoder_model_name_or_path,
        prior_prior=prior,
        prior_text_encoder=text_encoder,
        prior_tokenizer=tokenizer,
    )
    pipeline = pipeline.to(device, torch_dtype=dtype)

    # generate image
    while True:
        width = args.w
        height = args.h
        seed = args.seed
        negative_prompt = None

        if args.interactive:
            print("prompt:")
            prompt = input()
            if prompt == "":
                break

            # parse prompt
            prompt_args = prompt.split(" --")
            prompt = prompt_args[0]

            for parg in prompt_args[1:]:
                try:
                    m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                    if m:
                        width = int(m.group(1))
                        print(f"width: {width}")
                        continue

                    m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                    if m:
                        height = int(m.group(1))
                        print(f"height: {height}")
                        continue

                    m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                    if m:  # seed
                        seed = int(m.group(1))
                        print(f"seed: {seed}")
                        continue

                    m = re.match(r"n (.+)", parg, re.IGNORECASE)
                    if m:  # negative prompt
                        negative_prompt = m.group(1)
                        print(f"negative prompt: {negative_prompt}")
                        continue
                except ValueError as ex:
                    print(f"Exception in parsing / 解析エラー: {parg}")
                    print(ex)
        else:
            prompt = args.prompt
            negative_prompt = args.negative_prompt

        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)

        with torch.autocast(device):
            image = pipeline(
                prompt,
                negative_prompt=negative_prompt,
                # prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
                generator=generator,
                width=width,
                height=height,
            ).images[0]

        # save image
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        image.save(os.path.join(args.outdir, f"image_{timestamp}.png"))

        if not args.interactive:
            break

    print("Done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # train_util.add_sd_models_arguments(parser)
    parser.add_argument(
        "--pretrained_prior_model_name_or_path",
        type=str,
        default="warp-ai/wuerstchen-prior",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_decoder_model_name_or_path",
        type=str,
        default="warp-ai/wuerstchen",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    parser.add_argument("--prompt", type=str, default="A photo of a cat")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--outdir", type=str, default=".")
    parser.add_argument("--w", type=int, default=1024)
    parser.add_argument("--h", type=int, default=1024)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    generate(args)
