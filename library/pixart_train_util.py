import argparse
import math
import os
from typing import Optional

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from accelerate import init_empty_weights
from tqdm import tqdm

from transformers import T5EncoderModel, T5Tokenizer
from accelerate import Accelerator
from diffusers import AutoencoderKL
import pixart_model_util
import library.train_util as train_util
from library.pixart_pipeline import SimplePixartPipeline

from library import model_util
# Figure out weighting for T5?
#from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from .utils import setup_logging
import library.save_naming as save_naming
setup_logging()
import logging
logger = logging.getLogger(__name__)

TOKENIZER_PATH = "DeepFloyd/t5-v1_1-xxl"

# DEFAULT_NOISE_OFFSET = 0.0357

def get_hidden_states_pixart(
    max_token_length: int,
    input_ids: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    weight_dtype: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
):

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(input_ids, attention_mask=prompt_attention_mask)[0]
    prompt_attention_mask = prompt_attention_mask

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states, prompt_attention_mask

def load_target_model(args, accelerator, model_version: str, weight_dtype):
    model_dtype = match_mixed_precision(args, weight_dtype)  # prepare fp16/bf16
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            (
                load_stable_diffusion_format,
                text_encoder,
                vae,
                dit,
                ckpt_info,
            ) = _load_target_model(
                args.pretrained_model_name_or_path,
                args.resolution,
                args.vae,
                model_version,
                args.enable_ar_condition,
                args.max_token_length,
                args.text_encoder_path,
                args.load_t5_in_4bit,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                model_dtype,
            )

            # work on low-ram device
            if args.lowram:
                text_encoder.to(accelerator.device)
                dit.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    return load_stable_diffusion_format, text_encoder, vae, dit, ckpt_info

def load_vae(vae_id, dtype):
    logger.info(f"load VAE: {vae_id}")
    # Diffusers local/remote
    try:
        vae = AutoencoderKL.from_pretrained(vae_id, subfolder=None, torch_dtype=dtype)
    except EnvironmentError as e:
        logger.error(f"exception occurs in loading vae: {e}")
        logger.error("retry with subfolder='vae'")
        vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae", torch_dtype=dtype)
    return vae

def _load_target_model(
    name_or_path: str, base_resolution, vae_path: Optional[str], model_version: str, enable_ar_condition, max_token_length, text_encoder_path, load_t5_in_4bit, weight_dtype, device="cpu", model_dtype=None
):
    # model_dtype only work with full fp16/bf16
    name_or_path = os.readlink(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers

    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        (
            text_encoder,
            vae,
            dit,
            ckpt_info,
        ) = pixart_model_util.load_models_from_pixart_checkpoint(model_version, name_or_path, base_resolution, enable_ar_condition, max_token_length, text_encoder_path, load_t5_in_4bit, vae_path, device, model_dtype)
    else:
        raise Exception("kabachuha TODO")

    # VAEを読み込む
    if vae_path is not None:
        vae = model_util.load_vae(vae_path, weight_dtype)
        logger.info("additional VAE loaded")

    return load_stable_diffusion_format, text_encoder, vae, dit, ckpt_info


def load_tokenizers(args: argparse.Namespace):
    logger.info("prepare tokenizers")

    if not args.load_t5_in_4bit:
        print("WARNING: The T5 text encoder is being loaded not in 4bit mode. It's heavily recommended to use it in 4 bit quantization to enhance speed and save up the VRAM dramatically while not losing precision that much") # ("load_in_4bit=True")

    original_paths = [args.text_encoder_path]
    tokeniers = []
    for i, original_path in enumerate(original_paths):
        tokenizer: T5Tokenizer = None
        if args.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(args.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                logger.info(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = T5Tokenizer.from_pretrained(local_tokenizer_path)

        if tokenizer is None:
            tokenizer = T5Tokenizer.from_pretrained(original_path)

        if args.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            logger.info(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        if i == 1:
            tokenizer.pad_token_id = 0  # fix pad token id to make same as open clip tokenizer

        tokeniers.append(tokenizer)

    if hasattr(args, "max_token_length") and args.max_token_length is not None:
        logger.info(f"update token length: {args.max_token_length}")

    return tokeniers


def match_mixed_precision(args, weight_dtype):
    if args.full_fp16:
        assert (
            weight_dtype == torch.float16
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        return weight_dtype
    elif args.full_bf16:
        assert (
            weight_dtype == torch.bfloat16
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        return weight_dtype
    else:
        return None

# Same in PixArt_blocks as in other diffusions
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=timesteps.device
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_timestep_embedding(x, outdim):
    assert len(x.shape) == 2
    b, dims = x.shape[0], x.shape[1]
    x = torch.flatten(x)
    emb = timestep_embedding(x, outdim)
    emb = torch.reshape(emb, (b, dims * outdim))
    return emb

def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


def save_pixart_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder,
    dit,
    vae,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = {} # TODO: add metadata for pixart
        pixart_model_util.save_pixart_checkpoint(
            ckpt_file,
            text_encoder,
            dit,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            sai_metadata,
            save_dtype,
        )
    
    save_naming.save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, None, logger
    )

# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_pixart_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path,
    save_original_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder,
    dit,
    vae,
    ckpt_info,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = {} # kabachuha TODO
        pixart_model_util.save_pixart_checkpoint(
            ckpt_file,
            text_encoder,
            dit,
            epoch_no,
            global_step,
            ckpt_info,
            vae,
            sai_metadata,
            save_dtype,
        )

    def diffusers_saver(out_dir):
        raise NotImplementedError("kabachuha TODO")

    save_naming.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_original_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
        logger
    )


def add_pixart_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=TOKENIZER_PATH
    )
    parser.add_argument(
        "--load_t5_in_4bit",
        action="store_true",
        help="Launch T5 LLM in 4bit to save vram and speed"
    )
    parser.add_argument(
        "--enable_ar_conditioning",
        action="store_true",
        help="Experimental: enable conditioning the transformer on different aspect ratios"
    )


def verify_pixart_training_args(args: argparse.Namespace, supportTextEncoderCaching: bool = True):
    assert not args.v2, "v2 cannot be enabled in PixArt training / PixArt学習ではv2を有効にすることはできません"
    if args.v_parameterization:
        logger.warning("v_parameterization will be unexpected / PixArt学習ではv_parameterizationは想定外の動作になります")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in PixArt training currently / PixArt学習では今のところweighted_captionsを有効にすることはできません"

    if supportTextEncoderCaching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )

# accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet
def sample_images(*args, **kwargs):
    return train_util.sample_images_common(SimplePixartPipeline, *args, **kwargs)
