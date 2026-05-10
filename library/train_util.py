# common functions for training

import argparse
import ast
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import datetime
import importlib
import json
import logging
import pathlib
import re
import shutil
import time
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO
import toml

# from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from packaging.version import Version

import torch
from library.device_utils import init_ipex, clean_memory_on_device
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, TextEncodingStrategy

init_ipex()

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import transformers
from diffusers.optimization import (
    SchedulerType as DiffusersSchedulerType,
    TYPE_TO_SCHEDULER_FUNCTION as DIFFUSERS_TYPE_TO_SCHEDULER_FUNCTION,
)
from transformers.optimization import SchedulerType, TYPE_TO_SCHEDULER_FUNCTION
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    AutoencoderKL,
)
from library import custom_train_functions, sd3_utils
from library.original_unet import UNet2DConditionModel
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import imagesize
import cv2
import safetensors.torch
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
import library.model_util as model_util
import library.huggingface_util as huggingface_util
import library.sai_model_spec as sai_model_spec
import library.deepspeed_utils as deepspeed_utils
from library.utils import setup_logging, resize_image, validate_interpolation_fn

setup_logging()
import logging

logger = logging.getLogger(__name__)
# from library.attention_processors import FlashAttnProcessor
# from library.hypernetwork import replace_attentions_for_hypernetwork
from library.original_unet import UNet2DConditionModel

# Accelerator setup helpers have moved to library.accelerator_setup;
# re-exported here for backward compatibility.
# New code should import from library.accelerator_setup directly.
# HIGH_VRAM is mutated by enable_high_vram(); for legacy ``train_util.HIGH_VRAM``
# attribute reads we forward through a module-level __getattr__ below.
from library.accelerator_setup import (  # noqa: F401
    enable_high_vram,
    prepare_dataset_args,
    prepare_accelerator,
    prepare_dtype,
    patch_accelerator_for_fp16_training,
)


def __getattr__(name):
    if name == "HIGH_VRAM":
        from library import accelerator_setup
        return accelerator_setup.HIGH_VRAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

# Dataset core has moved to library.dataset; re-exported here for backward compatibility.
# New code should import from library.dataset directly.
from library.dataset import (  # noqa: F401
    IMAGE_EXTENSIONS,
    IMAGE_TRANSFORMS,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
    split_train_val,
    ImageInfo,
    BucketManager,
    BucketBatchIndex,
    AugHelper,
    BaseDataset,
    DatasetGroup,
    MinimalDataset,
    load_arbitrary_dataset,
    debug_dataset,
    glob_images,
    glob_images_pathlib,
    load_image,
)












# Subset classes have moved to library.subset; re-exported here for backward compatibility.
# New code should import from library.subset directly.
from library.subset import BaseSubset, DreamBoothSubset, FineTuningSubset, ControlNetSubset  # noqa: F401



# DreamBooth / FineTuning / ControlNet datasets have moved to dedicated modules;
# re-exported here for backward compatibility. New code should import from library.* directly.
from library.dreambooth_dataset import DreamBoothDataset  # noqa: F401
from library.finetuning_dataset import FineTuningDataset  # noqa: F401
from library.controlnet_dataset import ControlNetDataset  # noqa: F401


# Caching functions have moved to library.caching; re-exported here for backward compatibility.
# New code should import from library.caching directly.
from library.caching import (  # noqa: F401
    is_disk_cached_latents_is_expected,
    trim_and_resize_if_required,
    load_images_and_masks_for_caching,
    cache_batch_latents,
    cache_batch_text_encoder_outputs,
    cache_batch_text_encoder_outputs_sd3,
    save_text_encoder_outputs_to_disk,
    load_text_encoder_outputs_from_disk,
)


# 戻り値は、latents_tensor, (original_size width, original_size height), (crop left, crop top)
# TODO update to use CachingStrategy
# def load_latents_from_disk(
#     npz_path,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
#     npz = np.load(npz_path)
#     if "latents" not in npz:
#         raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

#     latents = npz["latents"]
#     original_size = npz["original_size"].tolist()
#     crop_ltrb = npz["crop_ltrb"].tolist()
#     flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
#     alpha_mask = npz["alpha_mask"] if "alpha_mask" in npz else None
#     return latents, original_size, crop_ltrb, flipped_latents, alpha_mask


# def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
#     kwargs = {}
#     if flipped_latents_tensor is not None:
#         kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
#     if alpha_mask is not None:
#         kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
#     np.savez(
#         npz_path,
#         latents=latents_tensor.float().cpu().numpy(),
#         original_size=np.array(original_size),
#         crop_ltrb=np.array(crop_ltrb),
#         **kwargs,
#     )




# 画像を読み込む。戻り値はnumpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)


# endregion

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# Model I/O, hashing and metadata helpers have moved to library.model_io;
# re-exported here for backward compatibility.
# New code should import from library.model_io directly.
from library.model_io import (  # noqa: F401
    model_hash,
    calculate_sha256,
    precalculate_safetensors_hashes,
    addnet_hash_legacy,
    addnet_hash_safetensors,
    get_git_revision_hash,
    replace_unet_modules,
    load_metadata_from_safetensors,
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
    build_minimum_network_metadata,
    get_sai_model_spec,
    get_sai_model_spec_dataclass,
    _load_target_model,
    load_target_model,
)



# def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
#     replace_attentions_for_hypernetwork()
#     # unet is not used currently, but it is here for future use
#     unet.enable_xformers_memory_efficient_attention()
#     return
#     if mem_eff_attn:
#         unet.set_attn_processor(FlashAttnProcessor())
#     elif xformers:
#         unet.enable_xformers_memory_efficient_attention()


# def replace_unet_cross_attn_to_xformers():
#     logger.info("CrossAttention.forward has been replaced to enable xformers.")
#     try:
#         import xformers.ops
#     except ImportError:
#         raise ImportError("No xformers / xformersがインストールされていないようです")

#     def forward_xformers(self, x, context=None, mask=None):
#         h = self.heads
#         q_in = self.to_q(x)

#         context = default(context, x)
#         context = context.to(x.dtype)

#         if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
#             context_k, context_v = self.hypernetwork.forward(x, context)
#             context_k = context_k.to(x.dtype)
#             context_v = context_v.to(x.dtype)
#         else:
#             context_k = context
#             context_v = context

#         k_in = self.to_k(context_k)
#         v_in = self.to_v(context_v)

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
#         del q_in, k_in, v_in

#         q = q.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

#         out = rearrange(out, "b n h d -> b n (h d)", h=h)

#         # diffusers 0.7.0~
#         out = self.to_out[0](out)
#         out = self.to_out[1](out)
#         return out


#     diffusers.models.attention.CrossAttention.forward = forward_xformers

"""
def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers):
    # vae is not used currently, but it is here for future use
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        # とりあえずDiffusersのxformersを使う。AttentionがあるのはMidBlockのみ
        logger.info("Use Diffusers xformers for VAE")
        vae.encoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)
        vae.decoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)


def replace_vae_attn_to_memory_efficient():
    logger.info("AttentionBlock.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states):
        logger.info("forward_flash_attn")
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    diffusers.models.attention.AttentionBlock.forward = forward_flash_attn
"""


# endregion


# region arguments






# Argument definitions and configuration helpers have moved to library.args;
# re-exported here for backward compatibility.
# New code should import from library.args directly.
from library.args import (  # noqa: F401
    add_sd_models_arguments,
    add_optimizer_arguments,
    add_training_arguments,
    add_masked_loss_arguments,
    add_dit_training_arguments,
    get_sanitized_config_or_none,
    verify_command_line_training_args,
    verify_training_args,
    add_dataset_arguments,
    add_sd_saving_arguments,
    read_config_from_file,
    resume_from_local_or_hf_if_specified,
)


# endregion

# region utils



# Optimizer / scheduler / LR-logging helpers have moved to library.optimizer;
# re-exported here for backward compatibility.
# New code should import from library.optimizer directly.
from library.optimizer import (  # noqa: F401
    get_optimizer,
    get_optimizer_train_eval_fn,
    is_schedulefree_optimizer,
    get_dummy_scheduler,
    get_scheduler_fix,
    append_lr_to_logs,
    append_lr_to_logs_with_names,
)




# Text encoder hidden-state helpers have moved to library.hidden_states;
# re-exported here for backward compatibility.
# New code should import from library.hidden_states directly.
from library.hidden_states import (  # noqa: F401
    get_hidden_states,
    pool_workaround,
    get_hidden_states_sdxl,
)


def default_if_none(value, default):
    return default if value is None else value


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        # 保存するか否かは呼び出し側で判断済み

        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch  # 例: 最初のepochの途中で保存したら0になる、SDモデルに保存される
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(args.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
        else:
            out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step))

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no))
            else:
                remove_out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no))

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if args.save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=True)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name, force_sync_upload=True)


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    # Sample a random timestep for each image
    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
    timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # This moves the alphas_cumprod back to the CPU after it is moved in noise_scheduler.add_noise
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps.cpu())
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full((b_size,), args.huber_c * args.huber_scale, device=timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str, huber_c: Optional[torch.Tensor] = None
):
    """
    NOTE: if you're using the scheduled version, huber_c has to depend on the timesteps already
    """
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        # Reshape huber_c to broadcast with model_pred (supports 4D and 5D tensors)
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        # Reshape huber_c to broadcast with model_pred (supports 4D and 5D tensors)
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss



# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def get_my_scheduler(
    *,
    sample_sampler: str,
    v_parameterization: bool,
):
    sched_init_args = {}
    if sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
    elif sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        steps_offset=1,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def sample_images(*args, **kwargs):
    return sample_images_common(StableDiffusionLongPromptWeightingPipeline, *args, **kwargs)


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img_diffusers
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps
                prompt_dict["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # guidance scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

            m = re.match(r"i (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["image"] = m.group(1).strip()
                continue

            m = re.match(r"ctr (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["cfg_trunc_ratio"] = float(m.group(1))
                continue

            m = re.match(r"rcfg (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["renorm_cfg"] = float(m.group(1))
                continue

            m = re.match(r"fs (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["flow_shift"] = m.group(1)
                continue

            m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
            if m:  # additional network multiplier(s) — comma-separated list, same as gen_img.py
                prompt_dict["additional_network_multiplier"] = [float(v) for v in m.group(1).split(",")]
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(ex)

    return prompt_dict


def load_prompts(prompt_file: str) -> List[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            from library.train_util import line_to_prompt_dict

            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def sample_images_common(
    pipe_class,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    steps: int,
    device,
    vae,
    tokenizer,
    text_encoder,
    unet_wrapped,
    prompt_replacement=None,
    controlnet=None,
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    TODO Use strategies here
    """

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

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(distributed_state.device)  # distributed_state.device is same as accelerator.device

    # unwrap unet and text_encoder(s)
    unet = accelerator.unwrap_model(unet_wrapped)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # read prompts
    if args.sample_prompts.endswith(".txt"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif args.sample_prompts.endswith(".toml"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif args.sample_prompts.endswith(".json"):
        with open(args.sample_prompts, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    default_scheduler = get_my_scheduler(sample_sampler=args.sample_sampler, v_parameterization=args.v_parameterization)

    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=default_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipeline.to(distributed_state.device)
    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

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
                    accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
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
                        accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
                    )

    # clear pipeline and cache to reduce vram usage
    del pipeline

    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    pipeline: Union[StableDiffusionLongPromptWeightingPipeline, SdxlStableDiffusionLongPromptWeightingPipeline],
    save_dir,
    prompt_dict,
    epoch,
    steps,
    prompt_replacement,
    controlnet=None,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()

    scheduler = get_my_scheduler(
        sample_sampler=sampler_name,
        v_parameterization=args.v_parameterization,
    )
    pipeline.scheduler = scheduler

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)

    # Round down so the latent shape is divisible by the UNet's largest stride:
    # SDXL has 2 downsamples (latent /4 → image /32); SD1.5/2.x has 3 (latent /8 → image /64).
    divisor = 32 if isinstance(pipeline, SdxlStableDiffusionLongPromptWeightingPipeline) else 64
    height = max(divisor, height - height % divisor)
    width = max(divisor, width - width % divisor)
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # Prepare inpainting source image and mask when training an inpainting model.
    # The prompt line should include "--i /path/to/image.jpg".
    # The mask is generated reproducibly from the prompt seed (or randomly if no seed).
    inpaint_image = None
    inpaint_mask = None
    if getattr(args, "train_inpainting", False):
        image_path = prompt_dict.get("image")
        if image_path:
            from library.mask_generator import wobbly_ellipse_mask as _gen_mask

            if not os.path.exists(image_path):
                logger.warning(f"inpaint image not found, skipping sample: {image_path}")
                return
            inpaint_image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
            inpaint_mask = _gen_mask(width, height, seed=seed)
            logger.info(f"inpaint image: {image_path}")
        else:
            logger.warning(
                "train_inpainting is set but no source image specified in the prompt line; "
                "skipping sample. Add '--i /path/to/image.jpg' to the prompt to enable inpainting sampling."
            )
            return

    with accelerator.autocast(), torch.no_grad():
        latents = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
            controlnet=controlnet,
            controlnet_image=controlnet_image,
            inpaint_image=inpaint_image,
            inpaint_mask=inpaint_mask,
        )

    if torch.cuda.is_available():
        with torch.cuda.device(torch.cuda.current_device()):
            torch.cuda.empty_cache()

    image = pipeline.latents_to_image(latents)[0]

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        import wandb

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)  # positive prompt as a caption


def init_trackers(accelerator: Accelerator, args: argparse.Namespace, default_tracker_name: str):
    """
    Initialize experiment trackers with tracker specific behaviors
    """
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            default_tracker_name if args.log_tracker_name is None else args.log_tracker_name,
            config=get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

        if "wandb" in [tracker.name for tracker in accelerator.trackers]:
            import wandb

            wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)

            # Define specific metrics to handle validation and epochs "steps"
            wandb_tracker.define_metric("epoch", hidden=True)
            wandb_tracker.define_metric("val_step", hidden=True)

            wandb_tracker.define_metric("global_step", hidden=True)


# endregion


# region 前処理用


class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません: {img_path}, error: {e}")
            return None

        return (tensor_pil, img_path)


# endregion


# collate_fn用 epoch,stepはmultiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0
        return self.loss_total / losses
