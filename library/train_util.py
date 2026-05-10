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


# Checkpoint filename templates and save / rotate helpers have moved to
# library.checkpoint_io; re-exported here for backward compatibility.
# New code should import from library.checkpoint_io directly.
from library.checkpoint_io import (  # noqa: F401
    EPOCH_STATE_NAME,
    EPOCH_FILE_NAME,
    EPOCH_DIFFUSERS_DIR_NAME,
    LAST_STATE_NAME,
    DEFAULT_EPOCH_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    DEFAULT_STEP_NAME,
    STEP_STATE_NAME,
    STEP_FILE_NAME,
    STEP_DIFFUSERS_DIR_NAME,
)

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


# Checkpoint save / rotate helpers have moved to library.checkpoint_io;
# re-exported here for backward compatibility.
# New code should import from library.checkpoint_io directly.
from library.checkpoint_io import (  # noqa: F401
    default_if_none,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    save_sd_model_on_epoch_end_or_stepwise,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    save_sd_model_on_train_end,
    save_sd_model_on_train_end_common,
)




# Loss / noise scheduling helpers have moved to library.loss;
# re-exported here for backward compatibility.
# New code should import from library.loss directly.
from library.loss import (  # noqa: F401
    get_timesteps,
    get_noise_noisy_latents_and_timesteps,
    get_huber_threshold_if_needed,
    conditional_loss,
)



# Sampling helpers (default scheduler, prompt parsing, sample generation)
# have moved to library.sampling; re-exported here for backward compatibility.
# New code should import from library.sampling directly.
from library.sampling import (  # noqa: F401, E402
    SCHEDULER_LINEAR_START,
    SCHEDULER_LINEAR_END,
    SCHEDULER_TIMESTEPS,
    SCHEDLER_SCHEDULE,
    get_my_scheduler,
    sample_images,
    line_to_prompt_dict,
    load_prompts,
    sample_images_common,
    sample_image_inference,
)


# Logging / tracker helpers have moved to library.logging_util;
# re-exported here for backward compatibility.
# New code should import from library.logging_util directly.
from library.logging_util import init_trackers  # noqa: F401, E402


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


# LossRecorder has moved to library.logging_util;
# re-exported here for backward compatibility.
# New code should import from library.logging_util directly.
from library.logging_util import LossRecorder  # noqa: F401, E402
