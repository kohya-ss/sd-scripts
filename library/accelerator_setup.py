"""Accelerator / dtype / dataset-args setup helpers.

Hosts the routines that prepare the Accelerator (incl. DeepSpeed plugin and
DDP options), resolve mixed-precision dtypes, normalise dataset-related
arguments, patch the fp16 grad scaler, and toggle the ``HIGH_VRAM`` mode flag.
``HIGH_VRAM`` is a mutable module-level flag toggled by ``enable_high_vram``
and read from ``library.caching`` / ``library.dataset`` / the strategy
modules. Extracted from ``library.train_util`` and re-exported there for
backward compatibility (the legacy ``train_util.HIGH_VRAM`` attribute is
served via a module-level ``__getattr__`` shim).
"""

import argparse
import datetime
import logging
import os

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
from packaging.version import Version

import library.deepspeed_utils as deepspeed_utils
from library.utils import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


# Mutable module-level flag toggled by enable_high_vram(). Read from caching /
# dataset / strategy modules to skip per-step CUDA cache clears on big-VRAM rigs.
HIGH_VRAM = False


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled / highvramが有効です")
        global HIGH_VRAM
        HIGH_VRAM = True


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
    if args.resolution is not None:
        args.resolution = tuple([int(r) for r in args.resolution.split(",")])
        if len(args.resolution) == 1:
            args.resolution = (args.resolution[0], args.resolution[0])
        assert (
            len(args.resolution) == 2
        ), f"resolution must be 'size' or 'width,height' / resolution（解像度）は'サイズ'または'幅','高さ'で指定してください: {args.resolution}"

    if args.skip_image_resolution is not None:
        args.skip_image_resolution = tuple([int(r) for r in args.skip_image_resolution.split(",")])
        if len(args.skip_image_resolution) == 1:
            args.skip_image_resolution = (args.skip_image_resolution[0], args.skip_image_resolution[0])
        assert (
            len(args.skip_image_resolution) == 2
        ), f"skip_image_resolution must be 'size' or 'width,height' / skip_image_resolutionは'サイズ'または'幅','高さ'で指定してください: {args.skip_image_resolution}"

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(",")])
        assert (
            len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            logger.warning(
                f"latents in npz is ignored when color_aug or random_crop is True / color_augまたはrandom_cropを有効にした場合、npzファイルのlatentsは無視されます"
            )


def prepare_accelerator(args: argparse.Namespace):
    """
    this function also prepares deepspeed plugin
    """
    import time

    if args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    dynamo_backend = "NO"
    if args.torch_compile:
        dynamo_backend = args.dynamo_backend

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_backend=dynamo_backend,
        deepspeed_plugin=deepspeed_plugin,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def patch_accelerator_for_fp16_training(accelerator):

    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer
