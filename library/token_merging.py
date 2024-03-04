# based on:
#   https://github.com/ethansmith2000/ImprovedTokenMerge
#   https://github.com/ethansmith2000/comfy-todo (MIT)

import math

import torch
import torch.nn.functional as F

from library.sdxl_original_unet import SdxlUNet2DConditionModel

from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1).permute(0, 3, 1, 2)
    item = F.interpolate(item, size=(new_h, new_w), mode=method).permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item


def compute_merge(x: torch.Tensor, tome_info: dict):
    original_h, original_w = tome_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    cur_h = original_h // downsample
    cur_w = original_w // downsample
    downsample_factor_1 = tome_info["args"]["downsample_factor_depth_1"]
    downsample_factor_2 = tome_info["args"]["downsample_factor_depth_2"]

    merge_op = lambda x: x
    if downsample == 1 and downsample_factor_1 > 1:
        new_h = int(cur_h / downsample_factor_1)
        new_w = int(cur_w / downsample_factor_1)
        merge_op = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h, tome_info["args"]["downsample_method"])
    elif downsample == 2 and downsample_factor_2 > 1:
        new_h = int(cur_h / downsample_factor_2)
        new_w = int(cur_w / downsample_factor_2)
        merge_op = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h, tome_info["args"]["downsample_method"])

    return merge_op


def hook_tome_model(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._tome_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_attention(attn: torch.nn.Module):
    """ Adds a forward pre hook to downsample attention keys and values. This hook can be removed with remove_patch. """
    def hook(module, args, kwargs):
        hidden_states = args[0]
        m = compute_merge(hidden_states, module._tome_info)
        kwargs["context"] = m(hidden_states)
        return args, kwargs

    attn._tome_info["hooks"].append(attn.register_forward_pre_hook(hook, with_kwargs=True))


def parse_todo_args(args, is_sdxl: bool) -> dict:
    if len(args.todo_factor) > 2:
        raise ValueError(f"--todo_factor expects 1 or 2 arguments, received {len(args.todo_factor)}")
    elif is_sdxl and len(args.todo_factor) > 1:
        raise ValueError(f"--todo_factor expects expects exactly 1 argument for SDXL, received {len(args.todo_factor)}")

    todo_kwargs = {
        "downsample_factor_depth_1": 1,
        "downsample_factor_depth_2": 1,
        "downsample_method": "nearest-exact",
    }

    if is_sdxl:
        # SDXL doesn't have depth 1, so default to depth 2
        todo_kwargs["downsample_factor_depth_2"] = args.todo_factor[0]
    else:
        todo_kwargs["downsample_factor_depth_1"] = args.todo_factor[0]
        todo_kwargs["downsample_factor_depth_2"] = args.todo_factor[1] if len(args.todo_factor) == 2 else 1

    if args.todo_args:
        for arg in args.todo_args:
            key, value = arg.split("=")
            todo_kwargs[key] = value
    todo_kwargs["downsample_factor_depth_1"] = float(todo_kwargs["downsample_factor_depth_1"])
    todo_kwargs["downsample_factor_depth_2"] = float(todo_kwargs["downsample_factor_depth_2"])

    logger.info(f"enable token downsampling optimization | {todo_kwargs}")

    return todo_kwargs


def patch_attention(unet: torch.nn.Module, args):
    """ Patches the UNet's transformer blocks to apply token downsampling. """
    is_sdxl = isinstance(unet, SdxlUNet2DConditionModel)
    todo_kwargs = parse_todo_args(args, is_sdxl)

    unet._tome_info = {
        "size": None,
        "hooks": [],
        "args": todo_kwargs,
    }
    hook_tome_model(unet)

    for _, module in unet.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            module.attn1._tome_info = unet._tome_info
            hook_attention(module.attn1)

    return unet


def remove_patch(unet: torch.nn.Module):
    if hasattr(unet, "_tome_info"):
        for hook in unet._tome_info["hooks"]:
            hook.remove()
        unet._tome_info["hooks"].clear()

    return unet
