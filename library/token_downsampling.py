# based on:
#   https://github.com/ethansmith2000/ImprovedTokenMerge
#   https://github.com/ethansmith2000/comfy-todo (MIT)

import math

import torch
import torch.nn.functional as F


def up_or_downsample(item, cur_w, cur_h, new_w, new_h, method="nearest-exact"):
    batch_size = item.shape[0]

    item = item.reshape(batch_size, cur_h, cur_w, -1).permute(0, 3, 1, 2)
    item = F.interpolate(item, size=(new_h, new_w), mode=method).permute(0, 2, 3, 1)
    item = item.reshape(batch_size, new_h * new_w, -1)

    return item


def compute_merge(x: torch.Tensor, todo_info: dict):
    original_h, original_w = todo_info["size"]
    original_tokens = original_h * original_w
    downsample = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    cur_h = original_h // downsample
    cur_w = original_w // downsample

    args = todo_info["args"]
    downsample_factor = args["downsample_factor"]

    merge_op = lambda x: x
    if downsample <= args["max_depth"]:
        new_h = int(cur_h / downsample_factor)
        new_w = int(cur_w / downsample_factor)
        merge_op = lambda x: up_or_downsample(x, cur_w, cur_h, new_w, new_h)

    return merge_op


def hook_unet(model: torch.nn.Module):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        module._todo_info["size"] = (args[0].shape[2], args[0].shape[3])
        return None

    model._todo_info["hooks"].append(model.register_forward_pre_hook(hook))


def hook_attention(attn: torch.nn.Module):
    """ Adds a forward pre hook to downsample attention keys and values. This hook can be removed with remove_patch. """
    def hook(module, args, kwargs):
        hidden_states = args[0]
        m = compute_merge(hidden_states, module._todo_info)
        kwargs["context"] = m(hidden_states)
        return args, kwargs

    attn._todo_info["hooks"].append(attn.register_forward_pre_hook(hook, with_kwargs=True))


def parse_todo_args(args, is_sdxl: bool = False) -> dict:
    if args.todo_max_depth is None:
        args.todo_max_depth = 2 if is_sdxl else 1
    if is_sdxl and args.todo_max_depth not in (2, 3):
        raise ValueError(f"--todo_max_depth for SDXL must be 2 or 3, received {args.todo_factor}")

    todo_kwargs = {
        "downsample_factor": args.todo_factor,
        "max_depth": 2**(args.todo_max_depth - 1),
    }

    return todo_kwargs


def apply_patch(unet: torch.nn.Module, args, is_sdxl=False):
    """ Patches the UNet's transformer blocks to apply token downsampling. """
    todo_kwargs = parse_todo_args(args, is_sdxl)

    unet._todo_info = {
        "size": None,
        "hooks": [],
        "args": todo_kwargs,
    }
    hook_unet(unet)

    for _, module in unet.named_modules():
        if module.__class__.__name__ == "BasicTransformerBlock":
            module.attn1._todo_info = unet._todo_info
            hook_attention(module.attn1)

    return unet


def remove_patch(unet: torch.nn.Module):
    if hasattr(unet, "_todo_info"):
        for hook in unet._todo_info["hooks"]:
            hook.remove()
        unet._todo_info["hooks"].clear()

    return unet
