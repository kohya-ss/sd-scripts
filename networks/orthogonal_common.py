# Shared helpers for orthogonal adapter network modules.

import ast
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]
UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]

LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"


def str_to_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y", "on")


def str_to_int(value, default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    return int(value)


def str_to_float(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    return float(value)


def parse_patterns(value) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [str(value)]
    return parsed if isinstance(parsed, list) else [parsed]


def text_encoder_list(text_encoder) -> List[torch.nn.Module]:
    if text_encoder is None:
        return []
    if isinstance(text_encoder, list):
        return [te for te in text_encoder if te is not None]
    return [text_encoder]


def load_weights_sd(file, weights_sd=None) -> Dict[str, torch.Tensor]:
    if weights_sd is not None:
        return weights_sd

    if os.path.isdir(file):
        safetensors_file = os.path.join(file, "adapter_model.safetensors")
        bin_file = os.path.join(file, "adapter_model.bin")
        if os.path.exists(safetensors_file):
            file = safetensors_file
        elif os.path.exists(bin_file):
            file = bin_file
        else:
            raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {file}")

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file

        return load_file(file)

    return torch.load(file, map_location="cpu")


def save_weights_sd(file, state_dict, dtype, metadata):
    if metadata is not None and len(metadata) == 0:
        metadata = None

    if dtype is not None:
        state_dict = {k: v.detach().clone().to("cpu").to(dtype) for k, v in state_dict.items()}
    else:
        state_dict = {k: v.detach().clone().to("cpu") for k, v in state_dict.items()}

    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import save_file
        from library import train_util

        if metadata is None:
            metadata = {}
        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash
        save_file(state_dict, file, metadata)
    else:
        torch.save(state_dict, file)


def is_linear_module(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "Linear"


def is_conv2d_module(module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "Conv2d"


def is_conv2d_1x1(module: torch.nn.Module) -> bool:
    return is_conv2d_module(module) and module.kernel_size == (1, 1)


def get_in_features(module: torch.nn.Module) -> int:
    if is_linear_module(module):
        return module.in_features
    if is_conv2d_module(module):
        if module.groups != 1:
            raise ValueError("Orthogonal adapters do not support grouped Conv2d layers")
        if module.dilation[0] > 1 or module.dilation[1] > 1:
            raise ValueError("Orthogonal adapters do not support Conv2d layers with dilation > 1")
        return module.in_channels * module.kernel_size[0] * module.kernel_size[1]
    raise TypeError(f"Unsupported module type: {module.__class__.__name__}")


def get_out_features(module: torch.nn.Module) -> int:
    if is_linear_module(module):
        return module.out_features
    if is_conv2d_module(module):
        return module.out_channels
    raise TypeError(f"Unsupported module type: {module.__class__.__name__}")


def adjust_block_size(in_features: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError("block size must be positive")
    if in_features % block_size == 0 and block_size <= in_features:
        return block_size
    if block_size >= in_features:
        return in_features

    higher = block_size
    while higher <= in_features and in_features % higher != 0:
        higher += 1

    lower = block_size
    while lower > 1 and in_features % lower != 0:
        lower -= 1

    if higher > in_features:
        return lower
    return lower if (block_size - lower) <= (higher - block_size) else higher


def triangular_block_size(n_elements: int) -> int:
    # n = b * (b - 1) / 2
    block_size = int((1 + math.sqrt(1 + 8 * n_elements)) / 2)
    if block_size * (block_size - 1) // 2 != n_elements:
        raise ValueError(f"Cannot infer OFT block size from {n_elements} triangular elements")
    return block_size


def native_prefix(root_prefix: str, original_name: str) -> str:
    return f"{root_prefix}.{original_name}".replace(".", "_")


def omi_prefix(root_prefix: str, original_name: str) -> Optional[str]:
    if root_prefix == LORA_PREFIX_UNET:
        return f"unet.{original_name}"
    if root_prefix == LORA_PREFIX_TEXT_ENCODER:
        return f"clip_l.{original_name}"
    if root_prefix == LORA_PREFIX_TEXT_ENCODER1:
        return f"clip_l.{original_name}"
    if root_prefix == LORA_PREFIX_TEXT_ENCODER2:
        return f"clip_g.{original_name}"
    return None


def candidate_module_prefixes(lora_name: str, root_prefix: str, original_name: str) -> List[str]:
    candidates = [
        lora_name,
        f"{root_prefix}.{original_name}",
    ]
    omi = omi_prefix(root_prefix, original_name)
    if omi is not None:
        candidates.append(omi)
    return candidates


def extract_module_state(
    weights_sd: Dict[str, torch.Tensor],
    lora_name: str,
    root_prefix: str,
    original_name: str,
    param_names: Iterable[str],
) -> Tuple[Dict[str, torch.Tensor], List[str]]:
    state = {}
    used = []
    keys = set(weights_sd.keys())
    prefixes = candidate_module_prefixes(lora_name, root_prefix, original_name)

    for param_name in param_names:
        found_key = None
        for prefix in prefixes:
            key = f"{prefix}.{param_name}"
            if key in keys:
                found_key = key
                break

        if found_key is None:
            suffix = f".{original_name}.{param_name}"
            for key in keys:
                if key.endswith(suffix):
                    found_key = key
                    break

        if found_key is not None:
            state[param_name] = weights_sd[found_key]
            used.append(found_key)

    return state, used


def has_any_module_state(
    weights_sd: Dict[str, torch.Tensor],
    lora_name: str,
    root_prefix: str,
    original_name: str,
    param_names: Iterable[str],
) -> bool:
    state, _ = extract_module_state(weights_sd, lora_name, root_prefix, original_name, param_names)
    return len(state) > 0
