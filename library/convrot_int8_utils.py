"""ConvRot INT8 optimization for frozen base weights during LoRA training (and inference).

Ported from Musubi Tuner (``modules/convrot_int8_utils.py`` + ``modules/comfy_quant_utils.py``).

Mirrors the structure of fp8_optimization_utils.py: quantize target Linear weights at load time, store the INT8 weight
(in the rotated basis) under the original ``.weight`` key and the per-channel scale under a sibling ``.scale_weight``
key, then monkey-patch the matching ``nn.Linear`` modules' ``forward``.

Unlike the fp8 path, the forward goes through a custom ``torch.autograd.Function``: the fused Triton kernel (rotation +
dynamic row-wise INT8 quantization + INT8 GEMM with dequantization epilogue) has no autograd support, and the base weight
is frozen so only grad_x is needed in backward: grad_x = rotate(g @ W_rot).

Keeping the module an ``nn.Linear`` (patched forward, INT8 ``.weight``) is load-bearing: LoRA targets modules by class
name "Linear", block swap streams ``module.weight.data`` of Linear-named modules, and compile exclusion also keys on the
class name.

ComfyUI pre-quantized checkpoints (``.weight`` int8 + ``.weight_scale`` fp32 [N, 1] + ``.comfy_quant`` JSON spec) are
loaded as they are. ComfyUI also quantizes ``nn.Embedding`` weights (token embeddings of the text encoders) with the same
row-wise ConvRot INT8 recipe: those modules get a dequantizing lookup (``int8 row * scale`` rotated back to the original
basis).
"""

import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from library.convrot_int8_kernels import (
    HAS_TRITON,
    _build_hadamard,
    _rotate_activation,
    int8_linear,
    quantize_int8_convrot_weight,
)
from library.device_utils import clean_memory_on_device
from library.safetensors_utils import MemoryEfficientSafeOpen, TensorWeightAdapter, WeightTransformHooks
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

CONVROT_GROUPSIZE = 256

# ComfyUI pre-quantized layers: `.weight` (int8, rotated basis) + `.weight_scale` (fp32 [N, 1]) + `.comfy_quant`
# (uint8 bytes of a JSON spec). `.comfy_attention.config` is a per-block runtime hint of ComfyUI (attention backend),
# not a weight: it is skipped on load.
COMFY_QUANT_SUFFIX = ".comfy_quant"
COMFY_WEIGHT_SCALE_SUFFIX = ".weight_scale"
COMFY_ATTENTION_CONFIG_SUFFIX = ".comfy_attention.config"
COMFY_QUANT_FORMAT_INT8 = "int8_tensorwise"

# Rows per chunk when quantizing very tall matrices (token embeddings: 248k x 4096). The quantization is row-wise, so
# chunking is exact; it only bounds the fp32 transient (two [rows, K] fp32 copies) on the calc device.
QUANTIZE_ROW_CHUNK = 16384


def _is_power_of_4(value: int) -> bool:
    return value >= 4 and (value & (value - 1)) == 0 and math.log(value, 4) % 1 == 0


# region ComfyUI comfy_quant spec


def decode_comfy_quant_spec(key: str, tensor: torch.Tensor) -> dict:
    """Decode a ``.comfy_quant`` tensor (uint8 bytes of JSON) into a dict, format-agnostic."""
    if tensor.dtype != torch.uint8 or tensor.ndim != 1:
        raise ValueError(f"Invalid comfy_quant tensor for {key}: expected 1D uint8, got {tensor.dtype} ndim={tensor.ndim}")
    try:
        spec = json.loads(bytes(tensor.tolist()).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid comfy_quant JSON for {key}: {e}") from e
    if not isinstance(spec, dict):
        raise ValueError(f"Invalid comfy_quant spec for {key}: expected a JSON object, got {type(spec).__name__}")
    return spec


def parse_comfy_quant_spec(key: str, tensor: torch.Tensor) -> dict:
    """Decode and validate a ``.comfy_quant`` JSON spec tensor (uint8 bytes).

    Only ConvRot INT8 (``int8_tensorwise`` + ``convrot``) is supported; other formats (e.g. nvfp4, fp8) raise with a clear
    message.
    """
    spec = decode_comfy_quant_spec(key, tensor)
    quant_format = spec.get("format")
    if quant_format != COMFY_QUANT_FORMAT_INT8 or not spec.get("convrot"):
        raise ValueError(
            f"Unsupported comfy_quant format for {key}: {spec}. Only ConvRot INT8"
            f' ("{COMFY_QUANT_FORMAT_INT8}" with "convrot": true) is supported.'
            f" / {key} の comfy_quant 形式はサポートされていません。ConvRot INT8 のみ対応しています。"
        )
    groupsize = spec.get("convrot_groupsize")
    if not isinstance(groupsize, int) or not _is_power_of_4(groupsize):
        raise ValueError(f"Invalid convrot_groupsize for {key}: {groupsize!r} (must be a power of 4, e.g. 64 or 256)")
    return spec


def has_comfy_quant_tensors(files: Iterable[Union[str, Path]], *, disable_numpy_memmap: bool = False) -> bool:
    """Header-only probe: does the checkpoint declare ComfyUI pre-quantized layers?

    Routing only: validation and conversion of the ``.comfy_quant`` triples happen in
    ``ConvRotInt8Quantizer.load_and_quantize``.
    """
    for file in files:
        with MemoryEfficientSafeOpen(str(file), disable_numpy_memmap=disable_numpy_memmap) as f:
            if any(key.endswith(COMFY_QUANT_SUFFIX) for key in f.keys()):
                return True
    return False


def canonicalize_convrot_int8_key(key: str) -> str:
    """Map the ComfyUI ``.weight_scale`` suffix to the ``.scale_weight`` layout used by the monkey patch."""
    if key.endswith(COMFY_WEIGHT_SCALE_SUFFIX):
        return key[: -len(COMFY_WEIGHT_SCALE_SUFFIX)] + ".scale_weight"
    return key


# endregion


# region quantization


def select_convrot_groupsize(in_features: int, allowed_groupsizes: Sequence[int]) -> Optional[int]:
    """Pick the largest allowed group size that divides ``in_features``, or None.

    The regular Hadamard construction only exists for power-of-4 sizes, so allowing a smaller fallback (e.g. 64) extends
    coverage to layers whose in_features is not a multiple of 256. This rule reproduces ComfyUI's published group size
    choices.
    """
    for groupsize in sorted(allowed_groupsizes, reverse=True):
        if in_features % groupsize == 0:
            return groupsize
    return None


def quantize_weight_convrot(key: str, tensor: torch.Tensor, allowed_groupsizes: Sequence[int] = (CONVROT_GROUPSIZE,)):
    """Quantize a single weight tensor with ConvRot INT8, or return None if not applicable.

    The quantization is always done with the eager implementation: it is deterministic and independent of triton
    availability, so the resulting state dict is identical across environments (and matches the ComfyUI files up to a
    handful of rounding ties).

    Returns:
        (quantized int8 [N, K] in rotated basis, scale float32 [N, 1], groupsize), or None if the tensor is not a 2D weight
        or its in_features is not divisible by any allowed groupsize.
    """
    if tensor.ndim != 2:
        logger.info(f"Skipping ConvRot INT8 for {key}: not a 2D weight (ndim={tensor.ndim})")
        return None
    groupsize = select_convrot_groupsize(tensor.shape[1], allowed_groupsizes)
    if groupsize is None:
        logger.info(f"Skipping ConvRot INT8 for {key}: in_features {tensor.shape[1]} not divisible by any of {allowed_groupsizes}")
        return None
    if tensor.shape[0] <= QUANTIZE_ROW_CHUNK:
        quantized_weight, scale_tensor = quantize_int8_convrot_weight(tensor, groupsize)
    else:
        # row-wise quantization is independent per row: chunk tall matrices to bound the fp32 transient
        quantized_chunks, scale_chunks = [], []
        for start in range(0, tensor.shape[0], QUANTIZE_ROW_CHUNK):
            q, s = quantize_int8_convrot_weight(tensor[start : start + QUANTIZE_ROW_CHUNK], groupsize)
            quantized_chunks.append(q)
            scale_chunks.append(s)
        quantized_weight = torch.cat(quantized_chunks, dim=0)
        scale_tensor = torch.cat(scale_chunks, dim=0)
    return quantized_weight, scale_tensor, groupsize


class ConvRotInt8Quantizer:
    """Strategy object that streams safetensors files and quantizes target weights.

    Passed as ``quantizer`` to ``lora_utils.load_safetensors_with_lora_and_fp8``; carries its own streaming loader so the
    fp8 path stays untouched.

    Two sources are supported per tensor, decided on the fly:

    - bf16/fp16/fp32 target weights are dynamically quantized (rotation + row-wise INT8), picking the largest
      ``allowed_groupsizes`` entry that divides in_features.
    - Pre-quantized ComfyUI layers (int8 ``.weight`` + ``.weight_scale`` + ``.comfy_quant``) are converted in place: the
      scale is renamed to ``.scale_weight`` and the group size is taken from the JSON spec. The file dictates which layers
      these are, independent of the target/exclude patterns.

    ``module_groupsizes`` maps module paths to their group size after loading; pass it to
    ``apply_convrot_int8_monkey_patch`` so mixed group sizes dispatch correctly.
    """

    def __init__(
        self,
        target_layer_keys: Optional[List[str]] = None,
        exclude_layer_keys: Optional[List[str]] = None,
        allowed_groupsizes: Sequence[int] = (CONVROT_GROUPSIZE,),
    ):
        for groupsize in allowed_groupsizes:
            if not _is_power_of_4(groupsize):
                raise ValueError(f"ConvRot group sizes must be powers of 4, got {groupsize}")
        self.target_layer_keys = target_layer_keys
        self.exclude_layer_keys = exclude_layer_keys
        self.allowed_groupsizes = tuple(allowed_groupsizes)
        self.module_groupsizes: Dict[str, int] = {}
        self.optimized_count = 0
        self.prequantized_count = 0

    def is_target_key(self, key: str) -> bool:
        """``target_layer_keys=None`` targets every ``.weight``; an empty list disables dynamic quantization."""
        is_target = (self.target_layer_keys is None or any(pattern in key for pattern in self.target_layer_keys)) and key.endswith(
            ".weight"
        )
        is_excluded = self.exclude_layer_keys is not None and any(pattern in key for pattern in self.exclude_layer_keys)
        return is_target and not is_excluded

    def load_and_quantize(
        self,
        model_files: List[str],
        calc_device: Union[str, torch.device, None],
        move_to_device: bool = False,
        weight_hook: Optional[Callable] = None,
        disable_numpy_memmap: bool = False,
        weight_transform_hooks: Optional[WeightTransformHooks] = None,
        key_filter: Optional[Callable[[str], bool]] = None,
    ) -> dict:
        """Load state dict from safetensors files, quantizing target weights to ConvRot INT8.

        Same streaming contract as ``load_safetensors_with_fp8_optimization``: the LoRA merge ``weight_hook`` runs on the
        raw (bf16) weight before quantization. Pre-quantized ComfyUI ConvRot layers are converted instead; they cannot be
        combined with a LoRA merge hook (INT8 weights cannot be merged into).

        ``key_filter(key) -> bool`` (keys after ``weight_transform_hooks``) drops tensors which are not part of the model
        (e.g. the vision tower of a VLM checkpoint) before they are read.
        """
        optimized_count = 0
        prequantized_count = 0
        state_dict = {}
        prequantized_groupsizes: Dict[str, int] = {}  # spans all shards
        for model_file in model_files:
            with MemoryEfficientSafeOpen(model_file, disable_numpy_memmap=disable_numpy_memmap) as original_f:
                f = TensorWeightAdapter(weight_transform_hooks, original_f) if weight_transform_hooks is not None else original_f

                keys = [key for key in f.keys() if not key.endswith(COMFY_ATTENTION_CONFIG_SUFFIX)]
                if key_filter is not None:
                    keys = [key for key in keys if key_filter(key)]

                # Pre-scan the tiny `.comfy_quant` spec tensors so the per-module group size is known before the (possibly
                # earlier-iterated) weight/scale keys arrive.
                for key in keys:
                    if key.endswith(COMFY_QUANT_SUFFIX):
                        module_path = key[: -len(COMFY_QUANT_SUFFIX)]
                        spec = parse_comfy_quant_spec(key, f.get_tensor(key))
                        prequantized_groupsizes[module_path] = spec["convrot_groupsize"]
                if prequantized_groupsizes and weight_hook is not None:
                    raise ValueError(
                        f"Cannot merge LoRA weights into pre-quantized ConvRot INT8 checkpoint {model_file}."
                        " Use the original BF16 weights to merge LoRA at load time."
                        f" / 事前量子化済みConvRot INT8チェックポイント {model_file} にはLoRAをマージできません。"
                        "BF16の元重みを使用してロード時マージしてください。"
                    )

                for key in tqdm(keys, desc=f"Loading {os.path.basename(model_file)}", unit="key", leave=False):
                    if key.endswith(COMFY_QUANT_SUFFIX):
                        continue  # consumed in the pre-scan, not a model tensor

                    value = f.get_tensor(key)
                    original_device = value.device  # usually cpu
                    passthrough_device = calc_device if (calc_device is not None and move_to_device) else original_device

                    if key.endswith(COMFY_WEIGHT_SCALE_SUFFIX):
                        module_path = key[: -len(COMFY_WEIGHT_SCALE_SUFFIX)]
                        if module_path not in prequantized_groupsizes:
                            raise ValueError(f"Found {key} without a matching {module_path}{COMFY_QUANT_SUFFIX} spec")
                        if value.dtype is not torch.float32:
                            raise ValueError(f"Pre-quantized ConvRot scale {key} must be F32, got {value.dtype}")
                        # rename to the `.scale_weight` layout; the fp32 [N, 1] shape is shared as-is
                        state_dict[module_path + ".scale_weight"] = value.to(passthrough_device)
                        continue

                    module_path = key[: -len(".weight")] if key.endswith(".weight") else None
                    if module_path is not None and module_path in prequantized_groupsizes:
                        if value.dtype != torch.int8:
                            raise ValueError(f"Pre-quantized ConvRot layer {key} must be int8, got {value.dtype}")
                        groupsize = prequantized_groupsizes[module_path]
                        if value.shape[1] % groupsize != 0:
                            raise ValueError(
                                f"Pre-quantized ConvRot layer {key}: in_features {value.shape[1]} not divisible by"
                                f" group size {groupsize}"
                            )
                        self.module_groupsizes[module_path] = groupsize
                        state_dict[key] = value.to(passthrough_device)
                        prequantized_count += 1
                        continue

                    if module_path is not None and value.dtype.itemsize == 1:
                        raise ValueError(
                            f"Layer {key} is already in {value.dtype} format but has no {COMFY_QUANT_SUFFIX} spec."
                            " Only ComfyUI ConvRot INT8 pre-quantized checkpoints or fp16/bf16/float32 weights are"
                            f" supported. / レイヤー {key} は既に{value.dtype}形式ですが {COMFY_QUANT_SUFFIX} がありません。"
                            "ComfyUI ConvRot INT8形式の事前量子化済み重み、またはFP16/BF16/Float32の重みを使用してください。"
                        )

                    if weight_hook is not None:
                        value = weight_hook(key, value, keep_on_calc_device=(calc_device is not None))

                    if not self.is_target_key(key):
                        state_dict[key] = value.to(passthrough_device)
                        continue

                    if calc_device is not None:
                        value = value.to(calc_device)

                    result = quantize_weight_convrot(key, value, self.allowed_groupsizes)
                    if result is None:
                        # leave the layer unquantized (bf16)
                        if not move_to_device:
                            value = value.to(original_device)
                        state_dict[key] = value
                        continue
                    quantized_weight, scale_tensor, groupsize = result
                    self.module_groupsizes[module_path] = groupsize

                    scale_key = key.replace(".weight", ".scale_weight")
                    assert key != scale_key, "weight key and scale key must be different"

                    if not move_to_device:
                        quantized_weight = quantized_weight.to(original_device)

                    # scale stays float32 [N, 1]: the Triton epilogue and the backward want fp32, and the shape maps 1:1 to
                    # ComfyUI's `weight_scale`
                    state_dict[key] = quantized_weight
                    state_dict[scale_key] = scale_tensor.to(device=quantized_weight.device)

                    optimized_count += 1

                    if calc_device is not None and optimized_count % 10 == 0:
                        clean_memory_on_device(calc_device)

        # every declared `.comfy_quant` spec must have received its int8 weight and fp32 [N, 1] scale; a partial triple
        # would otherwise assign-load into an unpatched module
        for module_path in prequantized_groupsizes:
            weight_key = module_path + ".weight"
            scale_key = module_path + ".scale_weight"
            missing = [key for key in (weight_key, scale_key) if key not in state_dict]
            if missing:
                raise ValueError(f"Pre-quantized ConvRot layer {module_path} is missing tensors {missing}")
            expected_scale_shape = (state_dict[weight_key].shape[0], 1)
            if tuple(state_dict[scale_key].shape) != expected_scale_shape:
                raise ValueError(
                    f"Pre-quantized ConvRot layer {module_path}: scale shape must be {expected_scale_shape},"
                    f" got {tuple(state_dict[scale_key].shape)}"
                )

        self.optimized_count += optimized_count
        self.prequantized_count += prequantized_count
        logger.info(
            f"Number of ConvRot INT8 layers: {optimized_count} dynamically quantized, {prequantized_count} loaded pre-quantized"
        )
        return state_dict


# endregion


# region forward / backward


class ConvRotInt8LinearFn(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, wq, w_scale, bias, groupsize, bwd_mode):
        # x: [..., K] bf16/fp16, wq: [N, K] int8 (rotated basis), w_scale: [N, 1] fp32
        # F.linear casts its inputs to the autocast dtype under autocast; the fused kernel bypasses F.linear, so replicate
        # that here (fp32 modulation outputs are promoted to fp32 otherwise).
        if torch.is_autocast_enabled(x.device.type):
            cast_dtype = torch.get_autocast_dtype(x.device.type)
            x = x.to(cast_dtype)
            if bias is not None:
                bias = bias.to(cast_dtype)
        if HAS_TRITON and x.is_cuda:
            out = int8_linear(x, wq, w_scale.reshape(-1), bias, x.dtype, True, groupsize)
        else:
            # eager fallback: rotation + transient dequantized matmul, no activation quantization
            h = _build_hadamard(groupsize, device=x.device, dtype=x.dtype)
            x_rot = _rotate_activation(x, h, groupsize)
            w_rot = wq.to(x.dtype) * w_scale.reshape(-1, 1).to(x.dtype)
            out = F.linear(x_rot, w_rot, bias)
        # wq/w_scale are live buffers, so saving them adds no activation memory; x is not saved (base is frozen, no
        # grad_weight needed)
        ctx.save_for_backward(wq, w_scale)
        ctx.groupsize = groupsize
        ctx.bwd_mode = bwd_mode
        ctx.bias_needs_grad = bias is not None and bias.requires_grad
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        wq, w_scale = ctx.saved_tensors
        gs = ctx.groupsize
        g2d = grad_out.reshape(-1, grad_out.shape[-1])  # [M, N]

        grad_x = None
        if ctx.needs_input_grad[0]:
            # grad_x = g @ W = g @ (W_rot R) = rotate(g @ W_rot), R = block-diag Hadamard
            if ctx.bwd_mode == "int8":
                if not g2d.is_cuda:
                    raise RuntimeError("ConvRot INT8 backward mode 'int8' requires CUDA tensors")
                # fold per-channel weight scale into g, then reuse the fused Triton GEMM (row-wise quant of g + int8 GEMM +
                # dequant epilogue in one pipeline). transient int8 transpose of wq: [K, N], ~1 byte/param, freed after mm
                g_scaled = g2d * w_scale.reshape(1, -1).to(g2d.dtype)
                one = torch.ones(1, device=g2d.device, dtype=torch.float32)
                gx_rot = int8_linear(g_scaled, wq.t().contiguous(), one, None, grad_out.dtype, False, gs)
            else:
                # transient bf16 dequant of the rotated weight (stays in rotated basis)
                w_rot = wq.to(grad_out.dtype) * w_scale.reshape(-1, 1).to(grad_out.dtype)
                gx_rot = g2d @ w_rot  # [M, K]
            h = _build_hadamard(gs, device=gx_rot.device, dtype=gx_rot.dtype)
            grad_x = _rotate_activation(gx_rot, h, gs).reshape(*grad_out.shape[:-1], wq.shape[1])

        grad_bias = g2d.sum(dim=0) if ctx.bias_needs_grad else None
        return grad_x, None, None, grad_bias, None, None


def convrot_int8_linear_forward_patch(self: nn.Linear, x):
    return ConvRotInt8LinearFn.apply(x, self.weight, self.scale_weight, self.bias, self._convrot_groupsize, self._convrot_bwd_mode)


def convrot_int8_embedding_forward_patch(self: nn.Embedding, input: torch.Tensor) -> torch.Tensor:
    """Dequantizing lookup for a ConvRot INT8 token embedding: ``int8 row * scale`` rotated back to the original basis.

    The rows are stored in the rotated basis like a Linear weight (W_rot = W @ H^T per group), so the lookup needs the
    inverse rotation. H is symmetric and orthogonal, hence the same block-diagonal product undoes it.
    """
    rows = self.weight[input]  # index_select works on int8; padding_idx etc. only affect training of the embedding
    dequantized = rows.float() * self.scale_weight[input]
    h = _build_hadamard(self._convrot_groupsize, device=dequantized.device, dtype=torch.float32)
    return _rotate_activation(dequantized, h, self._convrot_groupsize).to(self._convrot_embedding_dtype)


def _validate_convrot_bwd_mode(bwd_mode: str) -> None:
    if bwd_mode not in ("bf16", "int8"):
        raise ValueError(f"Unsupported ConvRot INT8 backward mode: {bwd_mode}")
    if bwd_mode == "int8" and not HAS_TRITON:
        raise ValueError("ConvRot INT8 backward mode 'int8' requires triton. Install triton (triton-windows on Windows).")
    if not HAS_TRITON:
        logger.warning(
            "triton is not available: ConvRot INT8 falls back to transient dequantization in forward."
            " Weight VRAM is still reduced, but there is no speedup. Install triton (triton-windows on Windows)"
            " for the fused INT8 kernels."
        )


def _patch_convrot_int8_linear(module: nn.Linear, groupsize: int, bwd_mode: str) -> None:
    module._convrot_groupsize = groupsize
    module._convrot_bwd_mode = bwd_mode
    module.forward = convrot_int8_linear_forward_patch.__get__(module, type(module))


def _patch_convrot_int8_embedding(module: nn.Embedding, groupsize: int, embedding_dtype: torch.dtype) -> None:
    module._convrot_groupsize = groupsize
    module._convrot_embedding_dtype = embedding_dtype
    module.forward = convrot_int8_embedding_forward_patch.__get__(module, type(module))


def apply_convrot_int8_monkey_patch(
    model: nn.Module,
    optimized_state_dict: dict,
    bwd_mode: str = "bf16",
    groupsize: int = CONVROT_GROUPSIZE,
    groupsize_map: Optional[Dict[str, int]] = None,
    embedding_dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """
    Apply monkey patching to a model using a ConvRot INT8 optimized state dict. Call before ``load_state_dict(assign=True)``
    and freeze the model (``requires_grad_(False)``) first: int8 tensors cannot require grad, and an assign load re-wraps
    the incoming tensors with the meta parameters' ``requires_grad``.

    Args:
        model (nn.Module): Model instance to patch
        optimized_state_dict (dict): state dict produced by ConvRotInt8Quantizer
        bwd_mode (str): "bf16" (transient dequant, safe default) or "int8" (quantizes gradients, faster, requires triton)
        groupsize (int): ConvRot group size for modules not listed in ``groupsize_map``
        groupsize_map (Optional[Dict[str, int]]): per-module group sizes (``ConvRotInt8Quantizer.module_groupsizes``);
            needed when a checkpoint mixes group sizes
        embedding_dtype (torch.dtype): output dtype of the dequantizing lookup of quantized ``nn.Embedding`` modules

    Returns:
        nn.Module: The patched model (same instance, modified in-place)
    """
    _validate_convrot_bwd_mode(bwd_mode)

    scale_keys = [k for k in optimized_state_dict.keys() if k.endswith(".scale_weight")]

    patched_module_paths = set()
    scale_shape_info = {}
    for scale_key in scale_keys:
        module_path = scale_key.rsplit(".scale_weight", 1)[0]
        patched_module_paths.add(module_path)
        scale_shape_info[module_path] = optimized_state_dict[scale_key].shape

    patched_paths = set()
    embedding_count = 0
    for name, module in model.named_modules():
        if name not in patched_module_paths:
            continue
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            # register the scale_weight as a buffer to load the state_dict
            module.register_buffer("scale_weight", torch.ones(scale_shape_info[name], dtype=torch.float32))
            module_groupsize = groupsize_map.get(name, groupsize) if groupsize_map is not None else groupsize
            if isinstance(module, nn.Linear):
                _patch_convrot_int8_linear(module, module_groupsize, bwd_mode)
            else:
                _patch_convrot_int8_embedding(module, module_groupsize, embedding_dtype)
                embedding_count += 1
            patched_paths.add(name)
        else:
            raise ValueError(f"ConvRot INT8 state dict declares {name}, which is neither nn.Linear nor nn.Embedding ({type(module).__name__})")

    unmatched = sorted(patched_module_paths - patched_paths)
    if unmatched:
        raise ValueError(f"ConvRot INT8 state dict declares missing module {unmatched[0]} (total {len(unmatched)} unmatched)")

    model.is_convrot_int8 = True
    model.convrot_int8_layer_count = len(patched_paths)
    logger.info(
        f"Number of ConvRot INT8 monkey-patched modules: {len(patched_paths)}"
        + (f" (including {embedding_count} embeddings)" if embedding_count else "")
    )
    return model


def is_convrot_int8_model(model: nn.Module) -> bool:
    return bool(getattr(model, "is_convrot_int8", False))


# endregion
