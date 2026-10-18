# MOSS-Image model loading and text encoding utilities

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights

from library import convrot_int8_utils, moss_image_models
from library.safetensors_utils import MemoryEfficientSafeOpen, WeightTransformHooks, get_split_weight_filenames
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# The tokenizer is downloaded from the Hub (about 20MB). The weights are never downloaded.
TOKENIZER_ID = "OpenMOSS-Team/MOSS-Image-Base"
TOKENIZER_SUBFOLDER = "tokenizer"
TEXT_ENCODER_CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "moss_image_qwen35_9b")

MAX_TOKEN_LENGTH = 4096  # maximum text length of the official pipeline
TEXT_ENCODER_HIDDEN_SIZE = 4096

# Prefixes of the DiT state dict keys which are removed on load
DIT_KEY_PREFIXES = ["model.diffusion_model.", "diffusion_model."]

# Prefixes of the text encoder keys: diffusers format (language_model.*) and ComfyUI format (model.language_model.*)
TEXT_ENCODER_KEY_PREFIXES = ["model.language_model.", "language_model."]
# Keys which are not needed for text encoding: vision tower, multi-token prediction head and LM head
TEXT_ENCODER_SKIP_PREFIXES = ["visual.", "model.visual.", "mtp.", "lm_head."]

# ConvRot INT8 scope. DiT: the Linear layers of the 42 main blocks (`layers.N.proj.{q,k,v,o,ffn_up_gate,ffn_down}`), the
# same as the ComfyUI int8 files (which fuse q/k/v). The text embedder, modulation, embedders and the head stay bf16.
DIT_CONVROT_INT8_TARGET_KEYS = ["layers."]
DIT_CONVROT_INT8_EXCLUDE_KEYS = ["norm"]
# Text encoder: the attention / GDN / MLP projections of all layers and the token embedding, the same as the ComfyUI int8
# file (`in_proj_a` / `in_proj_b` with 32 output rows and the norms stay bf16).
TEXT_ENCODER_CONVROT_INT8_TARGET_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "embed_tokens",
]
TEXT_ENCODER_CONVROT_INT8_EXCLUDE_KEYS = []


# region DiT


def _strip_dit_prefix(key: str) -> str:
    for prefix in DIT_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def split_fused_qkv(key: str, tensor: Optional[torch.Tensor], dim: int = moss_image_models.MOSS_IMAGE_CONFIG["dim"]):
    """Split hook for the ComfyUI format: `layers.N.proj.qkv.weight` [3*dim, dim] -> q / k / v weights.

    The ConvRot INT8 files of ComfyUI quantize the fused `qkv`: the int8 `.weight` [3*dim, dim] and the per-row
    `.weight_scale` [3*dim, 1] are split along the rows as well (the rotation acts on the input features, so the rows are
    independent), and the `.comfy_quant` spec is duplicated for q / k / v.

    Returns (new_keys, new_tensors). (None, None) if the key is not a fused qkv tensor. If tensor is None, only the keys
    are returned (the loader calls the hook without a tensor first).
    """
    stripped = _strip_dit_prefix(key)
    if not stripped.startswith("layers."):
        return None, None
    suffix = None
    for candidate in (".proj.qkv.weight", ".proj.qkv.weight_scale", ".proj.qkv.comfy_quant"):
        if stripped.endswith(candidate):
            suffix = candidate[len(".proj.qkv") :]  # ".weight", ".weight_scale" or ".comfy_quant"
            break
    if suffix is None:
        return None, None
    base = stripped[: -len("qkv" + suffix)]
    new_keys = [base + "q" + suffix, base + "k" + suffix, base + "v" + suffix]
    if tensor is None:
        return new_keys, None
    if suffix == ".comfy_quant":
        return new_keys, [tensor, tensor.clone(), tensor.clone()]
    assert tensor.shape[0] == 3 * dim, f"Unexpected fused qkv tensor shape {tuple(tensor.shape)} for {key}"
    q, k, v = torch.split(tensor, dim, dim=0)
    return new_keys, [q.contiguous(), k.contiguous(), v.contiguous()]


def _cast_floating_tensors(sd: Dict[str, torch.Tensor], dtype: Optional[torch.dtype]) -> None:
    """Cast the floating point tensors of a (partially) quantized state dict in place, keeping the fp32 `.scale_weight`."""
    if dtype is None:
        return
    for key, value in sd.items():
        if value.is_floating_point() and not key.endswith(".scale_weight"):
            sd[key] = value.to(dtype)


def load_moss_image_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
    disable_mmap: bool = False,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
) -> moss_image_models.MossImage:
    """Load the MOSS-Image DiT.

    Both the official diffusers shards (specify the first file, `diffusion_pytorch_model-00001-of-00003.safetensors`)
    and the ComfyUI single file (fused qkv) are supported. LoRA weights (sd-scripts format, `lora_unet_*`) can be merged
    on load.

    ConvRot INT8 (`convrot_int8=True`) quantizes the Linear layers of the main blocks at load time (after the LoRA merge).
    ComfyUI pre-quantized ConvRot INT8 files (`*_int8_convrot.safetensors`) are detected automatically and loaded as they
    are (LoRA cannot be merged into them). The patched model is frozen (`requires_grad=False`).

    Args:
        device: device used for LoRA merging and quantization
        dit_path: path to the checkpoint (first shard or single file)
        attn_mode: "torch", "flash", "sageattn" or "xformers"
        split_attn: process each sample separately in attention (saves memory for variable length batches)
        loading_device: device to load the weights to
        dit_weight_dtype: dtype of the weights. None loads the weights as is (bf16 for the official checkpoints). With
            ConvRot INT8, the dtype of the layers which are not quantized
        convrot_int8: quantize the base weights to ConvRot INT8
        convrot_int8_bwd: backward mode of the ConvRot INT8 layers, "bf16" (transient dequantization) or "int8"
    """
    # imported here: lora_utils depends on the `networks` package, which is not importable from tools/ scripts
    from library.lora_utils import load_safetensors_with_lora_and_fp8

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    files = get_split_weight_filenames(dit_path) or [dit_path]
    prequantized = convrot_int8_utils.has_comfy_quant_tensors(files, disable_numpy_memmap=disable_mmap)
    if prequantized and not convrot_int8:
        logger.info("Pre-quantized ConvRot INT8 checkpoint detected: the quantized layers are used as they are")
    use_convrot_int8 = convrot_int8 or prequantized

    with init_empty_weights():
        model = moss_image_models.MossImage(**moss_image_models.MOSS_IMAGE_CONFIG, attn_mode=attn_mode, split_attn=split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(
        f"Loading MOSS-Image DiT from {dit_path}, device={loading_device}, dtype={dit_weight_dtype}"
        + (" (ConvRot INT8)" if use_convrot_int8 else "")
    )

    hooks = WeightTransformHooks()
    hooks.rename_hook = _strip_dit_prefix
    hooks.split_hook = split_fused_qkv

    quantizer = None
    if use_convrot_int8:
        # an empty target list only converts the pre-quantized layers of the file
        quantizer = convrot_int8_utils.ConvRotInt8Quantizer(
            DIT_CONVROT_INT8_TARGET_KEYS if convrot_int8 else [], DIT_CONVROT_INT8_EXCLUDE_KEYS
        )

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=False,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=None if use_convrot_int8 else dit_weight_dtype,
        disable_numpy_memmap=disable_mmap,
        weight_transform_hooks=hooks,
        quantizer=quantizer,
    )

    # `rope.bands` is a derived constant: discard the checkpoint value and keep the recomputed buffer
    sd.pop("rope.bands", None)

    if use_convrot_int8:
        _cast_floating_tensors(sd, dit_weight_dtype)
        convrot_int8_utils.apply_convrot_int8_monkey_patch(
            model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=quantizer.module_groupsizes
        )
        # int8 tensors cannot require grad, and load_state_dict(assign=True) re-wraps the incoming tensors with the meta
        # parameters' requires_grad (default True). The base weights are frozen for LoRA training anyway.
        model.requires_grad_(False)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if k != "rope.bands"]
    if missing:
        raise RuntimeError(f"Missing keys in checkpoint: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:10]}{'...' if len(unexpected) > 10 else ''}")

    if loading_device.type != "cpu":
        model.rope.bands = model.rope._default_bands(loading_device)

    logger.info(f"Loaded MOSS-Image DiT from {dit_path}")
    return model


# endregion


# region Text Encoder


def load_text_encoder_config():
    from transformers import Qwen3_5TextConfig

    config_path = os.path.join(TEXT_ENCODER_CONFIG_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Text encoder config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config = Qwen3_5TextConfig(**config_dict)
    config.use_cache = False
    return config


def _strip_text_encoder_key(key: str) -> Optional[str]:
    """Return the key without the prefix, or None if the key is not part of the text model."""
    for prefix in TEXT_ENCODER_SKIP_PREFIXES:
        if key.startswith(prefix):
            return None
    for prefix in TEXT_ENCODER_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def load_text_encoder(
    ckpt_path: str,
    dtype: Optional[torch.dtype] = torch.bfloat16,
    device: Union[str, torch.device] = "cpu",
    disable_mmap: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
    convrot_int8: bool = False,
    convrot_int8_bwd: str = "bf16",
    calc_device: Optional[Union[str, torch.device]] = None,
):
    """Load the Qwen3.5-9B text encoder (text model only, the vision tower is not loaded).

    Both the diffusers shards (specify the first file, `model-00001-of-00004.safetensors`) and the ComfyUI single file
    are supported. The config is always built from `configs/moss_image_qwen35_9b/config.json`.

    LoRA weights for the text encoder can be merged on load. The keys must be relative to the text model without the
    `lora_te_` prefix (e.g. `layers_0_self_attn_q_proj.lora_down.weight`): strip the prefix beforehand.

    ConvRot INT8 (`convrot_int8=True`) quantizes the projections of all layers and the token embedding at load time
    (after the LoRA merge), the same scope as the ComfyUI `qwen3.5_9b_int8_convrot.safetensors`, which is detected
    automatically and loaded as it is (LoRA cannot be merged into it). The patched model is frozen. `calc_device` is the
    device used for the quantization (and the LoRA merge) when it differs from `device` (e.g. quantize on the GPU, keep the
    weights on the CPU).
    """
    from transformers import Qwen3_5TextModel

    device = torch.device(device)
    calc_device = device if calc_device is None else torch.device(calc_device)
    config = load_text_encoder_config()

    files = get_split_weight_filenames(ckpt_path) or [ckpt_path]
    prequantized = convrot_int8_utils.has_comfy_quant_tensors(files, disable_numpy_memmap=disable_mmap)
    if prequantized and not convrot_int8:
        logger.info("Pre-quantized ConvRot INT8 text encoder detected: the quantized layers are used as they are")
    use_convrot_int8 = convrot_int8 or prequantized

    logger.info(
        f"Loading MOSS-Image text encoder (Qwen3.5-9B) from {ckpt_path}, device={device}, dtype={dtype}"
        + (" (ConvRot INT8)" if use_convrot_int8 else "")
    )
    with init_empty_weights():
        # Do not cast the model here: the weights are assigned from the state dict in the requested dtype, and the
        # non-persistent buffers (rotary inv_freq) must stay in float32 like `from_pretrained`.
        model = Qwen3_5TextModel(config)

    weight_hook = None
    list_of_lora_weight_keys = []
    if lora_weights_list:
        # imported here: lora_utils depends on the `networks` package, which is not importable from tools/ scripts
        from library.lora_utils import create_lora_merge_hook

        logger.info(f"Merging LoRA weights into the text encoder. multipliers: {lora_multipliers}")
        weight_hook, list_of_lora_weight_keys, _ = create_lora_merge_hook(lora_weights_list, lora_multipliers, calc_device)

    if use_convrot_int8:
        # the quantizer streams the files itself: the prefixes are removed by the rename hook, the vision tower etc. are
        # dropped by the key filter. The LoRA merge (float32) happens before the quantization; the layers which are not
        # quantized are cast back to `dtype` afterwards.
        hooks = WeightTransformHooks()
        hooks.rename_hook = lambda key: _strip_text_encoder_key(key) or key
        quantizer = convrot_int8_utils.ConvRotInt8Quantizer(
            TEXT_ENCODER_CONVROT_INT8_TARGET_KEYS if convrot_int8 else [], TEXT_ENCODER_CONVROT_INT8_EXCLUDE_KEYS
        )
        sd = quantizer.load_and_quantize(
            files,
            calc_device=calc_device,
            move_to_device=(calc_device == device),
            weight_hook=weight_hook,
            disable_numpy_memmap=disable_mmap,
            weight_transform_hooks=hooks,
            key_filter=lambda key: _strip_text_encoder_key(key) is not None,
        )
        _cast_floating_tensors(sd, dtype)
        convrot_int8_utils.apply_convrot_int8_monkey_patch(
            model, sd, bwd_mode=convrot_int8_bwd, groupsize_map=quantizer.module_groupsizes, embedding_dtype=dtype
        )
        # int8 tensors cannot require grad (see load_moss_image_model)
        model.requires_grad_(False)
    else:
        sd = {}
        for file in files:
            with MemoryEfficientSafeOpen(file, disable_numpy_memmap=disable_mmap) as f:
                for key in f.keys():
                    new_key = _strip_text_encoder_key(key)
                    if new_key is None:
                        continue
                    tensor = f.get_tensor(key, device=device, dtype=dtype)
                    if weight_hook is not None:
                        # the merge is computed in the LoRA dtype (float32): cast back to the text encoder dtype
                        tensor = weight_hook(new_key, tensor, keep_on_calc_device=False).to(device, tensor.dtype)
                    sd[new_key] = tensor

    for lora_weight_keys in list_of_lora_weight_keys:
        if len(lora_weight_keys) > 0:
            logger.warning(f"Warning: not all LoRA keys are used: {', '.join(sorted(lora_weight_keys))}")

    info = model.load_state_dict(sd, strict=True, assign=True)
    logger.info(f"Loaded MOSS-Image text encoder: {info}")

    # non-persistent buffers (rotary inv_freq) were created on CPU by init_empty_weights
    model.to(device)
    model.eval()
    return model


def load_tokenizer(tokenizer_cache_dir: Optional[str] = None):
    """Load the Qwen3.5 tokenizer.

    If `tokenizer_cache_dir` contains `tokenizer_config.json` (a local tokenizer directory), it is loaded from there.
    Otherwise it is downloaded from the Hub (`OpenMOSS-Team/MOSS-Image-Base`, subfolder `tokenizer`) using
    `tokenizer_cache_dir` as the Hugging Face cache directory.
    """
    from transformers import AutoTokenizer

    if tokenizer_cache_dir is not None and os.path.exists(os.path.join(tokenizer_cache_dir, "tokenizer_config.json")):
        logger.info(f"Loading tokenizer from local directory {tokenizer_cache_dir}")
        return AutoTokenizer.from_pretrained(tokenizer_cache_dir, local_files_only=True)

    logger.info(f"Loading tokenizer from {TOKENIZER_ID} (subfolder {TOKENIZER_SUBFOLDER}), cache_dir={tokenizer_cache_dir}")
    return AutoTokenizer.from_pretrained(TOKENIZER_ID, subfolder=TOKENIZER_SUBFOLDER, cache_dir=tokenizer_cache_dir)


def format_prompt(prompt: str) -> str:
    """Apply the chat template used by the official pipeline (`add_generation_prompt=True, enable_thinking=True`)."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def tokenize_prompts(tokenizer, prompts: List[str], max_length: int = MAX_TOKEN_LENGTH) -> List[torch.Tensor]:
    """Tokenize prompts without padding. Returns a list of 1D int64 tensors (truncated to max_length)."""
    input_ids_list = []
    for prompt in prompts:
        inputs = tokenizer(format_prompt(prompt), return_tensors="pt", truncation=True, max_length=max_length)
        input_ids_list.append(inputs["input_ids"][0])
    return input_ids_list


def encode_tokens(text_encoder, input_ids_list: List[torch.Tensor], device: Optional[torch.device] = None) -> List[torch.Tensor]:
    """Encode tokenized prompts one by one (no padding, to avoid padding effects in Qwen GDN layers).

    Returns a list of `last_hidden_state` tensors [L_i, hidden_size] in the text encoder dtype.
    """
    if device is None:
        device = next(text_encoder.parameters()).device
    outputs = []
    for input_ids in input_ids_list:
        input_ids = input_ids.reshape(1, -1).to(device)
        output = text_encoder(input_ids=input_ids, use_cache=False, return_dict=True)
        outputs.append(output.last_hidden_state[0])
    return outputs


def pad_text_embeds(embeds_list: List[torch.Tensor], max_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of [L_i, C] tensors to [B, L, C] and return the mask [B, L] (int32, 1 = valid)."""
    length = max(e.shape[0] for e in embeds_list)
    if max_length is not None:
        length = max(length, max_length)
    first = embeds_list[0]
    embeds = first.new_zeros(len(embeds_list), length, first.shape[-1])
    mask = torch.zeros(len(embeds_list), length, dtype=torch.int32, device=first.device)
    for i, e in enumerate(embeds_list):
        embeds[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = 1
    return embeds, mask


# endregion
