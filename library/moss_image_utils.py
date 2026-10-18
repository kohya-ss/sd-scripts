# MOSS-Image model loading and text encoding utilities

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from accelerate import init_empty_weights

from library import moss_image_models
from library.lora_utils import load_safetensors_with_lora_and_fp8
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


# region DiT


def _strip_dit_prefix(key: str) -> str:
    for prefix in DIT_KEY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def split_fused_qkv(key: str, tensor: Optional[torch.Tensor], dim: int = moss_image_models.MOSS_IMAGE_CONFIG["dim"]):
    """Split hook for the ComfyUI format: `layers.N.proj.qkv.weight` [3*dim, dim] -> q / k / v weights.

    Returns (new_keys, new_tensors). (None, None) if the key is not a fused qkv weight. If tensor is None, only the keys
    are returned (the loader calls the hook without a tensor first).
    """
    stripped = _strip_dit_prefix(key)
    if not (stripped.startswith("layers.") and stripped.endswith(".proj.qkv.weight")):
        return None, None
    base = stripped[: -len("qkv.weight")]
    new_keys = [base + "q.weight", base + "k.weight", base + "v.weight"]
    if tensor is None:
        return new_keys, None
    assert tensor.shape[0] == 3 * dim, f"Unexpected fused qkv weight shape {tuple(tensor.shape)} for {key}"
    q, k, v = torch.split(tensor, dim, dim=0)
    return new_keys, [q.contiguous(), k.contiguous(), v.contiguous()]


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
) -> moss_image_models.MossImage:
    """Load the MOSS-Image DiT.

    Both the official diffusers shards (specify the first file, `diffusion_pytorch_model-00001-of-00003.safetensors`)
    and the ComfyUI single file (fused qkv) are supported. LoRA weights (sd-scripts format, `lora_unet_*`) can be merged
    on load.

    Args:
        device: device used for LoRA merging
        dit_path: path to the checkpoint (first shard or single file)
        attn_mode: "torch", "flash", "sageattn" or "xformers"
        split_attn: process each sample separately in attention (saves memory for variable length batches)
        loading_device: device to load the weights to
        dit_weight_dtype: dtype of the weights. None loads the weights as is (bf16 for the official checkpoints)
    """
    device = torch.device(device)
    loading_device = torch.device(loading_device)

    with init_empty_weights():
        model = moss_image_models.MossImage(**moss_image_models.MOSS_IMAGE_CONFIG, attn_mode=attn_mode, split_attn=split_attn)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    logger.info(f"Loading MOSS-Image DiT from {dit_path}, device={loading_device}, dtype={dit_weight_dtype}")

    hooks = WeightTransformHooks()
    hooks.rename_hook = _strip_dit_prefix
    hooks.split_hook = split_fused_qkv

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=False,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        disable_numpy_memmap=disable_mmap,
        weight_transform_hooks=hooks,
    )

    # `rope.bands` is a derived constant: discard the checkpoint value and keep the recomputed buffer
    sd.pop("rope.bands", None)

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
):
    """Load the Qwen3.5-9B text encoder (text model only, the vision tower is not loaded).

    Both the diffusers shards (specify the first file, `model-00001-of-00004.safetensors`) and the ComfyUI single file
    are supported. The config is always built from `configs/moss_image_qwen35_9b/config.json`.
    """
    from transformers import Qwen3_5TextModel

    device = torch.device(device)
    config = load_text_encoder_config()

    logger.info(f"Loading MOSS-Image text encoder (Qwen3.5-9B) from {ckpt_path}, device={device}, dtype={dtype}")
    with init_empty_weights():
        # Do not cast the model here: the weights are assigned from the state dict in the requested dtype, and the
        # non-persistent buffers (rotary inv_freq) must stay in float32 like `from_pretrained`.
        model = Qwen3_5TextModel(config)

    files = get_split_weight_filenames(ckpt_path) or [ckpt_path]
    sd = {}
    for file in files:
        with MemoryEfficientSafeOpen(file, disable_numpy_memmap=disable_mmap) as f:
            for key in f.keys():
                new_key = _strip_text_encoder_key(key)
                if new_key is None:
                    continue
                sd[new_key] = f.get_tensor(key, device=device, dtype=dtype)

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
