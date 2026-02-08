# Anima model loading/saving utilities

import os
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from accelerate.utils import set_module_tensor_to_device  # kept for potential future use
from accelerate import init_empty_weights

from library.fp8_optimization_utils import apply_fp8_monkey_patch
from library.lora_utils import load_safetensors_with_lora_and_fp8
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import anima_models


# Keys that should stay in high precision (float32/bfloat16, not quantized)
KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer']


def load_safetensors(path: str, device: str = "cpu", dtype: Optional[torch.dtype] = None) -> Dict[str, torch.Tensor]:
    """Load a safetensors file and optionally cast to dtype."""
    sd = load_file(path, device=device)
    if dtype is not None:
        sd = {k: v.to(dtype) for k, v in sd.items()}
    return sd


def load_anima_dit(
    dit_path: str,
    dtype: torch.dtype,
    device: Union[str, torch.device] = "cpu",
    transformer_dtype: Optional[torch.dtype] = None,
    llm_adapter_path: Optional[str] = None,
    disable_mmap: bool = False,
) -> anima_models.MiniTrainDIT:
    """Load the MiniTrainDIT model from safetensors.

    Args:
        dit_path: Path to DiT safetensors file
        dtype: Base dtype for model parameters
        device: Device to load to
        transformer_dtype: Optional separate dtype for transformer blocks (lower precision)
        llm_adapter_path: Optional separate path for LLM adapter weights
        disable_mmap: If True, disable memory-mapped loading (reduces peak memory)
    """
    if transformer_dtype is None:
        transformer_dtype = dtype

    logger.info(f"Loading Anima DiT from {dit_path}")
    if disable_mmap:
        from library.safetensors_utils import load_safetensors as load_safetensors_no_mmap
        state_dict = load_safetensors_no_mmap(dit_path, device="cpu", disable_mmap=True)
    else:
        state_dict = load_file(dit_path, device="cpu")

    # Remove 'net.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            k = k[len('net.'):]
        new_state_dict[k] = v
    state_dict = new_state_dict

    # Derive config from state_dict
    dit_config = anima_models.get_dit_config(state_dict)

    # Detect LLM adapter
    if llm_adapter_path is not None:
        use_llm_adapter = True
        dit_config['use_llm_adapter'] = True
        llm_adapter_state_dict = load_safetensors(llm_adapter_path, device="cpu")
    elif 'llm_adapter.out_proj.weight' in state_dict:
        use_llm_adapter = True
        dit_config['use_llm_adapter'] = True
        llm_adapter_state_dict = None  # Loaded as part of DiT
    else:
        use_llm_adapter = False
        llm_adapter_state_dict = None

    logger.info(f"DiT config: model_channels={dit_config['model_channels']}, num_blocks={dit_config['num_blocks']}, "
                f"num_heads={dit_config['num_heads']}, use_llm_adapter={use_llm_adapter}")

    # Build model normally on CPU — buffers get proper values from __init__
    dit = anima_models.MiniTrainDIT(**dit_config)

    # Merge LLM adapter weights into state_dict if loaded separately
    if use_llm_adapter and llm_adapter_state_dict is not None:
        for k, v in llm_adapter_state_dict.items():
            state_dict[f"llm_adapter.{k}"] = v

    # Load checkpoint: strict=False keeps buffers not in checkpoint (e.g. pos_embedder.seq)
    missing, unexpected = dit.load_state_dict(state_dict, strict=False)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [k for k in missing if not any(
            buf_name in k for buf_name in ('seq', 'dim_spatial_range', 'dim_temporal_range', 'inv_freq')
        )]
        if unexpected_missing:
            logger.warning(f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}")
    if unexpected:
        logger.info(f"Unexpected keys in checkpoint (ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Apply per-parameter dtype (high precision for 1D/critical, transformer_dtype for rest)
    for name, p in dit.named_parameters():
        dtype_to_use = dtype if (
            any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1
        ) else transformer_dtype
        p.data = p.data.to(dtype=dtype_to_use)

    dit.to(device)
    logger.info(f"Loaded Anima DiT successfully. Parameters: {sum(p.numel() for p in dit.parameters()):,}")
    return dit


FP8_OPTIMIZATION_TARGET_KEYS = ["blocks", ""]
FP8_OPTIMIZATION_EXCLUDE_KEYS = ["_embedder", "norm", "adaln", "final_layer"]


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[Dict[str, torch.Tensor]] = None,
    lora_multipliers: Optional[list[float]] = None,
) -> anima_models.Anima:
    """
    Load a HunyuanImage model from the specified checkpoint.

    Args:
        device (Union[str, torch.device]): Device for optimization or merging
        dit_path (str): Path to the DiT model checkpoint.
        attn_mode (str): Attention mode to use, e.g., "torch", "flash", etc.
        split_attn (bool): Whether to use split attention.
        loading_device (Union[str, torch.device]): Device to load the model weights on.
        dit_weight_dtype (Optional[torch.dtype]): Data type of the DiT weights.
            If None, it will be loaded as is (same as the state_dict) or scaled for fp8. if not None, model weights will be casted to this dtype.
        fp8_scaled (bool): Whether to use fp8 scaling for the model weights.
        lora_weights_list (Optional[Dict[str, torch.Tensor]]): LoRA weights to apply, if any.
        lora_multipliers (Optional[List[float]]): LoRA multipliers for the weights, if any.
    """
    # dit_weight_dtype is None for fp8_scaled
    assert (not fp8_scaled and dit_weight_dtype is not None) or (fp8_scaled and dit_weight_dtype is None)

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # We currently support fixed DiT config for Anima models
    dit_config={'max_img_h': 512, 'max_img_w': 512, 'max_frames': 128, 'in_channels': 16, 'out_channels': 16, 'patch_spatial': 2, 'patch_temporal': 1, 'model_channels': 2048, 'concat_padding_mask': True, 'crossattn_emb_channels': 1024, 'pos_emb_cls': 'rope3d', 'pos_emb_learnable': True, 'pos_emb_interpolation': 'crop', 'min_fps': 1, 'max_fps': 30, 'use_adaln_lora': True, 'adaln_lora_dim': 256, 'num_blocks': 28, 'num_heads': 16, 'extra_per_block_abs_pos_emb': False, 'rope_h_extrapolation_ratio': 4.0, 'rope_w_extrapolation_ratio': 4.0, 'rope_t_extrapolation_ratio': 1.0, 'extra_h_extrapolation_ratio': 1.0, 'extra_w_extrapolation_ratio': 1.0, 'extra_t_extrapolation_ratio': 1.0, 'rope_enable_fps_modulation': False, 'use_llm_adapter': True, 'attn_mode': attn_mode, 'split_attn': split_attn}
    # model = create_model(attn_mode, split_attn, dit_weight_dtype)
    with init_empty_weights():
        model = anima_models.Anima(dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # load model weights with dynamic fp8 optimization and LoRA merging if needed
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")

    sd = load_safetensors_with_lora_and_fp8(
        model_files=dit_path,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        fp8_optimization=fp8_scaled,
        calc_device=device,
        move_to_device=(loading_device == device),
        dit_weight_dtype=dit_weight_dtype,
        target_keys=FP8_OPTIMIZATION_TARGET_KEYS,
        exclude_keys=FP8_OPTIMIZATION_EXCLUDE_KEYS,
    )

    if fp8_scaled:
        apply_fp8_monkey_patch(model, sd, use_scaled_mm=False)

        if loading_device.type != "cpu":
            # make sure all the model weights are on the loading_device
            logger.info(f"Moving weights to {loading_device}")
            for key in sd.keys():
                sd[key] = sd[key].to(loading_device)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        # Filter out expected missing buffers (initialized in __init__, not saved in checkpoint)
        unexpected_missing = [k for k in missing if not any(
            buf_name in k for buf_name in ('seq', 'dim_spatial_range', 'dim_temporal_range', 'inv_freq')
        )]
        if unexpected_missing:
            # Raise error to avoid silent failures
            raise RuntimeError(f"Missing keys in checkpoint: {unexpected_missing[:10]}{'...' if len(unexpected_missing) > 10 else ''}")
        missing = {}  # all missing keys were expected
    if unexpected:
        # Raise error to avoid silent failures
        raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    logger.info(f"Loaded DiT model from {dit_path}, unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

    return model



def load_anima_vae(vae_path: str, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    """Load WanVAE from a safetensors/pth file.

    Returns (vae_model, mean_tensor, std_tensor, scale).
    """
    from library.anima_models import ANIMA_VAE_MEAN, ANIMA_VAE_STD

    logger.info(f"Loading Anima VAE from {vae_path}")

    # VAE config (fixed for WanVAE)
    vae_config = dict(
        dim=96,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )

    from library.anima_vae import WanVAE_

    # Build model
    with torch.device('meta'):
        vae = WanVAE_(**vae_config)

    # Load state dict
    if vae_path.endswith('.safetensors'):
        vae_sd = load_file(vae_path, device='cpu')
    else:
        vae_sd = torch.load(vae_path, map_location='cpu', weights_only=True)

    vae.load_state_dict(vae_sd, assign=True)
    vae = vae.eval().requires_grad_(False).to(device, dtype=dtype)

    # Create normalization tensors
    mean = torch.tensor(ANIMA_VAE_MEAN, dtype=dtype, device=device)
    std = torch.tensor(ANIMA_VAE_STD, dtype=dtype, device=device)
    scale = [mean, 1.0 / std]

    logger.info(f"Loaded Anima VAE successfully.")
    return vae, mean, std, scale


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model).

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file.
                     If a directory, loads tokenizer from it directly.
                     If a file, uses configs/qwen3_06b/ for tokenizer config.
    Returns:
        tokenizer
    """
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'qwen3_06b')
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(qwen3_path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cpu"):
    """Load Qwen3-0.6B text encoder.

    Args:
        qwen3_path: Path to either a directory with model files or a safetensors file
        dtype: Model dtype
        device: Device to load to

    Returns:
        (text_encoder_model, tokenizer)
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        # Directory with full model
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path, torch_dtype=dtype, local_files_only=True
        ).model
    else:
        # Single safetensors file - use configs/qwen3_06b/ for config
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'qwen3_06b')
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        # Load weights
        if qwen3_path.endswith('.safetensors'):
            state_dict = load_file(qwen3_path, device='cpu')
        else:
            state_dict = torch.load(qwen3_path, map_location='cpu', weights_only=True)

        # Remove 'model.' prefix if present
        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_sd[k[len('model.'):]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens.

    Args:
        t5_tokenizer_path: Optional path to T5 tokenizer directory. If None, uses default configs.
    """
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    # Use bundled config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 't5_old')
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, 'spiece.model'),
            tokenizer_file=os.path.join(config_dir, 'tokenizer.json'),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json. "
        "You can download these from the google/t5-v1_1-xxl HuggingFace repository."
    )


def save_anima_model(save_path: str, dit_state_dict: Dict[str, torch.Tensor], dtype: Optional[torch.dtype] = None):
    """Save Anima DiT model with 'net.' prefix for ComfyUI compatibility.

    Args:
        save_path: Output path (.safetensors)
        dit_state_dict: State dict from dit.state_dict()
        dtype: Optional dtype to cast to before saving
    """
    prefixed_sd = {}
    for k, v in dit_state_dict.items():
        if dtype is not None:
            v = v.to(dtype)
        prefixed_sd['net.' + k] = v.contiguous()

    save_file(prefixed_sd, save_path, metadata={'format': 'pt'})
    logger.info(f"Saved Anima model to {save_path}")


def vae_encode(tensor: torch.Tensor, vae, scale):
    """Encode tensor through WanVAE with normalization.

    Args:
        tensor: Input tensor (B, C, T, H, W) in [-1, 1] range
        vae: WanVAE_ model
        scale: [mean, 1/std] list

    Returns:
        Normalized latents
    """
    return vae.encode(tensor, scale)


def vae_decode(latents: torch.Tensor, vae, scale):
    """Decode latents through WanVAE with denormalization.

    Args:
        latents: Normalized latents
        vae: WanVAE_ model
        scale: [mean, 1/std] list

    Returns:
        Decoded tensor in [-1, 1] range
    """
    return vae.decode(latents, scale)
