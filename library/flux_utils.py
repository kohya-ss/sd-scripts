import json
from typing import Optional, Union
import einops
import torch

from safetensors.torch import load_file
from accelerate import init_empty_weights
from transformers import CLIPTextModel, CLIPConfig, T5EncoderModel, T5Config

from library import flux_models

from library.utils import setup_logging, MemoryEfficientSafeOpen

setup_logging()
import logging

logger = logging.getLogger(__name__)

MODEL_VERSION_FLUX_V1 = "flux1"


# temporary copy from sd3_utils TODO refactor
def load_safetensors(
    path: str, device: Union[str, torch.device], disable_mmap: bool = False, dtype: Optional[torch.dtype] = torch.float32
):
    if disable_mmap:
        # return safetensors.torch.load(open(path, "rb").read())
        # use experimental loader
        logger.info(f"Loading without mmap (experimental)")
        state_dict = {}
        with MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key).to(device, dtype=dtype)
        return state_dict
    else:
        try:
            return load_file(path, device=device)
        except:
            return load_file(path)  # prevent device invalid Error


def load_flow_model(
    name: str, ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False
) -> flux_models.Flux:
    logger.info(f"Building Flux model {name}")
    with torch.device("meta"):
        model = flux_models.Flux(flux_models.configs[name].params)
        if dtype is not None:
            model = model.to(dtype)

    # load_sft doesn't support torch.device
    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = model.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded Flux: {info}")
    return model


def load_ae(
    name: str, ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> flux_models.AutoEncoder:
    logger.info("Building AutoEncoder")
    with torch.device("meta"):
        ae = flux_models.AutoEncoder(flux_models.configs[name].ae_params).to(dtype)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae


def load_clip_l(ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False) -> CLIPTextModel:
    logger.info("Building CLIP")
    CLIPL_CONFIG = {
        "_name_or_path": "clip-vit-large-patch14/",
        "architectures": ["CLIPModel"],
        "initializer_factor": 1.0,
        "logit_scale_init_value": 2.6592,
        "model_type": "clip",
        "projection_dim": 768,
        # "text_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "bos_token_id": 0,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0.0,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_norm_eps": 1e-05,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 77,
        "min_length": 0,
        "model_type": "clip_text_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 1,
        "prefix": None,
        "problem_type": None,
        "projection_dim": 768,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": None,
        "torchscript": False,
        "transformers_version": "4.16.0.dev0",
        "use_bfloat16": False,
        "vocab_size": 49408,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        # },
        # "text_config_dict": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "projection_dim": 768,
        # },
        # "torch_dtype": "float32",
        # "transformers_version": None,
    }
    config = CLIPConfig(**CLIPL_CONFIG)
    with init_empty_weights():
        clip = CLIPTextModel._from_config(config)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = clip.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded CLIP: {info}")
    return clip


def load_t5xxl(
    ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False
) -> T5EncoderModel:
    T5_CONFIG_JSON = """
{
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    config = json.loads(T5_CONFIG_JSON)
    config = T5Config(**config)
    with init_empty_weights():
        t5xxl = T5EncoderModel._from_config(config)

    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = t5xxl.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded T5xxl: {info}")
    return t5xxl


def get_t5xxl_actual_dtype(t5xxl: T5EncoderModel) -> torch.dtype:
    # nn.Embedding is the first layer, but it could be casted to bfloat16 or float32
    return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype


def prepare_img_ids(batch_size: int, packed_latent_height: int, packed_latent_width: int):
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    """
    x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """
    x: [b c (h ph) (w pw)] -> [b (h w) (c ph pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x
