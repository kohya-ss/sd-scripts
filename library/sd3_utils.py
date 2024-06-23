import math
from typing import Dict, Optional, Union
import torch
import safetensors
from safetensors.torch import load_file
from accelerate import init_empty_weights

from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import sd3_models

# TODO move some of functions to model_util.py
from library import sdxl_model_util

# region models


def load_models(
    ckpt_path: str,
    clip_l_path: str,
    clip_g_path: str,
    t5xxl_path: str,
    vae_path: str,
    attn_mode: str,
    device: Union[str, torch.device],
    weight_dtype: torch.dtype,
    disable_mmap: bool = False,
    t5xxl_device: Optional[str] = None,
    t5xxl_dtype: Optional[str] = None,
):
    def load_state_dict(path: str, dvc: Union[str, torch.device] = device):
        if disable_mmap:
            return safetensors.torch.load(open(path, "rb").read())
        else:
            try:
                return load_file(path, device=dvc)
            except:
                return load_file(path)  # prevent device invalid Error

    t5xxl_device = t5xxl_device or device

    logger.info(f"Loading SD3 models from {ckpt_path}...")
    state_dict = load_state_dict(ckpt_path)

    # load clip_l
    clip_l_sd = None
    if clip_l_path:
        logger.info(f"Loading clip_l from {clip_l_path}...")
        clip_l_sd = load_state_dict(clip_l_path)
        for key in list(clip_l_sd.keys()):
            clip_l_sd["transformer." + key] = clip_l_sd.pop(key)
    else:
        if "text_encoders.clip_l.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
            # found clip_l: remove prefix "text_encoders.clip_l."
            logger.info("clip_l is included in the checkpoint")
            clip_l_sd = {}
            prefix = "text_encoders.clip_l."
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    clip_l_sd[k[len(prefix) :]] = state_dict.pop(k)

    # load clip_g
    clip_g_sd = None
    if clip_g_path:
        logger.info(f"Loading clip_g from {clip_g_path}...")
        clip_g_sd = load_state_dict(clip_g_path)
        for key in list(clip_g_sd.keys()):
            clip_g_sd["transformer." + key] = clip_g_sd.pop(key)
    else:
        if "text_encoders.clip_g.transformer.text_model.embeddings.position_embedding.weight" in state_dict:
            # found clip_g: remove prefix "text_encoders.clip_g."
            logger.info("clip_g is included in the checkpoint")
            clip_g_sd = {}
            prefix = "text_encoders.clip_g."
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    clip_g_sd[k[len(prefix) :]] = state_dict.pop(k)

    # load t5xxl
    t5xxl_sd = None
    if t5xxl_path:
        logger.info(f"Loading t5xxl from {t5xxl_path}...")
        t5xxl_sd = load_state_dict(t5xxl_path, t5xxl_device)
        for key in list(t5xxl_sd.keys()):
            t5xxl_sd["transformer." + key] = t5xxl_sd.pop(key)
    else:
        if "text_encoders.t5xxl.transformer.encoder.block.0.layer.0.SelfAttention.k.weight" in state_dict:
            # found t5xxl: remove prefix "text_encoders.t5xxl."
            logger.info("t5xxl is included in the checkpoint")
            t5xxl_sd = {}
            prefix = "text_encoders.t5xxl."
            for k in list(state_dict.keys()):
                if k.startswith(prefix):
                    t5xxl_sd[k[len(prefix) :]] = state_dict.pop(k)

    # MMDiT and VAE
    vae_sd = {}
    if vae_path:
        logger.info(f"Loading VAE from {vae_path}...")
        vae_sd = load_state_dict(vae_path)
    else:
        # remove prefix "first_stage_model."
        vae_sd = {}
        vae_prefix = "first_stage_model."
        for k in list(state_dict.keys()):
            if k.startswith(vae_prefix):
                vae_sd[k[len(vae_prefix) :]] = state_dict.pop(k)

    mmdit_prefix = "model.diffusion_model."
    for k in list(state_dict.keys()):
        if k.startswith(mmdit_prefix):
            state_dict[k[len(mmdit_prefix) :]] = state_dict.pop(k)
        else:
            state_dict.pop(k)  # remove other keys

    # load MMDiT
    logger.info("Building MMDit")
    with init_empty_weights():
        mmdit = sd3_models.create_mmdit_sd3_medium_configs(attn_mode)

    logger.info("Loading state dict...")
    info = sdxl_model_util._load_state_dict_on_device(mmdit, state_dict, device, weight_dtype)
    logger.info(f"Loaded MMDiT: {info}")

    # load ClipG and ClipL
    if clip_l_sd is None:
        clip_l = None
    else:
        logger.info("Building ClipL")
        clip_l = sd3_models.create_clip_l(device, weight_dtype, clip_l_sd)
        logger.info("Loading state dict...")
        info = clip_l.load_state_dict(clip_l_sd)
        logger.info(f"Loaded ClipL: {info}")
        clip_l.set_attn_mode(attn_mode)

    if clip_g_sd is None:
        clip_g = None
    else:
        logger.info("Building ClipG")
        clip_g = sd3_models.create_clip_g(device, weight_dtype, clip_g_sd)
        logger.info("Loading state dict...")
        info = clip_g.load_state_dict(clip_g_sd)
        logger.info(f"Loaded ClipG: {info}")
        clip_g.set_attn_mode(attn_mode)

    # load T5XXL
    if t5xxl_sd is None:
        t5xxl = None
    else:
        logger.info("Building T5XXL")
        t5xxl = sd3_models.create_t5xxl(t5xxl_device, t5xxl_dtype, t5xxl_sd)
        logger.info("Loading state dict...")
        info = t5xxl.load_state_dict(t5xxl_sd)
        logger.info(f"Loaded T5XXL: {info}")
        t5xxl.set_attn_mode(attn_mode)

    # load VAE
    logger.info("Building VAE")
    vae = sd3_models.SDVAE()
    logger.info("Loading state dict...")
    info = vae.load_state_dict(vae_sd)
    logger.info(f"Loaded VAE: {info}")

    return mmdit, clip_l, clip_g, t5xxl, vae


# endregion
# region utils


def get_cond(
    prompt: str,
    tokenizer: sd3_models.SD3Tokenizer,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    l_tokens, g_tokens, t5_tokens = tokenizer.tokenize_with_weights(prompt)
    return get_cond_from_tokens(l_tokens, g_tokens, t5_tokens, clip_l, clip_g, t5xxl, device=device, dtype=dtype)


def get_cond_from_tokens(
    l_tokens,
    g_tokens,
    t5_tokens,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    l_out, l_pooled = clip_l.encode_token_weights(l_tokens)
    g_out, g_pooled = clip_g.encode_token_weights(g_tokens)
    lg_out = torch.cat([l_out, g_out], dim=-1)
    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
    if device is not None:
        lg_out = lg_out.to(device=device)
        l_pooled = l_pooled.to(device=device)
        g_pooled = g_pooled.to(device=device)
    if dtype is not None:
        lg_out = lg_out.to(dtype=dtype)
        l_pooled = l_pooled.to(dtype=dtype)
        g_pooled = g_pooled.to(dtype=dtype)

    # t5xxl may be in another device (eg. cpu)
    if t5_tokens is None:
        t5_out = torch.zeros((lg_out.shape[0], 77, 4096), device=lg_out.device, dtype=lg_out.dtype)
    else:
        t5_out, _ = t5xxl.encode_token_weights(t5_tokens)  # t5_out is [1, 77, 4096], t5_pooled is None
        if device is not None:
            t5_out = t5_out.to(device=device)
        if dtype is not None:
            t5_out = t5_out.to(dtype=dtype)

    # return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)
    return lg_out, t5_out, torch.cat((l_pooled, g_pooled), dim=-1)


# used if other sd3 models is available
r"""
def get_sd3_configs(state_dict: Dict):
    # Important configuration values can be quickly determined by checking shapes in the source file
    # Some of these will vary between models (eg 2B vs 8B primarily differ in their depth, but also other details change)
    # prefix = "model.diffusion_model."
    prefix = ""

    patch_size = state_dict[prefix + "x_embedder.proj.weight"].shape[2]
    depth = state_dict[prefix + "x_embedder.proj.weight"].shape[0] // 64
    num_patches = state_dict[prefix + "pos_embed"].shape[1]
    pos_embed_max_size = round(math.sqrt(num_patches))
    adm_in_channels = state_dict[prefix + "y_embedder.mlp.0.weight"].shape[1]
    context_shape = state_dict[prefix + "context_embedder.weight"].shape
    context_embedder_config = {
        "target": "torch.nn.Linear",
        "params": {"in_features": context_shape[1], "out_features": context_shape[0]},
    }
    return {
        "patch_size": patch_size,
        "depth": depth,
        "num_patches": num_patches,
        "pos_embed_max_size": pos_embed_max_size,
        "adm_in_channels": adm_in_channels,
        "context_embedder": context_embedder_config,
    }


def create_mmdit_from_sd3_checkpoint(state_dict: Dict, attn_mode: str = "xformers"):
    ""
    Doesn't load state dict.
    ""
    sd3_configs = get_sd3_configs(state_dict)

    mmdit = sd3_models.MMDiT(
        input_size=None,
        pos_embed_max_size=sd3_configs["pos_embed_max_size"],
        patch_size=sd3_configs["patch_size"],
        in_channels=16,
        adm_in_channels=sd3_configs["adm_in_channels"],
        depth=sd3_configs["depth"],
        mlp_ratio=4,
        qk_norm=None,
        num_patches=sd3_configs["num_patches"],
        context_size=4096,
        attn_mode=attn_mode,
    )
    return mmdit
"""


class ModelSamplingDiscreteFlow:
    """Helper for sampler scheduling (ie timestep/sigma calculations) for Discrete Flow models"""

    def __init__(self, shift=1.0):
        self.shift = shift
        timesteps = 1000
        self.sigmas = self.sigma(torch.arange(1, timesteps + 1, 1))

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return self.shift * timestep / (1 + (self.shift - 1) * timestep)

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        # assert max_denoise is False, "max_denoise not implemented"
        # max_denoise is always True, I'm not sure why it's there
        return sigma * noise + (1.0 - sigma) * latent_image


# endregion
