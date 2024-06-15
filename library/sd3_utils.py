import math
from typing import Dict
import torch

from library import sd3_models


def get_cond(
    prompt: str,
    tokenizer: sd3_models.SD3Tokenizer,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: sd3_models.T5XXLModel,
):
    l_tokens, g_tokens, t5_tokens = tokenizer.tokenize_with_weights(prompt)
    l_out, l_pooled = clip_l.encode_token_weights(l_tokens)
    g_out, g_pooled = clip_g.encode_token_weights(g_tokens)
    lg_out = torch.cat([l_out, g_out], dim=-1)
    lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))

    if t5_tokens is None:
        t5_out = torch.zeros((lg_out.shape[0], 77, 4096), device=lg_out.device)
    else:
        t5_out, t5_pooled = t5xxl.encode_token_weights(t5_tokens)  # t5_out is [1, 77, 4096], t5_pooled is None
        t5_out = t5_out.to(lg_out.dtype)

    return torch.cat([lg_out, t5_out], dim=-2), torch.cat((l_pooled, g_pooled), dim=-1)


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
