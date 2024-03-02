# adapted from tomesd, which uses the MIT license
# https://github.com/dbolya/tomesd/blob/main/tomesd/patch.py

from typing import Type

import torch
from tomesd.patch import compute_merge, hook_tome_model, remove_patch
from tomesd.utils import isinstance_str


def make_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class on the fly so we don't have to import any specific modules.
    This patch applies ToMe to the forward function of the block.
    """

    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(self, hidden_states: torch.Tensor, context: torch.Tensor = None, timestep=None) -> torch.Tensor:
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            # 1. Self-Attention
            norm_hidden_states = m_a(self.norm1(hidden_states))
            hidden_states = u_a(self.attn1(norm_hidden_states)) + hidden_states

            # 2. Cross-Attention
            norm_hidden_states = m_c(self.norm2(hidden_states))
            hidden_states = u_c(self.attn2(norm_hidden_states, context=context)) + hidden_states

            # 3. Feed-forward
            hidden_states = u_m(self.ff(m_m(self.norm3(hidden_states)))) + hidden_states

            return hidden_states

    return ToMeBlock


def apply_patch(
        model: torch.nn.Module,
        ratio: float = 0.5,
        max_downsample: int = 1,
        sx: int = 2, sy: int = 2,
        use_rand: bool = True,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with ToMe. Modified to work with kohya-ss/sd-scripts
    Apply this to the unet.

    Important Args:
     - model: A Stable Diffusion unet module to patch in place.
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(model)

    diffusion_model = model

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx, "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    hook_tome_model(diffusion_model)

    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            module.__class__ = make_tome_block(module.__class__)
            module._tome_info = diffusion_model._tome_info

            # Something introduced in SD 2.0 (LDM only)
            if not hasattr(module, "disable_self_attn"):
                module.disable_self_attn = False

    return model
