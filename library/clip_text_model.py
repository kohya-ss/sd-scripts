"""Compatibility wrapper for `transformers.CLIPTextModel`.

transformers 5.6.0 flattened `CLIPTextModel`: the `text_model` submodule (`CLIPTextTransformer`)
was removed and `embeddings` / `encoder` / `final_layer_norm` became direct children.
`from_pretrained` / `save_pretrained` convert the keys transparently, but `load_state_dict`,
`state_dict()` and `named_modules()` expose the flattened layout. This changes

- the state dict keys that our checkpoint loaders / savers produce (`text_model.` prefix), and
- the module names that LoRA / OFT etc. derive their weight names from
  (`lora_te_text_model_encoder_layers_*`), which must stay stable for compatibility with
  existing LoRA files and other tools.

`CLIPTextModelWrapper` puts the flattened model back under a `text_model` attribute so that
state dict keys, module names and `text_encoder.text_model.*` attribute access keep the
pre-5.6 layout. Only the model construction sites need to call `wrap_clip_text_model`;
everything else keeps working unchanged. With transformers < 5.6 the wrapper is not applied.

`CLIPTextModelWithProjection` is not affected (it still has `text_model`).
"""

from typing import Any, Optional, Union

import torch
from torch import nn


def is_flattened_clip_text_model(model: nn.Module) -> bool:
    """True if `model` is a `CLIPTextModel` with the flattened (transformers >= 5.6) layout."""
    return (
        not isinstance(model, CLIPTextModelWrapper)
        and hasattr(model, "encoder")
        and hasattr(model, "final_layer_norm")
        and not hasattr(model, "text_model")
    )


class CLIPTextModelWrapper(nn.Module):
    """Holds a flattened `CLIPTextModel` as `self.text_model` and delegates everything to it.

    `state_dict()` / `load_state_dict()` / `named_modules()` see the `text_model.` prefix, so the
    wrapped model is indistinguishable from a pre-5.6 `CLIPTextModel` for loaders, savers and
    network modules. `forward` and the commonly used `PreTrainedModel` helpers are delegated;
    any other attribute not found on the wrapper falls back to the inner model.
    """

    def __init__(self, text_model: nn.Module):
        super().__init__()
        self.text_model = text_model

    # --- forward -----------------------------------------------------------------------------

    def forward(self, *args, **kwargs):
        return self.text_model(*args, **kwargs)

    # --- PreTrainedModel-like helpers ----------------------------------------------------------

    @property
    def config(self):
        return self.text_model.config

    @property
    def dtype(self) -> torch.dtype:
        return self.text_model.dtype

    @property
    def device(self) -> torch.device:
        return self.text_model.device

    def get_input_embeddings(self) -> nn.Module:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value: nn.Module):
        self.text_model.embeddings.token_embedding = value

    def resize_token_embeddings(self, *args, **kwargs):
        return self.text_model.resize_token_embeddings(*args, **kwargs)

    def gradient_checkpointing_enable(self, *args, **kwargs):
        return self.text_model.gradient_checkpointing_enable(*args, **kwargs)

    def gradient_checkpointing_disable(self, *args, **kwargs):
        return self.text_model.gradient_checkpointing_disable(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        # the inner model converts the keys back to the layout of the installed transformers
        return self.text_model.save_pretrained(*args, **kwargs)

    # --- fallback ------------------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # nn.Module.__getattr__ only resolves parameters / buffers / submodules; anything else
        # (e.g. `eos_token_id`, `main_input_name`, `_no_split_modules`) comes from the inner model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "text_model":  # not yet registered (during __init__) or a real error
                raise
            return getattr(self.text_model, name)


def wrap_clip_text_model(model: nn.Module) -> nn.Module:
    """Wrap `model` with `CLIPTextModelWrapper` if it has the flattened layout, else return as is.

    Call this right after constructing a `CLIPTextModel` (via `_from_config` / `from_pretrained` /
    a Diffusers pipeline) and before `load_state_dict`. Idempotent.
    """
    if is_flattened_clip_text_model(model):
        return CLIPTextModelWrapper(model)
    return model


def unwrap_clip_text_model(model: nn.Module) -> nn.Module:
    """Return the inner `CLIPTextModel` if `model` is wrapped, else `model` itself.

    Use this when handing the model to code that needs the real transformers class, e.g. a
    Diffusers pipeline for `save_pretrained` (which records the class name in model_index.json).
    """
    if isinstance(model, CLIPTextModelWrapper):
        return model.text_model
    return model
