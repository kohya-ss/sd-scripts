# Generic LoRA network module built on `networks.network_base` (ArchConfig + AdditionalNetwork).
#
# The architecture (target modules, key prefixes, default excludes) is detected from the models by
# `network_base.detect_arch_config`, so this module works for every architecture registered there. It is currently used
# by MOSS-Image (`moss_image_train_network.py`) and is the first step of unifying the per-model `networks/lora_*.py`.
#
# network_args:
#   exclude_patterns / include_patterns: Python list literal of regular expressions (full match) on the module name
#   rank_dropout / module_dropout: float
#   conv_dim / conv_alpha: LoRA for Conv2d 3x3 (architectures with unet_conv_target_modules only)
#   network_reg_dims / network_reg_lrs: "regex=value,..." per-module dim / learning rate
#   loraplus_lr_ratio / loraplus_unet_lr_ratio / loraplus_text_encoder_lr_ratio: LoRA+
#   train_llm_adapter / train_text_embedder: "true" to add the adapter modules of the architecture
#     (Anima: LLM adapter, MOSS-Image: text embedder = input projection + refiner blocks). The two names are aliases.
#   verbose: "true" to log the created modules

import ast
import math
import os
from typing import Dict, List, Optional

import torch

from .network_base import AdditionalNetwork, ArchConfig, detect_arch_config, _parse_kv_pairs
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class LoRAModule(torch.nn.Module):
    """Replaces the forward method of the original Linear / Conv2d (the original module is kept)."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        **kwargs,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # treated as a constant

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # removed in apply_to
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if isinstance(self.lora_down, torch.nn.Conv2d):
                # Conv2d: lora_dim is at dim 1 -> [B, dim, 1, 1]
                mask = mask.unsqueeze(-1).unsqueeze(-1)
            else:
                # Linear: lora_dim is at last dim -> [B, 1, ..., 1, dim]
                for _ in range(len(lx.size()) - 2):
                    mask = mask.unsqueeze(1)
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # for backup / restore of the original weights
        self.enabled = True
        self.network: Optional[AdditionalNetwork] = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
        """Merge the LoRA weights (from a state dict) into the original module."""
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        down_weight = sd["lora_down.weight"].to(torch.float).to(device)
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)

        weight = weight.to(torch.float).to(device)  # calc in float

        if len(weight.size()) == 2:
            # linear
            weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = weight + self.multiplier * conved * self.scale

        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        """Return the weight difference of this module (for restorable merging)."""
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        if len(down_weight.size()) == 2:
            # linear
            weight = multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = multiplier * conved * self.scale

        return weight

    def default_forward(self, x):
        lx = self.lora_down(x)
        lx = self.lora_up(lx)
        return self.org_forward(x) + lx * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


class LoRANetwork(AdditionalNetwork):
    """AdditionalNetwork with LoRA specific features (max norm regularization)."""

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)


def _parse_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).lower() == "true"


def _parse_patterns(value) -> Optional[List[str]]:
    if value is None:
        return None
    patterns = ast.literal_eval(value)
    if not isinstance(patterns, list):
        patterns = [patterns]
    return patterns


def is_adapter_training_requested(kwargs: dict) -> bool:
    """`train_llm_adapter` (Anima / LoHa / LoKr naming) and `train_text_embedder` (MOSS-Image naming) are aliases."""
    return _parse_bool(kwargs.get("train_llm_adapter", None)) or _parse_bool(kwargs.get("train_text_embedder", None))


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoder,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    """Create a LoRA network. Called by train_network.py via network_module.create_network()."""
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)

    train_adapter = is_adapter_training_requested(kwargs)

    exclude_patterns = _parse_patterns(kwargs.get("exclude_patterns", None)) or []
    exclude_patterns.extend(arch_config.default_excludes)
    include_patterns = _parse_patterns(kwargs.get("include_patterns", None))

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # conv dim/alpha for Conv2d 3x3
    conv_lora_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_lora_dim is not None:
        conv_lora_dim = int(conv_lora_dim)
        conv_alpha = 1.0 if conv_alpha is None else float(conv_alpha)

    verbose = _parse_bool(kwargs.get("verbose", None))

    # regex-specific learning rates / dimensions
    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    reg_lrs = _parse_kv_pairs(network_reg_lrs, is_int=False) if network_reg_lrs is not None else None
    network_reg_dims = kwargs.get("network_reg_dims", None)
    reg_dims = _parse_kv_pairs(network_reg_dims, is_int=True) if network_reg_dims is not None else None

    network = LoRANetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        module_class=LoRAModule,
        conv_lora_dim=conv_lora_dim,
        conv_alpha=conv_alpha,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        train_llm_adapter=train_adapter,
        verbose=verbose,
    )

    # LoRA+
    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    """Create a LoRA network from saved weights (dim / alpha per module are taken from the weights)."""
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)

    module_class = LoRAInfModule if for_inference else LoRAModule

    # the adapter modules are included in the search: only the modules found in the weights are created
    network = LoRANetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=True,
    )
    return network, weights_sd
