# LoHa (Low-rank Hadamard Product) network module
# Reference: https://arxiv.org/abs/2108.06098
#
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

import ast
import os
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_base import ArchConfig, AdditionalNetwork, detect_arch_config, _parse_kv_pairs
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class HadaWeight(torch.autograd.Function):
    """Efficient Hadamard product forward/backward for LoHa.

    Computes ((w1a @ w1b) * (w2a @ w2b)) * scale with custom backward
    that recomputes intermediates instead of storing them.
    """

    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=None):
        if scale is None:
            scale = torch.tensor(1, device=w1a.device, dtype=w1a.dtype)
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out * (w2a @ w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a @ w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp

        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class LoHaModule(torch.nn.Module):
    """LoHa module for training. Replaces forward method of the original Linear/Conv2d."""

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
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        is_conv2d = org_module.__class__.__name__ == "Conv2d"
        if is_conv2d:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            self.is_conv = True
            if org_module.kernel_size != (1, 1):
                raise ValueError("LoHa Conv2d 3x3 (Tucker decomposition) is not supported yet")
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_conv = False

        # Hadamard product parameters: ΔW = (w1a @ w1b) * (w2a @ w2b)
        self.hada_w1_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, in_dim))
        self.hada_w2_a = nn.Parameter(torch.empty(out_dim, lora_dim))
        self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, in_dim))

        # Initialization: w1_a normal(0.1), w1_b normal(1.0), w2_a = 0, w2_b normal(1.0)
        # Ensures ΔW = 0 at init since w2_a = 0
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1.0)
        torch.nn.init.constant_(self.hada_w2_a, 0)
        torch.nn.init.normal_(self.hada_w2_b, std=1.0)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def get_diff_weight(self):
        """Return materialized weight delta as a 2D matrix."""
        scale = torch.tensor(self.scale, dtype=self.hada_w1_a.dtype, device=self.hada_w1_a.device)
        return HadaWeight.apply(self.hada_w1_a, self.hada_w1_b, self.hada_w2_a, self.hada_w2_b, scale)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()

        # rank dropout (applied on output dimension)
        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            drop = drop.view(-1, 1)
            diff_weight = diff_weight * drop
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        if self.is_conv:
            # Conv2d 1x1: reshape to 4D for conv operation
            diff_weight = diff_weight.unsqueeze(2).unsqueeze(3)
            return org_forwarded + F.conv2d(x, diff_weight) * self.multiplier * scale
        else:
            return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class LoHaInfModule(LoHaModule):
    """LoHa module for inference. Supports merge_to and get_weight."""

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

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: AdditionalNetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(torch.float)

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        # get LoHa weights
        w1a = sd["hada_w1_a"].to(torch.float).to(device)
        w1b = sd["hada_w1_b"].to(torch.float).to(device)
        w2a = sd["hada_w2_a"].to(torch.float).to(device)
        w2b = sd["hada_w2_b"].to(torch.float).to(device)

        # compute ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
        diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * self.scale

        if self.is_conv:
            diff_weight = diff_weight.unsqueeze(2).unsqueeze(3)

        weight = weight.to(device) + self.multiplier * diff_weight

        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        w1a = self.hada_w1_a.to(torch.float)
        w1b = self.hada_w1_b.to(torch.float)
        w2a = self.hada_w2_a.to(torch.float)
        w2b = self.hada_w2_b.to(torch.float)

        weight = ((w1a @ w1b) * (w2a @ w2b)) * self.scale * multiplier

        if self.is_conv:
            weight = weight.unsqueeze(2).unsqueeze(3)

        return weight

    def default_forward(self, x):
        diff_weight = self.get_diff_weight()
        if self.is_conv:
            diff_weight = diff_weight.unsqueeze(2).unsqueeze(3)
            return self.org_forward(x) + F.conv2d(x, diff_weight) * self.multiplier
        else:
            return self.org_forward(x) + F.linear(x, diff_weight) * self.multiplier

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


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
    """Create a LoHa network. Called by train_network.py via network_module.create_network()."""
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    # handle text_encoder as list
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    # detect architecture
    arch_config = detect_arch_config(unet, text_encoders)

    # train LLM adapter
    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    if train_llm_adapter is not None:
        train_llm_adapter = True if str(train_llm_adapter).lower() == "true" else False

    # exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    # add default exclude patterns from arch config
    exclude_patterns.extend(arch_config.default_excludes)

    # include patterns
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # conv dim/alpha (for future Conv2d 3x3 support)
    conv_lora_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_lora_dim is not None:
        conv_lora_dim = int(conv_lora_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # verbose
    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if str(verbose).lower() == "true" else False

    # regex-specific learning rates / dimensions
    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    reg_lrs = _parse_kv_pairs(network_reg_lrs, is_int=False) if network_reg_lrs is not None else None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    reg_dims = _parse_kv_pairs(network_reg_dims, is_int=True) if network_reg_dims is not None else None

    network = AdditionalNetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        module_class=LoHaModule,
        conv_lora_dim=conv_lora_dim,
        conv_alpha=conv_alpha,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
    )

    # LoRA+ support
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
    """Create a LoHa network from saved weights. Called by train_network.py."""
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # detect dim/alpha from weights
    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "hada_w1_b" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    # handle text_encoder as list
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    # detect architecture
    arch_config = detect_arch_config(unet, text_encoders)

    module_class = LoHaInfModule if for_inference else LoHaModule

    network = AdditionalNetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        train_llm_adapter=train_llm_adapter,
    )
    return network, weights_sd


def merge_weights_to_tensor(
    model_weight: torch.Tensor,
    lora_name: str,
    lora_sd: Dict[str, torch.Tensor],
    lora_weight_keys: set,
    multiplier: float,
    calc_device: torch.device,
) -> torch.Tensor:
    """Merge LoHa weights directly into a model weight tensor.

    No Module/Network creation needed. Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoHa keys found.
    """
    w1a_key = lora_name + ".hada_w1_a"
    w1b_key = lora_name + ".hada_w1_b"
    w2a_key = lora_name + ".hada_w2_a"
    w2b_key = lora_name + ".hada_w2_b"
    alpha_key = lora_name + ".alpha"

    if w1a_key not in lora_weight_keys:
        return model_weight

    w1a = lora_sd[w1a_key].to(calc_device)
    w1b = lora_sd[w1b_key].to(calc_device)
    w2a = lora_sd[w2a_key].to(calc_device)
    w2b = lora_sd[w2b_key].to(calc_device)

    dim = w1b.shape[0]
    alpha = lora_sd.get(alpha_key, torch.tensor(dim))
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    scale = alpha / dim

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        w1a, w1b, w2a, w2b = w1a.to(torch.float16), w1b.to(torch.float16), w2a.to(torch.float16), w2b.to(torch.float16)

    # ΔW = ((w1a @ w1b) * (w2a @ w2b)) * scale
    diff_weight = ((w1a @ w1b) * (w2a @ w2b)) * scale

    # handle Conv2d 1x1 weights (4D tensors)
    if len(model_weight.shape) == 4:
        diff_weight = diff_weight.unsqueeze(2).unsqueeze(3)

    model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    # remove consumed keys
    for key in [w1a_key, w1b_key, w2a_key, w2b_key, alpha_key]:
        lora_weight_keys.discard(key)

    return model_weight
