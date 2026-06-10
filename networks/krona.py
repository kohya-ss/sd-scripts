# KronA (Kronecker Product Adaption) network module
# Compatible with LOKR format and inference
#
# References & Citation:
# - Paper: "DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Models" (WACV 2025)
#   arXiv: https://arxiv.org/abs/2402.17412
# - Project Website: https://diffusekrona.github.io/
# - Official Codebase: https://github.com/IBM/DiffuseKronA
#
# Implementation Differences (Official vs. Custom):
# 1. Computation Path:
#    - Official (diffusers): Reshapes hidden states and performs two consecutive small matrix multiplications
#      Y = B(X A^T) during forward pass to save training FLOPs and VRAM.
#    - Custom (sd-scripts / krona.py): Materializes the Kronecker product delta weight ΔW = B ⊗ A and applies
#      it via standard matrix multiplication Y = X ΔW^T or merges it directly into base weights.
#    - Equivalence: Due to the vectorization property vec(B X A^T) = (A ⊗ B) vec(X), both methods are 
#      mathematically 100% equivalent.
# 2. Ecosystem Compatibility:
#    - By using LoKr parameter naming conventions (lokr_w1, lokr_w2), this implementation outputs checkpoints
#      that are 100% compatible with ComfyUI, WebUI, and other standard LoKr inference loaders without any modification.
#
# Based on the lokr.py from sd-scripts


import ast
import math
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


def factorization_in(dimension: int) -> tuple:
    """Return a tuple of two values whose product equals dimension.
    For the input side, one factor is fixed to 4. If not divisible,
    decrements to 2 or 1.
    """
    for val in [4, 2, 1]:
        if dimension % val == 0:
            m = val
            n = dimension // val
            return m, n
    return 1, dimension


def factorization_out(dimension: int, max_val: int = 64) -> tuple:
    """Return a tuple of two values whose product equals dimension.
    For the output side, one factor is found by searching downwards
    from 64 for the largest integer that divides dimension.
    """
    for val in range(max_val, 0, -1):
        if dimension % val == 0:
            m = val
            n = dimension // val
            return m, n
    return 1, dimension


def make_kron(w1, w2, scale):
    """Compute Kronecker product of w1 and w2, scaled by scale."""
    if w1.dim() != w2.dim():
        for _ in range(w2.dim() - w1.dim()):
            w1 = w1.unsqueeze(-1)
    w2 = w2.contiguous()
    rebuild = torch.kron(w1, w2)
    if scale != 1:
        rebuild = rebuild * scale
    return rebuild


def rebuild_tucker(t, wa, wb):
    """Rebuild weight from Tucker decomposition."""
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


class KronaModule(torch.nn.Module):
    """Krona module for training. Replaces forward method of the original Linear/Conv2d.
    Uses parameter naming compatible with LoKr for seamless LOKR inference support.
    """

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
        use_tucker=False,
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        is_conv2d = org_module.__class__.__name__ == "Conv2d"
        if is_conv2d:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            self.is_conv = True
            self.stride = org_module.stride
            self.padding = org_module.padding
            self.dilation = org_module.dilation
            self.groups = org_module.groups
            self.kernel_size = kernel_size

            self.tucker = use_tucker and any(k != 1 for k in kernel_size)

            if kernel_size == (1, 1):
                self.conv_mode = "1x1"
            elif self.tucker:
                self.conv_mode = "tucker"
            else:
                self.conv_mode = "flat"
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_conv = False
            self.tucker = False
            self.conv_mode = None
            self.kernel_size = None

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_w2 = False

        # Apply KronA-specific factorization
        in_m, in_n = factorization_in(in_dim)
        out_l, out_k = factorization_out(out_dim)

        # To align with DiffuseKronA's B ⊗ A order while maintaining LoKr compatibility,
        # we assign B (large factor) to lokr_w1 and A (small factor) to lokr_w2.
        # Standard LoKr computations calculate: ΔW = lokr_w1 ⊗ lokr_w2 = B ⊗ A.
        self.lokr_w1 = nn.Parameter(torch.empty(out_k, in_n))

        if self.conv_mode in ("tucker", "flat"):
            k_size = kernel_size
            if lora_dim >= max(out_l, in_m) / 2:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(out_l, in_m, *k_size))
            elif self.tucker:
                self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *k_size))
                self.lokr_w2_a = nn.Parameter(torch.empty(lora_dim, out_l))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_m))
            else:
                k_prod = 1
                for k in k_size:
                    k_prod *= k
                self.lokr_w2_a = nn.Parameter(torch.empty(out_l, lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_m * k_prod))
        else:
            if lora_dim < max(out_l, in_m) / 2:
                self.lokr_w2_a = nn.Parameter(torch.empty(out_l, lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_m))
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(out_l, in_m))

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        if self.use_w2:
            alpha = lora_dim
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialization matching DiffuseKronA paper and codebase:
        # lokr_w1 (representing B matrix) initialized to zeros for zero initial delta weight
        torch.nn.init.zeros_(self.lokr_w1)
        
        # lokr_w2 (representing A matrix) initialized with normal distribution std=1/a1 (where a1 is out_l)
        if self.use_w2:
            torch.nn.init.normal_(self.lokr_w2, std=1.0 / self.lokr_w2.size(0))
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2_b)



        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def get_diff_weight(self):
        w1 = self.lokr_w1
        if self.use_w2:
            w2 = self.lokr_w2
        elif self.tucker:
            w2 = rebuild_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b

        result = make_kron(w1, w2, self.scale)
        if self.conv_mode == "flat" and result.dim() == 2:
            result = result.reshape(self.out_dim, self.in_dim, *self.kernel_size)
        return result

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()

        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            drop = drop.view(-1, *([1] * (diff_weight.dim() - 1)))
            diff_weight = diff_weight * drop
            scale = 1.0 / (1.0 - self.rank_dropout)
        else:
            scale = 1.0

        if self.is_conv:
            if self.conv_mode == "1x1":
                diff_weight = diff_weight.unsqueeze(2).unsqueeze(3)
                return org_forwarded + F.conv2d(
                    x, diff_weight, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups
                ) * self.multiplier * scale
            else:
                return org_forwarded + F.conv2d(
                    x, diff_weight, stride=self.stride, padding=self.padding,
                    dilation=self.dilation, groups=self.groups
                ) * self.multiplier * scale
        else:
            return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class KronaInfModule(KronaModule):
    """Krona module for inference. Supports merge_to and get_weight."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        use_tucker = kwargs.pop("use_tucker", False)
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, use_tucker=use_tucker)

        self.org_module_ref = [org_module]
        self.enabled = True
        self.network: AdditionalNetwork = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(torch.float)

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        w1 = sd["lokr_w1"].to(torch.float).to(device)

        if "lokr_w2" in sd:
            w2 = sd["lokr_w2"].to(torch.float).to(device)
        elif "lokr_t2" in sd:
            t2 = sd["lokr_t2"].to(torch.float).to(device)
            w2a = sd["lokr_w2_a"].to(torch.float).to(device)
            w2b = sd["lokr_w2_b"].to(torch.float).to(device)
            w2 = rebuild_tucker(t2, w2a, w2b)
        else:
            w2a = sd["lokr_w2_a"].to(torch.float).to(device)
            w2b = sd["lokr_w2_b"].to(torch.float).to(device)
            w2 = w2a @ w2b

        diff_weight = make_kron(w1, w2, self.scale)

        if diff_weight.shape != weight.shape:
            diff_weight = diff_weight.reshape(weight.shape)

        weight = weight.to(device) + self.multiplier * diff_weight
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        w1 = self.lokr_w1.to(torch.float)

        if self.use_w2:
            w2 = self.lokr_w2.to(torch.float)
        elif self.tucker:
            w2 = rebuild_tucker(
                self.lokr_t2.to(torch.float),
                self.lokr_w2_a.to(torch.float),
                self.lokr_w2_b.to(torch.float),
            )
        else:
            w2 = (self.lokr_w2_a @ self.lokr_w2_b).to(torch.float)

        weight = make_kron(w1, w2, self.scale) * multiplier

        if self.is_conv:
            if self.conv_mode == "1x1":
                weight = weight.unsqueeze(2).unsqueeze(3)
            elif self.conv_mode == "flat" and weight.dim() == 2:
                weight = weight.reshape(self.out_dim, self.in_dim, *self.kernel_size)

        return weight

    def default_forward(self, x):
        diff_weight = self.get_diff_weight()
        if self.is_conv:
            if self.conv_mode == "1x1":
                diff_weight = diff_weight.unsqueeze(2).unsqueeze(3)
            return self.org_forward(x) + F.conv2d(
                x, diff_weight, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            ) * self.multiplier
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
    """Create a Krona network (LOKR-compatible)."""
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)

    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    if train_llm_adapter is not None:
        train_llm_adapter = True if str(train_llm_adapter).lower() == "true" else False

    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)
        if not isinstance(exclude_patterns, list):
            exclude_patterns = [exclude_patterns]

    exclude_patterns.extend(arch_config.default_excludes)

    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None:
        include_patterns = ast.literal_eval(include_patterns)
        if not isinstance(include_patterns, list):
            include_patterns = [include_patterns]

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    conv_lora_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_lora_dim is not None:
        conv_lora_dim = int(conv_lora_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    use_tucker = kwargs.get("use_tucker", "false")
    if use_tucker is not None:
        use_tucker = True if str(use_tucker).lower() == "true" else False

    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if str(verbose).lower() == "true" else False

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
        module_class=KronaModule,
        module_kwargs={"use_tucker": use_tucker},
        conv_lora_dim=conv_lora_dim,
        conv_alpha=conv_alpha,
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
    )

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
    """Create a Krona network from saved weights (compatible with LOKR)."""
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    train_llm_adapter = False
    use_tucker = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lokr_w2_a" in key:
            if "lokr_t2" in key.replace("lokr_w2_a", "lokr_t2") and lora_name + ".lokr_t2" in weights_sd:
                dim = value.shape[0]
            else:
                dim = value.shape[1]
            modules_dim[lora_name] = dim
        elif "lokr_w2" in key and "lokr_w2_a" not in key and "lokr_w2_b" not in key:
            if lora_name not in modules_dim:
                modules_dim[lora_name] = max(value.shape[0], value.shape[1])

        if "lokr_t2" in key:
            use_tucker = True

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)

    module_class = KronaInfModule if for_inference else KronaModule
    module_kwargs = {"use_tucker": use_tucker}

    network = AdditionalNetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
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
    """Merge Krona weights directly into a model weight tensor using LoKr mapping."""
    # Reuse standard lokr merging function logic as keys are identical
    from .lokr import merge_weights_to_tensor as lokr_merge
    return lokr_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)
