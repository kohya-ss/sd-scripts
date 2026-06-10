# KronA (Kronecker Product Adaption) network module
# Compatible with LOKR format and inference
#
# References & Citation:
# - Paper: "DiffuseKronA: A Parameter Efficient Fine-tuning Method for Personalized Diffusion Models" (WACV 2025)
#   arXiv: https://arxiv.org/abs/2402.17412
# - Project Website: https://diffusekrona.github.io/
# - Official Codebase: https://github.com/IBM/DiffuseKronA
#

import ast
import math
import os
import logging
from typing import Dict, List, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_base import ArchConfig, AdditionalNetwork, detect_arch_config, _parse_kv_pairs
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def factorization_in(dimension: int, pref_val: int = 4) -> tuple:
    """Return a tuple of two values whose product equals dimension.
    For the input side, one factor is fixed to pref_val (default 4). If not divisible,
    decrements to 2 or 1.
    """
    for val in [pref_val, 4, 2, 1]:
        if val <= dimension and dimension % val == 0:
            m = val
            n = dimension // val
            return m, n
    return 1, dimension


def factorization_out(dimension: int, pref_val: int = 64) -> tuple:
    """Return a tuple of two values whose product equals dimension.
    For the output side, one factor is found by searching downwards
    from pref_val (default 64) for the largest integer that divides dimension.
    """
    for val in range(pref_val, 0, -1):
        if val <= dimension and dimension % val == 0:
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
    if scale != 1.0:
        rebuild = rebuild * scale
    return rebuild


class KronaModule(torch.nn.Module):
    """Krona module for training. Replaces forward method of the original Linear/Conv2d.
    Uses parameter naming compatible with LoKr for seamless LOKR inference support.
    Isomorphic design: r=1 Full Rank, supports customizable factorizations and initializations.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,           # Standard LoKr lora_dim (used for scale_lokr in degradation)
        alpha=1.0,            # Standard LoKr alpha (used for scale_lokr in degradation)
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        weight_decompose=False,
        wd_on_out=True,
        allora=False,
        allora_eta=2.0,
        **kwargs,
    ):
        super().__init__()

        self.lora_name = lora_name
        self.lora_dim = lora_dim

        # Customizable parameters. By default this follows DiffuseKronA-style
        # factorization: A=(factor_out, factor_in), B=(out/factor_out, in/factor_in).
        # If cdka_factor_in is provided, use CDKA-style factorization instead:
        # A=(factor_out, in/cdka_factor_in), B=(out/factor_out, cdka_factor_in).
        self.r = 1
        self.r1 = kwargs.get("factor_out", 64)
        self.r2 = kwargs.get("factor_in", 4)
        self.cdka_factor_in = kwargs.get("cdka_factor_in", None)
        w2_init = kwargs.get("w2_init", "normal")
        cdka_alpha = kwargs.get("cdka_alpha", None)
        if cdka_alpha is not None and str(cdka_alpha).lower() in ("none", "null", ""):
            cdka_alpha = None

        cdka_alpha = float(cdka_alpha) if cdka_alpha is not None else None

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

            if kernel_size == (1, 1):
                self.conv_mode = "1x1"
            else:
                self.conv_mode = "flat"
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.is_conv = False
            self.conv_mode = None
            self.kernel_size = None

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Flatten input dimension for Conv2d kernel product
        in_dim_flat = in_dim
        if self.conv_mode == "flat":
            k_prod = 1
            for k in kernel_size:
                k_prod *= k
            in_dim_flat = in_dim * k_prod

        # Apply factorization (pref_val can be customized)
        r1_val, out_k_val = factorization_out(out_dim, self.r1)
        if self.cdka_factor_in is not None:
            r2_val, in_m_val = factorization_in(in_dim_flat, int(self.cdka_factor_in))
        else:
            in_m_val, r2_val = factorization_in(in_dim_flat, self.r2)

        self.out_l = r1_val
        self.out_k = out_k_val
        self.in_n = r2_val
        self.in_m = in_m_val

        # Setup parameters (Full Rank)
        self.lokr_w1 = nn.Parameter(torch.empty(self.out_k, self.in_n))
        self.lokr_w2 = nn.Parameter(torch.empty(self.out_l, self.in_m))

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy().item()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.register_buffer("alpha", torch.tensor(alpha))

        # Setup scale. Full-rank LoKr inference ignores alpha, so state_dict()
        # folds this scale into lokr_w1 when exporting.
        if cdka_alpha is not None:
            self.scale = cdka_alpha / math.sqrt(self.in_n)
            self.alpha.copy_(torch.tensor(cdka_alpha))
        else:
            # Default KronA style: scale = 1.0 (ignores network_dim/alpha settings)
            self.scale = 1.0

        # Initialization
        # lokr_w1 (B) initialized to zeros

        nn.init.zeros_(self.lokr_w1)
        
        # lokr_w2 (A) initialized according to w2_init
        if w2_init == "normal":
            nn.init.normal_(self.lokr_w2, std=1.0 / self.out_l)
        elif w2_init == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.lokr_w2, a=math.sqrt(5))
        elif w2_init == "kaiming_normal":
            nn.init.kaiming_normal_(self.lokr_w2, a=math.sqrt(5))
        elif w2_init == "zeros":
            nn.init.zeros_(self.lokr_w2)
        else:
            raise ValueError(f"Unknown w2_init mode: {w2_init}")

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.wd = weight_decompose
        self.wd_on_out = wd_on_out
        if self.wd:
            self.register_buffer("org_weight", org_module.weight.data.clone(), persistent=False)
            org_weight_cpu = org_module.weight.data.cpu().clone().float()
            self.dora_norm_dims = org_weight_cpu.dim() - 1
            if self.wd_on_out:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight_cpu.reshape(org_weight_cpu.shape[0], -1),
                        dim=1,
                        keepdim=True,
                    ).reshape(org_weight_cpu.shape[0], *[1] * self.dora_norm_dims)
                ).float()
            else:
                self.dora_scale = nn.Parameter(
                    torch.norm(
                        org_weight_cpu.transpose(1, 0).reshape(org_weight_cpu.shape[1], -1),
                        dim=1,
                        keepdim=True,
                    )
                    .reshape(org_weight_cpu.shape[1], *[1] * self.dora_norm_dims)
                    .transpose(1, 0)
                ).float()

        self.allora = allora
        self.allora_eta = allora_eta



    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def get_diff_weight(self):
        w1 = self.lokr_w1
        w2 = self.lokr_w2
        result = make_kron(w1, w2, self.scale)
        if self.conv_mode == "flat" and result.dim() == 2:
            result = result.reshape(self.out_dim, self.in_dim, *self.kernel_size)
        return result

    def apply_weight_decompose(self, weight, multiplier=1):
        weight = weight.to(self.dora_scale.dtype)
        if self.wd_on_out:
            weight_norm = (
                weight.reshape(weight.shape[0], -1)
                .norm(dim=1)
                .reshape(weight.shape[0], *[1] * self.dora_norm_dims)
            ) + torch.finfo(weight.dtype).eps
        else:
            weight_norm = (
                weight.transpose(0, 1)
                .reshape(weight.shape[1], -1)
                .norm(dim=1, keepdim=True)
                .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
                .transpose(0, 1)
            ) + torch.finfo(weight.dtype).eps

        scale = self.dora_scale.to(weight.device) / weight_norm
        if multiplier != 1:
            scale = multiplier * (scale - 1) + 1

        return weight * scale

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.scale != 1.0:
            w1 = self.lokr_w1 * self.scale
            if not keep_vars:
                w1 = w1.detach()
            destination[prefix + "lokr_w1"] = w1
        if self.wd:
            destination[prefix + "dora_scale"] = self.dora_scale
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "lokr_w1"
        if key in state_dict and self.scale != 1.0:
            # Scale is folded into lokr_w1 when saving. To prevent repeated scaling during continue training,
            # we divide it back by the scale factor upon loading.
            state_dict = state_dict.copy()
            state_dict[key] = state_dict[key] / self.scale
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()
        diff_weight = diff_weight.to(x.dtype)

        if self.rank_dropout is not None and self.training:
            drop = (torch.rand(diff_weight.size(0), device=diff_weight.device) > self.rank_dropout).to(diff_weight.dtype)
            drop = drop.view(-1, *([1] * (diff_weight.dim() - 1)))
            diff_weight = diff_weight * drop
            dropout_scale = 1.0 / (1.0 - self.rank_dropout)
            diff_weight = diff_weight * dropout_scale

        if self.allora and self.training:
            diff_weight_static = diff_weight.detach()
            norms = torch.norm(diff_weight_static.reshape(diff_weight_static.shape[0], -1), dim=1)
            norms = norms.reshape(diff_weight_static.shape[0], *[1] * (diff_weight_static.dim() - 1))
            rsq_scale = 1.0 / (self.allora_eta ** 2)
            accelerate = 1.0 / torch.sqrt(norms + rsq_scale)
            acc_val = accelerate.to(diff_weight.device).to(diff_weight.dtype)
            diff_weight.register_hook(lambda grad: grad * acc_val)

        if self.wd:

            base_weight = self.org_weight.to(diff_weight.device)
            new_weight = self.apply_weight_decompose(base_weight + diff_weight, self.multiplier)
            delta_weight = (new_weight - base_weight).to(x.dtype)
            multiplier_val = 1.0
        else:
            delta_weight = diff_weight
            multiplier_val = self.multiplier

        if self.is_conv:
            if self.conv_mode == "1x1":
                delta_weight = delta_weight.unsqueeze(2).unsqueeze(3)
            return org_forwarded + F.conv2d(
                x, delta_weight, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            ) * multiplier_val
        else:
            return org_forwarded + F.linear(x, delta_weight) * multiplier_val


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
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, **kwargs)
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
        w2 = sd["lokr_w2"].to(torch.float).to(device)

        # Saved full LoKr weights already have scale folded into lokr_w1.
        diff_weight = make_kron(w1, w2, 1.0)

        if diff_weight.shape != weight.shape:
            diff_weight = diff_weight.reshape(weight.shape)

        if self.wd:
            if "dora_scale" in sd:
                with torch.no_grad():
                    self.dora_scale.copy_(sd["dora_scale"])
            weight = self.apply_weight_decompose(weight.to(device) + diff_weight.to(device), self.multiplier)
        else:
            weight = weight.to(device) + self.multiplier * diff_weight.to(device)


        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)


    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        w1 = self.lokr_w1.to(torch.float)
        w2 = self.lokr_w2.to(torch.float)

        weight = make_kron(w1, w2, self.scale) * multiplier

        if self.is_conv:
            if self.conv_mode == "1x1":
                weight = weight.unsqueeze(2).unsqueeze(3)
            elif self.conv_mode == "flat" and weight.dim() == 2:
                weight = weight.reshape(self.out_dim, self.in_dim, *self.kernel_size)

        return weight

    def default_forward(self, x):
        diff_weight = self.get_diff_weight()
        diff_weight = diff_weight.to(x.dtype)
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

    # Configurable factorization and init parameters from kwargs
    factor_in = kwargs.get("factor_in", 4)
    factor_out = kwargs.get("factor_out", 64)
    factor_in = int(factor_in) if factor_in is not None else 4
    factor_out = int(factor_out) if factor_out is not None else 64
    w2_init = kwargs.get("w2_init", "normal")
    cdka_factor_in = kwargs.get("cdka_factor_in", None)
    cdka_factor_in = int(cdka_factor_in) if cdka_factor_in is not None else None
    cdka_alpha = kwargs.get("cdka_alpha", None)
    cdka_alpha = float(cdka_alpha) if cdka_alpha is not None else None

    verbose = kwargs.get("verbose", "false")
    if verbose is not None:
        verbose = True if str(verbose).lower() == "true" else False

    weight_decompose = kwargs.get("weight_decompose", "false")
    weight_decompose = True if str(weight_decompose).lower() == "true" else False
    wd_on_out = kwargs.get("wd_on_out", "true")
    wd_on_out = True if str(wd_on_out).lower() == "true" else False

    allora = kwargs.get("allora", "false")
    allora = True if str(allora).lower() == "true" else False
    allora_eta = kwargs.get("allora_eta", None)
    allora_eta = float(allora_eta) if allora_eta is not None else 2.0

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
        module_kwargs={
            "factor_in": factor_in,
            "factor_out": factor_out,
            "w2_init": w2_init,
            "cdka_factor_in": cdka_factor_in,
            "cdka_alpha": cdka_alpha,
            "weight_decompose": weight_decompose,
            "wd_on_out": wd_on_out,
            "allora": allora,
            "allora_eta": allora_eta,
        },
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
    weight_decompose = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lokr_w2" in key:
            modules_dim[lora_name] = max(value.shape[0], value.shape[1])
        elif "dora_scale" in key:
            weight_decompose = True

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]
    arch_config = detect_arch_config(unet, text_encoders)

    module_class = KronaInfModule if for_inference else KronaModule

    network = AdditionalNetwork(
        text_encoders,
        unet,
        arch_config=arch_config,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs={
            "weight_decompose": weight_decompose,
        },
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
    from .lokr import merge_weights_to_tensor as lokr_merge
    return lokr_merge(model_weight, lora_name, lora_sd, lora_weight_keys, multiplier, calc_device)
