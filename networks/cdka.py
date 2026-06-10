# CDKA (Component Designed Kronecker Adapters) network module
# Supporting multi-component training and SVD degradation to single-component LoKr format on save.
#
# Reference Paper: 
# - "Diving into Kronecker Adapters: Component Design Matters" (arXiv:2602.01267, 2026)
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


def factorization_in(dimension: int, pref_val: int = 8) -> tuple:
    """Return a tuple of two values whose product equals dimension.
    For CDKA input side, we prefer pref_val (default 8). If not divisible, fallback to 4, 2, 1.
    """
    for val in [pref_val, 4, 2, 1]:
        if val <= dimension and dimension % val == 0:
            m = val
            n = dimension // val
            return m, n
    return 1, dimension


def factorization_out(dimension: int, pref_val: int = 2) -> tuple:
    """Return a tuple of two values whose product equals dimension.
    For CDKA output side, we prefer pref_val (default 2). If not divisible, fallback to 1.
    """
    for val in [pref_val, 1]:
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


class CdkaModule(torch.nn.Module):
    """Cdka module for training. Replaces forward method of the original Linear/Conv2d.
    Supports multi-component training with stabilization factor lambda,
    and SVD degradation to standard single-component LoKr on save.
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
        **kwargs,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        # CDKA-specific hyperparameters. Default to r=1 so LoKr export is lossless.
        self.r = kwargs.get("r", 1)
        self.r1 = kwargs.get("r1", 2)
        self.r2 = kwargs.get("r2", 8)

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

        # Apply CDKA factorization (Paper: small r1, large r2)
        # factorization_out returns (r1_val, out_k_val)
        # factorization_in returns (r2_val, in_m_val)
        r1_val, out_k_val = factorization_out(out_dim, self.r1)
        r2_val, in_m_val = factorization_in(in_dim_flat, self.r2)

        self.out_l = r1_val
        self.out_k = out_k_val
        self.in_n = r2_val
        self.in_m = in_m_val

        # Multi-component parameters:
        # lokr_w1_multi (B) shape: (r, out_k, in_n)
        # lokr_w2_multi (A) shape: (r, out_l, in_m)
        self.lokr_w1_multi = nn.Parameter(torch.empty(self.r, self.out_k, self.in_n))
        self.lokr_w2_multi = nn.Parameter(torch.empty(self.r, self.out_l, self.in_m))

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy().item()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.register_buffer("alpha", torch.tensor(alpha))

        # CDKA Stabilization Scaling Factor: lambda = alpha / sqrt(r * r2)
        # self.in_n represents r2 (input-side factor)
        self.scale = alpha / math.sqrt(self.r * self.in_n)

        # Initialization
        # lokr_w1_multi (B) initialized to zeros
        nn.init.zeros_(self.lokr_w1_multi)
        
        # lokr_w2_multi (A) initialized with Kaiming Uniform
        for i in range(self.r):
            nn.init.kaiming_uniform_(self.lokr_w2_multi[i], a=math.sqrt(5))

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
        # Accumulate Kronecker products over components
        diff_weight = 0
        for i in range(self.r):
            w1 = self.lokr_w1_multi[i]
            w2 = self.lokr_w2_multi[i]
            diff_weight += make_kron(w1, w2, 1.0)

        # Apply the scale factor lambda
        diff_weight = diff_weight * self.scale
        
        if self.conv_mode == "flat" and diff_weight.dim() == 2:
            diff_weight = diff_weight.reshape(self.out_dim, self.in_dim, *self.kernel_size)
        return diff_weight

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
            return org_forwarded + F.linear(x, diff_weight) * self.multiplier * scale

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        # 1. Fast path for a single component: export exact LoKr weights.
        if self.r == 1:
            with torch.no_grad():
                w1 = self.lokr_w1_multi[0] * self.scale
                w2 = self.lokr_w2_multi[0]
                if not keep_vars:
                    w1 = w1.detach()
                    w2 = w2.detach()
                destination[prefix + "lokr_w1"] = w1
                destination[prefix + "lokr_w2"] = w2
                destination[prefix + "alpha"] = self.alpha
            return destination

        # 2. Compute multi-component update delta_w
        with torch.no_grad():
            diff_weight = 0
            for i in range(self.r):
                w1 = self.lokr_w1_multi[i]
                w2 = self.lokr_w2_multi[i]
                diff_weight += make_kron(w1, w2, 1.0)
            
            delta_w = diff_weight * self.scale  # shape: (out_dim, in_dim_flat)
        
        # 3. Divide by the standard LoKr inference scale (alpha / lora_dim)
        scale_lokr = self.alpha.item() / self.lora_dim
        delta_w_for_svd = delta_w / scale_lokr
        
        # 4. Kreshape and perform SVD
        # delta_w_for_svd has shape (out_k * out_l, in_n * in_m)
        # We reshape and permute to (out_k * in_n, out_l * in_m)
        W_tilde = delta_w_for_svd.contiguous().view(
            self.out_k, self.out_l, self.in_n, self.in_m
        ).permute(0, 2, 1, 3).reshape(self.out_k * self.in_n, self.out_l * self.in_m)
        
        # Perform SVD on CPU to prevent backend support issues
        W_tilde_cpu = W_tilde.to(torch.float32).cpu()
        U, S, V = torch.linalg.svd(W_tilde_cpu, full_matrices=False)
        
        sigma_1 = S[0]
        u_1 = U[:, 0]
        v_1 = V[0, :]  # conjugate transpose V^H first row
        
        # SVD NKP solution:
        B_vec = torch.sqrt(sigma_1) * u_1
        A_vec = torch.sqrt(sigma_1) * v_1
        
        B_approx = B_vec.reshape(self.out_k, self.in_n).to(device=delta_w.device, dtype=delta_w.dtype)
        A_approx = A_vec.reshape(self.out_l, self.in_m).to(device=delta_w.device, dtype=delta_w.dtype)
        
        # 5. Save degraded weights into destination dictionary
        destination[prefix + "lokr_w1"] = B_approx
        destination[prefix + "lokr_w2"] = A_approx
        destination[prefix + "alpha"] = self.alpha
        
        return destination

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


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
    """Create a CDKA network."""
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

    # CDKA hyperparameters from kwargs
    r = kwargs.get("r", 1)
    r1 = kwargs.get("r1", 2)
    r2 = kwargs.get("r2", 8)
    r = int(r) if r is not None else 1
    r1 = int(r1) if r1 is not None else 2
    r2 = int(r2) if r2 is not None else 8

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
        module_class=CdkaModule,
        module_kwargs={"r": r, "r1": r1, "r2": r2},
        train_llm_adapter=train_llm_adapter,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
    )

    return network
