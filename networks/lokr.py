# LoKr (Low-rank Kronecker Product) network module
# Reference: https://arxiv.org/abs/2309.14859
#
# Based on the LyCORIS project by KohakuBlueleaf
# https://github.com/KohakuBlueleaf/LyCORIS

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


def factorization(dimension: int, factor: int = -1) -> tuple:
    """Return a tuple of two values whose product equals dimension,
    optimized for balanced factors.

    In LoKr, the first value is for the weight scale (smaller),
    and the second value is for the weight (larger).

    Examples:
        factor=-1: 128 -> (8, 16), 512 -> (16, 32), 1024 -> (32, 32)
        factor=4:  128 -> (4, 32), 512 -> (4, 128)
    """
    if factor > 0 and (dimension % factor) == 0:
        m = factor
        n = dimension // factor
        if m > n:
            n, m = m, n
        return m, n
    if factor < 0:
        factor = dimension
    m, n = 1, dimension
    length = m + n
    while m < n:
        new_m = m + 1
        while dimension % new_m != 0:
            new_m += 1
        new_n = dimension // new_m
        if new_m + new_n > length or new_m > factor:
            break
        else:
            m, n = new_m, new_n
    if m > n:
        n, m = m, n
    return m, n


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
    """Rebuild weight from Tucker decomposition: einsum("i j ..., i p, j r -> p r ...", t, wa, wb).

    Compatible with LyCORIS convention.
    """
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


class LoKrModule(torch.nn.Module):
    """LoKr module for training. Replaces forward method of the original Linear/Conv2d."""

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
        factor=-1,
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

        factor = int(factor)
        self.use_w2 = False

        # Factorize dimensions
        in_m, in_n = factorization(in_dim, factor)
        out_l, out_k = factorization(out_dim, factor)

        # w1 is always a full matrix (the "scale" factor, small)
        self.lokr_w1 = nn.Parameter(torch.empty(out_l, in_m))

        # w2: depends on mode
        if self.conv_mode in ("tucker", "flat"):
            # Conv2d 3x3+ modes
            k_size = kernel_size

            if lora_dim >= max(out_k, in_n) / 2:
                # Full matrix mode (includes kernel dimensions)
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n, *k_size))
                logger.warning(
                    f"LoKr: lora_dim {lora_dim} is large for dim={max(in_dim, out_dim)} "
                    f"and factor={factor}, using full matrix mode for Conv2d."
                )
            elif self.tucker:
                # Tucker mode: separate kernel into t2 tensor
                self.lokr_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, *k_size))
                self.lokr_w2_a = nn.Parameter(torch.empty(lora_dim, out_k))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
            else:
                # Non-Tucker: flatten kernel into w2_b
                k_prod = 1
                for k in k_size:
                    k_prod *= k
                self.lokr_w2_a = nn.Parameter(torch.empty(out_k, lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_n * k_prod))
        else:
            # Linear or Conv2d 1x1
            if lora_dim < max(out_k, in_n) / 2:
                self.lokr_w2_a = nn.Parameter(torch.empty(out_k, lora_dim))
                self.lokr_w2_b = nn.Parameter(torch.empty(lora_dim, in_n))
            else:
                self.use_w2 = True
                self.lokr_w2 = nn.Parameter(torch.empty(out_k, in_n))
                if lora_dim >= max(out_k, in_n) / 2:
                    logger.warning(
                        f"LoKr: lora_dim {lora_dim} is large for dim={max(in_dim, out_dim)} "
                        f"and factor={factor}, using full matrix mode."
                    )

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        # if both w1 and w2 are full matrices, use scale = 1
        if self.use_w2:
            alpha = lora_dim
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        # Initialization
        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        if self.use_w2:
            torch.nn.init.constant_(self.lokr_w2, 0)
        else:
            if self.tucker:
                torch.nn.init.kaiming_uniform_(self.lokr_t2, a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.constant_(self.lokr_w2_b, 0)
        # Ensures ΔW = kron(w1, 0) = 0 at init

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
        """Return materialized weight delta.

        Returns:
            - Linear: 2D tensor (out_dim, in_dim)
            - Conv2d 1x1: 2D tensor (out_dim, in_dim) — caller should unsqueeze for F.conv2d
            - Conv2d 3x3+ Tucker/full: 4D tensor (out_dim, in_dim, k1, k2)
            - Conv2d 3x3+ flat: 4D tensor (out_dim, in_dim, k1, k2) — reshaped from 2D
        """
        w1 = self.lokr_w1

        if self.use_w2:
            w2 = self.lokr_w2
        elif self.tucker:
            w2 = rebuild_tucker(self.lokr_t2, self.lokr_w2_a, self.lokr_w2_b)
        else:
            w2 = self.lokr_w2_a @ self.lokr_w2_b

        result = make_kron(w1, w2, self.scale)

        # For non-Tucker Conv2d 3x3+, result is 2D; reshape to 4D
        if self.conv_mode == "flat" and result.dim() == 2:
            result = result.reshape(self.out_dim, self.in_dim, *self.kernel_size)

        return result

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        diff_weight = self.get_diff_weight()

        # rank dropout
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
                # Conv2d 3x3+: diff_weight is already 4D from get_diff_weight
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


class LoKrInfModule(LoKrModule):
    """LoKr module for inference. Supports merge_to and get_weight."""

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference; pass factor and use_tucker from kwargs
        factor = kwargs.pop("factor", -1)
        use_tucker = kwargs.pop("use_tucker", False)
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, factor=factor, use_tucker=use_tucker)

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

        # get LoKr weights
        w1 = sd["lokr_w1"].to(torch.float).to(device)

        if "lokr_w2" in sd:
            w2 = sd["lokr_w2"].to(torch.float).to(device)
        elif "lokr_t2" in sd:
            # Tucker mode
            t2 = sd["lokr_t2"].to(torch.float).to(device)
            w2a = sd["lokr_w2_a"].to(torch.float).to(device)
            w2b = sd["lokr_w2_b"].to(torch.float).to(device)
            w2 = rebuild_tucker(t2, w2a, w2b)
        else:
            w2a = sd["lokr_w2_a"].to(torch.float).to(device)
            w2b = sd["lokr_w2_b"].to(torch.float).to(device)
            w2 = w2a @ w2b

        # compute ΔW via Kronecker product
        diff_weight = make_kron(w1, w2, self.scale)

        # reshape diff_weight to match original weight shape if needed
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

        # reshape to match original weight shape if needed
        if self.is_conv:
            if self.conv_mode == "1x1":
                weight = weight.unsqueeze(2).unsqueeze(3)
            elif self.conv_mode == "flat" and weight.dim() == 2:
                weight = weight.reshape(self.out_dim, self.in_dim, *self.kernel_size)
            # Tucker and full matrix modes: already 4D from kron

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
    """Create a LoKr network. Called by train_network.py via network_module.create_network()."""
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

    # conv dim/alpha for Conv2d 3x3
    conv_lora_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_lora_dim is not None:
        conv_lora_dim = int(conv_lora_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # Tucker decomposition for Conv2d 3x3
    use_tucker = kwargs.get("use_tucker", "false")
    if use_tucker is not None:
        use_tucker = True if str(use_tucker).lower() == "true" else False

    # factor for LoKr
    factor = int(kwargs.get("factor", -1))

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
        module_class=LoKrModule,
        module_kwargs={"factor": factor, "use_tucker": use_tucker},
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
    """Create a LoKr network from saved weights. Called by train_network.py."""
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
    use_tucker = False
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lokr_w2_a" in key:
            # low-rank mode: dim detection depends on Tucker vs non-Tucker
            if "lokr_t2" in key.replace("lokr_w2_a", "lokr_t2") and lora_name + ".lokr_t2" in weights_sd:
                # Tucker: w2_a = (rank, out_k) → dim = w2_a.shape[0]
                dim = value.shape[0]
            else:
                # Non-Tucker: w2_a = (out_k, rank) → dim = w2_a.shape[1]
                dim = value.shape[1]
            modules_dim[lora_name] = dim
        elif "lokr_w2" in key and "lokr_w2_a" not in key and "lokr_w2_b" not in key:
            # full matrix mode: set dim large enough to trigger full-matrix path
            if lora_name not in modules_dim:
                modules_dim[lora_name] = max(value.shape[0], value.shape[1])

        if "lokr_t2" in key:
            use_tucker = True

        if "llm_adapter" in lora_name:
            train_llm_adapter = True

    # handle text_encoder as list
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    # detect architecture
    arch_config = detect_arch_config(unet, text_encoders)

    # extract factor for LoKr
    factor = int(kwargs.get("factor", -1))

    module_class = LoKrInfModule if for_inference else LoKrModule
    module_kwargs = {"factor": factor, "use_tucker": use_tucker}

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
    """Merge LoKr weights directly into a model weight tensor.

    Supports standard LoKr, non-Tucker Conv2d 3x3, and Tucker Conv2d 3x3.
    No Module/Network creation needed. Consumed keys are removed from lora_weight_keys.
    Returns model_weight unchanged if no matching LoKr keys found.
    """
    w1_key = lora_name + ".lokr_w1"
    w2_key = lora_name + ".lokr_w2"
    w2a_key = lora_name + ".lokr_w2_a"
    w2b_key = lora_name + ".lokr_w2_b"
    t2_key = lora_name + ".lokr_t2"
    alpha_key = lora_name + ".alpha"

    if w1_key not in lora_weight_keys:
        return model_weight

    w1 = lora_sd[w1_key].to(calc_device)

    # determine mode: full matrix vs Tucker vs low-rank
    has_tucker = t2_key in lora_weight_keys

    if w2a_key in lora_weight_keys:
        w2a = lora_sd[w2a_key].to(calc_device)
        w2b = lora_sd[w2b_key].to(calc_device)

        if has_tucker:
            # Tucker: w2a = (rank, out_k), dim = rank
            dim = w2a.shape[0]
        else:
            # Non-Tucker low-rank: w2a = (out_k, rank), dim = rank
            dim = w2a.shape[1]

        consumed_keys = [w1_key, w2a_key, w2b_key, alpha_key]
        if has_tucker:
            consumed_keys.append(t2_key)
    elif w2_key in lora_weight_keys:
        # full matrix mode
        w2a = None
        w2b = None
        dim = None
        consumed_keys = [w1_key, w2_key, alpha_key]
    else:
        return model_weight

    alpha = lora_sd.get(alpha_key, None)
    if alpha is not None and isinstance(alpha, torch.Tensor):
        alpha = alpha.item()

    # compute scale
    if w2a is not None:
        if alpha is None:
            alpha = dim
        scale = alpha / dim
    else:
        # full matrix mode: scale = 1.0
        scale = 1.0

    original_dtype = model_weight.dtype
    if original_dtype.itemsize == 1:  # fp8
        model_weight = model_weight.to(torch.float16)
        w1 = w1.to(torch.float16)
        if w2a is not None:
            w2a, w2b = w2a.to(torch.float16), w2b.to(torch.float16)

    # compute w2
    if w2a is not None:
        if has_tucker:
            t2 = lora_sd[t2_key].to(calc_device)
            if original_dtype.itemsize == 1:
                t2 = t2.to(torch.float16)
            w2 = rebuild_tucker(t2, w2a, w2b)
        else:
            w2 = w2a @ w2b
    else:
        w2 = lora_sd[w2_key].to(calc_device)
        if original_dtype.itemsize == 1:
            w2 = w2.to(torch.float16)

    # ΔW = kron(w1, w2) * scale
    diff_weight = make_kron(w1, w2, scale)

    # Reshape diff_weight to match model_weight shape if needed
    # (handles Conv2d 1x1 unsqueeze, Conv2d 3x3 non-Tucker reshape, etc.)
    if diff_weight.shape != model_weight.shape:
        diff_weight = diff_weight.reshape(model_weight.shape)

    model_weight = model_weight + multiplier * diff_weight

    if original_dtype.itemsize == 1:
        model_weight = model_weight.to(original_dtype)

    # remove consumed keys
    for key in consumed_keys:
        lora_weight_keys.discard(key)

    return model_weight
