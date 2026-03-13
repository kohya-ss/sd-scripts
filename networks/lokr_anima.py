import ast
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class LoKrModule(torch.nn.Module):
    """LoKr-like adapter module for Linear / Conv2d layers.

    Delta weight is represented as kron(M1, M2), where
    M1 = W1_a @ W1_b and M2 = W2_a @ W2_b.
    """

    def __init__(
        self,
        lokr_name: str,
        org_module: torch.nn.Module,
        multiplier: float = 1.0,
        lokr_dim: int = 4,
        alpha: float = 1.0,
        lokr_factor: Optional[int] = 8,
        module_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.lokr_name = lokr_name
        self.multiplier = multiplier
        self.module_dropout = module_dropout

        if org_module.__class__.__name__ == "Linear":
            out_dim = org_module.out_features
            in_dim = org_module.in_features
        elif org_module.__class__.__name__ == "Conv2d":
            out_dim, in_per_group, k_h, k_w = org_module.weight.shape
            in_dim = in_per_group * k_h * k_w
        else:
            raise ValueError(f"Unsupported module type for LoKr: {org_module.__class__.__name__}")

        self.org_module = org_module
        self.org_shape = tuple(org_module.weight.shape)
        self.is_linear = org_module.__class__.__name__ == "Linear"
        if not self.is_linear:
            self.org_stride = org_module.stride
            self.org_padding = org_module.padding
            self.org_dilation = org_module.dilation
            self.org_groups = org_module.groups

        self.out_a, self.out_b = self._factor_pair(out_dim, lokr_factor)
        self.in_a, self.in_b = self._factor_pair(in_dim, lokr_factor)

        self.lokr_dim = max(1, int(lokr_dim))

        # Full-matrix mode for very large dim values.
        # This follows the common LoKr behavior where an oversized dim (e.g. 100000)
        # requests a dense adapter instead of low-rank Kronecker factors.
        self.use_full_matrix = self.lokr_dim >= min(out_dim, in_dim)
        self.effective_rank = min(out_dim, in_dim) if self.use_full_matrix else self.lokr_dim

        alpha = self.effective_rank if alpha is None or alpha == 0 else alpha
        self.scale = float(alpha) / float(self.effective_rank)
        self.register_buffer("alpha", torch.tensor(float(alpha)))

        if self.use_full_matrix:
            self.lokr_full = torch.nn.Parameter(torch.empty(out_dim, in_dim))
            torch.nn.init.zeros_(self.lokr_full)
            logger.info(f"{self.lokr_name}: full matrix mode (dim={self.lokr_dim}, shape={out_dim}x{in_dim})")
        else:
            self.lokr_w1_a = torch.nn.Parameter(torch.empty(self.out_a, self.lokr_dim))
            self.lokr_w1_b = torch.nn.Parameter(torch.empty(self.lokr_dim, self.in_a))
            self.lokr_w2_a = torch.nn.Parameter(torch.empty(self.out_b, self.lokr_dim))
            self.lokr_w2_b = torch.nn.Parameter(torch.empty(self.lokr_dim, self.in_b))

            torch.nn.init.kaiming_uniform_(self.lokr_w1_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w1_b)
            torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lokr_w2_b)

    @staticmethod
    def _factor_pair(dim: int, preferred_factor: Optional[int] = 8) -> Tuple[int, int]:
        if preferred_factor is not None:
            preferred_factor = int(preferred_factor)
            if preferred_factor > 1 and dim % preferred_factor == 0:
                return preferred_factor, dim // preferred_factor

        r = int(math.sqrt(dim))
        while r > 1 and dim % r != 0:
            r -= 1
        if dim % r == 0:
            return r, dim // r
        return 1, dim

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _get_delta_weight_2d(self, device, dtype) -> torch.Tensor:
        if self.use_full_matrix:
            delta = self.lokr_full
        else:
            m1 = self.lokr_w1_a @ self.lokr_w1_b
            m2 = self.lokr_w2_a @ self.lokr_w2_b
            delta = torch.kron(m1, m2)
        delta = delta.to(device=device, dtype=dtype)
        return delta * self.multiplier * self.scale

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        if self.module_dropout is not None and self.training and torch.rand(1, device=x.device) < self.module_dropout:
            return org_forwarded

        if self.is_linear:
            delta_w = self._get_delta_weight_2d(x.device, x.dtype)
            # x[..., in] @ delta_w.T[in, out]
            return org_forwarded + F.linear(x, delta_w)

        # Conv2d
        delta_w = self._get_delta_weight_2d(x.device, x.dtype).reshape(self.org_shape)
        return org_forwarded + F.conv2d(
            x,
            delta_w,
            bias=None,
            stride=self.org_stride,
            padding=self.org_padding,
            dilation=self.org_dilation,
            groups=self.org_groups,
        )


class LoKrNetwork(torch.nn.Module):
    LOKR_PREFIX_ANIMA = "lokr_anima"
    LOKR_PREFIX_TEXT_ENCODER = "lokr_te"

    def __init__(
        self,
        text_encoders: list,
        unet: torch.nn.Module,
        multiplier: float = 1.0,
        lokr_dim: int = 4,
        alpha: float = 1.0,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, float]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        lokr_factor: Optional[int] = 8,
        module_dropout: Optional[float] = None,
        train_llm_adapter: bool = False,
        train_text_encoder: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.lokr_dim = lokr_dim
        self.alpha = alpha
        self.lokr_factor = lokr_factor
        self.train_llm_adapter = train_llm_adapter

        if modules_dim is not None:
            logger.info("create LoKr network from weights")
        else:
            logger.info(f"create LoKr network. base dim: {lokr_dim}, alpha: {alpha}, factor: {lokr_factor}")

        def compile_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
            out = []
            if patterns is not None:
                for pattern in patterns:
                    out.append(re.compile(pattern))
            return out

        exclude_re = compile_patterns(exclude_patterns)
        include_re = compile_patterns(include_patterns)

        def should_use(original_name: str) -> bool:
            excluded = any(p.fullmatch(original_name) for p in exclude_re)
            included = any(p.fullmatch(original_name) for p in include_re)
            return (not excluded) or included

        def create_modules(prefix: str, root_module: torch.nn.Module):
            adapters = []
            skipped = []
            for name, module in root_module.named_modules():
                is_linear = module.__class__.__name__ == "Linear"
                is_conv2d = module.__class__.__name__ == "Conv2d"
                if not (is_linear or is_conv2d):
                    continue

                original_name = name
                lokr_name = f"{prefix}.{original_name}".replace(".", "_")

                if not self.train_llm_adapter and original_name.startswith("llm_adapter"):
                    if verbose:
                        logger.info(f"exclude llm_adapter module: {original_name}")
                    continue

                if not should_use(original_name):
                    if verbose:
                        logger.info(f"exclude: {original_name}")
                    continue

                if modules_dim is not None:
                    dim = modules_dim.get(lokr_name, None)
                    alpha_val = modules_alpha.get(lokr_name, dim) if dim is not None else None
                else:
                    dim = lokr_dim
                    alpha_val = alpha

                if dim is None or int(dim) <= 0:
                    skipped.append(lokr_name)
                    continue

                adapter = LoKrModule(
                    lokr_name=lokr_name,
                    org_module=module,
                    multiplier=multiplier,
                    lokr_dim=int(dim),
                    alpha=float(alpha_val),
                    lokr_factor=lokr_factor,
                    module_dropout=module_dropout,
                )
                adapters.append(adapter)

            return adapters, skipped

        self.text_encoder_lokrs = []
        skipped_te = []
        if train_text_encoder and text_encoders is not None:
            for i, te in enumerate(text_encoders):
                if te is None:
                    continue
                te_adapters, te_skipped = create_modules(f"{self.LOKR_PREFIX_TEXT_ENCODER}{i}", te)
                logger.info(f"create LoKr for Text Encoder {i + 1}: {len(te_adapters)} modules")
                self.text_encoder_lokrs.extend(te_adapters)
                skipped_te += te_skipped

        self.unet_lokrs, skipped_unet = create_modules(self.LOKR_PREFIX_ANIMA, unet)
        logger.info(f"create LoKr for Anima DiT: {len(self.unet_lokrs)} modules")

        if verbose and (len(skipped_te) + len(skipped_unet) > 0):
            logger.info(f"skipped {len(skipped_te) + len(skipped_unet)} modules")

        names = set()
        for m in self.text_encoder_lokrs + self.unet_lokrs:
            assert m.lokr_name not in names, f"duplicated lokr name: {m.lokr_name}"
            names.add(m.lokr_name)

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        if not apply_text_encoder:
            self.text_encoder_lokrs = []
        if not apply_unet:
            self.unet_lokrs = []

        for lokr in self.text_encoder_lokrs + self.unet_lokrs:
            lokr.apply_to()
            self.add_module(lokr.lokr_name, lokr)

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        self.requires_grad_(True)
        params = []
        desc = []

        if self.text_encoder_lokrs:
            te_params = list(torch.nn.ModuleList(self.text_encoder_lokrs).parameters())
            if len(te_params) > 0:
                p = {"params": te_params}
                lr = text_encoder_lr if text_encoder_lr is not None else default_lr
                if isinstance(lr, list):
                    lr = lr[0] if len(lr) > 0 else default_lr
                if lr is not None:
                    p["lr"] = lr
                params.append(p)
                desc.append("textencoder")

        if self.unet_lokrs:
            unet_params = list(torch.nn.ModuleList(self.unet_lokrs).parameters())
            if len(unet_params) > 0:
                p = {"params": unet_params}
                lr = unet_lr if unet_lr is not None else default_lr
                if lr is not None:
                    p["lr"] = lr
                params.append(p)
                desc.append("unet")

        return params, desc

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def enable_gradient_checkpointing(self):
        pass

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                state_dict[key] = state_dict[key].detach().clone().to("cpu").to(dtype)

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        return self.load_state_dict(weights_sd, False)

    def is_mergeable(self):
        return False

    def apply_max_norm_regularization(self, max_norm_value, device):
        return 0, 0.0, 0.0


def _parse_patterns_arg(arg_value: Optional[str]) -> Optional[List[str]]:
    if arg_value is None:
        return None
    parsed = ast.literal_eval(arg_value)
    if isinstance(parsed, list):
        return parsed
    return [parsed]


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae,
    text_encoders: list,
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4
    if network_alpha is None:
        network_alpha = 1.0

    exclude_patterns = _parse_patterns_arg(kwargs.get("exclude_patterns", None))
    include_patterns = _parse_patterns_arg(kwargs.get("include_patterns", None))

    module_dropout = kwargs.get("module_dropout", neuron_dropout)
    module_dropout = float(module_dropout) if module_dropout is not None else None

    train_text_encoder = kwargs.get("train_text_encoder", "false")
    if isinstance(train_text_encoder, str):
        train_text_encoder = train_text_encoder.lower() == "true"

    train_llm_adapter = kwargs.get("train_llm_adapter", "false")
    if isinstance(train_llm_adapter, str):
        train_llm_adapter = train_llm_adapter.lower() == "true"

    verbose = kwargs.get("verbose", "false")
    if isinstance(verbose, str):
        verbose = verbose.lower() == "true"

    lokr_factor = kwargs.get("lokr_factor", 8)
    lokr_factor = int(lokr_factor) if lokr_factor is not None else None
    if lokr_factor is not None and lokr_factor <= 0:
        raise ValueError("lokr_factor must be positive")

    network = LoKrNetwork(
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        lokr_dim=int(network_dim),
        alpha=float(network_alpha),
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        lokr_factor=lokr_factor,
        module_dropout=module_dropout,
        train_llm_adapter=train_llm_adapter,
        train_text_encoder=train_text_encoder,
        verbose=verbose,
    )

    return network


def create_network_from_weights(multiplier, file, vae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
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
        lokr_name = key.split(".")[0]
        if key.endswith(".alpha"):
            modules_alpha[lokr_name] = float(value.item() if torch.is_tensor(value) else value)
        elif key.endswith(".lokr_w1_b"):
            modules_dim[lokr_name] = value.shape[0]
        elif key.endswith(".lokr_full"):
            # force full matrix mode when recreating from weights
            modules_dim[lokr_name] = int(max(value.shape[0], value.shape[1]))

    network = LoKrNetwork(
        text_encoders=text_encoders,
        unet=unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        lokr_factor=None,
        train_text_encoder=True,
    )
    return network, weights_sd
