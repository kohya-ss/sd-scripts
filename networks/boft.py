# PEFT-style BOFT network module for SD1/SD2/SDXL training.

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from library.utils import setup_logging
from networks.orthogonal_common import (
    LORA_PREFIX_TEXT_ENCODER,
    LORA_PREFIX_TEXT_ENCODER1,
    LORA_PREFIX_TEXT_ENCODER2,
    LORA_PREFIX_UNET,
    TEXT_ENCODER_TARGET_REPLACE_MODULE,
    UNET_TARGET_REPLACE_MODULE,
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3,
    adjust_block_size,
    extract_module_state,
    get_in_features,
    get_out_features,
    is_conv2d_1x1,
    is_conv2d_module,
    is_linear_module,
    load_weights_sd,
    native_prefix,
    parse_patterns,
    save_weights_sd,
    str_to_bool,
    str_to_float,
    str_to_int,
    text_encoder_list,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MultiplicativeDropoutLayer(nn.Module):
    """PEFT-compatible multiplicative dropout for BOFT rotation blocks."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x

        if x.shape[-1] != x.shape[-2]:
            raise ValueError("The last two dimensions of input should be the same")

        n_factors, block_num, block_size, _ = x.shape
        n_random = torch.randint(0, n_factors, (1,), device=x.device).item()
        num_to_replace = int(self.p * block_num)
        if num_to_replace <= 0:
            return x

        mask = torch.cat(
            [
                torch.ones(num_to_replace, device=x.device, dtype=x.dtype),
                torch.zeros(block_num - num_to_replace, device=x.device, dtype=x.dtype),
            ]
        )
        mask = mask[torch.randperm(block_num, device=x.device)].view(1, block_num, 1, 1)
        full_mask = torch.zeros(n_factors, block_num, 1, 1, device=x.device, dtype=x.dtype)
        full_mask[n_random] = mask
        eye = torch.eye(block_size, device=x.device, dtype=x.dtype).repeat(n_factors, block_num, 1, 1)
        return (1 - full_mask) * x + full_mask * eye


def _valid_boft_shape(in_features: int, block_size: int, block_num: int, num_factors: int) -> bool:
    if block_size <= 0 or block_num <= 0 or block_size * block_num != in_features:
        return False
    internal_factors = num_factors - 1
    if internal_factors <= 0:
        return True
    return (
        block_num % (2**internal_factors) == 0
        and block_size % 2 == 0
        and block_num % 2 == 0
        and in_features % (block_size * (2**internal_factors)) == 0
    )


def _choose_boft_shape(
    in_features: int,
    requested_block_size: int,
    requested_block_num: int,
    num_factors: int,
    auto_adjust: bool,
):
    if requested_block_size and requested_block_num:
        raise ValueError("Specify only one of boft_block_size or boft_block_num")

    if requested_block_num:
        if in_features % requested_block_num != 0:
            raise ValueError(f"in_features ({in_features}) must be divisible by boft_block_num ({requested_block_num})")
        block_num = requested_block_num
        block_size = in_features // block_num
        if not _valid_boft_shape(in_features, block_size, block_num, num_factors):
            raise ValueError(
                f"Invalid BOFT shape: in_features={in_features}, block_size={block_size}, "
                f"block_num={block_num}, boft_n_butterfly_factor={num_factors}"
            )
        return block_size, block_num

    requested_block_size = requested_block_size or 4
    if in_features % requested_block_size == 0:
        block_size = requested_block_size
    elif auto_adjust:
        block_size = adjust_block_size(in_features, requested_block_size)
    else:
        raise ValueError(f"in_features ({in_features}) must be divisible by boft_block_size ({requested_block_size})")

    block_num = in_features // block_size
    if _valid_boft_shape(in_features, block_size, block_num, num_factors):
        return block_size, block_num

    if not auto_adjust:
        raise ValueError(
            f"Invalid BOFT shape: in_features={in_features}, block_size={block_size}, "
            f"block_num={block_num}, boft_n_butterfly_factor={num_factors}"
        )

    divisors = [d for d in range(1, in_features + 1) if in_features % d == 0]
    divisors.sort(key=lambda d: (abs(d - requested_block_size), d))
    for candidate_size in divisors:
        candidate_num = in_features // candidate_size
        if _valid_boft_shape(in_features, candidate_size, candidate_num, num_factors):
            return candidate_size, candidate_num

    raise ValueError(
        f"Cannot find a valid BOFT block shape for in_features={in_features}, "
        f"requested_block_size={requested_block_size}, boft_n_butterfly_factor={num_factors}"
    )


class BOFTModule(nn.Module):
    def __init__(
        self,
        lora_name: str,
        root_prefix: str,
        original_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        block_size: int = 4,
        block_num: int = 0,
        boft_n_butterfly_factor: int = 1,
        dropout: Optional[float] = None,
        auto_adjust: bool = True,
    ):
        super().__init__()
        if boft_n_butterfly_factor < 1:
            raise ValueError("boft_n_butterfly_factor must be a positive integer")

        self.lora_name = lora_name
        self.root_prefix = root_prefix
        self.original_name = original_name
        self.multiplier = multiplier
        self.enabled = True

        self.org_module_ref = [org_module]
        self.org_module = [org_module]
        self.org_forward = None

        self.in_features = get_in_features(org_module)
        self.out_features = get_out_features(org_module)
        self.boft_n_butterfly_factor = boft_n_butterfly_factor
        self.boft_block_size, self.boft_block_num = _choose_boft_shape(
            self.in_features,
            block_size,
            block_num,
            boft_n_butterfly_factor,
            auto_adjust,
        )

        self.boft_R = nn.Parameter(
            torch.zeros(boft_n_butterfly_factor, self.boft_block_num, self.boft_block_size, self.boft_block_size)
        )
        self.boft_s = nn.Parameter(torch.ones(self.out_features, 1))
        self.dropout = MultiplicativeDropoutLayer(p=dropout or 0.0)

        p = torch.empty(boft_n_butterfly_factor, self.in_features, self.in_features)
        internal_factors = boft_n_butterfly_factor - 1
        for i in range(boft_n_butterfly_factor):
            perm = self.block_butterfly_perm(
                self.in_features,
                int(self.boft_block_num / (2**i)),
                int(self.boft_block_size / 2),
                internal_factors,
            )
            p[i] = self.perm2mat(perm)
        self.register_buffer("boft_P", p, persistent=False)

    def apply_to(self):
        self.org_forward = self.org_module_ref[0].forward
        self.org_module_ref[0].forward = self.forward

    def set_network(self, network):
        self.network = network

    @staticmethod
    def perm2mat(indices):
        n = len(indices)
        perm_mat = torch.zeros((n, n))
        for i, idx in enumerate(indices):
            perm_mat[i, idx] = 1
        return perm_mat

    @staticmethod
    def block_butterfly_perm(n, b, r=3, n_butterfly_factor=1):
        if n_butterfly_factor == 0:
            return torch.arange(n)
        if b * r * 2 > n:
            raise ValueError("Invalid number of blocks")

        block_size = int(n // b)
        indices = torch.arange(n)

        def sort_block(block_len, base):
            step = block_len / base
            initial_order = torch.arange(block_len)
            sorted_order = torch.empty(block_len, dtype=torch.long)
            evens = torch.arange(0, step, 2)
            odds = torch.arange(1, step, 2)
            sorted_seq = torch.cat((evens, odds), dim=0)
            for j, pos in enumerate(sorted_seq):
                sorted_order[int(j * base) : int(j * base + base)] = initial_order[int(pos * base) : int(pos * base + base)]
            return sorted_order

        sorted_order = sort_block(block_size, r)
        for i in range(0, n, block_size):
            block_end = i + block_size
            tmp_indices = indices[i:block_end]
            indices[i:block_end] = tmp_indices[sorted_order]
        return indices

    @staticmethod
    def cayley_batch(data):
        previous_dtype = data.dtype
        data = data.to(torch.float32)
        batch_size, rows, cols = data.shape
        skew_mat = 0.5 * (data - data.transpose(1, 2))
        id_mat = torch.eye(rows, device=data.device, dtype=data.dtype).unsqueeze(0).expand(batch_size, rows, cols)
        return torch.linalg.solve(id_mat + skew_mat, id_mat - skew_mat, left=False).to(previous_dtype)

    def get_rotation_and_scale(self, apply_dropout=True):
        n_factors, block_num, block_size, _ = self.boft_R.shape
        rotation_blocks = self.cayley_batch(self.boft_R.view(n_factors * block_num, block_size, block_size))
        rotation_blocks = rotation_blocks.view(n_factors, block_num, block_size, block_size)
        if apply_dropout:
            rotation_blocks = self.dropout(rotation_blocks)

        matrices = []
        p = self.boft_P.to(device=rotation_blocks.device, dtype=rotation_blocks.dtype)
        for i in range(n_factors):
            block_diagonal = torch.block_diag(*torch.unbind(rotation_blocks[i]))
            matrices.append(p[i] @ (block_diagonal @ p[i].transpose(0, 1)))

        rotation = matrices[0]
        for i in range(1, len(matrices)):
            rotation = matrices[i] @ rotation
        return rotation, self.boft_s

    def _rotated_weight(self, weight, apply_dropout=True):
        rotation, scale = self.get_rotation_and_scale(apply_dropout=apply_dropout)
        weight_shape = weight.shape
        flat_weight = weight.reshape(weight_shape[0], -1)
        rotated = rotation.to(flat_weight.dtype) @ flat_weight.transpose(0, 1)
        rotated = rotated.transpose(0, 1) * scale.to(flat_weight.dtype)
        return rotated.reshape(weight_shape)

    def _adapt_state_for_local_shape(self, sd):
        if "boft_s" in sd and sd["boft_s"].shape != self.boft_s.shape:
            if sd["boft_s"].transpose(0, 1).shape == self.boft_s.shape:
                sd = dict(sd)
                sd["boft_s"] = sd["boft_s"].transpose(0, 1)
        return sd

    def load_local_state(self, sd):
        sd = self._adapt_state_for_local_shape(sd)
        return self.load_state_dict(sd, strict=False)

    def forward(self, x, *args, **kwargs):
        if not self.enabled or self.multiplier == 0.0:
            return self.org_forward(x, *args, **kwargs)

        org_module = self.org_module_ref[0]
        weight = org_module.weight
        rotated_weight = self._rotated_weight(weight, apply_dropout=True)
        if self.multiplier != 1.0:
            rotated_weight = weight + (rotated_weight.to(weight.dtype) - weight) * self.multiplier

        bias = org_module.bias.to(dtype=x.dtype) if org_module.bias is not None else None
        if is_linear_module(org_module):
            return F.linear(x, rotated_weight.to(dtype=x.dtype), bias)

        return F.conv2d(
            x,
            rotated_weight.to(dtype=x.dtype),
            bias,
            org_module.stride,
            org_module.padding,
            org_module.dilation,
            org_module.groups,
        )

    def merge_to(self, sd=None, dtype=None, device=None):
        if sd:
            sd = {k: v.to(device=device) if device is not None else v for k, v in sd.items()}
            self.load_local_state(sd)

        org_module = self.org_module_ref[0]
        org_sd = org_module.state_dict()
        org_weight = org_sd["weight"]
        if device is not None:
            self.to(device)
            org_weight = org_weight.to(device)

        rotated = self._rotated_weight(org_weight, apply_dropout=False)
        if self.multiplier != 1.0:
            rotated = org_weight + (rotated.to(org_weight.dtype) - org_weight) * self.multiplier

        org_sd["weight"] = rotated.to(dtype if dtype is not None else org_sd["weight"].dtype)
        org_module.load_state_dict(org_sd)


class BOFTInfModule(BOFTModule):
    pass


class BOFTNetwork(nn.Module):
    PARAM_NAMES = ("boft_R", "boft_s")

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier: float = 1.0,
        block_size: int = 4,
        block_num: int = 0,
        boft_n_butterfly_factor: int = 1,
        dropout: Optional[float] = None,
        enable_conv: bool = False,
        modules_state: Optional[Dict[str, torch.Tensor]] = None,
        module_class=BOFTModule,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        auto_adjust: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.block_size = block_size
        self.block_num = block_num
        self.boft_n_butterfly_factor = boft_n_butterfly_factor
        self.dropout = dropout
        self.enable_conv = enable_conv
        self.module_class = module_class
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.auto_adjust = auto_adjust

        logger.info(
            f"create BOFT network. block_size: {block_size}, block_num: {block_num}, "
            f"boft_n_butterfly_factor: {boft_n_butterfly_factor}, dropout: {dropout}, "
            f"enable_conv: {enable_conv}, multiplier: {multiplier}"
        )

        text_encoders = text_encoder_list(text_encoder)
        self.text_encoder_loras = []
        for i, text_encoder_item in enumerate(text_encoders):
            prefix = LORA_PREFIX_TEXT_ENCODER if len(text_encoders) == 1 else (
                LORA_PREFIX_TEXT_ENCODER1 if i == 0 else LORA_PREFIX_TEXT_ENCODER2
            )
            text_loras, skipped = self._create_modules(
                prefix,
                text_encoder_item,
                TEXT_ENCODER_TARGET_REPLACE_MODULE,
                modules_state,
            )
            self.text_encoder_loras.extend(text_loras)
            if verbose and skipped:
                logger.info(f"skipped Text Encoder modules: {skipped}")
            logger.info(f"create BOFT for Text Encoder {i + 1}: {len(text_loras)} modules.")

        target_modules = list(UNET_TARGET_REPLACE_MODULE)
        if enable_conv or modules_state is not None:
            target_modules.extend(UNET_TARGET_REPLACE_MODULE_CONV2D_3X3)
        self.unet_loras, skipped_unet = self._create_modules(LORA_PREFIX_UNET, unet, target_modules, modules_state)
        if verbose and skipped_unet:
            logger.info(f"skipped U-Net modules: {skipped_unet}")
        logger.info(f"create BOFT for U-Net: {len(self.unet_loras)} modules.")

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def _pattern_allowed(self, original_name: str) -> bool:
        import re

        if self.exclude_patterns and any(re.fullmatch(pattern, original_name) for pattern in self.exclude_patterns):
            if not self.include_patterns or not any(re.fullmatch(pattern, original_name) for pattern in self.include_patterns):
                return False
        if self.include_patterns:
            return any(re.fullmatch(pattern, original_name) for pattern in self.include_patterns)
        return True

    def _infer_from_state(self, state, block_size, block_num, num_factors):
        if "boft_R" not in state:
            return block_size, block_num, num_factors
        boft_r = state["boft_R"]
        return boft_r.shape[2], 0, boft_r.shape[0]

    def _create_modules(self, prefix, root_module, target_replace_modules, modules_state):
        loras = []
        skipped = []
        if root_module is None:
            return loras, skipped

        for name, module in root_module.named_modules():
            if module.__class__.__name__ not in target_replace_modules:
                continue
            for child_name, child_module in module.named_modules():
                if not (is_linear_module(child_module) or is_conv2d_module(child_module)):
                    continue
                if is_conv2d_module(child_module) and not (is_conv2d_1x1(child_module) or self.enable_conv or modules_state is not None):
                    continue

                original_name = f"{name}.{child_name}" if child_name else name
                if not self._pattern_allowed(original_name):
                    continue

                lora_name = native_prefix(prefix, original_name)
                state = {}
                if modules_state is not None:
                    state, _ = extract_module_state(modules_state, lora_name, prefix, original_name, self.PARAM_NAMES)
                    if not state:
                        skipped.append(lora_name)
                        continue

                module_block_size, module_block_num, module_num_factors = self._infer_from_state(
                    state, self.block_size, self.block_num, self.boft_n_butterfly_factor
                )
                lora = self.module_class(
                    lora_name,
                    prefix,
                    original_name,
                    child_module,
                    self.multiplier,
                    module_block_size,
                    module_block_num,
                    module_num_factors,
                    self.dropout,
                    self.auto_adjust,
                )
                if state:
                    lora.load_local_state(state)
                loras.append(lora)

        return loras, skipped

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def _adapt_state(self, lora, state):
        if "boft_s" in state and state["boft_s"].shape != lora.boft_s.shape:
            if state["boft_s"].transpose(0, 1).shape == lora.boft_s.shape:
                state = dict(state)
                state["boft_s"] = state["boft_s"].transpose(0, 1)
        return state

    def load_weights(self, file):
        weights_sd = load_weights_sd(file)
        converted = {}
        used = []
        for lora in self.text_encoder_loras + self.unet_loras:
            state, state_used = extract_module_state(weights_sd, lora.lora_name, lora.root_prefix, lora.original_name, self.PARAM_NAMES)
            state = self._adapt_state(lora, state)
            for key, value in state.items():
                converted[f"{lora.lora_name}.{key}"] = value
            used.extend(state_used)
        unused = sorted(set(weights_sd.keys()) - set(used))
        if unused:
            logger.warning(f"ignored {len(unused)} BOFT weight keys that do not match current targets")
        return self.load_state_dict(converted, strict=False)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if not apply_text_encoder:
            self.text_encoder_loras = []
        if not apply_unet:
            self.unet_loras = []

        logger.info(f"enable BOFT for text encoder: {len(self.text_encoder_loras)} modules")
        logger.info(f"enable BOFT for U-Net: {len(self.unet_loras)} modules")

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        for lora in self.text_encoder_loras + self.unet_loras:
            state, _ = extract_module_state(weights_sd, lora.lora_name, lora.root_prefix, lora.original_name, self.PARAM_NAMES)
            state = self._adapt_state(lora, state)
            if state:
                lora.merge_to(state, dtype, device)
        logger.info("BOFT weights are merged")

    def _assemble_params(self, loras, lr):
        params = []
        for lora in loras:
            params.extend(list(lora.parameters()))
        if not params:
            return []
        group = {"params": params}
        if lr is not None:
            group["lr"] = lr
        if group.get("lr", 1.0) == 0:
            return []
        return [group]

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr]
        elif isinstance(text_encoder_lr, (float, int)):
            text_encoder_lr = [float(text_encoder_lr)]

        self.requires_grad_(True)
        params = []
        descriptions = []

        for te_idx, prefix in enumerate((LORA_PREFIX_TEXT_ENCODER, LORA_PREFIX_TEXT_ENCODER1, LORA_PREFIX_TEXT_ENCODER2)):
            te_loras = [lora for lora in self.text_encoder_loras if lora.lora_name.startswith(prefix)]
            if not te_loras:
                continue
            lr = text_encoder_lr[te_idx] if te_idx < len(text_encoder_lr) else text_encoder_lr[0]
            groups = self._assemble_params(te_loras, lr)
            params.extend(groups)
            descriptions.extend([f"textencoder {te_idx + 1}"] * len(groups))

        unet_groups = self._assemble_params(self.unet_loras, unet_lr if unet_lr is not None else default_lr)
        params.extend(unet_groups)
        descriptions.extend(["unet"] * len(unet_groups))
        return params, descriptions

    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        return self.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, unet_lr, default_lr)[0]

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        save_weights_sd(file, self.state_dict(), dtype, metadata)

    def backup_weights(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                org_module._lora_org_weight = org_module.state_dict()["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        for lora in self.text_encoder_loras + self.unet_loras:
            org_module = lora.org_module_ref[0]
            lora.merge_to()
            org_module._lora_restored = False
            lora.enabled = False


def _create_network_args(network_dim, neuron_dropout, kwargs):
    explicit_block_size = kwargs.get("boft_block_size", kwargs.get("block_size", None))
    explicit_block_num = kwargs.get("boft_block_num", kwargs.get("block_num", None))
    block_size = str_to_int(explicit_block_size, None)
    block_num = str_to_int(explicit_block_num, 0)
    if block_size is None:
        block_size = network_dim if network_dim is not None else 4

    boft_n_butterfly_factor = str_to_int(kwargs.get("boft_n_butterfly_factor", None), 1)
    dropout = neuron_dropout if neuron_dropout is not None else str_to_float(kwargs.get("boft_dropout", kwargs.get("dropout", None)), None)
    enable_conv = str_to_bool(kwargs.get("enable_conv", None), False)
    auto_adjust = str_to_bool(kwargs.get("auto_adjust", None), True)
    include_patterns = parse_patterns(kwargs.get("include_patterns", None))
    exclude_patterns = parse_patterns(kwargs.get("exclude_patterns", None))
    return {
        "block_size": block_size,
        "block_num": block_num,
        "boft_n_butterfly_factor": boft_n_butterfly_factor,
        "dropout": dropout,
        "enable_conv": enable_conv,
        "auto_adjust": auto_adjust,
        "include_patterns": include_patterns,
        "exclude_patterns": exclude_patterns,
    }


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
    del network_alpha, vae
    args = _create_network_args(network_dim, neuron_dropout, kwargs)
    return BOFTNetwork(text_encoder, unet, multiplier=multiplier, verbose=True, **args)


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    del vae
    weights_sd = load_weights_sd(file, weights_sd)
    args = _create_network_args(kwargs.get("network_dim", None), None, kwargs)
    module_class = BOFTInfModule if for_inference else BOFTModule
    network = BOFTNetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_state=weights_sd,
        module_class=module_class,
        verbose=True,
        **args,
    )
    return network, weights_sd
