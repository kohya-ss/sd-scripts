# OneTrainer/PEFT-style OFTv2 network module for SD1/SD2/SDXL training.

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
    triangular_block_size,
)

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MultiplicativeDropoutLayer(nn.Module):
    """Randomly replace learned rotation blocks with identity matrices."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p <= 0:
            return x

        if x.shape[-1] != x.shape[-2]:
            raise ValueError("The last two dimensions of input should be the same")

        blocks, block_size, _ = x.shape
        if blocks == 1:
            return x

        keep_prob = 1.0 - self.p
        mask = torch.empty(blocks, 1, 1, device=x.device, dtype=x.dtype).bernoulli_(p=keep_prob)
        eye = torch.eye(block_size, device=x.device, dtype=x.dtype).repeat(blocks, 1, 1)
        return mask * x + (1 - mask) * eye


class OFTRotationModule(nn.Module):
    def __init__(
        self,
        r: int,
        n_elements: int,
        block_size: int,
        in_features: int,
        coft: bool = False,
        coft_eps: float = 6e-5,
        block_share: bool = False,
        use_cayley_neumann: bool = True,
        num_cayley_neumann_terms: int = 5,
        dropout_probability: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.n_elements = n_elements
        self.block_size = block_size
        self.in_features = in_features
        self.coft = coft
        self.coft_eps = coft_eps
        self.block_share = block_share
        self.use_cayley_neumann = use_cayley_neumann
        self.num_cayley_neumann_terms = num_cayley_neumann_terms

        self.weight = nn.Parameter(torch.zeros(r, n_elements))

        rows, cols = torch.triu_indices(block_size, block_size, 1)
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.dropout = MultiplicativeDropoutLayer(p=dropout_probability)

    def _pytorch_skew_symmetric(self, vec, block_size):
        batch_size = vec.shape[0]
        matrix = torch.zeros(batch_size, block_size, block_size, device=vec.device, dtype=vec.dtype)

        batch_idx = torch.arange(batch_size, device=vec.device)[:, None]
        matrix = matrix.index_put((batch_idx, self.rows, self.cols), vec)
        return matrix - matrix.transpose(-2, -1)

    def _pytorch_skew_symmetric_inv(self, matrix, block_size):
        return matrix[:, self.rows, self.cols]

    def _cayley_batch(self, q, block_size, use_cayley_neumann=True, num_neumann_terms=5):
        batch_size, _ = q.shape
        previous_dtype = q.dtype
        q = q.to(torch.float32)
        q_skew = self._pytorch_skew_symmetric(q, block_size)

        if use_cayley_neumann:
            r = torch.eye(block_size, device=q.device, dtype=q.dtype).repeat(batch_size, 1, 1)
            if num_neumann_terms > 1:
                r.add_(q_skew, alpha=2.0)
                if num_neumann_terms > 2:
                    q_squared = torch.bmm(q_skew, q_skew)
                    r.add_(q_squared, alpha=2.0)

                    q_power = q_squared
                    for _ in range(3, num_neumann_terms - 1):
                        q_power = torch.bmm(q_power, q_skew)
                        r.add_(q_power, alpha=2.0)
                    q_power = torch.bmm(q_power, q_skew)
                    r.add_(q_power)
        else:
            id_mat = torch.eye(q_skew.shape[-1], device=q_skew.device, dtype=q_skew.dtype).unsqueeze(0)
            id_mat = id_mat.expand(batch_size, q_skew.shape[-1], q_skew.shape[-1])
            r = torch.linalg.solve(id_mat + q_skew, id_mat - q_skew, left=False)

        return r.to(previous_dtype)

    def _project_batch(self, q, coft_eps=1e-4):
        oft_r = self._pytorch_skew_symmetric(q, self.block_size)
        coft_eps = coft_eps * 1 / torch.sqrt(torch.tensor(oft_r.shape[0], device=oft_r.device, dtype=oft_r.dtype))
        origin = torch.zeros((oft_r.size(1), oft_r.size(1)), device=oft_r.device, dtype=oft_r.dtype).unsqueeze(0)
        origin = origin.expand_as(oft_r)
        diff = oft_r - origin
        norm_diff = torch.norm(diff, dim=(1, 2), keepdim=True)
        mask = (norm_diff <= coft_eps).bool()
        out = torch.where(mask, oft_r, origin + coft_eps * (diff / norm_diff))
        return self._pytorch_skew_symmetric_inv(out, self.block_size)

    def get_rotation_blocks(self, apply_dropout=True):
        weight = self.weight
        if self.coft:
            with torch.no_grad():
                weight = self._project_batch(weight, coft_eps=self.coft_eps)
                self.weight.copy_(weight)

        rotation = self._cayley_batch(
            weight,
            self.block_size,
            self.use_cayley_neumann,
            self.num_cayley_neumann_terms,
        )
        if apply_dropout:
            rotation = self.dropout(rotation)

        if self.block_share:
            rank = self.in_features // self.block_size
            rotation = rotation.repeat(rank, 1, 1)

        return rotation

    def get_weight(self, apply_dropout=True):
        rotation = self.get_rotation_blocks(apply_dropout=apply_dropout)
        return torch.block_diag(*torch.unbind(rotation))

    def forward(self, x):
        required_dtype = x.dtype
        if required_dtype != self.weight.dtype:
            x = x.to(self.weight.dtype)

        orig_shape = x.shape
        rank = self.in_features // self.block_size
        batch_dims = x.shape[:-1]
        x_reshaped = x.reshape(*batch_dims, rank, self.block_size)
        rotation = self.get_rotation_blocks(apply_dropout=True)
        x_rotated = torch.einsum("...rk,rkc->...rc", x_reshaped, rotation)
        return x_rotated.reshape(*orig_shape).to(required_dtype)


class OFTv2Module(nn.Module):
    def __init__(
        self,
        lora_name: str,
        root_prefix: str,
        original_name: str,
        org_module: nn.Module,
        multiplier: float = 1.0,
        block_size: int = 32,
        coft: bool = False,
        coft_eps: float = 6e-5,
        block_share: bool = False,
        dropout: Optional[float] = None,
        auto_adjust: bool = True,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.root_prefix = root_prefix
        self.original_name = original_name
        self.multiplier = multiplier
        self.coft = coft
        self.coft_eps = coft_eps
        self.block_share = block_share
        self.enabled = True

        self.org_module_ref = [org_module]
        self.org_module = [org_module]
        self.org_forward = None

        self.in_features = get_in_features(org_module)
        if auto_adjust:
            block_size = adjust_block_size(self.in_features, block_size)
        elif self.in_features % block_size != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by block_size ({block_size})")

        self.block_size = block_size
        self.rank = self.in_features // self.block_size
        n_elements = self.block_size * (self.block_size - 1) // 2
        r = 1 if block_share else self.rank
        self.oft_R = OFTRotationModule(
            r=r,
            n_elements=n_elements,
            block_size=self.block_size,
            in_features=self.in_features,
            coft=coft,
            coft_eps=coft_eps,
            block_share=block_share,
            dropout_probability=dropout or 0.0,
        )

    def apply_to(self):
        self.org_forward = self.org_module_ref[0].forward
        self.org_module_ref[0].forward = self.forward

    def set_network(self, network):
        self.network = network

    def forward(self, x, *args, **kwargs):
        if not self.enabled or self.multiplier == 0.0:
            return self.org_forward(x, *args, **kwargs)

        org_module = self.org_module_ref[0]
        if is_linear_module(org_module):
            rotated_x = self.oft_R(x)
            if self.multiplier != 1.0:
                rotated_x = x + (rotated_x - x) * self.multiplier
            return self.org_forward(rotated_x, *args, **kwargs)

        rotation = self.oft_R.get_rotation_blocks(apply_dropout=True)
        weight = org_module.weight
        rotated_weight = self._rotate_conv_weight(weight, rotation)
        if self.multiplier != 1.0:
            rotated_weight = weight + (rotated_weight - weight) * self.multiplier
        return F.conv2d(
            x,
            rotated_weight.to(dtype=x.dtype),
            org_module.bias.to(dtype=x.dtype) if org_module.bias is not None else None,
            org_module.stride,
            org_module.padding,
            org_module.dilation,
            org_module.groups,
        )

    def _rotate_conv_weight(self, weight, rotation_blocks):
        weight_dtype = weight.dtype
        weight_float = weight.to(rotation_blocks.dtype)
        weight_reshaped = weight_float.reshape(weight.shape[0], self.rank, self.block_size)
        rotated = torch.einsum("ork,rkc->orc", weight_reshaped, rotation_blocks)
        return rotated.reshape(weight.shape).to(weight_dtype)

    def load_local_state(self, sd):
        return self.load_state_dict(sd, strict=False)

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

        if is_linear_module(org_module):
            rotation = self.oft_R.get_weight(apply_dropout=False).to(device=org_weight.device)
            rotated = org_weight.to(rotation.dtype).reshape(org_weight.shape[0], -1) @ rotation.transpose(0, 1)
            rotated = rotated.reshape(org_weight.shape)
        else:
            rotation_blocks = self.oft_R.get_rotation_blocks(apply_dropout=False).to(device=org_weight.device)
            rotated = self._rotate_conv_weight(org_weight, rotation_blocks)

        if self.multiplier != 1.0:
            rotated = org_weight + (rotated.to(org_weight.dtype) - org_weight) * self.multiplier

        org_sd["weight"] = rotated.to(dtype if dtype is not None else org_sd["weight"].dtype)
        org_module.load_state_dict(org_sd)


class OFTv2InfModule(OFTv2Module):
    pass


class OFTv2Network(nn.Module):
    PARAM_NAMES = ("oft_R.weight",)

    def __init__(
        self,
        text_encoder,
        unet,
        multiplier: float = 1.0,
        block_size: int = 32,
        coft: bool = False,
        coft_eps: float = 6e-5,
        block_share: bool = False,
        dropout: Optional[float] = None,
        enable_conv: bool = False,
        modules_state: Optional[Dict[str, torch.Tensor]] = None,
        module_class=OFTv2Module,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        auto_adjust: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.multiplier = multiplier
        self.block_size = block_size
        self.coft = coft
        self.coft_eps = coft_eps
        self.block_share = block_share
        self.dropout = dropout
        self.enable_conv = enable_conv
        self.module_class = module_class
        self.include_patterns = include_patterns
        self.exclude_patterns = exclude_patterns
        self.auto_adjust = auto_adjust

        logger.info(
            f"create OFTv2 network. block_size: {block_size}, coft: {coft}, block_share: {block_share}, "
            f"dropout: {dropout}, enable_conv: {enable_conv}, multiplier: {multiplier}"
        )

        self.text_encoder_loras = []
        for i, text_encoder_item in enumerate(text_encoder_list(text_encoder)):
            prefix = LORA_PREFIX_TEXT_ENCODER if len(text_encoder_list(text_encoder)) == 1 else (
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
            logger.info(f"create OFTv2 for Text Encoder {i + 1}: {len(text_loras)} modules.")

        target_modules = list(UNET_TARGET_REPLACE_MODULE)
        if enable_conv or modules_state is not None:
            target_modules.extend(UNET_TARGET_REPLACE_MODULE_CONV2D_3X3)
        self.unet_loras, skipped_unet = self._create_modules(LORA_PREFIX_UNET, unet, target_modules, modules_state)
        if verbose and skipped_unet:
            logger.info(f"skipped U-Net modules: {skipped_unet}")
        logger.info(f"create OFTv2 for U-Net: {len(self.unet_loras)} modules.")

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

    def _infer_from_state(self, state, org_module, block_size, block_share):
        if "oft_R.weight" not in state:
            return block_size, block_share
        weight = state["oft_R.weight"]
        inferred_block_size = triangular_block_size(weight.shape[1])
        rank = get_in_features(org_module) // inferred_block_size
        inferred_block_share = weight.shape[0] == 1 and rank > 1
        return inferred_block_size, inferred_block_share

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

                module_block_size, module_block_share = self._infer_from_state(
                    state, child_module, self.block_size, self.block_share
                )
                lora = self.module_class(
                    lora_name,
                    prefix,
                    original_name,
                    child_module,
                    self.multiplier,
                    module_block_size,
                    self.coft,
                    self.coft_eps,
                    module_block_share,
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

    def load_weights(self, file):
        weights_sd = load_weights_sd(file)
        converted = {}
        used = []
        for lora in self.text_encoder_loras + self.unet_loras:
            state, state_used = extract_module_state(weights_sd, lora.lora_name, lora.root_prefix, lora.original_name, self.PARAM_NAMES)
            for key, value in state.items():
                converted[f"{lora.lora_name}.{key}"] = value
            used.extend(state_used)
        unused = sorted(set(weights_sd.keys()) - set(used))
        if unused:
            logger.warning(f"ignored {len(unused)} OFTv2 weight keys that do not match current targets")
        return self.load_state_dict(converted, strict=False)

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if not apply_text_encoder:
            self.text_encoder_loras = []
        if not apply_unet:
            self.unet_loras = []

        logger.info(f"enable OFTv2 for text encoder: {len(self.text_encoder_loras)} modules")
        logger.info(f"enable OFTv2 for U-Net: {len(self.unet_loras)} modules")

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        for lora in self.text_encoder_loras + self.unet_loras:
            state, _ = extract_module_state(weights_sd, lora.lora_name, lora.root_prefix, lora.original_name, self.PARAM_NAMES)
            if state:
                lora.merge_to(state, dtype, device)
        logger.info("OFTv2 weights are merged")

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
    block_size = str_to_int(kwargs.get("oft_block_size", kwargs.get("block_size", network_dim)), 32)
    coft = str_to_bool(kwargs.get("oft_coft", kwargs.get("coft", None)), False)
    coft_eps = str_to_float(kwargs.get("coft_eps", None), 6e-5)
    block_share = str_to_bool(kwargs.get("oft_block_share", kwargs.get("block_share", None)), False)
    dropout = neuron_dropout if neuron_dropout is not None else str_to_float(
        kwargs.get("dropout_probability", kwargs.get("dropout", None)), None
    )
    enable_conv = str_to_bool(kwargs.get("enable_conv", None), False)
    auto_adjust = str_to_bool(kwargs.get("auto_adjust", None), True)
    include_patterns = parse_patterns(kwargs.get("include_patterns", None))
    exclude_patterns = parse_patterns(kwargs.get("exclude_patterns", None))
    return {
        "block_size": block_size,
        "coft": coft,
        "coft_eps": coft_eps,
        "block_share": block_share,
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
    del network_alpha
    args = _create_network_args(network_dim, neuron_dropout, kwargs)
    return OFTv2Network(text_encoder, unet, multiplier=multiplier, verbose=True, **args)


def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    del vae
    weights_sd = load_weights_sd(file, weights_sd)
    args = _create_network_args(kwargs.get("network_dim", None), None, kwargs)
    module_class = OFTv2InfModule if for_inference else OFTv2Module
    network = OFTv2Network(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_state=weights_sd,
        module_class=module_class,
        verbose=True,
        **args,
    )
    return network, weights_sd
