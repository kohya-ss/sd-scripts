import importlib
import argparse
import inspect
import math
import os
import sys
import random
import time
import json
from dataclasses import dataclass
from multiprocessing import Value
import toml
from collections import Counter, deque
import numpy as np

from typing import Any, Dict, List, Optional

from tqdm import tqdm

import torch
import torch.distributed as dist
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed, DistributedType
from diffusers import DDPMScheduler
from library import deepspeed_utils, model_util

import library.train_util as train_util
from library.train_util import DreamBoothDataset
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.avg_ckpt_util import (
    average_state_dicts,
    filter_lora_state_dict,
    collect_last_checkpoints_with_epochs,
    load_lora_state_dict,
    save_lora_state_dict,
)
from library.utils import setup_logging, add_logging_arguments
from library.rounding_util import round_parameters
from accelerate.utils import broadcast

setup_logging()
import logging

logger = logging.getLogger(__name__)

_TORCH_GET_TOTAL_NORM = getattr(torch.nn.utils, "get_total_norm", None)
_FOREACH_GRAD_NORM_DISABLED = False


def _set_delta_fake_quant_compat(network, step, mode, **kwargs):
    """Call older network implementations without Triton-only kwargs."""
    setter = network.set_delta_fake_quant
    parameters = inspect.signature(setter).parameters
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if not accepts_kwargs:
        for name in ("use_triton", "triton_stats"):
            if name not in parameters:
                kwargs.pop(name, None)
    setter(step, mode, **kwargs)


def resolve_avg_proxy_candidate_modes(avg_cp_mode: str, avg_promote_pick: str, avg_mode: str) -> List[str]:
    if avg_cp_mode == "promote" and avg_promote_pick == "fixed":
        return [avg_mode]

    candidate_modes = ["ema", "uniform"]
    if avg_mode not in candidate_modes:
        candidate_modes.append(avg_mode)
    return candidate_modes


def _legacy_grad_norm(grads: List[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.tensor(0.0)

    grad_norm_sqr = torch.tensor(0.0, device=grads[0].device)
    for grad in grads:
        detached_grad = grad.detach()
        grad_norm_sqr += (detached_grad * detached_grad).sum()
    return torch.sqrt(grad_norm_sqr)


def _can_use_foreach_grad_norm(grads: List[torch.Tensor]) -> bool:
    if _FOREACH_GRAD_NORM_DISABLED or not grads:
        return False
    if not callable(_TORCH_GET_TOTAL_NORM) and not callable(getattr(torch, "_foreach_norm", None)):
        return False

    first_grad = grads[0]
    if (
        first_grad.dtype != torch.float32
        or first_grad.layout != torch.strided
        or first_grad.device.type not in ("cpu", "cuda")
    ):
        return False

    return all(
        grad.dtype == first_grad.dtype and grad.layout == torch.strided and grad.device == first_grad.device
        for grad in grads[1:]
    )


def _foreach_grad_norm(grads: List[torch.Tensor]) -> torch.Tensor:
    if callable(_TORCH_GET_TOTAL_NORM):
        total_norm = _TORCH_GET_TOTAL_NORM(grads, norm_type=2.0, error_if_nonfinite=False, foreach=True)
    else:
        per_grad_norms = torch._foreach_norm(grads, 2.0)
        total_norm = torch.linalg.vector_norm(torch.stack(per_grad_norms), 2.0)

    # The legacy path squares FP32 values before taking the square root. Preserve
    # its overflow/underflow classification while keeping the multi-tensor norm.
    return torch.sqrt(total_norm * total_norm)


def _is_unsupported_foreach_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in (
            "not implemented",
            "not supported",
            "unsupported",
            "can't use the foreach",
            "cannot use the foreach",
        )
    )


def _calculate_grad_norm(grads: List[torch.Tensor]) -> torch.Tensor:
    global _FOREACH_GRAD_NORM_DISABLED

    if _can_use_foreach_grad_norm(grads):
        try:
            return _foreach_grad_norm(grads)
        except (TypeError, NotImplementedError):
            _FOREACH_GRAD_NORM_DISABLED = True
        except RuntimeError as error:
            if not _is_unsupported_foreach_error(error):
                raise
            _FOREACH_GRAD_NORM_DISABLED = True
    return _legacy_grad_norm(grads)


@dataclass
class GradNormGuardianConfig:
    skip_grad_norm: bool
    log_grad_norm: bool
    log_grad_scale: bool
    log_grad_cosine: bool
    skip_grad_norm_max: Optional[float]
    nan_to_window: bool
    inf_to_window: bool
    skip_nan_immediate: bool
    skip_inf_immediate: bool
    moving_avg_window: int = 200
    log_flush_interval: int = 100
    initial_threshold: float = 200_000.0


class GradNormGuardian:
    def __init__(
        self,
        config: GradNormGuardianConfig,
        scaler_for_log=None,
        log_file_path: Optional[str] = None,
    ):
        self.config = config
        self.scaler_for_log = scaler_for_log if config.log_grad_scale else None
        self.log_file_path = log_file_path if config.log_grad_norm else None

        self.moving_avg_window = deque(maxlen=config.moving_avg_window)
        self.log_buffer: List[str] = []
        self.prev_grad_map = None
        self.prev_grad_norm = None
        self._cached_model = None
        self._cached_parameters = None

        if self.config.log_grad_norm and self.log_file_path is not None:
            with open(self.log_file_path, "w") as f:
                header = "Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff"
                if self.config.log_grad_scale:
                    header += ",Scale"
                if self.config.log_grad_cosine:
                    header += ",CosineSim"
                f.write(header + "\n")

    def _get_parameters(self, model):
        if self._cached_model is not model:
            # Training fixes parameter topology before the first step; only grad
            # presence changes later due to module dropout or TE freezing.
            self._cached_model = model
            self._cached_parameters = tuple(model.parameters())
        return self._cached_parameters

    def observe(self, model, epoch: int, step: int, loss_val: float) -> bool:
        parameters = self._get_parameters(model)
        use_cosine = self.config.log_grad_cosine

        with torch.no_grad():
            if not use_cosine:
                # Keep scaler-applied grads (pre-unscale) to retain fp16 scaling behavior.
                grads = [
                    param.grad.detach()
                    for param in parameters
                    if param.grad is not None
                ]
                current_grad_norm_tensor = _calculate_grad_norm(grads)
            else:
                device = parameters[0].device if parameters else torch.device("cpu")
                grad_norm_sqr = torch.tensor(0.0, device=device)
                dot_sum = torch.tensor(0.0, device=device) if self.prev_grad_map is not None else None
                cur_grads = {}
                grad_topology_changed = False

                for param in parameters:
                    if param.grad is None:
                        continue
                    grad = param.grad  # NOTE: keep scaler-applied grads (pre-unscale) to retain fp16 scaling behavior
                    grad_norm_sqr += (grad.detach() * grad.detach()).sum()
                    param_id = id(param)
                    if self.prev_grad_map is not None:
                        prev_grad = self.prev_grad_map.get(param_id)
                        if prev_grad is None or prev_grad.shape != grad.shape:
                            grad_topology_changed = True
                        else:
                            dot_sum += (grad.detach() * prev_grad).sum()
                    cur_grads[param_id] = grad.detach().clone()

                if self.prev_grad_map is not None:
                    grad_topology_changed = grad_topology_changed or set(cur_grads.keys()) != set(
                        self.prev_grad_map.keys()
                    )
                current_grad_norm_tensor = torch.sqrt(grad_norm_sqr)

        current_grad_norm = current_grad_norm_tensor.item()
        cosine_sim = None
        if use_cosine:
            if (
                self.prev_grad_map is not None
                and dot_sum is not None
                and self.prev_grad_norm is not None
                and not grad_topology_changed
            ):
                denom = current_grad_norm * (self.prev_grad_norm + 1e-12)
                if denom == 0.0:
                    cosine_sim = float("nan")
                else:
                    cosine_sim = (dot_sum / denom).item()
            else:
                cosine_sim = float("nan")
            self.prev_grad_map = cur_grads
            self.prev_grad_norm = current_grad_norm

        is_nan = math.isnan(current_grad_norm)
        is_inf = math.isinf(current_grad_norm)

        if not is_nan and not is_inf:
            self.moving_avg_window.append(current_grad_norm)
        else:
            if is_nan and self.config.nan_to_window:
                self.moving_avg_window.append(current_grad_norm)  # NOTE: intentionally poison the window so threshold stays NaN
            if is_inf and self.config.inf_to_window:
                self.moving_avg_window.append(current_grad_norm)  # NOTE: same idea for Inf; keep threshold disabled until flushed out

        if len(self.moving_avg_window) == self.moving_avg_window.maxlen:
            mean_norm = np.mean(self.moving_avg_window)
            std_norm = np.std(self.moving_avg_window)
            dynamic_threshold_pre_cap = mean_norm + 2.5 * std_norm
        else:
            dynamic_threshold_pre_cap = self.config.initial_threshold

        dynamic_threshold = dynamic_threshold_pre_cap
        if self.config.skip_grad_norm_max is not None and dynamic_threshold > self.config.skip_grad_norm_max:
            dynamic_threshold = self.config.skip_grad_norm_max
        if len(self.moving_avg_window) < self.moving_avg_window.maxlen:
            dynamic_threshold = dynamic_threshold_pre_cap

        if self.config.log_grad_norm:
            scale_val = self.scaler_for_log.get_scale() if self.config.log_grad_scale and self.scaler_for_log else None
            flag = 1 if math.isnan(dynamic_threshold) else 0
            log_line = f"{epoch},{step},{current_grad_norm},{dynamic_threshold},{loss_val},{flag}"
            if self.config.log_grad_scale:
                log_line += f",{scale_val}"
            if self.config.log_grad_cosine:
                log_line += f",{cosine_sim}"
            self.log_buffer.append(log_line + "\n")
            if step % self.config.log_flush_interval == 0 and self.log_file_path is not None:
                with open(self.log_file_path, "a") as f:
                    f.writelines(self.log_buffer)
                self.log_buffer.clear()

        if not self.config.skip_grad_norm:
            return False

        if (is_nan and self.config.skip_nan_immediate) or (is_inf and self.config.skip_inf_immediate):
            return True

        return current_grad_norm > dynamic_threshold


class GroupLossTracker:
    def __init__(self, beta: float):
        self.beta = beta
        self.ema_by_group: Dict[str, float] = {}
        self.count_by_group: Dict[str, int] = {}
        self.epoch_count_by_group: Dict[str, int] = {}
        self.epoch_loss_sum_by_group: Dict[str, float] = {}

    def update(self, group: str, loss: float):
        prev_ema = self.ema_by_group.get(group)
        if prev_ema is None:
            ema = loss
        else:
            ema = prev_ema * self.beta + loss * (1.0 - self.beta)

        self.ema_by_group[group] = ema
        self.count_by_group[group] = self.count_by_group.get(group, 0) + 1
        self.epoch_count_by_group[group] = self.epoch_count_by_group.get(group, 0) + 1
        self.epoch_loss_sum_by_group[group] = self.epoch_loss_sum_by_group.get(group, 0.0) + loss
        return ema, self.count_by_group[group]

    def get_epoch_summary(self):
        summaries = []
        for group in sorted(self.epoch_count_by_group.keys()):
            count_epoch = self.epoch_count_by_group[group]
            if count_epoch <= 0:
                continue
            mean_loss_epoch = self.epoch_loss_sum_by_group[group] / count_epoch
            summaries.append((group, self.ema_by_group.get(group), count_epoch, mean_loss_epoch))
        return summaries

    def reset_epoch(self):
        self.epoch_count_by_group.clear()
        self.epoch_loss_sum_by_group.clear()


GRAD_NORM_PRESETS = {
    "stable": {
        "skip_grad_norm": True,
        "log_grad_norm": True,
        "log_grad_cosine": False,
        "skip_grad_norm_max": 200000.0,
        "nan_to_window": True,
        "inf_to_window": True,
        "skip_nan_immediate": False,
        "skip_inf_immediate": False,
    },
    "stable_no_threshoff": {
        "skip_grad_norm": True,
        "log_grad_norm": True,
        "log_grad_cosine": False,
        "skip_grad_norm_max": 200000.0,
        "nan_to_window": False,
        "inf_to_window": False,
        "skip_nan_immediate": False,
        "skip_inf_immediate": False,
    },
    "gamble": {
        "skip_grad_norm": True,
        "log_grad_norm": True,
        "log_grad_cosine": False,
        "skip_grad_norm_max": None,
        "nan_to_window": False,
        "inf_to_window": False,
        "skip_nan_immediate": True,
        "skip_inf_immediate": True,
    },
}


def resolve_grad_norm_settings(args):
    grad_norm_mode = getattr(args, "grad_norm_mode", None)
    if grad_norm_mode is not None:
        preset = GRAD_NORM_PRESETS[grad_norm_mode]
        skip_grad_norm = preset["skip_grad_norm"]
        log_grad_norm = preset["log_grad_norm"]
        log_grad_cosine = preset["log_grad_cosine"]
        skip_grad_norm_max = preset["skip_grad_norm_max"]
        nan_to_window = preset["nan_to_window"]
        inf_to_window = preset["inf_to_window"]
        skip_nan_immediate = preset["skip_nan_immediate"]
        skip_inf_immediate = preset["skip_inf_immediate"]

        # Allow only explicit negation flags to override preset behavior.
        if getattr(args, "skip_nan_immediate", True) is False:
            skip_nan_immediate = False
        if getattr(args, "skip_inf_immediate", True) is False:
            skip_inf_immediate = False
    else:
        skip_grad_norm = getattr(args, "skip_grad_norm", False)
        log_grad_norm = getattr(args, "grad_norm_log", False)
        log_grad_cosine = getattr(args, "grad_cosine_log", False)
        skip_grad_norm_max = getattr(args, "skip_grad_norm_max", None)
        nan_to_window = getattr(args, "nan_to_window", False)
        inf_to_window = getattr(args, "inf_to_window", False)
        skip_nan_immediate = getattr(args, "skip_nan_immediate", True)
        skip_inf_immediate = getattr(args, "skip_inf_immediate", True)

    log_grad_cosine = log_grad_norm and log_grad_cosine

    return (
        grad_norm_mode,
        skip_grad_norm,
        log_grad_norm,
        log_grad_cosine,
        skip_grad_norm_max,
        nan_to_window,
        inf_to_window,
        skip_nan_immediate,
        skip_inf_immediate,
    )


DQ_DELTA_AUTO_PRESETS = {
    "default": {
        "clip_low": 0.0005,
        "clip_high": 0.003,
    },
    "clip_rate_high": {
        "clip_low": 0.003,
        "clip_high": 0.005,
    },
    "clip_rate_high_narrow": {
        "clip_low": 0.0038,
        "clip_high": 0.0048,
    },
    "clip_rate_mid": {
        "clip_low": 0.002,
        "clip_high": 0.004,
    },
    "clip_rate_low": {
        "clip_low": 0.0005,
        "clip_high": 0.0022,
    },
    "clip_rate_low_auto": {
        "clip_low": 0.0005,
        "clip_high": 0.0022,
    },
}

DQ_DELTA_AUTO_BANDS = {
    "default": (0.0005, 0.003),
    "high": (0.003, 0.005),
    "high_narrow": (0.0038, 0.0048),
    "mid": (0.002, 0.004),
    "low": (0.0005, 0.0022),
}


def resolve_dq_delta_auto_settings(args):
    auto_preset = getattr(args, "dq_delta_auto_preset", None)
    if auto_preset is not None:
        preset = DQ_DELTA_AUTO_PRESETS[auto_preset]
        dq_auto_clip_low = preset["clip_low"]
        dq_auto_clip_high = preset["clip_high"]
        dq_auto_mul_up = float(getattr(args, "dq_delta_auto_mul_up", 1.01))
        dq_auto_mul_down = float(getattr(args, "dq_delta_auto_mul_down", 0.995))
    else:
        dq_auto_clip_low = float(getattr(args, "dq_delta_auto_clip_low", 0.0005))
        dq_auto_clip_high = float(getattr(args, "dq_delta_auto_clip_high", 0.003))
        dq_auto_mul_up = float(getattr(args, "dq_delta_auto_mul_up", 1.01))
        dq_auto_mul_down = float(getattr(args, "dq_delta_auto_mul_down", 0.995))
    return auto_preset, dq_auto_clip_low, dq_auto_clip_high, dq_auto_mul_up, dq_auto_mul_down


class NetworkTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False
        self._te_lr_after_cfg = None
        self._te_lr_after_resume_state = None
        self._te_lr_after_resumed = False
        self._te_lr_after_resume_step = None
        self._te_freeze_cfg = None
        self._te_frozen_param_ids = set()
        self._te_frozen_state_dict = {}

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        for lr_desc, lr, i in self._get_lr_group_items(args, lr_scheduler, lr_descriptions):
            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )

        return logs

    def _get_lr_group_items(
        self,
        args: argparse.Namespace,
        lr_scheduler,
        lr_descriptions,
    ):
        items = []
        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"
            items.append((lr_desc, lr, i))
        return items

    def collect_rank_log_lr_snapshot(
        self,
        args: argparse.Namespace,
        lr_scheduler,
        lr_descriptions,
    ):
        scope_to_values = {
            "unet": [],
            "te1": [],
            "te2": [],
        }

        for lr_desc, lr, _ in self._get_lr_group_items(args, lr_scheduler, lr_descriptions):
            base_desc = (lr_desc or "").split()[0].lower()
            if base_desc.startswith("textencoder2"):
                scope_to_values["te2"].append(float(lr))
            elif base_desc.startswith("textencoder1") or base_desc == "textencoder":
                scope_to_values["te1"].append(float(lr))
            elif base_desc.startswith("unet"):
                scope_to_values["unet"].append(float(lr))

        snapshot = {}
        for scope, values in scope_to_values.items():
            if values:
                snapshot[f"{scope}_lr_min"] = min(values)
                snapshot[f"{scope}_lr_max"] = max(values)
            else:
                snapshot[f"{scope}_lr_min"] = None
                snapshot[f"{scope}_lr_max"] = None
        return snapshot

    def _parse_te_lr_after_option(self, raw_option):
        if raw_option is None:
            return None

        def _flatten(value):
            if isinstance(value, (list, tuple)):
                for v in value:
                    yield from _flatten(v)
            else:
                yield value

        tokens: List[str] = []
        for item in _flatten(raw_option):
            if isinstance(item, str):
                pieces = item.replace(",", " ").split()
                tokens.extend(pieces)
            else:
                tokens.append(str(item))

        if len(tokens) not in (2, 3):
            raise ValueError(
                "--te-lr-after expects 2 or 3 values: <ratio> <multiplier> [target(both|te1|te2)] / "
                "--te-lr-after には <割合> <倍率> [対象(both|te1|te2)] を指定してください"
            )

        try:
            ratio = float(tokens[0])
            multiplier = float(tokens[1])
        except ValueError as exc:
            raise ValueError(
                "failed to parse --te-lr-after values as numbers / --te-lr-after の値を数値として解釈できませんでした"
            ) from exc

        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(
                "--te-lr-after ratio must be between 0 and 1 / --te-lr-after の割合は0〜1の範囲で指定してください"
            )
        target_key = tokens[2].lower() if len(tokens) == 3 else "both"
        target_map = {
            "both": {0, 1},
            "all": {0, 1},
            "te": {0, 1},
            "te12": {0, 1},
            "12": {0, 1},
            "te1": {0},
            "1": {0},
            "te2": {1},
            "2": {1},
        }
        if target_key not in target_map:
            raise ValueError(
                f"unsupported --te-lr-after target '{target_key}' (use both|te1|te2) / "
                f"--te-lr-after の対象 '{target_key}' は未対応です（both|te1|te2 を使用してください）"
            )

        return {
            "ratio": ratio,
            "mult": multiplier,
            "target_indices": set(target_map[target_key]),
            "target_label": target_key,
            "threshold_step": None,
            "group_indices": None,
            "group_labels": [],
            "applied": False,
            "applied_step": None,
        }

    def _parse_te_freeze_options(self, args):
        cfg = {}
        for te_index, option_name in (
            (0, "te1_freeze_at"),
            (1, "te2_freeze_at"),
        ):
            value = getattr(args, option_name, None)
            if value is None:
                continue
            freeze_at = float(value)
            if freeze_at < 0.0:
                raise ValueError(f"--{option_name} must be >= 0.")
            cfg[te_index] = {
                "freeze_at": freeze_at,
                "threshold_step": None,
                "group_indices": [],
                "group_labels": [],
                "applied": False,
                "applied_step": None,
            }
        return cfg or None

    @staticmethod
    def _te_group_matches_description(description: str, te_index: int) -> bool:
        if not description:
            return False
        base = description.split()[0]
        if not base.startswith("textencoder"):
            return False
        suffix = base[len("textencoder") :]
        if not suffix:
            return te_index == 0
        digits = "".join(ch for ch in suffix if ch.isdigit())
        if not digits:
            return False
        try:
            return int(digits) - 1 == te_index
        except ValueError:
            return False

    def _get_param_group_lr(self, optimizer, group_idx: int):
        stack = [optimizer]
        visited = set()
        while stack:
            opt = stack.pop()
            if opt is None:
                continue
            if id(opt) in visited:
                continue
            visited.add(id(opt))
            param_groups = getattr(opt, "param_groups", None)
            if param_groups is not None and len(param_groups) > group_idx:
                return param_groups[group_idx].get("lr")
            if hasattr(opt, "optimizer"):
                stack.append(getattr(opt, "optimizer"))
            if hasattr(opt, "optimizers"):
                inners = getattr(opt, "optimizers")
                if inners:
                    stack.extend(inners)
        return None

    def _update_optimizer_group_lr(self, optimizer, group_idx: int, new_lr: float):
        if optimizer is None:
            return
        stack = [optimizer]
        visited = set()
        while stack:
            opt = stack.pop()
            if opt is None:
                continue
            if id(opt) in visited:
                continue
            visited.add(id(opt))
            param_groups = getattr(opt, "param_groups", None)
            if param_groups is not None and len(param_groups) > group_idx:
                group = param_groups[group_idx]
                group["lr"] = new_lr
                if "initial_lr" in group:
                    group["initial_lr"] = new_lr
            if hasattr(opt, "optimizer"):
                stack.append(getattr(opt, "optimizer"))
            if hasattr(opt, "optimizers"):
                inners = getattr(opt, "optimizers")
                if inners:
                    stack.extend(inners)

    def _iter_schedulers(self, scheduler):
        stack = [scheduler]
        visited = set()
        while stack:
            sched = stack.pop()
            if sched is None:
                continue
            if id(sched) in visited:
                continue
            visited.add(id(sched))
            yield sched
            for attr in ("scheduler", "_scheduler", "lr_scheduler"):
                if hasattr(sched, attr):
                    stack.append(getattr(sched, attr))
            if hasattr(sched, "schedulers"):
                nested = getattr(sched, "schedulers")
                if nested:
                    stack.extend(nested)

    def _update_scheduler_state_after_lr_change(self, lr_scheduler, group_idx: int, multiplier: float, new_lr: float):
        if lr_scheduler is None:
            return
        for sched in self._iter_schedulers(lr_scheduler):
            base_lrs = getattr(sched, "base_lrs", None)
            if base_lrs is not None and len(base_lrs) > group_idx:
                base_lrs[group_idx] *= multiplier
            last_lr = getattr(sched, "_last_lr", None)
            if last_lr is not None and len(last_lr) > group_idx:
                last_lr[group_idx] = new_lr
            elif hasattr(sched, "last_lr") and isinstance(getattr(sched, "last_lr"), list):
                lr_list = getattr(sched, "last_lr")
                if len(lr_list) > group_idx:
                    lr_list[group_idx] = new_lr
            if hasattr(sched, "optimizers"):
                optimizers = getattr(sched, "optimizers")
                if optimizers:
                    for opt in optimizers:
                        self._update_optimizer_group_lr(opt, group_idx, new_lr)
            elif hasattr(sched, "optimizer"):
                self._update_optimizer_group_lr(getattr(sched, "optimizer"), group_idx, new_lr)

    def _freeze_optimizer_group_params(self, optimizer, group_idx: int) -> int:
        frozen = 0
        stack = [optimizer]
        visited = set()
        seen_params = set()
        while stack:
            opt = stack.pop()
            if opt is None:
                continue
            if id(opt) in visited:
                continue
            visited.add(id(opt))
            param_groups = getattr(opt, "param_groups", None)
            if param_groups is not None and len(param_groups) > group_idx:
                for param in param_groups[group_idx].get("params", []):
                    if id(param) in seen_params:
                        continue
                    seen_params.add(id(param))
                    param.requires_grad_(False)
                    param.grad = None
                    self._te_frozen_param_ids.add(id(param))
                    frozen += 1
            if hasattr(opt, "optimizer"):
                stack.append(getattr(opt, "optimizer"))
            if hasattr(opt, "optimizers"):
                inners = getattr(opt, "optimizers")
                if inners:
                    stack.extend(inners)
        return frozen

    def _capture_frozen_te_state_dict(self, network) -> int:
        if not self._te_frozen_param_ids:
            return 0

        captured = 0
        for name, param in network.named_parameters():
            if id(param) not in self._te_frozen_param_ids:
                continue
            if name in self._te_frozen_state_dict:
                continue
            self._te_frozen_state_dict[name] = param.detach().cpu().clone()
            captured += 1
        return captured

    def _restore_frozen_te_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self._te_frozen_state_dict:
            return state_dict

        for key, frozen_tensor in self._te_frozen_state_dict.items():
            if key in state_dict:
                state_dict[key] = frozen_tensor.clone()
        return state_dict

    def _apply_te_freeze_if_ready(self, optimizer, network, global_step: int):
        if not self._te_freeze_cfg:
            return

        for te_index, cfg in sorted(self._te_freeze_cfg.items()):
            if cfg.get("applied"):
                continue
            threshold_step = cfg.get("threshold_step")
            group_indices = cfg.get("group_indices") or []
            if threshold_step is None or not group_indices:
                continue
            if global_step < threshold_step:
                continue

            frozen_params = 0
            for group_idx in group_indices:
                frozen_params += self._freeze_optimizer_group_params(optimizer, group_idx)
            frozen_state_keys = self._capture_frozen_te_state_dict(network)
            cfg["applied"] = True
            cfg["applied_step"] = global_step
            labels = cfg.get("group_labels") or [f"TE{te_index + 1}"]
            logger.info(
                "froze %s at step %d (threshold=%d, params=%d, state_keys=%d) / TE freeze: %s step=%d threshold=%d params=%d state_keys=%d",
                ", ".join(labels),
                global_step,
                threshold_step,
                frozen_params,
                frozen_state_keys,
                ", ".join(labels),
                global_step,
                threshold_step,
                frozen_params,
                frozen_state_keys,
            )

    def _apply_max_norm_regularization(self, network, max_norm_value, device):
        apply_fn = network.apply_max_norm_regularization
        if "exclude_param_ids" in inspect.signature(apply_fn).parameters:
            return apply_fn(max_norm_value, device, exclude_param_ids=self._te_frozen_param_ids)
        return apply_fn(max_norm_value, device)

    def _apply_te_lr_after_if_ready(self, optimizer, lr_scheduler, next_step_idx: int):
        cfg = self._te_lr_after_cfg
        if (
            cfg is None
            or cfg.get("applied")
            or cfg.get("threshold_step") is None
            or cfg.get("group_indices") is None
        ):
            return

        if next_step_idx <= cfg["threshold_step"]:
            return

        multiplier = cfg["mult"]
        for group_idx in cfg["group_indices"]:
            current_lr = self._get_param_group_lr(optimizer, group_idx)
            if current_lr is None:
                continue
            new_lr = current_lr * multiplier
            self._update_optimizer_group_lr(optimizer, group_idx, new_lr)
            self._update_scheduler_state_after_lr_change(lr_scheduler, group_idx, multiplier, new_lr)

        cfg["applied"] = True
        cfg["applied_step"] = next_step_idx
        target_desc = cfg.get("group_labels") or [f"TE{idx + 1}" for idx in sorted(cfg["target_indices"])]
        logger.info(
            "applied te_lr_after at step %d: scaled %s lr by %.6f / te_lr_after: ステップ%d超で %s の学習率に倍率%.6fを適用しました",
            next_step_idx,
            ", ".join(target_desc),
            multiplier,
            next_step_idx,
            ", ".join(target_desc),
            multiplier,
        )

    def _handle_te_lr_after_resume(self):
        cfg = self._te_lr_after_cfg
        if not cfg:
            return

        resume_state = self._te_lr_after_resume_state
        if resume_state is not None:
            applied = bool(resume_state.get("applied", False))
            cfg["applied"] = applied
            cfg["applied_step"] = resume_state.get("applied_step")
            if applied:
                logger.info(
                    "te_lr_after: restored applied state from checkpoint (step=%s) / te_lr_after: チェックポイントから適用済み状態を復元しました (ステップ=%s)",
                    cfg["applied_step"],
                    cfg["applied_step"],
                )
            return

        resume_step = self._te_lr_after_resume_step
        threshold = cfg.get("threshold_step")
        completed_step = None
        if resume_step is not None:
            completed_step = max(0, resume_step - 1)
        if (
            self._te_lr_after_resumed
            and completed_step is not None
            and threshold is not None
            and completed_step > threshold
        ):
            cfg["applied"] = True
            cfg["applied_step"] = completed_step
            logger.info(
                "te_lr_after: last completed step %d exceeded threshold %d; assuming multiplier already applied / "
                "te_lr_after: 再開時点の完了ステップ %d がしきい値 %d を超えているため、倍率適用済みと見なします",
                completed_step,
                threshold,
                completed_step,
                threshold,
            )
    def assert_extra_args(self, args, train_dataset_group):
        train_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype
    ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
        return encoder_hidden_states

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        if accelerator.num_processes <= 1:
            return
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def _set_network_multiplier_from_batch(self, network, batch):
        if not hasattr(network, "set_multiplier") or "network_multipliers" not in batch or batch["network_multipliers"] is None:
            return

        multipliers = batch["network_multipliers"]
        if isinstance(multipliers, torch.Tensor):
            if torch.all(multipliers == multipliers[0]):
                multipliers = multipliers.reshape(-1)[0].item()
            else:
                raise NotImplementedError("multipliers for each sample is not supported yet")
        network.set_multiplier(multipliers)

    def _get_text_conds_for_batch(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype, grad_enabled: bool):
        with torch.set_grad_enabled(grad_enabled), accelerator.autocast():
            if args.weighted_captions:
                tokenizer_arg = tokenizers if len(tokenizers) != 1 else tokenizers[0]
                text_encoder_arg = text_encoders if len(text_encoders) != 1 else text_encoders[0]
                return get_weighted_text_embeddings(
                    tokenizer_arg,
                    text_encoder_arg,
                    batch["captions"],
                    accelerator.device,
                    args.max_token_length // 75 if args.max_token_length else 1,
                    clip_skip=args.clip_skip,
                )
            return self.get_text_cond(args, accelerator, batch, tokenizers, text_encoders, weight_dtype)

    def _compute_batch_loss(
        self,
        args,
        accelerator,
        batch,
        noise_scheduler,
        unet,
        text_encoder_conds,
        noisy_latents,
        timesteps,
        target,
        huber_c,
        weight_dtype,
        train_unet: bool,
    ):
        with accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        loss = train_util.conditional_loss(
            noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
        )
        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]
        loss = loss * loss_weights

        if args.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
        if args.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if args.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
        if args.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

        return loss.mean()

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)

        # reset te-lr-after resume tracking state for each training run
        self._te_lr_after_resume_state = None
        self._te_lr_after_resumed = False
        self._te_lr_after_resume_step = None

        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)
        logger.info(
            f"avg_cp: {args.avg_cp}, avg_window: {args.avg_window}, avg_begin: {args.avg_begin}, "
            f"avg_mode: {args.avg_mode}, avg_reset_stats: {args.avg_reset_stats}"
        )
        if args.round_lora_step is not None and args.round_lora_step > 0:
            logger.info(
                f"lora rounding: step={args.round_lora_step}, mode={args.round_lora_mode}, "
                f"every={args.round_lora_every}, begin={args.round_lora_begin}"
            )

        self._te_lr_after_cfg = None
        try:
            self._te_lr_after_cfg = self._parse_te_lr_after_option(getattr(args, "te_lr_after", None))
        except ValueError as exc:
            logger.error(str(exc))
            raise
        self._te_freeze_cfg = self._parse_te_freeze_options(args)
        self._te_frozen_param_ids = set()
        self._te_frozen_state_dict = {}
        dq_begin_after_lr_warmup = bool(getattr(args, "dq_delta_begin_after_lr_warmup", False))
        if dq_begin_after_lr_warmup:
            lr_warmup_steps = getattr(args, "lr_warmup_steps", 0)
            if lr_warmup_steps is None or (isinstance(lr_warmup_steps, (int, float)) and lr_warmup_steps <= 0):
                logger.error(
                    "dq_delta_begin_after_lr_warmup is enabled but lr_warmup_steps is not specified (>0 required). / "
                    "dq_delta_begin_after_lr_warmup が有効ですが lr_warmup_steps が指定されていません（>0 が必要）。"
                )
                raise ValueError("dq_delta_begin_after_lr_warmup requires lr_warmup_steps > 0")
        # parse bits schedule if provided
        def _parse_bits_sched(spec: str):
            items = []
            if not spec:
                return items
            for part in spec.split(","):
                if not part:
                    continue
                k, v = part.split(":")
                p = float(k)
                b = int(v)
                assert 0.0 <= p <= 1.0, "progress must be in [0,1]"
                assert b > 0, "bits must be > 0"
                items.append((p, b))
            items.sort(key=lambda x: x[0])
            return items

        dq_bits_sched = _parse_bits_sched(getattr(args, "dq_delta_bits_sched", None))

        if ((getattr(args, "dq_delta_step", None) is not None and args.dq_delta_step and args.dq_delta_step > 0)
            or (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits) or dq_bits_sched):
            dq_begin_info = f"begin={args.dq_delta_begin}"
            if dq_begin_after_lr_warmup:
                dq_begin_info = f"begin_after_lr_warmup={getattr(args,'lr_warmup_steps',None)}"
            logger.info(
                f"lora fake-quant: target={'z' if getattr(args,'dq_quantize_z', False) else 'delta'}, "
                f"step={getattr(args,'dq_delta_step',None)}, bits={getattr(args,'dq_delta_bits',None)}, "
                f"mode={args.dq_delta_mode}, {dq_begin_info}, granularity={getattr(args,'dq_delta_granularity',None)}, "
                f"stat={getattr(args,'dq_delta_stat',None)}, range_mul={getattr(args,'dq_delta_range_mul',None)}, "
                f"bits_sched={dq_bits_sched}, use_triton={getattr(args,'dq_delta_use_triton', False)}, "
                f"triton_stats={getattr(args,'dq_delta_triton_stats', False)}"
            )

        dq_log_enabled = bool(getattr(args, "dq_delta_log", False))
        dq_log_every = max(1, int(getattr(args, "dq_delta_log_every", 100)))
        dq_log_scope = getattr(args, "dq_delta_log_scope", None) or getattr(args, "dq_delta_scope", "both")
        dq_log_mode = getattr(args, "dq_delta_log_mode", "summary")
        dq_log_detail = getattr(args, "dq_delta_log_detail", "basic")
        dq_log_extra = set(getattr(args, "dq_delta_log_extra", []) or [])
        dq_triton_enabled = bool(getattr(args, "dq_delta_use_triton", False))
        dq_triton_stats_enabled = bool(getattr(args, "dq_delta_triton_stats", False))
        if dq_triton_stats_enabled and not dq_triton_enabled:
            logger.warning(
                "--dq_delta_triton_stats requires --dq_delta_use_triton; stats will use the PyTorch path."
            )
        if dq_triton_enabled:
            unvalidated = []
            if dq_bits_sched:
                unvalidated.append("bits schedule")
            elif getattr(args, "dq_delta_bits", None) != 8:
                unvalidated.append(f"bits={getattr(args, 'dq_delta_bits', None)}")
            if getattr(args, "dq_delta_granularity", None) != "channel":
                unvalidated.append(f"granularity={getattr(args, 'dq_delta_granularity', None)}")
            if getattr(args, "dq_delta_stat", None) != "rms":
                unvalidated.append(f"stat={getattr(args, 'dq_delta_stat', None)}")
            if getattr(args, "dq_delta_mode", None) != "stoch":
                unvalidated.append(f"mode={getattr(args, 'dq_delta_mode', None)}")
            if getattr(args, "dq_quantize_z", False):
                unvalidated.append("target=z")
            if getattr(args, "dq_delta_scope", None) != "unet":
                unvalidated.append(f"scope={getattr(args, 'dq_delta_scope', None)}")
            if getattr(args, "mixed_precision", None) != "fp16":
                unvalidated.append(f"mixed_precision={getattr(args, 'mixed_precision', None)}")
            if unvalidated:
                logger.warning(
                    "dq_delta Triton is outside the end-to-end validated training profile "
                    "(8bit/channel/rms/stoch, delta, UNet, fp16): %s. Supported kernels are used where eligible; "
                    "other work falls back to PyTorch.",
                    ", ".join(unvalidated),
                )
        if (
            dq_triton_enabled
            and dq_triton_stats_enabled
            and dq_log_enabled
            and (dq_log_detail == "full" or dq_log_mode == "per_module")
        ):
            logger.warning(
                "dq_delta Triton stats: full/per_module LogSteps require detail fields and therefore use PyTorch "
                "stats. Fake-quant forward remains on the normal Triton path; eligible Auto-only/basic steps may "
                "still use fused stats."
            )
        if (
            dq_triton_enabled
            and dq_triton_stats_enabled
            and (
                getattr(args, "dq_delta_mode", None) != "stoch"
                or not (getattr(args, "dq_delta_bits", None) or dq_bits_sched)
            )
        ):
            logger.warning(
                "dq_delta Triton fused stats require bits mode with stochastic rounding; stats will use PyTorch "
                "for the current quantization mode."
            )
        if dq_triton_stats_enabled and not (dq_log_enabled or getattr(args, "dq_delta_auto_range_mul", False)):
            logger.warning("--dq_delta_triton_stats has no effect unless dq_delta log or auto stats are enabled.")
        if (
            dq_triton_enabled
            and dq_triton_stats_enabled
            and getattr(args, "dq_delta_auto_range_mul", False)
            and getattr(args, "dq_delta_auto_preset", None) != "clip_rate_low_auto"
        ):
            logger.warning(
                "dq_delta Triton stats currently fuse the qerr-basic set used by summary/basic logs and "
                "clip_rate_low_auto. With other auto presets, AutoSteps that do not overlap an eligible basic "
                "LogStep use PyTorch stats."
            )
        rank_log_enabled = bool(getattr(args, "rank_log", False))
        rank_log_every = max(1, int(getattr(args, "rank_log_every", 100)))
        rank_log_mode = getattr(args, "rank_log_mode", "summary")

        dq_auto_enabled = bool(getattr(args, "dq_delta_auto_range_mul", False))
        (
            dq_auto_preset,
            dq_auto_clip_low,
            dq_auto_clip_high,
            dq_auto_mul_up,
            dq_auto_mul_down,
        ) = resolve_dq_delta_auto_settings(args)
        dq_low_auto_enabled = dq_auto_enabled and dq_auto_preset == "clip_rate_low_auto"
        dq_auto_active_band = "low" if dq_auto_preset in ("clip_rate_low", "clip_rate_low_auto") else (
            "mid" if dq_auto_preset == "clip_rate_mid" else (
                "high_narrow" if dq_auto_preset == "clip_rate_high_narrow" else (
                    "high" if dq_auto_preset == "clip_rate_high" else ("default" if dq_auto_preset == "default" else "custom")
                )
            )
        )
        dq_auto_active_clip_low = dq_auto_clip_low
        dq_auto_active_clip_high = dq_auto_clip_high
        if dq_auto_preset is not None:
            logger.info(
                "dq_delta_auto_preset: %s (clip_low=%s, clip_high=%s, mul_up=%s, mul_down=%s)",
                dq_auto_preset,
                dq_auto_clip_low,
                dq_auto_clip_high,
                dq_auto_mul_up,
                dq_auto_mul_down,
            )
        dq_auto_every = max(1, int(getattr(args, "dq_delta_auto_every", 50)))
        dq_auto_min = float(getattr(args, "dq_delta_auto_min", 1.0))
        dq_auto_max = float(getattr(args, "dq_delta_auto_max", 6.0))
        dq_auto_ema = float(getattr(args, "dq_delta_auto_ema", 0.95))
        dq_auto_use_raw = bool(getattr(args, "dq_delta_auto_use_raw", False))
        dq_qerr_per_clip_floor = max(1e-12, float(getattr(args, "dq_delta_qerr_per_clip_floor", 0.001)))
        dq_log_error_parts = False
        dq_low_auto_min_progress = max(0.0, min(1.0, float(getattr(args, "dq_delta_clip_rate_low_auto_min_progress", 0.25))))
        dq_low_auto_bad_streak_threshold = max(1, int(getattr(args, "dq_delta_clip_rate_low_auto_bad_streak", 3)))
        dq_low_auto_freeze_progress = float(getattr(args, "dq_delta_clip_rate_low_auto_freeze_progress", 0.90))
        dq_low_auto_qerr_ratio_threshold = float(getattr(args, "dq_delta_clip_rate_low_auto_qerr_ratio", 0.25))
        dq_low_auto_qerr_per_clip_threshold = float(getattr(args, "dq_delta_clip_rate_low_auto_qerr_per_clip", 130.0))
        dq_auto_warmup_enabled = dq_auto_enabled and bool(getattr(args, "dq_delta_auto_warmup", True))
        dq_auto_warmup_updates_override = int(getattr(args, "dq_delta_auto_warmup_updates", 0))
        dq_auto_warmup_updates = 0
        if dq_auto_warmup_enabled:
            if dq_auto_warmup_updates_override > 0:
                dq_auto_warmup_updates = dq_auto_warmup_updates_override
            elif 0.0 < dq_auto_ema < 1.0:
                dq_auto_warmup_updates = int(math.ceil(2.0 / (1.0 - dq_auto_ema)))
            else:
                dq_auto_warmup_enabled = False
                logger.warning(
                    "dq_delta_auto_warmup is enabled but dq_delta_auto_ema is not in (0,1); warmup will be disabled."
                )
        dq_auto_log_format = getattr(args, "dq_delta_auto_log_format", "minimal")
        dq_auto_init_applied = 0
        dq_auto_init_value = None
        dq_auto_init_clip_target = None
        if dq_auto_enabled and bool(getattr(args, "dq_delta_auto_init_range_mul_from_band", False)):
            if args.dq_delta_stat != "rms":
                logger.warning(
                    "dq_delta_auto_init_range_mul_from_band is enabled but dq_delta_stat is not rms; init will be skipped."
                )
            else:
                clip_target = (dq_auto_active_clip_low + dq_auto_active_clip_high) / 2.0
                p = 1.0 - (clip_target / 2.0)
                try:
                    range_mul_init = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item()
                    if math.isfinite(range_mul_init):
                        range_mul_init = max(dq_auto_min, min(dq_auto_max, range_mul_init))
                        args.dq_delta_range_mul = range_mul_init
                        dq_auto_init_applied = 1
                        dq_auto_init_value = range_mul_init
                        dq_auto_init_clip_target = clip_target
                        logger.info(
                            "dq_delta_auto_init_range_mul_from_band applied: clip_target=%.6g, range_mul=%.6g",
                            clip_target,
                            range_mul_init,
                        )
                    else:
                        logger.warning(
                            "dq_delta_auto_init_range_mul_from_band produced non-finite value (clip_target=%.6g); init will be skipped.",
                            clip_target,
                        )
                except Exception as exc:
                    logger.warning(
                        "dq_delta_auto_init_range_mul_from_band failed: %s",
                        str(exc),
                    )

        dq_log_path = None
        dq_auto_log_path = None
        rank_log_path = None
        if dq_log_enabled:
            dq_log_path = getattr(args, "dq_delta_log_file", None)
            if dq_log_path is None:
                dq_log_path = os.path.join(args.output_dir, f"dq_delta_logs+{args.output_name}.txt")
        if dq_auto_enabled:
            dq_auto_log_path = getattr(args, "dq_delta_auto_log_file", None)
            if dq_auto_log_path is None:
                dq_auto_log_path = os.path.join(args.output_dir, f"dq_delta_auto+{args.output_name}.txt")
        if rank_log_enabled:
            rank_log_path = getattr(args, "rank_log_file", None)
            if rank_log_path is None:
                rank_log_path = os.path.join(args.output_dir, f"rank_logs+{args.output_name}.txt")

        dq_log_header_written = False
        dq_auto_log_header_written = False
        rank_log_header_written = False

        def _write_csv(path: str, header: str, line: str):
            nonlocal dq_log_header_written, dq_auto_log_header_written, rank_log_header_written
            if not path:
                return
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            if path == dq_log_path:
                header_written = dq_log_header_written
            elif path == dq_auto_log_path:
                header_written = dq_auto_log_header_written
            else:
                header_written = rank_log_header_written
            if not header_written:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(header + "\n")
                if path == dq_log_path:
                    dq_log_header_written = True
                elif path == dq_auto_log_path:
                    dq_auto_log_header_written = True
                else:
                    rank_log_header_written = True
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

        if dq_auto_enabled:
            if args.dq_delta_stat != "rms":
                logger.warning("dq_delta_auto_range_mul is enabled but dq_delta_stat is not rms; auto will be inactive.")
            if not ((args.dq_delta_bits is not None and args.dq_delta_bits) or dq_bits_sched):
                logger.warning("dq_delta_auto_range_mul is enabled but dq_delta_bits/bits_sched is not set; auto will be inactive.")

        def _dq_format_value(v):
            if v is None:
                return ""
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, (float, int)):
                return f"{v:.6g}"
            return str(v)

        def _dq_log_header(
            log_mode: str,
            include_near_zero: bool,
            include_qerr_per_clip: bool = False,
            detail: str = "basic",
        ):
            full_detail = detail == "full" or log_mode == "per_module"
            cols = [
                "Epoch",
                "TrainStep",
                "Scope",
                "Target",
                "Bits",
                "DQStepSize",
                "RangeMul",
                "Stat",
                "Granularity",
                "Mode",
            ]
            if log_mode == "per_module":
                cols += ["Module", "Shape"]
            cols += [
                "RMS",
                "ClipRateRaw",
                "ClipRateEMA",
                "QuantErrRMSRaw",
                "QuantErrRMSEMA",
                "QuantErrRatioRaw",
                "QuantErrRatioEMA",
            ]
            if full_detail:
                cols[cols.index("ClipRateRaw"):cols.index("ClipRateRaw")] = [
                    "AbsMax",
                    "Range",
                    "ScaleMin",
                    "ScaleMean",
                    "ScaleMax",
                    "Qmax",
                ]
                cols.insert(cols.index("QuantErrRMSRaw"), "ZeroRate")
            if include_near_zero:
                cols.append("NearZeroRate")
            if include_qerr_per_clip:
                cols += [
                    "QErrPerClip",
                    "QErrPerClipClipFloor",
                ]
            cols += [
                "ActiveClipBand",
                "ActiveClipLow",
                "ActiveClipHigh",
                "ClipRateLowAutoState",
                "ClipRateLowAutoBad",
                "ClipRateLowAutoBadStreak",
                "TrainProgress",
                "ClipRateLowAutoMinProgress",
                "ClipRateLowAutoFreezeProgress",
                "ClipRateLowAutoThresholdQErrRatio",
                "ClipRateLowAutoThresholdQErrPerClip",
                "ClipRateLowAutoPhase",
            ]
            if dq_log_error_parts:
                cols += [
                    "ClipErrRMS",
                    "RoundErrRMS",
                    "ClipErrRatio",
                    "RoundErrRatio",
                    "ClipShare",
                    "RoundShare",
                ]
            cols += [
                "Numel",
                "AutoApplied",
                "RangeMulBefore",
                "RangeMulAfter",
                "WarmupActive",
                "WarmupRemain",
                "AutoReason",
                "AutoInitMulApplied",
                "AutoInitMulValue",
                "AutoInitClipTarget",
            ]
            return ",".join(cols)

        def _rank_log_header(log_mode: str):
            cols = [
                "Epoch",
                "TrainStep",
                "Scope",
                "UnetLRMin",
                "UnetLRMax",
                "Te1LRMin",
                "Te1LRMax",
                "Te2LRMin",
                "Te2LRMax",
            ]
            if log_mode == "summary":
                cols += [
                    "RankDim",
                    "RankSatWMean",
                    "RankSatP50",
                    "RankSatP95",
                    "RankSatMax",
                    "RankTop1P95",
                    "RankEnergySum",
                ]
            else:
                cols += [
                    "Module",
                    "RankDim",
                    "RankSat",
                    "RankTop1",
                    "RankEnergy",
                ]
            return ",".join(cols)

        def _dq_auto_log_header(full_schema: bool, include_near_zero: bool):
            if full_schema:
                return _dq_log_header("summary", include_near_zero, include_qerr_per_clip=True, detail="full")
            return (
                "TrainStep,Scope,Target,Bits,ClipRateRaw,ClipRateEMA,RangeMulBefore,RangeMulAfter,AutoApplied,"
                "WarmupActive,WarmupRemain,AutoReason,AutoInitMulApplied,AutoInitMulValue,AutoInitClipTarget,"
                "QErrPerClip,QErrPerClipClipFloor,ActiveClipBand,ActiveClipLow,ActiveClipHigh,"
                "ClipRateLowAutoState,ClipRateLowAutoDecision,ClipRateLowAutoReason,ClipRateLowAutoBad,ClipRateLowAutoBadStreak,"
                "TrainProgress,ClipRateLowAutoMinProgress,ClipRateLowAutoFreezeProgress,"
                "ClipRateLowAutoThresholdQErrRatio,ClipRateLowAutoThresholdQErrPerClip,"
                "ClipRateLowAutoPhase,ClipRateLowAutoCanEscape"
            )

        def _dq_qerr_per_clip(quant_err_ratio, clip_rate):
            if quant_err_ratio is None or clip_rate is None:
                return None
            return quant_err_ratio / max(clip_rate, dq_qerr_per_clip_floor)

        def _dq_reduce_stats(
            accum_by_scope,
            collect_full: bool,
            collect_zero: bool,
            collect_near_zero: bool,
            collect_detail: bool,
        ):
            if accelerator.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
                return accum_by_scope

            scopes = ["unet", "te"]
            sum_fields = []
            sum_refs = []
            for scope in scopes:
                acc = accum_by_scope[scope]
                sum_fields.append(acc.numel)
                sum_refs.append((acc, "numel"))
                sum_fields.append(acc.clip_count)
                sum_refs.append((acc, "clip_count"))
                if collect_zero:
                    sum_fields.append(acc.zero_count)
                    sum_refs.append((acc, "zero_count"))
                if collect_near_zero:
                    sum_fields.append(acc.near_zero_count)
                    sum_refs.append((acc, "near_zero_count"))
                if collect_full:
                    sum_fields.append(acc.sumsq)
                    sum_refs.append((acc, "sumsq"))
                    sum_fields.append(acc.xq_sumsq)
                    sum_refs.append((acc, "xq_sumsq"))
                    sum_fields.append(acc.xxq_sum)
                    sum_refs.append((acc, "xxq_sum"))
                    if collect_detail:
                        sum_fields.append(acc.scale_sum)
                        sum_refs.append((acc, "scale_sum"))
                        sum_fields.append(acc.scale_count)
                        sum_refs.append((acc, "scale_count"))
                    if getattr(acc, "clip_err_sumsq", None) is not None:
                        sum_fields.append(acc.clip_err_sumsq)
                        sum_refs.append((acc, "clip_err_sumsq"))
                    if getattr(acc, "round_err_sumsq", None) is not None:
                        sum_fields.append(acc.round_err_sumsq)
                        sum_refs.append((acc, "round_err_sumsq"))

            if sum_fields:
                sum_vec = torch.stack(sum_fields)
                dist.all_reduce(sum_vec, op=dist.ReduceOp.SUM)
                for idx, (acc, name) in enumerate(sum_refs):
                    setattr(acc, name, sum_vec[idx])

            if collect_detail:
                max_fields = []
                max_refs = []
                min_fields = []
                min_refs = []
                for scope in scopes:
                    acc = accum_by_scope[scope]
                    max_fields.append(acc.absmax)
                    max_refs.append((acc, "absmax"))
                    max_fields.append(acc.scale_max)
                    max_refs.append((acc, "scale_max"))
                    min_fields.append(acc.scale_min)
                    min_refs.append((acc, "scale_min"))

                if max_fields:
                    max_vec = torch.stack(max_fields)
                    dist.all_reduce(max_vec, op=dist.ReduceOp.MAX)
                    for idx, (acc, name) in enumerate(max_refs):
                        setattr(acc, name, max_vec[idx])
                if min_fields:
                    min_vec = torch.stack(min_fields)
                    dist.all_reduce(min_vec, op=dist.ReduceOp.MIN)
                    for idx, (acc, name) in enumerate(min_refs):
                        setattr(acc, name, min_vec[idx])

            return accum_by_scope

        def _dq_compute_metrics(
            acc,
            qmax,
            collect_full: bool,
            collect_zero: bool,
            collect_near_zero: bool,
            collect_detail: bool,
        ):
            numel = acc.numel.item() if acc.numel is not None else 0.0
            clip_rate = (acc.clip_count / acc.numel).item() if numel > 0 else None
            zero_rate = (acc.zero_count / acc.numel).item() if collect_zero and numel > 0 else None
            near_zero_rate = (acc.near_zero_count / acc.numel).item() if collect_near_zero and numel > 0 else None
            rms = absmax = scale_min = scale_max = scale_mean = range_val = None
            quant_err_rms = quant_err_ratio = None
            clip_err_rms = round_err_rms = clip_err_ratio = round_err_ratio = clip_share = round_share = None
            total_err_sumsq = None
            if collect_full and numel > 0:
                rms = math.sqrt((acc.sumsq / acc.numel).item()) if acc.sumsq is not None else None
                if collect_detail:
                    absmax = acc.absmax.item() if acc.absmax is not None else None
                    scale_min = acc.scale_min.item() if acc.scale_min is not None else None
                    scale_max = acc.scale_max.item() if acc.scale_max is not None else None
                    if acc.scale_sum is not None and acc.scale_count is not None and acc.scale_count.item() > 0:
                        scale_mean = (acc.scale_sum / acc.scale_count).item()
                if acc.xq_sumsq is not None and acc.xxq_sum is not None and acc.sumsq is not None:
                    err_sumsq = acc.sumsq + acc.xq_sumsq - (2.0 * acc.xxq_sum)
                    err_sumsq = torch.clamp(err_sumsq, min=0.0)
                    total_err_sumsq = err_sumsq
                    quant_err_rms = math.sqrt((err_sumsq / acc.numel).item())
                    if rms is not None:
                        quant_err_ratio = quant_err_rms / (rms + 1e-12)
                if getattr(acc, "clip_err_sumsq", None) is not None and acc.clip_err_sumsq is not None:
                    clip_err_rms = math.sqrt((torch.clamp(acc.clip_err_sumsq, min=0.0) / acc.numel).item())
                    if rms is not None:
                        clip_err_ratio = clip_err_rms / (rms + 1e-12)
                if getattr(acc, "round_err_sumsq", None) is not None and acc.round_err_sumsq is not None:
                    round_err_rms = math.sqrt((torch.clamp(acc.round_err_sumsq, min=0.0) / acc.numel).item())
                    if rms is not None:
                        round_err_ratio = round_err_rms / (rms + 1e-12)
                if total_err_sumsq is not None and total_err_sumsq.item() > 0:
                    if getattr(acc, "clip_err_sumsq", None) is not None and acc.clip_err_sumsq is not None:
                        clip_share = (torch.clamp(acc.clip_err_sumsq, min=0.0) / (total_err_sumsq + 1e-12)).item()
                    if getattr(acc, "round_err_sumsq", None) is not None and acc.round_err_sumsq is not None:
                        round_share = (torch.clamp(acc.round_err_sumsq, min=0.0) / (total_err_sumsq + 1e-12)).item()
            if scale_mean is not None and qmax is not None:
                range_val = scale_mean * qmax
            return {
                "numel": numel,
                "clip_rate": clip_rate,
                "zero_rate": zero_rate,
                "near_zero_rate": near_zero_rate,
                "quant_err_rms": quant_err_rms,
                "quant_err_ratio": quant_err_ratio,
                "rms": rms,
                "absmax": absmax,
                "scale_min": scale_min,
                "scale_max": scale_max,
                "scale_mean": scale_mean,
                "range": range_val,
                "clip_err_rms": clip_err_rms,
                "round_err_rms": round_err_rms,
                "clip_err_ratio": clip_err_ratio,
                "round_err_ratio": round_err_ratio,
                "clip_share": clip_share,
                "round_share": round_share,
            }

        def _dq_merge_acc(
            acc_a,
            acc_b,
            collect_full: bool,
            collect_zero: bool,
            collect_near_zero: bool,
            collect_detail: bool,
        ):
            numel = acc_a.numel + acc_b.numel
            clip = acc_a.clip_count + acc_b.clip_count
            zero = acc_a.zero_count + acc_b.zero_count if collect_zero else None
            near_zero = acc_a.near_zero_count + acc_b.near_zero_count if collect_near_zero else None
            sumsq = absmax = scale_min = scale_max = scale_sum = scale_count = None
            xq_sumsq = xxq_sum = None
            clip_err_sumsq = round_err_sumsq = None
            if collect_full:
                sumsq = acc_a.sumsq + acc_b.sumsq
                xq_sumsq = acc_a.xq_sumsq + acc_b.xq_sumsq
                xxq_sum = acc_a.xxq_sum + acc_b.xxq_sum
                if collect_detail:
                    absmax = torch.maximum(acc_a.absmax, acc_b.absmax)
                    scale_min = torch.minimum(acc_a.scale_min, acc_b.scale_min)
                    scale_max = torch.maximum(acc_a.scale_max, acc_b.scale_max)
                    scale_sum = acc_a.scale_sum + acc_b.scale_sum
                    scale_count = acc_a.scale_count + acc_b.scale_count
                if getattr(acc_a, "clip_err_sumsq", None) is not None and getattr(acc_b, "clip_err_sumsq", None) is not None:
                    clip_err_sumsq = acc_a.clip_err_sumsq + acc_b.clip_err_sumsq
                if getattr(acc_a, "round_err_sumsq", None) is not None and getattr(acc_b, "round_err_sumsq", None) is not None:
                    round_err_sumsq = acc_a.round_err_sumsq + acc_b.round_err_sumsq
            acc_cls = type(acc_a)
            if "collect_error_parts" in inspect.signature(acc_cls).parameters:
                temp_acc = acc_cls(
                    acc_a.numel.device,
                    collect_full,
                    collect_zero,
                    collect_near_zero,
                    collect_detail=collect_detail,
                    collect_error_parts=(
                        getattr(acc_a, "collect_error_parts", False) or getattr(acc_b, "collect_error_parts", False)
                    ),
                )
            else:
                temp_acc = acc_cls(acc_a.numel.device, collect_full, collect_zero, collect_near_zero)
            temp_acc.numel = numel
            temp_acc.clip_count = clip
            temp_acc.zero_count = zero
            temp_acc.near_zero_count = near_zero
            temp_acc.sumsq = sumsq
            temp_acc.xq_sumsq = xq_sumsq
            temp_acc.xxq_sum = xxq_sum
            temp_acc.absmax = absmax
            temp_acc.scale_min = scale_min
            temp_acc.scale_max = scale_max
            temp_acc.scale_sum = scale_sum
            temp_acc.scale_count = scale_count
            temp_acc.clip_err_sumsq = clip_err_sumsq
            temp_acc.round_err_sumsq = round_err_sumsq
            return temp_acc

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # tokenizerは単体またはリスト、tokenizersは必ずリスト：既存のコードとの互換性のため
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        self.assert_extra_args(args, train_dataset_group)

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        self.cache_text_encoder_outputs_if_needed(
            args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype
        )

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)

        if self._te_lr_after_cfg and not train_text_encoder:
            logger.warning(
                "ignore te_lr_after because text encoder training is disabled / Text Encoderを学習しないため te_lr_after は無視されます"
            )
            self._te_lr_after_cfg = None

        num_text_encoders = len(text_encoders)
        te_selection_indices: List[int] = []
        te_targets_for_network: Optional[List[int]] = None
        if train_text_encoder:
            if args.network_te_train_targets:
                idx_map = {"te1": 0, "te2": 1}
                selected = []
                for target in args.network_te_train_targets:
                    target_lower = target.lower()
                    if target_lower not in idx_map:
                        raise ValueError(
                            f"unsupported text encoder target '{target}' / 未対応のText Encoderターゲット'{target}'が指定されています"
                        )
                    idx = idx_map[target_lower]
                    if idx >= num_text_encoders:
                        raise ValueError(
                            f"text encoder target '{target}' is unavailable: this model provides {num_text_encoders} text encoder(s) / Text Encoderターゲット'{target}'は無効です。このモデルには{num_text_encoders}個のText Encoderしかありません"
                        )
                    if idx not in selected:
                        selected.append(idx)

                if len(selected) == 0:
                    te_selection_indices = []
                    te_targets_for_network = []
                else:
                    te_selection_indices = selected
                    te_targets_for_network = selected
            else:
                te_selection_indices = list(range(num_text_encoders))
                te_targets_for_network = None
        else:
            te_selection_indices = []
            te_targets_for_network = []

        if train_text_encoder and args.network_te_train_targets:
            logger.info(
                "enable LoRA training for Text Encoder target(s): %s",
                ", ".join(f"TE{idx + 1}" for idx in te_selection_indices) if te_selection_indices else "(none)",
            )

        train_text_encoder = train_text_encoder and len(te_selection_indices) > 0

        if hasattr(network, "set_te_train_targets"):
            network.set_te_train_targets(te_targets_for_network)

        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        if self._te_lr_after_cfg:
            active_targets = {idx for idx in self._te_lr_after_cfg["target_indices"] if idx in te_selection_indices}
            if not active_targets:
                logger.warning(
                    "ignore te_lr_after because the specified text encoder target(s) are not selected / 指定されたText Encoderが学習対象外のため te_lr_after は無視されます"
                )
                self._te_lr_after_cfg = None
            else:
                self._te_lr_after_cfg["target_indices"] = active_targets

        if self._te_freeze_cfg:
            active_freeze_cfg = {
                idx: cfg for idx, cfg in self._te_freeze_cfg.items() if idx in te_selection_indices
            }
            if not active_freeze_cfg:
                logger.warning(
                    "ignore te freeze options because the specified text encoder target(s) are not selected"
                )
                self._te_freeze_cfg = None
            else:
                skipped = sorted(set(self._te_freeze_cfg.keys()) - set(active_freeze_cfg.keys()))
                if skipped:
                    logger.warning(
                        "ignore te freeze option(s) for unselected target(s): %s",
                        ", ".join(f"TE{idx + 1}" for idx in skipped),
                    )
                self._te_freeze_cfg = active_freeze_cfg
                if args.network_train_text_encoder_only and set(active_freeze_cfg.keys()) >= set(te_selection_indices):
                    raise ValueError(
                        "te freeze cannot freeze all selected Text Encoders when --network_train_text_encoder_only is used. "
                        "At least one trainable LoRA group must remain after freeze."
                    )

        # Configure LoRA delta fake-quantization if available
        if (((getattr(args, "dq_delta_step", None) is not None and args.dq_delta_step) or (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits) or dq_bits_sched) and hasattr(network, "set_delta_fake_quant")):
            unwrapped = accelerator.unwrap_model(network)
            _set_delta_fake_quant_compat(
                unwrapped,
                getattr(args, "dq_delta_step", None),
                args.dq_delta_mode,
                granularity=args.dq_delta_granularity,
                stat=args.dq_delta_stat,
                bits=getattr(args, "dq_delta_bits", None),
                range_mul=getattr(args, "dq_delta_range_mul", None),
                on_z=getattr(args, "dq_quantize_z", False),
                use_triton=getattr(args, "dq_delta_use_triton", False),
                triton_stats=getattr(args, "dq_delta_triton_stats", False),
            )
            # no EMA-based stats to propagate (ema_* removed)
            # Scope control: unet / te / both
            scope = getattr(args, "dq_delta_scope", "both")
            if scope == "unet" and hasattr(unwrapped, "text_encoder_loras"):
                for l in unwrapped.text_encoder_loras:
                    l.delta_q_enabled = False
                for l in unwrapped.unet_loras:
                    l.delta_q_enabled = True
            elif scope == "te" and hasattr(unwrapped, "unet_loras"):
                for l in unwrapped.unet_loras:
                    l.delta_q_enabled = False
                for l in unwrapped.text_encoder_loras:
                    l.delta_q_enabled = True

        if args.network_weights is not None:
            # FIXME consider alpha of weights
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        te_lr_overrides: Dict[int, float] = {}
        if train_text_encoder:
            if args.text_encoder_lr1 is not None:
                if 0 in te_selection_indices:
                    te_lr_overrides[0] = args.text_encoder_lr1
                else:
                    logger.warning(
                        "ignore text_encoder_lr1 because Text Encoder 1 is not selected / Text Encoder 1を学習対象にしていないためtext_encoder_lr1は無視されます"
                    )
            if args.text_encoder_lr2 is not None:
                if 1 in te_selection_indices:
                    te_lr_overrides[1] = args.text_encoder_lr2
                else:
                    logger.warning(
                        "ignore text_encoder_lr2 because Text Encoder 2 is not selected / Text Encoder 2を学習対象にしていないためtext_encoder_lr2は無視されます"
                    )
        elif args.text_encoder_lr1 is not None or args.text_encoder_lr2 is not None:
            logger.warning(
                "ignore text_encoder_lr1/text_encoder_lr2 because text encoder training is disabled / Text Encoderを学習しないためtext_encoder_lr1とtext_encoder_lr2は無視されます"
            )

        lr_descriptions = None
        try:
            results = network.prepare_optimizer_params(
                args.text_encoder_lr,
                args.unet_lr,
                args.learning_rate,
                text_encoder_lrs=te_lr_overrides,
                active_text_encoder_indices=te_selection_indices,
            )
        except TypeError:
            if te_lr_overrides:
                logger.warning(
                    "network module does not support per-text-encoder learning rates; falling back to shared lr / ネットワークモジュールがText Encoderごとの学習率に対応していないため、共通の学習率を使用します"
                )
            try:
                results = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
            except TypeError:
                results = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)

        if isinstance(results, tuple):
            trainable_params = results[0]
            lr_descriptions = results[1]
        else:
            trainable_params = results

        # if len(trainable_params) == 0:
        #     accelerator.print("no trainable parameters found / 学習可能なパラメータが見つかりませんでした")
        # for params in trainable_params:
        #     for k, v in params.items():
        #         if type(v) == float:
        #             pass
        #         else:
        #             v = len(v)
        #         accelerator.print(f"trainable_params: {k} = {v}")

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        if self._te_lr_after_cfg:
            if lr_descriptions is None:
                logger.warning(
                    "te_lr_after requires optimizer group descriptions; disabling option / te_lr_after を利用するには学習率グループの情報が必要なため無効化します"
                )
                self._te_lr_after_cfg = None
            else:
                te_group_indices: List[int] = []
                missing_targets: List[int] = []
                for te_idx in sorted(self._te_lr_after_cfg["target_indices"]):
                    matches = [
                        idx for idx, desc in enumerate(lr_descriptions) if self._te_group_matches_description(desc, te_idx)
                    ]
                    if not matches:
                        missing_targets.append(te_idx)
                    else:
                        te_group_indices.extend(matches)
                if missing_targets:
                    target_names = ", ".join(f"TE{idx + 1}" for idx in missing_targets)
                    logger.warning(
                        "te_lr_after: targets %s have no optimizer groups; they will be skipped / te_lr_after: 対象 %s に対応するパラメーターグループが見つからなかったためスキップします",
                        target_names,
                        target_names,
                    )
                te_group_indices = sorted(set(te_group_indices))
                if not te_group_indices:
                    logger.warning(
                        "te_lr_after: no applicable text encoder parameter groups detected; disabling option / te_lr_after: 対応するText Encoderパラメーターが見つからなかったため無効化します"
                    )
                    self._te_lr_after_cfg = None
                else:
                    self._te_lr_after_cfg["group_indices"] = te_group_indices
                    self._te_lr_after_cfg["group_labels"] = [lr_descriptions[i] for i in te_group_indices]

        if self._te_freeze_cfg:
            if lr_descriptions is None:
                logger.warning("te freeze options require optimizer group descriptions; disabling option")
                self._te_freeze_cfg = None
            else:
                active_freeze_cfg = {}
                for te_idx, cfg in sorted(self._te_freeze_cfg.items()):
                    matches = [
                        idx for idx, desc in enumerate(lr_descriptions) if self._te_group_matches_description(desc, te_idx)
                    ]
                    if not matches:
                        logger.warning("te freeze: TE%d has no optimizer groups; skipping", te_idx + 1)
                        continue
                    cfg["group_indices"] = sorted(set(matches))
                    cfg["group_labels"] = [lr_descriptions[i] for i in cfg["group_indices"]]
                    active_freeze_cfg[te_idx] = cfg
                self._te_freeze_cfg = active_freeze_cfg or None

        # dataloaderを準備する
        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        if self._te_lr_after_cfg:
            total_steps = args.max_train_steps
            if total_steps is None or total_steps <= 0:
                logger.warning(
                    "disable te_lr_after because max_train_steps is not a positive number / max_train_steps が正の値ではないため te_lr_after は無効化されます"
                )
                self._te_lr_after_cfg = None
            else:
                threshold = math.floor(total_steps * self._te_lr_after_cfg["ratio"])
                self._te_lr_after_cfg["threshold_step"] = threshold
                labels = self._te_lr_after_cfg.get("group_labels")
                if not labels:
                    labels = [f"TE{idx + 1}" for idx in sorted(self._te_lr_after_cfg.get("target_indices", []))]
                mult = self._te_lr_after_cfg["mult"]
                ratio = self._te_lr_after_cfg["ratio"]
                status = "applied" if self._te_lr_after_cfg.get("applied") else "pending"
                applied_step = self._te_lr_after_cfg.get("applied_step")
                status_detail = f"{status}"
                if applied_step is not None:
                    status_detail += f" (step={applied_step})"
                logger.info(
                    "te_lr_after ready (%s): scale %s lr by %.6f after step > %d (ratio=%.4f) / "
                    "te_lr_after: 状態=%s。ステップ%d超で %s の学習率に倍率%.6f (割合=%.4f) を適用します",
                    status_detail,
                    ", ".join(labels),
                    mult,
                    threshold,
                    ratio,
                    status_detail,
                    threshold,
                    ", ".join(labels),
                    mult,
                    ratio,
                )

        # データセット側にも学習ステップを送信
        if self._te_freeze_cfg:
            total_steps = args.max_train_steps
            if total_steps is None or total_steps <= 0:
                logger.warning("disable te freeze options because max_train_steps is not a positive number")
                self._te_freeze_cfg = None
            else:
                for te_idx, cfg in sorted(self._te_freeze_cfg.items()):
                    freeze_at = cfg["freeze_at"]
                    threshold = math.floor(total_steps * freeze_at) if freeze_at <= 1.0 else int(freeze_at)
                    threshold = max(0, threshold)
                    cfg["threshold_step"] = threshold
                    labels = cfg.get("group_labels") or [f"TE{te_idx + 1}"]
                    logger.info(
                        "te freeze ready: freeze %s at step >= %d (value=%.4f)",
                        ", ".join(labels),
                        threshold,
                        freeze_at,
                    )

        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr schedulerを用意する
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes, lr_descriptions=lr_descriptions)

        dq_delta_begin_step = None
        if dq_begin_after_lr_warmup:
            if isinstance(args.lr_warmup_steps, float):
                if args.max_train_steps is None or args.max_train_steps <= 0:
                    logger.error(
                        "dq_delta_begin_after_lr_warmup requires positive max_train_steps when lr_warmup_steps is float. / "
                        "dq_delta_begin_after_lr_warmup では lr_warmup_steps が float の場合、max_train_steps が正の値である必要があります。"
                    )
                    raise ValueError("dq_delta_begin_after_lr_warmup requires max_train_steps > 0 for float lr_warmup_steps")
                num_training_steps = args.max_train_steps * accelerator.num_processes
                dq_delta_begin_step = int(args.lr_warmup_steps * num_training_steps)
            else:
                dq_delta_begin_step = int(args.lr_warmup_steps)
            dq_delta_begin_step = max(0, dq_delta_begin_step)
            logger.info(
                "dq_delta_begin_after_lr_warmup enabled: begin_step=%d (lr_warmup_steps=%s)",
                dq_delta_begin_step,
                args.lr_warmup_steps,
            )

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training.")
            unet_weight_dtype = torch.float8_e4m3fn
            te_weight_dtype = torch.float8_e4m3fn

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)
                # nn.Embedding not support FP8
                t_enc.text_model.embeddings.to(dtype=(weight_dtype if te_weight_dtype != weight_dtype else te_weight_dtype))

        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if train_text_encoder else None,
                text_encoder2=text_encoders[1] if train_text_encoder and len(text_encoders) > 1 else None,
                network=network,
            )
            ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, lr_scheduler
            )
            training_model = ds_model
        else:
            if train_unet:
                unet = accelerator.prepare(unet)
            else:
                unet.to(accelerator.device, dtype=unet_weight_dtype)  # move to device because unet is not prepared by accelerator
            if train_text_encoder:
                if len(text_encoders) > 1:
                    text_encoder = text_encoders = [accelerator.prepare(t_enc) for t_enc in text_encoders]
                else:
                    text_encoder = accelerator.prepare(text_encoder)
                    text_encoders = [text_encoder]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
            training_model = network

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            train_state = {
                "current_epoch": current_epoch.value,
                "current_step": current_step.value + 1,
            }
            if self._te_lr_after_cfg:
                train_state["te_lr_after"] = {
                    "applied": bool(self._te_lr_after_cfg.get("applied", False)),
                    "applied_step": self._te_lr_after_cfg.get("applied_step"),
                    "threshold_step": self._te_lr_after_cfg.get("threshold_step"),
                }
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump(train_state, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                step_value = data.get("current_step")
                try:
                    steps_from_state_local = int(step_value) if step_value is not None else None
                except (TypeError, ValueError):
                    steps_from_state_local = None
                steps_from_state = steps_from_state_local
                self._te_lr_after_resumed = True
                self._te_lr_after_resume_state = data.get("te_lr_after")
                self._te_lr_after_resume_step = steps_from_state_local
                logger.info(f"load train state from {train_state_file}: {data}")
            elif getattr(args, "resume", False):
                self._te_lr_after_resumed = True

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)
        if self._te_lr_after_cfg:
            # load_model_hook で復元された情報を反映する（resume前には未取得）
            self._handle_te_lr_after_resume()

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        avg_cp_mode = getattr(args, "avg_cp_mode", "live")
        avg_promote_pick = getattr(args, "avg_promote_pick", "fixed")
        shadow_mode = bool(args.avg_cp and avg_cp_mode == "shadow")
        promote_mode = bool(args.avg_cp and avg_cp_mode == "promote")
        proxy_scoring_mode = shadow_mode or promote_mode
        if proxy_scoring_mode and getattr(args, "resume", False):
            raise ValueError(f"avg_cp_mode={avg_cp_mode} does not support resume yet / avg_cp_mode={avg_cp_mode} は resume 未対応です")
        if proxy_scoring_mode and accelerator.num_processes > 1:
            raise ValueError(
                f"avg_cp_mode={avg_cp_mode} does not support multi-GPU or distributed training yet / "
                f"avg_cp_mode={avg_cp_mode} は複数GPU・分散学習に未対応です"
            )
        cp_window = deque(maxlen=args.avg_window) if args.avg_cp else None
        cp_window_epochs = deque(maxlen=args.avg_window) if args.avg_cp else None
        if args.avg_cp and args.resume:
            ext = "." + args.save_model_as
            model_name = train_util.default_if_none(args.output_name, train_util.DEFAULT_EPOCH_NAME)
            for epoch_no, p in collect_last_checkpoints_with_epochs(args.output_dir, model_name, ext, args.avg_window):
                cp_window.append(load_lora_state_dict(p))
                cp_window_epochs.append(epoch_no)

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        if bool(getattr(args, "group_loss_log", False)):
            datasets_for_check = getattr(train_dataset_group, "datasets", None)
            if isinstance(datasets_for_check, (list, tuple)) and len(datasets_for_check) > 0:
                dataset_batch_sizes = [int(getattr(d, "batch_size", -1)) for d in datasets_for_check]
                if any(bs != 1 for bs in dataset_batch_sizes):
                    raise ValueError(
                        f"group_loss_log supports only batch_size=1 datasets. found: {dataset_batch_sizes} / "
                        f"group_loss_log は batch_size=1 のデータセットのみ対応です。検出値: {dataset_batch_sizes}"
                    )
            else:
                logger.info(
                    "group_loss_log: dataset-level batch_size check is skipped because dataset_class does not expose `.datasets`; "
                    "batch_size will be validated at runtime / "
                    "group_loss_log: dataset_class が `.datasets` を持たないため事前のbatch_size検証をスキップします。"
                    "batch_size は実行時に検証します"
                )

        if proxy_scoring_mode:
            datasets_for_check = getattr(train_dataset_group, "datasets", None)
            if isinstance(datasets_for_check, (list, tuple)) and len(datasets_for_check) > 0:
                dataset_batch_sizes = [int(getattr(d, "batch_size", -1)) for d in datasets_for_check]
                if any(bs != 1 for bs in dataset_batch_sizes):
                    raise ValueError(
                        f"avg_cp_mode={avg_cp_mode} supports only batch_size=1 datasets. found: {dataset_batch_sizes} / "
                        f"avg_cp_mode={avg_cp_mode} は batch_size=1 のデータセットのみ対応です。検出値: {dataset_batch_sizes}"
                    )
            else:
                logger.info(
                    f"avg_cp_mode={avg_cp_mode}: dataset-level batch_size check is skipped because dataset_class does not expose `.datasets`; "
                    "batch_size will be validated at runtime / "
                    f"avg_cp_mode={avg_cp_mode}: dataset_class が `.datasets` を持たないため事前のbatch_size検証をスキップします。"
                    "batch_size は実行時に検証します"
                )

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_te1_lr_warmup_steps": args.te1_lr_warmup_steps,
            "ss_te2_lr_warmup_steps": args.te2_lr_warmup_steps,
            "ss_te1_freeze_at": args.te1_freeze_at,
            "ss_te2_freeze_at": args.te2_freeze_at,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_noise_offset_random_min_ratio": args.noise_offset_random_min_ratio,
            "ss_noise_offset_random_max_ratio": args.noise_offset_random_max_ratio,
            "ss_noise_offset_random_min_ratio_sched": args.noise_offset_random_min_ratio_sched,
            "ss_noise_offset_random_max_ratio_sched": args.noise_offset_random_max_ratio_sched,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_c": args.huber_c,
        }

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "subset_index": getattr(subset, "subset_index", -1),
                        "num_repeats": subset.num_repeats,
                        "group": getattr(subset, "group", None),
                        "group_adjust": bool(getattr(subset, "group_adjust", True)),
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # calculate steps to skip when resuming or starting from a specific step
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

        progress_bar = tqdm(
            range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )
        progress_bar_started = False

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
                initial_step *= args.gradient_accumulation_steps

                # set epoch to start to make initial_step less than len(train_dataloader)
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = 0
        skipped_steps = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        loss_recorder = train_util.LossRecorder()
        del train_dataset_group

        # prepare gradient skipping if enabled (複数 GPUではrankごとに判定がズレる恐れありらしい)
        (
            grad_norm_mode,
            skip_grad_norm,
            log_grad_norm,
            log_grad_cosine,
            skip_grad_norm_max,
            nan_to_window,
            inf_to_window,
            skip_nan_immediate,
            skip_inf_immediate,
        ) = resolve_grad_norm_settings(args)
        scaler_for_log = accelerator.scaler if hasattr(accelerator, "scaler") else None
        log_grad_scale = log_grad_norm and scaler_for_log is not None
        logger.info(
            f"grad_norm_mode: {grad_norm_mode}, skip_grad_norm: {skip_grad_norm}, grad_norm_log: {log_grad_norm}, "
            f"skip_grad_norm_max: {skip_grad_norm_max}, nan_to_window: {nan_to_window}, "
            f"inf_to_window: {inf_to_window}, skip_nan_immediate: {skip_nan_immediate}, "
            f"skip_inf_immediate: {skip_inf_immediate}"
        )
        model_name_for_logs = train_util.default_if_none(args.output_name, train_util.DEFAULT_LAST_OUTPUT_NAME)
        use_grad_norm = skip_grad_norm or log_grad_norm
        grad_norm_guardian: Optional[GradNormGuardian] = None
        if use_grad_norm:
            os.makedirs(args.output_dir, exist_ok=True)
            log_file_path = os.path.join(args.output_dir, f"gradient_logs+{model_name_for_logs}.txt")
            guardian_config = GradNormGuardianConfig(
                skip_grad_norm=skip_grad_norm,
                log_grad_norm=log_grad_norm,
                log_grad_scale=log_grad_scale,
                log_grad_cosine=log_grad_cosine,
                skip_grad_norm_max=skip_grad_norm_max,
                nan_to_window=nan_to_window,
                inf_to_window=inf_to_window,
                skip_nan_immediate=skip_nan_immediate,
                skip_inf_immediate=skip_inf_immediate,
            )
            grad_norm_guardian = GradNormGuardian(
                config=guardian_config,
                scaler_for_log=scaler_for_log if log_grad_scale else None,
                log_file_path=log_file_path if log_grad_norm else None,
            )

            def check_gradients_and_skip_update(model, epoch, step, loss_val):
                return grad_norm_guardian.observe(model, epoch, step, loss_val)
        else:
            def check_gradients_and_skip_update(model, epoch, step, loss_val):
                return False

        group_loss_log_enabled = bool(getattr(args, "group_loss_log", False))
        group_loss_tracker: Optional[GroupLossTracker] = None
        group_loss_log_every_n_steps = 100
        group_loss_epoch_summary = bool(getattr(args, "group_loss_epoch_summary", False))
        group_loss_step_log_path = None
        group_loss_epoch_log_path = None
        group_loss_step_header_written = False
        group_loss_epoch_header_written = False
        group_loss_step_header = "global_step,epoch,group,subset_index,loss,ema_loss_group,count_group,timestep,bucket_reso"
        group_loss_step_log_buffer: List[str] = []
        group_loss_last_flush_step = 0
        if proxy_scoring_mode and args.avg_shadow_bank_size < 1:
            raise ValueError("avg_shadow_bank_size must be >= 1 / avg_shadow_bank_size は 1 以上で指定してください")
        shadow_log_jsonl = proxy_scoring_mode and bool(getattr(args, "avg_shadow_log_jsonl", True))
        shadow_log_prefix = "avg_shadow" if shadow_mode else "avg_promote"
        shadow_log_path = (
            os.path.join(args.output_dir, f"{shadow_log_prefix}+{model_name_for_logs}.jsonl") if shadow_log_jsonl else None
        )
        shadow_candidate_pool: List[Dict[str, Any]] = []
        shadow_candidate_pool_target_size = max(args.avg_shadow_bank_size * 4, 32) if proxy_scoring_mode else 0
        shadow_candidate_pool_final_size: Optional[int] = None
        shadow_bank: List[Dict[str, Any]] = []
        shadow_virtual_win_streak = 0
        final_avg_raw_sd: Optional[Dict[str, torch.Tensor]] = None
        final_avg_center_sd: Optional[Dict[str, torch.Tensor]] = None

        def _group_csv_escape(value):
            if value is None:
                return ""
            text = str(value)
            if "," in text or '"' in text or "\n" in text:
                text = '"' + text.replace('"', '""') + '"'
            return text

        def _group_csv_format(value):
            if value is None:
                return ""
            if isinstance(value, torch.Tensor):
                value = value.item()
            if isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return ""
                return f"{value:.10g}"
            return _group_csv_escape(value)

        def _group_write_csv(path: Optional[str], header: str, row: List, epoch_log: bool):
            nonlocal group_loss_step_header_written, group_loss_epoch_header_written
            if path is None:
                return
            dirpath = os.path.dirname(path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)

            header_written = group_loss_epoch_header_written if epoch_log else group_loss_step_header_written
            if not header_written:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(header + "\n")
                if epoch_log:
                    group_loss_epoch_header_written = True
                else:
                    group_loss_step_header_written = True

            with open(path, "a", encoding="utf-8") as f:
                f.write(",".join(_group_csv_format(v) for v in row) + "\n")

        def _group_flush_step_buffer(force: bool = False, current_global_step: Optional[int] = None):
            nonlocal group_loss_step_header_written, group_loss_last_flush_step
            if not group_loss_log_enabled or not accelerator.is_main_process:
                return
            if group_loss_step_log_path is None:
                return
            if not force:
                if current_global_step is None:
                    return
                if current_global_step - group_loss_last_flush_step < group_loss_log_every_n_steps:
                    return
                group_loss_last_flush_step = current_global_step
            elif current_global_step is not None:
                group_loss_last_flush_step = current_global_step

            if len(group_loss_step_log_buffer) == 0:
                return

            dirpath = os.path.dirname(group_loss_step_log_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            if not group_loss_step_header_written:
                with open(group_loss_step_log_path, "w", encoding="utf-8") as f:
                    f.write(group_loss_step_header + "\n")
                group_loss_step_header_written = True

            with open(group_loss_step_log_path, "a", encoding="utf-8") as f:
                f.writelines(line + "\n" for line in group_loss_step_log_buffer)
            group_loss_step_log_buffer.clear()

        def _extract_first(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                if value.numel() < 1:
                    return None
                return value.reshape(-1)[0].item()
            if isinstance(value, (list, tuple)):
                if len(value) < 1:
                    return None
                return value[0]
            return value

        def _infer_runtime_batch_size(batch):
            preferred_keys = (
                "loss_weights",
                "latents",
                "images",
                "input_ids",
                "input_ids2",
                "network_multipliers",
                "captions",
                "alpha_masks",
                "original_sizes_hw",
                "crop_top_lefts",
                "target_sizes_hw",
                "groups",
                "subset_indices",
                "bucket_resos",
            )

            def _candidate_size(value):
                if isinstance(value, torch.Tensor):
                    if value.ndim == 0:
                        return None
                    return int(value.shape[0])
                if isinstance(value, list):
                    return len(value)
                return None

            for key in preferred_keys:
                if key not in batch:
                    continue
                size = _candidate_size(batch.get(key))
                if size is not None:
                    return size, key

            for key, value in batch.items():
                size = _candidate_size(value)
                if size is not None:
                    return size, key

            return None, None

        def _shadow_to_cpu(value, compact_float: bool = False):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                tensor = value.detach().cpu().clone()
                if compact_float and torch.is_floating_point(tensor):
                    tensor = tensor.to(torch.float16)
                return tensor
            if isinstance(value, list):
                return list(value)
            if isinstance(value, tuple):
                return list(value)
            return value

        def _shadow_normalize_labels(values):
            if values is None:
                return []
            if not isinstance(values, (list, tuple)):
                values = [values]
            normalized = []
            for value in values:
                if value is None:
                    continue
                text = str(value).strip()
                if text == "":
                    continue
                normalized.append(text)
            return normalized

        def _shadow_primary_label(values):
            labels = _shadow_normalize_labels(values)
            if len(labels) < 1:
                return None
            unique = list(dict.fromkeys(labels))
            if len(unique) == 1:
                return unique[0]
            return "||".join(unique)

        def _collect_shadow_bank_item(batch, noisy_latents, target, timesteps, huber_c):
            item: Dict[str, Any] = {
                "noisy_latents": _shadow_to_cpu(noisy_latents, compact_float=True),
                "target": _shadow_to_cpu(target, compact_float=True),
                "timesteps": _shadow_to_cpu(timesteps),
                "huber_c": _shadow_to_cpu(huber_c),
                "loss_weights": _shadow_to_cpu(batch["loss_weights"]),
            }
            tensor_keys = (
                "input_ids",
                "input_ids2",
                "network_multipliers",
                "alpha_masks",
                "original_sizes_hw",
                "crop_top_lefts",
                "target_sizes_hw",
                "text_encoder_outputs1_list",
                "text_encoder_outputs2_list",
                "text_encoder_pool2_list",
            )
            compact_float_keys = {"alpha_masks", "text_encoder_outputs1_list", "text_encoder_outputs2_list", "text_encoder_pool2_list"}
            for key in tensor_keys:
                if key in batch and batch[key] is not None:
                    item[key] = _shadow_to_cpu(batch[key], compact_float=(key in compact_float_keys))
            if "captions" in batch and batch["captions"] is not None:
                item["captions"] = list(batch["captions"])
            if "image_keys" in batch and batch["image_keys"] is not None:
                item["image_keys"] = list(batch["image_keys"])
            if "class_tokens" in batch and batch["class_tokens"] is not None:
                item["class_tokens"] = list(batch["class_tokens"])
            item["shadow_primary_class_token"] = _shadow_primary_label(item.get("class_tokens"))
            item["shadow_primary_image_key"] = _shadow_primary_label(item.get("image_keys"))
            return item

        def _materialize_shadow_batch(item):
            runtime_batch: Dict[str, Any] = {}
            for key, value in item.items():
                if key in (
                    "noisy_latents",
                    "target",
                    "timesteps",
                    "huber_c",
                    "shadow_primary_class_token",
                    "shadow_primary_image_key",
                ):
                    continue
                if isinstance(value, torch.Tensor):
                    runtime_batch[key] = value.to(accelerator.device)
                elif isinstance(value, list):
                    runtime_batch[key] = list(value)
                else:
                    runtime_batch[key] = value

            noisy_latents = item["noisy_latents"].to(accelerator.device, dtype=weight_dtype)
            target = item["target"].to(accelerator.device, dtype=weight_dtype)
            timesteps = item["timesteps"].to(accelerator.device)
            huber_c_value = item.get("huber_c")
            huber_c_runtime = huber_c_value.to(accelerator.device, dtype=weight_dtype) if isinstance(huber_c_value, torch.Tensor) else huber_c_value
            return runtime_batch, noisy_latents, target, timesteps, huber_c_runtime

        def _write_shadow_log(payload: Dict[str, Any]):
            if not shadow_log_jsonl or not accelerator.is_main_process or shadow_log_path is None:
                return
            dirpath = os.path.dirname(shadow_log_path)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(shadow_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def _shadow_select_final_bank(candidate_pool: List[Dict[str, Any]]):
            if len(candidate_pool) < args.avg_shadow_bank_size:
                return []

            grouped_candidates: Dict[str, List[Dict[str, Any]]] = {}
            group_order: List[str] = []
            for item in candidate_pool:
                class_key = item.get("shadow_primary_class_token") or "__no_class_tokens__"
                if class_key not in grouped_candidates:
                    grouped_candidates[class_key] = []
                    group_order.append(class_key)
                grouped_candidates[class_key].append(item)

            selected: List[Dict[str, Any]] = []
            used_item_ids = set()
            used_image_keys = set()

            while len(selected) < args.avg_shadow_bank_size:
                made_progress = False
                for class_key in group_order:
                    candidates = grouped_candidates[class_key]
                    selected_item = None
                    fallback_item = None
                    for item in candidates:
                        item_id = id(item)
                        if item_id in used_item_ids:
                            continue
                        if fallback_item is None:
                            fallback_item = item
                        image_key = item.get("shadow_primary_image_key")
                        if image_key is None or image_key not in used_image_keys:
                            selected_item = item
                            break
                    if selected_item is None:
                        selected_item = fallback_item
                    if selected_item is None:
                        continue

                    item_id = id(selected_item)
                    used_item_ids.add(item_id)
                    image_key = selected_item.get("shadow_primary_image_key")
                    if image_key is not None:
                        used_image_keys.add(image_key)
                    selected.append(selected_item)
                    made_progress = True

                    if len(selected) >= args.avg_shadow_bank_size:
                        break

                if not made_progress:
                    break

            if len(selected) < args.avg_shadow_bank_size:
                for item in candidate_pool:
                    item_id = id(item)
                    if item_id in used_item_ids:
                        continue
                    selected.append(item)
                    used_item_ids.add(item_id)
                    if len(selected) >= args.avg_shadow_bank_size:
                        break

            return selected

        def _shadow_timestep_bins(items: List[Dict[str, Any]]):
            total_timesteps = int(noise_scheduler.config.num_train_timesteps)
            low_cut = total_timesteps // 3
            high_cut = (total_timesteps * 2) // 3
            bins = {"low": 0, "mid": 0, "high": 0}
            for item in items:
                timesteps_tensor = item.get("timesteps")
                if not isinstance(timesteps_tensor, torch.Tensor):
                    continue
                for timestep in timesteps_tensor.reshape(-1).tolist():
                    timestep_value = int(timestep)
                    if timestep_value < low_cut:
                        bins["low"] += 1
                    elif timestep_value < high_cut:
                        bins["mid"] += 1
                    else:
                        bins["high"] += 1
            return bins

        def _shadow_bank_metadata(items: List[Dict[str, Any]]):
            class_counter: Counter = Counter()
            image_counter: Counter = Counter()
            for item in items:
                class_counter.update(_shadow_normalize_labels(item.get("class_tokens")))
                image_counter.update(_shadow_normalize_labels(item.get("image_keys")))

            return {
                "bank_unique_class_tokens": len(class_counter),
                "bank_max_per_class_tokens": max(class_counter.values()) if len(class_counter) > 0 else 0,
                "bank_unique_image_keys": len(image_counter),
                "bank_max_per_image_key": max(image_counter.values()) if len(image_counter) > 0 else 0,
                "bank_timestep_bins": _shadow_timestep_bins(items),
            }

        def _shadow_candidate_pool_size_for_log():
            if shadow_candidate_pool_final_size is not None:
                return shadow_candidate_pool_final_size
            return len(shadow_candidate_pool)

        def _finalize_shadow_bank_if_ready(is_last_epoch: bool):
            nonlocal shadow_candidate_pool_final_size
            if len(shadow_bank) > 0:
                return True
            if len(shadow_candidate_pool) < args.avg_shadow_bank_size:
                return False
            if len(shadow_candidate_pool) < shadow_candidate_pool_target_size and not is_last_epoch:
                return False

            shadow_bank.extend(_shadow_select_final_bank(shadow_candidate_pool))
            if len(shadow_bank) >= args.avg_shadow_bank_size:
                shadow_candidate_pool_final_size = len(shadow_candidate_pool)
                shadow_candidate_pool.clear()
            return len(shadow_bank) >= args.avg_shadow_bank_size

        def _reset_optimizer_avg_stats():
            for p_state in optimizer.state.values():
                p_state["step"] = p_state.get("step", 0)
                for buf in ("exp_avg", "exp_avg_sq", "exp_avg_max"):
                    if buf in p_state and isinstance(p_state[buf], torch.Tensor):
                        p_state[buf].zero_()

        def _save_avg_candidate(kind: str, state_dict: Dict[str, torch.Tensor], epoch_no: int, steps: int):
            if not accelerator.is_main_process:
                return
            os.makedirs(args.output_dir, exist_ok=True)
            save_sd = {}
            for key, value in state_dict.items():
                tensor = value.detach().cpu()
                if save_dtype is not None:
                    tensor = tensor.to(save_dtype)
                save_sd[key] = tensor
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)
            metadata_to_save = dict(minimum_metadata if args.no_metadata else metadata)
            metadata_to_save.update(train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False))
            metadata_to_save["ss_avg_shadow_variant"] = kind
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(save_sd, metadata_to_save)
            metadata_to_save["sshs_model_hash"] = model_hash
            metadata_to_save["sshs_legacy_hash"] = legacy_hash
            model_name = train_util.default_if_none(args.output_name, train_util.DEFAULT_LAST_OUTPUT_NAME)
            save_lora_state_dict(
                os.path.join(args.output_dir, f"{model_name}_{kind}.safetensors"), save_sd, dtype=None, metadata=metadata_to_save
            )

        def _capture_torch_rng_state():
            state = {"cpu": torch.get_rng_state()}
            if accelerator.device.type == "cuda" and torch.cuda.is_available():
                state["cuda"] = torch.cuda.get_rng_state(accelerator.device)
            return state

        def _restore_torch_rng_state(state):
            torch.set_rng_state(state["cpu"])
            if "cuda" in state:
                torch.cuda.set_rng_state(state["cuda"], accelerator.device)

        def _score_shadow_bank(unwrapped_network):
            if len(shadow_bank) < args.avg_shadow_bank_size:
                return None

            total_loss = 0.0
            total_weight = 0
            with torch.no_grad():
                for item in shadow_bank:
                    runtime_batch, noisy_latents_rt, target_rt, timesteps_rt, huber_c_rt = _materialize_shadow_batch(item)
                    self._set_network_multiplier_from_batch(unwrapped_network, runtime_batch)
                    text_encoder_conds = self._get_text_conds_for_batch(
                        args, accelerator, runtime_batch, tokenizers, text_encoders, weight_dtype, grad_enabled=False
                    )
                    loss = self._compute_batch_loss(
                        args,
                        accelerator,
                        runtime_batch,
                        noise_scheduler,
                        unet,
                        text_encoder_conds,
                        noisy_latents_rt,
                        timesteps_rt,
                        target_rt,
                        huber_c_rt,
                        weight_dtype,
                        train_unet=False,
                    )
                    batch_size_for_score = int(runtime_batch["loss_weights"].shape[0])
                    total_loss += loss.detach().item() * batch_size_for_score
                    total_weight += batch_size_for_score
                    del runtime_batch, noisy_latents_rt, target_rt, timesteps_rt, huber_c_rt, text_encoder_conds, loss

            clean_memory_on_device(accelerator.device)
            if total_weight < 1:
                return None
            return total_loss / total_weight

        if group_loss_log_enabled:
            group_loss_ema_beta = float(getattr(args, "group_loss_ema_beta", 0.98))
            if not (0.0 <= group_loss_ema_beta < 1.0):
                raise ValueError(
                    "group_loss_ema_beta must be in [0,1) / group_loss_ema_beta は [0,1) の範囲で指定してください"
                )
            group_loss_log_every_n_steps = max(1, int(getattr(args, "group_loss_log_every_n_steps", 100)))
            group_loss_tracker = GroupLossTracker(group_loss_ema_beta)
            group_loss_step_log_path = os.path.join(args.output_dir, f"group_loss_logs+{model_name_for_logs}.csv")
            if group_loss_epoch_summary:
                group_loss_epoch_log_path = os.path.join(args.output_dir, f"group_loss_epoch+{model_name_for_logs}.csv")

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first
        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                initial_step -= len(train_dataloader)
            global_step = initial_step

        dq_auto_ema_state = None
        dq_quant_err_rms_ema_state = None
        dq_quant_err_ratio_ema_state = None
        dq_low_auto_quant_err_ratio_ema_state = None
        dq_low_auto_start_step = None
        dq_low_auto_bad_streak = 0
        dq_low_auto_escaped = False
        dq_low_auto_state = "observe" if dq_low_auto_enabled else ""
        dq_low_auto_decision = ""
        dq_low_auto_reason = ""
        dq_low_auto_bad = ""
        dq_low_auto_qerr_per_clip = None
        dq_low_auto_phase = "pre_min_progress" if dq_low_auto_enabled else ""
        dq_low_auto_can_escape = 0 if dq_low_auto_enabled else ""
        dq_bits_changed_since_auto = False
        dq_auto_warmup_reset_updates = dq_auto_warmup_updates
        dq_auto_warmup_remaining = dq_auto_warmup_reset_updates
        dq_auto_warmup_inband_streak = 0
        if dq_auto_enabled and dq_auto_log_path and accelerator.is_main_process:
            include_near_zero = "near_zero_rate" in dq_log_extra
            header = _dq_auto_log_header(dq_auto_log_format == "full_schema", include_near_zero)
            cols = header.split(",")
            row = ["" for _ in cols]
            col_idx = {name: idx for idx, name in enumerate(cols)}
            if "TrainStep" in col_idx:
                row[col_idx["TrainStep"]] = 0
            if "AutoInitMulApplied" in col_idx:
                row[col_idx["AutoInitMulApplied"]] = dq_auto_init_applied
            if "AutoInitMulValue" in col_idx:
                row[col_idx["AutoInitMulValue"]] = dq_auto_init_value if dq_auto_init_value is not None else ""
            if "AutoInitClipTarget" in col_idx:
                row[col_idx["AutoInitClipTarget"]] = (
                    dq_auto_init_clip_target if dq_auto_init_clip_target is not None else ""
                )
            if "QErrPerClipClipFloor" in col_idx:
                row[col_idx["QErrPerClipClipFloor"]] = dq_qerr_per_clip_floor
            if "ActiveClipBand" in col_idx:
                row[col_idx["ActiveClipBand"]] = dq_auto_active_band
            if "ActiveClipLow" in col_idx:
                row[col_idx["ActiveClipLow"]] = dq_auto_active_clip_low
            if "ActiveClipHigh" in col_idx:
                row[col_idx["ActiveClipHigh"]] = dq_auto_active_clip_high
            if "TrainProgress" in col_idx:
                row[col_idx["TrainProgress"]] = 0
            if "ClipRateLowAutoMinProgress" in col_idx:
                row[col_idx["ClipRateLowAutoMinProgress"]] = dq_low_auto_min_progress
            if "ClipRateLowAutoFreezeProgress" in col_idx:
                row[col_idx["ClipRateLowAutoFreezeProgress"]] = dq_low_auto_freeze_progress
            if "ClipRateLowAutoThresholdQErrRatio" in col_idx:
                row[col_idx["ClipRateLowAutoThresholdQErrRatio"]] = dq_low_auto_qerr_ratio_threshold
            if "ClipRateLowAutoThresholdQErrPerClip" in col_idx:
                row[col_idx["ClipRateLowAutoThresholdQErrPerClip"]] = dq_low_auto_qerr_per_clip_threshold
            if "ClipRateLowAutoPhase" in col_idx:
                row[col_idx["ClipRateLowAutoPhase"]] = "pre_min_progress" if dq_low_auto_enabled else ""
            if "ClipRateLowAutoCanEscape" in col_idx:
                row[col_idx["ClipRateLowAutoCanEscape"]] = 0 if dq_low_auto_enabled else ""
            _write_csv(dq_auto_log_path, header, ",".join(_dq_format_value(v) for v in row))

        def _dq_bits_for_progress(progress_frac: float, default_bits: Optional[int]):
            if not dq_bits_sched:
                return default_bits
            cur_bits = default_bits
            for p, b in dq_bits_sched:
                if progress_frac >= p:
                    cur_bits = b
                else:
                    break
            return cur_bits

        # initialize last_applied_bits from args (avoid per-epoch reset)
        last_applied_bits = getattr(args, "dq_delta_bits", None)
        dq_bits_force_apply = bool(dq_bits_sched and last_applied_bits is None)

        def _dq_delta_quant_enabled(progress_frac: float, global_step: int) -> bool:
            if dq_delta_begin_step is not None:
                return global_step >= dq_delta_begin_step
            return progress_frac >= args.dq_delta_begin

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step
                if initial_step > 0:
                    initial_step -= 1
                    continue
                if not progress_bar_started:
                    elapsed = time.time() - training_started_at
                    if accelerator.is_main_process:
                        logger.info(
                            f"startup time before first training step: {elapsed:.2f} sec"
                            f" / 学習開始前の初期化時間: {elapsed:.2f} 秒"
                        )
                    # Reset timer to exclude init/data loading overhead from it/s.
                    progress_bar.start_t = time.time()
                    progress_bar.last_print_t = progress_bar.start_t
                    progress_bar_started = True
                skip_step_flag = False
                if proxy_scoring_mode:
                    step_batch_size, batch_size_source = _infer_runtime_batch_size(batch)
                    if step_batch_size is None:
                        raise ValueError(
                            f"avg_cp_mode={avg_cp_mode} could not infer batch_size from the runtime batch; "
                            f"disable avg_cp_mode={avg_cp_mode} or make the batch expose a batched tensor/list field / "
                            f"avg_cp_mode={avg_cp_mode} は実行時バッチから batch_size を推定できませんでした。"
                            f"avg_cp_mode={avg_cp_mode} を無効化するか、バッチにバッチ軸を持つ tensor/list フィールドを含めてください"
                        )
                    if step_batch_size != 1:
                        raise ValueError(
                            f"avg_cp_mode={avg_cp_mode} supports only batch_size=1. got batch size {step_batch_size} from `{batch_size_source}` at step {step} / "
                            f"avg_cp_mode={avg_cp_mode} は batch_size=1 のみ対応です。step {step} で `{batch_size_source}` から batch size={step_batch_size} を検出しました"
                        )
                if group_loss_log_enabled:
                    step_batch_size, batch_size_source = _infer_runtime_batch_size(batch)
                    if step_batch_size is None:
                        raise ValueError(
                            "group_loss_log could not infer batch_size from the runtime batch; "
                            "disable group_loss_log or make the batch expose a batched tensor/list field / "
                            "group_loss_log は実行時バッチから batch_size を推定できませんでした。"
                            "group_loss_log を無効化するか、バッチにバッチ軸を持つ tensor/list フィールドを含めてください"
                        )
                    if step_batch_size != 1:
                        raise ValueError(
                            f"group_loss_log supports only batch_size=1. got batch size {step_batch_size} from `{batch_size_source}` at step {step} / "
                            f"group_loss_log は batch_size=1 のみ対応です。step {step} で `{batch_size_source}` から batch size={step_batch_size} を検出しました"
                        )
                step_group = train_util.BaseDataset.normalize_group_name(_extract_first(batch.get("groups")))
                step_subset_index = _extract_first(batch.get("subset_indices"))
                step_bucket_reso = _extract_first(batch.get("bucket_resos"))
                step_timestep = None
                with accelerator.accumulate(training_model):
                    dq_bits_changed_this_step = False
                    # Toggle delta fake-quantization based on training progress
                    if hasattr(accelerator.unwrap_model(network), "set_delta_quant_enabled"):
                        dq_configured = (
                            (getattr(args, "dq_delta_step", None) is not None and args.dq_delta_step)
                            or (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits)
                            or bool(dq_bits_sched)
                        )
                        quant_enabled = False
                        progress_frac = 1.0
                        if dq_configured:
                            progress_frac = (global_step / float(args.max_train_steps)) if args.max_train_steps > 0 else 1.0
                            quant_enabled = _dq_delta_quant_enabled(progress_frac, global_step)
                            accelerator.unwrap_model(network).set_delta_quant_enabled(quant_enabled)

                            # Apply bits scheduling if specified
                            if dq_bits_sched:
                                cur_bits = last_applied_bits
                                for p, b in dq_bits_sched:
                                    if progress_frac >= p:
                                        cur_bits = b
                                    else:
                                        break
                                if dq_bits_force_apply or (cur_bits != last_applied_bits):
                                    _set_delta_fake_quant_compat(
                                        accelerator.unwrap_model(network),
                                        getattr(args, "dq_delta_step", None),
                                        args.dq_delta_mode,
                                        granularity=args.dq_delta_granularity,
                                        stat=args.dq_delta_stat,
                                        bits=cur_bits,
                                        range_mul=getattr(args, "dq_delta_range_mul", None),
                                        on_z=getattr(args, "dq_quantize_z", False),
                                        use_triton=getattr(args, "dq_delta_use_triton", False),
                                        triton_stats=getattr(args, "dq_delta_triton_stats", False),
                                    )
                                    last_applied_bits = cur_bits
                                    dq_bits_force_apply = False
                                    dq_bits_changed_this_step = True
                                    dq_bits_changed_since_auto = True

                        # dq_delta stats collection control (LogStep / AutoStep)
                        if hasattr(accelerator.unwrap_model(network), "set_dq_stats_state"):
                            step_idx = global_step + 1
                            do_log = dq_log_enabled and quant_enabled and (step_idx % dq_log_every == 0)
                            auto_eligible = dq_auto_enabled and quant_enabled and (
                                (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits) or bool(dq_bits_sched)
                            ) and (args.dq_delta_stat == "rms")
                            do_auto = auto_eligible and (step_idx % dq_auto_every == 0)
                            dq_log_full_detail = dq_log_detail == "full" or dq_log_mode == "per_module"
                            collect_full = bool(do_log or (do_auto and dq_low_auto_enabled))
                            collect_zero = bool(do_log and dq_log_full_detail)
                            collect_near_zero = bool(do_log and dq_log_full_detail and ("near_zero_rate" in dq_log_extra))
                            collect_detail = bool(do_log and dq_log_full_detail)
                            collect_error_parts = False
                            target = "z" if getattr(args, "dq_quantize_z", False) else "delta"

                            dq_stats_kwargs = dict(
                                step_idx=step_idx,
                                device=accelerator.device,
                                do_log=do_log,
                                do_auto=do_auto,
                                collect_full=collect_full,
                                collect_zero=collect_zero,
                                collect_near_zero=collect_near_zero,
                                collect_detail=collect_detail,
                                log_mode=dq_log_mode,
                                log_scope=dq_log_scope,
                                auto_scope=getattr(args, "dq_delta_scope", "both"),
                                target=target,
                            )
                            dq_stats_state_fn = accelerator.unwrap_model(network).set_dq_stats_state
                            dq_stats_params = inspect.signature(dq_stats_state_fn).parameters
                            if "collect_detail" not in dq_stats_params:
                                dq_stats_kwargs.pop("collect_detail", None)
                            supports_error_parts = "collect_error_parts" in dq_stats_params
                            if supports_error_parts:
                                dq_stats_state_fn(
                                    **dq_stats_kwargs,
                                    collect_error_parts=collect_error_parts,
                                )
                            else:
                                dq_stats_state_fn(**dq_stats_kwargs)

                    self._apply_te_freeze_if_ready(optimizer, accelerator.unwrap_model(network), global_step)

                    on_step_start(text_encoder, unet)

                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                            with torch.no_grad():
                                # latentに変換
                                latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype)
                        else:
                            chunks = [batch["images"][i:i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)]
                            list_latents = []
                            for chunk in chunks:
                                with torch.no_grad():
                                # latentに変換
                                    list_latents.append(vae.encode(chunk.to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype))
                            latents = torch.cat(list_latents, dim=0)
                            # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                    latents = latents * self.vae_scale_factor

                    self._set_network_multiplier_from_batch(accelerator.unwrap_model(network), batch)

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    progress_frac_for_noise = (global_step / float(args.max_train_steps)) if args.max_train_steps > 0 else 1.0
                    progress_frac_for_noise = max(0.0, min(1.0, progress_frac_for_noise))
                    noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents, progress_frac=progress_frac_for_noise
                    )
                    step_timestep = _extract_first(timesteps)

                    with torch.set_grad_enabled(train_text_encoder):
                        text_encoder_conds = self._get_text_conds_for_batch(
                            args, accelerator, batch, tokenizers, text_encoders, weight_dtype, grad_enabled=train_text_encoder
                        )

                    # ensure the hidden state will require grad
                    if args.gradient_checkpointing:
                        for x in noisy_latents:
                            x.requires_grad_(True)
                        for t in text_encoder_conds:
                            t.requires_grad_(True)

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise
                    if (
                        proxy_scoring_mode
                        and len(shadow_bank) < args.avg_shadow_bank_size
                        and len(shadow_candidate_pool) < shadow_candidate_pool_target_size
                        and (epoch + 1) / num_train_epochs >= args.avg_begin
                    ):
                        shadow_candidate_pool.append(_collect_shadow_bank_item(batch, noisy_latents, target, timesteps, huber_c))

                    loss = self._compute_batch_loss(
                        args,
                        accelerator,
                        batch,
                        noise_scheduler,
                        unet,
                        text_encoder_conds,
                        noisy_latents,
                        timesteps,
                        target,
                        huber_c,
                        weight_dtype,
                        train_unet=train_unet,
                    )

                    accelerator.backward(loss)
                    loss_scalar = loss.detach().item()
                    skip_step = False
                    if check_gradients_and_skip_update(network, epoch, step, loss_scalar):
                        accelerator.print(
                            f"\nSkipping update at Epoch: {epoch}, Step: {step} due to large gradients."
                        )
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        skip_step = True
                        skip_step_flag = True

                    if not skip_step:
                        if accelerator.sync_gradients:
                            self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                            if args.max_grad_norm != 0.0:
                                params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        self._apply_te_lr_after_if_ready(optimizer, lr_scheduler, global_step + 1)
                        optimizer.zero_grad(set_to_none=True)

                        # Optional: quantize/round LoRA trainable parameters after each optimizer step
                        if (
                            args.round_lora_step is not None
                            and args.round_lora_step > 0
                            and accelerator.sync_gradients
                        ):
                            # step index after this update
                            next_step_idx = global_step + 1
                            # respect warmup for rounding based on overall training progress
                            progress_frac = next_step_idx / float(args.max_train_steps)
                            if progress_frac >= args.round_lora_begin and (next_step_idx % max(1, args.round_lora_every) == 0):
                                round_parameters(
                                    accelerator.unwrap_model(network).get_trainable_params(),
                                    step=args.round_lora_step,
                                    mode=args.round_lora_mode,
                                    exclude_param_ids=self._te_frozen_param_ids,
                                )
                current_loss = loss_scalar

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = self._apply_max_norm_regularization(
                        accelerator.unwrap_model(network), args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if (
                        group_loss_log_enabled
                        and group_loss_tracker is not None
                        and (not skip_step_flag)
                        and math.isfinite(current_loss)
                    ):
                        ema_loss_group, count_group = group_loss_tracker.update(step_group, current_loss)
                        if accelerator.is_main_process:
                            bucket_reso_value = ""
                            if isinstance(step_bucket_reso, (list, tuple)) and len(step_bucket_reso) == 2:
                                bucket_reso_value = f"{step_bucket_reso[0]}x{step_bucket_reso[1]}"
                            row = [
                                global_step,
                                epoch + 1,
                                step_group,
                                step_subset_index,
                                current_loss,
                                ema_loss_group,
                                count_group,
                                step_timestep,
                                bucket_reso_value,
                            ]
                            group_loss_step_log_buffer.append(",".join(_group_csv_format(v) for v in row))
                    _group_flush_step_buffer(force=False, current_global_step=global_step)
                    if hasattr(accelerator.unwrap_model(network), "export_dq_stats"):
                        step_idx = global_step
                        if skip_step_flag:
                            accelerator.unwrap_model(network).discard_dq_stats_step(step_idx)
                        else:
                            dq_stats = accelerator.unwrap_model(network).export_dq_stats()
                            if dq_stats is not None and dq_stats.get("step_idx") == step_idx:
                                accum_by_scope = dq_stats["accum"]
                                collect_full = dq_stats["collect_full"]
                                collect_zero = dq_stats["collect_zero"]
                                collect_near_zero = dq_stats["collect_near_zero"]
                                collect_detail = dq_stats.get("collect_detail", collect_full)
                                _dq_reduce_stats(accum_by_scope, collect_full, collect_zero, collect_near_zero, collect_detail)

                                cur_bits = last_applied_bits
                                qmax = (1 << (cur_bits - 1)) - 1 if cur_bits is not None else None
                                metrics = {
                                    "unet": _dq_compute_metrics(
                                        accum_by_scope["unet"], qmax, collect_full, collect_zero, collect_near_zero, collect_detail
                                    ),
                                    "te": _dq_compute_metrics(
                                        accum_by_scope["te"], qmax, collect_full, collect_zero, collect_near_zero, collect_detail
                                    ),
                                }

                                auto_applied = 0
                                range_mul_before = getattr(args, "dq_delta_range_mul", None)
                                range_mul_after = range_mul_before
                                clip_rate_raw = None
                                clip_rate_ema = dq_auto_ema_state
                                warmup_active = 1 if (dq_auto_warmup_enabled and dq_auto_warmup_remaining > 0) else 0
                                warmup_remain = dq_auto_warmup_remaining if dq_auto_warmup_enabled else 0
                                if warmup_active:
                                    auto_reason = "warmup"
                                elif dq_auto_enabled:
                                    auto_reason = "in_band"
                                else:
                                    auto_reason = ""
                                low_auto_state = dq_low_auto_state
                                low_auto_decision = ""
                                low_auto_reason = ""
                                low_auto_bad = ""
                                low_auto_bad_streak = dq_low_auto_bad_streak
                                low_auto_qerr_per_clip = None
                                low_auto_phase = dq_low_auto_phase
                                low_auto_can_escape = dq_low_auto_can_escape

                                if dq_stats["do_auto"]:
                                    auto_scope = dq_stats["auto_scope"]
                                    if accelerator.is_main_process:
                                        if auto_scope == "unet":
                                            auto_metrics = metrics["unet"]
                                        elif auto_scope == "te":
                                            auto_metrics = metrics["te"]
                                        else:
                                            # combine unet + te
                                            temp_acc = _dq_merge_acc(
                                                accum_by_scope["unet"],
                                                accum_by_scope["te"],
                                                collect_full,
                                                collect_zero,
                                                collect_near_zero,
                                                collect_detail,
                                            )
                                            auto_metrics = _dq_compute_metrics(
                                                temp_acc, qmax, collect_full, collect_zero, collect_near_zero, collect_detail
                                            )

                                        clip_rate_raw = auto_metrics["clip_rate"]
                                        if clip_rate_raw is not None:
                                            if dq_bits_changed_since_auto:
                                                dq_auto_ema_state = clip_rate_raw
                                                dq_low_auto_quant_err_ratio_ema_state = None
                                                dq_bits_changed_since_auto = False
                                                if dq_auto_warmup_enabled:
                                                    dq_auto_warmup_remaining = dq_auto_warmup_reset_updates
                                                    dq_auto_warmup_inband_streak = 0
                                            else:
                                                if dq_auto_ema_state is None:
                                                    dq_auto_ema_state = clip_rate_raw
                                                else:
                                                    dq_auto_ema_state = dq_auto_ema_state * dq_auto_ema + (1.0 - dq_auto_ema) * clip_rate_raw
                                            clip_rate_ema = dq_auto_ema_state

                                            if range_mul_before is None:
                                                range_mul_before = getattr(args, "dq_delta_range_mul", 3.0)
                                            range_mul_after = range_mul_before

                                            if dq_low_auto_enabled and dq_low_auto_start_step is None:
                                                dq_low_auto_start_step = step_idx

                                            if dq_low_auto_enabled:
                                                qerr_ratio_raw_for_low_auto = auto_metrics.get("quant_err_ratio")
                                                if qerr_ratio_raw_for_low_auto is not None:
                                                    if dq_low_auto_quant_err_ratio_ema_state is None:
                                                        dq_low_auto_quant_err_ratio_ema_state = qerr_ratio_raw_for_low_auto
                                                    else:
                                                        dq_low_auto_quant_err_ratio_ema_state = (
                                                            dq_low_auto_quant_err_ratio_ema_state * dq_auto_ema
                                                            + (1.0 - dq_auto_ema) * qerr_ratio_raw_for_low_auto
                                                        )
                                                low_auto_qerr_per_clip = _dq_qerr_per_clip(
                                                    dq_low_auto_quant_err_ratio_ema_state,
                                                    clip_rate_ema,
                                                )

                                            warmup_step_active = dq_auto_warmup_enabled and dq_auto_warmup_remaining > 0
                                            if warmup_step_active:
                                                if dq_auto_active_clip_low <= clip_rate_ema <= dq_auto_active_clip_high:
                                                    dq_auto_warmup_inband_streak += 1
                                                else:
                                                    dq_auto_warmup_inband_streak = 0
                                                dq_auto_warmup_remaining = max(0, dq_auto_warmup_remaining - 1)
                                                if dq_auto_warmup_inband_streak >= 3:
                                                    dq_auto_warmup_remaining = 0
                                                auto_reason = "warmup"
                                                if dq_low_auto_enabled:
                                                    low_auto_state = "observe"
                                                    low_auto_decision = "observe"
                                                    low_auto_reason = "warmup"
                                                    low_auto_phase = "warmup"
                                                    low_auto_can_escape = 0
                                            else:
                                                if dq_low_auto_enabled:
                                                    low_auto_frozen = progress_frac >= dq_low_auto_freeze_progress
                                                    low_auto_stats_ready = (
                                                        dq_low_auto_quant_err_ratio_ema_state is not None
                                                        and low_auto_qerr_per_clip is not None
                                                    )
                                                    low_auto_is_bad = (
                                                        low_auto_stats_ready
                                                        and dq_low_auto_quant_err_ratio_ema_state >= dq_low_auto_qerr_ratio_threshold
                                                        and low_auto_qerr_per_clip >= dq_low_auto_qerr_per_clip_threshold
                                                    )
                                                    low_auto_bad = 1 if low_auto_is_bad else (0 if low_auto_stats_ready else "")
                                                    low_auto_can_escape = 1 if (
                                                        progress_frac >= dq_low_auto_min_progress
                                                        and not low_auto_frozen
                                                        and not dq_low_auto_escaped
                                                    ) else 0
                                                    if progress_frac < dq_low_auto_min_progress:
                                                        dq_low_auto_state = "observe"
                                                        low_auto_decision = "observe"
                                                        low_auto_reason = "min_progress"
                                                        low_auto_phase = "pre_min_progress"
                                                        dq_low_auto_bad_streak = 0
                                                    elif dq_low_auto_escaped:
                                                        dq_low_auto_state = "mid_lock"
                                                        low_auto_decision = "mid_lock"
                                                        low_auto_reason = "escaped_once"
                                                        low_auto_phase = "escaped"
                                                        dq_low_auto_bad_streak = 0
                                                    elif low_auto_frozen:
                                                        dq_low_auto_state = "frozen"
                                                        low_auto_decision = "frozen"
                                                        low_auto_reason = "freeze_progress"
                                                        low_auto_phase = "frozen"
                                                        if low_auto_is_bad:
                                                            dq_low_auto_bad_streak += 1
                                                        else:
                                                            dq_low_auto_bad_streak = 0
                                                    elif not low_auto_stats_ready:
                                                        dq_low_auto_state = "observe"
                                                        low_auto_decision = "observe"
                                                        low_auto_reason = "insufficient_qerr_stats"
                                                        low_auto_phase = "active"
                                                        dq_low_auto_bad_streak = 0
                                                    else:
                                                        low_auto_phase = "active"
                                                        if low_auto_is_bad:
                                                            dq_low_auto_bad_streak += 1
                                                            low_auto_decision = "bad_count"
                                                            low_auto_reason = "low_bad"
                                                        else:
                                                            dq_low_auto_bad_streak = 0
                                                            low_auto_decision = "keep_low"
                                                            low_auto_reason = "in_band"
                                                        if dq_low_auto_bad_streak >= dq_low_auto_bad_streak_threshold:
                                                            dq_low_auto_escaped = True
                                                            dq_auto_active_band = "mid"
                                                            dq_auto_active_clip_low, dq_auto_active_clip_high = DQ_DELTA_AUTO_BANDS["mid"]
                                                            dq_low_auto_state = "escape_to_mid"
                                                            low_auto_decision = "escape_to_mid"
                                                            low_auto_reason = "bad_streak_met"
                                                            low_auto_phase = "escaped"
                                                            low_auto_can_escape = 0
                                                low_auto_state = dq_low_auto_state
                                                low_auto_bad_streak = dq_low_auto_bad_streak
                                                dq_low_auto_phase = low_auto_phase
                                                dq_low_auto_can_escape = low_auto_can_escape

                                                if dq_auto_use_raw:
                                                    clip_high_hit = (
                                                        clip_rate_raw is not None
                                                        and clip_rate_ema > dq_auto_active_clip_high
                                                        and clip_rate_raw > dq_auto_active_clip_high
                                                    )
                                                    clip_low_hit = (
                                                        clip_rate_raw is not None
                                                        and clip_rate_ema < dq_auto_active_clip_low
                                                        and clip_rate_raw < dq_auto_active_clip_low
                                                    )
                                                else:
                                                    clip_high_hit = clip_rate_ema > dq_auto_active_clip_high
                                                    clip_low_hit = clip_rate_ema < dq_auto_active_clip_low
                                                if clip_high_hit:
                                                    range_mul_after = range_mul_before * dq_auto_mul_up
                                                    auto_reason = "clip_high"
                                                elif clip_low_hit:
                                                    range_mul_after = range_mul_before * dq_auto_mul_down
                                                    auto_reason = "clip_low"
                                                else:
                                                    auto_reason = "in_band"
                                                range_mul_after = max(dq_auto_min, min(dq_auto_max, range_mul_after))
                                                if range_mul_after != range_mul_before:
                                                    auto_applied = 1

                                            warmup_active = 1 if warmup_step_active else 0
                                            warmup_remain = dq_auto_warmup_remaining if dq_auto_warmup_enabled else 0
                                            low_auto_state = dq_low_auto_state
                                            low_auto_bad_streak = dq_low_auto_bad_streak

                                    if dist.is_available() and dist.is_initialized():
                                        range_tensor = torch.tensor(
                                            range_mul_after if accelerator.is_main_process else 0.0,
                                            device=accelerator.device,
                                            dtype=torch.float32,
                                        )
                                        dist.broadcast(range_tensor, src=0)
                                        range_mul_after = float(range_tensor.item())

                                    if range_mul_after is not None:
                                        args.dq_delta_range_mul = range_mul_after
                                        _set_delta_fake_quant_compat(
                                            accelerator.unwrap_model(network),
                                            getattr(args, "dq_delta_step", None),
                                            args.dq_delta_mode,
                                            granularity=args.dq_delta_granularity,
                                            stat=args.dq_delta_stat,
                                            bits=cur_bits,
                                            range_mul=range_mul_after,
                                            on_z=getattr(args, "dq_quantize_z", False),
                                            use_triton=getattr(args, "dq_delta_use_triton", False),
                                            triton_stats=getattr(args, "dq_delta_triton_stats", False),
                                        )

                                if dq_stats["do_log"] and accelerator.is_main_process and dq_log_path:
                                    log_full_detail = dq_log_detail == "full" or dq_log_mode == "per_module"
                                    include_near_zero = log_full_detail and "near_zero_rate" in dq_log_extra
                                    header = _dq_log_header(dq_log_mode, include_near_zero, detail=dq_log_detail)
                                    log_scopes = ["unet", "te"] if dq_stats["log_scope"] == "both" else [dq_stats["log_scope"]]
                                    quant_err_rms_ema = dq_quant_err_rms_ema_state
                                    quant_err_ratio_ema = dq_quant_err_ratio_ema_state
                                    if dq_stats["log_scope"] == "both":
                                        ema_acc = _dq_merge_acc(
                                            accum_by_scope["unet"],
                                            accum_by_scope["te"],
                                            collect_full,
                                            collect_zero,
                                            collect_near_zero,
                                            collect_detail,
                                        )
                                        ema_metrics = _dq_compute_metrics(
                                            ema_acc, qmax, collect_full, collect_zero, collect_near_zero, collect_detail
                                        )
                                    else:
                                        ema_metrics = metrics[dq_stats["log_scope"]]
                                    if ema_metrics is not None:
                                        quant_err_rms_raw = ema_metrics["quant_err_rms"]
                                        quant_err_ratio_raw = ema_metrics["quant_err_ratio"]
                                        if quant_err_rms_raw is not None:
                                            if dq_quant_err_rms_ema_state is None:
                                                dq_quant_err_rms_ema_state = quant_err_rms_raw
                                            else:
                                                dq_quant_err_rms_ema_state = (
                                                    dq_quant_err_rms_ema_state * dq_auto_ema
                                                    + (1.0 - dq_auto_ema) * quant_err_rms_raw
                                                )
                                        if quant_err_ratio_raw is not None:
                                            if dq_quant_err_ratio_ema_state is None:
                                                dq_quant_err_ratio_ema_state = quant_err_ratio_raw
                                            else:
                                                dq_quant_err_ratio_ema_state = (
                                                    dq_quant_err_ratio_ema_state * dq_auto_ema
                                                    + (1.0 - dq_auto_ema) * quant_err_ratio_raw
                                                )
                                        quant_err_rms_ema = dq_quant_err_rms_ema_state
                                        quant_err_ratio_ema = dq_quant_err_ratio_ema_state
                                    for scope in log_scopes:
                                        m = metrics[scope]
                                        values = [
                                            epoch + 1,
                                            step_idx,
                                            scope,
                                            dq_stats["target"],
                                            cur_bits if cur_bits is not None else "",
                                            getattr(args, "dq_delta_step", None) or "",
                                            range_mul_after if range_mul_after is not None else "",
                                            args.dq_delta_stat,
                                            args.dq_delta_granularity,
                                            args.dq_delta_mode,
                                        ]
                                        if dq_log_mode == "per_module":
                                            for item in dq_stats["per_module"]:
                                                if item["scope"] != scope:
                                                    continue
                                                numel = item["numel"].item()
                                                clip_rate = (item["clip_count"] / item["numel"]).item() if numel > 0 else None
                                                zero_rate = (item["zero_count"] / item["numel"]).item() if item["zero_count"] is not None and numel > 0 else None
                                                near_zero_rate = (item["near_zero_count"] / item["numel"]).item() if item["near_zero_count"] is not None and numel > 0 else None
                                                rms = math.sqrt((item["sumsq"] / item["numel"]).item()) if item["sumsq"] is not None and numel > 0 else None
                                                absmax = item["absmax"].item() if item["absmax"] is not None else None
                                                scale_min = item["scale_min"].item() if item["scale_min"] is not None else None
                                                scale_max = item["scale_max"].item() if item["scale_max"] is not None else None
                                                scale_mean = (item["scale_sum"] / item["scale_count"]).item() if item["scale_sum"] is not None and item["scale_count"] is not None and item["scale_count"].item() > 0 else None
                                                range_val = scale_mean * qmax if scale_mean is not None and qmax is not None else None
                                                quant_err_rms = quant_err_ratio = None
                                                clip_err_rms = round_err_rms = clip_err_ratio = round_err_ratio = clip_share = round_share = None
                                                if item["sumsq"] is not None and item["xq_sumsq"] is not None and item["xxq_sum"] is not None and numel > 0:
                                                    err_sumsq = item["sumsq"] + item["xq_sumsq"] - (2.0 * item["xxq_sum"])
                                                    err_sumsq = torch.clamp(err_sumsq, min=0.0)
                                                    quant_err_rms = math.sqrt((err_sumsq / item["numel"]).item())
                                                    if rms is not None:
                                                        quant_err_ratio = quant_err_rms / (rms + 1e-12)
                                                    if item.get("clip_err_sumsq") is not None:
                                                        clip_err_rms = math.sqrt((torch.clamp(item["clip_err_sumsq"], min=0.0) / item["numel"]).item())
                                                        if rms is not None:
                                                            clip_err_ratio = clip_err_rms / (rms + 1e-12)
                                                        if err_sumsq.item() > 0:
                                                            clip_share = (torch.clamp(item["clip_err_sumsq"], min=0.0) / (err_sumsq + 1e-12)).item()
                                                    if item.get("round_err_sumsq") is not None:
                                                        round_err_rms = math.sqrt((torch.clamp(item["round_err_sumsq"], min=0.0) / item["numel"]).item())
                                                        if rms is not None:
                                                            round_err_ratio = round_err_rms / (rms + 1e-12)
                                                        if err_sumsq.item() > 0:
                                                            round_share = (torch.clamp(item["round_err_sumsq"], min=0.0) / (err_sumsq + 1e-12)).item()
                                                row = values + [item["module"], item["shape"], rms]
                                                if log_full_detail:
                                                    row += [
                                                        absmax,
                                                        range_val,
                                                        scale_min,
                                                        scale_mean,
                                                        scale_max,
                                                        qmax if qmax is not None else "",
                                                    ]
                                                row += [
                                                    clip_rate,
                                                    clip_rate_ema if clip_rate_ema is not None else "",
                                                ]
                                                if log_full_detail:
                                                    row.append(zero_rate)
                                                row += [
                                                    quant_err_rms,
                                                    quant_err_rms_ema if quant_err_rms_ema is not None else "",
                                                    quant_err_ratio,
                                                    quant_err_ratio_ema if quant_err_ratio_ema is not None else "",
                                                ]
                                                if include_near_zero:
                                                    row.append(near_zero_rate)
                                                row += [
                                                    dq_auto_active_band,
                                                    dq_auto_active_clip_low,
                                                    dq_auto_active_clip_high,
                                                    low_auto_state if dq_low_auto_enabled else "",
                                                    low_auto_bad if dq_low_auto_enabled else "",
                                                    low_auto_bad_streak if dq_low_auto_enabled else "",
                                                    progress_frac,
                                                    dq_low_auto_min_progress if dq_low_auto_enabled else "",
                                                    dq_low_auto_freeze_progress if dq_low_auto_enabled else "",
                                                    dq_low_auto_qerr_ratio_threshold if dq_low_auto_enabled else "",
                                                    dq_low_auto_qerr_per_clip_threshold if dq_low_auto_enabled else "",
                                                    low_auto_phase if dq_low_auto_enabled else "",
                                                ]
                                                row += [
                                                    numel,
                                                    auto_applied,
                                                    range_mul_before if range_mul_before is not None else "",
                                                    range_mul_after if range_mul_after is not None else "",
                                                    warmup_active,
                                                    warmup_remain,
                                                    auto_reason,
                                                    dq_auto_init_applied,
                                                    dq_auto_init_value if dq_auto_init_value is not None else "",
                                                    dq_auto_init_clip_target if dq_auto_init_clip_target is not None else "",
                                                ]
                                                _write_csv(dq_log_path, header, ",".join(_dq_format_value(v) for v in row))
                                        else:
                                            row = values + [m["rms"]]
                                            if log_full_detail:
                                                row += [
                                                    m["absmax"],
                                                    m["range"],
                                                    m["scale_min"],
                                                    m["scale_mean"],
                                                    m["scale_max"],
                                                    qmax if qmax is not None else "",
                                                ]
                                            row += [
                                                m["clip_rate"],
                                                clip_rate_ema if clip_rate_ema is not None else "",
                                            ]
                                            if log_full_detail:
                                                row.append(m["zero_rate"])
                                            row += [
                                                m["quant_err_rms"],
                                                quant_err_rms_ema if quant_err_rms_ema is not None else "",
                                                m["quant_err_ratio"],
                                                quant_err_ratio_ema if quant_err_ratio_ema is not None else "",
                                            ]
                                            if include_near_zero:
                                                row.append(m["near_zero_rate"])
                                            row += [
                                                dq_auto_active_band,
                                                dq_auto_active_clip_low,
                                                dq_auto_active_clip_high,
                                                low_auto_state if dq_low_auto_enabled else "",
                                                low_auto_bad if dq_low_auto_enabled else "",
                                                low_auto_bad_streak if dq_low_auto_enabled else "",
                                                progress_frac,
                                                dq_low_auto_min_progress if dq_low_auto_enabled else "",
                                                dq_low_auto_freeze_progress if dq_low_auto_enabled else "",
                                                dq_low_auto_qerr_ratio_threshold if dq_low_auto_enabled else "",
                                                dq_low_auto_qerr_per_clip_threshold if dq_low_auto_enabled else "",
                                                low_auto_phase if dq_low_auto_enabled else "",
                                            ]
                                            row += [
                                                m["numel"],
                                                auto_applied,
                                                range_mul_before if range_mul_before is not None else "",
                                                range_mul_after if range_mul_after is not None else "",
                                                warmup_active,
                                                warmup_remain,
                                                auto_reason,
                                                dq_auto_init_applied,
                                                dq_auto_init_value if dq_auto_init_value is not None else "",
                                                dq_auto_init_clip_target if dq_auto_init_clip_target is not None else "",
                                            ]
                                            _write_csv(dq_log_path, header, ",".join(_dq_format_value(v) for v in row))

                                if dq_stats["do_auto"] and accelerator.is_main_process and dq_auto_log_path:
                                    include_near_zero = "near_zero_rate" in dq_log_extra
                                    header = _dq_auto_log_header(dq_auto_log_format == "full_schema", include_near_zero)
                                    if dq_auto_log_format == "full_schema":
                                        row = [
                                            epoch + 1,
                                            step_idx,
                                            dq_stats["auto_scope"],
                                            dq_stats["target"],
                                            cur_bits if cur_bits is not None else "",
                                            getattr(args, "dq_delta_step", None) or "",
                                            range_mul_after if range_mul_after is not None else "",
                                            args.dq_delta_stat,
                                            args.dq_delta_granularity,
                                            args.dq_delta_mode,
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            qmax if qmax is not None else "",
                                            clip_rate_raw if clip_rate_raw is not None else "",
                                            clip_rate_ema if clip_rate_ema is not None else "",
                                            "",
                                            "",
                                            "",
                                            "",
                                            "",
                                        ]
                                        if include_near_zero:
                                            row.append("")
                                        auto_qerr_per_clip = _dq_qerr_per_clip(
                                            dq_low_auto_quant_err_ratio_ema_state if dq_low_auto_enabled else None,
                                            clip_rate_ema,
                                        )
                                        row += [
                                            auto_qerr_per_clip if auto_qerr_per_clip is not None else "",
                                            dq_qerr_per_clip_floor,
                                            dq_auto_active_band,
                                            dq_auto_active_clip_low,
                                            dq_auto_active_clip_high,
                                            low_auto_state if dq_low_auto_enabled else "",
                                            low_auto_bad if dq_low_auto_enabled else "",
                                            low_auto_bad_streak if dq_low_auto_enabled else "",
                                            progress_frac,
                                            dq_low_auto_min_progress if dq_low_auto_enabled else "",
                                            dq_low_auto_freeze_progress if dq_low_auto_enabled else "",
                                            dq_low_auto_qerr_ratio_threshold if dq_low_auto_enabled else "",
                                            dq_low_auto_qerr_per_clip_threshold if dq_low_auto_enabled else "",
                                            low_auto_phase if dq_low_auto_enabled else "",
                                        ]
                                        if dq_log_error_parts:
                                            row += ["", "", "", "", "", ""]
                                        row += [
                                            "",
                                            auto_applied,
                                            range_mul_before if range_mul_before is not None else "",
                                            range_mul_after if range_mul_after is not None else "",
                                            warmup_active,
                                            warmup_remain,
                                            auto_reason,
                                            dq_auto_init_applied,
                                            dq_auto_init_value if dq_auto_init_value is not None else "",
                                            dq_auto_init_clip_target if dq_auto_init_clip_target is not None else "",
                                        ]
                                        _write_csv(dq_auto_log_path, header, ",".join(_dq_format_value(v) for v in row))
                                    else:
                                        row = [
                                            step_idx,
                                            dq_stats["auto_scope"],
                                            dq_stats["target"],
                                            cur_bits if cur_bits is not None else "",
                                            clip_rate_raw if clip_rate_raw is not None else "",
                                            clip_rate_ema if clip_rate_ema is not None else "",
                                            range_mul_before if range_mul_before is not None else "",
                                            range_mul_after if range_mul_after is not None else "",
                                            auto_applied,
                                            warmup_active,
                                            warmup_remain,
                                            auto_reason,
                                            dq_auto_init_applied,
                                            dq_auto_init_value if dq_auto_init_value is not None else "",
                                            dq_auto_init_clip_target if dq_auto_init_clip_target is not None else "",
                                            low_auto_qerr_per_clip if low_auto_qerr_per_clip is not None else "",
                                            dq_qerr_per_clip_floor,
                                            dq_auto_active_band,
                                            dq_auto_active_clip_low,
                                            dq_auto_active_clip_high,
                                            low_auto_state if dq_low_auto_enabled else "",
                                            low_auto_decision if dq_low_auto_enabled else "",
                                            low_auto_reason if dq_low_auto_enabled else "",
                                            low_auto_bad if dq_low_auto_enabled else "",
                                            low_auto_bad_streak if dq_low_auto_enabled else "",
                                            progress_frac,
                                            dq_low_auto_min_progress if dq_low_auto_enabled else "",
                                            dq_low_auto_freeze_progress if dq_low_auto_enabled else "",
                                            dq_low_auto_qerr_ratio_threshold if dq_low_auto_enabled else "",
                                            dq_low_auto_qerr_per_clip_threshold if dq_low_auto_enabled else "",
                                            low_auto_phase if dq_low_auto_enabled else "",
                                            low_auto_can_escape if dq_low_auto_enabled else "",
                                        ]
                                        _write_csv(dq_auto_log_path, header, ",".join(_dq_format_value(v) for v in row))
                    if rank_log_enabled and accelerator.is_main_process and rank_log_path and (not skip_step_flag):
                        step_idx = global_step
                        if step_idx % rank_log_every == 0:
                            unwrapped = accelerator.unwrap_model(network)
                            if hasattr(unwrapped, "compute_rank_stats"):
                                rank_stats = None
                                try:
                                    rank_stats = unwrapped.compute_rank_stats(scope="unet")
                                except Exception as exc:
                                    logger.warning("failed to compute rank stats: %s", str(exc))
                                if rank_stats is not None:
                                    header = _rank_log_header(rank_log_mode)
                                    lr_snapshot = self.collect_rank_log_lr_snapshot(args, lr_scheduler, lr_descriptions)
                                    if rank_log_mode == "per_module":
                                        for item in rank_stats.get("per_module", []):
                                            row = [
                                                epoch + 1,
                                                step_idx,
                                                "unet",
                                                lr_snapshot.get("unet_lr_min"),
                                                lr_snapshot.get("unet_lr_max"),
                                                lr_snapshot.get("te1_lr_min"),
                                                lr_snapshot.get("te1_lr_max"),
                                                lr_snapshot.get("te2_lr_min"),
                                                lr_snapshot.get("te2_lr_max"),
                                                item.get("module"),
                                                item.get("r"),
                                                item.get("sat"),
                                                item.get("top1"),
                                                item.get("energy"),
                                            ]
                                            _write_csv(rank_log_path, header, ",".join(_dq_format_value(v) for v in row))
                                    else:
                                        row = [
                                            epoch + 1,
                                            step_idx,
                                            "unet",
                                            lr_snapshot.get("unet_lr_min"),
                                            lr_snapshot.get("unet_lr_max"),
                                            lr_snapshot.get("te1_lr_min"),
                                            lr_snapshot.get("te1_lr_max"),
                                            lr_snapshot.get("te2_lr_min"),
                                            lr_snapshot.get("te2_lr_max"),
                                            rank_stats.get("rank_dim"),
                                            rank_stats.get("sat_wmean"),
                                            rank_stats.get("sat_p50"),
                                            rank_stats.get("sat_p95"),
                                            rank_stats.get("sat_max"),
                                            rank_stats.get("top1_p95"),
                                            rank_stats.get("energy_sum"),
                                        ]
                                        _write_csv(rank_log_path, header, ",".join(_dq_format_value(v) for v in row))
                    self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}
                if skip_grad_norm:
                    logs["skipped"] = skipped_steps
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, keys_scaled, mean_norm, maximum_norm
                    )
                    if skip_grad_norm:
                        logs["train/skipped_steps"] = skipped_steps
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            _group_flush_step_buffer(force=True, current_global_step=global_step)

            if group_loss_log_enabled and group_loss_epoch_summary and group_loss_tracker is not None:
                if accelerator.is_main_process:
                    for group, ema_loss_end, count_epoch, mean_loss_epoch in group_loss_tracker.get_epoch_summary():
                        row = [epoch + 1, group, ema_loss_end, count_epoch, mean_loss_epoch]
                        _group_write_csv(
                            group_loss_epoch_log_path,
                            "epoch,group,ema_loss_end,count_epoch,mean_loss_epoch",
                            row,
                            epoch_log=True,
                        )
                group_loss_tracker.reset_epoch()

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)
            if proxy_scoring_mode:
                clean_memory_on_device(accelerator.device)

            progress_ratio = (epoch + 1) / num_train_epochs
            if args.avg_cp:
                shadow_log_payload = None
                if proxy_scoring_mode:
                    shadow_log_payload = {
                        "epoch": epoch + 1,
                        "progress": round(progress_ratio, 6),
                        "construction_pool_size": _shadow_candidate_pool_size_for_log(),
                        "construction_pool_target_size": shadow_candidate_pool_target_size,
                        "bank_ready": len(shadow_bank) >= args.avg_shadow_bank_size,
                        "bank_size": len(shadow_bank),
                        "bank_unique_class_tokens": None,
                        "bank_max_per_class_tokens": None,
                        "bank_unique_image_keys": None,
                        "bank_max_per_image_key": None,
                        "bank_timestep_bins": None,
                        "avg_mode": args.avg_mode,
                        "avg_promote_pick": avg_promote_pick,
                        "avg_window": args.avg_window,
                        "raw_proxy_loss": None,
                        "center_proxy_loss": None,
                        "ema_proxy_loss": None,
                        "uniform_proxy_loss": None,
                        "best_candidate_mode": None,
                        "best_candidate_proxy_loss": None,
                        "selected_candidate_mode": args.avg_mode,
                        "selected_proxy_loss": None,
                        "delta_abs": None,
                        "delta_pct": None,
                        "winner": None,
                        "winner_mode": None,
                        "virtual_margin_ok": False,
                        "virtual_win_streak": shadow_virtual_win_streak,
                        "virtual_would_promote": False,
                        "promote_mode": promote_mode,
                        "promote_decision": None,
                        "promote_applied": False,
                        "next_epoch_source": "raw",
                        "optimizer_reset_applied": False,
                        "window_epochs": list(cp_window_epochs) if cp_window_epochs is not None else [],
                        "status": "before_avg_begin" if progress_ratio < args.avg_begin else "pending",
                    }

                if progress_ratio >= args.avg_begin:
                    unwrapped_network = accelerator.unwrap_model(network)
                    raw_sd = filter_lora_state_dict(unwrapped_network.state_dict())
                    final_avg_raw_sd = raw_sd
                    cp_window.append(raw_sd)
                    cp_window_epochs.append(epoch + 1)
                    if len(cp_window) >= args.avg_window:
                        final_avg_center_sd = self._restore_frozen_te_state_dict(
                            average_state_dicts(list(cp_window), args.avg_mode)
                        )

                    if proxy_scoring_mode:
                        shadow_log_payload["window_epochs"] = list(cp_window_epochs)
                        bank_ready = _finalize_shadow_bank_if_ready(is_last_epoch=((epoch + 1) >= num_train_epochs))
                        shadow_log_payload["construction_pool_size"] = _shadow_candidate_pool_size_for_log()
                        shadow_log_payload["bank_ready"] = bank_ready
                        shadow_log_payload["bank_size"] = len(shadow_bank)
                        if bank_ready:
                            shadow_log_payload.update(_shadow_bank_metadata(shadow_bank))

                        if not bank_ready:
                            shadow_log_payload["status"] = "building_pool"
                        elif len(cp_window) < args.avg_window:
                            shadow_log_payload["status"] = "waiting_window"
                        elif accelerator.is_main_process:
                            candidate_modes = resolve_avg_proxy_candidate_modes(avg_cp_mode, avg_promote_pick, args.avg_mode)
                            candidate_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {
                                candidate_mode: self._restore_frozen_te_state_dict(
                                    average_state_dicts(list(cp_window), candidate_mode)
                                )
                                for candidate_mode in candidate_modes
                            }

                            rng_state = _capture_torch_rng_state()
                            raw_score = None
                            candidate_scores: Dict[str, float] = {}
                            try:
                                raw_score = _score_shadow_bank(unwrapped_network)
                                for candidate_mode, candidate_sd in candidate_state_dicts.items():
                                    _restore_torch_rng_state(rng_state)
                                    unwrapped_network.load_state_dict(candidate_sd, strict=False)
                                    candidate_scores[candidate_mode] = _score_shadow_bank(unwrapped_network)
                            finally:
                                unwrapped_network.load_state_dict(raw_sd, strict=False)
                                _restore_torch_rng_state(rng_state)
                                clean_memory_on_device(accelerator.device)

                            center_score = candidate_scores[args.avg_mode]
                            if "ema" in candidate_scores and "uniform" in candidate_scores:
                                best_candidate_mode = min(("ema", "uniform"), key=lambda mode: candidate_scores[mode])
                                best_candidate_score = candidate_scores[best_candidate_mode]
                            else:
                                best_candidate_mode = None
                                best_candidate_score = None
                            if avg_promote_pick == "best":
                                selected_candidate_mode = best_candidate_mode
                            else:
                                selected_candidate_mode = args.avg_mode
                            selected_sd = candidate_state_dicts[selected_candidate_mode]
                            selected_score = candidate_scores[selected_candidate_mode]
                            delta_abs = selected_score - raw_score
                            delta_pct = (delta_abs / raw_score * 100.0) if raw_score not in (None, 0.0) else None
                            margin_ok = selected_score < raw_score * (1.0 - args.avg_shadow_margin)
                            winner = "center" if margin_ok else "raw"
                            winner_mode = selected_candidate_mode if margin_ok else "raw"
                            shadow_virtual_win_streak = shadow_virtual_win_streak + 1 if margin_ok else 0
                            would_promote = shadow_virtual_win_streak >= args.avg_shadow_patience
                            promote_applied = False
                            optimizer_reset_applied = False
                            next_epoch_source = "raw"

                            shadow_log_payload.update(
                                {
                                    "raw_proxy_loss": round(raw_score, 10),
                                    "center_proxy_loss": round(center_score, 10),
                                    "ema_proxy_loss": (
                                        round(candidate_scores["ema"], 10) if "ema" in candidate_scores else None
                                    ),
                                    "uniform_proxy_loss": (
                                        round(candidate_scores["uniform"], 10) if "uniform" in candidate_scores else None
                                    ),
                                    "best_candidate_mode": best_candidate_mode,
                                    "best_candidate_proxy_loss": (
                                        round(best_candidate_score, 10) if best_candidate_score is not None else None
                                    ),
                                    "selected_candidate_mode": selected_candidate_mode,
                                    "selected_proxy_loss": round(selected_score, 10),
                                    "delta_abs": round(delta_abs, 10),
                                    "delta_pct": round(delta_pct, 6) if delta_pct is not None else None,
                                    "winner": winner,
                                    "winner_mode": winner_mode,
                                    "virtual_margin_ok": margin_ok,
                                    "virtual_win_streak": shadow_virtual_win_streak,
                                    "virtual_would_promote": would_promote,
                                    "promote_decision": winner_mode,
                                    "status": "scored",
                                }
                            )

                            if promote_mode and winner == "center" and would_promote:
                                unwrapped_network.load_state_dict(selected_sd, strict=False)
                                promote_applied = True
                                next_epoch_source = "center"
                                if args.avg_reset_stats:
                                    _reset_optimizer_avg_stats()
                                    optimizer_reset_applied = True

                            shadow_log_payload.update(
                                {
                                    "promote_applied": promote_applied,
                                    "next_epoch_source": next_epoch_source,
                                    "optimizer_reset_applied": optimizer_reset_applied,
                                }
                            )

                            logger.info(
                                "avg_cp %s epoch %d: raw=%.6f center(%s)=%.6f ema=%s uniform=%s winner=%s promote_pick=%s selected=%s streak=%d promote=%s window=%s",
                                avg_cp_mode,
                                epoch + 1,
                                raw_score,
                                args.avg_mode,
                                center_score,
                                (
                                    f"{candidate_scores['ema']:.6f}"
                                    if "ema" in candidate_scores
                                    else "not_scored"
                                ),
                                (
                                    f"{candidate_scores['uniform']:.6f}"
                                    if "uniform" in candidate_scores
                                    else "not_scored"
                                ),
                                winner_mode,
                                avg_promote_pick,
                                selected_candidate_mode,
                                shadow_virtual_win_streak,
                                promote_applied,
                                list(cp_window_epochs),
                            )

                        accelerator.wait_for_everyone()
                    else:
                        if len(cp_window) == args.avg_window:
                            start_ep = epoch - args.avg_window + 2
                            if start_ep < 1:
                                start_ep = 1
                            logger.info(f"averaging checkpoints from epoch {start_ep} to {epoch + 1}")
                            avg_sd = self._restore_frozen_te_state_dict(
                                average_state_dicts(list(cp_window), args.avg_mode)
                            )
                            final_avg_center_sd = avg_sd
                            unwrapped_network.load_state_dict(avg_sd, strict=False)
                            if args.avg_reset_stats:
                                _reset_optimizer_avg_stats()
                            if accelerator.distributed_type != DistributedType.NO:
                                sd = broadcast(unwrapped_network.state_dict())
                                unwrapped_network.load_state_dict(sd, strict=False)
                                accelerator.wait_for_everyone()
                            else:
                                accelerator.wait_for_everyone()

                if proxy_scoring_mode:
                    _write_shadow_log(shadow_log_payload)

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())
        _group_flush_step_buffer(force=True, current_global_step=global_step)

        if log_grad_norm and grad_norm_guardian is not None and len(grad_norm_guardian.log_buffer) > 0:
            with open(log_file_path, "a") as f:
                f.writelines(grad_norm_guardian.log_buffer)
            grad_norm_guardian.log_buffer.clear()

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)
            if args.avg_cp and getattr(args, "avg_save_final_raw", False) and final_avg_raw_sd is not None:
                _save_avg_candidate("final_raw", final_avg_raw_sd, num_train_epochs, global_step)
            if proxy_scoring_mode and getattr(args, "avg_save_last_candidates", False):
                if final_avg_raw_sd is not None:
                    _save_avg_candidate("raw", final_avg_raw_sd, num_train_epochs, global_step)
                if final_avg_center_sd is not None:
                    _save_avg_candidate("center", final_avg_center_sd, num_train_epochs, global_step)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    def int_or_float_or_percent(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float_value
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")
    parser.add_argument(
        "--text_encoder_lr1",
        type=float,
        default=None,
        help="learning rate for Text Encoder 1 (ViT-L) / Text Encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--text_encoder_lr2",
        type=float,
        default=None,
        help="learning rate for Text Encoder 2 (BiG-G) / Text Encoder 2 (BiG-G)の学習率",
    )
    parser.add_argument(
        "--network_te_train_targets",
        type=str,
        nargs="+",
        choices=["te1", "te2"],
        default=None,
        help="LoRA targets to train in SDXL text encoders (te1=ViT-L, te2=BiG-G). Omit to train both / SDXLのText Encoderで学習するLoRA対象 (te1=ViT-L, te2=BiG-G)。未指定時は両方を学習",
    )
    parser.add_argument(
        "--te-lr-after",
        nargs="+",
        default=None,
        metavar="value",
        help=(
            "Apply a learning rate multiplier to text encoder(s) once training progress exceeds the specified ratio "
            "(single-step change). Provide ratio (0-1), multiplier, and optional target (both|te1|te2). / "
            "総ステップ数に対する割合を超えたタイミングでText Encoderの学習率に倍率を一度だけ適用します。"
            "指定は <割合> <倍率> [対象(both|te1|te2)] です。"
        ),
    )

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--te1_lr_warmup_steps",
        type=int_or_float_or_percent,
        default=None,
        help=(
            "warmup steps for Text Encoder 1 when using --lr_scheduler constant_with_warmup. "
            "Use an int for steps, a float below 1 for train-step ratio, or a percentage like 5%%."
        ),
    )
    parser.add_argument(
        "--te2_lr_warmup_steps",
        type=int_or_float_or_percent,
        default=None,
        help=(
            "warmup steps for Text Encoder 2 when using --lr_scheduler constant_with_warmup. "
            "Use an int for steps, a float below 1 for train-step ratio, or a percentage like 5%%."
        ),
    )
    parser.add_argument(
        "--te1_freeze_at",
        type=int_or_float_or_percent,
        default=None,
        help="freeze Text Encoder 1 training at this progress ratio or absolute optimizer step.",
    )
    parser.add_argument(
        "--te2_freeze_at",
        type=int_or_float_or_percent,
        default=None,
        help="freeze Text Encoder 2 training at this progress ratio or absolute optimizer step.",
    )

    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    parser.add_argument("--avg_cp", action="store_true", help="enable inter-epoch checkpoint averaging / エポック間のチェックポイント平均を有効化")
    parser.add_argument("--avg_window", type=int, default=4, help="number of checkpoints to average / 平均するチェックポイント数")
    parser.add_argument("--avg_begin", type=float, default=0.6, help="fraction of total epochs to start averaging / 学習の何割から平均を開始するか")
    parser.add_argument(
        "--avg_cp_mode",
        type=str,
        default="live",
        choices=["live", "shadow", "promote"],
        help="avg_cp behavior: live applies averaged weights immediately, shadow scores only, promote adopts center conditionally / avg_cp の動作。live は平均重みを即反映、shadow は採点のみ、promote は条件付きで center を採用",
    )
    parser.add_argument(
        "--avg_mode",
        type=str,
        default="ema",
        choices=["uniform", "ema", "metric"],
        help="averaging mode: uniform, ema or metric / 平均化モード",
    )
    parser.add_argument(
        "--avg_promote_pick",
        type=str,
        default="fixed",
        choices=["fixed", "best"],
        help="promote candidate selection: fixed scores only avg_mode in promote mode, best scores ema/uniform and picks the better one; shadow always scores comparison candidates / promote 候補の選び方。promote の fixed は avg_mode だけを採点、best は ema/uniform の良い方を使う。shadow は常に比較候補を採点する",
    )
    parser.add_argument(
        "--avg_shadow_bank_size",
        type=int,
        default=12,
        help="number of train_proxy batches to retain on CPU for avg_cp shadow scoring / avg_cp shadow 採点用に CPU 保持する train_proxy バッチ数",
    )
    parser.add_argument(
        "--avg_shadow_margin",
        type=float,
        default=0.003,
        help="center wins only when center_score < raw_score * (1 - margin) in shadow/promote mode / shadow/promote で center 勝ちとみなす閾値",
    )
    parser.add_argument(
        "--avg_shadow_patience",
        type=int,
        default=2,
        help="win streak threshold used by shadow logging and promote gating / shadow ログと promote 判定に使う連勝閾値",
    )
    parser.add_argument(
        "--avg_shadow_log_jsonl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write avg_cp shadow/promote epoch logs to jsonl under output_dir / avg_cp shadow/promote の epoch ログを output_dir 配下の jsonl に出力",
    )
    parser.add_argument(
        "--avg_reset_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reset optimizer stats after averaging / 平均化後にOptimizer統計をリセットする",
    )
    parser.add_argument(
        "--avg_save_last_candidates",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save final raw/center LoRA candidates as extra files in shadow/promote mode / shadow/promote 時に最終 raw/center の LoRA を追加保存する",
    )
    parser.add_argument(
        "--avg_save_final_raw",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="save final epoch raw LoRA before avg_cp averaging/promote adoption as <output_name>_final_raw.safetensors / avg_cp の平均反映・promote 採用前の最終 epoch raw LoRA を保存する",
    )
    # LoRA delta fake-quantization (on forward only)
    parser.add_argument(
        "--dq_delta_step",
        type=float,
        default=None,
        help="Fake-quantize only LoRA delta output per forward with this step (STE). None/<=0 to disable / LoRAの差分出力のみをこの刻みでフェイク量子化（STE）。Noneまたは<=0で無効",
    )
    parser.add_argument(
        "--dq_delta_mode",
        type=str,
        default="det",
        choices=["det", "stoch"],
        help="Fake-quant mode: det or stoch / フェイク量子化モード：det=最近傍、stoch=確率的",
    )
    parser.add_argument(
        "--dq_delta_begin",
        type=float,
        default=0.0,
        help="Enable fake-quant after this fraction of total steps [0-1] / 学習進行率がこの割合を超えてから有効化 [0-1]",
    )
    parser.add_argument(
        "--dq_delta_begin_after_lr_warmup",
        action="store_true",
        help=(
            "Begin dq_delta after lr warmup steps (overrides dq_delta_begin) / "
            "lrウォームアップ後にdq_deltaを開始（dq_delta_beginより優先）"
        ),
    )
    parser.add_argument(
        "--dq_delta_scope",
        type=str,
        default="both",
        choices=["unet", "te", "both"],
        help="Apply delta fake-quant to: unet, te, or both / Δのフェイク量子化の適用範囲（unet/te/both）",
    )
    parser.add_argument(
        "--dq_quantize_z",
        action="store_true",
        help="Quantize z=A(x) instead of delta: apply B(Q(z)) / Δではなくz=A(x)を量子化してB(Q(z))を適用する",
    )
    parser.add_argument(
        "--dq_delta_granularity",
        type=str,
        default="tensor",
        choices=["tensor", "channel"],
        help="Granularity of delta fake-quant: whole tensor or per-channel / Δのフェイク量子化の粒度（テンソル全体/チャネル別）",
    )
    parser.add_argument(
        "--dq_delta_stat",
        type=str,
        default="rms",
        choices=["rms", "absmax", "none"],
        help="Statistic for scale/step: rms/absmax/none. Channel-wise when granularity=channel. / スケール/ステップの統計：rms/absmax/none（granularity=channelでチャネル別）",
    )
    parser.add_argument(
        "--dq_delta_bits",
        type=int,
        default=None,
        help="If set, use N-bit symmetric fake-quant (overrides step path). Recommended: 8 / Nビット対称フェイク量子化（step指定より優先）。推奨: 8",
    )
    parser.add_argument(
        "--dq_delta_range_mul",
        type=float,
        default=3.0,
        help="When bits mode with stat=rms, dynamic range = range_mul * RMS. / bitsモードかつstat=rms時の有効レンジ倍率（range=倍率×RMS）",
    )
    parser.add_argument(
        "--dq_delta_bits_sched",
        type=str,
        default=None,
        help="Schedule bits by progress fraction, e.g. '0.0:6,0.5:8,0.8:10' / 学習進行率に応じたビット数スケジュール（例: '0.0:6,0.5:8,0.8:10'）",
    )
    parser.add_argument(
        "--dq_delta_use_triton",
        action="store_true",
        help="Use optional Triton kernels for eligible dq_delta scale and stochastic fake-quant work; falls back to PyTorch",
    )
    parser.add_argument(
        "--dq_delta_triton_stats",
        action="store_true",
        help="Fuse eligible dq_delta basic log/auto stats into the Triton stochastic fake-quant kernel; detail stats fall back to PyTorch",
    )
    # dq_delta logging / auto-tuning
    parser.add_argument(
        "--dq_delta_log",
        action="store_true",
        help="Enable dq_delta logging / dq_delta ログを有効化",
    )
    parser.add_argument(
        "--dq_delta_log_every",
        type=int,
        default=100,
        help="Log every N optimizer steps / ログ間隔（optimizer step）",
    )
    parser.add_argument(
        "--dq_delta_log_scope",
        type=str,
        default=None,
        choices=["unet", "te", "both"],
        help="Scope for dq_delta log; defaults to dq_delta_scope / dq_delta ログ対象（未指定時は dq_delta_scope）",
    )
    parser.add_argument(
        "--dq_delta_log_mode",
        type=str,
        default="summary",
        choices=["summary", "per_module"],
        help="dq_delta log mode: summary or per_module / dq_delta ログ粒度（summary/per_module）",
    )
    parser.add_argument(
        "--dq_delta_log_detail",
        type=str,
        default="basic",
        choices=["basic", "full"],
        help="dq_delta log detail: basic or full / dq_delta log detail（basic/full）",
    )
    parser.add_argument(
        "--dq_delta_log_file",
        type=str,
        default=None,
        help="Path for dq_delta log file / dq_delta ログ出力先",
    )
    parser.add_argument(
        "--dq_delta_log_extra",
        nargs="*",
        default=[],
        choices=["near_zero_rate"],
        help="Extra dq_delta log fields / dq_delta 追加ログ項目",
    )
    parser.add_argument(
        "--rank_log",
        action="store_true",
        help="Enable rank saturation logging / rank飽和ログを有効化",
    )
    parser.add_argument(
        "--rank_log_every",
        type=int,
        default=100,
        help="Log rank stats every N optimizer steps / rankログ間隔（optimizer step）",
    )
    parser.add_argument(
        "--rank_log_mode",
        type=str,
        default="summary",
        choices=["summary", "per_module"],
        help="rank log mode: summary or per_module / rankログ粒度（summary/per_module）",
    )
    parser.add_argument(
        "--rank_log_file",
        type=str,
        default=None,
        help="Path for rank log file / rankログ出力先",
    )
    parser.add_argument(
        "--dq_delta_auto_range_mul",
        action="store_true",
        help="Enable auto range_mul tuning / range_mul の自動調整を有効化",
    )
    parser.add_argument(
        "--dq_delta_auto_preset",
        type=str,
        default=None,
        choices=["default", "clip_rate_high", "clip_rate_high_narrow", "clip_rate_mid", "clip_rate_low", "clip_rate_low_auto"],
        help=(
            "Preset for auto range_mul tuning (overrides clip_low/high only) / "
            "auto range_mul 調整プリセット（clip_low/high のみ上書き）"
        ),
    )
    parser.add_argument(
        "--dq_delta_auto_every",
        type=int,
        default=50,
        help="Auto update interval in optimizer steps / 自動調整間隔（optimizer step）",
    )
    parser.add_argument(
        "--dq_delta_auto_clip_low",
        type=float,
        default=0.0005,
        help="Auto clip_rate low threshold / clip_rate 下限",
    )
    parser.add_argument(
        "--dq_delta_auto_clip_high",
        type=float,
        default=0.003,
        help="Auto clip_rate high threshold / clip_rate 上限",
    )
    parser.add_argument(
        "--dq_delta_auto_mul_up",
        type=float,
        default=1.01,
        help="Auto range_mul increase factor / range_mul 上げ係数",
    )
    parser.add_argument(
        "--dq_delta_auto_mul_down",
        type=float,
        default=0.995,
        help="Auto range_mul decrease factor / range_mul 下げ係数",
    )
    parser.add_argument(
        "--dq_delta_auto_min",
        type=float,
        default=1.0,
        help="Auto range_mul min / range_mul 下限",
    )
    parser.add_argument(
        "--dq_delta_auto_max",
        type=float,
        default=6.0,
        help="Auto range_mul max / range_mul 上限",
    )
    parser.add_argument(
        "--dq_delta_auto_ema",
        type=float,
        default=0.95,
        help="Auto clip_rate EMA / clip_rate EMA 係数",
    )
    parser.add_argument(
        "--dq_delta_auto_use_raw",
        action="store_true",
        help="Include clip_rate_raw in auto checks / auto判定にclip_rate_rawも使う",
    )
    parser.add_argument(
        "--dq_delta_auto_init_range_mul_from_band",
        action="store_true",
        help="Auto-init range_mul from clip band center / clip帯中心からrange_mul初期値を自動算出",
    )
    parser.add_argument(
        "--dq_delta_auto_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable warmup for auto range_mul (EMA/log only) / auto range_mul のウォームアップを有効化（EMA/ログのみ更新）",
    )
    parser.add_argument(
        "--dq_delta_auto_warmup_updates",
        type=int,
        default=0,
        help="Warmup updates override (0=auto) / warmup 回数の上書き（0=内部デフォルト）",
    )
    parser.add_argument(
        "--dq_delta_auto_log_file",
        type=str,
        default=None,
        help="Path for dq_delta auto log file / dq_delta auto ログ出力先",
    )
    parser.add_argument(
        "--dq_delta_auto_log_format",
        type=str,
        default="minimal",
        choices=["minimal", "full_schema"],
        help="Auto log format / auto ログ形式（minimal/full_schema）",
    )
    parser.add_argument(
        "--dq_delta_clip_rate_low_auto_min_progress",
        type=float,
        default=0.25,
        help="Training progress before clip_rate_low_auto escape checks / clip_rate_low_auto の判定開始までの学習進捗",
    )
    parser.add_argument(
        "--dq_delta_clip_rate_low_auto_bad_streak",
        type=int,
        default=3,
        help="Consecutive clip_rate_low_auto bad checks before escaping to mid / clip_rate_low_auto がmidへ逃がすまでの連続bad回数",
    )
    parser.add_argument(
        "--dq_delta_clip_rate_low_auto_freeze_progress",
        type=float,
        default=0.90,
        help="Training progress after which clip_rate_low_auto band switching is frozen / clip_rate_low_auto のband切替を凍結する学習進捗",
    )
    parser.add_argument(
        "--dq_delta_clip_rate_low_auto_qerr_ratio",
        type=float,
        default=0.25,
        help="clip_rate_low_auto QuantErrRatioEMA threshold / clip_rate_low_auto の QuantErrRatioEMA 閾値",
    )
    parser.add_argument(
        "--dq_delta_clip_rate_low_auto_qerr_per_clip",
        type=float,
        default=130.0,
        help="clip_rate_low_auto QErrPerClip threshold / clip_rate_low_auto の QErrPerClip 閾値",
    )
    parser.add_argument(
        "--dq_delta_qerr_per_clip_floor",
        type=float,
        default=0.001,
        help="ClipRate floor for QErrPerClip diagnostics / QErrPerClip 診断で使う ClipRate 下限",
    )
    # ema_* options removed
    # LoRA rounding options
    parser.add_argument(
        "--round_lora_step",
        type=float,
        default=None,
        help="Round LoRA trainable weights to multiples of this step after optimizer step (disabled if None or <= 0) / Optimizer更新後にLoRAの学習パラメータをこの刻みに丸める（Noneまたは<=0で無効）",
    )
    parser.add_argument(
        "--round_lora_mode",
        type=str,
        default="det",
        choices=["det", "stoch"],
        help="Rounding mode: det (deterministic) or stoch (stochastic) / 丸めモード：det=最近傍、stoch=確率的",
    )
    parser.add_argument(
        "--round_lora_every",
        type=int,
        default=1,
        help="Apply rounding every N optimizer steps (only when gradients sync) / 丸めを適用するステップ間隔（同期更新時のみ）",
    )
    parser.add_argument(
        "--round_lora_begin",
        type=float,
        default=0.0,
        help="Begin rounding after this fraction of total steps [0-1] / 学習全体のこの進行率以降で丸めを開始 [0-1]",
    )
    parser.add_argument(
        "--group_loss_log",
        action="store_true",
        help="enable group-wise loss EMA CSV logging / グループ別 loss EMA のCSVログを有効化",
    )
    parser.add_argument(
        "--group_loss_ema_beta",
        type=float,
        default=0.98,
        help="EMA beta for group-wise loss logging / グループ別loss EMAのbeta値",
    )
    parser.add_argument(
        "--group_loss_log_every_n_steps",
        type=int,
        default=100,
        help="flush buffered group-wise loss logs every N global steps / グループ別lossログのバッファを書き出す間隔（global step）",
    )
    parser.add_argument(
        "--group_loss_epoch_summary",
        action="store_true",
        help="append group-wise epoch summary CSV / グループ別のepochサマリCSVを追記出力",
    )
    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
