from __future__ import annotations

import copy
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch


def clone_state_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").clone()
    if isinstance(value, dict):
        return {key: clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = [item.clone().cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda_all" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _guardian_state(guardian: Any, network: torch.nn.Module) -> Optional[dict[str, Any]]:
    if guardian is None:
        return None
    id_to_name = {id(parameter): name for name, parameter in network.named_parameters()}
    previous = None
    if guardian.prev_grad_map is not None:
        previous = {
            id_to_name[param_id]: clone_state_to_cpu(grad)
            for param_id, grad in guardian.prev_grad_map.items()
            if param_id in id_to_name
        }
    return {
        "moving_avg_window": list(guardian.moving_avg_window),
        "moving_avg_maxlen": guardian.moving_avg_window.maxlen,
        "prev_grad_map": previous,
        "prev_grad_norm": guardian.prev_grad_norm,
        "log_buffer": list(guardian.log_buffer),
    }


def _restore_guardian(guardian: Any, network: torch.nn.Module, state: Optional[Mapping[str, Any]]) -> None:
    if guardian is None or state is None:
        return
    guardian.moving_avg_window = deque(state["moving_avg_window"], maxlen=state["moving_avg_maxlen"])
    name_to_parameter = dict(network.named_parameters())
    previous = state.get("prev_grad_map")
    guardian.prev_grad_map = None if previous is None else {
        id(name_to_parameter[name]): tensor.to(name_to_parameter[name].device)
        for name, tensor in previous.items()
        if name in name_to_parameter
    }
    guardian.prev_grad_norm = state.get("prev_grad_norm")
    guardian.log_buffer = list(state.get("log_buffer", []))
    guardian._cached_model = None
    guardian._cached_parameters = None


def _trainer_state(trainer: Any) -> dict[str, Any]:
    fields = (
        "_te_lr_after_cfg",
        "_te_lr_after_resume_state",
        "_te_lr_after_resumed",
        "_te_lr_after_resume_step",
        "_te_freeze_cfg",
        "_te_frozen_state_dict",
    )
    state = {field: clone_state_to_cpu(getattr(trainer, field, None)) for field in fields}
    state["_te_frozen_param_names"] = []
    return state


def _capture_network_runtime(network: torch.nn.Module) -> dict[str, Any]:
    module_modes = {name: module.training for name, module in network.named_modules()}
    requires_grad = {name: parameter.requires_grad for name, parameter in network.named_parameters()}
    runtime = {
        "module_modes": module_modes,
        "requires_grad": requires_grad,
    }
    for field in (
        "multiplier",
        "delta_q_step",
        "delta_q_mode",
        "delta_q_granularity",
        "delta_q_stat",
        "delta_q_bits",
        "delta_q_range_mul",
        "delta_q_on_z",
        "delta_q_use_triton",
        "delta_q_triton_stats",
    ):
        if hasattr(network, field):
            runtime[field] = copy.deepcopy(getattr(network, field))
    loras = getattr(network, "text_encoder_loras", []) + getattr(network, "unet_loras", [])
    runtime["lora_quant_enabled"] = {
        lora.lora_name: bool(getattr(lora, "delta_q_enabled", False)) for lora in loras
    }
    return runtime


def _restore_network_runtime(network: torch.nn.Module, state: Mapping[str, Any]) -> None:
    modules = dict(network.named_modules())
    for name, mode in state.get("module_modes", {}).items():
        if name in modules:
            modules[name].train(bool(mode))
    parameters = dict(network.named_parameters())
    for name, enabled in state.get("requires_grad", {}).items():
        if name in parameters:
            parameters[name].requires_grad_(bool(enabled))
            if not enabled:
                parameters[name].grad = None
    if hasattr(network, "set_multiplier") and "multiplier" in state:
        network.set_multiplier(state["multiplier"])
    for field, value in state.items():
        if field in {"module_modes", "requires_grad", "lora_quant_enabled"}:
            continue
        if hasattr(network, field):
            setattr(network, field, copy.deepcopy(value))
    enabled_map = state.get("lora_quant_enabled", {})
    loras = getattr(network, "text_encoder_loras", []) + getattr(network, "unet_loras", [])
    for lora in loras:
        if lora.lora_name in enabled_map:
            lora.delta_q_enabled = bool(enabled_map[lora.lora_name])


def _move_optimizer_state_to_parameter_devices(optimizer: Any) -> None:
    for group in getattr(optimizer, "param_groups", []):
        for parameter in group.get("params", []):
            state = optimizer.state.get(parameter)
            if not isinstance(state, dict):
                continue
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(parameter.device)


@dataclass
class TrainingSnapshot:
    network_state: dict[str, Any]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    scaler_state: Optional[dict[str, Any]]
    rng_state: dict[str, Any]
    network_runtime: dict[str, Any]
    trainer_state: dict[str, Any]
    guardian_state: Optional[dict[str, Any]]
    metadata: dict[str, Any]

    @classmethod
    def capture(
        cls,
        *,
        network: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        trainer: Any,
        guardian: Any,
        global_step: int,
        epoch: int,
        data_step: int,
    ) -> "TrainingSnapshot":
        frozen_ids = set(getattr(trainer, "_te_frozen_param_ids", set()))
        trainer_state = _trainer_state(trainer)
        trainer_state["_te_frozen_param_names"] = [
            name for name, parameter in network.named_parameters() if id(parameter) in frozen_ids
        ]
        metadata = {
            "global_step": int(global_step),
            "epoch": int(epoch),
            "data_step": int(data_step),
            "lr": [float(group.get("lr", 0.0)) for group in optimizer.param_groups],
        }
        return cls(
            network_state=clone_state_to_cpu(network.state_dict()),
            optimizer_state=clone_state_to_cpu(optimizer.state_dict()),
            scheduler_state=clone_state_to_cpu(scheduler.state_dict()),
            scaler_state=None if scaler is None else clone_state_to_cpu(scaler.state_dict()),
            rng_state=capture_rng_state(),
            network_runtime=_capture_network_runtime(network),
            trainer_state=trainer_state,
            guardian_state=_guardian_state(guardian, network),
            metadata=metadata,
        )

    def restore(
        self,
        *,
        network: torch.nn.Module,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        trainer: Any,
        guardian: Any,
    ) -> None:
        network.load_state_dict(clone_state_to_cpu(self.network_state), strict=True)
        optimizer.load_state_dict(clone_state_to_cpu(self.optimizer_state))
        _move_optimizer_state_to_parameter_devices(optimizer)
        scheduler.load_state_dict(clone_state_to_cpu(self.scheduler_state))
        if scaler is not None and self.scaler_state is not None:
            scaler.load_state_dict(clone_state_to_cpu(self.scaler_state))
        _restore_network_runtime(network, self.network_runtime)
        for field, value in self.trainer_state.items():
            if field == "_te_frozen_param_names":
                continue
            setattr(trainer, field, clone_state_to_cpu(value))
        names = set(self.trainer_state.get("_te_frozen_param_names", []))
        trainer._te_frozen_param_ids = {
            id(parameter) for name, parameter in network.named_parameters() if name in names
        }
        _restore_guardian(guardian, network, self.guardian_state)
        optimizer.zero_grad(set_to_none=True)
        restore_rng_state(self.rng_state)
