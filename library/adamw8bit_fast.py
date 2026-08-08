# The optimizer step control flow in this file is adapted from bitsandbytes,
# which is distributed under the MIT License.
# See third_party/bitsandbytes-LICENSE.txt.

from __future__ import annotations

import logging
from typing import Callable, Optional

import torch

try:
    import bitsandbytes as bnb
except ImportError as exc:  # pragma: no cover - handled by optimizer selection
    raise ImportError("AdamW8bitFast requires bitsandbytes") from exc

logger = logging.getLogger(__name__)
BITSANDBYTES_VERSION = getattr(bnb, "__version__", "unknown")


class AdamW8bitFast(bnb.optim.AdamW8bit):
    """AdamW8bit with one CUDA synchronization per optimizer step.

    bitsandbytes updates parameters one at a time and synchronizes the whole
    device after every parameter. LoRA commonly has many small parameter
    tensors, so those synchronizations dominate the optimizer step. CUDA
    stream ordering already keeps the queued updates in order; synchronizing
    once after the final update preserves the synchronous ``step()`` contract.

    The fast path is intentionally limited to ordinary parameters on one CUDA
    device. Paged, distributed, tensor-subclass, sparse, multi-device, CPU and
    closure-based calls use the unmodified bitsandbytes implementation.
    """

    _optimizer_display_name = "AdamW8bitFast"
    _stock_optimizer_display_name = "AdamW8bit"

    def _log_fast_path_once(self, device: torch.device) -> None:
        if getattr(self, "_adamw8bit_fast_path_logged", False):
            return
        logger.info(
            "%s: fast path enabled on %s (bitsandbytes %s)",
            self._optimizer_display_name,
            device,
            BITSANDBYTES_VERSION,
        )
        self._adamw8bit_fast_path_logged = True

    def _log_stock_fallback_once(self, reason: str) -> None:
        if getattr(self, "_adamw8bit_stock_fallback_logged", False):
            return
        logger.warning(
            "%s: using stock %s step (reason: %s, bitsandbytes %s)",
            self._optimizer_display_name,
            self._stock_optimizer_display_name,
            reason,
            BITSANDBYTES_VERSION,
        )
        self._adamw8bit_stock_fallback_logged = True

    def _has_active_gradients(self) -> bool:
        return any(param.grad is not None for group in self.param_groups for param in group["params"])

    def _unsafe_active_override_reason(self) -> Optional[str]:
        # GlobalOptimManager stores only overridden parameters here. Keep the
        # common path O(1), and inspect this usually-empty mapping instead of
        # rebuilding the effective config for every LoRA parameter each step.
        parameter_overrides = self.mng.index2config
        if not parameter_overrides:
            return None

        for (group_index, param_index), override in parameter_overrides.items():
            if override.get("percentile_clipping", 100) < 100:
                reason = "parameter override enables percentile_clipping"
            elif override.get("max_unorm", 0.0) > 0.0:
                reason = "parameter override enables max_unorm"
            else:
                continue

            # Ignore entries whose indices do not exist in this optimizer.
            # GlobalOptimManager is a singleton and may retain such entries.
            if group_index >= len(self.param_groups):
                continue
            params = self.param_groups[group_index]["params"]
            if param_index >= len(params):
                continue
            if params[param_index].grad is not None:
                return reason

        return None

    def _fast_path_device(self) -> tuple[bool, Optional[torch.device], Optional[str]]:
        if self.is_paged:
            return False, None, "paged optimizer"
        if getattr(self.args, "percentile_clipping", 100) < 100:
            return False, None, "percentile_clipping is enabled"
        if getattr(self.args, "max_unorm", 0.0) > 0.0:
            return False, None, "max_unorm is enabled"
        override_reason = self._unsafe_active_override_reason()
        if override_reason is not None:
            return False, None, override_reason

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                return False, None, f"distributed world size {torch.distributed.get_world_size()}"

        active_device: Optional[torch.device] = None
        for group in self.param_groups:
            for param in group["params"]:
                grad = param.grad
                if grad is None:
                    continue
                if type(param) is not torch.nn.Parameter or type(grad) is not torch.Tensor:
                    return False, None, "parameter or gradient uses a Tensor subclass"
                if param.device.type != "cuda":
                    return False, None, f"parameter is on {param.device.type}"
                if grad.device != param.device:
                    return False, None, "gradient and parameter are on different devices"
                if param.layout != torch.strided or grad.layout != torch.strided or grad.is_sparse:
                    return False, None, "parameter or gradient is not dense strided"
                if active_device is None:
                    active_device = param.device
                elif param.device != active_device:
                    return False, None, "active gradients span multiple CUDA devices"

        return True, active_device, None

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None):
        # A closure may replace gradients, so let bitsandbytes retain complete
        # control of closure evaluation and route selection in that uncommon
        # case.
        if closure is not None:
            self._log_stock_fallback_once("optimizer closure")
            return super().step(closure)

        # Module-based GlobalOptimManager overrides are mapped to parameter
        # indices by check_overrides(). Apply them before deciding whether the
        # effective per-parameter clipping configuration is safe for fast mode.
        if not self.initialized:
            self.check_overrides()

        can_use_fast_path, device, fallback_reason = self._fast_path_device()
        if not can_use_fast_path:
            if self._has_active_gradients():
                self._log_stock_fallback_once(fallback_reason or "unsupported configuration")
            return super().step()

        if device is not None:
            self._log_fast_path_once(device)

        if not self.initialized:
            self.to_gpu()
            self.initialized = True

        updated = False
        for group_index, group in enumerate(self.param_groups):
            for param_index, param in enumerate(group["params"]):
                if param.grad is None:
                    continue

                state = self.state[param]
                if len(state) == 0:
                    self.init_state(group, param, group_index, param_index)

                self.prefetch_state(param)
                self.update_step(group, param, group_index, param_index)
                updated = True

        if updated:
            # Stock bitsandbytes synchronizes after every parameter. One final
            # device sync still surfaces asynchronous CUDA errors before step()
            # returns without introducing per-parameter CPU/GPU round trips.
            torch.cuda.synchronize(device=device)

        return None
