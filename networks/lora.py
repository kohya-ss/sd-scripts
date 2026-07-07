# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import numpy as np
import torch
import re
import library.maruo_global_config as maruoCfg
from library.rounding_util import (
    fake_quantize,
    compute_per_channel_step,
    fake_quantize_levels,
    compute_scale_bits,
    _reduce_dims_and_shape,
)
from library.utils import setup_logging
from library.sdxl_original_unet import SdxlUNet2DConditionModel

setup_logging()
import logging

logger = logging.getLogger(__name__)

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


def _fake_quantize_levels_with_q(
    x: torch.Tensor,
    *,
    scale: Union[float, torch.Tensor],
    qmin: int,
    qmax: int,
    mode: str,
):
    if not isinstance(scale, torch.Tensor):
        s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    else:
        s = scale.to(device=x.device, dtype=torch.float32)
    y = x.to(torch.float32) / s
    q_clamp = torch.clamp(y, qmin, qmax)
    if mode == "det":
        q = torch.round(q_clamp)
    elif mode == "stoch":
        frac = q_clamp - torch.floor(q_clamp)
        probs = frac.clamp(0.0, 1.0)
        q = torch.floor(q_clamp) + (torch.rand_like(probs) < probs).to(q_clamp.dtype)
    else:
        raise ValueError(f"unknown round mode: {mode}")
    q = torch.clamp(q, qmin, qmax)
    q_out = (q * s).to(x.dtype)
    return x + (q_out - x).detach(), q_clamp, s


class DQStatsAccumulator:
    def __init__(
        self,
        device,
        collect_full: bool,
        collect_zero: bool,
        collect_near_zero: bool,
        collect_error_parts: bool = False,
    ):
        self.device = device
        self.collect_full = collect_full
        self.collect_zero = collect_zero
        self.collect_near_zero = collect_near_zero
        self.collect_error_parts = collect_error_parts
        self.numel = torch.zeros(1, device=device, dtype=torch.float32)
        self.clip_count = torch.zeros(1, device=device, dtype=torch.float32)
        self.zero_count = torch.zeros(1, device=device, dtype=torch.float32) if collect_zero else None
        self.near_zero_count = torch.zeros(1, device=device, dtype=torch.float32) if collect_near_zero else None
        self.sumsq = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.absmax = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.scale_min = torch.full((1,), float("inf"), device=device, dtype=torch.float32) if collect_full else None
        self.scale_max = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.scale_sum = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.scale_count = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.xq_sumsq = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.xxq_sum = torch.zeros(1, device=device, dtype=torch.float32) if collect_full else None
        self.clip_err_sumsq = torch.zeros(1, device=device, dtype=torch.float32) if collect_error_parts else None
        self.round_err_sumsq = torch.zeros(1, device=device, dtype=torch.float32) if collect_error_parts else None

    def add(
        self,
        *,
        numel: torch.Tensor,
        clip_count: torch.Tensor,
        zero_count: Optional[torch.Tensor] = None,
        near_zero_count: Optional[torch.Tensor] = None,
        sumsq: Optional[torch.Tensor] = None,
        xq_sumsq: Optional[torch.Tensor] = None,
        xxq_sum: Optional[torch.Tensor] = None,
        absmax: Optional[torch.Tensor] = None,
        scale_min: Optional[torch.Tensor] = None,
        scale_max: Optional[torch.Tensor] = None,
        scale_sum: Optional[torch.Tensor] = None,
        scale_count: Optional[torch.Tensor] = None,
        clip_err_sumsq: Optional[torch.Tensor] = None,
        round_err_sumsq: Optional[torch.Tensor] = None,
    ):
        self.numel += numel
        self.clip_count += clip_count
        if self.collect_zero and zero_count is not None:
            self.zero_count += zero_count
        if self.collect_near_zero and near_zero_count is not None:
            self.near_zero_count += near_zero_count
        if self.collect_full:
            if sumsq is not None:
                self.sumsq += sumsq
            if xq_sumsq is not None:
                self.xq_sumsq += xq_sumsq
            if xxq_sum is not None:
                self.xxq_sum += xxq_sum
            if absmax is not None:
                self.absmax = torch.maximum(self.absmax, absmax)
            if scale_min is not None:
                self.scale_min = torch.minimum(self.scale_min, scale_min)
            if scale_max is not None:
                self.scale_max = torch.maximum(self.scale_max, scale_max)
            if scale_sum is not None:
                self.scale_sum += scale_sum
            if scale_count is not None:
                self.scale_count += scale_count
        if self.collect_error_parts:
            if clip_err_sumsq is not None:
                self.clip_err_sumsq += clip_err_sumsq
            if round_err_sumsq is not None:
                self.round_err_sumsq += round_err_sumsq


class DQStatsManager:
    def __init__(self):
        self.active = False
        self.step_idx = None
        self.collect_full = False
        self.collect_zero = False
        self.collect_near_zero = False
        self.collect_error_parts = False
        self.log_mode = "summary"
        self.log_scope = "both"
        self.auto_scope = "both"
        self.target = "delta"
        self.do_log = False
        self.do_auto = False
        self.accum = {}
        self.per_module = []

    def _reset(self, device):
        self.accum = {
            "unet": DQStatsAccumulator(
                device, self.collect_full, self.collect_zero, self.collect_near_zero, self.collect_error_parts
            ),
            "te": DQStatsAccumulator(
                device, self.collect_full, self.collect_zero, self.collect_near_zero, self.collect_error_parts
            ),
        }
        self.per_module = []

    def begin_step(
        self,
        *,
        step_idx: int,
        device,
        do_log: bool,
        do_auto: bool,
        collect_full: bool,
        collect_zero: bool,
        collect_near_zero: bool,
        collect_error_parts: bool,
        log_mode: str,
        log_scope: str,
        auto_scope: str,
        target: str,
    ):
        if not do_log and not do_auto:
            self.active = False
            self.step_idx = None
            return
        if (not self.active) or (self.step_idx != step_idx):
            self.collect_full = collect_full
            self.collect_zero = collect_zero
            self.collect_near_zero = collect_near_zero
            self.collect_error_parts = collect_error_parts
            self.log_mode = log_mode
            self.log_scope = log_scope
            self.auto_scope = auto_scope
            self.target = target
            self.do_log = do_log
            self.do_auto = do_auto
            self.step_idx = step_idx
            self.active = True
            self._reset(device)
        else:
            self.collect_full = collect_full
            self.collect_zero = collect_zero
            self.collect_near_zero = collect_near_zero
            self.collect_error_parts = collect_error_parts
            self.log_mode = log_mode
            self.log_scope = log_scope
            self.auto_scope = auto_scope
            self.target = target
            self.do_log = do_log
            self.do_auto = do_auto

    def discard_step(self, step_idx: int):
        if self.step_idx == step_idx:
            self.active = False
            self.step_idx = None

    def _scope_enabled(self, scope: str) -> bool:
        if not self.active:
            return False
        log_ok = self.do_log and (self.log_scope == "both" or self.log_scope == scope)
        auto_ok = self.do_auto and (self.auto_scope == "both" or self.auto_scope == scope)
        return log_ok or auto_ok

    def wants_scope(self, scope: str) -> bool:
        return self._scope_enabled(scope)

    def add_stats(
        self,
        *,
        scope: str,
        module_name: str,
        shape: str,
        numel: torch.Tensor,
        clip_count: torch.Tensor,
        zero_count: Optional[torch.Tensor],
        near_zero_count: Optional[torch.Tensor],
        sumsq: Optional[torch.Tensor],
        xq_sumsq: Optional[torch.Tensor],
        xxq_sum: Optional[torch.Tensor],
        absmax: Optional[torch.Tensor],
        scale_min: Optional[torch.Tensor],
        scale_max: Optional[torch.Tensor],
        scale_sum: Optional[torch.Tensor],
        scale_count: Optional[torch.Tensor],
        clip_err_sumsq: Optional[torch.Tensor],
        round_err_sumsq: Optional[torch.Tensor],
    ):
        if not self._scope_enabled(scope):
            return
        acc = self.accum[scope]
        acc.add(
            numel=numel,
            clip_count=clip_count,
            zero_count=zero_count,
            near_zero_count=near_zero_count,
            sumsq=sumsq,
            xq_sumsq=xq_sumsq,
            xxq_sum=xxq_sum,
            absmax=absmax,
            scale_min=scale_min,
            scale_max=scale_max,
            scale_sum=scale_sum,
            scale_count=scale_count,
            clip_err_sumsq=clip_err_sumsq,
            round_err_sumsq=round_err_sumsq,
        )
        if self.do_log and self.log_mode == "per_module" and (self.log_scope == "both" or self.log_scope == scope):
            self.per_module.append(
                {
                    "scope": scope,
                    "module": module_name,
                    "shape": shape,
                    "numel": numel.detach(),
                    "clip_count": clip_count.detach(),
                    "zero_count": zero_count.detach() if zero_count is not None else None,
                    "near_zero_count": near_zero_count.detach() if near_zero_count is not None else None,
                    "sumsq": sumsq.detach() if sumsq is not None else None,
                    "xq_sumsq": xq_sumsq.detach() if xq_sumsq is not None else None,
                    "xxq_sum": xxq_sum.detach() if xxq_sum is not None else None,
                    "absmax": absmax.detach() if absmax is not None else None,
                    "scale_min": scale_min.detach() if scale_min is not None else None,
                    "scale_max": scale_max.detach() if scale_max is not None else None,
                    "scale_sum": scale_sum.detach() if scale_sum is not None else None,
                    "scale_count": scale_count.detach() if scale_count is not None else None,
                    "clip_err_sumsq": clip_err_sumsq.detach() if clip_err_sumsq is not None else None,
                    "round_err_sumsq": round_err_sumsq.detach() if round_err_sumsq is not None else None,
                }
            )

    def export(self):
        if not self.active:
            return None
        return {
            "step_idx": self.step_idx,
            "do_log": self.do_log,
            "do_auto": self.do_auto,
            "log_mode": self.log_mode,
            "log_scope": self.log_scope,
            "auto_scope": self.auto_scope,
            "target": self.target,
            "collect_full": self.collect_full,
            "collect_zero": self.collect_zero,
            "collect_near_zero": self.collect_near_zero,
            "collect_error_parts": self.collect_error_parts,
            "accum": self.accum,
            "per_module": self.per_module,
        }


def _compute_lora_effective_rank_stats(lora, eps: float = 1e-12):
    w_down = getattr(lora, "lora_down", None)
    w_up = getattr(lora, "lora_up", None)
    if w_down is None or w_up is None:
        return None
    w_down = w_down.weight
    w_up = w_up.weight
    if w_down is None or w_up is None:
        return None

    with torch.no_grad():
        a = w_down.to(dtype=torch.float32)
        b = w_up.to(dtype=torch.float32)
        if a.dim() == 4:
            a = a.reshape(a.shape[0], -1)
        if b.dim() == 4:
            b = b.reshape(b.shape[0], b.shape[1])
        r = a.shape[0]
        if r <= 0:
            return None

        p = b.transpose(0, 1) @ b
        q = a @ a.transpose(0, 1)

        eye = torch.eye(r, device=q.device, dtype=q.dtype)
        diag_mean = torch.mean(torch.diag(q)) if r > 0 else torch.tensor(0.0, device=q.device, dtype=q.dtype)
        jitter = eps * (diag_mean.abs() + 1.0)
        l, info = torch.linalg.cholesky_ex(q + jitter * eye)
        if int(info.item()) != 0:
            eigvals, eigvecs = torch.linalg.eigh(q)
            eigvals = torch.clamp(eigvals, min=0.0)
            sqrt_q = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.transpose(0, 1)
            s = sqrt_q @ p @ sqrt_q
        else:
            s = l.transpose(0, 1) @ p @ l

        s = (s + s.transpose(0, 1)) * 0.5
        eigvals = torch.linalg.eigvalsh(s)
        eigvals = torch.clamp(eigvals, min=0.0)
        energy = eigvals.sum()
        energy_val = float(energy.item())
        scale = float(getattr(lora, "scale", 1.0))
        mult = float(getattr(lora, "multiplier", 1.0))
        energy_val *= (scale * mult) ** 2
        if energy_val <= eps:
            return {
                "module": lora.lora_name,
                "r": int(r),
                "sat": 0.0,
                "top1": 0.0,
                "energy": energy_val,
            }

        w = eigvals / (energy + eps)
        entropy = -(w * torch.log(w + eps)).sum()
        r_eff = torch.exp(entropy)
        sat = float(r_eff.item()) / float(r)
        top1 = float(torch.max(w).item())
        return {
            "module": lora.lora_name,
            "r": int(r),
            "sat": sat,
            "top1": top1,
            "energy": energy_val,
        }


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
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
        delta_q_step: Optional[float] = None,
        delta_q_mode: str = "det",
        delta_q_granularity: str = "tensor",  # 'tensor' or 'channel'
        delta_q_stat: str = "rms",  # 'rms'|'absmax'|'none'
        delta_q_bits: Optional[int] = None,
        delta_q_range_mul: float = 3.0,
        delta_q_ema_decay: float = 0.99,
        delta_q_on_z: bool = False,
        delta_q_use_triton: bool = False,
        delta_q_triton_scale_only: bool = False,
        delta_q_triton_div_rn: bool = False,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        # if limit_rank:
        #   self.lora_dim = min(lora_dim, in_dim, out_dim)
        #   if self.lora_dim != lora_dim:
        #     logger.info(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")
        # else:
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ == "Conv2d":
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
        else:
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        # delta fake quantization (applied to LoRA delta output only)
        self.delta_q_step = float(delta_q_step) if (delta_q_step is not None) else None
        self.delta_q_mode = delta_q_mode
        self.delta_q_enabled = True  # toggled by network if needed
        self.delta_q_granularity = delta_q_granularity
        self.delta_q_stat = delta_q_stat
        self.delta_q_bits = delta_q_bits
        self.delta_q_range_mul = delta_q_range_mul
        self.delta_q_ema_decay = delta_q_ema_decay
        # when True, quantize z=A(x) and then apply B: Delta' = B(Q(z))
        # otherwise quantize Delta directly: Delta' = Q(B(z))
        self.delta_q_on_z = bool(delta_q_on_z)
        self.delta_q_use_triton = bool(delta_q_use_triton)
        self.delta_q_triton_scale_only = bool(delta_q_triton_scale_only)
        self.delta_q_triton_div_rn = bool(delta_q_triton_div_rn)
        self.dq_stats_manager: Optional[DQStatsManager] = None
        self.dq_scope = "te" if lora_name.startswith("lora_te") else "unet"

    # no EMA buffers/statistics for delta quantization (ema_* removed)

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def _record_dq_stats(
        self,
        x_in: torch.Tensor,
        quantized: torch.Tensor,
        q_clamp: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        qmax: Optional[int],
    ):
        mgr = self.dq_stats_manager
        if mgr is None or not mgr.active or not mgr.wants_scope(self.dq_scope):
            return
        if mgr.target == "z" and not self.delta_q_on_z:
            return
        if mgr.target == "delta" and self.delta_q_on_z:
            return
        with torch.no_grad():
            device = x_in.device
            numel = torch.tensor(float(x_in.numel()), device=device, dtype=torch.float32)
            if q_clamp is not None and qmax is not None:
                clip_count = (q_clamp.abs() >= qmax).to(torch.float32).sum()
            else:
                clip_count = torch.zeros(1, device=device, dtype=torch.float32)
            zero_count = (quantized == 0).to(torch.float32).sum() if mgr.collect_zero else None
            near_zero_count = None
            if mgr.collect_near_zero and scale is not None:
                near_zero_count = (x_in.abs() < (0.5 * scale)).to(torch.float32).sum()
            sumsq = xq_sumsq = xxq_sum = absmax = None
            clip_err_sumsq = round_err_sumsq = None
            if mgr.collect_full:
                x_fp32 = x_in.to(torch.float32)
                x_flat = x_fp32.reshape(-1)
                sumsq = torch.dot(x_flat, x_flat)
                q_fp32 = quantized.to(torch.float32)
                q_flat = q_fp32.reshape(-1)
                xq_sumsq = torch.dot(q_flat, q_flat)
                xxq_sum = torch.dot(x_flat, q_flat)
                absmax = x_in.abs().max()
                if mgr.collect_error_parts and q_clamp is not None and scale is not None:
                    x_clamped = q_clamp.to(torch.float32) * scale.to(device=device, dtype=torch.float32)
                    clip_err = x_fp32 - x_clamped
                    round_err = x_clamped - q_fp32
                    clip_err_flat = clip_err.reshape(-1)
                    round_err_flat = round_err.reshape(-1)
                    clip_err_sumsq = torch.dot(clip_err_flat, clip_err_flat)
                    round_err_sumsq = torch.dot(round_err_flat, round_err_flat)
            scale_min = scale_max = scale_sum = scale_count = None
            if mgr.collect_full and scale is not None:
                scale_min = scale.min()
                scale_max = scale.max()
                scale_sum = scale.sum()
                scale_count = torch.tensor(float(scale.numel()), device=device, dtype=torch.float32)

            mgr.add_stats(
                scope=self.dq_scope,
                module_name=self.lora_name,
                shape=str(tuple(x_in.shape)),
                numel=numel,
                clip_count=clip_count,
                zero_count=zero_count,
                near_zero_count=near_zero_count,
                sumsq=sumsq,
                xq_sumsq=xq_sumsq,
                xxq_sum=xxq_sum,
                absmax=absmax,
                scale_min=scale_min,
                scale_max=scale_max,
                scale_sum=scale_sum,
                scale_count=scale_count,
                clip_err_sumsq=clip_err_sumsq,
                round_err_sumsq=round_err_sumsq,
            )

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
            if len(lx.size()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.size()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        # Optionally apply fake quantization to z before up-projection
        if self.training and self.delta_q_enabled and self.delta_q_on_z:
            if self.delta_q_bits is not None and self.delta_q_bits > 0:
                qmax = (1 << (self.delta_q_bits - 1)) - 1
                x_in = lx
                if self.dq_stats_manager is not None and self.dq_stats_manager.wants_scope(self.dq_scope) and self.dq_stats_manager.active:
                    with torch.no_grad():
                        z_scale = compute_scale_bits(
                            lx,
                            bits=self.delta_q_bits,
                            granularity=self.delta_q_granularity,
                            stat=(self.delta_q_stat if self.delta_q_stat != "none" else "rms"),
                            range_mul=self.delta_q_range_mul,
                            use_triton=self.delta_q_use_triton,
                        )
                    lx, q_clamp, scale_t = _fake_quantize_levels_with_q(
                        x_in, scale=z_scale, qmin=-qmax, qmax=qmax, mode=self.delta_q_mode
                    )
                    self._record_dq_stats(x_in, lx, q_clamp, scale_t, qmax)
                else:
                    with torch.no_grad():
                        z_scale = compute_scale_bits(
                            lx,
                            bits=self.delta_q_bits,
                            granularity=self.delta_q_granularity,
                            stat=(self.delta_q_stat if self.delta_q_stat != "none" else "rms"),
                            range_mul=self.delta_q_range_mul,
                            use_triton=self.delta_q_use_triton,
                        )
                    lx = fake_quantize_levels(
                        lx,
                        scale=z_scale,
                        qmin=-qmax,
                        qmax=qmax,
                        mode=self.delta_q_mode,
                        use_triton=self.delta_q_use_triton and not self.delta_q_triton_scale_only,
                        triton_div_rn=self.delta_q_triton_div_rn,
                    )
            elif self.delta_q_step is not None and self.delta_q_step > 0:
                if self.delta_q_granularity == "channel":
                    with torch.no_grad():
                        step_t = compute_per_channel_step(lx, self.delta_q_step, stat=self.delta_q_stat)
                else:
                    step_t = self.delta_q_step
                x_in = lx
                lx = fake_quantize(lx, step=step_t, mode=self.delta_q_mode)
                if self.dq_stats_manager is not None and self.dq_stats_manager.wants_scope(self.dq_scope) and self.dq_stats_manager.active:
                    self._record_dq_stats(x_in, lx, None, None, None)
            # ensure memory contiguity for faster lora_up (matmul/conv)
            lx = lx.contiguous()

        lx = self.lora_up(lx)

        delta = lx * self.multiplier * scale
        # Apply fake quantization to delta only when on_z is False
        # EMA-based stats were removed to simplify and speed up training

        if self.training and self.delta_q_enabled and not self.delta_q_on_z:
            if self.delta_q_bits is not None and self.delta_q_bits > 0:
                # bits mode: compute scale per setting (tensor or per-channel)
                qmax = (1 << (self.delta_q_bits - 1)) - 1
                x_in = delta
                if self.dq_stats_manager is not None and self.dq_stats_manager.wants_scope(self.dq_scope) and self.dq_stats_manager.active:
                    with torch.no_grad():
                        d_scale = compute_scale_bits(
                            delta,
                            bits=self.delta_q_bits,
                            granularity=self.delta_q_granularity,
                            stat=(self.delta_q_stat if self.delta_q_stat != "none" else "rms"),
                            range_mul=self.delta_q_range_mul,
                            use_triton=self.delta_q_use_triton,
                        )
                    delta, q_clamp, scale_t = _fake_quantize_levels_with_q(
                        x_in, scale=d_scale, qmin=-qmax, qmax=qmax, mode=self.delta_q_mode
                    )
                    self._record_dq_stats(x_in, delta, q_clamp, scale_t, qmax)
                else:
                    with torch.no_grad():
                        d_scale = compute_scale_bits(
                            delta,
                            bits=self.delta_q_bits,
                            granularity=self.delta_q_granularity,
                            stat=(self.delta_q_stat if self.delta_q_stat != "none" else "rms"),
                            range_mul=self.delta_q_range_mul,
                            use_triton=self.delta_q_use_triton,
                        )
                    delta = fake_quantize_levels(
                        delta,
                        scale=d_scale,
                        qmin=-qmax,
                        qmax=qmax,
                        mode=self.delta_q_mode,
                        use_triton=self.delta_q_use_triton and not self.delta_q_triton_scale_only,
                        triton_div_rn=self.delta_q_triton_div_rn,
                    )
            elif self.delta_q_step is not None and self.delta_q_step > 0:
                if self.delta_q_granularity == "channel":
                    with torch.no_grad():
                        step_t = compute_per_channel_step(delta, self.delta_q_step, stat=self.delta_q_stat)
                else:
                    step_t = self.delta_q_step
                x_in = delta
                delta = fake_quantize(delta, step=step_t, mode=self.delta_q_mode)
                if self.dq_stats_manager is not None and self.dq_stats_manager.wants_scope(self.dq_scope) and self.dq_stats_manager.active:
                    self._record_dq_stats(x_in, delta, None, None, None)

        return org_forwarded + delta


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # 後から参照できるように
        self.enabled = True

        # check regional or not by lora_name
        self.text_encoder = False
        if lora_name.startswith("lora_te_"):
            self.regional = False
            self.use_sub_prompt = True
            self.text_encoder = True
        elif "attn2_to_k" in lora_name or "attn2_to_v" in lora_name:
            self.regional = False
            self.use_sub_prompt = True
        elif "time_emb" in lora_name:
            self.regional = False
            self.use_sub_prompt = False
        else:
            self.regional = True
            self.use_sub_prompt = False

        self.network: LoRANetwork = None

    def set_network(self, network):
        self.network = network

    # freezeしてマージする
    def merge_to(self, sd, dtype, device):
        # get up/down weight
        up_weight = sd["lora_up.weight"].to(torch.float).to(device)
        down_weight = sd["lora_down.weight"].to(torch.float).to(device)

        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"].to(torch.float)

        # merge weight
        if len(weight.size()) == 2:
            # linear
            weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                weight
                + self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            # logger.info(conved.size(), weight.size(), module.stride, module.padding)
            weight = weight + self.multiplier * conved * self.scale

        # set weight to org_module
        org_sd["weight"] = weight.to(dtype)
        self.org_module.load_state_dict(org_sd)

    # 復元できるマージのため、このモジュールのweightを返す
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        return self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)

        if self.network is None or self.network.sub_prompt_index is None:
            return self.default_forward(x)
        if not self.regional and not self.use_sub_prompt:
            return self.default_forward(x)

        if self.regional:
            return self.regional_forward(x)
        else:
            return self.sub_prompt_forward(x)

    def get_mask_for_x(self, x):
        # calculate size from shape of x
        if len(x.size()) == 4:
            h, w = x.size()[2:4]
            area = h * w
        else:
            area = x.size()[1]

        mask = self.network.mask_dic.get(area, None)
        if mask is None or len(x.size()) == 2:
            # emb_layers in SDXL doesn't have mask
            # if "emb" not in self.lora_name:
            #     print(f"mask is None for resolution {self.lora_name}, {area}, {x.size()}")
            mask_size = (1, x.size()[1]) if len(x.size()) == 2 else (1, *x.size()[1:-1], 1)
            return torch.ones(mask_size, dtype=x.dtype, device=x.device) / self.network.num_sub_prompts
        if len(x.size()) == 3:
            mask = torch.reshape(mask, (1, -1, 1))
        return mask

    def regional_forward(self, x):
        if "attn2_to_out" in self.lora_name:
            return self.to_out_forward(x)

        if self.network.mask_dic is None:  # sub_prompt_index >= 3
            return self.default_forward(x)

        # apply mask for LoRA result
        lx = self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        mask = self.get_mask_for_x(lx)
        # print("regional", self.lora_name, self.network.sub_prompt_index, lx.size(), mask.size())
        # if mask.ndim > lx.ndim:  # in some resolution, lx is 2d and mask is 3d (the reason is not checked)
        #     mask = mask.squeeze(-1)
        lx = lx * mask

        x = self.org_forward(x)
        x = x + lx

        if "attn2_to_q" in self.lora_name and self.network.is_last_network:
            x = self.postp_to_q(x)

        return x

    def postp_to_q(self, x):
        # repeat x to num_sub_prompts
        has_real_uncond = x.size()[0] // self.network.batch_size == 3
        qc = self.network.batch_size  # uncond
        qc += self.network.batch_size * self.network.num_sub_prompts  # cond
        if has_real_uncond:
            qc += self.network.batch_size  # real_uncond

        query = torch.zeros((qc, x.size()[1], x.size()[2]), device=x.device, dtype=x.dtype)
        query[: self.network.batch_size] = x[: self.network.batch_size]

        for i in range(self.network.batch_size):
            qi = self.network.batch_size + i * self.network.num_sub_prompts
            query[qi : qi + self.network.num_sub_prompts] = x[self.network.batch_size + i]

        if has_real_uncond:
            query[-self.network.batch_size :] = x[-self.network.batch_size :]

        # logger.info(f"postp_to_q {self.lora_name} {x.size()} {query.size()} {self.network.num_sub_prompts}")
        return query

    def sub_prompt_forward(self, x):
        if x.size()[0] == self.network.batch_size:  # if uncond in text_encoder, do not apply LoRA
            return self.org_forward(x)

        emb_idx = self.network.sub_prompt_index
        if not self.text_encoder:
            emb_idx += self.network.batch_size

        # apply sub prompt of X
        lx = x[emb_idx :: self.network.num_sub_prompts]
        lx = self.lora_up(self.lora_down(lx)) * self.multiplier * self.scale

        # logger.info(f"sub_prompt_forward {self.lora_name} {x.size()} {lx.size()} {emb_idx}")

        x = self.org_forward(x)
        x[emb_idx :: self.network.num_sub_prompts] += lx

        return x

    def to_out_forward(self, x):
        # logger.info(f"to_out_forward {self.lora_name} {x.size()} {self.network.is_last_network}")

        if self.network.is_last_network:
            masks = [None] * self.network.num_sub_prompts
            self.network.shared[self.lora_name] = (None, masks)
        else:
            lx, masks = self.network.shared[self.lora_name]

        # call own LoRA
        x1 = x[self.network.batch_size + self.network.sub_prompt_index :: self.network.num_sub_prompts]
        lx1 = self.lora_up(self.lora_down(x1)) * self.multiplier * self.scale

        if self.network.is_last_network:
            lx = torch.zeros(
                (self.network.num_sub_prompts * self.network.batch_size, *lx1.size()[1:]), device=lx1.device, dtype=lx1.dtype
            )
            self.network.shared[self.lora_name] = (lx, masks)

        # logger.info(f"to_out_forward {lx.size()} {lx1.size()} {self.network.sub_prompt_index} {self.network.num_sub_prompts}")
        lx[self.network.sub_prompt_index :: self.network.num_sub_prompts] += lx1
        masks[self.network.sub_prompt_index] = self.get_mask_for_x(lx1)

        # if not last network, return x and masks
        x = self.org_forward(x)
        if not self.network.is_last_network:
            return x

        lx, masks = self.network.shared.pop(self.lora_name)

        # if last network, combine separated x with mask weighted sum
        has_real_uncond = x.size()[0] // self.network.batch_size == self.network.num_sub_prompts + 2

        out = torch.zeros((self.network.batch_size * (3 if has_real_uncond else 2), *x.size()[1:]), device=x.device, dtype=x.dtype)
        out[: self.network.batch_size] = x[: self.network.batch_size]  # uncond
        if has_real_uncond:
            out[-self.network.batch_size :] = x[-self.network.batch_size :]  # real_uncond

        # logger.info(f"to_out_forward {self.lora_name} {self.network.sub_prompt_index} {self.network.num_sub_prompts}")
        # if num_sub_prompts > num of LoRAs, fill with zero
        for i in range(len(masks)):
            if masks[i] is None:
                masks[i] = torch.zeros_like(masks[0])

        mask = torch.cat(masks)
        mask_sum = torch.sum(mask, dim=0) + 1e-4
        for i in range(self.network.batch_size):
            # 1枚の画像ごとに処理する
            lx1 = lx[i * self.network.num_sub_prompts : (i + 1) * self.network.num_sub_prompts]
            lx1 = lx1 * mask
            lx1 = torch.sum(lx1, dim=0)

            xi = self.network.batch_size + i * self.network.num_sub_prompts
            x1 = x[xi : xi + self.network.num_sub_prompts]
            x1 = x1 * mask
            x1 = torch.sum(x1, dim=0)
            x1 = x1 / mask_sum

            x1 = x1 + lx1
            out[self.network.batch_size + i] = x1

        # logger.info(f"to_out_forward {x.size()} {out.size()} {has_real_uncond}")
        return out


def parse_block_lr_kwargs(is_sdxl: bool, nw_kwargs: Dict) -> Optional[List[float]]:
    down_lr_weight = nw_kwargs.get("down_lr_weight", None)
    mid_lr_weight = nw_kwargs.get("mid_lr_weight", None)
    up_lr_weight = nw_kwargs.get("up_lr_weight", None)

    # 以上のいずれにも設定がない場合は無効としてNoneを返す
    if down_lr_weight is None and mid_lr_weight is None and up_lr_weight is None:
        return None

    # extract learning rate weight for each block
    if down_lr_weight is not None:
        # if some parameters are not set, use zero
        if "," in down_lr_weight:
            down_lr_weight = [(float(s) if s else 0.0) for s in down_lr_weight.split(",")]

    if mid_lr_weight is not None:
        mid_lr_weight = [(float(s) if s else 0.0) for s in mid_lr_weight.split(",")]

    if up_lr_weight is not None:
        if "," in up_lr_weight:
            up_lr_weight = [(float(s) if s else 0.0) for s in up_lr_weight.split(",")]

    return get_block_lr_weight(
        is_sdxl, down_lr_weight, mid_lr_weight, up_lr_weight, float(nw_kwargs.get("block_lr_zero_threshold", 0.0))
    )


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # if unet is an instance of SdxlUNet2DConditionModel or subclass, set is_sdxl to True
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # block dim/alpha/lr
    block_dims = kwargs.get("block_dims", None)
    block_lr_weight = parse_block_lr_kwargs(is_sdxl, kwargs)

    # 以上のいずれかに指定があればblockごとのdim(rank)を有効にする
    if block_dims is not None or block_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
        )

        # remove block dim/alpha without learning rate
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = remove_block_dims_and_alphas(
            is_sdxl, block_dims, block_alphas, conv_block_dims, conv_block_alphas, block_lr_weight
        )

    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        varbose=True,
        is_sdxl=is_sdxl,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network


# このメソッドは外部から呼び出される可能性を考慮しておく
# network_dim, network_alpha にはデフォルト値が入っている。
# block_dims, block_alphas は両方ともNoneまたは両方とも値が入っている
# conv_dim, conv_alpha は両方ともNoneまたは両方とも値が入っている
def get_block_dims_and_alphas(
    is_sdxl, block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
):
    if not is_sdxl:
        num_total_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + LoRANetwork.NUM_OF_MID_BLOCKS
    else:
        # 1+9+3+9+1=23, no LoRA for emb_layers (0)
        num_total_blocks = 1 + LoRANetwork.SDXL_NUM_OF_BLOCKS * 2 + LoRANetwork.SDXL_NUM_OF_MID_BLOCKS + 1

    def parse_ints(s):
        return [int(i) for i in s.split(",")]

    def parse_floats(s):
        return [float(i) for i in s.split(",")]

    # block_dimsとblock_alphasをパースする。必ず値が入る
    if block_dims is not None:
        block_dims = parse_ints(block_dims)
        assert len(block_dims) == num_total_blocks, (
            f"block_dims must have {num_total_blocks} elements but {len(block_dims)} elements are given"
            + f" / block_dimsは{num_total_blocks}個指定してください（指定された個数: {len(block_dims)}）"
        )
    else:
        logger.warning(
            f"block_dims is not specified. all dims are set to {network_dim} / block_dimsが指定されていません。すべてのdimは{network_dim}になります"
        )
        block_dims = [network_dim] * num_total_blocks

    if block_alphas is not None:
        block_alphas = parse_floats(block_alphas)
        assert (
            len(block_alphas) == num_total_blocks
        ), f"block_alphas must have {num_total_blocks} elements / block_alphasは{num_total_blocks}個指定してください"
    else:
        logger.warning(
            f"block_alphas is not specified. all alphas are set to {network_alpha} / block_alphasが指定されていません。すべてのalphaは{network_alpha}になります"
        )
        block_alphas = [network_alpha] * num_total_blocks

    # conv_block_dimsとconv_block_alphasを、指定がある場合のみパースする。指定がなければconv_dimとconv_alphaを使う
    if conv_block_dims is not None:
        conv_block_dims = parse_ints(conv_block_dims)
        assert (
            len(conv_block_dims) == num_total_blocks
        ), f"conv_block_dims must have {num_total_blocks} elements / conv_block_dimsは{num_total_blocks}個指定してください"

        if conv_block_alphas is not None:
            conv_block_alphas = parse_floats(conv_block_alphas)
            assert (
                len(conv_block_alphas) == num_total_blocks
            ), f"conv_block_alphas must have {num_total_blocks} elements / conv_block_alphasは{num_total_blocks}個指定してください"
        else:
            if conv_alpha is None:
                conv_alpha = 1.0
            logger.warning(
                f"conv_block_alphas is not specified. all alphas are set to {conv_alpha} / conv_block_alphasが指定されていません。すべてのalphaは{conv_alpha}になります"
            )
            conv_block_alphas = [conv_alpha] * num_total_blocks
    else:
        if conv_dim is not None:
            logger.warning(
                f"conv_dim/alpha for all blocks are set to {conv_dim} and {conv_alpha} / すべてのブロックのconv_dimとalphaは{conv_dim}および{conv_alpha}になります"
            )
            conv_block_dims = [conv_dim] * num_total_blocks
            conv_block_alphas = [conv_alpha] * num_total_blocks
        else:
            conv_block_dims = None
            conv_block_alphas = None

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 層別学習率用に層ごとの学習率に対する倍率を定義する、外部から呼び出せるようにclass外に出しておく
# 戻り値は block ごとの倍率のリスト
def get_block_lr_weight(
    is_sdxl,
    down_lr_weight: Union[str, List[float]],
    mid_lr_weight: List[float],
    up_lr_weight: Union[str, List[float]],
    zero_threshold: float,
) -> Optional[List[float]]:
    # パラメータ未指定時は何もせず、今までと同じ動作とする
    if up_lr_weight is None and mid_lr_weight is None and down_lr_weight is None:
        return None

    if not is_sdxl:
        max_len_for_down_or_up = LoRANetwork.NUM_OF_BLOCKS
        max_len_for_mid = LoRANetwork.NUM_OF_MID_BLOCKS
    else:
        max_len_for_down_or_up = LoRANetwork.SDXL_NUM_OF_BLOCKS
        max_len_for_mid = LoRANetwork.SDXL_NUM_OF_MID_BLOCKS

    def get_list(name_with_suffix) -> List[float]:
        import math

        tokens = name_with_suffix.split("+")
        name = tokens[0]
        base_lr = float(tokens[1]) if len(tokens) > 1 else 0.0

        if name == "cosine":
            return [
                math.sin(math.pi * (i / (max_len_for_down_or_up - 1)) / 2) + base_lr
                for i in reversed(range(max_len_for_down_or_up))
            ]
        elif name == "sine":
            return [math.sin(math.pi * (i / (max_len_for_down_or_up - 1)) / 2) + base_lr for i in range(max_len_for_down_or_up)]
        elif name == "linear":
            return [i / (max_len_for_down_or_up - 1) + base_lr for i in range(max_len_for_down_or_up)]
        elif name == "reverse_linear":
            return [i / (max_len_for_down_or_up - 1) + base_lr for i in reversed(range(max_len_for_down_or_up))]
        elif name == "zeros":
            return [0.0 + base_lr] * max_len_for_down_or_up
        else:
            logger.error(
                "Unknown lr_weight argument %s is used. Valid arguments:  / 不明なlr_weightの引数 %s が使われました。有効な引数:\n\tcosine, sine, linear, reverse_linear, zeros"
                % (name)
            )
            return None

    if type(down_lr_weight) == str:
        down_lr_weight = get_list(down_lr_weight)
    if type(up_lr_weight) == str:
        up_lr_weight = get_list(up_lr_weight)

    if (up_lr_weight != None and len(up_lr_weight) > max_len_for_down_or_up) or (
        down_lr_weight != None and len(down_lr_weight) > max_len_for_down_or_up
    ):
        logger.warning("down_weight or up_weight is too long. Parameters after %d-th are ignored." % max_len_for_down_or_up)
        logger.warning("down_weightもしくはup_weightが長すぎます。%d個目以降のパラメータは無視されます。" % max_len_for_down_or_up)
        up_lr_weight = up_lr_weight[:max_len_for_down_or_up]
        down_lr_weight = down_lr_weight[:max_len_for_down_or_up]

    if mid_lr_weight != None and len(mid_lr_weight) > max_len_for_mid:
        logger.warning("mid_weight is too long. Parameters after %d-th are ignored." % max_len_for_mid)
        logger.warning("mid_weightが長すぎます。%d個目以降のパラメータは無視されます。" % max_len_for_mid)
        mid_lr_weight = mid_lr_weight[:max_len_for_mid]

    if (up_lr_weight != None and len(up_lr_weight) < max_len_for_down_or_up) or (
        down_lr_weight != None and len(down_lr_weight) < max_len_for_down_or_up
    ):
        logger.warning("down_weight or up_weight is too short. Parameters after %d-th are filled with 1." % max_len_for_down_or_up)
        logger.warning(
            "down_weightもしくはup_weightが短すぎます。%d個目までの不足したパラメータは1で補われます。" % max_len_for_down_or_up
        )

        if down_lr_weight != None and len(down_lr_weight) < max_len_for_down_or_up:
            down_lr_weight = down_lr_weight + [1.0] * (max_len_for_down_or_up - len(down_lr_weight))
        if up_lr_weight != None and len(up_lr_weight) < max_len_for_down_or_up:
            up_lr_weight = up_lr_weight + [1.0] * (max_len_for_down_or_up - len(up_lr_weight))

    if mid_lr_weight != None and len(mid_lr_weight) < max_len_for_mid:
        logger.warning("mid_weight is too short. Parameters after %d-th are filled with 1." % max_len_for_mid)
        logger.warning("mid_weightが短すぎます。%d個目までの不足したパラメータは1で補われます。" % max_len_for_mid)
        mid_lr_weight = mid_lr_weight + [1.0] * (max_len_for_mid - len(mid_lr_weight))

    if (up_lr_weight != None) or (mid_lr_weight != None) or (down_lr_weight != None):
        logger.info("apply block learning rate / 階層別学習率を適用します。")
        if down_lr_weight != None:
            down_lr_weight = [w if w > zero_threshold else 0 for w in down_lr_weight]
            logger.info(f"down_lr_weight (shallower -> deeper, 浅い層->深い層): {down_lr_weight}")
        else:
            down_lr_weight = [1.0] * max_len_for_down_or_up
            logger.info("down_lr_weight: all 1.0, すべて1.0")

        if mid_lr_weight != None:
            mid_lr_weight = [w if w > zero_threshold else 0 for w in mid_lr_weight]
            logger.info(f"mid_lr_weight: {mid_lr_weight}")
        else:
            mid_lr_weight = [1.0] * max_len_for_mid
            logger.info("mid_lr_weight: all 1.0, すべて1.0")

        if up_lr_weight != None:
            up_lr_weight = [w if w > zero_threshold else 0 for w in up_lr_weight]
            logger.info(f"up_lr_weight (deeper -> shallower, 深い層->浅い層): {up_lr_weight}")
        else:
            up_lr_weight = [1.0] * max_len_for_down_or_up
            logger.info("up_lr_weight: all 1.0, すべて1.0")

    lr_weight = down_lr_weight + mid_lr_weight + up_lr_weight

    if is_sdxl:
        lr_weight = [1.0] + lr_weight + [1.0]  # add 1.0 for emb_layers and out

    assert (not is_sdxl and len(lr_weight) == LoRANetwork.NUM_OF_BLOCKS * 2 + LoRANetwork.NUM_OF_MID_BLOCKS) or (
        is_sdxl and len(lr_weight) == 1 + LoRANetwork.SDXL_NUM_OF_BLOCKS * 2 + LoRANetwork.SDXL_NUM_OF_MID_BLOCKS + 1
    ), f"lr_weight length is invalid: {len(lr_weight)}"

    return lr_weight


# lr_weightが0のblockをblock_dimsから除外する、外部から呼び出す可能性を考慮しておく
def remove_block_dims_and_alphas(
    is_sdxl, block_dims, block_alphas, conv_block_dims, conv_block_alphas, block_lr_weight: Optional[List[float]]
):
    if block_lr_weight is not None:
        for i, lr in enumerate(block_lr_weight):
            if lr == 0:
                block_dims[i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[i] = 0
    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 外部から呼び出す可能性を考慮しておく
def get_block_index(lora_name: str, is_sdxl: bool = False) -> int:
    block_idx = -1  # invalid lora name
    if not is_sdxl:
        m = RE_UPDOWN.search(lora_name)
        if m:
            g = m.groups()
            i = int(g[1])
            j = int(g[3])
            if g[2] == "resnets":
                idx = 3 * i + j
            elif g[2] == "attentions":
                idx = 3 * i + j
            elif g[2] == "upsamplers" or g[2] == "downsamplers":
                idx = 3 * i + 2

            if g[0] == "down":
                block_idx = 1 + idx  # 0に該当するLoRAは存在しない
            elif g[0] == "up":
                block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx
        elif "mid_block_" in lora_name:
            block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12
    else:
        # copy from sdxl_train
        if lora_name.startswith("lora_unet_"):
            name = lora_name[len("lora_unet_") :]
            if name.startswith("time_embed_") or name.startswith("label_emb_"):  # No LoRA
                block_idx = 0  # 0
            elif name.startswith("input_blocks_"):  # 1-9
                block_idx = 1 + int(name.split("_")[2])
            elif name.startswith("middle_block_"):  # 10-12
                block_idx = 10 + int(name.split("_")[2])
            elif name.startswith("output_blocks_"):  # 13-21
                block_idx = 13 + int(name.split("_")[2])
            elif name.startswith("out_"):  # 22, out, no LoRA
                block_idx = 22

    return block_idx


def convert_diffusers_to_sai_if_needed(weights_sd):
    # only supports U-Net LoRA modules

    found_up_down_blocks = False
    for k in list(weights_sd.keys()):
        if "down_blocks" in k:
            found_up_down_blocks = True
            break
        if "up_blocks" in k:
            found_up_down_blocks = True
            break
    if not found_up_down_blocks:
        return

    from library.sdxl_model_util import make_unet_conversion_map

    unet_conversion_map = make_unet_conversion_map()
    unet_conversion_map = {hf.replace(".", "_")[:-1]: sd.replace(".", "_")[:-1] for sd, hf in unet_conversion_map}

    # # add extra conversion
    # unet_conversion_map["up_blocks_1_upsamplers_0"] = "lora_unet_output_blocks_2_2_conv"

    logger.info(f"Converting LoRA keys from Diffusers to SAI")
    lora_unet_prefix = "lora_unet_"
    for k in list(weights_sd.keys()):
        if not k.startswith(lora_unet_prefix):
            continue

        unet_module_name = k[len(lora_unet_prefix) :].split(".")[0]

        # search for conversion: this is slow because the algorithm is O(n^2), but the number of keys is small
        for hf_module_name, sd_module_name in unet_conversion_map.items():
            if hf_module_name in unet_module_name:
                new_key = (
                    lora_unet_prefix
                    + unet_module_name.replace(hf_module_name, sd_module_name)
                    + k[len(lora_unet_prefix) + len(unet_module_name) :]
                )
                weights_sd[new_key] = weights_sd.pop(k)
                found = True
                break

        if not found:
            logger.warning(f"Key {k} is not found in unet_conversion_map")


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, vae, text_encoder, unet, weights_sd=None, for_inference=False, **kwargs):
    # if unet is an instance of SdxlUNet2DConditionModel or subclass, set is_sdxl to True
    is_sdxl = unet is not None and issubclass(unet.__class__, SdxlUNet2DConditionModel)

    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # if keys are Diffusers based, convert to SAI based
    if is_sdxl:
        convert_diffusers_to_sai_if_needed(weights_sd)

    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)

    # support old LoRA without alpha
    for key in modules_dim.keys():
        if key not in modules_alpha:
            modules_alpha[key] = modules_dim[key]

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        is_sdxl=is_sdxl,
    )

    # block lr
    block_lr_weight = parse_block_lr_kwargs(is_sdxl, kwargs)
    if block_lr_weight is not None:
        network.set_block_lr_weight(block_lr_weight)

    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数
    NUM_OF_MID_BLOCKS = 1
    SDXL_NUM_OF_BLOCKS = 9  # SDXLのモデルでのinput/outputの層の数 total=1(base) 9(input) + 3(mid) + 9(output) + 1(out) = 23
    SDXL_NUM_OF_MID_BLOCKS = 3

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    if maruoCfg.te_mlp_fc_only:
        # 改造ルート
        TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]  # 昔のバージョンの状態と同じにしたい場合用(実験用)
    else:
        # 通常ルート
        TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPSdpaAttention", "CLIPMLP"]

    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        block_dims: Optional[List[int]] = None,
        block_alphas: Optional[List[float]] = None,
        conv_block_dims: Optional[List[int]] = None,
        conv_block_alphas: Optional[List[float]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        module_class: Type[object] = LoRAModule,
        varbose: Optional[bool] = False,
        is_sdxl: Optional[bool] = False,
        delta_q_step: Optional[float] = None,
        delta_q_mode: str = "det",
        delta_q_granularity: str = "tensor",
        delta_q_stat: str = "rms",
        delta_q_bits: Optional[int] = None,
        delta_q_range_mul: float = 3.0,
        delta_q_on_z: bool = False,
        delta_q_use_triton: bool = False,
        delta_q_triton_scale_only: bool = False,
        delta_q_triton_div_rn: bool = False,
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        # config for delta fake quantization (propagated to modules)
        self.delta_q_step = delta_q_step
        self.delta_q_mode = delta_q_mode
        self.delta_q_granularity = delta_q_granularity
        self.delta_q_stat = delta_q_stat
        self.delta_q_bits = delta_q_bits
        self.delta_q_range_mul = delta_q_range_mul
        self.delta_q_on_z = bool(delta_q_on_z)
        self.delta_q_use_triton = bool(delta_q_use_triton)
        self.delta_q_triton_scale_only = bool(delta_q_triton_scale_only)
        self.delta_q_triton_div_rn = bool(delta_q_triton_div_rn)
        self.dq_stats_manager = DQStatsManager()

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
        elif block_dims is not None:
            logger.info(f"create LoRA network from block_dims")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            logger.info(f"block_dims: {block_dims}")
            logger.info(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                logger.info(f"conv_block_dims: {conv_block_dims}")
                logger.info(f"conv_block_alphas: {conv_block_alphas}")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            if self.conv_lora_dim is not None:
                logger.info(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
                )

        # create module instances
        def create_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],  # None, 1, 2
            root_module: torch.nn.Module,
            target_replace_modules: List[torch.nn.Module],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if module.__class__.__name__ in target_replace_modules:
                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        if is_linear or is_conv2d:
                            lora_name = prefix + "." + name + "." + child_name
                            lora_name = lora_name.replace(".", "_")

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha[lora_name]
                            elif is_unet and block_dims is not None:
                                # U-Netでblock_dims指定あり
                                block_idx = get_block_index(lora_name, is_sdxl)
                                if is_linear or is_conv2d_1x1:
                                    dim = block_dims[block_idx]
                                    alpha = block_alphas[block_idx]
                                elif conv_block_dims is not None:
                                    dim = conv_block_dims[block_idx]
                                    alpha = conv_block_alphas[block_idx]
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None or conv_block_dims is not None):
                                    skipped.append(lora_name)
                                continue

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                delta_q_step=self.delta_q_step,
                                delta_q_mode=self.delta_q_mode,
                                delta_q_granularity=self.delta_q_granularity,
                                delta_q_stat=self.delta_q_stat,
                                delta_q_bits=self.delta_q_bits,
                                delta_q_range_mul=self.delta_q_range_mul,
                                delta_q_on_z=self.delta_q_on_z,
                                delta_q_use_triton=self.delta_q_use_triton,
                                delta_q_triton_scale_only=self.delta_q_triton_scale_only,
                                delta_q_triton_div_rn=self.delta_q_triton_div_rn,
                            )
                            lora.dq_stats_manager = self.dq_stats_manager
                            loras.append(lora)
            return loras, skipped

        text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        skipped_te = []
        self._text_encoder_loras_by_encoder: List[List[LoRAModule]] = []
        for i, text_encoder in enumerate(text_encoders):
            if len(text_encoders) > 1:
                index = i + 1
                logger.info(f"create LoRA for Text Encoder {index}:")
            else:
                index = None
                logger.info(f"create LoRA for Text Encoder:")

            text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
            self._text_encoder_loras_by_encoder.append(text_encoder_loras)
            skipped_te += skipped

        self._has_multiple_text_encoders = len(self._text_encoder_loras_by_encoder) > 1
        self._active_text_encoder_indices = [
            idx for idx, group in enumerate(self._text_encoder_loras_by_encoder) if len(group) > 0
        ]
        self.text_encoder_loras = [
            lora
            for idx, group in enumerate(self._text_encoder_loras_by_encoder)
            if idx in self._active_text_encoder_indices
            for lora in group
        ]
        logger.info(f"create LoRA for Text Encoder: {len(self.text_encoder_loras)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        self.unet_loras, skipped_un = create_modules(True, None, unet, target_modules)
        logger.info(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            logger.warning(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        self.block_lr_weight = None
        self.block_lr = False

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    # runtime control for delta fake-quant (enable/disable)
    def set_delta_fake_quant(
        self,
        step: Optional[float],
        mode: str = "det",
        granularity: Optional[str] = None,
        stat: Optional[str] = None,
        bits: Optional[int] = None,
        range_mul: Optional[float] = None,
        on_z: Optional[bool] = None,
        use_triton: Optional[bool] = None,
        triton_scale_only: Optional[bool] = None,
        triton_div_rn: Optional[bool] = None,
    ):
        self.delta_q_step = step
        self.delta_q_mode = mode
        if granularity is not None:
            self.delta_q_granularity = granularity
        if stat is not None:
            self.delta_q_stat = stat
        if bits is not None:
            self.delta_q_bits = bits
        if range_mul is not None:
            self.delta_q_range_mul = range_mul
        if on_z is not None:
            self.delta_q_on_z = bool(on_z)
        if use_triton is not None:
            self.delta_q_use_triton = bool(use_triton)
        if triton_scale_only is not None:
            self.delta_q_triton_scale_only = bool(triton_scale_only)
        if triton_div_rn is not None:
            self.delta_q_triton_div_rn = bool(triton_div_rn)
        for l in self.text_encoder_loras + self.unet_loras:
            l.delta_q_step = step
            l.delta_q_mode = mode
            if granularity is not None:
                l.delta_q_granularity = granularity
            if stat is not None:
                l.delta_q_stat = stat
            if bits is not None:
                l.delta_q_bits = bits
            if range_mul is not None:
                l.delta_q_range_mul = range_mul
            if on_z is not None:
                l.delta_q_on_z = bool(on_z)
            if use_triton is not None:
                l.delta_q_use_triton = bool(use_triton)
            if triton_scale_only is not None:
                l.delta_q_triton_scale_only = bool(triton_scale_only)
            if triton_div_rn is not None:
                l.delta_q_triton_div_rn = bool(triton_div_rn)

    def set_delta_quant_enabled(self, enabled: bool):
        for l in self.text_encoder_loras + self.unet_loras:
            l.delta_q_enabled = enabled

    def set_dq_stats_state(
        self,
        *,
        step_idx: int,
        device,
        do_log: bool,
        do_auto: bool,
        collect_full: bool,
        collect_zero: bool,
        collect_near_zero: bool,
        collect_error_parts: bool = False,
        log_mode: str,
        log_scope: str,
        auto_scope: str,
        target: str,
    ):
        if self.dq_stats_manager is None:
            return
        self.dq_stats_manager.begin_step(
            step_idx=step_idx,
            device=device,
            do_log=do_log,
            do_auto=do_auto,
            collect_full=collect_full,
            collect_zero=collect_zero,
            collect_near_zero=collect_near_zero,
            collect_error_parts=collect_error_parts,
            log_mode=log_mode,
            log_scope=log_scope,
            auto_scope=auto_scope,
            target=target,
        )

    def discard_dq_stats_step(self, step_idx: int):
        if self.dq_stats_manager is None:
            return
        self.dq_stats_manager.discard_step(step_idx)

    def export_dq_stats(self):
        if self.dq_stats_manager is None:
            return None
        return self.dq_stats_manager.export()

    def compute_rank_stats(self, scope: str = "unet", eps: float = 1e-12):
        if scope != "unet":
            return None
        loras = self.unet_loras
        if not loras:
            return None

        per_module = []
        with torch.no_grad():
            for lora in loras:
                stats = _compute_lora_effective_rank_stats(lora, eps=eps)
                if stats is not None:
                    per_module.append(stats)

        if not per_module:
            return None

        energy_sum = float(sum(item["energy"] for item in per_module))
        active = [item for item in per_module if item["energy"] > eps]
        if active:
            sat_values = [item["sat"] for item in active]
            top1_values = [item["top1"] for item in active]
        else:
            sat_values = [item["sat"] for item in per_module]
            top1_values = [item["top1"] for item in per_module]

        def _quantile(values, q):
            if not values:
                return None
            return float(np.quantile(np.asarray(values, dtype=np.float64), q))

        sat_p50 = _quantile(sat_values, 0.5)
        sat_p95 = _quantile(sat_values, 0.95)
        sat_max = float(max(sat_values)) if sat_values else None
        top1_p95 = _quantile(top1_values, 0.95)
        sat_wmean = None
        if energy_sum > eps:
            sat_wmean = float(sum(item["energy"] * item["sat"] for item in per_module) / energy_sum)
        else:
            sat_wmean = 0.0

        r_values = {item["r"] for item in per_module}
        rank_dim = next(iter(r_values)) if len(r_values) == 1 else None

        return {
            "per_module": per_module,
            "by_module": {item["module"]: item for item in per_module},
            "rank_dim": rank_dim,
            "sat_wmean": sat_wmean,
            "sat_p50": sat_p50,
            "sat_p95": sat_p95,
            "sat_max": sat_max,
            "top1_p95": top1_p95,
            "energy_sum": energy_sum,
        }

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(self, text_encoder, unet, apply_text_encoder=True, apply_unet=True):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoder, unet, weights_sd, dtype, device):
        apply_text_encoder = apply_unet = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER):
                apply_text_encoder = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_UNET):
                apply_unet = True

        if apply_text_encoder:
            logger.info("enable LoRA for text encoder")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info("enable LoRA for U-Net")
        else:
            self.unet_loras = []

        for lora in self.text_encoder_loras + self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            lora.merge_to(sd_for_lora, dtype, device)

        logger.info(f"weights are merged")

    # 層別学習率用に層ごとの学習率に対する倍率を定義する　引数の順番が逆だがとりあえず気にしない
    def set_block_lr_weight(self, block_lr_weight: Optional[List[float]]):
        self.block_lr = True
        self.block_lr_weight = block_lr_weight

    def get_lr_weight(self, block_idx: int) -> float:
        if not self.block_lr or self.block_lr_weight is None:
            return 1.0
        return self.block_lr_weight[block_idx]

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def set_te_train_targets(self, target_indices: Optional[Sequence[int]]):
        if not hasattr(self, "_text_encoder_loras_by_encoder"):
            return

        if target_indices is None:
            active = [idx for idx, group in enumerate(self._text_encoder_loras_by_encoder) if len(group) > 0]
        else:
            active = []
            for idx in target_indices:
                if not isinstance(idx, int):
                    continue
                if idx < 0 or idx >= len(self._text_encoder_loras_by_encoder):
                    continue
                if len(self._text_encoder_loras_by_encoder[idx]) == 0:
                    continue
                if idx not in active:
                    active.append(idx)

        disabled = sorted(i for i in range(len(self._text_encoder_loras_by_encoder)) if i not in active)
        if self._has_multiple_text_encoders and disabled:
            logger.info(
                "disable LoRA for Text Encoder target(s): %s",
                ", ".join(f"TE{i + 1}" for i in disabled),
            )

        self._active_text_encoder_indices = active
        self.text_encoder_loras = [
            lora
            for idx, group in enumerate(self._text_encoder_loras_by_encoder)
            if idx in self._active_text_encoder_indices
            for lora in group
        ]

    # 二つのText Encoderに別々の学習率を設定できるようにするといいかも
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr, **kwargs):
        # TODO warn if optimizer is not compatible with LoRA+ (but it will cause error so we don't need to check it here?)
        # if (
        #     self.loraplus_lr_ratio is not None
        #     or self.loraplus_text_encoder_lr_ratio is not None
        #     or self.loraplus_unet_lr_ratio is not None
        # ):
        #     assert (
        #         optimizer_type.lower() != "prodigy" and "dadapt" not in optimizer_type.lower()
        #     ), "LoRA+ and Prodigy/DAdaptation is not supported / LoRA+とProdigy/DAdaptationの組み合わせはサポートされていません"

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            return params, descriptions

        te_lr_overrides: Dict[int, float] = kwargs.get("text_encoder_lrs", {}) or {}
        active_indices = kwargs.get("active_text_encoder_indices")
        if active_indices is None:
            active_indices = getattr(self, "_active_text_encoder_indices", list(range(len(getattr(self, "_text_encoder_loras_by_encoder", [])))))

        if getattr(self, "_text_encoder_loras_by_encoder", None):
            for idx, loras in enumerate(self._text_encoder_loras_by_encoder):
                if idx not in active_indices or len(loras) == 0:
                    continue

                lr = te_lr_overrides.get(idx, text_encoder_lr if text_encoder_lr is not None else default_lr)
                params, descriptions = assemble_params(
                    loras,
                    lr,
                    self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio,
                )
                if params:
                    all_params.extend(params)
                    base_desc = "textencoder"
                    if self._has_multiple_text_encoders:
                        base_desc = f"textencoder{idx + 1}"
                    lr_descriptions.extend([base_desc + (" " + d if d else "") for d in descriptions])

        if self.unet_loras:
            if self.block_lr:
                is_sdxl = False
                for lora in self.unet_loras:
                    if "input_blocks" in lora.lora_name or "output_blocks" in lora.lora_name:
                        is_sdxl = True
                        break

                # 学習率のグラフをblockごとにしたいので、blockごとにloraを分類
                block_idx_to_lora = {}
                for lora in self.unet_loras:
                    idx = get_block_index(lora.lora_name, is_sdxl)
                    if idx not in block_idx_to_lora:
                        block_idx_to_lora[idx] = []
                    block_idx_to_lora[idx].append(lora)

                # blockごとにパラメータを設定する
                for idx, block_loras in block_idx_to_lora.items():
                    params, descriptions = assemble_params(
                        block_loras,
                        (unet_lr if unet_lr is not None else default_lr) * self.get_lr_weight(idx),
                        self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
                    )
                    all_params.extend(params)
                    lr_descriptions.extend([f"unet_block{idx}" + (" " + d if d else "") for d in descriptions])

            else:
                params, descriptions = assemble_params(
                    self.unet_loras,
                    unet_lr if unet_lr is not None else default_lr,
                    self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio,
                )
                all_params.extend(params)
                lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    # mask is a tensor with values from 0 to 1
    def set_region(self, sub_prompt_index, is_last_network, mask):
        if mask.max() == 0:
            mask = torch.ones_like(mask)

        self.mask = mask
        self.sub_prompt_index = sub_prompt_index
        self.is_last_network = is_last_network

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.set_network(self)

    def set_current_generation(self, batch_size, num_sub_prompts, width, height, shared, ds_ratio=None):
        self.batch_size = batch_size
        self.num_sub_prompts = num_sub_prompts
        self.current_size = (height, width)
        self.shared = shared

        # create masks
        mask = self.mask
        mask_dic = {}
        mask = mask.unsqueeze(0).unsqueeze(1)  # b(1),c(1),h,w
        ref_weight = self.text_encoder_loras[0].lora_down.weight if self.text_encoder_loras else self.unet_loras[0].lora_down.weight
        dtype = ref_weight.dtype
        device = ref_weight.device

        def resize_add(mh, mw):
            # logger.info(mh, mw, mh * mw)
            m = torch.nn.functional.interpolate(mask, (mh, mw), mode="bilinear")  # doesn't work in bf16
            m = m.to(device, dtype=dtype)
            mask_dic[mh * mw] = m

        h = height // 8
        w = width // 8
        for _ in range(4):
            resize_add(h, w)
            if h % 2 == 1 or w % 2 == 1:  # add extra shape if h/w is not divisible by 2
                resize_add(h + h % 2, w + w % 2)

            # deep shrink
            if ds_ratio is not None:
                hd = int(h * ds_ratio)
                wd = int(w * ds_ratio)
                resize_add(hd, wd)

            h = (h + 1) // 2
            w = (w + 1) // 2

        self.mask_dic = mask_dic

    def backup_weights(self):
        # 重みのバックアップを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        # 重みのリストアを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        # 事前計算を行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device, exclude_param_ids=None):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0
        exclude_param_ids = exclude_param_ids or set()
        named_params = dict(self.named_parameters())

        state_dict = self.state_dict()
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                upkey = key.replace("lora_down", "lora_up")
                down_param = named_params.get(key)
                up_param = named_params.get(upkey)
                if (down_param is not None and id(down_param) in exclude_param_ids) or (
                    up_param is not None and id(up_param) in exclude_param_ids
                ):
                    continue
                downkeys.append(key)
                upkeys.append(upkey)
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = desired.cpu() / norm.cpu()
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        if not norms:
            return keys_scaled, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)
