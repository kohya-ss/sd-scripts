from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch

from dq_profile.protocol import CandidateDefinition, deterministic_seed
from library.rounding_util import compute_scale_bits


@dataclass
class ShadowMetricRow:
    phase: str
    probe_or_step: str
    module_name: str
    target: str
    invocation: int
    repeat: int
    candidate: str
    numel: int
    clip_rate: float
    error_rms: float
    clip_error_rms: float
    round_error_rms: float
    fisher_error_mean: float
    fisher_clip_error_mean: float
    fisher_round_error_mean: float
    signed_impact_mean: float
    signed_clip_impact_mean: float
    signed_round_impact_mean: float
    grad_rms: float

    def to_dict(self) -> dict[str, Any]:
        return vars(self).copy()


class ProfileQuantContext:
    """Per-pass quantization context used only by ``copied_lora``.

    The context owns a private generator for every module invocation.  It never
    advances PyTorch's global RNG, which keeps model dropout/noise aligned
    across candidates.
    """

    def __init__(self, protocol_seed: int, *, rng_mode: str = "stateless") -> None:
        if rng_mode not in {"stateless", "legacy"}:
            raise ValueError(f"unknown profile quant RNG mode: {rng_mode}")
        self.protocol_seed = int(protocol_seed)
        self.rng_mode = rng_mode
        self.mode = "inactive"
        self.phase = ""
        self.probe_or_step = ""
        self.repeat = 0
        self.dropout_enabled = True
        self.grad_scale = 1.0
        self.mechanism = "full"
        self.shadow_candidates: tuple[CandidateDefinition, ...] = ()
        self.shadow_repeats = 0
        self._invocations: dict[str, int] = defaultdict(int)
        self._module_forwards: dict[str, int] = defaultdict(int)
        self._shadow_rows: list[ShadowMetricRow] = []
        self._dropout_hasher = hashlib.sha256()
        self._quant_hasher = hashlib.sha256()
        self._dropout_site_count = 0
        self._quant_rng_call_count = 0
        self._last_trace: dict[str, Any] = {}

    def begin_pass(
        self,
        *,
        mode: str,
        phase: str,
        probe_or_step: str | int,
        repeat: int = 0,
        dropout_enabled: bool = True,
        grad_scale: float = 1.0,
        mechanism: str = "full",
        shadow_candidates: tuple[CandidateDefinition, ...] = (),
        shadow_repeats: int = 0,
        control_rng_digest: Optional[str] = None,
    ) -> None:
        if mode not in {"inactive", "candidate", "shadow"}:
            raise ValueError(f"unknown profile pass mode: {mode}")
        if mechanism not in {"full", "clip_only", "round_only"}:
            raise ValueError(f"unknown profile quant mechanism: {mechanism}")
        self.mode = mode
        self.phase = str(phase)
        self.probe_or_step = str(probe_or_step)
        self.repeat = int(repeat)
        self.dropout_enabled = bool(dropout_enabled)
        self.grad_scale = max(float(grad_scale), 1.0)
        self.mechanism = mechanism
        self.shadow_candidates = tuple(shadow_candidates)
        self.shadow_repeats = max(0, int(shadow_repeats))
        self._invocations.clear()
        self._module_forwards.clear()
        self._shadow_rows.clear()
        self._dropout_hasher = hashlib.sha256()
        self._quant_hasher = hashlib.sha256()
        self._dropout_site_count = 0
        self._quant_rng_call_count = 0
        header = repr((self.phase, self.probe_or_step, self.repeat)).encode("utf-8")
        dropout_header = repr(
            (self.phase, self.probe_or_step, self.repeat, control_rng_digest)
        ).encode("utf-8")
        self._dropout_hasher.update(dropout_header)
        self._quant_hasher.update(header)

    def record_module_invocation(self, module_name: str) -> None:
        self._module_forwards[str(module_name)] += 1

    def record_dropout_site(
        self,
        *,
        module_name: str,
        kind: str,
        probability: float,
        shape: tuple[int, ...],
        actual: Optional[torch.Tensor | bool] = None,
    ) -> None:
        self._dropout_site_count += 1
        self._dropout_hasher.update(
            repr((module_name, kind, float(probability), tuple(int(value) for value in shape))).encode("utf-8")
        )
        if isinstance(actual, torch.Tensor):
            tensor = actual.detach().to("cpu").contiguous()
            self._dropout_hasher.update(str(tensor.dtype).encode("ascii"))
            self._dropout_hasher.update(tensor.numpy().tobytes())
        elif actual is not None:
            self._dropout_hasher.update(repr(bool(actual)).encode("ascii"))

    def pass_trace(self) -> dict[str, Any]:
        invocation_items = sorted((name, int(count)) for name, count in self._module_forwards.items())
        invocation_hasher = hashlib.sha256(repr(invocation_items).encode("utf-8"))
        return {
            "dropout_mask_digest": self._dropout_hasher.hexdigest(),
            "dropout_site_count": self._dropout_site_count,
            "quant_rng_digest": self._quant_hasher.hexdigest(),
            "quant_rng_call_count": self._quant_rng_call_count,
            "module_invocation_count": sum(count for _, count in invocation_items),
            "module_invocation_digest": invocation_hasher.hexdigest(),
        }

    @property
    def last_trace(self) -> dict[str, Any]:
        return dict(self._last_trace)

    def finish_pass(self) -> list[dict[str, Any]]:
        rows = [row.to_dict() for row in self._shadow_rows]
        self._last_trace = self.pass_trace()
        self.mode = "inactive"
        self.mechanism = "full"
        self.shadow_candidates = ()
        self.shadow_repeats = 0
        self._invocations.clear()
        self._shadow_rows.clear()
        return rows

    def claim_invocation(self, module_name: str) -> int:
        invocation = self._invocations[module_name]
        self._invocations[module_name] = invocation + 1
        return invocation

    def rand_for(
        self,
        x: torch.Tensor,
        *,
        module_name: str,
        invocation: Optional[int] = None,
        repeat: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        if self.rng_mode == "legacy":
            return None
        if invocation is None:
            invocation = self.claim_invocation(module_name)
        seed = deterministic_seed(
            self.protocol_seed,
            phase=self.phase,
            probe_or_step=self.probe_or_step,
            module_name=module_name,
            invocation=invocation,
            repeat=self.repeat if repeat is None else int(repeat),
        )
        self._quant_rng_call_count += 1
        self._quant_hasher.update(
            repr((module_name, int(invocation), self.repeat if repeat is None else int(repeat), seed, tuple(x.shape))).encode("utf-8")
        )
        generator = torch.Generator(device=x.device)
        generator.manual_seed(seed)
        return torch.rand(x.shape, dtype=torch.float32, device=x.device, generator=generator)

    def apply_mechanism(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        x_clamped: torch.Tensor,
    ) -> torch.Tensor:
        if self.mechanism == "full":
            return q
        if self.mechanism == "clip_only":
            error = x_clamped.to(x.dtype) - x
        elif self.mechanism == "round_only":
            error = q.detach().to(x.dtype) - x_clamped.to(x.dtype)
        else:  # guarded by begin_pass
            raise RuntimeError(f"unsupported quant mechanism: {self.mechanism}")
        return x + error.detach()


    @staticmethod
    def _virtual_quantize(
        x: torch.Tensor,
        *,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
        mode: str,
        rand: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x32 = x.to(torch.float32)
        scale32 = scale.to(device=x.device, dtype=torch.float32)
        normalized = x32 / scale32
        clamped_levels = torch.clamp(normalized, qmin, qmax)
        x_clamped = clamped_levels * scale32
        if mode == "det":
            rounded_levels = torch.round(clamped_levels)
        elif mode == "stoch":
            if rand is None:
                raise RuntimeError("shadow stochastic quantization requires explicit rand")
            floor = torch.floor(clamped_levels)
            probability = (clamped_levels - floor).clamp(0.0, 1.0)
            rounded_levels = floor + (rand.to(torch.float32) < probability).to(torch.float32)
        else:
            raise ValueError(f"unknown round mode: {mode}")
        q = torch.clamp(rounded_levels, qmin, qmax) * scale32
        return q, x_clamped, clamped_levels, normalized

    def attach_shadow_hook(
        self,
        x: torch.Tensor,
        *,
        module_name: str,
        target: str,
        bits: int,
        granularity: str,
        stat: str,
        quant_mode: str,
        use_triton_scale: bool,
    ) -> torch.Tensor:
        if self.mode != "shadow" or not x.requires_grad or not self.shadow_candidates:
            return x
        invocation = self.claim_invocation(module_name)
        x_reference = x.detach()
        phase = self.phase
        probe_or_step = self.probe_or_step
        grad_scale = self.grad_scale
        repeats = self.shadow_repeats
        candidates = self.shadow_candidates
        qmax = (1 << (int(bits) - 1)) - 1

        def collect(grad: torch.Tensor) -> torch.Tensor:
            grad_unscaled = grad.detach().to(torch.float32) / grad_scale
            x32 = x_reference.to(torch.float32)
            grad_sumsq = torch.sum(grad_unscaled * grad_unscaled)
            for candidate in candidates:
                if not candidate.quantized or candidate.initial_range_mul is None:
                    continue
                with torch.no_grad():
                    scale = compute_scale_bits(
                        x_reference,
                        bits=int(bits),
                        granularity=granularity,
                        stat=stat if stat != "none" else "rms",
                        range_mul=float(candidate.initial_range_mul),
                        use_triton=bool(use_triton_scale),
                    )
                    for repeat in range(repeats):
                        rand = self.rand_for(
                            x_reference,
                            module_name=module_name,
                            invocation=invocation,
                            repeat=repeat,
                        ) if quant_mode == "stoch" else None
                        q, x_clamped, clamped_levels, normalized = self._virtual_quantize(
                            x_reference,
                            scale=scale,
                            qmin=-qmax,
                            qmax=qmax,
                            mode=quant_mode,
                            rand=rand,
                        )
                        error = q - x32
                        clip_error = x_clamped - x32
                        round_error = q - x_clamped
                        count = int(x32.numel())
                        denom = max(count, 1)
                        self._shadow_rows.append(
                            ShadowMetricRow(
                                phase=phase,
                                probe_or_step=probe_or_step,
                                module_name=module_name,
                                target=target,
                                invocation=invocation,
                                repeat=repeat,
                                candidate=candidate.name,
                                numel=count,
                                clip_rate=float((clamped_levels.abs() >= qmax).to(torch.float32).mean().item()),
                                error_rms=float(torch.sqrt(torch.sum(error * error) / denom).item()),
                                clip_error_rms=float(torch.sqrt(torch.sum(clip_error * clip_error) / denom).item()),
                                round_error_rms=float(torch.sqrt(torch.sum(round_error * round_error) / denom).item()),
                                fisher_error_mean=float(torch.sum((grad_unscaled * error) ** 2).item() / denom),
                                fisher_clip_error_mean=float(torch.sum((grad_unscaled * clip_error) ** 2).item() / denom),
                                fisher_round_error_mean=float(torch.sum((grad_unscaled * round_error) ** 2).item() / denom),
                                signed_impact_mean=float(torch.sum(grad_unscaled * error).item() / denom),
                                signed_clip_impact_mean=float(torch.sum(grad_unscaled * clip_error).item() / denom),
                                signed_round_impact_mean=float(torch.sum(grad_unscaled * round_error).item() / denom),
                                grad_rms=float(torch.sqrt(grad_sumsq / denom).item()),
                            )
                        )
            return grad

        x.register_hook(collect)
        return x


def aggregate_shadow_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row.get("phase", "")),
                str(row.get("probe_or_step", "")),
                str(row.get("image_key", "")),
                str(row.get("timestep_bin", "")),
                str(row.get("probe_replica", "")),
                str(row.get("probe_regime", "")),
                str(row["candidate"]),
                str(row["module_name"]) + "\x1f" + str(row["target"]),
            )
        ].append(row)
    result: list[dict[str, Any]] = []
    metric_names = (
        "clip_rate",
        "error_rms",
        "clip_error_rms",
        "round_error_rms",
        "fisher_error_mean",
        "fisher_clip_error_mean",
        "fisher_round_error_mean",
        "signed_impact_mean",
        "signed_clip_impact_mean",
        "signed_round_impact_mean",
        "grad_rms",
    )
    for (phase, probe_or_step, image_key, timestep_bin, probe_replica, probe_regime, candidate, module_and_target), items in sorted(grouped.items()):
        module_name, target = module_and_target.split("\x1f", 1)
        out: dict[str, Any] = {
            "phase": phase,
            "probe_or_step": probe_or_step,
            "image_key": image_key,
            "timestep_bin": timestep_bin,
            "probe_replica": probe_replica,
            "probe_regime": probe_regime,
            "candidate": candidate,
            "module_name": module_name,
            "target": target,
            "repeat_count": len(items),
            "numel": max(int(item["numel"]) for item in items),
        }
        for metric in metric_names:
            values = [float(item[metric]) for item in items]
            out[f"{metric}_mean"] = sum(values) / len(values)
            out[f"{metric}_std"] = math.sqrt(sum((value - out[f"{metric}_mean"]) ** 2 for value in values) / len(values))
        result.append(out)
    return result
