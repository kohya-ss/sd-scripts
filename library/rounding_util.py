from __future__ import annotations

import os
from pathlib import Path

import torch
from typing import Iterable, Literal, Union, Optional

try:
    from library.triton_quant import triton_compute_scale_bits_channel_rms, triton_fake_quantize_levels_stoch
except Exception:
    triton_compute_scale_bits_channel_rms = None
    triton_fake_quantize_levels_stoch = None


RoundMode = Literal["det", "stoch"]
_DQ_TRITON_CAPTURE_COUNT = 0
_DQ_TRITON_CAPTURE_SEEN = 0
_DQ_TRITON_CAPTURE_GLOBAL_STEP: Optional[int] = None


def set_dq_triton_capture_global_step(global_step: Optional[int]) -> None:
    """Set the current training step for optional dq_delta capture filters.

    This is a diagnostic hook. It is intentionally independent from normal
    fake-quant computation unless DQ_TRITON_CAPTURE_DIR is set.
    """
    global _DQ_TRITON_CAPTURE_GLOBAL_STEP
    _DQ_TRITON_CAPTURE_GLOBAL_STEP = None if global_step is None else int(global_step)


def _read_optional_int_env(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _maybe_capture_fake_quant_levels(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    mode: str,
    use_triton: bool,
) -> None:
    """Optionally save real training tensors for Triton/PyTorch comparison.

    Enable with:
        DQ_TRITON_CAPTURE_DIR=path

    By default, only calls with use_triton=True are captured. Set
    DQ_TRITON_CAPTURE_USE_TRITON_ONLY=0 to capture all calls.
    """
    global _DQ_TRITON_CAPTURE_COUNT, _DQ_TRITON_CAPTURE_SEEN

    capture_dir = os.environ.get("DQ_TRITON_CAPTURE_DIR")
    if not capture_dir or mode != "stoch":
        return
    triton_only = os.environ.get("DQ_TRITON_CAPTURE_USE_TRITON_ONLY", "1").lower() not in ("0", "false", "no")
    if triton_only and not use_triton:
        return

    min_global_step = _read_optional_int_env("DQ_TRITON_CAPTURE_MIN_GLOBAL_STEP")
    max_global_step = _read_optional_int_env("DQ_TRITON_CAPTURE_MAX_GLOBAL_STEP")
    if min_global_step is not None or max_global_step is not None:
        if _DQ_TRITON_CAPTURE_GLOBAL_STEP is None:
            return
        if min_global_step is not None and _DQ_TRITON_CAPTURE_GLOBAL_STEP < min_global_step:
            return
        if max_global_step is not None and _DQ_TRITON_CAPTURE_GLOBAL_STEP > max_global_step:
            return

    try:
        limit = int(os.environ.get("DQ_TRITON_CAPTURE_LIMIT", "16"))
    except ValueError:
        limit = 16
    if limit <= 0 or _DQ_TRITON_CAPTURE_COUNT >= limit:
        return

    try:
        skip = int(os.environ.get("DQ_TRITON_CAPTURE_SKIP", "0"))
    except ValueError:
        skip = 0
    if _DQ_TRITON_CAPTURE_SEEN < max(0, skip):
        _DQ_TRITON_CAPTURE_SEEN += 1
        return
    _DQ_TRITON_CAPTURE_SEEN += 1

    capture_index = _DQ_TRITON_CAPTURE_COUNT
    _DQ_TRITON_CAPTURE_COUNT += 1

    try:
        path = Path(capture_dir)
        path.mkdir(parents=True, exist_ok=True)
        scale_cpu = scale.detach().cpu() if isinstance(scale, torch.Tensor) else torch.tensor(scale, dtype=torch.float32)
        payload = {
            "x": x.detach().cpu(),
            "scale": scale_cpu,
            "qmin": int(qmin),
            "qmax": int(qmax),
            "mode": mode,
            "use_triton": bool(use_triton),
            "global_step": _DQ_TRITON_CAPTURE_GLOBAL_STEP,
            "global_step_1based": None
            if _DQ_TRITON_CAPTURE_GLOBAL_STEP is None
            else int(_DQ_TRITON_CAPTURE_GLOBAL_STEP) + 1,
            "capture_seen": int(_DQ_TRITON_CAPTURE_SEEN),
            "x_shape": tuple(x.shape),
            "x_stride": tuple(x.stride()),
            "x_dtype": str(x.dtype),
            "x_is_contiguous": bool(x.is_contiguous()),
            "scale_shape": tuple(scale.shape),
            "scale_stride": tuple(scale.stride()),
            "scale_dtype": str(scale.dtype),
            "scale_is_contiguous": bool(scale.is_contiguous()),
        }
        torch.save(payload, path / f"dq_fake_quant_capture_{os.getpid()}_{capture_index:04d}.pt")
    except Exception:
        # Capture is diagnostic-only and must never disturb training.
        return


@torch.no_grad()
def round_tensor_det(x: torch.Tensor, step: float) -> torch.Tensor:
    """Deterministically round values of `x` to multiples of `step`.

    The computation is done in fp32 on the same device and cast back to the
    original dtype to avoid unexpected precision artifacts.
    """
    if step <= 0:
        return x
    dtype = x.dtype
    y = x.to(torch.float32)
    y = torch.round(y / step) * step
    return y.to(dtype)


@torch.no_grad()
def round_tensor_stoch(x: torch.Tensor, step: float) -> torch.Tensor:
    """Stochastically round values of `x` to neighboring multiples of `step`.

    For each value v, with q = floor(v/step), rounds to q*step or (q+1)*step
    with probability proportional to the fractional part.
    Computation is in fp32 on the same device.
    """
    if step <= 0:
        return x
    dtype = x.dtype
    y = x.to(torch.float32)
    q = torch.floor(y / step)
    r = (y / step) - q  # fractional part in [0,1)
    probs = r.clamp(0.0, 1.0)
    # Bernoulli via uniform comparison keeps it fast and vectorized
    incr = (torch.rand_like(probs) < probs).to(y.dtype)
    y = (q + incr) * step
    return y.to(dtype)


def _ste_from_quantized(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator wrapper: use quantized values in forward,
    but pass-through gradients as identity (d/dx ≈ 1).
    """
    return x + (q - x).detach()


def ste_round_tensor_det(x: torch.Tensor, step: Union[float, torch.Tensor]) -> torch.Tensor:
    """Deterministic rounding with STE (keeps gradients)."""
    if not isinstance(step, torch.Tensor):
        if step is None or step <= 0:
            return x
    dtype = x.dtype
    y = x.to(torch.float32)
    s = step
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, dtype=torch.float32, device=y.device)
    else:
        s = s.to(device=y.device, dtype=torch.float32)
    q = torch.round(y / s) * s
    q = q.to(dtype)
    return _ste_from_quantized(x, q)


def ste_round_tensor_stoch(x: torch.Tensor, step: Union[float, torch.Tensor]) -> torch.Tensor:
    """Stochastic rounding with STE (keeps gradients)."""
    if not isinstance(step, torch.Tensor):
        if step is None or step <= 0:
            return x
    dtype = x.dtype
    y = x.to(torch.float32)
    s = step
    if not isinstance(s, torch.Tensor):
        s = torch.tensor(s, dtype=torch.float32, device=y.device)
    else:
        s = s.to(device=y.device, dtype=torch.float32)
    q = torch.floor(y / s)
    r = (y / s) - q
    probs = r.clamp(0.0, 1.0)
    incr = (torch.rand_like(probs) < probs).to(y.dtype)
    q = (q + incr) * s
    q = q.to(dtype)
    return _ste_from_quantized(x, q)


def fake_quantize(
    x: torch.Tensor,
    *,
    step: Union[float, torch.Tensor],
    mode: RoundMode = "det",
) -> torch.Tensor:
    """Fake-quantize tensor values to multiples of `step` using STE.

    - Forward: rounds to the nearest (or stochastic) grid point.
    - Backward: gradient is approximated as identity (STE).
    """
    if not isinstance(step, torch.Tensor):
        if step is None or step <= 0:
            return x
    if mode == "det":
        return ste_round_tensor_det(x, step)
    if mode == "stoch":
        return ste_round_tensor_stoch(x, step)
    raise ValueError(f"unknown round mode: {mode}")


def compute_per_channel_step(
    x: torch.Tensor,
    base_step: float,
    *,
    stat: Literal["rms", "absmax", "none"] = "rms",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute a per-channel step tensor broadcastable to x.

    - If stat == 'none', returns a scalar tensor with `base_step`.
    - For 4D: channel dim=1; for 3D/2D: channel dim=-1.
    - Step is `base_step * stat_per_channel`.
    """
    device = x.device
    if stat == "none":
        return torch.tensor(base_step, dtype=torch.float32, device=device)

    if x.ndim == 4:
        # (N, C, H, W)
        reduce_dims = (0, 2, 3)
        shape = (1, x.size(1), 1, 1)
    elif x.ndim == 3:
        # (N, L, C)
        reduce_dims = (0, 1)
        shape = (1, 1, x.size(2))
    elif x.ndim == 2:
        # (N, C)
        reduce_dims = (0,)
        shape = (1, x.size(1))
    else:
        # fallback to scalar
        return torch.tensor(base_step, dtype=torch.float32, device=device)

    if stat == "rms":
        s = torch.sqrt(torch.mean(x.to(torch.float32) ** 2, dim=reduce_dims, keepdim=True) + eps)
    elif stat == "absmax":
        s = torch.amax(torch.abs(x.to(torch.float32)), dim=reduce_dims, keepdim=True) + eps
    else:
        raise ValueError(f"unknown stat: {stat}")

    step = (base_step * s).to(torch.float32)
    # reshape to channel-broadcastable
    # s is already keepdim=True, but make sure shape matches broadcast for safety
    return step.view(*s.shape)


def fake_quantize_levels(
    x: torch.Tensor,
    *,
    scale: Union[float, torch.Tensor],
    qmin: int,
    qmax: int,
    mode: RoundMode = "det",
    use_triton: bool = False,
) -> torch.Tensor:
    """STE fake-quantization with finite integer levels and (symmetric) clamp.

    y = clamp(round(x/scale), qmin, qmax) * scale  (det)
    or
    y = clamp(stoch_round(x/scale), qmin, qmax) * scale  (stoch)
    """
    if not isinstance(scale, torch.Tensor):
        s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    else:
        s = scale.to(device=x.device, dtype=torch.float32)

    _maybe_capture_fake_quant_levels(x, s, qmin, qmax, mode, use_triton)

    if use_triton and mode == "stoch" and triton_fake_quantize_levels_stoch is not None and isinstance(s, torch.Tensor):
        q_triton = triton_fake_quantize_levels_stoch(
            x,
            scale=s,
            qmin=qmin,
            qmax=qmax,
        )
        if q_triton is not None:
            return _ste_from_quantized(x, q_triton)

    y = x.to(torch.float32) / s
    if mode == "det":
        q = torch.round(y)
    elif mode == "stoch":
        frac = y - torch.floor(y)
        probs = frac.clamp(0.0, 1.0)
        q = torch.floor(y) + (torch.rand_like(probs) < probs).to(y.dtype)
    else:
        raise ValueError(f"unknown round mode: {mode}")
    q = torch.clamp(q, qmin, qmax)
    q = (q * s).to(x.dtype)
    return _ste_from_quantized(x, q)


def _reduce_dims_and_shape(x: torch.Tensor):
    if x.ndim == 4:
        return (0, 2, 3), (1, x.size(1), 1, 1)
    if x.ndim == 3:
        return (0, 1), (1, 1, x.size(2))
    if x.ndim == 2:
        return (0,), (1, x.size(1))
    return None, None


def compute_scale_bits(
    x: torch.Tensor,
    *,
    bits: int,
    granularity: Literal["tensor", "channel"] = "tensor",
    stat: Literal["rms", "absmax"] = "rms",
    range_mul: float = 3.0,
    eps: float = 1e-8,
    use_triton: bool = False,
) -> torch.Tensor:
    """Compute per-tensor/per-channel scale for symmetric signed N-bit quant.

    scale = range / qmax, where range = absmax or (range_mul * rms).
    Returns a tensor broadcastable to x.
    """
    assert bits > 0
    qmax = (1 << (bits - 1)) - 1  # e.g., 127 for 8-bit signed

    if granularity == "tensor":
        if stat == "absmax":
            rng = torch.amax(torch.abs(x.to(torch.float32))) + eps
        elif stat == "rms":
            rng = torch.sqrt(torch.mean(x.to(torch.float32) ** 2) + eps) * range_mul
        else:
            raise ValueError(f"unknown stat: {stat}")
        return (rng / qmax).to(torch.float32)

    # per-channel
    reduce_dims, shape = _reduce_dims_and_shape(x)
    if reduce_dims is None:
        # fallback to per-tensor
        return compute_scale_bits(x, bits=bits, granularity="tensor", stat=stat, range_mul=range_mul, eps=eps)

    if stat == "absmax":
        rng = torch.amax(torch.abs(x.to(torch.float32)), dim=reduce_dims, keepdim=True) + eps
    elif stat == "rms":
        if use_triton and triton_compute_scale_bits_channel_rms is not None:
            scale_triton = triton_compute_scale_bits_channel_rms(x, bits=bits, range_mul=range_mul, eps=eps)
            if scale_triton is not None:
                return scale_triton
        rng = torch.sqrt(torch.mean(x.to(torch.float32) ** 2, dim=reduce_dims, keepdim=True) + eps) * range_mul
    else:
        raise ValueError(f"unknown stat: {stat}")
    scale = (rng / qmax).to(torch.float32)
    return scale

@torch.no_grad()
def round_parameters(
    params: Iterable[torch.nn.Parameter],
    *,
    step: float,
    mode: RoundMode = "det",
    exclude_param_ids: Optional[set[int]] = None,
) -> None:
    """In-place rounding of given parameters to multiples of `step`.

    Only floating point tensors are affected. Non-floating tensors are skipped.
    """
    if step is None or step <= 0:
        return

    exclude_param_ids = exclude_param_ids or set()
    for p in params:
        if id(p) in exclude_param_ids:
            continue
        if not torch.is_floating_point(p.data):
            continue
        if mode == "det":
            p.data.copy_(round_tensor_det(p.data, step))
        elif mode == "stoch":
            p.data.copy_(round_tensor_stoch(p.data, step))
        else:
            raise ValueError(f"unknown round mode: {mode}")
