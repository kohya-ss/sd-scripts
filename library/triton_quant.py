from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception as e:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False
    _TRITON_IMPORT_ERROR = e
else:
    _TRITON_IMPORT_ERROR = None

_warned_messages: set[str] = set()


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE


def _warn_once(key: str, message: str) -> None:
    if key in _warned_messages:
        return
    _warned_messages.add(key)
    logger.warning(message)


if _TRITON_AVAILABLE:
    # Keep scale (A) and fake-quant (B) as separate kernels. A fused A+B
    # shortcut was tested, but changed dq_delta logs/training behavior more
    # than the separate-kernel path, so it is intentionally not implemented.

    @triton.jit
    def _scale_bits_channel_rms_kernel(
        x_ptr,
        scale_ptr,
        reduction_count: tl.constexpr,
        size0: tl.constexpr,
        size1: tl.constexpr,
        size2: tl.constexpr,
        size3: tl.constexpr,
        ndim: tl.constexpr,
        range_mul,
        qmax,
        eps,
        BLOCK_SIZE: tl.constexpr,
    ):
        c = tl.program_id(axis=0)
        r = tl.arange(0, BLOCK_SIZE)
        mask = r < reduction_count

        if ndim == 4:
            # NCHW contiguous: offset = ((n * C + c) * H * W) + hw
            hw_size = size2 * size3
            n = r // hw_size
            hw = r - n * hw_size
            offsets = ((n * size1 + c) * hw_size) + hw
        elif ndim == 3:
            # NLC contiguous: offset = (n * L + l) * C + c
            n = r // size1
            l = r - n * size1
            offsets = (n * size1 + l) * size2 + c
        else:
            # NC contiguous: offset = n * C + c
            offsets = r * size1 + c

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sumsq = tl.sum(x * x, axis=0)
        rms = tl.sqrt(sumsq / reduction_count + eps)
        scale = rms * range_mul / qmax
        tl.store(scale_ptr + c, scale)

    @triton.jit
    def _fake_quantize_levels_stoch_kernel(
        x_ptr,
        scale_ptr,
        out_ptr,
        rand_ptr,
        n_elements: tl.constexpr,
        scale_numel: tl.constexpr,
        dim1: tl.constexpr,
        dim2: tl.constexpr,
        dim3: tl.constexpr,
        ndim: tl.constexpr,
        qmin: tl.constexpr,
        qmax: tl.constexpr,
        USE_DIV_RN: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if scale_numel == 1:
            scale_offsets = tl.full((BLOCK_SIZE,), 0, tl.int64)
        elif ndim == 4:
            # Flattened NCHW: channel = (offset / (H * W)) % C
            scale_offsets = (offsets // (dim2 * dim3)) % dim1
        elif ndim == 3:
            # Flattened NLC: channel = offset % C
            scale_offsets = offsets % dim2
        else:
            # Flattened NC: channel = offset % C
            scale_offsets = offsets % dim1

        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=1.0).to(tl.float32)
        if USE_DIV_RN:
            y = tl.div_rn(x, scale)
        else:
            y = x / scale
        q_floor = tl.floor(y)
        frac = y - q_floor
        probs = tl.minimum(tl.maximum(frac, 0.0), 1.0)
        rnd = tl.load(rand_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
        q = q_floor + (rnd < probs).to(tl.float32)
        q = tl.minimum(tl.maximum(q, qmin), qmax)
        out = q * scale
        tl.store(out_ptr + offsets, out, mask=mask)


def triton_compute_scale_bits_channel_rms(
    x: torch.Tensor,
    *,
    bits: int,
    range_mul: float,
    eps: float,
) -> Optional[torch.Tensor]:
    """Mirror compute_scale_bits(..., granularity="channel", stat="rms").

    Python reference:
        sqrt(mean(x.float() ** 2, channel_reduce_dims, keepdim=True) + eps)
        * range_mul / qmax

    Returns a broadcastable float32 scale tensor, or None so callers can
    fall back to the Python/PyTorch implementation.
    """
    if not _TRITON_AVAILABLE:
        return None
    if not x.is_cuda or x.ndim not in (2, 3, 4):
        return None
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not x.is_contiguous():
        return None

    if x.ndim == 4:
        channel_count = x.shape[1]
        reduction_count = x.shape[0] * x.shape[2] * x.shape[3]
        out_shape = (1, channel_count, 1, 1)
    elif x.ndim == 3:
        channel_count = x.shape[2]
        reduction_count = x.shape[0] * x.shape[1]
        out_shape = (1, 1, channel_count)
    else:
        channel_count = x.shape[1]
        reduction_count = x.shape[0]
        out_shape = (1, channel_count)

    if reduction_count <= 0:
        return None

    block_size = triton.next_power_of_2(reduction_count)
    if block_size > 131072:
        return None

    scale_flat = torch.empty((channel_count,), device=x.device, dtype=torch.float32)
    qmax = float((1 << (bits - 1)) - 1)
    size0 = x.shape[0]
    size1 = x.shape[1] if x.ndim >= 2 else 1
    size2 = x.shape[2] if x.ndim >= 3 else 1
    size3 = x.shape[3] if x.ndim >= 4 else 1

    try:
        _scale_bits_channel_rms_kernel[(channel_count,)](
            x,
            scale_flat,
            reduction_count,
            size0,
            size1,
            size2,
            size3,
            x.ndim,
            float(range_mul),
            qmax,
            float(eps),
            BLOCK_SIZE=block_size,
        )
    except Exception as e:
        _warn_once("triton_scale_kernel", f"Triton scale kernel failed; falling back to PyTorch: {e}")
        return None

    return scale_flat.view(*out_shape)


def triton_fake_quantize_levels_stoch(
    x: torch.Tensor,
    *,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    use_div_rn: bool = False,
) -> Optional[torch.Tensor]:
    """Mirror fake_quantize_levels(..., mode="stoch").

    Python reference:
        y = x.float() / scale.float()
        q_floor = floor(y)
        probs = clamp(y - q_floor, 0, 1)
        q = q_floor + (torch.rand_like(probs) < probs)
        q = clamp(q, qmin, qmax)
        out = (q * scale).to(x.dtype)

    Random numbers are intentionally generated by PyTorch and passed to the
    Triton kernel. This matched the original stochastic path better in
    training tests than tl.rand. STE remains in Python as
    x + (out - x).detach().
    """
    if not _TRITON_AVAILABLE:
        _warn_once(
            "triton_import",
            f"Triton fake quant requested, but Triton is unavailable: {_TRITON_IMPORT_ERROR}",
        )
        return None

    if not x.is_cuda or not scale.is_cuda:
        return None
    if x.ndim not in (2, 3, 4):
        return None
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not x.is_contiguous() or not scale.is_contiguous():
        return None

    scale_flat = scale.reshape(-1)
    scale_numel = scale_flat.numel()
    if scale_numel not in (1, x.shape[1] if x.ndim in (2, 4) else x.shape[2]):
        return None

    out = torch.empty_like(x)
    rand = torch.rand_like(x, dtype=torch.float32)
    n_elements = x.numel()
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size),)
    dim1 = x.shape[1] if x.ndim >= 2 else 1
    dim2 = x.shape[2] if x.ndim >= 3 else 1
    dim3 = x.shape[3] if x.ndim >= 4 else 1
    try:
        _fake_quantize_levels_stoch_kernel[grid](
            x,
            scale_flat,
            out,
            rand,
            n_elements,
            scale_numel,
            dim1,
            dim2,
            dim3,
            x.ndim,
            qmin,
            qmax,
            bool(use_div_rn),
            BLOCK_SIZE=block_size,
        )
    except Exception as e:
        _warn_once("triton_kernel", f"Triton fake quant failed; falling back to PyTorch: {e}")
        return None

    return out
