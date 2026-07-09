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
    def _scale_bits_channel_rms_nlc_kernel(
        x_ptr,
        scale_ptr,
        reduction_count: tl.constexpr,
        channel_count: tl.constexpr,
        range_mul,
        qmax,
        eps,
        BLOCK_R: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        c_offsets = tl.program_id(axis=0) * BLOCK_C + tl.arange(0, BLOCK_C)
        r_offsets = tl.arange(0, BLOCK_R)
        mask = (r_offsets[:, None] < reduction_count) & (c_offsets[None, :] < channel_count)

        # NLC contiguous: rows are N * L, channels are C. This 2D tile keeps
        # channel loads contiguous instead of gathering one strided channel at a time.
        offsets = r_offsets[:, None] * channel_count + c_offsets[None, :]
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        sumsq = tl.sum(x * x, axis=0)
        rms = tl.sqrt(sumsq / reduction_count + eps)
        scale = rms * range_mul / qmax
        tl.store(scale_ptr + c_offsets, scale, mask=c_offsets < channel_count)

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

    @triton.jit
    def _fake_quantize_levels_stoch_stats_kernel(
        x_ptr,
        scale_ptr,
        out_ptr,
        rand_ptr,
        stats_ptr,
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
            scale_offsets = (offsets // (dim2 * dim3)) % dim1
        elif ndim == 3:
            scale_offsets = offsets % dim2
        else:
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
        out_stored = tl.load(out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        numel = tl.sum(mask.to(tl.float32), axis=0)
        clip_count = tl.sum(((tl.abs(y) >= qmax) & mask).to(tl.float32), axis=0)
        sumsq = tl.sum(x * x, axis=0)
        xq_sumsq = tl.sum(out_stored * out_stored, axis=0)
        xxq_sum = tl.sum(x * out_stored, axis=0)

        base = pid * 5
        tl.store(stats_ptr + base + 0, numel)
        tl.store(stats_ptr + base + 1, clip_count)
        tl.store(stats_ptr + base + 2, sumsq)
        tl.store(stats_ptr + base + 3, xq_sumsq)
        tl.store(stats_ptr + base + 4, xxq_sum)

    @triton.jit
    def _fake_quant_stats_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        stats_ptr,
        n_elements: tl.constexpr,
        scale_numel: tl.constexpr,
        dim1: tl.constexpr,
        dim2: tl.constexpr,
        dim3: tl.constexpr,
        ndim: tl.constexpr,
        qmax: tl.constexpr,
        collect_zero: tl.constexpr,
        collect_near_zero: tl.constexpr,
        collect_full: tl.constexpr,
        collect_detail: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        q = tl.load(q_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if scale_numel == 1:
            scale_offsets = tl.full((BLOCK_SIZE,), 0, tl.int64)
        elif ndim == 4:
            scale_offsets = (offsets // (dim2 * dim3)) % dim1
        elif ndim == 3:
            scale_offsets = offsets % dim2
        else:
            scale_offsets = offsets % dim1

        scale = tl.load(scale_ptr + scale_offsets, mask=mask, other=1.0).to(tl.float32)
        y = x / scale
        q_clamp = tl.minimum(tl.maximum(y, -qmax), qmax)
        clipped = tl.abs(y) >= qmax

        numel = tl.sum(mask.to(tl.float32), axis=0)
        clip_count = tl.sum((clipped & mask).to(tl.float32), axis=0)

        zero_count = 0.0
        if collect_zero:
            zero_count = tl.sum(((q == 0.0) & mask).to(tl.float32), axis=0)

        near_zero_count = 0.0
        if collect_near_zero:
            near_zero_count = tl.sum(((tl.abs(x) < (0.5 * scale)) & mask).to(tl.float32), axis=0)

        sumsq = 0.0
        xq_sumsq = 0.0
        xxq_sum = 0.0
        absmax = 0.0
        if collect_full:
            sumsq = tl.sum(x * x, axis=0)
            xq_sumsq = tl.sum(q * q, axis=0)
            xxq_sum = tl.sum(x * q, axis=0)
        if collect_detail:
            absmax = tl.max(tl.abs(x), axis=0)

        base = pid * 8
        tl.store(stats_ptr + base + 0, numel)
        tl.store(stats_ptr + base + 1, clip_count)
        tl.store(stats_ptr + base + 2, zero_count)
        tl.store(stats_ptr + base + 3, near_zero_count)
        tl.store(stats_ptr + base + 4, sumsq)
        tl.store(stats_ptr + base + 5, xq_sumsq)
        tl.store(stats_ptr + base + 6, xxq_sum)
        tl.store(stats_ptr + base + 7, absmax)


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

    if x.ndim == 3 and channel_count >= 4096 and block_size * 16 <= 131072:
        try:
            _scale_bits_channel_rms_nlc_kernel[(triton.cdiv(channel_count, 16),)](
                x,
                scale_flat,
                reduction_count,
                channel_count,
                float(range_mul),
                qmax,
                float(eps),
                BLOCK_R=block_size,
                BLOCK_C=16,
                num_warps=8,
            )
        except Exception as e:
            _warn_once("triton_scale_nlc_kernel", f"Triton NLC scale kernel failed; falling back to PyTorch: {e}")
            return None

        return scale_flat.view(*out_shape)

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


def _check_fake_quant_inputs(x: torch.Tensor, scale: torch.Tensor) -> Optional[tuple[torch.Tensor, int, int, int, int]]:
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

    dim1 = x.shape[1] if x.ndim >= 2 else 1
    dim2 = x.shape[2] if x.ndim >= 3 else 1
    dim3 = x.shape[3] if x.ndim >= 4 else 1
    return scale_flat, scale_numel, dim1, dim2, dim3


def triton_fake_quantize_levels_stoch_with_stats(
    x: torch.Tensor,
    *,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    use_div_rn: bool = False,
    rand: Optional[torch.Tensor] = None,
) -> Optional[tuple[torch.Tensor, dict[str, Optional[torch.Tensor]]]]:
    """Fuse stochastic fake quant B with the minimal dq_delta basic stats.

    Python reference for the forward value is fake_quantize_levels(...,
    mode="stoch"). The returned tensor is the dequantized forward value; STE is
    intentionally applied by the caller so gradients stay identical to the
    normal Python/Triton path.

    Collected stats are the qerr-basic subset:
        numel, clip_count, sumsq, xq_sumsq, xxq_sum
    """
    if not _TRITON_AVAILABLE:
        _warn_once(
            "triton_import",
            f"Triton fused fake quant requested, but Triton is unavailable: {_TRITON_IMPORT_ERROR}",
        )
        return None

    checked = _check_fake_quant_inputs(x, scale)
    if checked is None:
        return None
    scale_flat, scale_numel, dim1, dim2, dim3 = checked

    if rand is not None:
        if rand.shape != x.shape or not rand.is_cuda or not rand.is_contiguous():
            return None
        rand_t = rand.to(device=x.device, dtype=torch.float32)
    else:
        rand_t = torch.rand_like(x, dtype=torch.float32)

    n_elements = x.numel()
    if n_elements <= 0:
        return None
    out = torch.empty_like(x)
    block_size = 256
    n_blocks = triton.cdiv(n_elements, block_size)
    stats = torch.empty((n_blocks, 5), device=x.device, dtype=torch.float32)

    try:
        _fake_quantize_levels_stoch_stats_kernel[(n_blocks,)](
            x,
            scale_flat,
            out,
            rand_t,
            stats,
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
        _warn_once("triton_fused_stats_kernel", f"Triton fused fake quant stats failed; falling back: {e}")
        return None

    sums = stats.sum(dim=0)
    return out, {
        "numel": sums[0].reshape(1),
        "clip_count": sums[1].reshape(1),
        "zero_count": None,
        "near_zero_count": None,
        "sumsq": sums[2].reshape(1),
        "xq_sumsq": sums[3].reshape(1),
        "xxq_sum": sums[4].reshape(1),
        "absmax": None,
        "scale_min": None,
        "scale_max": None,
        "scale_sum": None,
        "scale_count": None,
    }


def triton_collect_fake_quant_stats(
    x: torch.Tensor,
    quantized: torch.Tensor,
    *,
    scale: torch.Tensor,
    qmax: int,
    collect_zero: bool,
    collect_near_zero: bool,
    collect_full: bool,
    collect_detail: bool = True,
) -> Optional[dict[str, Optional[torch.Tensor]]]:
    """Collect dq_delta fake-quant stats with a small Triton reduction.

    This mirrors LoRAModule._record_dq_stats for the common bits/channel/rms
    path, except clip/round error parts are intentionally not handled here.
    The quantized tensor is supplied by the normal fake_quantize_levels path so
    stats steps can observe the same forward output used for training.
    """
    if not _TRITON_AVAILABLE:
        return None
    if not x.is_cuda or not quantized.is_cuda or not scale.is_cuda:
        return None
    if x.shape != quantized.shape:
        return None
    if x.ndim not in (2, 3, 4):
        return None
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if quantized.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return None
    if not x.is_contiguous() or not quantized.is_contiguous() or not scale.is_contiguous():
        return None

    scale_flat = scale.reshape(-1)
    scale_numel = scale_flat.numel()
    if scale_numel not in (1, x.shape[1] if x.ndim in (2, 4) else x.shape[2]):
        return None

    n_elements = x.numel()
    if n_elements <= 0:
        return None
    block_size = 256
    n_blocks = triton.cdiv(n_elements, block_size)
    stats = torch.empty((n_blocks, 8), device=x.device, dtype=torch.float32)
    dim1 = x.shape[1] if x.ndim >= 2 else 1
    dim2 = x.shape[2] if x.ndim >= 3 else 1
    dim3 = x.shape[3] if x.ndim >= 4 else 1

    try:
        _fake_quant_stats_kernel[(n_blocks,)](
            x,
            quantized,
            scale_flat,
            stats,
            n_elements,
            scale_numel,
            dim1,
            dim2,
            dim3,
            x.ndim,
            int(qmax),
            bool(collect_zero),
            bool(collect_near_zero),
            bool(collect_full),
            bool(collect_detail),
            BLOCK_SIZE=block_size,
        )
    except Exception as e:
        _warn_once("triton_stats_kernel", f"Triton fake quant stats failed; falling back to PyTorch: {e}")
        return None

    sums = stats.sum(dim=0)
    absmax = stats[:, 7].max().reshape(1) if collect_detail else None
    return {
        "numel": sums[0].reshape(1),
        "clip_count": sums[1].reshape(1),
        "zero_count": sums[2].reshape(1) if collect_zero else None,
        "near_zero_count": sums[3].reshape(1) if collect_near_zero else None,
        "sumsq": sums[4].reshape(1) if collect_full else None,
        "xq_sumsq": sums[5].reshape(1) if collect_full else None,
        "xxq_sum": sums[6].reshape(1) if collect_full else None,
        "absmax": absmax,
        "scale_min": scale.min() if collect_detail else None,
        "scale_max": scale.max() if collect_detail else None,
        "scale_sum": scale.sum() if collect_detail else None,
        "scale_count": torch.tensor(float(scale.numel()), device=x.device, dtype=torch.float32) if collect_detail else None,
    }
