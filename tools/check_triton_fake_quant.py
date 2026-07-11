from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.checkpoint import checkpoint

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover - diagnostic script
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = e
else:
    TRITON_IMPORT_ERROR = None

import library.rounding_util as rounding_util
import networks.lora as lora_impl
from library.rounding_util import compute_scale_bits, fake_quantize_levels
try:
    from library.triton_quant import (
        _FUSED_STATS_LARGE_MIN_ELEMENTS,
        triton_fake_quantize_levels_stoch,
        triton_fake_quantize_levels_stoch_with_stats,
    )
except Exception:
    _FUSED_STATS_LARGE_MIN_ELEMENTS = None
    triton_fake_quantize_levels_stoch = None
    triton_fake_quantize_levels_stoch_with_stats = None


if triton is not None:

    @triton.jit
    def _debug_fake_quant_stoch_kernel(
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

    @triton.jit
    def _debug_clip_count_kernel(
        x_ptr,
        scale_ptr,
        stats_ptr,
        n_elements: tl.constexpr,
        scale_numel: tl.constexpr,
        dim1: tl.constexpr,
        dim2: tl.constexpr,
        dim3: tl.constexpr,
        ndim: tl.constexpr,
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
        clip_count = tl.sum(((tl.abs(y) >= qmax) & mask).to(tl.float32), axis=0)
        tl.store(stats_ptr + pid, clip_count)


@dataclass
class CaseResult:
    name: str
    dtype: str
    shape: str
    scale: str
    div: str
    equal: bool
    mismatches: int
    max_abs_diff: float
    mean_abs_diff: float


@dataclass
class ScaleResult:
    dtype: str
    shape: str
    max_abs_diff: float
    max_rel_diff: float
    mean_abs_diff: float


@dataclass
class FusedStatsResult:
    dtype: str
    shape: str
    div: str
    out_equal: bool
    out_mismatches: int
    out_max_abs_diff: float
    clip_count_abs_diff: float
    sumsq_rel_diff: float
    xq_sumsq_rel_diff: float
    xxq_sum_rel_diff: float


@dataclass
class EndToEndResult:
    name: str
    dtype: str
    shape: str
    scale: str
    equal: bool
    mismatches: int
    max_abs_diff: float
    mean_abs_diff: float
    scale_max_abs_diff: float
    scale_max_rel_diff: float


@dataclass
class RngResult:
    name: str
    dtype: str
    shape: str
    scale: str
    rng_after_equal: bool
    out_equal: bool
    mismatches: int
    max_abs_diff: float
    before_hash: str
    after_ref_hash: str
    after_tri_hash: str


@dataclass
class FallbackResult:
    out_equal: bool
    rand_equal: bool
    rng_after_equal: bool
    mismatches: int
    max_abs_diff: float


@dataclass
class FusedRouteResult:
    out_equal: bool
    rand_equal: bool
    rng_after_equal: bool
    stats_numel_equal: bool
    counters_equal: bool
    mismatches: int
    max_abs_diff: float


@dataclass
class FusedSteResult:
    ok: bool
    grad_min: float
    grad_max: float
    grad_mean: float
    stats_numel_equal: bool
    counters_equal: bool


@dataclass
class CheckpointTraceResult:
    use_reentrant: bool
    calls_after_forward: int
    calls_after_backward: int
    recompute_calls: int


@dataclass
class MutationResult:
    name: str
    dtype: str
    shape: str
    scale: str
    x_mutated: bool
    max_abs_diff: float
    out_aliases_x: bool
    x_ptr: int
    out_ptr: int
    version_before: int
    version_after: int


@dataclass
class CompareStats:
    equal: bool
    mismatches: int
    max_abs_diff: float
    fixed_tri_gt_ref: int
    fixed_tri_lt_ref: int
    signed_sum: float
    signed_mean_all: float
    signed_mean_mismatch: float
    mean_abs_out_diff: float
    mse_out_diff: float
    ref_quant_noise_l1: float
    ref_quant_noise_mse: float
    extra_l1_ratio: float
    extra_mse_ratio: float
    floor_diff_count: int
    compare_diff_count: int
    clamp_diff_count: int
    quant_index_diff_count: int
    same_index_value_diff_count: int
    unclassified_mismatch_count: int
    saturation_ref_count: int
    saturation_tri_count: int
    rand_margin_min: float
    rand_margin_median: float
    rand_margin_mean: float
    integer_boundary_min: float
    integer_boundary_median: float
    integer_boundary_mean: float


@dataclass
class CaptureResult:
    path: str
    global_step: str
    global_step_1based: str
    capture_seen: str
    dtype: str
    shape: str
    stride: str
    scale_shape: str
    scale_stride: str
    x_contig: bool
    scale_contig: bool
    fixed_default: CompareStats
    fixed_divrn: CompareStats
    rng_after_equal: bool
    production_equal: bool
    production_mismatches: int
    production_max_abs_diff: float
    x_mutated: bool
    x_mutation_max_abs_diff: float
    out_aliases_x: bool
    version_before: int
    version_after: int
    e2e_default: CompareStats
    e2e_divrn: CompareStats
    e2e_scale_max_rel_diff: float


def _scale_to_shape(scale_flat: torch.Tensor, x: torch.Tensor, scale_kind: str) -> torch.Tensor:
    if scale_kind == "scalar":
        return scale_flat[:1].contiguous()
    if x.ndim == 2:
        return scale_flat.view(1, x.shape[1]).contiguous()
    if x.ndim == 3:
        return scale_flat.view(1, 1, x.shape[2]).contiguous()
    if x.ndim == 4:
        return scale_flat.view(1, x.shape[1], 1, 1).contiguous()
    raise ValueError(f"unsupported ndim: {x.ndim}")


def _make_x(shape: tuple[int, ...], dtype: torch.dtype, scale: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "random":
        return (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.08).to(dtype).contiguous()

    # Stress values near integer boundaries of y = x / scale, where tiny
    # division differences can change floor(y).
    if scale.numel() == 1:
        scale_b = scale.reshape([1] * len(shape))
    elif len(shape) == 2:
        scale_b = scale.view(1, shape[1])
    elif len(shape) == 3:
        scale_b = scale.view(1, 1, shape[2])
    else:
        scale_b = scale.view(1, shape[1], 1, 1)
    base = torch.randint(-8, 9, shape, device="cuda", dtype=torch.int32).to(torch.float32)
    offsets = torch.tensor([-1e-5, -1e-6, 0.0, 1e-6, 1e-5, 0.49999, 0.5, 0.50001], device="cuda")
    frac = offsets[torch.arange(torch.tensor(shape).prod().item(), device="cuda") % offsets.numel()].view(shape)
    return ((base + frac) * scale_b).to(dtype).contiguous()


def ref_fake_quant_stoch_with_rand(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    rand: torch.Tensor,
) -> torch.Tensor:
    s = scale.to(device=x.device, dtype=torch.float32)
    y = x.to(torch.float32) / s
    q_floor = torch.floor(y)
    probs = (y - q_floor).clamp(0.0, 1.0)
    q = q_floor + (rand.to(torch.float32) < probs).to(torch.float32)
    q = torch.clamp(q, qmin, qmax)
    return (q * s).to(x.dtype)


def debug_triton_fake_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    rand: torch.Tensor,
    *,
    use_div_rn: bool,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError(f"Triton import failed: {TRITON_IMPORT_ERROR}")
    scale_flat = scale.reshape(-1).contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    block_size = 256
    dim1 = x.shape[1] if x.ndim >= 2 else 1
    dim2 = x.shape[2] if x.ndim >= 3 else 1
    dim3 = x.shape[3] if x.ndim >= 4 else 1
    _debug_fake_quant_stoch_kernel[(triton.cdiv(n_elements, block_size),)](
        x,
        scale_flat,
        out,
        rand.contiguous(),
        n_elements,
        scale_flat.numel(),
        dim1,
        dim2,
        dim3,
        x.ndim,
        qmin,
        qmax,
        use_div_rn,
        BLOCK_SIZE=block_size,
    )
    return out


def debug_triton_clip_count(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmax: int,
    *,
    use_div_rn: bool,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError(f"Triton import failed: {TRITON_IMPORT_ERROR}")
    scale_flat = scale.reshape(-1).contiguous()
    n_elements = x.numel()
    block_size = 256
    n_blocks = triton.cdiv(n_elements, block_size)
    stats = torch.empty((n_blocks,), device=x.device, dtype=torch.float32)
    dim1 = x.shape[1] if x.ndim >= 2 else 1
    dim2 = x.shape[2] if x.ndim >= 3 else 1
    dim3 = x.shape[3] if x.ndim >= 4 else 1
    _debug_clip_count_kernel[(n_blocks,)](
        x,
        scale_flat,
        stats,
        n_elements,
        scale_flat.numel(),
        dim1,
        dim2,
        dim3,
        x.ndim,
        qmax,
        use_div_rn,
        BLOCK_SIZE=block_size,
    )
    return stats.sum().reshape(1)


def _case_results(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    scale_kind: str,
    x_mode: str,
    use_div_rn: bool,
) -> CaseResult:
    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), scale_kind)
    x = _make_x(shape, dtype, scale, x_mode)
    rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()

    ref = ref_fake_quant_stoch_with_rand(x, scale, -127, 127, rand)
    tri = debug_triton_fake_quant(x, scale, -127, 127, rand, use_div_rn=use_div_rn)
    diff = (ref.to(torch.float32) - tri.to(torch.float32)).abs()
    mismatches = int((ref != tri).sum().item())
    return CaseResult(
        name=x_mode,
        dtype=str(dtype).replace("torch.", ""),
        shape=str(tuple(shape)),
        scale=scale_kind,
        div="div_rn" if use_div_rn else "default",
        equal=bool(torch.equal(ref, tri)),
        mismatches=mismatches,
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
    )


def _supported_dtypes() -> list[torch.dtype]:
    dtypes = [torch.float16, torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.insert(1, torch.bfloat16)
    return dtypes


def _rng_hash(state: torch.Tensor | None = None) -> str:
    if state is None:
        state = torch.cuda.get_rng_state()
    return hashlib.sha256(state.cpu().numpy().tobytes()).hexdigest()[:16]


def run_forward_checks(include_div_rn: bool) -> list[CaseResult]:
    torch.manual_seed(1234)
    shapes = [(17, 13), (3, 19, 11), (2, 7, 5, 3)]
    scale_kinds = ["scalar", "channel"]
    modes = ["random", "boundary"]
    div_modes = [False, True] if include_div_rn else [False]

    results: list[CaseResult] = []
    for dtype in _supported_dtypes():
        for shape in shapes:
            for scale_kind in scale_kinds:
                for mode in modes:
                    for use_div_rn in div_modes:
                        results.append(
                            _case_results(
                                dtype=dtype,
                                shape=shape,
                                scale_kind=scale_kind,
                                x_mode=mode,
                                use_div_rn=use_div_rn,
                            )
                        )
    return results


def _end_to_end_case(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    scale_kind: str,
    x_mode: str,
) -> EndToEndResult:
    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale_seed = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), scale_kind)
    x = _make_x(shape, dtype, scale_seed, x_mode)

    if scale_kind == "scalar":
        scale_ref = compute_scale_bits(
            x,
            bits=8,
            granularity="tensor",
            stat="rms",
            range_mul=3.0,
            use_triton=False,
        )
        # Triton acceleration is currently channel/rms only. Keep scalar
        # cases in the report as a PyTorch/PyTorch sanity baseline.
        scale_tri = compute_scale_bits(
            x,
            bits=8,
            granularity="tensor",
            stat="rms",
            range_mul=3.0,
            use_triton=True,
        )
    else:
        scale_ref = compute_scale_bits(
            x,
            bits=8,
            granularity="channel",
            stat="rms",
            range_mul=3.0,
            use_triton=False,
        )
        scale_tri = compute_scale_bits(
            x,
            bits=8,
            granularity="channel",
            stat="rms",
            range_mul=3.0,
            use_triton=True,
        )

    rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()
    ref = ref_fake_quant_stoch_with_rand(x, scale_ref, -127, 127, rand)
    tri = debug_triton_fake_quant(x, scale_tri, -127, 127, rand, use_div_rn=False)
    diff = (ref.to(torch.float32) - tri.to(torch.float32)).abs()
    scale_diff = (scale_ref.to(torch.float32) - scale_tri.to(torch.float32)).abs()
    scale_rel = scale_diff / scale_ref.to(torch.float32).abs().clamp_min(1e-30)
    return EndToEndResult(
        name=x_mode,
        dtype=str(dtype).replace("torch.", ""),
        shape=str(tuple(shape)),
        scale=scale_kind,
        equal=bool(torch.equal(ref, tri)),
        mismatches=int((ref != tri).sum().item()),
        max_abs_diff=float(diff.max().item()),
        mean_abs_diff=float(diff.mean().item()),
        scale_max_abs_diff=float(scale_diff.max().item()),
        scale_max_rel_diff=float(scale_rel.max().item()),
    )


def run_end_to_end_checks() -> list[EndToEndResult]:
    torch.manual_seed(3456)
    results: list[EndToEndResult] = []
    for dtype in _supported_dtypes():
        for shape in [(17, 13), (3, 19, 11), (2, 7, 5, 3)]:
            for scale_kind in ["scalar", "channel"]:
                for mode in ["random", "boundary"]:
                    results.append(
                        _end_to_end_case(
                            dtype=dtype,
                            shape=shape,
                            scale_kind=scale_kind,
                            x_mode=mode,
                        )
                    )
    return results


def _production_rng_case(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    scale_kind: str,
    x_mode: str,
) -> RngResult:
    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), scale_kind)
    x = _make_x(shape, dtype, scale, x_mode)

    # Compile/warm the Triton path before taking the RNG snapshot. This keeps
    # first-call compilation separate from the RNG consumption comparison.
    _ = fake_quantize_levels(x, scale=scale, qmin=-127, qmax=127, mode="stoch", use_triton=True)
    torch.cuda.synchronize()

    state0 = torch.cuda.get_rng_state()
    before_hash = _rng_hash(state0)

    torch.cuda.set_rng_state(state0)
    out_ref = fake_quantize_levels(x, scale=scale, qmin=-127, qmax=127, mode="stoch", use_triton=False)
    torch.cuda.synchronize()
    after_ref = torch.cuda.get_rng_state()

    torch.cuda.set_rng_state(state0)
    out_tri = fake_quantize_levels(x, scale=scale, qmin=-127, qmax=127, mode="stoch", use_triton=True)
    torch.cuda.synchronize()
    after_tri = torch.cuda.get_rng_state()

    diff = (out_ref.to(torch.float32) - out_tri.to(torch.float32)).abs()
    return RngResult(
        name=x_mode,
        dtype=str(dtype).replace("torch.", ""),
        shape=str(tuple(shape)),
        scale=scale_kind,
        rng_after_equal=bool(torch.equal(after_ref, after_tri)),
        out_equal=bool(torch.equal(out_ref, out_tri)),
        mismatches=int((out_ref != out_tri).sum().item()),
        max_abs_diff=float(diff.max().item()),
        before_hash=before_hash,
        after_ref_hash=_rng_hash(after_ref),
        after_tri_hash=_rng_hash(after_tri),
    )


def _production_compare_existing(x: torch.Tensor, scale: torch.Tensor, qmin: int, qmax: int) -> tuple[bool, bool, int, float]:
    _ = fake_quantize_levels(x, scale=scale, qmin=qmin, qmax=qmax, mode="stoch", use_triton=True)
    torch.cuda.synchronize()

    state0 = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state0)
    out_ref = fake_quantize_levels(x, scale=scale, qmin=qmin, qmax=qmax, mode="stoch", use_triton=False)
    torch.cuda.synchronize()
    after_ref = torch.cuda.get_rng_state()

    torch.cuda.set_rng_state(state0)
    out_tri = fake_quantize_levels(x, scale=scale, qmin=qmin, qmax=qmax, mode="stoch", use_triton=True)
    torch.cuda.synchronize()
    after_tri = torch.cuda.get_rng_state()

    diff = (out_ref.to(torch.float32) - out_tri.to(torch.float32)).abs()
    return (
        bool(torch.equal(after_ref, after_tri)),
        bool(torch.equal(out_ref, out_tri)),
        int((out_ref != out_tri).sum().item()),
        float(diff.max().item()),
    )


def _mutation_check_existing(
    x: torch.Tensor,
    scale: torch.Tensor,
    qmin: int,
    qmax: int,
    *,
    name: str,
    scale_kind: str,
) -> MutationResult:
    torch.cuda.synchronize()
    x_before = x.detach().clone()
    version_before = int(getattr(x, "_version", -1))
    x_ptr = int(x.data_ptr())

    out = fake_quantize_levels(x, scale=scale, qmin=qmin, qmax=qmax, mode="stoch", use_triton=True)
    torch.cuda.synchronize()

    diff = (x.to(torch.float32) - x_before.to(torch.float32)).abs()
    version_after = int(getattr(x, "_version", -1))
    return MutationResult(
        name=name,
        dtype=str(x.dtype).replace("torch.", ""),
        shape=str(tuple(x.shape)),
        scale=scale_kind,
        x_mutated=not bool(torch.equal(x, x_before)),
        max_abs_diff=float(diff.max().item()),
        out_aliases_x=bool(out.data_ptr() == x.data_ptr()),
        x_ptr=x_ptr,
        out_ptr=int(out.data_ptr()),
        version_before=version_before,
        version_after=version_after,
    )


def _mutation_case(
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...],
    scale_kind: str,
    x_mode: str,
) -> MutationResult:
    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), scale_kind)
    x = _make_x(shape, dtype, scale, x_mode)

    # Compile/warm the Triton path first so the mutation check only observes
    # the production wrapper call itself.
    _ = fake_quantize_levels(x, scale=scale, qmin=-127, qmax=127, mode="stoch", use_triton=True)
    torch.cuda.synchronize()

    return _mutation_check_existing(
        x,
        scale,
        -127,
        127,
        name=x_mode,
        scale_kind=scale_kind,
    )


def run_mutation_checks() -> list[MutationResult]:
    torch.manual_seed(4567)
    results: list[MutationResult] = []
    for dtype in _supported_dtypes():
        for shape in [(17, 13), (3, 19, 11), (2, 7, 5, 3)]:
            for scale_kind in ["scalar", "channel"]:
                for mode in ["random", "boundary"]:
                    results.append(
                        _mutation_case(
                            dtype=dtype,
                            shape=shape,
                            scale_kind=scale_kind,
                            x_mode=mode,
                        )
                    )
    return results


def _signed_diff_stats(ref: torch.Tensor, tri: torch.Tensor) -> tuple[int, int, float, float, float]:
    signed = tri.to(torch.float32) - ref.to(torch.float32)
    nonzero = signed != 0
    mismatch_count = int(nonzero.sum().item())
    tri_gt_ref = int((signed > 0).sum().item())
    tri_lt_ref = int((signed < 0).sum().item())
    signed_sum = float(signed.sum().item())
    signed_mean_all = float(signed.mean().item())
    if mismatch_count > 0:
        signed_mean_mismatch = float(signed[nonzero].mean().item())
    else:
        signed_mean_mismatch = 0.0
    return tri_gt_ref, tri_lt_ref, signed_sum, signed_mean_all, signed_mean_mismatch


def _broadcast_scale(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if scale.numel() == 1:
        return scale.reshape([1] * x.ndim)
    return scale


def _masked_stat(values: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    selected = values[mask]
    if selected.numel() == 0:
        return 0.0, 0.0, 0.0
    return (
        float(selected.min().item()),
        float(selected.median().item()),
        float(selected.mean().item()),
    )


def _compare_quant_outputs(
    *,
    x: torch.Tensor,
    scale_ref: torch.Tensor,
    scale_tri: torch.Tensor,
    qmin: int,
    qmax: int,
    rand: torch.Tensor,
    ref: torch.Tensor,
    tri: torch.Tensor,
) -> CompareStats:
    ref_f = ref.to(torch.float32)
    tri_f = tri.to(torch.float32)
    x_f = x.to(torch.float32)
    signed = tri_f - ref_f
    abs_diff = signed.abs()
    mismatch = ref != tri
    mismatches = int(mismatch.sum().item())
    tri_gt_ref = int((signed > 0).sum().item())
    tri_lt_ref = int((signed < 0).sum().item())
    signed_sum = float(signed.sum().item())
    signed_mean_all = float(signed.mean().item())
    signed_mean_mismatch = float(signed[mismatch].mean().item()) if mismatches > 0 else 0.0

    ref_noise = ref_f - x_f
    ref_noise_abs = ref_noise.abs()
    ref_noise_sq = ref_noise * ref_noise
    out_diff_sq = signed * signed
    mean_abs_out_diff = float(abs_diff.mean().item())
    mse_out_diff = float(out_diff_sq.mean().item())
    ref_quant_noise_l1 = float(ref_noise_abs.mean().item())
    ref_quant_noise_mse = float(ref_noise_sq.mean().item())
    eps = 1e-30
    extra_l1_ratio = mean_abs_out_diff / max(ref_quant_noise_l1, eps)
    extra_mse_ratio = mse_out_diff / max(ref_quant_noise_mse, eps)

    scale_ref_b = _broadcast_scale(scale_ref.to(device=x.device, dtype=torch.float32), x)
    scale_tri_b = _broadcast_scale(scale_tri.to(device=x.device, dtype=torch.float32), x)
    y_ref = x_f / scale_ref_b
    y_tri = x_f / scale_tri_b
    floor_ref = torch.floor(y_ref)
    floor_tri = torch.floor(y_tri)
    probs_ref = (y_ref - floor_ref).clamp(0.0, 1.0)
    probs_tri = (y_tri - floor_tri).clamp(0.0, 1.0)
    incr_ref = rand < probs_ref
    incr_tri = rand < probs_tri
    q_ref_raw = floor_ref + incr_ref.to(torch.float32)
    q_tri_raw = floor_tri + incr_tri.to(torch.float32)
    q_ref = torch.clamp(q_ref_raw, qmin, qmax)
    q_tri = torch.clamp(q_tri_raw, qmin, qmax)
    q_index_ref_from_out = torch.round(ref_f / scale_ref_b)
    q_index_tri_from_out = torch.round(tri_f / scale_tri_b)
    quant_index_diff = (q_index_ref_from_out != q_index_tri_from_out) & mismatch
    same_index_value_diff = (q_index_ref_from_out == q_index_tri_from_out) & mismatch
    clamp_diff = (q_ref != q_tri) & mismatch
    floor_diff = (floor_ref != floor_tri) & mismatch
    compare_diff = (floor_ref == floor_tri) & (incr_ref != incr_tri) & mismatch
    classified = floor_diff | compare_diff | clamp_diff | quant_index_diff | same_index_value_diff
    saturation_ref = (q_ref_raw < qmin) | (q_ref_raw > qmax)
    saturation_tri = (q_tri_raw < qmin) | (q_tri_raw > qmax)
    rand_margin = (rand - probs_ref).abs()
    frac_ref = y_ref - floor_ref
    integer_boundary_distance = torch.minimum(frac_ref.abs(), (1.0 - frac_ref).abs())
    rand_min, rand_median, rand_mean = _masked_stat(rand_margin, mismatch)
    boundary_min, boundary_median, boundary_mean = _masked_stat(integer_boundary_distance, mismatch)

    return CompareStats(
        equal=bool(torch.equal(ref, tri)),
        mismatches=mismatches,
        max_abs_diff=float(abs_diff.max().item()),
        fixed_tri_gt_ref=tri_gt_ref,
        fixed_tri_lt_ref=tri_lt_ref,
        signed_sum=signed_sum,
        signed_mean_all=signed_mean_all,
        signed_mean_mismatch=signed_mean_mismatch,
        mean_abs_out_diff=mean_abs_out_diff,
        mse_out_diff=mse_out_diff,
        ref_quant_noise_l1=ref_quant_noise_l1,
        ref_quant_noise_mse=ref_quant_noise_mse,
        extra_l1_ratio=extra_l1_ratio,
        extra_mse_ratio=extra_mse_ratio,
        floor_diff_count=int(floor_diff.sum().item()),
        compare_diff_count=int(compare_diff.sum().item()),
        clamp_diff_count=int(clamp_diff.sum().item()),
        quant_index_diff_count=int(quant_index_diff.sum().item()),
        same_index_value_diff_count=int(same_index_value_diff.sum().item()),
        unclassified_mismatch_count=int((mismatch & ~classified).sum().item()),
        saturation_ref_count=int(saturation_ref.sum().item()),
        saturation_tri_count=int(saturation_tri.sum().item()),
        rand_margin_min=rand_min,
        rand_margin_median=rand_median,
        rand_margin_mean=rand_mean,
        integer_boundary_min=boundary_min,
        integer_boundary_median=boundary_median,
        integer_boundary_mean=boundary_mean,
    )


def _empty_compare_stats() -> CompareStats:
    return CompareStats(
        equal=False,
        mismatches=-1,
        max_abs_diff=float("nan"),
        fixed_tri_gt_ref=0,
        fixed_tri_lt_ref=0,
        signed_sum=float("nan"),
        signed_mean_all=float("nan"),
        signed_mean_mismatch=float("nan"),
        mean_abs_out_diff=float("nan"),
        mse_out_diff=float("nan"),
        ref_quant_noise_l1=float("nan"),
        ref_quant_noise_mse=float("nan"),
        extra_l1_ratio=float("nan"),
        extra_mse_ratio=float("nan"),
        floor_diff_count=0,
        compare_diff_count=0,
        clamp_diff_count=0,
        quant_index_diff_count=0,
        same_index_value_diff_count=0,
        unclassified_mismatch_count=0,
        saturation_ref_count=0,
        saturation_tri_count=0,
        rand_margin_min=float("nan"),
        rand_margin_median=float("nan"),
        rand_margin_mean=float("nan"),
        integer_boundary_min=float("nan"),
        integer_boundary_median=float("nan"),
        integer_boundary_mean=float("nan"),
    )


def _capture_case(path: Path) -> CaptureResult:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x_cpu = payload["x"]
    scale_cpu = payload["scale"]
    x = x_cpu.to("cuda")
    scale = scale_cpu.to("cuda", dtype=torch.float32)
    qmin = int(payload["qmin"])
    qmax = int(payload["qmax"])

    fixed_default = _empty_compare_stats()
    fixed_divrn = _empty_compare_stats()
    if x.is_contiguous() and scale.is_contiguous() and x.ndim in (2, 3, 4):
        rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()
        ref = ref_fake_quant_stoch_with_rand(x, scale, qmin, qmax, rand)
        tri_default = debug_triton_fake_quant(x, scale, qmin, qmax, rand, use_div_rn=False)
        tri_divrn = debug_triton_fake_quant(x, scale, qmin, qmax, rand, use_div_rn=True)
        fixed_default = _compare_quant_outputs(
            x=x,
            scale_ref=scale,
            scale_tri=scale,
            qmin=qmin,
            qmax=qmax,
            rand=rand,
            ref=ref,
            tri=tri_default,
        )
        fixed_divrn = _compare_quant_outputs(
            x=x,
            scale_ref=scale,
            scale_tri=scale,
            qmin=qmin,
            qmax=qmax,
            rand=rand,
            ref=ref,
            tri=tri_divrn,
        )

    rng_after_equal, production_equal, production_mismatches, production_max_abs_diff = _production_compare_existing(
        x, scale, qmin, qmax
    )
    mutation = _mutation_check_existing(
        x,
        scale,
        qmin,
        qmax,
        name=path.name,
        scale_kind="channel" if scale.numel() != 1 else "scalar",
    )
    scale_ref = compute_scale_bits(
        x,
        bits=8,
        granularity="channel" if scale.numel() != 1 else "tensor",
        stat="rms",
        range_mul=3.0,
        use_triton=False,
    )
    scale_tri = compute_scale_bits(
        x,
        bits=8,
        granularity="channel" if scale.numel() != 1 else "tensor",
        stat="rms",
        range_mul=3.0,
        use_triton=True,
    )
    rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()
    e2e_ref = ref_fake_quant_stoch_with_rand(x, scale_ref, qmin, qmax, rand)
    e2e_tri_default = debug_triton_fake_quant(x, scale_tri, qmin, qmax, rand, use_div_rn=False)
    e2e_tri_divrn = debug_triton_fake_quant(x, scale_tri, qmin, qmax, rand, use_div_rn=True)
    e2e_default = _compare_quant_outputs(
        x=x,
        scale_ref=scale_ref,
        scale_tri=scale_tri,
        qmin=qmin,
        qmax=qmax,
        rand=rand,
        ref=e2e_ref,
        tri=e2e_tri_default,
    )
    e2e_divrn = _compare_quant_outputs(
        x=x,
        scale_ref=scale_ref,
        scale_tri=scale_tri,
        qmin=qmin,
        qmax=qmax,
        rand=rand,
        ref=e2e_ref,
        tri=e2e_tri_divrn,
    )
    e2e_scale_diff = (scale_ref.to(torch.float32) - scale_tri.to(torch.float32)).abs()
    e2e_scale_rel = e2e_scale_diff / scale_ref.to(torch.float32).abs().clamp_min(1e-30)
    return CaptureResult(
        path=str(path),
        global_step=str(payload.get("global_step", "")),
        global_step_1based=str(payload.get("global_step_1based", "")),
        capture_seen=str(payload.get("capture_seen", "")),
        dtype=str(x.dtype).replace("torch.", ""),
        shape=str(tuple(x.shape)),
        stride=str(tuple(x.stride())),
        scale_shape=str(tuple(scale.shape)),
        scale_stride=str(tuple(scale.stride())),
        x_contig=bool(x.is_contiguous()),
        scale_contig=bool(scale.is_contiguous()),
        fixed_default=fixed_default,
        fixed_divrn=fixed_divrn,
        rng_after_equal=rng_after_equal,
        production_equal=production_equal,
        production_mismatches=production_mismatches,
        production_max_abs_diff=production_max_abs_diff,
        x_mutated=mutation.x_mutated,
        x_mutation_max_abs_diff=mutation.max_abs_diff,
        out_aliases_x=mutation.out_aliases_x,
        version_before=mutation.version_before,
        version_after=mutation.version_after,
        e2e_default=e2e_default,
        e2e_divrn=e2e_divrn,
        e2e_scale_max_rel_diff=float(e2e_scale_rel.max().item()),
    )


def run_capture_checks(capture_dir: Path) -> list[CaptureResult]:
    files = sorted(capture_dir.glob("dq_fake_quant_capture_*.pt"))
    return [_capture_case(path) for path in files]


def run_production_rng_checks() -> list[RngResult]:
    torch.manual_seed(9012)
    results: list[RngResult] = []
    for dtype in _supported_dtypes():
        for shape in [(17, 13), (3, 19, 11), (2, 7, 5, 3)]:
            for scale_kind in ["scalar", "channel"]:
                for mode in ["random", "boundary"]:
                    results.append(
                        _production_rng_case(
                            dtype=dtype,
                            shape=shape,
                            scale_kind=scale_kind,
                            x_mode=mode,
                        )
                    )
    return results


def run_ste_check() -> tuple[bool, float, float, float]:
    x = torch.randn((3, 19, 11), device="cuda", dtype=torch.float16, requires_grad=True).contiguous()
    with torch.no_grad():
        scale = compute_scale_bits(
            x,
            bits=8,
            granularity="channel",
            stat="rms",
            range_mul=3.0,
            use_triton=True,
        )
    out = fake_quantize_levels(x, scale=scale, qmin=-127, qmax=127, mode="stoch", use_triton=True)
    loss = out.float().sum()
    loss.backward()
    grad = x.grad.detach().float()
    return bool(torch.allclose(grad, torch.ones_like(grad))), float(grad.min()), float(grad.max()), float(grad.mean())


def run_scale_checks() -> list[ScaleResult]:
    torch.manual_seed(5678)
    results: list[ScaleResult] = []
    for dtype in _supported_dtypes():
        for shape in [(17, 13), (3, 19, 11), (2, 7, 5, 3)]:
            x = (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.08).to(dtype).contiguous()
            scale_ref = compute_scale_bits(
                x,
                bits=8,
                granularity="channel",
                stat="rms",
                range_mul=3.0,
                use_triton=False,
            )
            scale_tri = compute_scale_bits(
                x,
                bits=8,
                granularity="channel",
                stat="rms",
                range_mul=3.0,
                use_triton=True,
            )
            diff = (scale_ref - scale_tri).abs()
            rel = diff / scale_ref.abs().clamp_min(1e-30)
            results.append(
                ScaleResult(
                    dtype=str(dtype).replace("torch.", ""),
                    shape=str(tuple(shape)),
                    max_abs_diff=float(diff.max().item()),
                    max_rel_diff=float(rel.max().item()),
                    mean_abs_diff=float(diff.mean().item()),
                )
            )
    return results


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a - b).abs() / a.abs().clamp_min(1e-30)).max().item())


def _fused_stats_case(*, dtype: torch.dtype, shape: tuple[int, ...], use_div_rn: bool) -> FusedStatsResult:
    if triton_fake_quantize_levels_stoch is None or triton_fake_quantize_levels_stoch_with_stats is None:
        raise RuntimeError("production Triton fake-quant wrappers are unavailable")

    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), "channel").contiguous()
    x = _make_x(shape, dtype, scale, "random").contiguous()
    rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()

    tri = triton_fake_quantize_levels_stoch(
        x,
        scale=scale,
        qmin=-127,
        qmax=127,
        use_div_rn=use_div_rn,
        rand=rand,
    )
    if tri is None:
        raise RuntimeError(f"production B returned None for shape={shape} dtype={dtype}")
    fused = triton_fake_quantize_levels_stoch_with_stats(
        x,
        scale=scale,
        qmin=-127,
        qmax=127,
        use_div_rn=use_div_rn,
        rand=rand,
    )
    if fused is None:
        raise RuntimeError(f"fused stats returned None for shape={shape} dtype={dtype}")
    fused_out, packed_stats = fused

    clip_count_ref = debug_triton_clip_count(x, scale, 127, use_div_rn=use_div_rn)
    x_fp32 = x.to(torch.float32)
    q_fp32 = fused_out.to(torch.float32)
    sumsq_ref = torch.dot(x_fp32.reshape(-1), x_fp32.reshape(-1)).reshape(1)
    xq_sumsq_ref = torch.dot(q_fp32.reshape(-1), q_fp32.reshape(-1)).reshape(1)
    xxq_sum_ref = torch.dot(x_fp32.reshape(-1), q_fp32.reshape(-1)).reshape(1)

    out_diff = (tri.to(torch.float32) - fused_out.to(torch.float32)).abs()
    return FusedStatsResult(
        dtype=str(dtype).replace("torch.", ""),
        shape=str(tuple(shape)),
        div="div_rn" if use_div_rn else "default",
        out_equal=bool(torch.equal(tri, fused_out)),
        out_mismatches=int((tri != fused_out).sum().item()),
        out_max_abs_diff=float(out_diff.max().item()),
        clip_count_abs_diff=float((clip_count_ref - packed_stats[1]).abs().max().item()),
        sumsq_rel_diff=_rel_diff(sumsq_ref, packed_stats[2]),
        xq_sumsq_rel_diff=_rel_diff(xq_sumsq_ref, packed_stats[3]),
        xxq_sum_rel_diff=_rel_diff(xxq_sum_ref, packed_stats[4]),
    )


def run_fused_stats_checks() -> list[FusedStatsResult]:
    torch.manual_seed(2468)
    results: list[FusedStatsResult] = []
    for dtype in _supported_dtypes():
        for shape in [(17, 13), (3, 19, 11), (2, 7, 5, 3), (1, 77, 1280)]:
            for use_div_rn in (False, True):
                results.append(_fused_stats_case(dtype=dtype, shape=shape, use_div_rn=use_div_rn))
    launch_boundary = int(_FUSED_STATS_LARGE_MIN_ELEMENTS)
    for shape in [
        (1, launch_boundary - 1),
        (1, launch_boundary),
        (480, 1280),
        (1, 320, 48, 40),
        (1, 480, 1280),
        (1, 480, 10240),
        (1, 468, 1280),
        (1, 468, 10240),
    ]:
        for use_div_rn in (False, True):
            results.append(_fused_stats_case(dtype=torch.float16, shape=shape, use_div_rn=use_div_rn))
    return results


def _fused_rng_case(*, shape: tuple[int, ...], use_div_rn: bool) -> RngResult:
    if triton_fake_quantize_levels_stoch is None or triton_fake_quantize_levels_stoch_with_stats is None:
        raise RuntimeError("production Triton fake-quant wrappers are unavailable")

    dtype = torch.float16
    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), "channel").contiguous()
    x = _make_x(shape, dtype, scale, "random").contiguous()
    warm_rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()
    _ = triton_fake_quantize_levels_stoch(
        x, scale=scale, qmin=-127, qmax=127, use_div_rn=use_div_rn, rand=warm_rand
    )
    _ = triton_fake_quantize_levels_stoch_with_stats(
        x, scale=scale, qmin=-127, qmax=127, use_div_rn=use_div_rn, rand=warm_rand
    )
    torch.cuda.synchronize()

    state0 = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state0)
    out_normal = triton_fake_quantize_levels_stoch(
        x, scale=scale, qmin=-127, qmax=127, use_div_rn=use_div_rn
    )
    torch.cuda.synchronize()
    after_normal = torch.cuda.get_rng_state()

    torch.cuda.set_rng_state(state0)
    fused = triton_fake_quantize_levels_stoch_with_stats(
        x, scale=scale, qmin=-127, qmax=127, use_div_rn=use_div_rn
    )
    torch.cuda.synchronize()
    after_fused = torch.cuda.get_rng_state()
    if out_normal is None or fused is None:
        raise RuntimeError(f"production RNG comparison failed to run for shape={shape}")
    out_fused, _ = fused
    diff = (out_normal.to(torch.float32) - out_fused.to(torch.float32)).abs()
    return RngResult(
        name=f"normal_vs_fused/{'div_rn' if use_div_rn else 'default'}",
        dtype=str(dtype).replace("torch.", ""),
        shape=str(tuple(shape)),
        scale="channel",
        rng_after_equal=bool(torch.equal(after_normal, after_fused)),
        out_equal=bool(torch.equal(out_normal, out_fused)),
        mismatches=int((out_normal != out_fused).sum().item()),
        max_abs_diff=float(diff.max().item()),
        before_hash=_rng_hash(state0),
        after_ref_hash=_rng_hash(after_normal),
        after_tri_hash=_rng_hash(after_fused),
    )


def run_fused_rng_checks() -> list[RngResult]:
    torch.manual_seed(8642)
    return [
        _fused_rng_case(shape=shape, use_div_rn=use_div_rn)
        for shape in [(3, 19, 11), (1, 480, 1280), (1, 468, 10240)]
        for use_div_rn in (False, True)
    ]


def run_forced_fallback_check() -> FallbackResult:
    shape = (3, 19, 11)
    dtype = torch.float16
    channel_count = shape[2]
    scale_flat = torch.linspace(0.0007, 0.0031, channel_count, device="cuda", dtype=torch.float32)
    scale = _scale_to_shape(scale_flat, torch.empty(shape, device="cuda"), "channel").contiguous()
    x = _make_x(shape, dtype, scale, "random").contiguous()
    captured: dict[str, torch.Tensor] = {}

    original = rounding_util.triton_fake_quantize_levels_stoch

    def forced_failure(*args, rand=None, **kwargs):
        if rand is not None:
            captured["rand"] = rand.detach().clone()
        return None

    state0 = torch.cuda.get_rng_state()
    try:
        rounding_util.triton_fake_quantize_levels_stoch = forced_failure
        torch.cuda.set_rng_state(state0)
        out = rounding_util.fake_quantize_levels(
            x,
            scale=scale,
            qmin=-127,
            qmax=127,
            mode="stoch",
            use_triton=True,
        )
        torch.cuda.synchronize()
        after_fallback = torch.cuda.get_rng_state()
    finally:
        rounding_util.triton_fake_quantize_levels_stoch = original

    torch.cuda.set_rng_state(state0)
    rand_ref = torch.rand_like(x, dtype=torch.float32)
    quantized_ref = ref_fake_quant_stoch_with_rand(x, scale, -127, 127, rand_ref)
    out_ref = x + (quantized_ref - x).detach()
    torch.cuda.synchronize()
    after_ref = torch.cuda.get_rng_state()
    diff = (out.detach().to(torch.float32) - out_ref.to(torch.float32)).abs()
    return FallbackResult(
        out_equal=bool(torch.equal(out.detach(), out_ref)),
        rand_equal=bool("rand" in captured and torch.equal(captured["rand"], rand_ref)),
        rng_after_equal=bool(torch.equal(after_fallback, after_ref)),
        mismatches=int((out.detach() != out_ref).sum().item()),
        max_abs_diff=float(diff.max().item()),
    )


def _make_fused_route_module() -> tuple[lora_impl.LoRAModule, lora_impl.DQStatsManager]:
    org_module = torch.nn.Linear(11, 11, bias=False, device="cuda", dtype=torch.float16)
    module = lora_impl.LoRAModule(
        "lora_unet_test",
        org_module,
        lora_dim=4,
        alpha=4,
        delta_q_mode="stoch",
        delta_q_granularity="channel",
        delta_q_stat="rms",
        delta_q_bits=8,
        delta_q_range_mul=3.0,
        delta_q_use_triton=True,
        delta_q_triton_div_rn=True,
        delta_q_triton_stats=True,
        delta_q_triton_stats_mode="fused",
    )
    manager = lora_impl.DQStatsManager()
    module.dq_stats_manager = manager
    manager.begin_step(
        step_idx=1,
        device=torch.device("cuda"),
        do_log=True,
        do_auto=False,
        collect_full=True,
        collect_zero=False,
        collect_near_zero=False,
        collect_detail=False,
        collect_error_parts=False,
        log_mode="summary",
        log_scope="unet",
        auto_scope="unet",
        target="delta",
    )
    return module, manager


def _fused_route_input() -> tuple[torch.Tensor, torch.Tensor]:
    shape = (3, 19, 11)
    seed_scale = torch.linspace(0.0007, 0.0031, shape[-1], device="cuda", dtype=torch.float32).view(1, 1, -1)
    x = _make_x(shape, torch.float16, seed_scale, "random").contiguous().detach().requires_grad_(True)
    scale = compute_scale_bits(
        x.detach(),
        bits=8,
        granularity="channel",
        stat="rms",
        range_mul=3.0,
        use_triton=True,
    ).contiguous()
    return x, scale


def run_fused_route_fallback_check() -> FusedRouteResult:
    module, manager = _make_fused_route_module()
    x, scale = _fused_route_input()
    captured: dict[str, torch.Tensor] = {}
    original_fused = lora_impl.triton_fake_quantize_levels_stoch_with_stats
    original_fake_quant = lora_impl.fake_quantize_levels

    def forced_fused_failure(*args, rand=None, **kwargs):
        if rand is not None:
            captured["fused_rand"] = rand.detach().clone()
        return None

    def normal_b_spy(*args, rand=None, **kwargs):
        if rand is not None:
            captured["normal_rand"] = rand.detach().clone()
        return original_fake_quant(*args, rand=rand, **kwargs)

    state0 = torch.cuda.get_rng_state()
    try:
        lora_impl.triton_fake_quantize_levels_stoch_with_stats = forced_fused_failure
        lora_impl.fake_quantize_levels = normal_b_spy
        torch.cuda.set_rng_state(state0)
        out = module._fake_quantize_levels_with_fused_stats(x, scale=scale, qmin=-127, qmax=127)
        torch.cuda.synchronize()
        after_fallback = torch.cuda.get_rng_state()
    finally:
        lora_impl.triton_fake_quantize_levels_stoch_with_stats = original_fused
        lora_impl.fake_quantize_levels = original_fake_quant

    if out is None:
        raise RuntimeError("LoRAModule fused fallback returned None")
    torch.cuda.set_rng_state(state0)
    rand_ref = torch.rand_like(x, dtype=torch.float32)
    raw_ref = triton_fake_quantize_levels_stoch(
        x.detach(),
        scale=scale,
        qmin=-127,
        qmax=127,
        use_div_rn=True,
        rand=rand_ref,
    )
    if raw_ref is None:
        raise RuntimeError("normal production B returned None in fused route fallback check")
    out_ref = x + (raw_ref - x).detach()
    torch.cuda.synchronize()
    after_ref = torch.cuda.get_rng_state()
    report = manager.get_path_report()
    stats_numel = float(manager.accum["unet"].numel.item())
    diff = (out.detach().to(torch.float32) - out_ref.detach().to(torch.float32)).abs()
    return FusedRouteResult(
        out_equal=bool(torch.equal(out.detach(), out_ref.detach())),
        rand_equal=bool(
            "fused_rand" in captured
            and "normal_rand" in captured
            and torch.equal(captured["fused_rand"], captured["normal_rand"])
            and torch.equal(captured["fused_rand"], rand_ref)
        ),
        rng_after_equal=bool(torch.equal(after_fallback, after_ref)),
        stats_numel_equal=stats_numel == float(x.numel()),
        counters_equal=bool(
            report["fused_stats_calls"] == 0
            and report["separate_stats_calls"] == 0
            and report["pytorch_stats_calls"] == 1
            and report["fused_fallback_calls"] == 1
        ),
        mismatches=int((out.detach() != out_ref.detach()).sum().item()),
        max_abs_diff=float(diff.max().item()),
    )


def run_fused_route_ste_check() -> FusedSteResult:
    module, manager = _make_fused_route_module()
    x, scale = _fused_route_input()
    out = module._fake_quantize_levels_with_fused_stats(x, scale=scale, qmin=-127, qmax=127)
    if out is None:
        raise RuntimeError("LoRAModule fused STE route returned None")
    out.to(torch.float32).sum().backward()
    grad = x.grad.detach().to(torch.float32)
    report = manager.get_path_report()
    stats_numel = float(manager.accum["unet"].numel.item())
    return FusedSteResult(
        ok=bool(torch.allclose(grad, torch.ones_like(grad), atol=0.0, rtol=0.0)),
        grad_min=float(grad.min().item()),
        grad_max=float(grad.max().item()),
        grad_mean=float(grad.mean().item()),
        stats_numel_equal=stats_numel == float(x.numel()),
        counters_equal=bool(report["fused_stats_calls"] == 1 and report["fused_elements"] == x.numel()),
    )


def _checkpoint_trace_case(use_reentrant: bool) -> CheckpointTraceResult:
    module, manager = _make_fused_route_module()
    x, scale = _fused_route_input()

    def checkpointed_quant(value: torch.Tensor) -> torch.Tensor:
        quantized = module._fake_quantize_levels_with_fused_stats(value, scale=scale, qmin=-127, qmax=127)
        if quantized is None:
            raise RuntimeError("checkpoint fused route returned None")
        return torch.sin(quantized.to(torch.float32)).sum()

    loss = checkpoint(checkpointed_quant, x, use_reentrant=use_reentrant, preserve_rng_state=True)
    calls_after_forward = manager.total_stats_calls()
    trace_snapshot = manager.trace_snapshot()
    loss.backward()
    calls_after_backward = manager.total_stats_calls()
    manager.record_backward_trace(trace_snapshot)
    report = manager.get_path_report()
    return CheckpointTraceResult(
        use_reentrant=use_reentrant,
        calls_after_forward=calls_after_forward,
        calls_after_backward=calls_after_backward,
        recompute_calls=report["backward_recompute_stats_calls"],
    )


def run_checkpoint_trace_checks() -> list[CheckpointTraceResult]:
    return [_checkpoint_trace_case(False), _checkpoint_trace_case(True)]


def print_results(results: Iterable[CaseResult]) -> int:
    failures = 0
    print("forward fixed-rand comparison")
    print("case,dtype,shape,scale,div,equal,mismatches,max_abs_diff,mean_abs_diff")
    for r in results:
        if not r.equal:
            failures += 1
        print(
            f"{r.name},{r.dtype},{r.shape},{r.scale},{r.div},"
            f"{r.equal},{r.mismatches},{r.max_abs_diff:.9g},{r.mean_abs_diff:.9g}"
        )
    return failures


def print_rng_results(results: Iterable[RngResult]) -> int:
    failures = 0
    print("production/fused wrapper same-rng-state comparison")
    print(
        "case,dtype,shape,scale,rng_after_equal,out_equal,mismatches,max_abs_diff,"
        "before_hash,after_ref_hash,after_tri_hash"
    )
    for r in results:
        if not (r.rng_after_equal and r.out_equal):
            failures += 1
        print(
            f"{r.name},{r.dtype},{r.shape},{r.scale},"
            f"{r.rng_after_equal},{r.out_equal},{r.mismatches},{r.max_abs_diff:.9g},"
            f"{r.before_hash},{r.after_ref_hash},{r.after_tri_hash}"
        )
    return failures


def print_fallback_result(result: FallbackResult) -> int:
    print("forced Triton failure fallback comparison")
    print("out_equal,rand_equal,rng_after_equal,mismatches,max_abs_diff")
    print(
        f"{result.out_equal},{result.rand_equal},{result.rng_after_equal},"
        f"{result.mismatches},{result.max_abs_diff:.9g}"
    )
    return 0 if result.out_equal and result.rand_equal and result.rng_after_equal else 1


def print_fused_route_results(fallback: FusedRouteResult, ste: FusedSteResult) -> int:
    print("LoRAModule fused fallback route comparison")
    print("out_equal,rand_equal,rng_after_equal,stats_numel_equal,counters_equal,mismatches,max_abs_diff")
    print(
        f"{fallback.out_equal},{fallback.rand_equal},{fallback.rng_after_equal},"
        f"{fallback.stats_numel_equal},{fallback.counters_equal},{fallback.mismatches},{fallback.max_abs_diff:.9g}"
    )
    print("LoRAModule fused STE check")
    print("ok,grad_min,grad_max,grad_mean,stats_numel_equal,counters_equal")
    print(
        f"{ste.ok},{ste.grad_min:.9g},{ste.grad_max:.9g},{ste.grad_mean:.9g},"
        f"{ste.stats_numel_equal},{ste.counters_equal}"
    )
    fallback_ok = (
        fallback.out_equal
        and fallback.rand_equal
        and fallback.rng_after_equal
        and fallback.stats_numel_equal
        and fallback.counters_equal
    )
    ste_ok = ste.ok and ste.stats_numel_equal and ste.counters_equal
    return 0 if fallback_ok and ste_ok else 1


def print_checkpoint_trace_results(results: Iterable[CheckpointTraceResult]) -> int:
    print("gradient checkpointing stats recompute trace (informational)")
    print("use_reentrant,calls_after_forward,calls_after_backward,recompute_calls")
    failures = 0
    for result in results:
        if result.calls_after_forward <= 0 or result.calls_after_backward < result.calls_after_forward:
            failures += 1
        print(
            f"{result.use_reentrant},{result.calls_after_forward},"
            f"{result.calls_after_backward},{result.recompute_calls}"
        )
    return failures


def print_mutation_results(results: Iterable[MutationResult]) -> int:
    failures = 0
    print("production wrapper input mutation/alias check")
    print(
        "case,dtype,shape,scale,x_mutated,max_abs_diff,out_aliases_x,"
        "x_ptr,out_ptr,version_before,version_after"
    )
    for r in results:
        if r.x_mutated or r.out_aliases_x:
            failures += 1
        print(
            f"{r.name},{r.dtype},{r.shape},{r.scale},"
            f"{r.x_mutated},{r.max_abs_diff:.9g},{r.out_aliases_x},"
            f"{r.x_ptr},{r.out_ptr},{r.version_before},{r.version_after}"
        )
    return failures


def print_end_to_end_results(results: Iterable[EndToEndResult]) -> int:
    failures = 0
    print("end-to-end scale+fake-quant fixed-rand comparison")
    print(
        "case,dtype,shape,scale,equal,mismatches,max_abs_diff,mean_abs_diff,"
        "scale_max_abs_diff,scale_max_rel_diff"
    )
    for r in results:
        if not r.equal:
            failures += 1
        print(
            f"{r.name},{r.dtype},{r.shape},{r.scale},"
            f"{r.equal},{r.mismatches},{r.max_abs_diff:.9g},{r.mean_abs_diff:.9g},"
            f"{r.scale_max_abs_diff:.9g},{r.scale_max_rel_diff:.9g}"
        )
    return failures


def print_fused_stats_results(results: Iterable[FusedStatsResult]) -> int:
    failures = 0
    print("fused B+stats fixed-rand comparison")
    print(
        "dtype,shape,div,out_equal,out_mismatches,out_max_abs_diff,"
        "clip_count_abs_diff,sumsq_rel_diff,xq_sumsq_rel_diff,xxq_sum_rel_diff"
    )
    for r in results:
        ok = (
            r.out_equal
            and r.clip_count_abs_diff == 0.0
            and r.sumsq_rel_diff < 1e-5
            and r.xq_sumsq_rel_diff < 1e-5
            and r.xxq_sum_rel_diff < 1e-5
        )
        if not ok:
            failures += 1
        print(
            f"{r.dtype},{r.shape},{r.div},{r.out_equal},{r.out_mismatches},{r.out_max_abs_diff:.9g},"
            f"{r.clip_count_abs_diff:.9g},{r.sumsq_rel_diff:.9g},"
            f"{r.xq_sumsq_rel_diff:.9g},{r.xxq_sum_rel_diff:.9g}"
        )
    return failures


def print_capture_results(results: Iterable[CaptureResult]) -> int:
    failures = 0
    totals: dict[str, dict[str, float]] = {}
    print("captured training tensor comparison")
    print(
        "path,global_step,global_step_1based,capture_seen,dtype,shape,stride,scale_shape,scale_stride,x_contig,scale_contig,"
        "rng_after_equal,production_equal,production_mismatches,production_max_abs_diff,"
        "x_mutated,x_mutation_max_abs_diff,out_aliases_x,version_before,version_after,"
        "kind,div,equal,mismatches,max_abs_diff,tri_gt_ref,tri_lt_ref,signed_sum,signed_mean_all,signed_mean_mismatch,"
        "mean_abs_out_diff,mse_out_diff,ref_quant_noise_l1,ref_quant_noise_mse,extra_l1_ratio,extra_mse_ratio,"
        "floor_diff_count,compare_diff_count,clamp_diff_count,quant_index_diff_count,same_index_value_diff_count,"
        "unclassified_mismatch_count,saturation_ref_count,saturation_tri_count,"
        "rand_margin_min,rand_margin_median,rand_margin_mean,"
        "integer_boundary_min,integer_boundary_median,integer_boundary_mean,e2e_scale_max_rel_diff"
    )
    count = 0
    for r in results:
        count += 1
        try:
            shape = tuple(int(part.strip()) for part in r.shape.strip("()").split(",") if part.strip())
            numel = 1
            for dim in shape:
                numel *= dim
        except Exception:
            numel = 0
        if not (r.fixed_default.equal and r.rng_after_equal and r.production_equal and not r.x_mutated and not r.out_aliases_x):
            failures += 1
        rows = [
            ("fixed", "default", r.fixed_default, 0.0),
            ("fixed", "div_rn", r.fixed_divrn, 0.0),
            ("e2e", "default", r.e2e_default, r.e2e_scale_max_rel_diff),
            ("e2e", "div_rn", r.e2e_divrn, r.e2e_scale_max_rel_diff),
        ]
        for kind, div, stats, scale_rel in rows:
            key = f"{kind}_{div}"
            total = totals.setdefault(
                key,
                {
                    "count": 0,
                    "elements": 0,
                    "mismatches": 0,
                    "tri_gt_ref": 0,
                    "tri_lt_ref": 0,
                    "signed_sum": 0.0,
                    "abs_sum": 0.0,
                    "sq_sum": 0.0,
                    "ref_noise_abs_sum": 0.0,
                    "ref_noise_sq_sum": 0.0,
                    "floor_diff": 0,
                    "compare_diff": 0,
                    "clamp_diff": 0,
                    "quant_index_diff": 0,
                    "same_index_value_diff": 0,
                    "unclassified_mismatch": 0,
                    "saturation_ref": 0,
                    "saturation_tri": 0,
                },
            )
            total["count"] += 1
            total["elements"] += numel
            total["mismatches"] += max(0, stats.mismatches)
            total["tri_gt_ref"] += stats.fixed_tri_gt_ref
            total["tri_lt_ref"] += stats.fixed_tri_lt_ref
            if stats.signed_sum == stats.signed_sum:
                total["signed_sum"] += stats.signed_sum
            if stats.mean_abs_out_diff == stats.mean_abs_out_diff:
                total["abs_sum"] += stats.mean_abs_out_diff * numel
            if stats.mse_out_diff == stats.mse_out_diff:
                total["sq_sum"] += stats.mse_out_diff * numel
            if stats.ref_quant_noise_l1 == stats.ref_quant_noise_l1:
                total["ref_noise_abs_sum"] += stats.ref_quant_noise_l1 * numel
            if stats.ref_quant_noise_mse == stats.ref_quant_noise_mse:
                total["ref_noise_sq_sum"] += stats.ref_quant_noise_mse * numel
            total["floor_diff"] += stats.floor_diff_count
            total["compare_diff"] += stats.compare_diff_count
            total["clamp_diff"] += stats.clamp_diff_count
            total["quant_index_diff"] += stats.quant_index_diff_count
            total["same_index_value_diff"] += stats.same_index_value_diff_count
            total["unclassified_mismatch"] += stats.unclassified_mismatch_count
            total["saturation_ref"] += stats.saturation_ref_count
            total["saturation_tri"] += stats.saturation_tri_count
            print(
                f"{r.path},{r.global_step},{r.global_step_1based},{r.capture_seen},"
                f"{r.dtype},{r.shape},{r.stride},{r.scale_shape},{r.scale_stride},"
                f"{r.x_contig},{r.scale_contig},"
                f"{r.rng_after_equal},{r.production_equal},{r.production_mismatches},{r.production_max_abs_diff:.9g},"
                f"{r.x_mutated},{r.x_mutation_max_abs_diff:.9g},{r.out_aliases_x},{r.version_before},{r.version_after},"
                f"{kind},{div},{stats.equal},{stats.mismatches},{stats.max_abs_diff:.9g},"
                f"{stats.fixed_tri_gt_ref},{stats.fixed_tri_lt_ref},{stats.signed_sum:.9g},"
                f"{stats.signed_mean_all:.9g},{stats.signed_mean_mismatch:.9g},"
                f"{stats.mean_abs_out_diff:.9g},{stats.mse_out_diff:.9g},"
                f"{stats.ref_quant_noise_l1:.9g},{stats.ref_quant_noise_mse:.9g},"
                f"{stats.extra_l1_ratio:.9g},{stats.extra_mse_ratio:.9g},"
                f"{stats.floor_diff_count},{stats.compare_diff_count},{stats.clamp_diff_count},"
                f"{stats.quant_index_diff_count},{stats.same_index_value_diff_count},{stats.unclassified_mismatch_count},"
                f"{stats.saturation_ref_count},{stats.saturation_tri_count},"
                f"{stats.rand_margin_min:.9g},{stats.rand_margin_median:.9g},{stats.rand_margin_mean:.9g},"
                f"{stats.integer_boundary_min:.9g},{stats.integer_boundary_median:.9g},{stats.integer_boundary_mean:.9g},"
                f"{scale_rel:.9g}"
            )
    if count == 0:
        print("no_capture_files_found")
        failures += 1
    else:
        print("captured training tensor comparison summary")
        print(
            "kind,count,total_elements,total_mismatches,tri_gt_ref,tri_lt_ref,"
            "signed_sum,signed_mean_all,signed_mean_mismatch,"
            "mean_abs_out_diff,mse_out_diff,ref_quant_noise_l1,ref_quant_noise_mse,"
            "extra_l1_ratio,extra_mse_ratio,floor_diff_count,compare_diff_count,clamp_diff_count,"
            "quant_index_diff_count,same_index_value_diff_count,unclassified_mismatch_count,"
            "saturation_ref_count,saturation_tri_count"
        )
        for key, total in totals.items():
            elements = int(total["elements"])
            mismatches = int(total["mismatches"])
            signed_sum = float(total["signed_sum"])
            mean_all = signed_sum / elements if elements > 0 else 0.0
            mean_mismatch = signed_sum / mismatches if mismatches > 0 else 0.0
            mean_abs = float(total["abs_sum"]) / elements if elements > 0 else 0.0
            mse = float(total["sq_sum"]) / elements if elements > 0 else 0.0
            noise_l1 = float(total["ref_noise_abs_sum"]) / elements if elements > 0 else 0.0
            noise_mse = float(total["ref_noise_sq_sum"]) / elements if elements > 0 else 0.0
            extra_l1 = mean_abs / max(noise_l1, 1e-30)
            extra_mse = mse / max(noise_mse, 1e-30)
            print(
                f"{key},{int(total['count'])},{elements},{mismatches},"
                f"{int(total['tri_gt_ref'])},{int(total['tri_lt_ref'])},"
                f"{signed_sum:.9g},{mean_all:.9g},{mean_mismatch:.9g},"
                f"{mean_abs:.9g},{mse:.9g},{noise_l1:.9g},{noise_mse:.9g},"
                f"{extra_l1:.9g},{extra_mse:.9g},"
                f"{int(total['floor_diff'])},{int(total['compare_diff'])},{int(total['clamp_diff'])},"
                f"{int(total['quant_index_diff'])},{int(total['same_index_value_diff'])},{int(total['unclassified_mismatch'])},"
                f"{int(total['saturation_ref'])},{int(total['saturation_tri'])}"
            )
    return failures


def print_scale_results(results: Iterable[ScaleResult]) -> int:
    failures = 0
    print("scale channel-rms comparison")
    print("dtype,shape,max_abs_diff,max_rel_diff,mean_abs_diff")
    for r in results:
        if r.max_rel_diff > 1e-5:
            failures += 1
        print(f"{r.dtype},{r.shape},{r.max_abs_diff:.9g},{r.max_rel_diff:.9g},{r.mean_abs_diff:.9g}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-div-rn", action="store_true", help="Skip tl.div_rn comparison kernel")
    parser.add_argument(
        "--skip-production-rng",
        action="store_true",
        help="Skip production wrapper same-RNG-state comparison",
    )
    parser.add_argument(
        "--skip-e2e",
        action="store_true",
        help="Skip end-to-end scale+fake-quant fixed-rand comparison",
    )
    parser.add_argument(
        "--skip-fused-stats",
        action="store_true",
        help="Skip fused B+stats fixed-rand comparison",
    )
    parser.add_argument(
        "--skip-mutation",
        action="store_true",
        help="Skip production wrapper input mutation/alias check",
    )
    parser.add_argument(
        "--skip-fused-route",
        action="store_true",
        help="Skip LoRAModule fused fallback and STE checks",
    )
    parser.add_argument(
        "--skip-checkpoint-trace",
        action="store_true",
        help="Skip informational gradient-checkpointing stats recompute trace",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=None,
        help="Validate tensors saved with DQ_TRITON_CAPTURE_DIR",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if triton is None:
        raise RuntimeError(f"Triton import failed: {TRITON_IMPORT_ERROR}")

    print(f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}")
    results = run_forward_checks(include_div_rn=not args.no_div_rn)
    failures = print_results(results)
    if not args.skip_production_rng:
        failures += print_rng_results(run_production_rng_checks() + run_fused_rng_checks())
        failures += print_fallback_result(run_forced_fallback_check())
    if not args.skip_fused_route:
        failures += print_fused_route_results(run_fused_route_fallback_check(), run_fused_route_ste_check())
    if not args.skip_checkpoint_trace:
        failures += print_checkpoint_trace_results(run_checkpoint_trace_checks())
    if not args.skip_mutation:
        failures += print_mutation_results(run_mutation_checks())
    if not args.skip_e2e:
        failures += print_end_to_end_results(run_end_to_end_checks())
    if not args.skip_fused_stats:
        failures += print_fused_stats_results(run_fused_stats_checks())
    if args.capture_dir is not None:
        failures += print_capture_results(run_capture_checks(args.capture_dir))
    failures += print_scale_results(run_scale_checks())
    ste_ok, grad_min, grad_max, grad_mean = run_ste_check()
    print(f"ste_check,ok={ste_ok},grad_min={grad_min:.6g},grad_max={grad_max:.6g},grad_mean={grad_mean:.6g}")

    return 1 if failures or not ste_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
