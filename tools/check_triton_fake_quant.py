from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

try:
    import triton
    import triton.language as tl
except Exception as e:  # pragma: no cover - diagnostic script
    triton = None
    tl = None
    TRITON_IMPORT_ERROR = e
else:
    TRITON_IMPORT_ERROR = None

from library.rounding_util import compute_scale_bits, fake_quantize_levels


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
    fixed_equal: bool
    fixed_mismatches: int
    fixed_max_abs_diff: float
    fixed_tri_gt_ref: int
    fixed_tri_lt_ref: int
    fixed_signed_sum: float
    fixed_signed_mean_all: float
    fixed_signed_mean_mismatch: float
    rng_after_equal: bool
    production_equal: bool
    production_mismatches: int
    production_max_abs_diff: float
    x_mutated: bool
    x_mutation_max_abs_diff: float
    out_aliases_x: bool
    version_before: int
    version_after: int
    e2e_equal: bool
    e2e_mismatches: int
    e2e_max_abs_diff: float
    e2e_tri_gt_ref: int
    e2e_tri_lt_ref: int
    e2e_signed_sum: float
    e2e_signed_mean_all: float
    e2e_signed_mean_mismatch: float
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


def _capture_case(path: Path) -> CaptureResult:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    x_cpu = payload["x"]
    scale_cpu = payload["scale"]
    x = x_cpu.to("cuda")
    scale = scale_cpu.to("cuda", dtype=torch.float32)
    qmin = int(payload["qmin"])
    qmax = int(payload["qmax"])

    fixed_equal = False
    fixed_mismatches = -1
    fixed_max_abs_diff = float("nan")
    fixed_tri_gt_ref = 0
    fixed_tri_lt_ref = 0
    fixed_signed_sum = float("nan")
    fixed_signed_mean_all = float("nan")
    fixed_signed_mean_mismatch = float("nan")
    if x.is_contiguous() and scale.is_contiguous() and x.ndim in (2, 3, 4):
        rand = torch.rand(x.shape, device="cuda", dtype=torch.float32).contiguous()
        ref = ref_fake_quant_stoch_with_rand(x, scale, qmin, qmax, rand)
        tri = debug_triton_fake_quant(x, scale, qmin, qmax, rand, use_div_rn=False)
        diff = (ref.to(torch.float32) - tri.to(torch.float32)).abs()
        fixed_equal = bool(torch.equal(ref, tri))
        fixed_mismatches = int((ref != tri).sum().item())
        fixed_max_abs_diff = float(diff.max().item())
        (
            fixed_tri_gt_ref,
            fixed_tri_lt_ref,
            fixed_signed_sum,
            fixed_signed_mean_all,
            fixed_signed_mean_mismatch,
        ) = _signed_diff_stats(ref, tri)

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
    e2e_tri = debug_triton_fake_quant(x, scale_tri, qmin, qmax, rand, use_div_rn=False)
    e2e_diff = (e2e_ref.to(torch.float32) - e2e_tri.to(torch.float32)).abs()
    (
        e2e_tri_gt_ref,
        e2e_tri_lt_ref,
        e2e_signed_sum,
        e2e_signed_mean_all,
        e2e_signed_mean_mismatch,
    ) = _signed_diff_stats(e2e_ref, e2e_tri)
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
        fixed_equal=fixed_equal,
        fixed_mismatches=fixed_mismatches,
        fixed_max_abs_diff=fixed_max_abs_diff,
        fixed_tri_gt_ref=fixed_tri_gt_ref,
        fixed_tri_lt_ref=fixed_tri_lt_ref,
        fixed_signed_sum=fixed_signed_sum,
        fixed_signed_mean_all=fixed_signed_mean_all,
        fixed_signed_mean_mismatch=fixed_signed_mean_mismatch,
        rng_after_equal=rng_after_equal,
        production_equal=production_equal,
        production_mismatches=production_mismatches,
        production_max_abs_diff=production_max_abs_diff,
        x_mutated=mutation.x_mutated,
        x_mutation_max_abs_diff=mutation.max_abs_diff,
        out_aliases_x=mutation.out_aliases_x,
        version_before=mutation.version_before,
        version_after=mutation.version_after,
        e2e_equal=bool(torch.equal(e2e_ref, e2e_tri)),
        e2e_mismatches=int((e2e_ref != e2e_tri).sum().item()),
        e2e_max_abs_diff=float(e2e_diff.max().item()),
        e2e_tri_gt_ref=e2e_tri_gt_ref,
        e2e_tri_lt_ref=e2e_tri_lt_ref,
        e2e_signed_sum=e2e_signed_sum,
        e2e_signed_mean_all=e2e_signed_mean_all,
        e2e_signed_mean_mismatch=e2e_signed_mean_mismatch,
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
    print("production wrapper same-rng-state comparison")
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


def print_capture_results(results: Iterable[CaptureResult]) -> int:
    failures = 0
    fixed_total_mismatches = 0
    fixed_total_tri_gt_ref = 0
    fixed_total_tri_lt_ref = 0
    fixed_total_signed_sum = 0.0
    fixed_total_elements = 0
    e2e_total_mismatches = 0
    e2e_total_tri_gt_ref = 0
    e2e_total_tri_lt_ref = 0
    e2e_total_signed_sum = 0.0
    e2e_total_elements = 0
    print("captured training tensor comparison")
    print(
        "path,global_step,global_step_1based,capture_seen,dtype,shape,stride,scale_shape,scale_stride,x_contig,scale_contig,"
        "fixed_equal,fixed_mismatches,fixed_max_abs_diff,"
        "fixed_tri_gt_ref,fixed_tri_lt_ref,fixed_signed_sum,fixed_signed_mean_all,fixed_signed_mean_mismatch,"
        "rng_after_equal,production_equal,production_mismatches,production_max_abs_diff,"
        "x_mutated,x_mutation_max_abs_diff,out_aliases_x,version_before,version_after,"
        "e2e_equal,e2e_mismatches,e2e_max_abs_diff,"
        "e2e_tri_gt_ref,e2e_tri_lt_ref,e2e_signed_sum,e2e_signed_mean_all,e2e_signed_mean_mismatch,"
        "e2e_scale_max_rel_diff"
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
        fixed_total_mismatches += max(0, r.fixed_mismatches)
        fixed_total_tri_gt_ref += r.fixed_tri_gt_ref
        fixed_total_tri_lt_ref += r.fixed_tri_lt_ref
        if r.fixed_signed_sum == r.fixed_signed_sum:
            fixed_total_signed_sum += r.fixed_signed_sum
        fixed_total_elements += numel
        e2e_total_mismatches += max(0, r.e2e_mismatches)
        e2e_total_tri_gt_ref += r.e2e_tri_gt_ref
        e2e_total_tri_lt_ref += r.e2e_tri_lt_ref
        if r.e2e_signed_sum == r.e2e_signed_sum:
            e2e_total_signed_sum += r.e2e_signed_sum
        e2e_total_elements += numel
        if not (r.fixed_equal and r.rng_after_equal and r.production_equal and not r.x_mutated and not r.out_aliases_x):
            failures += 1
        print(
            f"{r.path},{r.global_step},{r.global_step_1based},{r.capture_seen},"
            f"{r.dtype},{r.shape},{r.stride},{r.scale_shape},{r.scale_stride},"
            f"{r.x_contig},{r.scale_contig},"
            f"{r.fixed_equal},{r.fixed_mismatches},{r.fixed_max_abs_diff:.9g},"
            f"{r.fixed_tri_gt_ref},{r.fixed_tri_lt_ref},{r.fixed_signed_sum:.9g},"
            f"{r.fixed_signed_mean_all:.9g},{r.fixed_signed_mean_mismatch:.9g},"
            f"{r.rng_after_equal},{r.production_equal},{r.production_mismatches},{r.production_max_abs_diff:.9g},"
            f"{r.x_mutated},{r.x_mutation_max_abs_diff:.9g},{r.out_aliases_x},{r.version_before},{r.version_after},"
            f"{r.e2e_equal},{r.e2e_mismatches},{r.e2e_max_abs_diff:.9g},"
            f"{r.e2e_tri_gt_ref},{r.e2e_tri_lt_ref},{r.e2e_signed_sum:.9g},"
            f"{r.e2e_signed_mean_all:.9g},{r.e2e_signed_mean_mismatch:.9g},"
            f"{r.e2e_scale_max_rel_diff:.9g}"
        )
    if count == 0:
        print("no_capture_files_found")
        failures += 1
    else:
        fixed_mean_all = fixed_total_signed_sum / fixed_total_elements if fixed_total_elements > 0 else 0.0
        fixed_mean_mismatch = fixed_total_signed_sum / fixed_total_mismatches if fixed_total_mismatches > 0 else 0.0
        e2e_mean_all = e2e_total_signed_sum / e2e_total_elements if e2e_total_elements > 0 else 0.0
        e2e_mean_mismatch = e2e_total_signed_sum / e2e_total_mismatches if e2e_total_mismatches > 0 else 0.0
        print("captured training tensor signed-diff summary")
        print(
            "kind,count,total_elements,total_mismatches,tri_gt_ref,tri_lt_ref,"
            "signed_sum,signed_mean_all,signed_mean_mismatch"
        )
        print(
            f"fixed,{count},{fixed_total_elements},{fixed_total_mismatches},"
            f"{fixed_total_tri_gt_ref},{fixed_total_tri_lt_ref},"
            f"{fixed_total_signed_sum:.9g},{fixed_mean_all:.9g},{fixed_mean_mismatch:.9g}"
        )
        print(
            f"e2e,{count},{e2e_total_elements},{e2e_total_mismatches},"
            f"{e2e_total_tri_gt_ref},{e2e_total_tri_lt_ref},"
            f"{e2e_total_signed_sum:.9g},{e2e_mean_all:.9g},{e2e_mean_mismatch:.9g}"
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
        "--skip-mutation",
        action="store_true",
        help="Skip production wrapper input mutation/alias check",
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
        failures += print_rng_results(run_production_rng_checks())
    if not args.skip_mutation:
        failures += print_mutation_results(run_mutation_checks())
    if not args.skip_e2e:
        failures += print_end_to_end_results(run_end_to_end_checks())
    if args.capture_dir is not None:
        failures += print_capture_results(run_capture_checks(args.capture_dir))
    failures += print_scale_results(run_scale_checks())
    ste_ok, grad_min, grad_max, grad_mean = run_ste_check()
    print(f"ste_check,ok={ste_ok},grad_min={grad_min:.6g},grad_max={grad_max:.6g},grad_mean={grad_mean:.6g}")

    return 1 if failures or not ste_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
