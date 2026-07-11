from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import triton
except Exception as e:  # pragma: no cover - standalone diagnostic
    triton = None
    TRITON_IMPORT_ERROR = e
else:
    TRITON_IMPORT_ERROR = None

from library.triton_quant import (
    _FUSED_STATS_LARGE_BLOCK_SIZE,
    _FUSED_STATS_LARGE_MIN_ELEMENTS,
    _FUSED_STATS_SMALL_BLOCK_SIZE,
    triton_fake_quantize_levels_stoch,
    triton_fake_quantize_levels_stoch_with_stats,
)


DEFAULT_SHAPES = [
    (1, 32, 1280),
    (1, 77, 1280),
    (1, 480, 1280),
    (1, 480, 10240),
    (1, 468, 1280),
    (1, 468, 10240),
]


@dataclass
class BenchResult:
    operation: str
    median_ms: float
    min_ms: float


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid shape: {value}") from e
    if len(shape) not in (2, 3, 4) or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape must have 2, 3, or 4 positive dimensions")
    return shape


def channel_scale(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    channel_count = shape[1] if len(shape) in (2, 4) else shape[2]
    flat = torch.linspace(0.0007, 0.0031, channel_count, device=device, dtype=torch.float32)
    if len(shape) == 2:
        return flat.view(1, channel_count).contiguous()
    if len(shape) == 3:
        return flat.view(1, 1, channel_count).contiguous()
    return flat.view(1, channel_count, 1, 1).contiguous()


def require_result(value, name: str):
    if value is None:
        raise RuntimeError(f"{name} returned None")
    return value


def benchmark_cuda(
    fn: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)

    return BenchResult("", statistics.median(samples), min(samples))


def pytorch_basic_stats(
    x: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    qmax: int,
) -> torch.Tensor:
    x_fp32 = x.to(torch.float32)
    q_fp32 = quantized.to(torch.float32)
    x_flat = x_fp32.reshape(-1)
    q_flat = q_fp32.reshape(-1)
    y = x_fp32 / scale
    return torch.stack(
        (
            torch.tensor(float(x.numel()), device=x.device, dtype=torch.float32),
            (y.abs() >= qmax).to(torch.float32).sum(),
            torch.dot(x_flat, x_flat),
            torch.dot(q_flat, q_flat),
            torch.dot(x_flat, q_flat),
        )
    )


def run_shape(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[BenchResult]:
    device = torch.device("cuda")
    qmin, qmax = -127, 127
    x = (torch.randn(shape, device=device, dtype=torch.float32) * 0.05).to(dtype).contiguous()
    scale = channel_scale(shape, device)
    fixed_rand = torch.rand(shape, device=device, dtype=torch.float32).contiguous()
    fused_block_size = (
        _FUSED_STATS_LARGE_BLOCK_SIZE
        if x.numel() >= _FUSED_STATS_LARGE_MIN_ELEMENTS
        else _FUSED_STATS_SMALL_BLOCK_SIZE
    )
    partial_rows = math.ceil(x.numel() / fused_block_size)
    pytorch_accumulator = torch.zeros(5, device=device, dtype=torch.float32)
    fused_accumulator = torch.zeros(5, device=device, dtype=torch.float32)

    def normal_fixed():
        return require_result(
            triton_fake_quantize_levels_stoch(
                x,
                scale=scale,
                qmin=qmin,
                qmax=qmax,
                rand=fixed_rand,
            ),
            "normal B",
        )

    def normal_random():
        return require_result(
            triton_fake_quantize_levels_stoch(
                x,
                scale=scale,
                qmin=qmin,
                qmax=qmax,
            ),
            "normal B with PyTorch rand",
        )

    def normal_plus_pytorch_stats():
        quantized = normal_fixed()
        return pytorch_basic_stats(x, quantized, scale, qmax)

    def production_pytorch_stats_random():
        stats = pytorch_basic_stats(x, normal_random(), scale, qmax)
        return pytorch_accumulator.add_(stats)

    def fused_fixed():
        return require_result(
            triton_fake_quantize_levels_stoch_with_stats(
                x,
                scale=scale,
                qmin=qmin,
                qmax=qmax,
                rand=fixed_rand,
            ),
            "fused B+stats",
        )

    def fused_random():
        return require_result(
            triton_fake_quantize_levels_stoch_with_stats(
                x,
                scale=scale,
                qmin=qmin,
                qmax=qmax,
            ),
            "fused B+stats with PyTorch rand",
        )

    def production_fused_random():
        _, packed = fused_random()
        return fused_accumulator.add_(packed)

    operations = [
        ("normal_b_fixed_rand", normal_fixed),
        ("normal_b_pytorch_rand", normal_random),
        ("normal_b_plus_pytorch_stats", normal_plus_pytorch_stats),
        ("fused_stats_fixed_rand", fused_fixed),
        ("fused_stats_pytorch_rand", fused_random),
        ("production_pytorch_stats_rand", production_pytorch_stats_random),
        ("production_fused_pytorch_rand", production_fused_random),
    ]

    normal_out = normal_fixed()
    fused_out, _ = fused_fixed()
    if not torch.equal(normal_out, fused_out):
        mismatches = int((normal_out != fused_out).sum().item())
        raise RuntimeError(f"normal/fused output mismatch before benchmark: {mismatches}")

    results = []
    with torch.no_grad():
        for name, fn in operations:
            result = benchmark_cuda(fn, warmup=warmup, iterations=iterations, repeats=repeats)
            result.operation = name
            results.append(result)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="CUDA-event benchmark for dq_delta Triton fake quant stats")
    parser.add_argument("--shape", action="append", type=parse_shape, help="Shape such as 1,480,10240; repeatable")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Use a short 5/20/3 warmup/iteration/repeat run")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if triton is None:
        raise RuntimeError(f"Triton import failed: {TRITON_IMPORT_ERROR}")
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("warmup must be >= 0 and iterations/repeats must be > 0")

    warmup, iterations, repeats = args.warmup, args.iterations, args.repeats
    if args.quick:
        warmup, iterations, repeats = 5, 20, 3
    dtype = getattr(torch, args.dtype)
    shapes = args.shape or DEFAULT_SHAPES

    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__} "
        f"device={torch.cuda.get_device_name()} dtype={args.dtype} "
        f"warmup={warmup} iterations={iterations} repeats={repeats}"
    )
    print("shape,numel,fused_block_size,partial_rows,operation,median_ms,min_ms")
    for shape in shapes:
        results = run_shape(
            shape,
            dtype=dtype,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        by_name = {result.operation: result for result in results}
        numel = math.prod(shape)
        fused_block_size = (
            _FUSED_STATS_LARGE_BLOCK_SIZE
            if numel >= _FUSED_STATS_LARGE_MIN_ELEMENTS
            else _FUSED_STATS_SMALL_BLOCK_SIZE
        )
        partial_rows = math.ceil(numel / fused_block_size)
        for result in results:
            print(
                f'"{shape}",{numel},{fused_block_size},{partial_rows},'
                f"{result.operation},{result.median_ms:.9g},{result.min_ms:.9g}"
            )

        fused = by_name["fused_stats_fixed_rand"].median_ms
        pytorch_stats = by_name["normal_b_plus_pytorch_stats"].median_ms
        production_pytorch = by_name["production_pytorch_stats_rand"].median_ms
        production_fused = by_name["production_fused_pytorch_rand"].median_ms
        print(
            f'# summary shape={shape} pytorch_stats_to_fused={pytorch_stats / fused:.4f}x '
            f'production_pytorch_to_fused={production_pytorch / production_fused:.4f}x'
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
