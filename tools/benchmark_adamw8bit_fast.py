from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bitsandbytes as bnb

from library.adamw8bit_fast import AdamW8bitFast


@dataclass
class Result:
    name: str
    median_ms: float
    min_ms: float
    mean_ms: float
    steady_allocated_mib: float
    step_peak_extra_mib: float


def load_weight_shapes(checkpoint: Path) -> list[tuple[int, ...]]:
    with safe_open(checkpoint, framework="pt", device="cpu") as file:
        return [tuple(file.get_slice(key).get_shape()) for key in file.keys() if not key.endswith(".alpha")]


def make_parameters(shapes: list[tuple[int, ...]], seed: int, dtype: torch.dtype) -> list[torch.nn.Parameter]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    parameters = []
    for shape in shapes:
        parameter = torch.nn.Parameter(torch.randn(shape, device="cuda", dtype=dtype, generator=generator) * 0.01)
        parameter.grad = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
        parameters.append(parameter)
    return parameters


def benchmark(
    name: str,
    factory: Callable[[list[torch.nn.Parameter]], torch.optim.Optimizer],
    shapes: list[tuple[int, ...]],
    *,
    seed: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> Result:
    gc.collect()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated()

    parameters = make_parameters(shapes, seed, dtype)
    optimizer = factory(parameters)
    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize()

    steady_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        optimizer.step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)

    peak_allocated = torch.cuda.max_memory_allocated()
    result = Result(
        name=name,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        mean_ms=statistics.mean(samples),
        steady_allocated_mib=(steady_allocated - baseline_allocated) / (1024**2),
        step_peak_extra_mib=(peak_allocated - steady_allocated) / (1024**2),
    )

    del optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare stock AdamW8bit with AdamW8bitFast on LoRA shapes")
    parser.add_argument("checkpoint", type=Path, help="LoRA .safetensors checkpoint used only for tensor shapes")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="Trainable parameter/gradient dtype (ordinary mixed-precision LoRA normally uses float32)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be >= 1")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    dtype = getattr(torch, args.dtype)

    shapes = load_weight_shapes(args.checkpoint)
    if not shapes:
        raise ValueError("checkpoint has no non-alpha tensors")
    tensor_count = len(shapes)
    element_count = sum(math.prod(shape) for shape in shapes)
    below_threshold = sum(math.prod(shape) < 4096 for shape in shapes)

    print(
        f"torch={torch.__version__} cuda={torch.version.cuda} bnb={bnb.__version__} "
        f"device={torch.cuda.get_device_name()}"
    )
    print(
        f"checkpoint={args.checkpoint} dtype={args.dtype} tensors={tensor_count} elements={element_count} "
        f"below_min_8bit_size={below_threshold} warmup={args.warmup} iterations={args.iterations}"
    )

    factories = [
        ("AdamW8bit", lambda params: bnb.optim.AdamW8bit(params, lr=args.lr)),
        ("AdamW8bitFast", lambda params: AdamW8bitFast(params, lr=args.lr)),
    ]
    results = [
        benchmark(
            name,
            factory,
            shapes,
            seed=args.seed,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        for name, factory in factories
    ]

    print("optimizer,median_ms,min_ms,mean_ms,steady_allocated_mib,step_peak_extra_mib")
    for result in results:
        print(
            f"{result.name},{result.median_ms:.6f},{result.min_ms:.6f},{result.mean_ms:.6f},"
            f"{result.steady_allocated_mib:.3f},{result.step_peak_extra_mib:.3f}"
        )

    stock, fast = results
    print(
        f"# summary speedup={stock.median_ms / fast.median_ms:.4f}x "
        f"saved_ms={stock.median_ms - fast.median_ms:.6f} "
        f"estimated_8400_saved_sec={(stock.median_ms - fast.median_ms) * 8.4:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
