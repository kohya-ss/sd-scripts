from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch


def error_decomposition(
    x: torch.Tensor, q: torch.Tensor, x_clamped: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error = q - x
    clip_error = x_clamped - x
    round_error = q - x_clamped
    return error, clip_error, round_error


@dataclass
class ExactGradient:
    values: dict[str, torch.Tensor]
    norm_sq: float

    @classmethod
    def capture(cls, named_parameters: Iterable[tuple[str, torch.nn.Parameter]], *, scale: float = 1.0) -> "ExactGradient":
        scale = max(float(scale), 1.0)
        values: dict[str, torch.Tensor] = {}
        norm_sq = 0.0
        for name, parameter in named_parameters:
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().to(dtype=torch.float32, device="cpu") / scale
            grad = grad.contiguous().clone()
            values[name] = grad
            norm_sq += float(torch.sum(grad * grad).item())
        return cls(values, norm_sq)

    @property
    def norm(self) -> float:
        return math.sqrt(max(self.norm_sq, 0.0))

    def cosine(self, other: "ExactGradient") -> dict[str, Any]:
        common = sorted(set(self.values).intersection(other.values))
        topology_matches = set(self.values) == set(other.values)
        dot = 0.0
        left_sq = 0.0
        right_sq = 0.0
        difference_sq = 0.0
        for name in common:
            left = self.values[name]
            right = other.values[name]
            if left.shape != right.shape:
                topology_matches = False
                continue
            dot += float(torch.sum(left * right).item())
            left_sq += float(torch.sum(left * left).item())
            right_sq += float(torch.sum(right * right).item())
            difference = right - left
            difference_sq += float(torch.sum(difference * difference).item())
        denominator = math.sqrt(left_sq) * math.sqrt(right_sq)
        cosine = dot / denominator if denominator > 0.0 else float("nan")
        return {
            "cosine": cosine,
            "dot": dot,
            "reference_norm": math.sqrt(left_sq),
            "candidate_norm": math.sqrt(right_sq),
            "difference_norm": math.sqrt(max(difference_sq, 0.0)),
            "common_parameter_count": len(common),
            "topology_matches": topology_matches,
        }


class CountSketch:
    """Deterministic CountSketch reserved for image-gradient structure."""

    def __init__(self, width: int = 4096, seed: int = 0) -> None:
        if width <= 0:
            raise ValueError("CountSketch width must be positive")
        self.width = int(width)
        self.seed = int(seed)

    def sketch(self, gradient: ExactGradient) -> np.ndarray:
        output = np.zeros(self.width, dtype=np.float64)
        offset = 0
        for name in sorted(gradient.values):
            flat = gradient.values[name].numpy().reshape(-1).astype(np.float64, copy=False)
            indices = np.arange(offset, offset + flat.size, dtype=np.uint64)
            name_seed = np.uint64(int.from_bytes(name.encode("utf-8")[:8].ljust(8, b"\0"), "little") ^ self.seed)
            mixed = indices + name_seed + np.uint64(0x9E3779B97F4A7C15)
            mixed ^= mixed >> np.uint64(30)
            mixed *= np.uint64(0xBF58476D1CE4E5B9)
            mixed ^= mixed >> np.uint64(27)
            mixed *= np.uint64(0x94D049BB133111EB)
            mixed ^= mixed >> np.uint64(31)
            buckets = (mixed % np.uint64(self.width)).astype(np.int64)
            signs = np.where((mixed & np.uint64(1)) == 0, 1.0, -1.0)
            np.add.at(output, buckets, flat * signs)
            offset += flat.size
        return output


def gram_and_rank(sketches: Iterable[np.ndarray]) -> dict[str, Any]:
    matrix = np.asarray(list(sketches), dtype=np.float64)
    if matrix.size == 0:
        return {"gram": np.zeros((0, 0)), "eigenvalues": np.zeros(0), "effective_rank": 0.0, "stable_rank": 0.0}
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        effective_rank = 0.0
        stable_rank = 0.0
    else:
        probabilities = eigenvalues[eigenvalues > 0.0] / total
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        stable_rank = total / max(float(eigenvalues.max()), 1e-30)
    return {
        "gram": gram,
        "eigenvalues": eigenvalues,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
    }


def gradient_noise_scale(sketches: Iterable[np.ndarray]) -> Optional[float]:
    matrix = np.asarray(list(sketches), dtype=np.float64)
    if matrix.shape[0] < 2:
        return None
    mean = matrix.mean(axis=0)
    signal = float(np.dot(mean, mean))
    centered = matrix - mean
    noise = float(np.sum(centered * centered) / (matrix.shape[0] - 1))
    return noise / max(signal, 1e-30)


def aggregate_numeric(rows: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    rows = list(rows)
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        if not values:
            continue
        mean = sum(values) / len(values)
        output[f"{key}_mean"] = mean
        output[f"{key}_std"] = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return output
