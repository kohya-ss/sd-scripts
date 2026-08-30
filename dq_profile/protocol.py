from __future__ import annotations

import hashlib
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    import tomllib  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python 3.10 used by this project
    tomllib = None

import toml

from dq_profile import RUNTIME_PROTOCOL_VERSION as PROTOCOL_VERSION


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
AUTO_BANDS: dict[str, tuple[float, float]] = {
    "clip_rate_high": (0.003, 0.005),
    "clip_rate_low": (0.0005, 0.0022),
}
DEFAULT_V2_RANGE_MULS: tuple[float, ...] = (2.70, 2.85, 3.00, 3.15, 3.30, 3.45)
STATELESS_RNG_DEFINITION_VERSION = "sdxl-dq-profile-v1"


@dataclass(frozen=True)
class CandidateDefinition:
    name: str
    quantized: bool
    clip_low: Optional[float]
    clip_high: Optional[float]
    initial_range_mul: Optional[float]
    auto_enabled: bool
    mechanism: str = "full"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetSubsetSummary:
    image_dir: str
    repeats: int
    image_count: int
    weighted_count: int


@dataclass(frozen=True)
class PreflightSummary:
    dataset_config: str
    dataset_batch_sizes: tuple[int, ...]
    subsets: tuple[DatasetSubsetSummary, ...]
    unique_images: int
    repeat_weighted_samples: int
    steps_per_epoch: int
    normal_training_steps: int
    dq_begin_step: int
    branch_steps: int
    probe_images: int
    probe_points_per_replica: int
    stochastic_repeats: int
    standard_probe_replicas: int
    full_probe_replicas: int
    full_budget_steps: int
    full_budget_core_exceeded: bool
    estimated_standard_steps: int
    estimated_full_steps: int
    estimated_standard_epochs: float
    estimated_full_epochs: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["subsets"] = [asdict(item) for item in self.subsets]
        return payload


def canonical_json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def deterministic_seed(
    protocol_seed: int,
    *,
    phase: str,
    probe_or_step: str | int,
    module_name: str,
    invocation: int,
    repeat: int,
) -> int:
    """Return a stateless unsigned 64-bit seed.

    Candidate identity is intentionally absent so high/low candidates use
    common random numbers.
    """

    fields = (
        # Keep the v1 stateless stream stable when report/schema versions
        # evolve. Candidate identity remains intentionally absent.
        STATELESS_RNG_DEFINITION_VERSION,
        str(int(protocol_seed)),
        str(phase),
        str(probe_or_step),
        str(module_name),
        str(int(invocation)),
        str(int(repeat)),
    )
    digest = hashlib.blake2b("\x1f".join(fields).encode("utf-8"), digest_size=8, person=b"dqprofv1").digest()
    return int.from_bytes(digest, "little", signed=False)


def initial_range_mul(clip_low: float, clip_high: float, minimum: float = 1.0, maximum: float = 6.0) -> float:
    clip_target = (float(clip_low) + float(clip_high)) / 2.0
    if not 0.0 < clip_target < 1.0:
        raise ValueError(f"clip target must be in (0, 1), got {clip_target}")
    value = statistics.NormalDist().inv_cdf(1.0 - clip_target / 2.0)
    return max(float(minimum), min(float(maximum), float(value)))


def default_candidates() -> tuple[CandidateDefinition, ...]:
    candidates = [CandidateDefinition("no_quant", False, None, None, None, False)]
    for name in ("clip_rate_high", "clip_rate_low"):
        low, high = AUTO_BANDS[name]
        candidates.append(CandidateDefinition(name, True, low, high, initial_range_mul(low, high), True))
    return tuple(candidates)


def parse_range_muls(
    value: str | Sequence[float],
    *,
    minimum_count: int = 3,
    maximum_count: int | None = None,
) -> tuple[float, ...]:
    if minimum_count <= 0:
        raise ValueError("minimum_count must be positive")
    if maximum_count is not None and maximum_count < minimum_count:
        raise ValueError("maximum_count must be >= minimum_count")
    raw = value.split(",") if isinstance(value, str) else value
    parsed: list[float] = []
    for item in raw:
        number = float(str(item).strip())
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"range_mul must be finite and positive, got {item!r}")
        if not any(abs(number - existing) <= 1e-12 for existing in parsed):
            parsed.append(number)
    if len(parsed) < minimum_count:
        raise ValueError(
            "fixed range sweep requires at least "
            f"{minimum_count} distinct range_mul values"
        )
    if maximum_count is not None and len(parsed) > maximum_count:
        raise ValueError(
            "fixed range sweep accepts at most "
            f"{maximum_count} distinct range_mul values"
        )
    return tuple(sorted(parsed))


def parse_mechanism_muls(value: Optional[str | Sequence[float]]) -> tuple[float, ...]:
    if value in (None, ""):
        return ()
    raw = value.split(",") if isinstance(value, str) else value
    parsed: list[float] = []
    for item in raw:
        number = float(str(item).strip())
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"mechanism range_mul must be finite and positive, got {item!r}")
        if not any(abs(number - existing) <= 1e-12 for existing in parsed):
            parsed.append(number)
    if not parsed:
        raise ValueError("at least one mechanism range_mul is required")
    return tuple(sorted(parsed))


def fixed_range_candidates(
    range_muls: Sequence[float],
    *,
    minimum_count: int = 3,
    maximum_count: int | None = None,
) -> tuple[CandidateDefinition, ...]:
    candidates = [CandidateDefinition("no_quant", False, None, None, None, False)]
    for value in parse_range_muls(
        tuple(range_muls),
        minimum_count=minimum_count,
        maximum_count=maximum_count,
    ):
        candidates.append(
            CandidateDefinition(
                name=f"mul_{value:.3f}",
                quantized=True,
                clip_low=None,
                clip_high=None,
                initial_range_mul=float(value),
                auto_enabled=False,
                mechanism="full",
            )
        )
    return tuple(candidates)


def mechanism_candidates(range_mul: float) -> tuple[CandidateDefinition, ...]:
    value = float(range_mul)
    return (
        CandidateDefinition("no_quant", False, None, None, None, False),
        CandidateDefinition(f"mul_{value:.3f}__full", True, None, None, value, False, "full"),
        CandidateDefinition(f"mul_{value:.3f}__clip_only", True, None, None, value, False, "clip_only"),
        CandidateDefinition(f"mul_{value:.3f}__round_only", True, None, None, value, False, "round_only"),
    )


def calculate_dq_begin_step(lr_warmup_steps: int | float, max_train_steps: int, num_processes: int = 1) -> int:
    """Mirror the copied trainer's dq_delta_begin_after_lr_warmup rule."""

    if isinstance(lr_warmup_steps, float):
        if max_train_steps <= 0:
            raise ValueError("max_train_steps must be positive when lr_warmup_steps is a float")
        value = int(lr_warmup_steps * max_train_steps * num_processes)
    else:
        value = int(lr_warmup_steps)
    return max(0, value)


def resolve_branch_steps(total_steps: int, override: Optional[int]) -> int:
    if override is not None:
        if override <= 0:
            raise ValueError("dq_profile_branch_steps must be positive")
        return int(override)
    return max(64, min(256, int(math.ceil(total_steps * 0.025))))


def _load_toml(path: Path) -> Mapping[str, Any]:
    if tomllib is not None:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return toml.load(stream)


def _iter_dataset_sections(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        datasets = value.get("datasets")
        if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
            for dataset in datasets:
                if isinstance(dataset, Mapping):
                    yield dataset
        for child in value.values():
            if isinstance(child, Mapping):
                yield from _iter_dataset_sections(child)


def _count_images(image_dir: Path) -> int:
    if not image_dir.exists() or not image_dir.is_dir():
        return 0
    return sum(1 for item in image_dir.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def inspect_dataset_config(
    dataset_config: str | os.PathLike[str],
    *,
    max_train_epochs: Optional[int],
    max_train_steps: Optional[int],
    lr_warmup_steps: int | float,
    branch_steps_override: Optional[int],
    max_images: int,
    timestep_bins: int,
    stochastic_repeats: int,
) -> PreflightSummary:
    path = Path(dataset_config).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"dataset config was not found: {path}")
    config = _load_toml(path)
    dataset_sections = list(_iter_dataset_sections(config))
    if not dataset_sections and isinstance(config.get("datasets"), Sequence):
        dataset_sections = [x for x in config["datasets"] if isinstance(x, Mapping)]
    if not dataset_sections:
        raise ValueError(f"no [[datasets]] sections were found in {path}")

    dataset_batch_sizes: list[int] = []
    subset_summaries: list[DatasetSubsetSummary] = []
    unique_paths: set[str] = set()
    weighted_samples = 0
    for dataset in dataset_sections:
        batch_size = int(dataset.get("batch_size", 1))
        dataset_batch_sizes.append(batch_size)
        subsets = dataset.get("subsets", [])
        if not isinstance(subsets, Sequence):
            continue
        for subset in subsets:
            if not isinstance(subset, Mapping):
                continue
            raw_dir = subset.get("image_dir")
            if raw_dir is None:
                continue
            image_dir = Path(str(raw_dir)).expanduser()
            if not image_dir.is_absolute():
                image_dir = (path.parent / image_dir).resolve()
            repeats = max(1, int(subset.get("num_repeats", 1)))
            image_files = [
                item.resolve()
                for item in image_dir.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ] if image_dir.is_dir() else []
            count = len(image_files)
            unique_paths.update(os.path.normcase(str(item)) for item in image_files)
            weighted = count * repeats
            weighted_samples += weighted
            subset_summaries.append(DatasetSubsetSummary(str(image_dir), repeats, count, weighted))

    if any(size != 1 for size in dataset_batch_sizes):
        raise ValueError(f"v1 requires every dataset batch_size to be 1, got {dataset_batch_sizes}")
    if weighted_samples <= 0:
        raise ValueError(f"no training images were found through {path}")
    steps_per_epoch = weighted_samples
    if max_train_epochs is not None:
        normal_steps = int(max_train_epochs) * steps_per_epoch
    elif max_train_steps is not None and int(max_train_steps) > 0:
        normal_steps = int(max_train_steps)
    else:
        raise ValueError("max_train_epochs or positive max_train_steps is required")
    begin_step = calculate_dq_begin_step(lr_warmup_steps, normal_steps)
    branch_steps = resolve_branch_steps(normal_steps, branch_steps_override)
    probe_images = min(max(1, int(max_images)), max(1, len(unique_paths)))
    repeats = max(1, int(stochastic_repeats))
    probe_points = probe_images * max(1, int(timestep_bins))
    probe_backward_per_replica = probe_points * (1 + 2 * repeats)
    core_steps = begin_step + branch_steps * 3
    standard_steps = core_steps + probe_backward_per_replica
    requested_full_budget = min(standard_steps * 2, int(normal_steps * 0.75))
    full_budget_core_exceeded = requested_full_budget < standard_steps
    full_budget = max(standard_steps, requested_full_budget)
    full_probe_replicas = max(1, (full_budget - core_steps) // max(probe_backward_per_replica, 1))
    full_steps = core_steps + full_probe_replicas * probe_backward_per_replica
    return PreflightSummary(
        dataset_config=str(path),
        dataset_batch_sizes=tuple(dataset_batch_sizes),
        subsets=tuple(subset_summaries),
        unique_images=len(unique_paths),
        repeat_weighted_samples=weighted_samples,
        steps_per_epoch=steps_per_epoch,
        normal_training_steps=normal_steps,
        dq_begin_step=begin_step,
        branch_steps=branch_steps,
        probe_images=probe_images,
        probe_points_per_replica=probe_points,
        stochastic_repeats=repeats,
        standard_probe_replicas=1,
        full_probe_replicas=full_probe_replicas,
        full_budget_steps=full_budget,
        full_budget_core_exceeded=full_budget_core_exceeded,
        estimated_standard_steps=standard_steps,
        estimated_full_steps=full_steps,
        estimated_standard_epochs=standard_steps / steps_per_epoch,
        estimated_full_epochs=full_steps / steps_per_epoch,
    )


class AutoRangeController:
    """Candidate-local implementation of the production high/low controller."""

    def __init__(
        self,
        candidate: CandidateDefinition,
        *,
        every: int = 50,
        ema: float = 0.95,
        mul_up: float = 1.01,
        mul_down: float = 0.995,
        minimum: float = 1.0,
        maximum: float = 6.0,
        warmup: bool = True,
        warmup_updates: int = 0,
        use_raw: bool = False,
    ) -> None:
        if not candidate.quantized or candidate.clip_low is None or candidate.clip_high is None:
            raise ValueError("AutoRangeController requires a quantized candidate")
        self.candidate = candidate
        self.every = max(1, int(every))
        self.ema_decay = float(ema)
        self.mul_up = float(mul_up)
        self.mul_down = float(mul_down)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.use_raw = bool(use_raw)
        self.range_mul = float(candidate.initial_range_mul)
        self.ema_value: Optional[float] = None
        self.observation_count = 0
        self.post_warmup_observation_count = 0
        self.inband_streak = 0
        if warmup and 0.0 < self.ema_decay < 1.0:
            self.warmup_updates = int(warmup_updates) if int(warmup_updates) > 0 else int(math.ceil(2.0 / (1.0 - self.ema_decay)))
        else:
            self.warmup_updates = 0
        self.warmup_remaining = self.warmup_updates
        self.rows: list[dict[str, Any]] = []

    @property
    def warmup_completed(self) -> bool:
        return self.warmup_remaining <= 0

    def observe(self, step: int, clip_rate: Optional[float]) -> dict[str, Any]:
        before = self.range_mul
        reason = "not_observed"
        applied = False
        raw = None if clip_rate is None else float(clip_rate)
        warmup_active = self.warmup_remaining > 0
        if raw is not None and math.isfinite(raw):
            self.observation_count += 1
            self.ema_value = raw if self.ema_value is None else self.ema_value * self.ema_decay + raw * (1.0 - self.ema_decay)
            ema_value = self.ema_value
            low = float(self.candidate.clip_low)
            high = float(self.candidate.clip_high)
            if warmup_active:
                self.inband_streak = self.inband_streak + 1 if low <= ema_value <= high else 0
                self.warmup_remaining = max(0, self.warmup_remaining - 1)
                if self.inband_streak >= 3:
                    self.warmup_remaining = 0
                reason = "warmup"
            else:
                self.post_warmup_observation_count += 1
                high_hit = ema_value > high and (not self.use_raw or raw > high)
                low_hit = ema_value < low and (not self.use_raw or raw < low)
                if high_hit:
                    self.range_mul *= self.mul_up
                    reason = "clip_high"
                elif low_hit:
                    self.range_mul *= self.mul_down
                    reason = "clip_low"
                else:
                    reason = "in_band"
                self.range_mul = max(self.minimum, min(self.maximum, self.range_mul))
                applied = self.range_mul != before
        row = {
            "step": int(step),
            "candidate": self.candidate.name,
            "clip_rate_raw": raw,
            "clip_rate_ema": self.ema_value,
            "range_mul_before": before,
            "range_mul_after": self.range_mul,
            "auto_applied": applied,
            "auto_reason": reason,
            "warmup_active": warmup_active,
            "warmup_remaining": self.warmup_remaining,
        }
        self.rows.append(row)
        return row

    def validity(self) -> dict[str, Any]:
        valid = self.warmup_completed and self.post_warmup_observation_count >= 3
        if not self.warmup_completed:
            reason = "auto_warmup_not_completed"
        elif self.post_warmup_observation_count < 3:
            reason = "fewer_than_3_post_warmup_observations"
        else:
            reason = None
        return {
            "auto_observation_count": self.observation_count,
            "auto_post_warmup_observation_count": self.post_warmup_observation_count,
            "auto_warmup_completed": self.warmup_completed,
            "auto_trajectory_metrics_valid": valid,
            "auto_invalid_reason": reason,
        }
