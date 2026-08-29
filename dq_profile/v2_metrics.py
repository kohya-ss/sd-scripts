from __future__ import annotations

import itertools
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch


DIAGNOSTIC_TARGET = "no_quant_trajectory_stability"
UTILITY_ROPE = (0.45, 0.55)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _median(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


@dataclass(frozen=True)
class ParameterDelta:
    """Exact CPU FP32 cumulative parameter update from a shared snapshot."""

    values: dict[str, torch.Tensor]

    @classmethod
    def capture(
        cls,
        network: torch.nn.Module,
        reference: Mapping[str, torch.Tensor],
        parameter_names: Optional[Iterable[str]] = None,
    ) -> "ParameterDelta":
        values: dict[str, torch.Tensor] = {}
        selected = None if parameter_names is None else set(parameter_names)
        with torch.no_grad():
            for name, parameter in network.named_parameters():
                if name not in reference or (selected is not None and name not in selected):
                    continue
                current = parameter.detach().to(device="cpu", dtype=torch.float32)
                baseline = reference[name].detach().to(device="cpu", dtype=torch.float32)
                values[name] = (current - baseline).contiguous().clone()
        return cls(values)

    def norm(self, names: Optional[Iterable[str]] = None) -> float:
        selected = self.values if names is None else {name: self.values[name] for name in names if name in self.values}
        total = sum(float(torch.sum(value * value).item()) for value in selected.values())
        return math.sqrt(max(total, 0.0))


def parameter_module_groups(name: str) -> tuple[str, ...]:
    lowered = name.casefold()
    groups = ["all"]
    if any(token in lowered for token in ("input_blocks", "down_blocks", "down_block")):
        groups.append("region:input")
    elif any(token in lowered for token in ("middle_block", "mid_block")):
        groups.append("region:middle")
    elif any(token in lowered for token in ("output_blocks", "up_blocks", "up_block")):
        groups.append("region:output")
    else:
        groups.append("region:other")

    if "attn1" in lowered:
        groups.append("family:attn1")
    elif "attn2" in lowered:
        groups.append("family:attn2")
    elif any(token in lowered for token in ("_ff_", ".ff.", "feed_forward", "feedforward")):
        groups.append("family:ff")
    elif any(token in lowered for token in ("proj", "projection")):
        groups.append("family:projection")
    else:
        groups.append("family:other")
    return tuple(groups)


def compare_parameter_deltas(reference: ParameterDelta, candidate: ParameterDelta) -> list[dict[str, Any]]:
    common = sorted(set(reference.values).intersection(candidate.values))
    groups: dict[str, list[str]] = defaultdict(list)
    topology_matches = set(reference.values) == set(candidate.values)
    for name in common:
        if reference.values[name].shape != candidate.values[name].shape:
            topology_matches = False
            continue
        for group in parameter_module_groups(name):
            groups[group].append(name)

    rows: list[dict[str, Any]] = []
    for group, names in sorted(groups.items(), key=lambda item: (item[0] != "all", item[0])):
        dot = 0.0
        reference_sq = 0.0
        candidate_sq = 0.0
        difference_sq = 0.0
        for name in names:
            left = reference.values[name]
            right = candidate.values[name]
            dot += float(torch.sum(left * right).item())
            reference_sq += float(torch.sum(left * left).item())
            candidate_sq += float(torch.sum(right * right).item())
            difference = right - left
            difference_sq += float(torch.sum(difference * difference).item())
        reference_norm = math.sqrt(max(reference_sq, 0.0))
        candidate_norm = math.sqrt(max(candidate_sq, 0.0))
        denominator = reference_norm * candidate_norm
        valid = reference_sq > 1e-30 and denominator > 1e-30
        projection_gain = dot / reference_sq if reference_sq > 1e-30 else float("nan")
        orthogonal_sq = max(candidate_sq - (dot * dot / reference_sq), 0.0) if reference_sq > 1e-30 else float("nan")
        rows.append(
            {
                "module_group": group,
                "parameter_count": len(names),
                "topology_matches": topology_matches,
                "update_direction_valid": valid,
                "invalid_reason": None if valid else "no_quant_update_norm_too_small",
                "update_cosine": dot / denominator if valid else float("nan"),
                "projection_gain": projection_gain,
                "orthogonal_drift": math.sqrt(orthogonal_sq) / reference_norm if valid else float("nan"),
                "total_drift": math.sqrt(max(difference_sq, 0.0)) / reference_norm if valid else float("nan"),
                "update_norm_ratio": candidate_norm / reference_norm if reference_norm > 1e-30 else float("nan"),
                "no_quant_update_norm": reference_norm,
                "candidate_update_norm": candidate_norm,
            }
        )
    if not rows:
        rows.append(
            {
                "module_group": "all",
                "parameter_count": 0,
                "topology_matches": topology_matches,
                "update_direction_valid": False,
                "invalid_reason": "no_common_parameters",
                "update_cosine": float("nan"),
                "projection_gain": float("nan"),
                "orthogonal_drift": float("nan"),
                "total_drift": float("nan"),
                "update_norm_ratio": float("nan"),
                "no_quant_update_norm": 0.0,
                "candidate_update_norm": 0.0,
            }
        )
    return rows


def hard_safety_reason(
    *,
    loss: float,
    gradient_norm: float,
    matched_no_quant_gradient_norm: Optional[float],
    absolute_floor: float = 1.0e4,
    relative_multiplier: float = 100.0,
) -> Optional[str]:
    if not _finite(loss):
        return "candidate_nonfinite_loss"
    if not _finite(gradient_norm):
        return "candidate_nonfinite_gradient"
    reference = float(matched_no_quant_gradient_norm or 0.0)
    limit = max(float(absolute_floor), max(reference, 0.0) * float(relative_multiplier))
    if float(gradient_norm) > limit:
        return "candidate_gradient_explosion"
    return None


def _noise_floor(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    by_mul: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if _finite(row.get(metric)) and _finite(row.get("range_mul")):
            by_mul[float(row["range_mul"])].append(float(row[metric]))
    differences: list[float] = []
    for values in by_mul.values():
        for left, right in itertools.combinations(values, 2):
            differences.append(abs(left - right) / math.sqrt(2.0))
    return _median(differences) if differences else 0.0


def _contiguous_component(values: Sequence[float], anchor: Optional[float], grid: Sequence[float]) -> list[float]:
    selected = {float(value) for value in values}
    ordered = [float(value) for value in sorted(set(grid)) if float(value) in selected]
    if not ordered:
        return []
    components: list[list[float]] = [[ordered[0]]]
    positions = {float(value): index for index, value in enumerate(sorted(set(grid)))}
    for value in ordered[1:]:
        if positions[value] == positions[components[-1][-1]] + 1:
            components[-1].append(value)
        else:
            components.append([value])
    if anchor is not None:
        for component in components:
            if float(anchor) in component:
                return component
    return max(components, key=lambda component: (len(component), -abs(statistics.mean(component) - statistics.mean(grid))))


def _metric_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    grid: Sequence[float],
) -> dict[str, Any]:
    finite_rows = [
        row
        for row in rows
        if bool(row.get("update_direction_valid", True))
        and not bool(row.get("forced_safety_abort", False))
        and _finite(row.get(metric))
        and _finite(row.get("range_mul"))
    ]
    by_mul: dict[float, list[float]] = defaultdict(list)
    by_repeat: dict[int, dict[float, float]] = defaultdict(dict)
    for row in finite_rows:
        mul = float(row["range_mul"])
        repeat = int(row.get("repeat", 0))
        value = float(row[metric])
        by_mul[mul].append(value)
        by_repeat[repeat][mul] = value
    medians = {mul: _median(values) for mul, values in by_mul.items()}
    optimum = min(medians, key=lambda mul: (medians[mul], mul)) if medians else None
    repeat_best = {
        repeat: min(values, key=lambda mul: (values[mul], mul))
        for repeat, values in by_repeat.items()
        if values
    }
    noise_floor = _noise_floor(finite_rows, metric)
    noninferior = set(medians)
    for values in by_repeat.values():
        if not values:
            continue
        best_value = min(values.values())
        tolerance = max(2.0 * noise_floor, abs(best_value) * 0.10)
        noninferior.intersection_update({mul for mul, value in values.items() if value <= best_value + tolerance})
    plateau = _contiguous_component(sorted(noninferior), optimum, grid)
    sorted_medians = sorted(medians.items(), key=lambda item: (item[1], item[0]))
    runner_up_gap = (
        sorted_medians[1][1] - sorted_medians[0][1]
        if len(sorted_medians) > 1
        else float("nan")
    )
    return {
        "metric": metric,
        "optimum": optimum,
        "plateau_grid": plateau,
        "noise_floor": noise_floor,
        "repeat_best": repeat_best,
        "median_by_mul": medians,
        "runner_up_gap": runner_up_gap,
    }


def summarize_stability(
    rows: Sequence[Mapping[str, Any]],
    *,
    grid: Sequence[float],
    checkpoint: int = 64,
    guardian_mode: str = "common_skip",
    poor_threshold: float = 0.5,
) -> dict[str, Any]:
    endpoint_rows = [
        row
        for row in rows
        if int(row.get("checkpoint", -1)) == int(checkpoint)
        and str(row.get("guardian_mode", "")) == guardian_mode
        and str(row.get("module_group", "all")) == "all"
    ]
    direction = _metric_summary(endpoint_rows, metric="orthogonal_drift", grid=grid)
    total = _metric_summary(endpoint_rows, metric="total_drift", grid=grid)
    m_dir = direction["optimum"]
    m_total = total["optimum"]
    ordered = list(sorted(set(float(value) for value in grid)))
    if m_dir is None or m_total is None:
        diagnostic_optimum: Any = None
        confidence = "low"
        ambiguous = True
    else:
        distance = abs(ordered.index(float(m_dir)) - ordered.index(float(m_total)))
        if distance == 0:
            diagnostic_optimum = float(m_dir)
            confidence = "high"
            ambiguous = False
        elif distance == 1:
            diagnostic_optimum = sorted((float(m_dir), float(m_total)))
            confidence = "medium"
            ambiguous = False
        else:
            diagnostic_optimum = "ambiguous"
            confidence = "low"
            ambiguous = True

    common_plateau = sorted(set(direction["plateau_grid"]).intersection(total["plateau_grid"]))
    common_plateau = _contiguous_component(common_plateau, m_dir if m_dir == m_total else None, ordered)
    if not common_plateau:
        ambiguous = True
        diagnostic_optimum = "ambiguous"
        confidence = "low"

    reasons: list[str] = []
    for label, payload in (("dir", direction), ("total", total)):
        if len(set(payload["repeat_best"].values())) > 1:
            reasons.append(f"repeat_best_changed_{label}")
        gap = payload["runner_up_gap"]
        if _finite(gap) and float(gap) <= float(payload["noise_floor"]):
            reasons.append(f"best_runner_up_within_noise_{label}")
    if m_dir is not None and m_total is not None and m_dir != m_total:
        reasons.append("m_dir_differs_from_m_total")

    checkpoint32 = [
        row
        for row in rows
        if int(row.get("checkpoint", -1)) == 32
        and str(row.get("guardian_mode", "")) == guardian_mode
        and str(row.get("module_group", "all")) == "all"
    ]
    if checkpoint32:
        d32 = _metric_summary(checkpoint32, metric="orthogonal_drift", grid=grid)["optimum"]
        t32 = _metric_summary(checkpoint32, metric="total_drift", grid=grid)["optimum"]
        if d32 != m_dir:
            reasons.append("m_dir_changed_32_to_64")
        if t32 != m_total:
            reasons.append("m_total_changed_32_to_64")
    if not common_plateau:
        reasons.append("no_common_stability_plateau")
    if any(bool(row.get("forced_safety_abort", False)) for row in endpoint_rows):
        reasons.append("candidate_safety_abort")

    min_dir = min(direction["median_by_mul"].values(), default=float("inf"))
    min_total = min(total["median_by_mul"].values(), default=float("inf"))
    return {
        "diagnostic_target": DIAGNOSTIC_TARGET,
        "guardian_mode": guardian_mode,
        "checkpoint": int(checkpoint),
        "m_dir": m_dir,
        "m_total": m_total,
        "m_stability_diag": diagnostic_optimum,
        "diagnostic_optimum": "ambiguous" if ambiguous else diagnostic_optimum,
        "stability_confidence": confidence,
        "W_dir_grid": direction["plateau_grid"],
        "W_total_grid": total["plateau_grid"],
        "W_stability_grid": common_plateau,
        "all_candidates_poor": bool(min_dir > poor_threshold or min_total > poor_threshold),
        "m_utility": None,
        "third_repeat_required": bool(reasons),
        "third_repeat_reasons": sorted(set(reasons)),
        "direction_details": direction,
        "total_details": total,
    }


def guardian_dependence(
    intrinsic: Mapping[str, Any],
    native: Mapping[str, Any],
) -> dict[str, Any]:
    intrinsic_value = intrinsic.get("m_stability_diag")
    native_value = native.get("m_stability_diag")
    dependent = intrinsic_value != native_value or bool(native.get("diagnostic_optimum") == "ambiguous")
    return {
        "intrinsic_stability_result": dict(intrinsic),
        "guardian_adjusted_result": dict(native),
        "guardian_dependent": dependent,
    }


def mechanism_interaction(
    *,
    no_quant: float,
    full: float,
    clip_only: float,
    round_only: float,
) -> float:
    return (float(full) - float(no_quant)) - (float(clip_only) - float(no_quant)) - (
        float(round_only) - float(no_quant)
    )


def classify_utility_interval(
    estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    rope: tuple[float, float] = UTILITY_ROPE,
) -> str:
    del estimate  # Classification is intentionally interval-based.
    low, high = map(float, rope)
    if float(ci_low) > high:
        return "positive"
    if float(ci_high) < low:
        return "negative"
    if float(ci_low) >= low and float(ci_high) <= high:
        return "neutral"
    return "unknown"


def aggregate_training_seed_utility(seed_results: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    classifications = {
        int(seed): classify_utility_interval(
            float(result["estimate"]), float(result["ci_low"]), float(result["ci_high"])
        )
        for seed, result in seed_results.items()
    }
    if len(classifications) >= 2 and len(set(classifications.values())) == 1:
        selected = next(iter(classifications.values()))
        confidence = "moderate"
    else:
        selected = "unknown"
        confidence = "low"
    return {
        "utility_screen_seed39": classifications.get(39, "not_measured"),
        "U_selected_protocol": selected,
        "U_any_quantization": "unknown",
        "utility_confidence": confidence,
        "training_seed_results": classifications,
    }


def choose_minimax_pair(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    *,
    grid: Sequence[float],
) -> dict[str, Any]:
    def regrets(rows: Sequence[Mapping[str, Any]]) -> dict[float, float]:
        metrics: dict[str, dict[float, float]] = {}
        for metric in ("orthogonal_drift", "total_drift"):
            grouped: dict[float, list[float]] = defaultdict(list)
            for row in rows:
                if _finite(row.get("range_mul")) and _finite(row.get(metric)):
                    grouped[float(row["range_mul"])].append(float(row[metric]))
            medians = {mul: _median(values) for mul, values in grouped.items()}
            if not medians:
                metrics[metric] = {}
                continue
            lo, hi = min(medians.values()), max(medians.values())
            scale = max(hi - lo, 1e-12)
            metrics[metric] = {mul: (value - lo) / scale for mul, value in medians.items()}
        result: dict[float, float] = {}
        for mul in set(metrics["orthogonal_drift"]).intersection(metrics["total_drift"]):
            result[mul] = max(metrics["orthogonal_drift"][mul], metrics["total_drift"][mul])
        return result

    regret_a = regrets(rows_a)
    regret_b = regrets(rows_b)
    common = sorted(set(regret_a).intersection(regret_b).intersection(float(value) for value in grid))
    if not common:
        return {"m_pair": None, "valid": False, "invalid_reason": "no_common_safe_grid"}
    center = statistics.mean(sorted(float(value) for value in grid))
    scored = [(max(regret_a[mul], regret_b[mul]), abs(mul - center), mul) for mul in common]
    score, _, selected = min(scored)
    return {
        "m_pair": selected,
        "valid": True,
        "minimax_regret": score,
        "regret_a": regret_a,
        "regret_b": regret_b,
        "tie_break": "closest_to_grid_center_then_lower_mul",
    }


def _weighted_mean(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    total = float(weights.sum())
    if total <= 0.0:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    return np.sum(matrix * weights[:, None], axis=0) / total


def hierarchical_geometry_variance(
    sketches: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matrix = np.asarray(sketches, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(metadata) or matrix.shape[0] == 0:
        return {"valid": False, "invalid_reason": "sketch_metadata_shape_mismatch"}
    sources = [str(row.get("source_group", row.get("image_key", ""))) for row in metadata]
    images = [str(row.get("image_key", "")) for row in metadata]
    timesteps = [str(row.get("timestep_bin", "")) for row in metadata]
    repeats = [str(row.get("probe_replica", "")) for row in metadata]
    source_counts = Counter(sources)
    weights = np.asarray([1.0 / max(source_counts[source], 1) for source in sources], dtype=np.float64)
    grand = _weighted_mean(matrix, weights)

    def group_means(keys: Sequence[Any]) -> dict[Any, np.ndarray]:
        grouped: dict[Any, list[int]] = defaultdict(list)
        for index, key in enumerate(keys):
            grouped[key].append(index)
        return {
            key: _weighted_mean(matrix[indices], weights[indices])
            for key, indices in grouped.items()
        }

    source_means = group_means(sources)
    image_keys = list(zip(sources, images))
    image_means = group_means(image_keys)
    timestep_means = group_means(timesteps)
    source_timestep_keys = list(zip(sources, timesteps))
    source_timestep_means = group_means(source_timestep_keys)

    source_component = np.asarray([source_means[source] - grand for source in sources])
    image_component = np.asarray([image_means[(source, image)] - source_means[source] for source, image in image_keys])
    timestep_component = np.asarray([timestep_means[timestep] - grand for timestep in timesteps])
    interaction_component = np.asarray(
        [
            source_timestep_means[(source, timestep)]
            - source_means[source]
            - timestep_means[timestep]
            + grand
            for source, timestep in source_timestep_keys
        ]
    )
    fitted = grand + source_component + image_component + timestep_component + interaction_component
    residual_component = matrix - fitted

    def energy(component: np.ndarray) -> float:
        return float(np.sum(weights[:, None] * component * component) / max(float(weights.sum()), 1e-30))

    energies = {
        "source": energy(source_component),
        "image_within_source": energy(image_component),
        "timestep": energy(timestep_component),
        "source_timestep_interaction": energy(interaction_component),
        "repeat_noise_residual": energy(residual_component),
    }
    total = sum(max(value, 0.0) for value in energies.values())
    fractions = {f"{name}_fraction": max(value, 0.0) / max(total, 1e-30) for name, value in energies.items()}
    images_per_source = Counter(zip(sources, images))
    has_image_replication = any(sum(1 for key in images_per_source if key[0] == source) > 1 for source in source_counts)
    repeat_counts = Counter(zip(sources, images, timesteps))
    has_repeat_replication = any(count > 1 for count in repeat_counts.values())
    balanced = len(set(source_counts.values())) <= 1
    return {
        "valid": True,
        "method": "source_equal_weighted_descriptive_variance_v1",
        "source_count": len(source_counts),
        "image_count": len(set(images)),
        "timestep_count": len(set(timesteps)),
        "repeat_count": len(set(repeats)),
        "design_unbalanced": not balanced,
        "image_within_source_estimable": has_image_replication,
        "repeat_noise_residual_estimable": has_repeat_replication,
        **energies,
        **fractions,
    }


def sketch_agreement(first: np.ndarray, second: np.ndarray) -> dict[str, Any]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] < 2:
        return {"stable": False, "invalid_reason": "sketch_shape_mismatch"}

    def cosine_gram(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1)
        denominator = np.outer(norms, norms)
        return np.divide(matrix @ matrix.T, denominator, out=np.zeros_like(denominator), where=denominator > 0)

    gram_left = cosine_gram(left)
    gram_right = cosine_gram(right)
    indices = np.triu_indices(left.shape[0], 1)
    a, b = gram_left[indices], gram_right[indices]
    correlation = float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and np.std(a) > 0 and np.std(b) > 0 else 1.0

    def effective_rank(matrix: np.ndarray) -> float:
        eigenvalues = np.clip(np.linalg.eigvalsh(matrix @ matrix.T), 0.0, None)
        total = float(eigenvalues.sum())
        if total <= 0.0:
            return 0.0
        p = eigenvalues[eigenvalues > 0] / total
        return float(np.exp(-np.sum(p * np.log(p))))

    rank_left, rank_right = effective_rank(left), effective_rank(right)
    relative_rank_difference = abs(rank_left - rank_right) / max(rank_left, rank_right, 1e-30)
    return {
        "stable": bool(correlation >= 0.95 and relative_rank_difference <= 0.10),
        "off_diagonal_cosine_correlation": correlation,
        "effective_rank_first": rank_left,
        "effective_rank_second": rank_right,
        "effective_rank_relative_difference": relative_rank_difference,
    }


def caption_tag_metrics(captions: Iterable[str]) -> dict[str, Any]:
    rows = [tuple(tag.strip() for tag in str(caption).split(",") if tag.strip()) for caption in captions]
    counts = Counter(tag for row in rows for tag in set(row))
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()] if total else []
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    sorted_counts = sorted(counts.values())
    if sorted_counts and sum(sorted_counts) > 0:
        n = len(sorted_counts)
        gini = (2 * sum((index + 1) * value for index, value in enumerate(sorted_counts)) / (n * sum(sorted_counts))) - (n + 1) / n
    else:
        gini = 0.0
    tags = sorted(counts)
    cooccurrence = np.zeros((len(tags), len(tags)), dtype=np.float64)
    lookup = {tag: index for index, tag in enumerate(tags)}
    for row in rows:
        unique = sorted(set(row))
        for left in unique:
            for right in unique:
                cooccurrence[lookup[left], lookup[right]] += 1.0
    singular = np.linalg.svd(cooccurrence, compute_uv=False) if cooccurrence.size else np.zeros(0)
    power = singular * singular
    if power.sum() > 0:
        p = power[power > 0] / power.sum()
        cooccurrence_rank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        cooccurrence_rank = 0.0
    jaccards = []
    for left, right in itertools.combinations((set(row) for row in rows), 2):
        union = left | right
        jaccards.append(len(left & right) / len(union) if union else 1.0)
    return {
        "caption_count": len(rows),
        "unique_tag_count": len(counts),
        "tag_entropy": entropy,
        "tag_gini": gini,
        "tag_cooccurrence_effective_rank": cooccurrence_rank,
        "singleton_tag_fraction": sum(1 for count in counts.values() if count == 1) / max(len(counts), 1),
        "reusable_tag_fraction": sum(1 for count in counts.values() if count >= 2) / max(len(counts), 1),
        "caption_pair_jaccard_mean": _median(jaccards) if jaccards else None,
    }
