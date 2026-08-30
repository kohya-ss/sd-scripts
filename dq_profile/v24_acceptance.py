from __future__ import annotations

"""DQ Profiler v2.4 local numerical-acceptance analysis.

The diagnostic deliberately separates the usual local deformation (Body),
the worst timestep-bin deformation (Tail), and Tail/Body amplification.  It
does not predict final image quality.  Candidate reduction is conservative:
only candidates that are Pareto-dominated with high source-bootstrap support
may be removed.
"""

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dq_profile.v23_safety import _bootstrap_ci, canonical_json_sha256


LOCAL_ACCEPTANCE_SCHEMA_VERSION = "2.4.0-local"
LOCAL_ACCEPTANCE_METRIC_VERSION = "2.4.0"
LOCAL_SELECTION_SCHEMA_VERSION = "2.4.0-local-selection"

# Compatibility aliases retained for existing imports and serialized readers.
SCHEMA_VERSION = LOCAL_ACCEPTANCE_SCHEMA_VERSION
METRIC_DEFINITION_VERSION = LOCAL_ACCEPTANCE_METRIC_VERSION
SELECTION_SCHEMA_VERSION = LOCAL_SELECTION_SCHEMA_VERSION
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 2401
CORE_GRID = (2.70, 3.15, 3.45)
DOMINANCE_PROBABILITY = 0.80
MAX_FORMAL_CANDIDATES = 3
MINIMUM_IMAGES = 8
MINIMUM_SOURCE_GROUPS = 4

DISTANCE_FIELDS = {
    "relative": "relative_gradient_distance",
    "symmetric": "symmetric_gradient_distance",
    "angle": "angular_gradient_distance",
    "gain": "gradient_gain_distance",
}


def acceptance_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "diagnostic_target": "numerical_gradient_acceptance_by_fixed_range_mul",
        "not_quality_or_utility": True,
        "primary_channels": {
            "local_body": "source-balanced q95(d) pooled across timestep bins",
            "local_tail": "max_t source-balanced q95(d | timestep bin)",
            "tail_amplification": "local_tail/local_body",
            "trajectory": "measured only in the formal 128-step stage",
        },
        "relative_distance": "||g_m-g_0||/||g_0||",
        "cause_decomposition": {
            "symmetric": "2||g_m-g_0||/(||g_m||+||g_0||)",
            "angle": "sqrt(2(1-cosine))",
            "gain": "abs(log(||g_m||/||g_0||))",
            "absolute": [
                "grad_norm_noquant",
                "grad_norm_candidate",
                "grad_diff_norm",
            ],
        },
        "bootstrap": {
            "primary_unit": "source_group_cluster_equal_weight",
            "secondary_unit": "image_key_block",
            "shared_draws_across_candidates": True,
            "iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
        },
        "candidate_reduction": {
            "rule": "drop_only_robustly_body_tail_pareto_dominated",
            "dominance_probability": DOMINANCE_PROBABILITY,
            "always_retain": [
                "point_body_min",
                "point_tail_min",
                "bootstrap_modal_body",
                "bootstrap_modal_tail",
            ],
            "maximum_formal_candidates": MAX_FORMAL_CANDIDATES,
            "credible_set_at_least_4": "increase_local_evidence_or_abstain",
        },
        "core_grid": list(CORE_GRID),
        "edge_rule": (
            "a retained measured-grid endpoint requests one outside local-only "
            "point; no resolved claim while edge_unresolved"
        ),
        "score": (
            "100/(1+channel) may be shown only as a perturbation/fidelity gauge; "
            "it is not a quality score"
        ),
        "automated_claims_excluded": [
            "best_final_quality_mul",
            "training_success_guarantee",
            "failure_probability",
            "quantization_utility",
        ],
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def _finite(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected finite numeric value, got {value!r}")
    return number


def _optional_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _same_mul(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _weighted_quantile(
    values: Sequence[float],
    weights: Sequence[float],
    quantile: float,
) -> float:
    data = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(data) & np.isfinite(mass) & (mass > 0.0)
    data = data[valid]
    mass = mass[valid]
    if data.size == 0:
        raise ValueError("weighted quantile has no finite positive-weight values")
    order = np.argsort(data, kind="stable")
    data = data[order]
    mass = mass[order]
    cumulative = np.cumsum(mass)
    target = min(1.0, max(0.0, float(quantile))) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(data[min(index, data.size - 1)])


def _cluster_values(
    members_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_groups: Sequence[str],
    field: str,
) -> tuple[list[float], list[float]]:
    values: list[float] = []
    weights: list[float] = []
    for source_group in selected_groups:
        members = members_by_group.get(source_group, ())
        finite = [
            number
            for row in members
            if (number := _optional_finite(row.get(field))) is not None
        ]
        if not finite:
            continue
        per_observation = 1.0 / len(finite)
        values.extend(finite)
        weights.extend([per_observation] * len(finite))
    return values, weights


def _cluster_quantile(
    members_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_groups: Sequence[str],
    field: str,
    quantile: float = 0.95,
) -> float:
    values, weights = _cluster_values(members_by_group, selected_groups, field)
    return _weighted_quantile(values, weights, quantile)


def _cluster_rate(
    members_by_group: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_groups: Sequence[str],
    predicate: Any,
) -> float:
    source_rates: list[float] = []
    for source_group in selected_groups:
        members = list(members_by_group.get(source_group, ()))
        if members:
            source_rates.append(float(np.mean([bool(predicate(row)) for row in members])))
    return float(np.mean(source_rates)) if source_rates else float("nan")


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_bins: int,
) -> tuple[
    dict[str, list[Mapping[str, Any]]],
    dict[int, dict[str, list[Mapping[str, Any]]]],
]:
    all_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_bin: dict[int, dict[str, list[Mapping[str, Any]]]] = {
        index: defaultdict(list) for index in range(timestep_bins)
    }
    for row in rows:
        source_group = str(row["source_group"])
        timestep_bin = int(row["timestep_bin"])
        all_groups[source_group].append(row)
        by_bin[timestep_bin][source_group].append(row)
    return dict(all_groups), {
        key: dict(value) for key, value in by_bin.items()
    }


def _body_tail(
    all_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    by_bin: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    selected_groups: Sequence[str],
    *,
    field: str,
    timestep_bins: int,
) -> dict[str, Any]:
    body = _cluster_quantile(all_groups, selected_groups, field)
    bin_values = [
        _cluster_quantile(by_bin[index], selected_groups, field)
        for index in range(timestep_bins)
    ]
    worst_bin = min(
        range(timestep_bins),
        key=lambda index: (-bin_values[index], index),
    )
    tail = float(bin_values[worst_bin])
    return {
        "body": body,
        "tail": tail,
        "amplification": tail / max(body, 1e-30),
        "worst_timestep_bin": int(worst_bin),
        "per_bin": bin_values,
    }


def _winner_probabilities(
    candidates: Sequence[str],
    draws: Mapping[str, np.ndarray],
) -> dict[str, float]:
    if not candidates:
        return {}
    matrix = np.stack([draws[candidate] for candidate in candidates], axis=0)
    minimum = np.min(matrix, axis=0)
    ties = np.isclose(matrix, minimum[None, :], rtol=1e-12, atol=1e-12)
    weights = ties / np.maximum(np.sum(ties, axis=0, keepdims=True), 1)
    return {
        candidate: float(np.mean(weights[index]))
        for index, candidate in enumerate(candidates)
    }


def _candidate_contract(
    summary: Mapping[str, Any],
) -> tuple[list[str], dict[str, float], dict[str, bool]]:
    if str(summary.get("schema_version")) != "2.1.0":
        raise ValueError("v2.4 analysis requires a schema 2.1.0 runtime profile")
    protocol = str(summary.get("profile", {}).get("protocol"))
    if protocol not in {
        "v24-acceptance-local",
        "v24-acceptance-formal",
        "v24-trajectory-descriptive",
    }:
        raise ValueError("v2.4 analysis requires a v24 acceptance runtime profile")
    candidates: list[tuple[float, str]] = []
    hard_pass: dict[str, bool] = {}
    for row in summary.get("candidates", ()):  # type: ignore[union-attr]
        candidate = str(row.get("candidate"))
        if candidate == "no_quant":
            continue
        value = row.get("initial_range_mul")
        if value is None:
            continue
        candidates.append((_finite(value), candidate))
        hard_pass[candidate] = not bool(row.get("forced_safety_abort")) and not bool(
            row.get("invalid_reason")
        )
    candidates.sort()
    if not candidates:
        raise ValueError("v2.4 profile has no fixed-mul candidate")
    return (
        [candidate for _, candidate in candidates],
        {candidate: value for value, candidate in candidates},
        hard_pass,
    )


def _phenotype(core_rows: Sequence[Mapping[str, Any]]) -> str:
    if not core_rows:
        return "unknown"
    if all(
        max(float(row["local_body_ci_high"]), float(row["local_tail_ci_high"])) < 1.0
        for row in core_rows
    ):
        return "broad_observed_below_anchor"
    if all(
        max(float(row["local_body_ci_low"]), float(row["local_tail_ci_low"])) >= 1.0
        for row in core_rows
    ):
        return "all_high_perturbation"
    point_alarm = [
        max(float(row["local_body"]), float(row["local_tail"]))
        for row in core_rows
    ]
    if min(point_alarm) < 1.0 <= max(point_alarm):
        return "selective_window"
    return "unknown"


def analyze_natural_gradient_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_bins: int,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED + 1,
) -> dict[str, Any]:
    """Summarize local no-quant gradient variation using source clusters.

    These rows compare independent no-quant noise replicas for the same
    image/timestep probe. They are a descriptive natural-gradient baseline,
    not a denominator for the local candidate selector.
    """

    if timestep_bins <= 0 or bootstrap_iterations <= 0:
        raise ValueError("natural-gradient analysis requires positive bins/iterations")
    normalized: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if not bool(row.get("gradient_topology_matches", True)):
            raise ValueError("no-quant natural-gradient topology mismatch")
        image_key = str(row.get("image_key", ""))
        source_group = str(row.get("source_group", "") or image_key)
        if not image_key or not source_group:
            raise ValueError("no-quant natural-gradient row lacks image/source key")
        converted = {
            **row,
            "source_group": source_group,
            "relative_gradient_distance": _finite(
                row.get("relative_gradient_distance_a_to_b")
            ),
            "grad_norm_noquant": _finite(row.get("grad_norm_a")),
            "grad_norm_candidate": _finite(row.get("grad_norm_b")),
        }
        for field in (
            "grad_diff_norm",
            "symmetric_gradient_distance",
            "angular_gradient_distance",
            "gradient_gain_distance",
        ):
            converted[field] = _finite(row.get(field))
        normalized.append(converted)
    if not normalized:
        return {
            "valid": False,
            "invalid_reason": "no_local_natural_gradient_rows",
            "selector_input": False,
        }

    all_groups, by_bin = _index_rows(normalized, timestep_bins=timestep_bins)
    source_groups = sorted(all_groups)
    if len(source_groups) < MINIMUM_SOURCE_GROUPS:
        return {
            "valid": False,
            "invalid_reason": "fewer_than_four_source_groups",
            "source_group_count": len(source_groups),
            "selector_input": False,
        }
    rng = np.random.default_rng(int(bootstrap_seed))
    draw_indices = rng.integers(
        0,
        len(source_groups),
        size=(int(bootstrap_iterations), len(source_groups)),
    )
    point = {
        name: _body_tail(
            all_groups,
            by_bin,
            source_groups,
            field=field,
            timestep_bins=timestep_bins,
        )
        for name, field in DISTANCE_FIELDS.items()
    }
    body_draws = np.empty(bootstrap_iterations, dtype=np.float64)
    tail_draws = np.empty(bootstrap_iterations, dtype=np.float64)
    for iteration, indices in enumerate(draw_indices):
        selected = [source_groups[int(index)] for index in indices]
        values = _body_tail(
            all_groups,
            by_bin,
            selected,
            field=DISTANCE_FIELDS["relative"],
            timestep_bins=timestep_bins,
        )
        body_draws[iteration] = values["body"]
        tail_draws[iteration] = values["tail"]
    body_ci = _bootstrap_ci(body_draws)
    tail_ci = _bootstrap_ci(tail_draws)
    return {
        "valid": True,
        "probe_regime": "structural_dropout_off",
        "pair_count": len(normalized),
        "image_count": len({str(row["image_key"]) for row in normalized}),
        "source_group_count": len(source_groups),
        "local_body": point["relative"]["body"],
        "local_body_ci_low": body_ci["ci_low"],
        "local_body_ci_high": body_ci["ci_high"],
        "local_tail": point["relative"]["tail"],
        "local_tail_ci_low": tail_ci["ci_low"],
        "local_tail_ci_high": tail_ci["ci_high"],
        "tail_amplification": point["relative"]["amplification"],
        "worst_timestep_bin": point["relative"]["worst_timestep_bin"],
        "symmetric_body": point["symmetric"]["body"],
        "symmetric_tail": point["symmetric"]["tail"],
        "angle_body": point["angle"]["body"],
        "angle_tail": point["angle"]["tail"],
        "gain_body": point["gain"]["body"],
        "gain_tail": point["gain"]["tail"],
        "bootstrap_unit": "source_group_cluster_equal_weight",
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "selector_input": False,
        "interpretation": "no_quant_local_natural_gradient_variation",
    }


def analyze_local_profile(
    *,
    summary: Mapping[str, Any],
    gradient_tail_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    core_grid: Sequence[float] = CORE_GRID,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    selection_enabled: bool = True,
) -> dict[str, Any]:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    candidates, range_by_candidate, hard_pass = _candidate_contract(summary)
    timestep_bins = int(summary.get("profile", {}).get("timestep_bins", 0))
    if timestep_bins <= 0:
        raise ValueError("profile timestep bin count must be positive")
    samples: list[dict[str, Any]] = []
    for raw in gradient_tail_rows:
        if str(raw.get("record_type")) != "sample":
            continue
        candidate = str(raw.get("candidate"))
        if candidate not in range_by_candidate:
            continue
        row = dict(raw)
        image_key = str(row.get("image_key", ""))
        if not image_key:
            raise ValueError("gradient-tail sample has no image_key")
        source_group = str(row.get("source_group", "") or image_key)
        row["source_group"] = source_group
        for field in (
            "relative_gradient_distance",
            "gradient_cosine",
            "grad_norm_noquant",
            "grad_norm_candidate",
            "grad_diff_norm",
            "symmetric_gradient_distance",
            "angular_gradient_distance",
            "gradient_gain_distance",
        ):
            if _optional_finite(row.get(field)) is None:
                if field == "gradient_gain_distance":
                    continue
                raise ValueError(f"v2.4 sample is missing finite {field}: {row!r}")
        samples.append(row)
    if not samples:
        raise ValueError("gradient_tail.csv has no v2.4 candidate samples")

    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    key_sets: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    image_to_source: dict[str, str] = {}
    for row in samples:
        candidate = str(row["candidate"])
        by_candidate[candidate].append(row)
        image_key = str(row["image_key"])
        source_group = str(row["source_group"])
        previous = image_to_source.setdefault(image_key, source_group)
        if previous != source_group:
            raise ValueError(f"image maps to multiple source groups: {image_key}")
        key_sets[candidate].add(
            (
                image_key,
                int(row["timestep_bin"]),
                int(row["noise_replica"]),
                int(row["quant_repeat"]),
            )
        )
    reference_keys = key_sets[candidates[0]]
    for candidate in candidates:
        if key_sets[candidate] != reference_keys:
            raise ValueError(f"candidate probe key mismatch for {candidate}")

    image_keys = sorted(image_to_source)
    source_groups = sorted(set(image_to_source.values()))
    if len(image_keys) < MINIMUM_IMAGES:
        raise ValueError(
            f"v2.4 local analysis requires at least {MINIMUM_IMAGES} images"
        )
    if len(source_groups) < MINIMUM_SOURCE_GROUPS:
        raise ValueError(
            f"v2.4 source bootstrap requires at least {MINIMUM_SOURCE_GROUPS} groups"
        )

    indexes: dict[
        str,
        tuple[
            dict[str, list[Mapping[str, Any]]],
            dict[int, dict[str, list[Mapping[str, Any]]]],
        ],
    ] = {
        candidate: _index_rows(by_candidate[candidate], timestep_bins=timestep_bins)
        for candidate in candidates
    }
    rng = np.random.default_rng(int(bootstrap_seed))
    source_draw_indices = rng.integers(
        0,
        len(source_groups),
        size=(int(bootstrap_iterations), len(source_groups)),
    )
    image_draw_indices = rng.integers(
        0,
        len(image_keys),
        size=(int(bootstrap_iterations), len(image_keys)),
    )
    image_as_group: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    image_as_group_by_bin: dict[
        str,
        dict[int, dict[str, list[Mapping[str, Any]]]],
    ] = {}
    for candidate in candidates:
        all_images: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        bin_images: dict[int, dict[str, list[Mapping[str, Any]]]] = {
            index: defaultdict(list) for index in range(timestep_bins)
        }
        for row in by_candidate[candidate]:
            image_key = str(row["image_key"])
            bin_index = int(row["timestep_bin"])
            all_images[image_key].append(row)
            bin_images[bin_index][image_key].append(row)
        image_as_group[candidate] = dict(all_images)
        image_as_group_by_bin[candidate] = {
            key: dict(value) for key, value in bin_images.items()
        }

    point_by_candidate: dict[str, dict[str, Any]] = {}
    source_draws: dict[str, dict[str, np.ndarray]] = {}
    image_draws: dict[str, dict[str, np.ndarray]] = {}
    bootstrap_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        all_groups, by_bin = indexes[candidate]
        primary = _body_tail(
            all_groups,
            by_bin,
            source_groups,
            field=DISTANCE_FIELDS["relative"],
            timestep_bins=timestep_bins,
        )
        decompositions = {
            name: _body_tail(
                all_groups,
                by_bin,
                source_groups,
                field=field,
                timestep_bins=timestep_bins,
            )
            for name, field in DISTANCE_FIELDS.items()
        }
        point_by_candidate[candidate] = {
            "primary": primary,
            "decompositions": decompositions,
            "d_gt_1_rate": _cluster_rate(
                all_groups,
                source_groups,
                lambda row: float(row["relative_gradient_distance"]) > 1.0,
            ),
            "gradient_cosine_lt_0_rate": _cluster_rate(
                all_groups,
                source_groups,
                lambda row: float(row["gradient_cosine"]) < 0.0,
            ),
            "grad_norm_noquant_q05": _cluster_quantile(
                all_groups, source_groups, "grad_norm_noquant", 0.05
            ),
            "grad_norm_noquant_median": _cluster_quantile(
                all_groups, source_groups, "grad_norm_noquant", 0.50
            ),
            "grad_norm_candidate_median": _cluster_quantile(
                all_groups, source_groups, "grad_norm_candidate", 0.50
            ),
            "grad_diff_norm_q95": _cluster_quantile(
                all_groups, source_groups, "grad_diff_norm", 0.95
            ),
        }
        candidate_source = {
            "body": np.empty(bootstrap_iterations, dtype=np.float64),
            "tail": np.empty(bootstrap_iterations, dtype=np.float64),
            "amplification": np.empty(bootstrap_iterations, dtype=np.float64),
        }
        candidate_image = {
            "body": np.empty(bootstrap_iterations, dtype=np.float64),
            "tail": np.empty(bootstrap_iterations, dtype=np.float64),
        }
        for name in DISTANCE_FIELDS:
            if name != "relative":
                candidate_source[f"{name}_body"] = np.empty(
                    bootstrap_iterations, dtype=np.float64
                )
                candidate_source[f"{name}_tail"] = np.empty(
                    bootstrap_iterations, dtype=np.float64
                )
        for iteration in range(bootstrap_iterations):
            selected_sources = [
                source_groups[int(index)] for index in source_draw_indices[iteration]
            ]
            selected_images = [
                image_keys[int(index)] for index in image_draw_indices[iteration]
            ]
            source_primary = _body_tail(
                all_groups,
                by_bin,
                selected_sources,
                field=DISTANCE_FIELDS["relative"],
                timestep_bins=timestep_bins,
            )
            image_primary = _body_tail(
                image_as_group[candidate],
                image_as_group_by_bin[candidate],
                selected_images,
                field=DISTANCE_FIELDS["relative"],
                timestep_bins=timestep_bins,
            )
            candidate_source["body"][iteration] = source_primary["body"]
            candidate_source["tail"][iteration] = source_primary["tail"]
            candidate_source["amplification"][iteration] = source_primary[
                "amplification"
            ]
            candidate_image["body"][iteration] = image_primary["body"]
            candidate_image["tail"][iteration] = image_primary["tail"]
            for name, field in DISTANCE_FIELDS.items():
                if name == "relative":
                    continue
                values = _body_tail(
                    all_groups,
                    by_bin,
                    selected_sources,
                    field=field,
                    timestep_bins=timestep_bins,
                )
                candidate_source[f"{name}_body"][iteration] = values["body"]
                candidate_source[f"{name}_tail"][iteration] = values["tail"]
        source_draws[candidate] = candidate_source
        image_draws[candidate] = candidate_image
        for iteration in range(bootstrap_iterations):
            bootstrap_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "iteration": iteration,
                    "bootstrap_unit": "source_group_cluster_equal_weight",
                    "local_body": float(candidate_source["body"][iteration]),
                    "local_tail": float(candidate_source["tail"][iteration]),
                    "tail_amplification": float(
                        candidate_source["amplification"][iteration]
                    ),
                    "symmetric_body": float(
                        candidate_source["symmetric_body"][iteration]
                    ),
                    "symmetric_tail": float(
                        candidate_source["symmetric_tail"][iteration]
                    ),
                    "angle_body": float(candidate_source["angle_body"][iteration]),
                    "angle_tail": float(candidate_source["angle_tail"][iteration]),
                    "gain_body": float(candidate_source["gain_body"][iteration]),
                    "gain_tail": float(candidate_source["gain_tail"][iteration]),
                }
            )

    body_draw_matrix = np.stack([source_draws[name]["body"] for name in candidates])
    tail_draw_matrix = np.stack([source_draws[name]["tail"] for name in candidates])
    body_regret = body_draw_matrix - np.min(body_draw_matrix, axis=0, keepdims=True)
    tail_regret = tail_draw_matrix - np.min(tail_draw_matrix, axis=0, keepdims=True)
    body_probabilities = _winner_probabilities(
        candidates, {name: source_draws[name]["body"] for name in candidates}
    )
    tail_probabilities = _winner_probabilities(
        candidates, {name: source_draws[name]["tail"] for name in candidates}
    )
    body_modal = max(
        candidates,
        key=lambda name: (body_probabilities[name], -range_by_candidate[name]),
    )
    tail_modal = max(
        candidates,
        key=lambda name: (tail_probabilities[name], -range_by_candidate[name]),
    )
    body_point = min(
        candidates,
        key=lambda name: (
            point_by_candidate[name]["primary"]["body"],
            range_by_candidate[name],
        ),
    )
    tail_point = min(
        candidates,
        key=lambda name: (
            point_by_candidate[name]["primary"]["tail"],
            range_by_candidate[name],
        ),
    )

    dominance_rows: list[dict[str, Any]] = []
    dominated_by: dict[str, tuple[str, float] | None] = {
        candidate: None for candidate in candidates
    }
    for left in candidates:
        left_index = candidates.index(left)
        for right in candidates:
            if left == right:
                continue
            right_index = candidates.index(right)
            dominates = (
                (body_draw_matrix[left_index] <= body_draw_matrix[right_index])
                & (tail_draw_matrix[left_index] <= tail_draw_matrix[right_index])
                & (
                    (body_draw_matrix[left_index] < body_draw_matrix[right_index])
                    | (tail_draw_matrix[left_index] < tail_draw_matrix[right_index])
                )
            )
            probability = float(np.mean(dominates))
            dominance_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dominator": left,
                    "dominator_mul": range_by_candidate[left],
                    "candidate": right,
                    "candidate_mul": range_by_candidate[right],
                    "probability_body_tail_pareto_dominated": probability,
                    "elimination_boundary": DOMINANCE_PROBABILITY,
                }
            )
            previous = dominated_by[right]
            if probability >= DOMINANCE_PROBABILITY and (
                previous is None or probability > previous[1]
            ):
                dominated_by[right] = (left, probability)

    mandatory = {body_point, tail_point, body_modal, tail_modal}
    retained = [
        candidate
        for candidate in candidates
        if hard_pass[candidate]
        and (dominated_by[candidate] is None or candidate in mandatory)
    ]
    retained.sort(key=lambda name: range_by_candidate[name])
    formal_selection_valid = bool(
        selection_enabled
        and 1 <= len(retained) <= MAX_FORMAL_CANDIDATES
        and all(hard_pass[name] for name in retained)
    )

    ordered = sorted(candidates, key=lambda name: range_by_candidate[name])
    endpoint_retained = [
        candidate for candidate in (ordered[0], ordered[-1]) if candidate in retained
    ]
    edge_unresolved = bool(endpoint_retained)
    edge_recommendations: list[float] = []
    if edge_unresolved and len(ordered) >= 2:
        low = range_by_candidate[ordered[0]]
        high = range_by_candidate[ordered[-1]]
        if ordered[0] in endpoint_retained:
            step = range_by_candidate[ordered[1]] - low
            edge_recommendations.append(round(max(0.01, low - step), 2))
        if ordered[-1] in endpoint_retained:
            step = high - range_by_candidate[ordered[-2]]
            edge_recommendations.append(round(high + step, 2))

    core_values = tuple(float(value) for value in core_grid)
    core_candidates = [
        candidate
        for candidate in candidates
        if any(_same_mul(range_by_candidate[candidate], value) for value in core_values)
    ]
    if selection_enabled and len(core_candidates) != len(core_values):
        missing = [
            value
            for value in core_values
            if not any(_same_mul(range_by_candidate[name], value) for name in candidates)
        ]
        raise ValueError(f"v2.4 local run is missing common core grid points: {missing}")

    score_rows: list[dict[str, Any]] = []
    timestep_rows: list[dict[str, Any]] = []
    regret_rows: list[dict[str, Any]] = []
    source_loo_rows: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates):
        point = point_by_candidate[candidate]
        primary = point["primary"]
        body_ci = _bootstrap_ci(source_draws[candidate]["body"])
        tail_ci = _bootstrap_ci(source_draws[candidate]["tail"])
        amplification_ci = _bootstrap_ci(source_draws[candidate]["amplification"])
        image_body_ci = _bootstrap_ci(image_draws[candidate]["body"])
        image_tail_ci = _bootstrap_ci(image_draws[candidate]["tail"])
        body_floor = float(np.std(source_draws[candidate]["body"]))
        tail_floor = float(np.std(source_draws[candidate]["tail"]))
        dominated = dominated_by[candidate]
        role = (
            "core_grid"
            if candidate in core_candidates
            else "edge_extension"
        )
        row = {
            "dataset_id": dataset_id,
            "candidate": candidate,
            "range_mul": range_by_candidate[candidate],
            "grid_role": role,
            "hard_safety_pass": hard_pass[candidate],
            "local_body": float(primary["body"]),
            "local_body_ci_low": body_ci["ci_low"],
            "local_body_ci_high": body_ci["ci_high"],
            "local_tail": float(primary["tail"]),
            "local_tail_ci_low": tail_ci["ci_low"],
            "local_tail_ci_high": tail_ci["ci_high"],
            "tail_amplification": float(primary["amplification"]),
            "tail_amplification_ci_low": amplification_ci["ci_low"],
            "tail_amplification_ci_high": amplification_ci["ci_high"],
            "worst_timestep_bin": int(primary["worst_timestep_bin"]),
            "d_gt_1_rate": point["d_gt_1_rate"],
            "gradient_cosine_lt_0_rate": point["gradient_cosine_lt_0_rate"],
            "grad_norm_noquant_q05": point["grad_norm_noquant_q05"],
            "grad_norm_noquant_median": point["grad_norm_noquant_median"],
            "grad_norm_candidate_median": point["grad_norm_candidate_median"],
            "grad_diff_norm_q95": point["grad_diff_norm_q95"],
            "symmetric_body": point["decompositions"]["symmetric"]["body"],
            "symmetric_tail": point["decompositions"]["symmetric"]["tail"],
            "angle_body": point["decompositions"]["angle"]["body"],
            "angle_tail": point["decompositions"]["angle"]["tail"],
            "gain_body": point["decompositions"]["gain"]["body"],
            "gain_tail": point["decompositions"]["gain"]["tail"],
            "source_bootstrap_body_min_probability": body_probabilities[candidate],
            "source_bootstrap_tail_min_probability": tail_probabilities[candidate],
            "image_bootstrap_body_ci_low": image_body_ci["ci_low"],
            "image_bootstrap_body_ci_high": image_body_ci["ci_high"],
            "image_bootstrap_tail_ci_low": image_tail_ci["ci_low"],
            "image_bootstrap_tail_ci_high": image_tail_ci["ci_high"],
            "body_regret_median": float(np.median(body_regret[candidate_index])),
            "body_regret_ci_high": float(
                np.quantile(body_regret[candidate_index], 0.975)
            ),
            "tail_regret_median": float(np.median(tail_regret[candidate_index])),
            "tail_regret_ci_high": float(
                np.quantile(tail_regret[candidate_index], 0.975)
            ),
            "body_regret_within_own_sampling_sd_probability": float(
                np.mean(body_regret[candidate_index] <= body_floor)
            ),
            "tail_regret_within_own_sampling_sd_probability": float(
                np.mean(tail_regret[candidate_index] <= tail_floor)
            ),
            "robustly_dominated": dominated is not None and candidate not in mandatory,
            "dominated_by": None if dominated is None else dominated[0],
            "dominance_probability": None if dominated is None else dominated[1],
            "mandatory_retention_role": sorted(
                role_name
                for role_name, member in (
                    ("point_body_min", body_point),
                    ("point_tail_min", tail_point),
                    ("bootstrap_modal_body", body_modal),
                    ("bootstrap_modal_tail", tail_modal),
                )
                if member == candidate
            ),
            "retained_for_formal": candidate in retained,
            "trajectory_risk_T": None,
            "conservative_alarm_R": None,
            "perturbation_gauge": None,
            "not_quality_or_utility": True,
        }
        score_rows.append(row)
        for bin_index, value in enumerate(primary["per_bin"]):
            timestep_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "timestep_bin": bin_index,
                    "source_balanced_q95_relative_distance": float(value),
                    "is_worst_timestep_bin": bin_index
                    == int(primary["worst_timestep_bin"]),
                }
            )
        for iteration in range(bootstrap_iterations):
            regret_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "iteration": iteration,
                    "body_regret": float(body_regret[candidate_index, iteration]),
                    "tail_regret": float(tail_regret[candidate_index, iteration]),
                }
            )
        all_groups, by_bin = indexes[candidate]
        for omitted in source_groups:
            kept = [value for value in source_groups if value != omitted]
            values = _body_tail(
                all_groups,
                by_bin,
                kept,
                field=DISTANCE_FIELDS["relative"],
                timestep_bins=timestep_bins,
            )
            source_loo_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "omitted_source_group": omitted,
                    "remaining_source_groups": len(kept),
                    "local_body": values["body"],
                    "local_tail": values["tail"],
                    "tail_amplification": values["amplification"],
                }
            )

    core_rows = [row for row in score_rows if row["grid_role"] == "core_grid"]
    core_body_matrix = np.stack(
        [source_draws[row["candidate"]]["body"] for row in core_rows]
    )
    core_tail_matrix = np.stack(
        [source_draws[row["candidate"]]["tail"] for row in core_rows]
    )
    core_envelope = {
        "grid": list(core_values),
        "candidate_count": len(core_rows),
        "max_local_body": max(float(row["local_body"]) for row in core_rows),
        "max_local_tail": max(float(row["local_tail"]) for row in core_rows),
        "probability_all_core_body_below_anchor": float(
            np.mean(np.max(core_body_matrix, axis=0) < 1.0)
        ),
        "probability_all_core_tail_below_anchor": float(
            np.mean(np.max(core_tail_matrix, axis=0) < 1.0)
        ),
        "scope": "common_core_grid_only",
    }
    selection_reason = (
        "disabled_for_formal_reanalysis"
        if not selection_enabled
        else "credible_set_too_large_expand_local_or_abstain"
        if len(retained) > MAX_FORMAL_CANDIDATES
        else "no_hard_safety_candidate"
        if not retained
        else "valid"
    )
    selection = {
        "selection_valid": formal_selection_valid,
        "selection_status": (
            "edge_unresolved"
            if formal_selection_valid and edge_unresolved
            else "ready_for_formal"
            if formal_selection_valid
            else selection_reason
        ),
        "selected_candidates": retained if formal_selection_valid else [],
        "selected_muls": (
            [range_by_candidate[name] for name in retained]
            if formal_selection_valid
            else []
        ),
        "credible_candidates": retained,
        "credible_muls": [range_by_candidate[name] for name in retained],
        "credible_candidate_count": len(retained),
        "robustly_dominated_candidates": [
            name
            for name in candidates
            if dominated_by[name] is not None and name not in mandatory
        ],
        "point_body_min_candidate": body_point,
        "point_tail_min_candidate": tail_point,
        "bootstrap_modal_body_candidate": body_modal,
        "bootstrap_modal_tail_candidate": tail_modal,
        "edge_unresolved": edge_unresolved,
        "retained_endpoint_candidates": endpoint_retained,
        "edge_extension_recommended": edge_recommendations,
        "ranking_resolved": False,
        "best_quality_mul": None,
        "utility": "unknown",
    }
    contract = acceptance_contract()
    summary_result = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "dataset_id": dataset_id,
        "diagnostic_target": "numerical_gradient_acceptance_by_fixed_range_mul",
        "not_quality_or_utility": True,
        "candidate_grid": [range_by_candidate[name] for name in candidates],
        "core_grid": list(core_values),
        "edge_extension": [
            range_by_candidate[name]
            for name in candidates
            if name not in core_candidates
        ],
        "image_count": len(image_keys),
        "source_group_count": len(source_groups),
        "source_groups": source_groups,
        "timestep_bins": timestep_bins,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "primary_bootstrap_unit": "source_group_cluster_equal_weight",
        "secondary_bootstrap_unit": "image_key_block",
        "local_phenotype": _phenotype(core_rows),
        "core_grid_envelope": core_envelope,
        "selection": selection,
        "selection_rule": contract["candidate_reduction"],
        "acceptance_contract_sha256": contract["contract_sha256"],
        "confidence": (
            "limited_independent_sources"
            if len(source_groups) < 8
            else "moderate_descriptive"
        ),
        "threshold_anchor_1_has_mathematical_meaning": True,
        "threshold_0_5_is_not_a_formal_classification": True,
    }
    return {
        "contract": contract,
        "summary": summary_result,
        "score_rows": score_rows,
        "timestep_rows": timestep_rows,
        "bootstrap_rows": bootstrap_rows,
        "regret_rows": regret_rows,
        "dominance_rows": dominance_rows,
        "source_loo_rows": source_loo_rows,
        "selection": selection,
        "_source_draws": source_draws,
    }
