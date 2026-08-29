from __future__ import annotations

"""Local-only stage for the v2.3.2 numerical-acceptance protocol.

This module deliberately does not estimate the formal trajectory channel T.
It measures the local gradient-tail channel L and preregisters which two or
three fixed-mul candidates may proceed to the expensive 128-step stage.
"""

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence

import numpy as np

from dq_profile.v23_safety import (
    ANCHOR_RISK_BOUNDARY,
    TOLERANT_RISK_BOUNDARY,
    _bootstrap_ci,
    canonical_json_sha256,
)


SCHEMA_VERSION = "2.3.2-local"
METRIC_DEFINITION_VERSION = "2.3.2-local"
SELECTION_SCHEMA_VERSION = "2.3.2-local-selection"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 2322
ANCHOR_MUL = 3.15


def local_selection_rule() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": "point_min_anchor_nearest_neighbor_v1",
        "diagnostic_target": "local_gradient_acceptance_by_fixed_range_mul",
        "point_min": "minimum point L; ties select smaller range_mul",
        "anchor": "grid point nearest 3.15; ties select smaller range_mul",
        "neighbor": (
            "nearest unselected grid point to point_min; ties select smaller "
            "range_mul"
        ),
        "selected_count": "2-3 after role deduplication",
        "edge_unresolved": (
            "point_min is a tested-grid endpoint and point L >= 0.5"
        ),
        "formal_score_before_branch": "unknown",
        "local_score_ceiling": "100/(1+L)",
        "selector_or_utility_vote": False,
        "anchor_0_5_validated": False,
    }
    payload["sha256"] = canonical_json_sha256(payload)
    return payload


def _finite_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"expected a finite number, got {value!r}")
    return number


def _candidate_contract(summary: Mapping[str, Any]) -> tuple[list[str], dict[str, float]]:
    if str(summary.get("schema_version")) != "2.1.0":
        raise ValueError("local safety analysis requires a schema 2.1.0 profile")
    profile = summary.get("profile", {})
    if str(profile.get("protocol")) != "v23-safety-local":
        raise ValueError("local safety analysis requires v23-safety-local")
    candidates: list[tuple[float, str]] = []
    for row in summary.get("candidates", ()):  # type: ignore[union-attr]
        name = str(row.get("candidate"))
        if name == "no_quant":
            continue
        value = row.get("initial_range_mul")
        if value is None:
            continue
        candidates.append((_finite_float(value), name))
    candidates.sort()
    if len(candidates) < 3:
        raise ValueError("local safety scan requires at least three fixed-mul candidates")
    names = [name for _, name in candidates]
    return names, {name: value for value, name in candidates}


def _sample_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    values = np.asarray(
        [_finite_float(row["relative_gradient_distance"]) for row in rows],
        dtype=np.float64,
    )
    cosines = np.asarray(
        [_finite_float(row["gradient_cosine"]) for row in rows],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("cannot summarize an empty local-tail sample")
    return {
        "q90": float(np.quantile(values, 0.90)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "d_gt_1_rate": float(np.mean(values > 1.0)),
        "gradient_cosine_lt_0_rate": float(np.mean(cosines < 0.0)),
    }


def _classification(local_risk: float, *, hard_safety_pass: bool) -> str:
    if not hard_safety_pass:
        return "unsafe"
    if local_risk >= ANCHOR_RISK_BOUNDARY:
        return "local_anchor_exceeded"
    if local_risk >= TOLERANT_RISK_BOUNDARY:
        return "local_caution"
    return "local_observed_tolerant"


def _winner_probabilities(
    candidates: Sequence[str],
    local_draws: Mapping[str, np.ndarray],
) -> dict[str, float]:
    if not candidates:
        return {}
    iterations = len(local_draws[candidates[0]])
    counts = {candidate: 0.0 for candidate in candidates}
    for iteration in range(iterations):
        values = {
            candidate: float(local_draws[candidate][iteration])
            for candidate in candidates
        }
        best = min(values.values())
        winners = [
            candidate
            for candidate, value in values.items()
            if math.isclose(value, best, rel_tol=1e-12, abs_tol=1e-12)
        ]
        weight = 1.0 / len(winners)
        for candidate in winners:
            counts[candidate] += weight
    return {candidate: counts[candidate] / iterations for candidate in candidates}


def _select_candidates(
    score_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(score_rows, key=lambda row: float(row["range_mul"]))
    point_min = min(
        ordered,
        key=lambda row: (float(row["local_risk_L"]), float(row["range_mul"])),
    )
    anchor = min(
        ordered,
        key=lambda row: (abs(float(row["range_mul"]) - ANCHOR_MUL), float(row["range_mul"])),
    )
    selected_names: list[str] = []
    roles: dict[str, list[str]] = defaultdict(list)

    def add(row: Mapping[str, Any], role: str) -> None:
        candidate = str(row["candidate"])
        if candidate not in selected_names:
            selected_names.append(candidate)
        roles[candidate].append(role)

    add(point_min, "point_min_L")
    add(anchor, "anchor_nearest_3.15")
    remaining = [row for row in ordered if str(row["candidate"]) not in selected_names]
    if remaining:
        neighbor = min(
            remaining,
            key=lambda row: (
                abs(float(row["range_mul"]) - float(point_min["range_mul"])),
                float(row["range_mul"]),
            ),
        )
        add(neighbor, "nearest_unselected_neighbor_to_point_min")

    selected_rows = [
        row for row in ordered if str(row["candidate"]) in selected_names
    ]
    selected_muls = sorted(float(row["range_mul"]) for row in selected_rows)
    selected_candidates = [
        str(row["candidate"])
        for row in sorted(selected_rows, key=lambda row: float(row["range_mul"]))
    ]
    point_index = next(
        index
        for index, row in enumerate(ordered)
        if str(row["candidate"]) == str(point_min["candidate"])
    )
    endpoint = point_index in {0, len(ordered) - 1}
    edge_unresolved = endpoint and float(point_min["local_risk_L"]) >= TOLERANT_RISK_BOUNDARY
    return {
        "selection_valid": 2 <= len(selected_muls) <= 3,
        "point_min_candidate": str(point_min["candidate"]),
        "point_min_mul": float(point_min["range_mul"]),
        "point_min_local_risk_L": float(point_min["local_risk_L"]),
        "point_min_is_grid_endpoint": endpoint,
        "edge_unresolved": edge_unresolved,
        "edge_extension_recommended": edge_unresolved,
        "selected_muls": selected_muls,
        "selected_candidates": selected_candidates,
        "roles_by_candidate": {
            candidate: sorted(candidate_roles)
            for candidate, candidate_roles in sorted(roles.items())
        },
    }


def analyze_local_profile(
    *,
    summary: Mapping[str, Any],
    gradient_tail_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")
    candidates, range_by_candidate = _candidate_contract(summary)
    timestep_bins = int(summary.get("profile", {}).get("timestep_bins", 0))
    if timestep_bins <= 0:
        raise ValueError("profile timestep bin count must be positive")
    samples = [
        row
        for row in gradient_tail_rows
        if str(row.get("record_type")) == "sample"
        and str(row.get("candidate")) in range_by_candidate
    ]
    if not samples:
        raise ValueError("gradient_tail.csv has no local candidate samples")

    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    by_image: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    image_sets: dict[str, set[str]] = defaultdict(set)
    for row in samples:
        candidate = str(row["candidate"])
        timestep_bin = int(row["timestep_bin"])
        image_key = str(row["image_key"])
        groups[(candidate, timestep_bin)].append(row)
        by_image[(candidate, timestep_bin, image_key)].append(
            _finite_float(row["relative_gradient_distance"])
        )
        image_sets[candidate].add(image_key)
    reference_images = sorted(image_sets[candidates[0]])
    if len(reference_images) < 16:
        raise ValueError("local safety scan requires at least 16 unique images")
    for candidate in candidates:
        if image_sets[candidate] != set(reference_images):
            raise ValueError(f"candidate image set mismatch for {candidate}")
        for timestep_bin in range(timestep_bins):
            if not groups.get((candidate, timestep_bin)):
                raise ValueError(
                    f"missing local samples for {candidate}, timestep {timestep_bin}"
                )

    rng = np.random.default_rng(int(bootstrap_seed))
    image_draws = rng.integers(
        0,
        len(reference_images),
        size=(int(bootstrap_iterations), len(reference_images)),
    )
    timestep_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    local_draws: dict[str, np.ndarray] = {}
    ceiling_draws: dict[str, np.ndarray] = {}

    candidate_summary = {
        str(row.get("candidate")): row for row in summary.get("candidates", ())
    }
    for candidate in candidates:
        point_metrics: list[dict[str, Any]] = []
        for timestep_bin in range(timestep_bins):
            rows = groups[(candidate, timestep_bin)]
            metrics = _sample_metrics(rows)
            record = {
                "dataset_id": dataset_id,
                "candidate": candidate,
                "range_mul": range_by_candidate[candidate],
                "timestep_bin": timestep_bin,
                "sample_count": len(rows),
                "image_count": len({str(row["image_key"]) for row in rows}),
                **metrics,
            }
            point_metrics.append(record)
            timestep_rows.append(record)

        local = np.empty(int(bootstrap_iterations), dtype=np.float64)
        q99_draw = np.empty(int(bootstrap_iterations), dtype=np.float64)
        for iteration, sampled_indices in enumerate(image_draws):
            sampled_images = [reference_images[int(index)] for index in sampled_indices]
            per_bin_q95: list[float] = []
            per_bin_q99: list[float] = []
            for timestep_bin in range(timestep_bins):
                values = [
                    value
                    for image_key in sampled_images
                    for value in by_image[(candidate, timestep_bin, image_key)]
                ]
                per_bin_q95.append(float(np.quantile(values, 0.95)))
                per_bin_q99.append(float(np.quantile(values, 0.99)))
            local[iteration] = max(per_bin_q95)
            q99_draw[iteration] = max(per_bin_q99)
        local_draws[candidate] = local
        ceiling = 100.0 / (1.0 + local)
        ceiling_draws[candidate] = ceiling

        worst = max(
            point_metrics,
            key=lambda row: (float(row["q95"]), -int(row["timestep_bin"])),
        )
        catastrophic = max(
            point_metrics,
            key=lambda row: (float(row["q99"]), -int(row["timestep_bin"])),
        )
        candidate_state = candidate_summary.get(candidate, {})
        hard_pass = not bool(candidate_state.get("forced_safety_abort")) and not candidate_state.get(
            "invalid_reason"
        )
        point_l = float(worst["q95"])
        local_ci = _bootstrap_ci(local)
        ceiling_ci = _bootstrap_ci(ceiling)
        q99_ci = _bootstrap_ci(q99_draw)
        strata = {
            (int(row["noise_replica"]), int(row["quant_repeat"]))
            for row in groups[(candidate, int(worst["timestep_bin"]))]
        }
        reasons: list[str] = []
        if not hard_pass:
            reasons.append("hard_safety_failed")
        if point_l >= ANCHOR_RISK_BOUNDARY:
            reasons.append("worst_timestep_q95_anchor_exceeded")
        elif point_l >= TOLERANT_RISK_BOUNDARY:
            reasons.append("worst_timestep_q95_caution")
        if float(catastrophic["q99"]) >= ANCHOR_RISK_BOUNDARY:
            reasons.append("catastrophic_q99_anchor_exceeded_any_timestep")
        if len(strata) < 4:
            reasons.append("under_sampled_strata")
        row = {
            "dataset_id": dataset_id,
            "candidate": candidate,
            "range_mul": range_by_candidate[candidate],
            "hard_safety_pass": hard_pass,
            "evidence_complete": len(reference_images) >= 16 and len(strata) >= 4,
            "local_classification": _classification(point_l, hard_safety_pass=hard_pass),
            "local_risk_L": point_l,
            "local_risk_ci_low": local_ci["ci_low"],
            "local_risk_ci_high": local_ci["ci_high"],
            "worst_timestep_bin": int(worst["timestep_bin"]),
            "worst_timestep_q90_d": float(worst["q90"]),
            "worst_timestep_q95_d": float(worst["q95"]),
            "worst_timestep_q99_d": float(worst["q99"]),
            "worst_timestep_max_d": float(worst["max"]),
            "worst_timestep_d_gt_1_rate": float(worst["d_gt_1_rate"]),
            "worst_timestep_gradient_cosine_lt_0_rate": float(
                worst["gradient_cosine_lt_0_rate"]
            ),
            "catastrophic_q99_d": float(catastrophic["q99"]),
            "catastrophic_q99_timestep_bin": int(catastrophic["timestep_bin"]),
            "catastrophic_q99_ci_low": q99_ci["ci_low"],
            "catastrophic_q99_ci_high": q99_ci["ci_high"],
            "formal_score_ceiling_from_local": 100.0 / (1.0 + point_l),
            "formal_score_ceiling_ci_low": ceiling_ci["ci_low"],
            "formal_score_ceiling_ci_high": ceiling_ci["ci_high"],
            "trajectory_risk_T": None,
            "combined_risk_R": None,
            "display_score_S": None,
            "formal_score_status": "unknown_until_128_step_branch",
            "catastrophic_tail_included_in_L": False,
            "reason_codes": reasons,
        }
        score_rows.append(row)
        for iteration in range(int(bootstrap_iterations)):
            bootstrap_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "iteration": iteration,
                    "local_risk_L": float(local[iteration]),
                    "formal_score_ceiling_from_local": float(ceiling[iteration]),
                    "catastrophic_q99_d": float(q99_draw[iteration]),
                }
            )

    probabilities = _winner_probabilities(candidates, local_draws)
    for row in score_rows:
        row["bootstrap_point_min_probability"] = probabilities[str(row["candidate"])]
    selection = _select_candidates(score_rows)
    selected_point_probability = probabilities[selection["point_min_candidate"]]
    selection["point_min_bootstrap_probability"] = selected_point_probability
    selection["winner_probability_is_descriptive"] = True

    ranking_rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            ranking_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate_a": left,
                    "range_mul_a": range_by_candidate[left],
                    "candidate_b": right,
                    "range_mul_b": range_by_candidate[right],
                    "probability_L_a_lt_L_b": float(
                        np.mean(local_draws[left] < local_draws[right])
                    ),
                    "probability_L_b_lt_L_a": float(
                        np.mean(local_draws[right] < local_draws[left])
                    ),
                }
            )

    envelope_draw = np.max(
        np.stack([local_draws[candidate] for candidate in candidates]), axis=0
    )
    envelope_ci = _bootstrap_ci(envelope_draw)
    point_envelope = max(float(row["local_risk_L"]) for row in score_rows)
    result_summary = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "dataset_id": dataset_id,
        "diagnostic_target": "local_gradient_acceptance_by_fixed_range_mul",
        "not_quality_or_utility": True,
        "formal_score_status": "unknown_until_128_step_branch",
        "candidate_grid": [range_by_candidate[candidate] for candidate in candidates],
        "candidate_count": len(candidates),
        "image_count": len(reference_images),
        "timestep_bins": timestep_bins,
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "selection": selection,
        "local_tested_grid_envelope": {
            "definition": "max_m L(m); grid-dependent",
            "local_risk_L": point_envelope,
            "local_risk_ci_low": envelope_ci["ci_low"],
            "local_risk_ci_high": envelope_ci["ci_high"],
            "score_ceiling": 100.0 / (1.0 + point_envelope),
        },
        "selection_rule": local_selection_rule(),
    }
    return {
        "summary": result_summary,
        "score_rows": score_rows,
        "timestep_rows": timestep_rows,
        "bootstrap_rows": bootstrap_rows,
        "ranking_rows": ranking_rows,
        "selection": selection,
    }
