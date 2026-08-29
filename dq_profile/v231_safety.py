from __future__ import annotations

"""DQ Profiler v2.3.1 numerical-safety analysis.

This module intentionally leaves :mod:`dq_profile.v23_safety` frozen.  The
primary L/T/R/S definitions are unchanged; v2.3.1 adds paired ranking
uncertainty and an explicitly non-scoring catastrophic-tail channel.
"""

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dq_profile.v23_safety import (
    ANCHOR_RISK_BOUNDARY,
    TOLERANT_RISK_BOUNDARY,
    _as_bool,
    _bootstrap_ci,
    _candidate_rows,
    _contiguous_grid,
    _finite,
    _hard_safety,
    _quantile,
    _summary,
    canonical_json_sha256,
    classify_risk,
    display_score,
)


SCHEMA_VERSION = "2.3.1"
METRIC_DEFINITION_VERSION = "2.3.1"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 2311
RANKING_CONFIDENCE_BOUNDARY = 0.75


def safety_contract() -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "diagnostic_target": "numerical_gradient_acceptance_by_fixed_range_mul",
        "not_a_target": [
            "final_image_quality",
            "quantization_utility",
            "best_production_setting",
        ],
        "primary_metrics_unchanged_from": "2.3.0",
        "local_risk": {
            "symbol": "L",
            "definition": "max_t q95(||g_quant-g_no_quant||/||g_no_quant|| | mul,t)",
            "source": "gradient_tail.csv record_type=sample",
            "block": "image",
        },
        "trajectory_risk": {
            "symbol": "T",
            "definition": (
                "median(candidate checkpoint128 orthogonal drift) / "
                "q95(no_quant-pair checkpoint128 orthogonal drift)"
            ),
            "source": "cumulative_null_calibration.csv",
        },
        "combined_risk": {
            "symbol": "R",
            "definition": "max(L,T)",
            "reason": "weakest-link safety aggregation",
        },
        "display_score": {
            "symbol": "S",
            "definition": "100/(1+R)",
            "validated_acceptance_threshold": False,
            "quality_score": False,
        },
        "candidate_ranking": {
            "definition": (
                "probability that each tested mul has minimum R under shared "
                "image-block, branch-repeat, and no_quant-null bootstrap draws"
            ),
            "resolved_boundary": RANKING_CONFIDENCE_BOUNDARY,
            "boundary_validated": False,
            "point_estimate_is_recommendation": False,
        },
        "catastrophic_tail": {
            "symbol": "C99",
            "definition": "max_t q99(||g_quant-g_no_quant||/||g_no_quant|| | mul,t)",
            "additional_channels": [
                "max_t max(d)",
                "max_t rate(d>1)",
                "max_t rate(gradient cosine<0)",
            ],
            "included_in_combined_risk_R": False,
            "role": "predeclared explanatory alarm, not a post-hoc score vote",
        },
        "hard_safety_precedence": [
            "forced_safety_abort",
            "invalid_execution",
            "nonfinite_or_gradient_explosion",
        ],
        "descriptive_labels": {
            "unsafe": "hard safety failure",
            "anchor_exceeded_high_perturbation": "R >= 1",
            "caution": "0.5 <= R < 1",
            "observed_tolerant": "R < 0.5",
            "unknown": "insufficient or invalid evidence",
        },
        "boundary_note": (
            "R=1 has a mathematical anchor; R=0.5 and ranking probability 0.75 "
            "are provisional and require outcome calibration"
        ),
        "bootstrap": {
            "iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "local": "shared image-block resampling across candidates",
            "trajectory": (
                "shared candidate-repeat and no_quant-pair resampling across "
                "candidates; no_quant pairs are dependent, so CI is descriptive"
            ),
        },
        "edge_extension": {
            "upper": [3.75, 4.05],
            "lower": [2.4, 2.1],
            "rule": (
                "recommend as a curve-completion experiment when the point minimum "
                "R is at a grid edge, minimum R >= 0.5, and hard safety passes; "
                "report ranking uncertainty separately"
            ),
        },
        "required_regimes": {
            "local_probe": "structural_dropout_off",
            "short_branch": "training_dropout_on",
            "guardian": "common_skip",
        },
        "runtime_input_contract": "v2-tail-calibration schema 2.1 frozen",
        "utility": "unknown",
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def _repeat_maps(
    candidates: Sequence[str],
    rows_by_candidate: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, dict[int, float]], list[int], bool]:
    maps: dict[str, dict[int, float]] = {}
    repeat_sets: list[set[int]] = []
    for candidate in candidates:
        values: dict[int, float] = {}
        for row in rows_by_candidate.get(candidate, ()):
            repeat = int(float(row.get("repeat", 0)))
            if repeat in values:
                raise ValueError(f"duplicate checkpoint128 repeat for {candidate}: {repeat}")
            values[repeat] = float(row["orthogonal_drift"])
        maps[candidate] = values
        repeat_sets.append(set(values))
    if not repeat_sets:
        return maps, [], False
    first = repeat_sets[0]
    complete = bool(first) and all(current == first for current in repeat_sets[1:])
    return maps, sorted(first) if complete else [], complete


def _winner_probabilities(
    candidates: Sequence[str],
    risk_bootstrap: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not candidates:
        return {}, []
    matrix = np.stack([risk_bootstrap[candidate] for candidate in candidates])
    probabilities = {candidate: 0.0 for candidate in candidates}
    for iteration in range(matrix.shape[1]):
        values = matrix[:, iteration]
        minimum = float(np.min(values))
        winners = np.flatnonzero(np.isclose(values, minimum, rtol=0.0, atol=1e-12))
        share = 1.0 / len(winners)
        for index in winners:
            probabilities[candidates[int(index)]] += share
    probabilities = {
        candidate: value / matrix.shape[1]
        for candidate, value in probabilities.items()
    }
    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            left_values = risk_bootstrap[left]
            right_values = risk_bootstrap[right]
            ties = np.isclose(left_values, right_values, rtol=0.0, atol=1e-12)
            pairwise.append(
                {
                    "candidate_a": left,
                    "candidate_b": right,
                    "probability_a_lower_risk": float(np.mean(left_values < right_values)),
                    "probability_tie": float(np.mean(ties)),
                    "probability_b_lower_risk": float(np.mean(right_values < left_values)),
                }
            )
    return probabilities, pairwise


def analyze_profile(
    *,
    summary: Mapping[str, Any],
    gradient_tail_rows: Sequence[Mapping[str, Any]],
    cumulative_null_rows: Sequence[Mapping[str, Any]],
    range_sweep_rows: Sequence[Mapping[str, Any]],
    dataset_id: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if str(summary.get("schema_version")) != "2.1.0":
        raise ValueError("v2.3.1 safety analysis requires a schema 2.1.0 tail profile")
    if str(summary.get("profile", {}).get("protocol")) != "v2-tail-calibration":
        raise ValueError("v2.3.1 safety analysis requires v2-tail-calibration")
    if bootstrap_iterations <= 0:
        raise ValueError("bootstrap_iterations must be positive")

    contract = safety_contract()
    candidate_summaries = _candidate_rows(summary)
    candidates = sorted(
        candidate_summaries,
        key=lambda name: float(candidate_summaries[name]["initial_range_mul"]),
    )
    if not candidates:
        raise ValueError("tail profile has no fixed-mul candidates")
    range_by_candidate = {
        candidate: float(candidate_summaries[candidate]["initial_range_mul"])
        for candidate in candidates
    }
    grid = [range_by_candidate[candidate] for candidate in candidates]
    timestep_bins = int(summary["profile"]["timestep_bins"])

    sample_rows = [
        row
        for row in gradient_tail_rows
        if str(row.get("record_type")) == "sample"
        and str(row.get("candidate")) in candidate_summaries
    ]
    if not sample_rows:
        raise ValueError("gradient_tail.csv has no candidate sample rows")
    image_keys = sorted({str(row["image_key"]) for row in sample_rows})
    if not image_keys:
        raise ValueError("gradient tail has no image keys")

    values_by_key: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    row_groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sample_rows:
        candidate = str(row["candidate"])
        timestep_bin = int(row["timestep_bin"])
        image_key = str(row["image_key"])
        values_by_key[(candidate, timestep_bin, image_key)].append(
            float(row["relative_gradient_distance"])
        )
        row_groups[(candidate, timestep_bin)].append(row)

    natural_rows = [
        row
        for row in cumulative_null_rows
        if str(row.get("record_type")) == "no_quant_pair"
        and str(row.get("module_group")) == "all"
        and int(float(row.get("checkpoint", -1))) == 128
        and _finite(row.get("orthogonal_drift"))
    ]
    natural_drifts = np.asarray(
        [float(row["orthogonal_drift"]) for row in natural_rows],
        dtype=np.float64,
    )
    if natural_drifts.size == 0:
        raise ValueError("no checkpoint128 no_quant natural drift rows")
    natural_q95 = float(np.quantile(natural_drifts, 0.95))
    if natural_q95 <= 0:
        raise ValueError("no_quant natural drift q95 must be positive")

    drift_rows_by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cumulative_null_rows:
        candidate = str(row.get("candidate"))
        if (
            str(row.get("record_type")) == "candidate_vs_matched_no_quant"
            and candidate in candidate_summaries
            and str(row.get("module_group")) == "all"
            and int(float(row.get("checkpoint", -1))) == 128
            and _finite(row.get("orthogonal_drift"))
        ):
            drift_rows_by_candidate[candidate].append(row)
    drift_maps, paired_repeat_ids, repeat_pairing_complete = _repeat_maps(
        candidates,
        drift_rows_by_candidate,
    )

    rng = np.random.default_rng(int(bootstrap_seed))
    image_index = rng.integers(
        0,
        len(image_keys),
        size=(int(bootstrap_iterations), len(image_keys)),
    )
    natural_index = rng.integers(
        0,
        len(natural_drifts),
        size=(int(bootstrap_iterations), len(natural_drifts)),
    )
    null_q95_bootstrap = np.quantile(natural_drifts[natural_index], 0.95, axis=1)
    paired_repeat_index = (
        rng.integers(
            0,
            len(paired_repeat_ids),
            size=(int(bootstrap_iterations), len(paired_repeat_ids)),
        )
        if repeat_pairing_complete
        else None
    )

    score_rows: list[dict[str, Any]] = []
    timestep_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    risk_bootstrap: dict[str, np.ndarray] = {}
    bootstrap_payloads: dict[str, dict[str, np.ndarray]] = {}
    hard_pass_by_candidate: dict[str, bool] = {}

    for candidate in candidates:
        timestep_metrics: list[dict[str, Any]] = []
        for timestep_bin in range(timestep_bins):
            rows = row_groups.get((candidate, timestep_bin), [])
            values = [float(row["relative_gradient_distance"]) for row in rows]
            if not values:
                continue
            metrics = _summary(values)
            metrics.update(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "timestep_bin": timestep_bin,
                    "sample_count": len(values),
                    "image_count": len({str(row["image_key"]) for row in rows}),
                    "d_gt_1_rate": statistics.mean(
                        float(row["relative_gradient_distance"]) > 1.0 for row in rows
                    ),
                    "gradient_cosine_lt_0_rate": statistics.mean(
                        float(row["gradient_cosine"]) < 0.0 for row in rows
                    ),
                }
            )
            timestep_metrics.append(metrics)
            timestep_rows.append(dict(metrics))
        if len(timestep_metrics) != timestep_bins:
            raise ValueError(
                f"{candidate} has {len(timestep_metrics)} timestep bins, expected {timestep_bins}"
            )
        worst = max(
            timestep_metrics,
            key=lambda row: (float(row["q95"]), -int(row["timestep_bin"])),
        )
        catastrophic_q99 = max(
            timestep_metrics,
            key=lambda row: (float(row["q99"]), -int(row["timestep_bin"])),
        )
        catastrophic_max = max(
            timestep_metrics,
            key=lambda row: (float(row["max"]), -int(row["timestep_bin"])),
        )
        catastrophic_d_gt_1 = max(
            timestep_metrics,
            key=lambda row: (
                float(row["d_gt_1_rate"]),
                -int(row["timestep_bin"]),
            ),
        )
        catastrophic_cosine = max(
            timestep_metrics,
            key=lambda row: (
                float(row["gradient_cosine_lt_0_rate"]),
                -int(row["timestep_bin"]),
            ),
        )
        local_risk = float(worst["q95"])

        local_bootstrap = np.empty(int(bootstrap_iterations), dtype=np.float64)
        catastrophic_q99_bootstrap = np.empty(
            int(bootstrap_iterations),
            dtype=np.float64,
        )
        for iteration, sampled_indices in enumerate(image_index):
            sampled_images = [image_keys[int(index)] for index in sampled_indices]
            q95_values: list[float] = []
            q99_values: list[float] = []
            for timestep_bin in range(timestep_bins):
                sampled_values = [
                    value
                    for image_key in sampled_images
                    for value in values_by_key.get((candidate, timestep_bin, image_key), ())
                ]
                if not sampled_values:
                    raise ValueError(
                        f"bootstrap missing values for {candidate}, timestep {timestep_bin}"
                    )
                q95_values.append(_quantile(sampled_values, 0.95))
                q99_values.append(_quantile(sampled_values, 0.99))
            local_bootstrap[iteration] = max(q95_values)
            catastrophic_q99_bootstrap[iteration] = max(q99_values)

        repeat_values = drift_maps.get(candidate, {})
        candidate_repeat_ids = sorted(repeat_values)
        candidate_drifts = np.asarray(
            [repeat_values[repeat] for repeat in candidate_repeat_ids],
            dtype=np.float64,
        )
        evidence_complete = (
            len(candidate_drifts) >= 5
            and len(natural_rows) >= 5
            and len(image_keys) >= 16
        )
        if candidate_drifts.size == 0:
            trajectory_risk = math.nan
            trajectory_bootstrap = np.full(
                int(bootstrap_iterations),
                math.nan,
                dtype=np.float64,
            )
        else:
            trajectory_risk = float(np.median(candidate_drifts) / natural_q95)
            if repeat_pairing_complete:
                aligned = np.asarray(
                    [repeat_values[repeat] for repeat in paired_repeat_ids],
                    dtype=np.float64,
                )
                assert paired_repeat_index is not None
                candidate_median_bootstrap = np.median(
                    aligned[paired_repeat_index],
                    axis=1,
                )
            else:
                candidate_index = rng.integers(
                    0,
                    len(candidate_drifts),
                    size=(int(bootstrap_iterations), len(candidate_drifts)),
                )
                candidate_median_bootstrap = np.median(
                    candidate_drifts[candidate_index],
                    axis=1,
                )
            trajectory_bootstrap = candidate_median_bootstrap / np.maximum(
                null_q95_bootstrap,
                1e-30,
            )

        hard_pass, hard_reasons = _hard_safety(
            candidate,
            candidate_summaries[candidate],
            range_sweep_rows,
        )
        hard_pass_by_candidate[candidate] = hard_pass
        combined_risk = (
            max(local_risk, trajectory_risk)
            if math.isfinite(trajectory_risk)
            else None
        )
        label = classify_risk(
            combined_risk,
            hard_safety_pass=hard_pass,
            evidence_complete=evidence_complete,
        )
        score = (
            display_score(combined_risk)
            if combined_risk is not None and hard_pass
            else None
        )

        combined_bootstrap = np.maximum(local_bootstrap, trajectory_bootstrap)
        score_bootstrap = 100.0 / (1.0 + combined_bootstrap)
        valid_bootstrap = np.isfinite(combined_bootstrap)
        if not np.any(valid_bootstrap):
            raise ValueError(f"no finite bootstrap values for {candidate}")
        local_ci = _bootstrap_ci(local_bootstrap)
        trajectory_ci = _bootstrap_ci(trajectory_bootstrap[valid_bootstrap])
        risk_ci = _bootstrap_ci(combined_bootstrap[valid_bootstrap])
        score_ci = _bootstrap_ci(score_bootstrap[valid_bootstrap])
        catastrophic_q99_ci = _bootstrap_ci(catastrophic_q99_bootstrap)

        worst_rows = row_groups[(candidate, int(worst["timestep_bin"]))]
        stratum_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in worst_rows:
            stratum_values[
                (int(row["noise_replica"]), int(row["quant_repeat"]))
            ].append(float(row["relative_gradient_distance"]))
        stratum_q95 = [_quantile(values, 0.95) for values in stratum_values.values()]
        stratum_cv = (
            statistics.pstdev(stratum_q95)
            / max(abs(statistics.mean(stratum_q95)), 1e-30)
            if len(stratum_q95) > 1
            else None
        )

        reasons = list(hard_reasons)
        if local_risk >= ANCHOR_RISK_BOUNDARY:
            reasons.append("worst_timestep_q95_anchor_exceeded")
        elif local_risk >= TOLERANT_RISK_BOUNDARY:
            reasons.append("worst_timestep_q95_caution")
        if math.isfinite(trajectory_risk):
            if trajectory_risk >= ANCHOR_RISK_BOUNDARY:
                reasons.append("trajectory_natural_q95_anchor_exceeded")
            elif trajectory_risk >= TOLERANT_RISK_BOUNDARY:
                reasons.append("trajectory_caution")
        if float(catastrophic_q99["q99"]) >= ANCHOR_RISK_BOUNDARY:
            reasons.append("catastrophic_q99_anchor_exceeded_any_timestep")
        if float(catastrophic_d_gt_1["d_gt_1_rate"]) > 0:
            reasons.append("observed_d_gt_1_tail_any_timestep")
        if float(catastrophic_cosine["gradient_cosine_lt_0_rate"]) > 0:
            reasons.append("observed_gradient_direction_reversal_any_timestep")
        if len(stratum_q95) < 4:
            reasons.append("under_sampled_strata")

        row = {
            "dataset_id": dataset_id,
            "candidate": candidate,
            "range_mul": range_by_candidate[candidate],
            "hard_safety_pass": hard_pass,
            "evidence_complete": evidence_complete,
            "classification": label,
            "local_risk_L": local_risk,
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
            "catastrophic_q99_d": float(catastrophic_q99["q99"]),
            "catastrophic_q99_timestep_bin": int(catastrophic_q99["timestep_bin"]),
            "catastrophic_q99_ci_low": catastrophic_q99_ci["ci_low"],
            "catastrophic_q99_ci_high": catastrophic_q99_ci["ci_high"],
            "catastrophic_max_d": float(catastrophic_max["max"]),
            "catastrophic_max_timestep_bin": int(catastrophic_max["timestep_bin"]),
            "catastrophic_d_gt_1_rate": float(catastrophic_d_gt_1["d_gt_1_rate"]),
            "catastrophic_d_gt_1_timestep_bin": int(
                catastrophic_d_gt_1["timestep_bin"]
            ),
            "catastrophic_gradient_cosine_lt_0_rate": float(
                catastrophic_cosine["gradient_cosine_lt_0_rate"]
            ),
            "catastrophic_gradient_cosine_timestep_bin": int(
                catastrophic_cosine["timestep_bin"]
            ),
            "catastrophic_tail_included_in_R": False,
            "trajectory_risk_T": trajectory_risk,
            "trajectory_risk_ci_low": trajectory_ci["ci_low"],
            "trajectory_risk_ci_high": trajectory_ci["ci_high"],
            "candidate_orthogonal_drift_median": float(np.median(candidate_drifts)),
            "no_quant_natural_orthogonal_drift_q95": natural_q95,
            "combined_risk_R": combined_risk,
            "combined_risk_ci_low": risk_ci["ci_low"],
            "combined_risk_ci_high": risk_ci["ci_high"],
            "display_score_S": score,
            "display_score_ci_low": score_ci["ci_low"],
            "display_score_ci_high": score_ci["ci_high"],
            "bootstrap_best_probability": None,
            "stratum_count": len(stratum_q95),
            "worst_timestep_stratum_q95_cv": stratum_cv,
            "branch_repeat_count": len(candidate_drifts),
            "image_count": len(image_keys),
            "reason_codes": sorted(set(reasons)),
            "score_is_quality_or_utility": False,
            "acceptance_threshold_validated": False,
        }
        score_rows.append(row)
        risk_bootstrap[candidate] = combined_bootstrap
        bootstrap_payloads[candidate] = {
            "local": local_bootstrap,
            "trajectory": trajectory_bootstrap,
            "combined": combined_bootstrap,
            "score": score_bootstrap,
            "catastrophic_q99": catastrophic_q99_bootstrap,
        }

    ranking_candidates = [
        candidate
        for candidate in candidates
        if hard_pass_by_candidate.get(candidate, False)
        and np.all(np.isfinite(risk_bootstrap[candidate]))
    ]
    ranking_probabilities, ranking_rows = _winner_probabilities(
        ranking_candidates,
        risk_bootstrap,
    )
    for row in score_rows:
        row["bootstrap_best_probability"] = ranking_probabilities.get(
            str(row["candidate"])
        )
    for candidate in candidates:
        payload = bootstrap_payloads[candidate]
        for iteration in range(int(bootstrap_iterations)):
            bootstrap_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "iteration": iteration,
                    "local_risk_L": float(payload["local"][iteration]),
                    "trajectory_risk_T": float(payload["trajectory"][iteration]),
                    "combined_risk_R": float(payload["combined"][iteration]),
                    "display_score_S": float(payload["score"][iteration]),
                    "catastrophic_q99_d": float(
                        payload["catastrophic_q99"][iteration]
                    ),
                }
            )

    valid_rows = [
        row
        for row in score_rows
        if row["hard_safety_pass"]
        and row["combined_risk_R"] is not None
        and math.isfinite(float(row["combined_risk_R"]))
    ]
    point_best = (
        min(
            valid_rows,
            key=lambda row: (float(row["combined_risk_R"]), float(row["range_mul"])),
        )
        if valid_rows
        else None
    )
    modal_candidate = (
        max(
            ranking_candidates,
            key=lambda candidate: (
                ranking_probabilities[candidate],
                -range_by_candidate[candidate],
            ),
        )
        if ranking_candidates
        else None
    )
    modal_probability = (
        ranking_probabilities[modal_candidate] if modal_candidate is not None else None
    )
    ranking_resolved = bool(
        repeat_pairing_complete
        and len(ranking_candidates) >= 2
        and modal_probability is not None
        and modal_probability >= RANKING_CONFIDENCE_BOUNDARY
    )
    ranking_status = (
        "resolved"
        if ranking_resolved
        else "indistinguishable"
        if len(ranking_candidates) >= 2
        else "insufficient_competitors"
    )
    if not ranking_resolved:
        for row in score_rows:
            row["reason_codes"] = sorted(
                set([*row["reason_codes"], "candidate_ranking_unresolved"])
            )

    tolerant_values = {
        float(row["range_mul"])
        for row in valid_rows
        if float(row["combined_risk_R"]) < TOLERANT_RISK_BOUNDARY
    }
    below_anchor_values = {
        float(row["range_mul"])
        for row in valid_rows
        if float(row["combined_risk_R"]) < ANCHOR_RISK_BOUNDARY
    }
    hard_failure_values = [
        float(row["range_mul"]) for row in score_rows if not row["hard_safety_pass"]
    ]
    edge_direction: str | None = None
    edge_values: list[float] = []
    if (
        point_best is not None
        and float(point_best["combined_risk_R"]) >= TOLERANT_RISK_BOUNDARY
        and not hard_failure_values
    ):
        best_mul = float(point_best["range_mul"])
        if math.isclose(best_mul, max(grid)):
            edge_direction = "upper"
            edge_values = [3.75, 4.05]
        elif math.isclose(best_mul, min(grid)):
            edge_direction = "lower"
            edge_values = [2.4, 2.1]

    risk_values = [float(row["combined_risk_R"]) for row in valid_rows]
    if hard_failure_values:
        phenotype = "hard_safety_failure_observed"
    elif risk_values and all(value < TOLERANT_RISK_BOUNDARY for value in risk_values):
        phenotype = "broadly_observed_tolerant"
    elif risk_values and all(value >= ANCHOR_RISK_BOUNDARY for value in risk_values):
        phenotype = "high_perturbation_all_tested"
    elif (
        risk_values
        and min(risk_values) < TOLERANT_RISK_BOUNDARY
        and max(risk_values) >= ANCHOR_RISK_BOUNDARY
    ):
        phenotype = "mixed_or_narrow_tolerance"
    elif risk_values and min(risk_values) < TOLERANT_RISK_BOUNDARY:
        phenotype = "partially_observed_tolerant"
    elif risk_values and min(risk_values) < ANCHOR_RISK_BOUNDARY:
        phenotype = "caution_window"
    else:
        phenotype = "unknown"

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "dataset_id": dataset_id,
        "diagnostic_target": contract["diagnostic_target"],
        "safety_contract_sha256": contract["contract_sha256"],
        "source_profile_schema_version": summary["schema_version"],
        "source_profile_metric_definition_version": summary[
            "metric_definition_version"
        ],
        "grid": grid,
        "phenotype": phenotype,
        "point_estimate_best_mul": (
            float(point_best["range_mul"]) if point_best is not None else None
        ),
        "point_estimate_best_score": (
            float(point_best["display_score_S"]) if point_best is not None else None
        ),
        "best_tested_mul_for_numerical_safety": (
            float(point_best["range_mul"]) if point_best is not None else None
        ),
        "best_tested_mul_interpretation": "point_estimate_only",
        "bootstrap_modal_best_mul": (
            range_by_candidate[modal_candidate] if modal_candidate is not None else None
        ),
        "bootstrap_modal_best_probability": modal_probability,
        "ranking_confidence_boundary": RANKING_CONFIDENCE_BOUNDARY,
        "ranking_confidence_boundary_validated": False,
        "ranking_resolved": ranking_resolved,
        "ranking_status": ranking_status,
        "numerical_safety_preferred_mul": (
            range_by_candidate[modal_candidate] if ranking_resolved else None
        ),
        "bootstrap_best_probability_by_candidate": {
            candidate: ranking_probabilities.get(candidate)
            for candidate in candidates
        },
        "repeat_pairing_complete": repeat_pairing_complete,
        "paired_repeat_ids": paired_repeat_ids,
        "tested_tolerant_envelope": _contiguous_grid(grid, tolerant_values),
        "tested_below_anchor_envelope": _contiguous_grid(
            grid,
            below_anchor_values,
        ),
        "hard_safety_failure_muls": hard_failure_values,
        "edge_unresolved": edge_direction is not None,
        "edge_extension_direction": edge_direction,
        "edge_extension_recommended_muls": edge_values,
        "edge_extension_basis": (
            "point_estimate_minimum_at_edge" if edge_direction else None
        ),
        "edge_extension_ranking_resolved": (
            ranking_resolved if edge_direction else None
        ),
        "all_tested_high_perturbation": bool(
            risk_values and all(value >= ANCHOR_RISK_BOUNDARY for value in risk_values)
        ),
        "risk_min": min(risk_values) if risk_values else None,
        "risk_max": max(risk_values) if risk_values else None,
        "risk_span": max(risk_values) - min(risk_values) if risk_values else None,
        "no_quant_natural_orthogonal_drift_q95": natural_q95,
        "no_quant_natural_pair_count": len(natural_rows),
        "image_count": len(image_keys),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "score_rows": score_rows,
        "ranking_rows": ranking_rows,
        "interpretation": {
            "safety_not_utility": True,
            "quality_unknown": True,
            "threshold_0_5_provisional": True,
            "ranking_0_75_provisional": True,
            "anchor_1_has_mathematical_meaning": True,
            "catastrophic_tail_not_in_primary_R": True,
        },
    }
    return {
        "contract": contract,
        "summary": summary_payload,
        "score_rows": score_rows,
        "timestep_rows": timestep_rows,
        "bootstrap_rows": bootstrap_rows,
        "ranking_rows": ranking_rows,
    }


def analyze_profile_directory(
    profile_dir: Path,
    *,
    dataset_id: str,
    bootstrap_iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    profile_dir = Path(profile_dir).resolve()

    def read_json(name: str) -> dict[str, Any]:
        return json.loads((profile_dir / name).read_text(encoding="utf-8-sig"))

    def read_csv(name: str) -> list[dict[str, str]]:
        import csv

        with (profile_dir / name).open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as stream:
            return list(csv.DictReader(stream))

    result = analyze_profile(
        summary=read_json("summary.json"),
        gradient_tail_rows=read_csv("gradient_tail.csv"),
        cumulative_null_rows=read_csv("cumulative_null_calibration.csv"),
        range_sweep_rows=read_csv("range_sweep.csv"),
        dataset_id=dataset_id,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    result["summary"]["source_profile"] = str(profile_dir)
    result["summary"]["source_summary_sha256"] = hashlib.sha256(
        (profile_dir / "summary.json").read_bytes()
    ).hexdigest()
    return result
