from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "2.3.0"
METRIC_DEFINITION_VERSION = "2.3.0"
DEFAULT_BOOTSTRAP_ITERATIONS = 2000
DEFAULT_BOOTSTRAP_SEED = 2301
TOLERANT_RISK_BOUNDARY = 0.5
ANCHOR_RISK_BOUNDARY = 1.0


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            "R=1 has a mathematical anchor; R=0.5 is provisional and must be "
            "calibrated against controlled outcomes"
        ),
        "bootstrap": {
            "iterations": DEFAULT_BOOTSTRAP_ITERATIONS,
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "local": "image-block resampling with all timestep/noise/quant rows retained",
            "trajectory": (
                "candidate-repeat and no_quant-pair resampling; no_quant pairs are "
                "dependent, so trajectory CI is descriptive"
            ),
        },
        "edge_extension": {
            "upper": [3.75, 4.05],
            "lower": [2.4, 2.1],
            "rule": (
                "recommend only when the minimum observed R is at a grid edge, "
                "minimum R >= 0.5, and no hard safety failure invalidates the curve"
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _quantile(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("quantile requires at least one value")
    return float(np.quantile(array, probability))


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("summary requires at least one value")
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def classify_risk(
    risk: float | None,
    *,
    hard_safety_pass: bool,
    evidence_complete: bool,
) -> str:
    if not hard_safety_pass:
        return "unsafe"
    if not evidence_complete or risk is None or not math.isfinite(risk):
        return "unknown"
    if risk >= ANCHOR_RISK_BOUNDARY:
        return "anchor_exceeded_high_perturbation"
    if risk >= TOLERANT_RISK_BOUNDARY:
        return "caution"
    return "observed_tolerant"


def display_score(risk: float) -> float:
    if risk < 0 or not math.isfinite(risk):
        raise ValueError(f"risk must be finite and non-negative: {risk!r}")
    return 100.0 / (1.0 + risk)


def _bootstrap_ci(values: np.ndarray) -> dict[str, float]:
    return {
        "ci_low": float(np.quantile(values, 0.025)),
        "median": float(np.quantile(values, 0.5)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def _contiguous_grid(values: Sequence[float], accepted: set[float]) -> list[list[float]]:
    output: list[list[float]] = []
    current: list[float] = []
    for value in sorted(values):
        if value in accepted:
            current.append(float(value))
        elif current:
            output.append(current)
            current = []
    if current:
        output.append(current)
    return output


def _candidate_name(range_mul: float) -> str:
    return f"mul_{float(range_mul):.3f}"


def _candidate_rows(
    summary: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    return {
        str(row["candidate"]): row
        for row in summary.get("candidates", [])
        if str(row.get("candidate")) != "no_quant"
    }


def _hard_safety(
    candidate: str,
    candidate_summary: Mapping[str, Any],
    range_rows: Sequence[Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _as_bool(candidate_summary.get("forced_safety_abort", False)):
        reasons.append("forced_safety_abort")
    if candidate_summary.get("invalid_reason") not in (None, ""):
        reasons.append(f"candidate_invalid:{candidate_summary.get('invalid_reason')}")
    relevant = [
        row
        for row in range_rows
        if str(row.get("candidate")) == candidate
        and int(float(row.get("checkpoint", -1))) == 128
        and str(row.get("module_group")) == "all"
    ]
    if not relevant:
        reasons.append("missing_checkpoint128_range_rows")
    for row in relevant:
        if _as_bool(row.get("forced_safety_abort", False)):
            reasons.append("range_row_forced_safety_abort")
        if row.get("invalid_reason") not in (None, ""):
            reasons.append(f"range_row_invalid:{row.get('invalid_reason')}")
        if not _as_bool(row.get("common_skip_matched", True)):
            reasons.append("common_skip_not_matched")
        for field in ("orthogonal_drift", "total_drift"):
            if not _finite(row.get(field)):
                reasons.append(f"nonfinite_{field}")
    return not reasons, sorted(set(reasons))


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
        raise ValueError("v2.3 safety analysis requires a schema 2.1.0 tail profile")
    if str(summary.get("profile", {}).get("protocol")) != "v2-tail-calibration":
        raise ValueError("v2.3 safety analysis requires v2-tail-calibration")
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
        value = float(row["relative_gradient_distance"])
        values_by_key[(candidate, timestep_bin, image_key)].append(value)
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

    candidate_drift_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in cumulative_null_rows:
        candidate = str(row.get("candidate"))
        if (
            str(row.get("record_type")) == "candidate_vs_matched_no_quant"
            and candidate in candidate_summaries
            and str(row.get("module_group")) == "all"
            and int(float(row.get("checkpoint", -1))) == 128
            and _finite(row.get("orthogonal_drift"))
        ):
            candidate_drift_rows[candidate].append(row)

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

    score_rows: list[dict[str, Any]] = []
    timestep_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
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
        worst = max(timestep_metrics, key=lambda row: (float(row["q95"]), -int(row["timestep_bin"])))
        local_risk = float(worst["q95"])

        local_bootstrap = np.empty(int(bootstrap_iterations), dtype=np.float64)
        for iteration, sampled_indices in enumerate(image_index):
            worst_q95 = 0.0
            sampled_images = [image_keys[int(index)] for index in sampled_indices]
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
                worst_q95 = max(worst_q95, _quantile(sampled_values, 0.95))
            local_bootstrap[iteration] = worst_q95

        drift_rows = sorted(
            candidate_drift_rows.get(candidate, []),
            key=lambda row: int(float(row.get("repeat", 0))),
        )
        candidate_drifts = np.asarray(
            [float(row["orthogonal_drift"]) for row in drift_rows],
            dtype=np.float64,
        )
        evidence_complete = (
            len(drift_rows) >= 5
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

        worst_rows = row_groups[(candidate, int(worst["timestep_bin"]))]
        stratum_values: dict[tuple[int, int], list[float]] = defaultdict(list)
        for row in worst_rows:
            stratum_values[
                (int(row["noise_replica"]), int(row["quant_repeat"]))
            ].append(float(row["relative_gradient_distance"]))
        stratum_q95 = [_quantile(values, 0.95) for values in stratum_values.values()]
        stratum_cv = (
            statistics.pstdev(stratum_q95) / max(abs(statistics.mean(stratum_q95)), 1e-30)
            if len(stratum_q95) > 1
            else None
        )

        reasons = list(hard_reasons)
        if local_risk >= 1.0:
            reasons.append("worst_timestep_q95_anchor_exceeded")
        elif local_risk >= 0.5:
            reasons.append("worst_timestep_q95_caution")
        if math.isfinite(trajectory_risk):
            if trajectory_risk >= 1.0:
                reasons.append("trajectory_natural_q95_anchor_exceeded")
            elif trajectory_risk >= 0.5:
                reasons.append("trajectory_caution")
        if float(worst["q99"]) >= 1.0:
            reasons.append("worst_timestep_q99_anchor_exceeded")
        if float(worst["d_gt_1_rate"]) > 0:
            reasons.append("observed_d_gt_1_tail")
        if float(worst["gradient_cosine_lt_0_rate"]) > 0:
            reasons.append("observed_gradient_direction_reversal")
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
            "stratum_count": len(stratum_q95),
            "worst_timestep_stratum_q95_cv": stratum_cv,
            "branch_repeat_count": len(candidate_drifts),
            "image_count": len(image_keys),
            "reason_codes": sorted(set(reasons)),
            "score_is_quality_or_utility": False,
            "acceptance_threshold_validated": False,
        }
        score_rows.append(row)
        for iteration in range(int(bootstrap_iterations)):
            bootstrap_rows.append(
                {
                    "dataset_id": dataset_id,
                    "candidate": candidate,
                    "range_mul": range_by_candidate[candidate],
                    "iteration": iteration,
                    "local_risk_L": float(local_bootstrap[iteration]),
                    "trajectory_risk_T": float(trajectory_bootstrap[iteration]),
                    "combined_risk_R": float(combined_bootstrap[iteration]),
                    "display_score_S": float(score_bootstrap[iteration]),
                }
            )

    valid_rows = [
        row
        for row in score_rows
        if row["hard_safety_pass"]
        and row["combined_risk_R"] is not None
        and math.isfinite(float(row["combined_risk_R"]))
    ]
    if not valid_rows:
        best_row = None
    else:
        best_row = min(
            valid_rows,
            key=lambda row: (float(row["combined_risk_R"]), float(row["range_mul"])),
        )
    tolerant_values = {
        float(row["range_mul"])
        for row in valid_rows
        if float(row["combined_risk_R"]) < 0.5
    }
    below_anchor_values = {
        float(row["range_mul"])
        for row in valid_rows
        if float(row["combined_risk_R"]) < 1.0
    }
    hard_failure_values = [
        float(row["range_mul"]) for row in score_rows if not row["hard_safety_pass"]
    ]
    edge_direction: str | None = None
    edge_values: list[float] = []
    if (
        best_row is not None
        and float(best_row["combined_risk_R"]) >= 0.5
        and not hard_failure_values
    ):
        best_mul = float(best_row["range_mul"])
        if math.isclose(best_mul, max(grid)):
            edge_direction = "upper"
            edge_values = [3.75, 4.05]
        elif math.isclose(best_mul, min(grid)):
            edge_direction = "lower"
            edge_values = [2.4, 2.1]

    risk_values = [float(row["combined_risk_R"]) for row in valid_rows]
    if hard_failure_values:
        phenotype = "hard_safety_failure_observed"
    elif risk_values and all(value < 0.5 for value in risk_values):
        phenotype = "broadly_observed_tolerant"
    elif risk_values and all(value >= 1.0 for value in risk_values):
        phenotype = "high_perturbation_all_tested"
    elif risk_values and min(risk_values) < 0.5 and max(risk_values) >= 1.0:
        phenotype = "mixed_or_narrow_tolerance"
    elif risk_values and min(risk_values) < 0.5:
        phenotype = "partially_observed_tolerant"
    elif risk_values and min(risk_values) < 1.0:
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
        "best_tested_mul_for_numerical_safety": (
            float(best_row["range_mul"]) if best_row is not None else None
        ),
        "best_tested_score": (
            float(best_row["display_score_S"]) if best_row is not None else None
        ),
        "tested_tolerant_envelope": _contiguous_grid(grid, tolerant_values),
        "tested_below_anchor_envelope": _contiguous_grid(
            grid,
            below_anchor_values,
        ),
        "hard_safety_failure_muls": hard_failure_values,
        "edge_unresolved": edge_direction is not None,
        "edge_extension_direction": edge_direction,
        "edge_extension_recommended_muls": edge_values,
        "all_tested_high_perturbation": bool(
            risk_values and all(value >= 1.0 for value in risk_values)
        ),
        "risk_min": min(risk_values) if risk_values else None,
        "risk_max": max(risk_values) if risk_values else None,
        "risk_span": (
            max(risk_values) - min(risk_values) if risk_values else None
        ),
        "no_quant_natural_orthogonal_drift_q95": natural_q95,
        "no_quant_natural_pair_count": len(natural_rows),
        "image_count": len(image_keys),
        "bootstrap_iterations": int(bootstrap_iterations),
        "bootstrap_seed": int(bootstrap_seed),
        "score_rows": score_rows,
        "interpretation": {
            "safety_not_utility": True,
            "quality_unknown": True,
            "threshold_0_5_provisional": True,
            "anchor_1_has_mathematical_meaning": True,
        },
    }
    return {
        "contract": contract,
        "summary": summary_payload,
        "score_rows": score_rows,
        "timestep_rows": timestep_rows,
        "bootstrap_rows": bootstrap_rows,
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
        return json.loads(
            (profile_dir / name).read_text(encoding="utf-8-sig")
        )

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
