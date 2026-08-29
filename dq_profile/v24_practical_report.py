from __future__ import annotations

"""Practical, descriptive report helpers for SDXL DQ Profiler v2.4.

The report is a numerical Safety/Fidelity chart. It deliberately does not
predict final image quality, recommend quantization over no_quant, or name a
best-quality mul.
"""

import ast
from collections import Counter, defaultdict
import html
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "2.4.2-practical-report-prototype"
METRIC_DEFINITION_VERSION = "2.4.0"
REPORT_CONTRACT_VERSION = "1.2.0-prototype"
ABSOLUTE_REFERENCE_DISTANCE = 1.0
AFFINITY_FIXED_Y_MAX = 4.0
AFFINITY_SCALE_CALIBRATION = {
    "mode": "fixed_primary_with_dataset_auto_zoom",
    "y_min": 0.0,
    "y_max": AFFINITY_FIXED_Y_MAX,
    "calibration_dataset_configs": 10,
    "calibration_point_estimates": 80,
    "observed_point_max": 3.79473914493573,
    "observed_point_q95": 1.95326672423944,
    "observed_point_q99": 2.16682267166094,
    "overflow_policy": (
        "keep the fixed primary scale, clip at the upper edge with an overflow "
        "indicator, and preserve exact values in the candidate table"
    ),
}
PRACTICAL_PRESET_MARKERS = (
    ("clip_rate_high 初期値", 2.878, "#0f766e"),
    ("clip_rate_low 初期値", 3.205, "#be185d"),
)
TIMESTEP_BIN_LABELS = {
    0: ("0–249", "低noise"),
    1: ("250–499", "中低noise"),
    2: ("500–749", "中高noise"),
    3: ("750–999", "高noise"),
}
ABSOLUTE_LEVEL_LABELS = {
    "low_perturbation": "低摂動",
    "tail_attention": "Tail注意",
    "high_perturbation": "高摂動",
    "hard_unsafe": "Hard unsafe",
    "unmeasurable": "測定不能",
}
DATASET_ABSOLUTE_RESPONSE_LABELS = {
    "all_low_perturbation": "全候補が低摂動",
    "all_tail_attention": "全候補でTail注意",
    "all_high_perturbation": "全候補が高摂動",
    "mixed_absolute_response": "mulにより摂動帯が変化",
    "includes_hard_unsafe": "Hard unsafeを含む",
}


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass", "passed"}


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return [value]
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]
    return [value]


def _candidate_mul(row: Mapping[str, Any]) -> float:
    value = _optional_float(row.get("range_mul"))
    if value is None:
        value = _optional_float(row.get("initial_range_mul"))
    if value is None:
        raise ValueError(f"candidate row has no finite range_mul: {row!r}")
    return value


def _candidate_name(row: Mapping[str, Any]) -> str:
    value = str(row.get("candidate", "")).strip()
    if value:
        return value
    return f"mul_{_candidate_mul(row):.3f}"


def _same_mul(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def fidelity_gauge(distance: Any) -> float | None:
    """Map a non-negative deformation to an explanatory 0-100 gauge."""

    number = _optional_float(distance)
    if number is None or number < 0.0:
        return None
    return 100.0 / (1.0 + number)


def local_fidelity_gauge(body: Any, tail: Any) -> float | None:
    values = [fidelity_gauge(body), fidelity_gauge(tail)]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def report_contract() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "report_contract_version": REPORT_CONTRACT_VERSION,
        "diagnostic_target": "numerical_safety_fidelity_card_by_fixed_range_mul",
        "not_quality_or_utility": True,
        "layers": [
            "action_summary",
            "candidate_card",
            "mul_affinity_curve",
            "detailed_diagnostics",
            "measurement_quality",
        ],
        "primary_channels": {
            "body": "source-balanced q95 relative gradient deformation pooled over timestep bins",
            "tail": "maximum source-balanced q95 deformation over timestep bins",
            "trajectory": "128-step cumulative drift relative to no_quant natural drift",
        },
        "gauge": {
            "channel_formula": "100/(1+channel)",
            "local_only_formula": "min(body_gauge, tail_gauge)",
            "overall_formula": "min(body_gauge, tail_gauge, trajectory_gauge)",
            "trajectory_missing_policy": (
                "show Local Fidelity Gauge and keep Overall unavailable; "
                "never silently substitute local-only for full Overall"
            ),
            "forbidden_names": ["quality_score", "success_probability", "safety_probability"],
        },
        "absolute_interpretation": {
            "reference": ABSOLUTE_REFERENCE_DISTANCE,
            "reference_meaning": (
                "mathematical equal-distance reference against no_quant; "
                "not a quality pass/fail boundary"
            ),
            "low_perturbation": "Body < 1 and Tail < 1",
            "tail_attention": "Body < 1 and Tail >= 1",
            "high_perturbation": "Body >= 1",
            "hard_unsafe": "nonfinite or forced safety abort",
        },
        "affinity_curve_scale": dict(AFFINITY_SCALE_CALIBRATION),
        "candidate_sets": {
            "hard_safety_pass_candidates": (
                "hard-safety pass; candidate is not automatically deleted"
            ),
            "fidelity_retained_set": (
                "not robustly Body/Tail dominated under the current protocol"
            ),
            "stronger_perturbation_within_candidates": (
                "hard-safe but robustly Body/Tail dominated; retained as an "
                "exploration comparison option"
            ),
        },
        "evidence_separation": {
            "measurement_qa": "protocol/gate integrity only",
            "local_comparison_confidence": (
                "uncertainty of within-dataset Body/Tail comparison"
            ),
            "recommendation_maturity": (
                "availability of trajectory and external quality utility evidence"
            ),
            "natural_baseline": (
                "scale context only; never used as selector evidence"
            ),
        },
        "relative_statuses": [
            "near_best_plateau",
            "relative_retained",
            "trade_off",
            "dominated",
            "edge_unresolved",
            "inconclusive",
            "hard_unsafe",
        ],
        "automated_claims_excluded": [
            "best_final_quality_mul",
            "quantization_better_than_no_quant",
            "training_success_guarantee",
            "final_image_quality_prediction",
        ],
    }


def pairwise_win_matrix(
    bootstrap_rows: Sequence[Mapping[str, Any]],
    *,
    metric: str,
) -> dict[str, Any]:
    """P(row candidate has a lower metric than column candidate).

    Shared bootstrap iteration IDs are required. Ties contribute half a win.
    """

    by_candidate: dict[str, dict[int, float]] = defaultdict(dict)
    muls: dict[str, float] = {}
    for row in bootstrap_rows:
        value = _optional_float(row.get(metric))
        if value is None:
            continue
        candidate = _candidate_name(row)
        iteration = int(row["iteration"])
        by_candidate[candidate][iteration] = value
        muls[candidate] = _candidate_mul(row)
    candidates = sorted(by_candidate, key=lambda name: (muls[name], name))
    matrix: dict[str, dict[str, float | None]] = {}
    for left in candidates:
        matrix[left] = {}
        for right in candidates:
            if left == right:
                matrix[left][right] = None
                continue
            shared = sorted(set(by_candidate[left]) & set(by_candidate[right]))
            if not shared:
                matrix[left][right] = None
                continue
            wins = 0.0
            for iteration in shared:
                left_value = by_candidate[left][iteration]
                right_value = by_candidate[right][iteration]
                wins += 1.0 if left_value < right_value else 0.5 if left_value == right_value else 0.0
            matrix[left][right] = wins / len(shared)
    return {
        "metric": metric,
        "candidates": candidates,
        "muls": muls,
        "matrix": matrix,
    }


def source_loo_stability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    by_omitted: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        omitted = str(row.get("omitted_source_group", "")).strip()
        if omitted:
            by_omitted[omitted].append(row)
    if not by_omitted:
        return None

    body_winners: list[str] = []
    tail_winners: list[str] = []
    for omitted, members in sorted(by_omitted.items()):
        valid = [
            row
            for row in members
            if _optional_float(row.get("local_body")) is not None
            and _optional_float(row.get("local_tail")) is not None
        ]
        if not valid:
            continue
        body_winners.append(
            _candidate_name(
                min(valid, key=lambda row: (_optional_float(row["local_body"]), _candidate_mul(row)))
            )
        )
        tail_winners.append(
            _candidate_name(
                min(valid, key=lambda row: (_optional_float(row["local_tail"]), _candidate_mul(row)))
            )
        )

    def summarize(values: Sequence[str]) -> dict[str, Any] | None:
        if not values:
            return None
        counts = Counter(values)
        winner, count = min(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        return {
            "modal_candidate": winner,
            "modal_count": count,
            "total": len(values),
            "consistency": count / len(values),
            "counts": dict(sorted(counts.items())),
        }

    return {
        "omitted_source_count": len(by_omitted),
        "body": summarize(body_winners),
        "tail": summarize(tail_winners),
    }


def _phenotype_tags(
    phenotype: str,
    *,
    edge_unresolved: bool,
    body_min: str | None,
    tail_min: str | None,
    hard_safety_all_pass: bool,
) -> list[dict[str, str]]:
    mapping = {
        "broad_observed_below_anchor": ("broad_tolerant", "Broad tolerant", "試した通常範囲を比較的穏やかに受容"),
        "selective_window": ("selective_window", "Selective window", "一部のmulだけが比較的穏やか"),
        "all_high_perturbation": ("high_perturbation", "High perturbation", "試した範囲全体でno_quantとの差が大きい"),
        "unknown": ("unknown", "Unstable / Unknown", "現在の証拠だけでは体質を単純化できない"),
    }
    code, label, explanation = mapping.get(
        phenotype,
        ("unknown", "Unstable / Unknown", "現在の証拠だけでは体質を単純化できない"),
    )
    tags = [{"code": code, "label": label, "explanation": explanation}]
    if body_min and tail_min and body_min != tail_min:
        tags.append(
            {
                "code": "trade_off",
                "label": "Trade-off",
                "explanation": "BodyとTailで最も穏やかな候補が異なる",
            }
        )
    if edge_unresolved:
        tags.append(
            {
                "code": "edge_seeking",
                "label": "Edge seeking",
                "explanation": "測定範囲の端でも改善傾向が続き、真の最小点は未観測",
            }
        )
    if not hard_safety_all_pass:
        tags.append(
            {
                "code": "hard_unsafe",
                "label": "Hard unsafe",
                "explanation": "nonfiniteまたは強制安全停止を含む",
            }
        )
    return tags


def absolute_perturbation_level(
    body: Any,
    tail: Any,
    *,
    hard_safety_pass: bool,
) -> str:
    """Classify absolute local deformation without making a quality claim."""

    if not hard_safety_pass:
        return "hard_unsafe"
    body_value = _optional_float(body)
    tail_value = _optional_float(tail)
    if body_value is None or tail_value is None:
        return "unmeasurable"
    if body_value >= ABSOLUTE_REFERENCE_DISTANCE:
        return "high_perturbation"
    if tail_value >= ABSOLUTE_REFERENCE_DISTANCE:
        return "tail_attention"
    return "low_perturbation"


def _measurement_quality(
    *,
    hard_safety_all_pass: bool,
    required_gates_pass: bool | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if required_gates_pass is False or not hard_safety_all_pass:
        level = "FAIL"
        reasons.append("required gateまたはhard safetyに未通過があります")
    elif required_gates_pass is True:
        level = "PASS"
        reasons.append("利用可能なrequired gateとhard safetyは通過しています")
    else:
        level = "PARTIAL"
        reasons.append("この集約レポートでは全required gateの詳細を利用できません")
        if hard_safety_all_pass:
            reasons.append("保存済み候補のhard safety集約は全件PASSです")
    return {"level": level, "reasons": reasons, "scope": "measurement_qa"}


def _local_comparison_confidence(
    *,
    source_count: int,
    edge_unresolved: bool,
    loo: Mapping[str, Any] | None,
    detailed_bootstrap: bool,
    absolute_norm_available: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    loo_consistencies = [
        float(loo[channel]["consistency"])
        for channel in ("body", "tail")
        if loo and loo.get(channel) and loo[channel].get("consistency") is not None
    ]
    if source_count < 8 or not detailed_bootstrap:
        level = "Low"
    elif (
        source_count >= 12
        and len(loo_consistencies) == 2
        and min(loo_consistencies) >= 0.75
    ):
        level = "High"
    else:
        level = "Medium"
    if source_count < 8:
        reasons.append(f"独立sourceが{source_count}群で、8群未満です")
    else:
        reasons.append(f"独立sourceを{source_count}群使用しています")
    if detailed_bootstrap:
        reasons.append("source-cluster bootstrapの詳細分布を利用しています")
    else:
        reasons.append("保存済みcommon-core要約のみで、詳細bootstrap行は未収録です")
    if loo and loo.get("body") and loo.get("tail"):
        reasons.append(
            "source LOOの最頻一致率は"
            f"Body {loo['body']['modal_count']}/{loo['body']['total']}、"
            f"Tail {loo['tail']['modal_count']}/{loo['tail']['total']}です"
        )
    if edge_unresolved:
        reasons.append("測定範囲の端が未解決です")
    if not absolute_norm_available:
        reasons.append("旧runでは絶対gradient normが未記録です")
    return {
        "level": level,
        "reasons": reasons,
        "scope": "descriptive_local_fidelity",
    }


def _recommendation_maturity(
    *,
    trajectory_available: bool,
    edge_unresolved: bool,
) -> dict[str, Any]:
    reasons = [
        "画質Utilityを直接検証する40 epoch blind比較は未実施です",
        "独立training seedでの再現確認も未実施です",
    ]
    if trajectory_available:
        level = "Trajectory-informed"
        reasons.insert(0, "128-step Trajectoryを含む説明が可能です")
    else:
        level = "Local-only"
        reasons.insert(0, "現在は局所probeのみで、128-step Trajectoryは未測定です")
    if edge_unresolved:
        reasons.append("探索gridの端が未解決です")
    return {
        "level": level,
        "stage": "Experimental",
        "utility_evidence": "None",
        "reasons": reasons,
        "scope": "recommendation_maturity",
    }


def _reason_codes(
    row: Mapping[str, Any],
    *,
    absolute_perturbation: str,
    relative_status: str,
    edge_endpoint: bool,
    trajectory_available: bool,
) -> list[str]:
    codes = ["HARD_SAFETY_PASS"] if _as_bool(row.get("hard_safety_pass")) else ["HARD_SAFETY_FAIL"]
    codes.append(f"ABSOLUTE_{absolute_perturbation.upper()}")
    if relative_status == "dominated":
        codes.append("ROBUSTLY_BODY_TAIL_DOMINATED")
    if relative_status in {
        "relative_retained",
        "near_best_plateau",
        "trade_off",
        "edge_unresolved",
    }:
        codes.append("FIDELITY_RETAINED_SET")
    if edge_endpoint:
        codes.append("EDGE_ENDPOINT_RETAINED")
    for value in _as_list(row.get("mandatory_retention_role")):
        code = str(value).strip().upper()
        if code:
            codes.append(code)
    if not trajectory_available:
        codes.append("TRAJECTORY_NOT_MEASURED")
    return list(dict.fromkeys(codes))


def _relative_status(
    row: Mapping[str, Any],
    *,
    credible_names: set[str],
    endpoint_names: set[str],
    body_min: str | None,
    tail_min: str | None,
) -> str:
    name = _candidate_name(row)
    if not _as_bool(row.get("hard_safety_pass")):
        return "hard_unsafe"
    if _as_bool(row.get("robustly_dominated")):
        return "dominated"
    if name in endpoint_names and name in credible_names:
        return "edge_unresolved"
    if name in credible_names:
        if name == body_min or name == tail_min:
            if name == body_min and name == tail_min:
                return "near_best_plateau"
            return "trade_off"
        return "relative_retained"
    return "inconclusive"


def _classification(absolute_perturbation: str) -> str:
    return ABSOLUTE_LEVEL_LABELS.get(absolute_perturbation, "測定不能")


def _candidate_explanation(card: Mapping[str, Any]) -> str:
    status = str(card["relative_status"])
    absolute = str(card["absolute_perturbation"])
    mul = float(card["range_mul"])
    parts: list[str] = []
    if absolute == "hard_unsafe":
        return f"mul {mul:.2f}はhard safetyを通過していないため、数値比較の対象外です。"
    if absolute == "unmeasurable":
        parts.append("BodyまたはTailが未測定のため、絶対的な摂動量を分類できません。")
    elif absolute == "high_perturbation":
        parts.append("Bodyが基準1.0以上で、候補内では高摂動側です。")
    elif absolute == "tail_attention":
        parts.append("Bodyは1.0未満ですがTailが1.0以上で、厳しいtimestep帯に注意が必要です。")
    else:
        parts.append("BodyとTailがともに1.0未満で、観測上は低摂動です。")
    parts.append("基準1.0は数学的な距離基準であり、画質の合否線ではありません。")
    if status == "dominated":
        dominator = card.get("dominated_by_mul")
        probability = card.get("dominance_probability")
        detail = (
            f"mul {float(dominator):.2f}"
            if dominator is not None
            else "別候補"
        )
        if probability is not None:
            parts.append(
                f"BodyとTailの両方で{detail}より大きいというbootstrap支持が"
                f"{100.0 * float(probability):.1f}%あります。"
            )
        else:
            parts.append(f"BodyとTailの両方で{detail}より明瞭に大きい摂動です。")
        parts.append("hard unsafeではありませんが、候補内ではより強い摂動側なので比較優先度を下げます。")
    elif status == "edge_unresolved":
        parts.append("相対的なFidelity retained setに残りました。")
        parts.append("測定範囲の端点なので、真の最小摂動点は未観測です。")
    elif status == "near_best_plateau":
        parts.append("相対比較ではNear-best plateauに残り、他のretained候補との差を明確に区別できません。")
    elif status == "relative_retained":
        parts.append("相対比較でFidelity retained setに残りました。")
    elif status == "trade_off":
        parts.append("BodyとTailで相対位置が異なり、単純な一順位にできない候補です。")
    else:
        parts.append("現在の測定では相対位置を明瞭に分類できません。")

    angle = _optional_float(card.get("angle_tail"))
    gain = _optional_float(card.get("gain_tail"))
    if angle is not None and gain is not None:
        if angle > gain * 1.25:
            parts.append("Tailの原因分解では勾配方向の回転がgain変化より大きく見えます。")
        elif gain > angle * 1.25:
            parts.append("Tailの原因分解では勾配normのgain変化が方向回転より大きく見えます。")
        else:
            parts.append("Tailの方向回転とgain変化は同程度で、混合型です。")
    if card.get("trajectory") is None:
        parts.append("128-stepでの蓄積は未測定です。")
    return "".join(parts)


def build_dataset_card(
    *,
    evaluation: Mapping[str, Any],
    candidate_rows: Sequence[Mapping[str, Any]],
    detail: Mapping[str, Any] | None = None,
    gate_rows: Sequence[Mapping[str, Any]] = (),
    prospective: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidate_rows:
        raise ValueError(f"dataset {evaluation.get('dataset_id')} has no candidates")
    detail = detail or {}
    prospective = prospective or {}
    selection = dict(detail.get("selection") or {})
    credible_muls = [
        float(value)
        for value in (
            selection.get("credible_muls")
            or _as_list(evaluation.get("credible_muls"))
        )
    ]
    credible_names = {
        _candidate_name(row)
        for row in candidate_rows
        if any(_same_mul(_candidate_mul(row), value) for value in credible_muls)
    }
    body_min = str(
        selection.get("point_body_min_candidate")
        or evaluation.get("point_body_min_candidate")
        or ""
    ) or None
    tail_min = str(
        selection.get("point_tail_min_candidate")
        or evaluation.get("point_tail_min_candidate")
        or ""
    ) or None
    edge_unresolved = bool(
        selection.get("edge_unresolved")
        if "edge_unresolved" in selection
        else prospective.get(
            "edge_unresolved",
            evaluation.get("edge_unresolved_within_core", False),
        )
    )
    ordered_rows = sorted(candidate_rows, key=_candidate_mul)
    endpoint_names: set[str] = set()
    if edge_unresolved:
        retained_endpoints = {
            str(value) for value in _as_list(selection.get("retained_endpoint_candidates"))
        }
        if retained_endpoints:
            endpoint_names = retained_endpoints
        else:
            for row in (ordered_rows[0], ordered_rows[-1]):
                if _candidate_name(row) in credible_names:
                    endpoint_names.add(_candidate_name(row))

    trajectory_status = str(
        prospective.get("trajectory_status")
        or selection.get("trajectory_status")
        or "not_measured"
    )
    trajectory_available = trajectory_status == "available"
    cards: list[dict[str, Any]] = []
    for raw in ordered_rows:
        row = dict(raw)
        name = _candidate_name(row)
        body = _optional_float(row.get("local_body"))
        tail = _optional_float(row.get("local_tail"))
        trajectory = _optional_float(row.get("trajectory_risk_T"))
        hard_safety_pass = _as_bool(row.get("hard_safety_pass"))
        absolute_perturbation = absolute_perturbation_level(
            body,
            tail,
            hard_safety_pass=hard_safety_pass,
        )
        relative_status = _relative_status(
            row,
            credible_names=credible_names,
            endpoint_names=endpoint_names,
            body_min=body_min,
            tail_min=tail_min,
        )
        dominated_by = str(row.get("dominated_by", "") or "")
        dominated_by_mul = None
        if dominated_by.startswith("mul_"):
            dominated_by_mul = _optional_float(dominated_by[4:])
        card = {
            "candidate": name,
            "range_mul": _candidate_mul(row),
            "grid_role": str(row.get("grid_role", "measured")),
            "hard_safety_pass": hard_safety_pass,
            "body": body,
            "body_ci_low": _optional_float(row.get("local_body_ci_low")),
            "body_ci_high": _optional_float(row.get("local_body_ci_high")),
            "tail": tail,
            "tail_ci_low": _optional_float(row.get("local_tail_ci_low")),
            "tail_ci_high": _optional_float(row.get("local_tail_ci_high")),
            "tail_amplification": _optional_float(row.get("tail_amplification")),
            "worst_timestep_bin": (
                int(float(row["worst_timestep_bin"]))
                if _optional_float(row.get("worst_timestep_bin")) is not None
                else None
            ),
            "d_gt_1_rate": _optional_float(row.get("d_gt_1_rate")),
            "gradient_cosine_lt_0_rate": _optional_float(
                row.get("gradient_cosine_lt_0_rate")
            ),
            "symmetric_body": _optional_float(row.get("symmetric_body")),
            "symmetric_tail": _optional_float(row.get("symmetric_tail")),
            "angle_body": _optional_float(row.get("angle_body")),
            "angle_tail": _optional_float(row.get("angle_tail")),
            "gain_body": _optional_float(row.get("gain_body")),
            "gain_tail": _optional_float(row.get("gain_tail")),
            "grad_norm_noquant_median": _optional_float(
                row.get("grad_norm_noquant_median")
            ),
            "grad_norm_candidate_median": _optional_float(
                row.get("grad_norm_candidate_median")
            ),
            "grad_diff_norm_q95": _optional_float(row.get("grad_diff_norm_q95")),
            "body_min_probability": _optional_float(
                row.get("source_bootstrap_body_min_probability")
            ),
            "tail_min_probability": _optional_float(
                row.get("source_bootstrap_tail_min_probability")
            ),
            "robustly_dominated": _as_bool(row.get("robustly_dominated")),
            "dominated_by": dominated_by or None,
            "dominated_by_mul": dominated_by_mul,
            "dominance_probability": _optional_float(
                row.get("dominance_probability")
            ),
            "absolute_perturbation": absolute_perturbation,
            "absolute_perturbation_label": _classification(absolute_perturbation),
            "relative_status": relative_status,
            "classification": _classification(absolute_perturbation),
            "edge_endpoint": name in endpoint_names,
            "trajectory": trajectory,
            "trajectory_status": trajectory_status,
            "body_gauge": fidelity_gauge(body),
            "tail_gauge": fidelity_gauge(tail),
            "trajectory_gauge": fidelity_gauge(trajectory),
            "local_fidelity_gauge": local_fidelity_gauge(body, tail),
            "overall_fidelity_gauge": (
                min(
                    value
                    for value in (
                        fidelity_gauge(body),
                        fidelity_gauge(tail),
                        fidelity_gauge(trajectory),
                    )
                    if value is not None
                )
                if trajectory is not None
                else None
            ),
            "not_quality_or_utility": True,
        }
        card["reason_codes"] = _reason_codes(
            row,
            absolute_perturbation=absolute_perturbation,
            relative_status=relative_status,
            edge_endpoint=card["edge_endpoint"],
            trajectory_available=trajectory_available,
        )
        card["explanation_ja"] = _candidate_explanation(card)
        cards.append(card)

    bootstrap_rows = list(detail.get("bootstrap_rows") or [])
    loo = source_loo_stability(list(detail.get("source_loo_rows") or []))
    gates_required = [
        row for row in gate_rows if _as_bool(row.get("required"))
    ]
    required_gates_pass: bool | None = (
        all(_as_bool(row.get("passed")) for row in gates_required)
        if gates_required
        else None
    )
    source_count = int(
        detail.get("summary", {}).get(
            "source_group_count",
            evaluation.get("source_group_count", 0),
        )
    )
    image_count = int(
        detail.get("summary", {}).get(
            "image_count",
            evaluation.get("image_count", 0),
        )
    )
    absolute_norm_available = _as_bool(
        evaluation.get("absolute_gradient_norm_available", True)
    )
    hard_safety_all_pass = all(card["hard_safety_pass"] for card in cards)
    measurement_quality = _measurement_quality(
        hard_safety_all_pass=hard_safety_all_pass,
        required_gates_pass=required_gates_pass,
    )
    local_comparison_confidence = _local_comparison_confidence(
        source_count=source_count,
        edge_unresolved=edge_unresolved,
        loo=loo,
        detailed_bootstrap=bool(bootstrap_rows),
        absolute_norm_available=absolute_norm_available,
    )
    recommendation_maturity = _recommendation_maturity(
        trajectory_available=trajectory_available,
        edge_unresolved=edge_unresolved,
    )
    phenotype = str(
        detail.get("summary", {}).get(
            "local_phenotype",
            evaluation.get("phenotype", "unknown"),
        )
    )
    phenotype_tags = _phenotype_tags(
        phenotype,
        edge_unresolved=edge_unresolved,
        body_min=body_min,
        tail_min=tail_min,
        hard_safety_all_pass=hard_safety_all_pass,
    )

    hard_safety_pass_candidates = [
        card for card in cards if card["hard_safety_pass"]
    ]
    fidelity_retained = [
        card
        for card in hard_safety_pass_candidates
        if card["candidate"] in credible_names and not card["robustly_dominated"]
    ]
    stronger_perturbation = [
        card
        for card in hard_safety_pass_candidates
        if card["relative_status"] == "dominated"
    ]
    inconclusive = [
        card
        for card in hard_safety_pass_candidates
        if card not in fidelity_retained and card not in stronger_perturbation
    ]
    body_representative = next(
        (card for card in fidelity_retained if card["candidate"] == body_min),
        None,
    )
    tail_representative = next(
        (card for card in fidelity_retained if card["candidate"] == tail_min),
        None,
    )
    all_candidates_retained = bool(fidelity_retained) and (
        len(fidelity_retained) == len(hard_safety_pass_candidates)
    )
    single_representative: dict[str, Any] | None = None
    if not fidelity_retained:
        representative_selection_state = "no_retained_candidate"
        representative_selection_reason = (
            "相対的なFidelity retained候補がないため、単一代表を選びません。"
        )
    elif len(fidelity_retained) == 1:
        single_representative = fidelity_retained[0]
        representative_selection_state = "provisional_single"
        representative_selection_reason = (
            "Fidelity retained候補が1点だけなので暫定代表とします。"
        )
    elif all_candidates_retained:
        representative_selection_state = "no_single_all_retained"
        representative_selection_reason = (
            "全候補が相対的に残っており、単一代表へ削減する根拠がありません。"
        )
    elif (
        body_representative
        and tail_representative
        and body_representative["candidate"] != tail_representative["candidate"]
    ):
        representative_selection_state = "no_single_body_tail_tradeoff"
        representative_selection_reason = (
            "Body代表とTail代表が異なるため、単一代表を作りません。"
        )
    else:
        single_representative = min(
            fidelity_retained,
            key=lambda card: (abs(card["range_mul"] - 3.205), card["range_mul"]),
        )
        representative_selection_state = "deterministic_plateau_representative"
        representative_selection_reason = (
            "plateau内でclip_rate_low初期値3.205に最も近い点を決定論的に選択しました。"
            "固定mulとauto presetは同一ではありません。"
        )
    stronger_representative = (
        max(
            stronger_perturbation,
            key=lambda card: (
                max(card["body"] or -math.inf, card["tail"] or -math.inf),
                -card["range_mul"],
            ),
        )
        if stronger_perturbation
        else None
    )
    absolute_levels = {
        card["absolute_perturbation"] for card in hard_safety_pass_candidates
    }
    if not hard_safety_all_pass:
        absolute_response = "includes_hard_unsafe"
    elif absolute_levels == {"low_perturbation"}:
        absolute_response = "all_low_perturbation"
    elif absolute_levels == {"tail_attention"}:
        absolute_response = "all_tail_attention"
    elif absolute_levels == {"high_perturbation"}:
        absolute_response = "all_high_perturbation"
    else:
        absolute_response = "mixed_absolute_response"
    edge_direction = "resolved"
    if edge_unresolved and cards:
        lower_edge = cards[0]["candidate"] in endpoint_names
        upper_edge = cards[-1]["candidate"] in endpoint_names
        edge_direction = (
            "both" if lower_edge and upper_edge else "lower" if lower_edge else "upper" if upper_edge else "unknown"
        )
    natural = detail.get("natural_baseline")
    if natural and natural.get("valid"):
        natural_body = _optional_float(natural.get("local_body"))
        natural_tail = _optional_float(natural.get("local_tail"))
        for card in cards:
            card["body_vs_natural"] = (
                card["body"] / natural_body
                if card["body"] is not None and natural_body
                else None
            )
            card["tail_vs_natural"] = (
                card["tail"] / natural_tail
                if card["tail"] is not None and natural_tail
                else None
            )
    else:
        natural = None
        for card in cards:
            card["body_vs_natural"] = None
            card["tail_vs_natural"] = None

    if single_representative:
        minimum_comparison_set = [
            "no_quant",
            f"mul {single_representative['range_mul']:.2f}（暫定代表）",
        ]
    elif (
        representative_selection_state == "no_single_body_tail_tradeoff"
        and body_representative
        and tail_representative
    ):
        minimum_comparison_set = [
            "no_quant",
            f"mul {body_representative['range_mul']:.2f}（Body代表）",
            f"mul {tail_representative['range_mul']:.2f}（Tail代表）",
        ]
    elif all_candidates_retained:
        minimum_comparison_set = ["no_quant"] + [
            f"mul {card['range_mul']:.2f}" for card in fidelity_retained
        ]
    else:
        minimum_comparison_set = ["no_quant"]
    actions = {
        "stability_first": ["no_quant"] + [
            f"mul {card['range_mul']:.2f}" for card in fidelity_retained
        ],
        "minimum_comparison_set": minimum_comparison_set,
        "minimum_comparison_state": representative_selection_state,
        "minimum_comparison_reason": representative_selection_reason,
        "exploration_comparison_set": ["no_quant"] + [
            f"mul {card['range_mul']:.2f}" for card in hard_safety_pass_candidates
        ],
        "stronger_perturbation_options": [
            f"mul {card['range_mul']:.2f}" for card in stronger_perturbation
        ],
        "additional_measurement": (
            "必要なら独立profile seedでLocal候補集合とedge傾向を再確認してください。"
            + (
                "上端でも改善中ですが、最良mul探索のために無制限なedge拡張は行いません。"
                if edge_unresolved
                else ""
            )
            + "128-step Trajectoryは研究用で、通常の候補削減には使いません。"
        ),
    }
    pairwise = {
        "body": pairwise_win_matrix(bootstrap_rows, metric="local_body")
        if bootstrap_rows
        else None,
        "tail": pairwise_win_matrix(bootstrap_rows, metric="local_tail")
        if bootstrap_rows
        else None,
    }
    return {
        "dataset_id": str(evaluation["dataset_id"]),
        "label": str(evaluation.get("label") or evaluation["dataset_id"]),
        "family_id": str(evaluation.get("family_id") or evaluation["dataset_id"]),
        "evidence_role": str(evaluation.get("evidence_role", "")),
        "protocol_scope": (
            "full_local_grid"
            if detail
            else "common_core_summary_only"
        ),
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "source_group_count": source_count,
        "image_count": image_count,
        "candidate_grid": [card["range_mul"] for card in cards],
        "candidate_count": len(cards),
        "hard_safety_pass_count": len(hard_safety_pass_candidates),
        "fidelity_retained_count": len(fidelity_retained),
        "relatively_stronger_count": len(stronger_perturbation),
        "inconclusive_count": len(inconclusive),
        "relative_reduction_count": (
            len(hard_safety_pass_candidates) - len(fidelity_retained)
        ),
        "absolute_response": absolute_response,
        "absolute_response_label": DATASET_ABSOLUTE_RESPONSE_LABELS[absolute_response],
        "local_comparison_summary": representative_selection_state,
        "edge_unresolved": edge_unresolved,
        "edge_direction": edge_direction,
        "edge_extension_recommended": [
            float(value)
            for value in _as_list(
                selection.get("edge_extension_recommended")
                or prospective.get("edge_extension_recommended")
            )
            if _optional_float(value) is not None
        ],
        "trajectory_status": trajectory_status,
        "trajectory_available": trajectory_available,
        "hard_safety_all_pass": hard_safety_all_pass,
        "required_gates_pass": required_gates_pass,
        "phenotype_tags": phenotype_tags,
        "measurement_quality": measurement_quality,
        "local_comparison_confidence": local_comparison_confidence,
        "recommendation_maturity": recommendation_maturity,
        "utility_evidence": recommendation_maturity["utility_evidence"],
        "candidate_cards": cards,
        "hard_safety_pass_muls": [
            card["range_mul"] for card in hard_safety_pass_candidates
        ],
        "fidelity_retained_muls": [
            card["range_mul"] for card in fidelity_retained
        ],
        "relatively_stronger_muls": [
            card["range_mul"] for card in stronger_perturbation
        ],
        "body_representative_mul": (
            body_representative["range_mul"] if body_representative else None
        ),
        "tail_representative_mul": (
            tail_representative["range_mul"] if tail_representative else None
        ),
        "single_representative_mul": (
            single_representative["range_mul"] if single_representative else None
        ),
        "representative_selection_state": representative_selection_state,
        "representative_selection_reason": representative_selection_reason,
        "stronger_representative_mul": (
            stronger_representative["range_mul"]
            if stronger_representative
            else None
        ),
        "actions": actions,
        "natural_baseline": natural,
        "pairwise": pairwise,
        "source_loo": loo,
        "timestep_rows": list(detail.get("timestep_rows") or []),
        "gate_rows": [dict(row) for row in gate_rows],
        "known_note": str(evaluation.get("known_winner_note", "")),
        "known_direction": str(evaluation.get("known_direction", "")),
        "known_labels_used_only_after_selector": True,
        "not_quality_or_utility": True,
    }


def build_report_model(
    *,
    cross_summary: Mapping[str, Any],
    evaluations: Sequence[Mapping[str, Any]],
    candidate_rows_by_dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    detail_by_dataset: Mapping[str, Mapping[str, Any]] | None = None,
    gate_rows: Sequence[Mapping[str, Any]] = (),
    prospective_runs: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    detail_by_dataset = detail_by_dataset or {}
    gates_by_dataset: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in gate_rows:
        gates_by_dataset[str(row.get("dataset_id", ""))].append(row)
    prospective_by_dataset = {
        str(row.get("dataset_id")): row for row in prospective_runs
    }
    datasets = []
    for evaluation in evaluations:
        dataset_id = str(evaluation["dataset_id"])
        datasets.append(
            build_dataset_card(
                evaluation=evaluation,
                candidate_rows=candidate_rows_by_dataset[dataset_id],
                detail=detail_by_dataset.get(dataset_id),
                gate_rows=gates_by_dataset.get(dataset_id, ()),
                prospective=prospective_by_dataset.get(dataset_id),
            )
        )

    common_345 = []
    for dataset in datasets:
        for card in dataset["candidate_cards"]:
            if _same_mul(card["range_mul"], 3.45):
                common_345.append(
                    {
                        "dataset_id": dataset["dataset_id"],
                        "label": dataset["label"],
                        "body": card["body"],
                        "tail": card["tail"],
                        "local_fidelity_gauge": card["local_fidelity_gauge"],
                        "absolute_perturbation": card["absolute_perturbation"],
                        "absolute_perturbation_label": card[
                            "absolute_perturbation_label"
                        ],
                        "relative_status": card["relative_status"],
                        "measurement_quality": dataset["measurement_quality"]["level"],
                        "local_comparison_confidence": dataset[
                            "local_comparison_confidence"
                        ]["level"],
                    }
                )
                break
    common_345.sort(
        key=lambda row: (
            row["tail"] if row["tail"] is not None else math.inf,
            row["dataset_id"],
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "report_contract": report_contract(),
        "affinity_curve_scale": dict(AFFINITY_SCALE_CALIBRATION),
        "diagnostic_target": "practical_numerical_safety_fidelity_card",
        "not_quality_or_utility": True,
        "cross_summary": dict(cross_summary),
        "dataset_count": len(datasets),
        "datasets": datasets,
        "cross_dataset_common_mul_3_45": common_345,
        "provenance": dict(provenance or {}),
    }


def build_single_dataset_report_model(
    *,
    dataset_id: str,
    candidate_rows: Sequence[Mapping[str, Any]],
    detail: Mapping[str, Any],
    label: str | None = None,
    gate_rows: Sequence[Mapping[str, Any]] = (),
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the production Local-only card for one completed dataset analysis."""
    normalized_id = str(dataset_id).strip()
    if not normalized_id:
        raise ValueError("dataset_id must be non-empty")
    if not candidate_rows:
        raise ValueError("candidate_rows must be non-empty")
    summary = dict(detail.get("summary") or {})
    selection = dict(detail.get("selection") or {})
    evaluation = {
        "dataset_id": normalized_id,
        "label": str(label or normalized_id),
        "family_id": normalized_id,
        "evidence_role": "single_dataset_local_diagnostic",
        "source_group_count": int(summary.get("source_group_count", 0)),
        "image_count": int(summary.get("image_count", 0)),
        "phenotype": str(summary.get("local_phenotype", "unknown")),
        "credible_muls": list(selection.get("credible_muls") or ()),
        "point_body_min_candidate": selection.get("point_body_min_candidate"),
        "point_tail_min_candidate": selection.get("point_tail_min_candidate"),
        "edge_unresolved_within_core": bool(selection.get("edge_unresolved")),
        "absolute_gradient_norm_available": True,
    }
    detail_payload = dict(detail)
    detail_payload["summary"] = summary
    detail_payload["selection"] = selection
    return build_report_model(
        cross_summary={
            "scope": "single_dataset_local_only",
            "not_quality_or_utility": True,
            "trajectory_product_role": "research_only_keep_product_local_only",
        },
        evaluations=[evaluation],
        candidate_rows_by_dataset={normalized_id: list(candidate_rows)},
        detail_by_dataset={normalized_id: detail_payload},
        gate_rows=gate_rows,
        prospective_runs=[
            {
                "dataset_id": normalized_id,
                "trajectory_status": "not_measured_research_only",
            }
        ],
        provenance=provenance,
    )

def _fmt(value: Any, digits: int = 3, missing: str = "未測定") -> str:
    number = _optional_float(value)
    return missing if number is None else f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 1, missing: str = "未測定") -> str:
    number = _optional_float(value)
    return missing if number is None else f"{100.0 * number:.{digits}f}%"


def _status_ja(status: str) -> str:
    return {
        "near_best_plateau": "Near-best plateau",
        "relative_retained": "Relative retained",
        "trade_off": "Trade-off",
        "dominated": "Dominated",
        "edge_unresolved": "Edge unresolved",
        "inconclusive": "Inconclusive",
        "hard_unsafe": "Hard unsafe",
    }.get(status, status)


def _pill(text: str, kind: str = "neutral") -> str:
    return f'<span class="pill {html.escape(kind)}">{html.escape(text)}</span>'


def _mul_list(values: Sequence[float]) -> str:
    if not values:
        return "該当なし"
    return " / ".join(f"mul {float(value):.2f}" for value in values)


def _curve_svg(
    cards: Sequence[Mapping[str, Any]],
    *,
    fixed_y_max: float | None = None,
    edge_direction: str = "resolved",
) -> str:
    width, height = 760, 330
    left, right, top, bottom = 58, 24, 28, 54
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [float(card["range_mul"]) for card in cards]
    if not xs:
        return ""
    x_min, x_max = min(xs), max(xs)
    if _same_mul(x_min, x_max):
        x_min -= 0.1
        x_max += 0.1
    candidates_y = []
    for card in cards:
        for key in ("body_ci_high", "tail_ci_high", "body", "tail"):
            value = _optional_float(card.get(key))
            if value is not None:
                candidates_y.append(value)
    auto_y_max = max(candidates_y or [ABSOLUTE_REFERENCE_DISTANCE]) * 1.08
    auto_y_max = max(auto_y_max, ABSOLUTE_REFERENCE_DISTANCE * 1.08)
    if fixed_y_max is not None and float(fixed_y_max) <= 0.0:
        raise ValueError("fixed_y_max must be positive")
    y_max = float(fixed_y_max) if fixed_y_max is not None else auto_y_max
    scale_mode = "fixed" if fixed_y_max is not None else "dataset-auto"
    overflow_values = [
        (float(card["range_mul"]), key, value)
        for card in cards
        for key in ("body_ci_high", "tail_ci_high", "body", "tail")
        if (value := _optional_float(card.get(key))) is not None and value > y_max
    ]

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + plot_h - min(value, y_max) / y_max * plot_h

    def line_points(key: str) -> str:
        return " ".join(
            f"{sx(float(card['range_mul'])):.1f},{sy(float(card[key])):.1f}"
            for card in cards
            if _optional_float(card.get(key)) is not None
        )

    body_errors = []
    tail_errors = []
    for card in cards:
        x = sx(float(card["range_mul"]))
        for key_low, key_high, color, target in (
            ("body_ci_low", "body_ci_high", "#2563eb", body_errors),
            ("tail_ci_low", "tail_ci_high", "#d97706", tail_errors),
        ):
            low = _optional_float(card.get(key_low))
            high = _optional_float(card.get(key_high))
            if low is not None and high is not None:
                target.append(
                    f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(low):.1f}" y2="{sy(high):.1f}" '
                    f'stroke="{color}" stroke-width="7" opacity=".16"/>'
                )
    y_ticks = []
    for index in range(5):
        value = y_max * index / 4
        y = sy(value)
        y_ticks.append(
            f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#d7dee9" stroke-width="1"/>'
            f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" class="svg-label">{value:.2f}</text>'
        )
    x_ticks = [
        f'<text x="{sx(value):.1f}" y="{height-20}" text-anchor="middle" class="svg-label">{value:.2f}</text>'
        for value in xs
    ]
    edge_lines = [
        f'<line x1="{sx(float(card["range_mul"])):.1f}" x2="{sx(float(card["range_mul"])):.1f}" '
        f'y1="{top}" y2="{top+plot_h}" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 5"/>'
        for card in cards
        if card.get("edge_endpoint")
    ]
    overflow_markers = []
    for card in cards:
        x = sx(float(card["range_mul"]))
        for prefix, color, offset in (
            ("body", "#2563eb", -6.0),
            ("tail", "#d97706", 6.0),
        ):
            values = [
                _optional_float(card.get(prefix)),
                _optional_float(card.get(f"{prefix}_ci_high")),
            ]
            if any(value is not None and value > y_max for value in values):
                marker_x = x + offset
                overflow_markers.append(
                    f'<polygon points="{marker_x-5:.1f},{top+9:.1f} '
                    f'{marker_x+5:.1f},{top+9:.1f} {marker_x:.1f},{top+1:.1f}" '
                    f'fill="{color}"><title>mul {float(card["range_mul"]):.2f} '
                    f'{prefix} はY軸上限 {y_max:.1f} を超過</title></polygon>'
                )
    reference_y = sy(ABSOLUTE_REFERENCE_DISTANCE)
    reference_line = (
        f'<line x1="{left}" x2="{width-right}" y1="{reference_y:.1f}" '
        f'y2="{reference_y:.1f}" stroke="#b91c1c" stroke-width="1.5" '
        'stroke-dasharray="7 5"/>'
        f'<text x="{width-right-4}" y="{reference_y-6:.1f}" text-anchor="end" '
        'class="svg-reference">基準 1.0（画質の合否線ではない）</text>'
    )
    preset_lines = []
    preset_label_y = top + 34
    for label, value, color in PRACTICAL_PRESET_MARKERS:
        if x_min <= value <= x_max:
            x = sx(value)
            preset_lines.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top+plot_h}" '
                f'stroke="{color}" stroke-width="1" stroke-dasharray="2 5" opacity=".8"/>'
                f'<text x="{x:.1f}" y="{preset_label_y}" text-anchor="middle" '
                f'class="svg-preset" fill="{color}">{html.escape(label)} ≈ {value:.3f}</text>'
            )
            preset_label_y += 14
    edge_hint = ""
    if edge_direction in {"upper", "both"}:
        edge_hint += (
            f'<text x="{width-right}" y="{height-34}" text-anchor="end" '
            'class="svg-edge">上側未解決 →</text>'
        )
    if edge_direction in {"lower", "both"}:
        edge_hint += (
            f'<text x="{left}" y="{height-34}" text-anchor="start" '
            'class="svg-edge">← 下側未解決</text>'
        )
    if fixed_y_max is not None:
        scale_note = (
            f'<p class="micro scale-note"><strong>固定Y軸 0–{y_max:.1f}:</strong> '
            f'現行{AFFINITY_SCALE_CALIBRATION["calibration_dataset_configs"]}設定・'
            f'{AFFINITY_SCALE_CALIBRATION["calibration_point_estimates"]}個のBody/Tail点推定を校正母集団とし、'
            f'観測最大{AFFINITY_SCALE_CALIBRATION["observed_point_max"]:.3f}を含む表示範囲です。'
        )
        if overflow_values:
            scale_note += (
                f' CI上限等{len(overflow_values)}点が上限を超えたため、上向き三角で示して上端で打ち切りました。'
                '正確な値は下の候補表を参照してください。'
            )
        scale_note += "</p>"
    else:
        scale_note = (
            '<p class="micro scale-note">この拡大図だけはdatasetごとの自動Y軸です。'
            '曲線の細かな形を見る用途で、別datasetとの高さ比較には使いません。</p>'
        )
    return f"""
<svg class="chart" data-scale-mode="{scale_mode}" data-y-max="{y_max:.6f}" viewBox="0 0 {width} {height}" role="img" aria-label="mulに対するBodyとTailの曲線">
  <style>.svg-label{{font:12px system-ui,sans-serif;fill:#536174}}.svg-reference{{font:11px system-ui,sans-serif;fill:#991b1b;font-weight:700}}.svg-preset{{font:10px system-ui,sans-serif;font-weight:700}}.svg-edge{{font:11px system-ui,sans-serif;fill:#6d28d9;font-weight:700}}</style>
  {''.join(y_ticks)}
  {reference_line}
  {''.join(preset_lines)}
  {''.join(edge_lines)}
  {''.join(overflow_markers)}
  {''.join(body_errors)}
  {''.join(tail_errors)}
  <polyline points="{line_points('body')}" fill="none" stroke="#2563eb" stroke-width="3"/>
  <polyline points="{line_points('tail')}" fill="none" stroke="#d97706" stroke-width="3"/>
  {''.join(f'<circle cx="{sx(float(card["range_mul"])):.1f}" cy="{sy(float(card["body"])):.1f}" r="4" fill="#2563eb"/>' for card in cards if card.get("body") is not None)}
  {''.join(f'<rect x="{sx(float(card["range_mul"]))-4:.1f}" y="{sy(float(card["tail"]))-4:.1f}" width="8" height="8" fill="#d97706"/>' for card in cards if card.get("tail") is not None)}
  {''.join(x_ticks)}
  {edge_hint}
  <text x="{width/2}" y="{height-3}" text-anchor="middle" class="svg-label">range_mul</text>
  <text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" class="svg-label">gradient deformation（小さいほどno_quantに近い）</text>
  <g transform="translate({left+8},{top+8})">
    <circle cx="0" cy="0" r="4" fill="#2563eb"/><text x="10" y="4" class="svg-label">Body</text>
    <rect x="70" y="-4" width="8" height="8" fill="#d97706"/><text x="84" y="4" class="svg-label">Tail</text>
    <line x1="132" x2="150" y1="0" y2="0" stroke="#7c3aed" stroke-width="2" stroke-dasharray="5 5"/><text x="156" y="4" class="svg-label">edge unresolved</text>
  </g>
</svg>
<p class="micro">preset線は初期値の位置だけを示します。固定mul測定とauto presetの挙動は同一ではありません。</p>
{scale_note}
"""


def _cause_svg(cards: Sequence[Mapping[str, Any]]) -> str:
    width = 760
    row_h = 34
    height = 56 + row_h * len(cards)
    values = [
        float(value)
        for card in cards
        for value in (
            card.get("symmetric_tail"),
            card.get("angle_tail"),
            card.get("gain_tail"),
        )
        if _optional_float(value) is not None
    ]
    maximum = max(values or [1.0])
    left = 88
    plot_w = width - left - 32
    colors = {"symmetric_tail": "#6b7280", "angle_tail": "#7c3aed", "gain_tail": "#0f766e"}
    bars = []
    for index, card in enumerate(cards):
        y = 38 + index * row_h
        bars.append(
            f'<text x="{left-12}" y="{y+11}" text-anchor="end" class="svg-label">{float(card["range_mul"]):.2f}</text>'
        )
        for offset, key in enumerate(("symmetric_tail", "angle_tail", "gain_tail")):
            value = _optional_float(card.get(key))
            if value is None:
                continue
            bar_w = value / maximum * plot_w
            yy = y + offset * 8
            bars.append(
                f'<rect x="{left}" y="{yy}" width="{bar_w:.1f}" height="6" rx="3" fill="{colors[key]}"/>'
                f'<text x="{min(left + bar_w + 5, width - 34):.1f}" y="{yy + 6}" '
                f'class="svg-value">{value:.3f}</text>'
            )
    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="Tail原因分解">
  <style>.svg-label{{font:12px system-ui,sans-serif;fill:#536174}}.svg-value{{font:10px system-ui,sans-serif;fill:#334155;font-weight:700}}</style>
  {''.join(bars)}
  <g transform="translate({left},{height-12})">
    <rect x="0" y="-7" width="12" height="7" fill="#6b7280"/><text x="17" y="0" class="svg-label">symmetric</text>
    <rect x="100" y="-7" width="12" height="7" fill="#7c3aed"/><text x="117" y="0" class="svg-label">angle</text>
    <rect x="183" y="-7" width="12" height="7" fill="#0f766e"/><text x="200" y="0" class="svg-label">gain</text>
  </g>
</svg>
"""


def _pairwise_table(matrix: Mapping[str, Any] | None, title: str) -> str:
    if not matrix:
        return '<p class="muted">詳細bootstrap行がないため、このprototypeでは勝率行列を表示できません。</p>'
    candidates = list(matrix["candidates"])
    muls = matrix["muls"]
    rows = []
    for left in candidates:
        cells = []
        for right in candidates:
            value = matrix["matrix"][left][right]
            if value is None:
                cells.append('<td class="diag">―</td>')
            else:
                kind = "win-high" if value >= 0.8 else "win-low" if value <= 0.2 else "win-mid"
                cells.append(f'<td class="{kind}">{100*value:.1f}%</td>')
        rows.append(
            f'<tr><th scope="row">{float(muls[left]):.2f}</th>{"".join(cells)}</tr>'
        )
    headers = "".join(f'<th scope="col">{float(muls[name]):.2f}</th>' for name in candidates)
    return f"""
<div class="matrix-wrap">
<h4>{html.escape(title)}</h4>
<p class="micro">行のmulが列のmulより数値的に穏やか（値が小さい）であるbootstrap確率。色だけでなく数値を併記します。</p>
<table class="matrix"><thead><tr><th>行＼列</th>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table>
</div>
"""


def _timestep_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return '<p class="muted">timestep別詳細は、この保存済み集計には含まれていません。</p>'
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    muls: dict[str, float] = {}
    bins: set[int] = set()
    for row in rows:
        candidate = _candidate_name(row)
        by_candidate[candidate].append(row)
        muls[candidate] = _candidate_mul(row)
        bins.add(int(row["timestep_bin"]))
    header = "".join(
        (
            f"<th>bin {index}<div class=\"micro\">"
            f"t={html.escape(TIMESTEP_BIN_LABELS.get(index, ('範囲不明', 'noise不明'))[0])}・"
            f"{html.escape(TIMESTEP_BIN_LABELS.get(index, ('範囲不明', 'noise不明'))[1])}"
            "</div></th>"
        )
        for index in sorted(bins)
    )
    body = []
    for candidate in sorted(by_candidate, key=lambda name: muls[name]):
        values = {int(row["timestep_bin"]): row for row in by_candidate[candidate]}
        cells = []
        for index in sorted(bins):
            row = values.get(index)
            if not row:
                cells.append("<td>未測定</td>")
                continue
            value = _optional_float(row.get("source_balanced_q95_relative_distance"))
            cls = "worst" if _as_bool(row.get("is_worst_timestep_bin")) else ""
            cells.append(f'<td class="{cls}">{_fmt(value)}</td>')
        body.append(f'<tr><th scope="row">{muls[candidate]:.2f}</th>{"".join(cells)}</tr>')
    return f'<table><thead><tr><th>mul</th>{header}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _candidate_table(dataset: Mapping[str, Any]) -> str:
    rows = []
    for card in dataset["candidate_cards"]:
        gauge = (
            f'{card["overall_fidelity_gauge"]:.0f}'
            if card["overall_fidelity_gauge"] is not None
            else f'Local {card["local_fidelity_gauge"]:.0f}'
            if card["local_fidelity_gauge"] is not None
            else "未測定"
        )
        trajectory = (
            _fmt(card["trajectory"])
            if card["trajectory"] is not None
            else "未測定"
        )
        worst_bin = card["worst_timestep_bin"]
        timestep_detail = (
            f"bin {worst_bin}・t={TIMESTEP_BIN_LABELS.get(worst_bin, ('範囲不明', 'noise不明'))[0]}・"
            f"{TIMESTEP_BIN_LABELS.get(worst_bin, ('範囲不明', 'noise不明'))[1]}"
            if worst_bin is not None
            else "worst bin未測定"
        )
        rows.append(
            f"""
<tr>
  <th scope="row">{card["range_mul"]:.2f}</th>
  <td><strong>{gauge}</strong><div class="micro">Body {_fmt(card["body_gauge"], 0)} / Tail {_fmt(card["tail_gauge"], 0)} / Trajectory {_fmt(card["trajectory_gauge"], 0)}</div></td>
  <td>{_fmt(card["body"])}<div class="micro">CI [{_fmt(card["body_ci_low"])}, {_fmt(card["body_ci_high"])}]</div></td>
  <td>{_fmt(card["tail"])}<div class="micro">CI [{_fmt(card["tail_ci_low"])}, {_fmt(card["tail_ci_high"])}]</div></td>
  <td>{_fmt(card["tail_amplification"], 2)}×<div class="micro">{html.escape(timestep_detail)}</div></td>
  <td>{trajectory}<div class="micro">{html.escape(card["trajectory_status"])}</div></td>
  <td>d&gt;1 {_pct(card["d_gt_1_rate"])}<div class="micro">cos&lt;0 {_pct(card["gradient_cosine_lt_0_rate"])}</div></td>
  <td>{_pill(card["absolute_perturbation_label"], card["absolute_perturbation"])}</td>
  <td>{_pill(_status_ja(card["relative_status"]), card["relative_status"])}</td>
</tr>
<tr class="explanation-row"><td></td><td colspan="8">{html.escape(card["explanation_ja"])}</td></tr>
"""
        )
    return "".join(rows)


def _mark(active: bool) -> str:
    if active:
        return '<span class="matrix-mark on" aria-label="該当">○</span>'
    return '<span class="matrix-mark off" aria-label="非該当">—</span>'


def _dataset_behavior_table(dataset: Mapping[str, Any]) -> str:
    active_code = str(dataset["absolute_response"])
    cards = list(dataset["candidate_cards"])
    unsafe_muls = [
        float(card["range_mul"])
        for card in cards
        if not _as_bool(card.get("hard_safety_pass"))
    ]
    level_muls: dict[str, list[float]] = defaultdict(list)
    for card in cards:
        level_muls[str(card["absolute_perturbation"])].append(
            float(card["range_mul"])
        )
    mixed_detail = " / ".join(
        f'{ABSOLUTE_LEVEL_LABELS.get(level, level)}: {_mul_list(values)}'
        for level, values in level_muls.items()
    )
    definitions = [
        (
            "all_low_perturbation",
            "絶対",
            "全候補が低摂動",
            "全候補でBody・Tailとも基準1.0未満。",
            "全測定mul",
        ),
        (
            "all_tail_attention",
            "絶対",
            "全候補でTail注意",
            "通常域のBodyは1未満だが、厳しいtimestep帯のTailは1以上。",
            "全測定mul",
        ),
        (
            "all_high_perturbation",
            "絶対",
            "全候補が高摂動",
            "全候補でBodyが基準1.0以上。",
            "全測定mul",
        ),
        (
            "mixed_absolute_response",
            "絶対",
            "mulにより摂動帯が変化",
            "mulを変えると低摂動・Tail注意・高摂動の帯をまたぐ。",
            mixed_detail,
        ),
        (
            "includes_hard_unsafe",
            "安全",
            "Hard unsafeを含む",
            "nonfinite、強制安全停止など、数値比較より前の問題を含む。",
            _mul_list(unsafe_muls),
        ),
        (
            "relatively_stronger",
            "相対",
            "候補内でより強い摂動あり",
            "hard-safeだが、同じdataset内でBody・Tailの両方が明瞭に大きい候補がある。",
            _mul_list(dataset["relatively_stronger_muls"]),
        ),
    ]
    rows = []
    for code, channel, label, meaning, relevant in definitions:
        active = (
            bool(dataset["relatively_stronger_muls"])
            if code == "relatively_stronger"
            else active_code == code
        )
        rows.append(
            "<tr>"
            f'<td class="mark-cell">{_mark(active)}</td>'
            f"<td>{html.escape(channel)}</td>"
            f"<th scope=\"row\">{html.escape(label)}</th>"
            f"<td>{html.escape(meaning)}</td>"
            f"<td>{html.escape(relevant) if active else '—'}</td>"
            "</tr>"
        )
    return (
        '<div class="table-wrap"><table class="decision-table">'
        '<thead><tr><th>該当</th><th>軸</th><th>診断パターン</th>'
        '<th>初心者向けの意味</th><th>該当mul</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _candidate_role_matrix(dataset: Mapping[str, Any]) -> str:
    cards = list(dataset["candidate_cards"])
    muls = [float(card["range_mul"]) for card in cards]
    hard_safe = {
        float(card["range_mul"])
        for card in cards
        if _as_bool(card.get("hard_safety_pass"))
    }
    role_rows = [
        (
            "Hard-safety pass",
            "nonfinite等の安全停止なし",
            hard_safe,
        ),
        (
            "Fidelity retained",
            "候補内比較で明瞭には除外されない",
            set(float(value) for value in dataset["fidelity_retained_muls"]),
        ),
        (
            "候補内でより強い摂動",
            "hard-safeだが相対的にBody/Tailが強い",
            set(float(value) for value in dataset["relatively_stronger_muls"]),
        ),
        (
            "Body代表",
            "通常域の変形が最小",
            (
                {float(dataset["body_representative_mul"])}
                if dataset["body_representative_mul"] is not None
                else set()
            ),
        ),
        (
            "Tail代表",
            "厳しいtimestep帯の変形が最小",
            (
                {float(dataset["tail_representative_mul"])}
                if dataset["tail_representative_mul"] is not None
                else set()
            ),
        ),
        (
            "単一代表",
            "Body/Tailを1候補へ統合できた場合だけ",
            (
                {float(dataset["single_representative_mul"])}
                if dataset["single_representative_mul"] is not None
                else set()
            ),
        ),
    ]
    header = "".join(f"<th>mul<br>{mul:.2f}</th>" for mul in muls)
    rows = []
    for label, meaning, active_muls in role_rows:
        cells = "".join(
            f'<td class="mark-cell">{_mark(any(_same_mul(mul, active) for active in active_muls))}</td>'
            for mul in muls
        )
        rows.append(
            f'<tr><th scope="row">{html.escape(label)}'
            f'<span class="row-help">{html.escape(meaning)}</span></th>{cells}</tr>'
        )
    return (
        '<div class="table-wrap"><table class="role-matrix">'
        f'<thead><tr><th>役割</th>{header}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="micro">○ = 該当、— = 非該当。○がない行は「選べなかった」という診断結果です。</p>'
    )


def _overview_behavior_table(datasets: Sequence[Mapping[str, Any]]) -> str:
    columns = [
        ("all_low_perturbation", "全候補低"),
        ("all_tail_attention", "全候補Tail注意"),
        ("all_high_perturbation", "全候補高"),
        ("mixed_absolute_response", "mulで帯が変化"),
        ("includes_hard_unsafe", "Hard unsafe"),
    ]
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows = []
    for dataset in datasets:
        cells = "".join(
            f'<td class="mark-cell">{_mark(dataset["absolute_response"] == code)}</td>'
            for code, _ in columns
        )
        cells += (
            f'<td class="mark-cell">{_mark(bool(dataset["relatively_stronger_muls"]))}</td>'
        )
        rows.append(
            f'<tr><th scope="row"><button class="link-button" data-open="{html.escape(dataset["dataset_id"])}">'
            f'{html.escape(dataset["label"])}</button></th>{cells}</tr>'
        )
    return (
        '<div class="table-wrap"><table class="overview-behavior-table">'
        f'<thead><tr><th>Dataset</th>{header}<th>候補内で強め</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def _dataset_section(
    dataset: Mapping[str, Any],
    *,
    initially_visible: bool = False,
) -> str:
    tag_html = "".join(
        _pill(tag["label"], tag["code"]) for tag in dataset["phenotype_tags"]
    )
    measurement_reasons = "".join(
        f"<li>{html.escape(reason)}</li>"
        for reason in dataset["measurement_quality"]["reasons"]
    )
    local_confidence_reasons = "".join(
        f"<li>{html.escape(reason)}</li>"
        for reason in dataset["local_comparison_confidence"]["reasons"]
    )
    maturity_reasons = "".join(
        f"<li>{html.escape(reason)}</li>"
        for reason in dataset["recommendation_maturity"]["reasons"]
    )
    edge_message = (
        "測定範囲の端でも改善傾向が残っています。真の最小摂動点は未観測です。"
        if dataset["edge_unresolved"]
        else "測定範囲内で端点未解決は検出されていません。"
    )
    trajectory_message = (
        "128-step Trajectoryを利用できます。"
        if dataset["trajectory_available"]
        else "128-step Trajectoryは未測定です。このレポートはLocal-onlyです。"
    )
    natural = dataset.get("natural_baseline")
    natural_rows = "".join(
        f'<tr><th scope="row">{card["range_mul"]:.2f}</th>'
        f'<td>{_fmt(card.get("body_vs_natural"), 2)}×</td>'
        f'<td>{_fmt(card.get("tail_vs_natural"), 2)}×</td></tr>'
        for card in dataset["candidate_cards"]
    )
    natural_html = (
        f"""
<details>
  <summary>no_quant自然変動との規模比較（selectorには不使用）</summary>
  <p class="warning-inline"><strong>注意:</strong> 系統的な量子化差と、noise変更によるランダムな自然変動は同じ意味ではありません。この比は規模感の参考だけです。</p>
  <div class="qa-grid"><div><span>自然Body</span><strong>{_fmt(natural.get("local_body"))}</strong></div><div><span>自然Tail</span><strong>{_fmt(natural.get("local_tail"))}</strong></div></div>
  <table><thead><tr><th>mul</th><th>Body / 自然Body</th><th>Tail / 自然Tail</th></tr></thead><tbody>{natural_rows}</tbody></table>
</details>
"""
        if natural
        else """
<details><summary>no_quant自然変動との規模比較</summary><p class="muted">この保存済みrunでは比較用baselineを利用できません。</p></details>
"""
    )
    action = dataset["actions"]
    loo = dataset.get("source_loo")
    loo_html = (
        f"""
<p>sourceを1群ずつ外したときの最頻候補:
Body {html.escape(str(loo["body"]["modal_candidate"]))}（{loo["body"]["modal_count"]}/{loo["body"]["total"]}）、
Tail {html.escape(str(loo["tail"]["modal_candidate"]))}（{loo["tail"]["modal_count"]}/{loo["tail"]["total"]}）。</p>
"""
        if loo and loo.get("body") and loo.get("tail")
        else '<p class="muted">source LOO詳細は利用できません。</p>'
    )
    qa_rows = "".join(
        f'<tr><td>{html.escape(str(row.get("gate", "")))}</td><td>{html.escape(str(row.get("status", "")))}</td><td>{"PASS" if _as_bool(row.get("passed")) else "n/a"}</td></tr>'
        for row in dataset.get("gate_rows", [])
    ) or '<tr><td colspan="3">この集約では詳細gate行なし。hard safety集約は候補表に反映済みです。</td></tr>'
    hidden = "" if initially_visible else " hidden"
    max_measured_mul = max(
        (float(card["range_mul"]) for card in dataset["candidate_cards"]),
        default=0.0,
    )
    high_mul_note = (
        """
<div class="high-mul-note">
  <strong>mul 3.7～4.0付近も「固定mul」としては有望候補になり得ます。</strong>
  ただしauto presetが通常到達しない領域であり、この図が示すのは局所的なSafety/Fidelityだけです。
  128-stepの軌道と最終画質Utilityは別に確認します。
</div>
"""
        if max_measured_mul >= 3.7
        else ""
    )
    return f"""
<article id="dataset-{html.escape(dataset["dataset_id"])}" class="view dataset-view"{hidden}>
  <section class="dataset-intro">
    <div class="dataset-intro-main">
      <div class="eyebrow">Dataset diagnostic card ・ {html.escape(dataset["protocol_scope"])}</div>
      <h1>{html.escape(dataset["label"])}</h1>
      <div class="tag-row">{tag_html}</div>
      <p class="lead">{html.escape(edge_message)}</p>
    </div>
    <div class="evidence-strip" aria-label="証拠の状態">
      <div class="evidence-chip">
        <span>Measurement QA</span>
        <strong>{html.escape(dataset["measurement_quality"]["level"])}</strong>
      </div>
      <div class="evidence-chip">
        <span>Local比較</span>
        <strong>{html.escape(dataset["local_comparison_confidence"]["level"])}</strong>
      </div>
      <div class="evidence-chip">
        <span>推薦成熟度</span>
        <strong>{html.escape(dataset["recommendation_maturity"]["level"])}</strong>
      </div>
    </div>
  </section>

  <section class="affinity-first" aria-labelledby="affinity-{html.escape(dataset["dataset_id"])}">
    <div class="section-heading-inline">
      <div>
        <div class="eyebrow">最初に見る図</div>
        <h2 id="affinity-{html.escape(dataset["dataset_id"])}">Mul affinity curve</h2>
        <p class="section-help">mulごとに、no_quantの勾配からどれだけ離れたかをBody（普段の領域）とTail（厳しい領域）で見ます。</p>
      </div>
      <span class="scale-badge">Y軸固定 0–{AFFINITY_FIXED_Y_MAX:.1f}</span>
    </div>
    <div class="chart-card primary-chart">
      {_curve_svg(dataset["candidate_cards"], fixed_y_max=AFFINITY_FIXED_Y_MAX, edge_direction=dataset["edge_direction"])}
      <div class="interpretation-strip" aria-label="gradient deformationの読み方">
        <div><strong>0</strong><span>no_quantと一致</span></div>
        <div><strong>0～1未満</strong><span>差分normが基準勾配norm未満。よりno_quantに近い</span></div>
        <div><strong>1.0</strong><span>差分と基準勾配が同程度。画質の合否線ではない</span></div>
        <div><strong>1超</strong><span>差分normが基準勾配normを上回る</span></div>
      </div>
      <p class="metric-definition"><code>d = ||g_quant − g_noquant|| / ||g_noquant||</code>。
      Body/Tailは個々の全sampleではなく、q95を中心にまとめた厳しめの代表値です。小さいほど数値的にno_quantへ近いだけで、最終画質が良いとは限りません。</p>
      {high_mul_note}
      <details>
        <summary>このdatasetだけの自動スケールで曲線を拡大</summary>
        {_curve_svg(dataset["candidate_cards"], edge_direction=dataset["edge_direction"])}
      </details>
      <p class="trajectory-status"><strong>Trajectory:</strong> {html.escape(trajectory_message)}</p>
    </div>
  </section>

  <section class="warning">
    <strong>これは画質点ではありません。</strong>
    no_quantの学習信号に対する数値的な近さを示します。強い摂動が正則化として画質に役立つ可能性は残ります。基準1.0も画質の合否線ではありません。
  </section>

  <section>
    <h2>1. 今回どう動くか</h2>
    <p class="section-help">上の表はdataset全体の「体質」、下の表は試したmulごとの役割です。文章を読み比べなくても、○の位置で別datasetとの違いを確認できます。</p>
    <h3>Datasetの動き方</h3>
    {_dataset_behavior_table(dataset)}
    <h3 class="subheading">試したmulと役割</h3>
    {_candidate_role_matrix(dataset)}
    <div class="callout-grid beginner-actions">
      <div><strong>no_quantへの近さを優先</strong><br>{html.escape(" / ".join(action["stability_first"]))}</div>
      <div><strong>最小限の比較</strong><br>{html.escape(" / ".join(action["minimum_comparison_set"]))}<div class="micro">{html.escape(action["minimum_comparison_reason"])}</div></div>
      <div><strong>違いも含めて探索</strong><br>{html.escape(" / ".join(action["exploration_comparison_set"]))}</div>
      <div><strong>追加測定</strong><br>{html.escape(action["additional_measurement"])}</div>
    </div>
  </section>

  <section>
    <h2>2. 各mulの診断カルテ</h2>
    <p class="section-help"><strong>Body</strong> = 普段の学習信号の変形、<strong>Tail</strong> = 厳しいtimestep帯の変形、<strong>Trajectory</strong> = 短期学習中の蓄積です。小さいほどno_quantに近い数値的Fidelityです。</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>mul</th><th>Fidelity gauge</th><th>Body</th><th>Tail</th><th>Tail amplification</th><th>Trajectory</th><th>Rare events</th><th>絶対摂動</th><th>相対状態</th></tr></thead>
        <tbody>{_candidate_table(dataset)}</tbody>
      </table>
    </div>
    <p class="micro">Local gauge = min(100/(1+Body), 100/(1+Tail))。絶対摂動は基準1.0との比較、相対状態は同じdataset内の順位です。Trajectoryがない間はOverall gaugeを出しません。</p>
  </section>

  <section>
    <h2>3. 詳細診断</h2>
    <div class="chart-card">
      <h3>Tailの原因分解</h3>
      <p class="section-help">symmetric、方向回転（angle）、勾配gain変化を説明用に表示します。候補選定の追加票にはしません。</p>
      {_cause_svg(dataset["candidate_cards"])}
    </div>
    <details open>
      <summary>timestep別のTail候補</summary>
      <div class="table-wrap">{_timestep_table(dataset["timestep_rows"])}</div>
    </details>
    <details>
      <summary>候補間の勝率マトリクス</summary>
      <div class="matrix-grid">
        {_pairwise_table(dataset["pairwise"]["body"], "Body")}
        {_pairwise_table(dataset["pairwise"]["tail"], "Tail")}
      </div>
    </details>
    <details>
      <summary>source leave-one-out</summary>
      {loo_html}
    </details>
    {natural_html}
  </section>

  <section>
    <h2>4. 推奨アクション</h2>
    <div class="action-box">
      <p><strong>安定性重視:</strong> no_quantを必ず残し、Fidelity retained setを比較対象にします。</p>
      <p><strong>正則化の違いを探索:</strong> hard-safeだが候補内でより強い摂動の点を、「画質的に良い」と断言せず比較へ追加できます。</p>
      <p><strong>比較を最小化:</strong> {html.escape(" / ".join(action["minimum_comparison_set"]))}</p>
      <p><strong>代表選択:</strong> {html.escape(action["minimum_comparison_reason"])}</p>
      <p><strong>注意:</strong> edge未解決でもTrajectoryは説明目的で測定可能ですが、best mulは出しません。</p>
    </div>
  </section>

  <section>
    <h2>5. 測定品質</h2>
    <details>
      <summary>QA / provenance</summary>
      <div class="qa-grid">
        <div><span>Protocol</span><strong>{html.escape(dataset["metric_definition_version"])}</strong></div>
        <div><span>Source groups</span><strong>{dataset["source_group_count"]}</strong></div>
        <div><span>Images</span><strong>{dataset["image_count"]}</strong></div>
        <div><span>Hard safety</span><strong>{"PASS" if dataset["hard_safety_all_pass"] else "FAIL"}</strong></div>
        <div><span>Measurement QA</span><strong>{html.escape(dataset["measurement_quality"]["level"])}</strong></div>
        <div><span>Local confidence</span><strong>{html.escape(dataset["local_comparison_confidence"]["level"])}</strong></div>
        <div><span>Trajectory</span><strong>{html.escape(dataset["trajectory_status"])}</strong></div>
        <div><span>Edge</span><strong>{"UNRESOLVED" if dataset["edge_unresolved"] else "resolved in grid"}</strong></div>
      </div>
      <div class="qa-reason-grid">
        <div><h4>Measurement QAの理由</h4><ul>{measurement_reasons}</ul></div>
        <div><h4>Local comparison confidenceの理由</h4><ul>{local_confidence_reasons}</ul></div>
        <div><h4>Recommendation maturityの理由</h4><ul>{maturity_reasons}</ul></div>
      </div>
      <table><thead><tr><th>Gate</th><th>Status</th><th>Result</th></tr></thead><tbody>{qa_rows}</tbody></table>
      <p class="micro">既知画質ラベルはselector後の照合専用です。{html.escape(dataset["known_note"])}</p>
    </details>
  </section>
</article>
"""


def render_report(model: Mapping[str, Any]) -> str:
    datasets = list(model["datasets"])
    if not datasets:
        raise ValueError("at least one dataset is required")
    single_dataset = len(datasets) == 1
    initial_view = datasets[0]["dataset_id"] if single_dataset else "overview"
    options = "".join(
        f'<option value="{html.escape(dataset["dataset_id"])}"'
        f'{" selected" if dataset["dataset_id"] == initial_view else ""}>'
        f'{html.escape(dataset["label"])}</option>'
        for dataset in datasets
    )
    dataset_cards = "".join(
        f"""
<button class="dataset-tile" type="button" data-open="{html.escape(dataset["dataset_id"])}">
  <span class="eyebrow">{html.escape(dataset["dataset_id"])}</span>
  <strong>{html.escape(dataset["label"])}</strong>
  <span>{html.escape(" + ".join(tag["label"] for tag in dataset["phenotype_tags"]))}</span>
  <span>絶対摂動: {html.escape(dataset["absolute_response_label"])}</span>
  <span>Fidelity retained: {html.escape(_mul_list(dataset["fidelity_retained_muls"]))}</span>
  <span>QA {html.escape(dataset["measurement_quality"]["level"])} / Local confidence {html.escape(dataset["local_comparison_confidence"]["level"])}</span>
</button>
"""
        for dataset in datasets
    )
    common_rows = "".join(
        f"""
<tr>
  <td><button class="link-button" data-open="{html.escape(row["dataset_id"])}">{html.escape(row["label"])}</button></td>
  <td>{_fmt(row["body"])}</td><td>{_fmt(row["tail"])}</td>
  <td>{_fmt(row["local_fidelity_gauge"], 0)}</td>
  <td>{html.escape(str(row["absolute_perturbation_label"]))}</td>
  <td>{html.escape(_status_ja(str(row["relative_status"])))}</td>
  <td>{html.escape(str(row["measurement_quality"]))}</td>
  <td>{html.escape(str(row["local_comparison_confidence"]))}</td>
</tr>
"""
        for row in model["cross_dataset_common_mul_3_45"]
    )
    summary = model["cross_summary"]
    dataset_sections = "".join(
        _dataset_section(
            dataset,
            initially_visible=dataset["dataset_id"] == initial_view,
        )
        for dataset in datasets
    )
    overview_hidden = " hidden" if single_dataset else ""
    overview_selected = "" if single_dataset else " selected"
    selector_class = "dataset-selector is-single" if single_dataset else "dataset-selector"
    behavior_overview = _overview_behavior_table(datasets)
    payload = html.escape(
        json.dumps(
            {
                "schema_version": model["schema_version"],
                "dataset_ids": [dataset["dataset_id"] for dataset in datasets],
                "initial_view": initial_view,
                "affinity_curve_scale": model.get(
                    "affinity_curve_scale", AFFINITY_SCALE_CALIBRATION
                ),
                "not_quality_or_utility": True,
            },
            ensure_ascii=False,
        )
    )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SDXL DQ 診断カルテ v2.4.2 prototype</title>
<style>
:root{{--ink:#172033;--muted:#607086;--line:#d9e1ec;--paper:#f5f7fb;--card:#fff;--blue:#2563eb;--blue-soft:#eaf1ff;--orange:#d97706;--orange-soft:#fff4e2;--purple:#7c3aed;--purple-soft:#f2ebff;--teal:#0f766e;--shadow:0 12px 34px rgba(25,38,70,.09)}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.65 system-ui,-apple-system,"Segoe UI","Yu Gothic UI",sans-serif}}
header{{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.94);backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}}
.header-inner{{max-width:1280px;margin:auto;padding:12px 24px;display:flex;gap:18px;align-items:center;justify-content:space-between}}
.brand{{font-weight:800;letter-spacing:.01em}} .brand small{{display:block;color:var(--muted);font-weight:500}}
select{{padding:9px 36px 9px 12px;border:1px solid #b8c5d8;border-radius:9px;background:white;color:var(--ink);font-weight:650}}
main{{max-width:1280px;margin:auto;padding:18px 24px 80px}} section{{margin:0 0 26px}}
.hero{{display:grid;grid-template-columns:1.6fr 1fr;gap:24px;align-items:stretch;background:linear-gradient(130deg,#fff 0%,#f2f6ff 100%);padding:30px;border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}}
h1{{font-size:clamp(28px,4vw,48px);line-height:1.08;margin:5px 0 14px}} h2{{font-size:24px;margin:34px 0 12px}} h3{{font-size:17px;margin:0 0 9px}} h4{{margin:0 0 5px}}
.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:#65758c;font-weight:800}} .lead{{font-size:17px;color:#3e4d62;max-width:72ch}}
.dataset-intro{{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(380px,1fr);gap:20px;align-items:center;background:linear-gradient(130deg,#fff 0%,#f2f6ff 100%);padding:17px 22px;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}}
.dataset-intro h1{{font-size:clamp(25px,3vw,36px);margin:3px 0 9px}} .dataset-intro .lead{{font-size:14px;margin:8px 0 0}}
.evidence-strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}} .evidence-chip{{display:grid;gap:2px;min-width:0;padding:10px 11px;background:white;border:1px solid var(--line);border-radius:10px}} .evidence-chip span{{font-size:10px;color:var(--muted);font-weight:750}} .evidence-chip strong{{font-size:17px;color:var(--purple);overflow-wrap:anywhere}}
.section-heading-inline{{display:flex;justify-content:space-between;gap:16px;align-items:end;margin:0 0 8px}} .section-heading-inline h2{{margin:0}} .section-heading-inline p{{margin:3px 0 0}}
.scale-badge{{white-space:nowrap;border-radius:999px;padding:6px 11px;background:#e8eef8;color:#334155;font-size:12px;font-weight:800}}
.primary-chart{{padding:10px 18px 14px}} .primary-chart>.chart{{width:100%;height:315px;max-width:1040px;margin:0 auto}} .primary-chart details .chart{{max-height:390px}}
.interpretation-strip{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:8px 0}} .interpretation-strip>div{{display:grid;gap:1px;padding:9px 10px;background:#f4f7fb;border-radius:8px;border-top:3px solid #9aabc0}} .interpretation-strip strong{{font-size:14px}} .interpretation-strip span{{font-size:11px;color:var(--muted)}}
.metric-definition{{margin:9px 0;color:#415168;font-size:12px}} .metric-definition code{{font-weight:800;color:#27364d}} .high-mul-note{{padding:10px 12px;background:#eefbf8;border-left:4px solid var(--teal);border-radius:8px;font-size:12px}} .trajectory-status{{margin:8px 0 0;font-size:12px;color:var(--muted)}}
.confidence-card,.mini-card,.chart-card,.action-box{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px}}
.evidence-stack{{display:grid;gap:9px}} .confidence-card.compact{{padding:12px 15px}} .confidence-value{{font-size:30px;font-weight:850;color:var(--purple)}} .confidence-value.small{{font-size:21px}} .confidence-card ul{{padding-left:19px;margin:5px 0 0;color:#4d5d72;font-size:12px}}
.warning{{padding:15px 18px;border-left:5px solid var(--orange);background:var(--orange-soft);border-radius:8px}}
.warning-inline{{padding:10px 12px;border-left:4px solid var(--orange);background:var(--orange-soft);border-radius:7px}}
.tag-row{{display:flex;flex-wrap:wrap;gap:7px}} .pill{{display:inline-flex;align-items:center;border:1px solid #bcc9da;border-radius:999px;padding:3px 9px;font-size:12px;font-weight:750;background:white}}
.pill.low_perturbation,.pill.relative_retained,.pill.near_best_plateau{{background:var(--blue-soft);border-color:#9bb9f7;color:#17499f}} .pill.tail_attention,.pill.dominated{{background:var(--orange-soft);border-color:#efbd79;color:#8c4800}}
.pill.high_perturbation{{background:#fee2e2;border-color:#fca5a5;color:#991b1b}} .pill.edge_unresolved,.pill.edge_seeking{{background:var(--purple-soft);border-color:#c6a8f4;color:#5b21b6}} .pill.unmeasurable,.pill.unsafe,.pill.hard_unsafe,.pill.unknown{{background:#eee;border-color:#aaa;color:#333}}
.pill.trade_off{{background:#e8faf7;border-color:#7cc9bf;color:#0f615b}} .pill.broad_tolerant{{background:#eaf1ff;color:#17499f}} .pill.selective_window{{background:#f2ebff;color:#5b21b6}}
.summary-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}} .mini-card.accent{{border-top:4px solid var(--blue)}} .mini-card.warm{{border-top:4px solid var(--orange)}}
.big{{font-size:18px;font-weight:800;margin:6px 0}} .metric-line{{display:flex;justify-content:space-between;border-bottom:1px dashed var(--line);padding:5px 0}}
.representative-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:12px}} .representative-grid>div{{display:grid;gap:3px;padding:13px;background:white;border:1px solid var(--line);border-radius:10px}} .representative-grid span,.representative-grid small{{color:var(--muted)}}
.callout-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}} .callout-grid>div{{padding:13px;background:#edf2f8;border-radius:10px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:12px;background:white}} table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:10px 9px;text-align:left;vertical-align:top}} thead th{{position:sticky;top:0;background:#edf2f8;white-space:nowrap}} .explanation-row td{{background:#fafbfd;color:#4a596d;padding-top:7px;padding-bottom:12px}}
.subheading{{margin-top:22px}} .mark-cell{{text-align:center!important;vertical-align:middle}} .matrix-mark{{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;font-weight:900}} .matrix-mark.on{{background:#dbeafe;color:#17499f}} .matrix-mark.off{{color:#a2acba;background:#f3f4f6}} .decision-table th[scope="row"]{{min-width:190px}} .decision-table td:last-child{{min-width:180px}} .role-matrix th:first-child{{min-width:245px}} .role-matrix td{{text-align:center;vertical-align:middle}} .row-help{{display:block;color:var(--muted);font-weight:400;font-size:10px;margin-top:2px}} .overview-behavior-table td{{text-align:center}}
.micro{{font-size:11px;color:var(--muted);margin:3px 0}} .muted,.section-help{{color:var(--muted)}} .chart-grid{{display:grid;grid-template-columns:2fr 1fr;gap:15px}} .chart{{display:block;width:100%;height:auto}}
.trajectory-empty{{display:grid;place-content:center;text-align:center;min-height:270px}} .empty-icon{{font-size:64px;color:#a4aec0}}
details{{background:white;border:1px solid var(--line);border-radius:12px;padding:12px 15px;margin:10px 0}} summary{{cursor:pointer;font-weight:800}} details>table,details>.matrix-grid,details>p{{margin-top:12px}}
.matrix-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .matrix-wrap{{max-width:100%;overflow:auto}} .matrix td,.matrix th{{text-align:center}} .matrix .win-high{{background:#dce9ff;font-weight:800}} .matrix .win-low{{background:#fff0dd}} .matrix .win-mid{{background:#f1edf9}} .matrix .diag{{background:#eee;color:#777}} .worst{{outline:2px solid var(--orange);outline-offset:-3px;font-weight:800}}
.qa-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}} .qa-grid>div{{display:flex;justify-content:space-between;padding:9px;background:#f4f6fa;border-radius:8px}} .qa-grid span{{color:var(--muted)}}
.qa-reason-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:12px 0}} .qa-reason-grid>div{{padding:12px;background:#f8fafc;border:1px solid var(--line);border-radius:8px}} .qa-reason-grid ul{{margin:5px 0 0;padding-left:18px;font-size:12px;color:var(--muted)}}
.dataset-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}} .dataset-tile{{appearance:none;text-align:left;border:1px solid var(--line);background:white;border-radius:13px;padding:16px;display:grid;gap:5px;color:var(--ink);cursor:pointer;box-shadow:0 5px 16px rgba(25,38,70,.04)}} .dataset-tile:hover,.dataset-tile:focus{{border-color:#7fa5ed;transform:translateY(-1px)}} .dataset-tile strong{{font-size:16px}} .dataset-tile span:not(.eyebrow){{color:var(--muted);font-size:12px}}
.overview-hero{{background:linear-gradient(135deg,#172554,#1e3a8a);color:white;border-radius:18px;padding:32px}} .overview-hero .eyebrow,.overview-hero .lead{{color:#d8e5ff}} .overview-kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-top:20px}} .overview-kpis>div{{background:rgba(255,255,255,.1);padding:14px;border:1px solid rgba(255,255,255,.18);border-radius:11px}} .overview-kpis strong{{display:block;font-size:27px}}
.link-button{{border:0;background:none;color:#1d4ed8;text-decoration:underline;cursor:pointer;padding:0;font:inherit}} .dataset-selector.is-single{{display:none}} footer{{color:var(--muted);font-size:12px;padding-top:30px}}
@media(max-width:900px){{.hero,.dataset-intro,.chart-grid,.matrix-grid{{grid-template-columns:1fr}}.summary-grid,.callout-grid,.overview-kpis{{grid-template-columns:repeat(2,1fr)}}.dataset-grid{{grid-template-columns:1fr 1fr}}.dataset-intro{{gap:12px}}}}
@media(max-width:580px){{main{{padding:12px 10px 55px}}.header-inner{{padding:9px 12px;align-items:flex-start;flex-direction:column}}.summary-grid,.callout-grid,.overview-kpis,.dataset-grid,.qa-grid,.qa-reason-grid,.representative-grid,.interpretation-strip{{grid-template-columns:1fr}}.evidence-strip{{grid-template-columns:repeat(3,minmax(0,1fr))}}.dataset-intro{{padding:13px}}.dataset-intro h1{{font-size:24px}}.evidence-chip{{padding:7px}}.evidence-chip strong{{font-size:13px}}.section-heading-inline{{align-items:flex-start;flex-direction:column;gap:6px}}.primary-chart{{padding:8px}}.primary-chart>.chart{{height:250px}}}}
@media print{{header{{position:static}}.view[hidden]{{display:block!important;page-break-before:always}}button,select{{display:none}}body{{background:white}}main{{max-width:none}}}}
</style>
</head>
<body>
<header><div class="header-inner"><div class="brand">SDXL DQ 診断カルテ <small>v2.4.2 practical report prototype ・ Safety/Fidelity ≠ Utility</small></div><label class="{selector_class}">表示 <select id="dataset-select"><option value="overview"{overview_selected}>横断概要</option>{options}</select></label></div></header>
<main>
<section id="overview" class="view"{overview_hidden}>
  <div class="overview-hero">
    <div class="eyebrow">Practical diagnostic report prototype</div>
    <h1>結論から詳細へ潜れる、mul別の診断カルテ</h1>
    <p class="lead">数学基準1.0に対する絶対的な摂動帯と、同じdataset内の相対順位を分けて表示します。最終画質のbest mulや量子化採用可否は判定しません。</p>
    <div class="overview-kpis">
      <div><span>測定設定数</span><strong>{model["dataset_count"]}</strong></div>
      <div><span>独立family数</span><strong>{int(summary.get("family_count", summary.get("family_vote_count", 0)))}</strong></div>
      <div><span>共通core除外</span><strong>{int(summary.get("aggregate_candidate_reduction_count", 0))}/{int(summary.get("aggregate_candidate_count", 0))}</strong></div>
      <div><span>拡張local除外</span><strong>{int(summary.get("prospective_full_grid_candidate_reduction_count", 0))}/{int(summary.get("prospective_full_grid_candidate_count", 0))}</strong></div>
      <div><span>有効Trajectory</span><strong>{int(summary.get("prospective_valid_trajectory_count", 0))}</strong></div>
    </div>
  </div>
  <section class="warning"><strong>条件付きGo:</strong> 測定・QA・Body/Tailカルテは実用化へ進めます。Fidelity retained setとrobust dominanceはbeta、画質Utility推薦は未実装です。</section>
  <h2>Datasetを選ぶ</h2>
  <div class="dataset-grid">{dataset_cards}</div>
  <h2>Dataset体質の○表</h2>
  <p class="section-help">定型パターンのどこに○が付くかを横並びで比較します。「候補内で強め」は絶対的な危険ではなく、同じdataset内の相対状態です。</p>
  {behavior_overview}
  <h2>同一mul 3.45の横断比較</h2>
  <p class="section-help">metric definition 2.4の共通coreだけを比較。同じmulでもTailがdatasetごとに大きく異なることを確認できます。</p>
  <div class="table-wrap"><table><thead><tr><th>Dataset</th><th>Body</th><th>Tail</th><th>Local gauge</th><th>絶対摂動</th><th>相対状態</th><th>Measurement QA</th><th>Local confidence</th></tr></thead><tbody>{common_rows}</tbody></table></div>
  <h2>このprototypeで確認すること</h2>
  <div class="action-box"><ol><li>30秒で絶対摂動帯、Fidelity retained set、強い摂動候補、edgeが分かるか。</li><li>各mulの絶対値と相対状態を混同しないか。</li><li>Measurement QA・Local confidence・Recommendation maturityを混同しないか。</li><li>単一代表を選べない場合に、その理由が明示されるか。</li><li>初心者説明からbootstrap・source LOOまで段階的に潜れるか。</li></ol></div>
</section>
{dataset_sections}
<footer>Schema {html.escape(str(model["schema_version"]))} ・ Metric {html.escape(str(model["metric_definition_version"]))} ・ Primary affinity Y-scale 0–{AFFINITY_FIXED_Y_MAX:.1f} ・ Numerical Safety/Fidelity report only. No external CDN.</footer>
</main>
<script id="report-meta" type="application/json">{payload}</script>
<script>
(function(){{
  const select=document.getElementById("dataset-select");
  const views=[...document.querySelectorAll(".view")];
  const initialView={json.dumps(str(initial_view), ensure_ascii=False)};
  function show(id,scroll=true){{
    views.forEach(v=>{{v.hidden=v.id!==(id==="overview"?"overview":"dataset-"+id);}});
    select.value=id;
    if(scroll) window.scrollTo({{top:0,behavior:"instant"}});
  }}
  select.addEventListener("change",()=>show(select.value));
  document.querySelectorAll("[data-open]").forEach(button=>button.addEventListener("click",()=>show(button.dataset.open)));
  show(initialView,false);
}})();
</script>
</body>
</html>
"""


def candidate_csv_rows(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in model["datasets"]:
        for card in dataset["candidate_cards"]:
            rows.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "label": dataset["label"],
                    "family_id": dataset["family_id"],
                    "protocol_scope": dataset["protocol_scope"],
                    "measurement_quality": dataset["measurement_quality"]["level"],
                    "local_comparison_confidence": dataset[
                        "local_comparison_confidence"
                    ]["level"],
                    "recommendation_maturity": dataset[
                        "recommendation_maturity"
                    ]["level"],
                    "utility_evidence": dataset["utility_evidence"],
                    "phenotype": "+".join(tag["code"] for tag in dataset["phenotype_tags"]),
                    "range_mul": card["range_mul"],
                    "hard_safety_pass": card["hard_safety_pass"],
                    "fidelity_retained": (
                        card["range_mul"] in dataset["fidelity_retained_muls"]
                    ),
                    "absolute_perturbation": card["absolute_perturbation"],
                    "absolute_perturbation_label": card[
                        "absolute_perturbation_label"
                    ],
                    "classification": card["classification"],
                    "relative_status": card["relative_status"],
                    "local_fidelity_gauge": card["local_fidelity_gauge"],
                    "overall_fidelity_gauge": card["overall_fidelity_gauge"],
                    "body": card["body"],
                    "body_ci_low": card["body_ci_low"],
                    "body_ci_high": card["body_ci_high"],
                    "tail": card["tail"],
                    "tail_ci_low": card["tail_ci_low"],
                    "tail_ci_high": card["tail_ci_high"],
                    "tail_amplification": card["tail_amplification"],
                    "trajectory": card["trajectory"],
                    "trajectory_status": card["trajectory_status"],
                    "d_gt_1_rate": card["d_gt_1_rate"],
                    "gradient_cosine_lt_0_rate": card["gradient_cosine_lt_0_rate"],
                    "edge_endpoint": card["edge_endpoint"],
                    "reason_codes": "|".join(card["reason_codes"]),
                    "not_quality_or_utility": True,
                }
            )
    return rows


def reason_code_rows(model: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in model["datasets"]:
        for card in dataset["candidate_cards"]:
            for code in card["reason_codes"]:
                rows.append(
                    {
                        "dataset_id": dataset["dataset_id"],
                        "candidate": card["candidate"],
                        "range_mul": card["range_mul"],
                        "reason_code": code,
                        "explanation_ja": card["explanation_ja"],
                    }
                )
    return rows
