from __future__ import annotations

import math

from dq_profile.v24_practical_report import (
    absolute_perturbation_level,
    build_dataset_card,
    build_report_model,
    build_single_dataset_report_model,
    fidelity_gauge,
    local_fidelity_gauge,
    pairwise_win_matrix,
    render_report,
    report_contract,
    source_loo_stability,
)


def _candidate(
    mul: float,
    body: float,
    tail: float,
    *,
    dominated_by: float | None = None,
) -> dict:
    return {
        "dataset_id": "SYN",
        "candidate": f"mul_{mul:.3f}",
        "range_mul": mul,
        "grid_role": "core_grid",
        "hard_safety_pass": True,
        "local_body": body,
        "local_body_ci_low": body * 0.8,
        "local_body_ci_high": body * 1.2,
        "local_tail": tail,
        "local_tail_ci_low": tail * 0.8,
        "local_tail_ci_high": tail * 1.2,
        "tail_amplification": tail / body,
        "worst_timestep_bin": 3,
        "d_gt_1_rate": 0.01,
        "gradient_cosine_lt_0_rate": 0.0,
        "symmetric_body": body * 0.8,
        "symmetric_tail": tail * 0.8,
        "angle_body": body * 0.7,
        "angle_tail": tail * 0.7,
        "gain_body": body * 0.3,
        "gain_tail": tail * 0.3,
        "source_bootstrap_body_min_probability": 0.4,
        "source_bootstrap_tail_min_probability": 0.4,
        "robustly_dominated": dominated_by is not None,
        "dominated_by": f"mul_{dominated_by:.3f}" if dominated_by else "",
        "dominance_probability": 0.9 if dominated_by else "",
        "mandatory_retention_role": [],
        "trajectory_risk_T": "",
    }


def _detail() -> dict:
    bootstrap_rows = []
    values = {
        2.70: [(1.2, 1.3), (1.1, 1.4)],
        3.15: [(0.7, 0.8), (0.8, 0.9)],
        3.45: [(0.8, 0.9), (0.9, 0.8)],
    }
    for mul, pairs in values.items():
        for iteration, (body, tail) in enumerate(pairs):
            bootstrap_rows.append(
                {
                    "dataset_id": "SYN",
                    "candidate": f"mul_{mul:.3f}",
                    "range_mul": mul,
                    "iteration": iteration,
                    "local_body": body,
                    "local_tail": tail,
                }
            )
    loo_rows = []
    for omitted in ("source-a", "source-b"):
        for mul, body, tail in ((2.70, 1.2, 1.3), (3.15, 0.7, 0.8), (3.45, 0.8, 0.9)):
            loo_rows.append(
                {
                    "candidate": f"mul_{mul:.3f}",
                    "range_mul": mul,
                    "omitted_source_group": omitted,
                    "local_body": body,
                    "local_tail": tail,
                }
            )
    return {
        "summary": {
            "source_group_count": 9,
            "image_count": 18,
            "local_phenotype": "selective_window",
        },
        "selection": {
            "credible_muls": [3.15, 3.45],
            "point_body_min_candidate": "mul_3.150",
            "point_tail_min_candidate": "mul_3.150",
            "edge_unresolved": True,
            "retained_endpoint_candidates": ["mul_3.450"],
            "edge_extension_recommended": [3.75],
            "trajectory_status": "unknown_until_128_step_formal",
        },
        "bootstrap_rows": bootstrap_rows,
        "source_loo_rows": loo_rows,
        "timestep_rows": [],
        "natural_baseline": {
            "valid": True,
            "local_body": 2.0,
            "local_tail": 3.0,
        },
    }


def _evaluation() -> dict:
    return {
        "dataset_id": "SYN",
        "label": "Synthetic",
        "family_id": "SYN",
        "evidence_role": "test",
        "source_group_count": 9,
        "image_count": 18,
        "phenotype": "selective_window",
        "credible_muls": "[3.15, 3.45]",
        "point_body_min_candidate": "mul_3.150",
        "point_tail_min_candidate": "mul_3.150",
        "edge_unresolved_within_core": True,
        "absolute_gradient_norm_available": True,
    }


def test_fidelity_gauge_is_descriptive_worst_channel() -> None:
    assert fidelity_gauge(0.0) == 100.0
    assert math.isclose(fidelity_gauge(1.0), 50.0)
    assert fidelity_gauge(-1.0) is None
    assert local_fidelity_gauge(0.25, 1.0) == 50.0
    contract = report_contract()
    assert contract["not_quality_or_utility"] is True
    assert "quality_score" in contract["gauge"]["forbidden_names"]
    assert contract["absolute_interpretation"]["reference"] == 1.0
    assert "fidelity_retained_set" in contract["candidate_sets"]
    assert contract["affinity_curve_scale"]["y_max"] == 4.0
    assert (
        contract["affinity_curve_scale"]["mode"]
        == "fixed_primary_with_dataset_auto_zoom"
    )


def test_absolute_perturbation_is_independent_of_relative_rank() -> None:
    assert (
        absolute_perturbation_level(0.8, 0.9, hard_safety_pass=True)
        == "low_perturbation"
    )
    assert (
        absolute_perturbation_level(0.8, 1.1, hard_safety_pass=True)
        == "tail_attention"
    )
    assert (
        absolute_perturbation_level(1.0, 0.7, hard_safety_pass=True)
        == "high_perturbation"
    )
    assert (
        absolute_perturbation_level(0.2, 0.2, hard_safety_pass=False)
        == "hard_unsafe"
    )


def test_pairwise_matrix_uses_shared_iterations_and_half_ties() -> None:
    rows = [
        {"candidate": "mul_2.700", "range_mul": 2.7, "iteration": 0, "local_body": 1.0},
        {"candidate": "mul_2.700", "range_mul": 2.7, "iteration": 1, "local_body": 2.0},
        {"candidate": "mul_3.150", "range_mul": 3.15, "iteration": 0, "local_body": 1.0},
        {"candidate": "mul_3.150", "range_mul": 3.15, "iteration": 1, "local_body": 3.0},
    ]
    result = pairwise_win_matrix(rows, metric="local_body")
    assert result["matrix"]["mul_2.700"]["mul_3.150"] == 0.75
    assert result["matrix"]["mul_3.150"]["mul_2.700"] == 0.25
    assert result["matrix"]["mul_2.700"]["mul_2.700"] is None


def test_source_loo_reports_modal_candidate_consistency() -> None:
    stability = source_loo_stability(_detail()["source_loo_rows"])
    assert stability is not None
    assert stability["omitted_source_count"] == 2
    assert stability["body"]["modal_candidate"] == "mul_3.150"
    assert stability["body"]["consistency"] == 1.0
    assert stability["tail"]["modal_candidate"] == "mul_3.150"


def test_dataset_card_keeps_local_only_and_edge_uncertainty() -> None:
    rows = [
        _candidate(2.70, 1.2, 1.3, dominated_by=3.15),
        _candidate(3.15, 0.7, 0.8),
        _candidate(3.45, 0.8, 0.9),
    ]
    card = build_dataset_card(
        evaluation=_evaluation(),
        candidate_rows=rows,
        detail=_detail(),
        gate_rows=[{"required": True, "passed": True}],
        prospective={"trajectory_status": "unknown_until_valid_128_step_formal"},
    )
    assert card["trajectory_available"] is False
    assert card["fidelity_retained_muls"] == [3.15, 3.45]
    assert card["relatively_stronger_muls"] == [2.70]
    by_mul = {item["range_mul"]: item for item in card["candidate_cards"]}
    assert by_mul[2.70]["relative_status"] == "dominated"
    assert by_mul[2.70]["absolute_perturbation"] == "high_perturbation"
    assert by_mul[3.15]["absolute_perturbation"] == "low_perturbation"
    assert by_mul[3.45]["relative_status"] == "edge_unresolved"
    assert by_mul[3.15]["overall_fidelity_gauge"] is None
    assert "TRAJECTORY_NOT_MEASURED" in by_mul[3.15]["reason_codes"]
    assert card["measurement_quality"]["level"] == "PASS"
    assert card["local_comparison_confidence"]["level"] == "Medium"
    assert card["recommendation_maturity"]["level"] == "Local-only"
    assert card["single_representative_mul"] is None
    assert card["representative_selection_state"] == "no_single_edge_unresolved"
    assert card["actions"]["minimum_comparison_set"] == [
        "no_quant",
        "mul 3.15（観測上のBody／Tail代表）",
    ]
    assert card["not_quality_or_utility"] is True


def test_single_dataset_report_is_local_only_and_hides_trajectory_from_selection() -> None:
    rows = [
        _candidate(2.70, 1.2, 1.3, dominated_by=3.15),
        _candidate(3.15, 0.7, 0.8),
        _candidate(3.45, 0.8, 0.9),
    ]
    model = build_single_dataset_report_model(
        dataset_id="SYN",
        candidate_rows=rows,
        detail=_detail(),
    )
    assert model["dataset_count"] == 1
    assert model["cross_summary"]["scope"] == "single_dataset_local_only"
    dataset = model["datasets"][0]
    assert dataset["trajectory_available"] is False
    assert dataset["trajectory_status"] == "not_measured_research_only"
    assert "研究用" in dataset["actions"]["additional_measurement"]
    assert "Trajectoryを取得してください" not in dataset["actions"]["additional_measurement"]
    html = render_report(model)
    assert "dataset-selector is-single" in html
    assert "Safety/Fidelity ≠ Utility" in html

def test_all_retained_candidates_do_not_force_a_single_representative() -> None:
    rows = [
        _candidate(2.70, 0.85, 0.95),
        _candidate(3.15, 0.70, 0.80),
        _candidate(3.45, 0.75, 0.82),
    ]
    detail = _detail()
    detail["selection"].update(
        {
            "credible_muls": [2.70, 3.15, 3.45],
            "edge_unresolved": False,
            "retained_endpoint_candidates": [],
        }
    )
    evaluation = _evaluation()
    evaluation["edge_unresolved_within_core"] = False
    card = build_dataset_card(
        evaluation=evaluation,
        candidate_rows=rows,
        detail=detail,
    )
    assert card["fidelity_retained_muls"] == [2.70, 3.15, 3.45]
    assert card["single_representative_mul"] is None
    assert card["representative_selection_state"] == "no_single_all_retained"


def test_edge_unresolved_with_one_retained_candidate_still_abstains() -> None:
    rows = [
        _candidate(2.70, 1.20, 1.30, dominated_by=3.45),
        _candidate(3.45, 0.72, 0.81),
    ]
    detail = _detail()
    detail["selection"].update(
        {
            "credible_muls": [3.45],
            "point_body_min_candidate": "mul_3.450",
            "point_tail_min_candidate": "mul_3.450",
            "edge_unresolved": True,
            "retained_endpoint_candidates": ["mul_3.450"],
        }
    )
    evaluation = _evaluation()
    evaluation.update(
        {
            "credible_muls": "[3.45]",
            "point_body_min_candidate": "mul_3.450",
            "point_tail_min_candidate": "mul_3.450",
            "edge_unresolved_within_core": True,
        }
    )
    card = build_dataset_card(
        evaluation=evaluation,
        candidate_rows=rows,
        detail=detail,
    )
    assert card["body_representative_mul"] == 3.45
    assert card["tail_representative_mul"] == 3.45
    assert card["single_representative_mul"] is None
    assert card["representative_selection_state"] == "no_single_edge_unresolved"
    assert card["actions"]["minimum_comparison_set"] == [
        "no_quant",
        "mul 3.45（観測上のBody／Tail代表）",
    ]


def test_body_tail_tradeoff_keeps_two_representatives() -> None:
    rows = [
        _candidate(2.70, 1.20, 1.30, dominated_by=3.15),
        _candidate(3.15, 0.70, 0.90),
        _candidate(3.45, 0.85, 0.75),
    ]
    detail = _detail()
    detail["selection"].update(
        {
            "point_body_min_candidate": "mul_3.150",
            "point_tail_min_candidate": "mul_3.450",
            "edge_unresolved": False,
            "retained_endpoint_candidates": [],
        }
    )
    evaluation = _evaluation()
    evaluation.update(
        {
            "point_body_min_candidate": "mul_3.150",
            "point_tail_min_candidate": "mul_3.450",
            "edge_unresolved_within_core": False,
        }
    )
    card = build_dataset_card(
        evaluation=evaluation,
        candidate_rows=rows,
        detail=detail,
    )
    assert card["body_representative_mul"] == 3.15
    assert card["tail_representative_mul"] == 3.45
    assert card["single_representative_mul"] is None
    assert (
        card["representative_selection_state"]
        == "no_single_body_tail_tradeoff"
    )
    assert card["actions"]["minimum_comparison_set"] == [
        "no_quant",
        "mul 3.15（Body代表）",
        "mul 3.45（Tail代表）",
    ]


def test_rendered_report_states_scope_and_does_not_force_overall() -> None:
    rows = [
        _candidate(2.70, 1.2, 1.3, dominated_by=3.15),
        _candidate(3.15, 0.7, 0.8),
        _candidate(3.45, 0.8, 0.9),
    ]
    model = build_report_model(
        cross_summary={"family_vote_count": 1},
        evaluations=[_evaluation()],
        candidate_rows_by_dataset={"SYN": rows},
        detail_by_dataset={"SYN": _detail()},
        gate_rows=[{"dataset_id": "SYN", "required": True, "passed": True}],
        prospective_runs=[
            {"dataset_id": "SYN", "trajectory_status": "unknown_until_valid_128_step_formal"}
        ],
    )
    rendered = render_report(model)
    assert "Local-only" in rendered
    assert "Trajectoryがない間はOverall gaugeを出しません" in rendered
    assert "最終画質のbest mulや量子化採用可否は判定しません" in rendered
    assert "Edge unresolved" in rendered
    assert "候補間の勝率マトリクス" in rendered
    assert "基準 1.0（画質の合否線ではない）" in rendered
    assert "clip_rate_high 初期値" in rendered
    assert "clip_rate_low 初期値" in rendered
    assert "Fidelity retained set" in rendered
    assert "Measurement QA" in rendered
    assert "Local comparison confidence" in rendered
    assert "Recommendation maturity" in rendered
    assert "CI [" in rendered
    assert '<td class="diag">―</td>' in rendered
    assert '<td class="diag">?</td>' not in rendered
    assert "no_quant自然変動との規模比較（selectorには不使用）" in rendered
    assert "Gentle set" not in rendered
    assert "Accepted set" not in rendered
    assert model["not_quality_or_utility"] is True
    assert rendered.index("Mul affinity curve") < rendered.index("1. 今回どう動くか")
    assert 'data-scale-mode="fixed" data-y-max="4.000000"' in rendered
    assert "固定Y軸 0–4.0" in rendered
    assert "0～1未満" in rendered
    assert "差分normが基準勾配norm未満" in rendered
    assert "Datasetの動き方" in rendered
    assert "試したmulと役割" in rendered
    assert "Hard-safety pass" in rendered
    assert "Fidelity retained" in rendered
    assert '<span class="matrix-mark on" aria-label="該当">○</span>' in rendered
    assert '<article id="dataset-SYN" class="view dataset-view">' in rendered
    assert '<section id="overview" class="view" hidden>' in rendered
    assert "dataset-selector is-single" in rendered
    assert "このdatasetだけの自動スケールで曲線を拡大" in rendered
    assert "全dataset共通スケールで見る" not in rendered


def test_fixed_affinity_scale_marks_overflow_without_rescaling() -> None:
    rows = [
        _candidate(2.70, 1.2, 1.3, dominated_by=3.15),
        _candidate(3.15, 0.7, 0.8),
        _candidate(3.45, 0.8, 0.9),
    ]
    rows[0]["local_tail_ci_high"] = 5.2
    model = build_report_model(
        cross_summary={"family_vote_count": 1},
        evaluations=[_evaluation()],
        candidate_rows_by_dataset={"SYN": rows},
        detail_by_dataset={"SYN": _detail()},
    )
    rendered = render_report(model)
    assert model["affinity_curve_scale"]["y_max"] == 4.0
    assert 'data-scale-mode="fixed" data-y-max="4.000000"' in rendered
    assert "上端で打ち切りました" in rendered
    assert "正確な値は下の候補表" in rendered
    assert "Y軸上限 4.0 を超過" in rendered
