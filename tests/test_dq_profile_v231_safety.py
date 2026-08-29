from __future__ import annotations

import copy

import pytest

from dq_profile.v23_safety import analyze_profile as analyze_v23
from dq_profile.v231_safety import (
    analyze_profile,
    canonical_json_sha256,
    safety_contract,
)


def _fixture(
    *,
    lower_local: float = 0.9,
    lower_drift: float = 0.8,
    upper_local: float = 0.4,
    upper_drift: float = 0.4,
    identical: bool = False,
    force_upper_abort: bool = False,
    rare_upper_catastrophe: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    if identical:
        upper_local = lower_local
        upper_drift = lower_drift
    summary = {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "profile": {"protocol": "v2-tail-calibration", "timestep_bins": 2},
        "candidates": [
            {
                "candidate": "no_quant",
                "initial_range_mul": None,
                "forced_safety_abort": False,
                "invalid_reason": None,
            },
            {
                "candidate": "mul_2.700",
                "initial_range_mul": 2.7,
                "forced_safety_abort": False,
                "invalid_reason": None,
            },
            {
                "candidate": "mul_3.450",
                "initial_range_mul": 3.45,
                "forced_safety_abort": force_upper_abort,
                "invalid_reason": None,
            },
        ],
    }
    gradient_rows: list[dict] = []
    for image_index in range(32):
        for timestep_bin in range(2):
            for candidate, base in (
                ("mul_2.700", lower_local if timestep_bin == 1 else lower_local * 0.4),
                ("mul_3.450", upper_local if timestep_bin == 1 else upper_local * 0.4),
            ):
                value = base + (image_index - 15.5) * 0.0002
                if (
                    rare_upper_catastrophe
                    and candidate == "mul_3.450"
                    and timestep_bin == 0
                    and image_index == 31
                ):
                    value = 20.0
                gradient_rows.append(
                    {
                        "record_type": "sample",
                        "image_key": f"image_{image_index:02d}",
                        "candidate": candidate,
                        "timestep_bin": timestep_bin,
                        "noise_replica": image_index % 2,
                        "quant_repeat": (image_index // 2) % 2,
                        "relative_gradient_distance": value,
                        "gradient_cosine": -0.1 if value > 10 else 0.9,
                    }
                )
    cumulative_rows: list[dict] = []
    for index, value in enumerate((0.8, 0.9, 1.0, 1.1, 1.2)):
        cumulative_rows.append(
            {
                "record_type": "no_quant_pair",
                "module_group": "all",
                "checkpoint": 128,
                "orthogonal_drift": value,
                "repeat_a": index,
                "repeat_b": index + 1,
            }
        )
    for candidate, center in (
        ("mul_2.700", lower_drift),
        ("mul_3.450", upper_drift),
    ):
        for repeat in range(5):
            cumulative_rows.append(
                {
                    "record_type": "candidate_vs_matched_no_quant",
                    "module_group": "all",
                    "checkpoint": 128,
                    "candidate": candidate,
                    "repeat": repeat,
                    "orthogonal_drift": center + (repeat - 2) * 0.005,
                }
            )
    range_rows: list[dict] = []
    for candidate in ("mul_2.700", "mul_3.450"):
        for repeat in range(5):
            range_rows.append(
                {
                    "candidate": candidate,
                    "module_group": "all",
                    "checkpoint": 128,
                    "orthogonal_drift": 1.0,
                    "total_drift": 1.0,
                    "forced_safety_abort": False,
                    "invalid_reason": None,
                    "common_skip_matched": True,
                    "repeat": repeat,
                }
            )
    return summary, gradient_rows, cumulative_rows, range_rows


def _analyze(fixture: tuple[dict, list[dict], list[dict], list[dict]], **kwargs):
    return analyze_profile(
        summary=copy.deepcopy(fixture[0]),
        gradient_tail_rows=copy.deepcopy(fixture[1]),
        cumulative_null_rows=copy.deepcopy(fixture[2]),
        range_sweep_rows=copy.deepcopy(fixture[3]),
        dataset_id="synthetic",
        bootstrap_iterations=kwargs.get("iterations", 200),
        bootstrap_seed=kwargs.get("seed", 17),
    )


def test_contract_hash_excludes_self_reference() -> None:
    contract = safety_contract()
    expected = contract.pop("contract_sha256")
    assert canonical_json_sha256(contract) == expected
    assert contract["catastrophic_tail"]["included_in_combined_risk_R"] is False


def test_v231_preserves_v23_point_estimate_metrics() -> None:
    fixture = _fixture()
    old = analyze_v23(
        summary=copy.deepcopy(fixture[0]),
        gradient_tail_rows=copy.deepcopy(fixture[1]),
        cumulative_null_rows=copy.deepcopy(fixture[2]),
        range_sweep_rows=copy.deepcopy(fixture[3]),
        dataset_id="synthetic",
        bootstrap_iterations=40,
        bootstrap_seed=17,
    )
    new = _analyze(fixture, iterations=40)
    old_rows = {row["candidate"]: row for row in old["score_rows"]}
    new_rows = {row["candidate"]: row for row in new["score_rows"]}
    for candidate in old_rows:
        for key in (
            "local_risk_L",
            "trajectory_risk_T",
            "combined_risk_R",
            "display_score_S",
        ):
            assert new_rows[candidate][key] == pytest.approx(old_rows[candidate][key])


def test_catastrophic_tail_uses_all_bins_but_does_not_change_R() -> None:
    plain = _analyze(_fixture(rare_upper_catastrophe=False), iterations=60)
    rare = _analyze(_fixture(rare_upper_catastrophe=True), iterations=60)
    plain_upper = next(
        row for row in plain["score_rows"] if row["candidate"] == "mul_3.450"
    )
    rare_upper = next(
        row for row in rare["score_rows"] if row["candidate"] == "mul_3.450"
    )
    assert rare_upper["worst_timestep_bin"] == 1
    assert rare_upper["catastrophic_q99_timestep_bin"] == 0
    assert rare_upper["catastrophic_q99_d"] > plain_upper["catastrophic_q99_d"]
    assert rare_upper["catastrophic_max_d"] == pytest.approx(20.0)
    assert rare_upper["catastrophic_tail_included_in_R"] is False
    assert rare_upper["combined_risk_R"] == pytest.approx(
        plain_upper["combined_risk_R"]
    )


def test_dominant_candidate_resolves_ranking() -> None:
    result = _analyze(
        _fixture(
            lower_local=1.8,
            lower_drift=1.7,
            upper_local=0.2,
            upper_drift=0.2,
        ),
        iterations=300,
    )
    assert result["summary"]["repeat_pairing_complete"] is True
    assert result["summary"]["paired_repeat_ids"] == [0, 1, 2, 3, 4]
    assert result["summary"]["ranking_status"] == "resolved"
    assert result["summary"]["numerical_safety_preferred_mul"] == pytest.approx(3.45)
    assert result["summary"]["bootstrap_modal_best_probability"] >= 0.75
    assert sum(
        result["summary"]["bootstrap_best_probability_by_candidate"].values()
    ) == pytest.approx(1.0)


def test_identical_candidates_are_indistinguishable() -> None:
    result = _analyze(_fixture(identical=True), iterations=120)
    probabilities = result["summary"][
        "bootstrap_best_probability_by_candidate"
    ]
    assert probabilities["mul_2.700"] == pytest.approx(0.5)
    assert probabilities["mul_3.450"] == pytest.approx(0.5)
    assert result["summary"]["ranking_status"] == "indistinguishable"
    assert result["summary"]["numerical_safety_preferred_mul"] is None


def test_hard_safety_candidate_is_excluded_from_ranking() -> None:
    result = _analyze(
        _fixture(
            lower_local=0.8,
            lower_drift=0.8,
            upper_local=0.1,
            upper_drift=0.1,
            force_upper_abort=True,
        ),
        iterations=80,
    )
    rows = {row["candidate"]: row for row in result["score_rows"]}
    assert rows["mul_3.450"]["classification"] == "unsafe"
    assert rows["mul_3.450"]["bootstrap_best_probability"] is None
    assert result["summary"]["ranking_status"] == "insufficient_competitors"
    assert result["summary"]["numerical_safety_preferred_mul"] is None


def test_shared_bootstrap_is_deterministic() -> None:
    fixture = _fixture(lower_local=0.7, upper_local=0.69)
    first = _analyze(fixture, iterations=100, seed=29)
    second = _analyze(fixture, iterations=100, seed=29)
    assert first["score_rows"] == second["score_rows"]
    assert first["bootstrap_rows"] == second["bootstrap_rows"]
    assert first["ranking_rows"] == second["ranking_rows"]
