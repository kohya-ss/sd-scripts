from __future__ import annotations

import copy

import pytest

from dq_profile.v23_safety import (
    analyze_profile,
    canonical_json_sha256,
    classify_risk,
    display_score,
    safety_contract,
)


def _fixture(
    *,
    upper_local: float = 0.7,
    upper_drift: float = 0.6,
    force_upper_abort: bool = False,
) -> tuple[dict, list[dict], list[dict], list[dict]]:
    summary = {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "profile": {
            "protocol": "v2-tail-calibration",
            "timestep_bins": 2,
        },
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
    for image_index in range(16):
        for timestep_bin in range(2):
            for candidate, base in (
                ("mul_2.700", 1.2 if timestep_bin == 1 else 0.8),
                (
                    "mul_3.450",
                    upper_local if timestep_bin == 1 else upper_local * 0.5,
                ),
            ):
                gradient_rows.append(
                    {
                        "record_type": "sample",
                        "image_key": f"image_{image_index:02d}",
                        "candidate": candidate,
                        "timestep_bin": timestep_bin,
                        "noise_replica": 0,
                        "quant_repeat": 0,
                        "relative_gradient_distance": base
                        + (image_index - 7.5) * 0.001,
                        "gradient_cosine": 0.9,
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
    for candidate, value in (
        ("mul_2.700", 1.3),
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
                    "orthogonal_drift": value + (repeat - 2) * 0.01,
                }
            )
    range_rows = []
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


def test_contract_hash_excludes_self_reference() -> None:
    contract = safety_contract()
    expected = contract.pop("contract_sha256")
    assert canonical_json_sha256(contract) == expected


@pytest.mark.parametrize(
    ("risk", "hard_pass", "complete", "expected"),
    [
        (0.49, True, True, "observed_tolerant"),
        (0.5, True, True, "caution"),
        (0.999, True, True, "caution"),
        (1.0, True, True, "anchor_exceeded_high_perturbation"),
        (0.1, False, True, "unsafe"),
        (0.1, True, False, "unknown"),
    ],
)
def test_classification_boundaries(
    risk: float,
    hard_pass: bool,
    complete: bool,
    expected: str,
) -> None:
    assert (
        classify_risk(
            risk,
            hard_safety_pass=hard_pass,
            evidence_complete=complete,
        )
        == expected
    )


def test_display_score_anchor() -> None:
    assert display_score(0.0) == pytest.approx(100.0)
    assert display_score(1.0) == pytest.approx(50.0)


def test_analysis_uses_worst_timestep_and_weakest_link() -> None:
    summary, gradient, cumulative, ranges = _fixture(
        upper_local=0.7,
        upper_drift=0.3,
    )
    result = analyze_profile(
        summary=summary,
        gradient_tail_rows=gradient,
        cumulative_null_rows=cumulative,
        range_sweep_rows=ranges,
        dataset_id="synthetic",
        bootstrap_iterations=40,
        bootstrap_seed=7,
    )
    rows = {row["candidate"]: row for row in result["score_rows"]}
    lower = rows["mul_2.700"]
    upper = rows["mul_3.450"]
    assert lower["worst_timestep_bin"] == 1
    assert lower["local_risk_L"] > 1.19
    assert lower["combined_risk_R"] == pytest.approx(lower["local_risk_L"])
    assert lower["classification"] == "anchor_exceeded_high_perturbation"
    assert upper["local_risk_L"] == pytest.approx(0.70675)
    assert upper["trajectory_risk_T"] < upper["local_risk_L"]
    assert upper["combined_risk_R"] == pytest.approx(upper["local_risk_L"])
    assert upper["classification"] == "caution"
    assert result["summary"]["edge_unresolved"] is True
    assert result["summary"]["edge_extension_direction"] == "upper"
    assert result["summary"]["edge_extension_recommended_muls"] == [3.75, 4.05]


def test_hard_safety_overrides_numeric_score() -> None:
    summary, gradient, cumulative, ranges = _fixture(
        upper_local=0.2,
        upper_drift=0.2,
        force_upper_abort=True,
    )
    result = analyze_profile(
        summary=summary,
        gradient_tail_rows=gradient,
        cumulative_null_rows=cumulative,
        range_sweep_rows=ranges,
        dataset_id="synthetic",
        bootstrap_iterations=20,
        bootstrap_seed=9,
    )
    upper = next(
        row for row in result["score_rows"] if row["candidate"] == "mul_3.450"
    )
    assert upper["hard_safety_pass"] is False
    assert upper["classification"] == "unsafe"
    assert upper["display_score_S"] is None
    assert "forced_safety_abort" in upper["reason_codes"]


def test_bootstrap_is_deterministic() -> None:
    fixture = _fixture(upper_local=0.4, upper_drift=0.4)
    first = analyze_profile(
        summary=copy.deepcopy(fixture[0]),
        gradient_tail_rows=copy.deepcopy(fixture[1]),
        cumulative_null_rows=copy.deepcopy(fixture[2]),
        range_sweep_rows=copy.deepcopy(fixture[3]),
        dataset_id="synthetic",
        bootstrap_iterations=25,
        bootstrap_seed=11,
    )
    second = analyze_profile(
        summary=copy.deepcopy(fixture[0]),
        gradient_tail_rows=copy.deepcopy(fixture[1]),
        cumulative_null_rows=copy.deepcopy(fixture[2]),
        range_sweep_rows=copy.deepcopy(fixture[3]),
        dataset_id="synthetic",
        bootstrap_iterations=25,
        bootstrap_seed=11,
    )
    assert first["score_rows"] == second["score_rows"]
    assert first["bootstrap_rows"] == second["bootstrap_rows"]
