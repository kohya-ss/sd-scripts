from __future__ import annotations

import copy

import pytest

from dq_profile.v2_calibration import (
    angular_gradient_distance,
    gradient_gain_distance,
    symmetric_gradient_distance,
)
from dq_profile.v23_safety import canonical_json_sha256
from dq_profile.v24_acceptance import (
    CORE_GRID,
    _weighted_quantile,
    acceptance_contract,
    analyze_natural_gradient_rows,
    analyze_local_profile,
)


def _fixture(
    risk_by_mul: dict[float, float] | None = None,
    *,
    image_count: int = 13,
    source_count: int = 6,
) -> tuple[dict, list[dict]]:
    risk_by_mul = risk_by_mul or {2.70: 0.90, 3.15: 0.35, 3.45: 0.60}
    summary = {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "profile": {"protocol": "v24-acceptance-local", "timestep_bins": 4},
        "candidates": [
            {
                "candidate": "no_quant",
                "initial_range_mul": None,
                "forced_safety_abort": False,
                "invalid_reason": None,
            },
            *[
                {
                    "candidate": f"mul_{value:.3f}",
                    "initial_range_mul": value,
                    "forced_safety_abort": False,
                    "invalid_reason": None,
                }
                for value in sorted(risk_by_mul)
            ],
        ],
    }
    rows: list[dict] = []
    for image in range(image_count):
        source_group = f"source-{image % source_count:02d}"
        for timestep_bin in range(4):
            bin_scale = (0.50, 0.65, 0.80, 1.0)[timestep_bin]
            for value, risk in sorted(risk_by_mul.items()):
                for noise_replica in range(2):
                    for quant_repeat in range(2):
                        relative = risk * bin_scale + image * 1e-5
                        g0 = 1.0 + image * 1e-4
                        gm = g0 * (1.0 + 0.1 * relative)
                        diff = relative * g0
                        cosine = max(-1.0, min(1.0, 1.0 - relative * relative / 2.0))
                        rows.append(
                            {
                                "record_type": "sample",
                                "image_key": f"image-{image:02d}",
                                "source_group": source_group,
                                "candidate": f"mul_{value:.3f}",
                                "timestep_bin": timestep_bin,
                                "noise_replica": noise_replica,
                                "quant_repeat": quant_repeat,
                                "relative_gradient_distance": relative,
                                "gradient_cosine": cosine,
                                "grad_norm_noquant": g0,
                                "grad_norm_candidate": gm,
                                "grad_diff_norm": diff,
                                "symmetric_gradient_distance": 2.0 * diff / (g0 + gm),
                                "angular_gradient_distance": (2.0 * (1.0 - cosine)) ** 0.5,
                                "gradient_gain_distance": abs(__import__("math").log(gm / g0)),
                            }
                        )
    return summary, rows


def _analyze(
    fixture: tuple[dict, list[dict]],
    *,
    iterations: int = 160,
    seed: int = 2401,
):
    return analyze_local_profile(
        summary=copy.deepcopy(fixture[0]),
        gradient_tail_rows=copy.deepcopy(fixture[1]),
        dataset_id="synthetic",
        bootstrap_iterations=iterations,
        bootstrap_seed=seed,
    )


def test_contract_hash_excludes_self_reference_and_disclaims_quality() -> None:
    contract = acceptance_contract()
    digest = contract.pop("contract_sha256")
    assert canonical_json_sha256(contract) == digest
    assert contract["not_quality_or_utility"] is True
    assert contract["core_grid"] == list(CORE_GRID)
    assert contract["bootstrap"]["primary_unit"] == "source_group_cluster_equal_weight"


def test_distance_decomposition_channels_are_exact() -> None:
    assert symmetric_gradient_distance(1.0, 3.0, 2.0) == pytest.approx(1.0)
    assert angular_gradient_distance(0.5) == pytest.approx(1.0)
    assert gradient_gain_distance(1.0, 2.0) == pytest.approx(__import__("math").log(2.0))
    assert gradient_gain_distance(0.0, 2.0) == float("inf")


def test_source_equal_weight_quantile_is_not_observation_weighted() -> None:
    # The large source has ten low rows and the small source has one high row.
    # Equal source mass gives the high source half of the probability mass.
    groups = {
        "large": [{"value": 0.0} for _ in range(10)],
        "small": [{"value": 10.0}],
    }
    values = []
    weights = []
    for group in ("large", "small"):
        members = groups[group]
        values.extend(row["value"] for row in members)
        weights.extend([1.0 / len(members)] * len(members))
    assert _weighted_quantile(values, weights, 0.75) == pytest.approx(10.0)


def test_thirteen_images_six_sources_body_tail_and_decomposition() -> None:
    result = _analyze(_fixture(), iterations=120, seed=17)
    summary = result["summary"]
    assert summary["image_count"] == 13
    assert summary["source_group_count"] == 6
    assert summary["primary_bootstrap_unit"] == "source_group_cluster_equal_weight"
    rows = {row["range_mul"]: row for row in result["score_rows"]}
    for row in rows.values():
        assert row["local_tail"] >= row["local_body"]
        assert row["tail_amplification"] >= 1.0
        assert row["symmetric_body"] >= 0.0
        assert row["angle_body"] >= 0.0
        assert row["gain_body"] >= 0.0
        assert row["grad_norm_noquant_q05"] > 0.0
        assert row["trajectory_risk_T"] is None
        assert row["perturbation_gauge"] is None


def test_shared_source_bootstrap_and_robust_pareto_drop() -> None:
    result = _analyze(_fixture(), iterations=200, seed=29)
    selection = result["selection"]
    assert selection["selected_muls"] == [3.15]
    assert selection["robustly_dominated_candidates"] == [
        "mul_2.700",
        "mul_3.450",
    ]
    assert selection["selection_valid"] is True
    assert selection["edge_unresolved"] is False
    rows = {row["range_mul"]: row for row in result["score_rows"]}
    assert rows[3.15]["source_bootstrap_body_min_probability"] == pytest.approx(1.0)
    assert rows[3.15]["source_bootstrap_tail_min_probability"] == pytest.approx(1.0)


def test_equal_four_candidate_credible_set_abstains_before_formal() -> None:
    fixture = _fixture({2.70: 0.5, 3.15: 0.5, 3.45: 0.5, 3.75: 0.5})
    result = _analyze(fixture, iterations=80, seed=31)
    selection = result["selection"]
    assert selection["credible_candidate_count"] == 4
    assert selection["selection_valid"] is False
    assert selection["selection_status"] == "credible_set_too_large_expand_local_or_abstain"
    assert selection["selected_muls"] == []
    assert selection["ranking_resolved"] is False


def test_credible_endpoint_requests_outside_local_only_and_never_resolves() -> None:
    fixture = _fixture({2.70: 0.2, 3.15: 0.6, 3.45: 0.9})
    result = _analyze(fixture, iterations=100, seed=37)
    selection = result["selection"]
    assert selection["selected_muls"] == [2.70]
    assert selection["edge_unresolved"] is True
    assert selection["edge_extension_recommended"] == [2.25]
    assert selection["selection_status"] == "edge_unresolved"
    assert selection["ranking_resolved"] is False


def test_core_grid_and_edge_extension_are_separate() -> None:
    fixture = _fixture({2.25: 0.25, 2.70: 0.30, 3.15: 0.70, 3.45: 0.90})
    result = _analyze(fixture, iterations=100, seed=41)
    summary = result["summary"]
    assert summary["core_grid"] == list(CORE_GRID)
    assert summary["edge_extension"] == [2.25]
    roles = {row["range_mul"]: row["grid_role"] for row in result["score_rows"]}
    assert roles[2.25] == "edge_extension"
    assert roles[2.70] == "core_grid"
    assert summary["core_grid_envelope"]["grid"] == list(CORE_GRID)


def test_source_bootstrap_is_deterministic() -> None:
    fixture = _fixture()
    first = _analyze(fixture, iterations=60, seed=47)
    second = _analyze(fixture, iterations=60, seed=47)
    for key in (
        "summary",
        "score_rows",
        "timestep_rows",
        "bootstrap_rows",
        "regret_rows",
        "dominance_rows",
        "source_loo_rows",
        "selection",
    ):
        assert first[key] == second[key]


def test_missing_common_core_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing common core grid"):
        _analyze(_fixture({2.70: 0.2, 3.15: 0.3, 3.30: 0.4}), iterations=20)


def test_candidate_probe_topology_must_match() -> None:
    summary, rows = _fixture()
    rows.pop()
    with pytest.raises(ValueError, match="probe key mismatch"):
        _analyze((summary, rows), iterations=20)


def test_hard_safety_candidate_is_never_selected() -> None:
    summary, rows = _fixture()
    for candidate in summary["candidates"]:
        if candidate["candidate"] == "mul_3.150":
            candidate["forced_safety_abort"] = True
    result = _analyze((summary, rows), iterations=60, seed=53)
    assert 3.15 not in result["selection"]["selected_muls"]
    assert result["selection"]["best_quality_mul"] is None
    assert result["selection"]["utility"] == "unknown"


def test_no_quant_natural_gradient_baseline_uses_source_clusters() -> None:
    rows = []
    for image in range(13):
        for timestep_bin in range(4):
            for left, right in ((0, 1), (0, 2), (1, 2)):
                value = 0.1 + timestep_bin * 0.02 + image * 1e-4
                rows.append(
                    {
                        "image_key": f"image-{image:02d}",
                        "source_group": f"source-{image % 6:02d}",
                        "timestep_bin": timestep_bin,
                        "noise_replica_a": left,
                        "noise_replica_b": right,
                        "grad_norm_a": 1.0,
                        "grad_norm_b": 1.05,
                        "grad_diff_norm": value,
                        "gradient_cosine": 0.99,
                        "relative_gradient_distance_a_to_b": value,
                        "symmetric_gradient_distance": 2 * value / 2.05,
                        "angular_gradient_distance": (2 * (1 - 0.99)) ** 0.5,
                        "gradient_gain_distance": __import__("math").log(1.05),
                        "gradient_topology_matches": True,
                    }
                )
    result = analyze_natural_gradient_rows(
        rows,
        timestep_bins=4,
        bootstrap_iterations=80,
        bootstrap_seed=59,
    )
    assert result["valid"] is True
    assert result["image_count"] == 13
    assert result["source_group_count"] == 6
    assert result["pair_count"] == 13 * 4 * 3
    assert result["local_tail"] >= result["local_body"]
    assert result["selector_input"] is False
