from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from dq_profile.protocol import parse_mechanism_muls
from dq_profile.v2_metrics import (
    ParameterDelta,
    aggregate_training_seed_utility,
    caption_tag_metrics,
    choose_minimax_pair,
    classify_utility_interval,
    compare_parameter_deltas,
    hard_safety_reason,
    hierarchical_geometry_variance,
    mechanism_interaction,
    sketch_agreement,
    summarize_stability,
)


def test_mechanism_range_parser_accepts_one_or_two_points_and_rejects_invalid_values():
    assert parse_mechanism_muls("3.15") == (3.15,)
    assert parse_mechanism_muls("3.30,3.15,3.30") == (3.15, 3.30)
    assert parse_mechanism_muls(None) == ()
    with pytest.raises(ValueError):
        parse_mechanism_muls("nan")


def test_exact_cumulative_update_direction_metrics():
    reference_model = torch.nn.Linear(2, 1, bias=False)
    baseline = {name: value.detach().clone() for name, value in reference_model.state_dict().items()}
    with torch.no_grad():
        reference_model.weight.add_(torch.tensor([[1.0, 0.0]]))
    no_quant = ParameterDelta.capture(reference_model, baseline)

    candidate_model = torch.nn.Linear(2, 1, bias=False)
    candidate_model.load_state_dict(baseline)
    with torch.no_grad():
        candidate_model.weight.add_(torch.tensor([[1.0, 1.0]]))
    candidate = ParameterDelta.capture(candidate_model, baseline)
    row = next(item for item in compare_parameter_deltas(no_quant, candidate) if item["module_group"] == "all")
    assert row["update_cosine"] == pytest.approx(1 / math.sqrt(2))
    assert row["projection_gain"] == pytest.approx(1.0)
    assert row["orthogonal_drift"] == pytest.approx(1.0)
    assert row["total_drift"] == pytest.approx(1.0)
    assert row["update_norm_ratio"] == pytest.approx(math.sqrt(2))


def test_parameter_delta_can_be_limited_to_snapshot_trainable_names():
    model = torch.nn.Linear(2, 1, bias=True)
    baseline = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
        model.bias.add_(10.0)
    delta = ParameterDelta.capture(model, baseline, parameter_names=("weight",))
    assert set(delta.values) == {"weight"}
    assert delta.norm() == pytest.approx(math.sqrt(2.0))


def _stability_rows(best_by_repeat=(3.0, 3.0), best32=3.0):
    rows = []
    grid = (2.85, 3.0, 3.15)
    for repeat, best in enumerate(best_by_repeat):
        for checkpoint in (32, 64):
            selected = best32 if checkpoint == 32 else best
            for mul in grid:
                distance = abs(mul - selected)
                rows.append(
                    {
                        "range_mul": mul,
                        "repeat": repeat,
                        "checkpoint": checkpoint,
                        "guardian_mode": "common_skip",
                        "module_group": "all",
                        "update_direction_valid": True,
                        "forced_safety_abort": False,
                        "orthogonal_drift": 0.1 + distance,
                        "total_drift": 0.2 + distance,
                    }
                )
    return rows


def test_stability_summary_reports_point_plateau_and_repeat_trigger():
    stable = summarize_stability(_stability_rows(), grid=(2.85, 3.0, 3.15))
    assert stable["m_dir"] == pytest.approx(3.0)
    assert stable["m_total"] == pytest.approx(3.0)
    assert stable["m_stability_diag"] == pytest.approx(3.0)
    assert stable["stability_confidence"] == "high"
    assert stable["third_repeat_required"] is False

    changed = summarize_stability(
        _stability_rows(best_by_repeat=(3.0, 3.15)), grid=(2.85, 3.0, 3.15)
    )
    assert changed["third_repeat_required"] is True
    assert any("repeat_best_changed" in reason for reason in changed["third_repeat_reasons"])


def test_stability_summary_detects_32_to_64_change():
    result = summarize_stability(
        _stability_rows(best_by_repeat=(3.15, 3.15), best32=3.0), grid=(2.85, 3.0, 3.15)
    )
    assert result["third_repeat_required"] is True
    assert "m_dir_changed_32_to_64" in result["third_repeat_reasons"]


def test_hard_safety_aborts_nonfinite_and_extreme_gradients():
    assert hard_safety_reason(loss=float("nan"), gradient_norm=1.0, matched_no_quant_gradient_norm=1.0) == "candidate_nonfinite_loss"
    assert hard_safety_reason(loss=1.0, gradient_norm=float("inf"), matched_no_quant_gradient_norm=1.0) == "candidate_nonfinite_gradient"
    assert hard_safety_reason(loss=1.0, gradient_norm=20_000.0, matched_no_quant_gradient_norm=1.0) == "candidate_gradient_explosion"
    assert hard_safety_reason(loss=1.0, gradient_norm=2.0, matched_no_quant_gradient_norm=1.0) is None


def test_mechanism_interaction_uses_no_quant_relative_effects():
    assert mechanism_interaction(no_quant=10.0, full=16.0, clip_only=12.0, round_only=13.0) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("ci_low", "ci_high", "expected"),
    [(0.56, 0.70, "positive"), (0.20, 0.44, "negative"), (0.46, 0.54, "neutral"), (0.44, 0.56, "unknown")],
)
def test_utility_uses_rope(ci_low, ci_high, expected):
    assert classify_utility_interval(0.5, ci_low, ci_high) == expected


def test_dataset_utility_requires_two_training_seeds_to_agree():
    one = aggregate_training_seed_utility({39: {"estimate": 0.65, "ci_low": 0.56, "ci_high": 0.72}})
    assert one["U_selected_protocol"] == "unknown"
    assert one["utility_confidence"] == "low"
    two = aggregate_training_seed_utility(
        {
            39: {"estimate": 0.65, "ci_low": 0.56, "ci_high": 0.72},
            40: {"estimate": 0.64, "ci_low": 0.57, "ci_high": 0.70},
        }
    )
    assert two["U_selected_protocol"] == "positive"
    assert two["utility_confidence"] == "moderate"


def test_d4_pair_uses_minimax_regret_and_fixed_tie_break():
    grid = (2.85, 3.0, 3.15)
    rows_a = [
        {"range_mul": mul, "orthogonal_drift": abs(mul - 2.85), "total_drift": abs(mul - 2.85)}
        for mul in grid
    ]
    rows_b = [
        {"range_mul": mul, "orthogonal_drift": abs(mul - 3.15), "total_drift": abs(mul - 3.15)}
        for mul in grid
    ]
    result = choose_minimax_pair(rows_a, rows_b, grid=grid)
    assert result["valid"] is True
    assert result["m_pair"] == pytest.approx(3.0)


def test_hierarchical_geometry_and_sketch_agreement_are_explicit():
    sketches = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [1.0, 0.0],
            [1.1, 0.0],
            [0.0, 1.0],
            [0.1, 1.0],
            [1.0, 1.0],
            [1.1, 1.0],
        ]
    )
    metadata = []
    for source in ("s0", "s1"):
        for image in ("i0", "i1"):
            for timestep in (0, 1):
                metadata.append(
                    {
                        "source_group": source,
                        "image_key": f"{source}_{image}",
                        "timestep_bin": timestep,
                        "probe_replica": 0,
                    }
                )
    result = hierarchical_geometry_variance(sketches, metadata)
    assert result["valid"] is True
    fractions = [value for key, value in result.items() if key.endswith("_fraction")]
    assert sum(fractions) == pytest.approx(1.0)
    agreement = sketch_agreement(sketches, sketches.copy())
    assert agreement["stable"] is True


def test_caption_metrics_distinguish_singletons_and_reusable_tags():
    result = caption_tag_metrics(
        ["character, outfit_a, unique_01", "character, outfit_b, unique_02"]
    )
    assert result["reusable_tag_fraction"] == pytest.approx(1 / 5)
    assert result["singleton_tag_fraction"] == pytest.approx(4 / 5)
    assert result["tag_cooccurrence_effective_rank"] > 0
