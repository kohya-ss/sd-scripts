from __future__ import annotations

import math

import numpy as np
import torch

from dq_profile import (
    METRIC_DEFINITION_VERSION,
    PREFIX_GATE_METRIC_VERSION,
    RUNTIME_METRIC_DEFINITION_VERSION,
    RUNTIME_PROTOCOL_VERSION,
    RUNTIME_SCHEMA_VERSION,
    SCHEMA_VERSION,
)
from dq_profile.protocol import CandidateDefinition
from dq_profile.quant_context import ProfileQuantContext
from dq_profile.v2_calibration import (
    aggregate_prefix_gate,
    bootstrap_tail_winner,
    evaluate_prefix_pair,
    intrinsic_noise_rows,
    relative_gradient_distance,
)
from dq_profile.v2_metrics import ParameterDelta
from dq_profile.v2_runtime import BranchExecution, V2ExperimentRunner


def _candidate(name: str = "mul_3.150", value: float = 3.15) -> CandidateDefinition:
    return CandidateDefinition(name, True, None, None, value, False)


def _state(value: float, fingerprint: str) -> dict:
    tensor = torch.tensor([value], dtype=torch.float32)
    return {
        "fingerprints": {
            "network": fingerprint,
            "optimizer": fingerprint,
            "scheduler": "same",
            "scaler": "same",
            "guardian": "same",
            "trainer": "same",
        },
        "numeric": {
            "network": {"w": tensor},
            "optimizer": {"state": tensor},
        },
    }


def _trace_row(step: int) -> dict:
    return {
        "loss": 1.0 + step * 0.01,
        "gradient_norm": 2.0 + step * 0.01,
        "gradient_hash": f"g-{step}",
        "replay_digest": f"r-{step}",
        "noise_digest": f"n-{step}",
        "timestep_digest": f"t-{step}",
        "dropout_mask_digest": f"d-{step}",
        "quant_rng_digest": f"q-{step}",
        "rng_digest_before": f"rb-{step}",
        "rng_digest_after": f"ra-{step}",
        "module_invocation_count": 4,
        "module_invocation_digest": "modules",
        "update_skipped": False,
        "optimizer_step_performed": True,
        "lr_before": [1e-4],
        "lr_after": [1e-4],
    }


def test_schema_and_metric_definition_are_v21():
    assert SCHEMA_VERSION == "2.1.0"
    assert RUNTIME_SCHEMA_VERSION == SCHEMA_VERSION
    assert RUNTIME_METRIC_DEFINITION_VERSION == METRIC_DEFINITION_VERSION
    assert PREFIX_GATE_METRIC_VERSION == METRIC_DEFINITION_VERSION
    assert RUNTIME_PROTOCOL_VERSION == "sdxl-dq-profile-v2.1"
    assert METRIC_DEFINITION_VERSION == "2.1.0"


def test_reference_mismatch_is_invalid_instead_of_comparing_deltas():
    delta = ParameterDelta({"w": torch.tensor([1.0])})
    reference = BranchExecution(
        _candidate("no_quant", 0.0),
        0,
        "native_guardian",
        64,
        execution_id="reference",
        cohort_id="core64",
        phase="v2_core",
        deltas={64: delta},
    )
    execution = BranchExecution(
        _candidate(),
        0,
        "common_skip",
        64,
        execution_id="candidate",
        cohort_id="extension128",
        phase="v2_core",
        reference_execution_id="reference",
        deltas={64: delta},
    )
    rows = V2ExperimentRunner._direction_rows(
        execution,
        reference,
        guardian_mode="common_skip",
    )
    assert len(rows) == 1
    assert rows[0]["update_direction_valid"] is False
    assert rows[0]["invalid_reason"] == "reference_mismatch"
    assert rows[0]["execution_id"] == "candidate"
    assert rows[0]["reference_execution_id"] == "reference"
    sweep_rows = V2ExperimentRunner._range_sweep_rows([execution], rows)
    assert sweep_rows[0]["invalid_reason"] == "reference_mismatch"
    assert sweep_rows[0]["loss_mean_to_checkpoint"] is None


def test_matching_direction_rows_include_full_provenance():
    delta = ParameterDelta({"w": torch.tensor([1.0, 0.0])})
    reference = BranchExecution(
        _candidate("no_quant", 0.0),
        1,
        "native_guardian",
        128,
        execution_id="reference",
        cohort_id="tail-r1",
        phase="tail",
        deltas={128: delta},
    )
    execution = BranchExecution(
        _candidate(),
        1,
        "common_skip",
        128,
        execution_id="candidate",
        cohort_id="tail-r1",
        phase="tail",
        reference_execution_id="reference",
        deltas={128: delta},
    )
    rows = V2ExperimentRunner._direction_rows(
        execution,
        reference,
        guardian_mode="common_skip",
    )
    row = next(item for item in rows if item["module_group"] == "all")
    assert row["update_direction_valid"] is True
    assert row["requested_max_steps"] == 128
    assert row["actual_checkpoint"] == 128
    assert row["cohort_id"] == "tail-r1"


def test_prefix_pair_exact_numeric_and_control_failure():
    traces = [_trace_row(step) for step in range(2)]
    states = {
        checkpoint: _state(float(checkpoint + 1), "exact")
        for checkpoint in (0, 1, 32, 64)
    }
    rows, exact = evaluate_prefix_pair(
        reference_rows=traces,
        candidate_rows=[dict(row) for row in traces],
        reference_states=states,
        candidate_states=states,
        comparison="64A_vs_64B",
        candidate_name="no_quant",
    )
    assert exact["status"] == "pass_exact"
    assert all(row["exact"] for row in rows)

    numeric_states = {
        checkpoint: _state(float(checkpoint + 1) + 5e-8, "different")
        for checkpoint in (0, 1, 32, 64)
    }
    _, numeric = evaluate_prefix_pair(
        reference_rows=traces,
        candidate_rows=[dict(row) for row in traces],
        reference_states=states,
        candidate_states=numeric_states,
        comparison="64A_vs_128_at64",
        candidate_name="no_quant",
    )
    assert numeric["status"] == "pass_numeric"

    missing_state = dict(states)
    missing_state.pop(32)
    _, missing = evaluate_prefix_pair(
        reference_rows=traces,
        candidate_rows=[dict(row) for row in traces],
        reference_states=states,
        candidate_states=missing_state,
        comparison="64A_vs_64B",
        candidate_name="no_quant",
    )
    assert missing["status"] == "fail"
    assert missing["first_divergence"] == {
        "step": 32,
        "component": "state:checkpoint_presence",
    }

    changed = [dict(row) for row in traces]
    changed[0]["dropout_mask_digest"] = "different"
    _, failed = evaluate_prefix_pair(
        reference_rows=traces,
        candidate_rows=changed,
        reference_states=states,
        candidate_states=states,
        comparison="64A_vs_64B",
        candidate_name="mul_3.150",
    )
    gate = aggregate_prefix_gate([exact, numeric, failed])
    assert failed["status"] == "fail"
    assert failed["first_divergence"] == {
        "step": 1,
        "component": "control:dropout_mask_digest",
    }
    assert gate["passed"] is False


def test_dropout_and_quant_digests_cover_sites_and_invocations_without_global_rng():
    context = ProfileQuantContext(39)
    context.begin_pass(
        mode="candidate",
        phase="prefix",
        probe_or_step=3,
        repeat=0,
    )
    before = torch.get_rng_state().clone()
    context.record_module_invocation("lora.a")
    context.record_dropout_site(
        module_name="lora.a",
        kind="network_dropout",
        probability=0.3,
        shape=(1, 4, 8, 8),
    )
    context.record_dropout_site(
        module_name="lora.a",
        kind="rank_dropout",
        probability=0.2,
        shape=(1, 4),
        actual=torch.tensor([[True, False, True, True]]),
    )
    first = context.rand_for(
        torch.zeros(2, 3),
        module_name="lora.a",
        invocation=0,
    )
    after = torch.get_rng_state().clone()
    context.finish_pass()
    trace = context.last_trace
    assert torch.equal(before, after)
    assert trace["dropout_site_count"] == 2
    assert trace["module_invocation_count"] == 1
    assert trace["quant_rng_call_count"] == 1

    context.begin_pass(
        mode="candidate",
        phase="prefix",
        probe_or_step=3,
        repeat=0,
    )
    context.record_module_invocation("lora.a")
    context.record_dropout_site(
        module_name="lora.a",
        kind="network_dropout",
        probability=0.3,
        shape=(1, 4, 8, 8),
    )
    context.record_dropout_site(
        module_name="lora.a",
        kind="rank_dropout",
        probability=0.2,
        shape=(1, 4),
        actual=torch.tensor([[True, False, True, True]]),
    )
    second = context.rand_for(
        torch.zeros(2, 3),
        module_name="lora.a",
        invocation=0,
    )
    context.finish_pass()
    assert torch.equal(first, second)
    assert context.last_trace == trace


def test_relative_gradient_distance_formula_is_exact():
    assert relative_gradient_distance(1.0, 1.0) == 0.0
    assert relative_gradient_distance(1.0, 0.0) == math.sqrt(2.0)
    assert relative_gradient_distance(2.0, 1.0) == 1.0


def _tail_samples(lower: float, upper315: float, upper345: float) -> list[dict]:
    rows = []
    for image_index in range(12):
        for noise_replica in range(2):
            for quant_repeat in range(2):
                for candidate, value in (
                    ("mul_2.700", lower),
                    ("mul_3.150", upper315),
                    ("mul_3.450", upper345),
                ):
                    rows.append(
                        {
                            "image_key": f"image-{image_index}",
                            "timestep_bin": 3,
                            "noise_replica": noise_replica,
                            "quant_repeat": quant_repeat,
                            "candidate": candidate,
                            "relative_gradient_distance": value
                            + image_index * 1e-4,
                        }
                    )
    return rows


def test_image_block_tail_bootstrap_support_contradiction_and_abstention():
    supported = bootstrap_tail_winner(
        _tail_samples(0.8, 0.4, 0.5),
        timestep_bins=4,
        iterations=200,
        seed=7,
    )
    assert supported["upper_support_probability"] == 1.0
    assert supported["upper_strata_wins"] == 4
    assert supported["decision"] == "supported_on_development_dataset"

    contradiction = bootstrap_tail_winner(
        _tail_samples(0.3, 0.7, 0.8),
        timestep_bins=4,
        iterations=200,
        seed=7,
    )
    assert contradiction["lower_support_probability"] == 1.0
    assert contradiction["decision"] == "lower_contradiction"

    abstain = bootstrap_tail_winner(
        _tail_samples(0.5, 0.5, 0.6),
        timestep_bins=4,
        iterations=200,
        seed=7,
    )
    assert abstain["tie_probability"] == 1.0
    assert abstain["decision"] == "abstain"


def test_intrinsic_noise_is_sample_cv_within_image_and_timestep():
    rows = []
    for image in ("a", "b"):
        for timestep_bin in (0, 1):
            for loss in (1.0, 2.0, 3.0, 4.0, 5.0):
                rows.append(
                    {
                        "image_key": image,
                        "timestep_bin": timestep_bin,
                        "loss": loss,
                    }
                )
    result = intrinsic_noise_rows(rows)
    expected = np.std([1, 2, 3, 4, 5], ddof=1) / 3.0
    assert len(result) == 4
    assert all(math.isclose(row["loss_cv"], expected) for row in result)



def test_dropout_digest_detects_network_rng_module_and_rank_outcomes():
    def digest(control_rng, module_outcome, rank_mask):
        context = ProfileQuantContext(39)
        context.begin_pass(
            mode="candidate",
            phase="prefix",
            probe_or_step=1,
            repeat=0,
            control_rng_digest=control_rng,
        )
        context.record_dropout_site(
            module_name="lora.a",
            kind="network_dropout",
            probability=0.3,
            shape=(1, 4, 8, 8),
        )
        context.record_dropout_site(
            module_name="lora.a",
            kind="module_dropout",
            probability=0.1,
            shape=(1,),
            actual=module_outcome,
        )
        context.record_dropout_site(
            module_name="lora.a",
            kind="rank_dropout",
            probability=0.2,
            shape=(1, 4),
            actual=torch.tensor([rank_mask], dtype=torch.bool),
        )
        context.finish_pass()
        return context.last_trace["dropout_mask_digest"]

    baseline = digest("rng-a", False, [True, False, True, True])
    assert digest("rng-b", False, [True, False, True, True]) != baseline
    assert digest("rng-a", True, [True, False, True, True]) != baseline
    assert digest("rng-a", False, [False, False, True, True]) != baseline
