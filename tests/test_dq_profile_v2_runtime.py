from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from dq_profile.geometry import SourceGroupMap
from dq_profile.metrics import ExactGradient
from dq_profile.protocol import CandidateDefinition
from dq_profile.quant_context import ProfileQuantContext
from dq_profile.replay import ReplayBatch, ReplaySequence, seed_step_rng
from dq_profile.trainer_runtime import DiagnosticProfileRuntime
from dq_profile.v2_runtime import BranchExecution, V2ExperimentRunner


def test_quant_mechanism_outputs_are_exact_components_and_keep_ste_gradient():
    x = torch.tensor([-2.0, -0.2, 0.8, 2.0], requires_grad=True)
    x_clamped = torch.tensor([-1.0, -0.2, 0.8, 1.0])
    q_value = x + (torch.tensor([-1.0, 0.0, 1.0, 1.0]) - x).detach()
    context = ProfileQuantContext(39)

    expected = {
        "full": torch.tensor([-1.0, 0.0, 1.0, 1.0]),
        "clip_only": torch.tensor([-1.0, -0.2, 0.8, 1.0]),
        "round_only": torch.tensor([-2.0, 0.0, 1.0, 2.0]),
    }
    for mechanism, target in expected.items():
        context.begin_pass(
            mode="candidate",
            phase="test",
            probe_or_step=0,
            mechanism=mechanism,
        )
        output = context.apply_mechanism(x, q_value, x_clamped)
        assert torch.equal(output.detach(), target)
        x.grad = None
        output.sum().backward(retain_graph=True)
        assert torch.equal(x.grad, torch.ones_like(x))
        context.finish_pass()


def test_model_rng_repeat_is_reproducible_and_separated():
    first_seed = seed_step_rng(39, 7, phase="v2_training_model", repeat=1)
    first = torch.rand(4)
    again_seed = seed_step_rng(39, 7, phase="v2_training_model", repeat=1)
    again = torch.rand(4)
    other_seed = seed_step_rng(39, 7, phase="v2_training_model", repeat=2)
    other = torch.rand(4)
    assert first_seed == again_seed
    assert torch.equal(first, again)
    assert other_seed != first_seed
    assert not torch.equal(first, other)


def test_structural_model_seed_accepts_stable_string_identity():
    identity = "image:0|bin:2|noise:1|key:D:/images/a.png"
    first = seed_step_rng(39, identity, phase="v2_tail_structural_model", repeat=0)
    second = seed_step_rng(39, identity, phase="v2_tail_structural_model", repeat=0)
    other = seed_step_rng(
        39,
        "image:1|bin:2|noise:1|key:D:/images/b.png",
        phase="v2_tail_structural_model",
        repeat=0,
    )
    assert first == second
    assert first != other


@pytest.mark.parametrize("protocol", ["v23-safety-local", "v23-safety-formal"])
def test_safety_protocols_dispatch_to_lightweight_tail_probe(protocol):
    runtime = object.__new__(DiagnosticProfileRuntime)
    runtime.profile_protocol = protocol
    marker = ([], [], [], {}, [], {"path": "tail"})
    runtime._run_tail_probes = lambda **_: marker
    result = runtime._run_counterfactual_probes(
        sequence=None,
        snapshot=None,
        accelerator=None,
        network=None,
        optimizer=None,
        lr_scheduler=None,
        grad_norm_guardian=None,
        unet=None,
        text_encoders=[],
        tokenizers=[],
        train_unet=True,
        train_text_encoder=False,
        training_model=None,
        on_step_start=None,
        weight_dtype=torch.float32,
        noise_scheduler=None,
    )
    assert result is marker


def test_source_group_map_supports_exact_and_longest_prefix(tmp_path):
    source = tmp_path / "sources.json"
    source.write_text(
        json.dumps(
            [
                {"pattern": "D:/images/", "source_group": "all", "match": "prefix"},
                {"pattern": "D:/images/a", "source_group": "source-a", "match": "prefix"},
                {"image_key": "D:/images/b/crop2.png", "source_group": "source-b", "match": "exact"},
            ]
        ),
        encoding="utf-8",
    )
    mapping = SourceGroupMap.load(source)
    assert mapping.resolve(r"D:\images\a\crop1.png") == "source-a"
    assert mapping.resolve(r"D:\images\b\crop2.png") == "source-b"
    assert mapping.resolve(r"D:\images\c\crop3.png") == "all"


class _FakeAccelerator:
    device = torch.device("cpu")
    scaler = None

    @staticmethod
    def unwrap_model(network):
        return network


class _FakeScheduler:
    def __init__(self):
        self.steps = 0

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


class _FakeOptimizer:
    param_groups = []

    @staticmethod
    def state_dict():
        return {"state": {}, "param_groups": []}


class _FakeSnapshot:
    def __init__(self, network, scheduler):
        self.metadata = {"global_step": 10, "epoch": 2}
        self.network_state = {name: value.detach().clone() for name, value in network.state_dict().items()}
        self.network_runtime = {"requires_grad": {name: value.requires_grad for name, value in network.named_parameters()}}
        self._scheduler = scheduler

    def restore(self, *, network, **_):
        network.load_state_dict(self.network_state)
        self._scheduler.steps = 0


class _FakeRuntime:
    def __init__(self, network, scheduler):
        self.args = SimpleNamespace(max_train_steps=100, v_parameterization=False)
        self.protocol_seed = 39
        self.trainer = object()
        self.profile_protocol = "v2-core"
        self.network = network
        self.scheduler = scheduler

    def _run_pass(self, *, candidate, forced_skip, repeat, **_):
        step = int(_["probe_or_step"])
        native_would_skip = (step % 3 == 0) if candidate.name == "no_quant" else (step % 2 == 0)
        skipped = native_would_skip if forced_skip is None else bool(forced_skip)
        if not skipped:
            with torch.no_grad():
                increment = torch.tensor([[1.0, 0.0]])
                if candidate.quantized:
                    increment = torch.tensor([[1.0, 0.1]])
                self.network.weight.add_(increment)
            self.scheduler.steps += 1
        gradient = ExactGradient.capture(self.network.named_parameters())
        row = {
            "candidate": candidate.name,
            "repeat": repeat,
            "loss": 1.0,
            "gradient_norm": 1.0,
            "range_mul": candidate.initial_range_mul,
            "update_skipped": skipped,
            "native_would_skip": native_would_skip,
            "common_skip_matched": None if forced_skip is None else True,
            "forced_safety_abort": False,
            "invalid_reason": None,
            "optimizer_step_performed": not skipped,
            "mechanism": candidate.mechanism,
        }
        return row, gradient, []


def _materialized_sequence(count=64):
    sequence = ReplaySequence()
    for index in range(count):
        tensor = torch.zeros(1, 1)
        sequence.append(
            ReplayBatch(
                index=index,
                source_epoch=0,
                source_step=index,
                global_step=index,
                batch={"image_keys": [f"image-{index}"]},
                latents=tensor,
                noise=tensor,
                noisy_latents=tensor,
                timesteps=torch.zeros(1, dtype=torch.long),
                target=tensor,
                huber_c=None,
            )
        )
    sequence.seal()
    return sequence


def test_common_skip_matches_no_quant_updates_and_scheduler():
    network = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        network.weight.zero_()
    scheduler = _FakeScheduler()
    runtime = _FakeRuntime(network, scheduler)
    snapshot = _FakeSnapshot(network, scheduler)
    sequence = _materialized_sequence()
    runner = V2ExperimentRunner(
        runtime=runtime,
        sequence=sequence,
        snapshot=snapshot,
        pass_context={
            "accelerator": _FakeAccelerator(),
            "network": network,
            "optimizer": _FakeOptimizer(),
            "lr_scheduler": scheduler,
            "grad_norm_guardian": None,
            "noise_scheduler": object(),
            "weight_dtype": torch.float32,
        },
    )
    runner._repeat_sequences[0] = sequence
    reference = runner._reference(0, 64)
    reference_updates = sum(not value for value in reference.skip_mask)
    candidate = CandidateDefinition("mul_3.150", True, None, None, 3.15, False)
    execution = runner._execute(
        candidate=candidate,
        repeat=0,
        guardian_mode="common_skip",
        max_steps=64,
        reference=reference,
    )
    assert execution.skip_mask == reference.skip_mask
    assert scheduler.steps == reference_updates
    assert any(
        row["native_would_skip"] != row["update_skipped"]
        for row in execution.rows
    )
    assert all(row["common_skip_matched"] for row in execution.rows)
    assert set(execution.deltas) == {32, 64}


def test_mechanism_runs_every_core_gate_approved_range_without_forcing_one_winner():
    network = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        network.weight.zero_()
    scheduler = _FakeScheduler()
    runtime = _FakeRuntime(network, scheduler)
    runtime.profile_protocol = "v2-mechanism"
    snapshot = _FakeSnapshot(network, scheduler)
    sequence = _materialized_sequence()
    runner = V2ExperimentRunner(
        runtime=runtime,
        sequence=sequence,
        snapshot=snapshot,
        pass_context={
            "accelerator": _FakeAccelerator(),
            "network": network,
            "optimizer": _FakeOptimizer(),
            "lr_scheduler": scheduler,
            "grad_norm_guardian": None,
            "noise_scheduler": object(),
            "weight_dtype": torch.float32,
        },
    )
    runner._repeat_sequences[0] = sequence
    rows, trajectory, summary = runner._mechanism(
        selected_muls=(3.0, 3.15),
        repeats=(0,),
        max_steps=64,
    )
    interactions = [
        row
        for row in rows
        if row.get("mechanism") == "interaction"
        and row.get("module_group") == "all"
        and row.get("checkpoint") == 64
    ]
    assert summary["valid"] is True
    assert summary["selected_muls"] == [3.0, 3.15]
    assert {row["range_mul"] for row in interactions} == {3.0, 3.15}
    assert {row["mechanism"] for row in trajectory} == {"full", "clip_only", "round_only"}


def test_prefix_smoke_keeps_64a_64b_and_128_cohorts_distinct_and_prefix_equal():
    network = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        network.weight.zero_()
    scheduler = _FakeScheduler()
    runtime = _FakeRuntime(network, scheduler)
    runtime.profile_protocol = "v2-prefix-smoke"
    snapshot = _FakeSnapshot(network, scheduler)
    sequence = _materialized_sequence(128)
    runner = V2ExperimentRunner(
        runtime=runtime,
        sequence=sequence,
        snapshot=snapshot,
        pass_context={
            "accelerator": _FakeAccelerator(),
            "network": network,
            "optimizer": _FakeOptimizer(),
            "lr_scheduler": scheduler,
            "grad_norm_guardian": None,
            "noise_scheduler": object(),
            "weight_dtype": torch.float32,
        },
    )
    runner._repeat_sequences[0] = sequence

    result = runner.run()

    assert result["calibration_gate"]["gate"] == "pass_exact"
    assert result["calibration_gate"]["passed"] is True
    manifests = result["execution_manifest_rows"]
    assert len(manifests) == 6
    assert len({row["execution_id"] for row in manifests}) == 6
    assert {row["requested_max_steps"] for row in manifests} == {64, 128}
    assert {row["cohort_id"].split(".")[1] for row in manifests} == {"64A", "64B", "128"}
    assert all(row["phase"] == "v2_prefix_smoke" for row in manifests)
    assert len(runner._executions) == 6
    assert all(row["phase"] == "v2_prefix_smoke" for row in result["update_direction_rows"])


def test_range_sweep_uses_execution_id_when_same_candidate_exists_in_two_cohorts():
    candidate = CandidateDefinition("mul_3.150", True, None, None, 3.15, False)
    short = BranchExecution(
        candidate,
        0,
        "common_skip",
        64,
        execution_id="short",
        cohort_id="core64",
        rows=[{"loss": 1.0, "gradient_norm": 2.0, "optimizer_step_performed": True}],
    )
    long = BranchExecution(
        candidate,
        0,
        "common_skip",
        128,
        execution_id="long",
        cohort_id="extension128",
        rows=[{"loss": 9.0, "gradient_norm": 8.0, "optimizer_step_performed": True}],
    )
    rows = V2ExperimentRunner._range_sweep_rows(
        [short, long],
        [
            {
                "execution_id": "short",
                "candidate": candidate.name,
                "repeat": 0,
                "guardian_mode": "common_skip",
                "module_group": "all",
                "checkpoint": 1,
            },
            {
                "execution_id": "long",
                "candidate": candidate.name,
                "repeat": 0,
                "guardian_mode": "common_skip",
                "module_group": "all",
                "checkpoint": 1,
            },
        ],
    )
    assert [row["loss_mean_to_checkpoint"] for row in rows] == [1.0, 9.0]
