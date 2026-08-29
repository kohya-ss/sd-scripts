from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dq_profile.metrics import CountSketch, ExactGradient, error_decomposition, gram_and_rank
from dq_profile.protocol import (
    AutoRangeController,
    calculate_dq_begin_step,
    canonical_sha256,
    default_candidates,
    deterministic_seed,
    initial_range_mul,
    inspect_dataset_config,
)
from dq_profile.quant_context import ProfileQuantContext, aggregate_shadow_rows
from dq_profile.replay import ReplayBatch, ReplaySequence, replay_digest
from dq_profile.report import ProfileArtifacts, build_report
from dq_profile.snapshot import TrainingSnapshot


def test_stateless_seed_is_repeatable_and_candidate_free():
    kwargs = dict(phase="probe", probe_or_step="7", module_name="lora_x", invocation=2, repeat=1)
    assert deterministic_seed(39, **kwargs) == deterministic_seed(39, **kwargs)
    assert deterministic_seed(39, **kwargs) != deterministic_seed(39, **{**kwargs, "module_name": "lora_y"})
    assert deterministic_seed(39, **kwargs) != deterministic_seed(39, **{**kwargs, "invocation": 3})
    assert deterministic_seed(39, **kwargs) != deterministic_seed(39, **{**kwargs, "repeat": 2})
    assert 0 <= deterministic_seed(39, **kwargs) < 2**64
    assert "candidate" not in deterministic_seed.__code__.co_varnames


def test_band_initializers_and_boundary_match_production_values():
    candidates = {candidate.name: candidate for candidate in default_candidates()}
    assert candidates["clip_rate_high"].initial_range_mul == pytest.approx(2.8781617391)
    assert candidates["clip_rate_low"].initial_range_mul == pytest.approx(3.2051331802)
    assert initial_range_mul(0.003, 0.005) == candidates["clip_rate_high"].initial_range_mul
    assert calculate_dq_begin_step(0.05, 15_200) == 760
    assert calculate_dq_begin_step(760, 15_200) == 760


def test_dataset_preflight_counts_repeats_and_cost(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    for index in range(2):
        (first / f"{index}.png").write_bytes(b"x")
    (second / "0.webp").write_bytes(b"x")
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[[datasets]]\n"
        "batch_size=1\n"
        "[[datasets.subsets]]\n"
        f"image_dir={json.dumps(str(first))}\n"
        "num_repeats=3\n"
        "[[datasets.subsets]]\n"
        f"image_dir={json.dumps(str(second))}\n"
        "num_repeats=4\n",
        encoding="utf-8",
    )
    result = inspect_dataset_config(
        config,
        max_train_epochs=10,
        max_train_steps=None,
        lr_warmup_steps=0.1,
        branch_steps_override=None,
        max_images=32,
        timestep_bins=4,
        stochastic_repeats=3,
    )
    assert result.unique_images == 3
    assert result.repeat_weighted_samples == 10
    assert result.normal_training_steps == 100
    assert result.dq_begin_step == 10
    assert result.branch_steps == 64
    assert result.full_budget_core_exceeded is True
    assert result.full_probe_replicas == 1


def test_full_profile_expands_only_probe_replicas_within_budget(tmp_path: Path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(16):
        (image_dir / f"{index}.png").write_bytes(b"x")
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[[datasets]]\n"
        "batch_size=1\n"
        "[[datasets.subsets]]\n"
        f"image_dir={json.dumps(str(image_dir))}\n"
        "num_repeats=1\n",
        encoding="utf-8",
    )
    result = inspect_dataset_config(
        config,
        max_train_epochs=None,
        max_train_steps=15_200,
        lr_warmup_steps=0.05,
        branch_steps_override=256,
        max_images=32,
        timestep_bins=4,
        stochastic_repeats=3,
    )
    assert result.estimated_standard_steps == 1_976
    assert result.full_budget_steps == 3_952
    assert result.full_probe_replicas == 5
    assert result.estimated_full_steps == 3_768
    assert result.estimated_full_steps <= result.full_budget_steps
    assert result.branch_steps == 256
    assert result.probe_images == 16


def test_error_decomposition_identity_and_exact_gradient_cosine():
    x = torch.tensor([-2.0, -0.2, 0.8, 3.0])
    x_clamped = torch.tensor([-1.0, -0.2, 0.8, 1.0])
    q = torch.tensor([-1.0, 0.0, 1.0, 1.0])
    error, clip_error, round_error = error_decomposition(x, q, x_clamped)
    torch.testing.assert_close(error, clip_error + round_error)

    left_parameter = torch.nn.Parameter(torch.ones(3))
    right_parameter = torch.nn.Parameter(torch.ones(3))
    left_parameter.grad = torch.tensor([1.0, 2.0, 3.0])
    right_parameter.grad = torch.tensor([2.0, 4.0, 6.0])
    left = ExactGradient.capture([("p", left_parameter)])
    right = ExactGradient.capture([("p", right_parameter)])
    result = left.cosine(right)
    assert result["cosine"] == pytest.approx(1.0)
    assert result["topology_matches"] is True


def test_countsketch_is_only_a_structural_projection():
    parameter = torch.nn.Parameter(torch.ones(8))
    parameter.grad = torch.arange(8, dtype=torch.float32)
    gradient = ExactGradient.capture([("p", parameter)])
    sketcher = CountSketch(width=32, seed=11)
    first = sketcher.sketch(gradient)
    second = sketcher.sketch(gradient)
    np.testing.assert_array_equal(first, second)
    result = gram_and_rank([first, second])
    assert result["gram"].shape == (2, 2)
    assert result["effective_rank"] == pytest.approx(1.0)


def test_shadow_pass_preserves_forward_and_unscales_activation_gradient():
    context = ProfileQuantContext(39)
    context.begin_pass(
        mode="shadow",
        phase="counterfactual",
        probe_or_step="image:bin",
        grad_scale=8.0,
        dropout_enabled=False,
        shadow_candidates=tuple(candidate for candidate in default_candidates() if candidate.quantized),
        shadow_repeats=3,
    )
    x = torch.tensor([[-1.1, -0.2, 0.7, 1.8]], requires_grad=True)
    output = context.attach_shadow_hook(
        x,
        module_name="lora_test",
        target="delta",
        bits=4,
        granularity="tensor",
        stat="rms",
        quant_mode="stoch",
        use_triton_scale=False,
    )
    torch.testing.assert_close(output, x)
    (output * 8.0).sum().backward()
    rows = context.finish_pass()
    assert len(rows) == 6
    assert {row["candidate"] for row in rows} == {"clip_rate_high", "clip_rate_low"}
    assert all(row["grad_rms"] == pytest.approx(1.0) for row in rows)
    assert all(
        row["signed_impact_mean"]
        == pytest.approx(row["signed_clip_impact_mean"] + row["signed_round_impact_mean"])
        for row in rows
    )
    for row in rows:
        row["probe_regime"] = "structural_dropout_off"
    aggregate = aggregate_shadow_rows(rows)
    assert all(row["repeat_count"] == 3 for row in aggregate)
    assert all(row["probe_regime"] == "structural_dropout_off" for row in aggregate)


def test_copied_lora_stateless_quant_does_not_advance_global_rng():
    from dq_profile.copied_lora import LoRAModule

    base = torch.nn.Linear(4, 4, bias=False)
    module = LoRAModule(
        "lora_unet_test",
        base,
        lora_dim=2,
        alpha=2,
        delta_q_mode="stoch",
        delta_q_bits=4,
        delta_q_range_mul=2.8,
    )
    module.apply_to()
    with torch.no_grad():
        module.lora_up.weight.fill_(0.25)
    context = ProfileQuantContext(39)
    module.dq_profile_context = context
    module.train()
    x = torch.arange(8, dtype=torch.float32).view(2, 4)
    context.begin_pass(mode="candidate", phase="branch", probe_or_step=0)
    torch.manual_seed(1234)
    state_before = torch.get_rng_state().clone()
    first = base(x)
    state_after = torch.get_rng_state().clone()
    torch.testing.assert_close(state_after, state_before)
    context.begin_pass(mode="candidate", phase="branch", probe_or_step=0)
    second = base(x)
    torch.testing.assert_close(first, second)


def _run_lora_legacy_sequence(module_class, *, copied: bool, mode: str) -> tuple[list[float], dict[str, torch.Tensor]]:
    torch.manual_seed(91)
    base = torch.nn.Linear(4, 4, bias=False)
    module = module_class(
        "lora_unet_parity",
        base,
        lora_dim=2,
        alpha=2,
        dropout=0.2,
        rank_dropout=0.1,
        delta_q_mode=mode,
        delta_q_bits=4,
        delta_q_range_mul=2.8781617390954826,
    )
    module.apply_to()
    with torch.no_grad():
        module.lora_up.weight.fill_(0.1)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    context = ProfileQuantContext(39, rng_mode="legacy") if copied else None
    if copied:
        module.dq_profile_context = context
    inputs = [torch.arange(8, dtype=torch.float32).view(2, 4) * (index + 1) / 16 for index in range(16)]
    losses = []
    torch.manual_seed(2026)
    for index, value in enumerate(inputs):
        optimizer.zero_grad(set_to_none=True)
        if copied:
            context.begin_pass(mode="candidate", phase="parity", probe_or_step=index, dropout_enabled=True)
        loss = base(value).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().item()))
    return losses, {name: tensor.detach().clone() for name, tensor in module.state_dict().items()}


@pytest.mark.parametrize("mode", ["stoch", "det"])
def test_original_and_copied_lora_have_16_step_legacy_parity_with_dropout(mode: str):
    from networks.lora import LoRAModule as OriginalLoRAModule
    from dq_profile.copied_lora import LoRAModule as CopiedLoRAModule

    original_losses, original_state = _run_lora_legacy_sequence(OriginalLoRAModule, copied=False, mode=mode)
    copied_losses, copied_state = _run_lora_legacy_sequence(CopiedLoRAModule, copied=True, mode=mode)
    assert copied_losses == pytest.approx(original_losses, rel=0.0, abs=0.0)
    assert copied_state.keys() == original_state.keys()
    for name in original_state:
        torch.testing.assert_close(copied_state[name], original_state[name], rtol=0.0, atol=0.0)


def test_auto_validity_does_not_shorten_production_warmup():
    candidate = default_candidates()[1]
    controller = AutoRangeController(candidate, every=50, ema=0.95, warmup=True)
    assert controller.warmup_updates == 40
    for step in range(2):
        controller.observe((step + 1) * 50, 0.004)
    assert controller.warmup_completed is False
    assert controller.validity()["auto_trajectory_metrics_valid"] is False
    controller.observe(150, 0.004)
    assert controller.warmup_completed is True  # production early completion: 3 in-band observations
    for step in range(3):
        controller.observe(200 + step * 50, 0.004)
    assert controller.validity()["auto_trajectory_metrics_valid"] is True


def test_replay_sequence_is_sealed_and_has_no_loader_reference():
    sequence = ReplaySequence()
    item = ReplayBatch(0, 0, 0, 10, {"image_keys": ["a"], "x": torch.arange(3)})
    sequence.append(item)
    with pytest.raises(RuntimeError):
        list(sequence)
    sequence.seal()
    assert list(sequence) == [item]
    assert sequence.manifest()[0]["digest"] == replay_digest(item.batch)
    assert sequence.manifest()[0]["batch_digest"] == replay_digest(item.batch)
    item.latents = torch.ones(1)
    item.noise = torch.ones(1)
    item.noisy_latents = torch.ones(1)
    item.timesteps = torch.ones(1, dtype=torch.long)
    item.target = torch.ones(1)
    item.refresh_digest()
    assert item.digest != item.batch_digest
    with pytest.raises(RuntimeError):
        sequence.append(item)
    assert not any("loader" in name.lower() for name in vars(sequence))


class _TrainerState:
    _te_lr_after_cfg = {"applied": False}
    _te_lr_after_resume_state = None
    _te_lr_after_resumed = False
    _te_lr_after_resume_step = None
    _te_freeze_cfg = None
    _te_frozen_param_ids: set[int] = set()
    _te_frozen_state_dict: dict[str, torch.Tensor] = {}


def test_training_snapshot_restores_model_optimizer_scheduler_rng_and_requires_grad():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    trainer = _TrainerState()
    loss = model(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    snapshot = TrainingSnapshot.capture(
        network=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        trainer=trainer,
        guardian=None,
        global_step=1,
        epoch=0,
        data_step=1,
    )
    expected_random = torch.rand(4)
    with torch.no_grad():
        model.weight.add_(10)
    model.bias.requires_grad_(False)
    optimizer.param_groups[0]["lr"] = 0.5
    snapshot.restore(
        network=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        trainer=trainer,
        guardian=None,
    )
    actual_random = torch.rand(4)
    torch.testing.assert_close(actual_random, expected_random)
    assert model.bias.requires_grad is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)


def test_branch_parameter_update_norm_is_exact_on_cpu():
    from dq_profile.trainer_runtime import _gradient_update_norm

    model = torch.nn.Linear(3, 2, bias=False)
    reference = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(2.0)
    assert _gradient_update_norm(model, reference) == pytest.approx((model.weight.numel() * 4.0) ** 0.5)


def test_known_result_controlled_comparison_is_conservative():
    from dq_profile.trainer_runtime import _evaluate_known_result_controls

    current = {
        "network_dim": 4,
        "optimizer": "AdamW8bitFast",
        "mixed_precision": "fp16",
        "save_precision": "fp16",
        "fp16_safe_norms_mode": "strict",
        "training_steps": 15200,
        "dataset_sha256": "a" * 64,
        "dq_bits": 8,
        "dq_granularity": "channel",
        "dq_stat": "rms",
        "dq_mode": "stoch",
        "dq_scope": "unet",
    }
    known = {"comparison_controlled": True, "control_differences": []}
    known.update({f"past_{key}": value for key, value in current.items()})
    accepted = _evaluate_known_result_controls(known, current)
    assert accepted["comparison_controlled_effective"] is True
    known["past_network_dim"] = 5
    rejected = _evaluate_known_result_controls(known, current)
    assert rejected["comparison_controlled_effective"] is False
    assert any("network_dim" in item for item in rejected["detected_control_differences"])


def test_report_and_known_result_are_self_contained(tmp_path: Path):
    artifacts = ProfileArtifacts(tmp_path)
    artifacts.initialize()
    artifacts.ensure_known_result()
    known = (tmp_path / "known_result.toml").read_text(encoding="utf-8")
    assert "comparison_controlled = false" in known
    assert "past_mixed_precision" in known
    known_result = artifacts.read_known_result()
    summary = {
        "schema_version": "1.0.0",
        "metric_definition_version": "1.0.0",
        "source_manifest_sha256": "a" * 64,
        "dataset": {"unique_images": 2},
        "snapshot": {"global_step": 10},
        "probe_regime": "structural_dropout_off",
        "branch_regime": "training_dropout_on",
        "known_result": known_result,
        "warnings": [],
    }
    report = build_report(summary=summary, candidate_rows=[], trajectory_rows=[], shadow_rows=[], structural_rows=[])
    assert "<!doctype html>" in report.lower()
    assert "https://" not in report and "http://" not in report
    assert "Known result (reference only)" in report
    assert "not used by this profiler to recommend" in report


def test_diagnostic_wrapper_does_not_inherit_original_trainer():
    import train_network
    from dq_profile import copied_train_network
    from dq_profile.sdxl_profile_trainer import SdxlNetworkTrainer

    assert issubclass(SdxlNetworkTrainer, copied_train_network.NetworkTrainer)
    assert not issubclass(SdxlNetworkTrainer, train_network.NetworkTrainer)


def test_source_manifest_hashes_external_additional_inputs(tmp_path: Path):
    from dq_profile.manifest import build_source_manifest

    fake_repo = tmp_path / "repo"
    fake_repo.mkdir()
    external = tmp_path / "core_gate.json"
    external.write_text('{"core_gate_passed":true}\n', encoding="utf-8")
    manifest, _ = build_source_manifest(
        fake_repo,
        quant_rng_mode="stateless",
        additional_files=(external,),
    )
    record = manifest["additional_input_files"][0]
    assert record["path"] == str(external.resolve())
    assert record["repository_relative"] is False
    assert record["sha256"] == hashlib.sha256(external.read_bytes()).hexdigest()


def test_manifest_hash_is_canonical_and_non_self_referential():
    manifest = {"b": 2, "a": {"x": 1}}
    first = canonical_sha256(manifest)
    second = canonical_sha256({"a": {"x": 1}, "b": 2})
    assert first == second
    assert len(first) == 64
