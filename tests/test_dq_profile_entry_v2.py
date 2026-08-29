from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pytest

from dq_profile.manifest import build_source_manifest
from dq_profile.v2_calibration import source_contract_from_manifest
from dq_profile.v23_safety import canonical_json_sha256
from dq_profile.v232_local import local_selection_rule
from dq_profile.v24_acceptance import acceptance_contract
from dq_profile.v24_trajectory import build_trajectory_contract
from sdxl_dq_dataset_profile import _resolved_candidate_definitions, _validate_and_isolate


def _args(tmp_path, gate_path, *, mechanism_muls=None):
    dataset = tmp_path / "dataset.toml"
    dataset.write_text("[general]\n", encoding="utf-8")
    return argparse.Namespace(
        dataset_config=str(dataset),
        gradient_accumulation_steps=1,
        resume=None,
        network_module="networks.lora",
        network_dim=4,
        network_args=["rank_dropout=0.2"],
        dq_delta_bits=8,
        dq_delta_bits_sched=None,
        dq_delta_step=None,
        dq_delta_use_triton=True,
        dq_delta_auto_range_mul=True,
        lr_warmup_steps=0.05,
        optimizer_type="AdamW8bitFast",
        seed=39,
        fp16_safe_norms_mode="strict",
        network_dropout=0.3,
        dq_profile_protocol="v2-mechanism",
        dq_profile_output_dir=str(tmp_path / "out"),
        dq_profile_name="mechanism-output",
        dq_profile_level="standard",
        dq_profile_max_images=32,
        dq_profile_timestep_bins=4,
        dq_profile_stochastic_repeats=3,
        dq_profile_branch_steps=None,
        dq_profile_seed=None,
        dq_profile_range_muls="2.70,2.85,3.00,3.15,3.30,3.45",
        dq_profile_sweep_steps=64,
        dq_profile_branch_repeats=2,
        dq_profile_guardian_ablation="common_then_native",
        dq_profile_sketch_width=512,
        dq_profile_sketch_seeds=2,
        dq_profile_source_group_map=None,
        dq_profile_core_gate_file=str(gate_path),
        dq_profile_prefix_gate_file=None,
        dq_profile_safety_local_selection_file=None,
        dq_profile_core_profile_key="D2",
        dq_profile_mechanism_muls=mechanism_muls,
        max_data_loader_n_workers=1,
        persistent_data_loader_workers=True,
        output_dir="ignored",
        output_name="ignored",
    )


def _gate(tmp_path):
    path = tmp_path / "core_gate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "core_gate_passed": True,
                "mechanism_selected_muls": {"D2": [3.15, 3.30]},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_mechanism_entry_uses_only_core_gate_approved_ranges(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    _validate_and_isolate(args)
    assert args.network_module == "dq_profile.copied_lora"
    assert args.dq_profile_mechanism_muls_resolved == (3.15, 3.30)
    assert args.dq_profile_mechanism_selection_source == "core_gate"
    assert args.max_data_loader_n_workers == 0
    assert args.persistent_data_loader_workers is False


def test_mechanism_entry_rejects_range_not_approved_by_core_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path), mechanism_muls="3.00")
    with pytest.raises(ValueError, match="exactly match"):
        _validate_and_isolate(args)


def _prefix_gate_for_args(tmp_path, args, *, schema="2.1.0", contract_override=None):
    repo_root = Path(__file__).resolve().parents[1]
    manifest, _ = build_source_manifest(
        repo_root,
        quant_rng_mode="stateless",
        additional_files=(args.dataset_config,),
    )
    contract = source_contract_from_manifest(manifest)["sha256"]
    gate = tmp_path / f"prefix_gate_{schema}.json"
    gate.write_text(
        json.dumps(
            {
                "schema_version": schema,
                "metric_definition_version": schema,
                "gate": "pass_exact",
                "passed": True,
                "source_contract_sha256": contract_override or contract,
            }
        ),
        encoding="utf-8",
    )
    return gate


def test_tail_entry_requires_v21_gate_and_matching_source_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v2-tail-calibration"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_prefix_gate_file = str(_prefix_gate_for_args(tmp_path, args))

    _validate_and_isolate(args)

    assert args.dq_profile_range_muls_resolved == (2.70, 3.15, 3.45)
    assert args.dq_profile_sweep_steps == 128
    assert args.dq_profile_branch_repeats == 5
    assert args.dq_profile_capture_steps == 128
    assert args.dq_profile_current_source_contract_sha256 == args.dq_profile_expected_source_contract_sha256
    assert not Path(args.dq_profile_run_dir).exists()


def test_tail_entry_rejects_schema20_gate_without_creating_run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v2-tail-calibration"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_name = "tail-must-not-exist"
    args.dq_profile_prefix_gate_file = str(_prefix_gate_for_args(tmp_path, args, schema="2.0.0"))

    with pytest.raises(ValueError, match="schema/metric 2.1.0"):
        _validate_and_isolate(args)
    assert not (tmp_path / "out" / args.dq_profile_name).exists()


def test_tail_entry_rejects_changed_source_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v2-tail-calibration"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_prefix_gate_file = str(
        _prefix_gate_for_args(tmp_path, args, contract_override="0" * 64)
    )

    with pytest.raises(ValueError, match="source contract changed"):
        _validate_and_isolate(args)

def test_prefix_candidate_pair_does_not_use_core_three_point_parser():
    args = argparse.Namespace(
        dq_profile_protocol="v2-prefix-smoke",
        dq_profile_range_muls_resolved=(3.15,),
    )
    candidates = _resolved_candidate_definitions(args)
    assert [candidate.name for candidate in candidates] == ["no_quant", "mul_3.150"]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formal_selection(tmp_path, gate_path):
    local_profile = tmp_path / "local-profile"
    local_analysis = tmp_path / "local-analysis"
    local_profile.mkdir()
    local_analysis.mkdir()
    local_summary = local_profile / "summary.json"
    local_summary.write_text(
        json.dumps(
            {
                "schema_version": "2.1.0",
                "metric_definition_version": "2.1.0",
                "profile": {"protocol": "v23-safety-local"},
            }
        ),
        encoding="utf-8",
    )
    analysis_summary = local_analysis / "summary.json"
    analysis_summary.write_text(json.dumps({"schema_version": "2.3.2-local"}), encoding="utf-8")
    rule = local_selection_rule()
    rule_sha = rule.pop("sha256")
    contract = json.loads(Path(gate_path).read_text(encoding="utf-8"))["source_contract_sha256"]
    selection = local_analysis / "local_selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "2.3.2-local-selection",
                "selection_valid": True,
                "source_contract_sha256": contract,
                "local_profile_dir": str(local_profile),
                "local_summary_path": str(local_summary),
                "local_summary_sha256": _file_sha256(local_summary),
                "local_analysis_summary_path": str(analysis_summary),
                "local_analysis_summary_sha256": _file_sha256(analysis_summary),
                "selection_rule": rule,
                "selection_rule_sha256": rule_sha,
                "local_grid": [2.7, 3.15, 3.45, 3.75],
                "selected_muls": [3.15, 3.45],
                "selected_candidates": ["mul_3.150", "mul_3.450"],
            }
        ),
        encoding="utf-8",
    )
    return selection, local_summary


def test_formal_entry_accepts_two_local_selected_candidates(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v23-safety-formal"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_range_muls = "3.15,3.45"
    gate = _prefix_gate_for_args(tmp_path, args)
    selection, _ = _formal_selection(tmp_path, gate)
    args.dq_profile_prefix_gate_file = str(gate)
    args.dq_profile_safety_local_selection_file = str(selection)

    _validate_and_isolate(args)

    assert args.dq_profile_range_muls_resolved == (3.15, 3.45)
    assert args.dq_profile_branch_repeats == 5
    assert [candidate.name for candidate in _resolved_candidate_definitions(args)] == [
        "no_quant",
        "mul_3.150",
        "mul_3.450",
    ]


def test_formal_entry_rejects_tampered_local_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v23-safety-formal"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_range_muls = "3.15,3.45"
    gate = _prefix_gate_for_args(tmp_path, args)
    selection, local_summary = _formal_selection(tmp_path, gate)
    local_summary.write_text("{}", encoding="utf-8")
    args.dq_profile_prefix_gate_file = str(gate)
    args.dq_profile_safety_local_selection_file = str(selection)

    with pytest.raises(ValueError, match="valid schema 2.3.2 local selection"):
        _validate_and_isolate(args)


@pytest.mark.parametrize(
    "protocol",
    (
        "v2-prefix-smoke",
        "v2-tail-calibration",
        "v23-safety-local",
        "v23-safety-formal",
        "v24-acceptance-local",
        "v24-acceptance-formal",
        "v24-trajectory-descriptive",
    ),
)
def test_reproducibility_kernel_policy_enables_deterministic_torch(
    monkeypatch, protocol
):
    import torch

    from sdxl_dq_dataset_profile import _configure_prefix_kernel_policy

    previous = torch.are_deterministic_algorithms_enabled()
    previous_cudnn = torch.backends.cudnn.deterministic
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        args = argparse.Namespace(
            dq_profile_protocol=protocol,
            dq_profile_prefix_kernel_mode="deterministic",
        )
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
        policy = _configure_prefix_kernel_policy(args)
        assert policy["enabled"] is True
        assert policy["cublas_workspace_config"] == ":4096:8"
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.use_deterministic_algorithms(previous)
        torch.backends.cudnn.deterministic = previous_cudnn
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cuda.matmul.allow_tf32 = previous_matmul_tf32
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32


def test_snapshot_only_is_prefix_only_and_captures_one_batch(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v2-prefix-smoke"
    args.dq_profile_snapshot_only = True
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None

    _validate_and_isolate(args)

    assert args.dq_profile_capture_steps == 1
    assert args.network_module == "dq_profile.copied_lora"

    args.dq_profile_protocol = "v23-safety-local"
    with pytest.raises(ValueError, match="requires.*v2-prefix-smoke"):
        _validate_and_isolate(args)


def test_v24_formal_entry_accepts_one_pareto_selected_candidate(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v24-acceptance-formal"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_range_muls = "3.15"
    gate = _prefix_gate_for_args(tmp_path, args)
    local_profile = tmp_path / "v24-local-profile"
    local_analysis = tmp_path / "v24-local-analysis"
    local_profile.mkdir()
    local_analysis.mkdir()
    local_summary = local_profile / "summary.json"
    local_summary.write_text(
        json.dumps(
            {
                "schema_version": "2.1.0",
                "metric_definition_version": "2.1.0",
                "profile": {"protocol": "v24-acceptance-local"},
            }
        ),
        encoding="utf-8",
    )
    analysis_summary = local_analysis / "summary.json"
    analysis_summary.write_text(
        json.dumps({"schema_version": "2.4.0-local"}), encoding="utf-8"
    )
    rule = acceptance_contract()["candidate_reduction"]
    contract = json.loads(gate.read_text(encoding="utf-8"))["source_contract_sha256"]
    selection = local_analysis / "local_selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema_version": "2.4.0-local-selection",
                "selection_valid": True,
                "source_contract_sha256": contract,
                "local_profile_dir": str(local_profile),
                "local_summary_path": str(local_summary),
                "local_summary_sha256": _file_sha256(local_summary),
                "local_analysis_summary_path": str(analysis_summary),
                "local_analysis_summary_sha256": _file_sha256(analysis_summary),
                "selection_rule": rule,
                "selection_rule_sha256": canonical_json_sha256(rule),
                "local_grid": [2.7, 3.15, 3.45],
                "selected_muls": [3.15],
                "selected_candidates": ["mul_3.150"],
            }
        ),
        encoding="utf-8",
    )
    args.dq_profile_prefix_gate_file = str(gate)
    args.dq_profile_safety_local_selection_file = str(selection)
    _validate_and_isolate(args)
    assert args.dq_profile_range_muls_resolved == (3.15,)
    assert [candidate.name for candidate in _resolved_candidate_definitions(args)] == [
        "no_quant",
        "mul_3.150",
    ]


def test_v24_trajectory_entry_validates_preregistered_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    args = _args(tmp_path, _gate(tmp_path))
    args.dq_profile_protocol = "v24-trajectory-descriptive"
    args.dq_profile_core_gate_file = None
    args.dq_profile_core_profile_key = None
    args.dq_profile_mechanism_muls = None
    args.dq_profile_range_muls = "3.15,3.45,3.75"
    gate = _prefix_gate_for_args(tmp_path, args)
    source_contract = json.loads(gate.read_text(encoding="utf-8"))[
        "source_contract_sha256"
    ]

    local = tmp_path / "trajectory-local"
    analysis = tmp_path / "trajectory-analysis"
    local.mkdir()
    analysis.mkdir()
    (local / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (local / "summary.json").write_text(
        json.dumps({"profile": {"protocol": "v24-acceptance-local"}}),
        encoding="utf-8",
    )
    (local / "source_manifest.json").write_text(
        json.dumps({"source_contract": {"sha256": source_contract}}),
        encoding="utf-8",
    )
    (analysis / "status.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (analysis / "summary.json").write_text(
        json.dumps({"schema_version": "2.4.0-local"}), encoding="utf-8"
    )
    (analysis / "local_selection.json").write_text(
        json.dumps(
            {
                "schema_version": "2.4.0-local-selection",
                "selection_valid": True,
                "selection_status": "edge_unresolved",
                "source_contract_sha256": source_contract,
                "local_grid": [2.70, 3.15, 3.45, 3.75],
                "edge_unresolved": True,
            }
        ),
        encoding="utf-8",
    )
    with (analysis / "local_acceptance.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["candidate", "hard_safety_pass"]
        )
        writer.writeheader()
        for value in (2.70, 3.15, 3.45, 3.75):
            writer.writerow(
                {
                    "candidate": f"mul_{value:.3f}",
                    "hard_safety_pass": True,
                }
            )
    contract = build_trajectory_contract(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        prefix_gate_path=gate,
        trajectory_muls=(3.15, 3.45, 3.75),
        candidate_roles={
            3.15: "local_rejected_control",
            3.45: "retained",
            3.75: "retained_edge",
        },
        purpose="entry test",
    )
    contract_path = tmp_path / "trajectory_contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    args.dq_profile_prefix_gate_file = str(gate)
    args.dq_profile_trajectory_contract_file = str(contract_path)

    _validate_and_isolate(args)

    assert args.dq_profile_range_muls_resolved == (3.15, 3.45, 3.75)
    assert args.dq_profile_branch_repeats == 5
    assert args.dq_profile_trajectory_edge_unresolved is True
    assert [candidate.name for candidate in _resolved_candidate_definitions(args)] == [
        "no_quant",
        "mul_3.150",
        "mul_3.450",
        "mul_3.750",
    ]