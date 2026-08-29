from __future__ import annotations

import json

import torch

from dq_profile.snapshot_parity import compare_snapshot_outputs
from dq_profile.v2_calibration import fingerprint_tree


def _write_snapshot(root, state, *, contract="contract", warmup="warmup"):
    root.mkdir()
    fingerprints = {
        name: fingerprint_tree(value) for name, value in state.items()
    }
    fingerprints["combined"] = fingerprint_tree(fingerprints)
    torch.save(state, root / "snapshot_state.pt")
    summary = {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "source_contract_sha256": contract,
        "warmup_contract_sha256": warmup,
        "snapshot": {
            "global_step": 10,
            "epoch": 1,
            "data_step": 10,
            "lr": [0.001],
            "dq_delta_begin_step": 10,
            "first_quantized_batch_fingerprint": "batch",
            "fingerprints": fingerprints,
            "state_file": "snapshot_state.pt",
        },
    }
    (root / "summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (root / "resolved_args.json").write_text(
        json.dumps(
            {
                "dq_profile_prefix_kernel_policy": {
                    "mode": "deterministic",
                    "enabled": True,
                }
            }
        ),
        encoding="utf-8",
    )


def _state(value=1.0):
    return {
        "network": {
            "weight": torch.tensor([value, 2.0], dtype=torch.float64)
        },
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.25])}}},
        "scheduler": {"last_epoch": 10},
        "scaler": {"scale": 65536.0},
        "rng": {"torch_cpu": torch.arange(8, dtype=torch.uint8)},
        "network_runtime": {"training": True},
        "trainer": {"frozen": []},
        "guardian": {"window": [1.0, 2.0]},
        "metadata": {"global_step": 10},
    }


def test_snapshot_parity_passes_exact(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_snapshot(left, _state())
    _write_snapshot(right, _state())

    result = compare_snapshot_outputs(left, right)

    assert result["gate"] == "pass_exact"
    assert result["passed"] is True
    assert result["fingerprints_exact"] is True


def test_snapshot_parity_allows_only_tiny_floating_difference(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_snapshot(left, _state())
    _write_snapshot(right, _state(1.0 + 5e-8))

    result = compare_snapshot_outputs(left, right)

    assert result["gate"] == "pass_numeric"
    assert result["passed"] is True
    assert result["fingerprints_exact"] is False
    assert result["numeric_passed"] is True


def test_snapshot_parity_fails_control_or_large_state_difference(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_snapshot(left, _state(), contract="left-contract")
    _write_snapshot(right, _state(1.01), contract="right-contract")

    result = compare_snapshot_outputs(left, right)

    assert result["gate"] == "fail"
    assert result["passed"] is False
    assert result["controls_passed"] is False
    assert result["numeric_passed"] is False
