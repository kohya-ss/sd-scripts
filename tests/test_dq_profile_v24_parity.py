from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from dq_profile.v232_parity import canonical_sha256, sha256_file
from dq_profile.v24_parity import check_local_extension_parity, check_local_formal_parity


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(root: Path, *, image_count: int = 13):
    local = root / "local"
    analysis = root / "analysis"
    formal = root / "formal"
    for path in (local, analysis, formal):
        _write_json(path / "status.json", {"status": "complete"})
    candidates = ["mul_2.700", "mul_3.450"]
    local_summary = {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "profile": {"protocol": "v24-acceptance-local"},
        "snapshot": {"fingerprints": {"network": "same"}},
        "candidates": [
            {"candidate": "no_quant", "initial_range_mul": None},
            *[{"candidate": name, "initial_range_mul": float(name[4:])} for name in candidates],
        ],
    }
    formal_summary = {
        **local_summary,
        "profile": {"protocol": "v24-acceptance-formal"},
    }
    _write_json(local / "summary.json", local_summary)
    _write_json(formal / "summary.json", formal_summary)
    ordered = [
        {
            "image_key": f"image-{index:02d}",
            "batch_digest": f"batch-{index}",
            "replay_digest": f"replay-{index}",
            "model_seed": index,
        }
        for index in range(image_count)
    ]
    probe = {
        "ordered_probe_contract": ordered,
        "ordered_probe_contract_sha256": canonical_sha256(ordered),
    }
    _write_json(local / "probe_manifest.json", probe)
    _write_json(formal / "probe_manifest.json", probe)
    local_rows: list[dict] = []
    formal_rows: list[dict] = []
    local_tail: list[dict] = []
    formal_tail: list[dict] = []
    for image in range(image_count):
        for timestep_bin in range(4):
            for candidate in ["no_quant", *candidates]:
                noise_count = 3 if candidate == "no_quant" else 2
                repeat_count = 1 if candidate == "no_quant" else 2
                for noise in range(noise_count):
                    for quant_repeat in range(repeat_count):
                        quant = "" if candidate == "no_quant" else quant_repeat
                        row = {
                            "candidate": candidate,
                            "image_key": f"image-{image:02d}",
                            "source_group": f"source-{image % 6:02d}",
                            "timestep_bin": timestep_bin,
                            "timestep": 100 + timestep_bin,
                            "noise_replica": noise,
                            "quant_repeat": quant,
                            "probe_regime": "structural_dropout_off",
                            "gradient_topology_matches": True,
                            "phase": "v2_tail_probe",
                            "probe_or_step": f"probe-{image}-{timestep_bin}-{noise}",
                            "repeat": quant_repeat,
                            "range_mul": "" if candidate == "no_quant" else candidate[4:],
                            "gradient_hash": f"hash-{candidate}-{image}-{timestep_bin}-{noise}-{quant}",
                            "replay_digest": f"replay-{image}",
                            "noise_digest": f"noise-{image}-{timestep_bin}-{noise}",
                            "timestep_digest": f"time-{timestep_bin}",
                            "loss": 0.1 + image * 1e-5,
                            "gradient_norm": 1.0,
                            "parameter_gradient_cosine": 1.0 if candidate == "no_quant" else 0.9,
                        }
                        local_rows.append(row)
                        formal_rows.append(dict(row))
                        if candidate != "no_quant":
                            value = 0.2 + timestep_bin * 0.01
                            tail = {
                                "record_type": "sample",
                                **{key: row[key] for key in (
                                    "candidate", "image_key", "source_group", "timestep_bin",
                                    "timestep", "noise_replica", "quant_repeat", "probe_regime",
                                    "gradient_topology_matches",
                                )},
                                "gradient_cosine": 0.98,
                                "gradient_norm_ratio": 1.1,
                                "grad_norm_noquant": 1.0,
                                "grad_norm_candidate": 1.1,
                                "grad_diff_norm": value,
                                "relative_gradient_distance": value,
                                "symmetric_gradient_distance": 2 * value / 2.1,
                                "angular_gradient_distance": 0.2,
                                "gradient_gain_distance": 0.095,
                            }
                            local_tail.append(tail)
                            formal_tail.append(dict(tail))
    _write_csv(local / "per_image.csv", local_rows)
    _write_csv(formal / "per_image.csv", formal_rows)
    _write_csv(local / "gradient_tail.csv", local_tail)
    _write_csv(formal / "gradient_tail.csv", formal_tail)
    selection = {
        "schema_version": "2.4.0-local-selection",
        "selection_valid": True,
        "source_contract_sha256": "contract",
        "selected_candidates": candidates,
        "selected_muls": [2.7, 3.45],
        "local_summary_sha256": sha256_file(local / "summary.json"),
    }
    _write_json(analysis / "local_selection.json", selection)
    _write_json(local / "source_manifest.json", {"source_contract": {"sha256": "contract"}})
    _write_json(
        formal / "source_manifest.json",
        {
            "source_contract": {"sha256": "contract"},
            "safety_local_selection": {
                "canonical_sha256": canonical_sha256(selection),
                "matched": True,
            },
        },
    )
    return local, analysis, formal


def test_v24_parity_accepts_thirteen_shared_images(tmp_path: Path) -> None:
    local, analysis, formal = _fixture(tmp_path)
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert result["gate"] == "pass_exact"
    assert result["passed"] is True
    assert result["shared_image_count"] == 13


def test_v24_parity_accepts_numeric_tolerance(tmp_path: Path) -> None:
    local, analysis, formal = _fixture(tmp_path)
    rows = list(csv.DictReader((formal / "gradient_tail.csv").open(encoding="utf-8")))
    rows[0]["relative_gradient_distance"] = str(float(rows[0]["relative_gradient_distance"]) + 1e-10)
    _write_csv(formal / "gradient_tail.csv", rows)
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert result["gate"] == "pass_numeric"
    assert result["passed"] is True


def test_v24_parity_rejects_control_divergence(tmp_path: Path) -> None:
    local, analysis, formal = _fixture(tmp_path)
    rows = list(csv.DictReader((formal / "per_image.csv").open(encoding="utf-8")))
    rows[0]["gradient_hash"] = "changed"
    _write_csv(formal / "per_image.csv", rows)
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert result["gate"] == "fail"
    assert result["passed"] is False


def test_v24_parity_requires_at_least_eight_images(tmp_path: Path) -> None:
    local, analysis, formal = _fixture(tmp_path, image_count=7)
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert result["gate"] == "fail"
    assert result["shared_image_count"] == 7


def test_v24_local_edge_extension_preserves_shared_core(tmp_path: Path) -> None:
    core, _, _ = _fixture(tmp_path)
    extension = tmp_path / "extension"
    shutil.copytree(core, extension)
    summary = json.loads((extension / "summary.json").read_text(encoding="utf-8"))
    summary["candidates"].append(
        {"candidate": "mul_3.900", "initial_range_mul": 3.90}
    )
    _write_json(extension / "summary.json", summary)
    result = check_local_extension_parity(
        core_profile_dir=core,
        extension_profile_dir=extension,
        common_muls=(2.70, 3.45),
    )
    assert result["gate"] == "pass_exact"
    assert result["passed"] is True
    rows = list(csv.DictReader((extension / "gradient_tail.csv").open(encoding="utf-8")))
    rows[0]["relative_gradient_distance"] = "9.0"
    _write_csv(extension / "gradient_tail.csv", rows)
    failed = check_local_extension_parity(
        core_profile_dir=core, extension_profile_dir=extension, common_muls=(2.70, 3.45)
    )
    assert failed["passed"] is False


def test_v24_trajectory_parity_binds_descriptive_contract(tmp_path: Path) -> None:
    local, analysis, formal = _fixture(tmp_path)
    summary = json.loads((formal / "summary.json").read_text(encoding="utf-8"))
    summary["profile"]["protocol"] = "v24-trajectory-descriptive"
    _write_json(formal / "summary.json", summary)
    contract = {
        "schema_version": "2.4.1-trajectory-descriptive",
        "source_contract_sha256": "contract",
        "local_summary_sha256": sha256_file(local / "summary.json"),
        "trajectory_candidates": ["mul_2.700", "mul_3.450"],
        "trajectory_muls": [2.70, 3.45],
        "contract_sha256": "content",
        "edge_unresolved": True,
        "descriptive_only": True,
        "recommendation_allowed": False,
        "not_quality_or_utility": True,
    }
    contract_path = tmp_path / "trajectory_contract.json"
    _write_json(contract_path, contract)
    _write_json(
        formal / "source_manifest.json",
        {
            "source_contract": {"sha256": "contract"},
            "trajectory_contract": {
                "canonical_sha256": canonical_sha256(contract),
                "content_sha256": "content",
                "recommendation_allowed": False,
                "matched": True,
            },
        },
    )
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
        trajectory_contract_path=contract_path,
    )
    assert result["gate"] == "pass_exact"
    assert result["passed"] is True