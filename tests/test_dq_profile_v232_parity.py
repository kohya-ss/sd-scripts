from __future__ import annotations

import csv
import json
from pathlib import Path

from dq_profile.v232_parity import canonical_sha256, check_local_formal_parity, sha256_file


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _build_tree(tmp_path: Path):
    local = tmp_path / "local"
    analysis = tmp_path / "analysis"
    formal = tmp_path / "formal"
    for path in (local, analysis, formal):
        path.mkdir()
        _write_json(path / "status.json", {"status": "complete"})
    images = [f"image-{index:02d}" for index in range(16)]
    fingerprints = {"combined": "snapshot"}
    local_summary = {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "profile": {"protocol": "v23-safety-local"},
        "snapshot": {"fingerprints": fingerprints},
        "candidates": [
            {"candidate": "no_quant"},
            {"candidate": "mul_3.150"},
            {"candidate": "mul_3.450"},
            {"candidate": "mul_3.750"},
        ],
    }
    formal_summary = {
        **local_summary,
        "profile": {"protocol": "v23-safety-formal"},
        "candidates": [
            {"candidate": "no_quant"},
            {"candidate": "mul_3.150"},
            {"candidate": "mul_3.450"},
        ],
    }
    _write_json(local / "summary.json", local_summary)
    _write_json(formal / "summary.json", formal_summary)
    _write_json(local / "source_manifest.json", {"source_contract": {"sha256": "contract"}})
    for path in (local, formal):
        _write_json(
            path / "probe_manifest.json",
            {
                "first_16_probe_contract_sha256": "probe-contract",
                "ordered_probe_contract": [{"image_key": image} for image in images],
            },
        )
    selection = {
        "selected_candidates": ["mul_3.150", "mul_3.450"],
        "source_contract_sha256": "contract",
        "local_summary_sha256": sha256_file(local / "summary.json"),
    }
    _write_json(analysis / "local_selection.json", selection)
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

    rows = []
    for image in images:
        for candidate in ("no_quant", "mul_3.150", "mul_3.450"):
            noise_count = 3 if candidate == "no_quant" else 2
            quant_values = (None,) if candidate == "no_quant" else (0, 1)
            for noise in range(noise_count):
                for quant in quant_values:
                    rows.append(
                        {
                            "candidate": candidate,
                            "phase": "v2_tail_probe",
                            "probe_or_step": f"{image}:0:{noise}",
                            "repeat": "0" if quant is None else str(quant),
                            "gradient_hash": f"gradient:{candidate}:{image}:{noise}:{quant}",
                            "replay_digest": f"replay:{image}",
                            "noise_digest": f"noise:{image}:{noise}",
                            "timestep_digest": "timestep",
                            "rng_digest_before": "rng",
                            "rng_digest_after": "rng",
                            "dropout_mask_digest": "dropout-off",
                            "quant_rng_digest": "quant" if quant is not None else "none",
                            "module_invocation_digest": "modules",
                            "image_key": image,
                            "timestep_bin": 0,
                            "timestep": 125,
                            "noise_replica": noise,
                            "probe_regime": "structural_dropout_off",
                            "quant_repeat": "" if quant is None else quant,
                            "gradient_topology_matches": True,
                            "loss": 0.1,
                            "gradient_norm": 0.2,
                            "parameter_gradient_cosine": 1.0 if quant is None else 0.9,
                        }
                    )
    _write_csv(local / "per_image.csv", rows)
    _write_csv(formal / "per_image.csv", rows)
    return local, analysis, formal


def test_local_formal_parity_passes_exact_and_detects_control_drift(tmp_path) -> None:
    local, analysis, formal = _build_tree(tmp_path)
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert result["gate"] == "pass_exact"
    assert result["passed"] is True
    json.dumps(result)

    rows = list(csv.DictReader((formal / "per_image.csv").open(encoding="utf-8")))
    rows[0]["noise_digest"] = "changed"
    _write_csv(formal / "per_image.csv", rows)
    failed = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert failed["gate"] == "fail"
    assert failed["passed"] is False


def test_local_formal_parity_allows_small_numeric_only_difference(tmp_path) -> None:
    local, analysis, formal = _build_tree(tmp_path)
    rows = list(csv.DictReader((formal / "per_image.csv").open(encoding="utf-8")))
    rows[0]["loss"] = str(float(rows[0]["loss"]) + 1e-10)
    _write_csv(formal / "per_image.csv", rows)
    result = check_local_formal_parity(
        local_profile_dir=local,
        local_analysis_dir=analysis,
        formal_profile_dir=formal,
    )
    assert result["gate"] == "pass_numeric"
    assert result["passed"] is True
