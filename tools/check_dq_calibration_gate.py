from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dq_profile.manifest import build_source_manifest
from dq_profile.protocol import canonical_sha256
from dq_profile.v2_calibration import source_contract_from_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--source-group-map")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-json")
    args = parser.parse_args()

    gate_path = Path(args.gate).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
    errors: list[str] = []
    if gate.get("schema_version") != "2.1.0":
        errors.append("gate_schema_is_not_2.1.0")
    if gate.get("metric_definition_version") != "2.1.0":
        errors.append("gate_metric_definition_is_not_2.1.0")
    if gate.get("gate") not in {"pass_exact", "pass_numeric"} or gate.get("passed") is not True:
        errors.append("prefix_gate_did_not_pass")
    expected = str(gate.get("source_contract_sha256") or "")
    if not expected:
        errors.append("source_contract_hash_missing")

    manifest_path = gate_path.parent / "source_manifest.json"
    smoke_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest_path.is_file()
        else {}
    )
    if not smoke_manifest:
        errors.append("smoke_source_manifest_missing")
    elif source_contract_from_manifest(smoke_manifest)["sha256"] != expected:
        errors.append("smoke_manifest_contract_hash_mismatch")

    additional_files = [Path(args.dataset_config).resolve()]
    if args.source_group_map:
        additional_files.append(Path(args.source_group_map).resolve())
    current_manifest, _ = build_source_manifest(
        Path(args.repo_root).resolve(),
        quant_rng_mode="stateless",
        additional_files=tuple(additional_files),
    )
    current_contract = source_contract_from_manifest(current_manifest)
    if expected and current_contract["sha256"] != expected:
        errors.append("current_source_contract_changed")

    result = {
        "schema_version": "2.1.0",
        "gate_file": str(gate_path),
        "gate_file_sha256": canonical_sha256(gate),
        "expected_source_contract_sha256": expected,
        "current_source_contract_sha256": current_contract["sha256"],
        "passed": not errors,
        "errors": errors,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        output = Path(args.output_json).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
