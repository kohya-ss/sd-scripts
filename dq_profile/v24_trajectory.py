from __future__ import annotations

"""Frozen contract for descriptive v2.4 128-step trajectory checks.

This module deliberately separates a preregistered descriptive comparison from
the normal v2.4 local candidate-reduction path.  It may carry an edge-unresolved
local result into a 128-step measurement, but it can never authorize a best-mul,
quality, utility, or training-success claim.
"""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "2.4.1-trajectory-descriptive"
METRIC_DEFINITION_VERSION = "2.4.0"
ALLOWED_CANDIDATE_ROLES = {
    "local_rejected_control",
    "retained",
    "retained_edge",
    "preregistered_control",
}


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_name(range_mul: float) -> str:
    return f"mul_{float(range_mul):.3f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _contract_digest(payload: Mapping[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("contract_sha256", None)
    return canonical_sha256(unsigned)


def _normalise_muls(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted(float(value) for value in values))
    if not 1 <= len(result) <= 3 or len(set(result)) != len(result):
        raise ValueError("descriptive trajectory requires one to three unique muls")
    if any(not math.isfinite(value) or value <= 0.0 for value in result):
        raise ValueError("descriptive trajectory muls must be finite and positive")
    return result


def _muls_equal(left: Sequence[float], right: Sequence[float]) -> bool:
    a = tuple(sorted(float(value) for value in left))
    b = tuple(sorted(float(value) for value in right))
    return len(a) == len(b) and all(
        math.isclose(x, y, rel_tol=0.0, abs_tol=1e-12)
        for x, y in zip(a, b, strict=True)
    )


def _hard_safety_by_candidate(path: Path) -> dict[str, bool]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {
        str(row.get("candidate")): str(row.get("hard_safety_pass", "")).strip().casefold()
        in {"1", "true", "yes"}
        for row in rows
    }


def build_trajectory_contract(
    *,
    local_profile_dir: Path,
    local_analysis_dir: Path,
    prefix_gate_path: Path,
    trajectory_muls: Sequence[float],
    candidate_roles: Mapping[float, str],
    purpose: str,
) -> dict[str, Any]:
    """Build an immutable, descriptive-only contract from complete local data."""

    local_profile_dir = Path(local_profile_dir).resolve()
    local_analysis_dir = Path(local_analysis_dir).resolve()
    prefix_gate_path = Path(prefix_gate_path).resolve()
    muls = _normalise_muls(trajectory_muls)
    paths = {
        "local_status": local_profile_dir / "status.json",
        "local_summary": local_profile_dir / "summary.json",
        "local_manifest": local_profile_dir / "source_manifest.json",
        "analysis_status": local_analysis_dir / "status.json",
        "analysis_summary": local_analysis_dir / "summary.json",
        "local_selection": local_analysis_dir / "local_selection.json",
        "local_acceptance": local_analysis_dir / "local_acceptance.csv",
        "prefix_gate": prefix_gate_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"trajectory contract inputs are missing: {missing}")
    if _read_json(paths["local_status"]).get("status") != "complete":
        raise ValueError("trajectory contract requires a complete local profile")
    if _read_json(paths["analysis_status"]).get("status") != "complete":
        raise ValueError("trajectory contract requires a complete local analysis")

    summary = _read_json(paths["local_summary"])
    manifest = _read_json(paths["local_manifest"])
    selection = _read_json(paths["local_selection"])
    gate = _read_json(paths["prefix_gate"])
    source_contract = str(manifest.get("source_contract", {}).get("sha256", ""))
    if str(summary.get("profile", {}).get("protocol")) != "v24-acceptance-local":
        raise ValueError("trajectory contract requires v24-acceptance-local provenance")
    if selection.get("schema_version") != "2.4.0-local-selection":
        raise ValueError("trajectory contract requires schema 2.4 local selection")
    if selection.get("selection_valid") is not True:
        raise ValueError("trajectory contract requires a valid local selection")
    if not source_contract or selection.get("source_contract_sha256") != source_contract:
        raise ValueError("local selection and profile source contracts differ")
    if (
        gate.get("passed") is not True
        or gate.get("gate") not in {"pass_exact", "pass_numeric"}
        or gate.get("source_contract_sha256") != source_contract
    ):
        raise ValueError("trajectory contract requires a matching passed prefix gate")

    local_grid = tuple(float(value) for value in selection.get("local_grid", ()))
    for value in muls:
        if not any(math.isclose(value, grid, rel_tol=0.0, abs_tol=1e-12) for grid in local_grid):
            raise ValueError(f"trajectory mul {value} was not measured by the local profile")
    hard_safety = _hard_safety_by_candidate(paths["local_acceptance"])
    failed = [candidate_name(value) for value in muls if hard_safety.get(candidate_name(value)) is not True]
    if failed:
        raise ValueError(f"trajectory candidates did not pass local hard safety: {failed}")

    role_rows: list[dict[str, Any]] = []
    for value in muls:
        role = str(candidate_roles.get(value, ""))
        if role not in ALLOWED_CANDIDATE_ROLES:
            raise ValueError(f"invalid or missing trajectory role for {value}: {role!r}")
        role_rows.append(
            {"candidate": candidate_name(value), "range_mul": value, "role": role}
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "protocol": "v24-trajectory-descriptive",
        "diagnostic_target": "descriptive_128_step_local_to_trajectory_consistency",
        "descriptive_only": True,
        "recommendation_allowed": False,
        "ranking_resolved": False,
        "not_quality_or_utility": True,
        "automatic_followup_allowed": False,
        "purpose": str(purpose),
        "source_contract_sha256": source_contract,
        "prefix_gate_path": str(prefix_gate_path),
        "prefix_gate_canonical_sha256": canonical_sha256(gate),
        "local_profile_dir": str(local_profile_dir),
        "local_analysis_dir": str(local_analysis_dir),
        "local_summary_path": str(paths["local_summary"]),
        "local_summary_sha256": sha256_file(paths["local_summary"]),
        "local_analysis_summary_path": str(paths["analysis_summary"]),
        "local_analysis_summary_sha256": sha256_file(paths["analysis_summary"]),
        "local_selection_path": str(paths["local_selection"]),
        "local_selection_sha256": sha256_file(paths["local_selection"]),
        "local_acceptance_path": str(paths["local_acceptance"]),
        "local_acceptance_sha256": sha256_file(paths["local_acceptance"]),
        "local_grid": list(local_grid),
        "local_selection_status": selection.get("selection_status"),
        "edge_unresolved": bool(selection.get("edge_unresolved")),
        "trajectory_muls": list(muls),
        "trajectory_candidates": [candidate_name(value) for value in muls],
        "candidate_roles": role_rows,
        "measurement": {
            "branch_steps": 128,
            "branch_repeats": 5,
            "guardian_mode": "common_skip",
            "training_dropout": "enabled",
            "shared_local_probe_parity_required": True,
        },
        "claims_excluded": [
            "best_final_quality_mul",
            "best_range_mul",
            "quantization_utility",
            "training_success_guarantee",
        ],
    }
    payload["contract_sha256"] = _contract_digest(payload)
    return payload


def validate_trajectory_contract(
    payload: Mapping[str, Any],
    *,
    requested_muls: Sequence[float],
    expected_source_contract_sha256: str,
    expected_prefix_gate_sha256: str,
) -> dict[str, Any]:
    """Validate a trajectory contract and every file it binds."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"trajectory contract must use {SCHEMA_VERSION}")
    if payload.get("metric_definition_version") != METRIC_DEFINITION_VERSION:
        raise ValueError("trajectory contract metric definition is not v2.4.0")
    if payload.get("protocol") != "v24-trajectory-descriptive":
        raise ValueError("trajectory contract protocol is invalid")
    if (
        payload.get("descriptive_only") is not True
        or payload.get("recommendation_allowed") is not False
        or payload.get("ranking_resolved") is not False
        or payload.get("not_quality_or_utility") is not True
        or payload.get("automatic_followup_allowed") is not False
    ):
        raise ValueError("trajectory contract must prohibit recommendation and utility claims")
    if payload.get("contract_sha256") != _contract_digest(payload):
        raise ValueError("trajectory contract hash does not match its contents")
    if payload.get("source_contract_sha256") != expected_source_contract_sha256:
        raise ValueError("trajectory and prefix source contracts differ")
    if payload.get("prefix_gate_canonical_sha256") != expected_prefix_gate_sha256:
        raise ValueError("trajectory contract is bound to a different prefix gate")

    trajectory_muls = _normalise_muls(payload.get("trajectory_muls", ()))
    if not _muls_equal(trajectory_muls, requested_muls):
        raise ValueError("requested muls do not exactly match the trajectory contract")
    expected_candidates = [candidate_name(value) for value in trajectory_muls]
    if list(payload.get("trajectory_candidates", ())) != expected_candidates:
        raise ValueError("trajectory candidate names do not match trajectory muls")
    role_rows = list(payload.get("candidate_roles", ()))
    if len(role_rows) != len(trajectory_muls):
        raise ValueError("trajectory candidate roles are incomplete")
    for value, candidate, row in zip(
        trajectory_muls, expected_candidates, role_rows, strict=True
    ):
        if (
            row.get("candidate") != candidate
            or not math.isclose(float(row.get("range_mul")), value, rel_tol=0.0, abs_tol=1e-12)
            or row.get("role") not in ALLOWED_CANDIDATE_ROLES
        ):
            raise ValueError("trajectory candidate role topology is invalid")

    bound_paths = {
        "local_summary": ("local_summary_path", "local_summary_sha256"),
        "local_analysis_summary": (
            "local_analysis_summary_path",
            "local_analysis_summary_sha256",
        ),
        "local_selection": ("local_selection_path", "local_selection_sha256"),
        "local_acceptance": ("local_acceptance_path", "local_acceptance_sha256"),
    }
    resolved: dict[str, Path] = {}
    for name, (path_key, hash_key) in bound_paths.items():
        path = Path(str(payload.get(path_key, ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != payload.get(hash_key):
            raise ValueError(f"trajectory contract bound input changed: {name}")
        resolved[name] = path

    local_summary = _read_json(resolved["local_summary"])
    local_selection = _read_json(resolved["local_selection"])
    if str(local_summary.get("profile", {}).get("protocol")) != "v24-acceptance-local":
        raise ValueError("trajectory local summary protocol is incompatible")
    if (
        local_selection.get("schema_version") != "2.4.0-local-selection"
        or local_selection.get("selection_valid") is not True
        or local_selection.get("source_contract_sha256")
        != expected_source_contract_sha256
    ):
        raise ValueError("trajectory local selection provenance is invalid")
    if bool(local_selection.get("edge_unresolved")) != bool(payload.get("edge_unresolved")):
        raise ValueError("trajectory edge-unresolved provenance changed")
    local_grid = tuple(float(value) for value in local_selection.get("local_grid", ()))
    if not _muls_equal(local_grid, payload.get("local_grid", ())):
        raise ValueError("trajectory local grid provenance changed")
    for value in trajectory_muls:
        if not any(math.isclose(value, grid, rel_tol=0.0, abs_tol=1e-12) for grid in local_grid):
            raise ValueError(f"trajectory mul {value} is absent from the bound local grid")
    hard_safety = _hard_safety_by_candidate(resolved["local_acceptance"])
    if any(hard_safety.get(candidate) is not True for candidate in expected_candidates):
        raise ValueError("trajectory includes a candidate without local hard-safety pass")

    measurement = payload.get("measurement", {})
    if (
        int(measurement.get("branch_steps", 0)) != 128
        or int(measurement.get("branch_repeats", 0)) != 5
        or measurement.get("guardian_mode") != "common_skip"
        or measurement.get("shared_local_probe_parity_required") is not True
    ):
        raise ValueError("trajectory measurement contract is not the frozen 128-step design")
    return {
        "trajectory_muls": trajectory_muls,
        "trajectory_candidates": tuple(expected_candidates),
        "candidate_roles": tuple(dict(row) for row in role_rows),
        "local_grid": local_grid,
        "local_profile_dir": str(Path(str(payload["local_profile_dir"])).resolve()),
        "local_analysis_dir": str(Path(str(payload["local_analysis_dir"])).resolve()),
        "local_summary_sha256": str(payload["local_summary_sha256"]),
        "edge_unresolved": bool(payload.get("edge_unresolved")),
        "contract_sha256": str(payload["contract_sha256"]),
    }
