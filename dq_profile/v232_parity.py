from __future__ import annotations

"""Parity checks between v2.3.2 local and formal safety stages."""

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "2.3.2-local-formal-parity"
NUMERIC_RTOL = 1e-7
NUMERIC_ATOL = 1e-9


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _source_contract(manifest: Mapping[str, Any]) -> str:
    return str(manifest.get("source_contract", {}).get("sha256", ""))


def _sample_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, int | None]:
    quant_value = row.get("quant_repeat")
    quant_repeat = None if quant_value in (None, "") else int(float(quant_value))
    return (
        str(row["candidate"]),
        str(row["image_key"]),
        int(float(row["timestep_bin"])),
        int(float(row["noise_replica"])),
        quant_repeat,
    )


def _filtered_probe_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: set[str],
    image_keys: set[str],
) -> dict[tuple[str, str, int, int, int | None], Mapping[str, Any]]:
    selected: dict[tuple[str, str, int, int, int | None], Mapping[str, Any]] = {}
    for row in rows:
        candidate = str(row.get("candidate"))
        image_key = str(row.get("image_key"))
        if candidate not in candidates or image_key not in image_keys:
            continue
        if str(row.get("probe_regime")) != "structural_dropout_off":
            continue
        noise_replica = int(float(row["noise_replica"]))
        if candidate == "no_quant" and noise_replica >= 3:
            continue
        if candidate != "no_quant" and noise_replica >= 2:
            continue
        key = _sample_key(row)
        if key in selected:
            raise ValueError(f"duplicate local/formal probe key: {key}")
        selected[key] = row
    return selected


def _float_close(left: Any, right: Any) -> tuple[bool, float, float]:
    if left in (None, "") and right in (None, ""):
        return True, 0.0, 0.0
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return False, math.inf, math.inf
    if not math.isfinite(a) or not math.isfinite(b):
        return a == b, math.inf, math.inf
    absolute = abs(a - b)
    relative = absolute / max(abs(a), abs(b), 1e-30)
    return math.isclose(a, b, rel_tol=NUMERIC_RTOL, abs_tol=NUMERIC_ATOL), absolute, relative


def check_local_formal_parity(
    *,
    local_profile_dir: Path,
    local_analysis_dir: Path,
    formal_profile_dir: Path,
) -> dict[str, Any]:
    local_profile_dir = Path(local_profile_dir).resolve()
    local_analysis_dir = Path(local_analysis_dir).resolve()
    formal_profile_dir = Path(formal_profile_dir).resolve()
    required = [
        local_profile_dir / "status.json",
        local_profile_dir / "summary.json",
        local_profile_dir / "source_manifest.json",
        local_profile_dir / "probe_manifest.json",
        local_profile_dir / "per_image.csv",
        local_analysis_dir / "status.json",
        local_analysis_dir / "local_selection.json",
        formal_profile_dir / "status.json",
        formal_profile_dir / "summary.json",
        formal_profile_dir / "source_manifest.json",
        formal_profile_dir / "probe_manifest.json",
        formal_profile_dir / "per_image.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"local/formal parity inputs are missing: {missing}")

    statuses = {
        "local_profile": _json(local_profile_dir / "status.json").get("status"),
        "local_analysis": _json(local_analysis_dir / "status.json").get("status"),
        "formal_profile": _json(formal_profile_dir / "status.json").get("status"),
    }
    if set(statuses.values()) != {"complete"}:
        raise ValueError(f"local/formal parity requires complete inputs: {statuses}")

    local_summary = _json(local_profile_dir / "summary.json")
    formal_summary = _json(formal_profile_dir / "summary.json")
    selection = _json(local_analysis_dir / "local_selection.json")
    local_manifest = _json(local_profile_dir / "source_manifest.json")
    formal_manifest = _json(formal_profile_dir / "source_manifest.json")
    local_probe = _json(local_profile_dir / "probe_manifest.json")
    formal_probe = _json(formal_profile_dir / "probe_manifest.json")

    checks: list[dict[str, Any]] = []

    def exact_check(component: str, left: Any, right: Any) -> None:
        checks.append(
            {
                "component": component,
                "status": "pass_exact" if left == right else "fail",
                "local": left,
                "formal": right,
            }
        )

    exact_check(
        "runtime_schema",
        (local_summary.get("schema_version"), local_summary.get("metric_definition_version")),
        (formal_summary.get("schema_version"), formal_summary.get("metric_definition_version")),
    )
    exact_check(
        "profile_protocol_pair",
        (
            local_summary.get("profile", {}).get("protocol"),
            formal_summary.get("profile", {}).get("protocol"),
        ),
        ("v23-safety-local", "v23-safety-formal"),
    )
    local_contract = _source_contract(local_manifest)
    formal_contract = _source_contract(formal_manifest)
    exact_check("source_contract_sha256", local_contract, formal_contract)
    exact_check("selection_source_contract", selection.get("source_contract_sha256"), local_contract)
    exact_check(
        "selection_local_summary_sha256",
        selection.get("local_summary_sha256"),
        sha256_file(local_profile_dir / "summary.json"),
    )
    exact_check(
        "snapshot_fingerprints",
        local_summary.get("snapshot", {}).get("fingerprints"),
        formal_summary.get("snapshot", {}).get("fingerprints"),
    )
    exact_check(
        "first_16_probe_contract_sha256",
        local_probe.get("first_16_probe_contract_sha256"),
        formal_probe.get("first_16_probe_contract_sha256"),
    )

    selected_candidates = {str(value) for value in selection.get("selected_candidates", ())}
    formal_candidates = {
        str(row.get("candidate"))
        for row in formal_summary.get("candidates", ())
        if str(row.get("candidate")) != "no_quant"
    }
    exact_check(
        "formal_candidates_match_selection",
        sorted(selected_candidates),
        sorted(formal_candidates),
    )
    formal_selection_provenance = formal_manifest.get("safety_local_selection", {})
    exact_check(
        "formal_selection_canonical_sha256",
        formal_selection_provenance.get("canonical_sha256"),
        canonical_sha256(selection),
    )
    exact_check(
        "formal_selection_marked_matched",
        formal_selection_provenance.get("matched"),
        True,
    )

    local_contract_rows = local_probe.get("ordered_probe_contract", ())[:16]
    image_keys = {str(row.get("image_key")) for row in local_contract_rows}
    if len(image_keys) != 16:
        checks.append(
            {
                "component": "local_first_16_unique_images",
                "status": "fail",
                "count": len(image_keys),
            }
        )
    else:
        checks.append(
            {
                "component": "local_first_16_unique_images",
                "status": "pass_exact",
                "count": 16,
            }
        )

    candidates = set(selected_candidates)
    candidates.add("no_quant")
    local_rows = _filtered_probe_rows(
        _csv(local_profile_dir / "per_image.csv"),
        candidates=candidates,
        image_keys=image_keys,
    )
    formal_rows = _filtered_probe_rows(
        _csv(formal_profile_dir / "per_image.csv"),
        candidates=candidates,
        image_keys=image_keys,
    )
    exact_check("overlapping_probe_key_set", sorted(local_rows), sorted(formal_rows))

    exact_fields = (
        "candidate",
        "phase",
        "probe_or_step",
        "repeat",
        "range_mul",
        "update_skipped",
        "native_would_skip",
        "forced_safety_abort",
        "invalid_reason",
        "optimizer_step_performed",
        "mechanism",
        "gradient_hash",
        "replay_digest",
        "noise_digest",
        "timestep_digest",
        "rng_digest_before",
        "rng_digest_after",
        "dropout_mask_digest",
        "dropout_site_count",
        "quant_rng_digest",
        "quant_rng_call_count",
        "module_invocation_count",
        "module_invocation_digest",
        "image_key",
        "timestep_bin",
        "timestep",
        "noise_replica",
        "probe_regime",
        "quant_repeat",
        "gradient_topology_matches",
    )
    numeric_fields = (
        "loss",
        "gradient_norm",
        "clip_rate",
        "quant_error_rms",
        "quant_error_ratio",
        "clip_error_rms",
        "round_error_rms",
        "parameter_gradient_cosine",
    )
    first_divergence: dict[str, Any] | None = None
    max_abs = 0.0
    max_rel = 0.0
    numeric_exact = True
    numeric_close = True
    exact_controls = True
    for key in sorted(set(local_rows) & set(formal_rows)):
        left = local_rows[key]
        right = formal_rows[key]
        for field in exact_fields:
            if str(left.get(field, "")) != str(right.get(field, "")):
                exact_controls = False
                if first_divergence is None:
                    first_divergence = {
                        "key": list(key),
                        "field": field,
                        "local": left.get(field),
                        "formal": right.get(field),
                    }
        for field in numeric_fields:
            left_value = left.get(field)
            right_value = right.get(field)
            if str(left_value or "") != str(right_value or ""):
                numeric_exact = False
            close, absolute, relative = _float_close(left_value, right_value)
            max_abs = max(max_abs, absolute if math.isfinite(absolute) else 0.0)
            max_rel = max(max_rel, relative if math.isfinite(relative) else 0.0)
            if not close:
                numeric_close = False
                if first_divergence is None:
                    first_divergence = {
                        "key": list(key),
                        "field": field,
                        "local": left_value,
                        "formal": right_value,
                    }
    key_sets_equal = set(local_rows) == set(formal_rows)
    if exact_controls and numeric_exact and key_sets_equal:
        probe_status = "pass_exact"
    elif exact_controls and numeric_close and key_sets_equal:
        probe_status = "pass_numeric"
    else:
        probe_status = "fail"
    checks.append(
        {
            "component": "overlapping_raw_probe_rows",
            "status": probe_status,
            "row_count": len(local_rows),
            "numeric_rtol": NUMERIC_RTOL,
            "numeric_atol": NUMERIC_ATOL,
            "max_abs_difference": max_abs,
            "max_relative_difference": max_rel,
            "first_divergence": first_divergence,
        }
    )

    failed = [check for check in checks if check["status"] == "fail"]
    numeric = [check for check in checks if check["status"] == "pass_numeric"]
    gate = "fail" if failed else "pass_numeric" if numeric else "pass_exact"
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "passed": gate in {"pass_exact", "pass_numeric"},
        "checks": checks,
        "first_divergence": failed[0] if failed else first_divergence,
        "selected_candidates": sorted(selected_candidates),
        "source_contract_sha256": local_contract,
        "local_profile_dir": str(local_profile_dir),
        "local_analysis_dir": str(local_analysis_dir),
        "formal_profile_dir": str(formal_profile_dir),
        "safety_not_utility": True,
    }
