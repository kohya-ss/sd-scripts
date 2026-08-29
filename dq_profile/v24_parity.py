from __future__ import annotations

"""Parity gate between v2.4 local and formal acceptance probes."""

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from dq_profile.v232_parity import canonical_sha256, sha256_file


SCHEMA_VERSION = "2.4.0-local-formal-parity"
NUMERIC_RTOL = 1e-7
NUMERIC_ATOL = 1e-9
MINIMUM_SHARED_IMAGES = 8


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


def _filtered_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: set[str],
    image_keys: set[str],
    require_sample_record: bool,
) -> dict[tuple[str, str, int, int, int | None], Mapping[str, Any]]:
    selected: dict[tuple[str, str, int, int, int | None], Mapping[str, Any]] = {}
    for row in rows:
        candidate = str(row.get("candidate"))
        image_key = str(row.get("image_key"))
        if candidate not in candidates or image_key not in image_keys:
            continue
        if require_sample_record and str(row.get("record_type")) != "sample":
            continue
        if row.get("probe_regime") not in (None, "", "structural_dropout_off"):
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
    return (
        math.isclose(a, b, rel_tol=NUMERIC_RTOL, abs_tol=NUMERIC_ATOL),
        absolute,
        relative,
    )


def _compare_row_maps(
    *,
    component: str,
    local_rows: Mapping[tuple[Any, ...], Mapping[str, Any]],
    formal_rows: Mapping[tuple[Any, ...], Mapping[str, Any]],
    exact_fields: Sequence[str],
    numeric_fields: Sequence[str],
) -> dict[str, Any]:
    keys_equal = set(local_rows) == set(formal_rows)
    exact_controls = True
    numeric_exact = True
    numeric_close = True
    max_abs = 0.0
    max_rel = 0.0
    first_divergence: dict[str, Any] | None = None
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
            if math.isfinite(absolute):
                max_abs = max(max_abs, absolute)
            if math.isfinite(relative):
                max_rel = max(max_rel, relative)
            if not close:
                numeric_close = False
                if first_divergence is None:
                    first_divergence = {
                        "key": list(key),
                        "field": field,
                        "local": left_value,
                        "formal": right_value,
                    }
    if keys_equal and exact_controls and numeric_exact:
        status = "pass_exact"
    elif keys_equal and exact_controls and numeric_close:
        status = "pass_numeric"
    else:
        status = "fail"
    return {
        "component": component,
        "status": status,
        "row_count": len(local_rows),
        "key_sets_equal": keys_equal,
        "numeric_rtol": NUMERIC_RTOL,
        "numeric_atol": NUMERIC_ATOL,
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "first_divergence": first_divergence,
    }


def check_local_formal_parity(
    *,
    local_profile_dir: Path,
    local_analysis_dir: Path,
    formal_profile_dir: Path,
    trajectory_contract_path: Path | None = None,
) -> dict[str, Any]:
    local_profile_dir = Path(local_profile_dir).resolve()
    local_analysis_dir = Path(local_analysis_dir).resolve()
    formal_profile_dir = Path(formal_profile_dir).resolve()
    trajectory_contract_path = (
        Path(trajectory_contract_path).resolve()
        if trajectory_contract_path is not None
        else None
    )
    required = [
        local_profile_dir / "status.json",
        local_profile_dir / "summary.json",
        local_profile_dir / "source_manifest.json",
        local_profile_dir / "probe_manifest.json",
        local_profile_dir / "per_image.csv",
        local_profile_dir / "gradient_tail.csv",
        local_analysis_dir / "status.json",
        local_analysis_dir / "local_selection.json",
        formal_profile_dir / "status.json",
        formal_profile_dir / "summary.json",
        formal_profile_dir / "source_manifest.json",
        formal_profile_dir / "probe_manifest.json",
        formal_profile_dir / "per_image.csv",
        formal_profile_dir / "gradient_tail.csv",
    ]
    if trajectory_contract_path is not None:
        required.append(trajectory_contract_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v2.4 local/formal parity inputs are missing: {missing}")
    statuses = {
        "local_profile": _json(local_profile_dir / "status.json").get("status"),
        "local_analysis": _json(local_analysis_dir / "status.json").get("status"),
        "formal_profile": _json(formal_profile_dir / "status.json").get("status"),
    }
    if set(statuses.values()) != {"complete"}:
        raise ValueError(f"v2.4 parity requires complete inputs: {statuses}")

    local_summary = _json(local_profile_dir / "summary.json")
    formal_summary = _json(formal_profile_dir / "summary.json")
    selection = _json(
        trajectory_contract_path
        if trajectory_contract_path is not None
        else local_analysis_dir / "local_selection.json"
    )
    is_trajectory = trajectory_contract_path is not None
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
        (
            "v24-acceptance-local",
            "v24-trajectory-descriptive"
            if is_trajectory
            else "v24-acceptance-formal",
        ),
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
    local_ordered = list(local_probe.get("ordered_probe_contract", ()))
    formal_ordered = list(formal_probe.get("ordered_probe_contract", ()))
    exact_check("ordered_probe_contract", local_ordered, formal_ordered)
    exact_check(
        "ordered_probe_contract_sha256",
        local_probe.get("ordered_probe_contract_sha256"),
        formal_probe.get("ordered_probe_contract_sha256"),
    )
    ordered_image_keys = [str(row.get("image_key")) for row in local_ordered]
    unique_image_keys = set(ordered_image_keys)
    checks.append(
        {
            "component": "shared_unique_probe_images",
            "status": (
                "pass_exact"
                if len(unique_image_keys) == len(ordered_image_keys) >= MINIMUM_SHARED_IMAGES
                else "fail"
            ),
            "count": len(unique_image_keys),
            "minimum": MINIMUM_SHARED_IMAGES,
        }
    )

    selected_candidates = {
        str(value)
        for value in selection.get(
            "trajectory_candidates" if is_trajectory else "selected_candidates",
            (),
        )
    }
    formal_candidates = {
        str(row.get("candidate"))
        for row in formal_summary.get("candidates", ())
        if str(row.get("candidate")) != "no_quant"
    }
    exact_check("formal_candidates_match_selection", sorted(selected_candidates), sorted(formal_candidates))
    provenance = formal_manifest.get(
        "trajectory_contract" if is_trajectory else "safety_local_selection",
        {},
    )
    exact_check("formal_selection_canonical_sha256", provenance.get("canonical_sha256"), canonical_sha256(selection))
    if is_trajectory:
        exact_check("trajectory_content_sha256", provenance.get("content_sha256"), selection.get("contract_sha256"))
        exact_check("trajectory_recommendation_prohibited", provenance.get("recommendation_allowed"), False)
    exact_check("formal_selection_marked_matched", provenance.get("matched"), True)

    candidates = set(selected_candidates)
    candidates.add("no_quant")
    local_per_image = _filtered_rows(
        _csv(local_profile_dir / "per_image.csv"),
        candidates=candidates,
        image_keys=unique_image_keys,
        require_sample_record=False,
    )
    formal_per_image = _filtered_rows(
        _csv(formal_profile_dir / "per_image.csv"),
        candidates=candidates,
        image_keys=unique_image_keys,
        require_sample_record=False,
    )
    checks.append(
        _compare_row_maps(
            component="overlapping_raw_probe_rows",
            local_rows=local_per_image,
            formal_rows=formal_per_image,
            exact_fields=(
                "candidate", "phase", "probe_or_step", "repeat", "range_mul",
                "update_skipped", "native_would_skip", "forced_safety_abort",
                "invalid_reason", "optimizer_step_performed", "mechanism",
                "gradient_hash", "replay_digest", "noise_digest", "timestep_digest",
                "rng_digest_before", "rng_digest_after", "dropout_mask_digest",
                "dropout_site_count", "quant_rng_digest", "quant_rng_call_count",
                "module_invocation_count", "module_invocation_digest", "image_key",
                "source_group", "timestep_bin", "timestep", "noise_replica",
                "probe_regime", "quant_repeat", "gradient_topology_matches",
            ),
            numeric_fields=(
                "loss", "gradient_norm", "clip_rate", "quant_error_rms",
                "quant_error_ratio", "clip_error_rms", "round_error_rms",
                "parameter_gradient_cosine",
            ),
        )
    )
    local_tail = _filtered_rows(
        _csv(local_profile_dir / "gradient_tail.csv"),
        candidates=selected_candidates,
        image_keys=unique_image_keys,
        require_sample_record=True,
    )
    formal_tail = _filtered_rows(
        _csv(formal_profile_dir / "gradient_tail.csv"),
        candidates=selected_candidates,
        image_keys=unique_image_keys,
        require_sample_record=True,
    )
    checks.append(
        _compare_row_maps(
            component="derived_gradient_distance_rows",
            local_rows=local_tail,
            formal_rows=formal_tail,
            exact_fields=(
                "candidate", "image_key", "source_group", "timestep_bin",
                "timestep", "noise_replica", "quant_repeat", "probe_regime",
                "gradient_topology_matches",
            ),
            numeric_fields=(
                "gradient_cosine", "gradient_norm_ratio", "grad_norm_noquant",
                "grad_norm_candidate", "grad_diff_norm", "relative_gradient_distance",
                "symmetric_gradient_distance", "angular_gradient_distance",
                "gradient_gain_distance",
            ),
        )
    )

    failed = [check for check in checks if check["status"] == "fail"]
    numeric = [check for check in checks if check["status"] == "pass_numeric"]
    gate = "fail" if failed else "pass_numeric" if numeric else "pass_exact"
    return {
        "schema_version": SCHEMA_VERSION,
        "gate": gate,
        "passed": gate in {"pass_exact", "pass_numeric"},
        "checks": checks,
        "first_divergence": failed[0] if failed else None,
        "selected_candidates": sorted(selected_candidates),
        "shared_image_count": len(unique_image_keys),
        "shared_ordered_image_keys": ordered_image_keys,
        "source_contract_sha256": local_contract,
        "local_profile_dir": str(local_profile_dir),
        "local_analysis_dir": str(local_analysis_dir),
        "formal_profile_dir": str(formal_profile_dir),
        "safety_not_utility": True,
    }


def check_local_extension_parity(
    *,
    core_profile_dir: Path,
    extension_profile_dir: Path,
    common_muls: Sequence[float],
) -> dict[str, Any]:
    """Verify that adding edge candidates did not alter shared local probes."""

    core_profile_dir = Path(core_profile_dir).resolve()
    extension_profile_dir = Path(extension_profile_dir).resolve()
    required = [
        root / name
        for root in (core_profile_dir, extension_profile_dir)
        for name in (
            "status.json",
            "summary.json",
            "source_manifest.json",
            "probe_manifest.json",
            "per_image.csv",
            "gradient_tail.csv",
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v2.4 local extension parity inputs are missing: {missing}")
    core_summary = _json(core_profile_dir / "summary.json")
    extension_summary = _json(extension_profile_dir / "summary.json")
    core_manifest = _json(core_profile_dir / "source_manifest.json")
    extension_manifest = _json(extension_profile_dir / "source_manifest.json")
    core_probe = _json(core_profile_dir / "probe_manifest.json")
    extension_probe = _json(extension_profile_dir / "probe_manifest.json")
    checks: list[dict[str, Any]] = []

    def exact_check(component: str, left: Any, right: Any) -> None:
        checks.append(
            {
                "component": component,
                "status": "pass_exact" if left == right else "fail",
                "core": left,
                "extension": right,
            }
        )

    exact_check(
        "complete_status",
        (_json(core_profile_dir / "status.json").get("status"), _json(extension_profile_dir / "status.json").get("status")),
        ("complete", "complete"),
    )
    exact_check(
        "profile_protocols",
        (
            core_summary.get("profile", {}).get("protocol"),
            extension_summary.get("profile", {}).get("protocol"),
        ),
        ("v24-acceptance-local", "v24-acceptance-local"),
    )
    exact_check("source_contract_sha256", _source_contract(core_manifest), _source_contract(extension_manifest))
    exact_check(
        "snapshot_fingerprints",
        core_summary.get("snapshot", {}).get("fingerprints"),
        extension_summary.get("snapshot", {}).get("fingerprints"),
    )
    core_ordered = list(core_probe.get("ordered_probe_contract", ()))
    extension_ordered = list(extension_probe.get("ordered_probe_contract", ()))
    exact_check("ordered_probe_contract", core_ordered, extension_ordered)
    image_keys = {str(row.get("image_key")) for row in core_ordered}
    checks.append(
        {
            "component": "shared_unique_probe_images",
            "status": "pass_exact" if len(image_keys) == len(core_ordered) >= MINIMUM_SHARED_IMAGES else "fail",
            "count": len(image_keys),
            "minimum": MINIMUM_SHARED_IMAGES,
        }
    )

    def candidate_map(summary: Mapping[str, Any]) -> dict[float, str]:
        result: dict[float, str] = {}
        for row in summary.get("candidates", ()):  # type: ignore[union-attr]
            if str(row.get("candidate")) == "no_quant" or row.get("initial_range_mul") is None:
                continue
            result[float(row["initial_range_mul"])] = str(row["candidate"])
        return result

    core_map = candidate_map(core_summary)
    extension_map = candidate_map(extension_summary)
    common_candidates: set[str] = {"no_quant"}
    missing_muls: list[float] = []
    for value in common_muls:
        matches_core = [name for mul, name in core_map.items() if math.isclose(mul, float(value), rel_tol=0.0, abs_tol=1e-12)]
        matches_extension = [name for mul, name in extension_map.items() if math.isclose(mul, float(value), rel_tol=0.0, abs_tol=1e-12)]
        if len(matches_core) != 1 or matches_core != matches_extension:
            missing_muls.append(float(value))
        else:
            common_candidates.add(matches_core[0])
    exact_check("common_mul_candidate_identity", missing_muls, [])
    core_rows = _filtered_rows(
        _csv(core_profile_dir / "per_image.csv"),
        candidates=common_candidates,
        image_keys=image_keys,
        require_sample_record=False,
    )
    extension_rows = _filtered_rows(
        _csv(extension_profile_dir / "per_image.csv"),
        candidates=common_candidates,
        image_keys=image_keys,
        require_sample_record=False,
    )
    checks.append(
        _compare_row_maps(
            component="shared_core_raw_probe_rows",
            local_rows=core_rows,
            formal_rows=extension_rows,
            exact_fields=(
                "candidate", "phase", "probe_or_step", "repeat", "range_mul",
                "gradient_hash", "replay_digest", "noise_digest", "timestep_digest",
                "rng_digest_before", "rng_digest_after", "dropout_mask_digest",
                "quant_rng_digest", "module_invocation_digest", "image_key", "source_group",
                "timestep_bin", "timestep", "noise_replica", "probe_regime",
                "quant_repeat", "gradient_topology_matches",
            ),
            numeric_fields=("loss", "gradient_norm", "parameter_gradient_cosine"),
        )
    )
    core_tail = _filtered_rows(
        _csv(core_profile_dir / "gradient_tail.csv"),
        candidates=common_candidates - {"no_quant"},
        image_keys=image_keys,
        require_sample_record=True,
    )
    extension_tail = _filtered_rows(
        _csv(extension_profile_dir / "gradient_tail.csv"),
        candidates=common_candidates - {"no_quant"},
        image_keys=image_keys,
        require_sample_record=True,
    )
    checks.append(
        _compare_row_maps(
            component="shared_core_gradient_distance_rows",
            local_rows=core_tail,
            formal_rows=extension_tail,
            exact_fields=("candidate", "image_key", "source_group", "timestep_bin", "timestep", "noise_replica", "quant_repeat"),
            numeric_fields=("gradient_cosine", "grad_norm_noquant", "grad_norm_candidate", "grad_diff_norm", "relative_gradient_distance", "symmetric_gradient_distance", "angular_gradient_distance", "gradient_gain_distance"),
        )
    )
    failed = [check for check in checks if check["status"] == "fail"]
    numeric = [check for check in checks if check["status"] == "pass_numeric"]
    gate = "fail" if failed else "pass_numeric" if numeric else "pass_exact"
    return {
        "schema_version": "2.4.0-local-extension-parity",
        "gate": gate,
        "passed": gate in {"pass_exact", "pass_numeric"},
        "checks": checks,
        "first_divergence": failed[0] if failed else None,
        "common_muls": [float(value) for value in common_muls],
        "shared_image_count": len(image_keys),
        "core_profile_dir": str(core_profile_dir),
        "extension_profile_dir": str(extension_profile_dir),
        "safety_not_utility": True,
    }
