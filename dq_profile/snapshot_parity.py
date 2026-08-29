from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def _read_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_snapshot_output(path: str | Path) -> dict[str, Any]:
    root = Path(path).expanduser().resolve()
    summary_path = root / "summary.json" if root.is_dir() else root
    if not summary_path.is_file():
        raise FileNotFoundError(f"snapshot summary was not found: {summary_path}")
    root = summary_path.parent
    summary = _read_json(summary_path)
    snapshot = summary.get("snapshot")
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"summary has no snapshot object: {summary_path}")
    fingerprints = snapshot.get("fingerprints")
    if not isinstance(fingerprints, Mapping) or not fingerprints:
        raise ValueError(f"summary has no snapshot fingerprints: {summary_path}")
    resolved_path = root / "resolved_args.json"
    resolved = _read_json(resolved_path) if resolved_path.is_file() else {}
    state_name = snapshot.get("state_file", "snapshot_state.pt")
    state_path = root / str(state_name)
    return {
        "root": root,
        "summary_path": summary_path,
        "summary": summary,
        "snapshot": dict(snapshot),
        "fingerprints": {str(key): str(value) for key, value in fingerprints.items()},
        "resolved_args": resolved,
        "state_path": state_path if state_path.is_file() else None,
    }


def _relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = (right - left).to(torch.float64)
    denominator = float(torch.linalg.vector_norm(left.to(torch.float64)).item())
    return float(torch.linalg.vector_norm(delta).item()) / max(denominator, 1e-30)


def _compare_tree(
    left: Any,
    right: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    differences: list[dict[str, Any]],
) -> tuple[bool, bool, float, float, int]:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            differences.append({"path": path, "reason": "tensor_type_mismatch"})
            return False, False, math.inf, math.inf, 0
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            differences.append(
                {
                    "path": path,
                    "reason": "tensor_schema_mismatch",
                    "left_dtype": str(left.dtype),
                    "right_dtype": str(right.dtype),
                    "left_shape": list(left.shape),
                    "right_shape": list(right.shape),
                }
            )
            return False, False, math.inf, math.inf, int(left.numel())
        left_cpu = left.detach().to("cpu").contiguous()
        right_cpu = right.detach().to("cpu").contiguous()
        exact = bool(torch.equal(left_cpu, right_cpu))
        if exact:
            return True, True, 0.0, 0.0, int(left_cpu.numel())
        if not (left_cpu.is_floating_point() or left_cpu.is_complex()):
            differences.append({"path": path, "reason": "nonfloating_tensor_mismatch"})
            return False, False, math.inf, math.inf, int(left_cpu.numel())
        finite = bool(torch.isfinite(left_cpu).all() and torch.isfinite(right_cpu).all())
        if not finite:
            differences.append({"path": path, "reason": "nonfinite_tensor_mismatch"})
            return False, False, math.inf, math.inf, int(left_cpu.numel())
        max_abs = float(
            torch.max(
                torch.abs(
                    right_cpu.to(torch.float64) - left_cpu.to(torch.float64)
                )
            ).item()
        ) if left_cpu.numel() else 0.0
        relative_l2 = _relative_l2(left_cpu, right_cpu)
        numeric = max_abs <= atol and relative_l2 <= rtol
        differences.append(
            {
                "path": path,
                "reason": "floating_tensor_difference",
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "numeric_pass": numeric,
            }
        )
        return False, numeric, max_abs, relative_l2, int(left_cpu.numel())

    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            differences.append({"path": path, "reason": "ndarray_type_mismatch"})
            return False, False, math.inf, math.inf, 0
        return _compare_tree(
            torch.from_numpy(np.ascontiguousarray(left)),
            torch.from_numpy(np.ascontiguousarray(right)),
            path=path,
            atol=atol,
            rtol=rtol,
            differences=differences,
        )

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            differences.append({"path": path, "reason": "mapping_type_mismatch"})
            return False, False, math.inf, math.inf, 0
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            differences.append(
                {
                    "path": path,
                    "reason": "mapping_key_mismatch",
                    "left_only": sorted((repr(key) for key in left_keys - right_keys)),
                    "right_only": sorted((repr(key) for key in right_keys - left_keys)),
                }
            )
            return False, False, math.inf, math.inf, 0
        exact = True
        numeric = True
        max_abs = 0.0
        max_relative_l2 = 0.0
        tensor_count = 0
        for key in sorted(left_keys, key=repr):
            result = _compare_tree(
                left[key],
                right[key],
                path=f"{path}.{key!r}",
                atol=atol,
                rtol=rtol,
                differences=differences,
            )
            exact = exact and result[0]
            numeric = numeric and result[1]
            max_abs = max(max_abs, result[2])
            max_relative_l2 = max(max_relative_l2, result[3])
            tensor_count += result[4]
        return exact, numeric, max_abs, max_relative_l2, tensor_count

    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            differences.append({"path": path, "reason": "sequence_schema_mismatch"})
            return False, False, math.inf, math.inf, 0
        exact = True
        numeric = True
        max_abs = 0.0
        max_relative_l2 = 0.0
        tensor_count = 0
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result = _compare_tree(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                atol=atol,
                rtol=rtol,
                differences=differences,
            )
            exact = exact and result[0]
            numeric = numeric and result[1]
            max_abs = max(max_abs, result[2])
            max_relative_l2 = max(max_relative_l2, result[3])
            tensor_count += result[4]
        return exact, numeric, max_abs, max_relative_l2, tensor_count

    if isinstance(left, float) or isinstance(right, float):
        try:
            left_float = float(left)
            right_float = float(right)
        except (TypeError, ValueError):
            differences.append({"path": path, "reason": "scalar_type_mismatch"})
            return False, False, math.inf, math.inf, 0
        exact = left_float == right_float
        if exact:
            return True, True, 0.0, 0.0, 0
        delta = abs(right_float - left_float)
        relative = delta / max(abs(left_float), 1e-30)
        numeric = math.isfinite(delta) and delta <= atol and relative <= rtol
        differences.append(
            {
                "path": path,
                "reason": "floating_scalar_difference",
                "max_abs": delta,
                "relative_l2": relative,
                "numeric_pass": numeric,
            }
        )
        return False, numeric, delta, relative, 0

    exact = type(left) is type(right) and left == right
    if not exact:
        differences.append(
            {
                "path": path,
                "reason": "value_mismatch",
                "left": repr(left),
                "right": repr(right),
            }
        )
    return exact, exact, 0.0 if exact else math.inf, 0.0 if exact else math.inf, 0


def _control_checks(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    left_summary = left["summary"]
    right_summary = right["summary"]
    left_snapshot = left["snapshot"]
    right_snapshot = right["snapshot"]
    pairs = {
        "schema_version": (
            left_summary.get("schema_version"),
            right_summary.get("schema_version"),
        ),
        "metric_definition_version": (
            left_summary.get("metric_definition_version"),
            right_summary.get("metric_definition_version"),
        ),
        "source_contract_sha256": (
            left_summary.get("source_contract_sha256")
            or left_summary.get("v2", {}).get("source_contract_sha256"),
            right_summary.get("source_contract_sha256")
            or right_summary.get("v2", {}).get("source_contract_sha256"),
        ),
        "global_step": (
            left_snapshot.get("global_step"),
            right_snapshot.get("global_step"),
        ),
        "epoch": (left_snapshot.get("epoch"), right_snapshot.get("epoch")),
        "data_step": (
            left_snapshot.get("data_step"),
            right_snapshot.get("data_step"),
        ),
        "lr": (left_snapshot.get("lr"), right_snapshot.get("lr")),
        "dq_delta_begin_step": (
            left_snapshot.get("dq_delta_begin_step"),
            right_snapshot.get("dq_delta_begin_step"),
        ),
        "kernel_policy": (
            left["resolved_args"].get("dq_profile_prefix_kernel_policy"),
            right["resolved_args"].get("dq_profile_prefix_kernel_policy"),
        ),
    }
    if (
        left_summary.get("warmup_contract_sha256") is not None
        and right_summary.get("warmup_contract_sha256") is not None
    ):
        pairs["warmup_contract_sha256"] = (
            left_summary.get("warmup_contract_sha256"),
            right_summary.get("warmup_contract_sha256"),
        )
    if (
        left_snapshot.get("first_quantized_batch_fingerprint") is not None
        and right_snapshot.get("first_quantized_batch_fingerprint") is not None
    ):
        pairs["first_quantized_batch_fingerprint"] = (
            left_snapshot.get("first_quantized_batch_fingerprint"),
            right_snapshot.get("first_quantized_batch_fingerprint"),
        )
    return [
        {
            "name": name,
            "passed": left_value == right_value and left_value is not None,
            "left": left_value,
            "right": right_value,
        }
        for name, (left_value, right_value) in pairs.items()
    ]


def compare_snapshot_outputs(
    left_path: str | Path,
    right_path: str | Path,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-6,
) -> dict[str, Any]:
    left = load_snapshot_output(left_path)
    right = load_snapshot_output(right_path)
    controls = _control_checks(left, right)
    controls_passed = all(item["passed"] for item in controls)
    component_names_match = set(left["fingerprints"]) == set(right["fingerprints"])
    component_names = sorted(
        set(left["fingerprints"]) | set(right["fingerprints"])
    )
    fingerprint_checks = [
        {
            "component": name,
            "passed": left["fingerprints"].get(name)
            == right["fingerprints"].get(name),
            "left": left["fingerprints"].get(name),
            "right": right["fingerprints"].get(name),
        }
        for name in component_names
    ]
    fingerprints_exact = component_names_match and all(
        item["passed"] for item in fingerprint_checks
    )

    numeric_available = (
        left["state_path"] is not None and right["state_path"] is not None
    )
    component_numeric: list[dict[str, Any]] = []
    differences: list[dict[str, Any]] = []
    numeric_passed = False
    if not fingerprints_exact and numeric_available:
        left_state = torch.load(
            left["state_path"], map_location="cpu", weights_only=False
        )
        right_state = torch.load(
            right["state_path"], map_location="cpu", weights_only=False
        )
        if set(left_state) == set(right_state):
            numeric_passed = True
            for name in sorted(left_state):
                before = len(differences)
                result = _compare_tree(
                    left_state[name],
                    right_state[name],
                    path=name,
                    atol=atol,
                    rtol=rtol,
                    differences=differences,
                )
                component_numeric.append(
                    {
                        "component": name,
                        "exact": result[0],
                        "numeric_pass": result[1],
                        "max_abs": result[2],
                        "relative_l2": result[3],
                        "tensor_elements": result[4],
                        "difference_count": len(differences) - before,
                    }
                )
                numeric_passed = numeric_passed and result[1]
        else:
            differences.append(
                {
                    "path": "<root>",
                    "reason": "snapshot_component_key_mismatch",
                    "left_only": sorted(set(left_state) - set(right_state)),
                    "right_only": sorted(set(right_state) - set(left_state)),
                }
            )

    if controls_passed and fingerprints_exact:
        gate = "pass_exact"
    elif controls_passed and numeric_available and numeric_passed:
        gate = "pass_numeric"
    else:
        gate = "fail"
    return {
        "schema_version": "snapshot-parity-v1",
        "gate": gate,
        "passed": gate in {"pass_exact", "pass_numeric"},
        "tolerances": {"max_abs": float(atol), "relative_l2": float(rtol)},
        "left": str(left["root"]),
        "right": str(right["root"]),
        "controls_passed": controls_passed,
        "control_checks": controls,
        "component_names_match": component_names_match,
        "fingerprints_exact": fingerprints_exact,
        "fingerprint_checks": fingerprint_checks,
        "numeric_comparison_available": numeric_available,
        "numeric_passed": numeric_passed if numeric_available else None,
        "component_numeric": component_numeric,
        "first_differences": differences[:50],
    }
