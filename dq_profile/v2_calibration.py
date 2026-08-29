from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from dq_profile.metrics import ExactGradient
from dq_profile.protocol import canonical_sha256
from dq_profile.snapshot import clone_state_to_cpu


PREFIX_CONTROL_FIELDS = (
    "replay_digest",
    "noise_digest",
    "timestep_digest",
    "dropout_mask_digest",
    "quant_rng_digest",
    "rng_digest_before",
    "rng_digest_after",
    "module_invocation_count",
    "module_invocation_digest",
    "update_skipped",
    "optimizer_step_performed",
    "lr_before",
    "lr_after",
)


def _hash_update(hasher: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().to("cpu").contiguous()
        hasher.update(b"tensor")
        hasher.update(str(tensor.dtype).encode("ascii"))
        hasher.update(repr(tuple(tensor.shape)).encode("ascii"))
        hasher.update(tensor.numpy().tobytes())
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        hasher.update(b"ndarray")
        hasher.update(str(array.dtype).encode("ascii"))
        hasher.update(repr(array.shape).encode("ascii"))
        hasher.update(array.tobytes())
        return
    if isinstance(value, Mapping):
        hasher.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_update(hasher, key)
            _hash_update(hasher, value[key])
        return
    if isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_update(hasher, item)
        return
    if isinstance(value, set):
        hasher.update(b"set")
        for item in sorted(value, key=repr):
            _hash_update(hasher, item)
        return
    hasher.update(type(value).__name__.encode("utf-8", errors="replace"))
    hasher.update(repr(value).encode("utf-8", errors="replace"))


def fingerprint_tree(value: Any) -> str:
    hasher = hashlib.sha256()
    _hash_update(hasher, value)
    return hasher.hexdigest()


def rng_fingerprint() -> str:
    payload: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return fingerprint_tree(payload)


def exact_gradient_fingerprint(gradient: ExactGradient) -> str:
    return fingerprint_tree(gradient.values)


def _guardian_payload(guardian: Any, network: torch.nn.Module) -> Any:
    if guardian is None:
        return None
    names = {id(parameter): name for name, parameter in network.named_parameters()}
    previous = getattr(guardian, "prev_grad_map", None)
    return {
        "moving_avg_window": list(getattr(guardian, "moving_avg_window", ())),
        "moving_avg_maxlen": getattr(getattr(guardian, "moving_avg_window", None), "maxlen", None),
        "prev_grad_map": None
        if previous is None
        else {
            names.get(parameter_id, f"unknown:{parameter_id}"): tensor
            for parameter_id, tensor in previous.items()
        },
        "prev_grad_norm": getattr(guardian, "prev_grad_norm", None),
        "log_buffer": list(getattr(guardian, "log_buffer", ())),
    }


def _trainer_payload(trainer: Any, network: torch.nn.Module) -> dict[str, Any]:
    fields = (
        "_te_lr_after_cfg",
        "_te_lr_after_resume_state",
        "_te_lr_after_resumed",
        "_te_lr_after_resume_step",
        "_te_freeze_cfg",
        "_te_frozen_state_dict",
    )
    payload = {field: getattr(trainer, field, None) for field in fields}
    frozen_ids = set(getattr(trainer, "_te_frozen_param_ids", set()))
    payload["_te_frozen_param_names"] = sorted(
        name for name, parameter in network.named_parameters() if id(parameter) in frozen_ids
    )
    return payload


def _network_runtime_payload(network: torch.nn.Module) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "module_modes": {
            name: bool(module.training) for name, module in network.named_modules()
        },
        "requires_grad": {
            name: bool(parameter.requires_grad)
            for name, parameter in network.named_parameters()
        },
    }
    for field in (
        "multiplier",
        "delta_q_step",
        "delta_q_mode",
        "delta_q_granularity",
        "delta_q_stat",
        "delta_q_bits",
        "delta_q_range_mul",
        "delta_q_on_z",
        "delta_q_use_triton",
        "delta_q_triton_stats",
    ):
        if hasattr(network, field):
            payload[field] = getattr(network, field)
    loras = list(getattr(network, "text_encoder_loras", ())) + list(
        getattr(network, "unet_loras", ())
    )
    payload["lora_quant_enabled"] = {
        str(lora.lora_name): bool(getattr(lora, "delta_q_enabled", False))
        for lora in loras
    }
    return payload


def capture_state_bundle(
    *,
    network: torch.nn.Module,
    optimizer: Any,
    scheduler: Any,
    scaler: Any,
    trainer: Any,
    guardian: Any,
) -> dict[str, Any]:
    network_state = clone_state_to_cpu(network.state_dict())
    optimizer_state = clone_state_to_cpu(optimizer.state_dict())
    components = {
        "network": network_state,
        "optimizer": optimizer_state,
        "network_runtime": clone_state_to_cpu(_network_runtime_payload(network)),
        "scheduler": clone_state_to_cpu(scheduler.state_dict()),
        "scaler": None if scaler is None else clone_state_to_cpu(scaler.state_dict()),
        "guardian": clone_state_to_cpu(_guardian_payload(guardian, network)),
        "trainer": clone_state_to_cpu(_trainer_payload(trainer, network)),
    }
    return {
        "fingerprints": {name: fingerprint_tree(value) for name, value in components.items()},
        "numeric": {
            "network": network_state,
            "optimizer": optimizer_state,
        },
    }


def _tensor_paths(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        result[prefix or "$"] = value.detach().to(device="cpu", dtype=torch.float64)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            result.update(_tensor_paths(value[key], f"{prefix}/{repr(key)}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            result.update(_tensor_paths(item, f"{prefix}/{index}"))
    return result


def compare_numeric_trees(reference: Any, candidate: Any) -> dict[str, Any]:
    left = _tensor_paths(reference)
    right = _tensor_paths(candidate)
    topology_matches = set(left) == set(right)
    common = sorted(set(left).intersection(right))
    max_abs = 0.0
    diff_sq = 0.0
    reference_sq = 0.0
    shape_matches = True
    for path in common:
        if left[path].shape != right[path].shape:
            shape_matches = False
            continue
        difference = right[path] - left[path]
        if difference.numel():
            max_abs = max(max_abs, float(torch.max(torch.abs(difference)).item()))
        diff_sq += float(torch.sum(difference * difference).item())
        reference_sq += float(torch.sum(left[path] * left[path]).item())
    relative_l2 = math.sqrt(max(diff_sq, 0.0)) / max(math.sqrt(max(reference_sq, 0.0)), 1e-30)
    return {
        "topology_matches": bool(topology_matches and shape_matches),
        "tensor_count": len(common),
        "max_abs": max_abs,
        "relative_l2": relative_l2,
    }


def relative_gradient_distance(norm_ratio: float, cosine: float) -> float:
    value = 1.0 + float(norm_ratio) ** 2 - 2.0 * float(norm_ratio) * float(cosine)
    return math.sqrt(max(value, 0.0))


def symmetric_gradient_distance(
    reference_norm: float,
    candidate_norm: float,
    difference_norm: float,
) -> float:
    denominator = float(reference_norm) + float(candidate_norm)
    if denominator <= 0.0:
        return 0.0 if float(difference_norm) <= 0.0 else float("nan")
    return 2.0 * float(difference_norm) / denominator


def angular_gradient_distance(cosine: float) -> float:
    clipped = min(1.0, max(-1.0, float(cosine)))
    return math.sqrt(max(2.0 * (1.0 - clipped), 0.0))


def gradient_gain_distance(reference_norm: float, candidate_norm: float) -> float:
    reference = float(reference_norm)
    candidate = float(candidate_norm)
    if reference <= 0.0 and candidate <= 0.0:
        return 0.0
    if reference <= 0.0 or candidate <= 0.0:
        return float("inf")
    return abs(math.log(candidate / reference))


def _isclose(left: Any, right: Any, *, rtol: float = 1e-6, atol: float = 1e-8) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=rtol, abs_tol=atol)
    except (TypeError, ValueError):
        return left == right


def evaluate_prefix_pair(
    *,
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    reference_states: Mapping[int, Mapping[str, Any]],
    candidate_states: Mapping[int, Mapping[str, Any]],
    comparison: str,
    candidate_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    exact = len(reference_rows) == len(candidate_rows)
    numeric = exact
    first_divergence: Optional[dict[str, Any]] = None
    if len(reference_rows) != len(candidate_rows):
        first_divergence = {
            "step": min(len(reference_rows), len(candidate_rows)) + 1,
            "component": "trajectory_length",
        }
    for index, (left, right) in enumerate(zip(reference_rows, candidate_rows), start=1):
        controls = {field: left.get(field) == right.get(field) for field in PREFIX_CONTROL_FIELDS}
        controls_match = all(controls.values())
        loss_exact = left.get("loss") == right.get("loss")
        gradient_exact = left.get("gradient_hash") == right.get("gradient_hash")
        loss_close = _isclose(left.get("loss"), right.get("loss"))
        gradient_norm_close = _isclose(left.get("gradient_norm"), right.get("gradient_norm"))
        row_exact = controls_match and loss_exact and gradient_exact
        row_numeric = controls_match and loss_close and gradient_norm_close
        exact = exact and row_exact
        numeric = numeric and row_numeric
        differing = [field for field, matched in controls.items() if not matched]
        component = (
            f"control:{differing[0]}" if differing else
            "loss" if not loss_close else
            "gradient" if not gradient_norm_close else
            None
        )
        if component is not None and first_divergence is None:
            first_divergence = {"step": index, "component": component}
        rows.append(
            {
                "comparison": comparison,
                "candidate": candidate_name,
                "record_type": "step",
                "step": index,
                "controls_match": controls_match,
                "loss_exact": loss_exact,
                "loss_numeric_close": loss_close,
                "gradient_exact": gradient_exact,
                "gradient_numeric_close": gradient_norm_close,
                "exact": row_exact,
                "numeric": row_numeric,
                "differing_controls": differing,
            }
        )

    required_checkpoints = {0, 1, 32, 64}
    reference_checkpoints = set(reference_states)
    candidate_checkpoints = set(candidate_states)
    checkpoint_topology_matches = (
        reference_checkpoints == candidate_checkpoints
        and required_checkpoints.issubset(reference_checkpoints)
    )
    state_components_exact = checkpoint_topology_matches
    state_numeric = checkpoint_topology_matches
    if not checkpoint_topology_matches and first_divergence is None:
        missing_or_extra = sorted(reference_checkpoints.symmetric_difference(candidate_checkpoints))
        required_missing = sorted(required_checkpoints.difference(reference_checkpoints.intersection(candidate_checkpoints)))
        first_divergence = {
            "step": (missing_or_extra or required_missing or [0])[0],
            "component": "state:checkpoint_presence",
        }
    for checkpoint in sorted(reference_checkpoints.union(candidate_checkpoints)):
        if checkpoint not in reference_states or checkpoint not in candidate_states:
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate_name,
                    "record_type": "state",
                    "step": checkpoint,
                    "component": "checkpoint_presence",
                    "exact": False,
                    "numeric": False,
                    "reference_present": checkpoint in reference_states,
                    "candidate_present": checkpoint in candidate_states,
                }
            )
            continue
        left_bundle = reference_states[checkpoint]
        right_bundle = candidate_states[checkpoint]
        left_fingerprints = left_bundle["fingerprints"]
        right_fingerprints = right_bundle["fingerprints"]
        for component in sorted(set(left_fingerprints).union(right_fingerprints)):
            component_exact = left_fingerprints.get(component) == right_fingerprints.get(component)
            numeric_comparison = None
            component_numeric = component_exact
            if component in {"network", "optimizer"}:
                numeric_comparison = compare_numeric_trees(
                    left_bundle["numeric"].get(component),
                    right_bundle["numeric"].get(component),
                )
                component_numeric = bool(
                    numeric_comparison["topology_matches"]
                    and numeric_comparison["max_abs"] <= 1e-7
                    and numeric_comparison["relative_l2"] <= 1e-6
                )
            state_components_exact = state_components_exact and component_exact
            state_numeric = state_numeric and component_numeric
            if not component_numeric and first_divergence is None:
                first_divergence = {"step": checkpoint, "component": f"state:{component}"}
            rows.append(
                {
                    "comparison": comparison,
                    "candidate": candidate_name,
                    "record_type": "state",
                    "step": checkpoint,
                    "component": component,
                    "exact": component_exact,
                    "numeric": component_numeric,
                    "max_abs": None if numeric_comparison is None else numeric_comparison["max_abs"],
                    "relative_l2": None if numeric_comparison is None else numeric_comparison["relative_l2"],
                    "topology_matches": None
                    if numeric_comparison is None
                    else numeric_comparison["topology_matches"],
                }
            )
    exact = exact and state_components_exact
    numeric = numeric and state_numeric
    status = "pass_exact" if exact else ("pass_numeric" if numeric else "fail")
    return rows, {
        "comparison": comparison,
        "candidate": candidate_name,
        "status": status,
        "first_divergence": first_divergence,
        "step_count_reference": len(reference_rows),
        "step_count_candidate": len(candidate_rows),
        "state_checkpoints": sorted(reference_checkpoints.intersection(candidate_checkpoints)),
        "state_checkpoint_topology_matches": checkpoint_topology_matches,
    }


def aggregate_prefix_gate(pair_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = [str(item.get("status", "fail")) for item in pair_summaries]
    if statuses and all(status == "pass_exact" for status in statuses):
        status = "pass_exact"
    elif statuses and all(status in {"pass_exact", "pass_numeric"} for status in statuses):
        status = "pass_numeric"
    else:
        status = "fail"
    first = next((item.get("first_divergence") for item in pair_summaries if item.get("first_divergence")), None)
    return {
        "schema_version": "2.1.0",
        "metric_definition_version": "2.1.0",
        "gate": status,
        "passed": status in {"pass_exact", "pass_numeric"},
        "pair_results": list(pair_summaries),
        "first_divergence": first,
        "numeric_tolerance": {
            "state_max_abs": 1e-7,
            "state_relative_l2": 1e-6,
            "loss_gradient_rtol": 1e-6,
            "loss_gradient_atol": 1e-8,
        },
    }


def _sample_cv(values: Sequence[float]) -> Optional[float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite) < 2:
        return None
    mean = statistics.mean(finite)
    return statistics.stdev(finite) / max(abs(mean), 1e-30)


def intrinsic_noise_rows(no_quant_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in no_quant_rows:
        grouped[(str(row["image_key"]), int(row["timestep_bin"]))].append(float(row["loss"]))
    rows: list[dict[str, Any]] = []
    for (image_key, timestep_bin), values in sorted(grouped.items()):
        cv = _sample_cv(values)
        if cv is not None:
            rows.append(
                {
                    "image_key": image_key,
                    "timestep_bin": timestep_bin,
                    "noise_replica_count": len(values),
                    "loss_mean": statistics.mean(values),
                    "loss_std": statistics.stdev(values),
                    "loss_cv": cv,
                    "probe_regime": "structural_dropout_off",
                }
            )
    return rows


def bootstrap_intrinsic_noise(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_bins: int,
    iterations: int = 2000,
    seed: int = 39,
) -> dict[str, Any]:
    images = sorted({str(row["image_key"]) for row in rows})
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_image[str(row["image_key"])].append(row)
    rng = np.random.default_rng(int(seed))
    all_values: list[float] = []
    high_values: list[float] = []
    for _ in range(int(iterations)):
        sampled = rng.choice(images, size=len(images), replace=True)
        selected = [row for image in sampled for row in by_image[str(image)]]
        all_values.append(float(np.mean([float(row["loss_cv"]) for row in selected])))
        high = [float(row["loss_cv"]) for row in selected if int(row["timestep_bin"]) == int(timestep_bins) - 1]
        high_values.append(float(np.mean(high)))
    def summary(values: Sequence[float]) -> dict[str, float]:
        return {
            "estimate": float(np.mean(values)),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
    observed_all = [float(row["loss_cv"]) for row in rows]
    observed_high = [
        float(row["loss_cv"]) for row in rows if int(row["timestep_bin"]) == int(timestep_bins) - 1
    ]
    return {
        "definition": "mean per-image/per-timestep sample loss CV across independent diffusion-noise replicas",
        "all_timestep": {**summary(all_values), "observed": float(np.mean(observed_all))},
        "max_timestep_bin": {**summary(high_values), "observed": float(np.mean(observed_high))},
        "bootstrap_iterations": int(iterations),
        "block": "image",
        "seed": int(seed),
    }


def gradient_tail_rows(per_image_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    references: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    for row in per_image_rows:
        if row.get("candidate") == "no_quant":
            key = (str(row["image_key"]), int(row["timestep_bin"]), int(row["noise_replica"]))
            references[key] = row
    output: list[dict[str, Any]] = []
    for row in per_image_rows:
        if row.get("candidate") == "no_quant":
            continue
        key = (str(row["image_key"]), int(row["timestep_bin"]), int(row["noise_replica"]))
        reference = references.get(key)
        if reference is None:
            continue
        reference_norm = float(reference["gradient_norm"])
        candidate_norm = float(row["gradient_norm"])
        norm_ratio = candidate_norm / max(reference_norm, 1e-30)
        cosine = float(row["parameter_gradient_cosine"])
        relative_distance = relative_gradient_distance(norm_ratio, cosine)
        difference_sq = (
            reference_norm * reference_norm
            + candidate_norm * candidate_norm
            - 2.0 * reference_norm * candidate_norm * cosine
        )
        difference_norm = math.sqrt(max(difference_sq, 0.0))
        output.append(
            {
                "image_key": key[0],
                "source_group": str(
                    row.get(
                        "source_group",
                        reference.get("source_group", key[0]),
                    )
                ),
                "timestep_bin": key[1],
                "noise_replica": key[2],
                "quant_repeat": int(row["quant_repeat"]),
                "candidate": str(row["candidate"]),
                "range_mul": float(row["range_mul"]),
                "gradient_cosine": cosine,
                "gradient_norm_ratio": norm_ratio,
                "grad_norm_noquant": reference_norm,
                "grad_norm_candidate": candidate_norm,
                "grad_diff_norm": difference_norm,
                "relative_gradient_distance": relative_distance,
                "symmetric_gradient_distance": symmetric_gradient_distance(
                    reference_norm,
                    candidate_norm,
                    difference_norm,
                ),
                "angular_gradient_distance": angular_gradient_distance(cosine),
                "gradient_gain_distance": gradient_gain_distance(
                    reference_norm,
                    candidate_norm,
                ),
                "d_gt_1": relative_distance > 1.0,
                "gradient_cosine_lt_0": cosine < 0.0,
                "probe_regime": "structural_dropout_off",
            }
        )
    return output


def summarize_gradient_tail(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_bins: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, Optional[int], Optional[int]], list[float]] = defaultdict(list)
    for row in rows:
        candidate = str(row["candidate"])
        timestep_bin = int(row["timestep_bin"])
        value = float(row["relative_gradient_distance"])
        grouped[(candidate, timestep_bin, None, None)].append(value)
        grouped[(candidate, timestep_bin, int(row["noise_replica"]), int(row["quant_repeat"]))].append(value)
    output: list[dict[str, Any]] = []
    for (candidate, timestep_bin, noise_replica, quant_repeat), values in sorted(grouped.items(), key=repr):
        members = [
            row
            for row in rows
            if str(row["candidate"]) == candidate
            and int(row["timestep_bin"]) == timestep_bin
            and (noise_replica is None or int(row["noise_replica"]) == noise_replica)
            and (quant_repeat is None or int(row["quant_repeat"]) == quant_repeat)
        ]
        output.append(
            {
                "candidate": candidate,
                "timestep_bin": timestep_bin,
                "is_max_timestep_bin": timestep_bin == int(timestep_bins) - 1,
                "noise_replica": noise_replica,
                "quant_repeat": quant_repeat,
                "stratum": "pooled" if noise_replica is None else f"noise_{noise_replica}.quant_{quant_repeat}",
                "count": len(values),
                "q90_d": float(np.quantile(values, 0.90)),
                "q95_d": float(np.quantile(values, 0.95)),
                "q99_d": float(np.quantile(values, 0.99)),
                "max_d": max(values),
                "d_gt_1_rate": statistics.mean(bool(row["d_gt_1"]) for row in members),
                "gradient_cosine_lt_0_rate": statistics.mean(
                    bool(row["gradient_cosine_lt_0"]) for row in members
                ),
            }
        )
    return output


def bootstrap_tail_winner(
    rows: Sequence[Mapping[str, Any]],
    *,
    timestep_bins: int,
    lower_candidate: str = "mul_2.700",
    upper_candidates: Sequence[str] = ("mul_3.150", "mul_3.450"),
    iterations: int = 2000,
    seed: int = 39,
) -> dict[str, Any]:
    high = [row for row in rows if int(row["timestep_bin"]) == int(timestep_bins) - 1]
    images = sorted({str(row["image_key"]) for row in high})
    by_image: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in high:
        by_image[str(row["image_key"])].append(row)
    rng = np.random.default_rng(int(seed))
    upper_wins = 0
    lower_wins = 0
    ties = 0
    winner_candidates: defaultdict[str, int] = defaultdict(int)
    for _ in range(int(iterations)):
        sampled = rng.choice(images, size=len(images), replace=True)
        selected = [row for image in sampled for row in by_image[str(image)]]
        q95: dict[str, float] = {}
        for candidate in (lower_candidate, *upper_candidates):
            values = [
                float(row["relative_gradient_distance"])
                for row in selected
                if str(row["candidate"]) == candidate
            ]
            q95[candidate] = float(np.quantile(values, 0.95))
        upper_name = min(upper_candidates, key=lambda name: (q95[name], name))
        winner_candidates[upper_name] += 1
        if q95[upper_name] < q95[lower_candidate]:
            upper_wins += 1
        elif q95[lower_candidate] < q95[upper_name]:
            lower_wins += 1
        else:
            ties += 1
    strata: list[dict[str, Any]] = []
    for noise_replica in sorted({int(row["noise_replica"]) for row in high}):
        for quant_repeat in sorted({int(row["quant_repeat"]) for row in high}):
            members = [
                row
                for row in high
                if int(row["noise_replica"]) == noise_replica
                and int(row["quant_repeat"]) == quant_repeat
            ]
            q95 = {
                candidate: float(
                    np.quantile(
                        [
                            float(row["relative_gradient_distance"])
                            for row in members
                            if str(row["candidate"]) == candidate
                        ],
                        0.95,
                    )
                )
                for candidate in (lower_candidate, *upper_candidates)
            }
            upper_name = min(upper_candidates, key=lambda name: (q95[name], name))
            winner = "upper" if q95[upper_name] < q95[lower_candidate] else (
                "lower" if q95[lower_candidate] < q95[upper_name] else "tie"
            )
            strata.append(
                {
                    "noise_replica": noise_replica,
                    "quant_repeat": quant_repeat,
                    "winner": winner,
                    "upper_candidate": upper_name,
                    "q95": q95,
                }
            )
    upper_support = upper_wins / max(int(iterations), 1)
    lower_support = lower_wins / max(int(iterations), 1)
    upper_strata_wins = sum(item["winner"] == "upper" for item in strata)
    if upper_support >= 0.75 and upper_strata_wins >= 3:
        decision = "supported_on_development_dataset"
    elif lower_support >= 0.75:
        decision = "lower_contradiction"
    else:
        decision = "abstain"
    return {
        "primary_metric": "q95(relative_gradient_distance) in max timestep bin",
        "relative_gradient_distance_definition": "sqrt(1+r^2-2*r*cosine)",
        "lower_candidate": lower_candidate,
        "upper_candidates": list(upper_candidates),
        "upper_support_probability": upper_support,
        "lower_support_probability": lower_support,
        "tie_probability": ties / max(int(iterations), 1),
        "upper_candidate_win_probability": {
            name: winner_candidates[name] / max(int(iterations), 1) for name in upper_candidates
        },
        "strata": strata,
        "upper_strata_wins": upper_strata_wins,
        "decision": decision,
        "development_dataset_only": True,
        "bootstrap_iterations": int(iterations),
        "block": "image",
        "seed": int(seed),
    }


def source_contract_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "schema_version": manifest.get("schema_version"),
        "metric_definition_version": manifest.get("metric_definition_version"),
        "protocol_version": manifest.get("protocol_version"),
        "quant_rng_mode": manifest.get("quant_rng_mode"),
        "python": manifest.get("python"),
        "platform": manifest.get("platform"),
        "packages": manifest.get("packages"),
        "cuda": manifest.get("cuda"),
        "explicit_source_files": manifest.get("explicit_source_files"),
        "additional_input_files": manifest.get("additional_input_files"),
    }
    return {"sha256": canonical_sha256(contract), "contract": contract}
