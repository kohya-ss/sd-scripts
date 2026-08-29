from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from dq_profile.manifest import build_source_manifest
from dq_profile.protocol import (
    DEFAULT_V2_RANGE_MULS,
    CandidateDefinition,
    canonical_sha256,
    default_candidates,
    fixed_range_candidates,
    inspect_dataset_config,
    parse_mechanism_muls,
    parse_range_muls,
)
from dq_profile.report import ProfileArtifacts
from dq_profile.v2_calibration import source_contract_from_manifest
from dq_profile.v24_trajectory import (
    canonical_sha256 as trajectory_canonical_sha256,
    validate_trajectory_contract,
)
from dq_profile.sdxl_profile_trainer import SdxlNetworkTrainer, configure_sdxl_globals, setup_parser as setup_training_parser
from library import train_util


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_parser() -> argparse.ArgumentParser:
    parser = setup_training_parser()
    group = parser.add_argument_group("DQ Dataset Profiler")
    group.add_argument("--dq_profile_output_dir", type=str, default="dq_profile_output")
    group.add_argument("--dq_profile_name", type=str, default="dq_profile")
    group.add_argument("--dq_profile_level", choices=("standard", "full"), default="standard")
    group.add_argument(
        "--dq_profile_protocol",
        choices=(
            "v1",
            "v2-core",
            "v2-mechanism",
            "v2-prefix-smoke",
            "v2-tail-calibration",
            "v23-safety-local",
            "v23-safety-formal",
            "v24-acceptance-local",
            "v24-acceptance-formal",
            "v24-trajectory-descriptive",
        ),
        default="v2-core",
    )
    group.add_argument("--dq_profile_max_images", type=int, default=32)
    group.add_argument("--dq_profile_timestep_bins", type=int, default=4)
    group.add_argument("--dq_profile_stochastic_repeats", type=int, default=3)
    group.add_argument("--dq_profile_branch_steps", type=int, default=None)
    group.add_argument("--dq_profile_seed", type=int, default=None)
    group.add_argument("--dq_profile_range_muls", type=str, default=",".join(str(value) for value in DEFAULT_V2_RANGE_MULS))
    group.add_argument("--dq_profile_sweep_steps", type=int, default=64)
    group.add_argument("--dq_profile_branch_repeats", type=int, default=2)
    group.add_argument("--dq_profile_guardian_ablation", choices=("common_then_native", "common_only", "native_only"), default="common_then_native")
    group.add_argument("--dq_profile_sketch_width", type=int, default=512)
    group.add_argument("--dq_profile_sketch_seeds", type=int, default=2)
    group.add_argument("--dq_profile_source_group_map", type=str, default=None)
    group.add_argument("--dq_profile_core_gate_file", type=str, default=None)
    group.add_argument("--dq_profile_prefix_gate_file", type=str, default=None)
    group.add_argument(
        "--dq_profile_safety_local_selection_file",
        type=str,
        default=None,
        help="v2.3.2 local-selection JSON required by v23-safety-formal",
    )
    group.add_argument(
        "--dq_profile_trajectory_contract_file",
        type=str,
        default=None,
        help="preregistered descriptive-only 128-step trajectory contract",
    )
    group.add_argument(
        "--dq_profile_disable_progress_bar",
        action="store_true",
        help="diagnostic-only: disable tqdm output for redirected unattended runs",
    )
    group.add_argument(
        "--dq_profile_prefix_kernel_mode",
        choices=("deterministic", "native"),
        default="deterministic",
        help=(
            "GPU kernel policy for reproducibility-gated v2 protocols "
            "(prefix, tail calibration, and v2.3 safety local/formal)"
        ),
    )
    group.add_argument(
        "--dq_profile_snapshot_only",
        action="store_true",
        help=(
            "diagnostic-only: stop at the common warmup boundary, export the "
            "captured state, and omit probes/branches; valid only with "
            "v2-prefix-smoke"
        ),
    )
    group.add_argument("--dq_profile_core_profile_key", type=str, default=None)
    group.add_argument("--dq_profile_mechanism_muls", type=str, default=None)
    group.add_argument("--dq_profile_dry_run", action="store_true")
    return parser


def _resolved_candidate_definitions(args: argparse.Namespace) -> tuple[CandidateDefinition, ...]:
    if args.dq_profile_protocol == "v1":
        return default_candidates()
    if args.dq_profile_protocol == "v2-prefix-smoke":
        return (
            CandidateDefinition("no_quant", False, None, None, None, False),
            CandidateDefinition("mul_3.150", True, None, None, 3.15, False),
        )
    minimum_count = (
        1
        if args.dq_profile_protocol in {
            "v24-acceptance-formal",
            "v24-trajectory-descriptive",
        }
        else 2
        if args.dq_profile_protocol == "v23-safety-formal"
        else 3
    )
    return fixed_range_candidates(
        args.dq_profile_range_muls_resolved,
        minimum_count=minimum_count,
        maximum_count=3 if args.dq_profile_protocol.endswith("-formal") or args.dq_profile_protocol == "v24-trajectory-descriptive" else None,
    )

def _validate_and_isolate(args: argparse.Namespace) -> None:
    protocol = str(getattr(args, "dq_profile_protocol", "v1"))
    if bool(getattr(args, "dq_profile_snapshot_only", False)) and protocol != "v2-prefix-smoke":
        raise ValueError(
            "--dq_profile_snapshot_only is a diagnostic boundary check and "
            "requires --dq_profile_protocol=v2-prefix-smoke"
        )
    if not args.dataset_config:
        raise ValueError("--dataset_config is required")
    if int(getattr(args, "gradient_accumulation_steps", 1)) != 1:
        raise ValueError("DQ Dataset Profiler requires --gradient_accumulation_steps=1")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("DQ Dataset Profiler requires a single process/GPU")
    if getattr(args, "resume", None):
        raise ValueError("--resume is not supported by the isolated profiler")
    requested_module = getattr(args, "network_module", None)
    allowed = {None, "", "networks.lora", "dq_profile.copied_lora"}
    if requested_module not in allowed:
        raise ValueError(
            "DQ Dataset Profiler only accepts the normal LoRA module; "
            f"requested network_module={requested_module!r}"
        )
    if not getattr(args, "dq_delta_bits", None):
        raise ValueError("DQ Dataset Profiler requires a fixed positive --dq_delta_bits")
    if getattr(args, "dq_delta_bits_sched", None):
        raise ValueError("DQ Dataset Profiler does not support --dq_delta_bits_sched; use fixed --dq_delta_bits")
    if getattr(args, "dq_delta_step", None):
        raise ValueError("DQ Dataset Profiler requires bits mode, not dq_delta_step")
    if getattr(args, "lr_warmup_steps", 0) is None or float(args.lr_warmup_steps) <= 0.0:
        raise ValueError("DQ Dataset Profiler requires a positive LR warmup for the common no-quant snapshot")
    for name in (
        "dq_profile_max_images", "dq_profile_timestep_bins", "dq_profile_stochastic_repeats",
        "dq_profile_sketch_width", "dq_profile_sketch_seeds", "dq_profile_sweep_steps", "dq_profile_branch_repeats",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.dq_profile_source_group_map and not Path(args.dq_profile_source_group_map).expanduser().is_file():
        raise FileNotFoundError(f"source group map was not found: {args.dq_profile_source_group_map}")
    profile_name = str(args.dq_profile_name).strip()
    if not profile_name or Path(profile_name).name != profile_name:
        raise ValueError("--dq_profile_name must be a single non-empty path component")
    prefix_required_protocols = {
        "v2-tail-calibration",
        "v23-safety-local",
        "v23-safety-formal",
        "v24-acceptance-local",
        "v24-acceptance-formal",
        "v24-trajectory-descriptive",
    }
    if protocol in prefix_required_protocols:
        if not getattr(args, "dq_profile_prefix_gate_file", None):
            raise ValueError(
                f"{protocol} requires --dq_profile_prefix_gate_file from "
                "schema 2.1 prefix smoke"
            )
        prefix_gate_path = Path(args.dq_profile_prefix_gate_file).expanduser().resolve()
        if not prefix_gate_path.is_file():
            raise FileNotFoundError(f"prefix gate file was not found: {prefix_gate_path}")
        prefix_gate = json.loads(prefix_gate_path.read_text(encoding="utf-8-sig"))
        if (
            prefix_gate.get("schema_version") != "2.1.0"
            or prefix_gate.get("metric_definition_version") != "2.1.0"
            or prefix_gate.get("passed") is not True
            or prefix_gate.get("gate") not in {"pass_exact", "pass_numeric"}
            or not prefix_gate.get("source_contract_sha256")
        ):
            raise ValueError(
                f"{protocol} requires a passed schema/metric 2.1.0 prefix gate "
                "with source contract"
            )
        args.dq_profile_prefix_gate_file = str(prefix_gate_path)
        args.dq_profile_prefix_gate_sha256 = canonical_sha256(prefix_gate)
        args.dq_profile_expected_source_contract_sha256 = str(
            prefix_gate["source_contract_sha256"]
        )
        contract_inputs = [args.dataset_config]
        if getattr(args, "dq_profile_source_group_map", None):
            contract_inputs.append(args.dq_profile_source_group_map)
        current_manifest, _ = build_source_manifest(
            Path(__file__).resolve().parent,
            quant_rng_mode="stateless",
            additional_files=tuple(contract_inputs),
        )
        current_contract = source_contract_from_manifest(current_manifest)
        args.dq_profile_current_source_contract_sha256 = current_contract["sha256"]
        if (
            args.dq_profile_current_source_contract_sha256
            != args.dq_profile_expected_source_contract_sha256
        ):
            raise ValueError(
                f"source contract changed after prefix smoke; {protocol} was not started "
                f"(expected {args.dq_profile_expected_source_contract_sha256}, "
                f"current {args.dq_profile_current_source_contract_sha256})"
            )
    elif getattr(args, "dq_profile_prefix_gate_file", None):
        raise ValueError(
            "--dq_profile_prefix_gate_file is only valid with prefix-gated "
            "calibration/safety protocols"
        )

    if protocol in {"v23-safety-formal", "v24-acceptance-formal"}:
        v24_formal = protocol == "v24-acceptance-formal"
        formal_minimum = 1 if v24_formal else 2
        selection_schema = (
            "2.4.0-local-selection" if v24_formal else "2.3.2-local-selection"
        )
        selection_schema_display = "2.4.0" if v24_formal else "2.3.2"
        local_protocol = "v24-acceptance-local" if v24_formal else "v23-safety-local"
        selection_value = getattr(
            args, "dq_profile_safety_local_selection_file", None
        )
        if not selection_value:
            raise ValueError(
                f"{protocol} requires "
                "--dq_profile_safety_local_selection_file"
            )
        selection_path = Path(selection_value).expanduser().resolve()
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"safety local selection file was not found: {selection_path}"
            )
        selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
        requested_muls = parse_range_muls(
            args.dq_profile_range_muls,
            minimum_count=formal_minimum,
            maximum_count=3,
        )
        selected_muls = parse_range_muls(
            selection.get("selected_muls", ()),
            minimum_count=formal_minimum,
            maximum_count=3,
        )
        local_grid = parse_range_muls(selection.get("local_grid", ()))
        rule = selection.get("selection_rule")
        local_summary_path = Path(
            str(selection.get("local_summary_path", ""))
        ).expanduser().resolve()
        local_analysis_summary_path = Path(
            str(selection.get("local_analysis_summary_path", ""))
        ).expanduser().resolve()
        if (
            selection.get("schema_version") != selection_schema
            or selection.get("selection_valid") is not True
            or selection.get("source_contract_sha256")
            != getattr(args, "dq_profile_expected_source_contract_sha256", None)
            or not selection.get("local_summary_sha256")
            or not selection.get("selection_rule_sha256")
            or not isinstance(rule, dict)
            or canonical_sha256(rule) != selection.get("selection_rule_sha256")
            or not local_summary_path.is_file()
            or _sha256_file(local_summary_path)
            != selection.get("local_summary_sha256")
            or not local_analysis_summary_path.is_file()
            or _sha256_file(local_analysis_summary_path)
            != selection.get("local_analysis_summary_sha256")
        ):
            raise ValueError(
                f"{protocol} requires a valid schema {selection_schema_display} local selection "
                f"({selection_schema}) "
                "with the same source contract and untampered inputs"
            )
        local_summary = json.loads(
            local_summary_path.read_text(encoding="utf-8-sig")
        )
        if (
            str(local_summary.get("schema_version")) != "2.1.0"
            or str(local_summary.get("profile", {}).get("protocol"))
            != local_protocol
            or any(
                not any(math.isclose(value, grid, rel_tol=0.0, abs_tol=1e-12) for grid in local_grid)
                for value in selected_muls
            )
        ):
            raise ValueError(
                "formal safety local provenance is not a compatible local profile"
            )
        if selected_muls != requested_muls or not formal_minimum <= len(selected_muls) <= 3:
            raise ValueError(
                "--dq_profile_range_muls must exactly match the local-selected "
                f"muls: {list(selected_muls)}"
            )
        args.dq_profile_safety_local_selection_file = str(selection_path)
        args.dq_profile_safety_local_selection_sha256 = canonical_sha256(
            selection
        )
        args.dq_profile_safety_local_summary_sha256 = str(
            selection["local_summary_sha256"]
        )
        args.dq_profile_safety_local_profile_dir = str(
            Path(str(selection["local_profile_dir"])).expanduser().resolve()
        )
        args.dq_profile_safety_local_grid = local_grid
    elif getattr(args, "dq_profile_safety_local_selection_file", None):
        raise ValueError(
            "--dq_profile_safety_local_selection_file is only valid with "
            "a safety/acceptance formal protocol"
        )
    if protocol == "v24-trajectory-descriptive":
        contract_value = getattr(args, "dq_profile_trajectory_contract_file", None)
        if not contract_value:
            raise ValueError(
                "v24-trajectory-descriptive requires "
                "--dq_profile_trajectory_contract_file"
            )
        contract_path = Path(contract_value).expanduser().resolve()
        if not contract_path.is_file():
            raise FileNotFoundError(
                f"trajectory contract file was not found: {contract_path}"
            )
        trajectory_contract = json.loads(
            contract_path.read_text(encoding="utf-8-sig")
        )
        requested_muls = parse_range_muls(
            args.dq_profile_range_muls,
            minimum_count=1,
            maximum_count=3,
        )
        trajectory_info = validate_trajectory_contract(
            trajectory_contract,
            requested_muls=requested_muls,
            expected_source_contract_sha256=str(
                args.dq_profile_expected_source_contract_sha256
            ),
            expected_prefix_gate_sha256=str(args.dq_profile_prefix_gate_sha256),
        )
        args.dq_profile_trajectory_contract_file = str(contract_path)
        args.dq_profile_trajectory_contract_sha256 = (
            trajectory_canonical_sha256(trajectory_contract)
        )
        args.dq_profile_trajectory_content_sha256 = str(
            trajectory_info["contract_sha256"]
        )
        args.dq_profile_trajectory_local_profile_dir = str(
            trajectory_info["local_profile_dir"]
        )
        args.dq_profile_trajectory_local_analysis_dir = str(
            trajectory_info["local_analysis_dir"]
        )
        args.dq_profile_trajectory_local_summary_sha256 = str(
            trajectory_info["local_summary_sha256"]
        )
        args.dq_profile_trajectory_local_grid = tuple(
            trajectory_info["local_grid"]
        )
        args.dq_profile_trajectory_candidate_roles = tuple(
            trajectory_info["candidate_roles"]
        )
        args.dq_profile_trajectory_edge_unresolved = bool(
            trajectory_info["edge_unresolved"]
        )
    elif getattr(args, "dq_profile_trajectory_contract_file", None):
        raise ValueError(
            "--dq_profile_trajectory_contract_file is only valid with "
            "v24-trajectory-descriptive"
        )
    if protocol == "v2-mechanism":
        if not args.dq_profile_core_gate_file:
            raise ValueError("v2-mechanism requires --dq_profile_core_gate_file from the five-profile Core review")
        gate_path = Path(args.dq_profile_core_gate_file).expanduser().resolve()
        if not gate_path.is_file():
            raise FileNotFoundError(f"Core gate file was not found: {gate_path}")
        gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
        if gate.get("schema_version") != "1.0.0" or gate.get("core_gate_passed") is not True:
            raise ValueError(
                "v2-mechanism is gated until core_gate.json has schema_version=1.0.0 and core_gate_passed=true"
            )
        args.dq_profile_core_gate_file = str(gate_path)
        args.dq_profile_core_gate_sha256 = canonical_sha256(gate)
        approved = gate.get("mechanism_selected_muls") or {}
        profile_key = str(args.dq_profile_core_profile_key or profile_name)
        if profile_key not in approved:
            raise ValueError(
                "Core gate does not contain an approved mechanism range for "
                f"profile key {profile_key!r}; use --dq_profile_core_profile_key from core_gate.json"
            )
        approved_values = parse_mechanism_muls(approved[profile_key])
        requested_values = parse_mechanism_muls(args.dq_profile_mechanism_muls)
        if requested_values and requested_values != approved_values:
            raise ValueError(
                "--dq_profile_mechanism_muls must exactly match the Core-gate-approved values "
                f"for {profile_key!r}: {list(approved_values)}"
            )
        args.dq_profile_core_profile_key = profile_key
        args.dq_profile_mechanism_muls_resolved = approved_values
        args.dq_profile_mechanism_selection_source = "core_gate"
    elif args.dq_profile_core_gate_file:
        gate_path = Path(args.dq_profile_core_gate_file).expanduser().resolve()
        if not gate_path.is_file():
            raise FileNotFoundError(f"Core gate file was not found: {gate_path}")
        args.dq_profile_core_gate_file = str(gate_path)
        args.dq_profile_core_gate_sha256 = canonical_sha256(json.loads(gate_path.read_text(encoding="utf-8-sig")))
    if protocol != "v2-mechanism":
        if args.dq_profile_mechanism_muls or args.dq_profile_core_profile_key:
            raise ValueError("mechanism selection arguments are only valid with --dq_profile_protocol=v2-mechanism")
        args.dq_profile_mechanism_muls_resolved = ()

    if protocol != "v1":
        canonical_errors: list[str] = []
        if str(getattr(args, "optimizer_type", "")).casefold() != "adamw8bitfast":
            canonical_errors.append("--optimizer_type=AdamW8bitFast")
        if int(getattr(args, "network_dim", 0) or 0) != 4:
            canonical_errors.append("--network_dim=4")
        if int(getattr(args, "seed", 0) or 0) != 39:
            canonical_errors.append("--seed=39")
        if str(getattr(args, "fp16_safe_norms_mode", "") or "").casefold() != "strict":
            canonical_errors.append("--fp16_safe_norms_mode=strict")
        if not bool(getattr(args, "dq_delta_use_triton", False)):
            canonical_errors.append("--dq_delta_use_triton")
        network_dropout = float(getattr(args, "network_dropout", 0.0) or 0.0)
        if not math.isclose(network_dropout, 0.3, rel_tol=0.0, abs_tol=1e-12):
            canonical_errors.append("--network_dropout=0.3")
        network_args = tuple(str(value).replace(" ", "") for value in (getattr(args, "network_args", None) or ()))
        if "rank_dropout=0.2" not in network_args:
            canonical_errors.append('--network_args "rank_dropout=0.2"')
        if canonical_errors:
            raise ValueError(
                "v2 canonical contract is not satisfied; required: " + ", ".join(canonical_errors)
            )

    args.dq_profile_requested_range_muls = args.dq_profile_range_muls
    if protocol == "v2-prefix-smoke":
        args.dq_profile_range_muls_resolved = (3.15,)
        args.dq_profile_sweep_steps = 128
        args.dq_profile_branch_repeats = 1
        args.dq_profile_guardian_ablation = "common_only"
    elif protocol == "v2-tail-calibration":
        args.dq_profile_range_muls_resolved = (2.70, 3.15, 3.45)
        args.dq_profile_sweep_steps = 128
        args.dq_profile_branch_repeats = 5
        args.dq_profile_guardian_ablation = "common_only"
        args.dq_profile_max_images = 32
        args.dq_profile_timestep_bins = 4
        args.dq_profile_safety_no_quant_noise_replicas_resolved = 5
        args.dq_profile_safety_candidate_noise_replicas_resolved = 2
        args.dq_profile_safety_quant_repeats_resolved = 2
    elif protocol == "v23-safety-local":
        args.dq_profile_range_muls_resolved = parse_range_muls(
            args.dq_profile_range_muls
        )
        args.dq_profile_sweep_steps = 128
        args.dq_profile_branch_repeats = 0
        args.dq_profile_guardian_ablation = "common_only"
        args.dq_profile_max_images = 16
        args.dq_profile_timestep_bins = 4
        args.dq_profile_safety_no_quant_noise_replicas_resolved = 3
        args.dq_profile_safety_candidate_noise_replicas_resolved = 2
        args.dq_profile_safety_quant_repeats_resolved = 2
    elif protocol == "v23-safety-formal":
        args.dq_profile_range_muls_resolved = parse_range_muls(
            args.dq_profile_range_muls,
            minimum_count=2,
            maximum_count=3,
        )
        args.dq_profile_sweep_steps = 128
        args.dq_profile_branch_repeats = 5
        args.dq_profile_guardian_ablation = "common_only"
        args.dq_profile_max_images = 32
        args.dq_profile_timestep_bins = 4
        args.dq_profile_safety_no_quant_noise_replicas_resolved = 5
        args.dq_profile_safety_candidate_noise_replicas_resolved = 2
        args.dq_profile_safety_quant_repeats_resolved = 2
    elif protocol == "v24-acceptance-local":
        args.dq_profile_range_muls_resolved = parse_range_muls(
            args.dq_profile_range_muls,
            minimum_count=3,
        )
        args.dq_profile_sweep_steps = 128
        args.dq_profile_branch_repeats = 0
        args.dq_profile_guardian_ablation = "common_only"
        args.dq_profile_timestep_bins = 4
        args.dq_profile_safety_no_quant_noise_replicas_resolved = 3
        args.dq_profile_safety_candidate_noise_replicas_resolved = 2
        args.dq_profile_safety_quant_repeats_resolved = 2
    elif protocol in {"v24-acceptance-formal", "v24-trajectory-descriptive"}:
        args.dq_profile_range_muls_resolved = parse_range_muls(
            args.dq_profile_range_muls,
            minimum_count=1,
            maximum_count=3,
        )
        args.dq_profile_sweep_steps = 128
        args.dq_profile_branch_repeats = 5
        args.dq_profile_guardian_ablation = "common_only"
        args.dq_profile_timestep_bins = 4
        args.dq_profile_safety_no_quant_noise_replicas_resolved = 5
        args.dq_profile_safety_candidate_noise_replicas_resolved = 2
        args.dq_profile_safety_quant_repeats_resolved = 2
    else:
        args.dq_profile_range_muls_resolved = parse_range_muls(args.dq_profile_range_muls)
    if protocol not in {
        "v1",
        "v2-prefix-smoke",
        "v23-safety-local",
        "v24-acceptance-local",
    } and int(args.dq_profile_branch_repeats) < 2:
        raise ValueError("v2 requires --dq_profile_branch_repeats>=2")

    args.dq_profile_requested_network_module = requested_module
    args.dq_profile_requested_output_dir = getattr(args, "output_dir", None)
    args.dq_profile_requested_output_name = getattr(args, "output_name", None)
    args.dq_profile_requested_max_data_loader_n_workers = getattr(args, "max_data_loader_n_workers", None)
    args.dq_profile_requested_persistent_data_loader_workers = getattr(args, "persistent_data_loader_workers", None)
    args.dq_profile_requested_dq_delta_auto_range_mul = getattr(args, "dq_delta_auto_range_mul", None)
    for name in ("cache_latents_to_disk", "cache_text_encoder_outputs_to_disk", "logging_dir"):
        if hasattr(args, name):
            setattr(args, f"dq_profile_requested_{name}", getattr(args, name))
    args.network_module = "dq_profile.copied_lora"
    args.dq_profile_quant_rng_mode = "stateless"
    args.dq_profile_enabled = True
    args.dq_delta_begin_after_lr_warmup = True
    args.max_data_loader_n_workers = 0
    args.persistent_data_loader_workers = False
    args.output_name = profile_name
    args.dq_profile_run_dir = str((Path(args.dq_profile_output_dir).expanduser().resolve() / profile_name).resolve())
    args.output_dir = args.dq_profile_run_dir
    if hasattr(args, "cache_latents_to_disk"):
        args.cache_latents_to_disk = False
    if hasattr(args, "cache_text_encoder_outputs_to_disk"):
        args.cache_text_encoder_outputs_to_disk = False
    if hasattr(args, "logging_dir"):
        args.logging_dir = None

    # The copied trainer is used only for the common warmup.  All model/state
    # writes and unrelated logging are disabled before it starts.
    for name in (
        "save_every_n_epochs",
        "save_every_n_steps",
        "save_last_n_epochs",
        "save_last_n_steps",
        "sample_every_n_epochs",
        "sample_every_n_steps",
        "huggingface_repo_id",
        "log_with",
    ):
        if hasattr(args, name):
            setattr(args, f"dq_profile_requested_{name}", getattr(args, name))
            setattr(args, name, None)
    for name in (
        "save_state",
        "save_state_on_train_end",
        "save_state_to_huggingface",
        "sample_at_first",
        "avg_cp",
        "group_loss_log",
        "dq_delta_log",
        "rank_log",
    ):
        if hasattr(args, name):
            setattr(args, f"dq_profile_requested_{name}", getattr(args, name))
            setattr(args, name, False)
    # Candidate-local auto controllers are created from the production values.
    # Disable the warmup loop's controller so it cannot mutate the shared args.
    args.dq_delta_auto_range_mul = False
    if args.dq_profile_seed is None:
        args.dq_profile_seed = int(getattr(args, "seed", 0) or 0)
    if bool(getattr(args, "dq_profile_snapshot_only", False)):
        args.dq_profile_capture_steps = 1
    elif protocol in {
        "v2-prefix-smoke",
        "v2-tail-calibration",
        "v23-safety-local",
        "v23-safety-formal",
        "v24-acceptance-local",
        "v24-acceptance-formal",
        "v24-trajectory-descriptive",
    }:
        args.dq_profile_capture_steps = max(
            int(args.dq_profile_branch_steps or 0),
            128,
        )
    else:
        args.dq_profile_capture_steps = max(
            int(args.dq_profile_sweep_steps) * 2,
            int(args.dq_profile_branch_steps or 0),
            64,
        )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    summary = inspect_dataset_config(
        args.dataset_config,
        max_train_epochs=getattr(args, "max_train_epochs", None),
        max_train_steps=getattr(args, "max_train_steps", None),
        lr_warmup_steps=args.lr_warmup_steps,
        branch_steps_override=args.dq_profile_branch_steps,
        max_images=args.dq_profile_max_images,
        timestep_bins=args.dq_profile_timestep_bins,
        stochastic_repeats=args.dq_profile_stochastic_repeats,
    )
    args.dq_profile_branch_steps_resolved = summary.branch_steps
    args.dq_profile_probe_replicas_resolved = (
        summary.full_probe_replicas if args.dq_profile_level == "full" else summary.standard_probe_replicas
    )
    if args.dq_profile_protocol != "v1":
        args.dq_profile_probe_replicas_resolved = max(2, int(args.dq_profile_probe_replicas_resolved))
    payload = summary.to_dict()
    payload["v2"] = {
        "protocol": args.dq_profile_protocol,
        "range_muls": list(args.dq_profile_range_muls_resolved),
        "sweep_steps": int(args.dq_profile_sweep_steps),
        "branch_repeats": int(args.dq_profile_branch_repeats),
        "capture_steps": int(args.dq_profile_capture_steps),
    }
    if args.dq_profile_protocol == "v2-prefix-smoke":
        args.dq_profile_probe_replicas_resolved = 0
        snapshot_only = bool(getattr(args, "dq_profile_snapshot_only", False))
        branch_steps = 0 if snapshot_only else 2 * 64 * 2 + 2 * 128
        estimated = int(summary.dq_begin_step) + branch_steps
        payload.update(
            {
                "branch_steps": 0 if snapshot_only else 128,
                "standard_probe_replicas": 0,
                "full_probe_replicas": 0,
                "estimated_standard_steps": estimated,
                "estimated_full_steps": estimated,
                "estimated_standard_epochs": estimated / max(1, summary.steps_per_epoch),
                "estimated_full_epochs": estimated / max(1, summary.steps_per_epoch),
            }
        )
        payload["v2"]["cost_components"] = {
            "warmup_steps": int(summary.dq_begin_step),
            "prefix_branch_steps": branch_steps,
            "structural_probe_steps": 0,
        }
        payload["v2"]["snapshot_only"] = snapshot_only
        args.dq_profile_preflight = payload
        return payload
    if args.dq_profile_protocol in {
        "v23-safety-local",
        "v23-safety-formal",
        "v24-acceptance-local",
        "v24-acceptance-formal",
        "v24-trajectory-descriptive",
    }:
        local_only = args.dq_profile_protocol.endswith("-local")
        v24 = args.dq_profile_protocol.startswith("v24-")
        image_count = int(summary.probe_images) if v24 else 16 if local_only else 32
        no_quant_replicas = 3 if local_only else 5
        candidate_noise_replicas = 2
        quant_repeats = 2
        grid_count = len(args.dq_profile_range_muls_resolved)
        no_quant_probe_steps = image_count * 4 * no_quant_replicas
        candidate_probe_steps = (
            image_count
            * 4
            * candidate_noise_replicas
            * grid_count
            * quant_repeats
        )
        branch_steps = (
            0
            if local_only
            else int(args.dq_profile_branch_repeats)
            * (1 + grid_count)
            * int(args.dq_profile_sweep_steps)
        )
        estimated = (
            int(summary.dq_begin_step)
            + no_quant_probe_steps
            + candidate_probe_steps
            + branch_steps
        )
        args.dq_profile_probe_replicas_resolved = no_quant_replicas
        payload.update(
            {
                "branch_steps": 0 if local_only else 128,
                "standard_probe_replicas": no_quant_replicas,
                "full_probe_replicas": no_quant_replicas,
                "estimated_standard_steps": estimated,
                "estimated_full_steps": estimated,
                "estimated_standard_epochs": estimated
                / max(1, summary.steps_per_epoch),
                "estimated_full_epochs": estimated
                / max(1, summary.steps_per_epoch),
            }
        )
        payload["v2"]["cost_components"] = {
            "warmup_steps": int(summary.dq_begin_step),
            "no_quant_structural_steps": no_quant_probe_steps,
            "candidate_local_steps": candidate_probe_steps,
            "common_skip_branch_steps": branch_steps,
            "geometry_steps": 0,
        }
        payload["v2"]["safety_stage"] = "local" if local_only else "formal"
        args.dq_profile_preflight = payload
        return payload
    if args.dq_profile_protocol == "v2-tail-calibration":
        args.dq_profile_probe_replicas_resolved = 5
        no_quant_probe_steps = 32 * 4 * 5
        candidate_probe_steps = 32 * 4 * 2 * 3 * 2
        branch_steps = 5 * 4 * 128
        estimated = int(summary.dq_begin_step) + no_quant_probe_steps + candidate_probe_steps + branch_steps
        payload.update(
            {
                "branch_steps": 128,
                "standard_probe_replicas": 5,
                "full_probe_replicas": 5,
                "estimated_standard_steps": estimated,
                "estimated_full_steps": estimated,
                "estimated_standard_epochs": estimated / max(1, summary.steps_per_epoch),
                "estimated_full_epochs": estimated / max(1, summary.steps_per_epoch),
            }
        )
        payload["v2"]["cost_components"] = {
            "warmup_steps": int(summary.dq_begin_step),
            "no_quant_structural_steps": no_quant_probe_steps,
            "candidate_tail_steps": candidate_probe_steps,
            "common_skip_branch_steps": branch_steps,
        }
        args.dq_profile_preflight = payload
        return payload
    if args.dq_profile_protocol != "v1":
        grid_count = len(args.dq_profile_range_muls_resolved)
        representative_gradient_candidates = min(3, grid_count)
        branch_repeats = int(args.dq_profile_branch_repeats)
        sweep_steps = int(args.dq_profile_sweep_steps)
        common_sweep_steps = branch_repeats * (1 + grid_count) * sweep_steps
        native_confirmation_steps = (
            0
            if args.dq_profile_guardian_ablation == "common_only"
            else branch_repeats * min(3, grid_count) * sweep_steps
        )
        mechanism_steps = (
            branch_repeats * (1 + 3 * len(args.dq_profile_mechanism_muls_resolved)) * 64
            if args.dq_profile_protocol == "v2-mechanism" else 0
        )

        def v2_cost(probe_replicas: int) -> int:
            structural_points = summary.probe_points_per_replica * int(probe_replicas)
            structural_backward_steps = structural_points * (1 + representative_gradient_candidates)
            return (
                int(summary.dq_begin_step)
                + structural_backward_steps
                + common_sweep_steps
                + native_confirmation_steps
                + mechanism_steps
            )

        standard_replicas = 2
        full_replicas = max(2, int(summary.full_probe_replicas))
        standard_steps = v2_cost(standard_replicas)
        full_budget_steps = min(standard_steps * 2, int(summary.normal_training_steps * 0.75))
        fixed_steps = v2_cost(0)
        per_replica_steps = summary.probe_points_per_replica * (1 + representative_gradient_candidates)
        maximum_budget_replicas = max(
            standard_replicas,
            (full_budget_steps - fixed_steps) // max(1, per_replica_steps),
        )
        full_replicas = min(full_replicas, int(maximum_budget_replicas))
        full_steps = v2_cost(full_replicas)
        args.dq_profile_probe_replicas_resolved = (
            full_replicas if args.dq_profile_level == "full" else standard_replicas
        )
        conditional_extra = {
            "third_repeat": (1 + grid_count + min(3, grid_count)) * sweep_steps,
            "edge_extension_one_side": branch_repeats * 2 * sweep_steps,
            "selected_128_extension": branch_repeats * 4 * 64,
        }
        payload.update(
            {
                "branch_steps": sweep_steps,
                "standard_probe_replicas": standard_replicas,
                "full_probe_replicas": full_replicas,
                "estimated_standard_steps": standard_steps,
                "estimated_full_steps": full_steps,
                "estimated_standard_epochs": standard_steps / max(1, summary.steps_per_epoch),
                "estimated_full_epochs": full_steps / max(1, summary.steps_per_epoch),
                "full_budget_steps": full_budget_steps,
                "full_budget_core_exceeded": full_steps > full_budget_steps,
            }
        )
        payload["v2"]["cost_components"] = {
            "warmup_steps": int(summary.dq_begin_step),
            "representative_parameter_gradient_candidates": representative_gradient_candidates,
            "common_sweep_steps": common_sweep_steps,
            "native_confirmation_steps": native_confirmation_steps,
            "mechanism_steps": mechanism_steps,
            "conditional_extra_steps": conditional_extra,
            "full_budget_excludes_conditional_extras": True,
            "full_probe_replicas_reduced_for_budget": full_replicas < max(2, int(summary.full_probe_replicas)),
        }
    args.dq_profile_preflight = payload
    return payload


def _configure_prefix_kernel_policy(args: argparse.Namespace) -> dict[str, Any]:
    mode = str(getattr(args, "dq_profile_prefix_kernel_mode", "deterministic"))
    deterministic_protocols = {
        "v2-prefix-smoke",
        "v2-tail-calibration",
        "v23-safety-local",
        "v23-safety-formal",
        "v24-acceptance-local",
        "v24-acceptance-formal",
        "v24-trajectory-descriptive",
    }
    enabled = args.dq_profile_protocol in deterministic_protocols and mode == "deterministic"
    payload: dict[str, Any] = {
        "mode": mode,
        "protocol": str(args.dq_profile_protocol),
        "enabled": enabled,
        "cublas_workspace_config": None,
        "torch_deterministic_algorithms": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": None,
        "cuda_matmul_allow_tf32": None,
        "cudnn_allow_tf32": None,
    }
    if not enabled:
        args.dq_profile_prefix_kernel_policy = payload
        return payload

    # Prefix smoke validates snapshot/restore and max-step independence.  The
    # gated tail/local/formal processes must use the same policy as the prefix;
    # otherwise separate warmups can silently produce different snapshots.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    payload.update(
        {
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_matmul_allow_tf32": bool(
                torch.backends.cuda.matmul.allow_tf32
            ),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
    )
    args.dq_profile_prefix_kernel_policy = payload
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = setup_parser()
    args = parser.parse_args(argv)
    if getattr(args, "output_config", False):
        raise ValueError("--output_config is not supported because the profiler writes only inside its diagnostic run directory")
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    # Apply the deterministic CUDA/cuBLAS policy before validation builds a
    # source manifest and queries the GPU.  Local/formal runs are separate
    # processes, so late configuration can invalidate their snapshot parity.
    _configure_prefix_kernel_policy(args)
    _validate_and_isolate(args)
    preflight = _preflight(args)
    configure_sdxl_globals(args)

    if args.dq_profile_dry_run:
        print(
            json.dumps(
                {
                    "preflight": preflight,
                    "profile_level": args.dq_profile_level,
                    "profile_protocol": args.dq_profile_protocol,
                    "selected_probe_replicas": args.dq_profile_probe_replicas_resolved,
                    "estimated_selected_steps": preflight[
                        "estimated_full_steps" if args.dq_profile_level == "full" else "estimated_standard_steps"
                    ],
                    "run_dir": args.dq_profile_run_dir,
                    "forced_network_module": args.network_module,
                    "quant_rng_mode": args.dq_profile_quant_rng_mode,
                    "mechanism_muls": list(args.dq_profile_mechanism_muls_resolved),
                    "candidates": [
                        candidate.to_dict()
                        for candidate in _resolved_candidate_definitions(args)
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    artifacts = ProfileArtifacts(args.dq_profile_run_dir)
    artifacts.initialize()
    artifacts.ensure_known_result()
    args.dq_profile_execution_log_path = str(artifacts.root / "execution.log")
    trainer = None
    try:
        trainer = SdxlNetworkTrainer()
        trainer.train(args)
    except BaseException as error:
        artifacts.mark_failed(error)
        raise
    finally:
        execution_handler = None if trainer is None else getattr(trainer, "_dq_profile_execution_log_handler", None)
        if execution_handler is not None:
            logging.getLogger().removeHandler(execution_handler)
            execution_handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
