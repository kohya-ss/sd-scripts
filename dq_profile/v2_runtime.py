from __future__ import annotations

import itertools
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from dq_profile.protocol import CandidateDefinition, fixed_range_candidates, mechanism_candidates
from dq_profile.replay import ReplayBatch, ReplaySequence, seed_step_rng
from dq_profile.snapshot import TrainingSnapshot
from dq_profile.v2_calibration import (
    aggregate_prefix_gate,
    capture_state_bundle,
    evaluate_prefix_pair,
)
from dq_profile.v2_metrics import (
    DIAGNOSTIC_TARGET,
    ParameterDelta,
    compare_parameter_deltas,
    guardian_dependence,
    mechanism_interaction,
    summarize_stability,
)
from library import train_util


@dataclass
class BranchExecution:
    candidate: CandidateDefinition
    repeat: int
    guardian_mode: str
    max_steps: int
    execution_id: str = ""
    cohort_id: str = ""
    phase: str = "v2_core"
    reference_execution_id: Optional[str] = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    deltas: dict[int, ParameterDelta] = field(default_factory=dict)
    skip_mask: list[bool] = field(default_factory=list)
    gradient_norms: list[float] = field(default_factory=list)
    forced_safety_abort: bool = False
    invalid_reason: Optional[str] = None
    last_safe_step: int = 0
    state_values: dict[int, dict[str, Any]] = field(default_factory=dict)
    state_records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def actual_steps(self) -> int:
        return len(self.rows)

    @property
    def actual_checkpoint(self) -> int:
        return max(self.deltas, default=0)


def _candidate_for_mul(value: float, *, mechanism: str = "full") -> CandidateDefinition:
    suffix = "" if mechanism == "full" else f"__{mechanism}"
    return CandidateDefinition(
        name=f"mul_{float(value):.3f}{suffix}",
        quantized=True,
        clip_low=None,
        clip_high=None,
        initial_range_mul=float(value),
        auto_enabled=False,
        mechanism=mechanism,
    )


def _no_quant_candidate() -> CandidateDefinition:
    return CandidateDefinition("no_quant", False, None, None, None, False)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


class V2ExperimentRunner:
    """Execute fixed-range branches from a single immutable snapshot/replay."""

    def __init__(
        self,
        *,
        runtime: Any,
        sequence: ReplaySequence,
        snapshot: TrainingSnapshot,
        pass_context: Mapping[str, Any],
    ) -> None:
        self.runtime = runtime
        self.args = runtime.args
        self.sequence = sequence
        self.snapshot = snapshot
        self.pass_context = dict(pass_context)
        self.accelerator = self.pass_context["accelerator"]
        self.network = self.pass_context["network"]
        self.optimizer = self.pass_context["optimizer"]
        self.scheduler = self.pass_context["lr_scheduler"]
        self.guardian = self.pass_context["grad_norm_guardian"]
        self.noise_scheduler = self.pass_context["noise_scheduler"]
        self.unwrapped = self.accelerator.unwrap_model(self.network)
        self._repeat_sequences: dict[int, ReplaySequence] = {}
        # Run length and cohort are identity, not cache reuse hints.  A freshly
        # restored 128-step execution must never replace a 64-step cohort.
        self._executions: dict[tuple[str, str, str, int, str, int], BranchExecution] = {}

    @staticmethod
    def _execution_id(
        *,
        cohort_id: str,
        phase: str,
        guardian_mode: str,
        candidate: CandidateDefinition,
        repeat: int,
        max_steps: int,
    ) -> str:
        return (
            f"{cohort_id}__{phase}__{guardian_mode}__{candidate.name}"
            f"__repeat_{int(repeat)}__steps_{int(max_steps)}"
        )

    def _restore(self) -> None:
        self.snapshot.restore(
            network=self.unwrapped,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=getattr(self.accelerator, "scaler", None),
            trainer=self.runtime.trainer,
            guardian=self.guardian,
        )

    def _capture_prefix_state(self, execution: BranchExecution, checkpoint: int) -> None:
        bundle = capture_state_bundle(
            network=self.unwrapped,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=getattr(self.accelerator, "scaler", None),
            trainer=self.runtime.trainer,
            guardian=self.guardian,
        )
        execution.state_values[int(checkpoint)] = bundle
        for component, fingerprint in sorted(bundle["fingerprints"].items()):
            execution.state_records.append(
                {
                    "execution_id": execution.execution_id,
                    "reference_execution_id": execution.reference_execution_id,
                    "cohort_id": execution.cohort_id,
                    "phase": execution.phase,
                    "candidate": execution.candidate.name,
                    "repeat": execution.repeat,
                    "requested_max_steps": execution.max_steps,
                    "actual_checkpoint": int(checkpoint),
                    "component": component,
                    "sha256": fingerprint,
                }
            )

    def _repeat_sequence(self, repeat: int) -> ReplaySequence:
        cached = self._repeat_sequences.get(int(repeat))
        if cached is not None:
            return cached
        result = ReplaySequence()
        weight_dtype = self.pass_context["weight_dtype"]
        device = self.accelerator.device
        for source in self.sequence:
            model_seed = seed_step_rng(
                self.runtime.protocol_seed,
                source.index,
                phase="v2_materialize",
                repeat=int(repeat),
            )
            latents = source.latents.to(device=device, dtype=weight_dtype)
            progress = source.global_step / float(max(1, self.args.max_train_steps))
            noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                self.args,
                self.noise_scheduler,
                latents,
                progress_frac=max(0.0, min(1.0, progress)),
            )
            target = (
                self.noise_scheduler.get_velocity(latents, noise, timesteps)
                if self.args.v_parameterization
                else noise
            )
            result.append(
                ReplayBatch(
                    index=source.index,
                    source_epoch=source.source_epoch,
                    source_step=source.source_step,
                    global_step=source.global_step,
                    batch=source.batch,
                    latents=latents,
                    noise=noise,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    target=target,
                    huber_c=huber_c,
                    model_seed=model_seed,
                )
            )
        result.seal()
        self._repeat_sequences[int(repeat)] = result
        return result

    def _execute(
        self,
        *,
        candidate: CandidateDefinition,
        repeat: int,
        guardian_mode: str,
        max_steps: int,
        reference: Optional[BranchExecution],
        quant_phase: str = "v2_core",
        cohort_id: Optional[str] = None,
        capture_prefix_state: bool = False,
    ) -> BranchExecution:
        if guardian_mode not in {"common_skip", "native_guardian"}:
            raise ValueError(f"unknown Guardian mode: {guardian_mode}")
        cohort_id = str(cohort_id or f"{quant_phase}.steps_{int(max_steps)}.repeat_{int(repeat)}")
        key = (cohort_id, guardian_mode, candidate.name, int(repeat), quant_phase, int(max_steps))
        existing = self._executions.get(key)
        if existing is not None:
            return existing
        if candidate.quantized and guardian_mode == "common_skip" and reference is None:
            raise ValueError("common-skip quantized branch requires a no-quant reference")
        if reference is not None and reference.max_steps < int(max_steps):
            raise ValueError("no-quant reference is shorter than the requested candidate branch")
        if reference is not None and (
            reference.cohort_id != cohort_id
            or reference.repeat != int(repeat)
            or reference.phase != quant_phase
            or reference.max_steps != int(max_steps)
        ):
            raise ValueError(
                "reference_mismatch: candidate/reference must have identical "
                "cohort, repeat, phase, and requested max steps"
            )

        self._restore()
        execution_id = self._execution_id(
            cohort_id=cohort_id,
            phase=quant_phase,
            guardian_mode=guardian_mode,
            candidate=candidate,
            repeat=int(repeat),
            max_steps=int(max_steps),
        )
        execution = BranchExecution(
            candidate,
            int(repeat),
            guardian_mode,
            int(max_steps),
            execution_id=execution_id,
            cohort_id=cohort_id,
            phase=quant_phase,
            reference_execution_id=None if reference is None else reference.execution_id,
        )
        if capture_prefix_state:
            self._capture_prefix_state(execution, 0)
        checkpoints = {value for value in (32, 64, 128, int(max_steps)) if 0 < value <= int(max_steps)}
        replay_sequence = self._repeat_sequence(int(repeat))
        for branch_step, replay in enumerate(replay_sequence):
            if branch_step >= int(max_steps):
                break
            seed_step_rng(
                self.runtime.protocol_seed,
                branch_step,
                phase="v2_training_model",
                repeat=int(repeat),
            )
            forced_skip: Optional[bool] = None
            matched_gradient_norm: Optional[float] = None
            if candidate.quantized and guardian_mode == "common_skip":
                if reference is None or branch_step >= len(reference.skip_mask):
                    execution.forced_safety_abort = True
                    execution.invalid_reason = "no_quant_reference_ended_before_candidate"
                    break
                forced_skip = bool(reference.skip_mask[branch_step])
                matched_gradient_norm = float(reference.gradient_norms[branch_step])

            absolute_step = int(self.snapshot.metadata["global_step"]) + branch_step
            row, _, _ = self.runtime._run_pass(
                replay=replay,
                candidate=candidate,
                range_mul=candidate.initial_range_mul,
                phase=quant_phase,
                probe_or_step=branch_step,
                repeat=int(repeat),
                dropout_enabled=True,
                shadow=False,
                update=True,
                do_auto_observation=False,
                absolute_step=absolute_step,
                epoch=int(self.snapshot.metadata["epoch"]),
                forced_skip=forced_skip,
                matched_no_quant_gradient_norm=matched_gradient_norm,
                hard_safety=True,
                **self.pass_context,
            )
            row.update(
                {
                    "branch_step": branch_step,
                    "absolute_step": absolute_step + 1,
                    "image_keys": "|".join(replay.image_keys),
                    "replay_digest": replay.digest,
                    "branch_regime": "training_dropout_on",
                    "guardian_mode": guardian_mode,
                    "diagnostic_target": DIAGNOSTIC_TARGET,
                    "execution_id": execution.execution_id,
                    "reference_execution_id": execution.reference_execution_id,
                    "cohort_id": execution.cohort_id,
                    "requested_max_steps": execution.max_steps,
                    "actual_checkpoint": branch_step + 1,
                }
            )
            execution.rows.append(row)
            execution.skip_mask.append(bool(row["update_skipped"]))
            execution.gradient_norms.append(float(row["gradient_norm"]))
            if bool(row.get("forced_safety_abort", False)):
                execution.forced_safety_abort = True
                execution.invalid_reason = str(row.get("invalid_reason") or "hard_safety_abort")
                row["last_safe_step"] = execution.last_safe_step
                break
            execution.last_safe_step = branch_step + 1
            row["last_safe_step"] = execution.last_safe_step
            if branch_step + 1 in checkpoints:
                execution.deltas[branch_step + 1] = ParameterDelta.capture(
                    self.unwrapped,
                    self.snapshot.network_state,
                    parameter_names=(
                        name
                        for name, enabled in self.snapshot.network_runtime.get("requires_grad", {}).items()
                        if bool(enabled)
                    ),
                )

            if capture_prefix_state and branch_step + 1 in {1, 32, 64}:
                self._capture_prefix_state(execution, branch_step + 1)
        if len(execution.rows) < int(max_steps) and not execution.forced_safety_abort:
            execution.forced_safety_abort = True
            execution.invalid_reason = "replay_sequence_shorter_than_requested_branch"
        for row in execution.rows:
            row["actual_steps"] = execution.actual_steps
        self._executions[key] = execution
        return execution

    def _reference(
        self,
        repeat: int,
        max_steps: int,
        *,
        quant_phase: str = "v2_core",
        cohort_id: Optional[str] = None,
        capture_prefix_state: bool = False,
    ) -> BranchExecution:
        return self._execute(
            candidate=_no_quant_candidate(),
            repeat=repeat,
            guardian_mode="native_guardian",
            max_steps=max_steps,
            reference=None,
            quant_phase=quant_phase,
            cohort_id=cohort_id,
            capture_prefix_state=capture_prefix_state,
        )

    @staticmethod
    def _direction_rows(
        execution: BranchExecution,
        reference: BranchExecution,
        *,
        guardian_mode: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        provenance = {
            "execution_id": execution.execution_id,
            "reference_execution_id": reference.execution_id,
            "cohort_id": execution.cohort_id,
            "phase": execution.phase,
            "requested_max_steps": execution.max_steps,
            "actual_steps": execution.actual_steps,
        }
        reference_matches = (
            execution.cohort_id == reference.cohort_id
            and execution.repeat == reference.repeat
            and execution.phase == reference.phase
            and execution.max_steps == reference.max_steps
        )
        if not reference_matches:
            return [
                {
                    **provenance,
                    "candidate": execution.candidate.name,
                    "range_mul": execution.candidate.initial_range_mul,
                    "mechanism": execution.candidate.mechanism,
                    "repeat": execution.repeat,
                    "checkpoint": None,
                    "actual_checkpoint": execution.actual_checkpoint,
                    "guardian_mode": guardian_mode,
                    "module_group": "all",
                    "update_direction_valid": False,
                    "invalid_reason": "reference_mismatch",
                    "forced_safety_abort": execution.forced_safety_abort,
                    "last_safe_step": execution.last_safe_step,
                }
            ]
        checkpoints = sorted(set(reference.deltas).union(execution.deltas))
        for checkpoint in checkpoints:
            reference_delta = reference.deltas.get(checkpoint)
            candidate_delta = execution.deltas.get(checkpoint)
            if reference_delta is None or candidate_delta is None:
                rows.append(
                    {
                        **provenance,
                        "candidate": execution.candidate.name,
                        "range_mul": execution.candidate.initial_range_mul,
                        "mechanism": execution.candidate.mechanism,
                        "repeat": execution.repeat,
                        "checkpoint": checkpoint,
                        "actual_checkpoint": checkpoint,
                        "guardian_mode": guardian_mode,
                        "module_group": "all",
                        "update_direction_valid": False,
                        "invalid_reason": execution.invalid_reason or "checkpoint_not_reached",
                        "forced_safety_abort": execution.forced_safety_abort,
                        "last_safe_step": execution.last_safe_step,
                    }
                )
                continue
            for comparison in compare_parameter_deltas(reference_delta, candidate_delta):
                rows.append(
                    {
                        **comparison,
                        **provenance,
                        "candidate": execution.candidate.name,
                        "range_mul": execution.candidate.initial_range_mul,
                        "mechanism": execution.candidate.mechanism,
                        "repeat": execution.repeat,
                        "checkpoint": checkpoint,
                        "actual_checkpoint": checkpoint,
                        "guardian_mode": guardian_mode,
                        "forced_safety_abort": execution.forced_safety_abort,
                        "last_safe_step": execution.last_safe_step,
                        "common_skip_matched": (
                            None
                            if guardian_mode == "native_guardian"
                            else bool(
                                not execution.forced_safety_abort
                                and all(row.get("common_skip_matched") is True for row in execution.rows)
                            )
                        ),
                    }
                )
        return rows

    @staticmethod
    def _anchor(rows: Sequence[Mapping[str, Any]], grid: Sequence[float], guardian_mode: str) -> Optional[float]:
        endpoint = [
            row
            for row in rows
            if int(row.get("checkpoint", -1)) == 64
            and str(row.get("guardian_mode")) == guardian_mode
            and str(row.get("module_group")) == "all"
            and bool(row.get("update_direction_valid", False))
            and not bool(row.get("forced_safety_abort", False))
        ]
        grouped: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in endpoint:
            mul = row.get("range_mul")
            if not _finite(mul):
                continue
            for metric in ("orthogonal_drift", "total_drift"):
                if _finite(row.get(metric)):
                    grouped[float(mul)][metric].append(float(row[metric]))
        medians: dict[float, dict[str, float]] = {
            mul: {metric: float(statistics.median(values)) for metric, values in metrics.items() if values}
            for mul, metrics in grouped.items()
        }
        medians = {mul: values for mul, values in medians.items() if len(values) == 2}
        if not medians:
            return None
        normalized: dict[str, dict[float, float]] = {}
        for metric in ("orthogonal_drift", "total_drift"):
            values = {mul: metrics[metric] for mul, metrics in medians.items()}
            lo, hi = min(values.values()), max(values.values())
            scale = max(hi - lo, 1e-12)
            normalized[metric] = {mul: (value - lo) / scale for mul, value in values.items()}
        center = statistics.mean(sorted(float(value) for value in grid))
        return min(
            medians,
            key=lambda mul: (
                max(normalized["orthogonal_drift"][mul], normalized["total_drift"][mul]),
                abs(mul - center),
                mul,
            ),
        )

    def _collect_direction_rows(
        self,
        executions: Sequence[BranchExecution],
        references: Mapping[int, BranchExecution],
        *,
        guardian_mode: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for execution in executions:
            rows.extend(
                self._direction_rows(
                    execution,
                    references[execution.repeat],
                    guardian_mode=guardian_mode,
                )
            )
        return rows

    def _run_common_candidates(
        self,
        candidates: Sequence[CandidateDefinition],
        repeats: Sequence[int],
        max_steps: int,
        *,
        quant_phase: str = "v2_core",
        cohort_label: Optional[str] = None,
    ) -> tuple[dict[int, BranchExecution], list[BranchExecution]]:
        cohort_ids = {
            repeat: f"{cohort_label or quant_phase}.repeat_{int(repeat)}.steps_{int(max_steps)}"
            for repeat in repeats
        }
        references = {
            repeat: self._reference(
                repeat,
                max_steps,
                quant_phase=quant_phase,
                cohort_id=cohort_ids[repeat],
            )
            for repeat in repeats
        }
        executions = [
            self._execute(
                candidate=candidate,
                repeat=repeat,
                guardian_mode="common_skip",
                max_steps=max_steps,
                reference=references[repeat],
                quant_phase=quant_phase,
                cohort_id=cohort_ids[repeat],
            )
            for repeat in repeats
            for candidate in candidates
        ]
        return references, executions

    @staticmethod
    def _edge_extensions(summary: Mapping[str, Any], grid: Sequence[float]) -> list[float]:
        ordered = sorted(float(value) for value in grid)
        selected = {summary.get("m_dir"), summary.get("m_total")}
        result: list[float] = []
        if ordered and ordered[0] in selected:
            result.extend((2.55, 2.40))
        if ordered and ordered[-1] in selected:
            result.extend((3.60, 3.75))
        return [value for value in result if value not in ordered]

    @staticmethod
    def _neighbor_candidates(anchor: Optional[float], grid: Sequence[float]) -> list[CandidateDefinition]:
        if anchor is None:
            return []
        ordered = sorted(float(value) for value in grid)
        index = ordered.index(float(anchor))
        values = ordered[max(0, index - 1) : min(len(ordered), index + 2)]
        return [_candidate_for_mul(value) for value in values]

    def _native_confirmation(
        self,
        *,
        common_rows: Sequence[Mapping[str, Any]],
        intrinsic: Mapping[str, Any],
        grid: Sequence[float],
        repeats: Sequence[int],
        max_steps: int,
        references: Mapping[int, BranchExecution],
    ) -> tuple[list[BranchExecution], list[dict[str, Any]], dict[str, Any], Optional[float]]:
        anchor = self._anchor(common_rows, grid, "common_skip")
        candidates = self._neighbor_candidates(anchor, grid)
        tested = {float(candidate.initial_range_mul) for candidate in candidates}
        executions: list[BranchExecution] = []

        def run_missing() -> None:
            known = {(item.candidate.name, item.repeat) for item in executions}
            for repeat in repeats:
                for candidate in candidates:
                    if (candidate.name, repeat) in known:
                        continue
                    executions.append(
                        self._execute(
                            candidate=candidate,
                            repeat=repeat,
                            guardian_mode="native_guardian",
                            max_steps=max_steps,
                            reference=references[repeat],
                            cohort_id=references[repeat].cohort_id,
                        )
                    )

        run_missing()
        rows = self._collect_direction_rows(executions, references, guardian_mode="native_guardian")
        native = summarize_stability(rows, grid=grid, checkpoint=64, guardian_mode="native_guardian")
        ordered = sorted(float(value) for value in grid)
        for _ in range(len(ordered)):
            optima = [value for value in (native.get("m_dir"), native.get("m_total")) if _finite(value)]
            additions: list[float] = []
            for optimum in optima:
                index = ordered.index(float(optimum))
                if float(optimum) == min(tested) and index > 0 and ordered[index - 1] not in tested:
                    additions.append(ordered[index - 1])
                if float(optimum) == max(tested) and index + 1 < len(ordered) and ordered[index + 1] not in tested:
                    additions.append(ordered[index + 1])
            if not additions:
                break
            for value in sorted(set(additions)):
                candidates.append(_candidate_for_mul(value))
                tested.add(value)
            run_missing()
            rows = self._collect_direction_rows(executions, references, guardian_mode="native_guardian")
            native = summarize_stability(rows, grid=grid, checkpoint=64, guardian_mode="native_guardian")
        return executions, rows, native, anchor

    @staticmethod
    def _summarize_candidates(
        executions: Sequence[BranchExecution],
        direction_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        by_candidate: dict[str, list[BranchExecution]] = defaultdict(list)
        for execution in executions:
            by_candidate[execution.candidate.name].append(execution)
        for name, candidate_executions in by_candidate.items():
            trajectory = [row for execution in candidate_executions for row in execution.rows]
            endpoint = [
                row
                for row in direction_rows
                if row.get("candidate") == name
                and int(row.get("checkpoint", -1)) == 64
                and row.get("module_group") == "all"
                and bool(row.get("update_direction_valid", False))
            ]
            candidate = candidate_executions[0].candidate
            payload: dict[str, Any] = {
                "candidate": name,
                "initial_range_mul": candidate.initial_range_mul,
                "final_range_mul": candidate.initial_range_mul,
                "mechanism": candidate.mechanism,
                "branch_loss_mean": (
                    float(statistics.mean(float(row["loss"]) for row in trajectory)) if trajectory else None
                ),
                "branch_loss_std": (
                    float(statistics.pstdev(float(row["loss"]) for row in trajectory)) if len(trajectory) > 1 else 0.0
                ),
                "branch_steps": max((len(execution.rows) for execution in candidate_executions), default=0),
                "branch_repeats": len(candidate_executions),
                "branch_regime": "training_dropout_on",
                "guardian_mode": candidate_executions[0].guardian_mode,
                "native_would_skip_count": sum(
                    int(bool(row.get("native_would_skip", False))) for row in trajectory
                ),
                "forced_safety_abort": any(item.forced_safety_abort for item in candidate_executions),
                "invalid_reason": next(
                    (item.invalid_reason for item in candidate_executions if item.invalid_reason),
                    None,
                ),
                "auto_observation_count": 0,
                "auto_post_warmup_observation_count": 0,
                "auto_warmup_completed": None,
                "auto_trajectory_metrics_valid": False,
                "auto_invalid_reason": "fixed_range_auto_disabled",
            }
            for metric in (
                "update_cosine",
                "projection_gain",
                "orthogonal_drift",
                "total_drift",
                "update_norm_ratio",
                "no_quant_update_norm",
                "candidate_update_norm",
            ):
                values = [float(row[metric]) for row in endpoint if _finite(row.get(metric))]
                payload[f"checkpoint64_{metric}_median"] = float(statistics.median(values)) if values else None
            result[name] = payload
        return result

    @staticmethod
    def _range_sweep_rows(
        executions: Sequence[BranchExecution],
        direction_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        lookup_by_execution_id = {
            item.execution_id: item for item in executions if item.execution_id
        }
        fallback_lookup = {
            (item.candidate.name, item.repeat, item.guardian_mode): item for item in executions
        }
        rows: list[dict[str, Any]] = []
        for row in direction_rows:
            if row.get("module_group") != "all":
                continue
            execution = lookup_by_execution_id.get(str(row.get("execution_id", "")))
            if execution is None:
                execution = fallback_lookup.get(
                    (row["candidate"], int(row["repeat"]), str(row["guardian_mode"]))
                )
            checkpoint_value = row.get("checkpoint")
            checkpoint = None if checkpoint_value is None else int(checkpoint_value)
            trajectory = (
                []
                if execution is None or checkpoint is None
                else execution.rows[:checkpoint]
            )
            rows.append(
                {
                    **dict(row),
                    "loss_mean_to_checkpoint": (
                        float(statistics.mean(float(item["loss"]) for item in trajectory)) if trajectory else None
                    ),
                    "gradient_norm_mean_to_checkpoint": (
                        float(statistics.mean(float(item["gradient_norm"]) for item in trajectory))
                        if trajectory
                        else None
                    ),
                    "optimizer_updates_to_checkpoint": sum(
                        int(bool(item.get("optimizer_step_performed", False))) for item in trajectory
                    ),
                    "native_would_skip_to_checkpoint": sum(
                        int(bool(item.get("native_would_skip", False))) for item in trajectory
                    ),
                }
            )
        return rows

    def _mechanism(
        self,
        *,
        selected_muls: Sequence[float],
        repeats: Sequence[int],
        max_steps: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        if not selected_muls:
            return [], [], {"valid": False, "invalid_reason": "no_core_gate_approved_mechanism_range"}
        references = {
            repeat: self._reference(repeat, max_steps, quant_phase="v2_mechanism") for repeat in repeats
        }
        executions: list[BranchExecution] = []
        direction_rows: list[dict[str, Any]] = []
        for repeat in repeats:
            reference = references[repeat]
            direction_rows.extend(
                self._direction_rows(reference, reference, guardian_mode="common_skip")
            )
            for selected_mul in selected_muls:
                for candidate in mechanism_candidates(float(selected_mul))[1:]:
                    execution = self._execute(
                        candidate=candidate,
                        repeat=repeat,
                        guardian_mode="common_skip",
                        max_steps=max_steps,
                        reference=reference,
                        quant_phase="v2_mechanism",
                        cohort_id=reference.cohort_id,
                    )
                    executions.append(execution)
                    direction_rows.extend(
                        self._direction_rows(execution, reference, guardian_mode="common_skip")
                    )

        output_rows = list(direction_rows)
        reference_by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
        by_key: dict[tuple[int, int, str, float], dict[str, Mapping[str, Any]]] = defaultdict(dict)
        for row in direction_rows:
            if not bool(row.get("update_direction_valid", False)):
                continue
            base_key = (int(row["repeat"]), int(row["checkpoint"]), str(row["module_group"]))
            if row["candidate"] == "no_quant":
                reference_by_key[base_key] = row
                continue
            key = (*base_key, float(row["range_mul"]))
            by_key[key][str(row.get("mechanism", "full"))] = row
        interaction_count = 0
        metrics = ("update_cosine", "projection_gain", "orthogonal_drift", "total_drift", "update_norm_ratio")
        for (repeat, checkpoint, module_group, selected_mul), values in sorted(by_key.items()):
            no_quant = reference_by_key.get((repeat, checkpoint, module_group))
            if no_quant is None or not {"full", "clip_only", "round_only"}.issubset(values):
                continue
            interaction_row: dict[str, Any] = {
                "execution_id": (
                    f"derived_interaction__{no_quant.get('cohort_id')}__repeat_{repeat}"
                    f"__checkpoint_{checkpoint}__mul_{float(selected_mul):.3f}"
                ),
                "reference_execution_id": no_quant.get("execution_id"),
                "cohort_id": no_quant.get("cohort_id"),
                "phase": "v2_mechanism_interaction",
                "requested_max_steps": no_quant.get("requested_max_steps"),
                "actual_steps": no_quant.get("actual_steps"),
                "actual_checkpoint": checkpoint,
                "candidate": "clip_round_interaction",
                "range_mul": float(selected_mul),
                "mechanism": "interaction",
                "repeat": repeat,
                "checkpoint": checkpoint,
                "guardian_mode": "common_skip",
                "module_group": module_group,
                "update_direction_valid": True,
            }
            for metric in metrics:
                interaction_row[f"interaction_{metric}"] = mechanism_interaction(
                    no_quant=float(no_quant[metric]),
                    full=float(values["full"][metric]),
                    clip_only=float(values["clip_only"][metric]),
                    round_only=float(values["round_only"][metric]),
                )
            output_rows.append(interaction_row)
            interaction_count += 1
        trajectory = [row for execution in executions for row in execution.rows]
        return output_rows, trajectory, {
            "valid": interaction_count > 0,
            "selected_muls": [float(value) for value in selected_muls],
            "definition": {
                "clip_only": "x + (x_clamped - x)",
                "round_only": "x + (q - x_clamped)",
                "full": "x + (x_clamped - x) + (q - x_clamped)",
                "interaction": "(full-no_quant)-(clip-no_quant)-(round-no_quant)",
            },
            "interaction_row_count": interaction_count,
            "counterfactual_only": True,
        }

    def _run_tail_calibration(self) -> dict[str, Any]:
        safety_formal = self.runtime.profile_protocol in {
            "v23-safety-formal",
            "v24-acceptance-formal",
            "v24-trajectory-descriptive",
        }
        required_steps = int(self.args.dq_profile_sweep_steps)
        if len(self.sequence) < required_steps:
            raise ValueError(
                f"{self.runtime.profile_protocol} requires at least "
                f"{required_steps} materialized replay batches"
            )
        grid = (
            sorted(
                float(value)
                for value in self.args.dq_profile_range_muls_resolved
            )
            if safety_formal
            else [2.70, 3.15, 3.45]
        )
        candidates = [_candidate_for_mul(value) for value in grid]
        repeats = (
            list(range(int(self.args.dq_profile_branch_repeats)))
            if safety_formal
            else list(range(5))
        )
        quant_phase = (
            self.runtime.profile_protocol.replace("-", "_")
            if safety_formal
            else "v2_tail_calibration"
        )
        cohort_label = quant_phase
        references, executions = self._run_common_candidates(
            candidates,
            repeats,
            required_steps,
            quant_phase=quant_phase,
            cohort_label=cohort_label,
        )
        direction_rows = self._collect_direction_rows(
            executions,
            references,
            guardian_mode="common_skip",
        )
        for row in direction_rows:
            row["result_scope"] = (
                "safety_formal_128" if safety_formal else "tail_calibration_128"
            )
        for execution in executions:
            for row in execution.rows:
                row["result_scope"] = (
                    "safety_formal_128"
                    if safety_formal
                    else "tail_calibration_128"
                )
        for reference in references.values():
            for row in reference.rows:
                row["result_scope"] = (
                    "safety_formal_128"
                    if safety_formal
                    else "tail_calibration_128"
                )

        natural_rows: list[dict[str, Any]] = []
        natural_by_metric: dict[tuple[int, str], list[float]] = defaultdict(list)
        for left_repeat, right_repeat in itertools.combinations(repeats, 2):
            left = references[left_repeat]
            right = references[right_repeat]
            for checkpoint in (32, 64, 128):
                if checkpoint not in left.deltas or checkpoint not in right.deltas:
                    continue
                comparisons = compare_parameter_deltas(
                    left.deltas[checkpoint],
                    right.deltas[checkpoint],
                )
                all_row = next(
                    (row for row in comparisons if row["module_group"] == "all"),
                    None,
                )
                if all_row is None:
                    continue
                payload = {
                    **all_row,
                    "record_type": "no_quant_pair",
                    "candidate": "no_quant",
                    "repeat_a": left_repeat,
                    "repeat_b": right_repeat,
                    "checkpoint": checkpoint,
                    "diagnostic_target": "no_quant_repeat_natural_drift",
                }
                natural_rows.append(payload)
                for metric in ("orthogonal_drift", "total_drift"):
                    if _finite(all_row.get(metric)):
                        natural_by_metric[(checkpoint, metric)].append(float(all_row[metric]))

        calibrated_candidate_rows: list[dict[str, Any]] = []
        for row in direction_rows:
            if row.get("module_group") != "all" or not bool(row.get("update_direction_valid", False)):
                continue
            checkpoint = int(row["checkpoint"])
            payload = {
                **dict(row),
                "record_type": "candidate_vs_matched_no_quant",
                "diagnostic_target": "candidate_drift_null_excess",
            }
            for metric in ("orthogonal_drift", "total_drift"):
                natural = natural_by_metric.get((checkpoint, metric), [])
                if natural and _finite(row.get(metric)):
                    null_median = float(np.median(natural))
                    null_q95 = float(np.quantile(natural, 0.95))
                    value = float(row[metric])
                    payload[f"{metric}_null_median"] = null_median
                    payload[f"{metric}_null_q95"] = null_q95
                    payload[f"{metric}_null_excess"] = value - null_q95
                    payload[f"{metric}_null_percentile"] = statistics.mean(
                        sample <= value for sample in natural
                    )
            calibrated_candidate_rows.append(payload)
        cumulative_null_rows = natural_rows + calibrated_candidate_rows

        endpoint = [
            row
            for row in direction_rows
            if row.get("module_group") == "all"
            and int(row.get("checkpoint", -1)) == 128
            and bool(row.get("update_direction_valid", False))
        ]
        best_by_repeat: dict[int, str] = {}
        for repeat in repeats:
            members = [row for row in endpoint if int(row["repeat"]) == repeat]
            if members:
                winner = min(
                    members,
                    key=lambda row: (float(row["total_drift"]), str(row["candidate"])),
                )
                best_by_repeat[repeat] = str(winner["candidate"])
        winner_counts: dict[str, int] = defaultdict(int)
        for winner in best_by_repeat.values():
            winner_counts[winner] += 1
        winner_instability = (
            1.0 - max(winner_counts.values()) / len(best_by_repeat)
            if best_by_repeat
            else None
        )

        probe_result = dict(getattr(self.runtime, "_tail_probe_result", {}))
        tail_records = list(probe_result.get("gradient_tail_rows", []))
        pooled_tail = [
            row
            for row in tail_records
            if row.get("record_type") == "summary" and row.get("stratum") == "pooled"
        ]
        minimum_by_bin: list[float] = []
        for timestep_bin in range(int(self.args.dq_profile_timestep_bins)):
            values = [
                float(row["q95_d"])
                for row in pooled_tail
                if int(row["timestep_bin"]) == timestep_bin
            ]
            if values:
                minimum_by_bin.append(min(values))
        timestep_variation = (
            statistics.pstdev(minimum_by_bin) / max(abs(statistics.mean(minimum_by_bin)), 1e-30)
            if len(minimum_by_bin) > 1
            else None
        )
        high_values = [
            float(row["q95_d"])
            for row in pooled_tail
            if bool(row.get("is_max_timestep_bin", False))
        ]
        fragility_diag = {
            "winner_instability": winner_instability,
            "timestep_variation_cv_of_min_q95": timestep_variation,
            "minimum_max_timestep_q95": min(high_values) if high_values else None,
            "best_candidate_by_branch_repeat": best_by_repeat,
            "threshold_classification": None,
            "interpretation": "continuous development-dataset diagnostics only",
        }
        tail_bootstrap = dict(probe_result.get("tail_bootstrap", {}))
        tail_bootstrap["fragility_diag"] = fragility_diag
        tail_bootstrap["structural_repeat_noise_residual_fraction"] = probe_result.get(
            "structural_repeat_noise_residual_fraction"
        )
        decision = tail_bootstrap.get("decision", "abstain")
        calibration_gate = {
            "schema_version": "2.1.0",
            "metric_definition_version": "2.1.0",
            "protocol": self.runtime.profile_protocol,
            "completed": True,
            "decision": decision,
            "development_dataset_only": not safety_formal,
            "continue_to_d1_d2": (
                decision == "supported_on_development_dataset"
                if not safety_formal
                else False
            ),
            "safety_acceptance_not_selector_or_utility": safety_formal,
            "automatic_followup_started": False,
        }

        branch_summaries = self._summarize_candidates(executions, direction_rows)
        no_quant_summary = self._summarize_candidates(list(references.values()), [])
        if "no_quant" in no_quant_summary:
            branch_summaries["no_quant"] = no_quant_summary["no_quant"]
            branch_summaries["no_quant"]["guardian_mode"] = "common_skip_reference"
        all_executions = list(references.values()) + executions
        execution_manifest_rows = [
            {
                "execution_id": item.execution_id,
                "reference_execution_id": item.reference_execution_id,
                "cohort_id": item.cohort_id,
                "phase": item.phase,
                "candidate": item.candidate.name,
                "range_mul": item.candidate.initial_range_mul,
                "repeat": item.repeat,
                "guardian_mode": item.guardian_mode,
                "requested_max_steps": item.max_steps,
                "actual_steps": item.actual_steps,
                "actual_checkpoint": item.actual_checkpoint,
                "forced_safety_abort": item.forced_safety_abort,
                "invalid_reason": item.invalid_reason,
                "last_safe_step": item.last_safe_step,
            }
            for item in sorted(all_executions, key=lambda value: value.execution_id)
        ]
        intrinsic = summarize_stability(
            direction_rows,
            grid=grid,
            checkpoint=128,
            guardian_mode="common_skip",
        )
        intrinsic["role"] = "secondary_cumulative_update_diagnostic"
        intrinsic["primary_tail_decision"] = decision
        self._restore()
        return {
            "trajectory_rows": [row for item in all_executions for row in item.rows],
            "branch_summaries": branch_summaries,
            "update_direction_rows": direction_rows,
            "range_sweep_rows": self._range_sweep_rows(executions, direction_rows),
            "guardian_ablation_rows": [],
            "mechanism_ablation_rows": [],
            "execution_manifest_rows": execution_manifest_rows,
            "intrinsic_noise_rows": probe_result.get("intrinsic_noise_rows", []),
            "intrinsic_noise_summary": probe_result.get("intrinsic_noise_summary"),
            "local_natural_gradient_rows": probe_result.get(
                "local_natural_gradient_rows", []
            ),
            "gradient_tail_rows": tail_records,
            "tail_bootstrap": tail_bootstrap,
            "cumulative_null_rows": cumulative_null_rows,
            "calibration_gate": calibration_gate,
            "fragility_diag": fragility_diag,
            "intrinsic_stability_result": intrinsic,
            "guardian_result": {
                "guardian_adjusted_result": None,
                "guardian_dependent": None,
                "invalid_reason": (
                    "safety_formal_uses_common_skip_only"
                    if safety_formal
                    else "tail_calibration_uses_common_skip_only"
                ),
            },
            "mechanism_result": {"valid": False, "invalid_reason": "not_requested"},
            "range_grid": grid,
            "edge_extensions": [],
            "branch_repeats_executed": repeats,
            "third_repeat_performed": True,
            "extension_128_performed": False,
            "extension_128_result": None,
            "guardian_anchor": None,
            "repeat_replay_manifests": {
                str(repeat): self._repeat_sequence(repeat).manifest()
                for repeat in repeats
            },
        }


    def _run_prefix_smoke(self) -> dict[str, Any]:
        if len(self.sequence) < 128:
            raise ValueError("v2-prefix-smoke requires at least 128 materialized replay batches")
        candidate = _candidate_for_mul(3.15)
        cohorts: dict[str, tuple[BranchExecution, BranchExecution]] = {}

        def execute_cohort(label: str, max_steps: int) -> tuple[BranchExecution, BranchExecution]:
            cohort_id = f"v2_prefix_smoke.{label}.repeat_0.steps_{max_steps}"
            reference = self._reference(
                0,
                max_steps,
                quant_phase="v2_prefix_smoke",
                cohort_id=cohort_id,
                capture_prefix_state=True,
            )
            execution = self._execute(
                candidate=candidate,
                repeat=0,
                guardian_mode="common_skip",
                max_steps=max_steps,
                reference=reference,
                quant_phase="v2_prefix_smoke",
                cohort_id=cohort_id,
                capture_prefix_state=True,
            )
            return reference, execution

        # Keep the 64A baseline in memory, compare each challenger immediately,
        # then release its numeric state tensors. Fingerprint rows remain, but
        # the smoke test never retains all six optimizer snapshots at once.
        cohorts["64A"] = execute_cohort("64A", 64)
        parity_rows: list[dict[str, Any]] = []
        pair_summaries: list[dict[str, Any]] = []
        for label, max_steps, comparison in (
            ("64B", 64, "64A_vs_64B"),
            ("128", 128, "64A_vs_128_at64"),
        ):
            cohorts[label] = execute_cohort(label, max_steps)
            for candidate_index, candidate_name in ((0, "no_quant"), (1, candidate.name)):
                baseline = cohorts["64A"][candidate_index]
                challenger = cohorts[label][candidate_index]
                rows, summary = evaluate_prefix_pair(
                    reference_rows=baseline.rows[:64],
                    candidate_rows=challenger.rows[:64],
                    reference_states=baseline.state_values,
                    candidate_states=challenger.state_values,
                    comparison=comparison,
                    candidate_name=candidate_name,
                )
                parity_provenance = {
                    "reference_execution_id": baseline.execution_id,
                    "execution_id": challenger.execution_id,
                    "reference_cohort_id": baseline.cohort_id,
                    "cohort_id": challenger.cohort_id,
                    "requested_max_steps": challenger.max_steps,
                    "actual_steps": challenger.actual_steps,
                }
                for row in rows:
                    row.update(parity_provenance)
                summary.update(parity_provenance)
                parity_rows.extend(rows)
                pair_summaries.append(summary)
                challenger.state_values.clear()
        gate = aggregate_prefix_gate(pair_summaries)
        gate["kernel_policy"] = dict(
            getattr(
                self.args,
                "dq_profile_prefix_kernel_policy",
                {"mode": "native", "enabled": False},
            )
        )

        direction_rows: list[dict[str, Any]] = []
        for label, (reference, execution) in cohorts.items():
            rows = self._direction_rows(execution, reference, guardian_mode="common_skip")
            for row in rows:
                row["result_scope"] = f"prefix_{label}"
            direction_rows.extend(rows)
            for row in reference.rows:
                row["result_scope"] = f"prefix_{label}"
            for row in execution.rows:
                row["result_scope"] = f"prefix_{label}"

        baseline_reference, baseline_candidate = cohorts["64A"]
        branch_summaries = self._summarize_candidates(
            [baseline_candidate],
            self._direction_rows(baseline_candidate, baseline_reference, guardian_mode="common_skip"),
        )
        no_quant_summary = self._summarize_candidates([baseline_reference], [])
        if "no_quant" in no_quant_summary:
            branch_summaries["no_quant"] = no_quant_summary["no_quant"]
            branch_summaries["no_quant"]["guardian_mode"] = "common_skip_reference"

        executions = [item for pair in cohorts.values() for item in pair]
        state_rows = [row for execution in executions for row in execution.state_records]
        for execution in cohorts["64A"]:
            execution.state_values.clear()
        execution_manifest_rows = [
            {
                "execution_id": item.execution_id,
                "reference_execution_id": item.reference_execution_id,
                "cohort_id": item.cohort_id,
                "phase": item.phase,
                "candidate": item.candidate.name,
                "range_mul": item.candidate.initial_range_mul,
                "repeat": item.repeat,
                "guardian_mode": item.guardian_mode,
                "requested_max_steps": item.max_steps,
                "actual_steps": item.actual_steps,
                "actual_checkpoint": item.actual_checkpoint,
                "forced_safety_abort": item.forced_safety_abort,
                "invalid_reason": item.invalid_reason,
                "last_safe_step": item.last_safe_step,
            }
            for item in sorted(executions, key=lambda value: value.execution_id)
        ]
        self._restore()
        return {
            "trajectory_rows": [row for execution in executions for row in execution.rows],
            "branch_summaries": branch_summaries,
            "update_direction_rows": direction_rows,
            "range_sweep_rows": self._range_sweep_rows(
                [pair[1] for pair in cohorts.values()],
                direction_rows,
            ),
            "guardian_ablation_rows": [],
            "mechanism_ablation_rows": [],
            "execution_manifest_rows": execution_manifest_rows,
            "state_fingerprint_rows": state_rows,
            "prefix_parity_rows": parity_rows,
            "prefix_parity": gate,
            "calibration_gate": gate,
            "intrinsic_stability_result": {
                "diagnostic_target": "prefix_reproducibility",
                "diagnostic_optimum": None,
                "invalid_reason": "range_selection_not_part_of_prefix_smoke",
            },
            "guardian_result": {
                "guardian_adjusted_result": None,
                "guardian_dependent": None,
            },
            "mechanism_result": {"valid": False, "invalid_reason": "not_requested"},
            "range_grid": [3.15],
            "edge_extensions": [],
            "branch_repeats_executed": [0],
            "third_repeat_performed": False,
            "extension_128_performed": False,
            "extension_128_result": None,
            "guardian_anchor": None,
            "repeat_replay_manifests": {"0": self._repeat_sequence(0).manifest()},
        }


    def _run_safety_local(self) -> dict[str, Any]:
        probe_result = dict(getattr(self.runtime, "_tail_probe_result", {}))
        grid = sorted(
            float(value) for value in self.args.dq_profile_range_muls_resolved
        )
        branch_summaries = {
            candidate.name: {
                "candidate": candidate.name,
                "initial_range_mul": candidate.initial_range_mul,
                "mechanism": candidate.mechanism,
                "guardian_mode": "not_run_local_probe_only",
                "branch_steps": 0,
                "branch_repeats": 0,
                "forced_safety_abort": False,
                "invalid_reason": None,
            }
            for candidate in self.runtime.candidates
        }
        gate = {
            "schema_version": "2.1.0",
            "metric_definition_version": "2.1.0",
            "protocol": self.runtime.profile_protocol,
            "completed": True,
            "decision": "local_curve_measured_formal_score_unknown",
            "formal_branch_performed": False,
            "selector_or_utility_vote": False,
        }
        self._restore()
        return {
            "trajectory_rows": [],
            "branch_summaries": branch_summaries,
            "update_direction_rows": [],
            "range_sweep_rows": [],
            "guardian_ablation_rows": [],
            "mechanism_ablation_rows": [],
            "execution_manifest_rows": [],
            "intrinsic_noise_rows": probe_result.get("intrinsic_noise_rows", []),
            "intrinsic_noise_summary": probe_result.get(
                "intrinsic_noise_summary"
            ),
            "local_natural_gradient_rows": probe_result.get(
                "local_natural_gradient_rows", []
            ),
            "gradient_tail_rows": probe_result.get("gradient_tail_rows", []),
            "tail_bootstrap": probe_result.get("tail_bootstrap"),
            "cumulative_null_rows": [],
            "calibration_gate": gate,
            "fragility_diag": None,
            "intrinsic_stability_result": {
                "diagnostic_target": "local_gradient_acceptance",
                "diagnostic_optimum": None,
                "invalid_reason": "formal_128_step_branch_not_run",
            },
            "guardian_result": {
                "guardian_adjusted_result": None,
                "guardian_dependent": None,
                "invalid_reason": "local_probe_only",
            },
            "mechanism_result": {
                "valid": False,
                "invalid_reason": "not_requested",
            },
            "range_grid": grid,
            "edge_extensions": [],
            "branch_repeats_executed": [],
            "third_repeat_performed": False,
            "extension_128_performed": False,
            "extension_128_result": None,
            "guardian_anchor": None,
            "repeat_replay_manifests": {},
        }


    def run(self) -> dict[str, Any]:
        if self.runtime.profile_protocol in {
            "v2-tail-calibration",
            "v23-safety-formal",
            "v24-acceptance-formal",
            "v24-trajectory-descriptive",
        }:
            return self._run_tail_calibration()
        if self.runtime.profile_protocol in {
            "v23-safety-local",
            "v24-acceptance-local",
        }:
            return self._run_safety_local()
        if self.runtime.profile_protocol == "v2-prefix-smoke":
            return self._run_prefix_smoke()
        sweep_steps = int(self.args.dq_profile_sweep_steps)
        if sweep_steps < 64:
            raise ValueError("v2 Core requires at least 64 sweep steps")
        requested_repeats = int(self.args.dq_profile_branch_repeats)
        repeats = list(range(max(2, requested_repeats)))
        grid = sorted(float(value) for value in self.args.dq_profile_range_muls_resolved)
        candidates = list(fixed_range_candidates(grid)[1:])

        references, common_executions = self._run_common_candidates(candidates, repeats, sweep_steps)
        common_rows = self._collect_direction_rows(
            common_executions,
            references,
            guardian_mode="common_skip",
        )
        intrinsic = summarize_stability(common_rows, grid=grid, checkpoint=64, guardian_mode="common_skip")

        edge_added = self._edge_extensions(intrinsic, grid)
        if edge_added:
            grid = sorted(set(grid).union(edge_added))
            new_candidates = [_candidate_for_mul(value) for value in edge_added]
            candidates.extend(new_candidates)
            _, new_executions = self._run_common_candidates(new_candidates, repeats, sweep_steps)
            common_executions.extend(new_executions)
            common_rows = self._collect_direction_rows(
                common_executions,
                references,
                guardian_mode="common_skip",
            )
            intrinsic = summarize_stability(common_rows, grid=grid, checkpoint=64, guardian_mode="common_skip")

        third_repeat_performed = 2 in repeats
        if intrinsic.get("third_repeat_required") and 2 not in repeats:
            repeats.append(2)
            third_repeat_performed = True
            new_references, new_executions = self._run_common_candidates(candidates, [2], sweep_steps)
            references.update(new_references)
            common_executions.extend(new_executions)
            common_rows = self._collect_direction_rows(
                common_executions,
                references,
                guardian_mode="common_skip",
            )
            intrinsic = summarize_stability(common_rows, grid=grid, checkpoint=64, guardian_mode="common_skip")

        guardian_mode = str(self.args.dq_profile_guardian_ablation)
        native_executions: list[BranchExecution] = []
        native_rows: list[dict[str, Any]] = []
        native_result: dict[str, Any]
        guardian_anchor: Optional[float] = None
        if guardian_mode == "common_only":
            native_result = {
                "diagnostic_target": DIAGNOSTIC_TARGET,
                "diagnostic_optimum": None,
                "invalid_reason": "native_guardian_not_requested",
            }
            guardian_result = {
                "intrinsic_stability_result": intrinsic,
                "guardian_adjusted_result": native_result,
                "guardian_dependent": None,
            }
        else:
            if guardian_mode == "native_only":
                guardian_anchor = self._anchor(common_rows, grid, "common_skip")
                native_candidates = candidates
                for repeat in repeats:
                    for candidate in native_candidates:
                        native_executions.append(
                            self._execute(
                                candidate=candidate,
                                repeat=repeat,
                                guardian_mode="native_guardian",
                                max_steps=sweep_steps,
                                reference=references[repeat],
                                cohort_id=references[repeat].cohort_id,
                            )
                        )
                native_rows = self._collect_direction_rows(
                    native_executions,
                    references,
                    guardian_mode="native_guardian",
                )
                native_result = summarize_stability(
                    native_rows,
                    grid=grid,
                    checkpoint=64,
                    guardian_mode="native_guardian",
                )
            else:
                native_executions, native_rows, native_result, guardian_anchor = self._native_confirmation(
                    common_rows=common_rows,
                    intrinsic=intrinsic,
                    grid=grid,
                    repeats=repeats,
                    max_steps=sweep_steps,
                    references=references,
                )
            guardian_result = guardian_dependence(intrinsic, native_result)

            if bool(guardian_result["guardian_dependent"]) and 2 not in repeats:
                repeats.append(2)
                third_repeat_performed = True
                new_references, new_executions = self._run_common_candidates(candidates, [2], sweep_steps)
                references.update(new_references)
                common_executions.extend(new_executions)
                common_rows = self._collect_direction_rows(
                    common_executions,
                    references,
                    guardian_mode="common_skip",
                )
                intrinsic = summarize_stability(
                    common_rows,
                    grid=grid,
                    checkpoint=64,
                    guardian_mode="common_skip",
                )
                tested_native = sorted(
                    {float(item.candidate.initial_range_mul) for item in native_executions}
                )
                for candidate in [_candidate_for_mul(value) for value in tested_native]:
                    native_executions.append(
                        self._execute(
                            candidate=candidate,
                            repeat=2,
                            guardian_mode="native_guardian",
                            max_steps=sweep_steps,
                            reference=references[2],
                            cohort_id=references[2].cohort_id,
                        )
                    )
                native_rows = self._collect_direction_rows(
                    native_executions,
                    references,
                    guardian_mode="native_guardian",
                )
                native_result = summarize_stability(
                    native_rows,
                    grid=grid,
                    checkpoint=64,
                    guardian_mode="native_guardian",
                )
                guardian_result = guardian_dependence(intrinsic, native_result)

        extension_128 = any(
            reason in set(intrinsic.get("third_repeat_reasons", ()))
            for reason in ("m_dir_changed_32_to_64", "m_total_changed_32_to_64")
        )
        extension_references: dict[int, BranchExecution] = {}
        extension_executions: list[BranchExecution] = []
        extension_rows: list[dict[str, Any]] = []
        extension_summary: dict[str, Any] = {
            "performed": False,
            "ranking_scope": "separate_from_core_64",
        }
        if extension_128:
            anchor = self._anchor(common_rows, grid, "common_skip")
            extension_candidates = self._neighbor_candidates(anchor, grid)
            extension_references, extension_executions = self._run_common_candidates(
                extension_candidates,
                repeats,
                128,
                quant_phase="v2_core",
                cohort_label="v2_core_extension_128",
            )
            extension_rows = self._collect_direction_rows(
                extension_executions,
                extension_references,
                guardian_mode="common_skip",
            )
            for row in extension_rows:
                row["result_scope"] = "extension_128"
            extension_grid = [float(candidate.initial_range_mul) for candidate in extension_candidates]
            extension_summary = {
                "performed": True,
                "ranking_scope": "candidate_subset_within_extension_128_only",
                "candidate_grid": extension_grid,
                "stability_at_128": summarize_stability(
                    extension_rows,
                    grid=extension_grid,
                    checkpoint=128,
                    guardian_mode="common_skip",
                ),
            }

        selected_muls = tuple(
            float(value)
            for value in getattr(self.args, "dq_profile_mechanism_muls_resolved", ())
        )
        mechanism_rows: list[dict[str, Any]] = []
        mechanism_trajectory: list[dict[str, Any]] = []
        mechanism_summary: dict[str, Any] = {"valid": False, "invalid_reason": "not_requested"}
        if self.runtime.profile_protocol == "v2-mechanism":
            mechanism_rows, mechanism_trajectory, mechanism_summary = self._mechanism(
                selected_muls=selected_muls,
                repeats=repeats,
                max_steps=64,
            )

        for row in common_rows:
            row["result_scope"] = "core_64"
        for row in native_rows:
            row["result_scope"] = "guardian_core_64"
        for execution in common_executions:
            for row in execution.rows:
                row["result_scope"] = "core_64"
        for reference in references.values():
            for row in reference.rows:
                row["result_scope"] = "core_64"
        for execution in native_executions:
            for row in execution.rows:
                row["result_scope"] = "guardian_core_64"
        for execution in extension_executions:
            for row in execution.rows:
                row["result_scope"] = "extension_128"
        for reference in extension_references.values():
            for row in reference.rows:
                row["result_scope"] = "extension_128"

        common_summary = self._summarize_candidates(common_executions, common_rows)
        no_quant_summaries = self._summarize_candidates(list(references.values()), [])
        if "no_quant" in no_quant_summaries:
            common_summary["no_quant"] = no_quant_summaries["no_quant"]
            common_summary["no_quant"]["guardian_mode"] = "common_skip_reference"

        trajectory_rows = [row for execution in common_executions for row in execution.rows]
        trajectory_rows.extend(row for reference in references.values() for row in reference.rows)
        trajectory_rows.extend(row for execution in native_executions for row in execution.rows)
        trajectory_rows.extend(row for execution in extension_executions for row in execution.rows)
        trajectory_rows.extend(row for reference in extension_references.values() for row in reference.rows)
        trajectory_rows.extend(mechanism_trajectory)
        range_sweep_rows = self._range_sweep_rows(common_executions, common_rows)
        extension_range_rows = self._range_sweep_rows(extension_executions, extension_rows)
        guardian_rows = self._range_sweep_rows(native_executions, native_rows)
        execution_manifest_rows = [
            {
                "execution_id": item.execution_id,
                "reference_execution_id": item.reference_execution_id,
                "cohort_id": item.cohort_id,
                "phase": item.phase,
                "candidate": item.candidate.name,
                "range_mul": item.candidate.initial_range_mul,
                "repeat": item.repeat,
                "guardian_mode": item.guardian_mode,
                "requested_max_steps": item.max_steps,
                "actual_steps": item.actual_steps,
                "actual_checkpoint": item.actual_checkpoint,
                "forced_safety_abort": item.forced_safety_abort,
                "invalid_reason": item.invalid_reason,
                "last_safe_step": item.last_safe_step,
            }
            for item in sorted(self._executions.values(), key=lambda value: value.execution_id)
        ]
        self._restore()
        return {
            "trajectory_rows": trajectory_rows,
            "branch_summaries": common_summary,
            "update_direction_rows": common_rows + native_rows + extension_rows,
            "range_sweep_rows": range_sweep_rows + extension_range_rows,
            "execution_manifest_rows": execution_manifest_rows,
            "guardian_ablation_rows": guardian_rows,
            "mechanism_ablation_rows": mechanism_rows,
            "intrinsic_stability_result": intrinsic,
            "guardian_result": guardian_result,
            "mechanism_result": mechanism_summary,
            "range_grid": grid,
            "edge_extensions": edge_added,
            "branch_repeats_executed": repeats,
            "third_repeat_performed": third_repeat_performed,
            "extension_128_performed": extension_128,
            "extension_128_result": extension_summary,
            "guardian_anchor": guardian_anchor,
            "selected_stability_mul_for_mechanism": selected_muls[0] if len(selected_muls) == 1 else None,
            "selected_stability_muls_for_mechanism": list(selected_muls),
            "repeat_replay_manifests": {
                str(repeat): self._repeat_sequence(repeat).manifest() for repeat in repeats
            },
        }


def run_v2_experiments(
    *,
    runtime: Any,
    sequence: ReplaySequence,
    snapshot: TrainingSnapshot,
    pass_context: Mapping[str, Any],
) -> dict[str, Any]:
    return V2ExperimentRunner(
        runtime=runtime,
        sequence=sequence,
        snapshot=snapshot,
        pass_context=pass_context,
    ).run()
