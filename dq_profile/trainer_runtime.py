from __future__ import annotations

import inspect
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import torch

from dq_profile.geometry import SourceGroupMap, captions_from_batches
from dq_profile import METRIC_DEFINITION_VERSION, PROTOCOL_VERSION, SCHEMA_VERSION
from dq_profile.manifest import build_source_manifest, sha256_file
from dq_profile.metrics import CountSketch, ExactGradient, aggregate_numeric, gradient_noise_scale, gram_and_rank
from dq_profile.protocol import (
    AutoRangeController,
    CandidateDefinition,
    canonical_sha256,
    default_candidates,
    fixed_range_candidates,
)
from dq_profile.quant_context import ProfileQuantContext, aggregate_shadow_rows
from dq_profile.replay import ReplayBatch, ReplaySequence, replay_digest, seed_step_rng
from dq_profile.v2_calibration import (
    angular_gradient_distance,
    bootstrap_intrinsic_noise,
    bootstrap_tail_winner,
    exact_gradient_fingerprint,
    fingerprint_tree,
    gradient_gain_distance,
    gradient_tail_rows,
    intrinsic_noise_rows,
    rng_fingerprint,
    source_contract_from_manifest,
    summarize_gradient_tail,
    symmetric_gradient_distance,
)
from dq_profile.v2_metrics import (
    caption_tag_metrics,
    hard_safety_reason,
    hierarchical_geometry_variance,
    sketch_agreement,
)
from dq_profile.v2_runtime import run_v2_experiments
from dq_profile.report import ProfileArtifacts, trajectory_svg, write_report
from dq_profile.snapshot import TrainingSnapshot
from library import train_util
from library.rounding_util import round_parameters


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def _finite_json(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    return _json_safe(value)


def _current_control_contract(args: Any, preflight: Mapping[str, Any], dataset_sha256: str) -> dict[str, Any]:
    return {
        "network_dim": int(getattr(args, "network_dim", 0) or 0),
        "optimizer": str(getattr(args, "optimizer_type", "") or ""),
        "mixed_precision": str(getattr(args, "mixed_precision", "") or ""),
        "save_precision": str(getattr(args, "save_precision", "") or ""),
        "fp16_safe_norms_mode": str(getattr(args, "fp16_safe_norms_mode_resolved", "") or ""),
        "training_steps": int(preflight.get("normal_training_steps", 0) or 0),
        "dataset_sha256": str(dataset_sha256),
        "dq_bits": int(getattr(args, "dq_delta_bits", 0) or 0),
        "dq_granularity": str(getattr(args, "dq_delta_granularity", "") or ""),
        "dq_stat": str(getattr(args, "dq_delta_stat", "") or ""),
        "dq_mode": str(getattr(args, "dq_delta_mode", "") or ""),
        "dq_scope": str(getattr(args, "dq_delta_scope", "") or ""),
    }


def _snapshot_warmup_contract(
    args: Any,
    preflight: Mapping[str, Any],
    *,
    dataset_sha256: str,
    source_contract_sha256: str,
    dq_delta_begin_step: int,
) -> dict[str, Any]:
    """Controls that are allowed to influence the common no-quant warmup."""

    names = (
        "pretrained_model_name_or_path",
        "seed",
        "network_dim",
        "network_alpha",
        "network_args",
        "network_dropout",
        "learning_rate",
        "text_encoder_lr",
        "text_encoder_lr1",
        "text_encoder_lr2",
        "unet_lr",
        "optimizer_type",
        "optimizer_args",
        "mixed_precision",
        "fp16_safe_norms_mode_resolved",
        "gradient_accumulation_steps",
        "max_train_epochs",
        "max_train_steps",
        "lr_scheduler",
        "lr_warmup_steps",
        "lr_scheduler_num_cycles",
        "lr_scheduler_power",
        "noise_offset",
        "adaptive_noise_scale",
        "min_snr_gamma",
        "ip_noise_gamma",
        "ip_noise_gamma_random_strength",
        "cache_latents",
        "enable_bucket",
        "min_bucket_reso",
        "max_bucket_reso",
        "bucket_reso_steps",
        "bucket_no_upscale",
        "max_data_loader_n_workers",
        "persistent_data_loader_workers",
        "downscale_freq_shift",
        "te_mlp_fc_only",
        "grad_norm_mode",
        "dq_delta_bits",
        "dq_delta_granularity",
        "dq_delta_stat",
        "dq_delta_mode",
        "dq_delta_scope",
        "dq_delta_begin_after_lr_warmup",
        "dq_delta_use_triton",
        "dq_delta_triton_stats",
    )
    return _finite_json(
        {
            "contract_version": "snapshot-warmup-v1",
            "dataset_config_sha256": str(dataset_sha256),
            "source_contract_sha256": str(source_contract_sha256),
            "dq_delta_begin_step": int(dq_delta_begin_step),
            "steps_per_epoch": int(preflight.get("steps_per_epoch", 0) or 0),
            "kernel_policy": getattr(args, "dq_profile_prefix_kernel_policy", None),
            "controls": {name: getattr(args, name, None) for name in names},
        }
    )


def _evaluate_known_result_controls(known_result: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(known_result)
    requested = bool(result.get("comparison_controlled", False))
    detected: list[str] = []
    if requested:
        supplied = result.get("control_differences", [])
        if isinstance(supplied, (list, tuple)):
            detected.extend(str(item) for item in supplied if str(item).strip())
        for key, current_value in current.items():
            past_key = f"past_{key}"
            past_value = result.get(past_key)
            missing = past_value is None or past_value == "" or (isinstance(past_value, (int, float)) and past_value == 0)
            if missing:
                detected.append(f"{past_key} is not recorded")
                continue
            if isinstance(current_value, str):
                matches = str(past_value).casefold() == current_value.casefold()
            else:
                matches = past_value == current_value
            if not matches:
                detected.append(f"{key}: past={past_value!r}, current={current_value!r}")
    result["comparison_controlled_requested"] = requested
    result["comparison_controlled_effective"] = bool(requested and not detected)
    result["detected_control_differences"] = detected
    return result


def _scalar(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(torch.float32).sum().item())
    return float(value)


def _combined_dq_metrics(exported: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    if not exported:
        return {
            "clip_rate": None,
            "quant_error_rms": None,
            "quant_error_ratio": None,
            "clip_error_rms": None,
            "round_error_rms": None,
        }
    accumulators = exported.get("accum", {})
    totals = defaultdict(float)
    for scope in ("unet", "te"):
        accumulator = accumulators.get(scope)
        if accumulator is None:
            continue
        for name in (
            "numel",
            "clip_count",
            "sumsq",
            "xq_sumsq",
            "xxq_sum",
            "clip_err_sumsq",
            "round_err_sumsq",
        ):
            value = getattr(accumulator, name, None)
            if value is not None:
                totals[name] += _scalar(value)
    count = totals["numel"]
    if count <= 0.0:
        return {
            "clip_rate": None,
            "quant_error_rms": None,
            "quant_error_ratio": None,
            "clip_error_rms": None,
            "round_error_rms": None,
        }
    error_sq = max(0.0, totals["sumsq"] + totals["xq_sumsq"] - 2.0 * totals["xxq_sum"])
    error_rms = math.sqrt(error_sq / count)
    x_rms = math.sqrt(max(totals["sumsq"], 0.0) / count)
    return {
        "clip_rate": totals["clip_count"] / count,
        "quant_error_rms": error_rms,
        "quant_error_ratio": error_rms / max(x_rms, 1e-30),
        "clip_error_rms": math.sqrt(max(totals["clip_err_sumsq"], 0.0) / count),
        "round_error_rms": math.sqrt(max(totals["round_err_sumsq"], 0.0) / count),
    }


def _set_fake_quant(network: Any, args: Any, *, enabled: bool, range_mul: Optional[float]) -> None:
    if hasattr(network, "set_delta_fake_quant"):
        kwargs = {
            "granularity": args.dq_delta_granularity,
            "stat": args.dq_delta_stat,
            "bits": args.dq_delta_bits,
            "range_mul": range_mul if range_mul is not None else args.dq_delta_range_mul,
            "on_z": bool(args.dq_quantize_z),
            "use_triton": bool(args.dq_delta_use_triton),
            "triton_stats": False,
        }
        setter = network.set_delta_fake_quant
        parameters = inspect.signature(setter).parameters
        for key in tuple(kwargs):
            if key not in parameters and not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                kwargs.pop(key)
        setter(args.dq_delta_step, args.dq_delta_mode, **kwargs)
    if hasattr(network, "set_delta_quant_enabled"):
        network.set_delta_quant_enabled(bool(enabled))
    if enabled:
        scope = getattr(args, "dq_delta_scope", "both")
        if scope == "unet":
            for lora in getattr(network, "text_encoder_loras", []):
                lora.delta_q_enabled = False
        elif scope == "te":
            for lora in getattr(network, "unet_loras", []):
                lora.delta_q_enabled = False


def _gradient_update_norm(network: torch.nn.Module, reference: Mapping[str, torch.Tensor]) -> float:
    total = 0.0
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            if name not in reference:
                continue
            delta = parameter.detach().to(device="cpu", dtype=torch.float32) - reference[name].to(torch.float32)
            total += float(torch.sum(delta * delta).item())
    return math.sqrt(max(total, 0.0))


def _tree_is_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.detach()).all().item())
    if isinstance(value, Mapping):
        return all(_tree_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_tree_is_finite(item) for item in value)
    return True
def _source_stratified_replay_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("candidate", "")),
            str(row.get("source_group", row.get("image_key", ""))),
            int(row.get("timestep_bin", -1)),
            str(row.get("probe_regime", "structural_dropout_off")),
        )
        grouped[key].append(row)
    output: list[dict[str, Any]] = []
    for (candidate, source_group, timestep_bin, regime), members in sorted(grouped.items()):
        metrics = aggregate_numeric(
            members,
            (
                "loss",
                "gradient_norm",
                "parameter_gradient_cosine",
                "clip_rate",
                "quant_error_rms",
                "quant_error_ratio",
            ),
        )
        output.append(
            {
                "evaluation_type": "source-stratified replay evaluation",
                "generalization_claim": False,
                "candidate": candidate,
                "source_group": source_group,
                "timestep_bin": timestep_bin,
                "probe_regime": regime,
                "row_count": len(members),
                "image_count": len({str(row.get("image_key", "")) for row in members}),
                **metrics,
            }
        )
    return output




def _post_step_nonfinite_reason(network: torch.nn.Module, optimizer: Any) -> Optional[str]:
    with torch.no_grad():
        for parameter in network.parameters():
            if not bool(torch.isfinite(parameter.detach()).all().item()):
                return "candidate_nonfinite_parameter_after_step"
    state = getattr(optimizer, "state", None)
    if state is None and hasattr(optimizer, "optimizer"):
        state = getattr(optimizer.optimizer, "state", None)
    if state is not None and not _tree_is_finite(state):
        return "candidate_nonfinite_optimizer_state_after_step"
    return None




class DiagnosticProfileRuntime:
    def __init__(self, *, args: Any, trainer: Any) -> None:
        self.args = args
        self.trainer = trainer
        self.protocol_seed = int(getattr(args, "dq_profile_seed", getattr(args, "seed", 0) or 0))
        self.profile_protocol = str(getattr(args, "dq_profile_protocol", "v1"))
        if self.profile_protocol == "v1":
            self.candidates = default_candidates()
        elif self.profile_protocol == "v2-prefix-smoke":
            self.candidates = (
                CandidateDefinition("no_quant", False, None, None, None, False),
                CandidateDefinition("mul_3.150", True, None, None, 3.15, False),
            )
        else:
            minimum_count = (
                1
                if self.profile_protocol in {
                    "v24-acceptance-formal",
                    "v24-trajectory-descriptive",
                }
                else 2
                if self.profile_protocol == "v23-safety-formal"
                else 3
            )
            self.candidates = fixed_range_candidates(
                getattr(
                    args,
                    "dq_profile_range_muls_resolved",
                    getattr(args, "dq_profile_range_muls", ()),
                ),
                minimum_count=minimum_count,
                maximum_count=3 if self.profile_protocol.endswith("-formal") else None,
            )
        self.quant_context = ProfileQuantContext(
            self.protocol_seed,
            rng_mode=str(getattr(args, "dq_profile_quant_rng_mode", "stateless")),
        )
        self.artifacts = ProfileArtifacts(args.dq_profile_run_dir)
        self._stats_sequence = 1_000_000
        self._warnings: list[str] = []

    def _next_stats_step(self) -> int:
        self._stats_sequence += 1
        return self._stats_sequence

    def _capture_batches(
        self,
        *,
        first_batch: Mapping[str, Any],
        epoch_iterator: Iterable[Any],
        train_dataloader: Iterable[Any],
        current_epoch: Any,
        current_step: Any,
        global_step: int,
        epoch: int,
        data_step: int,
        count: int,
    ) -> ReplaySequence:
        sequence = ReplaySequence()
        iterator = iter(epoch_iterator)
        current_batch = first_batch
        source_epoch = int(epoch)
        source_step = int(data_step)
        for index in range(count):
            if index > 0:
                # Match the production for-loop/collator ordering: the next
                # batch is collated while current_step still identifies the
                # previously completed optimizer step.
                current_step.value = global_step + index - 1
                try:
                    current_batch = next(iterator)
                    source_step += 1
                except StopIteration:
                    source_epoch += 1
                    current_epoch.value = source_epoch + 1
                    source_step = 0
                    iterator = iter(train_dataloader)
                    current_batch = next(iterator)
            sequence.append(
                ReplayBatch(
                    index=index,
                    source_epoch=source_epoch,
                    source_step=source_step,
                    global_step=global_step + index,
                    batch=dict(current_batch),
                )
            )
        sequence.seal()
        return sequence

    def _materialize_replay(
        self,
        sequence: ReplaySequence,
        *,
        accelerator: Any,
        vae: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        noise_scheduler: Any,
    ) -> None:
        for item in sequence:
            seed = seed_step_rng(self.protocol_seed, item.index, phase="materialize")
            batch = item.runtime_batch(accelerator.device)
            if "latents" in batch and batch["latents"] is not None:
                latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
            else:
                images = batch["images"].to(accelerator.device, dtype=vae_dtype)
                with torch.no_grad():
                    latents = vae.encode(images).latent_dist.sample().to(dtype=weight_dtype)
            if torch.any(torch.isnan(latents)):
                latents = torch.nan_to_num(latents, 0)
            latents = latents * self.trainer.vae_scale_factor
            progress = item.global_step / float(max(1, self.args.max_train_steps))
            noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                self.args,
                noise_scheduler,
                latents,
                progress_frac=max(0.0, min(1.0, progress)),
            )
            target = noise_scheduler.get_velocity(latents, noise, timesteps) if self.args.v_parameterization else noise
            item.latents = latents.detach().to("cpu").clone()
            item.noise = noise.detach().to("cpu").clone()
            item.noisy_latents = noisy_latents.detach().to("cpu").clone()
            item.timesteps = timesteps.detach().to("cpu").clone()
            item.target = target.detach().to("cpu").clone()
            item.huber_c = None if huber_c is None else huber_c.detach().to("cpu").clone()
            item.model_seed = seed
            item.refresh_digest()

    def _fixed_timestep_item(
        self,
        source: ReplayBatch,
        *,
        bin_index: int,
        bin_count: int,
        probe_replica: int,
        noise_scheduler: Any,
    ) -> ReplayBatch:
        if not source.materialized():
            raise RuntimeError("replay batch was not materialized")
        total = int(noise_scheduler.config.num_train_timesteps)
        center = min(total - 1, max(0, int((bin_index + 0.5) * total / bin_count)))
        latents = source.latents.to(torch.float32)
        model_seed = source.model_seed
        if int(probe_replica) == 0:
            noise = source.noise.to(torch.float32)
        else:
            model_seed = seed_step_rng(
                self.protocol_seed,
                source.index * max(1, int(bin_count)) + int(bin_index),
                phase="structural_probe_noise",
                repeat=int(probe_replica),
            )
            device = noise_scheduler.alphas_cumprod.device
            progress = source.global_step / float(max(1, self.args.max_train_steps))
            noise, _, _, _ = train_util.get_noise_noisy_latents_and_timesteps(
                self.args,
                noise_scheduler,
                latents.to(device),
                progress_frac=max(0.0, min(1.0, progress)),
            )
            noise = noise.to("cpu", dtype=torch.float32)
        timesteps = torch.full((latents.shape[0],), center, dtype=torch.long)
        device = noise_scheduler.alphas_cumprod.device
        noisy = noise_scheduler.add_noise(latents.to(device), noise.to(device), timesteps.to(device)).to("cpu")
        if self.args.v_parameterization:
            target = noise_scheduler.get_velocity(latents.to(device), noise.to(device), timesteps.to(device)).to("cpu")
        else:
            target = noise
        huber_c = self._fixed_huber_c(timesteps, noise_scheduler)
        return ReplayBatch(
            index=source.index,
            source_epoch=source.source_epoch,
            source_step=source.source_step,
            global_step=source.global_step,
            batch=source.batch,
            latents=latents,
            noise=noise,
            noisy_latents=noisy,
            timesteps=timesteps,
            target=target,
            huber_c=huber_c,
            model_seed=model_seed,
        )

    def _fixed_huber_c(self, timesteps: torch.Tensor, noise_scheduler: Any) -> Optional[torch.Tensor]:
        if self.args.loss_type == "l2":
            return None
        if self.args.huber_schedule == "exponential":
            alpha = -math.log(self.args.huber_c) / noise_scheduler.config.num_train_timesteps
            return torch.exp(-alpha * timesteps.to(torch.float32))
        if self.args.huber_schedule == "snr":
            alphas = torch.index_select(noise_scheduler.alphas_cumprod.to("cpu"), 0, timesteps.to("cpu"))
            sigmas = ((1.0 - alphas) / alphas) ** 0.5
            return (1 - self.args.huber_c) / (1 + sigmas) ** 2 + self.args.huber_c
        return torch.full((timesteps.shape[0],), self.args.huber_c)

    def _run_pass(
        self,
        *,
        replay: ReplayBatch,
        candidate: CandidateDefinition,
        range_mul: Optional[float],
        phase: str,
        probe_or_step: str | int,
        repeat: int,
        dropout_enabled: bool,
        shadow: bool,
        update: bool,
        do_auto_observation: bool,
        absolute_step: int,
        epoch: int,
        accelerator: Any,
        network: Any,
        optimizer: Any,
        lr_scheduler: Any,
        grad_norm_guardian: Any,
        unet: Any,
        text_encoders: list[Any],
        tokenizers: list[Any],
        train_unet: bool,
        train_text_encoder: bool,
        training_model: Any,
        on_step_start: Any,
        weight_dtype: torch.dtype,
        noise_scheduler: Any,
        forced_skip: Optional[bool] = None,
        matched_no_quant_gradient_norm: Optional[float] = None,
        hard_safety: bool = False,
    ) -> tuple[dict[str, Any], ExactGradient, list[dict[str, Any]]]:
        if not replay.materialized():
            raise RuntimeError("profile pass requires a fully materialized replay batch")
        rng_digest_before = rng_fingerprint()
        unwrapped = accelerator.unwrap_model(network)
        scaler = getattr(accelerator, "scaler", None)
        grad_scale = float(scaler.get_scale()) if scaler is not None and hasattr(scaler, "get_scale") else 1.0
        mode = "shadow" if shadow else ("candidate" if candidate.quantized else "inactive")
        self.quant_context.begin_pass(
            mode=mode,
            phase=phase,
            probe_or_step=probe_or_step,
            repeat=repeat,
            dropout_enabled=dropout_enabled,
            grad_scale=grad_scale,
            shadow_candidates=tuple(item for item in self.candidates if item.quantized) if shadow else (),
            mechanism=candidate.mechanism,
            shadow_repeats=int(self.args.dq_profile_stochastic_repeats) if shadow else 0,
            control_rng_digest=rng_digest_before,
        )
        if hasattr(unwrapped, "set_dq_profile_context"):
            unwrapped.set_dq_profile_context(self.quant_context)
        _set_fake_quant(
            unwrapped,
            self.args,
            enabled=bool((candidate.quantized and not shadow) or shadow),
            range_mul=range_mul,
        )

        stats_step = self._next_stats_step()
        if hasattr(unwrapped, "set_dq_stats_state"):
            unwrapped.set_dq_stats_state(
                step_idx=stats_step,
                device=accelerator.device,
                do_log=bool(candidate.quantized and not shadow),
                do_auto=bool(candidate.quantized and not shadow and do_auto_observation),
                collect_full=True,
                collect_zero=False,
                collect_near_zero=False,
                collect_detail=False,
                collect_error_parts=True,
                log_mode="summary",
                log_scope="both",
                auto_scope=getattr(self.args, "dq_delta_scope", "both"),
                target="z" if self.args.dq_quantize_z else "delta",
            )

        old_unet_mode = bool(unet.training)
        old_te_modes = [bool(encoder.training) for encoder in text_encoders]
        if not dropout_enabled:
            unet.eval()
            for encoder in text_encoders:
                encoder.eval()
        network.train()
        optimizer.zero_grad(set_to_none=True)
        batch = replay.runtime_batch(accelerator.device)
        self.trainer._set_network_multiplier_from_batch(unwrapped, batch)
        if update:
            self.trainer._apply_te_freeze_if_ready(optimizer, unwrapped, absolute_step)
        text_encoder_arg = text_encoders if len(text_encoders) > 1 else text_encoders[0]
        on_step_start(text_encoder_arg, unet)
        noisy_latents = replay.noisy_latents.to(accelerator.device, dtype=weight_dtype)
        timesteps = replay.timesteps.to(accelerator.device)
        target = replay.target.to(accelerator.device, dtype=weight_dtype)
        huber_c = replay.huber_c
        if isinstance(huber_c, torch.Tensor):
            huber_c = huber_c.to(accelerator.device)

        exported = None
        shadow_rows: list[dict[str, Any]] = []
        try:
            with torch.set_grad_enabled(train_text_encoder):
                text_conditions = self.trainer._get_text_conds_for_batch(
                    self.args,
                    accelerator,
                    batch,
                    tokenizers,
                    text_encoders,
                    weight_dtype,
                    grad_enabled=train_text_encoder,
                )
            if self.args.gradient_checkpointing:
                for value in noisy_latents:
                    value.requires_grad_(True)
                for value in text_conditions:
                    value.requires_grad_(True)
            loss = self.trainer._compute_batch_loss(
                self.args,
                accelerator,
                batch,
                noise_scheduler,
                unet,
                text_conditions,
                noisy_latents,
                timesteps,
                target,
                huber_c,
                weight_dtype,
                train_unet=train_unet,
            )
            accelerator.backward(loss)
            exact_gradient = ExactGradient.capture(unwrapped.named_parameters(), scale=grad_scale)
            shadow_rows = self.quant_context.finish_pass()
            profile_trace = self.quant_context.last_trace
            if hasattr(unwrapped, "export_dq_stats"):
                exported = unwrapped.export_dq_stats()
            dq_metrics = _combined_dq_metrics(exported)
            native_would_skip = False
            if update and grad_norm_guardian is not None:
                native_would_skip = bool(
                    grad_norm_guardian.observe(network, epoch, absolute_step, float(loss.detach().item()))
                )
            safety_reason = None
            if update and hard_safety:
                safety_reason = hard_safety_reason(
                    loss=float(loss.detach().item()),
                    gradient_norm=exact_gradient.norm,
                    matched_no_quant_gradient_norm=matched_no_quant_gradient_norm,
                )
            skip = native_would_skip if forced_skip is None else bool(forced_skip)
            if safety_reason is not None:
                skip = True
            lr_before = [float(group.get("lr", 0.0)) for group in optimizer.param_groups]
            if update and not skip:
                if accelerator.sync_gradients:
                    self.trainer.all_reduce_network(accelerator, network)
                    if self.args.max_grad_norm != 0.0:
                        accelerator.clip_grad_norm_(unwrapped.get_trainable_params(), self.args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                self.trainer._apply_te_lr_after_if_ready(optimizer, lr_scheduler, absolute_step + 1)
                if self.args.round_lora_step is not None and self.args.round_lora_step > 0:
                    next_step = absolute_step + 1
                    progress = next_step / float(max(1, self.args.max_train_steps))
                    if progress >= self.args.round_lora_begin and next_step % max(1, self.args.round_lora_every) == 0:
                        round_parameters(
                            unwrapped.get_trainable_params(),
                            step=self.args.round_lora_step,
                            mode=self.args.round_lora_mode,
                            exclude_param_ids=self.trainer._te_frozen_param_ids,
                        )
                if self.args.scale_weight_norms:
                    self.trainer._apply_max_norm_regularization(unwrapped, self.args.scale_weight_norms, accelerator.device)
            if update and not skip and hard_safety:
                post_step_reason = _post_step_nonfinite_reason(unwrapped, optimizer)
                if post_step_reason is not None:
                    safety_reason = post_step_reason
            common_skip_matched = None if forced_skip is None else safety_reason is None
            optimizer_step_performed = bool(update and not skip)
            optimizer.zero_grad(set_to_none=True)
            rng_digest_after = rng_fingerprint()
            row = {
                "candidate": candidate.name,
                "phase": phase,
                "probe_or_step": str(probe_or_step),
                "repeat": int(repeat),
                "loss": float(loss.detach().item()),
                "gradient_norm": exact_gradient.norm,
                "range_mul": range_mul,
                "update_skipped": skip,
                "lr_before": lr_before,
                "lr_after": [float(group.get("lr", 0.0)) for group in optimizer.param_groups],
                "native_would_skip": native_would_skip,
                "common_skip_matched": common_skip_matched,
                "forced_safety_abort": safety_reason is not None,
                "invalid_reason": safety_reason,
                "optimizer_step_performed": optimizer_step_performed,
                "mechanism": candidate.mechanism,
                "gradient_hash": exact_gradient_fingerprint(exact_gradient),
                "replay_digest": replay.digest,
                "noise_digest": replay_digest(replay.noise),
                "timestep_digest": replay_digest(replay.timesteps),
                "rng_digest_before": rng_digest_before,
                "rng_digest_after": rng_digest_after,
                **profile_trace,
                **dq_metrics,
            }
            return row, exact_gradient, shadow_rows
        finally:
            if hasattr(unwrapped, "discard_dq_stats_step"):
                unwrapped.discard_dq_stats_step(stats_step)
            if hasattr(unwrapped, "set_dq_profile_context"):
                unwrapped.set_dq_profile_context(None)
            unet.train(old_unet_mode)
            for encoder, old_mode in zip(text_encoders, old_te_modes):
                encoder.train(old_mode)

    def _run_tail_probes(
        self,
        *,
        sequence: ReplaySequence,
        snapshot: TrainingSnapshot,
        accelerator: Any,
        network: Any,
        optimizer: Any,
        lr_scheduler: Any,
        grad_norm_guardian: Any,
        unet: Any,
        text_encoders: list[Any],
        tokenizers: list[Any],
        train_unet: bool,
        train_text_encoder: bool,
        training_model: Any,
        on_step_start: Any,
        weight_dtype: torch.dtype,
        noise_scheduler: Any,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, np.ndarray],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        unwrapped = accelerator.unwrap_model(network)
        snapshot.restore(
            network=unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=getattr(accelerator, "scaler", None),
            trainer=self.trainer,
            guardian=grad_norm_guardian,
        )
        selected = sequence.unique_image_items(int(self.args.dq_profile_max_images))
        protocol_minimum = (
            8 if self.profile_protocol.startswith("v24-") else 16
        )
        minimum_images = min(int(self.args.dq_profile_max_images), protocol_minimum)
        if len(selected) < minimum_images:
            raise ValueError(
                "safety probe replay does not contain enough unique images: "
                f"requested={int(self.args.dq_profile_max_images)}, "
                f"minimum={minimum_images}, available={len(selected)}"
            )
        bins = int(self.args.dq_profile_timestep_bins)
        no_quant_noise_replicas = int(
            getattr(
                self.args,
                "dq_profile_safety_no_quant_noise_replicas_resolved",
                5,
            )
        )
        candidate_noise_replicas = int(
            getattr(
                self.args,
                "dq_profile_safety_candidate_noise_replicas_resolved",
                2,
            )
        )
        quant_repeats = int(
            getattr(
                self.args,
                "dq_profile_safety_quant_repeats_resolved",
                2,
            )
        )
        geometry_enabled = self.profile_protocol == "v2-tail-calibration"
        no_quant = self.candidates[0]
        candidates = list(self.candidates[1:])
        expected = tuple(
            float(value) for value in self.args.dq_profile_range_muls_resolved
        )
        actual = tuple(float(item.initial_range_mul) for item in candidates)
        if actual != expected:
            raise ValueError(
                "safety probe candidate contract mismatch: "
                f"expected={expected!r}, actual={actual!r}"
            )

        sketchers = (
            {
                f"tail_seed_{index}": CountSketch(
                    width=512,
                    seed=self.protocol_seed + 1_000_003 * index,
                )
                for index in range(2)
            }
            if geometry_enabled
            else {}
        )
        sketches: dict[str, list[np.ndarray]] = {name: [] for name in sketchers}
        sketches_by_bin: dict[str, dict[int, list[np.ndarray]]] = {
            name: defaultdict(list) for name in sketchers
        }
        sketch_metadata: list[dict[str, Any]] = []
        source_group_map = SourceGroupMap.load(getattr(self.args, "dq_profile_source_group_map", None))
        per_image_rows: list[dict[str, Any]] = []
        natural_gradient_enabled = self.profile_protocol.startswith("v24-")
        local_natural_gradient_rows: list[dict[str, Any]] = []

        for image_index, source in enumerate(selected):
            image_key = source.image_keys[0] if source.image_keys else source.digest
            source_group = source_group_map.resolve(image_key)
            for bin_index in range(bins):
                bin_reference_gradients: list[tuple[int, ExactGradient]] = []
                for noise_replica in range(no_quant_noise_replicas):
                    probe = self._fixed_timestep_item(
                        source,
                        probe_replica=noise_replica,
                        bin_index=bin_index,
                        bin_count=bins,
                        noise_scheduler=noise_scheduler,
                    )
                    probe_id = f"tail:{image_index}:{bin_index}:{noise_replica}:{image_key}"
                    # Stable across local/formal protocols, candidate grids,
                    # replica-count changes, and branch horizons.
                    model_seed_id = (
                        f"image:{image_index}|bin:{bin_index}|"
                        f"noise:{noise_replica}|key:{image_key}"
                    )
                    seed_step_rng(
                        self.protocol_seed,
                        model_seed_id,
                        phase="v2_tail_structural_model",
                        repeat=0,
                    )
                    reference_row, reference_gradient, _ = self._run_pass(
                        replay=probe,
                        candidate=no_quant,
                        range_mul=None,
                        phase="v2_tail_probe",
                        probe_or_step=probe_id,
                        repeat=0,
                        dropout_enabled=False,
                        shadow=False,
                        update=False,
                        do_auto_observation=False,
                        absolute_step=int(snapshot.metadata["global_step"]),
                        epoch=int(snapshot.metadata["epoch"]),
                        accelerator=accelerator,
                        network=network,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        grad_norm_guardian=grad_norm_guardian,
                        unet=unet,
                        text_encoders=text_encoders,
                        tokenizers=tokenizers,
                        train_unet=train_unet,
                        train_text_encoder=train_text_encoder,
                        training_model=training_model,
                        on_step_start=on_step_start,
                        weight_dtype=weight_dtype,
                        noise_scheduler=noise_scheduler,
                    )
                    base_metadata = {
                        "image_key": image_key,
                        "source_group": source_group,
                        "timestep_bin": bin_index,
                        "timestep": int(probe.timesteps.reshape(-1)[0].item()),
                        "noise_replica": noise_replica,
                        "probe_replica": noise_replica,
                        "probe_regime": "structural_dropout_off",
                    }
                    per_image_rows.append(
                        {
                            **reference_row,
                            **base_metadata,
                            "quant_repeat": None,
                            "parameter_gradient_cosine": 1.0,
                            "gradient_topology_matches": True,
                        }
                    )
                    sketch_metadata.append(dict(base_metadata))
                    for sketch_name, sketcher in sketchers.items():
                        value = sketcher.sketch(reference_gradient)
                        sketches[sketch_name].append(value)
                        sketches_by_bin[sketch_name][bin_index].append(value)

                    prior_gradients = bin_reference_gradients if natural_gradient_enabled else ()
                    for previous_noise_replica, previous_gradient in prior_gradients:
                        comparison = previous_gradient.cosine(reference_gradient)
                        previous_norm = float(comparison["reference_norm"])
                        current_norm = float(comparison["candidate_norm"])
                        difference_norm = float(comparison["difference_norm"])
                        cosine = float(comparison["cosine"])
                        local_natural_gradient_rows.append(
                            {
                                "image_key": image_key,
                                "source_group": source_group,
                                "timestep_bin": bin_index,
                                "timestep": int(probe.timesteps.reshape(-1)[0].item()),
                                "noise_replica_a": previous_noise_replica,
                                "noise_replica_b": noise_replica,
                                "grad_norm_a": previous_norm,
                                "grad_norm_b": current_norm,
                                "grad_diff_norm": difference_norm,
                                "gradient_cosine": cosine,
                                "relative_gradient_distance_a_to_b": (
                                    difference_norm / max(previous_norm, 1e-30)
                                ),
                                "symmetric_gradient_distance": symmetric_gradient_distance(
                                    previous_norm,
                                    current_norm,
                                    difference_norm,
                                ),
                                "angular_gradient_distance": angular_gradient_distance(cosine),
                                "gradient_gain_distance": gradient_gain_distance(
                                    previous_norm,
                                    current_norm,
                                ),
                                "gradient_topology_matches": bool(
                                    comparison["topology_matches"]
                                ),
                                "probe_regime": "structural_dropout_off",
                            }
                        )
                    if natural_gradient_enabled:
                        bin_reference_gradients.append((noise_replica, reference_gradient))

                    if noise_replica >= candidate_noise_replicas:
                        continue
                    for candidate in candidates:
                        for quant_repeat in range(quant_repeats):
                            seed_step_rng(
                                self.protocol_seed,
                                model_seed_id,
                                phase="v2_tail_structural_model",
                                repeat=0,
                            )
                            candidate_row, candidate_gradient, _ = self._run_pass(
                                replay=probe,
                                candidate=candidate,
                                range_mul=candidate.initial_range_mul,
                                phase="v2_tail_probe",
                                probe_or_step=probe_id,
                                repeat=quant_repeat,
                                dropout_enabled=False,
                                shadow=False,
                                update=False,
                                do_auto_observation=False,
                                absolute_step=int(snapshot.metadata["global_step"]),
                                epoch=int(snapshot.metadata["epoch"]),
                                accelerator=accelerator,
                                network=network,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                grad_norm_guardian=grad_norm_guardian,
                                unet=unet,
                                text_encoders=text_encoders,
                                tokenizers=tokenizers,
                                train_unet=train_unet,
                                train_text_encoder=train_text_encoder,
                                training_model=training_model,
                                on_step_start=on_step_start,
                                weight_dtype=weight_dtype,
                                noise_scheduler=noise_scheduler,
                            )
                            cosine = reference_gradient.cosine(candidate_gradient)
                            per_image_rows.append(
                                {
                                    **candidate_row,
                                    **base_metadata,
                                    "quant_repeat": quant_repeat,
                                    "parameter_gradient_cosine": cosine["cosine"],
                                    "gradient_topology_matches": cosine["topology_matches"],
                                }
                            )

        intrinsic_rows = intrinsic_noise_rows(
            [row for row in per_image_rows if row["candidate"] == "no_quant"]
        )
        intrinsic_summary = bootstrap_intrinsic_noise(
            intrinsic_rows,
            timestep_bins=bins,
            iterations=2000,
            seed=self.protocol_seed + 2101,
        )
        tail_samples = gradient_tail_rows(per_image_rows)
        tail_summaries = summarize_gradient_tail(tail_samples, timestep_bins=bins)
        if self.profile_protocol == "v2-tail-calibration":
            tail_bootstrap = bootstrap_tail_winner(
                tail_samples,
                timestep_bins=bins,
                iterations=2000,
                seed=self.protocol_seed + 2102,
            )
        else:
            tail_bootstrap = {
                "primary_metric": "max_t q95(relative_gradient_distance)",
                "relative_gradient_distance_definition": (
                    "sqrt(1+r^2-2*r*cosine)"
                ),
                "decision": "not_applicable_safety_acceptance_profile",
                "candidate_grid": [
                    float(candidate.initial_range_mul) for candidate in candidates
                ],
                "bootstrap_deferred_to_v232_analysis": True,
                "selector_or_utility_vote": False,
            }
        for row in tail_samples:
            row["record_type"] = "sample"
        for row in tail_summaries:
            row["record_type"] = "summary"

        if geometry_enabled:
            sketch_names = sorted(sketchers)
            agreement = sketch_agreement(
                np.asarray(sketches[sketch_names[0]]),
                np.asarray(sketches[sketch_names[1]]),
            )
            geometry = hierarchical_geometry_variance(
                np.asarray(sketches[sketch_names[0]]),
                sketch_metadata,
            )
            structural_rows: list[dict[str, Any]] = []
            for bin_index in range(bins):
                values = sketches_by_bin[sketch_names[0]][bin_index]
                rank = gram_and_rank(values)
                structural_rows.append(
                    {
                        "probe_regime": "structural_dropout_off",
                        "timestep_bin": bin_index,
                        "image_count": len(selected),
                        "noise_replica_count": no_quant_noise_replicas,
                        "probe_count": len(values),
                        "effective_rank": rank["effective_rank"],
                        "stable_rank": rank["stable_rank"],
                        "gradient_noise_scale": gradient_noise_scale(values),
                        "sketch_name": sketch_names[0],
                        "sketch_width": 512,
                        "sketch_stable": agreement.get("stable"),
                    }
                )
            components = (
                "source",
                "image_within_source",
                "timestep",
                "source_timestep_interaction",
                "repeat_noise_residual",
            )
            geometry_rows = [
                {
                    "probe_regime": "structural_dropout_off",
                    "component": component,
                    "energy": geometry.get(component),
                    "fraction": geometry.get(f"{component}_fraction"),
                    "valid": geometry.get("valid", False),
                    "design_unbalanced": geometry.get("design_unbalanced"),
                    "estimable": geometry.get(
                        f"{component}_estimable",
                        True,
                    ),
                    "effective_sketch": sketch_names[0],
                    "sketch_stable": agreement.get("stable"),
                }
                for component in components
            ]
            geometry_summary = {
                "probe_regime": "structural_dropout_off",
                "variance": geometry,
                "source_group_map": source_group_map.manifest(),
                "primary_sketch_width": 512,
                "primary_sketch_seed_count": 2,
                "primary_sketch_agreement": agreement,
                "effective_sketch": sketch_names[0],
                "effective_sketch_agreement": agreement,
                "parameter_gradient_probe_candidates": [
                    item.name for item in candidates
                ],
                "parameter_gradient_probe_repeats": quant_repeats,
                "range_confidence_modifiers": [],
                "geometry_is_candidate_vote": False,
            }
            arrays = {
                name: np.asarray(values) for name, values in sketches.items()
            }
            arrays["tail_image_keys"] = np.asarray(
                [str(item["image_key"]) for item in sketch_metadata],
                dtype=str,
            )
            arrays["tail_timestep_bins"] = np.asarray(
                [int(item["timestep_bin"]) for item in sketch_metadata],
                dtype=np.int16,
            )
            arrays["tail_noise_replicas"] = np.asarray(
                [int(item["noise_replica"]) for item in sketch_metadata],
                dtype=np.int16,
            )
        else:
            structural_rows = []
            geometry_rows = []
            arrays = {}
            geometry = {}
            geometry_summary = {
                "probe_regime": "structural_dropout_off",
                "valid": False,
                "invalid_reason": "disabled_in_default_safety_acceptance_protocol",
                "source_group_map": source_group_map.manifest(),
                "effective_sketch_agreement": None,
                "geometry_is_candidate_vote": False,
            }
        self._tail_probe_result = {
            "intrinsic_noise_rows": intrinsic_rows,
            "intrinsic_noise_summary": intrinsic_summary,
            "local_natural_gradient_rows": local_natural_gradient_rows,
            "gradient_tail_rows": tail_samples + tail_summaries,
            "gradient_tail_samples": tail_samples,
            "tail_bootstrap": tail_bootstrap,
            "structural_repeat_noise_residual_fraction": geometry.get(
                "repeat_noise_residual_fraction"
            ),
        }
        snapshot.restore(
            network=unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=getattr(accelerator, "scaler", None),
            trainer=self.trainer,
            guardian=grad_norm_guardian,
        )
        return (
            per_image_rows,
            [],
            structural_rows,
            arrays,
            geometry_rows,
            geometry_summary,
        )


    def _run_counterfactual_probes(
        self,
        *,
        sequence: ReplaySequence,
        snapshot: TrainingSnapshot,
        accelerator: Any,
        network: Any,
        optimizer: Any,
        lr_scheduler: Any,
        grad_norm_guardian: Any,
        unet: Any,
        text_encoders: list[Any],
        tokenizers: list[Any],
        train_unet: bool,
        train_text_encoder: bool,
        training_model: Any,
        on_step_start: Any,
        weight_dtype: torch.dtype,
        noise_scheduler: Any,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
        dict[str, np.ndarray],
        list[dict[str, Any]],
        dict[str, Any],
    ]:
        if self.profile_protocol in {
            "v2-tail-calibration",
            "v23-safety-local",
            "v23-safety-formal",
            "v24-acceptance-local",
            "v24-acceptance-formal",
            "v24-trajectory-descriptive",
        }:
            return self._run_tail_probes(
                sequence=sequence,
                snapshot=snapshot,
                accelerator=accelerator,
                network=network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_norm_guardian=grad_norm_guardian,
                unet=unet,
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                train_unet=train_unet,
                train_text_encoder=train_text_encoder,
                training_model=training_model,
                on_step_start=on_step_start,
                weight_dtype=weight_dtype,
                noise_scheduler=noise_scheduler,
            )
        if self.profile_protocol == "v2-prefix-smoke":
            return (
                [],
                [],
                [],
                {},
                [],
                {
                    "probe_regime": "not_run",
                    "valid": False,
                    "invalid_reason": "prefix_smoke_omits_structural_and_shadow_probes",
                    "source_group_map": None,
                    "effective_sketch_agreement": None,
                },
            )
        unwrapped = accelerator.unwrap_model(network)
        snapshot.restore(
            network=unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=getattr(accelerator, "scaler", None),
            trainer=self.trainer,
            guardian=grad_norm_guardian,
        )
        selected = sequence.unique_image_items(int(self.args.dq_profile_max_images))
        bins = int(self.args.dq_profile_timestep_bins)
        repeats = int(self.args.dq_profile_stochastic_repeats)
        probe_replicas = int(getattr(self.args, "dq_profile_probe_replicas_resolved", 1))
        sketch_width = int(getattr(self.args, "dq_profile_sketch_width", 4096))
        sketch_seed_count = (
            max(2, int(getattr(self.args, "dq_profile_sketch_seeds", 2)))
            if self.profile_protocol != "v1"
            else 1
        )
        sketchers: dict[str, CountSketch] = {
            f"primary_seed_{index}": CountSketch(
                width=sketch_width,
                seed=self.protocol_seed + 1_000_003 * index,
            )
            for index in range(sketch_seed_count)
        }
        if self.profile_protocol != "v1" and sketch_width < 1024:
            for index in range(2):
                sketchers[f"fallback1024_seed_{index}"] = CountSketch(
                    width=1024,
                    seed=self.protocol_seed + 9_000_001 + 1_000_003 * index,
                )
        sketches_all: dict[str, list[np.ndarray]] = {name: [] for name in sketchers}
        sketches_by_bin: dict[str, dict[int, list[np.ndarray]]] = {
            name: defaultdict(list) for name in sketchers
        }
        sketch_ids_by_bin: dict[int, list[str]] = defaultdict(list)
        sketch_metadata: list[dict[str, Any]] = []
        source_group_map = SourceGroupMap.load(getattr(self.args, "dq_profile_source_group_map", None))
        per_image_rows: list[dict[str, Any]] = []
        raw_shadow_rows: list[dict[str, Any]] = []
        no_quant = self.candidates[0]
        probe_candidates = list(self.candidates[1:])
        candidate_gradient_repeats = repeats
        if self.profile_protocol != "v1" and probe_candidates:
            ordered = sorted(probe_candidates, key=lambda item: float(item.initial_range_mul))
            representative_indices = {0, len(ordered) - 1}
            representative_indices.add(
                min(
                    range(len(ordered)),
                    key=lambda index: abs(float(ordered[index].initial_range_mul) - 3.15),
                )
            )
            probe_candidates = [ordered[index] for index in sorted(representative_indices)]
            candidate_gradient_repeats = 1

        for image_index, source in enumerate(selected):
            image_key = source.image_keys[0] if source.image_keys else source.digest
            source_group = source_group_map.resolve(image_key)
            for bin_index, probe_replica in (
                (bin_index, probe_replica)
                for bin_index in range(bins)
                for probe_replica in range(probe_replicas)
            ):
                probe = self._fixed_timestep_item(
                    source,
                    probe_replica=probe_replica,
                    bin_index=bin_index,
                    bin_count=bins,
                    noise_scheduler=noise_scheduler,
                )
                probe_id = f"{image_index}:{bin_index}:{probe_replica}:{image_key}"
                model_seed_id = (image_index * bins + bin_index) * probe_replicas + probe_replica
                seed_step_rng(self.protocol_seed, model_seed_id, phase="counterfactual_model")
                reference_row, reference_gradient, shadow_rows = self._run_pass(
                    replay=probe,
                    candidate=no_quant,
                    range_mul=None,
                    phase="counterfactual",
                    probe_or_step=probe_id,
                    repeat=0,
                    dropout_enabled=False,
                    shadow=True,
                    update=False,
                    do_auto_observation=False,
                    absolute_step=int(snapshot.metadata["global_step"]),
                    epoch=int(snapshot.metadata["epoch"]),
                    accelerator=accelerator,
                    network=network,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    grad_norm_guardian=grad_norm_guardian,
                    unet=unet,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    train_unet=train_unet,
                    train_text_encoder=train_text_encoder,
                    training_model=training_model,
                    on_step_start=on_step_start,
                    weight_dtype=weight_dtype,
                    noise_scheduler=noise_scheduler,
                )
                raw_shadow_rows.extend(
                    {
                        **row,
                        "source_group": source_group,
                        "image_key": image_key,
                        "timestep_bin": bin_index,
                        "probe_replica": probe_replica,
                        "probe_regime": "structural_dropout_off",
                    }
                    for row in shadow_rows
                )
                probe_metadata = {
                    "image_key": image_key,
                    "source_group": source_group,
                    "timestep_bin": bin_index,
                    "probe_replica": probe_replica,
                    "timestep": int(probe.timesteps.reshape(-1)[0].item()),
                }
                sketch_metadata.append(probe_metadata)
                for sketch_name, sketcher in sketchers.items():
                    sketch = sketcher.sketch(reference_gradient)
                    sketches_all[sketch_name].append(sketch)
                    if probe_replica == 0:
                        sketches_by_bin[sketch_name][bin_index].append(sketch)
                if probe_replica == 0:
                    sketch_ids_by_bin[bin_index].append(image_key)
                per_image_rows.append(
                    {
                        **reference_row,
                        "image_key": image_key,
                        "source_group": source_group,
                        "timestep_bin": bin_index,
                        "probe_replica": probe_replica,
                        "timestep": int(probe.timesteps.reshape(-1)[0].item()),
                        "parameter_gradient_cosine": 1.0,
                        "probe_regime": "structural_dropout_off",
                    }
                )
                for candidate in probe_candidates:
                    for repeat in range(candidate_gradient_repeats):
                        seed_step_rng(self.protocol_seed, model_seed_id, phase="counterfactual_model")
                        candidate_row, candidate_gradient, _ = self._run_pass(
                            replay=probe,
                            candidate=candidate,
                            range_mul=candidate.initial_range_mul,
                            phase="counterfactual",
                            probe_or_step=probe_id,
                            repeat=repeat,
                            dropout_enabled=False,
                            shadow=False,
                            update=False,
                            do_auto_observation=False,
                            absolute_step=int(snapshot.metadata["global_step"]),
                            epoch=int(snapshot.metadata["epoch"]),
                            accelerator=accelerator,
                            network=network,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            grad_norm_guardian=grad_norm_guardian,
                            unet=unet,
                            text_encoders=text_encoders,
                            tokenizers=tokenizers,
                            train_unet=train_unet,
                            train_text_encoder=train_text_encoder,
                            training_model=training_model,
                            on_step_start=on_step_start,
                            weight_dtype=weight_dtype,
                            noise_scheduler=noise_scheduler,
                        )
                        cosine = reference_gradient.cosine(candidate_gradient)
                        per_image_rows.append(
                            {
                                **candidate_row,
                                "image_key": image_key,
                                "source_group": source_group,
                                "timestep_bin": bin_index,
                                "probe_replica": probe_replica,
                                "timestep": int(probe.timesteps.reshape(-1)[0].item()),
                                "parameter_gradient_cosine": cosine["cosine"],
                                "gradient_topology_matches": cosine["topology_matches"],
                                "probe_regime": "structural_dropout_off",
                            }
                        )

        structural_rows: list[dict[str, Any]] = []
        arrays: dict[str, np.ndarray] = {}
        primary_names = sorted(name for name in sketchers if name.startswith("primary_seed_"))
        fallback_names = sorted(name for name in sketchers if name.startswith("fallback1024_seed_"))
        primary_agreement = (
            sketch_agreement(
                np.asarray(sketches_all[primary_names[0]]),
                np.asarray(sketches_all[primary_names[1]]),
            )
            if len(primary_names) >= 2
            else {"stable": True, "not_measured": True, "reason": "v1_single_sketch_seed"}
        )
        fallback_agreement = (
            sketch_agreement(
                np.asarray(sketches_all[fallback_names[0]]),
                np.asarray(sketches_all[fallback_names[1]]),
            )
            if len(fallback_names) >= 2
            else None
        )
        fallback_used = bool(not primary_agreement.get("stable", False) and fallback_names)
        effective_names = fallback_names if fallback_used else primary_names
        effective_name = effective_names[0]
        effective_agreement = fallback_agreement if fallback_used else primary_agreement
        geometry = hierarchical_geometry_variance(
            np.asarray(sketches_all[effective_name]),
            sketch_metadata,
        )
        captions = captions_from_batches(item.batch for item in selected)
        caption_metrics = caption_tag_metrics(captions)
        caption_metrics["common_tag_reuse_rate"] = caption_metrics.get("reusable_tag_fraction")
        caption_metrics["throwaway_or_singleton_tag_fraction"] = caption_metrics.get("singleton_tag_fraction")
        modifiers: list[str] = []
        if not source_group_map.rules:
            modifiers.append("source_group_map_missing_image_within_source_not_estimable")
        if not bool((effective_agreement or {}).get("stable", False)):
            modifiers.append("downgrade_confidence_sketch_instability")
        if bool(geometry.get("design_unbalanced", False)):
            modifiers.append("caution_unbalanced_source_design")
        if float(geometry.get("repeat_noise_residual_fraction", 0.0) or 0.0) > 0.25:
            modifiers.append("additional_repeat_recommended")
        if float(geometry.get("timestep_fraction", 0.0) or 0.0) > 0.40:
            modifiers.append("timestep_nonstationarity_high")
        geometry_summary = {
            "probe_regime": "structural_dropout_off",
            "variance": geometry,
            "source_group_map": source_group_map.manifest(),
            "caption_metrics": caption_metrics,
            "primary_sketch_width": sketch_width,
            "primary_sketch_seed_count": len(primary_names),
            "primary_sketch_agreement": primary_agreement,
            "fallback_1024_available": bool(fallback_names),
            "parameter_gradient_probe_candidates": [candidate.name for candidate in probe_candidates],
            "parameter_gradient_probe_repeats": candidate_gradient_repeats,
            "fallback_1024_used": fallback_used,
            "fallback_1024_agreement": fallback_agreement,
            "effective_sketch": effective_name,
            "effective_sketch_agreement": effective_agreement,
            "range_confidence_modifiers": modifiers,
            "geometry_is_candidate_vote": False,
        }
        components = (
            "source",
            "image_within_source",
            "timestep",
            "source_timestep_interaction",
            "repeat_noise_residual",
        )
        geometry_rows = [
            {
                "probe_regime": "structural_dropout_off",
                "component": component,
                "energy": geometry.get(component),
                "fraction": geometry.get(f"{component}_fraction"),
                "valid": geometry.get("valid", False),
                "design_unbalanced": geometry.get("design_unbalanced"),
                "estimable": (
                    geometry.get("image_within_source_estimable")
                    if component == "image_within_source"
                    else geometry.get("repeat_noise_residual_estimable")
                    if component == "repeat_noise_residual"
                    else True
                ),
                "effective_sketch": effective_name,
                "sketch_stable": (effective_agreement or {}).get("stable"),
            }
            for component in components
        ]
        for bin_index in range(bins):
            sketches = sketches_by_bin[effective_name][bin_index]
            result = gram_and_rank(sketches)
            structural_rows.append(
                {
                    "probe_regime": "structural_dropout_off",
                    "timestep_bin": bin_index,
                    "image_count": len(sketches),
                    "effective_rank": result["effective_rank"],
                    "stable_rank": result["stable_rank"],
                    "gradient_noise_scale": gradient_noise_scale(sketches),
                    "sketch_name": effective_name,
                    "sketch_width": int(sketchers[effective_name].width),
                    "sketch_stable": (effective_agreement or {}).get("stable"),
                }
            )
            arrays[f"sketches_bin_{bin_index}"] = np.asarray(sketches)
            arrays[f"gram_bin_{bin_index}"] = result["gram"]
            arrays[f"eigenvalues_bin_{bin_index}"] = result["eigenvalues"]
            arrays[f"image_ids_bin_{bin_index}"] = np.asarray(sketch_ids_by_bin[bin_index], dtype=str)
        for sketch_name, sketches in sketches_all.items():
            arrays[f"{sketch_name}_all"] = np.asarray(sketches)
            for bin_index in range(bins):
                arrays[f"{sketch_name}_bin_{bin_index}"] = np.asarray(sketches_by_bin[sketch_name][bin_index])
        arrays["geometry_image_keys"] = np.asarray([row["image_key"] for row in sketch_metadata], dtype=str)
        arrays["geometry_source_groups"] = np.asarray([row["source_group"] for row in sketch_metadata], dtype=str)
        arrays["geometry_timestep_bins"] = np.asarray([row["timestep_bin"] for row in sketch_metadata], dtype=np.int64)
        arrays["geometry_probe_replicas"] = np.asarray([row["probe_replica"] for row in sketch_metadata], dtype=np.int64)
        return per_image_rows, raw_shadow_rows, structural_rows, arrays, geometry_rows, geometry_summary

    def _run_branches(
        self,
        *,
        sequence: ReplaySequence,
        snapshot: TrainingSnapshot,
        accelerator: Any,
        network: Any,
        optimizer: Any,
        lr_scheduler: Any,
        grad_norm_guardian: Any,
        unet: Any,
        text_encoders: list[Any],
        tokenizers: list[Any],
        train_unet: bool,
        train_text_encoder: bool,
        training_model: Any,
        on_step_start: Any,
        weight_dtype: torch.dtype,
        noise_scheduler: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        unwrapped = accelerator.unwrap_model(network)
        trajectory_rows: list[dict[str, Any]] = []
        branch_summaries: dict[str, dict[str, Any]] = {}
        for candidate in self.candidates:
            snapshot.restore(
                network=unwrapped,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                scaler=getattr(accelerator, "scaler", None),
                trainer=self.trainer,
                guardian=grad_norm_guardian,
            )
            controller = None
            if candidate.quantized:
                controller = AutoRangeController(
                    candidate,
                    every=int(self.args.dq_delta_auto_every),
                    ema=float(self.args.dq_delta_auto_ema),
                    mul_up=float(self.args.dq_delta_auto_mul_up),
                    mul_down=float(self.args.dq_delta_auto_mul_down),
                    minimum=float(self.args.dq_delta_auto_min),
                    maximum=float(self.args.dq_delta_auto_max),
                    warmup=bool(self.args.dq_delta_auto_warmup),
                    warmup_updates=int(self.args.dq_delta_auto_warmup_updates),
                    use_raw=bool(self.args.dq_delta_auto_use_raw),
                )
            candidate_rows: list[dict[str, Any]] = []
            for branch_step, replay in enumerate(sequence):
                absolute_step = int(snapshot.metadata["global_step"]) + branch_step
                seed_step_rng(self.protocol_seed, branch_step, phase="branch_model")
                observe_auto = bool(
                    controller is not None
                    and ((absolute_step + 1) % int(self.args.dq_delta_auto_every) == 0)
                )
                range_mul = None if controller is None else controller.range_mul
                row, _, _ = self._run_pass(
                    replay=replay,
                    candidate=candidate,
                    range_mul=range_mul,
                    phase="branch",
                    probe_or_step=branch_step,
                    repeat=0,
                    dropout_enabled=True,
                    shadow=False,
                    update=True,
                    do_auto_observation=observe_auto,
                    absolute_step=absolute_step,
                    epoch=int(snapshot.metadata["epoch"]),
                    accelerator=accelerator,
                    network=network,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    grad_norm_guardian=grad_norm_guardian,
                    unet=unet,
                    text_encoders=text_encoders,
                    tokenizers=tokenizers,
                    train_unet=train_unet,
                    train_text_encoder=train_text_encoder,
                    training_model=training_model,
                    on_step_start=on_step_start,
                    weight_dtype=weight_dtype,
                    noise_scheduler=noise_scheduler,
                )
                auto_row: dict[str, Any] = {
                    "auto_observed": False,
                    "auto_reason": "not_observation_step",
                    "range_mul_before": range_mul,
                    "range_mul_after": range_mul,
                }
                if controller is not None and observe_auto:
                    auto_row = {"auto_observed": True, **controller.observe(absolute_step + 1, row.get("clip_rate"))}
                row.update(
                    {
                        **auto_row,
                        "branch_step": branch_step,
                        "absolute_step": absolute_step + 1,
                        "image_keys": "|".join(replay.image_keys),
                        "replay_digest": replay.digest,
                        "branch_regime": "training_dropout_on",
                    }
                )
                candidate_rows.append(row)
                trajectory_rows.append(row)

            losses = [float(row["loss"]) for row in candidate_rows]
            validity = (
                controller.validity()
                if controller is not None
                else {
                    "auto_observation_count": 0,
                    "auto_post_warmup_observation_count": 0,
                    "auto_warmup_completed": None,
                    "auto_trajectory_metrics_valid": False,
                    "auto_invalid_reason": "not_applicable_no_quant",
                }
            )
            branch_summaries[candidate.name] = {
                "candidate": candidate.name,
                "initial_range_mul": candidate.initial_range_mul,
                "final_range_mul": None if controller is None else controller.range_mul,
                "branch_loss_mean": sum(losses) / max(len(losses), 1),
                "branch_loss_std": math.sqrt(
                    sum((value - (sum(losses) / max(len(losses), 1))) ** 2 for value in losses) / max(len(losses), 1)
                ),
                "branch_parameter_update_norm": _gradient_update_norm(unwrapped, snapshot.network_state),
                "branch_steps": len(candidate_rows),
                "branch_regime": "training_dropout_on",
                **validity,
            }
        snapshot.restore(
            network=unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=getattr(accelerator, "scaler", None),
            trainer=self.trainer,
            guardian=grad_norm_guardian,
        )
        return trajectory_rows, branch_summaries

    def _finish_snapshot_only(
        self,
        *,
        accelerator: Any,
        snapshot: TrainingSnapshot,
        snapshot_components: Mapping[str, Any],
        snapshot_fingerprints: Mapping[str, str],
        first_batch: Mapping[str, Any],
        dq_delta_begin_step: int,
        num_train_epochs: int,
    ) -> dict[str, Any]:
        """Persist the warmup boundary without running any probe or branch."""

        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = Path(self.args.dataset_config).resolve()
        manifest_additional_files = [dataset_path]
        if getattr(self.args, "dq_profile_source_group_map", None):
            manifest_additional_files.append(
                Path(self.args.dq_profile_source_group_map).resolve()
            )
        source_manifest, _ = build_source_manifest(
            repo_root,
            quant_rng_mode=self.quant_context.rng_mode,
            additional_files=tuple(manifest_additional_files),
        )
        source_manifest["dataset_config"] = {
            "path": str(dataset_path),
            "size": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
        }
        source_manifest["dq_profile_v2"] = {
            "profile_protocol": self.profile_protocol,
            "diagnostic_target": "cross_process_snapshot_reproducibility",
            "snapshot_only": True,
            "kernel_policy": getattr(
                self.args, "dq_profile_prefix_kernel_policy", None
            ),
        }
        source_contract = source_contract_from_manifest(source_manifest)
        source_manifest["source_contract"] = {
            "sha256": source_contract["sha256"],
            "definition": (
                "schema/protocol/runtime/packages/CUDA/explicit "
                "sources/additional inputs"
            ),
        }
        source_manifest_hash = canonical_sha256(source_manifest)
        preflight = _finite_json(getattr(self.args, "dq_profile_preflight", {}))
        warmup_contract = _snapshot_warmup_contract(
            self.args,
            preflight,
            dataset_sha256=source_manifest["dataset_config"]["sha256"],
            source_contract_sha256=source_contract["sha256"],
            dq_delta_begin_step=dq_delta_begin_step,
        )

        state_path = self.artifacts.root / "snapshot_state.pt"
        temporary_state_path = self.artifacts.root / ".snapshot_state.pt.tmp"
        try:
            torch.save(dict(snapshot_components), temporary_state_path)
            os.replace(temporary_state_path, state_path)
        finally:
            if temporary_state_path.exists():
                temporary_state_path.unlink()
        state_sha256 = sha256_file(state_path)

        snapshot_payload = {
            **snapshot.metadata,
            "fingerprints": dict(snapshot_fingerprints),
            "dq_delta_begin_step": int(dq_delta_begin_step),
            "boundary": (
                "after_last_unquantized_update_before_first_quantized_batch"
            ),
            "first_quantized_batch_fingerprint": fingerprint_tree(first_batch),
            "state_file": state_path.name,
            "state_file_sha256": state_sha256,
        }
        summary = _finite_json(
            {
                "schema_version": SCHEMA_VERSION,
                "metric_definition_version": METRIC_DEFINITION_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "source_manifest_sha256": source_manifest_hash,
                "source_contract_sha256": source_contract["sha256"],
                "diagnostic_target": "cross_process_snapshot_reproducibility",
                "dataset": preflight,
                "snapshot": snapshot_payload,
                "warmup_contract": warmup_contract,
                "warmup_contract_sha256": canonical_sha256(warmup_contract),
                "profile": {
                    "protocol": self.profile_protocol,
                    "snapshot_only": True,
                    "quant_rng_mode": self.quant_context.rng_mode,
                    "num_train_epochs": int(num_train_epochs),
                    "kernel_policy": getattr(
                        self.args, "dq_profile_prefix_kernel_policy", None
                    ),
                },
                "probes_performed": False,
                "branches_performed": False,
                "warnings": list(self._warnings),
            }
        )
        sensitive_arg_names = {"wandb_api_key", "huggingface_token"}
        resolved_args: dict[str, Any] = {}
        for key, value in sorted(vars(self.args).items()):
            lowered = key.lower()
            if (
                key in sensitive_arg_names
                or "password" in lowered
                or "secret" in lowered
            ):
                resolved_args[key] = (
                    None if value in (None, "") else "<redacted>"
                )
            else:
                resolved_args[key] = _finite_json(value)

        self.artifacts.write_json(
            "probe_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "profile_protocol": self.profile_protocol,
                "probe_regime": "not_run",
                "branch_regime": "not_run",
                "snapshot_only": True,
                "first_quantized_batch_fingerprint": snapshot_payload[
                    "first_quantized_batch_fingerprint"
                ],
                "snapshot_fingerprints": dict(snapshot_fingerprints),
            },
        )
        self.artifacts.write_json("source_manifest.json", _finite_json(source_manifest))
        self.artifacts.write_json("summary.json", summary)
        self.artifacts.write_json("snapshot_smoke.json", summary)
        self.artifacts.write_json("resolved_args.json", resolved_args)
        (self.artifacts.root / "profile.log").write_text(
            json.dumps(
                {
                    "snapshot_step": snapshot.metadata["global_step"],
                    "snapshot_only": True,
                    "probes_performed": False,
                    "branches_performed": False,
                    "snapshot_state_bytes": state_path.stat().st_size,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_hash = canonical_sha256(summary)
        self.artifacts.mark_complete(summary_hash)
        accelerator.end_training()
        return summary

    def run_from_boundary(
        self,
        *,
        accelerator: Any,
        network: Any,
        optimizer: Any,
        lr_scheduler: Any,
        grad_norm_guardian: Any,
        global_step: int,
        epoch: int,
        data_step: int,
        first_batch: Mapping[str, Any],
        epoch_iterator: Iterable[Any],
        train_dataloader: Iterable[Any],
        current_epoch: Any,
        current_step: Any,
        num_train_epochs: int,
        vae: Any,
        vae_dtype: torch.dtype,
        weight_dtype: torch.dtype,
        noise_scheduler: Any,
        unet: Any,
        text_encoders: list[Any],
        tokenizers: list[Any],
        train_unet: bool,
        train_text_encoder: bool,
        training_model: Any,
        on_step_start: Any,
        dq_delta_begin_step: int,
    ) -> dict[str, Any]:
        if int(global_step) != int(dq_delta_begin_step):
            raise RuntimeError(
                f"profile snapshot boundary mismatch: global_step={global_step}, dq_delta_begin_step={dq_delta_begin_step}"
            )
        if accelerator.num_processes != 1:
            raise ValueError("DQ Dataset Profiler v1 requires exactly one process/GPU")
        unwrapped = accelerator.unwrap_model(network)
        if type(unwrapped).__module__ != "dq_profile.copied_lora":
            raise RuntimeError(
                "diagnostic isolation violation: expected network from dq_profile.copied_lora, "
                f"got {type(unwrapped).__module__}.{type(unwrapped).__name__}"
            )
        self.artifacts.initialize()
        self.artifacts.ensure_known_result()
        self.artifacts.copy_dataset_config(self.args.dataset_config)
        if getattr(self.args, "dq_delta_auto_preset", None) == "clip_rate_low_auto":
            self._warnings.append(
                "The requested normal-training preset clip_rate_low_auto is not a v1 candidate. "
                "The profiler uses clip_rate_low; early branches do not evaluate the low-auto escape condition."
            )
        snapshot = TrainingSnapshot.capture(
            network=unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=getattr(accelerator, "scaler", None),
            trainer=self.trainer,
            guardian=grad_norm_guardian,
            global_step=global_step,
            epoch=epoch,
            data_step=data_step,
        )
        snapshot_components = {
            "network": snapshot.network_state,
            "optimizer": snapshot.optimizer_state,
            "scheduler": snapshot.scheduler_state,
            "scaler": snapshot.scaler_state,
            "rng": snapshot.rng_state,
            "network_runtime": snapshot.network_runtime,
            "trainer": snapshot.trainer_state,
            "guardian": snapshot.guardian_state,
            "metadata": snapshot.metadata,
        }
        snapshot_fingerprints = {
            name: fingerprint_tree(value)
            for name, value in snapshot_components.items()
        }
        snapshot_fingerprints["combined"] = canonical_sha256(
            snapshot_fingerprints
        )
        if bool(getattr(self.args, "dq_profile_snapshot_only", False)):
            return self._finish_snapshot_only(
                accelerator=accelerator,
                snapshot=snapshot,
                snapshot_components=snapshot_components,
                snapshot_fingerprints=snapshot_fingerprints,
                first_batch=first_batch,
                dq_delta_begin_step=dq_delta_begin_step,
                num_train_epochs=num_train_epochs,
            )
        sequence = self._capture_batches(
            first_batch=first_batch,
            epoch_iterator=epoch_iterator,
            train_dataloader=train_dataloader,
            current_epoch=current_epoch,
            current_step=current_step,
            global_step=global_step,
            epoch=epoch,
            data_step=data_step,
            count=int(
                self.args.dq_profile_branch_steps_resolved
                if self.profile_protocol == "v1"
                else self.args.dq_profile_capture_steps
            ),
        )
        snapshot.restore(
            network=unwrapped,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            scaler=getattr(accelerator, "scaler", None),
            trainer=self.trainer,
            guardian=grad_norm_guardian,
        )
        self._materialize_replay(
            sequence,
            accelerator=accelerator,
            vae=vae,
            vae_dtype=vae_dtype,
            weight_dtype=weight_dtype,
            noise_scheduler=noise_scheduler,
        )
        bin_count = int(self.args.dq_profile_timestep_bins)
        diffusion_steps = int(noise_scheduler.config.num_train_timesteps)
        selected_probe_items = (
            []
            if self.profile_protocol == "v2-prefix-smoke"
            else sequence.unique_image_items(int(self.args.dq_profile_max_images))
        )
        ordered_probe_contract = [
            {
                "image_key": (
                    item.image_keys[0] if item.image_keys else item.digest
                ),
                "batch_digest": item.batch_digest,
                "replay_digest": item.digest,
                "model_seed": item.model_seed,
            }
            for item in selected_probe_items
        ]
        self.artifacts.write_json(
            "probe_manifest.json",
            _finite_json(
                {
                    "schema_version": SCHEMA_VERSION,
                    "branch_regime": "training_dropout_on",
                    "probe_regime": (
                        "not_run"
                        if self.profile_protocol == "v2-prefix-smoke"
                        else "structural_dropout_off"
                    ),
                    "worker_count": 0,
                    "live_dataloader_reused_by_branches": False,
                    "branch_batches": sequence.manifest(),
                    "structural_probe": {
                        "performed": self.profile_protocol != "v2-prefix-smoke",
                        "max_images": (
                            0
                            if self.profile_protocol == "v2-prefix-smoke"
                            else int(self.args.dq_profile_max_images)
                        ),
                        "timestep_bins": bin_count,
                        "timestep_centers": [
                            min(diffusion_steps - 1, max(0, int((index + 0.5) * diffusion_steps / bin_count)))
                            for index in range(bin_count)
                        ],
                        "stochastic_repeats": int(self.args.dq_profile_stochastic_repeats),
                        "probe_replicas": int(getattr(self.args, "dq_profile_probe_replicas_resolved", 1)),
                    },
                    "quant_seed_fields": [
                        "protocol_seed",
                        "phase",
                        "probe_or_step",
                        "module_name",
                        "invocation",
                        "repeat",
                    ],
                    "quant_seed_includes_candidate": False,
                    "snapshot_fingerprints": snapshot_fingerprints,
                    "ordered_probe_contract": ordered_probe_contract,
                    "ordered_probe_contract_sha256": canonical_sha256(
                        ordered_probe_contract
                    ),
                    "first_16_probe_contract_sha256": canonical_sha256(
                        ordered_probe_contract[:16]
                    ),
                }
            ),
        )
        if getattr(self.args, "ip_noise_gamma", 0):
            self._warnings.append(
                "Fixed-timestep structural probes reuse the materialized base noise and omit the additional "
                "input-perturbation noise draw; training-dropout branches retain the production noisy latents."
            )
        preflight_contract = getattr(self.args, "dq_profile_preflight", {})
        if self.args.dq_profile_level == "full" and bool(preflight_contract.get("full_budget_core_exceeded", False)):
            self._warnings.append(
                "The full-profile budget is smaller than the mandatory warmup, core probes, and fixed branch length. "
                "Core work was preserved and no automatic branch/probe reduction was applied."
            )

        (
            per_image_rows,
            raw_shadow_rows,
            structural_rows,
            sketch_arrays,
            geometry_rows,
            geometry_summary,
        ) = self._run_counterfactual_probes(
            sequence=sequence,
            snapshot=snapshot,
            accelerator=accelerator,
            network=network,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            grad_norm_guardian=grad_norm_guardian,
            unet=unet,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
            train_unet=train_unet,
            train_text_encoder=train_text_encoder,
            training_model=training_model,
            on_step_start=on_step_start,
            weight_dtype=weight_dtype,
            noise_scheduler=noise_scheduler,
        )
        v2_result: dict[str, Any] = {
            "update_direction_rows": [],
            "range_sweep_rows": [],
            "guardian_ablation_rows": [],
            "mechanism_ablation_rows": [],
            "execution_manifest_rows": [],
        }
        if self.profile_protocol == "v1":
            trajectory_rows, branch_summaries = self._run_branches(
                sequence=sequence,
                snapshot=snapshot,
                accelerator=accelerator,
                network=network,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                grad_norm_guardian=grad_norm_guardian,
                unet=unet,
                text_encoders=text_encoders,
                tokenizers=tokenizers,
                train_unet=train_unet,
                train_text_encoder=train_text_encoder,
                training_model=training_model,
                on_step_start=on_step_start,
                weight_dtype=weight_dtype,
                noise_scheduler=noise_scheduler,
            )
        else:
            v2_result = run_v2_experiments(
                runtime=self,
                sequence=sequence,
                snapshot=snapshot,
                pass_context={
                    "accelerator": accelerator,
                    "network": network,
                    "optimizer": optimizer,
                    "lr_scheduler": lr_scheduler,
                    "grad_norm_guardian": grad_norm_guardian,
                    "unet": unet,
                    "text_encoders": text_encoders,
                    "tokenizers": tokenizers,
                    "train_unet": train_unet,
                    "train_text_encoder": train_text_encoder,
                    "training_model": training_model,
                    "on_step_start": on_step_start,
                    "weight_dtype": weight_dtype,
                    "noise_scheduler": noise_scheduler,
                },
            )
            trajectory_rows = list(v2_result["trajectory_rows"])
            branch_summaries = dict(v2_result["branch_summaries"])
            probe_manifest_path = self.artifacts.root / "probe_manifest.json"
            probe_manifest_payload = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
            probe_manifest_payload.update(
                {
                    "profile_protocol": self.profile_protocol,
                    "repeat_replay_manifests": v2_result.get("repeat_replay_manifests", {}),
                    "source_group_map": geometry_summary.get("source_group_map"),
                    "sketch_protocol": geometry_summary.get("effective_sketch_agreement"),
                    "common_random_numbers_across_candidates": True,
                }
            )
            self.artifacts.write_json("probe_manifest.json", _finite_json(probe_manifest_payload))
        v2_result["geometry_variance_rows"] = geometry_rows
        v2_result["geometry_summary"] = geometry_summary
        shadow_aggregate = aggregate_shadow_rows(raw_shadow_rows)
        source_stratified_rows = _source_stratified_replay_rows(per_image_rows)

        candidate_rows: list[dict[str, Any]] = []
        candidate_names = (
            [candidate.name for candidate in self.candidates]
            if self.profile_protocol == "v1"
            else sorted(
                branch_summaries,
                key=lambda name: (
                    name != "no_quant",
                    float(branch_summaries[name].get("initial_range_mul") or 0.0),
                ),
            )
        )
        for candidate_name in candidate_names:
            probe_rows = [row for row in per_image_rows if row["candidate"] == candidate_name]
            raw_probe_metrics = aggregate_numeric(
                probe_rows,
                (
                    "loss",
                    "gradient_norm",
                    "parameter_gradient_cosine",
                    "clip_rate",
                    "quant_error_rms",
                    "quant_error_ratio",
                ),
            )
            probe_metrics = {f"probe_{key}": value for key, value in raw_probe_metrics.items()}
            merged = {
                **branch_summaries[candidate_name],
                **probe_metrics,
                "probe_regime": (
                    "not_run"
                    if self.profile_protocol == "v2-prefix-smoke"
                    else "structural_dropout_off"
                ),
            }
            candidate_rows.append(merged)

        manifest_additional_files = [self.args.dataset_config]
        if getattr(self.args, "dq_profile_source_group_map", None):
            manifest_additional_files.append(self.args.dq_profile_source_group_map)
        if getattr(self.args, "dq_profile_core_gate_file", None):
            manifest_additional_files.append(self.args.dq_profile_core_gate_file)
        repo_root = Path(__file__).resolve().parents[1]
        source_manifest, _ = build_source_manifest(
            repo_root,
            quant_rng_mode=self.quant_context.rng_mode,
            additional_files=tuple(manifest_additional_files),
        )
        dataset_path = Path(self.args.dataset_config).resolve()
        source_manifest["dataset_config"] = {
            "path": str(dataset_path),
            "size": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
        }
        source_manifest["dq_profile_v2"] = {
            "profile_protocol": self.profile_protocol,
            "diagnostic_target": (
                "numerical_gradient_acceptance_by_fixed_range_mul"
                if self.profile_protocol
                in {
                    "v23-safety-local",
                    "v23-safety-formal",
                    "v24-acceptance-local",
                    "v24-acceptance-formal",
                    "v24-trajectory-descriptive",
                }
                else "no_quant_trajectory_stability"
                if self.profile_protocol != "v1"
                else None
            ),
            "range_grid": v2_result.get("range_grid", []),
            "branch_repeats_executed": v2_result.get("branch_repeats_executed", []),
            "guardian_ablation": getattr(self.args, "dq_profile_guardian_ablation", None),
            "core_gate": (
                {
                    "path": str(getattr(self.args, "dq_profile_core_gate_file")),
                    "canonical_sha256": getattr(self.args, "dq_profile_core_gate_sha256", None),
                    "profile_key": getattr(self.args, "dq_profile_core_profile_key", None),
                    "approved_mechanism_muls": list(getattr(self.args, "dq_profile_mechanism_muls_resolved", ())),
                }
                if getattr(self.args, "dq_profile_core_gate_file", None) else None
            ),
        }
        source_contract = source_contract_from_manifest(source_manifest)
        source_manifest["source_contract"] = {
            "sha256": source_contract["sha256"],
            "definition": "schema/protocol/runtime/packages/CUDA/explicit sources/additional inputs",
        }
        if self.profile_protocol == "v2-prefix-smoke":
            calibration_gate = dict(v2_result.get("calibration_gate") or {})
            calibration_gate["source_contract_sha256"] = source_contract["sha256"]
            calibration_gate["formal_input_schema"] = SCHEMA_VERSION
            v2_result["calibration_gate"] = calibration_gate
            v2_result["prefix_parity"] = calibration_gate
        elif self.profile_protocol in {
            "v2-tail-calibration",
            "v23-safety-local",
            "v23-safety-formal",
            "v24-acceptance-local",
            "v24-acceptance-formal",
            "v24-trajectory-descriptive",
        }:
            expected_contract = str(
                getattr(self.args, "dq_profile_expected_source_contract_sha256", "")
            )
            if source_contract["sha256"] != expected_contract:
                raise RuntimeError(
                    "source contract changed after preflight; refusing "
                    f"{self.profile_protocol} result"
                )
            source_manifest["prefix_gate"] = {
                "path": str(getattr(self.args, "dq_profile_prefix_gate_file")),
                "sha256": getattr(self.args, "dq_profile_prefix_gate_sha256", None),
                "expected_source_contract_sha256": expected_contract,
                "matched": True,
            }
            if self.profile_protocol in {
                "v23-safety-formal",
                "v24-acceptance-formal",
            }:
                source_manifest["safety_local_selection"] = {
                    "path": str(
                        getattr(
                            self.args,
                            "dq_profile_safety_local_selection_file",
                        )
                    ),
                    "canonical_sha256": getattr(
                        self.args,
                        "dq_profile_safety_local_selection_sha256",
                        None,
                    ),
                    "local_summary_sha256": getattr(
                        self.args,
                        "dq_profile_safety_local_summary_sha256",
                        None,
                    ),
                    "local_profile_dir": getattr(
                        self.args,
                        "dq_profile_safety_local_profile_dir",
                        None,
                    ),
                    "local_grid": list(
                        getattr(
                            self.args,
                            "dq_profile_safety_local_grid",
                            (),
                        )
                    ),
                    "matched": True,
                }
            if self.profile_protocol == "v24-trajectory-descriptive":
                source_manifest["trajectory_contract"] = {
                    "path": str(
                        getattr(
                            self.args,
                            "dq_profile_trajectory_contract_file",
                        )
                    ),
                    "canonical_sha256": getattr(
                        self.args,
                        "dq_profile_trajectory_contract_sha256",
                        None,
                    ),
                    "content_sha256": getattr(
                        self.args,
                        "dq_profile_trajectory_content_sha256",
                        None,
                    ),
                    "local_summary_sha256": getattr(
                        self.args,
                        "dq_profile_trajectory_local_summary_sha256",
                        None,
                    ),
                    "local_profile_dir": getattr(
                        self.args,
                        "dq_profile_trajectory_local_profile_dir",
                        None,
                    ),
                    "local_analysis_dir": getattr(
                        self.args,
                        "dq_profile_trajectory_local_analysis_dir",
                        None,
                    ),
                    "local_grid": list(
                        getattr(
                            self.args,
                            "dq_profile_trajectory_local_grid",
                            (),
                        )
                    ),
                    "candidate_roles": list(
                        getattr(
                            self.args,
                            "dq_profile_trajectory_candidate_roles",
                            (),
                        )
                    ),
                    "edge_unresolved": bool(
                        getattr(
                            self.args,
                            "dq_profile_trajectory_edge_unresolved",
                            False,
                        )
                    ),
                    "descriptive_only": True,
                    "recommendation_allowed": False,
                    "matched": True,
                }
            calibration_gate = dict(v2_result.get("calibration_gate") or {})
            calibration_gate.update(
                {
                    "prefix_gate_sha256": getattr(
                        self.args, "dq_profile_prefix_gate_sha256", None
                    ),
                    "source_contract_sha256": source_contract["sha256"],
                    "source_contract_matched": True,
                }
            )
            v2_result["calibration_gate"] = calibration_gate
        source_manifest_hash = canonical_sha256(source_manifest)
        preflight = _finite_json(getattr(self.args, "dq_profile_preflight", {}))
        current_controls = _current_control_contract(
            self.args,
            preflight,
            source_manifest["dataset_config"]["sha256"],
        )
        known_result = _evaluate_known_result_controls(self.artifacts.read_known_result(), current_controls)
        if known_result["comparison_controlled_requested"] and not known_result["comparison_controlled_effective"]:
            self._warnings.append(
                "known_result comparison_controlled=true was not accepted because controls are missing or different; "
                "see detected_control_differences"
            )
        known_result = _finite_json(known_result)
        utility_summary = {
            "utility_screen_seed39": "not_measured",
            "U_selected_protocol": "unknown",
            "U_any_quantization": "unknown",
            "utility_confidence": "low",
            "quality_margin": None,
            "m_utility": None,
            "checkpoint_primary": "final avg center",
            "checkpoint_secondary": "final raw",
            "rope": [0.45, 0.55],
        }
        v2_summary = {
            "enabled": self.profile_protocol != "v1",
            "profile_protocol": self.profile_protocol,
            "diagnostic_target": (
                "numerical_gradient_acceptance_by_fixed_range_mul"
                if self.profile_protocol
                in {
                    "v23-safety-local",
                    "v23-safety-formal",
                    "v24-acceptance-local",
                    "v24-acceptance-formal",
                    "v24-trajectory-descriptive",
                }
                else "no_quant_trajectory_stability"
            ),
            "intrinsic_stability_result": v2_result.get("intrinsic_stability_result"),
            "guardian_adjusted_result": (
                v2_result.get("guardian_result", {}).get("guardian_adjusted_result")
            ),
            "guardian_dependent": v2_result.get("guardian_result", {}).get("guardian_dependent"),
            "mechanism_result": v2_result.get("mechanism_result"),
            "source_evaluation": {
                "type": "source-stratified replay evaluation",
                "generalization_claim": False,
                "formal_heldout_performed": False,
            },
            "range_grid": v2_result.get("range_grid", []),
            "geometry": v2_result.get("geometry_summary"),
            "edge_extensions": v2_result.get("edge_extensions", []),
            "branch_repeats_executed": v2_result.get("branch_repeats_executed", []),
            "third_repeat_performed": v2_result.get("third_repeat_performed", False),
            "extension_128_performed": v2_result.get("extension_128_performed", False),
            "extension_128_result": v2_result.get("extension_128_result"),
            "prefix_parity": v2_result.get("prefix_parity"),
            "calibration_gate": v2_result.get("calibration_gate"),
            "intrinsic_noise": v2_result.get("intrinsic_noise_summary"),
            "tail_bootstrap": v2_result.get("tail_bootstrap"),
            "fragility_diag": v2_result.get("fragility_diag"),
            "cumulative_null_row_count": len(
                v2_result.get("cumulative_null_rows", [])
            ),
            "source_contract_sha256": source_contract["sha256"],
            "guardian_anchor": v2_result.get("guardian_anchor"),
            "core_continuation_assessment": "requires_cross_dataset_evaluation",
            "utility": utility_summary,
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "source_manifest_sha256": source_manifest_hash,
            "probe_regime": (
                "not_run"
                if self.profile_protocol == "v2-prefix-smoke"
                else "structural_dropout_off"
            ),
            "diagnostic_target": v2_summary["diagnostic_target"] if v2_summary["enabled"] else None,
            "v2": v2_summary,
            "utility": utility_summary,
            "branch_regime": "training_dropout_on",
            "dataset": preflight,
            "snapshot": {
                **snapshot.metadata,
                "fingerprints": snapshot_fingerprints,
                "dq_delta_begin_step": dq_delta_begin_step,
                "boundary": "after_last_unquantized_update_before_first_quantized_batch",
                "first_quantized_batch_digest": sequence[0].digest,
            },
            "profile": {
                "protocol": self.profile_protocol,
                "level": self.args.dq_profile_level,
                "branch_steps": (
                    len(sequence)
                    if self.profile_protocol == "v1"
                    else 0
                    if self.profile_protocol.endswith("-local")
                    else int(self.args.dq_profile_sweep_steps)
                ),
                "replay_capture_batches": len(sequence),
                "probe_images": (
                    0
                    if self.profile_protocol == "v2-prefix-smoke"
                    else len(sequence.unique_image_items(int(self.args.dq_profile_max_images)))
                ),
                "timestep_bins": int(self.args.dq_profile_timestep_bins),
                "stochastic_repeats": int(self.args.dq_profile_stochastic_repeats),
                "probe_replicas": int(getattr(self.args, "dq_profile_probe_replicas_resolved", 1)),
                "quant_rng_mode": self.quant_context.rng_mode,
                "num_train_epochs": int(num_train_epochs),
            },
            "current_controls": current_controls,
            "candidates": candidate_rows,
            "auto_validity": {
                row["candidate"]: {
                    key: row.get(key)
                    for key in (
                        "auto_observation_count",
                        "auto_post_warmup_observation_count",
                        "auto_warmup_completed",
                        "auto_trajectory_metrics_valid",
                        "auto_invalid_reason",
                    )
                }
                for row in candidate_rows
            },
            "structural": structural_rows,
            "known_result": known_result,
            "warnings": self._warnings,
        }
        summary = _finite_json(summary)
        sensitive_arg_names = {"wandb_api_key", "huggingface_token"}
        resolved_args = {}
        for key, value in sorted(vars(self.args).items()):
            lowered = key.lower()
            if key in sensitive_arg_names or "password" in lowered or "secret" in lowered:
                resolved_args[key] = None if value in (None, "") else "<redacted>"
            else:
                resolved_args[key] = _finite_json(value)
        dataset_rows = []
        for subset in preflight.get("subsets", []) if isinstance(preflight, dict) else []:
            dataset_rows.append(subset)
        if self.profile_protocol == "v1":
            candidate_definitions = [candidate.to_dict() for candidate in self.candidates]
        else:
            candidate_definitions = [
                {
                    "name": row["candidate"],
                    "quantized": row["candidate"] != "no_quant",
                    "initial_range_mul": row.get("initial_range_mul"),
                    "auto_enabled": False,
                    "mechanism": row.get("mechanism", "full"),
                }
                for row in candidate_rows
            ]

        self.artifacts.write_json("source_manifest.json", _finite_json(source_manifest))
        self.artifacts.write_json("summary.json", summary)
        self.artifacts.write_json("resolved_args.json", resolved_args)
        self.artifacts.write_json(
            "candidate_definitions.json",
            {
                "protocol_version": PROTOCOL_VERSION,
                "quant_rng_mode": self.quant_context.rng_mode,
                "profile_protocol": self.profile_protocol,
                "diagnostic_target": v2_summary["diagnostic_target"] if v2_summary["enabled"] else None,
                "candidates": candidate_definitions,
            },
        )
        self.artifacts.write_csv("dataset.csv", dataset_rows)
        self.artifacts.write_csv("candidate.csv", candidate_rows)
        self.artifacts.write_csv("per_image.csv", per_image_rows)
        self.artifacts.write_csv("per_module.csv", shadow_aggregate)
        self.artifacts.write_csv("shadow_quant_repeats.csv", raw_shadow_rows)
        self.artifacts.write_csv("trajectory.csv", trajectory_rows)
        self.artifacts.write_csv("update_direction.csv", v2_result.get("update_direction_rows", []))
        self.artifacts.write_csv("execution_manifest.csv", v2_result.get("execution_manifest_rows", []))
        if v2_result.get("prefix_parity") is not None:
            self.artifacts.write_csv("prefix_parity.csv", v2_result.get("prefix_parity_rows", []))
            self.artifacts.write_json("prefix_parity.json", _finite_json(v2_result["prefix_parity"]))
        if v2_result.get("state_fingerprint_rows") is not None:
            self.artifacts.write_jsonl(
                "state_fingerprints.jsonl",
                _finite_json(v2_result.get("state_fingerprint_rows", [])),
            )
        if v2_result.get("calibration_gate") is not None:
            self.artifacts.write_json("calibration_gate.json", _finite_json(v2_result["calibration_gate"]))
        if v2_result.get("intrinsic_noise_rows") is not None:
            self.artifacts.write_csv("intrinsic_noise.csv", v2_result.get("intrinsic_noise_rows", []))
        if (
            self.profile_protocol.startswith("v24-")
            and v2_result.get("local_natural_gradient_rows") is not None
        ):
            self.artifacts.write_csv(
                "local_natural_gradient.csv",
                v2_result.get("local_natural_gradient_rows", []),
            )
        if v2_result.get("gradient_tail_rows") is not None:
            self.artifacts.write_csv("gradient_tail.csv", v2_result.get("gradient_tail_rows", []))
        if v2_result.get("tail_bootstrap") is not None:
            self.artifacts.write_json("tail_bootstrap.json", _finite_json(v2_result["tail_bootstrap"]))
        if v2_result.get("cumulative_null_rows") is not None:
            self.artifacts.write_csv(
                "cumulative_null_calibration.csv",
                v2_result.get("cumulative_null_rows", []),
            )
        self.artifacts.write_csv("source_stratified_replay.csv", source_stratified_rows)
        self.artifacts.write_csv("range_sweep.csv", v2_result.get("range_sweep_rows", []))
        self.artifacts.write_csv("guardian_ablation.csv", v2_result.get("guardian_ablation_rows", []))
        self.artifacts.write_csv("mechanism_ablation.csv", v2_result.get("mechanism_ablation_rows", []))
        self.artifacts.write_csv("structural.csv", structural_rows)
        self.artifacts.write_csv("geometry_variance.csv", geometry_rows)
        self.artifacts.write_json("geometry_summary.json", _finite_json(geometry_summary))
        self.artifacts.write_npz("structural_sketches.npz", **sketch_arrays)
        svg = trajectory_svg(trajectory_rows)
        (self.artifacts.figures / "loss_trajectory.svg").write_text(svg, encoding="utf-8")
        write_report(
            self.artifacts.root / "report.html",
            summary=summary,
            candidate_rows=candidate_rows,
            trajectory_rows=trajectory_rows,
            shadow_rows=shadow_aggregate,
            structural_rows=structural_rows,
            update_direction_rows=v2_result.get("update_direction_rows", []),
            range_sweep_rows=v2_result.get("range_sweep_rows", []),
            guardian_rows=v2_result.get("guardian_ablation_rows", []),
            mechanism_rows=v2_result.get("mechanism_ablation_rows", []),
            geometry_rows=v2_result.get("geometry_variance_rows", []),
        )
        (self.artifacts.root / "profile.log").write_text(
            json.dumps(
                {
                    "snapshot_step": global_step,
                    "replay_batches": len(sequence),
                    "counterfactual_rows": len(per_image_rows),
                    "shadow_rows": len(raw_shadow_rows),
                    "trajectory_rows": len(trajectory_rows),
                    "update_direction_rows": len(v2_result.get("update_direction_rows", [])),
                    "range_sweep_rows": len(v2_result.get("range_sweep_rows", [])),
                    "source_stratified_rows": len(source_stratified_rows),
                    "guardian_ablation_rows": len(v2_result.get("guardian_ablation_rows", [])),
                    "mechanism_ablation_rows": len(v2_result.get("mechanism_ablation_rows", [])),
                    "geometry_variance_rows": len(geometry_rows),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_hash = canonical_sha256(summary)
        self.artifacts.mark_complete(summary_hash)
        accelerator.end_training()
        return summary
