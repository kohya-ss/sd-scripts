from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


PRESET_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DiagnosticPreset:
    name: str
    metric_definition_version: str
    description: str
    expected_explicit: Mapping[str, Any]
    ignored_explicit: Mapping[str, str]
    unsupported_explicit: Mapping[str, str]
    training_tokens: tuple[str, ...]
    core_grid: tuple[float, ...]
    max_images: int
    timestep_bins: int
    stochastic_repeats: int
    sweep_steps: int
    branch_repeats: int
    sketch_width: int
    sketch_seeds: int
    max_edge_extension_rounds: int
    num_cpu_threads_per_process: int

    def contract(self) -> dict[str, Any]:
        return {
            "schema_version": PRESET_SCHEMA_VERSION,
            "name": self.name,
            "metric_definition_version": self.metric_definition_version,
            "description": self.description,
            "expected_explicit": dict(self.expected_explicit),
            "ignored_explicit": dict(self.ignored_explicit),
            "unsupported_explicit": dict(self.unsupported_explicit),
            "training_tokens": list(self.training_tokens),
            "core_grid": list(self.core_grid),
            "max_images": self.max_images,
            "timestep_bins": self.timestep_bins,
            "stochastic_repeats": self.stochastic_repeats,
            "sweep_steps": self.sweep_steps,
            "branch_repeats": self.branch_repeats,
            "sketch_width": self.sketch_width,
            "sketch_seeds": self.sketch_seeds,
            "max_edge_extension_rounds": self.max_edge_extension_rounds,
            "num_cpu_threads_per_process": self.num_cpu_threads_per_process,
            "diagnostic_target": "numerical_gradient_acceptance_by_fixed_range_mul",
            "not_quality_or_utility": True,
            "product_scope": "local_body_tail_only",
        }


CANONICAL_V1 = DiagnosticPreset(
    name="canonical-v1",
    metric_definition_version="2.4.0",
    description=(
        "Validated SDXL rank-4 Local Body/Tail preset. This is a numerical "
        "Safety/Fidelity diagnostic, not a quality or Utility predictor."
    ),
    expected_explicit={
        "prior_loss_weight": 1.0,
        "learning_rate": 3.5e-4,
        "max_train_epochs": 40,
        "optimizer_type": "AdamW8bitFast",
        "sdpa": True,
        "mixed_precision": "fp16",
        "seed": 39,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "network_module": "networks.lora",
        "network_dim": 4,
        "enable_bucket": True,
        "min_bucket_reso": 384,
        "max_bucket_reso": 1024,
        "noise_offset": 0.15,
        "adaptive_noise_scale": 0.1,
        "network_dropout": 0.3,
        "cache_latents": True,
        "text_encoder_lr": 2e-4,
        "text_encoder_lr1": 3e-4,
        "text_encoder_lr2": 2e-4,
        "downscale_freq_shift": True,
        "te_mlp_fc_only": True,
        "grad_norm_mode": "stable_no_threshoff",
        "avg_cp": True,
        "avg_cp_mode": "promote",
        "avg_window": 4,
        "avg_begin": 0.6,
        "avg_mode": "ema",
        "avg_shadow_bank_size": 12,
        "avg_reset_stats": False,
        "avg_save_final_raw": True,
        "dq_delta_bits": 8,
        "dq_delta_granularity": "channel",
        "dq_delta_stat": "rms",
        "dq_delta_mode": "stoch",
        "dq_delta_begin_after_lr_warmup": True,
        "dq_delta_scope": "unet",
        "dq_delta_log": True,
        "dq_delta_log_detail": "basic",
        "dq_delta_use_triton": True,
        "dq_delta_triton_stats": True,
        "lr_scheduler": "constant_with_warmup",
        "lr_warmup_steps": 0.05,
        "rank_log": True,
        "rank_log_mode": "per_module",
    },
    ignored_explicit={
        "output_dir": "normal checkpoint output is not used; diagnostics write only to the run directory",
        "save_precision": "the profiler does not save a normal training checkpoint",
        "save_model_as": "the profiler does not save a normal training checkpoint",
        "save_every_n_epochs": "the profiler does not save epoch checkpoints",
        "save_every_n_steps": "the profiler does not save step checkpoints",
        "training_comment": "replaced by a versioned diagnostic provenance comment",
        "max_data_loader_n_workers": "forced to 0 for deterministic replay materialization",
        "dq_delta_range_mul": "replaced by the fixed diagnostic mul grid",
        "dq_delta_auto_range_mul": "auto range is disabled during the fixed diagnostic scan",
        "dq_delta_auto_preset": "auto range is disabled during the fixed diagnostic scan",
        "dq_delta_auto_init_range_mul_from_band": "auto range is disabled during the fixed diagnostic scan",
        "dq_delta_auto_use_raw": "auto range is disabled during the fixed diagnostic scan",
        "dq_delta_log_every": "diagnostic logging cadence is owned by the protocol",
        "dq_delta_log_scope": "diagnostic logging scope is owned by the protocol",
        "dq_delta_log_mode": "diagnostic logging mode is owned by the protocol",
        "dq_delta_log_error_parts": "the Local Body/Tail protocol records its own error decomposition",
    },
    unsupported_explicit={
        "resume": "resume is unsupported because every diagnostic run starts from a fresh common snapshot",
        "resume_from_huggingface": "resume is unsupported in diagnostic mode",
        "network_weights": "loading existing network weights would invalidate the fresh-snapshot comparison",
        "max_train_steps": "canonical-v1 derives its warmup boundary from max_train_epochs=40",
        "config_file": "config-file expansion is not supported yet; pass the relevant training options explicitly",
        "full_fp16": "canonical-v1 requires mixed_precision=fp16 without full-fp16 gradients",
        "fp8_base": "canonical-v1 has not been validated with fp8 base weights",
        "dq_delta_bits_sched": "bit schedules are incompatible with the fixed 8-bit diagnostic contract",
        "dq_delta_step": "step-based quantization is outside canonical-v1",
        "dq_quantize_z": "z quantization is outside canonical-v1",
        "optimizer_args": "custom optimizer arguments are outside canonical-v1",
        "network_alpha": "custom network alpha is outside canonical-v1",
    },
    training_tokens=(
        "--prior_loss_weight=1.0",
        "--learning_rate=3.5e-4",
        "--max_train_epochs=40",
        "--optimizer_type=AdamW8bitFast",
        "--sdpa",
        "--mixed_precision=fp16",
        "--save_precision=fp16",
        "--seed=39",
        "--save_model_as=safetensors",
        "--max_data_loader_n_workers=0",
        "--gradient_accumulation_steps=1",
        "--network_module=networks.lora",
        "--network_dim=4",
        "--network_args",
        "rank_dropout=0.2",
        "--enable_bucket",
        "--min_bucket_reso=384",
        "--max_bucket_reso=1024",
        "--noise_offset=0.15",
        "--adaptive_noise_scale=0.1",
        "--network_dropout=0.3",
        "--cache_latents",
        "--text_encoder_lr=2e-4",
        "--text_encoder_lr1=3e-4",
        "--text_encoder_lr2=2e-4",
        "--downscale_freq_shift",
        "--te_mlp_fc_only",
        "--grad_norm_mode=stable_no_threshoff",
        "--avg_cp",
        "--avg_cp_mode=promote",
        "--avg_window=4",
        "--avg_begin=0.6",
        "--avg_mode=ema",
        "--avg_shadow_bank_size=12",
        "--no-avg_reset_stats",
        "--avg_save_final_raw",
        "--fp16_safe_norms_mode=strict",
        "--dq_delta_bits=8",
        "--dq_delta_granularity=channel",
        "--dq_delta_stat=rms",
        "--dq_delta_range_mul=3.0",
        "--dq_delta_mode=stoch",
        "--dq_delta_begin_after_lr_warmup",
        "--dq_delta_scope=unet",
        "--dq_delta_log",
        "--dq_delta_log_detail=basic",
        "--dq_delta_use_triton",
        "--dq_delta_triton_stats",
        "--lr_scheduler=constant_with_warmup",
        "--lr_warmup_steps=0.05",
        "--rank_log",
        "--rank_log_mode=per_module",
    ),
    core_grid=(2.70, 3.15, 3.45),
    max_images=32,
    timestep_bins=4,
    stochastic_repeats=2,
    sweep_steps=128,
    branch_repeats=5,
    sketch_width=512,
    sketch_seeds=2,
    max_edge_extension_rounds=2,
    num_cpu_threads_per_process=8,
)


PRESETS = {CANONICAL_V1.name: CANONICAL_V1}


def get_preset(name: str) -> DiagnosticPreset:
    try:
        return PRESETS[name]
    except KeyError as error:
        supported = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown DQ profile preset {name!r}; supported: {supported}") from error
