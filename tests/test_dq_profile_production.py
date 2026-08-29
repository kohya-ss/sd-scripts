from __future__ import annotations

import json
from pathlib import Path

import pytest

import dq_profile.production_runner as production_runner
import dq_profile.__main__ as production_main
from dq_profile.production_cli import ProfileCompatibilityError, resolve_training_cli
from dq_profile.production_runner import (
    DEFAULT_OUTPUT_BASE,
    ProductionRunOptions,
    allocate_run_directory,
    profile_command,
    promote_analysis,
    promote_profile_provenance,
    sanitize_profile_name,
    source_dirs_from_dataset_config,
    validate_output_base,
)


MODEL = r"D:\models\model.safetensors"
DATASET = r"D:\data set\日本語\dataset.toml"


def minimal_cli(*extra: str) -> list[str]:
    return [
        f"--pretrained_model_name_or_path={MODEL}",
        f"--dataset_config={DATASET}",
        *extra,
    ]


def test_minimal_cli_uses_versioned_preset_and_japanese_paths() -> None:
    request = resolve_training_cli(minimal_cli())
    assert request.preset.name == "canonical-v1"
    assert request.dataset_config.name == "dataset.toml"
    assert request.output_name == "dataset"
    assert {row["action"] for row in request.dispositions} == {"consumed"}


def test_full_legacy_style_cli_is_explicit_about_overrides() -> None:
    request = resolve_training_cli(
        minimal_cli(
            "--output_dir=D:\\lora_output",
            "--output_name=sample_r4",
            "--optimizer_type=AdamW8bitFast",
            "--network_module=networks.lora",
            "--network_dim=4",
            "--network_args",
            "rank_dropout=0.2",
            "--mixed_precision=fp16",
            "--fp16_safe_norms_mode=strict",
            "--network_dropout=0.3",
            "--max_data_loader_n_workers=1",
            "--dq_delta_auto_range_mul",
            "--dq_delta_auto_preset=clip_rate_low_auto",
        )
    )
    actions = {(row["destination"], row["action"]) for row in request.dispositions}
    assert ("optimizer_type", "matched_preset") in actions
    assert ("max_data_loader_n_workers", "overridden_with_reason") in actions
    assert ("dq_delta_auto_preset", "overridden_with_reason") in actions
    assert request.output_name == "sample_r4"


def test_canonical_long_training_command_can_be_reused() -> None:
    request = resolve_training_cli(
        minimal_cli(
            "--prior_loss_weight=1.0", "--output_dir=..\\lora_output",
            "--output_name=long_command", "--learning_rate=3.5e-4",
            "--max_train_epochs=40", "--optimizer_type=AdamW8bitFast", "--sdpa",
            "--mixed_precision=fp16", "--save_precision=fp16", "--seed=39",
            "--save_model_as=safetensors", "--save_every_n_epochs=1",
            "--max_data_loader_n_workers=1", "--network_module=networks.lora",
            "--network_dim=4", "--network_args", "rank_dropout=0.2",
            "--enable_bucket", "--min_bucket_reso=384", "--max_bucket_reso=1024",
            "--noise_offset=0.15", "--adaptive_noise_scale=0.1", "--network_dropout=0.3",
            "--cache_latents", "--text_encoder_lr=2e-4", "--downscale_freq_shift",
            "--te_mlp_fc_only", "--grad_norm_mode=stable_no_threshoff", "--avg_cp",
            "--avg_cp_mode=promote", "--avg_window=4", "--avg_begin=0.6",
            "--avg_mode=ema", "--avg_shadow_bank_size=12", "--no-avg_reset_stats",
            "--avg_save_final_raw", "--fp16_safe_norms", "--dq_delta_bits=8",
            "--dq_delta_granularity=channel", "--dq_delta_stat=rms",
            "--dq_delta_range_mul=3.0", "--dq_delta_mode=stoch",
            "--dq_delta_begin_after_lr_warmup", "--dq_delta_scope=unet", "--dq_delta_log",
            "--dq_delta_log_detail=basic", "--dq_delta_auto_range_mul",
            "--dq_delta_auto_preset=clip_rate_low_auto",
            "--dq_delta_auto_init_range_mul_from_band", "--dq_delta_auto_use_raw",
            "--dq_delta_use_triton", "--dq_delta_triton_stats",
            "--text_encoder_lr1=3e-4", "--text_encoder_lr2=2e-4",
            "--lr_scheduler=constant_with_warmup", "--lr_warmup_steps=0.05",
            "--rank_log", "--rank_log_mode=per_module",
        )
    )
    assert request.output_name == "long_command"
    assert any(
        row["destination"] == "dq_delta_range_mul" and row["action"] == "overridden_with_reason"
        for row in request.dispositions
    )


@pytest.mark.parametrize(
    ("option", "required"),
    (("--optimizer_type=AdamW8bit", "AdamW8bitFast"), ("--network_dim=8", "network_dim=4")),
)
def test_conflicting_canonical_option_is_rejected(option: str, required: str) -> None:
    with pytest.raises(ProfileCompatibilityError, match=required):
        resolve_training_cli(minimal_cli(option))


def test_resume_and_unknown_options_are_rejected() -> None:
    with pytest.raises(ProfileCompatibilityError, match="resume is unsupported"):
        resolve_training_cli(minimal_cli("--resume=D:\\checkpoint"))
    with pytest.raises(ProfileCompatibilityError, match="unknown to the SDXL training parser"):
        resolve_training_cli(minimal_cli("--not_a_real_option=1"))


@pytest.mark.parametrize(
    "unknown_tokens",
    (("--api-token=do-not-print",), ("--api-token", "do-not-print")),
)
def test_unknown_sensitive_option_value_is_redacted(unknown_tokens: tuple[str, ...]) -> None:
    with pytest.raises(ProfileCompatibilityError) as captured:
        resolve_training_cli(minimal_cli(*unknown_tokens))
    message = str(captured.value)
    assert "do-not-print" not in message
    assert "<redacted>" in message


def test_fp16_safe_norms_alias_is_accepted() -> None:
    request = resolve_training_cli(minimal_cli("--fp16_safe_norms"))
    assert any(row["destination"] == "fp16_safe_norms" for row in request.dispositions)


def test_source_dirs_follow_toml_order_and_reject_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "画像 A"
    second = tmp_path / "画像 B"
    first.mkdir()
    second.mkdir()
    config = tmp_path / "dataset.toml"
    config.write_text(
        "[[datasets]]\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(first))}\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(second))}\n",
        encoding="utf-8",
    )
    assert source_dirs_from_dataset_config(config) == (first.resolve(), second.resolve())
    config.write_text(
        "[[datasets]]\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(first))}\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {json.dumps(str(first))}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate image_dir"):
        source_dirs_from_dataset_config(config)


def test_output_base_policy_and_unique_run_ids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(production_runner, "REPO_ROOT", repo)
    source = tmp_path / "dataset"
    source.mkdir()
    base = tmp_path / "reports"
    assert validate_output_base(base, source_dirs=(source,), normal_output_dir=None) == base.resolve()
    with pytest.raises(ValueError, match="overlap dataset"):
        validate_output_base(source / "report", source_dirs=(source,), normal_output_dir=None)
    first = allocate_run_directory(base, "日本語 profile", "a" * 64)
    second = allocate_run_directory(base, "日本語 profile", "a" * 64)
    assert first != second
    assert first.parent.name == "日本語_profile"


def test_default_output_location_is_project_lora_output() -> None:
    assert DEFAULT_OUTPUT_BASE.parts[-2:] == ("lora_output", "dq_dataset_profiler")


def test_profile_command_uses_request_paths_and_python_module(tmp_path: Path) -> None:
    request = resolve_training_cli(minimal_cli("--output_name=portable"))
    command = profile_command(
        request,
        run_dir=tmp_path,
        source_map=tmp_path / "source.json",
        name="01_core",
        protocol="v24-acceptance-local",
        range_muls=(2.70, 3.15, 3.45),
        max_images=16,
    )
    assert command[1:4] == ["-m", "accelerate.commands.launch", "--num_cpu_threads_per_process"]
    assert f"--pretrained_model_name_or_path={request.model_path}" in command
    assert f"--dataset_config={request.dataset_config}" in command
    assert "--dq_profile_protocol=v24-acceptance-local" in command
    assert not any("noobaiXLNAIXL" in item for item in command)


def test_product_artifacts_are_promoted_to_run_root(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    analysis = tmp_path / "analysis"
    profile = tmp_path / "profile"
    run_dir.mkdir()
    analysis.mkdir()
    profile.mkdir()
    for name in ("report.html", "technical_report.html", "summary.json", "practical_report.json"):
        (analysis / name).write_text(name, encoding="utf-8")
    (profile / "source_manifest.json").write_text("{}", encoding="utf-8")
    (profile / "candidate_definitions.json").write_text("{}", encoding="utf-8")
    promoted = promote_analysis(run_dir, analysis)
    promoted.extend(promote_profile_provenance(run_dir, profile))
    assert (run_dir / "report.html").is_file()
    assert (run_dir / "source_manifest.json").is_file()
    assert "candidate_definitions.json" in promoted


def test_direct_module_entry_preserves_training_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run_profile_mode(training_argv: list[str], **kwargs: object) -> int:
        captured["training_argv"] = training_argv
        captured["kwargs"] = kwargs
        return 17

    monkeypatch.setattr(production_main, "run_profile_mode", fake_run_profile_mode)
    training = [
        "--dataset_config=D:\\space path\\日本語.toml",
        "--pretrained_model_name_or_path=D:\\models\\model.safetensors",
        "--network_args",
        "rank_dropout=0.2",
    ]
    result = production_main.main(["--dq-profile-name=test", *training])
    assert result == 17
    assert captured["training_argv"] == training
    assert captured["kwargs"] == {
        "preset_name": "canonical-v1",
        "output_base": DEFAULT_OUTPUT_BASE,
        "profile_name": "test",
        "preflight_only": False,
        "dry_run": False,
        "open_report": False,
    }


def test_profile_name_sanitization_is_windows_safe() -> None:
    assert sanitize_profile_name("  A/B: test  ") == "A_B__test"
    assert sanitize_profile_name("CON") == "dq_CON"
