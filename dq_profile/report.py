from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import toml

from dq_profile import (
    RUNTIME_METRIC_DEFINITION_VERSION as METRIC_DEFINITION_VERSION,
    RUNTIME_SCHEMA_VERSION as SCHEMA_VERSION,
)


KNOWN_RESULT_DEFAULTS: dict[str, Any] = {
    "best": "",
    "ranking": [],
    "confidence": "",
    "comparison_controlled": False,
    "utility_screen_seed39": "not_measured",
    "U_selected_protocol": "unknown",
    "U_any_quantization": "unknown",
    "utility_confidence": "low",
    "quality_margin": "",
    "m_utility": "",
    "past_network_dim": 0,
    "past_optimizer": "",
    "past_mixed_precision": "",
    "past_save_precision": "",
    "past_fp16_safe_norms_mode": "",
    "past_training_steps": 0,
    "past_dataset_sha256": "",
    "past_dq_preset": "",
    "past_dq_bits": 0,
    "past_dq_granularity": "",
    "past_dq_stat": "",
    "past_dq_mode": "",
    "past_dq_scope": "",
    "control_differences": [],
    "notes": "",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _format(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "+∞" if value > 0 else "−∞"
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_format(item) for item in value)
    return str(value)


class ProfileArtifacts:
    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.root = Path(output_dir).resolve()
        self.figures = self.root / "figures"

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.write_json("status.json", {"status": "running", "schema_version": SCHEMA_VERSION})

    def write_json(self, name: str, payload: Any) -> None:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n"
        _atomic_text(self.root / name, text)

    def write_csv(self, name: str, rows: Iterable[Mapping[str, Any]]) -> None:
        rows = list(rows)
        keys: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
                if keys:
                    writer.writeheader()
                    writer.writerows(rows)
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def write_jsonl(self, name: str, rows: Iterable[Mapping[str, Any]]) -> None:
        text = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default, allow_nan=False) + "\n"
            for row in rows
        )
        _atomic_text(self.root / name, text)

    def copy_dataset_config(self, source: str | os.PathLike[str]) -> None:
        shutil.copy2(Path(source).resolve(), self.root / "dataset_config.toml")

    def ensure_known_result(self) -> None:
        path = self.root / "known_result.toml"
        if path.exists():
            return
        _atomic_text(path, toml.dumps(KNOWN_RESULT_DEFAULTS))

    def read_known_result(self) -> dict[str, Any]:
        self.ensure_known_result()
        loaded = toml.load(self.root / "known_result.toml")
        result = dict(KNOWN_RESULT_DEFAULTS)
        if isinstance(loaded, Mapping):
            result.update(loaded)
        return result

    def write_npz(self, name: str, **arrays: Any) -> None:
        target = self.root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        os.close(fd)
        try:
            np.savez_compressed(temporary, **arrays)
            generated = Path(temporary + ".npz")
            os.replace(generated, target)
        finally:
            for candidate in (Path(temporary), Path(temporary + ".npz")):
                if candidate.exists():
                    candidate.unlink()

    def mark_failed(self, error: BaseException) -> None:
        self.write_json(
            "status.json",
            {
                "status": "failed",
                "schema_version": SCHEMA_VERSION,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            },
        )

    def mark_complete(self, summary_sha256: str) -> None:
        self.write_json(
            "status.json",
            {"status": "complete", "schema_version": SCHEMA_VERSION, "summary_sha256": summary_sha256},
        )


def trajectory_svg(rows: Iterable[Mapping[str, Any]], width: int = 900, height: int = 320) -> str:
    rows = [row for row in rows if row.get("loss") is not None]
    if not rows:
        return "<svg viewBox='0 0 900 120' role='img'><text x='20' y='60'>No trajectory data</text></svg>"
    candidates = sorted({str(row["candidate"]) for row in rows})
    palette = {"no_quant": "#64748b", "clip_rate_high": "#dc2626", "clip_rate_low": "#2563eb"}
    steps = [int(row["branch_step"]) for row in rows]
    losses = [float(row["loss"]) for row in rows if math.isfinite(float(row["loss"]))]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(losses), max(losses)
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    margin = 44

    def point(step: int, loss: float) -> tuple[float, float]:
        x = margin + (step - x_min) / max(x_max - x_min, 1) * (width - margin * 2)
        y = height - margin - (loss - y_min) / max(y_max - y_min, 1e-12) * (height - margin * 2)
        return x, y

    lines = [f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Branch loss trajectories'>"]
    lines.append(f"<rect width='{width}' height='{height}' fill='#fff'/>")
    lines.append(f"<line x1='{margin}' y1='{height-margin}' x2='{width-margin}' y2='{height-margin}' stroke='#94a3b8'/>")
    lines.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height-margin}' stroke='#94a3b8'/>")
    for candidate in candidates:
        candidate_rows = sorted((row for row in rows if str(row["candidate"]) == candidate), key=lambda row: int(row["branch_step"]))
        points = " ".join(f"{x:.2f},{y:.2f}" for x, y in (point(int(row["branch_step"]), float(row["loss"])) for row in candidate_rows))
        color = palette.get(candidate, "#7c3aed")
        lines.append(f"<polyline fill='none' stroke='{color}' stroke-width='2' points='{points}'/>")
    legend_x = margin
    for candidate in candidates:
        color = palette.get(candidate, "#7c3aed")
        lines.append(f"<rect x='{legend_x}' y='10' width='12' height='12' fill='{color}'/><text x='{legend_x+17}' y='21' font-size='12'>{html.escape(candidate)}</text>")
        legend_x += 150
    lines.append("</svg>")
    return "".join(lines)


def _table(rows: Iterable[Mapping[str, Any]], columns: list[tuple[str, str]]) -> str:
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_format(row.get(key)))}</td>" for key, _ in columns)
        body.append(f"<tr>{cells}</tr>")
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def build_report(
    *,
    summary: Mapping[str, Any],
    candidate_rows: Iterable[Mapping[str, Any]],
    trajectory_rows: Iterable[Mapping[str, Any]],
    shadow_rows: Iterable[Mapping[str, Any]],
    structural_rows: Iterable[Mapping[str, Any]],
    update_direction_rows: Iterable[Mapping[str, Any]] = (),
    range_sweep_rows: Iterable[Mapping[str, Any]] = (),
    guardian_rows: Iterable[Mapping[str, Any]] = (),
    mechanism_rows: Iterable[Mapping[str, Any]] = (),
    geometry_rows: Iterable[Mapping[str, Any]] = (),
) -> str:
    candidate_rows = list(candidate_rows)
    trajectory_rows = list(trajectory_rows)
    shadow_rows = list(shadow_rows)
    structural_rows = list(structural_rows)
    update_direction_rows = list(update_direction_rows)
    range_sweep_rows = list(range_sweep_rows)
    guardian_rows = list(guardian_rows)
    mechanism_rows = list(mechanism_rows)
    geometry_rows = list(geometry_rows)
    warnings = summary.get("warnings", [])
    known_result = summary.get("known_result") or KNOWN_RESULT_DEFAULTS
    warning_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in warnings) or "<li>None</li>"
    candidate_table = _table(
        candidate_rows,
        [
            ("candidate", "Candidate"),
            ("initial_range_mul", "Initial range_mul"),
            ("final_range_mul", "Final range_mul"),
            ("branch_regime", "Branch regime"),
            ("branch_loss_mean", "Branch loss mean"),
            ("branch_parameter_update_norm", "Branch update norm"),
            ("checkpoint64_update_cosine_median", "Update cosine @64"),
            ("checkpoint64_projection_gain_median", "Projection gain @64"),
            ("checkpoint64_orthogonal_drift_median", "Orthogonal drift @64"),
            ("checkpoint64_total_drift_median", "Total drift @64"),
            ("forced_safety_abort", "Safety abort"),
            ("probe_regime", "Probe regime"),
            ("probe_parameter_gradient_cosine_mean", "Probe gradient cosine"),
            ("auto_trajectory_metrics_valid", "Auto valid"),
            ("auto_invalid_reason", "Invalid reason"),
        ],
    )
    known_result_table = _table(
        [known_result],
        [
            ("best", "Known best"),
            ("ranking", "Known ranking"),
            ("confidence", "Confidence"),
            ("comparison_controlled", "Controlled comparison"),
            ("comparison_controlled_effective", "Controlled (effective)"),
            ("past_network_dim", "Past dim"),
            ("past_optimizer", "Past optimizer"),
            ("past_mixed_precision", "Past mixed precision"),
            ("past_fp16_safe_norms_mode", "Past safe norms"),
            ("past_dq_preset", "Past DQ preset"),
            ("past_dq_bits", "Past DQ bits"),
            ("past_dq_granularity", "Past DQ granularity"),
            ("past_dq_stat", "Past DQ stat"),
            ("utility_screen_seed39", "Utility screen seed39"),
            ("U_selected_protocol", "Selected-protocol utility"),
            ("U_any_quantization", "Any-quantization utility"),
            ("utility_confidence", "Utility confidence"),
            ("quality_margin", "Quality margin"),
            ("m_utility", "Utility range"),
            ("past_dq_mode", "Past DQ mode"),
            ("past_dq_scope", "Past DQ scope"),
            ("control_differences", "Control differences"),
            ("detected_control_differences", "Detected differences"),
            ("notes", "Notes"),
        ],
    )
    shadow_table = _table(
        shadow_rows[:100],
        [
            ("candidate", "Candidate"),
            ("module_name", "Module"),
            ("error_rms_mean", "Error RMS"),
            ("clip_error_rms_mean", "Clip RMS"),
            ("round_error_rms_mean", "Round RMS"),
            ("fisher_error_mean_mean", "Fisher error"),
            ("fisher_clip_error_mean_mean", "Fisher clip"),
            ("fisher_round_error_mean_mean", "Fisher round"),
            ("signed_impact_mean_mean", "Signed impact"),
            ("signed_clip_impact_mean_mean", "Signed clip"),
            ("signed_round_impact_mean_mean", "Signed round"),
        ],
    )
    structural_table = _table(
        structural_rows,
        [
            ("probe_regime", "Regime"),
            ("timestep_bin", "Timestep bin"),
            ("image_count", "Images"),
            ("effective_rank", "Effective rank"),
            ("stable_rank", "Stable rank"),
            ("gradient_noise_scale", "GNS"),
        ],
    )
    v2 = summary.get("v2") or {}
    intrinsic = v2.get("intrinsic_stability_result") or {}
    guardian_adjusted = v2.get("guardian_adjusted_result") or {}
    stability_table = _table(
        [
            {
                "result": "Intrinsic / common skip",
                **intrinsic,
            },
            {
                "result": "Guardian adjusted / native",
                **guardian_adjusted,
            },
        ],
        [
            ("result", "Result"),
            ("m_dir", "m_dir"),
            ("m_total", "m_total"),
            ("m_stability_diag", "m_stability_diag"),
            ("W_dir_grid", "W_dir_grid"),
            ("W_total_grid", "W_total_grid"),
            ("W_stability_grid", "W_stability_grid"),
            ("stability_confidence", "Confidence"),
            ("all_candidates_poor", "All candidates poor"),
            ("third_repeat_required", "Third repeat indicated"),
            ("third_repeat_reasons", "Reasons"),
        ],
    )
    range_table = _table(
        [
            row
            for row in range_sweep_rows
            if int(row.get("checkpoint", -1)) in {64, 128}
            and str(row.get("module_group", "all")) == "all"
        ],
        [
            ("candidate", "Candidate"),
            ("range_mul", "range_mul"),
            ("repeat", "Repeat"),
            ("checkpoint", "Checkpoint"),
            ("update_cosine", "Update cosine"),
            ("projection_gain", "Projection gain"),
            ("orthogonal_drift", "Orthogonal drift"),
            ("total_drift", "Total drift"),
            ("update_norm_ratio", "Update norm ratio"),
            ("native_would_skip_to_checkpoint", "Native would-skip"),
            ("forced_safety_abort", "Safety abort"),
            ("invalid_reason", "Invalid reason"),
        ],
    )
    guardian_table = _table(
        [row for row in guardian_rows if int(row.get("checkpoint", -1)) == 64],
        [
            ("candidate", "Candidate"),
            ("range_mul", "range_mul"),
            ("repeat", "Repeat"),
            ("update_cosine", "Update cosine"),
            ("orthogonal_drift", "Orthogonal drift"),
            ("total_drift", "Total drift"),
            ("native_would_skip_to_checkpoint", "Native skips"),
        ],
    )
    mechanism_table = _table(
        [row for row in mechanism_rows if int(row.get("checkpoint", -1)) == 64 and row.get("module_group") == "all"],
        [
            ("candidate", "Mechanism"),
            ("range_mul", "range_mul"),
            ("repeat", "Repeat"),
            ("update_cosine", "Update cosine"),
            ("orthogonal_drift", "Orthogonal drift"),
            ("total_drift", "Total drift"),
            ("interaction_update_cosine", "Interaction cosine"),
            ("interaction_orthogonal_drift", "Interaction orthogonal"),
            ("interaction_total_drift", "Interaction total"),
        ],
    )
    geometry_table = _table(
        geometry_rows,
        [("component", "Geometry component"), ("energy", "Energy"), ("fraction", "Fraction"), ("valid", "Valid")],
    )
    calibration_gate = v2.get("calibration_gate") or {}
    prefix_parity = v2.get("prefix_parity") or {}
    tail_bootstrap = v2.get("tail_bootstrap") or {}
    intrinsic_noise = v2.get("intrinsic_noise") or {}
    fragility = v2.get("fragility_diag") or {}
    prefix_table = _table(
        prefix_parity.get("pair_results", []),
        [
            ("comparison", "Comparison"),
            ("candidate", "Candidate"),
            ("status", "Status"),
            ("first_divergence", "First divergence"),
            ("step_count_reference", "Reference steps"),
            ("step_count_candidate", "Candidate steps"),
        ],
    )
    tail_table = _table(
        [
            {
                "decision": tail_bootstrap.get("decision"),
                "upper_support": tail_bootstrap.get("upper_support_probability"),
                "lower_support": tail_bootstrap.get("lower_support_probability"),
                "upper_strata_wins": tail_bootstrap.get("upper_strata_wins"),
                "loss_cv_all": (intrinsic_noise.get("all_timestep") or {}).get("observed"),
                "loss_cv_high": (intrinsic_noise.get("max_timestep_bin") or {}).get("observed"),
                "winner_instability": fragility.get("winner_instability"),
                "timestep_variation": fragility.get("timestep_variation_cv_of_min_q95"),
                "minimum_tail": fragility.get("minimum_max_timestep_q95"),
            }
        ],
        [
            ("decision", "Tail decision"),
            ("upper_support", "Upper support"),
            ("lower_support", "Lower support"),
            ("upper_strata_wins", "Upper strata wins"),
            ("loss_cv_all", "Loss CV all"),
            ("loss_cv_high", "Loss CV high"),
            ("winner_instability", "Winner instability"),
            ("timestep_variation", "Timestep variation"),
            ("minimum_tail", "Minimum high-bin q95"),
        ],
    )
    svg = trajectory_svg(trajectory_rows)
    dataset_summary = summary.get("dataset") or {}
    snapshot_summary = summary.get("snapshot") or {}
    profile_summary = summary.get("profile") or {}
    return f"""<!doctype html>
<html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>SDXL DQ Dataset Profile</title>
<style>
:root{{--ink:#172033;--muted:#64748b;--line:#dbe3ee;--panel:#f8fafc;--accent:#2563eb}}
body{{font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;margin:0;color:var(--ink);background:#fff}}
main{{max-width:1200px;margin:auto;padding:28px}}h1{{margin:0 0 8px}}h2{{margin-top:34px;border-bottom:1px solid var(--line);padding-bottom:8px}}
.meta{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:18px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}}.value{{font-size:1.35rem;font-weight:650;margin-top:5px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:var(--panel);position:sticky;top:0}}
svg{{width:100%;height:auto;border:1px solid var(--line);border-radius:8px}}code{{background:#eef2ff;padding:2px 5px;border-radius:4px}}
</style></head><body><main>
<h1>SDXL DQ Dataset Profile</h1>
<p class='meta'>Schema {html.escape(str(summary.get('schema_version')))} · metrics {html.escape(str(summary.get('metric_definition_version')))} · source <code>{html.escape(str(summary.get('source_manifest_sha256',''))[:16])}</code></p>
<div class='cards'>
<div class='card'>Dataset images<div class='value'>{html.escape(_format(dataset_summary.get('unique_images')))}</div></div>
<div class='card'>Snapshot step<div class='value'>{html.escape(_format(snapshot_summary.get('global_step')))}</div></div>
<div class='card'>Probe regime<div class='value'>{html.escape(_format(summary.get('probe_regime')))}</div></div>
<div class='card'>Branch regime<div class='value'>{html.escape(_format(summary.get('branch_regime')))}</div></div>
<div class='card'>Protocol<div class='value'>{html.escape(_format(profile_summary.get('protocol')))}</div></div>
<div class='card'>Stability range<div class='value'>{html.escape(_format(intrinsic.get('m_stability_diag')))}</div></div>
<div class='card'>Guardian dependent<div class='value'>{html.escape(_format(v2.get('guardian_dependent')))}</div></div>
<div class='card'>Calibration gate<div class='value'>{html.escape(_format(calibration_gate.get('gate') or calibration_gate.get('decision')))}</div></div>
<div class='card'>Tail decision<div class='value'>{html.escape(_format(tail_bootstrap.get('decision')))}</div></div>
</div>
<h2>Warnings and validity</h2><ul>{warning_html}</ul>
<h2>v2.1 staged calibration</h2>
<p class='meta'>Prefix smoke checks whether independent 64-step runs and the first 64 steps of a 128-step run reproduce. Tail calibration is shown only for its pre-registered high-timestep q95 rule; descriptive tail metrics do not receive extra votes.</p>
<h3>Prefix parity</h3>{prefix_table}
<h3>D3 tail calibration</h3>{tail_table}
<p class='meta'>A supported result is evidence on the development dataset only. An abstain result deliberately avoids manufacturing a range winner.</p>
<h2>v2 trajectory-stability diagnosis</h2>
<p class='meta'><code>m_stability_diag</code> is the fixed range that stayed closest to the no-quant update trajectory in this short diagnostic. It is not a claim about final image quality, and <code>W_stability_grid</code> is only a plateau among the measured grid points.</p>
{stability_table}
<h3>Fixed range sweep (common skip)</h3>{range_table}
<h3>Guardian confirmation</h3>
<p class='meta'>The sweep uses the no-quant update mask. Native Guardian confirmation then shows whether candidate-specific skip decisions change the result.</p>
{guardian_table}
<h3>Clip / round mechanism counterfactual</h3>
<p class='meta'>These branches explain the measured perturbation; clip-only and round-only are not production recommendations.</p>
{mechanism_table}
<h3>Hierarchical dataset geometry</h3>
<p class='meta'>Geometry modifies confidence and the measured plateau; it does not vote for a quantization candidate.</p>
{geometry_table}
<h2>Candidate comparison</h2>{candidate_table}
<h2>Known result (reference only)</h2>{known_result_table}
<p class='meta'>Known results are displayed for later cross-run analysis and are not used by this profiler to recommend or rank candidates.</p>
<h2>Training-dropout branch trajectories</h2>{svg}
<h2>Shadow quantization (no-quant reference gradient)</h2>{shadow_table}
<h2>Structural image-gradient probes</h2>{structural_table}
<p class='meta'>CountSketch is used only for image-gradient Gram/rank statistics. Candidate parameter-gradient cosine is accumulated exactly.</p>
</main></body></html>"""


def write_report(path: str | os.PathLike[str], **kwargs: Any) -> None:
    _atomic_text(Path(path), build_report(**kwargs))
