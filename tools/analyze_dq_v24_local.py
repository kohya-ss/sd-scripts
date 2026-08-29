from __future__ import annotations

import argparse
import csv
import html
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dq_profile.v23_safety import canonical_json_sha256
from dq_profile.v24_acceptance import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    METRIC_DEFINITION_VERSION,
    SCHEMA_VERSION,
    SELECTION_SCHEMA_VERSION,
    acceptance_contract,
    analyze_local_profile,
    analyze_natural_gradient_rows,
)
from dq_profile.v24_practical_report import (
    build_single_dataset_report_model,
    render_report as render_practical_report,
    report_contract as practical_report_contract,
)
from tools.analyze_dq_v231_safety import sha256_file, write_csv, write_json


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{number:.{digits}f}" if math.isfinite(number) else "—"


def _interval(low: Any, high: Any) -> str:
    return f"[{_number(low)}, {_number(high)}]"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def render_report(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    natural: dict[str, Any],
) -> str:
    selection = summary["selection"]
    table_rows: list[str] = []
    for row in rows:
        roles = ", ".join(row["mandatory_retention_role"]) or "—"
        table_rows.append(
            "<tr>"
            f"<td>{float(row['range_mul']):.2f}</td>"
            f"<td>{html.escape(str(row['grid_role']))}</td>"
            f"<td>{_number(row['local_body'])}</td>"
            f"<td>{_interval(row['local_body_ci_low'], row['local_body_ci_high'])}</td>"
            f"<td>{_number(row['local_tail'])}</td>"
            f"<td>{_interval(row['local_tail_ci_low'], row['local_tail_ci_high'])}</td>"
            f"<td>{_number(row['tail_amplification'])}</td>"
            f"<td>{_number(row['symmetric_body'])} / {_number(row['symmetric_tail'])}</td>"
            f"<td>{_number(row['angle_body'])} / {_number(row['angle_tail'])}</td>"
            f"<td>{_number(row['gain_body'])} / {_number(row['gain_tail'])}</td>"
            f"<td>{100*float(row['source_bootstrap_body_min_probability']):.1f}%</td>"
            f"<td>{100*float(row['source_bootstrap_tail_min_probability']):.1f}%</td>"
            f"<td>{'yes' if row['robustly_dominated'] else 'no'}</td>"
            f"<td>{'retain' if row['retained_for_formal'] else 'drop'}</td>"
            f"<td>{html.escape(roles)}</td>"
            "</tr>"
        )
    selected = ", ".join(f"{value:.2f}" for value in selection["selected_muls"]) or "なし"
    credible = ", ".join(f"{value:.2f}" for value in selection["credible_muls"]) or "なし"
    edge = ", ".join(
        f"{value:.2f}" for value in selection["edge_extension_recommended"]
    ) or "不要"
    natural_text = (
        f"Body {_number(natural.get('local_body'))} {_interval(natural.get('local_body_ci_low'), natural.get('local_body_ci_high'))}; "
        f"Tail {_number(natural.get('local_tail'))} {_interval(natural.get('local_tail_ci_low'), natural.get('local_tail_ci_high'))}; "
        f"A {_number(natural.get('tail_amplification'))}"
        if natural.get("valid")
        else f"invalid: {html.escape(str(natural.get('invalid_reason')))}"
    )
    envelope = summary["core_grid_envelope"]
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>DQ v2.4 Acceptance — {html.escape(summary['dataset_id'])}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:28px;color:#20272b;line-height:1.55}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px}}
.card{{border:1px solid #ccd5d9;border-radius:9px;padding:14px 18px;margin:12px 0}}
.warn{{background:#fff5e6;border-color:#e6b566}}.info{{background:#eef5fb;border-color:#8db7d6}}
table{{border-collapse:collapse;width:100%;font-size:10px}}th,td{{border:1px solid #ccd5d9;padding:5px;text-align:right}}
th:nth-child(2),td:nth-child(2),th:nth-child(n+13),td:nth-child(n+13){{text-align:left}}
code{{background:#f3f5f6;padding:2px 4px}}
</style></head><body>
<h1>DQ Profiler v2.4 — Local numerical acceptance</h1>
<div class="grid">
<div class="card"><b>Dataset</b><br>{html.escape(summary['dataset_id'])}<br>{summary['image_count']} images / {summary['source_group_count']} source groups</div>
<div class="card"><b>Phenotype (local only)</b><br>{html.escape(summary['local_phenotype'])}</div>
<div class="card"><b>Credible set</b><br>{credible}<br><b>Formal selection</b>: {selected}</div>
<div class="card"><b>Edge</b><br>{html.escape(selection['selection_status'])}<br>next local-only: {edge}</div>
</div>
<div class="card warn"><b>これは画質点・成功確率・Utilityではありません。</b><br>
Bodyは通常範囲、Tailは最悪timestep帯、AはTail増幅です。候補はsource-cluster bootstrapでBody/Tailの両方が80%以上の確率でPareto劣位な場合だけ除外します。
Trajectoryは128-step formalで別軸として測り、ここでは未測定です。</div>
<div class="card info"><b>No-quant natural local baseline</b><br>{natural_text}<br>候補選択票には使いません。</div>
<div class="card"><b>Common core envelope</b><br>grid {html.escape(str(envelope['grid']))}; max Body {_number(envelope['max_local_body'])}; max Tail {_number(envelope['max_local_tail'])}<br>
P(all core Body&lt;1) {100*float(envelope['probability_all_core_body_below_anchor']):.1f}% / P(all core Tail&lt;1) {100*float(envelope['probability_all_core_tail_below_anchor']):.1f}%</div>
<h2>候補別カルテ</h2>
<table><thead><tr><th>mul</th><th>role</th><th>Body</th><th>Body CI</th><th>Tail</th><th>Tail CI</th><th>A</th><th>sym B/T</th><th>angle B/T</th><th>gain B/T</th><th>P body min</th><th>P tail min</th><th>dominated</th><th>formal</th><th>mandatory role</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<h2>読み方</h2><p><code>d=||g_mul-g_noquant||/||g_noquant||</code>を主距離として残し、対称距離・方向回転・gainを原因分解として併記します。1.0は元勾配と同程度の差という数学的anchorですが「失敗」を意味しません。edge未解決中は最良mulを確定しません。</p>
<p>Schema / metric: {SCHEMA_VERSION} / {METRIC_DEFINITION_VERSION}<br>Acceptance contract: <code>{summary['acceptance_contract_sha256']}</code></p>
</body></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a v2.4 local acceptance profile and preregister its formal credible set."
    )
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "status": "running",
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        summary_path = profile_dir / "summary.json"
        tail_path = profile_dir / "gradient_tail.csv"
        natural_path = profile_dir / "local_natural_gradient.csv"
        manifest_path = profile_dir / "source_manifest.json"
        for path in (summary_path, tail_path, natural_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(f"required v2.4 local input is missing: {path}")
        raw_summary = _read_json(summary_path)
        source_manifest = _read_json(manifest_path)
        source_contract = str(source_manifest.get("source_contract", {}).get("sha256", ""))
        if not source_contract:
            raise ValueError("local profile source_manifest has no source contract")
        result = analyze_local_profile(
            summary=raw_summary,
            gradient_tail_rows=_read_csv(tail_path),
            dataset_id=str(args.dataset_id),
            bootstrap_iterations=int(args.iterations),
            bootstrap_seed=int(args.seed),
        )
        natural = analyze_natural_gradient_rows(
            _read_csv(natural_path),
            timestep_bins=int(raw_summary["profile"]["timestep_bins"]),
            bootstrap_iterations=int(args.iterations),
            bootstrap_seed=int(args.seed) + 1,
        )
        analysis_summary = result["summary"]
        analysis_summary.update(
            {
                "source_profile": str(profile_dir),
                "source_contract_sha256": source_contract,
                "local_summary_sha256": sha256_file(summary_path),
                "local_gradient_tail_sha256": sha256_file(tail_path),
                "local_natural_gradient_sha256": sha256_file(natural_path),
                "no_quant_natural_local_baseline": natural,
            }
        )
        write_json(output_dir / "acceptance_contract.json", result["contract"])
        write_json(output_dir / "summary.json", analysis_summary)
        write_json(output_dir / "natural_gradient_baseline.json", natural)
        write_csv(output_dir / "local_acceptance.csv", result["score_rows"])
        write_csv(output_dir / "local_timestep.csv", result["timestep_rows"])
        write_csv(output_dir / "source_bootstrap.csv", result["bootstrap_rows"])
        write_csv(output_dir / "bootstrap_regret.csv", result["regret_rows"])
        write_csv(output_dir / "robust_dominance.csv", result["dominance_rows"])
        write_csv(output_dir / "source_loo.csv", result["source_loo_rows"])
        (output_dir / "technical_report.html").write_text(
            render_report(analysis_summary, result["score_rows"], natural),
            encoding="utf-8",
        )

        rule = dict(analysis_summary["selection_rule"])
        rule_sha = canonical_json_sha256(rule)
        selection = {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "selection_valid": bool(result["selection"]["selection_valid"]),
            "diagnostic_target": "numerical_gradient_acceptance_by_fixed_range_mul",
            "not_quality_or_utility": True,
            "source_contract_sha256": source_contract,
            "local_profile_protocol": "v24-acceptance-local",
            "local_profile_dir": str(profile_dir),
            "local_summary_path": str(summary_path),
            "local_summary_sha256": sha256_file(summary_path),
            "local_gradient_tail_path": str(tail_path),
            "local_gradient_tail_sha256": sha256_file(tail_path),
            "local_natural_gradient_path": str(natural_path),
            "local_natural_gradient_sha256": sha256_file(natural_path),
            "local_analysis_summary_path": str(output_dir / "summary.json"),
            "local_analysis_summary_sha256": sha256_file(output_dir / "summary.json"),
            "selection_rule": rule,
            "selection_rule_sha256": rule_sha,
            "acceptance_contract_sha256": acceptance_contract()["contract_sha256"],
            "local_grid": analysis_summary["candidate_grid"],
            "core_grid": analysis_summary["core_grid"],
            "edge_extension": analysis_summary["edge_extension"],
            **result["selection"],
            "trajectory_status": "unknown_until_128_step_formal",
            "selector_or_utility_vote": False,
        }
        write_json(output_dir / "local_selection.json", selection)
        gate_path = profile_dir / "calibration_gate.json"
        gate_payload = _read_json(gate_path) if gate_path.is_file() else {}
        gate_rows = [
            {
                "dataset_id": str(args.dataset_id),
                "gate": "prefix_source_contract",
                "required": True,
                "passed": bool(
                    gate_payload.get("completed")
                    and gate_payload.get("source_contract_matched")
                ),
            }
        ]
        practical_model = build_single_dataset_report_model(
            dataset_id=str(args.dataset_id),
            candidate_rows=result["score_rows"],
            detail={
                "summary": analysis_summary,
                "selection": selection,
                "bootstrap_rows": result["bootstrap_rows"],
                "source_loo_rows": result["source_loo_rows"],
                "timestep_rows": result["timestep_rows"],
                "natural_baseline": natural,
            },
            gate_rows=gate_rows,
            provenance={
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_profile": str(profile_dir),
                "source_contract_sha256": source_contract,
                "report_scope": "single_dataset_local_only",
                "trajectory_product_role": "research_only_keep_product_local_only",
            },
        )
        write_json(output_dir / "practical_report.json", practical_model)
        write_json(output_dir / "report_contract.json", practical_report_contract())
        (output_dir / "report.html").write_text(
            render_practical_report(practical_model),
            encoding="utf-8",
        )
        write_json(
            output_dir / "analysis_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "metric_definition_version": METRIC_DEFINITION_VERSION,
                "inputs": {
                    path.name: {"path": str(path), "sha256": sha256_file(path)}
                    for path in (summary_path, tail_path, natural_path, manifest_path)
                },
                "source_contract_sha256": source_contract,
                "selection_rule_sha256": rule_sha,
                "acceptance_contract_sha256": analysis_summary["acceptance_contract_sha256"],
                "reports": {
                    "primary": {
                        "path": str(output_dir / "report.html"),
                        "sha256": sha256_file(output_dir / "report.html"),
                        "scope": "single_dataset_local_only",
                    },
                    "technical": {
                        "path": str(output_dir / "technical_report.html"),
                        "sha256": sha256_file(output_dir / "technical_report.html"),
                    },
                    "contract": {
                        "path": str(output_dir / "report_contract.json"),
                        "sha256": sha256_file(output_dir / "report_contract.json"),
                    },
                },
                "safety_not_utility": True,
            },
        )
        write_json(
            output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "selection_valid": selection["selection_valid"],
                "edge_unresolved": selection["edge_unresolved"],
                "primary_report": "report.html",
                "technical_report": "technical_report.html",
                "report_scope": "single_dataset_local_only",
                "trajectory_product_role": "research_only_keep_product_local_only",
            },
        )
        print(json.dumps(analysis_summary, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        write_json(
            output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
