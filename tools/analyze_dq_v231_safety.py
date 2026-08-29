from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from dq_profile.v231_safety import (
    DEFAULT_BOOTSTRAP_ITERATIONS,
    DEFAULT_BOOTSTRAP_SEED,
    METRIC_DEFINITION_VERSION,
    SCHEMA_VERSION,
    analyze_profile_directory,
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            payload = dict(row)
            if isinstance(payload.get("reason_codes"), list):
                payload["reason_codes"] = ";".join(payload["reason_codes"])
            writer.writerow(payload)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def render_chart(score_rows: list[dict[str, Any]]) -> str:
    width, height = 800, 340
    left, right, top, bottom = 72, 30, 28, 62
    plot_width = width - left - right
    plot_height = height - top - bottom
    muls = [float(row["range_mul"]) for row in score_rows]
    risk_values = [
        float(row[key])
        for row in score_rows
        for key in ("local_risk_L", "trajectory_risk_T", "combined_risk_R")
        if row.get(key) is not None and math.isfinite(float(row[key]))
    ]
    y_max = max(1.25, max(risk_values, default=1.0) * 1.12)
    x_min, x_max = min(muls), max(muls)

    def x(value: float) -> float:
        if math.isclose(x_min, x_max):
            return left + plot_width / 2
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y(value: float) -> float:
        return top + plot_height - value / y_max * plot_height

    series = (
        ("local_risk_L", "Local L", "#d95f02"),
        ("trajectory_risk_T", "Trajectory T", "#1b9e77"),
        ("combined_risk_R", "Combined R", "#5e3c99"),
    )
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="range mul numerical safety risk curve">',
        f'<rect width="{width}" height="{height}" fill="#fff"/>',
    ]
    for boundary, label, color in (
        (0.5, "provisional tolerant", "#d8b365"),
        (1.0, "mathematical anchor", "#c51b7d"),
    ):
        y_pos = y(boundary)
        parts.append(
            f'<line x1="{left}" y1="{y_pos:.2f}" x2="{width-right}" '
            f'y2="{y_pos:.2f}" stroke="{color}" stroke-dasharray="6 5"/>'
        )
        parts.append(
            f'<text x="{left+5}" y="{y_pos-5:.2f}" fill="{color}" '
            f'font-size="12">{label} ({boundary:g})</text>'
        )
    parts.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top+plot_height}" stroke="#333"/>',
            f'<line x1="{left}" y1="{top+plot_height}" x2="{width-right}" '
            f'y2="{top+plot_height}" stroke="#333"/>',
        ]
    )
    for tick in range(6):
        value = y_max * tick / 5
        y_pos = y(value)
        parts.append(
            f'<text x="{left-8}" y="{y_pos+4:.2f}" text-anchor="end" '
            f'font-size="11">{value:.2f}</text>'
        )
    for mul in muls:
        x_pos = x(mul)
        parts.append(
            f'<text x="{x_pos:.2f}" y="{top+plot_height+24}" '
            f'text-anchor="middle" font-size="12">{mul:g}</text>'
        )
    for key, label, color in series:
        points = " ".join(
            f'{x(float(row["range_mul"])):.2f},{y(float(row[key])):.2f}'
            for row in score_rows
            if row.get(key) is not None and math.isfinite(float(row[key]))
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.5"/>'
        )
        for row in score_rows:
            if row.get(key) is None or not math.isfinite(float(row[key])):
                continue
            parts.append(
                f'<circle cx="{x(float(row["range_mul"])):.2f}" '
                f'cy="{y(float(row[key])):.2f}" r="4" fill="{color}"/>'
            )
        legend_x = left + 410 + 115 * list(series).index((key, label, color))
        parts.append(
            f'<line x1="{legend_x}" y1="15" x2="{legend_x+18}" y2="15" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{legend_x+23}" y="19" font-size="11">{label}</text>'
        )
    parts.append(
        f'<text x="{left+plot_width/2:.2f}" y="{height-10}" '
        'text-anchor="middle" font-size="13">range_mul</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def render_report(summary: dict[str, Any], score_rows: list[dict[str, Any]]) -> str:
    table_rows: list[str] = []
    for row in score_rows:
        reasons = ", ".join(row.get("reason_codes", [])) or "none"
        table_rows.append(
            "<tr>"
            f"<td>{float(row['range_mul']):.2f}</td>"
            f"<td>{_number(row['local_risk_L'])}</td>"
            f"<td>{_number(row['trajectory_risk_T'])}</td>"
            f"<td>{_number(row['combined_risk_R'])}</td>"
            f"<td>{_number(row['display_score_S'], 1)}</td>"
            f"<td>{_number(100 * float(row['bootstrap_best_probability']), 1)}%</td>"
            f"<td>{_number(row['catastrophic_q99_d'])}</td>"
            f"<td>{int(row['catastrophic_q99_timestep_bin'])}</td>"
            f"<td>{_number(row['catastrophic_max_d'])}</td>"
            f"<td>{html.escape(str(row['classification']))}</td>"
            f"<td>{html.escape(reasons)}</td>"
            "</tr>"
        )
    edge = (
        f"{summary['edge_extension_direction']}: "
        f"{summary['edge_extension_recommended_muls']}"
        if summary["edge_unresolved"]
        else "not recommended"
    )
    preferred = summary.get("numerical_safety_preferred_mul")
    preferred_text = str(preferred) if preferred is not None else "indistinguishable"
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>DQ Profiler v2.3.1 Safety — {html.escape(summary['dataset_id'])}</title>
<style>
body{{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:28px;color:#20272b;line-height:1.55}}
h1,h2{{color:#263238}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}}
.card{{border:1px solid #ccd5d9;border-radius:9px;padding:14px 18px;margin:12px 0}}
.warn{{background:#fff5e6;border-color:#e6b566}} .info{{background:#eef5fb;border-color:#8db7d6}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border:1px solid #ccd5d9;padding:6px;text-align:right}}
th:nth-child(10),td:nth-child(10),th:last-child,td:last-child{{text-align:left}} code{{background:#f3f5f6;padding:2px 4px}}
</style>
</head>
<body>
<h1>DQ Profiler v2.3.1 — Numerical Safety</h1>
<div class="grid">
<div class="card"><b>Dataset</b><br>{html.escape(summary['dataset_id'])}</div>
<div class="card"><b>Safety phenotype</b><br>{html.escape(summary['phenotype'])}</div>
<div class="card"><b>点推定の最小R</b><br>mul {summary['point_estimate_best_mul']} / S {_number(summary['point_estimate_best_score'], 1)}</div>
<div class="card"><b>順位判定</b><br>{html.escape(summary['ranking_status'])}<br>preferred: {html.escape(preferred_text)}</div>
<div class="card"><b>Bootstrap最頻勝者</b><br>mul {summary['bootstrap_modal_best_mul']} / {_number(100 * float(summary['bootstrap_modal_best_probability']), 1)}%</div>
<div class="card"><b>観測tolerant envelope</b><br>{html.escape(str(summary['tested_tolerant_envelope']))}</div>
</div>
<div class="card warn">
<b>SafetyはQualityでもUtilityでもありません。</b><br>
点推定でRが最小のmulは推薦ではありません。候補順位は共有bootstrapで最頻勝率75%以上の場合だけ確定します。
S=100/(1+R)のR=0.5境界は暫定で、R=1だけがno_quant自然変動q95という数学的anchorを持ちます。
</div>
<div class="card info">
<b>C99は稀な破綻の説明用警報です。</b><br>
C99=max_t q99(d) と全timestep最大値を表示しますが、主指標R=max(L,T)には加えていません。
後から都合のよい票を増やさないための分離です。Edge extension: {html.escape(edge)}
</div>
<h2>Risk curve</h2>
{render_chart(score_rows)}
<h2>mul別結果</h2>
<table>
<thead><tr><th>mul</th><th>L</th><th>T</th><th>R</th><th>S</th><th>P(best)</th><th>C99</th><th>C99 bin</th><th>Cmax</th><th>判定</th><th>理由</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<h2>契約</h2>
<p>Schema / metric: {SCHEMA_VERSION} / {METRIC_DEFINITION_VERSION}<br>
Contract SHA-256: <code>{summary['safety_contract_sha256']}</code></p>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a frozen v2.1 tail profile as v2.3.1 numerical safety "
            "with paired ranking uncertainty and all-timestep catastrophic tails."
        )
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
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        result = analyze_profile_directory(
            profile_dir,
            dataset_id=str(args.dataset_id),
            bootstrap_iterations=int(args.iterations),
            bootstrap_seed=int(args.seed),
        )
        summary = result["summary"]
        write_json(output_dir / "safety_contract.json", result["contract"])
        write_json(output_dir / "summary.json", summary)
        write_json(
            output_dir / "safety_scores.json",
            {
                "schema_version": SCHEMA_VERSION,
                "metric_definition_version": METRIC_DEFINITION_VERSION,
                "dataset_id": args.dataset_id,
                "score_rows": result["score_rows"],
            },
        )
        write_csv(output_dir / "safety_scores.csv", result["score_rows"])
        write_csv(output_dir / "timestep_risk.csv", result["timestep_rows"])
        write_csv(output_dir / "safety_bootstrap.csv", result["bootstrap_rows"])
        write_csv(output_dir / "candidate_ranking.csv", result["ranking_rows"])
        (output_dir / "report.html").write_text(
            render_report(summary, result["score_rows"]),
            encoding="utf-8",
        )
        manifest_entries = []
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name not in {
                "analysis_manifest.json",
                "status.json",
            }:
                manifest_entries.append(
                    {
                        "path": path.name,
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        write_json(
            output_dir / "analysis_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "metric_definition_version": METRIC_DEFINITION_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_id": args.dataset_id,
                "source_profile": str(profile_dir),
                "source_summary_sha256": summary["source_summary_sha256"],
                "safety_contract_sha256": summary["safety_contract_sha256"],
                "entries": manifest_entries,
            },
        )
        write_json(
            output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "complete",
                "summary_sha256": sha256_file(output_dir / "summary.json"),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as error:
        write_json(
            output_dir / "status.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
