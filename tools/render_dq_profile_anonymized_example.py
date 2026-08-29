from __future__ import annotations

"""Render the checked-in, anonymized DQ Profiler comparison example.

The input fixture contains rounded numerical Safety/Fidelity measurements only.
It deliberately excludes dataset names, captions, paths, image identifiers,
hashes, and final-quality labels.  The generated HTML is self-contained and
uses inline SVG rather than external images or scripts.
"""

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "examples"
    / "dq_dataset_profiler_anonymized_data.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "docs"
    / "examples"
    / "dq_dataset_profiler_anonymized_example.html"
)

BODY_COLOR = "#2563eb"
TAIL_COLOR = "#d97706"
EDGE_COLOR = "#7c3aed"
REFERENCE_COLOR = "#b91c1c"


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _same_mul(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= 1e-9


def _curve_svg(dataset: Mapping[str, Any], *, y_max: float) -> str:
    cards = list(dataset["candidates"])
    width, height = 620, 286
    left, right, top, bottom = 52, 20, 24, 48
    plot_width = width - left - right
    plot_height = height - top - bottom
    muls = [float(card["mul"]) for card in cards]
    x_min, x_max = min(muls), max(muls)
    if _same_mul(x_min, x_max):
        x_min -= 0.1
        x_max += 0.1

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        clipped = min(max(0.0, float(value)), y_max)
        return top + plot_height - clipped / y_max * plot_height

    grid: list[str] = []
    for index in range(5):
        value = y_max * index / 4
        y = sy(value)
        grid.append(
            f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" '
            'stroke="#dbe3ef" stroke-width="1"/>'
            f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" '
            f'class="tick">{value:.1f}</text>'
        )

    x_ticks = [
        f'<text x="{sx(value):.1f}" y="{height-17}" text-anchor="middle" '
        f'class="tick">{value:.2f}</text>'
        for value in muls
    ]

    errors: list[str] = []
    overflow: list[str] = []
    for card in cards:
        x = sx(float(card["mul"]))
        for channel, color, x_offset in (
            ("body", BODY_COLOR, -5.0),
            ("tail", TAIL_COLOR, 5.0),
        ):
            low, high = (float(value) for value in card[f"{channel}_ci"])
            errors.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(low):.1f}" '
                f'y2="{sy(high):.1f}" stroke="{color}" stroke-width="7" '
                f'opacity=".17"><title>{channel} 95% interval '
                f'[{low:.3f}, {high:.3f}]</title></line>'
            )
            if high > y_max:
                marker_x = x + x_offset
                overflow.append(
                    f'<polygon points="{marker_x-4:.1f},{top+8:.1f} '
                    f'{marker_x+4:.1f},{top+8:.1f} {marker_x:.1f},{top:.1f}" '
                    f'fill="{color}"><title>95% interval上限 {high:.3f} は'
                    f'固定Y軸 {y_max:.1f} を超過</title></polygon>'
                )

    def points(channel: str) -> str:
        return " ".join(
            f'{sx(float(card["mul"])):.1f},{sy(float(card[channel])):.1f}'
            for card in cards
        )

    markers: list[str] = []
    for card in cards:
        x = sx(float(card["mul"]))
        markers.append(
            f'<circle cx="{x:.1f}" cy="{sy(float(card["body"])):.1f}" r="4" '
            f'fill="{BODY_COLOR}"><title>mul {float(card["mul"]):.2f}; '
            f'Body {_fmt(card["body"])}</title></circle>'
        )
        markers.append(
            f'<rect x="{x-4:.1f}" y="{sy(float(card["tail"]))-4:.1f}" '
            f'width="8" height="8" fill="{TAIL_COLOR}"><title>'
            f'mul {float(card["mul"]):.2f}; Tail {_fmt(card["tail"])}</title></rect>'
        )

    reference_y = sy(1.0)
    edge_line = ""
    if dataset.get("edge_unresolved"):
        edge_x = sx(max(muls))
        edge_line = (
            f'<line x1="{edge_x:.1f}" x2="{edge_x:.1f}" y1="{top}" '
            f'y2="{top+plot_height}" stroke="{EDGE_COLOR}" stroke-width="2" '
            'stroke-dasharray="5 5"/>'
            f'<text x="{width-right}" y="{height-31}" text-anchor="end" '
            'class="edge">上側を未解決 →</text>'
        )

    return f"""
<svg class="chart" viewBox="0 0 {width} {height}" role="img"
     aria-label="{html.escape(str(dataset['id']))}のmul affinity curve。青はBody、橙はTail、半透明の棒は95%区間">
  <style>
    .tick{{font:11px system-ui,sans-serif;fill:#526176}}
    .axis-label{{font:11px system-ui,sans-serif;fill:#334155}}
    .reference{{font:10px system-ui,sans-serif;fill:#991b1b;font-weight:700}}
    .edge{{font:10px system-ui,sans-serif;fill:#6d28d9;font-weight:700}}
  </style>
  {''.join(grid)}
  <line x1="{left}" x2="{width-right}" y1="{reference_y:.1f}" y2="{reference_y:.1f}"
        stroke="{REFERENCE_COLOR}" stroke-width="1.5" stroke-dasharray="7 5"/>
  <text x="{width-right-3}" y="{reference_y-5:.1f}" text-anchor="end"
        class="reference">距離1.0（画質の合否線ではない）</text>
  {edge_line}
  {''.join(errors)}
  {''.join(overflow)}
  <polyline points="{points('body')}" fill="none" stroke="{BODY_COLOR}" stroke-width="3"/>
  <polyline points="{points('tail')}" fill="none" stroke="{TAIL_COLOR}" stroke-width="3"/>
  {''.join(markers)}
  {''.join(x_ticks)}
  <text x="{width/2}" y="{height-2}" text-anchor="middle" class="axis-label">range_mul</text>
  <text x="14" y="{height/2}" transform="rotate(-90 14 {height/2})"
        text-anchor="middle" class="axis-label">gradient deformation</text>
</svg>
"""


def _dataset_card(dataset: Mapping[str, Any], *, y_max: float) -> str:
    cards = list(dataset["candidates"])
    body_min = min(cards, key=lambda item: (float(item["body"]), float(item["mul"])))
    tail_min = min(cards, key=lambda item: (float(item["tail"]), float(item["mul"])))
    tail_max = max(float(item["tail"]) for item in cards)
    retained = " / ".join(f'{float(value):.2f}' for value in dataset["retained_muls"])
    return f"""
<article class="dataset-card">
  <header>
    <div>
      <h3>{html.escape(str(dataset['id']))}</h3>
      <p>{html.escape(str(dataset['pattern']))}</p>
    </div>
    <span class="edge-pill">{'EDGE UNRESOLVED' if dataset.get('edge_unresolved') else 'EDGE RESOLVED'}</span>
  </header>
  {_curve_svg(dataset, y_max=y_max)}
  <div class="mini-grid">
    <div><span>Body最小点</span><strong>mul {float(body_min['mul']):.2f} / {_fmt(body_min['body'])}</strong></div>
    <div><span>Tail最小点</span><strong>mul {float(tail_min['mul']):.2f} / {_fmt(tail_min['tail'])}</strong></div>
    <div><span>観測Tail最大</span><strong>{tail_max:.3f}</strong></div>
    <div><span>Fidelity retained</span><strong>{html.escape(retained)}</strong></div>
  </div>
</article>
"""


def _summary_row(dataset: Mapping[str, Any]) -> str:
    cards = list(dataset["candidates"])
    body_values = [float(card["body"]) for card in cards]
    tail_values = [float(card["tail"]) for card in cards]
    body_min = min(cards, key=lambda item: float(item["body"]))
    tail_min = min(cards, key=lambda item: float(item["tail"]))
    return (
        "<tr>"
        f'<th>{html.escape(str(dataset["id"]))}</th>'
        f'<td>{html.escape(str(dataset["pattern"]))}</td>'
        f'<td>{min(body_values):.3f}–{max(body_values):.3f}</td>'
        f'<td>{min(tail_values):.3f}–{max(tail_values):.3f}</td>'
        f'<td>{float(body_min["mul"]):.2f}</td>'
        f'<td>{float(tail_min["mul"]):.2f}</td>'
        f'<td>{int(dataset["source_group_count"])}</td>'
        "</tr>"
    )


def render(payload: Mapping[str, Any]) -> str:
    datasets = list(payload["datasets"])
    y_max = float(payload["fixed_y_max"])
    body_points = [float(card["body"]) for item in datasets for card in item["candidates"]]
    tail_points = [float(card["tail"]) for item in datasets for card in item["candidates"]]
    dataset_cards = "".join(_dataset_card(item, y_max=y_max) for item in datasets)
    summary_rows = "".join(_summary_row(item) for item in datasets)
    paired = [item for item in datasets if str(item["id"]).startswith("Dataset F")]
    paired_note = ""
    if len(paired) == 2:
        first, second = paired
        first_by_mul = {float(card["mul"]): card for card in first["candidates"]}
        second_by_mul = {float(card["mul"]): card for card in second["candidates"]}
        shared = sorted(set(first_by_mul) & set(second_by_mul))
        pair_rows = "".join(
            "<tr>"
            f"<th>{mul:.2f}</th>"
            f'<td>{float(first_by_mul[mul]["body"]):.3f}</td>'
            f'<td>{float(second_by_mul[mul]["body"]):.3f}</td>'
            f'<td>{float(first_by_mul[mul]["tail"]):.3f}</td>'
            f'<td>{float(second_by_mul[mul]["tail"]):.3f}</td>'
            "</tr>"
            for mul in shared
        )
        paired_note = f"""
<section>
  <h2>同一画像・タグ設計だけを変えたpaired例</h2>
  <p>Dataset F1とF2は画像集合を共通にし、タグの組み立てだけを変更した匿名paired例です。
  mul 3.45ではBodyが0.966から0.659へ変わる一方、Tailは1.484と1.471で近いなど、
  caption設計だけでも曲線の形が変わることを確認できます。ただしsourceが6群なので、
  長い95%区間を含み、細かな優劣の断定には向きません。</p>
  <div class="table-wrap"><table>
    <thead><tr><th>mul</th><th>F1 Body</th><th>F2 Body</th><th>F1 Tail</th><th>F2 Tail</th></tr></thead>
    <tbody>{pair_rows}</tbody>
  </table></div>
</section>
"""

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(payload['title']))}</title>
<style>
:root{{--ink:#172033;--muted:#5b677a;--line:#dbe3ef;--panel:#fff;--bg:#f4f7fb;--blue:{BODY_COLOR};--orange:{TAIL_COLOR};--purple:{EDGE_COLOR}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;line-height:1.65}}
main{{max-width:1320px;margin:auto;padding:28px 22px 64px}} h1{{font-size:clamp(1.8rem,4vw,3rem);line-height:1.15;margin:.2em 0}} h2{{margin-top:2.2em}} h3{{font-size:1.3rem;margin:0}}
.hero,.notice,section,.dataset-card{{background:var(--panel);border:1px solid var(--line);border-radius:16px;box-shadow:0 7px 24px rgba(30,48,80,.06)}}
.hero{{padding:28px;background:linear-gradient(135deg,#eef5ff,#fff 62%,#f6f0ff)}} .eyebrow{{font-size:.78rem;letter-spacing:.14em;font-weight:800;color:#4f46e5}}
.hero p{{max-width:900px}} .metrics{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:20px}} .metrics div,.mini-grid div{{padding:12px;border-radius:11px;background:#f7f9fc;border:1px solid #e5eaf2}}
.metrics span,.mini-grid span{{display:block;color:var(--muted);font-size:.78rem}} .metrics strong,.mini-grid strong{{display:block;font-size:1.08rem}}
.notice,section{{padding:22px;margin-top:20px}} .warning{{border-left:5px solid #dc2626}} .legend{{display:flex;flex-wrap:wrap;gap:16px;color:var(--muted);font-size:.9rem}}
.dot,.square,.bar{{display:inline-block;margin-right:7px;vertical-align:middle}} .dot{{width:10px;height:10px;border-radius:50%;background:var(--blue)}} .square{{width:10px;height:10px;background:var(--orange)}} .bar{{width:7px;height:24px;background:rgba(37,99,235,.18)}}
.dataset-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px}} .dataset-card{{overflow:hidden}} .dataset-card header{{padding:18px 20px 0;display:flex;justify-content:space-between;gap:14px;align-items:flex-start}} .dataset-card header p{{margin:.3em 0;color:var(--muted)}}
.edge-pill{{font-size:.68rem;font-weight:800;color:#5b21b6;background:#f1eaff;border:1px solid #c7aaf4;border-radius:999px;padding:5px 8px;white-space:nowrap}} .chart{{display:block;width:100%;height:auto;padding:4px 8px}}
.mini-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:0 18px 18px}} .table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse;font-size:.9rem}} th,td{{border-bottom:1px solid var(--line);padding:9px 11px;text-align:right}} th:first-child,td:first-child,td:nth-child(2){{text-align:left}} thead th{{background:#eef3f9;color:#405069;position:sticky;top:0}}
code{{background:#eef2f7;border-radius:5px;padding:.1em .35em}} footer{{margin-top:30px;color:var(--muted);font-size:.82rem}} a{{color:#1d4ed8}}
@media(max-width:850px){{.dataset-grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<article class="hero">
  <div class="eyebrow">ANONYMIZED MEASURED EXAMPLE · NUMERICAL SAFETY/FIDELITY</div>
  <h1>{html.escape(str(payload['title']))}</h1>
  <p>保存済み実測から点推定とsource-cluster bootstrap区間だけを抜粋し、名称・パス・caption・画像識別子を除いた比較例です。すべてのchartを固定Y軸0–{y_max:.1f}で描き、dataset間の高さを直接比較できます。</p>
  <div class="metrics">
    <div><span>匿名dataset config</span><strong>{len(datasets)}件</strong></div>
    <div><span>Body点推定の観測幅</span><strong>{min(body_points):.3f}–{max(body_points):.3f}</strong></div>
    <div><span>Tail点推定の観測幅</span><strong>{min(tail_points):.3f}–{max(tail_points):.3f}</strong></div>
  </div>
</article>

<aside class="notice warning">
  <strong>この例は画質比較ではありません。</strong>
  値が小さいほどno-quant勾配に近いことを示しますが、最終画像が良いことや量子化のUtilityを保証しません。
</aside>

<section>
  <h2>グラフの読み方</h2>
  <div class="legend">
    <span><i class="dot"></i>Body: 通常範囲の変形</span>
    <span><i class="square"></i>Tail: 最も厳しいtimestep帯</span>
    <span><i class="bar"></i>上下の半透明棒: source-cluster bootstrap 95%区間</span>
  </div>
  <p>棒が長いほど、どの独立sourceを含めるかによる不確実性が大きく、候補の細かな順位を断定しにくいことを示します。量子化が必ず悪いという意味ではありません。CI上限が固定Y軸4.0を超える場合は、chart上端の三角で示し、値を打ち切っています。</p>
</section>

<section>
  <h2>同じmulでもdatasetごとに違う</h2>
  <p>Dataset Aは全候補でBody/Tailが1未満ですが、Dataset Cは全候補の点推定が1を超えます。Dataset BはBodyよりTailが大きく、Dataset Dはmul増加に沿って両方が低下します。このように、単一の共通mulだけでは説明できない反応差が観測されています。</p>
  <div class="table-wrap"><table>
    <thead><tr><th>匿名ID</th><th>観測パターン</th><th>Body範囲</th><th>Tail範囲</th><th>Body最小mul</th><th>Tail最小mul</th><th>source群</th></tr></thead>
    <tbody>{summary_rows}</tbody>
  </table></div>
</section>

<div class="dataset-grid">{dataset_cards}</div>
{paired_note}

<section>
  <h2>この例から確認できること／できないこと</h2>
  <ul>
    <li>datasetごとにBody/Tailの絶対水準、mul依存、Tail増幅、区間幅が異なります。</li>
    <li>同一画像でもタグ設計を変えると、勾配変形曲線が変わる例があります。</li>
    <li>上端候補が残ったdatasetは<code>edge_unresolved</code>であり、best mulを宣言できません。</li>
    <li>Fidelity retainedは数値的な候補集合で、最終画質の推薦ではありません。</li>
    <li>最終画質との対応には、別のcontrolled Utility Bridgeが必要です。</li>
  </ul>
</section>

<footer>
  Checked-in anonymized example schema {html.escape(str(payload['schema_version']))}. Values are rounded snapshots of measured v2.4 numerical outputs. No external CDN, images, fonts, or scripts are used; all charts are inline SVG.
</footer>
</main></body></html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8-sig"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(payload), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
