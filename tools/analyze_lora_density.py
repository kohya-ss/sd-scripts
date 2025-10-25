import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from safetensors.torch import load_file

from library.utils import add_logging_arguments, setup_logging


logger = logging.getLogger(__name__)


@dataclass
class TensorAnalysis:
    name: str
    kind: str
    shape: List[int]
    dtype: str
    num_params: int
    mean: float
    std: float
    median: float
    min: float
    max: float
    abs_mean: float
    max_abs: float
    l1_norm: float
    l2_norm: float
    rms: float
    sparsity: float
    entropy: float
    entropy_norm: float
    skewness: float
    kurtosis: float
    svd_rows: Optional[int] = None
    svd_cols: Optional[int] = None
    svd_top_energy_ratio: Optional[float] = None
    svd_top4_energy_ratio: Optional[float] = None
    svd_rank90: Optional[int] = None
    svd_effective_rank: Optional[float] = None


@dataclass
class BlockReport:
    name: str
    module: str
    alpha: Optional[float]
    total_params: int
    density_score: float
    density_components: Dict[str, float]
    tensors: List[TensorAnalysis]
    aggregated: Dict[str, Any]


def infer_module_name(layer_name: str) -> str:
    base = layer_name
    if base.startswith("lora_"):
        base = base[len("lora_") :]
    parts = base.split("_")
    if not parts:
        return "unknown"
    prefix = parts[0]
    if prefix in {"unet", "te1", "te2"}:
        return prefix
    if prefix == "textencoder":
        return "te1"
    if prefix == "clip":
        return "te2"
    return prefix


def infer_unet_block_label(layer_name: str) -> str:
    if not layer_name.startswith("lora_unet_"):
        return "unet"
    suffix = layer_name[len("lora_unet_") :]
    tokens = suffix.split("_")
    if len(tokens) >= 3 and tokens[0] in {"input", "output"} and tokens[1] == "blocks":
        return f"{tokens[0]}_blocks_{tokens[2]}"
    if len(tokens) >= 2 and tokens[0] == "middle" and tokens[1] == "block":
        return "middle_block"
    return tokens[0]


def _quantile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    fraction = position - lower_index
    return float(lower_value + (upper_value - lower_value) * fraction)


def compute_distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "q1": None, "q3": None}
    sorted_values = sorted(values)
    total = sum(values)
    count = len(values)
    return {
        "min": float(sorted_values[0]),
        "max": float(sorted_values[-1]),
        "mean": float(total / count),
        "median": _quantile(sorted_values, 0.5),
        "q1": _quantile(sorted_values, 0.25),
        "q3": _quantile(sorted_values, 0.75),
    }


def summarize_block(report: BlockReport) -> Dict[str, Any]:
    aggregated = report.aggregated
    components = report.density_components
    top_energy_values = [t.svd_top_energy_ratio for t in report.tensors if t.svd_top_energy_ratio is not None]
    effective_rank_values = [t.svd_effective_rank for t in report.tensors if t.svd_effective_rank is not None]

    def average(values: List[Optional[float]]) -> Optional[float]:
        usable = [v for v in values if v is not None]
        if not usable:
            return None
        return float(sum(usable) / len(usable))

    return {
        "name": report.name,
        "module": report.module,
        "alpha": report.alpha,
        "total_params": report.total_params,
        "density_score": report.density_score,
        "rms": aggregated.get("rms"),
        "entropy_norm": aggregated.get("entropy_norm"),
        "non_zero_ratio": components.get("non_zero_ratio"),
        "sparsity": aggregated.get("sparsity"),
        "mean": aggregated.get("mean"),
        "std": aggregated.get("std"),
        "max_abs": aggregated.get("max_abs"),
        "l1_norm": aggregated.get("l1_norm"),
        "l2_norm": aggregated.get("l2_norm"),
        "skewness": aggregated.get("skewness"),
        "kurtosis": aggregated.get("kurtosis"),
        "svd_top_energy_ratio_max": max(top_energy_values) if top_energy_values else None,
        "svd_top_energy_ratio_mean": average(top_energy_values),
        "svd_effective_rank_mean": average(effective_rank_values),
        "tensor_count": len(report.tensors),
    }


def load_lora_state(path: str) -> Dict[str, torch.Tensor]:
    logger.info("Loading LoRA: %s", path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu")


def group_lora_layers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    layers: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in state_dict.items():
        if key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            layers.setdefault(base, {})["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            base = key[: -len(".lora_up.weight")]
            layers.setdefault(base, {})["up"] = tensor
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            layers.setdefault(base, {})["alpha"] = tensor
        else:
            layers.setdefault(key, {})[os.path.split(key)[-1]] = tensor
    return layers


def compute_histogram_entropy(values: torch.Tensor, bins: int) -> Tuple[float, float]:
    finite_mask = torch.isfinite(values)
    if not bool(finite_mask.all()):
        values = values[finite_mask]
    if values.numel() == 0:
        return 0.0, 0.0

    min_val = float(values.min().item())
    max_val = float(values.max().item())
    if math.isclose(min_val, max_val):
        adjustment = 1e-6 if min_val >= 0 else -1e-6
        max_val = min_val + adjustment

    hist = torch.histc(values, bins=bins, min=min_val, max=max_val)
    counts = hist.float()

    total = float(sum(counts))
    if total <= 0:
        entropy = 0.0
        entropy_norm = 0.0
    else:
        probabilities = counts / total
        non_zero = probabilities > 0
        entropy = float(-(probabilities[non_zero] * torch.log(probabilities[non_zero])).sum().item())
        max_entropy = math.log(bins)
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(entropy), float(entropy_norm)


def compute_svd_metrics(tensor: torch.Tensor) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[float], Optional[int], Optional[float]]:
    if tensor.ndim == 0:
        return None, None, None, None, None, None

    mat = tensor.detach().cpu().float()
    if mat.ndim == 1:
        mat = mat.unsqueeze(0)
    elif mat.ndim > 2:
        mat = mat.reshape(mat.shape[0], -1)

    rows, cols = mat.shape
    if rows == 0 or cols == 0:
        return rows, cols, None, None, None, None

    try:
        singular_values = torch.linalg.svdvals(mat)
    except RuntimeError as exc:
        logger.debug("SVD failed for shape %s: %s", list(mat.shape), exc)
        return rows, cols, None, None, None, None

    if singular_values.numel() == 0:
        return rows, cols, None, None, None, None

    energy = singular_values.pow(2)
    energy_sum = energy.sum().item()
    if energy_sum == 0.0:
        return rows, cols, None, None, None, 0.0

    energy_prob = energy / energy_sum
    cumulative = torch.cumsum(energy_prob, dim=0)
    rank90_tensor = torch.nonzero(cumulative >= 0.9, as_tuple=False)
    rank90 = int(rank90_tensor[0].item() + 1) if rank90_tensor.numel() > 0 else singular_values.numel()

    top_energy_ratio = energy_prob[0].item()
    top4_energy_ratio = energy_prob[:4].sum().item() if energy_prob.numel() >= 4 else energy_prob.sum().item()

    effective_rank = float(torch.exp(-(energy_prob * torch.log(energy_prob)).sum()).item())

    return rows, cols, float(top_energy_ratio), float(top4_energy_ratio), rank90, effective_rank


def compute_tensor_analysis(
    name: str,
    kind: str,
    tensor: torch.Tensor,
    bins: int,
) -> TensorAnalysis:
    tensor_cpu = tensor.detach().cpu()
    shape = list(tensor_cpu.shape)
    num_params = int(torch.numel(tensor_cpu))

    if num_params == 0:
        mean = std = median = min_val = max_val = abs_mean = max_abs = 0.0
        l1_norm = l2_norm = rms = 0.0
        sparsity = 0.0
        skewness = kurtosis = 0.0
        entropy = entropy_norm = 0.0
    else:
        values = tensor_cpu.flatten().float()
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).item())
        median = float(values.median().item())
        min_val = float(values.min().item())
        max_val = float(values.max().item())
        abs_mean = float(values.abs().mean().item())
        max_abs = float(values.abs().max().item())
        l1_norm = float(values.abs().sum().item())
        l2_norm = float(torch.linalg.vector_norm(values).item())
        rms = float(torch.sqrt((values.pow(2).mean())).item())
        zero_ratio = float((values == 0).sum().item()) / float(num_params)
        sparsity = zero_ratio

        centered = values - mean
        if std > 0:
            skewness = float((centered.pow(3).mean() / (std ** 3)).item())
            kurtosis = float((centered.pow(4).mean() / (std ** 4)).item())
        else:
            skewness = 0.0
            kurtosis = 0.0

        entropy, entropy_norm = compute_histogram_entropy(values, bins)

    svd_rows, svd_cols, top_energy_ratio, top4_energy_ratio, rank90, eff_rank = compute_svd_metrics(tensor_cpu)

    return TensorAnalysis(
        name=name,
        kind=kind,
        shape=shape,
        dtype=str(tensor_cpu.dtype),
        num_params=num_params,
        mean=mean,
        std=std,
        median=median,
        min=min_val,
        max=max_val,
        abs_mean=abs_mean,
        max_abs=max_abs,
        l1_norm=l1_norm,
        l2_norm=l2_norm,
        rms=rms,
        sparsity=sparsity,
        entropy=entropy,
        entropy_norm=entropy_norm,
        skewness=skewness,
        kurtosis=kurtosis,
        svd_rows=svd_rows,
        svd_cols=svd_cols,
        svd_top_energy_ratio=top_energy_ratio,
        svd_top4_energy_ratio=top4_energy_ratio,
        svd_rank90=rank90,
        svd_effective_rank=eff_rank,
    )


def build_block_reports(
    layers: Dict[str, Dict[str, torch.Tensor]],
    bins: int,
) -> Tuple[List[BlockReport], Dict[str, Any]]:
    block_reports: List[BlockReport] = []
    aggregated_rms_values: List[float] = []

    for name in sorted(layers.keys()):
        layer = layers[name]
        tensors: List[TensorAnalysis] = []
        total_params = 0

        if "down" in layer:
            analysis = compute_tensor_analysis(name, "lora_down", layer["down"], bins)
            tensors.append(analysis)
            total_params += analysis.num_params
        if "up" in layer:
            analysis = compute_tensor_analysis(name, "lora_up", layer["up"], bins)
            tensors.append(analysis)
            total_params += analysis.num_params

        flattened_parts: List[torch.Tensor] = []
        for tensor_key in ("down", "up"):
            if tensor_key in layer:
                flattened_parts.append(layer[tensor_key].detach().cpu().float().flatten())

        aggregated_stats: Dict[str, Any] = {}
        if flattened_parts:
            combined = torch.cat(flattened_parts)
            combined_analysis = compute_tensor_analysis(name, "combined", combined, bins)
            aggregated_stats = {
                "num_params": combined_analysis.num_params,
                "mean": combined_analysis.mean,
                "std": combined_analysis.std,
                "median": combined_analysis.median,
                "min": combined_analysis.min,
                "max": combined_analysis.max,
                "abs_mean": combined_analysis.abs_mean,
                "max_abs": combined_analysis.max_abs,
                "l1_norm": combined_analysis.l1_norm,
                "l2_norm": combined_analysis.l2_norm,
                "rms": combined_analysis.rms,
                "sparsity": combined_analysis.sparsity,
                "entropy": combined_analysis.entropy,
                "entropy_norm": combined_analysis.entropy_norm,
                "skewness": combined_analysis.skewness,
                "kurtosis": combined_analysis.kurtosis,
            }
            aggregated_rms_values.append(combined_analysis.rms)
        else:
            aggregated_stats = {
                "num_params": 0,
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "abs_mean": 0.0,
                "max_abs": 0.0,
                "l1_norm": 0.0,
                "l2_norm": 0.0,
                "rms": 0.0,
                "sparsity": 0.0,
                "entropy": 0.0,
                "entropy_norm": 0.0,
                "skewness": 0.0,
                "kurtosis": 0.0,
            }

        alpha_tensor = layer.get("alpha")
        alpha_value = None
        if isinstance(alpha_tensor, torch.Tensor):
            if alpha_tensor.numel() == 1:
                alpha_value = float(alpha_tensor.item())
            else:
                alpha_value = float(alpha_tensor.detach().cpu().float().mean().item())

        density_components = {
            "entropy_norm": aggregated_stats["entropy_norm"],
            "non_zero_ratio": 1.0 - aggregated_stats["sparsity"],
            "rms": aggregated_stats["rms"],
        }

        block_reports.append(
            BlockReport(
                name=name,
                module=infer_module_name(name),
                alpha=alpha_value,
                total_params=total_params,
                density_score=0.0,
                density_components=density_components,
                tensors=tensors,
                aggregated=aggregated_stats,
            )
        )

    max_rms = max(aggregated_rms_values) if aggregated_rms_values else 0.0
    for report in block_reports:
        components = report.density_components
        entropy_term = components["entropy_norm"]
        sparsity_term = components["non_zero_ratio"]
        if max_rms > 0:
            rms_term = components["rms"] / max_rms
        else:
            rms_term = 0.0
        report.density_score = entropy_term * sparsity_term * rms_term
        report.density_components["rms_norm"] = rms_term

    global_summary = {
        "total_blocks": len(block_reports),
        "total_params": sum(r.total_params for r in block_reports),
        "max_rms": max_rms,
    }

    return block_reports, global_summary




def build_html_report(data: Dict[str, Any]) -> str:
    report_json = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    generated_at = data.get('generated_at', '')
    model_path = data.get('model_path', '')
    summary = data.get('summary', {})
    module_summary = data.get('module_summary', [])
    module_rankings = data.get('module_rankings', [])
    unet_block_summary = data.get('unet_block_summary', [])
    sparse_blocks = data.get('sparse_blocks', [])
    blocks = data.get('blocks', [])
    settings = data.get('settings', {})

    module_density_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for entry in module_summary:
        density_stats = entry.get('density', {}) or {}
        module_density_bounds[entry.get('module', '-') ] = (
            density_stats.get('min'),
            density_stats.get('max'),
        )

    def format_float(val: Any, precision: int = 4) -> str:
        if val is None:
            return '-'
        if isinstance(val, (int, float)):
            if not math.isfinite(val):
                return str(val)
            return f"{val:.{precision}f}"
        return str(val)

    def format_int(val: Any) -> str:
        if val is None:
            return '-'
        if isinstance(val, (int, float)):
            return f"{int(val):,}"
        return str(val)

    def module_label(module: str) -> str:
        lookup = {
            'unet': 'UNet',
            'te1': 'TE1 (Text Encoder 1)',
            'te2': 'TE2 (Text Encoder 2)',
        }
        return lookup.get(module, module.upper())

    def format_unet_block(label: str) -> str:
        if label.startswith("input_blocks_"):
            return f"Input {label.split('_')[-1]}"
        if label.startswith("output_blocks_"):
            return f"Output {label.split('_')[-1]}"
        if label == "middle_block":
            return "Middle Block"
        return label.replace("_", " ").title()

    def density_color(norm: Optional[float]) -> str:
        if norm is None:
            return ''
        norm = max(0.0, min(1.0, norm))
        low = (252, 165, 165)    # red-300
        mid = (255, 255, 255)    # white
        high = (74, 222, 128)    # green-400

        def blend(color_a, color_b, t: float) -> Tuple[int, int, int]:
            return tuple(int(round(a + (b - a) * t)) for a, b in zip(color_a, color_b))

        if norm < 0.5:
            ratio = norm / 0.5
            rgb = blend(low, mid, ratio)
        else:
            ratio = (norm - 0.5) / 0.5
            rgb = blend(mid, high, ratio)
        return '#%02x%02x%02x' % rgb

    def render_cards(cards: List[Tuple[str, str, str]]) -> str:
        parts: List[str] = []
        for title, value, subtitle in cards:
            parts.append(
                f'<div class="card"><h3>{title}</h3><p class="value">{value}</p><p class="sub">{subtitle}</p></div>'
            )
        return ''.join(parts)

    def render_module_table(items: List[Dict[str, Any]]) -> str:
        rows: List[str] = []
        for item in items:
            density = item.get('density', {}) or {}
            rms = item.get('rms', {}) or {}
            entropy = item.get('entropy_norm', {}) or {}
            cells = (
                f"<td>{module_label(item.get('module', '-'))}</td>"
                f'<td class="num">{format_int(item.get("block_count"))}</td>'
                f'<td class="num">{format_int(item.get("total_params"))}</td>'
                f'<td class="num">{format_float(density.get("mean"))}</td>'
                f'<td class="num">{format_float(density.get("median"))}</td>'
                f'<td class="num">{format_float(rms.get("median"))}</td>'
                f'<td class="num">{format_float(entropy.get("median"))}</td>'
            )
            rows.append(f'<tr>{cells}</tr>')
        if not rows:
            rows.append("<tr><td colspan='7' class='empty'>データがありません</td></tr>")
        return ''.join(rows)

    def render_block_table(
        items: List[Dict[str, Any]],
        columns: List[Tuple[str, str]],
        module_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
        highlight_density: bool = False,
        include_module_badge: bool = False,
    ) -> str:
        header = ''.join(f'<th>{label}</th>' for _, label in columns)
        rows: List[str] = []
        for item in items:
            row_style = ''
            if highlight_density and module_bounds:
                norm = item.get('module_density_norm')
                if norm is None:
                    bounds = module_bounds.get(item.get('module')) if module_bounds else None
                    if bounds:
                        low, high = bounds
                        value = item.get('density_score')
                        if value is not None and low is not None and high is not None:
                            if math.isclose(high, low):
                                norm = 0.5
                            else:
                                norm = (value - low) / (high - low)
                if norm is not None:
                    color = density_color(norm)
                    if color:
                        row_style = f' style="background-color: {color};"'
            cells: List[str] = []
            for key, _ in columns:
                value = item.get(key)
                if key in {'total_params', 'tensor_count'}:
                    cells.append(f"<td class='num'>{format_int(value)}</td>")
                elif key in {
                    'density_score',
                    'rms',
                    'entropy_norm',
                    'non_zero_ratio',
                    'svd_top_energy_ratio_max',
                    'svd_top_energy_ratio_mean',
                    'svd_effective_rank_mean',
                    'mean',
                    'std',
                    'skewness',
                    'kurtosis',
                    'alpha',
                }:
                    cells.append(f"<td class='num'>{format_float(value)}</td>")
                else:
                    display = '-' if value is None else value
                    if key == 'name' and include_module_badge:
                        module = item.get('module')
                        badge = ''
                        if module:
                            badge = f"<span class='module-tag'>{module_label(module)}</span>"
                        cells.append(f"<td>{display}{badge}</td>")
                    else:
                        cells.append(f"<td>{display}</td>")
            rows.append(f"<tr{row_style}>" + ''.join(cells) + "</tr>")
        if not rows:
            rows.append(f"<tr><td colspan='{len(columns)}' class='empty'>データがありません</td></tr>")
        return (
            '<table>'
            '<thead><tr>' + header + '</tr></thead>'
            '<tbody>' + ''.join(rows) + '</tbody>'
            '</table>'
        )
    
    def render_unet_block_table(items: List[Dict[str, Any]]) -> str:
        if not items:
            return "<p class='empty'>UNetブロックの統計はありません。</p>"
        rows: List[str] = []
        for item in items:
            density = item.get('density', {}) or {}
            rms = item.get('rms', {}) or {}
            cells = (
                f"<td>{format_unet_block(item.get('label', '-'))}</td>"
                f"<td class='num'>{format_int(item.get('block_count'))}</td>"
                f"<td class='num'>{format_int(item.get('total_params'))}</td>"
                f"<td class='num'>{format_float(density.get('mean'))}</td>"
                f"<td class='num'>{format_float(density.get('median'))}</td>"
                f"<td class='num'>{format_float(rms.get('median'))}</td>"
            )
            rows.append(f"<tr>{cells}</tr>")
        return (
            "<table>"
            "<thead><tr>"
            "<th>UNetブロック</th>"
            "<th class='num'>LoRA数</th>"
            "<th class='num'>総パラメータ</th>"
            "<th class='num'>情報密度平均</th>"
            "<th class='num'>情報密度中央値</th>"
            "<th class='num'>RMS中央値</th>"
            "</tr></thead>"
            "<tbody>"
            + ''.join(rows)
            + "</tbody></table>"
        )

    density_stats = summary.get('density', {}) or {}
    rms_stats = summary.get('rms', {}) or {}
    entropy_stats = summary.get('entropy_norm', {}) or {}

    overview_cards = [
        ('総ブロック数', format_int(summary.get('total_blocks')), f"総パラメータ: {format_int(summary.get('total_params'))}"),
        ('情報密度中央値', format_float(density_stats.get('median')), f"範囲 {format_float(density_stats.get('min'))} – {format_float(density_stats.get('max'))}"),
        ('RMS中央値', format_float(rms_stats.get('median')), f"範囲 {format_float(rms_stats.get('min'))} – {format_float(rms_stats.get('max'))}"),
        ('エントロピー中央値', format_float(entropy_stats.get('median')), f"範囲 {format_float(entropy_stats.get('min'))} – {format_float(entropy_stats.get('max'))}"),
    ]

    block_columns = [
        ('name', 'ブロック'),
        ('density_score', '情報密度'),
        ('rms', 'RMS'),
        ('entropy_norm', 'エントロピー'),
        ('non_zero_ratio', '非ゼロ率'),
        ('alpha', 'α'),
        ('total_params', 'パラメータ数'),
        ('svd_top_energy_ratio_max', 'SVDトップ比'),
    ]
    shortlist_columns = block_columns[:5]

    module_sections: List[str] = []
    for entry in module_rankings:
        module = entry.get('module', '-')
        label = module_label(module)
        top_blocks = entry.get('top', [])
        bottom_blocks = entry.get('bottom', [])
        top_html = render_block_table(top_blocks, shortlist_columns, module_density_bounds)
        bottom_html = render_block_table(bottom_blocks, shortlist_columns, module_density_bounds)
        module_sections.append(
            f"""
            <article class='module-ranking'>
                <h3>{label}</h3>
                <div class='module-tables'>
                    <div>
                        <h4>上位 {entry.get('top_count', len(top_blocks))} ブロック</h4>
                        {top_html}
                    </div>
                    <div>
                        <h4>下位 {entry.get('bottom_count', len(bottom_blocks))} ブロック</h4>
                        {bottom_html}
                    </div>
                </div>
            </article>
            """
        )
    module_rankings_html = ''.join(module_sections) if module_sections else "<p class='empty'>データがありません</p>"

    if sparse_blocks:
        sparse_table = render_block_table(
            sparse_blocks,
            block_columns,
            module_density_bounds,
            highlight_density=True,
            include_module_badge=True,
        )
    else:
        sparse_table = "<p>しきい値未満のスパースブロックはありません。</p>"

    all_blocks_table = render_block_table(
        blocks,
        block_columns,
        module_density_bounds,
        highlight_density=True,
        include_module_badge=True,
    )

    html = f"""<!DOCTYPE html>
<html lang='ja'>
<head>
    <meta charset='utf-8'>
    <title>LoRA 情報密度レポート</title>
    <style>
        body {{
            font-family: "Segoe UI", "Hiragino Kaku Gothic ProN", Meiryo, sans-serif;
            margin: 0 auto;
            max-width: 1080px;
            padding: 0 1.5rem 4rem;
            background-color: #f8fafc;
            color: #1f2937;
        }}
        header.page-header {{
            padding: 1.5rem 0;
            border-bottom: 1px solid #d9dde5;
            margin-bottom: 1.5rem;
        }}
        h1 {{
            margin: 0 0 0.5rem;
        }}
        .meta {{
            color: #4b5563;
            font-size: 0.95rem;
        }}
        .cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .card {{
            background: #ffffff;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }}
        .card h3 {{
            margin: 0;
            font-size: 0.95rem;
            color: #4b5563;
        }}
        .card p.value {{
            margin: 0.4rem 0 0.2rem;
            font-size: 1.6rem;
            font-weight: 600;
            color: #1f2937;
        }}
        .card p.sub {{
            margin: 0;
            color: #6b7280;
            font-size: 0.9rem;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: #ffffff;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.5rem;
        }}
        th, td {{
            border-bottom: 1px solid #e5e7eb;
            padding: 0.45rem 0.6rem;
            text-align: left;
        }}
        th {{
            font-weight: 600;
            color: #374151;
        }}
        td.num, th.num {{
            text-align: right;
            font-variant-numeric: tabular-nums;
        }}
        .module-tag {{
            display: inline-block;
            margin-left: 0.4rem;
            padding: 0.12rem 0.5rem;
            background: #dbeafe;
            color: #1d4ed8;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        section {{
            margin-bottom: 2.5rem;
        }}
        section > h2 {{
            margin-bottom: 0.75rem;
            color: #111827;
        }}
        .module-ranking {{
            background: #ffffff;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
            margin-bottom: 1.2rem;
        }}
        .module-ranking h3 {{
            margin: 0 0 0.6rem;
            color: #1f2937;
        }}
        .module-tables {{
            display: grid;
            gap: 1rem;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        }}
        .module-tables > div {{
            background: #f9fafb;
            border-radius: 10px;
            padding: 0.6rem;
            box-shadow: inset 0 0 0 1px #e5e7eb;
            overflow-x: auto;
        }}
        .module-tables table {{
            box-shadow: none;
            margin-bottom: 0;
        }}
        .module-tables table th, .module-tables table td {{
            border-bottom: 1px solid #e5e7eb;
        }}
        .module-tables h4 {{
            margin: 0 0 0.5rem;
            font-size: 1rem;
            color: #1f2937;
        }}
        .empty {{
            text-align: center;
            padding: 1rem 0;
            color: #6b7280;
        }}
        footer {{
            margin-top: 3rem;
            color: #6b7280;
            font-size: 0.85rem;
            text-align: center;
        }}
    </style>
</head>
<body>
    <header class='page-header'>
        <h1>LoRA 情報密度レポート</h1>
        <p class='meta'>モデル: {model_path} | 生成日時: {generated_at}</p>
        <p class='meta'>ヒストグラムビン数: {settings.get('bins')} | 全体Top/Bottom: {settings.get('top_n')} | モジュール別Top/Bottom: {settings.get('module_top_n')}</p>
    </header>
    <section id='overview'>
        <h2>サマリー</h2>
        <div class='cards'>
            {render_cards(overview_cards)}
        </div>
    </section>
    <section id='modules'>
        <h2>モジュール別統計</h2>
        <table>
            <thead>
                <tr>
                    <th>モジュール</th>
                    <th class='num'>ブロック数</th>
                    <th class='num'>総パラメータ</th>
                    <th class='num'>情報密度平均</th>
                    <th class='num'>情報密度中央値</th>
                    <th class='num'>RMS中央値</th>
                    <th class='num'>エントロピー中央値</th>
                </tr>
            </thead>
            <tbody>
                {render_module_table(module_summary)}
            </tbody>
        </table>
    </section>
    <section id='unet-blocks'>
        <h2>UNetブロック別概要</h2>
        {render_unet_block_table(unet_block_summary)}
    </section>
    <section id='module-rankings'>
        <h2>モジュール別 上位/下位ブロック</h2>
        {module_rankings_html}
    </section>
    <section id='sparse-blocks'>
        <h2>スパースブロック（非ゼロ率 &lt; {format_float(settings.get('sparse_threshold'))}）</h2>
        {sparse_table}
    </section>
    <section id='all-blocks'>
        <h2>全ブロック一覧（主要指標）</h2>
        {all_blocks_table}
    </section>
    <footer>
        <p>このレポートは sd-scripts の analyze_lora_density ツールで生成されました。</p>
    </footer>
    <script>
        window.__LORA_DENSITY_REPORT__ = {report_json};
    </script>
</body>
</html>
"""
    return html

def analyze_lora_density(
    model_path: str,
    report_path: Optional[str],
    json_path: Optional[str],
    bins: int,
) -> Dict[str, Any]:
    state_dict = load_lora_state(model_path)
    layers = group_lora_layers(state_dict)
    blocks, base_summary = build_block_reports(layers, bins)

    block_snapshots = [summarize_block(block) for block in blocks]

    density_values = [snap["density_score"] for snap in block_snapshots]
    rms_values = [snap["rms"] for snap in block_snapshots if snap["rms"] is not None]
    entropy_values = [snap["entropy_norm"] for snap in block_snapshots if snap["entropy_norm"] is not None]
    sparsity_values = [snap["sparsity"] for snap in block_snapshots if snap["sparsity"] is not None]

    summary = {
        "total_blocks": base_summary.get("total_blocks"),
        "total_params": base_summary.get("total_params"),
        "max_rms": base_summary.get("max_rms"),
        "density": compute_distribution(density_values),
        "rms": compute_distribution(rms_values),
        "entropy_norm": compute_distribution(entropy_values),
        "sparsity": compute_distribution(sparsity_values),
    }

    module_groups: Dict[str, List[Dict[str, Any]]] = {}
    for snapshot in block_snapshots:
        module_groups.setdefault(snapshot["module"], []).append(snapshot)

    module_summary = []
    module_density_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for module in sorted(module_groups.keys()):
        group = module_groups[module]
        density_values_group = [item["density_score"] for item in group]
        rms_values_group = [item["rms"] for item in group if item["rms"] is not None]
        entropy_values_group = [item["entropy_norm"] for item in group if item["entropy_norm"] is not None]

        density_stats_group = compute_distribution(density_values_group)
        module_density_bounds[module] = (density_stats_group["min"], density_stats_group["max"])

        module_summary.append(
            {
                "module": module,
                "block_count": len(group),
                "total_params": sum(item["total_params"] for item in group),
                "density": density_stats_group,
                "rms": compute_distribution(rms_values_group),
                "entropy_norm": compute_distribution(entropy_values_group),
            }
        )

    for snapshot in block_snapshots:
        module = snapshot["module"]
        bounds = module_density_bounds.get(module, (None, None))
        low, high = bounds
        norm = None
        if low is not None and high is not None:
            if math.isclose(high, low):
                norm = 0.5
            else:
                norm = (snapshot["density_score"] - low) / (high - low)
        snapshot["module_density_norm"] = None if norm is None else max(0.0, min(1.0, norm))

    top_n_global = min(12, len(block_snapshots))
    top_blocks = sorted(block_snapshots, key=lambda item: item["density_score"], reverse=True)[:top_n_global]
    bottom_blocks = sorted(block_snapshots, key=lambda item: item["density_score"])[:top_n_global]

    module_rank_limit = 8
    module_rankings = []
    for module in sorted(module_groups.keys()):
        group = module_groups[module]
        top_group = sorted(group, key=lambda item: item["density_score"], reverse=True)[: module_rank_limit]
        bottom_group = sorted(group, key=lambda item: item["density_score"])[: module_rank_limit]
        module_rankings.append(
            {
                "module": module,
                "top": top_group,
                "bottom": bottom_group,
                "top_count": len(top_group),
                "bottom_count": len(bottom_group),
            }
        )

    unet_block_groups: Dict[str, List[Dict[str, Any]]] = {}
    for snapshot in block_snapshots:
        if snapshot["module"] != "unet":
            continue
        block_label = infer_unet_block_label(snapshot["name"])
        unet_block_groups.setdefault(block_label, []).append(snapshot)

    unet_block_summary = []
    for block_label in sorted(unet_block_groups.keys()):
        group = unet_block_groups[block_label]
        density_stats_group = compute_distribution([item["density_score"] for item in group])
        unet_block_summary.append(
            {
                "label": block_label,
                "block_count": len(group),
                "total_params": sum(item["total_params"] for item in group),
                "density": density_stats_group,
                "rms": compute_distribution([item["rms"] for item in group if item["rms"] is not None]),
            }
        )

    sparse_threshold = 0.99
    sparse_blocks = [
        item for item in block_snapshots if item["non_zero_ratio"] is not None and item["non_zero_ratio"] < sparse_threshold
    ]
    sparse_blocks = sorted(sparse_blocks, key=lambda item: item["non_zero_ratio"])

    report_data = {
        "model_path": model_path,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "settings": {
            "bins": bins,
            "top_n": top_n_global,
            "module_top_n": module_rank_limit,
            "sparse_threshold": sparse_threshold,
        },
        "summary": summary,
        "module_summary": module_summary,
        "module_rankings": module_rankings,
        "unet_block_summary": unet_block_summary,
        "top_blocks": top_blocks,
        "bottom_blocks": bottom_blocks,
        "sparse_blocks": sparse_blocks,
        "blocks": block_snapshots,
    }

    if json_path:
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(report_data, fp, ensure_ascii=False, indent=2)
        logger.info("Saved JSON report to %s", json_path)

    if report_path:
        html = build_html_report(report_data)
        with open(report_path, "w", encoding="utf-8") as fp:
            fp.write(html)
        logger.info("Saved HTML report to %s", report_path)

    return report_data


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRAブロック密度分析レポート生成ツール")
    parser.add_argument("model", type=str, help="解析対象のLoRAファイル（.safetensors など）")
    parser.add_argument("--report", type=str, default=None, help="HTMLレポートの出力パス")
    parser.add_argument("--json", type=str, default=None, help="分析結果JSONの出力パス")
    parser.add_argument("--bins", type=int, default=128, help="ヒストグラムのビン数")
    add_logging_arguments(parser)
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    setup_logging(args, reset=True)
    analyze_lora_density(
        model_path=args.model,
        report_path=args.report,
        json_path=args.json,
        bins=args.bins,
    )


if __name__ == "__main__":
    main()
