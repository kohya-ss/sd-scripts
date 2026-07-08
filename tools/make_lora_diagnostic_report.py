import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

ColorPalette = [
    "#0f766e",
    "#2563eb",
    "#dc2626",
    "#7c3aed",
    "#d97706",
    "#16a34a",
    "#0891b2",
    "#be185d",
]

DEFAULT_LORA_EPOCH_TREND_MAX_SERIES = 18
DQ_LOW_QERR_PER_CLIP_THRESHOLD = 130.0


def module_label(module: str) -> str:
    mapping = {
        "unet": "UNet",
        "te1": "TE1 (Text Encoder 1)",
        "te2": "TE2 (Text Encoder 2)",
    }
    return mapping.get(module, str(module).upper())


def format_unet_block_label(label: str) -> str:
    if not label:
        return "-"
    if label.startswith("input_blocks_"):
        suffix = label.split("_")[-1]
        return f"Input {suffix}"
    if label.startswith("output_blocks_"):
        suffix = label.split("_")[-1]
        return f"Output {suffix}"
    if label == "middle_block":
        return "Middle Block"
    return label.replace("_", " ").title()


def import_lora_analysis_tools() -> Tuple[Any, Any, Any]:
    try:
        try:
            from tools.analyze_lora_density import (  # type: ignore
                build_checkpoint_history,
                collect_single_analysis,
                find_checkpoint_series,
            )
        except Exception:
            from analyze_lora_density import build_checkpoint_history, collect_single_analysis, find_checkpoint_series  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "analyze_lora_density.py の読み込みに失敗しました。torch/safetensors が必要です。"
        ) from exc
    return collect_single_analysis, find_checkpoint_series, build_checkpoint_history


def load_lora_module_param_counts(model_path: str) -> Dict[str, int]:
    try:
        try:
            from tools.analyze_lora_density import group_lora_layers, load_lora_state  # type: ignore
        except Exception:
            from analyze_lora_density import group_lora_layers, load_lora_state  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "analyze_lora_density.py の読み込みに失敗しました。torch/safetensors が必要です。"
        ) from exc

    state_dict = load_lora_state(model_path)
    layers = group_lora_layers(state_dict)
    param_counts: Dict[str, int] = {}
    for name, layer in layers.items():
        total = 0
        if "down" in layer:
            total += int(layer["down"].numel())
        if "up" in layer:
            total += int(layer["up"].numel())
        if total > 0:
            param_counts[name] = total
    return param_counts


def color_for_index(index: int, total: int) -> str:
    if total <= 0:
        return "#2563eb"
    hue = (index * 137.508) % 360.0
    saturation = 68.0
    lightness = 45.0 if (index % 2 == 0) else 55.0
    c = (1.0 - abs(2.0 * (lightness / 100.0) - 1.0)) * (saturation / 100.0)
    x = c * (1.0 - abs((hue / 60.0) % 2.0 - 1.0))
    m = (lightness / 100.0) - c / 2.0
    if hue < 60:
        rp, gp, bp = c, x, 0
    elif hue < 120:
        rp, gp, bp = x, c, 0
    elif hue < 180:
        rp, gp, bp = 0, c, x
    elif hue < 240:
        rp, gp, bp = 0, x, c
    elif hue < 300:
        rp, gp, bp = x, 0, c
    else:
        rp, gp, bp = c, 0, x
    r = int(round((rp + m) * 255))
    g = int(round((gp + m) * 255))
    b = int(round((bp + m) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def parse_te_layer_index(label: str, display_label: str) -> Optional[int]:
    patterns = [label, display_label]
    for source in patterns:
        text = (source or "").strip().lower()
        match = re.search(r"layer[_ ](\d+)", text)
        if match:
            return int(match.group(1))
    return None


def lora_trend_sort_key(module: str, entry: Dict[str, Any]) -> Tuple[int, int, str]:
    label = str(entry.get("label") or "")
    display = str(entry.get("display_label") or "")
    if module == "unet":
        if label.startswith("input_blocks_"):
            suffix = label.split("_")[-1]
            idx = int(suffix) if suffix.isdigit() else 999
            return (0, idx, display)
        if label == "middle_block":
            return (1, 0, display)
        if label.startswith("output_blocks_"):
            suffix = label.split("_")[-1]
            idx = int(suffix) if suffix.isdigit() else 999
            return (2, idx, display)
        return (3, 999, display)

    layer_idx = parse_te_layer_index(label, display)
    if layer_idx is not None:
        return (0, layer_idx, display)
    return (1, 999, display)


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_int(value: Any) -> Optional[int]:
    number = safe_float(value)
    if number is None:
        return None
    return int(number)


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        return [dict(row) for row in reader]


def moving_average(values: List[Optional[float]], window: int) -> List[Optional[float]]:
    if window <= 1:
        return list(values)
    result: List[Optional[float]] = []
    queue: List[float] = []
    running_sum = 0.0
    for value in values:
        if value is None:
            result.append(None)
            continue
        queue.append(value)
        running_sum += value
        if len(queue) > window:
            running_sum -= queue.pop(0)
        result.append(running_sum / len(queue))
    return result


def mean_tail(values: List[Optional[float]], ratio: float, take_tail: bool) -> Optional[float]:
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    count = max(1, int(len(valid) * ratio))
    target = valid[-count:] if take_tail else valid[:count]
    return float(sum(target) / len(target))


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_percent(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def infer_steps_per_epoch(rows: List[Dict[str, str]]) -> int:
    if not rows:
        return 1
    step_max_by_epoch: Dict[int, int] = {}
    for row in rows:
        epoch = safe_int(row.get("Epoch"))
        step = safe_int(row.get("Step"))
        if epoch is None or step is None:
            continue
        current = step_max_by_epoch.get(epoch, 0)
        if step > current:
            step_max_by_epoch[epoch] = step
    if not step_max_by_epoch:
        return 1
    candidates = [value + 1 for value in step_max_by_epoch.values() if value >= 0]
    if not candidates:
        return 1
    return int(statistics.median(candidates))


def parse_grad_log(path: str, ma_window: int) -> Dict[str, Any]:
    rows = read_csv_rows(path)
    steps_per_epoch = infer_steps_per_epoch(rows)

    x_values: List[int] = []
    epochs: List[Optional[int]] = []
    gradient_norm: List[Optional[float]] = []
    threshold: List[Optional[float]] = []
    loss: List[Optional[float]] = []
    thresh_off: List[Optional[float]] = []
    scale: List[Optional[float]] = []
    cosine: List[Optional[float]] = []
    first_step_by_epoch: Dict[int, int] = {}

    threshold_valid = 0
    threshold_exceeded = 0
    thresh_off_count = 0
    cosine_valid = 0

    for row in rows:
        epoch_value = safe_int(row.get("Epoch"))
        step_value = safe_int(row.get("Step"))
        if epoch_value is None or step_value is None:
            continue
        global_step = epoch_value * steps_per_epoch + step_value
        x_values.append(global_step)
        epochs.append(epoch_value + 1)
        if epoch_value not in first_step_by_epoch:
            first_step_by_epoch[epoch_value] = global_step

        grad = safe_float(row.get("Gradient Norm"))
        th = safe_float(row.get("Threshold"))
        lo = safe_float(row.get("Loss"))
        toff = safe_float(row.get("ThreshOff"))
        sc = safe_float(row.get("Scale"))
        cs = safe_float(row.get("CosineSim"))

        gradient_norm.append(grad)
        threshold.append(th)
        loss.append(lo)
        thresh_off.append(toff)
        scale.append(sc)
        cosine.append(cs)

        if toff is not None and toff > 0.0:
            thresh_off_count += 1
        if cs is not None:
            cosine_valid += 1
        if grad is not None and th is not None and th > 0:
            threshold_valid += 1
            if grad > th:
                threshold_exceeded += 1

    loss_ma = moving_average(loss, ma_window)
    loss_start = mean_tail(loss_ma, 0.15, take_tail=False)
    loss_end = mean_tail(loss_ma, 0.15, take_tail=True)
    loss_drop = None
    if loss_start is not None and loss_end is not None and loss_start > 0:
        loss_drop = (loss_start - loss_end) / loss_start

    markers = [
        {"x": step_value, "label": f"E{epoch + 1}"}
        for epoch, step_value in sorted(first_step_by_epoch.items(), key=lambda item: item[0])
    ]
    return {
        "path": path,
        "rows": len(x_values),
        "steps_per_epoch": steps_per_epoch,
        "x": x_values,
        "epochs": epochs,
        "markers": markers,
        "gradient_norm": gradient_norm,
        "threshold": threshold,
        "loss": loss,
        "loss_ma": loss_ma,
        "thresh_off": thresh_off,
        "scale": scale,
        "cosine": cosine,
        "summary": {
            "threshold_valid_count": threshold_valid,
            "threshold_exceeded_count": threshold_exceeded,
            "threshold_exceeded_ratio": safe_ratio(float(threshold_exceeded), float(threshold_valid))
            if threshold_valid
            else None,
            "thresh_off_ratio": safe_ratio(float(thresh_off_count), float(len(x_values))) if x_values else None,
            "loss_ma_start": loss_start,
            "loss_ma_end": loss_end,
            "loss_ma_drop_ratio": loss_drop,
            "cosine_valid_ratio": safe_ratio(float(cosine_valid), float(len(x_values))) if x_values else None,
            "max_grad_norm": max((v for v in gradient_norm if v is not None), default=None),
        },
    }


def infer_epoch_from_step(train_step: int, grad_data: Optional[Dict[str, Any]]) -> Optional[int]:
    if not grad_data:
        return None
    steps_per_epoch = grad_data.get("steps_per_epoch")
    if not isinstance(steps_per_epoch, int) or steps_per_epoch <= 0:
        return None
    return (train_step // steps_per_epoch) + 1


def parse_dq_logs(
    dq_log_path: Optional[str],
    dq_auto_path: Optional[str],
    grad_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    rows = read_csv_rows(dq_log_path) if dq_log_path else []
    parsed_rows: List[Dict[str, Any]] = []
    for row in rows:
        train_step = safe_int(row.get("TrainStep"))
        if train_step is None:
            continue
        epoch = safe_int(row.get("Epoch"))
        if epoch is None:
            epoch = infer_epoch_from_step(train_step, grad_data)

        parsed_rows.append(
            {
                "TrainStep": train_step,
                "Epoch": epoch,
                "Bits": safe_float(row.get("Bits")),
                "RangeMul": safe_float(row.get("RangeMul")),
                "ClipRateRaw": safe_float(row.get("ClipRateRaw")),
                "ClipRateEMA": safe_float(row.get("ClipRateEMA")),
                "QuantErrRatioRaw": safe_float(row.get("QuantErrRatioRaw")),
                "QuantErrRatioEMA": safe_float(row.get("QuantErrRatioEMA")),
                "QuantErrRMSRaw": safe_float(row.get("QuantErrRMSRaw")),
                "QuantErrRMSEMA": safe_float(row.get("QuantErrRMSEMA")),
                "QErrPerClip": safe_float(row.get("QErrPerClip")),
                "QErrPerClipClipFloor": safe_float(row.get("QErrPerClipClipFloor")),
                "ActiveClipBand": (row.get("ActiveClipBand") or "").strip(),
                "ActiveClipLow": safe_float(row.get("ActiveClipLow")),
                "ActiveClipHigh": safe_float(row.get("ActiveClipHigh")),
                "ClipRateLowAutoState": (row.get("ClipRateLowAutoState") or "").strip(),
                "ClipRateLowAutoBad": safe_int(row.get("ClipRateLowAutoBad")),
                "ClipRateLowAutoBadStreak": safe_int(row.get("ClipRateLowAutoBadStreak")),
                "TrainProgress": safe_float(row.get("TrainProgress")),
                "ClipRateLowAutoMinProgress": safe_float(row.get("ClipRateLowAutoMinProgress")),
                "ClipRateLowAutoFreezeProgress": safe_float(row.get("ClipRateLowAutoFreezeProgress")),
                "ClipRateLowAutoThresholdQErrRatio": safe_float(row.get("ClipRateLowAutoThresholdQErrRatio")),
                "ClipRateLowAutoThresholdQErrPerClip": safe_float(row.get("ClipRateLowAutoThresholdQErrPerClip")),
                "ClipRateLowAutoPhase": (row.get("ClipRateLowAutoPhase") or "").strip(),
                "ClipErrRMS": safe_float(row.get("ClipErrRMS")),
                "RoundErrRMS": safe_float(row.get("RoundErrRMS")),
                "ClipErrRatio": safe_float(row.get("ClipErrRatio")),
                "RoundErrRatio": safe_float(row.get("RoundErrRatio")),
                "ClipShare": safe_float(row.get("ClipShare")),
                "RoundShare": safe_float(row.get("RoundShare")),
                "ZeroRate": safe_float(row.get("ZeroRate")),
                "AbsMax": safe_float(row.get("AbsMax")),
                "Range": safe_float(row.get("Range")),
                "AutoReason": (row.get("AutoReason") or "").strip(),
            }
        )
    parsed_rows.sort(key=lambda item: item["TrainStep"])

    auto_rows: List[Dict[str, Any]] = []
    if dq_auto_path and os.path.exists(dq_auto_path):
        raw_auto_rows = read_csv_rows(dq_auto_path)
        for row in raw_auto_rows:
            step = safe_int(row.get("TrainStep"))
            if step is None:
                continue
            auto_rows.append(
                {
                    "TrainStep": step,
                    "Bits": safe_float(row.get("Bits")),
                    "ClipRateRaw": safe_float(row.get("ClipRateRaw")),
                    "ClipRateEMA": safe_float(row.get("ClipRateEMA")),
                    "RangeMul": safe_float(row.get("RangeMul") or row.get("RangeMulAfter")),
                    "RangeMulBefore": safe_float(row.get("RangeMulBefore")),
                    "RangeMulAfter": safe_float(row.get("RangeMulAfter")),
                    "AutoApplied": safe_int(row.get("AutoApplied")),
                    "WarmupActive": safe_int(row.get("WarmupActive")),
                    "AutoReason": (row.get("AutoReason") or "").strip(),
                    "AutoInitClipTarget": safe_float(row.get("AutoInitClipTarget")),
                    "QErrPerClip": safe_float(row.get("QErrPerClip")),
                    "QErrPerClipClipFloor": safe_float(row.get("QErrPerClipClipFloor")),
                    "ActiveClipBand": (row.get("ActiveClipBand") or "").strip(),
                    "ActiveClipLow": safe_float(row.get("ActiveClipLow")),
                    "ActiveClipHigh": safe_float(row.get("ActiveClipHigh")),
                    "ClipRateLowAutoState": (row.get("ClipRateLowAutoState") or "").strip(),
                    "ClipRateLowAutoDecision": (row.get("ClipRateLowAutoDecision") or "").strip(),
                    "ClipRateLowAutoReason": (row.get("ClipRateLowAutoReason") or "").strip(),
                    "ClipRateLowAutoBad": safe_int(row.get("ClipRateLowAutoBad")),
                    "ClipRateLowAutoBadStreak": safe_int(row.get("ClipRateLowAutoBadStreak")),
                    "TrainProgress": safe_float(row.get("TrainProgress")),
                    "ClipRateLowAutoMinProgress": safe_float(row.get("ClipRateLowAutoMinProgress")),
                    "ClipRateLowAutoFreezeProgress": safe_float(row.get("ClipRateLowAutoFreezeProgress")),
                    "ClipRateLowAutoThresholdQErrRatio": safe_float(row.get("ClipRateLowAutoThresholdQErrRatio")),
                    "ClipRateLowAutoThresholdQErrPerClip": safe_float(row.get("ClipRateLowAutoThresholdQErrPerClip")),
                    "ClipRateLowAutoPhase": (row.get("ClipRateLowAutoPhase") or "").strip(),
                    "ClipRateLowAutoCanEscape": safe_int(row.get("ClipRateLowAutoCanEscape")),
                }
            )
        auto_rows.sort(key=lambda item: item["TrainStep"])

    markers: List[Dict[str, Any]] = []
    last_epoch: Optional[int] = None
    for item in parsed_rows:
        epoch = item.get("Epoch")
        if epoch is None:
            continue
        if last_epoch != epoch:
            markers.append({"x": item["TrainStep"], "label": f"E{epoch}"})
            last_epoch = epoch

    bits_source = parsed_rows if parsed_rows else auto_rows
    bits_values = [item["Bits"] for item in bits_source if item.get("Bits") is not None]
    bit_switches = 0
    if bits_values:
        prev = bits_values[0]
        for current in bits_values[1:]:
            if current != prev:
                bit_switches += 1
            prev = current

    auto_reason_counter = Counter()
    auto_applied_count = 0
    non_warmup_count = 0
    in_band_count = 0
    clip_target_values: List[float] = []
    for row in auto_rows:
        reason = row.get("AutoReason") or ""
        if reason:
            auto_reason_counter[reason] += 1
        if row.get("AutoApplied") == 1:
            auto_applied_count += 1
        warmup = row.get("WarmupActive")
        if warmup == 0:
            non_warmup_count += 1
            if reason == "in_band":
                in_band_count += 1
        target = row.get("AutoInitClipTarget")
        if target is not None:
            clip_target_values.append(target)

    latest = parsed_rows[-1] if parsed_rows else (auto_rows[-1] if auto_rows else {})
    clip_source = parsed_rows if parsed_rows else auto_rows
    clip_ema_values = [item["ClipRateEMA"] for item in clip_source if item.get("ClipRateEMA") is not None]
    clip_ema_cv = None
    if len(clip_ema_values) >= 8:
        tail = clip_ema_values[-max(8, len(clip_ema_values) // 4) :]
        tail_mean = sum(tail) / len(tail)
        if tail_mean > 0:
            variance = sum((x - tail_mean) ** 2 for x in tail) / len(tail)
            clip_ema_cv = math.sqrt(variance) / tail_mean

    auto_has_qerr_per_clip = any(item.get("QErrPerClip") is not None for item in auto_rows)
    low_diag_source = auto_rows if auto_has_qerr_per_clip else parsed_rows
    qerr_per_clip_values = [item["QErrPerClip"] for item in low_diag_source if item.get("QErrPerClip") is not None]
    qerr_tail_mean = mean_tail(qerr_per_clip_values, 0.25, True) if qerr_per_clip_values else None
    qerr_max = max(qerr_per_clip_values) if qerr_per_clip_values else None
    final_qerr_per_clip = qerr_per_clip_values[-1] if qerr_per_clip_values else None
    low_auto_bad_values = [
        item["ClipRateLowAutoBad"] for item in low_diag_source if item.get("ClipRateLowAutoBad") is not None
    ]
    low_auto_bad_count = sum(1 for value in low_auto_bad_values if value == 1)
    low_auto_bad_ratio = safe_ratio(float(low_auto_bad_count), float(len(low_auto_bad_values))) if low_auto_bad_values else None
    low_auto_streak_values = [
        item["ClipRateLowAutoBadStreak"]
        for item in low_diag_source
        if item.get("ClipRateLowAutoBadStreak") is not None
    ]
    low_auto_states = [str(item.get("ClipRateLowAutoState") or "") for item in low_diag_source]
    low_auto_decisions = [str(item.get("ClipRateLowAutoDecision") or "") for item in low_diag_source]
    low_auto_reasons = [str(item.get("ClipRateLowAutoReason") or "") for item in low_diag_source]
    low_auto_phases = [str(item.get("ClipRateLowAutoPhase") or "") for item in low_diag_source]
    low_auto_escaped = any(value in ("escape_to_mid", "mid_lock") for value in low_auto_states + low_auto_decisions)
    low_auto_frozen_bad_count = sum(
        1
        for item in low_diag_source
        if item.get("ClipRateLowAutoBad") == 1
        and (
            item.get("ClipRateLowAutoState") == "frozen"
            or item.get("ClipRateLowAutoReason") == "freeze_progress"
        )
    )
    active_bands = [str(item.get("ActiveClipBand") or "") for item in low_diag_source if item.get("ActiveClipBand")]
    active_band_counts = dict(Counter(active_bands))
    active_clip_band_final = active_bands[-1] if active_bands else None
    low_band_seen = any(band == "low" for band in active_bands) or bool(low_auto_bad_values)
    run_qerr_ratio_thresholds = [
        item["ClipRateLowAutoThresholdQErrRatio"]
        for item in low_diag_source
        if item.get("ClipRateLowAutoThresholdQErrRatio") is not None
    ]
    run_qerr_per_clip_thresholds = [
        item["ClipRateLowAutoThresholdQErrPerClip"]
        for item in low_diag_source
        if item.get("ClipRateLowAutoThresholdQErrPerClip") is not None
    ]
    min_progress_values = [
        item["ClipRateLowAutoMinProgress"]
        for item in low_diag_source
        if item.get("ClipRateLowAutoMinProgress") is not None
    ]
    freeze_progress_values = [
        item["ClipRateLowAutoFreezeProgress"]
        for item in low_diag_source
        if item.get("ClipRateLowAutoFreezeProgress") is not None
    ]
    progress_rows = [item for item in low_diag_source if item.get("TrainProgress") is not None]
    min_progress_value = min_progress_values[-1] if min_progress_values else None
    freeze_progress_value = freeze_progress_values[-1] if freeze_progress_values else None
    min_progress_step = None
    freeze_progress_step = None
    if min_progress_value is not None:
        for item in progress_rows:
            if item.get("TrainProgress") is not None and item["TrainProgress"] >= min_progress_value:
                min_progress_step = item.get("TrainStep")
                break
    if freeze_progress_value is not None:
        for item in progress_rows:
            if item.get("TrainProgress") is not None and item["TrainProgress"] >= freeze_progress_value:
                freeze_progress_step = item.get("TrainStep")
                break

    clip_share_values = [item["ClipShare"] for item in parsed_rows if item.get("ClipShare") is not None]
    round_share_values = [item["RoundShare"] for item in parsed_rows if item.get("RoundShare") is not None]
    clip_err_ratio_values = [item["ClipErrRatio"] for item in parsed_rows if item.get("ClipErrRatio") is not None]
    round_err_ratio_values = [item["RoundErrRatio"] for item in parsed_rows if item.get("RoundErrRatio") is not None]

    summary = {
        "rows": len(parsed_rows),
        "auto_rows": len(auto_rows),
        "bit_switches": bit_switches,
        "bits_unique": sorted(list({int(v) for v in bits_values})) if bits_values else [],
        "auto_applied_count": auto_applied_count,
        "in_band_ratio": safe_ratio(float(in_band_count), float(non_warmup_count)) if non_warmup_count else None,
        "auto_reason_counts": dict(auto_reason_counter),
        "auto_clip_target_median": statistics.median(clip_target_values) if clip_target_values else None,
        "clip_ema_cv": clip_ema_cv,
        "final_clip_rate_ema": latest.get("ClipRateEMA"),
        "final_quant_err_ratio_ema": latest.get("QuantErrRatioEMA"),
        "final_quant_err_rms_ema": latest.get("QuantErrRMSEMA"),
        "final_qerr_per_clip": final_qerr_per_clip,
        "max_qerr_per_clip": qerr_max,
        "tail_mean_qerr_per_clip": qerr_tail_mean,
        "qerr_per_clip_threshold": DQ_LOW_QERR_PER_CLIP_THRESHOLD,
        "run_qerr_ratio_threshold": run_qerr_ratio_thresholds[-1] if run_qerr_ratio_thresholds else None,
        "run_qerr_per_clip_threshold": run_qerr_per_clip_thresholds[-1] if run_qerr_per_clip_thresholds else None,
        "low_band_seen": low_band_seen,
        "low_auto_bad_count": low_auto_bad_count if low_auto_bad_values else None,
        "low_auto_bad_ratio": low_auto_bad_ratio,
        "low_auto_max_bad_streak": max(low_auto_streak_values) if low_auto_streak_values else None,
        "low_auto_escaped": low_auto_escaped if low_auto_bad_values or low_auto_states or low_auto_decisions else None,
        "low_auto_frozen_bad_count": low_auto_frozen_bad_count if low_auto_bad_values else None,
        "low_auto_state_counts": dict(Counter(value for value in low_auto_states if value)),
        "low_auto_decision_counts": dict(Counter(value for value in low_auto_decisions if value)),
        "low_auto_reason_counts": dict(Counter(value for value in low_auto_reasons if value)),
        "low_auto_phase_counts": dict(Counter(value for value in low_auto_phases if value)),
        "low_auto_min_progress": min_progress_value,
        "low_auto_freeze_progress": freeze_progress_value,
        "low_auto_min_progress_step": min_progress_step,
        "low_auto_freeze_progress_step": freeze_progress_step,
        "active_clip_band_final": active_clip_band_final,
        "active_clip_band_counts": active_band_counts,
        "clip_share_tail_mean": mean_tail(clip_share_values, 0.25, True) if clip_share_values else None,
        "round_share_tail_mean": mean_tail(round_share_values, 0.25, True) if round_share_values else None,
        "clip_err_ratio_tail_mean": mean_tail(clip_err_ratio_values, 0.25, True) if clip_err_ratio_values else None,
        "round_err_ratio_tail_mean": mean_tail(round_err_ratio_values, 0.25, True) if round_err_ratio_values else None,
        "final_zero_rate": latest.get("ZeroRate"),
    }

    return {
        "path": dq_log_path,
        "auto_path": dq_auto_path,
        "rows": parsed_rows,
        "auto_rows": auto_rows,
        "markers": markers,
        "summary": summary,
    }


def parse_rank_logs(
    rank_log_path: str,
    grad_data: Optional[Dict[str, Any]],
    module_param_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    rank_lr_keys = [
        "UnetLRMin",
        "UnetLRMax",
        "Te1LRMin",
        "Te1LRMax",
        "Te2LRMin",
        "Te2LRMax",
    ]
    empty_grouped = {
        "path": {"labels": [], "rows": []},
        "role": {"labels": [], "rows": []},
    }
    empty_final_heatmaps = {}
    rows = read_csv_rows(rank_log_path)
    if not rows:
        return {
            "path": rank_log_path,
            "rows": [],
            "markers": [],
            "grouped": empty_grouped,
            "final_heatmaps": empty_final_heatmaps,
            "summary": {
                "rows": 0,
                "final_rank_sat_p95": None,
                "final_rank_energy_sum": None,
            },
        }

    def _quantile(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        if len(values) == 1:
            return float(values[0])
        sorted_values = sorted(values)
        position = (len(sorted_values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(sorted_values[lower])
        weight_upper = position - lower
        weight_lower = 1.0 - weight_upper
        return float(sorted_values[lower] * weight_lower + sorted_values[upper] * weight_upper)

    def _classify_path_group(module_name: str) -> str:
        lowered = (module_name or "").lower()
        if "down_blocks_" in lowered or "input_blocks_" in lowered:
            return "down"
        if "mid_block_" in lowered or "middle_block_" in lowered:
            return "mid"
        if "up_blocks_" in lowered or "output_blocks_" in lowered:
            return "up"
        return "other"

    def _classify_role_group(module_name: str) -> str:
        lowered = (module_name or "").lower()
        if "_to_q" in lowered:
            return "to_q"
        if "_to_k" in lowered:
            return "to_k"
        if "_to_v" in lowered:
            return "to_v"
        if "_to_out" in lowered:
            return "to_out"
        if "ff_net" in lowered or "_ff_" in lowered:
            return "ff"
        if "resnets_" in lowered:
            return "resnet"
        if "upsamplers_" in lowered or "downsamplers_" in lowered:
            return "sampler"
        if "conv" in lowered:
            return "conv"
        return "other"

    def _collapse_heatmap_role_label(role_label: str) -> str:
        if role_label in {"to_q", "to_k", "to_v", "to_out", "ff"}:
            return role_label
        return "other"

    def _order_group_labels(group_name: str, labels: List[str]) -> List[str]:
        preferred_orders = {
            "path": ["down", "mid", "up", "other"],
            "role": ["to_q", "to_k", "to_v", "to_out", "ff", "resnet", "sampler", "conv", "other"],
        }
        ordered = [label for label in preferred_orders.get(group_name, []) if label in labels]
        ordered.extend(sorted(label for label in labels if label not in ordered))
        return ordered

    def _finalize_grouped_rows(group_name: str, group_buckets: Dict[Tuple[int, str], Dict[str, Any]], eps: float):
        grouped_rows: List[Dict[str, Any]] = []
        labels = set()
        for bucket in group_buckets.values():
            groups_out: Dict[str, Dict[str, Any]] = {}
            total_energy = sum(info["energy_sum"] for info in bucket["groups"].values())
            total_energy_per_param = 0.0
            for info in bucket["groups"].values():
                param_count = info.get("param_count_sum")
                if param_count and param_count > 0:
                    total_energy_per_param += info["energy_sum"] / param_count
            for label, info in bucket["groups"].items():
                labels.add(label)
                sat_wmean = None
                if info["sat_weighted_energy"] > eps:
                    sat_wmean = info["sat_weighted_sum"] / info["sat_weighted_energy"]
                elif info["sat_values"]:
                    sat_wmean = statistics.mean(info["sat_values"])
                energy_share = None
                if total_energy > eps:
                    energy_share = info["energy_sum"] / total_energy
                energy_per_param = None
                energy_share_per_param = None
                param_count = info.get("param_count_sum")
                if param_count and param_count > 0:
                    energy_per_param = info["energy_sum"] / param_count
                    if total_energy_per_param > eps:
                        energy_share_per_param = energy_per_param / total_energy_per_param
                groups_out[label] = {
                    "RankEnergySum": info["energy_sum"],
                    "RankEnergyShare": energy_share,
                    "RankEnergyPerParam": energy_per_param,
                    "RankEnergySharePerParam": energy_share_per_param,
                    "RankSatWMean": sat_wmean,
                    "ParamCount": param_count,
                }

            row_out = {
                "TrainStep": bucket["TrainStep"],
                "Epoch": bucket.get("Epoch"),
                "Scope": bucket["Scope"],
                "groups": groups_out,
            }
            for lr_key in rank_lr_keys:
                row_out[lr_key] = bucket.get(lr_key)
            grouped_rows.append(row_out)

        grouped_rows.sort(key=lambda item: item["TrainStep"])
        return {
            "labels": _order_group_labels(group_name, list(labels)),
            "rows": grouped_rows,
        }

    first_columns = set(rows[0].keys())
    is_per_module_schema = (
        "RankSatWMean" not in first_columns
        and {"RankSat", "RankTop1", "RankEnergy"}.issubset(first_columns)
    )

    parsed_rows: List[Dict[str, Any]] = []
    grouped = empty_grouped
    final_heatmaps = empty_final_heatmaps

    if is_per_module_schema:
        buckets: Dict[Tuple[int, str], Dict[str, Any]] = {}
        eps = 1e-12
        grouped_buckets = {
            "path": {},
            "role": {},
        }
        final_step_modules: List[Dict[str, Any]] = []
        for row in rows:
            train_step = safe_int(row.get("TrainStep"))
            if train_step is None:
                continue
            epoch = safe_int(row.get("Epoch"))
            if epoch is None:
                epoch = infer_epoch_from_step(train_step, grad_data)
            scope = (row.get("Scope") or "").strip()
            key = (train_step, scope)
            if key not in buckets:
                buckets[key] = {
                    "TrainStep": train_step,
                    "Epoch": epoch,
                    "Scope": scope,
                    "rank_dims": [],
                    "sat_values": [],
                    "top1_values": [],
                    "sat_weighted": [],
                    "top1_weighted": [],
                    "energy_values": [],
                }
                for lr_key in rank_lr_keys:
                    buckets[key][lr_key] = None
            bucket = buckets[key]
            if bucket.get("Epoch") is None and epoch is not None:
                bucket["Epoch"] = epoch

            lr_values = {lr_key: safe_float(row.get(lr_key)) for lr_key in rank_lr_keys}
            for lr_key, lr_value in lr_values.items():
                if bucket.get(lr_key) is None and lr_value is not None:
                    bucket[lr_key] = lr_value

            rank_dim = safe_float(row.get("RankDim"))
            sat = safe_float(row.get("RankSat"))
            top1 = safe_float(row.get("RankTop1"))
            energy = safe_float(row.get("RankEnergy"))
            module_name = (row.get("Module") or "").strip()
            module_param_count = module_param_counts.get(module_name) if module_param_counts is not None else None

            if rank_dim is not None:
                bucket["rank_dims"].append(rank_dim)
            if sat is not None:
                bucket["sat_values"].append(sat)
            if top1 is not None:
                bucket["top1_values"].append(top1)
            if energy is not None:
                bucket["energy_values"].append(energy)
            if sat is not None and energy is not None:
                bucket["sat_weighted"].append((sat, energy))
            if top1 is not None and energy is not None:
                bucket["top1_weighted"].append((top1, energy))

            group_names = {
                "path": _classify_path_group(module_name),
                "role": _classify_role_group(module_name),
            }
            for grouping_name, label in group_names.items():
                grouping_buckets = grouped_buckets[grouping_name]
                if key not in grouping_buckets:
                    grouping_buckets[key] = {
                        "TrainStep": train_step,
                        "Epoch": epoch,
                        "Scope": scope,
                        "groups": {},
                    }
                    for lr_key in rank_lr_keys:
                        grouping_buckets[key][lr_key] = None
                grouped_bucket = grouping_buckets[key]
                if grouped_bucket.get("Epoch") is None and epoch is not None:
                    grouped_bucket["Epoch"] = epoch
                for lr_key, lr_value in lr_values.items():
                    if grouped_bucket.get(lr_key) is None and lr_value is not None:
                        grouped_bucket[lr_key] = lr_value
                if label not in grouped_bucket["groups"]:
                    grouped_bucket["groups"][label] = {
                        "energy_sum": 0.0,
                        "sat_weighted_sum": 0.0,
                        "sat_weighted_energy": 0.0,
                        "sat_values": [],
                        "param_count_sum": 0,
                    }
                info = grouped_bucket["groups"][label]
                if energy is not None:
                    info["energy_sum"] += energy
                if module_param_count is not None:
                    info["param_count_sum"] += int(module_param_count)
                if sat is not None:
                    if energy is not None and energy > eps:
                        info["sat_weighted_sum"] += sat * energy
                        info["sat_weighted_energy"] += energy
                    else:
                        info["sat_values"].append(sat)

            final_step_modules.append(
                {
                    "TrainStep": train_step,
                    "Epoch": epoch,
                    "Scope": scope,
                    "Module": module_name,
                    "PathGroup": group_names["path"],
                    "RoleGroup": _collapse_heatmap_role_label(group_names["role"]),
                    "RankSat": sat,
                    "RankEnergy": energy,
                    "ParamCount": module_param_count,
                }
            )

        for bucket in buckets.values():
            energy_sum = sum(bucket["energy_values"]) if bucket["energy_values"] else None
            active_sat_values = [sat for sat, energy in bucket["sat_weighted"] if energy > eps]
            active_top1_values = [top1 for top1, energy in bucket["top1_weighted"] if energy > eps]
            sat_values = active_sat_values if active_sat_values else bucket["sat_values"]
            top1_values = active_top1_values if active_top1_values else bucket["top1_values"]

            sat_wmean = None
            if bucket["sat_weighted"] and energy_sum is not None:
                if energy_sum > eps:
                    sat_wmean = sum(sat * energy for sat, energy in bucket["sat_weighted"]) / energy_sum
                else:
                    sat_wmean = 0.0

            rank_dim = None
            if bucket["rank_dims"]:
                rank_dim_set = set(bucket["rank_dims"])
                if len(rank_dim_set) == 1:
                    rank_dim = next(iter(rank_dim_set))

            parsed_rows.append(
                {
                    "TrainStep": bucket["TrainStep"],
                    "Epoch": bucket.get("Epoch"),
                    "Scope": bucket["Scope"],
                    "RankDim": rank_dim,
                    "RankSatWMean": sat_wmean,
                    "RankSatP50": _quantile(sat_values, 0.5),
                    "RankSatP95": _quantile(sat_values, 0.95),
                    "RankSatMax": max(sat_values) if sat_values else None,
                    "RankTop1P95": _quantile(top1_values, 0.95),
                    "RankEnergySum": energy_sum,
                }
            )
            for lr_key in rank_lr_keys:
                parsed_rows[-1][lr_key] = bucket.get(lr_key)

        grouped = {
            "path": _finalize_grouped_rows("path", grouped_buckets["path"], eps),
            "role": _finalize_grouped_rows("role", grouped_buckets["role"], eps),
        }

        if final_step_modules:
            final_step = max(item["TrainStep"] for item in final_step_modules)
            final_modules = [item for item in final_step_modules if item["TrainStep"] == final_step]
            heatmap_path_labels = ["down", "mid", "up"]
            heatmap_role_labels = ["to_q", "to_k", "to_v", "to_out", "ff", "other"]
            cell_buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}
            for path_label in heatmap_path_labels:
                for role_label in heatmap_role_labels:
                    cell_buckets[(path_label, role_label)] = {
                        "energy_sum": 0.0,
                        "sat_weighted_sum": 0.0,
                        "sat_weighted_energy": 0.0,
                        "sat_values": [],
                        "param_count_sum": 0,
                    }

            final_epoch = None
            final_scope = None
            for item in final_modules:
                path_label = item.get("PathGroup")
                role_label = item.get("RoleGroup")
                if path_label not in heatmap_path_labels or role_label not in heatmap_role_labels:
                    continue
                if final_epoch is None and item.get("Epoch") is not None:
                    final_epoch = item.get("Epoch")
                if final_scope is None and item.get("Scope"):
                    final_scope = item.get("Scope")
                info = cell_buckets[(path_label, role_label)]
                energy = item.get("RankEnergy")
                sat = item.get("RankSat")
                param_count = item.get("ParamCount")
                if energy is not None:
                    info["energy_sum"] += energy
                if param_count is not None:
                    info["param_count_sum"] += int(param_count)
                if sat is not None:
                    if energy is not None and energy > eps:
                        info["sat_weighted_sum"] += sat * energy
                        info["sat_weighted_energy"] += energy
                    else:
                        info["sat_values"].append(sat)

            total_energy_per_param = 0.0
            for info in cell_buckets.values():
                if info["param_count_sum"] > 0:
                    total_energy_per_param += info["energy_sum"] / info["param_count_sum"]

            def _build_heatmap(metric_key: str) -> Dict[str, Any]:
                rows_out = []
                max_value = 0.0
                for path_label in heatmap_path_labels:
                    cells = []
                    for role_label in heatmap_role_labels:
                        info = cell_buckets[(path_label, role_label)]
                        value = None
                        if metric_key == "RankEnergySharePerParam":
                            if info["param_count_sum"] > 0:
                                energy_per_param = info["energy_sum"] / info["param_count_sum"]
                                if total_energy_per_param > eps:
                                    value = energy_per_param / total_energy_per_param
                        elif metric_key == "RankSatWMean":
                            if info["sat_weighted_energy"] > eps:
                                value = info["sat_weighted_sum"] / info["sat_weighted_energy"]
                            elif info["sat_values"]:
                                value = statistics.mean(info["sat_values"])
                        if value is not None:
                            max_value = max(max_value, value)
                        cells.append(
                            {
                                "path": path_label,
                                "role": role_label,
                                "value": value,
                                "RankEnergySum": info["energy_sum"],
                                "ParamCount": info["param_count_sum"] or None,
                            }
                        )
                    rows_out.append({"label": path_label, "cells": cells})
                return {
                    "rows": rows_out,
                    "path_labels": heatmap_path_labels,
                    "role_labels": heatmap_role_labels,
                    "max_value": max_value if max_value > 0 else None,
                    "TrainStep": final_step,
                    "Epoch": final_epoch,
                    "Scope": final_scope,
                }

            final_heatmaps = {
                "energy_share_per_param": _build_heatmap("RankEnergySharePerParam"),
                "rank_sat_wmean": _build_heatmap("RankSatWMean"),
            }
    else:
        for row in rows:
            train_step = safe_int(row.get("TrainStep"))
            if train_step is None:
                continue
            epoch = safe_int(row.get("Epoch"))
            if epoch is None:
                epoch = infer_epoch_from_step(train_step, grad_data)
            parsed_rows.append(
                {
                    "TrainStep": train_step,
                    "Epoch": epoch,
                    "Scope": (row.get("Scope") or "").strip(),
                    "RankDim": safe_float(row.get("RankDim")),
                    "RankSatWMean": safe_float(row.get("RankSatWMean")),
                    "RankSatP50": safe_float(row.get("RankSatP50")),
                    "RankSatP95": safe_float(row.get("RankSatP95")),
                    "RankSatMax": safe_float(row.get("RankSatMax")),
                    "RankTop1P95": safe_float(row.get("RankTop1P95")),
                    "RankEnergySum": safe_float(row.get("RankEnergySum")),
                }
            )
            for lr_key in rank_lr_keys:
                parsed_rows[-1][lr_key] = safe_float(row.get(lr_key))
    parsed_rows.sort(key=lambda item: item["TrainStep"])

    markers: List[Dict[str, Any]] = []
    last_epoch: Optional[int] = None
    for item in parsed_rows:
        epoch = item.get("Epoch")
        if epoch is None:
            continue
        if last_epoch != epoch:
            markers.append({"x": item["TrainStep"], "label": f"E{epoch}"})
            last_epoch = epoch

    latest = parsed_rows[-1] if parsed_rows else {}
    summary = {
        "rows": len(parsed_rows),
        "final_rank_sat_p95": latest.get("RankSatP95"),
        "final_rank_energy_sum": latest.get("RankEnergySum"),
    }

    return {
        "path": rank_log_path,
        "rows": parsed_rows,
        "markers": markers,
        "grouped": grouped,
        "final_heatmaps": final_heatmaps,
        "summary": summary,
    }


def parse_group_loss_logs(step_log_path: Optional[str], epoch_log_path: Optional[str]) -> Optional[Dict[str, Any]]:
    step_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []
    step_markers: List[Dict[str, Any]] = []

    if step_log_path and os.path.exists(step_log_path):
        for row in read_csv_rows(step_log_path):
            global_step = safe_int(row.get("global_step"))
            if global_step is None:
                continue
            step_rows.append(
                {
                    "global_step": global_step,
                    "epoch": safe_int(row.get("epoch")),
                    "group": (row.get("group") or "").strip() or "other",
                    "loss": safe_float(row.get("loss")),
                    "ema_loss_group": safe_float(row.get("ema_loss_group")),
                    "count_group": safe_int(row.get("count_group")),
                }
            )
        step_rows.sort(key=lambda item: item["global_step"])

        last_epoch: Optional[int] = None
        for item in step_rows:
            epoch = item.get("epoch")
            if epoch is None or epoch == last_epoch:
                continue
            step_markers.append({"x": item["global_step"], "label": f"E{epoch}"})
            last_epoch = epoch

    if epoch_log_path and os.path.exists(epoch_log_path):
        for row in read_csv_rows(epoch_log_path):
            epoch = safe_int(row.get("epoch"))
            if epoch is None:
                continue
            epoch_rows.append(
                {
                    "epoch": epoch,
                    "group": (row.get("group") or "").strip() or "other",
                    "ema_loss_end": safe_float(row.get("ema_loss_end")),
                    "count_epoch": safe_int(row.get("count_epoch")),
                    "mean_loss_epoch": safe_float(row.get("mean_loss_epoch")),
                }
            )
        epoch_rows.sort(key=lambda item: (item["epoch"], item["group"]))

    if not step_rows and not epoch_rows:
        return None

    return {
        "step_path": step_log_path if step_log_path and os.path.exists(step_log_path) else None,
        "epoch_path": epoch_log_path if epoch_log_path and os.path.exists(epoch_log_path) else None,
        "step_rows": step_rows,
        "epoch_rows": epoch_rows,
        "step_markers": step_markers,
    }


def analyze_lora_checkpoint(model_path: str, bins: int) -> Dict[str, Any]:
    collect_single_analysis, _, _ = import_lora_analysis_tools()
    report = collect_single_analysis(model_path, bins)
    summary = report.get("summary", {}) or {}
    module_summary = report.get("module_summary", []) or []
    unet_block_summary = report.get("unet_block_summary", []) or []
    density_stats = summary.get("density", {}) or {}
    rms_stats = summary.get("rms", {}) or {}
    entropy_stats = summary.get("entropy_norm", {}) or {}
    sparsity_stats = summary.get("sparsity", {}) or {}

    module_density_medians = [
        item.get("density", {}).get("median")
        for item in module_summary
        if isinstance(item.get("density", {}).get("median"), (float, int))
    ]
    module_balance_ratio = None
    if module_density_medians:
        minimum = min(module_density_medians)
        maximum = max(module_density_medians)
        if minimum > 0:
            module_balance_ratio = maximum / minimum

    return {
        "path": model_path,
        "summary": summary,
        "summary_cards": {
            "total_blocks": summary.get("total_blocks"),
            "total_params": summary.get("total_params"),
            "density_median": density_stats.get("median"),
            "rms_median": rms_stats.get("median"),
            "entropy_median": entropy_stats.get("median"),
            "sparsity_median": sparsity_stats.get("median"),
        },
        "module_summary": module_summary,
        "unet_block_summary": unet_block_summary,
        "diagnostic": {
            "module_balance_ratio": module_balance_ratio,
            "unet_block_count": len(unet_block_summary),
            "density_min": density_stats.get("min"),
            "density_max": density_stats.get("max"),
        },
    }


def analyze_lora_epoch_trend(model_path: str, bins: int) -> Dict[str, Any]:
    collect_single_analysis, find_checkpoint_series, build_checkpoint_history = import_lora_analysis_tools()

    checkpoint_series = find_checkpoint_series(model_path)
    series_data: List[Tuple[int, str, bool, Dict[str, Any]]] = []
    total = len(checkpoint_series)
    print(f"[lora_epoch_trend] checkpoint series found: {total}", flush=True)
    for idx, (epoch, path, is_final) in enumerate(checkpoint_series, start=1):
        marker = " [final]" if is_final else ""
        print(
            f"[lora_epoch_trend] analyzing {idx}/{total}: epoch={epoch}{marker} file={os.path.basename(path)}",
            flush=True,
        )
        report = collect_single_analysis(path, bins)
        series_data.append((epoch, path, is_final, report))
    print("[lora_epoch_trend] checkpoint analysis done", flush=True)

    history = build_checkpoint_history(series_data)
    module_series = (history.get("series") or {}) if isinstance(history, dict) else {}

    trend_modules: List[Dict[str, Any]] = []
    for module in ("unet", "te1", "te2"):
        entries = module_series.get(module) or []
        entry_map = {str(item.get("label") or ""): item for item in entries}

        if module == "te2":
            selected: List[Dict[str, Any]] = []
            for layer_idx in range(32):
                key = f"layer_{layer_idx:02d}"
                if key in entry_map:
                    selected.append(entry_map[key])
                else:
                    selected.append(
                        {
                            "label": key,
                            "display_label": f"Layer {layer_idx:02d}",
                            "values": [],
                            "start_density": None,
                            "end_density": None,
                            "delta_density": None,
                        }
                    )
        else:
            sorted_entries = sorted(entries, key=lambda item: lora_trend_sort_key(module, item))
            selected = sorted_entries[: max(1, DEFAULT_LORA_EPOCH_TREND_MAX_SERIES)] if sorted_entries else []

        colors = [color_for_index(i, len(selected)) for i in range(len(selected))]
        series_items: List[Dict[str, Any]] = []
        legend_rows: List[Dict[str, Any]] = []
        for idx, entry in enumerate(selected):
            values = entry.get("values") or []
            x = [point.get("epoch") for point in values]
            y = [point.get("density_mean") for point in values]
            label = entry.get("display_label") or entry.get("label") or f"{module}_{idx + 1}"
            color = colors[idx]
            series_items.append({"name": label, "color": color, "x": x, "y": y})
            legend_rows.append(
                {
                    "color": color,
                    "name": label,
                    "start_density": entry.get("start_density"),
                    "end_density": entry.get("end_density"),
                    "delta_density": entry.get("delta_density"),
                }
            )

        trend_modules.append(
            {
                "module": module,
                "module_label": module_label(module),
                "total_series": len(entries),
                "selected_series": len(selected),
                "series_limit": DEFAULT_LORA_EPOCH_TREND_MAX_SERIES,
                "legend_rows": legend_rows,
                "series": series_items,
            }
        )
        print(
            f"[lora_epoch_trend] prepared module={module} series={len(series_items)} "
            f"(available={len(entries)})",
            flush=True,
        )

    files = history.get("files") if isinstance(history, dict) else []
    print("[lora_epoch_trend] trend payload ready", flush=True)
    return {
        "model_path": model_path,
        "checkpoints": files,
        "modules": trend_modules,
    }


def grade_metric(value: Optional[float], good_max: float, warn_max: float) -> Tuple[str, str]:
    if value is None:
        return "info", "データ不足"
    if value <= good_max:
        return "good", "良好"
    if value <= warn_max:
        return "warn", "注意"
    return "bad", "要改善"


def build_diagnostics(
    grad_data: Optional[Dict[str, Any]],
    dq_data: Optional[Dict[str, Any]],
    rank_data: Optional[Dict[str, Any]],
    lora_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []

    if grad_data:
        grad_summary = grad_data.get("summary", {})
        spike_ratio = grad_summary.get("threshold_exceeded_ratio")
        level, text = grade_metric(spike_ratio, good_max=0.01, warn_max=0.03)
        checks.append(
            {
                "section": "GradNorm",
                "name": "しきい値超過率",
                "value": fmt_percent(spike_ratio),
                "status": level,
                "note": text + " (超過が多いと更新スキップ増加の可能性)",
            }
        )

        thresh_off_ratio = grad_summary.get("thresh_off_ratio")
        level, text = grade_metric(thresh_off_ratio, good_max=0.01, warn_max=0.05)
        checks.append(
            {
                "section": "GradNorm",
                "name": "ThreshOff発生率",
                "value": fmt_percent(thresh_off_ratio),
                "status": level,
                "note": text + " (高い場合はしきい値判定が無効化されがち)",
            }
        )

        loss_drop = grad_summary.get("loss_ma_drop_ratio")
        if loss_drop is None:
            status = "info"
            note = "データ不足"
        elif loss_drop >= 0.10:
            status = "good"
            note = "良好 (Loss移動平均が十分低下)"
        elif loss_drop >= 0.0:
            status = "warn"
            note = "注意 (Loss低下が弱い)"
        else:
            status = "bad"
            note = "要改善 (Loss移動平均が上昇)"
        checks.append(
            {
                "section": "GradNorm",
                "name": "Loss移動平均の低下率",
                "value": fmt_percent(loss_drop),
                "status": status,
                "note": note,
            }
        )

    if dq_data:
        dq_summary = dq_data.get("summary", {})
        in_band_ratio = dq_summary.get("in_band_ratio")
        if in_band_ratio is None:
            status = "info"
            note = "autoログ不足"
        elif in_band_ratio >= 0.70:
            status = "good"
            note = "良好 (auto判定が帯域内で安定)"
        elif in_band_ratio >= 0.40:
            status = "warn"
            note = "注意 (帯域外判定がやや多い)"
        else:
            status = "bad"
            note = "要改善 (帯域外判定が多い)"
        checks.append(
            {
                "section": "DQ",
                "name": "Auto in-band比率",
                "value": fmt_percent(in_band_ratio),
                "status": status,
                "note": note,
            }
        )

        quant_err_ratio = dq_summary.get("final_quant_err_ratio_ema")
        level, text = grade_metric(quant_err_ratio, good_max=0.35, warn_max=0.50)
        checks.append(
            {
                "section": "DQ",
                "name": "最終 QuantErrRatioEMA",
                "value": fmt_float(quant_err_ratio, 4),
                "status": level,
                "note": text + " (高いほど量子化誤差が強い)",
            }
        )

        zero_rate = dq_summary.get("final_zero_rate")
        level, text = grade_metric(zero_rate, good_max=0.05, warn_max=0.10)
        checks.append(
            {
                "section": "DQ",
                "name": "最終 ZeroRate",
                "value": fmt_percent(zero_rate),
                "status": level,
                "note": text + " (高いほど情報が潰れやすい)",
            }
        )

        clip_cv = dq_summary.get("clip_ema_cv")
        if clip_cv is None:
            status = "info"
            note = "データ不足"
        elif clip_cv <= 0.25:
            status = "good"
            note = "良好 (ClipRateEMA変動が小さい)"
        elif clip_cv <= 0.50:
            status = "warn"
            note = "注意 (ClipRateEMAに揺れがある)"
        else:
            status = "bad"
            note = "要改善 (ClipRateEMAの揺れが大きい)"
        checks.append(
            {
                "section": "DQ",
                "name": "ClipRateEMAの変動係数",
                "value": fmt_float(clip_cv, 4),
                "status": status,
                "note": note,
            }
        )

        if dq_summary.get("low_band_seen"):
            qerr_per_clip_tail = dq_summary.get("tail_mean_qerr_per_clip")
            qerr_per_clip_max = dq_summary.get("max_qerr_per_clip")
            max_bad_streak = dq_summary.get("low_auto_max_bad_streak")
            frozen_bad_count = dq_summary.get("low_auto_frozen_bad_count")
            escaped = dq_summary.get("low_auto_escaped")
            bad_count = dq_summary.get("low_auto_bad_count")
            run_qerr_threshold = dq_summary.get("run_qerr_per_clip_threshold")
            run_threshold_note = ""
            if run_qerr_threshold is not None and abs(run_qerr_threshold - DQ_LOW_QERR_PER_CLIP_THRESHOLD) > 1e-9:
                run_threshold_note = f" run指定閾値は{run_qerr_threshold:g}。"
            if escaped or (frozen_bad_count is not None and frozen_bad_count > 0):
                status = "bad"
                note = "要改善 (low帯では誤差負荷が高い可能性。clip_rate_midまたはclip_rate_highでの再実行を推奨)"
            elif qerr_per_clip_tail is not None and qerr_per_clip_tail >= DQ_LOW_QERR_PER_CLIP_THRESHOLD:
                status = "bad"
                note = "要改善 (QErrPerClipが固定診断閾値130以上で継続)"
            elif (
                (qerr_per_clip_max is not None and qerr_per_clip_max >= DQ_LOW_QERR_PER_CLIP_THRESHOLD)
                or (max_bad_streak is not None and max_bad_streak >= 2)
                or (bad_count is not None and bad_count > 0)
            ):
                status = "warn"
                note = "注意 (low帯で一時的に誤差負荷が高い。mid/highとの比較候補)"
            elif qerr_per_clip_tail is None:
                status = "info"
                note = "データ不足 (QErrPerClip列が無いためlow適性は判定不可)"
            else:
                status = "good"
                note = "良好 (low clip帯でもQErrPerClipは安定)"
            checks.append(
                {
                    "section": "DQ",
                    "name": "clip_rate_low適性",
                    "value": fmt_float(qerr_per_clip_tail, 2),
                    "status": status,
                    "note": note + run_threshold_note,
                }
            )

            if frozen_bad_count is not None and frozen_bad_count > 0:
                checks.append(
                    {
                        "section": "DQ",
                        "name": "low_auto凍結後bad",
                        "value": str(frozen_bad_count),
                        "status": "warn",
                        "note": "freeze_progress後にbadが出ています。この実行ではmidへ逃げませんが、次回はlow以外の比較を推奨します。",
                    }
                )

    if rank_data:
        rank_summary = rank_data.get("summary", {})
        rank_sat_p95 = rank_summary.get("final_rank_sat_p95")
        level, text = grade_metric(rank_sat_p95, good_max=0.90, warn_max=0.97)
        checks.append(
            {
                "section": "Rank",
                "name": "最終 RankSatP95",
                "value": fmt_float(rank_sat_p95, 4),
                "status": level,
                "note": text + " (高止まりはrank飽和の兆候)",
            }
        )

    if lora_data:
        module_balance = lora_data.get("diagnostic", {}).get("module_balance_ratio")
        if module_balance is None:
            status = "info"
            note = "データ不足"
        elif module_balance <= 2.0:
            status = "good"
            note = "良好 (モジュール間密度の偏りが小さい)"
        elif module_balance <= 3.0:
            status = "warn"
            note = "注意 (モジュール間密度の偏りがやや大きい)"
        else:
            status = "bad"
            note = "要改善 (モジュール間密度の偏りが大きい)"
        checks.append(
            {
                "section": "LoRA",
                "name": "モジュール密度バランス比 (max/min)",
                "value": fmt_float(module_balance, 4),
                "status": status,
                "note": note,
            }
        )

    if not checks:
        return {
            "score": None,
            "overall_status": "情報不足",
            "overall_class": "info",
            "checks": [
                {
                    "section": "診断",
                    "name": "スコア算出可否",
                    "value": "-",
                    "status": "info",
                    "note": "Grad/DQ/Rank/LoRA の診断対象データが不足しているため総合スコアを算出できません",
                }
            ],
        }

    score = 100
    for check in checks:
        if check["status"] == "warn":
            score -= 10
        elif check["status"] == "bad":
            score -= 20
    score = max(0, score)
    if score >= 80:
        overall_status = "良好"
        overall_class = "good"
    elif score >= 60:
        overall_status = "注意"
        overall_class = "warn"
    else:
        overall_status = "要改善"
        overall_class = "bad"

    return {
        "score": score,
        "overall_status": overall_status,
        "overall_class": overall_class,
        "checks": checks,
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_json(v) for v in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    return value


def build_chart_payload(
    grad_data: Optional[Dict[str, Any]],
    dq_data: Optional[Dict[str, Any]],
    rank_data: Optional[Dict[str, Any]],
    group_loss_data: Optional[Dict[str, Any]],
    lora_trend_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"grad": [], "dq": [], "rank": [], "group_loss": [], "lora_trend": []}

    if grad_data:
        x = grad_data.get("x", [])
        markers = grad_data.get("markers", [])
        payload["grad"] = [
            {
                "id": "grad_norm",
                "title": "Gradient Norm / Threshold",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": [
                    {"name": "Gradient Norm", "color": ColorPalette[0], "y": grad_data.get("gradient_norm", [])},
                    {"name": "Threshold", "color": ColorPalette[2], "y": grad_data.get("threshold", [])},
                ],
            },
            {
                "id": "loss_ma",
                "title": "Loss (raw / moving average)",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": [
                    {"name": "Loss", "color": "#94a3b8", "y": grad_data.get("loss", [])},
                    {"name": "Loss MA", "color": ColorPalette[1], "y": grad_data.get("loss_ma", [])},
                ],
            },
            {
                "id": "thresh_off",
                "title": "ThreshOff",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 1.0,
                "y_tick_integer": True,
                "y_min_fixed": 0.0,
                "y_max_fixed": 2.0,
                "series": [{"name": "ThreshOff", "color": ColorPalette[3], "y": grad_data.get("thresh_off", [])}],
            },
            {
                "id": "scale",
                "title": "GradScaler Scale",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": [{"name": "Scale", "color": ColorPalette[4], "y": grad_data.get("scale", [])}],
            },
            {
                "id": "cosine",
                "title": "Grad Cosine Similarity",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "series": [{"name": "CosineSim", "color": ColorPalette[5], "y": grad_data.get("cosine", [])}],
            },
        ]

    if dq_data:
        log_rows = dq_data.get("rows", [])
        auto_rows = dq_data.get("auto_rows", [])
        rows = log_rows if log_rows else auto_rows
        auto_has_qerr_per_clip = any(item.get("QErrPerClip") is not None for item in auto_rows)
        x = [item.get("TrainStep") for item in rows]
        markers = list(dq_data.get("markers", []))
        dq_summary = dq_data.get("summary", {})
        if dq_summary.get("low_auto_min_progress_step") is not None:
            markers.append({"x": dq_summary.get("low_auto_min_progress_step"), "label": "min_progress"})
        if dq_summary.get("low_auto_freeze_progress_step") is not None:
            markers.append({"x": dq_summary.get("low_auto_freeze_progress_step"), "label": "freeze_progress"})

        def series_for(keys: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
            output = []
            for field, label, color in keys:
                output.append({"name": label, "color": color, "y": [item.get(field) for item in rows]})
            return output

        def series_for_source(source_rows: List[Dict[str, Any]], keys: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
            output = []
            for field, label, color in keys:
                output.append({"name": label, "color": color, "y": [item.get(field) for item in source_rows]})
            return output

        def has_series_value(field: str) -> bool:
            return any(item.get(field) is not None for item in rows)

        def has_source_series_value(source_rows: List[Dict[str, Any]], field: str) -> bool:
            return any(item.get(field) is not None for item in source_rows)

        payload["dq"] = []

        if has_series_value("Bits"):
            payload["dq"].append(
                {
                    "id": "dq_bits",
                    "title": "Bits",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_tick_step": 1.0,
                    "y_tick_integer": True,
                    "y_min_floor": 0.0,
                    "series": series_for([("Bits", "Bits", ColorPalette[0])]),
                }
            )

        if has_series_value("RangeMul"):
            payload["dq"].append(
                {
                    "id": "dq_range_mul",
                    "title": "RangeMul",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_tick_step": 0.1,
                    "y_tick_precision": 1,
                    "series": series_for([("RangeMul", "RangeMul", ColorPalette[1])]),
                }
            )

        if has_series_value("ClipRateRaw") or has_series_value("ClipRateEMA"):
            payload["dq"].append(
                {
                    "id": "dq_clip",
                    "title": "ClipRateRaw / ClipRateEMA",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_max_fixed": 0.008,
                    "y_tick_step": 0.001,
                    "y_tick_precision": 3,
                    "series": series_for(
                        [
                            ("ClipRateRaw", "ClipRateRaw", ColorPalette[2]),
                            ("ClipRateEMA", "ClipRateEMA", ColorPalette[0]),
                            ("ActiveClipLow", "ActiveClipLow", "#64748b"),
                            ("ActiveClipHigh", "ActiveClipHigh", "#334155"),
                        ]
                    ),
                }
            )

        if has_series_value("QuantErrRatioRaw") or has_series_value("QuantErrRatioEMA"):
            payload["dq"].append(
                {
                    "id": "dq_qerr_ratio",
                    "title": "QuantErrRatioRaw / QuantErrRatioEMA",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_max_fixed": 0.8,
                    "y_tick_step": 0.1,
                    "y_tick_precision": 1,
                    "series": series_for(
                        [
                            ("QuantErrRatioRaw", "QuantErrRatioRaw", ColorPalette[2]),
                            ("QuantErrRatioEMA", "QuantErrRatioEMA", ColorPalette[0]),
                        ]
                    ),
                }
            )

        if has_series_value("QuantErrRMSRaw") or has_series_value("QuantErrRMSEMA"):
            payload["dq"].append(
                {
                    "id": "dq_qerr_rms",
                    "title": "QuantErrRMSRaw / QuantErrRMSEMA",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.05,
                    "y_tick_precision": 2,
                    "series": series_for(
                        [
                            ("QuantErrRMSRaw", "QuantErrRMSRaw", ColorPalette[2]),
                            ("QuantErrRMSEMA", "QuantErrRMSEMA", ColorPalette[0]),
                        ]
                    ),
                }
            )

        if has_series_value("ZeroRate"):
            payload["dq"].append(
                {
                    "id": "dq_zero_rate",
                    "title": "ZeroRate",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.02,
                    "y_tick_precision": 2,
                    "series": series_for([("ZeroRate", "ZeroRate", ColorPalette[3])]),
                }
            )

        if has_series_value("AbsMax"):
            payload["dq"].append(
                {
                    "id": "dq_absmax",
                    "title": "AbsMax",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_max_fixed": 1000.0,
                    "y_tick_nice_integer": True,
                    "y_min_floor": 0.0,
                    "series": series_for([("AbsMax", "AbsMax", ColorPalette[4])]),
                }
            )

        if has_series_value("Range"):
            payload["dq"].append(
                {
                    "id": "dq_range",
                    "title": "Range",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.1,
                    "y_tick_precision": 1,
                    "series": series_for([("Range", "Range", ColorPalette[5])]),
                }
            )

        qerr_chart_rows = auto_rows if auto_has_qerr_per_clip else rows
        qerr_chart_x = [item.get("TrainStep") for item in qerr_chart_rows]
        if has_source_series_value(qerr_chart_rows, "QErrPerClip"):
            run_qerr_threshold = dq_summary.get("run_qerr_per_clip_threshold") or DQ_LOW_QERR_PER_CLIP_THRESHOLD
            qerr_threshold_series = [
                {
                    "name": f"Run threshold {run_qerr_threshold:g}",
                    "color": ColorPalette[2],
                    "y": [run_qerr_threshold if step is not None else None for step in qerr_chart_x],
                }
            ]
            if abs(run_qerr_threshold - DQ_LOW_QERR_PER_CLIP_THRESHOLD) > 1e-9:
                qerr_threshold_series.append(
                    {
                        "name": "Fixed reference 130",
                        "color": "#94a3b8",
                        "y": [DQ_LOW_QERR_PER_CLIP_THRESHOLD if step is not None else None for step in qerr_chart_x],
                    }
                )
            payload["dq"].append(
                {
                    "id": "dq_qerr_per_clip",
                    "title": "QErrPerClip",
                    "subtitle": "QErrPerClip = auto quant-error EMA / max(ClipRateEMA, floor).",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": qerr_chart_x,
                    "y_min_fixed": 0.0,
                    "y_max_ceil": max(run_qerr_threshold, DQ_LOW_QERR_PER_CLIP_THRESHOLD),
                    "y_tick_step": 25.0,
                    "y_tick_precision": 0,
                    "series": series_for_source(qerr_chart_rows, [("QErrPerClip", "QErrPerClip", ColorPalette[0])])
                    + qerr_threshold_series,
                }
            )

        low_auto_chart_rows = (
            rows
            if has_series_value("ClipRateLowAutoBad") or has_series_value("ClipRateLowAutoBadStreak")
            else auto_rows
        )
        low_auto_chart_x = [item.get("TrainStep") for item in low_auto_chart_rows]
        if has_source_series_value(low_auto_chart_rows, "ClipRateLowAutoBad") or has_source_series_value(
            low_auto_chart_rows, "ClipRateLowAutoBadStreak"
        ):
            payload["dq"].append(
                {
                    "id": "dq_low_auto_bad",
                    "title": "clip_rate_low_auto Bad / Streak",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": low_auto_chart_x,
                    "y_min_fixed": 0.0,
                    "y_tick_step": 1.0,
                    "y_tick_integer": True,
                    "series": series_for_source(
                        low_auto_chart_rows,
                        [
                            ("ClipRateLowAutoBad", "Bad", ColorPalette[2]),
                            ("ClipRateLowAutoBadStreak", "BadStreak", ColorPalette[4]),
                        ]
                    ),
                }
            )

        if has_series_value("ClipErrRatio") or has_series_value("RoundErrRatio"):
            payload["dq"].append(
                {
                    "id": "dq_error_part_ratio",
                    "title": "ClipErrRatio / RoundErrRatio",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.05,
                    "y_tick_precision": 2,
                    "series": series_for(
                        [
                            ("ClipErrRatio", "ClipErrRatio", ColorPalette[2]),
                            ("RoundErrRatio", "RoundErrRatio", ColorPalette[0]),
                        ]
                    ),
                }
            )

        if has_series_value("ClipShare") or has_series_value("RoundShare"):
            payload["dq"].append(
                {
                    "id": "dq_error_part_share",
                    "title": "ClipShare / RoundShare",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "y_min_fixed": 0.0,
                    "y_max_fixed": 1.0,
                    "y_tick_step": 0.1,
                    "y_tick_precision": 1,
                    "series": series_for(
                        [
                            ("ClipShare", "ClipShare", ColorPalette[2]),
                            ("RoundShare", "RoundShare", ColorPalette[0]),
                        ]
                    ),
                }
            )

    if rank_data:
        rows = rank_data.get("rows", [])
        x = [item.get("TrainStep") for item in rows]
        markers = rank_data.get("markers", [])
        rank_charts = [
            {
                "id": "rank_dim",
                "title": "RankDim",
                "subtitle": "LoRA rank の設定値",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_step": 1.0,
                "y_tick_integer": True,
                "y_min_floor": 0.0,
                "series": [{"name": "RankDim", "color": ColorPalette[6], "y": [item.get("RankDim") for item in rows]}],
            },
            {
                "id": "rank_sat",
                "title": "RankSatWMean / P50 / P95 / Max / Top1P95",
                "subtitle": "rank の使われ方の広さと偏りの要約",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_min_fixed": 0.0,
                "y_max_fixed": 1.1,
                "y_tick_step": 0.1,
                "y_tick_precision": 1,
                "legend_max_rows": 2,
                "series": [
                    {"name": "RankSatWMean", "color": ColorPalette[0], "y": [item.get("RankSatWMean") for item in rows]},
                    {"name": "RankSatP50", "color": ColorPalette[1], "y": [item.get("RankSatP50") for item in rows]},
                    {"name": "RankSatP95", "color": ColorPalette[2], "y": [item.get("RankSatP95") for item in rows]},
                    {"name": "RankSatMax", "color": ColorPalette[3], "y": [item.get("RankSatMax") for item in rows]},
                    {"name": "RankTop1P95", "color": ColorPalette[4], "y": [item.get("RankTop1P95") for item in rows]},
                ],
            },
            {
                "id": "rank_energy",
                "title": "RankEnergySum",
                "subtitle": "LoRA 重み量の総量",
                "x_label": "TrainStep",
                "markers": markers,
                "x": x,
                "y_tick_nice_integer": True,
                "y_min_fixed": 0.0,
                "series": [{"name": "RankEnergySum", "color": ColorPalette[7], "y": [item.get("RankEnergySum") for item in rows]}],
            },
        ]

        lr_scope_specs = [
            ("Unet", "UnetLRMin", "UnetLRMax"),
            ("Te1", "Te1LRMin", "Te1LRMax"),
            ("Te2", "Te2LRMin", "Te2LRMax"),
        ]
        lr_series = []
        lr_color_index = 0
        for scope_label, min_key, max_key in lr_scope_specs:
            min_values = [item.get(min_key) for item in rows]
            max_values = [item.get(max_key) for item in rows]
            if not any(value is not None for value in min_values + max_values):
                continue
            if min_values == max_values:
                lr_series.append(
                    {
                        "name": f"{scope_label}LR",
                        "color": color_for_index(lr_color_index, 6),
                        "y": max_values,
                    }
                )
                lr_color_index += 1
            else:
                lr_series.append(
                    {
                        "name": f"{scope_label}LRMin",
                        "color": color_for_index(lr_color_index, 6),
                        "y": min_values,
                    }
                )
                lr_color_index += 1
                lr_series.append(
                    {
                        "name": f"{scope_label}LRMax",
                        "color": color_for_index(lr_color_index, 6),
                        "y": max_values,
                    }
                )
                lr_color_index += 1

        if lr_series:
            rank_charts.append(
                {
                    "id": "rank_lr",
                    "title": "Rank Log LR Snapshot",
                    "subtitle": "学習率の推移",
                    "x_label": "TrainStep",
                    "markers": markers,
                    "x": x,
                    "legend_max_rows": 2,
                    "series": lr_series,
                }
            )

        grouped = rank_data.get("grouped") or {}
        group_display_names = {
            "path": {
                "down": "Down",
                "mid": "Mid",
                "up": "Up",
                "other": "Other",
            },
            "role": {
                "to_q": "Q",
                "to_k": "K",
                "to_v": "V",
                "to_out": "Out",
                "ff": "FF",
                "resnet": "ResNet",
                "sampler": "Sampler",
                "conv": "Conv",
                "other": "Other",
            },
        }

        def _append_grouped_rank_chart(
            chart_id: str,
            title: str,
            subtitle: str,
            group_kind: str,
            metric_key: str,
            *,
            y_min_fixed: Optional[float] = None,
            y_max_fixed: Optional[float] = None,
            y_tick_step: Optional[float] = None,
            y_tick_precision: Optional[int] = None,
        ):
            group_data = grouped.get(group_kind) or {}
            group_rows = group_data.get("rows", []) or []
            labels = group_data.get("labels", []) or []
            if not group_rows or not labels:
                return

            group_x = [item.get("TrainStep") for item in group_rows]
            series = []
            for idx, label in enumerate(labels):
                y = [((item.get("groups") or {}).get(label) or {}).get(metric_key) for item in group_rows]
                if not any(value is not None for value in y):
                    continue
                series.append(
                    {
                        "name": group_display_names.get(group_kind, {}).get(label, label),
                        "color": color_for_index(idx, max(1, len(labels))),
                        "y": y,
                    }
                )
            if not series:
                return

            chart = {
                "id": chart_id,
                "title": title,
                "subtitle": subtitle,
                "x_label": "TrainStep",
                "markers": markers,
                "x": group_x,
                "legend_max_rows": 2,
                "series": series,
            }
            if y_min_fixed is not None:
                chart["y_min_fixed"] = y_min_fixed
            if y_max_fixed is not None:
                chart["y_max_fixed"] = y_max_fixed
            if y_tick_step is not None:
                chart["y_tick_step"] = y_tick_step
            if y_tick_precision is not None:
                chart["y_tick_precision"] = y_tick_precision
            rank_charts.append(chart)

        _append_grouped_rank_chart(
            "rank_group_path_energy_share",
            "Path Group Energy Share",
            "ブロック位置ごとの重み量シェア（合計1.0）",
            "path",
            "RankEnergyShare",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_path_energy_share_per_param",
            "Path Group Energy Share Per Param",
            "ブロック位置ごとの正規化重み量シェア（合計1.0）",
            "path",
            "RankEnergySharePerParam",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_path_sat",
            "Path Group RankSatWMean",
            "ブロック位置ごとの rank の使われ方の広さ",
            "path",
            "RankSatWMean",
            y_min_fixed=0.0,
            y_max_fixed=1.1,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_role_energy_share",
            "Role Group Energy Share",
            "役割ごとの重み量シェア（合計1.0）",
            "role",
            "RankEnergyShare",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_role_energy_share_per_param",
            "Role Group Energy Share Per Param",
            "役割ごとの正規化重み量シェア（合計1.0）",
            "role",
            "RankEnergySharePerParam",
            y_min_fixed=0.0,
            y_max_fixed=1.0,
            y_tick_step=0.1,
            y_tick_precision=1,
        )
        _append_grouped_rank_chart(
            "rank_group_role_sat",
            "Role Group RankSatWMean",
            "役割ごとの rank の使われ方の広さ",
            "role",
            "RankSatWMean",
            y_min_fixed=0.0,
            y_max_fixed=1.1,
            y_tick_step=0.1,
            y_tick_precision=1,
        )

        payload["rank"] = rank_charts

    if group_loss_data:
        step_rows = group_loss_data.get("step_rows", []) or []
        epoch_rows = group_loss_data.get("epoch_rows", []) or []
        group_loss_charts: List[Dict[str, Any]] = []

        if step_rows:
            step_points: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": []})
            for item in step_rows:
                x = item.get("global_step")
                y = item.get("ema_loss_group")
                group = item.get("group") or "other"
                if x is None or y is None:
                    continue
                step_points[group]["x"].append(x)
                step_points[group]["y"].append(y)

            step_groups = sorted(step_points.keys())
            step_series = []
            for idx, group in enumerate(step_groups):
                step_series.append(
                    {
                        "name": group,
                        "color": color_for_index(idx, len(step_groups)),
                        "x": step_points[group]["x"],
                        "y": step_points[group]["y"],
                    }
                )

            group_loss_charts.append(
                {
                    "id": "group_loss_step_ema",
                    "title": "Group Loss EMA (step)",
                    "x_label": "Global Step",
                    "markers": group_loss_data.get("step_markers", []),
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.05,
                    "y_tick_precision": 2,
                    "series": step_series,
                }
            )

        if epoch_rows:
            epoch_points: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"x": [], "y": []})
            for item in epoch_rows:
                x = item.get("epoch")
                y = item.get("ema_loss_end")
                group = item.get("group") or "other"
                if x is None or y is None:
                    continue
                epoch_points[group]["x"].append(x)
                epoch_points[group]["y"].append(y)

            epoch_groups = sorted(epoch_points.keys())
            epoch_series = []
            for idx, group in enumerate(epoch_groups):
                epoch_series.append(
                    {
                        "name": group,
                        "color": color_for_index(idx, len(epoch_groups)),
                        "x": epoch_points[group]["x"],
                        "y": epoch_points[group]["y"],
                    }
                )

            group_loss_charts.append(
                {
                    "id": "group_loss_epoch_ema",
                    "title": "Group Loss EMA (epoch summary)",
                    "x_label": "Epoch",
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.05,
                    "y_tick_precision": 2,
                    "series": epoch_series,
                }
            )

        payload["group_loss"] = group_loss_charts

    if lora_trend_data:
        checkpoints = lora_trend_data.get("checkpoints", []) or []
        epoch_markers = []
        for item in checkpoints:
            epoch = item.get("epoch")
            if epoch is None:
                continue
            label = item.get("label") or f"E{epoch}"
            epoch_markers.append({"x": epoch, "label": label})

        module_entries = lora_trend_data.get("modules", []) or []
        trend_charts: List[Dict[str, Any]] = []
        for entry in module_entries:
            module = entry.get("module") or "-"
            series_rows = entry.get("series", []) or []
            chart_series = []
            for row in series_rows:
                chart_series.append(
                    {
                        "name": row.get("name"),
                        "color": row.get("color"),
                        "x": row.get("x", []),
                        "y": row.get("y", []),
                    }
                )
            selected_count = entry.get("selected_series", 0)
            total_count = entry.get("total_series", 0)
            trend_charts.append(
                {
                    "id": f"lora_trend_{module}",
                    "title": f"{entry.get('module_label', module)} 情報密度推移 ({selected_count}/{total_count}系列)",
                    "x_label": "Epoch",
                    "markers": epoch_markers,
                    "x": [item.get("epoch") for item in checkpoints if item.get("epoch") is not None],
                    "y_min_fixed": 0.0,
                    "y_tick_step": 0.1,
                    "y_tick_precision": 1,
                    "series": chart_series,
                    "legend_rows": entry.get("legend_rows", []),
                    "no_inline_legend": True,
                }
            )
        payload["lora_trend"] = trend_charts

    return payload


def render_module_rows(module_items: List[Dict[str, Any]]) -> str:
    if not module_items:
        return "<tr><td colspan='7' class='muted'>データがありません</td></tr>"
    rows: List[str] = []
    for item in module_items:
        density = item.get("density", {}) or {}
        rms = item.get("rms", {}) or {}
        row = (
            f"<tr>"
            f"<td>{module_label(item.get('module', '-'))}</td>"
            f"<td class='num'>{fmt_int(item.get('block_count'))}</td>"
            f"<td class='num'>{fmt_int(item.get('total_params'))}</td>"
            f"<td class='num'>{fmt_float(density.get('mean'))}</td>"
            f"<td class='num'>{fmt_float(density.get('median'))}</td>"
            f"<td class='num'>{fmt_float(rms.get('median'))}</td>"
            f"<td class='num'>{fmt_float((item.get('entropy_norm', {}) or {}).get('median'))}</td>"
            f"</tr>"
        )
        rows.append(row)
    return "".join(rows)


def render_unet_rows(items: List[Dict[str, Any]]) -> str:
    if not items:
        return "<tr><td colspan='6' class='muted'>データがありません</td></tr>"
    rows: List[str] = []
    for item in items:
        density = item.get("density", {}) or {}
        rms = item.get("rms", {}) or {}
        row = (
            f"<tr>"
            f"<td>{format_unet_block_label(item.get('label', '-'))}</td>"
            f"<td class='num'>{fmt_int(item.get('block_count'))}</td>"
            f"<td class='num'>{fmt_int(item.get('total_params'))}</td>"
            f"<td class='num'>{fmt_float(density.get('mean'))}</td>"
            f"<td class='num'>{fmt_float(density.get('median'))}</td>"
            f"<td class='num'>{fmt_float(rms.get('median'))}</td>"
            f"</tr>"
        )
        rows.append(row)
    return "".join(rows)


def status_label(status: str) -> str:
    mapping = {"good": "良好", "warn": "注意", "bad": "要改善", "info": "情報"}
    return mapping.get(status, status)


def render_check_rows(checks: List[Dict[str, str]]) -> str:
    if not checks:
        return "<tr><td colspan='5' class='muted'>診断チェックがありません</td></tr>"
    rows: List[str] = []
    for item in checks:
        status = item.get("status", "info")
        rows.append(
            "<tr>"
            f"<td>{item.get('section', '-')}</td>"
            f"<td>{item.get('name', '-')}</td>"
            f"<td class='num'>{item.get('value', '-')}</td>"
            f"<td><span class='badge {status}'>{status_label(status)}</span></td>"
            f"<td>{item.get('note', '-')}</td>"
            "</tr>"
        )
    return "".join(rows)


def heatmap_display_label(kind: str, label: str) -> str:
    if kind == "path":
        return {"down": "Down", "mid": "Mid", "up": "Up"}.get(label, label)
    if kind == "role":
        return {
            "to_q": "Q",
            "to_k": "K",
            "to_v": "V",
            "to_out": "Out",
            "ff": "FF",
            "other": "Other",
        }.get(label, label)
    return label


def heatmap_cell_style(value: Optional[float], max_value: Optional[float], *, fixed_max: Optional[float] = None) -> str:
    if value is None:
        return "background: #f8fafc; color: #94a3b8;"
    denom = fixed_max if fixed_max is not None else max_value
    if not denom or denom <= 0:
        ratio = 0.0
    else:
        ratio = max(0.0, min(1.0, value / denom))
    hue = 210 - int(22 * ratio)
    sat = 70
    light = 98 - int(34 * ratio)
    text_light = 18 if ratio >= 0.48 else 24
    return f"background: hsl({hue} {sat}% {light}%); color: hsl(217 33% {text_light}%);"


def render_rank_heatmap_card(
    title: str,
    subtitle: str,
    heatmap: Optional[Dict[str, Any]],
    *,
    fixed_max: Optional[float] = None,
) -> str:
    if not heatmap or not (heatmap.get("rows") or []):
        return ""
    role_labels = heatmap.get("role_labels") or []
    rows = heatmap.get("rows") or []
    max_value = heatmap.get("max_value")
    train_step = heatmap.get("TrainStep")
    epoch = heatmap.get("Epoch")
    header_cells = "".join(f"<th class='num'>{heatmap_display_label('role', label)}</th>" for label in role_labels)
    body_rows: List[str] = []
    for row in rows:
        cells_html: List[str] = []
        for cell in row.get("cells") or []:
            value = cell.get("value")
            text = fmt_float(value, 3) if value is not None else "-"
            style = heatmap_cell_style(value, max_value, fixed_max=fixed_max)
            cells_html.append(f"<td class='num heatmap-cell' style='{style}'>{text}</td>")
        body_rows.append(
            "<tr>"
            f"<th>{heatmap_display_label('path', row.get('label', '-'))}</th>"
            f"{''.join(cells_html)}"
            "</tr>"
        )
    caption_bits = []
    if epoch is not None:
        caption_bits.append(f"Epoch {epoch}")
    if train_step is not None:
        caption_bits.append(f"TrainStep {train_step}")
    caption = " / ".join(caption_bits)
    caption_html = f"<p class='sub' style='margin: 0 0 8px 0;'>{caption}</p>" if caption else ""
    return (
        "<div class='heatmap-card'>"
        f"<div class='chart-title'>{title}</div>"
        f"<div class='chart-subtitle'>{subtitle}</div>"
        f"{caption_html}"
        "<table class='heatmap-table'>"
        "<thead><tr><th></th>"
        f"{header_cells}"
        "</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
        "</div>"
    )


def build_html(report: Dict[str, Any]) -> str:
    report_json = json.dumps(sanitize_json(report), ensure_ascii=False).replace("</", "<\\/")
    diagnostics = report.get("diagnostics", {})
    rank_charts = ((report.get("charts") or {}).get("rank") or [])
    rank_chart_ids = {item.get("id") for item in rank_charts if isinstance(item, dict)}
    has_rank_grouped_charts = any(
        chart_id in rank_chart_ids
        for chart_id in (
            "rank_group_path_energy_share",
            "rank_group_path_energy_share_per_param",
            "rank_group_path_sat",
            "rank_group_role_energy_share",
            "rank_group_role_energy_share_per_param",
            "rank_group_role_sat",
        )
    )
    score_value = diagnostics.get("score")
    score_text = "-" if score_value is None else f"{score_value}点"
    lora_data = report.get("lora")
    lora_cards = {}
    module_rows = "<tr><td colspan='7' class='muted'>LoRA解析を実行していません</td></tr>"
    unet_rows = "<tr><td colspan='6' class='muted'>LoRA解析を実行していません</td></tr>"
    if lora_data:
        lora_cards = lora_data.get("summary_cards", {})
        module_rows = render_module_rows(lora_data.get("module_summary", []))
        unet_rows = render_unet_rows(lora_data.get("unet_block_summary", []))
    lora_error = report.get("lora_error")
    lora_error_html = f"<p class='sub' style='color: var(--bad);'>{lora_error}</p>" if lora_error else ""
    lora_trend_error = report.get("lora_trend_error")
    lora_trend_error_html = f"<p class='sub' style='color: var(--bad);'>{lora_trend_error}</p>" if lora_trend_error else ""
    rank_final_heatmaps = ((report.get("rank") or {}).get("final_heatmaps") or {})
    rank_heatmap_cards_html = ""
    if has_rank_grouped_charts:
        rank_heatmap_cards_html = (
            render_rank_heatmap_card(
                "Final Energy Share Per Param Heatmap",
                "最終 step における、位置×役割ごとの正規化重み量シェア（合計1.0）",
                rank_final_heatmaps.get("energy_share_per_param"),
            )
            + render_rank_heatmap_card(
                "Final RankSatWMean Heatmap",
                "最終 step における、位置×役割ごとの rank の使われ方の広さ",
                rank_final_heatmaps.get("rank_sat_wmean"),
                fixed_max=1.0,
            )
        )
    rank_grouped_help_html = ""
    if has_rank_grouped_charts:
        rank_grouped_help_html = """
      <div class="callout" style="margin: 0 0 16px 0;">
        <strong>重み量と rank の広がりの違い</strong>
        <p class="sub" style="margin: 8px 0 0 0;">
          `Energy Share` や `Energy Share Per Param` は、LoRA 重みがどれだけ大きく育っているかを見る指標です。
          一方で `RankSatWMean` は、設定した rank の成分をどれだけ広く使っているかを見る指標です。
        </p>
        <p class="sub" style="margin: 8px 0 0 0;">
          この2つは一致するとは限りません。
          重み量が大きくても、実際には少数の rank 成分に偏っていることがあります。
          逆に、重み量はそれほど大きくなくても、複数の rank 成分を広く使っていることがあります。
        </p>
        <p class="sub" style="margin: 8px 0 0 0;">
          そのため、`重み量` と `rank の広がり` を並べて見ると、
          「どこが強く学習しているか」と「どこで rank を広く使えているか」を分けて確認できます。
        </p>
      </div>
      <div class="callout" style="margin: 0 0 16px 0;">
        <strong>RankSat 要約グラフとの関係</strong>
        <p class="sub" style="margin: 8px 0 0 0;">
          `Path Group RankSatWMean` と `Role Group RankSatWMean` は、上の `RankSatWMean / P50 / P95 / Max / Top1P95` と同じく、
          rank の使われ方の広さを見る指標です。
          上のグラフが全体要約なのに対し、こちらは block 位置別・役割別の内訳を見ます。
        </p>
      </div>
      <div class="callout" style="margin: 12px 0 16px 0;">
        <strong>Path 系列の見方</strong>
        <p class="sub" style="margin: 8px 0 0 0;">
          `Down / Mid / Up` は、UNet のどの位置のブロックかを表します。
          `Down` は入力側で細かい情報を取り込む前半、`Mid` は中央、`Up` は出力側で絵を組み立て直す後半です。
          3つの Path グラフは同じ系列を共有しているので、どの位置が強く学習しているか、サイズ差を補正するとどう見えるか、rank を広く使えているかを並べて読めます。
        </p>
      </div>
      <div class="callout" style="margin: 0 0 16px 0;">
        <strong>Role 系列の見方</strong>
        <p class="sub" style="margin: 8px 0 0 0;">
          `Q / K / V / Out` は Attention の中の役割です。
          `Q` は「何を見たいか」、`K` は「どんな特徴を持つか」、`V` は「実際に取り出して混ぜる中身」、`Out` は混ぜた結果を次へ渡す出口です。
          `FF` は Attention の後ろにある特徴変換用の全結合層、`Other` はそのどちらにも素直に入らない周辺の層です。
          3つの Role グラフは同じ系列を共有しているので、どの役割が主戦場か、サイズ差を除いても強いか、rank の使い方に偏りがあるかを比べてください。
        </p>
      </div>
"""

    return f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>LoRA Diagnostic Report</title>
<style>
:root {{
  --bg: #f4f6fb;
  --card: #ffffff;
  --text: #0f172a;
  --muted: #64748b;
  --line: #d7deea;
  --good: #15803d;
  --good-bg: #dcfce7;
  --warn: #b45309;
  --warn-bg: #ffedd5;
  --bad: #b91c1c;
  --bad-bg: #fee2e2;
  --info: #0f766e;
  --info-bg: #ccfbf1;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: linear-gradient(160deg, #eef2ff 0%, #f8fafc 45%, #f5f5f4 100%);
  color: var(--text);
  font-family: "Segoe UI", "Noto Sans JP", "Hiragino Kaku Gothic ProN", sans-serif;
}}
.container {{
  max-width: 1520px;
  margin: 0 auto;
  padding: 22px;
}}
h1, h2 {{
  margin: 0;
}}
.sub {{
  color: var(--muted);
  margin-top: 8px;
}}
.panel {{
  margin-top: 16px;
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 14px 16px;
}}
.grid {{
  display: grid;
  gap: 12px;
}}
.grid.cards {{
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
}}
.card {{
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 12px;
  background: #fbfdff;
}}
.card .title {{
  color: var(--muted);
  font-size: 12px;
}}
.card .value {{
  margin-top: 4px;
  font-size: 24px;
  font-weight: 700;
}}
.card .caption {{
  margin-top: 4px;
  color: var(--muted);
  font-size: 12px;
}}
.overall {{
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: 700;
}}
.score {{
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 15px;
}}
.score.good {{ background: var(--good-bg); color: var(--good); }}
.score.warn {{ background: var(--warn-bg); color: var(--warn); }}
.score.bad {{ background: var(--bad-bg); color: var(--bad); }}
.score.info {{ background: var(--info-bg); color: var(--info); }}
table {{
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
  font-size: 13px;
}}
th, td {{
  border-bottom: 1px solid var(--line);
  padding: 8px 6px;
  text-align: left;
  vertical-align: top;
}}
th {{
  color: #334155;
  font-weight: 600;
  background: #f8fafc;
}}
td.num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.muted {{ color: var(--muted); }}
.badge {{
  border-radius: 999px;
  display: inline-block;
  padding: 2px 8px;
  font-size: 12px;
  font-weight: 600;
}}
.badge.good {{ color: var(--good); background: var(--good-bg); }}
.badge.warn {{ color: var(--warn); background: var(--warn-bg); }}
.badge.bad {{ color: var(--bad); background: var(--bad-bg); }}
.badge.info {{ color: var(--info); background: var(--info-bg); }}
.chart-grid {{
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
}}
.chart-card {{
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #ffffff;
  padding: 8px;
}}
.chart-title {{
  font-size: 13px;
  color: #0f172a;
  margin: 2px 2px 4px;
}}
.chart-subtitle {{
  font-size: 12px;
  color: var(--muted);
  margin: 0 2px 8px;
}}
.chart-canvas {{
  width: 100%;
  height: 240px;
  display: block;
}}
.series-legend {{
  margin-top: 8px;
  border-top: 1px solid var(--line);
  padding-top: 8px;
}}
.series-row {{
  display: grid;
  grid-template-columns: 14px 1fr 88px 88px 88px;
  gap: 8px;
  align-items: center;
  font-size: 12px;
  padding: 3px 0;
}}
.series-row.header {{
  color: var(--muted);
  font-weight: 600;
  padding-top: 1px;
}}
.series-chip {{
  width: 12px;
  height: 12px;
  border-radius: 3px;
  border: 1px solid #ffffff;
  box-shadow: 0 0 0 1px #cbd5e1 inset;
}}
.series-row .num {{
  text-align: right;
  font-variant-numeric: tabular-nums;
}}
.heatmap-grid {{
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  margin-top: 16px;
}}
.heatmap-card {{
  border: 1px solid var(--line);
  border-radius: 12px;
  background: #ffffff;
  padding: 10px;
}}
.heatmap-table {{
  margin-top: 0;
}}
.heatmap-table th,
.heatmap-table td {{
  text-align: center;
  vertical-align: middle;
}}
.heatmap-table th.num,
.heatmap-table td.num {{
  text-align: center;
}}
.heatmap-cell {{
  font-weight: 600;
}}
.file-list {{
  line-height: 1.8;
  font-size: 13px;
}}
@media (max-width: 700px) {{
  .container {{ padding: 12px; }}
  .chart-canvas {{ height: 210px; }}
}}
</style>
</head>
<body>
  <div class="container">
    <h1>LoRA Diagnostic Report</h1>
    <p class="sub">{report.get("base_name", "-")} / generated at {report.get("generated_at", "-")}</p>

    <section class="panel">
      <div class="overall">
        総合診断:
        <span class="score {diagnostics.get("overall_class", "info")}">{diagnostics.get("overall_status", "-")} ({score_text})</span>
      </div>
      <div class="grid cards" style="margin-top: 12px;">
        <div class="card">
          <div class="title">Grad しきい値超過率</div>
          <div class="value">{fmt_percent((((report.get("grad") or {}).get("summary") or {}).get("threshold_exceeded_ratio")), 2)}</div>
          <div class="caption">低いほど更新スキップの偏りが小さい</div>
        </div>
        <div class="card">
          <div class="title">Loss MA 低下率</div>
          <div class="value">{fmt_percent((((report.get("grad") or {}).get("summary") or {}).get("loss_ma_drop_ratio")), 2)}</div>
          <div class="caption">高いほど収束方向</div>
        </div>
        <div class="card">
          <div class="title">DQ Auto in-band比率</div>
          <div class="value">{fmt_percent((((report.get("dq") or {}).get("summary") or {}).get("in_band_ratio")), 2)}</div>
          <div class="caption">高いほど auto 制御が安定</div>
        </div>
        <div class="card">
          <div class="title">最終 QuantErrRatioEMA</div>
          <div class="value">{fmt_float((((report.get("dq") or {}).get("summary") or {}).get("final_quant_err_ratio_ema")), 4)}</div>
          <div class="caption">低いほど量子化誤差が小さい</div>
        </div>
        <div class="card">
          <div class="title">最終 RankSatP95</div>
          <div class="value">{fmt_float((((report.get("rank") or {}).get("summary") or {}).get("final_rank_sat_p95")), 4)}</div>
          <div class="caption">高すぎると rank 飽和の兆候</div>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>カテゴリ</th>
            <th>項目</th>
            <th class="num">値</th>
            <th>判定</th>
            <th>メモ</th>
          </tr>
        </thead>
        <tbody>{render_check_rows(diagnostics.get("checks", []))}</tbody>
      </table>
      <p class="sub">判定はヒューリスティクスです。プロジェクト特性に合わせてしきい値を調整してください。</p>
    </section>

    <section class="panel">
      <h2>GradNorm Dashboard</h2>
      <p class="sub">既存 `make_dashboard.py` 相当の5グラフを統合表示しています。</p>
      <div id="gradCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>DQ Delta Dashboard</h2>
      <p class="sub">X軸は TrainStep。縦線ラベルで Epoch を表示します。</p>
      <div id="dqCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>Rank Dashboard</h2>
      <p class="sub">LoRA重みから推定した rank 飽和指標の推移です。</p>
      <div class="callout" style="margin: 12px 0 16px 0;">
        <strong>RankSat グラフの見方</strong>
        <p class="sub" style="margin: 8px 0 0 0;">
          このグラフは、LoRA の各モジュールが「設定した rank をどのくらい広く使えているか」をまとめて見せるものです。
          値が高いほど、複数の rank 成分を使って学習しており、値が低いほど、少数の成分に偏って学習している傾向があります。
        </p>
        <p class="sub" style="margin: 8px 0 0 0;">
          `RankSatWMean` は、値が小さいほど少数の rank 成分への偏りが強く、値が大きいほど複数の rank 成分を広く使っています。
        </p>
        <p class="sub" style="margin: 8px 0 0 0;">
          `RankSatWMean` は全モジュールを重み付きで平均した代表値で、全体としての rank の使われ方を見ます。
          `P50` は中央値で、典型的なモジュールがどの程度 rank を使っているかを見ます。
          `P95` は上位 5% 側の値で、一部のモジュールが強く rank を使っていないかを見ます。
          `Max` は最も高いモジュールの値で、局所的に rank を使い切っている層があるかを見ます。
          `Top1P95` は「1つ目の成分への偏り」の強いモジュールがどの程度あるかを見る指標で、高いほど、実質的に少数成分へ寄っている可能性があります。
        </p>
        <p class="sub" style="margin: 8px 0 0 0;">
          `WMean` や `P50` が高いと、全体として rank を広く使えている可能性があります。
          `P95` や `Max` だけ高い場合は、一部のモジュールだけが強く rank を使っている可能性があります。
          `Top1P95` も高い場合は、rank はあるが、実際には少数成分への偏りが強い可能性があります。
        </p>
      </div>
      {rank_grouped_help_html}
      <div id="rankCharts" class="chart-grid"></div>
      {"<div class='heatmap-grid'>" + rank_heatmap_cards_html + "</div>" if rank_heatmap_cards_html else ""}
    </section>

    <section class="panel">
      <h2>Group Loss Dashboard</h2>
      <p class="sub">group lossログが存在する場合に、step単位とepoch単位のEMA推移を表示します。</p>
      <div id="groupLossCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>LoRA Checkpoint Analysis</h2>
      {lora_error_html}
      <div class="grid cards">
        <div class="card">
          <div class="title">総ブロック数</div>
          <div class="value">{fmt_int(lora_cards.get("total_blocks"))}</div>
          <div class="caption">解析対象LoRAブロック数</div>
        </div>
        <div class="card">
          <div class="title">総パラメータ</div>
          <div class="value">{fmt_int(lora_cards.get("total_params"))}</div>
          <div class="caption">up/down重み合算</div>
        </div>
        <div class="card">
          <div class="title">情報密度中央値</div>
          <div class="value">{fmt_float(lora_cards.get("density_median"))}</div>
          <div class="caption">高すぎ/低すぎの偏りを確認</div>
        </div>
        <div class="card">
          <div class="title">RMS中央値</div>
          <div class="value">{fmt_float(lora_cards.get("rms_median"))}</div>
          <div class="caption">更新量の強さの目安</div>
        </div>
        <div class="card">
          <div class="title">Entropy中央値</div>
          <div class="value">{fmt_float(lora_cards.get("entropy_median"))}</div>
          <div class="caption">分布の広がりの目安</div>
        </div>
        <div class="card">
          <div class="title">Sparsity中央値</div>
          <div class="value">{fmt_float(lora_cards.get("sparsity_median"))}</div>
          <div class="caption">高すぎると情報不足の可能性</div>
        </div>
      </div>

      <h3 style="margin-top: 16px;">モジュール別統計</h3>
      <table>
        <thead>
          <tr>
            <th>モジュール</th>
            <th class="num">ブロック数</th>
            <th class="num">総パラメータ</th>
            <th class="num">情報密度平均</th>
            <th class="num">情報密度中央値</th>
            <th class="num">RMS中央値</th>
            <th class="num">Entropy中央値</th>
          </tr>
        </thead>
        <tbody>{module_rows}</tbody>
      </table>

      <h3 style="margin-top: 16px;">UNetブロック別概要</h3>
      <table>
        <thead>
          <tr>
            <th>UNetブロック</th>
            <th class="num">LoRA数</th>
            <th class="num">総パラメータ</th>
            <th class="num">情報密度平均</th>
            <th class="num">情報密度中央値</th>
            <th class="num">RMS中央値</th>
          </tr>
        </thead>
        <tbody>{unet_rows}</tbody>
      </table>
    </section>

    <section class="panel">
      <h2>LoRA 情報密度エポック推移</h2>
      <p class="sub">`--lora_epoch_trend` 有効時のみ表示。系列ごとに色を固定し、下の凡例でブロック名との対応を示します。</p>
      {lora_trend_error_html}
      <div id="loraTrendCharts" class="chart-grid"></div>
    </section>

    <section class="panel">
      <h2>Input Files</h2>
      <div class="file-list">
        <div>Grad Log: <code>{(report.get("grad") or {}).get("path", "-")}</code></div>
        <div>DQ Log: <code>{(report.get("dq") or {}).get("path", "-")}</code></div>
        <div>DQ Auto Log: <code>{(report.get("dq") or {}).get("auto_path", "-")}</code></div>
        <div>Rank Log: <code>{(report.get("rank") or {}).get("path", "-")}</code></div>
        <div>Group Loss Step Log: <code>{(report.get("group_loss") or {}).get("step_path", "-")}</code></div>
        <div>Group Loss Epoch Log: <code>{(report.get("group_loss") or {}).get("epoch_path", "-")}</code></div>
        <div>LoRA Final Checkpoint: <code>{(report.get("lora") or {}).get("path", "-")}</code></div>
      </div>
    </section>
  </div>

<script>
const reportData = {report_json};

function withAlpha(hex, alpha) {{
  const clean = hex.replace('#', '');
  const r = parseInt(clean.slice(0, 2), 16);
  const g = parseInt(clean.slice(2, 4), 16);
  const b = parseInt(clean.slice(4, 6), 16);
  return `rgba(${{r}}, ${{g}}, ${{b}}, ${{alpha}})`;
}}

function finite(value) {{
  return typeof value === 'number' && Number.isFinite(value);
}}

function reduceMarkers(markers, maxCount = 14) {{
  if (!Array.isArray(markers) || markers.length <= maxCount) return markers || [];
  const step = Math.ceil(markers.length / maxCount);
  const out = [];
  for (let i = 0; i < markers.length; i += step) {{
    out.push(markers[i]);
  }}
  const last = markers[markers.length - 1];
  if (out[out.length - 1] !== last) out.push(last);
  return out;
}}

function decimalsFromStep(step) {{
  if (!finite(step) || step <= 0) return 4;
  const txt = String(step);
  if (!txt.includes('.')) return 0;
  return Math.min(6, txt.split('.')[1].length);
}}

function buildNiceIntegerStep(minVal, maxVal, targetTicks = 6) {{
  const range = Math.max(1e-9, maxVal - minVal);
  const rough = range / Math.max(2, targetTicks);
  const base = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / base;
  let nice = 1;
  if (scaled > 5) {{
    nice = 10;
  }} else if (scaled > 2) {{
    nice = 5;
  }} else if (scaled > 1) {{
    nice = 2;
  }}
  return Math.max(1, Math.round(nice * base));
}}

function buildYAxisTicks(bounds, chart) {{
  let minVal = bounds.yMin;
  let maxVal = bounds.yMax;
  if (finite(chart.y_min_fixed)) {{
    minVal = chart.y_min_fixed;
  }}
  if (finite(chart.y_max_fixed)) {{
    maxVal = chart.y_max_fixed;
  }}
  if (finite(chart.y_min_floor)) {{
    minVal = Math.min(minVal, chart.y_min_floor);
  }}
  if (finite(chart.y_max_ceil)) {{
    maxVal = Math.max(maxVal, chart.y_max_ceil);
  }}
  if (!(maxVal > minVal)) {{
    const pad = Math.abs(maxVal || 1) * 0.05 + 1e-6;
    minVal -= pad;
    maxVal += pad;
  }}

  let step = null;
  let ticks = [];
  if (finite(chart.y_tick_step) && chart.y_tick_step > 0) {{
    step = chart.y_tick_step;
    let start = Math.floor(minVal / step) * step;
    let end = Math.ceil(maxVal / step) * step;
    let count = Math.round((end - start) / step) + 1;
    if (count > 24) {{
      const mul = Math.ceil(count / 24);
      step *= mul;
      start = Math.floor(minVal / step) * step;
      end = Math.ceil(maxVal / step) * step;
      count = Math.round((end - start) / step) + 1;
    }}
    for (let i = 0; i < count; i += 1) {{
      ticks.push(start + i * step);
    }}
    minVal = start;
    maxVal = end;
  }} else if (chart.y_tick_nice_integer) {{
    step = buildNiceIntegerStep(minVal, maxVal, 6);
    const start = Math.floor(minVal / step) * step;
    const end = Math.ceil(maxVal / step) * step;
    const count = Math.round((end - start) / step) + 1;
    for (let i = 0; i < count; i += 1) {{
      ticks.push(start + i * step);
    }}
    minVal = start;
    maxVal = end;
  }} else {{
    const tickCount = 5;
    for (let i = 0; i <= tickCount; i += 1) {{
      ticks.push(minVal + ((maxVal - minVal) * i) / tickCount);
    }}
    step = (maxVal - minVal) / tickCount;
  }}
  return {{ min: minVal, max: maxVal, ticks, step }};
}}

function formatTickValue(value, chart, step) {{
  if (chart.y_tick_integer || chart.y_tick_nice_integer) {{
    return Math.round(value).toString();
  }}
  const precision = finite(chart.y_tick_precision) ? chart.y_tick_precision : decimalsFromStep(step);
  return Number(value).toFixed(Math.max(0, precision));
}}

function layoutInlineLegend(ctx, series, width, marginLeft, marginRight, maxRows) {{
  const rows = Math.max(1, maxRows || 2);
  const rowHeight = 14;
  const entries = [];
  let x = marginLeft;
  let row = 0;
  const rightLimit = Math.max(marginLeft + 20, width - marginRight);
  ctx.font = '11px sans-serif';
  (series || []).forEach((seriesItem) => {{
    const name = seriesItem.name || 'series';
    const itemWidth = 14 + ctx.measureText(name).width + 12;
    if (x + itemWidth > rightLimit && row + 1 < rows) {{
      row += 1;
      x = marginLeft;
    }}
    entries.push({{ series: seriesItem, x, y: 12 + row * rowHeight }});
    x += itemWidth;
  }});
  return {{ entries, height: rowHeight * (row + 1) + 4 }};
}}

function computeBounds(chart) {{
  let xMin = Infinity;
  let xMax = -Infinity;
  let yMin = Infinity;
  let yMax = -Infinity;
  let hasData = false;

  (chart.series || []).forEach((series) => {{
    const xVals = Array.isArray(series.x) && series.x.length ? series.x : (Array.isArray(chart.x) ? chart.x : []);
    const yVals = Array.isArray(series.y) ? series.y : [];
    const n = Math.min(xVals.length, yVals.length);
    for (let i = 0; i < n; i += 1) {{
      const x = xVals[i];
      const y = yVals[i];
      if (!finite(x) || !finite(y)) continue;
      hasData = true;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    }}
  }});

  if (!hasData) return null;
  if (Math.abs(yMax - yMin) < 1e-12) {{
    const pad = Math.abs(yMax || 1) * 0.05 + 1e-6;
    yMin -= pad;
    yMax += pad;
  }} else {{
    const pad = (yMax - yMin) * 0.10;
    yMin -= pad;
    yMax += pad;
  }}
  if (xMin === xMax) {{
    xMin -= 1;
    xMax += 1;
  }}
  return {{ xMin, xMax, yMin, yMax }};
}}

function drawChart(canvas, chart) {{
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(320, Math.floor(canvas.clientWidth));
  const height = Math.max(180, Math.floor(canvas.clientHeight));
  const pxW = Math.floor(width * ratio);
  const pxH = Math.floor(height * ratio);
  if (canvas.width !== pxW || canvas.height !== pxH) {{
    canvas.width = pxW;
    canvas.height = pxH;
  }}
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const bounds = computeBounds(chart);
  if (!bounds) {{
    ctx.fillStyle = '#64748b';
    ctx.font = '13px sans-serif';
    ctx.fillText('有効な描画データがありません', 12, 28);
    return;
  }}

  let legendLayout = null;
  let inlineLegendHeight = 0;
  if (!chart.no_inline_legend) {{
    legendLayout = layoutInlineLegend(
      ctx,
      chart.series || [],
      width,
      58,
      14,
      finite(chart.legend_max_rows) ? chart.legend_max_rows : 2
    );
    inlineLegendHeight = legendLayout.height;
  }}

  const margin = {{ top: 18 + inlineLegendHeight, right: 14, bottom: 34, left: 58 }};
  const chartW = Math.max(10, width - margin.left - margin.right);
  const chartH = Math.max(10, height - margin.top - margin.bottom);
  const yAxis = buildYAxisTicks(bounds, chart);
  const yRange = Math.max(1e-12, yAxis.max - yAxis.min);

  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = '#e2e8f0';
  ctx.lineWidth = 1;
  yAxis.ticks.forEach((tick) => {{
    const y = margin.top + chartH - ((tick - yAxis.min) / yRange) * chartH;
    ctx.beginPath();
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + chartW, y);
    ctx.stroke();
  }});

  const markers = reduceMarkers(chart.markers, 12);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#cbd5e1';
  markers.forEach((marker) => {{
    if (!finite(marker.x)) return;
    const t = (marker.x - bounds.xMin) / (bounds.xMax - bounds.xMin);
    const x = margin.left + t * chartW;
    if (x < margin.left || x > margin.left + chartW) return;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + chartH);
    ctx.stroke();
    ctx.save();
    ctx.fillStyle = '#64748b';
    ctx.font = '10px sans-serif';
    ctx.fillText(marker.label || '', x + 2, margin.top + 10);
    ctx.restore();
  }});
  ctx.setLineDash([]);

  (chart.series || []).forEach((series) => {{
    const xVals = (Array.isArray(series.x) && series.x.length) ? series.x : (chart.x || []);
    const yVals = series.y || [];
    const n = Math.min(xVals.length, yVals.length);
    ctx.strokeStyle = series.color || '#2563eb';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    let moved = false;
    for (let i = 0; i < n; i += 1) {{
      const x = xVals[i];
      const y = yVals[i];
      if (!finite(x) || !finite(y)) {{
        moved = false;
        continue;
      }}
      const px = margin.left + ((x - bounds.xMin) / (bounds.xMax - bounds.xMin)) * chartW;
      const py = margin.top + chartH - ((y - yAxis.min) / yRange) * chartH;
      if (!moved) {{
        ctx.moveTo(px, py);
        moved = true;
      }} else {{
        ctx.lineTo(px, py);
      }}
    }}
    ctx.stroke();
  }});

  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + chartH);
  ctx.lineTo(margin.left + chartW, margin.top + chartH);
  ctx.stroke();

  ctx.fillStyle = '#475569';
  ctx.font = '11px sans-serif';
  yAxis.ticks.forEach((tick) => {{
    const y = margin.top + chartH - ((tick - yAxis.min) / yRange) * chartH;
    ctx.fillText(formatTickValue(tick, chart, yAxis.step), 6, y + 4);
  }});
  ctx.fillText(Math.round(bounds.xMin).toString(), margin.left, margin.top + chartH + 18);
  const xMid = (bounds.xMin + bounds.xMax) / 2;
  const midText = Math.round(xMid).toString();
  const midW = ctx.measureText(midText).width;
  ctx.fillText(midText, margin.left + chartW / 2 - midW / 2, margin.top + chartH + 18);
  const maxText = Math.round(bounds.xMax).toString();
  const maxW = ctx.measureText(maxText).width;
  ctx.fillText(maxText, margin.left + chartW - maxW, margin.top + chartH + 18);

  if (!chart.no_inline_legend && legendLayout) {{
    legendLayout.entries.forEach((entry) => {{
      const series = entry.series || {{}};
      const color = series.color || '#2563eb';
      const name = series.name || 'series';
      ctx.fillStyle = color;
      ctx.fillRect(entry.x, entry.y - 8, 10, 10);
      ctx.fillStyle = '#334155';
      ctx.font = '11px sans-serif';
      ctx.fillText(name, entry.x + 14, entry.y);
    }});
  }}
}}

function appendSeriesLegend(parent, chart) {{
  const rows = Array.isArray(chart.legend_rows) ? chart.legend_rows : [];
  if (!rows.length) return;
  const wrap = document.createElement('div');
  wrap.className = 'series-legend';

  const header = document.createElement('div');
  header.className = 'series-row header';
  header.innerHTML = '<div></div><div>ブロック</div><div class="num">開始密度</div><div class="num">終了密度</div><div class="num">差分</div>';
  wrap.appendChild(header);

  rows.forEach((row) => {{
    const el = document.createElement('div');
    el.className = 'series-row';
    const startVal = finite(row.start_density) ? Number(row.start_density).toFixed(4) : '-';
    const endVal = finite(row.end_density) ? Number(row.end_density).toFixed(4) : '-';
    const deltaVal = finite(row.delta_density) ? Number(row.delta_density).toFixed(4) : '-';
    el.innerHTML =
      `<div class="series-chip" style="background:${{row.color || '#2563eb'}};"></div>` +
      `<div>${{row.name || '-'}}</div>` +
      `<div class="num">${{startVal}}</div>` +
      `<div class="num">${{endVal}}</div>` +
      `<div class="num">${{deltaVal}}</div>`;
    wrap.appendChild(el);
  }});
  parent.appendChild(wrap);
}}

function mountCharts(containerId, charts) {{
  const container = document.getElementById(containerId);
  if (!container || !Array.isArray(charts)) return;
  if (charts.length === 0) {{
    const empty = document.createElement('div');
    empty.className = 'muted';
    empty.textContent = '表示できるデータがありません。';
    container.appendChild(empty);
    return;
  }}
  const states = [];
  charts.forEach((chart, idx) => {{
    const card = document.createElement('div');
    card.className = 'chart-card';
    const title = document.createElement('div');
    title.className = 'chart-title';
    title.textContent = chart.title || `Chart ${{idx + 1}}`;
    const subtitle = document.createElement('div');
    subtitle.className = 'chart-subtitle';
    subtitle.textContent = chart.subtitle || '';
    const canvas = document.createElement('canvas');
    canvas.className = 'chart-canvas';
    card.appendChild(title);
    if (chart.subtitle) {{
      card.appendChild(subtitle);
    }}
    card.appendChild(canvas);
    appendSeriesLegend(card, chart);
    container.appendChild(card);
    states.push({{ canvas, chart }});
  }});

  const renderAll = () => {{
    states.forEach((state) => drawChart(state.canvas, state.chart));
  }};
  renderAll();
  window.addEventListener('resize', renderAll);
}}

mountCharts('gradCharts', ((reportData.charts || {{}}).grad || []));
mountCharts('dqCharts', ((reportData.charts || {{}}).dq || []));
mountCharts('rankCharts', ((reportData.charts || {{}}).rank || []));
mountCharts('groupLossCharts', ((reportData.charts || {{}}).group_loss || []));
mountCharts('loraTrendCharts', ((reportData.charts || {{}}).lora_trend || []));
</script>
</body>
</html>
"""


def resolve_paths(args: argparse.Namespace) -> Dict[str, Optional[str]]:
    input_dir = args.input_dir
    base_name = args.base_name

    grad_log = os.path.join(input_dir, f"gradient_logs+{base_name}.txt")
    dq_log = os.path.join(input_dir, f"dq_delta_logs+{base_name}.txt")
    dq_auto_log = os.path.join(input_dir, f"dq_delta_auto+{base_name}.txt")
    rank_log = os.path.join(input_dir, f"rank_logs+{base_name}.txt")
    group_loss_step_log = os.path.join(input_dir, f"group_loss_logs+{base_name}.csv")
    group_loss_epoch_log = os.path.join(input_dir, f"group_loss_epoch+{base_name}.csv")
    model = os.path.join(input_dir, f"{base_name}.safetensors")

    return {
        "grad_log": grad_log,
        "dq_log": dq_log,
        "dq_auto_log": dq_auto_log,
        "rank_log": rank_log,
        "group_loss_step_log": group_loss_step_log,
        "group_loss_epoch_log": group_loss_epoch_log,
        "model": model,
    }


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LoRA学習ログ診断ダッシュボード生成ツール")
    parser.add_argument("--base_name", required=True, help="LoRAのベース名 (例: brak_xl31c_noob075V)")
    parser.add_argument(
        "--input_dir",
        default=".",
        help="ログとモデルがあるディレクトリ（ログ名は gradient_logs+/dq_delta_logs+/dq_delta_auto+/rank_logs+ の固定形式）",
    )
    parser.add_argument("--loss_ma_window", type=int, default=100, help="Loss移動平均の窓サイズ")
    parser.add_argument("--lora_bins", type=int, default=128, help="LoRA重み解析のヒストグラムビン数")
    parser.add_argument("--skip_lora_analysis", action="store_true", help="LoRAチェックポイント解析をスキップ")
    parser.add_argument("--lora_epoch_trend", action="store_true", help="情報密度のエポック推移解析を有効化")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="出力ディレクトリ（未指定時は --input_dir/diagnostic_report）",
    )
    parser.add_argument("--output_html", default=None, help="HTML出力先。未指定時は output_dir に自動生成")
    parser.add_argument("--output_json", default=None, help="JSON出力先。未指定時は output_dir に自動生成")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    paths = resolve_paths(args)
    grad_log_path = paths["grad_log"]
    dq_log_path = paths["dq_log"]
    dq_auto_log_path = paths["dq_auto_log"]
    rank_log_path = paths["rank_log"]
    group_loss_step_log_path = paths["group_loss_step_log"]
    group_loss_epoch_log_path = paths["group_loss_epoch_log"]
    model_path = paths["model"]

    if grad_log_path and not os.path.exists(grad_log_path):
        grad_log_path = None
    if dq_log_path and not os.path.exists(dq_log_path):
        dq_log_path = None
    if dq_auto_log_path and not os.path.exists(dq_auto_log_path):
        dq_auto_log_path = None
    if rank_log_path and not os.path.exists(rank_log_path):
        rank_log_path = None
    if group_loss_step_log_path and not os.path.exists(group_loss_step_log_path):
        group_loss_step_log_path = None
    if group_loss_epoch_log_path and not os.path.exists(group_loss_epoch_log_path):
        group_loss_epoch_log_path = None

    if not any([grad_log_path, dq_log_path, dq_auto_log_path, rank_log_path, group_loss_step_log_path, group_loss_epoch_log_path]):
        raise FileNotFoundError(
            f"入力ログが見つかりません: {args.input_dir} (gradient_logs+/dq_delta_logs+/dq_delta_auto+/rank_logs+/group_loss_logs+/group_loss_epoch+)"
        )

    grad_data = parse_grad_log(grad_log_path, args.loss_ma_window) if grad_log_path else None
    dq_data = parse_dq_logs(dq_log_path, dq_auto_log_path, grad_data) if (dq_log_path or dq_auto_log_path) else None
    rank_module_param_counts = None
    if rank_log_path and os.path.exists(model_path):
        try:
            rank_module_param_counts = load_lora_module_param_counts(model_path)
        except Exception as exc:
            print(f"[rank_norm] skip module param normalization: {exc}", flush=True)
    rank_data = parse_rank_logs(rank_log_path, grad_data, module_param_counts=rank_module_param_counts) if rank_log_path else None
    group_loss_data = parse_group_loss_logs(group_loss_step_log_path, group_loss_epoch_log_path)

    lora_data = None
    lora_error = None
    lora_trend_data = None
    lora_trend_error = None
    if not args.skip_lora_analysis:
        if not os.path.exists(model_path):
            lora_error = f"LoRAチェックポイントが見つかりません: {model_path}"
        else:
            try:
                lora_data = analyze_lora_checkpoint(model_path, args.lora_bins)
            except Exception as exc:
                lora_error = f"LoRA解析に失敗しました: {exc}"
            if args.lora_epoch_trend:
                try:
                    print("[lora_epoch_trend] start", flush=True)
                    lora_trend_data = analyze_lora_epoch_trend(
                        model_path=model_path,
                        bins=args.lora_bins,
                    )
                    print("[lora_epoch_trend] complete", flush=True)
                except Exception as exc:
                    lora_trend_error = f"LoRAエポック推移解析に失敗しました: {exc}"
    elif args.lora_epoch_trend:
        lora_trend_error = "--skip_lora_analysis 指定時は --lora_epoch_trend を実行できません。"

    diagnostics = build_diagnostics(grad_data, dq_data, rank_data, lora_data)
    charts = build_chart_payload(grad_data, dq_data, rank_data, group_loss_data, lora_trend_data)

    report = {
        "base_name": args.base_name,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "grad": grad_data,
        "dq": dq_data,
        "rank": rank_data,
        "group_loss": group_loss_data,
        "lora": lora_data,
        "lora_error": lora_error,
        "lora_trend": lora_trend_data,
        "lora_trend_error": lora_trend_error,
        "diagnostics": diagnostics,
        "charts": charts,
    }

    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, "diagnostic_report")
    os.makedirs(output_dir, exist_ok=True)
    html_path = args.output_html or os.path.join(output_dir, f"{args.base_name}_diagnostic.html")
    json_path = args.output_json or os.path.join(output_dir, f"{args.base_name}_diagnostic.json")

    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(sanitize_json(report), fp, ensure_ascii=False, indent=2)
    html = build_html(report)
    with open(html_path, "w", encoding="utf-8") as fp:
        fp.write(html)

    print(f"JSON saved: {json_path}")
    print(f"HTML saved: {html_path}")
    if lora_error:
        print(f"[warn] {lora_error}")
    if lora_trend_error:
        print(f"[warn] {lora_trend_error}")


if __name__ == "__main__":
    main()
