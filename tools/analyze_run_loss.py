"""Summarize the per-sample loss log written by `--run_control` (`<output_dir>/<output_name>_run/loss.jsonl`).

The loss of a sample depends strongly on the timestep which was drawn for it, so the raw loss is not comparable
between samples. This script therefore uses the *residual*: the relative difference from the mean loss of the samples
in the same timestep bin (0 = average, +0.5 = 50% higher than the average at that noise level).

It prints the loss by timestep, by resolution, by batch size and by directory, and the images with the highest and the
lowest residual. For each image, `slope` is the change of the residual per epoch: a high residual which does not
decrease may indicate a problem of the image or its caption, but an image is seen only once per epoch (times the number
of repeats), so the numbers are noisy when `count` is small.

This script only reads the log. It has no dependencies other than the standard library.

Usage:
    python tools/analyze_run_loss.py <run directory or loss.jsonl> [--top 20] [--json]
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Optional

LOSS_FILE = "loss.jsonl"


def load_runs(path: str) -> dict[str, list[dict]]:
    """Returns {run_id: rows} in the order of the file. Each run starts with a line which has `run_id` only."""
    if os.path.isdir(path):
        path = os.path.join(path, LOSS_FILE)
    runs: dict[str, list[dict]] = {}
    current: Optional[list[dict]] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue  # the last line may be incomplete if the training is running or was killed
            if "image" not in row:
                if "run_id" in row:
                    current = runs.setdefault(row["run_id"], [])
                continue
            if current is None:
                current = runs.setdefault("(unknown)", [])
            if row.get("loss") is None:
                continue  # non-finite loss
            current.append(row)
    return runs


def slope(xs: list[float], ys: list[float]) -> Optional[float]:
    """Least-squares slope of ys against xs. None if it cannot be calculated."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denominator


def add_residuals(rows: list[dict], num_bins: int) -> list[dict]:
    """Adds `bin` and `residual` to each row, and returns the table of the timestep bins."""
    timesteps = [r["timestep"] for r in rows if r.get("timestep") is not None]
    t_min, t_max = (min(timesteps), max(timesteps)) if timesteps else (0.0, 0.0)
    width = (t_max - t_min) / num_bins if t_max > t_min else 0.0

    sums = defaultdict(float)
    counts = defaultdict(int)
    for r in rows:
        t = r.get("timestep")
        r["bin"] = 0 if (t is None or width == 0.0) else min(num_bins - 1, int((t - t_min) / width))
        sums[r["bin"]] += r["loss"]
        counts[r["bin"]] += 1

    bin_means = {b: sums[b] / counts[b] for b in counts}
    for r in rows:
        mean = bin_means[r["bin"]]
        r["residual"] = (r["loss"] / mean - 1.0) if mean > 0 else 0.0

    table = []
    for b in sorted(counts):
        table.append(
            {
                "timestep_from": t_min + width * b,
                "timestep_to": t_min + width * (b + 1) if width > 0 else t_max,
                "count": counts[b],
                "mean_loss": bin_means[b],
            }
        )
    return table


def group_table(rows: list[dict], key_fn, key_name: str) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    table = []
    for key, group in groups.items():
        table.append(
            {
                key_name: key,
                "count": len(group),
                "images": len({r["image"] for r in group}),
                "mean_loss": sum(r["loss"] for r in group) / len(group),
                "mean_residual": sum(r["residual"] for r in group) / len(group),
            }
        )
    return table


def image_table(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["image"]].append(r)
    table = []
    for image, group in groups.items():
        by_epoch = defaultdict(list)
        for r in group:
            if r.get("epoch") is not None:
                by_epoch[r["epoch"]].append(r["residual"])
        epochs = sorted(by_epoch)
        residual_slope = None
        if len(epochs) >= 3:  # a slope from two points is too noisy
            residual_slope = slope([float(e) for e in epochs], [sum(by_epoch[e]) / len(by_epoch[e]) for e in epochs])
        table.append(
            {
                "image": image,
                "count": len(group),
                "epochs": len(epochs),
                "mean_loss": sum(r["loss"] for r in group) / len(group),
                "mean_residual": sum(r["residual"] for r in group) / len(group),
                "slope": residual_slope,
            }
        )
    table.sort(key=lambda x: x["mean_residual"], reverse=True)
    return table


def resolution_key(row: dict) -> str:
    if row.get("width") is None or row.get("height") is None:
        return "(unknown)"
    return f"{row['width']}x{row['height']}"


def analyze(rows: list[dict], num_bins: int = 10) -> dict:
    """Returns the summary and the tables. `rows` is modified (`bin` and `residual` are added)."""
    timestep_bins = add_residuals(rows, num_bins)

    by_resolution = group_table(rows, resolution_key, "resolution")
    by_resolution.sort(key=lambda x: x["count"], reverse=True)
    by_batch_size = group_table(rows, lambda r: r.get("batch_size"), "batch_size")
    by_batch_size.sort(key=lambda x: (x["batch_size"] is None, x["batch_size"]))
    by_directory = group_table(rows, lambda r: os.path.dirname(str(r["image"])), "directory")
    by_directory.sort(key=lambda x: x["directory"])

    steps = [r["step"] for r in rows if r.get("step") is not None]
    epochs = [r["epoch"] for r in rows if r.get("epoch") is not None]
    return {
        "summary": {
            "samples": len(rows),
            "images": len({r["image"] for r in rows}),
            "step_from": min(steps) if steps else None,
            "step_to": max(steps) if steps else None,
            "epoch_from": min(epochs) if epochs else None,
            "epoch_to": max(epochs) if epochs else None,
            "mean_loss": sum(r["loss"] for r in rows) / len(rows) if rows else None,
        },
        "by_timestep": timestep_bins,
        "by_resolution": by_resolution,
        "by_batch_size": by_batch_size,
        "by_directory": by_directory,
        "by_image": image_table(rows),
    }


def format_value(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def print_table(title: str, table: list[dict], columns: list[str]):
    print(f"\n## {title}")
    if not table:
        print("(no data)")
        return
    cells = [[format_value(row.get(c)) for c in columns] for row in table]
    widths = [max(len(c), *(len(row[i]) for row in cells)) for i, c in enumerate(columns)]
    # the first column is a name (left aligned), the others are numbers (right aligned)
    print("  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(columns, widths))))
    for row in cells:
        print("  ".join(v.ljust(w) if i == 0 else v.rjust(w) for i, (v, w) in enumerate(zip(row, widths))))


def print_report(run_id: str, result: dict, top: int, min_count: int):
    summary = result["summary"]
    print(f"run_id: {run_id}")
    print(
        f"samples: {summary['samples']}, images: {summary['images']}, steps: {summary['step_from']}-{summary['step_to']},"
        f" epochs: {summary['epoch_from']}-{summary['epoch_to']}, mean loss: {format_value(summary['mean_loss'])}"
    )
    print("residual: relative difference from the mean loss of the same timestep bin (0 = average)")

    print_table("Loss by timestep", result["by_timestep"], ["timestep_from", "timestep_to", "count", "mean_loss"])
    group_columns = ["count", "images", "mean_loss", "mean_residual"]
    print_table("Loss by resolution", result["by_resolution"], ["resolution"] + group_columns)
    print_table("Loss by batch size", result["by_batch_size"], ["batch_size"] + group_columns)
    print_table("Loss by directory", result["by_directory"], ["directory"] + group_columns)

    images = [x for x in result["by_image"] if x["count"] >= min_count]
    skipped = len(result["by_image"]) - len(images)
    image_columns = ["image", "count", "epochs", "mean_loss", "mean_residual", "slope"]
    note = f" (count >= {min_count}, {skipped} images skipped)" if skipped > 0 else ""
    print_table(f"Images with the highest residual{note}", images[:top], image_columns)
    print_table(f"Images with the lowest residual{note}", images[::-1][:top], image_columns)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize the per-sample loss log of --run_control (loss.jsonl)")
    parser.add_argument("path", type=str, help="run directory (<output_dir>/<output_name>_run) or loss.jsonl")
    parser.add_argument("--run_id", type=str, default=None, help="run to analyze. default is the last run in the file")
    parser.add_argument("--all_runs", action="store_true", help="analyze all runs in the file together, e.g. a resumed training")
    parser.add_argument("--top", type=int, default=20, help="number of images to show in the highest / lowest lists")
    parser.add_argument(
        "--min_count", type=int, default=3, help="images which were trained less than this number of times are not listed"
    )
    parser.add_argument("--timestep_bins", type=int, default=10, help="number of timestep bins for the residual")
    parser.add_argument("--json", action="store_true", help="print the full result (all images) as JSON")
    return parser


def main():
    args = setup_parser().parse_args()

    runs = load_runs(args.path)
    if not runs:
        sys.exit(f"no run is found in {args.path}")

    if args.all_runs:
        run_id = ",".join(runs.keys())
        rows = [r for run_rows in runs.values() for r in run_rows]
    else:
        run_id = args.run_id if args.run_id is not None else list(runs.keys())[-1]
        if run_id not in runs:
            sys.exit(f"run_id {run_id} is not found. runs in the file: {', '.join(runs.keys())}")
        rows = runs[run_id]
    if not rows:
        sys.exit(f"no sample is recorded for the run {run_id}")

    result = analyze(rows, args.timestep_bins)
    if args.json:
        print(json.dumps({"run_id": run_id, **result}, ensure_ascii=False, indent=2))
    else:
        print_report(run_id, result, args.top, args.min_count)


if __name__ == "__main__":
    main()
