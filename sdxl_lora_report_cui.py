#!/usr/bin/env python
import argparse
import csv
import datetime as dt
import html
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path


def sanitize_id(value: str, fallback: str) -> str:
    value = (value or "").strip()
    if not value:
        value = fallback
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.ASCII)
    value = value.strip("._-")
    return value or fallback


def resolve_path(path: str | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_prompt_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".tsv" or looks_like_prompt_tsv(path):
        return parse_prompt_tsv(path)
    return parse_prompt_pipe_text(path)


def looks_like_prompt_tsv(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" not in line:
                    return False
                fields = [normalize_prompt_field(part) for part in next(csv.reader([line], delimiter="\t"))]
                return "prompt" in fields
    except OSError:
        return False
    return False


def parse_prompt_pipe_text(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split("|")]
            if len(parts) == 1:
                prompt_id = f"p{len(prompts) + 1:03d}"
                prompt = parts[0]
                negative = ""
                width = None
                height = None
            else:
                prompt_id = parts[0] or f"p{len(prompts) + 1:03d}"
                prompt = parts[1] if len(parts) > 1 else ""
                negative = parts[2] if len(parts) > 2 else ""
                width = int(parts[3]) if len(parts) > 3 and parts[3] else None
                height = int(parts[4]) if len(parts) > 4 and parts[4] else None

            if not prompt:
                raise ValueError(f"Prompt is empty at {path}:{line_no}")

            prompts.append(
                {
                    "id": sanitize_id(prompt_id, f"p{len(prompts) + 1:03d}"),
                    "prompt": prompt,
                    "negative": negative,
                    "width": width,
                    "height": height,
                    "line_no": line_no,
                }
            )

    if not prompts:
        raise ValueError(f"No prompts found: {path}")
    return prompts


def parse_prompt_tsv(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        lines = []
        line_numbers = []
        found_header = False
        for line_no, raw_line in enumerate(f, 1):
            if not found_header:
                stripped = raw_line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                found_header = True
            lines.append(raw_line)
            line_numbers.append(line_no)

        reader = csv.DictReader(lines, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No TSV header found: {path}")
        normalized_fields = {normalize_prompt_field(name): name for name in reader.fieldnames}
        if "prompt" not in normalized_fields:
            raise ValueError(f"TSV prompt file requires a 'prompt' column: {path}")

        for row_offset, row in enumerate(reader, 1):
            row_index = line_numbers[row_offset] if row_offset < len(line_numbers) else row_offset + 1
            if is_empty_prompt_row(row):
                continue
            prompt = cell(row, normalized_fields, "prompt")
            if not prompt or prompt.startswith("#"):
                continue

            prompt_id = cell(row, normalized_fields, "id") or f"p{len(prompts) + 1:03d}"
            negative = cell(row, normalized_fields, "negative")
            width = parse_optional_int(cell(row, normalized_fields, "width"), path, row_index, "width")
            height = parse_optional_int(cell(row, normalized_fields, "height"), path, row_index, "height")

            prompts.append(
                {
                    "id": sanitize_id(prompt_id, f"p{len(prompts) + 1:03d}"),
                    "prompt": prompt,
                    "negative": negative,
                    "width": width,
                    "height": height,
                    "line_no": row_index,
                }
            )

    if not prompts:
        raise ValueError(f"No prompts found: {path}")
    return prompts


def normalize_prompt_field(name: str | None) -> str:
    value = (name or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "prompt_id": "id",
        "negative_prompt": "negative",
        "neg": "negative",
        "w": "width",
        "h": "height",
    }
    return aliases.get(value, value)


def cell(row: dict, fields: dict[str, str], name: str) -> str:
    source = fields.get(name)
    if source is None:
        return ""
    return (row.get(source) or "").strip()


def is_empty_prompt_row(row: dict) -> bool:
    return not any((value or "").strip() for value in row.values())


def parse_optional_int(value: str, path: Path, line_no: int, column: str) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {column} value at {path}:{line_no}: {value}") from exc


def validate_image_size(width: int | None, height: int | None, source: str):
    if width is None or height is None:
        raise ValueError(f"{source}: width and height are required")
    for name, value in (("width", width), ("height", height)):
        if int(value) <= 0:
            raise ValueError(f"{source}: {name} must be positive: {value}")
        if int(value) % 64 != 0:
            raise ValueError(
                f"{source}: {name} must be a multiple of 64 for SDXL generation: {value}. "
                "Use a nearby value such as 896, 960, or 1024."
            )


def build_seeds(seed_config: dict) -> list[int]:
    values = [int(v) for v in seed_config.get("values", [])]
    negative_values = [value for value in values if value < 0]
    if negative_values:
        raise ValueError(f"Seed values must be non-negative because sdxl_gen_img.py prompt seeds do not accept negatives: {negative_values}")

    random_count = int(seed_config.get("random_count", 0) or 0)
    if random_count > 0:
        rnd = random.Random(seed_config.get("random_source_seed"))
        min_seed = int(seed_config.get("random_min", 0))
        max_seed = int(seed_config.get("random_max", 2**32 - 1))
        if min_seed < 0:
            raise ValueError(f"seeds.random_min must be non-negative: {min_seed}")
        if max_seed < min_seed:
            raise ValueError(f"seeds.random_max must be greater than or equal to seeds.random_min: {max_seed} < {min_seed}")
        values.extend(rnd.randint(min_seed, max_seed) for _ in range(random_count))
    if not values:
        raise ValueError("At least one seed is required. Set seeds.values or seeds.random_count.")
    return values


def normalize_lora_item(item: dict, config_dir: Path, force_lbw_module: bool, condition_id: str, item_index: int) -> dict:
    path = resolve_path(item.get("path"), config_dir)
    if path is None:
        raise ValueError(f"LoRA condition '{condition_id}' item {item_index} requires path")

    lbw = item.get("lbw")
    if force_lbw_module and lbw is None:
        raise ValueError(f"LoRA condition '{condition_id}' item {item_index} requires lbw when the condition uses LBW")

    module = item.get("module")
    if not module:
        module = "networks.lora_lbw" if force_lbw_module or lbw is not None else "networks.lora"

    return {
        "name": item.get("name") or Path(path).stem,
        "path": str(path),
        "strength": float(item.get("strength", 1.0)),
        "lbw": lbw,
        "module": module,
    }


def build_lora_conditions(config: dict, config_dir: Path) -> list[dict]:
    conditions = []
    if config.get("include_baseline", True):
        conditions.append({"id": "baseline", "name": "baseline", "items": []})

    for idx, raw in enumerate(config.get("loras", []), 1):
        condition_id = sanitize_id(raw.get("id") or raw.get("name"), f"lora_{idx:02d}")
        items = raw.get("items")
        if items is None:
            items = [raw]
        if not isinstance(items, list) or len(items) == 0:
            raise ValueError(f"LoRA condition '{condition_id}' requires a non-empty items list")
        force_lbw_module = any(item.get("lbw") is not None for item in items)
        normalized_items = [
            normalize_lora_item(item, config_dir, force_lbw_module, condition_id, item_index)
            for item_index, item in enumerate(items, 1)
        ]
        conditions.append(
            {
                "id": condition_id,
                "name": raw.get("name") or condition_id,
                "items": normalized_items,
            }
        )

    if not conditions:
        raise ValueError("No LoRA conditions found. Add loras or set include_baseline=true.")
    return conditions


def run_report_worker(
    output_dir: Path,
    gen_config: dict,
    conditions: list[dict],
    jobs: list[dict],
    dry_run: bool,
) -> int:
    active_jobs = list(jobs)
    if not active_jobs:
        return 0

    active_jobs = order_jobs_for_generation(active_jobs, conditions)

    worker_dir = output_dir / "worker"
    worker_outdir = worker_dir / "images"
    worker_dir.mkdir(parents=True, exist_ok=True)
    job_plan_path = worker_dir / "worker_job.json"
    result_path = worker_dir / "worker_result.json"
    job_index = {id(job): index for index, job in enumerate(jobs)}
    job_plan = {
        "sdxl_gen_img": gen_config,
        "conditions": conditions,
        "work_outdir": str(worker_outdir),
        "jobs": [
            {
                "job_index": job_index[id(job)],
                "prompt_id": job["prompt_id"],
                "prompt": job["prompt"],
                "negative": job.get("negative", ""),
                "width": job.get("width"),
                "height": job.get("height"),
                "seed": job["seed"],
                "condition_id": job["condition_id"],
                "condition_items": job["condition_items"],
                "target_path": job["target_path"],
            }
            for job in active_jobs
        ],
    }
    with job_plan_path.open("w", encoding="utf-8") as f:
        json.dump(job_plan, f, ensure_ascii=False, indent=2)

    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "sdxl_lora_report_worker.py"),
        "--job-json",
        str(job_plan_path),
        "--result-json",
        str(result_path),
    ]
    if dry_run:
        command.append("--dry-run")

    result = subprocess.run(command, cwd=Path(__file__).resolve().parent)
    if result_path.exists():
        try:
            with result_path.open("r", encoding="utf-8") as f:
                worker_result = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            mark_worker_failed(jobs, result.returncode, command, f"worker result JSON could not be read: {exc}")
        else:
            for item in worker_result.get("results", []):
                job = jobs[item["job_index"]]
                job["status"] = item.get("status", "failed")
                job["returncode"] = worker_result.get("returncode")
                job["worker_command"] = worker_result.get("command", [])
                job["worker_slots"] = worker_result.get("slots", [])
                if item.get("error"):
                    job["error"] = item["error"]
    elif result.returncode != 0:
        mark_worker_failed(jobs, result.returncode, command, "worker did not write result JSON")
    return result.returncode


def mark_worker_failed(jobs: list[dict], returncode: int, command: list[str], error: str):
    for job in jobs:
        job["status"] = "failed"
        job["returncode"] = returncode
        job["worker_command"] = command
        job["worker_slots"] = []
        job["error"] = error


def order_jobs_for_generation(jobs: list[dict], conditions: list[dict]) -> list[dict]:
    condition_order = {condition["id"]: index for index, condition in enumerate(conditions)}
    original_order = {id(job): index for index, job in enumerate(jobs)}
    return sorted(
        jobs,
        key=lambda job: (
            condition_order.get(job["condition_id"], len(condition_order)),
            original_order[id(job)],
        ),
    )


def copy_run_inputs(config_path: Path, prompt_path: Path, output_dir: Path, config: dict, prompts: list[dict]):
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    shutil.copy2(prompt_path, output_dir / "prompts.txt")
    with (output_dir / "prompts.parsed.json").open("w", encoding="utf-8") as f:
        json.dump({"source": str(config_path), "prompts": prompts}, f, ensure_ascii=False, indent=2)


def make_image_name(prompt: dict, prompt_index: int, seed: int) -> str:
    return f"p{prompt_index:04d}_{prompt['id']}_seed{seed}.png"


def target_path_key(path: Path) -> str:
    return os.path.normcase(str(path))


def generate_jobs(config_path: Path, config: dict) -> tuple[Path, list[dict], list[dict], list[int], list[dict]]:
    config_dir = config_path.parent
    output_root = resolve_path(config.get("output_root", "lora_reports"), config_dir)
    run_name = sanitize_id(config.get("run_name"), "lora_report")
    now = dt.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    output_dir = output_root / f"{timestamp}_{run_name}"

    prompt_path = resolve_path(config.get("prompt_file"), config_dir)
    if prompt_path is None:
        raise ValueError("prompt_file is required")

    prompts = parse_prompt_file(prompt_path)
    seeds = build_seeds(config.get("seeds", {}))
    conditions = build_lora_conditions(config, config_dir)
    gen_config = config.get("sdxl_gen_img", {})
    base_width = gen_config.get("width")
    base_height = gen_config.get("height")
    validate_image_size(base_width, base_height, "sdxl_gen_img")

    jobs = []
    seen_target_paths = {}
    for prompt_index, prompt in enumerate(prompts, 1):
        width = prompt.get("width") or base_width
        height = prompt.get("height") or base_height
        validate_image_size(width, height, f"{prompt_path}:{prompt['line_no']} ({prompt['id']})")
        for seed in seeds:
            for condition in conditions:
                condition_dir = output_dir / "images" / condition["id"]
                target_path = condition_dir / make_image_name(prompt, prompt_index, seed)
                path_key = target_path_key(target_path)
                if path_key in seen_target_paths:
                    other = seen_target_paths[path_key]
                    raise ValueError(
                        "Multiple report jobs resolve to the same image path: "
                        f"{target_path}. "
                        f"First: condition='{other['condition_id']}', prompt='{other['prompt_id']}', seed={other['seed']}. "
                        f"Second: condition='{condition['id']}', prompt='{prompt['id']}', seed={seed}. "
                        "Use unique LoRA condition ids/names and avoid duplicate seeds for the same prompt."
                    )
                seen_target_paths[path_key] = {
                    "condition_id": condition["id"],
                    "prompt_id": prompt["id"],
                    "seed": seed,
                }
                jobs.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt_id": prompt["id"],
                        "prompt": prompt["prompt"],
                        "negative": prompt.get("negative", ""),
                        "width": width,
                        "height": height,
                        "seed": seed,
                        "condition_id": condition["id"],
                        "condition_name": condition["name"],
                        "condition_items": condition["items"],
                        "image": str(target_path.relative_to(output_dir)).replace("\\", "/"),
                        "target_path": str(target_path),
                        "returncode": None,
                        "status": "pending",
                    }
                )

    return output_dir, prompts, conditions, seeds, jobs


def write_metadata(output_dir: Path, prompts: list[dict], conditions: list[dict], seeds: list[int], jobs: list[dict]):
    serializable_jobs = []
    for job in jobs:
        item = dict(job)
        item.pop("target_path", None)
        serializable_jobs.append(item)
    metadata = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "prompts": prompts,
        "conditions": conditions,
        "seeds": seeds,
        "jobs": serializable_jobs,
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def json_for_script(data: dict) -> str:
    return (
        json.dumps(data, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def write_report(output_dir: Path, metadata: dict):
    data_json = json_for_script(metadata)
    report = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDXL LoRA Report</title>
<style>
:root {{ color-scheme: light dark; --image-width: 320px; --card-width: calc(var(--image-width) + 18px); }}
body {{ margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f5f5f2; color: #202124; }}
header {{ position: sticky; top: 0; z-index: 3; background: #ffffffee; border-bottom: 1px solid #d8d8d0; padding: 12px 18px; backdrop-filter: blur(8px); }}
h1 {{ font-size: 18px; margin: 0 0 10px; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }}
.panel {{ padding: 12px 18px; border-bottom: 1px solid #ddd; background: #fbfbf8; }}
.filter-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.filter-box {{ max-height: 150px; overflow: auto; border: 1px solid #d6d6ce; padding: 8px; background: white; }}
.filter-box label {{ display: block; margin: 3px 0; white-space: nowrap; }}
main {{ padding: 18px; }}
.group {{ margin: 0 0 18px; background: white; border: 1px solid #d8d8d0; box-shadow: 0 1px 3px #0001; }}
.group h2 {{ margin: 0; padding: 10px 12px; font-size: 13px; background: #ecece6; border-bottom: 1px solid #d8d8d0; }}
.cards {{ display: flex; flex-wrap: wrap; align-items: flex-start; gap: 14px; padding: 14px; }}
.card {{ width: var(--card-width); border: 1px solid #d8d8d0; background: #fbfbf8; padding: 8px; box-sizing: border-box; }}
.card-title {{ width: var(--image-width); margin: 0 0 6px; font-size: 12px; font-weight: 650; overflow-wrap: anywhere; }}
.card img {{ width: var(--image-width); max-width: none; height: auto; display: block; cursor: zoom-in; border: 1px solid #ccc; background: #eee; box-sizing: border-box; }}
.missing {{ width: var(--image-width); min-height: 120px; display: grid; place-items: center; border: 1px dashed #aaa; color: #777; font-size: 12px; box-sizing: border-box; }}
.meta {{ margin-top: 6px; font-size: 11px; color: #555; line-height: 1.35; }}
dialog {{ max-width: 96vw; max-height: 96vh; border: 0; padding: 0; background: transparent; }}
.viewer-card {{ margin: 0; color: #eee; }}
#viewerTitle {{ padding: 8px 10px; font-size: 13px; font-weight: 650; background: #111; overflow-wrap: anywhere; }}
dialog img {{ max-width: 96vw; max-height: 88vh; display: block; background: #111; }}
#viewerMeta {{ padding: 7px 10px; font-size: 12px; background: #111; color: #ccc; }}
dialog::backdrop {{ background: rgba(0,0,0,.78); }}
button, select, input {{ font: inherit; }}
@media (prefers-color-scheme: dark) {{
  body {{ background: #1f211f; color: #eee; }}
  header, .panel, .group, .card, .filter-box {{ background: #282b28; }}
  .group h2 {{ background: #343832; }}
  header, .panel, .filter-box, .group, .group h2, .card {{ border-color: #474b43; }}
  .meta {{ color: #bbb; }}
}}
</style>
</head>
<body>
<header>
  <h1>SDXL LoRA Report</h1>
  <div class="toolbar">
    <label>Axis <select id="axis"><option value="condition">X: LoRA / Y: Prompt+Seed</option><option value="case">X: Prompt+Seed / Y: LoRA</option></select></label>
    <label>Image size <input id="size" type="range" min="160" max="768" step="16" value="320"> <span id="sizeLabel">320px</span></label>
    <button id="showAll">Show all</button>
  </div>
</header>
<section class="panel">
  <div class="filter-grid">
    <div><strong>LoRA</strong><div id="conditionFilters" class="filter-box"></div></div>
    <div><strong>Prompt</strong><div id="promptFilters" class="filter-box"></div></div>
    <div><strong>Seed</strong><div id="seedFilters" class="filter-box"></div></div>
  </div>
</section>
<main id="report"></main>
<dialog id="viewer"><figure class="viewer-card"><figcaption id="viewerTitle"></figcaption><img id="viewerImage" alt=""><div id="viewerMeta"></div></figure></dialog>
<script>
const reportData = {data_json};
const state = {{ axis: "condition", conditions: new Set(), prompts: new Set(), seeds: new Set(), size: 320 }};
let renderedJobs = [];
let renderedJobMap = new Map();
let viewerJobs = [];
let viewerIndex = -1;
const byId = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({{"&":"&amp;","<":"&lt;",">":"&gt;","\\"":"&quot;","'":"&#39;"}}[ch]));
const promptPosition = job => `p${{String(job.prompt_index ?? 0).padStart(4, "0")}}`;
const caseId = job => `${{promptPosition(job)}} / ${{job.prompt_id}} / seed ${{job.seed}}`;
const jobKey = job => `${{job.condition_id}}@@${{caseId(job)}}`;
const conditionOrder = new Map(reportData.conditions.map((condition, index) => [condition.id, index]));
function initState() {{
  reportData.conditions.forEach(c => state.conditions.add(c.id));
  reportData.prompts.forEach(p => state.prompts.add(p.id));
  reportData.seeds.forEach(s => state.seeds.add(String(s)));
}}
function checkboxList(root, items, selected, labelFn) {{
  root.innerHTML = items.map(item => {{
    const id = String(item.id ?? item);
    return `<label><input type="checkbox" data-id="${{esc(id)}}" checked> ${{esc(labelFn(item))}}</label>`;
  }}).join("");
  root.querySelectorAll("input").forEach(input => {{
    input.addEventListener("change", () => {{
      if (input.checked) selected.add(input.dataset.id); else selected.delete(input.dataset.id);
      render();
    }});
  }});
}}
function selectedJobs() {{
  return reportData.jobs.filter(j => state.conditions.has(j.condition_id) && state.prompts.has(j.prompt_id) && state.seeds.has(String(j.seed)));
}}
function card(job, title) {{
  if (!job) return `<article class="card"><div class="card-title">${{esc(title || "missing")}}</div><div class="missing">no job</div></article>`;
  const image = job.status === "done"
    ? `<img src="${{esc(job.image)}}" alt="" loading="lazy" data-job-key="${{esc(jobKey(job))}}">`
    : `<div class="missing">${{esc(job.status)}}</div>`;
  const items = (job.condition_items || []).map(item => `${{esc(item.name || item.path)}} x${{esc(item.strength)}} lbw=${{esc(item.lbw ?? "")}}`).join("<br>");
  return `<article class="card"><div class="card-title">${{esc(title || job.condition_name)}}</div>${{image}}<div class="meta">${{items}}<br>${{esc(caseId(job))}}<br>${{esc(job.width)}}x${{esc(job.height)}}</div></article>`;
}}
function setImageSize(value) {{
  state.size = Number(value);
  document.documentElement.style.setProperty("--image-width", `${{state.size}}px`);
  byId("sizeLabel").textContent = `${{state.size}}px`;
}}
function openViewer(key) {{
  const job = renderedJobMap.get(key);
  if (!job) return;
  viewerJobs = renderedJobs
    .filter(item => item.status === "done" && caseId(item) === caseId(job))
    .sort((a, b) => (conditionOrder.get(a.condition_id) ?? 9999) - (conditionOrder.get(b.condition_id) ?? 9999));
  viewerIndex = Math.max(0, viewerJobs.findIndex(item => jobKey(item) === key));
  updateViewer();
  byId("viewer").showModal();
}}
function updateViewer() {{
  const job = viewerJobs[viewerIndex];
  if (!job) return;
  byId("viewerImage").src = job.image;
  byId("viewerTitle").textContent = job.condition_name || "";
  byId("viewerMeta").textContent = `${{caseId(job)}} / ${{job.width}}x${{job.height}}`;
}}
function moveViewer(delta) {{
  if (!byId("viewer").open || viewerJobs.length === 0) return;
  viewerIndex = (viewerIndex + delta + viewerJobs.length) % viewerJobs.length;
  updateViewer();
}}
function render() {{
  setImageSize(state.size);
  const jobs = selectedJobs();
  renderedJobs = jobs;
  renderedJobMap = new Map(jobs.map(j => [jobKey(j), j]));
  const report = byId("report");
  const axis = state.axis;
  const jobMap = renderedJobMap;
  const cases = Array.from(new Set(jobs.map(caseId)));
  const conditions = reportData.conditions.filter(c => state.conditions.has(c.id));
  let html = "";
  if (axis === "condition") {{
    for (const caze of cases) {{
      html += `<section class="group"><h2>${{esc(caze)}}</h2><div class="cards">`;
      for (const condition of conditions) html += card(jobMap.get(`${{condition.id}}@@${{caze}}`), condition.name);
      html += "</div></section>";
    }}
  }} else {{
    for (const condition of conditions) {{
      html += `<section class="group"><h2>${{esc(condition.name)}}</h2><div class="cards">`;
      for (const caze of cases) html += card(jobMap.get(`${{condition.id}}@@${{caze}}`), condition.name);
      html += "</div></section>";
    }}
  }}
  report.innerHTML = html;
  report.querySelectorAll("img").forEach(img => img.addEventListener("click", () => {{
    openViewer(img.dataset.jobKey);
  }}));
}}
initState();
checkboxList(byId("conditionFilters"), reportData.conditions, state.conditions, item => item.name);
checkboxList(byId("promptFilters"), reportData.prompts, state.prompts, item => item.id);
checkboxList(byId("seedFilters"), reportData.seeds, state.seeds, item => item);
byId("axis").addEventListener("change", event => {{ state.axis = event.target.value; render(); }});
byId("size").addEventListener("input", event => {{ setImageSize(event.target.value); render(); }});
byId("showAll").addEventListener("click", () => location.reload());
byId("viewer").addEventListener("click", () => byId("viewer").close());
document.addEventListener("keydown", event => {{
  if (!byId("viewer").open) return;
  if (event.key === "ArrowLeft") {{ event.preventDefault(); moveViewer(-1); }}
  if (event.key === "ArrowRight") {{ event.preventDefault(); moveViewer(1); }}
}});
render();
</script>
</body>
</html>
"""
    with (output_dir / "report.html").open("w", encoding="utf-8") as f:
        f.write(report)


def write_blind_report(output_dir: Path, metadata: dict):
    data_json = json_for_script(metadata)
    report = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SDXL LoRA Blind Report</title>
<style>
:root { color-scheme: light dark; --image-width: 320px; --card-width: calc(var(--image-width) + 22px); }
body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; background: #f5f5f2; color: #202124; }
header { position: sticky; top: 0; z-index: 3; background: #ffffffee; border-bottom: 1px solid #d8d8d0; padding: 12px 18px; backdrop-filter: blur(8px); }
h1 { font-size: 18px; margin: 0 0 10px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; font-size: 13px; }
main { padding: 18px; }
.group { margin: 0 0 22px; background: white; border: 1px solid #d8d8d0; box-shadow: 0 1px 3px #0001; }
.group h2 { margin: 0; padding: 10px 12px; font-size: 13px; background: #ecece6; border-bottom: 1px solid #d8d8d0; }
.choices { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 14px; padding: 14px; }
.choice { width: var(--card-width); border: 1px solid #d8d8d0; background: #fbfbf8; padding: 10px; box-sizing: border-box; }
.choice img { width: var(--image-width); max-width: none; height: auto; display: block; cursor: zoom-in; border: 1px solid #ccc; background: #eee; box-sizing: border-box; }
.choice label { display: inline-flex; gap: 7px; align-items: center; margin-top: 8px; font-weight: 650; }
.missing { width: var(--image-width); min-height: 140px; display: grid; place-items: center; border: 1px dashed #aaa; color: #777; font-size: 12px; box-sizing: border-box; }
.answer { display: none; margin-top: 7px; font-size: 12px; color: #555; line-height: 1.35; }
body.revealed .answer { display: block; }
#results { margin: 0 18px 22px; padding: 12px; border: 1px solid #d8d8d0; background: white; display: none; }
#results table { border-collapse: collapse; }
#results th, #results td { border: 1px solid #d8d8d0; padding: 6px 10px; text-align: left; }
dialog { max-width: 96vw; max-height: 96vh; border: 0; padding: 0; background: transparent; }
.viewer-card { margin: 0; color: #eee; }
#viewerTitle { padding: 8px 10px; font-size: 13px; font-weight: 650; background: #111; overflow-wrap: anywhere; }
dialog img { max-width: 96vw; max-height: 88vh; display: block; background: #111; }
dialog::backdrop { background: rgba(0,0,0,.78); }
button, input { font: inherit; }
@media (prefers-color-scheme: dark) {
  body { background: #1f211f; color: #eee; }
  header, .group, .choice, #results { background: #282b28; }
  .group h2 { background: #343832; }
  header, .group, .group h2, .choice, #results, #results th, #results td { border-color: #474b43; }
  .answer { color: #bbb; }
}
</style>
</head>
<body>
<header>
  <h1>SDXL LoRA Blind Report</h1>
  <div class="toolbar">
    <label>Image size <input id="size" type="range" min="160" max="768" step="16" value="320"> <span id="sizeLabel">320px</span></label>
    <button id="reveal">Reveal / 答え合わせ</button>
  </div>
</header>
<section id="results"></section>
<main id="report"></main>
<dialog id="viewer"><figure class="viewer-card"><figcaption id="viewerTitle"></figcaption><img id="viewerImage" alt=""></figure></dialog>
<script>
const reportData = __REPORT_DATA__;
let renderedGroups = [];
let viewerJobs = [];
let viewerIndex = -1;
const byId = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => {
  if (ch === "&") return "&amp;";
  if (ch === "<") return "&lt;";
  if (ch === ">") return "&gt;";
  if (ch === '"') return "&quot;";
  return "&#39;";
});
const promptPosition = job => `p${String(job.prompt_index ?? 0).padStart(4, "0")}`;
const caseId = job => `${promptPosition(job)} / ${job.prompt_id} / seed ${job.seed}`;
function shuffle(items) {
  const result = [...items];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
function conditionLabel(job) {
  const items = (job.condition_items || []).map(item => `${esc(item.name || item.path)} x${esc(item.strength)} lbw=${esc(item.lbw ?? "")}`).join("<br>");
  return `${esc(job.condition_name)}${items ? "<br>" + items : ""}`;
}
function buildGroups() {
  const groups = new Map();
  for (const job of reportData.jobs) {
    if (!groups.has(caseId(job))) groups.set(caseId(job), []);
    groups.get(caseId(job)).push(job);
  }
  return [...groups.entries()].map(([label, jobs], index) => ({ label, index, jobs: shuffle(jobs) }));
}
function choice(job, groupIndex, choiceIndex) {
  const image = job.status === "done"
    ? `<img src="${esc(job.image)}" alt="" loading="lazy" data-group-index="${groupIndex}" data-choice-index="${choiceIndex}">`
    : `<div class="missing">${esc(job.status)}</div>`;
  return `<article class="choice" data-condition="${esc(job.condition_id)}">
    ${image}
    <label><input type="radio" name="best_${groupIndex}" value="${esc(job.condition_id)}"> Best</label>
    <div class="answer">#${choiceIndex + 1}<br>${conditionLabel(job)}</div>
  </article>`;
}
function setImageSize(value) {
  document.documentElement.style.setProperty("--image-width", `${value}px`);
  byId("sizeLabel").textContent = `${value}px`;
}
function openViewer(groupIndex, choiceIndex) {
  const group = renderedGroups[groupIndex];
  if (!group) return;
  viewerJobs = group.jobs.filter(job => job.status === "done");
  const clickedJob = group.jobs[choiceIndex];
  viewerIndex = Math.max(0, viewerJobs.findIndex(job => job === clickedJob));
  updateViewer();
  byId("viewer").showModal();
}
function updateViewer() {
  const job = viewerJobs[viewerIndex];
  if (!job) return;
  byId("viewerImage").src = job.image;
  byId("viewerTitle").textContent = caseId(job);
}
function moveViewer(delta) {
  if (!byId("viewer").open || viewerJobs.length === 0) return;
  viewerIndex = (viewerIndex + delta + viewerJobs.length) % viewerJobs.length;
  updateViewer();
}
function render() {
  setImageSize(byId("size").value);
  renderedGroups = buildGroups();
  byId("report").innerHTML = renderedGroups.map(group => `
    <section class="group">
      <h2>${esc(group.label)}</h2>
      <div class="choices">${group.jobs.map((job, index) => choice(job, group.index, index)).join("")}</div>
    </section>
  `).join("");
  byId("report").querySelectorAll("img").forEach(img => img.addEventListener("click", () => {
    openViewer(Number(img.dataset.groupIndex), Number(img.dataset.choiceIndex));
  }));
}
function reveal() {
  document.body.classList.add("revealed");
  const votes = new Map(reportData.conditions.map(condition => [condition.id, { name: condition.name, count: 0 }]));
  document.querySelectorAll('input[type="radio"]:checked').forEach(input => {
    if (votes.has(input.value)) votes.get(input.value).count += 1;
  });
  const rows = [...votes.values()].sort((a, b) => b.count - a.count || a.name.localeCompare(b.name));
  byId("results").style.display = "block";
  byId("results").innerHTML = `<strong>Vote results</strong>
    <table><thead><tr><th>LoRA</th><th>Votes</th></tr></thead><tbody>
    ${rows.map(row => `<tr><td>${esc(row.name)}</td><td>${row.count}</td></tr>`).join("")}
    </tbody></table>`;
}
byId("size").addEventListener("input", event => {
  setImageSize(event.target.value);
});
byId("reveal").addEventListener("click", reveal);
byId("viewer").addEventListener("click", () => byId("viewer").close());
document.addEventListener("keydown", event => {
  if (!byId("viewer").open) return;
  if (event.key === "ArrowLeft") { event.preventDefault(); moveViewer(-1); }
  if (event.key === "ArrowRight") { event.preventDefault(); moveViewer(1); }
});
render();
</script>
</body>
</html>
""".replace("__REPORT_DATA__", data_json)
    with (output_dir / "blind_report.html").open("w", encoding="utf-8") as f:
        f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Generate SDXL LoRA comparison images and an HTML report.")
    parser.add_argument("--config", required=True, help="Path to report JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Write metadata/report without running image generation.")
    args = parser.parse_args()

    try:
        run(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


def run(args):
    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    output_dir, prompts, conditions, seeds, jobs = generate_jobs(config_path, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = resolve_path(config.get("prompt_file"), config_path.parent)
    copy_run_inputs(config_path, prompt_path, output_dir, config, prompts)

    print(f"Output: {output_dir}")
    print(f"Jobs: {len(jobs)}")
    gen_config = config.get("sdxl_gen_img", {})
    print(f"[1/1] worker ({len(jobs)} jobs, {len(conditions)} conditions)")
    returncode = run_report_worker(output_dir, gen_config, conditions, jobs, args.dry_run)
    if returncode != 0:
        metadata = write_metadata(output_dir, prompts, conditions, seeds, jobs)
        write_report(output_dir, metadata)
        write_blind_report(output_dir, metadata)
        print_report_paths(output_dir)
        raise SystemExit(returncode)

    metadata = write_metadata(output_dir, prompts, conditions, seeds, jobs)
    write_report(output_dir, metadata)
    write_blind_report(output_dir, metadata)
    print_report_paths(output_dir)


def print_report_paths(output_dir: Path):
    print(f"Report: {output_dir / 'report.html'}")
    print(f"Blind report: {output_dir / 'blind_report.html'}")


if __name__ == "__main__":
    main()
