#!/usr/bin/env python
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from library.generation_lora_strengths import flatten_strength_specs, normalize_strength_spec


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DEFAULT_LBW = "ALL"
REPORT_LORA_MODULE = "networks.lora_lbw"
BLOCKED_COMMON_ARGS = {"--network_merge", "--network_merge_n_models"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_arg(command: list[str], option: str, value):
    if value is not None:
        command.extend([option, str(value)])


def list_images(directory: Path) -> set[Path]:
    if not directory.exists():
        return set()
    return {p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}


def normalize_lora_slot_item(item: dict) -> dict:
    normalized = dict(item)
    module = normalized.get("module")
    lbw = normalized.get("lbw")
    if module in (None, "networks.lora", REPORT_LORA_MODULE):
        normalized["module"] = REPORT_LORA_MODULE
        normalized["lbw"] = lbw or DEFAULT_LBW
    return normalized


def lora_slot_key(item: dict) -> tuple[str, str, str | None]:
    normalized = normalize_lora_slot_item(item)
    return (normalized["module"], normalized["path"], normalized.get("lbw"))


def build_lora_slots(conditions: list[dict]) -> list[dict]:
    slots = []
    slot_by_key = {}
    for condition in conditions:
        for item in condition["items"]:
            key = lora_slot_key(item)
            if key in slot_by_key:
                continue
            slot_by_key[key] = len(slots)
            slots.append(normalize_lora_slot_item(item))
    return slots


def build_lora_slots_from_jobs(jobs: list[dict]) -> list[dict]:
    return build_lora_slots([{"items": job.get("condition_items", [])} for job in jobs])


def format_multiplier(value: float) -> str:
    text = f"{float(value):.12f}".rstrip("0").rstrip(".")
    return text or "0"


def prompt_line(job: dict, slots: list[dict]) -> str:
    text = job["prompt"]
    if job.get("negative"):
        text += f" --n {job['negative']}"
    text += f" --d {job['seed']}"
    if job.get("width"):
        text += f" --w {job['width']}"
    if job.get("height"):
        text += f" --h {job['height']}"

    if slots:
        strength_specs = [(0.0,)] * len(slots)
        slot_by_key = {lora_slot_key(slot): index for index, slot in enumerate(slots)}
        for item in job["condition_items"]:
            strength_specs[slot_by_key[lora_slot_key(item)]] = normalize_strength_spec(item["strength"])
        multipliers = flatten_strength_specs(strength_specs)
        text += " --am " + ",".join(format_multiplier(v) for v in multipliers)

    return text


def build_command(script_dir: Path, job_plan: dict, prompt_file: Path, outdir: Path, slots: list[dict]) -> list[str]:
    gen_config = job_plan["sdxl_gen_img"]
    command = [sys.executable, str(script_dir / "sdxl_gen_img.py")]

    append_arg(command, "--ckpt", gen_config.get("ckpt"))
    append_arg(command, "--vae", gen_config.get("vae"))
    command.extend(["--outdir", str(outdir)])
    append_arg(command, "--W", gen_config.get("width"))
    append_arg(command, "--H", gen_config.get("height"))
    append_arg(command, "--steps", gen_config.get("steps"))
    append_arg(command, "--sampler", gen_config.get("sampler"))
    append_arg(command, "--scale", gen_config.get("scale"))
    append_arg(command, "--batch_size", gen_config.get("batch_size", 1))
    append_arg(command, "--images_per_prompt", gen_config.get("images_per_prompt", 1))

    common_args = gen_config.get("common_args", [])
    if not isinstance(common_args, list):
        raise ValueError("sdxl_gen_img.common_args must be a list")
    for arg in common_args:
        option = str(arg).split("=", 1)[0]
        if option in BLOCKED_COMMON_ARGS:
            raise ValueError(
                f"{option} cannot be used in sdxl_lora_report common_args because it merges LoRA weights "
                "and prevents per-job LoRA switching."
            )
    command.extend(str(arg) for arg in common_args)

    if slots:
        command.append("--network_module")
        command.extend(slot["module"] for slot in slots)
        command.append("--network_weights")
        command.extend(slot["path"] for slot in slots)
        command.append("--network_mul")
        command.extend("1.0" for _ in slots)
        if any(slot.get("lbw") is not None for slot in slots):
            command.append("--network_lbw")
            command.extend(str(slot.get("lbw") or DEFAULT_LBW) for slot in slots)

    command.extend(["--from_file", str(prompt_file), "--sequential_file_name"])
    return command


def select_job_output_images(created: list[Path], job_count: int) -> tuple[list[Path], str | None]:
    highres_final = [path for path in created if path.name.startswith("im_1")]
    if len(highres_final) == job_count:
        return highres_final, None
    if len(created) == job_count:
        return created, None
    if len(created) < job_count:
        return [], f"expected {job_count} images but found {len(created)}"
    if highres_final:
        return [], f"expected {job_count} final highres images but found {len(highres_final)} final images and {len(created)} total images"
    return [], f"expected {job_count} images but found {len(created)}; extra images make output pairing ambiguous"


def move_outputs(outdir: Path, before: set[Path], jobs: list[dict]) -> list[dict]:
    after = list_images(outdir)
    created = sorted(after - before, key=lambda p: p.name)
    images, error = select_job_output_images(created, len(jobs))
    if error is not None:
        return [
            {
                "job_index": job["job_index"],
                "status": "missing_image",
                "output": job["target_path"],
                "error": error,
            }
            for job in jobs
        ]

    results = []
    for image, job in zip(images, jobs):
        target = Path(job["target_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        shutil.move(str(image), str(target))
        results.append({"job_index": job["job_index"], "status": "done", "output": str(target), "error": None})
    return results


def run_worker(job_plan: dict, dry_run: bool) -> dict:
    script_dir = Path(__file__).resolve().parent
    gen_config = job_plan["sdxl_gen_img"]
    if int(gen_config.get("images_per_prompt", 1)) != 1:
        raise ValueError("LoRA report worker currently requires images_per_prompt=1")

    outdir = Path(job_plan["work_outdir"]).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = job_plan["jobs"]
    slots = build_lora_slots_from_jobs(jobs)

    with tempfile.TemporaryDirectory(prefix="lora_report_worker_") as tmpdir:
        prompt_file = Path(tmpdir) / "prompts.txt"
        with prompt_file.open("w", encoding="utf-8") as f:
            for job in jobs:
                f.write(prompt_line(job, slots) + "\n")

        command = build_command(script_dir, job_plan, prompt_file, outdir, slots)
        if dry_run:
            return {
                "status": "dry_run",
                "returncode": 0,
                "command": command,
                "slots": slots,
                "results": [
                    {"job_index": job["job_index"], "status": "dry_run", "output": job["target_path"], "error": None}
                    for job in jobs
                ],
            }

        before = list_images(outdir)
        result = subprocess.run(command, cwd=script_dir)
        if result.returncode != 0:
            return {
                "status": "failed",
                "returncode": result.returncode,
                "command": command,
                "slots": slots,
                "results": [
                    {"job_index": job["job_index"], "status": "failed", "output": job["target_path"], "error": "worker command failed"}
                    for job in jobs
                ],
            }

    results = move_outputs(outdir, before, jobs)
    status = "done" if all(r["status"] == "done" for r in results) else "missing_image"
    returncode = 0 if status == "done" else 1
    return {"status": status, "returncode": returncode, "command": command, "slots": slots, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Generate all LoRA report conditions in one sdxl_gen_img.py process.")
    parser.add_argument("--job-json", required=True, help="Worker job JSON path.")
    parser.add_argument("--result-json", required=True, help="Worker result JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Do not run image generation.")
    args = parser.parse_args()

    job_plan_path = Path(args.job_json).resolve()
    result_path = Path(args.result_json).resolve()
    job_plan = load_json(job_plan_path)
    try:
        result = run_worker(job_plan, args.dry_run)
    except Exception as exc:
        result = {
            "status": "failed",
            "returncode": 1,
            "command": [],
            "slots": [],
            "error": str(exc),
            "results": [
                {
                    "job_index": job["job_index"],
                    "status": "failed",
                    "output": job["target_path"],
                    "error": str(exc),
                }
                for job in job_plan.get("jobs", [])
            ],
        }
    write_json(result_path, result)
    raise SystemExit(int(result.get("returncode", 1)))


if __name__ == "__main__":
    main()
