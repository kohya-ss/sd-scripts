from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import toml

from dq_profile.production_cli import ResolvedProfileRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
PROFILE_SCRIPT = REPO_ROOT / "sdxl_dq_dataset_profile.py"
DEFAULT_OUTPUT_BASE = PROJECT_ROOT / "lora_output" / "dq_dataset_profiler"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".avif", ".jxl"}
RUN_SCHEMA_VERSION = "1.0-beta"
INVALID_FILENAME = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def console_write(text: str) -> None:
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe)
    sys.stdout.flush()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def git_tracked_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return bool(result.stdout.strip())


def sanitize_profile_name(value: str) -> str:
    name = INVALID_FILENAME.sub("_", str(value).strip()).strip(" .")
    name = re.sub(r"\s+", "_", name)
    if not name or name in {".", ".."}:
        name = "dq_profile"
    if name.upper() in WINDOWS_RESERVED:
        name = f"dq_{name}"
    return name[:80].rstrip(" .") or "dq_profile"


def source_dirs_from_dataset_config(dataset_config: Path) -> tuple[Path, ...]:
    payload = toml.load(dataset_config)
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("dataset TOML must contain at least one [[datasets]] entry")
    source_dirs: list[Path] = []
    seen: set[str] = set()
    for dataset_index, dataset in enumerate(datasets):
        subsets = dataset.get("subsets") if isinstance(dataset, dict) else None
        if not isinstance(subsets, list) or not subsets:
            raise ValueError(f"datasets[{dataset_index}] has no [[datasets.subsets]] entries")
        for subset_index, subset in enumerate(subsets):
            image_dir = subset.get("image_dir") if isinstance(subset, dict) else None
            if not image_dir:
                raise ValueError(
                    f"datasets[{dataset_index}].subsets[{subset_index}] has no image_dir; "
                    "metadata-only datasets are not supported by canonical-v1"
                )
            source_dir = Path(str(image_dir)).expanduser()
            if not source_dir.is_absolute():
                source_dir = dataset_config.parent / source_dir
            source_dir = source_dir.resolve()
            if not source_dir.is_dir():
                raise FileNotFoundError(f"dataset image_dir was not found: {source_dir}")
            folded = str(source_dir).casefold()
            if folded in seen:
                raise ValueError(f"duplicate image_dir is ambiguous for source bootstrap: {source_dir}")
            for existing in source_dirs:
                if existing in source_dir.parents or source_dir in existing.parents:
                    raise ValueError(f"nested image_dir values are ambiguous: {existing} and {source_dir}")
            seen.add(folded)
            source_dirs.append(source_dir)
    return tuple(source_dirs)


def _inventory(source_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(source_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file():
            continue
        rows.append(
            {
                "name": str(path.relative_to(source_dir)).replace("\\", "/"),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "is_image": path.suffix.casefold() in IMAGE_SUFFIXES,
            }
        )
    return rows


def build_source_map(source_dirs: Sequence[Path], *, dataset_key: str) -> tuple[list[dict[str, Any]], int]:
    payload: list[dict[str, Any]] = []
    for index, source_dir in enumerate(source_dirs, start=1):
        files = _inventory(source_dir)
        payload.append(
            {
                "pattern": str(source_dir.resolve()) + os.sep,
                "source_group": f"{dataset_key}-source-{index:02d}",
                "match": "prefix",
                "directory": str(source_dir.resolve()),
                "files": files,
                "image_count": sum(bool(row["is_image"]) for row in files),
            }
        )
    image_count = sum(int(row["image_count"]) for row in payload)
    if image_count < 8:
        raise ValueError(f"canonical-v1 requires at least 8 images; found {image_count}")
    return payload, image_count


def _is_within(path: Path, parent: Path) -> bool:
    path = path.resolve()
    parent = parent.resolve()
    return path == parent or parent in path.parents


def validate_output_base(
    output_base: Path,
    *,
    source_dirs: Sequence[Path],
    normal_output_dir: Path | None,
) -> Path:
    output_base = output_base.expanduser().resolve()
    if output_base.parent == output_base or output_base == PROJECT_ROOT.resolve():
        raise ValueError(f"diagnostic output base is too broad: {output_base}")
    forbidden = (REPO_ROOT, REPO_ROOT / ".git", REPO_ROOT / "venv")
    if any(_is_within(output_base, item) for item in forbidden):
        raise ValueError(f"diagnostic output must stay outside the repository/.git/venv: {output_base}")
    for source_dir in source_dirs:
        if _is_within(output_base, source_dir) or _is_within(source_dir, output_base):
            raise ValueError(f"diagnostic output must not overlap dataset image_dir: {source_dir}")
    if normal_output_dir is not None and output_base == normal_output_dir.resolve():
        raise ValueError(
            "diagnostic output base cannot be the exact normal checkpoint output directory; "
            "use its dq_dataset_profiler child or another directory"
        )
    output_base.mkdir(parents=True, exist_ok=True)
    probe = output_base / ".dq_profile_write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()
    return output_base


def _model_identity(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"pretrained model path was not found: {path}")
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "kind": "directory" if path.is_dir() else "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def build_protocol_fingerprint(
    request: ResolvedProfileRequest,
    *,
    source_map_payload: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    code_files = (
        Path(__file__).resolve(),
        REPO_ROOT / "dq_profile" / "production_cli.py",
        REPO_ROOT / "dq_profile" / "production_preset.py",
        PROFILE_SCRIPT,
        REPO_ROOT / "tools" / "analyze_dq_v24_local.py",
        REPO_ROOT / "tools" / "check_dq_calibration_gate.py",
    )
    payload = {
        "schema_version": RUN_SCHEMA_VERSION,
        "git_head": git_head(),
        "git_tracked_dirty": git_tracked_dirty(),
        "preset": request.preset.contract(),
        "model": _model_identity(request.model_path),
        "dataset_config": {
            "path": str(request.dataset_config),
            "sha256": sha256_file(request.dataset_config),
        },
        "source_map_sha256": canonical_sha256(source_map_payload),
        "code": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path)
            for path in code_files
        },
        "scope": "local_body_tail_only",
        "trajectory": "research_only_not_run",
        "not_quality_or_utility": True,
    }
    return {**payload, "fingerprint_sha256": canonical_sha256(payload)}


def allocate_run_directory(output_base: Path, profile_name: str, fingerprint: str) -> Path:
    profile_root = output_base / sanitize_profile_name(profile_name)
    profile_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{timestamp}_{fingerprint[:10]}"
    for index in range(1, 100):
        suffix = "" if index == 1 else f"_{index:02d}"
        candidate = profile_root / f"{stem}{suffix}"
        try:
            candidate.mkdir(parents=False, exist_ok=False)
            return candidate
        except FileExistsError:
            continue
    raise RuntimeError(f"could not allocate a unique diagnostic run directory under {profile_root}")


def is_complete(path: Path) -> bool:
    status_path = path / "status.json"
    return status_path.is_file() and read_json(status_path).get("status") == "complete"


class Launcher:
    def __init__(self, run_dir: Path, *, dry_run: bool = False) -> None:
        self.run_dir = run_dir
        self.dry_run = dry_run
        self.log_path = run_dir / "run.log"

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        console_write(line + "\n")
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")

    def run(self, command: Sequence[str], *, label: str) -> None:
        rendered = subprocess.list2cmdline([str(item) for item in command])
        self.log(f"RUN {label}: {rendered}")
        if self.dry_run:
            return
        environment = dict(os.environ)
        environment["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        with self.log_path.open("a", encoding="utf-8") as log_stream:
            for line in process.stdout:
                console_write(line)
                log_stream.write(line)
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"{label} failed with exit code {return_code}")
        self.log(f"DONE {label}")


@dataclass(frozen=True)
class ProductionRunOptions:
    output_base: Path = DEFAULT_OUTPUT_BASE
    profile_name: str | None = None
    preflight_only: bool = False
    dry_run: bool = False
    open_report: bool = False


@dataclass(frozen=True)
class ProductionRunResult:
    run_dir: Path
    status: str
    report: Path | None


def stage_names() -> dict[str, str]:
    return {
        "snapshot_a": "00a_snapshot_a",
        "snapshot_b": "00b_snapshot_b",
        "snapshot_parity": "00c_snapshot_parity.json",
        "prefix": "00d_prefix",
        "prefix_snapshot_parity": "00e_prefix_snapshot_parity.json",
        "prefix_source_gate": "00f_prefix_source_gate.json",
        "core_raw": "01_core_local_raw",
        "core_analysis": "02_core_local_analysis",
    }


def edge_names(round_index: int) -> dict[str, str]:
    return {
        "raw": f"{2 + round_index * 3:02d}_edge{round_index}_local_raw",
        "analysis": f"{3 + round_index * 3:02d}_edge{round_index}_local_analysis",
        "parity": f"{4 + round_index * 3:02d}_edge{round_index}_parity.json",
    }


def accelerate_prefix(preset: Any) -> list[str]:
    return [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        str(preset.num_cpu_threads_per_process),
        str(PROFILE_SCRIPT),
    ]


def profile_command(
    request: ResolvedProfileRequest,
    *,
    run_dir: Path,
    source_map: Path,
    name: str,
    protocol: str,
    range_muls: Sequence[float],
    max_images: int,
    gate: Path | None = None,
    snapshot_only: bool = False,
) -> list[str]:
    preset = request.preset
    command = [
        *accelerate_prefix(preset),
        f"--pretrained_model_name_or_path={request.model_path}",
        f"--dataset_config={request.dataset_config}",
        f"--output_dir={run_dir / '_internal_training_output'}",
        f"--output_name={request.output_name}",
        *preset.training_tokens,
        f"--training_comment=DQ Profiler Beta {preset.name} Local-only",
        f"--dq_profile_output_dir={run_dir}",
        f"--dq_profile_name={name}",
        "--dq_profile_level=standard",
        f"--dq_profile_protocol={protocol}",
        "--dq_profile_prefix_kernel_mode=deterministic",
        f"--dq_profile_max_images={max_images}",
        f"--dq_profile_timestep_bins={preset.timestep_bins}",
        f"--dq_profile_stochastic_repeats={preset.stochastic_repeats}",
        "--dq_profile_range_muls=" + ",".join(f"{value:.2f}" for value in range_muls),
        f"--dq_profile_sweep_steps={preset.sweep_steps}",
        f"--dq_profile_branch_repeats={preset.branch_repeats}",
        "--dq_profile_guardian_ablation=common_only",
        f"--dq_profile_sketch_width={preset.sketch_width}",
        f"--dq_profile_sketch_seeds={preset.sketch_seeds}",
        f"--dq_profile_source_group_map={source_map}",
    ]
    if gate is not None:
        command.append(f"--dq_profile_prefix_gate_file={gate}")
    if snapshot_only:
        command.append("--dq_profile_snapshot_only")
    return command


def run_profile(
    launcher: Launcher,
    request: ResolvedProfileRequest,
    *,
    source_map: Path,
    name: str,
    protocol: str,
    range_muls: Sequence[float],
    max_images: int,
    gate: Path | None = None,
    snapshot_only: bool = False,
) -> Path:
    output = launcher.run_dir / name
    if output.exists():
        raise RuntimeError(f"stage output already exists and will not be overwritten: {output}")
    launcher.run(
        profile_command(
            request,
            run_dir=launcher.run_dir,
            source_map=source_map,
            name=name,
            protocol=protocol,
            range_muls=range_muls,
            max_images=max_images,
            gate=gate,
            snapshot_only=snapshot_only,
        ),
        label=f"{protocol} / {name}",
    )
    if not launcher.dry_run and not is_complete(output):
        raise RuntimeError(f"profile returned success but stage status is not complete: {output}")
    return output


def run_snapshot_parity(launcher: Launcher, left: Path, right: Path, output_name: str) -> Path:
    output = launcher.run_dir / output_name
    launcher.run(
        [
            sys.executable,
            "-m",
            "tools.check_dq_snapshot_parity",
            "--left",
            str(left),
            "--right",
            str(right),
            "--output-json",
            str(output),
        ],
        label=f"snapshot parity {left.name} vs {right.name}",
    )
    if not launcher.dry_run and read_json(output).get("passed") is not True:
        raise RuntimeError(f"snapshot parity failed: {output}")
    return output


def run_local_analysis(
    launcher: Launcher,
    *,
    profile: Path,
    output_name: str,
    dataset_id: str,
) -> Path:
    output = launcher.run_dir / output_name
    if output.exists():
        raise RuntimeError(f"analysis output already exists and will not be overwritten: {output}")
    launcher.run(
        [
            sys.executable,
            "-m",
            "tools.analyze_dq_v24_local",
            "--profile-dir",
            str(profile),
            "--output-dir",
            str(output),
            "--dataset-id",
            dataset_id,
            "--iterations",
            "2000",
            "--seed",
            "2401",
        ],
        label=f"Local Body/Tail analysis / {output_name}",
    )
    if not launcher.dry_run and not is_complete(output):
        raise RuntimeError(f"Local analysis did not complete: {output}")
    return output


def update_status(run_dir: Path, *, status: str, current_stage: str, **extra: Any) -> None:
    existing = read_json(run_dir / "status.json") if (run_dir / "status.json").is_file() else {}
    write_json(
        run_dir / "status.json",
        {
            **existing,
            "schema_version": RUN_SCHEMA_VERSION,
            "status": status,
            "current_stage": current_stage,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "not_quality_or_utility": True,
            **extra,
        },
    )


def run_local_pipeline(
    launcher: Launcher,
    request: ResolvedProfileRequest,
    *,
    source_map: Path,
    probe_budget: int,
    dataset_id: str,
) -> tuple[Path, Path, tuple[float, ...], int]:
    names = stage_names()
    update_status(launcher.run_dir, status="running", current_stage="snapshot_a")
    snapshot_a = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["snapshot_a"],
        protocol="v2-prefix-smoke",
        range_muls=(3.15,),
        max_images=8,
        snapshot_only=True,
    )
    update_status(launcher.run_dir, status="running", current_stage="snapshot_b")
    snapshot_b = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["snapshot_b"],
        protocol="v2-prefix-smoke",
        range_muls=(3.15,),
        max_images=probe_budget,
        snapshot_only=True,
    )
    run_snapshot_parity(launcher, snapshot_a, snapshot_b, names["snapshot_parity"])

    update_status(launcher.run_dir, status="running", current_stage="prefix_gate")
    prefix = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["prefix"],
        protocol="v2-prefix-smoke",
        range_muls=(3.15,),
        max_images=probe_budget,
    )
    run_snapshot_parity(launcher, snapshot_b, prefix, names["prefix_snapshot_parity"])
    gate = prefix / "calibration_gate.json"
    launcher.run(
        [
            sys.executable,
            "-m",
            "tools.check_dq_calibration_gate",
            "--gate",
            str(gate),
            "--dataset-config",
            str(request.dataset_config),
            "--source-group-map",
            str(source_map),
            "--repo-root",
            str(REPO_ROOT),
            "--output-json",
            str(launcher.run_dir / names["prefix_source_gate"]),
        ],
        label="prefix/source contract gate",
    )

    update_status(launcher.run_dir, status="running", current_stage="core_local")
    active_grid = tuple(float(value) for value in request.preset.core_grid)
    active_profile = run_profile(
        launcher,
        request,
        source_map=source_map,
        name=names["core_raw"],
        protocol="v24-acceptance-local",
        range_muls=active_grid,
        max_images=probe_budget,
        gate=gate,
    )
    active_analysis = run_local_analysis(
        launcher,
        profile=active_profile,
        output_name=names["core_analysis"],
        dataset_id=dataset_id,
    )
    if launcher.dry_run:
        return active_profile, active_analysis, active_grid, 0

    selection = read_json(active_analysis / "local_selection.json")
    completed_edge_rounds = 0
    for round_index in range(1, request.preset.max_edge_extension_rounds + 1):
        if selection.get("selection_valid") is not True:
            break
        additions = tuple(float(value) for value in selection.get("edge_extension_recommended", ()))
        expanded_grid = tuple(sorted(set((*active_grid, *additions))))
        if not additions or expanded_grid == active_grid:
            break
        round_names = edge_names(round_index)
        update_status(
            launcher.run_dir,
            status="running",
            current_stage=f"edge_extension_{round_index}",
            measured_grid=list(expanded_grid),
        )
        edge_profile = run_profile(
            launcher,
            request,
            source_map=source_map,
            name=round_names["raw"],
            protocol="v24-acceptance-local",
            range_muls=expanded_grid,
            max_images=probe_budget,
            gate=gate,
        )
        edge_analysis = run_local_analysis(
            launcher,
            profile=edge_profile,
            output_name=round_names["analysis"],
            dataset_id=dataset_id,
        )
        launcher.run(
            [
                sys.executable,
                "-m",
                "tools.check_dq_v24_local_extension_parity",
                "--core-profile-dir",
                str(active_profile),
                "--extension-profile-dir",
                str(edge_profile),
                "--common-muls",
                ",".join(f"{value:.2f}" for value in active_grid),
                "--output-json",
                str(launcher.run_dir / round_names["parity"]),
            ],
            label=f"edge extension {round_index} shared-probe parity",
        )
        active_profile = edge_profile
        active_analysis = edge_analysis
        active_grid = expanded_grid
        completed_edge_rounds = round_index
        selection = read_json(active_analysis / "local_selection.json")
    return active_profile, active_analysis, active_grid, completed_edge_rounds


PROMOTED_ANALYSIS_FILES = (
    "report.html",
    "technical_report.html",
    "summary.json",
    "practical_report.json",
    "report_contract.json",
    "analysis_manifest.json",
    "local_selection.json",
    "local_acceptance.csv",
    "local_timestep.csv",
    "source_bootstrap.csv",
    "bootstrap_regret.csv",
    "robust_dominance.csv",
    "source_loo.csv",
    "natural_gradient_baseline.json",
    "acceptance_contract.json",
)


def promote_analysis(run_dir: Path, analysis_dir: Path) -> list[str]:
    promoted: list[str] = []
    for name in PROMOTED_ANALYSIS_FILES:
        source = analysis_dir / name
        if source.is_file():
            shutil.copy2(source, run_dir / name)
            promoted.append(name)
    for required in ("report.html", "technical_report.html", "summary.json", "practical_report.json"):
        if required not in promoted:
            raise FileNotFoundError(f"required product artifact was not generated: {analysis_dir / required}")
    return promoted


def promote_profile_provenance(run_dir: Path, profile_dir: Path) -> list[str]:
    mapping = {
        "source_manifest.json": "source_manifest.json",
        "candidate_definitions.json": "candidate_definitions.json",
        "probe_manifest.json": "probe_manifest.json",
        "calibration_gate.json": "profile_calibration_gate.json",
        "gradient_tail.csv": "raw_gradient_tail.csv",
        "local_natural_gradient.csv": "raw_local_natural_gradient.csv",
    }
    promoted: list[str] = []
    for source_name, target_name in mapping.items():
        source = profile_dir / source_name
        if source.is_file():
            shutil.copy2(source, run_dir / target_name)
            promoted.append(target_name)
    if "source_manifest.json" not in promoted:
        raise FileNotFoundError(f"active profile has no source_manifest.json: {profile_dir}")
    return promoted


def preflight(request: ResolvedProfileRequest, source_dirs: Sequence[Path]) -> None:
    required_files = (
        PROFILE_SCRIPT,
        REPO_ROOT / "tools" / "analyze_dq_v24_local.py",
        REPO_ROOT / "tools" / "check_dq_calibration_gate.py",
        REPO_ROOT / "tools" / "check_dq_snapshot_parity.py",
        REPO_ROOT / "tools" / "check_dq_v24_local_extension_parity.py",
        request.dataset_config,
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"DQ profile preflight is missing required files: {missing}")
    if not request.model_path.exists():
        raise FileNotFoundError(f"pretrained model path was not found: {request.model_path}")
    if not source_dirs:
        raise ValueError("dataset has no image_dir source groups")
    if importlib.util.find_spec("accelerate.commands.launch") is None:
        raise ModuleNotFoundError(
            f"accelerate is not available in the selected Python runtime: {sys.executable}"
        )


def _write_initial_artifacts(
    run_dir: Path,
    request: ResolvedProfileRequest,
    *,
    fingerprint: Mapping[str, Any],
    source_map_payload: Sequence[Mapping[str, Any]],
    image_count: int,
    probe_budget: int,
) -> None:
    write_json(run_dir / "resolved_args.json", request.provenance())
    write_json(run_dir / "protocol_fingerprint.json", fingerprint)
    write_json(run_dir / "source_group_map.json", source_map_payload)
    shutil.copy2(request.dataset_config, run_dir / "dataset_config_snapshot.toml")
    write_json(
        run_dir / "status.json",
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "status": "preparing",
            "current_stage": "preflight",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "preset": request.preset.name,
            "profile_scope": "local_body_tail_only",
            "image_count": image_count,
            "probe_budget": probe_budget,
            "source_group_count": len(source_map_payload),
            "fingerprint_sha256": fingerprint["fingerprint_sha256"],
            "not_quality_or_utility": True,
            "normal_training_sources_modified": False,
        },
    )


def run_profile_request(
    request: ResolvedProfileRequest,
    options: ProductionRunOptions,
) -> ProductionRunResult:
    source_dirs = source_dirs_from_dataset_config(request.dataset_config)
    preflight(request, source_dirs)
    output_base = validate_output_base(
        options.output_base,
        source_dirs=source_dirs,
        normal_output_dir=request.normal_output_dir,
    )
    source_map_payload, image_count = build_source_map(
        source_dirs,
        dataset_key=sanitize_profile_name(options.profile_name or request.output_name).casefold(),
    )
    probe_budget = min(image_count, request.preset.max_images)
    fingerprint = build_protocol_fingerprint(request, source_map_payload=source_map_payload)
    run_dir = allocate_run_directory(
        output_base,
        options.profile_name or request.output_name,
        str(fingerprint["fingerprint_sha256"]),
    )
    _write_initial_artifacts(
        run_dir,
        request,
        fingerprint=fingerprint,
        source_map_payload=source_map_payload,
        image_count=image_count,
        probe_budget=probe_budget,
    )
    launcher = Launcher(run_dir, dry_run=options.dry_run)
    launcher.log(f"DQ Profiler Beta run directory: {run_dir}")
    launcher.log("Scope: Local Body/Tail numerical Safety/Fidelity only; not quality or Utility")
    try:
        if options.preflight_only:
            update_status(
                run_dir,
                status="preflight_complete",
                current_stage="none",
                execution_started=False,
                report=None,
            )
            launcher.log("Preflight completed; GPU execution was not started")
            return ProductionRunResult(run_dir, "preflight_complete", None)
        if options.dry_run:
            command = profile_command(
                request,
                run_dir=run_dir,
                source_map=run_dir / "source_group_map.json",
                name=stage_names()["core_raw"],
                protocol="v24-acceptance-local",
                range_muls=request.preset.core_grid,
                max_images=probe_budget,
            )
            write_json(run_dir / "dry_run_command.json", {"argv": command})
            update_status(
                run_dir,
                status="dry_run_complete",
                current_stage="none",
                execution_started=False,
                report=None,
            )
            launcher.log("Dry run completed; no GPU process was started")
            return ProductionRunResult(run_dir, "dry_run_complete", None)

        active_profile, active_analysis, measured_grid, edge_rounds = run_local_pipeline(
            launcher,
            request,
            source_map=run_dir / "source_group_map.json",
            probe_budget=probe_budget,
            dataset_id=sanitize_profile_name(options.profile_name or request.output_name),
        )
        promoted = promote_analysis(run_dir, active_analysis)
        promoted.extend(promote_profile_provenance(run_dir, active_profile))
        selection = read_json(run_dir / "local_selection.json")
        update_status(
            run_dir,
            status="complete",
            current_stage="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            report="report.html",
            technical_report="technical_report.html",
            active_profile=str(active_profile),
            active_analysis=str(active_analysis),
            measured_grid=list(measured_grid),
            edge_extension_rounds=edge_rounds,
            edge_unresolved=bool(selection.get("edge_unresolved")),
            selection_valid=bool(selection.get("selection_valid")),
            retained_candidates=selection.get("credible_muls", []),
            promoted_artifacts=promoted,
            trajectory_status="research_only_not_run",
        )
        report = run_dir / "report.html"
        launcher.log(f"Primary report: {report}")
        launcher.log(f"Technical report: {run_dir / 'technical_report.html'}")
        if options.open_report and os.name == "nt":
            os.startfile(report)  # type: ignore[attr-defined]
        return ProductionRunResult(run_dir, "complete", report)
    except Exception as error:
        update_status(
            run_dir,
            status="failed",
            current_stage="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            error=repr(error),
            automatic_retry=False,
            existing_outputs_overwritten=False,
        )
        launcher.log(f"FAILED: {error!r}")
        raise
