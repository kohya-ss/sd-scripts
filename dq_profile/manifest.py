from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from dq_profile import (
    RUNTIME_METRIC_DEFINITION_VERSION as METRIC_DEFINITION_VERSION,
    RUNTIME_PROTOCOL_VERSION as PROTOCOL_VERSION,
    RUNTIME_SCHEMA_VERSION as SCHEMA_VERSION,
)
from dq_profile.protocol import canonical_sha256


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args], cwd=repo_root, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return result.stdout.strip()
    except Exception:
        return None


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _relative_file_record(repo_root: Path, path: Path) -> Optional[dict[str, Any]]:
    try:
        resolved = path.resolve()
        relative = resolved.relative_to(repo_root.resolve())
    except (OSError, ValueError):
        return None
    if not resolved.is_file():
        return None
    return {"path": relative.as_posix(), "size": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def _input_file_record(repo_root: Path, path: Path) -> Optional[dict[str, Any]]:
    try:
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    try:
        display_path = resolved.relative_to(repo_root.resolve()).as_posix()
        repository_relative = True
    except ValueError:
        display_path = str(resolved)
        repository_relative = False
    return {
        "path": display_path,
        "repository_relative": repository_relative,
        "size": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _loaded_repository_files(repo_root: Path) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for module in tuple(sys.modules.values()):
        raw_path = getattr(module, "__file__", None)
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.suffix in {".pyc", ".pyo"}:
            source_path = Path(str(path)[:-1])
            if source_path.exists():
                path = source_path
        record = _relative_file_record(repo_root, path)
        if record is not None:
            found[record["path"]] = record
    return [found[key] for key in sorted(found)]


def _gpu_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "available": torch.cuda.is_available(),
        "torch_compiled_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": [],
        "driver": None,
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result["devices"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory": properties.total_memory,
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
        try:
            query = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            result["driver"] = query.stdout.splitlines()[0].strip() if query.stdout else None
        except Exception:
            pass
    return result


def build_source_manifest(
    repo_root: str | os.PathLike[str],
    *,
    quant_rng_mode: str,
    additional_files: Iterable[str | os.PathLike[str]] = (),
) -> tuple[dict[str, Any], str]:
    root = Path(repo_root).resolve()
    additional_paths = tuple(Path(path).resolve() for path in additional_files)
    explicit = {
        root / "train_network.py",
        root / "networks" / "lora.py",
        root / "sdxl_train_network.py",
        root / "sdxl_dq_dataset_profile.py",
        root / "dq_profile" / "__init__.py",
        root / "dq_profile" / "copied_train_network.py",
        root / "dq_profile" / "copied_lora.py",
        root / "dq_profile" / "copied_sources.json",
        root / "dq_profile" / "sdxl_profile_trainer.py",
        root / "dq_profile" / "trainer_runtime.py",
        root / "dq_profile" / "v2_runtime.py",
        root / "dq_profile" / "v2_calibration.py",
        root / "dq_profile" / "v24_trajectory.py",
        root / "dq_profile" / "v2_metrics.py",
        root / "dq_profile" / "metrics.py",
        root / "dq_profile" / "quant_context.py",
        root / "dq_profile" / "protocol.py",
        root / "dq_profile" / "replay.py",
        root / "dq_profile" / "snapshot.py",
        root / "dq_profile" / "snapshot_parity.py",
        root / "dq_profile" / "report.py",
        root / "dq_profile" / "manifest.py",
        root / "tools" / "check_dq_profile_copy_drift.py",
        root / "library" / "train_util.py",
        root / "library" / "config_util.py",
        root / "library" / "sdxl_train_util.py",
        root / "library" / "sdxl_model_util.py",
        root / "library" / "sdxl_original_unet.py",
        root / "library" / "rounding_util.py",
        root / "library" / "triton_quant.py",
        root / "library" / "adamw8bit_fast.py",
        root / "library" / "maruo_global_config.py",
    }
    explicit_records = []
    for path in sorted(explicit, key=lambda item: str(item).lower()):
        record = _relative_file_record(root, path)
        if record is not None:
            explicit_records.append(record)
    additional_records = [
        record
        for path in sorted(set(additional_paths), key=lambda item: str(item).lower())
        if (record := _input_file_record(root, path)) is not None
    ]

    status = _git(root, "status", "--porcelain=v1")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "metric_definition_version": METRIC_DEFINITION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "quant_rng_mode": quant_rng_mode,
        "git": {
            "commit": _git(root, "rev-parse", "HEAD"),
            "branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status": [] if not status else status.splitlines(),
        },
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "packages": {
            "torch": torch.__version__,
            "bitsandbytes": _version("bitsandbytes"),
            "accelerate": _version("accelerate"),
            "transformers": _version("transformers"),
            "diffusers": _version("diffusers"),
            "triton": _version("triton"),
            "numpy": _version("numpy"),
        },
        "cuda": _gpu_environment(),
        "explicit_source_files": explicit_records,
        "additional_input_files": additional_records,
        "loaded_repository_modules": _loaded_repository_files(root),
    }
    return manifest, canonical_sha256(manifest)
