from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "dq_profile" / "copied_sources.json"


class CopyDriftError(RuntimeError):
    """Raised when an ordinary source or diagnostic copy no longer matches metadata."""


def normalized_sha256(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise CopyDriftError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def validate_copy_manifest(
    repo_root: Path = REPO_ROOT,
    manifest_path: Path | None = None,
    *,
    verify_git: bool = True,
) -> dict[str, Any]:
    root = repo_root.resolve()
    path = (manifest_path or root / "dq_profile" / "copied_sources.json").resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0.0":
        raise CopyDriftError("copied_sources.json must use schema_version 1.0.0")
    source_commit = str(payload.get("source_commit", "")).strip()
    files = payload.get("files")
    if not source_commit:
        raise CopyDriftError("copied_sources.json has no source_commit")
    if not isinstance(files, Mapping) or not files:
        raise CopyDriftError("copied_sources.json has no file records")

    errors: list[str] = []
    checked: list[dict[str, str]] = []
    if verify_git:
        resolved_commit = _git(root, "rev-parse", "--verify", f"{source_commit}^{{commit}}")
        if resolved_commit != source_commit:
            errors.append(
                f"source_commit resolved to {resolved_commit}, expected {source_commit}"
            )

    for copied_name, raw_record in sorted(files.items()):
        if not isinstance(raw_record, Mapping):
            errors.append(f"{copied_name}: record must be an object")
            continue
        source_name = str(raw_record.get("source", "")).strip()
        copied_path = root / str(copied_name)
        source_path = root / source_name
        if not source_path.is_file():
            errors.append(f"{copied_name}: source file is missing: {source_name}")
            continue
        if not copied_path.is_file():
            errors.append(f"{copied_name}: diagnostic copy is missing")
            continue
        source_hash = normalized_sha256(source_path)
        copied_hash = normalized_sha256(copied_path)
        expected_source_hash = str(raw_record.get("source_normalized_sha256", ""))
        expected_copied_hash = str(raw_record.get("copied_normalized_sha256", ""))
        if source_hash != expected_source_hash:
            errors.append(
                f"{source_name}: source drifted ({source_hash} != {expected_source_hash})"
            )
        if copied_hash != expected_copied_hash:
            errors.append(
                f"{copied_name}: diagnostic copy drifted "
                f"({copied_hash} != {expected_copied_hash})"
            )
        if verify_git:
            blob_oid = _git(root, "rev-parse", f"{source_commit}:{source_name}")
            expected_blob_oid = str(raw_record.get("source_git_blob_oid", ""))
            if blob_oid != expected_blob_oid:
                errors.append(
                    f"{source_name}: source commit blob changed "
                    f"({blob_oid} != {expected_blob_oid})"
                )
        checked.append(
            {
                "source": source_name,
                "copy": str(copied_name),
                "source_normalized_sha256": source_hash,
                "copied_normalized_sha256": copied_hash,
            }
        )

    if errors:
        raise CopyDriftError("copy drift check failed:\n- " + "\n- ".join(errors))
    return {
        "status": "pass",
        "source_commit": source_commit,
        "manifest": str(path),
        "files": checked,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check DQ profiler diagnostic copies against recorded source provenance."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    try:
        result = validate_copy_manifest(REPO_ROOT, args.manifest)
    except (CopyDriftError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            "DQ profiler copy drift check: PASS "
            f"({len(result['files'])} copies, source {result['source_commit'][:12]})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
