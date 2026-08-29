from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from dq_profile.production_cli import resolve_training_cli
from dq_profile.production_runner import (
    DEFAULT_OUTPUT_BASE,
    ProductionRunOptions,
    run_profile_request,
)


def run_profile_mode(
    training_argv: Sequence[str],
    *,
    preset_name: str = "canonical-v1",
    output_base: Path = DEFAULT_OUTPUT_BASE,
    profile_name: str | None = None,
    preflight_only: bool = False,
    dry_run: bool = False,
    open_report: bool = False,
) -> int:
    request = resolve_training_cli(training_argv, preset_name=preset_name)
    result = run_profile_request(
        request,
        ProductionRunOptions(
            output_base=output_base,
            profile_name=profile_name,
            preflight_only=preflight_only,
            dry_run=dry_run,
            open_report=open_report,
        ),
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "run_dir": str(result.run_dir),
                "report": str(result.report) if result.report else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0
