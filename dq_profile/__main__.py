from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from dq_profile.production_entry import run_profile_mode
from dq_profile.production_runner import DEFAULT_OUTPUT_BASE


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dq_profile",
        allow_abbrev=False,
        description=(
            "Run the Experimental Local Body/Tail DQ diagnostic. Pass this module the model "
            "and dataset options directly; do not wrap it in accelerate launch."
        ),
        epilog=(
            "Minimum example: python -m dq_profile "
            "--pretrained_model_name_or_path=MODEL --dataset_config=DATASET"
        ),
    )
    parser.add_argument(
        "--dq-profile-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_BASE,
        help=f"diagnostic base directory (default: {DEFAULT_OUTPUT_BASE})",
    )
    parser.add_argument(
        "--dq-profile-name",
        help="dataset folder name; defaults to --output_name or the dataset TOML stem",
    )
    parser.add_argument(
        "--dq-profile-preset",
        default="canonical-v1",
        help="versioned compatibility preset (default: canonical-v1)",
    )
    parser.add_argument(
        "--dq-profile-preflight",
        action="store_true",
        help="validate inputs and write provenance without starting GPU stages",
    )
    parser.add_argument(
        "--dq-profile-dry-run",
        action="store_true",
        help="write the resolved command plan without starting GPU stages",
    )
    parser.add_argument(
        "--dq-profile-open-report",
        action="store_true",
        help="open report.html after a successful Windows run",
    )
    selectors, training_argv = parser.parse_known_args(list(sys.argv[1:] if argv is None else argv))
    return run_profile_mode(
        training_argv,
        preset_name=selectors.dq_profile_preset,
        output_base=selectors.dq_profile_output_dir,
        profile_name=selectors.dq_profile_name,
        preflight_only=bool(selectors.dq_profile_preflight),
        dry_run=bool(selectors.dq_profile_dry_run),
        open_report=bool(selectors.dq_profile_open_report),
    )


if __name__ == "__main__":
    raise SystemExit(main())
