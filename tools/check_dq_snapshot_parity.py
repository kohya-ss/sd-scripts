from __future__ import annotations

import argparse
import json
from pathlib import Path

from dq_profile.snapshot_parity import compare_snapshot_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two DQ Profiler warmup-boundary snapshots across processes."
        )
    )
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=1e-7)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_snapshot_outputs(
        args.left,
        args.right,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
