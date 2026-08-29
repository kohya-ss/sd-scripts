from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from dq_profile.v24_parity import check_local_extension_parity


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check v2.4 shared core probes before/after local edge extension."
    )
    parser.add_argument("--core-profile-dir", type=Path, required=True)
    parser.add_argument("--extension-profile-dir", type=Path, required=True)
    parser.add_argument("--common-muls", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        common_muls = tuple(
            float(value.strip())
            for value in str(args.common_muls).split(",")
            if value.strip()
        )
        if not common_muls:
            raise ValueError("--common-muls is empty")
        result = check_local_extension_parity(
            core_profile_dir=args.core_profile_dir,
            extension_profile_dir=args.extension_profile_dir,
            common_muls=common_muls,
        )
    except Exception as error:
        result = {
            "schema_version": "2.4.0-local-extension-parity",
            "gate": "fail",
            "passed": False,
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "safety_not_utility": True,
        }
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
