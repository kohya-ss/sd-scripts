"""Send a command to a training which is running with `--run_control`, or show its status.

A command is a JSON file put into `<run directory>/commands/`. The training executes it after the next optimizer step
and writes the result to `<run directory>/commands/done/` with the same file name. This script writes the file, waits
for the result and prints it as JSON. See docs/run_control.md for the details.

This script has no dependencies other than the standard library, and can be used as an example to send commands
from other programs.

Usage:
    python tools/run_control_cli.py <run directory> status
    python tools/run_control_cli.py <run directory> sample --prompt "a cat --w 1024 --h 1024 --d 1"
    python tools/run_control_cli.py <run directory> sample --index 0 2 --seed 42
    python tools/run_control_cli.py <run directory> save
    python tools/run_control_cli.py <run directory> save_state
    python tools/run_control_cli.py <run directory> stop
    python tools/run_control_cli.py <run directory> pause
    python tools/run_control_cli.py <run directory> resume

Exit code: 0 = done, 1 = the command failed or was not executed, 2 = timeout while waiting for the result.
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

STATUS_FILE = "status.json"
COMMANDS_DIR = "commands"
COMMANDS_DONE_DIR = "done"

FINAL_STATES = ["finished", "stopped", "failed", "interrupted", "aborted"]
POLL_INTERVAL_SEC = 0.5


def read_json(path: str) -> Optional[dict]:
    """Returns None if the file does not exist or cannot be read now (it may be being replaced)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def send_command(run_dir: str, command: dict) -> str:
    """Write the command file atomically. Returns the file name, which is also the name of the result file."""
    commands_dir = os.path.join(run_dir, COMMANDS_DIR)
    os.makedirs(commands_dir, exist_ok=True)
    name = f"{time.time_ns()}_{command['command']}.json"  # executed in the order of the names
    path = os.path.join(commands_dir, name)
    tmp_path = path + ".tmp"  # the training reads *.json only
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(command, f, ensure_ascii=False)
    os.replace(tmp_path, path)
    return name


def wait_for_result(run_dir: str, name: str, timeout: float) -> Optional[dict]:
    """Returns the result, or None if it is not written within `timeout` seconds."""
    result_path = os.path.join(run_dir, COMMANDS_DIR, COMMANDS_DONE_DIR, name)
    deadline = time.time() + timeout
    while True:
        result = read_json(result_path)
        if result is not None:
            return result
        if time.time() >= deadline:
            return None
        time.sleep(POLL_INTERVAL_SEC)


def build_command(args: argparse.Namespace) -> dict:
    command: dict = {"command": args.command}
    if args.command == "sample":
        if args.prompt:
            command["prompts"] = args.prompt
        if args.index:
            command["indices"] = args.index
        overrides = {
            "seed": args.seed,
            "sample_steps": args.steps,
            "width": args.width,
            "height": args.height,
            "scale": args.scale,
            "negative_prompt": args.negative_prompt,
        }
        overrides = {k: v for k, v in overrides.items() if v is not None}
        if overrides:
            command["overrides"] = overrides
    elif args.command == "stop":
        command["save"] = not args.no_save
        command["save_state"] = not args.no_save_state
    return command


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Send a command to a training running with --run_control, or show its status")
    parser.add_argument("run_dir", type=str, help="run directory of the training (<output_dir>/<output_name>_run)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="print status.json")

    def add_command_parser(name: str, help: str) -> argparse.ArgumentParser:
        p = subparsers.add_parser(name, help=help)
        p.add_argument("--no_wait", action="store_true", help="do not wait for the result, only print the name of the command file")
        p.add_argument(
            "--timeout", type=float, default=3600.0, help="seconds to wait for the result (default 3600). The command is not canceled"
        )
        return p

    p = add_command_parser("sample", "generate sample images now with the current weights")
    p.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=None,
        help='prompts in the syntax of the prompt file, e.g. "a cat --w 1024 --h 1024 --d 1 --n low quality".'
        " A prompt which is not in the prompt file needs the text encoder in the training process",
    )
    p.add_argument(
        "--index", type=int, nargs="+", default=None, help="indices of the prompts in the prompt file (--sample_prompts), from 0"
    )
    p.add_argument("--seed", type=int, default=None, help="seed for all prompts of this command")
    p.add_argument("--steps", type=int, default=None, help="number of sampling steps for all prompts of this command")
    p.add_argument("--width", type=int, default=None, help="width for all prompts of this command")
    p.add_argument("--height", type=int, default=None, help="height for all prompts of this command")
    p.add_argument("--scale", type=float, default=None, help="CFG scale for all prompts of this command")
    p.add_argument("--negative_prompt", type=str, default=None, help="negative prompt for all prompts of this command")

    add_command_parser("save", "save the model now, with the name of the current step")
    add_command_parser("save_state", "save the state now, with the name of the current step")

    p = add_command_parser("stop", "save the model and the state with the name of the current step, and end the training")
    p.add_argument("--no_save", action="store_true", help="do not save the model")
    p.add_argument("--no_save_state", action="store_true", help="do not save the state")

    add_command_parser("pause", "move the models to CPU to free the VRAM, and wait for `resume` (or `stop`)")
    add_command_parser("resume", "move the models back and continue a paused training")
    return parser


def main() -> int:
    args = setup_parser().parse_args()

    status = read_json(os.path.join(args.run_dir, STATUS_FILE))
    if status is None:
        print(f"status.json is not found in {args.run_dir}. Is the training started with --run_control?", file=sys.stderr)
        return 1

    if args.command == "status":
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0

    if status.get("state") in FINAL_STATES:
        print(f"the training is not running (state: {status.get('state')})", file=sys.stderr)
        return 1

    name = send_command(args.run_dir, build_command(args))
    if args.no_wait:
        print(json.dumps({"file": name}))
        return 0

    result = wait_for_result(args.run_dir, name, args.timeout)
    if result is None:
        print(f"timeout: no result of {name} in {args.timeout} seconds. The command is not canceled", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
