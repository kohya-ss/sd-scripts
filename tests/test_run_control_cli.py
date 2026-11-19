import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import run_control_cli  # noqa: E402
from library import run_control  # noqa: E402
from library.run_control import RunController  # noqa: E402


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    monkeypatch.setattr(run_control_cli, "POLL_INTERVAL_SEC", 0.01)
    rc = RunController(str(tmp_path / "model_run"), status_interval=0.0)
    rc.start({}, max_train_steps=10, num_train_epochs=1)
    rc.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    return rc


def run_cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["run_control_cli.py", *argv])
    code = run_control_cli.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_constants_match_the_library():
    assert run_control_cli.STATUS_FILE == run_control.STATUS_FILE
    assert run_control_cli.COMMANDS_DIR == run_control.COMMANDS_DIR
    assert run_control_cli.COMMANDS_DONE_DIR == run_control.COMMANDS_DONE_DIR


def test_status(controller, monkeypatch, capsys):
    code, out, _ = run_cli(monkeypatch, capsys, controller.run_dir, "status")
    assert code == 0
    assert json.loads(out)["state"] == "running"


def test_no_run_directory(tmp_path, monkeypatch, capsys):
    code, _, err = run_cli(monkeypatch, capsys, str(tmp_path / "missing"), "save")
    assert code == 1
    assert "status.json is not found" in err


def test_command_is_refused_after_the_training_ended(controller, monkeypatch, capsys):
    controller.finish()
    code, _, err = run_cli(monkeypatch, capsys, controller.run_dir, "save")
    assert code == 1
    assert "not running" in err
    assert not os.path.exists(os.path.join(controller.run_dir, run_control.COMMANDS_DIR))


def test_sample_command_round_trip(controller, monkeypatch, capsys):
    received = []

    def sample(command):
        received.append(command)
        return {"files": ["a.png"]}

    # the "training" processes the command while the CLI is waiting
    stop = threading.Event()

    def training_loop():
        step = 1
        while not stop.is_set():
            controller.process_commands({"sample": sample}, step=step)
            step += 1
            stop.wait(0.01)

    thread = threading.Thread(target=training_loop)
    thread.start()
    try:
        code, out, _ = run_cli(
            monkeypatch, capsys, controller.run_dir,
            "sample", "--prompt", "a cat --w 1024", "a dog", "--index", "0", "2", "--seed", "42", "--steps", "8", "--timeout", "30",
        )  # fmt: skip
    finally:
        stop.set()
        thread.join()

    assert code == 0
    result = json.loads(out)
    assert result["status"] == "ok"
    assert result["result"] == {"files": ["a.png"]}
    assert received == [
        {
            "command": "sample",
            "prompts": ["a cat --w 1024", "a dog"],
            "indices": [0, 2],
            "overrides": {"seed": 42, "sample_steps": 8},
        }
    ]


def test_error_result_gives_exit_code_1(controller, monkeypatch, capsys):
    code, out, _ = run_cli(monkeypatch, capsys, controller.run_dir, "stop", "--no_save_state", "--no_wait")
    assert code == 0
    name = json.loads(out)["file"]
    assert name.endswith("_stop.json")
    command = json.load(open(os.path.join(controller.run_dir, run_control.COMMANDS_DIR, name), encoding="utf-8"))
    assert command == {"command": "stop", "save": True, "save_state": False}

    def stop(command):
        raise ValueError("cannot stop")

    controller.process_commands({"stop": stop}, step=2)
    result = run_control_cli.wait_for_result(controller.run_dir, name, timeout=1.0)
    assert result["status"] == "error"


def test_timeout(controller, monkeypatch, capsys):
    code, _, err = run_cli(monkeypatch, capsys, controller.run_dir, "save", "--timeout", "0.05")
    assert code == 2
    assert "timeout" in err
