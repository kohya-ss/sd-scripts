import argparse
import json
import os
import sys
import time

import pytest

from library import run_control
from library.run_control import RunController


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_events(run_dir):
    with open(os.path.join(run_dir, run_control.EVENTS_FILE), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_status(run_dir):
    return read_json(os.path.join(run_dir, run_control.STATUS_FILE))


@pytest.fixture
def controller(tmp_path, monkeypatch):
    # do not leave hooks in the test process
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    sample_dir = tmp_path / "sample"
    rc = RunController(str(tmp_path / "model_run"), sample_dir=str(sample_dir), status_interval=0.0)
    rc.start({"script": "test"}, max_train_steps=10, num_train_epochs=2)
    return rc


def make_args(**kwargs):
    defaults = dict(output_dir="out", output_name="model", logging_dir=None, run_control=True)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_disabled_controller_writes_nothing(tmp_path):
    for rc in [
        run_control.create_run_controller(make_args(output_dir=str(tmp_path), run_control=False)),
        run_control.create_run_controller(make_args(output_dir=str(tmp_path)), is_main_process=False),
    ]:
        assert not rc.enabled
        rc.start({})
        rc.on_batch_fetched()
        rc.on_epoch_start(epoch=1)
        rc.on_step_end(global_step=1, loss=0.1, average_loss=0.1)
        rc.record_checkpoint("a.safetensors", step=1)
        rc.record_state("a-state", step=1)
        rc.check_new_samples(step=1)
        rc.on_epoch_end(global_step=1)
        rc.finish()
    assert os.listdir(tmp_path) == []


def test_run_dir_name(tmp_path):
    rc = run_control.create_run_controller(make_args(output_dir=str(tmp_path)))
    assert rc.run_dir == os.path.join(str(tmp_path), "model_run")
    rc = run_control.create_run_controller(make_args(output_dir=str(tmp_path), output_name=None))
    assert rc.run_dir == os.path.join(str(tmp_path), "last_run")


def test_start_writes_manifest_status_and_event(controller):
    manifest = read_json(os.path.join(controller.run_dir, run_control.MANIFEST_FILE))
    assert manifest["schema_version"] == run_control.SCHEMA_VERSION
    assert manifest["run_id"] == controller.run_id
    assert manifest["script"] == "test"

    status = read_status(controller.run_dir)
    assert status["state"] == "starting"
    assert status["progress"] == {"global_step": 0, "max_train_steps": 10, "epoch": 0, "num_train_epochs": 2}

    events = read_events(controller.run_dir)
    assert [e["event"] for e in events] == ["run_start"]
    assert events[0]["run_id"] == controller.run_id


def test_step_updates_status(controller):
    logs = {"loss/current": 0.5, "loss/average": 0.4, "lr/unet": 1e-4, "norm/avg_key_norm": 0.2}
    controller.on_epoch_start(epoch=1)
    controller.on_batch_fetched()
    controller.on_step_end(global_step=1, loss=0.5, average_loss=0.4, logs=logs)
    time.sleep(0.05)  # longer than the granularity of the clock
    controller.on_batch_fetched()
    controller.on_step_end(global_step=2, loss=0.3, average_loss=0.35, logs=logs)

    status = read_status(controller.run_dir)
    assert status["state"] == "running"
    assert status["progress"]["global_step"] == 2
    assert status["progress"]["epoch"] == 1
    assert status["loss"] == {"current": 0.3, "average": 0.35}
    assert status["lr"] == {"unet": 1e-4}
    assert status["metrics"] == {"norm/avg_key_norm": 0.2}
    assert status["speed"]["steps_per_sec"] > 0
    assert status["speed"]["eta_sec"] >= 0
    assert 0 <= status["timing"]["data_wait_ratio"] <= 1


def test_status_is_throttled(tmp_path, monkeypatch):
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    rc = RunController(str(tmp_path / "run"), status_interval=3600.0)
    rc.start({})
    rc.on_step_end(global_step=1, loss=0.5, average_loss=0.5)  # state changes to running: written
    rc.on_step_end(global_step=2, loss=0.5, average_loss=0.5)  # within the interval: not written
    assert read_status(rc.run_dir)["progress"]["global_step"] == 1
    rc.finish()  # always written
    status = read_status(rc.run_dir)
    assert status["progress"]["global_step"] == 2
    assert status["state"] == "finished"


def test_non_finite_loss_is_strict_json_and_reported_once(controller):
    controller.on_step_end(global_step=1, loss=float("nan"), average_loss=float("nan"))
    controller.on_step_end(global_step=2, loss=float("inf"), average_loss=float("nan"))

    with open(os.path.join(controller.run_dir, run_control.STATUS_FILE), encoding="utf-8") as f:
        text = f.read()
    assert "NaN" not in text and "Infinity" not in text
    status = json.loads(text)
    assert status["loss"] == {"current": None, "average": None}
    assert status["anomalies"] == {"non_finite_loss": 2}

    anomalies = [e for e in read_events(controller.run_dir) if e["event"] == "anomaly"]
    assert len(anomalies) == 1
    assert anomalies[0]["kind"] == "non_finite_loss"
    assert anomalies[0]["step"] == 1


def test_checkpoint_and_state_are_recorded(controller, tmp_path):
    ckpt = str(tmp_path / "model-step00000002.safetensors")
    controller.on_epoch_start(epoch=1)
    controller.record_checkpoint(ckpt, step=2)
    controller.record_state(str(tmp_path / "model-step00000002-state"), step=2)
    controller.record_state(None, step=3)  # nothing was saved

    status = read_status(controller.run_dir)
    assert status["last_checkpoint"]["path"] == os.path.abspath(ckpt)
    assert status["last_checkpoint"]["step"] == 2
    assert status["last_checkpoint"]["epoch"] == 1
    assert status["last_state"]["step"] == 2

    events = read_events(controller.run_dir)
    assert [e["event"] for e in events] == ["run_start", "checkpoint_saved", "state_saved"]
    assert [e["epoch"] for e in events] == [0, 1, 1]  # every event has the current epoch


def test_new_samples_are_detected(controller):
    sample_dir = controller.sample_dir
    controller.check_new_samples(step=1)  # the directory does not exist yet
    assert read_status(controller.run_dir)["last_sample"] is None

    os.makedirs(sample_dir)
    for name in ["a_000002_00.png", "a_000002_01.png"]:
        open(os.path.join(sample_dir, name), "w").close()
    controller.check_new_samples(step=2)

    last_sample = read_status(controller.run_dir)["last_sample"]
    assert last_sample["files"] == ["a_000002_00.png", "a_000002_01.png"]
    assert last_sample["step"] == 2

    controller.check_new_samples(step=3)  # nothing new
    samples = [e for e in read_events(controller.run_dir) if e["event"] == "samples_generated"]
    assert len(samples) == 1


def test_sample_dir_is_not_listed_again_once_settled(controller, monkeypatch):
    os.makedirs(controller.sample_dir)
    controller.check_new_samples(step=1)
    # pretend that the last listing was long after the last change of the directory
    controller._sample_dir_scan_time += 10 * run_control.SAMPLE_DIR_SETTLE_NS

    monkeypatch.setattr(run_control.os, "listdir", lambda path: pytest.fail("listed again"))
    controller.check_new_samples(step=2)


def test_existing_samples_are_not_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir()
    (sample_dir / "old.png").touch()
    rc = RunController(str(tmp_path / "run"), sample_dir=str(sample_dir), status_interval=0.0)
    rc.start({})
    (sample_dir / "new.png").touch()
    rc.check_new_samples(step=1)
    assert read_status(rc.run_dir)["last_sample"]["files"] == ["new.png"]


def test_finish_and_exit_states(controller):
    controller.on_epoch_start(epoch=1)
    controller.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    controller.on_epoch_end(global_step=1, average_loss=0.5)
    controller.finish()
    controller._on_exit()  # already finished: nothing happens

    assert read_status(controller.run_dir)["state"] == "finished"
    events = [e["event"] for e in read_events(controller.run_dir)]
    assert events == ["run_start", "epoch_end", "run_end"]


@pytest.mark.parametrize(
    "error, expected_state",
    [
        (None, "aborted"),
        ({"type": "KeyboardInterrupt", "message": "", "traceback": []}, "interrupted"),
        ({"type": "RuntimeError", "message": "CUDA out of memory", "traceback": ["..."]}, "failed"),
    ],
)
def test_exit_without_finish(controller, error, expected_state):
    controller._error = error
    controller._on_exit()
    status = read_status(controller.run_dir)
    assert status["state"] == expected_state
    assert status["error"] == error
    assert read_events(controller.run_dir)[-1]["state"] == expected_state


def test_excepthook_records_error_and_chains(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(sys, "excepthook", lambda *a: called.append(a))
    monkeypatch.setattr(run_control.atexit, "register", lambda f: None)
    rc = RunController(str(tmp_path / "run"))
    rc.start({})

    try:
        raise RuntimeError("boom")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())

    assert len(called) == 1  # the previous hook is still called
    assert rc._error["type"] == "RuntimeError"
    assert rc._error["message"] == "boom"
    assert any("boom" in line for line in rc._error["traceback"])


def test_status_write_failure_is_ignored_and_retried(controller, monkeypatch):
    calls = []

    def failing_write(path, obj):
        calls.append(path)
        return False

    monkeypatch.setattr(run_control, "write_json_atomic", failing_write)
    controller.status_interval = 3600.0
    controller._last_status_write = 0.0
    controller.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    controller.on_step_end(global_step=2, loss=0.5, average_loss=0.5)
    assert len(calls) == 2  # retried at the next step even within the interval


def test_redaction():
    args = make_args(wandb_api_key="secret1", huggingface_token=None, learning_rate=1e-4)
    redacted = run_control.redact_args(args)
    assert redacted["wandb_api_key"] == run_control.REDACTED
    assert redacted["huggingface_token"] is None
    assert redacted["learning_rate"] == 1e-4

    argv = ["--wandb_api_key", "secret1", "--huggingface_token=secret2", "--output_name", "model"]
    assert run_control.redact_command_line(argv) == [
        "--wandb_api_key",
        run_control.REDACTED,
        f"--huggingface_token={run_control.REDACTED}",
        "--output_name",
        "model",
    ]


def test_build_manifest(tmp_path):
    args = make_args(output_dir=str(tmp_path), huggingface_token="secret", network_args=["conv_dim=4"])
    metadata = {
        "ss_num_train_images": "10",
        "ss_datasets": json.dumps(
            [{"resolution": [1024, 1024], "tag_frequency": {"dir": {"tag": 3}}, "subsets": [{"img_count": 10}]}]
        ),
        "ss_tag_frequency": json.dumps({"dir": {"tag": 3}}),
    }
    manifest = run_control.build_manifest(args, metadata, training={"max_train_steps": 100})

    assert manifest["args"]["huggingface_token"] == run_control.REDACTED
    assert manifest["training"] == {"max_train_steps": 100}
    assert manifest["metadata"]["ss_datasets"][0]["subsets"][0]["img_count"] == 10  # nested JSON is decoded
    # the tag frequency is in the model metadata only, not copied to the manifest
    assert "tag_frequency" not in json.dumps(manifest["metadata"])
    assert manifest["metadata"]["ss_datasets"][0]["resolution"] == [1024, 1024]
    assert json.loads(metadata["ss_datasets"])[0]["tag_frequency"] == {"dir": {"tag": 3}}  # the source is not modified
    assert manifest["outputs"]["output_dir"] == os.path.abspath(str(tmp_path))
    assert "torch" in manifest["environment"]["packages"]
    assert "secret" not in json.dumps(run_control.to_json_safe(manifest))


def test_vram_and_gpus_come_from_device_utils(controller, monkeypatch):
    # the same code path is used for CUDA, XPU and MPS. MPS has no peak and may not have the total memory
    gb = 1024**3
    monkeypatch.setattr(
        run_control.device_utils,
        "get_device_memory_stats",
        lambda: [{"device": "mps", "allocated": 2 * gb, "reserved": 3 * gb, "peak_allocated": None}],
    )
    monkeypatch.setattr(
        run_control.device_utils,
        "get_device_descriptions",
        lambda: [{"device": "mps", "name": "Apple MPS", "total_memory": None}],
    )
    controller.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    assert read_status(controller.run_dir)["vram"] == [
        {"device": "mps", "allocated_mb": 2048, "reserved_mb": 3072, "peak_allocated_mb": None}
    ]
    assert run_control.get_environment_info()["gpus"] == [{"device": "mps", "name": "Apple MPS", "total_memory_mb": None}]


def test_no_accelerator_device(controller, monkeypatch):
    monkeypatch.setattr(run_control.device_utils, "get_device_memory_stats", lambda: [])
    controller.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    assert read_status(controller.run_dir)["vram"] == []


def test_to_json_safe():
    import numpy as np
    import torch

    obj = {1: float("nan"), "t": torch.tensor(1.5), "n": np.float32(2.5), "l": (1, 2), "o": object}
    safe = run_control.to_json_safe(obj)
    assert safe["1"] is None
    assert safe["t"] == 1.5
    assert safe["n"] == 2.5
    assert safe["l"] == [1, 2]
    assert isinstance(safe["o"], str)
    json.dumps(safe, allow_nan=False)
