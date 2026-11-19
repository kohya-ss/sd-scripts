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
    rc = RunController(str(tmp_path / "model_run"), status_interval=0.0)
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
        with rc.watch_samples(step=1):
            pass
        rc.process_commands({"save": lambda command: pytest.fail("called")}, step=1)
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


def test_samples_saved_in_the_context_are_recorded(controller, tmp_path):
    from library import sampling

    controller.on_epoch_start(epoch=1)
    with controller.watch_samples(step=1):
        pass  # the sampling function did not generate images at this step
    assert read_status(controller.run_dir)["last_sample"] is None

    with controller.watch_samples(step=2):
        sampling.notify_sample_saved(str(tmp_path / "sample" / "a_000002_00.png"))
        sampling.notify_sample_saved(str(tmp_path / "sample" / "a_000002_01.png"))
    sampling.notify_sample_saved(str(tmp_path / "sample" / "outside.png"))  # not in the context

    last_sample = read_status(controller.run_dir)["last_sample"]
    assert last_sample["files"] == ["a_000002_00.png", "a_000002_01.png"]
    assert last_sample["dir"] == os.path.abspath(str(tmp_path / "sample"))
    assert (last_sample["step"], last_sample["epoch"]) == (2, 1)
    samples = [e for e in read_events(controller.run_dir) if e["event"] == "samples_generated"]
    assert len(samples) == 1


def put_command(run_dir, name, command):
    commands_dir = os.path.join(run_dir, run_control.COMMANDS_DIR)
    os.makedirs(commands_dir, exist_ok=True)
    path = os.path.join(commands_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(command if isinstance(command, str) else json.dumps(command))
    return path


def read_command_result(run_dir, name):
    return read_json(os.path.join(run_dir, run_control.COMMANDS_DIR, run_control.COMMANDS_DONE_DIR, name))


def test_commands_are_executed_in_order_and_results_are_written(controller):
    calls = []

    def save(command):
        calls.append(command)
        return {"path": "model-step00000003.safetensors"}

    controller.process_commands({"save": save}, step=1)  # no commands directory yet
    controller.on_epoch_start(epoch=1)
    path2 = put_command(controller.run_dir, "0002_save.json", {"command": "save", "note": "second"})
    path1 = put_command(controller.run_dir, "0001_save.json", {"command": "save", "note": "first"})
    put_command(controller.run_dir, "0003_save.json.tmp", {"command": "save", "note": "being written"})
    controller.process_commands({"save": save}, step=3)

    assert [c["note"] for c in calls] == ["first", "second"]
    assert not os.path.exists(path1) and not os.path.exists(path2)  # moved to the done directory
    result = read_command_result(controller.run_dir, "0001_save.json")
    assert result["status"] == "ok"
    assert result["result"] == {"path": "model-step00000003.safetensors"}
    assert result["command"] == {"command": "save", "note": "first"}
    assert (result["step"], result["epoch"], result["run_id"]) == (3, 1, controller.run_id)

    events = [(e["event"], e.get("status")) for e in read_events(controller.run_dir) if e["event"].startswith("command")]
    assert events == [("command_received", None), ("command_finished", "ok")] * 2
    assert read_status(controller.run_dir)["last_command"]["file"] == "0002_save.json"

    controller.process_commands({"save": save}, step=4)  # nothing is executed twice
    assert len(calls) == 2


def test_command_errors_are_reported_and_training_continues(controller):
    def sample(command):
        raise ValueError("index 9 is out of range")

    put_command(controller.run_dir, "1_sample.json", {"command": "sample", "indices": [9]})
    put_command(controller.run_dir, "2_unknown.json", {"command": "pause"})
    controller.process_commands({"sample": sample}, step=1)

    result = read_command_result(controller.run_dir, "1_sample.json")
    assert (result["status"], result["error"]) == ("error", "index 9 is out of range")
    result = read_command_result(controller.run_dir, "2_unknown.json")
    assert result["status"] == "error"
    assert "unknown command `pause`" in result["error"] and "sample" in result["error"]


def test_unexpected_error_of_a_command_is_reported_and_raised(controller):
    def save(command):
        raise RuntimeError("disk full")

    put_command(controller.run_dir, "1_save.json", {"command": "save"})
    with pytest.raises(RuntimeError):
        controller.process_commands({"save": save}, step=1)
    result = read_command_result(controller.run_dir, "1_save.json")
    assert (result["status"], result["error"]) == ("error", "RuntimeError: disk full")


def test_invalid_command_file_is_retried_then_rejected(controller, monkeypatch):
    path = put_command(controller.run_dir, "1_save.json", '{"command": "sa')  # may be still being written
    controller.process_commands({}, step=1)
    assert os.path.exists(path)  # left for the next step

    monkeypatch.setattr(run_control, "COMMAND_PARSE_GRACE_SEC", -1.0)  # negative: the clocks may differ a little
    controller.process_commands({}, step=2)
    assert not os.path.exists(path)
    result = read_command_result(controller.run_dir, "1_save.json")
    assert result["status"] == "error" and "invalid command file" in result["error"]

    put_command(controller.run_dir, "2_list.json", "[1, 2]")
    controller.process_commands({}, step=3)
    assert "must be a JSON object" in read_command_result(controller.run_dir, "2_list.json")["error"]


def test_stop_command_ends_the_processing_and_the_rest_expires(controller):
    def stop(command):
        controller.request_stop()
        return {"checkpoint": "a.safetensors"}

    put_command(controller.run_dir, "1_stop.json", {"command": "stop"})
    put_command(controller.run_dir, "2_save.json", {"command": "save"})
    assert not controller.stop_requested
    controller.process_commands({"stop": stop, "save": lambda command: pytest.fail("called after stop")}, step=5)
    assert controller.stop_requested

    controller.finish("stopped")
    assert read_status(controller.run_dir)["state"] == "stopped"
    result = read_command_result(controller.run_dir, "2_save.json")
    assert result["status"] == "expired"
    controller.process_commands({"save": lambda command: pytest.fail("called after finish")}, step=6)


def test_disabled_commands_are_answered_with_the_reason(controller):
    controller.disable_commands("commands are not supported in multi-process training")
    put_command(controller.run_dir, "1_save.json", {"command": "save"})
    controller.process_commands({"save": lambda command: pytest.fail("called")}, step=1)
    result = read_command_result(controller.run_dir, "1_save.json")
    assert (result["status"], result["error"]) == ("error", "commands are not supported in multi-process training")


def test_commands_sent_before_the_run_are_expired(tmp_path, monkeypatch):
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    run_dir = str(tmp_path / "run")
    put_command(run_dir, "1_stop.json", {"command": "stop"})  # e.g. sent to the previous run after it ended
    rc = RunController(run_dir, status_interval=0.0)
    rc.start({})
    rc.process_commands({"stop": lambda command: pytest.fail("called")}, step=1)
    result = read_command_result(run_dir, "1_stop.json")
    assert result["status"] == "expired"
    assert "before the training started" in result["error"]


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


def read_loss_rows(run_dir):
    with open(os.path.join(run_dir, run_control.LOSS_FILE), encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def make_batch(image_keys, sizes_hw=None):
    import torch

    batch = {"image_keys": image_keys, "loss_weights": torch.ones(len(image_keys))}
    batch["subset_indices"] = [(0, i) for i in range(len(image_keys))]
    if sizes_hw is not None:
        batch["target_sizes_hw"] = torch.LongTensor(sizes_hw)
    return batch


def test_sample_losses_are_recorded(controller):
    import torch

    controller.on_epoch_start(epoch=1)
    batch = make_batch(["a.png", "b.png"], [[512, 768], [512, 768]])
    controller.on_batch_fetched(batch, global_step=0)
    losses = torch.tensor([0.25, 0.5], requires_grad=True)
    controller.record_sample_losses(batch, torch.tensor([100, 900]), losses)
    controller.on_step_end(global_step=1, loss=0.375, average_loss=0.375)  # written together with the status

    rows = read_loss_rows(controller.run_dir)
    assert rows[0]["run_id"] == controller.run_id and "image" not in rows[0]  # each run starts with its run_id
    assert rows[1:] == [
        {"step": 1, "epoch": 1, "image": "a.png", "dataset": 0, "subset": 0, "timestep": 100.0, "loss": 0.25, "width": 768, "height": 512, "batch_size": 2, "loss_weight": 1.0},
        {"step": 1, "epoch": 1, "image": "b.png", "dataset": 0, "subset": 1, "timestep": 900.0, "loss": 0.5, "width": 768, "height": 512, "batch_size": 2, "loss_weight": 1.0},
    ]


def test_sample_losses_keep_the_step_of_their_batch(controller):
    # in replica training a batch is processed later on another thread, when the next batch is already fetched
    import torch

    controller.on_epoch_start(epoch=1)
    batch1 = make_batch(["a.png"])
    batch2 = make_batch(["b.png"])
    controller.on_batch_fetched(batch1, global_step=0)
    controller.on_batch_fetched(batch2, global_step=1)
    controller.on_epoch_start(epoch=2)
    moved_batch1 = dict(batch1)  # a copy on another device
    controller.record_sample_losses(batch2, torch.tensor([0.5]), torch.tensor([0.2]))
    controller.record_sample_losses(moved_batch1, torch.tensor([0.5]), torch.tensor([0.1]))
    controller.finish()  # the pending rows are written at the end

    rows = read_loss_rows(controller.run_dir)[1:]
    assert [(r["image"], r["step"], r["epoch"]) for r in rows] == [("b.png", 2, 1), ("a.png", 1, 1)]


def test_sample_losses_with_missing_values(controller):
    import torch

    controller.record_sample_losses({"loss_weights": torch.ones(1)}, None, torch.tensor([0.1]))  # no image keys: skipped
    batch = {"image_keys": ["a.png", "b.png"]}  # no size, no weight, not tagged
    controller.record_sample_losses(batch, None, torch.tensor([0.1, float("nan")]))
    controller.record_sample_losses(batch, torch.tensor(500), torch.tensor([0.1, 0.2]))  # timestep is not per sample
    controller.finish()

    rows = read_loss_rows(controller.run_dir)[1:]
    assert len(rows) == 4
    assert rows[0] == {
        "step": None, "epoch": 0, "image": "a.png", "dataset": None, "subset": None, "timestep": None, "loss": pytest.approx(0.1),
        "width": None, "height": None, "batch_size": 2, "loss_weight": None,
    }  # fmt: skip
    assert rows[1]["loss"] is None  # non-finite
    assert rows[2]["timestep"] is None


def test_sample_losses_are_not_recorded_when_disabled():
    import torch

    rc = RunController(None)
    rc.on_batch_fetched({"image_keys": ["a.png"]}, global_step=0)
    rc.record_sample_losses({"image_keys": ["a.png"]}, None, torch.tensor([0.1]))
    assert rc._pending_losses == []


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
