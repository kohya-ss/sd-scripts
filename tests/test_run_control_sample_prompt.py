import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import run_control_cli  # noqa: E402
import run_control_sample_prompt  # noqa: E402
from library import run_control, sampling  # noqa: E402
from library.run_control import RunController  # noqa: E402


def make_outputs(length=5, width=8, dtype=torch.bfloat16):
    """Two tensors per text, like a model whose text encoder returns the embeddings and the mask."""
    return [torch.randn(1, length, width).to(dtype), torch.ones(1, length, dtype=torch.int32)]


# region the file of the text encoder outputs


def test_te_outputs_round_trip(tmp_path):
    path = str(tmp_path / "te.safetensors")
    outputs = {"a cat": make_outputs(5), "": make_outputs(1), "低品質": make_outputs(3)}
    sampling.save_prompt_te_outputs(path, outputs)

    loaded = sampling.load_prompt_te_outputs(path)
    assert list(loaded.keys()) == ["a cat", "", "低品質"]
    for text, tensors in outputs.items():
        assert len(loaded[text]) == 2
        for a, b in zip(loaded[text], tensors):
            assert a.dtype == b.dtype and torch.equal(a, b)


def test_load_te_outputs_rejects_other_files(tmp_path):
    from safetensors.torch import save_file

    with pytest.raises(ValueError, match="failed to load"):
        sampling.load_prompt_te_outputs(str(tmp_path / "missing.safetensors"))

    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not a safetensors file")
    with pytest.raises(ValueError, match="failed to load"):
        sampling.load_prompt_te_outputs(str(broken))

    other = str(tmp_path / "other.safetensors")
    save_file({"0.0": torch.zeros(1)}, other, metadata={"format": "something else"})
    with pytest.raises(ValueError, match="not a file of text encoder outputs"):
        sampling.load_prompt_te_outputs(other)

    no_texts = str(tmp_path / "no_texts.safetensors")
    save_file(
        {"0.0": torch.zeros(1)},
        no_texts,
        metadata={"format": sampling.PROMPT_TE_OUTPUTS_FORMAT, "version": sampling.PROMPT_TE_OUTPUTS_VERSION, "texts": "{}"},
    )
    with pytest.raises(ValueError, match="texts"):
        sampling.load_prompt_te_outputs(no_texts)


def test_verify_te_outputs():
    reference = make_outputs(5)
    sampling.verify_prompt_te_outputs({"a": make_outputs(5), "b": make_outputs(77)}, reference)  # the length may differ
    sampling.verify_prompt_te_outputs({"a": make_outputs(5, dtype=torch.float32)}, reference)  # another float dtype is fine

    bad = {
        "tensors, expected 2": [make_outputs()[0]],
        "has 2 dimensions, expected 3": [torch.randn(1, 5), torch.ones(1, 5, dtype=torch.int32)],
        "batch size 2": [torch.randn(2, 5, 8), torch.ones(1, 5, dtype=torch.int32)],
        "has the shape": make_outputs(width=16),
        "has the dtype": [torch.randn(1, 5, 8), torch.ones(1, 5)],
        "non-finite": [torch.full((1, 5, 8), float("nan")), torch.ones(1, 5, dtype=torch.int32)],
        "is empty": [torch.randn(1, 0, 8), torch.ones(1, 5, dtype=torch.int32)],
    }
    for message, tensors in bad.items():
        with pytest.raises(ValueError, match=message):
            sampling.verify_prompt_te_outputs({"a": tensors}, reference)


# endregion

# region the training side and the tool


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    monkeypatch.setattr(run_control_cli, "POLL_INTERVAL_SEC", 0.01)
    rc = RunController(str(tmp_path / "model_run"), status_interval=0.0)
    manifest = {
        "script": "fake_train_network.py",
        "cwd": str(tmp_path),
        "args": {"cache_text_encoder_outputs": True, "sample_prompts": "prompts.txt", "text_encoder_blocks_to_swap": None},
    }
    rc.start(manifest, max_train_steps=10, num_train_epochs=1)
    rc.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    return rc


def test_resolve_command_data_path(controller, tmp_path):
    commands_dir = os.path.abspath(os.path.join(controller.run_dir, run_control.COMMANDS_DIR))
    path = controller.resolve_command_data_path("te_outputs/1.safetensors")
    assert path == os.path.join(commands_dir, "te_outputs", "1.safetensors")

    for bad in ["../status.json", "te_outputs/../../x", str(tmp_path / "x.safetensors"), "", ".", None, 1]:
        with pytest.raises(run_control.CommandError):
            controller.resolve_command_data_path(bad)
    with pytest.raises(run_control.CommandError):
        RunController(None).resolve_command_data_path("te_outputs/1.safetensors")


def run_tool(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["run_control_sample_prompt.py", *argv])
    monkeypatch.chdir(os.getcwd())  # the tool changes the working directory: restore it after the test
    code = run_control_sample_prompt.main()
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_collect_texts():
    texts = run_control_sample_prompt.collect_texts(["a cat --w 512 --n blurry", "a dog --d 1", "a cat --n ugly"], "low quality")
    assert texts == ["a cat", "blurry", "a dog", "ugly", "low quality", ""]
    assert run_control_sample_prompt.collect_texts(["a cat"], None) == ["a cat", ""]


def test_tool_encodes_and_sends_the_command(controller, monkeypatch, capsys):
    calls = []

    def fake_encoder(train_args, texts, device, blocks_to_swap):
        calls.append((texts, str(device), blocks_to_swap))
        return {text: make_outputs(len(text) + 1) for text in texts}

    monkeypatch.setitem(run_control_sample_prompt.ENCODERS, "fake_train_network.py", fake_encoder)
    code, out, _ = run_tool(
        monkeypatch, capsys, controller.run_dir, "--prompt", "a cat --w 512 --n blurry", "--seed", "42", "--no_wait"
    )
    assert code == 0
    assert calls == [(["a cat", "blurry", ""], "cpu", None)]

    commands_dir = os.path.join(controller.run_dir, run_control.COMMANDS_DIR)
    command = json.load(open(os.path.join(commands_dir, json.loads(out)["file"]), encoding="utf-8"))
    te_outputs = command.pop("te_outputs")
    assert command == {"command": "sample", "prompts": ["a cat --w 512 --n blurry"], "overrides": {"seed": 42}}

    # the training side can resolve and load the file
    loaded = sampling.load_prompt_te_outputs(controller.resolve_command_data_path(te_outputs))
    assert list(loaded.keys()) == ["a cat", "blurry", ""]
    sampling.verify_prompt_te_outputs(loaded, make_outputs(3))


def test_tool_sends_the_usual_command_when_the_text_encoder_is_in_the_training(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    rc = RunController(str(tmp_path / "model_run"), status_interval=0.0)
    rc.start({"script": "anima_train_network.py", "args": {"cache_text_encoder_outputs": False}}, max_train_steps=10)
    rc.on_step_end(global_step=1, loss=0.5, average_loss=0.5)

    code, out, err = run_tool(monkeypatch, capsys, rc.run_dir, "--prompt", "a cat", "--no_wait")
    assert code == 0
    assert "not cached" in err
    command = json.load(open(os.path.join(rc.run_dir, run_control.COMMANDS_DIR, json.loads(out)["file"]), encoding="utf-8"))
    assert command == {"command": "sample", "prompts": ["a cat"]}


def test_tool_refuses_unsupported_trainings(controller, tmp_path, monkeypatch, capsys):
    # the script is not supported
    code, _, err = run_tool(monkeypatch, capsys, controller.run_dir, "--prompt", "a cat", "--no_wait")
    assert code == 1
    assert "is not supported" in err

    # no prompt file
    monkeypatch.setattr(RunController, "_install_exit_hooks", lambda self: None)
    rc = RunController(str(tmp_path / "other_run"), status_interval=0.0)
    rc.start({"script": "anima_train_network.py", "args": {"cache_text_encoder_outputs": True, "sample_prompts": None}})
    rc.on_step_end(global_step=1, loss=0.5, average_loss=0.5)
    code, _, err = run_tool(monkeypatch, capsys, rc.run_dir, "--prompt", "a cat", "--no_wait")
    assert code == 1
    assert "--sample_prompts" in err

    # the training ended
    rc.finish()
    code, _, err = run_tool(monkeypatch, capsys, rc.run_dir, "--prompt", "a cat", "--no_wait")
    assert code == 1
    assert "not running" in err

    assert not os.path.exists(os.path.join(rc.run_dir, run_control.COMMANDS_DIR))


# endregion
