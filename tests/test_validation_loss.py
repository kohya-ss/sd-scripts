from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
import torch

import train_network
from library.logging_util import LossRecorder
from train_network import NetworkTrainer


@pytest.fixture
def validation(monkeypatch):
    trainer = NetworkTrainer()
    args = Namespace(min_timestep=-10, max_timestep=-1)
    network = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        network.weight.fill_(1)
    progress = Mock()
    progress_factory = Mock(return_value=progress)
    monkeypatch.setattr(train_network, "tqdm", progress_factory)

    events = []

    def process_batch(batch, *unused, **flags):
        events.append(("process", args.min_timestep, args.max_timestep))
        # A tiny CPU model gives a distinct scalar for each batch/timestep pair.
        return network(batch).mean() * args.min_timestep

    monkeypatch.setattr(trainer, "process_batch", Mock(side_effect=process_batch))
    monkeypatch.setattr(
        trainer,
        "on_step_start",
        Mock(side_effect=lambda *a, **k: events.append(("start", args.min_timestep, args.max_timestep))),
    )
    monkeypatch.setattr(
        trainer,
        "on_validation_step_end",
        Mock(side_effect=lambda *a, **k: events.append(("end", args.min_timestep, args.max_timestep))),
    )

    tensorboard_logs, wandb_logs = [], []
    trackers = {
        "tensorboard": SimpleNamespace(
            name="tensorboard", log=lambda logs, step: tensorboard_logs.append((dict(logs), step))
        ),
        "wandb": SimpleNamespace(name="wandb", log=lambda logs: wandb_logs.append(dict(logs))),
    }
    accelerator = SimpleNamespace(
        is_local_main_process=True, trackers=list(trackers.values()), get_tracker=trackers.__getitem__
    )
    training = LossRecorder()
    training.add(epoch=0, step=0, loss=2)
    training.add(epoch=0, step=1, loss=4)
    return SimpleNamespace(
        trainer=trainer,
        args=args,
        network=network,
        accelerator=accelerator,
        training=training,
        progress=progress,
        progress_factory=progress_factory,
        events=events,
        tensorboard_logs=tensorboard_logs,
        wandb_logs=wandb_logs,
    )


def _run(
    validation, losses, *, mode="step", epoch=0, global_step=10, timesteps=(1,), limit=None, total=None, **options
):
    batches = [torch.as_tensor(value, dtype=torch.float32).reshape(-1, 1) for value in losses]
    validation.events.clear()
    validation.progress.reset_mock()
    validation.args.min_timestep, validation.args.max_timestep = -10, -1
    kwargs = {
        "mode": mode,
        "accelerator": validation.accelerator,
        "args": validation.args,
        "network": validation.network,
        "text_encoders": [],
        "unet": None,
        "vae": None,
        "noise_scheduler": None,
        "vae_dtype": torch.float32,
        "weight_dtype": torch.float32,
        "text_encoding_strategy": None,
        "tokenize_strategy": None,
        "val_dataloader": batches,
        "validation_steps": len(batches) if limit is None else limit,
        "validation_timesteps": timesteps,
        "validation_total_steps": len(batches) * len(timesteps) if total is None else total,
        "train_text_encoder": False,
        "train_unet": True,
        "epoch": epoch,
        "global_step": global_step,
        "is_tracking": True,
        "train_loss_recorder": validation.training,
    }
    kwargs.update(options)
    validation.trainer._run_validation_loop(**kwargs)
    return kwargs


def _postfixes(validation, mode="step"):
    key = "val_avg_loss" if mode == "step" else "val_epoch_avg_loss"
    return [entry.args[0][key] for entry in validation.progress.set_postfix.call_args_list]


def test_repeated_step_validation_does_not_keep_previous_checkpoint_losses(validation):
    for epoch, losses, step in [(0, [1, 3], 10), (0, [5, 7], 20), (1, [9, 11], 30)]:
        _run(validation, losses, epoch=epoch, global_step=step)

    assert validation.tensorboard_logs == [
        ({"loss/validation/step_average": 2, "loss/validation/step_divergence": -1}, 10),
        ({"loss/validation/step_average": 6, "loss/validation/step_divergence": 3}, 20),
        ({"loss/validation/step_average": 10, "loss/validation/step_divergence": 7}, 30),
    ]
    assert [entry["epoch"] for entry in validation.wandb_logs] == [1, 1, 2]
    assert [entry["global_step"] for entry in validation.wandb_logs] == [10, 20, 30]


def test_once_per_epoch_validation_preserves_final_metrics_and_logging_axes(validation):
    for epoch, losses in enumerate(([1, 3], [5, 7], [9, 11])):
        _run(validation, losses, mode="epoch", epoch=epoch, global_step=(epoch + 1) * 10)

    assert validation.tensorboard_logs == [
        ({"loss/validation/epoch_average": 2, "loss/validation/epoch_divergence": -1}, 1),
        ({"loss/validation/epoch_average": 6, "loss/validation/epoch_divergence": 3}, 2),
        ({"loss/validation/epoch_average": 10, "loss/validation/epoch_divergence": 7}, 3),
    ]
    assert [entry["epoch"] for entry in validation.wandb_logs] == [1, 2, 3]
    assert [entry["global_step"] for entry in validation.wandb_logs] == [10, 20, 30]


@pytest.mark.parametrize("mode", ["step", "epoch"])
def test_progress_reports_only_observations_from_the_current_pass(validation, mode):
    _run(validation, [1, 3], mode=mode)
    _run(validation, [5, 7], mode=mode, epoch=1)

    assert _postfixes(validation, mode) == [5, 6]
    assert validation.progress.update.call_args_list == [call(1), call(1)]


def test_resumed_validation_and_shorter_pass_do_not_retain_an_old_tail(validation):
    _run(validation, [1, 3, 5], epoch=5)
    _run(validation, [7], epoch=6)

    assert [entry[0]["loss/validation/step_average"] for entry in validation.tensorboard_logs] == [3, 7]
    assert _postfixes(validation) == [7]


@pytest.mark.parametrize(
    "losses,limit,expected,postfixes",
    [([[1, 3], [6], [99]], 2, 8, [2, 4, 14 / 3, 8]), ([[1, 3]], 4, 4, [2, 4])],
)
def test_mean_counts_executed_batch_timestep_pairs(validation, losses, limit, expected, postfixes):
    _run(validation, losses, timesteps=(1, 3), limit=limit, total=99)

    assert validation.tensorboard_logs[-1][0] == {
        "loss/validation/step_average": expected,
        "loss/validation/step_divergence": expected - 3,
    }
    assert _postfixes(validation) == pytest.approx(postfixes)
    assert validation.progress.update.call_count == len(postfixes)
    assert validation.progress_factory.call_args.args[0] == range(99)


def test_validation_preserves_pair_hook_order_arguments_and_training_flags(validation):
    kwargs = _run(validation, [2], timesteps=(11, 17), train_text_encoder=True, train_unet=False)
    assert validation.events == [
        ("start", -10, -1), ("process", 11, 11), ("end", 11, 11),
        ("start", 11, 11), ("process", 17, 17), ("end", 17, 17),
    ]
    batch = kwargs["val_dataloader"][0]
    hook_args = (
        validation.args, validation.accelerator, validation.network, kwargs["text_encoders"], None, batch, torch.float32
    )
    assert validation.trainer.on_step_start.call_args_list == [call(*hook_args, is_train=False)] * 2
    assert validation.trainer.on_validation_step_end.call_args_list == [call(*hook_args)] * 2
    for entry in validation.trainer.process_batch.call_args_list:
        assert entry.args == (
            batch, kwargs["text_encoders"], None, validation.network, None, None,
            torch.float32, torch.float32, validation.accelerator, validation.args, None, None,
        )
        assert entry.kwargs == {"is_train": False, "train_text_encoder": True, "train_unet": False}
    assert [entry.args[0]["timestep"] for entry in validation.progress.set_postfix.call_args_list] == [11, 17]


def test_tracking_disabled_still_computes_current_pass_progress_and_hooks(validation):
    _run(validation, [1, 3], is_tracking=False)
    _run(validation, [5, 7], is_tracking=False)

    assert _postfixes(validation) == [5, 6]
    assert [event[0] for event in validation.events] == ["start", "process", "end"] * 2
    assert validation.tensorboard_logs == []
    assert validation.wandb_logs == []


@pytest.mark.parametrize("mode", ["step", "epoch"])
def test_empty_validation_logs_zero_without_processing_or_stale_history(validation, mode):
    _run(validation, [11], mode=mode)
    for losses, options in [([99], {"limit": 0}), ([99], {"timesteps": ()}), ([], {})]:
        _run(validation, losses, mode=mode, **options)
        assert validation.tensorboard_logs[-1][0] == {
            f"loss/validation/{mode}_average": 0,
            f"loss/validation/{mode}_divergence": -3,
        }
        assert validation.events == []
        validation.progress.update.assert_not_called()
        validation.progress.set_postfix.assert_not_called()


def test_validation_leaves_training_loss_smoothing_unchanged(validation):
    for mode in ("step", "epoch"):
        _run(validation, [100, 200], mode=mode)
    assert validation.training.loss_list == [2, 4]
    assert validation.training.loss_total == 6
    assert validation.training.moving_average == 3

    validation.training.add(epoch=1, step=0, loss=10)
    assert validation.training.loss_list == [10, 4]
    assert validation.training.moving_average == 7
