import sys
import types
from unittest.mock import patch

import pytest
import torch
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR

from library.optimizer import get_scheduler_fix
from train_network import setup_parser


class SchedulerWithSteps(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps=0, num_training_steps=1000, last_epoch=-1):
        self.received = (num_warmup_steps, num_training_steps)
        super().__init__(optimizer, lambda step: 1.0, last_epoch)


class SchedulerWithTotalSteps(LambdaLR):
    def __init__(self, optimizer, num_training_steps=1000):
        self.received = num_training_steps
        super().__init__(optimizer, lambda step: 1.0)


@pytest.fixture(autouse=True)
def custom_scheduler_module(monkeypatch):
    module = types.ModuleType("custom_sched_for_test")
    module.SchedulerWithSteps = SchedulerWithSteps
    module.SchedulerWithTotalSteps = SchedulerWithTotalSteps
    monkeypatch.setitem(sys.modules, "custom_sched_for_test", module)


def get_scheduler(argv, num_processes=1):
    with patch("sys.argv", [""] + argv):
        args = setup_parser().parse_args()
    optimizer = torch.optim.AdamW([Parameter(torch.zeros(1))], lr=1e-3)
    return get_scheduler_fix(args, optimizer, num_processes)


def test_pass_steps_to_custom_scheduler():
    scheduler = get_scheduler(
        ["--lr_scheduler_type", "custom_sched_for_test.SchedulerWithSteps", "--max_train_steps", "500", "--lr_warmup_steps", "0.1"],
        num_processes=2,
    )
    assert scheduler.received == (100, 1000)


def test_explicit_scheduler_args_take_priority():
    scheduler = get_scheduler(
        [
            "--lr_scheduler_type",
            "custom_sched_for_test.SchedulerWithSteps",
            "--max_train_steps",
            "500",
            "--lr_warmup_steps",
            "10",
            "--lr_scheduler_args",
            "num_training_steps=7",
            "num_warmup_steps=3",
        ]
    )
    assert scheduler.received == (3, 7)


def test_pass_only_accepted_arguments():
    scheduler = get_scheduler(["--lr_scheduler_type", "custom_sched_for_test.SchedulerWithTotalSteps", "--max_train_steps", "500"])
    assert scheduler.received == 500


def test_warmup_steps_rejected_if_not_accepted():
    with pytest.raises(ValueError):
        get_scheduler(
            ["--lr_scheduler_type", "custom_sched_for_test.SchedulerWithTotalSteps", "--max_train_steps", "500", "--lr_warmup_steps", "10"]
        )


def test_torch_scheduler_unchanged():
    scheduler = get_scheduler(["--lr_scheduler_type", "StepLR", "--max_train_steps", "500", "--lr_scheduler_args", "step_size=10"])
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
