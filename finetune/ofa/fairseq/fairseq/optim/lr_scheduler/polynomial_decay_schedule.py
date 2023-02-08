# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import II

from fairseq.dataclass import FairseqDataclass
from fairseq.optim.lr_scheduler import FairseqLRScheduler, register_lr_scheduler


@dataclass
class PolynomialDecayLRScheduleConfig(FairseqDataclass):
    warmup_updates: int = field(
        default=0,
        metadata={"help": "warmup the learning rate linearly for the first N updates"},
    )
    warmup_ratio: float = field(
        default=0,
        metadata={"help": "warmup ratio"},
    )
    force_anneal: Optional[int] = field(
        default=None,
        metadata={"help": "force annealing at specified epoch"},
    )
    end_learning_rate: float = field(
        default=0.0,
        metadata={"help": "learning rate to decay to"},
    )
    power: float = field(
        default=1.0,
        metadata={"help": "decay exponent"},
    )
    total_num_update: Optional[float] = field(
        default=1000000,
        metadata={"help": "total number of updates over which to decay learning rate"},
    )
    lr: List[float] = II("optimization.lr")


@register_lr_scheduler("polynomial_decay", dataclass=PolynomialDecayLRScheduleConfig)
class PolynomialDecayLRSchedule(FairseqLRScheduler):
    """Decay the LR on a fixed schedule."""

    def __init__(self, cfg: PolynomialDecayLRScheduleConfig, optimizer):
        super().__init__(cfg, optimizer)

        assert cfg.total_num_update > 0
        # set defaults
        cfg.warmup_updates = getattr(cfg, 'warmup_updates', 0) or 0

        self.lr = cfg.lr[0]
        self.warmup_updates = cfg.warmup_updates
        if self.warmup_updates > 0:
            self.warmup_factor = 1.0 / self.warmup_updates
        else:
            self.warmup_factor = 1
        self.end_learning_rate = cfg.end_learning_rate
        self.total_num_update = cfg.total_num_update
        self.power = cfg.power
        self.optimizer.set_lr(self.warmup_factor * self.lr)

    def get_next_lr(self, epoch):
        lrs = self.cfg.lr
        if self.cfg.force_anneal is None or epoch < self.cfg.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if self.warmup_updates > 0 and num_updates <= self.warmup_updates:
            self.warmup_factor = num_updates / float(self.warmup_updates)
            lr = self.warmup_factor * self.lr
        elif num_updates >= self.total_num_update:
            lr = self.end_learning_rate
        else:
            warmup = self.warmup_updates
            lr_range = self.lr - self.end_learning_rate
            pct_remaining = 1 - (num_updates - warmup) / (self.total_num_update - warmup)
            lr = lr_range * pct_remaining ** (self.power) + self.end_learning_rate
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()

    def reinit(self, total_num_update, num_updates):
        # only enable this when set warmup_ratio
        if self.cfg.warmup_ratio <= 0:
            return
        # re init this according to the real number of updates
        self.total_num_update = total_num_update
        self.warmup_updates = int(self.total_num_update * self.cfg.warmup_ratio)
        if num_updates > 0:
            self.warmup_factor = min(1.0, num_updates / float(self.warmup_updates))
            self.step_update(num_updates)
        else:
            self.warmup_factor = 1.0 / self.warmup_updates
            self.optimizer.set_lr(self.warmup_factor * self.lr)
        print('Total steps {}, warmup steps {}, warmup_factor {}'.format(self.total_num_update, self.warmup_updates,
                                                                         self.warmup_factor))