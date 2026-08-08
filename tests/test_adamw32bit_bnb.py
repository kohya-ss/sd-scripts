from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

bnb = pytest.importorskip("bitsandbytes")

from library.adamw_bnb import BNB32_MIN_8BIT_SIZE, AdamWBnb
from library.adamw8bit_fast import AdamW8bitFast


def _optimizer_args(optimizer_args=None):
    return SimpleNamespace(
        optimizer_type="AdamWBnb",
        use_8bit_adam=False,
        use_lion_optimizer=False,
        fused_backward_pass=False,
        gradient_accumulation_steps=1,
        optimizer_args=optimizer_args,
        learning_rate=1e-3,
    )


def test_optimizer_selection_accepts_adamwbnb_and_forces_32bit_path():
    from library.train_util import get_optimizer

    param = torch.nn.Parameter(torch.ones(8))
    optimizer_name, optimizer_args, optimizer = get_optimizer(_optimizer_args(), [param])

    assert optimizer_name == "library.adamw_bnb.AdamWBnb"
    assert optimizer_args == ""
    assert isinstance(optimizer, AdamWBnb)
    assert isinstance(optimizer, AdamW8bitFast)
    assert optimizer.args.min_8bit_size == BNB32_MIN_8BIT_SIZE


def test_matching_legacy_minimum_is_accepted():
    optimizer = AdamWBnb(
        [torch.nn.Parameter(torch.ones(8))],
        min_8bit_size=BNB32_MIN_8BIT_SIZE,
    )

    assert optimizer.args.min_8bit_size == BNB32_MIN_8BIT_SIZE


def test_conflicting_minimum_is_rejected():
    with pytest.raises(ValueError, match="fixes min_8bit_size"):
        AdamWBnb(
            [torch.nn.Parameter(torch.ones(8))],
            min_8bit_size=4096,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_large_tensor_uses_float32_states_and_fast_route():
    param = torch.nn.Parameter(torch.randn(8192, device="cuda", dtype=torch.float32))
    optimizer = AdamWBnb([param], lr=1e-3)
    param.grad = torch.randn_like(param)

    can_use_fast_path, device, reason = optimizer._fast_path_device()
    assert can_use_fast_path
    assert device == param.device
    assert reason is None

    optimizer.step()

    state = optimizer.state[param]
    assert state["state1"].dtype == torch.float32
    assert state["state2"].dtype == torch.float32
