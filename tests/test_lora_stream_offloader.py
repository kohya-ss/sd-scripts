"""Tests for the H2D-only block swap (``LoRAStreamOffloader``) and its configuration / factory.

The CPU-only tests cover ``BlockSwapConfig`` / ``create_offloader`` / the swap tensor selector. The CUDA tests run a
small homogeneous block model through the offloader in training (forward + backward with gradient checkpointing) and
forward-only modes and compare the results with a fully resident reference model.
"""

import argparse
import copy

import pytest
import torch
import torch.nn as nn
import torch.utils.checkpoint

from library.custom_offloading_utils import (
    BlockSwapConfig,
    LoRAStreamOffloader,
    ModelOffloader,
    create_offloader,
    default_swap_tensor_selector,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.register_buffer("scale", torch.ones(1))
        self.gradient_checkpointing = False

    def _forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x)))) * self.scale

    def forward(self, x):
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)


class LoRAAdapter(nn.Module):
    """Minimal LoRA applied the way the sd-scripts networks do it: the adapter is *not* a submodule of the block, and it
    replaces the ``forward`` of the target Linear, so the block's internals require grad even if its input does not."""

    def __init__(self, target: nn.Linear, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(target.in_features, rank, bias=False)
        self.up = nn.Linear(rank, target.out_features, bias=False)
        nn.init.normal_(self.down.weight, std=0.1)
        nn.init.normal_(self.up.weight, std=0.1)
        self.org_forward = target.forward
        target.forward = self.forward

    def forward(self, x):
        return self.org_forward(x) + self.up(self.down(x))


class Model(nn.Module):
    def __init__(self, num_blocks: int, dim: int = 32):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim) for _ in range(num_blocks)])
        self.adapters = nn.ModuleList([LoRAAdapter(b.fc1) for b in self.blocks])  # LoRA on fc1 of every block
        self.head = nn.Linear(dim, dim)
        self.offloader = None
        self.blocks_to_swap = 0

    @property
    def gradient_checkpointing(self):
        return self.blocks[0].gradient_checkpointing

    @gradient_checkpointing.setter
    def gradient_checkpointing(self, value):
        for b in self.blocks:
            b.gradient_checkpointing = value

    def enable_block_swap(self, blocks_to_swap: int, config: BlockSwapConfig):
        self.blocks_to_swap = blocks_to_swap
        self.offloader = create_offloader("block", self.blocks, len(self.blocks), blocks_to_swap, config)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if self.offloader is not None:
                self.offloader.wait_for_block(i)
            x = block(x)
            if self.offloader is not None:
                self.offloader.submit_move_blocks_forward(self.blocks, i)
        return self.head(x)


def freeze_base(model: Model):
    model.blocks.requires_grad_(False)
    model.head.requires_grad_(False)


def _args(**kwargs):
    ns = argparse.Namespace(block_swap_h2d_only=False, block_swap_ring_size=2, use_pinned_memory_for_block_swap=False, gradient_checkpointing=False)
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns


# ----------------------------------------------------------------------------- CPU tests


def test_default_swap_tensor_selector_picks_linear_weights_only():
    block = Block(8)
    jobs = default_swap_tensor_selector(block)
    assert [(m.__class__.__name__, name) for m, name in jobs] == [("Linear", "weight"), ("Linear", "weight")]
    assert jobs[0][0] is block.fc1 and jobs[1][0] is block.fc2


def test_from_args_defaults_to_classic():
    cfg = BlockSwapConfig.from_args(_args(), torch.device("cpu"), supports_backward=True)
    assert cfg.h2d_only is False and cfg.ring_size == 2 and cfg.use_pinned_memory is False and cfg.supports_backward is True


def test_from_args_tolerates_missing_knobs():
    cfg = BlockSwapConfig.from_args(argparse.Namespace(), torch.device("cpu"), supports_backward=False)
    assert cfg.h2d_only is False and cfg.ring_size == 2


def test_from_args_h2d_only_requires_gradient_checkpointing_for_training():
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        BlockSwapConfig.from_args(_args(block_swap_h2d_only=True), torch.device("cpu"), supports_backward=True)
    # inference (no backward) does not need it
    cfg = BlockSwapConfig.from_args(_args(block_swap_h2d_only=True), torch.device("cpu"), supports_backward=False)
    assert cfg.h2d_only is True
    cfg = BlockSwapConfig.from_args(_args(block_swap_h2d_only=True, gradient_checkpointing=True), torch.device("cpu"), supports_backward=True)
    assert cfg.h2d_only is True


def test_from_args_rejects_invalid_ring_size():
    with pytest.raises(ValueError, match="ring_size"):
        BlockSwapConfig.from_args(_args(block_swap_ring_size=0), torch.device("cpu"), supports_backward=False)


def test_create_offloader_classic():
    blocks = nn.ModuleList([Block(8) for _ in range(4)])
    off = create_offloader("block", blocks, 4, 2, BlockSwapConfig(torch.device("cpu"), supports_backward=False))
    assert isinstance(off, ModelOffloader)
    assert off.blocks_to_swap == 2 and off.forward_only is True


@cuda_only
def test_create_offloader_h2d_only_and_streaming_placement():
    blocks = nn.ModuleList([Block(8) for _ in range(10)])
    blocks.requires_grad_(False)
    cfg = BlockSwapConfig(torch.device("cuda"), supports_backward=True, h2d_only=True, ring_size=2)
    off = create_offloader("block", blocks, 10, 4, cfg)
    assert isinstance(off, LoRAStreamOffloader)
    assert off.S == 4 and off.B == 2
    assert off.stream_idx == [1, 3, 6, 8]  # evenly spaced (midpoint formula)
    assert sum(off.is_stream) == 4


@cuda_only
def test_h2d_only_rejects_trainable_base_weights():
    blocks = nn.ModuleList([Block(8) for _ in range(4)])  # requires_grad=True by default
    cfg = BlockSwapConfig(torch.device("cuda"), supports_backward=True, h2d_only=True)
    with pytest.raises(AssertionError, match="frozen base weights"):
        create_offloader("block", blocks, 4, 2, cfg)


# ----------------------------------------------------------------------------- CUDA tests


def _run_step(model: Model, x: torch.Tensor):
    out = model(x)
    loss = out.float().pow(2).mean()
    loss.backward()
    grads = [a.up.weight.grad.detach().clone() for a in model.adapters] + [a.down.weight.grad.detach().clone() for a in model.adapters]
    return out.detach().clone(), grads


@cuda_only
@pytest.mark.parametrize("ring_size", [1, 2, 3])
@pytest.mark.parametrize("use_pinned_memory", [False, True])
@pytest.mark.parametrize("blocks_to_swap", [1, 3, 6])
def test_h2d_only_training_matches_reference(blocks_to_swap, use_pinned_memory, ring_size):
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_blocks = 8
    ref = Model(num_blocks)
    freeze_base(ref)
    ref.gradient_checkpointing = True
    swapped = copy.deepcopy(ref)  # same weights, starts on CPU like a model loaded for block swap
    ref.to(device)

    cfg = BlockSwapConfig(device, supports_backward=True, h2d_only=True, ring_size=ring_size, use_pinned_memory=use_pinned_memory)
    swapped.enable_block_swap(blocks_to_swap, cfg)
    # non-swapped parts are moved by the script; the offloader places the blocks
    swapped.adapters.to(device)
    swapped.head.to(device)
    swapped.offloader.prepare_block_devices_before_forward(swapped.blocks)

    # streaming blocks keep their Linear weights on the CPU (masters) between loads, everything else is on the device
    off = swapped.offloader
    for i, b in enumerate(swapped.blocks):
        assert b.norm.weight.device.type == "cuda" and b.scale.device.type == "cuda" and b.fc1.bias.device.type == "cuda"
        if not off.is_stream[i]:
            assert b.fc1.weight.device.type == "cuda"

    x = torch.randn(2, 5, 32, device=device)
    for step in range(3):  # several steps: ring reuse, forward<->backward turn-around, no cold start on step 2+
        ref.zero_grad(set_to_none=True)
        swapped.zero_grad(set_to_none=True)
        out_ref, grads_ref = _run_step(ref, x)
        out_sw, grads_sw = _run_step(swapped, x)
        torch.cuda.synchronize()
        assert torch.allclose(out_ref, out_sw, atol=1e-5, rtol=1e-5), f"step {step}: output mismatch"
        for g_ref, g_sw in zip(grads_ref, grads_sw):
            assert torch.allclose(g_ref, g_sw, atol=1e-5, rtol=1e-5), f"step {step}: grad mismatch"
        # no gradient must reach the frozen streamed weights (they are not even on the device)
        for b in swapped.blocks:
            assert b.fc1.weight.grad is None and b.fc2.weight.grad is None
        x = x + 0.1


@cuda_only
def test_h2d_only_re_prepare_keeps_masters_on_cpu():
    """Re-preparing (as done around in-training sampling) must not drag the CPU masters onto the GPU."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = Model(6)
    freeze_base(model)
    model.gradient_checkpointing = True
    cfg = BlockSwapConfig(device, supports_backward=True, h2d_only=True, ring_size=2)
    model.enable_block_swap(3, cfg)
    model.adapters.to(device)
    model.head.to(device)
    off = model.offloader
    x = torch.randn(1, 4, 32, device=device)

    for _ in range(2):
        off.prepare_block_devices_before_forward(model.blocks)
        _run_step(model, x)
        torch.cuda.synchronize()
        # after a step, the masters still back the streaming blocks' weights on the CPU
        for i in off.stream_idx:
            for p in off.cpu_master[i]:
                assert p.device.type == "cpu"
        assert len(off.ring_flat) == 2


@cuda_only
def test_h2d_only_forward_only_matches_reference():
    """Inference-style use: supports_backward=False, repeated forward passes (wrap-around preload / self-heal)."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    ref = Model(8)
    freeze_base(ref)
    ref.adapters.requires_grad_(False)
    swapped = copy.deepcopy(ref)
    ref.to(device)

    cfg = BlockSwapConfig(device, supports_backward=False, h2d_only=True, ring_size=2)
    swapped.enable_block_swap(5, cfg)
    swapped.adapters.to(device)
    swapped.head.to(device)
    swapped.offloader.prepare_block_devices_before_forward(swapped.blocks)

    with torch.no_grad():
        for _ in range(3):
            x = torch.randn(2, 5, 32, device=device)
            out_ref = ref(x)
            out_sw = swapped(x)
            torch.cuda.synchronize()
            assert torch.allclose(out_ref, out_sw, atol=1e-5, rtol=1e-5)


@cuda_only
def test_h2d_only_switch_between_training_and_inference():
    """The training offloader can be switched to forward-only (sampling) and back, like switch_block_swap_for_*."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    ref = Model(8)
    freeze_base(ref)
    ref.gradient_checkpointing = True
    swapped = copy.deepcopy(ref)
    ref.to(device)

    cfg = BlockSwapConfig(device, supports_backward=True, h2d_only=True, ring_size=2)
    swapped.enable_block_swap(4, cfg)
    swapped.adapters.to(device)
    swapped.head.to(device)
    off = swapped.offloader
    off.prepare_block_devices_before_forward(swapped.blocks)
    x = torch.randn(2, 5, 32, device=device)

    out_ref, grads_ref = _run_step(ref, x)
    out_sw, grads_sw = _run_step(swapped, x)
    assert torch.allclose(out_ref, out_sw, atol=1e-5, rtol=1e-5)

    # sampling
    off.set_forward_only(True)
    off.prepare_block_devices_before_forward(swapped.blocks)
    with torch.no_grad():
        for _ in range(2):
            assert torch.allclose(ref(x), swapped(x), atol=1e-5, rtol=1e-5)

    # back to training
    off.set_forward_only(False)
    off.prepare_block_devices_before_forward(swapped.blocks)
    ref.zero_grad(set_to_none=True)
    swapped.zero_grad(set_to_none=True)
    out_ref, grads_ref = _run_step(ref, x)
    out_sw, grads_sw = _run_step(swapped, x)
    torch.cuda.synchronize()
    assert torch.allclose(out_ref, out_sw, atol=1e-5, rtol=1e-5)
    for g_ref, g_sw in zip(grads_ref, grads_sw):
        assert torch.allclose(g_ref, g_sw, atol=1e-5, rtol=1e-5)


@cuda_only
def test_h2d_only_without_gradient_checkpointing_raises_version_error():
    """Documents why --gradient_checkpointing is required: the ring buffer overwrite advances the autograd version."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    model = Model(8)
    freeze_base(model)
    model.gradient_checkpointing = False
    cfg = BlockSwapConfig(device, supports_backward=True, h2d_only=True, ring_size=1)
    model.enable_block_swap(6, cfg)
    model.adapters.to(device)
    model.head.to(device)
    model.offloader.prepare_block_devices_before_forward(model.blocks)
    x = torch.randn(1, 4, 32, device=device)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        _run_step(model, x)


@cuda_only
@pytest.mark.parametrize("ring_size", [1, 2])
def test_h2d_only_adapters_only_on_later_blocks(ring_size):
    """The first block that takes part in the backward pass has an input that does not require grad, so its backward
    hook fires before its own backward (PyTorch fires full backward hooks at output time when no positional input
    requires grad). The offloader must not evict that block in its hook. Here the adapters of the early blocks are
    frozen, so the backward starts in the middle of the block list, at a streaming block with rank >= ring_size."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_blocks = 8
    ref = Model(num_blocks)
    freeze_base(ref)
    ref.gradient_checkpointing = True
    swapped = copy.deepcopy(ref)
    ref.to(device)

    cfg = BlockSwapConfig(device, supports_backward=True, h2d_only=True, ring_size=ring_size)
    swapped.enable_block_swap(6, cfg)  # streaming indices [0, 2, 3, 4, 6, 7]
    off = swapped.offloader
    first_trainable = 6  # streaming block with rank 4 >= ring_size; blocks 0..5 take no part in the backward
    assert off.is_stream[first_trainable] and off.rank[first_trainable] >= ring_size
    for m in (ref, swapped):
        for i in range(first_trainable):
            m.adapters[i].requires_grad_(False)
    swapped.adapters.to(device)
    swapped.head.to(device)
    off.prepare_block_devices_before_forward(swapped.blocks)

    x = torch.randn(2, 5, 32, device=device)
    for step in range(2):
        ref.zero_grad(set_to_none=True)
        swapped.zero_grad(set_to_none=True)
        out_ref = ref(x)
        out_sw = swapped(x)
        out_ref.float().pow(2).mean().backward()
        out_sw.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        assert torch.allclose(out_ref, out_sw, atol=1e-5, rtol=1e-5)
        for i in range(first_trainable, num_blocks):
            assert torch.allclose(ref.adapters[i].up.weight.grad, swapped.adapters[i].up.weight.grad, atol=1e-5, rtol=1e-5), f"step {step} block {i}"


@cuda_only
def test_h2d_only_keyword_only_inputs_fail_loudly():
    """A forward loop that passes the activation as a keyword argument makes the backward hook fire before the block's
    backward; the offloader then skips the hook work and the next streaming block is not loaded, which must surface as
    an error rather than silently wrong results."""

    class KwModel(Model):
        def forward(self, x):
            for i, block in enumerate(self.blocks):
                self.offloader.wait_for_block(i)
                x = block(x=x)  # keyword-only activation input (not allowed)
                self.offloader.submit_move_blocks_forward(self.blocks, i)
            return self.head(x)

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = KwModel(8)
    freeze_base(model)
    model.gradient_checkpointing = True
    cfg = BlockSwapConfig(device, supports_backward=True, h2d_only=True, ring_size=1)
    model.enable_block_swap(6, cfg)
    model.adapters.to(device)
    model.head.to(device)
    model.offloader.prepare_block_devices_before_forward(model.blocks)
    x = torch.randn(1, 4, 32, device=device)
    with pytest.raises(RuntimeError):
        model(x).float().pow(2).mean().backward()
