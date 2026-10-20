from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
import gc
import os
import time
from typing import Any, Optional, Union, Callable, Tuple
import torch
import torch.nn as nn


# Keep these functions here for portability, and private to avoid confusion with the ones in device_utils.py
def _clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs: list[Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
    #     print(module_to_cpu.__class__, module_to_cuda.__class__)
    #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
    #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    # print(
                    #     f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, so not swapping and moving to device"
                    # )
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.Stream(device="cuda")
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs: list[Tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor]] = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=True)


class Offloader:
    """
    common offloading class
    """

    def __init__(self, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        self.thread_pool = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        self.cuda_available = device.type == "cuda"

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(f"Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}")

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(f"Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter() - start_time:.2f}s")
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        self.futures[block_idx_to_cuda] = self.thread_pool.submit(
            move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
        )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


# Gradient tensors
_grad_t = Union[tuple[torch.Tensor, ...], torch.Tensor]


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        blocks: Union[list[nn.Module], nn.ModuleList],
        blocks_to_swap: int,
        device: torch.device,
        supports_backward: bool = True,
        debug: bool = False,
    ):
        super().__init__(len(blocks), blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        # switching must wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(
        self, blocks: Union[list[nn.Module], nn.ModuleList], block_index: int
    ) -> Optional[Callable[[nn.Module, _grad_t, _grad_t], Union[None, _grad_t]]]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module: nn.Module, grad_input: _grad_t, grad_output: _grad_t):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: Union[list[nn.Module], nn.ModuleList]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"Prepare block devices before forward")

        # wait for all pending transfers
        for block_idx in list(self.futures.keys()):
            self._wait_blocks_move(block_idx)

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, torch.device("cpu"))  # make sure weights are on cpu

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: Union[list[nn.Module], nn.ModuleList], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        # this works for forward-only offloading. move upstream blocks to cuda
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)


# endregion


# region H2D-only block swap (ported from Musubi Tuner)


def default_swap_tensor_selector(block: nn.Module) -> list[tuple[nn.Module, str]]:
    """Default streaming selection rule: the ``weight`` of every ``*Linear`` module.

    An offloader can be given a different selector to also stream tensors that a quantization scheme hangs off
    the same modules (per-group scales etc.); each entry is ``(module, attribute_name)`` where the attribute is a
    parameter or a registered buffer of the module.
    """
    jobs = []
    for _, module in block.named_modules():
        if hasattr(module, "weight") and module.weight is not None and module.__class__.__name__.endswith("Linear"):
            jobs.append((module, "weight"))
    return jobs


@dataclass
class BlockSwapConfig:
    """
    Construction policy for a block-swap offloader, assembled once by the training/inference script and
    passed through each architecture's ``enable_block_swap`` to ``create_offloader``.

    It holds everything the offloader constructor needs that is *not* architecture-specific. The
    architecture-specific arguments (the block-type label, the block list, and its counts) are supplied by
    ``enable_block_swap`` per block list. Adding a new offloader knob (or a whole new offloader type) means
    adding a field here and a branch in ``create_offloader`` -- the per-architecture ``enable_block_swap``
    signatures stay ``(blocks_to_swap, config)`` and never change.
    """

    device: torch.device
    supports_backward: bool
    use_pinned_memory: bool = False
    h2d_only: bool = False  # frozen-base (LoRA / LoHa / LoKr) only: H2D-only streaming, no device->host copy
    ring_size: int = 2  # (h2d_only) number of GPU ring buffers for streamed blocks; 2 = double buffering
    debug: bool = False

    @classmethod
    def from_args(cls, args, device: torch.device, supports_backward: bool) -> "BlockSwapConfig":
        """Build from a parsed-args namespace, tolerating scripts whose parser lacks the optional knobs."""
        h2d_only = getattr(args, "block_swap_h2d_only", False)

        # H2D-only streams frozen weights into a reused GPU ring buffer and loads them with an in-place copy_,
        # which bumps the autograd version of the saved weight view. Gradient checkpointing re-reads the weight
        # at recompute time (a short, safe window), so the version stays consistent; without it the backward
        # pass fails with an opaque "modified by an inplace operation ... version N expected M" error that does
        # not point at the offloader. Fail early with an actionable message instead. (Inference / forward-only
        # has no backward, so it is unaffected.)
        if h2d_only and supports_backward and not getattr(args, "gradient_checkpointing", False):
            raise ValueError(
                "--block_swap_h2d_only requires --gradient_checkpointing for training. H2D-only block swap streams"
                " frozen weights through a reused GPU ring buffer, which advances the autograd version of weights"
                " saved for backward; gradient checkpointing re-reads them at recompute time and avoids this."
                " / --block_swap_h2d_only は学習時に --gradient_checkpointing が必須です（リングバッファの上書きで"
                "backward 用に保存された重みの version が進むため。gradient checkpointing は再計算時に読み直すので回避できます）。"
            )

        ring_size = getattr(args, "block_swap_ring_size", 2)
        if ring_size < 1:
            raise ValueError("--block_swap_ring_size must be >= 1")

        return cls(
            device=device,
            supports_backward=supports_backward,
            use_pinned_memory=getattr(args, "use_pinned_memory_for_block_swap", False),
            h2d_only=h2d_only,
            ring_size=ring_size,
        )


def create_offloader(block_type: str, blocks: list[nn.Module], num_blocks: int, blocks_to_swap: int, config: BlockSwapConfig):
    """
    Create the block-swap offloader for one block list. This is the single place that selects the offloader
    implementation from ``config``; ``enable_block_swap`` only supplies the per-block-list arguments, so a new
    offloader type plugs in here without touching any architecture.
    """
    if config.h2d_only:
        # H2D-only streaming for frozen-base (LoRA) training: keep a CPU master, copy Host->Device only.
        return LoRAStreamOffloader(
            block_type,
            blocks,
            num_blocks,
            blocks_to_swap,
            config.supports_backward,
            config.device,
            ring_size=config.ring_size,
            use_pinned_memory=config.use_pinned_memory,
            debug=config.debug,
        )
    # classic exchange offloader. `use_pinned_memory` is not used by this implementation.
    return ModelOffloader(blocks, blocks_to_swap, config.device, supports_backward=config.supports_backward, debug=config.debug)


class _DirectCopier:
    """
    Transfer engine for ``LoRAStreamOffloader`` when the masters are *pinned*: copies a flat pinned host
    buffer straight into a flat device buffer on a private copy stream (async Host->Device only).

    This is the seam that isolates all CUDA stream/event primitives from the residency logic. The residency
    side only ever calls ``submit`` / ``wait``; ``_StagedCopier`` (pageable master + worker thread) and a
    future non-CUDA device shim plug in behind the same interface without the residency code changing. Keep
    every ``torch.cuda.*`` transfer primitive inside copiers like this one.
    """

    def __init__(self, device: torch.device, debug: bool = False):
        self.device = device
        self.debug = debug
        self.copy_stream = torch.cuda.Stream(device=device)
        self._events = {}  # key -> cuda.Event recording H2D completion
        self._xfer = {}  # key -> (ev_a, ev_b) transfer-timing events (debug only)

    def submit(self, key, dst_flat: torch.Tensor, src_flat: torch.Tensor, gate_event=None):
        """Begin the H2D copy of ``src_flat`` into ``dst_flat``, gated behind ``gate_event`` (compute done with dst)."""
        ev_a = ev_b = None
        with torch.cuda.stream(self.copy_stream):
            if gate_event is not None:
                self.copy_stream.wait_event(gate_event)  # do not overwrite a buffer still used by compute
            if self.debug:
                ev_a = torch.cuda.Event(enable_timing=True)
                ev_b = torch.cuda.Event(enable_timing=True)
                ev_a.record(self.copy_stream)
            dst_flat.copy_(src_flat, non_blocking=True)  # src is pinned -> async H2D
            if self.debug:
                ev_b.record(self.copy_stream)
                self._xfer[key] = (ev_a, ev_b)
        self._events[key] = self.copy_stream.record_event()

    def wait(self, key):
        """Return the H2D-completion event for ``key`` (None if nothing is in flight for it)."""
        return self._events.get(key)

    def pop_xfer_timing(self, key):
        """Pop the (start, end) transfer-timing events recorded for ``key`` (debug only)."""
        return self._xfer.pop(key, None)

    def sync(self):
        self.copy_stream.synchronize()

    def reset(self):
        self._events.clear()
        self._xfer.clear()


class _StagedCopier:
    """
    Transfer engine for ``LoRAStreamOffloader`` when the masters are kept in *pageable* host memory: each
    H2D is staged through a small pool of pinned buffers filled by a background worker thread. Same
    ``submit`` / ``wait`` interface as ``_DirectCopier``, so the residency code is identical.

    Why: pinning every streaming master (what ``_DirectCopier`` needs) can cost tens of GB of non-pageable
    "shared GPU memory" (notably on Windows). Here the masters stay pageable and only ``num_staging`` pinned
    buffers exist. A pageable->device copy is, inside the CUDA driver, already a synchronous serial
    pageable->pinned->device transfer; doing it ourselves on a worker thread lets the pageable->pinned
    memcpy overlap the previous block's pinned->device DMA and -- crucially -- keeps it off the main thread
    so compute kernel launches are never blocked. That is the whole speed difference versus the old
    synchronous pageable path this replaces.

    Hazards handled:
      - a staging buffer is reused only after the H2D that consumed it has completed (one consume-event per
        buffer; the worker waits on it before overwriting);
      - ``wait`` blocks the caller until the worker has actually enqueued the H2D and recorded its
        completion event, so the residency layer can ``current_stream().wait_event`` it as usual.
    """

    def __init__(self, device: torch.device, num_staging: int = 2, debug: bool = False):
        self.device = device
        self.num_staging = max(1, num_staging)
        self.debug = debug
        self.copy_stream = torch.cuda.Stream(device=device)
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._futures = {}  # key -> Future -> (load_event, (ev_a, ev_b))
        self._xfer = {}  # key -> (ev_a, ev_b) transfer-timing events (debug only)
        self._staging = None  # list[pinned uint8 tensor] (lazily sized to the flat block)
        self._staging_free = None  # list[cuda.Event | None]: the H2D that last consumed each buffer
        self._rr = 0  # round-robin staging index (touched only on the worker thread)

    def _ensure_staging(self, nbytes: int):
        if self._staging is None:
            self._staging = [torch.empty(nbytes, dtype=torch.uint8).pin_memory(device=self.device) for _ in range(self.num_staging)]
            self._staging_free = [None] * self.num_staging

    def _run(self, dst_flat: torch.Tensor, src_flat: torch.Tensor, gate_event):
        # runs on the worker thread: pageable->pinned memcpy, then enqueue the async pinned->device H2D
        idx = self._rr
        self._rr = (idx + 1) % self.num_staging
        staging = self._staging[idx]

        free_ev = self._staging_free[idx]
        if free_ev is not None:
            free_ev.synchronize()  # CPU(worker) waits until the prior H2D released this buffer -- off main thread

        # pageable -> pinned host memcpy. Not `staging.copy_(src_flat)`: torch parallelizes a large CPU copy over all
        # intra-op threads and the OpenMP workers keep spinning afterwards, so the repeated per-block copies kept
        # nearly every core busy (46 of 64 cores on average in a MOSS-Image run). A single-threaded memmove
        # (about 18 GB/s here) is still faster than the PCIe DMA that consumes the buffer, releases the GIL and
        # does not touch the OpenMP pool.
        ctypes.memmove(staging.data_ptr(), src_flat.data_ptr(), src_flat.numel())

        ev_a = ev_b = None
        with torch.cuda.stream(self.copy_stream):
            if gate_event is not None:
                self.copy_stream.wait_event(gate_event)  # do not overwrite a dst still used by compute
            if self.debug:
                ev_a = torch.cuda.Event(enable_timing=True)
                ev_b = torch.cuda.Event(enable_timing=True)
                ev_a.record(self.copy_stream)
            dst_flat.copy_(staging, non_blocking=True)  # pinned -> device, async
            if self.debug:
                ev_b.record(self.copy_stream)
            done = self.copy_stream.record_event()  # H2D complete == this staging buffer is free again
        self._staging_free[idx] = done
        return done, (ev_a, ev_b)

    def submit(self, key, dst_flat: torch.Tensor, src_flat: torch.Tensor, gate_event=None):
        self._ensure_staging(src_flat.numel())  # flat uint8 -> numel == byte size
        self._futures[key] = self._pool.submit(self._run, dst_flat, src_flat, gate_event)

    def wait(self, key):
        fut = self._futures.get(key)
        if fut is None:
            return None
        load_event, xfer = fut.result()  # block until the worker enqueued the H2D (tiny if it ran ahead)
        if self.debug and xfer[0] is not None:
            self._xfer[key] = xfer
        return load_event

    def pop_xfer_timing(self, key):
        return self._xfer.pop(key, None)

    def _flush(self):
        for fut in list(self._futures.values()):
            fut.result()

    def sync(self):
        self._flush()
        self.copy_stream.synchronize()

    def reset(self):
        self._flush()
        self.copy_stream.synchronize()  # ensure all in-flight H2D are done before clearing staging state
        self._futures.clear()
        self._xfer.clear()
        if self._staging is not None:
            self._staging_free = [None] * self.num_staging

    def __del__(self):
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.shutdown(wait=False)


class LoRAStreamOffloader:
    """
    H2D-only offloader for training where the base weights are frozen (e.g. LoRA / LoHa / LoKr).

    The classic block swap (``ModelOffloader``) *exchanges* a block: it copies one block back to the
    CPU (Device->Host) while copying the next one in (Host->Device). When the base weights never change,
    the CPU already holds an identical copy, so the D2H half is pure overhead. This offloader keeps a
    permanent (pinned) master copy of every streamed weight on the CPU and only ever copies Host->Device,
    removing the D2H transfer and the CPU-side staging memcpy entirely.

    Layout (N = num_blocks, S = blocks_to_swap, B = ring_size):
      - ``N - S`` blocks stay resident on the GPU permanently.
      - ``S`` "streaming" blocks are placed at even intervals; their ``Linear`` weights live on the CPU
        (pinned) as masters and are streamed into a small ring of ``B`` GPU buffers (default 2: one being
        computed on, one being prefetched). Non-Linear params / buffers of streaming blocks stay resident.

    Mechanics:
      - Because the streamed weights are frozen (no ``.grad`` to preserve), we swap the whole ``module.weight``
        *reference* between its CPU master Parameter and a preallocated GPU ring Parameter, instead of the
        ``.data`` storage trick used by ``ModelOffloader``. No per-step ``cudaMalloc``.
      - What is streamed is decided by ``swap_tensor_selector`` (default: every ``*Linear`` ``weight``). A
        selector may return several tensors per module -- e.g. a quantized weight together with its scale
        buffers -- as ``(module, attribute_name)`` pairs; parameters are rebound as ``nn.Parameter`` ring
        views, registered buffers as plain tensor views.
      - All swap weights of a block are packed into a single contiguous (byte) buffer -- one pinned CPU
        buffer per streaming block and one GPU buffer per ring slot -- and the individual weights are
        dtype/shape views into it. Loading a block is therefore a single H2D ``copy_`` instead of one
        per Linear, which removes the per-tensor launch/driver overhead on PCIe/CPU-bound systems.
      - Eviction is free: repoint ``module.weight`` back to the CPU master (no transfer). A ring slot is
        overwritten only when its block is consumed, so the last ``B`` streamed blocks of a pass stay
        resident. On the reverse pass they are exactly the first blocks needed, so the forward<->backward
        turn-around needs no reload.
      - Direction is implicit in the callback: ``submit_move_blocks_forward`` prefetches the block ``B``
        streaming-slots *ahead* (larger index); the backward hook prefetches ``B`` slots *behind* (smaller
        index). No explicit scheduler / no ``prepare_*_before_backward``.
      - ``wait_for_block`` self-heals: if a needed streaming block is not resident in its slot (cold start,
        or an inference pass boundary), it is loaded on demand.

    Requirement for the model's forward loop: the block's activation input(s) must be passed *positionally*
    (``block(x, ...)``, not ``block(x=x, ...)``). The backward hooks are full backward hooks, which PyTorch fires
    after the block's backward only when a positional input requires grad; otherwise they fire when the gradients of
    the outputs become available, before the block's backward, and the slot would be evicted too early.
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        ring_size: int = 2,
        use_pinned_memory: bool = True,
        debug: bool = False,
        swap_tensor_selector=None,
    ):
        self.block_type = block_type
        self._blocks = blocks
        self.num_blocks = num_blocks  # N
        self.blocks_to_swap = blocks_to_swap  # requested S
        self.device = device
        self.use_pinned_memory = use_pinned_memory
        self.supports_backward = supports_backward
        self.forward_only = not supports_backward
        self.swap_tensor_selector = swap_tensor_selector if swap_tensor_selector is not None else default_swap_tensor_selector

        if not debug:
            debug = os.getenv("MUSUBI_TUNER_OFFLOADER_DEBUG", "0") == "1"
        self.debug = debug
        self.debug_interval = int(os.getenv("MUSUBI_TUNER_OFFLOADER_DEBUG_INTERVAL", "10"))  # steps between prints

        assert device.type == "cuda", "LoRAStreamOffloader currently supports CUDA only"

        # ---- streaming placement: S evenly spaced block indices (midpoint formula -> distinct for S <= N) ----
        stream_idx = sorted({((2 * i + 1) * num_blocks) // (2 * blocks_to_swap) for i in range(blocks_to_swap)})
        self.stream_idx = stream_idx
        self.S = len(stream_idx)  # actual streaming count (>=1; dedup is a no-op unless S is close to N)
        self.rank = {b: k for k, b in enumerate(stream_idx)}  # block_idx -> position in stream_idx
        self.is_stream = [b in self.rank for b in range(num_blocks)]
        self.B = min(ring_size, self.S)  # clamp ring to number of streaming blocks

        # ---- finetuning guard: H2D-only must never lose weight updates ----
        self._job_cache = {}  # block_idx -> [(module, attr_name, is_param) per swap tensor]
        for b in stream_idx:
            for m, name, is_param in self._jobs(b):
                assert not (is_param and getattr(m, name).requires_grad), (
                    "LoRAStreamOffloader requires frozen base weights (LoRA / no full fine-tune). "
                    f"Found a trainable swap tensor {name} in block {b}."
                )

        # ---- transfer engine: owns the copy stream and all H2D primitives ----
        # pinned masters -> direct async H2D; pageable masters -> stage through a worker thread + pinned pool
        # (low "shared GPU memory" footprint, e.g. on Windows). Both expose the same submit/wait interface.
        if use_pinned_memory:
            self.copier = _DirectCopier(device, debug=self.debug)
        else:
            self.copier = _StagedCopier(device, num_staging=self.B, debug=self.debug)

        # ---- runtime state (GPU buffers allocated lazily in prepare_block_devices_before_forward) ----
        self.cpu_master = {}  # block_idx -> [CPU (pinned) Parameter / buffer tensor per swap tensor] (views into cpu_flat)
        self.cpu_flat = {}  # block_idx -> flat (pinned) uint8 CPU tensor backing the masters
        self.ring_param = None  # [slot] -> [GPU nn.Parameter / buffer tensor per swap tensor] (views into ring_flat)
        self.ring_flat = None  # [slot] -> flat uint8 GPU tensor backing the ring params
        self._layout = None  # ([byte offset per swap tensor], total bytes) shared by all streaming blocks
        self.in_slot = [None] * self.B  # slot -> block_idx currently bound to this slot (or None)
        self.free_event = [None] * self.B  # slot -> cuda.Event recording when compute finished using the slot

        self._wait_ctx = "fwd"  # context tag for wait/load timing ("fwd" while in the forward loop, "bwd" in hooks)
        if self.debug:
            self._dbg_reset()

        # ---- backward hooks: only where there is work (prefetch this block, or wait the previous one) ----
        if supports_backward:
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self._create_backward_hook(i)
                if hook is not None:
                    self.remove_handles.append(block.register_full_backward_hook(hook))

        print(
            f"LoRAStreamOffloader[{block_type}]: H2D-only block swap. "
            f"{self.S} streaming / {num_blocks} blocks, ring={self.B}, pinned={use_pinned_memory}. "
            f"streaming indices: {stream_idx}"
        )

    # ------------------------------------------------------------------ helpers

    def _jobs(self, block_idx: int) -> list[tuple[nn.Module, str, bool]]:
        """Swap jobs of a block: ``(module, attr_name, is_param)`` per streamed tensor, cached."""
        cached = self._job_cache.get(block_idx)
        if cached is None:
            cached = []
            for m, name in self.swap_tensor_selector(self._blocks[block_idx]):
                if name in m._parameters:
                    is_param = True
                elif name in m._buffers:
                    is_param = False
                else:
                    raise ValueError(
                        f"swap tensor selector returned {name!r}, which is neither a parameter nor a"
                        f" registered buffer of {m.__class__.__name__} (block {block_idx})"
                    )
                cached.append((m, name, is_param))
            self._job_cache[block_idx] = cached
        return cached

    def _bind(self, block_idx: int, tensors: list[torch.Tensor]):
        # parameters arrive as nn.Parameter, buffers as plain tensors; setattr keeps them in the
        # module's _parameters / _buffers dicts respectively (buffer names are already registered)
        for (m, name, _is_param), t in zip(self._jobs(block_idx), tensors):
            setattr(m, name, t)

    @staticmethod
    def _compute_layout(weights: list[torch.Tensor]) -> tuple[list[int], int]:
        """Byte offsets (aligned so ``view(dtype)`` is valid for any element size) and total size of the flat buffer."""
        align = 256
        offsets = []
        total = 0
        for w in weights:
            total = (total + align - 1) // align * align
            offsets.append(total)
            total += w.numel() * w.element_size()
        return offsets, total

    @staticmethod
    def _flat_views(flat: torch.Tensor, weights: list[torch.Tensor], layout: tuple[list[int], int]) -> list[torch.Tensor]:
        """dtype/shape views into a flat uint8 buffer, one per swap tensor, following ``layout``."""
        offsets, _ = layout
        return [flat[off : off + w.numel() * w.element_size()].view(w.dtype).view(w.shape) for off, w in zip(offsets, weights)]

    def _load(self, rank: int, slot: int, ctx: str = "fwd"):
        """Stream the streaming block at ``rank`` into ring ``slot`` (H2D only), repointing weights."""
        blk = self.stream_idx[rank]

        # already resident (e.g. ring survived a re-prepare): the slot content is still identical to the
        # immutable CPU master, so just rebind the weights to the ring -- no transfer needed
        if self.in_slot[slot] == blk:
            self._bind(blk, self.ring_param[slot])
            return

        # evict the slot's current owner: repoint its weights back to the CPU master (no transfer)
        prev = self.in_slot[slot]
        if prev is not None:
            self._bind(prev, self.cpu_master[prev])

        # single coalesced H2D copy of the whole block, gated behind the slot's free_event (compute done
        # with it). The ring params are views into ring_flat, so the copy bumps their autograd version
        # counter: safe under gradient checkpointing (weights are re-read at recompute time, and a slot is
        # only overwritten after its free_event); without checkpointing autograd raises a version error
        # instead of silently using stale-saved weights -- which the old per-tensor `.data.copy_` hid.
        self.copier.submit(blk, self.ring_flat[slot], self.cpu_flat[blk], self.free_event[slot])

        self._bind(blk, self.ring_param[slot])  # reference swap (no .data trick, no cudaMalloc)
        self.in_slot[slot] = blk

        if self.debug:
            c = self._cur[ctx]
            c["loads"] += 1
            xfer = self.copier.pop_xfer_timing(blk)
            if xfer is not None:
                c["xfer_ev"].append(xfer)

    # ------------------------------------------------------------------ public interface (mirrors ModelOffloader)

    def set_forward_only(self, forward_only: bool):
        self.copier.sync()
        self.forward_only = forward_only

    def __del__(self):
        if getattr(self, "supports_backward", False):
            for handle in getattr(self, "remove_handles", []):
                handle.remove()

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.S == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward (H2D-only)")

        first_time = not self.cpu_master  # device placement is only needed on the very first call
        cpu_device = torch.device("cpu")
        for i, b in enumerate(blocks):
            if not self.is_stream[i]:
                if first_time:
                    b.to(self.device)  # resident block: buffers + Linear weights on GPU permanently
                    weighs_to_device(b, self.device)
                continue

            if not first_time:
                # re-prepare (e.g. around in-training sampling): the non-swap parts are already on the device
                # and the CPU masters have stayed on the CPU throughout, so just repoint weights to the masters.
                # IMPORTANT: do NOT call b.to(device) here -- it would drag the CPU masters onto the GPU and
                # they would never be released (every swap block would end up resident).
                self._bind(i, self.cpu_master[i])
                continue

            jobs = self._jobs(i)
            tensors = [getattr(m, name).data for m, name, _ in jobs]
            if self._layout is None:
                self._layout = self._compute_layout(tensors)  # first streaming block defines the shared layout
            flat = torch.empty(self._layout[1], dtype=torch.uint8, device=cpu_device)
            if self.use_pinned_memory:
                flat = flat.pin_memory(device=self.device)
            views = self._flat_views(flat, tensors, self._layout)
            master = []

            if all(t.device.type == "cpu" for t in tensors):
                # CPU-direct: build the flat masters host-to-host, detach the swap tensors from the
                # module so the block's device move cannot drag them, then restore them as the masters.
                # Avoids the H2D+D2H round trip of every streamed tensor that the general path below
                # pays when the model starts on the CPU.
                for (m, name, is_param), t, view in zip(jobs, tensors, views):
                    view.copy_(t)  # host-to-host memcpy into the flat master
                    placeholder = torch.empty(0, dtype=t.dtype, device=cpu_device)
                    if is_param:
                        m._parameters[name].data = placeholder
                    else:
                        m._buffers[name] = placeholder
                b.to(self.device)  # non-swap params/buffers to the device; the placeholders are empty
                for (m, name, is_param), view in zip(jobs, views):
                    if is_param:
                        p = m._parameters[name]
                        p.data = view
                        master.append(p)  # keep the original Parameter object as the persistent master
                    else:
                        m._buffers[name] = view
                        master.append(view)
            else:
                # swap tensors already live on an accelerator: move the whole block to the device
                # (buffers, norms, bias), then pull the swap tensors back into the flat master (one-time D2H)
                b.to(self.device)
                for (m, name, is_param), view in zip(jobs, views):
                    t = getattr(m, name)
                    view.copy_(t.data)
                    if is_param:
                        t.data = view
                        master.append(t)  # keep the original Parameter object as the persistent master
                    else:
                        m._buffers[name] = view
                        master.append(view)

            self.cpu_flat[i] = flat
            self.cpu_master[i] = master

        # validate homogeneous streaming blocks (shared ring template)
        template = self.cpu_master[self.stream_idx[0]]
        for b in self.stream_idx[1:]:
            assert len(self.cpu_master[b]) == len(template), f"block {b} has a different number of swap tensors"
            for p, t in zip(self.cpu_master[b], template):
                assert p.data.shape == t.data.shape and p.data.dtype == t.data.dtype, (
                    f"block {b} swap-tensor shape/dtype differs from the streaming template"
                )

        # preallocate the GPU ring (once); copies happen into these flat buffers, never reallocated
        if self.ring_param is None:
            template_jobs = self._jobs(self.stream_idx[0])
            template_weights = [p.data for p in template]
            self.ring_flat = [torch.empty(self._layout[1], dtype=torch.uint8, device=self.device) for _ in range(self.B)]
            self.ring_param = [
                [
                    nn.Parameter(view, requires_grad=False) if is_param else view
                    for (_m, _name, is_param), view in zip(template_jobs, self._flat_views(flat, template_weights, self._layout))
                ]
                for flat in self.ring_flat
            ]

        # reset transfer state and preload the first B streaming blocks. in_slot deliberately survives:
        # after a completed backward pass the ring already holds ranks 0..B-1, so these loads become
        # no-op rebinds (see _load) instead of redundant cold-start transfers.
        self.free_event = [None] * self.B
        self.copier.reset()
        for k in range(self.B):
            self._load(k, k)

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.S == 0 or not self.is_stream[block_idx]:
            return
        ctx = self._wait_ctx
        if self.debug and ctx == "fwd" and block_idx == self.stream_idx[0]:
            self._dbg_boundary()  # a forward pass is starting -> finalize the previous step's stats
        j = self.rank[block_idx]
        slot = j % self.B
        if self.in_slot[slot] != block_idx:
            # self-healing: not resident (cold start / inference pass boundary / same-direction restart)
            if self.debug:
                self._cur[ctx]["self_heal"] += 1
            self._load(j, slot, ctx)
        ev = self.copier.wait(block_idx)
        if ev is None:
            return
        if not self.debug:
            torch.cuda.current_stream().wait_event(ev)  # compute waits for H2D (GPU-side, does not block CPU)
            return
        # debug: measure the actual GPU stall (idle time the compute stream spends on wait_event)
        s = torch.cuda.current_stream()
        c = self._cur[ctx]
        c["waits"] += 1
        if not ev.query():
            c["not_ready"] += 1  # H2D not finished when compute reached it -> a real stall (cheap, no sync)
        ev_a = torch.cuda.Event(enable_timing=True)
        ev_b = torch.cuda.Event(enable_timing=True)
        ev_a.record(s)
        s.wait_event(ev)
        ev_b.record(s)
        c["stall_ev"].append((ev_a, ev_b))
        # staged copier records its transfer-timing events on the worker thread, so they only become
        # available once wait() has joined it; the direct copier already popped them in _load (returns None)
        xfer = self.copier.pop_xfer_timing(block_idx)
        if xfer is not None:
            c["xfer_ev"].append(xfer)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # consumed block_idx in the forward pass: free its slot and prefetch the block B streaming-slots ahead
        if self.S == 0 or not self.is_stream[block_idx]:
            return
        j = self.rank[block_idx]
        self.free_event[j % self.B] = torch.cuda.current_stream().record_event()
        if j + self.B < self.S:
            self._load(j + self.B, (j + self.B) % self.B, "fwd")  # same slot as rank j; evicts rank j
        elif self.forward_only and j == self.S - 1 and self.S > self.B:
            # inference: no backward pass will consume the retained tail, and the next forward pass starts
            # from rank 0 again. Wrap around: all tail slots are free once the last streaming block is
            # consumed, so preload the first B ranks now. The H2D overlaps with the remaining (resident)
            # blocks and the output layers, instead of stalling at the start of the next pass (self-heal).
            for k in range(self.B):
                self._load(k, k, "fwd")

    def _create_backward_hook(self, block_index: int):
        prefetch = self.is_stream[block_index]
        wait_prev = block_index - 1 >= 0 and self.is_stream[block_index - 1]
        if not prefetch and not wait_prev:
            return None

        def backward_hook(module, grad_input, grad_output):
            if not any(g is not None for g in grad_input):
                # No positional input of this block requires grad, so PyTorch fired this hook when the gradients of the
                # *outputs* became available, i.e. before the block's own backward (and its checkpoint recompute) ran.
                # This is the case for the first block that takes part in the backward pass (e.g. when the LoRA is
                # applied only to later blocks): no block runs after it, so there is nothing to free, prefetch or wait
                # for, and evicting it here would break its recompute. (The same happens when a forward loop passes all
                # tensors as keyword arguments, which it must not do; see the model forward loops.)
                return None
            if prefetch:
                # consumed block_index in backward: free its slot and prefetch B streaming-slots behind
                j = self.rank[block_index]
                self.free_event[j % self.B] = torch.cuda.current_stream().record_event()
                if j - self.B >= 0:
                    self._load(j - self.B, (j - self.B) % self.B, "bwd")  # same slot as rank j; evicts rank j
            if wait_prev:
                self._wait_ctx = "bwd"
                self.wait_for_block(block_index - 1)  # ensure the next block to run is ready
                self._wait_ctx = "fwd"
            return None

        return backward_hook

    # ------------------------------------------------------------------ debug timing

    def _dbg_reset(self):
        self._dbg_empty = lambda: {"stall_ev": [], "xfer_ev": [], "waits": 0, "not_ready": 0, "self_heal": 0, "loads": 0}
        self._cur = {"fwd": self._dbg_empty(), "bwd": self._dbg_empty()}
        self._roll = {"fwd": defaultdict(float), "bwd": defaultdict(float)}
        self._step = 0

    def _dbg_boundary(self):
        # finalize the events accumulated for the step that just ended, then print every debug_interval steps
        if not any(self._cur[c]["waits"] or self._cur[c]["loads"] for c in ("fwd", "bwd")):
            return  # very first forward: nothing finished yet
        torch.cuda.synchronize()  # ensure timing events are complete (debug only)
        for ctx in ("fwd", "bwd"):
            c, r = self._cur[ctx], self._roll[ctx]
            r["stall_ms"] += sum(a.elapsed_time(b) for a, b in c["stall_ev"])
            r["xfer_ms"] += sum(a.elapsed_time(b) for a, b in c["xfer_ev"])
            for k in ("waits", "not_ready", "self_heal", "loads"):
                r[k] += c[k]
            self._cur[ctx] = self._dbg_empty()
        self._step += 1
        if self._step % self.debug_interval == 0:
            self._dbg_print()

    def _dbg_print(self):
        n = self.debug_interval
        print(f"[{self.block_type}] H2D-only timing (avg over {n} steps):")
        for ctx in ("fwd", "bwd"):
            r = self._roll[ctx]
            print(
                f"  {ctx}: stall {r['stall_ms'] / n:6.2f} ms/step, H2D {r['xfer_ms'] / n:6.2f} ms/step"
                f" | waits {r['waits'] / n:.1f}, not-ready {r['not_ready'] / n:.1f},"
                f" self-heal {r['self_heal'] / n:.1f}, loads {r['loads'] / n:.1f}"
            )
            self._roll[ctx] = defaultdict(float)


# endregion

# region cpu offload utils


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.

    Args:
        x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.

    Returns:
        The same structure as x, with all torch.Tensor objects moved to CPU.
        Non-tensor objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


# Unsloth Offloaded Gradient Checkpointing
# Based on Unsloth Zoo by Daniel Han-Chen & the Unsloth team
try:
    from deepspeed.runtime.activation_checkpointing.checkpointing import detach_variable
except ImportError:

    def detach_variable(inputs, device=None):
        """Detach tensors from computation graph, optionally moving to a device.

        Reimplementation of deepspeed.runtime.activation_checkpointing.checkpointing.detach_variable
        for environments without DeepSpeed installed.
        """
        if isinstance(inputs, tuple):
            out = []
            for inp in inputs:
                if not isinstance(inp, torch.Tensor):
                    out.append(inp)
                    continue
                requires_grad = inp.requires_grad
                if device is not None:
                    x = inp.to(device=device)
                else:
                    x = inp
                x = x.detach()
                x.requires_grad = requires_grad
                out.append(x)
            return tuple(out)
        else:
            raise RuntimeError(
                "Only tuple of tensors is supported. Got Unsupported input type: ",
                type(inputs).__name__,
            )


class UnslothOffloadedGradientCheckpointer(torch.autograd.Function):
    """Saves VRAM by offloading activations to CPU RAM using non-blocking transfers.

    Compared to standard cpu_offload_checkpointing which uses blocking transfers,
    this uses non_blocking=True to hide CPU<->GPU transfer latency behind compute.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, forward_function, hidden_states, *args):
        # Remember the original device for backward pass (multi-GPU support)
        ctx.input_device = hidden_states.device
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        # NOTE: args stored directly on ctx (not via save_for_backward) because
        # the training loop holds references to these tensors, preventing GC.
        # Using save_for_backward for all args would add complexity for no benefit.
        ctx.args = args
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, *grads):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to(ctx.input_device, non_blocking=True).detach()
        hidden_states.requires_grad_(True)
        args = detach_variable(ctx.args)
        inputs = (hidden_states,) + args
        with torch.enable_grad():
            outputs = ctx.forward_function(*inputs)

        output_tensors = []
        grad_tensors = []
        for out, grad in zip(
            outputs if isinstance(outputs, tuple) else (outputs,), grads if isinstance(grads, tuple) else (grads,)
        ):
            if isinstance(out, torch.Tensor) and out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)
        return (None,) + tuple(inp.grad if isinstance(inp, torch.Tensor) else None for inp in inputs)


@torch._disable_dynamo
def unsloth_checkpoint(function, *args):
    """Wrapper for UnslothOffloadedGradientCheckpointer."""
    return UnslothOffloadedGradientCheckpointer.apply(function, *args)


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.

    Args:
        func: The original function to wrap.
        device: The device to move outputs back to.

    Returns:
        A wrapped function that offloads inputs to CPU and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)


# endregion
