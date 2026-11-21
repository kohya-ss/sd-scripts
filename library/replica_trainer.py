"""Single-process multi-GPU training for network (LoRA etc.) trainers: the replica trainer.

One Python process drives one *master* (the accelerator device, usually cuda:0) and one *replica* per additional GPU
(``--replica_devices``). Each replica holds its own copy of the frozen base model and of the network. The data loader
micro-steps are assigned to the devices in turn (replicas first, master last), the replica micro-steps run in a worker
thread per device, and at each optimizer step the replica gradients are summed into the master, the master is updated
as usual, and the replicas pull the new weights before their next micro-step.

Compared with DDP (``accelerate launch --multi_gpu``) there is no process group: no torchrun, no NCCL / gloo, one data
loader, and the text encoder / VAE are needed on the master only (their outputs are cached). ``train_batch_size`` is per
device as with DDP: ``gradient_accumulation_steps`` is multiplied by the number of devices internally so that the step
count, the learning rate schedule and the progress bar are the same as with DDP with the same arguments.

The trainer class is composed at runtime: ``wrap_trainer_class(MossImageNetworkTrainer, args)`` returns a subclass with
``ReplicaTrainerMixin`` in front. The mixin plugs into the existing hooks of ``NetworkTrainer`` (``assert_extra_args``,
``post_process_network``, ``forward_backward``, ``all_reduce_network``, ``update_metadata``); ``train()`` is unchanged.

Current limitations (v1): CUDA only, Anima and MOSS-Image, U-Net (DiT) LoRA only with cached latents and cached text
encoder outputs, ``mixed_precision`` bf16 or no, no DeepSpeed, no ``--base_weights``.

Implementation notes (see references/replica_training/README.md for the measurements behind them):
- Issuing the micro-steps of all devices from one thread serializes them: the CUDA launch queue fills and the thread
  blocks on the busiest device. Hence the worker thread per replica.
- The loss returned for a replica micro-step is the loss of the previous micro-step of that replica (the current one
  is still running). The logged loss therefore lags by one dispatch for the replica steps; the gradients do not.
- torch.compile is not thread-safe while compiling, so the first ``--replica_warmup_steps`` optimizer steps run the
  replica micro-steps serially.
- Cross-device copies through pinned host memory (``PinnedStager``): the direct ``tensor.to(other_device)`` path is
  slow on Windows (no P2P).
- The ``torch._dynamo`` / ``torch._inductor`` config overrides (``--compile_cache_size_limit`` etc.) are stored in
  ``contextvars``, which a new thread does not inherit: the worker runs each micro-step in a copy of the main thread's
  context (``Replica.submit``), otherwise the replica frames would use the default recompile limit.
"""

import argparse
import contextvars
import importlib
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# region arguments


def add_replica_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--replica_devices",
        type=str,
        default=None,
        help="single-process multi-GPU training: comma-separated CUDA devices for the additional replicas, e.g. `1` or"
        " `cuda:1,cuda:2`. The master runs on the accelerator device. Requires --cache_latents, --cache_text_encoder_outputs"
        " and --network_train_unet_only; train_batch_size is per device as with DDP"
        " / 1プロセスでのマルチGPU学習: 追加レプリカを置くCUDAデバイスをカンマ区切りで指定 (例: `1`, `cuda:1,cuda:2`)。"
        "マスターはacceleratorのデバイスで動作。--cache_latents, --cache_text_encoder_outputs, --network_train_unet_onlyが必要。"
        "train_batch_sizeはDDPと同様にデバイスごとの値",
    )
    parser.add_argument(
        "--replica_warmup_steps",
        type=int,
        default=None,
        help="number of optimizer steps at the start of the training during which the replica micro-steps run serially"
        " (torch.compile is not thread-safe while compiling). Default: 2 with --compile, 0 otherwise"
        " / 学習開始時にレプリカのmicro-stepを直列実行するoptimizer step数 (torch.compileはコンパイル中スレッドセーフでない)。"
        "既定: --compile時は2、それ以外は0",
    )


def parse_replica_devices(spec: str) -> List[torch.device]:
    """``"1,2"`` / ``"cuda:1, cuda:2"`` -> ``[cuda:1, cuda:2]``. Each entry must name a CUDA device with an index."""
    devices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token.isdigit():
            token = f"cuda:{token}"
        device = torch.device(token)
        if device.type != "cuda" or device.index is None:
            raise ValueError(f"--replica_devices entries must be CUDA devices with an index (e.g. `1` or `cuda:1`), got `{token}`")
        devices.append(device)
    if not devices:
        raise ValueError("--replica_devices is empty")
    return devices


def build_schedule(num_devices: int) -> List[int]:
    """Device index (0 = master) of each micro-step within a round: replicas first, master last.

    The master micro-step runs synchronously in the main thread, so dispatching the replicas first lets them run while
    the master is busy. Extending this list (e.g. ``[1, 1, 0]``) gives a weighted assignment.
    """
    return list(range(1, num_devices)) + [0]


# endregion


# region accelerator proxy


class AcceleratorDeviceProxy:
    """An ``Accelerator`` look-alike for one replica: ``device`` is the replica's device, ``prepare`` / ``unwrap_model``
    are the identity (the replica trainer places the models itself and accelerate must not register them), everything
    else is delegated to the real accelerator (``autocast``, ``backward``, ``print``, ...)."""

    def __init__(self, accelerator, device: torch.device):
        self._accelerator = accelerator
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def prepare(self, *args, device_placement=None, **kwargs):
        # like accelerate: move the modules to the device unless device_placement says otherwise (block swap passes
        # [False] and places the model itself); nothing is wrapped or registered
        for i, obj in enumerate(args):
            place = device_placement[i] if device_placement is not None else True
            if place and isinstance(obj, torch.nn.Module):
                obj.to(self._device)
        return args[0] if len(args) == 1 else args

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        return model

    def __getattr__(self, name):
        return getattr(self._accelerator, name)


# endregion


# region cross-device copies


class PinnedStager:
    """Copies a list of tensors from one CUDA device to another through a pinned host buffer.

    The copies are stream-ordered on both sides (D2H on the source's current stream, H2D on the destination's current
    stream after an event) and do not block the CPU, except when a staging buffer is reused before its previous H2D
    finished. One buffer is kept per ``key`` and dtype.
    """

    def __init__(self):
        self._buffers: Dict[Tuple[Any, torch.dtype], torch.Tensor] = {}
        self._in_flight: Dict[Tuple[Any, torch.dtype], torch.cuda.Event] = {}

    def _buffer(self, key, numel: int, dtype: torch.dtype) -> torch.Tensor:
        k = (key, dtype)
        buf = self._buffers.get(k)
        if buf is None or buf.numel() < numel:
            buf = torch.empty(numel, dtype=dtype, pin_memory=True)
            self._buffers[k] = buf
        return buf

    @torch.no_grad()
    def copy_flat(self, srcs: Sequence[torch.Tensor], dsts: Sequence[torch.Tensor], key, accumulate: bool = False):
        """``dsts[i] (+)= srcs[i]`` for all i. All ``srcs`` are on one device, all ``dsts`` on one device (may be the same)."""
        assert len(srcs) == len(dsts)
        if not srcs:
            return
        by_dtype: Dict[torch.dtype, List[int]] = {}
        for i, t in enumerate(srcs):
            by_dtype.setdefault(t.dtype, []).append(i)

        src_device = srcs[0].device
        dst_device = dsts[0].device
        for dtype, indices in by_dtype.items():
            flat = torch.cat([srcs[i].reshape(-1) for i in indices])
            if src_device == dst_device:
                out = flat
            else:
                k = (key, dtype)
                pending = self._in_flight.get(k)
                if pending is not None:
                    pending.synchronize()  # the previous H2D read of this buffer must be done before overwriting it
                buf = self._buffer(key, flat.numel(), dtype)[: flat.numel()]
                buf.copy_(flat, non_blocking=True)  # D2H on the source's current stream
                done = torch.cuda.Event()
                done.record(torch.cuda.current_stream(src_device))
                dst_stream = torch.cuda.current_stream(dst_device)
                dst_stream.wait_event(done)
                with torch.cuda.device(dst_device), torch.cuda.stream(dst_stream):
                    out = buf.to(dst_device, non_blocking=True)
                read = torch.cuda.Event()
                read.record(dst_stream)
                self._in_flight[k] = read
            # scatter with one foreach call instead of one kernel launch per tensor (hundreds of LoRA tensors)
            views, targets = [], []
            offset = 0
            for i in indices:
                d = dsts[i]
                views.append(out[offset : offset + d.numel()].view_as(d))
                targets.append(d)
                offset += d.numel()
            if accumulate:
                torch._foreach_add_(targets, views)
            else:
                torch._foreach_copy_(targets, views)


# endregion


def move_batch_to_device(batch, device: torch.device):
    """Return a copy of the data loader batch with every tensor (also inside lists / tuples / dicts) on ``device``.

    The prepared data loader places the batches on the accelerator device (the master); the replica micro-steps need
    them on their own device (``loss_weights``, ``alpha_masks``, ... are used as they are by ``process_batch``).
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(move_batch_to_device(v, device) for v in batch)
    return batch


def batch_shape_signature(batch) -> Tuple[int, ...]:
    """The input shapes of a micro-step that can trigger a torch.compile recompilation, as a flat tuple.

    Latents (batch size, height, width), the cached text encoder outputs (text length) and, for batches of more than one
    sample, the valid lengths of the text masks (the per-sample sequence lengths are passed to the compiled blocks as
    Python ints). The positions in the tuple identify the dimensions for ``ShapeTracker``.
    """
    sig: List[int] = []
    latents = batch.get("latents")
    if latents is not None:
        sig.extend(latents.shape)

    def visit(obj):
        if isinstance(obj, torch.Tensor):
            sig.extend(obj.shape)
            if obj.ndim == 2 and obj.shape[0] > 1 and not obj.is_floating_point():
                sig.extend(int(n) for n in obj.sum(dim=1).tolist())  # mask row sums (device sync, batch size > 1 only)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                visit(v)

    visit(batch.get("text_encoder_outputs_list"))
    return tuple(sig)


class ShapeTracker:
    """Decides whether a micro-step may trigger a torch.compile (re)compilation on a device, from its input shapes.

    Mirrors Dynamo's automatic dynamic shapes: a dimension is specialized to its first value, recompiled as dynamic when
    a second value is seen, and free afterwards. So a micro-step is "new" for a device when any dimension has a value
    not seen before on that device while fewer than two distinct values are known for it. With ``compile_dynamic``
    ``"true"`` only the very first shape is new, with ``"false"`` every new value is.
    """

    def __init__(self, compile_dynamic: Optional[str]):
        self.mode = (compile_dynamic or "auto").lower()
        self._seen: Dict[int, List[set]] = {}

    def is_new(self, device_index: int, signature: Tuple[int, ...]) -> bool:
        seen = self._seen.get(device_index)
        if seen is None or len(seen) != len(signature):
            self._seen[device_index] = [{v} for v in signature]
            return True  # first micro-step on this device (or a different batch structure)
        if self.mode == "true":
            return False
        new = False
        for values, v in zip(seen, signature):
            if v not in values:
                if self.mode == "false" or len(values) < 2:
                    new = True
                values.add(v)
        return new


class _Timing:
    """Mean durations of the micro-step phases, for SD_REPLICA_TIMING=1 (thread-safe enough for logging)."""

    def __init__(self):
        self._values: Dict[str, List[float]] = defaultdict(list)
        self.last_master_start: Optional[float] = None

    def add(self, **durations: float):
        for k, v in durations.items():
            self._values[k].append(v)

    def summary(self) -> str:
        parts = []
        for k in sorted(self._values):
            v = self._values[k]
            if v:
                parts.append(f"{k}={sum(v) / len(v):.3f}")
        self._values.clear()
        return " ".join(parts)


# region replica


@dataclass
class Replica:
    index: int  # 1-based (0 is the master)
    device: torch.device
    proxy: AcceleratorDeviceProxy
    unet: torch.nn.Module
    network: torch.nn.Module
    text_encoders: list
    executor: ThreadPoolExecutor
    params: List[torch.nn.Parameter] = field(default_factory=list)
    last_future: Optional[Future] = None
    weights_dirty: bool = True  # pull the master weights before the next micro-step

    def submit(self, fn) -> Future:
        # run in a copy of the caller's context: the torch._dynamo / torch._inductor config overrides made in the main
        # thread (e.g. --compile_cache_size_limit) live in contextvars and are not inherited by the worker thread
        self.last_future = self.executor.submit(contextvars.copy_context().run, fn)
        return self.last_future

    def join(self):
        """Wait for the in-flight micro-step (re-raises its exception in the caller)."""
        if self.last_future is not None:
            self.last_future.result()


class ReplicaTrainerMixin:
    """Mixin in front of a ``NetworkTrainer`` subclass; see the module docstring. Compose with ``wrap_trainer_class``."""

    # region hooks

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if not torch.cuda.is_available():
            raise ValueError("--replica_devices requires CUDA")
        self.replica_devices = parse_replica_devices(args.replica_devices)
        num_devices = 1 + len(self.replica_devices)

        def require(cond: bool, what: str):
            if not cond:
                raise ValueError(f"--replica_devices requires {what} / --replica_devicesには{what}が必要です")

        require(args.cache_latents, "--cache_latents")
        require(args.cache_text_encoder_outputs, "--cache_text_encoder_outputs")
        require(args.network_train_unet_only, "--network_train_unet_only")
        require(args.mixed_precision != "fp16", "--mixed_precision bf16 or no (fp16 is not supported)")
        require(not args.deepspeed, "no DeepSpeed")
        require(not args.torch_compile, "--compile instead of --torch_compile")
        require(args.base_weights is None, "no --base_weights")
        require(not (args.fp8_base or args.fp8_base_unet), "no --fp8_base")

        self._replica_orig_gradient_accumulation_steps = args.gradient_accumulation_steps
        args.gradient_accumulation_steps *= num_devices
        self._replica_warmup_steps = args.replica_warmup_steps
        if self._replica_warmup_steps is None:
            self._replica_warmup_steps = 2 if args.compile else 0
        # torch.compile is not thread-safe while compiling (FX tracing patches nn.Module.__call__ process-wide), and the
        # compile lock makes a coordinated wait impossible. A micro-step that may compile must therefore run alone: all
        # micro-steps during the warmup, and later the micro-steps whose input shapes are new to their device.
        self._replica_shapes = ShapeTracker(args.compile_dynamic) if args.compile else None
        # SD_REPLICA_TIMING=1 logs the mean durations of the micro-step phases every 10 optimizer steps (development aid)
        self._replica_timing = _Timing() if os.environ.get("SD_REPLICA_TIMING") else None
        logger.info(
            f"replica training: master + {len(self.replica_devices)} replica(s) on {[str(d) for d in self.replica_devices]},"
            f" gradient_accumulation_steps {self._replica_orig_gradient_accumulation_steps} x {num_devices} ="
            f" {args.gradient_accumulation_steps} micro-steps per optimizer step, {self._replica_warmup_steps} serial warmup step(s)"
        )

        self._replicas: List[Replica] = []
        self._replica_schedule = build_schedule(num_devices)
        self._replica_micro_step = 0
        self._replica_optimizer_steps = 0
        self._replica_master_params: Optional[List[torch.nn.Parameter]] = None
        self._replica_stager = PinnedStager()

    def load_target_model(self, args, weight_dtype, accelerator):
        result = super().load_target_model(args, weight_dtype, accelerator)
        # train() passes the text encoder(s) to create_network / apply_to exactly as returned here (a list for Anima and
        # MOSS-Image, a single module for others); keep the same object for the replicas
        self._replica_text_encoder_arg = result[1]
        return result

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)

        if accelerator.num_processes != 1:
            raise ValueError("--replica_devices cannot be combined with multi-process (DDP) training: launch a single process")
        if accelerator.device.type != "cuda":
            raise ValueError(f"--replica_devices requires the master on a CUDA device, got {accelerator.device}")
        for device in self.replica_devices:
            if device == accelerator.device:
                logger.warning(f"replica device {device} is the master device; this only makes sense for testing")

        from library import accelerator_setup

        weight_dtype, _ = accelerator_setup.prepare_dtype(args)
        network_module = importlib.import_module(args.network_module)
        for i, device in enumerate(self.replica_devices):
            self._replicas.append(self._create_replica(args, accelerator, i + 1, device, network_module, text_encoders, weight_dtype))

    def forward_backward(
        self,
        args,
        accelerator,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        text_encoding_strategy,
        tokenize_strategy,
        train_text_encoder,
        train_unet,
    ) -> torch.Tensor:
        target = self._replica_schedule[self._replica_micro_step % len(self._replica_schedule)]
        self._replica_micro_step += 1

        if target == 0:
            return self._run_master(
                args,
                batch,
                super().forward_backward,
                args,
                accelerator,
                batch,
                text_encoders,
                unet,
                network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                text_encoding_strategy,
                tokenize_strategy,
                train_text_encoder,
                train_unet,
            )

        replica = self._replicas[target - 1]
        self._ensure_param_lists(accelerator, network)
        timing = self._replica_timing

        def run_micro_step() -> float:
            t0 = time.perf_counter()
            if replica.weights_dirty:
                self._replica_stager.copy_flat(self._replica_master_params, replica.params, key=("weights", replica.index))
                replica.weights_dirty = False
            t1 = time.perf_counter()
            moved = move_batch_to_device(batch, replica.device)
            t2 = time.perf_counter()
            loss = self.process_batch(
                moved,
                replica.text_encoders,
                replica.unet,
                replica.network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                replica.proxy,
                args,
                text_encoding_strategy,
                tokenize_strategy,
                is_train=True,
                train_text_encoder=False,
                train_unet=train_unet,
            )
            accelerator.backward(loss)
            t3 = time.perf_counter()
            value = loss.detach().item()  # waits for this replica's GPU only (worker thread)
            if timing is not None:
                t4 = time.perf_counter()
                timing.add(replica_pull=t1 - t0, replica_move=t2 - t1, replica_cpu=t3 - t2, replica_item=t4 - t3, replica_total=t4 - t0)
            return value

        previous = replica.last_future
        t_dispatch = time.perf_counter()
        future = replica.submit(run_micro_step)
        if previous is None or self._must_run_alone(replica.index, batch):
            value = self._wait(future)  # first micro-step of this replica, warmup or new shapes: wait for the current one
        else:
            value = self._wait(previous)  # the loss of the previous micro-step of this replica (lags by one dispatch)
        if timing is not None:
            timing.add(dispatch_wait=time.perf_counter() - t_dispatch)
        return torch.tensor(value)

    def _run_master(self, args, batch, forward_backward_super, *fb_args) -> torch.Tensor:
        if self._must_run_alone(0, batch):
            for replica in self._replicas:
                replica.join()  # a compilation on the master must not overlap with a running replica micro-step
        t0 = time.perf_counter()
        try:
            loss = forward_backward_super(*fb_args)
        except RuntimeError as e:
            raise self._annotate_compile_race(e) from e
        if self._replica_timing is not None:
            t1 = time.perf_counter()
            self._replica_timing.add(master_cpu=t1 - t0)
            if self._replica_timing.last_master_start is not None:
                self._replica_timing.add(round_wall=t0 - self._replica_timing.last_master_start)
            self._replica_timing.last_master_start = t0
        return loss

    def _must_run_alone(self, device_index: int, batch) -> bool:
        """True when this micro-step may compile: during the warmup, or when its input shapes are new to the device."""
        if self._replica_optimizer_steps < self._replica_warmup_steps:
            return True
        if self._replica_shapes is None:
            return False
        return self._replica_shapes.is_new(device_index, batch_shape_signature(batch))

    @staticmethod
    def _wait(future: Future):
        try:
            return future.result()
        except RuntimeError as e:
            raise ReplicaTrainerMixin._annotate_compile_race(e) from e

    @staticmethod
    def _annotate_compile_race(e: RuntimeError) -> RuntimeError:
        if "symbolically trace a dynamo-optimized function" not in str(e):
            return e
        return RuntimeError(
            "torch.compile compiled a block while another device was running (not thread-safe). Increase"
            " --replica_warmup_steps, or train without --compile when using --replica_devices, and please report the case"
            " / torch.compile のコンパイルが他デバイスの実行と重なりました（スレッドセーフではありません）。"
            "--replica_warmup_steps を増やすか、--replica_devices 使用時は --compile を外してください。報告もお願いします"
        )

    def all_reduce_network(self, accelerator, network):
        # single process: nothing to do for the master itself; sum the replica gradients into the master
        self._ensure_param_lists(accelerator, network)
        master_params = self._replica_master_params
        t0 = time.perf_counter()
        for replica in self._replicas:
            replica.join()
            t1 = time.perf_counter()
            grads = [p.grad for p in replica.params]
            if all(g is None for g in grads):
                continue  # no micro-step on this replica since the last step (partial round at the end of the data loader)
            if any(g is None for g in grads):
                raise RuntimeError(f"replica {replica.index}: some parameters have no gradient")
            for p in master_params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            self._replica_stager.copy_flat(grads, [p.grad for p in master_params], key=("grads", replica.index), accumulate=True)
            for p in replica.params:
                p.grad = None
            replica.weights_dirty = True
            if self._replica_timing is not None:
                self._replica_timing.add(sync_join=t1 - t0, sync_reduce=time.perf_counter() - t1)
        self._replica_optimizer_steps += 1
        if self._replica_timing is not None and self._replica_optimizer_steps % 10 == 0:
            logger.info(f"replica timing (mean s over the last 10 steps): {self._replica_timing.summary()}")

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_gradient_accumulation_steps"] = self._replica_orig_gradient_accumulation_steps
        metadata["ss_replica_devices"] = ",".join(str(d) for d in self.replica_devices)

    # endregion

    def _ensure_param_lists(self, accelerator, network):
        if self._replica_master_params is not None:
            return
        master = accelerator.unwrap_model(network)
        self._replica_master_params = [p for p in master.parameters() if p.requires_grad]
        for replica in self._replicas:
            replica.params = [p for p in replica.network.parameters() if p.requires_grad]
            if len(replica.params) != len(self._replica_master_params) or any(
                a.shape != b.shape or a.dtype != b.dtype for a, b in zip(replica.params, self._replica_master_params)
            ):
                raise RuntimeError(
                    f"replica {replica.index}: trainable parameters do not match the master"
                    f" ({len(replica.params)} vs {len(self._replica_master_params)})"
                )
        logger.info(f"replica training: {len(self._replica_master_params)} trainable tensors synchronized per step")

    def _create_replica(self, args, accelerator, index: int, device: torch.device, network_module, text_encoders, weight_dtype) -> Replica:
        """Build the frozen base model and the network of one replica. Mirrors the setup of the master in train()."""
        logger.info(f"creating replica {index} on {device}")
        proxy = AcceleratorDeviceProxy(accelerator, device)
        text_encoder = getattr(self, "_replica_text_encoder_arg", text_encoders)  # as passed by train() (see load_target_model)
        with torch.cuda.device(device):
            unet, _ = self.load_unet_lazily(args, weight_dtype, proxy, text_encoders)

            # vae is only used by networks that train the VAE (none of the supported ones)
            network = self.create_network(args, network_module, None, text_encoder, unet, self.parse_network_args(args))
            if network is None:
                raise RuntimeError("create_network returned None for the replica")
            if hasattr(network, "prepare_network"):
                network.prepare_network(args)
            network.apply_to(text_encoder, unet, False, True)  # U-Net (DiT) only

            # --- mirror of the master setup in NetworkTrainer.train() ---
            if args.gradient_checkpointing:
                unet.enable_gradient_checkpointing(cpu_offload=args.cpu_offload_checkpointing)
                network.enable_gradient_checkpointing()  # may have no effect
            if args.full_fp16 or args.full_bf16:
                network.to(weight_dtype)
            unet.requires_grad_(False)
            if self.cast_unet(args):
                unet.to(dtype=weight_dtype)
            unet = self.prepare_unet_with_accelerator(args, proxy, unet)  # block swap placement, torch.compile
            network.to(device)
            if args.gradient_checkpointing:
                unet.train()
            else:
                unet.eval()
            network.prepare_grad_etc(text_encoder, unet)
            network.train()  # what network.on_epoch_start() does for the master
            # --- end of mirror ---

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"replica{index}", initializer=torch.cuda.set_device, initargs=(device,))
        return Replica(index=index, device=device, proxy=proxy, unet=unet, network=network, text_encoders=text_encoders, executor=executor)


def wrap_trainer_class(trainer_class: type, args: argparse.Namespace) -> type:
    """Return ``trainer_class`` extended with ``ReplicaTrainerMixin`` when ``--replica_devices`` is given, else as is."""
    if getattr(args, "replica_devices", None) is None:
        return trainer_class
    return type(f"Replica{trainer_class.__name__}", (ReplicaTrainerMixin, trainer_class), {})


# endregion
