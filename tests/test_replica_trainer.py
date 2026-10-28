"""Tests for library/replica_trainer.py (single-process multi-GPU network training).

The end-to-end tests build a toy trainer on top of NetworkTrainer and run the mixin hooks directly (no train() loop);
they place the replica on the master device so that one GPU is enough. Cross-device copies are tested when two GPUs
are available.
"""

import argparse
import contextlib
import types

import pytest
import torch
import torch.nn as nn

from library import replica_trainer
from library.replica_trainer import AcceleratorDeviceProxy, PinnedStager, ReplicaTrainerMixin, build_schedule, parse_replica_devices

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# region pure python


def test_parse_replica_devices():
    assert parse_replica_devices("1") == [torch.device("cuda:1")]
    assert parse_replica_devices("cuda:1, 2") == [torch.device("cuda:1"), torch.device("cuda:2")]
    with pytest.raises(ValueError):
        parse_replica_devices("cpu")
    with pytest.raises(ValueError):
        parse_replica_devices("cuda")  # no index
    with pytest.raises(ValueError):
        parse_replica_devices(" , ")


def test_build_schedule_replicas_first_master_last():
    assert build_schedule(1) == [0]
    assert build_schedule(2) == [1, 0]
    assert build_schedule(3) == [1, 2, 0]


def test_accelerator_proxy_delegates_everything_but_device_and_prepare():
    real = types.SimpleNamespace(device=torch.device("cuda:9"), num_processes=1, print=lambda *a: "printed", unwrap_model=lambda m: "unwrapped")
    proxy = AcceleratorDeviceProxy(real, torch.device("cpu"))
    assert proxy.device == torch.device("cpu")
    assert proxy.num_processes == 1
    assert proxy.print() == "printed"
    module = nn.Linear(2, 2)
    assert proxy.prepare(module) is module  # moved to the proxy device (a no-op here), not wrapped
    assert proxy.prepare(module, device_placement=[False]) is module
    assert proxy.prepare(module, module) == (module, module)
    assert proxy.unwrap_model(module) is module  # not the real accelerator's
    if torch.cuda.is_available():
        gpu_proxy = AcceleratorDeviceProxy(real, torch.device("cuda:0"))
        on_cpu = nn.Linear(2, 2)
        on_cpu.register_buffer("table", torch.zeros(3))
        gpu_proxy.prepare(on_cpu, device_placement=[False])
        assert on_cpu.table.device.type == "cpu"  # block swap style: the trainer places the model itself
        gpu_proxy.prepare(on_cpu)
        assert on_cpu.weight.device.type == "cuda" and on_cpu.table.device.type == "cuda"  # buffers too


def test_wrap_trainer_class():
    class Base:
        pass

    assert replica_trainer.wrap_trainer_class(Base, argparse.Namespace(replica_devices=None)) is Base
    wrapped = replica_trainer.wrap_trainer_class(Base, argparse.Namespace(replica_devices="1"))
    assert issubclass(wrapped, Base) and issubclass(wrapped, ReplicaTrainerMixin)
    assert wrapped.__mro__.index(ReplicaTrainerMixin) < wrapped.__mro__.index(Base)


# endregion


def test_move_batch_to_device_recurses_and_keeps_non_tensors():
    batch = {"latents": torch.zeros(2), "list": [torch.ones(1), (torch.ones(1), "caption")], "captions": ["a", "b"], "n": 3}
    moved = replica_trainer.move_batch_to_device(batch, torch.device("cpu"))
    assert moved is not batch and moved["captions"] == ["a", "b"] and moved["n"] == 3
    assert isinstance(moved["list"][1], tuple) and moved["list"][1][1] == "caption"
    if torch.cuda.is_available():
        moved = replica_trainer.move_batch_to_device(batch, torch.device("cuda:0"))
        assert moved["latents"].device.type == "cuda" and moved["list"][1][0].device.type == "cuda"
        assert batch["latents"].device.type == "cpu"  # the original is untouched


def test_shape_tracker_follows_automatic_dynamic_shapes():
    from library.replica_trainer import ShapeTracker

    t = ShapeTracker(None)  # auto
    assert t.is_new(0, (1, 64, 64, 300))  # first shape on device 0
    assert not t.is_new(0, (1, 64, 64, 300))
    assert t.is_new(1, (1, 64, 64, 300))  # first shape on device 1
    assert t.is_new(0, (1, 64, 64, 420))  # second text length: recompiled as dynamic
    assert not t.is_new(0, (1, 64, 64, 512))  # third: dynamic already
    assert t.is_new(0, (1, 48, 80, 512))  # new bucket (two dims with their second value)
    assert not t.is_new(0, (1, 80, 48, 200))
    assert t.is_new(0, (2, 80, 48, 200))  # batch size change

    static = ShapeTracker("false")
    assert static.is_new(0, (1, 64, 64, 300)) and static.is_new(0, (1, 64, 64, 420)) and static.is_new(0, (1, 64, 64, 512))
    assert not static.is_new(0, (1, 64, 64, 420))

    dynamic = ShapeTracker("true")
    assert dynamic.is_new(0, (1, 64, 64, 300)) and not dynamic.is_new(0, (2, 32, 32, 7))


def test_batch_shape_signature():
    batch = {
        "latents": torch.zeros(2, 32, 64, 64),
        "text_encoder_outputs_list": [torch.zeros(2, 300, 4096), torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.int32), [0.1, 0.0]],
    }
    sig = replica_trainer.batch_shape_signature(batch)
    assert sig == (2, 32, 64, 64, 2, 300, 4096, 2, 3, 3, 2)  # mask row sums appended for batch size > 1
    batch["latents"] = torch.zeros(1, 32, 64, 64)
    batch["text_encoder_outputs_list"][1] = torch.ones(1, 3, dtype=torch.int32)
    assert replica_trainer.batch_shape_signature(batch) == (1, 32, 64, 64, 2, 300, 4096, 1, 3)  # no sums for batch size 1


# region pinned stager


@cuda_only
def test_pinned_stager_same_device_copy_and_accumulate():
    dev = torch.device("cuda:0")
    stager = PinnedStager()
    srcs = [torch.randn(3, 4, device=dev), torch.randn(5, device=dev, dtype=torch.bfloat16)]
    dsts = [torch.zeros(3, 4, device=dev), torch.zeros(5, device=dev, dtype=torch.bfloat16)]
    stager.copy_flat(srcs, dsts, key="k")
    for s, d in zip(srcs, dsts):
        assert torch.equal(s, d)
    stager.copy_flat(srcs, dsts, key="k", accumulate=True)
    for s, d in zip(srcs, dsts):
        assert torch.equal(s * 2, d)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two GPUs required")
def test_pinned_stager_cross_device_copy():
    a, b = torch.device("cuda:0"), torch.device("cuda:1")
    stager = PinnedStager()
    srcs = [torch.randn(1000, device=a), torch.randn(7, 3, device=a)]
    dsts = [torch.zeros(1000, device=b), torch.ones(7, 3, device=b)]
    stager.copy_flat(srcs, dsts, key="g", accumulate=True)
    torch.cuda.synchronize(b)
    assert torch.equal(dsts[0].cpu(), srcs[0].cpu())
    assert torch.equal(dsts[1].cpu(), srcs[1].cpu() + 1)
    # reuse the staging buffer (waits for the previous H2D) and copy back the other way
    stager.copy_flat(dsts, srcs, key="g")
    torch.cuda.synchronize(a)
    assert torch.equal(srcs[1].cpu(), dsts[1].cpu())


# endregion


# region end-to-end with a toy trainer


class ToyNetwork(nn.Module):
    """Minimal network: a per-feature scale applied to the base model output in ToyTrainer.process_batch."""

    def __init__(self, features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def apply_to(self, text_encoders, unet, apply_text_encoder=True, apply_unet=True):
        pass

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)


class FakeAccelerator:
    def __init__(self, device, gradient_accumulation_steps):
        self.device = device
        self.num_processes = 1
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def backward(self, loss):
        (loss / self.gradient_accumulation_steps).backward()  # what accelerate does

    def unwrap_model(self, model):
        return model

    def autocast(self):
        return contextlib.nullcontext()

    def print(self, *args, **kwargs):
        pass

    def prepare(self, *args, **kwargs):
        return args[0] if len(args) == 1 else args


def make_toy_trainer_class():
    import train_network

    class ToyTrainer(train_network.NetworkTrainer):
        features = 4

        def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders):
            torch.manual_seed(0)
            unet = nn.Linear(self.features, self.features).to(accelerator.device)
            return unet, text_encoders

        def create_network(self, args, network_module, vae, text_encoder, unet, net_kwargs):
            return ToyNetwork(self.features)

        def cast_unet(self, args):
            return False

        def process_batch(self, batch, text_encoders, unet, network, vae, noise_scheduler, vae_dtype, weight_dtype, accelerator, args, text_encoding_strategy, tokenize_strategy, is_train=True, train_text_encoder=True, train_unet=True):
            x = batch["x"].to(accelerator.device)
            y = batch["y"].to(accelerator.device)
            with accelerator.autocast():
                pred = unet(x) * network.scale + network.bias
            return ((pred - y) ** 2).mean()

    return replica_trainer.wrap_trainer_class(ToyTrainer, argparse.Namespace(replica_devices="0"))


def toy_args(replica_devices="0", gradient_accumulation_steps=1, compile=False, replica_warmup_steps=None):
    return argparse.Namespace(
        replica_devices=replica_devices,
        replica_warmup_steps=replica_warmup_steps,
        cache_latents=True,
        cache_text_encoder_outputs=True,
        network_train_unet_only=True,
        mixed_precision="no",
        save_precision=None,
        deepspeed=False,
        torch_compile=False,
        compile=compile,
        base_weights=None,
        fp8_base=False,
        fp8_base_unet=False,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=False,
        cpu_offload_checkpointing=False,
        full_fp16=False,
        full_bf16=False,
        network_module="networks.lora_generic",  # imported but not used by ToyTrainer.create_network
        network_args=None,
        compile_dynamic=None,
    )


class _DatasetGroupStub:
    def verify_bucket_reso_steps(self, steps):
        pass


def setup_toy(args):
    """Returns (trainer, accelerator, master_unet, master_network, text_encoders)."""
    trainer_cls = make_toy_trainer_class()
    trainer = trainer_cls()
    trainer.assert_extra_args(args, _DatasetGroupStub(), None)
    accelerator = FakeAccelerator(torch.device("cuda:0"), args.gradient_accumulation_steps)
    text_encoders = [nn.Identity()]
    unet, _ = trainer.load_unet_lazily(args, torch.float32, accelerator, text_encoders)
    network = trainer.create_network(args, None, None, text_encoders[0], unet, {})
    trainer.post_process_network(args, accelerator, network, text_encoders, unet)
    # master setup as in train(): apply, requires_grad, device
    network.apply_to(text_encoders[0], unet, False, True)
    unet.requires_grad_(False)
    network.to(accelerator.device)
    network.prepare_grad_etc(text_encoders[0], unet)
    with torch.no_grad():  # make the master differ from the freshly created replica
        network.scale.mul_(1.5)
        network.bias.fill_(0.25)
    return trainer, accelerator, unet, network, text_encoders


def fb(trainer, args, accelerator, batch, unet, network, text_encoders):
    return trainer.forward_backward(args, accelerator, batch, text_encoders, unet, network, None, None, None, torch.float32, None, None, False, True)


def make_batch(seed, features=4, bs=2):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(bs, features, generator=g), "y": torch.randn(bs, features, generator=g)}


def reference_grads(unet, network, batches, ga):
    """Gradient of sum(loss_i / ga) over the batches on a copy of the master."""
    import copy

    net_ref = copy.deepcopy(network)
    for p in net_ref.parameters():
        p.grad = None
    for b in batches:
        x, y = b["x"].to(unet.weight.device), b["y"].to(unet.weight.device)
        pred = unet(x) * net_ref.scale + net_ref.bias
        (((pred - y) ** 2).mean() / ga).backward()
    return [p.grad.clone() for p in net_ref.parameters()]


@cuda_only
def test_assert_extra_args_multiplies_gradient_accumulation_and_validates():
    trainer_cls = make_toy_trainer_class()
    args = toy_args(replica_devices="0,0", gradient_accumulation_steps=3)
    trainer_cls().assert_extra_args(args, _DatasetGroupStub(), None)
    assert args.gradient_accumulation_steps == 9  # 3 devices
    assert args.replica_warmup_steps is None

    with_compile = toy_args(compile=True)
    trainer_cls().assert_extra_args(with_compile, _DatasetGroupStub(), None)
    assert with_compile.replica_warmup_steps is None  # resolved internally to 2 with --compile

    bad = toy_args()
    bad.cache_latents = False
    with pytest.raises(ValueError, match="cache_latents"):
        trainer_cls().assert_extra_args(bad, _DatasetGroupStub(), None)
    bad = toy_args()
    bad.mixed_precision = "fp16"
    with pytest.raises(ValueError, match="fp16"):
        trainer_cls().assert_extra_args(bad, _DatasetGroupStub(), None)


@cuda_only
def test_one_step_matches_reference_and_weights_are_pulled():
    args = toy_args(replica_devices="0", gradient_accumulation_steps=1)  # GA' = 2 micro-steps: replica then master
    trainer, accelerator, unet, network, tes = setup_toy(args)
    assert len(trainer._replicas) == 1
    replica = trainer._replicas[0]

    b1, b2 = make_batch(1), make_batch(2)
    loss1 = fb(trainer, args, accelerator, b1, unet, network, tes)  # replica (first dispatch: waits, exact loss)
    loss2 = fb(trainer, args, accelerator, b2, unet, network, tes)  # master
    trainer.all_reduce_network(accelerator, network)

    # the replica pulled the master weights before its micro-step, so both micro-steps used the master weights
    for pm, pr in zip(trainer._replica_master_params, replica.params):
        assert torch.equal(pm.detach(), pr.detach())
        assert pr.grad is None  # cleared after the reduction

    ref = reference_grads(unet, network, [b1, b2], ga=2)
    for pm, g in zip(trainer._replica_master_params, ref):
        assert torch.allclose(pm.grad, g, atol=1e-6), (pm.grad, g)

    # the logged losses are the real ones for the first dispatch and the master
    x, y = b1["x"].cuda(), b1["y"].cuda()
    expected1 = (((unet(x) * network.scale + network.bias) - y) ** 2).mean().item()
    assert loss1.item() == pytest.approx(expected1, rel=1e-5)
    assert loss2.requires_grad  # master loss is the live tensor

    # optimizer step on the master, then the next replica micro-step pulls the new weights
    with torch.no_grad():
        for p in trainer._replica_master_params:
            p.add_(-0.1 * p.grad)
            p.grad = None
    assert replica.weights_dirty
    loss3 = fb(trainer, args, accelerator, make_batch(3), unet, network, tes)  # replica, returns the previous (lagged) loss
    assert loss3.item() == pytest.approx(expected1, rel=1e-5)  # == loss of micro-step 1 (reported again once)
    replica.join()
    for pm, pr in zip(trainer._replica_master_params, replica.params):
        assert torch.equal(pm.detach(), pr.detach())
    assert not replica.weights_dirty


@cuda_only
def test_partial_round_at_end_of_dataloader():
    args = toy_args(replica_devices="0", gradient_accumulation_steps=2)  # GA' = 4
    trainer, accelerator, unet, network, tes = setup_toy(args)
    # only one replica micro-step before a forced sync (end of the data loader): the master has no gradient yet
    b = make_batch(5)
    fb(trainer, args, accelerator, b, unet, network, tes)
    trainer.all_reduce_network(accelerator, network)
    ref = reference_grads(unet, network, [b], ga=4)
    for pm, g in zip(trainer._replica_master_params, ref):
        assert torch.allclose(pm.grad, g, atol=1e-6)
    # a sync with nothing dispatched to the replica leaves the master gradient untouched
    before = [p.grad.clone() for p in trainer._replica_master_params]
    trainer.all_reduce_network(accelerator, network)
    for pm, g in zip(trainer._replica_master_params, before):
        assert torch.equal(pm.grad, g)


@cuda_only
def test_serial_warmup_returns_current_loss():
    args = toy_args(replica_devices="0", gradient_accumulation_steps=1, replica_warmup_steps=1)
    trainer, accelerator, unet, network, tes = setup_toy(args)
    assert trainer._replica_warmup_steps == 1
    b1, b2, b3 = make_batch(11), make_batch(12), make_batch(13)
    l1 = fb(trainer, args, accelerator, b1, unet, network, tes)  # replica, warmup: waits for the current micro-step
    fb(trainer, args, accelerator, b2, unet, network, tes)  # master
    trainer.all_reduce_network(accelerator, network)
    for p in trainer._replica_master_params:
        p.grad = None
    l3 = fb(trainer, args, accelerator, b3, unet, network, tes)  # replica, warmup over: lagged -> loss of b1 again
    assert l3.item() == pytest.approx(l1.item())


@cuda_only
def test_update_metadata_records_original_gradient_accumulation():
    args = toy_args(replica_devices="0", gradient_accumulation_steps=3)
    trainer_cls = make_toy_trainer_class()
    trainer = trainer_cls()
    trainer.assert_extra_args(args, _DatasetGroupStub(), None)
    metadata = {"ss_gradient_accumulation_steps": args.gradient_accumulation_steps}
    trainer.update_metadata(metadata, args)
    assert metadata["ss_gradient_accumulation_steps"] == 3
    assert metadata["ss_replica_devices"] == "cuda:0"


# endregion
