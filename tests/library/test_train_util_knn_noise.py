import argparse
import importlib.machinery
import sys
import types

import pytest
import torch
import transformers.utils

try:
    import xformers.ops  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    xformers_stub = types.ModuleType("xformers")
    xformers_stub.__spec__ = importlib.machinery.ModuleSpec("xformers", loader=None)
    xformers_stub.__version__ = "0.0.0"
    xformers_ops_stub = types.ModuleType("xformers.ops")
    xformers_ops_stub.__spec__ = importlib.machinery.ModuleSpec("xformers.ops", loader=None)
    xformers_stub.ops = xformers_ops_stub
    sys.modules["xformers"] = xformers_stub
    sys.modules["xformers.ops"] = xformers_ops_stub

if not hasattr(transformers.utils, "FLAX_WEIGHTS_NAME"):
    transformers.utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"

from library import train_util


def _build_verify_args(**overrides):
    base = dict(
        highvram=False,
        v2=False,
        clip_skip=None,
        cache_latents_to_disk=False,
        cache_latents=False,
        train_inpainting=False,
        knn_noise_k=1,
        adaptive_noise_scale=None,
        noise_offset=None,
        scale_v_pred_loss_like_noise_pred=False,
        v_parameterization=False,
        v_pred_like_loss=0.0,
        zero_terminal_snr=False,
        sample_every_n_epochs=None,
        sample_every_n_steps=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_sample_knn_noise_selects_nearest_candidate(monkeypatch):
    latents = torch.tensor([[[[0.1]]], [[[10.0]]]], dtype=torch.float32)
    expected = torch.tensor([[[[0.2]]], [[[9.7]]]], dtype=torch.float32)
    candidate_noise = torch.tensor(
        [
            [[[[0.5]]], [[[0.2]]], [[[-1.0]]]],
            [[[[1.0]]], [[[9.7]]], [[[13.0]]]],
        ],
        dtype=torch.float32,
    )

    real_randn = torch.randn

    def fake_randn(size, *args, **kwargs):
        if tuple(size) == tuple(candidate_noise.shape):
            return candidate_noise.to(device=kwargs.get("device"), dtype=kwargs.get("dtype"))
        return real_randn(size, *args, **kwargs)

    monkeypatch.setattr(train_util.torch, "randn", fake_randn)

    noise = train_util.sample_knn_noise(latents, k=3)

    assert torch.allclose(noise, expected)


def test_sample_training_noise_uses_gaussian_when_knn_noise_k_is_one(monkeypatch):
    latents = torch.zeros(2, 4, 8, 8, dtype=torch.float32)
    expected = torch.ones_like(latents)
    args = argparse.Namespace(knn_noise_k=1)

    monkeypatch.setattr(train_util.torch, "randn_like", lambda *_, **__: expected)

    noise = train_util.sample_training_noise(args, latents)

    assert torch.equal(noise, expected)


def test_knn_distance_is_computed_in_float16(monkeypatch):
    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    candidates = torch.randn(2, 3, 4, 8, 8, dtype=torch.float32)
    captured = {}
    real_vector_norm = torch.linalg.vector_norm

    def fake_vector_norm(input, *args, **kwargs):
        captured["dtype"] = input.dtype
        return real_vector_norm(input, *args, **kwargs)

    monkeypatch.setattr(train_util.torch.linalg, "vector_norm", fake_vector_norm)

    train_util.select_nearest_noise_candidate(latents, candidates)

    assert captured["dtype"] == torch.float16


def test_sample_knn_noise_rejects_invalid_k():
    latents = torch.randn(2, 4, 8, 8)
    with pytest.raises(ValueError, match="greater than or equal to 1"):
        train_util.sample_knn_noise(latents, k=0)


def test_verify_training_args_rejects_invalid_knn_noise_k():
    args = _build_verify_args(knn_noise_k=0)
    with pytest.raises(ValueError, match="knn_noise_k"):
        train_util.verify_training_args(args)


def test_get_noise_noisy_latents_and_timesteps_uses_knn_selection(monkeypatch):
    latents = torch.randn(2, 4, 8, 8, dtype=torch.float32)
    expected_noise = torch.full_like(latents, 0.25)

    class DummyScheduler:
        def __init__(self):
            self.config = types.SimpleNamespace(num_train_timesteps=1000)
            self.alphas_cumprod = torch.tensor([1.0], dtype=torch.float32)

        def add_noise(self, latents, noise, timesteps):
            return latents + noise

    args = argparse.Namespace(
        knn_noise_k=8,
        noise_offset=None,
        noise_offset_random_strength=False,
        adaptive_noise_scale=None,
        multires_noise_iterations=None,
        multires_noise_discount=0.3,
        min_timestep=None,
        max_timestep=None,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
    )
    called = {"knn": False}

    def fake_sample_training_noise(args, latents):
        if args.knn_noise_k > 0:
            called["knn"] = True
        return expected_noise

    monkeypatch.setattr(train_util, "sample_training_noise", fake_sample_training_noise)

    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, DummyScheduler(), latents)

    assert called["knn"] is True
    assert torch.equal(noise, expected_noise)
    assert torch.equal(noisy_latents, latents + expected_noise)
    assert timesteps.shape == (latents.shape[0],)
