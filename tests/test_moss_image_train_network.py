"""Tests for moss_image_train_network.py and the trainer utilities (CPU, small random DiT)."""

import argparse
import contextlib
from types import SimpleNamespace

import pytest
import torch

from library import flux2_vae, moss_image_models, moss_image_train_utils

SMALL_CONFIG = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)


class FakeAccelerator:
    device = torch.device("cpu")

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model


def make_dit(seed=0):
    torch.manual_seed(seed)
    return moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=False).eval()


def make_args(**overrides):
    args = dict(
        timestep_sampling="sigma",
        weighting_scheme="logit_normal",
        logit_mean=-1.09,
        logit_std=0.95,
        mode_scale=1.29,
        sigmoid_scale=1.0,
        discrete_flow_shift=1.0,
        ip_noise_gamma=None,
        ip_noise_gamma_random_strength=False,
        gradient_checkpointing=False,
        show_timesteps="console",
        show_timesteps_resolution="1024",
        show_timesteps_offset=0.0,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_setup_parser_defaults():
    import moss_image_train_network

    parser = moss_image_train_network.setup_parser()
    args = parser.parse_args(["--pretrained_model_name_or_path", "dit.safetensors", "--text_encoder", "te.safetensors"])
    assert args.text_encoder == "te.safetensors"
    assert args.moss_max_token_length == 4096
    assert args.timestep_sampling == "sigmoid" and args.discrete_flow_shift == 1.0
    assert args.sample_uncond_skip_layers == "22"
    assert args.compile is False and args.cuda_allow_tf32 is False
    assert args.tokenizer_cache_dir is None and args.vae is None


def test_shift_scale_latents_matches_vae_helper():
    import moss_image_train_network

    trainer = moss_image_train_network.MossImageNetworkTrainer()
    torch.manual_seed(0)
    trainer.bn_running_mean = torch.randn(128)
    trainer.bn_running_var = torch.rand(128) + 0.5
    raw = torch.randn(2, 32, 8, 12)
    out = trainer.shift_scale_latents(None, raw)
    assert out.shape == (2, 128, 4, 6)
    expected = flux2_vae.normalize_latents(flux2_vae.patchify_latents(raw), trainer.bn_running_mean, trainer.bn_running_var)
    torch.testing.assert_close(out, expected)
    # round trip through the inverse (used for sample image decoding)
    restored = flux2_vae.unpatchify_latents(flux2_vae.denormalize_latents(out, trainer.bn_running_mean, trainer.bn_running_var))
    torch.testing.assert_close(restored, raw, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("timestep_sampling", ["sigma", "sigmoid", "shift"])
def test_get_noise_pred_and_target(timestep_sampling):
    import moss_image_train_network
    from library import sd3_train_utils

    trainer = moss_image_train_network.MossImageNetworkTrainer()
    args = make_args(timestep_sampling=timestep_sampling)
    dit = make_dit()
    accelerator = FakeAccelerator()
    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)

    torch.manual_seed(1)
    latents = torch.randn(2, SMALL_CONFIG["latent_channels"], 3, 5)
    embeds = torch.randn(2, 7, SMALL_CONFIG["text_dim"])
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.int32)
    rates = torch.zeros(2)
    batch = {"custom_attributes": [{"timestep_sampling": {"offset": 0.0}}, {}]}

    torch.manual_seed(2)
    model_pred, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args, accelerator, noise_scheduler, latents, batch, [embeds, mask, rates], dit, None, torch.bfloat16, True, is_train=True
    )
    assert model_pred.shape == latents.shape
    assert target.shape == latents.shape
    assert timesteps.shape == (2,) and timesteps.dtype == torch.float32
    assert torch.all(timesteps > 0) and torch.all(timesteps <= 1000)
    assert weighting.shape == (2, 1, 1, 1) and torch.all(weighting == 1.0)  # logit_normal: no loss weighting

    # target is v = noise - x0: recover the noise and check the timesteps are sigma * 1000 (not divided by 1000)
    noise = target + latents
    sigmas = (timesteps / 1000).view(-1, 1, 1, 1)
    torch.manual_seed(2)
    expected_noise = torch.randn_like(latents)
    torch.testing.assert_close(noise, expected_noise)
    assert torch.all(sigmas > 0) and torch.all(sigmas <= 1)


def test_show_timesteps_console(capsys):
    args = make_args()
    moss_image_train_utils.show_timesteps(args)
    out = capsys.readouterr().out
    assert "timestep_sampling=sigma" in out
    assert "latent 64x64" in out  # 1024 / 16


def test_generate_latents_shapes_and_determinism():
    dit = make_dit()
    embeds = torch.randn(1, 5, SMALL_CONFIG["text_dim"])
    mask = torch.ones(1, 5, dtype=torch.int32)
    neg = torch.randn(1, 3, SMALL_CONFIG["text_dim"])
    neg_mask = torch.ones(1, 3, dtype=torch.int32)
    common = dict(height=48, width=80, steps=4, flow_shift=5.0, uncond_skip_layers=(1,), seed=42, device=torch.device("cpu"))

    a = moss_image_train_utils.generate_latents(dit, embeds, mask, neg, neg_mask, guidance_scale=3.0, show_progress=False, **common)
    b = moss_image_train_utils.generate_latents(dit, embeds, mask, neg, neg_mask, guidance_scale=3.0, show_progress=False, **common)
    assert a.shape == (1, SMALL_CONFIG["latent_channels"], 3, 5) and a.dtype == torch.float32
    torch.testing.assert_close(a, b)

    # guidance 1.0 disables CFG (the negative branch is not evaluated), so the result differs
    c = moss_image_train_utils.generate_latents(dit, embeds, mask, neg, neg_mask, guidance_scale=1.0, show_progress=False, **common)
    assert not torch.allclose(a, c)
    d = moss_image_train_utils.generate_latents(dit, embeds, mask, None, None, guidance_scale=3.0, show_progress=False, **common)
    torch.testing.assert_close(c, d)


def test_sampling_sigmas_snap_to_training_grid():
    sigmas = moss_image_train_utils.get_sampling_sigmas(10, shift=5.0)
    assert sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert torch.all(sigmas[:-1] > sigmas[1:])
    training = moss_image_train_utils._sigma_grid(1000, 1.0, 1000)
    for s in sigmas[:-1]:
        assert torch.any(torch.isclose(training, s))
