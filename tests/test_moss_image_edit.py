"""Tests for the reference-conditioned editing of MOSS-Image (moss_image_train_network_edit.py): reference tokens in the
DiT, reference latents in the sampler, and the conditioning latents cache of the ControlNet dataset (CPU, small random DiT)."""

import argparse
import contextlib
import os
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image

from library import config_util, flux2_vae, moss_image_models, moss_image_train_utils, strategy_base
from library.controlnet_dataset import ControlNetDataset
from library.subset import ControlNetSubset

SMALL_CONFIG = dict(dim=64, ffn_dim=96, num_heads=4, num_layers=3, text_dim=24, freq_dim=16, latent_channels=8, rope_base=10000.0)
NUM_BANDS = SMALL_CONFIG["dim"] // SMALL_CONFIG["num_heads"] // 8  # 2 bands per axis


class FakeAccelerator:
    device = torch.device("cpu")
    num_processes = 1
    process_index = 0

    def autocast(self):
        return contextlib.nullcontext()

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self):
        pass


def make_dit(seed=0, split_attn=False):
    torch.manual_seed(seed)
    return moss_image_models.MossImage(**SMALL_CONFIG, attn_mode="torch", split_attn=split_attn).eval()


def make_inputs(ref_hw=(3, 5)):
    torch.manual_seed(1)
    batch, height, width = 2, 3, 5
    lens = [4, 7]
    x = torch.randn(batch, SMALL_CONFIG["latent_channels"], height, width)
    ref = torch.randn(batch, SMALL_CONFIG["latent_channels"], *ref_hw)
    context = torch.randn(batch, max(lens), SMALL_CONFIG["text_dim"])
    mask = torch.zeros(batch, max(lens), dtype=torch.int32)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    context = context * mask[..., None]
    timesteps = torch.tensor([750.0, 120.0])
    return x, ref, timesteps, context, mask


# region RoPE


def _r_axis_dims():
    # layout: [sin (r, f, h, w) x bands, cos (r, f, h, w) x bands]
    half = 4 * NUM_BANDS
    return list(range(0, NUM_BANDS)) + list(range(half, half + NUM_BANDS))


def test_rope_reference_tokens_follow_image_tokens():
    rope = moss_image_models.MossImageRoPE(SMALL_CONFIG["dim"] // SMALL_CONFIG["num_heads"], SMALL_CONFIG["rope_base"])
    text_lens = torch.tensor([4, 7])
    base = rope(3, 5, 7, text_lens)
    with_ref_r0 = rope(3, 5, 7, text_lens, ref_size=(3, 5), ref_r=0.0)
    with_ref_r1 = rope(3, 5, 7, text_lens, ref_size=(3, 5), ref_r=1.0)
    n = 15
    assert base.shape == (2, n + 7, 16)
    assert with_ref_r0.shape == (2, 2 * n + 7, 16)

    # image and text parts are unchanged
    torch.testing.assert_close(with_ref_r1[:, :n], base[:, :n])
    torch.testing.assert_close(with_ref_r1[:, 2 * n :], base[:, n:])

    # r = 0: the reference tokens have exactly the positions of the image tokens
    torch.testing.assert_close(with_ref_r0[:, n : 2 * n], base[:, :n])

    # r = 1: only the r-axis dims differ from the image tokens (sin(0) = 0 / cos(0) = 1 for the image tokens)
    diff = (with_ref_r1[:, n : 2 * n] - base[:, :n]).abs().amax(dim=(0, 1))
    r_dims = _r_axis_dims()
    other_dims = [d for d in range(16) if d not in r_dims]
    assert torch.all(diff[other_dims] == 0)
    assert torch.all(diff[r_dims] > 0)


def test_rope_reference_grid_size_can_differ():
    rope = moss_image_models.MossImageRoPE(SMALL_CONFIG["dim"] // SMALL_CONFIG["num_heads"], SMALL_CONFIG["rope_base"])
    text_lens = torch.tensor([4, 7])
    out = rope(3, 5, 7, text_lens, ref_size=(2, 4), ref_r=1.0)
    assert out.shape == (2, 15 + 8 + 7, 16)


# endregion


# region DiT forward


def test_forward_with_reference_shapes_and_effect():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs()
    with torch.no_grad():
        base = model(x, timesteps, context, mask)
        with_ref = model(x, timesteps, context, mask, ref=ref, ref_rope_r=1.0)
        with_ref_r0 = model(x, timesteps, context, mask, ref=ref, ref_rope_r=0.0)
        with_other_ref = model(x, timesteps, context, mask, ref=torch.randn_like(ref), ref_rope_r=1.0)
    assert with_ref.shape == x.shape  # only the image tokens are output
    assert not torch.allclose(with_ref, base)
    assert not torch.allclose(with_ref, with_ref_r0)
    assert not torch.allclose(with_ref, with_other_ref)


def test_forward_with_reference_of_different_size():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs(ref_hw=(2, 4))
    with torch.no_grad():
        out = model(x, timesteps, context, mask, ref=ref)
    assert out.shape == x.shape


def test_forward_with_reference_rejects_bad_shapes():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs()
    with pytest.raises(ValueError):
        model(x, timesteps, context, mask, ref=ref[:1])
    with pytest.raises(ValueError):
        model(x, timesteps, context, mask, ref=ref[:, :4])


@pytest.mark.parametrize("split_attn", [False, True])
def test_batched_forward_with_reference_matches_per_sample(split_attn):
    # the attention mask handling with the extended image length: batched (padded text) == per-sample (true text length)
    model = make_dit(split_attn=split_attn)
    x, ref, timesteps, context, mask = make_inputs()
    with torch.no_grad():
        batched = model(x, timesteps, context, mask, ref=ref, ref_rope_r=1.0)
        for i in range(x.shape[0]):
            l = int(mask[i].sum())
            single = model(x[i : i + 1], timesteps[i : i + 1], context[i : i + 1, :l], None, ref=ref[i : i + 1], ref_rope_r=1.0)
            torch.testing.assert_close(batched[i : i + 1], single, atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_with_reference_matches_plain_forward():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs()
    with torch.no_grad():
        expected = model(x, timesteps, context, mask, ref=ref)
    model.train()
    model.enable_gradient_checkpointing()
    x_grad = x.clone().requires_grad_(True)
    ref_grad = ref.clone().requires_grad_(True)
    out = model(x_grad, timesteps, context, mask, ref=ref_grad)
    out.sum().backward()
    torch.testing.assert_close(out.detach(), expected, atol=1e-5, rtol=1e-5)
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()
    assert ref_grad.grad is not None and ref_grad.grad.abs().sum() > 0  # the reference influences the output


# endregion


# region sampler


def test_generate_latents_with_reference():
    dit = make_dit()
    embeds = torch.randn(1, 5, SMALL_CONFIG["text_dim"])
    mask = torch.ones(1, 5, dtype=torch.int32)
    neg = torch.randn(1, 3, SMALL_CONFIG["text_dim"])
    neg_mask = torch.ones(1, 3, dtype=torch.int32)
    ref = torch.randn(1, SMALL_CONFIG["latent_channels"], 3, 5)
    common = dict(height=48, width=80, steps=4, flow_shift=5.0, uncond_skip_layers=(1,), seed=42, device=torch.device("cpu"))

    a = moss_image_train_utils.generate_latents(dit, embeds, mask, neg, neg_mask, guidance_scale=3.0, show_progress=False, ref=ref, **common)
    b = moss_image_train_utils.generate_latents(dit, embeds, mask, neg, neg_mask, guidance_scale=3.0, show_progress=False, ref=ref, **common)
    c = moss_image_train_utils.generate_latents(dit, embeds, mask, neg, neg_mask, guidance_scale=3.0, show_progress=False, **common)
    assert a.shape == (1, SMALL_CONFIG["latent_channels"], 3, 5)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


# endregion


# region trainer


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
        ref_dropout_rate=0.0,
        ref_rope_r=1.0,
        cache_ref_latents=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def test_setup_parser_edit_defaults():
    import moss_image_train_network_edit

    parser = moss_image_train_network_edit.setup_parser()
    args = parser.parse_args([])
    assert args.ref_dropout_rate == 0.0
    assert args.ref_rope_r == 1.0
    assert args.cache_ref_latents is False
    assert args.cache_ref_latents_to_disk is False
    assert args.ref_latents_cache_dir is None
    assert args.ref_mask_source == "none"
    assert args.ref_mask_space == "latent"
    assert args.ref_mask_fill_pixel == 128
    assert args.ref_mask_fill_latent == 0.0


def test_random_reference_masks():
    import moss_image_train_network_edit

    masks = moss_image_train_network_edit.generate_random_reference_masks(2, 64, 96)
    assert masks.shape == (2, 64, 96) and masks.dtype == torch.float32
    for i in range(2):
        assert set(masks[i].unique().tolist()) == {0.0, 1.0}
        assert 0 < masks[i].mean() < 1

    torch.manual_seed(0)
    images = torch.rand(2, 3, 64, 96) * 2 - 1
    masked = moss_image_train_network_edit.apply_random_reference_masks_pixel(images, 128)
    assert masked.shape == images.shape
    value = 128 / 127.5 - 1.0
    for i in range(2):
        changed = (masked[i] != images[i]).any(0)  # [H, W]
        assert changed.any() and not changed.all()
        assert torch.all(masked[i][:, changed] == value)
    assert not torch.equal(images, masked)  # the input is not modified in place


def test_latent_mask_blend_has_sub_token_precision():
    torch.manual_seed(0)
    # pixel mask [B, H, W] covering exactly one 8px latent cell (the top-left cell of the second token) plus a half-covered cell
    mask = torch.zeros(1, 64, 96)
    mask[0, 0:8, 16:24] = 1.0  # one full latent cell -> m = 1
    mask[0, 32:36, 0:8] = 1.0  # half of a cell -> m = 0.5
    m = moss_image_train_utils.downsample_mask_to_latents(mask)
    assert m.shape == (1, 1, 8, 12)
    assert m[0, 0, 0, 2] == 1.0 and m[0, 0, 4, 0] == 0.5 and m.sum() == 1.5

    ref = torch.randn(1, 128, 4, 6)  # normalized, patchified latents of a 64x96 image
    fill = 0.0
    out = moss_image_train_utils.apply_reference_latent_mask(ref, m, fill)
    assert out.shape == ref.shape
    # per latent cell: unpatchify and compare with the 1/8 resolution blend
    ref_cells = flux2_vae.unpatchify_latents(ref)  # [1, 32, 8, 12]
    out_cells = flux2_vae.unpatchify_latents(out)
    expected = ref_cells * (1 - m) + fill * m
    torch.testing.assert_close(out_cells, expected)
    assert torch.all(out_cells[0, :, 0, 2] == fill)  # the masked cell is filled
    assert torch.all(out_cells[0, :, 0, 3] == ref_cells[0, :, 0, 3])  # the other cell of the same token is untouched
    torch.testing.assert_close(out_cells[0, :, 4, 0], 0.5 * ref_cells[0, :, 4, 0])  # the half covered cell is blended


def test_edit_trainer_call_dit_and_reference_dropout():
    import moss_image_train_network_edit
    from library import sd3_train_utils

    trainer = moss_image_train_network_edit.MossImageEditNetworkTrainer()
    assert trainer.use_controlnet_dataset(make_args()) is True

    dit = make_dit()
    accelerator = FakeAccelerator()
    x, ref, timesteps, context, mask = make_inputs()
    batch = {"ref_latents": ref}

    args = make_args()
    out = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, is_train=True)
    with torch.no_grad():
        expected = dit(x, timesteps, context, mask, ref=ref, ref_rope_r=1.0)
    torch.testing.assert_close(out, expected)

    # reference dropout 1.0 replaces the reference with zeros (training only)
    args = make_args(ref_dropout_rate=1.0)
    dropped = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, is_train=True)
    with torch.no_grad():
        expected = dit(x, timesteps, context, mask, ref=torch.zeros_like(ref), ref_rope_r=1.0)
    torch.testing.assert_close(dropped, expected)
    kept = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, is_train=False)
    assert not torch.allclose(kept, dropped)

    # the full get_noise_pred_and_target path uses the reference
    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    args = make_args()
    batch = {"ref_latents": ref, "custom_attributes": [{}, {}]}
    model_pred, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args, accelerator, noise_scheduler, x, batch, [context, mask, torch.zeros(2)], dit, None, torch.float32, True, is_train=True
    )
    assert model_pred.shape == x.shape and target.shape == x.shape


def test_load_reference_pixels_with_alpha_or_mask(tmp_path):
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[..., 0] = 200
    alpha = np.zeros((32, 48), dtype=np.uint8)
    alpha[8:16, 8:24] = 255
    Image.fromarray(np.dstack([rgb, alpha])).save(tmp_path / "ref.png")
    Image.fromarray(rgb).save(tmp_path / "ref_rgb.png")
    Image.fromarray(alpha).save(tmp_path / "mask.png")

    # default: the alpha channel is ignored, no mask
    pixels, mask = moss_image_train_utils.load_reference_pixels(str(tmp_path / "ref.png"), 48, 32)
    assert pixels.shape == (1, 3, 32, 48) and mask is None
    torch.testing.assert_close(pixels[0, 0], torch.full((32, 48), 200 / 127.5 - 1.0))

    # mask from the alpha channel (pixels unchanged)
    pixels2, mask = moss_image_train_utils.load_reference_pixels(str(tmp_path / "ref.png"), 48, 32, mask_from_alpha=True)
    torch.testing.assert_close(pixels2, pixels)
    assert mask.shape == (1, 32, 48)
    assert torch.all(mask[0, 8:16, 8:24] == 1.0) and mask.sum() == 8 * 16

    # mask file, with a resize
    pixels3, mask = moss_image_train_utils.load_reference_pixels(str(tmp_path / "ref_rgb.png"), 96, 64, mask_path=str(tmp_path / "mask.png"))
    assert pixels3.shape == (1, 3, 64, 96) and mask.shape == (1, 64, 96)
    assert torch.all(mask[0, 18:30, 18:46] == 1.0) and torch.all(mask[0, :14] == 0.0)

    # no alpha and no mask: warning, mask None
    _, mask = moss_image_train_utils.load_reference_pixels(str(tmp_path / "ref_rgb.png"), 48, 32, mask_from_alpha=True)
    assert mask is None


# endregion


# region ControlNet dataset conditioning latents cache


class FakeTokenizeStrategy:
    def tokenize(self, text):
        return [torch.zeros(1, 4, dtype=torch.long)]


class FakeVAE(torch.nn.Module):
    """8x downscale 'VAE' for the caching strategy: [B, 3, H, W] -> [B, 3, H/8, W/8]"""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    @property
    def device(self):
        return self.dummy.device

    @property
    def dtype(self):
        return self.dummy.dtype

    def encode(self, x):
        self.calls += 1
        return torch.nn.functional.avg_pool2d(x, 8)


def _write_image(path, size, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)


def _write_rgba_image(path, size, seed):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    alpha = np.zeros((size[1], size[0]), dtype=np.uint8)
    alpha[size[1] // 4 : size[1] // 2, size[0] // 4 : size[0] // 2] = 255
    Image.fromarray(np.dstack([rgb, alpha])).save(path)


def make_controlnet_dataset(tmp_path, flip_aug=False, enable_bucket=True, rgba_cond=False):
    image_dir = tmp_path / "images"
    cond_dir = tmp_path / "cond"
    image_dir.mkdir(exist_ok=True)
    cond_dir.mkdir(exist_ok=True)
    sizes = {"a": (64, 64), "b": (96, 64), "c": (64, 64)}
    for i, (name, size) in enumerate(sizes.items()):
        _write_image(image_dir / f"{name}.png", size, i)
        if rgba_cond:
            _write_rgba_image(cond_dir / f"{name}.png", size, 100 + i)
        else:
            _write_image(cond_dir / f"{name}.png", size, 100 + i)
        (image_dir / f"{name}.txt").write_text(f"caption {name}", encoding="utf-8")

    subset = ControlNetSubset(
        **asdict(
            config_util.ControlNetSubsetParams(
                image_dir=str(image_dir), conditioning_data_dir=str(cond_dir), caption_extension=".txt", flip_aug=flip_aug
            )
        )
    )
    dataset = ControlNetDataset(
        subsets=[subset],
        **asdict(
            config_util.ControlNetDatasetParams(
                resolution=(64, 64), batch_size=2, enable_bucket=enable_bucket, min_bucket_reso=32, max_bucket_reso=128, bucket_reso_steps=32
            )
        ),
    )
    dataset.make_buckets()
    dataset.dreambooth_dataset_delegate.tokenize_strategy = FakeTokenizeStrategy()
    return dataset


class FakeLatentsCachingStrategy(strategy_base.LatentsCachingStrategy):
    SUFFIX = "_fake.npz"

    @property
    def cache_suffix(self):
        return self.SUFFIX

    def get_latents_npz_path(self, absolute_path, image_size):
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso, npz_path, flip_aug, alpha_mask):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(self, npz_path, bucket_reso):
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos, flip_aug, alpha_mask, random_crop):
        self._default_cache_batch_latents(vae.encode, vae.device, vae.dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True)


def _collect_examples(dataset):
    examples = [dataset[i] for i in range(len(dataset))]
    assert len(examples) >= 1
    return examples


@pytest.mark.parametrize("cache_to_disk", [False, True])
def test_controlnet_dataset_caches_conditioning_latents(tmp_path, cache_to_disk):
    dataset = make_controlnet_dataset(tmp_path, flip_aug=False)

    # pixel path before caching
    example = dataset[0]
    assert "conditioning_images" in example and "conditioning_latents" not in example
    cond_images = example["conditioning_images"]
    assert cond_images.shape[1] == 3 and cond_images.shape[-2:] == tuple(example["images"].shape[-2:])

    vae = FakeVAE()
    strategy = FakeLatentsCachingStrategy(cache_to_disk, batch_size=2, skip_disk_cache_validity_check=False)
    dataset.new_cache_conditioning_latents(vae, FakeAccelerator(), strategy)
    assert vae.calls > 0

    for info in dataset.image_data.values():
        if cache_to_disk:
            assert info.cond_latents_npz is not None and os.path.exists(info.cond_latents_npz)
            assert os.path.dirname(info.cond_latents_npz) == os.path.dirname(info.cond_img_path)  # next to the conditioning image
        else:
            assert info.cond_latents is not None and info.cond_latents_flipped is None

    # latents path after caching: the cached latents equal the encoding of the (deterministically resized) conditioning images
    for example in _collect_examples(dataset):
        assert "conditioning_images" not in example
        latents = example["conditioning_latents"]
        images = example["images"]
        assert latents.shape == (images.shape[0], 3, images.shape[-2] // 8, images.shape[-1] // 8)

    # the cached latents match the pixel path of the same item (deterministic transform, no flip)
    dataset.cond_latents_cache_mode = None
    for i in range(len(dataset)):
        pixel_example = dataset[i]
        dataset.cond_latents_cache_mode = "disk" if cache_to_disk else "memory"
        cached = dataset[i]["conditioning_latents"]
        dataset.cond_latents_cache_mode = None
        torch.testing.assert_close(cached, vae.encode(pixel_example["conditioning_images"]), atol=1e-5, rtol=1e-5)

    if cache_to_disk:
        # a second pass finds the cache and encodes nothing
        calls = vae.calls
        dataset.new_cache_conditioning_latents(vae, FakeAccelerator(), strategy)
        assert vae.calls == calls


def test_controlnet_dataset_conditioning_latents_cache_dir_and_flip(tmp_path):
    dataset = make_controlnet_dataset(tmp_path, flip_aug=True)
    vae = FakeVAE()
    strategy = FakeLatentsCachingStrategy(True, batch_size=4, skip_disk_cache_validity_check=False)
    cache_dir = tmp_path / "cache"
    dataset.new_cache_conditioning_latents(vae, FakeAccelerator(), strategy, str(cache_dir))

    npz_paths = [info.cond_latents_npz for info in dataset.image_data.values()]
    assert all(os.path.dirname(p) == str(cache_dir) for p in npz_paths)
    assert len(set(npz_paths)) == len(npz_paths)
    for p in npz_paths:
        npz = np.load(p)
        assert any(k.startswith("latents_flipped") for k in npz.files)

    # flipped items get the flipped latents
    torch.manual_seed(0)
    import random

    random.seed(0)
    seen_flipped = False
    for _ in range(8):
        for i in range(len(dataset)):
            example = dataset[i]
            bucket_index = dataset.buckets_indices[i]
            bucket = dataset.bucket_manager.buckets[bucket_index.bucket_index]
            start = bucket_index.batch_index * bucket_index.bucket_batch_size
            keys = bucket[start : start + bucket_index.bucket_batch_size]
            for j, flipped in enumerate(example["flippeds"]):
                latents = example["conditioning_latents"][j]
                info = dataset.image_data[keys[j]]
                npz = np.load(info.cond_latents_npz)
                reso_suffix = f"_{latents.shape[-2]}x{latents.shape[-1]}"
                expected = npz[("latents_flipped" if flipped else "latents") + reso_suffix]
                torch.testing.assert_close(latents, torch.from_numpy(expected))
                seen_flipped |= bool(flipped)
    assert seen_flipped


def test_controlnet_dataset_cond_alpha_fill(tmp_path):
    dataset = make_controlnet_dataset(tmp_path, rgba_cond=True)
    gray = 128 / 127.5 - 1.0

    # without fill the alpha is dropped: no gray block
    example = dataset[0]
    cond = example["conditioning_images"]
    assert not torch.all(cond[:, :, 16:32, 16:32] == gray)

    # with fill: the alpha region (a quarter block at 1/4..1/2) is gray in the pixel path ...
    dataset.cond_alpha_fill = 128
    example = dataset[0]
    cond_filled = example["conditioning_images"]
    h, w = cond_filled.shape[-2:]
    block = cond_filled[:, :, h // 4 : h // 2, w // 4 : w // 2]
    torch.testing.assert_close(block, torch.full_like(block, gray))
    torch.testing.assert_close(cond_filled[:, :, : h // 4], cond[:, :, : h // 4])

    # ... and in the cached latents path (same encoding of the same filled pixels)
    vae = FakeVAE()
    strategy = FakeLatentsCachingStrategy(False, batch_size=2, skip_disk_cache_validity_check=False)
    dataset.new_cache_conditioning_latents(vae, FakeAccelerator(), strategy)
    cached = dataset[0]["conditioning_latents"]
    torch.testing.assert_close(cached, vae.encode(cond_filled), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cache_to_disk", [False, True])
def test_controlnet_dataset_cond_load_alpha_returns_masks(tmp_path, cache_to_disk):
    dataset = make_controlnet_dataset(tmp_path, flip_aug=True, rgba_cond=True)
    dataset.cond_load_alpha = True

    def check(example):
        images_or_latents = example["conditioning_images"] if "conditioning_images" in example else example["conditioning_latents"]
        masks = example["conditioning_masks"]
        h, w = example["images"].shape[-2:]
        assert masks.shape == (images_or_latents.shape[0], h, w)
        for j, flipped in enumerate(example["flippeds"]):
            m = masks[j]
            assert set(m.unique().tolist()) <= {0.0, 1.0} and 0 < m.mean() < 1
            # the hole block is at rows 1/4..1/2 and columns 1/4..1/2 of the source, mirrored when flipped
            cols = slice(w // 4, w // 2) if not flipped else slice(w - w // 2, w - w // 4)
            assert torch.all(m[h // 4 : h // 2, cols] == 1.0)
            assert torch.all(m[: h // 4] == 0.0)
        return masks

    # pixel path: RGB images + masks from the alpha channel
    import random

    random.seed(1)
    pixel_masks = {}
    for i in range(len(dataset)):
        example = dataset[i]
        assert example["conditioning_images"].shape[1] == 3
        check(example)

    # cached path: masks are cached with the latents (npz alpha_mask key on disk) and flipped with them
    vae = FakeVAE()
    strategy = FakeLatentsCachingStrategy(cache_to_disk, batch_size=2, skip_disk_cache_validity_check=False)
    dataset.new_cache_conditioning_latents(vae, FakeAccelerator(), strategy)
    if cache_to_disk:
        for info in dataset.image_data.values():
            assert any(k.startswith("alpha_mask") for k in np.load(info.cond_latents_npz).files)
    seen = set()
    for _ in range(6):
        for i in range(len(dataset)):
            example = dataset[i]
            assert "conditioning_images" not in example
            check(example)
            seen.update(bool(f) for f in example["flippeds"])
    assert seen == {True, False}


# endregion
