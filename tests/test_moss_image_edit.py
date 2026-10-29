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
        ref_mode="concat",
        ref_timestep=0.0,
        ref_kv_blocks=None,
        ref_kv_lora="on",
        ref_gate=False,
        ref_gate_bands="950,850,700,500",
        ref_gate_init=2.0,
        ref_gate_lr=1e-2,
        ref_gate_budget=None,
        ref_gate_budget_weight=0.1,
        ref_gate_balanced_timesteps=False,
        ref_gate_binarize=False,
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
    assert args.ref_mode == "concat"
    assert args.ref_timestep == 0.0
    assert args.ref_kv_blocks is None
    assert args.ref_kv_lora == "on"
    assert args.ref_gate is False and args.ref_gate_binarize is False
    assert args.ref_gate_bands == "950,850,700,500" and args.ref_gate_init == 2.0 and args.ref_gate_lr == 1e-2
    assert args.ref_gate_budget is None and args.ref_gate_budget_weight == 0.1 and args.ref_gate_balanced_timesteps is False


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
    out = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, None, is_train=True)
    with torch.no_grad():
        expected = dit(x, timesteps, context, mask, ref=ref, ref_rope_r=1.0)
    torch.testing.assert_close(out, expected)

    # reference dropout 1.0 replaces the reference with zeros (training only)
    args = make_args(ref_dropout_rate=1.0)
    dropped = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, None, is_train=True)
    with torch.no_grad():
        expected = dit(x, timesteps, context, mask, ref=torch.zeros_like(ref), ref_rope_r=1.0)
    torch.testing.assert_close(dropped, expected)
    kept = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, None, is_train=False)
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


# region one-way reference with cached K/V (Step B)

from library.attention import AttentionParams, attention


def _manual_attention(q, k, v, q_valid, k_valid):
    """Reference attention [B, L, H, D] with per-sample valid lengths of the queries and keys (fp32 softmax)."""
    out = torch.zeros_like(q)
    scale = q.shape[-1] ** -0.5
    for i in range(q.shape[0]):
        qi = q[i, : q_valid[i]].transpose(0, 1).float()  # [H, Lq, D]
        ki = k[i, : k_valid[i]].transpose(0, 1).float()
        vi = v[i, : k_valid[i]].transpose(0, 1).float()
        w = torch.softmax(qi @ ki.transpose(1, 2) * scale, dim=-1)
        out[i, : q_valid[i]] = (w @ vi).transpose(0, 1).to(q.dtype)
    return out


@pytest.mark.parametrize("split_attn", [False, True])
@pytest.mark.parametrize("lens", [[4, 7], [7, 7]])
def test_attention_with_extra_key_tokens_matches_manual(split_attn, lens):
    torch.manual_seed(0)
    batch, img_len, extra, text_len, heads, dim = 2, 6, 5, 7, 2, 8
    mask = torch.zeros(batch, text_len, dtype=torch.int32)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    q = torch.randn(batch, img_len + text_len, heads, dim)
    k_img, k_ref, k_txt = torch.randn(batch, img_len, heads, dim), torch.randn(batch, extra, heads, dim), torch.randn(batch, text_len, heads, dim)
    v_img, v_ref, v_txt = torch.randn(batch, img_len, heads, dim), torch.randn(batch, extra, heads, dim), torch.randn(batch, text_len, heads, dim)
    k = torch.cat([k_img, k_ref, k_txt], 1)
    v = torch.cat([v_img, v_ref, v_txt], 1)

    q_valid = [img_len + l for l in lens]

    def assert_valid_close(actual, expected):
        # the outputs of the padding queries are not defined (they are ignored downstream)
        for i, n in enumerate(q_valid):
            torch.testing.assert_close(actual[i, :n], expected[i, :n], atol=1e-5, rtol=1e-5)

    params = AttentionParams.create_attention_params_from_mask("torch", split_attn, img_len, mask, img_len + extra)
    out = attention([q, k, v], attn_params=params)
    expected = _manual_attention(q, k, v, q_valid, [img_len + extra + l for l in lens]).flatten(2)
    assert_valid_close(out, expected)

    # without the extra tokens the result is the plain attention
    params0 = AttentionParams.create_attention_params_from_mask("torch", split_attn, img_len, mask)
    k0 = torch.cat([k_img, k_txt], 1)
    v0 = torch.cat([v_img, v_txt], 1)
    out0 = attention([q, k0, v0], attn_params=params0)
    expected0 = _manual_attention(q, k0, v0, q_valid, q_valid).flatten(2)
    assert_valid_close(out0, expected0)
    assert not torch.allclose(out[:, :img_len], out0[:, :img_len])

    # no mask: all tokens valid, the keys are simply longer than the queries
    params_none = AttentionParams.create_attention_params_from_mask("torch", split_attn, img_len, None)
    out_none = attention([q, k, v], attn_params=params_none)
    expected_none = _manual_attention(q, k, v, [q.shape[1]] * batch, [k.shape[1]] * batch).flatten(2)
    torch.testing.assert_close(out_none, expected_none, atol=1e-5, rtol=1e-5)


def test_rope_image_r():
    rope = moss_image_models.MossImageRoPE(SMALL_CONFIG["dim"] // SMALL_CONFIG["num_heads"], SMALL_CONFIG["rope_base"])
    text_lens = torch.tensor([4, 7])
    base = rope(3, 5, 7, text_lens)
    shifted = rope(3, 5, 7, text_lens, image_r=1.0)
    with_ref = rope(3, 5, 7, text_lens, ref_size=(3, 5), ref_r=1.0)
    torch.testing.assert_close(shifted[:, :15], with_ref[:, 15:30])  # the image tokens at r=1 == the reference tokens at r=1
    torch.testing.assert_close(shifted[:, 15:], base[:, 15:])  # text unchanged


def test_forward_ref_kv_shapes_blocks_and_early_exit():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs(ref_hw=(2, 4))
    heads, head_dim = SMALL_CONFIG["num_heads"], SMALL_CONFIG["dim"] // SMALL_CONFIG["num_heads"]
    with torch.no_grad():
        kvs = model.forward_ref_kv(ref, torch.zeros(2), context, mask, ref_rope_r=1.0)
    assert len(kvs) == SMALL_CONFIG["num_layers"]
    for k, v in kvs:
        assert k.shape == (2, 8, heads, head_dim) and v.shape == (2, 8, heads, head_dim)

    calls = []
    for i, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda m, a, o, i=i: calls.append(i))
    with torch.no_grad():
        kvs_sub = model.forward_ref_kv(ref, torch.zeros(2), context, mask, ref_rope_r=1.0, blocks=[0, 1])
    assert calls == [0, 1]  # the forward stops after the last requested block
    assert kvs_sub[2] is None
    torch.testing.assert_close(kvs_sub[0][0], kvs[0][0])
    torch.testing.assert_close(kvs_sub[1][1], kvs[1][1])


def test_forward_with_ref_kv_effect_and_exclusivity():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs(ref_hw=(2, 4))
    with torch.no_grad():
        kvs = model.forward_ref_kv(ref, torch.zeros(2), context, mask)
        base = model(x, timesteps, context, mask)
        none_kv = model(x, timesteps, context, mask, ref_kv=[None] * len(kvs))
        with_kv = model(x, timesteps, context, mask, ref_kv=kvs)
        partial = model(x, timesteps, context, mask, ref_kv=[kvs[0], None, None])
        skipped = model(x, timesteps, context, mask, ref_kv=kvs, skip_layers=(1,))
    torch.testing.assert_close(none_kv, base)
    assert with_kv.shape == x.shape
    assert not torch.allclose(with_kv, base)
    assert not torch.allclose(partial, with_kv) and not torch.allclose(partial, base)
    assert not torch.allclose(skipped, with_kv)
    with pytest.raises(ValueError):
        model(x, timesteps, context, mask, ref=ref, ref_kv=kvs)
    with pytest.raises(ValueError):
        model(x, timesteps, context, mask, ref_kv=kvs[:2])


@pytest.mark.parametrize("split_attn", [False, True])
def test_batched_forward_with_ref_kv_matches_per_sample(split_attn):
    # the key side has extra tokens: padded batch == per-sample with the true text length (mask / cu_seqlens handling)
    model = make_dit(split_attn=split_attn)
    x, ref, timesteps, context, mask = make_inputs(ref_hw=(2, 4))
    with torch.no_grad():
        kvs = model.forward_ref_kv(ref, torch.zeros(2), context, mask)
        batched = model(x, timesteps, context, mask, ref_kv=kvs)
        for i in range(x.shape[0]):
            l = int(mask[i].sum())
            kv_i = [(k[i : i + 1], v[i : i + 1]) for k, v in kvs]
            kvs_single = model.forward_ref_kv(ref[i : i + 1], torch.zeros(1), context[i : i + 1, :l], None)
            for (k, v), (ks, vs) in zip(kv_i, kvs_single):
                torch.testing.assert_close(k, ks, atol=1e-5, rtol=1e-5)
                torch.testing.assert_close(v, vs, atol=1e-5, rtol=1e-5)
            single = model(x[i : i + 1], timesteps[i : i + 1], context[i : i + 1, :l], None, ref_kv=kv_i)
            torch.testing.assert_close(batched[i : i + 1], single, atol=1e-5, rtol=1e-5)


def test_gradient_checkpointing_with_ref_kv_matches_plain_forward():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs(ref_hw=(2, 4))
    with torch.no_grad():
        kvs = model.forward_ref_kv(ref, torch.zeros(2), context, mask)
        expected = model(x, timesteps, context, mask, ref_kv=kvs)
    model.train()
    model.enable_gradient_checkpointing()
    x_grad = x.clone().requires_grad_(True)
    out = model(x_grad, timesteps, context, mask, ref_kv=kvs)
    out.sum().backward()
    torch.testing.assert_close(out.detach(), expected, atol=1e-5, rtol=1e-5)
    assert x_grad.grad is not None and torch.isfinite(x_grad.grad).all()


def test_compute_reference_kv_and_generate_latents():
    dit = make_dit()
    ref = torch.randn(1, SMALL_CONFIG["latent_channels"], 3, 5)
    uncond = torch.randn(1, 3, SMALL_CONFIG["text_dim"])
    uncond_mask = torch.ones(1, 3, dtype=torch.int32)

    entered = []

    @contextlib.contextmanager
    def lora_off():
        entered.append(True)
        yield

    kvs = moss_image_train_utils.compute_reference_kv(dit, ref, 0.0, uncond, uncond_mask, 1.0, None, lora_off)
    assert entered == [True] and len(kvs) == SMALL_CONFIG["num_layers"]
    assert not kvs[0][0].requires_grad
    kvs_grad = moss_image_train_utils.compute_reference_kv(dit, ref, 0.0, uncond, uncond_mask, 1.0, None, None, enable_grad=True)
    assert kvs_grad[0][0].requires_grad
    torch.testing.assert_close(kvs_grad[0][0].detach(), kvs[0][0])
    # batch 1 embeddings are expanded to the batch of the reference
    kvs2 = moss_image_train_utils.compute_reference_kv(dit, ref.expand(2, -1, -1, -1), 0.0, uncond, uncond_mask)
    assert kvs2[0][0].shape[0] == 2

    embeds = torch.randn(1, 5, SMALL_CONFIG["text_dim"])
    mask = torch.ones(1, 5, dtype=torch.int32)
    common = dict(height=48, width=80, steps=4, flow_shift=5.0, uncond_skip_layers=(1,), seed=42, device=torch.device("cpu"))
    a = moss_image_train_utils.generate_latents(dit, embeds, mask, uncond, uncond_mask, guidance_scale=3.0, show_progress=False, ref_kv=kvs, **common)
    b = moss_image_train_utils.generate_latents(dit, embeds, mask, uncond, uncond_mask, guidance_scale=3.0, show_progress=False, ref_kv=kvs, **common)
    c = moss_image_train_utils.generate_latents(dit, embeds, mask, uncond, uncond_mask, guidance_scale=3.0, show_progress=False, **common)
    assert a.shape == (1, SMALL_CONFIG["latent_channels"], 3, 5)
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)


class FakeNetwork:
    def __init__(self):
        self.multiplier = 0.7
        self.history = []

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        self.history.append(multiplier)


def test_edit_trainer_kv_mode_call_dit():
    import moss_image_train_network_edit

    trainer = moss_image_train_network_edit.MossImageEditNetworkTrainer()
    network = FakeNetwork()
    trainer.post_process_network(make_args(), None, network, None, None)

    dit = make_dit()
    accelerator = FakeAccelerator()
    x, ref, timesteps, context, mask = make_inputs()
    uncond = torch.randn(2, 4, SMALL_CONFIG["text_dim"])
    uncond_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.int32)
    batch = {"text_encoder_outputs_list": [context, mask, uncond, uncond_mask, torch.zeros(2)]}

    # the empty prompt embeddings are taken from the cached outputs (padding dropped) and cached in the trainer
    embeds, emask = trainer.get_uncond_embeds(batch, None, None, None, torch.device("cpu"), torch.float32)
    assert embeds.shape == (1, 3, SMALL_CONFIG["text_dim"]) and emask.shape == (1, 3)
    torch.testing.assert_close(embeds, uncond[:1, :3])
    embeds2, _ = trainer.get_uncond_embeds({}, None, None, None, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(embeds2, embeds)

    # off: the LoRA is disabled during the reference forward and restored, no gradients through the reference
    args = make_args(ref_mode="kv", ref_timestep=0.0, ref_kv_blocks=None, ref_kv_lora="off")
    trainer.ref_kv_blocks = None
    batch = {"ref_latents": ref, "ref_uncond_embeds": embeds, "ref_uncond_mask": emask}
    out = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, network, is_train=True)
    assert out.shape == x.shape
    assert network.history == [0.0, 0.7]

    with torch.no_grad():
        kvs = moss_image_train_utils.compute_reference_kv(dit, ref, 0.0, embeds, emask, 1.0)
        expected = dit(x, timesteps, context, mask, ref_kv=kvs)
    torch.testing.assert_close(out, expected)

    # a subset of blocks
    trainer.ref_kv_blocks = frozenset([0, 2])
    out_sub = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, network, is_train=True)
    assert not torch.allclose(out_sub, out)
    trainer.ref_kv_blocks = None

    # on / on_no_grad: the LoRA stays enabled (the fake network has no modules, so the outputs equal "off"), and only
    # "on" lets the loss gradient reach the weights through the reference forward
    def grad_of_k_proj(mode):
        network.history.clear()
        dit.zero_grad()
        args = make_args(ref_mode="kv", ref_timestep=0.0, ref_kv_blocks=None, ref_kv_lora=mode)
        out_mode = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, network, is_train=True)
        torch.testing.assert_close(out_mode, out)
        assert network.history == []
        out_mode.square().sum().backward()
        return dit.layers[0].proj.k.weight.grad.clone()

    grad_on = grad_of_k_proj("on")
    grad_no_grad = grad_of_k_proj("on_no_grad")
    assert not torch.allclose(grad_on, grad_no_grad)

    # validation (is_train False) never builds a graph
    args = make_args(ref_mode="kv", ref_timestep=0.0, ref_kv_blocks=None, ref_kv_lora="on")
    out_val = trainer.call_dit(args, accelerator, dit, batch, x, timesteps, context, mask, network, is_train=False)
    assert not out_val.requires_grad


# endregion


# region learned reference gate (Step C)

from library import moss_image_ref_gate
from library.moss_image_ref_gate import ReferenceGate, resolve_block_gates


def test_reference_gate_bands_and_table():
    assert moss_image_ref_gate.parse_band_edges(None) == [950.0, 850.0, 700.0, 500.0]
    assert moss_image_ref_gate.parse_band_edges("500, 850") == [850.0, 500.0]
    with pytest.raises(ValueError):
        moss_image_ref_gate.parse_band_edges("1000,500")

    gate = ReferenceGate(3, [950, 850, 700, 500], init_logit=2.0)
    assert gate.num_bands == 5 and gate.trainable
    assert gate.band_ranges() == [(950.0, 1000.0), (850.0, 950.0), (700.0, 850.0), (500.0, 700.0), (0.0, 500.0)]
    t = torch.tensor([1000.0, 951.0, 950.0, 860.0, 700.0, 10.0])
    assert gate.band_index(t).tolist() == [0, 0, 1, 1, 3, 4]
    log_g = gate.log_gates(t)
    assert log_g.shape == (6, 3)
    torch.testing.assert_close(log_g.exp(), torch.full((6, 3), torch.sigmoid(torch.tensor(2.0)).item()))

    # budget penalty: 3 blocks * sigmoid(2) = 2.64 open per band
    torch.testing.assert_close(gate.budget_penalty(2.0), torch.tensor(3 * torch.sigmoid(torch.tensor(2.0)).item() - 2.0))
    assert gate.budget_penalty(3.0) == 0.0
    assert "block" in gate.table_string()

    # binarize: top-k per band, frozen values, state dict keeps the table and the band edges
    with torch.no_grad():
        gate.logits.copy_(torch.tensor([[3.0, -1.0, 0.0, 0.5, 0.2], [1.0, 2.0, 0.0, -0.5, 0.1], [-2.0, 1.0, 5.0, 0.0, 0.3]]))
    assert not gate.is_binarized()
    gate.binarize_(1)
    assert gate.is_binarized()
    assert [gate.open_blocks(b) for b in range(5)] == [[0], [1], [2], [0], [2]]
    assert set(gate.state_dict().keys()) == {"band_edges", "logits"}

    frozen = ReferenceGate(3, [950, 850, 700, 500], trainable=False)
    assert not frozen.trainable and len(list(frozen.parameters())) == 0
    frozen.load_state_dict(gate.state_dict())
    assert frozen.is_binarized() and frozen.open_blocks(2) == [2]


def test_resolve_block_gates():
    kvs = [(torch.zeros(2, 3, 1, 4), torch.zeros(2, 3, 1, 4)) for _ in range(4)]
    kvs[3] = None
    big = moss_image_ref_gate.BINARIZED_LOGIT
    logits = torch.tensor([[big, -big, 1.0, big], [big, -big, -big, big]])  # [B, blocks]
    log_g = torch.nn.functional.logsigmoid(logits)
    out_kv, out_gates = resolve_block_gates(kvs, log_g)
    assert out_kv[0] is kvs[0] and out_gates[0] is None  # all open: plain concatenation
    assert out_kv[1] is None and out_gates[1] is None  # all closed: skipped
    assert out_kv[2] is kvs[2] and out_gates[2] is not None  # mixed: bias path with per-sample log gates
    torch.testing.assert_close(out_gates[2], log_g[:, 2].clamp(min=moss_image_ref_gate.OFF_LOG_GATE))
    assert out_kv[3] is None and out_gates[3] is None
    # no bias path needed at all -> None
    _, gates_none = resolve_block_gates(kvs[:1], log_g[:, :1])
    assert gates_none is None
    # None log gates: pass through
    same_kv, none_gates = resolve_block_gates(kvs, None)
    assert same_kv == kvs and none_gates is None


@pytest.mark.parametrize("split_attn", [False, True])
def test_gated_reference_attention_matches_manual_and_limits(split_attn):
    torch.manual_seed(0)
    batch, img_len, extra, text_len, heads, dim = 2, 6, 5, 7, 2, 8
    lens = [4, 7]
    mask = torch.zeros(batch, text_len, dtype=torch.int32)
    for i, l in enumerate(lens):
        mask[i, :l] = 1
    q = torch.randn(batch, img_len + text_len, heads, dim)
    k = torch.randn(batch, img_len + extra + text_len, heads, dim)
    v = torch.randn(batch, img_len + extra + text_len, heads, dim)
    params = AttentionParams.create_attention_params_from_mask("torch", split_attn, img_len, mask, img_len + extra)
    q_valid = [img_len + l for l in lens]
    k_valid = [img_len + extra + l for l in lens]

    def manual(log_g):
        out = torch.zeros(batch, q.shape[1], heads, dim)
        for i in range(batch):
            qi = q[i, : q_valid[i]].transpose(0, 1)
            ki = k[i, : k_valid[i]].transpose(0, 1)
            vi = v[i, : k_valid[i]].transpose(0, 1)
            logits = qi @ ki.transpose(1, 2) * dim**-0.5
            logits[:, :, img_len : img_len + extra] += log_g[i]
            out[i, : q_valid[i]] = (torch.softmax(logits, -1) @ vi).transpose(0, 1)
        return out.flatten(2)

    def assert_valid_close(actual, expected):
        for i, n in enumerate(q_valid):
            torch.testing.assert_close(actual[i, :n], expected[i, :n], atol=1e-5, rtol=1e-5)

    log_g = torch.tensor([-0.7, -2.3])
    out = moss_image_models.gated_reference_attention(q, k, v, img_len, extra, log_g, params)
    assert out.shape == (batch, q.shape[1], heads * dim)
    assert_valid_close(out, manual(log_g))

    # g = 1: identical to the plain concatenation
    out1 = moss_image_models.gated_reference_attention(q, k, v, img_len, extra, torch.zeros(batch), params)
    assert_valid_close(out1, attention([q, k, v], attn_params=params))

    # g -> 0: identical to the attention without the reference keys
    out0 = moss_image_models.gated_reference_attention(
        q, k, v, img_len, extra, torch.full((batch,), moss_image_ref_gate.OFF_LOG_GATE), params
    )
    params0 = AttentionParams.create_attention_params_from_mask("torch", split_attn, img_len, mask)
    k0 = torch.cat([k[:, :img_len], k[:, img_len + extra :]], 1)
    v0 = torch.cat([v[:, :img_len], v[:, img_len + extra :]], 1)
    assert_valid_close(out0, attention([q, k0, v0], attn_params=params0))

    # differentiable w.r.t. the gate
    log_g_grad = log_g.clone().requires_grad_(True)
    moss_image_models.gated_reference_attention(q, k, v, img_len, extra, log_g_grad, params)[:, :img_len].sum().backward()
    assert log_g_grad.grad is not None and torch.all(log_g_grad.grad != 0)


def test_forward_with_gated_ref_kv():
    model = make_dit()
    x, ref, timesteps, context, mask = make_inputs(ref_hw=(2, 4))
    with torch.no_grad():
        kvs = model.forward_ref_kv(ref, torch.zeros(2), context, mask)
        plain = model(x, timesteps, context, mask, ref_kv=kvs)
        base = model(x, timesteps, context, mask)
    gates_one = [torch.zeros(2) for _ in kvs]
    gates_off = [torch.full((2,), moss_image_ref_gate.OFF_LOG_GATE) for _ in kvs]
    with torch.no_grad():
        out_one = model(x, timesteps, context, mask, ref_kv=kvs, ref_gate=gates_one)
        out_off = model(x, timesteps, context, mask, ref_kv=kvs, ref_gate=gates_off)
        out_mixed = model(x, timesteps, context, mask, ref_kv=kvs, ref_gate=[torch.tensor([0.0, -1.0])] + [None] * (len(kvs) - 1))
    torch.testing.assert_close(out_one, plain, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_off, base, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(out_mixed, plain)
    with pytest.raises(ValueError):
        model(x, timesteps, context, mask, ref_kv=kvs, ref_gate=gates_one[:1])

    # the table drives the per-block gates; its logits get gradients through the bias path
    gate = ReferenceGate(len(kvs), [950, 850, 700, 500], init_logit=1.0)
    kv_r, g_r = resolve_block_gates(kvs, gate.log_gates(timesteps))
    out = model(x, timesteps, context, mask, ref_kv=kv_r, ref_gate=g_r)
    out.square().sum().backward()
    assert gate.logits.grad is not None
    bands = gate.band_index(timesteps).tolist()
    assert torch.all(gate.logits.grad[:, bands] != 0)
    other = [b for b in range(gate.num_bands) if b not in bands]
    assert torch.all(gate.logits.grad[:, other] == 0)


def test_network_extra_trainable_module_saved_and_loaded(tmp_path):
    from networks import lora_generic

    dit = make_dit()
    network = lora_generic.create_network(1.0, 4, 1.0, None, [None], dit)
    gate = ReferenceGate(3, [950, 850, 700, 500], init_logit=2.0)
    network.add_extra_trainable_module("ref_gate", gate, 1e-2)
    network.apply_to([None], dit, False, True)

    params, descriptions = network.prepare_optimizer_params_with_multiple_te_lrs(None, 1e-4, 1e-4)
    assert descriptions[-1] == "ref_gate" and params[-1]["lr"] == 1e-2
    assert any(p is gate.logits for p in params[-1]["params"])
    assert any(p is gate.logits for p in network.get_trainable_params())

    with torch.no_grad():
        gate.logits.fill_(0.5)
    path = str(tmp_path / "lora.safetensors")
    network.save_weights(path, torch.float32, {})
    from safetensors.torch import load_file

    sd = load_file(path)
    assert "ref_gate.logits" in sd and "ref_gate.band_edges" in sd
    assert sd["ref_gate.band_edges"].tolist() == [950.0, 850.0, 700.0, 500.0]

    network2 = lora_generic.create_network(1.0, 4, 1.0, None, [None], make_dit())
    gate2 = ReferenceGate(3, [950, 850, 700, 500], init_logit=0.0, trainable=False)
    network2.add_extra_trainable_module("ref_gate", gate2, None)
    network2.load_weights(path)
    torch.testing.assert_close(gate2.logits, torch.full((3, 5), 0.5))
    # the frozen table adds no optimizer group
    _, descriptions2 = network2.prepare_optimizer_params_with_multiple_te_lrs(None, 1e-4, 1e-4)
    assert "ref_gate" not in descriptions2


def test_edit_trainer_balanced_timesteps():
    import moss_image_train_network_edit
    from library import sd3_train_utils

    trainer = moss_image_train_network_edit.MossImageEditNetworkTrainer()
    trainer.ref_gate = ReferenceGate(SMALL_CONFIG["num_layers"], [950, 850, 700, 500])
    dit = make_dit()
    accelerator = FakeAccelerator()
    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    args = make_args(ref_mode="kv", ref_gate=True, ref_gate_balanced_timesteps=True)

    torch.manual_seed(0)
    bsz = 64
    latents = torch.randn(bsz, SMALL_CONFIG["latent_channels"], 3, 5)
    ref = torch.randn(bsz, SMALL_CONFIG["latent_channels"], 3, 5)
    context = torch.randn(bsz, 5, SMALL_CONFIG["text_dim"])
    mask = torch.ones(bsz, 5, dtype=torch.int32)
    uncond = torch.randn(1, 3, SMALL_CONFIG["text_dim"])
    uncond_mask = torch.ones(1, 3, dtype=torch.int32)
    batch = {"ref_latents": ref, "ref_uncond_embeds": uncond, "ref_uncond_mask": uncond_mask}

    model_pred, target, timesteps, weighting = trainer.get_noise_pred_and_target(
        args, accelerator, noise_scheduler, latents, batch, [context, mask, torch.zeros(bsz)], dit, None, torch.float32, True, is_train=True
    )
    assert model_pred.shape == latents.shape and timesteps.shape == (bsz,) and timesteps.dtype == torch.float32
    assert torch.all(timesteps > 0) and torch.all(timesteps <= 1000)
    bands = trainer.ref_gate.band_index(timesteps)
    assert set(bands.tolist()) == set(range(5))  # every band is drawn (64 draws over 5 bands)
    # v = noise - x0 target: the noisy input follows (1 - sigma) x0 + sigma noise
    noise = target + latents
    assert torch.all(weighting == 1.0)

    # validation keeps the normal sampling
    out_val = trainer.get_noise_pred_and_target(
        args, accelerator, noise_scheduler, latents[:2], {k: v[:2] if k == "ref_latents" else v for k, v in batch.items()},
        [context[:2], mask[:2], torch.zeros(2)], dit, None, torch.float32, True, is_train=False,
    )
    assert out_val[0].shape == latents[:2].shape


def test_generate_latents_with_gate_table():
    dit = make_dit()
    ref = torch.randn(1, SMALL_CONFIG["latent_channels"], 3, 5)
    uncond = torch.randn(1, 3, SMALL_CONFIG["text_dim"])
    uncond_mask = torch.ones(1, 3, dtype=torch.int32)
    kvs = moss_image_train_utils.compute_reference_kv(dit, ref, 0.0, uncond, uncond_mask)
    embeds = torch.randn(1, 5, SMALL_CONFIG["text_dim"])
    mask = torch.ones(1, 5, dtype=torch.int32)
    common = dict(height=48, width=80, steps=4, flow_shift=5.0, uncond_skip_layers=(1,), seed=42, device=torch.device("cpu"))

    plain = moss_image_train_utils.generate_latents(dit, embeds, mask, uncond, uncond_mask, guidance_scale=3.0, show_progress=False, ref_kv=kvs, **common)
    gate = ReferenceGate(SMALL_CONFIG["num_layers"], [950, 850, 700, 500], init_logit=moss_image_ref_gate.BINARIZED_LOGIT, trainable=False)
    all_open = moss_image_train_utils.generate_latents(
        dit, embeds, mask, uncond, uncond_mask, guidance_scale=3.0, show_progress=False, ref_kv=kvs, ref_gate=gate, **common
    )
    torch.testing.assert_close(all_open, plain)
    with torch.no_grad():
        gate.logits.fill_(0.0)  # continuous g = 0.5 everywhere: bias path
    half = moss_image_train_utils.generate_latents(
        dit, embeds, mask, uncond, uncond_mask, guidance_scale=3.0, show_progress=False, ref_kv=kvs, ref_gate=gate, **common
    )
    assert not torch.allclose(half, plain)


# endregion


# region replica training (--replica_devices)


def test_edit_trainer_replica_wrapping_and_parser():
    import moss_image_train_network_edit
    from library import replica_trainer

    parser = moss_image_train_network_edit.setup_parser()
    assert parser.parse_args([]).replica_devices is None  # the replica arguments come with the MOSS-Image parser

    cls = replica_trainer.wrap_trainer_class(moss_image_train_network_edit.MossImageEditNetworkTrainer, make_args(replica_devices="1"))
    assert cls.__mro__[0] is cls and cls.__mro__[1] is replica_trainer.ReplicaTrainerMixin
    assert cls.__mro__[2] is moss_image_train_network_edit.MossImageEditNetworkTrainer
    plain = replica_trainer.wrap_trainer_class(moss_image_train_network_edit.MossImageEditNetworkTrainer, make_args(replica_devices=None))
    assert plain is moss_image_train_network_edit.MossImageEditNetworkTrainer


def test_edit_trainer_gate_per_network_and_frozen_table_sync():
    """Every network created by the trainer (the master's and the replicas') carries its own gate table; call_dit uses the
    table of the network of the micro-step; the binarized (frozen) table of the master is copied to the other tables once."""
    import moss_image_train_network_edit
    from networks import lora_generic

    net_args = dict(dim_from_weights=False, network_dim=4, network_alpha=1.0, network_dropout=None, network_args=None)
    accelerator = FakeAccelerator()
    dits = [make_dit(), make_dit()]

    # first stage: trainable tables
    trainer = moss_image_train_network_edit.MossImageEditNetworkTrainer()
    args = make_args(ref_mode="kv", ref_gate=True, ref_gate_budget=1, **net_args)
    networks = [trainer.create_network(args, lora_generic, None, [None], d, {}) for d in dits]
    assert trainer._ref_gates == [n.ref_gate for n in networks]
    assert all(g.trainable for g in trainer._ref_gates)
    for n in networks:
        assert any(p is n.ref_gate.logits for p in n.get_trainable_params())  # synchronized like the LoRA weights
    trainer.post_process_network(args, accelerator, networks[0], [None], dits[0])
    assert trainer.ref_gate is networks[0].ref_gate
    assert trainer.ref_gate_of(accelerator, networks[1]) is networks[1].ref_gate
    assert trainer.ref_gate_of(accelerator, None) is networks[0].ref_gate  # no network given: the master's table

    # call_dit uses the table of the given network, not the master's
    x, ref, timesteps, context, mask = make_inputs()
    uncond = torch.randn(1, 3, SMALL_CONFIG["text_dim"])
    batch = {"ref_latents": ref, "ref_uncond_embeds": uncond, "ref_uncond_mask": torch.ones(1, 3, dtype=torch.int32)}

    def run(network):
        return trainer.call_dit(args, accelerator, dits[1], batch, x, timesteps, context, mask, network, is_train=False)

    base = run(networks[1])
    with torch.no_grad():
        networks[0].ref_gate.logits.fill_(-20.0)  # closing the master's table does not change the replica's micro-step
    torch.testing.assert_close(run(networks[1]), base)
    with torch.no_grad():
        networks[1].ref_gate.logits.fill_(-20.0)
    assert not torch.allclose(run(networks[1]), base)

    # without --ref_gate no table is attached
    plain = moss_image_train_network_edit.MossImageEditNetworkTrainer()
    network = plain.create_network(make_args(**net_args), lora_generic, None, [None], make_dit(), {})
    assert not hasattr(network, "ref_gate") and plain._ref_gates == [] and plain.ref_gate_of(accelerator, network) is None

    # second stage: frozen tables; the master's table (loaded from --network_weights) is binarized once and copied
    trainer2 = moss_image_train_network_edit.MossImageEditNetworkTrainer()
    args2 = make_args(ref_mode="kv", ref_gate=True, ref_gate_binarize=True, ref_gate_budget=1, **net_args)
    networks2 = [trainer2.create_network(args2, lora_generic, None, [None], d, {}) for d in dits]
    assert not any(g.trainable for g in trainer2._ref_gates)
    trainer2.post_process_network(args2, accelerator, networks2[0], [None], dits[0])
    with torch.no_grad():
        trainer2.ref_gate.logits.copy_(torch.randn_like(trainer2.ref_gate.logits))  # as loaded by network.load_weights
    assert not torch.equal(networks2[1].ref_gate.logits, trainer2.ref_gate.logits)
    trainer2.prepare_ref_gate_after_load(args2)
    assert trainer2.ref_gate.is_binarized()
    torch.testing.assert_close(networks2[1].ref_gate.logits, trainer2.ref_gate.logits)
    with torch.no_grad():
        networks2[1].ref_gate.logits.zero_()
    trainer2.prepare_ref_gate_after_load(args2)  # done once: later calls do not touch the tables
    assert torch.all(networks2[1].ref_gate.logits == 0)


# endregion
