import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from finetune.merge_captions_to_metadata import main as merge_captions_to_metadata
from library.finetuning_dataset import FineTuningDataset
from library.strategy_base import LatentsCachingStrategy
from library.strategy_sd import SdSdxlLatentsCachingStrategy
from library.subset import FineTuningSubset


class _Tokenizer:
    def tokenize(self, caption):
        return [torch.tensor([[1, 2, 3]])]


@pytest.fixture
def latents_strategy(request, monkeypatch):
    family = getattr(request, "param", "sdxl")
    options = {"cache_to_disk": True, "batch_size": 1, "skip_disk_cache_validity_check": False}
    if family in ("sd", "sdxl"):
        strategy = SdSdxlLatentsCachingStrategy(sd=family == "sd", **options)
    else:
        class_name = {
            "sd3": "Sd3LatentsCachingStrategy",
            "flux": "FluxLatentsCachingStrategy",
            "lumina": "LuminaLatentsCachingStrategy",
            "hunyuan_image": "HunyuanImageLatentsCachingStrategy",
            "anima": "AnimaLatentsCachingStrategy",
        }[family]
        module = importlib.import_module(f"library.strategy_{family}")
        strategy = getattr(module, class_name)(**options)
    monkeypatch.setattr(LatentsCachingStrategy, "_strategy", strategy)
    return strategy


def _make_dataset(directory, metadata, **subset_options):
    metadata_file = directory / "metadata.json"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")
    options = {
        "image_dir": str(directory),
        "metadata_file": str(metadata_file),
        "alpha_mask": False,
        "num_repeats": 1,
        "shuffle_caption": False,
        "caption_separator": ",",
        "keep_tokens": 0,
        "keep_tokens_separator": None,
        "secondary_separator": None,
        "enable_wildcard": False,
        "color_aug": False,
        "flip_aug": False,
        "face_crop_aug_range": None,
        "random_crop": False,
        "caption_dropout_rate": 0,
        "caption_dropout_every_n_epochs": 0,
        "caption_tag_dropout_rate": 0,
        "caption_prefix": None,
        "caption_suffix": None,
        "token_warmup_min": 1,
        "token_warmup_step": 0,
    }
    options.update(subset_options)
    dataset = FineTuningDataset(
        [FineTuningSubset(**options)],
        batch_size=1,
        resolution=(64, 64),
        network_multiplier=1.0,
        enable_bucket=False,
        min_bucket_reso=64,
        max_bucket_reso=64,
        bucket_reso_steps=64,
        bucket_no_upscale=False,
        train_inpainting=False,
        debug_dataset=False,
        validation_seed=0,
        validation_split=0,
        resize_interpolation=None,
    )
    dataset.make_buckets()
    dataset.set_current_strategies()
    dataset.tokenize_strategy = _Tokenizer()
    return dataset


def _write_cache(strategy, image_path, value, *, legacy=False):
    if legacy:
        cache_path = str(image_path.with_suffix(".npz"))
    else:
        cache_path = strategy.get_latents_npz_path(str(image_path), (64, 64))
    if strategy.cache_suffix == "_hi.npz":
        shape = (64, 2, 2)
    elif strategy.cache_suffix in ("_sd.npz", "_sdxl.npz"):
        shape = (4, 8, 8)
    else:
        shape = (16, 8, 8)
    latents = torch.full(shape, float(value))
    strategy.save_latents_to_disk(
        cache_path,
        latents,
        (64, 64),
        (0, 0, 64, 64),
        key_reso_suffix="" if legacy else f"_{shape[-2]}x{shape[-1]}",
    )
    return cache_path


@pytest.mark.parametrize(
    "latents_strategy", ["sd", "sdxl", "sd3", "flux", "lumina", "hunyuan_image", "anima"], indirect=True
)
def test_cached_latents_remain_paired_with_metadata_caption(tmp_path, latents_strategy):
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), "red").save(image_path)
    own_cache = _write_cache(latents_strategy, image_path, 11)
    sibling_path = tmp_path / "photo_crop.png"
    Image.new("RGB", (64, 64), "blue").save(sibling_path)
    _write_cache(latents_strategy, sibling_path, 99)

    # Selecting a subset through metadata must not pick a longer sibling's cache.
    dataset = _make_dataset(tmp_path, {"photo.png": {"caption": "a red photograph"}})
    example = dataset[0]

    assert example["captions"] == ["a red photograph"]
    assert torch.all(example["latents"] == 11)
    assert example["images"] is None
    assert dataset.image_data["photo.png"].latents_npz == own_cache


def test_unrelated_cache_does_not_replace_uncached_image(tmp_path, latents_strategy):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "photo.png")
    _write_cache(latents_strategy, tmp_path / "photo_crop.png", 99)

    dataset = _make_dataset(tmp_path, {"photo.png": {"caption": "a red photograph"}})
    example = dataset[0]

    assert example["latents"] is None
    assert example["captions"] == ["a red photograph"]
    torch.testing.assert_close(example["images"][0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
    assert dataset.image_data["photo.png"].latents_npz is None


def test_sibling_with_metadata_size_does_not_donate_its_cache(tmp_path, latents_strategy):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "photo.png")
    Image.new("RGB", (64, 64), "blue").save(tmp_path / "photo_crop.png")
    own_cache = _write_cache(latents_strategy, tmp_path / "photo.png", 11)
    _write_cache(latents_strategy, tmp_path / "photo_crop.png", 99)
    dataset = _make_dataset(
        tmp_path,
        {
            "photo.png": {"caption": "red"},
            "photo_crop.png": {"caption": "blue", "image_size": [64, 64]},
        },
    )
    examples = {example["captions"][0]: example for example in dataset}

    assert torch.all(examples["red"]["latents"] == 11)
    assert dataset.image_data["photo.png"].latents_npz == own_cache
    assert dataset.image_data["photo_crop.png"].latents_npz is None
    assert examples["blue"]["latents"] is None


@pytest.mark.parametrize(
    "filename,metadata_key",
    [
        ("photo.png", "photo"),
        ("photo.v2.png", "photo.v2.png"),
        ("photo.v2.png", "photo.v2"),
        ("photo_crop.png", "photo_crop.png"),
        ("photo_0064x0064.png", "photo_0064x0064.png"),
        ("photo.png", "nested/../photo.png"),
    ],
)
def test_matching_cache_accepts_resolved_metadata_paths(tmp_path, latents_strategy, filename, metadata_key):
    (tmp_path / "nested").mkdir()
    image_path = tmp_path / filename
    Image.new("RGB", (64, 64), "red").save(image_path)
    own_cache = _write_cache(latents_strategy, image_path, 11)
    dataset = _make_dataset(tmp_path, {metadata_key: {"caption": "selected"}})

    assert Path(dataset.image_data[metadata_key].latents_npz) == Path(own_cache)
    assert torch.all(dataset[0]["latents"] == 11)


def test_absolute_metadata_path_without_image_dir(tmp_path, latents_strategy):
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), "red").save(image_path)
    own_cache = _write_cache(latents_strategy, image_path, 11)
    dataset = _make_dataset(tmp_path, {str(image_path): {"caption": "selected"}}, image_dir=None)

    assert dataset.image_data[str(image_path)].latents_npz == own_cache
    assert torch.all(dataset[0]["latents"] == 11)


@pytest.mark.parametrize(
    "filename,metadata_key", [("photo.png", "photo.png"), ("photo.png", "photo"), ("photo.v2.png", "photo.v2")]
)
def test_matching_cache_works_without_source_image(tmp_path, latents_strategy, filename, metadata_key):
    own_cache = _write_cache(latents_strategy, tmp_path / filename, 11)
    dataset = _make_dataset(tmp_path, {metadata_key: {"caption": "selected"}})

    assert dataset.image_data[metadata_key].latents_npz == own_cache
    assert torch.all(dataset[0]["latents"] == 11)


@pytest.mark.parametrize("full_path", [False, True])
def test_generated_metadata_reuses_cache_when_stem_ends_in_image_extension(tmp_path, latents_strategy, full_path):
    image_path = tmp_path / "photo.png.webp"
    Image.new("RGB", (64, 64), "red").save(image_path)
    image_path.with_suffix(".caption").write_text("a red photograph", encoding="utf-8")
    metadata_file = tmp_path / "generated.json"
    merge_captions_to_metadata(
        SimpleNamespace(
            train_data_dir=str(tmp_path),
            out_json=str(metadata_file),
            in_json=None,
            recursive=False,
            full_path=full_path,
            caption_extension=".caption",
            debug=False,
        )
    )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    image_key = str(image_path) if full_path else "photo.png"
    assert list(metadata) == [image_key]
    own_cache = _write_cache(latents_strategy, image_path, 11)
    image_path.unlink()

    dataset = _make_dataset(tmp_path, metadata)

    assert dataset.image_data[image_key].latents_npz == own_cache
    assert dataset[0]["captions"] == ["a red photograph"]
    assert torch.all(dataset[0]["latents"] == 11)


@pytest.mark.parametrize("include_explicit_sibling", [False, True])
def test_cache_only_metadata_rejects_two_different_matching_stems(tmp_path, latents_strategy, include_explicit_sibling):
    _write_cache(latents_strategy, tmp_path / "photo.png", 11)
    _write_cache(latents_strategy, tmp_path / "photo.png.webp", 99)
    metadata = {"photo.png": {"caption": "ambiguous"}}
    if include_explicit_sibling:
        # Consuming a cache for an earlier entry must not hide the ambiguity.
        metadata["photo.png.webp"] = {"caption": "explicit sibling"}

    with pytest.raises(ValueError, match="Ambiguous latent caches"):
        _make_dataset(tmp_path, metadata)


def test_existing_image_disambiguates_cache_only_stem_alternative(tmp_path, latents_strategy):
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), "red").save(image_path)
    own_cache = _write_cache(latents_strategy, image_path, 11)
    _write_cache(latents_strategy, tmp_path / "photo.png.webp", 99)

    dataset = _make_dataset(tmp_path, {"photo.png": {"caption": "selected"}})

    assert dataset.image_data["photo.png"].latents_npz == own_cache
    assert torch.all(dataset[0]["latents"] == 11)


def test_multiple_cache_resolutions_of_one_stem_are_not_ambiguous(tmp_path, latents_strategy):
    image_path = tmp_path / "photo.png"
    own_cache = _write_cache(latents_strategy, image_path, 11)
    second_cache = latents_strategy.get_latents_npz_path(str(image_path), (128, 64))
    latents_strategy.save_latents_to_disk(
        second_cache, torch.full((4, 8, 8), 22.0), (128, 64), (32, 0, 96, 64), key_reso_suffix="_8x8"
    )

    dataset = _make_dataset(tmp_path, {"photo.png": {"caption": "selected"}})

    selected_cache = dataset.image_data["photo.png"].latents_npz
    assert selected_cache in (own_cache, second_cache)
    assert torch.all(dataset[0]["latents"] == (11 if selected_cache == own_cache else 22))


@pytest.mark.parametrize("latents_strategy", ["sd", "sdxl"], indirect=True)
def test_legacy_cache_remains_available_with_prefix_sharing_modern_cache(tmp_path, latents_strategy):
    image_path = tmp_path / "photo.png"
    Image.new("RGB", (64, 64), "red").save(image_path)
    legacy_cache = _write_cache(latents_strategy, image_path, 11, legacy=True)
    _write_cache(latents_strategy, tmp_path / "photo_crop.png", 99)
    dataset = _make_dataset(tmp_path, {"photo.png": {"caption": "selected"}})

    assert dataset.image_data["photo.png"].latents_npz is None
    # A valid legacy cache is reused without needing a VAE or accelerator device.
    dataset.new_cache_latents(None, SimpleNamespace(num_processes=1, process_index=0))

    assert dataset.image_data["photo.png"].latents_npz == legacy_cache
    assert torch.all(dataset[0]["latents"] == 11)


def test_dataset_without_caching_strategy_loads_source_image(tmp_path, latents_strategy, monkeypatch):
    Image.new("RGB", (64, 64), "red").save(tmp_path / "photo.png")
    _write_cache(latents_strategy, tmp_path / "photo.png", 11)
    monkeypatch.setattr(LatentsCachingStrategy, "_strategy", None)
    dataset = _make_dataset(tmp_path, {"photo.png": {"caption": "selected"}})

    assert dataset.image_data["photo.png"].latents_npz is None
    assert dataset[0]["latents"] is None
    torch.testing.assert_close(dataset[0]["images"][0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
