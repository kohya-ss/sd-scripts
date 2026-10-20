"""Caption processing when caching Text Encoder outputs (issue #2453).

The deterministic part of ``process_caption`` (prefix/suffix, secondary separator, replacements, first-line
truncation) must be applied to the captions passed to ``cache_batch_outputs``.
"""

import logging
import random
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from library.dataset import BaseDataset, BucketBatchIndex, ImageInfo
from library.strategy_base import TextEncoderOutputsCachingStrategy, TextEncodingStrategy, TokenizeStrategy
from library.subset import DEFAULT_MULTILINE_CAPTION_SEPARATOR

SEP = DEFAULT_MULTILINE_CAPTION_SEPARATOR


def make_subset(**overrides):
    params = dict(
        enable_multiline_caption=False,
        multiline_caption_separator=SEP,
        enable_wildcard=False,
        caption_prefix=None,
        caption_suffix=None,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        shuffle_caption=False,
        token_warmup_min=1,
        token_warmup_step=0,
        keep_tokens=0,
        keep_tokens_separator="",
        caption_separator=",",
        secondary_separator=None,
    )
    params.update(overrides)
    return SimpleNamespace(**params)


def make_dataset():
    ds = SimpleNamespace(
        current_epoch=0, current_step=0, max_train_steps=100, replacements={}, has_multiple_caption_candidates=False
    )
    # process_caption_for_cache delegates to this method
    ds.process_caption_candidates_for_cache = lambda subset, caption: BaseDataset.process_caption_candidates_for_cache(
        ds, subset, caption
    )
    return ds


def for_cache(subset, caption, replacements=None):
    ds = make_dataset()
    if replacements:
        ds.replacements = replacements
    return BaseDataset.process_caption_for_cache(ds, subset, caption)


# ---------------------------------------------------------------------------
# process_caption_for_cache
# ---------------------------------------------------------------------------


def test_for_cache_applies_prefix_suffix_and_first_line():
    subset = make_subset(caption_prefix="pre", caption_suffix="suf")
    assert for_cache(subset, "line1\nline2") == "pre line1 suf"


def test_for_cache_applies_secondary_separator_and_replacements():
    subset = make_subset(secondary_separator=";;;")
    assert for_cache(subset, "a;;;b, girl", replacements={"girl": "woman"}) == "a,b, woman"


def test_for_cache_is_deterministic_and_ignores_random_options():
    # dropout / shuffle / tag dropout are not applied (they are rejected by is_text_encoder_output_cacheable anyway)
    subset = make_subset(caption_dropout_rate=1.0, shuffle_caption=True, caption_tag_dropout_rate=1.0, caption_prefix="pre")
    assert for_cache(subset, "a, b, c") == "pre a, b, c"


def test_for_cache_multiline_single_candidate_is_normalized():
    subset = make_subset(enable_multiline_caption=True, caption_suffix="suf")
    assert for_cache(subset, "  a  \nb\n\n") == "  a\nb suf"


def test_for_cache_wildcard_caption_is_kept_as_is():
    subset = make_subset(enable_wildcard=True, caption_prefix="pre")
    assert for_cache(subset, "a {x|y}\nb") == "pre a {x|y}\nb"


def test_for_cache_matches_process_caption_for_plain_captions():
    subset = make_subset(caption_prefix="pre", caption_suffix="suf", secondary_separator=";;;")
    ds = make_dataset()
    caption = "a;;;b, c"
    assert BaseDataset.process_caption_for_cache(ds, subset, caption) == BaseDataset.process_caption(ds, subset, caption)


# ---------------------------------------------------------------------------
# is_text_encoder_output_cacheable: wildcard warning
# ---------------------------------------------------------------------------


def test_wildcard_warns_but_is_cacheable(caplog):
    ds = make_dataset()
    ds.subsets = [make_subset(enable_wildcard=True)]
    with caplog.at_level(logging.WARNING, logger="library.dataset"):
        assert BaseDataset.is_text_encoder_output_cacheable(ds) is True
    assert any("enable_wildcard" in r.getMessage() for r in caplog.records)


def test_no_wildcard_no_warning(caplog):
    ds = make_dataset()
    ds.subsets = [make_subset()]
    with caplog.at_level(logging.WARNING, logger="library.dataset"):
        assert BaseDataset.is_text_encoder_output_cacheable(ds) is True
    assert not any("enable_wildcard" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# new_cache_text_encoder_outputs passes processed captions to the caching strategy
# ---------------------------------------------------------------------------


class RecordingCachingStrategy(TextEncoderOutputsCachingStrategy):
    def __init__(self, batch_size):
        super().__init__(cache_to_disk=False, batch_size=batch_size, skip_disk_cache_validity_check=False)
        self.calls = []

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch, captions):
        self.calls.append(([info.image_key for info in batch], list(captions)))


def test_new_cache_text_encoder_outputs_uses_processed_captions():
    ds = BaseDataset((64, 64), 1.0, False, False)
    ds.batch_size = 2
    subset_a = make_subset(caption_prefix="pre", secondary_separator=";;;")
    subset_b = make_subset(enable_multiline_caption=True, caption_suffix="suf")
    ds.subsets = [subset_a, subset_b]

    infos = [
        (ImageInfo("img0", 1, "a;;;b\nignored", False, "img0.png"), subset_a),
        (ImageInfo("img1", 1, "c", False, "img1.png"), subset_a),
        (ImageInfo("img2", 1, "x\ny\n", False, "img2.png"), subset_b),
    ]
    for info, subset in infos:
        ds.register_image(info, subset)

    caching = RecordingCachingStrategy(batch_size=2)
    accelerator = SimpleNamespace(num_processes=1, process_index=0)
    with patch.object(TokenizeStrategy, "get_strategy", return_value=object()), patch.object(
        TextEncodingStrategy, "get_strategy", return_value=object()
    ), patch.object(TextEncoderOutputsCachingStrategy, "get_strategy", return_value=caching):
        BaseDataset.new_cache_text_encoder_outputs(ds, models=[], accelerator=accelerator)

    assert caching.calls == [
        (["img0", "img1"], ["pre a,b", "pre c"]),
        (["img2"], ["x\ny suf"]),
    ]
    # the raw captions are untouched
    assert [info.caption for info, _ in infos] == ["a;;;b\nignored", "c", "x\ny\n"]


# ---------------------------------------------------------------------------
# multiple caption candidates: process_caption_candidates_for_cache
# ---------------------------------------------------------------------------


def candidates_for_cache(subset, caption):
    return BaseDataset.process_caption_candidates_for_cache(make_dataset(), subset, caption)


def test_candidates_for_cache_multiline_returns_all_candidates_with_prefix_suffix():
    subset = make_subset(enable_multiline_caption=True, caption_prefix="pre", caption_suffix="suf")
    caption = f"a\nb\n{SEP} # comment\nc\n{SEP}\n\n"
    assert candidates_for_cache(subset, caption) == ["pre a\nb suf", "pre c suf"]


def test_candidates_for_cache_single_candidate_modes():
    assert candidates_for_cache(make_subset(), "a\nb") == ["a"]
    assert candidates_for_cache(make_subset(enable_wildcard=True), "a {x|y}\nb") == ["a {x|y}\nb"]
    assert candidates_for_cache(make_subset(enable_multiline_caption=True), "a\nb") == ["a\nb"]
    # process_caption_for_cache is the first candidate
    subset = make_subset(enable_multiline_caption=True)
    assert for_cache(subset, f"a\n{SEP}\nb") == "a"


def test_cacheable_with_multiple_candidates_only_if_supported(caplog):
    ds = make_dataset()
    ds.subsets = [make_subset(enable_multiline_caption=True)]
    ds.has_multiple_caption_candidates = True
    with caplog.at_level(logging.INFO, logger="library.dataset"):
        assert BaseDataset.is_text_encoder_output_cacheable(ds) is False
        assert BaseDataset.is_text_encoder_output_cacheable(ds, supports_multiple_caption_candidates=True) is True
    assert any("multiple candidates" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    assert any("all candidates" in r.getMessage() and r.levelno == logging.INFO for r in caplog.records)
    # the other restrictions still apply
    ds.subsets = [make_subset(enable_multiline_caption=True, shuffle_caption=True)]
    assert BaseDataset.is_text_encoder_output_cacheable(ds, supports_multiple_caption_candidates=True) is False


# ---------------------------------------------------------------------------
# new_cache_text_encoder_outputs with multiple candidates
# ---------------------------------------------------------------------------


class RecordingMultiCachingStrategy(TextEncoderOutputsCachingStrategy):
    def __init__(self, batch_size, cache_to_disk=False, valid_npz=()):
        super().__init__(cache_to_disk=cache_to_disk, batch_size=batch_size, skip_disk_cache_validity_check=False)
        self.calls = []
        self.checked = []
        self.valid_npz = set(valid_npz)

    @property
    def supports_multiple_caption_candidates(self):
        return True

    def get_outputs_npz_path(self, image_abs_path):
        return image_abs_path + "_te.npz"

    def is_disk_cached_outputs_expected(self, npz_path, captions=None):
        self.checked.append((npz_path, captions))
        return npz_path in self.valid_npz

    def cache_batch_outputs(self, tokenize_strategy, models, text_encoding_strategy, batch, captions):
        raise AssertionError("cache_batch_outputs must not be called for a strategy with multiple candidate support")

    def cache_batch_outputs_multi(self, tokenize_strategy, models, text_encoding_strategy, batch, captions_list):
        self.calls.append(([info.image_key for info in batch], [list(c) for c in captions_list]))


def make_multi_candidate_dataset():
    ds = BaseDataset((64, 64), 1.0, False, False)
    ds.batch_size = 3
    subset = make_subset(enable_multiline_caption=True, caption_prefix="pre")
    ds.subsets = [subset]
    infos = [
        ImageInfo("img0", 1, f"a\n{SEP}\nb\n{SEP}\nc", False, "img0.png"),  # 3 candidates
        ImageInfo("img1", 1, "d", False, "img1.png"),  # 1 candidate
        ImageInfo("img2", 1, f"e\n{SEP}\nf", False, "img2.png"),  # 2 candidates
        ImageInfo("img3", 1, "g", False, "img3.png"),  # 1 candidate
    ]
    for info in infos:
        ds.register_image(info, subset)
    ds.has_multiple_caption_candidates = True
    return ds, infos


def run_cache(ds, caching):
    accelerator = SimpleNamespace(num_processes=1, process_index=0)
    with patch.object(TokenizeStrategy, "get_strategy", return_value=object()), patch.object(
        TextEncodingStrategy, "get_strategy", return_value=object()
    ), patch.object(TextEncoderOutputsCachingStrategy, "get_strategy", return_value=caching):
        BaseDataset.new_cache_text_encoder_outputs(ds, models=[], accelerator=accelerator)


def test_new_cache_text_encoder_outputs_batches_by_number_of_candidates():
    ds, _ = make_multi_candidate_dataset()
    caching = RecordingMultiCachingStrategy(batch_size=3)
    run_cache(ds, caching)
    # the batch is flushed when the number of candidates reaches batch_size (3): img0 alone, img1+img2, img3
    assert caching.calls == [
        (["img0"], [["pre a", "pre b", "pre c"]]),
        (["img1", "img2"], [["pre d"], ["pre e", "pre f"]]),
        (["img3"], [["pre g"]]),
    ]


def test_new_cache_text_encoder_outputs_checks_disk_cache_with_candidates():
    ds, infos = make_multi_candidate_dataset()
    caching = RecordingMultiCachingStrategy(batch_size=10, cache_to_disk=True, valid_npz={"img1.png_te.npz"})
    run_cache(ds, caching)
    assert caching.checked == [
        ("img0.png_te.npz", ["pre a", "pre b", "pre c"]),
        ("img1.png_te.npz", ["pre d"]),
        ("img2.png_te.npz", ["pre e", "pre f"]),
        ("img3.png_te.npz", ["pre g"]),
    ]
    assert caching.calls == [(["img0", "img2", "img3"], [["pre a", "pre b", "pre c"], ["pre e", "pre f"], ["pre g"]])]
    assert [info.text_encoder_outputs_npz for info in infos] == [f"img{i}.png_te.npz" for i in range(4)]


def test_new_cache_text_encoder_outputs_rejects_multiple_candidates_without_support():
    ds, _ = make_multi_candidate_dataset()
    caching = RecordingCachingStrategy(batch_size=3)  # no multiple candidate support
    with pytest.raises(ValueError, match="multiple candidates"):
        run_cache(ds, caching)
    assert caching.calls == []


# ---------------------------------------------------------------------------
# __getitem__ chooses one candidate and returns the matching caption
# ---------------------------------------------------------------------------


class IndexRecordingStrategy(RecordingMultiCachingStrategy):
    def __init__(self):
        super().__init__(batch_size=1, cache_to_disk=True)
        self.loaded = []

    def load_outputs_npz(self, npz_path, candidate_index=0):
        self.loaded.append((npz_path, candidate_index))
        # outputs of candidate k: embeds filled with k, mask of ones, rate
        return [
            np.full((2, 4), float(candidate_index), dtype=np.float32),
            np.ones(2, dtype=np.int32),
            np.array(0.0, dtype=np.float32),
        ]


def make_getitem_dataset(in_memory: bool):
    ds = BaseDataset((64, 64), 1.0, False, False)
    ds.batch_size = 2
    ds.prior_loss_weight = 1.0
    subset = make_subset(enable_multiline_caption=True, caption_suffix="suf")
    subset.flip_aug = False
    subset.custom_attributes = {}
    ds.subsets = [subset]
    infos = [
        ImageInfo("img0", 1, f"a\n{SEP}\nb\n{SEP}\nc", False, "img0.png"),
        ImageInfo("img1", 1, "d", False, "img1.png"),
    ]
    strategy = IndexRecordingStrategy()
    for info in infos:
        ds.register_image(info, subset)
        info.latents = torch.zeros(4, 8, 8)
        info.latents_original_size = (64, 64)
        info.latents_crop_ltrb = (0, 0, 0, 0)
        info.alpha_mask = None
        info.bucket_reso = (64, 64)
        if in_memory:
            num = len(BaseDataset.process_caption_candidates_for_cache(ds, subset, info.caption))
            info.text_encoder_outputs = [strategy.load_outputs_npz("mem", k) for k in range(num)]
        else:
            info.text_encoder_outputs_npz = strategy.get_outputs_npz_path(info.absolute_path)
    strategy.loaded.clear()
    ds.text_encoder_output_caching_strategy = strategy
    ds.bucket_manager = SimpleNamespace(buckets=[["img0", "img1"]])
    ds.buckets_indices = [BucketBatchIndex(0, 2, 0)]
    return ds, strategy


@pytest.mark.parametrize("in_memory", [False, True])
def test_getitem_chooses_candidate_and_matching_caption(in_memory):
    ds, strategy = make_getitem_dataset(in_memory)
    chosen = set()
    for seed in range(20):
        random.seed(seed)
        example = ds[0]
        embeds = example["text_encoder_outputs_list"][0]  # [B, L, C]
        k0 = int(embeds[0, 0, 0].item())
        chosen.add(k0)
        assert example["captions"] == [["a suf", "b suf", "c suf"][k0], "d suf"]
        assert int(embeds[1, 0, 0].item()) == 0
        assert example["input_ids_list"] is None
        if not in_memory:
            assert strategy.loaded[-2:] == [("img0.png_te.npz", k0), ("img1.png_te.npz", 0)]
    assert chosen == {0, 1, 2}
