"""Caption processing when caching Text Encoder outputs (issue #2453).

The deterministic part of ``process_caption`` (prefix/suffix, secondary separator, replacements, first-line
truncation) must be applied to the captions passed to ``cache_batch_outputs``.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

from library.dataset import BaseDataset, ImageInfo
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
    return SimpleNamespace(current_epoch=0, current_step=0, max_train_steps=100, replacements={}, has_multiple_caption_candidates=False)


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
