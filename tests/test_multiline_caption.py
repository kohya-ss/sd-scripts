"""Tests for multi-line captions (``enable_multiline_caption``) and caption candidate separator lines."""

import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from library.config_util import DreamBoothSubsetParams, FineTuningSubsetParams
from library.dataset import BaseDataset, normalize_multiline_caption, split_caption_candidates
from library.dreambooth_dataset import read_caption
from library.finetuning_dataset import build_finetuning_caption
from library.sampling import line_to_prompt_dict
from library.subset import DEFAULT_MULTILINE_CAPTION_SEPARATOR, DreamBoothSubset, FineTuningSubset

SEP = DEFAULT_MULTILINE_CAPTION_SEPARATOR


def make_subset(**overrides):
    """Minimal subset stand-in with the attributes ``process_caption`` reads."""
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
    """Minimal dataset stand-in; ``BaseDataset`` methods are called unbound on it."""
    return SimpleNamespace(
        current_epoch=0, current_step=0, max_train_steps=100, replacements={}, has_multiple_caption_candidates=False
    )


def process(subset, caption, seed=0):
    random.seed(seed)
    return BaseDataset.process_caption(make_dataset(), subset, caption)


# ---------------------------------------------------------------------------
# normalize_multiline_caption / split_caption_candidates
# ---------------------------------------------------------------------------


def test_normalize_keeps_inner_blank_lines_and_indentation():
    text = "\n\n  first line  \r\n\nsecond\t\n\n\n"
    assert normalize_multiline_caption(text) == "  first line\n\nsecond"


def test_split_single_candidate_without_separator():
    assert split_caption_candidates("a\nb\n", SEP) == ["a\nb"]


def test_split_multiple_candidates_with_comments():
    text = f"a\nb\n{SEP}\nc\n{SEP} # tag style\n1girl, solo\n"
    assert split_caption_candidates(text, SEP) == ["a\nb", "c", "1girl, solo"]


def test_split_separator_is_prefix_match_and_tolerates_surrounding_spaces():
    text = f"a\n  {SEP}   \nb"
    assert split_caption_candidates(text, SEP) == ["a", "b"]


def test_split_drops_empty_candidates():
    text = f"{SEP}\na\n{SEP}\n\n{SEP}\n"
    assert split_caption_candidates(text, SEP) == ["a"]
    assert split_caption_candidates("", SEP) == [""]
    assert split_caption_candidates(f"{SEP}\n{SEP}", SEP) == [""]


def test_split_reserved_text_after_separator_raises():
    with pytest.raises(ValueError):
        split_caption_candidates(f"a\n{SEP} 0.2\nb", SEP)
    # a comment is fine, text before the comment is not
    assert split_caption_candidates(f"a\n{SEP}#0.2\nb", SEP) == ["a", "b"]
    with pytest.raises(ValueError):
        split_caption_candidates(f"a\n{SEP} weight # comment\nb", SEP)


def test_split_custom_separator():
    assert split_caption_candidates("a\n---\nb", "---") == ["a", "b"]
    # the default separator is not special with a custom one
    assert split_caption_candidates(f"a\n{SEP}\nb", "===") == [f"a\n{SEP}\nb"]
    # prefix match: a content line starting with the custom separator followed by other text is an error
    with pytest.raises(ValueError):
        split_caption_candidates(f"a\n{SEP}\nb", "---")


# ---------------------------------------------------------------------------
# process_caption
# ---------------------------------------------------------------------------


def test_default_mode_uses_first_line_and_applies_prefix_suffix():
    subset = make_subset(caption_prefix="pre", caption_suffix="suf")
    assert process(subset, "line1\nline2") == "pre line1 suf"


def test_multiline_mode_keeps_line_breaks_and_applies_prefix_suffix():
    subset = make_subset(enable_multiline_caption=True, caption_prefix="pre", caption_suffix="suf")
    assert process(subset, "line1\nline2\n") == "pre line1\nline2 suf"


def test_multiline_mode_chooses_one_candidate():
    subset = make_subset(enable_multiline_caption=True, caption_suffix="suf")
    caption = f"a\nb\n{SEP}\nc\n{SEP} # comment\nd"
    seen = {process(subset, caption, seed=s) for s in range(50)}
    assert seen == {"a\nb suf", "c suf", "d suf"}


def test_wildcard_mode_chooses_one_line_then_applies_prefix_suffix():
    # regression: prefix/suffix used to be added before the line selection, so the prefix was attached only to
    # the first line and the suffix only to the last line
    subset = make_subset(enable_wildcard=True, caption_prefix="pre", caption_suffix="suf")
    seen = {process(subset, "a\nb", seed=s) for s in range(50)}
    assert seen == {"pre a suf", "pre b suf"}


def test_multiline_mode_with_wildcard_replaces_braces_but_does_not_split_lines():
    subset = make_subset(enable_multiline_caption=True, enable_wildcard=True)
    seen = {process(subset, "a {x|y}\nb", seed=s) for s in range(50)}
    assert seen == {"a x\nb", "a y\nb"}


def test_multiline_mode_shuffle_applies_only_to_single_line_candidates():
    subset = make_subset(enable_multiline_caption=True, shuffle_caption=True)
    multi = "a, b\nc, d"
    assert all(process(subset, multi, seed=s) == multi for s in range(20))
    single = "a, b, c, d"
    results = {process(subset, single, seed=s) for s in range(50)}
    assert len(results) > 1
    assert all(sorted(r.split(", ")) == ["a", "b", "c", "d"] for r in results)


def test_multiline_mode_tag_dropout_applies_only_to_single_line_candidates():
    subset = make_subset(enable_multiline_caption=True, caption_tag_dropout_rate=1.0)
    assert process(subset, "a, b\nc") == "a, b\nc"
    assert process(subset, "a, b") == ""


def test_default_mode_shuffle_is_unchanged():
    subset = make_subset(shuffle_caption=True)
    results = {process(subset, "a, b, c, d", seed=s) for s in range(50)}
    assert len(results) > 1
    assert all(sorted(r.split(", ")) == ["a", "b", "c", "d"] for r in results)


def test_multiline_mode_secondary_separator_and_replacements_apply():
    subset = make_subset(enable_multiline_caption=True, secondary_separator=";;;")
    ds = make_dataset()
    ds.replacements = {"girl": "woman"}
    assert BaseDataset.process_caption(ds, subset, "a;;;b\ngirl") == "a,b\nwoman"


def test_caption_dropout_in_multiline_mode():
    subset = make_subset(enable_multiline_caption=True, caption_dropout_rate=1.0)
    assert process(subset, f"a\n{SEP}\nb") == ""


# ---------------------------------------------------------------------------
# validate_caption_candidates / is_text_encoder_output_cacheable
# ---------------------------------------------------------------------------


def test_validate_sets_multiple_candidates_flag():
    ds = make_dataset()
    subset = make_subset(enable_multiline_caption=True)
    BaseDataset.validate_caption_candidates(ds, subset, ["a\nb", "c"])
    assert ds.has_multiple_caption_candidates is False
    BaseDataset.validate_caption_candidates(ds, subset, ["a", f"b\n{SEP}\nc"])
    assert ds.has_multiple_caption_candidates is True


def test_validate_raises_on_invalid_separator_line():
    ds = make_dataset()
    subset = make_subset(enable_multiline_caption=True)
    with pytest.raises(ValueError):
        BaseDataset.validate_caption_candidates(ds, subset, [f"a\n{SEP} 0.5\nb"])


def test_validate_is_noop_without_multiline():
    ds = make_dataset()
    subset = make_subset(enable_multiline_caption=False)
    BaseDataset.validate_caption_candidates(ds, subset, [f"a\n{SEP} 0.5\nb"])  # no error, separator is not special
    assert ds.has_multiple_caption_candidates is False


def test_text_encoder_output_not_cacheable_with_multiple_candidates():
    ds = make_dataset()
    ds.subsets = [make_subset(enable_multiline_caption=True)]
    assert BaseDataset.is_text_encoder_output_cacheable(ds) is True
    ds.has_multiple_caption_candidates = True
    assert BaseDataset.is_text_encoder_output_cacheable(ds) is False


# ---------------------------------------------------------------------------
# read_caption (DreamBooth)
# ---------------------------------------------------------------------------


@pytest.fixture
def caption_dir(tmp_path):
    (tmp_path / "img.png").write_bytes(b"")
    # write bytes to keep CRLF as is (text mode would translate "\n" again on Windows)
    (tmp_path / "img.txt").write_bytes(f"  first  \r\nsecond\r\n\r\n{SEP}\r\nthird\r\n\r\n".encode("utf-8"))
    return tmp_path


def test_read_caption_default_first_line(caption_dir):
    assert read_caption(str(caption_dir / "img.png"), ".txt", False) == "first"


def test_read_caption_wildcard_joins_non_empty_lines(caption_dir):
    assert read_caption(str(caption_dir / "img.png"), ".txt", True) == f"first\nsecond\n{SEP}\nthird"


def test_read_caption_multiline_keeps_whole_file(caption_dir):
    raw = read_caption(str(caption_dir / "img.png"), ".txt", False, True)
    assert raw == f"  first  \nsecond\n\n{SEP}\nthird\n\n"  # CRLF is normalized by text mode
    assert split_caption_candidates(raw, SEP) == ["  first\nsecond", "third"]


def test_read_caption_missing_file(tmp_path):
    (tmp_path / "img.png").write_bytes(b"")
    assert read_caption(str(tmp_path / "img.png"), ".txt", False, True) is None


# ---------------------------------------------------------------------------
# build_finetuning_caption (fine-tuning metadata)
# ---------------------------------------------------------------------------


def test_finetuning_caption_default_appends_tags():
    subset = make_subset()
    assert build_finetuning_caption(subset, "a", "t1, t2") == "a,t1, t2"
    assert build_finetuning_caption(subset, "", "t1") == "t1"
    assert build_finetuning_caption(subset, None, None) == ""
    assert build_finetuning_caption(subset, "a", None) == "a"


def test_finetuning_caption_wildcard_appends_tags_to_each_line():
    subset = make_subset(enable_wildcard=True)
    assert build_finetuning_caption(subset, "a\n\nb", "t1\nt2") == "a,t1,t2\nb,t1,t2"
    assert build_finetuning_caption(subset, "", "t1") == "t1"


def test_finetuning_caption_multiline_string_appends_tags_to_each_candidate():
    subset = make_subset(enable_multiline_caption=True)
    caption = f"a\nb\n{SEP} # c\nc"
    assert build_finetuning_caption(subset, caption, "t") == f"a\nb,t\n{SEP}\nc,t"
    assert build_finetuning_caption(subset, caption, None) == caption  # unchanged when there are no tags
    assert build_finetuning_caption(subset, "", "t") == "t"


def test_finetuning_caption_multiline_list():
    subset = make_subset(enable_multiline_caption=True)
    joined = build_finetuning_caption(subset, ["a\nb", "c"], None)
    assert split_caption_candidates(joined, SEP) == ["a\nb", "c"]
    with_tags = build_finetuning_caption(subset, ["a\nb", "c"], "t")
    assert split_caption_candidates(with_tags, SEP) == ["a\nb,t", "c,t"]


def test_finetuning_caption_list_requires_multiline():
    with pytest.raises(ValueError):
        build_finetuning_caption(make_subset(), ["a", "b"], None)
    with pytest.raises(ValueError):
        build_finetuning_caption(make_subset(enable_multiline_caption=True), ["a", 1], None)


def test_finetuning_caption_list_roundtrips_through_json():
    subset = make_subset(enable_multiline_caption=True)
    md = json.loads(json.dumps({"caption": ["x\ny", "z"], "tags": "t"}))
    assert split_caption_candidates(build_finetuning_caption(subset, md["caption"], md["tags"]), SEP) == ["x\ny,t", "z,t"]


# ---------------------------------------------------------------------------
# Subset / config plumbing
# ---------------------------------------------------------------------------


def test_subset_params_roundtrip_to_subset_classes():
    db = DreamBoothSubset(**asdict(DreamBoothSubsetParams(image_dir="x", enable_multiline_caption=True)))
    assert db.enable_multiline_caption is True
    assert db.multiline_caption_separator == SEP

    ft = FineTuningSubset(**asdict(FineTuningSubsetParams(metadata_file="m.json", multiline_caption_separator="---")))
    assert ft.enable_multiline_caption is False
    assert ft.multiline_caption_separator == "---"


def test_subset_rejects_empty_separator_when_multiline_enabled():
    with pytest.raises(AssertionError):
        DreamBoothSubset(
            **asdict(DreamBoothSubsetParams(image_dir="x", enable_multiline_caption=True, multiline_caption_separator=""))
        )
    # allowed when the feature is disabled
    DreamBoothSubset(**asdict(DreamBoothSubsetParams(image_dir="x", multiline_caption_separator="")))


# ---------------------------------------------------------------------------
# sample prompts
# ---------------------------------------------------------------------------


def test_line_to_prompt_dict_converts_backslash_n():
    d = line_to_prompt_dict(r"hello\nworld --n bad\nneg --w 512")
    assert d["prompt"] == "hello\nworld"
    assert d["negative_prompt"] == "bad\nneg"
    assert d["width"] == 512


def test_line_to_prompt_dict_without_escape_is_unchanged():
    d = line_to_prompt_dict("hello world --n bad")
    assert d["prompt"] == "hello world"
    assert d["negative_prompt"] == "bad"
