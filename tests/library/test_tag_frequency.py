from types import SimpleNamespace

from library.dataset import BaseDataset


def _frequency(captions, **kwargs):
    dataset = SimpleNamespace(tag_frequency={})
    BaseDataset.set_tag_frequency(dataset, "dir", captions, **kwargs)
    return dataset.tag_frequency["dir"]


def test_default_comma_separator_is_unchanged():
    assert _frequency(["1girl, Solo", "1girl"]) == {"1girl": 2, "solo": 1}


def test_keep_tokens_separator_splits_tags():
    freq = _frequency(["1boy ||| standing, red", "1boy, blue ||| smile"], keep_tokens_separator="|||")
    assert freq == {"1boy": 2, "standing": 1, "red": 1, "blue": 1, "smile": 1}


def test_custom_caption_separator():
    freq = _frequency(["1girl; solo ### smile"], caption_separator=";", keep_tokens_separator="###")
    assert freq == {"1girl": 1, "solo": 1, "smile": 1}
