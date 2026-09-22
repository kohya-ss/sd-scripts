"""Tests for library.clip_tokenizer (legacy CLIP text normalization on top of the fast tokenizer).

The tokenizer itself needs files from the Hugging Face Hub, so only the pure normalization is tested here.
The end-to-end token ids are covered by the local regression harness (tests/local).
"""

from library.clip_tokenizer import _clean, clean_clip_text


def test_clean_clip_text_matches_legacy_normalization():
    # curly quotes are straightened, full-width characters become ASCII, HTML entities are unescaped,
    # mojibake is repaired, whitespace is collapsed and stripped (ftfy.fix_text + whitespace_clean)
    assert clean_clip_text("“quoted”") == '"quoted"'
    assert clean_clip_text("ＡＢＣ　１２３") == "ABC 123"
    assert clean_clip_text("a &amp; b") == "a & b"
    assert clean_clip_text("caf\u00c3\u00a9") == "caf\u00e9"
    assert clean_clip_text("  many   spaces\n\tand lines  ") == "many spaces and lines"
    # plain text is unchanged
    assert clean_clip_text("a photo of a cat, masterpiece") == "a photo of a cat, masterpiece"
    assert clean_clip_text("") == ""


def test_clean_handles_batches_pairs_and_pretokenized_input():
    assert _clean(None) is None
    assert _clean(["“a”", "b"]) == ['"a"', "b"]
    assert _clean(("“a”", "“b”")) == ('"a"', '"b"')
    assert _clean([["“a”", "b"], ["c"]]) == [['"a"', "b"], ["c"]]
    assert _clean(3) == 3
