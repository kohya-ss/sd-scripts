"""CLIPTokenizer with the text normalization of the legacy (slow) CLIP tokenizer.

Up to transformers 4.x, ``CLIPTokenizer`` was a Python (slow) tokenizer that ran
``ftfy.fix_text`` and whitespace cleanup on the text before BPE, exactly like the
original OpenAI CLIP tokenizer the models were trained with. ``ftfy.fix_text``
straightens curly quotes, converts full-width characters to ASCII, unescapes
HTML entities and repairs mojibake.

In transformers 5.x ``CLIPTokenizer`` is the fast (Rust) tokenizer, whose
normalizer only applies NFC, lowercasing and whitespace collapsing, so such
strings are tokenized differently. This subclass restores the ftfy step for the
fast tokenizer, which makes the token ids identical across transformers
versions. With a slow tokenizer it is a no-op (the cleanup is already applied
inside the tokenizer).
"""

import re
from typing import Any, List, Tuple, Union

import ftfy
from transformers import CLIPTokenizer as _CLIPTokenizer

_WHITESPACE_RE = re.compile(r"\s+")


def clean_clip_text(text: str) -> str:
    """Same normalization as the legacy slow ``CLIPTokenizer`` (``whitespace_clean(ftfy.fix_text(text))``)."""
    return _WHITESPACE_RE.sub(" ", ftfy.fix_text(text)).strip()


def _clean(value: Any) -> Any:
    # str, pairs of str (text, text_pair), pre-tokenized lists and batches of those
    if isinstance(value, str):
        return clean_clip_text(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_clean(v) for v in value)
    return value


def as_legacy_clip_tokenizer(tokenizer: _CLIPTokenizer) -> _CLIPTokenizer:
    """Give an existing ``transformers.CLIPTokenizer`` instance (e.g. one loaded by a diffusers pipeline) the legacy normalization."""
    if type(tokenizer) is _CLIPTokenizer:
        tokenizer.__class__ = CLIPTokenizer
    return tokenizer


class CLIPTokenizer(_CLIPTokenizer):
    """Drop-in replacement for ``transformers.CLIPTokenizer`` that keeps the legacy text normalization."""

    def _encode_plus(self, text, text_pair=None, *args, **kwargs):
        if self.is_fast:
            text, text_pair = _clean(text), _clean(text_pair)
        return super()._encode_plus(text, text_pair, *args, **kwargs)

    def _batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        if self.is_fast:
            batch_text_or_text_pairs = _clean(batch_text_or_text_pairs)
        return super()._batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)
