"""
Tests for multi-caption text encoder output caching in Lumina strategy.
Covers both disk cache (NPZ) and memory cache paths, including backward compatibility.
"""

import numpy as np
import pytest

from library.strategy_lumina import LuminaTextEncoderOutputsCachingStrategy


class ImageInfo:
    """Minimal stub for testing multi-caption caching without importing train_util."""
    def __init__(self, image_key, num_repeats, caption, is_reg, absolute_path):
        self.image_key = image_key
        self.caption = caption
        self.num_caption_variants = 1
        self.text_encoder_outputs = None


SEQ_LEN = 256
HIDDEN_SIZE = 2304


def _make_single_caption_npz(npz_path: str):
    """Create a single-caption NPZ file (old format, no num_captions key)."""
    np.savez(
        npz_path,
        hidden_state=np.random.randn(SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
        attention_mask=np.ones(SEQ_LEN, dtype=np.float32),
        input_ids=np.ones(SEQ_LEN, dtype=np.int64),
    )


def _make_multi_caption_npz(npz_path: str, num_captions: int = 3):
    """Create a multi-caption NPZ file (new format, with num_captions key)."""
    np.savez(
        npz_path,
        hidden_state=np.random.randn(num_captions, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
        attention_mask=np.ones((num_captions, SEQ_LEN), dtype=np.float32),
        input_ids=np.ones((num_captions, SEQ_LEN), dtype=np.int64),
    )


class TestLoadOutputsNpz:
    """Tests for load_outputs_npz with single and multi-caption NPZ files."""

    def setup_method(self):
        self.strategy = LuminaTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False
        )

    def test_load_single_caption_npz(self, tmp_path):
        """Old-format NPZ should return single arrays unchanged."""
        npz_path = str(tmp_path / "test_lumina_te.npz")
        _make_single_caption_npz(npz_path)

        result = self.strategy.load_outputs_npz(npz_path)
        assert len(result) == 3
        hidden_state, input_ids, attention_mask = result
        assert hidden_state.shape == (SEQ_LEN, HIDDEN_SIZE)
        assert input_ids.shape == (SEQ_LEN,)
        assert attention_mask.shape == (SEQ_LEN,)

    def test_load_multi_caption_npz_returns_single(self, tmp_path):
        """Multi-caption NPZ should return a single randomly selected caption."""
        npz_path = str(tmp_path / "test_lumina_te.npz")
        num_captions = 5
        _make_multi_caption_npz(npz_path, num_captions)

        result = self.strategy.load_outputs_npz(npz_path)
        assert len(result) == 3
        hidden_state, input_ids, attention_mask = result
        assert hidden_state.shape == (SEQ_LEN, HIDDEN_SIZE)
        assert input_ids.shape == (SEQ_LEN,)
        assert attention_mask.shape == (SEQ_LEN,)

    def test_load_multi_caption_npz_randomness(self, tmp_path):
        """Multiple loads from multi-caption NPZ should not always return the same index."""
        npz_path = str(tmp_path / "test_lumina_te.npz")
        num_captions = 10
        hidden_states = np.arange(num_captions).reshape(num_captions, 1, 1) * np.ones(
            (1, SEQ_LEN, HIDDEN_SIZE), dtype=np.float32
        )
        np.savez(
            npz_path,
            hidden_state=hidden_states.astype(np.float32),
            attention_mask=np.ones((num_captions, SEQ_LEN), dtype=np.float32),
            input_ids=np.ones((num_captions, SEQ_LEN), dtype=np.int64),
        )

        selected_values = set()
        for _ in range(50):
            result = self.strategy.load_outputs_npz(npz_path)
            selected_values.add(float(result[0][0, 0]))

        assert len(selected_values) > 1, "Random selection should produce different results"


class TestIsDiskCachedOutputsExpected:
    """Tests for cache validity check with both old and new NPZ formats."""

    def setup_method(self):
        self.strategy = LuminaTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False
        )

    def test_single_caption_npz_valid(self, tmp_path):
        npz_path = str(tmp_path / "test_lumina_te.npz")
        _make_single_caption_npz(npz_path)
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is True

    def test_multi_caption_npz_valid(self, tmp_path):
        npz_path = str(tmp_path / "test_lumina_te.npz")
        _make_multi_caption_npz(npz_path, num_captions=3)
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is True

    def test_missing_file(self, tmp_path):
        npz_path = str(tmp_path / "nonexistent_lumina_te.npz")
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is False

    def test_incomplete_npz(self, tmp_path):
        npz_path = str(tmp_path / "test_lumina_te.npz")
        np.savez(npz_path, hidden_state=np.zeros(10))  # missing attention_mask and input_ids
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is False


class TestImageInfoMultiCaption:
    """Tests for ImageInfo.num_caption_variants field."""

    def test_default_num_caption_variants(self):
        info = ImageInfo("key", 1, "a cat", False, "/path/to/img.png")
        assert info.num_caption_variants == 1

    def test_set_num_caption_variants(self):
        info = ImageInfo("key", 1, "a cat", False, "/path/to/img.png")
        info.num_caption_variants = 5
        assert info.num_caption_variants == 5


class TestMemoryCacheMultiCaption:
    """Tests for memory cache random selection in __getitem__ pattern."""

    def test_single_caption_memory_no_selection(self):
        """Single caption (num_caption_variants=1) should return data as-is."""
        info = ImageInfo("key", 1, "a cat", False, "/path/to/img.png")
        info.num_caption_variants = 1
        info.text_encoder_outputs = [
            np.random.randn(SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
            np.ones(SEQ_LEN, dtype=np.int64),
            np.ones(SEQ_LEN, dtype=np.float32),
        ]

        text_encoder_outputs = info.text_encoder_outputs
        if info.num_caption_variants > 1:
            import random
            idx = random.randint(0, info.num_caption_variants - 1)
            text_encoder_outputs = [arr[idx] for arr in text_encoder_outputs]

        assert text_encoder_outputs[0].shape == (SEQ_LEN, HIDDEN_SIZE)
        assert text_encoder_outputs[1].shape == (SEQ_LEN,)
        assert text_encoder_outputs[2].shape == (SEQ_LEN,)

    def test_multi_caption_memory_selection(self):
        """Multi-caption (num_caption_variants>1) should select one variant."""
        import random

        num_captions = 4
        info = ImageInfo("key", 1, "line1\nline2\nline3\nline4", False, "/path/to/img.png")
        info.num_caption_variants = num_captions
        info.text_encoder_outputs = [
            np.random.randn(num_captions, SEQ_LEN, HIDDEN_SIZE).astype(np.float32),
            np.ones((num_captions, SEQ_LEN), dtype=np.int64),
            np.ones((num_captions, SEQ_LEN), dtype=np.float32),
        ]

        text_encoder_outputs = info.text_encoder_outputs
        if info.num_caption_variants > 1:
            idx = random.randint(0, info.num_caption_variants - 1)
            text_encoder_outputs = [arr[idx] for arr in text_encoder_outputs]

        assert text_encoder_outputs[0].shape == (SEQ_LEN, HIDDEN_SIZE)
        assert text_encoder_outputs[1].shape == (SEQ_LEN,)
        assert text_encoder_outputs[2].shape == (SEQ_LEN,)
