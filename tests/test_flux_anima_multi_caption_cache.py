"""
Tests for multi-caption text encoder output caching in Flux and Anima strategies.
Covers both disk cache (NPZ) and memory cache paths, including backward compatibility.
"""

import numpy as np
import pytest

from library.strategy_flux import FluxTextEncoderOutputsCachingStrategy
from library.strategy_anima import AnimaTextEncoderOutputsCachingStrategy


class ImageInfo:
    """Minimal stub for testing multi-caption caching without importing train_util."""
    def __init__(self, image_key, num_repeats, caption, is_reg, absolute_path):
        self.image_key = image_key
        self.caption = caption
        self.num_caption_variants = 1
        self.text_encoder_outputs = None


# --- Flux Constants ---
FLUX_SEQ_LEN = 512
FLUX_T5_HIDDEN = 4096
FLUX_CLIP_POOLED = 768


def _make_flux_single_npz(npz_path: str):
    """Create a single-caption Flux NPZ file (old format)."""
    np.savez(
        npz_path,
        l_pooled=np.random.randn(FLUX_CLIP_POOLED).astype(np.float32),
        t5_out=np.random.randn(FLUX_SEQ_LEN, FLUX_T5_HIDDEN).astype(np.float32),
        txt_ids=np.zeros((FLUX_SEQ_LEN, 3), dtype=np.float32),
        t5_attn_mask=np.ones(FLUX_SEQ_LEN, dtype=np.float32),
        apply_t5_attn_mask=False,
    )


def _make_flux_multi_npz(npz_path: str, num_captions: int = 3):
    """Create a multi-caption Flux NPZ file (new format)."""
    np.savez(
        npz_path,
        l_pooled=np.random.randn(num_captions, FLUX_CLIP_POOLED).astype(np.float32),
        t5_out=np.random.randn(num_captions, FLUX_SEQ_LEN, FLUX_T5_HIDDEN).astype(np.float32),
        txt_ids=np.zeros((num_captions, FLUX_SEQ_LEN, 3), dtype=np.float32),
        t5_attn_mask=np.ones((num_captions, FLUX_SEQ_LEN), dtype=np.float32),
        apply_t5_attn_mask=False,
    )


# --- Anima Constants ---
ANIMA_SEQ_LEN = 512
ANIMA_HIDDEN = 2048


def _make_anima_single_npz(npz_path: str):
    """Create a single-caption Anima NPZ file (old format)."""
    np.savez(
        npz_path,
        prompt_embeds=np.random.randn(ANIMA_SEQ_LEN, ANIMA_HIDDEN).astype(np.float32),
        attn_mask=np.ones(ANIMA_SEQ_LEN, dtype=np.float32),
        t5_input_ids=np.ones(ANIMA_SEQ_LEN, dtype=np.int32),
        t5_attn_mask=np.ones(ANIMA_SEQ_LEN, dtype=np.int32),
        caption_dropout_rate=np.float32(0.0),
    )


def _make_anima_multi_npz(npz_path: str, num_captions: int = 3):
    """Create a multi-caption Anima NPZ file (new format)."""
    np.savez(
        npz_path,
        prompt_embeds=np.random.randn(num_captions, ANIMA_SEQ_LEN, ANIMA_HIDDEN).astype(np.float32),
        attn_mask=np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.float32),
        t5_input_ids=np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.int32),
        t5_attn_mask=np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.int32),
        caption_dropout_rate=np.float32(0.0),
    )


# ==================== Flux Tests ====================


class TestFluxLoadOutputsNpz:

    def setup_method(self):
        self.strategy = FluxTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False, apply_t5_attn_mask=False,
        )

    def test_load_single_caption(self, tmp_path):
        npz_path = str(tmp_path / "test_flux_te.npz")
        _make_flux_single_npz(npz_path)

        result = self.strategy.load_outputs_npz(npz_path)
        assert len(result) == 4
        l_pooled, t5_out, txt_ids, t5_attn_mask = result
        assert l_pooled.shape == (FLUX_CLIP_POOLED,)
        assert t5_out.shape == (FLUX_SEQ_LEN, FLUX_T5_HIDDEN)
        assert txt_ids.shape == (FLUX_SEQ_LEN, 3)
        assert t5_attn_mask.shape == (FLUX_SEQ_LEN,)

    def test_load_multi_caption_returns_single(self, tmp_path):
        npz_path = str(tmp_path / "test_flux_te.npz")
        _make_flux_multi_npz(npz_path, num_captions=5)

        result = self.strategy.load_outputs_npz(npz_path)
        assert len(result) == 4
        l_pooled, t5_out, txt_ids, t5_attn_mask = result
        assert l_pooled.shape == (FLUX_CLIP_POOLED,)
        assert t5_out.shape == (FLUX_SEQ_LEN, FLUX_T5_HIDDEN)
        assert txt_ids.shape == (FLUX_SEQ_LEN, 3)
        assert t5_attn_mask.shape == (FLUX_SEQ_LEN,)

    def test_load_multi_caption_randomness(self, tmp_path):
        npz_path = str(tmp_path / "test_flux_te.npz")
        num_captions = 10
        l_pooled_data = np.arange(num_captions, dtype=np.float32).reshape(num_captions, 1) * np.ones(
            (1, FLUX_CLIP_POOLED), dtype=np.float32
        )
        np.savez(
            npz_path,
            l_pooled=l_pooled_data,
            t5_out=np.random.randn(num_captions, FLUX_SEQ_LEN, FLUX_T5_HIDDEN).astype(np.float32),
            txt_ids=np.zeros((num_captions, FLUX_SEQ_LEN, 3), dtype=np.float32),
            t5_attn_mask=np.ones((num_captions, FLUX_SEQ_LEN), dtype=np.float32),
            apply_t5_attn_mask=False,
        )

        selected_values = set()
        for _ in range(50):
            result = self.strategy.load_outputs_npz(npz_path)
            selected_values.add(float(result[0][0]))

        assert len(selected_values) > 1, "Random selection should produce different results"


class TestFluxCacheValidity:

    def setup_method(self):
        self.strategy = FluxTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False, apply_t5_attn_mask=False,
        )

    def test_single_caption_valid(self, tmp_path):
        npz_path = str(tmp_path / "test_flux_te.npz")
        _make_flux_single_npz(npz_path)
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is True

    def test_multi_caption_valid(self, tmp_path):
        npz_path = str(tmp_path / "test_flux_te.npz")
        _make_flux_multi_npz(npz_path, num_captions=3)
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is True

    def test_missing_file(self, tmp_path):
        npz_path = str(tmp_path / "nonexistent.npz")
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is False


# ==================== Anima Tests ====================


class TestAnimaLoadOutputsNpz:

    def setup_method(self):
        self.strategy = AnimaTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False,
        )

    def test_load_single_caption(self, tmp_path):
        npz_path = str(tmp_path / "test_anima_te.npz")
        _make_anima_single_npz(npz_path)

        result = self.strategy.load_outputs_npz(npz_path)
        assert len(result) == 5
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate = result
        assert prompt_embeds.shape == (ANIMA_SEQ_LEN, ANIMA_HIDDEN)
        assert attn_mask.shape == (ANIMA_SEQ_LEN,)
        assert t5_input_ids.shape == (ANIMA_SEQ_LEN,)
        assert t5_attn_mask.shape == (ANIMA_SEQ_LEN,)
        assert caption_dropout_rate.shape == ()  # scalar

    def test_load_multi_caption_returns_single(self, tmp_path):
        npz_path = str(tmp_path / "test_anima_te.npz")
        _make_anima_multi_npz(npz_path, num_captions=5)

        result = self.strategy.load_outputs_npz(npz_path)
        assert len(result) == 5
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, caption_dropout_rate = result
        assert prompt_embeds.shape == (ANIMA_SEQ_LEN, ANIMA_HIDDEN)
        assert attn_mask.shape == (ANIMA_SEQ_LEN,)
        assert t5_input_ids.shape == (ANIMA_SEQ_LEN,)
        assert t5_attn_mask.shape == (ANIMA_SEQ_LEN,)
        # caption_dropout_rate should remain scalar (NOT indexed)
        assert caption_dropout_rate.shape == ()

    def test_load_multi_caption_randomness(self, tmp_path):
        npz_path = str(tmp_path / "test_anima_te.npz")
        num_captions = 10
        prompt_data = np.arange(num_captions, dtype=np.float32).reshape(num_captions, 1, 1) * np.ones(
            (1, ANIMA_SEQ_LEN, ANIMA_HIDDEN), dtype=np.float32
        )
        np.savez(
            npz_path,
            prompt_embeds=prompt_data,
            attn_mask=np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.float32),
            t5_input_ids=np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.int32),
            t5_attn_mask=np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.int32),
            caption_dropout_rate=np.float32(0.1),
        )

        selected_values = set()
        for _ in range(50):
            result = self.strategy.load_outputs_npz(npz_path)
            selected_values.add(float(result[0][0, 0]))

        assert len(selected_values) > 1, "Random selection should produce different results"


class TestAnimaCacheValidity:

    def setup_method(self):
        self.strategy = AnimaTextEncoderOutputsCachingStrategy(
            cache_to_disk=True, batch_size=1, skip_disk_cache_validity_check=False,
        )

    def test_single_caption_valid(self, tmp_path):
        npz_path = str(tmp_path / "test_anima_te.npz")
        _make_anima_single_npz(npz_path)
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is True

    def test_multi_caption_valid(self, tmp_path):
        npz_path = str(tmp_path / "test_anima_te.npz")
        _make_anima_multi_npz(npz_path, num_captions=3)
        assert self.strategy.is_disk_cached_outputs_expected(npz_path) is True


# ==================== Memory Cache Tests ====================


class TestMemoryCacheMultiCaptionFlux:

    def test_single_caption_no_selection(self):
        info = ImageInfo("key", 1, "a cat", False, "/path/to/img.png")
        info.num_caption_variants = 1
        info.text_encoder_outputs = (
            np.random.randn(FLUX_CLIP_POOLED).astype(np.float32),
            np.random.randn(FLUX_SEQ_LEN, FLUX_T5_HIDDEN).astype(np.float32),
            np.zeros((FLUX_SEQ_LEN, 3), dtype=np.float32),
            np.ones(FLUX_SEQ_LEN, dtype=np.float32),
        )

        text_encoder_outputs = info.text_encoder_outputs
        if info.num_caption_variants > 1:
            import random
            idx = random.randint(0, info.num_caption_variants - 1)
            text_encoder_outputs = [arr[idx] for arr in text_encoder_outputs]

        assert text_encoder_outputs[0].shape == (FLUX_CLIP_POOLED,)
        assert text_encoder_outputs[1].shape == (FLUX_SEQ_LEN, FLUX_T5_HIDDEN)

    def test_multi_caption_selection(self):
        import random
        num_captions = 4
        info = ImageInfo("key", 1, "l1\nl2\nl3\nl4", False, "/path/to/img.png")
        info.num_caption_variants = num_captions
        info.text_encoder_outputs = (
            np.random.randn(num_captions, FLUX_CLIP_POOLED).astype(np.float32),
            np.random.randn(num_captions, FLUX_SEQ_LEN, FLUX_T5_HIDDEN).astype(np.float32),
            np.zeros((num_captions, FLUX_SEQ_LEN, 3), dtype=np.float32),
            np.ones((num_captions, FLUX_SEQ_LEN), dtype=np.float32),
        )

        text_encoder_outputs = info.text_encoder_outputs
        if info.num_caption_variants > 1:
            idx = random.randint(0, info.num_caption_variants - 1)
            text_encoder_outputs = [arr[idx] for arr in text_encoder_outputs]

        assert text_encoder_outputs[0].shape == (FLUX_CLIP_POOLED,)
        assert text_encoder_outputs[1].shape == (FLUX_SEQ_LEN, FLUX_T5_HIDDEN)
        assert text_encoder_outputs[2].shape == (FLUX_SEQ_LEN, 3)
        assert text_encoder_outputs[3].shape == (FLUX_SEQ_LEN,)


class TestMemoryCacheMultiCaptionAnima:

    def test_multi_caption_selection_preserves_scalar(self):
        """caption_dropout_rate is the 5th element and is a scalar - should NOT be indexed."""
        import random
        import torch
        num_captions = 3
        info = ImageInfo("key", 1, "l1\nl2\nl3", False, "/path/to/img.png")
        info.num_caption_variants = num_captions
        info.text_encoder_outputs = (
            np.random.randn(num_captions, ANIMA_SEQ_LEN, ANIMA_HIDDEN).astype(np.float32),
            np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.float32),
            np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.int32),
            np.ones((num_captions, ANIMA_SEQ_LEN), dtype=np.int32),
            torch.tensor(0.1, dtype=torch.float32),  # scalar caption_dropout_rate
        )

        # Simulate __getitem__ logic with ndim check (same as train_util.py)
        text_encoder_outputs = info.text_encoder_outputs
        if info.num_caption_variants > 1:
            idx = random.randint(0, info.num_caption_variants - 1)
            text_encoder_outputs = [arr[idx] if hasattr(arr, 'ndim') and arr.ndim > 0 else arr for arr in text_encoder_outputs]

        assert text_encoder_outputs[0].shape == (ANIMA_SEQ_LEN, ANIMA_HIDDEN)
        assert text_encoder_outputs[1].shape == (ANIMA_SEQ_LEN,)
        assert text_encoder_outputs[2].shape == (ANIMA_SEQ_LEN,)
        assert text_encoder_outputs[3].shape == (ANIMA_SEQ_LEN,)
        # caption_dropout_rate should remain scalar (0-dim), not indexed
        assert text_encoder_outputs[4].ndim == 0
        assert float(text_encoder_outputs[4]) == pytest.approx(0.1)
