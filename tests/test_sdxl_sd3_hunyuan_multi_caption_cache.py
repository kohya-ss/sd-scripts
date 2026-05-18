"""
Tests for multi-caption text encoder output caching in SDXL, SD3, and HunyuanImage strategies.
"""

import numpy as np
import pytest

from library.strategy_sdxl import SdxlTextEncoderOutputsCachingStrategy
from library.strategy_sd3 import Sd3TextEncoderOutputsCachingStrategy
from library.strategy_hunyuan_image import HunyuanImageTextEncoderOutputsCachingStrategy


# ==================== SDXL ====================

SDXL_SEQ = 77
SDXL_H1 = 768
SDXL_H2 = 1280


def _sdxl_single(p):
    np.savez(p, hidden_state1=np.random.randn(SDXL_SEQ, SDXL_H1).astype(np.float32),
             hidden_state2=np.random.randn(SDXL_SEQ, SDXL_H2).astype(np.float32),
             pool2=np.random.randn(SDXL_H2).astype(np.float32))


def _sdxl_multi(p, n=3):
    np.savez(p, hidden_state1=np.random.randn(n, SDXL_SEQ, SDXL_H1).astype(np.float32),
             hidden_state2=np.random.randn(n, SDXL_SEQ, SDXL_H2).astype(np.float32),
             pool2=np.random.randn(n, SDXL_H2).astype(np.float32))


class TestSdxlLoad:
    def setup_method(self):
        self.s = SdxlTextEncoderOutputsCachingStrategy(True, 1, False)

    def test_single(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _sdxl_single(p)
        r = self.s.load_outputs_npz(p)
        assert r[0].shape == (SDXL_SEQ, SDXL_H1)
        assert r[1].shape == (SDXL_SEQ, SDXL_H2)
        assert r[2].shape == (SDXL_H2,)

    def test_multi(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _sdxl_multi(p, 5)
        r = self.s.load_outputs_npz(p)
        assert r[0].shape == (SDXL_SEQ, SDXL_H1)
        assert r[1].shape == (SDXL_SEQ, SDXL_H2)
        assert r[2].shape == (SDXL_H2,)

    def test_randomness(self, tmp_path):
        p = str(tmp_path / "t.npz")
        n = 10
        np.savez(p,
                 hidden_state1=np.arange(n, dtype=np.float32).reshape(n, 1, 1) * np.ones((1, SDXL_SEQ, SDXL_H1), dtype=np.float32),
                 hidden_state2=np.random.randn(n, SDXL_SEQ, SDXL_H2).astype(np.float32),
                 pool2=np.random.randn(n, SDXL_H2).astype(np.float32))
        vals = set(float(self.s.load_outputs_npz(p)[0][0, 0]) for _ in range(50))
        assert len(vals) > 1

    def test_validity_single(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _sdxl_single(p)
        assert self.s.is_disk_cached_outputs_expected(p) is True

    def test_validity_multi(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _sdxl_multi(p)
        assert self.s.is_disk_cached_outputs_expected(p) is True


# ==================== SD3 ====================

SD3_SEQ_LG = 77
SD3_SEQ_T5 = 256
SD3_LG_DIM = 2048
SD3_T5_DIM = 4096
SD3_POOL = 2048


def _sd3_single(p):
    np.savez(p, lg_out=np.random.randn(SD3_SEQ_LG, SD3_LG_DIM).astype(np.float32),
             lg_pooled=np.random.randn(SD3_POOL).astype(np.float32),
             t5_out=np.random.randn(SD3_SEQ_T5, SD3_T5_DIM).astype(np.float32),
             clip_l_attn_mask=np.ones(SD3_SEQ_LG, dtype=np.float32),
             clip_g_attn_mask=np.ones(SD3_SEQ_LG, dtype=np.float32),
             t5_attn_mask=np.ones(SD3_SEQ_T5, dtype=np.float32),
             apply_lg_attn_mask=False, apply_t5_attn_mask=False)


def _sd3_multi(p, n=3):
    np.savez(p, lg_out=np.random.randn(n, SD3_SEQ_LG, SD3_LG_DIM).astype(np.float32),
             lg_pooled=np.random.randn(n, SD3_POOL).astype(np.float32),
             t5_out=np.random.randn(n, SD3_SEQ_T5, SD3_T5_DIM).astype(np.float32),
             clip_l_attn_mask=np.ones((n, SD3_SEQ_LG), dtype=np.float32),
             clip_g_attn_mask=np.ones((n, SD3_SEQ_LG), dtype=np.float32),
             t5_attn_mask=np.ones((n, SD3_SEQ_T5), dtype=np.float32),
             apply_lg_attn_mask=False, apply_t5_attn_mask=False)


class TestSd3Load:
    def setup_method(self):
        self.s = Sd3TextEncoderOutputsCachingStrategy(True, 1, False)

    def test_single(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _sd3_single(p)
        r = self.s.load_outputs_npz(p)
        assert len(r) == 6
        assert r[0].shape == (SD3_SEQ_LG, SD3_LG_DIM)  # lg_out
        assert r[1].shape == (SD3_SEQ_T5, SD3_T5_DIM)   # t5_out
        assert r[2].shape == (SD3_POOL,)                  # lg_pooled

    def test_multi(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _sd3_multi(p, 5)
        r = self.s.load_outputs_npz(p)
        assert r[0].shape == (SD3_SEQ_LG, SD3_LG_DIM)
        assert r[1].shape == (SD3_SEQ_T5, SD3_T5_DIM)
        assert r[2].shape == (SD3_POOL,)
        assert r[3].shape == (SD3_SEQ_LG,)  # l_attn_mask
        assert r[4].shape == (SD3_SEQ_LG,)  # g_attn_mask
        assert r[5].shape == (SD3_SEQ_T5,)  # t5_attn_mask

    def test_validity_both(self, tmp_path):
        p1 = str(tmp_path / "s.npz")
        p2 = str(tmp_path / "m.npz")
        _sd3_single(p1)
        _sd3_multi(p2)
        assert self.s.is_disk_cached_outputs_expected(p1) is True
        assert self.s.is_disk_cached_outputs_expected(p2) is True


# ==================== HunyuanImage ====================

HI_VLM_SEQ = 256
HI_VLM_DIM = 2048
HI_BYT5_SEQ = 128
HI_BYT5_DIM = 1472


def _hi_single(p):
    np.savez(p, vlm_embed=np.random.randn(HI_VLM_SEQ, HI_VLM_DIM).astype(np.float32),
             vlm_mask=np.ones(HI_VLM_SEQ, dtype=np.float32),
             byt5_embed=np.random.randn(HI_BYT5_SEQ, HI_BYT5_DIM).astype(np.float32),
             byt5_mask=np.ones(HI_BYT5_SEQ, dtype=np.float32),
             ocr_mask=np.array(False))


def _hi_multi(p, n=3):
    np.savez(p, vlm_embed=np.random.randn(n, HI_VLM_SEQ, HI_VLM_DIM).astype(np.float32),
             vlm_mask=np.ones((n, HI_VLM_SEQ), dtype=np.float32),
             byt5_embed=np.random.randn(n, HI_BYT5_SEQ, HI_BYT5_DIM).astype(np.float32),
             byt5_mask=np.ones((n, HI_BYT5_SEQ), dtype=np.float32),
             ocr_mask=np.array([False, True, False][:n]))


class TestHunyuanImageLoad:
    def setup_method(self):
        self.s = HunyuanImageTextEncoderOutputsCachingStrategy(True, 1, False)

    def test_single(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _hi_single(p)
        r = self.s.load_outputs_npz(p)
        assert len(r) == 5
        assert r[0].shape == (HI_VLM_SEQ, HI_VLM_DIM)  # vlm_embed
        assert r[1].shape == (HI_VLM_SEQ,)               # vlm_mask
        assert r[2].shape == (HI_BYT5_SEQ, HI_BYT5_DIM) # byt5_embed
        assert r[3].shape == (HI_BYT5_SEQ,)              # byt5_mask

    def test_multi(self, tmp_path):
        p = str(tmp_path / "t.npz")
        _hi_multi(p, 3)
        r = self.s.load_outputs_npz(p)
        assert r[0].shape == (HI_VLM_SEQ, HI_VLM_DIM)
        assert r[2].shape == (HI_BYT5_SEQ, HI_BYT5_DIM)
        # ocr_mask should be scalar after selection
        assert r[4].ndim == 0

    def test_validity_both(self, tmp_path):
        p1 = str(tmp_path / "s.npz")
        p2 = str(tmp_path / "m.npz")
        _hi_single(p1)
        _hi_multi(p2)
        assert self.s.is_disk_cached_outputs_expected(p1) is True
        assert self.s.is_disk_cached_outputs_expected(p2) is True
