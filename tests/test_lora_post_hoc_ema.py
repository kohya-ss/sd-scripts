"""Tests for networks/lora_post_hoc_ema.py."""

import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from library.safetensors_utils import MemoryEfficientSafeOpen
from networks.lora_post_hoc_ema import merge_lora_weights_with_post_hoc_ema, sigma_rel_to_gamma

PREFIX = "lora_unet_block"


def _make_checkpoints(tmp_path, n, dtype=torch.float32, alphas=None, metadata=None):
    paths, sds = [], []
    for i in range(n):
        g = torch.Generator().manual_seed(i)
        sd = {
            f"{PREFIX}.lora_down.weight": torch.randn(4, 16, generator=g).to(dtype),
            f"{PREFIX}.lora_up.weight": torch.randn(8, 4, generator=g).to(dtype),
            f"{PREFIX}.alpha": torch.tensor(4.0 if alphas is None else alphas[i]).to(dtype),
        }
        path = str(tmp_path / f"ckpt-{i:02d}.safetensors")
        save_file(sd, path, metadata=metadata)
        os.utime(path, (1_000_000 + i * 100, 1_000_000 + i * 100))  # older index = older file
        paths.append(path)
        sds.append(sd)
    return paths, sds


def _expected(sds, betas):
    merged = {k: v.to(torch.float32) for k, v in sds[0].items()}
    for sd, beta in zip(sds[1:], betas):
        for k in merged:
            if not k.endswith(".alpha"):
                merged[k] = merged[k] * beta + sd[k].to(torch.float32) * (1 - beta)
    return merged


def test_sigma_rel_to_gamma_matches_the_paper():
    # values from the EDM2 paper (https://arxiv.org/abs/2312.02696): sigma_rel 0.05 -> 16.97, 0.10 -> 6.94
    assert sigma_rel_to_gamma(0.05) == pytest.approx(16.97, abs=0.01)
    assert sigma_rel_to_gamma(0.10) == pytest.approx(6.94, abs=0.01)


def test_constant_beta(tmp_path):
    paths, sds = _make_checkpoints(tmp_path, 4)
    out = str(tmp_path / "out.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.9, 0.9, None, out)

    merged = load_file(out)
    expected = _expected(sds, [0.9] * 3)
    for k in expected:
        assert torch.allclose(merged[k], expected[k], atol=1e-6), k
    assert merged[f"{PREFIX}.alpha"].item() == 4.0


def test_beta_is_interpolated_to_beta2(tmp_path):
    paths, sds = _make_checkpoints(tmp_path, 4)
    out = str(tmp_path / "out.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.8, 0.9, None, out)

    merged = load_file(out)
    expected = _expected(sds, [0.8, 0.85, 0.9])
    for k in expected:
        assert torch.allclose(merged[k], expected[k], atol=1e-6), k


def test_sigma_rel(tmp_path):
    paths, sds = _make_checkpoints(tmp_path, 4)
    out = str(tmp_path / "out.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.9, 0.9, 0.2, out)

    gamma = sigma_rel_to_gamma(0.2)
    betas = [(1 - 1 / t) ** (gamma + 1) for t in (1, 2, 3)]
    assert betas[0] == 0.0  # the first file is replaced by the second one
    merged = load_file(out)
    expected = _expected(sds, betas)
    for k in expected:
        assert torch.allclose(merged[k], expected[k], atol=1e-6), k


def test_files_are_sorted_by_modification_time(tmp_path):
    paths, sds = _make_checkpoints(tmp_path, 3)
    expected = _expected(sds, [0.5] * 2)

    out = str(tmp_path / "sorted.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(reversed(paths)), False, 0.5, 0.5, None, out)
    merged = load_file(out)
    for k in expected:
        assert torch.allclose(merged[k], expected[k], atol=1e-6), k

    out = str(tmp_path / "unsorted.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(reversed(paths)), True, 0.5, 0.5, None, out)
    merged = load_file(out)
    expected_reversed = _expected(list(reversed(sds)), [0.5] * 2)
    for k in expected_reversed:
        assert torch.allclose(merged[k], expected_reversed[k], atol=1e-6), k


def test_dtype_is_kept(tmp_path):
    paths, _ = _make_checkpoints(tmp_path, 3, dtype=torch.bfloat16)
    out = str(tmp_path / "out.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.9, 0.9, None, out)
    assert all(v.dtype == torch.bfloat16 for v in load_file(out).values())


def test_different_alpha_is_an_error(tmp_path):
    paths, _ = _make_checkpoints(tmp_path, 3, alphas=[4.0, 4.0, 8.0])
    with pytest.raises(ValueError, match="[Aa]lpha"):
        merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.9, 0.9, None, str(tmp_path / "out.safetensors"))


def test_metadata_of_the_last_file_is_kept_and_hashes_are_updated(tmp_path):
    metadata = {"ss_network_dim": "4", "sshs_model_hash": "old", "sshs_legacy_hash": "old"}
    paths, _ = _make_checkpoints(tmp_path, 3, metadata=metadata)
    out = str(tmp_path / "out.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.9, 0.9, None, out)

    with MemoryEfficientSafeOpen(out) as f:
        merged_metadata = f.metadata()
    assert merged_metadata["ss_network_dim"] == "4"
    assert merged_metadata["sshs_model_hash"] not in ("old", "")
    assert merged_metadata["sshs_legacy_hash"] not in ("old", "")


def test_no_metadata(tmp_path):
    paths, _ = _make_checkpoints(tmp_path, 2)
    out = str(tmp_path / "out.safetensors")
    merge_lora_weights_with_post_hoc_ema(list(paths), False, 0.9, 0.9, None, out)
    with MemoryEfficientSafeOpen(out) as f:
        assert not f.metadata()
