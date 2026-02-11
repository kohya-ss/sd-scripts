"""
Test CDC config hash generation and cache invalidation
"""

import pytest
import torch
from pathlib import Path

from library.cdc_fm import CDCPreprocessor


class TestCDCConfigHash:
    """
    Test that CDC config hash properly invalidates cache when dataset or parameters change
    """

    def test_same_config_produces_same_hash(self, tmp_path):
        """
        Test that identical configurations produce identical hashes
        """
        preprocessor1 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        preprocessor2 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        assert preprocessor1.config_hash == preprocessor2.config_hash

    def test_different_dataset_dirs_produce_different_hash(self, tmp_path):
        """
        Test that different dataset directories produce different hashes
        """
        preprocessor1 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        preprocessor2 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset2")]
        )

        assert preprocessor1.config_hash != preprocessor2.config_hash

    def test_different_k_neighbors_produces_different_hash(self, tmp_path):
        """
        Test that different k_neighbors values produce different hashes
        """
        preprocessor1 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        preprocessor2 = CDCPreprocessor(
            k_neighbors=10, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        assert preprocessor1.config_hash != preprocessor2.config_hash

    def test_different_d_cdc_produces_different_hash(self, tmp_path):
        """
        Test that different d_cdc values produce different hashes
        """
        preprocessor1 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        preprocessor2 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=8, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        assert preprocessor1.config_hash != preprocessor2.config_hash

    def test_different_gamma_produces_different_hash(self, tmp_path):
        """
        Test that different gamma values produce different hashes
        """
        preprocessor1 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        preprocessor2 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=2.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        assert preprocessor1.config_hash != preprocessor2.config_hash

    def test_multiple_dataset_dirs_order_independent(self, tmp_path):
        """
        Test that dataset directory order doesn't affect hash (they are sorted)
        """
        preprocessor1 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu",
            dataset_dirs=[str(tmp_path / "dataset1"), str(tmp_path / "dataset2")]
        )

        preprocessor2 = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu",
            dataset_dirs=[str(tmp_path / "dataset2"), str(tmp_path / "dataset1")]
        )

        assert preprocessor1.config_hash == preprocessor2.config_hash

    def test_hash_length_is_8_chars(self, tmp_path):
        """
        Test that hash is exactly 8 characters (hex)
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        assert len(preprocessor.config_hash) == 8
        # Verify it's hex
        int(preprocessor.config_hash, 16)  # Should not raise

    def test_filename_includes_hash(self, tmp_path):
        """
        Test that CDC filenames include the config hash
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0,
            device="cpu", dataset_dirs=[str(tmp_path / "dataset1")]
        )

        latents_path = str(tmp_path / "image_0512x0768_flux.npz")
        cdc_path = CDCPreprocessor.get_cdc_npz_path(latents_path, preprocessor.config_hash)

        # Should be: image_0512x0768_flux_cdc_<hash>.npz
        expected = str(tmp_path / f"image_0512x0768_flux_cdc_{preprocessor.config_hash}.npz")
        assert cdc_path == expected

    def test_backward_compatibility_no_hash(self, tmp_path):
        """
        Test that get_cdc_npz_path works without hash (backward compatibility)
        """
        latents_path = str(tmp_path / "image_0512x0768_flux.npz")
        cdc_path = CDCPreprocessor.get_cdc_npz_path(latents_path, config_hash=None)

        # Should be: image_0512x0768_flux_cdc.npz (no hash suffix)
        expected = str(tmp_path / "image_0512x0768_flux_cdc.npz")
        assert cdc_path == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
