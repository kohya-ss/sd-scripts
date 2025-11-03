"""
Test CDC cache detection with multi-resolution filenames

This test verifies that _check_cdc_caches_exist() correctly detects CDC cache files
that include resolution information in their filenames (e.g., image_flux_cdc_104x80_hash.npz).

This was a bug where the check was looking for files without resolution
(image_flux_cdc_hash.npz) while the actual files had resolution in the name.
"""

import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pytest

from library.train_util import DatasetGroup, ImageInfo
from library.cdc_fm import CDCPreprocessor


class MockDataset:
    """Mock dataset for testing"""
    def __init__(self, image_data):
        self.image_data = image_data
        self.image_dir = "/mock/dataset"
        self.num_train_images = len(image_data)
        self.num_reg_images = 0

    def __len__(self):
        return len(self.image_data)


def test_cdc_cache_detection_with_resolution():
    """
    Test that CDC cache files with resolution in filename are properly detected.

    This reproduces the bug where:
    - CDC files are created with resolution: image_flux_cdc_104x80_hash.npz
    - But check looked for: image_flux_cdc_hash.npz
    - Result: Files not detected, unnecessary regeneration
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        # Setup: Create a mock latent cache file and corresponding CDC cache
        config_hash = "test1234"

        # Create latent cache file with multi-resolution format
        latent_path = Path(tmpdir) / "image_0832x0640_flux.npz"
        latent_shape = (16, 104, 80)  # C, H, W for resolution 832x640 (832/8=104, 640/8=80)

        # Save a mock latent file
        np.savez(
            latent_path,
            **{f"latents_{latent_shape[1]}x{latent_shape[2]}": np.random.randn(*latent_shape).astype(np.float32)}
        )

        # Create the CDC cache file with resolution in filename (as it's actually created)
        cdc_path = CDCPreprocessor.get_cdc_npz_path(
            str(latent_path),
            config_hash,
            latent_shape
        )

        # Verify the CDC path includes resolution
        assert "104x80" in cdc_path, f"CDC path should include resolution: {cdc_path}"

        # Create a mock CDC file
        np.savez(
            cdc_path,
            eigenvectors=np.random.randn(8, 16*104*80).astype(np.float16),
            eigenvalues=np.random.randn(8).astype(np.float16),
            shape=np.array(latent_shape),
            k_neighbors=256,
            d_cdc=8,
            gamma=1.0
        )

        # Setup mock dataset
        image_info = ImageInfo(
            image_key="test_image",
            num_repeats=1,
            caption="test",
            is_reg=False,
            absolute_path=str(Path(tmpdir) / "image.png")
        )
        image_info.latents_npz = str(latent_path)
        image_info.bucket_reso = (640, 832)  # W, H (note: reversed from latent shape H,W)
        image_info.latents = None  # Not in memory

        mock_dataset = MockDataset({"test_image": image_info})
        dataset_group = DatasetGroup([mock_dataset])

        # Test: Check if CDC cache is detected
        result = dataset_group._check_cdc_caches_exist(config_hash)

        # Verify: Should return True since the CDC file exists
        assert result is True, "CDC cache file should be detected when it exists with resolution in filename"


def test_cdc_cache_detection_missing_file():
    """
    Test that missing CDC cache files are correctly identified as missing.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        config_hash = "test5678"

        # Create latent cache file but NO CDC cache
        latent_path = Path(tmpdir) / "image_0768x0512_flux.npz"
        latent_shape = (16, 96, 64)  # C, H, W

        np.savez(
            latent_path,
            **{f"latents_{latent_shape[1]}x{latent_shape[2]}": np.random.randn(*latent_shape).astype(np.float32)}
        )

        # Setup mock dataset (CDC file does NOT exist)
        image_info = ImageInfo(
            image_key="test_image",
            num_repeats=1,
            caption="test",
            is_reg=False,
            absolute_path=str(Path(tmpdir) / "image.png")
        )
        image_info.latents_npz = str(latent_path)
        image_info.bucket_reso = (512, 768)  # W, H
        image_info.latents = None

        mock_dataset = MockDataset({"test_image": image_info})
        dataset_group = DatasetGroup([mock_dataset])

        # Test: Check if CDC cache is detected
        result = dataset_group._check_cdc_caches_exist(config_hash)

        # Verify: Should return False since CDC file doesn't exist
        assert result is False, "Should detect that CDC cache file is missing"


def test_cdc_cache_detection_with_in_memory_latent():
    """
    Test CDC cache detection when latent is already in memory (faster path).
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        config_hash = "test_mem1"

        # Create latent cache file path (file may or may not exist)
        latent_path = Path(tmpdir) / "image_1024x1024_flux.npz"
        latent_shape = (16, 128, 128)  # C, H, W

        # Create the CDC cache file
        cdc_path = CDCPreprocessor.get_cdc_npz_path(
            str(latent_path),
            config_hash,
            latent_shape
        )

        np.savez(
            cdc_path,
            eigenvectors=np.random.randn(8, 16*128*128).astype(np.float16),
            eigenvalues=np.random.randn(8).astype(np.float16),
            shape=np.array(latent_shape),
            k_neighbors=256,
            d_cdc=8,
            gamma=1.0
        )

        # Setup mock dataset with latent in memory
        import torch
        image_info = ImageInfo(
            image_key="test_image",
            num_repeats=1,
            caption="test",
            is_reg=False,
            absolute_path=str(Path(tmpdir) / "image.png")
        )
        image_info.latents_npz = str(latent_path)
        image_info.bucket_reso = (1024, 1024)  # W, H
        image_info.latents = torch.randn(latent_shape)  # In memory!

        mock_dataset = MockDataset({"test_image": image_info})
        dataset_group = DatasetGroup([mock_dataset])

        # Test: Check if CDC cache is detected (should use faster in-memory path)
        result = dataset_group._check_cdc_caches_exist(config_hash)

        # Verify: Should return True
        assert result is True, "CDC cache should be detected using in-memory latent shape"


def test_cdc_cache_detection_partial_cache():
    """
    Test that partial cache (some files exist, some don't) is correctly identified.
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        config_hash = "testpart"

        # Create two latent files
        latent_path1 = Path(tmpdir) / "image1_0640x0512_flux.npz"
        latent_path2 = Path(tmpdir) / "image2_0640x0512_flux.npz"
        latent_shape = (16, 80, 64)

        for latent_path in [latent_path1, latent_path2]:
            np.savez(
                latent_path,
                **{f"latents_{latent_shape[1]}x{latent_shape[2]}": np.random.randn(*latent_shape).astype(np.float32)}
            )

        # Create CDC cache for ONLY the first image
        cdc_path1 = CDCPreprocessor.get_cdc_npz_path(str(latent_path1), config_hash, latent_shape)
        np.savez(
            cdc_path1,
            eigenvectors=np.random.randn(8, 16*80*64).astype(np.float16),
            eigenvalues=np.random.randn(8).astype(np.float16),
            shape=np.array(latent_shape),
            k_neighbors=256,
            d_cdc=8,
            gamma=1.0
        )

        # CDC cache for second image does NOT exist

        # Setup mock dataset with both images
        info1 = ImageInfo("img1", 1, "test", False, str(Path(tmpdir) / "img1.png"))
        info1.latents_npz = str(latent_path1)
        info1.bucket_reso = (512, 640)
        info1.latents = None

        info2 = ImageInfo("img2", 1, "test", False, str(Path(tmpdir) / "img2.png"))
        info2.latents_npz = str(latent_path2)
        info2.bucket_reso = (512, 640)
        info2.latents = None

        mock_dataset = MockDataset({"img1": info1, "img2": info2})
        dataset_group = DatasetGroup([mock_dataset])

        # Test: Check if all CDC caches exist
        result = dataset_group._check_cdc_caches_exist(config_hash)

        # Verify: Should return False since not all files exist
        assert result is False, "Should detect that some CDC cache files are missing"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v"])
