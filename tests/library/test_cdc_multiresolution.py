"""
Test CDC-FM multi-resolution support

This test verifies that CDC files are correctly created and loaded for different
resolutions, preventing dimension mismatch errors in multi-resolution training.
"""

import torch
import numpy as np
from pathlib import Path
import pytest

from library.cdc_fm import CDCPreprocessor, GammaBDataset


class TestCDCMultiResolution:
    """Test CDC multi-resolution caching and loading"""

    def test_different_resolutions_create_separate_cdc_files(self, tmp_path):
        """
        Test that the same image with different latent resolutions creates
        separate CDC cache files.
        """
        # Create preprocessor
        preprocessor = CDCPreprocessor(
            k_neighbors=5,
            k_bandwidth=3,
            d_cdc=4,
            gamma=1.0,
            device="cpu",
            dataset_dirs=[str(tmp_path)]
        )

        # Same image, two different resolutions
        image_base_path = str(tmp_path / "test_image_1200x1500_flux.npz")

        # Resolution 1: 64x48 (simulating resolution=512 training)
        latent_64x48 = torch.randn(16, 64, 48, dtype=torch.float32)
        for i in range(10):  # Need multiple samples for CDC
            preprocessor.add_latent(
                latent=latent_64x48,
                global_idx=i,
                latents_npz_path=image_base_path,
                shape=latent_64x48.shape,
                metadata={'image_key': f'test_image_{i}'}
            )

        # Compute and save
        files_saved = preprocessor.compute_all()
        assert files_saved == 10

        # Verify CDC file for 64x48 exists with shape in filename
        cdc_path_64x48 = CDCPreprocessor.get_cdc_npz_path(
            image_base_path,
            preprocessor.config_hash,
            latent_shape=(16, 64, 48)
        )
        assert Path(cdc_path_64x48).exists()
        assert "64x48" in cdc_path_64x48

        # Create new preprocessor for resolution 2
        preprocessor2 = CDCPreprocessor(
            k_neighbors=5,
            k_bandwidth=3,
            d_cdc=4,
            gamma=1.0,
            device="cpu",
            dataset_dirs=[str(tmp_path)]
        )

        # Resolution 2: 104x80 (simulating resolution=768 training)
        latent_104x80 = torch.randn(16, 104, 80, dtype=torch.float32)
        for i in range(10):
            preprocessor2.add_latent(
                latent=latent_104x80,
                global_idx=i,
                latents_npz_path=image_base_path,
                shape=latent_104x80.shape,
                metadata={'image_key': f'test_image_{i}'}
            )

        files_saved2 = preprocessor2.compute_all()
        assert files_saved2 == 10

        # Verify CDC file for 104x80 exists with different shape in filename
        cdc_path_104x80 = CDCPreprocessor.get_cdc_npz_path(
            image_base_path,
            preprocessor2.config_hash,
            latent_shape=(16, 104, 80)
        )
        assert Path(cdc_path_104x80).exists()
        assert "104x80" in cdc_path_104x80

        # Verify both files exist and are different
        assert cdc_path_64x48 != cdc_path_104x80
        assert Path(cdc_path_64x48).exists()
        assert Path(cdc_path_104x80).exists()

        # Verify the CDC files have different dimensions
        data_64x48 = np.load(cdc_path_64x48)
        data_104x80 = np.load(cdc_path_104x80)

        # 64x48 -> flattened dim = 16 * 64 * 48 = 49152
        # 104x80 -> flattened dim = 16 * 104 * 80 = 133120
        assert data_64x48['eigenvectors'].shape[1] == 16 * 64 * 48
        assert data_104x80['eigenvectors'].shape[1] == 16 * 104 * 80

    def test_loading_correct_cdc_for_resolution(self, tmp_path):
        """
        Test that GammaBDataset loads the correct CDC file based on latent_shape
        """
        # Create and save CDC files for two resolutions
        config_hash = "testHash"

        image_path = str(tmp_path / "test_image_flux.npz")

        # Create CDC file for 64x48
        cdc_path_64x48 = CDCPreprocessor.get_cdc_npz_path(
            image_path,
            config_hash,
            latent_shape=(16, 64, 48)
        )
        eigvecs_64x48 = np.random.randn(4, 16 * 64 * 48).astype(np.float16)
        eigvals_64x48 = np.random.randn(4).astype(np.float16)
        np.savez(
            cdc_path_64x48,
            eigenvectors=eigvecs_64x48,
            eigenvalues=eigvals_64x48,
            shape=np.array([16, 64, 48])
        )

        # Create CDC file for 104x80
        cdc_path_104x80 = CDCPreprocessor.get_cdc_npz_path(
            image_path,
            config_hash,
            latent_shape=(16, 104, 80)
        )
        eigvecs_104x80 = np.random.randn(4, 16 * 104 * 80).astype(np.float16)
        eigvals_104x80 = np.random.randn(4).astype(np.float16)
        np.savez(
            cdc_path_104x80,
            eigenvectors=eigvecs_104x80,
            eigenvalues=eigvals_104x80,
            shape=np.array([16, 104, 80])
        )

        # Create GammaBDataset
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)

        # Load with 64x48 shape
        eigvecs_loaded, eigvals_loaded = gamma_b_dataset.get_gamma_b_sqrt(
            [image_path],
            device="cpu",
            latent_shape=(16, 64, 48)
        )
        assert eigvecs_loaded.shape == (1, 4, 16 * 64 * 48)

        # Load with 104x80 shape
        eigvecs_loaded2, eigvals_loaded2 = gamma_b_dataset.get_gamma_b_sqrt(
            [image_path],
            device="cpu",
            latent_shape=(16, 104, 80)
        )
        assert eigvecs_loaded2.shape == (1, 4, 16 * 104 * 80)

        # Verify different dimensions were loaded
        assert eigvecs_loaded.shape[2] != eigvecs_loaded2.shape[2]

    def test_error_when_latent_shape_not_provided_for_multireso(self, tmp_path):
        """
        Test that loading without latent_shape still works for backward compatibility
        but will use old filename format without resolution
        """
        config_hash = "testHash"
        image_path = str(tmp_path / "test_image_flux.npz")

        # Create CDC file with old naming (no latent shape)
        cdc_path_old = CDCPreprocessor.get_cdc_npz_path(
            image_path,
            config_hash,
            latent_shape=None  # Old format
        )
        eigvecs = np.random.randn(4, 16 * 64 * 48).astype(np.float16)
        eigvals = np.random.randn(4).astype(np.float16)
        np.savez(
            cdc_path_old,
            eigenvectors=eigvecs,
            eigenvalues=eigvals,
            shape=np.array([16, 64, 48])
        )

        # Load without latent_shape (backward compatibility)
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)
        eigvecs_loaded, eigvals_loaded = gamma_b_dataset.get_gamma_b_sqrt(
            [image_path],
            device="cpu",
            latent_shape=None
        )
        assert eigvecs_loaded.shape == (1, 4, 16 * 64 * 48)

    def test_filename_format_with_latent_shape(self):
        """Test that CDC filenames include latent dimensions correctly"""
        base_path = "/path/to/image_1200x1500_flux.npz"
        config_hash = "abc123de"

        # With latent shape
        cdc_path = CDCPreprocessor.get_cdc_npz_path(
            base_path,
            config_hash,
            latent_shape=(16, 104, 80)
        )

        # Should include latent H×W in filename
        assert "104x80" in cdc_path
        assert config_hash in cdc_path
        assert cdc_path.endswith("_flux_cdc_104x80_abc123de.npz")

    def test_filename_format_without_latent_shape(self):
        """Test backward compatible filename without latent shape"""
        base_path = "/path/to/image_1200x1500_flux.npz"
        config_hash = "abc123de"

        # Without latent shape (old format)
        cdc_path = CDCPreprocessor.get_cdc_npz_path(
            base_path,
            config_hash,
            latent_shape=None
        )

        # Should NOT include latent dimensions
        assert "104x80" not in cdc_path
        assert "64x48" not in cdc_path
        assert config_hash in cdc_path
        assert cdc_path.endswith("_flux_cdc_abc123de.npz")
