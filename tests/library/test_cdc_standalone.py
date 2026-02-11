"""
Standalone tests for CDC-FM per-file caching.

These tests focus on the current CDC-FM per-file caching implementation
with hash-based cache validation.
"""

from pathlib import Path

import pytest
import torch
import numpy as np

from library.cdc_fm import CDCPreprocessor, GammaBDataset


class TestCDCPreprocessor:
    """Test CDC preprocessing functionality with per-file caching"""

    def test_cdc_preprocessor_basic_workflow(self, tmp_path):
        """Test basic CDC preprocessing with small dataset"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]
        )

        # Add 10 small latents
        for i in range(10):
            latent = torch.randn(16, 4, 4, dtype=torch.float32)  # C, H, W
            latents_npz_path = str(tmp_path / f"test_image_{i}_0004x0004_flux.npz")
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=latent.shape,
                metadata=metadata
            )

        # Compute and save (creates per-file CDC caches)
        files_saved = preprocessor.compute_all()

        # Verify files were created
        assert files_saved == 10

        # Verify first CDC file structure
        latents_npz_path = str(tmp_path / "test_image_0_0004x0004_flux.npz")
        latent_shape = (16, 4, 4)
        cdc_path = Path(CDCPreprocessor.get_cdc_npz_path(latents_npz_path, preprocessor.config_hash, latent_shape))
        assert cdc_path.exists()

        data = np.load(cdc_path)
        assert data['k_neighbors'] == 5
        assert data['d_cdc'] == 4

        # Check eigenvectors and eigenvalues
        eigvecs = data['eigenvectors']
        eigvals = data['eigenvalues']

        assert eigvecs.shape[0] == 4  # d_cdc
        assert eigvals.shape[0] == 4  # d_cdc

    def test_cdc_preprocessor_different_shapes(self, tmp_path):
        """Test CDC preprocessing with variable-size latents (bucketing)"""
        preprocessor = CDCPreprocessor(
            k_neighbors=3, k_bandwidth=2, d_cdc=2, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]
        )

        # Add 5 latents of shape (16, 4, 4)
        for i in range(5):
            latent = torch.randn(16, 4, 4, dtype=torch.float32)
            latents_npz_path = str(tmp_path / f"test_image_{i}_0004x0004_flux.npz")
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=latent.shape,
                metadata=metadata
            )

        # Add 5 latents of different shape (16, 8, 8)
        for i in range(5, 10):
            latent = torch.randn(16, 8, 8, dtype=torch.float32)
            latents_npz_path = str(tmp_path / f"test_image_{i}_0008x0008_flux.npz")
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=latent.shape,
                metadata=metadata
            )

        # Compute and save
        files_saved = preprocessor.compute_all()

        # Verify both shape groups were processed
        assert files_saved == 10

        # Check shapes are stored in individual files
        cdc_path_0 = CDCPreprocessor.get_cdc_npz_path(
            str(tmp_path / "test_image_0_0004x0004_flux.npz"), preprocessor.config_hash, latent_shape=(16, 4, 4)
        )
        cdc_path_5 = CDCPreprocessor.get_cdc_npz_path(
            str(tmp_path / "test_image_5_0008x0008_flux.npz"), preprocessor.config_hash, latent_shape=(16, 8, 8)
        )

        data_0 = np.load(cdc_path_0)
        data_5 = np.load(cdc_path_5)

        assert tuple(data_0['shape']) == (16, 4, 4)
        assert tuple(data_5['shape']) == (16, 8, 8)


class TestGammaBDataset:
    """Test GammaBDataset loading and retrieval with per-file caching"""

    @pytest.fixture
    def sample_cdc_cache(self, tmp_path):
        """Create sample CDC cache files for testing"""
        # Use 20 samples to ensure proper k-NN computation
        # (minimum 256 neighbors recommended, but 20 samples with k=5 is sufficient for testing)
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)],
            adaptive_k=True,  # Enable adaptive k for small dataset
            min_bucket_size=5
        )

        # Create 20 samples
        latents_npz_paths = []
        for i in range(20):
            latent = torch.randn(16, 8, 8, dtype=torch.float32)  # C=16, d=1024 when flattened
            latents_npz_path = str(tmp_path / f"test_{i}_0008x0008_flux.npz")
            latents_npz_paths.append(latents_npz_path)
            metadata = {'image_key': f'test_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=latent.shape,
                metadata=metadata
            )

        preprocessor.compute_all()
        return tmp_path, latents_npz_paths, preprocessor.config_hash

    def test_gamma_b_dataset_loads_metadata(self, sample_cdc_cache):
        """Test that GammaBDataset loads CDC files correctly"""
        tmp_path, latents_npz_paths, config_hash = sample_cdc_cache
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)

        # Get components for first sample
        latent_shape = (16, 8, 8)
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt([latents_npz_paths[0]], device="cpu", latent_shape=latent_shape)

        # Check shapes
        assert eigvecs.shape[0] == 1  # batch size
        assert eigvecs.shape[1] == 4  # d_cdc
        assert eigvals.shape == (1, 4)  # batch, d_cdc

    def test_gamma_b_dataset_get_gamma_b_sqrt(self, sample_cdc_cache):
        """Test retrieving Γ_b^(1/2) components"""
        tmp_path, latents_npz_paths, config_hash = sample_cdc_cache
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)

        # Get Γ_b for paths [0, 2, 4]
        paths = [latents_npz_paths[0], latents_npz_paths[2], latents_npz_paths[4]]
        latent_shape = (16, 8, 8)
        eigenvectors, eigenvalues = gamma_b_dataset.get_gamma_b_sqrt(paths, device="cpu", latent_shape=latent_shape)

        # Check shapes
        assert eigenvectors.shape[0] == 3  # batch
        assert eigenvectors.shape[1] == 4  # d_cdc
        assert eigenvalues.shape == (3, 4)  # (batch, d_cdc)

        # Check values are positive
        assert torch.all(eigenvalues > 0)

    def test_gamma_b_dataset_compute_sigma_t_x_at_t0(self, sample_cdc_cache):
        """Test compute_sigma_t_x returns x unchanged at t=0"""
        tmp_path, latents_npz_paths, config_hash = sample_cdc_cache
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)

        # Create test latents (batch of 3, matching d=1024 flattened)
        x = torch.randn(3, 1024)  # B, d (flattened)
        t = torch.zeros(3)  # t = 0 for all samples

        # Get Γ_b components
        paths = [latents_npz_paths[0], latents_npz_paths[1], latents_npz_paths[2]]
        latent_shape = (16, 8, 8)
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt(paths, device="cpu", latent_shape=latent_shape)

        sigma_t_x = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, x, t)

        # At t=0, should return x unchanged
        assert torch.allclose(sigma_t_x, x, atol=1e-6)

    def test_gamma_b_dataset_compute_sigma_t_x_shape(self, sample_cdc_cache):
        """Test compute_sigma_t_x returns correct shape"""
        tmp_path, latents_npz_paths, config_hash = sample_cdc_cache
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)

        x = torch.randn(2, 1024)  # B, d (flattened)
        t = torch.tensor([0.3, 0.7])

        # Get Γ_b components
        paths = [latents_npz_paths[1], latents_npz_paths[3]]
        latent_shape = (16, 8, 8)
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt(paths, device="cpu", latent_shape=latent_shape)

        sigma_t_x = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, x, t)

        # Should return same shape as input
        assert sigma_t_x.shape == x.shape

    def test_gamma_b_dataset_compute_sigma_t_x_no_nans(self, sample_cdc_cache):
        """Test compute_sigma_t_x produces finite values"""
        tmp_path, latents_npz_paths, config_hash = sample_cdc_cache
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=config_hash)

        x = torch.randn(3, 1024)  # B, d (flattened)
        t = torch.rand(3)  # Random timesteps in [0, 1]

        # Get Γ_b components
        paths = [latents_npz_paths[0], latents_npz_paths[2], latents_npz_paths[4]]
        latent_shape = (16, 8, 8)
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt(paths, device="cpu", latent_shape=latent_shape)

        sigma_t_x = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, x, t)

        # Should not contain NaNs or Infs
        assert not torch.isnan(sigma_t_x).any()
        assert torch.isfinite(sigma_t_x).all()


class TestCDCEndToEnd:
    """End-to-end CDC workflow tests"""

    def test_full_preprocessing_and_usage_workflow(self, tmp_path):
        """Test complete workflow: preprocess -> save -> load -> use"""
        # Step 1: Preprocess latents
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]
        )

        num_samples = 10
        latents_npz_paths = []
        for i in range(num_samples):
            latent = torch.randn(16, 4, 4, dtype=torch.float32)
            latents_npz_path = str(tmp_path / f"test_image_{i}_0004x0004_flux.npz")
            latents_npz_paths.append(latents_npz_path)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=latent.shape,
                metadata=metadata
            )

        files_saved = preprocessor.compute_all()
        assert files_saved == num_samples

        # Step 2: Load with GammaBDataset
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=preprocessor.config_hash)

        # Step 3: Use in mock training scenario
        batch_size = 3
        batch_latents_flat = torch.randn(batch_size, 256)  # B, d (flattened 16*4*4=256)
        batch_t = torch.rand(batch_size)
        paths_batch = [latents_npz_paths[0], latents_npz_paths[5], latents_npz_paths[9]]

        # Get Γ_b components
        latent_shape = (16, 4, 4)
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt(paths_batch, device="cpu", latent_shape=latent_shape)

        # Compute geometry-aware noise
        sigma_t_x = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, batch_latents_flat, batch_t)

        # Verify output is reasonable
        assert sigma_t_x.shape == batch_latents_flat.shape
        assert not torch.isnan(sigma_t_x).any()
        assert torch.isfinite(sigma_t_x).all()

        # Verify that noise changes with different timesteps
        sigma_t0 = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, batch_latents_flat, torch.zeros(batch_size))
        sigma_t1 = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, batch_latents_flat, torch.ones(batch_size))

        # At t=0, should be close to x; at t=1, should be different
        assert torch.allclose(sigma_t0, batch_latents_flat, atol=1e-6)
        assert not torch.allclose(sigma_t1, batch_latents_flat, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
