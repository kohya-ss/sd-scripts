"""
Standalone tests for CDC-FM integration.

These tests focus on CDC-FM specific functionality without importing
the full training infrastructure that has problematic dependencies.
"""

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from library.cdc_fm import CDCPreprocessor, GammaBDataset


class TestCDCPreprocessor:
    """Test CDC preprocessing functionality"""

    def test_cdc_preprocessor_basic_workflow(self, tmp_path):
        """Test basic CDC preprocessing with small dataset"""
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        # Add 10 small latents
        for i in range(10):
            latent = torch.randn(16, 4, 4, dtype=torch.float32)  # C, H, W
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        # Compute and save
        output_path = tmp_path / "test_gamma_b.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        # Verify file was created
        assert Path(result_path).exists()

        # Verify structure
        from safetensors import safe_open

        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            assert f.get_tensor("metadata/num_samples").item() == 10
            assert f.get_tensor("metadata/k_neighbors").item() == 5
            assert f.get_tensor("metadata/d_cdc").item() == 4

            # Check first sample
            eigvecs = f.get_tensor("eigenvectors/test_image_0")
            eigvals = f.get_tensor("eigenvalues/test_image_0")

            assert eigvecs.shape[0] == 4  # d_cdc
            assert eigvals.shape[0] == 4  # d_cdc

    def test_cdc_preprocessor_different_shapes(self, tmp_path):
        """Test CDC preprocessing with variable-size latents (bucketing)"""
        preprocessor = CDCPreprocessor(
            k_neighbors=3, k_bandwidth=2, d_cdc=2, gamma=1.0, device="cpu"
        )

        # Add 5 latents of shape (16, 4, 4)
        for i in range(5):
            latent = torch.randn(16, 4, 4, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        # Add 5 latents of different shape (16, 8, 8)
        for i in range(5, 10):
            latent = torch.randn(16, 8, 8, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        # Compute and save
        output_path = tmp_path / "test_gamma_b_multi.safetensors"
        result_path = preprocessor.compute_all(save_path=output_path)

        # Verify both shape groups were processed
        from safetensors import safe_open

        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            # Check shapes are stored
            shape_0 = f.get_tensor("shapes/test_image_0")
            shape_5 = f.get_tensor("shapes/test_image_5")

            assert tuple(shape_0.tolist()) == (16, 4, 4)
            assert tuple(shape_5.tolist()) == (16, 8, 8)


class TestGammaBDataset:
    """Test GammaBDataset loading and retrieval"""

    @pytest.fixture
    def sample_cdc_cache(self, tmp_path):
        """Create a sample CDC cache file for testing"""
        cache_path = tmp_path / "test_gamma_b.safetensors"

        # Create mock Γ_b data for 5 samples
        tensors = {
            "metadata/num_samples": torch.tensor([5]),
            "metadata/k_neighbors": torch.tensor([10]),
            "metadata/d_cdc": torch.tensor([4]),
            "metadata/gamma": torch.tensor([1.0]),
        }

        # Add shape and CDC data for each sample
        for i in range(5):
            tensors[f"shapes/{i}"] = torch.tensor([16, 8, 8])  # C, H, W
            tensors[f"eigenvectors/{i}"] = torch.randn(4, 1024, dtype=torch.float32)  # d_cdc x d
            tensors[f"eigenvalues/{i}"] = torch.rand(4, dtype=torch.float32) + 0.1  # positive

        save_file(tensors, str(cache_path))
        return cache_path

    def test_gamma_b_dataset_loads_metadata(self, sample_cdc_cache):
        """Test that GammaBDataset loads metadata correctly"""
        gamma_b_dataset = GammaBDataset(gamma_b_path=sample_cdc_cache, device="cpu")

        assert gamma_b_dataset.num_samples == 5
        assert gamma_b_dataset.d_cdc == 4

    def test_gamma_b_dataset_get_gamma_b_sqrt(self, sample_cdc_cache):
        """Test retrieving Γ_b^(1/2) components"""
        gamma_b_dataset = GammaBDataset(gamma_b_path=sample_cdc_cache, device="cpu")

        # Get Γ_b for indices [0, 2, 4]
        indices = [0, 2, 4]
        eigenvectors, eigenvalues = gamma_b_dataset.get_gamma_b_sqrt(indices, device="cpu")

        # Check shapes
        assert eigenvectors.shape == (3, 4, 1024)  # (batch, d_cdc, d)
        assert eigenvalues.shape == (3, 4)  # (batch, d_cdc)

        # Check values are positive
        assert torch.all(eigenvalues > 0)

    def test_gamma_b_dataset_compute_sigma_t_x_at_t0(self, sample_cdc_cache):
        """Test compute_sigma_t_x returns x unchanged at t=0"""
        gamma_b_dataset = GammaBDataset(gamma_b_path=sample_cdc_cache, device="cpu")

        # Create test latents (batch of 3, matching d=1024 flattened)
        x = torch.randn(3, 1024)  # B, d (flattened)
        t = torch.zeros(3)  # t = 0 for all samples

        # Get Γ_b components
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt([0, 1, 2], device="cpu")

        sigma_t_x = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, x, t)

        # At t=0, should return x unchanged
        assert torch.allclose(sigma_t_x, x, atol=1e-6)

    def test_gamma_b_dataset_compute_sigma_t_x_shape(self, sample_cdc_cache):
        """Test compute_sigma_t_x returns correct shape"""
        gamma_b_dataset = GammaBDataset(gamma_b_path=sample_cdc_cache, device="cpu")

        x = torch.randn(2, 1024)  # B, d (flattened)
        t = torch.tensor([0.3, 0.7])

        # Get Γ_b components
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt([1, 3], device="cpu")

        sigma_t_x = gamma_b_dataset.compute_sigma_t_x(eigvecs, eigvals, x, t)

        # Should return same shape as input
        assert sigma_t_x.shape == x.shape

    def test_gamma_b_dataset_compute_sigma_t_x_no_nans(self, sample_cdc_cache):
        """Test compute_sigma_t_x produces finite values"""
        gamma_b_dataset = GammaBDataset(gamma_b_path=sample_cdc_cache, device="cpu")

        x = torch.randn(3, 1024)  # B, d (flattened)
        t = torch.rand(3)  # Random timesteps in [0, 1]

        # Get Γ_b components
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt([0, 2, 4], device="cpu")

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
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu"
        )

        num_samples = 10
        for i in range(num_samples):
            latent = torch.randn(16, 4, 4, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=latent.shape, metadata=metadata)

        output_path = tmp_path / "cdc_gamma_b.safetensors"
        cdc_path = preprocessor.compute_all(save_path=output_path)

        # Step 2: Load with GammaBDataset
        gamma_b_dataset = GammaBDataset(gamma_b_path=cdc_path, device="cpu")

        assert gamma_b_dataset.num_samples == num_samples

        # Step 3: Use in mock training scenario
        batch_size = 3
        batch_latents_flat = torch.randn(batch_size, 256)  # B, d (flattened 16*4*4=256)
        batch_t = torch.rand(batch_size)
        image_keys = ['test_image_0', 'test_image_5', 'test_image_9']

        # Get Γ_b components
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt(image_keys, device="cpu")

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
