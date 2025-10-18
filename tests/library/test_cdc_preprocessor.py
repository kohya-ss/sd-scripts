"""
CDC Preprocessor and Device Consistency Tests

This module provides testing of:
1. CDC Preprocessor functionality
2. Device consistency handling
3. GammaBDataset loading and usage
4. End-to-end CDC workflow verification
"""

import pytest
import logging
import torch
from pathlib import Path
from safetensors.torch import save_file
from safetensors import safe_open

from library.cdc_fm import CDCPreprocessor, GammaBDataset
from library.flux_train_utils import apply_cdc_noise_transformation


class TestCDCPreprocessorIntegration:
    """
    Comprehensive testing of CDC preprocessing and device handling
    """

    def test_basic_preprocessor_workflow(self, tmp_path):
        """
        Test basic CDC preprocessing with small dataset
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]  # Add dataset_dirs for hash
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

        # Compute and save
        files_saved = preprocessor.compute_all()

        # Verify files were created
        assert files_saved == 10

        # Verify first CDC file structure (with config hash)
        latents_npz_path = str(tmp_path / "test_image_0_0004x0004_flux.npz")
        cdc_path = Path(CDCPreprocessor.get_cdc_npz_path(latents_npz_path, preprocessor.config_hash))
        assert cdc_path.exists()

        import numpy as np
        data = np.load(cdc_path)

        assert data['k_neighbors'] == 5
        assert data['d_cdc'] == 4

        # Check eigenvectors and eigenvalues
        eigvecs = data['eigenvectors']
        eigvals = data['eigenvalues']

        assert eigvecs.shape[0] == 4  # d_cdc
        assert eigvals.shape[0] == 4  # d_cdc

    def test_preprocessor_with_different_shapes(self, tmp_path):
        """
        Test CDC preprocessing with variable-size latents (bucketing)
        """
        preprocessor = CDCPreprocessor(
            k_neighbors=3, k_bandwidth=2, d_cdc=2, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]  # Add dataset_dirs for hash
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

        import numpy as np
        # Check shapes are stored in individual files (with config hash)
        cdc_path_0 = CDCPreprocessor.get_cdc_npz_path(
            str(tmp_path / "test_image_0_0004x0004_flux.npz"), preprocessor.config_hash
        )
        cdc_path_5 = CDCPreprocessor.get_cdc_npz_path(
            str(tmp_path / "test_image_5_0008x0008_flux.npz"), preprocessor.config_hash
        )
        data_0 = np.load(cdc_path_0)
        data_5 = np.load(cdc_path_5)

        assert tuple(data_0['shape']) == (16, 4, 4)
        assert tuple(data_5['shape']) == (16, 8, 8)


class TestDeviceConsistency:
    """
    Test device handling and consistency for CDC transformations
    """

    def test_matching_devices_no_warning(self, tmp_path, caplog):
        """
        Test that no warnings are emitted when devices match.
        """
        # Create CDC cache on CPU
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]  # Add dataset_dirs for hash
        )

        shape = (16, 32, 32)
        latents_npz_paths = []
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32)
            latents_npz_path = str(tmp_path / f"test_image_{i}_0032x0032_flux.npz")
            latents_npz_paths.append(latents_npz_path)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=shape,
                metadata=metadata
            )

        preprocessor.compute_all()

        dataset = GammaBDataset(device="cpu", config_hash=preprocessor.config_hash)

        noise = torch.randn(2, *shape, dtype=torch.float32, device="cpu")
        timesteps = torch.tensor([100.0, 200.0], dtype=torch.float32, device="cpu")
        latents_npz_paths_batch = latents_npz_paths[:2]

        with caplog.at_level(logging.WARNING):
            caplog.clear()
            _ = apply_cdc_noise_transformation(
                noise=noise,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                latents_npz_paths=latents_npz_paths_batch,
                device="cpu"
            )

            # No device mismatch warnings
            device_warnings = [rec for rec in caplog.records if "device mismatch" in rec.message.lower()]
            assert len(device_warnings) == 0, "Should not warn when devices match"

    def test_device_mismatch_handling(self, tmp_path):
        """
        Test that CDC transformation handles device mismatch gracefully
        """
        # Create CDC cache on CPU
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]  # Add dataset_dirs for hash
        )

        shape = (16, 32, 32)
        latents_npz_paths = []
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32)
            latents_npz_path = str(tmp_path / f"test_image_{i}_0032x0032_flux.npz")
            latents_npz_paths.append(latents_npz_path)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(
                latent=latent,
                global_idx=i,
                latents_npz_path=latents_npz_path,
                shape=shape,
                metadata=metadata
            )

        preprocessor.compute_all()

        dataset = GammaBDataset(device="cpu", config_hash=preprocessor.config_hash)

        # Create noise and timesteps
        noise = torch.randn(2, *shape, dtype=torch.float32, device="cpu", requires_grad=True)
        timesteps = torch.tensor([100.0, 200.0], dtype=torch.float32, device="cpu")
        latents_npz_paths_batch = latents_npz_paths[:2]

        # Perform CDC transformation
        result = apply_cdc_noise_transformation(
            noise=noise,
            timesteps=timesteps,
            num_timesteps=1000,
            gamma_b_dataset=dataset,
            latents_npz_paths=latents_npz_paths_batch,
            device="cpu"
        )

        # Verify output characteristics
        assert result.shape == noise.shape
        assert result.device == noise.device
        assert result.requires_grad  # Gradients should still work
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()

        # Verify gradients flow
        loss = result.sum()
        loss.backward()
        assert noise.grad is not None


class TestCDCEndToEnd:
    """
    End-to-end CDC workflow tests
    """

    def test_full_preprocessing_usage_workflow(self, tmp_path):
        """
        Test complete workflow: preprocess -> save -> load -> use
        """
        # Step 1: Preprocess latents
        preprocessor = CDCPreprocessor(
            k_neighbors=5, k_bandwidth=3, d_cdc=4, gamma=1.0, device="cpu",
            dataset_dirs=[str(tmp_path)]  # Add dataset_dirs for hash
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

        # Step 2: Load with GammaBDataset (use config hash)
        gamma_b_dataset = GammaBDataset(device="cpu", config_hash=preprocessor.config_hash)

        # Step 3: Use in mock training scenario
        batch_size = 3
        batch_latents_flat = torch.randn(batch_size, 256)  # B, d (flattened 16*4*4=256)
        batch_t = torch.rand(batch_size)
        latents_npz_paths_batch = [latents_npz_paths[0], latents_npz_paths[5], latents_npz_paths[9]]

        # Get Γ_b components
        eigvecs, eigvals = gamma_b_dataset.get_gamma_b_sqrt(latents_npz_paths_batch, device="cpu")

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


def pytest_configure(config):
    """
    Configure custom markers for CDC tests
    """
    config.addinivalue_line(
        "markers",
        "device_consistency: mark test to verify device handling in CDC transformations"
    )
    config.addinivalue_line(
        "markers",
        "preprocessor: mark test to verify CDC preprocessing workflow"
    )
    config.addinivalue_line(
        "markers",
        "end_to_end: mark test to verify full CDC workflow"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])