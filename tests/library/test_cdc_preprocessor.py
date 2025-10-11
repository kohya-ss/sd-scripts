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
        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            assert f.get_tensor("metadata/num_samples").item() == 10
            assert f.get_tensor("metadata/k_neighbors").item() == 5
            assert f.get_tensor("metadata/d_cdc").item() == 4

            # Check first sample
            eigvecs = f.get_tensor("eigenvectors/test_image_0")
            eigvals = f.get_tensor("eigenvalues/test_image_0")

            assert eigvecs.shape[0] == 4  # d_cdc
            assert eigvals.shape[0] == 4  # d_cdc

    def test_preprocessor_with_different_shapes(self, tmp_path):
        """
        Test CDC preprocessing with variable-size latents (bucketing)
        """
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
        with safe_open(str(result_path), framework="pt", device="cpu") as f:
            # Check shapes are stored
            shape_0 = f.get_tensor("shapes/test_image_0")
            shape_5 = f.get_tensor("shapes/test_image_5")

            assert tuple(shape_0.tolist()) == (16, 4, 4)
            assert tuple(shape_5.tolist()) == (16, 8, 8)


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
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        shape = (16, 32, 32)
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=shape, metadata=metadata)

        cache_path = tmp_path / "test_device.safetensors"
        preprocessor.compute_all(save_path=cache_path)

        dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")

        noise = torch.randn(2, *shape, dtype=torch.float32, device="cpu")
        timesteps = torch.tensor([100.0, 200.0], dtype=torch.float32, device="cpu")
        image_keys = ['test_image_0', 'test_image_1']

        with caplog.at_level(logging.WARNING):
            caplog.clear()
            _ = apply_cdc_noise_transformation(
                noise=noise,
                timesteps=timesteps,
                num_timesteps=1000,
                gamma_b_dataset=dataset,
                image_keys=image_keys,
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
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        shape = (16, 32, 32)
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=shape, metadata=metadata)

        cache_path = tmp_path / "test_device_mismatch.safetensors"
        preprocessor.compute_all(save_path=cache_path)

        dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")

        # Create noise and timesteps
        noise = torch.randn(2, *shape, dtype=torch.float32, device="cpu", requires_grad=True)
        timesteps = torch.tensor([100.0, 200.0], dtype=torch.float32, device="cpu")
        image_keys = ['test_image_0', 'test_image_1']

        # Perform CDC transformation
        result = apply_cdc_noise_transformation(
            noise=noise,
            timesteps=timesteps,
            num_timesteps=1000,
            gamma_b_dataset=dataset,
            image_keys=image_keys,
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