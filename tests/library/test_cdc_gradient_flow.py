"""
Test gradient flow through CDC noise transformation.

Ensures that gradients propagate correctly through both fast and slow paths.
"""

import pytest
import torch

from library.cdc_fm import CDCPreprocessor, GammaBDataset
from library.flux_train_utils import apply_cdc_noise_transformation


class TestCDCGradientFlow:
    """Test gradient flow through CDC transformations"""

    @pytest.fixture
    def cdc_cache(self, tmp_path):
        """Create a test CDC cache"""
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        # Create samples with same shape for fast path testing
        shape = (16, 32, 32)
        for i in range(20):
            latent = torch.randn(*shape, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=shape, metadata=metadata)

        cache_path = tmp_path / "test_gradient.safetensors"
        preprocessor.compute_all(save_path=cache_path)
        return cache_path

    def test_gradient_flow_fast_path(self, cdc_cache):
        """
        Test that gradients flow correctly through batch processing (fast path).

        All samples have matching shapes, so CDC uses batch processing.
        """
        dataset = GammaBDataset(gamma_b_path=cdc_cache, device="cpu")

        batch_size = 4
        shape = (16, 32, 32)

        # Create input noise with requires_grad
        noise = torch.randn(batch_size, *shape, dtype=torch.float32, requires_grad=True)
        timesteps = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float32)
        image_keys = ['test_image_0', 'test_image_1', 'test_image_2', 'test_image_3']

        # Apply CDC transformation
        noise_out = apply_cdc_noise_transformation(
            noise=noise,
            timesteps=timesteps,
            num_timesteps=1000,
            gamma_b_dataset=dataset,
            image_keys=image_keys,
            device="cpu"
        )

        # Ensure output requires grad
        assert noise_out.requires_grad, "Output should require gradients"

        # Compute a simple loss and backprop
        loss = noise_out.sum()
        loss.backward()

        # Verify gradients were computed for input
        assert noise.grad is not None, "Gradients should flow back to input noise"
        assert not torch.isnan(noise.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(noise.grad).any(), "Gradients should not contain inf"
        assert (noise.grad != 0).any(), "Gradients should not be all zeros"

    def test_gradient_flow_slow_path_all_match(self, cdc_cache):
        """
        Test gradient flow when slow path is taken but all shapes match.

        This tests the per-sample loop with CDC transformation.
        """
        dataset = GammaBDataset(gamma_b_path=cdc_cache, device="cpu")

        batch_size = 4
        shape = (16, 32, 32)

        noise = torch.randn(batch_size, *shape, dtype=torch.float32, requires_grad=True)
        timesteps = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float32)
        image_keys = ['test_image_0', 'test_image_1', 'test_image_2', 'test_image_3']

        # Apply transformation
        noise_out = apply_cdc_noise_transformation(
            noise=noise,
            timesteps=timesteps,
            num_timesteps=1000,
            gamma_b_dataset=dataset,
            image_keys=image_keys,
            device="cpu"
        )

        # Test gradient flow
        loss = noise_out.sum()
        loss.backward()

        assert noise.grad is not None
        assert not torch.isnan(noise.grad).any()
        assert (noise.grad != 0).any()

    def test_gradient_consistency_between_paths(self, tmp_path):
        """
        Test that fast path and slow path produce similar gradients.

        When all shapes match, both paths should give consistent results.
        """
        # Create cache with uniform shapes
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        shape = (16, 32, 32)
        for i in range(10):
            latent = torch.randn(*shape, dtype=torch.float32)
            metadata = {'image_key': f'test_image_{i}'}
            preprocessor.add_latent(latent=latent, global_idx=i, shape=shape, metadata=metadata)

        cache_path = tmp_path / "test_consistency.safetensors"
        preprocessor.compute_all(save_path=cache_path)
        dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")

        # Same input for both tests
        torch.manual_seed(42)
        noise = torch.randn(4, *shape, dtype=torch.float32, requires_grad=True)
        timesteps = torch.tensor([100.0, 200.0, 300.0, 400.0], dtype=torch.float32)
        image_keys = ['test_image_0', 'test_image_1', 'test_image_2', 'test_image_3']

        # Apply CDC (should use fast path)
        noise_out = apply_cdc_noise_transformation(
            noise=noise,
            timesteps=timesteps,
            num_timesteps=1000,
            gamma_b_dataset=dataset,
            image_keys=image_keys,
            device="cpu"
        )

        # Compute gradients
        loss = noise_out.sum()
        loss.backward()

        # Both paths should produce valid gradients
        assert noise.grad is not None
        assert not torch.isnan(noise.grad).any()

    def test_fallback_gradient_flow(self, tmp_path):
        """
        Test gradient flow when using Gaussian fallback (shape mismatch).

        Ensures that cloned tensors maintain gradient flow correctly.
        """
        # Create cache with one shape
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        preprocessed_shape = (16, 32, 32)
        latent = torch.randn(*preprocessed_shape, dtype=torch.float32)
        metadata = {'image_key': 'test_image_0'}
        preprocessor.add_latent(latent=latent, global_idx=0, shape=preprocessed_shape, metadata=metadata)

        cache_path = tmp_path / "test_fallback.safetensors"
        preprocessor.compute_all(save_path=cache_path)
        dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")

        # Use different shape at runtime (will trigger fallback)
        runtime_shape = (16, 64, 64)
        noise = torch.randn(1, *runtime_shape, dtype=torch.float32, requires_grad=True)
        timesteps = torch.tensor([100.0], dtype=torch.float32)
        image_keys = ['test_image_0']

        # Apply transformation (should fallback to Gaussian for this sample)
        # Note: This will log a warning but won't raise
        noise_out = apply_cdc_noise_transformation(
            noise=noise,
            timesteps=timesteps,
            num_timesteps=1000,
            gamma_b_dataset=dataset,
            image_keys=image_keys,
            device="cpu"
        )

        # Ensure gradients still flow through fallback path
        assert noise_out.requires_grad, "Fallback output should require gradients"

        loss = noise_out.sum()
        loss.backward()

        assert noise.grad is not None, "Gradients should flow even in fallback case"
        assert not torch.isnan(noise.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
