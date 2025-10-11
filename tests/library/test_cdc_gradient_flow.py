"""
CDC Gradient Flow Verification Tests

This module provides testing of:
1. Mock dataset gradient preservation
2. Real dataset gradient flow
3. Various time steps and computation paths
4. Fallback and edge case scenarios
"""

import pytest
import torch

from library.cdc_fm import CDCPreprocessor, GammaBDataset
from library.flux_train_utils import apply_cdc_noise_transformation


class MockGammaBDataset:
    """
    Mock implementation of GammaBDataset for testing gradient flow
    """
    def __init__(self, *args, **kwargs):
        """
        Simple initialization that doesn't require file loading
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_sigma_t_x(
        self,
        eigenvectors: torch.Tensor,
        eigenvalues: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified implementation of compute_sigma_t_x for testing
        """
        # Store original shape to restore later
        orig_shape = x.shape

        # Flatten x if it's 4D
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.reshape(B, -1)  # (B, C*H*W)

        # Validate dimensions
        assert eigenvectors.shape[0] == x.shape[0], "Batch size mismatch"
        assert eigenvectors.shape[2] == x.shape[1], "Dimension mismatch"

        # Early return for t=0 with gradient preservation
        if torch.allclose(t, torch.zeros_like(t), atol=1e-8) and not t.requires_grad:
            return x.reshape(orig_shape)

        # Compute Σ_t @ x
        # V^T x
        Vt_x = torch.einsum('bkd,bd->bk', eigenvectors, x)

        # sqrt(λ) * V^T x
        sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=1e-10))
        sqrt_lambda_Vt_x = sqrt_eigenvalues * Vt_x

        # V @ (sqrt(λ) * V^T x)
        gamma_sqrt_x = torch.einsum('bkd,bk->bd', eigenvectors, sqrt_lambda_Vt_x)

        # Interpolate between original and noisy latent
        result = (1 - t) * x + t * gamma_sqrt_x

        # Restore original shape
        result = result.reshape(orig_shape)

        return result


class TestCDCGradientFlow:
    """
    Gradient flow testing for CDC noise transformations
    """

    def setup_method(self):
        """Prepare consistent test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_mock_gradient_flow_near_zero_time_step(self):
        """
        Verify gradient flow preservation for near-zero time steps
        using mock dataset with learnable time embeddings
        """
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Create a learnable time embedding with small initial value
        t = torch.tensor(0.001, requires_grad=True, device=self.device, dtype=torch.float32)

        # Generate mock latent and CDC components
        batch_size, latent_dim = 4, 64
        latent = torch.randn(batch_size, latent_dim, device=self.device, requires_grad=True)

        # Create mock eigenvectors and eigenvalues
        eigenvectors = torch.randn(batch_size, 8, latent_dim, device=self.device)
        eigenvalues = torch.rand(batch_size, 8, device=self.device)

        # Ensure eigenvectors and eigenvalues are meaningful
        eigenvectors /= torch.norm(eigenvectors, dim=-1, keepdim=True)
        eigenvalues = torch.clamp(eigenvalues, min=1e-4, max=1.0)

        # Use the mock dataset
        mock_dataset = MockGammaBDataset()

        # Compute noisy latent with gradient tracking
        noisy_latent = mock_dataset.compute_sigma_t_x(
            eigenvectors,
            eigenvalues,
            latent,
            t
        )

        # Compute a dummy loss to check gradient flow
        loss = noisy_latent.sum()

        # Compute gradients
        loss.backward()

        # Assertions to verify gradient flow
        assert t.grad is not None, "Time embedding gradient should be computed"
        assert latent.grad is not None, "Input latent gradient should be computed"

        # Check gradient magnitudes are non-zero
        t_grad_magnitude = torch.abs(t.grad).sum()
        latent_grad_magnitude = torch.abs(latent.grad).sum()

        assert t_grad_magnitude > 0, f"Time embedding gradient is zero: {t_grad_magnitude}"
        assert latent_grad_magnitude > 0, f"Input latent gradient is zero: {latent_grad_magnitude}"

    def test_gradient_flow_with_multiple_time_steps(self):
        """
        Verify gradient flow across different time step values
        """
        # Test time steps
        time_steps = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0]

        for time_val in time_steps:
            # Create a learnable time embedding
            t = torch.tensor(time_val, requires_grad=True, device=self.device, dtype=torch.float32)

            # Generate mock latent and CDC components
            batch_size, latent_dim = 4, 64
            latent = torch.randn(batch_size, latent_dim, device=self.device, requires_grad=True)

            # Create mock eigenvectors and eigenvalues
            eigenvectors = torch.randn(batch_size, 8, latent_dim, device=self.device)
            eigenvalues = torch.rand(batch_size, 8, device=self.device)

            # Ensure eigenvectors and eigenvalues are meaningful
            eigenvectors /= torch.norm(eigenvectors, dim=-1, keepdim=True)
            eigenvalues = torch.clamp(eigenvalues, min=1e-4, max=1.0)

            # Use the mock dataset
            mock_dataset = MockGammaBDataset()

            # Compute noisy latent with gradient tracking
            noisy_latent = mock_dataset.compute_sigma_t_x(
                eigenvectors,
                eigenvalues,
                latent,
                t
            )

            # Compute a dummy loss to check gradient flow
            loss = noisy_latent.sum()

            # Compute gradients
            loss.backward()

            # Assertions to verify gradient flow
            t_grad_magnitude = torch.abs(t.grad).sum()
            latent_grad_magnitude = torch.abs(latent.grad).sum()

            assert t_grad_magnitude > 0, f"Time embedding gradient is zero for t={time_val}"
            assert latent_grad_magnitude > 0, f"Input latent gradient is zero for t={time_val}"

            # Reset gradients for next iteration
            t.grad.zero_() if t.grad is not None else None
            latent.grad.zero_() if latent.grad is not None else None

    def test_gradient_flow_with_real_dataset(self, tmp_path):
        """
        Test gradient flow with real CDC dataset
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

        cache_path = tmp_path / "test_gradient.safetensors"
        preprocessor.compute_all(save_path=cache_path)
        dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")

        # Prepare test noise
        torch.manual_seed(42)
        noise = torch.randn(4, *shape, dtype=torch.float32, requires_grad=True)
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

        # Verify gradient flow
        assert noise_out.requires_grad, "Output should require gradients"

        loss = noise_out.sum()
        loss.backward()

        assert noise.grad is not None, "Gradients should flow back to input noise"
        assert not torch.isnan(noise.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(noise.grad).any(), "Gradients should not contain inf"
        assert (noise.grad != 0).any(), "Gradients should not be all zeros"

    def test_gradient_flow_with_fallback(self, tmp_path):
        """
        Test gradient flow when using Gaussian fallback (shape mismatch)

        Ensures that cloned tensors maintain gradient flow correctly
        even when shape mismatch triggers Gaussian noise
        """
        # Create cache with one shape
        preprocessor = CDCPreprocessor(
            k_neighbors=8, k_bandwidth=3, d_cdc=8, gamma=1.0, device="cpu"
        )

        preprocessed_shape = (16, 32, 32)
        latent = torch.randn(*preprocessed_shape, dtype=torch.float32)
        metadata = {'image_key': 'test_image_0'}
        preprocessor.add_latent(latent=latent, global_idx=0, shape=preprocessed_shape, metadata=metadata)

        cache_path = tmp_path / "test_fallback_gradient.safetensors"
        preprocessor.compute_all(save_path=cache_path)
        dataset = GammaBDataset(gamma_b_path=cache_path, device="cpu")

        # Use different shape at runtime (will trigger fallback)
        runtime_shape = (16, 64, 64)
        noise = torch.randn(1, *runtime_shape, dtype=torch.float32, requires_grad=True)
        timesteps = torch.tensor([100.0], dtype=torch.float32)
        image_keys = ['test_image_0']

        # Apply transformation (should fallback to Gaussian for this sample)
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
        assert not torch.isnan(noise.grad).any(), "Fallback gradients should not contain NaN"


def pytest_configure(config):
    """
    Configure custom markers for CDC gradient flow tests
    """
    config.addinivalue_line(
        "markers",
        "gradient_flow: mark test to verify gradient preservation in CDC Flow Matching"
    )
    config.addinivalue_line(
        "markers",
        "mock_dataset: mark test using mock dataset for simplified gradient testing"
    )
    config.addinivalue_line(
        "markers",
        "real_dataset: mark test using real dataset for comprehensive gradient testing"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])