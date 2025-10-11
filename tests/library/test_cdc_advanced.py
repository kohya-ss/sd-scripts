import torch
from typing import Union


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
        t: Union[float, torch.Tensor]
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

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

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

class TestCDCAdvanced:
    def setup_method(self):
        """Prepare consistent test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_gradient_flow_preservation(self):
        """
        Verify that gradient flow is preserved even for near-zero time steps
        with learnable time embeddings
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

        # Optional: Print gradient details for debugging
        print(f"Time embedding gradient magnitude: {t_grad_magnitude}")
        print(f"Latent gradient magnitude: {latent_grad_magnitude}")

    def test_gradient_flow_with_different_time_steps(self):
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
            if t.grad is not None:
                t.grad.zero_()
            if latent.grad is not None:
                latent.grad.zero_()

def pytest_configure(config):
    """
    Add custom markers for CDC-FM tests
    """
    config.addinivalue_line(
        "markers",
        "gradient_flow: mark test to verify gradient preservation in CDC Flow Matching"
    )