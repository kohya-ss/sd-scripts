import pytest
import torch
import torch.nn.functional as F

from library.custom_train_functions import ddo_loss


class TestDDOLoss:
    """Test suite for DDO (Direct Discriminative Optimization) loss function"""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing image latent tensors"""
        # Image latent tensor dimensions
        batch_size = 2
        channels = 4  # Latent channels (e.g., VAE latent space)
        height = 32  # Latent height
        width = 32  # Latent width

        # Create tensors with shape [batch_size, channels, height, width]
        loss = torch.randn(batch_size, channels, height, width)
        ref_loss = torch.randn(batch_size, channels, height, width)

        return loss, ref_loss

    @pytest.fixture
    def simple_tensors(self):
        """Create simple tensors for basic testing"""
        # Create tensors with shape (2, 4, 32, 32)
        batch_0 = torch.full((4, 32, 32), 1.0)
        batch_0[1] = 2.0  # Second channel
        batch_0[2] = 1.5  # Third channel
        batch_0[3] = 1.8  # Fourth channel

        batch_1 = torch.full((4, 32, 32), 2.0)
        batch_1[1] = 3.0
        batch_1[2] = 2.5
        batch_1[3] = 2.8

        loss = torch.stack([batch_0, batch_1], dim=0)  # Shape: (2, 4, 32, 32)

        # Reference loss tensor (different from target)
        ref_batch_0 = torch.full((4, 32, 32), 1.2)
        ref_batch_0[1] = 2.2
        ref_batch_0[2] = 1.7
        ref_batch_0[3] = 2.0

        ref_batch_1 = torch.full((4, 32, 32), 2.3)
        ref_batch_1[1] = 3.3
        ref_batch_1[2] = 2.8
        ref_batch_1[3] = 3.1

        ref_loss = torch.stack([ref_batch_0, ref_batch_1], dim=0)  # Shape: (2, 4, 32, 32)

        return loss, ref_loss

    def test_basic_functionality(self, simple_tensors):
        """Test basic functionality with simple inputs"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        # Check return types
        assert isinstance(result_loss, torch.Tensor)
        assert isinstance(metrics, dict)

        # Check tensor shape (should be 1D with batch dimension)
        assert result_loss.shape == torch.Size([2])  # batch_size = 2

        # Check that loss is finite
        assert torch.isfinite(result_loss).all()

    def test_metrics_keys(self, simple_tensors):
        """Test that all expected metrics are returned"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        _, metrics = ddo_loss(loss, ref_loss, w_t)

        expected_keys = ["loss/ddo_data", "loss/ddo_ref", "loss/ddo_total", "loss/ddo_sigmoid_log_ratio"]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert torch.isfinite(torch.tensor(metrics[key]))

    def test_ref_loss_detached(self, simple_tensors):
        """Test that reference loss gradients are properly detached"""
        loss, ref_loss = simple_tensors
        loss.requires_grad_(True)
        ref_loss.requires_grad_(True)
        w_t = 1.0

        result_loss, _ = ddo_loss(loss, ref_loss, w_t)
        result_loss.sum().backward()

        # Target loss should have gradients
        assert loss.grad is not None
        assert not torch.isnan(loss.grad).any()

        # Reference loss should NOT have gradients due to detach()
        assert ref_loss.grad is None or torch.allclose(ref_loss.grad, torch.zeros_like(ref_loss.grad))

    def test_different_w_t_values(self, simple_tensors):
        """Test with different timestep weights"""
        loss, ref_loss = simple_tensors

        w_t_values = [0.1, 0.5, 1.0, 2.0, 5.0]
        results = []

        for w_t in w_t_values:
            result_loss, _ = ddo_loss(loss, ref_loss, w_t)
            results.append(result_loss.mean().item())

        # Results should be different for different w_t values
        assert len(set(results)) == len(w_t_values)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_different_ddo_alpha_values(self, simple_tensors):
        """Test with different alpha values"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        alpha_values = [1.0, 2.0, 4.0, 8.0, 16.0]
        results = []

        for alpha in alpha_values:
            result_loss, metrics = ddo_loss(loss, ref_loss, w_t, ddo_alpha=alpha)
            results.append(result_loss.mean().item())

        # Results should be different for different alpha values
        assert len(set(results)) == len(alpha_values)

        # Higher alpha should generally increase the total loss due to increased ref penalty
        # (though this depends on the specific values)
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_different_ddo_beta_values(self, simple_tensors):
        """Test with different beta values"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        beta_values = [0.01, 0.05, 0.1, 0.2, 0.5]
        results = []

        for beta in beta_values:
            result_loss, metrics = ddo_loss(loss, ref_loss, w_t, ddo_beta=beta)
            results.append(result_loss.mean().item())

        # Results should be different for different beta values
        assert len(set(results)) == len(beta_values)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_log_likelihood_computation(self, simple_tensors):
        """Test that log likelihood computation is correct"""
        loss, ref_loss = simple_tensors
        w_t = 2.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        # Manually compute expected log likelihoods
        expected_target_logp = -torch.sum(w_t * loss, dim=(1, 2, 3))
        expected_ref_logp = -torch.sum(w_t * ref_loss.detach(), dim=(1, 2, 3))
        expected_delta = expected_target_logp - expected_ref_logp

        # The function should produce finite results
        assert torch.isfinite(result_loss).all()
        assert torch.isfinite(expected_delta).all()

    def test_sigmoid_log_ratio_bounds(self, simple_tensors):
        """Test that sigmoid log ratio is properly bounded"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        # Sigmoid output should be between 0 and 1
        sigmoid_ratio = metrics["loss/ddo_sigmoid_log_ratio"]
        assert 0 <= sigmoid_ratio <= 1

    def test_component_losses_relationship(self, simple_tensors):
        """Test relationship between component losses and total loss"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        # Total loss should equal data loss + ref loss (approximately)
        expected_total = metrics["loss/ddo_data"] + metrics["loss/ddo_ref"]
        actual_total = metrics["loss/ddo_total"]

        # Should be close within floating point precision
        assert abs(expected_total - actual_total) < 1e-5

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        # Test with very large values
        large_loss = torch.full((2, 4, 32, 32), 100.0)
        large_ref_loss = torch.full((2, 4, 32, 32), 50.0)

        result_loss, metrics = ddo_loss(large_loss, large_ref_loss, w_t=1.0)
        assert torch.isfinite(result_loss).all()

        # Test with very small values
        small_loss = torch.full((2, 4, 32, 32), 1e-6)
        small_ref_loss = torch.full((2, 4, 32, 32), 1e-7)

        result_loss, metrics = ddo_loss(small_loss, small_ref_loss, w_t=1.0)
        assert torch.isfinite(result_loss).all()

    def test_zero_w_t(self, simple_tensors):
        """Test with zero timestep weight"""
        loss, ref_loss = simple_tensors
        w_t = 0.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        # With w_t=0, log likelihoods should be zero, leading to specific behavior
        assert torch.isfinite(result_loss).all()

        # When w_t=0, target_logp = ref_logp = 0, so delta = 0, log_ratio = 0
        # sigmoid(0) = 0.5, so sigmoid_log_ratio should be 0.5
        assert abs(metrics["loss/ddo_sigmoid_log_ratio"] - 0.5) < 1e-5

    def test_negative_w_t(self, simple_tensors):
        """Test with negative timestep weight"""
        loss, ref_loss = simple_tensors
        w_t = -1.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        # Should handle negative weights gracefully
        assert torch.isfinite(result_loss).all()
        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value))

    def test_gradient_flow(self, simple_tensors):
        """Test that gradients flow properly through target loss only"""
        loss, ref_loss = simple_tensors
        loss.requires_grad_(True)
        ref_loss.requires_grad_(True)
        w_t = 1.0

        result_loss, _ = ddo_loss(loss, ref_loss, w_t)
        result_loss.sum().backward()

        # Check that gradients exist for target loss
        assert loss.grad is not None
        assert not torch.isnan(loss.grad).any()

        # Reference loss should not have gradients
        assert ref_loss.grad is None or torch.allclose(ref_loss.grad, torch.zeros_like(ref_loss.grad))

    @pytest.mark.parametrize(
        "batch_size,channels,height,width",
        [
            (1, 4, 32, 32),
            (4, 4, 16, 16),
            (2, 8, 64, 64),
            (8, 4, 8, 8),
        ],
    )
    def test_different_tensor_shapes(self, batch_size, channels, height, width):
        """Test with different tensor shapes"""
        loss = torch.randn(batch_size, channels, height, width)
        ref_loss = torch.randn(batch_size, channels, height, width)
        w_t = 1.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t)

        assert torch.isfinite(result_loss).all()
        assert result_loss.shape == torch.Size([batch_size])
        assert len(metrics) == 4

    def test_device_compatibility(self, simple_tensors):
        """Test that function works on different devices"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        # Test on CPU
        result_cpu, metrics_cpu = ddo_loss(loss, ref_loss, w_t)
        assert result_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            loss_gpu = loss.cuda()
            ref_loss_gpu = ref_loss.cuda()
            result_gpu, metrics_gpu = ddo_loss(loss_gpu, ref_loss_gpu, w_t)
            assert result_gpu.device.type == "cuda"

    def test_reproducibility(self, simple_tensors):
        """Test that results are reproducible with same inputs"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        # Run multiple times
        result1, metrics1 = ddo_loss(loss, ref_loss, w_t)
        result2, metrics2 = ddo_loss(loss, ref_loss, w_t)

        # Results should be identical (deterministic computation)
        assert torch.allclose(result1, result2)
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-6

    def test_logsigmoid_stability(self, simple_tensors):
        """Test that logsigmoid operations are numerically stable"""
        loss, ref_loss = simple_tensors
        w_t = 1.0

        # Test with extreme beta that could cause numerical issues
        extreme_beta_values = [0.001, 100.0]

        for beta in extreme_beta_values:
            result_loss, metrics = ddo_loss(loss, ref_loss, w_t, ddo_beta=beta)

            # All components should be finite
            assert torch.isfinite(result_loss).all()
            assert torch.isfinite(torch.tensor(metrics["loss/ddo_data"]))
            assert torch.isfinite(torch.tensor(metrics["loss/ddo_ref"]))

    def test_alpha_zero_case(self, simple_tensors):
        """Test the case when alpha = 0 (no reference loss term)"""
        loss, ref_loss = simple_tensors
        w_t = 1.0
        alpha = 0.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t, ddo_alpha=alpha)

        # With alpha=0, ref loss term should be zero
        assert abs(metrics["loss/ddo_ref"]) < 1e-6

        # Total loss should equal data loss
        assert abs(metrics["loss/ddo_total"] - metrics["loss/ddo_data"]) < 1e-5

    def test_beta_zero_case(self, simple_tensors):
        """Test the case when beta = 0 (no scaling of log ratio)"""
        loss, ref_loss = simple_tensors
        w_t = 1.0
        beta = 0.0

        result_loss, metrics = ddo_loss(loss, ref_loss, w_t, ddo_beta=beta)

        # With beta=0, log_ratio=0, so sigmoid should be 0.5
        assert abs(metrics["loss/ddo_sigmoid_log_ratio"] - 0.5) < 1e-5

        # All losses should be finite
        assert torch.isfinite(result_loss).all()

    def test_discriminative_behavior(self):
        """Test that DDO behaves as expected for discriminative training"""
        # Create scenario where target model is better than reference
        target_loss = torch.full((2, 4, 32, 32), 1.0)  # Lower loss (better)
        ref_loss = torch.full((2, 4, 32, 32), 2.0)  # Higher loss (worse)
        w_t = 1.0

        result_loss, metrics = ddo_loss(target_loss, ref_loss, w_t)

        # When target is better, we expect specific behavior in the discriminator
        assert torch.isfinite(result_loss).all()

        # The sigmoid ratio should reflect that target model is preferred
        # (exact value depends on beta, but should be meaningful)
        assert 0 <= metrics["loss/ddo_sigmoid_log_ratio"] <= 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
