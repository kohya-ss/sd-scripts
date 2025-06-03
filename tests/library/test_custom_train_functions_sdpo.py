import pytest
import torch

from library.custom_train_functions import sdpo_loss


class TestSDPOLoss:
    """Test suite for SDPO loss function"""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing image latent tensors"""
        # Image latent tensor dimensions
        batch_size = 1  # Will be doubled to 2 for preferred/dispreferred pairs
        channels = 4  # Latent channels (e.g., VAE latent space)
        height = 32  # Latent height
        width = 32  # Latent width

        # Create tensors with shape [2*batch_size, channels, height, width]
        # First half represents preferred (w), second half dispreferred (l)
        loss = torch.randn(2 * batch_size, channels, height, width)
        ref_loss = torch.randn(2 * batch_size, channels, height, width)

        return loss, ref_loss

    @pytest.fixture
    def simple_tensors(self):
        """Create simple tensors for basic testing"""
        # Create tensors with shape (2, 4, 32, 32)
        # First tensor (batch 0)
        batch_0 = torch.full((4, 32, 32), 1.0)
        batch_0[1] = 2.0  # Second channel
        batch_0[2] = 2.0  # Third channel
        batch_0[3] = 3.0  # Fourth channel

        # Second tensor (batch 1)
        batch_1 = torch.full((4, 32, 32), 3.0)
        batch_1[1] = 4.0
        batch_1[2] = 5.0
        batch_1[3] = 2.0

        loss = torch.stack([batch_0, batch_1], dim=0)  # Shape: (2, 4, 32, 32)

        # Reference loss tensor
        ref_batch_0 = torch.full((4, 32, 32), 0.5)
        ref_batch_0[1] = 1.5
        ref_batch_0[2] = 3.5
        ref_batch_0[3] = 9.5

        ref_batch_1 = torch.full((4, 32, 32), 2.5)
        ref_batch_1[1] = 3.5
        ref_batch_1[2] = 4.5
        ref_batch_1[3] = 3.5

        ref_loss = torch.stack([ref_batch_0, ref_batch_1], dim=0)  # Shape: (2, 4, 32, 32)

        return loss, ref_loss

    def test_basic_functionality(self, simple_tensors):
        """Test basic functionality with simple inputs"""
        loss, ref_loss = simple_tensors

        print(loss.shape, ref_loss.shape)

        result_loss, metrics = sdpo_loss(loss, ref_loss)

        # Check return types
        assert isinstance(result_loss, torch.Tensor)
        assert isinstance(metrics, dict)

        # Check tensor shape (should be scalar after mean reduction)
        assert result_loss.shape == torch.Size([1])

        # Check that loss is finite and positive
        assert torch.isfinite(result_loss)
        assert result_loss >= 0

    def test_metrics_keys(self, simple_tensors):
        """Test that all expected metrics are returned"""
        loss, ref_loss = simple_tensors

        _, metrics = sdpo_loss(loss, ref_loss)

        expected_keys = [
            "loss/sdpo_log_ratio_w",
            "loss/sdpo_log_ratio_l",
            "loss/sdpo_w_theta_max",
            "loss/sdpo_w_theta_w",
            "loss/sdpo_w_theta_l",
        ]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert not torch.isnan(torch.tensor(metrics[key]))

    def test_different_beta_values(self, simple_tensors):
        """Test with different beta values"""
        loss, ref_loss = simple_tensors

        print(loss.shape, ref_loss.shape)

        beta_values = [0.01, 0.02, 0.05, 0.1]
        results = []

        for beta in beta_values:
            result_loss, _ = sdpo_loss(loss, ref_loss, beta=beta)
            results.append(result_loss.item())

        # Results should be different for different beta values
        assert len(set(results)) == len(beta_values)

    def test_different_epsilon_values(self, simple_tensors):
        """Test with different epsilon values"""
        loss, ref_loss = simple_tensors

        epsilon_values = [0.05, 0.1, 0.2, 0.5]
        results = []

        for epsilon in epsilon_values:
            result_loss, _ = sdpo_loss(loss, ref_loss, epsilon=epsilon)
            results.append(result_loss.item())

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_tensor_chunking(self, sample_tensors):
        """Test that tensor chunking works correctly"""
        loss, ref_loss = sample_tensors

        result_loss, metrics = sdpo_loss(loss, ref_loss)

        # The function should handle chunking internally
        assert torch.isfinite(result_loss)
        assert len(metrics) == 5

    def test_gradient_flow(self, simple_tensors):
        """Test that gradients can flow through the loss"""
        loss, ref_loss = simple_tensors
        loss.requires_grad_(True)
        ref_loss.requires_grad_(True)

        result_loss, _ = sdpo_loss(loss, ref_loss)
        result_loss.backward()

        # Check that gradients exist
        assert loss.grad is not None
        assert ref_loss.grad is not None
        assert not torch.isnan(loss.grad).any()
        assert not torch.isnan(ref_loss.grad).any()

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Test with very large values
        large_loss = torch.full((4, 2, 32, 32), 100.0)
        large_ref_loss = torch.full((4, 2, 32, 32), 50.0)

        result_loss, metrics = sdpo_loss(large_loss, large_ref_loss)
        assert torch.isfinite(result_loss.mean())

        # Test with very small values
        small_loss = torch.full((4, 2, 32, 32), 1e-6)
        small_ref_loss = torch.full((4, 2, 32, 32), 1e-7)

        result_loss, metrics = sdpo_loss(small_loss, small_ref_loss)
        assert torch.isfinite(result_loss.mean())

    def test_zero_inputs(self):
        """Test with zero inputs"""
        zero_loss = torch.zeros(4, 2, 32, 32)
        zero_ref_loss = torch.zeros(4, 2, 32, 32)

        result_loss, metrics = sdpo_loss(zero_loss, zero_ref_loss)

        # Should handle zero inputs gracefully
        assert torch.isfinite(result_loss.mean())
        for key, value in metrics.items():
            assert torch.isfinite(torch.tensor(value))

    def test_asymmetric_preference(self):
        """Test that the function properly handles preferred vs dispreferred samples"""
        # Create scenario where preferred samples have lower loss
        loss_w = torch.tensor([[[[1.0, 1.0]]]])  # preferred (lower loss)
        loss_l = torch.tensor([[[[2.0, 3.0]]]])  # dispreferred (higher loss)
        loss = torch.cat([loss_w, loss_l], dim=0)

        ref_loss_w = torch.tensor([[[[2.0, 2.0]]]])
        ref_loss_l = torch.tensor([[[[2.0, 2.0]]]])
        ref_loss = torch.cat([ref_loss_w, ref_loss_l], dim=0)

        result_loss, metrics = sdpo_loss(loss, ref_loss)

        # The loss should be finite and reflect the preference structure
        assert torch.isfinite(result_loss)
        assert result_loss >= 0

        # Log ratios should reflect the preference structure
        assert metrics["loss/sdpo_log_ratio_w"] > metrics["loss/sdpo_log_ratio_l"]

    @pytest.mark.parametrize(
        "batch_size,channel,height,width",
        [
            (2, 4, 16, 16),
            (8, 16, 32, 32),
            (4, 4, 16, 16),
        ],
    )
    def test_different_tensor_shapes(self, batch_size, channel, height, width):
        """Test with different tensor shapes"""
        loss = torch.randn(2 * batch_size, channel, height, width)
        ref_loss = torch.randn(2 * batch_size, channel, height, width)

        result_loss, metrics = sdpo_loss(loss, ref_loss)

        assert torch.isfinite(result_loss.mean())
        assert result_loss.shape == torch.Size([batch_size])
        assert len(metrics) == 5

    def test_device_compatibility(self, simple_tensors):
        """Test that function works on different devices"""
        loss, ref_loss = simple_tensors

        # Test on CPU
        result_cpu, metrics_cpu = sdpo_loss(loss, ref_loss)
        assert result_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            loss_gpu = loss.cuda()
            ref_loss_gpu = ref_loss.cuda()
            result_gpu, metrics_gpu = sdpo_loss(loss_gpu, ref_loss_gpu)
            assert result_gpu.device.type == "cuda"

    def test_reproducibility(self, simple_tensors):
        """Test that results are reproducible with same inputs"""
        loss, ref_loss = simple_tensors

        # Run multiple times with same seed
        torch.manual_seed(42)
        result1, metrics1 = sdpo_loss(loss, ref_loss)

        torch.manual_seed(42)
        result2, metrics2 = sdpo_loss(loss, ref_loss)

        # Results should be identical
        assert torch.allclose(result1, result2)
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-6


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
