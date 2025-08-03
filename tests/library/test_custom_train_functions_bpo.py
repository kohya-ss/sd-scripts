import pytest
import torch

from library.custom_train_functions import bpo_loss


class TestBPOLoss:
    """Test suite for BPO loss function"""

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

    @torch.no_grad()
    def test_basic_functionality(self, simple_tensors):
        """Test basic functionality with simple inputs"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.5

        result_loss, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        # Check return types
        assert isinstance(result_loss, torch.Tensor)
        assert isinstance(metrics, dict)

        # Check tensor shape (should be scalar after mean reduction)
        assert result_loss.shape == torch.Size([1])

        # Check that loss is finite
        assert torch.isfinite(result_loss)

    @torch.no_grad()
    def test_metrics_keys(self, simple_tensors):
        """Test that all expected metrics are returned"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.5

        _, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        expected_keys = ["loss/bpo_reward_margin", "loss/bpo_R"]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert torch.isfinite(torch.tensor(metrics[key]))

    @torch.no_grad()
    def test_lambda_zero_case(self, simple_tensors):
        """Test the special case when lambda = 0.0"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.0

        result_loss, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        # Should handle lambda=0 case (R + log(R))
        assert torch.isfinite(result_loss)
        assert "loss/bpo_reward_margin" in metrics
        assert "loss/bpo_R" in metrics

    @torch.no_grad()
    def test_different_beta_values(self, simple_tensors):
        """Test with different beta values"""
        loss, ref_loss = simple_tensors
        lambda_ = 0.5

        beta_values = [0.01, 0.1, 0.5, 1.0]
        results = []

        for beta in beta_values:
            result_loss, _ = bpo_loss(loss, ref_loss, beta, lambda_)
            results.append(result_loss.item())

        # Results should be different for different beta values
        assert len(set(results)) == len(beta_values)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    @torch.no_grad()
    def test_different_lambda_values(self, simple_tensors):
        """Test with different lambda values"""
        loss, ref_loss = simple_tensors
        beta = 0.1

        lambda_values = [0.0, 0.1, 0.5, 1.0, 2.0]
        results = []

        for lambda_ in lambda_values:
            result_loss, _ = bpo_loss(loss, ref_loss, beta, lambda_)
            results.append(result_loss.item())

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    @torch.no_grad()
    def test_r_clipping(self, simple_tensors):
        """Test that R values are properly clipped to minimum 0.01"""
        loss, ref_loss = simple_tensors
        beta = 10.0  # Large beta to potentially create very small R values
        lambda_ = 0.5

        result_loss, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        # R should be >= 0.01 due to clipping
        assert metrics["loss/bpo_R"] >= 0.01
        assert torch.isfinite(result_loss)

    @torch.no_grad()
    def test_tensor_chunking(self, sample_tensors):
        """Test that tensor chunking works correctly"""
        loss, ref_loss = sample_tensors
        beta = 0.1
        lambda_ = 0.5

        result_loss, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        # The function should handle chunking internally
        assert torch.isfinite(result_loss)
        assert len(metrics) == 2

    def test_gradient_flow(self, simple_tensors):
        """Test that gradients can flow through the loss"""
        loss, ref_loss = simple_tensors
        loss.requires_grad_(True)
        ref_loss.requires_grad_(True)
        beta = 0.1
        lambda_ = 0.5

        result_loss, _ = bpo_loss(loss, ref_loss, beta, lambda_)
        result_loss.backward()

        # Check that gradients exist
        assert loss.grad is not None
        assert ref_loss.grad is not None
        assert not torch.isnan(loss.grad).any()
        assert not torch.isnan(ref_loss.grad).any()

    @torch.no_grad()
    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        # Test with very large values
        large_loss = torch.full((2, 4, 32, 32), 100.0)
        large_ref_loss = torch.full((2, 4, 32, 32), 50.0)

        result_loss, _ = bpo_loss(large_loss, large_ref_loss, beta=0.1, lambda_=0.5)
        assert torch.isfinite(result_loss)

        # Test with very small values
        small_loss = torch.full((2, 4, 32, 32), 1e-6)
        small_ref_loss = torch.full((2, 4, 32, 32), 1e-7)

        result_loss, _ = bpo_loss(small_loss, small_ref_loss, beta=0.1, lambda_=0.5)
        assert torch.isfinite(result_loss)

    @torch.no_grad()
    def test_negative_lambda_values(self, simple_tensors):
        """Test with negative lambda values"""
        loss, ref_loss = simple_tensors
        beta = 0.1

        # Test some negative lambda values
        lambda_values = [-0.5, -0.1, -0.9]

        for lambda_ in lambda_values:
            # Skip lambda = -1 as it causes division by zero
            if lambda_ != -1.0:
                result_loss, _ = bpo_loss(loss, ref_loss, beta, lambda_)
                assert torch.isfinite(result_loss)

    @torch.no_grad()
    def test_edge_case_lambda_near_negative_one(self, simple_tensors):
        """Test edge case near lambda = -1"""
        loss, ref_loss = simple_tensors
        beta = 0.1

        # Test values close to -1 but not exactly -1
        lambda_values = [-0.99, -0.999]

        for lambda_ in lambda_values:
            result_loss, _ = bpo_loss(loss, ref_loss, beta, lambda_)
            # Should still be finite even though close to the problematic value
            assert torch.isfinite(result_loss)

    @torch.no_grad()
    def test_asymmetric_preference_structure(self):
        """Test that the function properly handles preferred vs dispreferred samples"""
        # Create scenario where preferred samples have lower loss
        loss_w = torch.full((1, 4, 32, 32), 1.0)  # preferred (lower loss)
        loss_l = torch.full((1, 4, 32, 32), 3.0)  # dispreferred (higher loss)
        loss = torch.cat([loss_w, loss_l], dim=0)

        ref_loss_w = torch.full((1, 4, 32, 32), 2.0)
        ref_loss_l = torch.full((1, 4, 32, 32), 2.0)
        ref_loss = torch.cat([ref_loss_w, ref_loss_l], dim=0)

        result_loss, metrics = bpo_loss(loss, ref_loss, beta=0.1, lambda_=0.5)

        # The loss should be finite and reflect the preference structure
        assert torch.isfinite(result_loss)

        # The reward margin should reflect the preference (preferred - dispreferred)
        # In this case: (1-3) - (2-2) = -2, so reward_margin should be negative
        assert metrics["loss/bpo_reward_margin"] < 0

    @pytest.mark.parametrize(
        "batch_size,channels,height,width",
        [
            (2, 4, 32, 32),
            (2, 4, 16, 16),
            (2, 8, 64, 64),
        ],
    )
    @torch.no_grad()
    def test_different_tensor_shapes(self, batch_size, channels, height, width):
        """Test with different tensor shapes"""
        loss = torch.randn(2 * batch_size, channels, height, width)
        ref_loss = torch.randn(2 * batch_size, channels, height, width)

        result_loss, metrics = bpo_loss(loss, ref_loss, beta=0.1, lambda_=0.5)

        assert torch.isfinite(result_loss.mean())
        assert result_loss.shape == torch.Size([2])
        assert len(metrics) == 2

    def test_device_compatibility(self, simple_tensors):
        """Test that function works on different devices"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.5

        # Test on CPU
        result_cpu, _ = bpo_loss(loss, ref_loss, beta, lambda_)
        assert result_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            loss_gpu = loss.cuda()
            ref_loss_gpu = ref_loss.cuda()
            result_gpu, _ = bpo_loss(loss_gpu, ref_loss_gpu, beta, lambda_)
            assert result_gpu.device.type == "cuda"

    @torch.no_grad()
    def test_reproducibility(self, simple_tensors):
        """Test that results are reproducible with same inputs"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.5

        # Run multiple times with same seed
        torch.manual_seed(42)
        result1, metrics1 = bpo_loss(loss, ref_loss, beta, lambda_)

        torch.manual_seed(42)
        result2, metrics2 = bpo_loss(loss, ref_loss, beta, lambda_)

        # Results should be identical
        assert torch.allclose(result1, result2)
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-6

    @torch.no_grad()
    def test_zero_inputs(self):
        """Test with zero inputs"""
        zero_loss = torch.zeros(2, 4, 32, 32)
        zero_ref_loss = torch.zeros(2, 4, 32, 32)

        result_loss, metrics = bpo_loss(zero_loss, zero_ref_loss, beta=0.1, lambda_=0.5)

        # Should handle zero inputs gracefully
        assert torch.isfinite(result_loss)
        for value in metrics.values():
            assert torch.isfinite(torch.tensor(value))

    @torch.no_grad()
    def test_reward_margin_computation(self, simple_tensors):
        """Test that reward margin is computed correctly"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.5

        _, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        # Manually compute expected reward margin
        loss_w, loss_l = loss.chunk(2)
        ref_loss_w, ref_loss_l = ref_loss.chunk(2)
        expected_logits = loss_w - loss_l - ref_loss_w + ref_loss_l
        expected_reward_margin = beta * expected_logits

        # Compare with returned metric (within floating point precision)
        assert abs(metrics["loss/bpo_reward_margin"] - expected_reward_margin.mean().item()) < 1e-5

    @torch.no_grad()
    def test_r_value_computation(self, simple_tensors):
        """Test that R values are computed correctly"""
        loss, ref_loss = simple_tensors
        beta = 0.1
        lambda_ = 0.5

        _, metrics = bpo_loss(loss, ref_loss, beta, lambda_)

        # R should be positive and >= 0.01 due to clipping
        assert metrics["loss/bpo_R"] > 0
        assert metrics["loss/bpo_R"] >= 0.01


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
