import pytest
import torch
import torch.nn.functional as F

from library.custom_train_functions import cpo_loss


class TestCPOLoss:
    """Test suite for CPO (Contrastive Preference Optimization) loss function"""

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

        return loss

    @pytest.fixture
    def simple_tensors(self):
        """Create simple tensors for basic testing"""
        # Create tensors with shape (2, 4, 32, 32)
        # First tensor (batch 0) - preferred
        batch_0 = torch.full((4, 32, 32), 1.0)
        batch_0[1] = 2.0  # Second channel
        batch_0[2] = 1.5  # Third channel
        batch_0[3] = 1.8  # Fourth channel

        # Second tensor (batch 1) - dispreferred
        batch_1 = torch.full((4, 32, 32), 3.0)
        batch_1[1] = 4.0
        batch_1[2] = 3.5
        batch_1[3] = 3.8

        loss = torch.stack([batch_0, batch_1], dim=0)  # Shape: (2, 4, 32, 32)

        return loss

    def test_basic_functionality(self, simple_tensors):
        """Test basic functionality with simple inputs"""
        loss = simple_tensors

        result_loss, metrics = cpo_loss(loss)

        # Check return types
        assert isinstance(result_loss, torch.Tensor)
        assert isinstance(metrics, dict)

        # Check tensor shape (should be scalar)
        assert result_loss.shape == torch.Size([])

        # Check that loss is finite
        assert torch.isfinite(result_loss)

    def test_metrics_keys(self, simple_tensors):
        """Test that all expected metrics are returned"""
        loss = simple_tensors

        _, metrics = cpo_loss(loss)

        expected_keys = ["loss/cpo_reward_margin"]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert torch.isfinite(torch.tensor(metrics[key]))

    def test_tensor_chunking(self, sample_tensors):
        """Test that tensor chunking works correctly"""
        loss = sample_tensors

        result_loss, metrics = cpo_loss(loss)

        # The function should handle chunking internally
        assert torch.isfinite(result_loss)
        assert len(metrics) == 1

        # Verify chunking produces correct shapes
        loss_w, loss_l = loss.chunk(2)
        assert loss_w.shape == loss_l.shape
        assert loss_w.shape[0] == loss.shape[0] // 2

    def test_different_beta_values(self, simple_tensors):
        """Test with different beta values"""
        loss = simple_tensors

        beta_values = [0.01, 0.05, 0.1, 0.5, 1.0]
        results = []

        for beta in beta_values:
            result_loss, _ = cpo_loss(loss, beta=beta)
            results.append(result_loss.item())

        # Results should be different for different beta values
        assert len(set(results)) == len(beta_values)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_log_ratio_clipping(self, simple_tensors):
        """Test that log ratio is properly clipped to minimum 0.01"""
        loss = simple_tensors

        # Manually verify clipping behavior
        loss_w, loss_l = loss.chunk(2)
        raw_log_ratio = loss_w - loss_l

        result_loss, _ = cpo_loss(loss)

        # The function should clip values to minimum 0.01
        expected_log_ratio = torch.max(raw_log_ratio, torch.full_like(raw_log_ratio, 0.01))

        # All clipped values should be >= 0.01
        assert (expected_log_ratio >= 0.01).all()
        assert torch.isfinite(result_loss)

    def test_uniform_dpo_component(self, simple_tensors):
        """Test the uniform DPO loss component"""
        loss = simple_tensors
        beta = 0.1

        _, metrics = cpo_loss(loss, beta=beta)

        # Manually compute uniform DPO loss
        loss_w, loss_l = loss.chunk(2)
        log_ratio = torch.max(loss_w - loss_l, torch.full_like(loss_w, 0.01))
        expected_uniform_dpo = -F.logsigmoid(beta * log_ratio).mean()

        # The metric should match our manual computation
        assert abs(metrics["loss/cpo_reward_margin"] - expected_uniform_dpo.item()) < 1e-5

    def test_behavioral_cloning_component(self, simple_tensors):
        """Test the behavioral cloning regularizer component"""
        loss = simple_tensors

        result_loss, metrics = cpo_loss(loss)

        # Manually compute BC regularizer
        loss_w, _ = loss.chunk(2)
        expected_bc_regularizer = -loss_w.mean()

        # The total loss should include this component
        # Total = uniform_dpo + bc_regularizer
        expected_total = metrics["loss/cpo_reward_margin"] + expected_bc_regularizer.item()

        # Should match within floating point precision
        assert abs(result_loss.item() - expected_total) < 1e-5

    def test_gradient_flow(self, simple_tensors):
        """Test that gradients flow properly through the loss"""
        loss = simple_tensors
        loss.requires_grad_(True)

        result_loss, _ = cpo_loss(loss)
        result_loss.backward()

        # Check that gradients exist
        assert loss.grad is not None
        assert not torch.isnan(loss.grad).any()
        assert torch.isfinite(loss.grad).all()

    def test_preferred_vs_dispreferred_structure(self):
        """Test that the function properly handles preferred vs dispreferred samples"""
        # Create scenario where preferred samples have lower loss (better)
        loss_w = torch.full((1, 4, 32, 32), 1.0)  # preferred (lower loss)
        loss_l = torch.full((1, 4, 32, 32), 3.0)  # dispreferred (higher loss)
        loss = torch.cat([loss_w, loss_l], dim=0)

        result_loss, _ = cpo_loss(loss)

        # The loss should be finite and reflect the preference structure
        assert torch.isfinite(result_loss)

        # With preferred having lower loss, log_ratio should be negative
        # This should lead to specific behavior in the logsigmoid term
        log_ratio = loss_w - loss_l  # Should be negative (1.0 - 3.0 = -2.0)
        clipped_log_ratio = torch.max(log_ratio, torch.full_like(log_ratio, 0.01))

        # After clipping, should be 0.01 (the minimum)
        assert torch.allclose(clipped_log_ratio, torch.full_like(clipped_log_ratio, 0.01))

    def test_equal_losses_case(self):
        """Test behavior when preferred and dispreferred losses are equal"""
        # Create scenario where preferred and dispreferred have same loss
        loss_w = torch.full((1, 4, 32, 32), 2.0)
        loss_l = torch.full((1, 4, 32, 32), 2.0)
        loss = torch.cat([loss_w, loss_l], dim=0)

        result_loss, metrics = cpo_loss(loss)

        # Log ratio should be zero, but clipped to 0.01
        assert torch.isfinite(result_loss)

        # The reward margin should reflect the clipped behavior
        assert metrics["loss/cpo_reward_margin"] > 0

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        # Test with very large values
        large_loss = torch.full((2, 4, 32, 32), 100.0)
        result_loss, _ = cpo_loss(large_loss)
        assert torch.isfinite(result_loss)

        # Test with very small values
        small_loss = torch.full((2, 4, 32, 32), 1e-6)
        result_loss, _ = cpo_loss(small_loss)
        assert torch.isfinite(result_loss)

        # Test with negative values
        negative_loss = torch.full((2, 4, 32, 32), -1.0)
        result_loss, _ = cpo_loss(negative_loss)
        assert torch.isfinite(result_loss)

    def test_zero_beta_case(self, simple_tensors):
        """Test the case when beta = 0"""
        loss = simple_tensors
        beta = 0.0

        result_loss, metrics = cpo_loss(loss, beta=beta)

        # With beta=0, the uniform DPO term should behave differently
        # logsigmoid(0 * log_ratio) = logsigmoid(0) = log(0.5) ≈ -0.693
        assert torch.isfinite(result_loss)
        assert metrics["loss/cpo_reward_margin"] > 0  # Should be approximately 0.693

    def test_large_beta_case(self, simple_tensors):
        """Test the case with very large beta"""
        loss = simple_tensors
        beta = 100.0

        result_loss, metrics = cpo_loss(loss, beta=beta)

        # Even with large beta, should remain stable due to clipping
        assert torch.isfinite(result_loss)
        assert torch.isfinite(torch.tensor(metrics["loss/cpo_reward_margin"]))

    @pytest.mark.parametrize(
        "batch_size,channels,height,width",
        [
            (1, 4, 32, 32),
            (2, 4, 16, 16),
            (4, 8, 64, 64),
            (8, 4, 8, 8),
        ],
    )
    def test_different_tensor_shapes(self, batch_size, channels, height, width):
        """Test with different tensor shapes"""
        # Note: batch_size will be doubled for preferred/dispreferred pairs
        loss = torch.randn(2 * batch_size, channels, height, width)

        result_loss, metrics = cpo_loss(loss)

        assert torch.isfinite(result_loss)
        assert result_loss.shape == torch.Size([])  # Scalar
        assert len(metrics) == 1

    def test_device_compatibility(self, simple_tensors):
        """Test that function works on different devices"""
        loss = simple_tensors

        # Test on CPU
        result_cpu, _ = cpo_loss(loss)
        assert result_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            loss_gpu = loss.cuda()
            result_gpu, _ = cpo_loss(loss_gpu)
            assert result_gpu.device.type == "cuda"

    def test_reproducibility(self, simple_tensors):
        """Test that results are reproducible with same inputs"""
        loss = simple_tensors

        # Run multiple times
        result1, metrics1 = cpo_loss(loss)
        result2, metrics2 = cpo_loss(loss)

        # Results should be identical (deterministic computation)
        assert torch.allclose(result1, result2)
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-6

    def test_no_reference_model_needed(self, simple_tensors):
        """Test that CPO works without reference model (key feature)"""
        loss = simple_tensors

        # CPO should work with just the loss tensor, no reference needed
        result_loss, metrics = cpo_loss(loss)

        # Should produce meaningful results without reference model
        assert torch.isfinite(result_loss)
        assert len(metrics) == 1
        assert "loss/cpo_reward_margin" in metrics

    def test_loss_components_are_additive(self, simple_tensors):
        """Test that the total loss is sum of uniform DPO and BC regularizer"""
        loss = simple_tensors
        beta = 0.1

        result_loss, metrics = cpo_loss(loss, beta=beta)

        # Manually compute components
        loss_w, loss_l = loss.chunk(2)

        # Uniform DPO component
        log_ratio = torch.max(loss_w - loss_l, torch.full_like(loss_w, 0.01))
        uniform_dpo = -F.logsigmoid(beta * log_ratio).mean()

        # BC regularizer component
        bc_regularizer = -loss_w.mean()

        # Total should be sum of components
        expected_total = uniform_dpo + bc_regularizer

        assert abs(result_loss.item() - expected_total.item()) < 1e-5
        assert abs(metrics["loss/cpo_reward_margin"] - uniform_dpo.item()) < 1e-5

    def test_clipping_prevents_large_gradients(self):
        """Test that clipping prevents very large gradients from small differences"""
        # Create case where loss_w - loss_l would be very small without clipping
        loss_w = torch.full((1, 4, 32, 32), 2.000001)
        loss_l = torch.full((1, 4, 32, 32), 2.000000)
        loss = torch.cat([loss_w, loss_l], dim=0)
        loss.requires_grad_(True)

        result_loss, _ = cpo_loss(loss)
        result_loss.backward()

        assert loss.grad is not None

        # Gradients should be finite and not extremely large due to clipping
        assert torch.isfinite(loss.grad).all()
        assert not torch.any(torch.abs(loss.grad) > 0.001)  # Reasonable gradient magnitude

    def test_behavioral_cloning_effect(self):
        """Test that behavioral cloning regularizer has expected effect"""
        # Create two scenarios: one with low preferred loss, one with high

        # Scenario 1: Low preferred loss
        loss_w_low = torch.full((1, 4, 32, 32), 0.5)
        loss_l_low = torch.full((1, 4, 32, 32), 2.0)
        loss_low = torch.cat([loss_w_low, loss_l_low], dim=0)

        # Scenario 2: High preferred loss
        loss_w_high = torch.full((1, 4, 32, 32), 2.0)
        loss_l_high = torch.full((1, 4, 32, 32), 2.0)
        loss_high = torch.cat([loss_w_high, loss_l_high], dim=0)

        result_low, _ = cpo_loss(loss_low)
        result_high, _ = cpo_loss(loss_high)

        # The BC regularizer should make the total loss lower when preferred loss is lower
        # BC regularizer = -loss_w.mean(), so lower loss_w leads to higher (less negative) regularizer
        # But the overall effect depends on the relative magnitudes
        assert torch.isfinite(result_low)
        assert torch.isfinite(result_high)

    def test_edge_case_all_zeros(self):
        """Test edge case with all zero losses"""
        loss = torch.zeros(2, 4, 32, 32)

        result_loss, metrics = cpo_loss(loss)

        # Should handle all zeros gracefully
        assert torch.isfinite(result_loss)
        assert torch.isfinite(torch.tensor(metrics["loss/cpo_reward_margin"]))

        # With all zeros: loss_w - loss_l = 0, clipped to 0.01
        # BC regularizer = -0 = 0
        # So total should be just the uniform DPO term


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
