import pytest
import torch
import torch.nn.functional as F

from library.custom_train_functions import simpo_loss


class TestSimPOLoss:
    """Test suite for SimPO (Simple Preference Optimization) loss function"""

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
        # First tensor (batch 0) - preferred (lower loss is better)
        batch_0 = torch.full((4, 32, 32), 1.0)
        batch_0[1] = 0.8
        batch_0[2] = 1.2
        batch_0[3] = 0.9

        # Second tensor (batch 1) - dispreferred (higher loss)
        batch_1 = torch.full((4, 32, 32), 2.5)
        batch_1[1] = 2.8
        batch_1[2] = 2.2
        batch_1[3] = 2.7

        loss = torch.stack([batch_0, batch_1], dim=0)  # Shape: (2, 4, 32, 32)

        return loss

    def test_basic_functionality_sigmoid(self, simple_tensors):
        """Test basic functionality with sigmoid loss type"""
        loss = simple_tensors

        result_losses, metrics = simpo_loss(loss, loss_type="sigmoid")

        # Check return types
        assert isinstance(result_losses, torch.Tensor)
        assert isinstance(metrics, dict)

        # Check tensor shape (should match input preferred/dispreferred batch size)
        loss_w, _ = loss.chunk(2)
        assert result_losses.shape == loss_w.shape

        # Check that losses are finite
        assert torch.isfinite(result_losses).all()

    def test_basic_functionality_hinge(self, simple_tensors):
        """Test basic functionality with hinge loss type"""
        loss = simple_tensors

        result_losses, metrics = simpo_loss(loss, loss_type="hinge")

        # Check return types
        assert isinstance(result_losses, torch.Tensor)
        assert isinstance(metrics, dict)

        # Check tensor shape
        loss_w, _ = loss.chunk(2)
        assert result_losses.shape == loss_w.shape

        # Check that losses are finite and non-negative (ReLU property)
        assert torch.isfinite(result_losses).all()
        assert (result_losses >= 0).all()

    def test_metrics_keys(self, simple_tensors):
        """Test that all expected metrics are returned"""
        loss = simple_tensors

        _, metrics = simpo_loss(loss)

        expected_keys = ["loss/simpo_chosen_rewards", "loss/simpo_rejected_rewards", "loss/simpo_logratio"]

        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float))
            assert torch.isfinite(torch.tensor(metrics[key]))

    def test_loss_type_parameter(self, simple_tensors):
        """Test different loss types produce different results"""
        loss = simple_tensors

        sigmoid_losses, sigmoid_metrics = simpo_loss(loss, loss_type="sigmoid")
        hinge_losses, hinge_metrics = simpo_loss(loss, loss_type="hinge")

        # Results should be different
        assert not torch.allclose(sigmoid_losses, hinge_losses)

        # But metrics should be the same (they don't depend on loss type)
        assert sigmoid_metrics["loss/simpo_chosen_rewards"] == hinge_metrics["loss/simpo_chosen_rewards"]
        assert sigmoid_metrics["loss/simpo_rejected_rewards"] == hinge_metrics["loss/simpo_rejected_rewards"]
        assert sigmoid_metrics["loss/simpo_logratio"] == hinge_metrics["loss/simpo_logratio"]

    def test_invalid_loss_type(self, simple_tensors):
        """Test that invalid loss type raises ValueError"""
        loss = simple_tensors

        with pytest.raises(ValueError, match="Unknown loss type: invalid"):
            simpo_loss(loss, loss_type="invalid")

    def test_gamma_beta_ratio_effect(self, simple_tensors):
        """Test that gamma_beta_ratio parameter affects results"""
        loss = simple_tensors

        results = []
        gamma_ratios = [0.0, 0.25, 0.5, 1.0]

        for gamma_ratio in gamma_ratios:
            result_losses, _ = simpo_loss(loss, gamma_beta_ratio=gamma_ratio)
            results.append(result_losses.mean().item())

        # Results should be different for different gamma_beta_ratio values
        assert len(set(results)) == len(gamma_ratios)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_beta_parameter_effect(self, simple_tensors):
        """Test that beta parameter affects results"""
        loss = simple_tensors

        results = []
        beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]

        for beta in beta_values:
            result_losses, _ = simpo_loss(loss, beta=beta)
            results.append(result_losses.mean().item())

        # Results should be different for different beta values
        assert len(set(results)) == len(beta_values)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_smoothing_parameter_sigmoid(self, simple_tensors):
        """Test smoothing parameter with sigmoid loss"""
        loss = simple_tensors

        # Test different smoothing values
        smoothing_values = [0.0, 0.1, 0.3, 0.5]
        results = []

        for smoothing in smoothing_values:
            result_losses, _ = simpo_loss(loss, loss_type="sigmoid", smoothing=smoothing)
            results.append(result_losses.mean().item())

        # Results should be different for different smoothing values
        assert len(set(results)) == len(smoothing_values)

        # All results should be finite
        for result in results:
            assert torch.isfinite(torch.tensor(result))

    def test_smoothing_parameter_hinge(self, simple_tensors):
        """Test that smoothing parameter doesn't affect hinge loss"""
        loss = simple_tensors

        # Smoothing should not affect hinge loss
        result_no_smooth, _ = simpo_loss(loss, loss_type="hinge", smoothing=0.0)
        result_with_smooth, _ = simpo_loss(loss, loss_type="hinge", smoothing=0.5)

        # Results should be identical for hinge loss regardless of smoothing
        assert torch.allclose(result_no_smooth, result_with_smooth)

    def test_tensor_chunking(self, sample_tensors):
        """Test that tensor chunking works correctly"""
        loss = sample_tensors

        result_losses, metrics = simpo_loss(loss)

        # The function should handle chunking internally
        assert torch.isfinite(result_losses).all()
        assert len(metrics) == 3

        # Verify chunking produces correct shapes
        loss_w, loss_l = loss.chunk(2)
        assert loss_w.shape == loss_l.shape
        assert loss_w.shape[0] == loss.shape[0] // 2
        assert result_losses.shape == loss_w.shape

    def test_logits_computation(self, simple_tensors):
        """Test the logits computation (pi_logratios - gamma_beta_ratio)"""
        loss = simple_tensors
        gamma_beta_ratio = 0.25

        _, metrics = simpo_loss(loss, gamma_beta_ratio=gamma_beta_ratio)

        # Manually compute logits
        loss_w, loss_l = loss.chunk(2)
        pi_logratios = loss_w - loss_l
        expected_logits = pi_logratios - gamma_beta_ratio

        # The logratio metric should match our manual pi_logratios computation
        # (Note: metric includes beta scaling)
        beta = 2.0  # default beta
        expected_logratio_metric = (beta * expected_logits).mean().item()

        assert abs(metrics["loss/simpo_logratio"] - expected_logratio_metric) < 1e-5

    def test_sigmoid_loss_manual_computation(self, simple_tensors):
        """Test sigmoid loss computation matches manual calculation"""
        loss = simple_tensors
        beta = 2.0
        gamma_beta_ratio = 0.25
        smoothing = 0.1

        result_losses, _ = simpo_loss(loss, loss_type="sigmoid", beta=beta, gamma_beta_ratio=gamma_beta_ratio, smoothing=smoothing)

        # Manual computation
        loss_w, loss_l = loss.chunk(2)
        pi_logratios = loss_w - loss_l
        logits = pi_logratios - gamma_beta_ratio
        expected_losses = -F.logsigmoid(beta * logits) * (1 - smoothing) - F.logsigmoid(-beta * logits) * smoothing

        assert torch.allclose(result_losses, expected_losses, atol=1e-6)

    def test_hinge_loss_manual_computation(self, simple_tensors):
        """Test hinge loss computation matches manual calculation"""
        loss = simple_tensors
        beta = 2.0
        gamma_beta_ratio = 0.25

        result_losses, _ = simpo_loss(loss, loss_type="hinge", beta=beta, gamma_beta_ratio=gamma_beta_ratio)

        # Manual computation
        loss_w, loss_l = loss.chunk(2)
        pi_logratios = loss_w - loss_l
        logits = pi_logratios - gamma_beta_ratio
        expected_losses = torch.relu(1 - beta * logits)

        assert torch.allclose(result_losses, expected_losses, atol=1e-6)

    def test_reward_metrics_computation(self, simple_tensors):
        """Test that reward metrics are computed correctly"""
        loss = simple_tensors
        beta = 2.0

        _, metrics = simpo_loss(loss, beta=beta)

        # Manual computation of rewards
        loss_w, loss_l = loss.chunk(2)
        expected_chosen_rewards = (beta * loss_w.detach()).mean().item()
        expected_rejected_rewards = (beta * loss_l.detach()).mean().item()

        assert abs(metrics["loss/simpo_chosen_rewards"] - expected_chosen_rewards) < 1e-6
        assert abs(metrics["loss/simpo_rejected_rewards"] - expected_rejected_rewards) < 1e-6

    def test_gradient_flow(self, simple_tensors):
        """Test that gradients flow properly through the loss"""
        loss = simple_tensors
        loss.requires_grad_(True)

        result_losses, _ = simpo_loss(loss)

        # Sum losses to get scalar for backward pass
        total_loss = result_losses.sum()
        total_loss.backward()

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

        result_losses, metrics = simpo_loss(loss)

        # The losses should be finite
        assert torch.isfinite(result_losses).all()

        # With preferred having lower loss, pi_logratios should be negative
        # This should lead to specific behavior in the loss computation
        pi_logratios = loss_w - loss_l  # Should be negative (1.0 - 3.0 = -2.0)

        assert pi_logratios.mean() == -2.0

        # Chosen rewards should be lower than rejected rewards (since loss_w < loss_l)
        assert metrics["loss/simpo_chosen_rewards"] < metrics["loss/simpo_rejected_rewards"]

    def test_equal_losses_case(self):
        """Test behavior when preferred and dispreferred losses are equal"""
        # Create scenario where preferred and dispreferred have same loss
        loss_w = torch.full((1, 4, 32, 32), 2.0)
        loss_l = torch.full((1, 4, 32, 32), 2.0)
        loss = torch.cat([loss_w, loss_l], dim=0)

        result_losses, metrics = simpo_loss(loss)

        # pi_logratios should be zero
        assert torch.isfinite(result_losses).all()

        # Chosen and rejected rewards should be equal
        assert abs(metrics["loss/simpo_chosen_rewards"] - metrics["loss/simpo_rejected_rewards"]) < 1e-6

        # Logratio should reflect the gamma_beta_ratio offset
        gamma_beta_ratio = 0.25  # default
        beta = 2.0  # default
        expected_logratio = -beta * gamma_beta_ratio  # Since pi_logratios = 0
        assert abs(metrics["loss/simpo_logratio"] - expected_logratio) < 1e-6

    def test_numerical_stability_extreme_values(self):
        """Test numerical stability with extreme values"""
        # Test with very large values
        large_loss = torch.full((2, 4, 32, 32), 100.0)
        result_losses, _ = simpo_loss(large_loss)
        assert torch.isfinite(result_losses).all()

        # Test with very small values
        small_loss = torch.full((2, 4, 32, 32), 1e-6)
        result_losses, _ = simpo_loss(small_loss)
        assert torch.isfinite(result_losses).all()

        # Test with negative values
        negative_loss = torch.full((2, 4, 32, 32), -10.0)
        result_losses, _ = simpo_loss(negative_loss)
        assert torch.isfinite(result_losses).all()

    def test_zero_beta_case(self, simple_tensors):
        """Test the case when beta = 0"""
        loss = simple_tensors
        beta = 0.0

        result_losses, metrics = simpo_loss(loss, beta=beta)

        # With beta=0, both loss types should give specific results
        assert torch.isfinite(result_losses).all()

        # For sigmoid: logsigmoid(0) = log(0.5) ≈ -0.693
        # For hinge: relu(1 - 0) = 1

        # Rewards should be zero
        assert abs(metrics["loss/simpo_chosen_rewards"]) < 1e-6
        assert abs(metrics["loss/simpo_rejected_rewards"]) < 1e-6
        assert abs(metrics["loss/simpo_logratio"]) < 1e-6

    def test_large_beta_case(self, simple_tensors):
        """Test the case with very large beta"""
        loss = simple_tensors
        beta = 1000.0

        result_losses, metrics = simpo_loss(loss, beta=beta)

        # Even with large beta, should remain stable
        assert torch.isfinite(result_losses).all()
        assert torch.isfinite(torch.tensor(metrics["loss/simpo_chosen_rewards"]))
        assert torch.isfinite(torch.tensor(metrics["loss/simpo_rejected_rewards"]))
        assert torch.isfinite(torch.tensor(metrics["loss/simpo_logratio"]))

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

        result_losses, metrics = simpo_loss(loss)

        assert torch.isfinite(result_losses).all()
        assert result_losses.shape == (batch_size, channels, height, width)
        assert len(metrics) == 3

    def test_device_compatibility(self, simple_tensors):
        """Test that function works on different devices"""
        loss = simple_tensors

        # Test on CPU
        result_cpu, _ = simpo_loss(loss)
        assert result_cpu.device.type == "cpu"

        # Test on GPU if available
        if torch.cuda.is_available():
            loss_gpu = loss.cuda()
            result_gpu, _ = simpo_loss(loss_gpu)
            assert result_gpu.device.type == "cuda"

    def test_reproducibility(self, simple_tensors):
        """Test that results are reproducible with same inputs"""
        loss = simple_tensors

        # Run multiple times
        result1, metrics1 = simpo_loss(loss)
        result2, metrics2 = simpo_loss(loss)

        # Results should be identical (deterministic computation)
        assert torch.allclose(result1, result2)
        for key in metrics1:
            assert abs(metrics1[key] - metrics2[key]) < 1e-6

    def test_no_reference_model_needed(self, simple_tensors):
        """Test that SimPO works without reference model (key feature)"""
        loss = simple_tensors

        # SimPO should work with just the loss tensor, no reference needed
        result_losses, metrics = simpo_loss(loss)

        # Should produce meaningful results without reference model
        assert torch.isfinite(result_losses).all()
        assert len(metrics) == 3
        assert all(key in metrics for key in ["loss/simpo_chosen_rewards", "loss/simpo_rejected_rewards", "loss/simpo_logratio"])

    def test_smoothing_interpolation_sigmoid(self):
        """Test that smoothing interpolates between positive and negative logsigmoid"""
        loss_w = torch.full((1, 4, 32, 32), 1.0)
        loss_l = torch.full((1, 4, 32, 32), 2.0)
        loss = torch.cat([loss_w, loss_l], dim=0)

        # Test extreme smoothing values
        no_smooth, _ = simpo_loss(loss, loss_type="sigmoid", smoothing=0.0)
        full_smooth, _ = simpo_loss(loss, loss_type="sigmoid", smoothing=1.0)
        half_smooth, _ = simpo_loss(loss, loss_type="sigmoid", smoothing=0.5)

        # With smoothing=0.5, result should be between the extremes
        assert torch.isfinite(no_smooth).all()
        assert torch.isfinite(full_smooth).all()
        assert torch.isfinite(half_smooth).all()

        # The smoothed version should be different from both extremes
        assert not torch.allclose(no_smooth, full_smooth)
        assert not torch.allclose(half_smooth, no_smooth)
        assert not torch.allclose(half_smooth, full_smooth)

    def test_hinge_loss_properties(self):
        """Test specific properties of hinge loss"""
        # Create scenario where logits > 1/beta (should give zero loss)
        loss_w = torch.full((1, 4, 32, 32), -2.0)  # Very low preferred loss
        loss_l = torch.full((1, 4, 32, 32), 2.0)  # High dispreferred loss
        loss = torch.cat([loss_w, loss_l], dim=0)

        beta = 0.5  # Small beta
        gamma_beta_ratio = 0.25

        result_losses, _ = simpo_loss(loss, loss_type="hinge", beta=beta, gamma_beta_ratio=gamma_beta_ratio)

        # Calculate expected behavior
        pi_logratios = loss_w - loss_l  # -2 - 2 = -4
        logits = pi_logratios - gamma_beta_ratio  # -4 - 0.25 = -4.25
        # relu(1 - 0.5 * (-4.25)) = relu(1 + 2.125) = relu(3.125) = 3.125

        expected_value = 1 - beta * logits  # 1 - 0.5 * (-4.25) = 3.125
        assert torch.allclose(result_losses, expected_value)

    def test_edge_case_all_zeros(self):
        """Test edge case with all zero losses"""
        loss = torch.zeros(2, 4, 32, 32)

        result_losses, metrics = simpo_loss(loss)

        # Should handle all zeros gracefully
        assert torch.isfinite(result_losses).all()
        assert torch.isfinite(torch.tensor(metrics["loss/simpo_chosen_rewards"]))
        assert torch.isfinite(torch.tensor(metrics["loss/simpo_rejected_rewards"]))
        assert torch.isfinite(torch.tensor(metrics["loss/simpo_logratio"]))

        # With all zeros: chosen and rejected rewards should be zero
        assert abs(metrics["loss/simpo_chosen_rewards"]) < 1e-6
        assert abs(metrics["loss/simpo_rejected_rewards"]) < 1e-6

    def test_gamma_beta_ratio_as_margin(self):
        """Test that gamma_beta_ratio acts as a margin in the logits"""
        loss_w = torch.full((1, 4, 32, 32), 1.0)
        loss_l = torch.full((1, 4, 32, 32), 1.0)  # Equal losses
        loss = torch.cat([loss_w, loss_l], dim=0)

        # With equal losses, pi_logratios = 0, so logits = -gamma_beta_ratio
        gamma_ratios = [0.0, 0.5, 1.0]

        for gamma_ratio in gamma_ratios:
            _, metrics = simpo_loss(loss, gamma_beta_ratio=gamma_ratio)

            # logratio should be -beta * gamma_ratio
            beta = 2.0  # default
            expected_logratio = -beta * gamma_ratio
            assert abs(metrics["loss/simpo_logratio"] - expected_logratio) < 1e-6

    def test_return_tensor_vs_scalar_difference_from_cpo(self):
        """Test that SimPO returns tensor losses (not scalar like some other methods)"""
        loss = torch.randn(2, 4, 32, 32)

        result_losses, _ = simpo_loss(loss)

        # SimPO should return tensor with same shape as preferred batch
        loss_w, _ = loss.chunk(2)
        assert result_losses.shape == loss_w.shape
        assert result_losses.dim() > 0  # Not a scalar

    @pytest.mark.parametrize("loss_type", ["sigmoid", "hinge"])
    def test_parameter_combinations(self, simple_tensors, loss_type):
        """Test various parameter combinations work correctly"""
        loss = simple_tensors

        # Test different parameter combinations
        param_combinations = [
            {"beta": 0.5, "gamma_beta_ratio": 0.1, "smoothing": 0.0},
            {"beta": 2.0, "gamma_beta_ratio": 0.5, "smoothing": 0.1},
            {"beta": 5.0, "gamma_beta_ratio": 1.0, "smoothing": 0.3},
        ]

        for params in param_combinations:
            result_losses, metrics = simpo_loss(loss, loss_type=loss_type, **params)

            assert torch.isfinite(result_losses).all()
            assert len(metrics) == 3
            assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
