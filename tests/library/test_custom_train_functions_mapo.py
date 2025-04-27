import torch
import numpy as np

from library.custom_train_functions import mapo_loss


def test_mapo_loss_basic():
    batch_size = 4
    channels = 4
    height, width = 64, 64

    # Create dummy loss tensor with shape [B, C, H, W]
    loss = torch.rand(batch_size, channels, height, width)
    mapo_weight = 0.5

    result, metrics = mapo_loss(loss, mapo_weight)

    # Check return types
    assert isinstance(result, torch.Tensor)
    assert isinstance(metrics, dict)

    # Check required metrics are present
    expected_keys = ["total_loss", "ratio_loss", "model_losses_w", "model_losses_l", "win_score", "lose_score"]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_mapo_loss_different_shapes():
    # Test with different tensor shapes
    shapes = [
        (2, 4, 32, 32),  # Small tensor
        (4, 16, 64, 64),  # Medium tensor
        (6, 32, 128, 128),  # Larger tensor
    ]

    for shape in shapes:
        loss = torch.rand(*shape)
        result, metrics = mapo_loss(loss, 0.5)

        # The result should be a scalar tensor
        assert result.shape == torch.Size([])

        # All metrics should be scalars
        for val in metrics.values():
            assert np.isscalar(val)


def test_mapo_loss_with_zero_weight():
    loss = torch.rand(4, 3, 64, 64)
    result, metrics = mapo_loss(loss, 0.0)

    # With zero mapo_weight, ratio_loss should be zero
    assert metrics["ratio_loss"] == 0.0

    # result should be equal to mean of model_losses_w
    assert torch.isclose(result, torch.tensor(metrics["model_losses_w"]))


def test_mapo_loss_with_different_timesteps():
    loss = torch.rand(4, 4, 32, 32)

    # Test with different timestep values
    timesteps = [1, 10, 100, 1000]

    for ts in timesteps:
        result, metrics = mapo_loss(loss, 0.5, ts)

        # Check that the results are different for different timesteps
        if ts > 1:
            result_prev, metrics_prev = mapo_loss(loss, 0.5, ts // 10)
            # Log odds should be affected by timesteps, so ratio_loss should change
            assert metrics["ratio_loss"] != metrics_prev["ratio_loss"]


def test_mapo_loss_win_loss_scores():
    # Create a controlled input where win losses are lower than lose losses
    batch_size = 4
    channels = 4
    height, width = 64, 64

    # Create losses where winning examples have lower loss
    w_loss = torch.ones(batch_size // 2, channels, height, width) * 0.1
    l_loss = torch.ones(batch_size // 2, channels, height, width) * 0.9

    # Concatenate to create the full loss tensor
    loss = torch.cat([w_loss, l_loss], dim=0)

    # Run the function
    result, metrics = mapo_loss(loss, 0.5)

    # Win score should be higher than lose score (better performance)
    assert metrics["win_score"] > metrics["lose_score"]

    # Model losses for winners should be lower
    assert metrics["model_losses_w"] < metrics["model_losses_l"]


def test_mapo_loss_gradient_flow():
    # Test that gradients flow through the loss function
    batch_size = 4
    channels = 4
    height, width = 64, 64

    # Create a loss tensor that requires grad
    loss = torch.rand(batch_size, channels, height, width, requires_grad=True)
    mapo_weight = 0.5

    # Compute loss
    result, _ = mapo_loss(loss, mapo_weight)

    # Check that gradients flow
    result.backward()

    # If gradients flow, loss.grad should not be None
    assert loss.grad is not None
