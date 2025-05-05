import torch
import numpy as np

from library.custom_train_functions import mapo_loss


def test_mapo_loss_basic():
    batch_size = 8  # Must be even for chunking
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
    expected_keys = [
        "loss/mapo_total",
        "loss/mapo_ratio",
        "loss/mapo_w_loss",
        "loss/mapo_l_loss",
        "loss/mapo_win_score",
        "loss/mapo_lose_score",
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_mapo_loss_different_shapes():
    # Test with different tensor shapes
    shapes = [
        (4, 4, 32, 32),  # Small tensor
        (8, 16, 64, 64),  # Medium tensor
        (12, 32, 128, 128),  # Larger tensor
    ]
    for shape in shapes:
        loss = torch.rand(*shape)
        result, metrics = mapo_loss(loss.mean((1, 2, 3)), 0.5)
        # The result should have dimension batch_size//2
        assert result.shape == torch.Size([shape[0] // 2])
        # All metrics should be scalars
        for val in metrics.values():
            assert np.isscalar(val)


def test_mapo_loss_with_zero_weight():
    loss = torch.rand(8, 3, 64, 64)  # Batch size must be even
    loss_mean = loss.mean((1, 2, 3))
    result, metrics = mapo_loss(loss_mean, 0.0)
    
    # With zero mapo_weight, ratio_loss should be zero
    assert metrics["loss/mapo_ratio"] == 0.0
    
    # result should be equal to loss_w (first half of the batch)
    loss_w = loss_mean[:loss_mean.shape[0]//2]
    assert torch.allclose(result, loss_w)


def test_mapo_loss_with_different_timesteps():
    loss = torch.rand(8, 4, 32, 32)  # Batch size must be even
    # Test with different timestep values
    timesteps = [1, 10, 100, 1000]
    results = []
    for ts in timesteps:
        result, metrics = mapo_loss(loss, 0.5, ts)
        results.append(metrics["loss/mapo_ratio"])

    # Check that the results are different for different timesteps
    for i in range(1, len(results)):
        assert results[i] != results[i - 1]


def test_mapo_loss_win_loss_scores():
    batch_size = 8  # Must be even
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
    assert metrics["loss/mapo_win_score"] > metrics["loss/mapo_lose_score"]
    # Model losses for winners should be lower
    assert metrics["loss/mapo_w_loss"] < metrics["loss/mapo_l_loss"]


def test_mapo_loss_gradient_flow():
    batch_size = 8  # Must be even
    channels = 4
    height, width = 64, 64

    # Create a loss tensor that requires grad
    loss = torch.rand(batch_size, channels, height, width, requires_grad=True)
    mapo_weight = 0.5

    # Compute loss
    result, _ = mapo_loss(loss, mapo_weight)

    # Compute mean for backprop
    result.mean().backward()

    # If gradients flow, loss.grad should not be None
    assert loss.grad is not None
