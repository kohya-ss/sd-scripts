import torch

from library.custom_train_functions import diffusion_dpo_loss


def test_diffusion_dpo_loss_basic():
    # Test basic functionality with simple inputs
    batch_size = 4
    channels = 3
    height, width = 8, 8

    # Create dummy loss tensors
    loss = torch.rand(batch_size, channels, height, width)
    ref_loss = torch.rand(batch_size, channels, height, width)
    beta_dpo = 0.1

    result, metrics = diffusion_dpo_loss(loss.mean([1, 2, 3]), ref_loss.mean([1, 2, 3]), beta_dpo)

    # Check return types
    assert isinstance(result, torch.Tensor)
    assert isinstance(metrics, dict)

    # Check shape of result
    assert result.shape == torch.Size([batch_size // 2])

    # Check metrics
    expected_keys = [
        "loss/diffusion_dpo_total_loss",
        "loss/diffusion_dpo_raw_loss",
        "loss/diffusion_dpo_ref_loss",
        "loss/diffusion_dpo_implicit_acc",
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_diffusion_dpo_loss_different_shapes():
    # Test with different tensor shapes
    shapes = [
        (2, 3, 8, 8),  # Small tensor
        (4, 6, 16, 16),  # Medium tensor
        (6, 9, 32, 32),  # Larger tensor
    ]

    for shape in shapes:
        loss = torch.rand(*shape)
        ref_loss = torch.rand(*shape)

        result, metrics = diffusion_dpo_loss(loss.mean([1, 2, 3]), ref_loss.mean([1, 2, 3]), 0.1)

        # Result should have batch dimension halved
        assert result.shape == torch.Size([shape[0] // 2])

        # All metrics should be scalars
        for val in metrics.values():
            assert isinstance(val, float)


def test_diffusion_dpo_loss_beta_values():
    # Test with different beta values
    batch_size = 4
    channels = 3
    height, width = 8, 8

    loss = torch.rand(batch_size, channels, height, width)
    ref_loss = torch.rand(batch_size, channels, height, width)

    # Test with different beta values
    beta_values = [0.0, 0.5, 1.0, 10.0]
    results = []

    for beta in beta_values:
        result, _ = diffusion_dpo_loss(loss, ref_loss, beta)
        results.append(result.mean().item())

    # With different betas, results should vary
    assert len(set(results)) > 1, "Different beta values should produce different results"


def test_diffusion_dpo_loss_implicit_acc():
    # Test implicit accuracy calculation
    batch_size = 4
    channels = 3
    height, width = 8, 8

    # Create controlled test data where winners have lower loss
    loss_w = torch.ones(batch_size // 2, channels, height, width) * 0.2
    loss_l = torch.ones(batch_size // 2, channels, height, width) * 0.8
    loss = torch.cat([loss_w, loss_l], dim=0)

    # Make reference losses with opposite preference
    ref_w = torch.ones(batch_size // 2, channels, height, width) * 0.8
    ref_l = torch.ones(batch_size // 2, channels, height, width) * 0.2
    ref_loss = torch.cat([ref_w, ref_l], dim=0)

    # With beta=1.0, model_diff and ref_diff are opposite, should give low accuracy
    _, metrics = diffusion_dpo_loss(loss.mean((1, 2, 3)), ref_loss.mean((1, 2, 3)), 1.0)
    assert metrics["loss/diffusion_dpo_implicit_acc"] > 0.5

    # With beta=-1.0, the sign is flipped, should give high accuracy
    _, metrics = diffusion_dpo_loss(loss.mean((1, 2, 3)), ref_loss.mean((1, 2, 3)), -1.0)
    assert metrics["loss/diffusion_dpo_implicit_acc"] < 0.5


def test_diffusion_dpo_gradient_flow():
    # Test that gradients flow properly
    batch_size = 4
    channels = 3
    height, width = 8, 8

    # Create tensors that require gradients
    loss = torch.rand(batch_size, channels, height, width, requires_grad=True)
    ref_loss = torch.rand(batch_size, channels, height, width, requires_grad=False)

    # Compute loss
    result, _ = diffusion_dpo_loss(loss, ref_loss, 0.1)

    # Backpropagate
    result.mean().backward()

    # Verify gradients flowed through loss but not ref_loss
    assert loss.grad is not None
    assert ref_loss.grad is None  # Reference loss should be detached


def test_diffusion_dpo_loss_chunking():
    # Test chunking functionality
    batch_size = 4
    channels = 3
    height, width = 8, 8

    # Create controlled inputs where first half is clearly different from second half
    first_half = torch.zeros(batch_size // 2, channels, height, width)
    second_half = torch.ones(batch_size // 2, channels, height, width)

    # Test that the function correctly chunks inputs
    loss = torch.cat([first_half, second_half], dim=0)
    ref_loss = torch.cat([first_half, second_half], dim=0)

    result, metrics = diffusion_dpo_loss(loss.mean((1, 2, 3)), ref_loss.mean((1, 2, 3)), 1.0)

    # Since model_diff and ref_diff are identical, implicit acc should be 0.5
    assert abs(metrics["loss/diffusion_dpo_implicit_acc"] - 0.5) < 1e-5
