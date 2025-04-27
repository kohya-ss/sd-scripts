import torch
from unittest.mock import Mock

from library.custom_train_functions import diffusion_dpo_loss

def test_diffusion_dpo_loss_basic():
    batch_size = 4
    channels = 4
    height, width = 64, 64
    
    # Create dummy loss tensor
    loss = torch.rand(batch_size, channels, height, width)
    
    # Mock the call_unet and apply_loss functions
    mock_unet_output = torch.rand(batch_size, channels, height, width)
    call_unet = Mock(return_value=mock_unet_output)
    
    mock_loss_output = torch.rand(batch_size, channels, height, width)
    apply_loss = Mock(return_value=mock_loss_output)
    
    beta_dpo = 0.1
    
    result, metrics = diffusion_dpo_loss(loss, call_unet, apply_loss, beta_dpo)
    
    # Check return types
    assert isinstance(result, torch.Tensor)
    assert isinstance(metrics, dict)
    
    # Check expected metrics are present
    expected_keys = ["total_loss", "raw_model_loss", "ref_loss", "implicit_acc"]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)
    
    # Verify mocks were called correctly
    call_unet.assert_called_once()
    apply_loss.assert_called_once_with(mock_unet_output)

def test_diffusion_dpo_loss_shapes():
    # Test with different tensor shapes
    shapes = [
        (2, 4, 32, 32),    # Small tensor
        (4, 16, 64, 64),  # Medium tensor
        (6, 32, 128, 128),  # Larger tensor
    ]
    
    for shape in shapes:
        loss = torch.rand(*shape)
        
        # Create mocks
        mock_unet_output = torch.rand(*shape)
        call_unet = Mock(return_value=mock_unet_output)
        
        mock_loss_output = torch.rand(*shape)
        apply_loss = Mock(return_value=mock_loss_output)
        
        result, metrics = diffusion_dpo_loss(loss, call_unet, apply_loss, 0.1)
        
        # The result should be a scalar tensor
        assert result.shape == torch.Size([shape[0] // 2])
        
        # All metrics should be scalars
        for val in metrics.values():
            assert isinstance(val, float)

def test_diffusion_dpo_loss_beta_values():
    batch_size = 2
    channels = 4
    height, width = 64, 64
    
    loss = torch.rand(batch_size, channels, height, width)
    
    # Create consistent mock returns
    mock_unet_output = torch.rand(batch_size, channels, height, width)
    mock_loss_output = torch.rand(batch_size, channels, height, width)
    
    # Test with different beta values
    beta_values = [0.0, 0.1, 1.0, 10.0]
    results = []
    
    for beta in beta_values:
        call_unet = Mock(return_value=mock_unet_output)
        apply_loss = Mock(return_value=mock_loss_output)
        
        result, metrics = diffusion_dpo_loss(loss, call_unet, apply_loss, beta)
        results.append(result.item())
    
    # With increasing beta, results should be different
    # This test checks that beta affects the output
    assert len(set(results)) > 1, "Different beta values should produce different results"

def test_diffusion_dpo_implicit_acc():
    batch_size = 4
    channels = 4
    height, width = 64, 64
    
    # Create controlled test data where winners have lower loss
    w_loss = torch.ones(batch_size//2, channels, height, width) * 0.2
    l_loss = torch.ones(batch_size//2, channels, height, width) * 0.8
    loss = torch.cat([w_loss, l_loss], dim=0)
    
    # Make the reference loss similar but with less difference
    ref_w_loss = torch.ones(batch_size//2, channels, height, width) * 0.3
    ref_l_loss = torch.ones(batch_size//2, channels, height, width) * 0.7
    ref_loss = torch.cat([ref_w_loss, ref_l_loss], dim=0)
    
    call_unet = Mock(return_value=torch.zeros_like(loss))  # Dummy, won't be used
    apply_loss = Mock(return_value=ref_loss)
    
    # With a positive beta, model_diff > ref_diff should lead to high implicit accuracy
    result, metrics = diffusion_dpo_loss(loss, call_unet, apply_loss, 1.0)
    
    # Implicit accuracy should be high (model correctly identifies preferences)
    assert metrics["implicit_acc"] > 0.5

def test_diffusion_dpo_gradient_flow():
    batch_size = 4
    channels = 4
    height, width = 64, 64
    
    # Create loss tensor that requires gradients
    loss = torch.rand(batch_size, channels, height, width, requires_grad=True)
    
    # Create mock outputs
    mock_unet_output = torch.rand(batch_size, channels, height, width)
    call_unet = Mock(return_value=mock_unet_output)
    
    mock_loss_output = torch.rand(batch_size, channels, height, width)
    apply_loss = Mock(return_value=mock_loss_output)
    
    # Compute loss
    result, _ = diffusion_dpo_loss(loss, call_unet, apply_loss, 0.1)
    
    # Check that gradients flow
    result.mean().backward()
    
    # Verify gradients flowed through
    assert loss.grad is not None

def test_diffusion_dpo_no_ref_grad():
    batch_size = 4
    channels = 4
    height, width = 64, 64
    
    loss = torch.rand(batch_size, channels, height, width, requires_grad=True)
    
    # Set up mock that tracks if it was called with no_grad
    mock_unet_output = torch.rand(batch_size, channels, height, width)
    call_unet = Mock(return_value=mock_unet_output)
    
    mock_loss_output = torch.rand(batch_size, channels, height, width, requires_grad=True)
    apply_loss = Mock(return_value=mock_loss_output)
    
    # Run function
    result, _ = diffusion_dpo_loss(loss, call_unet, apply_loss, 0.1)
    result.mean().backward()
    
    # Check that the reference loss has no gradients (was computed with torch.no_grad())
    # This is a bit tricky to test directly, but we can verify call_unet was called
    call_unet.assert_called_once()
    apply_loss.assert_called_once()
    
    # The mock_loss_output should not receive gradients as it's used inside torch.no_grad()
    assert mock_loss_output.grad is None
