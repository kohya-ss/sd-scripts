import pytest
import torch
from library.model_utils import AID
from torch import nn
import torch.nn.functional as F

@pytest.fixture
def input_tensor():
    # Create a tensor with positive and negative values
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    return x

def test_aid_forward_train_mode(input_tensor):
    aid = AID(p=0.9)
    aid.train()
    
    # Run several forward passes to test stochastic behavior
    results = []
    for _ in range(10):
        output = aid(input_tensor)
        results.append(output.detach().clone())
    
    # Test that outputs vary (stochastic behavior)
    all_equal = all(torch.allclose(results[0], results[i]) for i in range(1, 10))
    assert not all_equal, "All outputs are identical, expected variability in training mode"
    
    # Test shape preservation
    assert results[0].shape == input_tensor.shape

def test_aid_forward_eval_mode(input_tensor):
    aid = AID(p=0.9)
    aid.eval()
    
    output = aid(input_tensor)
    
    # Test deterministic behavior
    output2 = aid(input_tensor)
    assert torch.allclose(output, output2), "Expected deterministic behavior in eval mode"
    
    # Test correct transformation
    expected = 0.9 * F.relu(input_tensor) + 0.1 * F.relu(-input_tensor) * -1
    assert torch.allclose(output, expected), "Incorrect evaluation mode transformation"

def test_aid_gradient_flow(input_tensor):
    aid = AID(p=0.9)
    aid.train()
    
    # Forward pass
    output = aid(input_tensor)
    
    # Check gradient flow
    assert output.requires_grad, "Output lost gradient tracking"
    
    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()
    
    # Verify gradients were computed
    assert input_tensor.grad is not None, "No gradients were recorded for input tensor"
    assert torch.any(input_tensor.grad != 0), "Gradients are all zeros"

def test_aid_extreme_p_values():
    # Test with p=1.0 (only positive values pass through)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    aid = AID(p=1.0)
    aid.eval()
    
    output = aid(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
    assert torch.allclose(output, expected), "Failed with p=1.0"
    
    # Test with p=0.0 (only negative values pass through)
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    aid = AID(p=0.0)
    aid.eval()
    
    output = aid(x)
    expected = torch.tensor([-2.0, -1.0, 0.0, 0.0, 0.0])
    assert torch.allclose(output, expected), "Failed with p=0.0"

def test_aid_with_all_positive_values():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    aid = AID(p=0.9)
    aid.train()
    
    # Run forward passes and check that only positive values are affected
    output = aid(x)
    
    # Backprop should work
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradients were recorded for all-positive input"

def test_aid_with_all_negative_values():
    x = torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], requires_grad=True)
    aid = AID(p=0.9)
    aid.train()
    
    # Run forward passes and check that only negative values are affected
    output = aid(x)
    
    # Backprop should work
    loss = output.sum()
    loss.backward()
    assert x.grad is not None, "No gradients were recorded for all-negative input"

def test_aid_with_zero_values():
    x = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    aid = AID(p=0.9)
    
    # Test training mode
    aid.train()
    output = aid(x)
    assert torch.allclose(output, torch.zeros_like(output)), "Expected zeros out for zero input"
    
    # Test eval mode
    aid.eval() 
    output = aid(x)
    assert torch.allclose(output, torch.zeros_like(output)), "Expected zeros out for zero input"

def test_aid_integration_with_linear_layer():
    # Test AID's compatibility with a linear layer
    linear = nn.Linear(5, 2)
    aid = AID(p=0.9)
    
    model = nn.Sequential(linear, aid)
    model.train()
    
    x = torch.randn(3, 5, requires_grad=True)
    output = model(x)
    
    # Check that gradients flow through the whole model
    loss = output.sum()
    loss.backward()
    
    assert linear.weight.grad is not None, "No gradients for linear layer weights"
    assert x.grad is not None, "No gradients for input tensor"
