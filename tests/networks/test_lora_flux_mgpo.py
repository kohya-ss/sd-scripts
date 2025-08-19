import pytest
import torch
import math
from networks.lora_flux import LoRAModule


class MockLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        return torch.matmul(x, self.weight.t())

    def state_dict(self):
        return {"weight": self.weight}


class MockOptimizer:
    def __init__(self, param):
        self.state = {param: {"exp_avg": torch.randn_like(param)}}


@pytest.fixture
def lora_module():
    org_module = MockLinear(10, 20)
    lora_module = LoRAModule(org_module, org_module, multiplier=1.0, lora_dim=4, alpha=1.0, mgpo_rho=0.1, mgpo_beta=0.9)
    # Manually set org_module_shape to match the original module's weight
    lora_module.org_module_shape = org_module.weight.shape
    return lora_module


def test_mgpo_parameter_initialization(lora_module):
    """Test MGPO-specific parameter initialization."""
    # Check MGPO-specific attributes
    assert hasattr(lora_module, "mgpo_rho")
    assert hasattr(lora_module, "mgpo_beta")
    assert lora_module.mgpo_rho == 0.1
    assert lora_module.mgpo_beta == 0.9

    # Check EMA parameters initialization
    assert hasattr(lora_module, "_grad_magnitude_ema_down")
    assert hasattr(lora_module, "_grad_magnitude_ema_up")
    assert isinstance(lora_module._grad_magnitude_ema_down, torch.nn.Parameter)
    assert isinstance(lora_module._grad_magnitude_ema_up, torch.nn.Parameter)
    assert lora_module._grad_magnitude_ema_down.requires_grad == False
    assert lora_module._grad_magnitude_ema_up.requires_grad == False
    assert lora_module._grad_magnitude_ema_down.item() == 1.0
    assert lora_module._grad_magnitude_ema_up.item() == 1.0


def test_update_gradient_ema(lora_module):
    """Test gradient EMA update method."""
    # Ensure method works when mgpo_beta is set
    lora_module.lora_down.weight.grad = torch.randn_like(lora_module.lora_down.weight)
    lora_module.lora_up.weight.grad = torch.randn_like(lora_module.lora_up.weight)

    # Store initial EMA values
    initial_down_ema = lora_module._grad_magnitude_ema_down.clone()
    initial_up_ema = lora_module._grad_magnitude_ema_up.clone()

    # Update gradient EMA
    lora_module.update_gradient_ema()

    # Check EMA update logic
    down_grad_norm = torch.norm(lora_module.lora_down.weight.grad, p=2)
    up_grad_norm = torch.norm(lora_module.lora_up.weight.grad, p=2)

    # Verify EMA calculation
    expected_down_ema = lora_module.mgpo_beta * initial_down_ema + (1 - lora_module.mgpo_beta) * down_grad_norm
    expected_up_ema = lora_module.mgpo_beta * initial_up_ema + (1 - lora_module.mgpo_beta) * up_grad_norm

    assert torch.allclose(lora_module._grad_magnitude_ema_down, expected_down_ema, rtol=1e-5)
    assert torch.allclose(lora_module._grad_magnitude_ema_up, expected_up_ema, rtol=1e-5)

    # Test when mgpo_beta is None
    lora_module.mgpo_beta = None
    lora_module.update_gradient_ema()  # Should not raise an exception


def test_get_mgpo_output_perturbation(lora_module):
    """Test MGPO perturbation generation."""
    # Create a mock optimizer
    mock_optimizer = MockOptimizer(lora_module.lora_down.weight)
    lora_module.register_optimizer(mock_optimizer)

    # Prepare input
    x = torch.randn(5, 10)  # batch × input_dim

    # Ensure method works with valid conditions
    perturbation = lora_module.get_mgpo_output_perturbation(x)

    # Verify perturbation characteristics
    assert perturbation is not None
    assert isinstance(perturbation, torch.Tensor)
    assert perturbation.shape == (x.shape[0], lora_module.org_module.out_features)

    # Test when conditions are not met
    lora_module.optimizer = None
    lora_module.mgpo_rho = None
    lora_module.mgpo_beta = None

    no_perturbation = lora_module.get_mgpo_output_perturbation(x)
    assert no_perturbation is None


def test_register_optimizer(lora_module):
    """Test optimizer registration method."""
    # Create a mock optimizer
    mock_optimizer = MockOptimizer(lora_module.lora_down.weight)

    # Register optimizer
    lora_module.register_optimizer(mock_optimizer)

    # Verify optimizer is correctly registered
    assert hasattr(lora_module, "optimizer")
    assert lora_module.optimizer == mock_optimizer
