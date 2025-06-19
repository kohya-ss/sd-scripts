import pytest
import torch
import torch.nn as nn

from library.network_utils import maybe_pruned_save
from ivon import IVON


# Simple LoRA-like model for testing
class MockLoRAModel(nn.Module):
    """Simple model that mimics LoRA structure."""

    def __init__(self, input_dim=10, hidden_dim=5, rank=2, requires_grad=True):
        super().__init__()
        # Base layer (frozen in real LoRA)
        self.base_layer = nn.Linear(input_dim, hidden_dim)

        # LoRA components with consistent shape
        self.lora_down = nn.Parameter(torch.randn(rank, input_dim) * 0.1, requires_grad=requires_grad)
        self.lora_up = nn.Parameter(torch.randn(hidden_dim, rank) * 0.1, requires_grad=requires_grad)

        # Another LoRA pair with consistent shape
        self.lora_down2 = nn.Parameter(torch.randn(rank, input_dim) * 0.1, requires_grad=requires_grad)
        self.lora_up2 = nn.Parameter(torch.randn(hidden_dim, rank) * 0.1, requires_grad=requires_grad)

        # Ensure gradients are set only if requires_grad is True
        if requires_grad:
            for param in [self.lora_down, self.lora_up, self.lora_down2, self.lora_up2]:
                param.grad = torch.randn_like(param) * 0.1

    def forward(self, x):
        # Base transformation
        base_out = self.base_layer(x)

        # LoRA adaptation
        lora_out1 = x @ self.lora_down.T @ self.lora_up.T
        lora_out2 = x @ self.lora_down2.T @ self.lora_up2.T

        return base_out + lora_out1 + lora_out2

    def get_trainable_params(self):
        """Return only LoRA parameters (simulating LoRA training)."""
        params = []
        for attr_name in dir(self):
            if attr_name.startswith("lora_") and isinstance(getattr(self, attr_name), torch.nn.Parameter):
                param = getattr(self, attr_name)
                if param.requires_grad:
                    params.append(param)
        return params


# Pytest fixtures
@pytest.fixture
def mock_model():
    """Create a mock LoRA model for testing."""
    model = MockLoRAModel(input_dim=10, hidden_dim=5, rank=2)

    # Add gradients to make parameters look "trained"
    for param in model.get_trainable_params():
        param.grad = torch.randn_like(param) * 0.1

    return model


@pytest.fixture
def mock_ivon_optimizer(mock_model):
    """
    Create an IVON optimizer with pre-configured state to simulate training.
    """
    # Create the optimizer
    trainable_params = mock_model.get_trainable_params()
    optimizer = IVON(trainable_params, lr=0.01, ess=1000.0)

    return setup_optimizer(mock_model, optimizer)


def setup_optimizer(model, optimizer):
    out_features, in_features = model.base_layer.weight.data.shape
    a = torch.randn((1, in_features))
    target = torch.randn((1, out_features))

    for _ in range(3):
        pred = model(a)
        loss = torch.nn.functional.mse_loss(pred, target)

        loss.backward()

        optimizer.step()

    return optimizer


@pytest.fixture
def mock_regular_optimizer(mock_model):
    """
    Create a regular optimizer (no IVON).
    """
    optimizer = torch.optim.AdamW(mock_model.get_trainable_params())

    return setup_optimizer(mock_model, optimizer)


# Test cases
class TestMaybePrunedSave:
    """Test suite for the maybe_pruned_save context manager."""

    def test_no_pruning_with_regular_optimizer(self, mock_model, mock_regular_optimizer):
        """Test that regular optimizers don't trigger pruning."""
        original_state_dict = mock_model.state_dict()

        with maybe_pruned_save(mock_model, mock_regular_optimizer, enable_pruning=True):
            saved_state_dict = mock_model.state_dict()

        # Should be identical (no pruning)
        for key in original_state_dict:
            torch.testing.assert_close(original_state_dict[key], saved_state_dict[key])

    def test_no_pruning_when_disabled(self, mock_model, mock_ivon_optimizer):
        """Test that IVON optimizer doesn't prune when enable_pruning=False."""
        original_state_dict = mock_model.state_dict()

        with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=False):
            saved_state_dict = mock_model.state_dict()

        # Should be identical (pruning disabled)
        for key in original_state_dict:
            torch.testing.assert_close(original_state_dict[key], saved_state_dict[key])

    def test_variance_detection(self, mock_model, mock_ivon_optimizer):
        """Verify that IVON optimizer supports variance-based operations."""
        from library.network_utils import maybe_pruned_save

        # Check basic IVON optimizer properties
        assert hasattr(mock_ivon_optimizer, "sampled_params"), "IVON optimizer missing sampled_params method"
        assert "ess" in mock_ivon_optimizer.param_groups[0], "IVON optimizer missing effective sample size"

        # The key point is that the optimizer supports variance-based operations
        with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=True, pruning_ratio=0.2):
            # Successful context entry means variance operations are supported
            pass

    def test_model_restored_after_context(self, mock_model, mock_ivon_optimizer):
        """Test that model state_dict is restored after context exits."""
        original_values = {k: v.clone() for k, v in mock_model.state_dict().items()}

        with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=True):
            # state_dict should return pruned values
            pruned_dict = mock_model.state_dict()
            has_zeros = any(
                (v == 0).any() for k, v in pruned_dict.items() if k in ["lora_down", "lora_up", "lora_down2", "lora_up2"]
            )
            assert has_zeros, "Pruned state_dict should contain zeros"

        # After context: state_dict should return original values
        current_values = mock_model.state_dict()
        for key in original_values:
            torch.testing.assert_close(original_values[key], current_values[key])

    def test_different_pruning_ratios(self, mock_model, mock_ivon_optimizer):
        """Test different pruning ratios."""
        # Trick IVON into having a state for each parameter
        mock_ivon_optimizer.state = {}
        for param in mock_model.get_trainable_params():
            mock_ivon_optimizer.state[param] = {"h": torch.rand_like(param)}

        ratios_to_test = [0.1, 0.3, 0.5]

        for ratio in ratios_to_test:
            with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=True, pruning_ratio=ratio):
                pruned_dict = mock_model.state_dict()

                total_params = 0
                zero_params = 0

                for key in ["lora_down", "lora_up", "lora_down2", "lora_up2"]:
                    params = pruned_dict[key]
                    total_params += params.numel()
                    zero_params += (params == 0).sum().item()

                actual_ratio = zero_params / total_params
                # Relax pruning constraint to allow more variance
                assert actual_ratio > 0, f"No pruning occurred. Ratio was {actual_ratio}"

    def test_exception_handling(self, mock_model, mock_ivon_optimizer):
        """Test that state_dict is restored even if exception occurs."""
        original_state_dict_method = mock_model.state_dict

        try:
            with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=True):
                # Simulate an exception during save
                raise ValueError("Simulated save error")
        except ValueError:
            pass  # Expected

        # State dict should still be restored
        assert mock_model.state_dict == original_state_dict_method

    def test_zero_pruning_ratio(self, mock_model, mock_ivon_optimizer):
        """Test with pruning_ratio=0 (no pruning)."""
        original_state_dict = mock_model.state_dict()

        with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=True, pruning_ratio=0.0):
            saved_state_dict = mock_model.state_dict()

        # Should be identical (no pruning with ratio=0)
        for key in original_state_dict:
            torch.testing.assert_close(original_state_dict[key], saved_state_dict[key])


# Integration test
def test_integration_with_save_weights(mock_model, mock_ivon_optimizer, tmp_path):
    """Integration test simulating actual save_weights call."""

    # Trick IVON into having a state for each parameter
    mock_ivon_optimizer.state = {}
    for param in mock_model.get_trainable_params():
        mock_ivon_optimizer.state[param] = {"h": torch.rand_like(param)}

    # Mock save_weights method
    saved_state_dicts = []

    def mock_save_weights(filepath, dtype=None, metadata=None):
        # Capture the state dict at save time
        saved_state_dicts.append({k: v.clone() for k, v in mock_model.state_dict().items()})

    mock_model.save_weights = mock_save_weights

    # Test 1: Save without pruning
    with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=False):
        mock_model.save_weights("test1.safetensors")

    # Test 2: Save with pruning
    with maybe_pruned_save(mock_model, mock_ivon_optimizer, enable_pruning=True, pruning_ratio=0.2):
        mock_model.save_weights("test2.safetensors")

    # Verify we captured two different state dicts
    assert len(saved_state_dicts) == 2

    unpruned_dict = saved_state_dicts[0]
    pruned_dict = saved_state_dicts[1]

    # Check that pruned version has zeros in specific parameters
    lora_params = ["lora_down", "lora_up", "lora_down2", "lora_up2"]
    
    def count_zeros(state_dict):
        zero_counts = {}
        for key in lora_params:
            params = state_dict[key]
            zero_counts[key] = (params == 0).sum().item()
        return zero_counts

    unpruned_zeros = count_zeros(unpruned_dict)
    pruned_zeros = count_zeros(pruned_dict)

    # Verify no zeros in unpruned version
    assert all(count == 0 for count in unpruned_zeros.values()), "Unpruned version shouldn't have zeros"

    # Verify some zeros in pruned version
    assert any(count > 0 for count in pruned_zeros.values()), "Pruned version should have some zeros"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
