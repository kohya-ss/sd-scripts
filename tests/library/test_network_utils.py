import torch
import pytest
from library.network_utils import lora_dropout_down, lora_dropout_up


@pytest.fixture
def setup_lora_dropout_dimensions():
    batch_size = 2
    in_dim = 32
    lora_dim = 8
    out_dim = 16

    x_in = torch.randn(batch_size, in_dim)
    x_mid = torch.randn(batch_size, lora_dim)

    down = torch.randn(lora_dim, in_dim)
    up = torch.randn(out_dim, lora_dim)

    return {
        "batch_size": batch_size,
        "in_dim": in_dim,
        "lora_dim": lora_dim,
        "out_dim": out_dim,
        "x_in": x_in,
        "x_mid": x_mid,
        "down": down,
        "up": up,
    }


# Tests
def test_lora_dropout_dimensions(setup_lora_dropout_dimensions):
    """Test if output dimensions are correct"""
    d = setup_lora_dropout_dimensions

    # Apply dropout
    mid_out = lora_dropout_down(d["down"], d["x_in"])
    final_out = lora_dropout_up(d["up"], mid_out)

    # Check dimensions
    assert mid_out.shape == (d["batch_size"], d["lora_dim"])
    assert final_out.shape == (d["batch_size"], d["out_dim"])


def test_lora_dropout_reproducibility():
    """Test if setting a seed makes dropout reproducible"""
    in_dim = 50
    lora_dim = 10
    batch_size = 3

    # Create sample inputs
    x_in = torch.randn(batch_size, in_dim)

    # Create weight matrix
    down = torch.randn(lora_dim, in_dim)

    # First run
    torch.manual_seed(123)
    result1 = lora_dropout_down(down, x_in)

    # Second run with same seed
    torch.manual_seed(123)
    result2 = lora_dropout_down(down, x_in)

    # They should be identical
    assert torch.allclose(result1, result2)


def test_lora_dropout_full_forward_path(setup_lora_dropout_dimensions):
    """Test a complete LoRA path with dropout"""
    torch.manual_seed(456)

    d = setup_lora_dropout_dimensions

    # Normal forward path without dropout
    mid_normal = d["x_in"] @ d["down"].t()
    out_normal = mid_normal @ d["up"].t()

    # Forward path with dropout
    mid_dropout = lora_dropout_down(d["down"], d["x_in"])
    out_dropout = lora_dropout_up(d["up"], mid_dropout)

    # The outputs should be different due to dropout
    assert not torch.allclose(out_normal, out_dropout)
