import torch
import pytest
from library.lora_util import initialize_pissa
from ..test_util import generate_synthetic_weights


def test_initialize_pissa_basic():
    # Create a simple linear layer
    org_module = torch.nn.Linear(10, 5)
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    torch.nn.init.xavier_uniform_(org_module.weight)
    torch.nn.init.zeros_(org_module.bias)

    # Create LoRA layers with matching shapes
    lora_down = torch.nn.Linear(10, 2)
    lora_up = torch.nn.Linear(2, 5)

    # Store original weight for comparison
    original_weight = org_module.weight.data.clone()

    # Call the initialization function
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=2)

    # Verify basic properties
    assert lora_down.weight.data is not None
    assert lora_up.weight.data is not None
    assert org_module.weight.data is not None

    # Check that the weights have been modified
    assert not torch.equal(original_weight, org_module.weight.data)


def test_initialize_pissa_rank_constraints():
    # Test with different rank values
    org_module = torch.nn.Linear(20, 10)
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    torch.nn.init.xavier_uniform_(org_module.weight)
    torch.nn.init.zeros_(org_module.bias)

    # Test with rank less than min dimension
    lora_down = torch.nn.Linear(20, 3)
    lora_up = torch.nn.Linear(3, 10)
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=3)

    # Test with rank equal to min dimension
    lora_down = torch.nn.Linear(20, 10)
    lora_up = torch.nn.Linear(10, 10)
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=10)


def test_initialize_pissa_shape_mismatch():
    # Test with shape mismatch to ensure warning is printed
    org_module = torch.nn.Linear(20, 10)

    # Intentionally mismatched shapes to test warning mechanism
    lora_down = torch.nn.Linear(20, 5)  # Different shape
    lora_up = torch.nn.Linear(3, 15)  # Different shape

    with pytest.warns(UserWarning):
        initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=3)


def test_initialize_pissa_scaling():
    # Test different scaling factors
    scales = [0.0, 0.1, 1.0]

    for scale in scales:
        org_module = torch.nn.Linear(10, 5)
        org_module.weight.data = generate_synthetic_weights(org_module.weight)
        original_weight = org_module.weight.data.clone()

        lora_down = torch.nn.Linear(10, 2)
        lora_up = torch.nn.Linear(2, 5)

        initialize_pissa(org_module, lora_down, lora_up, scale=scale, rank=2)

        # Check that the weight modification follows the scaling
        weight_diff = original_weight - org_module.weight.data
        expected_diff = scale * (lora_up.weight.data @ lora_down.weight.data)

        torch.testing.assert_close(weight_diff, expected_diff, rtol=1e-4, atol=1e-4)


def test_initialize_pissa_dtype():
    # Test with different data types
    dtypes = [torch.float16, torch.float32, torch.float64]

    for dtype in dtypes:
        org_module = torch.nn.Linear(10, 5).to(dtype=dtype)
        org_module.weight.data = generate_synthetic_weights(org_module.weight)

        lora_down = torch.nn.Linear(10, 2)
        lora_up = torch.nn.Linear(2, 5)

        initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=2)

        # Verify output dtype matches input
        assert org_module.weight.dtype == dtype


def test_initialize_pissa_svd_properties():
    # Verify SVD decomposition properties
    org_module = torch.nn.Linear(20, 10)
    lora_down = torch.nn.Linear(20, 3)
    lora_up = torch.nn.Linear(3, 10)

    org_module.weight.data = generate_synthetic_weights(org_module.weight)
    original_weight = org_module.weight.data.clone()

    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=3)

    # Reconstruct the weight
    reconstructed_weight = original_weight - 0.1 * (lora_up.weight.data @ lora_down.weight.data)

    # Check reconstruction is close to original
    torch.testing.assert_close(reconstructed_weight, org_module.weight.data, rtol=1e-4, atol=1e-4)


def test_initialize_pissa_device_handling():
    # Test different device scenarios
    devices = [torch.device("cpu"), torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")]

    for device in devices:
        # Create modules on specific device
        org_module = torch.nn.Linear(10, 5).to(device)
        lora_down = torch.nn.Linear(10, 2).to(device)
        lora_up = torch.nn.Linear(2, 5).to(device)

        # Test initialization with explicit device
        initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=2, device=device)

        # Verify modules are on the correct device
        assert org_module.weight.data.device.type == device.type
        assert lora_down.weight.data.device.type == device.type
        assert lora_up.weight.data.device.type == device.type


def test_initialize_pissa_dtype_preservation():
    # Test dtype preservation and conversion
    dtypes = [torch.float16, torch.float32, torch.float64]

    for dtype in dtypes:
        org_module = torch.nn.Linear(10, 5).to(dtype=dtype)
        lora_down = torch.nn.Linear(10, 2).to(dtype=dtype)
        lora_up = torch.nn.Linear(2, 5).to(dtype=dtype)

        initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=2)

        assert org_module.weight.dtype == dtype
        assert lora_down.weight.dtype == dtype
        assert lora_up.weight.dtype == dtype


def test_initialize_pissa_rank_limits():
    # Test rank limits
    org_module = torch.nn.Linear(10, 5)

    # Test minimum rank (should work)
    lora_down_min = torch.nn.Linear(10, 1)
    lora_up_min = torch.nn.Linear(1, 5)
    initialize_pissa(org_module, lora_down_min, lora_up_min, scale=0.1, rank=1)

    # Test maximum rank (rank = min(input_dim, output_dim))
    max_rank = min(10, 5)
    lora_down_max = torch.nn.Linear(10, max_rank)
    lora_up_max = torch.nn.Linear(max_rank, 5)
    initialize_pissa(org_module, lora_down_max, lora_up_max, scale=0.1, rank=max_rank)


def test_initialize_pissa_numerical_stability():
    # Test with numerically challenging scenarios
    scenarios = [
        torch.randn(20, 10) * 1e-10,  # Very small values
        torch.randn(20, 10) * 1e10,  # Very large values
        torch.zeros(20, 10),  # Zero matrix
    ]

    for i, weight_matrix in enumerate(scenarios):
        org_module = torch.nn.Linear(20, 10)
        org_module.weight.data = weight_matrix

        lora_down = torch.nn.Linear(10, 3)
        lora_up = torch.nn.Linear(3, 20)

        try:
            initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=3)
        except Exception as e:
            pytest.fail(f"Initialization failed for scenario ({i}): {e}")


def test_initialize_pissa_scale_effects():
    # Test different scaling factors
    org_module = torch.nn.Linear(10, 5)
    original_weight = org_module.weight.data.clone()

    test_scales = [0.0, 0.1, 0.5, 1.0]

    for scale in test_scales:
        # Reset module for each test
        org_module.weight.data = original_weight.clone()

        lora_down = torch.nn.Linear(10, 2)
        lora_up = torch.nn.Linear(2, 5)

        initialize_pissa(org_module, lora_down, lora_up, scale=scale, rank=2)

        # Verify weight modification proportional to scale
        weight_diff = original_weight - org_module.weight.data

        # Approximate check of scaling effect
        if scale == 0.0:
            torch.testing.assert_close(weight_diff, torch.zeros_like(weight_diff), rtol=1e-4, atol=1e-6)
        else:
            assert not torch.allclose(weight_diff, torch.zeros_like(weight_diff), rtol=1e-4, atol=1e-6)
