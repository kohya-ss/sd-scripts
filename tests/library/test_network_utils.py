import torch
import pytest
from library.network_utils import initialize_pissa
from library.test_util import generate_synthetic_weights


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


def test_initialize_pissa_with_ipca():
    # Test with IncrementalPCA option
    org_module = torch.nn.Linear(100, 50)  # Larger dimensions to test IPCA
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    lora_down = torch.nn.Linear(100, 8)
    lora_up = torch.nn.Linear(8, 50)

    original_weight = org_module.weight.data.clone()

    # Call with IPCA enabled
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=8, use_ipca=True)

    # Verify weights are changed
    assert not torch.equal(original_weight, org_module.weight.data)

    # Check that LoRA matrices have appropriate shapes
    assert lora_down.weight.shape == torch.Size([8, 100])
    assert lora_up.weight.shape == torch.Size([50, 8])


def test_initialize_pissa_with_lowrank():
    # Test with low-rank SVD option
    org_module = torch.nn.Linear(50, 30)
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    lora_down = torch.nn.Linear(50, 5)
    lora_up = torch.nn.Linear(5, 30)

    original_weight = org_module.weight.data.clone()

    # Call with low-rank SVD enabled
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=5, use_lowrank=True)

    # Verify weights are changed
    assert not torch.equal(original_weight, org_module.weight.data)


def test_initialize_pissa_with_lowrank_seed():
    # Test reproducibility with seed
    org_module = torch.nn.Linear(20, 10)
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    # First run with seed
    lora_down1 = torch.nn.Linear(20, 3)
    lora_up1 = torch.nn.Linear(3, 10)
    initialize_pissa(org_module, lora_down1, lora_up1, scale=0.1, rank=3, use_lowrank=True, lowrank_seed=42)

    result1_down = lora_down1.weight.data.clone()
    result1_up = lora_up1.weight.data.clone()

    # Reset module
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    # Second run with same seed
    lora_down2 = torch.nn.Linear(20, 3)
    lora_up2 = torch.nn.Linear(3, 10)
    initialize_pissa(org_module, lora_down2, lora_up2, scale=0.1, rank=3, use_lowrank=True, lowrank_seed=42)

    # Results should be identical
    torch.testing.assert_close(result1_down, lora_down2.weight.data)
    torch.testing.assert_close(result1_up, lora_up2.weight.data)


def test_initialize_pissa_ipca_with_lowrank():
    # Test IncrementalPCA with low-rank SVD enabled
    org_module = torch.nn.Linear(200, 100)  # Larger dimensions
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    lora_down = torch.nn.Linear(200, 10)
    lora_up = torch.nn.Linear(10, 100)

    # Call with both IPCA and low-rank enabled
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=10, use_ipca=True, use_lowrank=True, lowrank_q=20)

    # Check shapes of resulting matrices
    assert lora_down.weight.shape == torch.Size([10, 200])
    assert lora_up.weight.shape == torch.Size([100, 10])


def test_initialize_pissa_custom_lowrank_params():
    # Test with custom low-rank parameters
    org_module = torch.nn.Linear(30, 20)
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    lora_down = torch.nn.Linear(30, 5)
    lora_up = torch.nn.Linear(5, 20)

    # Test with custom q value and iterations
    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=5, use_lowrank=True, lowrank_q=12, lowrank_niter=6)

    # Check basic validity
    assert lora_down.weight.data is not None
    assert lora_up.weight.data is not None


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

        # Test with IPCA
        if device.type == "cpu":  # IPCA might be slow on CPU for large matrices
            org_module_small = torch.nn.Linear(20, 10).to(device)
            lora_down_small = torch.nn.Linear(20, 3).to(device)
            lora_up_small = torch.nn.Linear(3, 10).to(device)

            initialize_pissa(org_module_small, lora_down_small, lora_up_small, scale=0.1, rank=3, device=device, use_ipca=True)

            assert org_module_small.weight.data.device.type == device.type


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

        # Test with explicit dtype
        if dtype != torch.float16:  # Skip float16 for computational stability in SVD
            org_module2 = torch.nn.Linear(10, 5).to(dtype=torch.float32)
            lora_down2 = torch.nn.Linear(10, 2).to(dtype=torch.float32)
            lora_up2 = torch.nn.Linear(2, 5).to(dtype=torch.float32)

            initialize_pissa(org_module2, lora_down2, lora_up2, scale=0.1, rank=2, dtype=dtype)

            # Original module should be converted to specified dtype
            assert org_module2.weight.dtype == dtype



def test_initialize_pissa_numerical_stability():
    # Test with numerically challenging scenarios
    scenarios = [
        torch.randn(20, 10) * 1e-5,  # Small values
        torch.randn(20, 10) * 1e5,  # Large values
        torch.ones(20, 10),  # Uniform values
    ]

    for i, weight_matrix in enumerate(scenarios):
        org_module = torch.nn.Linear(20, 10)
        org_module.weight.data = weight_matrix

        lora_down = torch.nn.Linear(20, 3)
        lora_up = torch.nn.Linear(3, 10)

        try:
            initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=3)

            # Test IPCA as well
            lora_down_ipca = torch.nn.Linear(20, 3)
            lora_up_ipca = torch.nn.Linear(3, 10)
            initialize_pissa(org_module, lora_down_ipca, lora_up_ipca, scale=0.1, rank=3, use_ipca=True)
        except Exception as e:
            pytest.fail(f"Initialization failed for scenario ({i}): {e}")


def test_initialize_pissa_scale_effects():
    # Test effect of different scaling factors
    org_module = torch.nn.Linear(15, 10)
    original_weight = torch.randn_like(org_module.weight.data)
    org_module.weight.data = original_weight.clone()

    # Try different scales
    scales = [0.0, 0.01, 0.1, 1.0]

    for scale in scales:
        # Reset to original weights
        org_module.weight.data = original_weight.clone()

        lora_down = torch.nn.Linear(15, 4)
        lora_up = torch.nn.Linear(4, 10)

        initialize_pissa(org_module, lora_down, lora_up, scale=scale, rank=4)

        # Verify weight modification proportional to scale
        weight_diff = original_weight - org_module.weight.data

        # Approximate check of scaling effect
        if scale == 0.0:
            torch.testing.assert_close(weight_diff, torch.zeros_like(weight_diff), rtol=1e-4, atol=1e-6)
        else:
            # For non-zero scales, verify the magnitude of change is proportional to scale
            assert weight_diff.abs().sum() > 0

            # Do a second run with double the scale
            org_module2 = torch.nn.Linear(15, 10)
            org_module2.weight.data = original_weight.clone()

            lora_down2 = torch.nn.Linear(15, 4)
            lora_up2 = torch.nn.Linear(4, 10)

            initialize_pissa(org_module2, lora_down2, lora_up2, scale=scale * 2, rank=4)

            weight_diff2 = original_weight - org_module2.weight.data

            # The ratio of differences should be approximately 2
            # (allowing for numerical precision issues)
            ratio = weight_diff2.abs().sum() / (weight_diff.abs().sum() + 1e-10)
            assert 1.9 < ratio < 2.1

def test_initialize_pissa_large_matrix_performance():
    # Test with a large matrix to ensure it works well
    # This is particularly relevant for IPCA mode

    # Skip if running on CPU to avoid long test times
    if not torch.cuda.is_available():
        pytest.skip("Skipping large matrix test on CPU")

    org_module = torch.nn.Linear(1000, 500)
    org_module.weight.data = torch.randn_like(org_module.weight.data) * 0.1

    lora_down = torch.nn.Linear(1000, 16)
    lora_up = torch.nn.Linear(16, 500)

    # Test standard approach
    try:
        initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=16)
    except Exception as e:
        pytest.fail(f"Standard SVD failed on large matrix: {e}")

    # Test IPCA approach
    lora_down_ipca = torch.nn.Linear(1000, 16)
    lora_up_ipca = torch.nn.Linear(16, 500)

    try:
        initialize_pissa(org_module, lora_down_ipca, lora_up_ipca, scale=0.1, rank=16, use_ipca=True)
    except Exception as e:
        pytest.fail(f"IPCA approach failed on large matrix: {e}")

    # Test IPCA with lowrank
    lora_down_both = torch.nn.Linear(1000, 16)
    lora_up_both = torch.nn.Linear(16, 500)

    try:
        initialize_pissa(org_module, lora_down_both, lora_up_both, scale=0.1, rank=16, use_ipca=True, use_lowrank=True)
    except Exception as e:
        pytest.fail(f"Combined IPCA+lowrank approach failed on large matrix: {e}")


def test_initialize_pissa_requires_grad_preservation():
    # Test that requires_grad property is preserved
    org_module = torch.nn.Linear(20, 10)
    org_module.weight.requires_grad = False

    lora_down = torch.nn.Linear(20, 4)
    lora_up = torch.nn.Linear(4, 10)

    initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=4)

    # Check requires_grad is preserved
    assert not org_module.weight.requires_grad

    # Test with requires_grad=True
    org_module2 = torch.nn.Linear(20, 10)
    org_module2.weight.requires_grad = True

    initialize_pissa(org_module2, lora_down, lora_up, scale=0.1, rank=4)

    # Check requires_grad is preserved
    assert org_module2.weight.requires_grad


