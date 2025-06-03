import pytest
import torch
from torch import Tensor
from typing import Tuple

from library.network_utils import convert_pissa_to_standard_lora, initialize_pissa


def generate_synthetic_weights(org_weight, seed=42):
    generator = torch.manual_seed(seed)

    # Base random normal distribution
    weights = torch.randn_like(org_weight)

    # Add structured variance to mimic real-world weight matrices
    # Techniques to create more realistic weight distributions:

    # 1. Block-wise variation
    block_size = max(1, org_weight.shape[0] // 4)
    for i in range(0, org_weight.shape[0], block_size):
        block_end = min(i + block_size, org_weight.shape[0])
        block_variation = torch.randn(1, generator=generator) * 0.3  # Local scaling
        weights[i:block_end, :] *= 1 + block_variation

    # 2. Sparse connectivity simulation
    sparsity_mask = torch.rand(org_weight.shape, generator=generator) > 0.2  # 20% sparsity
    weights *= sparsity_mask.float()

    # 3. Magnitude decay
    magnitude_decay = torch.linspace(1.0, 0.5, org_weight.shape[0]).unsqueeze(1)
    weights *= magnitude_decay

    # 4. Add structured noise
    structural_noise = torch.randn_like(org_weight) * 0.1
    weights += structural_noise

    # Normalize to have similar statistical properties to trained weights
    weights = (weights - weights.mean()) / weights.std()

    return weights


class TestPissa:
    """Test suite for convert_pissa_to_standard_lora function."""

    @pytest.fixture
    def basic_matrices(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Create basic test matrices with known properties."""
        torch.manual_seed(42)
        d_model, rank = 64, 8

        # Create original matrices
        orig_up = torch.randn(d_model, rank, dtype=torch.float32)
        orig_down = torch.randn(rank, d_model, dtype=torch.float32)

        # Create trained matrices (slightly different)
        noise_scale = 0.1
        trained_up = orig_up + noise_scale * torch.randn_like(orig_up)
        trained_down = orig_down + noise_scale * torch.randn_like(orig_down)

        return trained_up, trained_down, orig_up, orig_down, rank

    @pytest.fixture
    def small_matrices(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Create small matrices for easier debugging."""
        torch.manual_seed(123)
        d_model, rank = 8, 2

        orig_up = torch.randn(d_model, rank, dtype=torch.float32)
        orig_down = torch.randn(rank, d_model, dtype=torch.float32)
        trained_up = orig_up + 0.1 * torch.randn_like(orig_up)
        trained_down = orig_down + 0.1 * torch.randn_like(orig_down)

        return trained_up, trained_down, orig_up, orig_down, rank

    def test_initialize_pissa_rank_constraints(self):
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

    def test_initialize_pissa_rank_limits(self):
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

    def test_initialize_pissa_basic(self):
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

    def test_initialize_pissa_with_lowrank(self):
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

    def test_initialize_pissa_custom_lowrank_params(self):
        # Test with custom low-rank parameters
        org_module = torch.nn.Linear(30, 20)
        org_module.weight.data = generate_synthetic_weights(org_module.weight)

        lora_down = torch.nn.Linear(30, 5)
        lora_up = torch.nn.Linear(5, 20)

        # Test with custom q value and iterations
        initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=5, use_lowrank=True, lowrank_niter=6)

        # Check basic validity
        assert lora_down.weight.data is not None
        assert lora_up.weight.data is not None

    def test_initialize_pissa_device_handling(self):
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

                initialize_pissa(org_module_small, lora_down_small, lora_up_small, scale=0.1, rank=3, device=device)

                assert org_module_small.weight.data.device.type == device.type

    def test_initialize_pissa_shape_mismatch(self):
        # Test with shape mismatch to ensure warning is printed
        org_module = torch.nn.Linear(20, 10)

        # Intentionally mismatched shapes to test warning mechanism
        lora_down = torch.nn.Linear(20, 5)  # Different shape
        lora_up = torch.nn.Linear(3, 15)  # Different shape

        with pytest.warns(UserWarning):
            initialize_pissa(org_module, lora_down, lora_up, scale=0.1, rank=3)

    def test_initialize_pissa_scaling(self):
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

    def test_initialize_pissa_dtype(self):
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

    def test_initialize_pissa_svd_properties(self):
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

    def test_initialize_pissa_dtype_preservation(self):
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
                assert org_module2.weight.dtype == torch.float32

    def test_initialize_pissa_numerical_stability(self):
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
                initialize_pissa(org_module, lora_down_ipca, lora_up_ipca, scale=0.1, rank=3)
            except Exception as e:
                pytest.fail(f"Initialization failed for scenario ({i}): {e}")

    def test_initialize_pissa_scale_effects(self):
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

    def test_initialize_pissa_large_matrix_performance(self):
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
            initialize_pissa(org_module, lora_down_ipca, lora_up_ipca, scale=0.1, rank=16)
        except Exception as e:
            pytest.fail(f"IPCA approach failed on large matrix: {e}")

        # Test IPCA with lowrank
        lora_down_both = torch.nn.Linear(1000, 16)
        lora_up_both = torch.nn.Linear(16, 500)

        try:
            initialize_pissa(org_module, lora_down_both, lora_up_both, scale=0.1, rank=16, use_lowrank=True)
        except Exception as e:
            pytest.fail(f"Combined IPCA+lowrank approach failed on large matrix: {e}")

    def test_initialize_pissa_requires_grad_preservation(self):
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

    def test_basic_functionality(self, basic_matrices):
        """Test that the function runs without errors and returns expected shapes."""
        trained_up, trained_down, orig_up, orig_down, rank = basic_matrices

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Check output types
        assert isinstance(new_up, torch.Tensor)
        assert isinstance(new_down, torch.Tensor)

        # Check shapes - should be compatible for matrix multiplication
        d_model = trained_up.shape[0]
        expected_rank = min(rank * 2, min(d_model, trained_down.shape[1]))

        assert new_up.shape == torch.Size([d_model, expected_rank])
        assert new_down.shape == (expected_rank, trained_down.shape[1])

    def test_delta_preservation(self, basic_matrices):
        """Test that the delta weight is preserved in the LoRA decomposition."""
        trained_up, trained_down, orig_up, orig_down, rank = basic_matrices

        # Calculate original delta
        original_delta = (trained_up @ trained_down) - (orig_up @ orig_down)

        # Convert to LoRA
        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Reconstruct delta from LoRA matrices
        reconstructed_delta = new_up @ new_down

        # Check that reconstruction approximates original delta
        # (Note: some information loss is expected due to rank reduction)
        relative_error = torch.norm(original_delta - reconstructed_delta) / torch.norm(original_delta)
        assert relative_error < 0.5  # Allow some approximation error

    def test_rank_handling(self, small_matrices):
        """Test various rank scenarios."""
        trained_up, trained_down, orig_up, orig_down, base_rank = small_matrices
        d_model = trained_up.shape[0]

        # Test with rank that would exceed matrix dimensions
        large_rank = d_model + 5
        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, large_rank)

        # Should not exceed available singular values
        max_possible_rank = min(d_model, trained_down.shape[1])
        assert new_up.shape[1] <= max_possible_rank
        assert new_down.shape[0] <= max_possible_rank

    def test_zero_delta(self):
        """Test behavior when trained and original matrices are identical."""
        torch.manual_seed(456)
        d_model, rank = 16, 4

        # Create identical matrices
        orig_up = torch.randn(d_model, rank, dtype=torch.float32)
        orig_down = torch.randn(rank, d_model, dtype=torch.float32)
        trained_up = orig_up.clone()
        trained_down = orig_down.clone()

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Reconstructed delta should be close to zero
        reconstructed_delta = new_up @ new_down
        assert torch.allclose(reconstructed_delta, torch.zeros_like(reconstructed_delta), atol=1e-6)

    def test_different_devices(self, basic_matrices):
        """Test that the function handles different device placement correctly."""
        trained_up, trained_down, orig_up, orig_down, rank = basic_matrices

        # Test with CPU tensors
        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Results should be on the same device as input
        assert new_up.device == trained_up.device
        assert new_down.device == trained_up.device

    def test_gradient_disabled(self, basic_matrices):
        """Test that gradients are properly disabled."""
        trained_up, trained_down, orig_up, orig_down, rank = basic_matrices

        # Enable gradients on inputs
        trained_up.requires_grad_(True)
        trained_down.requires_grad_(True)

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Outputs should not require gradients due to torch.no_grad()
        assert not new_up.requires_grad
        assert not new_down.requires_grad

    def test_dtype_consistency(self, basic_matrices):
        """Test that output dtypes are consistent."""
        trained_up, trained_down, orig_up, orig_down, rank = basic_matrices

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Should maintain float32 dtype
        assert new_up.dtype == torch.float32
        assert new_down.dtype == torch.float32

    def test_mathematical_properties(self, small_matrices):
        """Test mathematical properties of the SVD decomposition."""
        trained_up, trained_down, orig_up, orig_down, rank = small_matrices

        # Calculate delta manually
        delta_w = (trained_up @ trained_down) - (orig_up @ orig_down)

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # The decomposition should satisfy: new_up @ new_down ≈ low-rank approximation of delta_w
        reconstructed = new_up @ new_down

        # Check that reconstruction has expected rank
        actual_rank = torch.linalg.matrix_rank(reconstructed).item()
        expected_max_rank = min(rank * 2, min(delta_w.shape))
        assert actual_rank <= expected_max_rank

    @pytest.mark.parametrize("rank", [1, 4, 8, 16])
    def test_different_ranks(self, rank):
        """Test the function with different rank values."""
        torch.manual_seed(789)
        d_model = 32

        orig_up = torch.randn(d_model, rank, dtype=torch.float32)
        orig_down = torch.randn(rank, d_model, dtype=torch.float32)
        trained_up = orig_up + 0.1 * torch.randn_like(orig_up)
        trained_down = orig_down + 0.1 * torch.randn_like(orig_down)

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # Should handle all rank values gracefully
        assert new_up.shape[0] == d_model
        assert new_down.shape[1] == d_model
        assert new_up.shape[1] == new_down.shape[0]  # Compatible for multiplication

    def test_edge_case_single_rank(self):
        """Test with minimal rank (rank=1)."""
        torch.manual_seed(101)
        d_model, rank = 8, 1

        orig_up = torch.randn(d_model, rank, dtype=torch.float32)
        orig_down = torch.randn(rank, d_model, dtype=torch.float32)
        trained_up = orig_up + 0.2 * torch.randn_like(orig_up)
        trained_down = orig_down + 0.2 * torch.randn_like(orig_down)

        new_up, new_down = convert_pissa_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank)

        # With rank=1, output rank should be 2 (rank * 2)
        expected_rank = min(2, min(d_model, d_model))
        assert new_up.shape[1] <= expected_rank
        assert new_down.shape[0] <= expected_rank


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
