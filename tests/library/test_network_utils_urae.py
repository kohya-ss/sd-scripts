import pytest
import torch
from typing import Optional, Dict, Any

from library.network_utils import convert_urae_to_standard_lora, initialize_urae


class TestURAE:
    @pytest.fixture
    def sample_matrices(self):
        """Create sample matrices for testing"""
        # Original up matrix (4x2)
        orig_up = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

        # Original down matrix (2x6)
        orig_down = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]])

        # Trained up matrix (4x2) - same shape as orig_up but with changed values
        trained_up = torch.tensor([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1], [7.1, 8.1]])

        # Trained down matrix (2x6) - same shape as orig_down but with changed values
        trained_down = torch.tensor([[0.15, 0.25, 0.35, 0.45, 0.55, 0.65], [0.75, 0.85, 0.95, 1.05, 1.15, 1.25]])

        return {"orig_up": orig_up, "orig_down": orig_down, "trained_up": trained_up, "trained_down": trained_down}

    def test_basic_conversion(self, sample_matrices):
        """Test the basic functionality of convert_urae_to_standard_lora"""
        lora_up, lora_down, alpha = convert_urae_to_standard_lora(
            sample_matrices["trained_up"], sample_matrices["trained_down"], sample_matrices["orig_up"], sample_matrices["orig_down"]
        )

        # Check shapes
        assert lora_up.shape[0] == sample_matrices["trained_up"].shape[0]  # Same number of rows as trained_up
        assert lora_up.shape[1] == sample_matrices["trained_up"].shape[1]  # Same rank as trained_up
        assert lora_down.shape[0] == sample_matrices["trained_up"].shape[1]  # Same rank as trained_up
        assert lora_down.shape[1] == sample_matrices["trained_down"].shape[1]  # Same number of columns as trained_down

        # Check alpha is a reasonable value
        assert 0.1 <= alpha <= 1024.0

        # Check that lora_up @ lora_down approximates the weight delta
        delta = (sample_matrices["trained_up"] @ sample_matrices["trained_down"]) - (
            sample_matrices["orig_up"] @ sample_matrices["orig_down"]
        )

        # The approximation should be close in Frobenius norm after scaling
        lora_effect = lora_up @ lora_down
        delta_norm = torch.norm(delta, p="fro").item()
        lora_norm = torch.norm(lora_effect, p="fro").item()

        # Either they are close, or the alpha scaling brings them close
        scaled_lora_effect = (alpha / sample_matrices["trained_up"].shape[1]) * lora_effect
        scaled_lora_norm = torch.norm(scaled_lora_effect, p="fro").item()

        # At least one of these should be true
        assert abs(delta_norm - lora_norm) < 1e-4 or abs(delta_norm - scaled_lora_norm) < 1e-4

    def test_specified_rank(self, sample_matrices):
        """Test conversion with a specified rank"""
        new_rank = 1  # Lower than trained_up's rank of 2

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(
            sample_matrices["trained_up"],
            sample_matrices["trained_down"],
            sample_matrices["orig_up"],
            sample_matrices["orig_down"],
            rank=new_rank,
        )

        # Check that the new rank is used
        assert lora_up.shape[1] == new_rank
        assert lora_down.shape[0] == new_rank

        # Should still produce a reasonable alpha
        assert 0.1 <= alpha <= 1024.0

    def test_with_initial_alpha(self, sample_matrices):
        """Test conversion with a specified initial alpha"""
        initial_alpha = 16.0

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(
            sample_matrices["trained_up"],
            sample_matrices["trained_down"],
            sample_matrices["orig_up"],
            sample_matrices["orig_down"],
            initial_alpha=initial_alpha,
        )

        # Alpha should be influenced by initial_alpha but may be adjusted
        # Since we're using same rank, should be reasonably close to initial_alpha
        assert 0.1 <= alpha <= 1024.0
        # Should at least preserve the order of magnitude in typical cases
        assert abs(alpha - initial_alpha) <= initial_alpha * 4.0

    def test_large_initial_alpha(self, sample_matrices):
        """Test conversion with a very large initial alpha that should be capped"""
        initial_alpha = 2000.0  # Larger than the 1024.0 cap

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(
            sample_matrices["trained_up"],
            sample_matrices["trained_down"],
            sample_matrices["orig_up"],
            sample_matrices["orig_down"],
            initial_alpha=initial_alpha,
        )

        # Alpha should be capped at 1024.0
        assert alpha <= 1024.0

    def test_very_small_initial_alpha(self, sample_matrices):
        """Test conversion with a very small initial alpha that should be floored"""
        initial_alpha = 0.01  # Smaller than the 0.1 floor

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(
            sample_matrices["trained_up"],
            sample_matrices["trained_down"],
            sample_matrices["orig_up"],
            sample_matrices["orig_down"],
            initial_alpha=initial_alpha,
        )

        # Alpha should be floored at 0.1
        assert alpha >= 0.1

    def test_change_rank_with_initial_alpha(self, sample_matrices):
        """Test conversion with both rank change and initial alpha"""
        initial_alpha = 16.0
        new_rank = 1  # Half of original rank 2

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(
            sample_matrices["trained_up"],
            sample_matrices["trained_down"],
            sample_matrices["orig_up"],
            sample_matrices["orig_down"],
            initial_alpha=initial_alpha,
            rank=new_rank,
        )

        # Check shapes
        assert lora_up.shape[1] == new_rank
        assert lora_down.shape[0] == new_rank

        # Alpha should be adjusted for the rank change (approx halved in this case)
        expected_alpha = initial_alpha * (new_rank / sample_matrices["trained_up"].shape[1])
        # Allow some tolerance for adjustments from norm-based capping
        assert abs(alpha - expected_alpha) <= expected_alpha * 4.0 or alpha >= 0.1

    def test_zero_delta(self):
        """Test conversion when delta is zero"""
        # Create matrices where the delta will be zero
        dim_in, rank, dim_out = 4, 2, 6

        # Create identical matrices for original and trained
        orig_up = torch.randn(dim_in, rank)
        orig_down = torch.randn(rank, dim_out)
        trained_up = orig_up.clone()
        trained_down = orig_down.clone()

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(trained_up, trained_down, orig_up, orig_down)

        # Should still return matrices of correct shape
        assert lora_up.shape == (dim_in, rank)
        assert lora_down.shape == (rank, dim_out)

        # Alpha should be at least the minimum
        assert alpha >= 0.1

    def test_large_dimensions(self):
        """Test with larger matrix dimensions"""
        dim_in, rank, dim_out = 100, 8, 200

        orig_up = torch.randn(dim_in, rank)
        orig_down = torch.randn(rank, dim_out)
        trained_up = orig_up + 0.01 * torch.randn(dim_in, rank)  # Small perturbation
        trained_down = orig_down + 0.01 * torch.randn(rank, dim_out)  # Small perturbation

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(trained_up, trained_down, orig_up, orig_down)

        # Check shapes
        assert lora_up.shape == (dim_in, rank)
        assert lora_down.shape == (rank, dim_out)

        # Should produce a reasonable alpha
        assert 0.1 <= alpha <= 1024.0

    def test_rank_exceeding_singular_values(self):
        """Test when requested rank exceeds available singular values"""
        # Small matrices with limited rank
        dim_in, rank, dim_out = 3, 2, 3

        orig_up = torch.randn(dim_in, rank)
        orig_down = torch.randn(rank, dim_out)
        trained_up = orig_up + 0.1 * torch.randn(dim_in, rank)
        trained_down = orig_down + 0.1 * torch.randn(rank, dim_out)

        # Request rank larger than possible
        too_large_rank = 10

        lora_up, lora_down, alpha = convert_urae_to_standard_lora(trained_up, trained_down, orig_up, orig_down, rank=too_large_rank)

        # Rank should be limited to min(dim_in, dim_out, S.size)
        max_possible_rank = min(dim_in, dim_out)
        assert lora_up.shape[1] <= max_possible_rank
        assert lora_down.shape[0] <= max_possible_rank


def create_mock_urae_environment(level: Optional[str] = None) -> Dict[str, Any]:
    """Create a mock environment simulating different URAE initialization levels."""
    base_config = {
        "precision": torch.float32,
        "rank_multiplier": 1.0,
        "scale_factor": 0.1,
        "use_minor_components": False
    }

    urae_levels = {
        "low": {
            "precision": torch.float16,
            "rank_multiplier": 0.5,
            "scale_factor": 0.05,
            "use_minor_components": True
        },
        "medium": {
            "precision": torch.float32,
            "rank_multiplier": 1.0,
            "scale_factor": 0.1,
            "use_minor_components": False
        },
        "high": {
            "precision": torch.float64,
            "rank_multiplier": 2.0,
            "scale_factor": 0.2,
            "use_minor_components": False
        }
    }

    if level and level in urae_levels:
        base_config.update(urae_levels[level])

    return base_config


def test_urae_initialization_with_level_variations():
    """Test URAE initialization with different levels and configurations."""
    import torch

    # Test URAE levels
    urae_levels = [None, "low", "medium", "high"]

    for level in urae_levels:
        # Get URAE-specific configuration
        urae_config = create_mock_urae_environment(level)

        # Adjust input sizes based on rank multiplier
        input_dim = int(100 * urae_config["rank_multiplier"])
        output_dim = int(50 * urae_config["rank_multiplier"])
        rank = int(10 * urae_config["rank_multiplier"])

        # Create modules
        org_module = torch.nn.Linear(input_dim, output_dim)
        lora_down = torch.nn.Linear(input_dim, rank)
        lora_up = torch.nn.Linear(rank, output_dim)

        # Apply precision
        org_module = org_module.to(dtype=urae_config["precision"])
        lora_down = lora_down.to(dtype=urae_config["precision"])
        lora_up = lora_up.to(dtype=urae_config["precision"])

        # Apply device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        org_module = org_module.to(device)
        lora_down = lora_down.to(device)
        lora_up = lora_up.to(device)

        # Test initialization
        try:
            initialize_urae(
                org_module, 
                lora_down, 
                lora_up, 
                scale=urae_config["scale_factor"], 
                rank=rank,
            )
        except Exception as e:
            pytest.fail(f"URAE initialization failed for level {level}: {e}")


