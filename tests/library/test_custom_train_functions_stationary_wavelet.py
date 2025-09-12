import pytest
import torch
from torch import Tensor

from library.custom_train_functions import StationaryWaveletTransform


class TestStationaryWaveletTransform:
    @pytest.fixture
    def swt(self):
        """Fixture to create a StationaryWaveletTransform instance."""
        return StationaryWaveletTransform(wavelet="db4", device=torch.device("cpu"))

    @pytest.fixture
    def sample_image(self):
        """Fixture to create a sample image tensor for testing."""
        # Create a 2x2x32x32 sample image (batch x channels x height x width)
        return torch.randn(2, 2, 64, 64)

    def test_initialization(self, swt):
        """Test proper initialization of SWT with wavelet filters."""
        # Check if the base wavelet filters are initialized
        assert hasattr(swt, "dec_lo") and swt.dec_lo is not None
        assert hasattr(swt, "dec_hi") and swt.dec_hi is not None

        # Check filter dimensions for db4
        assert swt.dec_lo.size(0) == 8
        assert swt.dec_hi.size(0) == 8

    def test_swt_single_level(self, swt: StationaryWaveletTransform, sample_image: Tensor):
        """Test single-level SWT decomposition."""
        x = sample_image

        # Get level 0 filters (original filters)
        dec_lo, dec_hi = swt._get_filters_for_level(0)

        # Perform single-level decomposition
        ll, lh, hl, hh = swt._swt_single_level(x, dec_lo, dec_hi)

        # Check that all subbands have the same shape
        assert ll.shape == lh.shape == hl.shape == hh.shape

        # Check that batch and channel dimensions are preserved
        assert ll.shape[0] == x.shape[0]
        assert ll.shape[1] == x.shape[1]

        # SWT should maintain the same spatial dimensions as input
        assert ll.shape[2:] == x.shape[2:]

        # Test with different input sizes to verify consistency
        test_sizes = [(16, 16), (32, 32), (64, 64)]
        for h, w in test_sizes:
            test_input = torch.randn(2, 2, h, w)
            test_ll, test_lh, test_hl, test_hh = swt._swt_single_level(test_input, dec_lo, dec_hi)

            # Check output shape is same as input shape (no dimension change in SWT)
            assert test_ll.shape == test_input.shape
            assert test_lh.shape == test_input.shape
            assert test_hl.shape == test_input.shape
            assert test_hh.shape == test_input.shape

        # Check energy relationship
        input_energy = torch.sum(x**2).item()
        output_energy = torch.sum(ll**2).item() + torch.sum(lh**2).item() + torch.sum(hl**2).item() + torch.sum(hh**2).item()

        # For SWT, energy is not strictly preserved in the same way as DWT
        # But we can check the relationship is reasonable
        assert 0.5 <= output_energy / input_energy <= 5.0, (
            f"Energy ratio (output/input): {output_energy / input_energy:.4f} should be reasonable"
        )

    def test_decompose_structure(self, swt, sample_image):
        """Test structure of decomposition result."""
        x = sample_image
        level = 2

        # Perform decomposition
        result = swt.decompose(x, level=level)

        # Each entry should be a dictionary with aa, da, ad, dd keys
        for i in range(level):
            assert len(result["ll"]) == level
            assert len(result["lh"]) == level
            assert len(result["hl"]) == level
            assert len(result["hh"]) == level

    def test_decompose_shapes(self, swt: StationaryWaveletTransform, sample_image: Tensor):
        """Test shapes of decomposition coefficients."""
        x = sample_image
        level = 3

        # Perform decomposition
        result = swt.decompose(x, level=level)

        # All levels should maintain the same shape as the input
        expected_shape = x.shape

        # Check shapes of coefficients at each level
        for l in range(level):
            # Verify all bands at this level have the correct shape
            assert result["ll"][l].shape == expected_shape
            assert result["lh"][l].shape == expected_shape
            assert result["hl"][l].shape == expected_shape
            assert result["hh"][l].shape == expected_shape

    def test_decompose_different_levels(self, swt, sample_image):
        """Test decomposition with different levels."""
        x = sample_image

        # Test with different levels
        for level in [1, 2, 3]:
            result = swt.decompose(x, level=level)

            # Check number of levels
            assert len(result["ll"]) == level

            # All bands should maintain the same spatial dimensions
            for l in range(level):
                assert result["ll"][l].shape == x.shape
                assert result["lh"][l].shape == x.shape
                assert result["hl"][l].shape == x.shape
                assert result["hh"][l].shape == x.shape

    @pytest.mark.parametrize(
        "wavelet",
        [
            "db1",
            "db4",
            "sym4",
            "sym7",
            "haar",
            "coif3",
            "bior3.3",
            "rbio1.3",
            "dmey",
        ],
    )
    def test_different_wavelets(self, sample_image, wavelet):
        """Test SWT with different wavelet families."""
        swt = StationaryWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Simple test that decomposition works with this wavelet
        result = swt.decompose(sample_image, level=1)

        # Basic structure check
        assert len(result["ll"]) == 1

        # Check output dimensions match input
        assert result["ll"][0].shape == sample_image.shape
        assert result["lh"][0].shape == sample_image.shape
        assert result["hl"][0].shape == sample_image.shape
        assert result["hh"][0].shape == sample_image.shape

    @pytest.mark.parametrize(
        "wavelet",
        [
            "db1",
            "db4",
            "sym4",
            "haar",
        ],
    )
    def test_different_wavelets_different_sizes(self, wavelet):
        """Test SWT with different wavelet families and input sizes."""
        swt = StationaryWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Test with different input sizes to verify consistency
        test_sizes = [(16, 16), (32, 32), (64, 64)]

        for h, w in test_sizes:
            x = torch.randn(2, 2, h, w)

            # Perform decomposition
            result = swt.decompose(x, level=1)

            # Check shape matches input
            assert result["ll"][0].shape == x.shape
            assert result["lh"][0].shape == x.shape
            assert result["hl"][0].shape == x.shape
            assert result["hh"][0].shape == x.shape

    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 1, 128, 128), (4, 3, 120, 160)])
    def test_different_input_shapes(self, shape):
        """Test SWT with different input shapes."""
        swt = StationaryWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(*shape)

        # Perform decomposition
        result = swt.decompose(x, level=1)

        # SWT should maintain input dimensions
        expected_shape = shape

        # Check that all bands have the correct shape
        assert result["ll"][0].shape == expected_shape
        assert result["lh"][0].shape == expected_shape
        assert result["hl"][0].shape == expected_shape
        assert result["hh"][0].shape == expected_shape

        # Check energy relationship
        input_energy = torch.sum(x**2).item()

        # Calculate total energy across all subbands
        output_energy = (
            torch.sum(result["ll"][0] ** 2)
            + torch.sum(result["lh"][0] ** 2)
            + torch.sum(result["hl"][0] ** 2)
            + torch.sum(result["hh"][0] ** 2)
        ).item()

        # For SWT, energy relationship is different than DWT
        # Using a wider tolerance
        assert 0.5 <= output_energy / input_energy <= 5.0

    def test_device_support(self):
        """Test that SWT supports CPU and GPU (if available)."""
        # Test CPU
        cpu_device = torch.device("cpu")
        swt_cpu = StationaryWaveletTransform(device=cpu_device)
        assert swt_cpu.dec_lo.device == cpu_device
        assert swt_cpu.dec_hi.device == cpu_device

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda:0")
            swt_gpu = StationaryWaveletTransform(device=gpu_device)
            assert swt_gpu.dec_lo.device == gpu_device
            assert swt_gpu.dec_hi.device == gpu_device

    def test_multiple_level_decomposition(self, swt, sample_image):
        """Test multi-level SWT decomposition."""
        x = sample_image
        level = 3
        result = swt.decompose(x, level=level)

        # Check all levels maintain input dimensions
        for l in range(level):
            assert result["ll"][l].shape == x.shape
            assert result["lh"][l].shape == x.shape
            assert result["hl"][l].shape == x.shape
            assert result["hh"][l].shape == x.shape

    def test_odd_size_input(self):
        """Test SWT with odd-sized input."""
        swt = StationaryWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(2, 2, 33, 33)
        result = swt.decompose(x, level=1)

        # Check output shape matches input
        assert result["ll"][0].shape == x.shape
        assert result["lh"][0].shape == x.shape
        assert result["hl"][0].shape == x.shape
        assert result["hh"][0].shape == x.shape

    def test_small_input(self):
        """Test SWT with small input tensors."""
        swt = StationaryWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(2, 2, 16, 16)
        result = swt.decompose(x, level=1)

        # Check output shape matches input
        assert result["ll"][0].shape == x.shape
        assert result["lh"][0].shape == x.shape
        assert result["hl"][0].shape == x.shape
        assert result["hh"][0].shape == x.shape

    @pytest.mark.parametrize("input_size", [(12, 12), (15, 15), (20, 20)])
    def test_various_small_inputs(self, input_size):
        """Test SWT with various small input sizes."""
        swt = StationaryWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(2, 2, *input_size)
        result = swt.decompose(x, level=1)

        # Check output shape matches input
        assert result["ll"][0].shape == x.shape
        assert result["lh"][0].shape == x.shape
        assert result["hl"][0].shape == x.shape
        assert result["hh"][0].shape == x.shape

    def test_frequency_separation(self, swt, sample_image):
        """Test that SWT properly separates frequency components."""
        # Create synthetic image with distinct frequency components
        x = sample_image.clone()
        x[:, :, :, :] += 2.0
        result = swt.decompose(x, level=1)

        # The constant offset should be captured primarily in the LL band
        ll_mean = torch.mean(result["ll"][0]).item()
        lh_mean = torch.mean(result["lh"][0]).item()
        hl_mean = torch.mean(result["hl"][0]).item()
        hh_mean = torch.mean(result["hh"][0]).item()

        # LL should have the highest absolute mean
        assert abs(ll_mean) > abs(lh_mean)
        assert abs(ll_mean) > abs(hl_mean)
        assert abs(ll_mean) > abs(hh_mean)

    def test_level_progression(self, swt, sample_image):
        """Test that each level properly builds on the previous level."""
        x = sample_image
        level = 3
        result = swt.decompose(x, level=level)

        # Manually compute level-by-level to verify
        ll_current = x
        manual_results = []
        for l in range(level):
            # Get filters for current level
            dec_lo, dec_hi = swt._get_filters_for_level(l)
            ll_next, lh, hl, hh = swt._swt_single_level(ll_current, dec_lo, dec_hi)
            manual_results.append((ll_next, lh, hl, hh))
            ll_current = ll_next

        # Compare with the results from decompose
        for l in range(level):
            assert torch.allclose(manual_results[l][0], result["ll"][l])
            assert torch.allclose(manual_results[l][1], result["lh"][l])
            assert torch.allclose(manual_results[l][2], result["hl"][l])
            assert torch.allclose(manual_results[l][3], result["hh"][l])
