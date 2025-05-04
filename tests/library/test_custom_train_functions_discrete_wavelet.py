import pytest
import torch
from torch import Tensor

from library.custom_train_functions import DiscreteWaveletTransform, WaveletTransform


class TestDiscreteWaveletTransform:
    @pytest.fixture
    def dwt(self):
        """Fixture to create a DiscreteWaveletTransform instance."""
        return DiscreteWaveletTransform(wavelet="db4", device=torch.device("cpu"))

    @pytest.fixture
    def sample_image(self):
        """Fixture to create a sample image tensor for testing."""
        # Create a 2x2x32x32 sample image (batch x channels x height x width)
        return torch.randn(2, 2, 32, 32)

    def test_initialization(self, dwt):
        """Test proper initialization of DWT with wavelet filters."""
        # Check if the base wavelet filters are initialized
        assert hasattr(dwt, "dec_lo") and dwt.dec_lo is not None
        assert hasattr(dwt, "dec_hi") and dwt.dec_hi is not None

        # Check filter dimensions for db4
        assert dwt.dec_lo.size(0) == 8
        assert dwt.dec_hi.size(0) == 8

    def test_dwt_single_level(self, dwt: DiscreteWaveletTransform, sample_image: Tensor):
        """Test single-level DWT decomposition."""
        x = sample_image

        # Perform single-level decomposition
        ll, lh, hl, hh = dwt._dwt_single_level(x)

        # Check that all subbands have the same shape
        assert ll.shape == lh.shape == hl.shape == hh.shape

        # Check that batch and channel dimensions are preserved
        assert ll.shape[0] == x.shape[0]
        assert ll.shape[1] == x.shape[1]

        # Calculate expected output size based on PyTorch's conv2d output size formula:
        # output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1

        filter_size = dwt.dec_lo.size(0)  # 8 for db4
        padding = filter_size // 2  # 4 for db4
        stride = 2  # Downsampling factor

        # For each dimension
        padded_height = x.shape[2] + 2 * padding
        padded_width = x.shape[3] + 2 * padding

        # PyTorch's conv2d formula with stride=2
        expected_height = (padded_height - filter_size) // stride + 1
        expected_width = (padded_width - filter_size) // stride + 1

        expected_shape = (x.shape[0], x.shape[1], expected_height, expected_width)

        assert ll.shape == expected_shape, f"Expected {expected_shape}, got {ll.shape}"

        # Test with different input sizes to verify consistency
        test_sizes = [(8, 8), (32, 32), (64, 64)]

        for h, w in test_sizes:
            test_input = torch.randn(2, 2, h, w)
            test_ll, _, _, _ = dwt._dwt_single_level(test_input)

            # Calculate expected shape
            pad_h = test_input.shape[2] + 2 * padding
            pad_w = test_input.shape[3] + 2 * padding
            exp_h = (pad_h - filter_size) // stride + 1
            exp_w = (pad_w - filter_size) // stride + 1
            exp_shape = (test_input.shape[0], test_input.shape[1], exp_h, exp_w)

            assert test_ll.shape == exp_shape, f"For input {test_input.shape}, expected {exp_shape}, got {test_ll.shape}"

        # Check energy preservation
        input_energy = torch.sum(x**2).item()
        output_energy = torch.sum(ll**2).item() + torch.sum(lh**2).item() + torch.sum(hl**2).item() + torch.sum(hh**2).item()

        # For orthogonal wavelets like db4, energy should be approximately preserved
        assert 0.9 <= output_energy / input_energy <= 1.11, (
            f"Energy ratio (output/input): {output_energy / input_energy:.4f} should be close to 1.0"
        )

    def test_decompose_structure(self, dwt, sample_image):
        """Test structure of decomposition result."""
        x = sample_image
        level = 2

        # Perform decomposition
        result = dwt.decompose(x, level=level)

        # Check structure of result
        bands = ["ll", "lh", "hl", "hh"]

        for band in bands:
            assert band in result
            assert len(result[band]) == level

    def test_decompose_shapes(self, dwt: DiscreteWaveletTransform, sample_image: Tensor):
        """Test shapes of decomposition coefficients."""
        x = sample_image
        level = 3

        # Perform decomposition
        result = dwt.decompose(x, level=level)

        # Filter size and padding
        filter_size = dwt.dec_lo.size(0)  # 8 for db4
        padding = filter_size // 2  # 4 for db4
        stride = 2  # Downsampling factor

        # Calculate expected shapes at each level
        expected_shapes = []
        current_h, current_w = x.shape[2], x.shape[3]

        for l in range(level):
            # Calculate shape for this level using PyTorch's conv2d formula
            padded_h = current_h + 2 * padding
            padded_w = current_w + 2 * padding
            output_h = (padded_h - filter_size) // stride + 1
            output_w = (padded_w - filter_size) // stride + 1

            expected_shapes.append((x.shape[0], x.shape[1], output_h, output_w))

            # Update for next level
            current_h, current_w = output_h, output_w

        # Check shapes of coefficients at each level
        for l in range(level):
            expected_shape = expected_shapes[l]

            # Verify all bands at this level have the correct shape
            for band in ["ll", "lh", "hl", "hh"]:
                assert result[band][l].shape == expected_shape, (
                    f"Level {l}, {band}: expected {expected_shape}, got {result[band][l].shape}"
                )

        # Verify length of output lists
        for band in ["ll", "lh", "hl", "hh"]:
            assert len(result[band]) == level, f"Expected {level} levels for {band}, got {len(result[band])}"

    def test_decompose_different_levels(self, dwt, sample_image):
        """Test decomposition with different levels."""
        x = sample_image

        # Test with different levels
        for level in [1, 2, 3]:
            result = dwt.decompose(x, level=level)

            # Check number of coefficients at each level
            for band in ["ll", "lh", "hl", "hh"]:
                assert len(result[band]) == level

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
        """Test DWT with different wavelet families."""
        dwt = DiscreteWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Simple test that decomposition works with this wavelet
        result = dwt.decompose(sample_image, level=1)

        # Basic structure check
        assert all(band in result for band in ["ll", "lh", "hl", "hh"])

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
    def test_different_wavelets_different_sizes(self, sample_image, wavelet):
        """Test DWT with different wavelet families and input sizes."""
        dwt = DiscreteWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Test with different input sizes to verify consistency
        test_sizes = [(8, 8), (32, 32), (64, 64)]

        for h, w in test_sizes:
            x = torch.randn(2, 2, h, w)
            test_ll, _, _, _ = dwt._dwt_single_level(x)

            filter_size = dwt.dec_lo.size(0)
            padding = filter_size // 2
            stride = 2

            # Calculate expected shape
            pad_h = x.shape[2] + 2 * padding
            pad_w = x.shape[3] + 2 * padding
            exp_h = (pad_h - filter_size) // stride + 1
            exp_w = (pad_w - filter_size) // stride + 1
            exp_shape = (x.shape[0], x.shape[1], exp_h, exp_w)

            assert test_ll.shape == exp_shape, f"For input {x.shape}, expected {exp_shape}, got {test_ll.shape}"

    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 1, 128, 128), (4, 3, 120, 160)])
    def test_different_input_shapes(self, shape):
        """Test DWT with different input shapes."""
        dwt = DiscreteWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(*shape)

        # Perform decomposition
        result = dwt.decompose(x, level=1)

        # Calculate expected shape using the actual implementation formula
        filter_size = dwt.dec_lo.size(0)  # 8 for db4
        padding = filter_size // 2  # 4 for db4
        stride = 2  # Downsampling factor

        # Calculate shape for this level using PyTorch's conv2d formula
        padded_h = shape[2] + 2 * padding
        padded_w = shape[3] + 2 * padding
        output_h = (padded_h - filter_size) // stride + 1
        output_w = (padded_w - filter_size) // stride + 1

        expected_shape = (shape[0], shape[1], output_h, output_w)

        # Check that all bands have the correct shape
        for band in ["ll", "lh", "hl", "hh"]:
            assert result[band][0].shape == expected_shape, (
                f"For input {shape}, {band}: expected {expected_shape}, got {result[band][0].shape}"
            )

        # Check that the decomposition preserves energy
        input_energy = torch.sum(x**2).item()

        # Calculate total energy across all subbands
        output_energy = 0
        for band in ["ll", "lh", "hl", "hh"]:
            output_energy += torch.sum(result[band][0] ** 2).item()

        # For orthogonal wavelets, energy should be preserved
        assert 0.9 <= output_energy / input_energy <= 1.1, (
            f"Energy ratio (output/input): {output_energy / input_energy:.4f} should be close to 1.0"
        )

    def test_device_support(self):
        """Test that DWT supports CPU and GPU (if available)."""
        # Test CPU
        cpu_device = torch.device("cpu")
        dwt_cpu = DiscreteWaveletTransform(device=cpu_device)
        assert dwt_cpu.dec_lo.device == cpu_device
        assert dwt_cpu.dec_hi.device == cpu_device

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda:0")
            dwt_gpu = DiscreteWaveletTransform(device=gpu_device)
            assert dwt_gpu.dec_lo.device == gpu_device
            assert dwt_gpu.dec_hi.device == gpu_device

    def test_base_class_abstract_method(self):
        """Test that base class requires implementation of decompose."""
        base_transform = WaveletTransform(wavelet="db4", device=torch.device("cpu"))

        with pytest.raises(NotImplementedError):
            base_transform.decompose(torch.randn(2, 2, 32, 32))
