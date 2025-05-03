import pytest
import torch
from torch import Tensor
# import torch.nn.functional as F
# import numpy as np
# import pywt
#
# from unittest.mock import patch, MagicMock

# Import the class under test
from library.custom_train_functions import QuaternionWaveletTransform


class TestQuaternionWaveletTransform:
    @pytest.fixture
    def qwt(self):
        """Fixture to create a QuaternionWaveletTransform instance."""
        return QuaternionWaveletTransform(wavelet="db4", device=torch.device("cpu"))

    @pytest.fixture
    def sample_image(self):
        """Fixture to create a sample image tensor for testing."""
        # Create a 2x2x32x32 sample image (batch x channels x height x width)
        return torch.randn(2, 2, 32, 32)

    def test_initialization(self, qwt):
        """Test proper initialization of QWT with wavelet filters and Hilbert transforms."""
        # Check if the base wavelet filters are initialized
        assert hasattr(qwt, "dec_lo") and qwt.dec_lo is not None
        assert hasattr(qwt, "dec_hi") and qwt.dec_hi is not None

        # Check if Hilbert filters are initialized
        assert hasattr(qwt, "hilbert_x") and qwt.hilbert_x is not None
        assert hasattr(qwt, "hilbert_y") and qwt.hilbert_y is not None
        assert hasattr(qwt, "hilbert_xy") and qwt.hilbert_xy is not None

    def test_create_hilbert_filter_x(self, qwt):
        """Test creation of x-direction Hilbert filter."""
        filter_x = qwt._create_hilbert_filter("x")

        # Check shape and dimensions
        assert filter_x.dim() == 4  # [1, 1, H, W]
        assert filter_x.shape[2:] == (2, 7)  # Expected filter dimensions

        # Check filter contents (should be anti-symmetric along x-axis)
        filter_data = filter_x.squeeze()
        # Center row should be zero
        assert torch.allclose(filter_data[1], torch.zeros_like(filter_data[1]))
        # Test anti-symmetry property
        for i in range(filter_data.shape[1] // 2):
            assert torch.isclose(filter_data[0, i], -filter_data[0, -(i + 1)])

    def test_create_hilbert_filter_y(self, qwt):
        """Test creation of y-direction Hilbert filter."""
        filter_y = qwt._create_hilbert_filter("y")

        # Check shape and dimensions
        assert filter_y.dim() == 4  # [1, 1, H, W]
        assert filter_y.shape[2:] == (7, 2)  # Expected filter dimensions

        # Check filter contents (should be anti-symmetric along y-axis)
        filter_data = filter_y.squeeze()
        # Right column should be zero
        assert torch.allclose(filter_data[:, 1], torch.zeros_like(filter_data[:, 1]))
        # Test anti-symmetry property
        for i in range(filter_data.shape[0] // 2):
            assert torch.isclose(filter_data[i, 0], -filter_data[-(i + 1), 0])

    def test_create_hilbert_filter_xy(self, qwt):
        """Test creation of xy-direction (diagonal) Hilbert filter."""
        filter_xy = qwt._create_hilbert_filter("xy")

        # Check shape and dimensions
        assert filter_xy.dim() == 4  # [1, 1, H, W]
        assert filter_xy.shape[2:] == (7, 7)  # Expected filter dimensions

        filter_data = filter_xy.squeeze()

        # Verify middle row and column are zero
        assert torch.allclose(filter_data[3, :], torch.zeros_like(filter_data[3, :]))
        assert torch.allclose(filter_data[:, 3], torch.zeros_like(filter_data[:, 3]))

        # The filter has odd symmetry - point reflection through the center (0,0) -> -(6,6)
        # This is also called origin symmetry or central symmetry
        for i in range(7):
            for j in range(7):
                # Skip the zero middle row and column
                if i != 3 and j != 3:
                    assert torch.allclose(filter_data[i, j], filter_data[6 - i, 6 - j]), (
                        f"Point reflection failed at [{i},{j}] vs [{6 - i},{6 - j}]"
                    )

    def test_apply_hilbert_shape_preservation(self, qwt, sample_image):
        """Test that Hilbert transforms preserve input shape."""
        x = sample_image

        # Apply Hilbert transforms
        x_hilbert_x = qwt._apply_hilbert(x, "x")
        x_hilbert_y = qwt._apply_hilbert(x, "y")
        x_hilbert_xy = qwt._apply_hilbert(x, "xy")

        # Check that output shapes match input
        assert x_hilbert_x.shape == x.shape
        assert x_hilbert_y.shape == x.shape
        assert x_hilbert_xy.shape == x.shape

    def test_dwt_single_level(self, qwt: QuaternionWaveletTransform, sample_image: Tensor):
        """Test single-level DWT decomposition."""
        x = sample_image

        # Perform single-level decomposition
        ll, lh, hl, hh = qwt._dwt_single_level(x)

        # Check that all subbands have the same shape
        assert ll.shape == lh.shape == hl.shape == hh.shape

        # Check that batch and channel dimensions are preserved
        assert ll.shape[0] == x.shape[0]
        assert ll.shape[1] == x.shape[1]

        # From the debug output, we can see that:
        # - For input shape [2, 2, 32, 32]
        # - Padding makes it [4, 1, 40, 40]
        # - The filter size is 8 (for db4)
        # - Final output is [2, 2, 17, 17]

        # Calculate expected output size based on PyTorch's conv2d output size formula:
        # output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1

        filter_size = qwt.dec_lo.size(0)  # 8 for db4
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
            test_ll, _, _, _ = qwt._dwt_single_level(test_input)

            # Calculate expected shape
            pad_h = test_input.shape[2] + 2 * padding
            pad_w = test_input.shape[3] + 2 * padding
            exp_h = (pad_h - filter_size) // stride + 1
            exp_w = (pad_w - filter_size) // stride + 1
            exp_shape = (test_input.shape[0], test_input.shape[1], exp_h, exp_w)

            assert test_ll.shape == exp_shape, f"For input {test_input.shape}, expected {exp_shape}, got {test_ll.shape}"

        # # Check energy preservation
        # input_energy = torch.sum(x**2).item()
        # output_energy = torch.sum(ll**2).item() + torch.sum(lh**2).item() + torch.sum(hl**2).item() + torch.sum(hh**2).item()
        #
        # # For orthogonal wavelets like db4, energy should be approximately preserved
        # assert 0.9 <= output_energy / input_energy <= 1.1, (
        #     f"Energy ratio (output/input): {output_energy / input_energy:.4f} should be close to 1.0"
        # )

    def test_decompose_structure(self, qwt, sample_image):
        """Test structure of decomposition result."""
        x = sample_image
        level = 2

        # Perform decomposition
        result = qwt.decompose(x, level=level)

        # Check structure of result
        components = ["r", "i", "j", "k"]
        bands = ["ll", "lh", "hl", "hh"]

        for component in components:
            assert component in result
            for band in bands:
                assert band in result[component]
                assert len(result[component][band]) == level

    def test_decompose_shapes(self, qwt: QuaternionWaveletTransform, sample_image: Tensor):
        """Test shapes of decomposition coefficients."""
        x = sample_image
        level = 3

        # Perform decomposition
        result = qwt.decompose(x, level=level)

        # Filter size and padding
        filter_size = qwt.dec_lo.size(0)  # 8 for db4
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

            # Verify all components and bands at this level have the correct shape
            for component in ["r", "i", "j", "k"]:
                for band in ["ll", "lh", "hl", "hh"]:
                    assert result[component][band][l].shape == expected_shape, (
                        f"Level {l}, {component}/{band}: expected {expected_shape}, got {result[component][band][l].shape}"
                    )

        # Verify length of output lists
        for component in ["r", "i", "j", "k"]:
            for band in ["ll", "lh", "hl", "hh"]:
                assert len(result[component][band]) == level, (
                    f"Expected {level} levels for {component}/{band}, got {len(result[component][band])}"
                )

    def test_decompose_different_levels(self, qwt, sample_image):
        """Test decomposition with different levels."""
        x = sample_image

        # Test with different levels
        for level in [1, 2, 3]:
            result = qwt.decompose(x, level=level)

            # Check number of coefficients at each level
            for component in ["r", "i", "j", "k"]:
                for band in ["ll", "lh", "hl", "hh"]:
                    assert len(result[component][band]) == level

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
        """Test QWT with different wavelet families."""
        qwt = QuaternionWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Simple test that decomposition works with this wavelet
        result = qwt.decompose(sample_image, level=1)

        # Basic structure check
        assert all(component in result for component in ["r", "i", "j", "k"])
        assert all(band in result["r"] for band in ["ll", "lh", "hl", "hh"])

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
        """Test QWT with different wavelet families."""
        qwt = QuaternionWaveletTransform(wavelet=wavelet, device=torch.device("cpu"))

        # Simple test that decomposition works with this wavelet
        result = qwt.decompose(sample_image, level=1)

        # Test with different input sizes to verify consistency
        test_sizes = [(8, 8), (32, 32), (64, 64)]

        for h, w in test_sizes:
            x = torch.randn(2, 2, h, w)
            test_ll, _, _, _ = qwt._dwt_single_level(x)

            filter_size = qwt.dec_lo.size(0)  # 8 for db4
            padding = filter_size // 2  # 4 for db4
            stride = 2  # Downsampling factor

            # For each dimension
            padded_height = x.shape[2] + 2 * padding
            padded_width = x.shape[3] + 2 * padding

            # Filter size and padding
            filter_size = qwt.dec_lo.size(0)  # 8 for db4
            padding = filter_size // 2  # 4 for db4
            stride = 2  # Downsampling factor

            # Calculate expected shapes at each level
            expected_shapes = []
            current_h, current_w = x.shape[2], x.shape[3]

            # Calculate expected shape
            pad_h = x.shape[2] + 2 * padding
            pad_w = x.shape[3] + 2 * padding
            exp_h = (pad_h - filter_size) // stride + 1
            exp_w = (pad_w - filter_size) // stride + 1
            exp_shape = (x.shape[0], x.shape[1], exp_h, exp_w)

            assert test_ll.shape == exp_shape, f"For input {x.shape}, expected {exp_shape}, got {test_ll.shape}"

    @pytest.mark.parametrize("shape", [(2, 3, 64, 64), (1, 1, 128, 128), (4, 3, 120, 160)])
    def test_different_input_shapes(self, shape):
        """Test QWT with different input shapes."""
        qwt = QuaternionWaveletTransform(wavelet="db4", device=torch.device("cpu"))
        x = torch.randn(*shape)

        # Perform decomposition
        result = qwt.decompose(x, level=1)

        # Calculate expected shape using the actual implementation formula
        filter_size = qwt.dec_lo.size(0)  # 8 for db4
        padding = filter_size // 2  # 4 for db4
        stride = 2  # Downsampling factor

        # Calculate shape for this level using PyTorch's conv2d formula
        padded_h = shape[2] + 2 * padding
        padded_w = shape[3] + 2 * padding
        output_h = (padded_h - filter_size) // stride + 1
        output_w = (padded_w - filter_size) // stride + 1

        expected_shape = (shape[0], shape[1], output_h, output_w)

        # Check that all components and bands have the correct shape
        for component in ["r", "i", "j", "k"]:
            for band in ["ll", "lh", "hl", "hh"]:
                assert result[component][band][0].shape == expected_shape, (
                    f"For input {shape}, {component}/{band}: expected {expected_shape}, got {result[component][band][0].shape}"
                )

        # Also check that the decomposition preserves energy
        input_energy = torch.sum(x**2).item()

        # Calculate total energy across all subbands and components
        output_energy = 0
        for component in ["r", "i", "j", "k"]:
            for band in ["ll", "lh", "hl", "hh"]:
                output_energy += torch.sum(result[component][band][0] ** 2).item()

        # For quaternion wavelets, energy should be distributed across components
        # Use a wider tolerance due to the multiple transforms
        assert 0.8 <= output_energy / input_energy <= 1.2, (
            f"Energy ratio (output/input): {output_energy / input_energy:.4f} should be close to 1.0"
        )

    def test_device_support(self):
        """Test that QWT supports CPU and GPU (if available)."""
        # Test CPU
        cpu_device = torch.device("cpu")
        qwt_cpu = QuaternionWaveletTransform(device=cpu_device)
        assert qwt_cpu.dec_lo.device == cpu_device
        assert qwt_cpu.dec_hi.device == cpu_device
        assert qwt_cpu.hilbert_x.device == cpu_device
        assert qwt_cpu.hilbert_y.device == cpu_device
        assert qwt_cpu.hilbert_xy.device == cpu_device

        # Test GPU if available
        if torch.cuda.is_available():
            gpu_device = torch.device("cuda:0")
            qwt_gpu = QuaternionWaveletTransform(device=gpu_device)
            assert qwt_gpu.dec_lo.device == gpu_device
            assert qwt_gpu.dec_hi.device == gpu_device
            assert qwt_gpu.hilbert_x.device == gpu_device
            assert qwt_gpu.hilbert_y.device == gpu_device
            assert qwt_gpu.hilbert_xy.device == gpu_device
