import pytest
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from library.custom_train_functions import (
    WaveletLoss,
    DiscreteWaveletTransform,
    StationaryWaveletTransform,
    QuaternionWaveletTransform,
)


class TestWaveletLoss:
    @pytest.fixture(autouse=True)
    def no_grad_context(self):
        with torch.no_grad():
            yield

    @pytest.fixture
    def setup_inputs(self):
        # Create simple test inputs
        batch_size = 2
        channels = 3
        height = 64
        width = 64

        # Create predictable patterns for testing
        pred = torch.zeros(batch_size, channels, height, width)
        target = torch.zeros(batch_size, channels, height, width)

        # Add some patterns
        for b in range(batch_size):
            for c in range(channels):
                # Create different patterns for pred and target
                pred[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(1, -1) * torch.sin(
                    torch.linspace(0, 4 * np.pi, height)
                ).view(-1, 1)
                target[b, c] = torch.sin(torch.linspace(0, 4 * np.pi, width)).view(1, -1) * torch.sin(
                    torch.linspace(0, 4 * np.pi, height)
                ).view(-1, 1)

                # Add some differences
                if b == 1:
                    pred[b, c] += 0.2 * torch.randn(height, width)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return pred.to(device), target.to(device), device

    def test_init_dwt(self, setup_inputs):
        _, _, device = setup_inputs
        loss_fn = WaveletLoss(wavelet="db4", level=3, transform_type="dwt", device=device)

        assert loss_fn.level == 3
        assert loss_fn.wavelet == "db4"
        assert loss_fn.transform_type == "dwt"
        assert isinstance(loss_fn.transform, DiscreteWaveletTransform)
        assert hasattr(loss_fn, "dec_lo")
        assert hasattr(loss_fn, "dec_hi")

    def test_init_swt(self, setup_inputs):
        _, _, device = setup_inputs
        loss_fn = WaveletLoss(wavelet="db4", level=3, transform_type="swt", device=device)

        assert loss_fn.level == 3
        assert loss_fn.wavelet == "db4"
        assert loss_fn.transform_type == "swt"
        assert isinstance(loss_fn.transform, StationaryWaveletTransform)
        assert hasattr(loss_fn, "dec_lo")
        assert hasattr(loss_fn, "dec_hi")

    def test_init_qwt(self, setup_inputs):
        _, _, device = setup_inputs
        loss_fn = WaveletLoss(wavelet="db4", level=3, transform_type="qwt", device=device)

        assert loss_fn.level == 3
        assert loss_fn.wavelet == "db4"
        assert loss_fn.transform_type == "qwt"
        assert isinstance(loss_fn.transform, QuaternionWaveletTransform)
        assert hasattr(loss_fn, "dec_lo")
        assert hasattr(loss_fn, "dec_hi")
        assert hasattr(loss_fn, "hilbert_x")
        assert hasattr(loss_fn, "hilbert_y")
        assert hasattr(loss_fn, "hilbert_xy")

    def test_forward_dwt(self, setup_inputs):
        pred, target, device = setup_inputs
        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="dwt", device=device)

        # Test forward pass
        losses, details = loss_fn(pred, target)

        for loss in losses:
            # Check loss is a tensor of the right shape
            assert isinstance(loss, Tensor)
            assert loss.dim() == 4

        # Check details contains expected keys
        assert "combined_hf_pred" in details
        assert "combined_hf_target" in details

        # For identical inputs, loss should be small but not zero due to numerical precision
        same_losses, _ = loss_fn(target, target)
        for same_loss in same_losses:
            for item in same_loss:
                assert item.mean().item() < 1e-5

    def test_forward_swt(self, setup_inputs):
        pred, target, device = setup_inputs
        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="swt", device=device)

        # Test forward pass
        losses, details = loss_fn(pred, target)

        for loss in losses:
            # Check loss is a tensor of the right shape
            assert isinstance(loss, Tensor)
            assert loss.dim() == 4

        # Check details contains expected keys
        assert "combined_hf_pred" in details
        assert "combined_hf_target" in details

        # For identical inputs, loss should be small
        same_losses, _ = loss_fn(target, target)
        for same_loss in same_losses:
            for item in same_loss:
                assert item.mean().item() < 1e-5

    def test_forward_qwt(self, setup_inputs):
        pred, target, device = setup_inputs
        loss_fn = WaveletLoss(
            wavelet="db4",
            level=2,
            transform_type="qwt",
            device=device,
            quaternion_component_weights={"r": 1.0, "i": 0.5, "j": 0.5, "k": 0.2},
        )

        # Test forward pass
        losses, component_losses = loss_fn(pred, target)

        for loss in losses:
            # Check loss is a tensor of the right shape
            assert isinstance(loss, Tensor)
            assert loss.dim() == 4

        # Check component losses contain expected keys
        for level in range(2):
            for component in ["r", "i", "j", "k"]:
                for band in ["ll", "lh", "hl", "hh"]:
                    assert f"{component}_{band}_{level+1}" in component_losses

        # For identical inputs, loss should be small
        same_losses, _ = loss_fn(target, target)
        for same_loss in same_losses:
            for item in same_loss:
                assert item.mean().item() < 1e-5

    def test_custom_band_weights(self, setup_inputs):
        pred, target, device = setup_inputs

        # Define custom weights
        band_weights = {"ll": 0.5, "lh": 0.2, "hl": 0.2, "hh": 0.1}

        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="dwt", device=device, band_weights=band_weights)

        # Check weights are correctly set
        assert loss_fn.band_weights == band_weights

        # Test forward pass
        losses, _ = loss_fn(pred, target)

        for loss in losses:
            # Check loss is a tensor of the right shape
            assert isinstance(loss, Tensor)
            assert loss.dim() == 4

    def test_custom_band_level_weights(self, setup_inputs):
        pred, target, device = setup_inputs

        # Define custom level-specific weights
        band_level_weights = {"ll1": 0.3, "lh1": 0.1, "hl1": 0.1, "hh1": 0.1, "ll2": 0.2, "lh2": 0.05, "hl2": 0.05, "hh2": 0.1}

        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="dwt", device=device, band_level_weights=band_level_weights)

        # Check weights are correctly set
        assert loss_fn.band_level_weights == band_level_weights

        # Test forward pass
        losses, _ = loss_fn(pred, target)

        for loss in losses:
            # Check loss is a tensor of the right shape
            assert isinstance(loss, Tensor)
            assert loss.dim() == 4

    def test_ll_level_threshold(self, setup_inputs):
        pred, target, device = setup_inputs

        # Test with different ll_level_threshold values
        loss_fn1 = WaveletLoss(wavelet="db4", level=3, transform_type="dwt", device=device, ll_level_threshold=1)
        loss_fn2 = WaveletLoss(wavelet="db4", level=3, transform_type="dwt", device=device, ll_level_threshold=2)
        loss_fn3 = WaveletLoss(wavelet="db4", level=3, transform_type="dwt", device=device, ll_level_threshold=3)
        loss_fn4 = WaveletLoss(wavelet="db4", level=3, transform_type="dwt", device=device, ll_level_threshold=-1)

        losses1, _ = loss_fn1(pred, target)
        losses2, _ = loss_fn2(pred, target)
        losses3, _ = loss_fn3(pred, target)
        losses4, _ = loss_fn4(pred, target)

        # Loss with more ll levels should be different
        assert losses1[1].mean().item() != losses2[1].mean().item()

        for item1, item2, item3 in zip(losses1[2:], losses2[2:], losses3[2:]):
            # Loss with more ll levels should be different
            assert item3.mean().item() != item2.mean().item()
            assert item1.mean().item() != item3.mean().item()

        # ll threshold of -1 should be the same as 2 (3 - 1 == 2)
        assert losses2[2].mean().item() == losses4[2].mean().item()

    def test_set_loss_fn(self, setup_inputs):
        pred, target, device = setup_inputs

        # Initialize with MSE loss
        loss_fn = WaveletLoss(wavelet="db4", level=2, transform_type="dwt", device=device)
        assert loss_fn.loss_fn == F.mse_loss

        # Change to L1 loss
        loss_fn.set_loss_fn(F.l1_loss)
        assert loss_fn.loss_fn == F.l1_loss

        # Test with new loss function
        losses, _ = loss_fn(pred, target)
        for loss in losses:
            # Check loss is a tensor of the right shape
            assert isinstance(loss, Tensor)
            assert loss.dim() == 4
