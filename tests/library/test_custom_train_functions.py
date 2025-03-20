import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

# Import the functions we're testing
from library.custom_train_functions import (
    apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
    get_snr_scale,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
)


@pytest.fixture
def loss():
    return torch.ones(2, 1)


@pytest.fixture
def timesteps():
    return torch.tensor([[200, 200]], dtype=torch.int32)


@pytest.fixture
def noise_scheduler():
    scheduler = MagicMock()
    scheduler.get_snr_for_timestep = MagicMock(return_value=torch.tensor([10.0, 5.0]))
    scheduler.all_snr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])
    return scheduler


# Tests for apply_snr_weight
def test_apply_snr_weight_with_get_snr_method(loss, timesteps, noise_scheduler):
    image_size = 64
    gamma = 5.0

    result = apply_snr_weight(
        loss,
        timesteps,
        noise_scheduler,
        gamma,
        v_prediction=False,
        image_size=image_size,
    )

    expected_result = torch.tensor([[0.5, 1.0]])

    assert torch.allclose(result, expected_result, rtol=1e-4, atol=1e-4)


def test_apply_snr_weight_with_all_snr(loss, timesteps):
    gamma = 5.0

    # Modify the mock to not use get_snr_for_timestep
    mock_noise_scheduler_no_method = MagicMock()
    mock_noise_scheduler_no_method.get_snr_for_timestep = None
    mock_noise_scheduler_no_method.all_snr = torch.tensor([0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0])

    result = apply_snr_weight(loss, timesteps, mock_noise_scheduler_no_method, gamma, v_prediction=False)

    expected_result = torch.tensor([1.0, 1.0])
    assert torch.allclose(result, expected_result, rtol=1e-4, atol=1e-4)


def test_apply_snr_weight_with_v_prediction(loss, timesteps, noise_scheduler):
    gamma = 5.0

    result = apply_snr_weight(loss, timesteps, noise_scheduler, gamma, v_prediction=True)

    expected_result = torch.tensor([[0.4545, 0.8333], [0.4545, 0.8333]])

    assert torch.allclose(result, expected_result, rtol=1e-4, atol=1e-4)


# Tests for scale_v_prediction_loss_like_noise_prediction
def test_scale_v_prediction_loss(loss, timesteps, noise_scheduler):
    with patch("library.custom_train_functions.get_snr_scale") as mock_get_snr_scale:
        mock_get_snr_scale.return_value = torch.tensor([0.9, 0.8])

        result = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)

        mock_get_snr_scale.assert_called_once_with(timesteps, noise_scheduler, None)

        # Apply broadcasting for multiplication
        scale = torch.tensor([[0.9, 0.8], [0.9, 0.8]])
        expected_result = loss * scale
        assert torch.allclose(result, expected_result)


# Tests for get_snr_scale
def test_get_snr_scale_with_get_snr_method(timesteps, noise_scheduler):
    image_size = 64

    result = get_snr_scale(timesteps, noise_scheduler, image_size)

    # Verify the method was called correctly
    noise_scheduler.get_snr_for_timestep.assert_called_once_with(timesteps, image_size)

    # Calculate expected values (snr / (snr + 1))
    snr = torch.tensor([10.0, 5.0])
    expected_scale = snr / (snr + 1)

    assert torch.allclose(result, expected_scale)


def test_get_snr_scale_with_all_snr(timesteps):
    # Create a scheduler that only has all_snr
    mock_scheduler_all_snr = MagicMock()
    mock_scheduler_all_snr.get_snr_for_timestep = None
    mock_scheduler_all_snr.all_snr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])

    result = get_snr_scale(timesteps, mock_scheduler_all_snr)

    expected_scale = torch.tensor([[0.9524, 0.9524]])

    assert torch.allclose(result, expected_scale, rtol=1e-4, atol=1e-4)


def test_get_snr_scale_with_large_snr(timesteps, noise_scheduler):
    # Set up the mock with a very large SNR value
    noise_scheduler.get_snr_for_timestep.return_value = torch.tensor([2000.0, 5.0])

    result = get_snr_scale(timesteps, noise_scheduler)

    expected_scale = torch.tensor([0.9990, 0.8333])

    assert torch.allclose(result, expected_scale, rtol=1e-4, atol=1e-4)


# Tests for add_v_prediction_like_loss
def test_add_v_prediction_like_loss(loss, timesteps, noise_scheduler):
    v_pred_like_loss = torch.tensor([0.3, 0.2]).view(2, 1)

    with patch("library.custom_train_functions.get_snr_scale") as mock_get_snr_scale:
        mock_get_snr_scale.return_value = torch.tensor([0.9, 0.8])

        result = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, v_pred_like_loss)

        mock_get_snr_scale.assert_called_once_with(timesteps, noise_scheduler, None)

        expected_result = torch.tensor([[1.3333, 1.3750], [1.2222, 1.2500]])
        assert torch.allclose(result, expected_result, rtol=1e-4, atol=1e-4)


# Tests for apply_debiased_estimation
def test_apply_debiased_estimation_no_snr(loss, timesteps):
    # Create a scheduler without SNR methods
    scheduler_without_snr = MagicMock()
    # Need to explicitly set attribute to None instead of deleting
    scheduler_without_snr.get_snr_for_timestep = None

    result = apply_debiased_estimation(loss, timesteps, scheduler_without_snr)

    # When no SNR methods are available, the function should return the loss unchanged
    assert torch.equal(result, loss)


def test_apply_debiased_estimation_with_get_snr_method(loss, timesteps, noise_scheduler):
    # Test with v_prediction=False
    result_no_v = apply_debiased_estimation(loss, timesteps, noise_scheduler, v_prediction=False)

    expected_result_no_v = torch.tensor([[0.3162, 0.4472], [0.3162, 0.4472]])

    assert torch.allclose(result_no_v, expected_result_no_v, rtol=1e-4, atol=1e-4)

    # Test with v_prediction=True
    result_v = apply_debiased_estimation(loss, timesteps, noise_scheduler, v_prediction=True)

    expected_result_v = torch.tensor([[0.0909, 0.1667], [0.0909, 0.1667]])

    assert torch.allclose(result_v, expected_result_v, rtol=1e-4, atol=1e-4)


def test_apply_debiased_estimation_with_all_snr(loss, timesteps):
    # Create a scheduler that only has all_snr
    mock_scheduler_all_snr = MagicMock()
    mock_scheduler_all_snr.get_snr_for_timestep = None
    mock_scheduler_all_snr.all_snr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0])

    result = apply_debiased_estimation(loss, timesteps, mock_scheduler_all_snr, v_prediction=False)

    expected_result = torch.tensor([[1.0, 1.0]])

    assert torch.allclose(result, expected_result, rtol=1e-4, atol=1e-4)


def test_apply_debiased_estimation_with_large_snr(loss, timesteps, noise_scheduler):
    # Set up the mock with a very large SNR value
    noise_scheduler.get_snr_for_timestep.return_value = torch.tensor([2000.0, 5.0])

    result = apply_debiased_estimation(loss, timesteps, noise_scheduler, v_prediction=False)

    expected_result = torch.tensor([[0.0316, 0.4472], [0.0316, 0.4472]])

    assert torch.allclose(result, expected_result, rtol=1e-4, atol=1e-4)


# Additional edge cases
def test_empty_tensors(noise_scheduler):
    # Test with empty tensors
    loss = torch.tensor([], dtype=torch.float32)
    timesteps = torch.tensor([], dtype=torch.int32)

    assert isinstance(timesteps, torch.IntTensor)

    noise_scheduler.get_snr_for_timestep.return_value = torch.tensor([], dtype=torch.float32)

    result = apply_snr_weight(loss, timesteps, noise_scheduler, gamma=5.0)

    assert result.shape == loss.shape
    assert result.dtype == loss.dtype


def test_different_device_compatibility(loss, timesteps, noise_scheduler):
    gamma = 5.0
    device = torch.device("cpu")

    # For a real device test, we need to create actual tensors on devices
    loss_on_device = loss.to(device)

    # Mock the SNR tensor that would be returned with proper device handling
    snr_tensor = torch.tensor([0.204, 0.294], device=device)
    noise_scheduler.get_snr_for_timestep.return_value = snr_tensor

    result = apply_snr_weight(loss_on_device, timesteps, noise_scheduler, gamma)
