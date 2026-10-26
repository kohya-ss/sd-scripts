import pytest
import torch
from unittest.mock import MagicMock, patch
from library.flux_train_utils import (
    get_noisy_model_input_and_timesteps,
    get_show_timesteps_offset,
    sample_truncated_logit_normal_indices,
)
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

# Mock classes and functions
class MockNoiseScheduler:
    def __init__(self, num_train_timesteps=1000):
        self.config = MagicMock()
        self.config.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.arange(num_train_timesteps, dtype=torch.long)


# Create fixtures for commonly used objects
@pytest.fixture
def args():
    args = MagicMock()
    args.timestep_sampling = "uniform"
    args.weighting_scheme = "uniform"
    args.logit_mean = 0.0
    args.logit_std = 1.0
    args.mode_scale = 1.0
    args.sigmoid_scale = 1.0
    args.discrete_flow_shift = 3.1582
    args.timestep_cutoff = 0.95
    args.timestep_uniform_ratio = 0.01
    args.ip_noise_gamma = None
    args.ip_noise_gamma_random_strength = False
    return args


@pytest.fixture
def noise_scheduler():
    return MockNoiseScheduler(num_train_timesteps=1000)


@pytest.fixture
def real_noise_scheduler():
    # The real (descending, shift=1.0) training grid: timesteps[k] = 1000 - k, sigma = timesteps / 1000
    return FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)


@pytest.fixture
def latents():
    return torch.randn(2, 4, 8, 8)


@pytest.fixture
def noise():
    return torch.randn(2, 4, 8, 8)


@pytest.fixture
def device():
    # return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"


# Mock the required functions
@pytest.fixture(autouse=True)
def mock_functions():
    with (
        patch("torch.sigmoid", side_effect=torch.sigmoid),
        patch("torch.rand", side_effect=torch.rand),
        patch("torch.randn", side_effect=torch.randn),
    ):
        yield


# Test different timestep sampling methods
def test_uniform_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "uniform"
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)
    assert noisy_input.dtype == dtype
    assert timesteps.dtype == dtype


def test_sigmoid_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "sigmoid"
    args.sigmoid_scale = 1.0
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_shift_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "shift"
    args.sigmoid_scale = 1.0
    args.discrete_flow_shift = 3.1582
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_flux_shift_sampling(args, noise_scheduler, latents, noise, device):
    args.timestep_sampling = "flux_shift"
    args.sigmoid_scale = 1.0
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_weighting_scheme(args, noise_scheduler, latents, noise, device):
    # Mock the necessary functions for this specific test
    with patch("library.flux_train_utils.compute_density_for_timestep_sampling", 
               return_value=torch.tensor([0.3, 0.7], device=device)), \
         patch("library.flux_train_utils.get_sigmas", 
               return_value=torch.tensor([[0.3], [0.7]], device=device).view(-1, 1, 1, 1)):
               
        args.timestep_sampling = "other"  # Will trigger the weighting scheme path
        args.weighting_scheme = "uniform"
        args.logit_mean = 0.0
        args.logit_std = 1.0
        args.mode_scale = 1.0
        dtype = torch.float32
        
        noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        
        assert noisy_input.shape == latents.shape
        assert timesteps.shape == (latents.shape[0],)
        assert sigmas.shape == (latents.shape[0], 1, 1, 1)


# Test IP noise options
def test_with_ip_noise(args, noise_scheduler, latents, noise, device):
    args.ip_noise_gamma = 0.5
    args.ip_noise_gamma_random_strength = False
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


def test_with_random_ip_noise(args, noise_scheduler, latents, noise, device):
    args.ip_noise_gamma = 0.1
    args.ip_noise_gamma_random_strength = True
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (latents.shape[0],)
    assert sigmas.shape == (latents.shape[0], 1, 1, 1)


# Test different data types
def test_float16_dtype(args, noise_scheduler, latents, noise, device):
    dtype = torch.float16

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.dtype == dtype
    assert timesteps.dtype == dtype


# Test different batch sizes
def test_different_batch_size(args, noise_scheduler, device):
    latents = torch.randn(5, 4, 8, 8)  # batch size of 5
    noise = torch.randn(5, 4, 8, 8)
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (5,)
    assert sigmas.shape == (5, 1, 1, 1)


# Test different image sizes
def test_different_image_size(args, noise_scheduler, device):
    latents = torch.randn(2, 4, 16, 16)  # larger image size
    noise = torch.randn(2, 4, 16, 16)
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (2,)
    assert sigmas.shape == (2, 1, 1, 1)


# Test edge cases
def test_zero_batch_size(args, noise_scheduler, device):
    with pytest.raises(AssertionError):  # expecting an error with zero batch size
        latents = torch.randn(0, 4, 8, 8)
        noise = torch.randn(0, 4, 8, 8)
        dtype = torch.float32

        get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)


def test_different_timestep_count(args, device):
    noise_scheduler = MockNoiseScheduler(num_train_timesteps=500)  # different timestep count
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn(2, 4, 8, 8)
    dtype = torch.float32

    noisy_input, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype)

    assert noisy_input.shape == latents.shape
    assert timesteps.shape == (2,)
    # Check that timesteps are within the proper range
    assert torch.all(timesteps < 500)


# Tests for timestep_sampling_offset
class TestTimestepSamplingOffset:
    """Regression tests for per-subset timestep sampling offset."""

    @pytest.mark.parametrize("mode", ["sigmoid", "shift", "flux_shift"])
    def test_none_offset_matches_baseline(self, args, noise_scheduler, device, mode):
        """offset=None should produce identical results to no offset."""
        args.timestep_sampling = mode
        args.sigmoid_scale = 1.0
        bsz = 64
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        torch.manual_seed(42)
        _, ts_none, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype, timestep_sampling_offset=None
        )
        assert torch.allclose(ts_baseline, ts_none)

    @pytest.mark.parametrize("mode", ["sigmoid", "shift", "flux_shift"])
    def test_zero_offset_matches_baseline(self, args, noise_scheduler, device, mode):
        """offset=0.0 should produce identical results to no offset."""
        args.timestep_sampling = mode
        args.sigmoid_scale = 1.0
        bsz = 64
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        torch.manual_seed(42)
        _, ts_zero, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype,
            timestep_sampling_offset=torch.zeros(bsz)
        )
        assert torch.allclose(ts_baseline, ts_zero)

    @pytest.mark.parametrize("mode", ["sigmoid", "shift", "flux_shift"])
    def test_positive_offset_increases_mean_timestep(self, args, noise_scheduler, device, mode):
        """Positive offset should shift timestep distribution upward (higher noise)."""
        args.timestep_sampling = mode
        args.sigmoid_scale = 1.0
        bsz = 256
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        torch.manual_seed(42)
        _, ts_offset, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype,
            timestep_sampling_offset=torch.full((bsz,), 1.0)
        )
        assert ts_offset.mean() > ts_baseline.mean()

    @pytest.mark.parametrize("mode, weighting_scheme", [("truncated_logit_normal", "uniform"), ("sigma", "logit_normal")])
    def test_none_and_zero_offset_match_baseline_on_grid(self, args, real_noise_scheduler, device, mode, weighting_scheme):
        args.timestep_sampling = mode
        args.weighting_scheme = weighting_scheme
        bsz = 64
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(args, real_noise_scheduler, latents, noise, device, dtype)
        torch.manual_seed(42)
        _, ts_none, _ = get_noisy_model_input_and_timesteps(
            args, real_noise_scheduler, latents, noise, device, dtype, timestep_sampling_offset=None
        )
        torch.manual_seed(42)
        _, ts_zero, _ = get_noisy_model_input_and_timesteps(
            args, real_noise_scheduler, latents, noise, device, dtype, timestep_sampling_offset=torch.zeros(bsz)
        )
        assert torch.allclose(ts_baseline, ts_none)
        assert torch.allclose(ts_baseline, ts_zero)

    @pytest.mark.parametrize("mode, weighting_scheme", [("truncated_logit_normal", "uniform"), ("sigma", "logit_normal")])
    def test_positive_offset_increases_mean_timestep_on_grid(self, args, real_noise_scheduler, device, mode, weighting_scheme):
        """Grid-based modes: a positive offset must also mean higher noise (the normal sample is the logit of 1 - sigma)."""
        args.timestep_sampling = mode
        args.weighting_scheme = weighting_scheme
        args.logit_mean = -1.09
        args.logit_std = 0.95
        bsz = 1024
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(args, real_noise_scheduler, latents, noise, device, dtype)
        torch.manual_seed(42)
        _, ts_offset, _ = get_noisy_model_input_and_timesteps(
            args, real_noise_scheduler, latents, noise, device, dtype, timestep_sampling_offset=torch.full((bsz,), 1.0)
        )
        assert ts_offset.mean() > ts_baseline.mean() + 50

    def test_offset_ignored_for_sigma_without_logit_normal(self, args, real_noise_scheduler, device):
        """sigma with a weighting scheme that draws no normal sample ignores the offset."""
        args.timestep_sampling = "sigma"
        args.weighting_scheme = "uniform"
        bsz = 64
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(args, real_noise_scheduler, latents, noise, device, dtype)
        torch.manual_seed(42)
        _, ts_offset, _ = get_noisy_model_input_and_timesteps(
            args, real_noise_scheduler, latents, noise, device, dtype, timestep_sampling_offset=torch.full((bsz,), 1.0)
        )
        assert torch.allclose(ts_baseline, ts_offset)

    def test_offset_does_not_affect_uniform(self, args, noise_scheduler, device):
        """Uniform sampling ignores offset (offset code path is not reached)."""
        args.timestep_sampling = "uniform"
        bsz = 64
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        torch.manual_seed(42)
        _, ts_offset, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype,
            timestep_sampling_offset=torch.full((bsz,), 1.0)
        )
        assert torch.allclose(ts_baseline, ts_offset)

    def test_per_sample_offset_broadcasting(self, args, noise_scheduler, device):
        """Different offsets per sample should produce different shifts."""
        args.timestep_sampling = "sigmoid"
        args.sigmoid_scale = 1.0
        bsz = 4
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        dtype = torch.float32
        offset = torch.tensor([-1.0, 0.0, 0.0, 1.0])

        torch.manual_seed(42)
        _, ts_baseline, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype
        )
        torch.manual_seed(42)
        _, ts_offset, _ = get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, dtype,
            timestep_sampling_offset=offset
        )
        # Samples with offset=0 should match baseline
        assert torch.allclose(ts_offset[1], ts_baseline[1])
        assert torch.allclose(ts_offset[2], ts_baseline[2])
        # Negative offset → lower timestep, positive → higher
        assert ts_offset[0] < ts_baseline[0]
        assert ts_offset[3] > ts_baseline[3]


class TestGetShowTimestepsOffset:
    """Tests for the --show_timesteps_offset resolution helper."""

    @pytest.mark.parametrize("mode", ["sigmoid", "shift", "flux_shift", "truncated_logit_normal"])
    def test_offset_applied_for_supported_modes(self, args, mode):
        args.timestep_sampling = mode
        args.show_timesteps_offset = -0.5
        offset, note = get_show_timesteps_offset(args)
        assert offset == -0.5
        assert "IGNORED" not in note
        assert "-0.5" in note

    def test_offset_applied_for_sigma_with_logit_normal(self, args):
        args.timestep_sampling = "sigma"
        args.weighting_scheme = "logit_normal"
        args.show_timesteps_offset = -0.5
        offset, note = get_show_timesteps_offset(args)
        assert offset == -0.5
        assert "IGNORED" not in note

    @pytest.mark.parametrize("mode", ["uniform", "sigma"])
    def test_offset_ignored_for_unsupported_modes(self, args, mode):
        # the args fixture uses weighting_scheme="uniform", so sigma draws no normal sample here
        args.timestep_sampling = mode
        args.show_timesteps_offset = -0.5
        offset, note = get_show_timesteps_offset(args)
        assert offset is None
        assert "IGNORED" in note

    def test_zero_offset_returns_none(self, args):
        args.timestep_sampling = "shift"
        args.show_timesteps_offset = 0.0
        offset, note = get_show_timesteps_offset(args)
        assert offset is None
        assert note == ""


class TestTruncatedLogitNormal:
    """Tests for --timestep_sampling truncated_logit_normal (the MOSS-Image pre-training schedule)."""

    T = 1000

    def _sigmas(self, indices):
        # descending grid with shift=1.0: timesteps[k] = T - k
        return (self.T - indices).double() / self.T

    def test_indices_cover_the_grid(self):
        torch.manual_seed(0)
        idx = sample_truncated_logit_normal_indices(20000, self.T, -1.09, 0.95, 0.95, 0.01)
        assert idx.dtype == torch.long and idx.device.type == "cpu"
        assert idx.min() >= 0 and idx.max() <= self.T - 1
        # the rescaling covers the full range: both ends of the grid are reachable
        assert idx.min() == 0 and idx.max() == self.T - 1

    def test_matches_sigma_logit_normal_without_truncation(self):
        """cutoff=1.0 / ratio=0.0 is the plain logit-normal of `sigma` + `logit_normal` (floor(u * T) on the grid)."""
        n = 50000
        torch.manual_seed(0)
        idx = sample_truncated_logit_normal_indices(n, self.T, -1.09, 0.95, 1.0, 0.0)
        torch.manual_seed(1)
        u = torch.sigmoid(torch.normal(mean=-1.09, std=0.95, size=(n,)))
        idx_ref = (u * self.T).long()
        assert abs(idx.double().mean() - idx_ref.double().mean()) < 10  # std of the index is ~190, SE of the mean ~1
        assert abs(self._sigmas(idx).median() - self._sigmas(idx_ref).median()) < 0.02

    def test_reference_statistics(self):
        """The official schedule (mu -1.09, s 0.95, cutoff 0.95, 1% uniform) puts about 11% of the samples below sigma 0.5."""
        torch.manual_seed(0)
        idx = sample_truncated_logit_normal_indices(100000, self.T, -1.09, 0.95, 0.95, 0.01)
        sigmas = self._sigmas(idx)
        assert 0.10 < (sigmas < 0.5).double().mean() < 0.12
        assert 0.73 < sigmas.mean() < 0.76

    def test_uniform_ratio_one_is_uniform(self):
        torch.manual_seed(0)
        idx = sample_truncated_logit_normal_indices(50000, self.T, -1.09, 0.95, 0.95, 1.0)
        assert abs(idx.double().mean() - (self.T - 1) / 2) < 10
        counts = torch.bincount(idx, minlength=self.T)
        assert counts.min() > 0

    def test_smaller_cutoff_means_higher_noise(self):
        torch.manual_seed(0)
        idx_a = sample_truncated_logit_normal_indices(20000, self.T, -1.09, 0.95, 1.0, 0.0)
        torch.manual_seed(0)
        idx_b = sample_truncated_logit_normal_indices(20000, self.T, -1.09, 0.95, 0.5, 0.0)
        assert self._sigmas(idx_b).mean() > self._sigmas(idx_a).mean() + 0.05
        # truncation + rescaling keeps the full range
        assert idx_b.min() == 0

    def test_logit_offset_shifts_the_mean(self):
        torch.manual_seed(0)
        idx = sample_truncated_logit_normal_indices(20000, self.T, -1.09, 0.95, 0.95, 0.0)
        torch.manual_seed(0)
        idx_off = sample_truncated_logit_normal_indices(20000, self.T, -1.09, 0.95, 0.95, 0.0, torch.full((20000,), -1.0))
        # logit_offset < 0 lowers Z, so sigma = sigmoid(-Z) grows
        assert self._sigmas(idx_off).mean() > self._sigmas(idx).mean() + 0.05
        torch.manual_seed(0)
        idx_same = sample_truncated_logit_normal_indices(20000, self.T, -2.09, 0.95, 0.95, 0.0)
        assert torch.equal(idx_off, idx_same)  # identical to shifting logit_mean

    @pytest.mark.parametrize("cutoff, ratio", [(0.0, 0.0), (1.5, 0.0), (0.95, -0.1), (0.95, 1.1)])
    def test_invalid_parameters(self, cutoff, ratio):
        with pytest.raises(ValueError):
            sample_truncated_logit_normal_indices(4, self.T, -1.09, 0.95, cutoff, ratio)

    def test_get_noisy_model_input_and_timesteps(self, args, real_noise_scheduler, device):
        args.timestep_sampling = "truncated_logit_normal"
        args.logit_mean = -1.09
        args.logit_std = 0.95
        args.discrete_flow_shift = 1.0
        bsz = 256
        latents = torch.randn(bsz, 4, 8, 8)
        noise = torch.randn(bsz, 4, 8, 8)
        torch.manual_seed(0)
        noisy, timesteps, sigmas = get_noisy_model_input_and_timesteps(args, real_noise_scheduler, latents, noise, device, torch.float32)
        assert noisy.shape == latents.shape and timesteps.shape == (bsz,) and sigmas.shape == (bsz, 1, 1, 1)
        # on the shift=1.0 grid the timesteps are the integers 1..1000 and sigma = t / 1000
        assert torch.all(timesteps >= 1) and torch.all(timesteps <= 1000)
        assert torch.allclose(timesteps, timesteps.round())
        assert torch.allclose(sigmas.flatten(), timesteps / 1000)
        expected = (1.0 - sigmas) * latents + sigmas * noise
        assert torch.allclose(noisy, expected, atol=1e-5)
