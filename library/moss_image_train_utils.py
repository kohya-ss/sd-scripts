# MOSS-Image training / sampling utilities
# The sampler follows the official `MOSSImageScheduler` (pipeline_moss_image.py): the shifted inference grid is snapped
# to the nearest sigma of the training grid (shift 1, 1000 steps) and integrated with Euler steps in float32.

from typing import Optional, Sequence

import torch

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

NUM_TRAIN_TIMESTEPS = 1000
DEFAULT_UNCOND_SKIP_LAYERS = (22,)  # block skipped in the negative branch of CFG by the official pipeline


def _sigma_grid(count: int, shift: float, total: int) -> torch.Tensor:
    lower = shift / (total + shift - 1)
    values = torch.linspace(1, lower, count, dtype=torch.float32)
    return shift * values / (1 + (shift - 1) * values)


def get_sampling_sigmas(
    num_inference_steps: int, shift: float = 5.0, training_shift: float = 1.0, num_train_timesteps: int = NUM_TRAIN_TIMESTEPS
) -> torch.Tensor:
    """Return the sampling sigmas [steps + 1] (descending, the last value is 0) on the CPU in float32.

    The number of effective steps can be smaller than `num_inference_steps` because the shifted grid is snapped to the
    training grid and duplicates are removed (same as the official scheduler).
    """
    if num_inference_steps < 1 or shift <= 0 or training_shift <= 0:
        raise ValueError("Steps and shifts must be positive.")
    total = num_train_timesteps
    sigmas = _sigma_grid(num_inference_steps, shift, total)
    training = _sigma_grid(total, training_shift, total).flip(0)  # ascending
    upper = torch.searchsorted(training, sigmas).clamp(max=total - 1)
    lower = (upper - 1).clamp(min=0)
    nearest = torch.where((sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper)
    sigmas = torch.unique_consecutive(training[nearest])
    return torch.cat([sigmas, sigmas.new_zeros(1)])


def euler_step(latents: torch.Tensor, model_pred: torch.Tensor, sigmas: torch.Tensor, step_index: int) -> torch.Tensor:
    """One Euler step in float32: x_{i+1} = x_i + (sigma_{i+1} - sigma_i) * v"""
    dt = (sigmas[step_index + 1] - sigmas[step_index]).to(torch.float32)
    return latents.float() + dt * model_pred.float()


def parse_skip_layers(value: Optional[str]) -> tuple[int, ...]:
    """Parse a comma separated list of block indices. None or an empty string disables skipping."""
    if value is None:
        return ()
    value = value.strip()
    if not value:
        return ()
    return tuple(int(v.strip()) for v in value.split(",") if v.strip())
