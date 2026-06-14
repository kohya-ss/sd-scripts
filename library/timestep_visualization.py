"""Visualize the actual sampled-timestep / loss-weighting distribution for Flow Matching training.

This module is a *leaf* utility: it only knows how to render a distribution given
sampling and weighting callables. Model-specific assembly (which scheduler to build,
which ``get_noisy_model_input_and_timesteps`` to call) lives in the per-model
``*_train_utils`` modules so this file has no model dependencies.

Typical use from a training script::

    if args.show_timesteps:
        flux_train_utils.show_timesteps(args)  # builds the callables and calls show_timestep_distribution
    else:
        trainer.train(args)
"""

import logging
from typing import Callable, Optional

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONSOLE_WIDTH = 64
N_TIMESTEPS_PER_LINE = 25


def show_timestep_distribution(
    mode: str,
    sample_timesteps: Callable[[int], torch.Tensor],
    compute_weighting: Optional[Callable[[torch.Tensor], Optional[torch.Tensor]]] = None,
    num_train_timesteps: int = 1000,
    n_try: int = 100000,
    batch_size: int = 64,
    header: Optional[str] = None,
):
    """Sample timesteps many times and render their distribution (and loss weighting).

    Args:
        mode: ``"console"`` for an ASCII histogram, ``"image"`` for a matplotlib plot.
        sample_timesteps: ``fn(batch_size) -> Tensor`` returning sampled timesteps in
            ``[0, num_train_timesteps]`` (any shape; it is flattened).
        compute_weighting: ``fn(timesteps) -> Tensor | None`` returning the loss
            weighting for a 1-D tensor of integer timesteps ``[1 .. num_train_timesteps]``.
            ``None`` (or an all-``None`` return) is treated as uniform weighting of 1.0.
        num_train_timesteps: scheduler resolution (number of discrete timesteps).
        n_try: total number of timesteps to sample.
        batch_size: per-iteration sample count.
        header: optional multi-line text printed before the histogram (e.g. the args used).
    """
    if header:
        print(header)

    # --- sample the timestep distribution ---
    sampled_timesteps = np.zeros(num_train_timesteps, dtype=np.float64)
    for _ in tqdm(range(max(1, n_try // batch_size)), desc="sampling timesteps"):
        timesteps = sample_timesteps(batch_size).detach().flatten().float().cpu().numpy()
        indices = np.clip(timesteps.astype(np.int64), 0, num_train_timesteps - 1)
        np.add.at(sampled_timesteps, indices, 1.0)

    # --- compute the loss weighting per timestep ---
    sampled_weighting = np.ones(num_train_timesteps, dtype=np.float64)
    if compute_weighting is not None:
        all_timesteps = torch.arange(1, num_train_timesteps + 1, dtype=torch.float32)
        weighting = compute_weighting(all_timesteps)
        if weighting is not None:
            weighting = weighting.detach().float()
            # guard against inf / nan from schemes like sigma_sqrt at sigma==0
            weighting = torch.nan_to_num(weighting, nan=1.0, posinf=1.0, neginf=1.0)
            sampled_weighting = weighting.cpu().numpy().reshape(-1)

    if mode == "image":
        _show_image(sampled_timesteps, sampled_weighting)
    else:
        _show_console(sampled_timesteps, sampled_weighting, num_train_timesteps)


def _show_console(sampled_timesteps: np.ndarray, sampled_weighting: np.ndarray, num_train_timesteps: int):
    # average per line so the histogram fits the console
    n_lines = num_train_timesteps // N_TIMESTEPS_PER_LINE
    usable = n_lines * N_TIMESTEPS_PER_LINE
    ts = sampled_timesteps[:usable].reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)
    wt = sampled_weighting[:usable].reshape(-1, N_TIMESTEPS_PER_LINE).mean(axis=1)

    # display high -> low timestep (noisiest first), since t=num_train_timesteps is pure noise and t=0 is clean
    max_count = max(ts.max(), 1e-8)
    print(f"Sampled timesteps (top=noisy {num_train_timesteps}, bottom=clean 0): max count={max_count:.1f}")
    for i in reversed(range(len(ts))):
        line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: "
        line += "#" * int(ts[i] / max_count * CONSOLE_WIDTH)
        print(line)

    max_weighting = max(wt.max(), 1e-8)
    print(f"Sampled loss weighting (top=noisy {num_train_timesteps}, bottom=clean 0): max weighting={max_weighting:.2f}")
    for i in reversed(range(len(wt))):
        line = f"{i * N_TIMESTEPS_PER_LINE:4d}-{(i + 1) * N_TIMESTEPS_PER_LINE - 1:4d}: {wt[i]:8.2f} "
        line += "#" * int(wt[i] / max_weighting * CONSOLE_WIDTH)
        print(line)


def _show_image(sampled_timesteps: np.ndarray, sampled_weighting: np.ndarray):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is not installed; falling back to console output. / matplotlibが見つかりません。コンソール出力にフォールバックします。")
        _show_console(sampled_timesteps, sampled_weighting, len(sampled_timesteps))
        return

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(sampled_timesteps)), sampled_timesteps, width=1.0)
    plt.title("Sampled timesteps")
    plt.xlabel("Timestep (left=noisy, right=clean)")
    plt.ylabel("Count")
    plt.gca().invert_xaxis()  # noisiest timestep on the left

    plt.subplot(1, 2, 2)
    plt.bar(range(len(sampled_weighting)), sampled_weighting, width=1.0)
    plt.title("Sampled loss weighting")
    plt.xlabel("Timestep (left=noisy, right=clean)")
    plt.ylabel("Weighting")
    plt.gca().invert_xaxis()  # noisiest timestep on the left

    plt.tight_layout()
    plt.show()
