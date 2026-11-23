# Merge LoRA checkpoints of one training run with the Post-Hoc EMA method (ported from Musubi Tuner)
# 1. Sort the files for the specified path by modification time
# 2. Load the oldest file and initialize weights
# 3. Iterate through the remaining files, loading and merging their weights with decay rate beta
# 4. Save the final merged weights to a new file. The metadata is updated to reflect the new file

import argparse
import os
from typing import Optional

import numpy as np
import torch
from safetensors.torch import save_file

import library.model_io as model_io
from library.safetensors_utils import MemoryEfficientSafeOpen
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def sigma_rel_to_gamma(sigma_rel: float) -> float:
    """Implementation of Algorithm 2 from the paper: https://arxiv.org/pdf/2312.02696"""
    # solve the cubic equation γ^3 + 7γ^2 + (16 - 1/σ_rel^2)γ + (12 - 1/σ_rel^2) = 0
    t = sigma_rel**-2
    # coefficients [1, 7, 16-t, 12-t]
    coeffs = [1, 7, 16 - t, 12 - t]
    # positive real root is γ
    roots = np.roots(coeffs)
    gamma = roots[np.isreal(roots) & (roots.real >= 0)].real.max()
    return gamma


def merge_lora_weights_with_post_hoc_ema(
    path: list[str], no_sort: bool, beta1: float, beta2: float, sigma_rel: Optional[float], output_file: str
):
    # Sort the files by modification time
    if not no_sort:
        logger.info("Sorting files by modification time...")
        path.sort(key=lambda x: os.path.getmtime(x))

    # Load metadata from the last file
    logger.info(f"Loading metadata from {path[-1]}")
    with MemoryEfficientSafeOpen(path[-1]) as f:
        metadata = f.metadata()
    if not metadata:
        metadata = None
        logger.info("No metadata found in the last file, proceeding without metadata.")
    else:
        logger.info("Metadata found, using metadata from the last file.")

    # Load the oldest file and initialize weights
    logger.info(f"Loading weights from {path[0]}")
    with MemoryEfficientSafeOpen(path[0]) as f:
        original_dtypes = {}
        state_dict = {}
        for key in f.keys():
            value: torch.Tensor = f.get_tensor(key)

            if value.dtype.is_floating_point:
                original_dtypes[key] = value.dtype
                value = value.to(torch.float32)  # Convert to float32 for merging
            else:
                logger.info(f"Skipping non-floating point tensor: {key}")

            state_dict[key] = value

    # Iterate through the remaining files, loading and merging their weights with decay rate beta
    ema_count = len(path) - 1
    if sigma_rel is not None:
        gamma = sigma_rel_to_gamma(sigma_rel)
    else:
        gamma = None

    for i, file in enumerate(path[1:]):
        if sigma_rel is not None:
            # Calculate beta using Power Function EMA
            t = i + 1
            beta = (1 - 1 / t) ** (gamma + 1)
        else:
            beta = beta1 + (beta2 - beta1) * (i / (ema_count - 1)) if ema_count > 1 else beta1

        logger.info(f"Loading weights from {file} for merging with beta={beta:.4f}")
        with MemoryEfficientSafeOpen(file) as f:
            for key in f.keys():
                value = f.get_tensor(key)
                if key.endswith(".alpha"):
                    # compare alpha tensors and raise an error if they differ
                    if key not in state_dict or torch.allclose(state_dict[key], value.to(torch.float32)):
                        # If alpha tensors match, skip merging
                        continue
                    else:
                        raise ValueError(f"Alpha tensors for key {key} do not match across files.")

                if not value.dtype.is_floating_point:
                    # Skip non-floating point tensors
                    logger.info(f"Skipping non-floating point tensor: {key}")
                    continue

                if key in state_dict:
                    # Merge the weights with decay rate beta
                    value = value.to(torch.float32)
                    state_dict[key] = state_dict[key] * beta + value * (1 - beta)
                else:
                    raise KeyError(f"Key {key} not found in the initial state_dict.")

    # Convert the merged weights back to their original dtypes
    for key in state_dict:
        if key in original_dtypes:
            state_dict[key] = state_dict[key].to(original_dtypes[key])

    # update metadata with new hash
    if metadata is not None:
        logger.info("Updating metadata with new hashes.")
        model_hash, legacy_hash = model_io.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    # Save the final merged weights to a new file
    logger.info(f"Saving merged weights to {output_file}")
    save_file(state_dict, output_file, metadata=metadata)
    logger.info("Merging completed successfully.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge LoRA weights with Post-Hoc EMA method / LoRAの重みをPost-Hoc EMAでマージする"
    )
    parser.add_argument(
        "path",
        nargs="+",
        help="List of paths to the LoRA weight files (checkpoints of one training run) / LoRA重みファイルのパス（同じ学習のチェックポイント）",
    )
    parser.add_argument(
        "--no_sort",
        action="store_true",
        help="Do not sort the files by modification time (merge in the specified order) / ファイルを更新時刻でソートしない（指定順にマージする）",
    )
    parser.add_argument("--beta", type=float, default=0.95, help="Decay rate for merging weights / マージの減衰率")
    parser.add_argument(
        "--beta2",
        type=float,
        default=None,
        help="Decay rate at the last file; the decay rate is linearly interpolated from beta to beta2"
        + " / 最後のファイルでの減衰率。減衰率をbetaからbeta2へ線形補間する",
    )
    parser.add_argument(
        "--sigma_rel",
        type=float,
        default=None,
        help="Relative sigma for Power Function EMA. If specified, beta and beta2 are ignored"
        + " / Power Function EMAの相対シグマ。指定するとbetaとbeta2は無視される",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output file path for merged weights / マージした重みの出力先ファイル"
    )
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()

    beta2 = args.beta if args.beta2 is None else args.beta2
    merge_lora_weights_with_post_hoc_ema(args.path, args.no_sort, args.beta, beta2, args.sigma_rel, args.output_file)


if __name__ == "__main__":
    main()
