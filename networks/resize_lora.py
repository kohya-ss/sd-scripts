# Convert LoRA to different rank approximation (should only be used to go to lower rank)
# This code is based off the extract_lora_from_models.py file which is based on https://github.com/cloneofsimo/lora/blob/develop/lora_diffusion/cli_svd.py
# Thanks to cloneofsimo

import os
import argparse
import torch
from safetensors.torch import load_file, save_file, safe_open
from tqdm import tqdm
import numpy as np

import library.model_io as model_io
from library import model_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

MIN_SV = 1e-6

LORA_DOWN_UP_FORMATS = [
    ("lora_down", "lora_up"),  # sd-scripts LoRA
    ("lora_A", "lora_B"),  # PEFT LoRA
    ("down", "up"),  # ControlLoRA
]


# Model save and load functions


def load_state_dict(file_name, dtype):
    if model_util.is_safetensors(file_name):
        sd = load_file(file_name)
        with safe_open(file_name, framework="pt") as f:
            metadata = f.metadata()
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict, metadata):
    if model_util.is_safetensors(file_name):
        save_file(state_dict, file_name, metadata)
    else:
        torch.save(state_dict, file_name)


# Indexing functions


def index_sv_cumulative(S, target):
    original_sum = float(torch.sum(S))
    cumulative_sums = torch.cumsum(S, dim=0) / original_sum
    index = int(torch.searchsorted(cumulative_sums, target))
    index = max(0, min(index, len(S) - 1))

    return index


def index_sv_fro(S, target):
    S_squared = S.pow(2)
    S_fro_sq = float(torch.sum(S_squared))
    sum_S_squared = torch.cumsum(S_squared, dim=0) / S_fro_sq
    index = int(torch.searchsorted(sum_S_squared, target**2))
    index = max(0, min(index, len(S) - 1))

    return index


def index_sv_ratio(S, target):
    max_sv = S[0]
    min_sv = max_sv / target
    index = int(torch.sum(S > min_sv).item()) - 1
    index = max(0, min(index, len(S) - 1))

    return index


SVD_METHODS = ("merged", "qr")
DEFAULT_SVD_LOWRANK_NITER = 2


def svd_merged(lora_down, lora_up, lora_rank, device, svd_lowrank_niter=DEFAULT_SVD_LOWRANK_NITER):
    """SVD of the materialized product lora_up @ lora_down (the original method).

    For large matrices (both sides > 2048), torch.svd_lowrank is used unless svd_lowrank_niter is 0.
    """
    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    weight = lora_up @ lora_down  # (out, in)
    out_size, in_size = weight.size()
    del lora_up, lora_down

    if svd_lowrank_niter > 0 and out_size > 2048 and in_size > 2048:
        U, S, V = torch.svd_lowrank(weight, q=min(2 * lora_rank, out_size, in_size), niter=svd_lowrank_niter)
        Vh = V.T
    else:
        U, S, Vh = torch.linalg.svd(weight)
    return U, S, Vh


def svd_qr(lora_down, lora_up, lora_rank, device):
    """Exact SVD of lora_up @ lora_down without materializing the product.

    With lora_up = Q R and R @ lora_down = U S Vh, we get lora_up @ lora_down = (Q U) S Vh.

    If lora_rank is larger than the number of singular values, the result is padded up to lora_rank (capped by the
    input size) with zero singular values, zero columns of U and rows of Vh orthonormal to the existing ones. This keeps
    the requested rank like the merged method (whose extra singular values are near zero), and the padded lora_down
    stays non-zero so that the resized LoRA can be trained further.
    """
    lora_down = lora_down.to(device)
    lora_up = lora_up.to(device)

    Q, R = torch.linalg.qr(lora_up)  # Q: (out, k), R: (k, rank), k = min(out, rank)
    C = R @ lora_down  # C: (k, in)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)  # U: (k, m), S: (m,), Vh: (m, in), m = min(k, in)
    U = Q @ U

    num_pad = min(lora_rank, Vh.shape[1]) - len(S)
    if num_pad > 0:
        U = torch.cat([U, U.new_zeros(U.shape[0], num_pad)], dim=1)
        S = torch.cat([S, S.new_zeros(num_pad)])
        Vh = torch.cat([Vh, orthonormal_complement_rows(Vh, num_pad)], dim=0)
    return U, S, Vh


def orthonormal_complement_rows(Vh, num_rows, seed=0):
    """Return num_rows orthonormal rows orthogonal to the (orthonormal) rows of Vh. Deterministic for a given seed."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    G = torch.randn(num_rows, Vh.shape[1], generator=generator, dtype=torch.float64)
    Vh64 = Vh.to("cpu", torch.float64)
    for _ in range(2):  # project out twice for numerical stability
        G = G - (G @ Vh64.T) @ Vh64
    Qc, _ = torch.linalg.qr(G.T)  # (in, num_rows)
    return Qc.T.to(Vh.device, Vh.dtype)


def svd_lora(lora_down, lora_up, lora_rank, device, svd_method="merged", svd_lowrank_niter=DEFAULT_SVD_LOWRANK_NITER):
    if svd_method == "qr":
        return svd_qr(lora_down, lora_up, lora_rank, device)
    return svd_merged(lora_down, lora_up, lora_rank, device, svd_lowrank_niter)


# Modified from Kohaku-blueleaf's extract functions
def extract_conv(
    lora_down,
    lora_up,
    lora_rank,
    dynamic_method,
    dynamic_param,
    device,
    scale=1,
    svd_method="merged",
    svd_lowrank_niter=DEFAULT_SVD_LOWRANK_NITER,
):
    in_rank, in_size, kernel_size, k_ = lora_down.shape
    out_size, out_rank, _, _ = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"
    assert kernel_size == k_, f"kernel {kernel_size} {k_} mismatch"

    U, S, Vh = svd_lora(
        lora_down.reshape(in_rank, -1), lora_up.reshape(out_size, -1), lora_rank, device, svd_method, svd_lowrank_niter
    )

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank] * S[:lora_rank]
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size, kernel_size, kernel_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank, 1, 1).cpu()
    return param_dict


def extract_linear(
    lora_down,
    lora_up,
    lora_rank,
    dynamic_method,
    dynamic_param,
    device,
    scale=1,
    svd_method="merged",
    svd_lowrank_niter=DEFAULT_SVD_LOWRANK_NITER,
):
    in_rank, in_size = lora_down.shape
    out_size, out_rank = lora_up.shape
    assert in_rank == out_rank, f"rank {in_rank} {out_rank} mismatch"

    U, S, Vh = svd_lora(lora_down, lora_up, lora_rank, device, svd_method, svd_lowrank_niter)

    param_dict = rank_resize(S, lora_rank, dynamic_method, dynamic_param, scale)
    lora_rank = param_dict["new_rank"]

    U = U[:, :lora_rank] * S[:lora_rank]
    Vh = Vh[:lora_rank, :]

    param_dict["lora_down"] = Vh.reshape(lora_rank, in_size).cpu()
    param_dict["lora_up"] = U.reshape(out_size, lora_rank).cpu()
    return param_dict


# Calculate new rank


def rank_resize(S, rank, dynamic_method, dynamic_param, scale=1):
    param_dict = {}
    max_rank = min(rank, len(S))

    if dynamic_method == "sv_ratio":
        # Calculate new dim and alpha based off ratio
        new_rank = index_sv_ratio(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_cumulative":
        # Calculate new dim and alpha based off cumulative sum
        new_rank = index_sv_cumulative(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)

    elif dynamic_method == "sv_fro":
        # Calculate new dim and alpha based off sqrt sum of squares
        new_rank = index_sv_fro(S, dynamic_param) + 1
        new_alpha = float(scale * new_rank)
    else:
        new_rank = rank
        new_alpha = float(scale * new_rank)

    if S[0] <= MIN_SV:  # Zero matrix, set dim to 1
        new_rank = 1
        new_alpha = float(scale * new_rank)
    elif new_rank > max_rank:  # cap new rank at requested rank and available singular values
        new_rank = max_rank
        new_alpha = float(scale * new_rank)

    # Calculate resize info
    s_sum = torch.sum(torch.abs(S))
    s_rank = torch.sum(torch.abs(S[:new_rank]))

    S_squared = S.pow(2)
    s_fro = torch.sqrt(torch.sum(S_squared))
    s_red_fro = torch.sqrt(torch.sum(S_squared[:new_rank]))
    fro_percent = float(s_red_fro / s_fro)

    param_dict["new_rank"] = new_rank
    param_dict["new_alpha"] = new_alpha
    param_dict["sum_retained"] = (s_rank) / s_sum
    param_dict["fro_retained"] = fro_percent
    param_dict["max_ratio"] = S[0] / S[new_rank - 1]

    return param_dict


def resize_lora_model(
    lora_sd,
    new_rank,
    new_conv_rank,
    save_dtype,
    device,
    dynamic_method,
    dynamic_param,
    verbose,
    svd_lowrank_niter=None,
    svd_method="merged",
):
    if svd_method not in SVD_METHODS:
        raise ValueError(f"Unknown svd_method: {svd_method}, must be one of {SVD_METHODS}")

    if svd_method == "qr":
        if svd_lowrank_niter is not None:
            logger.warning("svd_lowrank_niter is ignored with svd_method 'qr' / svd_method 'qr' では svd_lowrank_niter は無視されます")
    else:
        if svd_lowrank_niter is None:
            svd_lowrank_niter = DEFAULT_SVD_LOWRANK_NITER
        logger.info(
            "'--svd_method qr' computes the exact SVD much faster without merging the LoRA weights"
            " / '--svd_method qr' を指定すると、LoRA の重みをマージせずに厳密な SVD を高速に計算できます"
        )

    max_old_rank = None
    new_alpha = None
    fro_list = []

    if dynamic_method:
        logger.info(
            f"Dynamically determining new alphas and dims based off {dynamic_method}: {dynamic_param}, max rank is {new_rank}"
        )

    lora_down_weight = None
    lora_up_weight = None

    o_lora_sd = lora_sd.copy()
    block_down_name = None
    block_up_name = None

    with torch.no_grad():
        for key, value in tqdm(lora_sd.items()):
            key_parts = key.split(".")
            block_down_name = None
            for _format in LORA_DOWN_UP_FORMATS:
                # Currently we only match lora_down_name in the last two parts of key
                # because ("down", "up") are general words and may appear in block_down_name
                if len(key_parts) >= 2 and _format[0] == key_parts[-2]:
                    block_down_name = ".".join(key_parts[:-2])
                    lora_down_name = "." + _format[0]
                    lora_up_name = "." + _format[1]
                    weight_name = "." + key_parts[-1]
                    break
                if len(key_parts) >= 1 and _format[0] == key_parts[-1]:
                    block_down_name = ".".join(key_parts[:-1])
                    lora_down_name = "." + _format[0]
                    lora_up_name = "." + _format[1]
                    weight_name = ""
                    break

            if block_down_name is None:
                # This parameter is not lora_down
                continue

            # Now weight_name can be ".weight" or ""
            # Find corresponding lora_up and alpha
            block_up_name = block_down_name
            lora_down_weight = value
            lora_up_weight = lora_sd.get(block_up_name + lora_up_name + weight_name, None)
            lora_alpha = lora_sd.get(block_down_name + ".alpha", None)

            weights_loaded = lora_down_weight is not None and lora_up_weight is not None

            if weights_loaded:

                conv2d = len(lora_down_weight.size()) == 4
                old_rank = lora_down_weight.size()[0]
                max_old_rank = max(max_old_rank or 0, old_rank)

                if lora_alpha is None:
                    scale = 1.0
                else:
                    scale = lora_alpha / old_rank

                if conv2d:
                    param_dict = extract_conv(
                        lora_down_weight,
                        lora_up_weight,
                        new_conv_rank,
                        dynamic_method,
                        dynamic_param,
                        device,
                        scale,
                        svd_method,
                        svd_lowrank_niter,
                    )
                else:
                    param_dict = extract_linear(
                        lora_down_weight,
                        lora_up_weight,
                        new_rank,
                        dynamic_method,
                        dynamic_param,
                        device,
                        scale,
                        svd_method,
                        svd_lowrank_niter,
                    )

                if verbose:
                    max_ratio = param_dict["max_ratio"]
                    sum_retained = param_dict["sum_retained"]
                    fro_retained = param_dict["fro_retained"]
                    if not np.isnan(fro_retained):
                        fro_list.append(float(fro_retained))

                    verbose_str = f"{block_down_name:75} | "
                    verbose_str += (
                        f"sum(S) retained: {sum_retained:.1%}, fro retained: {fro_retained:.1%}, max(S) ratio: {max_ratio:0.1f}"
                    )
                    if dynamic_method:
                        verbose_str += f", dynamic | dim: {param_dict['new_rank']}, alpha: {param_dict['new_alpha']}"
                    tqdm.write(verbose_str)

                new_alpha = param_dict["new_alpha"]
                o_lora_sd[block_down_name + lora_down_name + weight_name] = param_dict["lora_down"].to(save_dtype).contiguous()
                o_lora_sd[block_up_name + lora_up_name + weight_name] = param_dict["lora_up"].to(save_dtype).contiguous()
                o_lora_sd[block_down_name + ".alpha"] = torch.tensor(param_dict["new_alpha"]).to(save_dtype)

                block_down_name = None
                block_up_name = None
                lora_down_weight = None
                lora_up_weight = None
                weights_loaded = False
                del param_dict

    if verbose:
        print(f"Average Frobenius norm retention: {np.mean(fro_list):.2%} | std: {np.std(fro_list):0.3f}")
    logger.info("resizing complete")
    return o_lora_sd, max_old_rank, new_alpha


def resize(args):
    if args.save_to is None or not (
        args.save_to.endswith(".ckpt")
        or args.save_to.endswith(".pt")
        or args.save_to.endswith(".pth")
        or args.save_to.endswith(".safetensors")
    ):
        raise Exception("The --save_to argument must be specified and must be a .ckpt , .pt, .pth or .safetensors file.")

    args.new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    if args.dynamic_method and not args.dynamic_param:
        raise Exception("If using dynamic_method, then dynamic_param is required")

    merge_dtype = str_to_dtype("float")  # matmul method above only seems to work in float32
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    logger.info("loading Model...")
    lora_sd, metadata = load_state_dict(args.model, merge_dtype)

    logger.info("Resizing Lora...")
    state_dict, old_dim, new_alpha = resize_lora_model(
        lora_sd,
        args.new_rank,
        args.new_conv_rank,
        save_dtype,
        args.device,
        args.dynamic_method,
        args.dynamic_param,
        args.verbose,
        args.svd_lowrank_niter,
        args.svd_method,
    )

    # update metadata
    if metadata is None:
        metadata = {}

    comment = metadata.get("ss_training_comment", "")

    if not args.dynamic_method:
        conv_desc = "" if args.new_rank == args.new_conv_rank else f" (conv: {args.new_conv_rank})"
        metadata["ss_training_comment"] = f"dimension is resized from {old_dim} to {args.new_rank}{conv_desc}; {comment}"
        metadata["ss_network_dim"] = str(args.new_rank)
        metadata["ss_network_alpha"] = str(new_alpha)
    else:
        metadata["ss_training_comment"] = (
            f"Dynamic resize with {args.dynamic_method}: {args.dynamic_param} from {old_dim}; {comment}"
        )
        metadata["ss_network_dim"] = "Dynamic"
        metadata["ss_network_alpha"] = "Dynamic"

    # cast to save_dtype before calculating hashes
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
            state_dict[key] = value.to(save_dtype)

    model_hash, legacy_hash = model_io.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"saving model to: {args.save_to}")
    save_to_file(args.save_to, state_dict, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, float if omitted / 保存時の精度、未指定時はfloat",
    )
    parser.add_argument("--new_rank", type=int, default=4, help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LoRA model to resize at to new rank: ckpt or safetensors file / 読み込むLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Display verbose resizing information / rank変更時の詳細情報を出力する"
    )
    parser.add_argument(
        "--dynamic_method",
        type=str,
        default=None,
        choices=[None, "sv_ratio", "sv_fro", "sv_cumulative"],
        help="Specify dynamic resizing method, --new_rank is used as a hard limit for max rank",
    )
    parser.add_argument("--dynamic_param", type=float, default=None, help="Specify target for dynamic reduction")
    parser.add_argument(
        "--svd_lowrank_niter",
        type=int,
        default=None,
        help=f"Number of iterations for svd_lowrank on large matrices (>2048 dims) with svd_method 'merged', default is"
        f" {DEFAULT_SVD_LOWRANK_NITER}. 0 to disable and use full SVD. Ignored with svd_method 'qr'"
        f" / svd_method 'merged' で大行列(2048次元超)に対するsvd_lowrankの反復回数、デフォルトは{DEFAULT_SVD_LOWRANK_NITER}。"
        "0で無効化し完全SVDを使用。svd_method 'qr' では無視されます",
    )
    parser.add_argument(
        "--svd_method",
        type=str,
        default="merged",
        choices=SVD_METHODS,
        help="SVD method. 'merged' (default): SVD of the merged weight (lora_up @ lora_down), svd_lowrank is used for"
        " large matrices. 'qr': exact SVD via QR decomposition without merging the weight, much faster"
        " / SVDの方式。'merged'（デフォルト）: マージした重み (lora_up @ lora_down) をSVD、大行列にはsvd_lowrankを使用。"
        "'qr': QR分解により重みをマージせずに厳密なSVDを計算、大幅に高速",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    resize(args)
