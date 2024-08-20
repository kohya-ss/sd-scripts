import argparse
import math
import os
import time

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

import lora_flux as lora_flux
from library import sai_model_spec, train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_state_dict(file_name, dtype):
    if os.path.splitext(file_name)[1] == ".safetensors":
        sd = load_file(file_name)
        metadata = train_util.load_metadata_from_safetensors(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = {}

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def save_to_file(file_name, state_dict, dtype, metadata):
    if dtype is not None:
        logger.info(f"converting to {dtype}...")
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    logger.info(f"saving to: {file_name}")
    save_file(state_dict, file_name, metadata=metadata)


def merge_to_flux_model(
    loading_device, working_device, flux_model, models, ratios, merge_dtype, save_dtype
):
    logger.info(f"loading keys from FLUX.1 model: {flux_model}")
    flux_state_dict = load_file(flux_model, device=loading_device)

    def create_key_map(n_double_layers, n_single_layers, hidden_size):
        key_map = {}
        for index in range(n_double_layers):
            prefix_from = f"transformer_blocks.{index}"
            prefix_to = f"double_blocks.{index}"

            for end in ("weight", "bias"):
                k = f"{prefix_from}.attn."
                qkv_img = f"{prefix_to}.img_attn.qkv.{end}"
                qkv_txt = f"{prefix_to}.txt_attn.qkv.{end}"

                key_map[f"{k}to_q.{end}"] = (qkv_img, (0, 0, hidden_size))
                key_map[f"{k}to_k.{end}"] = (qkv_img, (0, hidden_size, hidden_size))
                key_map[f"{k}to_v.{end}"] = (qkv_img, (0, hidden_size * 2, hidden_size))
                key_map[f"{k}add_q_proj.{end}"] = (qkv_txt, (0, 0, hidden_size))
                key_map[f"{k}add_k_proj.{end}"] = (
                    qkv_txt,
                    (0, hidden_size, hidden_size),
                )
                key_map[f"{k}add_v_proj.{end}"] = (
                    qkv_txt,
                    (0, hidden_size * 2, hidden_size),
                )

            block_map = {
                "attn.to_out.0.weight": "img_attn.proj.weight",
                "attn.to_out.0.bias": "img_attn.proj.bias",
                "norm1.linear.weight": "img_mod.lin.weight",
                "norm1.linear.bias": "img_mod.lin.bias",
                "norm1_context.linear.weight": "txt_mod.lin.weight",
                "norm1_context.linear.bias": "txt_mod.lin.bias",
                "attn.to_add_out.weight": "txt_attn.proj.weight",
                "attn.to_add_out.bias": "txt_attn.proj.bias",
                "ff.net.0.proj.weight": "img_mlp.0.weight",
                "ff.net.0.proj.bias": "img_mlp.0.bias",
                "ff.net.2.weight": "img_mlp.2.weight",
                "ff.net.2.bias": "img_mlp.2.bias",
                "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
                "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
                "ff_context.net.2.weight": "txt_mlp.2.weight",
                "ff_context.net.2.bias": "txt_mlp.2.bias",
                "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
                "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
                "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
                "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
            }

            for k, v in block_map.items():
                key_map[f"{prefix_from}.{k}"] = f"{prefix_to}.{v}"

        for index in range(n_single_layers):
            prefix_from = f"single_transformer_blocks.{index}"
            prefix_to = f"single_blocks.{index}"

            for end in ("weight", "bias"):
                k = f"{prefix_from}.attn."
                qkv = f"{prefix_to}.linear1.{end}"
                key_map[f"{k}to_q.{end}"] = (qkv, (0, 0, hidden_size))
                key_map[f"{k}to_k.{end}"] = (qkv, (0, hidden_size, hidden_size))
                key_map[f"{k}to_v.{end}"] = (qkv, (0, hidden_size * 2, hidden_size))
                key_map[f"{prefix_from}.proj_mlp.{end}"] = (
                    qkv,
                    (0, hidden_size * 3, hidden_size * 4),
                )

            block_map = {
                "norm.linear.weight": "modulation.lin.weight",
                "norm.linear.bias": "modulation.lin.bias",
                "proj_out.weight": "linear2.weight",
                "proj_out.bias": "linear2.bias",
                "attn.norm_q.weight": "norm.query_norm.scale",
                "attn.norm_k.weight": "norm.key_norm.scale",
            }

            for k, v in block_map.items():
                key_map[f"{prefix_from}.{k}"] = f"{prefix_to}.{v}"

        return key_map

    key_map = create_key_map(
        18, 1, 2048
    )  # Assuming 18 double layers, 1 single layer, and hidden size of 2048

    def find_matching_key(flux_dict, lora_key):
        lora_key = lora_key.replace("diffusion_model.", "")
        lora_key = lora_key.replace("transformer.", "")
        lora_key = lora_key.replace("lora_A", "lora_down").replace("lora_B", "lora_up")
        lora_key = lora_key.replace("single_transformer_blocks", "single_blocks")
        lora_key = lora_key.replace("transformer_blocks", "double_blocks")

        double_block_map = {
            "attn.to_out.0": "img_attn.proj",
            "norm1.linear": "img_mod.lin",
            "norm1_context.linear": "txt_mod.lin",
            "attn.to_add_out": "txt_attn.proj",
            "ff.net.0.proj": "img_mlp.0",
            "ff.net.2": "img_mlp.2",
            "ff_context.net.0.proj": "txt_mlp.0",
            "ff_context.net.2": "txt_mlp.2",
            "attn.norm_q": "img_attn.norm.query_norm",
            "attn.norm_k": "img_attn.norm.key_norm",
            "attn.norm_added_q": "txt_attn.norm.query_norm",
            "attn.norm_added_k": "txt_attn.norm.key_norm",
            "attn.to_q": "img_attn.qkv",
            "attn.to_k": "img_attn.qkv",
            "attn.to_v": "img_attn.qkv",
            "attn.add_q_proj": "txt_attn.qkv",
            "attn.add_k_proj": "txt_attn.qkv",
            "attn.add_v_proj": "txt_attn.qkv",
        }

        single_block_map = {
            "norm.linear": "modulation.lin",
            "proj_out": "linear2",
            "attn.norm_q": "norm.query_norm",
            "attn.norm_k": "norm.key_norm",
            "attn.to_q": "linear1",
            "attn.to_k": "linear1",
            "attn.to_v": "linear1",
        }

        for old, new in double_block_map.items():
            lora_key = lora_key.replace(old, new)

        for old, new in single_block_map.items():
            lora_key = lora_key.replace(old, new)

        if lora_key in key_map:
            flux_key = key_map[lora_key]
            if isinstance(flux_key, tuple):
                flux_key = flux_key[0]
            logger.info(f"Found matching key: {flux_key}")
            return flux_key

        # If not found in key_map, try partial matching
        potential_key = lora_key + ".weight"
        logger.info(f"Searching for key: {potential_key}")
        matches = [k for k in flux_dict.keys() if potential_key in k]
        if matches:
            logger.info(f"Found matching key: {matches[0]}")
            return matches[0]
        return None

    merged_keys = set()
    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)

        logger.info("merging...")
        for key in tqdm(lora_sd.keys()):
            if "lora_down" in key or "lora_A" in key:
                lora_name = key[
                    : key.rfind(".lora_down" if "lora_down" in key else ".lora_A")
                ]
                up_key = key.replace("lora_down", "lora_up").replace("lora_A", "lora_B")
                alpha_key = (
                    key[: key.index("lora_down" if "lora_down" in key else "lora_A")]
                    + "alpha"
                )

                logger.info(f"Processing LoRA key: {lora_name}")
                flux_key = find_matching_key(flux_state_dict, lora_name)

                if flux_key is None:
                    logger.warning(f"no module found for LoRA weight: {key}")
                    continue

                logger.info(f"Merging LoRA key {lora_name} into Flux key {flux_key}")

                down_weight = lora_sd[key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                weight = flux_state_dict[flux_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                if lora_name.startswith("transformer."):
                    if "qkv" in flux_key:
                        hidden_size = weight.size(-1) // 3
                        update = ratio * (up_weight @ down_weight) * scale

                        if "img_attn" in flux_key or "txt_attn" in flux_key:
                            q, k, v = torch.chunk(weight, 3, dim=-1)
                            if "to_q" in lora_name or "add_q_proj" in lora_name:
                                q += update.reshape(q.shape)
                            elif "to_k" in lora_name or "add_k_proj" in lora_name:
                                k += update.reshape(k.shape)
                            elif "to_v" in lora_name or "add_v_proj" in lora_name:
                                v += update.reshape(v.shape)
                            weight = torch.cat([q, k, v], dim=-1)
                    else:
                        if len(weight.size()) == 2:
                            weight = weight + ratio * (up_weight @ down_weight) * scale
                        elif down_weight.size()[2:4] == (1, 1):
                            weight = (
                                weight
                                + ratio
                                * (
                                    up_weight.squeeze(3).squeeze(2)
                                    @ down_weight.squeeze(3).squeeze(2)
                                )
                                .unsqueeze(2)
                                .unsqueeze(3)
                                * scale
                            )
                        else:
                            conved = torch.nn.functional.conv2d(
                                down_weight.permute(1, 0, 2, 3), up_weight
                            ).permute(1, 0, 2, 3)
                            weight = weight + ratio * conved * scale
                else:
                    if len(weight.size()) == 2:
                        weight = weight + ratio * (up_weight @ down_weight) * scale
                    elif down_weight.size()[2:4] == (1, 1):
                        weight = (
                            weight
                            + ratio
                            * (
                                up_weight.squeeze(3).squeeze(2)
                                @ down_weight.squeeze(3).squeeze(2)
                            )
                            .unsqueeze(2)
                            .unsqueeze(3)
                            * scale
                        )
                    else:
                        conved = torch.nn.functional.conv2d(
                            down_weight.permute(1, 0, 2, 3), up_weight
                        ).permute(1, 0, 2, 3)
                        weight = weight + ratio * conved * scale

                flux_state_dict[flux_key] = weight.to(loading_device, save_dtype)
                merged_keys.add(flux_key)
                del up_weight
                del down_weight
                del weight

    logger.info(f"Merged keys: {sorted(list(merged_keys))}")
    return flux_state_dict


def merge_lora_models(models, ratios, merge_dtype, concat=False, shuffle=False):
    base_alphas = {}  # alpha for merged model
    base_dims = {}

    merged_sd = {}
    base_model = None
    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        if lora_metadata is not None:
            if base_model is None:
                base_model = lora_metadata.get(
                    train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None
                )

        # get alpha and dim
        alphas = {}  # alpha for current model
        dims = {}  # dims for current model
        for key in lora_sd.keys():
            if "alpha" in key:
                lora_module_name = key[: key.rfind(".alpha")]
                alpha = float(lora_sd[key].detach().numpy())
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha
            elif "lora_down" in key:
                lora_module_name = key[: key.rfind(".lora_down")]
                dim = lora_sd[key].size()[0]
                dims[lora_module_name] = dim
                if lora_module_name not in base_dims:
                    base_dims[lora_module_name] = dim

        for lora_module_name in dims.keys():
            if lora_module_name not in alphas:
                alpha = dims[lora_module_name]
                alphas[lora_module_name] = alpha
                if lora_module_name not in base_alphas:
                    base_alphas[lora_module_name] = alpha

        logger.info(
            f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}"
        )

        # merge
        logger.info("merging...")
        for key in tqdm(lora_sd.keys()):
            if "alpha" in key:
                continue

            if "lora_up" in key and concat:
                concat_dim = 1
            elif "lora_down" in key and concat:
                concat_dim = 0
            else:
                concat_dim = None

            lora_module_name = key[: key.rfind(".lora_")]

            base_alpha = base_alphas[lora_module_name]
            alpha = alphas[lora_module_name]

            scale = math.sqrt(alpha / base_alpha) * ratio
            scale = (
                abs(scale) if "lora_up" in key else scale
            )  # マイナスの重みに対応する。

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size()
                    or concat_dim is not None
                ), "weights shape mismatch, different dims? / 重みのサイズが合いません。dimが異なる可能性があります。"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat(
                        [merged_sd[key], lora_sd[key] * scale], dim=concat_dim
                    )
                else:
                    merged_sd[key] = merged_sd[key] + lora_sd[key] * scale
            else:
                merged_sd[key] = lora_sd[key] * scale

    # set alpha to sd
    for lora_module_name, alpha in base_alphas.items():
        key = lora_module_name + ".alpha"
        merged_sd[key] = torch.tensor(alpha)
        if shuffle:
            key_down = lora_module_name + ".lora_down.weight"
            key_up = lora_module_name + ".lora_up.weight"
            dim = merged_sd[key_down].shape[0]
            perm = torch.randperm(dim)
            merged_sd[key_down] = merged_sd[key_down][perm]
            merged_sd[key_up] = merged_sd[key_up][:, perm]

    logger.info("merged model")
    logger.info(
        f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}"
    )

    # check all dims are same
    dims_list = list(set(base_dims.values()))
    alphas_list = list(set(base_alphas.values()))
    all_same_dims = True
    all_same_alphas = True
    for dims in dims_list:
        if dims != dims_list[0]:
            all_same_dims = False
            break
    for alphas in alphas_list:
        if alphas != alphas_list[0]:
            all_same_alphas = False
            break

    # build minimum metadata
    dims = f"{dims_list[0]}" if all_same_dims else "Dynamic"
    alphas = f"{alphas_list[0]}" if all_same_alphas else "Dynamic"
    metadata = train_util.build_minimum_network_metadata(
        str(False), base_model, "networks.lora", dims, alphas, None
    )

    return merged_sd, metadata


def merge(args):
    assert (
        len(args.models) == len(args.ratios)
    ), "number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    def str_to_dtype(p):
        if p == "float":
            return torch.float
        if p == "fp16":
            return torch.float16
        if p == "bf16":
            return torch.bfloat16
        return None

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    dest_dir = os.path.dirname(args.save_to)
    if not os.path.exists(dest_dir):
        logger.info(f"creating directory: {dest_dir}")
        os.makedirs(dest_dir)

    if args.flux_model is not None:
        state_dict = merge_to_flux_model(
            args.loading_device,
            args.working_device,
            args.flux_model,
            args.models,
            args.ratios,
            merge_dtype,
            save_dtype,
        )

        if args.no_metadata:
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from(
                [args.flux_model] + args.models
            )
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None,
                False,
                False,
                False,
                False,
                False,
                time.time(),
                title=title,
                merged_from=merged_from,
                flux="dev",
            )

        logger.info(f"saving FLUX model to: {args.save_to}")
        save_to_file(args.save_to, state_dict, save_dtype, sai_metadata)

    else:
        state_dict, metadata = merge_lora_models(
            args.models, args.ratios, merge_dtype, args.concat, args.shuffle
        )

        logger.info("calculating hashes and creating metadata...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(
            state_dict, metadata
        )
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                state_dict,
                False,
                False,
                False,
                True,
                False,
                time.time(),
                title=title,
                merged_from=merged_from,
                flux="dev",
            )
            metadata.update(sai_metadata)

        logger.info(f"saving model to: {args.save_to}")
        save_to_file(args.save_to, state_dict, save_dtype, metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--flux_model",
        type=str,
        default=None,
        help="FLUX.1 model to load, merge LoRA models if omitted / 読み込むモデル、指定しない場合はLoRAモデルをマージする",
    )
    parser.add_argument(
        "--loading_device",
        type=str,
        default="cpu",
        help="device to load FLUX.1 model. LoRA models are loaded on CPU / FLUX.1モデルを読み込むデバイス。LoRAモデルはCPUで読み込まれます",
    )
    parser.add_argument(
        "--working_device",
        type=str,
        default="cpu",
        help="device to work (merge). Merging LoRA models are done on CPU."
        + " / 作業（マージ）するデバイス。LoRAモデルのマージはCPUで行われます。",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        help="destination file name: safetensors file / 保存先のファイル名、safetensorsファイル",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: safetensors file / マージするLoRAモデル、safetensorsファイル",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="*",
        help="ratios for each model / それぞれのLoRAモデルの比率",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="concat lora instead of merge (The dim(rank) of the output LoRA is the sum of the input dims) / "
        + "マージの代わりに結合する（LoRAのdim(rank)は入力dimの合計になる）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="shuffle lora weight./ " + "LoRAの重みをシャッフルする",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)
