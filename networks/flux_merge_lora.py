import math
import argparse
import os
import time
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, train_util
import networks.lora_flux as lora_flux
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


def merge_to_flux_model(loading_device, working_device, flux_model, models, ratios, merge_dtype, save_dtype):
    # create module map without loading state_dict
    logger.info(f"loading keys from FLUX.1 model: {flux_model}")
    lora_name_to_module_key = {}
    with safe_open(flux_model, framework="pt", device=loading_device) as flux_file:
        keys = list(flux_file.keys())
        for key in keys:
            if key.endswith(".weight"):
                module_name = ".".join(key.split(".")[:-1])
                lora_name = lora_flux.LoRANetwork.LORA_PREFIX_FLUX + "_" + module_name.replace(".", "_")
                lora_name_to_module_key[lora_name] = key

    flux_state_dict = load_file(flux_model, device=loading_device)
    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, _ = load_state_dict(model, merge_dtype)  # loading on CPU

        logger.info(f"merging...")
        for key in tqdm(lora_sd.keys()):
            if "lora_down" in key:
                lora_name = key[: key.rfind(".lora_down")]
                up_key = key.replace("lora_down", "lora_up")
                alpha_key = key[: key.index("lora_down")] + "alpha"

                if lora_name not in lora_name_to_module_key:
                    logger.warning(f"no module found for LoRA weight: {key}.  LoRA for Text Encoder is not supported yet.")
                    continue

                down_weight = lora_sd[key]
                up_weight = lora_sd[up_key]

                dim = down_weight.size()[0]
                alpha = lora_sd.get(alpha_key, dim)
                scale = alpha / dim

                # W <- W + U * D
                module_weight_key = lora_name_to_module_key[lora_name]
                if module_weight_key not in flux_state_dict:
                    weight = flux_file.get_tensor(module_weight_key)
                else:
                    weight = flux_state_dict[module_weight_key]

                weight = weight.to(working_device, merge_dtype)
                up_weight = up_weight.to(working_device, merge_dtype)
                down_weight = down_weight.to(working_device, merge_dtype)

                # logger.info(module_name, down_weight.size(), up_weight.size())
                if len(weight.size()) == 2:
                    # linear
                    weight = weight + ratio * (up_weight @ down_weight) * scale
                elif down_weight.size()[2:4] == (1, 1):
                    # conv2d 1x1
                    weight = (
                        weight
                        + ratio
                        * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                        * scale
                    )
                else:
                    # conv2d 3x3
                    conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                    # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                    weight = weight + ratio * conved * scale

                flux_state_dict[module_weight_key] = weight.to(loading_device, save_dtype)
                del up_weight
                del down_weight
                del weight

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
                base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

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

        logger.info(f"dim: {list(set(dims.values()))}, alpha: {list(set(alphas.values()))}")

        # merge
        logger.info(f"merging...")
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
            scale = abs(scale) if "lora_up" in key else scale  # マイナスの重みに対応する。

            if key in merged_sd:
                assert (
                    merged_sd[key].size() == lora_sd[key].size() or concat_dim is not None
                ), f"weights shape mismatch, different dims? / 重みのサイズが合いません。dimが異なる可能性があります。"
                if concat_dim is not None:
                    merged_sd[key] = torch.cat([merged_sd[key], lora_sd[key] * scale], dim=concat_dim)
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
    logger.info(f"dim: {list(set(base_dims.values()))}, alpha: {list(set(base_alphas.values()))}")

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
    metadata = train_util.build_minimum_network_metadata(str(False), base_model, "networks.lora", dims, alphas, None)

    return merged_sd, metadata


def merge(args):
    assert len(args.models) == len(
        args.ratios
    ), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

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
            args.loading_device, args.working_device, args.flux_model, args.models, args.ratios, merge_dtype, save_dtype
        )

        if args.no_metadata:
            sai_metadata = None
        else:
            merged_from = sai_model_spec.build_merged_from([args.flux_model] + args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                None, False, False, False, False, False, time.time(), title=title, merged_from=merged_from, flux="dev"
            )

        logger.info(f"saving FLUX model to: {args.save_to}")
        save_to_file(args.save_to, state_dict, save_dtype, sai_metadata)

    else:
        state_dict, metadata = merge_lora_models(args.models, args.ratios, merge_dtype, args.concat, args.shuffle)

        logger.info(f"calculating hashes and creating metadata...")

        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

        if not args.no_metadata:
            merged_from = sai_model_spec.build_merged_from(args.models)
            title = os.path.splitext(os.path.basename(args.save_to))[0]
            sai_metadata = sai_model_spec.build_metadata(
                state_dict, False, False, False, True, False, time.time(), title=title, merged_from=merged_from, flux="dev"
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
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
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
