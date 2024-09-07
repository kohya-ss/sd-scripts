import argparse
import itertools
import json
import os
import re
import time
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from library import sai_model_spec, train_util
import library.model_util as model_util
import lora
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

CLAMP_QUANTILE = 0.99

# copied from hako-mikan/sd-webui-lora-block-weight/scripts/lora_block_weight.py
BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]
BLOCKNUMS = [12,17,20,26]
BLOCKIDS=[BLOCKID12,BLOCKID17,BLOCKID20,BLOCKID26]

BLOCKS=["encoder",  # BASE
"diffusion_model_input_blocks_0_",  # IN00
"diffusion_model_input_blocks_1_",  # IN01
"diffusion_model_input_blocks_2_",  # IN02
"diffusion_model_input_blocks_3_",  # IN03
"diffusion_model_input_blocks_4_",  # IN04
"diffusion_model_input_blocks_5_",  # IN05
"diffusion_model_input_blocks_6_",  # IN06
"diffusion_model_input_blocks_7_",  # IN07
"diffusion_model_input_blocks_8_",  # IN08
"diffusion_model_input_blocks_9_",  # IN09
"diffusion_model_input_blocks_10_",  # IN10
"diffusion_model_input_blocks_11_",  # IN11
"diffusion_model_middle_block_",  # M00
"diffusion_model_output_blocks_0_",  # OUT00
"diffusion_model_output_blocks_1_",  # OUT01
"diffusion_model_output_blocks_2_",  # OUT02
"diffusion_model_output_blocks_3_",  # OUT03
"diffusion_model_output_blocks_4_",  # OUT04
"diffusion_model_output_blocks_5_",  # OUT05
"diffusion_model_output_blocks_6_",  # OUT06
"diffusion_model_output_blocks_7_",  # OUT07
"diffusion_model_output_blocks_8_",  # OUT08
"diffusion_model_output_blocks_9_",  # OUT09
"diffusion_model_output_blocks_10_",  # OUT10
"diffusion_model_output_blocks_11_",  # OUT11
"embedders",
"transformer_resblocks"]


def convert_diffusers_name_to_compvis(key, is_sd2):
    "copied from AUTOMATIC1111/stable-diffusion-webui/extensions-builtin/Lora/networks.py"

    # put original globals here
    re_digits = re.compile(r"\d+")
    re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
    re_compiled = {}

    suffix_conversion = {
        "attentions": {},
        "resnets": {
            "conv1": "in_layers_2",
            "conv2": "out_layers_3",
            "time_emb_proj": "emb_layers_1",
            "conv_shortcut": "skip_connection",
        }
    }  # end of original globals

    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key


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
        for key in list(state_dict.keys()):
            if type(state_dict[key]) == torch.Tensor:
                state_dict[key] = state_dict[key].to(dtype)

    if os.path.splitext(file_name)[1] == ".safetensors":
        save_file(state_dict, file_name, metadata=metadata)
    else:
        torch.save(state_dict, file_name)


def merge_lora_models(is_sd2, models, ratios, lbws, new_rank, new_conv_rank, device, merge_dtype):
    logger.info(f"new rank: {new_rank}, new conv rank: {new_conv_rank}")
    merged_sd = {}
    v2 = None
    base_model = None

    if lbws:
        try:
            # lbsは"[1,1,1,1,1,1,1,1,1,1,1,1]"のような文字列で与えられることを期待している
            lbws = [json.loads(lbw) for lbw in lbws]
        except Exception:
            raise ValueError(f"format of lbws are must be json / 層別適用率はJSON形式で書いてください")
        assert all(isinstance(lbw, list) for lbw in lbws), f"lbws are must be list / 層別適用率はリストにしてください"
        assert len(set(len(lbw) for lbw in lbws)) == 1, "all lbws should have the same length  / 層別適用率は同じ長さにしてください"
        assert all(len(lbw) in BLOCKNUMS for lbw in lbws), f"length of lbw are must be in {BLOCKNUMS} / 層別適用率の長さは{BLOCKNUMS}のいずれかにしてください"
        assert all(all(isinstance(weight, (int, float)) for weight in lbw) for lbw in lbws), f"values of lbs are must be numbers / 層別適用率の値はすべて数値にしてください"

        BLOCKID = BLOCKIDS[BLOCKNUMS.index(len(lbws[0]))]
        conditions = [blockid in BLOCKID for blockid in BLOCKID26]
        BLOCKS_ = [block for block, condition in zip(BLOCKS, conditions) if condition]

    for model, ratio, lbw in itertools.zip_longest(models, ratios, lbws):
        logger.info(f"loading: {model}")
        lora_sd, lora_metadata = load_state_dict(model, merge_dtype)

        if lora_metadata is not None:
            if v2 is None:
                v2 = lora_metadata.get(train_util.SS_METADATA_KEY_V2, None)  # return string
            if base_model is None:
                base_model = lora_metadata.get(train_util.SS_METADATA_KEY_BASE_MODEL_VERSION, None)

        # merge
        logger.info(f"merging...")
        for key in tqdm(list(lora_sd.keys())):
            if "lora_down" not in key:
                continue

            if lbw:
                # keyをlora_unet_down_blocks_0_のようなdiffusers形式から、
                # diffusion_model_input_blocks_0_のようなcompvis形式に変換する
                compvis_key = convert_diffusers_name_to_compvis(key, is_sd2)

                block_in_key = [block in compvis_key for block in BLOCKS_]
                is_lbw_target = any(block_in_key)
                if is_lbw_target:
                    index = [i for i, in_key in enumerate(block_in_key) if in_key][0]
                    lbw_weight = lbw[index]

            lora_module_name = key[: key.rfind(".lora_down")]

            down_weight = lora_sd[key]
            network_dim = down_weight.size()[0]

            up_weight = lora_sd[lora_module_name + ".lora_up.weight"]
            alpha = lora_sd.get(lora_module_name + ".alpha", network_dim)

            in_dim = down_weight.size()[1]
            out_dim = up_weight.size()[0]
            conv2d = len(down_weight.size()) == 4
            kernel_size = None if not conv2d else down_weight.size()[2:4]
            # logger.info(lora_module_name, network_dim, alpha, in_dim, out_dim, kernel_size)

            # make original weight if not exist
            if lora_module_name not in merged_sd:
                weight = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)
                if device:
                    weight = weight.to(device)
            else:
                weight = merged_sd[lora_module_name]

            # merge to weight
            if device:
                up_weight = up_weight.to(device)
                down_weight = down_weight.to(device)

            # W <- W + U * D
            scale = alpha / network_dim
            if lbw:
                if is_lbw_target:
                    scale *= lbw_weight  # keyがlbwの対象であれば、lbwの重みを掛ける

            if device:  # and isinstance(scale, torch.Tensor):
                scale = scale.to(device)

            if not conv2d:  # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif kernel_size == (1, 1):
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + ratio * conved * scale

            merged_sd[lora_module_name] = weight

    # extract from merged weights
    logger.info("extract new lora...")
    merged_lora_sd = {}
    with torch.no_grad():
        for lora_module_name, mat in tqdm(list(merged_sd.items())):
            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = mat.size()[0:2]

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
            module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

            U, S, Vh = torch.linalg.svd(mat)

            U = U[:, :module_new_rank]
            S = S[:module_new_rank]
            U = U @ torch.diag(S)

            Vh = Vh[:module_new_rank, :]

            dist = torch.cat([U.flatten(), Vh.flatten()])
            hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            low_val = -hi_val

            U = U.clamp(low_val, hi_val)
            Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, module_new_rank, 1, 1)
                Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])

            up_weight = U
            down_weight = Vh

            merged_lora_sd[lora_module_name + ".lora_up.weight"] = up_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".lora_down.weight"] = down_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".alpha"] = torch.tensor(module_new_rank)

    # build minimum metadata
    dims = f"{new_rank}"
    alphas = f"{new_rank}"
    if new_conv_rank is not None:
        network_args = {"conv_dim": new_conv_rank, "conv_alpha": new_conv_rank}
    else:
        network_args = None
    metadata = train_util.build_minimum_network_metadata(v2, base_model, "networks.lora", dims, alphas, network_args)

    return merged_lora_sd, metadata, v2 == "True", base_model


def merge(args):
    assert len(args.models) == len(args.ratios), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"
    if args.lbws:
        assert len(args.models) == len(args.lbws), f"number of models must be equal to number of ratios / モデルの数と層別適用率の数は合わせてください"
    else:
        args.lbws = []  # zip_longestで扱えるようにlbws未使用時には空のリストにしておく

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

    new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    state_dict, metadata, v2, base_model = merge_lora_models(
        args.sd2, args.models, args.ratios, args.lbws, args.new_rank, new_conv_rank, args.device, merge_dtype
    )

    logger.info(f"calculating hashes and creating metadata...")

    model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash

    if not args.no_metadata:
        is_sdxl = base_model is not None and base_model.lower().startswith("sdxl")
        merged_from = sai_model_spec.build_merged_from(args.models)
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        sai_metadata = sai_model_spec.build_metadata(
            state_dict, v2, v2, is_sdxl, True, False, time.time(), title=title, merged_from=merged_from
        )
        if v2:
            # TODO read sai modelspec
            logger.warning(
                "Cannot determine if LoRA is for v-prediction, so save metadata as v-prediction / LoRAがv-prediction用か否か不明なため、仮にv-prediction用としてmetadataを保存します"
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
        "--save_to", type=str, default=None, help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors"
    )
    parser.add_argument(
        "--sd2", action="store_true", help="set if LoRA models are for SD2 / マージするLoRAモデルがSD2用なら指定します"
    )
    parser.add_argument(
        "--models", type=str, nargs="*", help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors"
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument("--lbws", type=str, nargs="*", help="lbw for each model / それぞれのLoRAモデルの層別適用率")
    parser.add_argument("--new_rank", type=int, default=4, help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ",
    )
    parser.add_argument("--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う")
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    merge(args)
