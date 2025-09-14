import argparse
from safetensors.torch import save_file
from safetensors import safe_open
import torch


from library import train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def main(args):
    # load source safetensors
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)

    logger.info(f"Converting...")

    keys = list(state_dict.keys())
    count = 0
    for k in keys:
        if "double_blocks" in k:
            new_k = (
                k.replace("img_mlp_fc1", "img_mlp_0").replace("img_mlp_fc2", "img_mlp_2").replace("img_mod_linear", "img_mod_lin")
            )
            new_k = (
                new_k.replace("txt_mlp_fc1", "txt_mlp_0")
                .replace("txt_mlp_fc2", "txt_mlp_2")
                .replace("txt_mod_linear", "txt_mod_lin")
            )
            if new_k != k:
                state_dict[new_k] = state_dict.pop(k)
                count += 1
                # print(f"Renamed {k} to {new_k}")
        elif "single_blocks" in k:
            new_k = k.replace("modulation_linear", "modulation_lin")
            if new_k != k:
                state_dict[new_k] = state_dict.pop(k)
                count += 1
                # print(f"Renamed {k} to {new_k}")
    logger.info(f"Converted {count} keys")

    # Calculate hash
    if metadata is not None:
        logger.info(f"Calculating hashes and creating metadata...")
        model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    # save destination safetensors
    logger.info(f"Saving destination file {args.dst_path}")
    save_file(state_dict, args.dst_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LoRA format")
    parser.add_argument("src_path", type=str, default=None, help="source path, sd-scripts format")
    parser.add_argument("dst_path", type=str, default=None, help="destination path, ComfyUI format")
    args = parser.parse_args()
    main(args)
