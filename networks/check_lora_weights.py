import argparse
import os
import torch
from safetensors.torch import load_file
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

def main(file):
    logger.info(f"loading: {file}")
    if os.path.splitext(file)[1] == ".safetensors":
        sd = load_file(file)
    else:
        sd = torch.load(file, map_location="cpu")

    values = []

    keys = list(sd.keys())
    for key in keys:
        if "lora_up" in key or "lora_down" in key:
            values.append((key, sd[key]))
    logger.info(f"number of LoRA modules: {len(values)}")

    if args.show_all_keys:
        for key in [k for k in keys if k not in values]:
            values.append((key, sd[key]))
        logger.info(f"number of all modules: {len(values)}")

    for key, value in values:
        value = value.to(torch.float32)
        logger.info(f"{key},{str(tuple(value.size())).replace(', ', '-')},{torch.mean(torch.abs(value))},{torch.min(torch.abs(value))}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="model file to check / 重みを確認するモデルファイル")
    parser.add_argument("-s", "--show_all_keys", action="store_true", help="show all keys / 全てのキーを表示する")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    main(args.file)
