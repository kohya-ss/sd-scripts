import argparse
import os
import torch
from safetensors.torch import load_file
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging
logger = logging.getLogger(__name__)


def analyze_lora(file: str) -> None:
    logger.info(f"loading LoRA: {file}")
    if os.path.splitext(file)[1] == ".safetensors":
        state_dict = load_file(file)
    else:
        state_dict = torch.load(file, map_location="cpu")

    layers = {}
    for key, value in state_dict.items():
        if key.endswith(".lora_down.weight"):
            layer = key[: -len(".lora_down.weight")]
            layers.setdefault(layer, {})["down"] = value
        elif key.endswith(".lora_up.weight"):
            layer = key[: -len(".lora_up.weight")]
            layers.setdefault(layer, {})["up"] = value
        elif key.endswith(".alpha"):
            layer = key[: -len(".alpha")]
            layers.setdefault(layer, {})["alpha"] = value

    for name in sorted(layers.keys()):
        info = layers[name]
        if "down" not in info or "up" not in info:
            logger.warning(f"missing weights for layer {name}, skipped")
            continue
        down = info["down"].float()
        up = info["up"].float()
        num_params = down.numel() + up.numel()
        fro = torch.linalg.vector_norm(down) ** 2 + torch.linalg.vector_norm(up) ** 2
        fro = torch.sqrt(fro)
        print(f"{name},{num_params},{fro.item():.6f}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze LoRA layers")
    parser.add_argument("model", type=str, help="LoRA model file")
    add_logging_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    setup_logging(args, reset=True)
    analyze_lora(args.model)

