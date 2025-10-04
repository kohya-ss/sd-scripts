# SDXL用LoRAからテキストエンコーダー(TE1/TE2)向けのLoRA層を取り除いた派生ファイルを一括生成するユーティリティです。
# 元のLoRAを読み込み、TE1のみ削除・TE2のみ削除・両方削除の3パターンを出力します。
#
# 使い方の例:
#   python tools/sdxl_prune_te.py my_lora.safetensors --output-dir pruned --overwrite
#   python tools/sdxl_prune_te.py my_lora.safetensors --suffix-te1 _clipL --suffix-te2 _clipG

import argparse
import logging
import os
from typing import Dict, Iterable, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from library.utils import add_logging_arguments, setup_logging

logger = logging.getLogger(__name__)

SDXL_TE1_PREFIX = "lora_te1"
SDXL_TE2_PREFIX = "lora_te2"


def load_lora(path: str) -> Tuple[Dict[str, torch.Tensor], str, Dict[str, str]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        state_dict = load_file(path)
        metadata: Dict[str, str] = {}
        try:
            with safe_open(path, framework="pt", device="cpu") as st_file:
                metadata = dict(st_file.metadata())
        except Exception as ex:  # metadata が壊れていても処理は続行する
            logger.warning("メタデータの読み込みに失敗しました: %s", ex)
        return state_dict, "safetensors", metadata

    state_dict = torch.load(path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError("LoRAファイルはstate_dict形式である必要があります")
    return state_dict, "pt", {}


def save_lora(state_dict: Dict[str, torch.Tensor], path: str, fmt: str, metadata: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt == "safetensors":
        save_file(state_dict, path, metadata=metadata or {})
    else:
        torch.save(state_dict, path)


def strip_by_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Tuple[Dict[str, torch.Tensor], int]:
    drop_prefixes = tuple(prefixes)
    removed = 0
    filtered: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        lora_id = key.split(".")[0]
        if any(lora_id.startswith(prefix) for prefix in drop_prefixes):
            removed += 1
            continue
        filtered[key] = value
    return filtered, removed


def build_output_path(input_path: str, output_dir: str, suffix: str) -> str:
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    ext = os.path.splitext(input_path)[1]
    return os.path.join(output_dir, f"{base_name}{suffix}{ext}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDXL LoRAからText Encoder(TE)のLoRA層を削除したバリエーションを同時に出力するユーティリティ"
    )
    parser.add_argument("model", type=str, help="入力となるLoRAファイル (.safetensorsまたは.pt)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="出力先ディレクトリ。省略時は入力ファイルと同じ場所に保存",
    )
    parser.add_argument(
        "--suffix-te1",
        type=str,
        default="_no-te1",
        help="TE1を削除したファイル名に付加するサフィックス",
    )
    parser.add_argument(
        "--suffix-te2",
        type=str,
        default="_no-te2",
        help="TE2を削除したファイル名に付加するサフィックス",
    )
    parser.add_argument(
        "--suffix-both",
        type=str,
        default="_no-te12",
        help="TE1とTE2両方を削除したファイル名に付加するサフィックス",
    )
    parser.add_argument("--overwrite", action="store_true", help="既存の出力ファイルを上書きする")
    add_logging_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args, reset=True)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.model)) or "."

    logger.info("LoRAを読み込み中: %s", args.model)
    state_dict, fmt, metadata = load_lora(args.model)
    logger.info("テンソル数: %d", len(state_dict))

    variants = [
        (args.suffix_te1, (SDXL_TE1_PREFIX,), "TE1のみ削除"),
        (args.suffix_te2, (SDXL_TE2_PREFIX,), "TE2のみ削除"),
        (args.suffix_both, (SDXL_TE1_PREFIX, SDXL_TE2_PREFIX), "TE1+TE2を削除"),
    ]

    for suffix, prefixes, label in variants:
        output_path = build_output_path(args.model, output_dir, suffix)
        if os.path.exists(output_path) and not args.overwrite:
            logger.error("出力先が既に存在するためスキップしました: %s", output_path)
            continue

        filtered, removed = strip_by_prefix(state_dict, prefixes)
        logger.info("%s: %d 個のLoRAパラメータを削除", label, removed)
        if removed == 0:
            logger.warning("%s 対象のパラメータが見つかりませんでした", label)

        save_lora(filtered, output_path, fmt, metadata)
        logger.info("保存しました: %s (残りテンソル %d)", output_path, len(filtered))


if __name__ == "__main__":
    main()
