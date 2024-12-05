# add caption to images by Florence-2


import argparse
import json
import os
import glob
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from library import device_utils, train_util
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

import tagger_utils

TASK_PROMPT = "<MORE_DETAILED_CAPTION>"


def main(args):
    assert args.load_archive == (
        args.metadata is not None
    ), "load_archive must be used with metadata / load_archiveはmetadataと一緒に使う必要があります"

    device = args.device if args.device is not None else device_utils.get_preferred_device()
    if type(device) is str:
        device = torch.device(device)
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    logger.info(f"device: {device}, dtype: {torch_dtype}")

    logger.info("Loading Florence-2-large model / Florence-2-largeモデルをロード中")

    support_flash_attn = False
    try:
        import flash_attn

        support_flash_attn = True
    except ImportError:
        pass

    if support_flash_attn:
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True
        ).to(device)
    else:
        logger.info(
            "flash_attn is not available. Trying to load without it / flash_attnが利用できません。flash_attnを使わずにロードを試みます"
        )

        # https://github.com/huggingface/transformers/issues/31793#issuecomment-2295797330
        # Removing the unnecessary flash_attn import which causes issues on CPU or MPS backends
        from transformers.dynamic_module_utils import get_imports
        from unittest.mock import patch

        def fixed_get_imports(filename) -> list[str]:
            if not str(filename).endswith("modeling_florence2.py"):
                return get_imports(filename)
            imports = get_imports(filename)
            imports.remove("flash_attn")
            return imports

        # workaround for unnecessary flash_attn requirement
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True
            ).to(device)

    model.eval()
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    # 画像を読み込む
    if not args.load_archive:
        train_data_dir_path = Path(args.train_data_dir)
        image_paths = train_util.glob_images_pathlib(train_data_dir_path, args.recursive)
        logger.info(f"found {len(image_paths)} images.")
    else:
        archive_files = glob.glob(os.path.join(args.train_data_dir, "*.zip")) + glob.glob(
            os.path.join(args.train_data_dir, "*.tar")
        )
        image_paths = [Path(archive_file) for archive_file in archive_files]

    # load metadata if needed
    if args.metadata is not None:
        metadata = tagger_utils.load_metadata(args.metadata)
        images_metadata = metadata["images"]
    else:
        images_metadata = metadata = None

    # define preprocess_image function
    def preprocess_image(image: Image.Image):
        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device, torch_dtype)
        return inputs

    # prepare DataLoader or something similar :)
    # Loader returns: list of (image_path, processed_image_or_something, image_size)
    if args.load_archive:
        loader = tagger_utils.ArchiveImageLoader([str(p) for p in image_paths], args.batch_size, preprocess_image, args.debug)
    else:
        # we cannot use DataLoader with ImageLoadingPrepDataset because processor is not pickleable
        loader = tagger_utils.ImageLoader(image_paths, args.batch_size, preprocess_image, args.debug)

    def run_batch(
        list_of_path_inputs_size: list[tuple[str, dict[str, torch.Tensor], tuple[int, int]]],
        images_metadata: Optional[dict[str, Any]],
        caption_index: Optional[int] = None,
    ):
        input_ids = torch.cat([inputs["input_ids"] for _, inputs, _ in list_of_path_inputs_size])
        pixel_values = torch.cat([inputs["pixel_values"] for _, inputs, _ in list_of_path_inputs_size])

        if args.debug:
            logger.info(f"input_ids: {input_ids.shape}, pixel_values: {pixel_values.shape}")
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
            )
        if args.debug:
            logger.info(f"generate done: {generated_ids.shape}")
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        if args.debug:
            logger.info(f"decode done: {len(generated_texts)}")

        for generated_text, (image_path, _, image_size) in zip(generated_texts, list_of_path_inputs_size):
            parsed_answer = processor.post_process_generation(generated_text, task=TASK_PROMPT, image_size=image_size)
            caption_text = parsed_answer["<MORE_DETAILED_CAPTION>"]

            caption_text = caption_text.strip().replace("<pad>", "")
            original_caption_text = caption_text

            if args.remove_mood:
                p = caption_text.find("The overall ")
                if p != -1:
                    caption_text = caption_text[:p].strip()

            caption_file = os.path.splitext(image_path)[0] + args.caption_extension

            if images_metadata is None:
                with open(caption_file, "wt", encoding="utf-8") as f:
                    f.write(caption_text + "\n")
            else:
                image_md = images_metadata.get(image_path, None)
                if image_md is None:
                    image_md = {"image_size": list(image_size)}
                    images_metadata[image_path] = image_md
                if "caption" not in image_md:
                    image_md["caption"] = []
                if caption_index is None:
                    image_md["caption"].append(caption_text)
                else:
                    while len(image_md["caption"]) <= caption_index:
                        image_md["caption"].append("")
                    image_md["caption"][caption_index] = caption_text

            if args.debug:
                logger.info("")
                logger.info(f"{image_path}:")
                logger.info(f"\tCaption: {caption_text}")
                if args.remove_mood and original_caption_text != caption_text:
                    logger.info(f"\tCaption (prior to removing mood): {original_caption_text}")

    for data_entry in tqdm(loader, smoothing=0.0):
        b_imgs = data_entry
        b_imgs = [(str(image_path), image, size) for image_path, image, size in b_imgs]  # Convert image_path to string
        run_batch(b_imgs, images_metadata, args.caption_index)

    if args.metadata is not None:
        logger.info(f"saving metadata file: {args.metadata}")
        with open(args.metadata, "wt", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("done!")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_dir", type=str, help="directory for train images / 学習画像データのディレクトリ")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size in inference / 推論時のバッチサイズ")
    parser.add_argument(
        "--caption_extension", type=str, default=".txt", help="extension of caption file / 出力されるキャプションファイルの拡張子"
    )
    parser.add_argument("--recursive", action="store_true", help="search images recursively / 画像を再帰的に検索する")
    parser.add_argument(
        "--remove_mood", action="store_true", help="remove mood from the caption / キャプションからムードを削除する"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="maximum number of tokens to generate. default is 1024 / 生成するトークンの最大数。デフォルトは1024",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=3,
        help="number of beams for beam search. default is 3 / ビームサーチのビーム数。デフォルトは3",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="device for model. default is None, which means using an appropriate device / モデルのデバイス。デフォルトはNoneで、適切なデバイスを使用する",
    )
    parser.add_argument(
        "--caption_index",
        type=int,
        default=None,
        help="index of the caption in the metadata file. default is None, which means adding caption to the existing captions. 0>= to replace the caption"
        " / メタデータファイル内のキャプションのインデックス。デフォルトはNoneで、新しく追加する。0以上でキャプションを置き換える",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    tagger_utils.add_archive_arguments(parser)

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
