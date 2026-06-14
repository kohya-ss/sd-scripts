"""FineTuning dataset: pre-computed metadata.json driven dataset with cached latents.

Extracted from ``library.train_util`` as part of the dataset-split refactor;
imports the abstract :class:`~library.dataset.BaseDataset` and its
:class:`~library.subset.FineTuningSubset` configuration class.
"""

import glob
import json
import logging
import os
from typing import Optional, Sequence, Tuple

from library.dataset import (
    BaseDataset,
    ImageInfo,
    glob_images,
)
from library.strategy_base import LatentsCachingStrategy
from library.subset import FineTuningSubset
from library.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        resolution,
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        train_inpainting: bool,
        debug_dataset: bool,
        validation_seed: int,
        validation_split: float,
        resize_interpolation: Optional[str],
        skip_image_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__(
            resolution,
            network_multiplier,
            train_inpainting,
            debug_dataset,
            resize_interpolation,
            skip_image_resolution,
        )

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します"
                )
                continue

            # メタデータを読み込む
            if os.path.exists(subset.metadata_file):
                if subset.metadata_file.endswith(".jsonl"):
                    logger.info(f"loading existing JSOL metadata: {subset.metadata_file}")
                    # optional JSONL format
                    # {"image_path": "/path/to/image1.jpg", "caption": "A caption for image1", "image_size": [width, height]}
                    metadata = {}
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            line_md = json.loads(line)
                            image_md = {"caption": line_md.get("caption", "")}
                            if "image_size" in line_md:
                                image_md["image_size"] = line_md["image_size"]
                            if "width" in line_md and "height" in line_md:
                                image_md["image_size"] = [line_md["width"], line_md["height"]]
                            if "tags" in line_md:
                                image_md["tags"] = line_md["tags"]
                            metadata[line_md["image_path"]] = image_md
                else:
                    # standard JSON format
                    logger.info(f"loading existing metadata: {subset.metadata_file}")
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        metadata = json.load(f)
            else:
                raise ValueError(f"no metadata / メタデータファイルがありません: {subset.metadata_file}")

            if len(metadata) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / 画像に関するデータが見つからないためサブセットを無視します"
                )
                continue

            # Add full path for image
            image_dirs = set()
            if subset.image_dir is not None:
                image_dirs.add(subset.image_dir)
            for image_key in metadata.keys():
                if not os.path.isabs(image_key):
                    assert (
                        subset.image_dir is not None
                    ), f"image_dir is required when image paths are relative / 画像パスが相対パスの場合、image_dirの指定が必要です: {image_key}"
                    abs_path = os.path.join(subset.image_dir, image_key)
                else:
                    abs_path = image_key
                    image_dirs.add(os.path.dirname(abs_path))

                # if image_key does not have extension, try to find image file with supported extensions
                if not os.path.splitext(image_key)[1] or not os.path.exists(abs_path):  # no extension or file does not exist
                    paths = glob_images(os.path.dirname(abs_path), os.path.basename(image_key))
                    if len(paths) > 0:
                        abs_path = paths[0]
                    # If no file is found, we use *.npz file to get image size and for training

                metadata[image_key]["abs_path"] = abs_path

            # Enumerate existing npz files
            strategy = LatentsCachingStrategy.get_strategy()
            npz_paths = []
            if strategy is not None:    # If `cache_latents` is not enabled (and no strategy is set), skip this part
                for image_dir in image_dirs:
                    npz_paths.extend(glob.glob(os.path.join(image_dir, "*" + strategy.cache_suffix)))
            npz_paths = sorted(npz_paths, key=lambda item: len(os.path.basename(item)), reverse=True)  # longer paths first

            # Match image filename longer to shorter because some images share same prefix
            image_keys_sorted_by_length_desc = sorted(metadata.keys(), key=len, reverse=True)

            # Collect tags and sizes
            tags_list = []
            size_set_from_metadata = 0
            size_set_from_cache_filename = 0
            num_filtered = 0
            for image_key in image_keys_sorted_by_length_desc:
                img_md = metadata[image_key]
                caption = img_md.get("caption")
                tags = img_md.get("tags")
                image_size = img_md.get("image_size")
                abs_path = img_md.get("abs_path")

                # search npz if image_size is not given
                npz_path = None
                if image_size is None:
                    # match against the resolved absolute path and normalize separators, so that metadata
                    # paths written with "/" still match glob results using the OS-native separator
                    abs_path_without_ext = os.path.normpath(os.path.splitext(abs_path)[0])
                    for candidate in npz_paths:
                        if os.path.normpath(candidate).startswith(abs_path_without_ext):
                            npz_path = candidate
                            break
                    if npz_path is not None:
                        npz_paths.remove(npz_path)  # remove to avoid matching same file (share prefix)
                        abs_path = npz_path

                if caption is None:
                    caption = ""

                if subset.enable_wildcard:
                    # tags must be single line (split by caption separator)
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if tags is not None:
                        caption = "\n".join(
                            [f"{line}{subset.caption_separator}{tags}" for line in caption.split("\n") if line.strip() != ""]
                        )
                        tags_list.append(tags)
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        if len(caption) > 0:
                            caption = caption + subset.caption_separator
                        caption = caption + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path, subset.caption_dropout_rate)
                image_info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )

                if image_size is not None:
                    image_info.image_size = tuple(image_size)  # width, height
                    size_set_from_metadata += 1
                elif npz_path is not None:
                    # get image size from npz filename
                    w, h = strategy.get_image_size_from_disk_cache_path(abs_path, npz_path)
                    image_info.image_size = (w, h)
                    size_set_from_cache_filename += 1

                    # use the discovered cache file directly for latent caching/loading
                    image_info.latents_npz = npz_path

                if self.skip_image_resolution is not None:
                    size = image_info.image_size
                    if size is None:  # no image size in metadata or latents cache file, get image size by reading image file (slow)
                        size = self.get_image_size(abs_path)
                        image_info.image_size = size
                    skip_image_area = self.skip_image_resolution[0] * self.skip_image_resolution[1]
                    if size[0] * size[1] <= skip_image_area:
                        num_filtered += 1
                        continue

                self.register_image(image_info, subset)

            if size_set_from_cache_filename > 0:
                logger.info(
                    f"set image size from cache files: {size_set_from_cache_filename}/{len(image_keys_sorted_by_length_desc)}"
                )
            if size_set_from_metadata > 0:
                logger.info(f"set image size from metadata: {size_set_from_metadata}/{len(image_keys_sorted_by_length_desc)}")
            if num_filtered > 0:
                logger.info(f"filtered {num_filtered} images by original resolution from {subset.metadata_file}")
            self.num_train_images += len(metadata) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata)
            self.subsets.append(subset)
