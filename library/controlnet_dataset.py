"""ControlNet dataset: paired training images and conditioning (control) images.

Extracted from ``library.train_util`` as part of the dataset-split refactor;
imports the abstract :class:`~library.dataset.BaseDataset` and its
:class:`~library.subset.ControlNetSubset` configuration class.
"""

import hashlib
import logging
import os
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from tqdm import tqdm

from library.dataset import (
    IMAGE_TRANSFORMS,
    BaseDataset,
    ImageInfo,
    glob_images,
    load_image,
)
from library.dreambooth_dataset import DreamBoothDataset
from library.strategy_base import LatentsCachingStrategy
from library.subset import ControlNetSubset, DreamBoothSubset
from library.utils import resize_image, setup_logging, trim_and_resize_if_required

setup_logging()
logger = logging.getLogger(__name__)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
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
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
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

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_cropはControlNetDatasetではサポートされていません"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
                enable_multiline_caption=subset.enable_multiline_caption,
                multiline_caption_separator=subset.multiline_caption_separator,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            1.0,
            train_inpainting,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
            skip_image_resolution,
        )

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.has_multiple_caption_candidates = self.dreambooth_dataset_delegate.has_multiple_caption_candidates
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / 制御用画像が見つかりませんでした: {missing_imgs}"
        if len(extra_imgs) > 0:
            logger.warning(f"extra conditioning data for {len(extra_imgs)} images / 余分な制御用画像があります: {extra_imgs}")

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

        # conditioning latents cache (reference-conditioned training): None (serve pixels), "memory" or "disk"
        self.cond_latents_cache_mode: Optional[str] = None
        self.cond_latents_caching_strategy: Optional[LatentsCachingStrategy] = None
        # gray level (0-255) to fill the alpha region (alpha >= 128, e.g. the inpainting hole) of RGBA conditioning images with,
        # set by the trainer. None: the alpha channel is dropped
        self.cond_alpha_fill: Optional[int] = None
        self._warned_cond_alpha_missing = False

    def load_cond_image(self, path: str) -> np.ndarray:
        """Load a conditioning image as RGB uint8 [H, W, 3]. With `cond_alpha_fill`, the alpha region of an RGBA image is
        filled with the gray level (255 = fill, as the inpainting masks of the LLLite trainer)."""
        if self.cond_alpha_fill is None:
            return load_image(path)
        img = load_image(path, alpha=True)
        alpha = img[:, :, 3:4]
        if not self._warned_cond_alpha_missing and np.all(alpha == 255):
            logger.warning(
                f"conditioning image has no alpha channel (or the alpha is all 255): the whole image is filled with the gray level."
                f" This warning is shown once / 条件画像にアルファチャンネルがないか全て 255 です。画像全体がグレーで塗られます: {path}"
            )
            self._warned_cond_alpha_missing = True
        fill = np.full_like(img[:, :, :3], self.cond_alpha_fill)
        return np.where(alpha >= 128, fill, img[:, :, :3])

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def get_cond_latents_npz_path(self, caching_strategy: LatentsCachingStrategy, info: ImageInfo, cache_dir: Optional[str]) -> str:
        """npz path of the cached conditioning latents: next to the conditioning image by default, or in `cache_dir` with a
        short hash of the conditioning image path as prefix (conditioning directories of different subsets may share file names)."""
        npz_path = caching_strategy.get_latents_npz_path(info.cond_img_path, info.image_size)
        if cache_dir is None:
            return npz_path
        digest = hashlib.sha1(os.path.normcase(info.cond_img_path).encode("utf-8")).hexdigest()[:8]
        return os.path.join(cache_dir, f"{digest}_{os.path.basename(npz_path)}")

    def new_cache_conditioning_latents(
        self, model: Any, accelerator: Accelerator, caching_strategy: LatentsCachingStrategy, cache_dir: Optional[str] = None
    ):
        r"""
        Cache the VAE latents of the conditioning images (for trainers which use them as reference images), with the same
        VAE / npz handling as the latents of the training images: `caching_strategy` decides memory or disk and does the
        encoding. The conditioning image gets the same resize / center crop as its training image (random_crop is not
        supported by this dataset, so the result is deterministic), and the flipped latents are cached too when flip_aug
        is enabled. After this call `__getitem__` returns `conditioning_latents` [B, C, h, w] instead of `conditioning_images`.
        """
        logger.info("caching conditioning (reference) latents" + ("" if cache_dir is None else f" to {cache_dir}"))
        self.cond_latents_cache_mode = "disk" if caching_strategy.cache_to_disk else "memory"
        self.cond_latents_caching_strategy = caching_strategy
        if cache_dir is not None and caching_strategy.cache_to_disk:
            os.makedirs(cache_dir, exist_ok=True)

        image_infos = list(self.image_data.values())
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        def make_proxy(info: ImageInfo) -> ImageInfo:
            # a proxy ImageInfo of the conditioning image so that the caching strategy can be reused as is: same
            # bucket / resized size as the training image (the sizes are checked in __getitem__ for the pixel path)
            proxy = ImageInfo(info.image_key, 1, "", False, info.cond_img_path)
            proxy.image_size = info.image_size
            proxy.resized_size = info.resized_size
            proxy.bucket_reso = info.bucket_reso
            proxy.resize_interpolation = info.resize_interpolation
            proxy.latents_npz = info.cond_latents_npz
            cond_img = self.load_cond_image(info.cond_img_path)
            if cond_img.shape[0] != info.image_size[1] or cond_img.shape[1] != info.image_size[0]:
                if self.dreambooth_dataset_delegate.enable_bucket:
                    raise ValueError(
                        f"size of conditioning image is not match / 画像サイズが合いません: {info.cond_img_path} {cond_img.shape[1]}x{cond_img.shape[0]}"
                        f" vs {info.absolute_path} {info.image_size[0]}x{info.image_size[1]}"
                    )
                cond_img = resize_image(
                    cond_img, cond_img.shape[0], cond_img.shape[1], info.image_size[0], info.image_size[1], self.resize_interpolation
                )
            proxy.image = cond_img
            return proxy

        def submit_batch(batch: List[Tuple[ImageInfo, ImageInfo]], flip_aug: bool):
            proxies = [proxy for _, proxy in batch]
            caching_strategy.cache_batch_latents(model, proxies, flip_aug, False, False)
            for info, proxy in batch:
                proxy.image = None
                if self.cond_latents_cache_mode == "memory":
                    info.cond_latents = proxy.latents
                    info.cond_latents_flipped = proxy.latents_flipped

        batch: List[Tuple[ImageInfo, ImageInfo]] = []
        current_condition = None  # (bucket_reso, flip_aug)
        for i, info in enumerate(tqdm(image_infos)):
            subset = self.dreambooth_dataset_delegate.image_to_subset[info.image_key]

            if caching_strategy.cache_to_disk:
                info.cond_latents_npz = self.get_cond_latents_npz_path(caching_strategy, info, cache_dir)
                if i % num_processes != process_index:  # each process caches different latents
                    continue
                if caching_strategy.is_disk_cached_latents_expected(info.bucket_reso, info.cond_latents_npz, subset.flip_aug, False):
                    continue

            condition = (info.bucket_reso, subset.flip_aug)
            if len(batch) > 0 and current_condition != condition:
                submit_batch(batch, current_condition[1])
                batch = []

            batch.append((info, make_proxy(info)))
            current_condition = condition

            if len(batch) >= caching_strategy.batch_size:
                submit_batch(batch, current_condition[1])
                batch = []

        if len(batch) > 0:
            submit_batch(batch, current_condition[1])

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        if self.cond_latents_cache_mode is not None:
            example["conditioning_latents"] = self._get_cached_conditioning_latents(bucket, image_index, bucket_batch_size, example)
            return example

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = self.load_cond_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path}"

                cond_img, _, _ = trim_and_resize_if_required(
                    False,  # TODO support random crop
                    cond_img,
                    image_info.bucket_reso,
                    image_info.resized_size,
                    resize_interpolation=image_info.resize_interpolation,
                )
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(
                        cond_img,
                        cond_img.shape[0],
                        cond_img.shape[1],
                        target_size_hw[1],
                        target_size_hw[0],
                        self.resize_interpolation,
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example

    def _get_cached_conditioning_latents(self, bucket, image_index: int, bucket_batch_size: int, example) -> torch.Tensor:
        conditioning_latents = []
        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]
            flipped = example["flippeds"][i]

            if self.cond_latents_cache_mode == "memory":
                latents = image_info.cond_latents_flipped if flipped else image_info.cond_latents
                assert latents is not None, (
                    f"conditioning latents are not cached / 条件画像のlatentがキャッシュされていません: {image_info.cond_img_path}"
                )
            else:
                latents, _, _, flipped_latents, _ = self.cond_latents_caching_strategy.load_latents_from_disk(
                    image_info.cond_latents_npz, image_info.bucket_reso
                )
                if flipped:
                    assert flipped_latents is not None, (
                        f"flipped conditioning latents are not cached / 反転した条件画像のlatentがキャッシュされていません: {image_info.cond_latents_npz}"
                    )
                    latents = flipped_latents
                latents = torch.FloatTensor(latents)
            conditioning_latents.append(latents)

        return torch.stack(conditioning_latents).to(memory_format=torch.contiguous_format).float()
