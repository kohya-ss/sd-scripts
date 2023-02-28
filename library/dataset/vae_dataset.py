import pathlib
import random
from typing import List

import torch

from .base_datasets import BaseDataset
from .common import ImageInfo, KohyaDatasetException, IMAGE_EXTENSIONS


class VAEDataset(BaseDataset):
    def __init__(
        self,
        batch_size,
        root_train_data_dir,
        resolution,
        enable_bucket,
        min_bucket_reso,
        max_bucket_reso,
        bucket_reso_steps,
        bucket_no_upscale,
        flip_aug,
        color_aug,
        face_crop_aug_range,
        random_crop,
        debug_dataset,
        shuffling,
        gradient_accumulation_steps,
    ) -> None:
        """
        Training dataset for VAE.

        From my understanding of the code, there is no tokenization involved.

        :param batch_size:
        :param root_train_data_dir:
        :param resolution:
        :param enable_bucket:
        :param min_bucket_reso:
        :param max_bucket_reso:
        :param bucket_reso_steps:
        :param bucket_no_upscale:
        :param flip_aug:
        :param color_aug:
        :param face_crop_aug_range:
        :param random_crop:
        :param debug_dataset:
        :param shuffling:
        """
        super().__init__(
            None,
            255,
            False,
            False,
            resolution,
            flip_aug,
            color_aug,
            face_crop_aug_range,
            random_crop,
            debug_dataset,
        )

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.num_train_images = 0
        self.shuffling: bool = shuffling
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.enable_bucket = enable_bucket

        if self.enable_bucket:
            if min(resolution) >= min_bucket_reso:
                raise KohyaDatasetException(
                    "min_bucket_reso must be either equal or lesser than defined resolution\n"
                    "min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
                )
            elif max(resolution) <= max_bucket_reso:
                raise KohyaDatasetException(
                    "max_bucket_reso must be either equal or greater than defined resolution\n"
                    "max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
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

        print("Preparing train images...")

        # Prepare files
        train_counts = 0
        for concept_dir in root_train_data_dir.iterdir():
            if concept_dir.is_dir():
                repeats, image_paths = self.load_vae_dir(concept_dir)
                train_counts += repeats * len(image_paths)
                for image_pth in image_paths:
                    image_info = ImageInfo(
                        image_pth.name, repeats, "", False, str(image_pth)
                    )
                    self.register_image(image_info)
                self.dataset_dirs_info[concept_dir.name] = {
                    "n_repeats": repeats,
                    "img_count": len(image_paths),
                }

    def load_vae_dir(self, booth_dir: pathlib.Path):
        if not booth_dir.exists() and not booth_dir.is_dir():
            # print(f"ignore file: {dir}")
            return 0, [], []

        tokens = booth_dir.name.split("_")
        try:
            n_repeats = int(tokens[0])
        except ValueError as e:
            print(
                f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {booth_dir}"
            )
            return 0, [], []

        folder_caption = "_".join(tokens[1:])

        img_paths: List[pathlib.Path] = []
        for file in booth_dir.iterdir():
            if file.suffix.lower() in IMAGE_EXTENSIONS:
                img_paths.append(file)

        print(
            f"found directory {n_repeats}_{folder_caption} contains {len(img_paths)} image files"
        )

        return n_repeats, img_paths

    def make_buckets(self):
        super(VAEDataset, self).make_buckets()
        self._data_len_add = self.gradient_accumulation_steps - (
            self._data_len % self.gradient_accumulation_steps
        )

    def __len__(self):
        return self._data_len + self._data_len_add

    def __getitem__(self, index):
        if index == 0 and self.shuffling:
            self.shuffle_buckets()

        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size
        latents_list = []
        images = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]

            # image/latentsを処理する
            if image_info.latents is not None:
                latents = (
                    image_info.latents
                    if not self.flip_aug or random.random() < 0.5
                    else image_info.latents_flipped
                )
                image = None
            elif image_info.latents_npz is not None:
                latents = self.load_latents_from_npz(
                    image_info, self.flip_aug and random.random() >= 0.5
                )
                latents = torch.FloatTensor(latents)
                image = None
            else:
                raise KohyaDatasetException(
                    "Latents are required to be cached for VAE dataset! "
                    "Missing Latents."
                )

            images.append(image)
            latents_list.append(latents)

        example = {}

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["latents"] = (
            torch.stack(latents_list) if latents_list[0] is not None else None
        )

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example
