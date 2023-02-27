import pathlib
from .base_datasets import BaseDataset
from .buckets import BucketManager
from typing import Union, List
from .common import ImageInfo, KohyaDatasetException, IMAGE_EXTENSIONS

try:
    import orjson as json
except ImportError:
    import json

class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        json_file_name,
        batch_size,
        train_data_dir,
        tokenizer,
        max_token_length,
        shuffle_caption,
        shuffle_keep_tokens,
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
        dataset_repeats,
        debug_dataset,
    ) -> None:
        super().__init__(
            tokenizer,
            max_token_length,
            shuffle_caption,
            shuffle_keep_tokens,
            resolution,
            flip_aug,
            color_aug,
            face_crop_aug_range,
            random_crop,
            debug_dataset,
        )

        # メタデータを読み込む
        json_file = pathlib.Path(json_file_name).resolve()
        if json_file.exists() and json_file.is_file():
            print(f"loading existing metadata: {json_file_name}")
            metadata = json.loads(json_file.read_text())
        else:
            raise KohyaDatasetException(f"Metadata missing / メタデータファイルがありません: {json_file}")

        self.metadata = metadata
        self.train_data_dir = pathlib.Path(train_data_dir)
        self.batch_size = batch_size

        tags_list = []
        for image_key, img_md in metadata.items():
            # path情報を作る
            image_key = pathlib.Path(image_key)
            if image_key.exists():
                abs_path = image_key
            else:
                # わりといい加減だがいい方法が思いつかん
                abs_path = self.train_data_dir / image_key
                if not abs_path.exists():
                    raise KohyaDatasetException(f"Image file: {abs_path} does not exist.")

            caption = img_md.get("caption")
            tags = img_md.get("tags")
            if caption is None:
                caption = tags
            elif tags is not None and len(tags) > 0:
                caption = caption + ", " + tags
                tags_list.append(tags)
            if caption is None and not len(caption):
                raise KohyaDatasetException(f"caption or tag is required\n"
                                            f"キャプションまたはタグは必須です\n"
                                            f"{abs_path}"
                                            )


            image_info = ImageInfo(str(image_key.name), dataset_repeats, caption, False, str(abs_path))
            image_info.image_size = img_md.get("train_resolution")

            if not self.color_aug and not self.random_crop:
                # if npz exists, use them
                latents_npz, latents_npz_flipped = self.image_key_to_npz_file(image_key)
                if latents_npz:
                    image_info.latents_npz = str(latents_npz)
                if latents_npz_flipped:
                    image_info.latents_npz_flipped = str(latents_npz_flipped)

            self.register_image(image_info)
        self.num_train_images = len(metadata) * dataset_repeats
        self.num_reg_images = 0

        # TODO do not record tag freq when no tag
        self.set_tag_frequency(json_file.name, tags_list)
        self.dataset_dirs_info[json_file.name] = {
            "n_repeats": dataset_repeats,
            "img_count": len(metadata),
        }

        # check existence of all npz files
        use_npz_latents = not (self.color_aug or self.random_crop)
        if use_npz_latents:
            npz_any = False
            npz_all = True
            for image_info in self.image_data.values():
                has_npz = image_info.latents_npz is not None
                npz_any = npz_any or has_npz

                if self.flip_aug:
                    has_npz = has_npz and image_info.latents_npz_flipped is not None
                npz_all = npz_all and has_npz

                if npz_any and not npz_all:
                    break

            if not npz_any:
                use_npz_latents = False
                print(
                    f"npz file does not exist. ignore npz files / npzファイルが見つからないためnpzファイルを無視します"
                )
            elif not npz_all:
                use_npz_latents = False
                print(
                    f"some of npz file does not exist. ignore npz files / いくつかのnpzファイルが見つからないためnpzファイルを無視します"
                )
                if self.flip_aug:
                    print("maybe no flipped files / 反転されたnpzファイルがないのかもしれません")
        # else:
        #   print("npz files are not used with color_aug and/or random_crop / color_augまたはrandom_cropが指定されているためnpzファイルは使用されません")

        # check min/max bucket size
        sizes = set()
        resos = set()
        for image_info in self.image_data.values():
            if image_info.image_size is None:
                sizes = None  # not calculated
                break
            sizes.add(image_info.image_size[0])
            sizes.add(image_info.image_size[1])
            resos.add(tuple(image_info.image_size))

        if sizes is None:
            if use_npz_latents:
                use_npz_latents = False
                print(
                    f"npz files exist, but no bucket info in metadata. ignore npz files / メタデータにbucket情報がないためnpzファイルを無視します"
                )

            assert (
                resolution is not None
            ), "if metadata doesn't have bucket info, resolution is required / メタデータにbucket情報がない場合はresolutionを指定してください"

            self.enable_bucket = enable_bucket
            if self.enable_bucket:
                self.min_bucket_reso = min_bucket_reso
                self.max_bucket_reso = max_bucket_reso
                self.bucket_reso_steps = bucket_reso_steps
                self.bucket_no_upscale = bucket_no_upscale
        else:
            if not enable_bucket:
                print(
                    "metadata has bucket info, enable bucketing / メタデータにbucket情報があるためbucketを有効にします"
                )
            print("using bucket info in metadata / メタデータ内のbucket情報を使います")
            self.enable_bucket = True

            assert (
                not bucket_no_upscale
            ), "if metadata has bucket info, bucket reso is precalculated, so bucket_no_upscale cannot be used / メタデータ内にbucket情報がある場合はbucketの解像度は計算済みのため、bucket_no_upscaleは使えません"

            # bucket情報を初期化しておく、make_bucketsで再作成しない
            self.bucket_manager = BucketManager(False, None, None, None, None)
            self.bucket_manager.set_predefined_resos(resos)

        # npz情報をきれいにしておく
        if not use_npz_latents:
            for image_info in self.image_data.values():
                image_info.latents_npz = image_info.latents_npz_flipped = None

    def image_key_to_npz_file(self, image_key: pathlib.Path):

        npz_file_norm = image_key.with_suffix(".npz")

        if npz_file_norm.exists():
            # image_key is full path
            npz_file_flip = npz_file_norm.with_stem(npz_file_norm.stem + "_npz")
            if not npz_file_flip.exists():
                npz_file_flip = None
            return npz_file_norm, npz_file_flip

        # image_key is relative path
        npz_file_norm = self.train_data_dir / pathlib.Path(image_key.with_suffix(".npz"))
        npz_file_flip = npz_file_norm.with_stem(npz_file_norm.stem + "_npz")

        if npz_file_norm.exists():
            # image_key is full path
            npz_file_flip = npz_file_norm.with_stem(npz_file_norm.stem + "_npz")
            if not npz_file_flip.exists():
                npz_file_flip = None
            return npz_file_norm, npz_file_flip
        else:
            return None, None
