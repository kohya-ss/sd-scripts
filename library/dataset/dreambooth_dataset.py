import pathlib
from .base_datasets import BaseDataset
from typing import Union, List
from .common import ImageInfo, KohyaDatasetException, IMAGE_EXTENSIONS, with_stem


class DreamBoothDataset(BaseDataset):
    def __init__(
        self,
        batch_size,
        root_train_data_dir,
        reg_data_dir,
        tokenizer,
        max_token_length,
        caption_extension,
        shuffle_caption,
        shuffle_keep_tokens,
        resolution,
        enable_bucket,
        min_bucket_reso,
        max_bucket_reso,
        bucket_reso_steps,
        bucket_no_upscale,
        prior_loss_weight,
        flip_aug,
        color_aug,
        face_crop_aug_range,
        random_crop,
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

        if resolution is None or resolution == 0:
            raise KohyaDatasetException(
                "Resolution parameter is missing / resolution（解像度）指定は必須です"
            )

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.caption_extension = caption_extension

        self.num_reg_images = 0
        self.num_train_images = 0

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            if min(resolution) < min_bucket_reso:
                raise KohyaDatasetException(
                    f"min_bucket_reso must be either equal or lesser than defined resolution\n"
                    f"min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください\n"
                    f"Min res: {min(resolution)} | bucket size: {min_bucket_reso}"
                )
            elif max(resolution) > max_bucket_reso:
                raise KohyaDatasetException(
                    "max_bucket_reso must be either equal or greater than defined resolution\n"
                    "max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"
                    f"Max res: {max(resolution)} | bucket size: {max_bucket_reso}"
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

        if isinstance(root_train_data_dir, str):
            root_train_data_dir = pathlib.Path(root_train_data_dir)
            if not root_train_data_dir.resolve().exists():
                raise KohyaDatasetException(
                    "Root Train data folder does not exist.\n"
                    "root_train_data_dir フォルダーが存在しません。"
                    # TODO: Kohya, if possible, please translate this back to Japanese. Thanks!
                )

        print("Preparing train images...")

        train_counts = 0
        for concept_dir in root_train_data_dir.iterdir():
            if concept_dir.is_dir():
                repeats, image_paths, captions = self.load_dreambooth_dir(concept_dir)
                train_counts += repeats * len(image_paths)
                for image_pth, caption in zip(image_paths, captions):
                    image_info = ImageInfo(
                        image_pth.name, repeats, caption, False, str(image_pth)
                    )
                    self.register_image(image_info)
                self.dataset_dirs_info[concept_dir.name] = {
                    "n_repeats": repeats,
                    "img_count": len(image_paths),
                }

        print(f"{train_counts} train images with repeats.")
        self.num_train_images = train_counts

        # reg imageは数を数えて学習画像と同じ枚数にする
        print("Preparing reg images.")

        reg_infos: List[ImageInfo] = []
        if reg_data_dir:
            if isinstance(root_train_data_dir, str):
                reg_data_dir = pathlib.Path(reg_data_dir)
                if not reg_data_dir.resolve().exists():
                    raise KohyaDatasetException(
                        "Root reg data folder does not exist.\n"
                        "reg_data_dir フォルダーが存在しません。"
                        # TODO: Kohya, if possible, please translate this back to Japanese. Thanks!
                    )
            for reg_dir in reg_data_dir.iterdir():
                if reg_dir.is_dir():
                    repeats, image_paths, captions = self.load_dreambooth_dir(reg_dir)
                    train_counts += repeats * len(image_paths)
                    for image_pth, caption in zip(image_paths, captions):
                        image_info = ImageInfo(
                            image_pth.name, repeats, caption, False, str(image_pth)
                        )
                        reg_infos.append(image_info)
                    self.reg_dataset_dirs_info[reg_dir.name] = {
                        "n_repeats": repeats,
                        "img_count": len(image_paths),
                    }

        print(f"{self.num_reg_images} reg images.")
        if self.num_train_images < self.num_reg_images:
            print("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります")

        if self.num_reg_images == 0:
            print("no regularization images / 正則化画像が見つかりませんでした")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < self.num_reg_images:
                for info in reg_infos:
                    if first_loop:
                        self.register_image(info)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1
                        n += 1
                    if n >= self.num_train_images:
                        break
            first_loop = False

    def read_caption(self, img_path: pathlib.Path, new_method=False):
        """Finds and reads a caption file, given the image path.

        Args:
            img_path (pathlib.Path): Image path.
            new_method (bool, optional): Tries a new method which is the about the same as the old method.
            Defaults to False.

        Raises:
            UnicodeDecodeError: A file was invalid.
            KohyaDatasetException: A caption file was empty.

        Returns:
            _type_: _description_
        """
        # captionの候補ファイル名を作る

        if not new_method:
            base_name = img_path.stem
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])
            cap_paths = [
                img_path.with_suffix(self.caption_extension),
                with_stem(img_path, base_name_face_det).with_suffix(
                    self.caption_extension),
            ]
            for cap_path in cap_paths:
                if cap_path.is_file():
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            print(
                                f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path}"
                            )
                            raise e
                        if not lines:
                            raise KohyaDatasetException(
                                f"caption file is empty\n"
                                f"キャプションファイルが空です\n"
                                f"{cap_path}"
                            )
                        return lines[0].strip()
        else:
            caption_ext = self.caption_extension
            if not self.caption_extension.startswith("."):
                caption_ext = "." + caption_ext
            for file in img_path.parent.glob(f"{img_path.stem}{caption_ext}"):
                if file.is_file():
                    try:
                        captions = file.read_text(encoding="utf-8")
                    except UnicodeDecodeError as e:
                        print(
                            f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {file}"
                        )
                        raise e
                    return captions.split()[0].strip()

    def load_dreambooth_dir(self, booth_dir: pathlib.Path):
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

        # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
        captions = []
        for img_path in img_paths:
            cap_for_img = self.read_caption(img_path)
            captions.append(folder_caption if cap_for_img is None else cap_for_img)

        self.set_tag_frequency(booth_dir.name, captions)  # タグ頻度を記録

        return n_repeats, img_paths, captions
