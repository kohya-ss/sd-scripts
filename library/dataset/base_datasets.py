# Code for BaseDataset Class

from typing import Dict, List, Optional
import math
import os
import random

from tqdm import tqdm
import torch
import torch.utils.data
from torchvision import transforms
from transformers import CLIPTokenizer
import albumentations as albu
import numpy as np
from PIL import Image
from .buckets import BucketManager, BucketBatchIndex
from .common import ImageInfo, KohyaDatasetException
import cv2


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        max_token_length,
        shuffle_caption,
        shuffle_keep_tokens,
        resolution,
        flip_aug: bool,
        color_aug: bool,
        face_crop_aug_range,
        random_crop,
        debug_dataset: bool,
    ) -> None:
        super().__init__()
        self.buckets_indices = None
        self.tokenizer: CLIPTokenizer = tokenizer
        self.max_token_length = max_token_length
        self.shuffle_caption = shuffle_caption
        self.shuffle_keep_tokens = shuffle_keep_tokens
        # width/height is used when enable_bucket==False
        self.width: int
        self.height: int
        if not resolution:
            raise KohyaDatasetException("Missing Resolution!")
        else:
            self.width = resolution[0]
            self.height = resolution[1]
        self.face_crop_aug_range = face_crop_aug_range
        self.flip_aug = flip_aug
        self.color_aug = color_aug
        self.debug_dataset = debug_dataset
        self.random_crop = random_crop
        self.token_padding_disabled = False
        self.dataset_dirs_info = {}
        self.reg_dataset_dirs_info = {}
        self.tag_frequency = {}

        self.enable_bucket = False
        self.bucket_manager: BucketManager  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.tokenizer_max_length = (
            self.tokenizer.model_max_length
            if max_token_length is None
            else max_token_length + 2
        )

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ
        self.dropout_rate: float = 0
        self.dropout_every_n_epochs: int
        self.tag_dropout_rate: float = 0

        # augmentation
        flip_p = 0.5 if flip_aug else 0.0
        if color_aug:
            # わりと弱めの色合いaugmentation：brightness/contrastあたりは画像のpixel valueの最大値・最小値を変えてしまうのでよくないのではという想定でgamma/hueあたりを触る
            self.aug = albu.Compose(
                [
                    albu.OneOf(
                        [
                            albu.HueSaturationValue(8, 0, 0, p=0.5),
                            albu.RandomGamma((95, 105), p=0.5),
                        ],
                        p=0.33,
                    ),
                    albu.HorizontalFlip(p=flip_p),
                ],
                p=1.0,
            )
        elif flip_aug:
            self.aug = albu.Compose([albu.HorizontalFlip(p=flip_p)], p=1.0)
        else:
            self.aug = None

        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.image_data: Dict[str, ImageInfo] = {}

        self.replacements = {}

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def set_caption_dropout(
        self, dropout_rate, dropout_every_n_epochs, tag_dropout_rate
    ):
        # コンストラクタで渡さないのはTextual Inversionで意識したくないから（ということにしておく）
        self.dropout_rate = dropout_rate
        self.dropout_every_n_epochs = dropout_every_n_epochs
        self.tag_dropout_rate = tag_dropout_rate

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                if tag and not tag.isspace():
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, caption):
        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = self.dropout_rate > 0 and random.random() < self.dropout_rate
        is_drop_out = (
            is_drop_out
            or self.dropout_every_n_epochs
            and self.current_epoch % self.dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            if self.shuffle_caption or self.tag_dropout_rate > 0:

                def dropout_tags(tokens):
                    if self.tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= self.tag_dropout_rate:
                            l.append(token)
                    return l

                tokens = [t.strip() for t in caption.strip().split(",")]
                if self.shuffle_keep_tokens is None:
                    if self.shuffle_caption:
                        random.shuffle(tokens)

                    tokens = dropout_tags(tokens)
                else:
                    if len(tokens) > self.shuffle_keep_tokens:
                        keep_tokens = tokens[: self.shuffle_keep_tokens]
                        tokens = tokens[self.shuffle_keep_tokens :]

                        if self.shuffle_caption:
                            random.shuffle(tokens)

                        tokens = dropout_tags(tokens)

                        tokens = keep_tokens + tokens
                caption = ", ".join(tokens)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption):
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors="pt",
        ).input_ids

        if self.tokenizer_max_length > self.tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1,
                    self.tokenizer_max_length - self.tokenizer.model_max_length + 2,
                    self.tokenizer.model_max_length - 2,
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + self.tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(
                    1,
                    self.tokenizer_max_length - self.tokenizer.model_max_length + 2,
                    self.tokenizer.model_max_length - 2,
                ):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + self.tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if (
                        ids_chunk[-2] != self.tokenizer.eos_token_id
                        and ids_chunk[-2] != self.tokenizer.pad_token_id
                    ):
                        ids_chunk[-1] = self.tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == self.tokenizer.pad_token_id:
                        ids_chunk[1] = self.tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def register_image(self, info: ImageInfo):
        self.image_data[info.image_key] = info

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        print("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        if self.enable_bucket:
            print("make buckets")
        else:
            print("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        img_ar_errors = []
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                )
                if not self.bucket_no_upscale:
                    self.bucket_manager.make_buckets()
                else:
                    print(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            for image_info in self.image_data.values():
                if not image_info.image_size:
                    raise KohyaDatasetException(
                        f"image_size missing for {image_info.absolute_path}. Is the file valid?"
                    )
                image_width, image_height = image_info.image_size
                (
                    image_info.bucket_reso,
                    image_info.resized_size,
                    ar_error,
                ) = self.bucket_manager.select_bucket(image_width, image_height)

                # print(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(
                False, (self.width, self.height), None, None, None
            )
            self.bucket_manager.set_predefined_resos(
                [(self.width, self.height)]
            )  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                if not image_info.image_size:
                    raise KohyaDatasetException(
                        f"image_size missing for {image_info.absolute_path}. Is the file valid?"
                    )
                image_width, image_height = image_info.image_size
                (
                    image_info.bucket_reso,
                    image_info.resized_size,
                    _,
                ) = self.bucket_manager.select_bucket(image_width, image_height)

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(
                    image_info.bucket_reso, image_info.image_key
                )

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            print("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")
            for i, (reso, bucket) in enumerate(
                zip(self.bucket_manager.resos, self.bucket_manager.buckets)
            ):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {
                        "resolution": reso,
                        "count": len(bucket),
                    }
                    print(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            img_ar_errors = np.array(img_ar_errors)
            mean_img_ar_error = np.mean(np.abs(img_ar_errors))

            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error  # type: ignore
            print(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(
                    BucketBatchIndex(bucket_index, self.batch_size, batch_index)
                )

            # ↓以下はbucketごとのbatch件数があまりにも増えて混乱を招くので元に戻す
            # 　学習時はステップ数がランダムなので、同一画像が同一batch内にあってもそれほど悪影響はないであろう、と考えられる
            #
            # # bucketが細分化されることにより、ひとつのbucketに一種類の画像のみというケースが増え、つまりそれは
            # # ひとつのbatchが同じ画像で占められることになるので、さすがに良くないであろう
            # # そのためバッチサイズを画像種類までに制限する
            # # ただそれでも同一画像が同一バッチに含まれる可能性はあるので、繰り返し回数が少ないほうがshuffleの品質は良くなることは間違いない？
            # # TO DO 正則化画像をepochまたがりで利用する仕組み
            # num_of_image_types = len(set(bucket))
            # bucket_batch_size = min(self.batch_size, num_of_image_types)
            # batch_count = int(math.ceil(len(bucket) / bucket_batch_size))
            # # print(bucket_index, num_of_image_types, bucket_batch_size, batch_count)
            # for batch_index in range(batch_count):
            #   self.buckets_indices.append(BucketBatchIndex(bucket_index, bucket_batch_size, batch_index))
            # ↑ここまで

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def load_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        img = np.array(image, np.uint8)
        return img

    def trim_and_resize_if_required(self, image, reso, resized_size):
        image_height, image_width = image.shape[0:2]

        if image_width != resized_size[0] or image_height != resized_size[1]:
            # リサイズする
            image = cv2.resize(
                image, resized_size, interpolation=cv2.INTER_AREA
            )  # INTER_AREAでやりたいのでcv2でリサイズ

        image_height, image_width = image.shape[0:2]
        if image_width > reso[0]:
            trim_size = image_width - reso[0]
            p = trim_size // 2 if not self.random_crop else random.randint(0, trim_size)
            # print("w", trim_size, p)
            image = image[:, p : p + reso[0]]
        if image_height > reso[1]:
            trim_size = image_height - reso[1]
            p = trim_size // 2 if not self.random_crop else random.randint(0, trim_size)
            # print("h", trim_size, p)
            image = image[p : p + reso[1]]

        assert (
            image.shape[0] == reso[1] and image.shape[1] == reso[0]
        ), f"internal error, illegal trimmed size: {image.shape}, {reso}"
        return image

    def cache_latents(self, vae):
        # TODO ここを高速化したい
        print("caching latents.")
        for info in tqdm(self.image_data.values()):
            if info.latents_npz is not None:
                info.latents = self.load_latents_from_npz(info, False)
                info.latents = torch.FloatTensor(info.latents)
                info.latents_flipped = self.load_latents_from_npz(
                    info, True
                )  # might be None
                if info.latents_flipped is not None:
                    info.latents_flipped = torch.FloatTensor(info.latents_flipped)
                continue

            image = self.load_image(info.absolute_path)
            image = self.trim_and_resize_if_required(
                image, info.bucket_reso, info.resized_size
            )

            img_tensor = self.image_transforms(image)
            img_tensor = img_tensor.unsqueeze(0).to(device=vae.device, dtype=vae.dtype)
            info.latents = (
                vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
            )

            if self.flip_aug:
                image = image[:, ::-1].copy()  # cannot convert to Tensor without copy
                img_tensor = self.image_transforms(image)
                img_tensor = img_tensor.unsqueeze(0).to(
                    device=vae.device, dtype=vae.dtype
                )
                info.latents_flipped = (
                    vae.encode(img_tensor).latent_dist.sample().squeeze(0).to("cpu")
                )

    def get_image_size(self, image_path):
        image = Image.open(image_path)
        return image.size

    def load_image_with_face_info(self, image_path: str):
        img = self.load_image(image_path)

        face_cx = face_cy = face_w = face_h = 0
        if self.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        min_scale = max(
            self.height / height, self.width / width
        )  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(
            1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[1]))
        )  # 指定した顔最小サイズ
        max_scale = min(
            1.0, max(min_scale, self.size / (face_size * self.face_crop_aug_range[0]))
        )  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        if nh <= self.height or nw <= self.width:
            raise KohyaDatasetException(
                f"internal error. small scale {scale}, {width}*{height}"
            )
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(
            zip((self.height, self.width), (height, width), (face_cy, face_cx))
        ):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if self.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = (
                    p1 + (random.randint(0, range) + random.randint(0, range)) - range
                )  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if self.face_crop_aug_range[0] != self.face_crop_aug_range[1]:
                    if face_size > self.size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def load_latents_from_npz(self, image_info: ImageInfo, flipped):
        npz_file = image_info.latents_npz_flipped if flipped else image_info.latents_npz
        if npz_file is None:
            return None
        return np.load(npz_file)["arr_0"]

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index == 0:
            self.shuffle_buckets()

        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        images = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

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
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    image_info.absolute_path
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img = self.trim_and_resize_if_required(
                        img, image_info.bucket_reso, image_info.resized_size
                    )
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            self.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"

                # augmentation
                if self.aug is not None:
                    img = self.aug(image=img)["image"]

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる

            images.append(image)
            latents_list.append(latents)

            caption = self.process_caption(image_info.caption)
            captions.append(caption)
            if (
                not self.token_padding_disabled
            ):  # this option might be omitted in future
                input_ids_list.append(self.get_input_ids(caption))

        example = {}
        example["loss_weights"] = torch.FloatTensor(loss_weights)

        if self.token_padding_disabled:
            # padding=True means pad in the batch
            example["input_ids"] = self.tokenizer(
                captions, padding=True, truncation=True, return_tensors="pt"
            ).input_ids
        else:
            # batch processing seems to be good
            example["input_ids"] = torch.stack(input_ids_list)

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
            example["captions"] = captions
        return example
