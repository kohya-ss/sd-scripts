"""Dataset core: image bookkeeping, bucketing and the BaseDataset/DatasetGroup hierarchy.

This module owns the data-loading side of training:

* ``IMAGE_EXTENSIONS`` / ``TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX*`` constants
* ``ImageInfo`` (per-image metadata), ``BucketManager`` (aspect-ratio buckets),
  ``BucketBatchIndex``, ``AugHelper`` (color augmentation)
* ``BaseDataset`` (abstract dataset shared by DreamBooth / FineTuning / ControlNet)
* ``DatasetGroup`` (concat of multiple datasets)
* ``MinimalDataset`` and ``load_arbitrary_dataset`` (user-supplied dataset hook)
* ``debug_dataset`` viewer plus ``glob_images`` / ``glob_images_pathlib`` helpers
* ``split_train_val`` shared between training/validation dataset halves

Leaf image helpers (``IMAGE_TRANSFORMS`` / ``load_image`` / ``get_crop_ltrb`` /
``trim_and_resize_if_required``) live in ``library.utils`` and are re-exported
here for backward compatibility.

The DreamBooth / FineTuning / ControlNet specializations of ``BaseDataset`` live
in their dedicated modules: ``library.dreambooth_dataset``,
``library.finetuning_dataset`` and ``library.controlnet_dataset``.
``HIGH_VRAM`` (a mutable module-level flag toggled by ``enable_high_vram``) lives
in ``library.accelerator_setup``; that module has no cycle with this one so it
is imported directly.
"""

import glob
import importlib
import logging
import math
import os
import pathlib
import random
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import cv2
import imagesize
import numpy as np
import torch
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPTokenizer

import library.model_util as model_util
from library import accelerator_setup
from library.device_utils import clean_memory_on_device
from library.strategy_base import (
    LatentsCachingStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    TokenizeStrategy,
)
from library.subset import (
    BaseSubset,
    ControlNetSubset,
    DreamBoothSubset,
    FineTuningSubset,
)
from library.utils import (
    IMAGE_TRANSFORMS,
    get_crop_ltrb,
    load_image,
    resize_image,
    setup_logging,
    trim_and_resize_if_required,
    validate_interpolation_fn,
)

setup_logging()
logger = logging.getLogger(__name__)


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Linux and Windows
try:
    import pillow_jxl
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"
TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3 = "_sd3_te.npz"

def split_train_val(
    paths: List[str],
    sizes: List[Optional[Tuple[int, int]]],
    is_training_dataset: bool,
    validation_split: float,
    validation_seed: int | None,
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """
    Split the dataset into train and validation

    Shuffle the dataset based on the validation_seed or the current random seed.
    For example if the split of 0.2 of 100 images.
    [0:80] = 80 training images
    [80:] = 20 validation images
    """
    dataset = list(zip(paths, sizes))
    if validation_seed is not None:
        logging.info(f"Using validation seed: {validation_seed}")
        prevstate = random.getstate()
        random.seed(validation_seed)
        random.shuffle(dataset)
        random.setstate(prevstate)
    else:
        random.shuffle(dataset)

    paths, sizes = zip(*dataset)
    paths = list(paths)
    sizes = list(sizes)
    # Split the dataset between training and validation
    if is_training_dataset:
        # Training dataset we split to the first part
        split = math.ceil(len(paths) * (1 - validation_split))
        return paths[0:split], sizes[0:split]
    else:
        # Validation dataset we split to the second part
        split = len(paths) - round(len(paths) * validation_split)
        return paths[split:], sizes[split:]


class ImageInfo:
    def __init__(
        self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str, caption_dropout_rate: float = 0.0
    ) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.caption_dropout_rate: float = caption_dropout_rate
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_original_size: Optional[Tuple[int, int]] = None  # original image size, not latents size
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in original pixel size, not latents size
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = None  # filename. set in cache_text_encoder_outputs

        # new
        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        # old
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = None  # alpha mask can be flipped in runtime
        self.resize_interpolation: Optional[str] = None


class BucketManager:
    def __init__(self, no_upscale, max_reso, min_size, max_size, reso_steps) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        if not self.no_upscale:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # 縮小のみを行う
            if image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
            bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
            bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
            # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

            reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # implementation moved to library.utils.get_crop_ltrb; kept as a staticmethod for backward compatibility
        return get_crop_ltrb(bucket_reso, image_size)


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        # self.color_aug_method = albu.OneOf(
        #     [
        #         albu.HueSaturationValue(8, 0, 0, p=0.5),
        #         albu.RandomGamma((95, 105), p=0.5),
        #     ],
        #     p=0.33,
        # )
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        train_inpainting: bool,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None,
        skip_image_resolution: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_info = None  # for metadata

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # inpainting
        self.train_inpainting = train_inpainting

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        if resize_interpolation is not None:
            assert validate_interpolation_fn(
                resize_interpolation
            ), f'Resize interpolation "{resize_interpolation}" is not a valid interpolation'
        self.resize_interpolation = resize_interpolation

        self.skip_image_resolution = skip_image_resolution

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        self.tokenize_strategy = None
        self.text_encoder_output_caching_strategy = None
        self.latents_caching_strategy = None

    def set_current_strategies(self):
        self.tokenize_strategy = TokenizeStrategy.get_strategy()
        self.text_encoder_output_caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        self.latents_caching_strategy = LatentsCachingStrategy.get_strategy()

    def adjust_min_max_bucket_reso_by_steps(
        self, resolution: Tuple[int, int], min_bucket_reso: int, max_bucket_reso: int, bucket_reso_steps: int
    ) -> Tuple[int, int]:
        # make min/max bucket reso to be multiple of bucket_reso_steps
        if min_bucket_reso % bucket_reso_steps != 0:
            adjusted_min_bucket_reso = min_bucket_reso - min_bucket_reso % bucket_reso_steps
            logger.warning(
                f"min_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / min_bucket_resoがbucket_reso_stepsの倍数になるように調整されました: {min_bucket_reso} -> {adjusted_min_bucket_reso}"
            )
            min_bucket_reso = adjusted_min_bucket_reso
        if max_bucket_reso % bucket_reso_steps != 0:
            adjusted_max_bucket_reso = max_bucket_reso + bucket_reso_steps - max_bucket_reso % bucket_reso_steps
            logger.warning(
                f"max_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / max_bucket_resoがbucket_reso_stepsの倍数になるように調整されました: {max_bucket_reso} -> {adjusted_max_bucket_reso}"
            )
            max_bucket_reso = adjusted_max_bucket_reso

        assert (
            min(resolution) >= min_bucket_reso
        ), f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください"
        assert (
            max(resolution) <= max_bucket_reso
        ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください"

        return min_bucket_reso, max_bucket_reso

    def set_seed(self, seed):
        self.seed = seed

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epochが切り替わったらバケツをシャッフルする
            if epoch > self.current_epoch:
                logger.info("epoch is incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
                # self.current_epoch seem to be set to 0 again in the next epoch. it may be caused by skipped_dataloader?
            else:
                logger.warning("epoch is not incremented. current_epoch: {}, epoch: {}".format(self.current_epoch, epoch))
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def process_caption(self, subset: BaseSubset, caption):
        # caption に prefix/suffix を付ける
        if subset.caption_prefix:
            caption = subset.caption_prefix + " " + caption
        if subset.caption_suffix:
            caption = caption + " " + subset.caption_suffix

        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            if subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0:
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(subset.keep_tokens_separator, 1)
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(subset.keep_tokens_separator, 1)
                        fixed_suffix_tokens = [t.strip() for t in fixed_suffix_part.split(subset.caption_separator) if t.strip()]

                    fixed_tokens = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
                    flex_tokens = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
                else:
                    tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor(
                            (self.current_step) * ((len(flex_tokens) - subset.token_warmup_min) / (subset.token_warmup_step))
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(subset.secondary_separator, subset.caption_separator)

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

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        # # run in parallel
        # max_workers = min(os.cpu_count(), len(self.image_data))  # TODO consider multi-gpu (processes)
        # with ThreadPoolExecutor(max_workers) as executor:
        #     futures = []
        #     for info in tqdm(self.image_data.values(), desc="loading image sizes"):
        #         if info.image_size is None:
        #             def get_and_set_image_size(info):
        #                 info.image_size = self.get_image_size(info.absolute_path)
        #             futures.append(executor.submit(get_and_set_image_size, info))
        #             # consume futures to reduce memory usage and prevent Ctrl-C hang
        #             if len(futures) >= max_workers:
        #                 for future in futures:
        #                     future.result()
        #                 futures = []
        #     for future in futures:
        #         future.result()

        if self.enable_bucket:
            logger.info("make buckets")
        else:
            logger.info("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
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
                    logger.warning(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

                # logger.info(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            logger.info("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む）")
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            if len(img_ar_errors) == 0:
                mean_img_ar_error = 0  # avoid NaN
            else:
                img_ar_errors = np.array(img_ar_errors)
                mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        assert self.bucket_reso_steps is None or self.bucket_reso_steps % min_steps == 0, (
            f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
            + f"bucket_reso_stepsが{self.bucket_reso_steps}です。{min_steps}で割り切れる必要があります"
        )

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0
                    and not cache_supports_dropout
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                )
                for subset in self.subsets
            ]
        )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    other is not None
                    and self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batch: List[ImageInfo] = []
        current_condition = None

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        # define a function to submit a batch to cache
        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()  # future to image
            caching_strategy.cache_batch_latents(model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop)

            # remove image from memory
            for info in batch:
                info.image = None

        # define ThreadPoolExecutor to load images in parallel
        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(max_workers, caching_strategy.batch_size)  # max_workers should be less than batch_size
        executor = ThreadPoolExecutor(max_workers)

        try:
            # iterate images
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos)):
                subset = self.image_to_subset[info.image_key]

                if info.latents_npz is not None:  # fine tuning dataset
                    continue

                # check disk cache exists and size of latents
                if caching_strategy.cache_to_disk:
                    # info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                    info.latents_npz = caching_strategy.get_latents_npz_path(info.absolute_path, info.image_size)

                    # if the modulo of num_processes is not equal to process_index, skip caching
                    # this makes each process cache different latents
                    if i % num_processes != process_index:
                        continue

                    # print(f"{process_index}/{num_processes} {i}/{len(image_infos)} {info.latents_npz}")

                    cache_available = caching_strategy.is_disk_cached_latents_expected(
                        info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
                    )
                    if cache_available:  # do not add to batch
                        continue

                # if batch is not empty and condition is changed, flush the batch. Note that current_condition is not None if batch is not empty
                condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
                if len(batch) > 0 and current_condition != condition:
                    submit_batch(batch, current_condition)
                    batch = []
                if condition != current_condition and accelerator_setup.HIGH_VRAM:  # even with high VRAM, if shape is changed
                    clean_memory_on_device(accelerator.device)

                if info.image is None:
                    # load image in parallel
                    info.image = executor.submit(load_image, info.absolute_path, condition.alpha_mask)

                batch.append(info)
                current_condition = condition

                # if number of data in batch is enough, flush the batch
                if len(batch) >= caching_strategy.batch_size:
                    submit_batch(batch, current_condition)
                    batch = []
                    # current_condition = None  # keep current_condition to avoid next `clean_memory_on_device` call

            if len(batch) > 0:
                submit_batch(batch, current_condition)

        finally:
            executor.shutdown()

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        r"""
        a brand new method to cache text encoder outputs. This method caches text encoder outputs with caching strategy.
        """
        tokenize_strategy = TokenizeStrategy.get_strategy()
        text_encoding_strategy = TextEncodingStrategy.get_strategy()
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        batch_size = caching_strategy.batch_size or self.batch_size

        logger.info("caching Text Encoder outputs with caching strategy.")
        image_infos = list(self.image_data.values())

        # split by resolution
        batches = []
        batch = []

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos)):
            # check disk cache exists and size of text encoder outputs
            if caching_strategy.cache_to_disk:
                te_out_npz = caching_strategy.get_outputs_npz_path(info.absolute_path)
                info.text_encoder_outputs_npz = te_out_npz  # set npz filename regardless of cache availability

                # if the modulo of num_processes is not equal to process_index, skip caching
                # this makes each process cache different text encoder outputs
                if i % num_processes != process_index:
                    continue

                cache_available = caching_strategy.is_disk_cached_outputs_expected(te_out_npz)
                if cache_available:  # do not add to batch
                    continue

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if len(batches) == 0:
            logger.info("no Text Encoder outputs to cache")
            return

        # iterate batches
        logger.info("caching Text Encoder outputs...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            caching_strategy.cache_batch_outputs(tokenize_strategy, models, text_encoding_strategy, batch)

    def get_image_size(self, image_path):
        if image_path.endswith(".jxl") or image_path.endswith(".JXL"):
            return get_jxl_size(image_path)
        # return imagesize.get(image_path)
        image_size = imagesize.get(image_path)
        if image_size[0] <= 0:
            # imagesize doesn't work for some images, so use PIL as a fallback
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                logger.warning(f"failed to get image size: {image_path}, error: {e}")
                image_size = (0, 0)
        return image_size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str, alpha_mask=False):
        img = load_image(image_path, alpha_mask)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(self.height / height, self.width / width)  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # 指定した顔最小サイズ
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = resize_image(image, width, height, nw, nh, subset.resize_interpolation)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs_list = []
        custom_attributes = []
        masks = []
        masked_images = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5  # not flipped or flipped with 50% chance

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                    alpha_mask = image_info.alpha_mask
                else:
                    latents = image_info.latents_flipped
                    alpha_mask = None if image_info.alpha_mask is None else torch.flip(image_info.alpha_mask, [1])

                image = None
            elif image_info.latents_npz is not None:  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                latents, original_size, crop_ltrb, flipped_latents, alpha_mask = (
                    self.latents_caching_strategy.load_latents_from_disk(image_info.latents_npz, image_info.bucket_reso)
                )
                if flipped:
                    latents = flipped_latents
                    alpha_mask = None if alpha_mask is None else alpha_mask[:, ::-1].copy()  # copy to avoid negative stride problem
                    del flipped_latents
                latents = torch.FloatTensor(latents)
                if alpha_mask is not None:
                    alpha_mask = torch.FloatTensor(alpha_mask)

                image = None
            else:
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    subset, image_info.absolute_path, subset.alpha_mask
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                        resize_interpolation=image_info.resize_interpolation,
                    )
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
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

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    # augment RGB channels only
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                if subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]  # [H,W]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0  # 0.0~1.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
                else:
                    alpha_mask = None

                img = img[:, :, :3]  # remove alpha channel

                if self.train_inpainting:
                    pil_image = transforms.functional.to_pil_image(img)
                    mask = self.random_mask(pil_image.size)
                    mask, masked_image = self.prepare_mask_and_masked_image(pil_image, mask)

                    masks.append(mask)
                    masked_images.append(masked_image)

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # captionとtext encoder outputを処理する
            caption = image_info.caption  # default

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension

            input_ids_list.append(input_ids)
            captions.append(caption)

        def none_or_stack_elements(tensors_list, converter):
            # [[clip_l, clip_g, t5xxl], [clip_l, clip_g, t5xxl], ...] -> [torch.stack(clip_l), torch.stack(clip_g), torch.stack(t5xxl)]
            if len(tensors_list) == 0 or tensors_list[0] == None or len(tensors_list[0]) == 0 or tensors_list[0][0] is None:
                return None

            # old implementation without padding: all elements must have same length
            # return [torch.stack([converter(x[i]) for x in tensors_list]) for i in range(len(tensors_list[0]))]

            # new implementation with padding support
            result = []
            for i in range(len(tensors_list[0])):
                tensors = [x[i] for x in tensors_list]
                if tensors[0].ndim == 0:
                    # scalar value: e.g. ocr mask
                    result.append(torch.stack([converter(x[i]) for x in tensors_list]))
                    continue

                min_len = min([len(x) for x in tensors])
                max_len = max([len(x) for x in tensors])

                if min_len == max_len:
                    # no padding
                    result.append(torch.stack([converter(x) for x in tensors]))
                else:
                    # padding
                    tensors = [converter(x) for x in tensors]
                    if tensors[0].ndim == 1:
                        # input_ids or mask
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, max_len - x.shape[0]))) for x in tensors]))
                    else:
                        # text encoder outputs
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0]))) for x in tensors]))
            return result

        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.FloatTensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        # if one of alpha_masks is not None, we need to replace None with ones
        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones((images[i].shape[1], images[i].shape[2]), dtype=torch.float32)
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (latents_list[i].shape[1] * 8, latents_list[i].shape[2] * 8), dtype=torch.float32
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        example["masks"] = torch.stack(masks) if masks else None
        example["masked_images"] = torch.stack(masked_images) if masked_images else None

        example["latents"] = torch.stack(latents_list) if latents_list[0] is not None else None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    @staticmethod
    def prepare_mask_and_masked_image(image, mask):
        image = np.array(image.convert("RGB"))
        image = image.transpose(2, 0, 1)  # HWC -> CHW
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        mask = np.array(mask.convert("L"))
        mask = mask.astype(np.float32) / 255.0
        mask = mask[None]  # 1,H,W
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

        masked_image = image * (mask < 0.5)

        return mask, masked_image

    # generate random masks
    @staticmethod
    def random_mask(im_shape):
        from library.mask_generator import random_mask as _random_mask

        w, h = im_shape
        return _random_mask(w, h)



class DatasetGroup(torch.utils.data.ConcatDataset):
    # DreamBoothDataset / FineTuningDataset still live in library.train_util (PR-1d will move them).
    def __init__(self, datasets: 'Sequence[Union[DreamBoothDataset, FineTuningDataset]]'):
        self.datasets: 'List[Union[DreamBoothDataset, FineTuningDataset]]'

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def set_text_encoder_output_caching_strategy(self, strategy: TextEncoderOutputsCachingStrategy):
        """
        DataLoader is run in multiple processes, so we need to set the strategy manually.
        """
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False) -> bool:
        return all([dataset.is_text_encoder_output_cacheable(cache_supports_dropout) for dataset in self.datasets])

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()




def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / データセットの長さ（ステップ数）: {len(train_dataset)}")
    logger.info(
        "`S` for next step, `E` for next epoch no. , Escape for exit. / Sキーで次のステップ、Eキーで次のエポック、Escキーで中断、終了します"
    )

    epoch = 1
    while True:
        logger.info(f"")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    # example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(f"network multiplier: {example['network_multipliers'][j]}")
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                # if show_input_ids:
                #     logger.info(f"input ids: {iid}")
                #     if "input_ids2" in example:
                #         logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example or "masked_images" in example:
                        cond_img = example["conditioning_images"][j] if "conditioning_images" in example else example["masked_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1



def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths



def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths



class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, train_inpainting=False, debug_dataset=False):
        super().__init__(resolution, network_multiplier, train_inpainting, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []



def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group



# collate_fn 用 epoch, step は multiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]
