# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import random
import functools

import torch
import base64
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from PIL import Image, ImageFile

from zhconv import convert
import unicodedata

from ofa.data import data_utils
from ofa.data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def collate(samples, pad_idx, eos_idx):
    if len(samples) == 0:
        return {}

    def merge(key):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx=eos_idx,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source")
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
    patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens")
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
    }

    return batch


def ocr_resize(img, patch_image_size, is_document=False):
    img = img.convert("RGB")
    width, height = img.size

    if is_document:
        new_height, new_width = 64, 1920
    else:
        if width >= height:
            new_width = max(64, patch_image_size)
            new_height = max(64, int(patch_image_size * (height / width)))
            top = random.randint(0, patch_image_size - new_height)
            bottom = patch_image_size - new_height - top
            left, right = 0, 0
        else:
            new_height = max(64, patch_image_size)
            new_width = max(64, int(patch_image_size * (width / height)))
            left = random.randint(0, patch_image_size - new_width)
            right = patch_image_size - new_width - left
            top, bottom = 0, 0

    img_new = F.resize(
        img,
        [new_height, new_width],
        interpolation=InterpolationMode.BICUBIC,
    )

    if is_document:
        img_split = transforms.ToTensor()(img_new).chunk(4, dim=-1)
        img_new = transforms.ToPILImage()(torch.cat(img_split, dim=-2))
        new_width, new_height = img_new.size
        top = random.randint(0, patch_image_size - new_height)
        bottom = patch_image_size - new_height - top
        left, right = 0, 0

    img_new = F.pad(img_new, padding=[left, top, right, bottom], padding_mode="edge")
    assert img_new.size == (patch_image_size, patch_image_size)

    return img_new


class OcrDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=224,
        imagenet_default_mean_and_std=False,
        is_document=False,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        self.patch_resize_transform = transforms.Compose(
            [
                lambda image: ocr_resize(
                    image, patch_image_size, is_document=is_document
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.bpe = bpe
        if type(bpe).__name__ == 'GPT2BPE':
            self.prompt = " what are the texts on the image?"
        elif type(bpe).__name__ == 'BertBPE':
            self.prompt = "图片上的文字是什么?"

    def __getitem__(self, index):
        uniq_id, image, caption = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        patch_image = self.patch_resize_transform(image)
        patch_mask = torch.tensor([True])

        caption = unicodedata.normalize("NFKC", convert(caption, "zh-hans"))
        if type(self.bpe).__name__ == 'GPT2BPE':
            caption_token_list = caption.lower().strip().split()
            tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])
        elif type(self.bpe).__name__ == 'BertBPE':
            tgt_caption = caption[: self.max_tgt_length].lower()
        src_item = self.encode_text(self.prompt)
        tgt_item = self.encode_text(" {}".format(tgt_caption))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data required for the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
