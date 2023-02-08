# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from io import BytesIO

import logging
import warnings
import base64
import random

import numpy as np
import torch

from PIL import Image, ImageFile
from itertools import chain
from ofa.data.ofa_dataset import OFADataset
from ofa.data import data_utils

from PIL import Image
from io import BytesIO
import base64

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_source=False,
        left_pad_target=False,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
        )

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    code_images = np.array([s["code_image"] for s in samples])
    code_masks = torch.cat([sample['code_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target", left_pad=left_pad_target)
        tgt_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        )
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
    else:
        ntokens = src_lengths.sum().item()

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "code_masks": code_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "code_images": code_images,
        "target": target
    }

    return batch


def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x


class ImageGenDataset(OFADataset):
    def __init__(
            self,
            split,
            dataset,
            bpe,
            src_dict,
            tgt_dict=None,
            max_src_length=128,
            code_dict_size=8192,
            code_image_size=256,
            num_bins=1000
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length

        self.code_dict_size = code_dict_size
        self.num_codes = (code_image_size // 8) ** 2
        self.num_bins = num_bins

        slice_id = self.dataset.slice_id
        empty_img = Image.new('RGB', (code_image_size, code_image_size))
        empty_img.save(f'temp_{slice_id}.png')
        img = Image.open(f'temp_{slice_id}.png')
        img_buffer = BytesIO()
        img.save(img_buffer, format=img.format)
        byte_data = img_buffer.getvalue()
        self.empty_image_base64 = base64.urlsafe_b64encode(byte_data)

    def __getitem__(self, index):

        data = self.dataset[index]
        if len(data) == 2:
            uniq_id, text = data
            image_code = [0] * 1024
            image = self.empty_image_base64
        elif len(data) == 3:
            uniq_id, text, image_code = data
            image_code = [int(num) for num in image_code.strip().split()]
            image = self.empty_image_base64
        elif len(data) == 4:
            uniq_id, image, text, image_code = data
            image_code = [int(num) for num in image_code.strip().split()]
        else:
            raise NotImplementedError
        code_mask = torch.tensor([True])
        image_code = torch.LongTensor(image_code)
        tgt_item = image_code + len(self.src_dict) - self.code_dict_size - self.num_bins
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        caption_token_list = text.strip().split()
        caption = ' '.join(caption_token_list[:self.max_src_length])
        src_item = self.encode_text(
            " what is the complete image? caption: {}".format(caption),
            append_bos=True,
            append_eos=True
        )
        example = {
            "id": uniq_id,
            "source": src_item,
            "code_mask": code_mask,
            "code_image": image,
            "target": target_item,
            "prev_output_tokens": prev_output_item
        }
        return example

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return collate(samples, pad_idx=self.pad, eos_idx=self.eos)
