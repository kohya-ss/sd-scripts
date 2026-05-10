"""Latent and text-encoder output caching helpers.

Functions that move pre-computed VAE latents and text-encoder hidden states
between memory and ``.npz`` files on disk. Used both during the explicit
``cache_latents`` / ``cache_text_encoder_outputs`` preprocessing passes and on
the fly inside ``BaseDataset``.

``BucketManager`` / ``load_image`` / ``IMAGE_TRANSFORMS`` (and ``ImageInfo`` for
type checks) live in ``library.dataset``; ``library.dataset`` imports
``library.caching`` at top level so we pull the dataset symbols in lazily
through ``_ds()`` to avoid an import cycle.
``HIGH_VRAM`` lives in ``library.accelerator_setup``; it has no cycle with this
module so it is imported directly.
"""

import logging
import os
import random
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from diffusers import AutoencoderKL

from library import accelerator_setup
from library.device_utils import clean_memory_on_device
from library.utils import resize_image  # noqa: F401  (kept for symmetry / debugging)

if TYPE_CHECKING:
    from library.dataset import ImageInfo


def _ds():
    """Late-bound ``library.dataset`` accessor (BucketManager / load_image / IMAGE_TRANSFORMS)."""
    import library.dataset as m
    return m


logger = logging.getLogger(__name__)


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    try:
        npz = np.load(npz_path)
        if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
            return False
        if npz["latents"].shape[1:3] != expected_latents_size:
            return False

        if flip_aug:
            if "latents_flipped" not in npz:
                return False
            if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                return False

        if alpha_mask:
            if "alpha_mask" not in npz:
                return False
            if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                return False
        else:
            if "alpha_mask" in npz:
                return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        image = image[p : p + reso[1]]

    # random cropの場合のcropされた値をどうcrop left/topに反映するべきか全くアイデアがない
    # I have no idea how to reflect the cropped value in crop left/top in the case of random crop

    crop_ltrb = _ds().BucketManager.get_crop_ltrb(reso, original_size)

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List["ImageInfo"], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = (
            _ds().load_image(info.absolute_path, use_alpha_mask)
            if info.image is None
            else np.array(info.image, np.uint8)
        )
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = _ds().IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL,
    cache_to_disk: bool,
    image_infos: List["ImageInfo"],
    flip_aug: bool,
    use_alpha_mask: bool,
    random_crop: bool,
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = (
            _ds().load_image(info.absolute_path, use_alpha_mask)
            if info.image is None
            else np.array(info.image, np.uint8)
        )
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.FloatTensor(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        image = image[:, :, :3]  # remove alpha channel if exists
        image = _ds().IMAGE_TRANSFORMS(image)
        images.append(image)

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents, alpha_masks):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            # save_latents_to_disk(
            #     info.latents_npz,
            #     latent,
            #     info.latents_original_size,
            #     info.latents_crop_ltrb,
            #     flipped_latent,
            #     alpha_mask,
            # )
            pass
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent
            info.alpha_mask = alpha_mask

    if not accelerator_setup.HIGH_VRAM:
        clean_memory_on_device(vae.device)
