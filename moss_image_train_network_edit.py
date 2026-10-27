# MOSS-Image reference-conditioned editing LoRA training script
#
# Trains a LoRA for an editing task (e.g. occlusion removal, clothing change) conditioned on a reference image. The
# reference image is VAE-encoded and its tokens are concatenated to the token sequence of the noisy target image
# ([image, reference, text], "Control LoRA" / FLUX.1 Kontext style): the reference tokens are embedded by the same
# video_embedder, attend bidirectionally with the image and text tokens, and get `--ref_rope_r` as the r coordinate of
# the 4D RoPE (the r axis is always 0 in the base model, so this is an offset on an unused axis). The loss is computed
# on the target image tokens only.
#
# The dataset uses the ControlNet dataset config: the conditioning images are the reference images, and the normal
# images / captions are the edited targets (the caption describes the target image). The reference latents can be
# cached like the target latents (--cache_ref_latents / --cache_ref_latents_to_disk / --ref_latents_cache_dir).
#
# Inpainting-like tasks: `--ref_mask_fill G` fills the alpha region (255 = hole) of RGBA reference images with the gray level
# G before the VAE, and `--ref_random_mask` instead cuts a random hole (library.mask_generator) into the reference at every
# step (the reference images are then typically the target images themselves; not compatible with the reference cache).

import argparse
from typing import Optional, Union

import numpy as np
import torch
from accelerate import Accelerator

from library import moss_image_models, strategy_moss_image
from library.mask_generator import random_mask
import library.args as args_util
from library.controlnet_dataset import ControlNetDataset
from library.dataset import DatasetGroup, MinimalDataset
from library.device_utils import clean_memory_on_device
import moss_image_train_network
from library import moss_image_train_utils
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MossImageEditNetworkTrainer(moss_image_train_network.MossImageNetworkTrainer):
    def __init__(self):
        super().__init__()
        self.ref_latents_cached = False

    def use_controlnet_dataset(self, args) -> bool:
        return True  # conditioning images are the reference images

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        assert not args.masked_loss, (
            "masked_loss cannot be used with reference editing training: conditioning images are reference images, not masks"
            " / 参照編集学習ではmasked_lossは使えません（conditioning imageはマスクではなく参照画像です）"
        )

        if args.ref_random_mask:
            assert not args.cache_ref_latents and not args.cache_ref_latents_to_disk and args.ref_latents_cache_dir is None, (
                "ref_random_mask generates a new reference at every step and cannot be used with the reference latents cache"
                " / ref_random_mask は毎ステップ参照を生成するので参照 latent のキャッシュとは併用できません"
            )
            if args.ref_mask_fill is None:
                args.ref_mask_fill = DEFAULT_REF_MASK_FILL

        if args.ref_latents_cache_dir is not None and not args.cache_ref_latents_to_disk:
            logger.warning("ref_latents_cache_dir is specified, so cache_ref_latents_to_disk is also enabled")
            args.cache_ref_latents_to_disk = True
        if args.cache_ref_latents_to_disk and not args.cache_ref_latents:
            logger.warning("cache_ref_latents_to_disk is enabled, so cache_ref_latents is also enabled")
            args.cache_ref_latents = True

        for group in [train_dataset_group, val_dataset_group]:
            if group is None or not isinstance(group, DatasetGroup):
                continue
            for dataset in group.datasets:
                assert isinstance(dataset, ControlNetDataset), (
                    "reference editing training requires ControlNet dataset (conditioning_data_dir with reference images)"
                    " / 参照編集学習にはControlNetデータセット（参照画像のconditioning_data_dir）が必要です"
                )
                if args.ref_mask_fill is not None and not args.ref_random_mask:
                    dataset.cond_alpha_fill = args.ref_mask_fill  # fill the alpha region of the reference images

    def cache_conditioning_latents_if_needed(self, args, accelerator, vae, vae_dtype, dataset_group):
        if not args.cache_ref_latents or not isinstance(dataset_group, DatasetGroup):
            return

        # The same strategy class as the target latents, with its own memory / disk setting
        caching_strategy = strategy_moss_image.MossImageLatentsCachingStrategy(
            args.cache_ref_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        for dataset in dataset_group.datasets:
            if isinstance(dataset, ControlNetDataset):
                dataset.new_cache_conditioning_latents(vae, accelerator, caching_strategy, args.ref_latents_cache_dir)
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

        self.ref_latents_cached = True

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
    ) -> torch.Tensor:
        # Reference latents: cached by the dataset (raw VAE latents) or encoded here from the conditioning images
        cond_latents = batch.get("conditioning_latents")
        if cond_latents is None:
            cond_images = batch.get("conditioning_images")
            assert cond_images is not None, "conditioning images not found in batch / バッチにconditioning imageがありません"
            if args.ref_random_mask:
                cond_images = apply_random_reference_masks(cond_images, args.ref_mask_fill)
            if vae.device != accelerator.device:
                vae.to(accelerator.device)  # the VAE is on the CPU when the target latents are cached: keep it on the GPU from now on
            with torch.no_grad():
                cond_latents = self.encode_images_to_latents(args, vae, cond_images.to(accelerator.device, dtype=vae_dtype))
        else:
            cond_latents = cond_latents.to(accelerator.device)

        ref_latents = self.shift_scale_latents(args, cond_latents)  # patchify + normalize, same as the target latents
        batch["ref_latents"] = ref_latents.to(dtype=weight_dtype)

        return super().process_batch(
            batch,
            text_encoders,
            unet,
            network,
            vae,
            noise_scheduler,
            vae_dtype,
            weight_dtype,
            accelerator,
            args,
            text_encoding_strategy,
            tokenize_strategy,
            is_train,
            train_text_encoder,
            train_unet,
        )

    def call_dit(
        self,
        args,
        accelerator,
        dit: moss_image_models.MossImage,
        batch,
        noisy_model_input: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        is_train: bool,
    ) -> torch.Tensor:
        ref = batch["ref_latents"].to(device=noisy_model_input.device, dtype=noisy_model_input.dtype)

        # Reference dropout: replace the reference with zeros so that the model also learns the no-reference case
        # (enables reference-free CFG at inference)
        if is_train and args.ref_dropout_rate > 0.0:
            drop = torch.rand(ref.shape[0], device=ref.device) < args.ref_dropout_rate
            ref = torch.where(drop.view(-1, 1, 1, 1), torch.zeros_like(ref), ref)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            return dit(noisy_model_input, timesteps, prompt_embeds, attn_mask, ref=ref, ref_rope_r=args.ref_rope_r)

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_ref_dropout_rate"] = args.ref_dropout_rate
        metadata["ss_ref_rope_r"] = args.ref_rope_r
        metadata["ss_cache_ref_latents"] = bool(args.cache_ref_latents)
        metadata["ss_ref_mask_fill"] = args.ref_mask_fill
        metadata["ss_ref_random_mask"] = bool(args.ref_random_mask)


DEFAULT_REF_MASK_FILL = 128


def apply_random_reference_masks(images: torch.Tensor, fill: int) -> torch.Tensor:
    """Cut a random inpainting hole (library.mask_generator.random_mask) into each reference image [B, 3, H, W] (-1..1) and
    fill it with the gray level `fill` (0-255)."""
    images = images.clone()
    value = fill / 127.5 - 1.0
    for i in range(images.shape[0]):
        mask = random_mask(images.shape[-1], images.shape[-2])  # PIL "L", 255 = inpaint
        mask = torch.from_numpy(np.asarray(mask, dtype=np.float32) >= 127.5).to(images.device)  # [H, W] bool
        images[i] = torch.where(mask[None], torch.full_like(images[i], value), images[i])
    return images


def add_moss_image_edit_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ref_dropout_rate",
        type=float,
        default=0.0,
        help="rate to replace the reference latents with zeros during training (for reference-free CFG)"
        " / 学習中に参照latentをゼロに置き換える割合（参照なしCFG用）",
    )
    parser.add_argument(
        "--ref_rope_r",
        type=float,
        default=1.0,
        help="r coordinate of the reference tokens in the 4D RoPE (the base model always uses 0: 1.0 (default) gives the reference"
        " tokens an offset on the unused axis, 0.0 places them at the same positions as the image tokens). The same value must be"
        " used at inference (saved in the metadata)"
        " / 参照トークンの 4D RoPE の r 座標（ベースモデルは常に 0。デフォルトの 1.0 は未使用軸でのオフセット、0.0 は画像トークンと同じ位置）。"
        "推論時も同じ値を使うこと（メタデータに保存される）",
    )
    parser.add_argument(
        "--ref_mask_fill",
        type=int,
        default=None,
        help="gray level (0-255) to fill the alpha region (alpha 255 = hole) of RGBA reference images with before the VAE,"
        " for inpainting-like tasks. Default: the alpha channel is ignored. Also used for the sample images (`--cn` with an RGBA"
        f" image or `--mk` mask) and by --ref_random_mask (default {DEFAULT_REF_MASK_FILL} there)"
        " / RGBA の参照画像のアルファ領域（255 = 穴）を VAE に通す前にこのグレー値 (0-255) で塗りつぶす（inpainting 系のタスク用）。"
        "デフォルトではアルファは無視される。サンプル生成（RGBA の `--cn` または `--mk` マスク）と --ref_random_mask でも使われる",
    )
    parser.add_argument(
        "--ref_random_mask",
        action="store_true",
        help="cut a random hole (library.mask_generator) into the reference image at every step and fill it with --ref_mask_fill:"
        " the reference images are typically the target images themselves (conditioning_data_dir = image_dir)."
        " Not compatible with the reference latents cache"
        " / 毎ステップ参照画像にランダムな穴（library.mask_generator）を開けて --ref_mask_fill で塗る。参照画像は通常、目標画像そのもの"
        "（conditioning_data_dir = image_dir）。参照 latent のキャッシュとは併用不可",
    )
    parser.add_argument(
        "--cache_ref_latents",
        action="store_true",
        help="cache the VAE latents of the reference (conditioning) images in memory at startup, like --cache_latents"
        " / 参照画像（conditioning image）の VAE latent を起動時にメモリにキャッシュする（--cache_latents と同様）",
    )
    parser.add_argument(
        "--cache_ref_latents_to_disk",
        action="store_true",
        help="cache the reference latents to disk (npz next to the reference image by default), like --cache_latents_to_disk"
        " / 参照画像の latent をディスクにキャッシュする（デフォルトでは参照画像の隣に npz を置く。--cache_latents_to_disk と同様）",
    )
    parser.add_argument(
        "--ref_latents_cache_dir",
        type=str,
        default=None,
        help="directory for the reference latents cache files instead of next to the reference images (implies --cache_ref_latents_to_disk)"
        " / 参照画像の latent キャッシュを参照画像の隣ではなくこのディレクトリに保存する（--cache_ref_latents_to_disk を含意）",
    )


def setup_parser() -> argparse.ArgumentParser:
    parser = moss_image_train_network.setup_parser()
    add_moss_image_edit_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"  # backward compatibility

    if args.show_timesteps:
        moss_image_train_utils.show_timesteps(args)
    else:
        trainer = MossImageEditNetworkTrainer()
        trainer.train(args)
