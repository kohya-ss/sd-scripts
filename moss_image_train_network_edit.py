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
# --ref_mode kv (one-way reference with cached K/V): instead of concatenating the reference tokens, the clean reference is
# run through the DiT once per step (empty prompt, --ref_timestep) and the image / text tokens of the generation
# additionally attend to the per-block reference keys / values. The reference itself is not updated, so the FFN cost of
# the reference tokens is saved, and at inference the reference K/V are computed once and shared by all steps.
# --ref_kv_lora on (default): the LoRA is applied to the reference forward too and trained through it (matches inference
# with a LoRA-merged DiT, e.g. ComfyUI). on_no_grad: applied but not trained through it. off: reference K/V of the base
# model (the inference has to compute them without the LoRA). --ref_kv_blocks restricts the blocks which see the
# reference (all by default).
#
# Inpainting-like tasks (--ref_mask_source alpha | random): the hole of the reference is given by the alpha channel of the
# RGBA reference image (255 = hole) or cut randomly at every step (library.mask_generator). The hole is applied in the
# latent space by default (--ref_mask_space latent): the clean reference latents (cacheable) are blended with a constant
# fill value using the mask downsampled to the latent resolution (smooth), so the context outside the hole stays exact up
# to the boundary. --ref_mask_space pixel fills the hole with a gray level before the VAE instead (LLLite style).

import argparse
import contextlib
from typing import Optional, Union

import numpy as np
import torch
from accelerate import Accelerator

from library import moss_image_models, strategy_base, strategy_moss_image
import library.args as args_util
from library.controlnet_dataset import ControlNetDataset
from library.dataset import DatasetGroup, MinimalDataset
from library.device_utils import clean_memory_on_device
from library.mask_generator import random_mask
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
        self.network = None  # set in post_process_network, used to disable the LoRA during the reference forward
        self.ref_kv_blocks = None
        self._uncond_embeds = None  # cached embeddings of the empty prompt [1, L, C] and mask [1, L] (CPU)

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

        if args.ref_latents_cache_dir is not None and not args.cache_ref_latents_to_disk:
            logger.warning("ref_latents_cache_dir is specified, so cache_ref_latents_to_disk is also enabled")
            args.cache_ref_latents_to_disk = True
        if args.cache_ref_latents_to_disk and not args.cache_ref_latents:
            logger.warning("cache_ref_latents_to_disk is enabled, so cache_ref_latents is also enabled")
            args.cache_ref_latents = True

        if args.ref_mode == "kv":
            assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
                "ref_mode kv does not support blocks_to_swap / ref_mode kv は blocks_to_swap と併用できません"
            )
            self.ref_kv_blocks = moss_image_train_utils.parse_skip_layers(args.ref_kv_blocks) or None
            if self.ref_kv_blocks is not None:
                logger.info(f"reference K/V is used in blocks: {sorted(self.ref_kv_blocks)}")

        if args.ref_mask_source == "random" and args.ref_mask_space == "pixel":
            assert not args.cache_ref_latents, (
                "ref_mask_source random with ref_mask_space pixel generates a new reference image at every step and cannot be"
                " used with the reference latents cache (use ref_mask_space latent)"
                " / ref_mask_source random と ref_mask_space pixel の組み合わせは毎ステップ参照画像を生成するので参照 latent の"
                "キャッシュとは併用できません（ref_mask_space latent を使ってください）"
            )

        for group in [train_dataset_group, val_dataset_group]:
            if group is None or not isinstance(group, DatasetGroup):
                continue
            for dataset in group.datasets:
                assert isinstance(dataset, ControlNetDataset), (
                    "reference editing training requires ControlNet dataset (conditioning_data_dir with reference images)"
                    " / 参照編集学習にはControlNetデータセット（参照画像のconditioning_data_dir）が必要です"
                )
                if args.ref_mask_source == "alpha":
                    if args.ref_mask_space == "pixel":
                        dataset.cond_alpha_fill = args.ref_mask_fill_pixel  # fill the hole of the reference images at load time
                    else:
                        dataset.cond_load_alpha = True  # the dataset returns the hole mask (pixel resolution) with the reference

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        self.network = network

    @contextlib.contextmanager
    def lora_disabled(self):
        """Disable the LoRA (multiplier 0) during the reference forward of --ref_mode kv with --ref_kv_lora off"""
        network = self.network
        if network is None or not hasattr(network, "set_multiplier"):
            yield
            return
        multiplier = network.multiplier
        network.set_multiplier(0.0)
        try:
            yield
        finally:
            network.set_multiplier(multiplier)

    def get_uncond_embeds(self, batch, text_encoders, text_encoding_strategy, tokenize_strategy, device, weight_dtype):
        """Embeddings of the empty prompt for the reference forward: [1, L, C] and mask [1, L]. Taken from the cached text
        encoder outputs of the batch (every npz holds the unconditional outputs) or encoded once with the text encoder."""
        if self._uncond_embeds is None:
            te_outputs = batch.get("text_encoder_outputs_list")
            if te_outputs is not None and len(te_outputs) >= 5:
                # [prompt_embeds, attn_mask, uncond_embeds, uncond_mask, caption_dropout_rates] (before the caption dropout)
                uncond_embeds, uncond_mask = te_outputs[2][:1], te_outputs[3][:1]
                length = int(uncond_mask.sum())  # drop the padding (all samples share the same unconditional outputs)
                self._uncond_embeds = (uncond_embeds[:, :length].cpu(), uncond_mask[:, :length].cpu())
            else:
                te = text_encoders[0]
                tokens = tokenize_strategy.tokenize("")
                with torch.no_grad():
                    outputs = text_encoding_strategy.encode_tokens(tokenize_strategy, [te], tokens)
                self._uncond_embeds = (outputs[0].cpu(), outputs[1].cpu())
        embeds, mask = self._uncond_embeds
        return embeds.to(device, dtype=weight_dtype), mask.to(device)

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
        cond_masks = batch.get("conditioning_masks")  # [B, H, W] pixel resolution, 1 = hole (ref_mask_source alpha, latent space)
        if cond_latents is None:
            cond_images = batch.get("conditioning_images")
            assert cond_images is not None, "conditioning images not found in batch / バッチにconditioning imageがありません"
            if args.ref_mask_source == "random" and args.ref_mask_space == "pixel":
                cond_images = apply_random_reference_masks_pixel(cond_images, args.ref_mask_fill_pixel)
            if vae.device != accelerator.device:
                vae.to(accelerator.device)  # the VAE is on the CPU when the target latents are cached: keep it on the GPU from now on
            with torch.no_grad():
                cond_latents = self.encode_images_to_latents(args, vae, cond_images.to(accelerator.device, dtype=vae_dtype))
        else:
            cond_latents = cond_latents.to(accelerator.device)

        ref_latents = self.shift_scale_latents(args, cond_latents)  # patchify + normalize, same as the target latents

        # Latent space mask: blend the normalized reference latents with the fill value using the smooth mask
        if args.ref_mask_space == "latent" and args.ref_mask_source != "none":
            if args.ref_mask_source == "random":
                height, width = cond_latents.shape[-2] * 8, cond_latents.shape[-1] * 8  # pixel size of the reference
                cond_masks = generate_random_reference_masks(cond_latents.shape[0], height, width)
            assert cond_masks is not None, "conditioning masks not found in batch / バッチにconditioning maskがありません"
            mask_latents = moss_image_train_utils.downsample_mask_to_latents(cond_masks.to(accelerator.device))
            ref_latents = moss_image_train_utils.apply_reference_latent_mask(ref_latents, mask_latents, args.ref_mask_fill_latent)

        batch["ref_latents"] = ref_latents.to(dtype=weight_dtype)

        if args.ref_mode == "kv":
            # the empty prompt for the reference forward (must be taken before the caption dropout replaces the list)
            batch["ref_uncond_embeds"], batch["ref_uncond_mask"] = self.get_uncond_embeds(
                batch, text_encoders, text_encoding_strategy, tokenize_strategy, accelerator.device, weight_dtype
            )

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

        if args.ref_mode == "kv":
            # one-way reference: K/V of the clean reference (empty prompt, fixed timestep), then the generation forward
            # attends to them. --ref_kv_lora: on = LoRA applied and trained through the reference forward,
            # on_no_grad = applied but detached, off = base model (LoRA disabled)
            with accelerator.autocast():
                ref_kv = moss_image_train_utils.compute_reference_kv(
                    accelerator.unwrap_model(dit),
                    ref,
                    args.ref_timestep,
                    batch["ref_uncond_embeds"],
                    batch["ref_uncond_mask"],
                    args.ref_rope_r,
                    self.ref_kv_blocks,
                    self.lora_disabled if args.ref_kv_lora == "off" else None,
                    enable_grad=is_train and args.ref_kv_lora == "on",
                )
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                return dit(noisy_model_input, timesteps, prompt_embeds, attn_mask, ref_kv=ref_kv)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            return dit(noisy_model_input, timesteps, prompt_embeds, attn_mask, ref=ref, ref_rope_r=args.ref_rope_r)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]  # compatibility
        te = self.get_models_for_text_encoding(args, accelerator, text_encoders)
        te = te[0] if te is not None else None

        text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
        tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
        moss_image_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            te,
            tokenize_strategy,
            text_encoding_strategy,
            self.sample_prompts_te_outputs,
            ref_kv_context=self.lora_disabled if args.ref_kv_lora == "off" else None,
        )

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_ref_mode"] = args.ref_mode
        metadata["ss_ref_timestep"] = args.ref_timestep
        metadata["ss_ref_kv_blocks"] = args.ref_kv_blocks
        metadata["ss_ref_kv_lora"] = args.ref_kv_lora
        metadata["ss_ref_dropout_rate"] = args.ref_dropout_rate
        metadata["ss_ref_rope_r"] = args.ref_rope_r
        metadata["ss_cache_ref_latents"] = bool(args.cache_ref_latents)
        metadata["ss_ref_mask_source"] = args.ref_mask_source
        metadata["ss_ref_mask_space"] = args.ref_mask_space
        metadata["ss_ref_mask_fill_pixel"] = args.ref_mask_fill_pixel
        metadata["ss_ref_mask_fill_latent"] = args.ref_mask_fill_latent


def generate_random_reference_masks(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Random inpainting holes (library.mask_generator.random_mask) at pixel resolution: [B, H, W] float, 1 = hole"""
    masks = np.empty((batch_size, height, width), dtype=np.float32)
    for i in range(batch_size):
        mask = random_mask(width, height)  # PIL "L", 255 = inpaint
        masks[i] = np.asarray(mask, dtype=np.float32) >= 127.5
    return torch.from_numpy(masks)


def apply_random_reference_masks_pixel(images: torch.Tensor, fill: int) -> torch.Tensor:
    """Cut a random inpainting hole into each reference image [B, 3, H, W] (-1..1) and fill it with the gray level `fill`
    (0-255): the pixel space variant (--ref_mask_space pixel)."""
    masks = generate_random_reference_masks(images.shape[0], images.shape[-2], images.shape[-1]).to(images.device)
    value = fill / 127.5 - 1.0
    return torch.where(masks[:, None] > 0.5, torch.full_like(images, value), images)


def add_moss_image_edit_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ref_mode",
        type=str,
        default="concat",
        choices=["concat", "kv"],
        help="how the reference is given to the DiT. concat (default): the reference tokens are concatenated to the sequence"
        " (bidirectional, Kontext style). kv: one-way reference with cached K/V: the clean reference is run through the DiT"
        " once (empty prompt, --ref_timestep, LoRA off, no grad) and the generation attends to its per-block keys / values."
        " The same mode must be used at inference (metadata ss_ref_mode)"
        " / 参照の与え方。concat（デフォルト）: 参照トークンをシーケンスに連結（双方向、Kontext 型）。kv: 一方向の KV cache 参照。"
        "clean な参照を DiT に 1 回通し（空プロンプト、--ref_timestep、LoRA off、no grad）、生成側が block 毎の K/V に注意を向ける。"
        "推論時も同じモードを使うこと（メタデータ ss_ref_mode）",
    )
    parser.add_argument(
        "--ref_timestep",
        type=float,
        default=0.0,
        help="timestep (sigma * 1000, 0-1000) of the reference forward for --ref_mode kv (default 0 = clean). The reference"
        " latents themselves are not noised"
        " / --ref_mode kv の参照 forward の timestep（sigma × 1000、0〜1000、デフォルト 0 = clean）。参照 latent 自体にノイズは乗せない",
    )
    parser.add_argument(
        "--ref_kv_lora",
        type=str,
        default="on",
        choices=["on", "on_no_grad", "off"],
        help="LoRA in the reference forward of --ref_mode kv. on (default): the LoRA is applied and trained through the reference"
        " forward (the reference features are optimized; matches inference with a LoRA-merged DiT such as ComfyUI)."
        " on_no_grad: applied but detached (cheaper, the reference features only drift). off: reference K/V of the base model"
        " (ablation; the inference must compute the reference K/V without the LoRA: --ref_kv_without_lora of the inference script)"
        " / --ref_mode kv の参照 forward での LoRA。on（デフォルト）: LoRA を適用し参照 forward 経由でも学習する（参照特徴が最適化される。"
        "ComfyUI 等の LoRA マージ済み DiT での推論と一致）。on_no_grad: 適用するが勾配は流さない（軽いが参照特徴は副作用で変わるだけ）。"
        "off: ベースモデルの参照 K/V（ablation。推論側は LoRA なしで参照 K/V を計算する必要がある: 推論スクリプトの --ref_kv_without_lora）",
    )
    parser.add_argument(
        "--ref_kv_blocks",
        type=str,
        default=None,
        help="comma separated indices of the DiT blocks which attend to the reference K/V for --ref_mode kv (default: all 42"
        " blocks). The reference forward stops after the last one"
        " / --ref_mode kv で参照 K/V に注意を向ける DiT block の番号（カンマ区切り、デフォルト: 全 42 block）。参照 forward は最後の block で止まる",
    )
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
        "--ref_mask_source",
        type=str,
        default="none",
        choices=["none", "alpha", "random"],
        help="hole (mask) of the reference image for inpainting-like tasks. none (default): no mask. alpha: the alpha channel of the"
        " RGBA reference images (255 = hole). random: a random hole (library.mask_generator) at every step, the reference images"
        " are typically the target images themselves (conditioning_data_dir = image_dir)"
        " / inpainting 系タスク用の参照画像の穴（マスク）。none（デフォルト）: マスクなし。alpha: RGBA の参照画像のアルファ（255 = 穴）。"
        "random: 毎ステップ library.mask_generator でランダムな穴を開ける（参照画像は通常、目標画像そのもの、conditioning_data_dir = image_dir）",
    )
    parser.add_argument(
        "--ref_mask_space",
        type=str,
        default="latent",
        choices=["latent", "pixel"],
        help="where the hole is applied. latent (default): the clean reference latents (cacheable) are blended with"
        " --ref_mask_fill_latent using the mask downsampled to the latent resolution (smooth); the context outside the hole stays"
        " exact. pixel: the hole is filled with the gray level --ref_mask_fill_pixel before the VAE (random masks cannot be cached)"
        " / 穴を適用する空間。latent（デフォルト）: clean な参照 latent（キャッシュ可）を、latent 解像度に縮小した滑らかなマスクで"
        " --ref_mask_fill_latent と混合する。穴の外側の文脈は正確に残る。pixel: VAE の前にグレー値 --ref_mask_fill_pixel で塗る"
        "（ランダムマスクはキャッシュ不可）",
    )
    parser.add_argument(
        "--ref_mask_fill_pixel",
        type=int,
        default=128,
        help="gray level (0-255) of the hole for --ref_mask_space pixel (default 128) / --ref_mask_space pixel での穴のグレー値（デフォルト 128）",
    )
    parser.add_argument(
        "--ref_mask_fill_latent",
        type=float,
        default=0.0,
        help="fill value of the hole in the normalized latent space for --ref_mask_space latent (default 0.0 = the mean latent)"
        " / --ref_mask_space latent での穴の値（正規化後の latent 空間、デフォルト 0.0 = 平均）",
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
