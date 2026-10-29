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
# with a LoRA-merged DiT, e.g. ComfyUI). on_no_grad: applied but not trained through it. --ref_kv_blocks restricts the
# blocks which see the reference (all by default).
#
# --ref_gate (Step C, with --ref_mode kv): a learned gate table g[block, timestep band] decides how much each block attends to
# the reference K/V in each band (library/moss_image_ref_gate.py). The table is trained with the LoRA (own learning rate
# --ref_gate_lr) and saved into the LoRA file (keys `ref_gate.*`). --ref_gate_budget K adds a hinge penalty so that at most
# about K blocks per band stay open. The bands (--ref_gate_bands) are trained with the normal timestep distribution
# (--timestep_sampling etc.), so the share of the samples per band follows that distribution.
# Second stage: --ref_gate_binarize loads the table (and its bands) from --network_weights, keeps the top K blocks per band
# (frozen) and fine-tunes the LoRA with the static gates (the same gates are used at inference: closed blocks skip the
# reference).
#
# Inpainting-like tasks (--ref_mask_source alpha | random): the hole of the reference is given by the alpha channel of the
# RGBA reference image (255 = hole) or cut randomly at every step (library.mask_generator). The hole is applied in the
# latent space: the clean reference latents (cacheable) are blended with a constant fill value (--ref_mask_fill_latent)
# using the mask downsampled to the latent resolution (smooth), so the context outside the hole stays exact up to the
# boundary.
#
# --replica_devices (single-process multi-GPU, library/replica_trainer.py) is supported with --cache_ref_latents: every
# replica network gets its own reference gate table (created in `create_network`, synchronized like the LoRA weights).

import argparse
from typing import List, Optional, Union

import numpy as np
import torch
from accelerate import Accelerator

from library import moss_image_models, moss_image_ref_gate, strategy_base, strategy_moss_image
from library.moss_image_ref_gate import ReferenceGate
import library.args as args_util
from library.controlnet_dataset import ControlNetDataset
from library.dataset import DatasetGroup, MinimalDataset
from library.device_utils import clean_memory_on_device
from library.mask_generator import random_mask
import library.replica_trainer as replica_trainer
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
        self.ref_gate: Optional[ReferenceGate] = None  # learned reference gate table (Step C) of the master network
        self._ref_gates: List[ReferenceGate] = []  # the gate tables of all networks (master and replicas), see create_network
        self._ref_gate_prepared = False
        self._band_edges: Optional[List[float]] = None  # band edges of the gate tables, see resolve_band_edges

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

        if getattr(args, "replica_devices", None) is not None:
            # the VAE is not replicated (library/replica_trainer.py): the reference latents must come from the cache
            assert args.cache_ref_latents, (
                "--replica_devices requires --cache_ref_latents (the VAE is not replicated)"
                " / --replica_devices には --cache_ref_latents が必要です（VAE はレプリカに複製されません）"
            )

        if args.ref_mode == "kv":
            assert args.blocks_to_swap is None or args.blocks_to_swap == 0, (
                "ref_mode kv does not support blocks_to_swap / ref_mode kv は blocks_to_swap と併用できません"
            )
            self.ref_kv_blocks = moss_image_train_utils.parse_skip_layers(args.ref_kv_blocks) or None
            if self.ref_kv_blocks is not None:
                logger.info(f"reference K/V is used in blocks: {sorted(self.ref_kv_blocks)}")

        if args.ref_gate or args.ref_gate_binarize:
            assert args.ref_mode == "kv", "ref_gate requires ref_mode kv / ref_gate には ref_mode kv が必要です"
            assert self.ref_kv_blocks is None, (
                "ref_gate and ref_kv_blocks cannot be used together (the gate decides the blocks)"
                " / ref_gate と ref_kv_blocks は併用できません（block は gate が決めます）"
            )
            assert args.attn_mode != "sageattn" or args.ref_gate_binarize, (
                "the continuous reference gate (bias dimension, head_dim 136) is not supported by sageattn"
                " / 連続 gate（bias 次元、head_dim 136）は sageattn では使えません"
            )
            if args.ref_gate_binarize:
                assert args.network_weights is not None and args.ref_gate_budget is not None, (
                    "ref_gate_binarize needs network_weights (a LoRA file with the trained gate table) and ref_gate_budget"
                    " / ref_gate_binarize には network_weights（gate 表を含む LoRA ファイル）と ref_gate_budget が必要です"
                )
            args.ref_gate = True

        for group in [train_dataset_group, val_dataset_group]:
            if group is None or not isinstance(group, DatasetGroup):
                continue
            for dataset in group.datasets:
                assert isinstance(dataset, ControlNetDataset), (
                    "reference editing training requires ControlNet dataset (conditioning_data_dir with reference images)"
                    " / 参照編集学習にはControlNetデータセット（参照画像のconditioning_data_dir）が必要です"
                )
                if args.ref_mask_source == "alpha":
                    dataset.cond_load_alpha = True  # the dataset returns the hole mask (pixel resolution) with the reference

    def create_network(self, args, network_module, vae, text_encoder, unet, net_kwargs: dict):
        """The reference gate table (--ref_gate) is attached to every network created: the master's in train() and, with
        --replica_devices, the copy of each replica (library/replica_trainer.py builds them with this method)."""
        network = super().create_network(args, network_module, vae, text_encoder, unet, net_kwargs)
        if network is None or not args.ref_gate:
            return network

        num_blocks = len(unet.layers)
        band_edges = self.resolve_band_edges(args)
        # the binarized table is a frozen buffer; the learned table is a parameter with its own learning rate.
        # Both are saved into the LoRA file as `ref_gate.*` and loaded from --network_weights by network.load_weights
        gate = ReferenceGate(num_blocks, band_edges, trainable=not args.ref_gate_binarize)
        network.add_extra_trainable_module("ref_gate", gate, args.ref_gate_lr)
        self._ref_gates.append(gate)
        if len(self._ref_gates) == 1:
            logger.info(
                f"reference gate: {num_blocks} blocks x {gate.num_bands} bands {gate.band_ranges()}, "
                f"budget {args.ref_gate_budget}, binarize: {args.ref_gate_binarize}"
            )
        return network

    def resolve_band_edges(self, args) -> List[float]:
        """The band edges of the gate table: --ref_gate_bands, or in the second stage (--ref_gate_binarize) the edges saved
        with the table in --network_weights (the table is loaded into the same shape; --ref_gate_bands must match if given)."""
        if self._band_edges is not None:
            return self._band_edges
        edges = moss_image_ref_gate.parse_band_edges(args.ref_gate_bands)
        if args.ref_gate_binarize:
            saved = moss_image_ref_gate.load_band_edges(args.network_weights)
            assert saved is not None, (
                f"no reference gate table (ref_gate.*) in network_weights: {args.network_weights}"
                f" / network_weights に参照 gate 表（ref_gate.*）がありません: {args.network_weights}"
            )
            if args.ref_gate_bands is not None and edges != saved:
                raise ValueError(
                    f"ref_gate_bands {moss_image_ref_gate.format_band_edges(edges)} does not match the bands of the gate table"
                    f" in network_weights {moss_image_ref_gate.format_band_edges(saved)}: omit ref_gate_bands in the second stage"
                    f" / ref_gate_bands が network_weights の gate 表の帯と一致しません。第 2 段階では ref_gate_bands を省略してください"
                )
            edges = saved
        self._band_edges = edges
        return edges

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        super().post_process_network(args, accelerator, network, text_encoders, unet)
        self.network = network
        if args.ref_gate:
            self.ref_gate = self.ref_gate_of(accelerator, network)
            assert self.ref_gate is not None, (
                "the network has no reference gate table: --ref_gate needs a network created by create_network"
                " / network に参照 gate 表がありません（--ref_gate には create_network で作られた network が必要です）"
            )

    def ref_gate_of(self, accelerator, network) -> Optional[ReferenceGate]:
        """The gate table of `network` (the master's or a replica's), None without --ref_gate"""
        if self.ref_gate is None and not self._ref_gates:
            return None
        network = accelerator.unwrap_model(network) if accelerator is not None and network is not None else network
        gate = getattr(network, "ref_gate", None)
        return gate if gate is not None else self.ref_gate

    def prepare_ref_gate_after_load(self, args):
        """--ref_gate_binarize: the table loaded from --network_weights is binarized (top budget blocks per band) once the
        weights are loaded. The frozen table is not a parameter, so the replicas (--replica_devices) do not receive it with
        the weight synchronization: it is copied to their tables here."""
        if self.ref_gate is None or self._ref_gate_prepared:
            return
        self._ref_gate_prepared = True
        if not args.ref_gate_binarize:
            return
        if not self.ref_gate.is_binarized():
            self.ref_gate.binarize_(args.ref_gate_budget)
            logger.info("reference gate binarized:")
            print(self.ref_gate.table_string())  # print: the rich log handler would wrap the table
        with torch.no_grad():
            for gate in self._ref_gates:
                if gate is not self.ref_gate:
                    gate.logits.copy_(self.ref_gate.logits.to(gate.logits.device))

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
            if vae.device != accelerator.device:
                vae.to(accelerator.device)  # the VAE is on the CPU when the target latents are cached: keep it on the GPU from now on
            with torch.no_grad():
                cond_latents = self.encode_images_to_latents(args, vae, cond_images.to(accelerator.device, dtype=vae_dtype))
        else:
            cond_latents = cond_latents.to(accelerator.device)

        ref_latents = self.shift_scale_latents(args, cond_latents)  # patchify + normalize, same as the target latents

        # Latent space mask: blend the normalized reference latents with the fill value using the smooth mask
        if args.ref_mask_source != "none":
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

        loss = super().process_batch(
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

        # budget penalty of the learned reference gate (training only, not for the binarized table)
        gate = self.ref_gate_of(accelerator, network)
        if is_train and gate is not None and gate.trainable and args.ref_gate_budget is not None:
            loss = loss + args.ref_gate_budget_weight * gate.budget_penalty(args.ref_gate_budget).to(loss.device)
        return loss

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
        network,
        is_train: bool,
    ) -> torch.Tensor:
        # `network` and its gate table belong to the device of this micro-step (master or replica)
        network = accelerator.unwrap_model(network) if network is not None else None
        ref = batch["ref_latents"].to(device=noisy_model_input.device, dtype=noisy_model_input.dtype)

        # Reference dropout: replace the reference with zeros so that the model also learns the no-reference case
        # (enables reference-free CFG at inference)
        if is_train and args.ref_dropout_rate > 0.0:
            drop = torch.rand(ref.shape[0], device=ref.device) < args.ref_dropout_rate
            ref = torch.where(drop.view(-1, 1, 1, 1), torch.zeros_like(ref), ref)

        if args.ref_mode == "kv":
            # one-way reference: K/V of the clean reference (empty prompt, fixed timestep), then the generation forward
            # attends to them. --ref_kv_lora: on = LoRA applied and trained through the reference forward,
            # on_no_grad = applied but detached
            with accelerator.autocast():
                ref_kv = moss_image_train_utils.compute_reference_kv(
                    accelerator.unwrap_model(dit),
                    ref,
                    args.ref_timestep,
                    batch["ref_uncond_embeds"],
                    batch["ref_uncond_mask"],
                    args.ref_rope_r,
                    self.ref_kv_blocks,
                    enable_grad=is_train and args.ref_kv_lora == "on",
                )
            ref_gate = None
            gate = self.ref_gate_of(None, network)
            if gate is not None:
                # learned gate: per-sample log gates of the band of each timestep (bias path), binarized tables resolve to
                # plain concatenation / skipping of the reference per block
                ref_kv, ref_gate = moss_image_ref_gate.resolve_block_gates(ref_kv, gate.log_gates(timesteps))
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                return dit(noisy_model_input, timesteps, prompt_embeds, attn_mask, ref_kv=ref_kv, ref_gate=ref_gate)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            return dit(noisy_model_input, timesteps, prompt_embeds, attn_mask, ref=ref, ref_rope_r=args.ref_rope_r)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        if args.ref_mode == "kv" and self.sample_prompts_te_outputs is not None and "" not in self.sample_prompts_te_outputs:
            # the reference forward needs the empty prompt: take it from the cached outputs of the batches if the sample
            # prompt cache does not have it (older caches)
            if self._uncond_embeds is not None:
                self.sample_prompts_te_outputs[""] = list(self._uncond_embeds)
        if self.ref_gate is not None:
            self.prepare_ref_gate_after_load(args)  # binarize once the weights are loaded (first call is before training)
            if moss_image_train_utils.should_sample_images(args, epoch, global_step):
                logger.info(f"reference gate table (step {global_step}):")
                print(self.ref_gate.table_string())
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
            ref_gate=self.ref_gate,
        )

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train: bool = True):
        if self.ref_gate is not None:
            self.prepare_ref_gate_after_load(args)
        return super().on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train)

    def update_metadata(self, metadata, args):
        super().update_metadata(metadata, args)
        metadata["ss_ref_mode"] = args.ref_mode
        metadata["ss_ref_timestep"] = args.ref_timestep
        metadata["ss_ref_kv_blocks"] = args.ref_kv_blocks
        metadata["ss_ref_kv_lora"] = args.ref_kv_lora
        metadata["ss_ref_gate"] = bool(args.ref_gate)
        if args.ref_gate:
            metadata["ss_ref_gate_bands"] = moss_image_ref_gate.format_band_edges(self.resolve_band_edges(args))
            metadata["ss_ref_gate_lr"] = args.ref_gate_lr
            metadata["ss_ref_gate_budget"] = args.ref_gate_budget
            metadata["ss_ref_gate_budget_weight"] = args.ref_gate_budget_weight
            metadata["ss_ref_gate_binarize"] = bool(args.ref_gate_binarize)
        metadata["ss_ref_dropout_rate"] = args.ref_dropout_rate
        metadata["ss_ref_rope_r"] = args.ref_rope_r
        metadata["ss_cache_ref_latents"] = bool(args.cache_ref_latents)
        metadata["ss_ref_mask_source"] = args.ref_mask_source
        metadata["ss_ref_mask_fill_latent"] = args.ref_mask_fill_latent


def generate_random_reference_masks(batch_size: int, height: int, width: int) -> torch.Tensor:
    """Random inpainting holes (library.mask_generator.random_mask) at pixel resolution: [B, H, W] float, 1 = hole"""
    masks = np.empty((batch_size, height, width), dtype=np.float32)
    for i in range(batch_size):
        mask = random_mask(width, height)  # PIL "L", 255 = inpaint
        masks[i] = np.asarray(mask, dtype=np.float32) >= 127.5
    return torch.from_numpy(masks)


def add_moss_image_edit_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--ref_mode",
        type=str,
        default="kv",
        choices=["kv", "concat"],
        help="how the reference is given to the DiT. kv (default): one-way reference with cached K/V: the clean reference is"
        " run through the DiT once per step (empty prompt, --ref_timestep) and the generation attends to its per-block keys /"
        " values (at inference the reference K/V are computed once per image). concat: the reference tokens are concatenated"
        " to the sequence (bidirectional, Kontext style, slower at inference). The same mode must be used at inference"
        " (metadata ss_ref_mode)"
        " / 参照の与え方。kv（デフォルト）: 一方向の KV cache 参照。clean な参照を DiT に 1 回通し（空プロンプト、--ref_timestep）、"
        "生成側が block 毎の K/V に注意を向ける（推論時は参照 K/V を画像ごとに 1 回計算）。concat: 参照トークンをシーケンスに連結"
        "（双方向、Kontext 型、推論は遅い）。推論時も同じモードを使うこと（メタデータ ss_ref_mode）",
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
        choices=["on", "on_no_grad"],
        help="LoRA in the reference forward of --ref_mode kv. on (default): the LoRA is applied and trained through the reference"
        " forward (the reference features are optimized; matches inference with a LoRA-merged DiT such as ComfyUI)."
        " on_no_grad: applied but detached (cheaper, the reference features only drift)"
        " / --ref_mode kv の参照 forward での LoRA。on（デフォルト）: LoRA を適用し参照 forward 経由でも学習する（参照特徴が最適化される。"
        "ComfyUI 等の LoRA マージ済み DiT での推論と一致）。on_no_grad: 適用するが勾配は流さない（軽いが参照特徴は副作用で変わるだけ）",
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
        "--ref_gate",
        action="store_true",
        help="learn a reference gate table g[block, timestep band] (--ref_mode kv): how much each block attends to the reference"
        " K/V in each band (additive bias log g on the reference keys, head_dim padded to 136 in training). Trained with the"
        " LoRA, saved into the LoRA file (ref_gate.*)"
        " / 参照 gate 表 g[block, timestep 帯] を学習する（--ref_mode kv）。各 block が各帯で参照 K/V にどれだけ注意を向けるか"
        "（参照キーの logit に log g を加算、学習時は head_dim を 136 にパディング）。LoRA と一緒に学習され LoRA ファイルに保存される（ref_gate.*）",
    )
    parser.add_argument(
        "--ref_gate_bands",
        type=str,
        default=None,
        help="upper boundaries of the timestep bands of the gate table (sigma * 1000, descending, comma separated). The number"
        f" of bands is the number of boundaries + 1. Default {moss_image_ref_gate.DEFAULT_BAND_EDGES}: bands (1000,950]"
        ' (950,850] (850,700] (700,500] (500,0]. An empty string ("") gives a single band over all timesteps. The bands are'
        " trained with the normal timestep distribution (--timestep_sampling etc.): a band which that distribution rarely"
        " hits is trained slowly. Saved with the table; not needed (must match if given) with --ref_gate_binarize"
        " / gate 表の timestep 帯の境界（sigma × 1000、降順、カンマ区切り）。帯の数は境界の数 + 1。空文字列（\"\"）で全 timestep"
        " に共通の 1 帯。各帯は通常の timestep 分布（--timestep_sampling 等）で学習されるので、その分布がほとんど当たらない帯は"
        "学習が遅い。表と一緒に保存される。--ref_gate_binarize では不要（指定する場合は一致が必要）",
    )
    parser.add_argument(
        "--ref_gate_lr", type=float, default=1e-2, help="learning rate of the gate table (default 1e-2) / gate 表の学習率（デフォルト 1e-2）"
    )
    parser.add_argument(
        "--ref_gate_budget",
        type=int,
        default=None,
        help="target number of open blocks per band: hinge penalty relu(sum_blocks g - budget) per band (default: no penalty)."
        " Also the top-k of --ref_gate_binarize"
        " / 帯ごとの開いた block 数の目標。帯ごとに relu(Σ_block g − budget) の罰則（デフォルト: 罰則なし）。--ref_gate_binarize の top-k にも使う",
    )
    parser.add_argument(
        "--ref_gate_budget_weight", type=float, default=0.1, help="weight of the budget penalty (default 0.1) / 予算罰則の重み（デフォルト 0.1）"
    )
    parser.add_argument(
        "--ref_gate_binarize",
        action="store_true",
        help="second stage: load the gate table (and its bands) from --network_weights, keep the top --ref_gate_budget blocks"
        " per band (frozen 0 / 1 gates, closed blocks skip the reference) and fine-tune the LoRA"
        " / 第 2 段階: --network_weights から gate 表（と帯）を読み、帯ごとに上位 --ref_gate_budget 個の block を残して 0/1 に"
        "固定し（閉じた block は参照を見ない）、LoRA を再学習する",
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
        "--ref_mask_fill_latent",
        type=float,
        default=0.0,
        help="fill value of the hole: the clean reference latents (cacheable) are blended with this value in the normalized"
        " latent space using the mask downsampled to the latent resolution (smooth); the context outside the hole stays exact"
        " (default 0.0 = the mean latent)"
        " / 穴の値。clean な参照 latent（キャッシュ可）を、latent 解像度に縮小した滑らかなマスクでこの値と混合する（正規化後の"
        " latent 空間）。穴の外側の文脈は正確に残る（デフォルト 0.0 = 平均）",
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
        trainer = replica_trainer.wrap_trainer_class(MossImageEditNetworkTrainer, args)()
        trainer.train(args)
