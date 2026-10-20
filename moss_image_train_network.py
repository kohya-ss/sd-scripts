# MOSS-Image LoRA training script

import argparse
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device
from library.custom_offloading_utils import BlockSwapConfig

init_ipex()

from library import (
    convrot_int8_kernels,
    convrot_int8_utils,
    flux2_vae,
    flux_train_utils,
    moss_image_models,
    moss_image_train_utils,
    moss_image_utils,
    sd3_train_utils,
    strategy_base,
    strategy_moss_image,
    sampling,
)
import library.args as args_util
import library.compile_utils as compile_utils
import library.model_io as model_io
from library.dataset import DatasetGroup, MinimalDataset
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class MossImageNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks = False
        self._use_unsloth_offload_checkpointing = False
        # BatchNorm statistics of the FLUX.2 VAE, kept so that the latents can be normalized after the VAE is freed
        self.bn_running_mean: Optional[torch.Tensor] = None
        self.bn_running_var: Optional[torch.Tensor] = None
        # effective ConvRot INT8 state after loading (pre-quantized files are detected independently of the options)
        self.dit_convrot_int8 = False
        self.text_encoder_convrot_int8 = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        flux_train_utils.log_timestep_sampling_info(args)

        assert args.text_encoder is not None, "--text_encoder (Qwen3.5-9B checkpoint) is required / --text_encoder（Qwen3.5-9B のチェックポイント）を指定してください"
        assert args.vae is not None, "--vae (FLUX.2 VAE checkpoint) is required / --vae（FLUX.2 VAE のチェックポイント）を指定してください"

        if args.fp8_base or args.fp8_base_unet:
            logger.warning(
                "fp8_base and fp8_base_unet are not supported for MOSS-Image: use --convrot_int8 (and --convrot_int8_text_encoder)"
                " instead. They are disabled."
                " / fp8_base と fp8_base_unet は MOSS-Image ではサポートされていません。代わりに --convrot_int8（と"
                " --convrot_int8_text_encoder）を使ってください。無効化します。"
            )
            args.fp8_base = False
            args.fp8_base_unet = False
        args.fp8_scaled = False

        if args.convrot_int8_bwd == "int8":
            # the triton requirement is also checked when the layers are patched; fail early here
            assert convrot_int8_kernels.HAS_TRITON, (
                "--convrot_int8_bwd int8 requires triton (triton-windows on Windows)"
                " / --convrot_int8_bwd int8 には triton（Windows では triton-windows）が必要です"
            )
            if not (args.convrot_int8 or args.convrot_int8_text_encoder):
                logger.info(
                    "--convrot_int8_bwd int8 only takes effect with a ConvRot INT8 base (--convrot_int8, --convrot_int8_text_encoder"
                    " or a pre-quantized checkpoint)"
                )

        if args.mixed_precision == "fp16":
            logger.warning(
                "mixed_precision fp16 is not recommended for MOSS-Image: the text encoder and the DiT are bf16 models. Use bf16."
                " / MOSS-Image では mixed_precision fp16 は推奨されません。Text Encoder と DiT は bf16 のモデルです。bf16 を使ってください。"
            )

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning("cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled")
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            # caption dropout and multiple caption candidates are supported by MossImageTextEncoderOutputsCachingStrategy
            assert train_dataset_group.is_text_encoder_output_cacheable(
                cache_supports_dropout=True, supports_multiple_caption_candidates=True
            ), "when caching Text Encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        if self.is_train_text_encoder(args):
            logger.info(
                "Training the Text Encoder (Qwen3.5-9B) LoRA. Note: the official recommendation is to keep the Text Encoder frozen."
                " / Text Encoder（Qwen3.5-9B）の LoRA を学習します。公式は Text Encoder の凍結を推奨しています。"
            )

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.unsloth_offload_checkpointing:
            if not args.gradient_checkpointing:
                logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
                args.gradient_checkpointing = True
            assert (
                not args.cpu_offload_checkpointing
            ), "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"
            assert (
                args.blocks_to_swap is None or args.blocks_to_swap == 0
            ), "blocks_to_swap is not supported with unsloth_offload_checkpointing"

        if args.compile:
            assert not args.torch_compile, (
                "--compile (per-block torch.compile) and --torch_compile (accelerate dynamo) cannot be used together"
                " / --compile（ブロック単位torch.compile）と--torch_compile（accelerate dynamo）は併用できません"
            )
            assert not (args.compile_fullgraph and args.split_attn), (
                "--compile_fullgraph cannot be used with --split_attn (split attention uses dynamic control flow)"
                " / --compile_fullgraphは--split_attnと併用できません（split attentionは動的な制御フローを使用します）"
            )

        # FLUX.2 VAE spatial downscale = 8 and patch size = 2
        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        # Text encoder (Qwen3.5-9B) is always bf16 (see cast_text_encoder). Loaded to the CPU, moved to the GPU when needed.
        logger.info("Loading MOSS-Image text encoder (Qwen3.5-9B)...")
        text_encoder = moss_image_utils.load_text_encoder(
            args.text_encoder,
            dtype=torch.bfloat16,
            device="cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
            convrot_int8=args.convrot_int8_text_encoder,
            convrot_int8_bwd=args.convrot_int8_bwd,
            calc_device=accelerator.device,  # quantize on the GPU, keep the weights on the CPU
        )
        text_encoder.eval()
        self.text_encoder_convrot_int8 = convrot_int8_utils.is_convrot_int8_model(text_encoder)

        # VAE (FLUX.2) is always float32 (see cast_vae)
        logger.info("Loading FLUX.2 VAE...")
        vae = flux2_vae.load_flux2_vae(args.vae, dtype=torch.float32, device="cpu", disable_mmap=args.disable_mmap_load_safetensors)
        vae.eval()
        self.bn_running_mean = vae.bn.running_mean.detach().clone()
        self.bn_running_var = vae.bn.running_var.detach().clone()

        # Return format: (model_type, text_encoders, vae, unet)
        return "moss_image", [text_encoder], vae, None  # DiT is loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        logger.info(f"Loading MOSS-Image DiT with attn_mode={attn_mode}, split_attn: {args.split_attn}...")
        model = moss_image_utils.load_moss_image_model(
            accelerator.device,
            args.pretrained_model_name_or_path,
            attn_mode,
            args.split_attn,
            loading_device,
            weight_dtype,
            disable_mmap=args.disable_mmap_load_safetensors,
            convrot_int8=args.convrot_int8,
            convrot_int8_bwd=args.convrot_int8_bwd,
        )
        self.dit_convrot_int8 = convrot_int8_utils.is_convrot_int8_model(model)

        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=...); unsloth is re-applied in
        # prepare_unet_with_accelerator.
        self._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

        if self.is_swapping_blocks:
            # The base weights are frozen in LoRA training. Freeze them here, before enabling block swap, for the H2D-only offloader.
            model.requires_grad_(False)
            swap_config = BlockSwapConfig.from_args(args, accelerator.device, supports_backward=True)
            logger.info(
                f"enable block swap: blocks_to_swap={args.blocks_to_swap}, H2D-only: {swap_config.h2d_only}, "
                f"use pinned memory: {swap_config.use_pinned_memory}"
            )
            model.enable_block_swap(args.blocks_to_swap, swap_config)

        return model, text_encoders

    # region strategies

    def get_tokenize_strategy(self, args):
        return strategy_moss_image.MossImageTokenizeStrategy(args.moss_max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_moss_image.MossImageTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_moss_image.MossImageLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )

    def get_text_encoding_strategy(self, args):
        return strategy_moss_image.MossImageTextEncodingStrategy()

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_moss_image.MossImageTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    # endregion

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None  # no text encoders needed for encoding
        return text_encoders

    def cast_text_encoder(self, args):
        return False  # the text encoder stays in bf16 regardless of the mixed precision setting

    def cast_vae(self, args):
        return False  # the VAE stays in float32

    def cast_unet(self, args):
        # The DiT is loaded in weight_dtype by load_moss_image_model. `Module.to(dtype)` would also cast the fp32
        # `scale_weight` buffers of the ConvRot INT8 layers, so the cast is skipped here.
        return False

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # the DiT is not loaded yet, so only the VAE is moved
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoder to gpu")
            text_encoders[0].to(accelerator.device)

            # no autocast: the text encoder is bf16 and the official pipeline runs it in bf16
            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # cache sample prompts
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompts: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = sampling.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"  cache TE outputs for: {p}")
                                tokens = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, text_encoders, tokens
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            logger.info("move text encoder back to cpu")
            text_encoders[0].to("cpu")

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)

            clean_memory_on_device(accelerator.device)
        else:
            # move text encoder to device for encoding during training/validation
            text_encoders[0].to(accelerator.device)

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
        )

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        return sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=moss_image_train_utils.NUM_TRAIN_TIMESTEPS, shift=args.discrete_flow_shift
        )

    def encode_images_to_latents(self, args, vae: flux2_vae.AutoEncoder, images):
        # raw latents [B, 32, H/8, W/8] (same as the cache). The VAE is float32: run outside autocast.
        with torch.autocast(device_type=vae.device.type, enabled=False):
            return vae.encode(images.to(vae.dtype))

    def shift_scale_latents(self, args, latents):
        # raw latents [B, 32, H/8, W/8] -> DiT input [B, 128, H/16, W/16] (2x2 patchify + BatchNorm normalization)
        latents = flux2_vae.patchify_latents(latents)
        return flux2_vae.normalize_latents(latents, self.bn_running_mean, self.bn_running_var)

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        dit: moss_image_models.MossImage = unet

        noise = torch.randn_like(latents)

        # Per-sample timestep sampling offset from custom_attributes: timestep_sampling = { offset = ... }
        # Only applied during training; validation uses unbiased sampling for comparable loss.
        tso = None
        if is_train and "custom_attributes" in batch:
            offsets = [ca.get("timestep_sampling", {}).get("offset", 0.0) for ca in batch["custom_attributes"]]
            t = torch.tensor(offsets, dtype=torch.float32)
            if t.abs().sum() > 0:
                tso = t

        # Timesteps and sigmas are kept in float32 (bf16 would quantize sigma * 1000 to multiples of 4). The DiT casts
        # the noisy input to its own dtype.
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, torch.float32, timestep_sampling_offset=tso
        )
        # timesteps are sigma * 1000: passed to the DiT as is (the official pipeline does not divide by 1000)

        prompt_embeds, attn_mask = text_encoder_conds[:2]  # caption_dropout_rate (cached outputs) is not needed here
        prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
        attn_mask = attn_mask.to(accelerator.device)

        # Gradient checkpointing (non-reentrant) needs an input which requires grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            prompt_embeds.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = dit(noisy_model_input, timesteps, prompt_embeds, attn_mask)

        # Rectified flow target: v = noise - x0
        target = noise - latents

        # 1 for logit_normal (the official training uses no loss weighting)
        weighting = flux_train_utils.compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)

        return model_pred, target, timesteps, weighting

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
        """Caption dropout for cached text encoder outputs: dropped samples use the cached unconditional embeddings."""
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            moss_text_encoding_strategy: strategy_moss_image.MossImageTextEncodingStrategy = text_encoding_strategy
            caption_dropout_rates = text_encoder_outputs_list[-1]
            # [prompt_embeds, attn_mask, uncond_embeds, uncond_mask] -> [prompt_embeds, attn_mask]
            text_encoder_outputs_list = moss_text_encoding_strategy.drop_cached_text_encoder_outputs(
                *text_encoder_outputs_list[:-1], caption_dropout_rates=caption_dropout_rates
            )
            # Add the caption dropout rates back to the list for validation dataset (which is re-used batch items)
            batch["text_encoder_outputs_list"] = text_encoder_outputs_list + [caption_dropout_rates]

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

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return model_io.get_sai_model_spec_dataclass(None, args, False, True, False, moss_image="base").to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_convrot_int8"] = self.dit_convrot_int8
        metadata["ss_convrot_int8_text_encoder"] = self.text_encoder_convrot_int8

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        # Set the first floating point parameter's requires_grad to True to workaround Accelerate gradient checkpointing bug.
        # The first parameter (token embedding) is int8 with ConvRot INT8 and cannot require grad.
        first_param = next(p for p in text_encoder.parameters() if p.is_floating_point())
        first_param.requires_grad_(True)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        # The base NetworkTrainer only calls enable_gradient_checkpointing(cpu_offload=True/False),
        # so we re-apply with unsloth_offload if needed (after base has already enabled it).
        if self._use_unsloth_offload_checkpointing and args.gradient_checkpointing:
            unet.enable_gradient_checkpointing(unsloth_offload=True)

        if not self.is_swapping_blocks:
            model = super().prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            model = accelerator.prepare(unet, device_placement=[not self.is_swapping_blocks])
            accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
            accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        # CUDA perf switches are independent of torch.compile; apply whenever requested.
        compile_utils.apply_cuda_optimizations(args)

        if args.compile:
            # Per-block torch.compile of the 42 main blocks. Reach the real model via unwrap_model so the underlying
            # ModuleList is mutated regardless of any DDP wrapper.
            # ConvRot INT8 Linears are also excluded: the custom autograd.Function + autotuned Triton kernels do not compile.
            dit = accelerator.unwrap_model(model)
            compile_utils.compile_transformer(
                args, dit, [dit.layers], disable_linear=self.is_swapping_blocks or convrot_int8_utils.is_convrot_int8_model(dit)
            )

        return model

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            # prepare for next forward: because backward pass is not called, we need to prepare it here
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)
    moss_image_train_utils.add_moss_image_model_arguments(parser)
    moss_image_train_utils.add_moss_image_training_arguments(parser)
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
        trainer = MossImageNetworkTrainer()
        trainer.train(args)
