import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    lumina_models,
    flux_train_utils,
    lumina_util,
    lumina_train_util,
    sd3_train_utils,
    strategy_base,
    strategy_lumina,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class LuminaNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(self, args, train_dataset_group, val_dataset_group):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if (
            args.cache_text_encoder_outputs_to_disk
            and not args.cache_text_encoder_outputs
        ):
            logger.warning("Enabling cache_text_encoder_outputs due to disk caching")
            args.cache_text_encoder_outputs = True

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

        self.train_gemma2 = not args.network_train_unet_only

    def load_target_model(self, args, weight_dtype, accelerator):
        loading_dtype = None if args.fp8_base else weight_dtype

        model = lumina_util.load_lumina_model(
            args.pretrained_model_name_or_path,
            loading_dtype,
            "cpu",
            disable_mmap=args.disable_mmap_load_safetensors,
        )

        # if args.blocks_to_swap:
        #     logger.info(f'Enabling block swap: {args.blocks_to_swap}')
        #     model.enable_block_swap(args.blocks_to_swap, accelerator.device)
        #     self.is_swapping_blocks = True

        gemma2 = lumina_util.load_gemma2(
            args.gemma2, weight_dtype, "cpu"
        )
        ae = lumina_util.load_ae(
            args.ae, weight_dtype, "cpu"
        )

        return lumina_util.MODEL_VERSION_LUMINA_V2, [gemma2], ae, model

    def get_tokenize_strategy(self, args):
        return strategy_lumina.LuminaTokenizeStrategy(
            args.gemma2_max_token_length, args.tokenizer_cache_dir
        )

    def get_tokenizers(self, tokenize_strategy: strategy_lumina.LuminaTokenizeStrategy):
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_lumina.LuminaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, False
        )

    def get_text_encoding_strategy(self, args):
        return strategy_lumina.LuminaTextEncodingStrategy(args.apply_gemma2_attn_mask)

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [self.train_gemma2]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            # if the text encoders is trained, we need tokenization, so is_partial is True
            return strategy_lumina.LuminaTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                is_partial=self.train_gemma2,
                apply_gemma2_attn_mask=args.apply_gemma2_attn_mask,
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator: Accelerator,
        unet,
        vae,
        text_encoders,
        dataset,
        weight_dtype,
    ):
        for text_encoder in text_encoders:
            text_encoder_outputs_caching_strategy = (
                self.get_text_encoder_outputs_caching_strategy(args)
            )
            if text_encoder_outputs_caching_strategy is not None:
                text_encoder_outputs_caching_strategy.cache_batch_outputs(
                    self.get_tokenize_strategy(args),
                    [text_encoder],
                    self.get_text_encoding_strategy(args),
                    dataset,
                )

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        ae,
        tokenizer,
        text_encoder,
        lumina,
    ):
        lumina_train_util.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            lumina,
            ae,
            self.get_models_for_text_encoding(args, accelerator, text_encoder),
            self.sample_prompts_te_outputs,
        )

    # Remaining methods maintain similar structure to flux implementation
    # with Lumina-specific model calls and strategies

    def get_noise_scheduler(
        self, args: argparse.Namespace, device: torch.device
    ) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=args.discrete_flow_shift
        )
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        return vae.encode(images)

    # not sure, they use same flux vae
    def shift_scale_latents(self, args, latents):
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: lumina_models.NextDiT,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # get noisy model input and timesteps
        noisy_model_input, timesteps, sigmas = (
            flux_train_utils.get_noisy_model_input_and_timesteps(
                args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
            )
        )

        # pack latents and get img_ids - 这部分可以保留因为NextDiT也需要packed格式的输入
        packed_noisy_model_input = lumina_util.pack_latents(noisy_model_input)
        packed_latent_height, packed_latent_width = (
            noisy_model_input.shape[2] // 2,
            noisy_model_input.shape[3] // 2,
        )

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                if t is not None and t.dtype.is_floating_point:
                    t.requires_grad_(True)

        # Unpack Gemma2 outputs
        gemma2_hidden_states, gemma2_attn_mask, input_ids = text_encoder_conds
        if not args.apply_gemma2_attn_mask:
            gemma2_attn_mask = None

        def call_dit(img, gemma2_hidden_states, input_ids, timesteps, gemma2_attn_mask):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                # NextDiT forward expects (x, t, cap_feats, cap_mask)
                model_pred = unet(
                    x=img,  # packed latents
                    t=timesteps / 1000,  # timesteps需要除以1000来匹配模型预期
                    cap_feats=gemma2_hidden_states,  # Gemma2的hidden states作为caption features
                    cap_mask=gemma2_attn_mask,  # Gemma2的attention mask
                )
            return model_pred

        model_pred = call_dit(
            img=packed_noisy_model_input,
            gemma2_hidden_states=gemma2_hidden_states,
            input_ids=input_ids,
            timesteps=timesteps,
            gemma2_attn_mask=gemma2_attn_mask,
        )

        # unpack latents
        model_pred = lumina_util.unpack_latents(
            model_pred, packed_latent_height, packed_latent_width
        )

        # apply model prediction type
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(
            args, model_pred, noisy_model_input, sigmas
        )

        # flow matching loss: this is different from SD3
        target = noise - latents

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if (
                    "diff_output_preservation" in custom_attributes
                    and custom_attributes["diff_output_preservation"]
                ):
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                with torch.no_grad():
                    model_pred_prior = call_dit(
                        img=packed_noisy_model_input[diff_output_pr_indices],
                        gemma2_hidden_states=gemma2_hidden_states[
                            diff_output_pr_indices
                        ],
                        input_ids=input_ids[diff_output_pr_indices],
                        timesteps=timesteps[diff_output_pr_indices],
                        gemma2_attn_mask=(
                            gemma2_attn_mask[diff_output_pr_indices]
                            if gemma2_attn_mask is not None
                            else None
                        ),
                    )
                network.set_multiplier(1.0)

                model_pred_prior = lumina_util.unpack_latents(
                    model_pred_prior, packed_latent_height, packed_latent_width
                )
                model_pred_prior, _ = flux_train_utils.apply_model_prediction_type(
                    args,
                    model_pred_prior,
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(
            None, args, False, True, False, lumina="lumina2"
        )

    def update_metadata(self, metadata, args):
        metadata["ss_apply_gemma2_attn_mask"] = args.apply_gemma2_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        text_encoder.model.embed_tokens.requires_grad_(True)

    def prepare_text_encoder_fp8(
        self, index, text_encoder, te_weight_dtype, weight_dtype
    ):
        logger.info(
            f"prepare Gemma2 for fp8: set to {te_weight_dtype}, set embeddings to {weight_dtype}"
        )
        text_encoder.to(te_weight_dtype)  # fp8
        text_encoder.model.embed_tokens.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        # if we doesn't swap blocks, we can move the model to device
        nextdit: lumina_models.Nextdit = unet
        nextdit = accelerator.prepare(
            nextdit, device_placement=[not self.is_swapping_blocks]
        )
        accelerator.unwrap_model(nextdit).move_to_device_except_swap_blocks(
            accelerator.device
        )  # reduce peak memory usage
        accelerator.unwrap_model(nextdit).prepare_block_swap_before_forward()

        return nextdit


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    lumina_train_utils.add_lumina_train_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = LuminaNetworkTrainer()
    trainer.train(args)
