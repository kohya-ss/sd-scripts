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
        loading_dtype = None if args.fp8 else weight_dtype

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

        gemma2 = lumina_util.load_gemma2(args.gemma2, weight_dtype, "cpu")
        ae = lumina_util.load_ae(args.ae, weight_dtype, "cpu")

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

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, flux="dev")


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
