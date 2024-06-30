import warnings

warnings.filterwarnings("ignore")

import argparse

import torch
from diffusers import DDPMScheduler
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import (
    hunyuan_models,
    hunyuan_utils,
    sdxl_model_util,
    sdxl_train_util,
    train_util,
)
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class HunYuanNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        # sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = hunyuan_utils.load_target_model(
            args,
            accelerator,
            hunyuan_models.MODEL_VERSION_HUNYUAN_V1_1 if args.use_extra_cond else hunyuan_models.MODEL_VERSION_HUNYUAN_V1_2,
            weight_dtype,
            use_extra_cond=args.use_extra_cond,
        )

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return (
            hunyuan_models.MODEL_VERSION_HUNYUAN_V1_1 if args.use_extra_cond else hunyuan_models.MODEL_VERSION_HUNYUAN_V1_2,
            [text_encoder1, text_encoder2],
            vae,
            unet,
        )

    def load_tokenizer(self, args):
        tokenizer = hunyuan_utils.load_tokenizers()
        return tokenizer

    def load_noise_scheduler(self, args):
        return DDPMScheduler(
            beta_start=0.00085, 
            beta_end=args.beta_end,
            beta_schedule="scaled_linear", 
            num_train_timesteps=1000, 
            clip_sample=False,
            steps_offset=1
        )

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator,
        unet,
        vae,
        tokenizers,
        text_encoders,
        dataset: train_util.DatasetGroup,
        weight_dtype,
    ):
        if args.cache_text_encoder_outputs:
            raise NotImplementedError
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(
        self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype
    ):
        if (
            "text_encoder_outputs1_list" not in batch
            or batch["text_encoder_outputs1_list"] is None
        ):
            input_ids1 = batch["input_ids"]
            input_ids2 = batch["input_ids2"]
            logger.debug("input_ids1", input_ids1.shape)
            logger.debug("input_ids2", input_ids2.shape)
            with torch.enable_grad():
                input_ids1 = input_ids1.to(accelerator.device)
                input_ids2 = input_ids2.to(accelerator.device)
                encoder_hidden_states1, mask1, encoder_hidden_states2, mask2 = (
                    hunyuan_utils.hunyuan_get_hidden_states(
                        args.max_token_length,
                        input_ids1,
                        input_ids2,
                        tokenizers[0],
                        tokenizers[1],
                        text_encoders[0],
                        text_encoders[1],
                        None if not args.full_fp16 else weight_dtype,
                        accelerator=accelerator,
                    )
                )
                logger.debug("encoder_hidden_states1", encoder_hidden_states1.shape)
                logger.debug("encoder_hidden_states2", encoder_hidden_states2.shape)
        else:
            raise NotImplementedError
        return encoder_hidden_states1, mask1, encoder_hidden_states2, mask2

    def call_unet(
        self,
        args,
        accelerator,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
    ):
        noisy_latents = noisy_latents.to(
            weight_dtype
        )  # TODO check why noisy_latents is not weight_dtype
        B, C, H, W = noisy_latents.shape

        if args.use_extra_cond:
            # get size embeddings
            orig_size = batch["original_sizes_hw"]  # B, 2
            crop_size = batch["crop_top_lefts"]  # B, 2
            target_size = batch["target_sizes_hw"]  # B, 2

            style = torch.as_tensor([0] * B, device=accelerator.device)
            image_meta_size = torch.concat([orig_size, target_size, crop_size])
        else:
            style = None
            image_meta_size = None

        freqs_cis_img = hunyuan_utils.calc_rope(H * 8, W * 8, 2, 88)

        # concat embeddings
        encoder_hidden_states1, mask1, encoder_hidden_states2, mask2 = text_conds
        noise_pred = unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states1,
            text_embedding_mask=mask1,
            encoder_hidden_states_t5=encoder_hidden_states2,
            text_embedding_mask_t5=mask2,
            image_meta_size=image_meta_size,
            style=style,
            cos_cis_img=freqs_cis_img[0],
            sin_cis_img=freqs_cis_img[1],
        )
        # TODO Handle learned sigma correctly
        return noise_pred.chunk(2, dim=1)[0]

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
    ):
        steps = global_step
        if steps == 0:
            if not args.sample_at_first:
                return
        else:
            if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
                return
            if args.sample_every_n_epochs is not None:
                # sample_every_n_steps は無視する
                if epoch is None or epoch % args.sample_every_n_epochs != 0:
                    return
            else:
                if (
                    steps % args.sample_every_n_steps != 0 or epoch is not None
                ):  # steps is not divisible or end of epoch
                    return
        logger.warning("Sampling images not supported yet.")


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    hunyuan_utils.add_hydit_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = HunYuanNetworkTrainer()
    trainer.train(args)
