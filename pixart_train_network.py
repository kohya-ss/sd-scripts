import argparse

import torch
from library.device_utils import init_ipex, clean_memory_on_device
init_ipex()

from library import sdxl_model_util, sdxl_train_util, train_util
from library import train_util, pixart_aspect_ratios
from library import pixart_model_util, pixart_train_util

import train_network
from library.utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

class PixartNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR # vae same as in sdxl
        self.is_sdxl = True
        self.is_pixart = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        pixart_train_util.verify_pixart_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder,
            vae,
            dit,
            logit_scale,
            ckpt_info,
        ) = pixart_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        # kabachuha TODO: check args
        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # NOTE: having multiple text encoders may be handy when SD3 releases later
        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder], vae, dit

    def load_tokenizer(self, args):
        tokenizer = pixart_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, dit, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and dit to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = dit.device
                vae.to("cpu")
                dit.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            with accelerator.autocast():
                dataset.cache_text_encoder_outputs(
                    tokenizers,
                    text_encoders,
                    accelerator.device,
                    weight_dtype,
                    args.cache_text_encoder_outputs_to_disk,
                    accelerator.is_main_process,
                )

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and dit back to original device")
                vae.to(org_vae_device)
                dit.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        if "text_encoder_outputs_list" not in batch or batch["text_encoder_outputs_list"] is None:
            input_ids1 = batch["input_ids"]
            with torch.enable_grad():
                input_ids1 = input_ids1.to(accelerator.device)
                encoder_hidden_states = train_util.get_hidden_states_pixart(
                    args.max_token_length,
                    input_ids1,
                    tokenizers[0],
                    text_encoders[0],
                    None if not args.full_fp16 else weight_dtype,
                    accelerator=accelerator,
                )
        else:
            encoder_hidden_states = batch["text_encoder_outputs_list"].to(accelerator.device).to(weight_dtype)

        return encoder_hidden_states

    def call_dit(self, args, accelerator, dit, noisy_latents, timesteps, text_conds, attention_mask, batch, weight_dtype):

        aspect_ratio_table = pixart_aspect_ratios.select_aspect_ratio_table(f"ASPECT_RATIO_{args.resolution}")

        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"].cpu().numpy()
        # bz, 2

        sizes = []
        ars = []
        for h, w in orig_size:
            closest_size, closest_ratio = pixart_train_util.get_closest_ratio(h, w, aspect_ratio_table)
            sizes.append(torch.Tensor(closest_size))
            ars.append(torch.Tensor([closest_ratio]))
        resolution = torch.stack(sizes)
        aspect_ratio = torch.stack(ars)

        # Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if getattr(dit, 'module', dit).config.sample_size == 128 and args.micro_conditions:
            resolution = resolution.to(dtype=weight_dtype, device=noisy_latents.device)
            aspect_ratio = aspect_ratio.to(dtype=weight_dtype, device=noisy_latents.device)
            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        # Predict the noise residual and compute loss
        noise_pred = dit(noisy_latents, encoder_hidden_states=text_conds,
                                    encoder_attention_mask=attention_mask,
                                    timestep=timesteps,
                                    added_cond_kwargs=added_cond_kwargs).sample.chunk(2, 1)[0]

        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        pixart_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    pixart_train_util.add_pixart_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = PixartNetworkTrainer()
    trainer.train(args)
