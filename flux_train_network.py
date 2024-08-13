import argparse
import copy
import math
import random
from typing import Any

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import flux_models, flux_utils, sd3_train_utils, sd3_utils, sdxl_model_util, sdxl_train_util, strategy_flux, train_util
import train_network
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class FluxNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()

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

        train_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def load_target_model(self, args, weight_dtype, accelerator):
        # currently offload to cpu for some models
        name = "schnell" if "schnell" in args.pretrained_model_name_or_path else "dev"  # TODO change this to a more robust way
        # if we load to cpu, flux.to(fp8) takes a long time
        model = flux_utils.load_flow_model(name, args.pretrained_model_name_or_path, weight_dtype, "cpu")

        if args.split_mode:
            model = self.prepare_split_model(model, weight_dtype, accelerator)

        clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu")
        clip_l.eval()

        # loading t5xxl to cpu takes a long time, so we should load to gpu in future
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, weight_dtype, "cpu")
        t5xxl.eval()

        ae = flux_utils.load_ae(name, args.ae, weight_dtype, "cpu")

        return flux_utils.MODEL_VERSION_FLUX_V1, [clip_l, t5xxl], ae, model

    def prepare_split_model(self, model, weight_dtype, accelerator):
        from accelerate import init_empty_weights

        logger.info("prepare split model")
        with init_empty_weights():
            flux_upper = flux_models.FluxUpper(model.params)
            flux_lower = flux_models.FluxLower(model.params)
        sd = model.state_dict()

        # lower (trainable)
        logger.info("load state dict for lower")
        flux_lower.load_state_dict(sd, strict=False, assign=True)
        flux_lower.to(dtype=weight_dtype)

        # upper (frozen)
        logger.info("load state dict for upper")
        flux_upper.load_state_dict(sd, strict=False, assign=True)

        logger.info("prepare upper model")
        target_dtype = torch.float8_e4m3fn if args.fp8_base else weight_dtype
        flux_upper.to(accelerator.device, dtype=target_dtype)
        flux_upper.eval()

        if args.fp8_base:
            # this is required to run on fp8
            flux_upper = accelerator.prepare(flux_upper)

        flux_upper.to("cpu")

        self.flux_upper = flux_upper
        del model  # we don't need model anymore
        clean_memory_on_device(accelerator.device)

        logger.info("split model prepared")

        return flux_lower

    def get_tokenize_strategy(self, args):
        return strategy_flux.FluxTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        return [tokenize_strategy.clip_l, tokenize_strategy.t5xxl]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_flux.FluxLatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders  # + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, False, apply_t5_attn_mask=args.apply_t5_attn_mask
            )
        else:
            return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                # メモリ消費を減らす
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # When TE is not be trained, it will not be prepared so we need to use explicit autocast
            logger.info("move text encoders to gpu")
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator.is_main_process)
            accelerator.wait_for_everyone()

            logger.info("move text encoders back to cpu")
            text_encoders[0].to("cpu")  # , dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu")  # , dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    # def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
    #     noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

    #     # get size embeddings
    #     orig_size = batch["original_sizes_hw"]
    #     crop_size = batch["crop_top_lefts"]
    #     target_size = batch["target_sizes_hw"]
    #     embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

    #     # concat embeddings
    #     encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
    #     vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
    #     text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

    #     noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
    #     return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        # logger.warning("Sampling images is not supported for Flux model")
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, accelerator, vae, images):
        return vae.encode(images).latent_dist.sample()

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
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
    ):
        # copy from sd3_train.py and modified

        def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
            sigmas = self.noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
            schedule_timesteps = self.noise_scheduler_copy.timesteps.to(accelerator.device)
            timesteps = timesteps.to(accelerator.device)
            step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

            sigma = sigmas[step_indices].flatten()
            while len(sigma.shape) < n_dim:
                sigma = sigma.unsqueeze(-1)
            return sigma

        def compute_density_for_timestep_sampling(
            weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
        ):
            """Compute the density for sampling the timesteps when doing SD3 training.

            Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

            SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
            """
            if weighting_scheme == "logit_normal":
                # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
                u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
                u = torch.nn.functional.sigmoid(u)
            elif weighting_scheme == "mode":
                u = torch.rand(size=(batch_size,), device="cpu")
                u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
            else:
                u = torch.rand(size=(batch_size,), device="cpu")
            return u

        def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
            """Computes loss weighting scheme for SD3 training.

            Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

            SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
            """
            if weighting_scheme == "sigma_sqrt":
                weighting = (sigmas**-2.0).float()
            elif weighting_scheme == "cosmap":
                bot = 1 - 2 * sigmas + 2 * sigmas**2
                weighting = 2 / (math.pi * bot)
            else:
                weighting = torch.ones_like(sigmas)
            return weighting

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
            # Simple random t-based noise sampling
            if args.timestep_sampling == "sigmoid":
                # https://github.com/XLabs-AI/x-flux/tree/main
                t = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=accelerator.device))
            else:
                t = torch.rand((bsz,), device=accelerator.device)
            timesteps = t * 1000.0
            t = t.view(-1, 1, 1, 1)
            noisy_model_input = (1 - t) * latents + t * noise
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.logit_mean,
                logit_std=args.logit_std,
                mode_scale=args.mode_scale,
            )
            indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

            # Add noise according to flow matching.
            sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=weight_dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # pack latents and get img_ids
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)  # b, c, h*2, w*2 -> b, h*w, c*4
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)

        # get guidance
        guidance_vec = torch.full((bsz,), args.guidance_scale, device=accelerator.device)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            noisy_model_input.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)
            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        # Predict the noise residual
        l_pooled, t5_out, txt_ids = text_encoder_conds
        # print(
        #     f"model_input: {noisy_model_input.shape}, img_ids: {img_ids.shape}, t5_out: {t5_out.shape}, txt_ids: {txt_ids.shape}, l_pooled: {l_pooled.shape}, timesteps: {timesteps.shape}, guidance_vec: {guidance_vec.shape}"
        # )

        if not args.split_mode:
            # normal forward
            with accelerator.autocast():
                # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transformer model (we should not keep it but I want to keep the inputs same for the model for testing)
                model_pred = unet(
                    img=packed_noisy_model_input,
                    img_ids=img_ids,
                    txt=t5_out,
                    txt_ids=txt_ids,
                    y=l_pooled,
                    timesteps=timesteps / 1000,
                    guidance=guidance_vec,
                )
        else:
            # split forward to reduce memory usage
            assert network.train_blocks == "single", "train_blocks must be single for split mode"
            with accelerator.autocast():
                # move flux lower to cpu, and then move flux upper to gpu
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)
                self.flux_upper.to(accelerator.device)

                # upper model does not require grad
                with torch.no_grad():
                    intermediate_img, intermediate_txt, vec, pe = self.flux_upper(
                        img=packed_noisy_model_input,
                        img_ids=img_ids,
                        txt=t5_out,
                        txt_ids=txt_ids,
                        y=l_pooled,
                        timesteps=timesteps / 1000,
                        guidance=guidance_vec,
                    )

                # move flux upper back to cpu, and then move flux lower to gpu
                self.flux_upper.to("cpu")
                clean_memory_on_device(accelerator.device)
                unet.to(accelerator.device)

                # lower model requires grad
                intermediate_img.requires_grad_(True)
                intermediate_txt.requires_grad_(True)
                vec.requires_grad_(True)
                pe.requires_grad_(True)
                model_pred = unet(img=intermediate_img, txt=intermediate_txt, vec=vec, pe=pe)

        # unpack latents
        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)

        if args.model_prediction_type == "raw":
            # use model_pred as is
            weighting = None
        elif args.model_prediction_type == "additive":
            # add the model_pred to the noisy_model_input
            model_pred = model_pred + noisy_model_input
            weighting = None
        elif args.model_prediction_type == "sigma_scaled":
            # apply sigma scaling
            model_pred = model_pred * (-sigmas) + noisy_model_input

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

        # flow matching loss: this is different from SD3
        target = noise - latents

        return model_pred, target, timesteps, None, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, False, True, False, flux="dev")

    def update_metadata(self, metadata, args):
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    # sdxl_train_util.add_sdxl_training_arguments(parser)
    parser.add_argument("--clip_l", type=str, help="path to clip_l")
    parser.add_argument("--t5xxl", type=str, help="path to t5xxl")
    parser.add_argument("--ae", type=str, help="path to ae")
    parser.add_argument("--apply_t5_attn_mask", action="store_true")
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--split_mode",
        action="store_true",
        help="[EXPERIMENTAL] use split mode for Flux model, network arg `train_blocks=single` is required"
        + "/[実験的] Fluxモデルの分割モードを使用する。ネットワーク引数`train_blocks=single`が必要",
    )

    # copy from Diffusers
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid"],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, or sigmoid of random normal. / タイムステップをサンプリングする方法：sigma、random uniform、またはrandom normalのsigmoid。",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"の場合のみ有効）。',
    )
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw", "additive", "sigma_scaled"],
        default="sigma_scaled",
        help="How to interpret and process the model prediction: "
        "raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."
        " / モデル予測の解釈と処理方法："
        "raw（そのまま使用）、additive（ノイズ入力に加算）、sigma_scaled（シグマスケーリングを適用）。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 3.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは3.0。",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()
    trainer.train(args)
