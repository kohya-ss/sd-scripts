import argparse
import math
import os
import typing
from typing import Any, List, Union, Optional
import random
import time
import json
from multiprocessing import Value
from contextlib import nullcontext

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from accelerate import Accelerator
from diffusers import DDPMScheduler
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from library import deepspeed_utils, model_util, strategy_base, strategy_sd

import library.train_util as train_util
from library.train_util import DreamBoothDataset
import library.config_util as config_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.huggingface_util as huggingface_util
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)


class NativeTrainer:
    def __init__(self):
        self.vae_scale_factor = 0.18215
        self.is_sdxl = False

    # TODO 他のスクリプトと共通化する
    def generate_step_logs(
        self,
        args: argparse.Namespace,
        current_loss,
        avr_loss,
        lr_scheduler,
        lr_descriptions,
        optimizer=None,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
    ):
        # Assumed network_train_unet_only is False

        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/max_key_norm"] = maximum_norm
        if mean_norm is not None:
            logs["norm/avg_key_norm"] = mean_norm
        if mean_grad_norm is not None:
            logs["norm/avg_grad_norm"] = mean_grad_norm
        if mean_combined_norm is not None:
            logs["norm/avg_combined_norm"] = mean_combined_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i + 1
                if idx == -1:
                    lr_desc = "textencoder"
                else:
                    if len(lrs) > 2:
                        lr_desc = f"group{idx}"
                    else:
                        lr_desc = "unet"

            logs[f"lr/{lr_desc}"] = lr

            if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                # tracking d*lr value
                logs[f"lr/d*lr/{lr_desc}"] = (
                    lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                )
            if (
                args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
            ):  # tracking d*lr value of unet.
                logs["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )
        else:
            idx = 0
            logs["lr/textencoder"] = float(lrs[0])
            idx = 1

            for i in range(idx, len(lrs)):
                logs[f"lr/group{i}"] = float(lrs[i])
                if args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower():
                    logs[f"lr/d*lr/group{i}"] = (
                        lr_scheduler.optimizers[-1].param_groups[i]["d"] * lr_scheduler.optimizers[-1].param_groups[i]["lr"]
                    )
                if (
                    args.optimizer_type.lower().endswith("ProdigyPlusScheduleFree".lower()) and optimizer is not None
                ):
                    logs[f"lr/d*lr/group{i}"] = (
                        optimizer.param_groups[i]["d"] * optimizer.param_groups[i]["lr"]
                    )

        return logs

    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset], val_dataset_group: Optional[train_util.DatasetGroup]):
        train_dataset_group.verify_bucket_reso_steps(64)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    # Diffusers版のxformers使用フラグを設定する関数
    # Hint: Override load_target_model instead.
    def set_diffusers_xformers_flag(self, model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    def get_tokenize_strategy(self, args):
        return strategy_sd.SdTokenizeStrategy(args.v2, args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sd.SdTokenizeStrategy) -> List[Any]:
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            True, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy

    def get_text_encoding_strategy(self, args):
        return strategy_sd.SdTextEncodingStrategy(args.clip_skip)

    def get_text_encoder_outputs_caching_strategy(self, args):
        return None

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        """
        Returns a list of models that will be used for text encoding. SDXL uses wrapped and unwrapped models.
        FLUX.1 and SD3 may cache some outputs of the text encoder, so return the models that will be used for encoding (not cached).
        """
        return text_encoders

    def cache_text_encoder_outputs_if_needed(self, args, accelerator, unet, vae, text_encoders, dataset, weight_dtype):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def all_reduce_training_model(self, accelerator, training_model):
        for param in training_model.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype, **kwargs):
        noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizers[0], text_encoder, unet)

    def append_block_lr_to_logs(self, block_lrs, logs, lr_scheduler, optimizer_type):
        pass

    # region SD/SDXL

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: AutoencoderKL, images: torch.FloatTensor) -> torch.FloatTensor:
        return vae.encode(images).latent_dist.sample()

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        return latents * self.vae_scale_factor

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        weight_dtype,
        train_unet,
        is_train=True
    ):
        # network is removed: There is no multiplyer and it is no longer required.

        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        # Note: Flow Matching will bend the timesteps
        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

        # ensure the hidden state will require grad
        if args.gradient_checkpointing:
            for x in noisy_latents:
                x.requires_grad_(True)
            for t in text_encoder_conds:
                t.requires_grad_(True)

        # Predict the noise residual
        with torch.set_grad_enabled(is_train), accelerator.autocast():
            noise_pred = self.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )

        if args.flow_model:
            # Rectified Flow. Kind of vpred. Math is fun.
            target = noise - latents
        elif args.v_parameterization:
            # v-parameterization training
            # velocity = sqrt_alpha_prod * noise - sqrt_one_minus_alpha_prod * sample
            # target = (alphas_cumprod[timesteps] ** 0.5) * noise - (1 - alphas_cumprod[timesteps]) ** 0.5 * latents
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            # EPS mode
            target = noise

        # differential output preservation
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                with torch.no_grad(), accelerator.autocast():
                    noise_pred_prior = self.call_unet(
                        args,
                        accelerator,
                        unet,
                        noisy_latents,
                        timesteps,
                        text_encoder_conds,
                        batch,
                        weight_dtype,
                        indices=diff_output_pr_indices,
                    )
                target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

        # weighting is unused unless cosmap is used (See SD3 / Flux). 
        weighting = None
        # noise is used for Contrastive Flow Matching.
        return noise_pred, target, timesteps, weighting, noise

    def post_process_loss(self, loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
        if args.min_snr_gamma:
            loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
        if args.scale_v_pred_loss_like_noise_pred:
            loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
        if args.v_pred_like_loss:
            loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
        if args.debiased_estimation_loss:
            loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
        return loss

    def get_sai_model_spec(self, args):
        return train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)

    def update_metadata(self, metadata, args):
        pass

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        text_encoder.text_model.embeddings.to(dtype=weight_dtype)

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        return accelerator.prepare(unet)

    def on_step_start(self, args, accelerator, text_encoders, unet, batch, weight_dtype):
        pass

    def load_target_save_config(self, args):
        # verify load/save model formats
        if self.load_stable_diffusion_format:
            self.src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
            self.src_diffusers_model_path = None
        else:
            self.src_stable_diffusion_ckpt = None
            self.src_diffusers_model_path = args.pretrained_model_name_or_path

        if args.save_model_as is None:
            self.save_stable_diffusion_format = self.load_stable_diffusion_format
            self.use_safetensors = args.use_safetensors
        else:
            self.save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
            self.use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
            # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"

    def save_model_on_epoch_end_or_stepwise(self, args, on_epoch_end, accelerator, save_dtype, epoch, num_train_epochs, global_step, text_encoders, vae, unet):
        src_path = self.src_stable_diffusion_ckpt if self.save_stable_diffusion_format else self.src_diffusers_model_path
        train_util.save_sd_model_on_epoch_end_or_stepwise(
            args,
            on_epoch_end,
            accelerator,
            src_path,
            self.save_stable_diffusion_format,
            self.use_safetensors,
            save_dtype,
            epoch,
            num_train_epochs,
            global_step,
            accelerator.unwrap_model(text_encoders[0]), #text_encoder
            accelerator.unwrap_model(unet),
            vae,
        )

    def save_model_on_train_end(self, args, accelerator, save_dtype, epoch, global_step, text_encoders, vae, unet):
        src_path = self.src_stable_diffusion_ckpt if self.save_stable_diffusion_format else self.src_diffusers_model_path
        train_util.save_sd_model_on_train_end(
            args, 
            src_path, 
            self.save_stable_diffusion_format, 
            self.use_safetensors,
            save_dtype, 
            epoch, 
            global_step, 
            accelerator.unwrap_model(text_encoders[0]), #text_encoder
            accelerator.unwrap_model(unet),
            vae,
        )

    # endregion

    def process_batch(
        self, 
        batch, 
        text_encoders, 
        unet, 
        vae, 
        noise_scheduler, 
        vae_dtype, 
        weight_dtype, 
        accelerator, 
        args, 
        text_encoding_strategy: strategy_base.TextEncodingStrategy, 
        tokenize_strategy: strategy_base.TokenizeStrategy, 
        is_train=True, 
        train_text_encoder=True, 
        train_unet=True
    ) -> torch.Tensor:
        """
        Process a batch for the models
        """
        with torch.no_grad():
            if "latents" in batch and batch["latents"] is not None:
                latents = typing.cast(torch.FloatTensor, batch["latents"].to(accelerator.device))
            else:
                # latentに変換
                latents = self.encode_images_to_latents(args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))

                # NaNが含まれていれば警告を表示し0に置き換える
                if torch.any(torch.isnan(latents)):
                    accelerator.print("NaN found in latents, replacing with zeros")
                    latents = typing.cast(torch.FloatTensor, torch.nan_to_num(latents, 0, out=latents))

            latents = self.shift_scale_latents(args, latents)

        # Code guide: encoder_hidden_states1, encoder_hidden_states2, pool2 = text_encoder_outputs_list = text_encoder_conds
        # input_ids1, input_ids2 = batch["input_ids_list"]
        # Then the routine "get_noise_pred_and_target > call_unet" will handle the rest.
        text_encoder_conds = []
        text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
        if text_encoder_outputs_list is not None:
            text_encoder_conds = text_encoder_outputs_list  # List of text encoder outputs

        if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
            # TODO this does not work if 'some text_encoders are trained' and 'some are not and not cached'
            with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
                # Get the text embedding for conditioning
                if args.weighted_captions:
                    input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids_list,
                        weights_list,
                    )
                else:
                    input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                    encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                        tokenize_strategy,
                        self.get_models_for_text_encoding(args, accelerator, text_encoders),
                        input_ids,
                    )
                if args.full_fp16:
                    encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

            # if text_encoder_conds is not cached, use encoded_text_encoder_conds
            if len(text_encoder_conds) == 0:
                text_encoder_conds = encoded_text_encoder_conds
            else:
                # if encoded_text_encoder_conds is not None, update cached text_encoder_conds
                for i in range(len(encoded_text_encoder_conds)):
                    if encoded_text_encoder_conds[i] is not None:
                        text_encoder_conds[i] = encoded_text_encoder_conds[i]

        # sample noise, call unet, get target
        noise_pred, target, timesteps, weighting, noise = self.get_noise_pred_and_target(
            args,
            accelerator,
            noise_scheduler,
            latents,
            batch,
            text_encoder_conds,
            unet,
            weight_dtype,
            train_unet,
            is_train=is_train
        )

        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
        loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
        if weighting is not None:
            loss = loss * weighting
        if args.flow_model and args.contrastive_flow_matching and latents.size(0) > 1:
            # Original code accepts vpred, which is strange.
            negative_latents = latents.roll(1, 0)
            negative_noise = noise.roll(1, 0)
            #with torch.no_grad():    
            target_negative = negative_noise - negative_latents
            loss_contrastive = torch.nn.functional.mse_loss(noise_pred.float(), target_negative.float(), reduction="none")
            loss = loss - args.cfm_lambda * loss_contrastive
        if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
            loss = apply_masked_loss(loss, batch)
        loss = loss.mean([1, 2, 3])

        loss_weights = batch["loss_weights"]  # 各sampleごとのweight
        loss = loss * loss_weights

        loss = self.post_process_loss(loss, args, timesteps, noise_scheduler)

        # From Flow Matching code. So strange.
        # if loss.ndim != 0:
        #    loss = loss.mean()

        return loss.mean()

    def train(self, args):

        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.verify_fm_training_args(args)
        train_util.prepare_dataset_args(args, True)
        if args.skip_cache_check:
            train_util.set_skip_npz_path_check(True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)

        # Not a todo but only SDXL has such implementation.
        block_lrs = None
        if args.block_lr:
            assert (
                not args.weighted_captions or not args.cache_text_encoder_outputs
            ), "weighted_captions is not supported when caching text encoder outputs / cache_text_encoder_outputsを使うときはweighted_captionsはサポートされていません"
            assert (
                not args.train_text_encoder or not args.cache_text_encoder_outputs
            ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

            if args.block_lr:
                block_lrs = [float(lr) for lr in args.block_lr.split(",")]
                assert (
                    len(block_lrs) == self.unet_num_blocks_for_block_lr
                ), f"block_lr must have {self.unet_num_blocks_for_block_lr} values / block_lrは{self.unet_num_blocks_for_block_lr}個の値を指定してください"
            else:
                block_lrs = None

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)  # 乱数系列を初期化する

        tokenize_strategy = self.get_tokenize_strategy(args)
        strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)
        tokenizers = self.get_tokenizers(tokenize_strategy)  # will be removed after sample_image is refactored

        # prepare caching strategy: this must be set before preparing dataset. because dataset may use this strategy for initialization.
        # if args.cache_latents:
        latents_caching_strategy = self.get_latents_caching_strategy(args)
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

        # データセットを準備する
        if args.dataset_class is None:
            blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
            if use_user_config:
                logger.info(f"Loading dataset config from {args.dataset_config}")
                user_config = config_util.load_user_config(args.dataset_config)
                ignored = ["train_data_dir", "reg_data_dir", "in_json"]
                if any(getattr(args, attr) is not None for attr in ignored):
                    logger.warning(
                        "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                            ", ".join(ignored)
                        )
                    )
            else:
                if use_dreambooth_method:
                    logger.info("Using DreamBooth method.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                    args.train_data_dir, args.reg_data_dir
                                )
                            }
                        ]
                    }
                else:
                    logger.info("Training with captions.")
                    user_config = {
                        "datasets": [
                            {
                                "subsets": [
                                    {
                                        "image_dir": args.train_data_dir,
                                        "metadata_file": args.in_json,
                                    }
                                ]
                            }
                        ]
                    }

            blueprint = blueprint_generator.generate(user_config, args)
            train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args)
            val_dataset_group = None # placeholder until validation dataset supported for arbitrary

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
            train_util.debug_dataset(train_dataset_group)

            if val_dataset_group is not None:
                val_dataset_group.set_current_strategies()  # dataset needs to know the strategies explicitly
                train_util.debug_dataset(val_dataset_group)
            return
        if len(train_dataset_group) == 0:
            logger.error(
                "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / 画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
            )
            return

        if cache_latents:
            assert (
                train_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
            if val_dataset_group is not None:
                assert (
                    val_dataset_group.is_latent_cacheable()
                ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"
            if val_dataset_group is not None:
                assert (
                    val_dataset_group.is_text_encoder_output_cacheable()
                ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        self.assert_extra_args(args, train_dataset_group, val_dataset_group)  # may change some args

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        logger.info(f"Accelerator prepared at {accelerator.device} / process index : {accelerator.num_processes}, local process index : {accelerator.local_process_index}")
        logger.info(f"Waiting for everyone / 他のプロセスを待機中")
        accelerator.wait_for_everyone()
        logger.info("All processes are ready / すべてのプロセスが準備完了")

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む
        # TODO: SDXL Model Specific (vae vs ae, unet vs mmdit)
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        self.load_target_save_config(args)
        
        # TODO: SDXL Model Specific
        if self.is_sdxl:
            text_encoder1 = text_encoders[0]
            text_encoder2 = text_encoders[1]

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()

            train_dataset_group.new_cache_latents(vae, accelerator)
            if val_dataset_group is not None:
                val_dataset_group.new_cache_latents(vae, accelerator)

            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()
        
        # 学習を準備する：モデルを適切な状態にする
        if args.gradient_checkpointing:
            # cpu_offload throws error
            unet.enable_gradient_checkpointing()
     
        train_unet = args.learning_rate != 0
        
        train_text_encoder = False
        # TODO: SDXL Model Specific
        if self.is_sdxl:
            train_text_encoder1 = False
            train_text_encoder2 = False

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        text_encoding_strategy = self.get_text_encoding_strategy(args)
        strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

        # TODO: SDXL Model Specific
        if args.train_text_encoder:
            # TODO each option for two text encoders?
            accelerator.print("enable text encoder training")
            if args.gradient_checkpointing:
                text_encoder1.gradient_checkpointing_enable()
                text_encoder2.gradient_checkpointing_enable()
            lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
            lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
            train_text_encoder1 = lr_te1 != 0
            train_text_encoder2 = lr_te2 != 0

            # Used in process_batch. Seems that this is likely a AND gate.
            train_text_encoder = train_text_encoder1 and train_text_encoder2

            # caching one text encoder output is not supported
            if not train_text_encoder1:
                text_encoder1.to(weight_dtype)
            if not train_text_encoder2:
                text_encoder2.to(weight_dtype)
            text_encoder1.requires_grad_(train_text_encoder1)
            text_encoder2.requires_grad_(train_text_encoder2)
            text_encoder1.train(train_text_encoder1)
            text_encoder2.train(train_text_encoder2)
        else:
            text_encoder1.to(weight_dtype)
            text_encoder2.to(weight_dtype)
            text_encoder1.requires_grad_(False)
            text_encoder2.requires_grad_(False)
            text_encoder1.eval()
            text_encoder2.eval()

            # TextEncoderの出力をキャッシュする
            if args.cache_text_encoder_outputs:
                # Text Encodes are eval and no grad
                text_encoder_output_caching_strategy =  self.get_text_encoder_outputs_caching_strategy(args)
                strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_output_caching_strategy)

                self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)
                if val_dataset_group is not None:
                    self.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, val_dataset_group, weight_dtype)

            accelerator.wait_for_everyone()

        if not cache_latents:
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # TODO: Revisit for FP8
        # Default=True in pytorch. Comment out for adding requires_grad_ in specific layers (torch.Tensor). e.g. 63% UNET for 4x RTX3090
        #unet.requires_grad_(train_unet)
        if not train_unet:
            unet.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

        # TODO: SDXL Model Specific
        # TODO: Why casting to torch.tensor will slow down the performance so much? (20% slower)
        training_models = []
        params_to_optimize = []
        using_torchao = args.optimizer_type.endswith("4bit") or args.optimizer_type.endswith("Fp8")
        if train_unet:
            training_models.append(unet)
            if block_lrs is None:
                lr_unet = args.learning_rate
                params_to_optimize.append({"params": list(unet.parameters()), "lr": torch.tensor(lr_unet) if using_torchao else lr_unet})
            else:
                params_to_optimize.extend(self.get_block_params_to_optimize(unet, block_lrs))

        if train_text_encoder1:
            training_models.append(text_encoder1)
            lr_te1 = args.learning_rate_te1 or args.learning_rate
            params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": torch.tensor(lr_te1) if using_torchao else lr_te1})
        if train_text_encoder2:
            training_models.append(text_encoder2)
            lr_te2 = args.learning_rate_te2 or args.learning_rate
            params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": torch.tensor(lr_te2) if using_torchao else lr_te2})

        # calculate number of trainable parameters
        n_params = 0
        for group in params_to_optimize:
            for p in group["params"]:
                n_params += p.numel()

        accelerator.print(f"train unet: {train_unet}, text_encoder1: {train_text_encoder1}, text_encoder2: {train_text_encoder2}")
        accelerator.print(f"number of models: {len(training_models)}")
        accelerator.print(f"number of trainable parameters: {n_params}")

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        # network exclusive
        lr_descriptions = None

        if args.fused_optimizer_groups:
            # fused backward pass: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html
            # Instead of creating an optimizer for all parameters as in the tutorial, we create an optimizer for each group of parameters.
            # This balances memory usage and management complexity.

            # calculate total number of parameters
            n_total_params = sum(len(params["params"]) for params in params_to_optimize)
            params_per_group = math.ceil(n_total_params / args.fused_optimizer_groups)

            # split params into groups, keeping the learning rate the same for all params in a group
            # this will increase the number of groups if the learning rate is different for different params (e.g. U-Net and text encoders)
            grouped_params = []
            param_group = []
            param_group_lr = -1
            for group in params_to_optimize:
                lr = group["lr"]
                for p in group["params"]:
                    # if the learning rate is different for different params, start a new group
                    if lr != param_group_lr:
                        if param_group:
                            grouped_params.append({"params": param_group, "lr": param_group_lr})
                            param_group = []
                        param_group_lr = lr

                    param_group.append(p)

                    # if the group has enough parameters, start a new group
                    if len(param_group) == params_per_group:
                        grouped_params.append({"params": param_group, "lr": param_group_lr})
                        param_group = []
                        param_group_lr = -1

            if param_group:
                grouped_params.append({"params": param_group, "lr": param_group_lr})

            # prepare optimizers for each group
            optimizers = []
            for group in grouped_params:
                _, _, optimizer = train_util.get_optimizer(args, trainable_params=[group])
                optimizers.append(optimizer)
            optimizer = optimizers[0]  # avoid error in the following code

            logger.info(f"using {len(optimizers)} optimizers for fused optimizer groups")

        else:
            optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
            optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

        # prepare dataloader
        # strategies are set here because they cannot be referenced in another process. Copy them with the dataset
        # some strategies can be None
        train_dataset_group.set_current_strategies()
        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()

        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        pin_memory = args.pin_memory

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=args.persistent_data_loader_workers,
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset_group if val_dataset_group is not None else [],
            shuffle=False,
            batch_size=1,
            collate_fn=collator,
            num_workers=n_workers,
            pin_memory=pin_memory,
            persistent_workers=args.persistent_data_loader_workers,
        )

        # 学習ステップ数を計算する
        if args.max_train_epochs is not None:
            args.max_train_steps = args.max_train_epochs * math.ceil(
                len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
            )
            accelerator.print(
                f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
            )

        # データセット側にも学習ステップを送信
        train_dataset_group.set_max_train_steps(args.max_train_steps)

        # lr schedulerを用意する
        if args.fused_optimizer_groups:
            # prepare lr schedulers for each optimizer
            lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
            lr_scheduler = lr_schedulers[0]  # avoid error in the following code
        else:
            lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        # TODO: SDXL Model Specific
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            unet.to(weight_dtype)
            text_encoder1.to(weight_dtype)
            text_encoder2.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            unet.to(weight_dtype)
            text_encoder1.to(weight_dtype)
            text_encoder2.to(weight_dtype)

        # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
        if train_text_encoder1:
            text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
            text_encoder1.text_model.final_layer_norm.requires_grad_(False)

        # TODO: Revisit for FP8
        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base or args.fp8_base_unet:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training for U-Net.")
            unet_weight_dtype = torch.float8_e4m3fn

            if not args.fp8_base_unet:
                accelerator.print("enable fp8 training for Text Encoder.")
            te_weight_dtype = weight_dtype if args.fp8_base_unet else torch.float8_e4m3fn

            # unet.to(accelerator.device)  # this makes faster `to(dtype)` below, but consumes 23 GB VRAM
            # unet.to(dtype=unet_weight_dtype)  # without moving to gpu, this takes a lot of time and main memory

            # logger.info(f"set U-Net weight dtype to {unet_weight_dtype}, device to {accelerator.device}")
            # unet.to(accelerator.device, dtype=unet_weight_dtype)  # this seems to be safer than above
            logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
            unet.to(dtype=unet_weight_dtype)  # do not move to device because unet is not prepared by accelerator

            for i, t_enc in enumerate(text_encoders):
                t_enc.to(dtype=te_weight_dtype)
                # nn.Embedding not support FP8
                self.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)
           
        # TODO: SDXL Model Specific
        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoder1 if train_text_encoder1 else None,
                text_encoder2=text_encoder2 if train_text_encoder2 else None,
            )
            # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
            ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
            training_models = [ds_model]
        else:
            # acceleratorがなんかよろしくやってくれるらしい
            if train_unet:
                # default implementation is: unet = accelerator.prepare(unet)
                unet = self.prepare_unet_with_accelerator(args, accelerator, unet)  # accelerator does some magic here
            if train_text_encoder1:
                text_encoder1 = accelerator.prepare(text_encoder1)
            if train_text_encoder2:
                text_encoder2 = accelerator.prepare(text_encoder2)
            optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
                 optimizer, train_dataloader, val_dataloader, lr_scheduler
            )
            
        # TextEncoderの出力をキャッシュするときにはCPUへ移動する
        if args.cache_text_encoder_outputs:
            # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
            text_encoder1.to("cpu", dtype=torch.float32)
            text_encoder2.to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)
        else:
            # make sure Text Encoders are on GPU
            text_encoder1.to(accelerator.device)
            text_encoder2.to(accelerator.device)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
            # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
            train_util.patch_accelerator_for_fp16_training(accelerator)


        # Removed saving network weights, but preserving the steps_from_state. 
        # CLI will override this file, and to resume from a checkpoint (base model), you still have to modify the CLI.
        def save_model_hook(models, weights, output_dir):
            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # load current epoch and step to
            nonlocal steps_from_state
            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

        # resumeする
        train_util.resume_from_local_or_hf_if_specified(accelerator, args)

        if args.fused_backward_pass:
            # use fused optimizer for backward pass: other optimizers will be supported in the future
            import library.adafactor_fused

            library.adafactor_fused.patch_adafactor_fused(optimizer)
            for param_group in optimizer.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def __grad_hook(tensor: torch.Tensor, param_group=param_group):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, param_group)
                            tensor.grad = None

                        parameter.register_post_accumulate_grad_hook(__grad_hook)

        elif args.fused_optimizer_groups:
            # prepare for additional optimizers and lr schedulers
            for i in range(1, len(optimizers)):
                optimizers[i] = accelerator.prepare(optimizers[i])
                lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

            # counters are used to determine when to step the optimizer
            global optimizer_hooked_count
            global num_parameters_per_group
            global parameter_optimizer_map

            optimizer_hooked_count = {}
            num_parameters_per_group = [0] * len(optimizers)
            parameter_optimizer_map = {}

            for opt_idx, optimizer in enumerate(optimizers):
                for param_group in optimizer.param_groups:
                    for parameter in param_group["params"]:
                        if parameter.requires_grad:

                            def optimizer_hook(parameter: torch.Tensor):
                                if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                    accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                                i = parameter_optimizer_map[parameter]
                                optimizer_hooked_count[i] += 1
                                if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                    optimizers[i].step()
                                    optimizers[i].zero_grad(set_to_none=True)

                            parameter.register_post_accumulate_grad_hook(optimizer_hook)
                            parameter_optimizer_map[parameter] = opt_idx
                            num_parameters_per_group[opt_idx] += 1

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
        accelerator.print(f"  num validation images * repeats / 学習画像の数×繰り返し回数: {val_dataset_group.num_train_images if val_dataset_group is not None else 0}")
        accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
        accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
        accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
        accelerator.print(
            f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
        )
        # accelerator.print(f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}")
        accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
        accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

        # TODO refactor metadata creation and move to util
        metadata = {
            "ss_session_id": session_id,  # random integer indicating which group of epochs the model came from
            "ss_training_started_at": training_started_at,  # unix timestamp
            "ss_output_name": args.output_name,
            "ss_learning_rate": args.learning_rate,
            "ss_learning_rate_te1": args.learning_rate_te1,
            "ss_learning_rate_te2": args.learning_rate_te2,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_mixed_precision": args.mixed_precision,
            "ss_full_fp16": bool(args.full_fp16),
            "ss_v2": bool(args.v2),
            "ss_base_model_version": model_version,
            "ss_clip_skip": args.clip_skip,
            "ss_max_token_length": args.max_token_length,
            "ss_cache_latents": bool(args.cache_latents),
            "ss_seed": args.seed,
            "ss_lowram": args.lowram,
            "ss_noise_offset": args.noise_offset,
            "ss_multires_noise_iterations": args.multires_noise_iterations,
            "ss_multires_noise_discount": args.multires_noise_discount,
            "ss_adaptive_noise_scale": args.adaptive_noise_scale,
            "ss_zero_terminal_snr": args.zero_terminal_snr,
            "ss_training_comment": args.training_comment,  # will not be updated after training
            "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
            "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
            "ss_max_grad_norm": args.max_grad_norm,
            "ss_caption_dropout_rate": args.caption_dropout_rate,
            "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
            "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
            "ss_face_crop_aug_range": args.face_crop_aug_range,
            "ss_prior_loss_weight": args.prior_loss_weight,
            "ss_min_snr_gamma": args.min_snr_gamma,
            "ss_scale_weight_norms": args.scale_weight_norms,
            "ss_ip_noise_gamma": args.ip_noise_gamma,
            "ss_debiased_estimation": bool(args.debiased_estimation_loss),
            "ss_noise_offset_random_strength": args.noise_offset_random_strength,
            "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
            "ss_loss_type": args.loss_type,
            "ss_huber_schedule": args.huber_schedule,
            "ss_huber_scale": args.huber_scale,
            "ss_huber_c": args.huber_c,
            "ss_fp8_base": bool(args.fp8_base),
            "ss_fp8_base_unet": bool(args.fp8_base_unet),
            "ss_validation_seed": args.validation_seed, 
            "ss_validation_split": args.validation_split, 
            "ss_max_validation_steps": args.max_validation_steps, 
            "ss_validate_every_n_epochs": args.validate_every_n_epochs, 
            "ss_validate_every_n_steps": args.validate_every_n_steps,             
            "ss_resize_interpolation": args.resize_interpolation,
        }

        self.update_metadata(metadata, args)  # architecture specific metadata

        if use_user_config:
            # save metadata of multiple datasets
            # NOTE: pack "ss_datasets" value as json one time
            #   or should also pack nested collections as json?
            datasets_metadata = []
            tag_frequency = {}  # merge tag frequency for metadata editor
            dataset_dirs_info = {}  # merge subset dirs for metadata editor

            for dataset in train_dataset_group.datasets:
                is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
                dataset_metadata = {
                    "is_dreambooth": is_dreambooth_dataset,
                    "batch_size_per_device": dataset.batch_size,
                    "num_train_images": dataset.num_train_images,  # includes repeating
                    "num_reg_images": dataset.num_reg_images,
                    "resolution": (dataset.width, dataset.height),
                    "enable_bucket": bool(dataset.enable_bucket),
                    "min_bucket_reso": dataset.min_bucket_reso,
                    "max_bucket_reso": dataset.max_bucket_reso,
                    "tag_frequency": dataset.tag_frequency,
                    "bucket_info": dataset.bucket_info,                    
                    "resize_interpolation": dataset.resize_interpolation,
                }

                subsets_metadata = []
                for subset in dataset.subsets:
                    subset_metadata = {
                        "img_count": subset.img_count,
                        "num_repeats": subset.num_repeats,
                        "color_aug": bool(subset.color_aug),
                        "flip_aug": bool(subset.flip_aug),
                        "random_crop": bool(subset.random_crop),
                        "shuffle_caption": bool(subset.shuffle_caption),
                        "keep_tokens": subset.keep_tokens,
                        "keep_tokens_separator": subset.keep_tokens_separator,
                        "secondary_separator": subset.secondary_separator,
                        "enable_wildcard": bool(subset.enable_wildcard),
                        "caption_prefix": subset.caption_prefix,
                        "caption_suffix": subset.caption_suffix,
                        "resize_interpolation": subset.resize_interpolation,
                    }

                    image_dir_or_metadata_file = None
                    if subset.image_dir:
                        image_dir = os.path.basename(subset.image_dir)
                        subset_metadata["image_dir"] = image_dir
                        image_dir_or_metadata_file = image_dir

                    if is_dreambooth_dataset:
                        subset_metadata["class_tokens"] = subset.class_tokens
                        subset_metadata["is_reg"] = subset.is_reg
                        if subset.is_reg:
                            image_dir_or_metadata_file = None  # not merging reg dataset
                    else:
                        metadata_file = os.path.basename(subset.metadata_file)
                        subset_metadata["metadata_file"] = metadata_file
                        image_dir_or_metadata_file = metadata_file  # may overwrite

                    subsets_metadata.append(subset_metadata)

                    # merge dataset dir: not reg subset only
                    # TODO update additional-network extension to show detailed dataset config from metadata
                    if image_dir_or_metadata_file is not None:
                        # datasets may have a certain dir multiple times
                        v = image_dir_or_metadata_file
                        i = 2
                        while v in dataset_dirs_info:
                            v = image_dir_or_metadata_file + f" ({i})"
                            i += 1
                        image_dir_or_metadata_file = v

                        dataset_dirs_info[image_dir_or_metadata_file] = {
                            "n_repeats": subset.num_repeats,
                            "img_count": subset.img_count,
                        }

                dataset_metadata["subsets"] = subsets_metadata
                datasets_metadata.append(dataset_metadata)

                # merge tag frequency:
                for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
                    # あるディレクトリが複数のdatasetで使用されている場合、一度だけ数える
                    # もともと繰り返し回数を指定しているので、キャプション内でのタグの出現回数と、それが学習で何度使われるかは一致しない
                    # なので、ここで複数datasetの回数を合算してもあまり意味はない
                    if ds_dir_name in tag_frequency:
                        continue
                    tag_frequency[ds_dir_name] = ds_freq_for_dir

            metadata["ss_datasets"] = json.dumps(datasets_metadata)
            metadata["ss_tag_frequency"] = json.dumps(tag_frequency)
            metadata["ss_dataset_dirs"] = json.dumps(dataset_dirs_info)
        else:
            # conserving backward compatibility when using train_dataset_dir and reg_dataset_dir
            assert (
                len(train_dataset_group.datasets) == 1
            ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

            dataset = train_dataset_group.datasets[0]

            dataset_dirs_info = {}
            reg_dataset_dirs_info = {}
            if use_dreambooth_method:
                for subset in dataset.subsets:
                    info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
                    info[os.path.basename(subset.image_dir)] = {"n_repeats": subset.num_repeats, "img_count": subset.img_count}
            else:
                for subset in dataset.subsets:
                    dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                        "n_repeats": subset.num_repeats,
                        "img_count": subset.img_count,
                    }

            metadata.update(
                {
                    "ss_batch_size_per_device": args.train_batch_size,
                    "ss_total_batch_size": total_batch_size,
                    "ss_resolution": args.resolution,
                    "ss_color_aug": bool(args.color_aug),
                    "ss_flip_aug": bool(args.flip_aug),
                    "ss_random_crop": bool(args.random_crop),
                    "ss_shuffle_caption": bool(args.shuffle_caption),
                    "ss_enable_bucket": bool(dataset.enable_bucket),
                    "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
                    "ss_min_bucket_reso": dataset.min_bucket_reso,
                    "ss_max_bucket_reso": dataset.max_bucket_reso,
                    "ss_keep_tokens": args.keep_tokens,
                    "ss_dataset_dirs": json.dumps(dataset_dirs_info),
                    "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
                    "ss_tag_frequency": json.dumps(dataset.tag_frequency),
                    "ss_bucket_info": json.dumps(dataset.bucket_info),
                }
            )

        # model name and hash
        if args.pretrained_model_name_or_path is not None:
            sd_model_name = args.pretrained_model_name_or_path
            if os.path.exists(sd_model_name):
                metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
                metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
                sd_model_name = os.path.basename(sd_model_name)
            metadata["ss_sd_model_name"] = sd_model_name

        if args.vae is not None:
            vae_name = args.vae
            if os.path.exists(vae_name):
                metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
                metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
                vae_name = os.path.basename(vae_name)
            metadata["ss_vae_name"] = vae_name

        metadata = {k: str(v) for k, v in metadata.items()}

        # make minimum metadata for filtering
        minimum_metadata = {}
        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]

        # calculate steps to skip when resuming or starting from a specific step
        # this is not used for logging and file save. use global_step instead.
        initial_step = 0
        if args.initial_epoch is not None or args.initial_step is not None:
            # if initial_epoch or initial_step is specified, steps_from_state is ignored even when resuming
            if steps_from_state is not None:
                logger.warning(
                    "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
                )
            if args.initial_step is not None:
                initial_step = args.initial_step
            else:
                # num steps per epoch is calculated by num_processes and gradient_accumulation_steps
                initial_step = (args.initial_epoch - 1) * math.ceil(
                    len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
                )
        else:
            # if initial_epoch and initial_step are not specified, steps_from_state is used when resuming
            if steps_from_state is not None:
                initial_step = steps_from_state
                steps_from_state = None

        if initial_step > 0:
            assert (
                args.max_train_steps > initial_step
            ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

        progress_bar = tqdm(
            range(args.max_train_steps - initial_step), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps"
        )

        epoch_to_start = 0
        if initial_step > 0:
            if args.skip_until_initial_step:
                # if skip_until_initial_step is specified, load data and discard it to ensure the same data is used
                if not args.resume:
                    logger.info(
                        f"initial_step is specified but not resuming. lr scheduler will be started from the beginning / initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                    )
                logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")

                initial_step *= args.gradient_accumulation_steps

                # set epoch to start to make initial_step less than len(train_dataloader). Notice that initial_step has been multipled.
                epoch_to_start = initial_step // len(train_dataloader) 
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        # This variable is used for logging and file save: 
        global_step = 0

        noise_scheduler = self.get_noise_scheduler(args, accelerator.device)

        train_util.init_trackers(accelerator, args, "finetuning")

        loss_recorder = train_util.LossRecorder()
        val_step_loss_recorder = train_util.LossRecorder()
        val_epoch_loss_recorder = train_util.LossRecorder()

        # (code guide) train_network will explictly delete lots of models to reduce RAM. However it doesn't fit for the use case here (a buffed workstation with many system memory, storage, to make large scale finetune)
        #del train_dataset_group
        #if val_dataset_group is not None:
        #    del val_dataset_group
        # (code guide) meanwhile large scale finetune would like to save all the intermediate models for human evaluation (xy plot instead of live sampling).
        #def save_model()
        #def remove_model()

        # For --sample_at_first
        optimizer_eval_fn()
        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
        optimizer_train_fn()
        is_tracking = len(accelerator.trackers) > 0
        if is_tracking:
            # log empty object to commit the sample images to wandb
            accelerator.log({}, step=0)

        validation_steps = (
            min(args.max_validation_steps, len(val_dataloader)) 
            if args.max_validation_steps is not None 
            else len(val_dataloader)
        )

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                initial_step -= len(train_dataloader)
            # Divide back for proper logging.
            global_step = int(initial_step / args.gradient_accumulation_steps)

        # log device and dtype for each model
        logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
        for i, t_enc in enumerate(text_encoders):
            params_itr = t_enc.parameters()
            params_itr.__next__()  # skip the first parameter
            params_itr.__next__()  # skip the second parameter. because CLIP first two parameters are embeddings
            param_3rd = params_itr.__next__()
            logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")

        clean_memory_on_device(accelerator.device)

        enable_profiler = args.enable_profiler
        if enable_profiler:
            logger.warning(f"Pytorch profiler enabled. Disable after capturing traces. / Pytorch プロファイラーが有効になっています。トレースをキャプチャした後は無効にします。")

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}\n")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            for m in training_models:
                m.train()

            # TRAINING
            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                with accelerator.profile() if enable_profiler else nullcontext() as prof:
                    current_step.value = global_step
                    if initial_step > 0:
                        initial_step -= 1
                        continue

                    if args.fused_optimizer_groups:
                        optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

                    # Code guide: "network" here was misrepresented as training_model, however some features are capable for all "prepared" models. 
                    # Tne correct specific "network" operation has been removed.
                    # The process_batch will wrap all the inference logic (because it will be used for validation dataset also)
                    with accelerator.accumulate(*training_models):
                        # 250331: From HF guide
                        # 250406: No need
                        #optimizer.zero_grad(set_to_none=True)
                        
                        # temporary, for batch processing
                        self.on_step_start(args, accelerator, text_encoders, unet, batch, weight_dtype)

                        loss = self.process_batch(
                            batch, 
                            text_encoders, 
                            unet, 
                            vae, 
                            noise_scheduler, 
                            vae_dtype, 
                            weight_dtype, 
                            accelerator, 
                            args, 
                            text_encoding_strategy, 
                            tokenize_strategy, 
                            is_train=True, 
                            train_text_encoder=train_text_encoder, 
                            train_unet=train_unet
                        )

                        accelerator.backward(loss)

                        #250331: It is required to sync manually. See torch.Tensor.grad
                        if accelerator.sync_gradients:
                            for training_model in training_models:
                                self.all_reduce_training_model(accelerator, training_model)  # sync DDP grad manually                            
                                if (args.max_grad_norm != 0.0) and hasattr(training_model, "get_trainable_params"):
                                    params_to_clip = accelerator.unwrap_model(training_model).get_trainable_params()
                                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                                
                                # lora_flux exclusive
                                if hasattr(training_model, "update_grad_norms"):
                                    training_model.update_grad_norms()
                                if hasattr(training_model, "update_norms"):
                                    training_model.update_norms()

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    if args.scale_weight_norms:
                        for training_model in training_models:
                            keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(training_model).apply_max_norm_regularization(
                                args.scale_weight_norms, accelerator.device
                            )
                            # TODO: Multiple models
                            mean_grad_norm = None
                            mean_combined_norm = None
                            max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                    else:
                        keys_scaled, mean_norm, maximum_norm = None, None, None
                        mean_grad_norm = None
                        mean_combined_norm = None
                        max_mean_logs = {}

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        progress_bar.update(1)
                        global_step += 1

                        optimizer_eval_fn()
                        self.sample_images(
                            accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet
                        )

                        # 指定ステップごとにモデルを保存
                        if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                # Train network has different approach: It will upload to hf or remove old file immediately. 
                                # Train Native will keep the old *_train_utils.approach, however the class reference is so messy.
                                # Hint: self.load_target_model
                                self.save_model_on_epoch_end_or_stepwise(args, False, accelerator, save_dtype, epoch, num_train_epochs, global_step, text_encoders, vae, unet)
                        
                        optimizer_train_fn()

                    current_loss = loss.detach().item()

                    if len(accelerator.trackers) > 0:
                        logs = {"loss": current_loss}
                        if block_lrs is None:
                            train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_unet)
                        else:
                            self.append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)  # U-Net is included in block_lrs

                        accelerator.log(logs, step=global_step)

                    loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                    avr_loss: float = loss_recorder.moving_average
                    logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                    if is_tracking:
                        logs = self.generate_step_logs(
                            args, 
                            current_loss, 
                            avr_loss, 
                            lr_scheduler, 
                            lr_descriptions, 
                            optimizer, 
                            keys_scaled, 
                            mean_norm, 
                            maximum_norm,
                            mean_grad_norm,
                            mean_combined_norm,
                        )
                        accelerator.log(logs, step=global_step)

                    # VALIDATION PER STEP
                    should_validate_step = (
                        args.validate_every_n_steps is not None 
                        and global_step != 0 # Skip first step
                        and global_step % args.validate_every_n_steps == 0
                    )
                    if accelerator.sync_gradients and validation_steps > 0 and should_validate_step:
                        val_progress_bar = tqdm(
                            range(validation_steps), smoothing=0, 
                            disable=not accelerator.is_local_main_process, 
                            desc="validation steps"
                        )
                        for val_step, batch in enumerate(val_dataloader):
                            if val_step >= validation_steps:
                                break

                            # temporary, for batch processing
                            self.on_step_start(args, accelerator, text_encoders, unet, batch, weight_dtype)

                            loss = self.process_batch(
                                batch, 
                                text_encoders, 
                                unet, 
                                vae, 
                                noise_scheduler, 
                                vae_dtype, 
                                weight_dtype, 
                                accelerator, 
                                args, 
                                text_encoding_strategy, 
                                tokenize_strategy, 
                                is_train=False,
                                train_text_encoder=False, 
                                train_unet=False
                            )

                            current_loss = loss.detach().item()
                            val_step_loss_recorder.add(epoch=epoch, step=val_step, loss=current_loss)
                            val_progress_bar.update(1)
                            val_progress_bar.set_postfix({ "val_avg_loss": val_step_loss_recorder.moving_average })

                            if is_tracking:
                                logs = {
                                    "loss/validation/step_current": current_loss,
                                    "val_step": (epoch * validation_steps) + val_step,
                                }
                                accelerator.log(logs, step=global_step)

                        if is_tracking:
                            loss_validation_divergence = val_step_loss_recorder.moving_average - loss_recorder.moving_average
                            logs = {
                                "loss/validation/step_average": val_step_loss_recorder.moving_average, 
                                "loss/validation/step_divergence": loss_validation_divergence, 
                            }
                            accelerator.log(logs, step=global_step)
                                            
                    if global_step >= args.max_train_steps:
                        break

            # EPOCH VALIDATION
            should_validate_epoch = (
                (epoch + 1) % args.validate_every_n_epochs == 0 
                if args.validate_every_n_epochs is not None 
                else True
            )

            if should_validate_epoch and len(val_dataloader) > 0:
                val_progress_bar = tqdm(
                    range(validation_steps), smoothing=0, 
                    disable=not accelerator.is_local_main_process, 
                    desc="epoch validation steps"
                )

                for val_step, batch in enumerate(val_dataloader):
                    if val_step >= validation_steps:
                        break

                    # temporary, for batch processing
                    self.on_step_start(args, accelerator, text_encoders, unet, batch, weight_dtype)

                    loss = self.process_batch(
                        batch, 
                        text_encoders, 
                        unet,
                        vae, 
                        noise_scheduler, 
                        vae_dtype, 
                        weight_dtype, 
                        accelerator, 
                        args, 
                        text_encoding_strategy, 
                        tokenize_strategy, 
                        is_train=False,
                        train_text_encoder=False, 
                        train_unet=False
                    )

                    current_loss = loss.detach().item()
                    val_epoch_loss_recorder.add(epoch=epoch, step=val_step, loss=current_loss)
                    val_progress_bar.update(1)
                    val_progress_bar.set_postfix({ "val_epoch_avg_loss": val_epoch_loss_recorder.moving_average })

                    if is_tracking:
                        logs = {
                            "loss/validation/epoch_current": current_loss, 
                            "epoch": epoch + 1, 
                            "val_step": (epoch * validation_steps) + val_step
                        }
                        accelerator.log(logs, step=global_step)

                if is_tracking:
                    avr_loss: float = val_epoch_loss_recorder.moving_average
                    loss_validation_divergence = val_epoch_loss_recorder.moving_average - loss_recorder.moving_average 
                    logs = {
                        "loss/validation/epoch_average": avr_loss, 
                        "loss/validation/epoch_divergence": loss_validation_divergence, 
                        "epoch": epoch + 1
                    }
                    accelerator.log(logs, step=global_step)

            # END OF EPOCH
            if is_tracking:
                logs = {"loss/epoch_average": loss_recorder.moving_average, "epoch": epoch + 1}
                accelerator.log(logs, step=global_step)
                    
            if len(accelerator.trackers) > 0:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            optimizer_eval_fn()
            if args.save_every_n_epochs is not None:
                self.save_model_on_epoch_end_or_stepwise(args, True, accelerator, save_dtype, epoch, num_train_epochs, global_step, text_encoders, vae, unet)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
            optimizer_train_fn()

            # end of epoch

        # The sequence is rearranged. Looks like 

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        optimizer_eval_fn()

        if is_main_process:
            self.save_model_on_train_end(args, accelerator, save_dtype, num_train_epochs, global_step, text_encoders, vae, unet)
            logger.info("model saved.")

        accelerator.end_training()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        del accelerator  # この後メモリを使うのでこれは消す


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_fm_training_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    #Wrap to add_native_trainer_arguments(parser)?
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizer数",
    )

    #Wrap to add_runtime_arguments(parser)?
    parser.add_argument(
        "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )   

    #Append to add_training_arguments(parser)?
    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="[EXPERIMENTAL] enable offloading of tensors to CPU during checkpointing for U-Net or DiT, if supported"
        " / 勾配チェックポイント時にテンソルをCPUにオフロードする（U-NetまたはDiTのみ、サポートされている場合）",
    )
    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--fp8_base_unet",
        action="store_true",
        help="use fp8 for U-Net (or DiT), Text Encoder is fp16 or bf16"
        " / U-Net（またはDiT）にfp8を使用する。Text Encoderはfp16またはbf16",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    ) 
    parser.add_argument(
        "--skip_until_initial_step",
        action="store_true",
        help="skip training until initial_step is reached / initial_stepに到達するまで学習をスキップする",
    )
    parser.add_argument(
        "--initial_epoch",
        type=int,
        default=None,
        help="initial epoch number, 1 means first epoch (same as not specifying). NOTE: initial_epoch/step doesn't affect to lr scheduler. Which means lr scheduler will start from 0 without `--resume`."
        + " / 初期エポック数、1で最初のエポック（未指定時と同じ）。注意：initial_epoch/stepはlr schedulerに影響しないため、`--resume`しない場合はlr schedulerは0から始まる",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=None,
        help="initial step number including all epochs, 0 means first step (same as not specifying). overwrites initial_epoch."
        + " / 初期ステップ数、全エポックを含むステップ数、0で最初のステップ（未指定時と同じ）。initial_epochを上書きする",
    )
    parser.add_argument("--enable_profiler", action="store_true", help="Enable PyTorch Profiler for in depth analysis on tracing training process. Enable will make training very slow. / トレーニング プロセスのトレースに関する詳細な分析を行うには、PyTorch Profiler を有効にします。有効にすると、トレーニングが非常に遅くなります。")

    #Append to add_dataset_arguments(parser)?
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=None,
        help="Validation seed for shuffling validation dataset, training `--seed` used otherwise / 検証データセットをシャッフルするための検証シード、それ以外の場合はトレーニング `--seed` を使用する"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Split for validation images out of the training dataset / 学習画像から検証画像に分割する割合"
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=None,
        help="Run validation on validation dataset every N steps. By default, validation will only occur every epoch if a validation dataset is available / 検証データセットの検証をNステップごとに実行します。デフォルトでは、検証データセットが利用可能な場合にのみ、検証はエポックごとに実行されます"
    )
    parser.add_argument(
        "--validate_every_n_epochs",
        type=int,
        default=None,
        help="Run validation dataset every N epochs. By default, validation will run every epoch if a validation dataset is available / 検証データセットをNエポックごとに実行します。デフォルトでは、検証データセットが利用可能な場合、検証はエポックごとに実行されます"
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Max number of validation dataset items processed. By default, validation will run the entire validation dataset / 処理される検証データセット項目の最大数。デフォルトでは、検証は検証データセット全体を実行します"
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NativeTrainer()
    trainer.train(args)
