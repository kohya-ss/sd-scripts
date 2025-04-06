# training with captions

import argparse
from typing import List, Optional, Union

import torch
from accelerate import Accelerator
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import sdxl_model_util, strategy_sd, strategy_sdxl

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

import library.sdxl_train_util as sdxl_train_util
from library.sdxl_original_unet import SdxlUNet2DConditionModel
import train_native

setup_logging()
import logging

logger = logging.getLogger(__name__)

class SdxlNativeTrainer(train_native.NativeTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.unet_num_blocks_for_block_lr = sdxl_model_util.UNET_NUM_BLOCKS_FOR_BLOCK_LR
        self.is_sdxl = True
        self.arb_min_steps = sdxl_model_util.ARB_MIN_STEPS

    def get_block_params_to_optimize(self, unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
        block_params = [[] for _ in range(len(block_lrs))]

        for i, (name, param) in enumerate(unet.named_parameters()):
            if name.startswith("time_embed.") or name.startswith("label_emb."):
                block_index = 0  # 0
            elif name.startswith("input_blocks."):  # 1-9
                block_index = 1 + int(name.split(".")[1])
            elif name.startswith("middle_block."):  # 10-12
                block_index = 10 + int(name.split(".")[1])
            elif name.startswith("output_blocks."):  # 13-21
                block_index = 13 + int(name.split(".")[1])
            elif name.startswith("out."):  # 22
                block_index = 22
            else:
                raise ValueError(f"unexpected parameter name: {name}")

            block_params[block_index].append(param)

        params_to_optimize = []
        for i, params in enumerate(block_params):
            if block_lrs[i] == 0:  # 0のときは学習しない do not optimize when lr is 0
                continue
            params_to_optimize.append({"params": params, "lr": block_lrs[i]})

        return params_to_optimize

    def append_block_lr_to_logs(self, block_lrs, logs, lr_scheduler, optimizer_type):
        names = []
        block_index = 0
        while block_index < self.unet_num_blocks_for_block_lr + 2:
            if block_index < self.unet_num_blocks_for_block_lr:
                if block_lrs[block_index] == 0:
                    block_index += 1
                    continue
                names.append(f"block{block_index}")
            elif block_index == self.unet_num_blocks_for_block_lr:
                names.append("text_encoder1")
            elif block_index == self.unet_num_blocks_for_block_lr + 1:
                names.append("text_encoder2")

            block_index += 1

        train_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)

    def assert_extra_args(self, args, train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset], val_dataset_group: Optional[train_util.DatasetGroup]):
        #Disabled for 64 / 32 conflict. Has been checked below.
        #super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"
        train_dataset_group.verify_bucket_reso_steps(self.arb_min_steps)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(self.arb_min_steps)

    def get_tokenize_strategy(self, args):
        return strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_sdxl.SdxlTokenizeStrategy):
        return [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]  # will be removed in the future
    
    def get_latents_caching_strategy(self, args):
        latents_caching_strategy = strategy_sd.SdSdxlLatentsCachingStrategy(
            False, args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        return latents_caching_strategy
    
    def load_target_model(self, args, weight_dtype, accelerator):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        # モデルに xformers とか memory efficient attention を組み込む
        if args.diffusers_xformers:
            # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
            # How about Text encoders?
            accelerator.print("Use xformers by Diffusers")
            if not self.is_sdxl:
                self.set_diffusers_xformers_flag(unet, True)
            self.set_diffusers_xformers_flag(vae, True)
            self.set_diffusers_xformers_flag(text_encoder1, True)
            self.set_diffusers_xformers_flag(text_encoder2, True)
        else:
            # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
            accelerator.print("Disable Diffusers' xformers")
            train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
            if args.xformers and torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
                #vae.set_use_memory_efficient_attention_xformers(args.xformers)
                self.set_diffusers_xformers_flag(vae, True)
                self.set_diffusers_xformers_flag(text_encoder1, True)
                self.set_diffusers_xformers_flag(text_encoder2, True)

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def get_text_encoding_strategy(self, args):
        return strategy_sdxl.SdxlTextEncodingStrategy()
    
    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders + [accelerator.unwrap_model(text_encoders[-1])]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_sdxl.SdxlTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, None, args.skip_cache_check, is_weighted=args.weighted_captions
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
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders + [accelerator.unwrap_model(text_encoders[-1])], accelerator)
            accelerator.wait_for_everyone()

            text_encoders[0].to("cpu", dtype=torch.float32)  # Text Encoder doesn't work with fp16 on CPU
            text_encoders[1].to("cpu", dtype=torch.float32)
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            text_encoders[1].to(accelerator.device, dtype=weight_dtype)

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
        indices: Optional[List[int]] = None,
    ):
        noisy_latents = noisy_latents.to(weight_dtype)  # TODO check why noisy_latents is not weight_dtype

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, accelerator.device).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = torch.cat([pool2, embs], dim=1).to(weight_dtype)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2).to(weight_dtype)

        if indices is not None and len(indices) > 0:
            noisy_latents = noisy_latents[indices]
            timesteps = timesteps[indices]
            text_embedding = text_embedding[indices]
            vector_embedding = vector_embedding[indices]

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        sdxl_train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def save_model_on_epoch_end_or_stepwise(self, args, on_epoch_end, accelerator, save_dtype, epoch, num_train_epochs, global_step, text_encoders, vae, unet):
        src_path = self.src_stable_diffusion_ckpt if self.save_stable_diffusion_format else self.src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
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
            accelerator.unwrap_model(text_encoders[0]), #text_encoder1
            accelerator.unwrap_model(text_encoders[1]), #text_encoder2
            accelerator.unwrap_model(unet),
            vae,
            self.logit_scale,
            self.ckpt_info,
        )

    def save_model_on_train_end(self, args, accelerator, save_dtype, epoch, global_step, text_encoders, vae, unet):
        src_path = self.src_stable_diffusion_ckpt if self.save_stable_diffusion_format else self.src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            self.save_stable_diffusion_format,
            self.use_safetensors,
            save_dtype,
            epoch,
            global_step,
            accelerator.unwrap_model(text_encoders[0]), #text_encoder1
            accelerator.unwrap_model(text_encoders[1]), #text_encoder2
            accelerator.unwrap_model(unet),
            vae,
            self.logit_scale,
            self.ckpt_info,
        )

def setup_parser() -> argparse.ArgumentParser:
    parser = train_native.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument(
        "--learning_rate_te1",
        type=float,
        default=None,
        help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    )
    parser.add_argument(
        "--learning_rate_te2",
        type=float,
        default=None,
        help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {sdxl_model_util.UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{sdxl_model_util.UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )
    return parser

if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = SdxlNativeTrainer()
    trainer.train(args)
