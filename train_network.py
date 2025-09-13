import importlib
import argparse
import math
import os
import sys
import random
import time
import json
from multiprocessing import Value
import toml
from collections import deque
import numpy as np

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed, DistributedType
from diffusers import DDPMScheduler
from library import deepspeed_utils, model_util

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
    get_weighted_text_embeddings,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
)
from library.avg_ckpt_util import (
    average_state_dicts,
    filter_lora_state_dict,
    collect_last_checkpoints,
    load_lora_state_dict,
)
from library.utils import setup_logging, add_logging_arguments
from library.rounding_util import round_parameters
from accelerate.utils import broadcast

setup_logging()
import logging

logger = logging.getLogger(__name__)


class NetworkTrainer:
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
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
    ):
        logs = {"loss/current": current_loss, "loss/average": avr_loss}

        if keys_scaled is not None:
            logs["max_norm/keys_scaled"] = keys_scaled
            logs["max_norm/average_key_norm"] = mean_norm
            logs["max_norm/max_key_norm"] = maximum_norm

        lrs = lr_scheduler.get_last_lr()
        for i, lr in enumerate(lrs):
            if lr_descriptions is not None:
                lr_desc = lr_descriptions[i]
            else:
                idx = i - (0 if args.network_train_unet_only else -1)
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

        return logs

    def assert_extra_args(self, args, train_dataset_group):
        train_dataset_group.verify_bucket_reso_steps(64)

    def load_target_model(self, args, weight_dtype, accelerator):
        text_encoder, vae, unet, _ = train_util.load_target_model(args, weight_dtype, accelerator)
        return model_util.get_model_version_str_for_sd1_sd2(args.v2, args.v_parameterization), text_encoder, vae, unet

    def load_tokenizer(self, args):
        tokenizer = train_util.load_tokenizer(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return False

    def is_train_text_encoder(self, args):
        return not args.network_train_unet_only and not self.is_text_encoder_outputs_cached(args)

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator, unet, vae, tokenizers, text_encoders, data_loader, weight_dtype
    ):
        for t_enc in text_encoders:
            t_enc.to(accelerator.device, dtype=weight_dtype)

    def get_text_cond(self, args, accelerator, batch, tokenizers, text_encoders, weight_dtype):
        input_ids = batch["input_ids"].to(accelerator.device)
        encoder_hidden_states = train_util.get_hidden_states(args, input_ids, tokenizers[0], text_encoders[0], weight_dtype)
        return encoder_hidden_states

    def call_unet(self, args, accelerator, unet, noisy_latents, timesteps, text_conds, batch, weight_dtype):
        noise_pred = unet(noisy_latents, timesteps, text_conds).sample
        return noise_pred

    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet):
        train_util.sample_images(accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, unet)

    def train(self, args):
        session_id = random.randint(0, 2**32)
        training_started_at = time.time()
        train_util.verify_training_args(args)
        train_util.prepare_dataset_args(args, True)
        deepspeed_utils.prepare_deepspeed_args(args)
        setup_logging(args, reset=True)
        logger.info(
            f"avg_cp: {args.avg_cp}, avg_window: {args.avg_window}, avg_begin: {args.avg_begin}, "
            f"avg_mode: {args.avg_mode}, avg_reset_stats: {args.avg_reset_stats}"
        )
        if args.round_lora_step is not None and args.round_lora_step > 0:
            logger.info(
                f"lora rounding: step={args.round_lora_step}, mode={args.round_lora_mode}, "
                f"every={args.round_lora_every}, begin={args.round_lora_begin}"
            )
        # parse bits schedule if provided
        def _parse_bits_sched(spec: str):
            items = []
            if not spec:
                return items
            for part in spec.split(","):
                if not part:
                    continue
                k, v = part.split(":")
                p = float(k)
                b = int(v)
                assert 0.0 <= p <= 1.0, "progress must be in [0,1]"
                assert b > 0, "bits must be > 0"
                items.append((p, b))
            items.sort(key=lambda x: x[0])
            return items

        dq_bits_sched = _parse_bits_sched(getattr(args, "dq_delta_bits_sched", None))

        if ((getattr(args, "dq_delta_step", None) is not None and args.dq_delta_step and args.dq_delta_step > 0)
            or (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits) or dq_bits_sched):
            logger.info(
                f"lora delta fake-quant: step={getattr(args,'dq_delta_step',None)}, bits={getattr(args,'dq_delta_bits',None)}, "
                f"mode={args.dq_delta_mode}, begin={args.dq_delta_begin}, granularity={getattr(args,'dq_delta_granularity',None)}, "
                f"stat={getattr(args,'dq_delta_stat',None)}, range_mul={getattr(args,'dq_delta_range_mul',None)}, bits_sched={dq_bits_sched}"
            )

        cache_latents = args.cache_latents
        use_dreambooth_method = args.in_json is None
        use_user_config = args.dataset_config is not None

        if args.seed is None:
            args.seed = random.randint(0, 2**32)
        set_seed(args.seed)

        # tokenizerは単体またはリスト、tokenizersは必ずリスト：既存のコードとの互換性のため
        tokenizer = self.load_tokenizer(args)
        tokenizers = tokenizer if isinstance(tokenizer, list) else [tokenizer]

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

            blueprint = blueprint_generator.generate(user_config, args, tokenizer=tokenizer)
            train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
        else:
            # use arbitrary dataset class
            train_dataset_group = train_util.load_arbitrary_dataset(args, tokenizer)

        current_epoch = Value("i", 0)
        current_step = Value("i", 0)
        ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
        collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

        if args.debug_dataset:
            train_util.debug_dataset(train_dataset_group)
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

        self.assert_extra_args(args, train_dataset_group)

        # acceleratorを準備する
        logger.info("preparing accelerator")
        accelerator = train_util.prepare_accelerator(args)
        is_main_process = accelerator.is_main_process

        # mixed precisionに対応した型を用意しておき適宜castする
        weight_dtype, save_dtype = train_util.prepare_dtype(args)
        vae_dtype = torch.float32 if args.no_half_vae else weight_dtype

        # モデルを読み込む
        model_version, text_encoder, vae, unet = self.load_target_model(args, weight_dtype, accelerator)

        # text_encoder is List[CLIPTextModel] or CLIPTextModel
        text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

        # モデルに xformers とか memory efficient attention を組み込む
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)

        # 差分追加学習のためにモデルを読み込む
        sys.path.append(os.path.dirname(__file__))
        accelerator.print("import network module:", args.network_module)
        network_module = importlib.import_module(args.network_module)

        if args.base_weights is not None:
            # base_weights が指定されている場合は、指定された重みを読み込みマージする
            for i, weight_path in enumerate(args.base_weights):
                if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
                    multiplier = 1.0
                else:
                    multiplier = args.base_weights_multiplier[i]

                accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")

                module, weights_sd = network_module.create_network_from_weights(
                    multiplier, weight_path, vae, text_encoder, unet, for_inference=True
                )
                module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

            accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")

        # 学習を準備する
        if cache_latents:
            vae.to(accelerator.device, dtype=vae_dtype)
            vae.requires_grad_(False)
            vae.eval()
            with torch.no_grad():
                train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
            vae.to("cpu")
            clean_memory_on_device(accelerator.device)

            accelerator.wait_for_everyone()

        # 必要ならテキストエンコーダーの出力をキャッシュする: Text Encoderはcpuまたはgpuへ移される
        # cache text encoder outputs if needed: Text Encoder is moved to cpu or gpu
        self.cache_text_encoder_outputs_if_needed(
            args, accelerator, unet, vae, tokenizers, text_encoders, train_dataset_group, weight_dtype
        )

        # prepare network
        net_kwargs = {}
        if args.network_args is not None:
            for net_arg in args.network_args:
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        # if a new network is added in future, add if ~ then blocks for each network (;'∀')
        if args.dim_from_weights:
            network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
        else:
            if "dropout" not in net_kwargs:
                # workaround for LyCORIS (;^ω^)
                net_kwargs["dropout"] = args.network_dropout

            network = network_module.create_network(
                1.0,
                args.network_dim,
                args.network_alpha,
                vae,
                text_encoder,
                unet,
                neuron_dropout=args.network_dropout,
                **net_kwargs,
            )
        if network is None:
            return
        network_has_multiplier = hasattr(network, "set_multiplier")

        if hasattr(network, "prepare_network"):
            network.prepare_network(args)
        if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
            logger.warning(
                "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
            )
            args.scale_weight_norms = False

        train_unet = not args.network_train_text_encoder_only
        train_text_encoder = self.is_train_text_encoder(args)
        network.apply_to(text_encoder, unet, train_text_encoder, train_unet)

        # Configure LoRA delta fake-quantization if available
        if (((getattr(args, "dq_delta_step", None) is not None and args.dq_delta_step) or (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits) or dq_bits_sched) and hasattr(network, "set_delta_fake_quant")):
            unwrapped = accelerator.unwrap_model(network)
            unwrapped.set_delta_fake_quant(
                getattr(args, "dq_delta_step", None),
                args.dq_delta_mode,
                granularity=args.dq_delta_granularity,
                stat=args.dq_delta_stat,
                bits=getattr(args, "dq_delta_bits", None),
                range_mul=getattr(args, "dq_delta_range_mul", None),
            )
            # no EMA-based stats to propagate (ema_* removed)
            # Scope control: unet / te / both
            scope = getattr(args, "dq_delta_scope", "both")
            if scope == "unet" and hasattr(unwrapped, "text_encoder_loras"):
                for l in unwrapped.text_encoder_loras:
                    l.delta_q_enabled = False
                for l in unwrapped.unet_loras:
                    l.delta_q_enabled = True
            elif scope == "te" and hasattr(unwrapped, "unet_loras"):
                for l in unwrapped.unet_loras:
                    l.delta_q_enabled = False
                for l in unwrapped.text_encoder_loras:
                    l.delta_q_enabled = True

        if args.network_weights is not None:
            # FIXME consider alpha of weights
            info = network.load_weights(args.network_weights)
            accelerator.print(f"load network weights from {args.network_weights}: {info}")

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            for t_enc in text_encoders:
                t_enc.gradient_checkpointing_enable()
            del t_enc
            network.enable_gradient_checkpointing()  # may have no effect

        # 学習に必要なクラスを準備する
        accelerator.print("prepare optimizer, data loader etc.")

        # 後方互換性を確保するよ
        try:
            results = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr, args.learning_rate)
            if type(results) is tuple:
                trainable_params = results[0]
                lr_descriptions = results[1]
            else:
                trainable_params = results
                lr_descriptions = None
        except TypeError as e:
            # logger.warning(f"{e}")
            # accelerator.print(
            #     "Deprecated: use prepare_optimizer_params(text_encoder_lr, unet_lr, learning_rate) instead of prepare_optimizer_params(text_encoder_lr, unet_lr)"
            # )
            trainable_params = network.prepare_optimizer_params(args.text_encoder_lr, args.unet_lr)
            lr_descriptions = None

        # if len(trainable_params) == 0:
        #     accelerator.print("no trainable parameters found / 学習可能なパラメータが見つかりませんでした")
        # for params in trainable_params:
        #     for k, v in params.items():
        #         if type(v) == float:
        #             pass
        #         else:
        #             v = len(v)
        #         accelerator.print(f"trainable_params: {k} = {v}")

        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)

        # dataloaderを準備する
        # DataLoaderのプロセス数：0 は persistent_workers が使えないので注意
        n_workers = min(args.max_data_loader_n_workers, os.cpu_count())  # cpu_count or max_data_loader_n_workers

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_group,
            batch_size=1,
            shuffle=True,
            collate_fn=collator,
            num_workers=n_workers,
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
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

        # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
        if args.full_fp16:
            assert (
                args.mixed_precision == "fp16"
            ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
            accelerator.print("enable full fp16 training.")
            network.to(weight_dtype)
        elif args.full_bf16:
            assert (
                args.mixed_precision == "bf16"
            ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
            accelerator.print("enable full bf16 training.")
            network.to(weight_dtype)

        unet_weight_dtype = te_weight_dtype = weight_dtype
        # Experimental Feature: Put base model into fp8 to save vram
        if args.fp8_base:
            assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
            assert (
                args.mixed_precision != "no"
            ), "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
            accelerator.print("enable fp8 training.")
            unet_weight_dtype = torch.float8_e4m3fn
            te_weight_dtype = torch.float8_e4m3fn

        unet.requires_grad_(False)
        unet.to(dtype=unet_weight_dtype)
        for t_enc in text_encoders:
            t_enc.requires_grad_(False)

            # in case of cpu, dtype is already set to fp32 because cpu does not support fp8/fp16/bf16
            if t_enc.device.type != "cpu":
                t_enc.to(dtype=te_weight_dtype)
                # nn.Embedding not support FP8
                t_enc.text_model.embeddings.to(dtype=(weight_dtype if te_weight_dtype != weight_dtype else te_weight_dtype))

        # acceleratorがなんかよろしくやってくれるらしい / accelerator will do something good
        if args.deepspeed:
            ds_model = deepspeed_utils.prepare_deepspeed_model(
                args,
                unet=unet if train_unet else None,
                text_encoder1=text_encoders[0] if train_text_encoder else None,
                text_encoder2=text_encoders[1] if train_text_encoder and len(text_encoders) > 1 else None,
                network=network,
            )
            ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                ds_model, optimizer, train_dataloader, lr_scheduler
            )
            training_model = ds_model
        else:
            if train_unet:
                unet = accelerator.prepare(unet)
            else:
                unet.to(accelerator.device, dtype=unet_weight_dtype)  # move to device because unet is not prepared by accelerator
            if train_text_encoder:
                if len(text_encoders) > 1:
                    text_encoder = text_encoders = [accelerator.prepare(t_enc) for t_enc in text_encoders]
                else:
                    text_encoder = accelerator.prepare(text_encoder)
                    text_encoders = [text_encoder]
            else:
                pass  # if text_encoder is not trained, no need to prepare. and device and dtype are already set

            network, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                network, optimizer, train_dataloader, lr_scheduler
            )
            training_model = network

        if args.gradient_checkpointing:
            # according to TI example in Diffusers, train is required
            unet.train()
            for t_enc in text_encoders:
                t_enc.train()

                # set top parameter requires_grad = True for gradient checkpointing works
                if train_text_encoder:
                    t_enc.text_model.embeddings.requires_grad_(True)

        else:
            unet.eval()
            for t_enc in text_encoders:
                t_enc.eval()

        del t_enc

        accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)

        if not cache_latents:  # キャッシュしない場合はVAEを使うのでVAEを準備する
            vae.requires_grad_(False)
            vae.eval()
            vae.to(accelerator.device, dtype=vae_dtype)

        # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
        if args.full_fp16:
            train_util.patch_accelerator_for_fp16_training(accelerator)

        # before resuming make hook for saving/loading to save/load the network weights only
        def save_model_hook(models, weights, output_dir):
            # pop weights of other models than network to save only network weights
            # only main process or deepspeed https://github.com/huggingface/diffusers/issues/2606
            if accelerator.is_main_process or args.deepspeed:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, type(accelerator.unwrap_model(network))):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)
                # print(f"save model hook: {len(weights)} weights will be saved")

            # save current ecpoch and step
            train_state_file = os.path.join(output_dir, "train_state.json")
            # +1 is needed because the state is saved before current_step is set from global_step
            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {current_step.value+1}")
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump({"current_epoch": current_epoch.value, "current_step": current_step.value + 1}, f)

        steps_from_state = None

        def load_model_hook(models, input_dir):
            # remove models except network
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, type(accelerator.unwrap_model(network))):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)
            # print(f"load model hook: {len(models)} models will be loaded")

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

        # epoch数を計算する
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
            args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

        cp_window = deque(maxlen=args.avg_window) if args.avg_cp else None
        if args.avg_cp and args.resume:
            ext = "." + args.save_model_as
            model_name = train_util.default_if_none(args.output_name, train_util.DEFAULT_EPOCH_NAME)
            for p in collect_last_checkpoints(args.output_dir, model_name, ext, args.avg_window):
                cp_window.append(load_lora_state_dict(p))

        # 学習する
        # TODO: find a way to handle total batch size when there are multiple datasets
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        accelerator.print("running training / 学習開始")
        accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
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
            "ss_text_encoder_lr": args.text_encoder_lr,
            "ss_unet_lr": args.unet_lr,
            "ss_num_train_images": train_dataset_group.num_train_images,
            "ss_num_reg_images": train_dataset_group.num_reg_images,
            "ss_num_batches_per_epoch": len(train_dataloader),
            "ss_num_epochs": num_train_epochs,
            "ss_gradient_checkpointing": args.gradient_checkpointing,
            "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
            "ss_max_train_steps": args.max_train_steps,
            "ss_lr_warmup_steps": args.lr_warmup_steps,
            "ss_lr_scheduler": args.lr_scheduler,
            "ss_network_module": args.network_module,
            "ss_network_dim": args.network_dim,  # None means default because another network than LoRA may have another default dim
            "ss_network_alpha": args.network_alpha,  # some networks may not have alpha
            "ss_network_dropout": args.network_dropout,  # some networks may not have dropout
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
            "ss_huber_c": args.huber_c,
        }

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

        # add extra args
        if args.network_args:
            metadata["ss_network_args"] = json.dumps(net_kwargs)

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

                # set epoch to start to make initial_step less than len(train_dataloader)
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
            else:
                # if not, only epoch no is skipped for informative purpose
                epoch_to_start = initial_step // math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
                initial_step = 0  # do not skip

        global_step = 0
        skipped_steps = 0

        noise_scheduler = DDPMScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
        )
        prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
        if args.zero_terminal_snr:
            custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

        if accelerator.is_main_process:
            init_kwargs = {}
            if args.wandb_run_name:
                init_kwargs["wandb"] = {"name": args.wandb_run_name}
            if args.log_tracker_config is not None:
                init_kwargs = toml.load(args.log_tracker_config)
            accelerator.init_trackers(
                "network_train" if args.log_tracker_name is None else args.log_tracker_name,
                config=train_util.get_sanitized_config_or_none(args),
                init_kwargs=init_kwargs,
            )

        loss_recorder = train_util.LossRecorder()
        del train_dataset_group

        # prepare gradient skipping if enabled (複数 GPUではrankごとに判定がズレる恐れありらしい)
        skip_grad_norm = getattr(args, "skip_grad_norm", False)
        log_grad_norm = getattr(args, "grad_norm_log", False)
        scaler_for_log = accelerator.scaler if hasattr(accelerator, "scaler") else None
        log_grad_scale = log_grad_norm and scaler_for_log is not None
        log_grad_cosine = log_grad_norm and getattr(args, "grad_cosine_log", False)
        skip_grad_norm_max = getattr(args, "skip_grad_norm_max", None)
        nan_to_window = getattr(args, "nan_to_window", False)
        inf_to_window = getattr(args, "inf_to_window", False)
        skip_nan_immediate = getattr(args, "skip_nan_immediate", True)
        skip_inf_immediate = getattr(args, "skip_inf_immediate", True)
        nan_inf_until_step = getattr(args, "nan_inf_until_step", None)
        auto_cap_release = getattr(args, "auto_cap_release", False)
        cap_release_trigger_ratio = getattr(args, "cap_release_trigger_ratio", 0.66)
        cap_release_trigger_steps = getattr(args, "cap_release_trigger_steps", 200)
        cap_release_length = getattr(args, "cap_release_length", 200)
        cap_release_scale = getattr(args, "cap_release_scale", 3.0)
        idle_free_phase = getattr(args, "idle_free_phase", False)
        idle_max_steps = getattr(args, "idle_max_steps", 4000)
        idle_free_len = getattr(args, "idle_free_len", 200)
        nan_inf_switched = False
        logger.info(
            f"skip_grad_norm: {skip_grad_norm}, grad_norm_log: {log_grad_norm}, "
            f"skip_grad_norm_max: {skip_grad_norm_max}, nan_to_window: {nan_to_window}, "
            f"inf_to_window: {inf_to_window}, skip_nan_immediate: {skip_nan_immediate}, "
            f"skip_inf_immediate: {skip_inf_immediate}, nan_inf_until_step: {nan_inf_until_step}, auto_cap_release: {auto_cap_release}, "
            f"cap_release_trigger_ratio: {cap_release_trigger_ratio}, cap_release_trigger_steps: {cap_release_trigger_steps}, "
            f"cap_release_length: {cap_release_length}, cap_release_scale: {cap_release_scale}, "
            f"idle_free_phase: {idle_free_phase}, idle_max_steps: {idle_max_steps}, idle_free_len: {idle_free_len}"
        )
        use_grad_norm = skip_grad_norm or log_grad_norm
        if use_grad_norm:
            # 勾配ノルムの履歴を保持するキュー
            moving_avg_window = deque(maxlen=200)
            log_buffer = []
            model_name = train_util.default_if_none(args.output_name, train_util.DEFAULT_LAST_OUTPUT_NAME)
            log_file_path = f"gradient_logs+{model_name}.txt"

            # ログ用のヘッダーを書き出す
            if log_grad_norm:
                with open(log_file_path, "w") as f:
                    header = "Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff"
                    if log_grad_scale:
                        header += ",Scale"
                    if log_grad_cosine:
                        header += ",CosineSim"
                    f.write(header + "\n")

            trigger_count = 0
            cap_release_counter = 0
            # For cosine logging, keep previous gradients as per-parameter tensors to avoid huge concat each step
            prev_grad_list = None
            prev_grad_norm = None
            idle_counter = 0
            idle_free_counter = 0
            in_idle_free = False

            def check_gradients_and_skip_update(model, epoch, step, loss_val):
                nonlocal trigger_count, cap_release_counter
                nonlocal prev_grad_list, prev_grad_norm, idle_counter, idle_free_counter, in_idle_free
                device = next(model.parameters()).device
                grad_norm_sqr = torch.tensor(0.0, device=device)
                # accumulate dot product with previous grads without concatenation
                dot_sum = torch.tensor(0.0, device=device) if (log_grad_cosine and prev_grad_list is not None) else None
                cur_grads = [] if log_grad_cosine else None

                # 各パラメータの勾配ノルムの二乗を加算
                with torch.no_grad():
                    idx = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            grad = param.grad
                            # sum of squares
                            grad_norm_sqr += (grad.detach() * grad.detach()).sum()
                            if log_grad_cosine:
                                if prev_grad_list is not None and idx < len(prev_grad_list):
                                    # accumulate dot product per-parameter to avoid giant torch.cat
                                    dot_sum += (grad.detach() * prev_grad_list[idx]).sum()
                                cur_grads.append(grad.detach().clone())
                                idx += 1
                current_grad_norm = torch.sqrt(grad_norm_sqr).item()

                cosine_sim = None
                if log_grad_cosine:
                    if prev_grad_list is not None and dot_sum is not None and prev_grad_norm is not None:
                        denom = (current_grad_norm * (prev_grad_norm + 1e-12))
                        # avoid divide-by-zero
                        if denom == 0.0:
                            cosine_sim = float("nan")
                        else:
                            cosine_sim = (dot_sum / denom).item()
                    else:
                        cosine_sim = float("nan")
                    # store for next step
                    prev_grad_list = cur_grads
                    prev_grad_norm = current_grad_norm

                # 勾配ノルムが NaN / Inf かどうかを判定
                is_nan = math.isnan(current_grad_norm)
                is_inf = math.isinf(current_grad_norm)

                appended = False
                if not is_nan and not is_inf:
                    moving_avg_window.append(current_grad_norm)
                else:
                    # 設定に応じて NaN / Inf も移動平均窓へ入れる
                    if is_nan and nan_to_window:
                        moving_avg_window.append(current_grad_norm)
                        appended = True
                    if is_inf and inf_to_window:
                        moving_avg_window.append(current_grad_norm)
                        appended = True

                if in_idle_free:
                    idle_free_counter += 1
                    if idle_free_counter >= idle_free_len:
                        in_idle_free = False
                        idle_free_counter = 0
                else:
                    if appended:
                        idle_counter = 0
                    else:
                        idle_counter += 1
                        if idle_free_phase and idle_counter >= idle_max_steps:
                            in_idle_free = True
                            idle_counter = 0
                            idle_free_counter = 0

                # 窓がいっぱいになったら平均+2.5σを計算
                if len(moving_avg_window) == moving_avg_window.maxlen:
                    mean_norm = np.mean(moving_avg_window)
                    std_norm = np.std(moving_avg_window)
                    # 移動平均 + 2.5σ で動的閾値を算出
                    dynamic_threshold_pre_cap = mean_norm + 2.5 * std_norm
                else:
                    # 窓が埋まるまでは十分大きな閾値を与える
                    dynamic_threshold_pre_cap = 200_000

                # auto_cap_release の処理：連続して閾値を超えている場合に一時的に上限を緩和
                effective_max = skip_grad_norm_max
                if auto_cap_release and skip_grad_norm_max is not None:
                    if cap_release_counter > 0:
                        # キャップ解放中は閾値を強制的に拡大した値にする
                        effective_max = skip_grad_norm_max * cap_release_scale
                        dynamic_threshold_pre_cap = effective_max
                        cap_release_counter -= 1
                    else:
                        # 閾値超過が一定回数続いたらキャップ開放を開始
                        if not math.isnan(dynamic_threshold_pre_cap) and dynamic_threshold_pre_cap >= cap_release_trigger_ratio * skip_grad_norm_max:
                            trigger_count += 1
                            if trigger_count >= cap_release_trigger_steps:
                                cap_release_counter = cap_release_length
                                trigger_count = 0
                        else:
                            trigger_count = 0

                # 有効な上限値を反映した実際の閾値を求める
                dynamic_threshold = dynamic_threshold_pre_cap
                if effective_max is not None and dynamic_threshold > effective_max:
                    dynamic_threshold = effective_max
                # 窓が埋まるまでは計算値をそのまま使用する
                if len(moving_avg_window) < moving_avg_window.maxlen:
                    dynamic_threshold = dynamic_threshold_pre_cap

                # ログをファイルに出力
                if log_grad_norm:
                    scale_val = scaler_for_log.get_scale() if log_grad_scale else None
                    flag = 2 if in_idle_free else (1 if math.isnan(dynamic_threshold) else 0)
                    log_line = f"{epoch},{step},{current_grad_norm},{dynamic_threshold},{loss_val},{flag}"
                    if log_grad_scale:
                        log_line += f",{scale_val}"
                    if log_grad_cosine:
                        log_line += f",{cosine_sim}"
                    log_buffer.append(log_line + "\n")
                    if step % 100 == 0:
                        with open(log_file_path, "a") as f:
                            f.writelines(log_buffer)
                        log_buffer.clear()

                if not skip_grad_norm:
                    return False

                if (is_nan and skip_nan_immediate) or (is_inf and skip_inf_immediate):
                    return True

                if in_idle_free:
                    return False

                return current_grad_norm > dynamic_threshold
        else:
            def check_gradients_and_skip_update(model, epoch, step, loss_val):
                return False

        # callback for step start
        if hasattr(accelerator.unwrap_model(network), "on_step_start"):
            on_step_start = accelerator.unwrap_model(network).on_step_start
        else:
            on_step_start = lambda *args, **kwargs: None

        # function for saving/removing
        def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_file = os.path.join(args.output_dir, ckpt_name)

            accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
            metadata["ss_training_finished_at"] = str(time.time())
            metadata["ss_steps"] = str(steps)
            metadata["ss_epoch"] = str(epoch_no)

            metadata_to_save = minimum_metadata if args.no_metadata else metadata
            sai_metadata = train_util.get_sai_model_spec(None, args, self.is_sdxl, True, False)
            metadata_to_save.update(sai_metadata)

            unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)
            if args.huggingface_repo_id is not None:
                huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)

        def remove_model(old_ckpt_name):
            old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
            if os.path.exists(old_ckpt_file):
                accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
                os.remove(old_ckpt_file)

        # For --sample_at_first
        self.sample_images(accelerator, args, 0, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

        # training loop
        if initial_step > 0:  # only if skip_until_initial_step is specified
            for skip_epoch in range(epoch_to_start):  # skip epochs
                logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {initial_step}")
                initial_step -= len(train_dataloader)
            global_step = initial_step

        for epoch in range(epoch_to_start, num_train_epochs):
            accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
            current_epoch.value = epoch + 1

            metadata["ss_epoch"] = str(epoch + 1)

            accelerator.unwrap_model(network).on_epoch_start(text_encoder, unet)

            skipped_dataloader = None
            if initial_step > 0:
                skipped_dataloader = accelerator.skip_first_batches(train_dataloader, initial_step - 1)
                initial_step = 1

            # track last applied bits to avoid redundant updates
            last_applied_bits = getattr(args, "dq_delta_bits", None)

            for step, batch in enumerate(skipped_dataloader or train_dataloader):
                current_step.value = global_step
                if initial_step > 0:
                    initial_step -= 1
                    continue

                with accelerator.accumulate(training_model):
                    # Toggle delta fake-quantization based on training progress
                    if hasattr(accelerator.unwrap_model(network), "set_delta_quant_enabled"):
                        dq_configured = (
                            (getattr(args, "dq_delta_step", None) is not None and args.dq_delta_step)
                            or (getattr(args, "dq_delta_bits", None) is not None and args.dq_delta_bits)
                            or bool(dq_bits_sched)
                        )
                        if dq_configured:
                            progress_frac = (global_step / float(args.max_train_steps)) if args.max_train_steps > 0 else 1.0
                            accelerator.unwrap_model(network).set_delta_quant_enabled(progress_frac >= args.dq_delta_begin)

                            # Apply bits scheduling if specified
                            if dq_bits_sched:
                                cur_bits = last_applied_bits
                                for p, b in dq_bits_sched:
                                    if progress_frac >= p:
                                        cur_bits = b
                                    else:
                                        break
                                if cur_bits != last_applied_bits:
                                    accelerator.unwrap_model(network).set_delta_fake_quant(
                                        getattr(args, "dq_delta_step", None),
                                        args.dq_delta_mode,
                                        granularity=args.dq_delta_granularity,
                                        stat=args.dq_delta_stat,
                                        bits=cur_bits,
                                        range_mul=getattr(args, "dq_delta_range_mul", None),
                                    )
                                    last_applied_bits = cur_bits

                    on_step_start(text_encoder, unet)

                    if "latents" in batch and batch["latents"] is not None:
                        latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                    else:
                        if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                            with torch.no_grad():
                                # latentに変換
                                latents = vae.encode(batch["images"].to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype)
                        else:
                            chunks = [batch["images"][i:i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)]
                            list_latents = []
                            for chunk in chunks:
                                with torch.no_grad():
                                # latentに変換
                                    list_latents.append(vae.encode(chunk.to(dtype=vae_dtype)).latent_dist.sample().to(dtype=weight_dtype))
                            latents = torch.cat(list_latents, dim=0)
                            # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                    latents = latents * self.vae_scale_factor

                    # get multiplier for each sample
                    if network_has_multiplier:
                        multipliers = batch["network_multipliers"]
                        # if all multipliers are same, use single multiplier
                        if torch.all(multipliers == multipliers[0]):
                            multipliers = multipliers[0].item()
                        else:
                            raise NotImplementedError("multipliers for each sample is not supported yet")
                        # print(f"set multiplier: {multipliers}")
                        accelerator.unwrap_model(network).set_multiplier(multipliers)

                    with torch.set_grad_enabled(train_text_encoder), accelerator.autocast():
                        # Get the text embedding for conditioning
                        if args.weighted_captions:
                            text_encoder_conds = get_weighted_text_embeddings(
                                tokenizer,
                                text_encoder,
                                batch["captions"],
                                accelerator.device,
                                args.max_token_length // 75 if args.max_token_length else 1,
                                clip_skip=args.clip_skip,
                            )
                        else:
                            text_encoder_conds = self.get_text_cond(
                                args, accelerator, batch, tokenizers, text_encoders, weight_dtype
                            )

                    # Sample noise, sample a random timestep for each image, and add noise to the latents,
                    # with noise offset and/or multires noise if specified
                    noise, noisy_latents, timesteps, huber_c = train_util.get_noise_noisy_latents_and_timesteps(
                        args, noise_scheduler, latents
                    )

                    # ensure the hidden state will require grad
                    if args.gradient_checkpointing:
                        for x in noisy_latents:
                            x.requires_grad_(True)
                        for t in text_encoder_conds:
                            t.requires_grad_(True)

                    # Predict the noise residual
                    with accelerator.autocast():
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

                    if args.v_parameterization:
                        # v-parameterization training
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        target = noise

                    loss = train_util.conditional_loss(
                        noise_pred.float(), target.float(), reduction="none", loss_type=args.loss_type, huber_c=huber_c
                    )
                    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                        loss = apply_masked_loss(loss, batch)
                    loss = loss.mean([1, 2, 3])

                    loss_weights = batch["loss_weights"]  # 各sampleごとのweight
                    loss = loss * loss_weights

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
                    if args.debiased_estimation_loss:
                        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)

                    loss = loss.mean()  # 平均なのでbatch_sizeで割る必要なし

                    accelerator.backward(loss)
                    skip_step = False
                    if check_gradients_and_skip_update(network, epoch, step, loss.detach().item()):
                        accelerator.print(
                            f"\nSkipping update at Epoch: {epoch}, Step: {step} due to large gradients."
                        )
                        skipped_steps += 1
                        optimizer.zero_grad(set_to_none=True)
                        skip_step = True

                    if not skip_step:
                        if accelerator.sync_gradients:
                            self.all_reduce_network(accelerator, network)  # sync DDP grad manually
                            if args.max_grad_norm != 0.0:
                                params_to_clip = accelerator.unwrap_model(network).get_trainable_params()
                                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                        # Optional: quantize/round LoRA trainable parameters after each optimizer step
                        if (
                            args.round_lora_step is not None
                            and args.round_lora_step > 0
                            and accelerator.sync_gradients
                        ):
                            # step index after this update
                            next_step_idx = global_step + 1
                            # respect warmup for rounding based on overall training progress
                            progress_frac = next_step_idx / float(args.max_train_steps)
                            if progress_frac >= args.round_lora_begin and (next_step_idx % max(1, args.round_lora_every) == 0):
                                round_parameters(
                                    accelerator.unwrap_model(network).get_trainable_params(),
                                    step=args.round_lora_step,
                                    mode=args.round_lora_mode,
                                )

                if args.scale_weight_norms:
                    keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
                        args.scale_weight_norms, accelerator.device
                    )
                    max_mean_logs = {"Keys Scaled": keys_scaled, "Average key norm": mean_norm}
                else:
                    keys_scaled, mean_norm, maximum_norm = None, None, None

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if (
                        not nan_inf_switched
                        and nan_inf_until_step is not None
                        and global_step >= nan_inf_until_step
                    ):
                        nan_to_window = False
                        inf_to_window = False
                        skip_nan_immediate = True
                        skip_inf_immediate = True
                        nan_inf_switched = True
                        accelerator.print(
                            f"nan_inf switched at step {global_step}: nan_to_window={nan_to_window}, inf_to_window={inf_to_window}, "
                            f"skip_nan_immediate={skip_nan_immediate}, skip_inf_immediate={skip_inf_immediate}"
                        )

                    self.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

                    # 指定ステップごとにモデルを保存
                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
                            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch)

                            if args.save_state:
                                train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

                            remove_step_no = train_util.get_remove_step_no(args, global_step)
                            if remove_step_no is not None:
                                remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
                                remove_model(remove_ckpt_name)

                current_loss = loss.detach().item()
                loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
                avr_loss: float = loss_recorder.moving_average
                logs = {"avr_loss": avr_loss}
                if skip_grad_norm:
                    logs["skipped"] = skipped_steps
                progress_bar.set_postfix(**logs)

                if args.scale_weight_norms:
                    progress_bar.set_postfix(**{**max_mean_logs, **logs})

                if args.logging_dir is not None:
                    logs = self.generate_step_logs(
                        args, current_loss, avr_loss, lr_scheduler, lr_descriptions, keys_scaled, mean_norm, maximum_norm
                    )
                    if skip_grad_norm:
                        logs["train/skipped_steps"] = skipped_steps
                    accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if args.logging_dir is not None:
                logs = {"loss/epoch": loss_recorder.moving_average}
                accelerator.log(logs, step=epoch + 1)

            accelerator.wait_for_everyone()

            # 指定エポックごとにモデルを保存
            if args.save_every_n_epochs is not None:
                saving = (epoch + 1) % args.save_every_n_epochs == 0 and (epoch + 1) < num_train_epochs
                if is_main_process and saving:
                    ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch + 1)
                    save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch + 1)

                    remove_epoch_no = train_util.get_remove_epoch_no(args, epoch + 1)
                    if remove_epoch_no is not None:
                        remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                        remove_model(remove_ckpt_name)

                    if args.save_state:
                        train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch + 1)

            self.sample_images(accelerator, args, epoch + 1, global_step, accelerator.device, vae, tokenizer, text_encoder, unet)

            if args.avg_cp and (epoch + 1) / num_train_epochs >= args.avg_begin:
                sd = filter_lora_state_dict(accelerator.unwrap_model(network).state_dict())
                cp_window.append(sd)
                if len(cp_window) == args.avg_window:
                    start_ep = epoch - args.avg_window + 2
                    if start_ep < 1:
                        start_ep = 1
                    logger.info(f"averaging checkpoints from epoch {start_ep} to {epoch + 1}")
                    avg_sd = average_state_dicts(list(cp_window), args.avg_mode)
                    accelerator.unwrap_model(network).load_state_dict(avg_sd, strict=False)
                    if args.avg_reset_stats:
                        for p_state in optimizer.state.values():
                            p_state["step"] = p_state.get("step", 0)  # keep real count
                            for buf in ("exp_avg", "exp_avg_sq", "exp_avg_max"):
                                if buf in p_state and isinstance(p_state[buf], torch.Tensor):
                                    p_state[buf].zero_()
                    if accelerator.distributed_type != DistributedType.NO:
                        sd = broadcast(accelerator.unwrap_model(network).state_dict())
                        accelerator.unwrap_model(network).load_state_dict(sd, strict=False)
                        accelerator.wait_for_everyone()
                    else:
                        accelerator.wait_for_everyone()

            # end of epoch

        # metadata["ss_epoch"] = str(num_train_epochs)
        metadata["ss_training_finished_at"] = str(time.time())

        if log_grad_norm and len(log_buffer) > 0:
            with open(log_file_path, "a") as f:
                f.writelines(log_buffer)
            log_buffer.clear()

        if is_main_process:
            network = accelerator.unwrap_model(network)

        accelerator.end_training()

        if is_main_process and (args.save_state or args.save_state_on_train_end):
            train_util.save_state_on_train_end(args, accelerator)

        if is_main_process:
            ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
            save_model(ckpt_name, network, global_step, num_train_epochs, force_sync_upload=True)

            logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, True)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)

    parser.add_argument(
        "--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを出力先モデルに保存しない"
    )
    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )

    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")
    parser.add_argument("--text_encoder_lr", type=float, default=None, help="learning rate for Text Encoder / Text Encoderの学習率")

    parser.add_argument(
        "--network_weights", type=str, default=None, help="pretrained weights for network / 学習するネットワークの初期重み"
    )
    parser.add_argument(
        "--network_module", type=str, default=None, help="network module to train / 学習対象のネットワークのモジュール"
    )
    parser.add_argument(
        "--network_dim",
        type=int,
        default=None,
        help="network dimensions (depends on each network) / モジュールの次元数（ネットワークにより定義は異なります）",
    )
    parser.add_argument(
        "--network_alpha",
        type=float,
        default=1,
        help="alpha for LoRA weight scaling, default 1 (same as network_dim for same behavior as old version) / LoRaの重み調整のalpha値、デフォルト1（旧バージョンと同じ動作をするにはnetwork_dimと同じ値を指定）",
    )
    parser.add_argument(
        "--network_dropout",
        type=float,
        default=None,
        help="Drops neurons out of training every step (0 or None is default behavior (no dropout), 1 would drop all neurons) / 訓練時に毎ステップでニューロンをdropする（0またはNoneはdropoutなし、1は全ニューロンをdropout）",
    )
    parser.add_argument(
        "--network_args",
        type=str,
        default=None,
        nargs="*",
        help="additional arguments for network (key=value) / ネットワークへの追加の引数",
    )
    parser.add_argument(
        "--network_train_unet_only", action="store_true", help="only training U-Net part / U-Net関連部分のみ学習する"
    )
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="only training Text Encoder part / Text Encoder関連部分のみ学習する",
    )
    parser.add_argument(
        "--training_comment",
        type=str,
        default=None,
        help="arbitrary comment string stored in metadata / メタデータに記録する任意のコメント文字列",
    )
    parser.add_argument(
        "--dim_from_weights",
        action="store_true",
        help="automatically determine dim (rank) from network_weights / dim (rank)をnetwork_weightsで指定した重みから自動で決定する",
    )
    parser.add_argument(
        "--scale_weight_norms",
        type=float,
        default=None,
        help="Scale the weight of each key pair to help prevent overtraing via exploding gradients. (1 is a good starting point) / 重みの値をスケーリングして勾配爆発を防ぐ（1が初期値としては適当）",
    )
    parser.add_argument(
        "--base_weights",
        type=str,
        default=None,
        nargs="*",
        help="network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みファイル",
    )
    parser.add_argument(
        "--base_weights_multiplier",
        type=float,
        default=None,
        nargs="*",
        help="multiplier for network weights to merge into the model before training / 学習前にあらかじめモデルにマージするnetworkの重みの倍率",
    )
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
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
    parser.add_argument("--avg_cp", action="store_true", help="enable inter-epoch checkpoint averaging / エポック間のチェックポイント平均を有効化")
    parser.add_argument("--avg_window", type=int, default=4, help="number of checkpoints to average / 平均するチェックポイント数")
    parser.add_argument("--avg_begin", type=float, default=0.6, help="fraction of total epochs to start averaging / 学習の何割から平均を開始するか")
    parser.add_argument(
        "--avg_mode",
        type=str,
        default="ema",
        choices=["uniform", "ema", "metric"],
        help="averaging mode: uniform, ema or metric / 平均化モード",
    )
    parser.add_argument(
        "--avg_reset_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="reset optimizer stats after averaging / 平均化後にOptimizer統計をリセットする",
    )
    # LoRA delta fake-quantization (on forward only)
    parser.add_argument(
        "--dq_delta_step",
        type=float,
        default=None,
        help="Fake-quantize only LoRA delta output per forward with this step (STE). None/<=0 to disable / LoRAの差分出力のみをこの刻みでフェイク量子化（STE）。Noneまたは<=0で無効",
    )
    parser.add_argument(
        "--dq_delta_mode",
        type=str,
        default="det",
        choices=["det", "stoch"],
        help="Fake-quant mode: det or stoch / フェイク量子化モード：det=最近傍、stoch=確率的",
    )
    parser.add_argument(
        "--dq_delta_begin",
        type=float,
        default=0.0,
        help="Enable fake-quant after this fraction of total steps [0-1] / 学習進行率がこの割合を超えてから有効化 [0-1]",
    )
    parser.add_argument(
        "--dq_delta_scope",
        type=str,
        default="both",
        choices=["unet", "te", "both"],
        help="Apply delta fake-quant to: unet, te, or both / Δのフェイク量子化の適用範囲（unet/te/both）",
    )
    parser.add_argument(
        "--dq_delta_granularity",
        type=str,
        default="tensor",
        choices=["tensor", "channel"],
        help="Granularity of delta fake-quant: whole tensor or per-channel / Δのフェイク量子化の粒度（テンソル全体/チャネル別）",
    )
    parser.add_argument(
        "--dq_delta_stat",
        type=str,
        default="rms",
        choices=["rms", "absmax", "none"],
        help="Statistic for scale/step: rms/absmax/none. Channel-wise when granularity=channel. / スケール/ステップの統計：rms/absmax/none（granularity=channelでチャネル別）",
    )
    parser.add_argument(
        "--dq_delta_bits",
        type=int,
        default=None,
        help="If set, use N-bit symmetric fake-quant (overrides step path). Recommended: 8 / Nビット対称フェイク量子化（step指定より優先）。推奨: 8",
    )
    parser.add_argument(
        "--dq_delta_range_mul",
        type=float,
        default=3.0,
        help="When bits mode with stat=rms, dynamic range = range_mul * RMS. / bitsモードかつstat=rms時の有効レンジ倍率（range=倍率×RMS）",
    )
    parser.add_argument(
        "--dq_delta_bits_sched",
        type=str,
        default=None,
        help="Schedule bits by progress fraction, e.g. '0.0:6,0.5:8,0.8:10' / 学習進行率に応じたビット数スケジュール（例: '0.0:6,0.5:8,0.8:10'）",
    )
    # ema_* options removed
    # LoRA rounding options
    parser.add_argument(
        "--round_lora_step",
        type=float,
        default=None,
        help="Round LoRA trainable weights to multiples of this step after optimizer step (disabled if None or <= 0) / Optimizer更新後にLoRAの学習パラメータをこの刻みに丸める（Noneまたは<=0で無効）",
    )
    parser.add_argument(
        "--round_lora_mode",
        type=str,
        default="det",
        choices=["det", "stoch"],
        help="Rounding mode: det (deterministic) or stoch (stochastic) / 丸めモード：det=最近傍、stoch=確率的",
    )
    parser.add_argument(
        "--round_lora_every",
        type=int,
        default=1,
        help="Apply rounding every N optimizer steps (only when gradients sync) / 丸めを適用するステップ間隔（同期更新時のみ）",
    )
    parser.add_argument(
        "--round_lora_begin",
        type=float,
        default=0.0,
        help="Begin rounding after this fraction of total steps [0-1] / 学習全体のこの進行率以降で丸めを開始 [0-1]",
    )
    # parser.add_argument("--loraplus_lr_ratio", default=None, type=float, help="LoRA+ learning rate ratio")
    # parser.add_argument("--loraplus_unet_lr_ratio", default=None, type=float, help="LoRA+ UNet learning rate ratio")
    # parser.add_argument("--loraplus_text_encoder_lr_ratio", default=None, type=float, help="LoRA+ text encoder learning rate ratio")
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = NetworkTrainer()
    trainer.train(args)
