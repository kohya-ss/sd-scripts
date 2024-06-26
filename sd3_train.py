# training with captions

import argparse
import copy
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library.device_utils import init_ipex, clean_memory_on_device


init_ipex()

from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import deepspeed_utils, sd3_models, sd3_train_utils, sd3_utils

# , sdxl_model_util

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

# import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions

# from library.custom_train_functions import (
#     apply_snr_weight,
#     prepare_scheduler_for_custom_training,
#     scale_v_prediction_loss_like_noise_prediction,
#     add_v_prediction_like_loss,
#     apply_debiased_estimation,
#     apply_masked_loss,
# )


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    # sdxl_train_util.verify_sdxl_training_args(args)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    assert (
        not args.weighted_captions
    ), "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    # if args.block_lr:
    #     block_lrs = [float(lr) for lr in args.block_lr.split(",")]
    #     assert (
    #         len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
    #     ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    # else:
    #     block_lrs = None

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # load tokenizer
    sd3_tokenizer = sd3_models.SD3Tokenizer()

    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
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

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[sd3_tokenizer])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [sd3_tokenizer])

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(8)  # TODO これでいいか確認

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

    # acceleratorを準備する
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = weight_dtype  # torch.float32 if args.no_half_vae else weight_dtype # SD3 VAE works with fp16

    t5xxl_dtype = weight_dtype
    if args.t5xxl_dtype is not None:
        if args.t5xxl_dtype == "fp16":
            t5xxl_dtype = torch.float16
        elif args.t5xxl_dtype == "bf16":
            t5xxl_dtype = torch.bfloat16
        elif args.t5xxl_dtype == "fp32" or args.t5xxl_dtype == "float":
            t5xxl_dtype = torch.float32
        else:
            raise ValueError(f"unexpected t5xxl_dtype: {args.t5xxl_dtype}")
    t5xxl_device = accelerator.device if args.t5xxl_device is None else args.t5xxl_device

    # モデルを読み込む
    attn_mode = "xformers" if args.xformers else "torch"

    assert (
        attn_mode == "torch"
    ), f"attn_mode {attn_mode} is not supported. Please use `--sdpa` instead of `--xformers`. / attn_mode {attn_mode} はサポートされていません。`--xformers`の代わりに`--sdpa`を使ってください。"

    mmdit, clip_l, clip_g, t5xxl, vae = sd3_train_utils.load_target_model(
        args, accelerator, attn_mode, weight_dtype, t5xxl_device, t5xxl_dtype
    )
    assert clip_l is not None, "clip_l is required / clip_lは必須です"
    assert clip_g is not None, "clip_g is required / clip_gは必須です"
    # logit_scale = logit_scale.to(accelerator.device, dtype=weight_dtype)

    # 学習を準備する
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()
        vae_wrapper = sd3_models.VAEWrapper(vae)  # make SD/SDXL compatible
        with torch.no_grad():
            train_dataset_group.cache_latents(
                vae_wrapper, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process, file_suffix="_sd3.npz"
            )
        vae.to("cpu")
        clean_memory_on_device(accelerator.device)

        accelerator.wait_for_everyone()

    # 学習を準備する：モデルを適切な状態にする
    if args.gradient_checkpointing:
        mmdit.enable_gradient_checkpointing()
    train_mmdit = args.learning_rate != 0
    train_clip_l = False
    train_clip_g = False
    train_t5xxl = False

    # if args.train_text_encoder:
    #     # TODO each option for two text encoders?
    #     accelerator.print("enable text encoder training")
    #     if args.gradient_checkpointing:
    #         text_encoder1.gradient_checkpointing_enable()
    #         text_encoder2.gradient_checkpointing_enable()
    #     lr_te1 = args.learning_rate_te1 if args.learning_rate_te1 is not None else args.learning_rate  # 0 means not train
    #     lr_te2 = args.learning_rate_te2 if args.learning_rate_te2 is not None else args.learning_rate  # 0 means not train
    #     train_clip_l = lr_te1 != 0
    #     train_clip_g = lr_te2 != 0

    #     # caching one text encoder output is not supported
    #     if not train_clip_l:
    #         text_encoder1.to(weight_dtype)
    #     if not train_clip_g:
    #         text_encoder2.to(weight_dtype)
    #     text_encoder1.requires_grad_(train_clip_l)
    #     text_encoder2.requires_grad_(train_clip_g)
    #     text_encoder1.train(train_clip_l)
    #     text_encoder2.train(train_clip_g)
    # else:
    clip_l.to(weight_dtype)
    clip_g.to(weight_dtype)
    clip_l.requires_grad_(False)
    clip_g.requires_grad_(False)
    clip_l.eval()
    clip_g.eval()
    if t5xxl is not None:
        t5xxl.to(t5xxl_dtype)
        t5xxl.requires_grad_(False)
        t5xxl.eval()

    # TextEncoderの出力をキャッシュする
    if args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad

        with torch.no_grad(), accelerator.autocast():
            train_dataset_group.cache_text_encoder_outputs_sd3(
                sd3_tokenizer,
                (clip_l, clip_g, t5xxl),
                (accelerator.device, accelerator.device, t5xxl_device),
                None,
                (None, None, None),
                args.cache_text_encoder_outputs_to_disk,
                accelerator.is_main_process,
                args.text_encoder_batch_size,
            )
        accelerator.wait_for_everyone()

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    mmdit.requires_grad_(train_mmdit)
    if not train_mmdit:
        mmdit.to(accelerator.device, dtype=weight_dtype)  # because of unet is not prepared

    training_models = []
    params_to_optimize = []
    # if train_unet:
    training_models.append(mmdit)
    # if block_lrs is None:
    params_to_optimize.append({"params": list(mmdit.parameters()), "lr": args.learning_rate})
    # else:
    #     params_to_optimize.extend(get_block_params_to_optimize(mmdit, block_lrs))

    # if train_clip_l:
    #     training_models.append(text_encoder1)
    #     params_to_optimize.append({"params": list(text_encoder1.parameters()), "lr": args.learning_rate_te1 or args.learning_rate})
    # if train_clip_g:
    #     training_models.append(text_encoder2)
    #     params_to_optimize.append({"params": list(text_encoder2.parameters()), "lr": args.learning_rate_te2 or args.learning_rate})

    # calculate number of trainable parameters
    n_params = 0
    for group in params_to_optimize:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train mmdit: {train_mmdit}")  # , text_encoder1: {train_clip_l}, text_encoder2: {train_clip_g}")
    accelerator.print(f"number of models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params}")

    # 学習に必要なクラスを準備する
    accelerator.print("prepare optimizer, data loader etc.")

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
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)

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
    if args.fused_optimizer_groups:
        # prepare lr schedulers for each optimizer
        lr_schedulers = [train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes) for optimizer in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in the following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        mmdit.to(weight_dtype)
        clip_l.to(weight_dtype)
        clip_g.to(weight_dtype)
        if t5xxl is not None:
            t5xxl.to(weight_dtype)  # TODO check works with fp16 or not
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        mmdit.to(weight_dtype)
        clip_l.to(weight_dtype)
        clip_g.to(weight_dtype)
        if t5xxl is not None:
            t5xxl.to(weight_dtype)

    # TODO check if this is necessary. SD3 uses pool for clip_l and clip_g
    # # freeze last layer and final_layer_norm in te1 since we use the output of the penultimate layer
    # if train_clip_l:
    #     text_encoder1.text_model.encoder.layers[-1].requires_grad_(False)
    #     text_encoder1.text_model.final_layer_norm.requires_grad_(False)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            mmdit=mmdit,
            # mmdie=mmdit if train_mmdit else None,
            # text_encoder1=text_encoder1 if train_clip_l else None,
            # text_encoder2=text_encoder2 if train_clip_g else None,
        )
        # most of ZeRO stage uses optimizer partitioning, so we have to prepare optimizer and ds_model at the same time. # pull/1139#issuecomment-1986790007
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]

    else:
        # acceleratorがなんかよろしくやってくれるらしい
        if train_mmdit:
            mmdit = accelerator.prepare(mmdit)
        # if train_clip_l:
        #     text_encoder1 = accelerator.prepare(text_encoder1)
        # if train_clip_g:
        #     text_encoder2 = accelerator.prepare(text_encoder2)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # TextEncoderの出力をキャッシュするときには、すでに出力を取得済みなのでCPUへ移動する
    if args.cache_text_encoder_outputs:
        # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        clip_l.to("cpu", dtype=torch.float32)
        clip_g.to("cpu", dtype=torch.float32)
        if t5xxl is not None:
            t5xxl.to("cpu", dtype=torch.float32)
        clean_memory_on_device(accelerator.device)
    else:
        # make sure Text Encoders are on GPU
        # TODO support CPU for text encoders
        clip_l.to(accelerator.device)
        clip_g.to(accelerator.device)
        if t5xxl is not None:
            t5xxl.to(accelerator.device)

    # TODO cache sample prompt's embeddings to free text encoder's memory
    if args.cache_text_encoder_outputs:
        if not args.save_t5xxl:
            t5xxl = None  # free memory
    clean_memory_on_device(accelerator.device)

    # 実験的機能：勾配も含めたfp16学習を行う　PyTorchにパッチを当ててfp16でのgrad scaleを有効にする
    if args.full_fp16:
        # During deepseed training, accelerate not handles fp16/bf16|mixed precision directly via scaler. Let deepspeed engine do.
        # -> But we think it's ok to patch accelerator even if deepspeed is enabled.
        train_util.patch_accelerator_for_fp16_training(accelerator)

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
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num examples / サンプル数: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {len(train_dataloader)}")
    accelerator.print(f"  num epochs / epoch数: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    # noise_scheduler = DDPMScheduler(
    #     beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    # )

    noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    # if args.zero_terminal_snr:
    #     custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    if accelerator.is_main_process:
        init_kwargs = {}
        if args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers(
            "finetuning" if args.log_tracker_name is None else args.log_tracker_name,
            config=train_util.get_sanitized_config_or_none(args),
            init_kwargs=init_kwargs,
        )

    # # For --sample_at_first
    # sd3_train_utils.sample_images(
    #     accelerator, args, 0, global_step, accelerator.device, vae, [tokenizer1, tokenizer2], [text_encoder1, text_encoder2], mmdit
    # )

    # following function will be moved to sd3_train_utils

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
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

    loss_recorder = train_util.LossRecorder()
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.fused_optimizer_groups:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device).to(dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # encode images to latents. images are [-1, 1]
                        latents = vae.encode(batch["images"].to(vae_dtype)).to(weight_dtype)

                        # NaNが含まれていれば警告を表示し0に置き換える
                        if torch.any(torch.isnan(latents)):
                            accelerator.print("NaN found in latents, replacing with zeros")
                            latents = torch.nan_to_num(latents, 0, out=latents)
                # latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
                latents = sd3_models.SDVAE.process_in(latents)

                if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
                    # not cached, get text encoder outputs
                    # XXX This does not work yet
                    input_ids_clip_l, input_ids_clip_g, input_ids_t5xxl = batch["input_ids"]
                    with torch.set_grad_enabled(args.train_text_encoder):
                        # TODO support weighted captions
                        # TODO support length > 75
                        input_ids_clip_l = input_ids_clip_l.to(accelerator.device)
                        input_ids_clip_g = input_ids_clip_g.to(accelerator.device)
                        input_ids_t5xxl = input_ids_t5xxl.to(accelerator.device)

                        # get text encoder outputs: outputs are concatenated
                        context, pool = sd3_utils.get_cond_from_tokens(
                            input_ids_clip_l, input_ids_clip_g, input_ids_t5xxl, clip_l, clip_g, t5xxl
                        )
                else:
                    # encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(accelerator.device).to(weight_dtype)
                    # encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(accelerator.device).to(weight_dtype)
                    # pool2 = batch["text_encoder_pool2_list"].to(accelerator.device).to(weight_dtype)
                    # TODO this reuses SDXL keys, it should be fixed
                    lg_out = batch["text_encoder_outputs1_list"]
                    t5_out = batch["text_encoder_outputs2_list"]
                    pool = batch["text_encoder_pool2_list"]
                    context = torch.cat([lg_out, t5_out], dim=-2)

                # TODO support some features for noise implemented in get_noise_noisy_latents_and_timesteps

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=latents.ndim, dtype=weight_dtype)
                noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

                # call model
                with accelerator.autocast():
                    model_pred = mmdit(noisy_model_input, timesteps, context=context, y=pool)

                # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
                # Preconditioning of the model outputs.
                model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                target = latents

                # Compute regular loss. TODO simplify this
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)

                if not (args.fused_backward_pass or args.fused_optimizer_groups):
                    if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                else:
                    # optimizer.step() and optimizer.zero_grad() are called in the optimizer hook
                    lr_scheduler.step()
                    if args.fused_optimizer_groups:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # sdxl_train_util.sample_images(
                #     accelerator,
                #     args,
                #     None,
                #     global_step,
                #     accelerator.device,
                #     vae,
                #     [tokenizer1, tokenizer2],
                #     [text_encoder1, text_encoder2],
                #     mmdit,
                # )

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        sd3_train_utils.save_sd3_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            clip_l if args.save_clip else None,
                            clip_g if args.save_clip else None,
                            t5xxl if args.save_t5xxl else None,
                            mmdit,
                            vae,
                        )

            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs(logs, lr_scheduler, args.optimizer_type, including_unet=train_mmdit)

                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                sd3_train_utils.save_sd3_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    clip_l if args.save_clip else None,
                    clip_g if args.save_clip else None,
                    t5xxl if args.save_t5xxl else None,
                    mmdit,
                    vae,
                )

        # sdxl_train_util.sample_images(
        #     accelerator,
        #     args,
        #     epoch + 1,
        #     global_step,
        #     accelerator.device,
        #     vae,
        #     [tokenizer1, tokenizer2],
        #     [text_encoder1, text_encoder2],
        #     mmdit,
        # )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    mmdit = accelerator.unwrap_model(mmdit)
    clip_l = accelerator.unwrap_model(clip_l)
    clip_g = accelerator.unwrap_model(clip_g)
    if t5xxl is not None:
        t5xxl = accelerator.unwrap_model(t5xxl)

    accelerator.end_training()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        sd3_train_utils.save_sd3_model_on_train_end(
            args,
            save_dtype,
            epoch,
            global_step,
            clip_l if args.save_clip else None,
            clip_g if args.save_clip else None,
            t5xxl if args.save_t5xxl else None,
            mmdit,
            vae,
        )
        logger.info("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    add_logging_arguments(parser)
    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_masked_loss_arguments(parser)
    deepspeed_utils.add_deepspeed_arguments(parser)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sd3_train_utils.add_sd3_training_arguments(parser)

    # TE training is disabled temporarily

    # parser.add_argument(
    #     "--learning_rate_te1",
    #     type=float,
    #     default=None,
    #     help="learning rate for text encoder 1 (ViT-L) / text encoder 1 (ViT-L)の学習率",
    # )
    # parser.add_argument(
    #     "--learning_rate_te2",
    #     type=float,
    #     default=None,
    #     help="learning rate for text encoder 2 (BiG-G) / text encoder 2 (BiG-G)の学習率",
    # )

    # parser.add_argument(
    #     "--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する"
    # )
    # parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    # parser.add_argument(
    #     "--no_half_vae",
    #     action="store_true",
    #     help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    # )
    # parser.add_argument(
    #     "--block_lr",
    #     type=str,
    #     default=None,
    #     help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
    #     + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    # )
    parser.add_argument(
        "--fused_optimizer_groups",
        type=int,
        default=None,
        help="number of optimizers for fused backward pass and optimizer step / fused backward passとoptimizer stepのためのoptimizer数",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
