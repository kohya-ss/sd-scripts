# training with captions
import faulthandler
faulthandler.enable()

import argparse
import gc
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm
import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        from library.ipex import ipex_init
        ipex_init()
except Exception:
    pass
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from library import sdxl_model_util

import library.train_util as train_util
import library.config_util as config_util
import library.sdxl_train_util as sdxl_train_util
from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
import library.custom_train_functions as custom_train_functions
from library.custom_train_functions import (
    apply_snr_weight,
    prepare_scheduler_for_custom_training,
    scale_v_prediction_loss_like_noise_prediction,
    add_v_prediction_like_loss,
)
from library.sdxl_original_unet import SdxlUNet2DConditionModel
import psutil



UNET_NUM_BLOCKS_FOR_BLOCK_LR = 23


def get_block_params_to_optimize(unet: SdxlUNet2DConditionModel, block_lrs: List[float]) -> List[dict]:
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


def append_block_lr_to_logs(block_lrs, logs, lr_scheduler, optimizer_type):
    lrs = lr_scheduler.get_last_lr()

    lr_index = 0
    block_index = 0
    while lr_index < len(lrs):
        if block_index < UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            name = f"block{block_index}"
            if block_lrs[block_index] == 0:
                block_index += 1
                continue
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR:
            name = "text_encoder1"
        elif block_index == UNET_NUM_BLOCKS_FOR_BLOCK_LR + 1:
            name = "text_encoder2"
        else:
            raise ValueError(f"unexpected block_index: {block_index}")

        block_index += 1

        logs["lr/" + name] = float(lrs[lr_index])

        if optimizer_type.lower().startswith("DAdapt".lower()) or optimizer_type.lower() == "Prodigy".lower():
            logs["lr/d*lr/" + name] = (
                lr_scheduler.optimizers[-1].param_groups[lr_index]["d"] * lr_scheduler.optimizers[-1].param_groups[lr_index]["lr"]
            )

        lr_index += 1

def test(training_models, accelerator, args, global_step, 
        text_encoder1, text_encoder2, tokenizer1, tokenizer2, vae, unet, weight_dtype, vae_dtype):
    print("start test")
    for m in training_models: #清空梯度
        for param in m.parameters():
            param.grad = None
        m.eval()
    torch.cuda.empty_cache()
    gc.collect()
    
    text_encoder1.to(accelerator.device, dtype=weight_dtype)
    text_encoder2.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=vae_dtype)
    with torch.no_grad():
        sdxl_train_util.sample_images(
            accelerator,
            args,
            None,
            global_step,
            accelerator.device,
            vae,
            [tokenizer1, tokenizer2],
            [text_encoder1, text_encoder2],
            unet,
        )

    all_model_to_training_mode(args, accelerator, text_encoder1, text_encoder2, vae, unet, weight_dtype, vae_dtype, training_models)
    print("start end")

def vae_encode(vae, images, device, vae_dtype):
    vae.to(device, dtype=vae_dtype)
    images = images.to(device, dtype=vae_dtype)
    with torch.no_grad():
        # latentに変換
        latents = vae.encode().latent_dist.sample()

        # NaNが含まれていれば警告を表示し0に置き換える
        if torch.any(torch.isnan(latents)):
            print("NaN found in latents, replacing with zeros")
            latents = torch.where(torch.isnan(latents), torch.zeros_like(latents), latents)
    return latents

def get_letents(vae, batch, device, vae_dtype, weight_dtype):
    if "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"].to(device, dtype=weight_dtype)
    else:
        latents = vae_encode(vae, batch["images"], device, vae_dtype)
    latents = latents * sdxl_model_util.VAE_SCALE_FACTOR
    return latents

def get_text_embedding(text_encoder1, text_encoder2, tokenizer1, tokenizer2, batch, device, weight_dtype, args):
    if "text_encoder_outputs1_list" not in batch or batch["text_encoder_outputs1_list"] is None:
        input_ids1 = batch["input_ids"]
        input_ids2 = batch["input_ids2"]
        with torch.set_grad_enabled(args.train_text_encoder):
            input_ids1 = input_ids1.to(device, dtype=weight_dtype)
            input_ids2 = input_ids2.to(device, dtype=weight_dtype)
            encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                args.max_token_length,
                input_ids1,
                input_ids2,
                tokenizer1,
                tokenizer2,
                text_encoder1,
                text_encoder2,
                None if not args.full_fp16 else weight_dtype,
            )
    else:
        encoder_hidden_states1 = batch["text_encoder_outputs1_list"].to(device, dtype=weight_dtype)
        encoder_hidden_states2 = batch["text_encoder_outputs2_list"].to(device, dtype=weight_dtype)
        pool2 = batch["text_encoder_pool2_list"].to(device, dtype=weight_dtype)
    return encoder_hidden_states1, encoder_hidden_states2, pool2
def get_preload_data(vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2, noise_scheduler,
                     batch, device, vae_dtype, weight_dtype, args):
    with torch.no_grad():
        latents = get_letents(vae, batch, device, vae_dtype, weight_dtype)
        print("latents done")

        encoder_hidden_states1, encoder_hidden_states2, pool2 = get_text_embedding(text_encoder1, text_encoder2, tokenizer1, tokenizer2, batch, device, weight_dtype, args)
        print("text encode done")

        # get size embeddings
        orig_size = batch["original_sizes_hw"]
        crop_size = batch["crop_top_lefts"]
        target_size = batch["target_sizes_hw"]
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, device)

        # concat embeddings
        vector_embedding = torch.cat([pool2, embs], dim=1)
        text_embedding = torch.cat([encoder_hidden_states1, encoder_hidden_states2], dim=2)

        # Sample noise, sample a random timestep for each image, and add noise to the latents,
        # with noise offset and/or multires noise if specified
        noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(args, noise_scheduler, latents)

        latents = latents.to(device, dtype=weight_dtype)
        noisy_latents = noisy_latents.to(device, dtype=weight_dtype)
        noise = noise.to(device, dtype=weight_dtype)
        timesteps = timesteps.to(device, dtype=weight_dtype)
        text_embedding = text_embedding.to(device, dtype=weight_dtype)
        vector_embedding = vector_embedding.to(device, dtype=weight_dtype)
    return noise, noisy_latents, timesteps, text_embedding, vector_embedding
def get_dataset_and_dataloader(args, tokenizer1, tokenizer2, current_epoch, current_step):
    use_dreambooth_method = args.in_json is None
    # データセットを準備する
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, False, True))
        if args.dataset_config is not None:
            print(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                print(
                    "ignore following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                print("Using DreamBooth method.")
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
                print("Training with captions.")
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

        blueprint = blueprint_generator.generate(user_config, args, tokenizer=[tokenizer1, tokenizer2])
        train_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args, [tokenizer1, tokenizer2])

    if args.cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used / text encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"



    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(32)

    if args.debug_dataset:
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        print(
            "No data found. Please verify the metadata file and train_data_dir option. / 画像がありません。メタデータおよびtrain_data_dirオプションを確認してください。"
        )
        return
    
    # dataloaderを準備する
    # DataLoaderのプロセス数：0はメインプロセスになる
    n_workers = min(args.max_data_loader_n_workers, os.cpu_count() - 1)  # cpu_count-1 ただし最大で指定された数まで
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # データセット側にも学習ステップを送信
    train_dataset_group.set_max_train_steps(args.max_train_steps)
    return train_dataset_group, train_dataloader

def sitting_xformers(vae, unet, args):
    # Diffusers版のxformers使用フラグを設定する関数
    def set_diffusers_xformers_flag(model, valid):
        def fn_recursive_set_mem_eff(module: torch.nn.Module):
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(valid)

            for child in module.children():
                fn_recursive_set_mem_eff(child)

        fn_recursive_set_mem_eff(model)

    # モデルに xformers とか memory efficient attention を組み込む
    if args.diffusers_xformers:
        # もうU-Netを独自にしたので動かないけどVAEのxformersは動くはず
        print("Use xformers by Diffusers")
        # set_diffusers_xformers_flag(unet, True)
        set_diffusers_xformers_flag(vae, True)
    else:
        # Windows版のxformersはfloatで学習できなかったりするのでxformersを使わない設定も可能にしておく必要がある
        print("Disable Diffusers' xformers")
        train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
        if torch.__version__ >= "2.0.0":  # PyTorch 2.0.0 以上対応のxformersなら以下が使える
            vae.set_use_memory_efficient_attention_xformers(args.xformers)
def apply_block_lr(block_lrs, training_models):
    if block_lrs is None:
        params = []
        for m in training_models:
            params.extend(m.parameters())
        params_to_optimize = params
    else:
        params_to_optimize = get_block_params_to_optimize(training_models[0], block_lrs)  # U-Net
        for m in training_models[1:]:  # Text Encoders if exists
            params_to_optimize.append({"params": m.parameters(), "lr": args.learning_rate})      
             
    return params_to_optimize
def calc_n_params(params_to_optimize):
    n_params = 0
    for params in params_to_optimize:
        if(type(params) == dict):
            for p in params["params"]:
                n_params += p.numel()
        else:
            n_params += params.numel()
    return n_params
def all_model_to_training_mode(args, accelerator, text_encoder1, text_encoder2, vae, unet, weight_dtype, vae_dtype, training_models):
    # 切換模型到訓練模式下正確的設備與精度
    if not args.train_text_encoder and args.cache_text_encoder_outputs: # move Text Encoders for sampling images. Text Encoder doesn't work on CPU with fp16
        text_encoder1.to("cpu", dtype=weight_dtype)
        text_encoder2.to("cpu", dtype=weight_dtype)
        text_encoder1.requires_grad_(False)
        text_encoder2.requires_grad_(False)
        text_encoder1.eval()
        text_encoder2.eval()
    else: # make sure Text Encoders are on GPU
        text_encoder1.to(accelerator.device, dtype=weight_dtype)
        text_encoder2.to(accelerator.device, dtype=weight_dtype)
        text_encoder1.requires_grad_(True)
        text_encoder2.requires_grad_(True)
        text_encoder1.train()
        text_encoder2.train()
        
    vae.to("cpu", dtype=vae_dtype)
    vae.requires_grad_(False)
    vae.eval()
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.requires_grad_(True)
    unet.train()
    for m in training_models:
        m.to(accelerator.device, dtype=weight_dtype)
        m.requires_grad_(True)
        m.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
def get_memory_usage():
    memory_info = psutil.virtual_memory()
    return memory_info.used / 1024 / 1024
def train(args):
    #參數確定與初始化
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    sdxl_train_util.verify_sdxl_training_args(args)

    assert not args.weighted_captions, "weighted_captions is not supported currently / weighted_captionsは現在サポートされていません"
    assert (
        not args.train_text_encoder or not args.cache_text_encoder_outputs
    ), "cache_text_encoder_outputs is not supported when training text encoder / text encoderを学習するときはcache_text_encoder_outputsはサポートされていません"

    if args.block_lr:
        block_lrs = [float(lr) for lr in args.block_lr.split(",")]
        assert (
            len(block_lrs) == UNET_NUM_BLOCKS_FOR_BLOCK_LR
        ), f"block_lr must have {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / block_lrは{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値を指定してください"
    else:
        block_lrs = None

    if args.seed is not None:
        set_seed(args.seed)  # 乱数系列を初期化する

    # mixed precisionに対応した型を用意しておき適宜castする
    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = torch.float32 if args.no_half_vae else weight_dtype
    # 実験的機能：勾配も含めたfp16/bf16学習を行う　モデル全体をfp16/bf16にする
    if args.full_fp16:
        assert (
            args.mixed_precision == "fp16"
        ), "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
    elif args.full_bf16:
        assert (
            args.mixed_precision == "bf16"
        ), "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"

    


    # acceleratorを準備する
    print("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    
    #載入模型
    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
    (
        load_stable_diffusion_format,
        text_encoder1,
        text_encoder2,
        vae,
        unet,
        logit_scale,
        ckpt_info,
    ) = sdxl_train_util.load_target_model(args, accelerator, "sdxl", weight_dtype)
    sitting_xformers(vae, unet, args)

    # verify load/save model formats
    if load_stable_diffusion_format:
        src_stable_diffusion_ckpt = args.pretrained_model_name_or_path
        src_diffusers_model_path = None
    else:
        src_stable_diffusion_ckpt = None
        src_diffusers_model_path = args.pretrained_model_name_or_path

    if args.save_model_as is None:
        save_stable_diffusion_format = load_stable_diffusion_format
        use_safetensors = args.use_safetensors
    else:
        save_stable_diffusion_format = args.save_model_as.lower() == "ckpt" or args.save_model_as.lower() == "safetensors"
        use_safetensors = args.use_safetensors or ("safetensors" in args.save_model_as.lower())
        # assert save_stable_diffusion_format, "save_model_as must be ckpt or safetensors / save_model_asはckptかsafetensorsである必要があります"
    

    #設定所有模型的梯度計算
    training_models = []
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    training_models.append(unet)
    all_model_to_training_mode(args, accelerator, text_encoder1, text_encoder2, vae, unet, weight_dtype, vae_dtype, training_models)

    #計算模型參數數量
    params_to_optimize = apply_block_lr(block_lrs, training_models)
    n_params = calc_n_params(params_to_optimize)
    print(f"number of models: {len(training_models)}")
    print(f"number of trainable parameters: {n_params}")
    
    #載入dataset and dataloader
    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    train_dataset_group, train_dataloader = get_dataset_and_dataloader(args, tokenizer1, tokenizer2, current_epoch, current_step)
    
    #預跑特徵提取 減少訓練記憶體需求
    if not args.train_text_encoder and args.cache_text_encoder_outputs:
        # Text Encodes are eval and no grad
        with torch.no_grad():
            train_dataset_group.cache_text_encoder_outputs(
                (tokenizer1, tokenizer2),
                (text_encoder1, text_encoder2),
                accelerator.device,
                None,
                args.cache_text_encoder_outputs_to_disk,
                accelerator.is_main_process,
            )
        accelerator.wait_for_everyone()
            
    if args.cache_latents:
        with torch.no_grad():
            train_dataset_group.cache_latents(vae, args.vae_batch_size, args.cache_latents_to_disk, accelerator.is_main_process)
        accelerator.wait_for_everyone()
    

    # 学習に必要なクラスを準備する
    print("prepare optimizer, data loader etc.")
    _, _, optimizer = train_util.get_optimizer(args, trainable_params=params_to_optimize)
    
    # 学習ステップ数を計算する
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}")

    # lr schedulerを用意する
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # acceleratorがなんかよろしくやってくれるらしい
    if args.train_text_encoder:
        unet, text_encoder1, text_encoder2, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder1, text_encoder2, optimizer, train_dataloader, lr_scheduler
        )

        # transform DDP after prepare
        # text_encoder1, text_encoder2, unet = train_util.transform_models_if_DDP([text_encoder1, text_encoder2, unet]) #目前沒有分散式訓練的需求 先關閉
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
        # (unet,) = train_util.transform_models_if_DDP([unet]) #目前沒有分散式訓練的需求 先關閉
    
    # resumeする
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

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
    accelerator.print(f"  batch size per device / バッチサイズ: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}")
    # accelerator.print(
    #     f"  total train batch size (with parallel & distributed & accumulation) / 総バッチサイズ（並列学習、勾配合計含む）: {total_batch_size}"
    # )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = args.global_step #設定起始步數

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, clip_sample=False
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.log_tracker_config is not None:
            init_kwargs = toml.load(args.log_tracker_config)
        accelerator.init_trackers("finetuning" if args.log_tracker_name is None else args.log_tracker_name, init_kwargs=init_kwargs)
    
    if (args.sample_every_n_steps is not None):
        test(training_models, accelerator, args, global_step, text_encoder1, text_encoder2, tokenizer1, tokenizer2, vae, unet, weight_dtype, vae_dtype)

    # input("確認記憶體有清除...")
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1
        loss_total = 0
        for step, batch in enumerate(train_dataloader):
            print("start")
            current_step.value = global_step
            noise, noisy_latents, timesteps, text_embedding, vector_embedding = get_preload_data(vae, text_encoder1, text_encoder2, tokenizer1, tokenizer2, noise_scheduler, batch, accelerator.device, vae_dtype, weight_dtype, args)
            
            with accelerator.accumulate(training_models[0]):  # 複数モデルに対応していない模様だがとりあえずこうしておく
                print("unet before")
                # print(f"noisy_latents:{noisy_latents.dtype}, timesteps:{timesteps.dtype}, text_embedding:{text_embedding.dtype}, vector_embedding:{vector_embedding.dtype}, unet:{unet.dtype}")
                with accelerator.autocast():
                    noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
                print("unet done")

                if args.min_snr_gamma or args.scale_v_pred_loss_like_noise_pred or args.v_pred_like_loss:
                    # do not mean over batch dimension for snr weight or scale v-pred loss
                    loss = torch.nn.functional.mse_loss(noise_pred, noise.float(), reduction="none")
                    loss = loss.mean([1, 2, 3])

                    if args.min_snr_gamma:
                        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma)
                    if args.scale_v_pred_loss_like_noise_pred:
                        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
                    if args.v_pred_like_loss:
                        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)

                    loss = loss.mean()  # mean over batch dimension
                else:
                    # print(f"noise_pred.dtype:{noise_pred.dtype}, noise.dtype:{noise.dtype}")
                    loss = torch.nn.functional.mse_loss(noise_pred, noise.float(), reduction="mean")


                print("loss done", loss)
                memory_usage_in_mb = get_memory_usage()
                print(f"Current memory usage: {memory_usage_in_mb:.2f} MB")
                # torch.cuda.synchronize()
                print("synchronize done")
                accelerator.backward(loss)
                # torch.cuda.synchronize()
                print("backward done")
                if(global_step % args.accumulation_n_steps == 0):
                    print("clip_grad_norm_")
                    if args.max_grad_norm != 0.0 and accelerator.sync_gradients:
                        params_to_clip = []
                        for m in training_models:
                            params_to_clip.extend(m.parameters())
                        print("accelerator clip_grad_norm_")
                        
                        for param in params_to_clip: # 遍歷所有參數
                            if param.grad is not None:
                                # 將 NaN 梯度替換為零
                                if(torch.isnan(param.grad).any()):
                                    print("nan found")
                                param.grad.data = torch.nan_to_num(param.grad.data)

                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                    print("optimizer step")
                    optimizer.step()
                    lr_scheduler.step()

                    print("optimizer zero_grad")
                    optimizer.zero_grad(set_to_none=True)
                print("accumulate done")
            
            current_loss = loss.detach().item()  # 平均なのでbatch sizeは関係ないはず
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                
                progress_bar.update(1)
                print("args.sample_every_n_steps", args.sample_every_n_steps)
                if args.sample_every_n_steps is not None and global_step % args.sample_every_n_steps == 0:
                    test(training_models, accelerator, args, global_step, text_encoder1, text_encoder2, tokenizer1, tokenizer2, vae, unet, weight_dtype, vae_dtype)

                # 指定ステップごとにモデルを保存
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                        sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            src_path,
                            save_stable_diffusion_format,
                            use_safetensors,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(text_encoder1),
                            accelerator.unwrap_model(text_encoder2),
                            accelerator.unwrap_model(unet),
                            vae,
                            logit_scale,
                            ckpt_info,
                        )

            
            if args.logging_dir is not None:
                logs = {"loss": current_loss}
                if block_lrs is None:
                    logs["lr"] = float(lr_scheduler.get_last_lr()[0])
                    if (
                        args.optimizer_type.lower().startswith("DAdapt".lower()) or args.optimizer_type.lower() == "Prodigy".lower()
                    ):  # tracking d*lr value
                        logs["lr/d*lr"] = (
                            lr_scheduler.optimizers[0].param_groups[0]["d"] * lr_scheduler.optimizers[0].param_groups[0]["lr"]
                        )
                else:
                    append_block_lr_to_logs(block_lrs, logs, lr_scheduler, args.optimizer_type)

                accelerator.log(logs)
            global_step += 1

            # TODO moving averageにする
            loss_total += current_loss
            avr_loss = loss_total / (step + 1)
            logs = {"loss": avr_loss}  # , "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            print("step done")

        if args.logging_dir is not None:
            logs = {"loss/epoch": loss_total / len(train_dataloader)}
            accelerator.log(logs)

        accelerator.wait_for_everyone()

        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
                sdxl_train_util.save_sd_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    src_path,
                    save_stable_diffusion_format,
                    use_safetensors,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(text_encoder1),
                    accelerator.unwrap_model(text_encoder2),
                    accelerator.unwrap_model(unet),
                    vae,
                    logit_scale,
                    ckpt_info,
                )
        if args.sample_every_n_epochs is not None:
            sdxl_train_util.sample_images(
                accelerator,
                args,
                epoch + 1,
                global_step,
                accelerator.device,
                vae,
                [tokenizer1, tokenizer2],
                [text_encoder1, text_encoder2],
                unet,
            )

    is_main_process = accelerator.is_main_process
    # if is_main_process:
    unet = accelerator.unwrap_model(unet)
    text_encoder1 = accelerator.unwrap_model(text_encoder1)
    text_encoder2 = accelerator.unwrap_model(text_encoder2)

    accelerator.end_training()

    if args.save_state:  # and is_main_process:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator  # この後メモリを使うのでこれは消す

    if is_main_process:
        src_path = src_stable_diffusion_ckpt if save_stable_diffusion_format else src_diffusers_model_path
        sdxl_train_util.save_sd_model_on_train_end(
            args,
            src_path,
            save_stable_diffusion_format,
            use_safetensors,
            save_dtype,
            epoch,
            global_step,
            text_encoder1,
            text_encoder2,
            unet,
            vae,
            logit_scale,
            ckpt_info,
        )
        print("model saved.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    train_util.add_sd_models_arguments(parser)
    train_util.add_dataset_arguments(parser, True, True, True)
    train_util.add_training_arguments(parser, False)
    train_util.add_sd_saving_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    config_util.add_config_arguments(parser)
    custom_train_functions.add_custom_train_arguments(parser)
    sdxl_train_util.add_sdxl_training_arguments(parser)

    parser.add_argument("--diffusers_xformers", action="store_true", help="use xformers by diffusers / Diffusersでxformersを使用する")
    parser.add_argument("--train_text_encoder", action="store_true", help="train text encoder / text encoderも学習する")
    parser.add_argument(
        "--no_half_vae",
        action="store_true",
        help="do not use fp16/bf16 VAE in mixed precision (use float VAE) / mixed precisionでも fp16/bf16 VAEを使わずfloat VAEを使う",
    )
    parser.add_argument(
        "--block_lr",
        type=str,
        default=None,
        help=f"learning rates for each block of U-Net, comma-separated, {UNET_NUM_BLOCKS_FOR_BLOCK_LR} values / "
        + f"U-Netの各ブロックの学習率、カンマ区切り、{UNET_NUM_BLOCKS_FOR_BLOCK_LR}個の値",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)

    train(args)
