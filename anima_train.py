# Anima full finetune training script

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import math
import os
from multiprocessing import Value
from typing import List
import toml

from tqdm import tqdm

import torch
from library import utils
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from accelerate.utils import set_seed
from library import deepspeed_utils, anima_models, anima_train_utils, anima_utils, strategy_base, strategy_anima, sai_model_spec

import library.train_util as train_util

from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

import library.config_util as config_util

from library.config_util import (
    ConfigSanitizer,
    BlueprintGenerator,
)
from library.custom_train_functions import apply_masked_loss, add_custom_train_arguments


def train(args):
    train_util.verify_training_args(args)
    train_util.prepare_dataset_args(args, True)
    deepspeed_utils.prepare_deepspeed_args(args)
    setup_logging(args, reset=True)

    # backward compatibility
    if not args.skip_cache_check:
        args.skip_cache_check = args.skip_latents_validity_check

    if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
        logger.warning(
            "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
        )
        args.cache_text_encoder_outputs = True

    if args.cpu_offload_checkpointing and not args.gradient_checkpointing:
        logger.warning("cpu_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
        args.gradient_checkpointing = True

    if getattr(args, 'unsloth_offload_checkpointing', False):
        if not args.gradient_checkpointing:
            logger.warning("unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled")
            args.gradient_checkpointing = True
        assert not args.cpu_offload_checkpointing, \
            "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing"

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

    assert (
        args.blocks_to_swap is None or args.blocks_to_swap == 0
    ) or not getattr(args, 'unsloth_offload_checkpointing', False), \
        "blocks_to_swap is not supported with unsloth_offload_checkpointing"

    # Flash attention: validate availability
    if getattr(args, 'flash_attn', False):
        try:
            import flash_attn  # noqa: F401
            logger.info("Flash Attention enabled for DiT blocks")
        except ImportError:
            logger.warning("flash_attn package not installed, falling back to PyTorch SDPA")
            args.flash_attn = False

    cache_latents = args.cache_latents
    use_dreambooth_method = args.in_json is None

    if args.seed is not None:
        set_seed(args.seed)

    # prepare caching strategy: must be set before preparing dataset
    if args.cache_latents:
        latents_caching_strategy = strategy_anima.AnimaLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check
        )
        strategy_base.LatentsCachingStrategy.set_strategy(latents_caching_strategy)

    # prepare dataset
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if args.dataset_config is not None:
            logger.info(f"Load dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignore following options because config file is found: {0}".format(", ".join(ignored))
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
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    train_dataset_group.verify_bucket_reso_steps(8)  # WanVAE spatial downscale = 8

    # Anima uses embedding-level dropout (in AnimaTextEncodingStrategy) instead of
    # dataset-level caption dropout, so we save the rate and zero out subset-level
    # caption_dropout_rate to allow text encoder output caching.
    caption_dropout_rate = getattr(args, 'caption_dropout_rate', 0.0)
    if caption_dropout_rate > 0:
        logger.info(f"Using embedding-level caption dropout rate: {caption_dropout_rate}")
        for dataset in train_dataset_group.datasets:
            for subset in dataset.subsets:
                subset.caption_dropout_rate = 0.0

    if args.debug_dataset:
        if args.cache_text_encoder_outputs:
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(
                strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
                    args.cache_text_encoder_outputs_to_disk,
                    args.text_encoder_batch_size,
                    False,
                    False,
                )
            )
        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group, True)
        return
    if len(train_dataset_group) == 0:
        logger.error("No data found. Please verify the metadata file and train_data_dir option.")
        return

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used"

    if args.cache_text_encoder_outputs:
        assert (
            train_dataset_group.is_text_encoder_output_cacheable()
        ), "when caching text encoder output, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

    # prepare accelerator
    logger.info("prepare accelerator")
    accelerator = train_util.prepare_accelerator(args)

    # mixed precision dtype
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    # parse transformer_dtype
    transformer_dtype = None
    if hasattr(args, 'transformer_dtype') and args.transformer_dtype is not None:
        transformer_dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        transformer_dtype = transformer_dtype_map.get(args.transformer_dtype, None)

    # Load tokenizers and set strategies
    logger.info("Loading tokenizers...")
    qwen3_text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        args.qwen3_path, dtype=weight_dtype, device="cpu"
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(
        getattr(args, 't5_tokenizer_path', None)
    )

    # Set tokenize strategy
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer,
        t5_tokenizer=t5_tokenizer,
        qwen3_max_length=args.qwen3_max_token_length,
        t5_max_length=args.t5_max_token_length,
    )
    strategy_base.TokenizeStrategy.set_strategy(tokenize_strategy)

    # Set text encoding strategy
    caption_dropout_rate = getattr(args, 'caption_dropout_rate', 0.0)
    text_encoding_strategy = strategy_anima.AnimaTextEncodingStrategy(
        dropout_rate=caption_dropout_rate,
    )
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    # Prepare text encoder (always frozen for Anima)
    qwen3_text_encoder.to(weight_dtype)
    qwen3_text_encoder.requires_grad_(False)

    # Cache text encoder outputs
    sample_prompts_te_outputs = None
    if args.cache_text_encoder_outputs:
        qwen3_text_encoder.to(accelerator.device)
        qwen3_text_encoder.eval()

        text_encoder_caching_strategy = strategy_anima.AnimaTextEncoderOutputsCachingStrategy(
            args.cache_text_encoder_outputs_to_disk,
            args.text_encoder_batch_size,
            args.skip_cache_check,
            is_partial=False,
        )
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_caching_strategy)

        with accelerator.autocast():
            train_dataset_group.new_cache_text_encoder_outputs([qwen3_text_encoder], accelerator)

        # cache sample prompt embeddings
        if args.sample_prompts is not None:
            logger.info(f"Cache Text Encoder outputs for sample prompts: {args.sample_prompts}")
            prompts = train_util.load_prompts(args.sample_prompts)
            sample_prompts_te_outputs = {}
            with accelerator.autocast(), torch.no_grad():
                for prompt_dict in prompts:
                    for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                        if p not in sample_prompts_te_outputs:
                            logger.info(f"  cache TE outputs for: {p}")
                            tokens_and_masks = tokenize_strategy.tokenize(p)
                            sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                tokenize_strategy,
                                [qwen3_text_encoder],
                                tokens_and_masks,
                                enable_dropout=False,
                            )

        accelerator.wait_for_everyone()

        # free text encoder memory
        qwen3_text_encoder = None
        clean_memory_on_device(accelerator.device)

    # Load VAE and cache latents
    logger.info("Loading Anima VAE...")
    vae, vae_mean, vae_std, vae_scale = anima_utils.load_anima_vae(args.vae_path, dtype=weight_dtype, device="cpu")

    if cache_latents:
        vae.to(accelerator.device, dtype=weight_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    # Load DiT (MiniTrainDIT + optional LLM Adapter)
    logger.info("Loading Anima DiT...")
    dit = anima_utils.load_anima_dit(
        args.dit_path,
        dtype=weight_dtype,
        device="cpu",
        transformer_dtype=transformer_dtype,
        llm_adapter_path=getattr(args, 'llm_adapter_path', None),
        disable_mmap=getattr(args, 'disable_mmap_load_safetensors', False),
    )

    if args.gradient_checkpointing:
        dit.enable_gradient_checkpointing(
            cpu_offload=args.cpu_offload_checkpointing,
            unsloth_offload=getattr(args, 'unsloth_offload_checkpointing', False),
        )

    if getattr(args, 'flash_attn', False):
        dit.set_flash_attn(True)

    train_dit = args.learning_rate != 0
    dit.requires_grad_(train_dit)
    if not train_dit:
        dit.to(accelerator.device, dtype=weight_dtype)

    # Block swap
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        logger.info(f"Enable block swap: blocks_to_swap={args.blocks_to_swap}")
        dit.enable_block_swap(args.blocks_to_swap, accelerator.device)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=weight_dtype)
        # Move scale tensors to same device as VAE for on-the-fly encoding
        vae_scale = [s.to(accelerator.device) if isinstance(s, torch.Tensor) else s for s in vae_scale]

    # Setup optimizer with parameter groups
    if train_dit:
        param_groups = anima_train_utils.get_anima_param_groups(
            dit,
            base_lr=args.learning_rate,
            self_attn_lr=getattr(args, 'self_attn_lr', None),
            cross_attn_lr=getattr(args, 'cross_attn_lr', None),
            mlp_lr=getattr(args, 'mlp_lr', None),
            mod_lr=getattr(args, 'mod_lr', None),
            llm_adapter_lr=getattr(args, 'llm_adapter_lr', None),
        )
    else:
        param_groups = []

    training_models = []
    if train_dit:
        training_models.append(dit)

    # calculate trainable parameters
    n_params = 0
    for group in param_groups:
        for p in group["params"]:
            n_params += p.numel()

    accelerator.print(f"train dit: {train_dit}")
    accelerator.print(f"number of training models: {len(training_models)}")
    accelerator.print(f"number of trainable parameters: {n_params:,}")

    # prepare optimizer
    accelerator.print("prepare optimizer, data loader etc.")

    if args.blockwise_fused_optimizers:
        # Split params into per-block groups for blockwise fused optimizer
        # Build param_id → lr mapping from param_groups to propagate per-component LRs
        param_lr_map = {}
        for group in param_groups:
            for p in group['params']:
                param_lr_map[id(p)] = group['lr']

        grouped_params = []
        param_group = {}
        named_parameters = list(dit.named_parameters())
        for name, p in named_parameters:
            if not p.requires_grad:
                continue
            # Determine block type and index
            if name.startswith("blocks."):
                block_index = int(name.split(".")[1])
                block_type = "blocks"
            elif name.startswith("llm_adapter.blocks."):
                block_index = int(name.split(".")[2])
                block_type = "llm_adapter"
            else:
                block_index = -1
                block_type = "other"

            param_group_key = (block_type, block_index)
            if param_group_key not in param_group:
                param_group[param_group_key] = []
            param_group[param_group_key].append(p)

        for param_group_key, params in param_group.items():
            # Use per-component LR from param_groups if available
            lr = param_lr_map.get(id(params[0]), args.learning_rate)
            grouped_params.append({"params": params, "lr": lr})
            num_params = sum(p.numel() for p in params)
            accelerator.print(f"block {param_group_key}: {num_params} parameters, lr={lr}")

        # Create per-group optimizers
        optimizers = []
        for group in grouped_params:
            _, _, opt = train_util.get_optimizer(args, trainable_params=[group])
            optimizers.append(opt)
        optimizer = optimizers[0]  # avoid error in following code

        logger.info(f"using {len(optimizers)} optimizers for blockwise fused optimizers")

        if train_util.is_schedulefree_optimizer(optimizers[0], args):
            raise ValueError("Schedule-free optimizer is not supported with blockwise fused optimizers")
        optimizer_train_fn = lambda: None
        optimizer_eval_fn = lambda: None
    elif args.fused_backward_pass:
        # Pass per-component param_groups directly to preserve per-component LRs
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=param_groups)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)
    else:
        _, _, optimizer = train_util.get_optimizer(args, trainable_params=param_groups)
        optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    # prepare dataloader
    train_dataset_group.set_current_strategies()

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    # calculate training steps
    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(f"override steps. steps for {args.max_train_epochs} epochs: {args.max_train_steps}")

    train_dataset_group.set_max_train_steps(args.max_train_steps)

    # lr scheduler
    if args.blockwise_fused_optimizers:
        lr_schedulers = [train_util.get_scheduler_fix(args, opt, accelerator.num_processes) for opt in optimizers]
        lr_scheduler = lr_schedulers[0]  # avoid error in following code
    else:
        lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    # full fp16/bf16 training
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed_precision='fp16'"
        accelerator.print("enable full fp16 training.")
        dit.to(weight_dtype)
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed_precision='bf16'"
        accelerator.print("enable full bf16 training.")
        dit.to(weight_dtype)

    # move text encoder to GPU if not cached
    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        qwen3_text_encoder.to(accelerator.device)

    clean_memory_on_device(accelerator.device)

    # Prepare with accelerator
    # Temporarily move non-training models off GPU to reduce memory during DDP init
    # if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
    #     qwen3_text_encoder.to("cpu")
    # if not cache_latents and vae is not None:
    #     vae.to("cpu")
    # clean_memory_on_device(accelerator.device)

    if args.deepspeed:
        ds_model = deepspeed_utils.prepare_deepspeed_model(args, mmdit=dit)
        ds_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, lr_scheduler
        )
        training_models = [ds_model]
    else:
        if train_dit:
            dit = accelerator.prepare(dit, device_placement=[not is_swapping_blocks])
            if is_swapping_blocks:
                accelerator.unwrap_model(dit).move_to_device_except_swap_blocks(accelerator.device)
        optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)

    # Move non-training models back to GPU
    if not args.cache_text_encoder_outputs and qwen3_text_encoder is not None:
        qwen3_text_encoder.to(accelerator.device)
    if not cache_latents and vae is not None:
        vae.to(accelerator.device, dtype=weight_dtype)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    # resume
    train_util.resume_from_local_or_hf_if_specified(accelerator, args)

    if args.fused_backward_pass:
        import library.adafactor_fused

        library.adafactor_fused.patch_adafactor_fused(optimizer)

        for param_group in optimizer.param_groups:
            for parameter in param_group["params"]:
                if parameter.requires_grad:

                    def create_grad_hook(p_group):
                        def grad_hook(tensor: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(tensor, args.max_grad_norm)
                            optimizer.step_param(tensor, p_group)
                            tensor.grad = None

                        return grad_hook

                    parameter.register_post_accumulate_grad_hook(create_grad_hook(param_group))

    elif args.blockwise_fused_optimizers:
        # Prepare additional optimizers and lr schedulers
        for i in range(1, len(optimizers)):
            optimizers[i] = accelerator.prepare(optimizers[i])
            lr_schedulers[i] = accelerator.prepare(lr_schedulers[i])

        # Counters for blockwise gradient hook
        optimizer_hooked_count = {}
        num_parameters_per_group = [0] * len(optimizers)
        parameter_optimizer_map = {}

        for opt_idx, opt in enumerate(optimizers):
            for param_group in opt.param_groups:
                for parameter in param_group["params"]:
                    if parameter.requires_grad:

                        def grad_hook(parameter: torch.Tensor):
                            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                                accelerator.clip_grad_norm_(parameter, args.max_grad_norm)

                            i = parameter_optimizer_map[parameter]
                            optimizer_hooked_count[i] += 1
                            if optimizer_hooked_count[i] == num_parameters_per_group[i]:
                                optimizers[i].step()
                                optimizers[i].zero_grad(set_to_none=True)

                        parameter.register_post_accumulate_grad_hook(grad_hook)
                        parameter_optimizer_map[parameter] = opt_idx
                        num_parameters_per_group[opt_idx] += 1

    # Training loop
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    accelerator.print("running training")
    accelerator.print(f"  num examples: {train_dataset_group.num_train_images}")
    accelerator.print(f"  num batches per epoch: {len(train_dataloader)}")
    accelerator.print(f"  num epochs: {num_train_epochs}")
    accelerator.print(
        f"  batch size per device: {', '.join([str(d.batch_size) for d in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps: {args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

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

    if is_swapping_blocks:
        accelerator.unwrap_model(dit).prepare_block_swap_before_forward()

    # For --sample_at_first
    optimizer_eval_fn()
    anima_train_utils.sample_images(
        accelerator, args, 0, global_step, dit, vae, vae_scale,
        qwen3_text_encoder, tokenize_strategy, text_encoding_strategy,
        sample_prompts_te_outputs,
    )
    optimizer_train_fn()
    if len(accelerator.trackers) > 0:
        accelerator.log({}, step=0)

    # Show model info
    unwrapped_dit = accelerator.unwrap_model(dit) if dit is not None else None
    if unwrapped_dit is not None:
        logger.info(f"dit device: {unwrapped_dit.t_embedding_norm.weight.device}, dtype: {unwrapped_dit.t_embedding_norm.weight.dtype}")
    if qwen3_text_encoder is not None:
        logger.info(f"qwen3 device: {next(qwen3_text_encoder.parameters()).device}")
    if vae is not None:
        logger.info(f"vae device: {next(vae.parameters()).device}")

    loss_recorder = train_util.LossRecorder()
    epoch = 0
    for epoch in range(num_train_epochs):
        accelerator.print(f"\nepoch {epoch+1}/{num_train_epochs}")
        current_epoch.value = epoch + 1

        for m in training_models:
            m.train()

        for step, batch in enumerate(train_dataloader):
            current_step.value = global_step

            if args.blockwise_fused_optimizers:
                optimizer_hooked_count = {i: 0 for i in range(len(optimizers))}  # reset counter for each step

            with accelerator.accumulate(*training_models):
                # Get latents
                if "latents" in batch and batch["latents"] is not None:
                    latents = batch["latents"].to(accelerator.device, dtype=weight_dtype)
                else:
                    with torch.no_grad():
                        # images are already [-1, 1] from IMAGE_TRANSFORMS, add temporal dim
                        images = batch["images"].to(accelerator.device, dtype=weight_dtype)
                        images = images.unsqueeze(2)  # (B, C, 1, H, W)
                        latents = vae.encode(images, vae_scale).to(accelerator.device, dtype=weight_dtype)

                    if torch.any(torch.isnan(latents)):
                        accelerator.print("NaN found in latents, replacing with zeros")
                        latents = torch.nan_to_num(latents, 0, out=latents)

                # Get text encoder outputs
                text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
                if text_encoder_outputs_list is not None:
                    # Cached outputs
                    text_encoder_outputs_list = text_encoding_strategy.drop_cached_text_encoder_outputs(
                        *text_encoder_outputs_list
                    )
                    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_outputs_list
                else:
                    # Encode on-the-fly
                    input_ids_list = batch["input_ids_list"]
                    qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = input_ids_list
                    with torch.no_grad():
                        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoding_strategy.encode_tokens(
                            tokenize_strategy,
                            [qwen3_text_encoder],
                            [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask],
                        )

                # Move to device
                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                attn_mask = attn_mask.to(accelerator.device)
                t5_input_ids = t5_input_ids.to(accelerator.device, dtype=torch.long)
                t5_attn_mask = t5_attn_mask.to(accelerator.device)

                # Noise and timesteps
                noise = torch.randn_like(latents)

                noisy_model_input, timesteps, sigmas = anima_train_utils.get_noisy_model_input_and_timesteps(
                    args, latents, noise, accelerator.device, weight_dtype
                )

                # NaN checks
                if torch.any(torch.isnan(noisy_model_input)):
                    accelerator.print("NaN found in noisy_model_input, replacing with zeros")
                    noisy_model_input = torch.nan_to_num(noisy_model_input, 0, out=noisy_model_input)

                # Create padding mask
                # padding_mask: (B, 1, H_latent, W_latent)
                bs = latents.shape[0]
                h_latent = latents.shape[-2]
                w_latent = latents.shape[-1]
                padding_mask = torch.zeros(
                    bs, 1, h_latent, w_latent,
                    dtype=weight_dtype, device=accelerator.device
                )

                # DiT forward (LLM adapter runs inside forward for DDP gradient sync)
                if is_swapping_blocks:
                    accelerator.unwrap_model(dit).prepare_block_swap_before_forward()

                with accelerator.autocast():
                    model_pred = dit(
                        noisy_model_input,
                        timesteps,
                        prompt_embeds,
                        padding_mask=padding_mask,
                        source_attention_mask=attn_mask,
                        t5_input_ids=t5_input_ids,
                        t5_attn_mask=t5_attn_mask,
                    )

                # Compute loss (rectified flow: target = noise - latents)
                target = noise - latents

                # Weighting
                weighting = anima_train_utils.compute_loss_weighting_for_anima(
                    weighting_scheme=args.weighting_scheme, sigmas=sigmas
                )

                # Loss
                huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, None)
                loss = train_util.conditional_loss(
                    model_pred.float(), target.float(), args.loss_type, "none", huber_c
                )
                if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
                    loss = apply_masked_loss(loss, batch)
                loss = loss.mean([1, 2, 3, 4])  # (B, C, T, H, W) -> (B,)

                if weighting is not None:
                    loss = loss * weighting

                loss_weights = batch["loss_weights"]
                loss = loss * loss_weights
                loss = loss.mean()

                accelerator.backward(loss)

                if not (args.fused_backward_pass or args.blockwise_fused_optimizers):
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
                    if args.blockwise_fused_optimizers:
                        for i in range(1, len(optimizers)):
                            lr_schedulers[i].step()

            # Checks if the accelerator has performed an optimization step
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                optimizer_eval_fn()
                anima_train_utils.sample_images(
                    accelerator, args, None, global_step, dit, vae, vae_scale,
                    qwen3_text_encoder, tokenize_strategy, text_encoding_strategy,
                    sample_prompts_te_outputs,
                )

                # Save at specific steps
                if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
                            args,
                            False,
                            accelerator,
                            save_dtype,
                            epoch,
                            num_train_epochs,
                            global_step,
                            accelerator.unwrap_model(dit) if train_dit else None,
                        )
                optimizer_train_fn()

            current_loss = loss.detach().item()
            if len(accelerator.trackers) > 0:
                logs = {"loss": current_loss}
                train_util.append_lr_to_logs_with_names(
                    logs, lr_scheduler, args.optimizer_type,
                    ["base", "self_attn", "cross_attn", "mlp", "mod", "llm_adapter"] if train_dit else []
                )
                accelerator.log(logs, step=global_step)

            loss_recorder.add(epoch=epoch, step=step, loss=current_loss)
            avr_loss: float = loss_recorder.moving_average
            logs = {"avr_loss": avr_loss}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if len(accelerator.trackers) > 0:
            logs = {"loss/epoch": loss_recorder.moving_average}
            accelerator.log(logs, step=epoch + 1)

        accelerator.wait_for_everyone()

        optimizer_eval_fn()
        if args.save_every_n_epochs is not None:
            if accelerator.is_main_process:
                anima_train_utils.save_anima_model_on_epoch_end_or_stepwise(
                    args,
                    True,
                    accelerator,
                    save_dtype,
                    epoch,
                    num_train_epochs,
                    global_step,
                    accelerator.unwrap_model(dit) if train_dit else None,
                )

        anima_train_utils.sample_images(
            accelerator, args, epoch + 1, global_step, dit, vae, vae_scale,
            qwen3_text_encoder, tokenize_strategy, text_encoding_strategy,
            sample_prompts_te_outputs,
        )

    # End training
    is_main_process = accelerator.is_main_process
    dit = accelerator.unwrap_model(dit)

    accelerator.end_training()
    optimizer_eval_fn()

    if args.save_state or args.save_state_on_train_end:
        train_util.save_state_on_train_end(args, accelerator)

    del accelerator

    if is_main_process and train_dit:
        anima_train_utils.save_anima_model_on_train_end(
            args,
            save_dtype,
            epoch,
            global_step,
            dit,
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
    add_custom_train_arguments(parser)
    train_util.add_dit_training_arguments(parser)
    anima_train_utils.add_anima_training_arguments(parser)
    sai_model_spec.add_model_spec_arguments(parser)

    parser.add_argument(
        "--blockwise_fused_optimizers",
        action="store_true",
        help="enable blockwise optimizers for fused backward pass and optimizer step",
    )
    parser.add_argument(
        "--cpu_offload_checkpointing",
        action="store_true",
        help="offload gradient checkpointing to CPU (reduces VRAM at cost of speed)",
    )
    parser.add_argument(
        "--unsloth_offload_checkpointing",
        action="store_true",
        help="offload activations to CPU RAM using async non-blocking transfers (faster than --cpu_offload_checkpointing). "
        "Cannot be used with --cpu_offload_checkpointing or --blocks_to_swap.",
    )
    parser.add_argument(
        "--skip_latents_validity_check",
        action="store_true",
        help="[Deprecated] use 'skip_cache_check' instead",
    )

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    train(args)
