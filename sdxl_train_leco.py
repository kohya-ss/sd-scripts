import argparse
import importlib
import random

import torch
from accelerate.utils import set_seed
from diffusers import DDPMScheduler
from tqdm import tqdm

from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from library import custom_train_functions, sdxl_model_util, sdxl_train_util, strategy_sdxl, train_util
from library.custom_train_functions import apply_snr_weight, prepare_scheduler_for_custom_training
from library.leco_train_util import (
    PromptEmbedsCache,
    apply_noise_offset,
    batch_add_time_ids,
    build_network_kwargs,
    concat_embeddings_xl,
    diffusion_xl,
    encode_prompt_sdxl,
    get_add_time_ids,
    get_initial_latents,
    get_random_resolution,
    load_prompt_settings,
    predict_noise_xl,
    save_weights,
)
from library.utils import add_logging_arguments, setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    train_util.add_sd_models_arguments(parser)
    train_util.add_optimizer_arguments(parser)
    train_util.add_training_arguments(parser, support_dreambooth=False)
    custom_train_functions.add_custom_train_arguments(parser, support_weighted_captions=False)
    sdxl_train_util.add_sdxl_training_arguments(parser, support_text_encoder_caching=False)
    add_logging_arguments(parser)

    parser.add_argument(
        "--save_model_as",
        type=str,
        default="safetensors",
        choices=[None, "ckpt", "pt", "safetensors"],
        help="format to save the model (default is .safetensors) / モデル保存時の形式（デフォルトはsafetensors）",
    )
    parser.add_argument("--no_metadata", action="store_true", help="do not save metadata in output model / メタデータを保存しない")

    parser.add_argument("--prompts_file", type=str, required=True, help="LECO prompt toml / LECO用のprompt toml")
    parser.add_argument(
        "--max_denoising_steps",
        type=int,
        default=40,
        help="number of partial denoising steps per iteration / 各イテレーションで部分デノイズするステップ数",
    )
    parser.add_argument(
        "--leco_denoise_guidance_scale",
        type=float,
        default=3.0,
        help="guidance scale for the partial denoising pass / 部分デノイズ時のguidance scale",
    )

    parser.add_argument("--network_weights", type=str, default=None, help="pretrained weights for network")
    parser.add_argument("--network_module", type=str, default="networks.lora", help="network module to train")
    parser.add_argument("--network_dim", type=int, default=4, help="network rank / ネットワークのrank")
    parser.add_argument("--network_alpha", type=float, default=1.0, help="network alpha / ネットワークのalpha")
    parser.add_argument("--network_dropout", type=float, default=None, help="network dropout / ネットワークのdropout")
    parser.add_argument("--network_args", type=str, default=None, nargs="*", help="additional network arguments")
    parser.add_argument(
        "--network_train_text_encoder_only",
        action="store_true",
        help="unsupported for LECO; kept for compatibility / LECOでは未対応",
    )
    parser.add_argument(
        "--network_train_unet_only",
        action="store_true",
        help="LECO always trains U-Net LoRA only / LECOは常にU-Net LoRAのみを学習",
    )
    parser.add_argument("--training_comment", type=str, default=None, help="comment stored in metadata")
    parser.add_argument("--dim_from_weights", action="store_true", help="infer network dim from network_weights")
    parser.add_argument("--unet_lr", type=float, default=None, help="learning rate for U-Net / U-Netの学習率")

    # dummy arguments required by train_util.verify_training_args / deepspeed_utils (LECO does not use datasets or deepspeed)
    parser.add_argument("--cache_latents", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--cache_latents_to_disk", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--deepspeed", action="store_true", default=False, help=argparse.SUPPRESS)

    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    args = train_util.read_config_from_file(args, parser)
    train_util.verify_training_args(args)
    sdxl_train_util.verify_sdxl_training_args(args, support_text_encoder_caching=False)

    if args.output_dir is None:
        raise ValueError("--output_dir is required")
    if args.network_train_text_encoder_only:
        raise ValueError("LECO does not support text encoder LoRA training")

    if args.seed is None:
        args.seed = random.randint(0, 2**32 - 1)
    set_seed(args.seed)

    accelerator = train_util.prepare_accelerator(args)
    weight_dtype, save_dtype = train_util.prepare_dtype(args)

    prompt_settings = load_prompt_settings(args.prompts_file)
    logger.info(f"loaded {len(prompt_settings)} LECO prompt settings from {args.prompts_file}")

    _, text_encoder1, text_encoder2, vae, unet, _, _ = sdxl_train_util.load_target_model(
        args, accelerator, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype
    )
    del vae
    text_encoders = [text_encoder1, text_encoder2]

    train_util.replace_unet_modules(unet, args.mem_eff_attn, args.xformers, args.sdpa)
    unet.requires_grad_(False)
    unet.to(accelerator.device, dtype=weight_dtype)
    unet.train()

    tokenize_strategy = strategy_sdxl.SdxlTokenizeStrategy(args.max_token_length, args.tokenizer_cache_dir)
    text_encoding_strategy = strategy_sdxl.SdxlTextEncodingStrategy()

    for text_encoder in text_encoders:
        text_encoder.to(accelerator.device, dtype=weight_dtype)
        text_encoder.requires_grad_(False)
        text_encoder.eval()

    prompt_cache = PromptEmbedsCache()
    unique_prompts = sorted(
        {
            prompt
            for setting in prompt_settings
            for prompt in (setting.target, setting.positive, setting.unconditional, setting.neutral)
        }
    )
    with torch.no_grad():
        for prompt in unique_prompts:
            prompt_cache[prompt] = encode_prompt_sdxl(tokenize_strategy, text_encoding_strategy, text_encoders, prompt)

    for text_encoder in text_encoders:
        text_encoder.to("cpu", dtype=torch.float32)
    clean_memory_on_device(accelerator.device)

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        clip_sample=False,
    )
    prepare_scheduler_for_custom_training(noise_scheduler, accelerator.device)
    if args.zero_terminal_snr:
        custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)

    network_module = importlib.import_module(args.network_module)
    net_kwargs = build_network_kwargs(args)
    if args.dim_from_weights:
        if args.network_weights is None:
            raise ValueError("--dim_from_weights requires --network_weights")
        network, _ = network_module.create_network_from_weights(1.0, args.network_weights, None, text_encoders, unet, **net_kwargs)
    else:
        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            None,
            text_encoders,
            unet,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )

    network.apply_to(text_encoders, unet, apply_text_encoder=False, apply_unet=True)
    network.set_multiplier(0.0)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        logger.info(f"loaded network weights from {args.network_weights}: {info}")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()

    unet_lr = args.unet_lr if args.unet_lr is not None else args.learning_rate
    trainable_params, _ = network.prepare_optimizer_params(None, unet_lr, args.learning_rate)
    _, _, optimizer = train_util.get_optimizer(args, trainable_params)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    network, optimizer, lr_scheduler = accelerator.prepare(network, optimizer, lr_scheduler)
    accelerator.unwrap_model(network).prepare_grad_etc(text_encoders, unet)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)

    optimizer_train_fn, _ = train_util.get_optimizer_train_eval_fn(optimizer, args)
    optimizer_train_fn()
    train_util.init_trackers(accelerator, args, "sdxl_leco_train")

    progress_bar = tqdm(total=args.max_train_steps, disable=not accelerator.is_local_main_process, desc="steps")
    global_step = 0

    while global_step < args.max_train_steps:
        with accelerator.accumulate(network):
            optimizer.zero_grad(set_to_none=True)

            setting = prompt_settings[torch.randint(0, len(prompt_settings), (1,)).item()]
            noise_scheduler.set_timesteps(args.max_denoising_steps, device=accelerator.device)

            timesteps_to = torch.randint(1, args.max_denoising_steps, (1,), device=accelerator.device).item()
            height, width = get_random_resolution(setting)

            latents = get_initial_latents(noise_scheduler, setting.batch_size, height, width, 1).to(
                accelerator.device, dtype=weight_dtype
            )
            latents = apply_noise_offset(latents, args.noise_offset)
            add_time_ids = get_add_time_ids(
                height,
                width,
                dynamic_crops=setting.dynamic_crops,
                dtype=weight_dtype,
                device=accelerator.device,
            )
            batched_time_ids = batch_add_time_ids(add_time_ids, setting.batch_size)

            network_multiplier = accelerator.unwrap_model(network)
            network_multiplier.set_multiplier(setting.multiplier)
            with accelerator.autocast():
                denoised_latents = diffusion_xl(
                    unet,
                    noise_scheduler,
                    latents,
                    concat_embeddings_xl(prompt_cache[setting.unconditional], prompt_cache[setting.target], setting.batch_size),
                    add_time_ids=batched_time_ids,
                    total_timesteps=timesteps_to,
                    guidance_scale=args.leco_denoise_guidance_scale,
                )

            noise_scheduler.set_timesteps(1000, device=accelerator.device)
            current_timestep_index = int(timesteps_to * 1000 / args.max_denoising_steps)
            current_timestep = noise_scheduler.timesteps[current_timestep_index]

            network_multiplier.set_multiplier(0.0)
            with torch.no_grad(), accelerator.autocast():
                positive_latents = predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    concat_embeddings_xl(prompt_cache[setting.unconditional], prompt_cache[setting.positive], setting.batch_size),
                    add_time_ids=batched_time_ids,
                    guidance_scale=1.0,
                )
                neutral_latents = predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    concat_embeddings_xl(prompt_cache[setting.unconditional], prompt_cache[setting.neutral], setting.batch_size),
                    add_time_ids=batched_time_ids,
                    guidance_scale=1.0,
                )
                unconditional_latents = predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    concat_embeddings_xl(prompt_cache[setting.unconditional], prompt_cache[setting.unconditional], setting.batch_size),
                    add_time_ids=batched_time_ids,
                    guidance_scale=1.0,
                )

            network_multiplier.set_multiplier(setting.multiplier)
            with accelerator.autocast():
                target_latents = predict_noise_xl(
                    unet,
                    noise_scheduler,
                    current_timestep,
                    denoised_latents,
                    concat_embeddings_xl(prompt_cache[setting.unconditional], prompt_cache[setting.target], setting.batch_size),
                    add_time_ids=batched_time_ids,
                    guidance_scale=1.0,
                )

                target = setting.build_target(positive_latents, neutral_latents, unconditional_latents)
                loss = torch.nn.functional.mse_loss(target_latents.float(), target.float(), reduction="none")
                loss = loss.mean(dim=(1, 2, 3))
                if args.min_snr_gamma is not None and args.min_snr_gamma > 0:
                    timesteps = torch.full((loss.shape[0],), current_timestep_index, device=loss.device, dtype=torch.long)
                    loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
                loss = loss.mean() * setting.weight

            accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm != 0.0:
                accelerator.clip_grad_norm_(network.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()

        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
            network_multiplier = accelerator.unwrap_model(network)
            network_multiplier.set_multiplier(0.0)

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "guidance_scale": setting.guidance_scale,
                "network_multiplier": setting.multiplier,
            }
            accelerator.log(logs, step=global_step)
            progress_bar.set_postfix(loss=f"{logs['loss']:.4f}")

            if args.save_every_n_steps and global_step % args.save_every_n_steps == 0 and global_step < args.max_train_steps:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    sdxl_extra = {"ss_base_model_version": sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0}
                    save_weights(accelerator, network, args, save_dtype, prompt_settings, global_step, last=False, extra_metadata=sdxl_extra)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        sdxl_extra = {"ss_base_model_version": sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0}
        save_weights(accelerator, network, args, save_dtype, prompt_settings, global_step, last=True, extra_metadata=sdxl_extra)

    accelerator.end_training()


if __name__ == "__main__":
    main()
