import argparse
import copy
import math
import random
import os
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator
from diffusers import AutoencoderKL, CogView4Transformer2DModel, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.cogview4.pipeline_cogview4 import CogView4Pipeline
from PIL import Image
from transformers import AutoTokenizer, GlmModel

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    sd3_train_utils,
    strategy_base,
    strategy_cogview4,
    train_util,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class CogView4NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        
        # CogView4 specific argument validation
        if hasattr(args, 'fp8_base_unet') and args.fp8_base_unet:
            logger.warning("FP8 training is not yet fully supported for CogView4. Disabling fp8_base_unet.")
            args.fp8_base_unet = False
            
        if hasattr(args, 'cache_text_encoder_outputs') and args.cache_text_encoder_outputs:
            logger.warning("Text encoder output caching is not yet implemented for CogView4. Disabling.")
            args.cache_text_encoder_outputs = False
            
        if hasattr(args, 'cache_text_encoder_outputs_to_disk') and args.cache_text_encoder_outputs_to_disk:
            logger.warning("Text encoder output disk caching is not yet implemented for CogView4. Disabling.")
            args.cache_text_encoder_outputs_to_disk = False
            
        # Set default values for CogView4
        if not hasattr(args, 'max_token_length'):
            args.max_token_length = 128  # Default token length for GLM
            
        if not hasattr(args, 'resolution'):
            args.resolution = 256  # Default resolution for CogView4
            
        # Update dataset resolution if needed
        train_dataset_group.set_resolution(args.resolution)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)  # TODO check this

    def load_target_model(self, args, weight_dtype, accelerator):
        """
        Load the CogView4 model components including tokenizer, text encoder, VAE, and transformer.
        """
        logger.info(f"Loading CogView4 model from {args.pretrained_model_name_or_path}")
        
        # Load tokenizer and text encoder (GLM)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, 
            subfolder="tokenizer",
            use_fast=False,
            trust_remote_code=True
        )
        
        self.text_encoder = GlmModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            torch_dtype=weight_dtype,
            trust_remote_code=True
        )
        self.text_encoder.eval()
        
        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=weight_dtype,
            trust_remote_code=True
        )
        self.vae.eval()
        
        # Load transformer
        self.transformer = CogView4Transformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="transformer",
            torch_dtype=weight_dtype,
            trust_remote_code=True
        )
        
        # Create noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
            prediction_type="epsilon",
        )
        
        # Move models to device
        device = accelerator.device
        self.text_encoder = self.text_encoder.to(device)
        self.vae = self.vae.to(device)
        self.transformer = self.transformer.to(device)
        
        # Set gradient checkpointing if enabled
        if args.gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
            # Text encoder gradient checkpointing is handled by prepare_text_encoder_grad_ckpt_workaround
            # called by the base trainer if args.train_text_encoder is true for that TE.
        
        # Store components for later use
        self.weight_dtype = weight_dtype
        
        # Return components in the expected format
        return "cogview4-v1", [self.text_encoder], self.vae, self.transformer

    def get_tokenize_strategy(self, args):
        # For CogView4, we use a fixed token length for GLM
        max_token_length = getattr(args, 'max_token_length', 128)
        logger.info(f"Using max_token_length: {max_token_length} for GLM tokenizer")
        return strategy_cogview4.CogView4TokenizeStrategy(max_token_length, args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy):
        # For CogView4, we only have one tokenizer (GLM)
        return [tokenize_strategy.tokenizer]

    def get_latents_caching_strategy(self, args):
        return strategy_cogview4.CogView4LatentsCachingStrategy(
            args.cache_latents_to_disk, 
            args.vae_batch_size, 
            skip_disk_cache_validity_check=False
        )

    def get_text_encoding_strategy(self, args):
        # For CogView4, we use GLM instead of T5, but maintain similar interface
        return strategy_cogview4.CogView4TextEncodingStrategy(
            apply_attention_mask=getattr(args, 'apply_attention_mask', True)
        )

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        # For CogView4, we always return the text encoder (GLM) as it's needed for encoding
        return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        # For CogView4, we only have one text encoder (GLM)
        return [getattr(args, 'train_text_encoder', False)]

    def get_text_encoder_outputs_caching_strategy(self, args):
        if getattr(args, 'cache_text_encoder_outputs', False):
            return strategy_cogview4.CogView4TextEncoderOutputsCachingStrategy(
                cache_to_disk=getattr(args, 'cache_text_encoder_outputs_to_disk', False),
                batch_size=getattr(args, 'text_encoder_batch_size', 1),
                skip_disk_cache_validity_check=getattr(args, 'skip_cache_check', False),
                is_partial=getattr(args, 'train_text_encoder', False)
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        """Cache text encoder outputs to speed up training.
        
        Args:
            args: Training arguments
            accelerator: Accelerator instance
            unet: UNet model
            vae: VAE model
            text_encoders: List containing the GLM text encoder
            dataset: Dataset to cache text encoder outputs for
            weight_dtype: Data type for weights
        """
        if getattr(args, 'cache_text_encoder_outputs', False):
            if not getattr(args, 'lowram', False):
                # Free up GPU memory by moving models to CPU
                logger.info("Moving VAE and UNet to CPU to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            # Move text encoder to GPU with proper dtype
            logger.info("Moving text encoder to GPU")
            text_encoder = text_encoders[0]  # CogView4 uses a single text encoder (GLM)
            text_encoder.to(accelerator.device, dtype=weight_dtype)

            # Cache text encoder outputs
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            # Cache sample prompts if provided
            if getattr(args, 'sample_prompts', None) is not None:
                logger.info(f"Caching text encoder outputs for sample prompts: {args.sample_prompts}")

                # Initialize CogView4 strategies
                tokenize_strategy = strategy_cogview4.CogView4TokenizeStrategy(
                    max_length=getattr(args, 'max_token_length', 128),
                    tokenizer_cache_dir=getattr(args, 'tokenizer_cache_dir', None)
                )
                text_encoding_strategy = strategy_cogview4.CogView4TextEncodingStrategy(
                    apply_attention_mask=getattr(args, 'apply_attention_mask', True)
                )

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # key: prompt, value: text encoder outputs
                
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p and p not in sample_prompts_te_outputs:  # Skip empty prompts and duplicates
                                logger.info(f"Caching text encoder outputs for prompt: {p}")
                                tokens = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = text_encoding_strategy.encode_tokens(
                                    tokenize_strategy=tokenize_strategy,
                                    models=text_encoders,
                                    tokens=tokens,
                                    apply_attention_mask=getattr(args, 'apply_attention_mask', True)
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # Move text encoder back to CPU if not training it
            if not getattr(args, 'train_text_encoder', False):
                logger.info("Moving text encoder back to CPU")
                text_encoder.to("cpu")
                clean_memory_on_device(accelerator.device)

            # Move VAE and UNet back to their original devices if not in lowram mode
            if not getattr(args, 'lowram', False):
                logger.info("Moving VAE and UNet back to original devices")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # Keep text encoder in GPU if we're not caching outputs
            if text_encoders:
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)

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

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizer, text_encoder, transformer):
        """
        Generate sample images during training to monitor progress.
        """
        logger.info(f"Generating sample images at step {global_step}")
        
        # Set models to eval mode
        was_training = transformer.training
        transformer.eval()
        vae.eval()
        
        # Sample prompts to use for generation
        sample_prompts = [
            "A high quality photo of a cat",
            "A beautiful landscape with mountains and a lake",
            "A futuristic city at night"
        ]
        
        # Generate images for each prompt
        all_images = []
        
        with torch.no_grad():
            for prompt in sample_prompts:
                # Tokenize the prompt
                text_input = self.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_input.input_ids.to(device)
                attention_mask = text_input.attention_mask.to(device)
                
                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    ).last_hidden_state
                
                # Sample random noise
                latents = torch.randn(
                    (1, 4, args.resolution // 8, args.resolution // 8),
                    device=device,
                    dtype=torch.float32
                )
                
                # Set the scheduler for inference
                self.noise_scheduler.set_timesteps(50, device=device)
                
                # Generate image using the denoising process
                for t in self.noise_scheduler.timesteps:
                    # Expand the latents if we are doing classifier-free guidance
                    latent_model_input = torch.cat([latents] * 2) if args.guidance_scale > 1.0 else latents
                    latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # Predict the noise residual
                    with torch.no_grad():
                        noise_pred = transformer(
                            latent_model_input,
                            t.unsqueeze(0).repeat(latent_model_input.shape[0]),
                            encoder_hidden_states=torch.cat([text_embeddings] * 2) if args.guidance_scale > 1.0 else text_embeddings,
                            attention_mask=attention_mask,
                            return_dict=True,
                        ).sample
                    
                    # Perform guidance
                    if args.guidance_scale > 1.0:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Compute the previous noisy sample x_t -> x_t-1
                    latents = self.noise_scheduler.step(noise_pred, t, latents).prev_sample
                
                # Scale and decode the image latents with vae
                latents = 1 / 0.18215 * latents
                with torch.no_grad():
                    image = vae.decode(latents).sample
                
                # Convert to PIL image
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                images = (image * 255).round().astype("uint8")
                pil_images = [Image.fromarray(image) for image in images]
                all_images.extend(pil_images)
        
        # Log images to tensorboard if available
        if accelerator.is_main_process and hasattr(accelerator, "log"):
            log_images = []
            for i, img in enumerate(all_images):
                # Convert PIL image to numpy for logging
                log_images.append(np.array(img))
                
                # Save individual images
                os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
                img.save(os.path.join(args.output_dir, "samples", f"sample_epoch{epoch}_step{global_step}_{i}.png"))
            
            # Log to tensorboard
            accelerator.log({
                "samples": [
                    wandb.Image(img, caption=f"{sample_prompts[i]}")
                    for i, img in enumerate(log_images)
                ]
            }, step=global_step)
        
        # Set models back to training mode if they were training before
        if was_training:
            transformer.train()
            vae.train()
            
        return all_images

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        # Return the noise scheduler that was created during model loading
        return self.noise_scheduler

    def encode_images_to_latents(self, args, vae, images):
        return vae.encode(images)

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
        transformer: CogView4Transformer2DModel,
        network,
        weight_dtype,
        train_unet=True,
        is_train=True,
    ):
        """
        Get noise prediction and target for the loss computation.
        """
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
        ).long()
        
        # Add noise to the latents according to the noise magnitude at each timestep
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        
        # Prepare text encoder outputs
        with torch.set_grad_enabled(self.train_text_encoder):
            text_embeddings = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            ).last_hidden_state
        
        # Predict the noise residual
        with torch.set_grad_enabled(train_unet):
            # Add conditioning dropout for classifier-free guidance
            if args.guidance_scale > 1.0:
                # Randomly drop text conditioning 5% of the time
                mask = (torch.rand(bsz, device=latents.device) < 0.05).float().unsqueeze(1).unsqueeze(1)
                text_embeddings = text_embeddings * (1 - mask) + torch.zeros_like(text_embeddings) * mask
            
            # Predict noise
            model_pred = transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
                attention_mask=attention_mask,
                return_dict=True,
            ).sample
        
        # Calculate target
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
        
        # For classifier-free guidance, we need to do two forward passes
        if args.guidance_scale > 1.0:
            # Predict the conditional and unconditional outputs
            model_pred_uncond = transformer(
                noisy_latents,
                timesteps,
                encoder_hidden_states=torch.zeros_like(text_embeddings),
                attention_mask=attention_mask,
                return_dict=True,
            ).sample
            
            # Perform classifier-free guidance
            model_pred = model_pred_uncond + args.guidance_scale * (model_pred - model_pred_uncond)
            
            # For training, we only compute the loss on the conditional prediction
            if is_train:
                model_pred = model_pred_uncond + args.guidance_scale * (model_pred - model_pred_uncond)
        
        # Simple weighting - can be adjusted based on timestep if needed
        weighting = torch.ones_like(timesteps, dtype=weight_dtype, device=latents.device)
        
        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        """
        Post-process the loss value.
        This can include applying timestep weighting, gradient clipping, etc.
        """
        # Apply timestep weighting if specified
        if hasattr(args, 'timestep_bias_portion') and args.timestep_bias_portion > 0.0:
            # Simple timestep weighting - can be made more sophisticated if needed
            weights = torch.ones_like(timesteps, dtype=torch.float32)
            if hasattr(args, 'timestep_bias_begin') and args.timestep_bias_begin > 0:
                mask = timesteps < args.timestep_bias_begin
                weights[mask] = 0.0
            if hasattr(args, 'timestep_bias_end') and args.timestep_bias_end < 1000:
                mask = timesteps > args.timestep_bias_end
                weights[mask] = 0.0
            if hasattr(args, 'timestep_bias_multiplier') and args.timestep_bias_multiplier != 1.0:
                weights = weights * args.timestep_bias_multiplier
            
            loss = loss * weights.to(loss.device)

        # Clip loss values if specified
        if hasattr(args, 'clip_grad_norm') and args.clip_grad_norm > 0.0:
            loss = torch.clamp(loss, -args.clip_grad_norm, args.clip_grad_norm)

        return loss

    def prepare_extra_step_kwargs(self, generator, eta):
        """
        Prepare extra kwargs for the scheduler step, such as the generator for reproducibility.
        """
        # Prepare extra step kwargs.
        # TODO: Logic should ideally just be moved to base class
        accepts_eta = "eta" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Check if the scheduler accepts a generator
        accepts_generator = "generator" in set(inspect.signature(self.noise_scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
            
        return extra_step_kwargs

    def update_metadata(self, metadata, args):
        metadata["ss_apply_attn_mask"] = args.apply_attn_mask
        metadata["ss_weighting_scheme"] = args.weighting_scheme
        metadata["ss_logit_mean"] = args.logit_mean
        metadata["ss_logit_std"] = args.logit_std
        metadata["ss_mode_scale"] = args.mode_scale
        metadata["ss_guidance_scale"] = args.guidance_scale
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        metadata["ss_model_prediction_type"] = args.model_prediction_type
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift

    def is_text_encoder_not_needed_for_training(self, args):
        """Check if text encoder outputs are cached and not being trained."""
        return getattr(args, 'cache_text_encoder_outputs', False) and not getattr(args, 'train_text_encoder', False)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        """Prepare text encoder for gradient checkpointing.
        
        For CogView4, we only have one text encoder (GLM) so we don't need index-based handling.
        The base class method handles enabling gradient checkpointing if args specify it.
        """
        return super().prepare_text_encoder_grad_ckpt_workaround(index, text_encoder)

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        """Prepare text encoder for FP8 training.
        
        Args:
            index: Text encoder index (always 0 for CogView4)
            text_encoder: The text encoder model (GLM)
            te_weight_dtype: Target weight dtype for the encoder
            weight_dtype: Base weight dtype for embeddings
        """
        if index != 0:
            logger.warning(f"Unexpected text encoder index {index} for CogView4, expecting 0.")
            # Still proceed, assuming it's the single GLM encoder

        logger.info(f"Preparing GLM text encoder (index {index}) for {te_weight_dtype}, embeddings to {weight_dtype}")
        text_encoder.to(te_weight_dtype)
        
        # Move embeddings to base weight dtype if they exist
        if hasattr(text_encoder, 'word_embeddings'): # GLM typically has word_embeddings
            text_encoder.word_embeddings.to(dtype=weight_dtype)
        if hasattr(text_encoder, 'position_embeddings'): # GLM might have position_embeddings
            text_encoder.position_embeddings.to(dtype=weight_dtype)
        # Add other relevant parts of GLM if they need specific dtype handling for FP8
        return text_encoder

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        """Called at the end of each validation step."""
        # No special handling needed for CogView4 (e.g., no block swapping)
        pass

    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        """Prepare UNet model (CogView4Transformer2DModel) with accelerator.
        
        For CogView4, we use standard model preparation.
        """
        return super().prepare_unet_with_accelerator(args, accelerator, unet)


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    return parser

if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = CogView4NetworkTrainer()
    trainer.train(args)
