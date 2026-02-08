# Anima Strategy Classes

import os
import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from library import anima_utils, train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class AnimaTokenizeStrategy(TokenizeStrategy):
    """Tokenize strategy for Anima: dual tokenization with Qwen3 + T5.

    Qwen3 tokens are used for the text encoder.
    T5 tokens are used as target input IDs for the LLM Adapter (NOT encoded by T5).

    Can be initialized with either pre-loaded tokenizer objects or paths to load from.
    """

    def __init__(
        self,
        qwen3_tokenizer=None,
        t5_tokenizer=None,
        qwen3_max_length: int = 512,
        t5_max_length: int = 512,
        qwen3_path: Optional[str] = None,
        t5_tokenizer_path: Optional[str] = None,
    ) -> None:
        # Load tokenizers from paths if not provided directly
        if qwen3_tokenizer is None:
            if qwen3_path is None:
                raise ValueError("Either qwen3_tokenizer or qwen3_path must be provided")
            qwen3_tokenizer = anima_utils.load_qwen3_tokenizer(qwen3_path)
        if t5_tokenizer is None:
            t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

        self.qwen3_tokenizer = qwen3_tokenizer
        self.qwen3_max_length = qwen3_max_length
        self.t5_tokenizer = t5_tokenizer
        self.t5_max_length = t5_max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        # Tokenize with Qwen3
        qwen3_encoding = self.qwen3_tokenizer.batch_encode_plus(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.qwen3_max_length,
        )
        qwen3_input_ids = qwen3_encoding["input_ids"]
        qwen3_attn_mask = qwen3_encoding["attention_mask"]

        # Tokenize with T5 (for LLM Adapter target tokens)
        t5_encoding = self.t5_tokenizer.batch_encode_plus(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.t5_max_length,
        )
        t5_input_ids = t5_encoding["input_ids"]
        t5_attn_mask = t5_encoding["attention_mask"]

        return [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncodingStrategy(TextEncodingStrategy):
    """Text encoding strategy for Anima.

    Encodes Qwen3 tokens through the Qwen3 text encoder to get hidden states.
    T5 tokens are passed through unchanged (only used by LLM Adapter).
    """

    def __init__(
        self,
        dropout_rate: float = 0.0,
    ) -> None:
        self.dropout_rate = dropout_rate
        # Cached unconditional embeddings (from encoding empty caption "")
        # Must be initialized via cache_uncond_embeddings() before text encoder is deleted
        self._uncond_prompt_embeds: Optional[torch.Tensor] = None  # (1, seq_len, hidden)
        self._uncond_attn_mask: Optional[torch.Tensor] = None      # (1, seq_len)
        self._uncond_t5_input_ids: Optional[torch.Tensor] = None   # (1, t5_seq_len)
        self._uncond_t5_attn_mask: Optional[torch.Tensor] = None   # (1, t5_seq_len)

    def cache_uncond_embeddings(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
    ) -> None:
        """Pre-encode empty caption "" and cache the unconditional embeddings.

        Must be called before the text encoder is deleted from GPU.
        This matches diffusion-pipe-main behavior where empty caption embeddings
        are pre-cached and swapped in during caption dropout.
        """
        logger.info("Caching unconditional embeddings for caption dropout (encoding empty caption)...")
        tokens = tokenize_strategy.tokenize("")
        with torch.no_grad():
            uncond_outputs = self.encode_tokens(tokenize_strategy, models, tokens, enable_dropout=False)
        # Store as CPU tensors (1, seq_len, ...) to avoid GPU memory waste
        self._uncond_prompt_embeds = uncond_outputs[0].cpu()
        self._uncond_attn_mask = uncond_outputs[1].cpu()
        self._uncond_t5_input_ids = uncond_outputs[2].cpu()
        self._uncond_t5_attn_mask = uncond_outputs[3].cpu()
        logger.info("  Unconditional embeddings cached successfully")

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        enable_dropout: bool = True,
    ) -> List[torch.Tensor]:
        """Encode Qwen3 tokens and return embeddings + T5 token IDs.

        Args:
            models: [qwen3_text_encoder]
            tokens: [qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask]

        Returns:
            [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]
        """

        qwen3_text_encoder = models[0]
        qwen3_input_ids, qwen3_attn_mask, t5_input_ids, t5_attn_mask = tokens

        # Handle dropout: replace dropped items with unconditional embeddings (matching diffusion-pipe-main)
        batch_size = qwen3_input_ids.shape[0]

        encoder_device = qwen3_text_encoder.device

        # Build drop mask
        if enable_dropout and self.dropout_rate > 0.0:
            drop_mask = torch.rand(batch_size) < self.dropout_rate
        else:
            drop_mask = torch.zeros(batch_size, dtype=torch.bool)
        keep_mask = ~drop_mask

        # Encode only kept items
        if keep_mask.any():
            nd_input_ids = qwen3_input_ids[keep_mask].to(encoder_device)
            nd_attn_mask = qwen3_attn_mask[keep_mask].to(encoder_device)
            outputs = qwen3_text_encoder(input_ids=nd_input_ids, attention_mask=nd_attn_mask)
            nd_encoded_text = outputs.last_hidden_state
            # Zero out padding positions
            nd_encoded_text[~nd_attn_mask.bool()] = 0
        else:
            nd_encoded_text = None

        # If no items are dropped, return directly
        if not drop_mask.any():
            prompt_embeds = nd_encoded_text
            attn_mask = qwen3_attn_mask.to(encoder_device)
            return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]

        # Get unconditional embeddings
        if self._uncond_prompt_embeds is not None:
            uncond_pe = self._uncond_prompt_embeds[0]
            uncond_am = self._uncond_attn_mask[0]
            uncond_t5_ids = self._uncond_t5_input_ids[0]
            uncond_t5_am = self._uncond_t5_attn_mask[0]
        else:
            # Recursively encode empty caption with no dropout
            uncond_tokens = tokenize_strategy.tokenize("")
            with torch.no_grad():
                uncond_pe, uncond_am, uncond_t5_ids, uncond_t5_am = self.encode_tokens(
                    tokenize_strategy, models, uncond_tokens, enable_dropout=False
                )

        seq_len = qwen3_input_ids.shape[1]
        hidden_size = (nd_encoded_text.shape[-1] if nd_encoded_text is not None else uncond_pe.shape[-1])
        dtype = (nd_encoded_text.dtype if nd_encoded_text is not None else uncond_pe.dtype)

        prompt_embeds = torch.zeros((batch_size, seq_len, hidden_size), device=encoder_device, dtype=dtype)
        attn_mask = torch.zeros((batch_size, seq_len), device=encoder_device, dtype=qwen3_attn_mask.dtype)

        if keep_mask.any():
            prompt_embeds[keep_mask] = nd_encoded_text
            attn_mask[keep_mask] = nd_attn_mask

        # Fill dropped items
        prompt_embeds[drop_mask] = uncond_pe.to(device=encoder_device, dtype=dtype)
        attn_mask[drop_mask] = uncond_am.to(device=encoder_device, dtype=qwen3_attn_mask.dtype)

        t5_input_ids = t5_input_ids.clone()
        t5_attn_mask = t5_attn_mask.clone()
        t5_input_ids[drop_mask] = uncond_t5_ids.to(device=t5_input_ids.device, dtype=t5_input_ids.dtype)
        t5_attn_mask[drop_mask] = uncond_t5_am.to(device=t5_attn_mask.device, dtype=t5_attn_mask.dtype)

        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]


    def drop_cached_text_encoder_outputs(
        self,
        prompt_embeds: torch.Tensor,
        attn_mask: torch.Tensor,
        t5_input_ids: torch.Tensor,
        t5_attn_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """Apply dropout to cached text encoder outputs.

        Called during training when using cached outputs.
        Replaces dropped items with pre-cached unconditional embeddings (from encoding "")
        to match diffusion-pipe-main behavior.
        """
        if prompt_embeds is not None and self.dropout_rate > 0.0:
            # Clone to avoid in-place modification of cached tensors
            prompt_embeds = prompt_embeds.clone()
            if attn_mask is not None:
                attn_mask = attn_mask.clone()
            if t5_input_ids is not None:
                t5_input_ids = t5_input_ids.clone()
            if t5_attn_mask is not None:
                t5_attn_mask = t5_attn_mask.clone()

            for i in range(prompt_embeds.shape[0]):
                if random.random() < self.dropout_rate:
                    if self._uncond_prompt_embeds is not None:
                        # Use pre-cached unconditional embeddings
                        prompt_embeds[i] = self._uncond_prompt_embeds[0].to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                        if attn_mask is not None:
                            attn_mask[i] = self._uncond_attn_mask[0].to(device=attn_mask.device, dtype=attn_mask.dtype)
                        if t5_input_ids is not None:
                            t5_input_ids[i] = self._uncond_t5_input_ids[0].to(device=t5_input_ids.device, dtype=t5_input_ids.dtype)
                        if t5_attn_mask is not None:
                            t5_attn_mask[i] = self._uncond_t5_attn_mask[0].to(device=t5_attn_mask.device, dtype=t5_attn_mask.dtype)
                    else:
                        # Fallback: zero out (should not happen if cache_uncond_embeddings was called)
                        logger.warning("Unconditional embeddings not cached, falling back to zeros for caption dropout")
                        prompt_embeds[i] = torch.zeros_like(prompt_embeds[i])
                        if attn_mask is not None:
                            attn_mask[i] = torch.zeros_like(attn_mask[i])
                        if t5_input_ids is not None:
                            t5_input_ids[i] = torch.zeros_like(t5_input_ids[i])
                        if t5_attn_mask is not None:
                            t5_attn_mask[i] = torch.zeros_like(t5_attn_mask[i])

        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]


class AnimaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    """Caching strategy for Anima text encoder outputs.

    Caches: prompt_embeds (float), attn_mask (int), t5_input_ids (int), t5_attn_mask (int)
    """

    ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_anima_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.ANIMA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "prompt_embeds" not in npz:
                return False
            if "attn_mask" not in npz:
                return False
            if "t5_input_ids" not in npz:
                return False
            if "t5_attn_mask" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        prompt_embeds = data["prompt_embeds"]
        attn_mask = data["attn_mask"]
        t5_input_ids = data["t5_input_ids"]
        t5_attn_mask = data["t5_attn_mask"]
        return [prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        anima_text_encoding_strategy: AnimaTextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            # Always disable dropout during caching
            prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = anima_text_encoding_strategy.encode_tokens(
                tokenize_strategy,
                models,
                tokens_and_masks,
                enable_dropout=False,
            )

        # Convert to numpy for caching
        if prompt_embeds.dtype == torch.bfloat16:
            prompt_embeds = prompt_embeds.float()
        prompt_embeds = prompt_embeds.cpu().numpy()
        attn_mask = attn_mask.cpu().numpy()
        t5_input_ids = t5_input_ids.cpu().numpy().astype(np.int32)
        t5_attn_mask = t5_attn_mask.cpu().numpy().astype(np.int32)

        for i, info in enumerate(infos):
            prompt_embeds_i = prompt_embeds[i]
            attn_mask_i = attn_mask[i]
            t5_input_ids_i = t5_input_ids[i]
            t5_attn_mask_i = t5_attn_mask[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    prompt_embeds=prompt_embeds_i,
                    attn_mask=attn_mask_i,
                    t5_input_ids=t5_input_ids_i,
                    t5_attn_mask=t5_attn_mask_i,
                )
            else:
                info.text_encoder_outputs = (prompt_embeds_i, attn_mask_i, t5_input_ids_i, t5_attn_mask_i)


class AnimaLatentsCachingStrategy(LatentsCachingStrategy):
    """Latent caching strategy for Anima using WanVAE.

    WanVAE produces 16-channel latents with spatial downscale 8x.
    Latent shape for images: (B, 16, 1, H/8, W/8)
    """

    ANIMA_LATENTS_NPZ_SUFFIX = "_anima.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return self.ANIMA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + self.ANIMA_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(
        self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool
    ):
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        """Cache batch of latents using WanVAE.

        vae is expected to be the WanVAE_ model (not the wrapper).
        The encoding function handles the mean/std normalization.
        """
        from library.anima_models import ANIMA_VAE_MEAN, ANIMA_VAE_STD

        vae_device = next(vae.parameters()).device
        vae_dtype = next(vae.parameters()).dtype

        # Create scale tensors on VAE device
        mean = torch.tensor(ANIMA_VAE_MEAN, dtype=vae_dtype, device=vae_device)
        std = torch.tensor(ANIMA_VAE_STD, dtype=vae_dtype, device=vae_device)
        scale = [mean, 1.0 / std]

        def encode_by_vae(img_tensor):
            """Encode image tensor to latents.

            img_tensor: (B, C, H, W) in [-1, 1] range (already normalized by IMAGE_TRANSFORMS)
            Need to add temporal dim to get (B, C, T=1, H, W) for WanVAE
            """
            # Add temporal dimension: (B, C, H, W) -> (B, C, 1, H, W)
            img_tensor = img_tensor.unsqueeze(2)
            img_tensor = img_tensor.to(vae_device, dtype=vae_dtype)

            latents = vae.encode(img_tensor, scale)
            return latents.to("cpu")

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae_device)
