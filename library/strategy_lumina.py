import glob
import os
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModel
from library import train_util
from library.strategy_base import (
    LatentsCachingStrategy,
    TokenizeStrategy,
    TextEncodingStrategy,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


GEMMA_ID = "google/gemma-2-2b"


class LuminaTokenizeStrategy(TokenizeStrategy):
    def __init__(
        self, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            GEMMA_ID, cache_dir=tokenizer_cache_dir
        )
        self.tokenizer.padding_side = "right"

        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        encodings = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
        )
        return [encodings.input_ids]

    def tokenize_with_weights(
        self, text: str | List[str]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # Gemma doesn't support weighted prompts, return uniform weights
        tokens = self.tokenize(text)
        weights = [torch.ones_like(t) for t in tokens]
        return tokens, weights


class LuminaTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, apply_gemma2_attn_mask: Optional[bool] = None) -> None:
        super().__init__()
        self.apply_gemma2_attn_mask = apply_gemma2_attn_mask

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        apply_gemma2_attn_mask: Optional[bool] = None,
    ) -> List[torch.Tensor]:

        if apply_gemma2_attn_mask is None:
            apply_gemma2_attn_mask = self.apply_gemma2_attn_mask

        text_encoder = models[0]
        input_ids = tokens[0].to(text_encoder.device)

        attention_mask = None
        position_ids = None
        if apply_gemma2_attn_mask:
            # Create attention mask (1 for non-padding, 0 for padding)
            attention_mask = (input_ids != tokenize_strategy.tokenizer.pad_token_id).to(
                text_encoder.device
            )

            # Create position IDs
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

        with torch.no_grad():
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            # Get the last hidden state
            hidden_states = outputs.last_hidden_state

        return [hidden_states]

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        # For simplicity, use uniform weighting
        return self.encode_tokens(tokenize_strategy, models, tokens_list)


class LuminaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    LUMINA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_lumina_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        apply_gemma2_attn_mask: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            is_partial,
        )
        self.apply_gemma2_attn_mask = apply_gemma2_attn_mask

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return (
            os.path.splitext(image_abs_path)[0]
            + LuminaTextEncoderOutputsCachingStrategy.LUMINA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX
        )

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hidden_state" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hidden_state = data["hidden_state"]
        return [hidden_state]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        infos: List,
    ):
        lumina_text_encoding_strategy: LuminaTextEncodingStrategy = (
            text_encoding_strategy
        )
        captions = [info.caption for info in infos]

        if self.is_weighted:
            tokens_list, weights_list = tokenize_strategy.tokenize_with_weights(
                captions
            )
            with torch.no_grad():
                hidden_state = lumina_text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy, models, tokens_list, weights_list
                )[0]
        else:
            tokens = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                hidden_state = lumina_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens
                )[0]

        if hidden_state.dtype == torch.bfloat16:
            hidden_state = hidden_state.float()

        hidden_state = hidden_state.cpu().numpy()

        for i, info in enumerate(infos):
            hidden_state_i = hidden_state[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state=hidden_state_i,
                )
            else:
                info.text_encoder_outputs = [hidden_state_i]


class LuminaLatentsCachingStrategy(LatentsCachingStrategy):
    LUMINA_LATENTS_NPZ_SUFFIX = "_lumina.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return LuminaLatentsCachingStrategy.LUMINA_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(
        self, absolute_path: str, image_size: Tuple[int, int]
    ) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + LuminaLatentsCachingStrategy.LUMINA_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        npz_path: str,
        flip_aug: bool,
        alpha_mask: bool,
    ):
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[List[int]],
        Optional[List[int]],
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        return self._default_load_latents_from_disk(
            8, npz_path, bucket_reso
        )  # support multi-resolution

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(
        self,
        vae,
        image_infos: List,
        flip_aug: bool,
        alpha_mask: bool,
        random_crop: bool,
    ):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae,
            vae_device,
            vae_dtype,
            image_infos,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True,
        )

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
