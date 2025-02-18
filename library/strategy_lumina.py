import glob
import os
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModel, GemmaTokenizerFast
from library import train_util
from library.strategy_base import (
    LatentsCachingStrategy,
    TokenizeStrategy,
    TextEncodingStrategy,
    TextEncoderOutputsCachingStrategy
)
import numpy as np
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


GEMMA_ID = "google/gemma-2-2b"


class LuminaTokenizeStrategy(TokenizeStrategy):
    def __init__(
        self, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None
    ) -> None:
        self.tokenizer: GemmaTokenizerFast = AutoTokenizer.from_pretrained(
            GEMMA_ID, cache_dir=tokenizer_cache_dir
        )
        self.tokenizer.padding_side = "right"

        if max_length is None:
            self.max_length = 256
        else:
            self.max_length = max_length

    def tokenize(self, text: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=8,
            truncation=True,
        )
        return [encodings.input_ids, encodings.attention_mask]

    def tokenize_with_weights(
        self, text: str | List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Gemma doesn't support weighted prompts, return uniform weights
        tokens, attention_masks = self.tokenize(text)
        weights = [torch.ones_like(t) for t in tokens]
        return tokens, attention_masks, weights


class LuminaTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        super().__init__()

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_encoder = models[0]
        input_ids, attention_masks = tokens

        outputs = text_encoder(
            input_ids=input_ids.to(text_encoder.device),
            attention_mask=attention_masks.to(text_encoder.device),
            output_hidden_states=True,
            return_dict=True,
        )

        return outputs.hidden_states[-2], input_ids, attention_masks

    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # For simplicity, use uniform weighting
        return self.encode_tokens(tokenize_strategy, models, tokens)


class LuminaTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    LUMINA_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_lumina_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            is_partial,
        )

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
            if "attention_mask" not in npz:
                return False
            if "input_ids" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        """
        Load outputs from a npz file

        Returns:
            List[np.ndarray]: hidden_state,  input_ids, attention_mask
        """
        data = np.load(npz_path)
        hidden_state = data["hidden_state"]
        attention_mask = data["attention_mask"]
        input_ids = data["input_ids"]
        return [hidden_state, input_ids, attention_mask]

    def cache_batch_outputs(
        self,
        tokenize_strategy: LuminaTokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: LuminaTextEncodingStrategy,
        infos: List,
    ):
        lumina_text_encoding_strategy: LuminaTextEncodingStrategy = (
            text_encoding_strategy
        )
        captions = [info.caption for info in infos]

        if self.is_weighted:
            tokens, weights_list = tokenize_strategy.tokenize_with_weights(
                captions
            )
            with torch.no_grad():
                hidden_state, input_ids, attention_masks = lumina_text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy, models, tokens, weights_list
                )
        else:
            tokens = tokenize_strategy.tokenize(captions)
            with torch.no_grad():
                hidden_state, input_ids, attention_masks = lumina_text_encoding_strategy.encode_tokens(
                    tokenize_strategy, models, tokens
                )

        if hidden_state.dtype != torch.float32:
            hidden_state = hidden_state.float()

        hidden_state = hidden_state.cpu().numpy()
        attention_mask = attention_masks.cpu().numpy()
        input_ids = tokens.cpu().numpy()


        for i, info in enumerate(infos):
            hidden_state_i = hidden_state[i]
            attention_mask_i = attention_mask[i]
            input_ids_i = input_ids[i]

            if self.cache_to_disk:
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_state=hidden_state_i,
                    attention_mask=attention_mask_i,
                    input_ids=input_ids_i
                )
            else:
                info.text_encoder_outputs = [hidden_state_i, attention_mask_i, input_ids_i]


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
