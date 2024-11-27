import os
import glob
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import flux_utils, train_util, utils
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class FluxTokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 512, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.t5xxl_max_length = t5xxl_max_length
        self.clip_l = self._load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        l_tokens = self.clip_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

        t5_attn_mask = t5_tokens["attention_mask"]
        l_tokens = l_tokens["input_ids"]
        t5_tokens = t5_tokens["input_ids"]

        return [l_tokens, t5_tokens, t5_attn_mask]


class FluxTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: Optional[bool] = None) -> None:
        """
        Args:
            apply_t5_attn_mask: Default value for apply_t5_attn_mask.
        """
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        apply_t5_attn_mask: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        # supports single model inference

        if apply_t5_attn_mask is None:
            apply_t5_attn_mask = self.apply_t5_attn_mask

        clip_l, t5xxl = models if len(models) == 2 else (models[0], None)
        l_tokens, t5_tokens = tokens[:2]
        t5_attn_mask = tokens[2] if len(tokens) > 2 else None

        # clip_l is None when using T5 only
        if clip_l is not None and l_tokens is not None:
            l_pooled = clip_l(l_tokens.to(clip_l.device))["pooler_output"]
        else:
            l_pooled = None

        # t5xxl is None when using CLIP only
        if t5xxl is not None and t5_tokens is not None:
            # t5_out is [b, max length, 4096]
            attention_mask = None if not apply_t5_attn_mask else t5_attn_mask.to(t5xxl.device)
            t5_out, _ = t5xxl(t5_tokens.to(t5xxl.device), attention_mask, return_dict=False, output_hidden_states=True)
            # if zero_pad_t5_output:
            #     t5_out = t5_out * t5_attn_mask.to(t5_out.device).unsqueeze(-1)
            txt_ids = torch.zeros(t5_out.shape[0], t5_out.shape[1], 3, device=t5_out.device)
        else:
            t5_out = None
            txt_ids = None
            t5_attn_mask = None  # caption may be dropped/shuffled, so t5_attn_mask should not be used to make sure the mask is same as the cached one

        return [l_pooled, t5_out, txt_ids, t5_attn_mask]  # returns t5_attn_mask for attention mask in transformer


class FluxTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    KEYS = ["l_pooled", "t5_out", "txt_ids"]
    KEYS_MASKED = ["t5_attn_mask", "apply_t5_attn_mask"]

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        max_token_length: int,
        masked: bool,
        is_partial: bool = False,
    ) -> None:
        super().__init__(
            FluxLatentsCachingStrategy.ARCHITECTURE,
            cache_to_disk,
            batch_size,
            skip_disk_cache_validity_check,
            max_token_length,
            masked,
            is_partial,
        )

        self.warn_fp8_weights = False

    def is_disk_cached_outputs_expected(
        self, cache_path: str, prompts: list[str], preferred_dtype: Optional[Union[str, torch.dtype]]
    ):
        keys = FluxTextEncoderOutputsCachingStrategy.KEYS
        if self.masked:
            keys += FluxTextEncoderOutputsCachingStrategy.KEYS_MASKED
        return self._default_is_disk_cached_outputs_expected(cache_path, prompts, keys, preferred_dtype)

    def load_from_disk(self, cache_path: str, caption_index: int) -> list[Optional[torch.Tensor]]:
        l_pooled, t5_out, txt_ids = self.load_from_disk_for_keys(
            cache_path, caption_index, FluxTextEncoderOutputsCachingStrategy.KEYS
        )
        if self.masked:
            t5_attn_mask = self.load_from_disk_for_keys(
                cache_path, caption_index, FluxTextEncoderOutputsCachingStrategy.KEYS_MASKED
            )[0]
        else:
            t5_attn_mask = None
        return [l_pooled, t5_out, txt_ids, t5_attn_mask]

    def cache_batch_outputs(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        text_encoding_strategy: TextEncodingStrategy,
        batch: list[tuple[utils.ImageInfo, int, str]],
    ):
        if not self.warn_fp8_weights:
            if flux_utils.get_t5xxl_actual_dtype(models[1]) == torch.float8_e4m3fn:
                logger.warning(
                    "T5 model is using fp8 weights for caching. This may affect the quality of the cached outputs."
                    " / T5モデルはfp8の重みを使用しています。これはキャッシュの品質に影響を与える可能性があります。"
                )
            self.warn_fp8_weights = True

        flux_text_encoding_strategy: FluxTextEncodingStrategy = text_encoding_strategy
        captions = [caption for _, _, caption in batch]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            # attn_mask is applied in text_encoding_strategy.encode_tokens if apply_t5_attn_mask is True
            l_pooled, t5_out, txt_ids, _ = flux_text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens_and_masks)

        l_pooled = l_pooled.cpu()
        t5_out = t5_out.cpu()
        txt_ids = txt_ids.cpu()
        t5_attn_mask = tokens_and_masks[2].cpu()

        keys = FluxTextEncoderOutputsCachingStrategy.KEYS
        if self.masked:
            keys += FluxTextEncoderOutputsCachingStrategy.KEYS_MASKED

        for i, (info, caption_index, caption) in enumerate(batch):
            l_pooled_i = l_pooled[i]
            t5_out_i = t5_out[i]
            txt_ids_i = txt_ids[i]
            t5_attn_mask_i = t5_attn_mask[i]

            if self.cache_to_disk:
                outputs = [l_pooled_i, t5_out_i, txt_ids_i]
                if self.masked:
                    outputs += [t5_attn_mask_i]
                self.save_outputs_to_disk(info.text_encoder_outputs_cache_path, caption_index, caption, keys, outputs)
            else:
                # it's fine that attn mask is not None. it's overwritten before calling the model if necessary
                while len(info.text_encoder_outputs) <= caption_index:
                    info.text_encoder_outputs.append(None)
                info.text_encoder_outputs[caption_index] = [l_pooled_i, t5_out_i, txt_ids_i, t5_attn_mask_i]


class FluxLatentsCachingStrategy(LatentsCachingStrategy):
    ARCHITECTURE = "flux"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(FluxLatentsCachingStrategy.ARCHITECTURE, 8, cache_to_disk, batch_size, skip_disk_cache_validity_check)

    def is_disk_cached_latents_expected(
        self,
        bucket_reso: Tuple[int, int],
        cache_path: str,
        flip_aug: bool,
        alpha_mask: bool,
        preferred_dtype: Optional[torch.dtype] = None,
    ):
        return self._default_is_disk_cached_latents_expected(bucket_reso, cache_path, flip_aug, alpha_mask, preferred_dtype)

    def load_latents_from_disk(
        self, cache_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[torch.Tensor, List[int], List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self._default_load_latents_from_disk(cache_path, bucket_reso)

    def cache_batch_latents(self, vae, image_infos: List[utils.ImageInfo], flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop)

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)


if __name__ == "__main__":
    # test code for FluxTokenizeStrategy
    # tokenizer = sd3_models.SD3Tokenizer()
    strategy = FluxTokenizeStrategy(256)
    text = "hello world"

    l_tokens, g_tokens, t5_tokens = strategy.tokenize(text)
    # print(l_tokens.shape)
    print(l_tokens)
    print(g_tokens)
    print(t5_tokens)

    texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
    l_tokens_2 = strategy.clip_l(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    g_tokens_2 = strategy.clip_g(texts, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
    t5_tokens_2 = strategy.t5xxl(
        texts, max_length=strategy.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    print(l_tokens_2)
    print(g_tokens_2)
    print(t5_tokens_2)

    # compare
    print(torch.allclose(l_tokens, l_tokens_2["input_ids"][0]))
    print(torch.allclose(g_tokens, g_tokens_2["input_ids"][0]))
    print(torch.allclose(t5_tokens, t5_tokens_2["input_ids"][0]))

    text = ",".join(["hello world! this is long text"] * 50)
    l_tokens, g_tokens, t5_tokens = strategy.tokenize(text)
    print(l_tokens)
    print(g_tokens)
    print(t5_tokens)

    print(f"model max length l: {strategy.clip_l.model_max_length}")
    print(f"model max length g: {strategy.clip_g.model_max_length}")
    print(f"model max length t5: {strategy.t5xxl.model_max_length}")
