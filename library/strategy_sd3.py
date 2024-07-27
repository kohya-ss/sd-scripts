import os
import glob
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast

from library import sd3_utils, train_util
from library import sd3_models
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_ID = "openai/clip-vit-large-patch14"
CLIP_G_TOKENIZER_ID = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
T5_XXL_TOKENIZER_ID = "google/t5-v1_1-xxl"


class Sd3TokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 256, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.t5xxl_max_length = t5xxl_max_length
        self.clip_l = self._load_tokenizer(CLIPTokenizer, CLIP_L_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.clip_g = self._load_tokenizer(CLIPTokenizer, CLIP_G_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.t5xxl = self._load_tokenizer(T5TokenizerFast, T5_XXL_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        self.clip_g.pad_token_id = 0  # use 0 as pad token for clip_g

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text

        l_tokens = self.clip_l(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        g_tokens = self.clip_g(text, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
        t5_tokens = self.t5xxl(text, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt")

        l_tokens = l_tokens["input_ids"]
        g_tokens = g_tokens["input_ids"]
        t5_tokens = t5_tokens["input_ids"]

        return [l_tokens, g_tokens, t5_tokens]


class Sd3TextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        clip_l, clip_g, t5xxl = models

        l_tokens, g_tokens, t5_tokens = tokens
        if l_tokens is None:
            assert g_tokens is None, "g_tokens must be None if l_tokens is None"
            lg_out = None
        else:
            assert g_tokens is not None, "g_tokens must not be None if l_tokens is not None"
            l_out, l_pooled = clip_l(l_tokens)
            g_out, g_pooled = clip_g(g_tokens)
            lg_out = torch.cat([l_out, g_out], dim=-1)

        if t5xxl is not None and t5_tokens is not None:
            t5_out, _ = t5xxl(t5_tokens)  # t5_out is [1, max length, 4096]
        else:
            t5_out = None

        lg_pooled = torch.cat((l_pooled, g_pooled), dim=-1) if l_tokens is not None else None
        return [lg_out, t5_out, lg_pooled]

    def concat_encodings(
        self, lg_out: torch.Tensor, t5_out: Optional[torch.Tensor], lg_pooled: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        if t5_out is None:
            t5_out = torch.zeros((lg_out.shape[0], 77, 4096), device=lg_out.device, dtype=lg_out.dtype)
        return torch.cat([lg_out, t5_out], dim=-2), lg_pooled


class Sd3TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    SD3_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_sd3_te.npz"

    def __init__(
        self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, is_partial: bool = False
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + Sd3TextEncoderOutputsCachingStrategy.SD3_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, abs_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(self.get_outputs_npz_path(abs_path)):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(self.get_outputs_npz_path(abs_path))
            if "clip_l" not in npz or "clip_g" not in npz:
                return False
            if "clip_l_pool" not in npz or "clip_g_pool" not in npz:
                return False
            # t5xxl is optional
        except Exception as e:
            logger.error(f"Error loading file: {self.get_outputs_npz_path(abs_path)}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        lg_out = data["lg_out"]
        lg_pooled = data["lg_pooled"]
        t5_out = data["t5_out"] if "t5_out" in data else None
        return [lg_out, t5_out, lg_pooled]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        captions = [info.caption for info in infos]

        clip_l_tokens, clip_g_tokens, t5xxl_tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            lg_out, t5_out, lg_pooled = text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, [clip_l_tokens, clip_g_tokens, t5xxl_tokens]
            )

        if lg_out.dtype == torch.bfloat16:
            lg_out = lg_out.float()
        if lg_pooled.dtype == torch.bfloat16:
            lg_pooled = lg_pooled.float()
        if t5_out is not None and t5_out.dtype == torch.bfloat16:
            t5_out = t5_out.float()

        lg_out = lg_out.cpu().numpy()
        lg_pooled = lg_pooled.cpu().numpy()
        if t5_out is not None:
            t5_out = t5_out.cpu().numpy()

        for i, info in enumerate(infos):
            lg_out_i = lg_out[i]
            t5_out_i = t5_out[i] if t5_out is not None else None
            lg_pooled_i = lg_pooled[i]

            if self.cache_to_disk:
                kwargs = {}
                if t5_out is not None:
                    kwargs["t5_out"] = t5_out_i
                np.savez(info.text_encoder_outputs_npz, lg_out=lg_out_i, lg_pooled=lg_pooled_i, **kwargs)
            else:
                info.text_encoder_outputs = (lg_out_i, t5_out_i, lg_pooled_i)


class Sd3LatentsCachingStrategy(LatentsCachingStrategy):
    SD3_LATENTS_NPZ_SUFFIX = "_sd3.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    def get_image_size_from_disk_cache_path(self, absolute_path: str) -> Tuple[Optional[int], Optional[int]]:
        npz_file = glob.glob(os.path.splitext(absolute_path)[0] + "_*" + Sd3LatentsCachingStrategy.SD3_LATENTS_NPZ_SUFFIX)
        if len(npz_file) == 0:
            return None, None
        w, h = os.path.splitext(npz_file[0])[0].split("_")[-2].split("x")
        return int(w), int(h)

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + Sd3LatentsCachingStrategy.SD3_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._defualt_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask)

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop)

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)


if __name__ == "__main__":
    # test code for Sd3TokenizeStrategy
    # tokenizer = sd3_models.SD3Tokenizer()
    strategy = Sd3TokenizeStrategy(256)
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
