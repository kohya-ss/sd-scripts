import glob
import os
from typing import Any, List, Optional, Tuple, Union

import torch
from transformers import CLIPTokenizer
from library import train_util
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncodingStrategy
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


TOKENIZER_ID = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_ID = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う v2とv2.1はtokenizer仕様は同じ


class SdTokenizeStrategy(TokenizeStrategy):
    def __init__(self, v2: bool, max_length: Optional[int], tokenizer_cache_dir: Optional[str] = None) -> None:
        """
        max_length does not include <BOS> and <EOS> (None, 75, 150, 225)
        """
        logger.info(f"Using {'v2' if v2 else 'v1'} tokenizer")
        if v2:
            self.tokenizer = self._load_tokenizer(
                CLIPTokenizer, V2_STABLE_DIFFUSION_ID, subfolder="tokenizer", tokenizer_cache_dir=tokenizer_cache_dir
            )
        else:
            self.tokenizer = self._load_tokenizer(CLIPTokenizer, TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
        else:
            self.max_length = max_length + 2
        
        self.break_separator = "BREAK"
    
    def _split_on_break(self, text: str) -> List[str]:
        """Split text on BREAK separator (case-sensitive), filtering empty segments."""
        segments = text.split(self.break_separator)
        # Filter out empty or whitespace-only segments
        filtered = [seg.strip() for seg in segments if seg.strip()]
        # Return at least one segment to maintain consistency
        return filtered if filtered else [""]
    
    def _tokenize_segments(self, segments: List[str], weighted: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Tokenize multiple segments and concatenate them."""
        if len(segments) == 1:
            # No BREAK present, use existing logic
            if weighted:
                return self._get_input_ids(self.tokenizer, segments[0], self.max_length, weighted=True)
            else:
                tokens = self._get_input_ids(self.tokenizer, segments[0], self.max_length)
                return tokens, None
        
        # Multiple segments - tokenize each separately
        all_tokens = []
        all_weights = [] if weighted else None
        
        for segment in segments:
            if weighted:
                seg_tokens, seg_weights = self._get_input_ids(self.tokenizer, segment, self.max_length, weighted=True)
                all_tokens.append(seg_tokens)
                all_weights.append(seg_weights)
            else:
                seg_tokens = self._get_input_ids(self.tokenizer, segment, self.max_length)
                all_tokens.append(seg_tokens)
        
        # Concatenate along the sequence dimension (dim=1 for tokens that are [batch, seq_len] or [n_chunks, seq_len])
        combined_tokens = torch.cat(all_tokens, dim=1) if all_tokens[0].dim() == 2 else torch.cat(all_tokens, dim=0)
        combined_weights = None
        if weighted:
            combined_weights = torch.cat(all_weights, dim=1) if all_weights[0].dim() == 2 else torch.cat(all_weights, dim=0)
        
        return combined_tokens, combined_weights
    
    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        
        tokens_list = []
        for t in text:
            segments = self._split_on_break(t)
            tokens, _ = self._tokenize_segments(segments, weighted=False)
            tokens_list.append(tokens)
        
        # Pad tokens to same length for stacking
        max_length = max(t.shape[-1] for t in tokens_list)
        padded_tokens = []
        for tokens in tokens_list:
            if tokens.shape[-1] < max_length:
                # Pad with pad_token_id
                pad_size = max_length - tokens.shape[-1]
                if tokens.dim() == 2:
                    padding = torch.full((tokens.shape[0], pad_size), self.tokenizer.pad_token_id, dtype=tokens.dtype)
                    tokens = torch.cat([tokens, padding], dim=1)
                else:
                    padding = torch.full((pad_size,), self.tokenizer.pad_token_id, dtype=tokens.dtype)
                    tokens = torch.cat([tokens, padding], dim=0)
            padded_tokens.append(tokens)
        
        return [torch.stack(padded_tokens, dim=0)]
    
    def tokenize_with_weights(self, text: str | List[str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        text = [text] if isinstance(text, str) else text
        
        tokens_list = []
        weights_list = []
        for t in text:
            segments = self._split_on_break(t)
            tokens, weights = self._tokenize_segments(segments, weighted=True)
            tokens_list.append(tokens)
            weights_list.append(weights)
        
        return [torch.stack(tokens_list, dim=0)], [torch.stack(weights_list, dim=0)]


class SdTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, clip_skip: Optional[int] = None) -> None:
        self.clip_skip = clip_skip
    
    def _encode_with_clip_skip(self, text_encoder: Any, tokens: torch.Tensor) -> torch.Tensor:
        """Encode tokens with optional CLIP skip."""
        if self.clip_skip is None:
            return text_encoder(tokens)[0]
        
        enc_out = text_encoder(tokens, output_hidden_states=True, return_dict=True)
        hidden_states = enc_out["hidden_states"][-self.clip_skip]
        return text_encoder.text_model.final_layer_norm(hidden_states)
    
    def _reconstruct_embeddings(self, encoder_hidden_states: torch.Tensor, tokens: torch.Tensor,
                                max_token_length: int, model_max_length: int, 
                                tokenizer: Any) -> torch.Tensor:
        """Reconstruct embeddings from chunked encoding."""
        v1 = tokenizer.pad_token_id == tokenizer.eos_token_id
        states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
        
        if not v1:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す
            for i in range(1, max_token_length, model_max_length):
                chunk = encoder_hidden_states[:, i : i + model_max_length - 2]
                if i > 0:
                    for j in range(len(chunk)):
                        if tokens[j, 1] == tokenizer.eos_token:
                            chunk[j, 0] = chunk[j, 1]
                states_list.append(chunk)
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            for i in range(1, max_token_length, model_max_length):
                states_list.append(encoder_hidden_states[:, i : i + model_max_length - 2])
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))
        
        return torch.cat(states_list, dim=1)
    
    def _apply_weights_single_chunk(self, encoder_hidden_states: torch.Tensor, 
                                     weights: torch.Tensor) -> torch.Tensor:
        """Apply weights for single chunk case (no max_token_length)."""
        return encoder_hidden_states * weights.squeeze(1).unsqueeze(2)
    
    def _apply_weights_multi_chunk(self, encoder_hidden_states: torch.Tensor, 
                                    weights: torch.Tensor) -> torch.Tensor:
        """Apply weights for multi-chunk case (with max_token_length)."""
        for i in range(weights.shape[1]):
            start_idx = i * 75 + 1
            end_idx = i * 75 + 76
            encoder_hidden_states[:, start_idx:end_idx] = (
                encoder_hidden_states[:, start_idx:end_idx] * weights[:, i, 1:-1].unsqueeze(-1)
            )
        return encoder_hidden_states
    
    def encode_tokens(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        text_encoder = models[0]
        tokens = tokens[0]
        sd_tokenize_strategy = tokenize_strategy  # type: SdTokenizeStrategy
        
        b_size = tokens.size()[0]
        max_token_length = tokens.size()[1] * tokens.size()[2]
        model_max_length = sd_tokenize_strategy.tokenizer.model_max_length
        
        tokens = tokens.reshape((-1, model_max_length))
        tokens = tokens.to(text_encoder.device)
        
        encoder_hidden_states = self._encode_with_clip_skip(text_encoder, tokens)
        encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))
        
        if max_token_length != model_max_length:
            encoder_hidden_states = self._reconstruct_embeddings(
                encoder_hidden_states, tokens, max_token_length, 
                model_max_length, sd_tokenize_strategy.tokenizer
            )
        
        return [encoder_hidden_states]
    
    def encode_tokens_with_weights(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens_list: List[torch.Tensor],
        weights_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        encoder_hidden_states = self.encode_tokens(tokenize_strategy, models, tokens_list)[0]
        weights = weights_list[0].to(encoder_hidden_states.device)
        
        if weights.shape[1] == 1:
            encoder_hidden_states = self._apply_weights_single_chunk(encoder_hidden_states, weights)
        else:
            encoder_hidden_states = self._apply_weights_multi_chunk(encoder_hidden_states, weights)
        
        return [encoder_hidden_states]

class SdSdxlLatentsCachingStrategy(LatentsCachingStrategy):
    # sd and sdxl share the same strategy. we can make them separate, but the difference is only the suffix.
    # and we keep the old npz for the backward compatibility.

    SD_OLD_LATENTS_NPZ_SUFFIX = ".npz"
    SD_LATENTS_NPZ_SUFFIX = "_sd.npz"
    SDXL_LATENTS_NPZ_SUFFIX = "_sdxl.npz"

    def __init__(self, sd: bool, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.sd = sd
        self.suffix = (
            SdSdxlLatentsCachingStrategy.SD_LATENTS_NPZ_SUFFIX if sd else SdSdxlLatentsCachingStrategy.SDXL_LATENTS_NPZ_SUFFIX
        )
    
    @property
    def cache_suffix(self) -> str:
        return self.suffix

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        # support old .npz
        old_npz_file = os.path.splitext(absolute_path)[0] + SdSdxlLatentsCachingStrategy.SD_OLD_LATENTS_NPZ_SUFFIX
        if os.path.exists(old_npz_file):
            return old_npz_file
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + self.suffix

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask)

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        encode_by_vae = lambda img_tensor: vae.encode(img_tensor).latent_dist.sample()
        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop)

        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)
