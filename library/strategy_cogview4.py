import os
import glob
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import AutoTokenizer

from library import train_util
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


GLM_TOKENIZER_ID = "THUDM/CogView4-6B"


class CogView4TokenizeStrategy(TokenizeStrategy):
    def __init__(self, max_length: int = 512, tokenizer_cache_dir: Optional[str] = None) -> None:
        self.max_length = max_length
        self.tokenizer = self._load_tokenizer(AutoTokenizer, GLM_TOKENIZER_ID, tokenizer_cache_dir=tokenizer_cache_dir)
        # Add special tokens if needed
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def tokenize(self, text: Union[str, List[str]]) -> List[torch.Tensor]:
        text = [text] if isinstance(text, str) else text
        
        # Tokenize with GLM tokenizer
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]
        
        return [input_ids, attention_mask]


class CogView4TextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, apply_attention_mask: bool = True) -> None:
        """
        Args:
            apply_attention_mask: Whether to apply attention mask during encoding.
        """
        self.apply_attention_mask = apply_attention_mask

    def encode_tokens(
        self,
        tokenize_strategy: TokenizeStrategy,
        models: List[Any],
        tokens: List[torch.Tensor],
        apply_attention_mask: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        # supports single model inference
        if apply_attention_mask is None:
            apply_attention_mask = self.apply_attention_mask

        # Get GLM model (should be the only model in the list)
        glm_model = models[0]
        input_ids = tokens[0]
        attention_mask = tokens[1] if len(tokens) > 1 else None

        # Move tensors to the correct device
        device = glm_model.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Get GLM model outputs
        with torch.no_grad():
            outputs = glm_model(
                input_ids=input_ids,
                attention_mask=attention_mask if apply_attention_mask else None,
                output_hidden_states=True,
                return_dict=True
            )

        # Get the last hidden state
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # For compatibility with existing code, we'll return a list similar to the original
        # but with GLM's hidden states instead of CLIP/T5 outputs
        return [
            hidden_states,  # Replaces l_pooled
            hidden_states,  # Replaces t5_out (same tensor for now, can be modified if needed)
            torch.zeros(hidden_states.shape[0], hidden_states.shape[1], 3, device=device),  # txt_ids placeholder
            attention_mask  # attention mask
        ]


class CogView4TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    COGVIEW4_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_cogview4_te.npz"

    def __init__(
        self,
        cache_to_disk: bool,
        batch_size: int,
        skip_disk_cache_validity_check: bool,
        is_partial: bool = False,
        apply_attention_mask: bool = True,
    ) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)
        self.apply_attention_mask = apply_attention_mask
        self.warn_fp8_weights = False

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + CogView4TextEncoderOutputsCachingStrategy.COGVIEW4_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            required_fields = ["hidden_states", "attention_mask", "apply_attention_mask"]
            for field in required_fields:
                if field not in npz:
                    return False
            
            npz_apply_attention_mask = bool(npz["apply_attention_mask"])
            if npz_apply_attention_mask != self.apply_attention_mask:
                return False
                
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            logger.exception(e)
            return False

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hidden_states = data["hidden_states"]
        attention_mask = data["attention_mask"]
        return [
            hidden_states,  # l_pooled replacement
            hidden_states,  # t5_out replacement
            np.zeros((hidden_states.shape[0], hidden_states.shape[1], 3), dtype=np.float32),  # txt_ids
            attention_mask  # attention mask
        ]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        if not self.warn_fp8_weights:
            model_dtype = next(models[0].parameters()).dtype
            if model_dtype == torch.float8_e4m3fn or model_dtype == torch.float8_e5m2:
                logger.warning(
                    "Model is using fp8 weights for caching. This may affect the quality of the cached outputs."
                    " / モデルはfp8の重みを使用しています。これはキャッシュの品質に影響を与える可能性があります。"
                )
            self.warn_fp8_weights = True

        captions = [info.caption for info in infos]

        tokens_and_masks = tokenize_strategy.tokenize(captions)
        
        with torch.no_grad():
            hidden_states, _, _, attention_mask = text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens_and_masks
            )

        if hidden_states.dtype == torch.bfloat16:
            hidden_states = hidden_states.float()
        
        hidden_states = hidden_states.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy() if attention_mask is not None else None

        for i, info in enumerate(infos):
            hidden_states_i = hidden_states[i]
            attention_mask_i = attention_mask[i] if attention_mask is not None else None

            if self.cache_to_disk and hasattr(info, 'text_encoder_outputs_npz'):
                np.savez(
                    info.text_encoder_outputs_npz,
                    hidden_states=hidden_states_i,
                    attention_mask=attention_mask_i,
                    apply_attention_mask=self.apply_attention_mask,
                )
            else:
                info.text_encoder_outputs = (hidden_states_i, hidden_states_i, np.zeros((hidden_states_i.shape[0], 3), dtype=np.float32), attention_mask_i)


class CogView4LatentsCachingStrategy(LatentsCachingStrategy):
    COGVIEW4_LATENTS_NPZ_SUFFIX = "_cogview4.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return CogView4LatentsCachingStrategy.COGVIEW4_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        """Get the path for cached latents.
        
        Args:
            absolute_path: Absolute path to the source image
            image_size: Tuple of (height, width) for the target resolution
            
        Returns:
            Path to the cached latents file
        """
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + CogView4LatentsCachingStrategy.COGVIEW4_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(
        self, 
        bucket_reso: Tuple[int, int], 
        npz_path: str, 
        flip_aug: bool, 
        alpha_mask: bool
    ) -> bool:
        """Check if the latents are already cached and valid.
        
        Args:
            bucket_reso: Target resolution as (height, width)
            npz_path: Path to the cached latents file
            flip_aug: Whether flip augmentation was applied
            alpha_mask: Whether alpha mask was used
            
        Returns:
            bool: True if valid cache exists, False otherwise
        """
        # Using 8 as the default number of frames for compatibility
        return self._default_is_disk_cached_latents_expected(
            8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True
        )

    def load_latents_from_disk(
        self, 
        npz_path: str, 
        bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load latents from disk.
        
        Args:
            npz_path: Path to the cached latents file
            bucket_reso: Target resolution as (height, width)
            
        Returns:
            Tuple containing:
            - latents: The loaded latents or None if loading failed
            - original_size: Original image size as [height, width]
            - crop_top_left: Crop offset as [top, left]
            - alpha_mask: Alpha mask if available
            - alpha_mask_origin: Original alpha mask if available
        """
        # Using 8 as the default number of frames for compatibility
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    def cache_batch_latents(
        self, 
        vae: Any, 
        image_infos: List[Any], 
        flip_aug: bool, 
        alpha_mask: bool, 
        random_crop: bool
    ) -> None:
        """Cache a batch of latents.
        
        Args:
            vae: The VAE model used for encoding
            image_infos: List of image information objects
            flip_aug: Whether to apply flip augmentation
            alpha_mask: Whether to use alpha mask
            random_crop: Whether to apply random crop
        """
        # Define encoding function that moves output to CPU
        def encode_by_vae(img_tensor: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return vae.encode(img_tensor).to("cpu")

        # Get VAE device and dtype
        vae_device = vae.device
        vae_dtype = vae.dtype

        # Cache latents using the default implementation
        self._default_cache_batch_latents(
            encode_by_vae, 
            vae_device, 
            vae_dtype, 
            image_infos, 
            flip_aug, 
            alpha_mask, 
            random_crop, 
            multi_resolution=True
        )

        # Clean up GPU memory if not in high VRAM mode
        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)


if __name__ == "__main__":
    # Test code for CogView4TokenizeStrategy
    tokenizer = CogView4TokenizeStrategy(512)
    text = "hello world"
    
    # Test single text tokenization
    input_ids, attention_mask = tokenizer.tokenize(text)
    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)
    
    # Test batch tokenization
    texts = ["hello world", "the quick brown fox jumps over the lazy dog"]
    batch_input_ids, batch_attention_mask = tokenizer.tokenize(texts)
    print("\nBatch Input IDs:", batch_input_ids.shape)
    print("Batch Attention Mask:", batch_attention_mask.shape)
    
    # Test with a long text
    long_text = ",".join(["hello world! this is long text"] * 10)
    long_input_ids, long_attention_mask = tokenizer.tokenize(long_text)
    print("\nLong text input IDs shape:", long_input_ids.shape)
    print("Long text attention mask shape:", long_attention_mask.shape)
    
    # Test text encoding strategy
    print("\nTesting text encoding strategy...")
    from transformers import AutoModel
    
    # Load a small GLM model for testing
    model = AutoModel.from_pretrained("THUDM/glm-10b-chinese", trust_remote_code=True)
    model.eval()
    
    encoding_strategy = CogView4TextEncodingStrategy()
    tokens = tokenizer.tokenize(texts)
    encoded = encoding_strategy.encode_tokens(tokenizer, [model], tokens)
    
    print(f"Number of outputs: {len(encoded)}")
    print(f"Hidden states shape: {encoded[0].shape}")
    print(f"Attention mask shape: {encoded[3].shape if encoded[3] is not None else 'None'}")
    
    # Test caching strategy
    print("\nTesting caching strategy...")
    import tempfile
    import os
    
    class DummyInfo:
        def __init__(self, caption):
            self.caption = caption
            self.text_encoder_outputs_npz = tempfile.mktemp(suffix=".npz")
    
    # Create test data
    infos = [DummyInfo(text) for text in texts]
    
    # Test caching
    caching_strategy = CogView4TextEncoderOutputsCachingStrategy(
        cache_to_disk=True,
        batch_size=2,
        skip_disk_cache_validity_check=False
    )
    
    # Cache the outputs
    caching_strategy.cache_batch_outputs(tokenizer, [model], encoding_strategy, infos)
    
    # Check if files were created
    for info in infos:
        exists = os.path.exists(info.text_encoder_outputs_npz)
        print(f"Cache file {info.text_encoder_outputs_npz} exists: {exists}")
    
    # Clean up
    for info in infos:
        if os.path.exists(info.text_encoder_outputs_npz):
            os.remove(info.text_encoder_outputs_npz)
