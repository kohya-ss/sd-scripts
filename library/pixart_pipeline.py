
# NOTE: very bare implementation to test it sooner
from PIL import Image
from typing import Callable, List, Optional, Union
from tqdm import tqdm
import numpy as np
import torch
import library.sdxl_model_util as sdxl_model_util
class SimplePixartPipeline:
    
    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet, # actually DiT
        scheduler,
        safety_checker,
        feature_extractor,
        requires_safety_checker: bool = True,
        clip_skip: int = 0,
    ):
        # clip skip is ignored currently
        # -- tokenizers and encoders are lists per train_network
        self.tokenizer = tokenizer[0]
        self.text_encoder = text_encoder[0]
        self.unet = unet
        self.scheduler = scheduler
        self.safety_checker = safety_checker
        self.feature_extractor = feature_extractor
        self.requires_safety_checker = requires_safety_checker
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.progress_bar = lambda x: tqdm(x, leave=False)

        self.clip_skip = clip_skip
        self.tokenizers = tokenizer
    
    @torch.no_grad()
    def decode_latents(self, latents):
        latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    
    # copy from pil_utils.py
    def numpy_to_pil(self, images: np.ndarray) -> Image.Image:
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def latents_to_image(self, latents):
        # 9. Post-processing
        image = self.decode_latents(latents.to(self.vae.dtype))
        image = self.numpy_to_pil(image)
        return image
    
    # latents = pipeline(
    #     prompt=prompt,
    #     height=height,
    #     width=width,
    #     num_inference_steps=sample_steps,
    #     guidance_scale=scale,
    #     negative_prompt=negative_prompt,
    #     controlnet=controlnet,
    #     controlnet_image=controlnet_image,
    # )
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        image: Union[torch.FloatTensor, Image.Image] = None,
        mask_image: Union[torch.FloatTensor, Image.Image] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        strength: float = 0.8,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        controlnet=None,
        controlnet_image=None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):
        raise NotImplementedError("kabachuha TODO")
