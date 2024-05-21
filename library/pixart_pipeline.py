from PIL import Image
from typing import Callable, List, Optional, Union
from tqdm import tqdm
import numpy as np
import torch
import library.sdxl_model_util as sdxl_model_util
from diffusers import SchedulerMixin, PixArtSigmaPipeline

# simple diffusers pipeline wrapper
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
        self.dit = unet
        self.vae = vae
        self.pipe = PixArtSigmaPipeline(tokenizer,text_encoder,vae,self.dit,scheduler)
        
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
        output_type: Optional[str] = "latents",
        return_dict: bool = True,
        controlnet=None,
        controlnet_image=None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: int = 1,
    ):
        return self.pipe(prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            timesteps=None,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_embeds=None,
            negative_prompt_attention_mask=None,
            output_type=output_type,
            return_dict=output_type,
            callback=callback,
            callback_steps=callback_steps,
            clean_caption=None,
            use_resolution_binningh=None,
            max_sequence_length=None,
        )

    # callable functions
    def latents_to_image(self, latents):
        # 9. Post-processing
        image = self.decode_latents(latents.to(self.vae.dtype))
        image = self.numpy_to_pil(image)
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
    
    def decode_latents(self, latents):
        with torch.no_grad():
            latents = 1 / sdxl_model_util.VAE_SCALE_FACTOR * latents

            # print("post_quant_conv dtype:", self.vae.post_quant_conv.weight.dtype)  # torch.float32
            # x = torch.nn.functional.conv2d(latents, self.vae.post_quant_conv.weight.detach(), stride=1, padding=0)
            # print("latents dtype:", latents.dtype, "x dtype:", x.dtype)  # torch.float32, torch.float16
            # self.vae.to("cpu")
            # self.vae.set_use_memory_efficient_attention_xformers(False)
            # image = self.vae.decode(latents.to("cpu")).sample

            image = self.vae.decode(latents.to(self.vae.dtype)).sample
            image = (image / 2 + 0.5).clamp(0, 1)
            # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
            return image