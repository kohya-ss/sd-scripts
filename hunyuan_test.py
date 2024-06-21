import numpy as np
import torch

from k_diffusion.external import DiscreteVDDPMDenoiser
from k_diffusion.sampling import sample_euler_ancestral

from PIL import Image
from pytorch_lightning import seed_everything

from library.hunyuan_models import *
from library.hunyuan_utils import *


PROMPT = """
Anime-style illustration of a young girl with long black hair, red eyes, and small red horns, 
wearing a traditional white kimono with delicate blue floral patterns and a red obi sash. 
She's standing in shallow, reflective water at night, waving with one hand. 
Her hair is adorned with red and white flowers. 
In the background, a large red torii gate stands prominently, silhouetted against a misty blue night sky. 
Multiple glowing paper lanterns float on the water, creating a warm, magical atmosphere. 
A ethereal blue waterfall cascades in the distance, surrounded by shadowy trees. 
The scene is illuminated by a soft, mystical light, 
highlighting the water's surface and creating a dreamy, fantastical mood. 
The color palette focuses on deep blues, vibrant reds, and soft whites. 
Highly detailed in anime art style with vibrant colors and smooth linework.
"""
NEG_PROMPT = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺"
CLIP_TOKENS = 75*1 + 2
ATTN_MODE = "xformers"
STEPS = 32
CFG_SCALE = 4
DEVICE = "cuda"
DTYPE = torch.bfloat16


if __name__ == "__main__":
    seed_everything(0)
    with torch.inference_mode(True):
        alphas, sigmas = load_scheduler_sigmas()
        denoiser, patch_size, head_dim, clip_tokenizer, clip_encoder, mt5_embedder, vae = (
            load_model("./model", dtype=DTYPE, device=DEVICE)
        )
        denoiser.enable_gradient_checkpointing()
        denoiser.set_attn_mode(ATTN_MODE)
        vae.requires_grad_(False)

        with torch.autocast("cuda"):
            clip_h, clip_m, mt5_h, mt5_m = get_cond(
                PROMPT,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                # Should be same as original implemention with max_length_clip=77
                # Support 75*n + 2
                max_length_clip=CLIP_TOKENS
            )
            neg_clip_h, neg_clip_m, neg_mt5_h, neg_mt5_m = get_cond(
                NEG_PROMPT,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                max_length_clip=CLIP_TOKENS
            )
            torch.cuda.empty_cache()

        style = torch.as_tensor([0], device=DEVICE)
        # src hw, dst hw, 0, 0
        size_cond = [1024, 1024, 1024, 1024, 0, 0]
        image_meta_size = torch.as_tensor([size_cond], device=DEVICE)
        freqs_cis_img = calc_rope(1024, 1024, patch_size, head_dim)

        denoiser_wrapper = DiscreteVDDPMDenoiser(
            # A quick patch for learn_sigma
            lambda *args, **kwargs: denoiser(*args, **kwargs).chunk(2, dim=1)[0],
            alphas,
            False,
        ).to(DEVICE)

        def cfg_denoise_func(x, sigma):
            cond, uncond = denoiser_wrapper(
                x.repeat(2, 1, 1, 1),
                sigma.repeat(2),
                encoder_hidden_states=torch.concat([clip_h, neg_clip_h], dim=0),
                text_embedding_mask=torch.concat([clip_m, neg_clip_m], dim=0),
                encoder_hidden_states_t5=torch.concat([mt5_h, neg_mt5_h], dim=0),
                text_embedding_mask_t5=torch.concat([mt5_m, neg_mt5_m], dim=0),
                image_meta_size=torch.concat([image_meta_size, image_meta_size], dim=0),
                style=torch.concat([style, style], dim=0),
                cos_cis_img=freqs_cis_img[0],
                sin_cis_img=freqs_cis_img[1],
            ).chunk(2, dim=0)
            return uncond + (cond - uncond) * CFG_SCALE

        sigmas = denoiser_wrapper.get_sigmas(STEPS).to(DEVICE)
        x1 = torch.randn(1, 4, 128, 128, dtype=torch.float16, device=DEVICE)

        with torch.autocast("cuda"):
            sample = sample_euler_ancestral(
                cfg_denoise_func,
                x1 * sigmas[0],
                sigmas,
            )
            torch.cuda.empty_cache()
            with torch.no_grad():
                latent = sample / 0.13025
                image = vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.permute(0, 2, 3, 1).cpu().numpy()
                image = (image * 255).round().astype(np.uint8)
                image = [Image.fromarray(im) for im in image]
                for im in image:
                    im.save("test.png")
