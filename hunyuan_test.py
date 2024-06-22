import numpy as np
import torch

from k_diffusion.external import DiscreteVDDPMDenoiser
from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential

from PIL import Image
from pytorch_lightning import seed_everything

from library.hunyuan_models import *
from library.hunyuan_utils import *

from lycoris.kohya import create_network_from_weights


PROMPT = """
2girls, multiple girls, 

cesario (umamusume), daring tact (umamusume), umamusume, 

shino (ponjiyuusu), 

aged down, ahoge, animal ears, black hair, blue hair, blue sky, braid, cloud, crown braid, 
day, green eyes, horse ears, horse girl, horse tail, loafers, long hair, mole, mole under eye, 
multicolored hair, outdoors, pleated skirt, purple eyes, purple sailor collar, purple shirt, 
purple skirt, purple thighhighs, sailor collar, school uniform, shirt, shoes, short hair, 
skirt, sky, tail, thighhighs, time paradox, tracen school uniform, two-tone hair, white hair, 
winter uniform, 

masterpiece, absurdres, latest, safe
"""
NEG_PROMPT = "错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺"
CLIP_TOKENS = 75 * 3 + 2
ATTN_MODE = "xformers"
H = 1024
W = 1024
STEPS = 16
CFG_SCALE = 5
DEVICE = "cuda"
DTYPE = torch.float16


if __name__ == "__main__":
    seed_everything(0)
    with torch.inference_mode(True), torch.no_grad():
        alphas, sigmas = load_scheduler_sigmas()
        (
            denoiser,
            patch_size,
            head_dim,
            clip_tokenizer,
            clip_encoder,
            mt5_embedder,
            vae,
        ) = load_model("./model", dtype=DTYPE, device=DEVICE)
        denoiser.eval()
        denoiser.disable_fp32_silu()
        denoiser.disable_fp32_layer_norm()
        denoiser.set_attn_mode(ATTN_MODE)
        vae.requires_grad_(False)
        mt5_embedder.to(torch.float16)

        lycoris_net, state_dict = create_network_from_weights(
            multiplier=1.0,
            file="./output/lokr-hy-test-step00000600.safetensors",
            vae=vae,
            text_encoder=[clip_encoder, mt5_embedder],
            unet=denoiser,
        )
        lycoris_net.merge_to(
            text_encoder=[clip_encoder, mt5_embedder],
            unet=denoiser,
            weights_sd=state_dict,
            dtype=DTYPE,
            device=DEVICE,
        )

        with torch.autocast("cuda"):
            clip_h, clip_m, mt5_h, mt5_m = get_cond(
                PROMPT,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                # Should be same as original implementation with max_length_clip=77
                # Support 75*n + 2
                max_length_clip=CLIP_TOKENS,
            )
            neg_clip_h, neg_clip_m, neg_mt5_h, neg_mt5_m = get_cond(
                NEG_PROMPT,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                max_length_clip=CLIP_TOKENS,
            )
            clip_h = torch.concat([clip_h, neg_clip_h], dim=0)
            clip_m = torch.concat([clip_m, neg_clip_m], dim=0)
            mt5_h = torch.concat([mt5_h, neg_mt5_h], dim=0)
            mt5_m = torch.concat([mt5_m, neg_mt5_m], dim=0)
            torch.cuda.empty_cache()

        style = torch.as_tensor([0] * 2, device=DEVICE)
        # src hw, dst hw, 0, 0
        size_cond = [H, W, H, W, 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * 2, device=DEVICE)
        freqs_cis_img = calc_rope(H, W, patch_size, head_dim)

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
                encoder_hidden_states=clip_h,
                text_embedding_mask=clip_m,
                encoder_hidden_states_t5=mt5_h,
                text_embedding_mask_t5=mt5_m,
                image_meta_size=image_meta_size,
                style=style,
                cos_cis_img=freqs_cis_img[0],
                sin_cis_img=freqs_cis_img[1],
            ).chunk(2, dim=0)
            return uncond + (cond - uncond) * CFG_SCALE

        sigmas = denoiser_wrapper.get_sigmas(STEPS).to(DEVICE)
        sigmas = get_sigmas_exponential(
            STEPS, denoiser_wrapper.sigma_min, denoiser_wrapper.sigma_max, DEVICE
        )
        x1 = torch.randn(1, 4, H//8, W//8, dtype=torch.float16, device=DEVICE)

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
