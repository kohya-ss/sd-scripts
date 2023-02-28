import torch
from diffusers import DDIMScheduler, AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTokenizer
from .loaders import load_checkpoint_with_text_encoder_conversion
from .converters import convert_vae_state_dict, \
    convert_text_encoder_state_dict_to_sd_v2, \
    convert_unet_state_dict_to_sd
from .model_utils2 import is_safetensors

DIFFUSERS_REF_MODEL_ID_V1 = "runwayml/stable-diffusion-v1-5"
DIFFUSERS_REF_MODEL_ID_V2 = "stabilityai/stable-diffusion-2-1"


def save_stable_diffusion_checkpoint(v2, output_file, text_encoder, unet, ckpt_path, epochs, steps, save_dtype=None,
                                     vae=None):
    if ckpt_path is not None:
        # epoch/stepを参照する。またVAEがメモリ上にないときなど、もう一度VAEを含めて読み込む
        checkpoint, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
        if checkpoint is None:  # safetensors または state_dictのckpt
            checkpoint = {}
            strict = False
        else:
            strict = True
        if "state_dict" in state_dict:
            del state_dict["state_dict"]
    else:
        # 新しく作る
        assert vae is not None, "VAE is required to save a checkpoint without a given checkpoint"
        checkpoint = {}
        state_dict = {}
        strict = False

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            assert not strict or key in state_dict, f"Illegal key in save SD: {key}"
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    unet_state_dict = convert_unet_state_dict_to_sd(v2, unet.state_dict())
    update_sd("model.diffusion_model.", unet_state_dict)

    # Convert the text encoder model
    if v2:
        make_dummy = ckpt_path is None  # 参照元のcheckpointがない場合は最後の層を前の層から複製して作るなどダミーの重みを入れる
        text_enc_dict = convert_text_encoder_state_dict_to_sd_v2(text_encoder.state_dict(), make_dummy)
        update_sd("cond_stage_model.model.", text_enc_dict)
    else:
        text_enc_dict = text_encoder.state_dict()
        update_sd("cond_stage_model.transformer.", text_enc_dict)

    # Convert the VAE
    if vae is not None:
        vae_dict = convert_vae_state_dict(vae.state_dict())
        update_sd("first_stage_model.", vae_dict)

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {'state_dict': state_dict}

    if 'epoch' in checkpoint:
        epochs += checkpoint['epoch']
    if 'global_step' in checkpoint:
        steps += checkpoint['global_step']

    new_ckpt['epoch'] = epochs
    new_ckpt['global_step'] = steps

    if is_safetensors(output_file):
        # TODO Tensor以外のdictの値を削除したほうがいいか
        save_file(state_dict, output_file)
    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(v2, output_dir, text_encoder, unet, pretrained_model_name_or_path, vae=None,
                              use_safetensors=False):
    if pretrained_model_name_or_path is None:
        # load default settings for v1/v2
        if v2:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V2
        else:
            pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_V1

    scheduler = DDIMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    pipeline = StableDiffusionPipeline(
        unet=unet,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=None,
    )
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)
