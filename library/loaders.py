import pathlib

from diffusers import UNet2DConditionModel, AutoencoderKL
from safetensors.torch import load_file
from transformers import CLIPTextConfig, CLIPTextModel

from .bootstrap import create_unet_diffusers_config, create_vae_diffusers_config
from .converters import (
    convert_ldm_unet_checkpoint,
    convert_ldm_vae_checkpoint,
    convert_ldm_clip_checkpoint_v2,
    convert_ldm_clip_checkpoint_v1,
)
from .dataset.common import KohyaException
from .model_utils2 import is_safetensors
import torch


def load_checkpoint_with_text_encoder_conversion(ckpt_path):
    # text encoderの格納形式が違うモデルに対応する ('text_model'がない)
    TEXT_ENCODER_KEY_REPLACEMENTS = [
        (
            "cond_stage_model.transformer.embeddings.",
            "cond_stage_model.transformer.text_model.embeddings.",
        ),
        (
            "cond_stage_model.transformer.encoder.",
            "cond_stage_model.transformer.text_model.encoder.",
        ),
        (
            "cond_stage_model.transformer.final_layer_norm.",
            "cond_stage_model.transformer.text_model.final_layer_norm.",
        ),
    ]

    if is_safetensors(ckpt_path):
        checkpoint = None
        state_dict = load_file(ckpt_path, "cpu")
    else:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
            checkpoint = None

    key_reps = []
    for rep_from, rep_to in TEXT_ENCODER_KEY_REPLACEMENTS:
        for key in state_dict.keys():
            if key.startswith(rep_from):
                new_key = rep_to + key[len(rep_from) :]
                key_reps.append((key, new_key))

    for key, new_key in key_reps:
        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    return checkpoint, state_dict


# TODO dtype指定の動作が怪しいので確認する text_encoderを指定形式で作れるか未確認
def load_models_from_stable_diffusion_checkpoint(v2, ckpt_path, dtype=None):
    _, state_dict = load_checkpoint_with_text_encoder_conversion(ckpt_path)
    if dtype is not None:
        for k, v in state_dict.items():
            if type(v) is torch.Tensor:
                state_dict[k] = v.to(dtype)

    # Convert the UNet2DConditionModel model.
    unet_config = create_unet_diffusers_config(v2)
    converted_unet_checkpoint = convert_ldm_unet_checkpoint(v2, state_dict, unet_config)

    unet = UNet2DConditionModel(**unet_config)
    info = unet.load_state_dict(converted_unet_checkpoint)
    print("loading u-net:", info)

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config()
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(state_dict, vae_config)

    vae = AutoencoderKL(**vae_config)
    info = vae.load_state_dict(converted_vae_checkpoint)
    print("loading vae:", info)

    # convert text_model
    if v2:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v2(
            state_dict, 77
        )
        cfg = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=23,
            num_attention_heads=16,
            max_position_embeddings=77,
            hidden_act="gelu",
            layer_norm_eps=1e-05,
            dropout=0.0,
            attention_dropout=0.0,
            initializer_range=0.02,
            initializer_factor=1.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
            model_type="clip_text_model",
            projection_dim=512,
            torch_dtype="float32",
            transformers_version="4.25.0.dev0",
        )
        text_model = CLIPTextModel._from_config(cfg)
        info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    else:
        converted_text_encoder_checkpoint = convert_ldm_clip_checkpoint_v1(state_dict)
        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        info = text_model.load_state_dict(converted_text_encoder_checkpoint)
    print("loading text encoder:", info)

    return text_model, vae, unet


VAE_PREFIX = "first_stage_model."


def load_vae(vae_id, dtype):
    vae_id = pathlib.Path(vae_id)
    if not vae_id.exists():
        raise KohyaException(f"VAE {vae_id} does not exist.")
    print(f"load VAE: {vae_id}")
    if vae_id.is_dir() or not vae_id.is_file():
        # Diffusers local/remote
        try:
            vae = AutoencoderKL.from_pretrained(
                vae_id, subfolder=None, torch_dtype=dtype
            )
        except EnvironmentError as e:
            print(f"exception occurs in loading vae: {e}")
            print("retry with subfolder='vae'")
            vae = AutoencoderKL.from_pretrained(
                vae_id, subfolder="vae", torch_dtype=dtype
            )
        return vae

    # local
    vae_config = create_vae_diffusers_config()

    if vae_id.suffix.endswith("bin"):
        # SD 1.5 VAE on Huggingface
        converted_vae_checkpoint = torch.load(vae_id, map_location="cpu")
    else:
        # StableDiffusion
        vae_model = (
            load_file(vae_id, "cpu")
            if is_safetensors(vae_id)
            else torch.load(vae_id, map_location="cpu")
        )
        vae_sd = vae_model["state_dict"] if "state_dict" in vae_model else vae_model

        # vae only or full model
        full_model = False
        for vae_key in vae_sd:
            if vae_key.startswith(VAE_PREFIX):
                full_model = True
                break
        if not full_model:
            sd = {}
            for key, value in vae_sd.items():
                sd[VAE_PREFIX + key] = value
            vae_sd = sd
            del sd

        # Convert the VAE model.
        converted_vae_checkpoint = convert_ldm_vae_checkpoint(vae_sd, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    return vae
