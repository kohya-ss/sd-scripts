import torch
from accelerate import init_empty_weights
from accelerate.utils.modeling import set_module_tensor_to_device
from safetensors.torch import load_file, save_file
from transformers import T5EncoderModel, T5Tokenizer
from typing import List
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from library import model_util
from library.nets.PixArtMS import PixArtMS_XL_2
from .utils import setup_logging
setup_logging()
import logging
logger = logging.getLogger(__name__)

VAE_SCALE_FACTOR = 0.13025
MODEL_VERSION_SDXL_BASE_V1_0 = "sdxl_base_v1-0"

# Diffusersの設定を読み込むための参照モデル
DIFFUSERS_REF_MODEL_ID_SDXL = "stabilityai/stable-diffusion-xl-base-1.0"


# load state_dict without allocating new tensors
def _load_state_dict_on_device(model, state_dict, device, dtype=None):
    # dtype will use fp32 as default
    missing_keys = list(model.state_dict().keys() - state_dict.keys())
    unexpected_keys = list(state_dict.keys() - model.state_dict().keys())

    # similar to model.load_state_dict()
    if not missing_keys and not unexpected_keys:
        for k in list(state_dict.keys()):
            set_module_tensor_to_device(model, k, device, value=state_dict.pop(k), dtype=dtype)
        return "<All keys matched successfully>"

    # error_msgs
    error_msgs: List[str] = []
    if missing_keys:
        error_msgs.insert(0, "Missing key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in missing_keys)))
    if unexpected_keys:
        error_msgs.insert(0, "Unexpected key(s) in state_dict: {}. ".format(", ".join('"{}"'.format(k) for k in unexpected_keys)))

    raise RuntimeError("Error(s) in loading state_dict for {}:\n\t{}".format(model.__class__.__name__, "\n\t".join(error_msgs)))


def load_models_from_pixart_checkpoint(model_version, ckpt_path, base_resolution, enable_ar_condition, max_token_length, text_encoder_path, vae_path, map_location, dtype=None):
    # model_version is reserved for future use
    # dtype is used for full_fp16/bf16 integration. Text Encoder will remain fp32, because it runs on CPU when caching

    # Load the state dict
    if model_util.is_safetensors(ckpt_path):
        checkpoint = None
        try:
            state_dict = load_file(ckpt_path, device=map_location)
        except:
            state_dict = load_file(ckpt_path)  # prevent device invalid Error
        epoch = None
        global_step = None
    else:
        checkpoint = torch.load(ckpt_path, map_location=map_location)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint.get("epoch", 0)
            global_step = checkpoint.get("global_step", 0)
        else:
            state_dict = checkpoint
            epoch = 0
            global_step = 0
        checkpoint = None
    
    if 'pos_embed' in state_dict:
        del state_dict['pos_embed']

    # DiT
    logger.info("building Diffusion Transformer")
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    with init_empty_weights():
        # as our img size is desired to be in [512, 1024, 2048], let's stop on MS version
        dit = PixArtMS_XL_2(
            input_size=base_resolution // 8,
            pe_interpolation=pe_interpolation[base_resolution],
            micro_condition=enable_ar_condition,
            model_max_length=max_token_length,
        )

    logger.info("loading DiT from checkpoint")

    info = _load_state_dict_on_device(dit, state_dict, device=map_location, dtype=dtype)
    logger.info(f"DiT: {info}")

    # Text Encoders
    logger.info("building T5 text encoder")

    # Text Encoder is same to Stability AI's DF
    logger.info("loading T5 text encoders from huggingface")

    tokenizer = T5Tokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, subfolder="text_encoder").to(map_location)

    null_caption_token = tokenizer("", max_length=max_token_length, padding="max_length", truncation=True, return_tensors="pt").to(map_location)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    # prepare vae
    logger.info("loading VAE from checkpoint")
    
    vae = AutoencoderKL.from_pretrained(f"{vae_path}/vae").to(map_location).to(dtype)

    logger.info(f"VAE: {info}")

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_encoder, vae, dit, ckpt_info


def make_unet_conversion_map():
    unet_conversion_map_layer = []

    for i in range(3):  # num_blocks is 3 in sdxl
        # loop over downblocks/upblocks
        for j in range(2):
            # loop over resnets/attentions for downblocks
            hf_down_res_prefix = f"down_blocks.{i}.resnets.{j}."
            sd_down_res_prefix = f"input_blocks.{3*i + j + 1}.0."
            unet_conversion_map_layer.append((sd_down_res_prefix, hf_down_res_prefix))

            if i < 3:
                # no attention layers in down_blocks.3
                hf_down_atn_prefix = f"down_blocks.{i}.attentions.{j}."
                sd_down_atn_prefix = f"input_blocks.{3*i + j + 1}.1."
                unet_conversion_map_layer.append((sd_down_atn_prefix, hf_down_atn_prefix))

        for j in range(3):
            # loop over resnets/attentions for upblocks
            hf_up_res_prefix = f"up_blocks.{i}.resnets.{j}."
            sd_up_res_prefix = f"output_blocks.{3*i + j}.0."
            unet_conversion_map_layer.append((sd_up_res_prefix, hf_up_res_prefix))

            # if i > 0: commentout for sdxl
            # no attention layers in up_blocks.0
            hf_up_atn_prefix = f"up_blocks.{i}.attentions.{j}."
            sd_up_atn_prefix = f"output_blocks.{3*i + j}.1."
            unet_conversion_map_layer.append((sd_up_atn_prefix, hf_up_atn_prefix))

        if i < 3:
            # no downsample in down_blocks.3
            hf_downsample_prefix = f"down_blocks.{i}.downsamplers.0.conv."
            sd_downsample_prefix = f"input_blocks.{3*(i+1)}.0.op."
            unet_conversion_map_layer.append((sd_downsample_prefix, hf_downsample_prefix))

            # no upsample in up_blocks.3
            hf_upsample_prefix = f"up_blocks.{i}.upsamplers.0."
            sd_upsample_prefix = f"output_blocks.{3*i + 2}.{2}."  # change for sdxl
            unet_conversion_map_layer.append((sd_upsample_prefix, hf_upsample_prefix))

    hf_mid_atn_prefix = "mid_block.attentions.0."
    sd_mid_atn_prefix = "middle_block.1."
    unet_conversion_map_layer.append((sd_mid_atn_prefix, hf_mid_atn_prefix))

    for j in range(2):
        hf_mid_res_prefix = f"mid_block.resnets.{j}."
        sd_mid_res_prefix = f"middle_block.{2*j}."
        unet_conversion_map_layer.append((sd_mid_res_prefix, hf_mid_res_prefix))

    unet_conversion_map_resnet = [
        # (stable-diffusion, HF Diffusers)
        ("in_layers.0.", "norm1."),
        ("in_layers.2.", "conv1."),
        ("out_layers.0.", "norm2."),
        ("out_layers.3.", "conv2."),
        ("emb_layers.1.", "time_emb_proj."),
        ("skip_connection.", "conv_shortcut."),
    ]

    unet_conversion_map = []
    for sd, hf in unet_conversion_map_layer:
        if "resnets" in hf:
            for sd_res, hf_res in unet_conversion_map_resnet:
                unet_conversion_map.append((sd + sd_res, hf + hf_res))
        else:
            unet_conversion_map.append((sd, hf))

    for j in range(2):
        hf_time_embed_prefix = f"time_embedding.linear_{j+1}."
        sd_time_embed_prefix = f"time_embed.{j*2}."
        unet_conversion_map.append((sd_time_embed_prefix, hf_time_embed_prefix))

    for j in range(2):
        hf_label_embed_prefix = f"add_embedding.linear_{j+1}."
        sd_label_embed_prefix = f"label_emb.0.{j*2}."
        unet_conversion_map.append((sd_label_embed_prefix, hf_label_embed_prefix))

    unet_conversion_map.append(("input_blocks.0.0.", "conv_in."))
    unet_conversion_map.append(("out.0.", "conv_norm_out."))
    unet_conversion_map.append(("out.2.", "conv_out."))

    return unet_conversion_map


def convert_diffusers_unet_state_dict_to_sdxl(du_sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_map = {hf: sd for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(du_sd, conversion_map)


def convert_unet_state_dict(src_sd, conversion_map):
    converted_sd = {}
    for src_key, value in src_sd.items():
        # さすがに全部回すのは時間がかかるので右から要素を削りつつprefixを探す
        src_key_fragments = src_key.split(".")[:-1]  # remove weight/bias
        while len(src_key_fragments) > 0:
            src_key_prefix = ".".join(src_key_fragments) + "."
            if src_key_prefix in conversion_map:
                converted_prefix = conversion_map[src_key_prefix]
                converted_key = converted_prefix + src_key[len(src_key_prefix) :]
                converted_sd[converted_key] = value
                break
            src_key_fragments.pop(-1)
        assert len(src_key_fragments) > 0, f"key {src_key} not found in conversion map"

    return converted_sd


def convert_sdxl_unet_state_dict_to_diffusers(sd):
    unet_conversion_map = make_unet_conversion_map()

    conversion_dict = {sd: hf for sd, hf in unet_conversion_map}
    return convert_unet_state_dict(sd, conversion_dict)


def convert_text_encoder_2_state_dict_to_sdxl(checkpoint, logit_scale):
    def convert_key(key):
        # position_idsの除去
        if ".position_ids" in key:
            return None

        # common
        key = key.replace("text_model.encoder.", "transformer.")
        key = key.replace("text_model.", "")
        if "layers" in key:
            # resblocks conversion
            key = key.replace(".layers.", ".resblocks.")
            if ".layer_norm" in key:
                key = key.replace(".layer_norm", ".ln_")
            elif ".mlp." in key:
                key = key.replace(".fc1.", ".c_fc.")
                key = key.replace(".fc2.", ".c_proj.")
            elif ".self_attn.out_proj" in key:
                key = key.replace(".self_attn.out_proj.", ".attn.out_proj.")
            elif ".self_attn." in key:
                key = None  # 特殊なので後で処理する
            else:
                raise ValueError(f"unexpected key in DiffUsers model: {key}")
        elif ".position_embedding" in key:
            key = key.replace("embeddings.position_embedding.weight", "positional_embedding")
        elif ".token_embedding" in key:
            key = key.replace("embeddings.token_embedding.weight", "token_embedding.weight")
        elif "text_projection" in key:  # no dot in key
            key = key.replace("text_projection.weight", "text_projection")
        elif "final_layer_norm" in key:
            key = key.replace("final_layer_norm", "ln_final")
        return key

    keys = list(checkpoint.keys())
    new_sd = {}
    for key in keys:
        new_key = convert_key(key)
        if new_key is None:
            continue
        new_sd[new_key] = checkpoint[key]

    # attnの変換
    for key in keys:
        if "layers" in key and "q_proj" in key:
            # 三つを結合
            key_q = key
            key_k = key.replace("q_proj", "k_proj")
            key_v = key.replace("q_proj", "v_proj")

            value_q = checkpoint[key_q]
            value_k = checkpoint[key_k]
            value_v = checkpoint[key_v]
            value = torch.cat([value_q, value_k, value_v])

            new_key = key.replace("text_model.encoder.layers.", "transformer.resblocks.")
            new_key = new_key.replace(".self_attn.q_proj.", ".attn.in_proj_")
            new_sd[new_key] = value

    if logit_scale is not None:
        new_sd["logit_scale"] = logit_scale

    return new_sd


def save_stable_diffusion_checkpoint(
    output_file,
    text_encoder1,
    text_encoder2,
    unet,
    epochs,
    steps,
    ckpt_info,
    vae,
    logit_scale,
    metadata,
    save_dtype=None,
):
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    # Convert the UNet model
    update_sd("model.diffusion_model.", unet.state_dict())

    # Convert the text encoders
    update_sd("conditioner.embedders.0.transformer.", text_encoder1.state_dict())

    text_enc2_dict = convert_text_encoder_2_state_dict_to_sdxl(text_encoder2.state_dict(), logit_scale)
    update_sd("conditioner.embedders.1.model.", text_enc2_dict)

    # Convert the VAE
    vae_dict = model_util.convert_vae_state_dict(vae.state_dict())
    update_sd("first_stage_model.", vae_dict)

    # Put together new checkpoint
    key_count = len(state_dict.keys())
    new_ckpt = {"state_dict": state_dict}

    # epoch and global_step are sometimes not int
    if ckpt_info is not None:
        epochs += ckpt_info[0]
        steps += ckpt_info[1]

    new_ckpt["epoch"] = epochs
    new_ckpt["global_step"] = steps

    if model_util.is_safetensors(output_file):
        save_file(state_dict, output_file, metadata)
    else:
        torch.save(new_ckpt, output_file)

    return key_count


def save_diffusers_checkpoint(
    output_dir, text_encoder1, text_encoder2, unet, pretrained_model_name_or_path, vae=None, use_safetensors=False, save_dtype=None
):
    from diffusers import StableDiffusionXLPipeline

    # convert U-Net
    unet_sd = unet.state_dict()
    du_unet_sd = convert_sdxl_unet_state_dict_to_diffusers(unet_sd)

    diffusers_unet = UNet2DConditionModel(**DIFFUSERS_SDXL_UNET_CONFIG)
    if save_dtype is not None:
        diffusers_unet.to(save_dtype)
    diffusers_unet.load_state_dict(du_unet_sd)

    # create pipeline to save
    if pretrained_model_name_or_path is None:
        pretrained_model_name_or_path = DIFFUSERS_REF_MODEL_ID_SDXL

    scheduler = EulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer1 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer2 = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2")
    if vae is None:
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # prevent local path from being saved
    def remove_name_or_path(model):
        if hasattr(model, "config"):
            model.config._name_or_path = None
            model.config._name_or_path = None

    remove_name_or_path(diffusers_unet)
    remove_name_or_path(text_encoder1)
    remove_name_or_path(text_encoder2)
    remove_name_or_path(scheduler)
    remove_name_or_path(tokenizer1)
    remove_name_or_path(tokenizer2)
    remove_name_or_path(vae)

    pipeline = StableDiffusionXLPipeline(
        unet=diffusers_unet,
        text_encoder=text_encoder1,
        text_encoder_2=text_encoder2,
        vae=vae,
        scheduler=scheduler,
        tokenizer=tokenizer1,
        tokenizer_2=tokenizer2,
    )
    if save_dtype is not None:
        pipeline.to(None, save_dtype)
    pipeline.save_pretrained(output_dir, safe_serialization=use_safetensors)
