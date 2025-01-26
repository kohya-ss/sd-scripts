import torch, os
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
MODEL_VERSION_PIXART_SIGMA = "sigma"

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


def load_models_from_pixart_checkpoint(model_version, ckpt_path, base_resolution, enable_ar_condition, max_token_length, text_encoder_path, load_t5_in_4bit, vae_path, map_location, dtype=None):
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

    # DiT
    logger.info("building Diffusion Transformer")
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}
    with init_empty_weights():
        # as our img size is desired to be in [512, 1024, 2048], let's stop on the newer version
        dit = PixArtMS_XL_2(
            input_size=base_resolution // 8,
            pe_interpolation=pe_interpolation[base_resolution],
            micro_condition=enable_ar_condition,
            model_max_length=max_token_length,
        )

    # Text Encoders
    logger.info("building T5 text encoder")

    # Text Encoder is same to Stability AI's DF
    logger.info("loading T5 text encoders from huggingface")

    if not load_t5_in_4bit:
        print("WARNING: you did not specify loading T5 in 4 bit. If you are running on a consumer hardware, it can cost you a lot of VRAM usage without much quality gain")

    tokenizer = T5Tokenizer.from_pretrained(text_encoder_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(text_encoder_path, load_in_4bit=load_t5_in_4bit, subfolder="text_encoder").to(map_location)

    logger.info("Creating null embeds")

    null_caption_token = tokenizer("", max_length=max_token_length, padding="max_length", truncation=True, return_tensors="pt").to(map_location)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    logger.info("loading DiT from checkpoint")

    fill_in_pixart_checkpoint(state_dict, dit, map_location, null_caption_embs, dtype)

    logger.info("DiT Loaded")

    # prepare vae
    logger.info("loading VAE from checkpoint")
    
    vae = AutoencoderKL.from_pretrained(f"{vae_path}/vae").to(map_location).to(dtype)

    logger.info("VAE Loaded")

    ckpt_info = (epoch, global_step) if epoch is not None else None
    return text_encoder, vae, dit, ckpt_info

def fill_in_pixart_checkpoint(state_dict,
                    dit,
                    map_location,
                    null_caption_embs,
                    dtype
                    ):

    state_dict_keys = ['pos_embed', 'base_model.pos_embed', 'model.pos_embed']
    for key in state_dict_keys:
        if key in state_dict:
            del state_dict[key]
            break

    state_dict['y_embedder.y_embedding'] = null_caption_embs

    _load_state_dict_on_device(dit, state_dict, device=map_location, dtype=dtype)

def save_pixart_checkpoint(
    output_file,
    text_encoder,
    dit,
    epochs,
    steps,
    ckpt_info,
    vae,
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
    update_sd("", dit.state_dict())

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
