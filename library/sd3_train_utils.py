import argparse
import math
import os
import toml
import json
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import save_file
from accelerate import Accelerator, PartialState
from tqdm import tqdm
from PIL import Image

from library import sd3_models, sd3_utils, strategy_base, train_util
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

# from transformers import CLIPTokenizer
# from library import model_util
# , sdxl_model_util, train_util, sdxl_original_unet
# from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
from .utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from .sdxl_train_util import match_mixed_precision


def load_target_model(
    model_type: str,
    args: argparse.Namespace,
    state_dict: dict,
    accelerator: Accelerator,
    attn_mode: str,
    model_dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> Union[
    sd3_models.MMDiT,
    Optional[sd3_models.SDClipModel],
    Optional[sd3_models.SDXLClipG],
    Optional[sd3_models.T5XXLModel],
    sd3_models.SDVAE,
]:
    loading_device = device if device is not None else (accelerator.device if args.lowram else "cpu")

    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            if model_type == "mmdit":
                model = sd3_utils.load_mmdit(state_dict, attn_mode, model_dtype, loading_device)
            elif model_type == "clip_l":
                model = sd3_utils.load_clip_l(state_dict, args.clip_l, attn_mode, model_dtype, loading_device)
            elif model_type == "clip_g":
                model = sd3_utils.load_clip_g(state_dict, args.clip_g, attn_mode, model_dtype, loading_device)
            elif model_type == "t5xxl":
                model = sd3_utils.load_t5xxl(state_dict, args.t5xxl, attn_mode, model_dtype, loading_device)
            elif model_type == "vae":
                model = sd3_utils.load_vae(state_dict, args.vae, model_dtype, loading_device)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            # work on low-ram device: models are already loaded on accelerator.device, but we ensure they are on device
            if args.lowram:
                model = model.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    return model


def save_models(
    ckpt_path: str,
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel],
    sai_metadata: Optional[dict],
    save_dtype: Optional[torch.dtype] = None,
):
    r"""
    Save models to checkpoint file. Only supports unified checkpoint format.
    """

    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    update_sd("model.diffusion_model.", mmdit.state_dict())
    update_sd("first_stage_model.", vae.state_dict())

    if clip_l is not None:
        update_sd("text_encoders.clip_l.", clip_l.state_dict())
    if clip_g is not None:
        update_sd("text_encoders.clip_g.", clip_g.state_dict())
    if t5xxl is not None:
        update_sd("text_encoders.t5xxl.", t5xxl.state_dict())

    save_file(state_dict, ckpt_path, metadata=sai_metadata)


def save_sd3_model_on_train_end(
    args: argparse.Namespace,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel],
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, sd3=mmdit.model_type
        )
        save_models(ckpt_file, mmdit, vae, clip_l, clip_g, t5xxl, sai_metadata, save_dtype)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd3_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    clip_l: sd3_models.SDClipModel,
    clip_g: sd3_models.SDXLClipG,
    t5xxl: Optional[sd3_models.T5XXLModel],
    mmdit: sd3_models.MMDiT,
    vae: sd3_models.SDVAE,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, sd3=mmdit.model_type
        )
        save_models(ckpt_file, mmdit, vae, clip_l, clip_g, t5xxl, sai_metadata, save_dtype)

    train_util.save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        True,
        True,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        None,
    )


def add_sd3_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)"
        + " / text encoderのバッチサイズ（デフォルト: None, データセットのバッチサイズを使用）",
    )
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )

    parser.add_argument(
        "--clip_l",
        type=str,
        required=False,
        help="CLIP-L model path. if not specified, use ckpt's state_dict / CLIP-Lモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--clip_g",
        type=str,
        required=False,
        help="CLIP-G model path. if not specified, use ckpt's state_dict / CLIP-Gモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--t5xxl",
        type=str,
        required=False,
        help="T5-XXL model path. if not specified, use ckpt's state_dict / T5-XXLモデルのパス。指定しない場合はckptのstate_dictを使用",
    )
    parser.add_argument(
        "--save_clip", action="store_true", help="save CLIP models to checkpoint / CLIPモデルをチェックポイントに保存する"
    )
    parser.add_argument(
        "--save_t5xxl", action="store_true", help="save T5-XXL model to checkpoint / T5-XXLモデルをチェックポイントに保存する"
    )

    parser.add_argument(
        "--t5xxl_device",
        type=str,
        default=None,
        help="T5-XXL device. if not specified, use accelerator's device / T5-XXLデバイス。指定しない場合はacceleratorのデバイスを使用",
    )
    parser.add_argument(
        "--t5xxl_dtype",
        type=str,
        default=None,
        help="T5-XXL dtype. if not specified, use default dtype (from mixed precision) / T5-XXL dtype。指定しない場合はデフォルトのdtype（mixed precisionから）を使用",
    )

    # copy from Diffusers
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument("--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme.")
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )


def verify_sdxl_training_args(args: argparse.Namespace, supportTextEncoderCaching: bool = True):
    assert not args.v2, "v2 cannot be enabled in SDXL training / SDXL学習ではv2を有効にすることはできません"
    if args.v_parameterization:
        logger.warning("v_parameterization will be unexpected / SDXL学習ではv_parameterizationは想定外の動作になります")

    if args.clip_skip is not None:
        logger.warning("clip_skip will be unexpected / SDXL学習ではclip_skipは動作しません")

    # if args.multires_noise_iterations:
    #     logger.info(
    #         f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET}, but noise_offset is disabled due to multires_noise_iterations / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されていますが、multires_noise_iterationsが有効になっているためnoise_offsetは無効になります"
    #     )
    # else:
    #     if args.noise_offset is None:
    #         args.noise_offset = DEFAULT_NOISE_OFFSET
    #     elif args.noise_offset != DEFAULT_NOISE_OFFSET:
    #         logger.info(
    #             f"Warning: SDXL has been trained with noise_offset={DEFAULT_NOISE_OFFSET} / SDXLはnoise_offset={DEFAULT_NOISE_OFFSET}で学習されています"
    #         )
    #     logger.info(f"noise_offset is set to {args.noise_offset} / noise_offsetが{args.noise_offset}に設定されました")

    assert (
        not hasattr(args, "weighted_captions") or not args.weighted_captions
    ), "weighted_captions cannot be enabled in SDXL training currently / SDXL学習では今のところweighted_captionsを有効にすることはできません"

    if supportTextEncoderCaching:
        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True
            logger.warning(
                "cache_text_encoder_outputs is enabled because cache_text_encoder_outputs_to_disk is enabled / "
                + "cache_text_encoder_outputs_to_diskが有効になっているためcache_text_encoder_outputsが有効になりました"
            )


# temporary copied from sd3_minimal_inferece.py


def get_sigmas(sampling: sd3_utils.ModelSamplingDiscreteFlow, steps):
    start = sampling.timestep(sampling.sigma_max)
    end = sampling.timestep(sampling.sigma_min)
    timesteps = torch.linspace(start, end, steps)
    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(sampling.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)


def max_denoise(model_sampling, sigmas):
    max_sigma = float(model_sampling.sigma_max)
    sigma = float(sigmas[0])
    return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma


def do_sample(
    height: int,
    width: int,
    seed: int,
    cond: Tuple[torch.Tensor, torch.Tensor],
    neg_cond: Tuple[torch.Tensor, torch.Tensor],
    mmdit: sd3_models.MMDiT,
    steps: int,
    guidance_scale: float,
    dtype: torch.dtype,
    device: str,
):
    latent = torch.zeros(1, 16, height // 8, width // 8, device=device)
    latent = latent.to(dtype).to(device)

    # noise = get_noise(seed, latent).to(device)
    if seed is not None:
        generator = torch.manual_seed(seed)
    noise = (
        torch.randn(latent.size(), dtype=torch.float32, layout=latent.layout, generator=generator, device="cpu")
        .to(latent.dtype)
        .to(device)
    )

    model_sampling = sd3_utils.ModelSamplingDiscreteFlow(shift=3.0)  # 3.0 is for SD3

    sigmas = get_sigmas(model_sampling, steps).to(device)

    noise_scaled = model_sampling.noise_scaling(sigmas[0], noise, latent, max_denoise(model_sampling, sigmas))

    c_crossattn = torch.cat([cond[0], neg_cond[0]]).to(device).to(dtype)
    y = torch.cat([cond[1], neg_cond[1]]).to(device).to(dtype)

    x = noise_scaled.to(device).to(dtype)
    # print(x.shape)

    with torch.no_grad():
        for i in tqdm(range(len(sigmas) - 1)):
            sigma_hat = sigmas[i]

            timestep = model_sampling.timestep(sigma_hat).float()
            timestep = torch.FloatTensor([timestep, timestep]).to(device)

            x_c_nc = torch.cat([x, x], dim=0)
            # print(x_c_nc.shape, timestep.shape, c_crossattn.shape, y.shape)

            model_output = mmdit(x_c_nc, timestep, context=c_crossattn, y=y)
            model_output = model_output.float()
            batched = model_sampling.calculate_denoised(sigma_hat, model_output, x)

            pos_out, neg_out = batched.chunk(2)
            denoised = neg_out + (pos_out - neg_out) * guidance_scale
            # print(denoised.shape)

            # d = to_d(x, sigma_hat, denoised)
            dims_to_append = x.ndim - sigma_hat.ndim
            sigma_hat_dims = sigma_hat[(...,) + (None,) * dims_to_append]
            # print(dims_to_append, x.shape, sigma_hat.shape, denoised.shape, sigma_hat_dims.shape)
            """Converts a denoiser output to a Karras ODE derivative."""
            d = (x - denoised) / sigma_hat_dims

            dt = sigmas[i + 1] - sigma_hat

            # Euler method
            x = x + d * dt
            x = x.to(dtype)

    return x


def load_prompts(prompt_file: str) -> List[Dict]:
    # read prompts
    if prompt_file.endswith(".txt"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        prompts = [line.strip() for line in lines if len(line.strip()) > 0 and line[0] != "#"]
    elif prompt_file.endswith(".toml"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            data = toml.load(f)
        prompts = [dict(**data["prompt"], **subset) for subset in data["prompt"]["subset"]]
    elif prompt_file.endswith(".json"):
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    # preprocess prompts
    for i in range(len(prompts)):
        prompt_dict = prompts[i]
        if isinstance(prompt_dict, str):
            from library.train_util import line_to_prompt_dict

            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    mmdit,
    vae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap unet and text_encoder(s)
    mmdit = accelerator.unwrap_model(mmdit)
    text_encoders = [accelerator.unwrap_model(te) for te in text_encoders]
    # print([(te.parameters().__next__().device if te is not None else None) for te in text_encoders])

    prompts = load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    org_vae_device = vae.device  # will be on cpu
    vae.to(distributed_state.device)  # distributed_state.device is same as accelerator.device

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    mmdit,
                    text_encoders,
                    vae,
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator,
                        args,
                        mmdit,
                        text_encoders,
                        vae,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    vae.to(org_vae_device)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    mmdit: sd3_models.MMDiT,
    text_encoders: List[Union[sd3_models.SDClipModel, sd3_models.SDXLClipG, sd3_models.T5XXLModel]],
    vae: sd3_models.SDVAE,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    # controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    # sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        torch.cuda.seed()

    if negative_prompt is None:
        negative_prompt = ""

    height = max(64, height - height % 8)  # round to divisible by 8
    width = max(64, width - width % 8)  # round to divisible by 8
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    # logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # encode prompts
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    if sample_prompts_te_outputs and prompt in sample_prompts_te_outputs:
        te_outputs = sample_prompts_te_outputs[prompt]
    else:
        l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(prompt)
        te_outputs = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, [l_tokens, g_tokens, t5_tokens])

    lg_out, t5_out, pooled = te_outputs
    cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

    # encode negative prompts
    if sample_prompts_te_outputs and negative_prompt in sample_prompts_te_outputs:
        neg_te_outputs = sample_prompts_te_outputs[negative_prompt]
    else:
        l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(negative_prompt)
        neg_te_outputs = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, [l_tokens, g_tokens, t5_tokens])

    lg_out, t5_out, pooled = neg_te_outputs
    neg_cond = encoding_strategy.concat_encodings(lg_out, t5_out, pooled)

    # sample image
    latents = do_sample(height, width, seed, cond, neg_cond, mmdit, sample_steps, scale, mmdit.dtype, accelerator.device)
    latents = vae.process_out(latents.to(vae.device, dtype=vae.dtype))

    # latent to image
    with torch.no_grad():
        image = vae.decode(latents)
    image = image.float()
    image = torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)[0]
    decoded_np = 255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)
    decoded_np = decoded_np.astype(np.uint8)

    image = Image.fromarray(decoded_np)
    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log(
            {f"sample_{i}": wandb.Image(
                image,
                caption=prompt # positive prompt as a caption
            )}, 
            commit=False
        )


# region Diffusers


from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import BaseOutput


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


class FlowMatchEulerDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        timesteps = np.linspace(1, num_train_timesteps, num_train_timesteps, dtype=np.float32)[::-1].copy()
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32)

        sigmas = timesteps / num_train_timesteps
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.timesteps = sigmas * num_train_timesteps

        self._step_index = None
        self._begin_index = None

        self.sigmas = sigmas.to("cpu")  # to avoid too much CPU/GPU communication
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward process in flow-matching

        Args:
            sample (`torch.FloatTensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `torch.FloatTensor`:
                A scaled input sample.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        sigma = self.sigmas[self.step_index]
        sample = sigma * noise + (1.0 - sigma) * sample

        return sample

    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        timesteps = np.linspace(self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps)

        sigmas = timesteps / self.config.num_train_timesteps
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)

        timesteps = sigmas * self.config.num_train_timesteps
        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or
                tuple.

        Returns:
            [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_euler_discrete.EulerDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """

        if isinstance(timestep, int) or isinstance(timestep, torch.IntTensor) or isinstance(timestep, torch.LongTensor):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.sigmas[self.step_index]

        gamma = min(s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

        noise = randn_tensor(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)

        eps = noise * s_noise
        sigma_hat = sigma * (gamma + 1)

        if gamma > 0:
            sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        # NOTE: "original_sample" should not be an expected prediction_type but is left in for
        # backwards compatibility

        # if self.config.prediction_type == "vector_field":

        denoised = sample - model_output * sigma
        # 2. Convert to an ODE derivative
        derivative = (sample - denoised) / sigma_hat

        dt = self.sigmas[self.step_index + 1] - sigma_hat

        prev_sample = sample + derivative * dt
        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)

    def __len__(self):
        return self.config.num_train_timesteps


# endregion
