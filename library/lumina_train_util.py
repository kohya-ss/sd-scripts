import argparse
import math
import os
import numpy as np
import time
from typing import Callable, Dict, List, Optional, Tuple, Any

import torch
from torch import Tensor
from accelerate import Accelerator, PartialState
from transformers import Gemma2Model
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file

from library import lumina_models, lumina_util, strategy_base, strategy_lumina, train_util
from library.device_utils import init_ipex, clean_memory_on_device
from library.sd3_train_utils import FlowMatchEulerDiscreteScheduler

init_ipex()

from .utils import setup_logging, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region sample images


@torch.no_grad()
def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    global_step: int,
    nextdit: lumina_models.NextDiT,
    vae: torch.nn.Module,
    gemma2_model: Gemma2Model,
    sample_prompts_gemma2_outputs: List[Tuple[Tensor, Tensor, Tensor]],
    prompt_replacement: Optional[Tuple[str, str]] = None,
    controlnet=None,
):
    """
    Generate sample images using the NextDiT model.

    Args:
        accelerator (Accelerator): Accelerator instance.
        args (argparse.Namespace): Command-line arguments.
        epoch (int): Current epoch number.
        global_step (int): Current global step number.
        nextdit (lumina_models.NextDiT): The NextDiT model instance.
        vae (torch.nn.Module): The VAE module.
        gemma2_model (Gemma2Model): The Gemma2 model instance.
        sample_prompts_gemma2_outputs (List[Tuple[Tensor, Tensor, Tensor]]): List of tuples containing the encoded prompts, text masks, and timestep for each sample.
        prompt_replacement (Optional[Tuple[str, str]], optional): Tuple containing the prompt and negative prompt replacements. Defaults to None.
        controlnet:: ControlNet model

    Returns:
        None
    """
    if global_step == 0:
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
            if global_step % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    assert (
        args.sample_prompts is not None
    ), "No sample prompts found. Provide `--sample_prompts` / サンプルプロンプトが見つかりません。`--sample_prompts` を指定してください"

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {global_step}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_gemma2_outputs is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap nextdit and gemma2_model
    nextdit = accelerator.unwrap_model(nextdit)
    if gemma2_model is not None:
        gemma2_model = accelerator.unwrap_model(gemma2_model)
    # if controlnet is not None:
    #     controlnet = accelerator.unwrap_model(controlnet)
    # print([(te.parameters().__next__().device if te is not None else None) for te in text_encoders])

    prompts = train_util.load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    # save random state to restore later
    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        # If only one device is available, just use the original prompt list. We don't need to care about the distribution of prompts.
        for prompt_dict in prompts:
            sample_image_inference(
                accelerator,
                args,
                nextdit,
                gemma2_model,
                vae,
                save_dir,
                prompt_dict,
                epoch,
                global_step,
                sample_prompts_gemma2_outputs,
                prompt_replacement,
                controlnet,
            )
    else:
        # Creating list with N elements, where each element is a list of prompt_dicts, and N is the number of processes available (number of devices available)
        # prompt_dicts are assigned to lists based on order of processes, to attempt to time the image creation time to match enum order. Probably only works when steps and sampler are identical.
        per_process_prompts = []  # list of lists
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
            for prompt_dict in prompt_dict_lists[0]:
                sample_image_inference(
                    accelerator,
                    args,
                    nextdit,
                    gemma2_model,
                    vae,
                    save_dir,
                    prompt_dict,
                    epoch,
                    global_step,
                    sample_prompts_gemma2_outputs,
                    prompt_replacement,
                    controlnet,
                )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


@torch.no_grad()
def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    nextdit: lumina_models.NextDiT,
    gemma2_model: Gemma2Model,
    vae: torch.nn.Module,
    save_dir: str,
    prompt_dict: Dict[str, str],
    epoch: int,
    global_step: int,
    sample_prompts_gemma2_outputs: List[Tuple[Tensor, Tensor, Tensor]],
    prompt_replacement: Optional[Tuple[str, str]] = None,
    controlnet=None,
):
    """
    Generates sample images

    Args:
        accelerator (Accelerator): Accelerator object
        args (argparse.Namespace): Arguments object
        nextdit (lumina_models.NextDiT): NextDiT model
        gemma2_model (Gemma2Model): Gemma2 model
        vae (torch.nn.Module): VAE model
        save_dir (str): Directory to save images
        prompt_dict (Dict[str, str]): Prompt dictionary
        epoch (int): Epoch number
        steps (int): Number of steps to run
        sample_prompts_gemma2_outputs (List[Tuple[Tensor, Tensor, Tensor]]): List of tuples containing gemma2 outputs
        prompt_replacement (Optional[Tuple[str, str]], optional): Replacement for positive and negative prompt. Defaults to None.

    Returns:
        None
    """
    assert isinstance(prompt_dict, dict)
    # negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 38)
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    guidance_scale: int = prompt_dict.get("scale", 3.5)
    seed: int = prompt_dict.get("seed", None)
    controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    negative_prompt: str = prompt_dict.get("negative_prompt", "")
    # sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator.manual_seed(seed)

    # if negative_prompt is None:
    #     negative_prompt = ""
    height = max(64, height - height % 16)  # round to divisible by 16
    width = max(64, width - width % 16)  # round to divisible by 16
    logger.info(f"prompt: {prompt}")
    # logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {guidance_scale}")
    # logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # encode prompts
    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    assert isinstance(tokenize_strategy, strategy_lumina.LuminaTokenizeStrategy)
    assert isinstance(encoding_strategy, strategy_lumina.LuminaTextEncodingStrategy)

    gemma2_conds = []
    if sample_prompts_gemma2_outputs and prompt in sample_prompts_gemma2_outputs:
        gemma2_conds = sample_prompts_gemma2_outputs[prompt]
        logger.info(f"Using cached Gemma2 outputs for prompt: {prompt}")
    if gemma2_model is not None:
        logger.info(f"Encoding prompt with Gemma2: {prompt}")
        tokens_and_masks = tokenize_strategy.tokenize(prompt)
        encoded_gemma2_conds = encoding_strategy.encode_tokens(tokenize_strategy, [gemma2_model], tokens_and_masks)

        # if gemma2_conds is not cached, use encoded_gemma2_conds
        if len(gemma2_conds) == 0:
            gemma2_conds = encoded_gemma2_conds
        else:
            # if encoded_gemma2_conds is not None, update cached gemma2_conds
            for i in range(len(encoded_gemma2_conds)):
                if encoded_gemma2_conds[i] is not None:
                    gemma2_conds[i] = encoded_gemma2_conds[i]

    # Unpack Gemma2 outputs
    gemma2_hidden_states, input_ids, gemma2_attn_mask = gemma2_conds

    # sample image
    weight_dtype = vae.dtype  # TOFO give dtype as argument
    latent_height = height // 8
    latent_width = width // 8
    noise = torch.randn(
        1,
        16,
        latent_height,
        latent_width,
        device=accelerator.device,
        dtype=weight_dtype,
        generator=generator,
    )
    # Prompts are paired positive/negative
    noise = noise.repeat(gemma2_attn_mask.shape[0], 1, 1, 1)

    timesteps = get_schedule(sample_steps, noise.shape[1], shift=True)
    # img_ids = lumina_util.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(accelerator.device, weight_dtype)
    gemma2_attn_mask = gemma2_attn_mask.to(accelerator.device)

    # if controlnet_image is not None:
    #     controlnet_image = Image.open(controlnet_image).convert("RGB")
    #     controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)
    #     controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
    #     controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(weight_dtype).to(accelerator.device)

    with accelerator.autocast():
        x = denoise(nextdit, noise, gemma2_hidden_states, gemma2_attn_mask, timesteps=timesteps, guidance=guidance_scale)

    # x = lumina_util.unpack_latents(x, packed_latent_height, packed_latent_width)

    # latent to image
    clean_memory_on_device(accelerator.device)
    org_vae_device = vae.device  # will be on cpu
    vae.to(accelerator.device)  # distributed_state.device is same as accelerator.device
    with accelerator.autocast():
        x = vae.decode(x)
    vae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    x = x.clamp(-1, 1)
    x = x.permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{global_step:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = int(prompt_dict.get("enum", 0))
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        import wandb

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)  # positive prompt as a caption


def time_shift(mu: float, sigma: float, t: Tensor):
    """
    Get time shift

    Args:
        mu (float): mu value.
        sigma (float): sigma value.
        t (Tensor): timestep.

    Return:
        float: time shift
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    """
    Get linear function

    Args:
        x1 (float, optional): x1 value. Defaults to 256.
        y1 (float, optional): y1 value. Defaults to 0.5.
        x2 (float, optional): x2 value. Defaults to 4096.
        y2 (float, optional): y2 value. Defaults to 1.15.

    Return:
        Callable[[float], float]: linear function
    """
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    """
    Get timesteps schedule

    Args:
        num_steps (int): Number of steps in the schedule.
        image_seq_len (int): Sequence length of the image.
        base_shift (float, optional): Base shift value. Defaults to 0.5.
        max_shift (float, optional): Maximum shift value. Defaults to 1.15.
        shift (bool, optional): Whether to shift the schedule. Defaults to True.

    Return:
        List[float]: timesteps schedule
    """
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: lumina_models.NextDiT, img: Tensor, txt: Tensor, txt_mask: Tensor, timesteps: List[float], guidance: float = 4.0
):
    """
    Denoise an image using the NextDiT model.

    Args:
        model (lumina_models.NextDiT): The NextDiT model instance.
        img (Tensor): The input image tensor.
        txt (Tensor): The input text tensor.
        txt_mask (Tensor): The input text mask tensor.
        timesteps (List[float]): A list of timesteps for the denoising process.
        guidance (float, optional): The guidance scale for the denoising process. Defaults to 4.0.

    Returns:
        img (Tensor): Denoised tensor
    """
    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # model.prepare_block_swap_before_forward()
        # block_samples = None
        # block_single_samples = None
        pred = model.forward_with_cfg(
            x=img,  # image latents (B, C, H, W)
            t=t_vec / 1000,  # timesteps需要除以1000来匹配模型预期
            cap_feats=txt,  # Gemma2的hidden states作为caption features
            cap_mask=txt_mask.to(dtype=torch.int32),  # Gemma2的attention mask
            cfg_scale=guidance,
        )

        img = img + (t_prev - t_curr) * pred

    # model.prepare_block_swap_before_forward()
    return img


# endregion


# region train
def get_sigmas(
    noise_scheduler: FlowMatchEulerDiscreteScheduler, timesteps: Tensor, device: torch.device, n_dim=4, dtype=torch.float32
) -> Tensor:
    """
    Get sigmas for timesteps

    Args:
        noise_scheduler (FlowMatchEulerDiscreteScheduler): The noise scheduler instance.
        timesteps (Tensor): A tensor of timesteps for the denoising process.
        device (torch.device): The device on which the tensors are stored.
        n_dim (int, optional): The number of dimensions for the output tensor. Defaults to 4.
        dtype (torch.dtype, optional): The data type for the output tensor. Defaults to torch.float32.

    Returns:
        sigmas (Tensor): The sigmas tensor.
    """
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.

    Args:
        weighting_scheme (str): The weighting scheme to use.
        batch_size (int): The batch size for the sampling process.
        logit_mean (float, optional): The mean of the logit distribution. Defaults to None.
        logit_std (float, optional): The standard deviation of the logit distribution. Defaults to None.
        mode_scale (float, optional): The mode scale for the mode weighting scheme. Defaults to None.

    Returns:
        u (Tensor): The sampled timesteps.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(size=(batch_size,), device="cpu")
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None) -> Tensor:
    """Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.

    Args:
        weighting_scheme (str): The weighting scheme to use.
        sigmas (Tensor, optional): The sigmas tensor. Defaults to None.

    Returns:
        u (Tensor): The sampled timesteps.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = torch.ones_like(sigmas)
    return weighting


def get_noisy_model_input_and_timesteps(args, noise_scheduler, latents, noise, device, dtype) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Get noisy model input and timesteps.

    Args:
        args (argparse.Namespace): Arguments.
        noise_scheduler (noise_scheduler): Noise scheduler.
        latents (Tensor): Latents.
        noise (Tensor): Latent noise.
        device (torch.device): Device.
        dtype (torch.dtype): Data type

    Return:
        Tuple[Tensor, Tensor, Tensor]:
            noisy model input
            timesteps
            sigmas
    """
    bsz, _, h, w = latents.shape
    sigmas = None

    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        # Simple random t-based noise sampling
        if args.timestep_sampling == "sigmoid":
            # https://github.com/XLabs-AI/x-flux/tree/main
            t = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device))
        else:
            t = torch.rand((bsz,), device=device)

        timesteps = t * 1000.0
        t = t.view(-1, 1, 1, 1)
        noisy_model_input = (1 - t) * latents + t * noise
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift
        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        timesteps = (timesteps * shift) / (1 + (shift - 1) * timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0
        noisy_model_input = (1 - t) * latents + t * noise
    elif args.timestep_sampling == "nextdit_shift":
        logits_norm = torch.randn(bsz, device=device)
        logits_norm = logits_norm * args.sigmoid_scale  # larger scale for more uniform sampling
        timesteps = logits_norm.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
        timesteps = time_shift(mu, 1.0, timesteps)

        t = timesteps.view(-1, 1, 1, 1)
        timesteps = timesteps * 1000.0
        noisy_model_input = (1 - t) * latents + t * noise
    else:
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme,
            batch_size=bsz,
            logit_mean=args.logit_mean,
            logit_std=args.logit_std,
            mode_scale=args.mode_scale,
        )
        indices = (u * noise_scheduler.config.num_train_timesteps).long()
        timesteps = noise_scheduler.timesteps[indices].to(device=device)

        # Add noise according to flow matching.
        sigmas = get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas


def apply_model_prediction_type(
    args, model_pred: Tensor, noisy_model_input: Tensor, sigmas: Tensor
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Apply model prediction type to the model prediction and the sigmas.

    Args:
        args (argparse.Namespace): Arguments.
        model_pred (Tensor): Model prediction.
        noisy_model_input (Tensor): Noisy model input.
        sigmas (Tensor): Sigmas.

    Return:
        Tuple[Tensor, Optional[Tensor]]:
    """
    weighting = None
    if args.model_prediction_type == "raw":
        pass
    elif args.model_prediction_type == "additive":
        # add the model_pred to the noisy_model_input
        model_pred = model_pred + noisy_model_input
    elif args.model_prediction_type == "sigma_scaled":
        # apply sigma scaling
        model_pred = model_pred * (-sigmas) + noisy_model_input

        # these weighting schemes use a uniform timestep sampling
        # and instead post-weight the loss
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

    return model_pred, weighting


def save_models(
    ckpt_path: str,
    lumina: lumina_models.NextDiT,
    sai_metadata: Dict[str, Any],
    save_dtype: Optional[torch.dtype] = None,
    use_mem_eff_save: bool = False,
):
    """
    Save the model to the checkpoint path.

    Args:
        ckpt_path (str): Path to the checkpoint.
        lumina (lumina_models.NextDiT): NextDIT model.
        sai_metadata (Optional[dict]): Metadata for the SAI model.
        save_dtype (Optional[torch.dtype]): Data

    Return:
        None
    """
    state_dict = {}

    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None and v.dtype != save_dtype:
                v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v

    update_sd("", lumina.state_dict())

    if not use_mem_eff_save:
        save_file(state_dict, ckpt_path, metadata=sai_metadata)
    else:
        mem_eff_save_file(state_dict, ckpt_path, metadata=sai_metadata)


def save_lumina_model_on_train_end(
    args: argparse.Namespace, save_dtype: torch.dtype, epoch: int, global_step: int, lumina: lumina_models.NextDiT
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = train_util.get_sai_model_spec(
            None, args, False, False, False, is_stable_diffusion_ckpt=True, lumina="lumina2"
        )
        save_models(ckpt_file, lumina, sai_metadata, save_dtype, args.mem_eff_save)

    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合してている
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_lumina_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator: Accelerator,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    lumina: lumina_models.NextDiT,
):
    """
    Save the model to the checkpoint path.

    Args:
        args (argparse.Namespace): Arguments.
        save_dtype (torch.dtype): Data type.
        epoch (int): Epoch.
        global_step (int): Global step.
        lumina (lumina_models.NextDiT): NextDIT model.

    Return:
        None
    """

    def sd_saver(ckpt_file: str, epoch_no: int, global_step: int):
        sai_metadata = train_util.get_sai_model_spec({}, args, False, False, False, is_stable_diffusion_ckpt=True, lumina="lumina2")
        save_models(ckpt_file, lumina, sai_metadata, save_dtype, args.mem_eff_save)

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


# endregion


def add_lumina_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--gemma2",
        type=str,
        help="path to gemma2 model (*.sft or *.safetensors), should be float16 / gemma2のパス（*.sftまたは*.safetensors）、float16が前提",
    )
    parser.add_argument("--ae", type=str, help="path to ae (*.sft or *.safetensors) / aeのパス（*.sftまたは*.safetensors）")
    parser.add_argument(
        "--gemma2_max_token_length",
        type=int,
        default=None,
        help="maximum token length for Gemma2. if omitted, 256 for schnell and 512 for dev"
        " / Gemma2の最大トークン長。省略された場合、schnellの場合は256、devの場合は512",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the NextDIT.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "nextdit_shift"],
        default="sigma",
        help="Method to sample timesteps: sigma-based, uniform random, sigmoid of random normal, shift of sigmoid and NextDIT.1 shifting."
        " / タイムステップをサンプリングする方法：sigma、random uniform、random normalのsigmoid、sigmoidのシフト、NextDIT.1のシフト。",
    )
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help='Scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid"). / sigmoidタイムステップサンプリングの倍率（timestep-samplingが"sigmoid"の場合のみ有効）。',
    )
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw", "additive", "sigma_scaled"],
        default="sigma_scaled",
        help="How to interpret and process the model prediction: "
        "raw (use as is), additive (add to noisy input), sigma_scaled (apply sigma scaling)."
        " / モデル予測の解釈と処理方法："
        "raw（そのまま使用）、additive（ノイズ入力に加算）、sigma_scaled（シグマスケーリングを適用）。",
    )
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=3.0,
        help="Discrete flow shift for the Euler Discrete Scheduler, default is 3.0. / Euler Discrete Schedulerの離散フローシフト、デフォルトは3.0。",
    )
