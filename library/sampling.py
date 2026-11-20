"""Sample-image generation helpers extracted from ``library.train_util``.

This module hosts the inference-time sampling utilities that train scripts
call periodically (every N steps / epochs) to produce preview images:

- Default noise scheduler constants for ``get_my_scheduler``.
- :func:`get_my_scheduler` — build a diffusers noise scheduler from
  ``--sample_sampler`` (ddim / pndm / euler / dpmsolver / ...).
- :func:`line_to_prompt_dict` / :func:`load_prompts` — parse prompt files
  (``.txt`` / ``.toml`` / ``.json``) into prompt dicts. ``--w / --h /
  --d / --s / --l / --g / --n / --ss / --cn / --mk / --i / --ctr / --rcfg /
  --fs / --am`` flags are recognised.
- :func:`sample_images` / :func:`sample_images_common` /
  :func:`sample_image_inference` — orchestrate one round of sample
  generation (multi-GPU split via ``accelerate.PartialState``, RNG
  save/restore, optional inpainting input, W&B image upload).

These used to live in ``library.train_util`` and are still re-exported
from there for backward compatibility. New code should import from this
module.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import re
import time
from typing import Dict, List, Optional, Union

import toml
import torch
from accelerate import Accelerator, PartialState
from PIL import Image

from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from library.device_utils import clean_memory_on_device
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline

# library.sdxl_lpw_stable_diffusion pulls in train_util / sdxl_train_util, which
# in turn re-export from this module. Importing the SDXL pipeline at module top
# therefore cycles via train_util.py's `from library.sdxl_lpw_stable_diffusion
# import ...`. We only use the class for an isinstance check inside
# sample_image_inference, so import it lazily there.
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"


def check_sampler_requirements(sample_sampler: str) -> None:
    """Raise a clear ``ImportError`` if an optional dependency of the sampler is missing.

    ``lms`` / ``k_lms`` (``LMSDiscreteScheduler``) need ``scipy``, which is not in ``requirements.txt``.
    Without this check the failure surfaces only when the scheduler is built, i.e. at the first sample
    generation after the models are loaded.
    """
    if sample_sampler in ("lms", "k_lms") and importlib.util.find_spec("scipy") is None:
        raise ImportError(
            f"The sampler '{sample_sampler}' (LMSDiscreteScheduler) requires the `scipy` package, which is not included in"
            " requirements.txt. Please run `pip install scipy`, or choose another sampler (e.g. euler, dpmsolver++)."
            f" / サンプラー '{sample_sampler}' (LMSDiscreteScheduler) には requirements.txt に含まれない `scipy` パッケージが必要です。"
            "`pip install scipy` を実行するか、他のサンプラー（euler、dpmsolver++ など）を指定してください。"
        )


def get_my_scheduler(
    *,
    sample_sampler: str,
    v_parameterization: bool,
):
    check_sampler_requirements(sample_sampler)

    sched_init_args = {}
    has_steps_offset = True
    if sample_sampler == "ddim":
        scheduler_cls = DDIMScheduler
    elif sample_sampler == "ddpm":  # ddpmはおかしくなるのでoptionから外してある
        scheduler_cls = DDPMScheduler
    elif sample_sampler == "pndm":
        scheduler_cls = PNDMScheduler
    elif sample_sampler == "lms" or sample_sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif sample_sampler == "euler" or sample_sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif sample_sampler == "euler_a" or sample_sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif sample_sampler == "dpmsolver" or sample_sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = sample_sampler
        if sample_sampler == "dpmsolver":
            # diffusers rejects the default final_sigmas_type="zero" for the (non-++) dpmsolver algorithm
            sched_init_args["final_sigmas_type"] = "sigma_min"
    elif sample_sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
        has_steps_offset = False  # DPMSolverSinglestepScheduler has no steps_offset argument
    elif sample_sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif sample_sampler == "dpm_2" or sample_sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif sample_sampler == "dpm_2_a" or sample_sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"
    if has_steps_offset:
        sched_init_args["steps_offset"] = 1

    scheduler = scheduler_cls(
        num_train_timesteps=SCHEDULER_TIMESTEPS,
        beta_start=SCHEDULER_LINEAR_START,
        beta_end=SCHEDULER_LINEAR_END,
        beta_schedule=SCHEDLER_SCHEDULE,
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # logger.info("set clip_sample to True")
        scheduler.config.clip_sample = True

    return scheduler


def sample_images(*args, **kwargs):
    return sample_images_common(StableDiffusionLongPromptWeightingPipeline, *args, **kwargs)


def line_to_prompt_dict(line: str) -> dict:
    # subset of gen_img.py
    prompt_args = line.split(" --")
    prompt_dict = {}
    prompt_dict["prompt"] = prompt_args[0]

    for parg in prompt_args:
        try:
            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["width"] = int(m.group(1))
                continue

            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["height"] = int(m.group(1))
                continue

            m = re.match(r"d (\d+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["seed"] = int(m.group(1))
                continue

            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
            if m:  # steps. 0 makes the prompt a reserved one, see `is_reserved_prompt`
                prompt_dict["sample_steps"] = max(0, min(1000, int(m.group(1))))
                continue

            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # scale
                prompt_dict["scale"] = float(m.group(1))
                continue

            m = re.match(r"g ([\d\.]+)", parg, re.IGNORECASE)
            if m:  # guidance scale
                prompt_dict["guidance_scale"] = float(m.group(1))
                continue

            m = re.match(r"n (.+)", parg, re.IGNORECASE)
            if m:  # negative prompt
                prompt_dict["negative_prompt"] = m.group(1)
                continue

            m = re.match(r"ss (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["sample_sampler"] = m.group(1)
                continue

            m = re.match(r"cn (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["controlnet_image"] = m.group(1)
                continue

            m = re.match(r"mk (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["mask_image"] = m.group(1).strip()
                continue

            m = re.match(r"i (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["image"] = m.group(1).strip()
                continue

            m = re.match(r"ctr (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["cfg_trunc_ratio"] = float(m.group(1))
                continue

            m = re.match(r"rcfg (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["renorm_cfg"] = float(m.group(1))
                continue

            m = re.match(r"fs (.+)", parg, re.IGNORECASE)
            if m:
                prompt_dict["flow_shift"] = m.group(1)
                continue

            m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
            if m:  # additional network multiplier(s) — comma-separated list, same as gen_img.py
                prompt_dict["additional_network_multiplier"] = [float(v) for v in m.group(1).split(",")]
                continue

        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(ex)

    # `\n` (backslash + n) in line-format prompts is converted to a line break, for the models whose text encoders
    # are sensitive to line breaks. .toml / .json (dict) prompts can contain real line breaks, so they are not converted
    for key in ("prompt", "negative_prompt"):
        if key in prompt_dict and prompt_dict[key] is not None:
            prompt_dict[key] = prompt_dict[key].replace("\\n", "\n")

    return prompt_dict


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
            prompt_dict = line_to_prompt_dict(prompt_dict)
            prompts[i] = prompt_dict
        assert isinstance(prompt_dict, dict)

        # Adds an enumerator to the dict based on prompt position. Used later to name image files. Also cleanup of extra data in original prompt dict.
        prompt_dict["enum"] = i
        prompt_dict.pop("subset", None)

    return prompts


# region shared by the sampling functions of all models

# When not None, the sampling functions generate these prompts regardless of the step / epoch and the prompt file.
# The training is single-threaded around the sampling, so plain module variables are enough.
_forced_prompts: Optional[List[Dict]] = None
_saved_sample_collectors: List[List[str]] = []

# The file names of the forced samples use numbers from this, not to overwrite the scheduled samples of the same step
FORCED_PROMPT_ENUM_OFFSET = 100


def should_sample_images(args: argparse.Namespace, epoch: Optional[int], steps: int) -> bool:
    """Whether to generate sample images now. `epoch` is None for the call at each step, the epoch number at the epoch end."""
    if _forced_prompts is not None:
        return True
    if steps == 0:
        return bool(args.sample_at_first)
    if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
        return False
    if args.sample_every_n_epochs is not None:
        # sample_every_n_steps is ignored
        return epoch is not None and epoch % args.sample_every_n_epochs == 0
    return steps % args.sample_every_n_steps == 0 and epoch is None  # divisible, and not the end of an epoch


def has_sample_prompts(args: argparse.Namespace) -> bool:
    if _forced_prompts is not None:
        return True
    return args.sample_prompts is not None and os.path.isfile(args.sample_prompts)


def is_reserved_prompt(prompt_dict: Dict) -> bool:
    """A prompt whose number of steps is 0 (`--s 0`) is a reserved prompt.

    It is not generated by the scheduled sampling, but it is in the prompt file, so its text encoder outputs are cached
    at the start of the training together with the other prompts (`load_prompts` returns it). A sampling request from
    outside can therefore use it by its index, even if the text encoder is not in memory any more.
    """
    return prompt_dict.get("sample_steps") == 0


def get_sample_prompts(args: argparse.Namespace) -> List[Dict]:
    """The prompts to generate now: the prompt file without the reserved prompts, or the forced prompts."""
    if _forced_prompts is not None:
        return [dict(p) for p in _forced_prompts]
    return [p for p in load_prompts(args.sample_prompts) if not is_reserved_prompt(p)]


def notify_sample_saved(path: str) -> None:
    """The sampling functions call this for each saved image."""
    for collector in _saved_sample_collectors:
        collector.append(path)


@contextlib.contextmanager
def collect_saved_samples():
    """Yields a list to which the paths of the images saved in the context are added."""
    collector: List[str] = []
    _saved_sample_collectors.append(collector)
    try:
        yield collector
    finally:
        for i, c in enumerate(_saved_sample_collectors):
            if c is collector:
                del _saved_sample_collectors[i]
                break


@contextlib.contextmanager
def force_sample_prompts(prompts: List[Dict]):
    """In the context, the sampling functions generate `prompts` (prompt dicts) at any step."""
    global _forced_prompts
    previous = _forced_prompts
    _forced_prompts = prompts
    try:
        yield
    finally:
        _forced_prompts = previous


def build_forced_prompts(
    args: argparse.Namespace,
    prompts: Optional[List[Union[str, Dict]]] = None,
    indices: Optional[List[int]] = None,
    overrides: Optional[Dict] = None,
) -> List[Dict]:
    """Build the prompt dicts for a sampling request from outside of the training schedule.

    Args:
        prompts: prompt lines in the syntax of the prompt file (e.g. "a cat --w 1024 --h 1024 --d 1"), or prompt dicts.
        indices: indices of the prompts in the prompt file (`--sample_prompts`), starting from 0.
        overrides: values to set to every prompt, with the keys of the prompt dict (`width`, `height`, `seed`,
            `sample_steps`, `scale`, `negative_prompt`, ...). None values are ignored.

    If neither `prompts` nor `indices` is given, the prompts of the scheduled sampling are used: all prompts in the
    prompt file except the reserved ones (`--s 0`). A reserved prompt is selected by its index, and is generated with
    the default number of steps of the model unless `sample_steps` is in `overrides`.
    Raises ValueError if the request is not valid.
    """
    result: List[Dict] = []

    if indices or not prompts:
        if args.sample_prompts is None or not os.path.isfile(args.sample_prompts):
            raise ValueError("the prompt file (--sample_prompts) is required unless prompts are given")
        file_prompts = load_prompts(args.sample_prompts)
        if indices:
            for index in indices:
                if not isinstance(index, int) or isinstance(index, bool) or not (0 <= index < len(file_prompts)):
                    raise ValueError(f"index {index} is out of range. the prompt file has {len(file_prompts)} prompts")
                result.append(dict(file_prompts[index]))
        else:
            result.extend(p for p in file_prompts if not is_reserved_prompt(p))
            if len(result) == 0:
                raise ValueError("the prompt file has no prompts for the scheduled sampling. select the prompts by index")

    for prompt in prompts or []:
        if isinstance(prompt, str):
            result.append(line_to_prompt_dict(prompt))
        elif isinstance(prompt, dict):
            result.append(dict(prompt))
        else:
            raise ValueError(f"a prompt must be a string or a dict: {prompt!r}")

    for i, prompt_dict in enumerate(result):
        if overrides:
            if not isinstance(overrides, dict):
                raise ValueError("overrides must be a dict")
            prompt_dict.update({k: v for k, v in overrides.items() if v is not None})
        if is_reserved_prompt(prompt_dict):
            del prompt_dict["sample_steps"]  # 0 steps cannot be generated: use the default of the model
        prompt_dict["enum"] = FORCED_PROMPT_ENUM_OFFSET + i
    return result


def verify_forced_prompts_are_cached(prompts: List[Dict], cached_outputs: Dict) -> None:
    """Raises ValueError if a prompt or a negative prompt is not in the cached text encoder outputs.

    When the text encoder outputs of the prompt file were cached at the start of the training, the text encoder may not
    be in memory any more, and the sampling functions cannot encode other prompts (some of them fail in the middle of
    the sampling). `cached_outputs` is the dict of the cached outputs, whose keys are the prompts.
    """
    for prompt_dict in prompts:
        texts = [("prompt", prompt_dict.get("prompt", ""))]
        if prompt_dict.get("negative_prompt") is not None:
            texts.append(("negative prompt", prompt_dict["negative_prompt"]))
        for kind, text in texts:
            if text not in cached_outputs:
                raise ValueError(
                    f"the {kind} `{text}` is not in the prompt file (--sample_prompts). Only the prompts and the negative prompts"
                    " of the prompt file can be used, because the text encoder outputs are cached"
                )


# Text encoder outputs of prompts which are encoded outside of the training process (tools/run_control_sample_prompt.py)
# and sent to it with a sampling request. The file is a safetensors file: the tensors of the i-th text have the keys
# `<i>.<j>` (the j-th element of the return value of `TextEncodingStrategy.encode_tokens`), and the texts are in the metadata.
PROMPT_TE_OUTPUTS_FORMAT = "sd-scripts-prompt-te-outputs"
PROMPT_TE_OUTPUTS_VERSION = "1"


def save_prompt_te_outputs(path: str, outputs: Dict[str, List[torch.Tensor]]) -> None:
    """Save the text encoder outputs of prompts. The keys of `outputs` are the texts (prompts and negative prompts)."""
    from safetensors.torch import save_file

    texts = list(outputs.keys())
    tensors = {}
    for i, text in enumerate(texts):
        for j, tensor in enumerate(outputs[text]):
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"the text encoder outputs must be tensors: {type(tensor)}")
            tensors[f"{i}.{j}"] = tensor.detach().to("cpu").contiguous()
    metadata = {
        "format": PROMPT_TE_OUTPUTS_FORMAT,
        "version": PROMPT_TE_OUTPUTS_VERSION,
        "texts": json.dumps(texts, ensure_ascii=False),
    }
    save_file(tensors, path, metadata=metadata)


def load_prompt_te_outputs(path: str) -> Dict[str, List[torch.Tensor]]:
    """Load the file written by `save_prompt_te_outputs`. The tensors are on CPU. Raises ValueError if the file is not valid."""
    from safetensors import safe_open

    try:
        outputs: Dict[str, List[torch.Tensor]] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if metadata.get("format") != PROMPT_TE_OUTPUTS_FORMAT or metadata.get("version") != PROMPT_TE_OUTPUTS_VERSION:
                raise ValueError(f"not a file of text encoder outputs of prompts (version {PROMPT_TE_OUTPUTS_VERSION})")
            texts = json.loads(metadata.get("texts", "null"))
            if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
                raise ValueError("the texts in the metadata are not valid")
            keys = set(f.keys())
            for i, text in enumerate(texts):
                tensors = []
                while f"{i}.{len(tensors)}" in keys:
                    tensors.append(f.get_tensor(f"{i}.{len(tensors)}"))
                outputs[text] = tensors
        return outputs
    except ValueError as e:
        raise ValueError(f"failed to load the text encoder outputs {path}: {e}")
    except Exception as e:  # OSError, SafetensorError, ...
        raise ValueError(f"failed to load the text encoder outputs {path}: {type(e).__name__}: {e}")


def verify_prompt_te_outputs(outputs: Dict[str, List[torch.Tensor]], reference: List[torch.Tensor]) -> None:
    """Raises ValueError if the text encoder outputs from outside do not have the same structure as `reference`.

    `reference` is the outputs of a prompt which were encoded in the training process. Outputs of another model or of a
    broken file would raise an error in the middle of the sampling, which ends the training, so they are rejected here.
    The length of the sequence is not compared, because it depends on the prompt in some models.
    """
    for text, tensors in outputs.items():
        if len(tensors) != len(reference):
            raise ValueError(f"the text encoder outputs of `{text}` have {len(tensors)} tensors, expected {len(reference)}")
        for j, (tensor, ref) in enumerate(zip(tensors, reference)):
            ref = torch.as_tensor(ref)
            name = f"the tensor {j} of the text encoder outputs of `{text}`"
            if tensor.ndim != ref.ndim or tensor.ndim == 0:
                raise ValueError(f"{name} has {tensor.ndim} dimensions, expected {ref.ndim}")
            if tensor.shape[0] != ref.shape[0]:
                raise ValueError(f"{name} has the batch size {tensor.shape[0]}, expected {ref.shape[0]}")
            if tensor.numel() == 0:
                raise ValueError(f"{name} is empty")
            if tensor.ndim >= 3 and tensor.shape[2:] != ref.shape[2:]:
                raise ValueError(f"{name} has the shape {tuple(tensor.shape)}, expected {tuple(ref.shape)} (except the length)")
            if tensor.is_floating_point() != ref.is_floating_point():
                raise ValueError(f"{name} has the dtype {tensor.dtype}, expected {ref.dtype}")
            if tensor.is_floating_point() and not torch.isfinite(tensor.float()).all():
                raise ValueError(f"{name} has non-finite values")


# endregion


def sample_images_common(
    pipe_class,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    steps: int,
    device,
    vae,
    tokenizer,
    text_encoder,
    unet_wrapped,
    prompt_replacement=None,
    controlnet=None,
):
    """
    StableDiffusionLongPromptWeightingPipelineの改造版を使うようにしたので、clip skipおよびプロンプトの重みづけに対応した
    TODO Use strategies here
    """

    if not should_sample_images(args, epoch, steps):
        return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not has_sample_prompts(args):
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    org_vae_device = vae.device  # CPUにいるはず
    vae.to(distributed_state.device)  # distributed_state.device is same as accelerator.device

    # unwrap unet and text_encoder(s)
    unet = accelerator.unwrap_model(unet_wrapped)
    if isinstance(text_encoder, (list, tuple)):
        text_encoder = [accelerator.unwrap_model(te) for te in text_encoder]
    else:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # read prompts
    prompts = get_sample_prompts(args)

    default_scheduler = get_my_scheduler(sample_sampler=args.sample_sampler, v_parameterization=args.v_parameterization)

    pipeline = pipe_class(
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        tokenizer=tokenizer,
        scheduler=default_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        clip_skip=args.clip_skip,
    )
    pipeline.to(distributed_state.device)
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
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
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
                        accelerator, args, pipeline, save_dir, prompt_dict, epoch, steps, prompt_replacement, controlnet=controlnet
                    )

    # clear pipeline and cache to reduce vram usage
    del pipeline

    torch.set_rng_state(rng_state)
    if torch.cuda.is_available() and cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    vae.to(org_vae_device)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    pipeline,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    prompt_replacement,
    controlnet=None,
):
    from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 30)
    width = prompt_dict.get("width", 512)
    height = prompt_dict.get("height", 512)
    scale = prompt_dict.get("scale", 7.5)
    seed = prompt_dict.get("seed")
    controlnet_image = prompt_dict.get("controlnet_image")
    prompt: str = prompt_dict.get("prompt", "")
    sampler_name: str = prompt_dict.get("sample_sampler", args.sample_sampler)

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    else:
        # True random sample image generation
        torch.seed()
        if torch.cuda.is_available():
            torch.cuda.seed()

    scheduler = get_my_scheduler(
        sample_sampler=sampler_name,
        v_parameterization=args.v_parameterization,
    )
    pipeline.scheduler = scheduler

    if controlnet_image is not None:
        controlnet_image = Image.open(controlnet_image).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)

    # Round down so the latent shape is divisible by the UNet's largest stride:
    # SDXL has 2 downsamples (latent /4 → image /32); SD1.5/2.x has 3 (latent /8 → image /64).
    divisor = 32 if isinstance(pipeline, SdxlStableDiffusionLongPromptWeightingPipeline) else 64
    height = max(divisor, height - height % divisor)
    width = max(divisor, width - width % divisor)
    logger.info(f"prompt: {prompt}")
    logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}")
    logger.info(f"width: {width}")
    logger.info(f"sample_steps: {sample_steps}")
    logger.info(f"scale: {scale}")
    logger.info(f"sample_sampler: {sampler_name}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    # Prepare inpainting source image and mask when training an inpainting model.
    # The prompt line should include "--i /path/to/image.jpg".
    # The mask is generated reproducibly from the prompt seed (or randomly if no seed).
    inpaint_image = None
    inpaint_mask = None
    if getattr(args, "train_inpainting", False):
        image_path = prompt_dict.get("image")
        if image_path:
            from library.mask_generator import wobbly_ellipse_mask as _gen_mask

            if not os.path.exists(image_path):
                logger.warning(f"inpaint image not found, skipping sample: {image_path}")
                return
            inpaint_image = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
            inpaint_mask = _gen_mask(width, height, seed=seed)
            logger.info(f"inpaint image: {image_path}")
        else:
            logger.warning(
                "train_inpainting is set but no source image specified in the prompt line; "
                "skipping sample. Add '--i /path/to/image.jpg' to the prompt to enable inpainting sampling."
            )
            return

    with accelerator.autocast(), torch.no_grad():
        latents = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=scale,
            negative_prompt=negative_prompt,
            controlnet=controlnet,
            controlnet_image=controlnet_image,
            inpaint_image=inpaint_image,
            inpaint_mask=inpaint_mask,
        )

    if torch.cuda.is_available():
        with torch.cuda.device(torch.cuda.current_device()):
            torch.cuda.empty_cache()

    image = pipeline.latents_to_image(latents)[0]

    # adding accelerator.wait_for_everyone() here should sync up and ensure that sample images are saved in the same order as the original prompt list
    # but adding 'enum' to the filename should be enough

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    img_path = os.path.join(save_dir, img_filename)
    image.save(img_path)
    notify_sample_saved(img_path)

    # send images to wandb if enabled
    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")

        import wandb

        # not to commit images to avoid inconsistency between training and logging steps
        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)  # positive prompt as a caption
