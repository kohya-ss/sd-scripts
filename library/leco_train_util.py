import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import toml
from torch.utils.checkpoint import checkpoint

from library import train_util

import logging

logger = logging.getLogger(__name__)


def build_network_kwargs(args: argparse.Namespace) -> Dict[str, str]:
    kwargs = {}
    if args.network_args:
        for net_arg in args.network_args:
            key, value = net_arg.split("=", 1)
            kwargs[key] = value
    if "dropout" not in kwargs:
        kwargs["dropout"] = args.network_dropout
    return kwargs


def get_save_extension(args: argparse.Namespace) -> str:
    if args.save_model_as == "ckpt":
        return ".ckpt"
    if args.save_model_as == "pt":
        return ".pt"
    return ".safetensors"


def save_weights(
    accelerator,
    network,
    args: argparse.Namespace,
    save_dtype,
    prompt_settings,
    global_step: int,
    last: bool = False,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    ext = get_save_extension(args)
    ckpt_name = train_util.get_last_ckpt_name(args, ext) if last else train_util.get_step_ckpt_name(args, ext, global_step)
    ckpt_file = os.path.join(args.output_dir, ckpt_name)

    metadata = None
    if not args.no_metadata:
        metadata = {
            "ss_network_module": args.network_module,
            "ss_network_dim": str(args.network_dim),
            "ss_network_alpha": str(args.network_alpha),
            "ss_leco_prompt_count": str(len(prompt_settings)),
            "ss_leco_prompts_file": os.path.basename(args.prompts_file),
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        if args.training_comment:
            metadata["ss_training_comment"] = args.training_comment
        metadata["ss_leco_preview"] = json.dumps(
            [
                {
                    "target": p.target,
                    "positive": p.positive,
                    "unconditional": p.unconditional,
                    "neutral": p.neutral,
                    "action": p.action,
                    "multiplier": p.multiplier,
                    "weight": p.weight,
                }
                for p in prompt_settings[:16]
            ],
            ensure_ascii=False,
        )

    unwrapped = accelerator.unwrap_model(network)
    unwrapped.save_weights(ckpt_file, save_dtype, metadata)
    logger.info(f"saved model to: {ckpt_file}")



ResolutionValue = Union[int, Tuple[int, int]]


@dataclass
class PromptEmbedsXL:
    text_embeds: torch.Tensor
    pooled_embeds: torch.Tensor


class PromptEmbedsCache:
    def __init__(self):
        self.prompts: dict[str, Any] = {}

    def __setitem__(self, name: str, value: Any) -> None:
        self.prompts[name] = value

    def __getitem__(self, name: str) -> Any:
        return self.prompts[name]


@dataclass
class PromptSettings:
    target: str
    positive: Optional[str] = None
    unconditional: str = ""
    neutral: Optional[str] = None
    action: str = "erase"
    guidance_scale: float = 1.0
    resolution: ResolutionValue = 512
    dynamic_resolution: bool = False
    batch_size: int = 1
    dynamic_crops: bool = False
    multiplier: float = 1.0
    weight: float = 1.0

    def __post_init__(self):
        if self.positive is None:
            self.positive = self.target
        if self.neutral is None:
            self.neutral = self.unconditional
        if self.action not in ("erase", "enhance"):
            raise ValueError(f"Invalid action: {self.action}")

        self.guidance_scale = float(self.guidance_scale)
        self.batch_size = int(self.batch_size)
        self.multiplier = float(self.multiplier)
        self.weight = float(self.weight)
        self.dynamic_resolution = bool(self.dynamic_resolution)
        self.dynamic_crops = bool(self.dynamic_crops)
        self.resolution = normalize_resolution(self.resolution)

    def get_resolution(self) -> Tuple[int, int]:
        if isinstance(self.resolution, tuple):
            return self.resolution
        return (self.resolution, self.resolution)

    def build_target(self, positive_latents, neutral_latents, unconditional_latents):
        offset = self.guidance_scale * (positive_latents - unconditional_latents)
        if self.action == "erase":
            return neutral_latents - offset
        return neutral_latents + offset


def normalize_resolution(value: Any) -> ResolutionValue:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"resolution tuple must have 2 items: {value}")
        return (int(value[0]), int(value[1]))
    if isinstance(value, list):
        if len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            return (int(value[0]), int(value[1]))
        raise ValueError(f"resolution list must have 2 numeric items: {value}")
    return int(value)


def _read_non_empty_lines(path: Union[str, Path]) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _recognized_prompt_keys() -> set[str]:
    return {
        "target",
        "positive",
        "unconditional",
        "neutral",
        "action",
        "guidance_scale",
        "resolution",
        "dynamic_resolution",
        "batch_size",
        "dynamic_crops",
        "multiplier",
        "weight",
    }


def _recognized_slider_keys() -> set[str]:
    return {
        "target_class",
        "positive",
        "negative",
        "neutral",
        "guidance_scale",
        "resolution",
        "resolutions",
        "dynamic_resolution",
        "batch_size",
        "dynamic_crops",
        "multiplier",
        "weight",
    }


def _merge_known_defaults(defaults: dict[str, Any], item: dict[str, Any], known_keys: Iterable[str]) -> dict[str, Any]:
    merged = {k: v for k, v in defaults.items() if k in known_keys}
    merged.update(item)
    return merged


def _normalize_resolution_values(value: Any) -> List[ResolutionValue]:
    if value is None:
        return [512]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        return [normalize_resolution(v) for v in value]
    return [normalize_resolution(value)]


def _expand_slider_target(target: dict[str, Any], neutral: str) -> List[PromptSettings]:
    target_class = str(target.get("target_class", ""))
    positive = str(target.get("positive", "") or "")
    negative = str(target.get("negative", "") or "")
    multiplier = target.get("multiplier", 1.0)
    resolutions = _normalize_resolution_values(target.get("resolutions", target.get("resolution", 512)))

    if not positive.strip() and not negative.strip():
        raise ValueError("slider target requires either positive or negative prompt")

    base = dict(
        target=target_class,
        neutral=neutral,
        guidance_scale=target.get("guidance_scale", 1.0),
        dynamic_resolution=target.get("dynamic_resolution", False),
        batch_size=target.get("batch_size", 1),
        dynamic_crops=target.get("dynamic_crops", False),
        weight=target.get("weight", 1.0),
    )

    # Build bidirectional (positive_prompt, unconditional_prompt, action, multiplier_sign) pairs.
    # With both positive and negative: 4 pairs; with only one: 2 pairs.
    pairs: list[tuple[str, str, str, float]] = []
    if positive.strip() and negative.strip():
        pairs = [
            (negative, positive, "erase", multiplier),
            (positive, negative, "enhance", multiplier),
            (positive, negative, "erase", -multiplier),
            (negative, positive, "enhance", -multiplier),
        ]
    elif negative.strip():
        pairs = [
            (negative, "", "erase", multiplier),
            (negative, "", "enhance", -multiplier),
        ]
    else:
        pairs = [
            (positive, "", "enhance", multiplier),
            (positive, "", "erase", -multiplier),
        ]

    prompt_settings: List[PromptSettings] = []
    for resolution in resolutions:
        for pos, uncond, action, mult in pairs:
            prompt_settings.append(
                PromptSettings(**base, positive=pos, unconditional=uncond, action=action, resolution=resolution, multiplier=mult)
            )

    return prompt_settings


def load_prompt_settings(path: Union[str, Path]) -> List[PromptSettings]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = toml.load(f)

    if not data:
        raise ValueError("prompt file is empty")

    default_prompt_values = {
        "guidance_scale": 1.0,
        "resolution": 512,
        "dynamic_resolution": False,
        "batch_size": 1,
        "dynamic_crops": False,
        "multiplier": 1.0,
        "weight": 1.0,
    }

    prompt_settings: List[PromptSettings] = []

    def append_prompt_item(item: dict[str, Any], defaults: dict[str, Any]) -> None:
        merged = _merge_known_defaults(defaults, item, _recognized_prompt_keys())
        prompt_settings.append(PromptSettings(**merged))

    def append_slider_item(item: dict[str, Any], defaults: dict[str, Any], neutral_values: Sequence[str]) -> None:
        merged = _merge_known_defaults(defaults, item, _recognized_slider_keys())
        if not neutral_values:
            neutral_values = [str(merged.get("neutral", "") or "")]
        for neutral in neutral_values:
            prompt_settings.extend(_expand_slider_target(merged, neutral))

    if "prompts" in data:
        defaults = {**default_prompt_values, **{k: v for k, v in data.items() if k in _recognized_prompt_keys()}}
        for item in data["prompts"]:
            if "target_class" in item:
                append_slider_item(item, defaults, [str(item.get("neutral", "") or "")])
            else:
                append_prompt_item(item, defaults)
    else:
        slider_config = data.get("slider", data)
        targets = slider_config.get("targets")
        if targets is None:
            if "target_class" in slider_config:
                targets = [slider_config]
            elif "target" in slider_config:
                targets = [slider_config]
            else:
                raise ValueError("prompt file does not contain prompts or slider targets")
        if len(targets) == 0:
            raise ValueError("prompt file contains an empty targets list")

        if "target" in targets[0]:
            defaults = {**default_prompt_values, **{k: v for k, v in slider_config.items() if k in _recognized_prompt_keys()}}
            for item in targets:
                append_prompt_item(item, defaults)
        else:
            defaults = {**default_prompt_values, **{k: v for k, v in slider_config.items() if k in _recognized_slider_keys()}}
            neutral_values: List[str] = []
            if "neutrals" in slider_config:
                neutral_values.extend(str(v) for v in slider_config["neutrals"])
            if "neutral_prompt_file" in slider_config:
                neutral_values.extend(_read_non_empty_lines(path.parent / slider_config["neutral_prompt_file"]))
            if "prompt_file" in slider_config:
                neutral_values.extend(_read_non_empty_lines(path.parent / slider_config["prompt_file"]))
            if not neutral_values:
                neutral_values = [str(slider_config.get("neutral", "") or "")]

            for item in targets:
                item_neutrals = neutral_values
                if "neutrals" in item:
                    item_neutrals = [str(v) for v in item["neutrals"]]
                elif "neutral_prompt_file" in item:
                    item_neutrals = _read_non_empty_lines(path.parent / item["neutral_prompt_file"])
                elif "prompt_file" in item:
                    item_neutrals = _read_non_empty_lines(path.parent / item["prompt_file"])
                elif "neutral" in item:
                    item_neutrals = [str(item["neutral"] or "")]

                append_slider_item(item, defaults, item_neutrals)

    if not prompt_settings:
        raise ValueError("no prompt settings found")

    return prompt_settings


def encode_prompt_sd(tokenize_strategy, text_encoding_strategy, text_encoder, prompt: str) -> torch.Tensor:
    tokens = tokenize_strategy.tokenize(prompt)
    return text_encoding_strategy.encode_tokens(tokenize_strategy, [text_encoder], tokens)[0]


def encode_prompt_sdxl(tokenize_strategy, text_encoding_strategy, text_encoders, prompt: str) -> PromptEmbedsXL:
    tokens = tokenize_strategy.tokenize(prompt)
    hidden1, hidden2, pool2 = text_encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens)
    return PromptEmbedsXL(torch.cat([hidden1, hidden2], dim=2), pool2)


def apply_noise_offset(latents: torch.Tensor, noise_offset: Optional[float]) -> torch.Tensor:
    if noise_offset is None:
        return latents
    noise = torch.randn((latents.shape[0], latents.shape[1], 1, 1), dtype=torch.float32, device="cpu")
    noise = noise.to(dtype=latents.dtype, device=latents.device)
    return latents + noise_offset * noise


def get_initial_latents(scheduler, batch_size: int, height: int, width: int, n_prompts: int = 1) -> torch.Tensor:
    noise = torch.randn(
        (batch_size, 4, height // 8, width // 8),
        device="cpu",
    ).repeat(n_prompts, 1, 1, 1)
    return noise * scheduler.init_noise_sigma


def concat_embeddings(unconditional: torch.Tensor, conditional: torch.Tensor, batch_size: int) -> torch.Tensor:
    return torch.cat([unconditional, conditional], dim=0).repeat_interleave(batch_size, dim=0)


def concat_embeddings_xl(unconditional: PromptEmbedsXL, conditional: PromptEmbedsXL, batch_size: int) -> PromptEmbedsXL:
    text_embeds = torch.cat([unconditional.text_embeds, conditional.text_embeds], dim=0).repeat_interleave(batch_size, dim=0)
    pooled_embeds = torch.cat([unconditional.pooled_embeds, conditional.pooled_embeds], dim=0).repeat_interleave(batch_size, dim=0)
    return PromptEmbedsXL(text_embeds=text_embeds, pooled_embeds=pooled_embeds)


def batch_add_time_ids(add_time_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Duplicate add_time_ids for CFG (unconditional + conditional) and repeat for the batch."""
    return torch.cat([add_time_ids, add_time_ids], dim=0).repeat_interleave(batch_size, dim=0)


def _run_with_checkpoint(function, *args):
    if torch.is_grad_enabled():
        return checkpoint(function, *args, use_reentrant=False)
    return function(*args)


def predict_noise(unet, scheduler, timestep, latents: torch.Tensor, text_embeddings: torch.Tensor, guidance_scale: float = 1.0):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    def run_unet(model_input, encoder_hidden_states):
        return unet(model_input, timestep, encoder_hidden_states=encoder_hidden_states).sample

    noise_pred = _run_with_checkpoint(run_unet, latent_model_input, text_embeddings)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


def diffusion(
    unet,
    scheduler,
    latents: torch.Tensor,
    text_embeddings: torch.Tensor,
    total_timesteps: int,
    start_timesteps: int = 0,
    guidance_scale: float = 3.0,
):
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        noise_pred = predict_noise(unet, scheduler, timestep, latents, text_embeddings, guidance_scale=guidance_scale)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents


def get_add_time_ids(
    height: int,
    width: int,
    dynamic_crops: bool = False,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if dynamic_crops:
        random_scale = torch.rand(1).item() * 2 + 1
        original_size = (int(height * random_scale), int(width * random_scale))
        crops_coords_top_left = (
            torch.randint(0, max(original_size[0] - height, 1), (1,)).item(),
            torch.randint(0, max(original_size[1] - width, 1), (1,)).item(),
        )
        target_size = (height, width)
    else:
        original_size = (height, width)
        crops_coords_top_left = (0, 0)
        target_size = (height, width)

    add_time_ids = torch.tensor([list(original_size + crops_coords_top_left + target_size)], dtype=dtype)
    if device is not None:
        add_time_ids = add_time_ids.to(device)
    return add_time_ids


def predict_noise_xl(
    unet,
    scheduler,
    timestep,
    latents: torch.Tensor,
    prompt_embeds: PromptEmbedsXL,
    add_time_ids: torch.Tensor,
    guidance_scale: float = 1.0,
):
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    orig_size = add_time_ids[:, :2]
    crop_size = add_time_ids[:, 2:4]
    target_size = add_time_ids[:, 4:6]
    from library import sdxl_train_util

    size_embeddings = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size, latent_model_input.device)
    vector_embedding = torch.cat([prompt_embeds.pooled_embeds, size_embeddings.to(prompt_embeds.pooled_embeds.dtype)], dim=1)

    def run_unet(model_input, text_embeds, vector_embeds):
        return unet(model_input, timestep, text_embeds, vector_embeds)

    noise_pred = _run_with_checkpoint(run_unet, latent_model_input, prompt_embeds.text_embeds, vector_embedding)
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    return noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


def diffusion_xl(
    unet,
    scheduler,
    latents: torch.Tensor,
    prompt_embeds: PromptEmbedsXL,
    add_time_ids: torch.Tensor,
    total_timesteps: int,
    start_timesteps: int = 0,
    guidance_scale: float = 3.0,
):
    for timestep in scheduler.timesteps[start_timesteps:total_timesteps]:
        noise_pred = predict_noise_xl(
            unet,
            scheduler,
            timestep,
            latents,
            prompt_embeds=prompt_embeds,
            add_time_ids=add_time_ids,
            guidance_scale=guidance_scale,
        )
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> Tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2
    step = 64
    min_step = min_resolution // step
    max_step = max_resolution // step
    height = torch.randint(min_step, max_step + 1, (1,)).item() * step
    width = torch.randint(min_step, max_step + 1, (1,)).item() * step
    return height, width


def get_random_resolution(prompt: PromptSettings) -> Tuple[int, int]:
    height, width = prompt.get_resolution()
    if prompt.dynamic_resolution and height == width:
        return get_random_resolution_in_bucket(height)
    return height, width
