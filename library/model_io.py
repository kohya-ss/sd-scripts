"""Model file I/O, hashing and metadata helpers.

Hosts the routines that load Stable Diffusion / Diffusers checkpoints into
the in-memory training pipeline (``load_target_model`` / ``_load_target_model``),
the SD-WebUI / additional-networks compatible hashes (``model_hash``,
``calculate_sha256``, ``addnet_hash_*``, ``precalculate_safetensors_hashes``),
the ``ss_*`` LoRA metadata helpers (``build_minimum_network_metadata`` and the
``SS_METADATA_*`` keys), the SAI ModelSpec wrappers
(``get_sai_model_spec``, ``get_sai_model_spec_dataclass``), the
``replace_unet_modules`` attention-implementation switch, the safetensors
metadata reader (``load_metadata_from_safetensors``) and
``get_git_revision_hash``. Extracted from ``library.train_util`` and
re-exported there for backward compatibility.
"""

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from io import BytesIO
from typing import TYPE_CHECKING, Optional

import safetensors
import safetensors.torch

import library.sai_model_spec as sai_model_spec
from library.device_utils import clean_memory_on_device
from library.utils import setup_logging

if TYPE_CHECKING:
    from library.original_unet import UNet2DConditionModel

# NOTE: diffusers / library.model_util / library.original_unet are imported lazily
# inside the load_target_model functions: importing them here adds several seconds
# of startup time to lightweight CLI tools (networks/*) that only need the
# hashing / metadata helpers from this module.

setup_logging()

logger = logging.getLogger(__name__)


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
    except:
        return "(unknown)"


def replace_unet_modules(unet: "UNet2DConditionModel", mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops  # noqa: F401
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


# this metadata is referred from train_network and various scripts, so we wrote here
SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def build_minimum_network_metadata(
    v2: Optional[str],
    base_model: Optional[str],
    network_module: str,
    network_dim: str,
    network_alpha: str,
    network_args: Optional[dict],
):
    # old LoRA doesn't have base_model
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def get_sai_model_spec(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,  # None for TI and LoRA
    sd3: str = None,
    flux: str = None,  # "dev", "schnell" or "chroma"
    lumina: str = None,
    optional_metadata: dict[str, str] | None = None,
):
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    # Convert individual model parameters to model_config dict
    # TODO: Update calls to this function to pass in the model config
    model_config = {}
    if sd3 is not None:
        model_config["sd3"] = sd3
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina

    # Extract metadata_* fields from args and merge with optional_metadata
    extracted_metadata = {}

    # Extract all metadata_* attributes from args
    for attr_name in dir(args):
        if attr_name.startswith("metadata_") and not attr_name.startswith("metadata___"):
            value = getattr(args, attr_name, None)
            if value is not None:
                # Remove metadata_ prefix and exclude already handled fields
                field_name = attr_name[9:]  # len("metadata_") = 9
                if field_name not in ["title", "author", "description", "license", "tags"]:
                    extracted_metadata[field_name] = value

    # Merge extracted metadata with provided optional_metadata
    all_optional_metadata = {**extracted_metadata}
    if optional_metadata:
        all_optional_metadata.update(optional_metadata)

    metadata = sai_model_spec.build_metadata(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,  # None or int
        model_config=model_config,
        optional_metadata=all_optional_metadata if all_optional_metadata else None,
    )
    return metadata


def get_sai_model_spec_dataclass(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,
    sd3: str = None,
    flux: str = None,
    lumina: str = None,
    hunyuan_image: str = None,
    anima: str = None,
    optional_metadata: dict[str, str] | None = None,
) -> sai_model_spec.ModelSpecMetadata:
    """
    Get ModelSpec metadata as a dataclass - preferred for new code.
    Automatically extracts metadata_* fields from args.
    """
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    # Convert individual model parameters to model_config dict
    model_config = {}
    if sd3 is not None:
        model_config["sd3"] = sd3
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina
    if hunyuan_image is not None:
        model_config["hunyuan_image"] = hunyuan_image
    if anima is not None:
        model_config["anima"] = anima
    # Use the dataclass function directly
    return sai_model_spec.build_metadata_dataclass(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,
        model_config=model_config,
        optional_metadata=optional_metadata,
    )


def _load_target_model(args: argparse.Namespace, weight_dtype, device="cpu", unet_use_linear_projection_in_v2=False):
    from diffusers import StableDiffusionPipeline

    import library.model_util as model_util
    from library.original_unet import UNet2DConditionModel

    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
            args.v2, name_or_path, device, unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2
        )
    else:
        # Diffusers model is loaded to CPU
        logger.info(f"load Diffusers pretrained models: {name_or_path}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path}"
            )
            raise ex
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        # TODO *.ckpt/*.safetensorsのv2と同じ形式にここで変換すると良さそう
        # logger.info(f"unet config: {unet.config}")
        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
        logger.info("U-Net converted to original U-Net")

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        logger.info("additional VAE loaded")

    return text_encoder, vae, unet, load_stable_diffusion_format


def load_target_model(args, weight_dtype, accelerator, unet_use_linear_projection_in_v2=False):
    import library.model_util as model_util

    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                args,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
            )

            # Expand 4-channel conv_in to 9 channels when training inpainting from a
            # standard (non-inpainting) checkpoint.
            if getattr(args, "train_inpainting", False) and getattr(unet, "in_channels", 4) == 4:
                logger.info(
                    "train_inpainting: expanding UNet conv_in from 4 to 9 channels "
                    "(standard checkpoint → inpainting training from scratch)"
                )
                model_util.expand_unet_to_inpainting(unet)

            # work on low-ram device
            if args.lowram:
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()
    return text_encoder, vae, unet, load_stable_diffusion_format
