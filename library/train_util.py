# common functions for training

import logging

from library.device_utils import init_ipex, clean_memory_on_device  # noqa: F401  (clean_memory_on_device re-exported for backward compatibility)
from library.utils import setup_logging

init_ipex()
setup_logging()

logger = logging.getLogger(__name__)


# Accelerator setup helpers have moved to library.accelerator_setup;
# re-exported here for backward compatibility.
# New code should import from library.accelerator_setup directly.
# HIGH_VRAM is mutated by enable_high_vram(); for legacy ``train_util.HIGH_VRAM``
# attribute reads we forward through a module-level __getattr__ below.
from library.accelerator_setup import (  # noqa: F401, E402
    enable_high_vram,
    prepare_dataset_args,
    prepare_accelerator,
    prepare_dtype,
    patch_accelerator_for_fp16_training,
)


def __getattr__(name):
    if name == "HIGH_VRAM":
        from library import accelerator_setup
        return accelerator_setup.HIGH_VRAM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Checkpoint filename templates and save / rotate helpers have moved to
# library.checkpoint_io; re-exported here for backward compatibility.
# New code should import from library.checkpoint_io directly.
from library.checkpoint_io import (  # noqa: F401, E402
    EPOCH_STATE_NAME,
    EPOCH_FILE_NAME,
    EPOCH_DIFFUSERS_DIR_NAME,
    LAST_STATE_NAME,
    DEFAULT_EPOCH_NAME,
    DEFAULT_LAST_OUTPUT_NAME,
    DEFAULT_STEP_NAME,
    STEP_STATE_NAME,
    STEP_FILE_NAME,
    STEP_DIFFUSERS_DIR_NAME,
)

# Dataset core has moved to library.dataset; re-exported here for backward compatibility.
# New code should import from library.dataset directly.
from library.dataset import (  # noqa: F401, E402
    IMAGE_EXTENSIONS,
    IMAGE_TRANSFORMS,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
    TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
    split_train_val,
    ImageInfo,
    BucketManager,
    BucketBatchIndex,
    AugHelper,
    BaseDataset,
    DatasetGroup,
    MinimalDataset,
    load_arbitrary_dataset,
    debug_dataset,
    glob_images,
    glob_images_pathlib,
    load_image,
    collator_class,
)


# Subset classes have moved to library.subset; re-exported here for backward compatibility.
# New code should import from library.subset directly.
from library.subset import BaseSubset, DreamBoothSubset, FineTuningSubset, ControlNetSubset  # noqa: F401, E402


# DreamBooth / FineTuning / ControlNet datasets have moved to dedicated modules;
# re-exported here for backward compatibility. New code should import from library.* directly.
from library.dreambooth_dataset import DreamBoothDataset  # noqa: F401, E402
from library.finetuning_dataset import FineTuningDataset  # noqa: F401, E402
from library.controlnet_dataset import ControlNetDataset  # noqa: F401, E402


# Caching functions have moved to library.caching; re-exported here for backward compatibility.
# New code should import from library.caching directly.
from library.caching import (  # noqa: F401, E402
    is_disk_cached_latents_is_expected,
    trim_and_resize_if_required,
    load_images_and_masks_for_caching,
    cache_batch_latents,
)


# Model I/O, hashing and metadata helpers have moved to library.model_io;
# re-exported here for backward compatibility.
# New code should import from library.model_io directly.
from library.model_io import (  # noqa: F401, E402
    model_hash,
    calculate_sha256,
    precalculate_safetensors_hashes,
    addnet_hash_legacy,
    addnet_hash_safetensors,
    get_git_revision_hash,
    replace_unet_modules,
    load_metadata_from_safetensors,
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
    SS_METADATA_MINIMUM_KEYS,
    build_minimum_network_metadata,
    get_sai_model_spec,
    get_sai_model_spec_dataclass,
    _load_target_model,
    load_target_model,
)


# Argument definitions and configuration helpers have moved to library.args;
# re-exported here for backward compatibility.
# New code should import from library.args directly.
from library.args import (  # noqa: F401, E402
    add_sd_models_arguments,
    add_optimizer_arguments,
    add_training_arguments,
    add_masked_loss_arguments,
    add_dit_training_arguments,
    get_sanitized_config_or_none,
    verify_command_line_training_args,
    verify_training_args,
    add_dataset_arguments,
    add_sd_saving_arguments,
    read_config_from_file,
    resume_from_local_or_hf_if_specified,
)


# Optimizer / scheduler / LR-logging helpers have moved to library.optimizer;
# re-exported here for backward compatibility.
# New code should import from library.optimizer directly.
from library.optimizer import (  # noqa: F401, E402
    get_optimizer,
    get_optimizer_train_eval_fn,
    is_schedulefree_optimizer,
    get_dummy_scheduler,
    get_scheduler_fix,
    append_lr_to_logs,
    append_lr_to_logs_with_names,
)


# Text encoder hidden-state helpers have moved to library.hidden_states;
# re-exported here for backward compatibility.
# New code should import from library.hidden_states directly.
from library.hidden_states import (  # noqa: F401, E402
    get_hidden_states,
    pool_workaround,
    get_hidden_states_sdxl,
)


# Checkpoint save / rotate helpers have moved to library.checkpoint_io;
# re-exported here for backward compatibility.
# New code should import from library.checkpoint_io directly.
from library.checkpoint_io import (  # noqa: F401, E402
    default_if_none,
    get_epoch_ckpt_name,
    get_step_ckpt_name,
    get_last_ckpt_name,
    get_remove_epoch_no,
    get_remove_step_no,
    save_sd_model_on_epoch_end_or_stepwise,
    save_sd_model_on_epoch_end_or_stepwise_common,
    save_and_remove_state_on_epoch_end,
    save_and_remove_state_stepwise,
    save_state_on_train_end,
    save_sd_model_on_train_end,
    save_sd_model_on_train_end_common,
)


# Loss / noise scheduling helpers have moved to library.loss;
# re-exported here for backward compatibility.
# New code should import from library.loss directly.
from library.loss import (  # noqa: F401, E402
    get_timesteps,
    get_noise_noisy_latents_and_timesteps,
    get_huber_threshold_if_needed,
    conditional_loss,
)


# Sampling helpers (default scheduler, prompt parsing, sample generation)
# have moved to library.sampling; re-exported here for backward compatibility.
# New code should import from library.sampling directly.
from library.sampling import (  # noqa: F401, E402
    SCHEDULER_LINEAR_START,
    SCHEDULER_LINEAR_END,
    SCHEDULER_TIMESTEPS,
    SCHEDLER_SCHEDULE,
    get_my_scheduler,
    sample_images,
    line_to_prompt_dict,
    load_prompts,
    sample_images_common,
    sample_image_inference,
)


# Logging / tracker helpers have moved to library.logging_util;
# re-exported here for backward compatibility.
# New code should import from library.logging_util directly.
from library.logging_util import init_trackers, LossRecorder  # noqa: F401, E402
