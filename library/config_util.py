from voluptuous import Schema, Optional, Required, Any, ExactSequence

class ConfigManager:
    def __init__(self, user_config: dict) -> None:
        self.user_config = user_config

    # TODO: エラー発生時のメッセージをわかりやすくする
    def verify_user_config(self):
        # common among general, dataset, subset
        subset_common_options = {
            Optional('caption_dropout_every_n_epochs'): int,
            Optional('caption_dropout_rate'): float,
            Optional('caption_extension'): str,
            Optional('caption_tag_dropout_rate'): float,
            Optional('color_aug'): bool,
            Optional('face_crop_aug_range'): ExactSequence([float, float]),
            Optional('flip_aug'): bool,
            Optional('keep_tokens'): int,
            Optional('num_repeats'): int,
            Optional('random_crop'): bool,
            Optional('shuffle_caption'): bool,
        }

        # common among general, dataset
        dataset_common_options = {
            Optional('cache_latents'): bool,
            Optional('bucket_no_upscale'): bool,
            Optional('bucket_reso_steps'): int,
            Optional('enable_bucket'): bool,
            Optional('max_bucket_reso'): int,
            Optional('min_bucket_reso'): int,
            Optional('resolution'): Any(int, ExactSequence([int, int])),
            Optional('train_batch_size'): int,
        }

        subset_options = subset_common_options + {
            Required('image_dir'): str,
            Optional('is_reg'): bool,
            Optional('class_tokens'): str,
        }

        dataset_options = subset_common_options + dataset_common_options + \
            {Optional('subset'): [subset_options], }

        general_options = subset_common_options + dataset_common_options

        user_config_options = {
            Optional('general'): general_options,
            Optional('dataset'): [dataset_options],
        }

        Schema(user_config_options)(self.user_config)
