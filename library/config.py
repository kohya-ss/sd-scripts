import json

import toml
from voluptuous import Schema, Optional, Required, Any, ExactSequence

# TODO: エラー発生時のメッセージをわかりやすくする
def verify_config(config: dict) -> dict:
    # common among general, item, subset
    subset_common_schema = Schema({
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
    })

    # common among general, item
    item_common_schema = Schema({
        Optional('batch_size'): int,
        Optional('bucket_no_upscale'): bool,
        Optional('bucket_reso_steps'): int,
        Optional('enable_bucket'): bool,
        Optional('max_bucket_reso'): int,
        Optional('min_bucket_reso'): int,
        Optional('resolution'): Any(int, ExactSequence([int, int])),
    })

    subset_schema = subset_common_schema.extend({
        Required('image_dir'): str,
        Optional('is_reg'): bool,
        Optional('class_tokens'): str,
    })

    item_schema = subset_common_schema.extend(item_common_schema.schema).extend(
        {Optional('subset'): [subset_schema], },
    )

    general_schema = subset_common_schema.extend(item_common_schema.schema).extend({
        Optional('debug_dataset'): bool,
    })

    dataset_schema = Schema({
        Optional('general'): general_schema,
        Optional('items'): [item_schema],
    })

    config_schema = Schema({
        Optional('dataset'): dataset_schema,
    })

    config_schema(config)

    return config


def load_config(fname: str) -> dict:
    if fname.lower().endswith('.json'):
        return verify_config(json.load(fname))
    elif fname.lower().endswith('.toml'):
        return verify_config(toml.load(fname))
    else:
        raise ValueError(f'not supported config file type: {fname}')
