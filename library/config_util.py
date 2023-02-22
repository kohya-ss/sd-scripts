import argparse
from typing import NamedTuple, Optional as TpOptional, Tuple

from voluptuous import Schema, Optional as VpOptional, Required, Any, ExactSequence, All


class SubsetParams(NamedTuple):
  image_dir: str
  num_repeats: int
  shuffle_caption: bool
  shuffle_keep_tokens: int
  cache_latents: bool
  color_aug: bool
  flip_aug: bool
  face_crop_aug_range: TpOptional[Tuple[float, float]]
  random_crop: bool
  caption_dropout_rate: float
  caption_dropout_every_n_epochs: TpOptional[int]
  caption_tag_dropout_rate: float

class ConfigManager:
  def __init__(self, support_dreambooth: bool, support_finetuning: bool, support_dropout: bool) -> None:
    assert support_dreambooth or support_finetuning, "Neither DreamBooth mode nor Fine tuning mode specified. This seems to be a bug. / バグである可能性が高いです"

    self.support_dreambooth = support_dreambooth
    self.support_finetuning = support_finetuning
    self.support_dropout = support_dropout

    self.general_opts = self.DATASET_DELEGATABLE_OPTIONS + self.SUBSET_DELEGATABLE_OPTIONS
    self.dreambooth_dataset_opts = self.DATASET_DELEGATABLE_OPTIONS + \
      self.DREAMBOOTH_SUBSET_DELEGATABLE_OPTIONS
    self.dreambooth_subset_opts = self.SUBSET_DISTINCT_OPTIONS + self.SUBSET_DELEGATABLE_OPTIONS + \
      self.DREAMBOOTH_SUBSET_DISTINCT_OPTIONS + \
        self.DREAMBOOTH_SUBSET_DELEGATABLE_OPTIONS
    self.finetuning_dataset_opts = self.DATASET_DELEGATABLE_OPTIONS
    self.finetuning_subset_opts = self.SUBSET_DISTINCT_OPTIONS + \
      self.SUBSET_DELEGATABLE_OPTIONS + self.FINETUNING_SUBSET_DISTINCT_OPTIONS

    if self.support_dreambooth:
      self.general_opts += self.DREAMBOOTH_SUBSET_DELEGATABLE_OPTIONS

    if self.support_dropout:
      self.general_opts += self.SUBSET_DROPOUT_DELEGATABLE_OPTIONS
      self.dreambooth_dataset_opts += self.SUBSET_DROPOUT_DELEGATABLE_OPTIONS
      self.dreambooth_subset_opts += self.SUBSET_DROPOUT_DELEGATABLE_OPTIONS
      self.finetuning_dataset_opts += self.SUBSET_DROPOUT_DELEGATABLE_OPTIONS
      self.finetuning_subset_opts += self.SUBSET_DROPOUT_DELEGATABLE_OPTIONS

  def convert_to_params(self, user_config: dict):
    self.verify_user_config(user_config)
    assert False, "convert_to_params is yet not implemented"

  # TODO: エラー発生時のメッセージをわかりやすくする
  def verify_user_config(self, user_config: dict):
    dreambooth_dataset_nested_opts = self.dreambooth_dataset_opts + {
      VpOptional('subset'): [self.dreambooth_subset_opts],
    }
    finetuning_dataset_nested_opts = self.finetuning_dataset_opts + {
      VpOptional('subset'): [self.finetuning_subset_opts],
    }

    if self.support_dreambooth and self.support_finetuning:
      dataset_opts = Any([dreambooth_dataset_nested_opts], [finetuning_dataset_nested_opts])
    elif self.support_dreambooth:
      dataset_opts = [dreambooth_dataset_nested_opts]
    else:
      dataset_opts = [finetuning_dataset_nested_opts]

    user_config_options = {
      VpOptional('general'): self.general_opts,
      VpOptional('dataset'): dataset_opts,
    }

    Schema(user_config_options)(self.user_config)

  # subset options
  SUBSET_DELEGATABLE_OPTIONS = {
    VpOptional('cache_latents'): bool,
    VpOptional('color_aug'): bool,
    VpOptional('face_crop_aug_range'): ExactSequence([float, float]),
    VpOptional('flip_aug'): bool,
    VpOptional('num_repeats'): int,
    VpOptional('random_crop'): bool,
    VpOptional('shuffle_caption'): bool,
    VpOptional('shuffle_keep_tokens'): int,
  }
  SUBSET_DISTINCT_OPTIONS = {
    Required('image_dir'): str,
  }
  SUBSET_DROPOUT_DELEGATABLE_OPTIONS = {
    VpOptional('caption_dropout_every_n_epochs'): int,
    VpOptional('caption_dropout_rate'): float,
    VpOptional('caption_tag_dropout_rate'): float,
  }
  DREAMBOOTH_SUBSET_DELEGATABLE_OPTIONS = {
    VpOptional('caption_extension'): str,
  }
  DREAMBOOTH_SUBSET_DISTINCT_OPTIONS = {
    VpOptional('is_reg'): bool,
    VpOptional('class_tokens'): str,
  }
  FINETUNING_SUBSET_DISTINCT_OPTIONS = {
    VpOptional('in_json'): str,
  }

  # dataset options
  DATASET_DELEGATABLE_OPTIONS = {
    VpOptional('batch_size'): int,
    VpOptional('bucket_no_upscale'): bool,
    VpOptional('bucket_reso_steps'): int,
    VpOptional('enable_bucket'): bool,
    VpOptional('max_bucket_reso'): int,
    VpOptional('min_bucket_reso'): int,
    VpOptional('resolution'): Any(int, ExactSequence([int, int])),
  }

def add_config_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--config", type=str, default=None, help="config file for detail settings (only supports DreamBooth method) / 詳細な設定用の設定ファイル（DreamBooth の手法のみ対応）")
