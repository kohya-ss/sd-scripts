import argparse
from pathlib import Path
from toolz import curry
from typing import (
  NamedTuple,
  Optional,
  Sequence,
  Tuple,
  Union,
)

from voluptuous import (
  Any,
  ExactSequence,
  Required,
  Schema,
)

import file_util
from train_util import add_dataset_arguments, add_training_arguments


def add_config_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--config_file", type=Path, default=None, help="config file for detail settings / 詳細な設定用の設定ファイル")

class BaseSubsetParams(NamedTuple):
  image_dir: str
  num_repeats: int
  shuffle_caption: bool
  shuffle_keep_tokens: int
  cache_latents: bool
  color_aug: bool
  flip_aug: bool
  face_crop_aug_range: Optional[Tuple[float, float]]
  random_crop: bool
  caption_dropout_rate: float
  caption_dropout_every_n_epochs: Optional[int]
  caption_tag_dropout_rate: float

class DreamBoothSubsetParams(BaseSubsetParams):
  caption_extension: str
  is_reg: bool
  class_tokens: str

class FineTuningSubsetParams(BaseSubsetParams):
  in_json: str

class BaseDatasetParams(NamedTuple):
  resolution: Optional[Tuple[int, int]]

class DreamBoothDatasetParams(BaseDatasetParams):
  batch_size: int
  enable_bucket: bool
  min_bucket_reso: int
  max_bucket_reso: int
  bucket_reso_steps: int
  bucket_no_upscale: bool

class FineTuningDatasetParams(BaseDatasetParams):
  batch_size: int
  enable_bucket: bool
  min_bucket_reso: int
  max_bucket_reso: int
  bucket_reso_steps: int
  bucket_no_upscale: bool

class SubsetBlueprint(NamedTuple):
  params: Union[DreamBoothSubsetParams, FineTuningSubsetParams]

class DatasetBlueprint(NamedTuple):
  is_dreambooth: bool
  params: Union[DreamBoothDatasetParams, FineTuningDatasetParams]
  subsets: Sequence[SubsetBlueprint] = []

class Blueprint(NamedTuple):
  datasets: Sequence[DatasetBlueprint] = []

class ConfigSanitizer:
  @curry
  @staticmethod
  def __validate_and_convert_twodim(klass, value: Sequence) -> Tuple:
    Schema(ExactSequence([klass, klass]))(value)
    return tuple(value)

  @curry
  @staticmethod
  def __validate_and_convert_scalar_or_twodim(klass, value: Union[float, Sequence]) -> Tuple:
    Schema(Any(klass, ExactSequence([klass, klass])))(value)
    try:
      Schema(klass)(value)
      return (value, value)
    except:
      return ConfigSanitizer.__validate_and_convert_twodim(value, klass)

  # subset options
  SUBSET_ASCENDABLE_SCHEMA = {
    "cache_latents": bool,
    "color_aug": bool,
    "face_crop_aug_range": __validate_and_convert_twodim(float),
    "flip_aug": bool,
    "num_repeats": int,
    "random_crop": bool,
    "shuffle_caption": bool,
    "shuffle_keep_tokens": int,
  }
  SUBSET_DISTINCT_SCHEMA = {
    Required('image_dir'): str,
  }
  # DO means DropOut
  DO_SUBSET_ASCENDABLE_SCHEMA = {
    "caption_dropout_every_n_epochs": int,
    "caption_dropout_rate": float,
    "caption_tag_dropout_rate": float,
  }
  # DB means DreamBooth
  DB_SUBSET_ASCENDABLE_SCHEMA = {
    "caption_extension": str,
    "class_tokens": str,
  }
  DB_SUBSET_DISTINCT_SCHEMA = {
    "is_reg": bool,
  }
  # FT means FineTuning
  FT_SUBSET_DISTINCT_SCHEMA = {
    Required('in_json'): str,
  }

  # datasets options
  DATASET_ASCENDABLE_SCHEMA = {
    "batch_size": int,
    "bucket_no_upscale": bool,
    "bucket_reso_steps": int,
    "enable_bucket": bool,
    "max_bucket_reso": int,
    "min_bucket_reso": int,
    "resolution": __validate_and_convert_scalar_or_twodim(int),
  }

  def __init__(self, support_dreambooth: bool, support_finetuning: bool, support_dropout: bool) -> None:
    assert support_dreambooth or support_finetuning, "Neither DreamBooth mode nor fine tuning mode specified. Please specify one mode or more. / DreamBooth モードか fine tuning モードのどちらも指定されていません。1つ以上指定してください。"

    self.support_dreambooth = support_dreambooth
    self.support_finetuning = support_finetuning
    self.support_dropout = support_dropout

    def merge_schema(schema_list: Sequence[dict]) -> dict:
      merged = {}
      for schema in schema_list:
        merged |= schema
      return merged

    self.db_subset_schema = merge_schema([
        self.SUBSET_DISTINCT_SCHEMA,
        self.SUBSET_ASCENDABLE_SCHEMA,
        self.DB_SUBSET_DISTINCT_SCHEMA,
        self.DB_SUBSET_ASCENDABLE_SCHEMA,
      ] + ([self.DO_SUBSET_ASCENDABLE_SCHEMA] if support_dropout else []))

    self.ft_subset_schema = merge_schema([
        self.SUBSET_DISTINCT_SCHEMA,
        self.SUBSET_ASCENDABLE_SCHEMA,
        self.FT_SUBSET_DISTINCT_SCHEMA,
      ] + ([self.DO_SUBSET_ASCENDABLE_SCHEMA] if support_dropout else []))

    self.db_dataset_schema = merge_schema([
        self.DATASET_ASCENDABLE_SCHEMA,
        self.SUBSET_ASCENDABLE_SCHEMA,
        self.DB_SUBSET_ASCENDABLE_SCHEMA,
      ] + ([self.DO_SUBSET_ASCENDABLE_SCHEMA] if support_dropout else []))

    self.ft_dataset_schema = merge_schema([
        self.DATASET_ASCENDABLE_SCHEMA,
        self.SUBSET_ASCENDABLE_SCHEMA,
      ] + ([self.DO_SUBSET_ASCENDABLE_SCHEMA] if support_dropout else []))

    self.general_schema = merge_schema([
        self.DATASET_ASCENDABLE_SCHEMA,
        self.SUBSET_ASCENDABLE_SCHEMA,
      ] + ([self.DO_SUBSET_ASCENDABLE_SCHEMA] if support_dropout else []) \
        + ([self.DB_SUBSET_ASCENDABLE_SCHEMA] if support_dreambooth else []))

  # TODO: エラー発生時のメッセージをわかりやすくする
  def sanitize(self, user_config: dict):
    db_dataset_nested_schema = self.db_dataset_schema | {
      "subsets": [self.db_subset_schema],
    }
    ft_dataset_nested_schema = self.ft_dataset_schema | {
      "subsets": [self.ft_subset_schema],
    }

    if self.support_dreambooth and self.support_finetuning:
      dataset_schema = Any(db_dataset_nested_schema, ft_dataset_nested_schema)
    elif self.support_dreambooth:
      dataset_schema = db_dataset_nested_schema
    else:
      dataset_schema = ft_dataset_nested_schema

    validator = Schema({
      "general": self.general_schema,
      "datasets": [dataset_schema],
    })

    return validator(user_config)

class BlueprintGenerator:
  TO_ARGPARSE_OPTNAME = {
    "shuffle_keep_tokens": "keep_tokens",
  }

  def __init__(self) -> None:
      pass

  # TODO: 書く
  def generate_blueprint(self, sanitized_config: dict, parsed_args: argparse.Namespace) -> dict:
    return {}

    parsed_args = vars(parsed_args)
    general_config = sanitized_config.get("general", {})

    dataset_blueprints = []
    for dataset_config in sanitized_config.get("datasets", []):
      # NOTE: for checking whether it is dreambooth dataset, searching "in_json" key seems to be the best way so far
      is_dreambooth = not any(["in_json" in dataset_config["subsets"]])

      subset_blueprints = []
      for subset_config in dataset_config["subsets"]:
        subset_blueprint = SubsetBlueprint()

      dataset_blueprint = DatasetBlueprint(is_dreambooth, dataset_params, subset_blueprints)
      dataset_blueprints.append(dataset_blueprint)

    return Blueprint(dataset_blueprints)

  def get_value(self, key: str, candidates: Sequence[dict], parsed_args: dict):
    # search candidates first
    for cand in candidates:
      value = cand.get(key)
      if value is not None:
        return value

    # search parsed arguments
    # we need to map option name because option name may differ among config and argparse
    mapped_key = self.TO_ARGPARSE_OPTNAME.get(key, key)
    return parsed_args.get(mapped_key)

# for config test
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--support_dreambooth", action="store_true")
  parser.add_argument("--support_finetuning", action="store_true")
  parser.add_argument("--support_dropout", action="store_true")
  parser.add_argument("config_file")
  config_args, remain = parser.parse_known_args()

  parser = argparse.ArgumentParser()
  add_dataset_arguments(parser, config_args.support_dreambooth, config_args.support_finetuning, config_args.support_dropout)
  add_training_arguments(parser, config_args.support_dreambooth)
  train_args = parser.parse_args(remain)

  config = file_util.load_user_config(config_args.config_file)

  sanitizer = ConfigSanitizer(config_args.support_dreambooth, config_args.support_finetuning, config_args.support_dropout)

  print("[train_args]")
  print(vars(train_args))
  print("\n[sanitized config]")
  print(sanitizer.sanitize(config))

