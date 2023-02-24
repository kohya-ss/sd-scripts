import argparse
from dataclasses import (
  asdict,
  dataclass,
)
import json
from pathlib import Path
from toolz import curry
from typing import (
  Optional,
  Sequence,
  Tuple,
  Union,
)

import toml
from voluptuous import (
  Any,
  ExactSequence,
  Required,
  Schema,
)
from transformers import CLIPTokenizer

from . import train_util
from .train_util import (
  DreamBoothSubset,
  FineTuningSubset,
  DreamBoothDataset,
  FineTuningDataset,
  DatasetGroup,
)


def add_config_arguments(parser: argparse.ArgumentParser):
  parser.add_argument("--config_file", type=Path, default=None, help="config file for detail settings / 詳細な設定用の設定ファイル")

# TODO: inherit Params class in Subset, Dataset

@dataclass
class BaseSubsetParams:
  image_dir: Optional[str] = None
  num_repeats: int = 1
  shuffle_caption: bool = False
  keep_tokens: int = 0
  color_aug: bool = False
  flip_aug: bool = False
  face_crop_aug_range: Optional[Tuple[float, float]] = None
  random_crop: bool = False
  caption_dropout_rate: float = 0.0
  caption_dropout_every_n_epochs: int = 0
  caption_tag_dropout_rate: float = 0.0

@dataclass
class DreamBoothSubsetParams(BaseSubsetParams):
  is_reg: bool = False
  class_tokens: Optional[str] = None
  caption_extension: str = ".caption"

@dataclass
class FineTuningSubsetParams(BaseSubsetParams):
  metadata_file: Optional[str] = None

@dataclass
class BaseDatasetParams:
  tokenizer: CLIPTokenizer = None
  max_token_length: int = None
  resolution: Optional[Tuple[int, int]] = None
  debug_dataset: bool = False

@dataclass
class DreamBoothDatasetParams(BaseDatasetParams):
  batch_size: int = 1
  enable_bucket: bool = False
  min_bucket_reso: int = 256
  max_bucket_reso: int = 1024
  bucket_reso_steps: int = 64
  bucket_no_upscale: bool = False
  prior_loss_weight: float = 1.0

@dataclass
class FineTuningDatasetParams(BaseDatasetParams):
  batch_size: int = 1
  enable_bucket: bool = False
  min_bucket_reso: int = 256
  max_bucket_reso: int = 1024
  bucket_reso_steps: int = 64
  bucket_no_upscale: bool = False

@dataclass
class SubsetBlueprint:
  params: Union[DreamBoothSubsetParams, FineTuningSubsetParams]

@dataclass
class DatasetBlueprint:
  is_dreambooth: bool
  params: Union[DreamBoothDatasetParams, FineTuningDatasetParams]
  subsets: Sequence[SubsetBlueprint]

@dataclass
class DatasetGroupBlueprint:
  datasets: Sequence[DatasetBlueprint]
@dataclass
class Blueprint:
  dataset_group: DatasetGroupBlueprint


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
      return ConfigSanitizer.__validate_and_convert_twodim(klass, value)

  # subset schema
  SUBSET_ASCENDABLE_SCHEMA = {
    "color_aug": bool,
    "face_crop_aug_range": __validate_and_convert_twodim(float),
    "flip_aug": bool,
    "num_repeats": int,
    "random_crop": bool,
    "shuffle_caption": bool,
    "keep_tokens": int,
  }
  SUBSET_DISTINCT_SCHEMA = {
    Required("image_dir"): str,
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
    Required("metadata_file"): str,
  }

  # datasets schema
  DATASET_ASCENDABLE_SCHEMA = {
    "batch_size": int,
    "bucket_no_upscale": bool,
    "bucket_reso_steps": int,
    "enable_bucket": bool,
    "max_bucket_reso": int,
    "min_bucket_reso": int,
    "resolution": __validate_and_convert_scalar_or_twodim(int),
  }

  # options handled by argparse but not handled by user config
  ARGPARSE_SPECIFIC_SCHEMA = {
    "debug_dataset": bool,
    "max_token_length": Any(None, int),
    "prior_loss_weight": float,
  }
  # for handling default None value of argparse
  ARGPARSE_NULLABLE_OPTNAMES = [
    "face_crop_aug_range",
    "resolution",
  ]
  # prepare map because option name may differ among argparse and user config
  ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME = {
    "train_batch_size": "batch_size",
    "dataset_repeats": "num_repeats",
  }

  def __init__(self, support_dreambooth: bool, support_finetuning: bool, support_dropout: bool) -> None:
    assert support_dreambooth or support_finetuning, "Neither DreamBooth mode nor fine tuning mode specified. Please specify one mode or more. / DreamBooth モードか fine tuning モードのどちらも指定されていません。1つ以上指定してください。"

    self.db_subset_schema = self.__merge_dict(
      self.SUBSET_DISTINCT_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DB_SUBSET_DISTINCT_SCHEMA,
      self.DB_SUBSET_ASCENDABLE_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
    )

    self.ft_subset_schema = self.__merge_dict(
      self.SUBSET_DISTINCT_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.FT_SUBSET_DISTINCT_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
    )

    self.db_dataset_schema = self.__merge_dict(
      self.DATASET_ASCENDABLE_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DB_SUBSET_ASCENDABLE_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
      {"subsets": [self.db_subset_schema]},
    )

    self.ft_dataset_schema = self.__merge_dict(
      self.DATASET_ASCENDABLE_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
      {"subsets": [self.ft_subset_schema]},
    )

    if support_dreambooth and support_finetuning:
      self.dataset_schema = Any(self.db_dataset_schema, self.ft_dataset_schema)
    elif support_dreambooth:
      self.dataset_schema = self.db_dataset_schema
    else:
      self.dataset_schema = self.ft_dataset_schema

    self.general_schema = self.__merge_dict(
      self.DATASET_ASCENDABLE_SCHEMA,
      self.SUBSET_ASCENDABLE_SCHEMA,
      self.DB_SUBSET_ASCENDABLE_SCHEMA if support_dreambooth else {},
      self.DO_SUBSET_ASCENDABLE_SCHEMA if support_dropout else {},
    )

    self.user_config_validator = Schema({
      "general": self.general_schema,
      "datasets": [self.dataset_schema],
    })

    self.argparse_schema = self.__merge_dict(
      self.general_schema,
      self.ARGPARSE_SPECIFIC_SCHEMA,
      {optname: Any(None, self.general_schema[optname]) for optname in self.ARGPARSE_NULLABLE_OPTNAMES}
    )

    self.argparse_config_validator = Schema(self.argparse_schema)

  # unify user config and argparse namespace into sanitized config
  # TODO: エラー発生時のメッセージをわかりやすくする
  def sanitize(self, user_config: dict, argparse_namespace: argparse.Namespace) -> dict:
    try:
      sanitized_user_config = self.user_config_validator(user_config)
    except Exception as e:
      print("Invalid user config / ユーザ設定の形式が正しくないようです")
      raise e

    # convert argparse namespace to dict like config
    # NOTE: list comprehension would be messy, so we use normal for statement
    argparse_config = {}
    for optname, value in vars(argparse_namespace).items():
      mapped_optname = self.ARGPARSE_OPTNAME_TO_CONFIG_OPTNAME.get(optname, optname)
      # only extract option in schema, because argparse contains many other options
      if mapped_optname in self.argparse_config_validator.schema:
        argparse_config[mapped_optname] = value

    try:
      sanitized_argparse_config = self.argparse_config_validator(argparse_config)
    except Exception as e:
      # XXX: this should be a bug
      print("Invalid cmdline parsed arguments. This should be a bug. / コマンドラインのパース結果が正しくないようです。プログラムのバグの可能性が高いです。")
      raise e

    return self.__merge_dict(sanitized_user_config, {"argparse": sanitized_argparse_config})

  # NOTE: value would be overwritten by latter dict if there is already the same key
  @staticmethod
  def __merge_dict(*dict_list: dict) -> dict:
    merged = {}
    for schema in dict_list:
      merged |= schema
    return merged


class BlueprintGenerator:
  BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME = {
  }

  def __init__(self, sanitizer: ConfigSanitizer):
    self.sanitizer = sanitizer

  # runtime_params is for parameters which is only configurable on runtime, such as tokenizer
  def generate(self, user_config: dict, argparse_namespace: argparse.Namespace, **runtime_params) -> Blueprint:
    sanitized_config = self.sanitizer.sanitize(user_config, argparse_namespace)

    general_config = sanitized_config.get("general", {})
    argparse_config = sanitized_config.get("argparse", {})

    dataset_blueprints = []
    for dataset_config in sanitized_config.get("datasets", []):
      # NOTE: if subsets have no "metadata_file", these are DreamBooth datasets/subsets
      subsets = dataset_config.get("subsets", [])
      is_dreambooth = all(["metadata_file" not in subset for subset in subsets])
      if is_dreambooth:
        subset_params_klass = DreamBoothSubsetParams
        dataset_params_klass = DreamBoothDatasetParams
      else:
        subset_params_klass = FineTuningSubsetParams
        dataset_params_klass = FineTuningDatasetParams

      subset_blueprints = []
      for subset_config in subsets:
        params = self.generate_params_by_fallbacks(subset_params_klass,
                                                   [subset_config, dataset_config, general_config, argparse_config, runtime_params])
        subset_blueprints.append(SubsetBlueprint(params))

      params = self.generate_params_by_fallbacks(dataset_params_klass,
                                                 [dataset_config, general_config, argparse_config, runtime_params])
      dataset_blueprints.append(DatasetBlueprint(is_dreambooth, params, subset_blueprints))

    dataset_group_blueprint = DatasetGroupBlueprint(dataset_blueprints)

    return Blueprint(dataset_group_blueprint)

  @staticmethod
  def generate_params_by_fallbacks(param_klass, fallbacks: Sequence[dict]):
    name_map = BlueprintGenerator.BLUEPRINT_PARAM_NAME_TO_CONFIG_OPTNAME
    search_value = BlueprintGenerator.search_value
    default_params = asdict(param_klass())
    param_names = default_params.keys()

    params = {name: search_value(name_map.get(name, name), fallbacks, default_params.get(name)) for name in param_names}

    return param_klass(**params)

  @staticmethod
  def search_value(key: str, fallbacks: Sequence[dict], default_value = None):
    for cand in fallbacks:
      value = cand.get(key)
      if value is not None:
        return value

    return default_value


def generate_dataset_group_by_blueprint(dataset_group_blueprint: DatasetGroupBlueprint):
  datasets= []
  for dataset_blueprint in dataset_group_blueprint.datasets:
    if dataset_blueprint.is_dreambooth:
      subset_klass = DreamBoothSubset
      dataset_klass = DreamBoothDataset
    else:
      subset_klass = FineTuningSubset
      dataset_klass = FineTuningDataset

    subsets = [subset_klass(**asdict(subset_blueprint.params)) for subset_blueprint in dataset_blueprint.subsets]
    dataset = dataset_klass(subsets=subsets, **asdict(dataset_blueprint.params))
    datasets.append(dataset)

  # make buckets first because it determines the length of dataset
  for dataset in datasets:
    dataset.make_buckets()

  return DatasetGroup(datasets)


def generate_dreambooth_subsets_config_by_subdirs(train_data_dir: Optional[str] = None, reg_data_dir: Optional[str] = None):
  def extract_dreambooth_params(name: str) -> Tuple[int, str]:
    tokens = name.split('_')
    try:
      n_repeats = int(tokens[0])
    except ValueError as e:
      print(f"ignore directory without repeats / 繰り返し回数のないディレクトリを無視します: {dir}")
      return 0, ""
    caption_by_folder = '_'.join(tokens[1:])
    return n_repeats, caption_by_folder

  def generate(base_dir: Optional[str], is_reg: bool):
    if base_dir is None:
      return []

    base_dir: Path = Path(base_dir)
    if not base_dir.is_dir():
      return []

    subsets_config = []
    for subdir in base_dir.iterdir():
      if not subdir.is_dir():
        continue

      num_repeats, class_tokens = extract_dreambooth_params(subdir.name)
      if num_repeats < 1:
        continue

      subset_config = {"image_dir": str(subdir), "is_reg": is_reg, "class_tokens": class_tokens}
      subsets_config.append(subset_config)

    return subsets_config

  subsets_config = []
  subsets_config += generate(train_data_dir, False)
  subsets_config += generate(reg_data_dir, True)

  return subsets_config


def load_user_config(file: str) -> dict:
  file: Path = Path(file)
  if not file.is_file():
    raise ValueError(f"file not found / ファイルが見つかりません: {file}")

  if file.name.lower().endswith('.json'):
    config = json.load(file)
  elif file.name.lower().endswith('.toml'):
    config = toml.load(file)
  else:
    raise ValueError(f"not supported config file format / 対応していない設定ファイルの形式です: {file}")

  return config


# for config test
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--support_dreambooth", action="store_true")
  parser.add_argument("--support_finetuning", action="store_true")
  parser.add_argument("--support_dropout", action="store_true")
  parser.add_argument("config_file")
  config_args, remain = parser.parse_known_args()

  parser = argparse.ArgumentParser()
  train_util.add_dataset_arguments(parser, config_args.support_dreambooth, config_args.support_finetuning, config_args.support_dropout)
  train_util.add_training_arguments(parser, config_args.support_dreambooth)
  argparse_namespace = parser.parse_args(remain)
  train_util.prepare_dataset_args(argparse_namespace, config_args.support_finetuning)

  user_config = load_user_config(config_args.config_file)

  sanitizer = ConfigSanitizer(config_args.support_dreambooth, config_args.support_finetuning, config_args.support_dropout)
  sanitized_config = sanitizer.sanitize(user_config, argparse_namespace)
  blueprint = BlueprintGenerator(sanitizer).generate(user_config, argparse_namespace)

  print("[argparse_namespace]")
  print(vars(argparse_namespace))
  print("\n[sanitized_config]")
  print(sanitized_config)
  print("\n[blueprint]")
  print(blueprint)
