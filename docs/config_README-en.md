Original Source by kohya-ss

A.I Translation by Model: NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO, editing by Darkstorm2150

# Config Readme

This README is about the configuration files that can be passed with the `--dataset_config` option.

## Overview

By passing a configuration file, users can make detailed settings.

* Multiple datasets can be configured
   * For example, by setting `resolution` for each dataset, they can be mixed and trained.
   * In training methods that support both the DreamBooth approach and the fine-tuning approach, datasets of the DreamBooth method and the fine-tuning method can be mixed.
* Settings can be changed for each subset
   * A subset is a partition of the dataset by image directory or metadata. Several subsets make up a dataset.
   * Options such as `keep_tokens` and `flip_aug` can be set for each subset. On the other hand, options such as `resolution` and `batch_size` can be set for each dataset, and their values are common among subsets belonging to the same dataset. More details will be provided later.

The configuration file format can be JSON or TOML. Considering the ease of writing, it is recommended to use [TOML](https://toml.io/ja/v1.0.0-rc.2). The following explanation assumes the use of TOML.


Here is an example of a configuration file written in TOML.

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# This is a DreamBooth-style dataset
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # This subset uses keep_tokens = 2 (the value of the parent datasets)

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# This is a fine-tuning dataset
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # This subset uses keep_tokens = 1 (the value of [general])
```

In this example, three directories are trained as a DreamBooth-style dataset at 512x512 (batch size 4), and one directory is trained as a fine-tuning dataset at 768x768 (batch size 2).

## Settings for datasets and subsets

Settings for datasets and subsets are divided into several registration locations.

* `[general]`
    * This is where options that apply to all datasets or all subsets are specified.
    * If there are options with the same name in the dataset-specific or subset-specific settings, the dataset-specific or subset-specific settings take precedence.
* `[[datasets]]`
    * `datasets` is where settings for datasets are registered. This is where options that apply individually to each dataset are specified.
	* If there are subset-specific settings, the subset-specific settings take precedence.
* `[[datasets.subsets]]`
    * `datasets.subsets` is where settings for subsets are registered. This is where options that apply individually to each subset are specified.

Here is an image showing the correspondence between image directories and registration locations in the previous example.

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

The image directory corresponds to each `[[datasets.subsets]]`. Then, multiple `[[datasets.subsets]]` are combined to form one `[[datasets]]`. All `[[datasets]]` and `[[datasets.subsets]]` belong to `[general]`.

The available options for each registration location may differ, but if the same option is specified, the value in the lower registration location will take precedence. You can check how the `keep_tokens` option is handled in the previous example for better understanding.

Additionally, the available options may vary depending on the method that the learning approach supports.

* Options specific to the DreamBooth method
* Options specific to the fine-tuning method
* Options available when using the caption dropout technique

When using both the DreamBooth method and the fine-tuning method, they can be used together with a learning approach that supports both.
When using them together, a point to note is that the method is determined based on the dataset, so it is not possible to mix DreamBooth method subsets and fine-tuning method subsets within the same dataset.
In other words, if you want to use both methods together, you need to set up subsets of different methods belonging to different datasets.

In terms of program behavior, if the `metadata_file` option exists, it is determined to be a subset of fine-tuning. Therefore, for subsets belonging to the same dataset, as long as they are either "all have the `metadata_file` option" or "all have no `metadata_file` option," there is no problem.

Below, the available options will be explained. For options with the same name as the command-line argument, the explanation will be omitted in principle. Please refer to other READMEs.

### Common options for all learning methods

These are options that can be specified regardless of the learning method.

#### Data set specific options

These are options related to the configuration of the data set. They cannot be described in `datasets.subsets`.


| Option Name | Example Setting | `[general]` | `[[datasets]]` |
| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * This corresponds to the command-line argument `--train_batch_size`.

These settings are fixed per dataset. That means that subsets belonging to the same dataset will share these settings. For example, if you want to prepare datasets with different resolutions, you can define them as separate datasets as shown in the example above, and set different resolutions for each.

#### Options for Subsets

These options are related to subset configuration.

| Option Name | Example | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |
| `caption_prefix` | `"masterpiece, best quality, "` | o | o | o |
| `caption_suffix` | `", from side"` | o | o | o |

* `num_repeats`
    * Specifies the number of repeats for images in a subset. This is equivalent to `--dataset_repeats` in fine-tuning but can be specified for any training method.
* `caption_prefix`, `caption_suffix`
    * Specifies the prefix and suffix strings to be appended to the captions. Shuffling is performed with these strings included. Be cautious when using `keep_tokens`.

### DreamBooth-specific options

DreamBooth-specific options only exist as subsets-specific options.

#### Subset-specific options

Options related to the configuration of DreamBooth subsets.

| Option Name | Example Setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `'C:\hoge'` | - | - | o (required) |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `"sks girl"` | - | - | o |
| `is_reg` | `false` | - | - | o |

Firstly, note that for `image_dir`, the path to the image files must be specified as being directly in the directory. Unlike the previous DreamBooth method, where images had to be placed in subdirectories, this is not compatible with that specification. Also, even if you name the folder something like "5_cat", the number of repeats of the image and the class name will not be reflected. If you want to set these individually, you will need to explicitly specify them using `num_repeats` and `class_tokens`.

* `image_dir`
    * Specifies the path to the image directory. This is a required option.
    * Images must be placed directly under the directory.
* `class_tokens`
    * Sets the class tokens.
    * Only used during training when a corresponding caption file does not exist. The determination of whether or not to use it is made on a per-image basis. If `class_tokens` is not specified and a caption file is not found, an error will occur.
* `is_reg`
    * Specifies whether the subset images are for normalization. If not specified, it is set to `false`, meaning that the images are not for normalization.

### Fine-tuning method specific options

The options for the fine-tuning method only exist for subset-specific options.

#### Subset-specific options

These options are related to the configuration of the fine-tuning method's subsets.

| Option name | Example setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `'C:\hoge'` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o (required) |

* `image_dir`
    * Specify the path to the image directory. Unlike the DreamBooth method, specifying it is not mandatory, but it is recommended to do so.
        * The case where it is not necessary to specify is when the `--full_path` is added to the command line when generating the metadata file.
    * The images must be placed directly under the directory.
* `metadata_file`
    * Specify the path to the metadata file used for the subset. This is a required option.
        * It is equivalent to the command-line argument `--in_json`.
    * Due to the specification that a metadata file must be specified for each subset, it is recommended to avoid creating a metadata file with images from different directories as a single metadata file. It is strongly recommended to prepare a separate metadata file for each image directory and register them as separate subsets.

### Options available when caption dropout method can be used

The options available when the caption dropout method can be used exist only for subsets. Regardless of whether it's the DreamBooth method or fine-tuning method, if it supports caption dropout, it can be specified.

#### Subset-specific options

Options related to the setting of subsets that caption dropout can be used for.

| Option Name | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## Behavior when there are duplicate subsets

In the case of the DreamBooth dataset, if there are multiple `image_dir` directories with the same content, they are considered to be duplicate subsets. For the fine-tuning dataset, if there are multiple `metadata_file` files with the same content, they are considered to be duplicate subsets. If duplicate subsets exist in the dataset, subsequent subsets will be ignored.

However, if they belong to different datasets, they are not considered duplicates. For example, if you have subsets with the same `image_dir` in different datasets, they will not be considered duplicates. This is useful when you want to train with the same image but with different resolutions.

```toml
# If data sets exist separately, they are not considered duplicates and are both used for training.

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## Command Line Argument and Configuration File

There are options in the configuration file that have overlapping roles with command line argument options.

The following command line argument options are ignored if a configuration file is passed:

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

The following command line argument options are given priority over the configuration file options if both are specified simultaneously. In most cases, they have the same names as the corresponding options in the configuration file.

| Command Line Argument Option   | Prioritized Configuration File Option |
| ------------------------------- | ------------------------------------- |
| `--bucket_no_upscale`           |                                       |
| `--bucket_reso_steps`           |                                       |
| `--caption_dropout_every_n_epochs` |                                       |
| `--caption_dropout_rate`        |                                       |
| `--caption_extension`           |                                       |
| `--caption_tag_dropout_rate`    |                                       |
| `--color_aug`                   |                                       |
| `--dataset_repeats`             | `num_repeats`                          |
| `--enable_bucket`               |                                       |
| `--face_crop_aug_range`         |                                       |
| `--flip_aug`                    |                                       |
| `--keep_tokens`                 |                                       |
| `--min_bucket_reso`              |                                       |
| `--random_crop`                 |                                       |
| `--resolution`                  |                                       |
| `--shuffle_caption`             |                                       |
| `--train_batch_size`            | `batch_size`                           |

## Error Guide

Currently, we are using an external library to check if the configuration file is written correctly, but the development has not been completed, and there is a problem that the error message is not clear. In the future, we plan to improve this problem.

As a temporary measure, we will list common errors and their solutions. If you encounter an error even though it should be correct or if the error content is not understandable, please contact us as it may be a bug.

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: This error occurs when a required option is not provided. It is highly likely that you forgot to specify the option or misspelled the option name.
  * The error location is indicated by `...` in the error message. For example, if you encounter an error like `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']`, it means that the `image_dir` option does not exist in the 0th `subsets` of the 0th `datasets` setting.
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: This error occurs when the specified value format is incorrect. It is highly likely that the value format is incorrect. The `int` part changes depending on the target option. The example configurations in this README may be helpful.
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: This error occurs when there is an option name that is not supported. It is highly likely that you misspelled the option name or mistakenly included it.


