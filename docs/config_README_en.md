This a ChatGPT-4 English Conversion of kohya-ss (config_README-ja.md)


This documentation explains the configuration file that can be passed using the `--dataset_config` option.

## Overview

By providing a configuration file, users can fine-tune various settings.

* Multiple datasets can be configured.
    * For example, you can set the `resolution` for each dataset and train them together.
    * In learning methods that support both DreamBooth and fine-tuning techniques, it is possible to mix datasets using DreamBooth and fine-tuning techniques.
* Settings can be changed for each subset.
    * A dataset is a collection of subsets, which are created by dividing the dataset into separate image directories or metadata.
    * Options such as `keep_tokens` and `flip_aug` can be set for each subset. On the other hand, options such as `resolution` and `batch_size` can be set for each dataset, and the values are shared among subsets belonging to the same dataset. More details are provided later.

The configuration file can be written in JSON or TOML format. Considering ease of writing, we recommend using [TOML](https://toml.io/ja/v1.0.0-rc.2). The following explanations assume the use of TOML.

Here is an example of a configuration file written in TOML:

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
  # This subset has keep_tokens = 2 (using the value of the parent datasets)

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# This is a fine-tuning-style dataset
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # This subset has keep_tokens = 1 (using the general value)
```

In this example, three directories are trained as DreamBooth-style datasets at 512x512 (batch size 4), and one directory is trained as a fine-tuning-style dataset at 768x768 (batch size 2).

## Dataset and Subset Configuration Settings

The settings for datasets and subsets are divided into several sections.

* `[general]`
    * This section specifies options that apply to all datasets or all subsets.
    * If an option with the same name exists in the dataset-specific and subset-specific settings, the dataset and subset-specific settings take precedence.
* `[[datasets]]`
    * `datasets` is the registration section for dataset settings. This section specifies options that apply individually to each dataset.
    * If subset-specific settings exist, the subset-specific settings take precedence.
* `[[datasets.subsets]]`
    * `datasets.subsets` is the registration section for subset settings. This section specifies options that apply individually to each subset.

The following is a conceptual diagram of the correspondence between the image directories and registration sections in the previous example:

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

Each image directory corresponds to one `[[datasets.subsets]]`. One or more `[[datasets.subsets]]` are combined to form a `[[datasets]]`. The `[general]` section includes all `[[datasets]]` and `[[datasets.subsets]]`.

Different options can be specified for each registration section, but if an option with the same name is specified, the value in the lower registration section takes precedence. It may be easier to understand by checking how the `keep_tokens` option is handled in the previous example.

In addition, the available options vary depending on the supported techniques of the learning method.

* DreamBooth-specific options
* Fine-tuning-specific options
* Options available when the caption dropout technique can be used

In learning methods that support both DreamBooth and fine-tuning techniques, both can be used together.
When using both, note that whether a dataset is a DreamBooth-style or fine-tuning-style is determined on a dataset-by-dataset basis, so it is not possible to mix DreamBooth-style subsets and fine-tuning-style subsets within the same dataset.
In other words, if you want to use both of these techniques, you need to set the subsets with different techniques to belong to different datasets.

Regarding the program's behavior, it is determined that a subset is a fine-tuning-style subset if the `metadata_file` option, which will be explained later, exists.
Therefore, for subsets belonging to the same dataset, there is no problem as long as they are either "all have the `metadata_file` option" or "all do not have the `metadata_file` option".

The following describes the available options. For options with the same name as command-line arguments, the basic explanation is omitted. Please refer to the other READMEs.

### Common Options for All Learning Methods

These options can be specified regardless of the learning method.

#### Dataset-specific Options

These options are related to dataset settings and cannot be written in `datasets.subsets`.

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
    * Equivalent to the command line argument `--train_batch_size`.

These settings are fixed for each dataset. In other words, subsets belonging to the same dataset will share these settings. For example, if you want to prepare datasets with different resolutions, you can define them as separate datasets, as shown in the example above, and set different resolutions.

#### Subset-specific options

These are options related to the configuration of subsets.

| Option name | Example | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |

* `num_repeats`
    * Specifies the number of times the images in the subset are repeated. It corresponds to `--dataset_repeats` in fine-tuning, but `num_repeats` can be specified for any learning method.

### Options exclusive to DreamBooth method

The options for the DreamBooth method exist only for subset-specific options.

#### Subset-specific options

These are options related to the configuration of subsets in the DreamBooth method.

| Option name | Example | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o (required) |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `“sks girl”` | - | - | o |
| `is_reg` | `false` | - | - | o |

Please note that the `image_dir` must specify a path where the image files are placed directly. In the traditional DreamBooth method, images needed to be placed in subdirectories, but this is not compatible with that specification. Also, even if you name the folder like `5_cat`, the repetition count and class name of the images will not be reflected. If you want to set these individually, you need to explicitly specify `num_repeats` and `class_tokens`.

* `image_dir`
    * Specifies the path of the image directory. This is a required option.
    * Images must be placed directly in the directory.
* `class_tokens`
    * Sets the class tokens.
    * It will be used during training only if there is no corresponding caption file for the image. The determination of whether to use it is made on a per-image basis. If you do not specify `class_tokens` and no caption file is found, an error will occur.
* `is_reg`
    * Specifies whether the images in the subset are for normalization or not. If not specified, it is treated as `false`, meaning the images are not for normalization.

### Options exclusive to fine-tuning method

The options for the fine-tuning method exist only for subset-specific options.

#### Subset-specific options

These are options related to the configuration of subsets in the fine-tuning method.

| Option name | Example | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o (required) |

* `image_dir`
    * Specifies the path of the image directory. Unlike the DreamBooth method, this is not a required specification, but it is recommended to set it.
        * The situation where you do not need to specify it is when you have executed with `--full_path` when creating the metadata file.
    * Images must be placed directly in the directory.
* `metadata_file`
    * Specifies the path of the metadata file used in the subset. This is a required option.
        * Equivalent to the command line argument `--in_json`.
    * Since the specification requires you to specify the metadata file for each subset, it is better to avoid creating metadata that spans directories in a single metadata file. It is strongly recommended to prepare a metadata file for each image directory and register them as separate subsets.

### Options available when the caption dropout method can be used

Caption dropout method options exist only for subset-specific options. Regardless of whether it is the DreamBooth method or the fine-tuning method, you can specify it if the learning method supports caption dropout.

#### Subset-specific options

These are options related to the configuration of subsets when the caption dropout method can be used.

| Option name | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## Behavior when duplicate subsets exist

For DreamBooth method datasets, subsets with the same `image_dir` are considered duplicates. For fine-tuning method datasets, subsets with the same `metadata_file` are considered duplicates. If duplicate subsets exist within the dataset, the second and subsequent ones will be ignored.

On the other hand, if they belong to different datasets, they are not considered duplicates. For example, if you put subsets with the same `image_dir` in different datasets, they are not considered duplicates. This is useful when you want to train the same images at different resolutions.

```toml
# If they exist in separate datasets, they are not considered duplicates and both will be used for training

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## Usage with command line arguments

Some options in the configuration file have overlapping roles with command line arguments.

The following command line argument options are ignored when passing a configuration file:

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

For the following command line argument options, if they are specified simultaneously in the command line argument and configuration file, the value in the configuration file takes precedence. Unless otherwise stated, the options have the same name.

| Command line argument option | Preferred configuration file option |
| ---------------------------------- | ---------------------------------- |
| `--bucket_no_upscale` | |
| `--bucket_reso_steps` | |
| `--caption_dropout_every_n_epochs` | |
| `--caption_dropout_rate` | |
| `--caption_extension` | |
| `--caption_tag_dropout_rate` | |
| `--color_aug` | |
| `--dataset_repeats` | `num_repeats` |
| `--enable_bucket` | |
| `--face_crop_aug_range` | |
| `--flip_aug` | |
| `--keep_tokens` | |
| `--min_bucket_reso` | |
| `--random_crop` | |
| `--resolution` | |
| `--shuffle_caption` | |
| `--train_batch_size` | `batch_size` |

## Error Handling Guide

Currently, we are using an external library to check whether the configuration file is written correctly or not. However, the system is not well-maintained, and the error messages can be difficult to understand. We plan to address this issue in the future.

As a temporary solution, we provide a list of frequently encountered errors and their solutions. If you encounter an error even though you believe everything is correct, or if you cannot understand the error message, please contact us as it may be a bug.

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: This error indicates that a required option has not been specified. You might have forgotten to include the option or may have entered the option name incorrectly.
  * The location of the error is indicated by the `...` part of the message. For example, if you see the error `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']`, it means that the `image_dir` setting is missing from the 0th `subsets` configuration within the 0th `datasets`.
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: This error indicates that the value format is incorrect. The format of the value is likely incorrect. The `int` part will vary depending on the target option. The "Example Settings" for the options listed in this README may be helpful.
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: This error occurs when there are unsupported option names present. You may have entered the option name incorrectly or accidentally included it.
