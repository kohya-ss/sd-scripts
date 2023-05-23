This is an explanation about the configuration file that can be passed with `--dataset_config`.

## Overview

By providing a configuration file, users are allowed to make detailed settings.

* Multiple datasets can be set
    * For example, you can set `resolution` for each dataset and mix them for training.
    * In learning methods compatible with both DreamBooth's method and fine tuning method, it is possible to mix datasets of DreamBooth's method and fine tuning method.
* Settings can be changed for each subset
    * Subsets are divisions of the dataset by image directories or metadata. Some subsets make up a dataset.
    * Options such as `keep_tokens` and `flip_aug` can be set for each subset. On the other hand, options such as `resolution` and `batch_size` can be set for each dataset and have the same value for subsets belonging to the same dataset. More details will be discussed later.

The format of the configuration file can be either JSON or TOML. Considering ease of writing, it is recommended to use [TOML](https://toml.io/ja/v1.0.0-rc.2). The following explanation assumes the use of TOML.

Here is an example of a configuration file written in TOML.

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# This is a dataset of DreamBooth's method
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # This subset uses keep_tokens = 2 (the value of the datasets to which it belongs)

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# This is a dataset of fine tuning method
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # This subset uses keep_tokens = 1 (the value of the general)
```

In this example, three directories are trained as a dataset of DreamBooth's method in 512x512 (batch size 4), and one directory is trained as a dataset of the fine tuning method in 768x768 (batch size 2).

## Settings for Datasets and Subsets

The settings for datasets and subsets are divided into several registrable places.

* `[general]`
    * This is where you specify options that apply to all datasets or all subsets.
    * If there are options with the same name in the settings for each dataset and each subset, the settings for each dataset and subset will take precedence.
* `[[datasets]]`
    * `datasets` is where you register settings for datasets. This is where you specify options that apply individually to each dataset.
    * If there are settings for each subset, the settings for each subset will take precedence.
* `[[datasets.subsets]]`
    * `datasets.subsets` is where you register settings for subsets. This is where you specify options that apply individually to each subset.

Here is an illustration of the correspondence between the image directories and registration places in the previous example.

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.

1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

Each image directory corresponds to one `[[datasets.subsets]]`. Then, one or more `[[datasets.subsets]]` together make up one `[[datasets]]`. All `[[datasets]]` and `[[datasets.subsets]]` belong to `[general]`.

Different options can be specified for each registration place. However, if the same name option is specified, the value in the lower registration place takes precedence. The treatment of the `keep_tokens` option in the previous example should be easy to understand.

In addition, the available options change depending on the method supported by the learning method.

* Options dedicated to DreamBooth's method
* Options dedicated to fine tuning method
* Options when the caption dropout method is available

In a learning method where both DreamBooth's method and fine tuning method are available, both can be used together. When using them together, note that whether it is DreamBooth's method or fine tuning method is determined on a dataset-by-dataset basis, so you cannot mix subsets of DreamBooth's method and fine tuning method in the same dataset. In other words, if you want to use both, you need to set different types of subsets to belong to different datasets.

In terms of program behavior, if the `metadata_file` option exists, it is determined to be a subset of the fine tuning method. Therefore, for subsets belonging to the same dataset, there is no problem as long as they are all "either all have the `metadata_file` option" or "none have the `metadata_file` option".

Below, I will explain the available options. The explanation will be omitted for options with the same name as command line arguments. Please refer to the other READMEs.

### Options Common to All Learning Methods

These are options that can be specified regardless of the learning method.

#### Dataset-Specific Options

These are options related to the setting of the dataset. They cannot be written in `datasets.subsets`.

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
    * Equivalent to the `--train_batch_size` command line argument.

These settings are fixed for each dataset.
In other words, subsets belonging to the same dataset will share these settings.
For example, if you want to prepare datasets with different resolutions, you can set different resolutions by defining them as separate datasets as shown in the previous example.

#### Subset-Specific Options

These are options related to the setting of subsets.

| Option Name | Example Setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o |

 o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |

* `num_repeats`
    * Specifies the number of times to repeat the images of the subset. This is equivalent to `--dataset_repeats` in fine tuning, but `num_repeats` can be specified in any learning method.

### Options Specific to DreamBooth's Method

Options for DreamBooth's method only exist as subset-specific options.

#### Subset-Specific Options

These are options related to the setting of subsets in DreamBooth's method.

| Option Name | Example Setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `'C:\hoge'` | - | - | o (Required) |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `"sks girl"` | - | - | o |
| `is_reg` | `false` | - | - | o |

First of all, it should be noted that you need to specify the path where image files are directly placed in `image_dir`. The traditional DreamBooth's method required you to place images in a subdirectory, but it is not compatible with that specification. Even if you name the folder as `5_cat`, the number of image repetitions and the class name will not be reflected. Please note that if you want to set these individually, you need to explicitly specify `num_repeats` and `class_tokens`.

* `image_dir`
    * Specifies the path to the image directory. This is a required option.
    * Images must be placed directly under the directory.
* `class_tokens`
    * Sets the class tokens.
    * They are used only during training if there is no corresponding caption file for the image. The decision to use them is made for each image. If a caption file is not found and `class_tokens` are not specified, an error will occur.
* `is_reg`
    * Specifies whether the images in the subset are for normalization. If not specified, it will be treated as `false`, meaning they are not normalization images.

### Options Specific to Fine Tuning Method

Options for the fine tuning method only exist as subset-specific options.

#### Subset-Specific Options

These are options related to the setting of subsets in the fine tuning method.

| Option Name | Example Setting | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `'C:\hoge'` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o (Required) |

* `image_dir`
    * Specifies the path to the image directory. Unlike DreamBooth's method, it is not required, but it is recommended to set it.
        * The situation where there is no need to specify is when `--full_path` was executed when generating the metadata file.
    * Images must be placed directly under the directory.
* `metadata_file`
    * Specifies the path to the metadata file used in the subset. This is a required option.
        * Equivalent to the `--in_json` command line argument.
    * Due to the specification that requires a metadata file to be specified for each subset, it is advisable to avoid creating metadata across directories as one metadata file. It is strongly recommended to prepare a metadata file for each image directory and register them as separate subsets.

### Options Available for Caption Dropout Technique

The options available for the caption dropout technique are only for subsets. Whether you're using the DreamBooth method or the fine-tuning method, you can specify them if your training method supports caption dropout.

#### Subset Specific Options

These are the options related to the settings of subsets where caption dropout can be used.

| Option Name | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## Behavior when Duplicates Subsets Exist

In the case of a DreamBooth style dataset, subsets with the same `image_dir` are considered duplicates. For fine tuning style datasets, subsets with the same `metadata_file` are considered duplicates. If duplicate subsets exist within a dataset, the second and subsequent ones will be ignored.

On the other hand, if they belong to different datasets, they are not considered duplicates. For instance, if you put subsets with the same `image_dir` into different datasets, as shown below, they are not considered duplicates. This can be useful when you want to train with the same images at different resolutions.

```toml
# If they exist in different datasets, they are not considered duplicates and are used for training.

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## Combining with Command Line Arguments

Some options in the configuration file overlap with the options of command line arguments.

The following command line argument options will be ignored if a configuration file is passed:

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

For the following command line argument options, if they are specified simultaneously in the command line arguments and the configuration file, the value in the configuration file will take precedence. Unless otherwise specified, they are the same option.

| Command Line Argument Options     | Prioritized Configuration File Options |
| ---------------------------------- | ---------------------------------- |
| `--bucket_no_upscale`              |                                    |
| `--bucket_reso_steps`              |                                    |
| `--caption_dropout_every_n_epochs` |                                    |
| `--caption_dropout_rate`           |                                    |
| `--caption_extension`              |                                    |
| `--caption_tag_dropout_rate`       |                                    |
| `--color_aug`                      |                                    |
| `--dataset_repeats`                | `num_repeats`                      |
| `--enable_bucket`                  |                                    |
| `--face_crop_aug_range`            |                                    |
| `--flip_aug`                       |                                    |
| `--keep_tokens`                    |                                    |
| `--min_bucket_reso`                |                                    |
| `--random_crop`                    |                                    |
| `--resolution`                     |                                    |
| `--shuffle_caption`                |                                    |
| `--train_batch_size`               | `batch_size`                       |

## Error Guide

We are currently using external libraries to check if the description of the configuration file is correct. However, there is a problem that the error message is not clear due to inadequate maintenance. We plan to work on improving this problem in the future.

As a second-best measure, we will post frequently occurring errors and how to deal with them. If you get an error that should be correct, or if you can't understand the error content, it might be a bug, so please contact us.

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: This is an error that a required option is not specified. You may have forgotten to specify it or may have made a mistake in the option name.
  * The location where the error occurred is listed in the `...` part. For example, if an error like `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']` occurs, it means that there is no `image_dir` in the settings of the 0th `subsets` in the 0th `datasets`.
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: This is an error where the format of the specified value is incorrect. The format of the value may be wrong. The `int` part changes depending on the target option. The "Example Settings" for the options listed in this README may be helpful.
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: This is an error that occurs when there is an option name that is not supported. You may have made a mistake in the option name or it may have accidentally slipped in.
