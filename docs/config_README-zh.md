这是一个关于如何使用`--dataset_config`参数来传递设置文件的解释。

## 概述

通过传递设置文件，用户可以进行更细致的设置。

* 可以设置多个数据集
    * 例如，可以为每个数据集设置`resolution`，并混合这些数据集进行学习。
    * 在同时支持DreamBooth方法和fine tuning方法的学习方法中，有可能混合DreamBooth方法和fine tuning方法的数据集。
* 可以为每个子集设置不同的参数
    * 被按图像目录或元数据分开的数据集部分称为子集。一些子集组合形成数据集。
    * `keep_tokens`或`flip_aug`等选项可以为每个子集设置。另一方面，如`resolution`或`batch_size`等选项可以为每个数据集设置，属于同一数据集的子集将共享相同的值。详细信息将在后文中解释。

设置文件可以采用JSON或TOML格式。考虑到编写方便性，我们推荐使用[TOML](https://toml.io/cn/v1.0.0-rc.2)。以下的解释将基于使用TOML。

以下是一个使用TOML编写的设置文件示例：

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# 这是DreamBooth方法的数据集
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # 这个子集的keep_tokens = 2（使用所属数据集的值）

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# 这是fine tuning方法的数据集
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # 这个子集的keep_tokens = 1（使用general的值）
```

在这个例子中，将使用512x512 (batch size 4)的设置，以DreamBooth方法学习三个目录；并且将使用768x768 (batch size 2)的设置，以fine tuning方法学习一个目录。

## 关于数据集和子集的设置

关于数据集和子集的设置，可以在几个不同的地方进行。

* `[general]`
    * 这是设置适用于所有数据集或所有子集的选项的地方。
    * 如果存在同名的选项在每个数据集或每个子集的设置中，那么数据集和子集的特定设置将优先。
* `[[datasets]]`
    * `datasets`是关于数据集设置的注册位置。这是为每个数据集指定特定选项的地方。
    * 如果存在每个子集的设置，那么子集特定的设置将优先。
* `[[datasets.subsets]]`
    * `datasets.subsets`是关于子集设置的注册位置。这是为每个子集指定特定选项的地方。

以下，是关于前文示例中图像目录和注册位置对应关系的示意图。

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

每个图像目录对应一个`[[datasets.subsets]]`。然后，一个或多个`[[datasets.subsets]]`组合形成一个`[[datasets]]`。所有`[[datasets]]`和`[[datasets.subsets]]`都属于`[general]`。

虽然每个注册位置可以指定的选项不同，但如果指定了同名的选项，那么优先级将给予位于较低注册位置的值。我认为，查看前文示例中`keep_tokens`选项的处理方式，将有助于理解这一概念。

此外，根据学习方法所支持的技术，可指定的选项也将发生变化。

* 仅适用于DreamBooth方法的选项
* 仅适用于fine tuning方法的选项
* 当可以使用caption dropout方法时的选项
* 
在同时支持DreamBooth方法和fine tuning方法的学习方法中，可以同时使用这两种方法。
需要注意的是，我们是以数据集为单位来判断是DreamBooth方法还是fine tuning方法，因此，在同一数据集中不能混合DreamBooth方法的子集和fine tuning方法的子集。
也就是说，如果想同时使用这两种方法，需要将不同方法的子集设置为属于不同的数据集。

在程序的执行中，如果存在后文将提到的`metadata_file`选项，将判断该子集为fine tuning方法。
因此，对于属于同一数据集的子集，只要所有子集都包含`metadata_file`选项，或者都不包含`metadata_file`选项，就不会有问题。

以下，将解释可用的选项。对于与命令行参数名称相同的选项，基本上将省略解释。请参考其他README。

### 所有学习方法共通的选项

无论学习方法如何，都可以指定的选项。

#### 针对数据集的选项

这些是与数据集设置相关的选项。不能在`datasets.subsets`中描述。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` |
| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * 等同于命令行参数的`--train_batch_size`。

这些设置是每个数据集固定的。
也就是说，属于数据集的子集将共享这些设置。
例如，如果想使用不同分辨率的数据集，可以像前文示例那样，将它们定义为不同的数据集，就可以设置不同的分辨率了。

#### 针对子集的选项

这些是与子集设置相关的选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |
| `caption_prefix` | `“masterpiece, best quality, ”` | o | o | o |
| `caption_suffix` | `“, from side”` | o | o | o |
| `caption_separator` | （通常不进行设置） | o | o | o |
| `keep_tokens_separator` | `“|||”` | o | o | o |
| `secondary_separator` | `“;;;”` | o | o | o |
| `enable_wildcard` | `true` | o | o | o |

* `num_repeats`
    * 指定子集图像的重复次数。这相当于fine tuning中的`--dataset_repeats`，但是`num_repeats`可以在任何学习方法中指定。
* `caption_prefix`, `caption_suffix`
    * 指定添加在标题前后的字符串。这些字符串也会在进行shuffle时包括在内。如果指定了`keep_tokens`，请注意。

* `caption_separator`
    * 指定用于分割标签的字符串。默认为`,`。此选项通常无需设置。

* `keep_tokens_separator`
    * 指定在标题中用于分割固定部分的字符串。例如，如果指定为`aaa, bbb ||| ccc, ddd, eee, fff ||| ggg, hhh`，那么`aaa, bbb`和`ggg, hhh`部分将不会被shuffle或drop，而是保留。中间的逗号是不必要的。结果，prompt将可能是`aaa, bbb, eee, ccc, fff, ggg, hhh`或`aaa, bbb, fff, ccc, eee, ggg, hhh`等。

* `secondary_separator`
    * 指定额外的分隔符。由这个分隔符分割的部分将作为一个标签处理，然后被shuffle或drop。之后，它将被替换为`caption_separator`。例如，如果指定为`aaa;;;bbb;;;ccc`，将被替换为`aaa,bbb,ccc`，或整体被drop。

* `enable_wildcard`
    * 启用通配符表示法和多行标题。关于通配符表示法和多行标题的详细信息，将在后文中解释。

### 仅适用于DreamBooth方法的选项

DreamBooth方法的选项只存在于针对子集的选项中。

#### 针对子集的选项

这些是与DreamBooth方法子集设置相关的选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o（必须） |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `“sks girl”` | - | - | o |
| `cache_info` | `false` | o | o | o | 
| `is_reg` | `false` | - | - | o |

首先，需要注意的是，`image_dir`中应指定图像文件直接位于其下的路径。在传统的DreamBooth方法中，需要将图像放置在子目录下，但与该方法不兼容。此外，即使使用如`5_cat`这样的文件夹名称，图像的重复次数和类名称也不会被反映。如果想分别设置这些，注意需要使用`num_repeats`和`class_tokens`明确指定。

* `image_dir`
    * 指定图像目录的路径。这是必须指定的选项。
    * 图像必须直接放置在目录下。
* `class_tokens`
    * 设置类令牌。
    * 仅在不存在与图像对应的caption文件时，在学习时使用。是否使用将逐个图像判断。如果没有指定`class_tokens`，并且没有找到caption文件，将导致错误。
* `cache_info`
    * 指定是否缓存图像大小和caption。如果没有指定，将默认为`false`。缓存将以`metadata_cache.json`的文件名保存在`image_dir`中。
    * 如果进行缓存，第二次及以后的数据集加载将加速。如果处理数千张以上的图像，这将非常有效。
* `is_reg`
    * 指定子集的图像是否用于规范化。如果没有指定，将视为`false`，即不是规范化图像。

### 仅适用于fine tuning方法的选项

fine tuning方法的选项只存在于针对子集的选项中。

#### 针对子集的选项

这些是与fine tuning方法子集设置相关的选项。

| 选项名称 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o（必须） |

* `image_dir`
    * 指定图像目录的路径。与DreamBooth的方法不同，虽然这不是必须的，但推荐进行设置。
        * 如果在生成元数据文件时使用了`--full_path`参数执行，则不需要指定。
    * 图像必须直接放置在目录下。
* `metadata_file`
    * 指定子集使用的元数据文件的路径。这是必须指定的选项。
        * 这相当于命令行参数的`--in_json`。
    * 鉴于每个子集需要指定元数据文件的规范，最好避免将跨目录的元数据合并为一个元数据文件。强烈推荐为每个图像目录准备元数据文件，并将它们作为不同的子集进行注册。

### 当可以使用caption dropout方法时可指定的选项

当可以使用caption dropout方法时的选项只存在于针对子集的选项中。
无论DreamBooth方法还是fine tuning方法，只要学习方法支持caption dropout，就可以指定。

#### 针对子集的选项

这些是与可以使用caption dropout的子集设置相关的选项。

| 选项名称 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## 关于存在重复子集时的行为

对于DreamBooth方法的数据集，其中具有相同`image_dir`的子集被视为重复。
对于fine tuning方法的数据集，其中具有相同`metadata_file`的子集被视为重复。
如果数据集中存在重复的子集，从第二个开始将被忽略。

另一方面，如果属于不同的数据集，则不会被视为重复。
例如，如果将具有相同`image_dir`的子集放入不同的数据集中，它们将不会被视为重复。
这在希望以不同分辨率学习相同图像时会很有用。

```toml
# 如果在不同的数据集中存在，不会被视为重复，两者都将用于学习

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## 与命令行参数的并用

设置文件的选项中，有一些与命令行参数的选项作用相同。

以下列出的命令行参数选项，在传递设置文件时将被忽略。

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

以下列出的命令行参数选项，在命令行参数和设置文件中同时指定时，将优先使用设置文件的值。除非特别说明，否则同名选项将优先使用设置文件中的值。

| 命令行参数的选项     | 优先使用的设置文件的选项 |
| ------------------ | ------------------ |
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

## 错误指南

目前，我们正在使用外部库检查设置文件的描述是否正确，但由于维护尚未完善，存在错误信息难以理解的问题。
未来，我们计划解决这个问题。

作为次优方案，我们将列出常见的错误及其处理方法。
如果正确无误仍然出现错误，或者无法理解错误内容，请告知我们，这可能是一个bug。

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`: 这是一个错误，表示未提供必须的键。很可能是因为忘记指定或者键名输入错误。
  * 在`...`部分，将显示发生错误的位置。例如，如果出现`voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']`这样的错误，这意味着在0号`datasets`中的0号`subsets`的设置中未找到`image_dir`。
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`: 这是一个错误，表示字典值的格式不正确。很可能是因为值的格式错误。`int`部分将根据目标选项而变化。这个README中列出的选项的“设置示例”可能有所帮助。
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`: 当存在不支持的选项名时将发生此错误。很可能是因为选项名输入错误或意外混入。

## 其他

### 多行标题

通过设置`enable_wildcard = true`，也可以同时启用多行标题。如果标题文件由多行组成，将随机选择其中一行作为标题使用。

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, microphone, stage
a girl with a microphone standing on a stage
detailed digital art of a girl with a microphone on a stage
```

可以将此和通配符表示法结合使用。

在元数据文件中也可以类似地指定多行标题。在元数据的.json中，使用`\n`表示换行。当标题文件由多行组成时，使用`merge_captions_to_metadata.py`，将以这种格式创建元数据文件。

元数据的标签(tags)将添加到标题的每一行中。

```json
{
    "/path/to/image.png": {
        "caption": "a cartoon of a frog with the word frog on it\ntest multiline caption1\ntest multiline caption2",
        "tags": "open mouth, simple background, standing, no humans, animal, black background, frog, animal costume, animal focus"
    },
    ...
}
```

在这种情况下，实际的标题将是`a cartoon of a frog with the word frog on it, open mouth, simple background ...`或`test multiline caption1, open mouth, simple background ...`,`test multiline caption2, open mouth, simple background ...`等。

### 设置文件描述示例：额外的分隔符、通配符表示法、`keep_tokens_separator`等

```toml
[general]
flip_aug = true
color_aug = false
resolution = [1024, 1024]

[[datasets]]
batch_size = 6
enable_bucket = true
bucket_no_upscale = true
caption_extension = ".txt"
keep_tokens_separator= "|||"
shuffle_caption = true
caption_tag_dropout_rate = 0.1
secondary_separator = ";;;" # 也可以在子集侧写入 / can be written in the subset side
enable_wildcard = true # 同上 / same as above

  [[datasets.subsets]]
  image_dir = "/path/to/image_dir"
  num_repeats = 1

  # `|||`前后不需要逗号（自动添加） / No comma is required before and after ||| (it is added automatically)
  caption_prefix = "1girl, hatsune miku, vocaloid |||" 
  
  # `|||`后的内容不会被shuffle或drop，而是保留 / After |||, it is not shuffled or dropped and remains
  # 由于只是简单地作为字符串连接，所以需要自己插入逗号等 / It is simply concatenated as a string, so you need to put commas yourself
  caption_suffix = ", anime screencap ||| masterpiece, rating: general"
```

### 标题描述示例，secondary_separator表示法：在`secondary_separator = ";;;"`的情况下

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, sky;;;cloud;;;day, outdoors
```
`sky;;;cloud;;;day`部分不会被shuffle或drop，而是被替换为`sky,cloud,day`。如果启用了shuffle或drop，将作为一个整体（作为一个标签）进行处理。也就是说，可能变成`vocaloid, 1girl, upper body, sky,cloud,day, outdoors, hatsune miku`（shuffle后）或`vocaloid, 1girl, outdoors, looking at viewer, upper body, hatsune miku`（drop情况）等。

### 标题描述示例，通配符表示法：在`enable_wildcard = true`的情况下

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, {simple|white} background
```
随机选择`simple`或`white`，变成`simple background`或`white background`。

```txt
1girl, hatsune miku, vocaloid, {{retro style}}
```
如果想在标签字符串中包含`{`或`}`本身，请使用`{{`或`}}`（在本例中，实际用于学习的标题将变为`{retro style}`）。

### 标题描述示例，`keep_tokens_separator`表示法：在`keep_tokens_separator = "|||"`的情况下

```txt
1girl, hatsune miku, vocaloid ||| stage, microphone, white shirt, smile ||| best quality, rating: general
```
可能变成`1girl, hatsune miku, vocaloid, microphone, stage, white shirt, best quality, rating: general`或`1girl, hatsune miku, vocaloid, white shirt, smile, stage, microphone, best quality, rating: general`等。
