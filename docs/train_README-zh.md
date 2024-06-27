__文档正在更新中，描述中可能存在错误。__

# 学习，通用篇

本仓库支持模型的fine tuning，DreamBooth，以及LoRA和Textual Inversion（包括[XTI:P+](https://github.com/kohya-ss/sd-scripts/pull/327)）的学习。本文件将解释它们共有的学习数据准备方法和选项等。

# 概览

请先参考这个仓库的README，进行环境准备。

以下将进行说明。

1. 学习数据的准备（使用设置文件的新格式）
1. 对用于学习的术语的简单解释
1. 以前的指定格式（不使用设置文件，直接在命令行中指定）
1. 学习过程中的样本图像生成
1. 各个脚本中常用的、通用的选项
1. fine tuning方式的元数据准备：captioning等

如果你只执行1，应该就可以开始学习了（具体学习方式请参考各脚本的文档）。从2开始，你可以根据需要参考。

# 关于学习数据的准备

你可以将学习数据的图像文件准备在任意文件夹（可以是多个）中。支持`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`。基本上不需要进行如resize等的预处理。

然而，我建议你不要使用比学习分辨率（后面会提到）小得多的图像，或者提前使用超分辨率AI等进行放大。此外，似乎存在超过极大图像（大约3000x3000像素？）的情况会导致错误，所以请提前缩小。

在学习时，你必须整理提供给模型学习的图像数据，并指定给脚本。根据学习数据的数量、学习目标、是否可以准备caption（图像的描述）等，有几种方式可以指定学习数据。有以下几种方式（各个名称不是通用的，而是本仓库独有的定义）。稍后将提到正则化图像。

1. DreamBooth、class+identifier方式（可以使用正则化图像）

    它学习将特定单词（identifier）与学习目标关联。你不需要准备caption。例如，如果你想学习特定角色，使用它就不需要准备caption，这很方便，但是，因为学习数据的所有元素都将与identifier关联并学习，因此可能会发生无法在生成时的prompt中改变服装、发型、背景等情况。

1. DreamBooth、caption方式（可以使用正则化图像）

    你通过为每个图像准备带有caption的文本文件来学习。例如，如果你学习特定角色，并在caption中描述图像的细节（穿着白色衣服的角色A，穿着红色衣服的角色A等），可以期待角色和其他元素分离，模型将更精确地学习角色。

1. fine tuning方式（不可以使用正则化图像）

    你提前在元数据文件中总结caption。它支持将标签和caption分开管理，或为了加速学习而提前缓存latents等功能（这些在其他文档中解释）。（虽然命名为fine tuning方式，但也可以用于非fine tuning。）

你可以根据你想学的内容和可用的指定方法组合来选择。

| 学习目标或方法 | 脚本 | DB / class+identifier | DB / caption | fine tuning |
| ----- | ----- | ----- | ----- | ----- |
| fine tuning模型 | `fine_tune.py`| x | x | o |
| DreamBooth模型 | `train_db.py`| o | o | x |
| LoRA | `train_network.py`| o | o | o |
| Textual Inversion | `train_textual_inversion.py`| o | o | o |

## 应该选择哪个？

如果你希望在没有准备字幕文件的情况下轻松训练，LoRA和Textual Inversion，DreamBooth class+identifier方式可能是好选择。如果有能力准备字幕文件，DreamBooth字幕方式会更佳。如果训练数据量大并且不使用正则化图像，也请考虑fine tuning方式。

对于DreamBooth同样适用，但是不能使用fine tuning方式。在fine tuning情况下，只能使用fine tuning方式。

# 关于各方式的指定方法

这里只对每种指定方法的典型模式进行说明。更详细的指定方法，请参阅 [数据集设置](./config_README-zh.md)。

# DreamBooth, class+identifier方式（可使用正则化图像）

本方式下，每张图像将像使用`class identifier`字幕（如`shs dog`）训练一样。

## 第一步. 决定identifier和class

确定要绑定的对象的标识符单词和该对象所属的class。

（尽管有多种叫法如instance等，但我们将遵循原始论文的称呼。）

以下是简要说明（更多细节请自己查找）。

class是学习目标的一般分类。例如，如果要学习特定品种的狗，class就是dog。如果目标是动画角色，根据模型可能变成boy或girl，1boy或1girl。

identifier是用于识别和学习学习目标的任何单词。根据原始论文，“在tokenizer中成为一个token的三个字符以下的罕见单词”是好的选择。

使用identifier和class，例如，通过`shs dog`训练模型，可以从class中区分并学习你想要学习的对象。

生成图像时，输入`shs dog`将生成你训练的狗品种的图像。

（作为参考，我最近使用的identifier有`shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny`等。实际上，更希望使用不在Danbooru Tag中的单词。）

## 第二步. 决定是否使用正则化图像，如果使用则生成正则化图像

正则化图像，是为了防止前文所述的class整体被学习目标拖累的图像（语言漂移）。如果不使用正则化图像，例如，当使用`shs 1girl`学习特定角色时，即使仅使用`1girl`这样的提示生成，图像也会趋向于那个角色。这是因为`1girl`包含在学习时的字幕中。

通过同时学习学习目标的图像和正则化图像，可以使class保持为class，在附加identifier到提示时才生成学习目标。

在LoRA或DreamBooth中，如果只需要特定角色出现，可以不用正则化图像。

在Textual Inversion中，可能不需要使用（因为如果学习的 token string 不包含在字幕中，将不会学习任何东西）。

通常，作为正则化图像，使用目标模型仅通过class名称生成的图像（如`1girl`）是普遍做法。但是，如果生成图像质量不好，可以修改提示或使用从网络上下载的其他图像。

（由于正则化图像也会被学习，其质量将影响模型。）

通常，准备几百张是理想的（如果数量过少，class的图像不会泛化，反而会吸收它们的特征）。

如果使用生成图像，通常，生成图像的大小应与训练分辨率（更准确地说是bucket的分辨率，后文详述）一致。

## 第二步. 配置文件的编写

创建文本文件，并将其扩展名改为`.toml`。可以像以下一样进行编写。

（以`#`开头的部分是注释，你可以直接复制粘贴并保持原样，或删除它们，这都没问题。）

```toml
[general]
enable_bucket = true                        # 是否使用Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 4                              # 批次大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定包含训练图像的文件夹
  class_tokens = 'hoge girl'                # 指定identifier class
  num_repeats = 10                          # 重复训练图像的次数

  # 仅在使用正则化图像时描述以下内容。不使用时请删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定包含正则化图像的文件夹
  class_tokens = 'girl'                     # 指定class
  num_repeats = 1                           # 正则化图像的重复次数，基本上1就足够
```

基本上，只需更改以下几点就能开始训练。

1. 训练分辨率

    指定一个数字将得到正方形图像（如`512`将得到512x512），如果用括号和逗号分隔指定两个数字，则将得到横x纵的图像（如`[512,768]`将得到512x768）。在SD1.x系列中，原始的训练分辨率是512。指定如`[512,768]`等较大的分辨率可能有助于在生成高宽比图像时减少失真。在SD2.x 768系列中，是`768`。

1. 批次大小

    这指定了同时训练多少数据。这取决于GPU的VRAM大小和训练分辨率。具体细节后述。在fine tuning/DreamBooth/LoRA等情况下也会有所不同，因此也请参阅各脚本的说明。

1. 文件夹指定

    指定训练图像，及正则化图像（如果使用）的文件夹。仅需指定包含图像数据的文件夹本身。

1. identifier和class的指定

    按照前文的示例进行。

1. 重复次数

    详细后述。

### 关于重复次数

重复次数用于调整正则化图像的数量和训练图像的数量。由于正则化图像的数量通常多于训练图像，通过重复训练图像使其数量匹配，以便以1:1的比例进行学习。

请指定重复次数，使“ __训练图像的重复次数×训练图像的数量≧正则化图像的重复次数×正则化图像的数量__ ”。

（1 epoch（数据循环一次为1 epoch）的数据量为“训练图像的重复次数×训练图像的数量”。如果正则化图像的数量超过这个数量，多余部分的正则化图像不会被使用。）

## 第三步. 开始训练

请参考各自的文档进行训练。

# DreamBooth, 字幕方式（可使用正则化图像）

本方式下，每张图像将根据字幕进行训练。

## 第一步. 准备字幕文件

在训练图像的文件夹中，放置与图像相同的文件名，扩展名为`.caption`（可以在设置中更改）的文件。每个文件仅需一行。编码应为`UTF-8`。

## 第二步. 决定是否使用正则化图像，如果使用则生成正则化图像

与class+identifier形式相同。然而，尽管可以为正则化图像添加字幕，但通常不需要这样做。

## 步骤2. 配置文件的描述

创建一个文本文件，并将其扩展名设为`.toml`。例如，可以像下面这样描述：

```toml
[general]
enable_bucket = true                        # 是否使用 Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # 学习分辨率
batch_size = 4                              # 批量大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定包含训练图像的文件夹
  caption_extension = '.caption'            # 标题文件的扩展名 如果使用 .txt 需要修改
  num_repeats = 10                          # 训练图像的重复次数

  # 以下仅在使用正则化图像时描述。如果不使用则删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定包含正则化图像的文件夹
  class_tokens = 'girl'                     # 指定类别
  num_repeats = 1                           # 正则化图像的重复次数，基本上1次即可
```

## 第二步. 编写配置文件

创建文本文件，并将其扩展名设置为`.toml`。例如，可以这样编写：

```toml
[general]
enable_bucket = true                        # 是否使用Aspect Ratio Bucketing

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 4                              # 批次大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定存放训练图像的文件夹
  caption_extension = '.caption'            # 字幕文件的扩展名，如果使用.txt则需更改
  num_repeats = 10                          # 训练图像的重复次数

  # 仅在使用正则化图像时描述以下内容，不使用时请删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定存放正则化图像的文件夹
  class_tokens = 'girl'                     # 指定class
  num_repeats = 1                           # 正则化图像的重复次数，基本上1就足够
```

基本上，只需更改以下几点就能开始训练。未提及的部分与class+identifier方式相同。

1. 训练分辨率
1. 批次大小
1. 文件夹指定
1. 字幕文件的扩展名

    可以指定任意扩展名。
1. 重复次数

## 第三步. 开始训练

请参考各自的文档进行训练。

# fine tuning 方式

## 第一步. 准备元数据

我们称汇总了字幕和标签的管理文件为元数据。其格式为json，扩展名为`.json`。创建方法较为繁琐，因此在文档末尾进行了说明。

## 第二步. 编写配置文件

创建文本文件，并将其扩展名设置为`.toml`。例如，可以这样编写：

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # 训练分辨率
batch_size = 4                                      # 批次大小

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # 指定存放训练图像的文件夹
  metadata_file = 'C:\piyo\piyo_md.json'            # 元数据文件名
```

基本上，只需更改以下几点就能开始训练。未提及的部分与DreamBooth, class+identifier方式相同。

1. 训练分辨率
1. 批次大小
1. 文件夹指定
1. 元数据文件名

    按照后述方法创建的元数据文件需要被指定。


## 第三步. 开始训练

请参考各自的文档进行训练。

# 训练中使用的术语的简短解释

省略了细节，且我也不完全理解，因此请自行详查。

## fine tuning（微调）

指的是对模型进行学习和微调。根据使用方式，其含义会有所不同，但在狭义上，fine tuning指的是在Stable Diffusion中，使用图像和字幕训练模型。可以说DreamBooth是狭义fine tuning的一种特殊做法。广义的fine tuning包含LoRA、Textual Inversion、Hypernetworks等，涵盖了所有训练模型的过程。

## 步骤

粗略地说，每计算一次训练数据就是一步。“将当前模型的训练数据字幕流过，比较输出的图像与训练数据的图像，稍作调整以使模型更接近训练数据”，这就是一步。

## 批次大小

批次大小是每步处理多少数据的值。通过合并计算，速度相对提升。据说精度通常也会提高。

`批次大小×步数` 就是训练中使用的数据量。因此，增加批次大小时，应该相应减少步数。

（但是，例如“批次大小1，1600步”与“批次大小4，400步”不会产生相同的结果。如果学习率相同，一般来说后者的训练不足。请调整学习率（例如设置为 `2e-6`）或步数（例如设置为500步）。）

增加批次大小会相应增加GPU内存消耗。如果内存不足，会导致错误，即使不出现错误，训练速度会下降。请同时使用任务管理器或 `nvidia-smi` 命令检查使用的内存量并进行调整。

顺便说一下，批次的意思大致是“一块数据”。

## 学习率

简单说，是表示每步变化多少的值。设置较大值可以加快学习速度，但可能因变化过大导致模型破坏，或无法到达最优状态。设置较小值会降低学习速度，同样可能无法达到最优状态。

在fine tuning、DreamBooth、LoRA中差异较大，且会根据训练数据、想要训练的模型、批次大小和步数变化。请从一般值开始，同时观察学习状态进行调整。

默认情况下，学习率在整个学习过程中固定。通过指定调度器，可以决定如何变化学习率，因此结果也会因调度器而异。

## 时期（epoch）

当所有训练数据学完一遍（数据循环一次）就是1个epoch。如果指定了重复次数，完成重复后的数据循环一次就是1个epoch。

1个epoch的步数基本上是 `数据量÷批次大小`，但是使用Aspect Ratio Bucketing时会略有增加（由于不同bucket的数据不能放在同一个批次中，因此步数会增加）。

## Aspect Ratio Bucketing

Stable Diffusion v1在512\*512分辨率下训练，但同时也以256\*1024或384\*640等分辨率训练。这可以减少裁剪部分，更准确地学习字幕和图像的关系。

此外，由于可以在任意分辨率下学习，因此无需事先统一图像数据的长宽比。

可以在设置中切换启用或禁用，但到目前为止的配置文件示例中，它是启用状态（设置为`true`）。

训练分辨率将在不超过给定参数分辨率的面积（=内存使用量）的范围内，以64像素为单位（默认，可更改）调整和创建长宽。

在机器学习中，通常需要统一输入大小，但这并没有特别的限制，实际上只要在同一批次内统一大小即可。NovelAI所说的bucketing是指，根据长宽比预先将训练数据分类到不同的学习分辨率。然后，通过在每个bucket内创建批次，统一批次的图像大小。

# 旧的指定形式（不使用设置文件，在命令行指定）

这是不指定`.toml`文件，而在命令行选项中指定的方式。有DreamBooth class+identifier方式、DreamBooth 字幕方式和fine tuning方式。

## DreamBooth, class+identifier方式

使用文件夹名指定重复次数。同时使用`train_data_dir`选项和`reg_data_dir`选项。

### 第一步. 准备训练图像

创建存储训练图像的文件夹。__在该文件夹中__，以以下名称创建目录。

```
<重复次数>_<identifier> <class>
```

请不要忘记之间的`_`。

例如，如果使用`sls frog`作为提示，并重复数据20次，它将变为`20_sls frog`。如下所示。

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)

### 多class，多目标（identifier）的学习

方法很简单，在训练图像文件夹中准备多个``重复次数_<identifier> <class>``的文件夹，在正则化图像文件夹中同样准备多个``重复次数_<class>``的文件夹。

例如，如果同时学习`sls frog`和`cpc rabbit`，如下所示。

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

当class只有一个，目标有多个时，一个正则化图像文件夹就足够了。例如，如果girl中有角色A和角色B，你可以这样做。

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### 第二步. 准备正则化图像

这是使用正则化图像的步骤。

创建存储正则化图像的文件夹。__在该文件夹中__创建名为``<重复次数>_<class>``的目录。

例如，如果使用`frog`作为提示，并不重复数据（仅一次），如下所示。

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)


### 步骤3. 执行学习

我们将运行每个学习脚本。请通过 `--train_data_dir` 选项指定前面提到的训练数据文件夹（__不是包含图片的文件夹，而是其父文件夹__），并通过 `--reg_data_dir` 选项指定正则化图片的文件夹（__不是包含图片的文件夹，而是其父文件夹__）。

## DreamBooth、字幕方法

在训练和正则化图片的文件夹中，放置与图片同名的文件，扩展名为.caption（可选项可以更改），脚本将从该文件读取字幕并作为提示进行学习。

※在这些图片的学习中，文件夹名（标识符类别）将不再使用。

字幕文件的默认扩展名为.caption。可以通过学习脚本的 `--caption_extension` 选项进行更改。使用 `--shuffle_caption` 选项可以在学习时，将逗号分隔的各部分字幕随机化进行学习。

## 精调方法

创建元数据的部分与使用设置文件的情况相同。通过 `in_json` 选项指定元数据文件。

# 学习过程中的样本输出

通过在学习过程中生成图片，可以检查学习的进展。在学习脚本中指定以下选项。

- `--sample_every_n_steps` / `--sample_every_n_epochs`
    
    指定进行样本输出的步数或周期数。每隔这个数量的步数或周期，就进行样本输出。如果同时指定两者，则周期数优先。

- `--sample_at_first`
    
    在学习开始前进行样本输出。可以与学习前的图片进行对比。

- `--sample_prompts`

    指定用于样本输出的提示文件。

- `--sample_sampler`

    指定用于样本输出的采样器。
    可以选择 `'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'` 之一。

为了进行样本输出，需要预先准备一个包含提示的文本文件。每行写一个提示。

例如，如下所示：

```txt
# 提示 1
masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# 提示 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

以 `#` 开头的行被视为注释。可以使用 `--n` 这样的「`--` + 小写英文字母」来指定生成图片的选项。以下是可用的选项：

- `--n` 将接下来的选项视为负面提示。
- `--w` 指定生成图片的宽度。
- `--h` 指定生成图片的高度。
- `--d` 指定生成图片的seed。
- `--l` 指定生成图片的CFG scale。
- `--s` 指定生成时的步数。

# 各脚本中常用的、频繁使用的选项

更新脚本后，文档可能无法及时更新。在这种情况下，请使用 `--help` 选项查看可用的选项。

## 指定用于学习的模型

- `--v2` / `--v_parameterization`
    
    如果使用Hugging Face的stable-diffusion-2-base，或者基于此的精调模型作为学习目标模型（如果在推理时被指示使用 `v2-inference.yaml` 的模型）则指定 `--v2` 选项；如果使用stable-diffusion-2或768-v-ema.ckpt，以及它们的精调模型（如果在推理时使用 `v2-inference-v.yaml` 的模型）则指定 `--v2` 和 `--v_parameterization` 两个选项。

    Stable Diffusion 2.0在以下方面有重大改变：

    1. 使用的Tokenizer
    2. 使用的Text Encoder和使用的输出层（2.0使用倒数第二层）
    3. Text Encoder的输出维度数（768->1024）
    4. U-Net的结构（CrossAttention的head数等）
    5. v-parameterization（采样方法似乎已被更改）

    其中，base使用1～4，而非base（768-v）使用1～5。启用1～4的为v2选项，启用5的为v_parameterization选项。

- `--pretrained_model_name_or_path` 
    
    指定用于追加学习的原始模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors），Diffusers的本地磁盘上的模型目录，或Diffusers的模型ID（如"stabilityai/stable-diffusion-2"）。

## 关于学习的设定

- `--output_dir`

    指定学习后模型保存的文件夹。

- `--output_name`

    指定模型的文件名，不包括扩展名。

- `--dataset_config`

    指定描述数据集设置的 `.toml` 文件。

- `--max_train_steps` / `--max_train_epochs`

    指定学习的步数或周期数。当两者都指定时，周期数优先。

- `--mixed_precision`

    为了节省内存，使用 mixed precision（混合精度）进行学习。例如，指定 `--mixed_precision="fp16"`。与没有混合精度（默认）相比，精度可能会降低，但所需GPU内存将大幅减少。

    （从RTX30系列开始，也可以指定 `bf16`。请与在环境准备期间对accelerate进行的设置保持一致）。

- `--gradient_checkpointing`

    通过不是一次性计算学习中的权重，而是逐步进行，以减少所需的GPU内存。开启或关闭不会影响精度，但开启后可以使用更大的批次大小，从而影响性能。

    此外，通常情况下，开启后速度会降低，但由于可以使用更大的批次大小，总的学习时间可能会更快。

- `--xformers` / `--mem_eff_attn`

    指定xformers选项将使用xformers的CrossAttention。如果未安装xformers或遇到错误（根据环境，比如 `mixed_precision="no"` 的情况），指定 `mem_eff_attn` 选项将使用内存节省版CrossAttention（速度会比xformers慢）。

- `--clip_skip`

    指定`2`时，使用Text Encoder (CLIP) 倒数第二层的输出。在1或未指定选项时使用最后一层。

    ※SD2.0默认使用倒数第二层，因此在SD2.0的学习中，请不要指定。

    如果学习的目标模型已经训练为使用第二层，指定2可能是好的。

    反之，如果使用的是最后一层，那么模型整体都是基于此进行训练的。因此，如果再次使用第二层进行学习，可能需要一定数量的训练数据和较长的学习时间才能获得理想的学习结果。

- `--max_token_length`

    默认值是75。通过指定 `150` 或 `225`，可以扩展令牌长度进行学习。在使用长标题进行学习时指定。

    但是，学习时的令牌扩展规范与 Automatic1111 先生的Web UI略有不同（如分割规范等），如果不需要，建议在75进行学习。

    与 `clip_skip` 类似，如果在与模型学习状态不同的长度下学习，可能需要一定数量的训练数据和较长的学习时间。

- `--weighted_captions`

    指定时，与Automatic1111先生的Web UI相同，启用加权标题。可以在 "Textual Inversion 和 XTI" 之外的学习中使用。对于DreamBooth方法的token字符串同样有效。

    加权标题的语法与Web UI几乎相同，可以使用(abc)、[abc]、(abc:1.23)等。可以嵌套。由于在括号内包含逗号会导致prompt的shuffle/dropout的括号匹配不正确，请不要在括号内包含逗号。

- `--persistent_data_loader_workers`

    在Windows环境下指定，可以在周期间大幅缩短等待时间。

- `--max_data_loader_n_workers`

    指定数据加载的进程数。进程数越多，数据加载越快，可以更有效地利用GPU，但会消耗主内存。默认是「`8` 或 `CPU并发线程数-1` 的较小值」，如果主内存不足或GPU使用率在90%左右或以上，请根据这些数字将值降低到 `2` 或 `1`。

- `--logging_dir` / `--log_prefix`

    关于保存学习日志的选项。请在logging_dir选项中指定日志保存目录。将以TensorBoard格式保存日志。

    例如，如果指定`--logging_dir=logs`，将在工作目录中创建logs目录，并在其中的日期时间目录中保存日志。
    此外，如果指定`--log_prefix`选项，将在日期时间前添加指定的字符串。请使用如 `--logging_dir=logs --log_prefix=db_style1_` 用于识别。

    要在TensorBoard中查看日志，请在另一个命令提示符中打开并输入以下内容，在工作目录中：

    ```
    tensorboard --logdir=logs
    ```

    （我认为tensorboard在环境准备时会一起安装，如果没有，请使用 `pip install tensorboard` 进行安装。）

    然后打开浏览器，访问http://localhost:6006/ 将显示结果。

- `--log_with` / `--log_tracker_name`

    关于保存学习日志的选项。不仅可以保存到 `tensorboard` ，还可以保存到 `wandb`。详情请参阅 [PR#428](https://github.com/kohya-ss/sd-scripts/pull/428)。

- `--noise_offset`

    这是以下文章的实现：https://www.crosslabs.org//blog/diffusion-with-offset-noise
    
    看起来整体较暗、较亮的图像生成结果可能会改善。看来在LoRA学习中也是有效的。指定 `0.1` 左右的值似乎是好的。

- `--adaptive_noise_scale` （实验性选项）

    这是一个选项，它会根据latents的每个通道的平均值的绝对值自动调整噪声偏移的值。与 `--noise_offset` 同时指定时生效。噪声偏移值由 `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale` 计算得出。由于latent接近正态分布，指定noise_offset的1/10到相同数量级的值可能是好的。

    也可以指定负值，在这种情况下，噪声偏移值将被裁剪到0以上。

- `--multires_noise_iterations` / `--multires_noise_discount`
    
    这是Multi resolution noise (pyramid noise)的设置。详情请参阅 [PR#471](https://github.com/kohya-ss/sd-scripts/pull/471) 以及此页面 [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)。
    
    当在 `--multires_noise_iterations` 中指定数字时，它将生效。值在6到10之间似乎较好。对于 `--multires_noise_discount`，请指定0.1~0.3 的值（在LoRA学习等数据集相对较小的情况下，PR作者推荐），或者指定0.8左右的值（原始文章推荐）（默认值是 0.3）。

- `--debug_dataset`

    使用这个选项，你可以在开始学习之前预览将使用何种图像数据和标题进行学习。按下Esc键将结束并返回命令行。使用`S`键进入下一个步骤（批次），使用`E`键进入下一个周期。

    ※在Linux环境（包括Colab）中，图像不会显示。

- `--vae`

    如果在vae选项中指定Stable Diffusion的checkpoint、VAE的checkpoint文件、Diffuses的模型或VAE（可以指定本地或Hugging Face的模型ID），则使用该VAE进行学习（在缓存latents或学习期间获取latents时）。

    在DreamBooth和fine tuning中，保存的模型将包含这个VAE。

- `--cache_latents` / `--cache_latents_to_disk`

    为了减少使用的VRAM，将VAE的输出缓存在主内存中。除`flip_aug`外，其他数据增强将不可用。此外，整体学习速度会稍微加快。

    如果指定`cache_latents_to_disk`，则将缓存保存到磁盘上。即使在关闭并重新启动脚本后，缓存也会保持有效。

- `--min_snr_gamma`

    指定Min-SNR Weighting策略。详情请参阅[这里](https://github.com/kohya-ss/sd-scripts/pull/308)。在论文中推荐的是`5`。

## 关于模型保存的设置

- `--save_precision`

    指定保存时的数据精度。如果在save_precision选项中指定float、fp16、或bf16，将以该格式保存模型（在DreamBooth或fine tuning中以Diffusers格式保存模型时无效）。当你想要减少模型大小等情况下请使用。

- `--save_every_n_epochs` / `--save_state` / `--resume`

    如果在save_every_n_epochs选项中指定数字，将在每个该周期保存学习过程中的模型。

    同时指定save_state选项时，将同时保存包括optimizer等的学习状态（从保存的模型中也可以恢复学习，但与之相比，可以期待精度提升和学习时间的缩短）。保存位置是一个文件夹。
    
    学习状态将保存在保存位置文件夹中的 `<output_name>-??????-state`（??????是周期数）的文件夹中。在长时间学习时请使用。

    要从保存的学习状态恢复学习，请使用resume选项。请指定学习状态的文件夹（不是`output_dir`，而是其中的state文件夹）。

    由于Accelerator的规格，周期数和全局步数不会被保存，即使在恢复时也会从1开始，请谅解。

- `--save_every_n_steps`

    如果在save_every_n_steps选项中指定数字，将每隔该步数保存学习过程中的模型。可以与save_every_n_epochs同时指定。

- `--save_model_as` （仅DreamBooth, fine tuning）

    可以从`ckpt, safetensors, diffusers, diffusers_safetensors`中选择模型的保存格式。
    
    例如，指定 `--save_model_as=safetensors`。读取Stable Diffusion格式（ckpt或safetensors），并以Diffusers格式保存时，会从Hugging Face下载缺失的信息，用v1.5或v2.1的信息进行补充。

- `--huggingface_repo_id` 等

    如果指定了huggingface_repo_id，在保存模型时会同时上传到HuggingFace。请注意处理访问令牌（请参阅HuggingFace文档）。

    例如，可以指定其他参数如下。

    -   `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`

    如果将huggingface_repo_visibility指定为`public`，则会公开仓库。省略或指定`private`（等非public选项）时，仓库将保持私有。

    如果在指定`--save_state`选项时指定`--save_state_to_huggingface`，state也会被上传。

    如果在指定`--resume`选项时指定`--resume_from_huggingface`，则会从HuggingFace下载state以恢复。此时的`--resume`选项将变为`--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`。
    
    示例：`--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`

    如果指定`--async_upload`选项，将异步进行上传。

## 优化器相关

- `--optimizer_type`
    指定优化器的类型。以下是可以指定的选项：
    - AdamW : [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    - 与过去版本未指定选项时相同。
    - AdamW8bit : 参数与上述相同。
    - PagedAdamW8bit : 参数与上述相同。
    - 与过去版本指定`--use_8bit_adam`时相同。
    - Lion : https://github.com/lucidrains/lion-pytorch
    - 与过去版本指定`--use_lion_optimizer`时相同。
    - Lion8bit : 参数与上述相同。
    - PagedLion8bit : 参数与上述相同。
    - SGDNesterov : [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html), nesterov=True
    - SGDNesterov8bit : 参数与上述相同。
    - DAdaptation(DAdaptAdamPreprint) : https://github.com/facebookresearch/dadaptation
    - DAdaptAdam : 参数与上述相同。
    - DAdaptAdaGrad : 参数与上述相同。
    - DAdaptAdan : 参数与上述相同。
    - DAdaptAdanIP : 参数与上述相同。
    - DAdaptLion : 参数与上述相同。
    - DAdaptSGD : 参数与上述相同。
    - Prodigy : https://github.com/konstmish/prodigy
    - AdaFactor : [Transformers AdaFactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
    - 任意的优化器

- `--learning_rate`

    指定学习率。适当的学習率因学习脚本而异，因此请参考各自的说明。

- `--lr_scheduler` / `--lr_warmup_steps` / `--lr_scheduler_num_cycles` / `--lr_scheduler_power`
  
    关于学习率的调度器相关设置。

    通过lr_scheduler选项，可以从linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup, 以及任意的调度器中选择学习率的调度器。默认是constant。
    
    通过lr_warmup_steps，可以指定调度器的预热（逐渐改变学习率）步数。
    
    lr_scheduler_num_cycles 是 cosine with restarts调度器的重启次数，lr_scheduler_power 是 polynomial调度器的多项式幂。

    详情请自行查找。

    使用任意的调度器时，与任意的优化器相同，可以通过`--scheduler_args`指定选项参数。

### 关于指定优化器

请通过`--optimizer_args`选项指定优化器的参数。以key=value的形式，可以指定多个值。此外，value可以由逗号分隔来指定多个值。例如，如果要为AdamW优化器指定参数，它将如下所示：``--optimizer_args weight_decay=0.01 betas=.9,.999``。

如果指定参数选项，请确认每个优化器的规范。

一些优化器有必需的参数，在省略时会自动添加（如SGDNesterov的momentum）。请检查控制台输出。

D-Adaptation优化器会自动调整学习率。在学习率选项中指定的值不是学习率本身，而是D-Adaptation决定的学习率的应用率，因此通常应指定1.0。如果想给Text Encoder指定U-Net一半的学习率，应指定为``--text_encoder_lr=0.5 --unet_lr=1.0``。

AdaFactor优化器如果指定relative_step=True，就可以自动调整学习率（省略时，默认值会自动添加）。如果要自动调整，将强制使用adafactor_scheduler作为学习率的调度器。看起来指定scale_parameter和warmup_init是有好处的。

对于自动调整的选项指定，例如，它会像这样：``--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True"``。

如果不自动调整学习率，请添加参数选项``relative_step=False``。在这种情况下，学习率的调度器将使用constant_with_warmup，且建议不执行梯度的clip norm。因此，参数将如下所示：``--optimizer_type=adafactor --optimizer_args "relative_step=False" --lr_scheduler="constant_with_warmup" --max_grad_norm=0.0``。

### 使用任意的优化器

如果要使用`torch.optim`中的优化器，只需指定类名（如`--optimizer_type=RMSprop`）；如果要使用其他模块的优化器，需要指定“模块名.类名”（如`--optimizer_type=bitsandbytes.optim.lamb.LAMB`）。

（它仅使用内部的importlib，其实际功能未经过验证。如果需要，请安装相应的包。）

<!-- 
## 在非方形图像上进行学习 --resolution
可以在非方形图像上进行学习。以“宽度,高度”的形式指定resolution，如“448,640”。宽度和高度必须是64的倍数。请调整训练图像和正则化图像的大小。

由于我经常需要生成纵向较长的图像，因此有时会以“448,640”等进行学习。

## 宽高比分桶 --enable_bucket / --min_bucket_reso / --max_bucket_reso
如果指定了enable_bucket选项，该功能将启用。Stable Diffusion在512x512的分辨率下进行训练，但除此之外，它还能够在例如256x768或384x640这样的分辨率下进行训练。

指定此选项时，无需将训练图像和正则化图像统一到特定的分辨率。它会从几个分辨率（宽高比）中选择最优的，并在该分辨率下进行训练。
由于分辨率以64像素为单位，所以可能与原图的宽高比不完全匹配，这种情况下，超出的部分会被轻微裁剪。

可以使用min_bucket_reso选项指定分辨率的最小尺寸，使用max_bucket_reso指定最大尺寸。默认值分别是256和1024。
例如，如果将最小尺寸指定为384，则不会使用256x1024或320x768等分辨率。
如果将分辨率设置为较大的值，如768x768，可能可以将最大尺寸指定为1280等。

然而，当启用Aspect Ratio Bucketing时，对于正则化图像，可能最好准备与训练图像有相似趋势的各种分辨率。

（为了防止一个批次内的图像偏向于训练图像或正则化图像。我认为这可能不会产生太大的影响……。）

## 数据增强 --color_aug / --flip_aug
数据增强是一种通过在训练时动态变换数据来提高模型性能的技术。通过color_aug微妙地改变颜色，通过flip_aug进行左右翻转，进行训练。

由于它是动态变换数据，因此不能与cache_latents选项同时指定。


## 全fp16的梯度训练（实验性功能） --full_fp16
如果指定了full_fp16选项，它将梯度从常规的float32改为float16（fp16）进行训练（这似乎不是混合精度，而是完全的fp16训练）。
这似乎使得在SD1.x的512x512尺寸下可以使用不到8GB的VRAM，在SD2.x的512x512尺寸下可以使用不到12GB的VRAM进行训练。

请预先通过accelerate config指定fp16，并通过选项设置为 ``mixed_precision="fp16"``（bf16不适用）。

为了最小化内存使用量，请指定xformers、use_8bit_adam、cache_latents和gradient_checkpointing的选项，并将train_batch_size设置为1。

（如果内存有余量，逐步增加train_batch_size应该会略微提高精度。）

这是通过向PyTorch源代码打补丁来强制实现的（在PyTorch 1.12.1和1.13.0中进行确认）。精度会大幅下降，训练失败的概率也会增加。
学习率和步数的设置似乎也很苛刻。在了解这些的基础上，请自行承担责任使用。

-->

# 创建元数据文件

## 准备训练数据

如前所述，准备好你想要训练的图像数据，并将其放入任意文件夹。

例如，可以这样存储图像：

![训练数据文件夹的截图](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## 自动加标题

如果你打算无标题仅使用标签进行训练，请跳过此步骤。

如果你打算手动准备标题，请在与训练数据图像相同的目录中，使用相同的文件名和.caption等扩展名准备标题。每个文件将是一个单行的文本文件。

### 使用BLIP进行标题添加

在最新版本中，不再需要下载BLIP、权重或添加虚拟环境。它现在可以直接运行。

在finetune文件夹中运行make_captions.py。

```
python finetune\make_captions.py --batch_size <batch_size> <训练数据文件夹>
```

例如，如果batch_size为8，训练数据位于父文件夹的train_data中，将如下所示：

```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

标题文件将在与训练数据图像相同的目录中创建，使用相同的文件名和.caption扩展名。

请根据GPU的VRAM容量调整batch_size。较大的值会运行得更快（我认为即使在VRAM为12GB时，也可以进一步增加）。
你可以使用max_length选项指定标题的最大长度。默认值是75。如果你打算使用225个令牌训练模型，可能可以设置更长的长度。
caption_extension选项可以更改标题的扩展名。默认是.caption（如果设置为.txt，会与后述的DeepDanbooru冲突）。

如果有多个训练数据文件夹，请对每个文件夹分别执行操作。

由于推理中存在随机性，每次运行结果都会不同。如果要固定结果，请使用--seed选项指定如 `--seed 42` 的随机数种子。

其他选项可通过 `--help` 查看帮助（参数的含义似乎没有整理成文档，可能需要查看源代码）。

默认情况下，将生成扩展名为.caption的标题文件。

![生成标题的文件夹](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

例如，可能会添加以下标题：

![标题和图像](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## 通过DeepDanbooru进行标签添加

如果不直接进行danbooru标签的标注，请跳至“标题和标签信息预处理”。

标签添加可使用DeepDanbooru或WD14Tagger进行。WD14Tagger似乎更准确。如果选择使用WD14Tagger进行标签添加，请跳至下一章节。

### 环境准备

从作业文件夹中克隆DeepDanbooru https://github.com/KichangKim/DeepDanbooru 或下载并解压zip文件。我选择了解压zip文件。
同时，从DeepDanbooru的Releases页面 https://github.com/KichangKim/DeepDanbooru/releases “DeepDanbooru Pretrained Model v3-20211112-sgd-e28”的Assets中，下载deepdanbooru-v3-20211112-sgd-e28.zip并解压至DeepDanbooru文件夹。

从以下链接下载。点击Assets打开并从那里下载。

![DeepDanbooru下载页面](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

请按照以下目录结构进行设置

![DeepDanbooru的目录结构](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)

安装Diffusers环境所需的库。在DeepDanbooru文件夹中进行安装（实际上只有tensorflow-io会被添加）。

```
pip install -r requirements.txt
```

接下来安装DeepDanbooru本身。

```
pip install .
```

至此，标签添加的环境准备已经完成。

### 执行标签添加
移动至DeepDanbooru文件夹，运行deepdanbooru进行标签添加。

```
deepdanbooru evaluate <教师数据文件夹> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

如果将教师数据放置于父文件夹的train_data中，则命令如下。

```
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

标签文件将在与教师数据图像相同的目录下创建，具有相同的文件名，扩展名为.txt。由于逐个处理，因此速度较慢。

如果有多个教师数据文件夹，请对每个文件夹分别执行。

生成如下：

![DeepDanbooru生成文件](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

标签将这样被添加（信息量巨大……）。

![DeepDanbooru标签和图像](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## 通过WD14Tagger进行标签添加

以下是使用WD14Tagger代替DeepDanbooru的步骤。

我们将使用Automatic1111先生WebUI中使用的tagger。参考了以下github页面的信息（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger ）。

在初次环境准备时，已安装所需的模块，权重会自动从Hugging Face下载。

### 执行标签添加

运行脚本进行标签添加。
```
python tag_images_by_wd14_tagger.py --batch_size <批处理大小> <教师数据文件夹>
```

如果将教师数据放置于父文件夹的train_data中，命令如下。
```
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

首次启动时，模型文件将自动下载至wd14_tagger_model文件夹（通过选项可更改文件夹）。结果如下：

![下载的文件](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

标签文件将在与教师数据图像相同的目录下创建，具有相同的文件名，扩展名为.txt。

![生成的标签文件](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![标签和图像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

通过thresh选项，可以指定判定的标签的confidence（确信度）高于多少时才添加标签。默认值为与WD14Tagger样本相同的0.35。降低此值会添加更多标签，但精度会下降。

batch_size应根据GPU的VRAM容量进行调整。值越大，速度越快（即使VRAM为12GB，我认为也可以进一步增加）。caption_extension选项可以更改标签文件的扩展名。默认为.txt。

model_dir选项可以指定模型的保存位置。

使用force_download选项，则即使保存位置已存在模型，也会重新下载。

如果有多个教师数据文件夹，请对每个文件夹分别执行。

## 标题和标签信息的预处理

为了便于处理脚本，我们将标题和标签汇总为一个文件的元数据。

### 标题的预处理

若要将标题添加至元数据，请在作业文件夹中运行以下命令（如果不使用标题进行训练，无需执行）（实际上，将是一行命令，以下同）。使用`--full_path`选项将图像文件的完整路径存储在元数据中。省略此选项将记录为相对路径，但需要在`.toml`文件中另行指定文件夹。

```
python merge_captions_to_metadata.py --full_path <教师数据文件夹>
    --in_json <要加载的元数据文件名> <元数据文件名>
```

元数据文件名可随意命名。
如果教师数据位于train_data中，无需加载的元数据文件，元数据文件名为meta_cap.json，则命令如下。

```
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

caption_extension选项可指定标题的扩展名。

如果有多个教师数据文件夹，请指定full_path参数，并对每个文件夹分别执行。

```
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```

省略in_json选项，如果存在写入目标元数据文件，则将从那里加载，并覆盖其内容。

__※推荐每次更改in_json选项和写入目标，安全地写入到不同的元数据文件中。__

### 标签预处理

同样地，我们也将标签汇总到元数据中（如果不使用标签进行训练，无需执行）。
```
python merge_dd_tags_to_metadata.py --full_path <教师数据文件夹> 
    --in_json <要加载的元数据文件名> <要写入的元数据文件名>
```

如果使用相同的目录结构，从meta_cap.json加载，并写入meta_cap_dd.json，命令如下。
```
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

如果有多个教师数据文件夹，请指定full_path参数，并对每个文件夹分别执行。

```
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

省略in_json选项，如果存在写入目标元数据文件，则将从那里加载，并覆盖其内容。

__※推荐每次更改in_json选项和写入目标，安全地写入到不同的元数据文件中。__

### 标题和标签的清理

到目前为止，元数据文件中已经汇总了标题和DeepDanbooru的标签。但是，自动标题可能有表示变化等问题，略显微妙（※）；同时，标签中可能包含下划线或有rating（在DeepDanbooru的情况下），因此，我们建议使用编辑器的替换功能等对标题和标签进行清理。

※例如，当学习动漫少女的图像时，标题中可能会有girl/girls/woman/women等不一致的描述。同时，"anime girl"等描述可能更适宜简化为"girl"。

我们已经准备了清理脚本，根据情况编辑脚本内容即可使用。

（无需指定教师数据文件夹。将清理元数据中的所有数据。）

```
python clean_captions_and_tags.py <要加载的元数据文件名> <要写入的元数据文件名>
```

请注意，无需使用--in_json。例如，命令如下。

```
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

至此，标题和标签的预处理已完成。

## 预获取latents

※ 此步骤并非必须。即使省略，也可以在训练时一边获取latents一边进行学习。
如果在训练时进行`random_crop`或`color_aug`等操作，则无法预获取latents（因为每次学习都需要改变图像）。如果不进行预获取，在此阶段的元数据即可用于训练。

我们将预先获取图像的潜在表示并保存在磁盘上。这样可以加快学习速度。同时，我们也会进行bucketing（根据教师数据的宽高比进行分类）。

请在作业文件夹中输入以下命令。
```
python prepare_buckets_latents.py --full_path <教师数据文件夹>  
    <要加载的元数据文件名> <要写入的元数据文件名> 
    <用于微调的模型名或checkpoint> 
    --batch_size <批处理大小> 
    --max_resolution <分辨率 宽,高> 
    --mixed_precision <精度>
```

如果模型为model.ckpt，批处理大小为4，学习分辨率为512*512，精度为no（float32），从meta_clean.json加载元数据，并写入至meta_lat.json，命令如下。

```
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

以numpy的npz格式在教师数据文件夹中保存latents。

可以使用--min_bucket_reso选项指定分辨率的最小尺寸，使用--max_bucket_reso指定最大尺寸。默认值分别为256和1024。例如，如果指定最小尺寸为384，将不会使用256*1024或320*768等分辨率。
如果将分辨率设置为768*768等较大值，建议将最大尺寸设置为1280等值。

如果指定--flip_aug选项，将进行左右翻转的augmentation（数据扩增）。虽然可以将数据量看似翻倍，但如果数据并非左右对称（例如角色的外观、发型等），指定此选项可能会导致学习失败。

（对于翻转的图像，也将获取latents，并保存为\*\_flip.npz文件。对于fline_tune.py，无需特别指定选项。如果存在\_flip的文件，会随机读取flip和无flip的文件。）

批处理大小可能在VRAM 12GB时可进一步增加。
分辨率应为可被64整除的数字，以"宽,高"指定。分辨率直接影响fine tuning时的内存大小。在VRAM 12GB时，512,512可能为极限（※）。在16GB时，可能可以将分辨率设置为512,704或512,768。然而，即使设置为256,256，也可能在VRAM 8GB时依然很紧张（因为参数和optimizer等所需的内存与分辨率无关，始终为固定值）。

※有报告称，在batch size 1的学习中，12GB VRAM，640,640分辨率可以运行。

将显示bucketing的结果如下：

![bucketing结果](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

如果有多个教师数据文件夹，请指定full_path参数，并对每个文件夹分别执行。
```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

```
加载源和写入目标可以相同，但分开更安全。

__※每次更改参数，并写入到不同的元数据文件中，会更安全。__

