这是针对NovelAI提出的训练方法，自动captioning，标签添加，以及对Windows和VRAM 12GB（在SD v1.x的情况下）环境进行了优化的fine tuning。在这里，fine tuning是指使用图像和caption来训练模型（不包括LoRA，Textual Inversion，Hypernetworks）。

请同时参阅[关于训练的通用文档](./train_README-zh.md)。

# 概述

我们使用Diffusers对Stable Diffusion的U-Net进行fine tuning。我们对应了NovelAI文章中提出的以下改进（对于Aspect Ratio Bucketing，我们参考了NovelAI的代码，但最终代码都是原创的）。

* 使用CLIP（Text Encoder）倒数第二层的输出，而非最后一层的输出。
* 支持非正方形分辨率的训练（Aspect Ratio Bucketing）。
* 将令牌长度从75扩展至225。
* 使用BLIP进行captioning（自动创建caption），使用DeepDanbooru或WD14Tagger进行自动标签添加。
* 支持Hypernetwork的训练。
* 支持Stable Diffusion v2.0（base和768/v）。
* 通过预先获取并保存VAE的输出到磁盘，实现训练的内存节省和加速。

默认情况下，不会训练Text Encoder。在对整个模型进行fine tuning时，似乎通常只训练U-Net（NovelAI似乎也是这样）。通过选项指定，也可以将Text Encoder设为训练目标。

# 关于附加功能

## CLIP输出的更改

为了将prompt反映到图像中，需要将文本转换为特征，这就是CLIP（Text Encoder）的作用。在Stable Diffusion中，使用的是CLIP最后层的输出，但我们可以改为使用倒数第二层的输出。根据NovelAI的说法，这样可以更准确地反映prompt。
也可以继续使用原始的，最后一层的输出。

※在Stable Diffusion 2.0中，默认使用倒数第二层。请勿指定clip_skip选项。

## 非正方形分辨率的训练

虽然Stable Diffusion是在512*512下训练的，但此方法还可以在如256*1024或384*640这样的分辨率下进行训练。这样可以减少裁剪的部分，期望可以更准确地学习prompt和图像的关系。
训练分辨率将在不超过作为参数给出的分辨率面积（=内存使用量）的范围内，以64像素为单位，在纵向和横向上进行调整和创建。

在机器学习中，通常会统一输入大小，但这并没有特别的限制，实际上，只要在同一个batch内统一就足够了。NovelAI所说的bucketing，似乎是指预先根据长宽比将教师数据分类至相应的训练分辨率。然后，通过从每个bucket内的图像创建batch，统一batch内的图像大小。

## 从75到225的令牌长度扩展

在Stable Diffusion中，最大为75令牌（包括开始和结束共77令牌），但我们将此扩展至225令牌。
然而，CLIP可以接受的最大长度为75令牌，因此，对于225令牌，我们将其简单地分为三部分，分别调用CLIP，然后链接结果。

※我并不完全确定这是理想的实现。但似乎确实可以运行。特别是在2.0中，没有可以参考的实现，所以我独自实现了它。

※Automatic1111先生的Web UI似乎有意识地使用逗号等进行分割，但我的实现并没有那么复杂，只是简单地分割。

# 训练的步骤

请参照此仓库的README，先进行环境准备。

## 数据准备

请参阅[关于准备训练数据](./train_README-zh.md)。fine tuning只支持使用元数据的fine tuning方法。

## 执行训练
例如，可以这样执行。以下是为了节省内存的设置。请根据需要更改每一行。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=<.ckpt或者.safetensors或者Diffusers版模型的目录> 
    --output_dir=<训练后模型的输出目录>  
    --output_name=<训练后模型输出时的文件名> 
    --dataset_config=<数据准备中创建的.toml文件> 
    --save_model_as=safetensors 
    --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=fp16
```

`num_cpu_threads_per_process`通常指定为1。

`pretrained_model_name_or_path`指定要进行额外训练的模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers的本地磁盘上的模型目录、或Diffusers的模型ID（如"stabilityai/stable-diffusion-2"）。

`output_dir`指定保存训练后模型的目录。`output_name`指定模型的文件名，不包括扩展名。`save_model_as`指定保存为safetensors格式。

`dataset_config`指定`.toml`文件。在文件中，为了最初减少内存消耗，将batch size设置为`1`。

我们将训练步数`max_train_steps`设置为10000。学习率`learning_rate`在这里设置为5e-6。

为了节省内存，我们指定`mixed_precision="fp16"`（在RTX30系列及以后的版本中也可以指定`bf16`。请与在环境准备时对accelerate进行的设置相匹配）。同时，我们指定`gradient_checkpointing`。

为了使用内存消耗较少的8bit AdamW作为优化器（将模型调整至适合训练数据的类），我们指定`optimizer_type="AdamW8bit"`。

指定`xformers`选项，使用xformers的CrossAttention。如果没有安装xformers或导致错误（取决于环境，如`mixed_precision="no"`的情况下），可以指定`mem_eff_attn`选项使用节省内存的CrossAttention（速度会变慢）。

如果内存足够，编辑`.toml`文件，将batch size增加至例如`4`（可能可以提高速度和精度）。

### 关于常用选项

在以下情况下，请参考关于选项的文档。

- 训练Stable Diffusion 2.x或其衍生模型
- 训练假定clip skip超过2的模型
- 使用超过75令牌的caption进行训练

### 关于batch size

与LoRA等训练相比，因为要训练整个模型，所以内存消耗会更多（与DreamBooth相同）。

### 关于学习率

大约1e-6到5e-6是常见的。请参考其他fine tuning的例子。

### 如果使用旧形式的数据集指定的命令行

您需要在选项中指定分辨率和批处理大小。命令行示例如下所示。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=model.ckpt 
    --in_json meta_lat.json 
    --train_data_dir=train_data 
    --output_dir=fine_tuned 
    --shuffle_caption 
    --train_batch_size=1 --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=bf16
    --save_every_n_epochs=4
```

<!-- 
### 使用fp16的梯度进行学习（实验功能）
如果指定full_fp16选项，将把梯度从通常的float32更改为float16（fp16）进行学习（这将变成完整的fp16学习，而非混合精度）。这样，似乎可以在SD1.x的512*512大小下使用不到8GB的VRAM，在SD2.x的512*512大小下使用不到12GB的VRAM进行学习。

请预先在accelerate config中指定fp16，并在选项中使用mixed_precision="fp16"（bf16将无法工作）。

为了将内存使用量降至最低，请指定xformers、use_8bit_adam、gradient_checkpointing的各选项，并将train_batch_size设为1。
（如果可能，逐步增加train_batch_size应该会稍微提高精度。）

我通过在PyTorch源代码上打补丁强行实现（在PyTorch 1.12.1和1.13.0中确认）。精度会大幅降低，而且学习失败的可能性也会增加。学习率和步数的设置似乎也很严格。在了解这些之后，请自行负责使用。
-->

# fine tuning特有的其他主要选项

请参考其他文档以获取所有选项的信息。

## `train_text_encoder`
将Text Encoder设为训练目标。这将略微增加内存使用量。

通常的fine tuning不会将Text Encoder设为训练目标（可能是因为U-Net被训练以遵循Text Encoder的输出），但在训练数据量较少的情况下，似乎像DreamBooth那样让Text Encoder进行学习也是有效的。

## `diffusers_xformers`
不使用脚本特有的xformers替换功能，而是利用Diffusers的xformers功能。这样将无法进行Hypernetwork的训练。
