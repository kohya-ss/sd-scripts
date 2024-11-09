# 关于LoRA的学习

这是将[LoRA: Large Language Models的低秩适应](https://arxiv.org/abs/2106.09685)（arxiv）、[LoRA](https://github.com/microsoft/LoRA)（github）应用到Stable Diffusion上的内容。

我大量参考了[cloneofsimo先生的仓库](https://github.com/cloneofsimo/lora)，非常感谢。

通常的LoRA仅应用于Linear和内核大小为1x1的Conv2d，但也可以扩展应用到内核大小为3x3的Conv2d。

对Conv2d 3x3的扩展是由[cloneofsimo先生](https://github.com/cloneofsimo/lora)首次发布，KohakuBlueleaf先生在他的[LoCon](https://github.com/KohakuBlueleaf/LoCon)中揭示了它的有效性。我深深感谢KohakuBlueleaf先生。

看起来它在8GB VRAM上也能勉强运行。

请同时查看[关于学习的通用文档](./train_README-zh.md)。

# 可以学习的LoRA的类型

我们将支持以下两种类型。以下是本仓库内独特的命名。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers，读作“リエラ”)

    应用于Linear和内核大小为1x1的Conv2d的LoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ onvolutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers，读作“セリア”)

    除了1.，还应用于内核大小为3x3的Conv2d的LoRA

与LoRA-LierLa相比，LoRA-C3Liar由于可以应用于更多的层，因此可能期待更高的精度。

在学习时，也可以使用__DyLoRA__（将在后文讨论）。

## 关于学习模型的注意事项

LoRA-LierLa 可以在AUTOMATIC1111先生的Web UI的LoRA功能中使用。

要使用LoRA-C3Liar在Web UI中生成，请使用这个[WebUI用extension](https://github.com/kohya-ss/sd-webui-additional-networks)。

你也可以使用本仓库内的脚本，将学习后的LoRA模型预先合并到Stable Diffusion的模型中。

它与cloneofsimo先生的仓库，以及d8ahazard先生的[Stable-Diffusion-WebUI的Dreambooth Extension](https://github.com/d8ahazard/sd_dreambooth_extension)，在当前阶段是不兼容的。这是因为我们进行了一些功能扩展（将在后文讨论）。

# 学习的步骤

请先参考这个仓库的README，进行环境准备。

## 准备数据

请参照[准备学习数据](./train_README-zh.md)。




## 执行学习

使用`train_network.py`。

在`train_network.py`中，通过`--network_module`选项指定学习目标的模块名。对于LoRA，应为`network.lora`，请指定这个。

学习率建议定得比通常的DreamBooth或fine tuning高一些，大约`1e-4`～`1e-3`。

以下是一个命令行的例子。

```
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=<.ckpt或者.safetensord或者Diffusers版模型的目录> 
    --dataset_config=<在数据准备中创建的.toml文件> 
    --output_dir=<学习后的模型输出的文件夹>  
    --output_name=<学习后的模型输出时的文件名> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=400 
    --learning_rate=1e-4 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora
```

在这个命令行中，将学习 LoRA-LierLa。

LoRA的模型将被保存在由`--output_dir`选项指定的文件夹中。关于其他选项，如优化器等，请参阅[学习的通用文档](./train_README-ja.md)中的“常用选项”。

此外，可以指定以下选项。

* `--network_dim`
  * 指定LoRA的RANK（如`--networkdim=4`）。如果省略，默认为4。数量越多，表达力越强，但所需的学习内存和时间也会增加。盲目增加似乎并不好。
* `--network_alpha`
  * 为了防止下溢并稳定学习，指定``alpha``值。默认是1。如果指定与`network_dim`相同的值，它将像以前版本一样运行。
* `--persistent_data_loader_workers`
  * 在Windows环境中指定此选项将大大缩短epoch之间的等待时间。
* `--max_data_loader_n_workers`
  * 指定数据加载的进程数。进程数越多，数据加载越快，GPU可以更有效地使用，但会消耗主内存。默认为“`8`或`CPU并发线程数-1`的较小值”，如果主内存不足或GPU使用率在90%左右或以上时，请根据这些数字将值减少到`2`或`1`左右。
* `--network_weights`
  * 在学习前加载预先训练的LoRA的权重，然后从那里继续学习。
* `--network_train_unet_only`
  * 只有效U-Net相关的LoRA模块。如果用于类似fine tuning的学习，可能效果良好。
* `--network_train_text_encoder_only`
  * 只有效Text Encoder相关的LoRA模块。也许可以期待Textual Inversion的效果。
* `--unet_lr`
  * 当使用与常规学习率（由`--learning_rate`选项指定）不同的学习率来学习与U-Net相关的LoRA模块时，指定此选项。
* `--text_encoder_lr`
  * 当使用与常规学习率（由`--learning_rate`选项指定）不同的学习率来学习与Text Encoder相关的LoRA模块时，指定此选项。据说将Text Encoder的学习率设置得稍低一些（如5e-5）可能更好。
* `--network_args`
  * 可以指定多个参数。下面将进行说明。

当`--network_train_unet_only`和`--network_train_text_encoder_only`两者都未指定时（默认），Text Encoder和U-Net的LoRA模块都将生效。

# 其他的学习方法

## 学习LoRA-C3Lier

请在`--network_args`中按以下方式指定。使用`conv_dim`指定Conv2d (3x3)的rank，使用`conv_alpha`指定alpha。

```
--network_args "conv_dim=4" "conv_alpha=1"
```

如果省略alpha，默认为1，如下所示。

```
--network_args "conv_dim=4"
```

## DyLoRA

DyLoRA是由以下论文提出的。[DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://arxiv.org/abs/2210.07558) 官方实现可以在[这里](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)找到。

根据论文，LoRA的rank并非越高越好，需要根据目标模型、数据集和任务寻找适当的rank。使用DyLoRA，你可以在指定的dim(rank)以下的各种rank同时学习LoRA。这可以省去为每个需要寻找最佳rank的麻烦。

本仓库的实现是在官方实现的基础上进行了一些自定义扩展（因此可能会存在一些问题）。

### 本仓库的DyLoRA的特点

学习后的DyLoRA模型文件与LoRA兼容。此外，你可以从模型文件中抽取指定dim(rank)以下的多个dim的LoRA。

你可以学习DyLoRA-LierLa或DyLoRA-C3Lier。

### 使用DyLoRA学习

请指定DyLoRA对应的`network.dylora`，如下所示`--network_module=networks.dylora`。

同时，在`--network_args`中，例如，指定`--network_args "unit=4"`中的`unit`。`unit`是分割rank的单位。例如，你可以指定`--network_dim=16 --network_args "unit=4"`。请将`unit`设置为可以整除`network_dim`的值（`network_dim`是`unit`的倍数）。

如果不指定`unit`，则视为`unit=1`。

示例如下。

```
--network_module=networks.dylora --network_dim=16 --network_args "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "unit=4"
```

对于DyLoRA-C3Lier，你可以在`--network_args`中指定`"conv_dim=4"`这样的`conv_dim`。与常规LoRA不同，`conv_dim`需要与`network_dim`的值相同。示例如下。

```
--network_module=networks.dylora --network_dim=16 --network_args "conv_dim=16" "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "conv_dim=32" "conv_alpha=16" "unit=8"
```

例如，如果在dim=16，unit=4（后面会详细说明）下学习，你可以学习并抽取4、8、12、16这4个rank的LoRA。通过使用每个抽取的模型生成图像并进行比较，你可以选择最佳rank的LoRA。

其他选项与常规LoRA相同。

※ `unit`是本仓库的自定义扩展，因为在DyLoRA中，与同dim(rank)的常规LoRA相比，学习时间可能会更长，因此我们增加了分割单位。

### 从DyLoRA模型中抽取LoRA模型

使用`networks`文件夹内的`extract_lora_from_dylora.py`。以指定的`unit`单位，从DyLoRA模型中抽取LoRA模型。

命令行示例如下。

```powershell
python networks\extract_lora_from_dylora.py --model "foldername/dylora-model.safetensors" --save_to "foldername/dylora-model-split.safetensors" --unit 4
```

在`--model`中，指定DyLoRA模型文件。在`--save_to`中，指定保存抽取模型的文件名（rank的数值会添加到文件名中）。在`--unit`中，指定DyLoRA学习时的`unit`。

## 层级学习率

详情请参阅[PR #355](https://github.com/kohya-ss/sd-scripts/pull/355)。

SDXL当前不支持。

你可以指定全模型的25个块的权重。不存在对应于第一个块的LoRA，但我们为了与层级LoRA应用等的兼容性而设置为25个。此外，即使在不扩展到conv2d3x3的情况下，部分块可能也不存在LoRA，但为了统一描述，我们始终需要指定25个值。

请在`--network_args`中指定以下参数。

- `down_lr_weight`：指定U-Net的down blocks的学习率的权重。你可以如下指定。
  - 每个块的权重：像`"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`这样指定12个数值。
  - 从预设指定：像`"down_lr_weight=sine"`这样指定（使用正弦曲线指定权重）。可以指定sine, cosine, linear, reverse_linear, zeros。此外，你可以通过像`"down_lr_weight=cosine+.25"`这样添加 `+数字` 来增加指定的数值（它会变成0.25~1.25）。
- `mid_lr_weight`：指定U-Net的mid block的学习率的权重。你只需要指定一个数值，像`"mid_lr_weight=0.5"`这样。
- `up_lr_weight`：指定U-Net的up blocks的学习率的权重。与down_lr_weight相同。
- 省略的部分将被视为1.0。此外，如果你将权重设置为0，那么该块的LoRA模块将不会创建。
- `block_lr_zero_threshold`：如果权重低于这个值，LoRA模块将不会创建。默认是0。

### 层级学习率命令行示例:

```powershell
--network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"

--network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"
```

### 层级学习率toml文件示例:

```toml
network_args = [ "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5", "mid_lr_weight=2.0", "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5",]

network_args = [ "block_lr_zero_threshold=0.1", "down_lr_weight=sine+.5", "mid_lr_weight=1.5", "up_lr_weight=cosine+.5", ]
```

## 层级dim (rank)

你可以指定全模型的25个块的dim (rank)。与层级学习率类似，部分块可能不存在LoRA，但你需要始终指定25个值。

请在`--network_args`中指定以下参数。

- `block_dims`：指定每个块的dim (rank)。像`"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`这样指定25个数值。
- `block_alphas`：指定每个块的alpha。与block_dims一样，你需要指定25个数值。如果省略，将使用network_alpha的值。
- `conv_block_dims`：扩展LoRA到Conv2d 3x3，并指定每个块的dim (rank)。
- `conv_block_alphas`：当你扩展LoRA到Conv2d 3x3时，指定每个块的alpha。如果省略，将使用conv_alpha的值。

### 层级dim (rank)命令行示例:

```powershell
--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"
```

### 层级dim (rank)toml文件示例:

```toml
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2",]
  
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2", "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",]
```

# 其他脚本

这是与LoRA相关的合并等脚本集合。

## 关于合并脚本

你可以使用merge_lora.py将LoRA的学习结果合并到Stable Diffusion的模型中，或者合并多个LoRA模型。

我们为SDXL准备了sdxl_merge_lora.py。选项等是相同的，所以请将以下merge_lora.py替换为阅读。

### 将LoRA模型合并到Stable Diffusion的模型中

合并后的模型可以像常规的Stable Diffusion ckpt一样处理。例如，命令行可能如下所示。

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors --ratios 0.8
```

如果你在Stable Diffusion v2.x的模型上进行了学习，并且要将其合并，请指定--v2选项。

--sd_model选项中，指定要作为合并源的Stable Diffusion模型文件（仅支持.ckpt或.safetensors，目前不支持Diffusers）。

在--save_to选项中，指定合并后模型的保存位置（.ckpt或.safetensors，根据扩展名自动判断）。

在--models中，指定学习过的LoRA模型文件。可以指定多个，如果是多个，将依次进行合并。

在--ratios中，指定每个模型的适用率（即多少权重反映到原模型中）为0~1.0的数值。例如，如果接近过度学习，降低适用率可能会有所改善。请指定与模型数量相同数量的值。

如果你指定多个，如下所示。

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.8 0.5
```

### 合并多个LoRA模型

如果你指定--concat选项，你可以将多个LoRA简单地组合起来创建一个新的LoRA模型。文件大小（以及dim/rank）将是指定的LoRA的总大小（如果你想在合并时改变dim (rank)，请使用`svd_merge_lora.py`）。

例如，命令行可能如下所示。

```
python networks\merge_lora.py --save_precision bf16 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 1.0 -1.0 --concat --shuffle
```

你指定--concat选项。

另外，添加--shuffle选项以打乱权重。如果不打乱，可以从合并后的LoRA中提取出原始的LoRA，因此在学习复印机等情况下，学习源数据将变得明显。请注意。

在--save_to选项中，指定合并后LoRA模型的保存位置（.ckpt或.safetensors，根据扩展名自动判断）。

在--models中，指定学习过的LoRA模型文件。你可以指定三个或更多。

在--ratios中，指定每个模型的比率（即多少权重反映到原模型中）为0~1.0的数值。如果你想一对一地合并两个模型，将是「0.5 0.5」。「1.0 1.0」将使总权重过大，结果可能不理想。

你不能合并v1中学习的LoRA和v2中学习的LoRA，也不能合并不同rank（维度数）的LoRA。理论上，你可能可以合并只有U-Net的LoRA和U-Net+Text Encoder的LoRA，但结果是未知的。

### 其他选项

* precision
  * 你可以从float、fp16、bf16中指定合并计算时的精度。如果省略，为了保证精度，将使用float。如果你想减少内存使用，请指定fp16/bf16。
* save_precision
  * 你可以从float、fp16、bf16中指定模型保存时的精度。如果省略，默认为与precision相同的精度。

还有其他几个选项，请使用--help来查看。

## 合并具有不同rank的多个LoRA模型

将多个LoRA近似为一个LoRA（无法完全重现）。使用`svd_merge_lora.py`。例如，命令行可能如下所示。

```
python networks\svd_merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 0.6 0.4 --new_rank 32 --device cuda
```

这与`merge_lora.py`的主要选项相同。以下选项被添加了。

- `--new_rank`
  - 指定要创建的LoRA的rank。
- `--new_conv_rank`
  - 指定要创建的Conv2d 3x3 LoRA的rank。如果省略，将与`new_rank`相同。
- `--device`
  - 如果指定为`--device cuda`，计算将在GPU上进行。处理会更快。

## 在本仓库的图像生成脚本中生成

请在gen_img_diffusers.py中添加--network_module和--network_weights的选项。它们的意义与学习时相同。

通过在--network_mul选项中指定0~1.0的数值，你可以改变LoRA的适用率。

## 在Diffusers的pipeline中生成

请参考以下示例。你只需要networks/lora.py这个文件。Diffusers的版本如果不是0.10.2，可能无法运行。

```python
import torch
from diffusers import StableDiffusionPipeline
from networks.lora import LoRAModule, create_network_from_weights
from safetensors.torch import load_file

# if the ckpt is CompVis based, convert it to Diffusers beforehand with tools/convert_diffusers20_original_sd.py. See --help for more details.

model_id_or_dir = r"model_id_on_hugging_face_or_dir"
device = "cuda"

# create pipe
print(f"creating pipe from {model_id_or_dir}...")
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_dir, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)
vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet

# load lora networks
print(f"loading lora networks...")

lora_path1 = r"lora1.safetensors"
sd = load_file(lora_path1)   # If the file is .ckpt, use torch.load instead.
network1, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
network1.apply_to(text_encoder, unet)
network1.load_state_dict(sd)
network1.to(device, dtype=torch.float16)

# # You can merge weights instead of apply_to+load_state_dict. network.set_multiplier does not work
# network.merge_to(text_encoder, unet, sd)

lora_path2 = r"lora2.safetensors"
sd = load_file(lora_path2) 
network2, sd = create_network_from_weights(0.7, None, vae, text_encoder,unet, sd)
network2.apply_to(text_encoder, unet)
network2.load_state_dict(sd)
network2.to(device, dtype=torch.float16)

lora_path3 = r"lora3.safetensors"
sd = load_file(lora_path3)
network3, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
network3.apply_to(text_encoder, unet)
network3.load_state_dict(sd)
network3.to(device, dtype=torch.float16)

# prompts
prompt = "masterpiece, best quality, 1girl, in white shirt, looking at viewer"
negative_prompt = "bad quality, worst quality, bad anatomy, bad hands"

# exec pipe
print("generating image...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]

# if not merged, you can use set_multiplier
# network1.set_multiplier(0.8)
# and generate image again...

# save image
image.save(r"by_diffusers..png")
```

## 从两个模型的差异创建LoRA模型

这是根据[这个讨论](https://github.com/cloneofsimo/lora/discussions/56)实施的。我直接使用了公式（虽然我没有完全理解，但似乎在近似中使用了奇异值分解）。

它将使用LoRA来近似两个模型（例如，fine tuning的源模型和fine tuning后的模型）的差异。

### 脚本的执行方法

请按以下方式指定。
```
python networks\extract_lora_from_models.py --model_org base-model.ckpt
    --model_tuned fine-tuned-model.ckpt 
    --save_to lora-weights.safetensors --dim 4
```

在--model_org选项中，指定原始的Stable Diffusion模型。如果要应用创建的LoRA模型，你将需要指定这个模型来应用。可以指定.ckpt或.safetensors。

在--model_tuned选项中，指定要从中提取差异的目标Stable Diffusion模型。例如，你可以指定fine tuning或DreamBooth后的模型。可以指定.ckpt或.safetensors。

在--save_to中，指定LoRA模型的保存位置。在--dim中，指定LoRA的维度数。

生成的LoRA模型可以像学习过的LoRA模型一样使用。

如果两个模型的Text Encoder相同，LoRA将成为仅U-Net的LoRA。

### 其他选项

- `--v2`
  - 如果你使用的是v2.x的Stable Diffusion模型，请指定此选项。
- `--device`
  - 如果你指定为`--device cuda`，计算将在GPU上进行。处理速度会更快（在CPU上也不会特别慢，最多可能是两到三倍的速度差异）。
- `--save_precision`
  - 你可以从"float", "fp16", "bf16"中指定LoRA的保存格式。如果省略，将默认为float。
- `--conv_dim`
  - 如果你指定，LoRA的应用范围将扩展到Conv2d 3x3。指定Conv2d 3x3的rank。

## 图像重置脚本

（稍后我会整理文档，但目前先在这里说明一下。）

通过Aspect Ratio Bucketing功能的扩展，现在可以不对小图像进行放大，而是直接使用它们作为训练数据。我收到了报告，称在训练数据中加入原始图像的缩小版本可以提高精度，同时我也收到了预处理脚本，所以我整理并添加了它。感谢bmaltais先生。

### 脚本的执行方法

请按以下方式指定。原始图像和调整大小后的图像将被保存在目标文件夹中。调整大小后的图像的文件名中会附加``+512x512``这样的目标分辨率（这与图像大小不同）。小于调整后分辨率的图像不会被放大。

```
python tools\resize_images_to_resolution.py --max_resolution 512x512,384x384,256x256 --save_as_png 
    --copy_associated_files 原始图像文件夹 目标文件夹
```

原始图像文件夹中的图像文件将被调整大小，以使其具有与指定的分辨率（可以指定多个）相同的面积，并保存在目标文件夹中。非图像文件将被直接复制。

在``--max_resolution``选项中，请像示例那样指定调整大小后的尺寸。图像将被调整大小，使其面积与该尺寸相匹配。如果你指定多个尺寸，图像将被调整为每个尺寸。如果你指定``512x512,384x384,256x256``，那么目标文件夹中的图像将有总共四张，包括原始大小和三张调整大小后的图像。

如果你指定``--save_as_png``选项，图像将以png格式保存。如果省略，将以jpeg格式（quality=100）保存。

如果你指定``--copy_associated_files``选项，与图像具有相同文件名（不包括扩展名，例如caption等）的文件将被复制，其文件名与调整大小后的图像的文件名相同。


### 其他选项

- divisible_by
  - 为了使调整大小后的图像的尺寸（高和宽）可以被这个值整除，它会从图像中心裁剪。
- interpolation
  - 指定缩小时的插值方法。可以从``area, cubic, lanczos4``中选择，默认为``area``。


# 额外信息

## 与cloneofsimo先生的仓库的区别

截至2022/12/25，本仓库已经扩展了LoRA的应用位置，包括Text Encoder的MLP，U-Net的FFN，Transformer的in/out projection，从而提高了表达力。然而，作为代价，内存消耗增加，现在是8GB的极限。

此外，模块替换机制完全不同。

## 关于将来扩展

除了LoRA，我们也可以支持其他扩展，所以我计划将它们添加进来。
