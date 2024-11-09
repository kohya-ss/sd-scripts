这是一个适用于SD 1.x和2.x模型、本仓库训练的LoRA、ControlNet（仅确认v1.0版本运行），基于Diffusers的推理（图像生成）脚本。它通过命令行使用。

# 概述

* 基于Diffusers (v0.10.2)的推理（图像生成）脚本。
* 支持SD 1.x和2.x (base/v-parameterization)模型。
* 支持txt2img、img2img、inpainting。
* 支持对话模式，以及从文件加载prompt，和连续生成。
* 可以指定每行prompt生成的图像数量。
* 可以指定整体的重复次数。
* 支持`fp16`和`bf16`。
* 支持xformers，可以进行高速生成。
    * 虽然通过xformers可以进行节省内存的生成，但与Automatic 1111先生的Web UI相比，优化程度较低，因此，在生成512*512图像时，大约需要使用6GB的VRAM。
* 支持prompt扩展至225令牌，支持negative prompt，权重指定。
* 支持Diffusers的各种sampler（与Web UI相比，sampler的数量较少）。
* 支持Text Encoder的clip skip（使用倒数第n层的输出）。
* 支持VAE的额外加载。
* 支持CLIP Guided Stable Diffusion、VGG16 Guided Stable Diffusion、Highres. fix、upscale。
    * Highres. fix是独立实现的，没有完全参考Web UI的实现，因此，输出结果可能不同。
* 支持LoRA。支持应用率指定、多个LoRA同时使用、权重合并。
    * 无法分别为Text Encoder和U-Net指定不同的应用率。
* 支持Attention Couple。
* 支持ControlNet v1.0。
* 虽然无法在过程中切换模型，但可以通过组合批处理文件来应对。
* 添加了我个人需要的各种功能。

在添加功能时，并没有对所有功能进行测试，因此，可能会影响以前的功能，导致部分功能无法运行。如果遇到任何问题，请告知我。

# 基本使用方法

## 使用对话模式生成图像

请这样输入。

```batchfile
python gen_img_diffusers.py --ckpt <模型名> --outdir <图像输出位置> --xformers --fp16 --interactive
```

使用`--ckpt`选项指定模型（Stable Diffusion的checkpoint文件，或Diffusers的模型文件夹），使用`--outdir`选项指定图像输出目录。

使用`--xformers`选项指定使用xformers（如果不使用xformers，请取消该选项）。使用`--fp16`选项指定使用fp16（单精度）进行推理。在RTX 30系列GPU上，也可以使用`--bf16`选项进行bf16（bfloat16）推理。

使用`--interactive`选项指定对话模式。

如果使用Stable Diffusion 2.0（或从那里进行额外学习的模型），请添加`--v2`选项。如果使用v-parameterization模型（`768-v-ema.ckpt`及从那里进行额外学习的模型），请进一步添加`--v_parameterization`。

如果`--v2`的指定有误，将在模型加载时出现错误。如果`--v_parameterization`的指定有误，将显示棕色的图像。

当显示`Type prompt:`时，请输入prompt。

![image](https://user-images.githubusercontent.com/52813779/235343115-f3b8ac82-456d-4aab-9724-0cc73c4534aa.png)

※如果图像未显示并出现错误，可能是安装了headless（无屏幕显示功能）的OpenCV。请使用`pip install opencv-python`安装标准的OpenCV。或者使用`--no_preview`选项停止显示图像。

选择图像窗口，然后按任意键，窗口将关闭，可输入下一个prompt。在prompt中按Ctrl+Z，然后按Enter，将关闭脚本。

## 使用单一prompt批量生成图像

这样输入（实际上是在一行中输入）。

```batchfile
python gen_img_diffusers.py --ckpt <模型名> --outdir <图像输出位置> 
    --xformers --fp16 --images_per_prompt <生成数量> --prompt "<prompt>"
```

使用`--images_per_prompt`选项，指定每条prompt的生成数量。使用`--prompt`选项指定prompt。如果包含空格，请用双引号包围。

可以使用`--batch_size`选项指定批处理大小（后述）。

## 从文件读取提示并批量生成

按照以下方式输入：

```batchfile
python gen_img_diffusers.py --ckpt <模型名称> --outdir <图像输出目录> 
    --xformers --fp16 --from_file <提示文件名>
```

使用`--from_file`选项，指定包含提示的文件。请确保每个提示占用一行。可以通过指定`--images_per_prompt`选项来设置每行生成的图像数量。

## 使用否定提示和权重

在提示选项中（在提示内部像`--x`一样指定，后文会详细说明）写入`--n`，之后的部分将被视为否定提示。

同时，可以使用与AUTOMATIC1111先生的Web UI相同的方式进行权重设置，例如使用`()`、`[]`，或者`(xxx:1.3)`（此功能的实现是从Diffusers的[Long Prompt Weighting Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion)复制的）。

无论是从命令行指定提示，还是从文件读取提示，都可以使用相同的方式进行设置。

![image](https://user-images.githubusercontent.com/52813779/235343128-e79cd768-ec59-46f5-8395-fce9bdc46208.png)

# 主要选项

请从命令行指定。

## 指定模型

- `--ckpt <模型名称>`：指定模型名称。`--ckpt`选项是必需的。可以指定Stable Diffusion的checkpoint文件，或者Diffusers的模型目录，或者Hugging Face的模型ID。

- `--v2`：在使用Stable Diffusion 2.x系列模型时指定。对于1.x系列的模型，无需指定。

- `--v_parameterization`：在使用v-parameterization模型时指定（例如`768-v-ema.ckpt`及其后续训练模型，Waifu Diffusion v1.5等）。
    
    如果`--v2`的指定与实际情况不符，会在加载模型时出现错误。如果`--v_parameterization`的指定与实际情况不符，生成的图像可能会呈现为棕色。

- `--vae`：指定要使用的VAE。如果不指定，将使用模型内部的VAE。

## 图像生成与输出

- `--interactive`：以交互模式运行。输入提示后将生成图像。

- `--prompt <提示>`：指定提示。如果包含空格，请用双引号括起来。

- `--from_file <提示文件名>`：指定包含提示的文件。请确保每个提示占用一行。注意，图像尺寸和指导规模可以在提示选项中指定（后文会详细说明）。

- `--W <图像宽度>`：指定图像宽度。默认为`512`。

- `--H <图像高度>`：指定图像高度。默认为`512`。

- `--steps <步数>`：指定采样步数。默认为`50`。

- `--scale <指导规模>`：指定无条件指导规模。默认为`7.5`。

- `--sampler <采样器名称>`：指定采样器。默认为`ddim`。可以指定Diffusers提供的ddim、pndm、dpmsolver、dpmsolver+++、lms、euler、euler_a（后三个也可以用k_lms、k_euler、k_euler_a指定）。

- `--outdir <图像输出目录>`：指定图像输出位置。

- `--images_per_prompt <生成数量>`：指定每个提示生成的图像数量。默认为`1`。

- `--clip_skip <跳过数量>`：指定使用CLIP的倒数第几层。省略时使用最后一层。

- `--max_embeddings_multiples <倍数>`：指定CLIP的输入输出长度为默认值（75）的多少倍。未指定时保持为75。例如，指定3将使输入输出长度变为225。

- `--negative_scale` : 单独指定无条件指导规模。这是根据gcem156先生的[这篇文章](https://note.com/gcem156/n/ne9a53e4a6f43)实现的。

## 调整内存使用量和生成速度

- `--batch_size <批处理大小>`：指定批处理大小。默认为`1`。批处理大小越大，消耗的内存越多，但生成速度越快。

- `--vae_batch_size <VAE的批处理大小>`：指定VAE的批处理大小。默认与批处理大小相同。
    VAE可能消耗更多内存，因此，在反噪化后（步骤达到100%后）可能会出现内存不足的情况。如果出现这种情况，请减小VAE的批处理大小。

- `--xformers`：如果使用xformers，应指定此选项。

- `--fp16`：使用fp16（半精度）进行推理。如果不指定`fp16`和`bf16`，则使用fp32（单精度）进行推理。

- `--bf16`：使用bf16（bfloat16）进行推理。仅在RTX 30系列GPU上可以指定。如果在RTX 30系列以外的GPU上指定`--bf16`选项，将会出现错误。与`fp16`相比，`bf16`在推理结果变为NaN（图像全黑）的可能性似乎更低。

## 使用附加网络（如LoRA）

- `--network_module`：指定要使用的附加网络。对于LoRA，指定`--network_module networks.lora`。如果使用多个LoRA，可以指定`--network_module networks.lora networks.lora networks.lora`。

- `--network_weights`：指定要使用的附加网络的权重文件。如`--network_weights model.safetensors`。如果使用多个LoRA，可以指定`--network_weights model1.safetensors model2.safetensors model3.safetensors`。参数的数量应与通过`--network_module`指定的数量相同。

- `--network_mul`：指定要使用的附加网络的权重倍数。默认为`1`。如`--network_mul 0.8`。如果使用多个LoRA，可以指定`--network_mul 0.4 0.5 0.7`。参数的数量应与通过`--network_module`指定的数量相同。

- `--network_merge`：使用预先指定的`--network_mul`的权重，提前合并要使用的附加网络的权重。不能与`--network_pre_calc`同时使用。虽然无法使用提示选项的`--am`和Regional LoRA，但生成速度可以提高到与未使用LoRA时相当。

- `--network_pre_calc`：对于每次生成，预先计算要使用的附加网络的权重。可以使用提示选项的`--am`。虽然生成速度可以提高到与未使用LoRA时相当，但在生成前需要计算权重的时间，而且会增加一些内存使用。在使用Regional LoRA时，此选项无效。

# 主要选项示例

以下是在批处理大小为4的情况下，对同一提示批量生成64张图像的示例。

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 64 
    --prompt "beautiful flowers --n monochrome"
```

以下是根据文件中编写的提示，每条提示分别生成10张图像，批处理大小为4的示例：

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 10 
    --from_file prompts.txt
```

这是使用Textual Inversion（后述）和LoRA的示例。

```batchfile
python gen_img_diffusers.py --ckpt model.safetensors 
    --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --fp16 --sampler k_euler_a 
    --textual_inversion_embeddings goodembed.safetensors negprompt.pt 
    --network_module networks.lora networks.lora 
    --network_weights model1.safetensors model2.safetensors 
    --network_mul 0.4 0.8 
    --clip_skip 2 --max_embeddings_multiples 1 
    --batch_size 8 --images_per_prompt 1 --interactive
```

# 提示选项

在提示内部，可以使用“双破折号加一个字母n”（如`--n`）的形式指定各种选项。无论是在对话模式、命令行，还是从文件指定提示，此功能均有效。

在提示选项`--n`的前后，请插入空格。

- `--n`：指定否定提示。

- `--w`：指定图像宽度。覆盖命令行的指定。

- `--h`：指定图像高度。覆盖命令行的指定。

- `--s`：指定步数。覆盖命令行的指定。

- `--d`：指定该图像的随机数种子。如果指定了`--images_per_prompt`，请以逗号分隔的形式指定多个值，如`--d 1,2,3,4`。
    *由于各种原因，即使使用相同的随机数种子，生成的图像也可能与Web UI的不同。

- `--l`：指定指导规模。覆盖命令行的指定。

- `--t`：指定img2img（后述）的强度。覆盖命令行的指定。

- `--nl`：指定否定提示的指导规模（后述）。覆盖命令行的指定。

- `--am`：指定附加网络的权重。覆盖命令行的指定。如果使用多个附加网络，可以以逗号分隔的形式指定，如`--am 0.8,0.5,0.3`。

*指定这些选项后，可能会导致实际运行的批处理大小小于指定的批处理大小（因为当这些值不同时无法进行批量生成）。（不过不必过于担心，从文件读取提示并生成图像时，将这些值相同的提示排列在一起，效率会更高。）

示例：
```
(masterpiece, best quality), 1girl, in shirt and plated skirt, standing at street under cherry blossoms, upper body, [from below], kind smile, looking at another, [goodembed] --n realistic, real life, (negprompt), (lowres:1.1), (worst quality:1.2), (low quality:1.1), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry --w 960 --h 640 --s 28 --d 1
```

![image](https://user-images.githubusercontent.com/52813779/235343446-25654172-fff4-4aaf-977a-20d262b51676.png)

# img2img

## 选项

- `--image_path`：指定用于img2img的图像。如`--image_path template.png`。如果指定目录，将依次使用该目录中的图像。

- `--strength`：指定img2img的强度。如`--strength 0.8`。默认为`0.8`。

- `--sequential_file_name`：指定是否将文件名设置为连续编号。如果指定，生成的文件名将从`im_000001.png`开始连续编号。

- `--use_original_file_name`：如果指定，生成文件名将与原始文件名相同。

## 从命令行执行示例

```batchfile
python gen_img_diffusers.py --ckpt trinart_characters_it4_v1_vae_merged.ckpt 
    --outdir outputs --xformers --fp16 --scale 12.5 --sampler k_euler --steps 32 
    --image_path template.png --strength 0.8 
    --prompt "1girl, cowboy shot, brown hair, pony tail, brown eyes, 
          sailor school uniform, outdoors 
          --n lowres, bad anatomy, bad hands, error, missing fingers, cropped, 
          worst quality, low quality, normal quality, jpeg artifacts, (blurry), 
          hair ornament, glasses" 
    --batch_size 8 --images_per_prompt 32
```

如果在`--image_path`选项中指定目录，将依次读取该目录中的图像。生成的图像数量不是基于图像数量，而是基于提示数量，因此，请使用`--images_per_prompt`选项来匹配用于img2img的图像数量和提示数量。

文件将按文件名排序读取。请注意，排序是按字符串顺序进行的（不是`1.jpg→2.jpg→10.jpg`，而是`1.jpg→10.jpg→2.jpg`），因此，请通过在前面填充0等方式进行调整（`01.jpg→02.jpg→10.jpg`）。

## 使用img2img进行上采样

在img2img过程中，如果通过命令行选项`--W`和`--H`指定生成图像的大小，将在进行img2img之前将原始图像调整为此大小。

另外，如果img2img的源图像是由本脚本生成的，如果省略提示，将从源图像的元数据中获取提示并直接使用。这样，可以仅执行Highres. fix的第二阶段操作。

## 在img2img时进行inpainting

可以指定图像和掩模图像进行inpainting（请注意，这并不支持inpainting模型，只是针对掩模区域进行img2img操作）。

选项如下：

- `--mask_image`：指定掩模图像。与`--img_path`相同，如果指定目录，将依次使用该目录中的图像。

掩模图像是灰度图像，其中白色部分将进行inpainting。如果在边界处使用渐变，可以使结果看起来更平滑，因此推荐这样做。

![image](https://user-images.githubusercontent.com/52813779/235343795-9eaa6d98-02ff-4f32-b089-80d1fc482453.png)

# 其他功能

## Textual Inversion

使用`--textual_inversion_embeddings`选项指定要使用的embeddings（可以指定多个）。在提示中使用不带扩展名的文件名，就可以使用这些embeddings（使用方式与Web UI相同）。在否定提示中也可以使用。

作为模型，可以使用本仓库训练的Textual Inversion模型，以及使用Web UI训练的Textual Inversion模型（不支持图像嵌入）。

## Extended Textual Inversion

请使用`--XTI_embeddings`选项代替`--textual_inversion_embeddings`。使用方法与`--textual_inversion_embeddings`相同。

## Highres. fix

这是类似于AUTOMATIC1111先生的Web UI中功能的一个类似功能（由于是独立实现的，也许在某些方面会有所不同）。首先生成较小的图像，然后基于该图像进行img2img操作，从而在防止图像整体出现失真的同时生成高分辨率的图像。

第二阶段的步数会根据`--steps`和`--strength`选项的值来计算（`steps*strength`）。

不能与img2img同时使用。

有以下选项：

- `--highres_fix_scale`：启用Highres. fix，并以倍率指定在第一阶段生成的图像大小。如果最终输出为1024x1024，而需要首先生成512x512的图像，可以指定`--highres_fix_scale 0.5`。请注意，这与在Web UI中的指定正好相反。

- `--highres_fix_steps`：指定第一阶段图像的步数。默认为`28`。

- `--highres_fix_save_1st`：指定是否保存第一阶段的图像。

- `--highres_fix_latents_upscaling`：如果指定，将在第二阶段图像生成时，基于latent方式上采样第一阶段的图像（仅支持双线性插值）。如果不指定，将以LANCZOS4方式上采样图像。

- `--highres_fix_upscaler`：在第二阶段使用任何上采样器。目前仅支持`--highres_fix_upscaler tools.latent_upscaler`。

- `--highres_fix_upscaler_args`：指定传递给通过`--highres_fix_upscaler`指定的上采样器的参数。
    对于`tools.latent_upscaler`，可以像`--highres_fix_upscaler_args "weights=D:\Work\SD\Models\others\etc\upscaler-v1-e100-220.safetensors"`那样指定权重文件。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 1024 --H 1024 --batch_size 1 --outdir ../txt2img 
    --steps 48 --sampler ddim --fp16 
    --xformers 
    --images_per_prompt 1  --interactive 
    --highres_fix_scale 0.5 --highres_fix_steps 28 --strength 0.5
```

## ControlNet

目前仅确认ControlNet 1.0运行正常。预处理仅支持Canny。

有以下选项：

- `--control_net_models`：指定ControlNet的模型文件。
    如果指定多个，将按步骤切换使用（与Web UI的ControlNet扩展的实现不同）。同时支持diff和正常模型。

- `--guide_image_path`：指定用于ControlNet的提示图像。与`--img_path`相同，如果指定目录，将依次使用该目录中的图像。对于非Canny模型，请预先进行预处理。

- `--control_net_preps`：指定ControlNet的预处理。可以像`--control_net_models`那样指定多个。目前仅支持canny。如果目标模型不使用预处理，请指定`none`。
   对于canny，可以像`--control_net_preps canny_63_191`那样，用'_'分隔阈值1和2进行指定。

- `--control_net_weights`：指定应用ControlNet时的权重（`1.0`为正常，`0.5`则以一半的影响度应用）。可以像`--control_net_models`那样指定多个。

- `--control_net_ratios`：指定应用ControlNet的步骤范围。如果指定`0.5`，则只在步骤数的一半内应用ControlNet。可以像`--control_net_models`那样指定多个。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py --ckpt model_ckpt --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --bf16 --sampler k_euler_a 
    --control_net_models diff_control_sd15_canny.safetensors --control_net_weights 1.0 
    --guide_image_path guide.png --control_net_ratios 1.0 --interactive
```

## Attention Couple + Regional LoRA

这是一个功能，可以将提示分割成几个部分，并指定每个提示应应用于图像的哪个区域。没有单独的选项，而是通过`mask_path`和提示进行指定。

首先，使用` AND `在提示中定义多个部分。可以为前三个部分指定区域，后续部分将应用于整个图像。否定提示将应用于整个图像。

以下示例中，我们使用` AND `定义了三个部分：

```
shs 2girls, looking at viewer, smile AND bsb 2girls, looking back AND 2girls --n bad quality, worst quality
```

然后，准备掩模图像。掩模图像是彩色图像，其中RGB的每个通道对应于由提示中的`AND`分隔的部分。此外，如果某个通道的值全为0，则将应用于整个图像。

在上面的例子中，R通道对应于`shs 2girls, looking at viewer, smile`，G通道对应于`bsb 2girls, looking back`，B通道对应于`2girls`。如果使用以下掩模图像，由于B通道没有指定，`2girls`将应用于整个图像。

![image](https://user-images.githubusercontent.com/52813779/235343061-b4dc9392-3dae-4831-8347-1e9ae5054251.png)

掩模图像是通过`--mask_path`指定的。目前仅支持一张图像。它将自动调整为指定的图像大小并应用。

可以与ControlNet结合使用（对于精细的位置指定，推荐与ControlNet结合使用）。

如果指定了LoRA，那么`--network_weights`中指定的多个LoRA将分别对应于`AND`的每个部分。目前的约束是，LoRA的数量必须与`AND`部分的数量相同。

## CLIP Guided Stable Diffusion

这是从Diffusers的Community Examples的[此自定义pipeline](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#clip-guided-stable-diffusion)复制并修改的代码。

除了常规提示生成的指定外，还会通过一个更大的CLIP获取提示文本的特征，然后在生成过程中控制生成的图像，使其特征接近于该文本的特征（这是我粗略的理解）。由于使用较大的CLIP，VRAM使用量将大幅增加（在8GB VRAM上，即使512*512也可能很紧张），并且生成时间也会增加。

可选择的采样器仅限于DDIM、PNDM和LMS。

通过`--clip_guidance_scale`选项，以数值指定要多大程度上反映CLIP的特征。在前面的示例中，设置为100，因此，可以从这个值开始，逐渐增加或减少。

默认情况下，将提示的前75个标记（不包括权重的特殊字符）传递给CLIP。通过提示的`--c`选项，可以单独指定传递给CLIP的文本，而不是常规提示（例如，CLIP可能无法识别DreamBooth的标识符或"1girl"等模型特有的单词，因此，省略这些单词的文本可能更合适）。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt v1-5-pruned-emaonly.ckpt --n_iter 1 
    --scale 2.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img --steps 36  
    --sampler ddim --fp16 --opt_channels_last --xformers --images_per_prompt 1  
    --interactive --clip_guidance_scale 100
```

## CLIP Image Guided Stable Diffusion

此功能不是将图像传递给CLIP，而是通过指定的图像的特征控制生成，使其接近该特征。请在`--clip_image_guidance_scale`选项中指定应用量的数值，在`--guide_image_path`选项中指定用于引导的图像（文件或目录）。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img 
    --steps 80 --sampler ddim --fp16 --opt_channels_last --xformers 
    --images_per_prompt 1  --interactive  --clip_image_guidance_scale 100 
    --guide_image_path YUKA160113420I9A4104_TP_V.jpg
```

### VGG16 Guided Stable Diffusion

这是一个功能，可以根据指定的图像生成相似的图像。除了常规提示生成的指定外，还会额外获取VGG16的特征，然后在生成过程中控制生成的图像，使其接近指定的引导图像。推荐在img2img中使用（在常规生成中，图像可能会显得模糊）。这是借鉴CLIP Guided Stable Diffusion机制的自定义功能。同时，灵感来源于使用VGG进行的风格转换。

可选择的采样器仅限于DDIM、PNDM和LMS。

通过`--vgg16_guidance_scale`选项，以数值指定要多大程度上反映VGG16的特征。根据我的经验，从100左右开始增减可能比较好。请通过`--guide_image_path`选项指定用于引导的图像（文件或目录）。

如果要批量转换多张图像的img2img，并且将源图像作为引导图像，只需为`--guide_image_path`和`--image_path`指定相同的值即可。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py --ckpt wd-v1-3-full-pruned-half.ckpt 
    --n_iter 1 --scale 5.5 --steps 60 --outdir ../txt2img 
    --xformers --sampler ddim --fp16 --W 512 --H 704 
    --batch_size 1 --images_per_prompt 1 
    --prompt "picturesque, 1girl, solo, anime face, skirt, beautiful face 
        --n lowres, bad anatomy, bad hands, error, missing fingers, 
        cropped, worst quality, low quality, normal quality, 
        jpeg artifacts, blurry, 3d, bad face, monochrome --d 1" 
    --strength 0.8 --image_path ..\src_image
    --vgg16_guidance_scale 100 --guide_image_path ..\src_image 
```

可以使用`--vgg16_guidance_layer`指定用于特征获取的VGG16层编号（默认为20，即conv4-2的ReLU）。据说，较高的层更倾向于表现画风，而较低的层更倾向于表现内容。

![image](https://user-images.githubusercontent.com/52813779/235343813-3c1f0d7a-4fb3-4274-98e4-b92d76b551df.png)

# 其他选项

- `--no_preview` : 在对话模式下不显示预览图像。如果未安装OpenCV，或者你直接检查输出文件时，请指定此选项。

- `--n_iter` : 指定重复生成的次数。默认为1。在从文件读取提示时，如果想要进行多次生成，请指定此选项。

- `--tokenizer_cache_dir` : 指定令牌化器的缓存目录。（正在开发中）

- `--seed` : 指定随机数种子。在生成单张图像时，这是该图像的种子；在生成多张图像时，这是用于生成每张图像种子的随机数的种子（当使用`--from_file`生成多张图像，并且指定`--seed`选项时，即使多次执行，每张图像也会具有相同的种子）。

- `--iter_same_seed` : 当提示中没有指定随机数种子时，在`--n_iter`的重复过程中使用相同的种子。在使用`--from_file`并想要统一多个提示之间的种子以便进行比较时，可以使用此选项。

- `--diffusers_xformers` : 使用Diffuser的xformers。

- `--opt_channels_last` : 在推理时将张量的通道放在最后。在某些情况下，这可能会加快速度。

- `--network_show_meta` : 显示附加网络的元数据。


---

# 关于Gradual Latent

Gradual Latent是一种Hires修正方法，它逐渐增大latent的尺寸。`gen_img.py`、`sdxl_gen_img.py`和`gen_img_diffusers.py`添加了以下选项。

- `--gradual_latent_timesteps`：指定开始增大latent尺寸的时刻。默认为None，意味着不使用Gradual Latent。最初可以尝试大约750。
- `--gradual_latent_ratio`：指定latent的初始尺寸。默认为0.5，意味着从默认latent尺寸的一半开始。
- `--gradual_latent_ratio_step`：指定增大latent尺寸的比例。默认为0.125，意味着latent尺寸将逐渐增加到0.625、0.75、0.875、1.0。
- `--gradual_latent_ratio_every_n_steps`：指定增大latent尺寸的间隔。默认为3，意味着每3步增大latent尺寸。

每个选项也可以通过提示选项指定，即`--glt`、`--glr`、`--gls`、`--gle`。

__请将`sampler`指定为`euler_a`。__ 因为对`sampler`的源代码进行了修改。它无法与其他`sampler`一起工作。

在SD 1.5中，其效果更为显著。在SDXL中，其效果相当微妙。

