有关 [Textual Inversion](https://textual-inversion.github.io/) 的学习说明。

请同时查看[共同文档](./train_README-zh.md)。

在实施过程中，我大量参考了 [https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion](https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion) 。

训练后的模型也可以直接在 Web UI 中使用。

# 学习步骤

请先参考此存储库的 README，进行环境设置。

## 准备数据

请参阅[准备训练数据](./train_README-zh.md)。

## 执行训练

使用 `train_textual_inversion.py` 进行学习。以下是命令行的示例（DreamBooth 方法）。

```
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py 
    --dataset_config=<在数据准备中创建的.toml文件> 
    --output_dir=<学习模型的输出目录>  
    --output_name=<学习模型输出时的文件名> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=1600 
    --learning_rate=1e-6 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --token_string=mychar4 --init_word=cute --num_vectors_per_token=4
```

``--token_string`` 用于指定训练时的标记字符串。在训练时，请确保提示包含此字符串（如果 token_string 是 mychar4，则例如为``mychar4 1girl``）。提示中此字符串的部分将被替换为 Textual Inversion 的新标记进行训练。对于 DreamBooth，作为类+标识符形式的数据集，将 ``token_string`` 设置为标记字符串是最简单且最可靠的方法。

通过使用 ``--debug_dataset``，您可以查看提示中的标记字符串是否被替换为后续的token id，例如，您可以通过检查是否存在从 ``49408`` 开始的token来确认提示中是否包含标记字符串。

```
input ids: tensor([[49406, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```

tokenizer已经包含的单词（通常的单词）不能使用。

``--init_word`` 在初始化嵌入时，指定用于复制源标记的字符串。最好选择与要学习的概念相近的单词。不能指定两个或更多标记的字符串。

``--num_vectors_per_token`` 指定在这次学习中使用多少标记。标记越多，表达力越强，但相应地会消耗更多的标记。例如，如果 num_vectors_per_token=8，则指定的标记字符串将消耗（通常限制为77个标记的）8个标记。

以上是用于Textual Inversion的主要选项。之后的步骤与其他学习脚本相似。

通常建议将 `num_cpu_threads_per_process` 设置为1。

`pretrained_model_name_or_path` 指定要进行追加训练的原始模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers的本地模型目录，或Diffusers的模型ID（例如"stabilityai/stable-diffusion-2"等）

在 `output_dir`  中指定保存经过学习后的模型的文件夹。在 `output_name` 中指定模型文件的名称，不包括扩展名。使用 `save_model_as` 选项指定以safetensors格式保存。

在 `dataset_config` 中指定  `.toml`文件。最初，为了减少内存消耗，将文件内的批处理大小设置为 `1`。

设置训练步数 `max_train_steps` 为10000。学习率 `learning_rate` 在这里设置为5e-6。

为了节省内存，指定 `mixed_precision="fp16"`（在 RTX30 系列及更高版本中，也可以指定为 `bf16`。请与环境设置时的 accelerate 设置保持一致）。同时指定 `gradient_checkpointing`。

为了使用内存消耗较少的 8 位 AdamW 优化器（将模型优化为与学习数据匹配），指定 `optimizer_type="AdamW8bit"`。

指定 `xformers` 选项，并使用 xformers 的 CrossAttention。如果未安装 xformers 或发生错误（取决于环境，例如 `mixed_precision="no"` 的情况），则可以指定 `mem_eff_attn` 选项，这将使用内存省略版本的 CrossAttention（速度较慢）。

如果有足够的内存，请编辑 `.toml` 文件，将批处理大小增加到例如 `8` 左右（可能提高速度和精度）。

### 常用选项说明

请在以下情况下参考选项文档。

- Stable Diffusion 2.x 或从其派生的模型的训练
- 训练前提条件为 clip skip 大于2的模型
- 训练超过75个标记的描述

### 关于 Textual Inversion 的批处理大小

由于相对于整个模型的训练（如 DreamBooth 或微调），内存使用量较少，因此批处理大小可以设置得较大。

＃ Textual Inversion 的其他主要选项

有关所有选项的详细信息，请参阅其他文档。

* `--weights`
  * 在训练之前加载预训练的嵌入，并在此基础上进行微调训练。
* `--use_object_template`
  * 不使用文本描述，而是使用默认的对象模板字符串（例如``a photo of a {}``）进行训练。与官方实现相同。将忽略文本描述。
* `--use_style_template`
  * 不使用文本描述，而是使用默认的样式模板字符串（例如``a painting in the style of {}``）进行训练。与官方实现相同。将忽略文本描述

## 在此代码库中使用的图像生成脚本进行生成。

在 gen_img_diffusers.py 中，请使用 ``--textual_inversion_embeddings``  选项指定训练的embeddings文件（可以指定多个）。在提示中使用embeddings文件的文件名（不包括扩展名），以应用该embeddings。
