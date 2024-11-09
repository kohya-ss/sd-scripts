# 使用WD14Tagger进行标签化

以下信息参考自这个github页面（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger ）。

建议使用onnx进行推理。请使用以下命令安装onnx。

```powershell
pip install onnx==1.15.0 onnxruntime-gpu==1.17.1
```

模型权重会自动从Hugging Face下载。

# 使用方法

运行脚本来执行标签化。
```
python fintune/tag_images_by_wd14_tagger.py --onnx --repo_id <模型的repo id> --batch_size <批量大小> <教师数据文件夹>
```

如果使用仓库中的 `SmilingWolf/wd-swinv2-tagger-v3` ，批量大小设置为4，教师数据放置在父目录的 `train_data` 中，将如下所示。

```
python tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 --batch_size 4 ..\train_data
```

首次启动时，模型文件将自动下载到 `wd14_tagger_model` 文件夹中（文件夹可通过选项更改）。

标签文件将在与教师数据图像相同的目录中创建，文件名相同，扩展名为.txt。

![生成的标签文件](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![标签与图像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

## 示例描述

如果你要以Animagine XL 3.1的方式输出，会如下所示（实际上，请在一行中输入）。

```
python tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 
    --batch_size 4  --remove_underscore --undesired_tags "PUT,YOUR,UNDESIRED,TAGS" --recursive 
    --use_rating_tags_as_last_tag --character_tags_first --character_tag_expand 
    --always_first_tags "1girl,1boy"  ..\train_data
```

## 可用的仓库ID

可以使用 [SmilingWolf 先生的V2、V3模型](https://huggingface.co/SmilingWolf)。请指定如 `SmilingWolf/wd-vit-tagger-v3`。默认情况下，如果省略，会使用 `SmilingWolf/wd-v1-4-convnext-tagger-v2`。

# 选项

## 通用选项

- `--onnx` : 使用ONNX进行推理。如果不指定，将使用TensorFlow。如果选择使用TensorFlow，需要额外安装TensorFlow。
- `--batch_size` : 每次处理的图像数量。默认值是1。请根据VRAM的容量进行调整。
- `--caption_extension` : 配文文件的扩展名。默认是 `.txt`。
- `--max_data_loader_n_workers` : DataLoader的最大工作线程数。如果指定一个大于1的数值，将使用DataLoader来加速图像加载。如果不指定，则不使用DataLoader。
- `--thresh` : 输出标签的置信度阈值。默认是0.35。降低该值将添加更多标签，但精度会降低。
- `--general_threshold` : 通用标签的置信度阈值。如果不指定，将与 `--thresh` 相同。
- `--character_threshold` : 角色标签的置信度阈值。如果不指定，将与 `--thresh` 相同。
- `--recursive` : 如果指定，将递归处理指定文件夹内的子文件夹。
- `--append_tags` : 将标签追加到现有的标签文件中。
- `--frequency_tags` : 输出标签的频率。
- `--debug` : 调试模式。如果指定，会输出调试信息。

## 下载模型

- `--model_dir` : 模型文件的保存目录。默认是 `wd14_tagger_model`。
- `--force_download` : 如果指定，会重新下载模型文件。

## 编辑标签相关

- `--remove_underscore` : 从输出的标签中删除下划线。
- `--undesired_tags` : 指定不输出的标签。可以使用逗号分隔指定多个，例如你可能指定 `black eyes,black hair`。
- `--use_rating_tags` : 在标签的开头输出评分标签。
- `--use_rating_tags_as_last_tag` : 在标签的末尾添加评分标签。
- `--character_tags_first` : 首先输出人物标签。
- `--character_tag_expand` : 展开人物标签的系列名称。例如，将 `chara_name_(series)` 标签拆分为 `chara_name, series`。
- `--always_first_tags` : 当某个标签出现在图像上时，指定该标签始终在首位输出的标签。可以使用逗号分隔指定多个，例如 `1girl,1boy`。
- `--caption_separator` : 在输出的文件中使用这个字符串分割标签。默认是 `, `。
- `--tag_replacement` : 进行标签替换。例如，你可以指定 `tag1,tag2;tag3,tag4`。如果使用 `,` 和 `;` ，请用 `\` 进行转义。
    比如 `aira tsubase,aira tsubase (uniform)` （当你想要训练特定的服装时），或者 `aira tsubase,aira tsubase\, heir of shadows` （当系列名不包含在标签中时）。

`tag_replacement` 会在 `character_tag_expand` 之后应用。

如果指定了 `remove_underscore`，请在 `undesired_tags`，`always_first_tags` 和 `tag_replacement` 中不要包含下划线。

如果指定了 `caption_separator`，请用 `caption_separator` 分割 `undesired_tags` 和 `always_first_tags`。`tag_replacement` 必须始终用 `,` 分割。

