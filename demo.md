## 镜像说明

1. 代码路径: `/root/sd-scripts` (已经切到 `sd` 分支最新代码,支持 `flux` 和 `sd3` 的模型训练, **日期: 2025.03.11**)
2. 运行脚本依赖已经安装
   1. 包含 `clip` 使用的词表文件，路径存放在 `~/.cache/huggingface/hub/models--google--t5-v1_1-xxl/` 和 `~/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/`
   2. `sd-scripts` 运行时所需要的 pip 依赖
   3. 已经装了 `tmux` 软件（如果不熟悉该软件，可以忽略）
3. flux.1 相关权重模型需要执行下面的脚本进行下载,下载路径存放在 `/root/autodl-tmp/models/Fluxgym-modles`

## 学术资源加速

> 声明：限于学术使用github和huggingface网络速度慢的问题，以下为方便用户学术用途使用相关资源提供的加速代理，不承诺稳定性保证。此外如遭遇恶意攻击等，将随时停止该加速服务

* 在终端执行以下命令：
  ```shell
  # 终端执行
  source /etc/network_turbo
  ```

## 下载 flux.1 模型

```
# 终端执行
cg down zealman/Fluxgym-modles/ae.sft -t /root/autodl-tmp/models/
cg down zealman/Fluxgym-modles/t5xxl_fp16.safetensors -t /root/autodl-tmp/models/
cg down zealman/Fluxgym-modles/clip_l.safetensors -t /root/autodl-tmp/models/
cg down zealman/Fluxgym-modles/flux1-dev.sft -t /root/autodl-tmp/models/
```

## 准数据配置文件

* 已经准备好了模板配置文件 `/root/sd-scripts/flux_data_config.toml`，可以直接修改。详情说明可以看官方说明：https://github.com/kohya-ss/sd-scripts/blob/sd3/docs/train_README-zh.md

```shell
> cat /root/sd-scripts/flux_data_config.toml

[general]
enable_bucket    = true     # 是否使用Aspect Ratio Bucketing

# ARB相关参数
bucket_no_upscale = false   # 设为true时，不会将图像放大到超过原始尺寸
bucket_reso_steps = 64      # 分桶的分辨率增量步长，默认64像素
min_bucket_reso  = 320      # 最小bucket分辨率
max_bucket_reso  = 1280     # 最大bucket分辨率

[[datasets]]
resolution = 1024           # 训练分辨率
batch_size = 2              # 批次大小

  [[datasets.subsets]]
  image_dir         = '/root/autodl-tmp/data/three_people_group_photo'  # 指定包含训练图像的文件夹
  caption_extension = '.txt'  # 若使用txt文件,更改此项
  num_repeats       = 10      # 训练图像的重复次数
```



## 准备训练脚本

* 已经准备好了模板脚本 `/root/sd-scripts/run_flux_lora.sh`, 可以直接修改。详情说明可以看官方说明：https://github.com/kohya-ss/sd-scripts/blob/sd3/README.md

```shell
> cat /root/sd-scripts/run_flux_lora.sh

#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/output"
mkdir -p $OUTPUT_DIR

# 模型路径设置 - 请根据实际路径修改
MODEL_PATH="/root/autodl-tmp/models/Fluxgym-modles/"
FLUX_MODEL="$MODEL_PATH/flux1-dev.sft"  # FLUX.1模型路径
CLIP_L_MODEL="$MODEL_PATH/clip_l.safetensors"   # CLIP-L模型路径
T5XXL_MODEL="$MODEL_PATH/t5xxl_fp16.safetensors" # T5XXL模型路径
VAE_MODEL="$MODEL_PATH/ae.sft"           # VAE模型路径
CONFIG_FILE="flux_data_config.toml"         # 数据配置文件

# 训练参数
NETWORK_DIM=16                    # LoRA维度
LEARNING_RATE=1e-4                # 学习率
MAX_EPOCHS=5                      # 最大训练轮数
SAVE_EVERY=1                      # 每多少轮保存一次
MODEL_NAME="three-people-group-photo-001"     # 输出模型名称

# 运行训练命令
accelerate launch --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py \
  --pretrained_model_name_or_path $FLUX_MODEL \
  --clip_l $CLIP_L_MODEL \
  --t5xxl $T5XXL_MODEL \
  --ae $VAE_MODEL \
  --cache_latents_to_disk \
  --save_model_as safetensors \
  --sdpa \
  --persistent_data_loader_workers \
  --max_data_loader_n_workers 2 \
  --seed 42 \
  --gradient_checkpointing \
  --mixed_precision bf16 \
  --save_precision bf16 \
  --network_module networks.lora_flux \
  --network_dim $NETWORK_DIM \
  --network_train_unet_only \
  --optimizer_type adamw8bit \
  --learning_rate $LEARNING_RATE \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk \
  --fp8_base \
  --highvram \
  --max_train_epochs $MAX_EPOCHS \
  --save_every_n_epochs $SAVE_EVERY \
  --dataset_config $CONFIG_FILE \
  --output_dir $OUTPUT_DIR \
  --output_name $MODEL_NAME \
  --timestep_sampling shift \
  --discrete_flow_shift 3.1582 \
  --model_prediction_type raw \
  --guidance_scale 1.0

echo "训练完成! 模型保存在: $OUTPUT_DIR"
```

## 执行训练

* 打开终端最好使用 `tmux` 运行以下命令
  ```bash
  cd /root/sd-scripts
  ./run_flux_lora.sh
  ```
* 运行截图
  ![sdscripts.png](https://codewithgpu-image-1310972338.cos.ap-beijing.myqcloud.com/6683-825545018-n5Jjtb1HXqJpLOW9GTop.png)



