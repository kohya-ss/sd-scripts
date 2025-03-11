#!/bin/bash

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0

# 创建输出目录
OUTPUT_DIR="/root/autodl-tmp/output"
mkdir -p $OUTPUT_DIR

# 模型路径设置 - 请根据实际路径修改
MODEL_PATH="/root/autodl-tmp/models/"
# FLUX_MODEL="$MODEL_PATH/Fluxgym-modles/flux1-dev.sft"  # FLUX.1模型路径
FLUX_MODEL="$MODEL_PATH/majicflus_v1/majicflus_v134.safetensors"  # 麦橘超然 MajicFlus 模型路径
CLIP_L_MODEL="$MODEL_PATH/Fluxgym-modles/clip_l.safetensors"   # CLIP-L模型路径
T5XXL_MODEL="$MODEL_PATH/Fluxgym-modles/t5xxl_fp16.safetensors" # T5XXL模型路径
VAE_MODEL="$MODEL_PATH/Fluxgym-modles/ae.sft"           # VAE模型路径
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
