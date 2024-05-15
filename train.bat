call venv/Scripts/activate

@REM 小資料集測試
@REM set CUDA_LAUNCH_BLOCKING=1

accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no --num_cpu_threads_per_process=1 "./sdxl_train.py" ^
    --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\recommend\sd_xl_base_1.0.safetensors" ^
    --train_data_dir="D:\AICGCode\checkpoint_train\dataset\dataset_train_tiny_symbol" ^
    --output_dir="D:/AIGC/model/slot_checkpoints" ^
    --logging_dir="D:/AIGC/model/slot_checkpoints" ^
    --sample_prompts="D:/AIGC/model/slot_checkpoints/data/sample_prompts_tiny_symbol.json" --sample_sampler="dpm_2" ^
    --output_name="slot_checkpoints" ^
    --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs ^
    --optimizer_type="Lion8bit" --max_data_loader_n_workers="0" --gradient_checkpointing --xformers ^
    --log_prefix=slot_checkpoints ^
    --save_model_as=safetensors ^
    --mixed_precision="bf16" --save_precision="bf16" --no_half_vae --full_bf16 --caption_extension=".txt" --cache_latents ^
    --train_batch_size=2 --accumulation_n_steps=4 ^
    --max_train_steps="22400" --sample_every_n_steps="200" --save_every_n_steps="4000000" ^
    --resolution="1024,1024" ^
    --learning_rate="1e-5" --min_timestep=100 --log_with="wandb"
    
@REM  --sample_every_n_steps="200"

@REM accelerate launch --num_cpu_threads_per_process=2 "./sdxl_train.py" ^
@REM     --pretrained_model_name_or_path="C:\ComfyUIModel\models\checkpoints\recommend\sd_xl_base_1.0.safetensors" ^
@REM     --train_data_dir="D:\AICGCode\checkpoint_train\dataset\dataset_train" ^
@REM     --output_dir="D:/AIGC/model/slot_checkpoints" ^
@REM     --logging_dir="D:/AIGC/model/slot_checkpoints" ^
@REM     --sample_prompts="D:/AIGC/model/slot_checkpoints/data/sample_prompts.json" --sample_sampler="dpm_2" ^
@REM     --output_name="slot_checkpoints" ^
@REM     --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs ^
@REM     --optimizer_type="AdamW8bit" --max_data_loader_n_workers="1" --persistent_data_loader_workers --gradient_checkpointing --xformers ^
@REM     --log_prefix=slot_checkpoints ^
@REM     --save_model_as=safetensors ^
@REM     --mixed_precision="bf16" --save_precision="bf16" --full_bf16 --caption_extension=".txt" --cache_latents ^
@REM     --train_batch_size=2 --accumulation_n_steps=4 ^
@REM     --max_train_steps="219300" --sample_every_n_steps="2193" --save_every_n_steps="21930"^
@REM     --resolution="1024,1024" ^
@REM     --learning_rate="1e-5" --min_timestep=200 --log_with="wandb" 





@REM 接續訓練
@REM accelerate launch --num_processes=1 --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no --num_cpu_threads_per_process=1 "./sdxl_train.py" ^
@REM     --train_data_dir="D:\AICGCode\checkpoint_train\dataset\dataset_train_v1" ^
@REM     --output_dir="D:/AIGC/model/slot_checkpoints" ^
@REM     --logging_dir="D:/AIGC/model/slot_checkpoints" ^
@REM     --sample_prompts="D:/AIGC/model/slot_checkpoints/data/sample_prompts.json" --sample_sampler="dpm_2" ^
@REM     --output_name="slot_checkpoints" ^
@REM     --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs ^
@REM     --optimizer_type="PagedAdamW8bit" --max_data_loader_n_workers="1" --persistent_data_loader_workers --gradient_checkpointing --xformers ^
@REM     --log_prefix=slot_checkpoints ^
@REM     --save_model_as=safetensors ^
@REM     --mixed_precision="bf16" --save_precision="bf16"  --no_half_vae --full_bf16 --caption_extension=".txt" --cache_latents ^
@REM     --train_batch_size=2 --accumulation_n_steps=1 ^
@REM     --max_train_steps="219300" --sample_every_n_steps="2193" --save_every_n_steps="8772"^
@REM     --resolution="1024,1024" ^
@REM     --learning_rate="1e-5" --min_timestep=200 --log_with="wandb" ^
@REM     --pretrained_model_name_or_path="D:/AIGC/model/slot_checkpoints/slot_checkpoints-step00052632.safetensors" ^
@REM     --global_step=52633
    
    
@REM --log_with="wandb" 
    
    

@REM  --lowram
@REM --accumulation_n_steps=4


@REM gradient_checkpointing
@REM --cache_text_encoder_outputs 
@REM --max_timestep 400
