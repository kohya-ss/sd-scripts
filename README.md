This repository contains training, generation and utility scripts for Stable Diffusion.

## FLUX.1 training (WIP)

This feature is experimental. The options and the training script may change in the future. Please let us know if you have any idea to improve the training.

__Please update PyTorch to 2.4.0. We have tested with `torch==2.4.0` and `torchvision==0.19.0` with CUDA 12.4. We also updated `accelerate` to 0.33.0 just to be safe. `requirements.txt` is also updated, so please update the requirements.__

The command to install PyTorch is as follows:
`pip3 install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124`

### Recent Updates

Oct 12, 2024 (update 1):

- [Experimental] FLUX.1 fine-tuning and LoRA training now support "FLUX.1 __compact__" models.
  - A compact model is a model that retains the FLUX.1 architecture but reduces the number of double/single blocks from the default 19/38.
  - The model is automatically determined based on the keys in *.safetensors.
  - Specifications for compact model safetensors:
    - Please specify the block indices as consecutive numbers. An error will occur if there are missing numbers. For example, if you reduce the double blocks to 15, the maximum key will be `double_blocks.14.*`. The same applies to single blocks.
  - LoRA training is unverified.
  - The trained model can be used for inference with `flux_minimal_inference.py`. Other inference environments are unverified.

Oct 12, 2024:

- Multi-GPU training now works on Windows. Thanks to Akegarasu for PR [#1686](https://github.com/kohya-ss/sd-scripts/pull/1686)!
  - In simple tests, SDXL and FLUX.1 LoRA training worked. FLUX.1 fine-tuning did not work, probably due to a PyTorch-related error. Other scripts are unverified.
  - Set up multi-GPU training with `accelerate config`.
  - Specify `--rdzv_backend=c10d` when launching `accelerate launch`. You can also edit `config.yaml` directly.
    ```
    accelerate launch --rdzv_backend=c10d sdxl_train_network.py ...
    ```
  - In multi-GPU training, the memory of multiple GPUs is not integrated. In other words, even if you have two 12GB VRAM GPUs, you cannot train the model that requires 24GB VRAM. Training that can be done with 12GB VRAM is executed at (up to) twice the speed.

Oct 11, 2024:
- ControlNet training for SDXL has been implemented in this branch. Please use `sdxl_train_control_net.py`. 
  - For details on defining the dataset, see [here](docs/train_lllite_README.md#creating-a-dataset-configuration-file).
  - The learning rate for the copy part of the U-Net is specified by `--learning_rate`. The learning rate for the added modules in ControlNet is specified by `--control_net_lr`. The optimal value is still unknown, but try around U-Net `1e-5` and ControlNet `1e-4`.
  - If you want to generate sample images, specify the control image as `--cn path/to/control/image`.
  - The trained weights are automatically converted and saved in Diffusers format. It should be available in ComfyUI.
- Weighting of prompts (captions) during training in SDXL is now supported (e.g., `(some text)`, `[some text]`, `(some text:1.4)`, etc.). The function is enabled by specifying `--weighted_captions`. 
  - The default is `False`. It is same as before, and the parentheses are used as normal text.
  - If `--weighted_captions` is specified, please use `\` to escape the parentheses in the prompt. For example, `\(some text:1.4\)`.

Oct 6, 2024:
- In FLUX.1 LoRA training and fine-tuning, the specified weight file (*.safetensors) is automatically determined to be dev or schnell. This allows schnell models to be loaded correctly. Note that LoRA training with schnell models and fine-tuning with schnell models are unverified.
- FLUX.1 LoRA training and fine-tuning can now load weights in Diffusers format in addition to BFL format (a single *.safetensors file). Please specify the parent directory of `transformer` or `diffusion_pytorch_model-00001-of-00003.safetensors` with the full path. However, Diffusers format CLIP/T5XXL is not supported. Saving is supported only in BFL format.

Sep 26, 2024:
The implementation of block swap during FLUX.1 fine-tuning has been changed to improve speed about 10% (depends on the environment). A new `--blocks_to_swap` option has been added, and `--double_blocks_to_swap` and `--single_blocks_to_swap` are deprecated. `--double_blocks_to_swap` and `--single_blocks_to_swap` are working as before, but they will be removed in the future. See [FLUX.1 fine-tuning](#flux1-fine-tuning) for details.


Sep 18, 2024 (update 1):
Fixed an issue where train()/eval() was not called properly with the schedule-free optimizer. The schedule-free optimizer can be used in FLUX.1 LoRA training and fine-tuning for now.

Sep 18, 2024:

- Schedule-free optimizer is added. Thanks to sdbds! See PR [#1600](https://github.com/kohya-ss/sd-scripts/pull/1600) for details.
  - Details of the schedule-free optimizer can be found in [facebookresearch/schedule_free](https://github.com/facebookresearch/schedule_free).
  - `schedulefree` is added to the dependencies. Please update the library if necessary.
  - AdamWScheduleFree or SGDScheduleFree can be used. Specify `adamwschedulefree` or `sgdschedulefree` in `--optimizer_type`.
  - Wrapper classes are not available for now.
  - These can be used not only for FLUX.1 training but also for other training scripts after merging to the dev/main branch.

Sep 16, 2024:

 Added `train_double_block_indices` and `train_double_block_indices` to the LoRA training script to specify the indices of the blocks to train. See [Specify blocks to train in FLUX.1 LoRA training](#specify-blocks-to-train-in-flux1-lora-training) for details.

Sep 15, 2024:

Added a script `convert_diffusers_to_flux.py` to convert Diffusers format FLUX.1 models (checkpoints) to BFL format. See `--help` for usage. Only Flux models are supported. AE/CLIP/T5XXL are not supported. 

The implementation is based on 2kpr's code. Thanks to 2kpr!

Sep 14, 2024:
- You can now specify the rank for each layer in FLUX.1. See [Specify rank for each layer in FLUX.1](#specify-rank-for-each-layer-in-flux1) for details.
- OFT is now supported with FLUX.1. See [FLUX.1 OFT training](#flux1-oft-training) for details.

Sep 11, 2024: 
Logging to wandb is improved. See PR [#1576](https://github.com/kohya-ss/sd-scripts/pull/1576) for details. Thanks to p1atdev!

Sep 10, 2024:
In FLUX.1 LoRA training, individual learning rates can be specified for CLIP-L and T5XXL. By specifying multiple numbers in `--text_encoder_lr`, you can set the learning rates for CLIP-L and T5XXL separately. Specify like `--text_encoder_lr 1e-4 1e-5`. The first value is the learning rate for CLIP-L, and the second value is for T5XXL. If you specify only one, the learning rates for CLIP-L and T5XXL will be the same.

Sep 9, 2024:
Added `--negative_prompt` and `--cfg_scale` to `flux_minimal_inference.py`. Negative prompts can be used. 

Sep 5, 2024 (update 1):

Added `--cpu_offload_checkpointing` option to LoRA training script. Offloads gradient checkpointing to CPU. This reduces up to 1GB of VRAM usage but slows down the training by about 15%. Cannot be used with `--split_mode`.

Sep 5, 2024:

The LoRA merge script now supports CLIP-L and T5XXL LoRA. Please specify `--clip_l` and `--t5xxl`. `--clip_l_save_to` and `--t5xxl_save_to` specify the save destination for CLIP-L and T5XXL. See [Merge LoRA to FLUX.1 checkpoint](#merge-lora-to-flux1-checkpoint) for details.

Sep 4, 2024:
- T5XXL LoRA is supported in LoRA training. Remove `--network_train_unet_only` and add `train_t5xxl=True` to `--network_args`. CLIP-L is also trained at the same time (T5XXL only cannot be trained). The trained model can be used with ComfyUI. See [Key Features for FLUX.1 LoRA training](#key-features-for-flux1-lora-training) for details.
- In LoRA training, when `--fp8_base` is specified, you can specify `t5xxl_fp8_e4m3fn.safetensors` as the T5XXL weights. However, it is recommended to use fp16 weights for caching.
- Fixed an issue where the training CLIP-L LoRA was not used in sample image generation during LoRA training.

Sep 1, 2024:
- `--timestamp_sampling` has `flux_shift` option. Thanks to sdbds!
  - This is the same shift as FLUX.1 dev inference, adjusting the timestep sampling depending on the resolution. `--discrete_flow_shift` is ignored when `flux_shift` is specified. It is not verified which is better, `shift` or `flux_shift`.

Aug 29, 2024: 
Please update `safetensors` to `0.4.4` to fix the error when using `--resume`. `requirements.txt` is updated.

### Contents

- [FLUX.1 LoRA training](#flux1-lora-training)
  - [Key Options for FLUX.1 LoRA training](#key-options-for-flux1-lora-training)
  - [Distribution of timesteps](#distribution-of-timesteps)
  - [Key Features for FLUX.1 LoRA training](#key-features-for-flux1-lora-training)
  - [Specify rank for each layer in FLUX.1](#specify-rank-for-each-layer-in-flux1)
  - [Specify blocks to train in FLUX.1 LoRA training](#specify-blocks-to-train-in-flux1-lora-training)
- [FLUX.1 OFT training](#flux1-oft-training)
- [Inference for FLUX.1 with LoRA model](#inference-for-flux1-with-lora-model)
- [FLUX.1 fine-tuning](#flux1-fine-tuning)
  - [Key Features for FLUX.1 fine-tuning](#key-features-for-flux1-fine-tuning)
- [Extract LoRA from FLUX.1 Models](#extract-lora-from-flux1-models)
- [Convert FLUX LoRA](#convert-flux-lora)
- [Merge LoRA to FLUX.1 checkpoint](#merge-lora-to-flux1-checkpoint)
- [FLUX.1 Multi-resolution training](#flux1-multi-resolution-training)
- [Convert Diffusers to FLUX.1](#convert-diffusers-to-flux1)

### FLUX.1 LoRA training

We have added a new training script for LoRA training. The script is `flux_train_network.py`. See `--help` for options. 

FLUX.1 model, CLIP-L, and T5XXL models are recommended to be in bf16/fp16 format. If you specify `--fp8_base`, you can use fp8 models for FLUX.1. The fp8 model is only compatible with `float8_e4m3fn` format.

Sample command is below. It will work with 24GB VRAM GPUs. 

```
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train_network.py 
--pretrained_model_name_or_path flux1-dev.safetensors --clip_l sd3/clip_l.safetensors --t5xxl sd3/t5xxl_fp16.safetensors 
--ae ae.safetensors --cache_latents_to_disk --save_model_as safetensors --sdpa --persistent_data_loader_workers 
--max_data_loader_n_workers 2 --seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 
--network_module networks.lora_flux --network_dim 4 --optimizer_type adamw8bit --learning_rate 1e-4 
--cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk --fp8_base 
--highvram --max_train_epochs 4 --save_every_n_epochs 1 --dataset_config dataset_1024_bs2.toml 
--output_dir path/to/output/dir --output_name flux-lora-name 
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 
```
(The command is multi-line for readability. Please combine it into one line.)

The training can be done with 16GB VRAM GPUs with Adafactor optimizer. Please use settings like below:

```
--optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
```

The training can be done with 12GB VRAM GPUs with Adafactor optimizer, `--split_mode` and `train_blocks=single` options. Please use settings like below:

```
--optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --split_mode --network_args "train_blocks=single" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
```

`--cpu_offload_checkpointing` offloads gradient checkpointing to CPU. This reduces up to 1GB of VRAM usage but slows down the training by about 15%. Cannot be used with `--split_mode`.

We also not sure how many epochs are needed for convergence, and how the learning rate should be adjusted.

The trained LoRA model can be used with ComfyUI. 

#### Key Options for FLUX.1 LoRA training

There are many unknown points in FLUX.1 training, so some settings can be specified by arguments. Here are the arguments. The arguments and sample settings are still experimental and may change in the future. Feedback on the settings is welcome.

- `--pretrained_model_name_or_path` is the path to the pretrained model (FLUX.1). bf16 (original BFL model) is recommended (`flux1-dev.safetensors` or `flux1-dev.sft`). If you specify `--fp8_base`, you can use fp8 models for FLUX.1. The fp8 model is only compatible with `float8_e4m3fn` format.
- `--clip_l` is the path to the CLIP-L model. 
- `--t5xxl` is the path to the T5XXL model. If you specify `--fp8_base`, you can use fp8 (float8_e4m3fn) models for T5XXL. However, it is recommended to use fp16 models for caching.
- `--ae` is the path to the autoencoder model (`ae.safetensors` or `ae.sft`).

- `--timestep_sampling` is the method to sample timesteps (0-1):
  - `sigma`: sigma-based, same as SD3
  - `uniform`: uniform random
  - `sigmoid`: sigmoid of random normal, same as x-flux, AI-toolkit etc.
  - `shift`: shifts the value of sigmoid of normal distribution random number
  - `flux_shift`: shifts the value of sigmoid of normal distribution random number, depending on the resolution (same as FLUX.1 dev inference). `--discrete_flow_shift` is ignored when `flux_shift` is specified.
- `--sigmoid_scale` is the scale factor for sigmoid timestep sampling (only used when timestep-sampling is "sigmoid"). The default is 1.0. Larger values will make the sampling more uniform.
  - This option is effective even when`--timestep_sampling shift` is specified.
  - Normally, leave it at 1.0. Larger values make the value before shift closer to a uniform distribution.
- `--model_prediction_type` is how to interpret and process the model prediction:
  - `raw`: use as is, same as x-flux
  - `additive`: add to noisy input
  - `sigma_scaled`: apply sigma scaling, same as SD3
- `--discrete_flow_shift` is the discrete flow shift for the Euler Discrete Scheduler, default is 3.0 (same as SD3).

The existing `--loss_type` option may be useful for FLUX.1 training. The default is `l2`.

~~In our experiments, `--timestep_sampling sigma --model_prediction_type raw --discrete_flow_shift 1.0` with `--loss_type l2` seems to work better than the default (SD3) settings. The multiplier of LoRA should be adjusted.~~

In our experiments, `--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0` (with the default `l2` loss_type) seems to work better. 

The settings in [AI Toolkit by Ostris](https://github.com/ostris/ai-toolkit) seems to be equivalent to `--timestep_sampling sigmoid --model_prediction_type raw --guidance_scale 1.0` (with the default `l2` loss_type). 

Other settings may work better, so please try different settings.

Other options are described below.

#### Distribution of timesteps

`--timestep_sampling` and `--sigmoid_scale`, `--discrete_flow_shift` adjust the distribution of timesteps. The distribution is shown in the figures below.

The effect of `--discrete_flow_shift` with `--timestep_sampling shift` (when `--sigmoid_scale` is not specified, the default is 1.0):
![Figure_2](https://github.com/user-attachments/assets/d9de42f9-f17d-40da-b88d-d964402569c6)

The difference between `--timestep_sampling sigmoid` and `--timestep_sampling uniform` (when `--timestep_sampling sigmoid` or `uniform` is specified, `--discrete_flow_shift` is ignored):
![Figure_3](https://github.com/user-attachments/assets/27029009-1f5d-4dc0-bb24-13d02ac4fdad)

The effect of `--timestep_sampling sigmoid` and `--sigmoid_scale` (when `--timestep_sampling sigmoid` is specified, `--discrete_flow_shift` is ignored):
![Figure_4](https://github.com/user-attachments/assets/08a2267c-e47e-48b7-826e-f9a080787cdc)

#### Key Features for FLUX.1 LoRA training

1. CLIP-L and T5XXL LoRA Support:
   - FLUX.1 LoRA training now supports CLIP-L and T5XXL LoRA training.
   - Remove `--network_train_unet_only` from your command.
   - Add `train_t5xxl=True` to `--network_args` to train T5XXL LoRA. CLIP-L is also trained at the same time.
   - T5XXL output can be cached for CLIP-L LoRA training. So, `--cache_text_encoder_outputs` or `--cache_text_encoder_outputs_to_disk` is also available.
   - The learning rates for CLIP-L and T5XXL can be specified separately. Multiple numbers can be specified in `--text_encoder_lr`. For example, `--text_encoder_lr 1e-4 1e-5`. The first value is the learning rate for CLIP-L, and the second value is for T5XXL. If you specify only one, the learning rates for CLIP-L and T5XXL will be the same. If `--text_encoder_lr` is not specified, the default learning rate `--learning_rate` is used for both CLIP-L and T5XXL.
   - The trained LoRA can be used with ComfyUI.
   - Note: `flux_extract_lora.py`, `convert_flux_lora.py`and `merge_flux_lora.py` do not support CLIP-L and T5XXL LoRA yet.

    | trained LoRA|option|network_args|cache_text_encoder_outputs (*1)|
    |---|---|---|---|
    |FLUX.1|`--network_train_unet_only`|-|o|
    |FLUX.1 + CLIP-L|-|-|o (*2)|
    |FLUX.1 + CLIP-L + T5XXL|-|`train_t5xxl=True`|-|
    |CLIP-L (*3)|`--network_train_text_encoder_only`|-|o (*2)|
    |CLIP-L + T5XXL (*3)|`--network_train_text_encoder_only`|`train_t5xxl=True`|-|

    - *1: `--cache_text_encoder_outputs` or `--cache_text_encoder_outputs_to_disk` is also available.
    - *2: T5XXL output can be cached for CLIP-L LoRA training.
    - *3: Not tested yet.

2. Experimental FP8/FP16 mixed training:
   - `--fp8_base_unet` enables training with fp8 for FLUX and bf16/fp16 for CLIP-L/T5XXL.
   - FLUX can be trained with fp8, and CLIP-L/T5XXL can be trained with bf16/fp16.
   - When specifying this option, the `--fp8_base` option is automatically enabled.

3. Split Q/K/V Projection Layers (Experimental):
   - Added an option to split the projection layers of q/k/v/txt in the attention and apply LoRA to each of them.
   - Specify `"split_qkv=True"` in network_args like `--network_args "split_qkv=True"` (`train_blocks` is also available).
   - May increase expressiveness but also training time.
   - The trained model is compatible with normal LoRA models in sd-scripts and can be used in environments like ComfyUI.
   - Converting to AI-toolkit (Diffusers) format with `convert_flux_lora.py` will reduce the size.
   
4. T5 Attention Mask Application:
   - T5 attention mask is applied when `--apply_t5_attn_mask` is specified.
   - Now applies mask when encoding T5 and in the attention of Double and Single Blocks
   - Affects fine-tuning, LoRA training, and inference in `flux_minimal_inference.py`.

5. Multi-resolution Training Support:
   - FLUX.1 now supports multi-resolution training, even with caching latents to disk.


Technical details of Q/K/V split: 

In the implementation of Black Forest Labs' model, the projection layers of q/k/v (and txt in single blocks) are concatenated into one. If LoRA is added there as it is, the LoRA module is only one, and the dimension is large. In contrast, in the implementation of Diffusers, the projection layers of q/k/v/txt are separated. Therefore, the LoRA module is applied to q/k/v/txt separately, and the dimension is smaller. This option is for training LoRA similar to the latter.

The compatibility of the saved model (state dict) is ensured by concatenating the weights of multiple LoRAs. However, since there are zero weights in some parts, the model size will be large.

#### Specify rank for each layer in FLUX.1

You can specify the rank for each layer in FLUX.1 by specifying the following network_args. If you specify `0`, LoRA will not be applied to that layer.

When network_args is not specified, the default value (`network_dim`) is applied, same as before.

|network_args|target layer|
|---|---|
|img_attn_dim|img_attn in DoubleStreamBlock|
|txt_attn_dim|txt_attn in DoubleStreamBlock|
|img_mlp_dim|img_mlp in DoubleStreamBlock|
|txt_mlp_dim|txt_mlp in DoubleStreamBlock|
|img_mod_dim|img_mod in DoubleStreamBlock|
|txt_mod_dim|txt_mod in DoubleStreamBlock|
|single_dim|linear1 and linear2 in SingleStreamBlock|
|single_mod_dim|modulation in SingleStreamBlock|

`"verbose=True"` is also available for debugging. It shows the rank of each layer.

example: 
```
--network_args "img_attn_dim=4" "img_mlp_dim=8" "txt_attn_dim=2" "txt_mlp_dim=2" 
"img_mod_dim=2" "txt_mod_dim=2" "single_dim=4" "single_mod_dim=2" "verbose=True"
```

You can apply LoRA to the conditioning layers of Flux by specifying `in_dims` in network_args. When specifying, be sure to specify 5 numbers in `[]` as a comma-separated list.

example: 
```
--network_args "in_dims=[4,2,2,2,4]"
```

Each number corresponds to `img_in`, `time_in`, `vector_in`, `guidance_in`, `txt_in`. The above example applies LoRA to all conditioning layers, with rank 4 for `img_in`, 2 for `time_in`, `vector_in`, `guidance_in`, and 4 for `txt_in`.

If you specify `0`, LoRA will not be applied to that layer. For example, `[4,0,0,0,4]` applies LoRA only to `img_in` and `txt_in`.

#### Specify blocks to train in FLUX.1 LoRA training

You can specify the blocks to train in FLUX.1 LoRA training by specifying `train_double_block_indices` and `train_single_block_indices` in network_args. The indices are 0-based. The default (when omitted) is to train all blocks. The indices are specified as a list of integers or a range of integers, like `0,1,5,8` or `0,1,4-5,7`. The number of double blocks is 19, and the number of single blocks is 38, so the valid range is 0-18 and 0-37, respectively. `all` is also available to train all blocks, `none` is also available to train no blocks.

example: 
```
--network_args "train_double_block_indices=0,1,8-12,18" "train_single_block_indices=3,10,20-25,37"
```

```
--network_args "train_double_block_indices=none" "train_single_block_indices=10-15"
```

If you specify one of `train_double_block_indices` or `train_single_block_indices`, the other will be trained as usual. 

### FLUX.1 OFT training

You can train OFT with almost the same options as LoRA, such as `--timestamp_sampling`. The following points are different.

- Change `--network_module` from `networks.lora_flux` to `networks.oft_flux`.
- `--network_dim` is the number of OFT blocks. Unlike LoRA rank, the smaller the dim, the larger the model. We recommend about 64 or 128. Please make the output dimension of the target layer of OFT divisible by the value of `--network_dim` (an error will occur if it is not divisible). Valid values are 64, 128, 256, 512, 1024, etc.
- `--network_alpha` is treated as a constraint for OFT. We recommend about 1e-2 to 1e-4. The default value when omitted is 1, which is too large, so be sure to specify it.
- CLIP/T5XXL is not supported. Specify `--network_train_unet_only`.
- `--network_args` specifies the hyperparameters of OFT. The following are valid:
    - Specify `enable_all_linear=True` to target all linear connections in the MLP layer. The default is False, which targets only attention.

Currently, there is no environment to infer FLUX.1 OFT. Inference is only possible with `flux_minimal_inference.py` (specify OFT model with `--lora`).

Sample command is below. It will work with 24GB VRAM GPUs with the batch size of 1.

```
--network_module networks.oft_flux  --network_dim 128 --network_alpha 1e-3 
--network_args "enable_all_linear=True" --learning_rate 1e-5 
```

The training can be done with 16GB VRAM GPUs without `--enable_all_linear` option and with Adafactor optimizer. 

### Inference for FLUX.1 with LoRA model

The inference script is also available. The script is `flux_minimal_inference.py`. See `--help` for options. 

```
python flux_minimal_inference.py --ckpt flux1-dev.safetensors --clip_l sd3/clip_l.safetensors --t5xxl sd3/t5xxl_fp16.safetensors --ae ae.safetensors --dtype bf16 --prompt "a cat holding a sign that says hello world" --out path/to/output/dir --seed 1 --flux_dtype fp8 --offload --lora lora-flux-name.safetensors;1.0
```

### FLUX.1 fine-tuning

The memory-efficient training with block swap is based on 2kpr's implementation. Thanks to 2kpr!

__`--double_blocks_to_swap` and `--single_blocks_to_swap` are deprecated. These options is still available, but they will be removed in the future. Please use `--blocks_to_swap` instead. These options are equivalent to specifying `double_blocks_to_swap + single_blocks_to_swap // 2` in `--blocks_to_swap`.__

Sample command for FLUX.1 fine-tuning is below. This will work with 24GB VRAM GPUs, and 64GB main memory is recommended. 

```
accelerate launch  --mixed_precision bf16 --num_cpu_threads_per_process 1 flux_train.py   
--pretrained_model_name_or_path flux1-dev.safetensors  --clip_l clip_l.safetensors --t5xxl t5xxl_fp16.safetensors --ae ae_dev.safetensors 
--save_model_as safetensors --sdpa --persistent_data_loader_workers --max_data_loader_n_workers 2 
--seed 42 --gradient_checkpointing --mixed_precision bf16 --save_precision bf16 
--dataset_config dataset_1024_bs1.toml  --output_dir path/to/output/dir --output_name output-name 
--learning_rate 5e-5 --max_train_epochs 4  --sdpa --highvram --cache_text_encoder_outputs_to_disk --cache_latents_to_disk --save_every_n_epochs 1 
--optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" 
--lr_scheduler constant_with_warmup --max_grad_norm 0.0 
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0 
--fused_backward_pass  --blocks_to_swap 8 --full_bf16 
```
(The command is multi-line for readability. Please combine it into one line.)

Options are almost the same as LoRA training. The difference is `--full_bf16`, `--fused_backward_pass` and  `--blocks_to_swap`. `--cpu_offload_checkpointing` is also available.

`--full_bf16` enables the training with bf16 (weights and gradients). 

`--fused_backward_pass` enables the fusing of the optimizer step into the backward pass for each parameter. This reduces the memory usage during training. Only Adafactor optimizer is supported for now. Stochastic rounding is also enabled when `--fused_backward_pass` and `--full_bf16` are specified.

`--blockwise_fused_optimizers` enables the fusing of the optimizer step into the backward pass for each block. This is similar to `--fused_backward_pass`. Any optimizer can be used, but Adafactor is recommended for memory efficiency and stochastic rounding. `--blockwise_fused_optimizers` cannot be used with `--fused_backward_pass`. Stochastic rounding is not supported for now.

`--blocks_to_swap` is the number of blocks to swap. The default is None (no swap). These options must be combined with `--fused_backward_pass` or `--blockwise_fused_optimizers`. The recommended maximum value is 36. 

`--cpu_offload_checkpointing` is to offload the gradient checkpointing to CPU. This reduces about 2GB of VRAM usage. 

All these options are experimental and may change in the future.

The increasing the number of blocks to swap may reduce the memory usage, but the training speed will be slower. `--cpu_offload_checkpointing` also slows down the training.

Swap 8 blocks without cpu offload checkpointing may be a good starting point for 24GB VRAM GPUs. Please try different settings according to VRAM usage and training speed.

The learning rate and the number of epochs are not optimized yet. Please adjust them according to the training results.

#### How to use block swap

There are two possible ways to use block swap. It is unknown which is better.

1. Swap the minimum number of blocks that fit in VRAM with batch size 1 and shorten the training speed of one step.

    The above command example is for this usage.

2. Swap many blocks to increase the batch size and shorten the training speed per data.

    For example, swapping 20 blocks seems to increase the batch size to about 6. In this case, the training speed per data will be relatively faster than 1.
  
#### Training with <24GB VRAM GPUs

Swap 28 blocks without cpu offload checkpointing may be working with 12GB VRAM GPUs. Please try different settings according to VRAM size of your GPU.

T5XXL requires about 10GB of VRAM, so 10GB of VRAM will be minimum requirement for FLUX.1 fine-tuning.

#### Key Features for FLUX.1 fine-tuning

1.  Technical details of block swap:
    - Reduce memory usage by transferring double and single blocks of FLUX.1 from GPU to CPU when they are not needed.
    - During forward pass, the weights of the blocks that have finished calculation are transferred to CPU, and the weights of the blocks to be calculated are transferred to GPU.
    - The same is true for the backward pass, but the order is reversed. The gradients remain on the GPU.
    - Since the transfer between CPU and GPU takes time, the training will be slower.
    - `--blocks_to_swap` specify the number of blocks to swap. 
    - About 640MB of memory can be saved per block.
    - Since the memory usage of one double block and two single blocks is almost the same, the transfer of single blocks is done in units of two. For example, consider the case of `--blocks_to_swap 6`.
      - Before the forward pass, all double blocks and 26 (=38-12) single blocks are on the GPU. The last 12 single blocks are on the CPU.
      - In the forward pass, the 6 double blocks that have finished calculation (the first 6 blocks) are transferred to the CPU, and the 12 single blocks to be calculated (the last 12 blocks) are transferred to the GPU.
      - The same is true for the backward pass, but in reverse order. The 12 single blocks that have finished calculation are transferred to the CPU, and the 6 double blocks to be calculated are transferred to the GPU. 
      - After the backward pass, the blocks are back to their original locations.

2. Sample Image Generation:
   - Sample image generation during training is now supported.
   - The prompts are cached and used for generation if `--cache_latents` is specified. So changing the prompts during training will not affect the generated images.
   - Specify options such as `--sample_prompts` and `--sample_every_n_epochs`.
   - Note: It will be very slow when `--split_mode` is specified.

3. Experimental Memory-Efficient Saving:
   - `--mem_eff_save` option can further reduce memory consumption during model saving (about 22GB).
   - This is a custom implementation and may cause unexpected issues. Use with caution.

4. T5XXL Token Length Control:
   - Added `--t5xxl_max_token_length` option to specify the maximum token length of T5XXL.
   - Default is 512 in dev and 256 in schnell models.

5. Multi-GPU Training Support:
   - Note: `--double_blocks_to_swap` and `--single_blocks_to_swap` cannot be used in multi-GPU training.

6. Disable mmap Load for Safetensors:
   - `--disable_mmap_load_safetensors` option now works in `flux_train.py`.
   - Speeds up model loading during training in WSL2.
   - Effective in reducing memory usage when loading models during multi-GPU training.


### Extract LoRA from FLUX.1 Models

Script: `networks/flux_extract_lora.py`

Extracts LoRA from the difference between two FLUX.1 models.

Offers memory-efficient option with `--mem_eff_safe_open`.

CLIP-L LoRA is not supported.

### Convert FLUX LoRA

Script: `convert_flux_lora.py`

Converts LoRA between sd-scripts format (BFL-based) and AI-toolkit format (Diffusers-based).

If you use LoRA in the inference environment, converting it to AI-toolkit format may reduce temporary memory usage.

Note that re-conversion will increase the size of LoRA.

CLIP-L/T5XXL LoRA is not supported.

### Merge LoRA to FLUX.1 checkpoint

`networks/flux_merge_lora.py` merges LoRA to FLUX.1 checkpoint, CLIP-L or T5XXL models. __The script is experimental.__ 

```
python networks/flux_merge_lora.py --flux_model flux1-dev.safetensors --save_to output.safetensors --models lora1.safetensors --ratios 2.0 --save_precision fp16 --loading_device cuda --working_device cpu
```

You can also merge multiple LoRA models into a FLUX.1 model. Specify multiple LoRA models in `--models`. Specify the same number of ratios in `--ratios`.

CLIP-L and T5XXL LoRA are supported. `--clip_l` and `--clip_l_save_to` are for CLIP-L, `--t5xxl` and `--t5xxl_save_to` are for T5XXL. Sample command is below.

```
--clip_l clip_l.safetensors --clip_l_save_to merged_clip_l.safetensors  --t5xxl t5xxl_fp16.safetensors --t5xxl_save_to merged_t5xxl.safetensors
```

FLUX.1, CLIP-L, and T5XXL can be merged together or separately for memory efficiency.

An experimental option `--mem_eff_load_save` is available. This option is for memory-efficient loading and saving. It may also speed up loading and saving. 

`--loading_device` is the device to load the LoRA models. `--working_device` is the device to merge (calculate) the models. Default is `cpu` for both. Loading / working device examples are below (in the case of `--save_precision fp16` or `--save_precision bf16`, `float32` will consume more memory):

- 'cpu' / 'cpu': Uses >50GB of RAM, but works on any machine.
- 'cuda' / 'cpu': Uses 24GB of VRAM, but requires 30GB of RAM.
- 'cpu' / 'cuda': Uses 4GB of VRAM, but requires 50GB of RAM, faster than 'cpu' / 'cpu' or 'cuda' / 'cpu'.
- 'cuda' / 'cuda': Uses 30GB of VRAM, but requires 30GB of RAM, faster than 'cpu' / 'cpu' or 'cuda' / 'cpu'.

`--save_precision` is the precision to save the merged model. In the case of LoRA models are trained with `bf16`, we are not sure which is better, `fp16` or `bf16` for `--save_precision`.

The script can merge multiple LoRA models. If you want to merge multiple LoRA models, specify `--concat` option to work the merged LoRA model properly.

### FLUX.1 Multi-resolution training

You can define multiple resolutions in the dataset configuration file.

The dataset configuration file is like below. You can define multiple resolutions with different batch sizes. The resolutions are defined in the `[[datasets]]` section. The `[[datasets.subsets]]` section is for the dataset directory. Please specify the same directory for each resolution.

```
[general]
# define common settings here
flip_aug = true
color_aug = false
keep_tokens_separator= "|||"
shuffle_caption = false
caption_tag_dropout_rate = 0
caption_extension = ".txt"

[[datasets]]
# define the first resolution here
batch_size = 2
enable_bucket = true
resolution = [1024, 1024]

  [[datasets.subsets]]
  image_dir = "path/to/image/dir"
  num_repeats = 1

[[datasets]]
# define the second resolution here
batch_size = 3
enable_bucket = true
resolution = [768, 768]

  [[datasets.subsets]]
  image_dir = "path/to/image/dir"
  num_repeats = 1

[[datasets]]
# define the third resolution here
batch_size = 4
enable_bucket = true
resolution = [512, 512]

  [[datasets.subsets]]
  image_dir = "path/to/image/dir"
  num_repeats = 1
```

### Convert Diffusers to FLUX.1

Script: `convert_diffusers_to_flux1.py`

Converts Diffusers models to FLUX.1 models. The script is experimental. See `--help` for options. schnell and dev models are supported. AE/CLIP/T5XXL are not supported. The diffusers folder is a parent folder of `rmer` folder.

```
python tools/convert_diffusers_to_flux.py --diffusers_path path/to/diffusers_folder_or_00001_safetensors --save_to path/to/flux1.safetensors --mem_eff_load_save --save_precision bf16
```

## SD3 training

SD3 training is done with `sd3_train.py`. 

__Sep 1, 2024__:
- `--num_last_block_to_freeze` is added to `sd3_train.py`. This option is to freeze the last n blocks of the MMDiT. See [#1417](https://github.com/kohya-ss/sd-scripts/pull/1417) for details. Thanks to sdbds!

__Jul  27, 2024__: 
- Latents and text encoder outputs caching mechanism is refactored significantly. 
  - Existing cache files for SD3 need to be recreated. Please delete the previous cache files. 
  - With this change, dataset initialization is significantly faster, especially for large datasets. 

- Architecture-dependent parts are extracted from the dataset (`train_util.py`). This is expected to make it easier to add future architectures.

- Architecture-dependent parts including the cache mechanism for SD1/2/SDXL are also extracted. The basic operation of SD1/2/SDXL training on the sd3 branch has been confirmed, but there may be bugs. Please use the main or dev branch for SD1/2/SDXL training.

---

`fp16` and `bf16` are available for mixed precision training. We are not sure which is better.

`optimizer_type = "adafactor"` is recommended for 24GB VRAM GPUs. `cache_text_encoder_outputs_to_disk` and `cache_latents_to_disk` are necessary currently. 

`clip_l`, `clip_g` and `t5xxl` can be specified if the checkpoint does not include them.  

t5xxl works with `fp16` now.

There are `t5xxl_device` and `t5xxl_dtype` options for `t5xxl` device and dtype. 

`text_encoder_batch_size` is added experimentally for caching faster.

```toml
learning_rate = 1e-6 # seems to depend on the batch size
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
cache_text_encoder_outputs = true
cache_text_encoder_outputs_to_disk = true
vae_batch_size = 1
text_encoder_batch_size = 4
cache_latents = true
cache_latents_to_disk = true
```

__2024/7/27:__

Latents およびテキストエンコーダ出力のキャッシュの仕組みを大きくリファクタリングしました。SD3 用の既存のキャッシュファイルの再作成が必要になりますが、ご了承ください（以前のキャッシュファイルは削除してください）。これにより、特にデータセットの規模が大きい場合のデータセット初期化が大幅に高速化されます。

データセット (`train_util.py`) からアーキテクチャ依存の部分を切り出しました。これにより将来的なアーキテクチャ追加が容易になると期待しています。

SD1/2/SDXL のキャッシュ機構を含むアーキテクチャ依存の部分も切り出しました。sd3 ブランチの SD1/2/SDXL 学習について、基本的な動作は確認していますが、不具合があるかもしれません。SD1/2/SDXL の学習には main または dev ブランチをお使いください。

--- 

[__Change History__](#change-history) is moved to the bottom of the page. 
更新履歴は[ページ末尾](#change-history)に移しました。

[日本語版READMEはこちら](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Textual Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

## About requirements.txt

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with Pytorch 2.1.2. 2.0.1 and 1.12.1 is not tested but should work.

## Links to usage documentation

Most of the documents are written in Japanese.

[English translation by darkstorm2150 is here](https://github.com/darkstorm2150/sd-scripts#links-to-usage-documentation). Thanks to darkstorm2150!

* [Training guide - common](./docs/train_README-ja.md) : data preparation, options etc... 
  * [Chinese version](./docs/train_README-zh.md)
* [SDXL training](./docs/train_SDXL-en.md) (English version)
* [Dataset config](./docs/config_README-ja.md) 
  * [English version](./docs/config_README-en.md)
* [DreamBooth training guide](./docs/train_db_README-ja.md)
* [Step by Step fine-tuning guide](./docs/fine_tune_README_ja.md):
* [Training LoRA](./docs/train_network_README-ja.md)
* [Training Textual Inversion](./docs/train_ti_README-ja.md)
* [Image generation](./docs/gen_img_README-ja.md)
* note.com [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad)

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- git: https://git-scm.com/download/win

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

## Windows Installation

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade -r requirements.txt
pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu118

accelerate config
```

If `python -m venv` shows only `python`, change `python` to `py`.

__Note:__ Now `bitsandbytes==0.43.0`, `prodigyopt==1.0` and `lion-pytorch==0.0.6` are included in the requirements.txt. If you'd like to use the another version, please install it manually.

This installation is for CUDA 11.8. If you use a different version of CUDA, please install the appropriate version of PyTorch and xformers. For example, if you use CUDA 12, please install `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` and `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121`.

<!-- 
cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py
-->
Answers to accelerate config:

```txt
- This machine
- No distributed training
- NO
- NO
- NO
- all
- fp16
```

If you'd like to use bf16, please answer `bf16` to the last question.

Note: Some user reports ``ValueError: fp16 mixed precision requires a GPU`` is occurred in training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Single GPU with id `0` will be used.)

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

### Upgrade PyTorch

If you want to upgrade PyTorch, you can upgrade it with `pip install` command in [Windows Installation](#windows-installation) section. `xformers` is also required to be upgraded when PyTorch is upgraded.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


## Change History

### Working in progress

- __important__ The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
  - bitsandbytes, transformers, accelerate and huggingface_hub are updated. 
  - If you encounter any issues, please report them.

- There was a bug where the min_bucket_reso/max_bucket_reso in the dataset configuration did not create the correct resolution bucket if it was not divisible by bucket_reso_steps. These values are now warned and automatically rounded to a divisible value. Thanks to Maru-mee for raising the issue. Related PR [#1632](https://github.com/kohya-ss/sd-scripts/pull/1632)

- `bitsandbytes` is updated to 0.44.0. Now you can use `AdEMAMix8bit` and `PagedAdEMAMix8bit` in the training script. PR [#1640](https://github.com/kohya-ss/sd-scripts/pull/1640) Thanks to sdbds!
  - There is no abbreviation, so please specify the full path like `--optimizer_type bitsandbytes.optim.AdEMAMix8bit` (not bnb but bitsandbytes).

- Fixed a bug in the cache of latents. When `flip_aug`, `alpha_mask`, and `random_crop` are different in multiple subsets in the dataset configuration file (.toml), the last subset is used instead of reflecting them correctly.

- Fixed an issue where the timesteps in the batch were the same when using Huber loss. PR [#1628](https://github.com/kohya-ss/sd-scripts/pull/1628) Thanks to recris!

- Improvements in OFT (Orthogonal Finetuning) Implementation
  1. Optimization of Calculation Order:
      - Changed the calculation order in the forward method from (Wx)R to W(xR).
      - This has improved computational efficiency and processing speed.
  2. Correction of Bias Application:
      - In the previous implementation, R was incorrectly applied to the bias.
      - The new implementation now correctly handles bias by using F.conv2d and F.linear.
  3. Efficiency Enhancement in Matrix Operations:
      - Introduced einsum in both the forward and merge_to methods.
      - This has optimized matrix operations, resulting in further speed improvements.
  4. Proper Handling of Data Types:
      - Improved to use torch.float32 during calculations and convert results back to the original data type.
      - This maintains precision while ensuring compatibility with the original model.
  5. Unified Processing for Conv2d and Linear Layers:
     - Implemented a consistent method for applying OFT to both layer types.
  - These changes have made the OFT implementation more efficient and accurate, potentially leading to improved model performance and training stability.

  - Additional Information
    * Recommended α value for OFT constraint: We recommend using α values between 1e-4 and 1e-2. This differs slightly from the original implementation of "(α\*out_dim\*out_dim)". Our implementation uses "(α\*out_dim)", hence we recommend higher values than the 1e-5 suggested in the original implementation.

    * Performance Improvement: Training speed has been improved by approximately 30%.

    * Inference Environment: This implementation is compatible with and operates within Stable Diffusion web UI (SD1/2 and SDXL).

- The INVERSE_SQRT, COSINE_WITH_MIN_LR, and WARMUP_STABLE_DECAY learning rate schedules are now available in the transformers library. See PR [#1393](https://github.com/kohya-ss/sd-scripts/pull/1393) for details. Thanks to sdbds!
  - See the [transformers documentation](https://huggingface.co/docs/transformers/v4.44.2/en/main_classes/optimizer_schedules#schedules) for details on each scheduler.
  - `--lr_warmup_steps` and `--lr_decay_steps` can now be specified as a ratio of the number of training steps, not just the step value. Example: `--lr_warmup_steps=0.1` or `--lr_warmup_steps=10%`, etc.

https://github.com/kohya-ss/sd-scripts/pull/1393
- When enlarging images in the script (when the size of the training image is small and bucket_no_upscale is not specified), it has been changed to use Pillow's resize and LANCZOS interpolation instead of OpenCV2's resize and Lanczos4 interpolation. The quality of the image enlargement may be slightly improved. PR [#1426](https://github.com/kohya-ss/sd-scripts/pull/1426) Thanks to sdbds!

- Sample image generation during training now works on non-CUDA devices. PR [#1433](https://github.com/kohya-ss/sd-scripts/pull/1433) Thanks to millie-v!

- `--v_parameterization` is available in `sdxl_train.py`. The results are unpredictable, so use with caution. PR [#1505](https://github.com/kohya-ss/sd-scripts/pull/1505) Thanks to liesened!

- Fused optimizer is available for SDXL training. PR [#1259](https://github.com/kohya-ss/sd-scripts/pull/1259) Thanks to 2kpr!
  - The memory usage during training is significantly reduced by integrating the optimizer's backward pass with step. The training results are the same as before, but if you have plenty of memory, the speed will be slower.
  - Specify the `--fused_backward_pass` option in `sdxl_train.py`. At this time, only Adafactor is supported. Gradient accumulation is not available.
  - Setting mixed precision to `no` seems to use less memory than `fp16` or `bf16`.
  - Training is possible with a memory usage of about 17GB with a batch size of 1 and fp32. If you specify the `--full_bf16` option, you can further reduce the memory usage (but the accuracy will be lower). With the same memory usage as before, you can increase the batch size.
  - PyTorch 2.1 or later is required because it uses the new API `Tensor.register_post_accumulate_grad_hook(hook)`.
  - Mechanism: Normally, backward -> step is performed for each parameter, so all gradients need to be temporarily stored in memory. "Fuse backward and step" reduces memory usage by performing backward/step for each parameter and reflecting the gradient immediately. The more parameters there are, the greater the effect, so it is not effective in other training scripts (LoRA, etc.) where the memory usage peak is elsewhere, and there are no plans to implement it in those training scripts.

- Optimizer groups feature is added to SDXL training. PR [#1319](https://github.com/kohya-ss/sd-scripts/pull/1319)
  - Memory usage is reduced by the same principle as Fused optimizer. The training results and speed are the same as Fused optimizer.
  - Specify the number of groups like `--fused_optimizer_groups 10` in `sdxl_train.py`. Increasing the number of groups reduces memory usage but slows down training. Since the effect is limited to a certain number, it is recommended to specify 4-10.
  - Any optimizer can be used, but optimizers that automatically calculate the learning rate (such as D-Adaptation and Prodigy) cannot be used. Gradient accumulation is not available.
  - `--fused_optimizer_groups` cannot be used with `--fused_backward_pass`. When using Adafactor, the memory usage is slightly larger than with Fused optimizer. PyTorch 2.1 or later is required.
  - Mechanism: While Fused optimizer performs backward/step for individual parameters within the optimizer, optimizer groups reduce memory usage by grouping parameters and creating multiple optimizers to perform backward/step for each group. Fused optimizer requires implementation on the optimizer side, while optimizer groups are implemented only on the training script side.

- LoRA+ is supported. PR [#1233](https://github.com/kohya-ss/sd-scripts/pull/1233) Thanks to rockerBOO!
  - LoRA+ is a method to improve training speed by increasing the learning rate of the UP side (LoRA-B) of LoRA. Specify the multiple. The original paper recommends 16, but adjust as needed. Please see the PR for details.
  - Specify `loraplus_lr_ratio` with `--network_args`. Example: `--network_args "loraplus_lr_ratio=16"`
  - `loraplus_unet_lr_ratio` and `loraplus_lr_ratio` can be specified separately for U-Net and Text Encoder.
    - Example: `--network_args "loraplus_unet_lr_ratio=16" "loraplus_text_encoder_lr_ratio=4"` or `--network_args "loraplus_lr_ratio=16" "loraplus_text_encoder_lr_ratio=4"` etc.
  - `network_module` `networks.lora` and `networks.dylora` are available.

- The feature to use the transparency (alpha channel) of the image as a mask in the loss calculation has been added. PR [#1223](https://github.com/kohya-ss/sd-scripts/pull/1223) Thanks to u-haru!
  - The transparent part is ignored during training. Specify the `--alpha_mask` option in the training script or specify `alpha_mask = true` in the dataset configuration file.
  - See [About masked loss](./docs/masked_loss_README.md) for details.

- LoRA training in SDXL now supports block-wise learning rates and block-wise dim (rank). PR [#1331](https://github.com/kohya-ss/sd-scripts/pull/1331) 
  - Specify the learning rate and dim (rank) for each block.
  - See [Block-wise learning rates in LoRA](./docs/train_network_README-ja.md#階層別学習率) for details (Japanese only).

- Negative learning rates can now be specified during SDXL model training. PR [#1277](https://github.com/kohya-ss/sd-scripts/pull/1277) Thanks to Cauldrath!
  - The model is trained to move away from the training images, so the model is easily collapsed. Use with caution. A value close to 0 is recommended.
  - When specifying from the command line, use `=` like `--learning_rate=-1e-7`.

- Training scripts can now output training settings to wandb or Tensor Board logs. Specify the `--log_config` option. PR [#1285](https://github.com/kohya-ss/sd-scripts/pull/1285)  Thanks to ccharest93, plucked, rockerBOO, and VelocityRa!
  - Some settings, such as API keys and directory specifications, are not output due to security issues.

- The ControlNet training script `train_controlnet.py` for SD1.5/2.x was not working, but it has been fixed. PR [#1284](https://github.com/kohya-ss/sd-scripts/pull/1284) Thanks to sdbds!

- `train_network.py` and `sdxl_train_network.py` now restore the order/position of data loading from DataSet when resuming training. PR [#1353](https://github.com/kohya-ss/sd-scripts/pull/1353) [#1359](https://github.com/kohya-ss/sd-scripts/pull/1359) Thanks to KohakuBlueleaf!
  - This resolves the issue where the order of data loading from DataSet changes when resuming training.
  - Specify the `--skip_until_initial_step` option to skip data loading until the specified step. If not specified, data loading starts from the beginning of the DataSet (same as before).
  - If `--resume` is specified, the step saved in the state is used.
  - Specify the `--initial_step` or `--initial_epoch` option to skip data loading until the specified step or epoch. Use these options in conjunction with `--skip_until_initial_step`. These options can be used without `--resume` (use them when resuming training with `--network_weights`).

- An option `--disable_mmap_load_safetensors` is added to disable memory mapping when loading the model's .safetensors in SDXL. PR [#1266](https://github.com/kohya-ss/sd-scripts/pull/1266) Thanks to Zovjsra!
  - It seems that the model file loading is faster in the WSL environment etc.
  - Available in `sdxl_train.py`, `sdxl_train_network.py`, `sdxl_train_textual_inversion.py`, and `sdxl_train_control_net_lllite.py`.

- When there is an error in the cached latents file on disk, the file name is now displayed. PR [#1278](https://github.com/kohya-ss/sd-scripts/pull/1278) Thanks to Cauldrath!

- Fixed an error that occurs when specifying `--max_dataloader_n_workers` in `tag_images_by_wd14_tagger.py` when Onnx is not used. PR [#1291](
https://github.com/kohya-ss/sd-scripts/pull/1291) issue [#1290](
https://github.com/kohya-ss/sd-scripts/pull/1290) Thanks to frodo821!

- Fixed a bug that `caption_separator` cannot be specified in the subset in the dataset settings .toml file.  [#1312](https://github.com/kohya-ss/sd-scripts/pull/1312) and [#1313](https://github.com/kohya-ss/sd-scripts/pull/1312) Thanks to rockerBOO!

- Fixed a potential bug in ControlNet-LLLite training. PR [#1322](https://github.com/kohya-ss/sd-scripts/pull/1322) Thanks to aria1th!

- Fixed some bugs when using DeepSpeed. Related [#1247](https://github.com/kohya-ss/sd-scripts/pull/1247)

- Added a prompt option `--f` to `gen_imgs.py` to specify the file name when saving. Also, Diffusers-based keys for LoRA weights are now supported.

- SDXL の学習時に Fused optimizer が使えるようになりました。PR [#1259](https://github.com/kohya-ss/sd-scripts/pull/1259) 2kpr 氏に感謝します。
  - optimizer の backward pass に step を統合することで学習時のメモリ使用量を大きく削減します。学習結果は未適用時と同一ですが、メモリが潤沢にある場合は速度は遅くなります。
  - `sdxl_train.py` に `--fused_backward_pass` オプションを指定してください。現時点では optimizer は Adafactor のみ対応しています。また gradient accumulation は使えません。
  - mixed precision は `no` のほうが `fp16` や `bf16` よりも使用メモリ量が少ないようです。
  - バッチサイズ 1、fp32 で 17GB 程度で学習可能なようです。`--full_bf16` オプションを指定するとさらに削減できます（精度は劣ります）。以前と同じメモリ使用量ではバッチサイズを増やせます。
  - PyTorch 2.1 以降の新 API `Tensor.register_post_accumulate_grad_hook(hook)` を使用しているため、PyTorch 2.1 以降が必要です。
  - 仕組み：通常は backward -> step の順で行うためすべての勾配を一時的にメモリに保持する必要があります。「backward と step の統合」はパラメータごとに backward/step を行って、勾配をすぐ反映することでメモリ使用量を削減します。パラメータ数が多いほど効果が大きいため、SDXL の学習以外（LoRA 等）ではほぼ効果がなく（メモリ使用量のピークが他の場所にあるため）、それらの学習スクリプトへの実装予定もありません。

- SDXL の学習時に optimizer group 機能を追加しました。PR [#1319](https://github.com/kohya-ss/sd-scripts/pull/1319)
  - Fused optimizer と同様の原理でメモリ使用量を削減します。学習結果や速度についても同様です。
  - `sdxl_train.py` に `--fused_optimizer_groups 10` のようにグループ数を指定してください。グループ数を増やすとメモリ使用量が削減されますが、速度は遅くなります。ある程度の数までしか効果がないため、4~10 程度を指定すると良いでしょう。
  - 任意の optimizer が使えますが、学習率を自動計算する optimizer （D-Adaptation や Prodigy など）は使えません。gradient accumulation は使えません。
  - `--fused_optimizer_groups` は `--fused_backward_pass` と併用できません。AdaFactor 使用時は Fused optimizer よりも若干メモリ使用量は大きくなります。PyTorch 2.1 以降が必要です。
  - 仕組み：Fused optimizer が optimizer 内で個別のパラメータについて backward/step を行っているのに対して、optimizer groups はパラメータをグループ化して複数の optimizer を作成し、それぞれ backward/step を行うことでメモリ使用量を削減します。Fused optimizer は optimizer 側の実装が必要ですが、optimizer groups は学習スクリプト側のみで実装されています。やはり SDXL の学習でのみ効果があります。

- LoRA+ がサポートされました。PR [#1233](https://github.com/kohya-ss/sd-scripts/pull/1233) rockerBOO 氏に感謝します。
  - LoRA の UP 側（LoRA-B）の学習率を上げることで学習速度の向上を図る手法です。倍数で指定します。元の論文では 16 が推奨されていますが、データセット等にもよりますので、適宜調整してください。PR もあわせてご覧ください。
  - `--network_args` で `loraplus_lr_ratio` を指定します。例：`--network_args "loraplus_lr_ratio=16"`
  - `loraplus_unet_lr_ratio` と `loraplus_lr_ratio` で、U-Net および Text Encoder に個別の値を指定することも可能です。
    - 例：`--network_args "loraplus_unet_lr_ratio=16" "loraplus_text_encoder_lr_ratio=4"` または `--network_args "loraplus_lr_ratio=16" "loraplus_text_encoder_lr_ratio=4"` など
  - `network_module` の `networks.lora` および `networks.dylora` で使用可能です。

- 画像の透明度（アルファチャネル）をロス計算時のマスクとして使用する機能が追加されました。PR [#1223](https://github.com/kohya-ss/sd-scripts/pull/1223) u-haru 氏に感謝します。
  - 透明部分が学習時に無視されるようになります。学習スクリプトに `--alpha_mask` オプションを指定するか、データセット設定ファイルに `alpha_mask = true` を指定してください。
  - 詳細は [マスクロスについて](./docs/masked_loss_README-ja.md) をご覧ください。

- SDXL の LoRA で階層別学習率、階層別 dim (rank) をサポートしました。PR [#1331](https://github.com/kohya-ss/sd-scripts/pull/1331) 
  - ブロックごとに学習率および dim (rank) を指定することができます。
  - 詳細は [LoRA の階層別学習率](./docs/train_network_README-ja.md#階層別学習率) をご覧ください。

- `sdxl_train.py` での SDXL モデル学習時に負の学習率が指定できるようになりました。PR [#1277](https://github.com/kohya-ss/sd-scripts/pull/1277) Cauldrath 氏に感謝します。
  - 学習画像から離れるように学習するため、モデルは容易に崩壊します。注意して使用してください。0 に近い値を推奨します。
  - コマンドラインから指定する場合、`--learning_rate=-1e-7` のように`=` を使ってください。

- 各学習スクリプトで学習設定を wandb や Tensor Board などのログに出力できるようになりました。`--log_config` オプションを指定してください。PR [#1285](https://github.com/kohya-ss/sd-scripts/pull/1285)  ccharest93 氏、plucked 氏、rockerBOO 氏および VelocityRa 氏に感謝します。
  - API キーや各種ディレクトリ指定など、一部の設定はセキュリティ上の問題があるため出力されません。

- SD1.5/2.x 用の ControlNet 学習スクリプト `train_controlnet.py` が動作しなくなっていたのが修正されました。PR [#1284](https://github.com/kohya-ss/sd-scripts/pull/1284) sdbds 氏に感謝します。

- `train_network.py` および `sdxl_train_network.py` で、学習再開時に DataSet の読み込み順についても復元できるようになりました。PR [#1353](https://github.com/kohya-ss/sd-scripts/pull/1353) [#1359](https://github.com/kohya-ss/sd-scripts/pull/1359) KohakuBlueleaf 氏に感謝します。
  - これにより、学習再開時に DataSet の読み込み順が変わってしまう問題が解消されます。
  - `--skip_until_initial_step` オプションを指定すると、指定したステップまで DataSet 読み込みをスキップします。指定しない場合の動作は変わりません（DataSet の最初から読み込みます）
  - `--resume` オプションを指定すると、state に保存されたステップ数が使用されます。
  - `--initial_step` または `--initial_epoch` オプションを指定すると、指定したステップまたはエポックまで DataSet 読み込みをスキップします。これらのオプションは `--skip_until_initial_step` と併用してください。またこれらのオプションは `--resume` と併用しなくても使えます（`--network_weights` を用いた学習再開時などにお使いください ）。

- SDXL でモデルの .safetensors を読み込む際にメモリマッピングを無効化するオプション `--disable_mmap_load_safetensors` が追加されました。PR [#1266](https://github.com/kohya-ss/sd-scripts/pull/1266) Zovjsra 氏に感謝します。
  - WSL 環境等でモデルファイルの読み込みが高速化されるようです。
  - `sdxl_train.py`、`sdxl_train_network.py`、`sdxl_train_textual_inversion.py`、`sdxl_train_control_net_lllite.py` で使用可能です。

- ディスクにキャッシュされた latents ファイルに何らかのエラーがあったとき、そのファイル名が表示されるようになりました。 PR [#1278](https://github.com/kohya-ss/sd-scripts/pull/1278) Cauldrath 氏に感謝します。

- `tag_images_by_wd14_tagger.py` で Onnx 未使用時に `--max_dataloader_n_workers` を指定するとエラーになる不具合が修正されました。 PR [#1291](
https://github.com/kohya-ss/sd-scripts/pull/1291) issue [#1290](
https://github.com/kohya-ss/sd-scripts/pull/1290) frodo821 氏に感謝します。

- データセット設定の .toml ファイルで、`caption_separator` が subset に指定できない不具合が修正されました。 PR [#1312](https://github.com/kohya-ss/sd-scripts/pull/1312) および [#1313](https://github.com/kohya-ss/sd-scripts/pull/1313) rockerBOO 氏に感謝します。

- ControlNet-LLLite 学習時の潜在バグが修正されました。 PR [#1322](https://github.com/kohya-ss/sd-scripts/pull/1322) aria1th 氏に感謝します。

- DeepSpeed 使用時のいくつかのバグを修正しました。関連 [#1247](https://github.com/kohya-ss/sd-scripts/pull/1247)

- `gen_imgs.py` のプロンプトオプションに、保存時のファイル名を指定する `--f` オプションを追加しました。また同スクリプトで Diffusers ベースのキーを持つ LoRA の重みに対応しました。


### Sep 13, 2024 / 2024-09-13: 

- `sdxl_merge_lora.py` now supports OFT. Thanks to Maru-mee for the PR [#1580](https://github.com/kohya-ss/sd-scripts/pull/1580). 
- `svd_merge_lora.py` now supports LBW. Thanks to terracottahaniwa. See PR [#1575](https://github.com/kohya-ss/sd-scripts/pull/1575) for details.
- `sdxl_merge_lora.py` also supports LBW. 
- See [LoRA Block Weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) by hako-mikan for details on LBW.
- These will be included in the next release.

- `sdxl_merge_lora.py` が OFT をサポートされました。PR [#1580](https://github.com/kohya-ss/sd-scripts/pull/1580) Maru-mee 氏に感謝します。
- `svd_merge_lora.py` で LBW がサポートされました。PR [#1575](https://github.com/kohya-ss/sd-scripts/pull/1575) terracottahaniwa 氏に感謝します。
- `sdxl_merge_lora.py` でも LBW がサポートされました。
- LBW の詳細は hako-mikan 氏の [LoRA Block Weight](https://github.com/hako-mikan/sd-webui-lora-block-weight) をご覧ください。
- 以上は次回リリースに含まれます。

### Jun 23, 2024 / 2024-06-23: 

- Fixed `cache_latents.py` and `cache_text_encoder_outputs.py` not working. (Will be included in the next release.)

- `cache_latents.py` および `cache_text_encoder_outputs.py` が動作しなくなっていたのを修正しました。（次回リリースに含まれます。）

### Apr 7, 2024 / 2024-04-07: v0.8.7

- The default value of `huber_schedule` in Scheduled Huber Loss is changed from `exponential` to `snr`, which is expected to give better results.

- Scheduled Huber Loss の `huber_schedule` のデフォルト値を `exponential` から、より良い結果が期待できる `snr` に変更しました。

### Apr 7, 2024 / 2024-04-07: v0.8.6

#### Highlights

- The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
  - Especially `imagesize` is newly added, so if you cannot update the libraries immediately, please install with `pip install imagesize==1.4.1` separately.
  - `bitsandbytes==0.43.0`, `prodigyopt==1.0`, `lion-pytorch==0.0.6` are included in the requirements.txt.
    - `bitsandbytes` no longer requires complex procedures as it now officially supports Windows.  
  - Also, the PyTorch version is updated to 2.1.2 (PyTorch does not need to be updated immediately). In the upgrade procedure, PyTorch is not updated, so please manually install or update torch, torchvision, xformers if necessary (see [Upgrade PyTorch](#upgrade-pytorch)).
- When logging to wandb is enabled, the entire command line is exposed. Therefore, it is recommended to write wandb API key and HuggingFace token in the configuration file (`.toml`). Thanks to bghira for raising the issue.
  - A warning is displayed at the start of training if such information is included in the command line.
  - Also, if there is an absolute path, the path may be exposed, so it is recommended to specify a relative path or write it in the configuration file. In such cases, an INFO log is displayed.
  - See [#1123](https://github.com/kohya-ss/sd-scripts/pull/1123) and PR [#1240](https://github.com/kohya-ss/sd-scripts/pull/1240) for details.
- Colab seems to stop with log output. Try specifying `--console_log_simple` option in the training script to disable rich logging.
- Other improvements include the addition of masked loss, scheduled Huber Loss, DeepSpeed support, dataset settings improvements, and image tagging improvements. See below for details.

#### Training scripts

- `train_network.py` and `sdxl_train_network.py` are modified to record some dataset settings in the metadata of the trained model (`caption_prefix`, `caption_suffix`, `keep_tokens_separator`, `secondary_separator`, `enable_wildcard`).
- Fixed a bug that U-Net and Text Encoders are included in the state in `train_network.py` and `sdxl_train_network.py`. The saving and loading of the state are faster, the file size is smaller, and the memory usage when loading is reduced.
- DeepSpeed is supported. PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101)  and [#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) Thanks to BootsofLagrangian! See PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) for details.
- The masked loss is supported in each training script. PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) See [Masked loss](#about-masked-loss) for details.
- Scheduled Huber Loss has been introduced to each training scripts. PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) Thanks to kabachuha for the PR and cheald, drhead, and others for the discussion! See the PR and [Scheduled Huber Loss](#about-scheduled-huber-loss) for details.
- The options `--noise_offset_random_strength` and `--ip_noise_gamma_random_strength` are added to each training script. These options can be used to vary the noise offset and ip noise gamma in the range of 0 to the specified value. PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) Thanks to KohakuBlueleaf!
- The options `--save_state_on_train_end` are added to each training script. PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) Thanks to gesen2egee!
- The options `--sample_every_n_epochs` and `--sample_every_n_steps` in each training script now display a warning and ignore them when a number less than or equal to `0` is specified. Thanks to S-Del for raising the issue.

#### Dataset settings

- The [English version of the dataset settings documentation](./docs/config_README-en.md) is added. PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) Thanks to darkstorm2150!
- The `.toml` file for the dataset config is now read in UTF-8 encoding. PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Thanks to Horizon1704!
- Fixed a bug that the last subset settings are applied to all images when multiple subsets of regularization images are specified in the dataset settings. The settings for each subset are correctly applied to each image. PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) Thanks to feffy380!
- Some features are added to the dataset subset settings.
  - `secondary_separator` is added to specify the tag separator that is not the target of shuffling or dropping. 
    - Specify `secondary_separator=";;;"`. When you specify `secondary_separator`, the part is not shuffled or dropped. 
  - `enable_wildcard` is added. When set to `true`, the wildcard notation `{aaa|bbb|ccc}` can be used. The multi-line caption is also enabled.
  - `keep_tokens_separator` is updated to be used twice in the caption. When you specify `keep_tokens_separator="|||"`, the part divided by the second `|||` is not shuffled or dropped and remains at the end.
  - The existing features `caption_prefix` and `caption_suffix` can be used together. `caption_prefix` and `caption_suffix` are processed first, and then `enable_wildcard`, `keep_tokens_separator`, shuffling and dropping, and `secondary_separator` are processed in order.
  - See [Dataset config](./docs/config_README-en.md) for details.
- The dataset with DreamBooth method supports caching image information (size, caption). PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178) and [#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) Thanks to KohakuBlueleaf! See [DreamBooth method specific options](./docs/config_README-en.md#dreambooth-specific-options) for details.

#### Image tagging

- The support for v3 repositories is added to `tag_image_by_wd14_tagger.py` (`--onnx` option only). PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) Thanks to sdbds!
  - Onnx may need to be updated. Onnx is not installed by default, so please install or update it with `pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` etc. Please also check the comments in `requirements.txt`.
- The model is now saved in the subdirectory as `--repo_id` in `tag_image_by_wd14_tagger.py` . This caches multiple repo_id models. Please delete unnecessary files under `--model_dir`.
- Some options are added to `tag_image_by_wd14_tagger.py`.
  - Some are added in PR [#1216](https://github.com/kohya-ss/sd-scripts/pull/1216) Thanks to Disty0!
  - Output rating tags `--use_rating_tags` and `--use_rating_tags_as_last_tag`
  - Output character tags first `--character_tags_first`
  - Expand character tags and series `--character_tag_expand`
  - Specify tags to output first `--always_first_tags`
  - Replace tags `--tag_replacement`
  - See [Tagging documentation](./docs/wd14_tagger_README-en.md) for details.
- Fixed an error when specifying `--beam_search` and a value of 2 or more for `--num_beams` in `make_captions.py`.

#### About Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

#### About Scheduled Huber Loss

Scheduled Huber Loss has been introduced to each training scripts. This is a method to improve robustness against outliers or anomalies (data corruption) in the training data.

With the traditional MSE (L2) loss function, the impact of outliers could be significant, potentially leading to a degradation in the quality of generated images. On the other hand, while the Huber loss function can suppress the influence of outliers, it tends to compromise the reproduction of fine details in images.

To address this, the proposed method employs a clever application of the Huber loss function. By scheduling the use of Huber loss in the early stages of training (when noise is high) and MSE in the later stages, it strikes a balance between outlier robustness and fine detail reproduction.

Experimental results have confirmed that this method achieves higher accuracy on data containing outliers compared to pure Huber loss or MSE. The increase in computational cost is minimal.

The newly added arguments loss_type, huber_schedule, and huber_c allow for the selection of the loss function type (Huber, smooth L1, MSE), scheduling method (exponential, constant, SNR), and Huber's parameter. This enables optimization based on the characteristics of the dataset.

See PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) for details.

- `loss_type`: Specify the loss function type. Choose `huber` for Huber loss, `smooth_l1` for smooth L1 loss, and `l2` for MSE loss. The default is `l2`, which is the same as before.
- `huber_schedule`: Specify the scheduling method. Choose `exponential`, `constant`, or `snr`. The default is `snr`.
- `huber_c`: Specify the Huber's parameter. The default is `0.1`.

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.

#### 主要な変更点

- 依存ライブラリが更新されました。[アップグレード](./README-ja.md#アップグレード) を参照しライブラリを更新してください。
  - 特に `imagesize` が新しく追加されていますので、すぐにライブラリの更新ができない場合は `pip install imagesize==1.4.1` で個別にインストールしてください。
  - `bitsandbytes==0.43.0`、`prodigyopt==1.0`、`lion-pytorch==0.0.6` が requirements.txt に含まれるようになりました。
    - `bitsandbytes` が公式に Windows をサポートしたため複雑な手順が不要になりました。
  - また PyTorch のバージョンを 2.1.2 に更新しました。PyTorch はすぐに更新する必要はありません。更新時は、アップグレードの手順では PyTorch が更新されませんので、torch、torchvision、xformers を手動でインストールしてください。
- wandb へのログ出力が有効の場合、コマンドライン全体が公開されます。そのため、コマンドラインに wandb の API キーや HuggingFace のトークンなどが含まれる場合、設定ファイル（`.toml`）への記載をお勧めします。問題提起していただいた bghira 氏に感謝します。
  - このような場合には学習開始時に警告が表示されます。
  - また絶対パスの指定がある場合、そのパスが公開される可能性がありますので、相対パスを指定するか設定ファイルに記載することをお勧めします。このような場合は INFO ログが表示されます。
  - 詳細は [#1123](https://github.com/kohya-ss/sd-scripts/pull/1123) および PR [#1240](https://github.com/kohya-ss/sd-scripts/pull/1240) をご覧ください。
- Colab での動作時、ログ出力で停止してしまうようです。学習スクリプトに `--console_log_simple` オプションを指定し、rich のロギングを無効してお試しください。
- その他、マスクロス追加、Scheduled Huber Loss 追加、DeepSpeed 対応、データセット設定の改善、画像タグ付けの改善などがあります。詳細は以下をご覧ください。

#### 学習スクリプト

- `train_network.py` および `sdxl_train_network.py` で、学習したモデルのメタデータに一部のデータセット設定が記録されるよう修正しました（`caption_prefix`、`caption_suffix`、`keep_tokens_separator`、`secondary_separator`、`enable_wildcard`）。
- `train_network.py` および `sdxl_train_network.py` で、state に U-Net および Text Encoder が含まれる不具合を修正しました。state の保存、読み込みが高速化され、ファイルサイズも小さくなり、また読み込み時のメモリ使用量も削減されます。
- DeepSpeed がサポートされました。PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) 、[#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) BootsofLagrangian 氏に感謝します。詳細は PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) をご覧ください。
- 各学習スクリプトでマスクロスをサポートしました。PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) 詳細は [マスクロスについて](#マスクロスについて) をご覧ください。
- 各学習スクリプトに Scheduled Huber Loss を追加しました。PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) ご提案いただいた kabachuha 氏、および議論を深めてくださった cheald 氏、drhead 氏を始めとする諸氏に感謝します。詳細は当該 PR および [Scheduled Huber Loss について](#scheduled-huber-loss-について) をご覧ください。
- 各学習スクリプトに、noise offset、ip noise gammaを、それぞれ 0~指定した値の範囲で変動させるオプション `--noise_offset_random_strength` および `--ip_noise_gamma_random_strength` が追加されました。 PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) KohakuBlueleaf 氏に感謝します。
- 各学習スクリプトに、学習終了時に state を保存する `--save_state_on_train_end` オプションが追加されました。 PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) gesen2egee 氏に感謝します。
- 各学習スクリプトで `--sample_every_n_epochs` および `--sample_every_n_steps` オプションに `0` 以下の数値を指定した時、警告を表示するとともにそれらを無視するよう変更しました。問題提起していただいた S-Del 氏に感謝します。

#### データセット設定

- データセット設定の `.toml` ファイルが UTF-8 encoding で読み込まれるようになりました。PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Horizon1704 氏に感謝します。
- データセット設定で、正則化画像のサブセットを複数指定した時、最後のサブセットの各種設定がすべてのサブセットの画像に適用される不具合が修正されました。それぞれのサブセットの設定が、それぞれの画像に正しく適用されます。PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) feffy380 氏に感謝します。
- データセットのサブセット設定にいくつかの機能を追加しました。
  - シャッフルの対象とならないタグ分割識別子の指定 `secondary_separator` を追加しました。`secondary_separator=";;;"` のように指定します。`secondary_separator` で区切ることで、その部分はシャッフル、drop 時にまとめて扱われます。
  - `enable_wildcard` を追加しました。`true` にするとワイルドカード記法 `{aaa|bbb|ccc}` が使えます。また複数行キャプションも有効になります。
  - `keep_tokens_separator` をキャプション内に 2 つ使えるようにしました。たとえば `keep_tokens_separator="|||"` と指定したとき、`1girl, hatsune miku, vocaloid ||| stage, mic ||| best quality, rating: general` とキャプションを指定すると、二番目の `|||` で分割された部分はシャッフル、drop されず末尾に残ります。
  - 既存の機能 `caption_prefix` と `caption_suffix` とあわせて使えます。`caption_prefix` と `caption_suffix` は一番最初に処理され、その後、ワイルドカード、`keep_tokens_separator`、シャッフルおよび drop、`secondary_separator` の順に処理されます。
  - 詳細は [データセット設定](./docs/config_README-ja.md) をご覧ください。
- DreamBooth 方式の DataSet で画像情報（サイズ、キャプション）をキャッシュする機能が追加されました。PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178)、[#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) KohakuBlueleaf 氏に感謝します。詳細は [データセット設定](./docs/config_README-ja.md#dreambooth-方式専用のオプション) をご覧ください。
- データセット設定の[英語版ドキュメント](./docs/config_README-en.md) が追加されました。PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) darkstorm2150 氏に感謝します。

#### 画像のタグ付け

- `tag_image_by_wd14_tagger.py` で v3 のリポジトリがサポートされました（`--onnx` 指定時のみ有効）。 PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) sdbds 氏に感謝します。
  - Onnx のバージョンアップが必要になるかもしれません。デフォルトでは Onnx はインストールされていませんので、`pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` 等でインストール、アップデートしてください。`requirements.txt` のコメントもあわせてご確認ください。
- `tag_image_by_wd14_tagger.py` で、モデルを`--repo_id` のサブディレクトリに保存するようにしました。これにより複数のモデルファイルがキャッシュされます。`--model_dir` 直下の不要なファイルは削除願います。
- `tag_image_by_wd14_tagger.py` にいくつかのオプションを追加しました。
  - 一部は PR [#1216](https://github.com/kohya-ss/sd-scripts/pull/1216) で追加されました。Disty0 氏に感謝します。
  - レーティングタグを出力する `--use_rating_tags` および `--use_rating_tags_as_last_tag`
  - キャラクタタグを最初に出力する `--character_tags_first`
  - キャラクタタグとシリーズを展開する `--character_tag_expand`
  - 常に最初に出力するタグを指定する `--always_first_tags`
  - タグを置換する `--tag_replacement`
  - 詳細は [タグ付けに関するドキュメント](./docs/wd14_tagger_README-ja.md) をご覧ください。
- `make_captions.py` で `--beam_search` を指定し `--num_beams` に2以上の値を指定した時のエラーを修正しました。

#### マスクロスについて

各学習スクリプトでマスクロスをサポートしました。マスクロスを有効にするには `--masked_loss` オプションを指定してください。

機能は完全にテストされていないため、不具合があるかもしれません。その場合は Issue を立てていただけると助かります。

マスクの指定には ControlNet データセットを使用します。マスク画像は RGB 画像である必要があります。R チャンネルのピクセル値 255 がロス計算対象、0 がロス計算対象外になります。0-255 の値は、0-1 の範囲に変換されます（つまりピクセル値 128 の部分はロスの重みが半分になります）。データセットの詳細は [LLLite ドキュメント](./docs/train_lllite_README-ja.md#データセットの準備) をご覧ください。

#### Scheduled Huber Loss について

各学習スクリプトに、学習データ中の異常値や外れ値（data corruption）への耐性を高めるための手法、Scheduled Huber Lossが導入されました。

従来のMSE（L2）損失関数では、異常値の影響を大きく受けてしまい、生成画像の品質低下を招く恐れがありました。一方、Huber損失関数は異常値の影響を抑えられますが、画像の細部再現性が損なわれがちでした。

この手法ではHuber損失関数の適用を工夫し、学習の初期段階（ノイズが大きい場合）ではHuber損失を、後期段階ではMSEを用いるようスケジューリングすることで、異常値耐性と細部再現性のバランスを取ります。

実験の結果では、この手法が純粋なHuber損失やMSEと比べ、異常値を含むデータでより高い精度を達成することが確認されています。また計算コストの増加はわずかです。

具体的には、新たに追加された引数loss_type、huber_schedule、huber_cで、損失関数の種類（Huber, smooth L1, MSE）とスケジューリング方法（exponential, constant, SNR）を選択できます。これによりデータセットに応じた最適化が可能になります。

詳細は PR [#1228](https://github.com/kohya-ss/sd-scripts/pull/1228/) をご覧ください。

- `loss_type` : 損失関数の種類を指定します。`huber` で Huber損失、`smooth_l1` で smooth L1 損失、`l2` で MSE 損失を選択します。デフォルトは `l2` で、従来と同様です。
- `huber_schedule` : スケジューリング方法を指定します。`exponential` で指数関数的、`constant` で一定、`snr` で信号対雑音比に基づくスケジューリングを選択します。デフォルトは `snr` です。
- `huber_c` : Huber損失のパラメータを指定します。デフォルトは `0.1` です。

PR 内でいくつかの比較が共有されています。この機能を試す場合、最初は `--loss_type smooth_l1 --huber_schedule snr --huber_c 0.1` などで試してみるとよいかもしれません。

最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

## Additional Information

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). 
<!-- 
LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension. 
-->

### Sample image generation during training
  A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.
