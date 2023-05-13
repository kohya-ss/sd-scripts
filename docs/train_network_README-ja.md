## This is a ChatGPT-4 English adaptation of the original document by kohya-ss ([train_network_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_network_README-ja.md))

# Learning LoRA

This is an implementation of [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) (arxiv) and [LoRA](https://github.com/microsoft/LoRA) (github) for Stable Diffusion.

We greatly appreciate the work of [cloneofsimo's repository](https://github.com/cloneofsimo/lora). Thank you.

While conventional LoRA is applied only to Linear and Conv2d with a kernel size of 1x1, it can also be extended to Conv2d with a kernel size of 3x3.

The expansion to Conv2d 3x3 was first released by [cloneofsimo](https://github.com/cloneofsimo/lora), and its effectiveness was demonstrated by KohakuBlueleaf in [LoCon](https://github.com/KohakuBlueleaf/LoCon). We deeply appreciate KohakuBlueleaf's work.

It seems to work just fine with 8GB VRAM.

Please also refer to the [common documentation on learning](./train_README-en.md).

# Types of LoRA that can be learned

We support the following two types, which have unique names within this repository:

1. __LoRA-LierLa__: LoRA applied to Linear and Conv2d with a kernel size of 1x1.

2. __LoRA-C3Lier__: LoRA applied to Linear, Conv2d with a kernel size of 1x1, and Conv2d with a kernel size of 3x3.

Compared to LoRA-LierLa, LoRA-C3Liar may offer higher accuracy due to the increased number of layers it is applied to.

During training, you can also use __DyLoRA__ (described later).

## Notes on trained models

LoRA-LierLa can be used with AUTOMATIC1111's Web UI LoRA feature.

To generate with LoRA-C3Liar in Web UI, use this [WebUI extension for additional networks](https://github.com/kohya-ss/sd-webui-additional-networks).

Both LoRA models can also be merged with the Stable Diffusion model using a script within this repository.

There is currently no compatibility with cloneofsimo's repository or d8ahazard's [Dreambooth Extension for Stable-Diffusion-WebUI](https://github.com/d8ahazard/sd_dreambooth_extension) due to some feature extensions (described later).

# Learning Procedure

Please refer to this repository's README and prepare the environment accordingly.

## Data Preparation

Refer to [preparing learning data](./train_README-en.md).

## Execute Learning

Use `train_network.py`.

Specify the target module name for the `--network_module` option in `train_network.py`. Since `network.lora` is compatible with LoRA, please specify that.

It seems that specifying a higher learning rate than the usual DreamBooth or fine-tuning, such as `1e-4` to `1e-3`, works well.

Here is an example command line:

```
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=<.ckpt, .safetensors, or directory of Diffusers version model> 
    --dataset_config=<.toml file created during data preparation> 
    --output_dir=<output folder for trained model>  
    --output_name=<filename for output of trained model> 
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

This command line will train LoRA-LierLa.

The LoRA model will be saved in the folder specified by the `--output_dir` option. For other options, optimizers, etc., please refer to the "Commonly Used Options" section in the [common learning documentation](./train_README-en.md).

Other available options include:

* `--network_dim`: Specifies the RANK of LoRA (`--networkdim=4`, etc.). The default is 4. A higher number increases expressiveness but requires more memory and time for learning. It is not always better to increase this number blindly.
* `--network_alpha`: Specifies the `alpha` value for stable learning and preventing underflow. The default is 1. Specifying the same value as `network_dim` results in the same behavior as previous versions.
* `--persistent_data_loader_workers`: Significantly reduces waiting time between epochs on Windows environments.
* `--max_data_loader_n_workers`: Specifies the number of processes for data loading. More processes result in faster data loading and more efficient GPU usage but consume main memory. By default, it is set to the smaller of `8` or `number of concurrent CPU threads - 1`. If you have limited main memory or GPU usage is above 90%, consider reducing this value to `2` or `1`, based on your hardware.
* `--network_weights`: Loads the weights of a pre-trained LoRA and trains from there.
* `--network_train_unet_only`: Enables only LoRA modules related to U-Net. This may be useful for fine-tuning-type learning.
* `--network_train_text_encoder_only`: Enables only LoRA modules related to Text Encoder. This may have a Textual Inversion-like effect.
* `--unet_lr`: Specifies a different learning rate for U-Net-related LoRA modules than the standard learning rate specified by the `--learning_rate` option.
* `--text_encoder_lr`: Specifies a different learning rate for Text Encoder-related LoRA modules than the standard learning rate specified by the `--learning_rate` option. There is some discussion that it may be better to use a slightly lower learning rate (e.g., 5e-5) for the Text Encoder.
* `--network_args`: Allows specifying multiple arguments. Described later.

If both `--network_train_unet_only` and `--network_train_text_encoder_only` are unspecified (default), both Text Encoder and U-Net LoRA modules are enabled.

# Other Learning Methods

## Learning LoRA-C3Lier

Specify `--network_args` like this: `conv_dim` for the rank of Conv2d (3x3) and `conv_alpha` for alpha.

```
--network_args "conv_dim=4" "conv_alpha=1"
```

If alpha is omitted, it defaults to 1.

```
--network_args "conv_dim=4"
```

## DyLoRA

DyLoRA is proposed in this paper: [DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation](https://arxiv.org/abs/2210.07558). The official implementation is available [here](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA).

According to the paper, LoRA's rank is not always better when it is higher. Instead, it is necessary to find the appropriate rank depending on the target model, dataset, task, etc. Using DyLoRA, you can simultaneously train LoRA with various ranks up to the specified dim(rank). This eliminates the need to search for the optimal rank separately for each.

Our implementation is based on the official implementation with some custom extensions (which may have bugs).

### Features of DyLoRA in this repository

DyLoRA's model files are compatible with LoRA. Additionally, LoRA models with multiple dims(ranks) up to the specified dim can be extracted from DyLoRA's model files.

Both DyLoRA-LierLa and DyLoRA-C3Lier can be learned.

### Learning with DyLoRA

Specify `network.dylora` for the `--network_module` option, which is compatible with DyLoRA.

Also, specify `unit` in `--network_args`, such as `--network_args "unit=4"`. `unit` is the unit for dividing rank. For example, specify `--network_dim=16 --network_args "unit=4"`. Please make sure that `unit` is a divisor of `network_dim` (i.e., `network_dim` is a multiple of `unit`).

If `unit` is not specified, it is treated as `unit=1`.

For example, when learning with dim=16 and unit=4 (described later), you can extract and learn four LoRA models with ranks 4, 8, 12, and 16. By generating images with each extracted model and comparing them, you can select the LoRA with the optimal rank.

Other options are the same as for regular LoRA.

*Note: `unit` is a custom extension of this repository. In DyLoRA, the learning time is expected to be longer than that of regular LoRA with the same dim(rank), so the split unit has been increased.

### Extracting LoRA models from DyLoRA models

Use `extract_lora_from_dylora.py` in the `networks` folder. This script extracts LoRA models from DyLoRA models at the specified `unit` intervals.

The command line looks like this:

```powershell
python networks\extract_lora_from_dylora.py --model "foldername/dylora-model.safetensors" --save_to "foldername/dylora-model-split.safetensors" --unit 4
```

Specify the DyLoRA model file in `--model`. Specify the file name to save the extracted model in `--save_to` (the rank value will be added to the file name). Specify the `unit` used during DyLoRA's learning in `--unit`.

## Hierarchical Learning Rate

For details, please refer to [PR #355](https://github.com/kohya-ss/sd-scripts/pull/355).

You can specify the weights of the 25 blocks of the full model. There is no LoRA corresponding to the first block, but it is set to 25 for compatibility with hierarchical LoRA applications. Also, in some blocks, LoRA does not exist if not extended to conv2d3x3, but please always specify 25 values to unify the description.

Specify the following arguments in `--network_args`.

- `down_lr_weight`: Specify the learning rate weights of the down blocks of U-Net. You can specify the following:
  - Weight for each block: Specify 12 values like `"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`.
  - Specification from preset: Specify like `"down_lr_weight=sine"` (weights are specified with a sine curve). You can specify sine, cosine, linear, reverse_linear, or zeros. Additionally, if you add `+number` like `"down_lr_weight=cosine+.25"`, the specified value will be added (it will become 0.25~1.25).
- `mid_lr_weight`: Specify the learning rate weight of the mid block of U-Net. Specify only one number like `"down_lr_weight=0.5"`.
- `up_lr_weight`: Specify the learning rate weights of the up blocks of U-Net. It is the same as down_lr_weight.
- Any unspecified parts will be treated as 1.0. Also, if the weight is set to 0, the LoRA module for that block will not be created.
- `block_lr_zero_threshold`: If the weight is less than or equal to this value, the LoRA module will not be created. The default is 0.

### Example of hierarchical learning rate command-line specification:

```powershell
--network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"

--network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"
```

### Example of hierarchical learning rate toml file specification:

```toml
network_args = [ "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5", "mid_lr_weight=2.0", "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5",]

network_args = [ "block_lr_zero_threshold=0.1", "down_lr_weight=sine+.5", "mid_lr_weight=1.5", "up_lr_weight=cosine+.5", ]
```

## Hierarchical dim (rank)

You can specify the dim (rank) of the 25 blocks of the full model. Like the hierarchical learning rate, LoRA may not exist in some blocks, but always specify 25 values.

Specify the following arguments in `--network_args`.

- `block_dims`: Specify the dim (rank) of each block. Specify 25 values like `"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`.
- `block_alphas`: Specify the alpha of each block. Specify 25 values like block_dims. If omitted, the value of network_alpha will be used.
- `conv_block_dims`: Extend LoRA to Conv2d 3x3 and specify the dim (rank) of each block.
- `conv_block_alphas`: Specify the alpha of each block when LoRA is extended to Conv2d 3x3. If omitted, the value of conv_alpha will be used.

### Example of hierarchical dim (rank) command-line specification:

```powershell
--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"
```

### Example of hierarchical dim (rank) toml file specification:

```toml
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2",]
  
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2", "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",]
```

# Other scripts

These are a set of scripts related to merging and LoRA.

## About the merge script

With merge_lora.py, you can merge the learning results of LoRA into the Stable Diffusion model or merge multiple LoRA models.

### Merging LoRA model into Stable Diffusion model

The merged model can be treated like a normal Stable Diffusion ckpt. For example, the command line will be as follows:

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors --ratios 0.8
```

If you are training with a Stable Diffusion v2.x model and want to merge it, please specify the --v2 option.

Specify the Stable Diffusion model file to be merged in the --sd_model option (.ckpt or .safetensors only, Diffusers are not currently supported).

Specify the destination to save the merged model in the --save_to option (.ckpt or .safetensors, automatically determined by the extension).

Specify the learned LoRA model file in --models. Multiple specifications are also possible, in which case they will be merged in order.

Specify the application rate (how much weight to reflect in the original model) for each model in --ratios with a value between 0 and 1.0. For example, if it seems close to overlearning, reducing the application rate may make it better. Please specify the same number as the number of models.

When specifying multiple models, it will be as follows:

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.8 0.5
```

### Merging Multiple LoRA Models

There will be subtle differences in the results due to the order of calculations when applying multiple LoRA models one by one to the SD model compared to merging multiple LoRA models and then applying them to the SD model.

For example, the command line would look like this:

```
python networks\merge_lora.py
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.6 0.4
```

The --sd_model option is not required.

Specify the save destination for the merged LoRA model with the --save_to option (automatically determined by .ckpt or .safetensors extension).

Specify the trained LoRA model files with the --models option. You can specify three or more models.

Specify the ratio of each model (how much weight to reflect in the original model) with the --ratios option as a numeric value between 0 and 1.0. If you want to merge two models one-to-one, use "0.5 0.5". With "1.0 1.0", the total weight will be too high, and the result will likely be undesirable.

LoRA models trained with v1 and v2, and those with different ranks (dimensions) or ``alpha`` values cannot be merged. LoRA models with only U-Net and those with U-Net + Text Encoder should be mergeable, but the results are unknown.

### Other Options

* precision
  * Specify the precision during merge calculation from float, fp16, and bf16. If omitted, it defaults to float to ensure accuracy. If you want to reduce memory usage, please specify fp16/bf16.
* save_precision
  * Specify the precision when saving the model from float, fp16, and bf16. If omitted, the precision will be the same as the precision option.

## Merging Multiple LoRA Models with Different Ranks

Approximate multiple LoRA models with a single LoRA model (an exact reproduction is not possible). Use `svd_merge_lora.py`. For example, the command line would look like this:

```
python networks\svd_merge_lora.py
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors
    --ratios 0.6 0.4 --new_rank 32 --device cuda
```

The main options are the same as `merge_lora.py`. The following options have been added:

- `--new_rank`
  - Specify the rank of the created LoRA model.
- `--new_conv_rank`
  - Specify the rank of the created Conv2d 3x3 LoRA model. If omitted, it defaults to the same value as `new_rank`.
- `--device`
  - Specify cuda with `--device cuda` to perform calculations on the GPU. This will speed up the process.

## Generate Images Using the Scripts in This Repository

Add the --network_module and --network_weights options to gen_img_diffusers.py. Their meanings are the same as during training.

Specify a numerical value between 0 and 1.0 with the --network_mul option to change the application rate of LoRA.

## Generate Images Using the Diffusers Pipeline

Refer to the example below. The only required file is networks/lora.py. The script may not work with Diffusers versions other than 0.10.2.

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
network3, sd = create_network_from_weights(0.5, None, vae, text_encoder, unet, sd)
network3.apply_to(text_encoder, unet)
network3.load_state_dict(sd)
network3.to(device, dtype=torch.float16)

# prompts
prompt = "masterpiece, best quality, 1girl, in white shirt, looking at viewer"
negative_prompt = "bad quality, worst quality, bad anatomy, bad hands"

# execute pipeline
print("generating image...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]

# if not merged, you can use set_multiplier
# network1.set_multiplier(0.8)
# and generate image again...

# save image
image.save(r"by_diffusers..png")
```

## Creating a LoRA Model from the Difference of Two Models

This implementation is based on [this discussion](https://github.com/cloneofsimo/lora/discussions/56). The equations are used as is (I don't fully understand them, but it seems that singular value decomposition is used for approximation).

The difference between two models (e.g., the base model and the fine-tuned model) is approximated using LoRA.

### How to Run the Script

Specify as follows:
```
python networks\extract_lora_from_models.py --model_org base-model.ckpt
    --model_tuned fine-tuned-model.ckpt 
    --save_to lora-weights.safetensors --dim 4
```

Specify the original Stable Diffusion model with the `--model_org` option. When applying the created LoRA model, specify this model. Both `.ckpt` and `.safetensors` formats are accepted.

Specify the Stable Diffusion model to extract the difference with the `--model_tuned` option. For example, specify a model after fine-tuning or DreamBooth. Both `.ckpt` and `.safetensors` formats are accepted.

Specify the destination for saving the LoRA model with `--save_to`, and specify the dimension of LoRA with `--dim`.

The generated LoRA model can be used in the same way as a trained LoRA model.

If the Text Encoder is the same in both models, the resulting LoRA will be a LoRA for U-Net only.

### Other Options

- `--v2`
  - Specify if you want to use the v2.x Stable Diffusion model.
- `--device`
  - Specify `--device cuda` to perform calculations on the GPU. This speeds up processing (it's not too slow on the CPU, but it's only about twice to several times faster).
- `--save_precision`
  - Specify the LoRA save format from "float", "fp16", or "bf16". Default is "float".
- `--conv_dim`
  - Specify to expand the range of LoRA application to Conv2d 3x3. Specify the rank of Conv2d 3x3.

## Image Resize Script

(Documentation will be organized later, but for now, here's an explanation.)

With the extension of Aspect Ratio Bucketing, it is now possible to use small images as they are without enlarging them as training data. We received a preprocessing script along with a report that adding resized images of the original training images improved accuracy, so we've added and refined it. Thanks to bmaltais.

### How to Run the Script

Specify as follows. Both the original image and the resized image are saved in the destination folder. The resized image has the resolution, such as `+512x512`, added to the file name (which is different from the image size). Images smaller than the destination resolution will not be enlarged.

```
python tools\resize_images_to_resolution.py --max_resolution 512x512,384x384,256x256 --save_as_png 
    --copy_associated_files source_image_folder destination_folder
```

Images in the source_image_folder are resized so that their area is the same as the specified resolution and saved in the destination_folder. Files other than images are copied as is.

Specify the destination size in the `--max_resolution` option as shown in the example. The images will be resized so that their area is the same as the specified size. If you specify multiple resolutions, the images will be resized for each resolution. If you specify `512x512,384x384,256x256`, the destination folder will contain four images: the original size and three resized sizes.

Specify the `--save_as_png` option to save the images in PNG format. If omitted, the images will be saved in JPEG format (quality=100).

When the `--copy_associated_files` option is specified, files with the same name as the image (excluding the extension, such as captions) are copied with the same name as the resized image file.

### Other Options

- divisible_by
  - The image is cropped from the center so that the size (both height and width) of the resized image is divisible by this value.
- interpolation
  - Specify the interpolation method for downsampling. Choose from `area, cubic, lanczos4`. The default is `area`.

## Differences from Cloneofsimo's Repository

As of December 25, 2022, this repository has expanded the application of LoRA to Text Encoder's MLP, U-Net's FFN, and Transformer's in/out projection, increasing its expressiveness. However, this has increased memory usage, making it barely fit within 8GB.

The module replacement mechanisms are also completely different.

## Future Extensions

In addition to LoRA, other extensions can also be supported, and they will be added in the future.
