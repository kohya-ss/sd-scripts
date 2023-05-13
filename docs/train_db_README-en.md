## <p align="center">Developed as a Guide to Comprehend the Fine-Tuning of Stable Diffusion Models</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/darkstorm2150/OpenGen/main/OpenGen%20Logo-768.jpg" alt="ALT_TEXT" height="256">
</p>

<h3><p align="center">This is a ChatGPT-4 English adaptation of the original document by kohya-ss (train_db_README-ja.md).</p></h3>

Introducing the DreamBooth Guide.

Please also see the [common document for learning](./train_README.md).

# Overview

DreamBooth is a technology that adds specific themes to image generation models through additional learning and generates them with specific identifiers. [View the paper here](https://arxiv.org/abs/2208.12242).

Specifically, it teaches the Stable Diffusion model characters and art styles, and allows them to be called with specific words like `shs` (to appear in the generated image).

The script is based on [Diffusers' DreamBooth](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth), but has added features like the following (some of which have been implemented in the original script as well).

The main features of the script are as follows:

- Memory-saving with 8-bit Adam optimizer and caching latents (similar to [Shivam Shrirao's version](https://github.com/ShivamShrirao/diffusers/tree/main/examples/dreambooth)).
- Memory-saving with xformers.
- Learning in sizes other than 512x512.
- Quality improvement with augmentation.
- Support for fine-tuning DreamBooth and Text Encoder + U-Net.
- Reading and writing models in Stable Diffusion format.
- Aspect Ratio Bucketing.
- Support for Stable Diffusion v2.0.

# Learning Procedure

Please refer to this repository's README for environment preparation.

## Data Preparation

Please refer to [Preparing Learning Data](./train_README.md).

## Running the Learning

Execute the script. The command for maximum memory saving is as follows (actually entered in one line). Please modify each line as necessary. It seems to work with about 12GB of VRAM.

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckpt or .safetensord or Diffusers model directory> 
    --dataset_config=<.toml file created in data preparation> 
    --output_dir=<output folder for the learned model>  
    --output_name=<file name when outputting the learned model> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=1600 
    --learning_rate=1e-6 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
```

It is usually best to specify 1 for `num_cpu_threads_per_process`.

Specify the base model for additional learning in `pretrained_model_name_or_path`. You can specify a Stable Diffusion checkpoint file (.ckpt or .safetensors), a Diffusers local disk model directory, or a Diffusers model ID (such as "stabilityai/stable-diffusion-2").

Specify the folder to save the learned model in `output_dir`. Specify the model's file name without the extension in `output_name`. Specify saving in safetensors format with `save_model_as`.

Specify the `.toml` file in `dataset_config`. For the initial batch size specification in the file, set it to `1` to keep memory consumption low.

`prior_loss_weight` is the weight of the regularization image loss. Normally, specify 1.0.

Set the number of training steps, `max_train_steps`, to 1600. The learning rate `learning_rate` is specified as 1e-6.

Specify `mixed_precision="fp16"` for memory saving (you can also specify `bf16` for RTX 30 series and later. Match the settings made in accelerate during environment preparation). Also, specify `gradient_checkpointing`.

To use the memory-efficient 8-bit AdamW optimizer, specify `optimizer_type="AdamW8bit"`.

Specify the `xformers` option and use xformers' CrossAttention. If you have not installed xformers or encounter an error (depending on the environment, such as when `mixed_precision="no"`), you can specify the `mem_eff_attn` option instead to use the memory-efficient CrossAttention (which will be slower).

Cache the VAE output for memory saving by specifying the `cache_latents` option.

If you have enough memory, edit the `.toml` file to increase the batch size to, for example, `4` (which may potentially speed up and improve accuracy). Additionally, removing `cache_latents` enables augmentation.

### Commonly Used Options

Please refer to the [common document for learning](./train_README.md) for the following cases:

- Learning Stable Diffusion 2.x or derived models
- Learning models that assume a clip skip of 2 or more
- Learning with captions exceeding 75 tokens

### Step Count in DreamBooth

In this script, for memory saving, the number of learning times per step is half that of the original script (because the target image and regularization image are divided into separate batches for learning).

To perform almost the same learning as the original Diffusers version and XavierXiao's Stable Diffusion version, double the number of steps.

(Strictly speaking, the order of the data changes because the learning image and regularization image are shuffled together, but it is thought to have little effect on learning.)

### Batch Size in DreamBooth

As the whole model is learned (similar to fine-tuning), memory consumption is higher compared to learning using LoRA and other methods.

### Learning Rate

The Diffusers version is 5e-6, but the Stable Diffusion version is 1e-6, so the sample above specifies 1e-6.

### Command Line for Specifying Dataset in Previous Format

Specify resolution and batch size as options. An example of the command line is as follows:

```
accelerate launch --num_cpu_threads_per_process 1 train_db.py 
    --pretrained_model_name_or_path=<.ckpt or .safetensors or Diffusers version model directory>
    --train_data_dir=<training data directory>
    --reg_data_dir=<regularization images directory>
    --output_dir=<trained model output directory>
    --output_name=<trained model output filename>
    --prior_loss_weight=1.0
    --resolution=512
    --train_batch_size=1
    --learning_rate=1e-6
    --max_train_steps=1600
    --use_8bit_adam
    --xformers
    --mixed_precision="bf16"
    --cache_latents
    --gradient_checkpointing
```

## Generating images with the trained model

Once the training is complete, a safetensors file will be output in the specified folder with the specified name.

For v1.4/1.5 and other derivative models, you can infer with this model using Automatic1111's WebUI. Please place it in the models\Stable-diffusion folder.

To generate images with the v2.x model in the WebUI, a separate .yaml file describing the model specifications is required. For the v2.x base, place the v2-inference.yaml in the same folder, and for the 768/v, place the v2-inference-v.yaml in the folder, and name the part before the extension the same as the model.

![image](https://user-images.githubusercontent.com/52813779/210776915-061d79c3-6582-42c2-8884-8b91d2f07313.png)

Each yaml file can be found in the [Stability AI's SD2.0 repository](https://github.com/Stability-AI/stablediffusion/tree/main/configs/stable-diffusion).

# Other major options specific to DreamBooth

Please refer to the separate document for all options.

## Do not train the Text Encoder from the middle --stop_text_encoder_training

By specifying a number for the stop_text_encoder_training option, the Text Encoder training will not be performed after that step, and only the U-Net will be trained. In some cases, this may lead to improved accuracy.

(It is suspected that the Text Encoder alone may overfit first, and this option may help prevent that, but the exact impact is unknown.)

## Do not pad the Tokenizer output --no_token_padding

By specifying the no_token_padding option, the output of the Tokenizer will not be padded (this is the same behavior as the old DreamBooth of the Diffusers version).
