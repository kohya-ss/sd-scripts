## This is a ChatGPT-4 English adaptation of the original document by kohya-ss ([fine_tune_README_ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/fine_tune_README_ja.md))

This is a fine-tuning method proposed by NovelAI, which is compatible with their learning approach, automatic captioning, tagging, and a Windows + VRAM 12GB (for SD v1.x) environment. Fine-tuning in this context refers to training the model using images and captions (LoRA, Textual Inversion, and Hypernetworks are not included).

Please also refer to the [common document on training](./train_README-en.md).

# Overview

We will perform fine-tuning of the U-Net in Stable Diffusion using Diffusers. We have implemented the following improvements proposed by NovelAI's article (regarding Aspect Ratio Bucketing, we referred to NovelAI's code, but the final code is entirely original):

* Use the second-to-last layer's output of the CLIP (Text Encoder) instead of the last layer's output.
* Train with non-square resolutions (Aspect Ratio Bucketing).
* Extend the token length from 75 to 225.
* Perform automatic captioning with BLIP and automatic tagging with DeepDanbooru or WD14Tagger.
* Support Hypernetwork training.
* Compatible with Stable Diffusion v2.0 (base and 768/v).
* Reduce memory usage and speed up training by pre-fetching VAE outputs and saving them to disk.

By default, training for the Text Encoder is not performed. In general, it seems that only the U-Net is trained when fine-tuning the entire model (this is also the case with NovelAI). The Text Encoder can be included in the training with an optional setting.

# Additional Features

## Changing the CLIP output

The CLIP (Text Encoder) is responsible for converting text features to reflect the prompt in the image. Stable Diffusion uses the output of the last layer of CLIP, but this can be changed to use the output of the second-to-last layer instead. According to NovelAI, this results in a more accurate reflection of the prompt. It is also possible to use the original last layer output.

*Note: In Stable Diffusion 2.0, the second-to-last layer is used by default. Do not specify the clip_skip option.

## Training with non-square resolutions

In addition to the 512x512 resolution used in Stable Diffusion, we also train with resolutions such as 256x1024 and 384x640. This reduces the amount of cropping and is expected to improve the learning of the relationship between prompts and images. The training resolution is adjusted in 64-pixel increments vertically and horizontally, within the range that does not exceed the area (i.e., memory usage) of the specified resolution.

In machine learning, it is common to unify input sizes across all inputs, but there is no specific constraint. In practice, it is sufficient to have a consistent image size within a single batch. NovelAI's bucketing seems to refer to pre-classifying training data by aspect ratio and learning resolution. Then, by creating batches with images from each bucket, the image size within the batch is unified.

## Extension of token length from 75 to 225

In Stable Diffusion, the maximum token length is 75 (77 tokens, including the start and end tokens), but this is extended to 225 tokens. However, since the maximum length accepted by CLIP is 75 tokens, in the case of 225 tokens, the input is simply divided into three parts, and CLIP is called for each part, then the results are concatenated.

*Note: It is not entirely clear whether this implementation is desirable. It seems to be working for now. There is no reference implementation for 2.0, so it has been implemented independently.

*Note: In Automatic1111's Web UI, it seems that some additional processing, such as splitting based on commas, is performed. In my case, the implementation is simpler and only involves basic splitting.

# Training Procedure

Please refer to the README of this repository and set up your environment beforehand.

## Data Preparation

Please refer to the [instructions for preparing training data](./train_README-en.md). Fine-tuning only supports the metadata-based fine-tuning method.

## Executing the Training

For example, run the following command. Modify each line according to your needs.

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=<.ckpt or .safetensord or Diffusers version model directory> 
    --output_dir=<output folder for trained model>  
    --output_name=<output file name for trained model without extension> 
    --dataset_config=<.toml file created during data preparation> 
    --save_model_as=safetensors 
    --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=fp16
```

It is generally a good idea to specify `1` for `num_cpu_threads_per_process`.

Specify the base model for additional training in `pretrained_model_name_or_path`. You can specify the Stable Diffusion checkpoint file (.ckpt or .safetensors), the Diffusers local disk model directory, or the Diffusers model ID (e.g., "stabilityai/stable-diffusion-2").

Specify the folder to save the trained model in `output_dir`. Specify the model file name without the extension in `output_name`. Save the model in safetensors format by specifying `save_model_as`.

Specify the `.toml` file in `dataset_config`. To begin with, set the batch size specified in the file to `1` to minimize memory consumption.

Set the number of training steps to `10000` with `max_train_steps`. In this example, a learning_rate of `5e-6` is specified.

Enable mixed precision with `mixed_precision="fp16"` to reduce memory usage (for RTX 30 series and later, you can also specify `bf16`. Match the settings with the accelerate configuration made during environment setup). Also, enable `gradient_checkpointing`.

For more information on commonly used options, refer to the separate documentation.

In summary, this fine-tuning approach is compatible with NovelAI's learning method, automatic captioning, tagging, and a Windows + VRAM 12GB environment. It includes several improvements, such as using the second-to-last layer of the CLIP, training with non-square resolutions, and extending the token length to 225. The training procedure involves preparing the data, executing the training, and using additional features such as training the Text Encoder.
