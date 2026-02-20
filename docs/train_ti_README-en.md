## This is a ChatGPT-4 English adaptation of the original document by kohya-ss ([train_ti_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_ti_README-ja.md))

This is an explanation about learning Textual Inversion (https://textual-inversion.github.io/).

Please also refer to the [common documentation on learning](./train_README-en.md).

The implementation was greatly inspired by https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion.

The learned model can be used directly in the Web UI.

# Learning procedure

Please refer to this repository's README beforehand and set up the environment.

## Data preparation

Refer to [Preparing Training Data](./train_README-en.md) for more information.

## Executing the training

Use `train_textual_inversion.py`. The following is an example of a command-line (DreamBooth method).

```
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py 
    --dataset_config=<.toml file created during data preparation> 
    --output_dir=<output folder for the trained model>  
    --output_name=<file name for the trained model output without extension> 
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

Specify the token string during training with `--token_string`. __Make sure your training prompt includes this string (e.g., if the token_string is mychar4, use "mychar4 1girl")__. This part of the prompt will be replaced with a new token for Textual Inversion and learned. For DreamBooth and class+identifier-style datasets, it is easiest and most reliable to make the `token_string` the token string.

You can check whether the token string is included in the prompt by using `--debug_dataset`. The replaced token id will be displayed, so you can check if there are tokens after `49408`, as shown below.

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

You cannot use words that the tokenizer already has (common words).

Specify the string of the source token for initializing embeddings with `--init_word`. It is better to choose something close to the concept you want to learn. You cannot specify a string that consists of two or more tokens.

Specify how many tokens to use in this training with `--num_vectors_per_token`. The more tokens you use, the more expressive the model will be, but the more tokens will be consumed. For example, if num_vectors_per_token=8, the specified token string will consume 8 tokens (out of the general prompt's 77-token limit).

These are the main options for Textual Inversion. The rest is similar to other training scripts.

Usually, it is better to specify `1` for `num_cpu_threads_per_process`.

Specify the base model for additional learning with `pretrained_model_name_or_path`. You can specify a Stable Diffusion checkpoint file (.ckpt or .safetensors), a Diffusers model directory on your local disk, or a Diffusers model ID (e.g., "stabilityai/stable-diffusion-2").

Specify the folder to save the trained model after learning with `output_dir`. Specify the model's filename without the extension in `output_name`. Specify saving the model in safetensors format with `save_model_as`.

Specify the `.toml` file in `dataset_config`. Set the batch size in the file to `1` initially to keep memory consumption low.

Set the number of training steps to 10000 with `max_train_steps`. Set the learning rate to 5e-6 with `learning_rate`.

To save memory, specify `mixed_precision="fp16"` (for RTX 30 series and later, you can also specify `bf16`. Match the setting you made in accelerate when setting up the environment). Also, specify `gradient_checkpointing`.

To use a low-memory consumption 8bit AdamW optimizer, specify `optimizer_type="AdamW8bit"`.

Specify the `xformers` option to use xformers' CrossAttention. If you have not installed xformers or if it causes errors (depending on the environment, such as when `mixed_precision="no"`), you can alternatively specify the `mem_eff_attn` option to use the memory-efficient CrossAttention (although it will be slower).

If you have enough memory, edit the `.toml` file to increase the batch size to, for example, `8` (this may speed up and potentially improve accuracy).

### Commonly used options

Please refer to the documentation on options in the following cases:

- Training a Stable Diffusion 2.x or derived model
- Training a model with a clip skip of 2 or more
- Training with captions exceeding 75 tokens

### Batch size for Textual Inversion

Compared to DreamBooth and fine-tuning, which train the entire model, Textual Inversion uses less memory, so you can set a larger batch size.

# Other main options for Textual Inversion

Please refer to another document for all options.

* `--weights`
  * Load pre-trained embeddings before training and learn further from them.
* `--use_object_template`
  * Learn with a default object template string (e.g., "a photo of a {}") instead of captions. This will be the same as the official implementation. Captions will be ignored.
* `--use_style_template`
  * Learn with a default style template string (e.g., "a painting in the style of {}") instead of captions. This will be the same as the official implementation. Captions will be ignored.

## Generating images with the script in this repository

Specify the learned embeddings file with the `--textual_inversion_embeddings` option in gen_img_diffusers.py (multiple files allowed). Use the filename (without the extension) of the embeddings file in the prompt, and the embeddings will be applied.
