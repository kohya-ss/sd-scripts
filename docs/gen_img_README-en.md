## This is a ChatGPT-4 English adaptation of the original document by kohya-ss ([gen_img_README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/docs/gen_img_README-ja.md))

This is a Diffusers-based inference (image generation) script compatible with SD 1.x and 2.x models, as well as LoRA and ControlNet (only confirmed to work with v1.0) trained in this repository. It is used from the command line.

# Overview

* Diffusers (v0.10.2) based inference (image generation) script.
* Supports SD 1.x and 2.x (base/v-parameterization) models.
* Supports txt2img, img2img, and inpainting.
* Supports interactive mode, prompt loading from files, and continuous generation.
* Allows specifying the number of images generated per prompt line.
* Allows specifying the total number of iterations.
* Supports not only `fp16` but also `bf16`.
* Supports xformers for fast generation.
    * Although xformers enable memory-efficient generation, the implementation is not as optimized as Automatic 1111's Web UI, and it uses approximately 6GB of VRAM for generating 512x512 images.
* Extension to 225 tokens for prompts. Supports negative prompts and weighting.
* Supports various Diffusers samplers (fewer than Web UI).
* Supports text encoder clip skip (using output from the nth-last layer).
* Supports separate loading of VAE.
* Supports CLIP Guided Stable Diffusion, VGG16 Guided Stable Diffusion, Highres. fix, and upscale.
    * Highres. fix is an independent implementation without fully confirming the Web UI's implementation, so the output may differ.
* Supports LoRA. Supports application rate specification, simultaneous use of multiple LoRAs, and weight merging.
    * It is not possible to specify different application rates for Text Encoder and U-Net.
* Supports Attention Couple.
* Supports ControlNet v1.0.
* Does not allow switching models during execution, but can be handled by creating a batch file.
* Adds various features that were personally desired.

Not all tests are performed when adding new features, so some previous features may be affected and may not work. Please let us know if you have any issues.

# Basic Usage

## Generating Images in Interactive Mode

Please enter the following:

```batchfile
python gen_img_diffusers.py --ckpt <model_name> --outdir <image_output_destination> --xformers --fp16 --interactive
```

Specify the model (Stable Diffusion checkpoint file or Diffusers model folder) with the `--ckpt` option and the image output destination folder with the `--outdir` option.

Specify the use of xformers with the `--xformers` option (remove it if not using xformers). Perform inference in fp16 (single-precision) with the `--fp16` option. Inference in bf16 (bfloat16) can also be performed on RTX 30 series GPUs with the `--bf16` option.

The `--interactive` option specifies interactive mode.

Please add the `--v2` option if using Stable Diffusion 2.0 (or additional training models from it). If using a model with v-parameterization (e.g., `768-v-ema.ckpt` and additional training models from it), also add the `--v_parameterization` option.

If the presence or absence of `--v2` is incorrect, an error will occur when loading the model. If the presence or absence of `--v_parameterization` is incorrect, a brown image will be displayed.

Please enter the prompt when "Type prompt:" is displayed.

![image](https://user-images.githubusercontent.com/52813779/235343115-f3b8ac82-456d-4aab-9724-0cc73c4534aa.png)

*If an error occurs and no image is displayed, headless (no display functionality) OpenCV may be installed. Install regular OpenCV with `pip install opencv-python` or stop displaying images with the `--no_preview` option.

Select the image window and press any key to close the window and enter the next prompt. Press Ctrl+Z followed by Enter in the prompt to close the script.

## Generating Multiple Images with a Single Prompt

Enter the following (on one line):

```batchfile
python gen_img_diffusers.py --ckpt <model_name> --outdir <image_output_destination> 
    --xformers --fp16 --images_per_prompt <number_of_images> --prompt "<prompt>"
```

Specify the number of images generated per prompt with the `--images_per_prompt` option. Specify the prompt with the `--prompt` option. Enclose the prompt in double quotes if it contains spaces.

You can specify the batch size with the `--batch_size` option (described later).

## Batch Generation by Loading Prompts from a File

Enter the following:

```batchfile
python gen_img_diffusers.py --ckpt <model_name> --outdir <image_output_destination> 
    --xformers --fp16 --from_file <prompt_filename>
```

Specify the file containing the prompts with the `--from_file` option. Write one prompt per line. You can specify the number of images generated per prompt line with the `--images_per_prompt` option.

## Using Negative Prompts and Weighting

By writing `--n` in the prompt option (specify within the prompt as `--x`), everything that follows becomes a negative prompt.

Also, like AUTOMATIC1111's Web UI, you can use `()` or `[]`, or `(xxx:1.3)` for weighting (implementation copied from Diffusers' [Long Prompt Weighting Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion)).

You can specify the same options when entering prompts from the command line or loading prompts from a file.

![image](https://user-images.githubusercontent.com/52813779/235343128-e79cd768-ec59-46f5-8395-fce9bdc46208.png)

# Main Options

Specify these options from the command line.

## Model Specification

- `--ckpt <model_name>`: Specifies the model name. The `--ckpt` option is required. You can specify the Stable Diffusion checkpoint file, Diffusers model folder, or Hugging Face model ID.

- `--v2`: Specifies the use of a Stable Diffusion 2.x series model. No specification is needed for 1.x series models.

- `--v_parameterization`: Specifies the use of a model with v-parameterization (`768-v-ema.ckpt` and additional training models from it, Waifu Diffusion v1.5, etc.).
    
    If the presence or absence of `--v2` is incorrect, an error will occur when loading the model. If the presence or absence of `--v_parameterization` is incorrect, a brown image will be displayed.

- `--vae`: Specifies the VAE to use. If not specified, the VAE within the model will be used.

## Image Generation and Output

- `--interactive`: Operates in interactive mode. Generates images when a prompt is entered.

- `--prompt <prompt>`: Specifies the prompt. Enclose the prompt in double quotes if it contains spaces.

- `--from_file <prompt_filename>`: Specifies the file containing the prompts. Write one prompt per line. You can specify the number of images generated per prompt line with the `--images_per_prompt` option.

- `--W <image_width>`: Specifies the image width. The default is `512`.

- `--H <image_height>`: Specifies the image height. The default is `512`.

- `--steps <number_of_steps>`: Specifies the sampling step count. The default is `50`.

- `--scale <guidance_scale>`: Specifies the unconditional guidance scale. The default is `7.5`.

- `--sampler <sampler_name>`: Specifies the sampler. The default is `ddim`. You can specify ddim, pndm, dpmsolver, dpmsolver+++, lms, euler, euler_a, which are provided by Diffusers (you can also specify the last three as k_lms, k_euler, k_euler_a).

- `--outdir <image_output_destination_folder>`: Specifies the destination folder for the images.

- `--images_per_prompt <number_of_images>`: Specifies the number of images generated per prompt. The default is `1`.

- `--clip_skip <skip_count>`: Specifies which layer from the back of CLIP to use. The default is the last layer.

- `--max_embeddings_multiples <multiplier>`: Specifies the multiplier for the default input/output length (75) of CLIP. The default is 75. For example, specifying 3 sets the input/output length to 225.

- `--negative_scale`: Specifies the guidance scale for uncoditioning. This implementation is based on [the article by gcem156](https://note.com/gcem156/n/ne9a53e4a6f43).



## Adjusting Memory Usage and Generation Speed

- `--batch_size <batch_size>`: Specifies the batch size. The default is `1`. A larger batch size consumes more memory but increases generation speed.

- `--vae_batch_size <VAE_batch_size>`: Specifies the VAE batch size. The default is the same as the batch size. VAE consumes more memory, and there may be cases where memory is insufficient after denoising (when the step is 100%). In such cases, reduce the VAE batch size.

- `--xformers`: Specify this option when using xformers.

- `--fp16`: Performs inference in fp16 (single precision). If neither `fp16` nor `bf16` is specified, inference is performed in fp32 (single precision).

- `--bf16`: Performs inference in bf16 (bfloat16). This option can only be specified on RTX 30 series GPUs. The `--bf16` option will result in an error on non-RTX 30 series GPUs. The likelihood of inference results becoming NaN (resulting in a completely black image) seems lower with `bf16` than with `fp16`.

## Using Additional Networks (such as LoRA)

- `--network_module`: Specifies the additional network to use. For LoRA, specify `--network_module networks.lora`. To use multiple LoRAs, specify them like `--network_module networks.lora networks.lora networks.lora`.

- `--network_weights`: Specifies the weight file of the additional network to use. Specify it like `--network_weights model.safetensors`. To use multiple LoRAs, specify them like `--network_weights model1.safetensors model2.safetensors model3.safetensors`. The number of arguments should be the same as the number specified in `--network_module`.

- `--network_mul`: Specifies how many times to multiply the weights of the additional network to use. The default is `1`. Specify it like `--network_mul 0.8`. To use multiple LoRAs, specify them like `--network_mul 0.4 0.5 0.7`. The number of arguments should be the same as the number specified in `--network_module`.

- `--network_merge`: Merges the weights of the additional network to use with the weights specified in `--network_mul` beforehand. This option cannot be used in combination with `--network_pre_calc`. The prompt option `--am` and Regional LoRA will no longer be available, but generation will be accelerated to the same extent as when not using LoRA.

- `--network_pre_calc`: Calculates the weights of the additional network to use beforehand for each generation. The prompt option `--am` can be used. Generation will be accelerated to the same extent as when not using LoRA, but additional time will be required to calculate the weights before generation, and memory usage will also increase slightly. This option is invalidated when using Regional LoRA.

# Examples of Main Option Specifications

The following example generates 64 images in a single prompt with a batch size of 4.

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 64 
    --prompt "beautiful flowers --n monochrome"
```

The following example generates 10 images for each prompt written in a file with a batch size of 4.

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 10 
    --from_file prompts.txt
```

Here's an example using Textual Inversion (explained later) and LoRA.

```batchfile
python gen_img_diffusers.py --ckpt model.safetensors 
    --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --fp16 --sampler k_euler_a 
    --textual_inversion_embeddings goodembed.safetensors negprompt.pt 
    --network_module networks.lora networks.lora 
    --network_weights model1.safetensors model2.safetensors 
    --network_mul 0.4 0.8 
    --clip_skip 2 --max_embeddings_multiples 1 
    --batch_size 8 --images_per_prompt 1 --interactive
```

# Prompt Options

Various options can be specified in the prompt using the format `--n` (two hyphens followed by an alphabet letter). This is valid whether specifying the prompt interactively, via the command line, or from a file.

Please insert a space before and after the prompt option `--n`.

- `--n`: Specifies a negative prompt.

- `--w`: Specifies the image width. This will overwrite the specification from the command line.

- `--h`: Specifies the image height. This will overwrite the specification from the command line.

- `--s`: Specifies the number of steps. This will overwrite the specification from the command line.

- `--d`: Specifies the random seed for this image. If `--images_per_prompt` is specified, please specify multiple seeds separated by commas, like `--d 1,2,3,4`.
    * Due to various reasons, the generated image may differ from the Web UI even with the same random seed.

- `--l`: Specifies the guidance scale. This will overwrite the specification from the command line.

- `--t`: Specifies the strength for img2img (explained later). This will overwrite the specification from the command line.

- `--nl`: Specifies the guidance scale for negative prompts (explained later). This will overwrite the specification from the command line.

- `--am`: Specifies the weights of the additional network. This will overwrite the specification from the command line. To use multiple additional networks, specify them separated by commas, like `--am 0.8,0.5,0.3`.

* When these options are specified, the batch may be executed with a smaller size than the batch size (as items with different values cannot be generated in bulk). (You don't need to worry too much about this, but when generating from a file with prompts, arranging prompts with the same values will improve efficiency.)

Example:
```
(masterpiece, best quality), 1girl, in shirt and plated skirt, standing at street under cherry blossoms, upper body, [from below], kind smile, looking at another, [goodembed] --n realistic, real life, (negprompt), (lowres:1.1), (worst quality:1.2), (low quality:1.1), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry --w 960 --h 640 --s 28 --d 1
```

![image](https://user-images.githubusercontent.com/52813779/235343446-25654172-fff4-4aaf-977a-20d262b51676.png)

# Image-to-Image Conversion

## Options

- `--image_path`: Specifies the image to be used for img2img conversion. Specify it like `--image_path template.png`. If a folder is specified, the images in the folder will be used sequentially.

- `--strength`: Specifies the strength of img2img. Specify it like `--strength 0.8`. The default value is `0.8`.

- `--sequential_file_name`: Specifies whether to use a sequential file name. If specified, the generated file names will be in the format `im_000001.png` and so on.

- `--use_original_file_name`: If specified, the generated file name will be the same as the original file name.

## Example of Execution from the Command Line

```batchfile
python gen_img_diffusers.py --ckpt trinart_characters_it4_v1_vae_merged.ckpt 
    --outdir outputs --xformers --fp16 --scale 12.5 --sampler k_euler --steps 32 
    --image_path template.png --strength 0.8 
    --prompt "1girl, cowboy shot, brown hair, pony tail, brown eyes, 
          sailor school uniform, outdoors 
          --n lowres, bad anatomy, bad hands, error, missing fingers, cropped, 
          worst quality, low quality, normal quality, jpeg artifacts, (blurry), 
          hair ornament, glasses" 
    --batch_size 8 --images_per_prompt 32
```

When specifying a folder for the `--image_path` option, the images in the folder will be loaded sequentially. The number of generated images is not based on the number of images, but on the number of prompts, so please match the number of images for img2img and the number of prompts using the `--images_per_prompt` option.

Files are sorted and loaded by file name. Note that the sort order is in string order (not `1.jpg→2.jpg→10.jpg`, but `1.jpg→10.jpg→2.jpg`), so please adjust accordingly by zero-padding the file names (e.g., `01.jpg→02.jpg→10.jpg`).

## Using img2img for Upscaling

When specifying the generated image size with the `--W` and `--H` command-line options during img2img, the original image will be resized to that size before performing img2img.

Additionally, if the source image for img2img is an image generated by this script, and the prompt is omitted, the prompt will be automatically retrieved from the metadata of the source image, allowing only the 2nd stage of Highres. fix to be performed.

## Inpainting during img2img

You can perform inpainting by specifying an image and a mask image (note that this does not support inpainting models and simply performs img2img only on the masked area).

The options are as follows:

- `--mask_image`: Specifies the mask image. Like `--img_path`, if you specify a folder, the images in that folder will be used sequentially.

The mask image is a grayscale image with the white parts being inpainted. It is recommended to use a gradient at the boundary to make the inpainted area look smoother.

![image](https://user-images.githubusercontent.com/52813779/235343795-9eaa6d98-02ff-4f32-b089-80d1fc482453.png)

# Other Functions

## Textual Inversion

Specify the embeddings to be used with the `--textual_inversion_embeddings` option (multiple can be specified). By using the file name without the extension in the prompt, those embeddings will be used (same usage as in the Web UI). This can also be used within negative prompts.

As models, you can use the Textual Inversion models trained in this repository and the Textual Inversion models trained in the Web UI (image embedding is not supported).

## Extended Textual Inversion

Please specify the `--XTI_embeddings` option instead of the `--textual_inversion_embeddings` option. The usage is the same as with `--textual_inversion_embeddings`.

## Highres. Fix

This is a similar feature to the one in AUTOMATIC1111's Web UI (it may differ in various aspects due to being a custom implementation). A smaller image is generated initially, and then img2img is performed on that image to prevent inconsistencies in the entire image while generating a higher resolution image.

The number of steps for the 2nd stage is calculated from the values of the `--steps` and `--strength` options (`steps*strength`).

img2img cannot be combined with this feature.

The following options are available:

- `--highres_fix_scale`: Enables Highres. fix and specifies the size of the image generated in the 1st stage as a ratio. If the final output is 1024x1024 and a 512x512 image is generated initially, specify it as `--highres_fix_scale 0.5`. Please note that this is the reciprocal of the value used in the Web UI.

- `--highres_fix_steps`: Specifies the number of steps for the 1st stage image. The default is `28`.

- `--highres_fix_save_1st`: Specifies whether to save the 1st stage image.

- `--highres_fix_latents_upscaling`: If specified, the 1st stage image is upscaled at the latent level during the 2nd stage image generation (only bilinear is supported). If not specified, LANCZOS4 upscaling is used.

- `--highres_fix_upscaler`: Specifies an arbitrary upscaler for the 2nd stage. Currently, only `--highres_fix_upscaler tools.latent_upscaler` is supported.

- `--highres_fix_upscaler_args`: Specifies the arguments to be passed to the upscaler specified with `--highres_fix_upscaler`.
    In the case of `tools.latent_upscaler`, specify the weight file like `--highres_fix_upscaler_args "weights=D:\Work\SD\Models\others\etc\upscaler-v1-e100-220.safetensors"`.

Example of a command line:

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 1024 --H 1024 --batch_size 1 --outdir ../txt2img 
    --steps 48 --sampler ddim --fp16 
    --xformers 
    --images_per_prompt 1  --interactive 
    --highres_fix_scale 0.5 --highres_fix_steps 28 --strength 0.5
```

## ControlNet

Currently, only ControlNet 1.0 has been tested. Only Canny preprocessing is supported.

The following options are available:

- `--control_net_models`: Specifies the ControlNet model file.
    Multiple models can be specified, and they will be switched between steps (different implementation from the ControlNet extension in the Web UI). Both diff and regular models are supported.

- `--guide_image_path`: Specifies the guide image to be used for ControlNet. Like `--img_path`, if you specify a folder, the images in that folder will be used sequentially. If you are using a model other than Canny, please preprocess the image beforehand.

- `--control_net_preps`: Specifies the preprocessing for ControlNet. Multiple can be specified, like `--control_net_models`. Currently, only Canny is supported. If preprocessing is not used for the target model, specify `none`.
   For Canny, specify the thresholds 1 and 2 separated by '_' like `--control_net_preps canny_63_191`.

- `--control_net_weights`: Specifies the weights when applying ControlNet (normal with `1.0`, half the influence with `0.5`). Multiple can be specified, like `--control_net_models`.

- `--control_net_ratios`: Specifies the range of steps where ControlNet is applied. If `0.5` is set, ControlNet will be applied up to half of the total number of steps. Multiple can be specified, like `--control_net_models`.

Example of a command line:

```batchfile
python gen_img_diffusers.py --ckpt model_ckpt --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --bf16 --sampler k_euler_a 
    --control_net_models diff_control_sd15_canny.safetensors --control_net_weights 1.0 
    --guide_image_path guide.png --control_net_ratios 1.0 --interactive
```

## Attention Couple + Regional LoRA

This feature allows you to divide the prompt into several parts and specify which area of the image to apply each prompt to. There are no individual options, but `mask_path` and the prompt are used for specification.

First, use `AND` in the prompt to define multiple parts. The first three parts can be assigned to specific regions, while the remaining parts will be applied to the entire image. Negative prompts will be applied to the entire image.

In the following example, three parts are defined with AND.

The following text describes various options and functionalities for generating images using Diffusers and CLIP:

Two girls are smiling and looking at the viewer while another two girls are looking back. The image quality is not good, in fact, it's the worst quality.

Next, prepare a mask image. The mask image is a color image, with each RGB channel corresponding to a separated part of the prompt by AND. If the value of a channel is all 0, it will be applied to the entire image.

In the example above, the R channel corresponds to "shs 2girls, looking at viewer, smile", the G channel to "bsb 2girls, looking back", and the B channel to "2girls". By using a mask image like the one below, since there is no specification in the B channel, "2girls" will be applied to the entire image.

![image](https://user-images.githubusercontent.com/52813779/235343061-b4dc9392-3dae-4831-8347-1e9ae5054251.png)

The mask image can be specified using `--mask_path`. Currently, only one image is supported. The specified image size will be automatically resized and applied.

It is also possible to combine this with ControlNet (recommended for fine-grained position control).

When specifying LoRA, multiple LoRA specified by `--network_weights` will correspond to each part of the AND. As a current constraint, the number of LoRA must be the same as the number of AND parts.

## CLIP Guided Stable Diffusion

This is a modified version of the custom pipeline from the Community Examples of Diffusers [here](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#clip-guided-stable-diffusion).

In addition to the regular prompt-based generation, it retrieves the text features from a larger CLIP model and controls the generated image to make its features closer to the text features. This increases VRAM usage considerably (may be difficult for 512x512 with 8GB VRAM) and takes longer to generate.

Only DDIM, PNDM, and LMS samplers can be selected.

You can specify the degree to which the CLIP features are reflected with the `--clip_guidance_scale` option. Starting from around 100 and adjusting up or down seems to work well.

By default, the first 75 tokens of the prompt (excluding special characters for weighting) are passed to CLIP. You can use the `--c` option in the prompt to specify a separate text for CLIP instead of the regular prompt (for example, CLIP may not recognize DreamBooth's identifiers or model-specific words like "1girl").

Here's an example command line:

```batchfile
python gen_img_diffusers.py  --ckpt v1-5-pruned-emaonly.ckpt --n_iter 1 
    --scale 2.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img --steps 36  
    --sampler ddim --fp16 --opt_channels_last --xformers --images_per_prompt 1  
    --interactive --clip_guidance_scale 100
```

## CLIP Image Guided Stable Diffusion

This feature allows you to control the generation by passing another image to CLIP instead of text, making the generated image's features closer to the guide image. Specify the application amount with the `--clip_image_guidance_scale` option and the guide image (file or folder) with the `--guide_image_path` option.

Here's an example command line:

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img 
    --steps 80 --sampler ddim --fp16 --opt_channels_last --xformers 
    --images_per_prompt 1  --interactive  --clip_image_guidance_scale 100 
    --guide_image_path YUKA160113420I9A4104_TP_V.jpg
```

### VGG16 Guided Stable Diffusion

This feature allows you to generate images that are closer to the specified image. In addition to the regular prompt-based generation, it retrieves the features from VGG16 and controls the generated image to make it closer to the specified guide image. This is recommended for use with img2img (as the generated image may be blurry with a regular generation). It is an original feature that utilizes the mechanism of CLIP Guided Stable Diffusion and the idea is borrowed from style transfer using VGG.

Only DDIM, PNDM, and LMS samplers can be selected.

You can specify the degree to which the VGG16 features are reflected with the `--vgg16_guidance_scale` option. Starting from around 100 and adjusting up or down seems to work well. Specify the guide image (file or folder) with the `--guide_image_path` option.

To convert multiple images in bulk with img2img and use the original image as the guide image, specify the same value for `--guide_image_path` and `--image_path`.

Here's an example command line:

```batchfile
python gen_img_diffusers.py --ckpt wd-v1-3-full-pruned-half.ckpt 
    --n_iter 1 --scale 5.5 --steps 60 --outdir ../txt2img 
    --xformers --sampler ddim --fp16 --W 512 --H 704 
    --batch_size 1 --images_per_prompt 1 
    --prompt "picturesque, 1girl, solo, anime face, skirt, beautiful face 
        --n lowres, bad anatomy, bad hands, error, missing fingers, 
        cropped, worst quality, low quality, normal quality, 
        jpeg artifacts, blurry, 3d, bad face, monochrome --d 1" 
    --strength 0.8 --image_path ..\src_image
    --vgg16_guidance_scale 100 --guide_image_path ..\src_image 
```

You can specify the VGG16 layer number to be used for feature extraction with `--vgg16_guidance_layer`. The default is 20 for the ReLU of conv4-2. It is said that the higher layers represent the style, while the lower layers represent the content.

![image](https://user-images.githubusercontent.com/52813779/235343813-3c1f0d7a-4fb3-4274-98e4-b92d76b551df.png)

# Other Options

- `--no_preview`: Do not display the preview image in interactive mode. Specify this if OpenCV is not installed or if you want to check the output file directly.

- `--n_iter`: Specify the number of times to repeat the generation. The default is 1. Specify this when you want to perform multiple generations while reading prompts from a file.

- `--tokenizer_cache_dir`: Specify the cache directory for the tokenizer. (Work in progress)

- `--seed`: Specify the random seed. This seed will be used for the image when generating one image and as the seed for generating the seeds for each image when generating multiple images.

- `--iter_same_seed`: If the prompt does not specify a random seed, use the same seed for all iterations within `--n_iter`. Use this when you want to unify seeds for comparison between multiple prompts specified with `--from_file`.

- `--diffusers_xformers`: Use Diffuser's xformers.

- `--opt_channels_last`: Place the tensor channels last during inference. This may result in faster performance in some cases.

- `--network_show_meta`: Display the metadata for the additional network.

