# Script for Mutual Conversion of Stable Diffusion Checkpoint and Diffusers Model (Compatible with SD2.0)

## Introduction
While Diffusers officially provides conversion scripts for Stable Diffusion v1.x, as of 12/3, they have not yet released one compatible with 2.0. Since I have already implemented model conversion itself in my DreamBooth training script, I added code to it to create a script that performs mutual conversion.
I made it compatible with both v1.x and v2.0.

## Downloading the Script
Please download from here. The .zip contains two scripts, so place them in the same folder.

*12/10 (v4) update: Added support for safetensors format of Diffusers model. Requires DiffUsers 0.10.2 (works with 0.10.0 or later, but 0.10.0 seems to have issues, so use 0.10.2). Update within the virtual environment with `pip install -U diffusers[torch]==0.10.2`.

*12/5 (v3) update: Made a tentative fix as there were reports of errors when saving in safetensors format for some models. If you get an error with v2, please try v3 (other specifications are the same). I will delete v2 once I have thoroughly tested it.

*12/5 (v2) update: Added support for safetensors format. Install safetensors with `pip install safetensors`.

## Usage
### Converting from Diffusers to Stable Diffusion .ckpt/.safetensors
Specify the source model folder and destination .ckpt file as follows (actually written on one line). The v1/v2 version is automatically detected.

```
python convert_diffusers20_original_sd.py ..\models\diffusers_model 
    ..\models\sd.ckpt
```

Note that the Text Encoder of Diffusers v2 only has 22 layers, so if you convert it directly to Stable Diffusion, the weights will be insufficient. Therefore, the weights of the 22nd layer are copied as the 23rd layer to add weights. The weights of the 23rd layer are not used during image generation, so there is no impact. Similarly, dummy weights are added to text_projection and logit_scale (they do not seem to be used for image generation).

If you use the .safetensors extension, it will automatically save in safetensors format.

### Converting from Stable Diffusion .ckpt/.safetensors to Diffusers  
Enter as follows.

```
python convert_diffusers20_original_sd.py ..\models\sd.ckpt 
    ..\models\diffusers_model 
    --v2 --reference_model stabilityai/stable-diffusion-2
```

Specify the .ckpt file (or .safetensors file) and output destination folder as arguments (the read format is automatically determined by the extension).
Automatic model detection is not possible, so use the `--v1` or `--v2` option according to the model.

Also, since .ckpt does not contain scheduler and tokenizer information, it is necessary to copy that information from some existing Diffusers model. Specify it with `--reference_model`. You can specify a HuggingFace id or a local model directory.

If you don't have a local model, for v2, specifying `"stabilityai/stable-diffusion-2"` or `"stabilityai/stable-diffusion-2-base"` should work well.
For v1.4/1.5, `"CompVis/stable-diffusion-v1-4"` should be fine (v1.4 and v1.5 seem to be the same).

Use the `--use_safetensors` option to save the Diffusers model in safetensors format.

## Other Options
### --fp16 / --bf16 / --float
You can specify the data format when saving the checkpoint. Only `--fp16` is also valid when loading the Diffusers model.

### --epoch / --global_step 
When saving the checkpoint, the specified values are written for epoch and global_step. If not specified, both will be 0.

## Conclusion
I think some people may be having trouble with the Diffusers model due to the poor inference environment. I hope this helps even a little.

(It is also possible to convert the data format from checkpoint to checkpoint, although untested.)
