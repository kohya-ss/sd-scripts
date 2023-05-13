This repository contains training, generation, and utility scripts for Stable Diffusion.

The [__Change History__](#change-history) has been moved to the bottom of the page.

For the Japanese version of the README, click [here]([./README-ja.md](https://github.com/kohya-ss/sd-scripts/blob/main/README-ja.md)).

For easier usage, including GUI and PowerShell scripts, please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Special thanks to @bmaltais!

This repository includes scripts for the following:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Textual Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffusion ckpt/safetensors, and Diffusers)

__The Stable Diffusion web UI now appears to support LoRA training with ``sd-scripts``.__ Thank you for the great work!

## About requirements.txt

These files do not include requirements for PyTorch, as the required versions depend on your specific environment. Please install PyTorch first (refer to the installation guide below).

The scripts have been tested with PyTorch 1.12.1 and 1.13.0, as well as Diffusers 0.10.2.

## Links to usage documentation

Most of the documents are written in Japanese.

* [Training guide - common](./docs/train_README-en.md): data preparation, options, etc.
  * [Chinese version](https://github.com/kohya-ss/sd-scripts/blob/main/docs/train_README-ja.md)
* [Dataset config](./docs/config_README-en.md)
* [DreamBooth training guide](./docs/train_db_README-en.md)
* [Step-by-step fine-tuning guide](./docs/fine_tune_README-en.md)
* [LoRA training](./docs/train_network_README-en.md)
* [Textual Inversion training](./docs/train_ti_README-en.md)
* [Image generation](./docs/gen_img_README-en.md)
* [Model conversion](https://note.com/kohya_ss/n/n374f316fe4ad) on note.com

## Windows Required Dependencies

Python 3.10.6 and Git:

- Python 3.10.6: https://www.python.org/ftp/python/3.10.6/python-3.10.6-amd64.exe
- Git: https://git-scm.com/download/win

Grant unrestricted script access to PowerShell so that venv can work:

- Open an administrator PowerShell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close the admin PowerShell window

## Windows Installation

Open a regular PowerShell terminal and enter the following commands:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install --upgrade -r requirements.txt
pip install -U -I --no-deps https://github.com/C43H66N12O12S2/stable-diffusion-webui/releases/download/f/xformers-0.0.14.dev0-cp310-cp310-win_amd64.whl

cp .\bitsandbytes_windows\*.dll .\venv\Lib\site-packages\bitsandbytes\
cp .\bitsandbytes_windows\cextension.py .\venv\Lib\site-packages\bitsandbytes\cextension.py
cp .\bitsandbytes_windows\main.py .\venv\Lib\site-packages\bitsandbytes\cuda_setup\main.py

accelerate config
```

Note: It is recommended to use `python -m venv venv` instead of `python -m venv --system-site-packages venv` to avoid potential issues with global Python packages.

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

Note: Some users have reported encountering a `ValueError: fp16 mixed precision requires a GPU` error during training. In this case, answer `0` for the 6th question: 
``What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]:`` 

(Only the single GPU with id `0` will be used.)

### About PyTorch and xformers

Other versions of PyTorch and xformers may cause problems during training.
If there are no other constraints, please install the specified version.

### Optional: Use Lion8bit

To use Lion8bit, you need to upgrade `bitsandbytes` to version 0.38.0 or later. Uninstall `bitsandbytes`, and for Windows, install the Windows version of the .whl file from [here](https://github.com/jllllll/bitsandbytes-windows-webui) or other sources, like:

```powershell
pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl
```

To upgrade, update this repository with `pip install .`, and upgrade the necessary packages manually.

## Upgrade

When a new release is available, you can upgrade your repository using the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have been executed successfully, you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for the excellent work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo, and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much, KohakuBlueleaf!

## License

The majority of the scripts are licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's, and LoCon). However, portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause

## Change History

### May 11, 2023

- Added an option `--dim_from_weights` to `train_network.py` to automatically determine the dim(rank) from the weight file. [PR #491](https://github.com/kohya-ss/sd-scripts/pull/491) Thanks to AI-Casanova!
  - It is useful in combination with `resize_lora.py`. Please see the PR for details.
- Fixed a bug where the noise resolution was incorrect with Multires noise. [PR #489](https://github.com/kohya-ss/sd-scripts/pull/489) Thanks to sdbds!
  - Please see the PR for details.
- The image generation scripts can now use img2img and highres fix simultaneously.
- Fixed a bug where the hint image of ControlNet was incorrectly BGR instead of RGB in the image generation scripts.
- Added a feature to the image generation scripts to use the memory-efficient VAE.
  - If you specify a number with the `--vae_slices` option, the memory-efficient VAE will be used. The maximum output size will be larger, but it will be slower. Please specify a value of about `16` or `32`.
  - The implementation of the VAE is in `library/slicing_vae.py`.

### May 7, 2023

- The documentation has been moved to the `docs` folder. If you have links, please update them accordingly.
- Removed `gradio` from `requirements.txt`.
- DAdaptAdaGrad, DAdaptAdan, and DAdaptSGD are now supported by DAdaptation. [PR#455](https://github.com/kohya-ss/sd-scripts/pull/455) Thanks to sdbds!
  - DAdaptation needs to be installed. Also, depending on the optimizer, DAdaptation may need to be updated. Please update with `pip install --upgrade dadaptation`.
- Added support for pre-calculation of LoRA weights in image generation scripts. Specify `--network_pre_calc`.
  - The prompt option `--am` is available. Also, it is disabled when Regional LoRA is used.
- Added Adaptive noise scale to each training script. Specify a number with `--adaptive_noise_scale` to enable it.
  - __This is an experimental option. It may be removed or changed in the future.__
  - This is an original implementation that automatically adjusts the value of the noise offset according to the absolute value of the mean of each channel of the latents. It is expected that appropriate noise offsets will be set for bright and dark images, respectively.
  - Specify it together with `--noise_offset`.
  - The actual value of the noise offset is calculated as `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale`. Since the latent is close to a normal distribution, it may be a good idea to specify a value of about 1/10 to the same as the noise offset.
  - Negative values can also be specified, in which case the noise offset will be clipped to 0 or more.
- Other minor fixes.

Please read the [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.

### Naming of LoRA

To avoid confusion, the LoRA supported by `train_network.py` has been assigned specific names. The documentation has been updated accordingly. The following are the names of LoRA types in this repository:

1. __LoRA-LierLa__: (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    This LoRA is applicable to Linear layers and Conv2d layers with a 1x1 kernel.

2. __LoRA-C3Lier__: (LoRA for __C__ onvolutional layers with a __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to the first type, this LoRA is applicable for Conv2d layers with a 3x3 kernel.

LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network argument). LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI or the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with the Web UI, please utilize our extension.

### Sample Image Generation During Training

An example prompt file might look like this:

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy, bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy, bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

Lines starting with `#` are considered comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used:

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

Prompt weightings, such as `( )` and `[ ]`, are functional.

## Generating Sample Images
The prompt file may look like the following:

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy, bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy, bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

Lines starting with `#` are treated as comments. Options can be specified in the format "double hyphen + lowercase letter," such as `--n`. The following options are available:

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

Weightings such as `( )` and `[ ]` also work.
