This repository contains training, generation and utility scripts for Stable Diffusion.

## FLUX.1 and SD3 training (WIP)

This feature is experimental. The options and the training script may change in the future. Please let us know if you have any idea to improve the training.

__Please update PyTorch to 2.6.0 or later. We have tested with `torch==2.6.0` and `torchvision==0.21.0` with CUDA 12.4. `requirements.txt` is also updated, so please update the requirements.__

The command to install PyTorch is as follows:
`pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`

For RTX 50 series GPUs, PyTorch 2.8.0 with CUDA 12.8/9 should be used. `requirements.txt` will work with this version.

If you are using DeepSpeed, please install DeepSpeed with `pip install deepspeed` (appropriate version is not confirmed yet).

### Recent Updates

Sep 23, 2025:
- HunyuanImage-2.1 LoRA training is supported. [PR #2198](https://github.com/kohya-ss/sd-scripts/pull/2198) for details.
  - Please see [HunyuanImage-2.1 Training](./docs/hunyuan_image_train_network.md) for details.
  - __HunyuanImage-2.1 training does not support LoRA modules for Text Encoders, so `--network_train_unet_only` is required.__
  - The training script is `hunyuan_image_train_network.py`.
  - This includes changes to `train_network.py`, the base of the training script. Please let us know if you encounter any issues.

Sep 13, 2025:
- The loading speed of `.safetensors` files has been improved for SD3, FLUX.1 and Lumina. See [PR #2200](https://github.com/kohya-ss/sd-scripts/pull/2200) for more details.
    - Model loading can be up to 1.5 times faster.
    - This is a wide-ranging update, so there may be bugs. Please let us know if you encounter any issues.

Sep 4, 2025:
- The information about FLUX.1 and SD3/SD3.5 training that was described in the README has been organized and divided into the following documents:
    - [LoRA Training Overview](./docs/train_network.md)
    - [SDXL Training](./docs/sdxl_train_network.md)
    - [Advanced Training](./docs/train_network_advanced.md)
    - [FLUX.1 Training](./docs/flux_train_network.md)
    - [SD3 Training](./docs/sd3_train_network.md)
    - [LUMINA Training](./docs/lumina_train_network.md)
    - [Validation](./docs/validation.md)
    - [Fine-tuning](./docs/fine_tune.md)
    - [Textual Inversion Training](./docs/train_textual_inversion.md)

Aug 28, 2025:
- In order to support the latest GPUs and features, we have updated the **PyTorch and library versions**. PR [#2178](https://github.com/kohya-ss/sd-scripts/pull/2178) There are many changes, so please let us know if you encounter any issues.
- The PyTorch version used for testing has been updated to 2.6.0. We have confirmed that it works with PyTorch 2.6.0 and later.
- The `requirements.txt` has been updated, so please update your dependencies.
    - You can update the dependencies with `pip install -r requirements.txt`.
    - The version specification for `bitsandbytes` has been removed. If you encounter errors on RTX 50 series GPUs, please update it with `pip install -U bitsandbytes`.
- We have modified each script to minimize warnings as much as possible.
    - The modified scripts will work in the old environment (library versions), but please update them when convenient.


## For Developers Using AI Coding Agents

This repository provides recommended instructions to help AI agents like Claude and Gemini understand our project context and coding standards.

To use them, you need to opt-in by creating your own configuration file in the project root.

**Quick Setup:**

1.  Create a `CLAUDE.md` and/or `GEMINI.md` file in the project root.
2.  Add the following line to your `CLAUDE.md` to import the repository's recommended prompt:

    ```markdown
    @./.ai/claude.prompt.md
    ```

    or for Gemini:

    ```markdown
    @./.ai/gemini.prompt.md
    ```

3.  You can now add your own personal instructions below the import line (e.g., `Always respond in Japanese.`).

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so it won't be committed to the repository.

--- 

[__Change History__](#change-history) is moved to the bottom of the page. 
更新履歴は[ページ末尾](#change-history)に移しました。

Latest update: 2025-03-21 (Version 0.9.1)

[日本語版READMEはこちら](./README-ja.md)

The development version is in the `dev` branch. Please check the dev branch for the latest changes.

FLUX.1 and SD3/SD3.5 support is done in the `sd3` branch. If you want to train them, please use the sd3 branch.


For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Textual Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

### Sponsors

We are grateful to the following companies for their generous sponsorship:

<a href="https://aihub.co.jp/top-en">
  <img src="./images/logo_aihub.png" alt="AiHUB Inc." title="AiHUB Inc." height="100px">
</a>

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!


## About requirements.txt

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with Pytorch 2.1.2. PyTorch 2.2 or later will work. Please install the appropriate version of PyTorch and xformers.

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

Python 3.10.x, 3.11.x, and 3.12.x will work but not tested.

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

Note: Now `bitsandbytes==0.44.0`, `prodigyopt==1.0` and `lion-pytorch==0.0.6` are included in the requirements.txt. If you'd like to use the another version, please install it manually.

This installation is for CUDA 11.8. If you use a different version of CUDA, please install the appropriate version of PyTorch and xformers. For example, if you use CUDA 12, please install `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` and `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121`.

If you use PyTorch 2.2 or later, please change `torch==2.1.2` and `torchvision==0.16.2` and `xformers==0.0.23.post1` to the appropriate version.

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

## DeepSpeed installation (experimental, Linux or WSL2 only)
  
To install DeepSpeed, run the following command in your activated virtual environment:

```bash
pip install deepspeed==0.16.7 
```

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

### Version 0.10.1 (2026-01-18)

- `sd3` branch is merged to `main` branch. From this version, FLUX.1 and SD3/SD3.5 are supported in the `main` branch.

## Additional Information

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). 
