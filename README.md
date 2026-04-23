# sd-scripts

[English](./README.md) / [日本語](./README-ja.md)

## Table of Contents
<details>
<summary>Click to expand</summary>

- [Introduction](#introduction)
    - [Supported Models](#supported-models)
    - [Features](#features)
    - [Sponsors](#sponsors)
    - [Support the Project](#support-the-project)
- [Documentation](#documentation)
    - [Training Documentation (English and Japanese)](#training-documentation-english-and-japanese)
    - [Other Documentation (English and Japanese)](#other-documentation-english-and-japanese)
- [For Developers Using AI Coding Agents](#for-developers-using-ai-coding-agents)
- [Windows Installation](#windows-installation)
    - [Windows Required Dependencies](#windows-required-dependencies)
    - [Installation Steps](#installation-steps)
    - [About requirements.txt and PyTorch](#about-requirementstxt-and-pytorch)
    - [xformers installation (optional)](#xformers-installation-optional)
- [Linux/WSL2 Installation](#linuxwsl2-installation)
    - [DeepSpeed installation (experimental, Linux or WSL2 only)](#deepspeed-installation-experimental-linux-or-wsl2-only)
- [Upgrade](#upgrade)
    - [Upgrade PyTorch](#upgrade-pytorch)
- [Credits](#credits)
- [License](#license)

</details>

## Introduction

This repository contains training, generation and utility scripts for Stable Diffusion and other image generation models.

### Sponsors

We are grateful to the following companies for their generous sponsorship:

<a href="https://aihub.co.jp/top-en">
  <img src="./images/logo_aihub.png" alt="AiHUB Inc." title="AiHUB Inc." height="100px">
</a>

### Support the Project

If you find this project helpful, please consider supporting its development via [GitHub Sponsors](https://github.com/sponsors/kohya-ss/). Your support is greatly appreciated!

### Change History

- **Changes planned for the next release:** The following are the main changes planned for the next release. Please note that these changes may be subject to change without notice before the release.
    - Added OFTv2 and BOFT network modules (`networks.oft_v2`, `networks.boft`) for SD1.x / SD2.x / SDXL training. [PR #2357](https://github.com/kohya-ss/sd-scripts/pull/2357)
        - Orthogonal fine-tuning adapters following the PEFT implementation. Weights in PEFT format can also be loaded. Thanks to umisetokikaze.
        - Note that `--network_dim` means the block size for these modules. For details, please refer to the [documentation](./docs/train_network_oft_boft.md).
    - Added per-subset timestep sampling offset (`custom_attributes.timestep_sampling.offset`) for FLUX.1 and Anima LoRA training. [PR #2401](https://github.com/kohya-ss/sd-scripts/pull/2401) Thanks to okdsf.
        - Shifts the timestep sampling distribution of each dataset subset toward lower- or higher-noise timesteps. For details, please refer to the [documentation](./docs/timestep_sampling_offset.md).
    - Added `--show_timesteps_offset` to preview the timestep distribution with the offset applied when using `--show_timesteps`. [PR #2410](https://github.com/kohya-ss/sd-scripts/pull/2410)
        - The documentation also describes how the offset behaves with `shift` / `flux_shift` timestep sampling.

- **Version 0.11.1 (2026-06-16):**
    - Added support for torch.compile in Anima LoRA/LLLite training. [PR #2379](https://github.com/kohya-ss/sd-scripts/pull/2379)
        - It seems to speed up training by about 20%. It requires Triton and MSVC compiler. For details, please refer to the [documentation](./docs/anima_torch_compile.md).
    - Added 2D-only Qwen-Image VAE. [PR #2382](https://github.com/kohya-ss/sd-scripts/pull/2382)
        - Based on the suggestion by woct0rdho in [issue #2369](https://github.com/kohya-ss/sd-scripts/issues/2369). Thanks to woct0rdho.
        - Enabled by specifying `--qwen_image_vae_2d`. The weights are the same as the standard (3D) version.
        - Expected to speed up latent pre-caching (training itself remains unchanged). For details, please refer to the [documentation](./docs/anima_train_network.md#memory-and-speed--メモリ速度関連).
    - Added support for LLLite inpainting model training. [PR #2378](https://github.com/kohya-ss/sd-scripts/pull/2378)
        - For details, please refer to the [documentation](./docs/anima_train_control_net_lllite.md).
    - Added logging of timestep sampling settings and visualization of timesteps distribution. [PR #2384](https://github.com/kohya-ss/sd-scripts/pull/2384)
        - Visualization makes it easier to understand how training is conducted at different timesteps.
        - For details, please refer to the [documentation](./docs/anima_train_network.md#visualizing-the-timestep-distribution).

- **Version 0.11.0 (2026-06-12):**
    - A major internal refactoring of the codebase has been performed to improve code quality and maintainability. [PR #2372](https://github.com/kohya-ss/sd-scripts/pull/2372)
        - We have made efforts to minimize direct impact on users. For details and bug reports, please refer to [this discussion](https://github.com/kohya-ss/sd-scripts/discussions/2358).

- **Version 0.10.6 (2026-06-12):**
    - Stable version before refactoring merge.

- **Version 0.10.5 (2026-05-08):**
    - Support for transformers version 5 and later has been added. Thanks to marcus165090-spec for [PR #2315](https://github.com/kohya-ss/sd-scripts/pull/2315) (followed by [PR #2316](https://github.com/kohya-ss/sd-scripts/pull/2316)).
        - The `transformers` version in `requirements.txt` remains 4.x, but it also works with 5.x. If you use 5.x for any reason, please also update `diffusers` to the latest version.
    - Support for ControlNet-LLLite training for Anima has been added. Thanks to [PR #2317](https://github.com/kohya-ss/sd-scripts/pull/2317).
        - For details, please refer to the [documentation](./docs/anima_train_control_net_lllite.md).

- **Version 0.10.4 (2026-05-07):**
    - Improved compatibility with Intel GPUs. Thanks to WhitePr for [PR #2307](https://github.com/kohya-ss/sd-scripts/pull/2307).
    - Support for training inpainting models for SD 1.5/SDXL has been added. Thanks to allanoepping for [PR #2309](https://github.com/kohya-ss/sd-scripts/pull/2309) (followed by [PR #2318](https://github.com/kohya-ss/sd-scripts/pull/2318)).
        - For details, please refer to the [documentation](./docs/inpainting_training.md).

### Supported Models

* **Stable Diffusion 1.x/2.x**
* **SDXL**
* **SD3/SD3.5**
* **FLUX.1**
* **LUMINA**
* **HunyuanImage-2.1**
* **Anima**

### Features

* LoRA training
* Fine-tuning (native training, DreamBooth): except for HunyuanImage-2.1
* Textual Inversion training: SD/SDXL
* Inpainting model training: SD1.5 and SDXL
* Image generation
* Other utilities such as model conversion, image tagging, LoRA merging, etc.

## Documentation

### Training Documentation (English and Japanese)

* [LoRA Training Overview](./docs/train_network.md)
* [Dataset config](./docs/config_README-en.md) / [Japanese version](./docs/config_README-ja.md)
* [Advanced Training](./docs/train_network_advanced.md)
* [OFTv2 / BOFT Training](./docs/train_network_oft_boft.md)
* [SDXL Training](./docs/sdxl_train_network.md)
* [SD3 Training](./docs/sd3_train_network.md)
* [FLUX.1 Training](./docs/flux_train_network.md)
* [LUMINA Training](./docs/lumina_train_network.md)
* [HunyuanImage-2.1 Training](./docs/hunyuan_image_train_network.md)
* [Fine-tuning](./docs/fine_tune.md)
* [Textual Inversion Training](./docs/train_textual_inversion.md)
* [ControlNet-LLLite Training](./docs/train_lllite_README.md) / [Japanese version](./docs/train_lllite_README-ja.md)
* [Anima ControlNet-LLLite Training Guide](./docs/anima_train_control_net_lllite.md)
* [Validation](./docs/validation.md)
* [Masked Loss Training](./docs/masked_loss_README.md) / [Japanese version](./docs/masked_loss_README-ja.md)
* [Inpainting Training](./docs/inpainting_training.md)

### Other Documentation (English and Japanese)

* [Image generation](./docs/gen_img_README.md) / [Japanese version](./docs/gen_img_README-ja.md)
* [Tagging images with WD14 Tagger](./docs/wd14_tagger_README-en.md) / [Japanese version](./docs/wd14_tagger_README-ja.md)

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

This approach ensures that you have full control over the instructions given to your agent while benefiting from the shared project context. Your `CLAUDE.md` and `GEMINI.md` are already listed in `.gitignore`, so they won't be committed to the repository.

## Windows Installation

### Windows Required Dependencies

Python 3.10.x and Git:

- Python 3.10.x: Download Windows installer (64-bit) from https://www.python.org/downloads/windows/
- git: Download latest installer from https://git-scm.com/download/win

Python 3.11.x, and 3.12.x will work but not tested.

Give unrestricted script access to powershell so venv can work:

- Open an administrator powershell window
- Type `Set-ExecutionPolicy Unrestricted` and answer A
- Close admin powershell window

### Installation Steps

Open a regular Powershell terminal and type the following inside:

```powershell
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts

python -m venv venv
.\venv\Scripts\activate

pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install --upgrade -r requirements.txt

accelerate config
```

If `python -m venv` shows only `python`, change `python` to `py`.

Note: `bitsandbytes`, `prodigyopt` and `lion-pytorch` are included in the requirements.txt. If you'd like to use another version, please install it manually.

This installation is for CUDA 12.4. If you use a different version of CUDA, please install the appropriate version of PyTorch. For example, if you use CUDA 12.1, please install `pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121`.

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

## About requirements.txt and PyTorch

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with PyTorch 2.6.0. PyTorch 2.6.0 or later is required.

For RTX 50 series GPUs, PyTorch 2.8.0 with CUDA 12.8/12.9 should be used. `requirements.txt` will work with this version.

### Installing without OpenCV (optional)

`opencv-python` is listed in `requirements.txt` as a default dependency, but the core training / dataset pipeline only uses a small subset of OpenCV (mainly `cv2.resize`, `cv2.cvtColor`, and a debug-only `cv2.imshow`). If you would rather avoid the large OpenCV install, use `requirements-no-opencv.txt` instead:

```bash
pip install --upgrade -r requirements-no-opencv.txt
```

When `opencv-python` is not available, a lightweight Pillow/NumPy fallback under `library/_cv2_stub` is automatically registered as `cv2`, so existing scripts continue to work. Note that:

- The default install (`requirements.txt`) keeps OpenCV, which remains the recommended path — `cv2.INTER_AREA` in particular produces better results for downscaling than the Pillow fallback.
- The following tools still require real `opencv-python` and will exit with a clear message when it is missing: `tools/canny.py`, `tools/detect_face_rotate.py`, and the ControlNet `canny` preprocessor used by `gen_img.py` / `sdxl_gen_img.py`.
- Debug-only features such as `cv2.imshow` during dataset inspection become no-ops without OpenCV.

### xformers installation (optional)

To install xformers, run the following command in your activated virtual environment:

```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu124
```

Please change the CUDA version in the URL according to your environment if necessary. xformers may not be available for some GPU architectures.

## Linux/WSL2 Installation

Linux or WSL2 installation steps are almost the same as Windows. Just change `venv\Scripts\activate` to `source venv/bin/activate`.

Note: Please make sure that NVIDIA driver and CUDA toolkit are installed in advance.

### DeepSpeed installation (experimental, Linux or WSL2 only)
  
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

If you want to upgrade PyTorch, you can upgrade it with `pip install` command in [Windows Installation](#windows-installation) section.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause
