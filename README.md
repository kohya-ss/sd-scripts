This repository contains training, generation and utility scripts for Stable Diffusion.

## FLUX.1 and SD3 training (WIP)

This feature is experimental. The options and the training script may change in the future. Please let us know if you have any idea to improve the training.

__Please update PyTorch to 2.6.0 or later. We have tested with `torch==2.6.0` and `torchvision==0.21.0` with CUDA 12.4. `requirements.txt` is also updated, so please update the requirements.__

The command to install PyTorch is as follows:
`pip3 install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124`

For RTX 50 series GPUs, PyTorch 2.8.0 with CUDA 12.8/9 should be used. `requirements.txt` will work with this version.

If you are using DeepSpeed, please install DeepSpeed with `pip install deepspeed` (appropriate version is not confirmed yet).

### Recent Updates

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

### Mar 21, 2025 /  2025-03-21 Version 0.9.1

- Fixed a bug where some of LoRA modules for CLIP Text Encoder were not trained. Thank you Nekotekina for PR [#1964](https://github.com/kohya-ss/sd-scripts/pull/1964)
  - The LoRA modules for CLIP Text Encoder are now 264 modules, which is the same as before. Only 88 modules were trained in the previous version. 

### Jan 17, 2025 /  2025-01-17 Version 0.9.0

- __important__ The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
  - bitsandbytes, transformers, accelerate and huggingface_hub are updated. 
  - If you encounter any issues, please report them.

- The dev branch is merged into main. The documentation is delayed, and I apologize for that. I will gradually improve it.
- The state just before the merge is released as Version 0.8.8, so please use it if you encounter any issues.
- The following changes are included.

#### Changes

- Fixed a bug where the loss weight was incorrect when `--debiased_estimation_loss` was specified with `--v_parameterization`. PR [#1715](https://github.com/kohya-ss/sd-scripts/pull/1715) Thanks to catboxanon! See [the PR](https://github.com/kohya-ss/sd-scripts/pull/1715) for details.
  - Removed the warning when `--v_parameterization` is specified in SDXL and SD1.5. PR [#1717](https://github.com/kohya-ss/sd-scripts/pull/1717)

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

#### 変更点

- devブランチがmainにマージされました。ドキュメントの整備が遅れており申し訳ありません。少しずつ整備していきます。
- マージ直前の状態が Version 0.8.8 としてリリースされていますので、問題があればそちらをご利用ください。
- 以下の変更が含まれます。

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


### Oct 27, 2024 / 2024-10-27:

- `svd_merge_lora.py` VRAM usage has been reduced. However, main memory usage will increase (32GB is sufficient).
- This will be included in the next release.
- `svd_merge_lora.py` のVRAM使用量を削減しました。ただし、メインメモリの使用量は増加します（32GBあれば十分です）。
- これは次回リリースに含まれます。

### Oct 26, 2024 / 2024-10-26: 

- Fixed a bug in `svd_merge_lora.py`, `sdxl_merge_lora.py`, and `resize_lora.py` where the hash value of LoRA metadata was not correctly calculated when the `save_precision` was different from the  `precision` used in the calculation. See issue [#1722](https://github.com/kohya-ss/sd-scripts/pull/1722) for details. Thanks to JujoHotaru for raising the issue.
- It will be included in the next release.

- `svd_merge_lora.py`、`sdxl_merge_lora.py`、`resize_lora.py`で、保存時の精度が計算時の精度と異なる場合、LoRAメタデータのハッシュ値が正しく計算されない不具合を修正しました。詳細は issue [#1722](https://github.com/kohya-ss/sd-scripts/pull/1722) をご覧ください。問題提起していただいた JujoHotaru 氏に感謝します。
- 以上は次回リリースに含まれます。

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

  * `--n` Negative prompt up to the next option. Ignored when CFG scale is `1.0`.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image. For FLUX.1 models, the default is `1.0`, which means no CFG. For Chroma models, set to around `4.0` to enable CFG.
  * `--g` Specifies the embedded guidance scale for the models with embedded guidance (FLUX.1), the default is `3.5`. Set to `0.0` for Chroma models.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.
