This repository contains training, generation and utility scripts for Stable Diffusion.

[__Change History__](#change-history) is moved to the bottom of the page. 
更新履歴は[ページ末尾](#change-history)に移しました。

[日本語版READMEはこちら](./README-ja.md)

For easier use (GUI and PowerShell scripts etc...), please visit [the repository maintained by bmaltais](https://github.com/bmaltais/kohya_ss). Thanks to @bmaltais!

This repository contains the scripts for:

* DreamBooth training, including U-Net and Text Encoder
* Fine-tuning (native training), including U-Net and Text Encoder
* LoRA training
* Textual Inversion training
* Image generation
* Model conversion (supports 1.x and 2.x, Stable Diffision ckpt/safetensors and Diffusers)

## About requirements.txt

The file does not contain requirements for PyTorch. Because the version of PyTorch depends on the environment, it is not included in the file. Please install PyTorch first according to the environment. See installation instructions below.

The scripts are tested with Pytorch 2.1.2. 2.0.1 and 1.12.1 is not tested but should work.

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

__Note:__ Now `bitsandbytes==0.43.0`, `prodigyopt==1.0` and `lion-pytorch==0.0.6` are included in the requirements.txt. If you'd like to use the another version, please install it manually.

This installation is for CUDA 11.8. If you use a different version of CUDA, please install the appropriate version of PyTorch and xformers. For example, if you use CUDA 12, please install `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121` and `pip install xformers==0.0.23.post1 --index-url https://download.pytorch.org/whl/cu121`.

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

## Upgrade

When a new release comes out you can upgrade your repo with the following command:

```powershell
cd sd-scripts
git pull
.\venv\Scripts\activate
pip install --use-pep517 --upgrade -r requirements.txt
```

Once the commands have completed successfully you should be ready to use the new version.

## Credits

The implementation for LoRA is based on [cloneofsimo's repo](https://github.com/cloneofsimo/lora). Thank you for great work!

The LoRA expansion to Conv2d 3x3 was initially released by cloneofsimo and its effectiveness was demonstrated at [LoCon](https://github.com/KohakuBlueleaf/LoCon) by KohakuBlueleaf. Thank you so much KohakuBlueleaf!

## License

The majority of scripts is licensed under ASL 2.0 (including codes from Diffusers, cloneofsimo's and LoCon), however portions of the project are available under separate license terms:

[Memory Efficient Attention Pytorch](https://github.com/lucidrains/memory-efficient-attention-pytorch): MIT

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes): MIT

[BLIP](https://github.com/salesforce/BLIP): BSD-3-Clause


## Change History

### Mar XX, 2024 / 2024/3/XX: v0.8.6


- The dependent libraries are updated. Please see [Upgrade](#upgrade) and update the libraries.
  - Especially `imagesize` is newly added, so if you cannot update immediately, please install with `pip install imagesize==1.4.1`.
  - `bitsandbytes==0.43.0`, `prodigyopt==1.0`, `lion-pytorch==0.0.6` are included in the requirements.txt.
- Colab seems to stop with log output. Try specifying `--console_log_simple` option in the training script to disable rich logging.
- The `.toml` file for the dataset config is now read in UTF-8 encoding. PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Thanks to Horizon1704!
- Fixed a bug that the last subset settings are applied to all images when multiple subsets of regularization images are specified in the dataset settings. The settings for each subset are correctly applied to each image. PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) Thanks to feffy380!
- `train_network.py` and `sdxl_train_network.py` are modified to record some dataset settings in the metadata of the trained model (`caption_prefix`, `caption_suffix`, `keep_tokens_separator`, `secondary_separator`, `enable_wildcard`).
- DeepSpeed is supported. PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101)  and [#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) Thanks to BootsofLagrangian! See PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) for details.
- The masked loss is supported in each training script. PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) See [Masked loss](#masked-loss) for details.
- Some features are added to the dataset subset settings.
  - `secondary_separator` is added to specify the tag separator that is not the target of shuffling or dropping. 
    - Specify `secondary_separator=";;;"`. When you specify `secondary_separator`, the part is not shuffled or dropped. 
  - `enable_wildcard` is added. When set to `true`, the wildcard notation `{aaa|bbb|ccc}` can be used. The multi-line caption is also enabled.
  - `keep_tokens_separator` is updated to be used twice in the caption. When you specify `keep_tokens_separator="|||"`, the part divided by the second `|||` is not shuffled or dropped and remains at the end.
  - The existing features `caption_prefix` and `caption_suffix` can be used together. `caption_prefix` and `caption_suffix` are processed first, and then `enable_wildcard`, `keep_tokens_separator`, shuffling and dropping, and `secondary_separator` are processed in order.
  - See [Dataset config](./docs/config_README-en.md) for details.
- The dataset with DreamBooth method supports caching image information (size, caption). PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178) and [#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) Thanks to KohakuBlueleaf! See [DreamBooth method specific options](./docs/config_README-en.md#dreambooth-specific-options) for details.
- The support for v3 repositories is added to `tag_image_by_wd14_tagger.py` (`--onnx` option only). PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) Thanks to sdbds!
  - Onnx may need to be updated. Onnx is not installed by default, so please install or update it with `pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` etc. Please also check the comments in `requirements.txt`.
- The model is now saved in the subdirectory as `--repo_id` in `tag_image_by_wd14_tagger.py` . This caches multiple repo_id models. Please delete unnecessary files under `--model_dir`.
- The options `--noise_offset_random_strength` and `--ip_noise_gamma_random_strength` are added to each training script. These options can be used to vary the noise offset and ip noise gamma in the range of 0 to the specified value. PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) Thanks to KohakuBlueleaf!
- The options `--save_state_on_train_end` are added to each training script. PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) Thanks to gesen2egee!
- The options `--sample_every_n_epochs` and `--sample_every_n_steps` in each training script now display a warning and ignore them when a number less than or equal to `0` is specified. Thanks to S-Del for raising the issue.
- The [English version of the dataset settings documentation](./docs/config_README-en.md) is added. PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) Thanks to darkstorm2150!

- 依存ライブラリが更新されました。[アップグレード](./README-ja.md#アップグレード) を参照しライブラリを更新してください。
  - 特に `imagesize` が新しく追加されていますので、すぐに更新ができない場合は `pip install imagesize==1.4.1` でインストールしてください。
  - `bitsandbytes==0.43.0`、`prodigyopt==1.0`、`lion-pytorch==0.0.6` が requirements.txt に含まれるようになりました。
- Colab での動作時、ログ出力で停止してしまうようです。学習スクリプトに `--console_log_simple` オプションを指定し、rich のロギングを無効してお試しください。
- データセット設定の `.toml` ファイルが UTF-8 encoding で読み込まれるようになりました。PR [#1167](https://github.com/kohya-ss/sd-scripts/pull/1167) Horizon1704 氏に感謝します。
- データセット設定で、正則化画像のサブセットを複数指定した時、最後のサブセットの各種設定がすべてのサブセットの画像に適用される不具合が修正されました。それぞれのサブセットの設定が、それぞれの画像に正しく適用されます。PR [#1205](https://github.com/kohya-ss/sd-scripts/pull/1205) feffy380 氏に感謝します。
- `train_network.py` および `sdxl_train_network.py` で、学習したモデルのメタデータに一部のデータセット設定が記録されるよう修正しました（`caption_prefix`、`caption_suffix`、`keep_tokens_separator`、`secondary_separator`、`enable_wildcard`）。
- DeepSpeed がサポートされました。PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) 、[#1139](https://github.com/kohya-ss/sd-scripts/pull/1139) BootsofLagrangian 氏に感謝します。詳細は PR [#1101](https://github.com/kohya-ss/sd-scripts/pull/1101) をご覧ください。
- 各学習スクリプトでマスクロスをサポートしました。PR [#1207](https://github.com/kohya-ss/sd-scripts/pull/1207) 詳細は [Masked loss](#masked-loss) をご覧ください。
- データセットのサブセット設定にいくつかの機能を追加しました。
  - シャッフルの対象とならないタグ分割識別子の指定 `secondary_separator` を追加しました。`secondary_separator=";;;"` のように指定します。`secondary_separator` で区切ることで、その部分はシャッフル、drop 時にまとめて扱われます。
  - `enable_wildcard` を追加しました。`true` にするとワイルドカード記法 `{aaa|bbb|ccc}` が使えます。また複数行キャプションも有効になります。
  - `keep_tokens_separator` をキャプション内に 2 つ使えるようにしました。たとえば `keep_tokens_separator="|||"` と指定したとき、`1girl, hatsune miku, vocaloid ||| stage, mic ||| best quality, rating: general` とキャプションを指定すると、二番目の `|||` で分割された部分はシャッフル、drop されず末尾に残ります。
  - 既存の機能 `caption_prefix` と `caption_suffix` とあわせて使えます。`caption_prefix` と `caption_suffix` は一番最初に処理され、その後、ワイルドカード、`keep_tokens_separator`、シャッフルおよび drop、`secondary_separator` の順に処理されます。
  - 詳細は [データセット設定](./docs/config_README-ja.md) をご覧ください。
- DreamBooth 方式の DataSet で画像情報（サイズ、キャプション）をキャッシュする機能が追加されました。PR [#1178](https://github.com/kohya-ss/sd-scripts/pull/1178)、[#1206](https://github.com/kohya-ss/sd-scripts/pull/1206) KohakuBlueleaf 氏に感謝します。詳細は [データセット設定](./docs/config_README-ja.md#dreambooth-方式専用のオプション) をご覧ください。
- `tag_image_by_wd14_tagger.py` で v3 のリポジトリがサポートされました（`--onnx` 指定時のみ有効）。 PR [#1192](https://github.com/kohya-ss/sd-scripts/pull/1192) sdbds 氏に感謝します。
  - Onnx のバージョンアップが必要になるかもしれません。デフォルトでは Onnx はインストールされていませんので、`pip install onnx==1.15.0 onnxruntime-gpu==1.17.1` 等でインストール、アップデートしてください。`requirements.txt` のコメントもあわせてご確認ください。
- `tag_image_by_wd14_tagger.py` で、モデルを`--repo_id` のサブディレクトリに保存するようにしました。これにより複数のモデルファイルがキャッシュされます。`--model_dir` 直下の不要なファイルは削除願います。
- 各学習スクリプトに、noise offset、ip noise gammaを、それぞれ 0~指定した値の範囲で変動させるオプション `--noise_offset_random_strength` および `--ip_noise_gamma_random_strength` が追加されました。 PR [#1177](https://github.com/kohya-ss/sd-scripts/pull/1177) KohakuBlueleaf 氏に感謝します。
- 各学習スクリプトに、学習終了時に state を保存する `--save_state_on_train_end` オプションが追加されました。 PR [#1168](https://github.com/kohya-ss/sd-scripts/pull/1168) gesen2egee 氏に感謝します。
- 各学習スクリプトで `--sample_every_n_epochs` および `--sample_every_n_steps` オプションに `0` 以下の数値を指定した時、警告を表示するとともにそれらを無視するよう変更しました。問題提起していただいた S-Del 氏に感謝します。
- データセット設定の[英語版ドキュメント](./docs/config_README-en.md) が追加されました。PR [#1175](https://github.com/kohya-ss/sd-scripts/pull/1175) darkstorm2150 氏に感謝します。

Please read [Releases](https://github.com/kohya-ss/sd-scripts/releases) for recent updates.
最近の更新情報は [Release](https://github.com/kohya-ss/sd-scripts/releases) をご覧ください。

#### Masked loss

The masked loss is supported in each training script. To enable the masked loss, specify the `--masked_loss` option.

The feature is not fully tested, so there may be bugs. If you find any issues, please open an Issue.

ControlNet dataset is used to specify the mask. The mask images should be the RGB images. The pixel value 255 in R channel is treated as the mask (the loss is calculated only for the pixels with the mask), and 0 is treated as the non-mask. The pixel values 0-255 are converted to 0-1 (i.e., the pixel value 128 is treated as the half weight of the loss). See details for the dataset specification in the [LLLite documentation](./docs/train_lllite_README.md#preparing-the-dataset).

各学習スクリプトでマスクロスをサポートしました。マスクロスを有効にするには `--masked_loss` オプションを指定してください。

機能は完全にテストされていないため、不具合があるかもしれません。その場合は Issue を立てていただけると助かります。

マスクの指定には ControlNet データセットを使用します。マスク画像は RGB 画像である必要があります。R チャンネルのピクセル値 255 がロス計算対象、0 がロス計算対象外になります。0-255 の値は、0-1 の範囲に変換されます（つまりピクセル値 128 の部分はロスの重みが半分になります）。データセットの詳細は [LLLite ドキュメント](./docs/train_lllite_README-ja.md#データセットの準備) をご覧ください。


## Additional Information

### Naming of LoRA

The LoRA supported by `train_network.py` has been named to avoid confusion. The documentation has been updated. The following are the names of LoRA types in this repository.

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers)

    LoRA for Linear layers and Conv2d layers with 1x1 kernel

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers)

    In addition to 1., LoRA for Conv2d layers with 3x3 kernel 
    
LoRA-LierLa is the default LoRA type for `train_network.py` (without `conv_dim` network arg). LoRA-LierLa can be used with [our extension](https://github.com/kohya-ss/sd-webui-additional-networks) for AUTOMATIC1111's Web UI, or with the built-in LoRA feature of the Web UI.

To use LoRA-C3Lier with Web UI, please use our extension.

### LoRAの名称について

`train_network.py` がサポートするLoRAについて、混乱を避けるため名前を付けました。ドキュメントは更新済みです。以下は当リポジトリ内の独自の名称です。

1. __LoRA-LierLa__ : (LoRA for __Li__ n __e__ a __r__  __La__ yers、リエラと読みます)

    Linear 層およびカーネルサイズ 1x1 の Conv2d 層に適用されるLoRA

2. __LoRA-C3Lier__ : (LoRA for __C__ olutional layers with __3__ x3 Kernel and  __Li__ n __e__ a __r__ layers、セリアと読みます)

    1.に加え、カーネルサイズ 3x3 の Conv2d 層に適用されるLoRA

LoRA-LierLa は[Web UI向け拡張](https://github.com/kohya-ss/sd-webui-additional-networks)、またはAUTOMATIC1111氏のWeb UIのLoRA機能で使用することができます。

LoRA-C3Lierを使いWeb UIで生成するには拡張を使用してください。

## Sample image generation during training
  A prompt file might look like this, for example

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  Lines beginning with `#` are comments. You can specify options for the generated image with options like `--n` after the prompt. The following can be used.

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  The prompt weighting such as `( )` and `[ ]` are working.

## サンプル画像生成
プロンプトファイルは例えば以下のようになります。

```
# prompt 1
masterpiece, best quality, (1girl), in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# prompt 2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n (low quality, worst quality), bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

  `#` で始まる行はコメントになります。`--n` のように「ハイフン二個＋英小文字」の形でオプションを指定できます。以下が使用可能できます。

  * `--n` Negative prompt up to the next option.
  * `--w` Specifies the width of the generated image.
  * `--h` Specifies the height of the generated image.
  * `--d` Specifies the seed of the generated image.
  * `--l` Specifies the CFG scale of the generated image.
  * `--s` Specifies the number of steps in the generation.

  `( )` や `[ ]` などの重みづけも動作します。

