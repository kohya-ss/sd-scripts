Status: reviewed

# LoRA Training Guide for FLUX.1 using `flux_train_network.py` / `flux_train_network.py` を用いたFLUX.1モデルのLoRA学習ガイド

This document explains how to train LoRA models for the FLUX.1 model using `flux_train_network.py` included in the `sd-scripts` repository.

## 1. Introduction / はじめに

`flux_train_network.py` trains additional networks such as LoRA on the FLUX.1 model, which uses a transformer-based architecture different from Stable Diffusion. Two text encoders, CLIP-L and T5-XXL, and a dedicated AutoEncoder are used.

This guide assumes you know the basics of LoRA training. For common options see [train_network.py](train_network.md) and [sdxl_train_network.py](sdxl_train_network.md).

**Prerequisites:**

* The repository is cloned and the Python environment is ready.
* A training dataset is prepared. See the dataset configuration guide.

## 2. Differences from `train_network.py` / `train_network.py` との違い

`flux_train_network.py` is based on `train_network.py` but adapted for FLUX.1. Main differences include required arguments for the FLUX.1 model, CLIP-L, T5-XXL and AE, different model structure, and some incompatible options from Stable Diffusion.

## 3. Preparation / 準備

Before starting training you need:

1. **Training script:** `flux_train_network.py`
2. **FLUX.1 model file** and text encoder files (`clip_l`, `t5xxl`) and AE file.
3. **Dataset definition file (.toml)** such as `my_flux_dataset_config.toml`.

## 4. Running the Training / 学習の実行

Run `flux_train_network.py` from the terminal with FLUX.1 specific arguments. Example:

```bash
accelerate launch --num_cpu_threads_per_process 1 flux_train_network.py \
  --pretrained_model_name_or_path="<path to FLUX.1 model>" \
  --clip_l="<path to CLIP-L model>" \
  --t5xxl="<path to T5-XXL model>" \
  --ae="<path to AE model>" \
  --dataset_config="my_flux_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_flux_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_flux \
  --network_dim=16 \
  --network_alpha=1 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --sdpa \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="fp16" \
  --gradient_checkpointing \
  --guidance_scale=1.0 \
  --timestep_sampling="flux_shift" \
  --blocks_to_swap=18 \
  --cache_text_encoder_outputs \
  --cache_latents
```

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

The script adds FLUX.1 specific arguments such as guidance scale, timestep sampling, block swapping, and options for training CLIP-L and T5-XXL LoRA modules. Some Stable Diffusion options like `--v2` and `--clip_skip` are not used.

### 4.2. Starting Training / 学習の開始

Training begins once you run the command with the required options. Log checking is the same as in `train_network.py`.

## 5. Using the Trained Model / 学習済みモデルの利用

After training, a LoRA model file is saved in `output_dir` and can be used in inference environments supporting FLUX.1 (e.g. ComfyUI + Flux nodes).

## 6. Others / その他

Additional notes on VRAM optimization, training options, multi-resolution datasets, block selection and text encoder LoRA are provided in the Japanese section.

<details>
<summary>日本語</summary>



# `flux_train_network.py` を用いたFLUX.1モデルのLoRA学習ガイド

このドキュメントでは、`sd-scripts`リポジトリに含まれる`flux_train_network.py`を使用して、FLUX.1モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

## 1. はじめに

`flux_train_network.py`は、FLUX.1モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。FLUX.1はStable Diffusionとは異なるアーキテクチャを持つ画像生成モデルであり、このスクリプトを使用することで、特定のキャラクターや画風を再現するLoRAモデルを作成できます。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sdxl_train_network.py`](sdxl_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

*   `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
*   学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）

## 2. `train_network.py` との違い

`flux_train_network.py`は`train_network.py`をベースに、FLUX.1モデルに対応するための変更が加えられています。主な違いは以下の通りです。

*   **対象モデル:** FLUX.1モデル（dev版またはschnell版）を対象とします。
*   **モデル構造:** Stable Diffusionとは異なり、FLUX.1はTransformerベースのアーキテクチャを持ちます。Text EncoderとしてCLIP-LとT5-XXLの二つを使用し、VAEの代わりに専用のAutoEncoder (AE) を使用します。
*   **必須の引数:** FLUX.1モデル、CLIP-L、T5-XXL、AEの各モデルファイルを指定する引数が追加されています。
*   **一部引数の非互換性:** Stable Diffusion向けの引数の一部（例: `--v2`, `--clip_skip`, `--max_token_length`）はFLUX.1の学習では使用されません。
*   **FLUX.1特有の引数:** タイムステップのサンプリング方法やガイダンススケールなど、FLUX.1特有の学習パラメータを指定する引数が追加されています。

## 3. 準備

学習を開始する前に、以下のファイルが必要です。

1.  **学習スクリプト:** `flux_train_network.py`
2.  **FLUX.1モデルファイル:** 学習のベースとなるFLUX.1モデルの`.safetensors`ファイル（例: `flux1-dev.safetensors`）。
3.  **Text Encoderモデルファイル:**
    *   CLIP-Lモデルの`.safetensors`ファイル。例として`clip_l.safetensors`を使用します。
    *   T5-XXLモデルの`.safetensors`ファイル。例として`t5xxl.safetensors`を使用します。
4.  **AutoEncoderモデルファイル:** FLUX.1に対応するAEモデルの`.safetensors`ファイル。例として`ae.safetensors`を使用します。
5.  **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）。

    *   例として`my_flux_dataset_config.toml`を使用します。

## 4. 学習の実行

学習は、ターミナルから`flux_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、FLUX.1特有の引数を指定する必要があります。

以下に、基本的なコマンドライン実行例を示します。

```bash
accelerate launch --num_cpu_threads_per_process 1 flux_train_network.py 
 --pretrained_model_name_or_path="<path to FLUX.1 model>" 
 --clip_l="<path to CLIP-L model>" 
 --t5xxl="<path to T5-XXL model>" 
 --ae="<path to AE model>" 
 --dataset_config="my_flux_dataset_config.toml" 
 --output_dir="<output directory for training results>" 
 --output_name="my_flux_lora" 
 --save_model_as=safetensors 
 --network_module=networks.lora_flux 
 --network_dim=16 
 --network_alpha=1 
 --learning_rate=1e-4 
 --optimizer_type="AdamW8bit" 
 --lr_scheduler="constant" 
 --sdpa  
 --max_train_epochs=10 
 --save_every_n_epochs=1 
 --mixed_precision="fp16" 
 --gradient_checkpointing 
 --guidance_scale=1.0 
 --timestep_sampling="flux_shift" 
 --blocks_to_swap=18
 --cache_text_encoder_outputs 
 --cache_latents
```

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。

### 4.1. 主要なコマンドライン引数の解説（`train_network.py`からの追加・変更点）

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のFLUX.1特有の引数を指定します。共通の引数（`--output_dir`, `--output_name`, `--network_module`, `--network_dim`, `--network_alpha`, `--learning_rate`など）については、上記ガイドを参照してください。

#### モデル関連 [必須]

*   `--pretrained_model_name_or_path="<path to FLUX.1 model>"` **[必須]**
    *   学習のベースとなるFLUX.1モデル（dev版またはschnell版）の`.safetensors`ファイルのパスを指定します。Diffusers形式のディレクトリは現在サポートされていません。
*   `--clip_l="<path to CLIP-L model>"` **[必須]**
    *   CLIP-L Text Encoderモデルの`.safetensors`ファイルのパスを指定します。
*   `--t5xxl="<path to T5-XXL model>"` **[必須]**
    *   T5-XXL Text Encoderモデルの`.safetensors`ファイルのパスを指定します。
*   `--ae="<path to AE model>"` **[必須]**
    *   FLUX.1に対応するAutoEncoderモデルの`.safetensors`ファイルのパスを指定します。

#### FLUX.1 学習パラメータ

*   `--guidance_scale=<float>`
    *   FLUX.1 dev版は特定のガイダンススケール値で蒸留されていますが、学習時には `1.0` を指定してガイダンススケールを無効化します。デフォルトは`3.5`ですので、必ず指定してください。schnell版では通常無視されます。
*   `--timestep_sampling=<choice>`
    *   学習時に使用するタイムステップ（ノイズレベル）のサンプリング方法を指定します。`sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift` から選択します。デフォルトは `sigma` です。推奨は `flux_shift` です。
*   `--sigmoid_scale=<float>`
    *   `timestep_sampling` に `sigmoid` または `shift`, `flux_shift` を指定した場合のスケール係数です。デフォルトおよび推奨値は`1.0`です。
*   `--model_prediction_type=<choice>`
    *   モデルが何を予測するかを指定します。`raw` (予測値をそのまま使用), `additive` (ノイズ入力に加算), `sigma_scaled` (シグマスケーリングを適用) から選択します。デフォルトは `sigma_scaled` です。推奨は `raw` です。
*   `--discrete_flow_shift=<float>`
    *   Flow Matchingで使用されるスケジューラのシフト値を指定します。デフォルトは`3.0`です。`timestep_sampling`に`flux_shift`を指定した場合は、この値は無視されます。

#### メモリ・速度関連

*   `--blocks_to_swap=<integer>` **[実験的機能]**
    *   VRAM使用量を削減するために、モデルの一部（Transformerブロック）をCPUとGPU間でスワップする設定です。スワップするブロック数を整数で指定します（例: `18`）。値を大きくするとVRAM使用量は減りますが、学習速度は低下します。GPUのVRAM容量に応じて調整してください。`gradient_checkpointing`と併用可能です。
    *   `--cpu_offload_checkpointing`とは併用できません。
* `--cache_text_encoder_outputs`
    *   CLIP-LおよびT5-XXLの出力をキャッシュします。これにより、メモリ使用量が削減されます。
* `--cache_latents`, `--cache_latents_to_disk`
    *   AEの出力をキャッシュします。[sdxl_train_network.py](sdxl_train_network.md)と同様の機能です。

#### 非互換・非推奨の引数

*   `--v2`, `--v_parameterization`, `--clip_skip`: Stable Diffusion特有の引数のため、FLUX.1学習では使用されません。
*   `--max_token_length`: Stable Diffusion v1/v2向けの引数です。FLUX.1では`--t5xxl_max_token_length`を使用してください。
*   `--split_mode`: 非推奨の引数です。代わりに`--blocks_to_swap`を使用してください。

### 4.2. 学習の開始

必要な引数を設定し、コマンドを実行すると学習が開始されます。基本的な流れやログの確認方法は[`train_network.py`のガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

## 5. 学習済みモデルの利用

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_flux_lora.safetensors`）が保存されます。このファイルは、FLUX.1モデルに対応した推論環境（例: ComfyUI + ComfyUI-FluxNodes）で使用できます。

## 6. その他

`flux_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python flux_train_network.py --help`) を参照してください。

# FLUX.1 LoRA学習の補足説明

以下は、以上の基本的なFLUX.1 LoRAの学習手順を補足するものです。より詳細な設定オプションなどについて説明します。

## 1. VRAM使用量の最適化

FLUX.1モデルは比較的大きなモデルであるため、十分なVRAMを持たないGPUでは工夫が必要です。以下に、VRAM使用量を削減するための設定を紹介します。

### 1.1 メモリ使用量別の推奨設定

| GPUメモリ | 推奨設定 |
|----------|----------|
| 24GB VRAM | 基本設定で問題なく動作します（バッチサイズ2） |
| 16GB VRAM | バッチサイズ1に設定し、`--blocks_to_swap`を使用 |
| 12GB VRAM | `--blocks_to_swap 16`と8bit AdamWを使用 |
| 10GB VRAM | `--blocks_to_swap 22`を使用、T5XXLはfp8形式を推奨 |
| 8GB VRAM | `--blocks_to_swap 28`を使用、T5XXLはfp8形式を推奨 |

### 1.2 主要なVRAM削減オプション

- **`--blocks_to_swap <数値>`**：
  CPUとGPU間でブロックをスワップしてVRAM使用量を削減します。数値が大きいほど多くのブロックをスワップし、より多くのVRAMを節約できますが、学習速度は低下します。FLUX.1では最大35ブロックまでスワップ可能です。

- **`--cpu_offload_checkpointing`**：
  勾配チェックポイントをCPUにオフロードします。これにより最大1GBのVRAM使用量を削減できますが、学習速度は約15%低下します。`--blocks_to_swap`とは併用できません。

- **`--cache_text_encoder_outputs` / `--cache_text_encoder_outputs_to_disk`**：
  CLIP-LとT5-XXLの出力をキャッシュします。これによりメモリ使用量を削減できます。

- **`--cache_latents` / `--cache_latents_to_disk`**：
  AEの出力をキャッシュします。メモリ使用量を削減できます。

- **Adafactor オプティマイザの使用**：
  8bit AdamWよりもVRAM使用量を削減できます。以下の設定を使用してください：
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

- **T5XXLのfp8形式の使用**：
  10GB未満のVRAMを持つGPUでは、T5XXLのfp8形式チェックポイントの使用を推奨します。[comfyanonymous/flux_text_encoders](https://huggingface.co/comfyanonymous/flux_text_encoders)から`t5xxl_fp8_e4m3fn.safetensors`をダウンロードできます（`scaled`なしで使用してください）。

## 2. FLUX.1 LoRA学習の重要な設定オプション

FLUX.1の学習には多くの未知の点があり、いくつかの設定は引数で指定できます。以下に重要な引数とその説明を示します。

### 2.1 タイムステップのサンプリング方法

`--timestep_sampling`オプションで、タイムステップ（0-1）のサンプリング方法を指定できます：

- `sigma`：SD3と同様のシグマベース
- `uniform`：一様ランダム
- `sigmoid`：正規分布乱数のシグモイド（x-flux、AI-toolkitなどと同様）
- `shift`：正規分布乱数のシグモイド値をシフト
- `flux_shift`：解像度に応じて正規分布乱数のシグモイド値をシフト（FLUX.1 dev推論と同様）。この設定では`--discrete_flow_shift`は無視されます。

### 2.2 モデル予測の処理方法

`--model_prediction_type`オプションで、モデルの予測をどのように解釈し処理するかを指定できます：

- `raw`：そのまま使用（x-fluxと同様）【推奨】
- `additive`：ノイズ入力に加算
- `sigma_scaled`：シグマスケーリングを適用（SD3と同様）

### 2.3 推奨設定

実験の結果、以下の設定が良好に動作することが確認されています：
```
--timestep_sampling shift --discrete_flow_shift 3.1582 --model_prediction_type raw --guidance_scale 1.0
```

ガイダンススケールについて：FLUX.1 dev版は特定のガイダンススケール値で蒸留されていますが、学習時には`--guidance_scale 1.0`を指定してガイダンススケールを無効化することを推奨します。

## 3. 各層に対するランク指定

FLUX.1の各層に対して異なるランク（network_dim）を指定できます。これにより、特定の層に対してLoRAの効果を強調したり、無効化したりできます。

以下のnetwork_argsを指定することで、各層のランクを指定できます。0を指定するとその層にはLoRAが適用されません。

| network_args | 対象レイヤー |
|--------------|--------------|
| img_attn_dim | DoubleStreamBlockのimg_attn |
| txt_attn_dim | DoubleStreamBlockのtxt_attn |
| img_mlp_dim | DoubleStreamBlockのimg_mlp |
| txt_mlp_dim | DoubleStreamBlockのtxt_mlp |
| img_mod_dim | DoubleStreamBlockのimg_mod |
| txt_mod_dim | DoubleStreamBlockのtxt_mod |
| single_dim | SingleStreamBlockのlinear1とlinear2 |
| single_mod_dim | SingleStreamBlockのmodulation |

使用例：
```
--network_args "img_attn_dim=4" "img_mlp_dim=8" "txt_attn_dim=2" "txt_mlp_dim=2" "img_mod_dim=2" "txt_mod_dim=2" "single_dim=4" "single_mod_dim=2"
```

さらに、FLUXの条件付けレイヤーにLoRAを適用するには、network_argsに`in_dims`を指定します。5つの数値をカンマ区切りのリストとして指定する必要があります。

例：
```
--network_args "in_dims=[4,2,2,2,4]"
```

各数値は、`img_in`、`time_in`、`vector_in`、`guidance_in`、`txt_in`に対応します。上記の例では、すべての条件付けレイヤーにLoRAを適用し、`img_in`と`txt_in`のランクを4、その他のランクを2に設定しています。

0を指定するとそのレイヤーにはLoRAが適用されません。例えば、`[4,0,0,0,4]`は`img_in`と`txt_in`にのみLoRAを適用します。

## 4. 学習するブロックの指定

FLUX.1 LoRA学習では、network_argsの`train_double_block_indices`と`train_single_block_indices`を指定することで、学習するブロックを指定できます。インデックスは0ベースです。省略した場合のデフォルトはすべてのブロックを学習することです。

インデックスは、`0,1,5,8`のような整数のリストや、`0,1,4-5,7`のような整数の範囲として指定します。
- double blocksの数は19なので、有効な範囲は0-18です
- single blocksの数は38なので、有効な範囲は0-37です
- `all`を指定するとすべてのブロックを学習します
- `none`を指定するとブロックを学習しません

使用例：
```
--network_args "train_double_block_indices=0,1,8-12,18" "train_single_block_indices=3,10,20-25,37"
```

または：
```
--network_args "train_double_block_indices=none" "train_single_block_indices=10-15"
```

`train_double_block_indices`または`train_single_block_indices`のどちらか一方だけを指定した場合、もう一方は通常通り学習されます。

## 5. Text Encoder LoRAのサポート

FLUX.1 LoRA学習は、CLIP-LとT5XXL LoRAのトレーニングもサポートしています。

- FLUX.1のみをトレーニングする場合は、`--network_train_unet_only`を指定します
- FLUX.1とCLIP-Lをトレーニングする場合は、`--network_train_unet_only`を省略します
- FLUX.1、CLIP-L、T5XXLすべてをトレーニングする場合は、`--network_train_unet_only`を省略し、`--network_args "train_t5xxl=True"`を追加します

CLIP-LとT5XXLの学習率は、`--text_encoder_lr`で個別に指定できます。例えば、`--text_encoder_lr 1e-4 1e-5`とすると、最初の値はCLIP-Lの学習率、2番目の値はT5XXLの学習率になります。1つだけ指定すると、CLIP-LとT5XXLの学習率は同じになります。`--text_encoder_lr`を指定しない場合、デフォルトの学習率`--learning_rate`が両方に使用されます。

## 6. マルチ解像度トレーニング

データセット設定ファイルで複数の解像度を定義できます。各解像度に対して異なるバッチサイズを指定することができます。

設定ファイルの例：
```toml
[general]
# 共通設定をここで定義
flip_aug = true
color_aug = false
keep_tokens_separator= "|||"
shuffle_caption = false
caption_tag_dropout_rate = 0
caption_extension = ".txt"

[[datasets]]
# 最初の解像度の設定
batch_size = 2
enable_bucket = true
resolution = [1024, 1024]

  [[datasets.subsets]]
  image_dir = "画像ディレクトリへのパス"
  num_repeats = 1

[[datasets]]
# 2番目の解像度の設定
batch_size = 3
enable_bucket = true
resolution = [768, 768]

  [[datasets.subsets]]
  image_dir = "画像ディレクトリへのパス"
  num_repeats = 1

[[datasets]]
# 3番目の解像度の設定
batch_size = 4
enable_bucket = true
resolution = [512, 512]

  [[datasets.subsets]]
  image_dir = "画像ディレクトリへのパス"
  num_repeats = 1
```

各解像度セクションの`[[datasets.subsets]]`部分は、データセットディレクトリを定義します。各解像度に対して同じディレクトリを指定してください。</details>
