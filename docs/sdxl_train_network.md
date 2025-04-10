はい、承知いたしました。`sd-scripts` リポジトリに含まれる `sdxl_train_network.py` を使用した SDXL LoRA 学習に関するドキュメントを作成します。`how_to_use_train_network.md` との差分を中心に、初心者ユーザー向けに解説します。

---

# SDXL LoRA学習スクリプト `sdxl_train_network.py` の使い方

このドキュメントでは、`sd-scripts` リポジトリに含まれる `sdxl_train_network.py` を使用して、SDXL (Stable Diffusion XL) モデルに対する LoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

## 1. はじめに

`sdxl_train_network.py` は、SDXL モデルに対して LoRA などの追加ネットワークを学習させるためのスクリプトです。基本的な使い方は `train_network.py` ([LoRA学習スクリプト `train_network.py` の使い方](how_to_use_train_network.md) 参照) と共通ですが、SDXL モデル特有の設定が必要となります。

このガイドでは、SDXL LoRA 学習に焦点を当て、`train_network.py` との主な違いや SDXL 特有の設定項目を中心に説明します。

**前提条件:**

*   `sd-scripts` リポジトリのクローンと Python 環境のセットアップが完了していること。
*   学習用データセットの準備が完了していること。（データセットの準備については[データセット準備ガイド](link/to/dataset/doc)を参照してください）
*   [LoRA学習スクリプト `train_network.py` の使い方](how_to_use_train_network.md) を一読していること。

## 2. 準備

学習を開始する前に、以下のファイルが必要です。

1.  **学習スクリプト:** `sdxl_train_network.py`
2.  **データセット定義ファイル (.toml):** 学習データセットの設定を記述した TOML 形式のファイル。

### データセット定義ファイルについて

データセット定義ファイル (`.toml`) の基本的な書き方は `train_network.py` と共通です。[データセット設定ガイド](link/to/dataset/config/doc) および [LoRA学習スクリプト `train_network.py` の使い方](how_to_use_train_network.md#データセット定義ファイルについて) を参照してください。

SDXL では、高解像度のデータセットや、アスペクト比バケツ機能 (`enable_bucket = true`) の利用が一般的です。

ここでは、例として `my_sdxl_dataset_config.toml` という名前のファイルを使用することにします。

## 3. 学習の実行

学習は、ターミナルから `sdxl_train_network.py` を実行することで開始します。

以下に、SDXL LoRA 学習における基本的なコマンドライン実行例を示します。

```bash
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py 
 --pretrained_model_name_or_path="<SDXLベースモデルのパス>" 
 --dataset_config="my_sdxl_dataset_config.toml" 
 --output_dir="<学習結果の出力先ディレクトリ>" 
 --output_name="my_sdxl_lora" 
 --save_model_as=safetensors 
 --network_module=networks.lora 
 --network_dim=32 
 --network_alpha=16 
 --learning_rate=1e-4 
 --unet_lr=1e-4 
 --text_encoder_lr1=1e-5 
 --text_encoder_lr2=1e-5 
 --optimizer_type="AdamW8bit" 
 --lr_scheduler="constant" 
 --max_train_epochs=10 
 --save_every_n_epochs=1 
 --mixed_precision="bf16" 
 --gradient_checkpointing 
 --cache_text_encoder_outputs 
 --cache_latents
```

`train_network.py` の実行例と比較すると、以下の点が異なります。

*   実行するスクリプトが `sdxl_train_network.py` になります。
*   `--pretrained_model_name_or_path` には SDXL のベースモデルを指定します。
*   `--text_encoder_lr` が `--text_encoder_lr1` と `--text_encoder_lr2` に分かれています（SDXL は2つの Text Encoder を持つため）。
*   `--mixed_precision` は `bf16` または `fp16` が推奨されます。
*   `--cache_text_encoder_outputs` や `--cache_latents` は VRAM 使用量を削減するために推奨されます。

次に、`train_network.py` との差分となる主要なコマンドライン引数について解説します。共通の引数については、[LoRA学習スクリプト `train_network.py` の使い方](how_to_use_train_network.md#31-主要なコマンドライン引数) を参照してください。

### 3.1. 主要なコマンドライン引数（差分）

#### モデル関連

*   `--pretrained_model_name_or_path="<モデルのパス>"` **[必須]**
    *   学習のベースとなる **SDXL モデル**を指定します。Hugging Face Hub のモデル ID (例: `"stabilityai/stable-diffusion-xl-base-1.0"`) や、ローカルの Diffusers 形式モデルのディレクトリ、`.safetensors` ファイルのパスを指定できます。
*   `--v2`, `--v_parameterization`
    *   これらの引数は SD1.x/2.x 用です。`sdxl_train_network.py` を使用する場合、SDXL モデルであることが前提となるため、通常は**指定する必要はありません**。

#### データセット関連

*   `--dataset_config="<設定ファイルのパス>"`
    *   `train_network.py` と共通です。
    *   SDXL では高解像度データやバケツ機能 (`.toml` で `enable_bucket = true` を指定) の利用が一般的です。

#### 出力・保存関連

*   `train_network.py` と共通です。

#### LoRA パラメータ

*   `train_network.py` と共通です。

#### 学習パラメータ

*   `--learning_rate=1e-4`
    *   全体の学習率。`unet_lr`, `text_encoder_lr1`, `text_encoder_lr2` が指定されない場合のデフォルト値となります。
*   `--unet_lr=1e-4`
    *   U-Net 部分の LoRA モジュールに対する学習率。指定しない場合は `--learning_rate` の値が使用されます。
*   `--text_encoder_lr1=1e-5`
    *   **Text Encoder 1 (OpenCLIP ViT-G/14) の LoRA モジュール**に対する学習率。指定しない場合は `--learning_rate` の値が使用されます。U-Net より小さめの値が推奨されます。
*   `--text_encoder_lr2=1e-5`
    *   **Text Encoder 2 (CLIP ViT-L/14) の LoRA モジュール**に対する学習率。指定しない場合は `--learning_rate` の値が使用されます。U-Net より小さめの値が推奨されます。
*   `--optimizer_type="AdamW8bit"`
    *   `train_network.py` と共通です。
*   `--lr_scheduler="constant"`
    *   `train_network.py` と共通です。
*   `--lr_warmup_steps`
    *   `train_network.py` と共通です。
*   `--max_train_steps`, `--max_train_epochs`
    *   `train_network.py` と共通です。
*   `--mixed_precision="bf16"`
    *   混合精度学習の設定。SDXL では `bf16` または `fp16` の使用が推奨されます。GPU が対応している方を選択してください。VRAM 使用量を削減し、学習速度を向上させます。
*   `--gradient_accumulation_steps=1`
    *   `train_network.py` と共通です。
*   `--gradient_checkpointing`
    *   `train_network.py` と共通です。SDXL はメモリ消費が大きいため、有効にすることが推奨されます。
*   `--cache_latents`
    *   VAE の出力をメモリ（または `--cache_latents_to_disk` 指定時はディスク）にキャッシュします。VAE の計算を省略できるため、VRAM 使用量を削減し、学習を高速化できます。画像に対する Augmentation (`--color_aug`, `--flip_aug`, `--random_crop` 等) が無効になります。SDXL 学習では推奨されるオプションです。
*   `--cache_latents_to_disk`
    *   `--cache_latents` と併用し、キャッシュ先をディスクにします。データセットを最初に読み込む際に、VAE の出力をディスクにキャッシュします。二回目以降の学習で VAE の計算を省略できるため、学習データの枚数が多い場合に推奨されます。
*   `--cache_text_encoder_outputs`
    *   Text Encoder の出力をメモリ（または `--cache_text_encoder_outputs_to_disk` 指定時はディスク）にキャッシュします。Text Encoder の計算を省略できるため、VRAM 使用量を削減し、学習を高速化できます。キャプションに対する Augmentation (`--shuffle_caption`, `--caption_dropout_rate` 等) が無効になります。
    *   **注意:** このオプションを使用する場合、Text Encoder の LoRA モジュールは学習できません (`--network_train_unet_only` の指定が必須です)。
*   `--cache_text_encoder_outputs_to_disk`
    *   `--cache_text_encoder_outputs` と併用し、キャッシュ先をディスクにします。
*   `--no_half_vae`
    *   混合精度 (`fp16`/`bf16`) 使用時でも VAE を `float32` で動作させます。SDXL の VAE は `float16` で不安定になることがあるため、`fp16` 指定時には有効にしてくだ
*   `--clip_skip`
    *   SDXL では通常使用しません。指定は不要です。
*   `--fused_backward_pass`
    *   勾配計算とオプティマイザのステップを融合し、VRAM使用量を削減します。SDXLで利用可能です。（現在 `Adafactor` オプティマイザのみ対応）

#### その他

*   `--seed`, `--logging_dir`, `--log_prefix` などは `train_network.py` と共通です。

### 3.2. 学習の開始

必要な引数を設定し、コマンドを実行すると学習が開始されます。学習の進行状況はコンソールに出力されます。基本的な流れは `train_network.py` と同じです。

## 4. 学習済みモデルの利用

学習が完了すると、`output_dir` で指定したディレクトリに、`output_name` で指定した名前の LoRA モデルファイル (`.safetensors` など) が保存されます。

このファイルは、AUTOMATIC1111/stable-diffusion-webui 、ComfyUI などの SDXL に対応した GUI ツールで利用できます。

## 5. 補足: `train_network.py` との主な違い

*   **対象モデル:** `sdxl_train_network.py` は SDXL モデル専用です。
*   **Text Encoder:** SDXL は 2 つの Text Encoder を持つため、学習率の指定 (`--text_encoder_lr1`, `--text_encoder_lr2`) などが異なります。
*   **キャッシュ機能:** `--cache_text_encoder_outputs` は SDXL で特に効果が高く、推奨されます。
*   **推奨設定:** VRAM 使用量が大きいため、`bf16` または `fp16` の混合精度、`gradient_checkpointing`、キャッシュ機能 (`--cache_latents`, `--cache_text_encoder_outputs`) の利用が推奨されます。`fp16` 指定時は、VAE は `--no_half_vae` で `float32` 動作を推奨します。

その他の詳細なオプションについては、スクリプトのヘルプ (`python sdxl_train_network.py --help`) やリポジトリ内の他のドキュメントを参照してください。