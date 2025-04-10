# `flux_train_network.py` を用いたFLUX.1モデルのLoRA学習ガイド

このドキュメントでは、`sd-scripts`リポジトリに含まれる`flux_train_network.py`を使用して、FLUX.1モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

## 1. はじめに

`flux_train_network.py`は、FLUX.1モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。FLUX.1はStable Diffusionとは異なるアーキテクチャを持つ画像生成モデルであり、このスクリプトを使用することで、特定のキャラクターや画風を再現するLoRAモデルを作成できます。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象とし、`train_network.py`での学習経験があることを前提としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](how_to_use_train_network.md)を参照してください。

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
    *   CLIP-Lモデルの`.safetensors`ファイル。
    *   T5-XXLモデルの`.safetensors`ファイル。
4.  **AutoEncoderモデルファイル:** FLUX.1に対応するAEモデルの`.safetensors`ファイル。
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
 --network_module=networks.lora 
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
 --apply_t5_attn_mask 
 --blocks_to_swap=18
```

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。

### 4.1. 主要なコマンドライン引数の解説（`train_network.py`からの追加・変更点）

[`train_network.py`のガイド](how_to_use_train_network.md)で説明されている引数に加え、以下のFLUX.1特有の引数を指定します。共通の引数（`--output_dir`, `--output_name`, `--network_module`, `--network_dim`, `--network_alpha`, `--learning_rate`など）については、上記ガイドを参照してください。

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

*   `--t5xxl_max_token_length=<integer>`
    *   T5-XXL Text Encoderで使用するトークンの最大長を指定します。省略した場合、モデルがschnell版なら256、dev版なら512が自動的に設定されます。データセットのキャプション長に合わせて調整が必要な場合があります。
*   `--apply_t5_attn_mask`
    *   T5-XXLの出力とFLUXモデル内部（Double Block）のアテンション計算時に、パディングトークンに対応するアテンションマスクを適用します。精度向上が期待できる場合がありますが、わずかに計算コストが増加します。
*   `--guidance_scale=<float>`
    *   FLUX.1 dev版は特定のガイダンススケール値で蒸留されているため、学習時にもその値を指定します。デフォルトは`3.5`です。schnell版では通常無視されます。
*   `--timestep_sampling=<choice>`
    *   学習時に使用するタイムステップ（ノイズレベル）のサンプリング方法を指定します。`sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift` から選択します。デフォルトは `sigma` です。
*   `--sigmoid_scale=<float>`
    *   `timestep_sampling` に `sigmoid` または `shift`, `flux_shift` を指定した場合のスケール係数です。デフォルトは`1.0`です。
*   `--model_prediction_type=<choice>`
    *   モデルが何を予測するかを指定します。`raw` (予測値をそのまま使用), `additive` (ノイズ入力に加算), `sigma_scaled` (シグマスケーリングを適用) から選択します。デフォルトは `sigma_scaled` です。
*   `--discrete_flow_shift=<float>`
    *   Flow Matchingで使用されるスケジューラのシフト値を指定します。デフォルトは`3.0`です。

#### メモリ・速度関連

*   `--blocks_to_swap=<integer>` **[実験的機能]**
    *   VRAM使用量を削減するために、モデルの一部（Transformerブロック）をCPUとGPU間でスワップする設定です。スワップするブロック数を整数で指定します（例: `18`）。値を大きくするとVRAM使用量は減りますが、学習速度は低下します。GPUのVRAM容量に応じて調整してください。`gradient_checkpointing`と併用可能です。
    *   `--cpu_offload_checkpointing`とは併用できません。

#### 非互換・非推奨の引数

*   `--v2`, `--v_parameterization`, `--clip_skip`: Stable Diffusion特有の引数のため、FLUX.1学習では使用されません。
*   `--max_token_length`: Stable Diffusion v1/v2向けの引数です。FLUX.1では`--t5xxl_max_token_length`を使用してください。
*   `--split_mode`: 非推奨の引数です。代わりに`--blocks_to_swap`を使用してください。

### 4.2. 学習の開始

必要な引数を設定し、コマンドを実行すると学習が開始されます。基本的な流れやログの確認方法は[`train_network.py`のガイド](how_to_use_train_network.md#32-starting-the-training--学習の開始)と同様です。

## 5. 学習済みモデルの利用

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_flux_lora.safetensors`）が保存されます。このファイルは、FLUX.1モデルに対応した推論環境（例: ComfyUI + ComfyUI-FluxNodes）で使用できます。

## 6. その他

`flux_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](how_to_use_train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python flux_train_network.py --help`) を参照してください。
