# `sd3_train_network.py` を用いたStable Diffusion 3/3.5モデルのLoRA学習ガイド

このドキュメントでは、`sd-scripts`リポジトリに含まれる`sd3_train_network.py`を使用して、Stable Diffusion 3 (SD3) および Stable Diffusion 3.5 (SD3.5) モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

## 1. はじめに

`sd3_train_network.py`は、Stable Diffusion 3/3.5モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。SD3は、MMDiT (Multi-Modal Diffusion Transformer) と呼ばれる新しいアーキテクチャを採用しており、従来のStable Diffusionモデルとは構造が異なります。このスクリプトを使用することで、SD3/3.5モデルに特化したLoRAモデルを作成できます。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sdxl_train_network.py`](sdxl_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

*   `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
*   学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）
*   学習対象のSD3/3.5モデルファイルが準備できていること。

## 2. `train_network.py` との違い

`sd3_train_network.py`は`train_network.py`をベースに、SD3/3.5モデルに対応するための変更が加えられています。主な違いは以下の通りです。

*   **対象モデル:** Stable Diffusion 3, 3.5 Medium / Large モデルを対象とします。
*   **モデル構造:** U-Netの代わりにMMDiT (Transformerベース) を使用します。Text EncoderとしてCLIP-L, CLIP-G, T5-XXLの三つを使用します。VAEはSDXLと互換性がありません。
*   **引数:** SD3/3.5モデル、Text Encoder群、VAEを指定する引数があります。ただし、単一ファイルの`.safetensors`形式であれば、内部で自動的に分離されるため、個別のパス指定は必須ではありません。
*   **一部引数の非互換性:** Stable Diffusion v1/v2向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）はSD3/3.5の学習では使用されません。
*   **SD3特有の引数:** Text Encoderのアテンションマスクやドロップアウト率、Positional Embeddingの調整（SD3.5向け）、タイムステップのサンプリングや損失の重み付けに関する引数が追加されています。

## 3. 準備

学習を開始する前に、以下のファイルが必要です。

1.  **学習スクリプト:** `sd3_train_network.py`
2.  **SD3/3.5モデルファイル:** 学習のベースとなるSD3/3.5モデルの`.safetensors`ファイル。またText Encoderをそれぞれ対応する引数でパスを指定します。
    * 単一ファイル形式も使用可能です。
3.  **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](link/to/dataset/config/doc)を参照してください）。
    *   例として`my_sd3_dataset_config.toml`を使用します。

## 4. 学習の実行

学習は、ターミナルから`sd3_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、SD3/3.5特有の引数を指定する必要があります。

以下に、基本的なコマンドライン実行例を示します。

```bash
accelerate launch --num_cpu_threads_per_process 1 sd3_train_network.py 
 --pretrained_model_name_or_path="<path to SD3 model>" 
 --clip_l="<path to CLIP-L model>"
 --clip_g="<path to CLIP-G model>" 
 --t5xxl="<path to T5-XXL model>" 
 --dataset_config="my_sd3_dataset_config.toml" 
 --output_dir="<output directory for training results>" 
 --output_name="my_sd3_lora" 
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
 --weighting_scheme="sigma_sqrt" 
 --blocks_to_swap=32
```

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。

### 4.1. 主要なコマンドライン引数の解説（`train_network.py`からの追加・変更点）

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のSD3/3.5特有の引数を指定します。共通の引数（`--output_dir`, `--output_name`, `--network_module`, `--network_dim`, `--network_alpha`, `--learning_rate`など）については、上記ガイドを参照してください。

#### モデル関連

*   `--pretrained_model_name_or_path="<path to SD3 model>"` **[必須]**
    *   学習のベースとなるSD3/3.5モデルの`.safetensors`ファイルのパスを指定します。
*   `--clip_l`, `--clip_g`, `--t5xxl`, `--vae`:
    *   ベースモデルが単一ファイル形式の場合、これらの指定は不要です（自動的にモデル内部から読み込まれます）。
    *   Text Encoderが別ファイルとして提供されている場合は、それぞれの`.safetensors`ファイルのパスを指定します。`--vae` はベースモデルに含まれているため、通常は指定する必要はありません（明示的に異なるVAEを使用する場合のみ指定）。

#### SD3/3.5 学習パラメータ

*   `--t5xxl_max_token_length=<integer>`
    *   T5-XXL Text Encoderで使用するトークンの最大長を指定します。SD3のデフォルトは`256`です。データセットのキャプション長に合わせて調整が必要な場合があります。
*   `--apply_lg_attn_mask`
    *   CLIP-LおよびCLIP-Gの出力に対して、パディングトークンに対応するアテンションマスク（ゼロ埋め）を適用します。
*   `--apply_t5_attn_mask`
    *   T5-XXLの出力に対して、パディングトークンに対応するアテンションマスク（ゼロ埋め）を適用します。
*   `--clip_l_dropout_rate`, `--clip_g_dropout_rate`, `--t5_dropout_rate`:
    *   各Text Encoderの出力に対して、指定した確率でドロップアウト（出力をゼロにする）を適用します。過学習の抑制に役立つ場合があります。デフォルトは`0.0`（ドロップアウトなし）です。
*   `--pos_emb_random_crop_rate=<float>` **[SD3.5向け]**
    *   MMDiTのPositional Embeddingに対してランダムクロップを適用する確率を指定します。[SD3.5M model card](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) で説明されています。デフォルトは`0.0`です。
*   `--enable_scaled_pos_embed` **[SD3.5向け]** **[実験的機能]**
    *   マルチ解像度学習時に、解像度に応じてPositional Embeddingをスケーリングします。デフォルトは`False`です。通常は指定不要です。
*   `--training_shift=<float>`
    *   学習時のタイムステップ（ノイズレベル）の分布を調整するためのシフト値です。`weighting_scheme`に加えて適用されます。`1.0`より大きい値はノイズの大きい（構造寄り）領域を、小さい値はノイズの小さい（詳細寄り）領域を重視する傾向になります。デフォルトは`1.0`です。通常はデフォルト値で問題ありません。
*   `--weighting_scheme=<choice>`
    *   損失計算時のタイムステップ（ノイズレベル）に応じた重み付け方法を指定します。`sigma_sqrt`, `logit_normal`, `mode`, `cosmap`, `uniform` (または`none`) から選択します。SD3の論文では`sigma_sqrt`が使用されています。デフォルトは`uniform`です。通常はデフォルト値で問題ありません。
*   `--logit_mean`, `--logit_std`, `--mode_scale`:
    *   `weighting_scheme`で`logit_normal`または`mode`を選択した場合に、その分布を制御するためのパラメータです。通常はデフォルト値で問題ありません。

#### メモリ・速度関連

*   `--blocks_to_swap=<integer>` **[実験的機能]**
    *   VRAM使用量を削減するために、モデルの一部（MMDiTのTransformerブロック）をCPUとGPU間でスワップする設定です。スワップするブロック数を整数で指定します（例: `32`）。値を大きくするとVRAM使用量は減りますが、学習速度は低下します。GPUのVRAM容量に応じて調整してください。`gradient_checkpointing`と併用可能です。
    *   `--cpu_offload_checkpointing`とは併用できません。

#### 非互換・非推奨の引数

*   `--v2`, `--v_parameterization`, `--clip_skip`: Stable Diffusion v1/v2特有の引数のため、SD3/3.5学習では使用されません。

### 4.2. 学習の開始

必要な引数を設定し、コマンドを実行すると学習が開始されます。基本的な流れやログの確認方法は[`train_network.py`のガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

## 5. 学習済みモデルの利用

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_sd3_lora.safetensors`）が保存されます。このファイルは、SD3/3.5モデルに対応した推論環境（例: ComfyUIなど）で使用できます。

## 6. その他

`sd3_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python sd3_train_network.py --help`) を参照してください。
