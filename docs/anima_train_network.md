# LoRA Training Guide for Anima using `anima_train_network.py` / `anima_train_network.py` を用いたAnima モデルのLoRA学習ガイド

This document explains how to train LoRA (Low-Rank Adaptation) models for Anima using `anima_train_network.py` in the `sd-scripts` repository.

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts`リポジトリに含まれる`anima_train_network.py`を使用して、Anima モデルに対するLoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

</details>

## 1. Introduction / はじめに

`anima_train_network.py` trains additional networks such as LoRA for Anima models. Anima adopts a DiT (Diffusion Transformer) architecture based on the MiniTrainDIT design with Rectified Flow training. It uses a Qwen3-0.6B text encoder, an LLM Adapter (6-layer transformer bridge from Qwen3 to T5-compatible space), and a WanVAE (16-channel, 8x spatial downscale).

This guide assumes you already understand the basics of LoRA training. For common usage and options, see the [train_network.py guide](train_network.md). Some parameters are similar to those in [`sd3_train_network.py`](sd3_train_network.md) and [`flux_train_network.py`](flux_train_network.md).

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](./config_README-en.md).
* Anima model files for training are available.

<details>
<summary>日本語</summary>

`anima_train_network.py`は、Anima モデルに対してLoRAなどの追加ネットワークを学習させるためのスクリプトです。AnimaはMiniTrainDIT設計に基づくDiT (Diffusion Transformer) アーキテクチャを採用しており、Rectified Flow学習を使用します。テキストエンコーダーとしてQwen3-0.6B、LLM Adapter (Qwen3からT5互換空間への6層Transformerブリッジ)、およびWanVAE (16チャンネル、8倍空間ダウンスケール) を使用します。

このガイドは、基本的なLoRA学習の手順を理解しているユーザーを対象としています。基本的な使い方や共通のオプションについては、[`train_network.py`のガイド](train_network.md)を参照してください。また一部のパラメータは [`sd3_train_network.py`](sd3_train_network.md) や [`flux_train_network.py`](flux_train_network.md) と同様のものがあるため、そちらも参考にしてください。

**前提条件:**

* `sd-scripts`リポジトリのクローンとPython環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（データセットの準備については[データセット設定ガイド](./config_README-en.md)を参照してください）
* 学習対象のAnimaモデルファイルが準備できていること。
</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`anima_train_network.py` is based on `train_network.py` but modified for Anima . Main differences are:

* **Target models:** Anima DiT models.
* **Model structure:** Uses a MiniTrainDIT (Transformer based) instead of U-Net. Employs a single text encoder (Qwen3-0.6B), an LLM Adapter that bridges Qwen3 embeddings to T5-compatible cross-attention space, and a WanVAE (16-channel latent space with 8x spatial downscale).
* **Arguments:** Options exist to specify the Anima DiT model, Qwen3 text encoder, WanVAE, LLM adapter, and T5 tokenizer separately.
* **Incompatible arguments:** Stable Diffusion v1/v2 options such as `--v2`, `--v_parameterization` and `--clip_skip` are not used.
* **Anima specific options:** Additional parameters for component-wise learning rates (self_attn, cross_attn, mlp, mod, llm_adapter), timestep sampling, discrete flow shift, and flash attention.
* **6 Parameter Groups:** Independent learning rates for `base`, `self_attn`, `cross_attn`, `mlp`, `adaln_modulation`, and `llm_adapter` components.

<details>
<summary>日本語</summary>

`anima_train_network.py`は`train_network.py`をベースに、Anima モデルに対応するための変更が加えられています。主な違いは以下の通りです。

* **対象モデル:** Anima DiTモデルを対象とします。
* **モデル構造:** U-Netの代わりにMiniTrainDIT (Transformerベース) を使用します。テキストエンコーダーとしてQwen3-0.6B、Qwen3埋め込みをT5互換のクロスアテンション空間に変換するLLM Adapter、およびWanVAE (16チャンネル潜在空間、8倍空間ダウンスケール) を使用します。
* **引数:** Anima DiTモデル、Qwen3テキストエンコーダー、WanVAE、LLM Adapter、T5トークナイザーを個別に指定する引数があります。
* **一部引数の非互換性:** Stable Diffusion v1/v2向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）はAnimaの学習では使用されません。
* **Anima特有の引数:** コンポーネント別学習率（self_attn, cross_attn, mlp, mod, llm_adapter）、タイムステップサンプリング、離散フローシフト、Flash Attentionに関する引数が追加されています。
* **6パラメータグループ:** `base`、`self_attn`、`cross_attn`、`mlp`、`adaln_modulation`、`llm_adapter`の各コンポーネントに対して独立した学習率を設定できます。
</details>

## 3. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `anima_train_network.py`
2. **Anima DiT model file:** `.safetensors` file for the base DiT model.
3. **Qwen3-0.6B text encoder:** Either a HuggingFace model directory or a single `.safetensors` file (requires `configs/qwen3_06b/` config files).
4. **WanVAE model file:** `.safetensors` or `.pth` file for the VAE.
5. **LLM Adapter model file (optional):** `.safetensors` file. If not provided separately, the adapter is loaded from the DiT file if the key `llm_adapter.out_proj.weight` exists.
6. **T5 Tokenizer (optional):** If not specified, uses the bundled tokenizer at `configs/t5_old/`.
7. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](./config_README-en.md).) In this document we use `my_anima_dataset_config.toml` as an example.

**Notes:**
* When using a single `.safetensors` file for Qwen3, download the `config.json`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.json` from the [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) HuggingFace repository into the `configs/qwen3_06b/` directory.
* The T5 tokenizer only needs the tokenizer files (not the T5 model weights). It uses the vocabulary from `google/t5-v1_1-xxl`.
* Models are saved with a `net.` prefix on all keys for ComfyUI compatibility.

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `anima_train_network.py`
2. **Anima DiTモデルファイル:** ベースとなるDiTモデルの`.safetensors`ファイル。
3. **Qwen3-0.6Bテキストエンコーダー:** HuggingFaceモデルディレクトリまたは単体の`.safetensors`ファイル（`configs/qwen3_06b/`の設定ファイルが必要）。
4. **WanVAEモデルファイル:** VAEの`.safetensors`または`.pth`ファイル。
5. **LLM Adapterモデルファイル（オプション）:** `.safetensors`ファイル。個別に指定しない場合、DiTファイル内に`llm_adapter.out_proj.weight`キーが存在すればそこから読み込まれます。
6. **T5トークナイザー（オプション）:** 指定しない場合、`configs/t5_old/`のバンドル版トークナイザーを使用します。
7. **データセット定義ファイル (.toml):** 学習データセットの設定を記述したTOML形式のファイル。（詳細は[データセット設定ガイド](./config_README-en.md)を参照してください）。例として`my_anima_dataset_config.toml`を使用します。

**注意:**
* Qwen3の単体`.safetensors`ファイルを使用する場合、[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) HuggingFaceリポジトリから`config.json`、`tokenizer.json`、`tokenizer_config.json`、`vocab.json`をダウンロードし、`configs/qwen3_06b/`ディレクトリに配置してください。
* T5トークナイザーはトークナイザーファイルのみ必要です（T5モデルの重みは不要）。`google/t5-v1_1-xxl`の語彙を使用します。
* モデルはComfyUI互換のため、すべてのキーに`net.`プレフィックスを付けて保存されます。
</details>

## 4. Running the Training / 学習の実行

Execute `anima_train_network.py` from the terminal to start training. The overall command-line format is the same as `train_network.py`, but Anima specific options must be supplied.

Example command:

```bash
accelerate launch --num_cpu_threads_per_process 1 anima_train_network.py \
  --dit_path="<path to Anima DiT model>" \
  --qwen3_path="<path to Qwen3-0.6B model or directory>" \
  --vae_path="<path to WanVAE model>" \
  --llm_adapter_path="<path to LLM adapter model>" \
  --dataset_config="my_anima_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_anima_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_anima \
  --network_dim=8 \
  --network_alpha=8 \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --timestep_sample_method="logit_normal" \
  --discrete_flow_shift=3.0 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_text_encoder_outputs \
  --blocks_to_swap=18
```

*(Write the command on one line or use `\` or `^` for line breaks.)*

<details>
<summary>日本語</summary>

学習は、ターミナルから`anima_train_network.py`を実行することで開始します。基本的なコマンドラインの構造は`train_network.py`と同様ですが、Anima特有の引数を指定する必要があります。

コマンドラインの例は英語のドキュメントを参照してください。

※実際には1行で書くか、適切な改行文字（`\` または `^`）を使用してください。
</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

Besides the arguments explained in the [train_network.py guide](train_network.md), specify the following Anima specific options. For shared options (`--output_dir`, `--output_name`, `--network_module`, etc.), see that guide.

#### Model Options [Required] / モデル関連 [必須]

* `--dit_path="<path to Anima DiT model>"` **[Required]**
  - Path to the Anima DiT model `.safetensors` file. The model config (channels, blocks, heads) is auto-detected from the state dict. ComfyUI format with `net.` prefix is supported.
* `--qwen3_path="<path to Qwen3-0.6B model>"` **[Required]**
  - Path to the Qwen3-0.6B text encoder. Can be a HuggingFace model directory or a single `.safetensors` file. The text encoder is always frozen during training.
* `--vae_path="<path to WanVAE model>"` **[Required]**
  - Path to the WanVAE model `.safetensors` or `.pth` file. Fixed config: `dim=96, z_dim=16`.
* `--llm_adapter_path="<path to LLM adapter>"` *[Optional]*
  - Path to a separate LLM adapter weights file. If omitted, the adapter is loaded from the DiT file when the key `llm_adapter.out_proj.weight` exists.
* `--t5_tokenizer_path="<path to T5 tokenizer>"` *[Optional]*
  - Path to the T5 tokenizer directory. If omitted, uses the bundled config at `configs/t5_old/`.

#### Anima Training Parameters / Anima 学習パラメータ

* `--timestep_sample_method=<choice>`
  - Timestep sampling method. Choose from `logit_normal` (default) or `uniform`.
* `--discrete_flow_shift=<float>`
  - Shift for the timestep distribution in Rectified Flow training. Default `3.0`. The shift formula is `t_shifted = (t * shift) / (1 + (shift - 1) * t)`.
* `--sigmoid_scale=<float>`
  - Scale factor for `logit_normal` timestep sampling. Default `1.0`.
* `--qwen3_max_token_length=<integer>`
  - Maximum token length for the Qwen3 tokenizer. Default `512`.
* `--t5_max_token_length=<integer>`
  - Maximum token length for the T5 tokenizer. Default `512`.
* `--apply_t5_attn_mask`
  - Apply attention mask to T5 tokens in the LLM adapter.
* `--flash_attn`
  - Use Flash Attention for DiT self/cross-attention. Requires `pip install flash-attn`. Falls back to PyTorch SDPA if the package is not installed. Note: Flash Attention is only applied to DiT blocks; the LLM Adapter uses standard attention because it requires attention masks.
* `--transformer_dtype=<choice>`
  - Separate dtype for transformer blocks. Choose from `float16`, `bfloat16`, `float32`. If not specified, uses the same dtype as `--mixed_precision`.

#### Component-wise Learning Rates / コンポーネント別学習率

Anima supports 6 independent learning rate groups. Set to `0` to freeze a component:

* `--self_attn_lr=<float>` - Learning rate for self-attention layers. Default: same as `--learning_rate`.
* `--cross_attn_lr=<float>` - Learning rate for cross-attention layers. Default: same as `--learning_rate`.
* `--mlp_lr=<float>` - Learning rate for MLP layers. Default: same as `--learning_rate`.
* `--mod_lr=<float>` - Learning rate for AdaLN modulation layers. Default: same as `--learning_rate`.
* `--llm_adapter_lr=<float>` - Learning rate for LLM adapter layers. Default: same as `--learning_rate`.

#### Memory and Speed / メモリ・速度関連

* `--blocks_to_swap=<integer>` **[Experimental]**
  - Number of Transformer blocks to swap between CPU and GPU. More blocks reduce VRAM but slow training. Maximum values depend on model size:
    - 28-block model: max **26**
    - 36-block model: max **34**
    - 20-block model: max **18**
  - Cannot be used with `--cpu_offload_checkpointing` or `--unsloth_offload_checkpointing`.
* `--unsloth_offload_checkpointing`
  - Offload activations to CPU RAM using async non-blocking transfers. Faster than `--cpu_offload_checkpointing`. Cannot be combined with `--cpu_offload_checkpointing` or `--blocks_to_swap`.
* `--cache_text_encoder_outputs`
  - Cache Qwen3 text encoder outputs to reduce VRAM usage. Recommended when not training text encoder LoRA.
* `--cache_text_encoder_outputs_to_disk`
  - Cache text encoder outputs to disk. Auto-enables `--cache_text_encoder_outputs`.
* `--cache_latents`, `--cache_latents_to_disk`
  - Cache WanVAE latent outputs.
* `--fp8_base`
  - Use FP8 precision for the base model to reduce VRAM usage.

#### Incompatible or Deprecated Options / 非互換・非推奨の引数

* `--v2`, `--v_parameterization`, `--clip_skip` - Options for Stable Diffusion v1/v2 that are not used for Anima training.

<details>
<summary>日本語</summary>

[`train_network.py`のガイド](train_network.md)で説明されている引数に加え、以下のAnima特有の引数を指定します。共通の引数については、上記ガイドを参照してください。

#### モデル関連 [必須]

* `--dit_path="<path to Anima DiT model>"` **[必須]** - Anima DiTモデルの`.safetensors`ファイルのパスを指定します。
* `--qwen3_path="<path to Qwen3-0.6B model>"` **[必須]** - Qwen3-0.6Bテキストエンコーダーのパスを指定します。
* `--vae_path="<path to WanVAE model>"` **[必須]** - WanVAEモデルのパスを指定します。
* `--llm_adapter_path="<path to LLM adapter>"` *[オプション]* - 個別のLLM Adapterの重みファイルのパス。
* `--t5_tokenizer_path="<path to T5 tokenizer>"` *[オプション]* - T5トークナイザーディレクトリのパス。

#### Anima 学習パラメータ

* `--timestep_sample_method` - タイムステップのサンプリング方法。`logit_normal`（デフォルト）または`uniform`。
* `--discrete_flow_shift` - Rectified Flow学習のタイムステップ分布シフト。デフォルト`3.0`。
* `--sigmoid_scale` - logit_normalタイムステップサンプリングのスケール係数。デフォルト`1.0`。
* `--qwen3_max_token_length` - Qwen3トークナイザーの最大トークン長。デフォルト`512`。
* `--t5_max_token_length` - T5トークナイザーの最大トークン長。デフォルト`512`。
* `--apply_t5_attn_mask` - LLM AdapterでT5トークンにアテンションマスクを適用。
* `--flash_attn` - DiTのself/cross-attentionにFlash Attentionを使用。`pip install flash-attn`が必要。
* `--transformer_dtype` - Transformerブロック用の個別dtype。

#### コンポーネント別学習率

Animaは6つの独立した学習率グループをサポートします。`0`に設定するとそのコンポーネントをフリーズします：

* `--self_attn_lr` - Self-attention層の学習率。
* `--cross_attn_lr` - Cross-attention層の学習率。
* `--mlp_lr` - MLP層の学習率。
* `--mod_lr` - AdaLNモジュレーション層の学習率。
* `--llm_adapter_lr` - LLM Adapter層の学習率。

#### メモリ・速度関連

* `--blocks_to_swap` **[実験的機能]** - TransformerブロックをCPUとGPUでスワップしてVRAMを節約。
* `--unsloth_offload_checkpointing` - 非同期転送でアクティベーションをCPU RAMにオフロード。
* `--cache_text_encoder_outputs` - Qwen3の出力をキャッシュしてメモリ使用量を削減。
* `--cache_latents`, `--cache_latents_to_disk` - WanVAEの出力をキャッシュ。
* `--fp8_base` - ベースモデルにFP8精度を使用。
</details>

### 4.2. Starting Training / 学習の開始

After setting the required arguments, run the command to begin training. The overall flow and how to check logs are the same as in the [train_network.py guide](train_network.md#32-starting-the-training--学習の開始).

<details>
<summary>日本語</summary>

必要な引数を設定したら、コマンドを実行して学習を開始します。全体の流れやログの確認方法は、[train_network.pyのガイド](train_network.md#32-starting-the-training--学習の開始)と同様です。

</details>

## 5. LoRA Target Modules / LoRAの学習対象モジュール

When training LoRA with `anima_train_network.py`, the following modules are targeted:

* **DiT Blocks (`Block`)**: Self-attention, cross-attention, MLP, and AdaLN modulation layers within each transformer block.
* **LLM Adapter Blocks (`LLMAdapterTransformerBlock`)**: Only when `--network_args "train_llm_adapter=True"` is specified.
* **Text Encoder (Qwen3)**: Only when `--network_train_unet_only` is NOT specified.

The LoRA network module is `networks.lora_anima`.

### 5.1. Layer-specific Rank Configuration / 各層に対するランク指定

You can specify different ranks (network_dim) for each component of the Anima model. Setting `0` disables LoRA for that component.

| network_args | Target Component |
|---|---|
| `self_attn_dim` | Self-attention layers in DiT blocks |
| `cross_attn_dim` | Cross-attention layers in DiT blocks |
| `mlp_dim` | MLP layers in DiT blocks |
| `mod_dim` | AdaLN modulation layers in DiT blocks |
| `llm_adapter_dim` | LLM adapter layers (requires `train_llm_adapter=True`) |

Example usage:
```
--network_args "self_attn_dim=8" "cross_attn_dim=4" "mlp_dim=8" "mod_dim=4"
```

### 5.2. Embedding Layer LoRA / 埋め込み層LoRA

You can apply LoRA to embedding/output layers by specifying `emb_dims` in network_args as a comma-separated list of 3 numbers:

```
--network_args "emb_dims=[8,4,8]"
```

Each number corresponds to:
1. `x_embedder` (patch embedding)
2. `t_embedder` (timestep embedding)
3. `final_layer` (output layer)

Setting `0` disables LoRA for that layer.

### 5.3. Block Selection for Training / 学習するブロックの指定

You can specify which DiT blocks to train using `train_block_indices` in network_args. The indices are 0-based. Default is to train all blocks.

Specify indices as comma-separated integers or ranges:

```
--network_args "train_block_indices=0-5,10,15-27"
```

Special values: `all` (train all blocks), `none` (skip all blocks).

### 5.4. LLM Adapter LoRA / LLM Adapter LoRA

To apply LoRA to the LLM Adapter blocks:

```
--network_args "train_llm_adapter=True" "llm_adapter_dim=4"
```

### 5.5. Other Network Args / その他のネットワーク引数

* `--network_args "verbose=True"` - Print all LoRA module names and their dimensions.
* `--network_args "rank_dropout=0.1"` - Rank dropout rate.
* `--network_args "module_dropout=0.1"` - Module dropout rate.
* `--network_args "loraplus_lr_ratio=2.0"` - LoRA+ learning rate ratio.
* `--network_args "loraplus_unet_lr_ratio=2.0"` - LoRA+ learning rate ratio for DiT only.
* `--network_args "loraplus_text_encoder_lr_ratio=2.0"` - LoRA+ learning rate ratio for text encoder only.

<details>
<summary>日本語</summary>

`anima_train_network.py`でLoRAを学習させる場合、デフォルトでは以下のモジュールが対象となります。

* **DiTブロック (`Block`)**: 各Transformerブロック内のSelf-attention、Cross-attention、MLP、AdaLNモジュレーション層。
* **LLM Adapterブロック (`LLMAdapterTransformerBlock`)**: `--network_args "train_llm_adapter=True"`を指定した場合のみ。
* **テキストエンコーダー (Qwen3)**: `--network_train_unet_only`を指定しない場合のみ。

### 5.1. 各層のランクを指定する

`--network_args`で各コンポーネントに異なるランクを指定できます。`0`を指定するとその層にはLoRAが適用されません。

|network_args|対象コンポーネント|
|---|---|
|`self_attn_dim`|DiTブロック内のSelf-attention層|
|`cross_attn_dim`|DiTブロック内のCross-attention層|
|`mlp_dim`|DiTブロック内のMLP層|
|`mod_dim`|DiTブロック内のAdaLNモジュレーション層|
|`llm_adapter_dim`|LLM Adapter層（`train_llm_adapter=True`が必要）|

### 5.2. 埋め込み層LoRA

`emb_dims`で埋め込み/出力層にLoRAを適用できます。3つの数値をカンマ区切りで指定します。

各数値は `x_embedder`（パッチ埋め込み）、`t_embedder`（タイムステップ埋め込み）、`final_layer`（出力層）に対応します。

### 5.3. 学習するブロックの指定

`train_block_indices`でLoRAを適用するDiTブロックを指定できます。

### 5.4. LLM Adapter LoRA

LLM AdapterブロックにLoRAを適用するには：`--network_args "train_llm_adapter=True" "llm_adapter_dim=4"`

### 5.5. その他のネットワーク引数

* `verbose=True` - 全LoRAモジュール名とdimを表示
* `rank_dropout` - ランクドロップアウト率
* `module_dropout` - モジュールドロップアウト率
* `loraplus_lr_ratio` - LoRA+学習率比率

</details>

## 6. Using the Trained Model / 学習済みモデルの利用

When training finishes, a LoRA model file (e.g. `my_anima_lora.safetensors`) is saved in the directory specified by `output_dir`. Use this file with inference environments that support Anima , such as ComfyUI with appropriate nodes.

<details>
<summary>日本語</summary>

学習が完了すると、指定した`output_dir`にLoRAモデルファイル（例: `my_anima_lora.safetensors`）が保存されます。このファイルは、Anima モデルに対応した推論環境（例: ComfyUI + 適切なノード）で使用できます。

</details>

## 7. Advanced Settings / 高度な設定

### 7.1. VRAM Usage Optimization / VRAM使用量の最適化

Anima models can be large, so GPUs with limited VRAM may require optimization:

#### Key VRAM Reduction Options

- **`--fp8_base`**: Enables training in FP8 format for the DiT model.

- **`--blocks_to_swap <number>`**: Swaps blocks between CPU and GPU to reduce VRAM usage. Higher numbers save more VRAM but reduce training speed. See model-specific max values in section 4.1.

- **`--unsloth_offload_checkpointing`**: Offloads gradient checkpoints to CPU using async non-blocking transfers. Faster than `--cpu_offload_checkpointing`. Cannot be combined with `--blocks_to_swap`.

- **`--gradient_checkpointing`**: Standard gradient checkpointing to reduce VRAM at the cost of compute.

- **`--cache_text_encoder_outputs`**: Caches Qwen3 outputs so the text encoder can be freed from VRAM during training.

- **`--cache_latents`**: Caches WanVAE outputs so the VAE can be freed from VRAM during training.

- **Using Adafactor optimizer**: Can reduce VRAM usage:
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

<details>
<summary>日本語</summary>

Animaモデルは大きい場合があるため、VRAMが限られたGPUでは最適化が必要です。

主要なVRAM削減オプション：
- `--fp8_base`: FP8形式での学習を有効化
- `--blocks_to_swap`: CPUとGPU間でブロックをスワップ
- `--unsloth_offload_checkpointing`: 非同期転送でアクティベーションをCPUにオフロード
- `--gradient_checkpointing`: 標準的な勾配チェックポイント
- `--cache_text_encoder_outputs`: Qwen3の出力をキャッシュ
- `--cache_latents`: WanVAEの出力をキャッシュ
- Adafactorオプティマイザの使用

</details>

### 7.2. Training Settings / 学習設定

#### Timestep Sampling

The `--timestep_sample_method` option specifies how timesteps (0-1) are sampled:

- `logit_normal` (default): Sample from Normal(0,1), multiply by `sigmoid_scale`, apply sigmoid. Good general-purpose option.
- `uniform`: Uniform random sampling from [0, 1].

#### Discrete Flow Shift

The `--discrete_flow_shift` option (default `3.0`) shifts the timestep distribution toward higher noise levels. The formula is:

```
t_shifted = (t * shift) / (1 + (shift - 1) * t)
```

Timesteps are clamped to `[1e-5, 1-1e-5]` after shifting.

#### Loss Weighting

The `--weighting_scheme` option specifies loss weighting by timestep:

- `uniform` (default): Equal weight for all timesteps.
- `sigma_sqrt`: Weight by `sigma^(-2)`.
- `cosmap`: Weight by `2 / (pi * (1 - 2*sigma + 2*sigma^2))`.
- `none`: Same as uniform.

#### Caption Dropout

Use `--caption_dropout_rate` for embedding-level caption dropout. This is handled by `AnimaTextEncodingStrategy` and is compatible with text encoder output caching. The subset-level `caption_dropout_rate` is automatically zeroed when this is set.

<details>
<summary>日本語</summary>

#### タイムステップサンプリング

`--timestep_sample_method`でタイムステップのサンプリング方法を指定します：
- `logit_normal`（デフォルト）: 正規分布からサンプリングし、sigmoidを適用。
- `uniform`: [0, 1]の一様分布からサンプリング。

#### 離散フローシフト

`--discrete_flow_shift`（デフォルト`3.0`）はタイムステップ分布を高ノイズ側にシフトします。

#### 損失の重み付け

`--weighting_scheme`でタイムステップごとの損失の重み付けを指定します。

#### キャプションドロップアウト

`--caption_dropout_rate`で埋め込みレベルのキャプションドロップアウトを使用します。テキストエンコーダー出力のキャッシュと互換性があります。

</details>

### 7.3. Text Encoder LoRA Support / Text Encoder LoRAのサポート

Anima LoRA training supports training Qwen3 text encoder LoRA:

- To train only DiT: specify `--network_train_unet_only`
- To train DiT and Qwen3: omit `--network_train_unet_only`

You can specify a separate learning rate for Qwen3 with `--text_encoder_lr`. If not specified, the default `--learning_rate` is used.

<details>
<summary>日本語</summary>

Anima LoRA学習では、Qwen3テキストエンコーダーのLoRAもトレーニングできます。

- DiTのみ学習: `--network_train_unet_only`を指定
- DiTとQwen3を学習: `--network_train_unet_only`を省略

</details>

## 8. Other Training Options / その他の学習オプション

- **`--loss_type`**: Loss function for training. Default `l2`.
  - `l1`: L1 loss.
  - `l2`: L2 loss (mean squared error).
  - `huber`: Huber loss.
  - `smooth_l1`: Smooth L1 loss.

- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: Parameters for Huber loss when `--loss_type` is `huber` or `smooth_l1`.

- **`--ip_noise_gamma`**, **`--ip_noise_gamma_random_strength`**: Input Perturbation noise gamma values.

- **`--fused_backward_pass`**: Fuses the backward pass and optimizer step to reduce VRAM usage. Only works with Adafactor. For details, see the [`sdxl_train_network.py` guide](sdxl_train_network.md).

- **`--weighting_scheme`**, **`--logit_mean`**, **`--logit_std`**, **`--mode_scale`**: Timestep loss weighting options. For details, refer to the [`sd3_train_network.md` guide](sd3_train_network.md).

<details>
<summary>日本語</summary>

- **`--loss_type`**: 学習に用いる損失関数。デフォルト`l2`。`l1`, `l2`, `huber`, `smooth_l1`から選択。
- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: Huber損失のパラメータ。
- **`--ip_noise_gamma`**: Input Perturbationノイズガンマ値。
- **`--fused_backward_pass`**: バックワードパスとオプティマイザステップの融合。
- **`--weighting_scheme`** 等: タイムステップ損失の重み付け。詳細は[`sd3_train_network.md`](sd3_train_network.md)を参照。

</details>

## 9. Others / その他

### Metadata Saved in LoRA Models

The following Anima-specific metadata is saved in the LoRA model file:

* `ss_apply_t5_attn_mask`
* `ss_weighting_scheme`
* `ss_discrete_flow_shift`
* `ss_timestep_sample_method`
* `ss_sigmoid_scale`

<details>
<summary>日本語</summary>

`anima_train_network.py`には、サンプル画像の生成 (`--sample_prompts`など) や詳細なオプティマイザ設定など、`train_network.py`と共通の機能も多く存在します。これらについては、[`train_network.py`のガイド](train_network.md#5-other-features--その他の機能)やスクリプトのヘルプ (`python anima_train_network.py --help`) を参照してください。

### LoRAモデルに保存されるメタデータ

以下のAnima固有のメタデータがLoRAモデルファイルに保存されます：

* `ss_apply_t5_attn_mask`
* `ss_weighting_scheme`
* `ss_discrete_flow_shift`
* `ss_timestep_sample_method`
* `ss_sigmoid_scale`

</details>
