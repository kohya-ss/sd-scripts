# LoRA Training Guide for MOSS-Image using `moss_image_train_network.py` / `moss_image_train_network.py` を用いた MOSS-Image モデルの LoRA 学習ガイド

This document explains how to train LoRA (Low-Rank Adaptation) models for MOSS-Image using `moss_image_train_network.py` in the `sd-scripts` repository.

<details>
<summary>日本語</summary>

このドキュメントでは、`sd-scripts` リポジトリに含まれる `moss_image_train_network.py` を使用して、MOSS-Image モデルに対する LoRA (Low-Rank Adaptation) モデルを学習する基本的な手順について解説します。

</details>

## 1. Introduction / はじめに

`moss_image_train_network.py` trains additional networks such as LoRA for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) models (OpenMOSS). MOSS-Image is a 5B single-stream DiT (Diffusion Transformer, 42 blocks) trained with Rectified Flow. It uses Qwen3.5-9B as the text encoder (the last hidden state, 4096 dimensions, is fed to the DiT) and the FLUX.2 VAE (32-channel latent space, 8x spatial downscale). The DiT patchifies the latents 2x2, so the effective downscale of the image is 16x.

This guide assumes you already understand the basics of LoRA training. For common usage and options, see the [train_network.py guide](train_network.md). Many options are shared with [`anima_train_network.py`](anima_train_network.md) and [`flux_train_network.py`](flux_train_network.md).

**Prerequisites:**

* The `sd-scripts` repository has been cloned and the Python environment is ready.
* A training dataset has been prepared. See the [Dataset Configuration Guide](./config_README-en.md).
* MOSS-Image model files for training are available.

<details>
<summary>日本語</summary>

`moss_image_train_network.py` は、[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)（OpenMOSS）モデルに対して LoRA などの追加ネットワークを学習させるためのスクリプトです。MOSS-Image は Rectified Flow で学習された 5B の single-stream DiT (Diffusion Transformer、42 ブロック) です。テキストエンコーダーとして Qwen3.5-9B（最終層の hidden state、4096 次元を DiT に入力）、VAE として FLUX.2 VAE（32 チャンネル潜在空間、8 倍空間ダウンスケール）を使用します。DiT は潜在変数を 2x2 で patchify するため、画像に対する実効的なダウンスケールは 16 倍です。

このガイドは、LoRA 学習の基本を理解している方を対象としています。共通の使い方やオプションについては [train_network.py のガイド](train_network.md) を参照してください。多くのオプションは [`anima_train_network.py`](anima_train_network.md) や [`flux_train_network.py`](flux_train_network.md) と共通です。

**前提条件:**

* `sd-scripts` リポジトリのクローンと Python 環境のセットアップが完了していること。
* 学習用データセットの準備が完了していること。（[データセット設定ガイド](./config_README-ja.md) を参照してください。）
* 学習対象の MOSS-Image モデルファイルが準備できていること。

</details>

## 2. Differences from `train_network.py` / `train_network.py` との違い

`moss_image_train_network.py` is based on `train_network.py` but modified for MOSS-Image. Main differences are:

* **Target models:** MOSS-Image DiT models (Base and SFT).
* **Model structure:** Uses a single-stream DiT instead of U-Net. Employs a single text encoder (Qwen3.5-9B) and the FLUX.2 VAE. The text encoder is bf16 and the VAE is fp32 regardless of `--mixed_precision`.
* **Arguments:** Uses the common `--pretrained_model_name_or_path` for the DiT model path, `--text_encoder` for the Qwen3.5-9B text encoder, and `--vae` for the FLUX.2 VAE.
* **Incompatible arguments:** Stable Diffusion v1/v2 options such as `--v2`, `--v_parameterization` and `--clip_skip` are not used. `--fp8_base` / `--fp8_base_unet` are not supported; use ConvRot INT8 (`--convrot_int8`) instead.
* **Timestep sampling:** Uses the same `--timestep_sampling` options as FLUX / Anima training (`sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`). The pre-training distribution of MOSS-Image can be reproduced with existing options (see [Section 7.2](#72-training-settings--学習設定)).
* **LoRA:** Uses the generic LoRA module `networks.lora_generic` with regex-based module selection and per-module rank/learning rate control (`network_reg_dims`, `network_reg_lrs`). Module exclusion/inclusion is controlled by `exclude_patterns` and `include_patterns`.
* **Text Encoder:** The text encoder is run one sample at a time without padding (the Qwen3.5 Gated DeltaNet layers are sensitive to padding), the same as the official pipeline. Captions can contain line breaks (`enable_multiline_caption`).
* **Caption dropout with cached text encoder outputs:** The unconditional embedding is stored in every cache file, so `caption_dropout_rate` works together with `--cache_text_encoder_outputs`.
* **Multiple caption candidates with cached text encoder outputs:** The outputs of all caption candidates of an image (`enable_multiline_caption` with separator lines) are stored in one cache file, and one of them is chosen randomly at each step. See [Section 3](#3-preparation--準備).

<details>
<summary>日本語</summary>

`moss_image_train_network.py` は `train_network.py` をベースに、MOSS-Image モデルに対応するための変更が加えられています。主な違いは以下の通りです。

* **対象モデル:** MOSS-Image の DiT モデル（Base および SFT）を対象とします。
* **モデル構造:** U-Net の代わりに single-stream DiT を使用します。テキストエンコーダーは Qwen3.5-9B の 1 つ、VAE は FLUX.2 VAE です。`--mixed_precision` の指定に関係なく、テキストエンコーダーは bf16、VAE は fp32 で動作します。
* **引数:** DiT モデルのパスには共通引数 `--pretrained_model_name_or_path` を、Qwen3.5-9B テキストエンコーダーには `--text_encoder` を、FLUX.2 VAE には `--vae` を使用します。
* **一部引数の非互換性:** Stable Diffusion v1/v2 向けの引数（例: `--v2`, `--v_parameterization`, `--clip_skip`）は使用されません。`--fp8_base` / `--fp8_base_unet` はサポートされません。代わりに ConvRot INT8（`--convrot_int8`）を使用してください。
* **タイムステップサンプリング:** FLUX / Anima 学習と同じ `--timestep_sampling` オプション（`sigma`、`uniform`、`sigmoid`、`shift`、`flux_shift`）を使用します。MOSS-Image の事前学習の分布は既存オプションの組み合わせで再現できます（[7.2 節](#72-training-settings--学習設定) を参照）。
* **LoRA:** 汎用 LoRA モジュール `networks.lora_generic` を使用し、正規表現ベースのモジュール選択とモジュール単位のランク/学習率制御（`network_reg_dims`、`network_reg_lrs`）を行います。モジュールの除外/包含は `exclude_patterns` と `include_patterns` で制御します。
* **テキストエンコーダー:** 公式パイプラインと同様に、パディングなしで 1 サンプルずつ実行します（Qwen3.5 の Gated DeltaNet 層はパディングの影響を受けます）。キャプションには改行を含められます（`enable_multiline_caption`）。
* **キャッシュ時のキャプションドロップアウト:** 各キャッシュファイルに unconditional の埋め込みも保存するため、`caption_dropout_rate` を `--cache_text_encoder_outputs` と併用できます。
* **キャッシュ時の複数キャプション候補:** 画像の全キャプション候補（`enable_multiline_caption` と区切り行）の出力を一つのキャッシュファイルに保存し、ステップごとにランダムに一つを選びます。[3 節](#3-preparation--準備)を参照してください。

</details>

## 3. Preparation / 準備

The following files are required before starting training:

1. **Training script:** `moss_image_train_network.py`
2. **MOSS-Image DiT model file:** Either of the following. Specify the path with `--pretrained_model_name_or_path`.
    * The official diffusers-format shards (`transformer/diffusion_pytorch_model-0000X-of-00003.safetensors` in the [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) repository). **Specify the first shard** (`...-00001-of-00003.safetensors`); the other shards in the same directory are loaded automatically. `index.json` is not used.
    * The ComfyUI single-file checkpoints (`moss_image_base_bf16.safetensors`, `moss_image_base_int8_convrot.safetensors`, and the `sft` variants). The ConvRot INT8 files are detected automatically (see [Section 7.1](#71-vram-usage-optimization--vram-使用量の最適化)).
3. **Qwen3.5-9B text encoder:** Either of the following. Specify the path with `--text_encoder`.
    * The official shards (`text_encoder/model-0000X-of-00004.safetensors`). **Specify the first shard** (`model-00001-of-00004.safetensors`).
    * The ComfyUI ConvRot INT8 file (`qwen3.5_9b_int8_convrot.safetensors`). The W4A8 GPTQ file (`qwen3.5_9b_w4a8_gptq.safetensors`) is **not** supported.
    * The vision tower of Qwen3.5 is not loaded. The model config is bundled in `configs/moss_image_qwen35_9b/`.
4. **FLUX.2 VAE model file:** The ComfyUI / BFL format `flux2-vae.safetensors` is recommended. The diffusers-format VAE in the official repository (`vae/`, FLUX.2 "small decoder") can also be used; both share the same latent space, so cached latents are interchangeable. Specify the path with `--vae`.
5. **Tokenizer:** Downloaded automatically from the Hugging Face Hub (`OpenMOSS-Team/MOSS-Image-Base`, subfolder `tokenizer`, about 20MB). For offline training, specify `--tokenizer_cache_dir` with either a Hugging Face cache directory or a local directory containing the tokenizer files (`tokenizer_config.json` etc.).
6. **Dataset definition file (.toml):** Dataset settings in TOML format. (See the [Dataset Configuration Guide](./config_README-en.md).) In this document we use `my_moss_dataset_config.toml` as an example.

**Notes:**

* The bucket resolutions must be multiples of 16 (`bucket_reso_steps` of 16, 32, 64, ... The default 64 is fine).
* MOSS-Image is trained with long natural-language captions written by an LLM. `enable_multiline_caption = true` in the dataset config lets you use captions that contain line breaks as they are. See [the dataset config documentation](./config_README-en.md#captions-with-line-breaks-and-multiple-caption-candidates-enable_multiline_caption--true).
* Multiple caption candidates for one image (separator lines with `enable_multiline_caption`, e.g. an original caption and recaptions of different lengths, as in the official training) can be used together with `--cache_text_encoder_outputs`: all candidates are encoded and stored in one cache file, and one of them is chosen randomly at each step. The cache size and the caching time are multiplied by the number of candidates.

<details>
<summary>日本語</summary>

学習を開始する前に、以下のファイルが必要です。

1. **学習スクリプト:** `moss_image_train_network.py`
2. **MOSS-Image DiT モデルファイル:** 以下のいずれか。`--pretrained_model_name_or_path` にパスを指定します。
    * 公式の diffusers 形式の分割ファイル（[OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) リポジトリの `transformer/diffusion_pytorch_model-0000X-of-00003.safetensors`）。**最初のファイル**（`...-00001-of-00003.safetensors`）を指定してください。同じディレクトリの残りのファイルは自動的に読み込まれます。`index.json` は使用しません。
    * ComfyUI 向けの単一ファイル（`moss_image_base_bf16.safetensors`、`moss_image_base_int8_convrot.safetensors`、および `sft` 版）。ConvRot INT8 のファイルは自動判別されます（[7.1 節](#71-vram-usage-optimization--vram-使用量の最適化) を参照）。
3. **Qwen3.5-9B テキストエンコーダー:** 以下のいずれか。`--text_encoder` にパスを指定します。
    * 公式の分割ファイル（`text_encoder/model-0000X-of-00004.safetensors`）。**最初のファイル**（`model-00001-of-00004.safetensors`）を指定してください。
    * ComfyUI 向けの ConvRot INT8 ファイル（`qwen3.5_9b_int8_convrot.safetensors`）。W4A8 GPTQ のファイル（`qwen3.5_9b_w4a8_gptq.safetensors`）は**サポートされません**。
    * Qwen3.5 の vision tower は読み込まれません。モデルの config は `configs/moss_image_qwen35_9b/` に同梱されています。
4. **FLUX.2 VAE モデルファイル:** ComfyUI / BFL 形式の `flux2-vae.safetensors` を推奨します。公式リポジトリの diffusers 形式 VAE（`vae/`、FLUX.2 の "small decoder" 版）も使用できます。両者の潜在空間は同一なので、latent のキャッシュは共用できます。`--vae` にパスを指定します。
5. **トークナイザー:** Hugging Face Hub（`OpenMOSS-Team/MOSS-Image-Base` の `tokenizer` サブフォルダ、約 20MB）から自動的にダウンロードされます。オフラインで学習する場合は、`--tokenizer_cache_dir` に Hugging Face のキャッシュディレクトリ、またはトークナイザーのファイル（`tokenizer_config.json` 等）を含むローカルディレクトリを指定してください。
6. **データセット定義ファイル (.toml):** 学習データセットの設定を記述した TOML 形式のファイル。（詳細は [データセット設定ガイド](./config_README-ja.md) を参照してください。）例として `my_moss_dataset_config.toml` を使用します。

**注意:**

* bucket の解像度は 16 の倍数である必要があります（`bucket_reso_steps` は 16、32、64 など。デフォルトの 64 で問題ありません）。
* MOSS-Image は LLM が書いた長い自然文のキャプションで学習されています。データセット設定で `enable_multiline_caption = true` を指定すると、改行を含むキャプションをそのまま使用できます。[データセット設定のドキュメント](./config_README-ja.md) を参照してください。
* 一つの画像に対する複数のキャプション候補（`enable_multiline_caption` の区切り行。公式の学習のように元のキャプションと長さの異なる再キャプションを用意する場合など）は、`--cache_text_encoder_outputs` と併用できます。全候補をエンコードして一つのキャッシュファイルに保存し、ステップごとにランダムに一つを選びます。キャッシュのサイズとキャッシュ時間は候補数倍になります。

</details>

## 4. Running the Training / 学習の実行

Execute `moss_image_train_network.py` from the terminal to start training. The overall command-line format is the same as `train_network.py`, but MOSS-Image specific options must be supplied.

Example command (DiT LoRA with cached text encoder outputs):

```bash
accelerate launch --num_cpu_threads_per_process 1 moss_image_train_network.py \
  --pretrained_model_name_or_path="<path to MOSS-Image DiT, e.g. diffusion_pytorch_model-00001-of-00003.safetensors>" \
  --text_encoder="<path to Qwen3.5-9B, e.g. model-00001-of-00004.safetensors>" \
  --vae="<path to flux2-vae.safetensors>" \
  --dataset_config="my_moss_dataset_config.toml" \
  --output_dir="<output directory>" \
  --output_name="my_moss_lora" \
  --save_model_as=safetensors \
  --network_module=networks.lora_generic \
  --network_dim=16 \
  --network_train_unet_only \
  --learning_rate=1e-4 \
  --optimizer_type="AdamW8bit" \
  --lr_scheduler="constant" \
  --timestep_sampling="sigma" \
  --weighting_scheme="logit_normal" \
  --logit_mean=-1.09 \
  --logit_std=0.95 \
  --discrete_flow_shift=1.0 \
  --max_train_epochs=10 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --cache_latents \
  --cache_latents_to_disk \
  --cache_text_encoder_outputs \
  --cache_text_encoder_outputs_to_disk
```

*(Write the command on one line or use `\` or `^` for line breaks.)*

**Note for Windows:** With the default `--max_data_loader_n_workers` (8), the DataLoader worker processes are re-created at every epoch on Windows, which can make the training several times slower (the effect is large with small datasets). Specify a small number of workers together with `--persistent_data_loader_workers`, e.g. `--max_data_loader_n_workers 2 --persistent_data_loader_workers` (more workers do not make it faster). Note that `--persistent_data_loader_workers` cannot be combined with `--max_data_loader_n_workers 0`.

The learning rate of `1e-4` is just an example. Adjust it according to your dataset and objectives. This value is for `alpha=1.0` (default). If increasing `--network_alpha`, consider lowering the learning rate.

`--timestep_sampling sigma --weighting_scheme logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0` reproduces the timestep distribution used in the pre-training of MOSS-Image (see [Section 7.2](#72-training-settings--学習設定)). A simpler setting with a similar distribution is `--timestep_sampling shift --discrete_flow_shift 3.0`.

Add `--convrot_int8` (and `--convrot_int8_text_encoder`) to reduce VRAM usage. See [Section 7.1](#71-vram-usage-optimization--vram-使用量の最適化).

<details>
<summary>日本語</summary>

学習は、ターミナルから `moss_image_train_network.py` を実行することで開始します。基本的なコマンドラインの構造は `train_network.py` と同様ですが、MOSS-Image 特有の引数を指定する必要があります。

コマンドラインの例（テキストエンコーダー出力をキャッシュした DiT の LoRA 学習）は英語のドキュメントを参照してください。

※実際には 1 行で書くか、適切な改行文字（`\` または `^`）を使用してください。

**Windows での注意:** `--max_data_loader_n_workers` のデフォルト（8）のままだと、Windows では DataLoader のワーカープロセスがエポックごとに再生成され、学習が数倍遅くなることがあります（データセットが小さいほど影響が大きい）。`--max_data_loader_n_workers 2 --persistent_data_loader_workers` のように、少ないワーカー数と `--persistent_data_loader_workers` を併せて指定してください（ワーカーを増やしても速くはなりません）。`--persistent_data_loader_workers` は `--max_data_loader_n_workers 0` とは併用できません。

学習率 1e-4 はあくまで一例です。データセットや目的に応じて適切に調整してください。またこの値は alpha=1.0（デフォルト）での値です。`--network_alpha` を増やす場合は学習率を下げることを検討してください。

`--timestep_sampling sigma --weighting_scheme logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0` は MOSS-Image の事前学習で使われたタイムステップ分布を再現します（[7.2 節](#72-training-settings--学習設定) を参照）。より簡潔で似た分布になる設定として `--timestep_sampling shift --discrete_flow_shift 3.0` もあります。

VRAM 使用量を減らすには `--convrot_int8`（および `--convrot_int8_text_encoder`）を追加してください。[7.1 節](#71-vram-usage-optimization--vram-使用量の最適化) を参照してください。

</details>

### 4.1. Explanation of Key Options / 主要なコマンドライン引数の解説

Besides the arguments explained in the [train_network.py guide](train_network.md), specify the following MOSS-Image specific options. For shared options (`--output_dir`, `--output_name`, `--network_module`, etc.), see that guide.

#### Model Options [Required] / モデル関連 [必須]

* `--pretrained_model_name_or_path="<path to MOSS-Image DiT>"` **[Required]**
  - Path to the DiT checkpoint: the first shard of the official diffusers format, or a ComfyUI single file (bf16 or ConvRot INT8). ConvRot INT8 files are detected automatically.
* `--text_encoder="<path to Qwen3.5-9B>"` **[Required]**
  - Path to the Qwen3.5-9B text encoder: the first shard of the official format, or the ComfyUI ConvRot INT8 file. The text encoder is always run in bf16 (or ConvRot INT8).
* `--vae="<path to FLUX.2 VAE>"` **[Required]**
  - Path to the FLUX.2 VAE (`flux2-vae.safetensors` in ComfyUI / BFL format, or the diffusers format). The VAE is always run in fp32.

#### Model Options [Optional] / モデル関連 [オプション]

* `--tokenizer_cache_dir="<directory>"` *[Optional]*
  - Hugging Face cache directory for the tokenizer, or a local directory containing the tokenizer files. Use this for offline training.
* `--moss_max_token_length=<integer>` *[Optional]*
  - Maximum number of text tokens including the chat template. Default and maximum: `4096` (the limit of the official pipeline). Longer captions are truncated. Note that with 1024x1024 images the image sequence is 4096 tokens, so captions of thousands of tokens make attention considerably more expensive.

#### MOSS-Image Training Parameters / MOSS-Image 学習パラメータ

* `--timestep_sampling=<choice>`
  - Timestep sampling method. Choose from `sigma`, `uniform`, `sigmoid` (default), `shift`, `flux_shift`. Same options as FLUX training. See the [flux_train_network.py guide](flux_train_network.md) for details on each method.
* `--discrete_flow_shift=<float>`
  - Shift for the timestep distribution in Rectified Flow training. Default `1.0`, which is the training grid of MOSS-Image. This value is used when `--timestep_sampling` is set to `sigma` or `shift`.
* `--sigmoid_scale=<float>`
  - Scale factor when `--timestep_sampling` is set to `sigmoid`, `shift`, or `flux_shift`. Default `1.0`.
* `--attn_mode=<choice>`
  - Attention implementation to use. Choose from `torch` (default), `xformers`, `flash`, `sageattn`. `xformers` requires `--split_attn`. `sageattn` does not support training (inference only). This option overrides `--xformers`.
* `--split_attn`
  - Split attention computation to reduce memory usage. Required when using `--attn_mode xformers`.
* `--sample_uncond_skip_layers=<comma separated integers>`
  - DiT block indices skipped in the negative (unconditional) branch of CFG during sample image generation. Default `22`, which is the setting of the official pipeline. An empty string disables skipping. See [Section 7.4](#74-sample-image-generation--サンプル画像生成).

#### Memory and Speed / メモリ・速度関連

* `--convrot_int8`
  - Quantize the base weights of the DiT (the Linear layers of the 42 main blocks) to ConvRot INT8 at load time. Weight VRAM is halved. Not needed for the ComfyUI `*_int8_convrot.safetensors` files, which are detected automatically. See [Section 7.1](#71-vram-usage-optimization--vram-使用量の最適化).
* `--convrot_int8_text_encoder`
  - Quantize the text encoder weights to ConvRot INT8 at load time (about 16GB -> 8GB). Not needed for the ComfyUI `qwen3.5_9b_int8_convrot.safetensors`.
* `--convrot_int8_bwd=<bf16|int8>`
  - Backward mode of the ConvRot INT8 layers. Default `bf16` (most accurate). `int8` reuses the fused int8 GEMM for the input gradient (requires triton and CUDA); it has no measurable benefit with gradient checkpointing.
* `--blocks_to_swap=<integer>`
  - Number of DiT blocks to swap between CPU and GPU. More blocks reduce VRAM but slow training. Maximum **40** (42 blocks - 2). Cannot be used with `--cpu_offload_checkpointing` or `--unsloth_offload_checkpointing`.
  - `--block_swap_h2d_only` enables the faster H2D-only block swap for LoRA training (requires `--gradient_checkpointing`). See [Block Swap](./block_swap.md).
* `--unsloth_offload_checkpointing`
  - Offload activations to CPU RAM using async non-blocking transfers (faster than `--cpu_offload_checkpointing`). Enables `--gradient_checkpointing`. Cannot be combined with `--cpu_offload_checkpointing` or `--blocks_to_swap`.
* `--cache_text_encoder_outputs`, `--cache_text_encoder_outputs_to_disk`
  - Cache the Qwen3.5-9B outputs. Strongly recommended when not training text encoder LoRA: the text encoder is removed from the GPU after caching. The cache files are `<image>_moss_te.npz` and contain the unconditional embedding as well, so `caption_dropout_rate` can be used. Requires `--network_train_unet_only`.
  - The outputs of all caption candidates of an image are stored in one file (`num_candidates`, `prompt_embeds_{k}` / `attn_mask_{k}` / `caption_{k}` for each candidate `k`). The processed caption texts (`caption_prefix` / `caption_suffix` applied) are stored as well, and the file is re-cached when the number of candidates or the texts differ from the current captions, unless `--skip_cache_check` is specified. The files created with version 0.12.1 (without `num_candidates`) are still valid for images with a single candidate.
* `--text_encoder_batch_size=<integer>`
  - Number of captions processed per call when caching text encoder outputs. The text encoder itself always runs one caption at a time (no padding), so this only affects the loop granularity.
* `--cache_latents`, `--cache_latents_to_disk`
  - Cache the FLUX.2 VAE outputs. The cache files are `<image>_<W>x<H>_moss.npz` and hold the raw VAE output (32 channels, 1/8 resolution); the patchify and normalization for the DiT are applied at training time.
* `--compile` and related options
  - Speed up training with per-block `torch.compile`. See the [torch.compile for Anima guide](anima_torch_compile.md) for details (the options are shared). With ConvRot INT8, the quantized Linear layers are excluded from compilation automatically.

#### Incompatible or Unsupported Options / 非互換・非サポートの引数

* `--v2`, `--v_parameterization`, `--clip_skip` - Options for Stable Diffusion v1/v2 that are not used for MOSS-Image training.
* `--fp8_base`, `--fp8_base_unet` - Not supported for MOSS-Image. If specified, they are disabled with a warning. Use `--convrot_int8` instead.
* `--mixed_precision fp16` - Not recommended: the DiT and the text encoder are bf16 models. Use `bf16`.

<details>
<summary>日本語</summary>

[`train_network.py` のガイド](train_network.md) で説明されている引数に加え、以下の MOSS-Image 特有の引数を指定します。共通の引数については、上記ガイドを参照してください。

#### モデル関連 [必須]

* `--pretrained_model_name_or_path="<path to MOSS-Image DiT>"` **[必須]** - DiT のチェックポイント。公式 diffusers 形式の最初の分割ファイル、または ComfyUI 向けの単一ファイル（bf16 または ConvRot INT8）。ConvRot INT8 のファイルは自動判別されます。
* `--text_encoder="<path to Qwen3.5-9B>"` **[必須]** - Qwen3.5-9B テキストエンコーダー。公式形式の最初の分割ファイル、または ComfyUI 向けの ConvRot INT8 ファイル。テキストエンコーダーは常に bf16（または ConvRot INT8）で動作します。
* `--vae="<path to FLUX.2 VAE>"` **[必須]** - FLUX.2 VAE（ComfyUI / BFL 形式の `flux2-vae.safetensors`、または diffusers 形式）。VAE は常に fp32 で動作します。

#### モデル関連 [オプション]

* `--tokenizer_cache_dir` - トークナイザーの Hugging Face キャッシュディレクトリ、またはトークナイザーのファイルを含むローカルディレクトリ。オフラインで学習する場合に指定します。
* `--moss_max_token_length` - chat template を含むテキストトークンの最大数。デフォルトおよび最大値は `4096`（公式パイプラインの上限）。これを超えるキャプションは切り詰められます。1024x1024 の画像トークンは 4096 なので、数千トークンのキャプションを使うと attention のコストが大きく増えます。

#### MOSS-Image 学習パラメータ

* `--timestep_sampling` - タイムステップのサンプリング方法。`sigma`、`uniform`、`sigmoid`（デフォルト）、`shift`、`flux_shift` から選択。FLUX 学習と同じオプションです。各方法の詳細は [flux_train_network.py のガイド](flux_train_network.md) を参照してください。
* `--discrete_flow_shift` - Rectified Flow 学習のタイムステップ分布シフト。デフォルト `1.0`（MOSS-Image の学習グリッド）。`--timestep_sampling` が `sigma` または `shift` の場合に使用されます。
* `--sigmoid_scale` - `sigmoid`、`shift`、`flux_shift` タイムステップサンプリングのスケール係数。デフォルト `1.0`。
* `--attn_mode` - 使用する Attention の実装。`torch`（デフォルト）、`xformers`、`flash`、`sageattn` から選択。`xformers` は `--split_attn` の指定が必要です。`sageattn` は学習をサポートしていません（推論のみ）。
* `--split_attn` - メモリ使用量を減らすために attention 時にバッチを分割します。`--attn_mode xformers` 使用時に必要です。
* `--sample_uncond_skip_layers` - サンプル画像生成で、CFG の negative（unconditional）側で飛ばす DiT ブロックのインデックス（カンマ区切り）。デフォルト `22`（公式パイプラインの設定）。空文字列で無効。[7.4 節](#74-sample-image-generation--サンプル画像生成) を参照。

#### メモリ・速度関連

* `--convrot_int8` - DiT のベース重み（42 個のメインブロックの Linear 層）をロード時に ConvRot INT8 に量子化します。重みの VRAM が半減します。ComfyUI の `*_int8_convrot.safetensors` は自動判別されるので指定不要です。[7.1 節](#71-vram-usage-optimization--vram-使用量の最適化) を参照。
* `--convrot_int8_text_encoder` - テキストエンコーダーの重みをロード時に ConvRot INT8 に量子化します（約 16GB → 8GB）。ComfyUI の `qwen3.5_9b_int8_convrot.safetensors` には不要です。
* `--convrot_int8_bwd` - ConvRot INT8 層の backward の方式。デフォルト `bf16`（最も高精度）。`int8` は入力勾配にも融合 int8 GEMM を使います（triton と CUDA が必要）。gradient checkpointing 使用時には速度差はほぼありません。
* `--blocks_to_swap` - DiT のブロックを CPU と GPU でスワップして VRAM を節約。最大 **40**（42 ブロック − 2）。`--cpu_offload_checkpointing` および `--unsloth_offload_checkpointing` とは併用できません。
  - `--block_swap_h2d_only`でLoRA学習向けの高速なH2D-onlyブロックスワップを有効にできます（`--gradient_checkpointing`が必須）。[ブロックスワップ](./block_swap.md)を参照してください。
* `--unsloth_offload_checkpointing` - 非同期転送でアクティベーションを CPU RAM にオフロード。`--gradient_checkpointing` を有効にします。`--cpu_offload_checkpointing` および `--blocks_to_swap` とは併用できません。
* `--cache_text_encoder_outputs`, `--cache_text_encoder_outputs_to_disk` - Qwen3.5-9B の出力をキャッシュします。テキストエンコーダーの LoRA を学習しない場合は強く推奨します（キャッシュ後にテキストエンコーダーは GPU から解放されます）。キャッシュファイルは `<image>_moss_te.npz` で、unconditional の埋め込みも保存されるため `caption_dropout_rate` を使用できます。`--network_train_unet_only` が必要です。画像の全キャプション候補の出力を一つのファイルに保存します（`num_candidates`、候補 `k` ごとの `prompt_embeds_{k}` / `attn_mask_{k}` / `caption_{k}`）。処理済みのキャプション文字列（`caption_prefix` / `caption_suffix` 適用後）も保存され、候補数や文字列が現在のキャプションと異なる場合は、`--skip_cache_check` を指定していなければ再キャッシュされます。バージョン 0.12.1 で作成したファイル（`num_candidates` なし）は、候補が一つの画像ではそのまま使えます。
* `--text_encoder_batch_size` - テキストエンコーダー出力のキャッシュ時に 1 回に処理するキャプション数。テキストエンコーダー自体は常に 1 件ずつ（パディングなし）実行されるため、ループの粒度のみに影響します。
* `--cache_latents`, `--cache_latents_to_disk` - FLUX.2 VAE の出力をキャッシュします。キャッシュファイルは `<image>_<W>x<H>_moss.npz` で、VAE の生の出力（32 チャンネル、1/8 解像度）を保存します。DiT 向けの patchify と正規化は学習時に行います。
* `--compile` 他 - ブロック単位の `torch.compile` で学習を高速化します。詳細は [Anima 向け torch.compile ガイド](anima_torch_compile.md) を参照してください（オプションは共通です）。ConvRot INT8 使用時は、量子化された Linear 層は自動的にコンパイル対象から除外されます。

#### 非互換・非サポートの引数

* `--v2`, `--v_parameterization`, `--clip_skip` - Stable Diffusion v1/v2 向けの引数。MOSS-Image の学習では使用されません。
* `--fp8_base`, `--fp8_base_unet` - MOSS-Image ではサポートされていません。指定した場合、警告とともに無効化されます。代わりに `--convrot_int8` を使用してください。
* `--mixed_precision fp16` - 推奨されません。DiT とテキストエンコーダーは bf16 のモデルです。`bf16` を使用してください。

</details>

### 4.2. Starting Training / 学習の開始

After setting the required arguments, run the command to begin training. The overall flow and how to check logs are the same as in the [train_network.py guide](train_network.md#32-starting-the-training--学習の開始).

<details>
<summary>日本語</summary>

必要な引数を設定したら、コマンドを実行して学習を開始します。全体の流れやログの確認方法は、[train_network.py のガイド](train_network.md#32-starting-the-training--学習の開始) と同様です。

</details>

## 5. LoRA Target Modules / LoRA の学習対象モジュール

The LoRA network module is `networks.lora_generic`, a generic LoRA implementation that detects the architecture from the model. When training LoRA with `moss_image_train_network.py`, the following modules are targeted by default:

* **DiT blocks (`MossImageBlock`, `layers.0` to `layers.41`)**: The attention projections `proj.q`, `proj.k`, `proj.v`, `proj.o` and the feed-forward layers `proj.ffn_up_gate`, `proj.ffn_down` of each block (252 modules in total). The modulation layers, norms, the patch embedder (`video_embedder`), the time embedding and the final layer are not LoRA targets.
* **Text embedder (`MossImageTextEmbedder`)**: The input projection of the text embedding (`text_embedder.proj`) and the two text refiner blocks (`text_embedder.blocks.0/1`). Only when `--network_args "train_text_embedder=True"` is specified (see [Section 5.3](#53-text-embedder-lora--text-embedder-lora)).
* **Text Encoder (Qwen3.5-9B)**: Only when `--network_train_unet_only` is NOT specified and `--cache_text_encoder_outputs` is NOT used. See [Section 7.3](#73-text-encoder-lora-support--text-encoder-lora-のサポート).

The LoRA weight keys are `lora_unet_layers_<N>_proj_q` etc. for the DiT and `lora_te_layers_<N>_...` for the text encoder. The DiT keys follow the official (diffusers) module names with separate q / k / v projections; ComfyUI maps them to its fused `qkv` weight when loading the LoRA, so no conversion is needed.

LoHa and LoKr (`--network_module networks.loha` / `networks.lokr`) use the same module selection. See [the LoHa / LoKr documentation](loha_lokr.md) for their options.

<details>
<summary>日本語</summary>

LoRA のネットワークモジュールは `networks.lora_generic` です。モデルからアーキテクチャを自動検出する汎用の LoRA 実装です。`moss_image_train_network.py` で LoRA を学習させる場合、デフォルトでは以下のモジュールが対象となります。

* **DiT ブロック（`MossImageBlock`、`layers.0`〜`layers.41`）**: 各ブロックの attention の射影 `proj.q`、`proj.k`、`proj.v`、`proj.o` と feed-forward 層 `proj.ffn_up_gate`、`proj.ffn_down`（合計 252 モジュール）。modulation 層、norm、patch embedder（`video_embedder`）、time embedding、final layer は LoRA の対象外です。
* **Text embedder（`MossImageTextEmbedder`）**: テキスト埋め込みの入力射影（`text_embedder.proj`）と 2 つの text refiner ブロック（`text_embedder.blocks.0/1`）。`--network_args "train_text_embedder=True"` を指定した場合のみ（[5.3 節](#53-text-embedder-lora--text-embedder-lora) を参照）。
* **テキストエンコーダー（Qwen3.5-9B）**: `--network_train_unet_only` を指定せず、かつ `--cache_text_encoder_outputs` を使用しない場合のみ。[7.3 節](#73-text-encoder-lora-support--text-encoder-lora-のサポート) を参照。

LoRA の重みのキーは、DiT が `lora_unet_layers_<N>_proj_q` 等、テキストエンコーダーが `lora_te_layers_<N>_...` です。DiT のキーは公式（diffusers）のモジュール名に従い q / k / v が別々になっています。ComfyUI は LoRA のロード時にこれらを fused な `qkv` の重みへ対応付けるため、変換は不要です。

LoHa と LoKr（`--network_module networks.loha` / `networks.lokr`）も同じモジュール選択を使用します。オプションは [LoHa / LoKr のドキュメント](loha_lokr.md) を参照してください。

</details>

### 5.1. Module Selection with Patterns / パターンによるモジュール選択

You can customize which modules are included or excluded using regex patterns in `--network_args`:

* `exclude_patterns` - Exclude modules matching these patterns (in addition to the default exclusion).
* `include_patterns` - Force-include modules matching these patterns, overriding exclusion.

Patterns are matched against the full module name (e.g. `layers.0.proj.q`, `text_embedder.proj`) using `re.fullmatch()`. The DiT has no default exclusion; the text encoder excludes `.*in_proj_[ab]` by default (see [Section 7.3](#73-text-encoder-lora-support--text-encoder-lora-のサポート)).

Example to train only the attention projections of the DiT:
```
--network_args "exclude_patterns=['.*ffn.*']"
```

Example to exclude block 22, which the official pipeline skips in the unconditional branch of CFG:
```
--network_args "exclude_patterns=['layers\.22\..*']"
```

<details>
<summary>日本語</summary>

`--network_args` で正規表現パターンを使用して、含めるモジュールと除外するモジュールをカスタマイズできます。

* `exclude_patterns` - これらのパターンにマッチするモジュールを除外（デフォルトの除外に追加）。
* `include_patterns` - これらのパターンにマッチするモジュールを強制的に含める（除外を上書き）。

パターンは完全なモジュール名（例: `layers.0.proj.q`、`text_embedder.proj`）に対して `re.fullmatch()` でマッチングされます。DiT にはデフォルトの除外はありません。テキストエンコーダーはデフォルトで `.*in_proj_[ab]` を除外します（[7.3 節](#73-text-encoder-lora-support--text-encoder-lora-のサポート) を参照）。

DiT の attention の射影のみを学習する例: `--network_args "exclude_patterns=['.*ffn.*']"`

公式パイプラインが CFG の unconditional 側で飛ばすブロック 22 を除外する例: `--network_args "exclude_patterns=['layers\.22\..*']"`

</details>

### 5.2. Regex-based Rank and Learning Rate Control / 正規表現によるランク・学習率の制御

You can specify different ranks (network_dim) and learning rates for modules matching specific regex patterns:

* `network_reg_dims`: Specify ranks for modules matching a regular expression. The format is a comma-separated string of `pattern=rank`.
    * Example: `--network_args "network_reg_dims=.*proj\.[qkvo]=8,.*ffn.*=16"`
    * This sets the rank to 8 for the attention projections and 16 for the feed-forward layers.
* `network_reg_lrs`: Specify learning rates for modules matching a regular expression. The format is a comma-separated string of `pattern=lr`.
    * Example: `--network_args "network_reg_lrs=.*text_embedder.*=5e-5"`

**Notes:**

* Settings via `network_reg_dims` and `network_reg_lrs` take precedence over the global `--network_dim` and `--learning_rate` settings.
* Patterns are matched using `re.fullmatch()` against the module's original name (e.g., `layers.0.proj.q`).

<details>
<summary>日本語</summary>

正規表現にマッチするモジュールに対して、異なるランクや学習率を指定できます。

* `network_reg_dims`: 正規表現にマッチするモジュールに対してランクを指定します。`pattern=rank` 形式の文字列をカンマで区切って指定します。
    * 例: `--network_args "network_reg_dims=.*proj\.[qkvo]=8,.*ffn.*=16"`（attention の射影を 8、feed-forward 層を 16 にする）
* `network_reg_lrs`: 正規表現にマッチするモジュールに対して学習率を指定します。`pattern=lr` 形式の文字列をカンマで区切って指定します。
    * 例: `--network_args "network_reg_lrs=.*text_embedder.*=5e-5"`

**注意点:**
* `network_reg_dims` および `network_reg_lrs` での設定は、全体設定である `--network_dim` や `--learning_rate` よりも優先されます。
* パターンはモジュールのオリジナル名（例: `layers.0.proj.q`）に対して `re.fullmatch()` でマッチングされます。

</details>

### 5.3. Text Embedder LoRA / Text Embedder LoRA

The text embedder of the DiT (the projection from the 4096-dimensional Qwen3.5 hidden states to the DiT width and the two text refiner blocks) is not trained by default. To apply LoRA to it:

```
--network_args "train_text_embedder=True"
```

(`train_llm_adapter=True`, the Anima naming, is accepted as an alias.) This is a way to adapt the text conditioning without training the text encoder itself, and it works together with `--cache_text_encoder_outputs`. Consider a lower learning rate for these modules, e.g. `"network_reg_lrs=.*text_embedder.*=5e-5"`.

<details>
<summary>日本語</summary>

DiT の text embedder（Qwen3.5 の 4096 次元の hidden state を DiT の幅へ射影する層と、2 つの text refiner ブロック）はデフォルトでは学習されません。LoRA を適用するには `--network_args "train_text_embedder=True"` を指定します（Anima の名前である `train_llm_adapter=True` も別名として使用できます）。

テキストエンコーダー自体を学習せずにテキスト条件付けを適応させる方法で、`--cache_text_encoder_outputs` と併用できます。これらのモジュールの学習率は `"network_reg_lrs=.*text_embedder.*=5e-5"` などで下げることを検討してください。

</details>

### 5.4. Other Network Args / その他のネットワーク引数

* `--network_args "verbose=True"` - Print all LoRA module names and their dimensions.
* `--network_args "rank_dropout=0.1"` - Rank dropout rate.
* `--network_args "module_dropout=0.1"` - Module dropout rate.
* `--network_args "loraplus_lr_ratio=2.0"` - LoRA+ learning rate ratio.
* `--network_args "loraplus_unet_lr_ratio=2.0"` - LoRA+ learning rate ratio for DiT only.
* `--network_args "loraplus_text_encoder_lr_ratio=2.0"` - LoRA+ learning rate ratio for text encoder only.

<details>
<summary>日本語</summary>

* `verbose=True` - 全 LoRA モジュール名と dim を表示
* `rank_dropout` - ランクドロップアウト率
* `module_dropout` - モジュールドロップアウト率
* `loraplus_lr_ratio` - LoRA+ 学習率比率
* `loraplus_unet_lr_ratio` - DiT 専用の LoRA+ 学習率比率
* `loraplus_text_encoder_lr_ratio` - テキストエンコーダー専用の LoRA+ 学習率比率

</details>

## 6. Using the Trained Model / 学習済みモデルの利用

When training finishes, a LoRA model file (e.g. `my_moss_lora.safetensors`) is saved in the directory specified by `output_dir`.

* **ComfyUI:** The LoRA can be loaded with the standard LoRA loader without conversion, both for the DiT and for the text encoder (ComfyUI maps the separate q / k / v keys of sd-scripts to its fused `qkv` weight). Use a version of ComfyUI that supports MOSS-Image.
* **`moss_image_minimal_inference.py`:** The inference script in this repository merges DiT and text encoder LoRA at load time. See [Section 9](#9-related-tools--関連ツール).

<details>
<summary>日本語</summary>

学習が完了すると、指定した `output_dir` に LoRA モデルファイル（例: `my_moss_lora.safetensors`）が保存されます。

* **ComfyUI:** DiT、テキストエンコーダーとも、LoRA は変換なしで標準の LoRA ローダーから読み込めます（ComfyUI が sd-scripts の q / k / v 別のキーを fused な `qkv` の重みへ対応付けます）。MOSS-Image に対応したバージョンの ComfyUI を使用してください。
* **`moss_image_minimal_inference.py`:** 当リポジトリの推論スクリプトは、DiT とテキストエンコーダーの LoRA をロード時にマージします。[9 節](#9-related-tools--関連ツール) を参照してください。

</details>

## 7. Advanced Settings / 高度な設定

### 7.1. VRAM Usage Optimization / VRAM 使用量の最適化

MOSS-Image is a large model: the DiT is about 10.2GB and the text encoder about 15.9GB in bf16. The following options reduce VRAM usage.

#### ConvRot INT8 / ConvRot INT8

`--convrot_int8` quantizes the frozen DiT base weights to int8 with ConvRot ([arXiv:2512.03673](https://arxiv.org/abs/2512.03673)): a block-diagonal Hadamard rotation smooths activation outliers, then weights are quantized per-channel to int8. The forward pass runs a fused Triton int8 GEMM (online activation rotation + dynamic row-wise int8 quantization + dequantization fused into the matmul). This is the **alternative to fp8** for MOSS-Image; `--fp8_base` is not supported. This is a port of the implementation in [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner).

- **Scope:** The Linear layers of the 42 main blocks of the DiT (`proj.q/k/v/o`, `ffn_up_gate`, `ffn_down`; 252 layers). The text embedder, modulation layers and the final layer stay bf16. The quantized layers are still LoRA-trainable (the LoRA branch receives the unrotated input and its gradients are unaffected by the base quantization).
- **Text encoder:** `--convrot_int8_text_encoder` quantizes the text encoder with the same recipe as the ComfyUI file (200 Linear layers and the token embedding; about 15.9GB -> 7.4GB). The text encoder is quantized on the GPU at load time and kept on the CPU.
- **Pre-quantized files:** The ComfyUI ConvRot INT8 checkpoints (`moss_image_base_int8_convrot.safetensors`, `qwen3.5_9b_int8_convrot.safetensors`; `weight` int8 + `weight_scale` + `comfy_quant` tensors) are detected automatically and do not need the options. The quantization recipe of sd-scripts is bit-identical to these files, so dynamic quantization of the bf16 weights gives the same model.
- **Requires triton** for the fused kernels (`pip install triton-windows` on Windows, matching your torch version). Without triton, training still works via a dequantizing bf16 fallback: VRAM savings remain but it is slower than bf16.
- `--convrot_int8_bwd int8` (opt-in) also routes the input gradient through the fused int8 GEMM: slightly quantized gradients, no measurable step-time gain under gradient checkpointing. The default `bf16` dequantizes transiently and is the most accurate.
- **Quality:** The output of the int8 DiT differs from bf16 by about PSNR 27dB on generated images (same composition and details). The text encoder outputs differ by about 5% relative error. The loss curves of LoRA training match bf16.
- **Speed:** On an RTX PRO 6000 Blackwell, LoRA training speed is about the same as bf16 (1.41 vs 1.45 s/step at 1024x1024, batch size 1, gradient checkpointing). On GPUs without fp8 support (RTX 30 series and older) a true int8 GEMM is faster than bf16. The first 20 steps or so are slower because of the Triton autotuning.
- Composes with `--blocks_to_swap`, `--gradient_checkpointing` and `--compile` (the quantized Linear layers are excluded from compilation automatically).
- **LoRA merge at inference:** Merging LoRA into a pre-quantized checkpoint is rejected by `moss_image_minimal_inference.py`. Use the bf16 checkpoint with `--convrot_int8` instead (the LoRA is merged first, then quantized). For training this does not matter, since the LoRA is a separate module.

#### Key VRAM Reduction Options / 主要な VRAM 削減オプション

- **`--convrot_int8` / `--convrot_int8_text_encoder`**: See above.
- **`--blocks_to_swap <number>`**: Swaps DiT blocks between CPU and GPU. Higher numbers save more VRAM but reduce training speed. Maximum 40.
- **`--gradient_checkpointing`**: Standard gradient checkpointing to reduce VRAM at the cost of compute. Practically required.
- **`--unsloth_offload_checkpointing`**: Offloads the checkpointed activations to CPU RAM using async transfers. Cannot be combined with `--blocks_to_swap`.
- **`--cache_text_encoder_outputs`**: Caches the Qwen3.5-9B outputs so the text encoder can be freed from the GPU during training. The text encoder is still loaded to CPU RAM (about 16GB in bf16, 8GB with ConvRot INT8) and moved to the GPU for caching.
- **`--cache_latents`**: Caches the FLUX.2 VAE outputs so the VAE can be freed from the GPU during training.
- **Using Adafactor optimizer**: Can reduce VRAM usage (LoRA optimizer states are small, so the effect is minor):
  ```
  --optimizer_type adafactor --optimizer_args "relative_step=False" "scale_parameter=False" "warmup_init=False" --lr_scheduler constant_with_warmup --max_grad_norm 0.0
  ```

#### Measured Memory Usage / メモリ使用量の実測値

Peak GPU memory as reported by the driver (`nvidia-smi`, includes the CUDA context) and peak CPU RAM (working set of the process) measured on Windows with an RTX PRO 6000 Blackwell, 1024x1024 images, batch size 1, LoRA dim 16, AdamW, `--gradient_checkpointing`, latents cached to disk. "Startup" includes moving the text encoder to the GPU for caching (this happens even when all outputs are already cached on disk); "Training" is the peak after the training loop started.

| Run | DiT | Text Encoder | GPU (startup) | GPU (training) | CPU RAM |
|---|---|---|---|---|---|
| DiT LoRA, TE outputs cached | bf16 | bf16 | 15.3GB | 13.2GB | 17.2GB |
| DiT LoRA, TE outputs cached | ConvRot INT8 | ConvRot INT8 | 9.1GB | 9.1GB | 9.9GB |
| + `--blocks_to_swap 20` | ConvRot INT8 | ConvRot INT8 | 7.9GB | 6.9GB | 14.0GB |
| + `--blocks_to_swap 36` | ConvRot INT8 | ConvRot INT8 | 7.9GB | 5.2GB | 15.6GB |
| + `--unsloth_offload_checkpointing` | ConvRot INT8 | ConvRot INT8 | 8.2GB | 8.2GB | 11.2GB |
| `--blocks_to_swap 20` | bf16 | ConvRot INT8 | 9.0GB | 9.0GB | 18.5GB |
| DiT + TE LoRA (no TE cache) | ConvRot INT8 | ConvRot INT8 | 17.1GB | 17.1GB | 8.4GB |
| DiT + TE LoRA (no TE cache) | bf16 | bf16 | 32.4GB | 32.4GB | 16.1GB |

Caching with the tools: `tools/cache_latents.py` (fp32 VAE, 1024x1024) 3.2GB; `tools/cache_text_encoder_outputs.py` 15.5GB with the bf16 text encoder, 8.3GB with ConvRot INT8 (pre-quantized or `--convrot_int8_text_encoder`).

Guidelines:

* **24GB GPU:** bf16 DiT with the text encoder outputs cached. Use the ConvRot INT8 text encoder (or cache the outputs with the tool beforehand) if the startup peak is a problem.
* **16GB GPU:** ConvRot INT8 for both the DiT and the text encoder (9GB). Also bf16 DiT with `--blocks_to_swap` and the INT8 text encoder.
* **12GB GPU:** ConvRot INT8 for both and `--blocks_to_swap` (20 to 36). The text encoder outputs must be cached. About 15GB of CPU RAM is needed for the swapped blocks and the text encoder.
* **8GB GPU:** Not supported at the moment. Training itself fits with ConvRot INT8 and `--blocks_to_swap 36` (5.2GB), but caching the text encoder outputs needs about 8.3GB even with the ConvRot INT8 text encoder.
* **Text encoder LoRA:** needs about 17GB with ConvRot INT8 (DiT and text encoder), 32GB in bf16.
* Note that on Windows the commit charge of the process is larger than the working set (up to about 38GB in the bf16 runs above), because the GPU allocations are backed by system memory. Make sure the page file is large enough.

<details>
<summary>日本語</summary>

MOSS-Image は大きなモデルです。bf16 で DiT が約 10.2GB、テキストエンコーダーが約 15.9GB あります。以下のオプションで VRAM 使用量を削減できます。

#### ConvRot INT8

`--convrot_int8` で、凍結された DiT のベース重みを ConvRot（[arXiv:2512.03673](https://arxiv.org/abs/2512.03673)）で int8 量子化します。ブロック対角 Hadamard 回転で activation の外れ値を平滑化してから、重みを per-channel で int8 量子化します。forward は融合 Triton int8 GEMM（オンラインの activation 回転＋動的な行単位 int8 量子化＋逆量子化を matmul に融合）で実行されます。MOSS-Image では **fp8 の代替**であり、`--fp8_base` はサポートされません。[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) の実装の移植です。

- **適用範囲:** DiT の 42 個のメインブロックの Linear 層（`proj.q/k/v/o`、`ffn_up_gate`、`ffn_down`、252 層）。text embedder、modulation 層、final layer は bf16 のままです。量子化された層も LoRA 学習可能です（LoRA 枝には回転前の入力が渡され、その勾配はベースの量子化の影響を受けません）。
- **テキストエンコーダー:** `--convrot_int8_text_encoder` で、ComfyUI のファイルと同じレシピ（200 個の Linear 層とトークン埋め込み）でテキストエンコーダーを量子化します（約 15.9GB → 7.4GB）。ロード時に GPU で量子化し、CPU に置きます。
- **事前量子化済みファイル:** ComfyUI の ConvRot INT8 チェックポイント（`moss_image_base_int8_convrot.safetensors`、`qwen3.5_9b_int8_convrot.safetensors`。`weight` int8＋`weight_scale`＋`comfy_quant`）は自動判別されるため、オプションの指定は不要です。sd-scripts の量子化レシピはこれらのファイルとビット単位で一致するので、bf16 の重みを動的に量子化しても同じモデルになります。
- 融合カーネルには **triton** が必要です（Windows では torch のバージョンに対応する `pip install triton-windows`）。triton がなくても逆量子化 bf16 フォールバックで学習は可能です（VRAM 削減は維持されますが、bf16 より遅くなります）。
- `--convrot_int8_bwd int8`（opt-in）を指定すると、入力勾配も融合 int8 GEMM を通ります。勾配がわずかに量子化され、gradient checkpointing 使用時には速度差はほぼありません。デフォルトの `bf16` は一時的に逆量子化する方式で、最も高精度です。
- **品質:** int8 の DiT の出力は、生成画像で bf16 と PSNR 約 27dB の差です（構図・細部は同じ）。テキストエンコーダーの出力は相対誤差約 5% です。LoRA 学習の loss curve は bf16 と一致します。
- **速度:** RTX PRO 6000 Blackwell では LoRA 学習の速度は bf16 とほぼ同じです（1024x1024、batch size 1、gradient checkpointing で 1.41 vs 1.45 s/step）。fp8 非対応の GPU（RTX 30 シリーズ以前）では、真の int8 GEMM が bf16 より高速になります。最初の 20 ステップほどは Triton の autotune のため遅くなります。
- `--blocks_to_swap`、`--gradient_checkpointing`、`--compile` と併用できます（量子化された Linear 層は自動的にコンパイル対象から除外されます）。
- **推論時の LoRA マージ:** 事前量子化済みチェックポイントへの LoRA マージは `moss_image_minimal_inference.py` でエラーになります。bf16 のチェックポイントに `--convrot_int8` を付けて使用してください（LoRA をマージしてから量子化します）。学習では LoRA は別モジュールなので関係ありません。

#### 主要な VRAM 削減オプション

- `--convrot_int8` / `--convrot_int8_text_encoder`: 上記を参照
- `--blocks_to_swap`: DiT のブロックを CPU と GPU 間でスワップ（最大 40）
- `--gradient_checkpointing`: 標準的な勾配チェックポイント（実質必須）
- `--unsloth_offload_checkpointing`: 非同期転送でアクティベーションを CPU にオフロード（`--blocks_to_swap` とは併用不可）
- `--cache_text_encoder_outputs`: Qwen3.5-9B の出力をキャッシュ。テキストエンコーダーは CPU RAM にはロードされ（bf16 で約 16GB、ConvRot INT8 で 8GB）、キャッシュ時に GPU へ移動します
- `--cache_latents`: FLUX.2 VAE の出力をキャッシュ
- Adafactor オプティマイザの使用（LoRA のオプティマイザ状態は小さいので効果は限定的）

#### メモリ使用量の実測値

英語のドキュメントの表を参照してください。

</details>

### 7.2. Training Settings / 学習設定

#### Timestep Sampling

The `--timestep_sampling` option specifies how timesteps are sampled. The available methods are the same as FLUX training:

- `sigma`: Sigma-based sampling like SD3: `u = sigmoid(N(logit_mean, logit_std))` is mapped to the discrete training grid shifted by `--discrete_flow_shift`.
- `uniform`: Uniform random sampling from [0, 1].
- `sigmoid` (default): Sample from Normal(0,1), multiply by `sigmoid_scale`, apply sigmoid. Good general-purpose option.
- `shift`: Like `sigmoid`, but applies the discrete flow shift formula: `t_shifted = (t * shift) / (1 + (shift - 1) * t)`.
- `flux_shift`: Resolution-dependent shift used in FLUX training.

See the [flux_train_network.py guide](flux_train_network.md) for detailed descriptions.

**Reproducing the pre-training distribution of MOSS-Image:** According to the technical report, the final stage of the pre-training sampled the timesteps from a logit-normal distribution (mean -1.09, std 0.95) on the 1000-step grid without shift, and the loss was not weighted by timestep. This is reproduced by:

```
--timestep_sampling sigma --weighting_scheme logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0
```

(With `--timestep_sampling sigma`, `--weighting_scheme logit_normal` selects the logit-normal distribution and the loss weight is 1.) The negative mean biases the samples toward the noisier half of the trajectory. The official schedule additionally truncates the distribution at 0.95 and mixes in 1% uniform samples; these details are not reproduced. This setting was used for the verification of this script.

`--timestep_sampling shift --discrete_flow_shift 3.0` gives a very similar distribution with fewer options (compare them with `--show_timesteps`, see below), so it is a good alternative. The default `sigmoid` sampling (mean 0, no shift) also trains, but its distribution is centered at t=500 and quite different from the pre-training distribution.

#### Discrete Flow Shift

The `--discrete_flow_shift` option (default `1.0`) applies when `--timestep_sampling` is set to `sigma` or `shift`. With `sigma` sampling, `1.0` matches the (unshifted) training grid of MOSS-Image; with `shift` sampling, `3.0` approximates the pre-training distribution as described above. The official sampler applies a shift of 5.0 at inference time only (see [Section 7.4](#74-sample-image-generation--サンプル画像生成)).

#### Loss Weighting

The `--weighting_scheme` option specifies loss weighting by timestep:

- `uniform` (default): Equal weight for all timesteps.
- `sigma_sqrt`: Weight by `sigma^(-2)`.
- `cosmap`: Weight by `2 / (pi * (1 - 2*sigma + 2*sigma^2))`.
- `none`: Same as uniform.
- `logit_normal`, `mode`: Additional schemes from SD3 training. See the [`sd3_train_network.md` guide](sd3_train_network.md) for details.

#### Visualizing the Timestep Distribution

To check how the above settings actually affect the sampled timesteps, two features are available (shared with FLUX / Anima training):

* **Training log:** At the start of every run, a one-line summary of the timestep sampling configuration is logged. It explicitly states whether `--discrete_flow_shift` is applied for the chosen `--timestep_sampling`.
* `--show_timesteps=<console|image>`: Visualize the *actual* sampled-timestep distribution and the loss weighting for the current settings, then exit without training. `console` prints an ASCII histogram; `image` shows a matplotlib plot (requires `matplotlib`).
* `--show_timesteps_resolution=<H | H,W>`: Image resolution assumed by `--show_timesteps` for resolution-dependent sampling such as `flux_shift`. Default `1024`.

#### Caption Dropout

Caption dropout uses the `caption_dropout_rate` setting from the dataset configuration (per-subset in TOML). When using `--cache_text_encoder_outputs`, the unconditional embedding (the encoded empty prompt) is stored in every cache file and the dropout is applied at training time, so caption dropout is compatible with text encoder output caching and the rate can be changed without regenerating the cache. The official pipeline uses the empty prompt as the default negative prompt for CFG, so caption dropout is useful if you plan to use CFG with an empty negative prompt.

<details>
<summary>日本語</summary>

#### タイムステップサンプリング

`--timestep_sampling` でタイムステップのサンプリング方法を指定します。FLUX 学習と同じ方法が利用できます。

- `sigma`: SD3 と同様のシグマベースサンプリング。`u = sigmoid(N(logit_mean, logit_std))` を、`--discrete_flow_shift` でシフトした離散的な学習グリッドに対応付けます。
- `uniform`: [0, 1] の一様分布からサンプリング。
- `sigmoid`（デフォルト）: 正規分布からサンプリングし、sigmoid を適用。汎用的なオプション。
- `shift`: `sigmoid` と同様だが、離散フローシフトの式を適用。
- `flux_shift`: FLUX 学習で使用される解像度依存のシフト。

詳細は [flux_train_network.py のガイド](flux_train_network.md) を参照してください。

**MOSS-Image の事前学習の分布の再現:** テクニカルレポートによると、事前学習の最終段階では、シフトなしの 1000 ステップのグリッド上で logit-normal 分布（平均 -1.09、標準偏差 0.95）からタイムステップをサンプリングし、loss のタイムステップによる重み付けは行っていません。これは次の指定で再現できます。

```
--timestep_sampling sigma --weighting_scheme logit_normal --logit_mean -1.09 --logit_std 0.95 --discrete_flow_shift 1.0
```

（`--timestep_sampling sigma` の場合、`--weighting_scheme logit_normal` は logit-normal 分布の選択を意味し、loss の重みは 1 になります。）負の平均により、サンプルはノイズが多い側に偏ります。公式のスケジュールはさらに分布を 0.95 で切り詰め、1% の一様サンプルを混ぜていますが、これらは再現していません。当スクリプトの検証はこの設定で行いました。

`--timestep_sampling shift --discrete_flow_shift 3.0` は、より少ないオプションでこれとよく似た分布になるため（`--show_timesteps` で比較できます。下記参照）、代わりの設定として適しています。デフォルトの `sigmoid`（平均 0、シフトなし）でも学習はできますが、分布の中心が t=500 付近になり、事前学習の分布とはかなり異なります。

#### 離散フローシフト

`--discrete_flow_shift`（デフォルト `1.0`）は `--timestep_sampling` が `sigma` または `shift` の場合に適用されます。`sigma` サンプリングでは `1.0` が MOSS-Image の（シフトなしの）学習グリッドに一致します。`shift` サンプリングでは、上記のとおり `3.0` が事前学習の分布の近似になります。公式のサンプラーは推論時のみシフト 5.0 を適用します（[7.4 節](#74-sample-image-generation--サンプル画像生成) を参照）。

#### 損失の重み付け

`--weighting_scheme` でタイムステップごとの損失の重み付けを指定します。詳細は [`sd3_train_network.md`](sd3_train_network.md) を参照してください。

#### タイムステップ分布の可視化

学習開始時のログに、タイムステップサンプリング設定の概要が 1 行出力されます。また `--show_timesteps=<console|image>` で、現在の設定で実際にサンプリングされるタイムステップ分布と loss weighting を可視化して終了します（学習は行いません）。`--show_timesteps_resolution` で想定する解像度を指定できます（デフォルト `1024`）。

#### キャプションドロップアウト

キャプションドロップアウトにはデータセット設定（TOML でのサブセット単位）の `caption_dropout_rate` を使用します。`--cache_text_encoder_outputs` 使用時は、unconditional の埋め込み（空のプロンプトのエンコード結果）が各キャッシュファイルに保存され、ドロップアウトは学習時に適用されます。このため、テキストエンコーダー出力のキャッシュと同時に使用でき、率を変更してもキャッシュの再生成は不要です。公式パイプラインは CFG のデフォルトの negative prompt として空のプロンプトを使うため、空の negative prompt で CFG を使う予定があればキャプションドロップアウトが有効です。

</details>

### 7.3. Text Encoder LoRA Support / Text Encoder LoRA のサポート

MOSS-Image LoRA training supports training Qwen3.5-9B text encoder LoRA. **The official recommendation is to keep the text encoder frozen**; an info message is logged when the text encoder is trained.

- To train only DiT: specify `--network_train_unet_only` (and `--cache_text_encoder_outputs`, recommended).
- To train DiT and the text encoder: omit `--network_train_unet_only` and do NOT use `--cache_text_encoder_outputs`.

You can specify a separate learning rate for the text encoder with `--text_encoder_lr`. If not specified, the default `--learning_rate` is used.

The default target modules of the text encoder are:

* `Qwen3_5Attention` (the 8 full-attention layers): `q_proj`, `k_proj`, `v_proj`, `o_proj`
* `Qwen3_5GatedDeltaNet` (the 24 linear-attention layers): `in_proj_qkv`, `in_proj_z`, `out_proj`. `in_proj_a` and `in_proj_b` (32 output features) are excluded by default (`.*in_proj_[ab]`); use `include_patterns` to add them.
* `Qwen3_5MLP`: `gate_proj`, `up_proj`, `down_proj`

The text encoder is run one caption at a time without padding, in the same way as the official pipeline. Without the `flash-linear-attention` package the Gated DeltaNet layers use the PyTorch fallback of `transformers`, which is slower (about 1.5x the step time of DiT-only training in our tests). The text encoder stays on the GPU during training; see the memory table in [Section 7.1](#71-vram-usage-optimization--vram-使用量の最適化) for the requirements. ConvRot INT8 of the text encoder (`--convrot_int8_text_encoder` or the pre-quantized file) can be combined with text encoder LoRA training.

Note: When `--cache_text_encoder_outputs` is used, text encoder outputs are pre-computed and the text encoder is removed from the GPU, so text encoder LoRA cannot be trained.

<details>
<summary>日本語</summary>

MOSS-Image の LoRA 学習では、Qwen3.5-9B テキストエンコーダーの LoRA も学習できます。**公式はテキストエンコーダーの凍結を推奨しています。** テキストエンコーダーを学習する場合は info ログが出力されます。

- DiT のみ学習: `--network_train_unet_only` を指定（`--cache_text_encoder_outputs` の併用を推奨）
- DiT とテキストエンコーダーを学習: `--network_train_unet_only` を省略し、`--cache_text_encoder_outputs` を使用しない

テキストエンコーダーに個別の学習率を指定するには `--text_encoder_lr` を使用します。未指定の場合は `--learning_rate` が使われます。

テキストエンコーダーのデフォルトの対象モジュールは以下の通りです。

* `Qwen3_5Attention`（8 層の full attention）: `q_proj`、`k_proj`、`v_proj`、`o_proj`
* `Qwen3_5GatedDeltaNet`（24 層の linear attention）: `in_proj_qkv`、`in_proj_z`、`out_proj`。`in_proj_a` と `in_proj_b`（出力 32 次元）はデフォルトで除外されます（`.*in_proj_[ab]`）。含めるには `include_patterns` を使用してください。
* `Qwen3_5MLP`: `gate_proj`、`up_proj`、`down_proj`

テキストエンコーダーは公式パイプラインと同様に、パディングなしで 1 件ずつ実行されます。`flash-linear-attention` パッケージがない環境では、Gated DeltaNet 層は `transformers` の PyTorch フォールバックで動作し、遅くなります（検証では DiT のみの学習の約 1.5 倍のステップ時間）。学習中はテキストエンコーダーが GPU に常駐します。必要なメモリは [7.1 節](#71-vram-usage-optimization--vram-使用量の最適化) の表を参照してください。テキストエンコーダーの ConvRot INT8（`--convrot_int8_text_encoder` または事前量子化済みファイル）は、テキストエンコーダー LoRA の学習と併用できます。

注意: `--cache_text_encoder_outputs` を使用する場合、テキストエンコーダーの出力が事前に計算され GPU から解放されるため、テキストエンコーダー LoRA は学習できません。

</details>

### 7.4. Sample Image Generation / サンプル画像生成

Sample images can be generated during training with `--sample_prompts`, `--sample_every_n_epochs` / `--sample_every_n_steps` and `--sample_at_first`, in the same way as the other scripts (see the [train_network.py guide](train_network.md)). The sampler follows the official pipeline: Euler steps on the shifted sampling grid snapped to the training grid, CFG with the empty prompt as the default negative prompt, and block 22 skipped in the unconditional branch.

The defaults of the per-prompt options are those of the official pipeline:

* `--s` steps: `50`
* `--l` CFG scale: `3.0`
* `--fs` flow shift of the sampler: `5.0`
* `--n` negative prompt: empty (the official default)
* `--w` / `--h` size: `1024`
* `--d` seed

Example prompt file:

```text
A photo of a refrigerator door with a handwritten note held by a magnet, kitchen interior, soft daylight. --w 1024 --h 1024 --s 20 --l 3.0 --d 42
```

`\n` in a prompt is converted to a line break. The block indices skipped in the unconditional branch are set with `--sample_uncond_skip_layers` (default `22`; an empty string disables skipping). When `--cache_text_encoder_outputs` is used, the text encoder outputs of the sample prompts are computed and cached before the text encoder is removed from the GPU, so the sample prompt file must be given at the start of training.

<details>
<summary>日本語</summary>

他のスクリプトと同様に、`--sample_prompts`、`--sample_every_n_epochs` / `--sample_every_n_steps`、`--sample_at_first` で学習中にサンプル画像を生成できます（[train_network.py のガイド](train_network.md) を参照）。サンプラーは公式パイプラインに従います。シフト付きのサンプリンググリッドを学習グリッドに丸めた上での Euler ステップ、空のプロンプトをデフォルトの negative prompt とする CFG、unconditional 側でのブロック 22 のスキップです。

プロンプトごとのオプションのデフォルトは公式パイプラインの値です。

* `--s` ステップ数: `50`
* `--l` CFG スケール: `3.0`
* `--fs` サンプラーの flow shift: `5.0`
* `--n` negative prompt: 空（公式のデフォルト）
* `--w` / `--h` サイズ: `1024`
* `--d` シード

プロンプト内の `\n` は改行に変換されます。unconditional 側で飛ばすブロックは `--sample_uncond_skip_layers` で指定します（デフォルト `22`、空文字列で無効）。`--cache_text_encoder_outputs` 使用時は、サンプルプロンプトのテキストエンコーダー出力はテキストエンコーダーが GPU から解放される前に計算・保存されるため、サンプルプロンプトファイルは学習開始時に指定しておく必要があります。

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

- **Per-subset timestep sampling offset** (`custom_attributes.timestep_sampling.offset` in the dataset config) is supported. See [the documentation](timestep_sampling_offset.md).

<details>
<summary>日本語</summary>

- **`--loss_type`**: 学習に用いる損失関数。デフォルト `l2`。`l1`, `l2`, `huber`, `smooth_l1` から選択。
- **`--huber_schedule`**, **`--huber_c`**, **`--huber_scale`**: Huber 損失のパラメータ。
- **`--ip_noise_gamma`**: Input Perturbation ノイズガンマ値。
- **`--fused_backward_pass`**: バックワードパスとオプティマイザステップの融合。
- **`--weighting_scheme`** 等: タイムステップ損失の重み付け。詳細は [`sd3_train_network.md`](sd3_train_network.md) を参照。
- データセット設定のサブセット単位の timestep sampling offset（`custom_attributes.timestep_sampling.offset`）に対応しています。[ドキュメント](timestep_sampling_offset.md) を参照。

</details>

## 9. Related Tools / 関連ツール

### `moss_image_minimal_inference.py`

A minimal inference script for MOSS-Image. It loads the DiT, the text encoder and the VAE in the same formats as the training script, and merges LoRA weights at load time.

```bash
python moss_image_minimal_inference.py \
  --dit "<path to DiT>" --text_encoder "<path to Qwen3.5-9B>" --vae "<path to flux2-vae.safetensors>" \
  --prompt "A cat sitting on a windowsill" --image_size 1024 1024 --infer_steps 50 --guidance_scale 3.0 \
  --seed 42 --save_path output \
  --lora_weight my_moss_lora.safetensors --lora_multiplier 1.0
```

Main options:

* `--dit`, `--text_encoder`, `--vae`: Model files (same formats as training). `--tokenizer_cache_dir` as in training.
* `--prompt`, `--negative_prompt` (default empty), `--image_size` (height width, multiples of 16), `--infer_steps` (default 50), `--guidance_scale` (default 3.0), `--flow_shift` (default 5.0), `--uncond_skip_layers` (default `22`), `--seed`.
* `--lora_weight` / `--lora_multiplier`: LoRA files and multipliers (multiple allowed). DiT LoRA (`lora_unet_*`) and text encoder LoRA (`lora_te_*`) are merged into the base weights at load time. Merging into a pre-quantized ConvRot INT8 checkpoint is not possible; use the bf16 checkpoint with `--convrot_int8`.
* `--convrot_int8`, `--convrot_int8_text_encoder`: ConvRot INT8 quantization (pre-quantized files are detected automatically). `--blocks_to_swap`: block swap of the DiT. `--text_encoder_cpu`: run the text encoder on the CPU. `--attn_mode`: attention implementation.
* `--from_file`: read prompts from a file (one per line, with the same per-prompt options as the sample prompts). `--interactive`: read prompts from the console. `--output_type latent` / `--latent_path`: save latents and decode them later.

### `tools/cache_latents.py` / `tools/cache_text_encoder_outputs.py`

Both tools support MOSS-Image with the `--moss_image` option, so the caches can be created before training (and shared between runs):

```bash
python tools/cache_latents.py --moss_image --vae "<path to flux2-vae.safetensors>" --dataset_config my_moss_dataset_config.toml --cache_latents_to_disk
python tools/cache_text_encoder_outputs.py --moss_image --text_encoder "<path to Qwen3.5-9B>" --dataset_config my_moss_dataset_config.toml --cache_text_encoder_outputs_to_disk [--convrot_int8_text_encoder]
```

The text encoder is always run in bf16 (or ConvRot INT8) and the VAE in fp32 regardless of `--mixed_precision`.

<details>
<summary>日本語</summary>

**`moss_image_minimal_inference.py`**

MOSS-Image の最小限の推論スクリプトです。学習スクリプトと同じ形式で DiT、テキストエンコーダー、VAE を読み込み、LoRA の重みをロード時にマージします。コマンド例は英語のドキュメントを参照してください。

主なオプション:

* `--dit`、`--text_encoder`、`--vae`: モデルファイル（学習時と同じ形式）。`--tokenizer_cache_dir` も学習時と同様。
* `--prompt`、`--negative_prompt`（デフォルト空）、`--image_size`（高さ 幅、16 の倍数）、`--infer_steps`（デフォルト 50）、`--guidance_scale`（デフォルト 3.0）、`--flow_shift`（デフォルト 5.0）、`--uncond_skip_layers`（デフォルト `22`）、`--seed`。
* `--lora_weight` / `--lora_multiplier`: LoRA ファイルと倍率（複数指定可）。DiT の LoRA（`lora_unet_*`）とテキストエンコーダーの LoRA（`lora_te_*`）がロード時にベースの重みへマージされます。事前量子化済みの ConvRot INT8 チェックポイントへのマージはできません。bf16 のチェックポイントに `--convrot_int8` を付けて使用してください。
* `--convrot_int8`、`--convrot_int8_text_encoder`: ConvRot INT8 量子化（事前量子化済みファイルは自動判別）。`--blocks_to_swap`: DiT の block swap。`--text_encoder_cpu`: テキストエンコーダーを CPU で実行。`--attn_mode`: attention の実装。
* `--from_file`: ファイルからプロンプトを読み込み（1 行 1 プロンプト、サンプルプロンプトと同じプロンプトごとのオプションが使えます）。`--interactive`: コンソールからプロンプトを入力。`--output_type latent` / `--latent_path`: latent の保存と後からのデコード。

**`tools/cache_latents.py` / `tools/cache_text_encoder_outputs.py`**

どちらも `--moss_image` オプションで MOSS-Image に対応しており、学習前にキャッシュを作成できます（複数の学習で共用できます）。コマンド例は英語のドキュメントを参照してください。`--mixed_precision` の指定に関係なく、テキストエンコーダーは bf16（または ConvRot INT8）、VAE は fp32 で実行されます。

</details>

## 10. Others / その他

### Metadata Saved in LoRA Models

The following metadata is saved in the LoRA model file in addition to the common ones:

* `ss_weighting_scheme`
* `ss_logit_mean`
* `ss_logit_std`
* `ss_mode_scale`
* `ss_timestep_sampling`
* `ss_sigmoid_scale`
* `ss_discrete_flow_shift`
* `ss_convrot_int8` / `ss_convrot_int8_text_encoder`: whether the DiT / text encoder base weights were ConvRot INT8 during training (including pre-quantized checkpoints)

The model spec metadata (`modelspec.architecture`) is `moss-image-base`.

<details>
<summary>日本語</summary>

`moss_image_train_network.py` には、詳細なオプティマイザ設定など、`train_network.py` と共通の機能も多く存在します。これらについては、[`train_network.py` のガイド](train_network.md#5-other-features--その他の機能) やスクリプトのヘルプ (`python moss_image_train_network.py --help`) を参照してください。

### LoRA モデルに保存されるメタデータ

共通のメタデータに加えて、以下が LoRA モデルファイルに保存されます。

* `ss_weighting_scheme`
* `ss_logit_mean`
* `ss_logit_std`
* `ss_mode_scale`
* `ss_timestep_sampling`
* `ss_sigmoid_scale`
* `ss_discrete_flow_shift`
* `ss_convrot_int8` / `ss_convrot_int8_text_encoder`: 学習時に DiT / テキストエンコーダーのベース重みが ConvRot INT8 だったか（事前量子化済みチェックポイントを含む）

model spec のメタデータ（`modelspec.architecture`）は `moss-image-base` です。

</details>
