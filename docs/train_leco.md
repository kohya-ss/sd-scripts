# LECO Training Guide / LECO 学習ガイド

LECO (Low-rank adaptation for Erasing COncepts from diffusion models) is a technique for training LoRA models that modify or erase concepts from a diffusion model **without requiring any image dataset**. It works by training a LoRA against the model's own noise predictions using text prompts only.

This repository provides two LECO training scripts:

- `train_leco.py` for Stable Diffusion 1.x / 2.x
- `sdxl_train_leco.py` for SDXL

<details>
<summary>日本語</summary>

LECO (Low-rank adaptation for Erasing COncepts from diffusion models) は、**画像データセットを一切必要とせず**、テキストプロンプトのみを使用してモデル自身のノイズ予測に対して LoRA を学習させる手法です。拡散モデルから概念を変更・消去する LoRA モデルを作成できます。

このリポジトリでは以下の2つの LECO 学習スクリプトを提供しています：

- `train_leco.py` : Stable Diffusion 1.x / 2.x 用
- `sdxl_train_leco.py` : SDXL 用
</details>

## 1. Overview / 概要

### What LECO Can Do / LECO でできること

LECO can be used for:

- **Concept erasing**: Remove a specific style or concept (e.g., erase "van gogh" style from generated images)
- **Concept enhancing**: Strengthen a specific attribute (e.g., make "detailed" more pronounced)
- **Slider LoRA**: Create a LoRA that controls an attribute bidirectionally (e.g., a slider between "short hair" and "long hair")

Unlike standard LoRA training, LECO does not use any training images. All training signals come from the difference between the model's own noise predictions on different text prompts.

<details>
<summary>日本語</summary>

LECO は以下の用途に使用できます：

- **概念の消去**: 特定のスタイルや概念を除去する（例：生成画像から「van gogh」スタイルを消去）
- **概念の強化**: 特定の属性を強化する（例：「detailed」をより顕著にする）
- **スライダー LoRA**: 属性を双方向に制御する LoRA を作成する（例：「short hair」と「long hair」の間のスライダー）

通常の LoRA 学習とは異なり、LECO は学習画像を一切使用しません。学習のシグナルは全て、異なるテキストプロンプトに対するモデル自身のノイズ予測の差分から得られます。
</details>

### Key Differences from Standard LoRA Training / 通常の LoRA 学習との違い

| | Standard LoRA | LECO |
|---|---|---|
| Training data | Image dataset required | **No images needed** |
| Configuration | Dataset TOML | Prompt TOML |
| Training target | U-Net and/or Text Encoder | **U-Net only** |
| Training unit | Epochs and steps | **Steps only** |
| Saving | Per-epoch or per-step | **Per-step only** (`--save_every_n_steps`) |

<details>
<summary>日本語</summary>

| | 通常の LoRA | LECO |
|---|---|---|
| 学習データ | 画像データセットが必要 | **画像不要** |
| 設定ファイル | データセット TOML | プロンプト TOML |
| 学習対象 | U-Net と Text Encoder | **U-Net のみ** |
| 学習単位 | エポックとステップ | **ステップのみ** |
| 保存 | エポック毎またはステップ毎 | **ステップ毎のみ** (`--save_every_n_steps`) |
</details>

## 2. Prompt Configuration File / プロンプト設定ファイル

LECO uses a TOML file to define training prompts. Two formats are supported: the **original LECO format** and the **slider target format** (ai-toolkit style).

<details>
<summary>日本語</summary>
LECO は学習プロンプトの定義に TOML ファイルを使用します。**オリジナル LECO 形式**と**スライダーターゲット形式**（ai-toolkit スタイル）の2つの形式に対応しています。
</details>

### 2.1. Original LECO Format / オリジナル LECO 形式

Use `[[prompts]]` sections to define prompt pairs directly. This gives you full control over each training pair.

```toml
[[prompts]]
target = "van gogh"
positive = "van gogh"
unconditional = ""
neutral = ""
action = "erase"
guidance_scale = 1.0
resolution = 512
batch_size = 1
multiplier = 1.0
weight = 1.0
```

Each `[[prompts]]` entry defines one training pair with the following fields:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `target` | Yes | - | The concept to be modified by the LoRA |
| `positive` | No | same as `target` | The "positive direction" prompt for building the training target |
| `unconditional` | No | `""` | The unconditional/negative prompt |
| `neutral` | No | `""` | The neutral baseline prompt |
| `action` | No | `"erase"` | `"erase"` to remove the concept, `"enhance"` to strengthen it |
| `guidance_scale` | No | `1.0` | Scale factor for target construction (higher = stronger effect) |
| `resolution` | No | `512` | Training resolution (int or `[height, width]`) |
| `batch_size` | No | `1` | Number of latent samples per training step for this prompt |
| `multiplier` | No | `1.0` | LoRA strength multiplier during training |
| `weight` | No | `1.0` | Loss weight for this prompt pair |

<details>
<summary>日本語</summary>

`[[prompts]]` セクションを使用して、プロンプトペアを直接定義します。各学習ペアを細かく制御できます。

各 `[[prompts]]` エントリのフィールド：

| フィールド | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `target` | はい | - | LoRA によって変更される概念 |
| `positive` | いいえ | `target` と同じ | 学習ターゲット構築時の「正方向」プロンプト |
| `unconditional` | いいえ | `""` | 無条件/ネガティブプロンプト |
| `neutral` | いいえ | `""` | ニュートラルベースラインプロンプト |
| `action` | いいえ | `"erase"` | `"erase"` で概念を除去、`"enhance"` で強化 |
| `guidance_scale` | いいえ | `1.0` | ターゲット構築時のスケール係数（大きいほど効果が強い） |
| `resolution` | いいえ | `512` | 学習解像度（整数または `[height, width]`） |
| `batch_size` | いいえ | `1` | このプロンプトの学習ステップごとの latent サンプル数 |
| `multiplier` | いいえ | `1.0` | 学習時の LoRA 強度乗数 |
| `weight` | いいえ | `1.0` | このプロンプトペアの loss 重み |
</details>

### 2.2. Slider Target Format / スライダーターゲット形式

Use `[[targets]]` sections to define slider-style LoRAs. Each target is automatically expanded into bidirectional training pairs (4 pairs when both `positive` and `negative` are provided, 2 pairs when only one is provided).

```toml
guidance_scale = 1.0
resolution = 1024
neutral = ""

[[targets]]
target_class = "1girl"
positive = "1girl, long hair"
negative = "1girl, short hair"
multiplier = 1.0
weight = 1.0
```

Top-level fields (`guidance_scale`, `resolution`, `neutral`, `batch_size`, etc.) serve as defaults for all targets.

Each `[[targets]]` entry supports the following fields:

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `target_class` | Yes | - | The base class/subject prompt |
| `positive` | No* | `""` | Prompt for the positive direction of the slider |
| `negative` | No* | `""` | Prompt for the negative direction of the slider |
| `multiplier` | No | `1.0` | LoRA strength multiplier |
| `weight` | No | `1.0` | Loss weight |

\* At least one of `positive` or `negative` must be provided.

<details>
<summary>日本語</summary>

`[[targets]]` セクションを使用してスライダースタイルの LoRA を定義します。各ターゲットは自動的に双方向の学習ペアに展開されます（`positive` と `negative` の両方がある場合は4ペア、片方のみの場合は2ペア）。

トップレベルのフィールド（`guidance_scale`、`resolution`、`neutral`、`batch_size` など）は全ターゲットのデフォルト値として機能します。

各 `[[targets]]` エントリのフィールド：

| フィールド | 必須 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `target_class` | はい | - | ベースとなるクラス/被写体プロンプト |
| `positive` | いいえ* | `""` | スライダーの正方向プロンプト |
| `negative` | いいえ* | `""` | スライダーの負方向プロンプト |
| `multiplier` | いいえ | `1.0` | LoRA 強度乗数 |
| `weight` | いいえ | `1.0` | loss 重み |

\* `positive` と `negative` のうち少なくとも一方を指定する必要があります。
</details>

### 2.3. Multiple Neutral Prompts / 複数のニュートラルプロンプト

You can provide multiple neutral prompts for slider targets. Each neutral prompt generates a separate set of training pairs, which can improve generalization.

```toml
guidance_scale = 1.5
resolution = 1024
neutrals = ["", "photo of a person", "cinematic portrait"]

[[targets]]
target_class = "person"
positive = "smiling person"
negative = "expressionless person"
```

You can also load neutral prompts from a text file (one prompt per line):

```toml
neutral_prompt_file = "neutrals.txt"

[[targets]]
target_class = ""
positive = "high detail"
negative = "low detail"
```

<details>
<summary>日本語</summary>

スライダーターゲットに対して複数のニュートラルプロンプトを指定できます。各ニュートラルプロンプトごとに個別の学習ペアが生成され、汎化性能の向上が期待できます。

ニュートラルプロンプトをテキストファイル（1行1プロンプト）から読み込むこともできます。
</details>

### 2.4. Converting from ai-toolkit YAML / ai-toolkit の YAML からの変換

If you have an existing ai-toolkit style YAML config, convert it to TOML as follows:

<details>
<summary>日本語</summary>
既存の ai-toolkit スタイルの YAML 設定がある場合、以下のように TOML に変換してください。
</details>

**YAML:**
```yaml
targets:
  - target_class: ""
    positive: "high detail"
    negative: "low detail"
    multiplier: 1.0
guidance_scale: 1.0
resolution: 512
```

**TOML:**
```toml
guidance_scale = 1.0
resolution = 512

[[targets]]
target_class = ""
positive = "high detail"
negative = "low detail"
multiplier = 1.0
```

Key syntax differences:

- Use `=` instead of `:` for key-value pairs
- Use `[[targets]]` header instead of `targets:` with `- ` list items
- Arrays use `[brackets]` (e.g., `neutrals = ["a", "b"]`)

<details>
<summary>日本語</summary>

主な構文の違い：

- キーと値の区切りに `:` ではなく `=` を使用
- `targets:` と `- ` のリスト記法ではなく `[[targets]]` ヘッダを使用
- 配列は `[brackets]` で記述（例：`neutrals = ["a", "b"]`）
</details>

## 3. Running the Training / 学習の実行

Training is started by executing the script from the terminal. Below are basic command-line examples.

In reality, you need to write the command in a single line, but it is shown with line breaks for readability. On Linux/Mac, add `\` at the end of each line; on Windows, add `^`.

<details>
<summary>日本語</summary>
学習はターミナルからスクリプトを実行して開始します。以下に基本的なコマンドライン例を示します。

実際には1行で書く必要がありますが、見やすさのために改行しています。Linux/Mac では各行末に `\` を、Windows では `^` を追加してください。
</details>

### SD 1.x / 2.x

```bash
accelerate launch --mixed_precision bf16 train_leco.py
  --pretrained_model_name_or_path="model.safetensors"
  --prompts_file="prompts.toml"
  --output_dir="output"
  --output_name="my_leco"
  --network_dim=8
  --network_alpha=4
  --learning_rate=1e-4
  --optimizer_type="AdamW8bit"
  --max_train_steps=500
  --max_denoising_steps=40
  --mixed_precision=bf16
  --sdpa
  --gradient_checkpointing
  --save_every_n_steps=100
```

### SDXL

```bash
accelerate launch --mixed_precision bf16 sdxl_train_leco.py
  --pretrained_model_name_or_path="sdxl_model.safetensors"
  --prompts_file="slider.toml"
  --output_dir="output"
  --output_name="my_sdxl_slider"
  --network_dim=8
  --network_alpha=4
  --learning_rate=1e-4
  --optimizer_type="AdamW8bit"
  --max_train_steps=1000
  --max_denoising_steps=40
  --mixed_precision=bf16
  --sdpa
  --gradient_checkpointing
  --save_every_n_steps=200
```

## 4. Command-Line Arguments / コマンドライン引数

### 4.1. LECO-Specific Arguments / LECO 固有の引数

These arguments are unique to LECO and not found in standard LoRA training scripts.

<details>
<summary>日本語</summary>
以下の引数は LECO 固有のもので、通常の LoRA 学習スクリプトにはありません。
</details>

* `--prompts_file="prompts.toml"` **[Required]**
  * Path to the LECO prompt configuration TOML file. See [Section 2](#2-prompt-configuration-file--プロンプト設定ファイル) for the file format.

* `--max_denoising_steps=40`
  * Number of partial denoising steps per training iteration. At each step, a random number of denoising steps (from 1 to this value) is performed. Default: `40`.

* `--leco_denoise_guidance_scale=3.0`
  * Guidance scale used during the partial denoising pass. This is separate from `guidance_scale` in the TOML file. Default: `3.0`.

<details>
<summary>日本語</summary>

* `--prompts_file="prompts.toml"` **[必須]**
  * LECO プロンプト設定 TOML ファイルのパス。ファイル形式については[セクション2](#2-prompt-configuration-file--プロンプト設定ファイル)を参照してください。

* `--max_denoising_steps=40`
  * 各学習イテレーションでの部分デノイズステップ数。各ステップで1からこの値の間のランダムなステップ数でデノイズが行われます。デフォルト: `40`。

* `--leco_denoise_guidance_scale=3.0`
  * 部分デノイズ時の guidance scale。TOML ファイル内の `guidance_scale` とは別のパラメータです。デフォルト: `3.0`。
</details>

#### Understanding the Two `guidance_scale` Parameters / 2つの `guidance_scale` の違い

There are two separate guidance scale parameters that control different aspects of LECO training:

1. **`--leco_denoise_guidance_scale` (command-line)**: Controls CFG strength during the partial denoising pass that generates intermediate latents. Higher values produce more prompt-adherent latents for the training signal.

2. **`guidance_scale` (in TOML file)**: Controls the magnitude of the concept offset when constructing the training target. Higher values produce a stronger erase/enhance effect. This can be set per-prompt or per-target.

If training results are too subtle, try increasing the TOML `guidance_scale` (e.g., `1.5` to `3.0`).

<details>
<summary>日本語</summary>

LECO の学習では、異なる役割を持つ2つの guidance scale パラメータがあります：

1. **`--leco_denoise_guidance_scale`（コマンドライン）**: 中間 latent を生成する部分デノイズパスの CFG 強度を制御します。大きな値にすると、プロンプトにより忠実な latent が学習シグナルとして生成されます。

2. **`guidance_scale`（TOML ファイル内）**: 学習ターゲット構築時の概念オフセットの大きさを制御します。大きな値にすると、消去/強化の効果が強くなります。プロンプトごと・ターゲットごとに設定可能です。

学習結果の効果が弱い場合は、TOML の `guidance_scale` を大きくしてみてください（例：`1.5` から `3.0`）。
</details>

### 4.2. Model Arguments / モデル引数

* `--pretrained_model_name_or_path="model.safetensors"` **[Required]**
  * Path to the base Stable Diffusion model (`.ckpt`, `.safetensors`, Diffusers directory, or Hugging Face model ID).

* `--v2` (SD 1.x/2.x only)
  * Specify when using a Stable Diffusion v2.x model.

* `--v_parameterization` (SD 1.x/2.x only)
  * Specify when using a v-prediction model (e.g., SD 2.x 768px models).

<details>
<summary>日本語</summary>

* `--pretrained_model_name_or_path="model.safetensors"` **[必須]**
  * ベースとなる Stable Diffusion モデルのパス（`.ckpt`、`.safetensors`、Diffusers ディレクトリ、Hugging Face モデル ID）。

* `--v2`（SD 1.x/2.x のみ）
  * Stable Diffusion v2.x モデルを使用する場合に指定します。

* `--v_parameterization`（SD 1.x/2.x のみ）
  * v-prediction モデル（SD 2.x 768px モデルなど）を使用する場合に指定します。
</details>

### 4.3. LoRA Network Arguments / LoRA ネットワーク引数

* `--network_module=networks.lora`
  * Network module to train. Default: `networks.lora`.

* `--network_dim=8`
  * LoRA rank (dimension). Higher values increase expressiveness but also file size. Typical values: `4` to `16`. Default: `4`.

* `--network_alpha=4`
  * LoRA alpha for learning rate scaling. A common choice is to set this to half of `network_dim`. Default: `1.0`.

* `--network_dropout=0.1`
  * Dropout rate for LoRA layers. Optional.

* `--network_args "key=value" ...`
  * Additional network-specific arguments. For example, `--network_args "conv_dim=4"` to enable Conv2d LoRA.

* `--network_weights="path/to/weights.safetensors"`
  * Load pretrained LoRA weights to continue training.

* `--dim_from_weights`
  * Infer `network_dim` from the weights specified by `--network_weights`. Requires `--network_weights`.

<details>
<summary>日本語</summary>

* `--network_module=networks.lora`
  * 学習するネットワークモジュール。デフォルト: `networks.lora`。

* `--network_dim=8`
  * LoRA のランク（次元数）。大きいほど表現力が上がりますがファイルサイズも増加します。一般的な値: `4` から `16`。デフォルト: `4`。

* `--network_alpha=4`
  * 学習率スケーリング用の LoRA alpha。`network_dim` の半分程度に設定するのが一般的です。デフォルト: `1.0`。

* `--network_dropout=0.1`
  * LoRA レイヤーのドロップアウト率。省略可。

* `--network_args "key=value" ...`
  * ネットワーク固有の追加引数。例：`--network_args "conv_dim=4"` で Conv2d LoRA を有効にします。

* `--network_weights="path/to/weights.safetensors"`
  * 事前学習済み LoRA ウェイトを読み込んで学習を続行します。

* `--dim_from_weights`
  * `--network_weights` で指定したウェイトから `network_dim` を推定します。`--network_weights` の指定が必要です。
</details>

### 4.4. Training Parameters / 学習パラメータ

* `--max_train_steps=500`
  * Total number of training steps. Default: `1600`. Typical range for LECO: `300` to `2000`.
  * Note: `--max_train_epochs` is **not supported** for LECO (the training loop is step-based only).

* `--learning_rate=1e-4`
  * Learning rate. Typical range for LECO: `1e-4` to `1e-3`.

* `--unet_lr=1e-4`
  * Separate learning rate for U-Net LoRA modules. If not specified, `--learning_rate` is used.

* `--optimizer_type="AdamW8bit"`
  * Optimizer type. Options include `AdamW8bit` (requires `bitsandbytes`), `AdamW`, `Lion`, `Adafactor`, etc.

* `--lr_scheduler="constant"`
  * Learning rate scheduler. Options: `constant`, `cosine`, `linear`, `constant_with_warmup`, etc.

* `--lr_warmup_steps=0`
  * Number of warmup steps for the learning rate scheduler.

* `--gradient_accumulation_steps=1`
  * Number of steps to accumulate gradients before updating. Effectively multiplies the batch size.

* `--max_grad_norm=1.0`
  * Maximum gradient norm for gradient clipping. Set to `0` to disable.

* `--min_snr_gamma=5.0`
  * Min-SNR weighting gamma. Applies SNR-based loss weighting. Optional.

<details>
<summary>日本語</summary>

* `--max_train_steps=500`
  * 学習の総ステップ数。デフォルト: `1600`。LECO の一般的な範囲: `300` から `2000`。
  * 注意: `--max_train_epochs` は LECO では**サポートされていません**（学習ループはステップベースのみです）。

* `--learning_rate=1e-4`
  * 学習率。LECO の一般的な範囲: `1e-4` から `1e-3`。

* `--unet_lr=1e-4`
  * U-Net LoRA モジュール用の個別の学習率。指定しない場合は `--learning_rate` が使用されます。

* `--optimizer_type="AdamW8bit"`
  * オプティマイザの種類。`AdamW8bit`（要 `bitsandbytes`）、`AdamW`、`Lion`、`Adafactor` 等が選択可能です。

* `--lr_scheduler="constant"`
  * 学習率スケジューラ。`constant`、`cosine`、`linear`、`constant_with_warmup` 等が選択可能です。

* `--lr_warmup_steps=0`
  * 学習率スケジューラのウォームアップステップ数。

* `--gradient_accumulation_steps=1`
  * 勾配を累積するステップ数。実質的にバッチサイズを増加させます。

* `--max_grad_norm=1.0`
  * 勾配クリッピングの最大勾配ノルム。`0` で無効化。

* `--min_snr_gamma=5.0`
  * Min-SNR 重み付けのガンマ値。SNR ベースの loss 重み付けを適用します。省略可。
</details>

### 4.5. Output and Save Arguments / 出力・保存引数

* `--output_dir="output"` **[Required]**
  * Directory for saving trained LoRA models and logs.

* `--output_name="my_leco"` **[Required]**
  * Base filename for the trained LoRA (without extension).

* `--save_model_as="safetensors"`
  * Model save format. Options: `safetensors` (default, recommended), `ckpt`, `pt`.

* `--save_every_n_steps=100`
  * Save an intermediate checkpoint every N steps. If not specified, only the final model is saved.
  * Note: `--save_every_n_epochs` is **not supported** for LECO.

* `--save_precision="fp16"`
  * Precision for saving the model. Options: `float`, `fp16`, `bf16`. If not specified, the training precision is used.

* `--no_metadata`
  * Do not write metadata into the saved model file.

* `--training_comment="my comment"`
  * A comment string stored in the model metadata.

<details>
<summary>日本語</summary>

* `--output_dir="output"` **[必須]**
  * 学習済み LoRA モデルとログの保存先ディレクトリ。

* `--output_name="my_leco"` **[必須]**
  * 学習済み LoRA のベースファイル名（拡張子なし）。

* `--save_model_as="safetensors"`
  * モデルの保存形式。`safetensors`（デフォルト、推奨）、`ckpt`、`pt` から選択。

* `--save_every_n_steps=100`
  * N ステップごとに中間チェックポイントを保存。指定しない場合は最終モデルのみ保存されます。
  * 注意: `--save_every_n_epochs` は LECO では**サポートされていません**。

* `--save_precision="fp16"`
  * モデル保存時の精度。`float`、`fp16`、`bf16` から選択。省略時は学習時の精度が使用されます。

* `--no_metadata`
  * 保存するモデルファイルにメタデータを書き込みません。

* `--training_comment="my comment"`
  * モデルのメタデータに保存されるコメント文字列。
</details>

### 4.6. Memory and Performance Arguments / メモリ・パフォーマンス引数

* `--mixed_precision="bf16"`
  * Mixed precision training. Options: `no`, `fp16`, `bf16`. Using `bf16` or `fp16` is recommended.

* `--full_fp16`
  * Train entirely in fp16 precision including gradients.

* `--full_bf16`
  * Train entirely in bf16 precision including gradients.

* `--gradient_checkpointing`
  * Enable gradient checkpointing to reduce VRAM usage at the cost of slightly slower training. **Recommended for LECO**, especially with larger models or higher resolutions.

* `--sdpa`
  * Use Scaled Dot-Product Attention. Reduces memory usage and can improve speed. Recommended.

* `--xformers`
  * Use xformers for memory-efficient attention (requires `xformers` package). Alternative to `--sdpa`.

* `--mem_eff_attn`
  * Use memory-efficient attention implementation. Another alternative to `--sdpa`.

<details>
<summary>日本語</summary>

* `--mixed_precision="bf16"`
  * 混合精度学習。`no`、`fp16`、`bf16` から選択。`bf16` または `fp16` の使用を推奨します。

* `--full_fp16`
  * 勾配を含め全体を fp16 精度で学習します。

* `--full_bf16`
  * 勾配を含め全体を bf16 精度で学習します。

* `--gradient_checkpointing`
  * gradient checkpointing を有効にしてVRAM使用量を削減します（学習速度は若干低下）。特に大きなモデルや高解像度での LECO 学習時に**推奨**です。

* `--sdpa`
  * Scaled Dot-Product Attention を使用します。メモリ使用量を削減し速度向上が期待できます。推奨。

* `--xformers`
  * xformers を使用したメモリ効率の良い attention（`xformers` パッケージが必要）。`--sdpa` の代替。

* `--mem_eff_attn`
  * メモリ効率の良い attention 実装を使用。`--sdpa` の別の代替。
</details>

### 4.7. Other Useful Arguments / その他の便利な引数

* `--seed=42`
  * Random seed for reproducibility. If not specified, a random seed is automatically generated.

* `--noise_offset=0.05`
  * Enable noise offset. Small values like `0.02` to `0.1` can help with training stability.

* `--zero_terminal_snr`
  * Fix noise scheduler betas to enforce zero terminal SNR.

* `--clip_skip=2` (SD 1.x/2.x only)
  * Use the output from the Nth-to-last layer of the text encoder. Common values: `1` (no skip) or `2`.

* `--logging_dir="logs"`
  * Directory for TensorBoard logs. Enables logging when specified.

* `--log_with="tensorboard"`
  * Logging tool. Options: `tensorboard`, `wandb`, `all`.

<details>
<summary>日本語</summary>

* `--seed=42`
  * 再現性のための乱数シード。指定しない場合は自動生成されます。

* `--noise_offset=0.05`
  * ノイズオフセットを有効にします。`0.02` から `0.1` 程度の小さい値で学習の安定性が向上する場合があります。

* `--zero_terminal_snr`
  * noise scheduler の betas を修正してゼロ終端 SNR を強制します。

* `--clip_skip=2`（SD 1.x/2.x のみ）
  * text encoder の後ろから N 番目の層の出力を使用します。一般的な値: `1`（スキップなし）または `2`。

* `--logging_dir="logs"`
  * TensorBoard ログの出力ディレクトリ。指定時にログ出力が有効になります。

* `--log_with="tensorboard"`
  * ログツール。`tensorboard`、`wandb`、`all` から選択。
</details>

## 5. Tips / ヒント

### Tuning the Effect Strength / 効果の強さの調整

If the trained LoRA has a weak or unnoticeable effect:

1. **Increase `guidance_scale` in TOML** (e.g., `1.5` to `3.0`). This is the most direct way to strengthen the effect.
2. **Increase `multiplier` in TOML** (e.g., `1.5` to `2.0`).
3. **Increase `--max_denoising_steps`** for more refined intermediate latents.
4. **Increase `--max_train_steps`** to train longer.
5. **Apply the LoRA with a higher weight** at inference time.

<details>
<summary>日本語</summary>

学習した LoRA の効果が弱い、または認識できない場合：

1. **TOML の `guidance_scale` を上げる**（例：`1.5` から `3.0`）。効果を強める最も直接的な方法です。
2. **TOML の `multiplier` を上げる**（例：`1.5` から `2.0`）。
3. **`--max_denoising_steps` を増やす**。より精緻な中間 latent が生成されます。
4. **`--max_train_steps` を増やして**、より長く学習する。
5. **推論時に LoRA のウェイトを大きくして**適用する。
</details>

### Recommended Starting Settings / 推奨の開始設定

| Parameter | SD 1.x/2.x | SDXL |
|-----------|-------------|------|
| `--network_dim` | `4`-`8` | `8`-`16` |
| `--learning_rate` | `1e-4` | `1e-4` |
| `--max_train_steps` | `300`-`1000` | `500`-`2000` |
| `resolution` (in TOML) | `512` | `1024` |
| `guidance_scale` (in TOML) | `1.0`-`2.0` | `1.0`-`3.0` |
| `batch_size` (in TOML) | `1`-`4` | `1`-`4` |

<details>
<summary>日本語</summary>

| パラメータ | SD 1.x/2.x | SDXL |
|-----------|-------------|------|
| `--network_dim` | `4`-`8` | `8`-`16` |
| `--learning_rate` | `1e-4` | `1e-4` |
| `--max_train_steps` | `300`-`1000` | `500`-`2000` |
| `resolution`（TOML内） | `512` | `1024` |
| `guidance_scale`（TOML内） | `1.0`-`2.0` | `1.0`-`3.0` |
| `batch_size`（TOML内） | `1`-`4` | `1`-`4` |
</details>

### Dynamic Resolution and Crops (SDXL) / 動的解像度とクロップ（SDXL）

For SDXL slider targets, you can enable dynamic resolution and crops in the TOML file:

```toml
resolution = 1024
dynamic_resolution = true
dynamic_crops = true

[[targets]]
target_class = ""
positive = "high detail"
negative = "low detail"
```

- `dynamic_resolution`: Randomly varies the training resolution around the base value using aspect ratio buckets.
- `dynamic_crops`: Randomizes crop positions in the SDXL size conditioning embeddings.

These options can improve the LoRA's generalization across different aspect ratios.

<details>
<summary>日本語</summary>

SDXL のスライダーターゲットでは、TOML ファイルで動的解像度とクロップを有効にできます。

- `dynamic_resolution`: アスペクト比バケツを使用して、ベース値の周囲で学習解像度をランダムに変化させます。
- `dynamic_crops`: SDXL のサイズ条件付け埋め込みでクロップ位置をランダム化します。

これらのオプションにより、異なるアスペクト比に対する LoRA の汎化性能が向上する場合があります。
</details>

## 6. Using the Trained Model / 学習済みモデルの利用

The trained LoRA file (`.safetensors`) is saved in the `--output_dir` directory. It can be used with GUI tools such as AUTOMATIC1111/stable-diffusion-webui, ComfyUI, etc.

For slider LoRAs, apply positive weights (e.g., `0.5` to `1.5`) to move in the positive direction, and negative weights (e.g., `-0.5` to `-1.5`) to move in the negative direction.

<details>
<summary>日本語</summary>

学習済みの LoRA ファイル（`.safetensors`）は `--output_dir` ディレクトリに保存されます。AUTOMATIC1111/stable-diffusion-webui、ComfyUI 等の GUI ツールで使用できます。

スライダー LoRA の場合、正のウェイト（例：`0.5` から `1.5`）で正方向に、負のウェイト（例：`-0.5` から `-1.5`）で負方向に効果を適用できます。
</details>
