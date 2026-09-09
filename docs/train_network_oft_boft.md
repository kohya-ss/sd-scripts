# OFTv2 / BOFT Training / OFTv2・BOFT の学習

This document explains how to train OFTv2 and BOFT adapters with `train_network.py` (SD1.x / SD2.x) and `sdxl_train_network.py` (SDXL). Both are orthogonal fine-tuning methods: instead of adding a low-rank update like LoRA, they multiply the input side of each target layer by a learned orthogonal (rotation) matrix. The pre-trained weights are only rotated, so the norm and pairwise angles of the weight vectors are preserved.

These modules were contributed by umisetokikaze in [PR #2357](https://github.com/kohya-ss/sd-scripts/pull/2357). The implementation follows the [PEFT](https://github.com/huggingface/peft) OFT / BOFT layers, and weights in PEFT format can be loaded.

The legacy `networks.oft` module (output-side rotation, different weight format) is unchanged and remains available.

<details>
<summary>日本語</summary>

このドキュメントでは、`train_network.py`（SD1.x / SD2.x）および `sdxl_train_network.py`（SDXL）で OFTv2 と BOFT のアダプタを学習する方法を説明します。いずれも直交変換によるファインチューニング手法で、LoRA のように低ランクの差分を加えるのではなく、対象層の入力側に学習した直交（回転）行列を掛けます。事前学習済み重みは回転されるだけなので、重みベクトルのノルムと相互の角度が保存されます。

これらのモジュールは umisetokikaze 氏により [PR #2357](https://github.com/kohya-ss/sd-scripts/pull/2357) で追加されました。実装は [PEFT](https://github.com/huggingface/peft) の OFT / BOFT 層に準拠しており、PEFT 形式の重みを読み込めます。

従来の `networks.oft` モジュール（出力側の回転、異なる重み形式）は変更されておらず、引き続き利用できます。

</details>

## 1. Overview / 概要

| Module | Method | Trainable parameters per layer |
|---|---|---|
| `networks.oft_v2` | OFTv2 (block-diagonal orthogonal transform, Cayley parameterization) | `oft_R.weight` |
| `networks.boft` | BOFT (butterfly-factorized orthogonal transform with per-output scale) | `boft_R`, `boft_s` |

Supported models: SD1.x, SD2.x, SDXL (U-Net and Text Encoders). FLUX.1, SD3, Lumina, HunyuanImage and Anima are not supported.

Target layers are the same as `networks.lora`: all Linear and 1x1 Conv2d layers inside `Transformer2DModel` blocks of the U-Net, and attention / MLP layers of the Text Encoders. 3x3 Conv2d layers can be added with `enable_conv=true`.

<details>
<summary>日本語</summary>

| モジュール | 手法 | 層ごとの学習パラメータ |
|---|---|---|
| `networks.oft_v2` | OFTv2（ブロック対角の直交変換、Cayley パラメータ化） | `oft_R.weight` |
| `networks.boft` | BOFT（バタフライ分解した直交変換と出力ごとのスケール） | `boft_R`、`boft_s` |

対応モデル: SD1.x、SD2.x、SDXL（U-Net と Text Encoder）。FLUX.1、SD3、Lumina、HunyuanImage、Anima には対応していません。

対象層は `networks.lora` と同じで、U-Net の `Transformer2DModel` ブロック内の全 Linear と 1x1 Conv2d、Text Encoder の attention / MLP 層です。`enable_conv=true` で 3x3 Conv2d も対象に追加できます。

</details>

## 2. Command Line Arguments / コマンドライン引数

Specify the module with `--network_module`. `--network_alpha` is ignored by both modules.

**Note on `--network_dim`:** for `networks.oft_v2` and `networks.boft`, `--network_dim` is interpreted as the **block size** (the size of each orthogonal block). This differs from the legacy `networks.oft`, where it means the number of blocks. If `--network_dim` is omitted, the default block size is 32 for OFTv2 and 4 for BOFT.

`--network_dropout` is mapped to the multiplicative dropout of the rotation blocks (randomly replacing blocks with the identity during training).

<details>
<summary>日本語</summary>

`--network_module` でモジュールを指定します。`--network_alpha` はどちらのモジュールでも無視されます。

**`--network_dim` について:** `networks.oft_v2` と `networks.boft` では、`--network_dim` は**ブロックサイズ**（直交ブロック 1 つの大きさ）として解釈されます。ブロック数を意味する従来の `networks.oft` とは異なります。`--network_dim` を省略した場合のデフォルトは OFTv2 が 32、BOFT が 4 です。

`--network_dropout` は回転ブロックの乗算的ドロップアウト（学習中にブロックをランダムに単位行列へ置き換える）に対応付けられます。

</details>

### 2.1. `networks.oft_v2` options (`--network_args`)

| Argument | Default | Description |
|---|---|---|
| `block_size` (or `oft_block_size`) | `--network_dim` (32) | Size of each orthogonal block. Overrides `--network_dim`. |
| `coft` (or `oft_coft`) | `false` | Enable constrained OFT (COFT). The rotation is projected onto an ε-ball each step. |
| `coft_eps` | `6e-5` | Constraint radius for COFT. |
| `block_share` (or `oft_block_share`) | `false` | Share one rotation block across all blocks of a layer (fewer parameters). |
| `dropout` (or `dropout_probability`) | `0.0` | Block dropout probability. `--network_dropout` takes precedence if given. |
| `enable_conv` | `false` | Also apply to 3x3 Conv2d layers (ResNet blocks, up/down samplers). |
| `auto_adjust` | `true` | If `in_features` is not divisible by the block size, pick the nearest divisor automatically. If `false`, an error is raised instead. |
| `include_patterns` / `exclude_patterns` | none | Python list of regular expressions matched (full match) against the module name, e.g. `"exclude_patterns=['.*attn2.*']"`. |

<details>
<summary>日本語</summary>

| 引数 | デフォルト | 説明 |
|---|---|---|
| `block_size`（または `oft_block_size`） | `--network_dim`（32） | 直交ブロック 1 つの大きさ。`--network_dim` より優先されます。 |
| `coft`（または `oft_coft`） | `false` | 制約付き OFT (COFT) を有効にします。毎ステップ回転を ε 球へ射影します。 |
| `coft_eps` | `6e-5` | COFT の制約半径。 |
| `block_share`（または `oft_block_share`） | `false` | 層内の全ブロックで 1 つの回転ブロックを共有します（パラメータ数が減ります）。 |
| `dropout`（または `dropout_probability`） | `0.0` | ブロックドロップアウトの確率。`--network_dropout` が指定されていればそちらが優先されます。 |
| `enable_conv` | `false` | 3x3 Conv2d 層（ResNet ブロック、up/down サンプラー）にも適用します。 |
| `auto_adjust` | `true` | `in_features` がブロックサイズで割り切れない場合、最も近い約数を自動で選びます。`false` の場合はエラーになります。 |
| `include_patterns` / `exclude_patterns` | なし | モジュール名に対して完全一致で照合する正規表現の Python リスト。例: `"exclude_patterns=['.*attn2.*']"` |

</details>

### 2.2. `networks.boft` options (`--network_args`)

| Argument | Default | Description |
|---|---|---|
| `block_size` (or `boft_block_size`) | `--network_dim` (4) | Size of each orthogonal block. Specify either this or `block_num`, not both. |
| `block_num` (or `boft_block_num`) | none | Number of blocks per layer. The block size is derived from `in_features`. |
| `boft_n_butterfly_factor` | `1` | Number of butterfly factors. `1` is a plain block-diagonal OFT with a scale; larger values add butterfly permutations. With values above 1, the block size and block count must be even and `in_features` must be divisible by `block_size * 2^(factor-1)`. |
| `dropout` (or `boft_dropout`) | `0.0` | Block dropout probability. `--network_dropout` takes precedence if given. |
| `enable_conv` | `false` | Also apply to 3x3 Conv2d layers. |
| `auto_adjust` | `true` | Automatically pick a valid block shape when the requested one does not fit the layer. |
| `include_patterns` / `exclude_patterns` | none | Same as OFTv2. |

<details>
<summary>日本語</summary>

| 引数 | デフォルト | 説明 |
|---|---|---|
| `block_size`（または `boft_block_size`） | `--network_dim`（4） | 直交ブロック 1 つの大きさ。`block_num` とどちらか一方のみ指定します。 |
| `block_num`（または `boft_block_num`） | なし | 層ごとのブロック数。ブロックサイズは `in_features` から求めます。 |
| `boft_n_butterfly_factor` | `1` | バタフライ因子の数。`1` はスケール付きの単純なブロック対角 OFT で、大きくするとバタフライ置換が加わります。1 より大きい場合、ブロックサイズとブロック数は偶数、`in_features` は `block_size * 2^(factor-1)` で割り切れる必要があります。 |
| `dropout`（または `boft_dropout`） | `0.0` | ブロックドロップアウトの確率。`--network_dropout` が指定されていればそちらが優先されます。 |
| `enable_conv` | `false` | 3x3 Conv2d 層にも適用します。 |
| `auto_adjust` | `true` | 指定した形状が層に合わない場合、有効なブロック形状を自動で選びます。 |
| `include_patterns` / `exclude_patterns` | なし | OFTv2 と同じです。 |

</details>

## 3. Example / 実行例

SDXL, OFTv2, U-Net only:

```bash
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py \
  --pretrained_model_name_or_path="/path/to/sdxl_model.safetensors" \
  --dataset_config="/path/to/config.toml" \
  --output_dir="./output" --output_name="sdxl_oftv2" --save_model_as=safetensors \
  --network_module=networks.oft_v2 --network_dim=32 \
  --network_args "coft=false" "block_share=false" \
  --network_train_unet_only \
  --learning_rate=1e-4 --max_train_steps=1000 --train_batch_size=1 \
  --mixed_precision=fp16 --sdpa --gradient_checkpointing --cache_latents
```

BOFT with two butterfly factors:

```bash
  --network_module=networks.boft --network_dim=8 \
  --network_args "boft_n_butterfly_factor=2"
```

Other options (`--unet_lr`, `--text_encoder_lr`, `--network_weights`, sample generation, etc.) work the same way as for LoRA. See [train_network.md](./train_network.md) and [sdxl_train_network.md](./sdxl_train_network.md).

<details>
<summary>日本語</summary>

SDXL、OFTv2、U-Net のみの学習例は上記のとおりです。BOFT でバタフライ因子を 2 にする場合は 2 つ目の例のように指定します。

その他のオプション（`--unet_lr`、`--text_encoder_lr`、`--network_weights`、サンプル生成など）は LoRA と同様に動作します。[train_network.md](./train_network.md) と [sdxl_train_network.md](./sdxl_train_network.md) を参照してください。

</details>

## 4. Weight Format and Compatibility / 重み形式と互換性

Weights are saved with the same key prefixes as LoRA (`lora_unet_...`, `lora_te_...`, `lora_te1_...`, `lora_te2_...`), followed by the parameter name, for example `lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.oft_R.weight`.

When loading (`--network_weights`, `gen_img.py`, merging), the following key formats are recognized in addition to the native one:

* OMI-style prefixes `unet.` / `clip_l.` / `clip_g.` with the original dotted module name.
* PEFT adapter format `base_model.model.<module name>.oft_R.weight` / `.boft_R` / `.boft_s`. A PEFT adapter directory containing `adapter_model.safetensors` can be passed directly.

Keys that do not match any target module are ignored with a warning. When loading PEFT weights into SDXL, note that PEFT keys do not distinguish the two Text Encoders.

<details>
<summary>日本語</summary>

重みは LoRA と同じプレフィックス（`lora_unet_...`、`lora_te_...`、`lora_te1_...`、`lora_te2_...`）にパラメータ名を付けたキーで保存されます。例: `lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q.oft_R.weight`

読み込み時（`--network_weights`、`gen_img.py`、マージ）には、native 形式に加えて次のキー形式を認識します。

* OMI 風のプレフィックス `unet.` / `clip_l.` / `clip_g.` と、元のドット区切りモジュール名。
* PEFT アダプタ形式 `base_model.model.<モジュール名>.oft_R.weight` / `.boft_R` / `.boft_s`。`adapter_model.safetensors` を含む PEFT アダプタのディレクトリを直接指定できます。

対象モジュールに一致しないキーは警告を出して無視されます。PEFT 形式の重みを SDXL に読み込む場合、PEFT のキーは 2 つの Text Encoder を区別しない点に注意してください。

</details>

## 5. Notes / 注意点

* **Speed:** the rotation matrices are rebuilt from the parameters at every forward pass with many small matrix operations. Per-step overhead is therefore higher than LoRA, and GPU utilization may stay low at small batch sizes. Larger batch sizes amortize this overhead. `--gradient_checkpointing` recomputes the rotations during backward as well.
* **Merging:** `merge_to` is supported, so the adapters can be merged into the base weights (for example in `gen_img.py`). The merged weights are exactly the rotated original weights.
* `auto_adjust=true` (default) may silently choose a different block size for some layers when the requested size does not divide `in_features`. The chosen shapes are inferred from the saved weights when loading, so this does not affect compatibility with your own checkpoints.

<details>
<summary>日本語</summary>

* **速度:** 回転行列は毎 forward でパラメータから多数の小さな行列演算により再構築されます。そのため LoRA よりステップあたりのオーバーヘッドが大きく、小さいバッチサイズでは GPU 使用率が上がらないことがあります。バッチサイズを大きくするとこのオーバーヘッドは相対的に小さくなります。`--gradient_checkpointing` を使うと backward 時にも回転の再計算が行われます。
* **マージ:** `merge_to` に対応しているため、アダプタをベースの重みにマージできます（`gen_img.py` など）。マージ後の重みは元の重みを回転したものそのものです。
* `auto_adjust=true`（デフォルト）では、指定したブロックサイズが `in_features` を割り切らない層で、別のブロックサイズが自動的に選ばれます。読み込み時には保存された重みから形状を推定するため、自分で学習したチェックポイントとの互換性には影響しません。

</details>
