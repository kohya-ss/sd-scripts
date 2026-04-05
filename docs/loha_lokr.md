> 📝 Click on the language section to expand / 言語をクリックして展開

# LoHa / LoKr (LyCORIS)

## Overview / 概要

In addition to standard LoRA, sd-scripts supports **LoHa** (Low-rank Hadamard Product) and **LoKr** (Low-rank Kronecker Product) as alternative parameter-efficient fine-tuning methods. These are based on techniques from the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) project.

- **LoHa**: Represents weight updates as a Hadamard (element-wise) product of two low-rank matrices. Reference: [FedPara (arXiv:2108.06098)](https://arxiv.org/abs/2108.06098)
- **LoKr**: Represents weight updates as a Kronecker product with optional low-rank decomposition. Reference: [LoKr (arXiv:2309.14859)](https://arxiv.org/abs/2309.14859)

The algorithms and recommended settings are described in the [LyCORIS documentation](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md) and [guidelines](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md).

Both methods target Linear and Conv2d layers. Conv2d 1x1 layers are treated similarly to Linear layers. For Conv2d 3x3+ layers, optional Tucker decomposition or flat (kernel-flattened) mode is available.

This feature is experimental.

<details>
<summary>日本語</summary>

sd-scriptsでは、標準的なLoRAに加え、代替のパラメータ効率の良いファインチューニング手法として **LoHa**（Low-rank Hadamard Product）と **LoKr**（Low-rank Kronecker Product）をサポートしています。これらは [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) プロジェクトの手法に基づいています。

- **LoHa**: 重みの更新を2つの低ランク行列のHadamard積（要素ごとの積）で表現します。参考文献: [FedPara (arXiv:2108.06098)](https://arxiv.org/abs/2108.06098)
- **LoKr**: 重みの更新をKronecker積と、オプションの低ランク分解で表現します。参考文献: [LoKr (arXiv:2309.14859)](https://arxiv.org/abs/2309.14859)

アルゴリズムと推奨設定は[LyCORISのアルゴリズム解説](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-List.md)と[ガイドライン](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Guidelines.md)を参照してください。

LinearおよびConv2d層の両方を対象としています。Conv2d 1x1層はLinear層と同様に扱われます。Conv2d 3x3+層については、オプションのTucker分解またはflat（カーネル平坦化）モードが利用可能です。

この機能は実験的なものです。

</details>

## Acknowledgments / 謝辞

The LoHa and LoKr implementations in sd-scripts are based on the [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) project by [KohakuBlueleaf](https://github.com/KohakuBlueleaf). We would like to express our sincere gratitude for the excellent research and open-source contributions that made this implementation possible.

<details>
<summary>日本語</summary>

sd-scriptsのLoHaおよびLoKrの実装は、[KohakuBlueleaf](https://github.com/KohakuBlueleaf)氏による[LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS)プロジェクトに基づいています。この実装を可能にしてくださった素晴らしい研究とオープンソースへの貢献に心から感謝いたします。

</details>

## Supported architectures / 対応アーキテクチャ

LoHa and LoKr automatically detect the model architecture and apply appropriate default settings. The following architectures are currently supported:

- **SDXL**: Targets `Transformer2DModel` for UNet and `CLIPAttention`/`CLIPMLP` for text encoders. Conv2d layers in `ResnetBlock2D`, `Downsample2D`, and `Upsample2D` are also supported when `conv_dim` is specified. No default `exclude_patterns`.
- **Anima**: Targets `Block`, `PatchEmbed`, `TimestepEmbedding`, and `FinalLayer` for DiT, and `Qwen3Attention`/`Qwen3MLP` for the text encoder. Default `exclude_patterns` automatically skips modulation, normalization, embedder, and final_layer modules.

<details>
<summary>日本語</summary>

LoHaとLoKrは、モデルのアーキテクチャを自動で検出し、適切なデフォルト設定を適用します。現在、以下のアーキテクチャに対応しています:

- **SDXL**: UNetの`Transformer2DModel`、テキストエンコーダの`CLIPAttention`/`CLIPMLP`を対象とします。`conv_dim`を指定した場合、`ResnetBlock2D`、`Downsample2D`、`Upsample2D`のConv2d層も対象になります。デフォルトの`exclude_patterns`はありません。
- **Anima**: DiTの`Block`、`PatchEmbed`、`TimestepEmbedding`、`FinalLayer`、テキストエンコーダの`Qwen3Attention`/`Qwen3MLP`を対象とします。デフォルトの`exclude_patterns`により、modulation、normalization、embedder、final_layerモジュールは自動的にスキップされます。

</details>

## Training / 学習

To use LoHa or LoKr, change the `--network_module` argument in your training command. All other training options (dataset config, optimizer, etc.) remain the same as LoRA.

<details>
<summary>日本語</summary>

LoHaまたはLoKrを使用するには、学習コマンドの `--network_module` 引数を変更します。その他の学習オプション（データセット設定、オプティマイザなど）はLoRAと同じです。

</details>

### LoHa (SDXL)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 sdxl_train_network.py \
    --pretrained_model_name_or_path path/to/sdxl.safetensors \
    --dataset_config path/to/toml \
    --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --network_module networks.loha --network_dim 32 --network_alpha 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 \
    --output_dir path/to/output --output_name my-loha
```

### LoKr (SDXL)

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 sdxl_train_network.py \
    --pretrained_model_name_or_path path/to/sdxl.safetensors \
    --dataset_config path/to/toml \
    --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --network_module networks.lokr --network_dim 32 --network_alpha 16 \
    --max_train_epochs 16 --save_every_n_epochs 1 \
    --output_dir path/to/output --output_name my-lokr
```

For Anima, replace `sdxl_train_network.py` with `anima_train_network.py` and use the appropriate model path and options.

<details>
<summary>日本語</summary>

Animaの場合は、`sdxl_train_network.py` を `anima_train_network.py` に置き換え、適切なモデルパスとオプションを使用してください。

</details>

### Common training options / 共通の学習オプション

The following `--network_args` options are available for both LoHa and LoKr, same as LoRA:

| Option | Description |
|---|---|
| `verbose=True` | Display detailed information about the network modules |
| `rank_dropout=0.1` | Apply dropout to the rank dimension during training |
| `module_dropout=0.1` | Randomly skip entire modules during training |
| `exclude_patterns=[r'...']` | Exclude modules matching the regex patterns (in addition to architecture defaults) |
| `include_patterns=[r'...']` | Override excludes: modules matching these regex patterns will be included even if they match `exclude_patterns` |
| `network_reg_lrs=regex1=lr1,regex2=lr2` | Set per-module learning rates using regex patterns |
| `network_reg_dims=regex1=dim1,regex2=dim2` | Set per-module dimensions (rank) using regex patterns |

<details>
<summary>日本語</summary>

以下の `--network_args` オプションは、LoRAと同様にLoHaとLoKrの両方で使用できます:

| オプション | 説明 |
|---|---|
| `verbose=True` | ネットワークモジュールの詳細情報を表示 |
| `rank_dropout=0.1` | 学習時にランク次元にドロップアウトを適用 |
| `module_dropout=0.1` | 学習時にモジュール全体をランダムにスキップ |
| `exclude_patterns=[r'...']` | 正規表現パターンに一致するモジュールを除外（アーキテクチャのデフォルトに追加） |
| `include_patterns=[r'...']` | 正規表現パターンに一致するモジュールのみを対象とする |
| `network_reg_lrs=regex1=lr1,regex2=lr2` | 正規表現パターンでモジュールごとの学習率を設定 |
| `network_reg_dims=regex1=dim1,regex2=dim2` | 正規表現パターンでモジュールごとの次元（ランク）を設定 |

</details>

### Conv2d support / Conv2dサポート

By default, LoHa and LoKr target Linear and Conv2d 1x1 layers. To also train Conv2d 3x3+ layers (e.g., in SDXL's ResNet blocks), use the `conv_dim` and `conv_alpha` options:

```bash
--network_args "conv_dim=16" "conv_alpha=8"
```

For Conv2d 3x3+ layers, you can enable Tucker decomposition for more efficient parameter representation:

```bash
--network_args "conv_dim=16" "conv_alpha=8" "use_tucker=True"
```

- Without `use_tucker`: The kernel dimensions are flattened into the input dimension (flat mode).
- With `use_tucker=True`: A separate Tucker tensor is used to handle the kernel dimensions, which can be more parameter-efficient.

<details>
<summary>日本語</summary>

デフォルトでは、LoHaとLoKrはLinearおよびConv2d 1x1層を対象とします。Conv2d 3x3+層（SDXLのResNetブロックなど）も学習するには、`conv_dim`と`conv_alpha`オプションを使用します:

```bash
--network_args "conv_dim=16" "conv_alpha=8"
```

Conv2d 3x3+層に対して、Tucker分解を有効にすることで、より効率的なパラメータ表現が可能です:

```bash
--network_args "conv_dim=16" "conv_alpha=8" "use_tucker=True"
```

- `use_tucker`なし: カーネル次元が入力次元に平坦化されます（flatモード）。
- `use_tucker=True`: カーネル次元を扱う別のTuckerテンソルが使用され、よりパラメータ効率が良くなる場合があります。

</details>

### LoKr-specific option: `factor` / LoKr固有のオプション: `factor`

LoKr decomposes weight dimensions using factorization. The `factor` option controls how dimensions are split:

- `factor=-1` (default): Automatically find balanced factors. For example, dimension 512 is split into (16, 32).
- `factor=N` (positive integer): Force factorization using the specified value. For example, `factor=4` splits dimension 512 into (4, 128).

```bash
--network_args "factor=4"
```

When `network_dim` (rank) is large enough relative to the factorized dimensions, LoKr uses a full matrix instead of a low-rank decomposition for the second factor. A warning will be logged in this case.

<details>
<summary>日本語</summary>

LoKrは重みの次元を因数分解して分割します。`factor` オプションでその分割方法を制御します:

- `factor=-1`（デフォルト）: バランスの良い因数を自動的に見つけます。例えば、次元512は(16, 32)に分割されます。
- `factor=N`（正の整数）: 指定した値で因数分解します。例えば、`factor=4` は次元512を(4, 128)に分割します。

```bash
--network_args "factor=4"
```

`network_dim`（ランク）が因数分解された次元に対して十分に大きい場合、LoKrは第2因子に低ランク分解ではなくフル行列を使用します。その場合、警告がログに出力されます。

</details>

### Anima-specific option: `train_llm_adapter` / Anima固有のオプション: `train_llm_adapter`

For Anima, you can additionally train the LLM adapter modules by specifying:

```bash
--network_args "train_llm_adapter=True"
```

This includes `LLMAdapterTransformerBlock` modules as training targets.

<details>
<summary>日本語</summary>

Animaでは、以下を指定することでLLMアダプターモジュールも追加で学習できます:

```bash
--network_args "train_llm_adapter=True"
```

これにより、`LLMAdapterTransformerBlock` モジュールが学習対象に含まれます。

</details>

### LoRA+ / LoRA+

LoRA+ (`loraplus_lr_ratio` etc. in `--network_args`) is supported with LoHa/LoKr. For LoHa, the second pair of matrices (`hada_w2_a`) is treated as the "plus" (higher learning rate) parameter group. For LoKr, the scale factor (`lokr_w1`) is treated as the "plus" parameter group.

```bash
--network_args "loraplus_lr_ratio=4"
```

This feature has been confirmed to work in basic testing, but feedback is welcome. If you encounter any issues, please report them.

<details>
<summary>日本語</summary>

LoRA+（`--network_args` の `loraplus_lr_ratio` 等）はLoHa/LoKrでもサポートされています。LoHaでは第2ペアの行列（`hada_w2_a`）が「plus」（より高い学習率）パラメータグループとして扱われます。LoKrではスケール係数（`lokr_w1`）が「plus」パラメータグループとして扱われます。

```bash
--network_args "loraplus_lr_ratio=4"
```

この機能は基本的なテストでは動作確認されていますが、フィードバックをお待ちしています。問題が発生した場合はご報告ください。

</details>

## How LoHa and LoKr work / LoHaとLoKrの仕組み

### LoHa

LoHa represents the weight update as a Hadamard (element-wise) product of two low-rank matrices:

```
ΔW = (W1a × W1b) ⊙ (W2a × W2b)
```

where `W1a`, `W1b`, `W2a`, `W2b` are low-rank matrices with rank `network_dim`. This means LoHa has roughly **twice the number of trainable parameters** compared to LoRA at the same rank, but can capture more complex weight structures due to the element-wise product.

For Conv2d 3x3+ layers with Tucker decomposition, each pair additionally has a Tucker tensor `T` and the reconstruction becomes: `einsum("i j ..., j r, i p -> p r ...", T, Wb, Wa)`.

### LoKr

LoKr represents the weight update using a Kronecker product:

```
ΔW = W1 ⊗ W2    (where W2 = W2a × W2b in low-rank mode)
```

The original weight dimensions are factorized (e.g., a 512×512 weight might be split so that W1 is 16×16 and W2 is 32×32). W1 is always a full matrix (small), while W2 can be either low-rank decomposed or a full matrix depending on the rank setting. LoKr tends to produce **smaller models** compared to LoRA at the same rank.

<details>
<summary>日本語</summary>

### LoHa

LoHaは重みの更新を2つの低ランク行列のHadamard積（要素ごとの積）で表現します:

```
ΔW = (W1a × W1b) ⊙ (W2a × W2b)
```

ここで `W1a`, `W1b`, `W2a`, `W2b` はランク `network_dim` の低ランク行列です。LoHaは同じランクのLoRAと比較して学習可能なパラメータ数が **約2倍** になりますが、要素ごとの積により、より複雑な重み構造を捉えることができます。

Conv2d 3x3+層でTucker分解を使用する場合、各ペアにはさらにTuckerテンソル `T` があり、再構成は `einsum("i j ..., j r, i p -> p r ...", T, Wb, Wa)` となります。

### LoKr

LoKrはKronecker積を使って重みの更新を表現します:

```
ΔW = W1 ⊗ W2    （低ランクモードでは W2 = W2a × W2b）
```

元の重みの次元が因数分解されます（例: 512×512の重みが、W1が16×16、W2が32×32に分割されます）。W1は常にフル行列（小さい）で、W2はランク設定に応じて低ランク分解またはフル行列になります。LoKrは同じランクのLoRAと比較して **より小さいモデル** を生成する傾向があります。

</details>

## Inference / 推論

Trained LoHa/LoKr weights are saved in safetensors format, just like LoRA.

<details>
<summary>日本語</summary>

学習済みのLoHa/LoKrの重みは、LoRAと同様にsafetensors形式で保存されます。

</details>

### SDXL

For SDXL, use `gen_img.py` with `--network_module` and `--network_weights`, the same way as LoRA:

```bash
python gen_img.py --ckpt path/to/sdxl.safetensors \
    --network_module networks.loha --network_weights path/to/loha.safetensors \
    --prompt "your prompt" ...
```

Replace `networks.loha` with `networks.lokr` for LoKr weights.

<details>
<summary>日本語</summary>

SDXLでは、LoRAと同様に `gen_img.py` で `--network_module` と `--network_weights` を指定します:

```bash
python gen_img.py --ckpt path/to/sdxl.safetensors \
    --network_module networks.loha --network_weights path/to/loha.safetensors \
    --prompt "your prompt" ...
```

LoKrの重みを使用する場合は `networks.loha` を `networks.lokr` に置き換えてください。

</details>

### Anima

For Anima, use `anima_minimal_inference.py` with the `--lora_weight` argument. LoRA, LoHa, and LoKr weights are automatically detected and merged:

```bash
python anima_minimal_inference.py --dit path/to/dit --prompt "your prompt" \
    --lora_weight path/to/loha_or_lokr.safetensors ...
```

<details>
<summary>日本語</summary>

Animaでは、`anima_minimal_inference.py` に `--lora_weight` 引数を指定します。LoRA、LoHa、LoKrの重みは自動的に判定されてマージされます:

```bash
python anima_minimal_inference.py --dit path/to/dit --prompt "your prompt" \
    --lora_weight path/to/loha_or_lokr.safetensors ...
```

</details>
