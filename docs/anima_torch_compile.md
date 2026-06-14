# `torch.compile` for Anima Training / Anima 学習における `torch.compile`

This document explains how to speed up Anima LoRA training (`anima_train_network.py`) with PyTorch's `torch.compile`. `torch.compile` is a just-in-time (JIT) compilation feature that can reduce per-step time by optimizing model execution.

This feature is ported from [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) and currently applies to **Anima only** in `sd-scripts`. It uses **per-block compilation**: each Transformer block of the DiT is compiled individually rather than compiling the whole model at once. This keeps the compilation cache small (one artifact reused across the structurally identical blocks), avoids excessive recompilation, and coexists with block swapping.

Note: `torch.compile` may not work well in every environment. If it fails, simply train without `--compile` (the traditional path is unchanged).

<details>
<summary>日本語</summary>

このドキュメントでは、PyTorch の `torch.compile` を用いて Anima の LoRA 学習（`anima_train_network.py`）を高速化する方法を説明します。`torch.compile` は、モデルの実行を最適化することで 1 ステップあたりの時間を短縮できるジャストインタイム (JIT) コンパイル機能です。

本機能は [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner) から移植したもので、現状 `sd-scripts` では **Anima のみ** に対応しています。**ブロック単位コンパイル**を採用しており、DiT 全体を一度にコンパイルするのではなく、DiT の各 Transformer ブロックを個別にコンパイルします。これによりコンパイルキャッシュが小さく保たれ（構造が同一なブロック間で 1 つの成果物を再利用）、過剰な再コンパイルを避けつつ、ブロックスワップとも共存できます。

※ `torch.compile` はすべての環境でうまく動作するとは限りません。失敗する場合は `--compile` を付けずに学習してください（従来の経路はそのまま使えます）。

</details>

## 1. Prerequisites / 前提条件

- **Triton** is required for `torch.compile` to be effective. On Windows, install it from the [triton-windows](https://github.com/woct0rdho/triton-windows) project.
- A recent PyTorch (2.x) is required.
- **MSVC compiler** (Visual Studio 2022 with C++ development tools, or Build Tools 2022) is required on Windows **only when using `--compile_dynamic true`**. See [Section 6](#6-windows-requirements-for---compile_dynamic--windows-での---compile_dynamic-の要件).

<details>
<summary>日本語</summary>

- `torch.compile` を効果的に動作させるには **Triton** が必要です。Windows では [triton-windows](https://github.com/woct0rdho/triton-windows) からインストールしてください。
- 比較的新しい PyTorch (2.x) が必要です。
- Windows では、**`--compile_dynamic true` を使用する場合のみ** **MSVC コンパイラ**（C++ 開発ツール入りの Visual Studio 2022、または Build Tools 2022）が必要です。[セクション 6](#6-windows-requirements-for---compile_dynamic--windows-での---compile_dynamic-の要件) を参照してください。

</details>

## 2. Command Line Arguments / コマンドライン引数

These arguments are added by `anima_train_network.py`. They are independent of the legacy `--torch_compile` option (which routes through accelerate's dynamo and is not the per-block path described here). **`--compile` and `--torch_compile` cannot be used together.**

### Basic arguments / 基本的な引数

- `--compile`: Enable per-block `torch.compile` for the DiT (requires Triton).
- `--compile_backend`: Backend to use (default: `inductor`).
- `--compile_mode`: Compilation mode (default: `default`, recommended for training).
  - Choices: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`
- `--compile_dynamic`: Dynamic shapes mode (default: `None`, equivalent to `auto`). On Windows, `true` requires the Visual Studio 2022 C++ compiler.
  - Choices: `true`, `false`, `auto`
- `--compile_fullgraph`: Enable fullgraph mode. Cannot be used with `--split_attn`.
- `--compile_cache_size_limit`: Set `torch._dynamo.config.cache_size_limit` (default: PyTorch default, typically 8-32; recommended: `32`).

### Additional performance arguments / 追加のパフォーマンス引数

These are independent of `--compile` and can be used on their own.

- `--cuda_allow_tf32`: Allow TF32 precision on Ampere or newer GPUs (improves performance).
- `--cuda_cudnn_benchmark`: Enable cuDNN benchmark mode (may improve performance).

<details>
<summary>日本語</summary>

これらの引数は `anima_train_network.py` で追加されます。既存の `--torch_compile`（accelerate の dynamo 経由で、本ドキュメントのブロック単位の経路とは別物）とは独立しています。**`--compile` と `--torch_compile` は併用できません。**

### 基本的な引数

- `--compile`: DiT のブロック単位 `torch.compile` を有効にする（Triton が必要）。
- `--compile_backend`: 使用するバックエンド（デフォルト: `inductor`）。
- `--compile_mode`: コンパイルモード（デフォルト: `default`、学習に推奨）。
  - 選択肢: `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`
- `--compile_dynamic`: 動的形状モード（デフォルト: `None`、`auto` と同じ）。Windows で `true` を使うには Visual Studio 2022 の C++ コンパイラが必要です。
  - 選択肢: `true`, `false`, `auto`
- `--compile_fullgraph`: フルグラフモードを有効にする。`--split_attn` とは併用できません。
- `--compile_cache_size_limit`: `torch._dynamo.config.cache_size_limit` を設定（デフォルト: PyTorch のデフォルト、通常 8-32、推奨: `32`）。

### 追加のパフォーマンス引数

これらは `--compile` とは独立しており、単独でも使用できます。

- `--cuda_allow_tf32`: Ampere 以降の GPU で TF32 精度を許可する（パフォーマンス向上）。
- `--cuda_cudnn_benchmark`: cuDNN のベンチマークモードを有効にする（パフォーマンスが向上する可能性がある）。

</details>

## 3. Usage Example / 使用例

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \
  anima_train_network.py \
  --pretrained_model_name_or_path path/to/anima_dit.safetensors \
  --qwen3 path/to/qwen3 \
  --vae path/to/vae \
  --dataset_config path/to/config.toml \
  --output_dir path/to/output \
  --output_name anima_lora \
  (... other training args ...) \
  --compile \
  --compile_mode default \
  --compile_cache_size_limit 32 \
  --cuda_allow_tf32 \
  --cuda_cudnn_benchmark
```

On Windows Command Prompt, use `^` instead of `\` for line continuation.

<details>
<summary>日本語</summary>

Windows のコマンドプロンプトでは、行末の継続文字は `\` ではなく `^` を使用してください。

</details>

## 4. Notes on Behavior / 挙動に関する注意

- **The first iterations are slow.** `torch.compile` performs JIT compilation on the first forward (and backward) pass with a given input shape. This is expected; subsequent steps are much faster.
- **Recompilation on shape changes.** If `--compile_dynamic` is not set to `true`, recompilation occurs whenever an input shape changes. With multi-resolution (aspect-ratio bucketing) datasets, each new bucket resolution triggers a recompilation, so the **first epoch can be noticeably slower** while later epochs benefit. Increasing `--compile_cache_size_limit` (e.g. to `32`) helps avoid cache eviction across many bucket shapes.
- **`max-autotune`** may not work in some cases. If you hit errors, fall back to `--compile_mode default`.
- **`--compile_fullgraph`** may not work depending on the configuration; in particular it cannot be combined with `--split_attn` (see Section 5).

<details>
<summary>日本語</summary>

- **最初のイテレーションは遅くなります。** `torch.compile` は、ある入力形状に対する最初の forward（および backward）で JIT コンパイルを行います。これは想定どおりの挙動で、以降のステップははるかに高速になります。
- **形状変化時の再コンパイル。** `--compile_dynamic` を `true` にしない場合、入力形状が変わるたびに再コンパイルが発生します。マルチ解像度（アスペクト比バケッティング）のデータセットでは、新しいバケット解像度ごとに再コンパイルが走るため、**最初のエポックは目に見えて遅くなる**ことがありますが、以降のエポックでは高速化の恩恵を受けられます。`--compile_cache_size_limit` を大きく（例: `32`）すると、多数のバケット形状にまたがるキャッシュの追い出しを避けやすくなります。
- **`max-autotune`** は場合によって動作しないことがあります。エラーが出る場合は `--compile_mode default` に戻してください。
- **`--compile_fullgraph`** は構成によっては動作しないことがあります。特に `--split_attn` とは併用できません（セクション 5 参照）。

</details>

## 5. Limitations and Known Issues / 制限事項と既知の問題

- **`--compile_fullgraph` and `--split_attn` cannot be used together.** `--split_attn` uses dynamic control flow that is incompatible with fullgraph mode. This combination is rejected at startup.
- **`--compile` and `--torch_compile` cannot be used together.** The new per-block path and the legacy accelerate-dynamo path are mutually exclusive and rejected at startup.
- **Block swapping (`--blocks_to_swap`).** When block swapping is enabled, the Linear layers inside swapped blocks are automatically excluded from compilation (their weights move between CPU and GPU each step, which conflicts with a compiled graph). The blocks are still compiled, but the speed-up may be smaller than without block swapping.

<details>
<summary>日本語</summary>

- **`--compile_fullgraph` と `--split_attn` は併用できません。** `--split_attn` はフルグラフモードと互換性のない動的な制御フローを使用します。この組み合わせは起動時にエラーになります。
- **`--compile` と `--torch_compile` は併用できません。** 新しいブロック単位の経路と、既存の accelerate-dynamo の経路は排他で、起動時にエラーになります。
- **ブロックスワップ（`--blocks_to_swap`）。** ブロックスワップを有効にすると、スワップ対象ブロック内の Linear レイヤーは自動的にコンパイル対象から除外されます（重みが毎ステップ CPU と GPU 間を移動するため、コンパイル済みグラフと衝突します）。ブロック自体はコンパイルされますが、ブロックスワップなしの場合より高速化の効果は小さくなる可能性があります。

</details>

## 6. Windows Requirements for `--compile_dynamic` / Windows での `--compile_dynamic` の要件

**IMPORTANT**: On Windows, using `--compile_dynamic true` requires:

1. **Visual Studio 2022** with C++ development tools installed.
2. Either:
   - Running the training script from the **"x64 Native Tools Command Prompt for VS 2022"**, or
   - Running it after executing `vcvars64.bat` to set the environment variables, e.g. `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`.

If you hit compilation errors with `--compile_dynamic true`, make sure you are running from the correct command prompt, or try without `--compile_dynamic`. See also the [PyTorch Inductor Windows documentation](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html#install-a-compiler).

<details>
<summary>日本語</summary>

**重要**: Windows で `--compile_dynamic true` を使用する場合、以下が必要です。

1. C++ 開発ツール入りの **Visual Studio 2022** のインストール。
2. 以下のいずれか:
   - **"x64 Native Tools Command Prompt for VS 2022"** からスクリプトを実行する、または
   - `vcvars64.bat` を実行して環境変数を設定してからスクリプトを実行する。例: `"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"`

`--compile_dynamic true` でコンパイルエラーが発生する場合は、正しいコマンドプロンプトから実行しているか確認するか、`--compile_dynamic` なしで試してください。[PyTorch Inductor Windows ドキュメント](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html#install-a-compiler) も参照してください。

</details>

## 7. Recommended Settings / 推奨設定

For training, the following is a good starting point:

```bash
--compile \
--compile_mode default \
--compile_cache_size_limit 32 \
--cuda_allow_tf32 \
--cuda_cudnn_benchmark
```

The performance gain depends on the GPU, settings, and dataset. As one example, with batch size 4 and gradient checkpointing enabled, the second-epoch training time improved from about 1:13 to about 1:03, and a 40-step window (steps 60-100) improved from about 24s to about 17s. Block swapping reduces the gain (see Section 5).

<details>
<summary>日本語</summary>

学習では、以下を出発点にするとよいでしょう。

```bash
--compile \
--compile_mode default \
--compile_cache_size_limit 32 \
--cuda_allow_tf32 \
--cuda_cudnn_benchmark
```

高速化の効果は GPU・設定・データセットによって異なります。一例として、バッチサイズ 4・gradient checkpointing 有効の条件で、2 エポック目の学習時間が約 1:13 から約 1:03 に、40 ステップ（step 60〜100）が約 24 秒から約 17 秒に短縮しました。ブロックスワップを使うと効果は小さくなります（セクション 5 参照）。

</details>

## 8. Troubleshooting / トラブルシューティング

- **Out of memory.** Try a smaller `--compile_cache_size_limit`, reduce batch size, or use `--compile_mode default` instead of `max-autotune`.
- **Compilation errors.** Ensure Triton is installed correctly. On Windows with `--compile_dynamic true`, verify the MSVC setup (Section 6). If problems persist, train without `--compile`.

<details>
<summary>日本語</summary>

- **メモリ不足。** `--compile_cache_size_limit` を小さくする、バッチサイズを減らす、`max-autotune` の代わりに `--compile_mode default` を使う、などを試してください。
- **コンパイルエラー。** Triton が正しくインストールされているか確認してください。Windows で `--compile_dynamic true` を使う場合は MSVC のセットアップ（セクション 6）を確認してください。解決しない場合は `--compile` を付けずに学習してください。

</details>

## Additional Resources / 追加リソース

- [PyTorch `torch.compile` tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)
- [PyTorch Inductor Windows documentation](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
- [triton-windows](https://github.com/woct0rdho/triton-windows)
- [Anima LoRA Training Guide](anima_train_network.md)
