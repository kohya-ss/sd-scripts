# Block Swap (CPU Offloading) / ブロックスワップ（CPUオフロード）

## Overview / 概要

Block swap reduces VRAM usage by keeping only some of the model's transformer blocks on the GPU at a time and offloading the rest to CPU (main RAM), streaming each block to the GPU just before it is needed. This lets you train models that would otherwise not fit in VRAM, at the cost of some speed and additional main RAM.

Enable it with `--blocks_to_swap N`, which offloads `N` blocks to CPU. A larger `N` saves more VRAM but uses more main RAM and may slow things down. The maximum value of `N` and the recommended VRAM/RAM amounts are architecture-specific; see each architecture's documentation (e.g. [FLUX.1](./flux_train_network.md), [Anima](./anima_train_network.md), [MOSS-Image](./moss_image_train_network.md)).

Block swap is available for the DiT-based models: SD3, FLUX.1 (and Chroma), Lumina Image 2.0, HunyuanImage-2.1, Anima and MOSS-Image. It requires `--gradient_checkpointing` for practical use, and cannot be combined with `--cpu_offload_checkpointing` / `--unsloth_offload_checkpointing`.

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

<details>
<summary>日本語</summary>

ブロックスワップは、モデルのtransformerブロックのうち一部のみをGPU上に置き、残りをCPU（メインRAM）にオフロードし、各ブロックを必要になる直前にGPUへ転送することで、VRAM使用量を削減します。これにより、本来VRAMに収まらないモデルでも学習が可能になりますが、速度の低下と追加のメインRAMが必要になります。

`--blocks_to_swap N`で有効化し、`N`個のブロックをCPUにオフロードします。`N`を大きくするほどVRAMを節約できますが、メインRAMの使用量が増え、速度が低下する場合があります。`N`の最大値や推奨VRAM/RAM量はアーキテクチャごとに異なるため、各アーキテクチャのドキュメント（[FLUX.1](./flux_train_network.md)、[Anima](./anima_train_network.md)、[MOSS-Image](./moss_image_train_network.md)など）を参照してください。

ブロックスワップはDiTベースのモデル（SD3、FLUX.1（およびChroma）、Lumina Image 2.0、HunyuanImage-2.1、Anima、MOSS-Image）で利用できます。実用上は`--gradient_checkpointing`が必要で、`--cpu_offload_checkpointing` / `--unsloth_offload_checkpointing`とは併用できません。

（block swapのアイデアは2kpr氏の実装に基づくものです。2kpr氏にあらためて感謝します。）

</details>

## Command-Line Options / コマンドラインオプション

- `--blocks_to_swap N`: Number of blocks to offload to CPU. Default: none (disabled).
- `--block_swap_h2d_only`: Use H2D-only block swap (frozen-base / LoRA training only). See [H2D-only Block Swap](#h2d-only-block-swap--h2dのみのブロックスワップ).
- `--block_swap_ring_size N`: (used with `--block_swap_h2d_only`) Number of GPU ring buffers for streamed blocks. `2` (default) = double buffering (one block being computed on while the next is prefetched); `1` = minimal VRAM but no transfer/compute overlap.
- `--use_pinned_memory_for_block_swap`: (used with `--block_swap_h2d_only`) Keep the CPU master copies of the streamed weights in pinned (page-locked) host memory. This speeds up Host→Device copies, but increases shared GPU memory usage on Windows. Without it, the transfers are staged through a small pool of pinned buffers on a worker thread, which is only slightly slower in most environments. This option has no effect on the classic block swap.

<details>
<summary>日本語</summary>

- `--blocks_to_swap N`: CPUにオフロードするブロック数。デフォルトはなし（無効）。
- `--block_swap_h2d_only`: H2Dのみのブロックスワップを使用します（ベース凍結＝LoRA学習専用）。[H2Dのみのブロックスワップ](#h2d-only-block-swap--h2dのみのブロックスワップ)を参照してください。
- `--block_swap_ring_size N`:（`--block_swap_h2d_only`と併用）ストリーミング用GPUリングバッファの数。`2`（デフォルト）でダブルバッファ（1つのブロックを計算しつつ次をプリフェッチ）、`1`で最小VRAM（ただし転送と計算のオーバーラップなし）。
- `--use_pinned_memory_for_block_swap`:（`--block_swap_h2d_only`と併用）ストリーミングする重みのCPUマスターコピーをピン留め（ページロック）メモリに置きます。Host→Device転送が高速化されますが、Windowsでは共有GPUメモリ使用量が増加します。指定しない場合は小さなピン留めバッファのプールを介してワーカースレッドで転送し、多くの環境ではわずかに遅くなる程度です。従来のブロックスワップには影響しません。

</details>

## H2D-only Block Swap / H2Dのみのブロックスワップ

H2D-only block swap (`--block_swap_h2d_only`) is an optimized mode for **frozen-base training** (LoRA / LoHa / LoKr, where the base model weights are not updated). It is ported from [Musubi Tuner](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/block_swap.md).

**※Currently this implementation is only available for CUDA devices.**

The classic block swap *exchanges* a block: as a block is consumed it is copied back to the CPU (Device→Host, "D2H") while the next one is copied in (Host→Device, "H2D"). When the base weights are frozen, the CPU already holds an identical copy, so the D2H half is pure overhead. H2D-only keeps a permanent master copy of every streamed weight on the CPU and only ever copies Host→Device, removing the D2H transfer entirely. The weights of a block are also packed into one contiguous buffer and transferred with a single copy. This can noticeably improve training throughput, and the benefit is largest with `--fp8_base` / `--fp8_scaled` / `--convrot_int8` (smaller weights mean less data to transfer).

With less transfer time, many models can achieve complete overlap of compute and transfer. In Musubi Tuner, LoRA training of Qwen-Image on an RTX 3090 with 40 swapped blocks runs at about 14 seconds/sample with the classic block swap and about 11 seconds/sample with H2D-only, a roughly 25% speedup.

In sd-scripts, LoRA training of MOSS-Image (bf16, 1024x1024, batch size 1, `--blocks_to_swap 36`) on an RTX 3090 (PCIe Gen3 x4) runs at about 11.7 seconds/step with the classic block swap and about 9.4 seconds/step with H2D-only (`--use_pinned_memory_for_block_swap`), a roughly 20% speedup, with about 7GB of VRAM. The transfer (about 2.3 seconds per pass) is completely hidden behind the computation, so the step time no longer depends on the number of swapped blocks.

Note that on Max-Q devices and similar, power limits may cause H2D-only to run slower than no block swap at all, even with compute-transfer overlap.

**Supported scripts:** `flux_train_network.py` (FLUX.1 and Chroma), `lumina_train_network.py`, `hunyuan_image_train_network.py`, `anima_train_network.py` and `moss_image_train_network.py`. **SD3 is not supported** (`sd3_train_network.py` stops with an error), because the joint blocks of SD3 / SD3.5 are not uniform and cannot share the ring buffer. The full fine-tuning scripts (`flux_train.py`, `sd3_train.py`, `anima_train.py`, `flux_train_control_net.py`) also reject the option. The inference scripts are not affected.

**Requirements:**

- **Frozen base weights only** (LoRA / LoHa / LoKr). It cannot be used for full fine-tuning; the trainer stops with an error if a trainable base weight is detected.
- **`--gradient_checkpointing` is required for training.** Without it, training fails during the backward pass with an autograd error such as `one of the variables needed for gradient computation has been modified by an inplace operation ... version N; expected version M`. H2D-only streams weights into a small, reused GPU ring buffer; gradient checkpointing re-reads each weight at recompute time, which keeps autograd's version tracking consistent. The trainer checks this up front and reports an actionable message, so a missing `--gradient_checkpointing` is caught immediately rather than as the opaque error above.

**`--block_swap_ring_size`** controls how many GPU buffers stream the swapped blocks. `2` (default) overlaps the transfer of the next block with computation on the current one. `1` uses the least VRAM but removes the overlap; combined with `--fp8_base` / `--fp8_scaled` this is a good minimal-memory configuration.

**VRAM:** With the same `--blocks_to_swap`, H2D-only uses slightly more VRAM than the classic block swap: the classic implementation reuses the GPU memory of the swapped-out blocks, whereas H2D-only keeps `N - blocks_to_swap` blocks resident plus `ring_size` additional block buffers. Increase `--blocks_to_swap` by `ring_size` (or use `--block_swap_ring_size 1`) for the same VRAM usage.

**Debugging:** Setting the environment variable `MUSUBI_TUNER_OFFLOADER_DEBUG=1` prints the H2D transfer time and the GPU stall time per step (every `MUSUBI_TUNER_OFFLOADER_DEBUG_INTERVAL` steps, default 10). A stall of zero means the transfer is fully hidden behind the computation.

<details>
<summary>日本語</summary>

H2Dのみのブロックスワップ（`--block_swap_h2d_only`）は、**ベース凍結の学習**（LoRA / LoHa / LoKr。ベースモデルの重みを更新しない学習）向けに最適化したモードです。[Musubi Tuner](https://github.com/kohya-ss/musubi-tuner/blob/main/docs/block_swap.md)から移植しました。

**※現在の実装はCUDAデバイスでのみ利用可能です。**

従来のブロックスワップはブロックを*入れ替え*ます。あるブロックを使い終わるとCPUへ書き戻し（Device→Host、「D2H」）、同時に次のブロックを転送（Host→Device、「H2D」）します。ベースの重みが凍結されている場合、CPU側には既に同一のコピーがあるため、D2Hは完全に無駄です。H2D-onlyは、ストリーミングする全重みの恒久的なマスターコピーをCPUに保持し、常にHost→Deviceのみコピーすることで、D2H転送を完全に排除します。また、ブロックの重みをひとつの連続したバッファにまとめ、1回のコピーで転送します。これにより学習スループットが向上することがあり、その効果は`--fp8_base` / `--fp8_scaled` / `--convrot_int8`使用時に最も大きくなります（重みが小さくなり転送データ量が減るため）。

転送時間が減るため多くのモデルで計算と転送は完全にオーバーラップ可能となります。Musubi Tunerでは、RTX 3090におけるQwen-ImageのLoRA学習で、40ブロックをスワップする構成で、通常のブロックスワップは約14秒／サンプルですが、H2D-onlyは約11秒／サンプルと約25%の速度向上が見られます。

sd-scriptsでは、RTX 3090（PCIe Gen3 x4）におけるMOSS-ImageのLoRA学習（bf16、1024x1024、バッチサイズ1、`--blocks_to_swap 36`）で、通常のブロックスワップは約11.7秒／ステップですが、H2D-only（`--use_pinned_memory_for_block_swap`指定）は約9.4秒／ステップと約20%の速度向上が見られ、VRAM使用量は約7GBでした。転送（1パスあたり約2.3秒）は計算に完全に隠れるため、ステップ時間はスワップするブロック数に依存しなくなります。

なお、計算と転送が完全にオーバーラップする場合でも、Max-Qデバイスなどでは電力制限により、ブロックスワップ未使用時よりも速度が低下することがあります。

**対応スクリプト:** `flux_train_network.py`（FLUX.1およびChroma）、`lumina_train_network.py`、`hunyuan_image_train_network.py`、`anima_train_network.py`、`moss_image_train_network.py`。**SD3は非対応**です（`sd3_train_network.py`はエラーで停止します）。SD3 / SD3.5のjoint blockは構造が均一でなく、リングバッファを共有できないためです。フルファインチューニング用スクリプト（`flux_train.py`、`sd3_train.py`、`anima_train.py`、`flux_train_control_net.py`）でもこのオプションは指定できません。推論スクリプトには影響しません。

**要件:**

- **ベース凍結の重みのみ**（LoRA / LoHa / LoKr）。フルファインチューニングでは使用できません。学習可能なベース重みが検出されると、エラーで停止します。
- **学習時は`--gradient_checkpointing`が必須です。** 指定しないと、バックワードパスで`one of the variables needed for gradient computation has been modified by an inplace operation ... version N; expected version M`のようなautogradエラーで失敗します。H2D-onlyは重みを小さな再利用GPUリングバッファへストリーミングしますが、gradient checkpointingは再計算時に各重みを読み直すため、autogradのversion管理が整合します。学習側で事前にチェックし分かりやすいメッセージを表示するので、`--gradient_checkpointing`の付け忘れは上記の難解なエラーではなく即座に検出されます。

**`--block_swap_ring_size`** は、スワップするブロックをストリーミングするGPUバッファの数を制御します。`2`（デフォルト）は次ブロックの転送を現ブロックの計算とオーバーラップさせます。`1`はVRAM使用量が最小ですがオーバーラップがなくなります。`--fp8_base` / `--fp8_scaled`と組み合わせると最小メモリ構成として有効です。

**VRAM:** 同じ`--blocks_to_swap`では、H2D-onlyは従来のブロックスワップよりわずかに多くVRAMを使用します。従来の実装はスワップアウトしたブロックのGPUメモリを再利用しますが、H2D-onlyは`N - blocks_to_swap`個のブロックを常駐させたうえで`ring_size`個分のバッファを追加で確保するためです。同じVRAM使用量にするには`--blocks_to_swap`を`ring_size`だけ増やすか、`--block_swap_ring_size 1`を指定してください。

**デバッグ:** 環境変数`MUSUBI_TUNER_OFFLOADER_DEBUG=1`を設定すると、ステップあたりのH2D転送時間とGPUの待ち時間（stall）が表示されます（`MUSUBI_TUNER_OFFLOADER_DEBUG_INTERVAL`ステップごと、デフォルト10）。stallが0であれば転送は計算に完全に隠れています。

</details>

## Notes / 補足

- **torch.compile:** Block swap is compatible with `--compile` (Anima, MOSS-Image), but when block swap is active the speed benefit of compilation is limited, because the swapped Linear layers are excluded from compilation.
- **Main RAM:** Block swap holds the offloaded blocks in main RAM, and `--use_pinned_memory_for_block_swap` makes that memory page-locked (H2D-only). Larger `--blocks_to_swap` increases main RAM usage; see each architecture's documentation for recommended amounts.
- **For developers:** The offloader implementation is in `library/custom_offloading_utils.py` and is shared with Musubi Tuner. `BlockSwapConfig` holds the policy (device, backward support, H2D-only, ring size, pinned memory), and `create_offloader` selects `ModelOffloader` (classic) or `LoRAStreamOffloader` (H2D-only). The models call `enable_block_swap(blocks_to_swap, config)`. The forward loop must pass the block's activation tensors as positional arguments (`block(x, ...)`, not `block(x=x, ...)`): the offloaders use full backward hooks, which PyTorch fires after the block's backward only for positional inputs that require grad.

<details>
<summary>日本語</summary>

- **torch.compile:** ブロックスワップは`--compile`（Anima、MOSS-Image）と併用できますが、ブロックスワップが有効なときはコンパイルによる高速化の効果は限定的です（スワップ対象のLinear層がコンパイルから除外されるため）。
- **メインRAM:** ブロックスワップはオフロードしたブロックをメインRAMに保持し、`--use_pinned_memory_for_block_swap`を指定するとそのメモリがページロックされます（H2D-only時）。`--blocks_to_swap`を大きくするとメインRAM使用量が増えます。推奨量は各アーキテクチャのドキュメントを参照してください。
- **開発者向け:** オフローダの実装は`library/custom_offloading_utils.py`にあり、Musubi Tunerと共通です。`BlockSwapConfig`がポリシー（デバイス、backward対応、H2D-only、リングサイズ、ピン留めメモリ）を保持し、`create_offloader`が`ModelOffloader`（従来）または`LoRAStreamOffloader`（H2D-only）を選択します。各モデルは`enable_block_swap(blocks_to_swap, config)`を呼び出します。forwardループではブロックのアクティベーションを位置引数で渡す必要があります（`block(x, ...)`であり、`block(x=x, ...)`は不可）。オフローダはfull backward hookを使用しており、PyTorchは位置引数の入力がgradを必要とする場合にのみブロックのbackward後にhookを呼び出すためです。

</details>
