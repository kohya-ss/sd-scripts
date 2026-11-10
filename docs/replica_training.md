# Single-Process Multi-GPU Training (Replica Training) / 1プロセスでのマルチGPU学習（レプリカ学習）

> **Experimental.** This feature is new and has been tested only with a limited set of configurations (Anima and MOSS-Image LoRA training, two GPUs, `--compile` on and off). The behavior and the options may change, and there may be problems with configurations not tested yet. Please report issues with the command line used.
>
> **実験的機能です。** 新しい機能のため、限られた構成（Anima と MOSS-Image の LoRA 学習、2 GPU、`--compile` あり/なし）でしかテストしていません。挙動やオプションは変更される可能性があり、未検証の構成では問題が起きるかもしれません。問題があれば使用したコマンドラインとともにご報告ください。

## Overview / 概要

`--replica_devices` trains a LoRA (or LoHa / LoKr) on several GPUs from a single Python process, without DDP. One *master* runs on the accelerator device (usually `cuda:0`) and one *replica* per listed device holds its own copy of the frozen base model and of the network. The data loader batches are assigned to the devices in turn, the replica gradients are summed into the master at each optimizer step, and the replicas receive the updated weights before their next batch.

Compared with `accelerate launch --multi_gpu` (DDP):

- No process group: no `torchrun`, no NCCL / gloo. It therefore also works on Windows, where the gloo backend does not support multi-GPU training with CUDA tensors.
- One data loader, one process. The text encoder and the VAE are needed on the master only (their outputs are cached).
- The script is started with plain `python` (or `accelerate launch` with `--num_processes 1`).

Supported: Anima (`anima_train_network.py`) and MOSS-Image (`moss_image_train_network.py`), CUDA only.

`--replica_devices` は、DDP を使わずに 1 つの Python プロセスから複数の GPU で LoRA（LoHa / LoKr も）を学習します。accelerator のデバイス（通常 `cuda:0`）で動く *マスター* と、指定したデバイスごとの *レプリカ*（凍結されたベースモデルとネットワークのコピーを持つ）で構成されます。データローダーのバッチは各デバイスへ順番に割り当てられ、optimizer のステップごとにレプリカの勾配がマスターに合計され、レプリカは次のバッチの前に更新済みの重みを受け取ります。

`accelerate launch --multi_gpu`（DDP）との違い:

- プロセスグループを使いません（`torchrun`、NCCL / gloo が不要）。そのため、gloo バックエンドが CUDA テンソルでのマルチ GPU 学習に対応していない Windows でも動作します。
- データローダーとプロセスは 1 つだけです。Text Encoder と VAE はマスターにだけ必要です（出力はキャッシュされます）。
- 通常の `python` で起動します（`accelerate launch` を使う場合は `--num_processes 1`）。

対応モデル: Anima（`anima_train_network.py`）と MOSS-Image（`moss_image_train_network.py`）、CUDA のみ。

## Usage / 使い方

```bash
python moss_image_train_network.py \
  --pretrained_model_name_or_path moss_image_base_bf16.safetensors \
  --text_encoder model-00001-of-00004.safetensors --vae diffusion_pytorch_model.safetensors \
  --dataset_config dataset.toml --output_dir output --output_name my_lora \
  --network_module networks.lora_generic --network_dim 16 --network_alpha 16 \
  --mixed_precision bf16 --gradient_checkpointing --compile \
  --cache_latents --cache_latents_to_disk --cache_text_encoder_outputs --cache_text_encoder_outputs_to_disk \
  --network_train_unet_only \
  --replica_devices 1
```

`--replica_devices 1` adds a replica on `cuda:1`; `--replica_devices 1,2` adds two. The master is the accelerator device (`cuda:0` unless `CUDA_VISIBLE_DEVICES` is set).

The following options are required:

- `--cache_latents` and `--cache_text_encoder_outputs`: the VAE and the text encoder are not replicated.
- `--network_train_unet_only`: only the DiT part of the network is trained (no text encoder LoRA).
- `--mixed_precision bf16` (or `no`); fp16 is not supported.

Not supported with `--replica_devices`: DeepSpeed, `--torch_compile` (use `--compile`), `--base_weights`, `--fp8_base`, multi-process launches (`--num_processes` > 1).

`--replica_devices 1` で `cuda:1` にレプリカを 1 つ、`--replica_devices 1,2` で 2 つ追加します。マスターは accelerator のデバイス（`CUDA_VISIBLE_DEVICES` を指定しなければ `cuda:0`）です。

以下のオプションが必須です。

- `--cache_latents` と `--cache_text_encoder_outputs`: VAE と Text Encoder はレプリカに複製されません。
- `--network_train_unet_only`: ネットワークの DiT 部分のみを学習します（Text Encoder の LoRA は不可）。
- `--mixed_precision bf16`（または `no`）。fp16 は非対応です。

`--replica_devices` と併用できないもの: DeepSpeed、`--torch_compile`（代わりに `--compile` を使用）、`--base_weights`、`--fp8_base`、マルチプロセス起動（`--num_processes` > 1）。

## Batch size and steps / バッチサイズとステップ数

`train_batch_size` is the batch size *per device*, as with DDP. With N devices (master + replicas), one optimizer step consumes `train_batch_size × N × gradient_accumulation_steps` images, and the number of steps per epoch is divided by N. Internally `gradient_accumulation_steps` is multiplied by N and each micro-step is sent to one device, so the step count, the learning rate schedule and the progress bar are the same as with DDP with the same arguments. The metadata of the saved LoRA records the original `gradient_accumulation_steps` and the replica devices (`ss_replica_devices`).

`train_batch_size` は DDP と同様に *デバイスごとの* バッチサイズです。N 台（マスター＋レプリカ）では、1 回の optimizer ステップで `train_batch_size × N × gradient_accumulation_steps` 枚を消費し、epoch あたりのステップ数は N 分の 1 になります。内部では `gradient_accumulation_steps` を N 倍し、micro-step を 1 つずつデバイスに送ることで、ステップ数、学習率スケジュール、進捗表示が同じ引数の DDP と一致するようにしています。保存される LoRA のメタデータには元の `gradient_accumulation_steps` とレプリカのデバイス（`ss_replica_devices`）が記録されます。

## Options / オプション

| Option | Description |
|---|---|
| `--replica_devices` | Comma-separated CUDA devices of the replicas, e.g. `1` or `cuda:1,cuda:2`. Enables replica training. |
| `--replica_warmup_steps` | Number of optimizer steps at the start during which the replica micro-steps run serially (default: 2 with `--compile`, 0 otherwise). See the notes on `torch.compile` below. |

| オプション | 説明 |
|---|---|
| `--replica_devices` | レプリカを置く CUDA デバイスをカンマ区切りで指定します（例: `1`、`cuda:1,cuda:2`）。指定するとレプリカ学習が有効になります。 |
| `--replica_warmup_steps` | 学習開始時にレプリカの micro-step を直列に実行する optimizer ステップ数（既定: `--compile` 時は 2、それ以外は 0）。下記の `torch.compile` の補足を参照してください。 |

## Notes / 補足

- **Speed.** The replica micro-steps run in a worker thread per device while the master micro-step runs in the main thread. The speed-up approaches the number of devices when the GPU time of a micro-step is well above its CPU launch overhead; `--compile` reduces that overhead considerably and is recommended. At low resolutions or with small batches without `--compile`, the gain is smaller.
- **Memory.** Each replica holds a full copy of the base model (and its activations). `--blocks_to_swap` works per replica; the CPU copies of the swapped blocks are kept once per replica.
- **`torch.compile`.** Compilation is not thread-safe, so a micro-step that may compile runs alone: all micro-steps of the first `--replica_warmup_steps` optimizer steps, and later every micro-step whose input shapes (bucket resolution, text length, batch size) are new to its device. Following Dynamo's automatic dynamic shapes, a dimension is considered settled once two different values have been seen on the device, so the serialized micro-steps become rare after the first bucket changes (with `--compile_dynamic false` every new value is serialized, with `true` only the first shape). If the training stops with a message about a compilation overlapping another device, increase `--replica_warmup_steps` and please report the case.
- **Logged loss.** The loss shown for a replica micro-step is the loss of the *previous* micro-step of that replica (the current one is still running when the value is needed). The loss log therefore lags by one step for the replica steps; the gradients are not affected.
- **Sample images, validation and saving** run on the master.
- **Reproducibility.** The results differ from single-GPU training with the same seed (the batches are distributed differently), in the same way as with DDP.

- **速度。** レプリカの micro-step はデバイスごとのワーカースレッドで、マスターの micro-step はメインスレッドで実行されます。micro-step の GPU 時間が CPU 側のカーネル発行コストより十分大きいときに、台数に近い高速化が得られます。`--compile` はこのコストを大きく減らすので推奨です。低解像度や小さいバッチで `--compile` を使わない場合、伸びは小さくなります。
- **メモリ。** 各レプリカはベースモデル全体（と活性化）を保持します。`--blocks_to_swap` はレプリカごとに動作し、スワップされるブロックの CPU 側コピーもレプリカごとに持ちます。
- **`torch.compile`。** コンパイルはスレッドセーフではないため、コンパイルが起きうる micro-step は単独で実行します。最初の `--replica_warmup_steps` 回の optimizer ステップの全 micro-step と、その後はそのデバイスで初めて見る入力形状（バケット解像度、テキスト長、バッチサイズ）の micro-step が対象です。Dynamo の automatic dynamic shapes に合わせて、ある次元で異なる値を 2 回見たらその次元は確定したものとみなすため、最初のバケット切り替えを過ぎると直列化される micro-step はまれになります（`--compile_dynamic false` では新しい値のたびに、`true` では最初の形状だけ直列化されます）。コンパイルが他デバイスの実行と重なった旨のメッセージで学習が停止した場合は `--replica_warmup_steps` を増やし、状況をご報告ください。
- **ログの loss。** レプリカの micro-step に表示される loss は、そのレプリカの *1 つ前の* micro-step の値です（値が必要な時点では現在の micro-step がまだ実行中のため）。そのためレプリカのステップでは loss のログが 1 ステップ遅れますが、勾配には影響しません。
- **サンプル画像、検証、保存** はマスターで行われます。
- **再現性。** 同じ seed でも単一 GPU の学習とは結果が異なります（バッチの配分が変わるため）。DDP と同様です。
