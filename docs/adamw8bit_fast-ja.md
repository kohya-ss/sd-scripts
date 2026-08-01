# AdamW8bitFast: LoRA向け同期集約版

## 概要

`AdamW8bitFast`は、bitsandbytesの`AdamW8bit`と同じparameter更新・optimizer stateを使いながら、CUDA同期回数だけを削減する実験用optimizerです。

bitsandbytesのforkや、venv内の`site-packages`の書き換えは不要です。このリポジトリ内の [library/adamw8bit_fast.py](../library/adamw8bit_fast.py) が`bnb.optim.AdamW8bit`を継承します。

使用方法:

```text
--optimizer_type AdamW8bitFast
```

`fused=True`はPyTorchの`torch.optim.AdamW`用であり、AdamW8bitFastには指定しません。学習率、`betas`、`eps`、`weight_decay`、`min_8bit_size`など、AdamW8bitが受け取る`--optimizer_args`はそのまま渡されます。

## 何を変えるか

通常のbitsandbytes optimizerは、概ね次の順序で処理します。

```text
parameter 1を更新 → GPU全体を同期
parameter 2を更新 → GPU全体を同期
...
parameter Nを更新 → GPU全体を同期
```

AdamW8bitFastは次のようにします。

```text
parameter 1を更新
parameter 2を更新
...
parameter Nを更新
GPU全体を1回だけ同期
```

同一CUDA streamへ投入したkernelは投入順に実行されます。最後の同期は残すため、非同期CUDAエラーも`step()`が戻る前に検出されます。

変更しないもの:

- AdamW8bitの更新式
- 8bit / FP32 optimizer stateの選択
- `min_8bit_size`の判定
- parameterとparameter groupの更新順序
- 学習率、weight decay、betas、eps
- state dict、LR scheduler、zero_gradの形式

## 高速経路の条件

次をすべて満たすと同期集約経路を使います。

- non-paged AdamW8bit
- optimizer全体およびParameter別overrideの実効設定が`percentile_clipping=100`（既定値）かつ`max_unorm=0`
- そのstepで`grad is not None`の更新対象が、通常の`torch.nn.Parameter`とdense strided gradient
- 更新対象のparameterとgradientが同じ1台のCUDA device上
- 分散world sizeが1
- optimizer closureを使わない

CPU、paged optimizer、複数GPU、DTensorなどのTensor subclass、sparse gradient、複数device、closure使用時は、bitsandbytes標準の`step()`へ戻ります。

最初にgradientを更新するstepで、実際に選ばれた経路とbitsandbytesのversionを一度だけログへ出します。

```text
AdamW8bitFast: fast path enabled on cuda:0 (bitsandbytes 0.48.2)
```

標準経路へ戻る場合は、理由を含むwarningを一度だけ出します。途中で条件が変わり、高速経路から標準経路へ移った場合も同様です。

```text
AdamW8bitFast: using stock AdamW8bit step (reason: parameter is on cpu, bitsandbytes 0.48.2)
```

## RTX 5080での実測

検証環境:

```text
PyTorch       2.9.1+cu130
bitsandbytes  0.48.2
GPU           NVIDIA GeForce RTX 5080
LoRA tensors  1,620
LoRA elements 12,646,400
Parameter dtype FP32
```

代表的なLoRA checkpointのtensor形状で、warmup 3回後に20回測定した結果です。

| optimizer | step中央値 | optimizer周辺の定常割当 |
|---|---:|---:|
| AdamW8bit | 85.34ms | 125.107MiB |
| AdamW8bitFast | 32.98ms | 125.107MiB |

optimizer stepは約2.59倍、約52.35ms/step短縮しました。単純換算では8,400 stepで約440秒（約7分20秒）ですが、実学習時間はデータ処理、forward/backward、保存、ログ、量子化処理にも左右されます。

再測定コマンド:

```powershell
python tools/benchmark_adamw8bit_fast.py "path\to\lora_checkpoint.safetensors"
```

benchmarkはcheckpointのtensor名とshapeだけを読み、重み値は使用・変更しません。

通常のmixed-precision学習ではLoRA parameterとgradientがFP32のため、benchmarkも既定でFP32を使います。`--full_fp16`相当を調べる場合は`--dtype float16`を追加します。

## 検証状況と注意点

- FP32とFP16の両方について、複数learning-rate group、8bit/FP32 state混在、途中の`grad=None`を含む7 stepで、stock AdamW8bitと全parameter・全optimizer stateのbit完全一致を確認済みです。
- `block_wise=False`でもstock AdamW8bitとのbit完全一致を確認しています。`percentile_clipping < 100`と`max_unorm > 0`は安全のため標準経路へ戻します。
- stock AdamW8bitからAdamW8bitFast、および逆方向のstate dict読み込み後も、次のstepで全parameter・全optimizer stateが一致することを確認しています。
- CUDA同期が1 stepにつき1回になることをテストしています。
- 実GPUテストは`requirements.txt`と同じbitsandbytes 0.48.2で行っています。
- まだ8,400 stepの長時間学習では未検証です。
- bitsandbytesの内部メソッドを継承しているため、bitsandbytes更新時は回帰テストとbenchmarkを再実行してください。
- optimizer stepの制御フローはMIT Licenseのbitsandbytesをもとにしています。ライセンス本文は [third_party/bitsandbytes-LICENSE.txt](../third_party/bitsandbytes-LICENSE.txt) を参照してください。
- 最初の学習試験では既存のAdamW8bit runを上書きせず、別のoutput nameを使用してください。
