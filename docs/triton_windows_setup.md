# dq_delta Triton 高速化: 設定・対応範囲・検証記録

この文書は、dq_delta fake quantのoptional Triton高速化について、利用方法、コード上の対応範囲、実際に学習で検証した範囲、fallback、検証手順をまとめたものです。

## 基本方針

- Tritonはoptionalとし、`requirements.txt`には追加しない。
- `--dq_delta_use_triton`を指定した場合だけTriton pathを試す。
- Triton未導入、未対応tensor、未対応設定、kernel失敗時はPyTorchへfallbackする。
- `--dq_delta_use_triton`を外せば、Tritonがインストール済みでも従来のPyTorch pathになる。
- 学習forwardの出力を優先し、statsだけが非対応の場合もforwardは通常Triton Bを維持する。

## インストールと検証環境

長時間学習まで確認した環境:

```text
PyTorch: 2.9.1+cu130
CUDA: 13.0
Python: CPython 3.10 venv
Triton Windows: triton-windows 3.5.1.post24
GPU: NVIDIA GeForce RTX 5080
```

インストール例:

```bash
python -m pip install -U "triton-windows>=3.5,<3.6"
python -c "import triton; print(triton.__version__)"
```

公開requirementsへは追加しないため、環境ごとに互換性のあるTritonを別途導入してください。

## CLIオプション

正式な学習CLIは2つです。

| オプション | 既定 | 動作 |
| --- | --- | --- |
| `--dq_delta_use_triton` | OFF | 対応するscale計算とstochastic fake quantをTritonで処理する。 |
| `--dq_delta_triton_stats` | OFF | 対応するbasic log/auto statsをBと同じTriton kernelへ融合する。`--dq_delta_use_triton`が必要。 |

XL18相当の推奨指定:

```text
--dq_delta_use_triton --dq_delta_triton_stats
```

`--dq_delta_log_detail basic`は既定値なので省略できます。

実験に使った`--dq_delta_triton_scale_only`、`--dq_delta_triton_div_rn`、`--dq_delta_triton_stats_mode`は正式化時に削除しました。現在は次の構成に固定されています。

```text
A/B                    : 別kernel
stochastic乱数         : PyTorch torch.rand_like
Bの除算                : tl.div_rn
basic stats            : B+stats fused
partial stats集約      : torch.sum(dim=0)
large launch           : BLOCK_SIZE=1024, num_warps=2
small launch           : BLOCK_SIZE=256, num_warps=4
```

## 処理の対応関係

```text
A: channel RMS scale計算
B: stochastic fake quant
C: basic log/auto stats
```

### A: channel RMS scale

Triton Aは [library/rounding_util.py](../library/rounding_util.py) の`compute_scale_bits`の次の式に対応します。

```python
rng = torch.sqrt(torch.mean(x.to(torch.float32) ** 2, dim=reduce_dims, keepdim=True) + eps) * range_mul
scale = (rng / qmax).to(torch.float32)
```

Triton Aのコード上の条件:

```text
bits mode
granularity=channel
stat=rms
x.ndim in (2, 3, 4)
x is contiguous CUDA tensor
x.dtype in (float16, bfloat16, float32)
```

3D NLCではchannel方向をまとめて読む専用2D kernelを使います。それ以外の対応shapeではgeneric channel kernelを使います。

`granularity=tensor`または`stat=absmax`の場合、scale計算はPyTorchですが、条件を満たせばBだけTritonになることがあります。

### B: stochastic fake quant

Triton Bは [library/rounding_util.py](../library/rounding_util.py) の`fake_quantize_levels(..., mode="stoch")`に対応します。

```python
y = x.to(torch.float32) / scale.to(torch.float32)
q_floor = torch.floor(y)
probs = (y - q_floor).clamp(0.0, 1.0)
q = q_floor + (torch.rand_like(probs) < probs).to(y.dtype)
q = torch.clamp(q, qmin, qmax)
q_out = (q * scale).to(x.dtype)
out = x + (q_out - x).detach()
```

Triton Bのコード上の条件:

```text
bits mode
mode=stoch
x.ndim in (2, 3, 4)
x and scale are contiguous CUDA tensors
x.dtype in (float16, bfloat16, float32)
scale is scalar or broadcastable per-channel scale
```

乱数はTritonの`tl.rand`ではなくPyTorchで1回生成し、通常Bまたはfused Bへ渡します。Triton失敗時も同じrandをPyTorch fallbackへ渡すため、失敗によってCUDA RNG stateが余分に進みません。

STEの`x + (q_out - x).detach()`はPython側に残します。

### C: fused basic stats

`--dq_delta_triton_stats`を指定した場合、対応stepではBの出力と次のpartial statsを同じkernelで作ります。

```text
numel
clip_count
sumsq
xq_sumsq
xxq_sum
```

fused statsの条件:

```text
--dq_delta_use_triton
--dq_delta_triton_stats
bits mode
mode=stoch
summary/basic相当の統計
ZeroRate / NearZeroRate / AbsMax / ScaleMin/Mean/Maxを要求しない
```

主な対象はsummary/basicのLogStepと、QErrを使う`clip_rate_low_auto`のAutoStepです。

`clip_rate_high`など、`clip_rate_low_auto`以外のauto presetはclip-only統計を使います。このため、そのAutoStepが対応するbasic LogStepと重ならない場合、statsはPyTorchへfallbackします。学習開始時にもwarningを表示します。

## full・per_moduleログの動作

`--dq_delta_log_detail full`または`--dq_delta_log_mode per_module`では、fused kernelが扱わない詳細統計が必要です。

```text
学習forward       : 通常Triton A/B
log stats         : PyTorch fallback
Auto-only basic   : 条件を満たせばfused stats
```

学習開始時にwarningを出すため、full LogStepがTriton stats非対応であることに気づかないままにはなりません。

`--dq_delta_triton_stats`を付けずにPyTorch statsを使うこともできます。過去ログとのQErr比較を厳密に揃えたい場合に利用できます。

## 設定別の実際の動作

| 設定 | A | B | stats | 備考 |
| --- | --- | --- | --- | --- |
| bits/channel/rms/stoch + basic | Triton | Triton | fused Triton | 正式推奨・長時間検証済み |
| bits/channel/rms/stoch + full/per_module | Triton | Triton | PyTorch | 開始時warning |
| bits/tensor/rms/stoch | PyTorch | Triton候補 | basicならfused候補 | 学習未検証 |
| bits/channel/absmax/stoch | PyTorch | Triton候補 | basicならfused候補 | 学習未検証 |
| bits/channel/rms/det | Triton候補 | PyTorch | PyTorch | 学習未検証 |
| `--dq_delta_step` | PyTorch | PyTorch | PyTorch | Triton非対応 |
| `--dq_quantize_z` | 条件次第 | 条件次第 | 条件次第 | コード上は到達可能だが長時間未検証 |
| Triton未導入 | PyTorch | PyTorch | PyTorch | warning後にfallback |
| 個別kernel失敗 | A失敗時はPyTorch | 通常B失敗時はPyTorch | fused失敗時はPyTorch | fused失敗時も通常Bは同じrandでTritonを再試行。他の対応kernelは継続 |

「候補」はコード上の条件を満たせばそのkernelを試すという意味です。長時間学習で品質まで確認済みという意味ではありません。

## 検証済み範囲

### GPU単体回帰で確認済み

- 2D / 3D / 4D tensor
- scalar scale / per-channel scale
- float16 / bfloat16 / float32
- random値 / 量子化境界付近の値
- 通常Bとfused Bのfixed-rand bit-for-bit一致
- small/large launch境界の前後
- 実学習で多かった大shape
- PyTorch RNG終了状態
- 入力tensor非破壊と出力非alias
- fused失敗時の同一rand fallback
- STE勾配が1で通ること
- Aのchannel RMS scaleとPyTorch基準の近似一致

### 長時間学習で確認済み

中心となる検証profile:

```text
SDXL LoRA
rank=4
mixed_precision=fp16
dq target=delta
bits=8
granularity=channel
stat=rms
mode=stoch
scope=unet
summary/basic log
clip_rate_low_auto
gradient checkpointingなし
```

最新構成のXL18は8400 stepを`1:37:07`で完走し、DQログにNaN/Infやauto制御異常はなく、生成画像も良好でした。通常PyTorchの比較ラン`1:47:06`に対し、学習時間で約9.3%短縮、処理速度換算で約10.3%高速化です。

QErr系はランごとの差が大きく、XL18は高めでしたが、同等以上の値はTriton stats最適化前のランにも存在しました。ClipRate、loss、rank、LoRA RMS、生成画像は正常で、最新変更による品質回帰の兆候はありませんでした。

### 長時間学習では未検証

- 8bit以外
- bits scheduleとの併用
- tensor granularity
- absmax / none stat
- det mode
- `--dq_delta_step`
- `--dq_quantize_z`
- TEのみ / UNet+TE scope
- bf16 / float32学習
- gradient checkpointing
- SDXL以外のモデル

Tritonを有効にして検証profileと異なる設定を指定すると、差分を開始時warningへ表示します。コード上対応している部分はTritonを試し、それ以外はPyTorchへfallbackします。

## gradient checkpointing

gradient checkpointingではbackward再計算中にstatsが再収集されることを短縮実験で確認しています。現在はこの組み合わせを長時間検証済みとはしません。

通常運用でgradient checkpointingを使わない場合には影響しません。将来対応する場合は、backward再計算中のstats二重加算を防ぐ必要があります。

## 採用・不採用の経緯

### 採用

- A/B別kernel
- PyTorch乱数
- `tl.div_rn`
- stats有効stepでも通常stepと同じforward B
- B+basic stats fused
- large/small launch切替
- `torch.sum(dim=0)`によるpartial stats集約

### 不採用または削除済み

| 実験 | 結果 |
| --- | --- |
| Triton `tl.rand` | PyTorch stochastic roundingからの挙動差が大きいため不採用 |
| A+B融合 | A/B別よりログと生成結果が変わりやすく不採用 |
| AだけTriton | 品質切り分けには有効だがBがPyTorchになり遅いためCLI削除 |
| default division | B単体差が`tl.div_rn`で消えたため削除 |
| separate Triton stats | fusedより遅いため学習コードから削除 |
| Triton second-stage reduction | tuned first-stage + `torch.sum`より遅いため削除 |
| B NLC 2D tile | shapeにより速度が分かれ、短縮学習でも改善なし |
| A NLC C=1280 sweep | 出力を維持しつつ平均5%以上速い候補なし |
| 学習tensor capture hook | 調査完了後、通常学習を軽くするため削除 |

不採用結果は、同じ案を再実装しないために記録しています。

## 検証ツール

以下はvenvを有効にしてリポジトリrootから実行します。

### GPU回帰

通常の変更後:

```bash
python tools/check_triton_fake_quant.py --skip-e2e
```

Aやbroadcastまで変更した場合:

```bash
python tools/check_triton_fake_quant.py
```

終了コード`0`を合格条件とします。fixed-randは乱数列の違いを除き、算術とindexingだけを比較するための内部検証機能です。通常学習用の固定rand CLIはありません。

### CUDAベンチ

```bash
python tools/benchmark_triton_quant.py --quick
python tools/benchmark_triton_quant.py --warmup 50 --iterations 1000 --repeats 7
```

`--quick`は起動確認用です。正式測定では代表shapeについて、通常B、PyTorch basic stats、fused B+stats、PyTorch乱数とaccumulator更新を含む本番相当処理をCUDA Eventsで比較します。

過去の採否測定では、fused statsはseparate statsより対象module処理で約2.09～2.59倍速く、本番相当比較でも約2.15～2.47倍でした。これはstats対象処理だけの倍率で、学習全体の倍率ではありません。

### PyTorch基準テスト

`pytest`導入済みの開発環境では次も利用できます。

```bash
python -m pytest tests/test_rounding.py
```

検証したRTX 5080のvenvには`pytest`を追加していません。公開requirementsにも、検証だけを目的に`pytest`やTritonを追加しません。

## 保守時の注意

- 対応するPyTorch式を [library/rounding_util.py](../library/rounding_util.py) と照合する。
- 通常Bとfused Bは、同じ入力・scale・fixed-randでbit-for-bit一致させる。
- PyTorch randを1回だけ生成し、fallbackでも再利用する。
- stats非対応でもforward経路を変えない。
- 新しい設定を「検証済み」と記載するのは、GPU回帰だけでなく短縮学習または長時間学習を通した後にする。
