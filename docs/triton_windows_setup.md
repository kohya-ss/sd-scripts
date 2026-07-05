# Triton Windows 設定メモ

このメモは、フェイク量子化の optional 高速化実験で使うローカル Triton 環境と実装方針を記録するためのものです。

## 方針

- `requirements.txt` には `triton` / `triton-windows` を追加しない。
- Triton 対応は optional backend として扱う。
- Triton がインストールされていて、ユーザーが明示的に有効化した場合だけ Triton path を使う。
- Triton がない、未対応条件、実行時エラーなどの場合は、既存の PyTorch 実装へ fallback する。

## 確認済みのローカル環境

ローカルの学習用 venv で確認したバージョン:

```text
Python: 3.10 venv
torch: 2.9.1+cu130
torch.version.cuda: 13.0
torchvision: 0.24.1+cu130
torchaudio: 2.9.1+cu130
accelerate: 0.30.0
triton-windows: 3.5.1.post24
triton.__version__: 3.5.1
```

venv の metadata では以下も確認した:

```text
venv Python config: 3.10.11
venv path: D:\python\maruo-main02\sd-scripts\venv
```

## 実行したインストールコマンド

venv を有効化した状態で以下を実行:

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -m pip install -U "triton-windows>=3.5,<3.6"
python -c "import triton; print(triton.__version__)"
```

確認できた出力:

```text
2.9.1+cu130 13.0
Successfully installed triton-windows-3.5.1.post24
3.5.1
```

## このバージョン範囲にした理由

この環境では PyTorch 2.9.1 + CUDA 13.0 を使っている。

最初の実験では、最新版の `triton-windows` へいきなり上げるのではなく、PyTorch 2.9 世代に近い Triton 3.5 系に合わせるため、以下の範囲を指定した。

```powershell
python -m pip install -U "triton-windows>=3.5,<3.6"
```

もし RTX 5080 環境でこの組み合わせがうまく動かない場合は、次の候補として `triton-windows` 3.7 系を試す。

```powershell
python -m pip install -U "triton-windows>=3.7,<3.8"
```

## フェイク量子化追加コストの大まかな分布

フェイク量子化で増えた時間を `+100` とした場合の、現時点での大まかな見立て:

```text
FQ追加コスト +100
  |
  |-- A. channel RMS scale計算        25-40
  |
  |-- B. stoch fake quant本体         35-50
  |
  |-- C. stats/log/auto用 reduction   10-25
  |
  |-- D. Python分岐/STE/contiguous等    5-10
```

各項目の意味:

```text
A:
  compute_scale_bits(..., granularity="channel", stat="rms")
  channel ごとに RMS を計算して scale を作る部分。

B:
  fake_quantize_levels(..., mode="stoch")
  scale で割る、floor、乱数、確率比較、clamp、scale を掛ける、dtype を戻す部分。

C:
  dq_delta_log / dq_delta_auto_range_mul 用の統計収集。
  clip_count、zero_count、sumsq、xq_sumsq、xxq_sum、absmax、clip_err/round_err など。

D:
  Python 側の分岐、関数呼び出し、STE の組み立て、必要に応じた contiguous 化など。
```

## このリポジトリでの使い方

実装済みの有効化フラグ:

```text
--dq_delta_use_triton
```

このフラグはデフォルト OFF。指定した場合のみ、対応条件を満たす通常 path の dq_delta fake quant 関連処理で Triton kernel を試す。

予定している optional path:

```text
Triton がインストールされている
+ 明示フラグで有効化されている（--dq_delta_use_triton）
+ CUDA tensor
+ 対応 dtype / shape
+ 対応 fake-quant mode
=> Triton kernel を使う

それ以外
=> 既存の PyTorch 実装を使う
```

現在 Triton 化している対象:

```text
A: channel RMS scale 計算
   x.float()
   x ** 2
   channel ごとに sum / mean
   sqrt
   range_mul / qmax を掛ける

B: stochastic fake quantization
   x / scale
   floor
   擬似乱数によるしきい値判定
   clamp
   * scale
   dtype 戻し
```

現在は `--dq_delta_use_triton` を付けた場合、対応条件を満たす A と B をそれぞれ別 kernel で Triton 実行する。

stats 収集は PyTorch のまま残す。

後続の実験候補:

```text
C: dq_delta log / auto 用 stats 収集
A+B: scale 計算と fake quant の融合
```

## Triton path に入っているか確認する方法

`--dq_delta_use_triton` を付けると、`networks.lora` にフラグが届いた時点で以下のログが出る。

```text
dq_delta_use_triton is enabled for networks.lora. Only the normal stochastic fake-quant path is eligible; stats/log steps may still use PyTorch.
```

大量に出る一時的な shape / summary ログは削除済み。

## PyTorch path と Triton path の確認メモ

開発中の短い単体ベンチでは以下のような結果になった:

```text
torch=2.9.1+cu130 cuda=13.0 device=NVIDIA GeForce RTX 5080
dtype=torch.float16 warmup=3 iters=5

      (1, 77, 768)  elements=     59136  A_scale torch=  0.0225 triton=  0.0057 speedup=  3.96x  B_quant torch=  0.0624 triton=  0.0103 speedup=  6.04x
    (1, 468, 1280)  elements=    599040  A_scale torch=  0.0302 triton=  0.0141 speedup=  2.14x  B_quant torch=  0.0680 triton=  0.0125 speedup=  5.45x
```

このベンチは A/B 単体の比較なので、学習全体の it/s 改善率とは一致しない。
学習全体で差が小さい場合は、C の stats/log/auto reduction、UNet 本体などが支配的な可能性がある。

学習中の実測では、`--dq_delta_use_triton` なし/ありで以下のような差が見えた。

```text
tritonなし: 1.33-1.37 it/s 程度
tritonあり: 1.45-1.48 it/s 程度
```

データセット順、bucket、GPUクロック、温度、Windows側の負荷で揺れるため、最終判断は十分な step 数の平均で見る。

## 注意点

- Triton 側の stochastic rounding は `torch.rand_like` と完全に同じ乱数列にはならない。
- bit-for-bit 一致ではなく、分布として同等であることを期待する。
- STE の `x + (q - x).detach()` は、強い理由がなければ Python/PyTorch 側に残す。
- benchmark では、速度だけでなく `ClipRate`、`QuantErrRatio`、生成サンプルも PyTorch path と比較する。
