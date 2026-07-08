# Triton Windows 設定メモ

このメモは、dq_delta fake quant の optional Triton 高速化実験と、現在採用している安全寄りルートを記録するためのものです。

## 基本方針

- `requirements.txt` には Triton を追加しない。
- `--dq_delta_use_triton` が指定され、かつ Triton が import でき、対応条件を満たす場合だけ Triton path を試す。
- Triton がない、未対応 shape/dtype/mode、kernel 実行エラーの場合は既存の PyTorch 実装へ fallback する。
- `--dq_delta_use_triton` を付けない場合は、Triton がインストールされていても既存の PyTorch path を使う。

## 検証環境

ユーザー環境で確認したバージョン:

```text
PyTorch: 2.9.1+cu130
CUDA: 13.0
Python: cp310 venv
Triton Windows: triton-windows 3.5.1.post24
GPU: RTX 5080
```

インストールコマンド:

```bash
python -m pip install -U "triton-windows>=3.5,<3.6"
python -c "import triton; print(triton.__version__)"
```

## 現在の採用ルート

現在の標準 Triton path は、A と B を別 kernel で実行する。

```text
A: channel RMS scale 計算
B: stochastic fake quant
```

`--dq_delta_use_triton` を付けると、対応条件を満たす通常stepで以下を行う。

```text
A: compute_scale_bits(..., granularity="channel", stat="rms") 相当を Triton で実行
B: fake_quantize_levels(..., mode="stoch") 相当を Triton で実行
```

stochastic rounding の乱数は Triton の `tl.rand` ではなく、PyTorch の `torch.rand_like` 系で生成し、その乱数テンソルを Triton kernel に渡す。これは元の PyTorch 実装の乱数挙動に寄せるため。

STE の `x + (q - x).detach()` は Python 側に残している。

## 実験用: scale only

品質差の切り分け用に `--dq_delta_triton_scale_only` を追加した。

```text
--dq_delta_use_triton --dq_delta_triton_scale_only
```

この指定では、A の channel RMS scale 計算だけを Triton にし、B の stochastic fake quant 本体は PyTorch 実装に戻す。

```text
A: compute_scale_bits(... channel/rms) -> Triton
B: fake_quantize_levels(... stoch)    -> PyTorch
```

目的は、B の Triton 化による stochastic rounding の細かな挙動差が品質に効いているかを確認すること。速度は A+B Triton より落ちる可能性があるが、元の PyTorch path への寄り具合を調べやすい。

## 実験用: div_rn

後半stepの capture 検証で、B 単体の default division 差は `tl.div_rn` で消えることを確認した。
本番の Triton B でも同じ挙動を試すため、`--dq_delta_triton_div_rn` を追加した。

```text
--dq_delta_use_triton --dq_delta_triton_div_rn
```

この指定では、Triton B の `y = x / scale` を `tl.div_rn(x, scale)` に切り替える。
`--dq_delta_triton_scale_only` を指定している場合は B が PyTorch になるため、このオプションは実質的に影響しない。

学習本体に入れていた fake quant 入力 capture hook は、通常学習を軽く保つため削除した。
追加調査が必要な場合は、学習本体ではなく `tools/check_triton_fake_quant.py` 側の検証で行う。

## Python 実装との対応関係

Triton A は [library/rounding_util.py](../library/rounding_util.py) の `compute_scale_bits` のうち、以下の条件に対応する。

```text
granularity == "channel"
stat == "rms"
x.ndim in (2, 3, 4)
x is contiguous CUDA tensor
```

対応する式:

```python
rng = torch.sqrt(torch.mean(x.to(torch.float32) ** 2, dim=reduce_dims, keepdim=True) + eps) * range_mul
scale = (rng / qmax).to(torch.float32)
```

Triton B は [library/rounding_util.py](../library/rounding_util.py) の `fake_quantize_levels(..., mode="stoch")` に対応する。

対応する式:

```python
y = x.to(torch.float32) / scale.to(torch.float32)
q_floor = torch.floor(y)
probs = (y - q_floor).clamp(0.0, 1.0)
q = q_floor + (torch.rand_like(probs) < probs).to(y.dtype)
q = torch.clamp(q, qmin, qmax)
q_out = (q * scale).to(x.dtype)
out = x + (q_out - x).detach()
```

Triton kernel 内では `q_out` までを計算し、STE は呼び出し元の Python で付ける。

## A+B 融合について

A+B 融合 kernel は実験したが、量子化ログと生成結果が A/B 別 kernel より変わりやすかったため削除した。

特に以下の実験では、融合ありよりも融合なしの方が通常 PyTorch path に近く、生成結果も良好だった。

```text
xl05:
  A/B別Triton
  PyTorch乱数
  A+B融合なし
```

そのため、現在の正式候補では `--dq_delta_use_triton` を付けても A+B 融合は使わない。

## z 量子化について

`--dq_quantize_z` との組み合わせは未検証。コード上は同じ `compute_scale_bits` / `fake_quantize_levels` を通るため、対応条件を満たせばTriton pathに入る可能性はある。

ただし、今回の主な検証対象は `delta` 量子化であり、`z` 量子化は experimental 扱い。

## log / auto step

`--dq_delta_use_triton` が有効な場合、stats が有効な log / auto step でも、学習 forward に使う fake quant 出力は通常 step と同じ `fake_quantize_levels(...)` で作る。

これにより、通常 step は Triton B、log / auto step だけ PyTorch B になる、という forward 経路の混在を避ける。

標準 Triton path では、forward の fake quant は通常 step と揃え、stats の集計自体は PyTorch のまま行う。

```text
--dq_delta_triton_stats
```

この指定では、対応できる場合だけ stats reduction も Triton で集計する。`--dq_delta_log_error_parts` が有効な場合や、テンソル条件が合わない場合は PyTorch stats path に fallback する。

`--dq_delta_triton_stats` は、log / auto step の高速化を調べるための実験用オプションとして残している。現時点では、通常の採用候補は `--dq_delta_use_triton` による forward 経路統一まで。

## 実測メモ

同一系統のテストランでの所要時間:

```text
xl01 通常 PyTorch: 1:47:06
xl03 Triton tl.rand / 融合系実験: 1:37:07
xl04 Triton PyTorch乱数 / 融合あり: 1:37:36
xl05 Triton PyTorch乱数 / 融合なし: 1:40:17
```

xl05 は通常より 409 秒短縮、学習時間で約 6.36% 短縮、処理速度で約 6.8% 高速化。

xl05 のログは通常 PyTorch path に近く、生成結果も良好だったため、現在の採用候補。

## 注意

- bit-for-bit 一致は目標にしない。
- `--dq_delta_use_triton` を外せば既存 PyTorch path に戻る。
- Triton path は optional であり、公開環境に Triton がなくても動く必要がある。
- 追加実装時は、対応する Python 実装の式をコメントやdocsに明記する。
