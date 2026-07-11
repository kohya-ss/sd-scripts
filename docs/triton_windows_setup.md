# Triton Windows 設定メモ

このメモは、dq_delta fake quant の optional Triton 高速化実験と、現在採用している安全寄りルートを記録するためのものです。

## 基本方針

- `requirements.txt` には Triton を追加しない。
- `--dq_delta_use_triton` が指定され、かつ Triton が import でき、対応条件を満たす場合だけ Triton path を試す。
- Triton がない、未対応 shape/dtype/mode、kernel 実行エラーの場合は既存の PyTorch 実装へ fallback する。
- `--dq_delta_use_triton` を付けない場合は、Triton がインストールされていても既存の PyTorch path を使う。

## まず読む現在地

公開時の安全な既定動作はPyTorch pathのままです。Tritonを使う場合の現時点の最終テスト候補は次の構成です。

```text
--dq_delta_use_triton
--dq_delta_triton_div_rn
--dq_delta_triton_stats
--dq_delta_triton_stats_mode fused
--dq_delta_log_detail basic
```

これは `8bit / channel / rms / stoch` のdq_deltaを主対象に検証した構成で、Tritonを公開環境の必須依存にはしません。CLI既定値と、検証中の推奨構成は意図的に分けています。

| オプション | CLI既定 | 現在の位置付け |
| --- | --- | --- |
| `--dq_delta_use_triton` | OFF | Triton全体の明示的な入口。今後もopt-inを維持する。 |
| `--dq_delta_triton_scale_only` | OFF | A/B差の切り分け専用。通常運用には不要。 |
| `--dq_delta_triton_div_rn` | OFF | PyTorch Bの除算に寄せる最終候補。最新構成の長時間確認後に統合・整理を判断する。 |
| `--dq_delta_triton_stats` | OFF | log/auto stats高速化の入口。未対応条件はPyTorchへfallbackする。 |
| `--dq_delta_triton_stats_mode` | `separate` | `fused`が速度面の採用候補。最新チューニング後の長時間確認までは既定値を変えない。 |
| `--dq_delta_log_detail` | `basic` | 常用向けとして採用済み。`full`は詳細診断用。 |

最新のfused stats launch設定は、GPU回帰、単体ベンチ、50-step短縮学習まで確認済みです。最新設定での通常間隔・8400 step相当の長時間ランは未実施なので、オプション削除や既定値変更はその確認後に行います。

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

## 検証の流れ

この実装は、次の順で計算差と速度差を切り分けた結果です。

1. PyTorch実装を基準に、A（channel RMS scale）とB（stochastic fake quant）を分離した。
2. 同じ入力・scale・固定randを使う単体比較と、実学習から一時取得したtensorの比較で、乱数差と算術差を分けた。学習本体のcapture hookは検証後に削除した。
3. A+B融合はログと生成結果が変わりやすかったため不採用とし、A/B別kernelとPyTorch乱数をforwardの採用ルートにした。
4. log/auto stepだけforwardがPyTorch Bへ戻らないよう、stats有効stepも通常stepと同じTriton Bへ統一した。
5. statsは、別kernelの`separate`から、Bと最小statsを同時に処理する`fused`へ段階的に進めた。通常Triton Bとfused Bは固定randでbit-for-bit一致を条件にした。
6. GPU回帰、CUDA単体ベンチ、全step statsの短縮学習、通常間隔の長時間学習の順で確認した。単体倍率だけで学習全体の短縮を判断しない。

以降の「不採用実験」は、同じ案を再実装しないための判断記録です。再現コマンドと各ツールの保証範囲は「検証ツールと再現手順」にまとめています。

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

## 不採用実験: B NLC 2D tile

B の stochastic fake quant 本体について、NLC tensor 専用の2D tile kernelを実験した。

```text
rows = N * L
cols = C
```

同じ PyTorch 乱数を渡した単体比較では、標準の1D B kernelと出力が完全一致することを確認した。

ただし、単体ベンチではshapeによって速い/遅いが分かれ、短時間学習 `220 steps` でも明確な速度改善は見えなかった。そのため、実装は削除し、正式Triton pathでは従来の1D B kernelを使う。

## 不採用実験: A NLC C=1280 config sweep

全stepで使うAのchannel RMSについて、generic kernelを使っている `C=1280` でもNLC 2Dが有利かを確認した。

```text
generic: num_warps=2/4/8
NLC 2D: BLOCK_C=8/16/32, num_warps=4/8
```

`L=32/77/468/480` と既存NLC pathの `C=10240` を比較したが、C=1280で現行genericを平均5%以上短縮し、かつ出力を維持する候補はなかった。一部のNLC候補はscaleに約 `2e-7` の相対差が出て、fixed-rand B出力も変化した。そのためAの実装は変更しない。

この比較に使ったconfig sweepスクリプトは一時検証用で、採否決定後に削除した。候補kernelも本番コードへ残していない。

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

3D NLC tensor では、A の scale 計算だけ NLC 専用の2D tile kernelを使う。これは同じ式を `rows = N * L`, `cols = C` として処理し、channel方向をまとめて読むことで、従来の channelごとの strided load を避けるため。

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

`--dq_delta_triton_stats` を指定しない標準 Triton pathでは、forwardのfake quantだけを通常stepと揃え、stats集計はPyTorchのまま行う。

```text
--dq_delta_triton_stats
```

この指定では、対応できる場合だけstats集計の主要部分もTritonで処理する。`separate`はstats専用Triton kernelの後にPyTorchで集約し、`fused`はBとpartial statsを同じTriton kernelで処理した後に `torch.sum(dim=0)` で集約する。テンソル条件が合わない場合はPyTorch stats pathへfallbackする。`--dq_delta_log_error_parts` は整理のため削除し、clip/round成分分解ログは新規ログでは出力しない。

`--dq_delta_log_detail basic` は常用向けの軽量ログで、`ZeroRate`, `AbsMax`, `Range`, `ScaleMin/Mean/Max` を計算・出力しない。詳細診断が必要な場合は `--dq_delta_log_detail full` を使う。

`--dq_delta_triton_stats` は、log / auto stepの高速化を調べるための明示的な実験オプションとして残している。fused v2は長時間ランまで正常完走しているが、全体時間の改善はラン差に隠れる規模なので、既定値にはしていない。

## 初期実測メモ

forward経路を決める過程で行った、初期の同一系統テストランの所要時間:

```text
xl01 通常 PyTorch: 1:47:06
xl03 Triton tl.rand / 融合系実験: 1:37:07
xl04 Triton PyTorch乱数 / 融合あり: 1:37:36
xl05 Triton PyTorch乱数 / 融合なし: 1:40:17
```

xl05 は通常より 409 秒短縮、学習時間で約 6.36% 短縮、処理速度で約 6.8% 高速化。

この初期実験を受けて、A/B別kernelとPyTorch乱数を使う現在のforward経路を採用した。その後、stats有効stepでも同じforward経路を使うよう統一し、fused stats v2まで追加している。

## 注意

- 通常PyTorch BとTriton Bのbit-for-bit一致は必須目標にしない。
- 通常Triton Bとfused Triton Bは、同じ入力・scale・固定randならbit-for-bit一致を必須とする。
- `--dq_delta_use_triton` を外せば既存 PyTorch path に戻る。
- Triton path は optional であり、公開環境に Triton がなくても動く必要がある。
- 追加実装時は、対応する Python 実装の式をコメントやdocsに明記する。

## B+stats fused 実験ルート

`--dq_delta_triton_stats_mode {separate,fused}` を追加した。

- `separate` は従来の stats 用Triton kernelを別に起動する方式。
- `fused` は stats有効stepだけ、stochastic fake quant B と basic stats の部分集計を同じTriton kernelで行う方式。
- 既定値は `separate`。`fused` は明示指定した場合だけ使う実験ルート。

`fused` が使われる条件は安全側に絞っている。

```text
--dq_delta_use_triton
--dq_delta_triton_stats
--dq_delta_triton_stats_mode fused
mode=stoch
--dq_delta_triton_scale_only なし
--dq_delta_log_detail basic 相当
per_module/near_zero/full detail なし
```

`--dq_delta_triton_stats_mode fused` で条件外になった場合は、通常Bでforwardを作り、statsはPyTorchで集計する。`separate` statsを使うのはmodeを明示的に`separate`にした場合だけ。`--dq_delta_triton_div_rn` を指定した場合、fused側のfake quantとclip_count判定でも同じ `tl.div_rn` 由来の値を使う。

fused v1で集計するのは basic/auto 判定に必要な最小統計のみ。

```text
numel
clip_count
sumsq
xq_sumsq
xxq_sum
```

### fused stats v2と後続チューニング

fused v2ではforward値とログ定義を変えず、stats後段だけを高速化する。

```text
B + partial stats Triton kernel
  -> partial statsの一括reduction
  -> 5要素 packed stats
  -> scope accumulatorへ1回のadd
```

初期v2はTritonのsecond-stage reductionを使っていた。その後のRTX 5080実測では、first-stageを大きくしてpartial行数を減らしたうえで `torch.sum(dim=0)` を1回使う方が一貫して速かったため、現在は次の構成にしている。

```text
n_elements >= 65536:
  BLOCK_SIZE=1024, num_warps=2
small tensor:
  BLOCK_SIZE=256, num_warps=4
partial stats:
  torch.sum(dim=0)
```

これは `--dq_delta_triton_stats_mode fused` を選んだ場合の正式構成として採用する。CLI既定値の`separate`は変更せず、比較用および互換ルートとして維持する。

学習側が使うのは5要素packed statsだけなので、moduleごとの辞書と個別view tensorも作らない。実験した出力値のレジスタ内castは一貫した高速化にならなかったため採用していない。旧Triton reduction helperは比較ベンチ用に残す。

通常Bとfused Bには内部検証用の固定randを渡せるが、通常学習ではこれまで通りPyTorch乱数を毎回1回生成する。fused失敗時は同じrandを通常B/PyTorch fallbackへ渡し、失敗の有無でCUDA RNG stateが余分に進まないようにする。

検証スクリプトでは、production Bとfused production Bを直接比較し、実戦大shape、RNG終了状態、強制fallbackを確認する。検証用randを指定するCLIは追加しない。

`ZeroRate`, `NearZeroRate`, `AbsMax`, `ScaleMin/Mean/Max` はfull/detail用として、fused v2では扱わない。

### fused v2の長時間確認

初期fused v2を使った8400 stepのテストラン（xl17）は`1:38:54`で完走した。DQログにNaN/Inf、auto判定異常、fallbackを疑う値はなく、生成用LoRAも正常だった。直前の同系統ランとの差は数十秒で、ラン変動を含むため、v2単独の全体速度改善量とは断定しない。

この完走後に、large tensorの`BLOCK_SIZE=1024, num_warps=2`化と、後段を`torch.sum(dim=0)`へ戻すチューニングを行った。こちらはGPU回帰、正式単体ベンチ、50-step短縮学習まで確認済みで、通常間隔の長時間ランは次の最終確認として残っている。

## 検証ツールと再現手順

以下のコマンドは、venvを有効にしてリポジトリrootから実行する。検証は目的の異なる5段階に分け、上の段階が通っても下の段階の代わりにはならない。

### PyTorch基準の単体テスト

```bash
python -m pytest tests/test_rounding.py
```

Tritonを使わず、元の丸め、fake quant、channel scale、STEの基本契約を確認する。Triton変更時にも、比較基準そのものを壊していないことを先に確認する。

これは`pytest`導入済みの開発環境向けです。上記のRTX 5080検証venvには`pytest`を追加しておらず、今回のTriton検証では次のGPU回帰スクリプトを主に使った。公開requirementsにも、この検証だけのために`pytest`やTritonを追加しない。

### Triton GPU回帰チェック

通常のB/stats変更後に使った短い回帰コマンド:

```bash
python tools/check_triton_fake_quant.py --skip-e2e
```

通常Bとfused Bのfixed-rand一致、大shape、PyTorch乱数を使った終了RNG state、入力非破壊、fallback時のrand再利用、LoRAModuleのSTE、stats値、gradient checkpointing時の再計算traceを確認する。終了コード`0`を合格条件とする。

Aのscale計算、broadcast、A+B end-to-endまで変更した場合は、`--skip-e2e`を外して実行する。

```bash
python tools/check_triton_fake_quant.py
```

fixed-randは、乱数列の違いを除いて算術とindexingだけを比較するために使う。`--capture-dir`は調査中に保存した既存captureを再検証するための互換機能であり、現在の学習本体にはcapture生成hookを残していない。

### 単体CUDAベンチ

モデルやデータセットを読み込まず、代表shapeの疑似tensorをCUDA Eventsで測る。

```bash
python tools/benchmark_triton_quant.py --quick
python tools/benchmark_triton_quant.py --warmup 50 --iterations 1000 --repeats 7
```

`--quick`はkernelが起動し、比較前のfixed-rand一致検査が通ることを見るsmoke testとして使う。速度採否には2行目の正式条件を使った。`--shape 1,480,10240`は繰り返し指定でき、`--dtype`と`--no-div-rn`で条件を変えられる。

主な測定対象:

```text
normal B (fixed rand / PyTorch rand)
normal B + PyTorch stats
normal B + separate Triton stats
fused v2
production-like separate (PyTorch rand + stats + scalar accumulator)
production-like fused (PyTorch rand + packed accumulator)
partial stats: torch.sum vs Triton reduction
packed accumulator add
```

`reduce_speedup`、`separate_to_fused`、`pytorch_stats_to_fused`、`production_separate_to_fused`は`1.0`より大きいほど右辺側の処理が速い。特に`production_separate_to_fused`が、PyTorch乱数生成とscope accumulator更新を含む最も本番に近い単体比較である。コンパイルをwarmupから除外し、各測定のmedianとminを出力する。

RTX 5080環境でwarmup 50回、1000 iteration、7 repeatを測った時点では、チューニング後fusedはseparate stats比で代表6 shapeすべて約2.09～2.59倍だった。PyTorch乱数とaccumulator更新を含む本番相当比較でも約2.15～2.47倍だった。`(1,32,1280)` のsmall launch側も約2.24倍で、65536要素の分岐より小さいshapeに悪化は見られなかった。forward出力はfixed-randで完全一致し、packed statsの相対差は最大でも約 `1.6e-7` だった。これはstats対象moduleの処理時間であり、学習全体の短縮率ではない。

全50 stepをlog/auto対象にした短縮学習では、separateが約40秒・1.24 it/s、チューニング後fusedが約38秒・1.30 it/sだった。通常の50/100 step間隔では対象stepが少ないため、8400 step全体への寄与は数秒程度と見積もる。

同じ50 stepログを比較すると、`QErrPerClip` は48/51行で異なり最大相対差は約15%だった。一方、`ClipRateLowAutoBad`, bad streak, `AutoReason`, auto decision, `RangeMulAfter` は全51行で一致した。派生QErr値にはreduction定義・順序の差が見えるが、この短縮確認ではauto制御結果は変わっていない。

### 全step statsの短縮学習

通常の学習コマンドへ次を追加し、`separate`と`fused`で出力名だけを変えて各50 step実行した。

```text
--max_train_steps 50
--dq_delta_log --dq_delta_log_every 1 --dq_delta_log_detail basic
--dq_delta_auto_range_mul --dq_delta_auto_every 1
--dq_delta_use_triton --dq_delta_triton_div_rn
--dq_delta_triton_stats --dq_delta_triton_stats_mode separate

# 比較側では最後だけ変更
--dq_delta_triton_stats_mode fused
```

これはlog/auto対象stepを意図的に増やし、stats経路の速度とログを短時間で比較する負荷試験である。品質評価や通常間隔での全体短縮率の代わりにはしない。同じseedと学習条件を使い、Tritonの初回コンパイルを含む最初の1回だけで採否を決めない。

### 通常間隔の最終学習

最終確認では`dq_delta_log_every=100`、`dq_delta_auto_every=50`の通常間隔へ戻し、「まず読む現在地」の候補構成で長時間ランを行う。確認項目は次の通り。

- 学習完走、生成画像、DQログとautoログに異常がない。
- 終了時の`fused_stats_calls`が0より大きく、`fused_fallback_calls=0`。
- `AutoReason`, auto decision, `RangeMulAfter`に不自然な変化がない。
- 所要時間は数十秒のラン変動を考慮し、近い条件の複数ランと比較する。

### 実学習のstats経路集計

statsを1回以上収集した学習では、終了時にmain rankのローカル集計を1行だけ出力する。

```text
dq_delta stats paths (main rank):
  fused_stats_calls=...
  separate_stats_calls=...
  pytorch_stats_calls=...
  fused_fallback_calls=...
  fused_elements=...
  backward_trace_windows=...
  backward_recompute_stats_calls=...
```

`fused_fallback_calls=0`なら、試行したfused kernelはすべて成功している。gradient checkpointing利用時に`backward_recompute_stats_calls>0`なら、backward再計算中にもstatsが再加算されている。現段階ではtraceのみで、収集停止処理はまだ入れていない。

同一条件の3 step短縮ランで全stepをlog/auto対象にした場合、gradient checkpointingなしでは`fused_stats_calls=2166`、ありでは`fused_stats_calls=4266`かつ`backward_recompute_stats_calls=2100`だった。実モデルでもbackward再計算による重複収集を確認した。`--gradient_checkpointing`を使わない通常運用には関係しないため、現時点では将来このオプションを使う場合の課題として記録するだけとし、収集停止処理は実装しない。
