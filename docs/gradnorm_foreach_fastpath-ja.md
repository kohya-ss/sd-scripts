# GradNorm Guardian foreach 高速化ガイド

## この文書について

この文書は、`train_network.py` の GradNorm Guardian に追加した foreach 高速化について、次の内容をまとめたものです。

- 何を高速化したのか
- なぜ従来処理が遅かったのか
- 高速経路と従来経路をどう切り替えているのか
- 高速経路を通る条件
- mixed precision、FP32、FP16、BF16との関係
- `--grad_cosine_log` を有効にした場合の影響
- フォールバックと互換性の考え方
- 今後コードを変更するときの注意点

GradNorm Guardianそのもののスキップ条件、移動平均、NaN/Inf処理、ログ列については、[skip_grad_norm機能仕様](skip_grad_norm_README-ja.md)を参照してください。

## 初心者向けの概要

### GradNorm Guardianとは

学習では、各stepで「パラメータをどの方向へ、どの程度動かすか」を表す勾配が作られます。

GradNorm Guardianは、学習対象になっているすべての勾配をまとめた大きさ（L2ノルム）を計算し、次の判断や記録に使う安全装置です。

- 勾配が異常に大きくなっていないか
- NaNやInfが発生していないか
- そのstepの更新をスキップするべきか
- 勾配ノルムをCSVへ記録するか

計算内容を簡略化すると次の形です。

```text
勾配ノルム = √(勾配1² + 勾配2² + 勾配3² + ……)
```

今回変更したのは、この値の意味やスキップ判定ではありません。同じ勾配ノルムを、GPUが処理しやすい方法で計算するようにしました。

### 従来処理が遅かった理由

代表的なLoRAでは、約907万要素の勾配が1個の大きなTensorに入っているのではなく、1,134個の小さなTensorに分かれています。

従来処理は、ParameterごとにPythonのループを回していました。

```text
Tensor 1を二乗 → 合計へ加算
Tensor 2を二乗 → 合計へ加算
Tensor 3を二乗 → 合計へ加算
……
Tensor 1134を二乗 → 合計へ加算
```

GPUは大きな計算をまとめて処理することが得意です。一方、小さなCUDA処理をCPUから何度も起動すると、計算そのものよりも起動や待ち合わせの負担が目立つことがあります。

運送に例えると、1,134個の荷物を1個ずつ発送し、そのたびに伝票を作っていた状態です。

### foreach高速化で変えたこと

PyTorchのforeach（multi-tensor）機能を使い、複数の勾配Tensorをまとめてノルム計算へ渡すようにしました。

```text
従来:
  Tensor 1を処理
  Tensor 2を処理
  Tensor 3を処理
  ……

foreach:
  対応するTensor群をまとめて処理
```

すべてが必ず1回のCUDA起動になるという意味ではありませんが、Parameterごとに逐次処理する場合と比べて、CUDA処理の起動回数を大幅に減らせます。

## 実装した内容

### 1. Parameter一覧のキャッシュ

`GradNormGuardian._get_parameters()`は、最初の呼び出しで`tuple(model.parameters())`を保存します。

学習中に毎step変わる可能性があるのは、Parameterそのものの一覧ではなく、主に各Parameterに勾配が存在するかどうかです。そのため、次のように分けています。

- Parameter一覧: 初回だけ取得してキャッシュ
- `param.grad is not None`の確認: 毎step実施

module dropoutやText Encoderのfreezeにより、有効な勾配が途中で変わっても対応できます。

このキャッシュは、同じ`GradNormGuardian`へ別のmodelオブジェクトが渡された場合には作り直されます。

### 2. コサインログ無効時の高速経路

`GradNormGuardian.observe()`は、`log_grad_cosine`が無効な場合だけ、有効な勾配をリストへ集めて`_calculate_grad_norm()`へ渡します。

```python
if not use_cosine:
    grads = [
        param.grad.detach()
        for param in parameters
        if param.grad is not None
    ]
    current_grad_norm_tensor = _calculate_grad_norm(grads)
else:
    # 従来のGradNorm計算とコサイン類似度計算
```

`detach()`は勾配Tensorを読み取り用として扱うためのもので、勾配の値を変更する処理ではありません。

### 3. 公開APIと旧PyTorch互換経路

foreachノルムの実装は、利用できるAPIによって次の順番で選択します。

1. `torch.nn.utils.get_total_norm(..., foreach=True)`
2. 公開APIがないPyTorchでは`torch._foreach_norm`

公開APIを優先しつつ、PyTorch 2.1系でも高速経路を利用できるようにしています。

### 4. 従来処理の保存

元のParameter単位の計算は、`_legacy_grad_norm()`として残しています。

高速経路を利用できない場合や、foreachが実行環境で未対応だった場合は、この従来処理へ戻ります。

## ソースコードの構造

主な実装は[train_network.py](../train_network.py)にあります。

| 関数・クラス | 役割 |
|---|---|
| `GradNormGuardian.observe()` | コサインの有無で大きな処理経路を分け、ノルムをskip判定やログへ渡す |
| `GradNormGuardian._get_parameters()` | modelのParameter一覧を初回だけキャッシュする |
| `_calculate_grad_norm()` | 高速経路を試し、利用できなければ従来経路へ戻す |
| `_can_use_foreach_grad_norm()` | dtype、device、layout、APIの有無などを確認する |
| `_foreach_grad_norm()` | 公開APIまたは`torch._foreach_norm`でmulti-tensor計算する |
| `_legacy_grad_norm()` | 元のParameter単位のノルム計算を行う |
| `_is_unsupported_foreach_error()` | foreach未対応として安全にフォールバックできるエラーか判定する |

処理全体は次のような構造です。

```text
GradNormGuardian.observe()
│
├─ Parameter一覧をキャッシュから取得
│
├─ コサインログがON
│   └─ 従来のTensor単位GradNorm計算
│      ＋前stepとのコサイン類似度計算
│
└─ コサインログがOFF
    └─ gradが存在するTensorだけを抽出
       └─ _calculate_grad_norm()
          │
          ├─ foreach利用条件を満たす
          │   └─ _foreach_grad_norm()
          │      ├─ 公開get_total_norm
          │      └─ 旧PyTorch用torch._foreach_norm
          │
          └─ 条件外またはforeach未対応
              └─ _legacy_grad_norm()
```

## 高速経路を通る条件

高速経路を使うには、次の条件をすべて満たす必要があります。

1. `--grad_cosine_log`が無効
2. `grad is not None`の勾配が1個以上ある
3. foreachノルムAPIが利用できる
4. すべての勾配がFP32
5. すべての勾配が通常のdense Tensor（`torch.strided`）
6. すべての勾配が同じdevice上にある
7. すべての勾配が同じdtypeである
8. deviceがCPUまたはCUDA
9. その実行環境でforeachが未対応として無効化されていない

主な対象は、単一GPUで通常のmixed-precision LoRA学習を行い、LoRAパラメータと勾配がFP32になっている環境です。

## mixed precisionと勾配dtype

### `mixed_precision="fp16"`でも高速化できる

ここで確認しているFP32は、学習全体の計算精度ではなく、`param.grad.dtype`です。

通常のmixed precisionでは、重いforward/backward計算にFP16やBF16を使いながら、学習対象のLoRAパラメータと勾配はFP32のままになることがあります。

```toml
mixed_precision = "fp16"
full_fp16 = false
```

この一般的な構成では、LoRA勾配がFP32なら高速経路の対象です。

GradScalerは勾配の数値を拡大しますが、FP32 Tensorを自動的にFP16 Tensorへ変更するものではありません。

### `full_fp16`と`full_bf16`は対象外

次の設定では、LoRAネットワーク自体が低精度dtypeへ変換されます。

```toml
mixed_precision = "fp16"
full_fp16 = true
```

または、

```toml
mixed_precision = "bf16"
full_bf16 = true
```

この場合、勾配もFP16またはBF16になるため、高速経路を使わず従来経路へ戻ります。

| 代表的な設定 | LoRA勾配の典型的なdtype | foreach高速経路 |
|---|---:|:---:|
| `mixed_precision="fp16"` | FP32 | 対象 |
| `mixed_precision="bf16"` | FP32 | 対象 |
| `full_fp16=true` | FP16 | 対象外 |
| `full_bf16=true` | BF16 | 対象外 |
| `fp8_base=true`のみ | 通常はFP32 | 対象 |
| カスタム処理でLoRAをhalf化 | FP16 | 対象外 |

実際のdtypeは、利用するnetwork moduleや追加処理によって変わる可能性があります。最終的な判定には、各stepの`param.grad.dtype`が使われます。

## コサイン類似度ログとの関係

`--grad_cosine_log`を有効にすると、今回のforeach高速経路は使いません。

コサイン類似度を計算するには、現在の勾配ノルムだけでなく、次の処理が必要だからです。

- 前stepの各勾配をParameter単位で保持
- 現在と前stepの勾配の内積をParameter単位で計算
- Parameter構成や有効勾配が変わっていないか確認

この処理は従来のループ構造と密接に結び付いているため、互換性を優先して既存経路を残しています。

```text
コサインログOFF:
  foreach高速経路を利用可能

コサインログON:
  従来のGradNorm計算
  ＋コサイン類似度計算
```

コサインログをONにしても、Parameter一覧のキャッシュは引き続き使われます。ただし、foreachによる大きな高速化効果は失われます。

また、前stepの勾配を複製保持するため、追加のVRAMを使用します。9,074,688要素のFP32勾配なら、保持中の前step勾配だけで概算約35MiBです。勾配を入れ替える途中では、新旧のコピーが一時的に共存する場合もあります。

コサイン類似度は学習結果を改善する機能ではなく、勾配方向を調査する診断機能です。通常学習ではOFFにし、必要な短時間だけONにする使い方を推奨します。

`stable`、`stable_no_threshoff`、`gamble`の各プリセットでは、コサインログは無効です。

## フォールバックの動作

### 条件に合わない場合

FP16、BF16、sparse、混在dtype、混在deviceなどは、foreachを試さず最初から`_legacy_grad_norm()`を使います。

高速化の対象外になるだけで、GradNorm Guardian自体が無効になるわけではありません。

### foreachが環境的に未対応の場合

条件を満たしていても、PyTorchやbackendがforeachに対応していない場合があります。

次のような「未対応」を示す例外が発生した場合は、従来処理へフォールバックし、その後のstepでも同じ失敗を繰り返さないように高速経路を無効化します。

- `TypeError`
- `NotImplementedError`
- foreachが未対応であることを示す`RuntimeError`

CUDA out of memoryなど、foreach未対応とは関係のないエラーは握りつぶさず、そのまま送出します。

## 数値と学習挙動の互換性

高速経路と従来経路は、数学的には同じL2ノルムを計算します。

ただし、GPU上で値を足し合わせる順番が変わるため、有限値にごく小さな浮動小数点差が出る可能性があります。テストでは許容誤差内で一致することを確認しています。

従来処理ではFP32値を先に二乗するため、極端に大きい値がInfになったり、極端に小さい値が0になったりします。高速経路でも最後に`√(norm²)`を通し、このオーバーフロー／アンダーフロー分類を従来処理に合わせています。

今回の変更では、次の仕様を変更していません。

- GradScalerがunscaleする前の、スケール適用済み勾配を使う
- 移動平均窓のサイズ
- 動的しきい値の計算
- `skip_grad_norm_max`
- NaN/Infの分類と処理
- skip条件
- CSVの列と出力形式
- CLIオプションとプリセット
- コサインログ有効時の既存処理
- 勾配Tensorそのものの値

## 性能測定

### GradNorm計算単体

1,134 tensors、9,074,688 elementsで測定した結果です。

| 経路 | 中央値 |
|---|---:|
| 従来処理 | 18.47ms |
| foreach処理 | 2.15ms |

- 約16.33ms短縮
- 約8.6倍高速化

### 実学習

同じ設定の8,400 stepで比較した結果です。

|  | 所要時間 |
|---|---:|
| 変更前 | 1時間34分35秒 |
| 変更後 | 1時間27分35秒 |

- 7分短縮
- 約7.4%短縮
- 観測上は約50ms/step短縮

GradNorm単体の測定差は約16msです。実学習の差には、CUDA処理の混雑や同期タイミング、温度、データ読み込みなどの実行時変動も含まれるため、約50ms/stepのすべてが直接的なforeach効果だとは断定できません。

## テスト

回帰テストは[tests/test_speed_optimizations.py](../tests/test_speed_optimizations.py)にあります。

主に次の項目を確認しています。

- 複数勾配
- ゼロ勾配
- すべて`grad=None`
- 一部だけ`grad=None`
- 従来計算との数値一致
- NaN/Infの分類
- 極端なFP32値の分類
- 計算前後で勾配が変更されないこと
- stepごとに有効勾配が変わる場合のParameterキャッシュ
- FP16、BF16、sparse、混在dtypeのフォールバック
- 公開APIがないPyTorchの互換経路
- foreach未対応時に1回だけ失敗して従来経路へ移ること
- 想定外のエラーを隠さないこと
- コサインログ有効時の既存経路
- moving window、しきい値上限、skip判定、ログ列

実行例:

```powershell
python -m pytest tests/test_speed_optimizations.py -q
```

## 保守時の注意点

### Parameter構成を途中で変更しない

Parameter一覧のキャッシュは、最初の`observe()`以降に同じmodelのParameter構成が変わらないことを前提にしています。

`grad=None`への変化、module dropout、freezeには対応します。一方、学習途中で新しいParameterを登録したり、Parameterオブジェクト自体を交換したりする機能を追加する場合は、キャッシュの無効化も必要です。

### 従来経路を削除しない

従来経路は単なる古いコードではなく、次のための安全網です。

- full FP16/BF16
- sparse勾配
- 特殊なdevice
- 混在dtype/device
- foreach非対応のPyTorch/backend
- コサイン類似度ログ

高速経路の対応範囲を広げる場合も、非対応環境のフォールバックと回帰テストを維持してください。

### `max_grad_norm`のgradient clippingとは別機能

GradNorm Guardianのノルム計算は、異常なstepをスキップするか判断するためのものです。

`--max_grad_norm`によるgradient clippingも別途ノルムを計算しますが、GradScalerや分散学習との実行順序が異なります。現在は安全性を優先し、両者のノルム計算を共有していません。
