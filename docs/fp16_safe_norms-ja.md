# fp16_safe_norms のモードと高速化検証

## 使い方

高速化版は、従来の `--fp16_safe_norms` を次に置き換えて使います。

```text
--fp16_safe_norms_mode native_accum
```

モードは次の3種類です。

- `off`: 安全正規化を無効化。
- `strict`: 従来の `--fp16_safe_norms` と同じ。LayerNorm/GroupNormと通常Attention経路のSoftmaxを明示的にfp32で計算し、出力をfp16へ戻す。
- `native_accum`: CUDA LayerNormだけ、fp16入出力のnative kernelとfp32のmean/rstdを使用する。GroupNormと通常Attention経路のSoftmaxは安全性のため `strict` のまま。

Softmaxのstrict処理は通常Attention経路だけに適用されます。`--sdpa` / `--xformers` 使用時のSoftmax内部精度は、それぞれのAttention実装に依存します。

従来の `--fp16_safe_norms` だけを指定した場合は、互換性のため `strict` になります。実際に解決されたモードは起動時ログと保存モデルの `ss_fp16_safe_norms_mode` metadataに記録されます。

`--torch_compile` 使用時は、コンパイル中だけTorchDynamoが追跡できるpublic autocast contextへ切り替えます。通常のeager実行では、オーバーヘッドの小さいprivate guardを引き続き使用します。

## 2026-08-03 検証結果

環境はRTX 5080 16GB、PyTorch 2.9.1+cu130、CUDA 13.0、fp16、SDPAです。実データの720px bucketに対応するLayerNorm形状を使いました。

### CUDA microbenchmark

値はCUDA Eventで計測した1回当たりのmedianです。

| 対象 | 形状 | strict | native_accum | 高速化 |
|---|---:|---:|---:|---:|
| LayerNorm forward | `[1,2016,640]` | 0.0364 ms | 0.0156 ms | 2.34倍 |
| LayerNorm forward+backward | `[1,2016,640]` | 0.1438 ms | 0.0830 ms | 1.73倍 |
| LayerNorm forward | `[1,504,1280]` | 0.0353 ms | 0.0154 ms | 2.29倍 |
| LayerNorm forward+backward | `[1,504,1280]` | 0.1474 ms | 0.0833 ms | 1.77倍 |
| SDPA Transformer Block forward | `[1,2016,640]` | 0.8111 ms | 0.7141 ms | 13.6%短縮 |
| SDPA Transformer Block forward+backward | `[1,2016,640]` | 2.4257 ms | 2.2925 ms | 5.8%短縮 |
| SDPA Transformer Block forward | `[1,504,1280]` | 0.5547 ms | 0.4808 ms | 15.4%短縮 |
| SDPA Transformer Block forward+backward | `[1,504,1280]` | 1.2240 ms | 1.1169 ms | 9.6%短縮 |

LayerNormと直後のQ/K/V Linearを10回実行したProfilerでは、`aten::_to_copy` の呼び出しが `off=60`、`strict=40`、`native_accum=0` でした。高速化の主因が明示castの削減であることを確認しています。

### 短時間の実学習A/B

実モデル、実データセット、batch 1、720px bucket、SDPA、LoRA dim 4、同じseedを使用しました。初期20 stepを除外し、測定区間の前後でCUDAを同期しています。

| 測定区間 | strict | native_accum | native_accumの短縮 |
|---|---:|---:|---:|
| 180 step | 325.14 ms/step | 316.31 ms/step | 2.72% |
| 100 step（逆順で再測定） | 312.52 ms/step | 293.83 ms/step | 5.98% |

比較用の `off` は同じ100-step区間で311.02 ms/stepでした。この条件では `strict` は `off` とほぼ同速、`native_accum` は両方より高速でした。

現在の本番設定にはText Encoder学習、DQ、rank log、checkpoint保存など、LayerNormと無関係な処理も含まれます。そのため、本番全体での改善率は上記より薄まる可能性があります。GPU clockやbucket順でも数%揺れるため、長時間学習では低い1桁%の改善候補として扱ってください。

## 数値安全性

- 実bucket形状と入力scale `1, 32, 256, 2048` で、LayerNormの出力と入力勾配は `strict` とbit一致。
- dim 640/1280の実SDPA Transformer Blockでも、出力・hidden state勾配・context勾配がbit一致。
- native LayerNormの出力はfp16、保存されるmean/rstdはfp32であることを確認。
- native GroupNormは保存統計もfp16になり、出力・勾配に最大0.00390625の差が出たため高速化対象にしていません。
- 短時間実学習でNaN/Infはなく、最終表示lossも一致。独立run間のLoRA重み差は、同一モードを繰り返した通常の非決定性の範囲内でした。

CPU上のfp16入力、またはLayerNorm affineのdevice/dtype不一致では、自動的に `strict` へフォールバックし、初回だけwarningを出します。bf16/fp32入力ではnative高速経路を使用せず、通常のLayerNormを実行します（warningは出ません）。
