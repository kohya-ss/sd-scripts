# DQ Dataset Profiler: Trajectory channel 検証判断

## 結論

128-stepの累積更新方向を使う`Trajectory` channelは、通常の診断レポートでは
**候補選択に使わない**。製品向け診断はLocal Body／Tailを既定とし、Trajectoryは
明示的に有効化する研究用の説明計測として残す。

この判断は、最終画質の良否ではなく、Local測定とTrajectory測定が同じ候補を
安定して順位付けできるかを検証した結果である。

## 検証条件

- 匿名化した1 dataset familyを使用した。
- Local gridは`2.70, 3.15, 3.45, 3.75`とした。
- Localが残した`3.45, 3.75`に、Localで棄却された`3.15`をcontrolとして加えた。
- 同一snapshot、同一probe順、同一noise／timestep／dropout契約を使用した。
- 128 step、5 branch repeats、`common_skip`で測定した。
- snapshot A/B、prefix、source contract、Local/Trajectory共有probeはすべて
  `pass_exact`だった。
- 3候補ともhard-safetyを通過し、強制安全停止は発生しなかった。

この検証は`descriptive_only=true`、`recommendation_allowed=false`として
事前登録し、結果を見てから候補や判定規則を変えていない。

## 観測結果

### Local Tail

小さい順は次のとおりだった。

| mul | Local Tail |
|---:|---:|
| 3.75 | 0.323 |
| 3.45 | 0.488 |
| 3.15 | 0.612 |

Localでは`3.15`が`3.45`にBody/Tailの両方で劣る確率が0.8405となり、
事前規則により候補集合から外れた。一方、端点`3.75`が残ったため、
Localの結論自体も`edge_unresolved`であり、最良mulは宣言していない。

### 128-step Trajectory

Trajectory risk `T`は、候補の累積更新drift中央値をno-quant repeat間の
自然drift q95で割った記述的な比率である。小さい順は次のとおりだった。

| mul | T | 95% bootstrap区間 |
|---:|---:|---:|
| 3.15 | 0.771 | 0.709–1.124 |
| 3.75 | 0.818 | 0.764–1.229 |
| 3.45 | 0.833 | 0.803–1.168 |

Local TailとTrajectoryの順位相関は`-0.5`で、明確な順位反転が起きた。
Localで棄却された`3.15`のTが`3.45`より低いbootstrap確率は1.000、
`3.75`より低い確率は0.934だった。反対に`3.45`と`3.75`は判別不能だった。

5通りのrepeat leave-one-outでは、どのrepeatを除いてもT最小は`3.15`だった。
したがって、この反転は単一repeatだけの事故ではない。ただし、これは
`3.15`の最終画質が良いことを意味せず、TrajectoryがLocalとは別の側面を
安定して測ったことを意味する。

## 解釈

Local Body／Tailは、固定snapshot上で量子化が勾配をどれだけ変形するかを測る。
Trajectoryは、dropout有効の短期更新を128 step累積したとき、no-quantの更新軌道から
どれだけ離れたかを測る。optimizer、skip、勾配の相殺、経路依存が入るため、
Localの単純な延長ではない。

今回のように両者が逆順位になる以上、`max(Local, T)`のような一つのscoreへ
まとめたり、Tを使って候補を落としたりすると、有効かもしれないmulを誤って
除外する危険がある。

## 製品仕様への反映

- 通常利用のcustom dataset runnerはLocal-onlyを既定とする。
- 通常レポートの主表示はHard-safety、Body、Tail、Tail Amplification、
  source bootstrap、credible set、edge uncertaintyとする。
- Fidelity retainedはLocalに基づくbeta候補削減であり、best qualityではない。
- Trajectoryは公開`python -m dq_profile`入口では実行せず、低レベル研究protocolを
  明示的に選んだ場合だけ実行する。
- Trajectoryは候補削減、best mul、Utility、学習成功保証へ使用しない。
- 最終画質との対応は、将来の固定blind比較によるUtility Bridgeで別に検証する。

## 現時点で残る研究

- 独立profile seedでLocal候補集合とedge傾向のrun-level再現性を確認する。
- 異なるdataset familyでもLocal/Trajectory順位反転が起きるかを調べる。
- 40 epoch Utility Bridgeで、Localの数値的Safety/Fidelityと最終画質を接続する。

これらが未完了でも、Local-onlyのSafety/Fidelity説明診断は通常学習から分離した
beta機能として利用できる。
