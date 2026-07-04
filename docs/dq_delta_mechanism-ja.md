# dq\_delta フェイク量子化と range\_mul オートチューン（SDXL LoRA / fp16 向け）

## 概要

本ドキュメントは、SDXL の LoRA 学習（特に fp16 前提）で利用する **dq\_delta フェイク量子化**と、その挙動を安定させるための \*\*range\_mul 自動調整（オートチューン）\*\*の仕組みを説明する。

-   dq\_delta: 学習中のあるテンソル（delta あるいは z）に対して **擬似的な量子化ノイズ**を入れる仕組み
    
-   range\_mul: 量子化レンジ（range）を統計量（RMS など）に倍率を掛けて決める係数
    
-   オートチューン: clip\_rate（クリップ発生率）を監視し、range\_mul を自動で上下させて「狙った量子化状態」に寄せる制御
    
-   ウォームアップ: 学習序盤の統計・EMA の立ち上がりで制御が暴れないように、**統計更新はするが range\_mul 更新は止める**期間
    

本機構の狙いは次の通り。

-   dq\_delta の「当たり外れ」の主要因になりやすい **クリップ過多/過少**を可視化する
    
-   range\_mul を自動調整して、clip\_rate を一定レンジに保ち、学習を安定化する
    
-   人力の試行回数を減らし、データセット差に強いプリセットを作れる状態にする
    

* * *

## 前提と対象

### 対象

-   SDXL LoRA 学習（UNet / Text Encoder）
    
-   フェイク量子化（dq\_delta / dq\_quantize\_z）
    
### dq\_delta オプション一覧（基本）

※ `--dq_delta_log` / `--dq_delta_auto_range_mul` 系は [docs/dq_delta_autotune_spec-ja.md](docs/dq_delta_autotune_spec-ja.md) に記載。

| オプション | 説明 |
| --- | --- |
| `--dq_delta_step <float>` | 等間隔の刻み幅を直接指定（step モード）。 |
| `--dq_delta_bits <int>` | Nビットの等間隔量子化を模擬（推奨: 8）。`stat` と `range_mul` からスケール算出し、`[-Qmax, Qmax]` にクランプ（`Qmax=2^(N-1)-1`）。 |
| `--dq_delta_mode {det,stoch}` | `det`=最近傍、`stoch`=確率的丸め。 |
| `--dq_delta_begin <0-1>` | 学習進行率。この割合以降のみ有効化。 |
| `--dq_delta_begin_after_lr_warmup` | lrウォームアップ後に dq_delta を開始（`--dq_delta_begin` より優先）。例: `--lr_scheduler constant_with_warmup --lr_warmup_steps 1000` と併用し、**lr_warmup 後に量子化を開始して学習を安定化させる**狙い。 |
| `--dq_delta_scope {unet,te,both}` | 適用対象の限定（U-Netのみ/Text Encoderのみ/両方）。 |
| `--dq_delta_granularity {tensor,channel}` | 粒度（テンソル全体/チャネル別）。 |
| `--dq_delta_stat {rms,absmax,none}` | bits/step のスケール基準（per-channel時はチャネルごとに計算）。 |
| `--dq_delta_range_mul <float>` | bits モード×`stat=rms`時の有効レンジ倍率（range=倍率×RMS、既定3.0）。 |
| `--dq_delta_bits_sched 'p1:bits1,p2:bits2,...'` | 学習進行率 p でビット数を段階的に切替（例: `0.0:6,0.5:8,0.8:10`）。 |
| `--dq_quantize_z` | Δ ではなく `z=A(x)` を量子化（`B(Q(z))`）。rank r の z を対象に統計を取るため軽量化。 |


### 量子化モードの前提

本書の「オートチューン」は主に以下を前提にする。

-   `--dq_delta_bits` または `--dq_delta_bits_sched` を使う（bits モード）
    
-   `--dq_delta_stat=rms`（range\_mul が統計レンジ決定に効く）
    
-   `--dq_delta_mode stoch`（確率丸め）または `det`（決定的丸め）でも概念は同じ
    

* * *

## 用語と直感

### 1) フェイク量子化（Fake Quantization）

学習中のテンソル `x` を

1.  スケール `scale` と最大整数値 `qmax` を決める
    
2.  `q = x / scale` を整数表現へ
    
3.  `q` を `[-qmax, qmax]` にクランプ
    
4.  丸め（det か stoch）して整数化
    
5.  `x̂ = q * scale` として戻す（dequantize）
    

という一連の処理で「量子化されたように見せる」。

ポイント:

-   実際に重みを int 化して保持するのではなく、学習時に **ノイズ注入のように振る舞う**
    
-   その結果、過学習の仕方や収束のクセが変わることがある
    

### 2) range と step（刻み幅）

bits モードでは概ね

-   `qmax = 2^(bits-1) - 1`（例: bits=8 -> 127）
    
-   `range = range_mul * stat(x)`（stat が RMS の場合、RMS に倍率）
    
-   `scale = range / qmax`
    
-   量子化の刻み幅（step size）は `scale`
    

直感:

-   **bits を上げる**: `qmax` が増える -> `scale` が小さくなる -> 量子化ノイズが弱くなる（細かい）
    
-   **range\_mul を上げる**: `range` が増える -> `scale` が大きくなる -> 量子化ノイズが強くなる（粗くなる）が、クリップは減る
    
-   **range\_mul を下げる**: `range` が減る -> `scale` が小さくなる -> 量子化ノイズは弱くなる（細かい）が、クリップが増える
    

つまり **bits と range\_mul はトレードオフの別方向ノブ**である。

* * *

## 監視指標: clip\_rate とその意味

### clip\_rate（クリップ率）

「レンジを超えた要素がどれくらいあるか」を表す割合。

-   理想: クリップがゼロに近すぎない（range が広すぎると刻みが粗くなって薄まる傾向が出やすい）
    
-   しかし: クリップが多すぎると歪みが強くなり、破綻・不安定化が起きやすい
    

量子化の世界では、

-   **クリップ歪み（rangeが狭い）**
    
-   **量子化誤差（rangeが広い/刻みが粗い）**
    の合計が小さくなる点を狙う発想が多い。
    

本機構は、そのうち **クリップ側を制御して “いい感じのレンジ” に寄せる**。

### clip\_rate の数え方（実装上の要点）

`mode=stoch` の場合、丸めの揺らぎで `|q|==qmax` を誤検知しやすいことがある。
そのため、clip の判定は

-   **clamp 後（round 前）の `q_clamp` を用いて `|q_clamp| >= qmax` を数える**
    

のが安定しやすい。

利点:

-   追加の `|x| > range` マスク生成を避けられる
    
-   stoch の丸め揺れによる誤判定が減る
    

* * *

## オートチューンの全体像

### 目的

`clip_rate` を目標レンジに保つように `range_mul` を自動調整する。

`--dq_delta_auto_preset clip_rate_low_auto` では、最初は `clip_rate_low` と同じ低clip帯を狙う。学習中に `QuantErrRatioEMA >= 0.25` かつ `QErrPerClip >= 130` が連続し、lowを維持するコストが高いと判断した場合は、目標clip帯を `clip_rate_mid` へ一方向に切り替える。これは低clipが合うデータではlowを維持し、低clipが合わないデータでは誤差が深くなる前に中間帯へ逃がすための保険である。

### 基本ロジック（AutoStep ごと）

AutoStep のたびに

1.  `clip_rate_raw` を計測
    
2.  `clip_rate_ema` を更新（平滑化）
    
3.  EMA が目標レンジから外れていれば `range_mul` を少し上下
    
    （※`--dq_delta_auto_use_raw` 指定時は EMA と raw の両方が閾値を超えた場合のみ）
    

という負帰還（フィードバック）を回す。

### 更新式（例）

-   `clip_rate_ema = ema * clip_rate_ema + (1-ema) * clip_rate_raw`
    

制御:

-   `clip_rate_ema > clip_high` -> `range_mul *= mul_up`（レンジ拡大、クリップ減）
    
-   `clip_rate_ema < clip_low` -> `range_mul *= mul_down`（レンジ縮小、クリップ増）

-   `--dq_delta_auto_use_raw` 指定時は `clip_rate_raw` も同時に判定

-   その後 `range_mul` を `[auto_min, auto_max]` にクランプ

補助診断:

-   `QErrPerClip = QuantErrRatioEMA / max(ClipRateEMA, 0.001)`

-   clip率に対して量子化誤差が重いかを見る指標。`clip_rate_low_auto` の判定にも使う。

-   `--dq_delta_log_error_parts` を指定すると、`QuantErr` を `ClipErr` と `RoundErr` に分解してログに出す。第一版では判定には使わず、後から「clip由来かround由来か」を確認するための診断列として扱う。


### bits スケジュールとの関係

`--dq_delta_bits_sched` で bits が切り替わると、qmax と刻みが変わるため挙動が変化する。

推奨挙動:

-   bits 切替 step では `clip_rate_ema` をリセット（例: `clip_rate_ema = clip_rate_raw`）
    
-   同時に、後述の warmup もリセットする（切替直後の誤制御を避ける）
    

* * *

## ウォームアップ（制御のみ停止）

### 背景: なぜ必要か

学習の序盤は、分布が急速に変化することが多く、さらに EMA は立ち上がり直後に「低く出やすい」。

その結果、

-   `clip_rate_raw` は上がり始めている
    
-   しかし `clip_rate_ema` はまだ低い
    
-   `clip_low` を下回っている判定になり、`mul_down` が連発
    
-   `range_mul` が意図せず下がりすぎる
    

という事故が起きやすい。

### ウォームアップの方針

-   **統計と EMA は通常どおり更新する**
    
-   ただし **range\_mul の更新（mul\_up/mul\_down）だけ止める**
    

つまり、

-   観測はする
    
-   判断材料は貯める
    
-   しかし「ハンドルを切る」のは少し待つ
    

### ウォームアップ期間の決め方（決め打ち）

EMA の時間定数に基づく自動決定を採用する。

-   `warmup_updates = ceil(2 / (1 - auto_ema))`
    

例:

-   `auto_ema=0.95` -> `warmup_updates=40`
    

warmup は **AutoStep の回数**で数える。

-   `auto_every=50` の場合、40 回 -> 2000 step 相当の時間スケール
    

### 早期終了（推奨）

ウォームアップ中でも「もう安定している」ケースがあるため、固定の早期終了を入れると扱いやすい。

-   `clip_low <= clip_rate_ema <= clip_high` が **3 回連続**で成立したら warmup 終了
    

### bits 切替時の扱い

bits 切替時は量子化の性質が変わるので、

-   `clip_rate_ema` をリセット
    
-   warmup カウントもリセット
    

するのが安全。

* * *

## ログ設計（観測と制御のための CSV）

### LogStep と AutoStep

-   LogStep: `TrainStep % dq_delta_log_every == 0`
    
    -   full stats（RMS、absmax、scale min/mean/max、zero\_rate など）を出す
        
-   AutoStep: `TrainStep % dq_delta_auto_every == 0`
    
    -   原則、制御に必要な \*\*最小統計（clip\_rate など）\*\*を出す
        
    -   LogStep と重なる場合は full stats を流用し、重複計算を避ける
        

### DDP での集約

-   各モジュールでローカルに統計を加算
    
-   DDP の all-reduce は \*\*1回/AutoStep（またはLogStep）\*\*に集約する（オーバーヘッド低減）
    
-   range\_mul 更新は main process のみで実施し、必要なら broadcast で共有
    

### 主要列（summary）

例（実装に合わせて適宜）:

-   基本: `Epoch, TrainStep, Scope, Target, Bits, RangeMul, ...`
    
-   計測: `RMS, AbsMax, ScaleMin/Mean/Max, Qmax`
    
-   監視: `ClipRateRaw, ClipRateEMA, ZeroRate`
    
-   補助: `QuantErrRMSRaw/QuantErrRMSEMA`, `QuantErrRatioRaw/QuantErrRatioEMA`（LogStep のみ）
    
-   制御: `AutoApplied, RangeMulBefore, RangeMulAfter`
    
-   warmup を入れる場合（推奨）:
    
    -   `WarmupActive`（0/1）
        
    -   `WarmupRemain`（残り AutoStep 回数）
        
    -   `AutoReason`（warmup / clip\_high / clip\_low / in\_band など）
        
### “clipだけ見ない” 2指標ログ（実装済み）

-   `QuantErrRMSRaw = RMS(x - x_q)` で量子化誤差を直接観測する
    
-   `QuantErrRatioRaw = QuantErrRMSRaw / (RMS(x) + eps)` で相対化（`eps=1e-12`）
    
-   LogStep のみ計測し、dq_delta_log が有効な時だけ出力する
    
読み方の目安:

-   `ClipRate` が低いのに `QuantErrRatio` が高い → 刻みが粗く「薄まり」方向  
    対処: `--dq_delta_range_mul` を下げる（例: -0.1、まだ薄いなら -0.2）。`ClipRate` が上がり過ぎたら戻す。  
    併用案: `--dq_delta_bits` を上げる、または終盤だけ10にする（例: `--dq_delta_bits_sched "0.0:8,0.9:10"`）。
    
-   `QuantErrRatio` が低いのに `ClipRate` が高い → クリップ歪みが支配的  
    対処: `--dq_delta_range_mul` を上げる、または `--dq_delta_bits` を上げて刻みを細かくする。  
    auto を使う場合は `auto_clip_low/high` を少し低めにして過度なクリップを抑える。
    
補足: `QuantErrRatio` は `RMS(x)` が小さい局面で見かけ上大きく出ることがあるため、**単独では判断せず** `QuantErrRMS` の絶対量と必ず併せて判断する。

読み方の目安2:

-   `ClipRate` が低いのに `QuantErrRatio` が高い場合、次の2系統がありうる：
    
    1.  **レンジが広く刻みが粗い**（`range_mul` 高め / `bits` 低め）
        　→ 対策：`range_mul` を下げる、`bits` を上げる（または `bits_sched` を早める）
        
    2.  **外れ値が大きく、少数でも誤差エネルギーを支配**（`AbsMax >> Range`）
        　→ 対策：`range_mul` を上げる（または `clip_high` を下げて早めに拡張させる）、必要なら `bits` を上げる
        
-   見分け方：`AbsMax / Range` が極端に大きい場合は (2) の可能性が高い。


### auto\_only ログ（dq\_delta\_auto\_log\_file）

「ログ OFF でも Auto の挙動だけ追える」用途。
AutoStep のみを出す CSV。

最小列の例:

-   `TrainStep, Scope, Target, Bits, ClipRateRaw, ClipRateEMA, RangeMulBefore, RangeMulAfter, AutoApplied`
    
-   ＋ `WarmupActive, WarmupRemain, AutoReason`（強く推奨）
    

* * *

## 典型的な挙動パターン

### パターンA: 序盤に1回調整が入り、その後安定

-   初期 range\_mul がほぼ適正
    
-   学習初期に分布が少し変わって 1 回だけ補正
    
-   以後は `clip_rate_ema` が目標帯域内で推移
    
-   `AutoApplied=0` が続く
    

この場合、データと設定の相性が良い可能性が高い。

### パターンB: 序盤が暴れて mul が連続更新

-   学習の立ち上がりで分布が急変
    
-   `clip_rate_raw` のブレが大きい
    
-   EMA が追いつくまで、閾値を跨いで更新が続きやすい
    

対策:

-   warmup を入れる（本書の結論）
    
-   `mul_up/mul_down` を小さくして更新を滑らかにする（例: 1.01 / 0.995）
    
-   目標帯域を用途に合わせて見直す（キャラ用途は「少しクリップ多め」の方が好みになる場合がある）
    

### パターンC: bits 切替が入っても clip\_rate があまり変わらない

これは起こり得る。理由は複合的で、

-   `stat=rms` と `range_mul` によってレンジが分布に追従している
    
-   granularity=channel でチャネルごとのスケールが効いている
    
-   「クリップ率」だけを見ると、bits 変更の差が見えにくいケースがある
    

ただし、bits は **クリップではなく “刻み幅” を変えるノブ**なので、
clip\_rate が似ていても量子化誤差（刻みの粗さ）は変わり得る点に注意。

QuantErrRatio を見ると差が出やすい。

* * *

## パラメータの考え方（bits と range\_mul）

### bits の役割

-   低い bits: ノイズ強め、特徴が薄まる（あるいは抽象化される）ことがある
    
-   高い bits: ノイズ弱め、dq\_delta の効果が薄くなる場合がある
    

キャラクター LoRA では体感的に

-   **bit=8 が実用中心**
    
-   bit=6 は薄まりやすい（用途次第）
    
-   bit=10 は「量子化が弱くなり過ぎて意味が薄い」寄りになりやすい
    

### range\_mul の役割

-   range\_mul を上げる: クリップは減るが刻みが粗くなる（薄まり方向に働く場合）
    
-   range\_mul を下げる: 刻みが細かくなるがクリップは増える（歪みや破綻が増える場合）
    

「キャラが薄い」と感じた場合、候補は次のどれか。

-   range\_mul が高めで刻みが粗い（薄まり）
    
-   bits が高めで dq\_delta の効果が弱い（差が出にくい）
    
-   目標 clip 帯域が低すぎて、結果的に range\_mul が上がりやすい設定になっている
    

* * *

## 実用プリセット例（キャラクター用途）

以下は “キャラ寄り” の実用を意図した一例。環境に合わせて調整する。

### 推奨ベース（例）

bash

コードをコピーする

`--dq_delta_bits 8 --dq_delta_bits_sched "0.0:8,0.9:10" --dq_delta_granularity channel --dq_delta_stat rms --dq_delta_mode stoch --dq_delta_scope unet --dq_delta_begin 0 --dq_delta_range_mul 3.0  --dq_delta_log --dq_delta_auto_range_mul --dq_delta_auto_every 50 --dq_delta_auto_ema 0.95  # キャラ用途で “ややクリップ許容” の一例 --dq_delta_auto_clip_low 0.003 --dq_delta_auto_clip_high 0.005 --dq_delta_auto_mul_up 1.01 --dq_delta_auto_mul_down 0.995  # ウォームアップ（実装したオプション名に合わせて） --dq_delta_auto_warmup`

意図:

-   序盤は warmup で暴れを抑える
    
-   その後は clip\_rate\_ema が 0.003 から 0.005 に入るように range\_mul を微調整
    
-   更新率を小さくして “じわじわ追従” にする（急に range\_mul が飛ばない）
    

* * *

## 性能とオーバーヘッドの見積もり（ざっくり）

### まず大前提

-   dq\_delta を入れた時点で処理が重くなる（量子化計算が乗る）
    
-   その上でのオートチューンは、設計次第で **誤差程度**に抑えられる
    

### オートチューンのコスト源

-   AutoStep ごとの `clip_count` と `numel` の集計（reduce）
    
-   DDP の all-reduce（1回/AutoStep にまとめるのが重要）
    

対策として本機構が採用している方針:

-   LogStep の full stats を AutoStep に流用
    
-   ログ OFF 時は最小統計のみ
    
-   all-reduce を 1回/AutoStep に集約
    

* * *

## 既知の課題と改善アイデア（今後の発展）

### 1) ターゲット clip 帯域のスケジュール

キャラ用途では「少しクリップ多めが良い」ことがある一方、
安定性重視用途では「クリップ少なめ」が良いことがある。

-   序盤: 低め（安定寄り）
    
-   中盤: やや高め（特徴寄り）
    
-   終盤: 少し下げて整える
    

のように、目的に応じて「狙い帯域」をスケジュールするのも考えられる。

### 2) scope別 range\_mul（UNet と TE を分離）

UNet と TE で分布が違う場合があるので、

-   UNet 用 range\_mul
    
-   TE 用 range\_mul
    

を別々に持つと、より狙い通りに寄せやすい。

* * *

## まとめ

-   dq\_delta のフェイク量子化は、当たると良いが、外れると逆効果になり得る
    
-   当たり外れの大きな要因は「クリップ過多/過少」と「刻みの粗さ」
    
-   そこで clip\_rate を監視し、range\_mul を自動で調整することで安定性が上がる
    
-   学習序盤は EMA が立ち上がり途中で誤制御しやすいので、**制御のみ停止するウォームアップ**が有効
    
-   ログ（clip\_rate\_raw/ema、quant\_err、range\_mul、auto\_applied、warmup情報）を残すことで、学習の挙動差を定量的に比較できる
    

* * *

## 付録: よくある質問

### Q. bits と range\_mul のどちらを先に触るべき？

-   まずは bits を「目的の強さ」に合わせる（キャラなら 8 を基点に）
    
-   次に range\_mul（固定 or auto）で clip 帯域に合わせる
    

### Q. クリップをわざと増やすのはアリ？

アリになり得るが、攻め方。
「クリップによる歪み」と「刻みの粗さ」のバランスが変わり、良い方向に出ることもあれば破綻も増える。
やるなら、ログで clip\_rate を見ながら、狙い帯域を用途別に設計
