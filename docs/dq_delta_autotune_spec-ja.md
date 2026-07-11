# dq_delta オートチューナ（方式A）仕様

本ドキュメントは、`--dq_delta_step / --dq_delta_bits` のフェイク量子化に対して、
「ログ追加」と「range_mul のフィードバック制御（自動調律）」を導入するための仕様です。

## 目的

- dq_delta の当たり外れ要因（特に clip 過多/過少）をログで可視化する
- `range_mul` を自動調整して、clip_rate を目標レンジに保つ
- SDXL/LoRA 学習の安定性を上げつつ、手動試行を減らす

## 対象範囲

- 対象は **フェイク量子化（dq_delta）** のみ
- `--dq_delta_bits` / `--dq_delta_bits_sched` を中心に設計（`--dq_delta_step` はログのみ）
- `--dq_quantize_z`（z 量子化）にも対応

## 追加ログ仕様

### 有効化オプション

- `--dq_delta_log` : dq_delta ログを有効化（デフォルト無効）
- `--dq_delta_log_every <int>` : ログ間隔（optimizer step 単位、デフォルト 100）
- `--dq_delta_log_scope {unet,te,both}` : 計測対象（未指定時は `dq_delta_scope` を継承）
- `--dq_delta_log_mode {summary,per_module}` : 出力粒度（デフォルト summary）
- `--dq_delta_log_detail {basic,full}` : summaryログの詳細度（デフォルト basic）。`basic` は常用向けに `ZeroRate`, `AbsMax`, `Range`, `ScaleMin/Mean/Max` を省略し、`full` は従来相当の詳細診断列を出力する。`per_module` は詳細診断扱い。
- `--dq_delta_log_file <path>` : 省略時は `--output_dir/dq_delta_logs+<output_name>.txt`
- `--dq_delta_log_extra {near_zero_rate}` : 追加ログ項目（デフォルト無効）
- `--dq_delta_log_error_parts` は削除。clip/round成分分解は常用診断から外し、新規ログでは出力しない。

### 計測ポイント

- 量子化の**直前**に計測（delta または z）
  - `--dq_quantize_z` 指定時は **z** を対象
  - それ以外は **delta** を対象
- bits モード/step モードともに計測可能
- DDP/複数プロセス時は **統計を all-reduce** してから summary を算出する
- ログ書き込みは **main process のみ**（または rank suffix 付きの別ファイル）
- `dq_delta_log` が無効な場合は **auto 用の最小統計**のみを計測する（後述）
- `dq_delta_log` が有効な場合は、**LogStep の統計を auto 判定に流用**し、重複計算は行わない
- **LogStep**: `TrainStep % dq_delta_log_every == 0`
- **AutoStep**: `TrainStep % dq_delta_auto_every == 0`
- LogStep と AutoStep が重なる場合は **full stats を流用**する
- 統計は各モジュールでローカルに加算し、DDP all-reduce は **reduce 種別ごとに各1回/AutoStep（またはLogStep）**にまとめる  
  例: sum 系を1回、max 系を1回、min 系を1回

### ログ項目（summary/basic）

- `epoch,TrainStep,scope,target,bits,DQStepSize,range_mul,stat,granularity,mode`
- `rms`
- `clip_rate_raw` : 今回計測した clip_rate
- `clip_rate_ema` : EMA 平滑化した clip_rate（制御に使用）
- `quant_err_rms_raw` : `RMS(x - x_q)`（フェイク量子化の誤差）
- `quant_err_rms_ema` : `quant_err_rms_raw` の EMA（係数は `dq_delta_auto_ema`）
- `quant_err_ratio_raw` : `quant_err_rms_raw / (rms + eps)`（`eps=1e-12`）
- `quant_err_ratio_ema` : `quant_err_ratio_raw` の EMA（係数は `dq_delta_auto_ema`）
  - EMA は scope 共通の 1 系列。`log_scope=both` の場合は unet+te 合算で更新。
- `active_clip_band` : 現在の目標clip帯（`default` / `low` / `mid` / `high` / `high_narrow` / `custom`）。
- `active_clip_low,active_clip_high` : その時点でauto制御が使っている目標clip帯。`clip_rate_low_auto` では途中で `low` から `mid` に変わる。
- `clip_rate_low_auto_state,clip_rate_low_auto_bad,clip_rate_low_auto_bad_streak` : `clip_rate_low_auto` 用の状態とbad判定。`clip_rate_low_auto` 以外では空欄。
- 旧ログ互換: 古いログに `clip_err_rms,round_err_rms,clip_err_ratio,round_err_ratio,clip_share,round_share` が含まれる場合、診断レポート側では読み取り可能。ただし新規ログでは出力しない。
- `numel` : 計測対象の要素数（集計値）
- `auto_applied` : AutoStep で range_mul 更新が適用された場合は 1、それ以外は 0
- `range_mul_before,range_mul_after` : AutoStep 以外は **同値で埋める**（`auto_applied=0`）
- `warmup_active` : warmup 中は 1、それ以外は 0
- `warmup_remain` : warmup の残り AutoStep 回数（0 なら warmup 終了）
- `auto_reason` : `warmup` / `clip_high` / `clip_low` / `in_band`
- `auto_init_mul_applied` : 初期 range_mul の自動上書きが行われた場合は 1
- `auto_init_mul_value` : 自動算出された初期 range_mul
- `auto_init_clip_target` : `(clip_low + clip_high) / 2`

### ログ項目（summary/full）

`summary/full` は `summary/basic` の全項目に、次の詳細診断列を追加する。

- `absmax,range,scale_min,scale_mean,scale_max,qmax`
- `zero_rate` : 量子化後に 0 になった割合
- 任意: `near_zero_rate` : `--dq_delta_log_extra near_zero_rate` 指定時の `|x| < 0.5*scale` の割合

※ `range = scale * qmax`、`scale_*` は bits モード時に有効。step モードは `range/scale_*/qmax` を空欄扱い。  
※ `granularity=channel` の場合、`scale_*` は `min/mean/max` を記録する（tensor の場合は同値）。

### ログ項目（per_module）

- summary に加え `module_name,shape` を出力
- 1行 = 1モジュールの計測
- DDP 時は **main rank のローカル値のみ**を出力（per_module は all-reduce しない）
- quant_error 系（`quant_err_*`）も summary と同じ列で出力

### 集計定義（summary）

- `numel_sum` : 計測対象の全要素数
- `sumsq_sum` : `x^2` の総和（fp32 で集計）
- `xq_sumsq_sum` : `x_q^2` の総和（fp32 で集計）
- `xxq_sum` : `x * x_q` の総和（fp32 で集計）
- `clip_count_sum` : クリップに該当した要素数（`|x| > range_elem` 相当）
- `zero_count_sum` : 量子化後に 0 になった要素数
- `rms = sqrt(sumsq_sum / numel_sum)`
- `absmax = max(|x|)`（全要素の最大値）
- `clip_rate = clip_count_sum / numel_sum`
- `zero_rate = zero_count_sum / numel_sum`
- `quant_err_rms = sqrt((sumsq_sum + xq_sumsq_sum - 2*xxq_sum) / numel_sum)`
- `quant_err_ratio = quant_err_rms / (rms + eps)`（`eps=1e-12`）
- `qerr_per_clip = quant_err_ratio_ema / max(clip_rate_ema, dq_delta_qerr_per_clip_floor)`
- clip/round成分分解（`clip_err_rms`, `round_err_rms`, `clip_share`, `round_share`）は旧ログ互換項目。新規ログでは出力しない。

※ `range_elem` は `granularity=channel` の場合はチャネル別 range をブロードキャストした per-element range を使う。  
※ 分散学習時は `numel_sum/sumsq_sum/clip_count_sum/zero_count_sum` を **all-reduce (sum)** してから算出する。  
※ `absmax` は **MAX reduce**、`scale_min` は **MIN reduce**、`scale_max` は **MAX reduce** とする。  
※ `scale_mean` は各rankの `scale_sum` と `scale_count` を all-reduce (sum) し、`scale_sum / scale_count` で算出する。
※ `clip_count_sum` は `|x| > range` の代わりに、**clamp 後（round 前）の q_clamp で |q_clamp| >= Qmax を数える**ことを推奨（低負荷）。  
  量子化で既に計算している q_clamp を利用し、追加で x.abs() や range_elem を生成して比較しない。  
  `mode=det` の場合に限り、round 後の q で近似してもよい。  

### Auto 用統計（低負荷）

- 通常autoの判定に必要なのは **clip_rate** のみなので、`dq_delta_log` が無効な場合は以下に限定する。
  - `numel_sum`
  - `clip_count_sum`
  - （任意）`zero_count_sum`
- **LogStep のみ**で **full stats**（`sumsq_sum/absmax/scale_*` など）を計測する。
  LogStep 以外の AutoStep は最小統計に限定する。
- `quant_err_*` は full stats の一部として **LogStep のみ**で計算する。
- 例外として `--dq_delta_auto_preset clip_rate_low_auto` では、AutoStep の判定に `QuantErrRatioEMA` / `QErrPerClip` を使うため、AutoStep でも full stats を計測する。
- clip/round 分解は常用診断から外し、`clip_rate_low_auto` の判定には使わない。

### 出力例（summary/basic）

```text
Epoch,TrainStep,Scope,Target,Bits,DQStepSize,RangeMul,Stat,Granularity,Mode,RMS,ClipRateRaw,ClipRateEMA,QuantErrRMSRaw,QuantErrRMSEMA,QuantErrRatioRaw,QuantErrRatioEMA,ActiveClipBand,ActiveClipLow,ActiveClipHigh,ClipRateLowAutoState,ClipRateLowAutoBad,ClipRateLowAutoBadStreak,TrainProgress,ClipRateLowAutoMinProgress,ClipRateLowAutoFreezeProgress,ClipRateLowAutoThresholdQErrRatio,ClipRateLowAutoThresholdQErrPerClip,ClipRateLowAutoPhase,Numel,AutoApplied,RangeMulBefore,RangeMulAfter,WarmupActive,WarmupRemain,AutoReason,AutoInitMulApplied,AutoInitMulValue,AutoInitClipTarget
```

### 出力例（summary/full）

```
Epoch,TrainStep,Scope,Target,Bits,DQStepSize,RangeMul,Stat,Granularity,Mode,RMS,AbsMax,Range,ScaleMin,ScaleMean,ScaleMax,Qmax,ClipRateRaw,ClipRateEMA,ZeroRate,QuantErrRMSRaw,QuantErrRMSEMA,QuantErrRatioRaw,QuantErrRatioEMA,ActiveClipBand,ActiveClipLow,ActiveClipHigh,ClipRateLowAutoState,ClipRateLowAutoBad,ClipRateLowAutoBadStreak,Numel,AutoApplied,RangeMulBefore,RangeMulAfter,WarmupActive,WarmupRemain,AutoReason,AutoInitMulApplied,AutoInitMulValue,AutoInitClipTarget
2,3400,unet,delta,8,,3.0,rms,channel,stoch,0.0123,0.0912,0.0369,0.00020,0.00029,0.00041,127,0.0008,0.0007,0.034,0.0015,0.0014,0.12,0.11,low,0.0005,0.0022,keep_low,0,0,12345678,1,3.0,3.21,0,0,clip_high,1,3.0,0.004
```

## ログの見方（初心者向け）

### logs（`--dq_delta_log`）: summary/basic と summary/full

以下の表は両形式をまとめたもの。`AbsMax`, `Range`, `ScaleMin/Mean/Max`, `Qmax`, `ZeroRate`, `NearZeroRate` は `full` のみで、`basic` には列自体が存在しない。

| 項目名 | 説明 | 読み取り方 |
| --- | --- | --- |
| Epoch | エポック番号 | 学習の周回数。進行の目安。 |
| TrainStep | optimizer step | ログのタイミング。100なら100回更新後。 |
| Scope | 対象範囲 | `unet`/`te`/`both`。どこを測っているか。 |
| Target | 量子化対象 | `delta` または `z`。 |
| Bits | 現在のビット数 | `bits_sched` 適用中の値。 |
| DQStepSize | step モードの刻み | bitsモードなら空欄。 |
| RangeMul | range_mul | 大きいほどレンジが広い。 |
| Stat | 統計方式 | `rms/absmax/none`。 |
| Granularity | 粒度 | `tensor`/`channel`。 |
| Mode | 丸め方式 | `det`/`stoch`。 |
| RMS | 入力のRMS | 大きいほどΔが大きい。 |
| AbsMax | 入力の最大絶対値（fullのみ） | 外れ値の指標。 |
| Range | 有効レンジ（fullのみ） | `ScaleMean * Qmax` の目安。 |
| ScaleMin | scale最小（fullのみ） | チャネル差の下側。 |
| ScaleMean | scale平均（fullのみ） | Range算出に使う中心値。 |
| ScaleMax | scale最大（fullのみ） | チャネル差の上側。 |
| Qmax | 量子化上限（fullのみ） | 6bitなら31など。 |
| ClipRateRaw | 生のクリップ率 | 目標帯域に収まるか確認。 |
| ClipRateEMA | EMA平滑値 | auto判定に使う値。 |
| ZeroRate | 量子化後0の割合（fullのみ） | 潰れの兆候。 |
| QuantErrRMSRaw | 量子化誤差のRMS | 「クリップは少ないが刻みが粗い」を検出。 |
| QuantErrRMSEMA | QuantErrRMS のEMA | ノイズの安定した傾向を見る。 |
| QuantErrRatioRaw | 誤差RMS/入力RMS | 入力に対する誤差の割合。 |
| QuantErrRatioEMA | QuantErrRatio のEMA | 比率の安定した傾向を見る。 |
| NearZeroRate | 0近傍の割合（fullのみ） | `--dq_delta_log_extra near_zero_rate`時のみ。 |
| ActiveClipBand | 現在の目標clip帯 | `clip_rate_low_auto` では `low` から `mid` へ切り替わる。通常presetでも比較用に出力。 |
| ActiveClipLow/High | 現在のclip帯下限/上限 | autoがどの帯を狙っているか確認する。 |
| ClipRateLowAutoState | clip_rate_low_auto 状態 | `observe`/`keep_low`/`escape_to_mid`/`mid_lock`/`frozen`。clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoBad | clip_rate_low_auto bad判定 | `QuantErrRatioEMA>=0.25` かつ `QErrPerClip>=130` の時 1。clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoBadStreak | bad連続回数 | 既定では3回連続で `mid` へ逃がす。clip_rate_low_auto以外は空欄。 |
| TrainProgress | 学習進捗 | `global_step / max_train_steps`。min/freeze境界を後から確認するための列。 |
| ClipRateLowAutoMinProgress | low_auto判定開始進捗 | `--dq_delta_clip_rate_low_auto_min_progress` の値。clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoFreezeProgress | low_auto切替凍結進捗 | `--dq_delta_clip_rate_low_auto_freeze_progress` の値。clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoThresholdQErrRatio | low_auto QuantErrRatio閾値 | `--dq_delta_clip_rate_low_auto_qerr_ratio` の値。clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoThresholdQErrPerClip | low_auto QErrPerClip閾値 | `--dq_delta_clip_rate_low_auto_qerr_per_clip` の値。clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoPhase | low_auto判定フェーズ | `warmup`/`pre_min_progress`/`active`/`frozen`/`escaped`。clip_rate_low_auto以外は空欄。 |
| ClipErrRMS/RoundErrRMS | clip/round誤差RMS | 旧ログ互換項目。新規ログでは出力しない。 |
| ClipErrRatio/RoundErrRatio | 各誤差RMS/RMS | 旧ログ互換項目。新規ログでは出力しない。 |
| ClipShare/RoundShare | 全QuantErr中のclip/round寄与 | lowを守るためにround誤差が支配的になっていないか確認する。 |
| Numel | 要素数 | 統計の母数。 |
| AutoApplied | auto適用 | 1ならrange_mulが変化。 |
| RangeMulBefore | 変更前 | auto適用時の前値。 |
| RangeMulAfter | 変更後 | auto適用時の後値。 |
| WarmupActive | warmup中 | 1なら range_mul は固定。 |
| WarmupRemain | warmup残り | 0なら warmup終了。 |
| AutoReason | 判定理由 | `warmup`/`clip_high`/`clip_low`/`in_band`。 |
| AutoInitMulApplied | 初期化適用 | 1なら初期 range_mul 上書きが行われた。 |
| AutoInitMulValue | 初期 range_mul | 自動算出された初期値（適用時）。 |
| AutoInitClipTarget | 初期 clip_target | `(clip_low+clip_high)/2`。 |
| QErrPerClip | QuantErrRatioEMA / max(ClipRateEMA, floor) | autoログ側に記録する low_auto 判定用の診断値。通常の `dq_delta_logs` には出力しない。 |
| ActiveClipBand/Low/High | 現在のclip目標帯 | clip_rate_low_autoのescape前後の目標帯を追う。 |
| ClipRateLowAutoState | clip_rate_low_auto 状態 | clip_rate_low_auto以外は空欄。 |
| ClipRateLowAutoDecision | clip_rate_low_auto判断 | `observe`/`keep_low`/`bad_count`/`escape_to_mid`/`mid_lock`/`frozen`。 |
| ClipRateLowAutoReason | clip_rate_low_auto判断理由 | `min_progress`/`low_bad`/`bad_streak_met`/`escaped_once`/`freeze_progress`/`insufficient_qerr_stats`。 |
| ClipRateLowAutoBad/ClipRateLowAutoBadStreak | bad判定と連続回数 | `mid` に逃がした根拠を追う。 |
| TrainProgress | 学習進捗 | `global_step / max_train_steps`。 |
| ClipRateLowAutoMinProgress/ClipRateLowAutoFreezeProgress | low_auto進捗閾値 | min/freeze境界をログ単体から復元する。 |
| ClipRateLowAutoThresholdQErrRatio/ClipRateLowAutoThresholdQErrPerClip | low_auto bad判定閾値 | CLIで指定した閾値をログ単体から復元する。 |
| ClipRateLowAutoPhase | low_auto判定フェーズ | `warmup`/`pre_min_progress`/`active`/`frozen`/`escaped`。 |
| ClipRateLowAutoCanEscape | low_auto切替可能状態 | min_progress以降、freeze_progress前、未escapeなら1。minimal autoログのみ。 |
補足（読み解きの目安）:
- `ClipRate` が低いのに `QuantErrRatio` が高い場合、**レンジが広く刻みが粗い**可能性がある（`range_mul` 高め / `bits` 低め）。
- `QuantErrRatio` が低いのに `ClipRate` が高い場合、**クリップ歪み**が支配的な可能性がある。
- Rank 系の指標は `dq_delta_log` には含まれません。必要な場合は `--rank_log` を併用し、`rank_logs+...txt` を参照してください。

### logs（`--dq_delta_log`）: per_module

| 項目名 | 説明 | 読み取り方 |
| --- | --- | --- |
| Module | LoRAモジュール名 | どの層の統計か。 |
| Shape | テンソル形状 | 層の規模感の目安。 |

※ per_module は上記 summary の列に `Module/Shape` が追加されます。

### auto（`--dq_delta_auto_log_file`）: minimal

| 項目名 | 説明 | 読み取り方 |
| --- | --- | --- |
| TrainStep | optimizer step | 自動調整の判定タイミング。 |
| Scope | 対象範囲 | `unet`/`te`/`both`。 |
| Target | 量子化対象 | `delta`/`z`。 |
| Bits | 現在のビット数 | スケジュールに依存。 |
| ClipRateRaw | 生のクリップ率 | 目標帯域に収まるか確認。 |
| ClipRateEMA | EMA平滑値 | auto判定値。 |
| RangeMulBefore | 変更前 | auto適用時の前値。 |
| RangeMulAfter | 変更後 | auto適用時の後値。 |
| AutoApplied | auto適用 | 1ならrange_mulが変化。 |
| WarmupActive | warmup中 | 1なら range_mul は固定。 |
| WarmupRemain | warmup残り | 0なら warmup終了。 |
| AutoReason | 判定理由 | `warmup`/`clip_high`/`clip_low`/`in_band`。 |
| AutoInitMulApplied | 初期化適用 | 1なら初期 range_mul 上書きが行われた。 |
| AutoInitMulValue | 初期 range_mul | 自動算出された初期値（適用時）。 |
| AutoInitClipTarget | 初期 clip_target | `(clip_low+clip_high)/2`。 |

### auto（`--dq_delta_auto_log_file`）: full_schema

full_schema は **logs（summary）と同じ列構成**で出力します。  
LogStep 以外の列は空欄（NA）で、追加統計は計算しません。
## range_mul フィードバック制御仕様

### 有効化オプション

- `--dq_delta_auto_range_mul` : range_mul の自動調整を有効化（デフォルト無効）
- `--dq_delta_auto_preset {default,clip_rate_high,clip_rate_high_narrow,clip_rate_mid,clip_rate_low,clip_rate_low_auto}` : auto range_mul のプリセット（指定時は clip_low/high のみ上書き。`clip_rate_low_auto` は途中でbandを切替）
- `--dq_delta_auto_every <int>` : 調整間隔（optimizer step 単位、デフォルト 50）
- `--dq_delta_auto_clip_low <float>` : clip_rate 下限（デフォルト 0.0005 = 0.05%）
- `--dq_delta_auto_clip_high <float>` : clip_rate 上限（デフォルト 0.003 = 0.3%）
- `--dq_delta_auto_mul_up <float>` : 上げ係数（デフォルト 1.01）
- `--dq_delta_auto_mul_down <float>` : 下げ係数（デフォルト 0.995）
- `--dq_delta_auto_min <float>` : range_mul 下限（デフォルト 1.0）
- `--dq_delta_auto_max <float>` : range_mul 上限（デフォルト 6.0）
- `--dq_delta_auto_ema <float>` : clip_rate の EMA 係数（デフォルト 0.95）
- `--dq_delta_auto_use_raw` : auto 判定に clip_rate_raw も使う（既定 OFF）。mul の変化をよりなだらかにする。
- `--dq_delta_auto_init_range_mul_from_band` : clip帯中心から range_mul 初期値を自動算出（auto有効時、`stat=rms` のみ）
- `--dq_delta_auto_warmup` / `--no-dq_delta_auto_warmup` : warmup 期間は range_mul を変更しない（auto 有効時のみ、既定 ON）
- `--dq_delta_auto_warmup_updates <int>` : warmup 回数の上書き（0=内部デフォルト）
- `--dq_delta_auto_log_file <path>` : 省略時は `--output_dir/dq_delta_auto+<output_name>.txt`（auto イベントのみ記録）
- `--dq_delta_auto_log_format {minimal,full_schema}` : auto ログの列形式（デフォルト minimal）
- `--dq_delta_clip_rate_low_auto_min_progress <float>` : `clip_rate_low_auto` の判定開始進捗（デフォルト 0.25）
- `--dq_delta_clip_rate_low_auto_bad_streak <int>` : `clip_rate_low_auto` が `mid` へ逃がすまでのbad連続回数（デフォルト 3）
- `--dq_delta_clip_rate_low_auto_freeze_progress <float>` : この学習進捗以降は `clip_rate_low_auto` のband切替を凍結（デフォルト 0.55）
- `--dq_delta_clip_rate_low_auto_qerr_ratio <float>` : `clip_rate_low_auto` bad判定の `QuantErrRatioEMA` 閾値（デフォルト 0.25）
- `--dq_delta_clip_rate_low_auto_qerr_per_clip <float>` : `clip_rate_low_auto` bad判定の `QErrPerClip` 閾値（デフォルト 130）
- `--dq_delta_qerr_per_clip_floor <float>` : `QErrPerClip` 計算時の `ClipRateEMA` 下限（デフォルト 0.001）

### auto プリセット一覧

※ preset は clip_low/high のみ切替。mul_up/mul_down は `--dq_delta_auto_mul_up/down` に従う（既定 1.01 / 0.995）。
※ `--dq_delta_auto_use_raw` 指定時は変化がなだらかになり、既存 preset では mul があまり動かないことがあるため、mul を積極的に動かしたい用途向けに `clip_rate_high_narrow` / `clip_rate_low` を追加。

| preset | clip_low | clip_high | mul_up | mul_down | 目的 |
| --- | --- | --- | --- | --- | --- |
| `default` | 0.0005 | 0.003 | args | args | 安全汎用。特徴が薄まりやすい。 |
| `clip_rate_high` | 0.003 | 0.005 | args | args | キャラクター学習向け。clip_rate 高めを狙う。 |
| `clip_rate_high_narrow` | 0.0038 | 0.0048 | args | args | キャラクター学習向け2。狭い範囲を狙う。 |
| `clip_rate_mid` | 0.002 | 0.004 | args | args | 中間帯の clip_rate を狙う。 |
| `clip_rate_low` | 0.0005 | 0.0022 | args | args | defaultより安定方向に振る。 |
| `clip_rate_low_auto` | 開始時 0.0005 | 開始時 0.0022 | args | args | low帯で開始し、low維持コストが高いと判断したら `clip_rate_mid` 帯へ一方向に逃がす。 |

### `clip_rate_low_auto` プリセット

`clip_rate_low` は低clipを狙うため、外れ値の影響を抑えられるデータでは有効なことがある。一方で、データによっては low帯を維持するために有効レンジを広げる必要があり、8bit量子化の刻みが粗くなって `QuantErr` が増えることがある。

`clip_rate_low_auto` はこの「lowを維持するコストが高い」ケースを本番ラン中に検出し、low帯から中間帯へ逃がす保険として用意する。highへは自動で移動せず、第一版では `low -> mid` の一方向のみ。

判定はシンプルに以下を使う。

```
QErrPerClip = QuantErrRatioEMA / max(ClipRateEMA, dq_delta_qerr_per_clip_floor)

low_bad =
  QuantErrRatioEMA >= dq_delta_clip_rate_low_auto_qerr_ratio
  and QErrPerClip >= dq_delta_clip_rate_low_auto_qerr_per_clip
```

既定値は `QuantErrRatioEMA >= 0.25` かつ `QErrPerClip >= 130`。`dq_delta_clip_rate_low_auto_min_progress` 経過後に `low_bad` が `dq_delta_clip_rate_low_auto_bad_streak` 回連続したら、目標bandを `clip_rate_mid`（0.002〜0.004）へ切り替える。一度 `mid` に逃げた後は `low` へ戻さない。`dq_delta_clip_rate_low_auto_freeze_progress` 以降は新規のband切替を行わない。

#### 既定閾値の前提

`clip_rate_low_auto` の既定閾値（`QuantErrRatioEMA=0.25`, `QErrPerClip=130`）は、主に `--dq_quantize_z` を使わない delta 量子化、`--dq_delta_mode stoch`、`8bit/channel/rms` のログを基準にした経験的な値。

`--dq_quantize_z` や `--dq_delta_mode det` を使う場合、`ZeroRate` や `QErrPerClip` の分布が変わるため、同じ閾値は保守的な警告として扱う。特に z 量子化では `QErrPerClip` が高めに出ることがあり、`clip_rate_low_auto` が `mid` へ逃がす挙動は安全側に働く可能性がある。

`QErrPerClip` は autoログ側の診断列として出力される。`dq_delta_logs` 側には `QuantErrRatioEMA` と `ClipRateEMA` を残すため、必要なら後段ツールで同等の比率を再計算できる。

### 発動条件（重要）

フィードバック制御は **以下を満たす時のみ** 発動する。

- `--dq_delta_bits` または `--dq_delta_bits_sched` が有効
- `--dq_delta_stat=rms`（range_mul が有効に使われるため）
- `dq_delta_begin` を通過後（量子化が有効化された後）

以下の場合は **自動調整を無効化** し、警告ログのみ出す。

- `--dq_delta_step` のみ（bits モード不使用）
- `--dq_delta_stat=absmax` または `none`

### 制御ロジック

- 各 `auto_every` optimizer step で clip_rate を集計（summary）
- EMA で平滑化し、既定は **EMA のみ**で判定
- `--dq_delta_auto_use_raw` 指定時は **EMA と raw の両方**が閾値を超えた場合のみ更新

```
if use_raw:
    if clip_rate_ema > clip_high and clip_rate_raw > clip_high: range_mul *= mul_up
    elif clip_rate_ema < clip_low and clip_rate_raw < clip_low: range_mul *= mul_down
else:
    if clip_rate_ema > clip_high: range_mul *= mul_up
    elif clip_rate_ema < clip_low: range_mul *= mul_down
range_mul = clamp(range_mul, auto_min, auto_max)
```

- 更新後の range_mul は **dq_delta_scope に含まれる全モジュールに反映**
- `--dq_quantize_z` 使用時も同様（対象テンソルが z になるだけ）
- `--dq_delta_bits_sched` 使用時は **現在の bits** に対して range_mul を調整
- `--dq_delta_bits_sched` で bits が切り替わった step では **clip_rate_ema をリセット**（`clip_rate_ema = clip_rate`）
- bits 切替時は warmup もリセット（残り回数と連続 in-band 回数を初期化）
- DDP/複数プロセス時は **main process のみ**が range_mul を更新し、**broadcast** で共有する

### 初期 range_mul の自動算出（任意）

`--dq_delta_auto_init_range_mul_from_band` を指定すると、学習開始前に **clip帯中心**から range_mul を機械的に初期化する。

- `clip_target = (clip_low + clip_high) / 2`
- `range_mul_init = inv_norm_cdf(1 - clip_target/2)`
  - `inv_norm_cdf` は標準正規の逆CDF
  - 実装は `Φ^{-1}(p) = sqrt(2) * erfinv(2p-1)`
- `--dq_delta_stat=rms` のときのみ有効（それ以外は警告してスキップ）
- 計算後に `dq_delta_auto_min/max` で clamp
- dq_delta_auto ログに `AutoInitMulApplied/AutoInitMulValue/AutoInitClipTarget` を開始時に出力

### ウォームアップ（range_mul 変更の抑制）

- warmup は **AutoStep の回数**で進行する（optimizer step そのものではない）
- AutoStep が skip されて呼ばれない場合、warmup も進行しない
- warmup は `--dq_delta_auto_warmup` が有効なときのみ機能
- warmup 回数は `--dq_delta_auto_warmup_updates` で上書き可能（0 なら内部デフォルト）

```
warmup_updates = ceil(2 / (1 - dq_delta_auto_ema))  # 未指定時のデフォルト
```

例: `dq_delta_auto_ema=0.95` -> 40 回、`0.90` -> 20 回、`0.98` -> 100 回

※ `1/(1-ema)` は **EMA の実効履歴長**の目安であり、warmup 回数とは別概念。

#### 早期終了

- `clip_low <= clip_rate_ema <= clip_high` が **3 回連続**で成立したら warmup を終了
- 連続回数は外部オプションにせず固定値（3 回）

#### warmup 中の挙動

- AutoStep では **必ず** clip_rate_raw 計測、EMA 更新、ログ出力を行う
- warmup 中は range_mul を変更しない
  - `auto_applied = 0`
  - `range_mul_before == range_mul_after`
  - `auto_reason = "warmup"`

#### warmup 終了後の挙動（既存通り）

```
if use_raw:
    if clip_rate_ema > clip_high and clip_rate_raw > clip_high: range_mul *= mul_up
    elif clip_rate_ema < clip_low and clip_rate_raw < clip_low: range_mul *= mul_down
else:
    if clip_rate_ema > clip_high: range_mul *= mul_up
    elif clip_rate_ema < clip_low: range_mul *= mul_down
# 条件に当たらない場合は変更なし
```

`auto_reason` は `clip_high` / `clip_low` / `in_band` を記録する。

### 更新間隔と判断期間の目安

- 更新間隔は `auto_every`（optimizer step 単位）
- EMA の実効的な履歴長は **約 `1/(1-ema)` 回の更新**  
  例: `auto_ema=0.95` なら約 20 回分の更新を反映  
  `auto_every=100` の場合、**約 2000 optimizer step** 相当の履歴
- 既定（`auto_every=50, auto_ema=0.95`）では **約 1000 optimizer step** 相当の履歴
- 間引き（`auto_every` を大きくする）と低負荷化は両立するが、**追従性は低下**する  
  分布が急変しない学習では 100〜200 でも実用域、急変が起きる設定では 50 前後が安全

### 反応速度の目安表（更新回数とstep換算）

EMAの「実効履歴長」は `約 1/(1-ema)` 回の更新に相当する。  
下表は代表値の目安（optimizer step換算）:

```
auto_every | auto_ema | 実効履歴(更新回) | 実効履歴(optimizer step)
---------- | -------- | ---------------- | ------------------------
50         | 0.90     | 10               | 500
50         | 0.95     | 20               | 1000
100        | 0.90     | 10               | 1000
100        | 0.95     | 20               | 2000
200        | 0.90     | 10               | 2000
200        | 0.95     | 20               | 4000
```

※ 実効履歴が短いほど反応は速いが、ブレやすい。  
※ bits 切替直後は EMA リセットにより即時追従する。

### ログ出力（自動調整時）

- `dq_delta_log` が有効な場合、既存ログに `clip_rate_raw / clip_rate_ema / range_mul_before / range_mul_after / auto_applied / warmup_active / warmup_remain / auto_reason` を追加
- 変更が発生した時のみ `logger.info` で 1 行出力
  - `dq_delta_log` が無効でも、変更時は `logger.info` だけ出す
- `dq_delta_auto_log_file` が指定されている場合は **AutoStep のみ**を CSV で追記する（ログOFF時の追跡用）
  - 最小列: `TrainStep,scope,target,bits,clip_rate_raw,clip_rate_ema,range_mul_before,range_mul_after,auto_applied,warmup_active,warmup_remain,auto_reason,auto_init_mul_applied,auto_init_mul_value,auto_init_clip_target`
  - `full_schema` 指定時は **main log と同じ列構成**で出力する  
    LogStep 以外の列は空欄（NA）で、追加統計は計算しない

## 既存オプションとの組み合わせ

- `--dq_delta_bits_sched` : 有効。bits はスケジュールに従い、range_mul は自動調整
- `--dq_delta_scope` : ログ/制御対象の範囲
- `--dq_quantize_z` : 対象テンソルが z に切り替わるのみ（挙動は同じ）
- `--dq_delta_granularity` : tensor / channel 両対応。clip_rate は **全要素で集計**

## 分散学習（DDP）での挙動

- summary の統計値は **全 rank で all-reduce** してから算出する
- range_mul の更新は **main process のみ**が行い、更新後に **broadcast** で共有する
- ログは **main process のみ**（または rank suffix を付けて分離出力）

## 学習再開（resume）時の扱い

- 既定は **初期値に戻す**（挙動が変わるため、ログに明記する）
- 将来拡張として、trainer state に range_mul を保存して復元する方式を検討

## 計算量・VRAM 見積もり

以下は **dq_delta 有効** を前提にした追加分の概算です。

### A) ログ追加のコスト

- **計算量**: 量子化対象テンソルに対する追加の reduce（`abs`, `mean`, `count`）が 1〜2 回
  - summary モード + 100 optimizer step ごと: **+0.5〜2% 程度**
  - summary モード + 毎 optimizer step: **+5〜10% 程度**
  - per_module + 毎 optimizer step: **+10〜25% 程度**
- **VRAM**: 追加の一時テンソル（`abs`, `mask` など）が最大で **対象テンソルと同サイズ**
  - fp32 変換を行う場合は **要素数×4byte** ぶんの一時領域が発生
  - summary + 100step ごとなら **ピーク増加は数十MB〜最大で 100MB 程度**（層サイズ依存）

### B) range_mul フィードバック制御のコスト

- **計算量**: clip_rate 計測 + EMA 更新 + 係数乗算のみ
  - 50 optimizer step ごとの調整でも **+0.5〜2% 程度**（最小統計のみの場合）
- **VRAM**: 追加の一時テンソルはログと同等
  - ログを無効にしても auto のために clip_rate 計測は必要

### 16GB VRAM 想定の注意

- summary モード + 100step ごとであれば、**16GB でも現実的**
- per_module を毎 step で使うと、**一時メモリの波が大きくなりやすい**ため非推奨

## 参考初期値（推奨）

- `--dq_delta_log --dq_delta_log_every 100 --dq_delta_log_mode summary`
- `--dq_delta_auto_range_mul --dq_delta_auto_every 50 --dq_delta_auto_ema 0.95`
- `--dq_delta_auto_clip_low 0.0005 --dq_delta_auto_clip_high 0.003`
- `--dq_delta_auto_min 1.0 --dq_delta_auto_max 6.0`

## 今後の検討事項

- TE と UNet で別々に range_mul を持つ（scope 分離）
- clip_rate のみならず zero_rate を使った下限制御の追加
- 量子化ノイズの推定（RMS 誤差）ログの追加

## Triton stats

`--dq_delta_use_triton --dq_delta_triton_stats`を指定すると、対応するstats有効stepでstochastic fake quant Bとbasic statsを同じTriton kernelで処理する。

`clip_rate_low_auto`以外のauto presetはclip-only統計を使うため、AutoStepが対応するbasic LogStepと重ならない場合はPyTorch statsへfallbackする。該当設定では学習開始時にwarningを表示する。

対象は`--dq_delta_log_detail basic`相当の`numel`, `clip_count`, `sumsq`, `xq_sumsq`, `xxq_sum`。`full`, `per_module`, `near_zero_rate`, `ZeroRate`, `AbsMax`, `ScaleMin/Mean/Max`が必要なLogStepはPyTorch statsへfallbackする。fake quantのforward自体は通常Triton pathを維持する。

full/per_moduleでfallbackする場合は学習開始時にwarningを出す。対応条件、長時間学習で検証済みの設定、検証コマンドは [triton_windows_setup.md](triton_windows_setup.md) を参照。
