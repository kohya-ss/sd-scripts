# `make_lora_diagnostic_report.py` 使い方

## 概要
`tools/make_lora_diagnostic_report.py` は、LoRA学習時のログとチェックポイントをまとめて解析し、診断用のHTML/JSONレポートを生成するツールです。

- `grad_norm` ログ可視化
- `dq_delta` ログ可視化
- LoRA重みの統計（任意）
- LoRA情報密度のエポック推移（任意）

出力先は、`--output_dir` 未指定時に `--input_dir/diagnostic_report` です。

## `--input_dir` 内で使うログファイル名（固定）
このツールは、`--input_dir` 配下の以下ファイル名を前提に読み込みます。

- `gradient_logs+<base_name>.txt`
- `dq_delta_logs+<base_name>.txt`
- `dq_delta_auto+<base_name>.txt`
- `rank_logs+<base_name>.txt`

LoRAチェックポイントは `--input_dir/<base_name>.safetensors` を使用します。  
`--lora_epoch_trend`オプション指定時は`--input_dir/<base_name>-000001.safetensors`などの途中のチェックポイントも使用します。

## オプション一覧
| オプション | 必須 | 既定値 | 説明 |
|---|---|---|---|
| `--base_name` | 必須 | なし | LoRAのベース名 |
| `--input_dir` | 任意 | `.` | ログ/チェックポイントのあるフォルダ |
| `--loss_ma_window` | 任意 | `100` | Loss移動平均の窓サイズ |
| `--lora_bins` | 任意 | `128` | LoRA解析ヒストグラムのビン数 |
| `--skip_lora_analysis` | 任意 | `False` | LoRA重み解析をスキップ |
| `--lora_epoch_trend` | 任意 | `False` | 情報密度のエポック推移解析を有効化 |
| `--output_dir` | 任意 | `<input_dir>/diagnostic_report` | 出力フォルダ |
| `--output_html` | 任意 | 自動生成 | HTML出力先を個別指定 |
| `--output_json` | 任意 | 自動生成 | JSON出力先を個別指定 |

## コマンド例
普段使いの想定コマンド:

```powershell
python tools\make_lora_diagnostic_report.py --base_name loraname --input_dir ..\lora_output --lora_epoch_trend
```

## ファイルが無い場合の動作
`gradient_logs` / `dq_delta_logs` / `rank_logs` / `group_loss_*` / `dq_delta_auto` のうち、1種類以上のログがあればレポートを生成します。

- どのログも無い場合  
  `FileNotFoundError: 入力ログが見つかりません: ...`

`dq_delta_auto` は任意です。無い場合は自動でスキップして続行します。  
`rank_logs` は任意です。無い場合は Rank セクションだけ空になります。

- `dq_delta_auto+<base_name>.txt` が無い  
  auto系の解析だけ省略し、HTML/JSONは生成されます。

LoRAチェックポイント（`.safetensors`）が無い場合は、LoRA重み解析セクションのみ警告表示になり、レポート生成は続行します。

## JSON構造リファレンス（AI解析向け）
この節は `tools/make_lora_diagnostic_report.py` の実装と、`sample/brak_xl32d_noob075V_diagnostic.json` を元にしています。  
JSONは時系列配列（`x`, `rows`, `series[*].y` など）が非常に長くなるため、AIに渡すときは必要な項目だけ抜粋するのがおすすめです。

### ルートキー
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `base_name` | 実行引数 `--base_name` | 学習/診断対象LoRAのベース名 |
| `generated_at` | レポート生成時刻 | JSON生成時刻（`YYYY-MM-DD HH:MM:SS`） |
| `grad` | `gradient_logs+<base_name>.txt` | GradNorm系ログの時系列と要約 |
| `dq` | `dq_delta_logs+<base_name>.txt`（+ 任意で `dq_delta_auto+...`） | DQ delta系ログの時系列と要約 |
| `rank` | `rank_logs+<base_name>.txt`（任意） | Rank系ログの時系列と要約 |
| `group_loss` | `group_loss_logs+<base_name>.csv` / `group_loss_epoch+<base_name>.csv`（任意） | グループ別lossのstep/epoch推移 |
| `lora` | `<base_name>.safetensors` のLoRA重み解析（任意） | 最終チェックポイントの統計 |
| `lora_error` | LoRA解析処理 | `lora` を作れなかった場合のエラー文字列（成功時は `null`） |
| `lora_trend` | `<base_name>-*.safetensors` + 最終 `<base_name>.safetensors`（`--lora_epoch_trend`時） | LoRA情報密度のエポック推移 |
| `lora_trend_error` | LoRA推移解析処理 | `lora_trend` を作れなかった場合のエラー文字列（成功時は `null`） |
| `diagnostics` | `grad` / `dq` / `rank` / `lora` の要約値から派生 | ヒューリスティクス診断スコアと判定一覧 |
| `charts` | `grad` / `dq` / `rank` / `group_loss` / `lora_trend` から派生 | HTML描画用のグラフペイロード |

### `grad` セクション
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `grad.path` | `gradient_logs` | 入力ログファイルパス |
| `grad.rows` | `gradient_logs` | 有効行数（`Epoch`/`Step`が読めた行） |
| `grad.steps_per_epoch` | `gradient_logs` | 推定1epochあたりstep数（中央値推定） |
| `grad.x` | `gradient_logs` | グローバルstep（`epoch * steps_per_epoch + step`） |
| `grad.epochs` | `gradient_logs` | エポック番号（1始まり） |
| `grad.markers[]` | `gradient_logs` | エポック開始位置のマーカー（`x`, `label`） |
| `grad.gradient_norm[]` | `gradient_logs` の `Gradient Norm` 列 | 勾配ノルム |
| `grad.threshold[]` | `gradient_logs` の `Threshold` 列 | しきい値 |
| `grad.loss[]` | `gradient_logs` の `Loss` 列 | 生loss |
| `grad.loss_ma[]` | `grad.loss[]` から派生 | Loss移動平均（窓は `--loss_ma_window`） |
| `grad.thresh_off[]` | `gradient_logs` の `ThreshOff` 列 | しきい値無効化フラグ系指標 |
| `grad.scale[]` | `gradient_logs` の `Scale` 列 | GradScalerのscale |
| `grad.cosine[]` | `gradient_logs` の `CosineSim` 列 | 勾配cos類似度 |
| `grad.summary.*` | 上記 `grad` 時系列から派生 | しきい値超過率、Loss MA低下率、最大gradなどの要約値 |

`grad.summary` の主なキー:

- `threshold_valid_count`
- `threshold_exceeded_count`
- `threshold_exceeded_ratio`
- `thresh_off_ratio`
- `loss_ma_start`
- `loss_ma_end`
- `loss_ma_drop_ratio`
- `cosine_valid_ratio`
- `max_grad_norm`

### `dq` セクション
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `dq.path` | `dq_delta_logs` | 入力DQログパス |
| `dq.auto_path` | `dq_delta_auto`（任意） | 入力DQ autoログパス（無い場合は `null`） |
| `dq.rows[]` | `dq_delta_logs` | DQ本体ログ行（`TrainStep`昇順） |
| `dq.auto_rows[]` | `dq_delta_auto`（任意） | DQ autoログ行（`TrainStep`昇順） |
| `dq.markers[]` | `dq.rows[]` | エポック境界マーカー（`x`, `label`） |
| `dq.summary.*` | `dq.rows[]` / `dq.auto_rows[]` から派生 | bit切替回数、最終EMA値、in-band比率などの要約 |

`dq.rows[]` の主なキー:

- `TrainStep`, `Epoch`
- `Bits`, `RangeMul`
- `ClipRateRaw`, `ClipRateEMA`
- `QuantErrRatioRaw`, `QuantErrRatioEMA`
- `QuantErrRMSRaw`, `QuantErrRMSEMA`
- `QErrPerClip`, `QErrPerClipClipFloor`
- `ActiveClipBand`, `ActiveClipLow`, `ActiveClipHigh`
- `ClipRateLowAutoState`, `ClipRateLowAutoBad`, `ClipRateLowAutoBadStreak`
- `TrainProgress`
- `ClipRateLowAutoMinProgress`, `ClipRateLowAutoFreezeProgress`
- `ClipRateLowAutoThresholdQErrRatio`, `ClipRateLowAutoThresholdQErrPerClip`
- `ClipRateLowAutoPhase`
- `ClipErrRMS`, `RoundErrRMS`, `ClipErrRatio`, `RoundErrRatio`, `ClipShare`, `RoundShare`
- `ZeroRate`, `AbsMax`, `Range`
- `AutoReason`

補足:

- `dq.rows[].Epoch` は `dq_delta_logs` 側に無ければ、`grad.steps_per_epoch` から推定補完されます。

`dq.auto_rows[]` の主なキー:

- `TrainStep`, `Bits`
- `ClipRateRaw`, `ClipRateEMA`
- `RangeMulBefore`, `RangeMulAfter`
- `AutoApplied`, `WarmupActive`, `AutoReason`
- `AutoInitClipTarget`
- `QErrPerClip`, `QErrPerClipClipFloor`
- `ActiveClipBand`, `ActiveClipLow`, `ActiveClipHigh`
- `ClipRateLowAutoState`, `ClipRateLowAutoDecision`, `ClipRateLowAutoReason`
- `ClipRateLowAutoBad`, `ClipRateLowAutoBadStreak`
- `TrainProgress`
- `ClipRateLowAutoMinProgress`, `ClipRateLowAutoFreezeProgress`
- `ClipRateLowAutoThresholdQErrRatio`, `ClipRateLowAutoThresholdQErrPerClip`
- `ClipRateLowAutoPhase`, `ClipRateLowAutoCanEscape`

`dq.summary` の主なキー:

- `rows`, `auto_rows`
- `bit_switches`, `bits_unique`
- `auto_applied_count`
- `in_band_ratio`
- `auto_reason_counts`
- `auto_clip_target_median`
- `clip_ema_cv`
- `final_clip_rate_ema`
- `final_quant_err_ratio_ema`
- `final_quant_err_rms_ema`
- `final_qerr_per_clip`, `max_qerr_per_clip`, `tail_mean_qerr_per_clip`
- `qerr_per_clip_threshold`
- `run_qerr_ratio_threshold`, `run_qerr_per_clip_threshold`
- `low_band_seen`, `low_auto_bad_count`, `low_auto_bad_ratio`, `low_auto_max_bad_streak`
- `low_auto_escaped`, `low_auto_frozen_bad_count`
- `low_auto_state_counts`, `low_auto_decision_counts`, `low_auto_reason_counts`
- `low_auto_phase_counts`
- `low_auto_min_progress`, `low_auto_freeze_progress`
- `low_auto_min_progress_step`, `low_auto_freeze_progress_step`
- `active_clip_band_final`, `active_clip_band_counts`
- `clip_share_tail_mean`, `round_share_tail_mean`
- `clip_err_ratio_tail_mean`, `round_err_ratio_tail_mean`
- `final_zero_rate`

### `rank` セクション
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `rank.path` | `rank_logs` | 入力Rankログパス |
| `rank.rows[]` | `rank_logs` | Rank本体ログ行（`TrainStep`昇順） |
| `rank.markers[]` | `rank.rows[]` | エポック境界マーカー（`x`, `label`） |
| `rank.summary.*` | `rank.rows[]` から派生 | 最終 RankSatP95 などの要約 |

`rank.rows[]` の主なキー:

- `TrainStep`, `Epoch`, `Scope`
- `UnetLRMin`, `UnetLRMax`, `Te1LRMin`, `Te1LRMax`, `Te2LRMin`, `Te2LRMax`
- `RankDim`
- `RankSatWMean`, `RankSatP50`, `RankSatP95`, `RankSatMax`
- `RankTop1P95`, `RankEnergySum`

`rank.summary` の主なキー:

- `rows`
- `final_rank_sat_p95`
- `final_rank_energy_sum`

`rank.grouped`（`rank_log_mode=per_module` 入力時）の主なキー:

- `path.labels`, `path.rows[]`
- `role.labels`, `role.rows[]`

補足:

- `path` は `down / mid / up / other` に粗く集約した推移です。
- `role` は `q / k / v / out / ff / resnet / sampler / conv / other` に粗く集約した推移です。
- 各 group row の `groups.<label>` は `RankEnergySum`, `RankEnergyShare`, `RankSatWMean` を持ちます。
- `summary` ログを入力した場合は `rank.grouped` は空です。

### `group_loss` セクション（任意）
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `group_loss` | group lossログ（任意） | ログが1つも無い場合は `null` |
| `group_loss.step_path` | `group_loss_logs+<base_name>.csv` | stepログパス |
| `group_loss.epoch_path` | `group_loss_epoch+<base_name>.csv` | epoch集計ログパス |
| `group_loss.step_rows[]` | stepログ | step単位のgroup別loss系列 |
| `group_loss.epoch_rows[]` | epochログ | epoch単位のgroup別集計系列 |
| `group_loss.step_markers[]` | `step_rows[]` | エポック境界マーカー |

`group_loss.step_rows[]` のキー:

- `global_step`, `epoch`, `group`
- `loss`, `ema_loss_group`, `count_group`

`group_loss.epoch_rows[]` のキー:

- `epoch`, `group`
- `ema_loss_end`, `count_epoch`, `mean_loss_epoch`

### `lora` セクション（任意）
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `lora.path` | `<base_name>.safetensors` | 解析対象LoRAパス |
| `lora.summary` | LoRA重み解析 | 全体統計（分布統計含む） |
| `lora.summary_cards` | `lora.summary` から派生 | カード表示用の主要値 |
| `lora.module_summary[]` | LoRA重み解析 | `unet/te1/te2` モジュール別統計 |
| `lora.unet_block_summary[]` | LoRA重み解析 | UNetブロック別統計 |
| `lora.diagnostic` | `lora.module_summary` から派生 | モジュール密度バランスなどの診断補助値 |

`lora.summary` の主なキー:

- `total_blocks`, `total_params`, `max_rms`
- `density`, `rms`, `entropy_norm`, `sparsity`

分布オブジェクト（例: `lora.summary.density`）のキー:

- `min`, `max`, `mean`, `median`, `q1`, `q3`

`lora.module_summary[]` の主なキー:

- `module`, `block_count`, `total_params`
- `density`, `rms`, `entropy_norm`（各キーは分布オブジェクト）

`lora.unet_block_summary[]` の主なキー:

- `label`, `block_count`, `total_params`
- `density`, `rms`（各キーは分布オブジェクト）

`lora.diagnostic` のキー:

- `module_balance_ratio`
- `unet_block_count`
- `density_min`, `density_max`

### `lora_trend` セクション（`--lora_epoch_trend`）
| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `lora_trend.model_path` | 最終 `<base_name>.safetensors` | 基準モデルパス |
| `lora_trend.checkpoints[]` | チェックポイント列挙 + LoRA重み解析 | 推移解析に使った各チェックポイント情報 |
| `lora_trend.modules[]` | `checkpoints` 各時点のLoRA重み解析 | モジュール別の情報密度時系列 |

`lora_trend.checkpoints[]` のキー:

- `epoch`, `label`
- `path`, `filename`
- `is_final`

`lora_trend.modules[]` のキー:

- `module`, `module_label`
- `total_series`, `selected_series`, `series_limit`
- `legend_rows[]`, `series[]`

`lora_trend.modules[].series[]` のキー:

- `name`, `color`
- `x`（epoch配列）, `y`（density_mean配列）

`lora_trend.modules[].legend_rows[]` のキー:

- `color`, `name`
- `start_density`, `end_density`, `delta_density`

### `diagnostics` セクション

DQ logsに `ActiveClipBand=low` または `ClipRateLowAuto*` 列がある場合のみ、`clip_rate_low適性` の診断を追加します。
`QErrPerClip` は固定診断基準 `130` を主基準として評価します。ログに `ClipRateLowAutoThresholdQErrPerClip` がある場合は、run指定閾値も補足表示します。

| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `diagnostics.score` | `grad.summary` / `dq.summary` / `lora.diagnostic` から派生 | 100点満点の総合スコア |
| `diagnostics.overall_status` | 同上 | 総合状態（`良好` / `注意` / `要改善`） |
| `diagnostics.overall_class` | 同上 | 状態クラス（`good` / `warn` / `bad`） |
| `diagnostics.checks[]` | 同上 | 項目別チェック結果 |

`diagnostics.checks[]` のキー:

- `section`（`GradNorm` / `DQ` / `LoRA`）
- `name`
- `value`（表示用文字列）
- `status`（`good` / `warn` / `bad` / `info`）
- `note`

### `charts` セクション

DQ logsに該当列がある場合は、`QErrPerClip`（run指定閾値の水平線つき。130と異なる場合は固定130も表示）、`clip_rate_low_auto Bad / Streak`、`ClipErrRatio / RoundErrRatio`、`ClipShare / RoundShare` のグラフを追加します。
`TrainProgress` と low_auto progress閾値がある場合は、min_progress / freeze_progress の位置をDQグラフの縦線マーカーに追加します。

| JSONパス | 情報源 | 内容の概説 |
|---|---|---|
| `charts.grad[]` | `grad` から派生 | Grad系チャート定義 |
| `charts.dq[]` | `dq` から派生 | DQ系チャート定義 |
| `charts.group_loss[]` | `group_loss` から派生 | Group loss系チャート定義 |
| `charts.lora_trend[]` | `lora_trend` から派生 | LoRA推移チャート定義 |

各チャート要素（`charts.*[]`）の共通キー:

- `id`, `title`, `x_label`
- `x`, `markers`
- `series[]`（`name`, `color`, `y` または `x`+`y`）
- 描画補助キー（`y_min_fixed`, `y_tick_step`, `y_tick_precision` など）

### AIに渡すときの最小セット例
用途別に、まずは以下だけ渡すと解釈しやすくなります。

- 学習安定性判定: `grad.summary`, `dq.summary`, `diagnostics`
- 重みの偏り判定: `lora.summary`, `lora.module_summary`, `lora.diagnostic`
- 学習推移判定: `lora_trend.checkpoints`, `lora_trend.modules[].legend_rows`
