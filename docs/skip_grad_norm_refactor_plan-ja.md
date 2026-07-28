# skip_grad_norm 系リファクタリングメモ

## 解読結果（現状挙動の整理）
- 対象ロジックは `train_network.py` の `GradNormGuardian` / `GradNormGuardianConfig` と、引数定義は `library/sdxl_train_util.py`。
- 判定タイミングは `accelerator.backward(loss)` の直後。`check_gradients_and_skip_update()` が呼ばれ、`True` ならその step は `optimizer.step()` を実行せずにスキップ（`global_step` は進む）。`skipped_steps` が進捗/ログに記録される。
- 勾配ノルムは **GradScaler による unscale 前の値（スケール適用済み）**で計算。スケール変動に対する補正は行わない。
- 移動平均窓は `moving_avg_window=200`。窓が満たされるまでの暫定しきい値は `initial_threshold=200000` 固定。
- 動的しきい値は `mean + 2.5 * std`。`skip_grad_norm_max` があれば上限キャップ。ただし **窓が埋まるまではキャップを無視**し、常に `initial_threshold` が使われる。
- NaN/Inf の扱い:
  - `nan_to_window` / `inf_to_window` が有効だと NaN/Inf を窓に入れる。窓が NaN/Inf で満たされると `threshold` が NaN になり比較が常に `False` → その間はスキップ無効（意図的な“ブレーキ解除”）。
  - `skip_nan_immediate` / `skip_inf_immediate` が `True` だと NaN/Inf の step は即スキップ（GradScaler の `found_inf` が働きにくい）。
- ログ（`--grad_norm_log`）は `gradient_logs+<output_name>.txt` に CSV 出力。`ThreshOff` は `0`=通常、`1`=しきい値が NaN で無効。`--grad_cosine_log` で追加列。

## リファクタリング計画（判定結果は不変）
1. **プリセット選択式の導入**
   - 新オプション: `--grad_norm_mode {stable,stable_no_threshoff,gamble}`。
   - `stable` は現行「基本設定」から勾配コサイン診断を無効化した設定（`--skip_grad_norm --grad_norm_log --skip_grad_norm_max 200000 --nan_to_window --inf_to_window --no-skip_nan_immediate --no-skip_inf_immediate`）。
   - `stable_no_threshoff` は `stable` 派生で、`--nan_to_window --inf_to_window` を使わず `ThreshOff` 区間を抑制する設定（`--skip_grad_norm --grad_norm_log --skip_grad_norm_max 200000 --no-skip_nan_immediate --no-skip_inf_immediate`）。
   - `gamble` は現行「博打設定」から勾配コサイン診断を無効化した設定（`--skip_grad_norm --grad_norm_log`）。
   - 既存オプションは **後方互換のため残すが、プリセット指定時は否定フラグのみ上書き可**にする。

2. **設定組み立ての共通化**
   - `train_network.py` 側の大量の `getattr(args, ...)` をヘルパー関数へ集約。
   - プリセット適用 → 明示オプション上書き → `GradNormGuardianConfig` 生成、の順で単一路線化。

3. **ログ出力の維持**
   - `--grad_norm_log` の CSV 形式を維持し、`ThreshOff` は `0/1` のみで運用する。
   - 全プリセットで任意列の `CosineSim` を省略し、それ以外の列と出力頻度は変えない。

4. **未使用オプションの削除**
   - `--nan_inf_until_step`、`--auto_cap_release`、`--idle_free_phase` を **ソースとドキュメントから削除**。
   - 関連する内部フラグや補助パラメータ（`cap_release_*` など）も併せて削除し、現行の判定結果を崩さない範囲で簡素化する。

5. **ドキュメント更新**
   - `docs/skip_grad_norm_README-ja.md` と `README.md` のプリセット記述を **新オプション表記へ置換**。
   - 既存オプションは「上級者向けの手動調整」として併記し、優先順位ルールを明記。

6. **検証方針**
   - 既存 2 プリセットで **スキップ判定が完全一致**することを確認。
   - `grad_norm_log` の出力列/頻度が変わらないことを差分比較（ログの行数・ヘッダ）。
