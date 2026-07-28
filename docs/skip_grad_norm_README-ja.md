# SDXL LoRA 向け `--skip_grad_norm` 機能仕様まとめ

## 概要
- 対象スクリプト: `train_network.py` の SDXL LoRA 学習パス。
- 主役オプション `--skip_grad_norm` は、**直近 200 ステップの勾配ノルム実績**から計算した動的しきい値を超えた更新をスキップして、fp16 学習時の破綻を抑える仕組み。
- 勾配ノルムの評価は **GradScaler が unscale する前の値（スケール適用済み）**で行い、スケール切替直後は平均値が追従するまでに遅れが出る仕様。
- スキップ判定の実行タイミングは `accelerator.backward(loss)` の直後で、スキップが発生しても `global_step` は前進する一方、`optimizer.step()` が実行されず更新は入らない。

## 処理フロー
1. `accelerator.backward(loss)` の後、`check_gradients_and_skip_update` が呼ばれる。
2. 全学習パラメータの `.grad` テンソルを走査し、L2 ノルム（スケール適用済み）を計算。
3. ノルム値を移動平均窓（最大 200 件）に追加。NaN/Inf の扱いは後述オプションに依存。
4. 窓が埋まるまでは暫定しきい値として `200000` を使用。埋まった後は `平均 + 2.5σ` を動的しきい値として採用し、`--skip_grad_norm_max` があれば上限を適用。
5. スキップ条件に合致すると `optimizer.zero_grad(set_to_none=True)` だけを実行し、そのステップの更新を取り消す。
6. `accelerator.sync_gradients` が有効なステップでは、スキップしていても進捗バーと `global_step` が進む。スキップ回数は `skipped_steps` として統計に残る。

## 動的しきい値と関連オプション
- **基本式**: `threshold = mean(window) + 2.5 * std(window)`（窓サイズ 200）。値は `numpy` 計算で、NaN が混入すると式全体が NaN になり、後続の比較は `False` 扱い。
- **`--skip_grad_norm_max <float>`**: 動的しきい値の絶対上限。設定しない場合は無制限。窓が埋まる前は常に 200000 を使用する点に注意。
- **`--grad_norm_log`**: `--output_dir/gradient_logs+<output_name>.txt` に CSV 形式で `Epoch,Step,Gradient Norm,Threshold,Loss,ThreshOff[,Scale][,CosineSim]` を出力。100 ステップごとに追記。`ThreshOff=1` は NaN によりしきい値が無効化された状態を示す。
- **`--grad_cosine_log`**: `--grad_norm_log` 併用時に有効。前ステップの勾配テンソルを複製保持し、コサイン類似度を計算・出力する。保持コストが掛かるが挙動には影響しない。
- **`--grad_norm_mode {stable,stable_no_threshoff,gamble}`**: 推奨プリセットを一括指定。`stable` は安定重視、`stable_no_threshoff` は `stable` から ThreshOff 区間を抑制した設定、`gamble` は当たり狙い。全プリセットで勾配コサインログは無効。指定時は個別オプションを無視し、`--no-skip_nan_immediate` / `--no-skip_inf_immediate` のみ上書き可能。
- **GradScaler との関係**: スケール適用前の勾配に切り替えると学習結果が変わったため、仕様として「スケール適用済みの勾配ノルムで平均を作る」ことを前提にしている。スケールが変動した瞬間は窓の平均が追随するまでスキップ判定が遅れる。

## NaN/Inf 取り扱い関連
- **`--nan_to_window` / `--inf_to_window`**: 勾配ノルムが NaN/Inf のときでも窓に挿入する。`numpy` の平均・分散が NaN になり、比較結果は常に `False`（＝スキップしない）になるため、「しばらくスキップ判定が停止した状態」が得られる。
- **`--skip_nan_immediate` / `--skip_inf_immediate`**: 既定値は `True`。オンのままだと NaN/Inf 発生時に即スキップし、GradScaler のヒステリシスが働かずスケールが上がり続けるケースがある。
  - `--no-skip_nan_immediate --no-skip_inf_immediate` を付けると NaN/Inf が出てもステップをスキップしないため、GradScaler が `found_inf` を検知してスケールを下げるトリガーが復活する。
  - このとき `--nan_to_window --inf_to_window` を併用すると、窓が NaN で満たされスキップが完全に停止する「隙間時間」がランダムに挿入され、実運用ではこの不規則性が好ましいとされている。

## よく使うプリセットの挙動
- **設定 1（`--grad_norm_mode stable`）**<br>`--grad_norm_mode stable`
  - 200 ステップ窓を維持しつつしきい値は 200000 にキャップ。ログはノルム・閾値・GradScaler スケールを記録し、勾配類似度は記録しない。
  - NaN/Inf を窓に混入させ、即スキップは無効化するため、ランダムな「スキップ停止区間」が生まれ、GradScaler によるスケール調整も維持される。
  - 実質的に「極端なスパイクのみスキップし、NaN/Inf は学習を止めずにスケール調整へ渡す」挙動になる。
- **設定 2（`--grad_norm_mode stable_no_threshoff`）**<br>`--grad_norm_mode stable_no_threshoff`
  - `stable` と同様にしきい値を 200000 にキャップし、即スキップは無効化する。
  - `nan_to_window` / `inf_to_window` を使わないため、しきい値が NaN 化しにくく、`ThreshOff=1` 区間を抑制できる。
  - 前ステップの勾配を保持せず、`CosineSim` をログ出力しない。
  - フェイク量子化（例: `--dq_delta_bits 8`）時に、`stable` のランダムなブレーキ解除を避けたい場合の比較用モード。
- **設定 3（`--grad_norm_mode gamble`）**<br>`--grad_norm_mode gamble`
  - 最小構成。移動平均 + 2.5σ を超えたステップをスキップし、ログにはノルム・閾値・Lossが出力される。勾配類似度は記録しない。
  - NaN/Inf は即時スキップする（デフォルト値）ため、GradScaler がトリガーされずスケールが上昇し続ける場合がある点に注意。

## 学習進行への影響
- スキップすると `optimizer.step()` と `lr_scheduler.step()` が実行されず、LoRA 重みは更新されない。一方 `global_step` は進むため、学習終了ステップに早期到達する反面、実際の更新回数は減る。
- `skipped_steps` はロガー (`accelerator.log`) に `train/skipped_steps` として記録され、進捗バーにも `skipped` フィールドで表示される。
- マルチ GPU では各 rank でノルム計算を行うため、監督値がずれる可能性がある（コード内コメントにも注意書きあり）。
- fp16 前提で GradScaler に依存しているため、bf16 や fp32 では効果が薄い／挙動が変わる可能性が高い。

## ログファイルの読み方
- ファイル名: `--output_dir/gradient_logs+<出力モデル名>.txt`（`--output_name` 未指定時は `last`）。
- 主な列:
  - `Gradient Norm`: そのステップの L2 ノルム（スケール適用済み）。
  - `Threshold`: 動的しきい値（NaN の場合は空欄になる）。
  - `ThreshOff`: `0`=通常、`1`=しきい値が NaN で無効。
  - `Scale`: `--grad_norm_log` かつ GradScaler 利用時のみ。`torch.cuda.amp.GradScaler.get_scale()` の生値。
  - `CosineSim`: `--grad_cosine_log` 追加時のみ。直前ステップと現ステップの勾配のコサイン類似度。前ステップが存在しない場合は `NaN`。

## 注意事項と既知の挙動
- `skip_grad_norm` をオフにしても `--grad_norm_log` だけでログ記録が可能。その場合はステップスキップは一切行われない。
- `--nan_to_window` で窓に NaN を入れると移動平均が NaN のまま残るが、窓から押し出されるまで自動で回復する。強制的にリセットする仕組みはない。
