# Observing a Running Training (Run Control) / 実行中の学習の観測（Run Control）

> **Experimental.** This feature is new. The file formats may change; `schema_version` in the files is incremented when they change incompatibly.
>
> **実験的機能です。** 新しい機能のため、ファイル形式は変更される可能性があります。互換性のない変更をするときは、ファイル内の `schema_version` を増やします。

## Overview / 概要

`--run_control` writes the state of the training to plain JSON files while it is running. A human, a script or an AI agent can read them to know how the training is going, without parsing the TensorBoard event files. TensorBoard / W&B logging is not affected and can be used together.

The files are written to the *run directory* `<output_dir>/<output_name>_run`:

| File | Content | Written |
|---|---|---|
| `manifest.json` | Resolved arguments, dataset summary, environment | Once at start |
| `status.json` | Current snapshot: state, step, loss, learning rate, speed, ETA, VRAM, last outputs | Overwritten about every 5 seconds |
| `events.jsonl` | Run start/end, checkpoints, states, sample images, anomalies. One JSON object per line | Appended |

This is observation only: the training itself does not change, and no file is written unless the option is specified. Sending commands to the running training (sample generation, saving, stopping) is planned as a later addition on the same directory.

Supported: the LoRA training scripts based on `train_network.py` (`train_network.py`, `sdxl_train_network.py`, `sd3_train_network.py`, `flux_train_network.py`, `lumina_train_network.py`, `hunyuan_image_train_network.py`, `anima_train_network.py`, `moss_image_train_network.py`). With multi-process training (`accelerate launch --multi_gpu`), only the main process writes.

`--run_control` を指定すると、学習中の状態を通常の JSON ファイルに書き出します。TensorBoard のイベントファイルを解析しなくても、人、スクリプト、AI エージェントが学習の様子を読み取れます。TensorBoard / W&B へのログ出力には影響せず、併用できます。

ファイルは *実行ディレクトリ* `<output_dir>/<output_name>_run` に書き出されます。

| ファイル | 内容 | 書き込み |
|---|---|---|
| `manifest.json` | 解決済みの引数、データセットの集計、実行環境 | 開始時に 1 回 |
| `status.json` | 現在のスナップショット: 状態、ステップ、loss、学習率、速度、ETA、VRAM、最後の出力 | 約 5 秒ごとに上書き |
| `events.jsonl` | 開始と終了、チェックポイント、state、サンプル画像、異常。1 行に 1 つの JSON オブジェクト | 追記 |

これは観測のみの機能です。学習そのものは変わらず、オプションを指定しない限りファイルは書き出されません。実行中の学習へのコマンド送信（サンプル生成、保存、停止）は、同じディレクトリへの追加として今後対応する予定です。

対応スクリプト: `train_network.py` を基にした LoRA 学習スクリプト（`train_network.py`、`sdxl_train_network.py`、`sd3_train_network.py`、`flux_train_network.py`、`lumina_train_network.py`、`hunyuan_image_train_network.py`、`anima_train_network.py`、`moss_image_train_network.py`）。マルチプロセス学習（`accelerate launch --multi_gpu`）では、メインプロセスだけが書き込みます。

## Usage / 使い方

Add `--run_control` to the training command.

```bash
accelerate launch anima_train_network.py \
  --output_dir output --output_name my_lora \
  ... \
  --run_control
```

The files are then in `output/my_lora_run/`. For example, to watch the progress:

```bash
cat output/my_lora_run/status.json
tail -f output/my_lora_run/events.jsonl
```

**Note:** Unlike the TensorBoard logs, which are kept in a separate directory for each run, the run directory is reused if the training is started again with the same `output_dir` and `output_name` (the saved models are overwritten in the same way). `manifest.json` and `status.json` of the previous run are then overwritten; only `events.jsonl` is appended. Copy or rename the run directory before starting if the files of the previous run should be kept. Each run has its own `run_id`, which is in every file and every event, so the events of each run can be told apart.

学習コマンドに `--run_control` を追加します。ファイルは上の例では `output/my_lora_run/` に出力されます。

**注意:** 実行ごとに別のディレクトリに残る TensorBoard のログとは異なり、同じ `output_dir` と `output_name` で再度学習を始めると、実行ディレクトリは再利用されます（保存されるモデルが上書きされるのと同じです）。このとき前回の `manifest.json` と `status.json` は上書きされ、`events.jsonl` だけが追記されます。前回のファイルを残したい場合は、開始前に実行ディレクトリをコピーするか名前を変更してください。実行ごとに固有の `run_id` がすべてのファイルとイベントに含まれるので、イベントは実行ごとに区別できます。

## status.json

```json
{
  "schema_version": 1,
  "run_id": "20261119-181931-13480",
  "pid": 13480,
  "state": "running",
  "started_at": 1795079971.87,
  "updated_at": 1795079994.73,
  "progress": {"global_step": 120, "max_train_steps": 1600, "epoch": 2, "num_train_epochs": 16},
  "loss": {"current": 0.2526, "average": 0.2287},
  "lr": {"unet": 0.0001},
  "metrics": {},
  "speed": {"steps_per_sec": 1.05, "sec_per_step": 0.95, "elapsed_sec": 130.2, "eta_sec": 1420.0},
  "timing": {"data_wait_sec_per_batch": 0.002, "data_wait_ratio": 0.002},
  "vram": [{"device": "cuda:0", "allocated_mb": 4178, "reserved_mb": 4206, "peak_allocated_mb": 6361}],
  "last_checkpoint": {"path": "...", "step": 100, "epoch": 1, "time": 1795079980.66},
  "last_state": {"path": "...", "step": 100, "epoch": 1, "time": 1795079980.93},
  "last_sample": {"dir": "...", "files": ["..."], "step": 100, "epoch": 1, "time": 1795079985.28},
  "anomalies": {},
  "error": null
}
```

| Field | Description |
|---|---|
| `state` | `starting` (before the first step, including the first sample generation), `running`, `finished`, `failed` (ended by an exception, see `error`), `interrupted` (Ctrl+C), `aborted` (exited before the training finished without an exception) |
| `updated_at` | Unix time of the last update. All times in the files are Unix time in seconds |
| `progress` | `global_step` is the number of optimizer steps. `epoch` is the current epoch, starting from 1 (`0` before the first epoch) |
| `loss` | Loss of the last batch and its moving average (the same values as the progress bar and TensorBoard). `null` if the value is not finite |
| `lr` | Learning rates by parameter group, the same as `lr/*` in TensorBoard |
| `metrics` | Other step logs for TensorBoard, e.g. `norm/*`, `max_norm/*` |
| `speed` | `steps_per_sec` and `sec_per_step` are measured since the previous update of the file, so they include the time for sampling and saving in that period. `eta_sec` is from the average since the first step |
| `timing` | Time spent waiting for the next batch from the DataLoader, per batch and as the ratio to the total time. A high ratio means that data loading is the bottleneck |
| `vram` | Memory of each device used by this process, as reported by PyTorch (`memory_allocated`, `memory_reserved`, `max_memory_allocated`). `device` is the device name in PyTorch, e.g. `cuda:0`. CUDA, XPU and MPS are supported; a value which the backend does not provide is `null` (MPS has no peak), and the list is empty on CPU |
| `last_checkpoint`, `last_state`, `last_sample` | The last saved model, the last saved state and the last generated sample images |
| `anomalies` | Count by kind. Currently `non_finite_loss` (the loss was NaN or infinity) |
| `error` | `type`, `message` and the last lines of `traceback` when the state is `failed` or `interrupted` |

The status does not change while a long operation between steps (sample generation, saving, validation) is running: `state` stays `running` and `updated_at` stays old.

If the process is killed (e.g. from the task manager, or by the out-of-memory killer of the OS), the final status cannot be written and `state` stays `running`. A reader can detect this by `updated_at` being old, or by checking whether the process of `pid` still exists.

| フィールド | 説明 |
|---|---|
| `state` | `starting`（最初のステップの前。最初のサンプル生成を含む）、`running`、`finished`、`failed`（例外で終了。`error` を参照）、`interrupted`（Ctrl+C）、`aborted`（例外なしで学習完了前に終了） |
| `updated_at` | 最後に更新した Unix 時刻。ファイル内の時刻はすべて秒単位の Unix 時刻です |
| `progress` | `global_step` は optimizer のステップ数。`epoch` は現在の epoch で 1 から始まります（最初の epoch の前は `0`） |
| `loss` | 最後のバッチの loss とその移動平均（プログレスバーや TensorBoard と同じ値）。有限の値でないときは `null` |
| `lr` | パラメータグループごとの学習率。TensorBoard の `lr/*` と同じです |
| `metrics` | TensorBoard 向けのその他のステップログ（`norm/*`、`max_norm/*` など） |
| `speed` | `steps_per_sec` と `sec_per_step` はファイルの前回更新からの計測値で、その間のサンプル生成や保存の時間を含みます。`eta_sec` は最初のステップからの平均で計算します |
| `timing` | DataLoader から次のバッチを待った時間（バッチあたり、および全体に対する割合）。割合が高い場合はデータ読み込みがボトルネックです |
| `vram` | このプロセスが使っている各デバイスのメモリ（PyTorch の `memory_allocated`、`memory_reserved`、`max_memory_allocated`）。`device` は `cuda:0` のような PyTorch 上のデバイス名です。CUDA、XPU、MPS に対応しています。バックエンドが提供しない値は `null` になり（MPS にはピーク値がありません）、CPU では空のリストになります |
| `last_checkpoint`、`last_state`、`last_sample` | 最後に保存したモデル、最後に保存した state、最後に生成したサンプル画像 |
| `anomalies` | 種類ごとの回数。現在は `non_finite_loss`（loss が NaN または無限大）のみ |
| `error` | `state` が `failed` または `interrupted` のときの `type`、`message`、`traceback` の末尾 |

ステップ間の長い処理（サンプル生成、保存、validation）の実行中は status は変わりません。`state` は `running` のまま、`updated_at` は古いままになります。

プロセスが強制終了された場合（タスクマネージャーからの終了、OS のメモリ不足による強制終了など）は、最後の status を書き込めないため `state` は `running` のまま残ります。読む側は、`updated_at` が古いこと、または `pid` のプロセスが存在しないことで検出できます。

## events.jsonl

Every line has `ts` (Unix time), `time` (local time in ISO 8601), `run_id`, `event`, `step` (the number of optimizer steps) and `epoch` (the current epoch, starting from 1, `0` before the first epoch).

```json
{"ts": 1795079980.66, "time": "2026-11-19T18:19:40+09:00", "run_id": "20261119-181931-13480", "event": "checkpoint_saved", "step": 5, "epoch": 1, "path": "..."}
```

| `event` | Additional fields | When |
|---|---|---|
| `run_start` | `run_dir`, `manifest` | The training starts |
| `checkpoint_saved` | `path` | A model is saved (every N steps / epochs, and at the end) |
| `state_saved` | `path` | A state is saved (`--save_state`) |
| `samples_generated` | `dir`, `files` | Sample images are generated |
| `epoch_end` | `average_loss` | An epoch ends, after saving and sampling |
| `anomaly` | `kind`, ... | The first non-finite loss of the run. Later ones are only counted in `status.json` |
| `run_end` | `state`, `error` | The training ends. `state` and `error` are the same as in `status.json` |

すべての行に `ts`（Unix 時刻）、`time`（ISO 8601 形式のローカル時刻）、`run_id`、`event`、`step`（optimizer のステップ数）、`epoch`（現在の epoch。1 から始まり、最初の epoch の前は `0`）があります。

| `event` | 追加フィールド | タイミング |
|---|---|---|
| `run_start` | `run_dir`、`manifest` | 学習の開始 |
| `checkpoint_saved` | `path` | モデルの保存（N ステップ / epoch ごと、および終了時） |
| `state_saved` | `path` | state の保存（`--save_state`） |
| `samples_generated` | `dir`、`files` | サンプル画像の生成 |
| `epoch_end` | `average_loss` | epoch の終了（保存とサンプル生成の後） |
| `anomaly` | `kind` など | その実行で最初の非有限 loss。2 回目以降は `status.json` の回数にのみ反映されます |
| `run_end` | `state`、`error` | 学習の終了。`state` と `error` は `status.json` と同じです |

## manifest.json

| Field | Description |
|---|---|
| `script`, `command_line`, `cwd` | How the training was started |
| `sd_scripts_commit_hash` | Git commit of sd-scripts |
| `training` | Numbers determined at runtime: `max_train_steps`, `num_train_epochs`, `num_batches_per_epoch`, `gradient_accumulation_steps`, `num_processes` |
| `outputs` | `output_dir`, `output_name`, `sample_dir`, `logging_dir` as absolute paths |
| `environment` | Python, platform, CUDA, versions of the main packages, GPUs |
| `args` | All arguments after merging the command line and `--config_file` |
| `metadata` | The training metadata stored in the model (`ss_*`), with nested values decoded. This has the dataset summary (`ss_datasets`: resolution, buckets, subsets, number of images). The tag frequency (`ss_tag_frequency`, and `tag_frequency` of each dataset) is not copied because it is large; it is stored in the model metadata as before |

`--wandb_api_key` and `--huggingface_token` are replaced with `***`. **The manifest contains local paths** (model, dataset, output). Check the content before sharing the file, for example in a bug report.

| フィールド | 説明 |
|---|---|
| `script`、`command_line`、`cwd` | 学習の起動方法 |
| `sd_scripts_commit_hash` | sd-scripts の Git コミット |
| `training` | 実行時に決まる値: `max_train_steps`、`num_train_epochs`、`num_batches_per_epoch`、`gradient_accumulation_steps`、`num_processes` |
| `outputs` | `output_dir`、`output_name`、`sample_dir`、`logging_dir` の絶対パス |
| `environment` | Python、プラットフォーム、CUDA、主なパッケージのバージョン、GPU |
| `args` | コマンドラインと `--config_file` をマージした後のすべての引数 |
| `metadata` | モデルに保存される学習メタデータ（`ss_*`）。入れ子の値はデコード済みです。データセットの集計（`ss_datasets`: 解像度、バケット、subset、画像枚数）が含まれます。タグ頻度（`ss_tag_frequency` と、各データセットの `tag_frequency`）は大きいためコピーしません。モデルのメタデータには従来どおり保存されます |

`--wandb_api_key` と `--huggingface_token` は `***` に置き換えられます。**マニフェストにはローカルのパス**（モデル、データセット、出力先）**が含まれます**。バグ報告などでファイルを共有する前に内容を確認してください。

## Notes for Readers / 読む側の注意

- `manifest.json` and `status.json` are written to a temporary file and then renamed, so a reader never sees a partially written file.
- On Windows, the rename fails while another process has `status.json` open. The update is then skipped and retried at the next step, so open the file only briefly (read it and close it).
- `events.jsonl` is only appended. A reader can remember the position and read the new lines.
- Errors in writing these files never stop the training. A warning is shown once.

- `manifest.json` と `status.json` は一時ファイルに書いてからリネームするため、書きかけのファイルが読まれることはありません。
- Windows では、他のプロセスが `status.json` を開いている間はリネームに失敗します。その場合は更新をスキップして次のステップで再試行するため、ファイルは短時間だけ開く（読んだらすぐ閉じる）ようにしてください。
- `events.jsonl` は追記のみです。読む側は位置を覚えておき、新しい行だけを読めます。
- これらのファイルの書き込みエラーで学習が止まることはありません。警告が 1 回表示されます。
