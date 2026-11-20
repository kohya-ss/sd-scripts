# Observing and Controlling a Running Training (Run Control) / 実行中の学習の観測と操作（Run Control）

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
| `loss.jsonl` | Loss of each training sample with its image, timestep and resolution. One JSON object per line | Appended |
| `commands/` | Commands to the training (sample generation, saving, stopping) are put here as JSON files; the results are written to `commands/done/` | By the sender / the training |

Nothing changes in the training unless the option is specified, and the files only record the training. The exception is `commands/`: the training can be controlled from outside by putting command files there (see [Sending Commands](#sending-commands--コマンドの送信)).

For AI agents which start and watch a training, [agent_guide.md](./agent_guide.md) has the procedure and the pitfalls in a short form (in English).

Supported: the LoRA training scripts based on `train_network.py` (`train_network.py`, `sdxl_train_network.py`, `sd3_train_network.py`, `flux_train_network.py`, `lumina_train_network.py`, `hunyuan_image_train_network.py`, `anima_train_network.py`, `moss_image_train_network.py`). With multi-process training (`accelerate launch --multi_gpu`), only the main process writes.

`--run_control` を指定すると、学習中の状態を通常の JSON ファイルに書き出します。TensorBoard のイベントファイルを解析しなくても、人、スクリプト、AI エージェントが学習の様子を読み取れます。TensorBoard / W&B へのログ出力には影響せず、併用できます。

ファイルは *実行ディレクトリ* `<output_dir>/<output_name>_run` に書き出されます。

| ファイル | 内容 | 書き込み |
|---|---|---|
| `manifest.json` | 解決済みの引数、データセットの集計、実行環境 | 開始時に 1 回 |
| `status.json` | 現在のスナップショット: 状態、ステップ、loss、学習率、速度、ETA、VRAM、最後の出力 | 約 5 秒ごとに上書き |
| `events.jsonl` | 開始と終了、チェックポイント、state、サンプル画像、異常。1 行に 1 つの JSON オブジェクト | 追記 |
| `loss.jsonl` | 学習した各サンプルの loss と、その画像、timestep、解像度。1 行に 1 つの JSON オブジェクト | 追記 |
| `commands/` | 学習へのコマンド（サンプル生成、保存、停止）を JSON ファイルとして置く場所。結果は `commands/done/` に書かれます | 送信側 / 学習側 |

オプションを指定しない限り学習には何の変化もなく、これらのファイルは学習を記録するだけです。例外は `commands/` で、ここにコマンドファイルを置くと外部から学習を操作できます（[コマンドの送信](#sending-commands--コマンドの送信)を参照）。

学習を起動して見守る AI エージェント向けに、手順と注意点を短くまとめた [agent_guide.md](./agent_guide.md)（英語）があります。

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

**Note:** Unlike the TensorBoard logs, which are kept in a separate directory for each run, the run directory is reused if the training is started again with the same `output_dir` and `output_name` (the saved models are overwritten in the same way). `manifest.json` and `status.json` of the previous run are then overwritten; `events.jsonl` and `loss.jsonl` are appended. Copy or rename the run directory before starting if the files of the previous run should be kept. Each run has its own `run_id`, which is in every file and every event, so the events of each run can be told apart (in `loss.jsonl`, each run starts with a line which has its `run_id`).

学習コマンドに `--run_control` を追加します。ファイルは上の例では `output/my_lora_run/` に出力されます。

**注意:** 実行ごとに別のディレクトリに残る TensorBoard のログとは異なり、同じ `output_dir` と `output_name` で再度学習を始めると、実行ディレクトリは再利用されます（保存されるモデルが上書きされるのと同じです）。このとき前回の `manifest.json` と `status.json` は上書きされ、`events.jsonl` と `loss.jsonl` には追記されます。前回のファイルを残したい場合は、開始前に実行ディレクトリをコピーするか名前を変更してください。実行ごとに固有の `run_id` がすべてのファイルとイベントに含まれるので、イベントは実行ごとに区別できます（`loss.jsonl` では、各実行の最初にその `run_id` を持つ行があります）。

## Sending Commands / コマンドの送信

A training running with `--run_control` accepts commands from outside. A command is executed after the next optimizer step, so the training is at a consistent point, and the training continues after it (except `stop`).

`tools/run_control_cli.py` sends a command, waits for the result and prints it as JSON. It needs only the Python standard library.

```bash
# generate sample images now, with the current weights
python tools/run_control_cli.py output/my_lora_run sample                       # all prompts in the prompt file
python tools/run_control_cli.py output/my_lora_run sample --index 0 2 --seed 42 # the 1st and the 3rd prompts, with another seed
python tools/run_control_cli.py output/my_lora_run sample --prompt "a cat on the sofa --w 1024 --h 1024 --d 1"

# save the model / the state now
python tools/run_control_cli.py output/my_lora_run save
python tools/run_control_cli.py output/my_lora_run save_state

# save the model and the state, and end the training
python tools/run_control_cli.py output/my_lora_run stop

# free the VRAM for a while, then continue
python tools/run_control_cli.py output/my_lora_run pause
python tools/run_control_cli.py output/my_lora_run resume

# print status.json
python tools/run_control_cli.py output/my_lora_run status
```

| Command | Action | Result |
|---|---|---|
| `sample` | Generate sample images with the current weights. `--index`: prompts in the prompt file (`--sample_prompts`) by position, from 0. `--prompt`: prompts in the syntax of the prompt file (`--w`, `--h`, `--d`, `--s`, `--l`, `--n` ...). Both can be given; without them, all prompts in the prompt file. `--seed`, `--steps`, `--width`, `--height`, `--scale`, `--negative_prompt` are applied to all prompts of the command | `files`: paths of the images |
| `save` | Save the model with the name of the current step (`<output_name>-step00001234.safetensors`) | `path` |
| `save_state` | Save the state with the name of the current step. `--save_state` is not needed | `path` |
| `stop` | Save the model and the state with the name of the current step (skip with `--no_save` / `--no_save_state`), then end the training normally | `checkpoint`, `state` |
| `pause` | Move the base models to CPU to free the VRAM, and wait. The process and all its state stay as they are | `vram_before`, `vram_after` |
| `resume` | Move the models back and continue a paused training | `paused_sec` |

Common options: `--no_wait` returns without waiting for the result, `--timeout` is the time to wait in seconds (default 3600; the command is not canceled by the timeout). The exit code is 0 if the command is done, 1 if it failed or was not executed, 2 for the timeout.

Notes:

- **Prompts which are not in the prompt file need the text encoder.** When the text encoder outputs are cached and the text encoder is not kept in memory (e.g. `--cache_text_encoder_outputs` with Anima, MOSS-Image, FLUX.1), only the prompts and negative prompts of the prompt file can be used; they were encoded at the start. `sample --prompt` with another prompt (or `--negative_prompt` with another negative prompt) is then rejected with an error before the generation starts, and the training continues. `--index` with `--seed`, `--steps`, `--width`, `--height` and `--scale` works in both cases.
- The images of the `sample` command are saved to the same directory as the usual sample images. Their file names have numbers from 100 (`..._000123_100_...png`), not to overwrite the scheduled sample images of the same step.
- `stop` does not save the model with the final name (`<output_name>.safetensors`), so that a stopped training is not taken as a finished one. `state` in `status.json` becomes `stopped`. The saved state can be used with `--resume` to continue the training.
- A model saved by `save` or `stop` is not counted for `--save_last_n_steps`; it is not removed automatically.
- If a command cannot be executed as requested (unknown command, index out of range, prompt which cannot be encoded), it is rejected before anything is done: the error is returned and the training continues. An unexpected error in a command (e.g. out of memory during the sample generation) ends the training, the same as an error in the scheduled sample generation or saving.
- Commands are executed one at a time in the order they were sent. Commands sent before the training started, and commands not yet executed when the training ends, are not executed (the result has the status `expired`).
- Commands are not accepted in multi-process training (`accelerate launch --multi_gpu`) for now. Replica training (`--replica_devices`) is supported.

`--run_control` で実行中の学習は、外部からのコマンドを受け付けます。コマンドは次の optimizer ステップの後に実行されるため、学習は整合性のある地点で処理され、コマンドの後も学習は続きます（`stop` を除く）。

`tools/run_control_cli.py` はコマンドを送り、結果を待って JSON で表示します。Python の標準ライブラリだけで動きます。

| コマンド | 動作 | 結果 |
|---|---|---|
| `sample` | 現在の重みでサンプル画像を生成します。`--index`: プロンプトファイル（`--sample_prompts`）内のプロンプトを位置（0 から）で指定します。`--prompt`: プロンプトファイルと同じ書式（`--w`、`--h`、`--d`、`--s`、`--l`、`--n` など）のプロンプトです。両方指定でき、どちらも省略するとプロンプトファイルのすべてのプロンプトを使います。`--seed`、`--steps`、`--width`、`--height`、`--scale`、`--negative_prompt` は、そのコマンドのすべてのプロンプトに適用されます | `files`: 画像のパス |
| `save` | 現在のステップの名前（`<output_name>-step00001234.safetensors`）でモデルを保存します | `path` |
| `save_state` | 現在のステップの名前で state を保存します。`--save_state` の指定は不要です | `path` |
| `stop` | 現在のステップの名前でモデルと state を保存し（`--no_save` / `--no_save_state` で省略可）、学習を正常に終了します | `checkpoint`、`state` |
| `pause` | ベースモデルを CPU に移して VRAM を空け、待機します。プロセスとその状態はそのまま残ります | `vram_before`、`vram_after` |
| `resume` | モデルを元に戻し、一時停止中の学習を続けます | `paused_sec` |

共通オプション: `--no_wait` は結果を待たずに戻ります。`--timeout` は待つ時間（秒）です（デフォルト 3600。タイムアウトしてもコマンドは取り消されません）。終了コードは、完了で 0、失敗または未実行で 1、タイムアウトで 2 です。

注意点:

- **プロンプトファイルにないプロンプトには Text Encoder が必要です。** Text Encoder の出力をキャッシュして Text Encoder をメモリに残さない構成（Anima、MOSS-Image、FLUX.1 での `--cache_text_encoder_outputs` など）では、プロンプトファイルにあるプロンプトとネガティブプロンプトしか使えません（これらは開始時にエンコード済みです）。この場合、別のプロンプトを `sample --prompt` で指定する（または別のネガティブプロンプトを `--negative_prompt` で指定する）と、生成を始める前にエラーとして拒否され、学習は続きます。`--index` と `--seed`、`--steps`、`--width`、`--height`、`--scale` の組み合わせは、どちらの構成でも使えます。
- `sample` コマンドの画像は、通常のサンプル画像と同じディレクトリに保存されます。同じステップの定期サンプル画像を上書きしないよう、ファイル名の番号は 100 から始まります（`..._000123_100_...png`）。
- `stop` は最終的な名前（`<output_name>.safetensors`）ではモデルを保存しません。途中で止めた学習が、完了した学習と区別できるようにするためです。`status.json` の `state` は `stopped` になります。保存された state は `--resume` に指定して学習を再開できます。
- `save` や `stop` で保存したモデルは `--save_last_n_steps` の対象外で、自動では削除されません。
- コマンドを指示どおりに実行できない場合（不明なコマンド、範囲外の番号、エンコードできないプロンプト）は、何も実行する前に拒否します。エラーを返して学習を続けます。コマンド中の予期しないエラー（サンプル生成中のメモリ不足など）では学習が終了します。これは定期的なサンプル生成や保存でのエラーと同じ扱いです。
- コマンドは送った順に 1 つずつ実行されます。学習の開始前に送られたコマンドと、学習の終了時にまだ実行されていないコマンドは実行されません（結果の status は `expired` になります）。
- マルチプロセス学習（`accelerate launch --multi_gpu`）では、現時点ではコマンドを受け付けません。レプリカ学習（`--replica_devices`）には対応しています。

### Pausing / 一時停止

`pause` frees the VRAM without ending the process, for example to use the GPU for something else for a while. `resume` continues the training from the same point; nothing has to be loaded again, and the training goes on exactly as if it had not been paused.

- The base models on the GPU (the U-Net / DiT, and the text encoders and the VAE if they are on the GPU) are moved to CPU, and the cache of PyTorch is released. `vram_before` and `vram_after` in the result show the memory of PyTorch before and after (the same format as `vram` in `status.json`).
- **Not everything is freed.** The network (LoRA) and the state of the optimizer stay on the GPU; they are small in LoRA training. The CUDA context of the process (a few hundred MB, not included in `vram_after`) also stays. For example, an Anima LoRA training went from about 4.2GB to 0.14GB allocated by PyTorch.
- The models need the main memory (RAM) while they are on CPU.
- While paused, `state` in `status.json` is `paused`, and `updated_at` keeps being updated. Only `resume` and `stop` are executed; other commands wait until the training is resumed. `stop` works while paused (the model and the state are saved, then the training ends).
- Moving the models takes time in proportion to their size, both for `pause` and for `resume`.
- The paused time is not counted for the speed and the ETA.
- Not supported yet: `--block_swap_h2d_only` (the usual block swap, `--blocks_to_swap` without it, is supported), replica training (`--replica_devices`) and DeepSpeed. `pause` returns an error in these cases and the training continues.

To free everything including the CUDA context and the RAM, use `stop` and continue later with `--resume` instead.

`pause` は、プロセスを終了せずに VRAM を空けます。GPU をしばらく別の用途に使いたいときなどに使います。`resume` で同じ地点から学習を続けます。何も読み込み直す必要はなく、学習は一時停止しなかった場合とまったく同じように進みます。

- GPU 上のベースモデル（U-Net / DiT。Text Encoder と VAE も GPU 上にあれば対象）を CPU に移し、PyTorch のキャッシュを解放します。結果の `vram_before` と `vram_after` は、その前後の PyTorch のメモリです（`status.json` の `vram` と同じ形式）。
- **すべてが解放されるわけではありません。** ネットワーク（LoRA）と optimizer の状態は GPU に残ります（LoRA 学習では小さいものです）。プロセスの CUDA コンテキスト（数百 MB。`vram_after` には含まれません）も残ります。例として、Anima の LoRA 学習では PyTorch の確保量が約 4.2GB から 0.14GB になりました。
- CPU に移している間、モデルの分のメインメモリ（RAM）が必要です。
- 一時停止中は `status.json` の `state` が `paused` になり、`updated_at` は更新され続けます。実行されるのは `resume` と `stop` だけで、他のコマンドは再開まで待たされます。`stop` は一時停止中でも使えます（モデルと state を保存して学習を終了します）。
- モデルの移動には、`pause` と `resume` のどちらでも、モデルの大きさに応じた時間がかかります。
- 一時停止していた時間は、速度と ETA の計算に含まれません。
- 未対応: `--block_swap_h2d_only`（通常の block swap、つまりこのオプションなしの `--blocks_to_swap` には対応しています）、レプリカ学習（`--replica_devices`）、DeepSpeed。これらの場合 `pause` はエラーを返し、学習は続きます。

CUDA コンテキストや RAM も含めてすべてを解放したい場合は、代わりに `stop` を使い、後で `--resume` で続きを学習してください。

### Command files / コマンドファイル

The CLI is a thin wrapper: a command is a JSON file in `<run directory>/commands/`, so any program (or an AI agent) can send commands by writing files.

1. Write the command to a temporary name in `commands/` and rename it to `<name>.json` (the training reads `*.json` only; the rename makes sure that it never reads a partially written file). Commands are executed in the order of the file names; the CLI uses `<time in nanoseconds>_<command>.json`.
2. The training executes the command, writes the result to `commands/done/<name>.json` and removes the command file.

```json
{"command": "sample", "prompts": ["a cat on the sofa --w 1024 --h 1024"], "indices": [0], "overrides": {"seed": 42}}
{"command": "save"}
{"command": "save_state"}
{"command": "stop", "save": true, "save_state": true}
{"command": "pause"}
{"command": "resume"}
```

For `sample`, `overrides` uses the keys of the prompt: `width`, `height`, `seed`, `sample_steps`, `scale`, `negative_prompt` and so on.

The result file:

```json
{
  "run_id": "20261119-222538-32888",
  "command": {"command": "sample", "indices": [0], "overrides": {"seed": 5}},
  "status": "ok",
  "step": 4,
  "epoch": 1,
  "finished_at": 1795094750.47,
  "result": {"files": ["/path/to/output/sample/my_lora_000004_100_20261119222550_5.png"]}
}
```

`status` is `ok`, `error` (with `error`: the message) or `expired`. `step` and `epoch` are the ones at which the command was executed.

CLI は薄いラッパーです。コマンドの実体は `<実行ディレクトリ>/commands/` に置く JSON ファイルなので、任意のプログラム（や AI エージェント）がファイルを書くだけでコマンドを送れます。

1. コマンドを `commands/` 内の一時的な名前で書き、`<名前>.json` にリネームします（学習側は `*.json` だけを読みます。リネームにより、書きかけのファイルが読まれることはありません）。コマンドはファイル名の順に実行されます。CLI は `<ナノ秒単位の時刻>_<コマンド>.json` という名前を使います。
2. 学習側はコマンドを実行し、結果を `commands/done/<名前>.json` に書いて、コマンドファイルを削除します。

`sample` の `overrides` には、プロンプトのキー（`width`、`height`、`seed`、`sample_steps`、`scale`、`negative_prompt` など）を使います。

結果ファイルの `status` は `ok`、`error`（`error` にメッセージ）、`expired` のいずれかです。`step` と `epoch` はコマンドを実行した時点の値です。

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
  "last_command": {"file": "1795094752976095000_save.json", "command": "save", "status": "ok", "step": 100, "time": 1795079990.11},
  "anomalies": {},
  "error": null
}
```

| Field | Description |
|---|---|
| `state` | `starting` (before the first step, including the first sample generation), `running`, `paused` (by the `pause` command, see `paused_since`), `finished`, `stopped` (ended by the `stop` command), `failed` (ended by an exception, see `error`), `interrupted` (Ctrl+C), `aborted` (exited before the training finished without an exception) |
| `updated_at` | Unix time of the last update. All times in the files are Unix time in seconds |
| `progress` | `global_step` is the number of optimizer steps. `epoch` is the current epoch, starting from 1 (`0` before the first epoch) |
| `loss` | Loss of the last batch and its moving average (the same values as the progress bar and TensorBoard). `null` if the value is not finite |
| `lr` | Learning rates by parameter group, the same as `lr/*` in TensorBoard |
| `metrics` | Other step logs for TensorBoard, e.g. `norm/*`, `max_norm/*` |
| `speed` | `steps_per_sec` and `sec_per_step` are measured since the previous update of the file, so they include the time for sampling and saving in that period. `eta_sec` is from the average since the first step |
| `timing` | Time spent waiting for the next batch from the DataLoader, per batch and as the ratio to the total time. A high ratio means that data loading is the bottleneck |
| `vram` | Memory of each device used by this process, as reported by PyTorch (`memory_allocated`, `memory_reserved`, `max_memory_allocated`). `device` is the device name in PyTorch, e.g. `cuda:0`. CUDA, XPU and MPS are supported; a value which the backend does not provide is `null` (MPS has no peak), and the list is empty on CPU |
| `last_checkpoint`, `last_state`, `last_sample` | The last saved model, the last saved state and the last generated sample images |
| `last_command` | The last processed command: the file name, the command, the status (`ok`, `error`, `expired`) and the step |
| `anomalies` | Count by kind. Currently `non_finite_loss` (the loss was NaN or infinity) |
| `error` | `type`, `message` and the last lines of `traceback` when the state is `failed` or `interrupted` |

The status does not change while a long operation between steps (sample generation, saving, validation) is running: `state` stays `running` and `updated_at` stays old.

If the process is killed (e.g. from the task manager, or by the out-of-memory killer of the OS), the final status cannot be written and `state` stays `running`. A reader can detect this by `updated_at` being old, or by checking whether the process of `pid` still exists.

| フィールド | 説明 |
|---|---|
| `state` | `starting`（最初のステップの前。最初のサンプル生成を含む）、`running`、`paused`（`pause` コマンドで一時停止中。`paused_since` を参照）、`finished`、`stopped`（`stop` コマンドで終了）、`failed`（例外で終了。`error` を参照）、`interrupted`（Ctrl+C）、`aborted`（例外なしで学習完了前に終了） |
| `updated_at` | 最後に更新した Unix 時刻。ファイル内の時刻はすべて秒単位の Unix 時刻です |
| `progress` | `global_step` は optimizer のステップ数。`epoch` は現在の epoch で 1 から始まります（最初の epoch の前は `0`） |
| `loss` | 最後のバッチの loss とその移動平均（プログレスバーや TensorBoard と同じ値）。有限の値でないときは `null` |
| `lr` | パラメータグループごとの学習率。TensorBoard の `lr/*` と同じです |
| `metrics` | TensorBoard 向けのその他のステップログ（`norm/*`、`max_norm/*` など） |
| `speed` | `steps_per_sec` と `sec_per_step` はファイルの前回更新からの計測値で、その間のサンプル生成や保存の時間を含みます。`eta_sec` は最初のステップからの平均で計算します |
| `timing` | DataLoader から次のバッチを待った時間（バッチあたり、および全体に対する割合）。割合が高い場合はデータ読み込みがボトルネックです |
| `vram` | このプロセスが使っている各デバイスのメモリ（PyTorch の `memory_allocated`、`memory_reserved`、`max_memory_allocated`）。`device` は `cuda:0` のような PyTorch 上のデバイス名です。CUDA、XPU、MPS に対応しています。バックエンドが提供しない値は `null` になり（MPS にはピーク値がありません）、CPU では空のリストになります |
| `last_checkpoint`、`last_state`、`last_sample` | 最後に保存したモデル、最後に保存した state、最後に生成したサンプル画像 |
| `last_command` | 最後に処理したコマンド: ファイル名、コマンド、status（`ok`、`error`、`expired`）、ステップ |
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
| `paused`, `resumed` | `paused_sec` (for `resumed`) | The training is paused / resumed by a command |
| `command_received` | `file`, `command` | A command is read and its execution starts |
| `command_finished` | `file`, `command`, `status`, `result` or `error` | A command is processed. The same content as the result file |
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
| `paused`、`resumed` | `paused_sec`（`resumed` のみ） | コマンドによる学習の一時停止 / 再開 |
| `command_received` | `file`、`command` | コマンドを読み取り、実行を開始 |
| `command_finished` | `file`、`command`、`status`、`result` または `error` | コマンドの処理完了。結果ファイルと同じ内容です |
| `run_end` | `state`、`error` | 学習の終了。`state` と `error` は `status.json` と同じです |

## loss.jsonl

The loss of each training sample, one line per sample. With this log the training can be examined per image, per resolution, per timestep and so on after (or during) the training. The recording is done in the training; the analysis is left to external tools (see the next section).

The first line of each run has `run_id` only. The lines after it belong to that run.

```json
{"run_id": "20261119-213416-23004", "ts": 1795091656.95}
{"step": 1, "epoch": 1, "image": "/path/to/img0.png", "dataset": 0, "subset": 0, "timestep": 0.6127, "loss": 0.1795, "width": 640, "height": 384, "batch_size": 2, "loss_weight": 1.0}
```

| Field | Description |
|---|---|
| `step` | The optimizer step to which the sample contributes, starting from 1. With gradient accumulation, several batches have the same step |
| `epoch` | Epoch, starting from 1 |
| `image` | Key of the image in the dataset: the path of the image file, or the key in the metadata file |
| `dataset`, `subset` | Index of the dataset, and index of the subset in the dataset, from 0. They are the positions in `metadata.ss_datasets` of `manifest.json`: `ss_datasets[dataset].subsets[subset]` has the settings of the subset (`image_dir`, `num_repeats`, `is_reg` ...). `null` if they are not available |
| `timestep` | Timestep of the sample. The scale depends on the model (e.g. 0 to 1000 for SD / SDXL, 0 to 1 for Anima). `null` if it is not available per sample |
| `loss` | Loss of the sample, averaged over the pixels (tokens) of the sample. The weighting by the timestep of the model (`--weighting_scheme`) and the masked loss are included; `loss_weight` and the weightings applied after it (e.g. `--min_snr_gamma`) are not. `null` if the value is not finite |
| `width`, `height` | Resolution of the sample (the bucket resolution) in pixels |
| `batch_size` | Number of samples in the batch of the sample |
| `loss_weight` | Weight of the sample in the batch: `--prior_loss_weight` for regularization images, 1 for the others |

The lines are written together with `status.json`, about every 5 seconds, so the last few steps may be missing if the process is killed. The size of the file is about 200 bytes plus the length of the image path per sample (e.g. about 10MB for 10,000 steps with batch size 4).

Each image is trained only once per epoch (times the number of repeats) with a random timestep, so the loss of one image is noisy. Take the number of the lines of the image into account when comparing images.

学習した各サンプルの loss を、1 サンプル 1 行で記録します。このログにより、学習後（または学習中）に、画像別、解像度別、timestep 別などで学習の様子を調べられます。記録は学習側で行い、分析は外部のツールに任せます（次の節を参照）。

各実行の最初の行は `run_id` だけを持ちます。それ以降の行がその実行のものです。

| フィールド | 説明 |
|---|---|
| `step` | そのサンプルが寄与する optimizer のステップ（1 から）。gradient accumulation では複数のバッチが同じステップになります |
| `epoch` | epoch（1 から） |
| `image` | データセット内の画像のキー。画像ファイルのパス、またはメタデータファイル内のキーです |
| `dataset`、`subset` | データセットの番号と、そのデータセット内での subset の番号（0 から）。`manifest.json` の `metadata.ss_datasets` 内の位置に対応し、`ss_datasets[dataset].subsets[subset]` にその subset の設定（`image_dir`、`num_repeats`、`is_reg` など）があります。得られない場合は `null` |
| `timestep` | そのサンプルの timestep。スケールはモデルによって異なります（SD / SDXL は 0〜1000、Anima は 0〜1 など）。サンプル単位で得られない場合は `null` |
| `loss` | そのサンプルの loss（サンプル内の画素（トークン）で平均した値）。モデルの timestep による重みづけ（`--weighting_scheme`）と masked loss は含まれ、`loss_weight` とその後に適用される重みづけ（`--min_snr_gamma` など）は含まれません。有限の値でないときは `null` |
| `width`、`height` | そのサンプルの解像度（バケットの解像度）。ピクセル単位です |
| `batch_size` | そのサンプルが属するバッチのサンプル数 |
| `loss_weight` | バッチ内でのサンプルの重み。正則化画像は `--prior_loss_weight`、それ以外は 1 です |

行は `status.json` と同時（約 5 秒ごと）に書き込まれるため、プロセスが強制終了された場合は最後の数ステップ分が欠けることがあります。ファイルサイズは 1 サンプルあたり約 200 バイトに画像パスの長さを加えた程度です（バッチサイズ 4 で 10,000 ステップなら約 10MB）。

各画像は 1 epoch に 1 回（× repeats 回）、ランダムな timestep でしか学習されないため、1 枚の画像の loss はばらつきが大きくなります。画像同士を比べるときは、その画像の行数も考慮してください。

## Analyzing the Loss Log / loss ログの分析

`tools/analyze_run_loss.py` summarizes `loss.jsonl`. It only reads the file and needs only the Python standard library, so it can also be run while the training is running, or on another machine.

```bash
python tools/analyze_run_loss.py output/my_lora_run
python tools/analyze_run_loss.py output/my_lora_run --top 50 --min_count 5
python tools/analyze_run_loss.py output/my_lora_run --json > loss_summary.json
```

### What the residual is

The loss of a sample depends strongly on the timestep (the noise level) which was randomly drawn for it. For example, the loss at one timestep may be around 0.15 for every image, and around 0.35 at another. A loss of 0.30 is therefore high at the former timestep and low at the latter, and an image may look "hard" only because it happened to get timesteps with a high loss. The raw loss cannot be compared between samples or between images.

The script uses the *residual* instead. It answers the question "how high is this loss compared with the other samples at a similar timestep?":

1. The range of the timesteps is divided into bins (10 by default), and the mean loss of all samples in each bin is calculated.
2. The residual of a sample is `loss / (mean loss of its bin) - 1`.

| Residual | Meaning |
|---|---|
| `0` | The same as the average at that noise level |
| `+0.5` | 50% higher than the average at that noise level (harder than the others) |
| `-0.2` | 20% lower than the average at that noise level (easier than the others) |

For example, if the mean loss of a bin is 0.20, a sample with the loss 0.30 in that bin has the residual `0.30 / 0.20 - 1 = +0.5`. The `mean_residual` of an image, a resolution and so on is the mean of the residuals of its samples.

The residual is relative to the other samples of the same training: it tells which images are harder or easier than the others, not whether the training as a whole is going well. It is also still noisy when the number of the samples is small, because the timestep is not the only random factor (the noise itself is random too).

It prints:

- The loss by timestep, by resolution, by batch size, by subset (`dataset-subset`) and by directory of the images.
- The images with the highest and the lowest mean residual. `slope` is the change of the residual per epoch (calculated when the image has three or more epochs). A high residual which does not go down may indicate a problem of the image or its caption, for example a mismatch between them; a high residual which is going down is an image that is being learned.

| Option | Description |
|---|---|
| `--run_id` | Run to analyze. The default is the last run in the file |
| `--all_runs` | Analyze all runs in the file together, e.g. for a resumed training |
| `--top` | Number of images in the highest / lowest lists (default 20) |
| `--min_count` | Images with fewer lines than this are not listed (default 3) |
| `--timestep_bins` | Number of timestep bins for the residual (default 10) |
| `--json` | Print the full result, including all images, as JSON |

This is one way to look at the log, intended as a starting point. The log is plain JSON Lines, so any other analysis is easy to write, or to ask an AI agent for.

`tools/analyze_run_loss.py` は `loss.jsonl` を集計します。ファイルを読むだけで、Python の標準ライブラリしか使わないため、学習の実行中や別のマシンでも実行できます。

### 残差とは

サンプルの loss は、そのサンプルにランダムに割り当てられた timestep（ノイズの強さ）に強く依存します。たとえば、ある timestep ではどの画像でも loss が 0.15 前後、別の timestep では 0.35 前後、ということが起こります。このとき 0.30 という loss は、前者の timestep では高く、後者では低い値です。たまたま loss の高い timestep が割り当てられただけの画像が「難しい画像」に見えてしまうため、生の loss はサンプル間や画像間で比較できません。

そこで、このスクリプトは代わりに *残差* を使います。「同じくらいの timestep の他のサンプルと比べて、この loss はどのくらい高いか」を表す値です。

1. timestep の範囲をいくつかの帯（デフォルトは 10）に分け、帯ごとに全サンプルの平均 loss を求めます。
2. サンプルの残差は `loss / (そのサンプルが属する帯の平均 loss) - 1` です。

| 残差 | 意味 |
|---|---|
| `0` | そのノイズレベルの平均と同じ |
| `+0.5` | そのノイズレベルの平均より 50% 高い（他より難しい） |
| `-0.2` | そのノイズレベルの平均より 20% 低い（他より易しい） |

たとえば、ある帯の平均 loss が 0.20 のとき、その帯で loss が 0.30 だったサンプルの残差は `0.30 / 0.20 - 1 = +0.5` です。画像や解像度などの `mean_residual` は、それに属するサンプルの残差の平均です。

残差は、同じ学習の中の他のサンプルとの相対的な値です。どの画像が他より難しいか易しいかは分かりますが、学習全体がうまく進んでいるかどうかは分かりません。また、ランダムな要素は timestep だけではない（ノイズそのものもランダム）ため、サンプル数が少ないうちは残差もばらつきます。

出力内容:

- timestep 別、解像度別、バッチサイズ別、subset 別（`データセット番号-subset 番号`）、画像のディレクトリ別の loss。
- 残差が最も高い画像と最も低い画像。`slope` は 1 epoch あたりの残差の変化です（その画像に 3 epoch 以上の記録があるときに計算します）。残差が高いまま下がらない画像は、画像またはキャプションに問題がある可能性があります（両者の不一致など）。残差が高くても下がっている画像は、学習が進んでいる途中の画像です。

| オプション | 説明 |
|---|---|
| `--run_id` | 分析する実行。省略時はファイル内の最後の実行 |
| `--all_runs` | ファイル内のすべての実行をまとめて分析します（学習を再開した場合など） |
| `--top` | 残差の高い / 低い画像の一覧に表示する枚数（デフォルト 20） |
| `--min_count` | 行数がこれより少ない画像は一覧に表示しません（デフォルト 3） |
| `--timestep_bins` | 残差の計算に使う timestep 帯の数（デフォルト 10） |
| `--json` | すべての画像を含む結果全体を JSON で出力します |

これはログの見方のひとつで、出発点として用意したものです。ログは通常の JSON Lines なので、他の分析も簡単に書けますし、AI エージェントに依頼することもできます。

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
- `events.jsonl` and `loss.jsonl` are only appended. A reader can remember the position and read the new lines. The last line may be incomplete while it is being written.
- Errors in writing these files never stop the training. A warning is shown once.

- `manifest.json` と `status.json` は一時ファイルに書いてからリネームするため、書きかけのファイルが読まれることはありません。
- Windows では、他のプロセスが `status.json` を開いている間はリネームに失敗します。その場合は更新をスキップして次のステップで再試行するため、ファイルは短時間だけ開く（読んだらすぐ閉じる）ようにしてください。
- `events.jsonl` と `loss.jsonl` は追記のみです。読む側は位置を覚えておき、新しい行だけを読めます。書き込み中は最後の行が不完全なことがあります。
- これらのファイルの書き込みエラーで学習が止まることはありません。警告が 1 回表示されます。
