# Guide for AI Agents Operating a Training

This document is for an AI coding agent (Claude Code, Codex, Gemini CLI and so on) which starts a training with a user and then watches it. It is not needed for working on the code of this repository.

It covers the procedure and the pitfalls. The reference for the file formats and the commands is [run_control.md](./run_control.md); read the relevant section there when you need the details of a field or an option.

Scope: the LoRA training scripts based on `train_network.py` (`train_network.py`, `sdxl_train_network.py`, `sd3_train_network.py`, `flux_train_network.py`, `lumina_train_network.py`, `hunyuan_image_train_network.py`, `anima_train_network.py`, `moss_image_train_network.py`), in a single process. The feature is experimental; check `schema_version` in the files (this document is written for `1`).

## How it works

With `--run_control`, the training writes its state to plain files in the *run directory* `<output_dir>/<output_name>_run`, and executes commands put there.

| File | Use it for |
|---|---|
| `status.json` | The current state: step, loss, speed, ETA, VRAM, last outputs. Overwritten about every 5 seconds |
| `events.jsonl` | What happened: checkpoints, sample images, commands, the end of the run. Appended |
| `manifest.json` | How the training was started: arguments, dataset summary, environment. Written once |
| `loss.jsonl` | The loss of every training sample. Appended. Read it with `tools/analyze_run_loss.py`, not directly |

`tools/run_control_cli.py` and `tools/analyze_run_loss.py` need only the Python standard library. Any Python can run them; the environment of the training is not needed.

## Starting a training so that it can be watched

The options of the training itself are described in the document of each model (`docs/*_train_network.md`). The points below are the ones that matter for watching it.

- **Add `--run_control`.** Without it there is no run directory and no command.
- **Use a new `output_name` for each run**, or move the old run directory away first. The run directory is reused when `output_dir` and `output_name` are the same: `manifest.json` and `status.json` are overwritten, and the two `.jsonl` files are appended.
- **Put every prompt that may be wanted later into the prompt file (`--sample_prompts`).** When the text encoder outputs are cached and the text encoder is not kept in memory (the usual setting for Anima, MOSS-Image and FLUX.1), a running training can only generate samples from the prompts of the prompt file. Ask the user for the prompts before starting. A prompt with `--s 0` (0 steps) is a *reserved prompt*: it is skipped by the scheduled sampling, but it is encoded at the start and can be generated later with `sample --index`. Use it for the prompts which are not needed at every scheduled sampling. The prompt file can be given without scheduled sampling (`--sample_every_n_steps` / `--sample_every_n_epochs`); it is then used only by the `sample` command.
- **Start the process so that it outlives your shell command**, as a background or detached job, with stdout and stderr redirected to a log file. A training takes from minutes to days; do not run it in a foreground command which has a timeout.
- **Add `--console_log_simple`** when the output is redirected. The log lines are then plain text, without the formatting for a terminal. Do not rely on `--console_log_file` instead of the redirection: it receives the log messages only, not the traceback of an exception, the progress bar or the errors of the argument parser.
- **Agree with the user on how you will watch**, in one short exchange before the user leaves: how often you report, and what you may do without asking. Propose the defaults of this document (report only; `sample` / `save` when it helps; never `pause` or `stop` on your own) and let the user change them. If the user wants you to act while they are away (e.g. "stop it if the loss becomes NaN"), get that condition in the user's words; such a condition counts as the user's request.
- **Keep the path of the run directory and of the log file.** Tell them to the user, so that another session can pick up the watching.

### The first minutes

The run directory is written only after the models are loaded and the caching of the latents and the text encoder outputs is done. This can take a long time for a large dataset.

1. Until `status.json` appears (or its `run_id` changes, if the run directory is reused), the only source of information is the log file. Errors in this period (wrong arguments, a missing file, out of memory while loading) are in the log file only.
2. `state` is `starting` until the first step is done. This includes the first sample generation if `--sample_at_first` is set.
3. When `state` is `running` and `progress.global_step` increases, the training is going. Report to the user: the speed, the ETA (`speed.eta_sec`), and the peak VRAM (`vram[].peak_allocated_mb`).

If the process ends before `status.json` is written, read the end of the log file and report the error.

## Checking on a running training

```bash
python tools/run_control_cli.py <run directory> status
```

Check in this order:

1. **Is it the run you started?** `run_id` and `pid` are those of the run. A `status.json` of a previous run stays in a reused run directory until the new run writes its own.
2. **Is it alive?** `state` is `running` (or `paused`), and `updated_at` is recent. See the pitfalls below for the cases where `updated_at` is old but the training is alive, and where `state` is `running` but the process is gone.
3. **Progress**: `progress.global_step` / `progress.max_train_steps`, `speed.eta_sec`.
4. **Problems**: `anomalies` (e.g. `non_finite_loss`: the loss was NaN or infinity), `error`, and `timing.data_wait_ratio` (a high ratio means that the data loading, not the GPU, limits the speed).
5. **New outputs**: `last_checkpoint`, `last_sample`. For the full list since the last check, read the new lines of `events.jsonl` (remember the number of lines you have read; the last line may be incomplete while it is being written).

When new sample images are generated (`samples_generated` in `events.jsonl`, with `files`), look at them if you can view images. The sample images tell more about the result than the loss does: the loss of a diffusion model training is noisy and depends on the timesteps which were drawn, and it often does not go down visibly even when the training is going well.

**How often**: `status.json` is updated every few seconds, but a check every few minutes is enough, and less often for a long training. Use the mechanism of your environment to wait (a scheduled wake-up, or a notification when the background process ends) instead of a loop of short sleeps. Checking does not affect the training.

**What to report**: the progress and the ETA, new sample images and checkpoints, and anything from "Problems" above. State what you read from the files; when you draw a conclusion from the sample images or the loss, say that it is your reading.

## Acting on a running training

```bash
python tools/run_control_cli.py <run directory> sample --index 0 2 --seed 42
python tools/run_control_cli.py <run directory> save
python tools/run_control_cli.py <run directory> save_state
python tools/run_control_cli.py <run directory> pause
python tools/run_control_cli.py <run directory> resume
python tools/run_control_cli.py <run directory> stop
```

The CLI waits for the result and prints it as JSON. The exit code is 0 if the command is done, 1 if it failed or was not executed, 2 if the result did not come within `--timeout`. A command is executed after the next optimizer step, and a `sample` takes as long as the sample generation does, so the wait can be long. A timeout does not cancel the command; it is still executed later.

| Command | What it does to the training | When |
|---|---|---|
| `status` | Nothing | Any time |
| `sample`, `save`, `save_state` | The training waits while the command is executed, then continues unchanged. They use time and disk space | When the user asks, or to answer a question of the user (e.g. "how does it look now?"). Reasonable to do on your own a few times; do not repeat them in a loop |
| `pause` / `resume` | The training stops making progress until `resume`. The process stays | Only when the user asks or agrees, or under a condition the user gave in advance |
| `stop` | The training ends. The model and the state are saved with the name of the current step | Only when the user asks or agrees, or under a condition the user gave in advance |

**Decisions about the training belong to the user.** Whether the training should be stopped early, continued, or started again with other settings depends on what the user wants from the model. If something looks wrong (the sample images are broken, the loss is NaN, the speed is far lower than expected), report what you see and ask. There is no rule here such as "stop if the loss is above X", because no such rule is reliable.

After a command, the result is also in `commands/done/` and in `events.jsonl` (`command_finished`).

## After the training ends

`state` in `status.json` (and `run_end` in `events.jsonl`) tells how it ended.

| `state` | Meaning | Next |
|---|---|---|
| `finished` | Completed. The final model is `<output_dir>/<output_name>.safetensors` | Report the final model, the checkpoints and the sample images |
| `stopped` | Ended by the `stop` command. There is no final model; the model and the state have the name of the step (see `last_checkpoint`, `last_state`) | To continue, start the same command again with `--resume <the state directory>` |
| `failed` | Ended by an exception | Read `error` (`type`, `message`, `traceback`) and the log file, and report. Do not start it again without knowing the cause |
| `interrupted` | Ctrl+C | Ask the user |
| `aborted` | Exited before the end without an exception | Read the log file |
| `running`, but the process is gone | Killed (task manager, the out-of-memory killer of the OS, a reboot). The final status could not be written | Read the log file and the logs of the OS |

To look into the training afterwards (or while it is running):

```bash
python tools/analyze_run_loss.py <run directory>
```

It shows the loss by timestep, by resolution and by subset, and the images whose loss is high or low compared with the other images at a similar timestep (the *residual*). A high residual which does not go down may indicate a problem of the image or its caption. It is relative to the other images of the same training, and noisy for an image with few samples: use it to pick the images to look at, not as a verdict. See [Analyzing the Loss Log](./run_control.md#analyzing-the-loss-log--loss-ログの分析).

## Pitfalls

Each of these has a reason, but none is obvious.

- **`updated_at` stops during a long operation between steps.** Sample generation, saving and validation do not update `status.json`; `state` stays `running`. An old `updated_at` alone does not mean that the training is dead. Check whether the process of `pid` exists, and wait for the time a sample generation or a save takes.
- **`state` stays `running` if the process is killed.** The final status is written by the process itself. If `updated_at` is old and the process of `pid` does not exist, the training is dead.
- **A previous run leaves its files.** In a reused run directory, `status.json` shows the previous run (e.g. `finished`) until the new run reaches the start of the training. Compare `run_id`. In `events.jsonl` and `loss.jsonl`, the lines of all runs are in one file; filter by `run_id` (`analyze_run_loss.py` uses the last run by default).
- **Do not keep `status.json` open.** On Windows, the training cannot replace the file while another process has it open, and skips the update. Read it and close it at once (the CLI does this). Do not follow it with a tool which keeps the file open.
- **`stop` does not make the final model.** `<output_name>.safetensors` exists only when the training completed. After `stop`, the model is `<output_name>-step<N>.safetensors`; the path is in the result of the command.
- **A prompt which is not in the prompt file may be rejected.** `sample --prompt` (and `--negative_prompt`) with a new text needs the text encoder in the training process. If it is not there, the command returns an error and the training continues. For Anima and MOSS-Image, `tools/run_control_sample_prompt.py <run directory> --prompt "..."` generates such a prompt: it encodes the prompt in its own process (on CPU by default) and sends the result to the training. It needs the Python environment of the training, and the training must have been started with a prompt file. `sample --index` with `--seed`, `--steps`, `--width`, `--height` and `--scale` always works.
- **Do not edit the prompt file while the training is running.** The `sample` command reads the file again each time, but the prompts were encoded at the start: a changed or added prompt is rejected in the same way, and the indices shift.
- **`pause` does not work in every configuration**, and does not free everything. It is rejected with `--block_swap_h2d_only`, replica training (`--replica_devices`) and DeepSpeed; the training continues. The LoRA, the state of the optimizer and the CUDA context stay on the GPU, and the models need RAM while they are on CPU. Check `vram_after` in the result before telling the user that the GPU is free.
- **While paused, only `resume` and `stop` are executed.** Other commands wait until the training is resumed, and the CLI waits with them.
- **Commands are not accepted in multi-process training** (`accelerate launch --multi_gpu`). A command sent before the training started, or left when it ended, is not executed (`expired`).
- **An unexpected error in a command ends the training**, the same as an error in the scheduled sampling or saving. For example, a `sample` with a much larger resolution than the training can run out of memory. Keep the size of on-demand samples close to that of the scheduled ones.
- **`manifest.json` and `loss.jsonl` have local paths** (model, dataset, images, output), and the log may have them too. Do not paste them into an issue, a web service or any other place outside the machine without the consent of the user.
