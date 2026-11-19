"""File-based observation of a running training process (``--run_control``).

A run directory ``<output_dir>/<output_name>_run`` receives plain files that a human, a script or an AI agent
can read while the training is running, without parsing TensorBoard event files:

- ``manifest.json``: written once at start. Resolved arguments, dataset summary, environment.
- ``status.json``: current snapshot (state, step, loss, lr, speed, ETA, VRAM, last outputs). Overwritten every few seconds.
- ``events.jsonl``: append-only log, one JSON object per line (run start/end, checkpoints, states, samples, anomalies).
- ``loss.jsonl``: append-only log of the loss of each training sample (image, timestep, loss, size). The analysis is
  left to external tools, e.g. ``tools/analyze_run_loss.py``.

Everything here is observation only and must never break the training: I/O errors are logged once and ignored.
Only the main process writes; other processes get a disabled controller, whose methods do nothing.

See ``docs/run_control.md`` for the file formats.
"""

import argparse
import atexit
import datetime
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
import traceback
from typing import Any, Optional

import torch

import library.device_utils as device_utils
from library.checkpoint_io import DEFAULT_LAST_OUTPUT_NAME
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


SCHEMA_VERSION = 1
MB = 1024 * 1024
RUN_DIR_SUFFIX = "_run"
MANIFEST_FILE = "manifest.json"
STATUS_FILE = "status.json"
EVENTS_FILE = "events.jsonl"
LOSS_FILE = "loss.jsonl"

# key of the batch dict in which the step and the epoch of the batch are kept for the per-sample loss log
BATCH_TAG_KEY = "run_control_step_epoch"

SENSITIVE_ARGS = ["wandb_api_key", "huggingface_token"]
REDACTED = "***"

# The tag frequency is large and already stored in the model metadata, so it is not copied to the manifest.
# It is also removed from each dataset in ss_datasets.
MANIFEST_EXCLUDED_METADATA_KEYS = ["ss_tag_frequency"]

PACKAGES_IN_MANIFEST = [
    "torch",
    "torchvision",
    "accelerate",
    "diffusers",
    "transformers",
    "safetensors",
    "numpy",
    "bitsandbytes",
    "xformers",
    "prodigyopt",
    "schedulefree",
]

TRACEBACK_MAX_LINES = 30
SAMPLE_DIR_SETTLE_NS = 2 * 1000**3


def add_run_control_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--run_control",
        action="store_true",
        help="write machine-readable manifest, status and event files of the training to `<output_dir>/<output_name>_run`,"
        " so that the training can be observed from outside (e.g. by scripts or AI agents)"
        " / 学習のマニフェスト、状態、イベントを機械可読なファイルとして `<output_dir>/<output_name>_run` に書き出し、"
        "外部（スクリプトやAIエージェントなど）から学習を観測できるようにする",
    )


def get_run_dir(args: argparse.Namespace) -> str:
    model_name = DEFAULT_LAST_OUTPUT_NAME if args.output_name is None else args.output_name
    return os.path.join(args.output_dir, model_name + RUN_DIR_SUFFIX)


def to_mb(num_bytes: Optional[int]) -> Optional[int]:
    return None if num_bytes is None else num_bytes // MB


def to_json_safe(obj: Any) -> Any:
    """Convert to something `json.dump` writes as strict JSON. Non-finite floats become None, unknown objects become str."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, torch.Tensor) and obj.numel() == 1:
        return to_json_safe(obj.item())
    if hasattr(obj, "item") and callable(obj.item):  # numpy scalar
        try:
            return to_json_safe(obj.item())
        except Exception:
            pass
    return str(obj)


def write_json_atomic(path: str, obj: Any) -> bool:
    """Write to a temporary file and rename, so that readers never see a partial file.

    Returns False if it could not be written, e.g. because another process has the file open on Windows.
    The caller is expected to retry later.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(to_json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(tmp_path, path)
        return True
    except OSError:
        return False


def redact_args(args: argparse.Namespace) -> dict:
    redacted = {}
    for k, v in vars(args).items():
        redacted[k] = REDACTED if (k in SENSITIVE_ARGS and v is not None) else v
    return redacted


def redact_command_line(argv: list[str]) -> list[str]:
    redacted = []
    redact_next = False
    for arg in argv:
        if redact_next:
            redacted.append(REDACTED)
            redact_next = False
            continue
        if arg.startswith("--"):
            name, sep, _ = arg[2:].partition("=")
            if name in SENSITIVE_ARGS:
                if sep:
                    redacted.append(f"--{name}={REDACTED}")
                else:
                    redacted.append(arg)
                    redact_next = True
                continue
        redacted.append(arg)
    return redacted


def decode_metadata(metadata: dict) -> dict:
    """The model metadata keeps nested values as JSON strings and numbers as strings. Decode the nested ones for readability."""
    decoded = {}
    for k, v in metadata.items():
        if k in MANIFEST_EXCLUDED_METADATA_KEYS:
            continue
        if isinstance(v, str) and v[:1] in ("{", "["):
            try:
                v = json.loads(v)
            except ValueError:
                pass
        if k == "ss_datasets" and isinstance(v, list):
            # the tag frequency is also in each dataset
            v = [{dk: dv for dk, dv in d.items() if dk != "tag_frequency"} if isinstance(d, dict) else d for d in v]
        decoded[k] = v
    return decoded


def get_environment_info() -> dict:
    packages = {}
    for name in PACKAGES_IN_MANIFEST:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass

    gpus = []
    for d in device_utils.get_device_descriptions():
        gpus.append(
            {"device": d["device"], "name": d["name"], "total_memory_mb": to_mb(d["total_memory"])}
        )

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda": torch.version.cuda,
        "packages": packages,
        "gpus": gpus,
    }


def build_manifest(args: argparse.Namespace, metadata: Optional[dict] = None, training: Optional[dict] = None) -> dict:
    """Build the content of manifest.json.

    Args:
        args: the resolved arguments (command line + config file). Secrets are redacted.
        metadata: the training metadata to be stored in the model (`ss_*`), which has the dataset summary.
        training: numbers determined at runtime, e.g. `max_train_steps`, `num_train_epochs`.
    """
    import library.model_io as model_io  # heavy import, only needed here

    return {
        "script": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else None,
        "command_line": redact_command_line(sys.argv[1:]),
        "cwd": os.getcwd(),
        "sd_scripts_commit_hash": model_io.get_git_revision_hash(),
        "training": training or {},
        "outputs": {
            "output_dir": os.path.abspath(args.output_dir),
            "output_name": args.output_name,
            "sample_dir": os.path.abspath(get_sample_dir(args)),
            "logging_dir": os.path.abspath(args.logging_dir) if getattr(args, "logging_dir", None) else None,
        },
        "environment": get_environment_info(),
        "args": redact_args(args),
        "metadata": decode_metadata(metadata) if metadata is not None else {},
    }


def get_sample_dir(args: argparse.Namespace) -> str:
    # same as the sampling functions of each model
    return args.output_dir + "/sample"


def create_run_controller(args: argparse.Namespace, is_main_process: bool = True) -> "RunController":
    """Returns an enabled controller only if `--run_control` is specified and this is the main process."""
    if not getattr(args, "run_control", False) or not is_main_process:
        return RunController(None)
    return RunController(get_run_dir(args), sample_dir=get_sample_dir(args))


class RunController:
    """Writes the status and the events of a training run. All methods are no-op if `run_dir` is None."""

    def __init__(self, run_dir: Optional[str], *, sample_dir: Optional[str] = None, status_interval: float = 5.0):
        self.run_dir = run_dir
        self.sample_dir = sample_dir
        self.status_interval = status_interval

        self.run_id: Optional[str] = None
        self._pid = os.getpid()
        self._started = False
        self._finished = False
        self._warned_io_error = False
        self._error: Optional[dict] = None
        self._prev_excepthook = None

        self._status: dict = {}
        self._epoch = 0  # current epoch, 1-based. 0 before the first epoch
        self._last_status_write = 0.0

        # speed and timing
        self._first_step: Optional[int] = None
        self._first_step_time = 0.0
        self._window_step = 0
        self._window_time = 0.0
        self._window_batches = 0
        self._window_data_wait = 0.0
        self._last_step_end: Optional[float] = None

        # per-sample losses, not yet written. (step, epoch, image keys, sizes, loss weights, timestep tensor, loss tensor)
        self._pending_losses: list[tuple] = []

        # sample detection
        self._known_samples: set[str] = set()
        self._sample_dir_mtime: Optional[int] = None
        self._sample_dir_scan_time = 0

    @property
    def enabled(self) -> bool:
        return self.run_dir is not None

    # region file writing

    def _warn_io_error(self, message: str):
        if not self._warned_io_error:
            self._warned_io_error = True
            logger.warning(f"run control: {message}. Training continues. This warning is shown only once.")

    def log_event(self, event: str, step: Optional[int] = None, **fields):
        """Append one line to events.jsonl."""
        if not self.enabled or not self._started:
            return
        now = time.time()
        record = {
            "ts": now,
            "time": datetime.datetime.fromtimestamp(now).astimezone().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "event": event,
            "step": step,
            "epoch": self._epoch,
        }
        record.update(fields)
        self._append_lines(EVENTS_FILE, [record])

    def _append_lines(self, file_name: str, records: list[dict]):
        try:
            with open(os.path.join(self.run_dir, file_name), "a", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(to_json_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
        except OSError as e:
            self._warn_io_error(f"failed to write to {file_name}: {e}")

    def _write_status(self, force: bool = False):
        if not self.enabled or not self._started:
            return
        now = time.time()
        if not force and now - self._last_status_write < self.status_interval:
            return
        self._flush_sample_losses()
        self._status["updated_at"] = now
        self._status["vram"] = self._get_vram()
        # if this fails (another process may be reading the file on Windows), it is retried at the next step
        if write_json_atomic(os.path.join(self.run_dir, STATUS_FILE), self._status):
            self._last_status_write = now

    @staticmethod
    def _get_vram() -> list[dict]:
        vram = []
        for stats in device_utils.get_device_memory_stats():
            vram.append(
                {
                    "device": stats["device"],
                    "allocated_mb": to_mb(stats["allocated"]),
                    "reserved_mb": to_mb(stats["reserved"]),
                    "peak_allocated_mb": to_mb(stats["peak_allocated"]),
                }
            )
        return vram

    # endregion

    # region lifecycle

    def start(self, manifest: dict, *, max_train_steps: Optional[int] = None, num_train_epochs: Optional[int] = None):
        """Create the run directory, write manifest.json and the first status.json."""
        if not self.enabled or self._started:
            return

        now = time.time()
        self.run_id = f"{datetime.datetime.fromtimestamp(now):%Y%m%d-%H%M%S}-{self._pid}"

        try:
            os.makedirs(self.run_dir, exist_ok=True)
        except OSError as e:
            self._warn_io_error(f"failed to create the run directory {self.run_dir}: {e}")
            self.run_dir = None  # disable
            return

        manifest = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "created_at": now, "pid": self._pid, **manifest}
        if not write_json_atomic(os.path.join(self.run_dir, MANIFEST_FILE), manifest):
            self._warn_io_error(f"failed to write {MANIFEST_FILE}")

        self._started = True
        self._status = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "pid": self._pid,
            "state": "starting",
            "started_at": now,
            "updated_at": now,
            "progress": {"global_step": 0, "max_train_steps": max_train_steps, "epoch": 0, "num_train_epochs": num_train_epochs},
            "loss": {"current": None, "average": None},
            "lr": {},
            "metrics": {},
            "speed": {"steps_per_sec": None, "sec_per_step": None, "elapsed_sec": 0.0, "eta_sec": None},
            "timing": {"data_wait_sec_per_batch": None, "data_wait_ratio": None},
            "vram": [],
            "last_checkpoint": None,
            "last_state": None,
            "last_sample": None,
            "anomalies": {},
            "error": None,
        }

        # samples which exist before the run are not reported
        self._scan_new_samples()

        self.log_event("run_start", step=0, run_dir=os.path.abspath(self.run_dir), manifest=MANIFEST_FILE)
        # the file is appended over the runs, so each run starts with a line which has the run_id
        self._append_lines(LOSS_FILE, [{"run_id": self.run_id, "ts": now}])
        self._write_status(force=True)
        self._install_exit_hooks()
        logger.info(f"run control is enabled. run directory: {self.run_dir}")

    def finish(self, state: str = "finished"):
        """Write the final status. `state` is one of `finished`, `failed`, `interrupted`, `aborted`."""
        if not self.enabled or not self._started or self._finished:
            return
        self._finished = True
        self._status["state"] = state
        self._status["error"] = self._error
        self._update_speed(time.time(), end_of_window=False)
        self.log_event("run_end", step=self._status["progress"]["global_step"], state=state, error=self._error)
        self._write_status(force=True)

    def _install_exit_hooks(self):
        # The training loop is long and has no single exit point, so the end of the process is watched instead.
        # A hard kill cannot be caught: the status stays `running` with an old `updated_at`.
        self._prev_excepthook = sys.excepthook

        def excepthook(exc_type, exc, tb):
            lines = traceback.format_exception(exc_type, exc, tb)
            self._error = {
                "type": exc_type.__name__,
                "message": str(exc),
                "traceback": "".join(lines).splitlines()[-TRACEBACK_MAX_LINES:],
            }
            self._prev_excepthook(exc_type, exc, tb)

        sys.excepthook = excepthook
        atexit.register(self._on_exit)

    def _on_exit(self):
        if self._finished or os.getpid() != self._pid:
            return
        if self._error is None:
            state = "aborted"  # exited without an exception before the training finished
        elif self._error["type"] == "KeyboardInterrupt":
            state = "interrupted"
        else:
            state = "failed"
        self.finish(state)

    # endregion

    # region training loop hooks

    def on_epoch_start(self, *, epoch: int):
        """Call at the start of each epoch. `epoch` is 1-based. The events after this are recorded with this epoch."""
        if not self.enabled or not self._started:
            return
        self._epoch = epoch
        self._status["progress"]["epoch"] = epoch

    def on_batch_fetched(self, batch: Optional[dict] = None, *, global_step: Optional[int] = None):
        """Call at the top of the loop body, to measure the time waiting for the DataLoader.

        If `batch` and `global_step` (the number of optimizer steps done before this batch) are given, the batch is
        tagged with the step and the epoch for `record_sample_losses`. The tag is kept in the batch because the batch
        may be processed later on another thread (replica training).
        """
        if not self.enabled or not self._started:
            return
        if isinstance(batch, dict) and global_step is not None:
            batch[BATCH_TAG_KEY] = (global_step + 1, self._epoch)
        if self._last_step_end is not None:
            self._window_data_wait += time.perf_counter() - self._last_step_end
            self._window_batches += 1

    def on_step_end(
        self,
        *,
        global_step: int,
        loss: Optional[float] = None,
        average_loss: Optional[float] = None,
        logs: Optional[dict] = None,
    ):
        """Call at the end of the loop body for each batch.

        Args:
            global_step: number of optimizer steps done so far.
            loss: loss of this batch.
            average_loss: moving average of the loss.
            logs: step logs for the trackers. `lr/*` goes to `lr`, the rest except `loss/*` goes to `metrics`.
        """
        if not self.enabled or not self._started:
            return

        now = time.time()
        status = self._status
        state_changed = status["state"] != "running"
        status["state"] = "running"
        status["progress"]["global_step"] = global_step
        status["loss"]["current"] = loss
        status["loss"]["average"] = average_loss
        if logs is not None:
            status["lr"] = {k[len("lr/") :]: v for k, v in logs.items() if k.startswith("lr/")}
            status["metrics"] = {k: v for k, v in logs.items() if not k.startswith("lr/") and not k.startswith("loss/")}

        if loss is not None and not math.isfinite(loss):
            count = status["anomalies"].get("non_finite_loss", 0) + 1
            status["anomalies"]["non_finite_loss"] = count
            if count == 1:  # only the first one is logged as an event, the status has the count
                self.log_event("anomaly", step=global_step, kind="non_finite_loss", loss=str(loss))
                state_changed = True

        if self._first_step is None:
            self._first_step = global_step
            self._first_step_time = now
            self._window_step = global_step
            self._window_time = now

        if state_changed or now - self._last_status_write >= self.status_interval:
            self._update_speed(now, end_of_window=True)
            self._write_status(force=True)

        self._last_step_end = time.perf_counter()

    def _update_speed(self, now: float, end_of_window: bool):
        if self._first_step is None:
            return
        status = self._status
        global_step = status["progress"]["global_step"]
        speed = status["speed"]
        speed["elapsed_sec"] = now - status["started_at"]

        window_steps = global_step - self._window_step
        window_sec = now - self._window_time
        if window_steps > 0 and window_sec > 0:
            speed["steps_per_sec"] = window_steps / window_sec
            speed["sec_per_step"] = window_sec / window_steps
            if self._window_batches > 0:
                status["timing"]["data_wait_sec_per_batch"] = self._window_data_wait / self._window_batches
                status["timing"]["data_wait_ratio"] = min(1.0, self._window_data_wait / window_sec)

        # ETA is from the average since the first step, which includes the time for sampling and saving
        total_steps = global_step - self._first_step
        total_sec = now - self._first_step_time
        max_train_steps = status["progress"]["max_train_steps"]
        if total_steps > 0 and total_sec > 0 and max_train_steps is not None:
            speed["eta_sec"] = max(0, max_train_steps - global_step) * total_sec / total_steps

        if end_of_window and window_steps > 0:
            self._window_step = global_step
            self._window_time = now
            self._window_batches = 0
            self._window_data_wait = 0.0

    def record_sample_losses(self, batch: dict, timesteps: Optional[torch.Tensor], losses: torch.Tensor):
        """Call for each training batch with the loss of each sample, shape (B,), before it is averaged over the batch.

        This only keeps the references to the (detached) tensors. They are converted and written to loss.jsonl later
        together with the status, so that the training is not synchronized with the device here.
        This may be called from another thread.
        """
        if not self.enabled or not self._started or self._finished:
            return
        image_keys = batch.get("image_keys")
        if image_keys is None:
            return  # the dataset does not provide the keys
        step, epoch = batch.get(BATCH_TAG_KEY, (None, self._epoch))
        sizes_hw = batch.get("target_sizes_hw")
        self._pending_losses.append(
            (
                step,
                epoch,
                list(image_keys),
                sizes_hw.tolist() if isinstance(sizes_hw, torch.Tensor) and not sizes_hw.is_cuda else sizes_hw,
                batch.get("loss_weights"),
                timesteps.detach() if isinstance(timesteps, torch.Tensor) else None,
                losses.detach(),
            )
        )

    def _flush_sample_losses(self):
        if not self._pending_losses:
            return
        pending, self._pending_losses = self._pending_losses, []

        def to_list(values, n):
            if isinstance(values, torch.Tensor):
                values = values.detach().float().reshape(-1).tolist() if values.numel() == n else None
            if isinstance(values, (list, tuple)) and len(values) == n:
                return list(values)
            return [None] * n

        rows = []
        for step, epoch, image_keys, sizes_hw, loss_weights, timesteps, losses in pending:
            try:
                n = len(image_keys)
                loss_values = to_list(losses, n)
                timestep_values = to_list(timesteps, n)
                weight_values = to_list(loss_weights, n)
                if isinstance(sizes_hw, torch.Tensor):
                    sizes_hw = sizes_hw.tolist()
                if not isinstance(sizes_hw, (list, tuple)) or len(sizes_hw) != n:
                    sizes_hw = [(None, None)] * n
                for i in range(n):
                    rows.append(
                        {
                            "step": step,
                            "epoch": epoch,
                            "image": image_keys[i],
                            "timestep": timestep_values[i],
                            "loss": loss_values[i],
                            "width": sizes_hw[i][1],
                            "height": sizes_hw[i][0],
                            "batch_size": n,
                            "loss_weight": weight_values[i],
                        }
                    )
            except Exception:
                pass  # e.g. the device is not usable any more after an error. the rows of this batch are lost
        if rows:
            self._append_lines(LOSS_FILE, rows)

    def on_epoch_end(self, *, global_step: int, average_loss: Optional[float] = None):
        """Call at the very end of each epoch, after saving and sampling."""
        if not self.enabled or not self._started:
            return
        self.log_event("epoch_end", step=global_step, average_loss=average_loss)
        # validation, saving and sampling at the end of the epoch must not be counted as waiting for data
        self._last_step_end = None

    def record_checkpoint(self, path: str, *, step: int):
        if not self.enabled or not self._started:
            return
        path = os.path.abspath(path)
        self._status["last_checkpoint"] = {"path": path, "step": step, "epoch": self._epoch, "time": time.time()}
        self.log_event("checkpoint_saved", step=step, path=path)
        self._write_status(force=True)

    def record_state(self, path: Optional[str], *, step: int):
        if not self.enabled or not self._started or path is None:
            return
        path = os.path.abspath(path)
        self._status["last_state"] = {"path": path, "step": step, "epoch": self._epoch, "time": time.time()}
        self.log_event("state_saved", step=step, path=path)
        self._write_status(force=True)

    def _scan_new_samples(self) -> list[str]:
        """Returns the files newly added to the sample directory since the last call."""
        if self.sample_dir is None:
            return []
        try:
            mtime = os.stat(self.sample_dir).st_mtime_ns
            # This is called at every step, so the directory is listed only if its modification time has changed.
            # The same modification time is trusted only if the last listing was clearly later than it, because
            # a file added within the timestamp granularity right after the listing does not change the time.
            settled = self._sample_dir_scan_time - mtime > SAMPLE_DIR_SETTLE_NS
            if mtime == self._sample_dir_mtime and settled:
                return []  # no file is added or removed
            scan_time = time.time_ns()
            files = set(os.listdir(self.sample_dir))
        except OSError:
            return []  # the directory does not exist yet
        self._sample_dir_mtime = mtime
        self._sample_dir_scan_time = scan_time
        new_files = sorted(files - self._known_samples)
        self._known_samples = files
        return new_files

    def check_new_samples(self, *, step: int):
        """Call after the sampling function, which decides by itself whether it generates images at this step.

        The generated images are detected by watching the sample directory, so this works for every model.
        """
        if not self.enabled or not self._started:
            return
        new_files = self._scan_new_samples()
        if not new_files:
            return
        sample_dir = os.path.abspath(self.sample_dir)
        self._status["last_sample"] = {
            "dir": sample_dir,
            "files": new_files,
            "step": step,
            "epoch": self._epoch,
            "time": time.time(),
        }
        self.log_event("samples_generated", step=step, dir=sample_dir, files=new_files)
        self._write_status(force=True)

    # endregion
