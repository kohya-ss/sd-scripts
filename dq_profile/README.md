# SDXL DQ Dataset Profiler

This profiler is an isolated diagnostic entry point for measuring SDXL LoRA
training behavior from a common snapshot. It measures numerical Safety/Fidelity,
not final image quality or quantization Utility.

## Production direct entry

Run the product orchestrator directly with ordinary Python, not through
`accelerate launch`. The versioned preset fills the validated training and
measurement settings, and output defaults to
`..\lora_output\dq_dataset_profiler`.

```powershell
python -m dq_profile `
  --dq-profile-name="example_dataset" `
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" `
  --dataset_config="D:\datasets\example\dataset.toml"
```

The complete public CLI, fixed `canonical-v1` contract, stage descriptions,
and runtime estimates are documented in
[`docs/dq_dataset_profiler-ja.md`](../docs/dq_dataset_profiler-ja.md).

The older staged custom-dataset runners remain research/reproduction tools;
they are not the ordinary product entry.

The 128-step Trajectory channel is not part of the production candidate
reduction rule. It is available only through explicit
`--with-trajectory-research`, remains descriptive, and cannot claim a best mul,
quality, Utility, or training-success verdict. See
[`docs/dq_dataset_profiler_trajectory_decision-ja.md`](../docs/dq_dataset_profiler_trajectory_decision-ja.md)
for the validation decision.

## Low-level protocol entry

For protocol development or exact reproduction, run from the `sd-scripts`
directory with the same training arguments used for a normal run, but replace
`sdxl_train_network.py` with `sdxl_dq_dataset_profile.py`.

```powershell
.\venv\Scripts\accelerate.exe launch --num_cpu_threads_per_process 8 `
  sdxl_dq_dataset_profile.py `
  --pretrained_model_name_or_path="D:\models\sdxl_base.safetensors" `
  --dataset_config="D:\datasets\example\dataset.toml" `
  --network_module=networks.lora --network_dim=4 --network_args "rank_dropout=0.2" `
  --optimizer_type=AdamW8bitFast --seed=39 `
  --mixed_precision=fp16 --fp16_safe_norms_mode=strict `
  --network_dropout=0.3 `
  --max_train_epochs=40 --gradient_accumulation_steps=1 `
  --lr_scheduler=constant_with_warmup --lr_warmup_steps=0.05 `
  --dq_delta_bits=8 --dq_delta_granularity=channel --dq_delta_stat=rms `
  --dq_delta_range_mul=3.0 --dq_delta_mode=stoch --dq_delta_scope=unet --dq_delta_use_triton `
  --dq_profile_protocol=v2-core `
  --dq_profile_range_muls=2.70,2.85,3.00,3.15,3.30,3.45 `
  --dq_profile_sweep_steps=64 --dq_profile_branch_repeats=2 `
  --dq_profile_output_dir=.\dq_profile_output --dq_profile_name=my_dataset
```

Add `--dq_profile_dry_run` to inspect image/repeat counts, the exact snapshot
step, branch length, and equivalent-epoch cost without loading a model or
writing diagnostic output.

## v2 staged protocol

`v2-core` runs the full fixed grid under `common_skip`, compares cumulative
LoRA parameter updates with the no-quant trajectory at steps 32 and 64, and
reports `m_dir`, `m_total`, `m_stability_diag`, and grid plateaus. These are
trajectory-stability diagnostics, not final image-quality optima. It then checks
the best point and neighboring points with native Guardian behavior. A third
repeat, edge extension, or selected 128-step extension is added only when the
predeclared instability rules require it. NaN/Inf and extreme gradients always
stop the affected branch even under common skip.

After five canonical Core profiles finish, use `dq_profile_v2_gate.py` with a
pre-written expectations JSON. Only a passing `core_gate.json` can authorize
`v2-mechanism`. The gate records the approved one or two stability points per
profile. A same-image paired dataset uses one shared paired-minimax point when
both Core update tables are available. A Mechanism run must identify the
original Core profile with
`--dq_profile_core_profile_key`; it cannot substitute an unapproved range.

```powershell
.\venv\Scripts\python.exe dq_profile_v2_gate.py `
  <profile-1-core-dir> <profile-2-core-dir> <profile-3-core-dir> `
  <paired-profile-a-core-dir> <paired-profile-b-core-dir> `
  --expectations dq_profile\core_expectations.example.json `
  --output_dir <new-core-gate-dir>
```

Use a new output name for the Mechanism run while retaining the Core identity:

```text
--dq_profile_protocol=v2-mechanism
--dq_profile_core_gate_file=<new-core-gate-dir>\core_gate.json
--dq_profile_core_profile_key=<original-Core-profile-name>
```

`v2-mechanism` measures the approved point(s) as explanatory counterfactuals:

- `clip-only = x + (x_clamped - x)`
- `round-only = x + (q - x_clamped)`
- `full = x + (x_clamped - x) + (q - x_clamped)`
- interaction = `(full-no_quant) - (clip-no_quant) - (round-no_quant)`

These mechanism branches are not recommendation candidates. Formal source-held-
out and 40-epoch utility experiments remain later gated phases. Utility protocol
and blind-judgment tools are provided by `dq_profile_v2_utility.py`; they fix
`final_avg_center` as primary, training seeds 39/40, prompt-level bootstrap, and
ROPE `[0.45, 0.55]`. One training seed is only a low-confidence screen.

For v2, `standard` uses two structural probe replicas. CountSketch defaults to
two independent 512-wide sketches, checks Gram/rank agreement, and falls back to
two precomputed 1024-wide sketches when necessary. Dataset geometry modifies
confidence and repeat requirements; it never votes for a quantization candidate.

To separate crop/file variation from independent source variation, provide a
JSON or CSV `--dq_profile_source_group_map` that maps each image key (exactly or
by directory prefix) to its original source. Without it, each image key is
conservatively treated as a separate source and `image_within_source` is marked
not estimable rather than inferred from folder layout.

Safety and reproducibility guarantees:

- The entry point forces `dq_profile.copied_lora` and uses
  `dq_profile.copied_train_network.NetworkTrainer`; it never inherits from the
  normal `train_network.NetworkTrainer`.
- Model/checkpoint/state saving, resume, external trackers, sampling, and
  Hugging Face uploads are disabled before the copied trainer starts.
- DataLoader workers are fixed to zero.  Branches consume only a sealed,
  post-collation replay sequence and cannot access a live DataLoader.
- The in-memory snapshot is taken at `global_step == dq_delta_begin_step`,
  after the final unquantized update and before processing the first quantized
  batch.
- Production profile quantization uses stateless per-module random tensors.
  Candidate names are not part of the seed, so all fixed-range candidates use common random
  numbers without advancing model/dropout/noise RNG streams.
- Structural probes disable dropout; short training branches keep production
  dropout enabled.  Their metrics are labeled separately in every output.

The output directory contains `report.html`, versioned JSON manifests, CSV
tables, structural sketches/Gram matrices, an SVG loss trajectory, status and
log files, a copy of the dataset TOML, and an editable `known_result.toml`.
Probe and branch metrics retain the distinct `structural_dropout_off` and
`training_dropout_on` regimes; candidate summaries use separate `probe_*` and
`branch_*` column names. `comparison_controlled` defaults to `false`; if it is
set to true, dim, optimizer, precision, safe-norm, dataset hash, training-step,
and DQ controls must all match for `comparison_controlled_effective` to become
true. Known results are displayed as reference only and never drive candidate
ranking or recommendation.

All protocols require one process/GPU, dataset batch size 1, gradient
accumulation 1, fixed bits mode, and a positive LR warmup. They reject resume,
bits schedules, arbitrary network modules, and step-based quantization. v2 also
enforces the canonical AdamW8bitFast, strict fp16 safe norms, Triton, dim 4,
seed 39, rank dropout 0.2, and network dropout 0.3 controls. Use
`--dq_profile_protocol=v1` only to reproduce the earlier three-branch protocol.
