# Per-subset timestep sampling offset (`timestep_sampling_offset`)

## Overview

For flow-matching models, the noise level (timestep) at which each sample is trained materially affects the final result. `timestep_sampling_offset` shifts the flow-matching timestep sampling distribution per dataset subset toward higher- or lower-noise regions.

It is applied as an offset to the pre-sigmoid normal sample of the logit-normal (sigmoid / shift / flux_shift) schedule. Default `0.0` leaves sampling unchanged.

- **Negative** value → biases toward lower-noise timesteps (detail-focused steps)
- **Positive** value → biases toward higher-noise timesteps (structure-focused steps)

## Motivation

Even at the same resolution, images with different semantic granularity benefit from a different noise emphasis:

- **Close-up / fine-detail images** (e.g. head shots, texture-heavy content): lower-noise emphasis lets the model focus on fine texture refinement.
- **Full-body / macro-structure images**: higher-noise emphasis pushes the model to learn overall structure and composition from heavier corruption.

For FLUX in particular, the model carries a strong photoreal prior. When fine-tuning on anime data, the model tends toward overly aggressive gradient updates. Applying a per-content noise offset smooths these update dynamics and stabilizes training.

## Background

This per-content noise treatment is motivated by Semantic Granularity Alignment (SGA):

> Xiong & Yuan, *"The Quadratic Geometry of Flow Matching: Semantic Granularity Alignment for Text-to-Image Synthesis"*, [arXiv:2603.10785](https://arxiv.org/abs/2603.10785)

SGA analyzes flow-matching fine-tuning as a quadratic form governed by a Neural Tangent Kernel. It shows that aligning data geometry with the optimization structure — here, offsetting the training noise per semantic granularity — mitigates gradient conflicts and improves both convergence efficiency and structural integrity.

## Usage

In your dataset TOML config, set `timestep_sampling_offset` per subset:

```toml
[[datasets.subsets]]
image_dir = "closeup_shots"
timestep_sampling_offset = -0.5   # lower noise → detail refinement

[[datasets.subsets]]
image_dir = "full_body_shots"
timestep_sampling_offset = 0.5    # higher noise → structure learning

[[datasets.subsets]]
image_dir = "other"
# default 0.0 = unchanged
```

Useful magnitudes are typically in the range of `-1.0` to `1.0`. The offset is applied before the sigmoid transform, so the effect on the final sigma distribution is nonlinear and saturates at large values.

## Scope

- Applies to `sigmoid`, `shift`, and `flux_shift` timestep sampling modes.
- `uniform` and `sigma` (density-based) modes are not affected.
- Currently wired in `anima_train_network.py`; other trainers can opt in with a one-line change.
