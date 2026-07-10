# Per-subset timestep sampling offset (`custom_attributes.timestep_sampling.offset`)

## Overview

For flow-matching models, the noise level (timestep) at which each sample is trained materially affects the final result. The `timestep_sampling.offset` custom attribute shifts the flow-matching timestep sampling distribution per dataset subset toward higher- or lower-noise regions.

It is applied as an offset to the pre-sigmoid normal sample of the logit-normal (sigmoid / shift / flux_shift) schedule. Default `0.0` (or omitted) leaves sampling unchanged.

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

This feature uses `custom_attributes` to avoid expanding the subset public schema. Set it per subset in your dataset TOML config:

```toml
[[datasets.subsets]]
image_dir = "closeup_shots"
[datasets.subsets.custom_attributes]
timestep_sampling = { offset = -0.5 }

[[datasets.subsets]]
image_dir = "full_body_shots"
[datasets.subsets.custom_attributes]
timestep_sampling = { offset = 0.5 }

[[datasets.subsets]]
image_dir = "other"
# no custom_attributes needed — default is no offset
```

Useful magnitudes are typically in the range of `-1.0` to `1.0`. The offset is applied before the sigmoid transform, so the effect on the final sigma distribution is nonlinear and saturates at large values.

## Scope

- Applies to `sigmoid`, `shift`, and `flux_shift` timestep sampling modes.
- `uniform` and `sigma` (density-based) modes are not affected.
- Currently consumed by `anima_train_network.py`. Other trainers (FLUX, SD3, etc.) do not read this attribute; setting it for those trainers is a no-op.
- The offset is applied only during **training**. Validation uses unbiased sampling for comparable loss metrics.
