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

**Recommended range: `-0.5` to `0.5`.** Tested across multiple datasets; values in this range produce moderate, stable improvements. Larger magnitudes (e.g. `±1.0`) cause extreme distribution skew — see [Understanding the offset](#understanding-the-offset) below.

The offset is applied before the sigmoid transform, so the effect on the final sigma distribution is nonlinear and saturates at large values.

## Scope

- Applies to `sigmoid`, `shift`, and `flux_shift` timestep sampling modes.
- `uniform` and `sigma` (density-based) modes are not affected.
- Currently consumed by `anima_train_network.py` and `flux_train_network.py`. Other trainers (SD3, Lumina, etc.) do not read this attribute; setting it for those trainers is a no-op.
- The offset is applied only during **training**. Validation uses unbiased sampling for comparable loss metrics.

## Understanding the offset

### Timestep ranges and learning behavior

| Range (approximate) | Noise level | What the model learns to restore |
|---|---|---|
| **High t** (0.7–1.0) | High | Global structure — composition, spatial layout, overall tone |
| **Mid t** (0.3–0.7) | Medium | Mid-level structure — proportions, lighting, regional color |
| **Low t** (0.0–0.3) | Low | Fine texture — line quality, material detail, high-frequency information |

Offset adjusts the learning emphasis across these ranges to match the training content. Note that FM and U-Net architectures have different inherent biases across these ranges, so the same offset value may produce different effects depending on the architecture.

### How offset shifts the distribution

Default `logit_normal` sampling draws `z ~ N(0, 1)` then `t = sigmoid(z)`, producing a symmetric bell curve centered at `t = 0.5`.

Adding offset shifts the mean: `z ~ N(offset, 1)`.

- **Positive offset** (e.g. +0.5): distribution shifts toward high t, mean moves from 0.500 → ≈0.622.
- **Negative offset** (e.g. −0.5): distribution shifts toward low t, mean moves from 0.500 → ≈0.378.

![offset distribution comparison](images/timestep_bias/offset_distribution_comparison.png)

Note: when `sigmoid_scale ≠ 1.0`, the effective shift is `sigmoid_scale × offset`. The means above assume the default scale of 1.0.

### Risk of excessive offset

The default logit-normal sampling is already a biased (bell-shaped) distribution. Offset stacks additional shift on top of it. Values beyond ±0.5 should be used with caution:

- **Tail starvation**: extreme offset skews the distribution so that some timestep ranges are rarely sampled, degrading detail quality (positive offset) or structural quality (negative offset).
- **Velocity field degradation**: the velocity field must be accurate across the full path `t ∈ [0, 1]`. Under-trained ranges degrade inference quality and may cause artifacts along the inference trajectory.

**Recommended range: ±0.5.** This range worked well in our experiments across several datasets, producing observable improvements without sacrificing coverage in any timestep range.
