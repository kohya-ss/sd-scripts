# LECO Training

This repository now includes dedicated LECO training entry points:

- `train_leco.py` for Stable Diffusion 1.x / 2.x
- `sdxl_train_leco.py` for SDXL

These scripts train a LoRA against the model's own noise predictions, so no image dataset is required.

## Current scope

- U-Net LoRA training only
- `networks.lora` is the default network module
- Prompt TOML supports both original LECO prompt pairs and ai-toolkit style slider targets
- Full ai-toolkit job YAML is not supported; convert your prompt/target settings to TOML format

## Example: SD 1.x / 2.x

```bash
accelerate launch train_leco.py ^
  --pretrained_model_name_or_path="model.safetensors" ^
  --output_dir="output" ^
  --output_name="detail_slider" ^
  --prompts_file="prompts.toml" ^
  --network_dim=8 ^
  --network_alpha=4 ^
  --learning_rate=1e-4 ^
  --max_train_steps=500 ^
  --max_denoising_steps=40 ^
  --mixed_precision=bf16
```

## Example: SDXL

```bash
accelerate launch sdxl_train_leco.py ^
  --pretrained_model_name_or_path="sdxl_model.safetensors" ^
  --output_dir="output" ^
  --output_name="detail_slider_xl" ^
  --prompts_file="slider.toml" ^
  --network_dim=8 ^
  --network_alpha=4 ^
  --learning_rate=1e-4 ^
  --max_train_steps=500 ^
  --max_denoising_steps=40 ^
  --mixed_precision=bf16
```

## Prompt TOML: original LECO format

```toml
[[prompts]]
target = "van gogh"
positive = "van gogh"
unconditional = ""
neutral = ""
action = "erase"
guidance_scale = 1.0
resolution = 512
batch_size = 1
multiplier = 1.0
weight = 1.0
```

## Prompt TOML: ai-toolkit style slider target

This expands internally into the bidirectional LECO pairs needed for slider-style behavior.

```toml
guidance_scale = 1.0
resolution = 512
neutral = ""

[[targets]]
target_class = ""
positive = "high detail, intricate, high quality"
negative = "blurry, low detail, low quality"
multiplier = 1.0
weight = 1.0
```

You can also provide multiple neutral prompts:

```toml
neutrals = ["", "studio photo", "cinematic lighting"]

[[targets]]
target_class = "person"
positive = "smiling person"
negative = "expressionless person"
```

## Converting from ai-toolkit YAML

If you have an existing ai-toolkit style YAML config, convert it to TOML as follows:

**YAML:**
```yaml
targets:
  - target_class: ""
    positive: "high detail"
    negative: "low detail"
    multiplier: 1.0
guidance_scale: 1.0
resolution: 512
```

**TOML equivalent:**
```toml
guidance_scale = 1.0
resolution = 512

[[targets]]
target_class = ""
positive = "high detail"
negative = "low detail"
multiplier = 1.0
```

Key differences:
- Use `=` instead of `:` for key-value pairs
- Use `[[targets]]` instead of `targets:` with `- ` list items
- String values use `"quotes"` (same as YAML)
- Arrays use `[brackets]` (e.g., `neutrals = ["a", "b"]`)
