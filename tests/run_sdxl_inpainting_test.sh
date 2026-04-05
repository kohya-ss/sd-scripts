#!/usr/bin/env bash
# Smoke test for --train_inpainting with sdxl_train.py.
#
# IMPORTANT: requires a Stable Diffusion XL *inpainting* checkpoint (in_channels=9).
# A standard SDXL base model has in_channels=4 and will fail at the UNet forward pass.
#
# Compatible models:
#   - diffusers/stable-diffusion-xl-1.0-inpainting-0.1  (HuggingFace hub ID or local path)
#   - Any local .safetensors converted from the above
#
# Usage:
#   bash tests/run_sdxl_inpainting_test.sh /path/to/sdxl-inpainting.safetensors
#
#   # Optional: use real downloaded data instead of synthetic images
#   bash tests/run_sdxl_inpainting_test.sh /path/to/model.safetensors tests/downloaded_data
#
# The test runs 20 optimiser steps using sdxl_train.py + sdxl_inpainting_test.toml.
# Success criterion: exit 0 and output .safetensors exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 /path/to/sdxl-inpainting-model.safetensors [data_dir]" >&2
    echo "" >&2
    echo "  The model must be an SDXL inpainting model (UNet in_channels=9)." >&2
    echo "  Standard SDXL base models have in_channels=4 and will not work." >&2
    exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Error: model not found: $MODEL_PATH" >&2
    exit 1
fi

# Optional second argument: training data directory (DreamBooth folder layout)
DATA_DIR="${2:-}"
OUTPUT_DIR="$SCRIPT_DIR/test_output_sdxl"

# ---------------------------------------------------------------------------
# Data: use supplied dir, downloaded data, or synthetic fallback
if [[ -z "$DATA_DIR" ]]; then
    SYNTHETIC_DIR="$SCRIPT_DIR/test_data"
    DOWNLOADED_DIR="$SCRIPT_DIR/downloaded_data"

    if [[ -d "$DOWNLOADED_DIR" ]] && [[ -n "$(ls -A "$DOWNLOADED_DIR")" ]]; then
        echo "==> Using previously downloaded data: $DOWNLOADED_DIR"
        DATA_DIR="$DOWNLOADED_DIR"
    else
        if [[ ! -d "$SYNTHETIC_DIR" ]]; then
            echo "==> Generating synthetic test images..."
            python3 "$SCRIPT_DIR/generate_inpainting_test_data.py"
        fi
        DATA_DIR="$SYNTHETIC_DIR"
        echo "==> Using synthetic test images: $DATA_DIR"
        echo "    (Pass a data dir as \$2, or pre-run download_training_data.py for real images)"
    fi
fi

mkdir -p "$OUTPUT_DIR"

# ---------------------------------------------------------------------------
# Write a runtime TOML that patches the four path fields into the base config.
# This avoids embedding absolute paths in the committed config file.
RUNTIME_TOML="$OUTPUT_DIR/sdxl_inpainting_test_runtime.toml"
BASE_TOML="$SCRIPT_DIR/sdxl_inpainting_test.toml"

"$REPO_DIR/venv/bin/python3" - <<PYEOF
import tomllib, toml
with open("$BASE_TOML", "rb") as f:
    cfg = tomllib.load(f)

cfg["pretrained_model_name_or_path"] = "$MODEL_PATH"
cfg["train_data_dir"] = "$DATA_DIR"
cfg["output_dir"] = "$OUTPUT_DIR"
cfg["output_name"] = "sdxl_inpainting_lora_test"

with open("$RUNTIME_TOML", "w") as f:
    toml.dump(cfg, f)
print(f"Runtime config written to: $RUNTIME_TOML")
PYEOF

echo ""
echo "==> Starting SDXL inpainting training smoke test"
echo "    model  : $MODEL_PATH"
echo "    data   : $DATA_DIR"
echo "    output : $OUTPUT_DIR"
echo "    config : $RUNTIME_TOML"
echo ""

accelerate launch \
    --dynamo_backend no \
    --dynamo_mode default \
    --mixed_precision bf16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 2 \
    "$REPO_DIR/sdxl_train.py" \
        --config_file "$RUNTIME_TOML"

# ---------------------------------------------------------------------------
EXPECTED="$OUTPUT_DIR/sdxl_inpainting_lora_test.safetensors"
if [[ -f "$EXPECTED" ]]; then
    echo ""
    echo "==> PASS: output model created at $EXPECTED"
else
    echo ""
    echo "==> FAIL: expected output not found: $EXPECTED" >&2
    exit 1
fi
