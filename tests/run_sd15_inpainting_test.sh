#!/usr/bin/env bash
# Smoke test for --train_inpainting with train_db.py (SD1.5 DreamBooth fine-tune).
#
# Compatible models:
#   - runwayml/stable-diffusion-inpainting  (HuggingFace hub ID or local path)
#   - Any local .safetensors SD1.5 inpainting checkpoint (in_channels=9)
#   - Any standard SD1.5 checkpoint (in_channels=4): conv_in is automatically
#     expanded to 9 channels when --train_inpainting is set.
#
# Usage:
#   bash tests/run_sd15_inpainting_test.sh /path/to/sd15-model.safetensors
#
#   # Optional: use real downloaded data instead of synthetic images
#   bash tests/run_sd15_inpainting_test.sh /path/to/model.safetensors tests/downloaded_data
#
# The test runs 20 optimiser steps using train_db.py + sd15_inpainting_test.toml.
# Success criterion: exit 0 and output .safetensors exists.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 /path/to/sd15-model.safetensors [data_dir]" >&2
    echo "" >&2
    echo "  Accepts both SD1.5 inpainting models (in_channels=9) and standard" >&2
    echo "  SD1.5 checkpoints (in_channels=4 — conv_in is expanded automatically)." >&2
    exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Error: model not found: $MODEL_PATH" >&2
    exit 1
fi

# Optional second argument: training data directory (DreamBooth folder layout)
DATA_DIR="${2:-}"
OUTPUT_DIR="$SCRIPT_DIR/test_output_sd15"

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
RUNTIME_TOML="$OUTPUT_DIR/sd15_inpainting_test_runtime.toml"
BASE_TOML="$SCRIPT_DIR/sd15_inpainting_test.toml"

"$REPO_DIR/venv/bin/python3" - <<PYEOF
import tomllib, toml
with open("$BASE_TOML", "rb") as f:
    cfg = tomllib.load(f)

cfg["pretrained_model_name_or_path"] = "$MODEL_PATH"
cfg["train_data_dir"] = "$DATA_DIR"
cfg["output_dir"] = "$OUTPUT_DIR"
cfg["output_name"] = "sd15_inpainting_test"

with open("$RUNTIME_TOML", "w") as f:
    toml.dump(cfg, f)
print(f"Runtime config written to: $RUNTIME_TOML")
PYEOF

echo ""
echo "==> Starting SD1.5 inpainting training smoke test"
echo "    model  : $MODEL_PATH"
echo "    data   : $DATA_DIR"
echo "    output : $OUTPUT_DIR"
echo "    config : $RUNTIME_TOML"
echo ""

accelerate launch \
    --dynamo_backend no \
    --dynamo_mode default \
    --mixed_precision fp16 \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 2 \
    "$REPO_DIR/train_db.py" \
        --config_file "$RUNTIME_TOML"

# ---------------------------------------------------------------------------
EXPECTED="$OUTPUT_DIR/sd15_inpainting_test.safetensors"
if [[ -f "$EXPECTED" ]]; then
    echo ""
    echo "==> PASS: output model created at $EXPECTED"
else
    echo ""
    echo "==> FAIL: expected output not found: $EXPECTED" >&2
    exit 1
fi
