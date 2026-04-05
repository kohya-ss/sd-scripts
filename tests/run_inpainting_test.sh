#!/usr/bin/env bash
# Smoke test for the --train_inpainting feature.
#
# IMPORTANT: requires a Stable Diffusion *inpainting* checkpoint (in_channels=9).
# A standard SD 1.x model will crash because the inpainting pipeline concatenates
# [noisy_latents(4ch), mask(1ch), masked_latents(4ch)] → 9 channels before the UNet.
#
# Compatible models (SD 1.x inpainting):
#   - runwayml/stable-diffusion-inpainting  (HuggingFace hub ID or local path)
#   - Any local .safetensors / .ckpt converted from the above
#
# Usage:
#   bash tests/run_inpainting_test.sh /path/to/sd-inpainting.safetensors
#
# The test runs only 20 optimiser steps and saves a LoRA checkpoint.
# Success criterion: script exits 0 and produces a .safetensors file in test_output/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
MODEL_PATH="${1:-}"
if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 /path/to/sd-inpainting-model.safetensors" >&2
    echo "" >&2
    echo "  The model must be an SD 1.x inpainting model (UNet in_channels=9)." >&2
    echo "  Standard SD models have in_channels=4 and will not work." >&2
    exit 1
fi

if [[ ! -e "$MODEL_PATH" ]]; then
    echo "Error: model not found: $MODEL_PATH" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
DATA_DIR="$SCRIPT_DIR/test_data"
OUTPUT_DIR="$SCRIPT_DIR/test_output"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "==> Generating synthetic test images..."
    python3 "$SCRIPT_DIR/generate_inpainting_test_data.py"
fi

mkdir -p "$OUTPUT_DIR"

echo "==> Starting inpainting training smoke test"
echo "    model  : $MODEL_PATH"
echo "    data   : $DATA_DIR"
echo "    output : $OUTPUT_DIR"
echo ""

accelerate launch \
    --num_cpu_threads_per_process 1 \
    "$REPO_DIR/train_network.py" \
        --pretrained_model_name_or_path="$MODEL_PATH" \
        --train_data_dir="$DATA_DIR" \
        --output_dir="$OUTPUT_DIR" \
        --output_name="inpainting_lora_test" \
        --resolution=512 \
        --train_batch_size=1 \
        --max_train_steps=20 \
        --save_every_n_steps=20 \
        --learning_rate=1e-4 \
        --network_module=networks.lora \
        --network_dim=4 \
        --network_alpha=1 \
        --caption_extension=".caption" \
        --save_model_as=safetensors \
        --mixed_precision=no \
        --train_inpainting

# ---------------------------------------------------------------------------
EXPECTED="$OUTPUT_DIR/inpainting_lora_test.safetensors"
if [[ -f "$EXPECTED" ]]; then
    echo ""
    echo "==> PASS: output model created at $EXPECTED"
else
    echo ""
    echo "==> FAIL: expected output not found: $EXPECTED" >&2
    exit 1
fi
