#!/usr/bin/env bash
# Smoke test for --train_inpainting on SDXL (both full FT and LoRA).
#
# Modes:
#   ft   — sdxl_train.py (full fine-tune)
#          Accepts SDXL inpainting checkpoints (in_channels=9) AND standard
#          SDXL base models (in_channels=4 — conv_in is auto-expanded).
#          Output is a full UNet checkpoint; verifier asserts conv_in=9ch.
#   lora — sdxl_train_network.py (LoRA)
#          Requires an SDXL inpainting checkpoint (in_channels=9). LoRA does
#          not extend conv_in, so a standard SDXL will fail at UNet forward.
#          Output is a LoRA-only file; verifier checks for lora_unet_* keys.
#
# Usage:
#   bash tests/run_sdxl_inpainting_test.sh --mode {ft|lora} --model PATH \
#                                          [--data DIR] [--steps N]
#
# Data resolution when --data is omitted:
#   tests/downloaded_data (if present and non-empty)
#     → tests/test_data    (synthetic; auto-generated if absent)
#
# Success criterion: training exits 0, expected .safetensors is produced, and
# the verifier (tests/_verify_inpainting_checkpoint.py) reports PASS.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

MODE=""
MODEL=""
DATA=""
STEPS=""

usage() {
    cat <<EOF
Usage: $0 --mode {ft|lora} --model PATH [--data DIR] [--steps N]

  --mode    ft   = sdxl_train.py (full FT; supports 4ch and 9ch base models)
            lora = sdxl_train_network.py (requires a 9ch inpainting base model)
  --model   path to .safetensors checkpoint
  --data    optional training data dir (DreamBooth folder layout)
  --steps   optional max_train_steps override (default: 20 from TOML)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)   MODE="$2";  shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --data)   DATA="$2";  shift 2 ;;
        --steps)  STEPS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

if [[ -z "$MODE" || -z "$MODEL" ]]; then
    usage >&2
    exit 1
fi
if [[ "$MODE" != "ft" && "$MODE" != "lora" ]]; then
    echo "Error: --mode must be 'ft' or 'lora' (got: $MODE)" >&2
    exit 1
fi
if [[ ! -e "$MODEL" ]]; then
    echo "Error: model not found: $MODEL" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Data resolution
# ---------------------------------------------------------------------------

if [[ -z "$DATA" ]]; then
    DOWNLOADED_DIR="$SCRIPT_DIR/downloaded_data"
    SYNTHETIC_DIR="$SCRIPT_DIR/test_data"

    if [[ -d "$DOWNLOADED_DIR" ]] && [[ -n "$(ls -A "$DOWNLOADED_DIR" 2>/dev/null || true)" ]]; then
        DATA="$DOWNLOADED_DIR"
        echo "==> Using downloaded data: $DATA"
    else
        if [[ ! -d "$SYNTHETIC_DIR" ]]; then
            echo "==> Generating synthetic test images..."
            python "$SCRIPT_DIR/generate_inpainting_test_data.py"
        fi
        DATA="$SYNTHETIC_DIR"
        echo "==> Using synthetic test images: $DATA"
        echo "    (Pass --data DIR or pre-run download_training_data.py for real images.)"
    fi
fi

# ---------------------------------------------------------------------------
# Mode-specific configuration
# ---------------------------------------------------------------------------

if [[ "$MODE" == "ft" ]]; then
    SCRIPT="$REPO_DIR/sdxl_train.py"
    BASE_TOML="$SCRIPT_DIR/sdxl_inpainting_test_ft.toml"
    OUTPUT_DIR="$SCRIPT_DIR/test_output_sdxl_ft"
    OUTPUT_NAME="sdxl_inpainting_test_ft"
    VERIFY_ARGS=(--expect-in-channels 9)
else
    SCRIPT="$REPO_DIR/sdxl_train_network.py"
    BASE_TOML="$SCRIPT_DIR/sdxl_inpainting_test_lora.toml"
    OUTPUT_DIR="$SCRIPT_DIR/test_output_sdxl_lora"
    OUTPUT_NAME="sdxl_inpainting_test_lora"
    VERIFY_ARGS=(--expect-lora)
fi

mkdir -p "$OUTPUT_DIR"

# Optional CLI overrides on top of the TOML
EXTRA=()
if [[ -n "$STEPS" ]]; then
    EXTRA+=( --max_train_steps "$STEPS" )
fi

echo ""
echo "==> SDXL inpainting smoke test (mode=$MODE)"
echo "    script : $SCRIPT"
echo "    config : $BASE_TOML"
echo "    model  : $MODEL"
echo "    data   : $DATA"
echo "    output : $OUTPUT_DIR/$OUTPUT_NAME.safetensors"
echo ""

accelerate launch \
    --dynamo_backend no \
    --dynamo_mode default \
    --num_processes 1 \
    --num_machines 1 \
    --num_cpu_threads_per_process 2 \
    "$SCRIPT" \
        --config_file "$BASE_TOML" \
        --pretrained_model_name_or_path "$MODEL" \
        --train_data_dir "$DATA" \
        --output_dir "$OUTPUT_DIR" \
        --output_name "$OUTPUT_NAME" \
        ${EXTRA[@]+"${EXTRA[@]}"}

# ---------------------------------------------------------------------------
# Verify output
# ---------------------------------------------------------------------------

EXPECTED="$OUTPUT_DIR/$OUTPUT_NAME.safetensors"
if [[ ! -f "$EXPECTED" ]]; then
    echo ""
    echo "==> FAIL: expected output not found: $EXPECTED" >&2
    exit 1
fi

python "$SCRIPT_DIR/_verify_inpainting_checkpoint.py" "$EXPECTED" "${VERIFY_ARGS[@]}"

echo ""
echo "==> PASS"
