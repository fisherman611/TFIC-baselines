#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

# Unified script to generate necessary offline checkpoints (artifacts) 
# for AWQ, FlatQuant, and SpinQuant before running full baselines.

MODEL_PATH=${MODEL_PATH:-"meta-llama/Meta-Llama-3.1-8B"}
MODEL_TAG=${MODEL_TAG:-"llama31_8b"}
OUTPUT_DIR=${OUTPUT_DIR:-"./quantized_models/artifacts_${MODEL_TAG}"}
LOG_DIR=${LOG_DIR:-"./logs"}

# Limit execution to GPU 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# Hyperparameters
BITS=${BITS:-3}
ACTIVATION_BITS=${ACTIVATION_BITS:-16}
SEQLEN=${SEQLEN:-2048}

SCHEMES=${SCHEMES:-"asymmetric symmetric"}
METHODS=${METHODS:-"awq flatquant spinquant"}

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/generate_ckpts_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Generation of offline artifacts started $(date) ==="
echo "Model: $MODEL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "Target Methods: $METHODS"
echo "Target Schemes: $SCHEMES"

for METHOD in $METHODS; do
  for SCHEME in $SCHEMES; do
    echo
    echo "############################################################"
    echo "# Generating artifacts for: $METHOD ($SCHEME)"
    echo "############################################################"

    if [[ "$METHOD" == "awq" ]]; then
      python -m scripts.generate_awq_scales \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR/awq" \
        --scheme "$SCHEME" \
        --seqlen "$SEQLEN"

    elif [[ "$METHOD" == "flatquant" ]]; then
      if [[ "$SCHEME" == "symmetric" ]]; then
        FQ_ARGS="--weight-symmetric --activation-symmetric"
      else
        FQ_ARGS="--no-weight-symmetric --no-activation-symmetric"
      fi
      python -m scripts.calibrate_flatquant \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR/flatquant_$SCHEME" \
        --weight-bits "$BITS" \
        --activation-bits "$ACTIVATION_BITS" \
        $FQ_ARGS
        
    elif [[ "$METHOD" == "spinquant" ]]; then
      # SpinQuant rotations are scheme-agnostic, only run once
      if [[ "$SCHEME" == "asymmetric" ]]; then
        python -m scripts.calibrate_spinquant \
          --model-path "$MODEL_PATH" \
          --output-dir "$OUTPUT_DIR/spinquant" \
          --seqlen "$SEQLEN"
      fi
        
    else
      echo "Unknown method: $METHOD. Skipping."
    fi
  done
done

echo
echo "=== Generation finished $(date) ==="
echo "Artifacts are saved in $OUTPUT_DIR"
echo "Log saved to $LOG_FILE"
