#!/usr/bin/env bash
set -euo pipefail

# Lightweight AWQ pipeline checks. Each run quantizes only the first Linear
# layer of the model. These checkpoints are not benchmark results.

MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}
AWQ_SCALES_PT=${AWQ_SCALES_PT:-./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt}
OUTPUT_DIR=${OUTPUT_DIR:-./quantized_models/assignment_smokes}
METHODS=${METHODS:-"rtn gptq flexround tfic"}

if [[ ! -f "$AWQ_SCALES_PT" ]]; then
  echo "missing AWQ scales: $AWQ_SCALES_PT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for METHOD in $METHODS; do
  EXTRA_ARGS=()
  if [[ "$METHOD" == "flexround" ]]; then
    EXTRA_ARGS+=(--flexround-steps 1 --flexround-lr 2e-4)
  fi

  echo
  echo "=== AWQ assignment smoke: $METHOD ==="
  PYTHONPATH=. python run_quantization_baseline.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "llama31_8b_awq_asym_${METHOD}_smoke" \
    --no-save \
    --grid awq \
    --awq-scales-pt "$AWQ_SCALES_PT" \
    --scheme asymmetric \
    --assignment "$METHOD" \
    --bits 3 \
    --group-size 128 \
    --calib-dataset c4 \
    --n-calib 1 \
    --seqlen 128 \
    --k 0 \
    --layer-batch-size 1 \
    --max-layers 1 \
    --device-map auto \
    --input-device auto \
    --stats-device layer \
    "${EXTRA_ARGS[@]}"
done

echo
echo "all assignment smoke runs completed (checkpoint saving disabled)"
