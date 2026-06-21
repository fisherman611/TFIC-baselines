#!/usr/bin/env bash
set -euo pipefail

# End-to-end AWQ assignment smoke for LLaMA:
# small calibration -> save checkpoint -> run small PPL eval for each method.
#
# This is a functionality check, not a benchmark. It quantizes only the first
# few Linear layers by default so the run finishes quickly while still testing
# calibration, AWQ grid construction, assignment, checkpoint saving, reload,
# and perplexity evaluation.

MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}
AWQ_SCALES_PT=${AWQ_SCALES_PT:-./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt}
OUTPUT_DIR=${OUTPUT_DIR:-./quantized_models/awq_assignment_smoke_eval}
RESULT_DIR=${RESULT_DIR:-./results/awq_assignment_smoke_eval}
RUN_PREFIX=${RUN_PREFIX:-llama31_8b_awq_asym}
METHODS=${METHODS:-"rtn gptq gptaq gptaq_rescomp flexround tfic"}

SCHEME=${SCHEME:-asymmetric}
BITS=${BITS:-3}
GROUP_SIZE=${GROUP_SIZE:-128}

CALIB_DATASET=${CALIB_DATASET:-c4}
N_CALIB=${N_CALIB:-2}
SEQLEN=${SEQLEN:-128}
MAX_LAYERS=${MAX_LAYERS:-4}
LAYER_BATCH_SIZE=${LAYER_BATCH_SIZE:-1}

EVAL_DATASETS=${EVAL_DATASETS:-"wikitext2 c4"}
C4_SAMPLES=${C4_SAMPLES:-4}
EVAL_SEED=${EVAL_SEED:-1234}

DEVICE_MAP=${DEVICE_MAP:-auto}
INPUT_DEVICE=${INPUT_DEVICE:-auto}
STATS_DEVICE=${STATS_DEVICE:-layer}

FLEXROUND_STEPS=${FLEXROUND_STEPS:-1}
FLEXROUND_LR=${FLEXROUND_LR:-2e-4}

if [[ ! -f "$AWQ_SCALES_PT" ]]; then
  echo "missing AWQ scales: $AWQ_SCALES_PT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR"

echo "=== LLaMA AWQ assignment smoke + eval ==="
echo "model:       $MODEL_PATH"
echo "awq scales:  $AWQ_SCALES_PT"
echo "methods:     $METHODS"
echo "output dir:  $OUTPUT_DIR"
echo "result dir:  $RESULT_DIR"
echo "calibration: dataset=$CALIB_DATASET n=$N_CALIB seqlen=$SEQLEN max_layers=$MAX_LAYERS"
echo "eval:        datasets=$EVAL_DATASETS c4_samples=$C4_SAMPLES"

for METHOD in $METHODS; do
  RUN_NAME="${RUN_PREFIX}_${METHOD}_w${BITS}g${GROUP_SIZE}_c${N_CALIB}s${SEQLEN}_l${MAX_LAYERS}"
  CHECKPOINT_DIR="$OUTPUT_DIR/$RUN_NAME"
  PPL_JSON="$RESULT_DIR/${RUN_NAME}_ppl.json"

  EXTRA_ARGS=()
  if [[ "$METHOD" == "flexround" ]]; then
    EXTRA_ARGS+=(--flexround-steps "$FLEXROUND_STEPS" --flexround-lr "$FLEXROUND_LR")
  fi

  echo
  echo "=== quantize: $METHOD ==="
  echo "checkpoint: $CHECKPOINT_DIR"

  PYTHONPATH=. python run_quantization_baseline.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --run-name "$RUN_NAME" \
    --grid awq \
    --awq-scales-pt "$AWQ_SCALES_PT" \
    --scheme "$SCHEME" \
    --assignment "$METHOD" \
    --bits "$BITS" \
    --group-size "$GROUP_SIZE" \
    --calib-dataset "$CALIB_DATASET" \
    --n-calib "$N_CALIB" \
    --seqlen "$SEQLEN" \
    --layer-batch-size "$LAYER_BATCH_SIZE" \
    --max-layers "$MAX_LAYERS" \
    --device-map "$DEVICE_MAP" \
    --input-device "$INPUT_DEVICE" \
    --stats-device "$STATS_DEVICE" \
    "${EXTRA_ARGS[@]}"

  echo
  echo "=== eval: $METHOD ==="
  echo "result json: $PPL_JSON"

  # shellcheck disable=SC2086
  PYTHONPATH=. python eval_ppl.py \
    --model-path "$CHECKPOINT_DIR" \
    --datasets $EVAL_DATASETS \
    --seqlen "$SEQLEN" \
    --c4-samples "$C4_SAMPLES" \
    --seed "$EVAL_SEED" \
    --out "$PPL_JSON" \
    --device-map "$DEVICE_MAP" \
    --input-device "$INPUT_DEVICE"
done

echo
echo "all AWQ assignment smoke + eval runs completed"
echo "checkpoints: $OUTPUT_DIR"
echo "ppl jsons:   $RESULT_DIR"
