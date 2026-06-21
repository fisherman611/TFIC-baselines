#!/usr/bin/env bash
set -euo pipefail

# Full experiment driver for the new modular baseline code:
#   grid_baselines/ + assignment_methods/ + run_quantization_baseline.py
#
# Default priority model: LLaMA-3.1-8B. Override paths/settings from the shell:
#
#   MODEL_PATH=/path/or/hf/id bash run_full_baselines.sh
#   MODEL_PATH=meta-llama/Meta-Llama-3.1-8B AWQ_SCALES_PT=./awq_scales.pt bash run_full_baselines.sh
#   AWQ_SCALES_PT_ASYMMETRIC=./awq_asym.pt AWQ_SCALES_PT_SYMMETRIC=./awq_sym.pt bash run_full_baselines.sh
#
# To run fewer cells:
#
#   GRIDS="vanilla" ASSIGNMENTS="rtn tfic" SCHEMES="asymmetric" bash run_full_baselines.sh
#
# To skip expensive downstream lm-eval:
#
#   RUN_LM_EVAL=0 bash run_full_baselines.sh

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}
MODEL_TAG=${MODEL_TAG:-llama31_8b}

OUTPUT_DIR=${OUTPUT_DIR:-./quantized_models/baselines_${MODEL_TAG}}
RESULT_DIR=${RESULT_DIR:-./results/baselines_${MODEL_TAG}}
LOG_DIR=${LOG_DIR:-./logs}

BITS=${BITS:-3}
GROUP_SIZE=${GROUP_SIZE:-128}
CALIB_DATASET=${CALIB_DATASET:-c4}
N_CALIB=${N_CALIB:-128}
SEQLEN=${SEQLEN:-2048}
CALIB_SEED=${CALIB_SEED:-42}
EVAL_SEED=${EVAL_SEED:-1234}
C4_SAMPLES=${C4_SAMPLES:-128}

MODEL_DEVICE_MAP=${MODEL_DEVICE_MAP:-auto}
INPUT_DEVICE=${INPUT_DEVICE:-auto}
STATS_DEVICE=${STATS_DEVICE:-layer}

GRIDS=${GRIDS:-"vanilla awq"}
SCHEMES=${SCHEMES:-"asymmetric symmetric"}
ASSIGNMENTS=${ASSIGNMENTS:-"rtn gptq gptaq gptaq_rescomp flexround tfic"}

FLEXROUND_STEPS=${FLEXROUND_STEPS:-5000}
FLEXROUND_LR=${FLEXROUND_LR:-2e-4}
FLEXROUND_LOG_DIVISOR_BOUND=${FLEXROUND_LOG_DIVISOR_BOUND:-6.0}

AWQ_SCALES_PT=${AWQ_SCALES_PT:-}
AWQ_SCALES_PT_ASYMMETRIC=${AWQ_SCALES_PT_ASYMMETRIC:-}
AWQ_SCALES_PT_SYMMETRIC=${AWQ_SCALES_PT_SYMMETRIC:-}

RUN_PPL=${RUN_PPL:-1}
RUN_LM_EVAL=${RUN_LM_EVAL:-1}
RUN_WANDB=${RUN_WANDB:-${USE_WANDB:-1}}
DELETE_CHECKPOINT=${DELETE_CHECKPOINT:-0}

WANDB_PROJECT=${WANDB_PROJECT:-tfic-baselines}
WANDB_ENTITY=${WANDB_ENTITY:-}

LM_EVAL_TASK_PRESET=${LM_EVAL_TASK_PRESET:-extended}
LM_EVAL_LIMIT=${LM_EVAL_LIMIT:-}
LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-auto}
LM_EVAL_DEVICE=${LM_EVAL_DEVICE:-cuda}
LM_EVAL_NUM_FEWSHOT=${LM_EVAL_NUM_FEWSHOT:-0}

mkdir -p "$OUTPUT_DIR" "$RESULT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_full_baselines_${MODEL_TAG}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== full baseline run started $(date) ==="
echo "model: $MODEL_PATH"
echo "model tag: $MODEL_TAG"
echo "output dir: $OUTPUT_DIR"
echo "result dir: $RESULT_DIR"
echo "log: $LOG_FILE"
echo "grids: $GRIDS"
echo "schemes: $SCHEMES"
echo "assignments: $ASSIGNMENTS"
echo "bits/group: W${BITS} g${GROUP_SIZE}"
echo "calibration: ${CALIB_DATASET}/${N_CALIB}/${SEQLEN} seed=${CALIB_SEED}"
echo "eval c4: samples=${C4_SAMPLES} seed=${EVAL_SEED}"
echo "devices: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all} model_device_map=$MODEL_DEVICE_MAP input_device=$INPUT_DEVICE stats_device=$STATS_DEVICE"
echo "run ppl: $RUN_PPL | run lm-eval: $RUN_LM_EVAL | wandb: $RUN_WANDB | delete checkpoints: $DELETE_CHECKPOINT"

for GRID in $GRIDS; do
  for SCHEME in $SCHEMES; do
    for ASSIGNMENT in $ASSIGNMENTS; do
      RUN_NAME="${MODEL_TAG}_${GRID}_${SCHEME}_${ASSIGNMENT}_w${BITS}g${GROUP_SIZE}_${CALIB_DATASET}n${N_CALIB}"
      CKPT_DIR="$OUTPUT_DIR/$RUN_NAME"

      case "$ASSIGNMENT" in
        rtn)
          LBS=${LBS_RTN:-16}
          EXTRA_ARGS=()
          ;;
        gptq)
          LBS=${LBS_GPTQ:-4}
          EXTRA_ARGS=(--eig-on-cpu)
          ;;
        gptaq)
          LBS=${LBS_GPTAQ:-1}
          EXTRA_ARGS=(--eig-on-cpu)
          ;;
        gptaq_rescomp)
          LBS=${LBS_GPTAQ_RESCOMP:-1}
          EXTRA_ARGS=(--eig-on-cpu)
          ;;
        flexround)
          LBS=${LBS_FLEXROUND:-4}
          EXTRA_ARGS=(
            --flexround-steps "$FLEXROUND_STEPS"
            --flexround-lr "$FLEXROUND_LR"
            --flexround-log-divisor-bound "$FLEXROUND_LOG_DIVISOR_BOUND"
          )
          ;;
        tfic)
          LBS=${LBS_TFIC:-4}
          EXTRA_ARGS=(--eig-on-cpu)
          ;;
        *)
          echo "!!! unknown assignment: $ASSIGNMENT"
          exit 1
          ;;
      esac

      GRID_ARGS=(--grid "$GRID")
      if [[ "$GRID" == "awq" ]]; then
        SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT"
        if [[ "$SCHEME" == "asymmetric" && -n "$AWQ_SCALES_PT_ASYMMETRIC" ]]; then
          SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT_ASYMMETRIC"
        fi
        if [[ "$SCHEME" == "symmetric" && -n "$AWQ_SCALES_PT_SYMMETRIC" ]]; then
          SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT_SYMMETRIC"
        fi
        if [[ -z "$SCHEME_AWQ_SCALES_PT" ]]; then
          echo
          echo "!!! skipping grid=awq scheme=$SCHEME because no AWQ scales path is set"
          echo "!!! set AWQ_SCALES_PT or AWQ_SCALES_PT_${SCHEME^^}"
          continue
        fi
        GRID_ARGS+=(--awq-scales-pt "$SCHEME_AWQ_SCALES_PT")
      fi

      echo
      echo "############################################################"
      echo "# grid=$GRID scheme=$SCHEME assignment=$ASSIGNMENT lbs=$LBS"
      echo "# run_name=$RUN_NAME"
      echo "############################################################"

      echo ">>> [1/3] quantize"
      PYTHONPATH=. python run_quantization_baseline.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_NAME" \
        "${GRID_ARGS[@]}" \
        --scheme "$SCHEME" \
        --assignment "$ASSIGNMENT" \
        --bits "$BITS" \
        --group-size "$GROUP_SIZE" \
        --calib-dataset "$CALIB_DATASET" \
        --n-calib "$N_CALIB" \
        --seqlen "$SEQLEN" \
        --seed "$CALIB_SEED" \
        --layer-batch-size "$LBS" \
        --device-map "$MODEL_DEVICE_MAP" \
        --input-device "$INPUT_DEVICE" \
        --stats-device "$STATS_DEVICE" \
        "${EXTRA_ARGS[@]}"

      if [[ ! -d "$CKPT_DIR" ]]; then
        echo "!!! expected checkpoint missing: $CKPT_DIR -- skipping eval"
        continue
      fi

      if [[ "$RUN_PPL" == "1" ]]; then
        echo ">>> [2/3] PPL eval: WikiText2 test + C4 validation"
        PYTHONPATH=. python eval_ppl.py \
          --model-path "$CKPT_DIR" \
          --datasets wikitext2 c4 \
          --seqlen "$SEQLEN" \
          --c4-samples "$C4_SAMPLES" \
          --seed "$EVAL_SEED" \
          --device-map "$MODEL_DEVICE_MAP" \
          --input-device "$INPUT_DEVICE" \
          --out "$RESULT_DIR/${RUN_NAME}_ppl.json"
      else
        echo ">>> [2/3] PPL eval skipped"
      fi

      if [[ "$RUN_LM_EVAL" == "1" ]]; then
        echo ">>> [3/3] lm-eval preset=$LM_EVAL_TASK_PRESET"
        PYTHONPATH=. python - "$CKPT_DIR" "$RUN_NAME" "$RESULT_DIR/lm_eval" <<PY
from __future__ import annotations

import json
import sys
from datetime import datetime

from lm_eval_runner import LMEvalHarnessRunner
from runtime_utils import DEFAULT_LM_EVAL_TASKS, load_runtime_env, resolve_hf_token

model_path, run_name, output_dir = sys.argv[1:4]
load_runtime_env()
tasks = list(DEFAULT_LM_EVAL_TASKS["$LM_EVAL_TASK_PRESET"])
limit_value = "$LM_EVAL_LIMIT"
limit = None if not limit_value else float(limit_value)

runner = LMEvalHarnessRunner(
    tasks=tasks,
    device="$LM_EVAL_DEVICE",
    batch_size="$LM_EVAL_BATCH_SIZE",
    num_fewshot=int("$LM_EVAL_NUM_FEWSHOT"),
    limit=limit,
    output_dir=output_dir,
    run_name=f"{run_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    hf_token=resolve_hf_token(),
)
results = runner.run({run_name: model_path})
summary_path = f"{output_dir}/{run_name}_summary.json"
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(results[run_name]["summary"], handle, indent=2, sort_keys=True)
print(f"lm-eval summary -> {summary_path}")
PY
      else
        echo ">>> [3/3] lm-eval skipped"
      fi

      if [[ "$RUN_WANDB" == "1" ]]; then
        echo ">>> [wandb] logging metrics"
        WANDB_ARGS=(
          --use-wandb
          --wandb-project "$WANDB_PROJECT"
          --run-name "$RUN_NAME"
          --model-path "$MODEL_PATH"
          --checkpoint-dir "$CKPT_DIR"
          --grid "$GRID"
          --scheme "$SCHEME"
          --assignment "$ASSIGNMENT"
          --bits "$BITS"
          --group-size "$GROUP_SIZE"
          --calib-dataset "$CALIB_DATASET"
          --n-calib "$N_CALIB"
          --seqlen "$SEQLEN"
        )
        if [[ -n "$WANDB_ENTITY" ]]; then
          WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
        fi
        if [[ -f "$RESULT_DIR/${RUN_NAME}_ppl.json" ]]; then
          WANDB_ARGS+=(--ppl-json "$RESULT_DIR/${RUN_NAME}_ppl.json")
        fi
        if [[ -f "$RESULT_DIR/lm_eval/${RUN_NAME}_summary.json" ]]; then
          WANDB_ARGS+=(--lm-eval-summary-json "$RESULT_DIR/lm_eval/${RUN_NAME}_summary.json")
        fi
        PYTHONPATH=. python log_wandb_results.py "${WANDB_ARGS[@]}"
      else
        echo ">>> [wandb] skipped"
      fi

      if [[ "$DELETE_CHECKPOINT" == "1" ]]; then
        echo ">>> deleting checkpoint: $CKPT_DIR"
        rm -rf "$CKPT_DIR"
      fi

      echo "<<< done $RUN_NAME"
    done
  done
done

echo
echo "=== full baseline run finished $(date) ==="
echo "results: $RESULT_DIR"
echo "log: $LOG_FILE"
