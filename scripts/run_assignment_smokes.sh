#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
cd "$ROOT_DIR"

# Grid-baseline x assignment-method driver.
#
# Default mode is a lightweight smoke test:
#   bash scripts/run_assignment_smokes.sh
#
# To save full checkpoints from the same file:
#   RUN_MODE=checkpoint bash scripts/run_assignment_smokes.sh

MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}
RUN_PREFIX=${RUN_PREFIX:-llama31_8b}
RUN_MODE=${RUN_MODE:-smoke}

case "$RUN_MODE" in
  smoke|checkpoint)
    ;;
  *)
    echo "unknown RUN_MODE=$RUN_MODE; use smoke or checkpoint" >&2
    exit 1
    ;;
esac

if [[ "$RUN_MODE" == "checkpoint" ]]; then
  OUTPUT_DIR=${OUTPUT_DIR:-./quantized_models/baselines_${RUN_PREFIX}}
  N_CALIB=${N_CALIB:-128}
  SEQLEN=${SEQLEN:-2048}
  MAX_LAYERS=${MAX_LAYERS:-}
  K=${K:-16}
  FLEXROUND_STEPS=${FLEXROUND_STEPS:-5000}
  SPINQUANT_RANDOM_ROTATIONS=${SPINQUANT_RANDOM_ROTATIONS:-0}
  AUTO_GENERATE_AWQ_SCALES=${AUTO_GENERATE_AWQ_SCALES:-1}
else
  OUTPUT_DIR=${OUTPUT_DIR:-./quantized_models/assignment_smokes}
  N_CALIB=${N_CALIB:-1}
  SEQLEN=${SEQLEN:-128}
  MAX_LAYERS=${MAX_LAYERS:-1}
  K=${K:-0}
  FLEXROUND_STEPS=${FLEXROUND_STEPS:-1}
  SPINQUANT_RANDOM_ROTATIONS=${SPINQUANT_RANDOM_ROTATIONS:-1}
  AUTO_GENERATE_AWQ_SCALES=${AUTO_GENERATE_AWQ_SCALES:-0}
fi

GRIDS=${GRIDS:-"vanilla awq flatquant spinquant_had neuqi"}
SCHEMES=${SCHEMES:-"asymmetric symmetric"}
METHODS=${METHODS:-"rtn gptq gptaq gptaq_rescomp flexround tfic"}

BITS=${BITS:-3}
GROUP_SIZE=${GROUP_SIZE:-128}
CALIB_DATASET=${CALIB_DATASET:-c4}
LAYER_BATCH_SIZE=${LAYER_BATCH_SIZE:-}

DEVICE_MAP=${DEVICE_MAP:-auto}
INPUT_DEVICE=${INPUT_DEVICE:-auto}
STATS_DEVICE=${STATS_DEVICE:-layer}

FLEXROUND_LR=${FLEXROUND_LR:-2e-4}

AWQ_SCALES_PT=${AWQ_SCALES_PT:-}
AWQ_SCALES_PT_ASYMMETRIC=${AWQ_SCALES_PT_ASYMMETRIC:-./outputs/awq_scales/${RUN_PREFIX}_awq_asym_w${BITS}g${GROUP_SIZE}_${CALIB_DATASET}n${N_CALIB}.pt}
AWQ_SCALES_PT_SYMMETRIC=${AWQ_SCALES_PT_SYMMETRIC:-./outputs/awq_scales/${RUN_PREFIX}_awq_sym_w${BITS}g${GROUP_SIZE}_${CALIB_DATASET}n${N_CALIB}.pt}
FLATQUANT_PARAMS_PT=${FLATQUANT_PARAMS_PT:-}
FLATQUANT_PARAMS_PT_ASYMMETRIC=${FLATQUANT_PARAMS_PT_ASYMMETRIC:-}
FLATQUANT_PARAMS_PT_SYMMETRIC=${FLATQUANT_PARAMS_PT_SYMMETRIC:-}
FLATQUANT_TRANSFORMS_PT=${FLATQUANT_TRANSFORMS_PT:-}
SPINQUANT_ROTATIONS_PT=${SPINQUANT_ROTATIONS_PT:-}
SPINQUANT_R4_PT=${SPINQUANT_R4_PT:-}
SPINQUANT_RANDOM_SEED=${SPINQUANT_RANDOM_SEED:-42}

NEUQI_SCALE_CANDIDATES=${NEUQI_SCALE_CANDIDATES:-16}
NEUQI_COARSE_CANDIDATES=${NEUQI_COARSE_CANDIDATES:-8}
NEUQI_CANDIDATE_CHUNK_SIZE=${NEUQI_CANDIDATE_CHUNK_SIZE:-8}
NEUQI_ROW_CHUNK_SIZE=${NEUQI_ROW_CHUNK_SIZE:-16}

mkdir -p "$OUTPUT_DIR"

if [[ "$RUN_MODE" == "checkpoint" ]]; then
  SAVE_ARGS=()
  MAX_LAYERS_ARGS=()
  RUN_KIND="checkpoint"
else
  SAVE_ARGS=(--no-save)
  MAX_LAYERS_ARGS=(--max-layers "$MAX_LAYERS")
  RUN_KIND="smoke"
fi

if [[ "$RUN_MODE" == "checkpoint" && -n "$MAX_LAYERS" ]]; then
  MAX_LAYERS_ARGS=(--max-layers "$MAX_LAYERS")
fi

generate_awq_scales_if_missing() {
  local scheme="$1"
  local out_path="$2"

  if [[ -z "$out_path" || -f "$out_path" || "$AUTO_GENERATE_AWQ_SCALES" != "1" ]]; then
    return
  fi

  echo
  echo ">>> generating missing AWQ scales: scheme=$scheme out=$out_path"
  mkdir -p "$(dirname "$out_path")"
  python -m scripts.generate_awq_scales \
    --model-path "$MODEL_PATH" \
    --scheme "$scheme" \
    --bits "$BITS" \
    --group-size "$GROUP_SIZE" \
    --calib-dataset "$CALIB_DATASET" \
    --n-calib "$N_CALIB" \
    --seqlen "$SEQLEN" \
    --out "$out_path"
}

default_layer_batch_size() {
  local method="$1"

  if [[ -n "$LAYER_BATCH_SIZE" ]]; then
    echo "$LAYER_BATCH_SIZE"
    return
  fi

  if [[ "$RUN_MODE" == "smoke" ]]; then
    echo 1
    return
  fi

  case "$method" in
    rtn) echo "${LBS_RTN:-16}" ;;
    gptq) echo "${LBS_GPTQ:-4}" ;;
    gptaq) echo "${LBS_GPTAQ:-1}" ;;
    gptaq_rescomp) echo "${LBS_GPTAQ_RESCOMP:-1}" ;;
    flexround) echo "${LBS_FLEXROUND:-4}" ;;
    tfic) echo "${LBS_TFIC:-4}" ;;
    *) echo 1 ;;
  esac
}

echo "=== assignment/grid $RUN_KIND run ==="
echo "mode:        $RUN_MODE"
echo "model:       $MODEL_PATH"
echo "output dir:  $OUTPUT_DIR"
echo "grids:       $GRIDS"
echo "schemes:     $SCHEMES"
echo "methods:     $METHODS"
echo "scope:       n_calib=$N_CALIB seqlen=$SEQLEN max_layers=${MAX_LAYERS:-all} save=$([[ "$RUN_MODE" == "checkpoint" ]] && echo 1 || echo 0)"
echo "awq auto:    $AUTO_GENERATE_AWQ_SCALES"
echo "spinquant:   random_rotations=$SPINQUANT_RANDOM_ROTATIONS rotations_pt=${SPINQUANT_ROTATIONS_PT:-unset}"

RUNS=0
SKIPS=0

for GRID in $GRIDS; do
  for SCHEME in $SCHEMES; do
    GRID_ARGS=(--grid "$GRID")
    if [[ "$GRID" == "awq" ]]; then
      SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT"
      if [[ "$SCHEME" == "asymmetric" && -n "$AWQ_SCALES_PT_ASYMMETRIC" ]]; then
        SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT_ASYMMETRIC"
      fi
      if [[ "$SCHEME" == "symmetric" && -n "$AWQ_SCALES_PT_SYMMETRIC" ]]; then
        SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT_SYMMETRIC"
      fi
      generate_awq_scales_if_missing "$SCHEME" "$SCHEME_AWQ_SCALES_PT"
      if [[ -z "$SCHEME_AWQ_SCALES_PT" || ! -f "$SCHEME_AWQ_SCALES_PT" ]]; then
        echo
        echo "!!! skipping grid=awq scheme=$SCHEME because AWQ scales are missing"
        echo "!!! set AWQ_SCALES_PT or AWQ_SCALES_PT_${SCHEME^^}"
        SKIPS=$((SKIPS + 1))
        continue
      fi
      GRID_ARGS+=(--awq-scales-pt "$SCHEME_AWQ_SCALES_PT")
    fi

    if [[ "$GRID" == "flatquant_diag" ]]; then
      SCHEME_FLATQUANT_PARAMS_PT="$FLATQUANT_PARAMS_PT"
      if [[ "$SCHEME" == "asymmetric" && -n "$FLATQUANT_PARAMS_PT_ASYMMETRIC" ]]; then
        SCHEME_FLATQUANT_PARAMS_PT="$FLATQUANT_PARAMS_PT_ASYMMETRIC"
      fi
      if [[ "$SCHEME" == "symmetric" && -n "$FLATQUANT_PARAMS_PT_SYMMETRIC" ]]; then
        SCHEME_FLATQUANT_PARAMS_PT="$FLATQUANT_PARAMS_PT_SYMMETRIC"
      fi
      if [[ -z "$SCHEME_FLATQUANT_PARAMS_PT" || ! -f "$SCHEME_FLATQUANT_PARAMS_PT" ]]; then
        echo
        echo "!!! skipping grid=flatquant_diag scheme=$SCHEME because FlatQuant diag params are missing"
        echo "!!! set FLATQUANT_PARAMS_PT or FLATQUANT_PARAMS_PT_${SCHEME^^}"
        SKIPS=$((SKIPS + 1))
        continue
      fi
      GRID_ARGS+=(--flatquant-params-pt "$SCHEME_FLATQUANT_PARAMS_PT")
    fi

    if [[ "$GRID" == "flatquant" ]]; then
      if [[ -z "$FLATQUANT_TRANSFORMS_PT" || ! -f "$FLATQUANT_TRANSFORMS_PT" ]]; then
        echo
        echo "!!! skipping grid=flatquant scheme=$SCHEME because transforms are missing"
        echo "!!! set FLATQUANT_TRANSFORMS_PT"
        SKIPS=$((SKIPS + 1))
        continue
      fi
      GRID_ARGS+=(--flatquant-transforms-pt "$FLATQUANT_TRANSFORMS_PT")
    fi

    if [[ "$GRID" == "spinquant" || "$GRID" == "spinquant_had" ]]; then
      if [[ -n "$SPINQUANT_ROTATIONS_PT" ]]; then
        if [[ ! -f "$SPINQUANT_ROTATIONS_PT" ]]; then
          echo
          echo "!!! skipping grid=$GRID scheme=$SCHEME because SpinQuant rotations are missing: $SPINQUANT_ROTATIONS_PT"
          SKIPS=$((SKIPS + 1))
          continue
        fi
        GRID_ARGS+=(--spinquant-rotations-pt "$SPINQUANT_ROTATIONS_PT")
      elif [[ "$SPINQUANT_RANDOM_ROTATIONS" == "1" ]]; then
        GRID_ARGS+=(--spinquant-random-rotations --spinquant-random-seed "$SPINQUANT_RANDOM_SEED")
      else
        echo
        echo "!!! skipping grid=$GRID scheme=$SCHEME because no SpinQuant rotations are set"
        echo "!!! set SPINQUANT_ROTATIONS_PT, or SPINQUANT_RANDOM_ROTATIONS=1 for smoke/debug runs"
        SKIPS=$((SKIPS + 1))
        continue
      fi
      if [[ "$GRID" == "spinquant_had" && -n "$SPINQUANT_R4_PT" ]]; then
        if [[ ! -f "$SPINQUANT_R4_PT" ]]; then
          echo "!!! skipping grid=spinquant_had because R4 is missing: $SPINQUANT_R4_PT"
          SKIPS=$((SKIPS + 1))
          continue
        fi
        GRID_ARGS+=(--spinquant-r4-pt "$SPINQUANT_R4_PT")
      fi
    fi

    if [[ "$GRID" == "neuqi" ]]; then
      GRID_ARGS+=(
        --neuqi-scale-candidates "$NEUQI_SCALE_CANDIDATES"
        --neuqi-coarse-candidates "$NEUQI_COARSE_CANDIDATES"
        --neuqi-candidate-chunk-size "$NEUQI_CANDIDATE_CHUNK_SIZE"
        --neuqi-row-chunk-size "$NEUQI_ROW_CHUNK_SIZE"
      )
    fi

    for METHOD in $METHODS; do
      EXTRA_ARGS=()
      if [[ "$METHOD" == "flexround" ]]; then
        EXTRA_ARGS+=(--flexround-steps "$FLEXROUND_STEPS" --flexround-lr "$FLEXROUND_LR")
      fi
      LBS=$(default_layer_batch_size "$METHOD")

      if [[ "$RUN_MODE" == "checkpoint" ]]; then
        RUN_NAME="${RUN_PREFIX}_${GRID}_${SCHEME}_${METHOD}_w${BITS}g${GROUP_SIZE}_${CALIB_DATASET}n${N_CALIB}"
      else
        RUN_NAME="${RUN_PREFIX}_${GRID}_${SCHEME}_${METHOD}_smoke"
      fi

      echo
      echo "=== assignment/grid $RUN_KIND: grid=$GRID scheme=$SCHEME method=$METHOD lbs=$LBS ==="
      python -m scripts.run_quantization_baseline \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_NAME" \
        "${SAVE_ARGS[@]}" \
        "${GRID_ARGS[@]}" \
        --scheme "$SCHEME" \
        --assignment "$METHOD" \
        --bits "$BITS" \
        --group-size "$GROUP_SIZE" \
        --calib-dataset "$CALIB_DATASET" \
        --n-calib "$N_CALIB" \
        --seqlen "$SEQLEN" \
        --k "$K" \
        --layer-batch-size "$LBS" \
        "${MAX_LAYERS_ARGS[@]}" \
        --device-map "$DEVICE_MAP" \
        --input-device "$INPUT_DEVICE" \
        --stats-device "$STATS_DEVICE" \
        "${EXTRA_ARGS[@]}"
      RUNS=$((RUNS + 1))
    done
  done
done

echo
if [[ "$RUN_MODE" == "checkpoint" ]]; then
  echo "assignment/grid checkpoint runs completed: ran=$RUNS skipped=$SKIPS"
  echo "checkpoints: $OUTPUT_DIR"
else
  echo "assignment/grid smoke runs completed: ran=$RUNS skipped=$SKIPS (checkpoint saving disabled)"
fi
