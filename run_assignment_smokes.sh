#!/usr/bin/env bash
set -euo pipefail

# Lightweight grid-baseline x assignment-method pipeline checks. Each run
# quantizes only the first Linear layer of the model. These checkpoints are
# not benchmark results.

MODEL_PATH=${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B}
AWQ_SCALES_PT=${AWQ_SCALES_PT:-./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt}
OUTPUT_DIR=${OUTPUT_DIR:-./quantized_models/assignment_smokes}
RUN_PREFIX=${RUN_PREFIX:-llama31_8b}

GRIDS=${GRIDS:-"vanilla awq flatquant_diag spinquant neuqi"}
SCHEMES=${SCHEMES:-"asymmetric symmetric"}
METHODS=${METHODS:-"rtn gptq gptaq gptaq_rescomp flexround tfic"}

BITS=${BITS:-3}
GROUP_SIZE=${GROUP_SIZE:-128}
CALIB_DATASET=${CALIB_DATASET:-c4}
N_CALIB=${N_CALIB:-1}
SEQLEN=${SEQLEN:-128}
MAX_LAYERS=${MAX_LAYERS:-1}
LAYER_BATCH_SIZE=${LAYER_BATCH_SIZE:-1}
K=${K:-0}

DEVICE_MAP=${DEVICE_MAP:-auto}
INPUT_DEVICE=${INPUT_DEVICE:-auto}
STATS_DEVICE=${STATS_DEVICE:-layer}

FLEXROUND_STEPS=${FLEXROUND_STEPS:-1}
FLEXROUND_LR=${FLEXROUND_LR:-2e-4}

AWQ_SCALES_PT_ASYMMETRIC=${AWQ_SCALES_PT_ASYMMETRIC:-}
AWQ_SCALES_PT_SYMMETRIC=${AWQ_SCALES_PT_SYMMETRIC:-}
FLATQUANT_PARAMS_PT=${FLATQUANT_PARAMS_PT:-}
FLATQUANT_PARAMS_PT_ASYMMETRIC=${FLATQUANT_PARAMS_PT_ASYMMETRIC:-}
FLATQUANT_PARAMS_PT_SYMMETRIC=${FLATQUANT_PARAMS_PT_SYMMETRIC:-}
SPINQUANT_ROTATIONS_PT=${SPINQUANT_ROTATIONS_PT:-}
SPINQUANT_RANDOM_ROTATIONS=${SPINQUANT_RANDOM_ROTATIONS:-1}
SPINQUANT_RANDOM_SEED=${SPINQUANT_RANDOM_SEED:-42}

NEUQI_SCALE_CANDIDATES=${NEUQI_SCALE_CANDIDATES:-16}
NEUQI_COARSE_CANDIDATES=${NEUQI_COARSE_CANDIDATES:-8}
NEUQI_CANDIDATE_CHUNK_SIZE=${NEUQI_CANDIDATE_CHUNK_SIZE:-8}
NEUQI_ROW_CHUNK_SIZE=${NEUQI_ROW_CHUNK_SIZE:-16}

mkdir -p "$OUTPUT_DIR"

echo "=== assignment/grid smoke run ==="
echo "model:       $MODEL_PATH"
echo "output dir:  $OUTPUT_DIR"
echo "grids:       $GRIDS"
echo "schemes:     $SCHEMES"
echo "methods:     $METHODS"
echo "scope:       n_calib=$N_CALIB seqlen=$SEQLEN max_layers=$MAX_LAYERS no_save=1"
echo "spinquant:   random_rotations=$SPINQUANT_RANDOM_ROTATIONS rotations_pt=${SPINQUANT_ROTATIONS_PT:-unset}"

RUNS=0
SKIPS=0

for GRID in $GRIDS; do
  for SCHEME in $SCHEMES; do
    if [[ "$GRID" == "neuqi" && "$SCHEME" != "asymmetric" ]]; then
      echo
      echo "!!! skipping grid=neuqi scheme=$SCHEME because NeUQI is asymmetric only"
      SKIPS=$((SKIPS + 1))
      continue
    fi

    GRID_ARGS=(--grid "$GRID")
    if [[ "$GRID" == "awq" ]]; then
      SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT"
      if [[ "$SCHEME" == "asymmetric" && -n "$AWQ_SCALES_PT_ASYMMETRIC" ]]; then
        SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT_ASYMMETRIC"
      fi
      if [[ "$SCHEME" == "symmetric" && -n "$AWQ_SCALES_PT_SYMMETRIC" ]]; then
        SCHEME_AWQ_SCALES_PT="$AWQ_SCALES_PT_SYMMETRIC"
      fi
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

    if [[ "$GRID" == "spinquant" ]]; then
      if [[ -n "$SPINQUANT_ROTATIONS_PT" ]]; then
        if [[ ! -f "$SPINQUANT_ROTATIONS_PT" ]]; then
          echo
          echo "!!! skipping grid=spinquant scheme=$SCHEME because SpinQuant rotations are missing: $SPINQUANT_ROTATIONS_PT"
          SKIPS=$((SKIPS + 1))
          continue
        fi
        GRID_ARGS+=(--spinquant-rotations-pt "$SPINQUANT_ROTATIONS_PT")
      elif [[ "$SPINQUANT_RANDOM_ROTATIONS" == "1" ]]; then
        GRID_ARGS+=(--spinquant-random-rotations --spinquant-random-seed "$SPINQUANT_RANDOM_SEED")
      else
        echo
        echo "!!! skipping grid=spinquant scheme=$SCHEME because no SpinQuant rotations are set"
        echo "!!! set SPINQUANT_ROTATIONS_PT, or SPINQUANT_RANDOM_ROTATIONS=1 for smoke/debug runs"
        SKIPS=$((SKIPS + 1))
        continue
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

      RUN_NAME="${RUN_PREFIX}_${GRID}_${SCHEME}_${METHOD}_smoke"

      echo
      echo "=== assignment/grid smoke: grid=$GRID scheme=$SCHEME method=$METHOD ==="
      PYTHONPATH=. python run_quantization_baseline.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --run-name "$RUN_NAME" \
        --no-save \
        "${GRID_ARGS[@]}" \
        --scheme "$SCHEME" \
        --assignment "$METHOD" \
        --bits "$BITS" \
        --group-size "$GROUP_SIZE" \
        --calib-dataset "$CALIB_DATASET" \
        --n-calib "$N_CALIB" \
        --seqlen "$SEQLEN" \
        --k "$K" \
        --layer-batch-size "$LAYER_BATCH_SIZE" \
        --max-layers "$MAX_LAYERS" \
        --device-map "$DEVICE_MAP" \
        --input-device "$INPUT_DEVICE" \
        --stats-device "$STATS_DEVICE" \
        "${EXTRA_ARGS[@]}"
      RUNS=$((RUNS + 1))
    done
  done
done

echo
echo "assignment/grid smoke runs completed: ran=$RUNS skipped=$SKIPS (checkpoint saving disabled)"
