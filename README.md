# TFIC Baselines

This repository contains post-training quantization baselines for studying:

- quantization grids / transforms: Vanilla and AWQ are implemented
- assignment methods: RTN, GPTQ, and TFIC are implemented
- evaluation: perplexity on WikiText2 and C4, plus `lm-eval`

The current modular pipeline is:

```text
grid_baselines/         build the quantization grid
assignment_methods/     assign integer codes on that grid
run_quantization_baseline.py
                         run one real-model grid x assignment experiment
run_full_baselines.sh   orchestrate full experiments
generate_awq_scales.py  generate AWQ per-layer scales
eval_ppl.py             evaluate WikiText2 / C4 perplexity
lm_eval_runner.py       run lm-evaluation-harness
log_wandb_results.py    log saved PPL/lm-eval JSON files to W&B
```

## Implemented Methods

Grid baselines:

```text
vanilla: symmetric and asymmetric
awq:     symmetric and asymmetric
```

Assignment methods:

```text
rtn
gptq
tfic
```

Default experiment setup:

```text
model:       meta-llama/Meta-Llama-3.1-8B
bits:        3
group size:  128
calibration: C4 train shard, 128 samples, seqlen 2048
PPL eval:    WikiText2 test + C4 validation
lm-eval:     extended preset
```

## Setup

Install `uv`, then sync dependencies from `pyproject.toml`:

```bash
pip install uv
uv sync --extra dev
```

For gated Hugging Face models such as LLaMA, put your token in `.env`:

```bash
echo 'HF_TOKEN=your_hf_token_here' > .env
```

Optional W&B logging also reads `.env`:

```bash
cat >> .env <<'EOF'
WANDB_API_KEY=your_wandb_key_here
WANDB_PROJECT=tfic-baselines
WANDB_ENTITY=your_entity_optional
EOF
```

Sanity check:

```bash
uv run pytest tests -q
```

## Quick Debug Run

Run one lightweight cell first. This quantizes only Vanilla/asymmetric/RTN and
skips downstream `lm-eval`:

```bash
GRIDS="vanilla" SCHEMES="asymmetric" ASSIGNMENTS="rtn" RUN_LM_EVAL=0 \
  uv run bash run_full_baselines.sh
```

`run_full_baselines.sh` logs to W&B by default. To disable W&B for a debug run:

```bash
RUN_WANDB=0 GRIDS="vanilla" SCHEMES="asymmetric" ASSIGNMENTS="rtn" RUN_LM_EVAL=0 \
  uv run bash run_full_baselines.sh
```

## Run Vanilla Baselines

Run all implemented assignment methods on the Vanilla grid:

```bash
GRIDS="vanilla" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq tfic" \
  uv run bash run_full_baselines.sh
```

Each cell runs:

```text
1. quantization with run_quantization_baseline.py
2. PPL eval on WikiText2 test and C4 validation
3. lm-eval with the extended task preset
```

## Generate AWQ Scales

AWQ requires per-layer scales before running the AWQ grid.

Asymmetric AWQ scales:

```bash
uv run python generate_awq_scales.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --scheme asymmetric \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --out ./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt
```

Symmetric AWQ scales:

```bash
uv run python generate_awq_scales.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --scheme symmetric \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --out ./outputs/awq_scales/llama31_8b_awq_sym_w3g128_c4n128.pt
```

## Run AWQ Baselines

Run AWQ asymmetric and symmetric:

```bash
AWQ_SCALES_PT_ASYMMETRIC=./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
AWQ_SCALES_PT_SYMMETRIC=./outputs/awq_scales/llama31_8b_awq_sym_w3g128_c4n128.pt \
GRIDS="awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq tfic" \
  uv run bash run_full_baselines.sh
```

Run both Vanilla and AWQ:

```bash
AWQ_SCALES_PT_ASYMMETRIC=./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
AWQ_SCALES_PT_SYMMETRIC=./outputs/awq_scales/llama31_8b_awq_sym_w3g128_c4n128.pt \
GRIDS="vanilla awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq tfic" \
  uv run bash run_full_baselines.sh
```

## Outputs

By default, `run_full_baselines.sh` uses:

```text
MODEL_PATH=meta-llama/Meta-Llama-3.1-8B
MODEL_TAG=llama31_8b
```

Outputs are written to:

```text
./quantized_models/baselines_llama31_8b/
./results/baselines_llama31_8b/
./logs/
```

To save disk after each cell is evaluated:

```bash
DELETE_CHECKPOINT=1 GRIDS="vanilla" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq tfic" \
  uv run bash run_full_baselines.sh
```

## Single-Cell Commands

Vanilla + TFIC:

```bash
uv run python run_quantization_baseline.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128 \
  --grid vanilla \
  --scheme asymmetric \
  --assignment tfic \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --layer-batch-size 4 \
  --eig-on-cpu
```

Evaluate PPL:

```bash
uv run python eval_ppl.py \
  --model-path ./quantized_models/baselines_llama31_8b/llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128 \
  --datasets wikitext2 c4 \
  --seqlen 2048 \
  --c4-samples 128 \
  --seed 1234
```

Evaluate PPL and log it to W&B:

```bash
uv run python eval_ppl.py \
  --model-path ./quantized_models/baselines_llama31_8b/llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128 \
  --datasets wikitext2 c4 \
  --seqlen 2048 \
  --c4-samples 128 \
  --seed 1234 \
  --use-wandb \
  --wandb-run-name llama31_8b_vanilla_asymmetric_tfic_ppl
```

Log existing result JSON files to W&B:

```bash
uv run python log_wandb_results.py \
  --use-wandb \
  --run-name llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128 \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --checkpoint-dir ./quantized_models/baselines_llama31_8b/llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128 \
  --ppl-json ./results/baselines_llama31_8b/llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128_ppl.json \
  --lm-eval-summary-json ./results/baselines_llama31_8b/lm_eval/llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128_summary.json \
  --grid vanilla \
  --scheme asymmetric \
  --assignment tfic \
  --bits 3 \
  --group-size 128
```

## Legacy EigenFlip Runner

The older runner is still available:

```bash
PYTHONPATH=. uv run python eigenflip/run_fast.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/eigenflip_3bit \
  --bits 3 --group-size 128 --k 16 \
  --base rtn --scheme asymmetric --encoder tfic_fast \
  --calib-dataset c4 --n-calib 128 --seqlen 2048 \
  --layer-batch-size 4 --eig-on-cpu

PYTHONPATH=. uv run python eval_ppl.py \
  --model-path ./quantized_models/eigenflip_3bit/rtn_asymmetric_tfic_fast \
  --datasets wikitext2 c4 --seqlen 2048
```

## Notes

- GPTQ and TFIC optimize weighted reconstruction energy, not plain MSE.
- Final model quality should be compared with PPL and downstream `lm-eval`.
- AWQ without `AWQ_SCALES_PT` is skipped by `run_full_baselines.sh`.
- `run_full_baselines.sh` logs W&B by default. Use `RUN_WANDB=0` to disable it, or `--use-wandb` for direct Python commands.
- Heavy GPU runs should be launched only after model access and disk space are ready.
