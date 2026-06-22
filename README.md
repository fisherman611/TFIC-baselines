# TFIC Baselines

This repository contains post-training quantization baselines for studying:

- quantization grids / transforms: Vanilla, AWQ, NeUQI, the FlatQuant
  diagonal-scale ablation grid, and SpinQuant no-had rotation absorption are implemented
- assignment methods: RTN, GPTQ, GPTAQ, GPTAQ+ResComp, fixed-grid FlexRound, and TFIC are implemented
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
flatquant_diag: symmetric and asymmetric per-channel scale/clipping grid
spinquant: learned or random no-had R1/R2 rotation absorption, then symmetric/asymmetric grid
neuqi: Hessian-diagonal weighted grid initialization, symmetric and asymmetric
```

Assignment methods:

```text
rtn
gptq
gptaq
gptaq_rescomp
flexround
tfic
```

`flexround` is the assignment-only variant required by this repository's
factorized benchmark. It keeps the selected grid's scale and zero-point fixed,
learns FlexRound's positive element-wise divisor (plus its output-channel
factor), and minimizes the calibration reconstruction surrogate
`H_tilde = diag(D) + V V^T`. As in the paper, the integer codes produced after
the final optimization step are returned directly. The learned divisors are
not stored in the checkpoint; inference still uses ordinary integer codes and
the original Vanilla/AWQ grid.

`gptaq` is implemented in `assignment_methods/gptaq.py` as a fixed-grid port of
the official GPTAQ lazy block update. The whole-model runner collects paired
full-precision and quantized-path layer inputs to form
`dXXT = E[(X_fp - X_quant)^T X_quant]`, then applies GPTAQ on the selected
Vanilla or AWQ grid. The paired collector forces sequential one-layer batches
for GPTAQ-style methods so later layers see the quantized path created by
earlier layers.

`gptaq_rescomp` is implemented separately in
`assignment_methods/gptaq_rescomp.py`. It adds the compensation-aware residual
from *Rethinking Residual Errors in Compensation-based LLM Quantization*,
reuses the paired GPTAQ statistics, and adds the `P2` correction from
`X_fp^T X_quant = H + dXXT` while keeping the same fixed Vanilla/AWQ grid.
The default residual scale follows the reference implementation
(`rescomp_alpha=0.25`), with 2-bit using the original GPTAQ-style update and
3-bit or higher using ResComp's `allw` update.

`flatquant_diag` is implemented as a fixed-grid-compatible ablation of
FlatQuant: the learned pair-wise per-channel scale `c` and optional learned
weight clipping threshold `alpha_w`. It consumes params produced elsewhere
through `--flatquant-params-pt`. This is not the full FlatQuant baseline from
the official repo. Full FlatQuant's non-diagonal Kronecker affine transforms
require online activation/KV-cache transforms and model reparameterization, so
they must be integrated at the model-forward level before assignment methods
can be applied correctly.

`spinquant` implements the SpinQuant no-had inference path for LLaMA/Mistral-
style models: it fuses RMSNorm scales, absorbs learned residual rotation `R1`
and per-layer value/output-head rotation `R2`, then applies the selected
assignment method to the ordinary uniform weight grid. Use
`--spinquant-rotations-pt` for a learned checkpoint containing `R1` and
`model.layers.{i}.self_attn.R2`. `--spinquant-random-rotations` exists only for
pipeline smoke/debug runs; it is not the learned SpinQuant baseline.

`neuqi` initializes a uniform grid by minimizing the diagonal-Hessian weighted
reconstruction loss from NeUQI. It uses `stats.diag_H` collected from
calibration inputs and searches scale with the paper defaults `T=2048`,
`T_c=64`. The asymmetric variant stores floating-point zero-points using the
paper's transition-point search: Algorithm 3 on the Eq. 8 approximation
followed by Algorithm 4 on Eq. 7 in the local interval. The symmetric variant
keeps zero-point fixed at 0 and searches the weighted scale/clipping factor on
signed integer codes.

```python
from assignment_methods import GPTAQAssignment, stats_from_paired_inputs

stats = stats_from_paired_inputs(x_quantized, x_full_precision)
weights, info = GPTAQAssignment(
    damp=0.01,
    block_size=128,
    alpha=0.25,
).apply_to_grid(grid, stats)
```

Assignment-level smoke test:

```bash
python -m pytest tests/test_gptaq_assignment.py -q
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
GRIDS="vanilla" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
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
GRIDS="awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
  uv run bash run_full_baselines.sh
```

Run both Vanilla and AWQ:

```bash
AWQ_SCALES_PT_ASYMMETRIC=./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
AWQ_SCALES_PT_SYMMETRIC=./outputs/awq_scales/llama31_8b_awq_sym_w3g128_c4n128.pt \
GRIDS="vanilla awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
  uv run bash run_full_baselines.sh
```

## Multi-GPU Runs

Use `CUDA_VISIBLE_DEVICES` to choose the GPUs visible to the run. The full
runner passes these settings into quantization and PPL eval:

```text
MODEL_DEVICE_MAP=auto       Transformers placement policy
INPUT_DEVICE=auto           input_ids go to the first model device
STATS_DEVICE=layer          streaming stats/Hessian live on each layer device
```

Recommended 2-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
MODEL_DEVICE_MAP=auto \
INPUT_DEVICE=auto \
STATS_DEVICE=layer \
GRIDS="vanilla" SCHEMES="asymmetric" ASSIGNMENTS="gptq tfic" \
uv run bash run_full_baselines.sh
```

Recommended 4-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DEVICE_MAP=balanced \
INPUT_DEVICE=auto \
STATS_DEVICE=layer \
GRIDS="vanilla awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
uv run bash run_full_baselines.sh
```

For lower GPU memory at the cost of speed:

```bash
STATS_DEVICE=cpu uv run bash run_full_baselines.sh
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
DELETE_CHECKPOINT=1 GRIDS="vanilla" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
  uv run bash run_full_baselines.sh
```

## Single-Cell Commands

AWQ + fixed-grid FlexRound:

```bash
uv run python run_quantization_baseline.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_awq_asymmetric_flexround_w3g128_c4n128 \
  --grid awq \
  --awq-scales-pt ./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
  --scheme asymmetric \
  --assignment flexround \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --k 16 \
  --layer-batch-size 4 \
  --flexround-steps 5000 \
  --flexround-lr 2e-4
```

For a quick pipeline check, lower `--flexround-steps`; use 5000 for the paper's
reconstruction-step budget. `--no-flexround-row-scale` disables the additional
output-channel factor. The full runner exposes the same settings through
`FLEXROUND_STEPS`, `FLEXROUND_LR`, and `FLEXROUND_LOG_DIVISOR_BOUND`.

FlatQuant diagonal-scale grid + RTN:

```bash
uv run python run_quantization_baseline.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_flatquant_diag_asymmetric_rtn_w3g128_c4n128 \
  --grid flatquant_diag \
  --flatquant-params-pt ./outputs/flatquant/llama31_8b_flatquant_params.pt \
  --scheme asymmetric \
  --assignment rtn \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048
```

The FlatQuant diagonal-scale params file can map each linear layer name to a scale tensor:

```python
{"model.layers.0.mlp.down_proj": scale_tensor}
```

or to a dict with optional clipping:

```python
{
    "model.layers.0.mlp.down_proj": {
        "scales": scale_tensor,
        "weight_clip": alpha_w,
    }
}
```

SpinQuant no-had + RTN with learned rotations:

```bash
uv run python run_quantization_baseline.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_spinquant_asymmetric_rtn_w3g128_c4n128 \
  --grid spinquant \
  --spinquant-rotations-pt ./outputs/spinquant/llama31_8b_R.bin \
  --scheme asymmetric \
  --assignment rtn \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048
```

For a one-layer smoke run without a learned checkpoint, replace the rotations
path with `--spinquant-random-rotations --spinquant-random-seed 42` and add
`--no-save --n-calib 1 --seqlen 128 --max-layers 1`.

NeUQI + RTN:

```bash
uv run python run_quantization_baseline.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_neuqi_asymmetric_rtn_w3g128_c4n128 \
  --grid neuqi \
  --scheme asymmetric \
  --assignment rtn \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048
```

Minimal AWQ FlexRound smoke run using the generated asymmetric AWQ scales:

```bash
python run_quantization_baseline.py \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/flexround_smoke \
  --run-name llama31_8b_awq_asym_flexround_smoke \
  --grid awq \
  --awq-scales-pt ./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
  --scheme asymmetric \
  --assignment flexround \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 1 \
  --seqlen 128 \
  --k 0 \
  --flexround-steps 1 \
  --flexround-lr 2e-4 \
  --device-map auto \
  --input-device auto \
  --stats-device layer
```

For FlexRound, `--k 0` selects the lightweight diagonal-plus-mean
reconstruction surrogate. It avoids full Gram matrices and collects all linear
layers in one calibration pass. This is intended only for smoke testing; use
`--k 16`, the full calibration settings, and more optimization steps for the
actual benchmark.

Run smoke checks for every implemented assignment method across grid baselines:

```bash
bash run_assignment_smokes.sh
```

The script runs `rtn`, `gptq`, `gptaq`, `gptaq_rescomp`, `flexround`, and
`tfic` over `vanilla`, `awq`, `flatquant_diag`, `spinquant`, and `neuqi` where
the required grid artifacts are available. Each cell uses one calibration
sample and quantizes only the first linear layer via `--max-layers 1`. It also
passes `--no-save`, so the smoke cells do not write full LLaMA checkpoints.
The script verifies model loading, calibration, grid construction, and
assignment only.

Run selected grids or methods only:

```bash
GRIDS="vanilla neuqi" SCHEMES="asymmetric symmetric" bash run_assignment_smokes.sh
METHODS="rtn flexround" bash run_assignment_smokes.sh
```

Save full checkpoints from the same script:

```bash
RUN_MODE=checkpoint \
GRIDS="vanilla awq neuqi" \
SCHEMES="asymmetric symmetric" \
METHODS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
bash run_assignment_smokes.sh
```

In checkpoint mode the script writes to
`./quantized_models/baselines_llama31_8b/`, uses full layers, defaults to
`n_calib=128` and `seqlen=2048`, and auto-generates missing AWQ scale files.
FlatQuant diagonal and learned SpinQuant checkpoints still need their external
artifact paths.

Override model or artifact paths when necessary:

```bash
MODEL_PATH=/path/to/model \
AWQ_SCALES_PT=/path/to/awq_scales.pt \
FLATQUANT_PARAMS_PT=/path/to/flatquant_diag_params.pt \
SPINQUANT_ROTATIONS_PT=/path/to/spinquant_rotations.pt \
bash run_assignment_smokes.sh
```

For smoke/debug only, SpinQuant uses random rotations by default when
`SPINQUANT_ROTATIONS_PT` is unset. Set `SPINQUANT_RANDOM_ROTATIONS=0` to skip
SpinQuant unless a learned rotations checkpoint is provided.

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
