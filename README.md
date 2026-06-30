# TFIC Baselines

This repository contains post-training quantization baselines for studying:

- quantization grids / transforms: Vanilla, AWQ, NeUQI, FlatQuant affine
  transforms, the FlatQuant diagonal ablation, and SpinQuant are implemented
- assignment methods: RTN, GPTQ, GPTAQ, GPTAQ+ResComp, FlexRound, and TFIC are implemented
- evaluation: perplexity on WikiText2 and C4, plus `lm-eval`

Repository layout:

```text
assignment_methods/       integer-code assignment algorithms
baseline_utils/           shared calibration, runtime, and W&B utilities
eigenflip/                EigenFlip/TFIC implementation
grid_baselines/           quantization-grid implementations
scripts/                  Python and shell entry points
tests/                    automated tests
docs/methods/             method notes and research documentation
docs/plans/               implementation plans
experiments/legacy/       archived exploratory code
```

Detailed notes are indexed in [docs/README.md](docs/README.md).

## Implemented Methods

Grid baselines:

```text
vanilla: symmetric and asymmetric
awq:     symmetric and asymmetric
flatquant:      online Kronecker activation transforms with transformed weights
flatquant_diag: (fixed-grid ablation) symmetric and asymmetric per-channel scale/clipping ablation
spinquant: learned or random no-had R1/R2, then symmetric/asymmetric grid
spinquant_had: R1/R2 plus online R3/R4 for low-bit activation/K-cache
neuqi: Hessian-diagonal weighted grid initialization, asymmetric plus symmetric extension
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

`flexround` follows the official FlexRound quantizer parameterization from
FlexRound_LRQ: `delta1` is initialized from the selected grid scale and learned
by default, while `delta2` is element-wise and `delta3` is the output-channel
factor. It still uses this repository's calibration reconstruction surrogate
`H_tilde = diag(D) + V V^T` instead of FlexRound_LRQ's cached transformer-block
input/output reconstruction loop. Set `--no-flexround-learn-layer-scale` to
recover the older fixed-grid assignment-only ablation.

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

`flatquant` implements the paper's central affine relation
`Q(XP) Q(P^-1 W^T)` for normalized per-linear Kronecker transform artifacts.
Weights are stored in transformed coordinates, activations are transformed
online, and assignment statistics are collected after that transform. Official
`kcache_trans` matrices are applied to Q/K after RoPE, with optional Q/K/V
cache fake quantization. The saved manifest restores the online path in PPL
and lm-eval.

`python -m scripts.calibrate_flatquant` performs repository-native block-wise
calibration optimization. Base-model weights remain frozen; only Kronecker
factors, diagonal scales, and weight/activation clipping parameters are
optimized against the floating-point block output. Generated artifacts contain
structural model identity and quantization settings, which the runner validates
before use. Official `flat_matrices.pth` files remain supported for experiments
that need learned `kcache_trans`/`vcache_trans` in addition to W/A transforms.

`python -m scripts.calibrate_flexround` performs the official-style FlexRound
block calibration loop. It captures floating-point block outputs, replaces the
seven LLaMA-family linear projections with trainable FlexRound quantizers, and
optimizes `delta1`, `delta2`, and `delta3` with block-output MSE before saving
a model-bound artifact.

`flatquant_diag` remains a fixed-grid-compatible ablation of FlatQuant: the
learned pair-wise per-channel scale `c` and optional learned weight clipping
threshold `alpha_w`. It consumes params produced elsewhere through
`--flatquant-params-pt`. Use `flatquant` for the non-diagonal Kronecker model
path.

`spinquant` is the project-default no-had path: it fuses RMSNorm scales and
absorbs learned residual rotation `R1` plus per-layer value/output-head
rotation `R2`. `spinquant_had` additionally applies factorized online `R4` to
the MLP down projection and post-RoPE `R3` for K-cache quantization. Assignment
methods operate on the exact coordinates consumed by each path. Use
`--spinquant-rotations-pt` for a learned checkpoint containing `R1` and
`model.layers.{i}.self_attn.R2`. `--spinquant-random-rotations` exists only for
pipeline smoke/debug runs; it is not the learned SpinQuant baseline.
`python -m scripts.calibrate_spinquant` calibrates repository-native R1/R2 rotation
artifacts on calibration data with a full fake-quantized model cross-entropy
objective by default, starting from random-signed Hadamard rotations. Its
defaults match this project's C4/128/2048 setting and save a loader-compatible
`--spinquant-rotations-pt` file. Use `--rotation-init identity` or
`--rotation-init random` only for ablations, and use `--objective reconstruction`
only for the older local linear MSE debug path.
For non-power-of-two intermediate widths in `spinquant_had`, pass
`--spinquant-r4-pt` containing the official `had_K` and `K`. Use
`--activation-bits`, `--v-bits`, and
`--k-bits` for A/V/K fake quantization. Runtime transforms and quantizer
settings are restored from the saved checkpoint manifest during evaluation.

`neuqi` initializes a uniform grid by minimizing the diagonal-Hessian weighted
reconstruction loss from NeUQI. It uses `stats.diag_H` collected from
calibration inputs and searches scale with the paper defaults `T=2048`,
`T_c=64`. The asymmetric variant stores floating-point zero-points using the
paper's transition-point search: Algorithm 3 on the Eq. 8 approximation
followed by Algorithm 4 on Eq. 7 in the local interval. The symmetric variant
keeps zero-point fixed at 0 and searches the weighted scale/clipping factor on
signed integer codes; this is a repository extension, not the primary paper or
official-implementation path. `--group-size -1` is supported for NeUQI and
means one channel-wise group per output row, matching the official
implementation's `group_size=-1` behavior. The whole-model runner quantizes
layers sequentially so later layer statistics are collected after earlier
layers have been quantized, but the exact official NeUQI path is still
different because this repository does not implement or use the official LDLQ
assignment.

```python
from assignment_methods import GPTAQAssignment, stats_from_paired_inputs

stats = stats_from_paired_inputs(x_quantized, x_full_precision)
weights, info = GPTAQAssignment(
    damp=0.01,
    block_size=128,
).apply_to_grid(grid, stats)
```

The default `alpha=1.0` applies the asymmetric correction exactly as written
in GPTAQ Algorithm 1. `--gptaq-alpha` remains available for ablations.

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
  uv run bash scripts/run_full_baselines.sh
```

`scripts/run_full_baselines.sh` logs to W&B by default. To disable W&B for a debug run:

```bash
RUN_WANDB=0 GRIDS="vanilla" SCHEMES="asymmetric" ASSIGNMENTS="rtn" RUN_LM_EVAL=0 \
  uv run bash scripts/run_full_baselines.sh
```

## Run Vanilla Baselines

Run all implemented assignment methods on the Vanilla grid:

```bash
GRIDS="vanilla" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
  uv run bash scripts/run_full_baselines.sh
```

Each cell runs:

```text
1. quantization with `python -m scripts.run_quantization_baseline`
2. PPL eval on WikiText2 test and C4 validation
3. lm-eval with the extended task preset
```

## Generate AWQ Scales

AWQ requires per-layer calibration artifacts before running the AWQ grid. The
generator uses the paper statistic `mean(abs(X))`, searches
`scale = activation_scale ** alpha`, and then searches symmetric clipping
bounds per output channel and weight group. As in the official AWQ code, Q/K
projections skip clipping because their error is coupled by the QK product.

New artifacts include model, bit-width, group-size, and quantization-scheme
metadata; the runner rejects mismatched artifacts. Legacy scale-only `.pt`
files remain loadable, but do not provide AWQ weight clipping.

Asymmetric AWQ scales:

```bash
uv run python -m scripts.generate_awq_scales \
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
uv run python -m scripts.generate_awq_scales \
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

This repository evaluates AWQ as a fixed accuracy grid and saves dense
dequantized checkpoints. It does not provide AWQ integer packing or TinyChat
kernels. Assignments other than `rtn` are intentional `AWQ grid x assignment`
hybrids rather than the standalone AWQ algorithm.

Run AWQ asymmetric and symmetric:

```bash
AWQ_SCALES_PT_ASYMMETRIC=./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
AWQ_SCALES_PT_SYMMETRIC=./outputs/awq_scales/llama31_8b_awq_sym_w3g128_c4n128.pt \
GRIDS="awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
  uv run bash scripts/run_full_baselines.sh
```

Run both Vanilla and AWQ:

```bash
AWQ_SCALES_PT_ASYMMETRIC=./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt \
AWQ_SCALES_PT_SYMMETRIC=./outputs/awq_scales/llama31_8b_awq_sym_w3g128_c4n128.pt \
GRIDS="vanilla awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
  uv run bash scripts/run_full_baselines.sh
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
uv run bash scripts/run_full_baselines.sh
```

Recommended 4-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MODEL_DEVICE_MAP=balanced \
INPUT_DEVICE=auto \
STATS_DEVICE=layer \
GRIDS="vanilla awq" SCHEMES="asymmetric symmetric" ASSIGNMENTS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
uv run bash scripts/run_full_baselines.sh
```

For lower GPU memory at the cost of speed:

```bash
STATS_DEVICE=cpu uv run bash scripts/run_full_baselines.sh
```

## Outputs

By default, `scripts/run_full_baselines.sh` uses:

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
  uv run bash scripts/run_full_baselines.sh
```

## Single-Cell Commands

Calibrate FlatQuant transforms:

```bash
uv run python -m scripts.calibrate_flatquant \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --weight-bits 4 \
  --activation-bits 4 \
  --weight-symmetric \
  --activation-symmetric \
  --weight-group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --epochs 15 \
  --batch-size 4 \
  --learning-rate 5e-3 \
  --out ./outputs/flatquant/llama31_8b_flatquant_w4a4_c4n128.pt
```

This command follows the paper's block-wise MSE optimization while retaining
the project's common group-size-128 weight grid. The official FlatQuant
W4A4KV4 preset instead uses per-output-channel weight quantization and
separately learned K/V-cache transforms. Use an official `flat_matrices.pth`
artifact when reproducing that exact deployment setting.

Calibrate FlexRound quantizers with cached block reconstruction:

```bash
uv run python -m scripts.calibrate_flexround \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./outputs/flexround \
  --weight-bits 4 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --iters 5000 \
  --batch-size 4 \
  --learning-rate 3e-3 \
  --save-model-dir ./quantized_models/flexround_llama31_8b_w4
```

This is the preferred FlexRound path: it learns `delta1`, `delta2`, and
`delta3` jointly against block-output MSE, applies the exported weights back to
the model, and propagates optimized quantized outputs into the next block.
The saved checkpoint can be evaluated directly. This remains weight-only;
the paper's QDrop activation-quantization schedule is not implemented.

Full affine FlatQuant + GPTQ on the project grid:

```bash
uv run python -m scripts.run_quantization_baseline \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_flatquant_gptq_w4a4 \
  --grid flatquant \
  --flatquant-transforms-pt ./outputs/flatquant/llama31_8b_transforms.pt \
  --assignment gptq \
  --scheme symmetric \
  --bits 4 \
  --activation-bits 4 \
  --activation-symmetric \
  --group-size 128
```

Verify that an artifact preserves the real model before quantization:

```bash
uv run python -m scripts.check_flatquant_parity \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --flatquant-transforms-pt ./outputs/flatquant/llama31_8b_flatquant_w4a4_c4n128.pt \
  --atol 5e-3 \
  --out ./outputs/flatquant/llama31_8b_parity.json
```

The normalized transform artifact maps each linear module to
`matrix_left`, `matrix_right`, and optionally `diagonal`,
or `weight_clip`.

AWQ + FlexRound:

```bash
uv run python -m scripts.run_quantization_baseline \
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
  --flexround-lr 3e-3
```

For a quick pipeline check, lower `--flexround-steps`; use 5000 for the paper's
reconstruction-step budget. `--no-flexround-row-scale` disables the additional
output-channel factor, and `--no-flexround-learn-layer-scale` disables learned
`delta1` and now preserves the selected grid exactly. This runner path uses the
low-rank layer surrogate and is an assignment ablation, not the preferred
block-reconstruction reproduction. The full runner exposes the same settings through
`FLEXROUND_STEPS`, `FLEXROUND_LR`, and `FLEXROUND_LOG_DIVISOR_BOUND`.

FlatQuant diagonal-scale grid + RTN:

```bash
uv run python -m scripts.run_quantization_baseline \
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

SpinQuant + RTN with learned rotations:

```bash
uv run python -m scripts.calibrate_spinquant \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --out ./outputs/spinquant/llama31_8b_R.pt \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048 \
  --weight-bits 4 \
  --weight-group-size 128
```

```bash
uv run python -m scripts.run_quantization_baseline \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/baselines_llama31_8b \
  --run-name llama31_8b_spinquant_asymmetric_rtn_w3g128_c4n128 \
  --grid spinquant \
  --spinquant-rotations-pt ./outputs/spinquant/llama31_8b_R.pt \
  --scheme asymmetric \
  --assignment rtn \
  --bits 3 \
  --group-size 128 \
  --calib-dataset c4 \
  --n-calib 128 \
  --seqlen 2048
```

Use `--grid spinquant_had --spinquant-r4-pt <R4.pt>` when activation or
K-cache quantization requires online R3/R4.

For a one-layer smoke run without a learned checkpoint, replace the rotations
path with `--spinquant-random-rotations --spinquant-random-seed 42` and add
`--no-save --n-calib 1 --seqlen 128 --max-layers 1`.

NeUQI + RTN:

```bash
uv run python -m scripts.run_quantization_baseline \
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

Use `--group-size -1` for the official channel-wise NeUQI grid shape.

Minimal AWQ FlexRound smoke run using the generated asymmetric AWQ scales:

```bash
python -m scripts.run_quantization_baseline \
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
  --flexround-lr 3e-3 \
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
bash scripts/run_assignment_smokes.sh
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
GRIDS="vanilla neuqi" SCHEMES="asymmetric symmetric" bash scripts/run_assignment_smokes.sh
METHODS="rtn flexround" bash scripts/run_assignment_smokes.sh
```

Save full checkpoints from the same script:

```bash
RUN_MODE=checkpoint \
GRIDS="vanilla awq neuqi" \
SCHEMES="asymmetric symmetric" \
METHODS="rtn gptq gptaq gptaq_rescomp flexround tfic" \
bash scripts/run_assignment_smokes.sh
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
bash scripts/run_assignment_smokes.sh
```

For smoke/debug only, SpinQuant uses random rotations by default when
`SPINQUANT_ROTATIONS_PT` is unset. Set `SPINQUANT_RANDOM_ROTATIONS=0` to skip
SpinQuant unless a learned rotations checkpoint is provided.

Vanilla + TFIC:

```bash
uv run python -m scripts.run_quantization_baseline \
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
uv run python -m scripts.eval_ppl \
  --model-path ./quantized_models/baselines_llama31_8b/llama31_8b_vanilla_asymmetric_tfic_w3g128_c4n128 \
  --datasets wikitext2 c4 \
  --seqlen 2048 \
  --c4-samples 128 \
  --seed 1234
```

Evaluate PPL and log it to W&B:

```bash
uv run python -m scripts.eval_ppl \
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
uv run python -m scripts.log_wandb_results \
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
uv run python -m eigenflip.run_fast \
  --model-path meta-llama/Meta-Llama-3.1-8B \
  --output-dir ./quantized_models/eigenflip_3bit \
  --bits 3 --group-size 128 --k 16 \
  --base rtn --scheme asymmetric --encoder tfic_fast \
  --calib-dataset c4 --n-calib 128 --seqlen 2048 \
  --layer-batch-size 4 --eig-on-cpu

uv run python -m scripts.eval_ppl \
  --model-path ./quantized_models/eigenflip_3bit/rtn_asymmetric_tfic_fast \
  --datasets wikitext2 c4 --seqlen 2048
```

## Notes

- GPTQ and TFIC optimize weighted reconstruction energy, not plain MSE.
- Final model quality should be compared with PPL and downstream `lm-eval`.
- AWQ without `AWQ_SCALES_PT` is skipped by `scripts/run_full_baselines.sh`.
- `scripts/run_full_baselines.sh` logs W&B by default. Use `RUN_WANDB=0` to disable it, or `--use-wandb` for direct Python commands.
- Heavy GPU runs should be launched only after model access and disk space are ready.
