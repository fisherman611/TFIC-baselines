# Baseline Summary

This file summarizes baselines only after they are implemented.

---

## Grid Baselines

### Vanilla Quantization Grid

Status: implemented in `grid_baselines/vanilla_quantization_grid.py`.

The vanilla quantization grid directly quantizes the original weight tensor
without AWQ scaling, rotation, or any other transformation. Once the scale and
zero-point are fixed, assignment methods choose integer codes on this fixed
uniform grid.

The implementation supports two variants:

- symmetric absmax quantization
- asymmetric min-max quantization

---

#### Symmetric Vanilla Grid

Symmetric quantization fixes the zero-point to zero and uses signed integer
codes.

For bit-width `b`:

```text
s = max(abs(W)) / (2^(b - 1) - 1)
z = 0
q_min = -2^(b - 1)
q_max =  2^(b - 1) - 1
```

Each weight is assigned by round-to-nearest:

```text
q = clip(round(w / s), q_min, q_max)
w_hat = s * q
```

The grid points are:

```text
..., -2s, -s, 0, s, 2s, ...
```

Code:

```python
from grid_baselines import build_symmetric_vanilla_quantization_grid

grid = build_symmetric_vanilla_quantization_grid(
    W,
    bits=bits,
    group_size=group_size,
)
q, W_hat = grid.round_to_nearest()
```

This is the default scheme in:

```python
build_vanilla_quantization_grid(...)
```

It is also available in the EigenFlip quantization state:

```python
IntegerQuantizedTensorState.from_rtn(
    W,
    bits=bits,
    group_size=group_size,
    scheme="symmetric",
)
```

---

#### Asymmetric Vanilla Grid

Asymmetric quantization uses the min and max values in each group and learns a
zero-point.

For bit-width `b`:

```text
s = (w_max - w_min) / (2^b - 1)
z = round(-w_min / s)
q_min = 0
q_max = 2^b - 1
```

Each weight is assigned by round-to-nearest:

```text
q = clip(round(w / s + z), q_min, q_max)
w_hat = s * (q - z)
```

Code:

```python
from grid_baselines import build_asymmetric_vanilla_quantization_grid

grid = build_asymmetric_vanilla_quantization_grid(
    W,
    bits=bits,
    group_size=group_size,
)
q, W_hat = grid.round_to_nearest()
```

This is the default EigenFlip RTN state for backward compatibility:

```python
IntegerQuantizedTensorState.from_rtn(...)
IntegerQuantizedTensorState.from_rtn(..., scheme="asymmetric")
```

---

### AWQ Quantization Grid

Status: implemented in `grid_baselines/awq_quantization_grid.py`.

AWQ rescales input channels before quantization, then folds the inverse scale
back into the dequantized weights.

Given per-input-channel AWQ scale `a`:

```text
W_scaled = W * a
```

The implementation supports both asymmetric and symmetric quantization on
`W_scaled`.

#### Asymmetric AWQ Grid

Asymmetric AWQ applies min-max quantization to `W_scaled`:

```text
s_q = (w_scaled_max - w_scaled_min) / (2^b - 1)
z = round(-w_scaled_min / s_q)
q_min = 0
q_max = 2^b - 1
q = clip(round(W_scaled / s_q + z), q_min, q_max)
```

The dequantized weight is mapped back to the original coordinate:

```text
W_hat = (q - z) * s_q / a
```

Equivalently, the effective dequantization scale stored by the grid is:

```text
s_eff = s_q / a
W_hat = (q - z) * s_eff
```

Code:

```python
from grid_baselines import build_asymmetric_awq_quantization_grid

grid = build_asymmetric_awq_quantization_grid(
    W,
    awq_scales,
    bits=bits,
    group_size=group_size,
)
q, W_hat = grid.round_to_nearest()
```

This module matches the existing EigenFlip AWQ base:

```python
IntegerQuantizedTensorState.from_awq(W, awq_scales, bits, group_size)
IntegerQuantizedTensorState.from_awq(
    W,
    awq_scales,
    bits,
    group_size,
    scheme="asymmetric",
)
```

#### Symmetric AWQ Grid

Symmetric AWQ applies signed absmax quantization after AWQ scaling:

```text
s_q = max(abs(W_scaled)) / (2^(b - 1) - 1)
z = 0
q_min = -2^(b - 1)
q_max =  2^(b - 1) - 1
q = clip(round(W_scaled / s_q), q_min, q_max)
```

The dequantized weight is still mapped back to the original coordinate:

```text
W_hat = q * s_q / a
```

Code:

```python
from grid_baselines import build_symmetric_awq_quantization_grid

grid = build_symmetric_awq_quantization_grid(
    W,
    awq_scales,
    bits=bits,
    group_size=group_size,
)
q, W_hat = grid.round_to_nearest()
```

It is also available through the EigenFlip AWQ state:

```python
IntegerQuantizedTensorState.from_awq(
    W,
    awq_scales,
    bits,
    group_size,
    scheme="symmetric",
)
```

---

## Assignment Methods

Status: implemented in `assignment_methods/`.

These modules expose RTN, GPTQ, GPTAQ, FlexRound, and TFIC through a common
fixed-grid assignment interface.

### RTN

Round-to-nearest assigns each weight independently to the nearest point on the
fixed grid:

```text
q = clip(round(w / s + z), q_min, q_max)
w_hat = s * (q - z)
```

Code:

```python
from assignment_methods import RTNAssignment

W_hat, info = RTNAssignment().apply_to_grid(grid)
```

### GPTQ

GPTQ uses a calibration Hessian / Gram matrix to assign codes sequentially while
compensating later coordinates for earlier quantization errors.

Objective:

```text
minimize Tr((W_hat - W) H (W_hat - W)^T)
H = E[x x^T]
```

Code:

```python
from assignment_methods import GPTQAssignment

W_hat, info = GPTQAssignment().apply_to_grid(grid, stats)
```

### GPTAQ

Status: implemented in `assignment_methods/gptaq.py` and wired into
`run_quantization_baseline.py`.

GPTAQ extends GPTQ with paired asymmetric calibration statistics:

```text
H = E[X_quant^T X_quant]
dXXT = E[(X_fp - X_quant)^T X_quant]
P = alpha * triu(dXXT U^T, diagonal=1) U
```

Here, `U` is the upper Cholesky factor of the damped inverse Hessian. The
assignment performs GPTQ's lazy-block update plus the correction involving
`P`, while keeping the selected Vanilla/AWQ scale and zero-point fixed.

```python
from assignment_methods import GPTAQAssignment, stats_from_paired_inputs

stats = stats_from_paired_inputs(x_quantized, x_full_precision)
W_hat, info = GPTAQAssignment(
    damp=0.01,
    block_size=128,
    alpha=0.25,
).apply_to_grid(grid, stats)
```

When `X_fp == X_quant`, `dXXT = 0`, so GPTAQ reduces exactly to GPTQ. It has no
RTN fallback and returns the final lazy-block assignments.

The whole-model collector now supports the paired FP/quantized activation
stream required by GPTAQ. For GPTAQ runs it forces sequential one-layer
batches, temporarily restores previous layers to their full-precision weights
to collect `X_fp`, restores the quantized path to collect `X_quant`, and stores
`delta_cross`. Missing `delta_cross` still raises an error instead of silently
running GPTQ under the GPTAQ label.

### FlexRound

Status: implemented in `assignment_methods/flexround.py`.

The repository implements FlexRound as an assignment-only method so it can be
compared fairly across the same fixed Vanilla or AWQ grid. The grid scale and
zero-point remain unchanged. FlexRound learns a positive divisor for each
weight and an optional output-channel factor:

```text
S = exp(log_S_element + log_S_row)
q = clip(round(W / (s * S) + z), q_min, q_max)
W_hat = s * (q - z)
```

`log_S_element` has the same shape as the padded weight matrix.
`log_S_row` has shape `[out_features, 1]` and corresponds to the additional
output-channel factor used by FlexRound for linear layers. Both are initialized
to zero, so `S = 1` and optimization starts exactly from RTN.

Rounding uses a straight-through estimator:

```text
round_ste(x) = x + stop_gradient(round(x) - x)
```

The assignment is optimized against the calibration reconstruction surrogate
already stored in `LayerStats`:

```text
H_tilde = diag(D) + V V^T
R = W_hat - W
L = Tr(R H_tilde R^T) / out_features
```

This is the layer-output reconstruction energy under the low-rank-plus-diagonal
second-moment approximation collected from calibration activations. Following
the paper, the method returns the hard integer codes obtained after the final
optimization step. It does not retain the best intermediate codes and does not
fall back to RTN.

Code:

```python
from assignment_methods import FlexRoundAssignment

assignment = FlexRoundAssignment(
    steps=5000,
    lr=2e-4,
    log_divisor_bound=6.0,
    learn_row_scale=True,
)
W_hat, info = assignment.apply_to_grid(grid, stats)
```

The returned `info` contains:

```text
variant = fixed_grid_surrogate
codes
initial_loss
final_loss
changed_codes
changed_fraction
```

The learned divisors are discarded after integer codes are selected. Inference
stores only the ordinary grid codes, scale, and zero-point, so FlexRound adds no
per-weight inference metadata.

Important scope note: this is the fixed-grid assignment variant required by
this repository's `grid x assignment` design. Full FlexRound from the paper
also learns the quantization grid scale and can reconstruct an entire block.
Learning the grid scale here would change both the grid and assignment at once,
so it is deliberately excluded from this baseline.

### TFIC

TFIC starts from base integer codes and edits them with spin-flip moves to
reduce reconstruction energy:

```text
E = Tr(R G R^T)
R = W_hat - W
G = H = E[x x^T]
```

Code:

```python
from assignment_methods import TFICAssignment

W_hat, info = TFICAssignment().apply_to_grid(grid, stats)
```

Current note: the wrapped EigenFlip TFIC implementation assumes non-negative
integer code updates internally. Signed symmetric grids are shifted to an
equivalent non-negative representation by `assignment_methods/state_adapter.py`;
the represented dequantized weights are unchanged.

---

## Real Model Runner

Status: implemented in `run_quantization_baseline.py`.

This runner uses the modular baseline folders directly:

```text
grid_baselines      -> builds the quantization grid
assignment_methods  -> assigns integer codes on that grid
```

### Calibration And Evaluation Protocol

Calibration is loaded in `calibration_utils.py`.

Default real-run calibration:

```text
dataset: C4 train shard
source: https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-train.00000-of-01024.json.gz
samples: 128
sequence length: 2048
sampling: random 2048-token window inside each long document
cache: ./calibration_cache
```

Perplexity evaluation is loaded in `eval_ppl.py`.

Evaluation datasets:

```text
WikiText2: wikitext-2-raw-v1, split=test
C4: allenai/c4 validation shard en/c4-validation.00000-of-00008.json.gz
```

So the normal protocol is:

```text
calibrate on C4 train windows
evaluate on WikiText2 test
evaluate on C4 validation windows
```

Use a different C4 eval seed from the calibration seed when you want to make
the separation explicit.

### Quantization Commands

Vanilla + RTN:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid vanilla `
  --scheme asymmetric `
  --assignment rtn `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --run-name vanilla_asym_rtn_w3g128_c4n128
```

Vanilla + GPTQ:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid vanilla `
  --scheme asymmetric `
  --assignment gptq `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --layer-batch-size 4 `
  --eig-on-cpu `
  --run-name vanilla_asym_gptq_w3g128_c4n128
```

Vanilla + TFIC:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid vanilla `
  --scheme asymmetric `
  --assignment tfic `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --layer-batch-size 4 `
  --eig-on-cpu `
  --run-name vanilla_asym_tfic_w3g128_c4n128
```

Vanilla + FlexRound:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid vanilla `
  --scheme asymmetric `
  --assignment flexround `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --k 16 `
  --layer-batch-size 4 `
  --flexround-steps 5000 `
  --flexround-lr 2e-4 `
  --run-name vanilla_asym_flexround_w3g128_c4n128
```

AWQ + RTN:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid awq `
  --awq-scales-pt <path-to-awq-scales.pt> `
  --scheme asymmetric `
  --assignment rtn `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --run-name awq_asym_rtn_w3g128_c4n128
```

AWQ + GPTQ:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid awq `
  --awq-scales-pt <path-to-awq-scales.pt> `
  --scheme asymmetric `
  --assignment gptq `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --layer-batch-size 4 `
  --eig-on-cpu `
  --run-name awq_asym_gptq_w3g128_c4n128
```

AWQ + TFIC:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid awq `
  --awq-scales-pt <path-to-awq-scales.pt> `
  --scheme asymmetric `
  --assignment tfic `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --layer-batch-size 4 `
  --eig-on-cpu `
  --run-name awq_asym_tfic_w3g128_c4n128
```

AWQ + FlexRound:

```powershell
python run_quantization_baseline.py `
  --model-path <hf-model-or-local-path> `
  --grid awq `
  --awq-scales-pt <path-to-awq-scales.pt> `
  --scheme asymmetric `
  --assignment flexround `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --seed 42 `
  --k 16 `
  --layer-batch-size 4 `
  --flexround-steps 5000 `
  --flexround-lr 2e-4 `
  --run-name awq_asym_flexround_w3g128_c4n128
```

Lightweight AWQ + FlexRound smoke run:

```powershell
python run_quantization_baseline.py `
  --model-path meta-llama/Meta-Llama-3.1-8B `
  --output-dir ./quantized_models/flexround_smoke `
  --run-name llama31_8b_awq_asym_flexround_smoke `
  --grid awq `
  --awq-scales-pt ./outputs/awq_scales/llama31_8b_awq_asym_w3g128_c4n128.pt `
  --scheme asymmetric `
  --assignment flexround `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 1 `
  --seqlen 128 `
  --k 0 `
  --flexround-steps 1 `
  --flexround-lr 2e-4 `
  --device-map auto `
  --input-device auto `
  --stats-device layer
```

For FlexRound, `--k 0` uses the diagonal-plus-mean surrogate and skips dense
Gram construction, allowing the collector to gather all linear-layer moments
in one model pass. This setting verifies the pipeline only; benchmark runs
should use `--k 16` and the full calibration/optimization budget.

Smoke all implemented assignment methods:

```powershell
bash run_assignment_smokes.sh
```

`run_assignment_smokes.sh` runs RTN, GPTQ, FlexRound, and TFIC on the existing
asymmetric AWQ grid. It uses one calibration sample, sequence length 128, and
`--max-layers 1`, so only the first linear layer is quantized. It uses
`--no-save` to avoid writing four full LLaMA checkpoints. These runs check model
loading, calibration, grid construction, and assignment only.

### PPL Evaluation Commands

Evaluate a saved checkpoint on WikiText2 test and C4 validation:

```powershell
python eval_ppl.py `
  --model-path ./quantized_models/baselines/vanilla_asym_tfic_w3g128_c4n128 `
  --datasets wikitext2 c4 `
  --seqlen 2048 `
  --c4-samples 128 `
  --seed 1234
```

Run only WikiText2:

```powershell
python eval_ppl.py `
  --model-path ./quantized_models/baselines/vanilla_asym_tfic_w3g128_c4n128 `
  --datasets wikitext2 `
  --seqlen 2048
```

Run only C4 validation:

```powershell
python eval_ppl.py `
  --model-path ./quantized_models/baselines/vanilla_asym_tfic_w3g128_c4n128 `
  --datasets c4 `
  --seqlen 2048 `
  --c4-samples 128 `
  --seed 1234
```

### Legacy EigenFlip Runner

The older `eigenflip/run_fast.py` entrypoint also supports both schemes now:

```powershell
python -m eigenflip.run_fast `
  --model-path <hf-model-or-local-path> `
  --output-dir ./quantized_models/eigenflip_3bit `
  --base rtn `
  --scheme asymmetric `
  --encoder tfic_fast `
  --bits 3 `
  --group-size 128 `
  --calib-dataset c4 `
  --n-calib 128 `
  --seqlen 2048 `
  --layer-batch-size 4 `
  --eig-on-cpu
```

The saved checkpoint path is:

```text
<output-dir>/<base>_<scheme>_<encoder>
```

For example:

```text
./quantized_models/eigenflip_3bit/rtn_asymmetric_tfic_fast
```
