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

These wrappers expose the existing EigenFlip assignment code with clearer
experiment labels.

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
integer codes, so the smoke test uses the asymmetric vanilla grid.
