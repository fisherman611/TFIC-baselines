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
