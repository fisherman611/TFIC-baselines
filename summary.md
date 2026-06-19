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

---

## Assignment Methods

No assignment-method baseline has been summarized here yet. Add entries only
after the corresponding code path is implemented or cleaned up for this task.
