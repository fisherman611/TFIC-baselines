# Full SpinQuant Integration Plan

## Current State

Implemented:

- `spinquant`: SpinQuant no-had model reparameterization for LLaMA/Mistral-style
  models.
- Learned `R1` and per-layer `self_attn.R2` checkpoint loading.
- Optional random orthogonal `R1`/`R2` generation for smoke/debug runs.
- RMSNorm scale fusion and architecture-level absorption of residual rotation
  `R1`.
- Per-head value/output projection pair absorption of `R2`.
- Fixed-grid assignment compatibility after rotation absorption.

Not implemented:

- joint training of SpinQuant rotations with Cayley SGD
- online Hadamard rotations `R3` and `R4`
- activation and KV-cache quantization

## Scope Boundary

The implemented baseline absorbs no-had `R1`/`R2` rotations into the model,
then builds the ordinary group-wise uniform grid on the rotated weights. This
matches the no-had inference parameterization for weight-only experiments when
a learned SpinQuant rotation checkpoint is provided, but it is not the full
ICLR 2025 SpinQuant model.

Full SpinQuant must reparameterize related Transformer weights together so the
rotations preserve the floating-point network while remaining absorbed for
low-bit inference.

## Remaining Phases

1. Add Cayley-SGD rotation training on C4 calibration samples.
2. Add optional `R3` and `R4` fast Hadamard forward operators.
3. Add activation and KV-cache fake quantization for W-A-KV experiments.
4. Compare a one-layer and full-model run against the official implementation.
