# Full FlatQuant Integration Plan

This plan tracks the work needed to implement the full FlatQuant baseline from
`ruikangliu/FlatQuant` in this repository.

## Current State

Implemented:

- `flatquant_diag`: a fixed-grid-compatible ablation using per-channel
  diagonal scale `c` and optional weight clipping.

Not implemented:

- Kronecker affine transforms `P = P1 x P2`
- online activation transforms `XP`
- K/V cache transforms
- activation and KV clipping/quantization
- FlatQuant model wrappers and reparameterization

## Why Full FlatQuant Needs Model Integration

The core FlatQuant relation is:

```text
Y = X W^T ~= Q(XP) Q(P^-1 W^T)
```

The transformed weight and transformed activation must be used together. A
checkpoint-only runner that only replaces `module.weight` cannot represent this
unless the activation path is also changed or the model is fully
reparameterized.

## Implementation Phases

### Phase 1: Official Parameter Compatibility

- Define a loader for official FlatQuant parameter files such as
  `flat_matrices.pth`.
- Normalize parameter names to this repo's layer names.
- Validate required transforms for attention and MLP layers:
  - attention: `ln_trans`, `kcache_trans`, `vcache_trans`, `o_trans`
  - MLP: `up_gate_trans`, `down_trans`
- Fail closed when transforms are missing or have incompatible shapes.

### Phase 2: Transform Modules

- Implement reusable Kronecker/SVD transform modules.
- Support applying transforms to:
  - activations
  - linear weights
  - per-head K/V cache tensors
- Add unit tests with toy matrices verifying:
  - `XP` matches explicit full-matrix multiplication
  - `P^-1 W^T` preserves the unquantized linear result with `XP`
  - per-head K/V transform shape handling is correct

### Phase 3: Model Wrappers

- Add LLaMA-style attention and MLP wrappers mirroring official FlatQuant.
- Keep LayerNorm, RoPE, and attention scores in higher precision.
- Add a guarded entrypoint such as:

```text
--grid flatquant
--flatquant-matrices-pt <path>
```

- Initially target LLaMA-3.1-8B only, then generalize.

### Phase 4: Quantization And Clipping

- Add activation quantization in the transformed activation path.
- Add learnable/frozen clipping thresholds for:
  - weights
  - activations
  - K/V cache
- Ensure assignment methods still own integer code selection for linear
  weights, while FlatQuant owns the transformed grid and activation path.

### Phase 5: Assignment Integration

- Collect Hessian/statistics from transformed activations, not original inputs.
- For GPTAQ/GPTAQ+ResComp, collect paired full/quantized transformed activation
  streams.
- Apply RTN/GPTQ/GPTAQ/GPTAQ+ResComp/FlexRound/TFIC on transformed weights and
  dequantization grids.

### Phase 6: Validation

- Smoke test with `--max-layers 1`.
- Verify unquantized transformed wrapper matches the original layer output.
- Compare RTN W4A4 behavior against official FlatQuant on a small layer/model
  slice before running full benchmarks.
- Only mark the main FlatQuant task complete after the full transformed forward
  path is exercised in evaluation.
