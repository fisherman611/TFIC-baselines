# Full SpinQuant Integration Plan

## Current State

Implemented:

- SpinQuant R1/R2 model reparameterization for LLaMA-style models.
- Learned `R1` and per-layer `self_attn.R2` checkpoint loading.
- Optional random orthogonal `R1`/`R2` generation for smoke/debug runs.
- RMSNorm scale fusion and architecture-level absorption of residual rotation
  `R1`.
- Per-head value/output projection pair absorption of `R2`.
- Fixed-grid assignment compatibility after rotation absorption.
- Per-token/group activation quantization with official `o_proj`, `down_proj`,
  `v_proj`, and `lm_head` scope rules.
- Factorized online R4 and transformed `down_proj` assignment coordinates.
- Post-RoPE R3 plus token-wise/head-wise K-cache quantization for LlamaAttention.
- Cayley-SGD orthogonality-preserving update primitive.
- Transform-aware checkpoint manifests for R3/R4 and A/V/K evaluation.
- Separate project-default `spinquant` no-had and explicit `spinquant_had` grids.
- Class-dispatched post-RoPE R3 adapters for LLaMA, Qwen2, and Mistral.
- Multi-family tiny-model parity, low-bit cache, and checkpoint round-trip tests.

Not implemented:

- joint training of SpinQuant rotations with Cayley SGD
- parity benchmark against official large-model checkpoints

## Multi-model requirements

- Make project-default `spinquant` the no-had R1/R2 path for every model;
  expose R3/R4 through a separate explicit had variant.
- Keep `LlamaAttention` as the LLaMA-3.1-8B reference implementation.
- Add model-specific post-RoPE R3 adapters for `Qwen2Attention` and
  `MistralAttention`, preserving their sliding-window and cache arguments.
- Preserve projection biases during R1/R2 absorption, especially on Qwen2.
- Derive hidden size, intermediate size, attention heads, KV heads, and head
  dimension from config and validate every rotation checkpoint against them.
- Provide or validate an exact R4 factorization for each model's intermediate
  size; fail closed when no supported factorization exists.
- Add tiny-config no-had parity, had parity, low-bit K/V cache, and checkpoint
  round-trip tests for LLaMA, Qwen2, and Mistral.

## Scope Boundary

The implemented model absorbs `R1`/`R2`, installs factorized R4, and installs
R3 only when K-cache quantization is requested. Weight assignment remains an
ordinary group-wise uniform grid after the reparameterization.

Full SpinQuant must reparameterize related Transformer weights together so the
rotations preserve the floating-point network while remaining absorbed for
low-bit inference.

## Remaining Phases

1. Add Cayley-SGD rotation training on C4 calibration samples.
2. Add a fused CUDA Hadamard kernel option; the current implementation is a
   correctness-oriented PyTorch factorization.
3. Compare a one-layer and full-model run against the official implementation.
