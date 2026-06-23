## Experimental Plan: Assignment Methods on Different Quantization Grids

### 1. Goal

The goal is to evaluate different **assignment / rounding / compensation methods** for post-training quantization of LLMs, and analyze how they perform under different **quantization grid or transformation settings**.

We will first establish results on the most important grid setting, **AWQ**, then gradually expand to other grid baselines.

---

### 2. Assignment Methods

We compare the following assignment methods:

#### Main baselines

- [X] ~~**RTN**~~
- [X] ~~**GPTQ**~~
- [X] ~~**FlexRound**~~
- [X] ~~**GPTAQ**~~
- [X] ~~**GPTAQ + ResComp**~~
- [X] ~~**TFIC**~~

`GPTAQ + ResComp` is based on *Rethinking Residual Errors in Compensation-based LLM Quantization*, ICLR 2026.

#### Backup baseline

- [ ] **AdaRound**

AdaRound will be run later if time and computational resources allow.

---

### 3. Quantization Grid / Transformation Baselines

The assignment methods will be applied on top of the following quantization grid or transformation settings.

#### Main grid baselines

- [X] ~~**Vanilla quantization grid**~~
- [X] ~~**AWQ**~~
- [X] ~~**FlatQuant**~~
- [X] ~~**SpinQuant**~~
- [X] ~~**NeUQI**~~

Implemented support note:

- [X] ~~**SpinQuant no-had learned rotation grid (`spinquant`)**~~
- [X] ~~**FlatQuant diagonal-scale fixed grid (`flatquant_diag`)**~~
- [X] ~~**NeUQI Hessian-diagonal affine grid (`neuqi`)**~~

`flatquant_diag` is not the full FlatQuant baseline. It only covers the
fixed-grid-compatible diagonal scale and weight clipping subset. Full
FlatQuant still requires model forward/reparameterization support for the
learned affine/Kronecker transforms.

`spinquant` absorbs supplied no-had `R1`/`R2` rotations into LLaMA/Mistral-style
models, then quantizes the rotated weights on the existing uniform grid. Full
SpinQuant still requires Cayley-SGD rotation learning plus optional online
Hadamard `R3`/`R4`, activation quantization, and KV-cache quantization paths.

`neuqi` initializes a uniform grid with calibration `diag_H` and a scale
search. The asymmetric variant uses floating-point zero-points; the symmetric
variant keeps zero-point fixed at 0 on signed integer codes. It is a grid
initialization baseline, so the existing assignment methods can run on top of
its fixed `scale`/`zero_point`.

AWQ is the first priority.

#### Backup grid baselines

- [ ] **SmoothQuant**
- [ ] **QuaRot**

These backup settings will be run after the main experiments are completed.

---

### 4. Running Priority

#### Phase 1: AWQ-first experiments

Start with the **AWQ grid** and run the main assignment methods:

- [ ] RTN
- [ ] GPTQ
- [ ] FlexRound
- [ ] GPTAQ
- [ ] GPTAQ + ResComp
- [ ] TFIC

This phase is used to debug the pipeline, verify implementation correctness, and obtain the first main comparison.

#### Phase 2: Extension to other main grids

After the AWQ experiments are stable, extend the same assignment methods to:

- [ ] Vanilla grid
- [ ] FlatQuant
- [ ] SpinQuant
- [ ] NeUQI

This phase evaluates whether the assignment method is robust across different grid constructions.

#### Phase 3: Backup experiments

If time and resources are sufficient, run:

- [ ] AdaRound as an additional assignment baseline
- [ ] SmoothQuant and QuaRot as additional grid baselines

---

### 5. Models

Experiments will be conducted on the following LLMs:

- [ ] **LLaMA-3.1-8B**
- [ ] **Qwen2.5-7B**
- [ ] **Mistral-V3-7B**

The recommended starting model is **LLaMA-3.1-8B**, since it can be used to debug and stabilize the full experimental pipeline before scaling to the other models.

#### 5.1 Architecture compatibility requirements

Every grid and assignment cell must start from a fresh base model. Grid
artifacts are model-specific and must not be reused across these architectures.
Use the same C4 document indices and random seed for fairness, but tokenize the
documents with each model's own tokenizer.

Common requirements for all three models:

- Pin the exact Hugging Face model ID and revision in every run. In
  particular, resolve whether `Mistral-V3-7B` means
  `mistralai/Mistral-7B-v0.3` before producing artifacts.
- Discover dimensions from `model.config`; do not hard-code hidden size,
  intermediate size, head count, KV-head count, or head dimension.
- Preserve GQA correctly: Q uses `num_attention_heads`, while K/V use
  `num_key_value_heads`.
- Preserve projection biases when fusing FlatQuant or SpinQuant transforms.
- Apply Q/K cache transforms after RoPE and before cache insertion.
- Preserve the installed Transformers cache API, attention implementation
  (`eager`, SDPA, or Flash Attention), generation with `use_cache=True`, and
  checkpoint save/reload.
- Validate that every block contains the expected `q_proj`, `k_proj`,
  `v_proj`, `o_proj`, `up_proj`, `gate_proj`, and `down_proj` modules before
  quantization.

Model-specific work:

- **LLaMA-3.1-8B**: use `LlamaAttention` as the reference implementation;
  validate GQA, long-context RoPE, and the non-power-of-two FFN dimension used
  by SpinQuant R4.
- **Qwen2.5-7B**: add a `Qwen2Attention` adapter, preserve Q/K/V biases and its
  `sliding_window` argument, support `Qwen2RMSNorm`, and validate FlatQuant's
  official Qwen parameter layout. SpinQuant-had must fail closed until an
  exact R4 factorization is available for the model's intermediate dimension.
- **Mistral-V3-7B**: after pinning the exact model ID, add its
  `MistralAttention` adapter, preserve sliding-window attention and cache
  semantics, support `MistralRMSNorm`, and add native FlatQuant artifact
  conversion because the current adapter is LLaMA-only.

Required tests for each model family:

- Tiny-config full-precision parity before and after FlatQuant transforms.
- Tiny-config full-precision parity after SpinQuant R1/R2 and optional R4.
- A/K/V low-bit forward with GQA and `use_cache=True`.
- Save/reload parity for transform-aware checkpoints.
- One real-model block smoke run before starting the full experiment matrix.

---

### 6. Calibration Setting

Use the following calibration setup:

- [ ] Calibration dataset: **C4**
- [ ] Number of calibration samples: **128**
- [ ] Sequence length: **2048**

In short:

```text
Calibration: C4 / 128 samples / 2048 sequence length
```

---

### 7. Evaluation

#### 7.1 Perplexity evaluation

Evaluate perplexity on:

- [X] ~~WikiText~~
- [X] ~~C4~~

#### 7.2 Downstream task evaluation

Use `lm-eval` on the following tasks:

- [X] ~~arc_easy~~
- [X] ~~arc_challenge~~
- [X] ~~hellaswag~~
- [X] ~~piqa~~
- [X] ~~winogrande~~
- [X] ~~boolq~~
- [X] ~~rte~~
- [X] ~~openbookqa~~
- [X] ~~lambada_openai~~

---

### 8. Expected Outputs

For each experiment, we should record:

- [ ] Model name
- [ ] Quantization grid / transformation method
- [ ] Assignment method
- [ ] Bit-width and group size
- [ ] Calibration setting
- [X] ~~WikiText perplexity~~
- [X] ~~C4 perplexity~~
- [X] ~~lm-eval results for each downstream task~~
- [ ] Runtime and memory usage, if available

---

### 9. Main Result Table

The main table can be organized as follows:

| Model | Grid / Transform | Assignment | Wiki PPL | C4 PPL | MMLU | GSM8K |
| ----- | ---------------- | ---------- | -------: | -----: | ---: | ----: |

A separate detailed table can be used to report the full results for all lm-eval tasks.
