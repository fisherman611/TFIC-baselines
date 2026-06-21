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
- [ ] **FlatQuant**
- [ ] **SpinQuant**
- [ ] **NeUQI**

Implemented support note:

- [X] ~~**FlatQuant diagonal-scale fixed grid (`flatquant_diag`)**~~

`flatquant_diag` is not the full FlatQuant baseline. It only covers the
fixed-grid-compatible diagonal scale and weight clipping subset. Full
FlatQuant still requires model forward/reparameterization support for the
learned affine/Kronecker transforms.

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
