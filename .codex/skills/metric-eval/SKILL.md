---
name: metric-eval
description: Use when adding or modifying evaluation metrics, benchmark reports, or result tables.
---

Rules:
1. Reuse existing metric interfaces.
2. Do not change old metric names unless required.
3. Add metrics as independent functions.
4. Save raw predictions and aggregate results separately.
5. Report both:
   - per-sample fields
   - aggregate metrics
6. Include a small example input/output.