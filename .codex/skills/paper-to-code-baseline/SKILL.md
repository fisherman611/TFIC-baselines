---
name: paper-to-code-baseline
description: Use when implementing a research paper baseline from a method description, equations, or pseudocode.
---

Workflow:
1. Extract only implementation-relevant parts:
   - inputs
   - outputs
   - objective/loss
   - algorithm steps
   - hyperparameters
2. Map each part to existing repo components.
3. Implement behind a flag/config.
4. Do not replace existing baselines.
5. Add a minimal sanity test or dry-run.
6. Document deviations from the paper clearly.