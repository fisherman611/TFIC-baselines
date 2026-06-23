---
name: experiment-runner
description: Use when adding, running, or debugging ML experiments, configs, baselines, metrics, or evaluation scripts.
---

Workflow:
1. Identify the experiment entrypoint.
2. Identify config files and output directory.
3. Never modify dataset files or checkpoints.
4. Prefer adding a new config over changing default configs.
5. Keep experiment names explicit.
6. For new baselines, add:
   - config
   - runner hook
   - metric logging
   - README command
7. Verification should use a tiny dry-run if available.