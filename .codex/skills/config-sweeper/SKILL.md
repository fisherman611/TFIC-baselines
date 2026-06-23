---
name: config-sweeper
description: Use when creating experiment grids, ablations, or batch scripts from existing configs.
---

Workflow:
1. Reuse existing config schema.
2. Generate compact sweep configs.
3. Do not duplicate full configs if inheritance/override exists.
4. Name runs with key variables only.
5. Add dry-run command.
6. Avoid launching expensive jobs unless explicitly asked.