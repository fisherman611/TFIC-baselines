# AGENTS.md

## Token-saving rules

- Prefer the smallest relevant context.
- Use repo-navigator before reading many files.
- Use minimal-patch for all code edits.
- Use test-first-debug for errors, failed tests, and stack traces.
- Use experiment-runner for ML/research experiment changes.
- Do not read data/, checkpoints/, outputs/, wandb/, logs/ unless needed.
- Do not refactor unrelated code.
- Report changed files and verification commands only.