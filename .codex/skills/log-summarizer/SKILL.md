---
name: log-summarizer
description: Use when analyzing long logs, training outputs, evaluation outputs, or stack traces.
---

Workflow:
1. Ignore progress bars unless they contain errors.
2. Extract:
   - first error
   - final error
   - stack trace root
   - failed command
   - relevant config values
3. Summarize in under 15 lines.
4. Suggest the smallest next debugging step.