---
name: minimal-patch
description: Use when making code changes. Keep diffs small, preserve existing APIs, and avoid broad refactors.
---

Rules:
1. Make the smallest correct change.
2. Do not rename functions/classes unless required.
3. Do not rewrite unrelated code.
4. Preserve current config formats and CLI arguments.
5. If a large refactor seems useful, propose it separately instead of doing it.
6. After editing, show only:
   - changed files
   - reason for each change
   - how to verify