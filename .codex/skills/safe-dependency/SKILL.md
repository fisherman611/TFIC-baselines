---
name: safe-dependency
description: Use when a task may require installing, adding, upgrading, or replacing dependencies.
---

Rules:
1. Prefer standard library or existing dependencies.
2. Before adding a dependency, check pyproject.toml, requirements.txt, package.json, or environment.yml.
3. Ask before adding production dependencies.
4. For research scripts, prefer optional dependency guards.
5. Document install command only if needed.