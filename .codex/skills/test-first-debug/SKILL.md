---
name: test-first-debug
description: Use when debugging errors, failed tests, stack traces, or broken scripts. Reproduce first, patch minimally, then verify.
---

Workflow:
1. Read the error message and stack trace carefully.
2. Locate the exact failing file/function.
3. Reproduce with the smallest command possible.
4. Patch only the root cause.
5. Run the narrowest relevant test first.
6. Only run full test/lint if the narrow test passes.
7. Report:
   - root cause
   - patch
   - verification command