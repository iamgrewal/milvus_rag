---
description: "Fusion strategies for combining vector and graph results."
globs: ["**/fusion/**", "**/hybrid_rag/**"]
alwaysApply: true
---
# Result Fusion Logic

- Use weighted reciprocal rank or hybrid scoring heuristics.
- Handle partial or missing results gracefully.
- Normalize scores before fusion.

@fusion/merging.py
