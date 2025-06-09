---
description: "Self-correction mechanism for RAG responses."
globs: ["**/self_correction/**", "**/hybrid_rag/**"]
alwaysApply: true
---
# Self-Correction

- Detect low confidence or hallucination in output.
- Trigger fallback or re-retrieval paths if `correction_needed=True`.
- Limit by `iteration_count` to avoid loops.

@self_correction/checks.py
