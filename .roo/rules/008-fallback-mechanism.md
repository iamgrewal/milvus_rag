---
description: "Fallback strategies for missing or low-confidence results."
globs: ["**/fallback/**", "**/hybrid_rag/**"]
alwaysApply: true
---
# Fallback Mechanism

- Trigger fallback when:
  - `confidence_score < 0.4`
  - No results returned
- Fallback to external search_files sources like Wikipedia, Bing.

@fallback/search_files.py
