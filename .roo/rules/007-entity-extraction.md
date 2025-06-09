---
description: "NER and relationship extraction pipeline."
globs: ["**/entity_extraction/**"]
alwaysApply: true
---
# Entity and Relationship Extraction

- Use transformer-backed NER models (e.g. `en_core_web_trf`).
- Parallelize entity and relation extraction with asyncio.
- Return metadata (type, span, confidence) for each entity.

@entity_extraction/ner_pipeline.py
