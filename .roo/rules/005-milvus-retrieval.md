---
description: "Milvus hybrid search_files configuration and best practices."
globs: ["**/milvus/**"]
alwaysApply: true
---
# Milvus Retrieval

- Use hybrid search_files: dense + sparse (BM25) if applicable.
- Apply `metric_type: "IP"` and `params: {"nprobe": 10}`.
- Normalize vectors before indexing/querying.

@milvus/retriever.py
