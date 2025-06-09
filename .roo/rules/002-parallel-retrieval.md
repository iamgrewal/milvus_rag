---
description: "Parallel async execution pattern for hybrid retrieval."
globs: ["**/retrieval/**", "**/hybrid_rag/**"]
alwaysApply: true
---
# Parallel Retrieval

- Always run vector and graph retrieval in parallel using asyncio.
- Await both tasks to optimize performance and reduce latency.

@retrieval/parallel.py

**Example:**
```python
vector_task = asyncio.create_task(vector_retrieve_async(query))
graph_task = asyncio.create_task(graph_traverse_async(query))
vector_results, graph_results = await asyncio.gather(vector_task, graph_task)
```
