---
description: "LangGraph orchestration rules for hybrid RAG."
globs: ["**/langgraph/**"]
alwaysApply: true
---
# LangGraph Orchestration

- Use `StateGraph` with typed states for managing routing, retrieval, fusion, correction, and fallback.
- Implement `conditional_edges` to branch logic across vector-only, graph-only, and hybrid paths.
- Include a `correction_needed` check and `iteration_count` to avoid infinite loops.

@langgraph/orchestration.py

**Example:**
```python
graph.add_conditional_edges(
    "query_router",
    route_decision,
    {
        "vector_only": "vector_retrieval",
        "graph_only": "graph_retrieval",
        "hybrid": ["vector_retrieval", "graph_retrieval"]
    }
)
```
