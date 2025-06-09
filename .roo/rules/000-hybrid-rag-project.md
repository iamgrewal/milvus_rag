---
description: Hybrid RAG implementation guidelines combining Neo4j knowledge graphs with Milvus vector search_files
globs: ["**/rag/**/*.py", "**/graph/**/*.cypher", "**/configs/rag*.yaml", "**/fusion/**/*.py", "**/self_correction/**/*.py"]
alwaysApply: false
---

# Hybrid RAG System Implementation Rules

@context {
  "architecture": "hybrid-rag",
  "components": ["neo4j", "milvus", "langgraph", "llm"],
  "phase": "production"
}

## LangGraph Orchestration Requirements

**State Management**
- Implement parallel processing for vector/graph retrieval using async/await patterns
- Maintain confidence scores in state transitions
- Enforce maximum 3 self-correction iterations

```python
class RAGState(TypedDict):
    query: str
    vector_results: List[dict]
    graph_results: List[dict]
    fused_results: List[dict]
    response: str
    confidence_score: float
```

```python
async def parallel_retrieval(state: RAGState):
    vector_task = asyncio.create_task(vector_retrieve_async(state["query"]))
    graph_task = asyncio.create_task(graph_traverse_async(state["query"]))
    state["vector_results"], state["graph_results"] = await asyncio.gather(vector_task, graph_task)
    return state
```

@langgraph/orchestration.py

## Neo4j Integration Standards

**Schema Design**
- Use labeled property graph model:
  - `Entity` nodes with `type` and `properties`
  - `RELATIONSHIP` edges with `strength` and `source`
  - `Document` nodes linked to entities

**Cypher Query Patterns**
```cypher
MATCH (e1:Entity)-[r:RELATES]->(e2:Entity)
WHERE r.strength > 0.7
RETURN e1, r, e2
```

@neo4j/graph.cypher

## Milvus Configuration Rules

**Collection Setup**
- Use `HNSW` index
- Partition sizing:
  - <1M: 2 partitions
  - 1–10M: 8 partitions
  - >10M: 16 partitions

**Hybrid Search Parameters**
```python
search_params = {
  "metric_type": "L2",
  "offset": 5,
  "ignore_growing": False,
  "params": {"ef": 32}
}
```

@milvus/retriever.py

## Fusion & Self-Correction

**Merging Results**
1. Apply weighted reciprocal rank
2. Deduplicate with content hashing
3. Filter by confidence threshold ≥ 0.65

**Hallucination Detection**
```python
def detect_hallucination(response: str, sources: List[str]) -> bool:
    source_entities = extract_entities(sources)
    response_entities = extract_entities(response)
    novel_entities = response_entities - source_entities
    return len(novel_entities) > 0
```

@fusion/fuser.py

## Production Readiness

**Security**
- Use TLS 1.3 for all services
- RBAC roles:
  - `rag_reader`
  - `rag_operator`
  - `rag_admin`

**Monitoring KPIs**
- `rag.latency.95percentile < 2s`
- `rag.accuracy.f1_score > 0.85`
- `rag.uptime >= 99.95%`

@deploy/security.yaml

## Implementation Validation

**Unit Test**
```python
def test_hybrid_retrieval():
    query = "Explain neural architecture search_files"
    vector_results = vector_retrieve(query)
    graph_results = graph_retrieve(query)
    assert len(vector_results) >= 3
    assert len(graph_results) >= 2
    assert validate_relationships(graph_results)
```

**Integration Tests**
- Validate cross-system consistency
- Test fallback (e.g., Tavily or Bing)
- Verify automatic Neo4j schema migration

@tests/test_hybrid.py
