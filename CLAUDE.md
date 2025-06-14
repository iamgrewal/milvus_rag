# CLAUDE.md - Milvus RAG Project Documentation

## Project Overview

**Project Name:** Milvus RAG Implementation  
**Type:** Hybrid Retrieval-Augmented Generation System  
**Architecture:** Milvus (Vector) + Neo4j (Graph) + LLM  
**Environment:** Debian WSL + Docker + Conda  
**Current Status:** Phase 1 - Foundation Optimization → Hybrid RAG Engine Evolution  

### Core Mission
Build a production-grade RAG system that combines vector similarity search with graph-based relationship retrieval for intelligent document querying, optimized for Onix's AI/Cloud solution offerings.

## Development Environment Setup

The project uses Conda for environment management with specific setup scripts:

### Environment Creation and Management
```bash
# Create environment (run once)
./create_conda.sh

# Run commands in the correct environment
./run_in_env.sh python your_script.py
./run_in_env.sh pytest
./run_in_env.sh python src/graphrag/rag_system/main.py
```

### Testing Commands
```bash
# Run all tests
./run_in_env.sh pytest

# Run specific test file
./run_in_env.sh pytest tests/test_hybrid_rag_system.py

# Run tests with coverage
./run_in_env.sh pytest --cov=src/graphrag

# Run async tests specifically
./run_in_env.sh pytest tests/test_hybrid_rag_system.py -v -s
```

### Code Quality and Linting
```bash
# Format code with black
./run_in_env.sh black src/ tests/

# Check types with mypy
./run_in_env.sh mypy src/graphrag

# Lint with ruff
./run_in_env.sh ruff check src/ tests/

# Fix linting issues automatically
./run_in_env.sh ruff check --fix src/ tests/
```

### Docker Development
```bash
# Start development environment
docker-compose up -d milvus-standalone neo4j

# Run full stack
docker-compose up

# Health check services
./health_check.sh
```

## Current Architecture Overview

### Core Components (Phase 1 - Current Implementation)

1. **Vector Store (Milvus)**: Stores entity embeddings for semantic similarity search
   - Located in: `src/graphrag/milvus/manager.py`
   - **Current:** IVF_FLAT index with Inner Product metric
   - **Target:** HNSW index with L2 metric for 2-3x performance improvement
   - Default embedding dimension: 384

2. **Graph Store (Neo4j)**: Manages entity relationships and knowledge graph
   - Located in: `src/graphrag/neo4j/manager.py`
   - Stores entities with relationships between them
   - **Optimization Target:** Batch operations to eliminate N+1 query problem

3. **NLP Processing**: Extracts entities and relationships from text
   - Located in: `src/graphrag/nlp/processor.py`
   - Uses spaCy for NER (Named Entity Recognition)
   - Extracts PERSON, ORG, GPE entities and co-occurrence relations

4. **Embedding Service**: Converts text to vector embeddings
   - Located in: `src/graphrag/embedding_service/service.py`
   - **Current:** Loads 300MB+ models on every initialization
   - **Target:** Singleton pattern with ModelCache for 67% memory reduction

### Package Structure
```
src/graphrag/
├── config/          # Configuration management
├── embedding_service/ # Text-to-vector conversion (OPTIMIZE: Model caching)
├── milvus/          # Vector database operations (OPTIMIZE: HNSW index)
├── neo4j/           # Graph database operations (OPTIMIZE: Batch queries)
├── nlp/             # Natural language processing
├── rag_system/      # Main RAG orchestration
└── logger.py        # Logging configuration
```

## Development Context

### Current Performance Profile
- **Query Latency:** 700ms (p95) → Target: 200ms
- **Throughput:** 10 QPS → Target: 50 QPS  
- **Memory Usage:** 2GB → Target: 800MB
- **Test Coverage:** <30% → Target: >80%

### Architecture Evolution Roadmap
**Current State:** Basic Milvus RAG (Phase 1)  
**Target State:** Hybrid RAG Engine with Neo4j + Advanced Capabilities  
**Evolution Path:** 4-phase incremental development with production validation at each stage

### Critical Issues Identified (Phase 1 Focus)
1. **Dependency Conflicts:** protobuf, httpx, grpcio-status version incompatibilities
2. **Synchronous Bottlenecks:** Sequential operations blocking pipeline
3. **Inefficient Indexing:** IVF_FLAT instead of HNSW
4. **Collection Recreation:** Data loss on restart
5. **N+1 Query Problem:** Individual Neo4j calls instead of batch
6. **Model Loading Overhead:** 300MB+ models loaded per request

## Claude Code Guidelines

### Phase-Based Development Strategy

**Phase 1 - Foundation Stabilization (Current Focus - Week 1-2):**
```bash
# Quick wins for immediate 2-3x performance improvement
claude code "Implement ModelCache singleton in src/utils/model_cache.py to eliminate 300MB model loading overhead"
claude code "Replace IVF_FLAT with HNSW index configuration in src/graphrag/milvus/manager.py for 2-3x query speed"
claude code "Remove collection recreation anti-pattern to preserve data and indexes"
claude code "Implement batch Neo4j operations to eliminate N+1 query problem"
```

**Phase 2 - Hybrid Foundation (Week 3-6):**
```bash
claude code "Implement Neo4j integration for dual-database hybrid search"
claude code "Add parallel vector + graph retrieval with asyncio.gather()"
claude code "Create basic result fusion engine for combining search results"
```

**Phase 3 - Intelligence Layer (Week 7-10):**
```bash
claude code "Implement ML-powered query classification for intelligent routing"
claude code "Add Weighted Reciprocal Rank Fusion (RRF) for result combination"
claude code "Create self-correction framework with hallucination detection"
```

**Phase 4 - Production Excellence (Week 11-16):**
```bash
claude code "Implement multi-tier fallback architecture with Tavily integration"
claude code "Add comprehensive observability with OpenTelemetry and Prometheus"
claude code "Create LangGraph orchestration for agentic workflows"
```

### Code Generation Preferences

**Architecture Patterns (All Phases):**
- **Async First:** All new code should use async/await
- **Batch Operations:** Prefer batch processing over individual calls
- **Singleton Models:** Use ModelCache for all ML model instances
- **Connection Pooling:** Database connections must use pooling
- **Error Handling:** Comprehensive try/catch with retry logic

**Performance Requirements:**
- **Sub-200ms Latency:** All query operations target <200ms
- **High Throughput:** Design for 50+ concurrent queries
- **Memory Efficiency:** Minimize model loading overhead
- **Caching Strategy:** Implement multi-level caching (query, embedding, entity)

**Code Style Standards:**
```python
# Preferred async pattern with observability
async def answer(self, question: str) -> RAGResponse:
    """Process query with parallel retrieval and fusion."""
    try:
        # Parallel operations for Phase 1 optimization
        vector_task = asyncio.create_task(self.vector_search(question))
        graph_task = asyncio.create_task(self.graph_traverse(question))
        
        vector_results, graph_results = await asyncio.gather(
            vector_task, graph_task
        )
        
        # Future: Weighted fusion (Phase 2+)
        fused_results = self.fusion_engine.combine(
            vector_results, graph_results, weights=(0.6, 0.4)
        )
        
        return await self.generate_response(question, fused_results)
        
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise RAGException(f"Query processing error: {e}")
```

### Database Optimization Patterns

**Milvus Best Practices:**
```python
# HNSW index configuration (Phase 1 priority)
index_params = {
    "index_type": "HNSW",
    "metric_type": "L2",
    "params": {"M": 16, "efConstruction": 500}
}

# Batch insert pattern
async def insert_batch(self, entities: List[Dict], batch_size: int = 1000):
    for i in range(0, len(entities), batch_size):
        batch = entities[i:i + batch_size]
        await self.collection.insert_async(batch)
    await self.collection.flush_async()
```

**Neo4j Best Practices:**
```python
# Batch query pattern (Phase 1 priority)
async def get_related_batch(self, entities: List[str]) -> Dict[str, List[str]]:
    async with self.driver.session() as session:
        result = await session.run("""
            UNWIND $entities AS entity_name
            MATCH (e:Entity {name: entity_name})--(n)
            RETURN entity_name, COLLECT(DISTINCT n.name) AS related
        """, entities=entities)
        return {r["entity_name"]: r["related"] async for r in result}
```

### Environment-Specific Considerations

**WSL Development:**
- Docker volume paths: `/mnt/d/projects/wslprojects/milvus_env`
- Use Docker Desktop integration
- Conda environment: `milvus_env` (Python 3.9)

**Docker Services:**
- Milvus: `localhost:19530`
- Neo4j: `localhost:7687` 
- MinIO: `localhost:9000`

**Dependency Management:**
- Pin compatible versions in requirements.txt
- Test all package imports before deployment
- Use conda for environment isolation

## Testing Strategy

### Current Test Architecture
- **Unit Tests:** Individual component testing
- **Integration Tests:** End-to-end RAG system testing (`test_hybrid_rag_system.py`)
- **Async Testing:** Uses pytest-asyncio for async components

### Key Test Files
- `tests/test_hybrid_rag_system.py`: Comprehensive hybrid RAG system tests
- `tests/test_nlp_processor.py`: NLP component tests
- `tests/test_rag_system.py`: Core RAG system tests

### Required Test Coverage by Phase
- **Phase 1:** >80% coverage with performance benchmarks
- **Phase 2:** Integration tests for hybrid operations
- **Phase 3:** ML model validation and confidence tracking
- **Phase 4:** Production load tests and health checks

### Performance Benchmark Tests
```python
# Performance validation (Phase 1 requirement)
@pytest.mark.asyncio
async def test_query_latency():
    """Validate <200ms query latency requirement."""
    start_time = time.time()
    response = await rag_system.answer("test question")
    latency = time.time() - start_time
    
    assert latency < 0.2, f"Query too slow: {latency}s"
    assert response.confidence > 0.7
```

## Migration Path: Basic → Advanced Hybrid RAG Engine

### Phase 1: Foundation Stabilization (Current - Week 1-2)
- Fix dependency conflicts and optimize current Milvus implementation
- Implement model caching and HNSW indexing
- Add comprehensive testing framework
- **Success Criteria:** 2-3x performance improvement, <30min implementation

### Phase 2: Dual-Database Integration (Week 3-6)
- Neo4j graph database integration
- Entity extraction and relationship mapping
- Basic hybrid retrieval (vector + graph)
- **Success Criteria:** Parallel search operations, improved context quality

### Phase 3: Intelligence Layer (Week 7-10)
- Advanced query routing and classification
- Weighted Reciprocal Rank Fusion (RRF)
- Self-correction and validation framework
- **Success Criteria:** <2s response time, >85% accuracy

### Phase 4: Enterprise Production (Week 11-16)
- Multi-tier fallback architecture with Tavily integration
- Circuit breakers and comprehensive observability
- Advanced agentic capabilities and monitoring
- **Success Criteria:** 99.9% uptime, enterprise-grade reliability

### Migration Strategy
1. **Risk-Minimized Evolution:** Each phase builds on proven foundation
2. **Production Validation:** Deploy and validate each phase before advancing
3. **Parallel Development:** Maintain basic system while building advanced features
4. **Gradual Cutover:** Phase-by-phase production migration with rollback capability

## Future Architecture Vision

### Hybrid RAG Engine (Target Architecture)
The ultimate goal is a sophisticated hybrid system combining:
- **Dual-Database Strategy:** Milvus (semantic) + Neo4j (relationships)
- **Intelligent Fallback:** Multi-tier strategy with Tavily web search integration
- **Self-Correction Engine:** Advanced validation and hallucination detection
- **Agentic Capabilities:** LangGraph orchestration with human-in-the-loop
- **Enterprise Observability:** Comprehensive monitoring with OpenTelemetry

**Key Success Metrics (Target):**
- Query Response Time: <2s (P95)
- Retrieval Accuracy: >85% precision, >90% recall
- System Availability: 99.9% uptime
- Hallucination Reduction: 60% improvement over basic RAG

*Note: This vision will be implemented incrementally through Phases 2-4*

## Monitoring & Production Requirements

### Key Metrics (Phase 1 Focus)
- **Query Latency:** p50, p95, p99 response times
- **Throughput:** Queries per second (QPS)
- **Cache Hit Rate:** Model and query cache efficiency  
- **Error Rate:** Failed queries per total queries
- **Resource Usage:** Memory, CPU, disk utilization

### Health Checks
```python
# Service health monitoring (current implementation)
async def health_check() -> HealthStatus:
    """Comprehensive system health validation."""
    checks = await asyncio.gather(
        check_milvus_connection(),
        check_neo4j_connection(),
        check_model_loading(),
        check_cache_performance()
    )
    return HealthStatus(all(checks))

# Health check script usage
./health_check.sh
```

### Service Health Monitoring
- Health check script: `./health_check.sh`
- Checks Neo4j HTTP endpoint, Milvus port connectivity, and app initialization
- Required for production deployment validation

## Claude Code Integration Notes

### Development Workflow by Phase
**Phase 1 (Current):** `claude code assess project-health` → Fix dependencies → Optimize performance  
**Phase 2:** `claude code "Implement Neo4j integration following hybrid RAG architecture patterns"`  
**Phase 3:** `claude code "Add intelligent query routing and validation framework"`  
**Phase 4:** `claude code "Implement production-grade fallback mechanisms and observability"`

### Phase-Specific Focus Areas
- **Phase 1:** Performance optimization, dependency management, testing foundation
- **Phase 2:** Graph database integration, entity extraction, hybrid search patterns  
- **Phase 3:** ML-powered classification, advanced fusion algorithms, validation systems
- **Phase 4:** Production infrastructure, monitoring, enterprise deployment patterns

### Architecture-Aware Development
Claude Code will understand the target hybrid architecture while maintaining focus on current phase requirements. Each generated solution should be compatible with the eventual dual-database strategy without over-engineering current needs.

## Business Context

### Onix Strategic Alignment
- **AI Solutions:** Showcase advanced RAG capabilities to clients
- **Cloud Expertise:** Demonstrate scalable AI infrastructure
- **Supply Chain:** Knowledge graph applications for logistics
- **Healthcare:** Intelligent document processing for medical records

### Success Metrics by Phase
- **Phase 1:** 2-3x performance improvement, foundation stability
- **Phase 2:** Enhanced context quality, parallel processing
- **Phase 3:** <2s response time, >85% accuracy
- **Phase 4:** 99.9% uptime, enterprise deployment ready

**Strategic Value:** Position Onix as the AI/Cloud partner that delivers enterprise-grade solutions with cutting-edge development practices.

---

**Last Updated:** June 2025  
**Project Lead:** Jay Grewal, Rhobyte 
**Technical Focus:** Production-grade RAG system for AI/Cloud solutions  
**Current Phase:** Phase 1 - Foundation Optimization
