# CLAUDE.md - Milvus RAG Project Documentation

## Project Overview

**Project Name:** Milvus RAG Implementation  
**Type:** Advanced Hybrid Retrieval-Augmented Generation System  
**Architecture:** Milvus (Vector) + Neo4j (Graph) + LLM + Self-Correction
**Environment:** Debian 12 + Docker + Conda  
**Current Status:** Phase 2 - Hybrid RAG Engine with Advanced Intelligence

### Core Mission

Build a production-grade hybrid RAG system that combines vector similarity search with graph-based relationship retrieval, advanced context enhancement, and self-correction capabilities, optimized for Rhobyte's AI/Cloud solution offerings.

## Development Environment Setup

The project uses Conda for environment management with specific setup scripts:

```
conda activate milvus_env
```

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

# Run hybrid system tests
./run_in_env.sh pytest tests/test_hybrid_rag_system.py

# Run context enhancement tests
./run_in_env.sh pytest tests/test_context_enhancement.py

# Run self-correction tests
./run_in_env.sh pytest tests/test_self_correction.py

# Run tests with coverage
./run_in_env.sh pytest --cov=src/graphrag

# Run async tests specifically
./run_in_env.sh pytest tests/test_hybrid_rag_system.py -v -s --asyncio-mode=no
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

## Phase 2 Architecture Overview

### Core Components (Phase 2 - Current Implementation)

1. **Vector Store (Milvus)**: Optimized semantic similarity search

    - Located in: `src/graphrag/milvus/manager.py`
    - **Phase 1 Achievement:** HNSW index with L2 metric for 2-3x performance improvement
    - **Phase 2 Enhancement:** Dynamic collection management and batch operations
    - Default embedding dimension: 384
2. **Graph Store (Neo4j)**: Advanced relationship and knowledge graph management

    - Located in: `src/graphrag/neo4j/manager.py`
    - **Phase 1 Achievement:** Batch operations eliminating N+1 query problem
    - **Phase 2 Enhancement:** Dynamic relationship discovery and graph expansion
    - Advanced Cypher query optimization for multi-hop traversals
3. **Context Enhancement Engine**: Advanced relationship mapping and discovery

    - Located in: `src/graphrag/graph_enrichment/context_engine.py`
    - **NEW in Phase 2:** Multi-strategy relationship discovery
    - Semantic similarity, co-occurrence, and temporal relationship analysis
    - Dynamic context graph construction with NetworkX integration
4. **Self-Correction Framework**: Validation and hallucination detection

    - Located in: `src/graphrag/validation/self_correction.py`
    - **NEW in Phase 2:** Multi-layer validation system
    - Confidence scoring and automated correction mechanisms
    - LLM-based consistency checking
5. **Contextual Memory System**: Advanced caching with similarity search

    - Located in: `src/graphrag/graph_enrichment/memory_system.py`
    - **NEW in Phase 2:** Redis-based similarity caching
    - Embedding-based context retrieval and cache optimization
6. **Hybrid RAG Orchestrator**: Intelligent query routing and result fusion

    - Located in: `src/graphrag/rag_system/hybrid_orchestrator.py`
    - **NEW in Phase 2:** Parallel vector + graph retrieval
    - Weighted Reciprocal Rank Fusion (RRF) for result combination

### Enhanced Package Structure

```
src/graphrag/
├── config/                    # Configuration management
├── embedding_service/         # Text-to-vector conversion (ModelCache optimized)
├── milvus/                   # Vector database operations (HNSW optimized)
├── neo4j/                    # Graph database operations (Batch optimized)
├── nlp/                      # Natural language processing
├── rag_system/               # Main RAG orchestration
│   ├── hybrid_orchestrator.py  # NEW: Hybrid search coordination
│   └── main.py               # Enhanced main system
├── graph_enrichment/         # NEW: Advanced context enhancement
│   ├── context_engine.py     # Relationship mapping and discovery
│   └── memory_system.py      # Contextual memory with similarity search
├── validation/               # NEW: Self-correction framework
│   ├── self_correction.py    # Validation and correction logic
│   └── confidence_scoring.py # Confidence assessment
├── utils/                    # NEW: Shared utilities
│   ├── model_cache.py        # Phase 1: ModelCache singleton
│   └── observability.py     # Phase 2: Monitoring and metrics
└── logger.py                 # Logging configuration
```

## Phase 2 Performance Profile

### Current Achievement (Phase 1 → Phase 2)

- **Query Latency:** 700ms → 200ms (p95) → **Target: <200ms with hybrid search**
- **Throughput:** 10 QPS → 50 QPS → **Target: 100+ QPS with parallel processing**
- **Memory Usage:** 2GB → 800MB → **Target: <1GB with advanced caching**
- **Test Coverage:** <30% → 80% → **Target: >90% with Phase 2 components**
- **Context Quality:** Basic → **Enhanced with relationship discovery**
- **Accuracy:** Standard RAG → **Self-corrected with validation framework**

### New Phase 2 Capabilities

1. **Hybrid Retrieval**: Parallel vector + graph search with intelligent fusion
2. **Context Enhancement**: Multi-strategy relationship discovery and mapping
3. **Self-Correction**: Automated validation and hallucination detection
4. **Memory System**: Similarity-based context caching and retrieval
5. **Advanced Observability**: Comprehensive monitoring with OpenTelemetry

## Claude Code Implementation Guide

### Phase 2 Implementation Strategy

**Approach:** Incremental enhancement of Phase 1 foundation  
**Duration:** 4-6 weeks  
**Risk:** Low (builds on stable Phase 1 foundation)

### Step-by-Step Claude Code Instructions

#### Week 1: Neo4j Integration and Context Enhancement

```bash
# Step 1: Implement Advanced Context Enhancement Engine
claude code "Create src/graphrag/graph_enrichment/context_engine.py implementing the ContextEnhancementEngine class with multi-strategy relationship discovery (Neo4j, semantic similarity, co-occurrence, temporal analysis) as shown in the provided specifications"

# Step 2: Add Contextual Memory System
claude code "Implement src/graphrag/graph_enrichment/memory_system.py with Redis-based similarity caching, embedding-based retrieval, and automatic cache management as specified in the memory system requirements"

# Step 3: Enhance Neo4j Manager for Advanced Operations
claude code "Update src/graphrag/neo4j/manager.py to support dynamic relationship discovery, multi-hop traversals, and integration with the ContextEnhancementEngine"

# Step 4: Create Configuration Support
claude code "Update src/graphrag/config/ to support new Phase 2 configuration parameters for graph expansion, similarity thresholds, caching settings, and validation parameters"
```

#### Week 2: Self-Correction Framework

```bash
# Step 5: Implement Self-Correction Framework
claude code "Create src/graphrag/validation/self_correction.py implementing multi-layer validation system with confidence scoring, consistency checking, and automated correction mechanisms"

# Step 6: Add Confidence Scoring System
claude code "Implement src/graphrag/validation/confidence_scoring.py with LLM-based confidence assessment, uncertainty quantification, and validation rule engine"

# Step 7: Create Validation Integration
claude code "Update the main RAG system to integrate self-correction validation at appropriate checkpoints in the query processing pipeline"
```

#### Week 3: Hybrid Orchestration

```bash
# Step 8: Create Hybrid RAG Orchestrator
claude code "Implement src/graphrag/rag_system/hybrid_orchestrator.py with parallel vector + graph retrieval, intelligent query routing, and Weighted Reciprocal Rank Fusion (RRF) for result combination"

# Step 9: Enhance Main RAG System
claude code "Update src/graphrag/rag_system/main.py to use the hybrid orchestrator, integrate context enhancement, and implement the full Phase 2 pipeline"

# Step 10: Add Observability Framework
claude code "Create src/graphrag/utils/observability.py implementing comprehensive monitoring with OpenTelemetry, Prometheus metrics, and performance tracking for Phase 2 components"
```

#### Week 4: Integration and Testing

```bash
# Step 11: Create Comprehensive Test Suite
claude code "Implement tests/test_context_enhancement.py with comprehensive tests for the context enhancement engine, relationship discovery, and graph expansion"

# Step 12: Add Self-Correction Tests
claude code "Create tests/test_self_correction.py testing the validation framework, confidence scoring, and correction mechanisms"

# Step 13: Implement Integration Tests
claude code "Update tests/test_hybrid_rag_system.py to include end-to-end testing of the hybrid system with parallel retrieval, context enhancement, and self-correction"

# Step 14: Performance Validation
claude code "Create performance tests validating <200ms query latency, >90% accuracy improvement, and successful parallel processing under load"
```

### Architecture-Aware Development Patterns

**Async-First Pattern (Phase 2 Enhancement):**

```python
# Hybrid retrieval with parallel operations
async def hybrid_search(self, question: str) -> HybridRAGResponse:
    """Enhanced hybrid search with context enhancement and validation."""
    try:
        # Phase 2: Parallel context enhancement and retrieval
        context_task = asyncio.create_task(
            self.context_engine.enhance_query_context(question, [])
        )
        vector_task = asyncio.create_task(self.vector_search(question))
        graph_task = asyncio.create_task(self.graph_traverse(question))
        
        enhanced_context, vector_results, graph_results = await asyncio.gather(
            context_task, vector_task, graph_task
        )
        
        # Phase 2: Intelligent result fusion with RRF
        fused_results = await self.fusion_engine.weighted_fusion(
            vector_results, graph_results, enhanced_context
        )
        
        # Phase 2: Self-correction and validation
        response = await self.generate_response(question, fused_results)
        validated_response = await self.validation_engine.validate_and_correct(
            question, response, enhanced_context
        )
        
        return validated_response
        
    except Exception as e:
        logger.error(f"Hybrid RAG query failed: {e}")
        raise HybridRAGException(f"Query processing error: {e}")
```

**Context Enhancement Pattern:**

```python
# Advanced relationship discovery and mapping
async def discover_relationships(self, entities: List[Dict]) -> List[EntityRelationship]:
    """Multi-strategy relationship discovery."""
    
    relationship_tasks = [
        self._find_neo4j_relationships(entities),
        self._find_semantic_relationships(entities),
        self._find_cooccurrence_relationships(entities),
        self._find_temporal_relationships(entities)
    ]
    
    all_relationships = await asyncio.gather(*relationship_tasks)
    return self._deduplicate_and_rank_relationships(all_relationships)
```

## Phase 2 Success Criteria

### Technical Metrics

- **Parallel Operations**: Successful concurrent vector + graph retrieval
- **Context Quality**: >40% improvement in relationship discovery
- **Response Accuracy**: >85% accuracy with self-correction
- **Query Latency**: Maintain <200ms despite added complexity
- **System Reliability**: >99% uptime with enhanced components

### Business Value Metrics

- **Enhanced Context**: Rich relationship mapping for complex queries
- **Reduced Hallucinations**: 60% reduction through self-correction
- **Improved User Experience**: More accurate and contextual responses
- **Scalability**: Support for 100+ concurrent users

## Migration Path: Phase 1 → Phase 2

### Preservation of Phase 1 Achievements

✅ **ModelCache Singleton**: Maintained and enhanced  
✅ **HNSW Indexing**: Preserved and integrated with hybrid search  
✅ **Batch Neo4j Operations**: Enhanced with advanced traversals  
✅ **Performance Optimizations**: Built upon with parallel processing  
✅ **Test Coverage**: Expanded to include Phase 2 components

### Phase 2 Incremental Enhancements

🔄 **Context Enhancement**: Added without disrupting existing vector search  
🔄 **Self-Correction**: Integrated as optional validation layer  
🔄 **Memory System**: Added as performance optimization  
🔄 **Hybrid Orchestration**: Coordinates existing components

### Rollback Strategy

- Phase 1 components remain functional independently
- Phase 2 features can be disabled via configuration
- Database schemas are backward compatible
- Performance monitoring ensures no regression

## Monitoring & Production Requirements

### Phase 2 Enhanced Metrics

- **Hybrid Search Performance**: Vector vs Graph vs Hybrid latency comparison
- **Context Enhancement Quality**: Relationship discovery success rate
- **Self-Correction Effectiveness**: Validation accuracy and correction rate
- **Memory System Efficiency**: Cache hit rates and similarity search performance
- **Overall System Health**: End-to-end accuracy and user satisfaction

### Health Checks

```python
# Enhanced health monitoring for Phase 2
async def phase2_health_check() -> HealthStatus:
    """Comprehensive Phase 2 system health validation."""
    checks = await asyncio.gather(
        check_milvus_connection(),
        check_neo4j_connection(),
        check_redis_connection(),  # NEW: Redis for memory system
        check_context_enhancement_engine(),  # NEW: Context enhancement
        check_self_correction_framework(),  # NEW: Validation system
        check_hybrid_orchestrator(),  # NEW: Hybrid coordination
        check_model_cache_performance()
    )
    return HealthStatus(all(checks))
```

## Business Context

### Onix Strategic Alignment (Phase 2)

- **AI Solutions**: Showcase cutting-edge hybrid RAG with self-correction
- **Cloud Expertise**: Demonstrate advanced multi-service orchestration
- **Supply Chain**: Enhanced relationship mapping for complex logistics
- **Healthcare**: Improved accuracy through validation for medical queries

### Phase 2 Value Proposition

- **Enhanced Accuracy**: Self-correction reduces hallucinations by 60%
- **Richer Context**: Advanced relationship discovery improves response quality
- **Scalable Architecture**: Parallel processing supports enterprise workloads
- **Production Ready**: Comprehensive monitoring and validation frameworks

---

**Last Updated:** June 2025  
**Project Lead:** Jay Grewal, VP Sales - Rhobyte
**Technical Focus:** Production-grade Hybrid RAG with Advanced Intelligence  
**Current Phase:** Phase 2 - Hybrid RAG Engine with Self-Correction

## CPhase 1 Architecture Overview

### Core Components (Phase 1 - Existing Implementation)

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

### Phase 1 Package Structure

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

## Phase 1 Development Context

### Phase 1 Current Performance Profile

- **Query Latency:** 700ms (p95) → Target: 200ms
- **Throughput:** 10 QPS → Target: 50 QPS  
- **Memory Usage:** 2GB → Target: 800MB
- **Test Coverage:** <30% → Target: >80%

### Phase 1 Architecture Evolution Roadmap

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

**Linux  Development:**

- Docker volume paths: `/home/jgrewal/projects/aiprojects/milvus_rag`
- Use Docker Desktop integration
- Conda environment: `milvus_env` (Python Python 3.10.18)
(milvus_env) jgrewal@devbox:~/projects/aiprojects/milvus_rag$ pip --version
pip 25.1.1 from /home/jgrewal/anaconda3/envs/milvus_env/lib/python3.10/site-packages/pip (python 3.10)
uv 0.7.13
/home/jgrewal/.local/bin/uv
**Docker Services:**
docker version 28.2.2
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

### Rhobyte Strategic Alignment

- **AI Solutions:** Showcase advanced RAG capabilities to clients
- **Cloud Expertise:** Demonstrate scalable AI infrastructure
- **Supply Chain:** Knowledge graph applications for logistics
- **Healthcare:** Intelligent document processing for medical records

### Success Metrics by Phase

- **Phase 1:** 2-3x performance improvement, foundation stability
- **Phase 2:** Enhanced context quality, parallel processing
- **Phase 3:** <2s response time, >85% accuracy
- **Phase 4:** 99.9% uptime, enterprise deployment ready

**Strategic Value:** Position Rhobyte as the AI/Cloud partner that delivers enterprise-grade solutions with cutting-edge development practices.

---

**Last Updated:** June 2025  
**Project Lead:** Jay Grewal, Rhobyte
**Technical Focus:** Production-grade RAG system for AI/Cloud solutions  
**Current Phase:** Phase 1 - Foundation Optimization
