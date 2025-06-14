# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Architecture Overview

This is a hybrid RAG (Retrieval-Augmented Generation) system that combines vector search with graph database relationships:

### Core Components

1. **Vector Store (Milvus)**: Stores entity embeddings for semantic similarity search
   - Located in: `src/graphrag/milvus/manager.py`
   - Uses IVF_FLAT index with Inner Product metric
   - Default embedding dimension: 384

2. **Graph Store (Neo4j)**: Manages entity relationships and knowledge graph
   - Located in: `src/graphrag/neo4j/manager.py`
   - Stores entities with relationships between them
   - Supports graph traversal for related entity discovery

3. **NLP Processing**: Extracts entities and relationships from text
   - Located in: `src/graphrag/nlp/processor.py`
   - Uses spaCy for NER (Named Entity Recognition)
   - Extracts PERSON, ORG, GPE entities and co-occurrence relations

4. **Embedding Service**: Converts text to vector embeddings
   - Located in: `src/graphrag/embedding_service/service.py`
   - Uses sentence-transformers (default: all-MiniLM-L6-v2)

### Data Flow

1. **Ingestion**: Text → NLP Processing → Entity Extraction → Vector Embeddings → Storage (Milvus + Neo4j)
2. **Retrieval**: Query → Vector Search (Milvus) → Related Entities (Neo4j) → Context Assembly → Answer Generation

### Configuration

Configuration is managed through environment variables and the `Config` class:
- Located in: `src/graphrag/config/settings.py`
- Key settings: MILVUS_HOST, NEO4J_URI, embedding models, collection names
- Uses `.env` file for environment-specific settings

## Testing Architecture

The project includes comprehensive testing with focus on:

### Test Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end RAG system testing (`test_hybrid_rag_system.py`)
- **Async Testing**: Uses pytest-asyncio for async components

### Key Test Files
- `tests/test_hybrid_rag_system.py`: Comprehensive hybrid RAG system tests
- `tests/test_nlp_processor.py`: NLP component tests
- `tests/test_rag_system.py`: Core RAG system tests

### Test Categories
1. **Orchestrator Tests**: LangGraph-based workflow testing
2. **Fusion Logic Tests**: Result merging and deduplication
3. **Confidence Tracking**: Score validation throughout pipeline
4. **Self-Correction**: Iterative improvement logic
5. **Production Readiness**: Error handling, timeouts, health checks

## Package Structure

```
src/graphrag/
├── config/          # Configuration management
├── embedding_service/ # Text-to-vector conversion
├── milvus/          # Vector database operations
├── neo4j/           # Graph database operations
├── nlp/             # Natural language processing
├── rag_system/      # Main RAG orchestration
└── logger.py        # Logging configuration
```

## Development Guidelines

### Running the Main Application
```bash
# Start the RAG system
./run_in_env.sh python src/graphrag/rag_system/main.py
```

### Working with Dependencies
- Use `uv` for fast package installation (installed via create_conda.sh)
- Dependencies are split between `requirements.txt` (production) and `dev-requirements.txt` (development)
- The project uses pydantic for configuration validation

### Code Navigation
- ctags are automatically generated in `.tags` file
- Use IDE support for Python with the conda environment path
- Main entry point: `src/graphrag/rag_system/main.py`

### Service Health Monitoring
- Health check script: `./health_check.sh`
- Checks Neo4j HTTP endpoint, Milvus port connectivity, and app initialization
- Required for production deployment validation

## Key Dependencies

- **Vector Database**: pymilvus (v2.5.10)
- **Graph Database**: neo4j, neo4j-graphrag
- **ML/NLP**: sentence-transformers, spacy, transformers
- **Orchestration**: langchain (for advanced RAG patterns)
- **Development**: pytest, ruff, black, mypy, pre-commit