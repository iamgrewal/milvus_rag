---
description: Documentation for the structure.sh file that sets up the project structure and boilerplate for the GraphRAG application.
globs: ['structure.sh']
alwaysApply: false
---

# structure.sh Documentation

## Overview
The `structure.sh` file is a Bash script designed to create the initial directory structure and boilerplate code for the GraphRAG application. It sets up essential directories, configuration files, and initial Python scripts that are crucial for the application's functionality.

## Key Components

### Directory Structure Creation
- **`mkdir -p graphrag/{...}`**: This command creates the main project directory `graphrag` along with subdirectories for configuration, logs, models, NLP processing, Neo4j integration, Milvus integration, embedding services, and tests.

### Configuration File
- **`touch graphrag/config/config.py`**: This command creates a configuration file where application settings can be defined.

### Logger Setup
- **`graphrag/logger.py`**: This script sets up a logging mechanism for the application, allowing for easy tracking of events and errors. It uses Python's built-in logging library to log messages with a specified format and level.

### Embedding Service
- **`graphrag/embedding_service/service.py`**: This class, `EmbeddingService`, is responsible for loading a pre-trained sentence transformer model and providing a method to embed text into vectors. It logs the loading of the model for debugging purposes.

### Milvus Manager
- **`graphrag/milvus/manager.py`**: The `MilvusManager` class connects to a Milvus database, manages collections, and provides methods for inserting and searching embeddings. It ensures that the collection is created with the appropriate schema and index.

### Neo4j Manager
- **`graphrag/neo4j/manager.py`**: This class manages connections to a Neo4j database, allowing for the creation of entities and relationships. It provides methods to create entities and retrieve related entities from the graph database.

### NLP Processor
- **`graphrag/nlp/processor.py`**: The `NLPProcessor` class handles text preprocessing and entity extraction using the SpaCy library. It also identifies relationships between entities in the text.

### RAG System
- **`graphrag/rag_system/main.py`**: This is the main entry point for the application, integrating all components. It provides methods for ingesting text, extracting entities and relationships, and answering questions based on the ingested data.

## Dependencies
This file does not import any other files in the repository, nor is it imported by any other files. It serves as a standalone script to set up the project structure.

## Usage Example
To use this script, simply run it in a terminal:
```bash
bash structure.sh
```
This will create the necessary directories and files for the GraphRAG application.

## Best Practices
- Ensure that the necessary dependencies (like `sentence_transformers`, `pymilvus`, `neo4j`, and `spacy`) are installed in your Python environment before running the application.
- Regularly update the configuration file with any new settings or parameters required by the application.
- Use the logging functionality to monitor the application's behavior and troubleshoot issues effectively.