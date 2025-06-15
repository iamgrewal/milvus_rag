# graphrag/config/settings.py

import os

class Config:
    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "graph_rag")
    EMBEDDING_DIM = 384

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # NLP and Model Settings
    NLP_MODEL = os.getenv("NLP_MODEL", "en_core_web_sm")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    QA_MODEL = os.getenv("QA_MODEL", "deepset/bert-base-cased-squad2")

    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
    
    # Memory System Configuration
    MEMORY_CACHE_TTL = int(os.getenv("MEMORY_CACHE_TTL", "3600"))  # 1 hour
    MEMORY_MAX_CACHE_SIZE = int(os.getenv("MEMORY_MAX_CACHE_SIZE", "10000"))
    MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.8"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
