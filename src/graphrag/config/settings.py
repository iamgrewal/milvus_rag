# graphrag/config/settings.py

import os


class Config:
    # ===============================================
    # Phase 1 - Core Database Configuration
    # ===============================================

    # Milvus Configuration
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "graph_rag")
    EMBEDDING_DIM = 384

    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_MAX_CONNECTION_LIFETIME = int(
        os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "3600")
    )
    NEO4J_MAX_CONNECTION_POOL_SIZE = int(
        os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")
    )

    # Redis Configuration
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

    # ===============================================
    # Phase 1 - Model and NLP Configuration
    # ===============================================

    # NLP and Model Settings
    NLP_MODEL = os.getenv("NLP_MODEL", "en_core_web_sm")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    QA_MODEL = os.getenv("QA_MODEL", "deepset/bert-base-cased-squad2")

    # ===============================================
    # Phase 2 - Context Enhancement Configuration
    # ===============================================

    # Context Enhancement Engine Settings
    CONTEXT_ENHANCEMENT_ENABLED = (
        os.getenv("CONTEXT_ENHANCEMENT_ENABLED", "true").lower() == "true"
    )
    CONTEXT_CACHE_TTL = int(os.getenv("CONTEXT_CACHE_TTL", "300"))  # 5 minutes
    CONTEXT_MAX_RELATIONSHIPS = int(os.getenv("CONTEXT_MAX_RELATIONSHIPS", "100"))
    CONTEXT_EXPANSION_DEPTH = int(os.getenv("CONTEXT_EXPANSION_DEPTH", "2"))

    # Multi-Strategy Relationship Discovery
    NEO4J_RELATIONSHIP_DISCOVERY_ENABLED = (
        os.getenv("NEO4J_RELATIONSHIP_DISCOVERY_ENABLED", "true").lower() == "true"
    )
    SEMANTIC_RELATIONSHIP_DISCOVERY_ENABLED = (
        os.getenv("SEMANTIC_RELATIONSHIP_DISCOVERY_ENABLED", "true").lower() == "true"
    )
    COOCCURRENCE_RELATIONSHIP_DISCOVERY_ENABLED = (
        os.getenv("COOCCURRENCE_RELATIONSHIP_DISCOVERY_ENABLED", "true").lower()
        == "true"
    )
    TEMPORAL_RELATIONSHIP_DISCOVERY_ENABLED = (
        os.getenv("TEMPORAL_RELATIONSHIP_DISCOVERY_ENABLED", "true").lower() == "true"
    )

    # Relationship Discovery Thresholds
    NEO4J_MIN_CONFIDENCE = float(os.getenv("NEO4J_MIN_CONFIDENCE", "0.3"))
    SEMANTIC_SIMILARITY_THRESHOLD = float(
        os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.7")
    )
    COOCCURRENCE_MIN_FREQUENCY = int(os.getenv("COOCCURRENCE_MIN_FREQUENCY", "2"))
    TEMPORAL_RELATIONSHIP_WINDOW_DAYS = int(
        os.getenv("TEMPORAL_RELATIONSHIP_WINDOW_DAYS", "30")
    )

    # ===============================================
    # Phase 2 - Graph Expansion Configuration
    # ===============================================

    # Graph Expansion Settings
    GRAPH_EXPANSION_ENABLED = (
        os.getenv("GRAPH_EXPANSION_ENABLED", "true").lower() == "true"
    )
    GRAPH_EXPANSION_FACTOR = float(os.getenv("GRAPH_EXPANSION_FACTOR", "2.0"))
    GRAPH_MAX_HOPS = int(os.getenv("GRAPH_MAX_HOPS", "3"))
    GRAPH_MIN_RELATIONSHIP_STRENGTH = float(
        os.getenv("GRAPH_MIN_RELATIONSHIP_STRENGTH", "0.1")
    )

    # Pattern Discovery Settings
    PATTERN_DISCOVERY_ENABLED = (
        os.getenv("PATTERN_DISCOVERY_ENABLED", "true").lower() == "true"
    )
    PATTERN_MIN_FREQUENCY = int(os.getenv("PATTERN_MIN_FREQUENCY", "2"))
    PATTERN_MAX_LENGTH = int(os.getenv("PATTERN_MAX_LENGTH", "3"))
    PATTERN_CACHE_DURATION = int(
        os.getenv("PATTERN_CACHE_DURATION", "1800")
    )  # 30 minutes

    # Multi-hop Traversal Settings
    MULTIHOP_MAX_PATHS = int(os.getenv("MULTIHOP_MAX_PATHS", "20"))
    MULTIHOP_MIN_CONFIDENCE = float(os.getenv("MULTIHOP_MIN_CONFIDENCE", "0.1"))
    MULTIHOP_TIMEOUT_SECONDS = int(os.getenv("MULTIHOP_TIMEOUT_SECONDS", "30"))

    # ===============================================
    # Phase 2 - Memory System Configuration
    # ===============================================

    # Memory System Configuration
    MEMORY_SYSTEM_ENABLED = os.getenv("MEMORY_SYSTEM_ENABLED", "true").lower() == "true"
    MEMORY_CACHE_TTL = int(os.getenv("MEMORY_CACHE_TTL", "3600"))  # 1 hour
    MEMORY_MAX_CACHE_SIZE = int(os.getenv("MEMORY_MAX_CACHE_SIZE", "10000"))
    MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.8"))

    # Memory Cleanup Settings
    MEMORY_CLEANUP_INTERVAL = int(
        os.getenv("MEMORY_CLEANUP_INTERVAL", "300")
    )  # 5 minutes
    MEMORY_LRU_CLEANUP_RATIO = float(
        os.getenv("MEMORY_LRU_CLEANUP_RATIO", "0.1")
    )  # Remove 10% when full
    MEMORY_MAX_MEMORY_MB = int(os.getenv("MEMORY_MAX_MEMORY_MB", "1024"))  # 1GB limit

    # ===============================================
    # Phase 2 - Similarity Thresholds Configuration
    # ===============================================

    # Vector Similarity Settings
    VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", "0.7"))
    VECTOR_MAX_RESULTS = int(os.getenv("VECTOR_MAX_RESULTS", "20"))
    VECTOR_SIMILARITY_METRIC = os.getenv(
        "VECTOR_SIMILARITY_METRIC", "cosine"
    )  # cosine, euclidean, dot_product

    # Graph Similarity Settings
    GRAPH_SIMILARITY_THRESHOLD = float(os.getenv("GRAPH_SIMILARITY_THRESHOLD", "0.5"))
    GRAPH_MAX_RESULTS = int(os.getenv("GRAPH_MAX_RESULTS", "15"))
    GRAPH_RELATIONSHIP_WEIGHT_DECAY = float(
        os.getenv("GRAPH_RELATIONSHIP_WEIGHT_DECAY", "0.8")
    )

    # Hybrid Fusion Settings
    HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
    HYBRID_GRAPH_WEIGHT = float(os.getenv("HYBRID_GRAPH_WEIGHT", "0.4"))
    HYBRID_SIMILARITY_THRESHOLD = float(os.getenv("HYBRID_SIMILARITY_THRESHOLD", "0.6"))
    HYBRID_MAX_RESULTS = int(os.getenv("HYBRID_MAX_RESULTS", "25"))

    # ===============================================
    # Phase 2 - Validation and Self-Correction Configuration
    # ===============================================

    # Self-Correction Framework
    SELF_CORRECTION_ENABLED = (
        os.getenv("SELF_CORRECTION_ENABLED", "true").lower() == "true"
    )
    VALIDATION_MIN_CONFIDENCE = float(os.getenv("VALIDATION_MIN_CONFIDENCE", "0.7"))
    VALIDATION_MAX_ITERATIONS = int(os.getenv("VALIDATION_MAX_ITERATIONS", "3"))
    VALIDATION_TIMEOUT_SECONDS = int(os.getenv("VALIDATION_TIMEOUT_SECONDS", "60"))

    # Confidence Scoring Settings
    CONFIDENCE_SCORING_ENABLED = (
        os.getenv("CONFIDENCE_SCORING_ENABLED", "true").lower() == "true"
    )
    CONFIDENCE_SOURCE_WEIGHT = float(os.getenv("CONFIDENCE_SOURCE_WEIGHT", "0.4"))
    CONFIDENCE_CONSISTENCY_WEIGHT = float(
        os.getenv("CONFIDENCE_CONSISTENCY_WEIGHT", "0.3")
    )
    CONFIDENCE_COMPLETENESS_WEIGHT = float(
        os.getenv("CONFIDENCE_COMPLETENESS_WEIGHT", "0.3")
    )

    # Hallucination Detection
    HALLUCINATION_DETECTION_ENABLED = (
        os.getenv("HALLUCINATION_DETECTION_ENABLED", "true").lower() == "true"
    )
    HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "0.3"))
    HALLUCINATION_CHECK_TIMEOUT = int(os.getenv("HALLUCINATION_CHECK_TIMEOUT", "30"))

    # Validation Rules
    VALIDATION_ENTITY_CONSISTENCY = (
        os.getenv("VALIDATION_ENTITY_CONSISTENCY", "true").lower() == "true"
    )
    VALIDATION_RELATIONSHIP_CONSISTENCY = (
        os.getenv("VALIDATION_RELATIONSHIP_CONSISTENCY", "true").lower() == "true"
    )
    VALIDATION_TEMPORAL_CONSISTENCY = (
        os.getenv("VALIDATION_TEMPORAL_CONSISTENCY", "true").lower() == "true"
    )
    VALIDATION_FACTUAL_CONSISTENCY = (
        os.getenv("VALIDATION_FACTUAL_CONSISTENCY", "true").lower() == "true"
    )

    # ===============================================
    # Phase 2 - Performance and Optimization Configuration
    # ===============================================

    # Performance Settings
    PERFORMANCE_MONITORING_ENABLED = (
        os.getenv("PERFORMANCE_MONITORING_ENABLED", "true").lower() == "true"
    )
    PERFORMANCE_STATS_TTL = int(os.getenv("PERFORMANCE_STATS_TTL", "3600"))  # 1 hour
    PERFORMANCE_SLOW_QUERY_THRESHOLD = float(
        os.getenv("PERFORMANCE_SLOW_QUERY_THRESHOLD", "1.0")
    )  # 1 second

    # Batch Processing Settings
    BATCH_SIZE_ENTITIES = int(os.getenv("BATCH_SIZE_ENTITIES", "1000"))
    BATCH_SIZE_RELATIONSHIPS = int(os.getenv("BATCH_SIZE_RELATIONSHIPS", "500"))
    BATCH_PROCESSING_TIMEOUT = int(
        os.getenv("BATCH_PROCESSING_TIMEOUT", "300")
    )  # 5 minutes

    # Connection Pooling
    CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "20"))
    CONNECTION_TIMEOUT = int(os.getenv("CONNECTION_TIMEOUT", "30"))
    CONNECTION_RETRY_ATTEMPTS = int(os.getenv("CONNECTION_RETRY_ATTEMPTS", "3"))

    # ===============================================
    # Phase 2 - Caching Configuration
    # ===============================================

    # Multi-Level Caching
    QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
    QUERY_CACHE_TTL = int(os.getenv("QUERY_CACHE_TTL", "1800"))  # 30 minutes
    QUERY_CACHE_MAX_SIZE = int(os.getenv("QUERY_CACHE_MAX_SIZE", "5000"))

    EMBEDDING_CACHE_ENABLED = (
        os.getenv("EMBEDDING_CACHE_ENABLED", "true").lower() == "true"
    )
    EMBEDDING_CACHE_TTL = int(os.getenv("EMBEDDING_CACHE_TTL", "7200"))  # 2 hours
    EMBEDDING_CACHE_MAX_SIZE = int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "50000"))

    ENTITY_CACHE_ENABLED = os.getenv("ENTITY_CACHE_ENABLED", "true").lower() == "true"
    ENTITY_CACHE_TTL = int(os.getenv("ENTITY_CACHE_TTL", "3600"))  # 1 hour
    ENTITY_CACHE_MAX_SIZE = int(os.getenv("ENTITY_CACHE_MAX_SIZE", "20000"))

    # Cache Eviction Policies
    CACHE_EVICTION_POLICY = os.getenv("CACHE_EVICTION_POLICY", "lru")  # lru, lfu, fifo
    CACHE_MEMORY_PRESSURE_THRESHOLD = float(
        os.getenv("CACHE_MEMORY_PRESSURE_THRESHOLD", "0.8")
    )  # 80%

    # ===============================================
    # Phase 2 - Quality and Accuracy Configuration
    # ===============================================

    # Result Quality Settings
    RESULT_DEDUPLICATION_ENABLED = (
        os.getenv("RESULT_DEDUPLICATION_ENABLED", "true").lower() == "true"
    )
    RESULT_DEDUPLICATION_THRESHOLD = float(
        os.getenv("RESULT_DEDUPLICATION_THRESHOLD", "0.95")
    )
    RESULT_RANKING_ENABLED = (
        os.getenv("RESULT_RANKING_ENABLED", "true").lower() == "true"
    )

    # Context Quality Settings
    CONTEXT_QUALITY_THRESHOLD = float(os.getenv("CONTEXT_QUALITY_THRESHOLD", "0.6"))
    CONTEXT_RELEVANCE_THRESHOLD = float(os.getenv("CONTEXT_RELEVANCE_THRESHOLD", "0.5"))
    CONTEXT_COMPLETENESS_THRESHOLD = float(
        os.getenv("CONTEXT_COMPLETENESS_THRESHOLD", "0.7")
    )

    # ===============================================
    # System Configuration
    # ===============================================

    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv(
        "LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    LOG_FILE_ENABLED = os.getenv("LOG_FILE_ENABLED", "false").lower() == "true"
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "logs/graphrag.log")

    # Health Check Configuration
    HEALTH_CHECK_ENABLED = os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true"
    HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))  # 1 minute
    HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))  # 10 seconds

    # Development and Debug Settings
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    VERBOSE_LOGGING = os.getenv("VERBOSE_LOGGING", "false").lower() == "true"
    PROFILE_PERFORMANCE = os.getenv("PROFILE_PERFORMANCE", "false").lower() == "true"

    # ===============================================
    # Helper Methods for Configuration Validation
    # ===============================================

    @classmethod
    def validate_configuration(cls) -> list[str]:
        """
        Validate configuration settings and return list of validation errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate threshold ranges
        if not 0.0 <= cls.SEMANTIC_SIMILARITY_THRESHOLD <= 1.0:
            errors.append("SEMANTIC_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")

        if not 0.0 <= cls.VECTOR_SIMILARITY_THRESHOLD <= 1.0:
            errors.append("VECTOR_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")

        if not 0.0 <= cls.GRAPH_SIMILARITY_THRESHOLD <= 1.0:
            errors.append("GRAPH_SIMILARITY_THRESHOLD must be between 0.0 and 1.0")

        if not 0.0 <= cls.VALIDATION_MIN_CONFIDENCE <= 1.0:
            errors.append("VALIDATION_MIN_CONFIDENCE must be between 0.0 and 1.0")

        # Validate hybrid weights sum to 1.0
        weight_sum = cls.HYBRID_VECTOR_WEIGHT + cls.HYBRID_GRAPH_WEIGHT
        if abs(weight_sum - 1.0) > 0.01:
            errors.append(
                f"HYBRID_VECTOR_WEIGHT + HYBRID_GRAPH_WEIGHT must equal 1.0 (current: {weight_sum})"
            )

        # Validate confidence scoring weights sum to 1.0
        confidence_weight_sum = (
            cls.CONFIDENCE_SOURCE_WEIGHT
            + cls.CONFIDENCE_CONSISTENCY_WEIGHT
            + cls.CONFIDENCE_COMPLETENESS_WEIGHT
        )
        if abs(confidence_weight_sum - 1.0) > 0.01:
            errors.append(
                f"Confidence scoring weights must sum to 1.0 (current: {confidence_weight_sum})"
            )

        # Validate positive integers
        if cls.GRAPH_MAX_HOPS <= 0:
            errors.append("GRAPH_MAX_HOPS must be positive")

        if cls.MEMORY_MAX_CACHE_SIZE <= 0:
            errors.append("MEMORY_MAX_CACHE_SIZE must be positive")

        if cls.VALIDATION_MAX_ITERATIONS <= 0:
            errors.append("VALIDATION_MAX_ITERATIONS must be positive")

        return errors

    @classmethod
    def get_phase2_enabled_features(cls) -> list[str]:
        """
        Get list of enabled Phase 2 features.

        Returns:
            List of enabled feature names
        """
        features = []

        if cls.CONTEXT_ENHANCEMENT_ENABLED:
            features.append("Context Enhancement")

        if cls.MEMORY_SYSTEM_ENABLED:
            features.append("Memory System")

        if cls.SELF_CORRECTION_ENABLED:
            features.append("Self-Correction")

        if cls.PERFORMANCE_MONITORING_ENABLED:
            features.append("Performance Monitoring")

        if cls.PATTERN_DISCOVERY_ENABLED:
            features.append("Pattern Discovery")

        if cls.GRAPH_EXPANSION_ENABLED:
            features.append("Graph Expansion")

        return features

    @classmethod
    def get_discovery_strategies(cls) -> list[str]:
        """
        Get list of enabled relationship discovery strategies.

        Returns:
            List of enabled strategy names
        """
        strategies = []

        if cls.NEO4J_RELATIONSHIP_DISCOVERY_ENABLED:
            strategies.append("Neo4j Graph")

        if cls.SEMANTIC_RELATIONSHIP_DISCOVERY_ENABLED:
            strategies.append("Semantic Similarity")

        if cls.COOCCURRENCE_RELATIONSHIP_DISCOVERY_ENABLED:
            strategies.append("Co-occurrence")

        if cls.TEMPORAL_RELATIONSHIP_DISCOVERY_ENABLED:
            strategies.append("Temporal")

        return strategies
