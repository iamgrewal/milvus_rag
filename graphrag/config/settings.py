# graphrag/config/settings.py

import os


class Config:
    """
    Production configuration for the Hybrid RAG System.

    Centralizes all configuration parameters for Milvus, Neo4j, NLP,
    and system-wide settings according to the rhoSearcher ruleset.
    """

    # === Milvus Vector Store Configuration ===
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "hybrid_rag_collection")
    EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "384"))

    # Milvus HNSW Index Configuration (according to ruleset)
    MILVUS_INDEX_TYPE = "HNSW"
    MILVUS_METRIC_TYPE = "L2"
    MILVUS_INDEX_M = int(os.getenv("MILVUS_INDEX_M", "16"))  # Bi-directional links
    MILVUS_INDEX_EF_CONSTRUCTION = int(os.getenv("MILVUS_INDEX_EF_CONSTRUCTION", "256"))

    # Milvus Search Parameters (according to ruleset)
    MILVUS_SEARCH_EF = int(os.getenv("MILVUS_SEARCH_EF", "32"))
    MILVUS_SEARCH_OFFSET = int(os.getenv("MILVUS_SEARCH_OFFSET", "5"))

    # Milvus Partitioning Strategy
    EXPECTED_DOC_COUNT = int(os.getenv("EXPECTED_DOC_COUNT", "1000000"))

    # === Neo4j Graph Store Configuration ===
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    # Neo4j Connection Pool Settings
    NEO4J_MAX_CONNECTION_LIFETIME = int(
        os.getenv("NEO4J_MAX_CONNECTION_LIFETIME", "1800")
    )  # 30 minutes
    NEO4J_MAX_CONNECTION_POOL_SIZE = int(
        os.getenv("NEO4J_MAX_CONNECTION_POOL_SIZE", "50")
    )
    NEO4J_CONNECTION_ACQUISITION_TIMEOUT = int(
        os.getenv("NEO4J_CONNECTION_ACQUISITION_TIMEOUT", "60")
    )

    # Neo4j Query Configuration (according to ruleset)
    NEO4J_MIN_RELATIONSHIP_STRENGTH = float(
        os.getenv("NEO4J_MIN_RELATIONSHIP_STRENGTH", "0.7")
    )
    NEO4J_MAX_TRAVERSAL_DEPTH = int(os.getenv("NEO4J_MAX_TRAVERSAL_DEPTH", "3"))
    NEO4J_MAX_RESULTS_PER_QUERY = int(os.getenv("NEO4J_MAX_RESULTS_PER_QUERY", "50"))

    # === NLP and Entity Extraction Configuration ===
    NLP_MODEL = os.getenv("NLP_MODEL", "en_core_web_trf")  # Transformer model
    NLP_FALLBACK_MODEL = os.getenv("NLP_FALLBACK_MODEL", "en_core_web_sm")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Entity Extraction Settings
    ENTITY_CONFIDENCE_THRESHOLD = float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.7"))
    MAX_ENTITY_LENGTH = int(os.getenv("MAX_ENTITY_LENGTH", "100"))

    # === Fusion and Orchestration Configuration ===
    # Fusion Parameters (according to ruleset)
    FUSION_VECTOR_WEIGHT = float(os.getenv("FUSION_VECTOR_WEIGHT", "0.6"))
    FUSION_GRAPH_WEIGHT = float(os.getenv("FUSION_GRAPH_WEIGHT", "0.4"))
    FUSION_RRF_K = int(
        os.getenv("FUSION_RRF_K", "60")
    )  # Reciprocal rank fusion parameter

    # Confidence and Quality Thresholds (according to ruleset)
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    HALLUCINATION_ENTITY_THRESHOLD = float(
        os.getenv("HALLUCINATION_ENTITY_THRESHOLD", "0.3")
    )

    # Self-Correction Limits (according to ruleset)
    MAX_CORRECTION_ITERATIONS = int(os.getenv("MAX_CORRECTION_ITERATIONS", "3"))

    # === Service Configuration ===
    # RAG Service Settings
    MAX_QUERY_LENGTH = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
    QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))
    ENABLE_MONITORING = os.getenv("ENABLE_MONITORING", "true").lower() == "true"

    # Retry Configuration
    MAX_RETRY_ATTEMPTS = int(os.getenv("MAX_RETRY_ATTEMPTS", "3"))
    RETRY_MIN_WAIT = int(os.getenv("RETRY_MIN_WAIT", "4"))
    RETRY_MAX_WAIT = int(os.getenv("RETRY_MAX_WAIT", "10"))

    # === External Service Configuration ===
    # LLM Service Configuration
    LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "")
    LLM_API_KEY = os.getenv("LLM_API_KEY", "")
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Embedding Service Configuration
    EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "")
    EMBEDDING_SERVICE_API_KEY = os.getenv("EMBEDDING_SERVICE_API_KEY", "")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

    # Fallback Search Services (according to ruleset)
    FALLBACK_SEARCH_ENABLED = (
        os.getenv("FALLBACK_SEARCH_ENABLED", "true").lower() == "true"
    )
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
    EXA_API_KEY = os.getenv("EXA_API_KEY", "")
    BING_API_KEY = os.getenv("BING_API_KEY", "")

    # === Performance and Monitoring Configuration ===
    # Performance Targets (according to ruleset KPIs)
    TARGET_LATENCY_95_PERCENTILE = float(
        os.getenv("TARGET_LATENCY_95_PERCENTILE", "2.0")
    )  # seconds
    TARGET_ACCURACY_F1_SCORE = float(os.getenv("TARGET_ACCURACY_F1_SCORE", "0.85"))
    TARGET_UPTIME = float(os.getenv("TARGET_UPTIME", "0.9995"))  # 99.95%
    TARGET_FALLBACK_RECOVERY = float(
        os.getenv("TARGET_FALLBACK_RECOVERY", "0.80")
    )  # 80%

    # Monitoring and Observability
    ENABLE_METRICS_COLLECTION = (
        os.getenv("ENABLE_METRICS_COLLECTION", "true").lower() == "true"
    )
    METRICS_EXPORT_INTERVAL = int(os.getenv("METRICS_EXPORT_INTERVAL", "60"))  # seconds
    PROMETHEUS_ENDPOINT = os.getenv("PROMETHEUS_ENDPOINT", "")
    GRAFANA_ENDPOINT = os.getenv("GRAFANA_ENDPOINT", "")

    # === Security Configuration ===
    # TLS and Security Settings (according to ruleset)
    ENABLE_TLS = os.getenv("ENABLE_TLS", "true").lower() == "true"
    TLS_CERT_PATH = os.getenv("TLS_CERT_PATH", "")
    TLS_KEY_PATH = os.getenv("TLS_KEY_PATH", "")

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE = int(
        os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60")
    )
    RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))

    # Authentication and Authorization
    ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
    JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    # RBAC Roles (according to ruleset)
    RBAC_ROLES = [
        "rag_reader",  # Read-only access to RAG queries
        "rag_operator",  # Standard operations and monitoring
        "rag_admin",  # Full administrative access
    ]

    # === Logging Configuration ===
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.getenv("LOG_FORMAT", "json")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "")
    ENABLE_STRUCTURED_LOGGING = (
        os.getenv("ENABLE_STRUCTURED_LOGGING", "true").lower() == "true"
    )

    # === Development and Testing Configuration ===
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
    ENABLE_PROFILING = os.getenv("ENABLE_PROFILING", "false").lower() == "true"

    # Testing Configuration
    TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", "")
    TEST_MODE = os.getenv("TEST_MODE", "false").lower() == "true"
    MOCK_EXTERNAL_SERVICES = (
        os.getenv("MOCK_EXTERNAL_SERVICES", "false").lower() == "true"
    )

    # === Deployment Configuration ===
    # Docker and Container Settings
    CONTAINER_NAME = os.getenv("CONTAINER_NAME", "hybrid-rag-service")
    CONTAINER_PORT = int(os.getenv("CONTAINER_PORT", "8000"))
    HEALTH_CHECK_ENDPOINT = os.getenv("HEALTH_CHECK_ENDPOINT", "/health")

    # Kubernetes Settings
    NAMESPACE = os.getenv("NAMESPACE", "default")
    SERVICE_NAME = os.getenv("SERVICE_NAME", "hybrid-rag")
    REPLICA_COUNT = int(os.getenv("REPLICA_COUNT", "1"))

    @classmethod
    def validate_config(cls) -> dict:
        """
        Validate configuration settings and return validation results.

        Returns:
            Dictionary containing validation status and any errors
        """
        errors = []
        warnings = []

        # Validate required settings
        required_settings = [
            ("MILVUS_HOST", cls.MILVUS_HOST),
            ("NEO4J_URI", cls.NEO4J_URI),
            ("NEO4J_USER", cls.NEO4J_USER),
            ("NEO4J_PASSWORD", cls.NEO4J_PASSWORD),
        ]

        for setting_name, setting_value in required_settings:
            if not setting_value:
                errors.append(f"Required setting {setting_name} is not configured")

        # Validate numeric ranges
        if not (0.0 <= cls.CONFIDENCE_THRESHOLD <= 1.0):
            errors.append("CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")

        if not (0.0 <= cls.FUSION_VECTOR_WEIGHT <= 1.0):
            errors.append("FUSION_VECTOR_WEIGHT must be between 0.0 and 1.0")

        if not (0.0 <= cls.FUSION_GRAPH_WEIGHT <= 1.0):
            errors.append("FUSION_GRAPH_WEIGHT must be between 0.0 and 1.0")

        if abs((cls.FUSION_VECTOR_WEIGHT + cls.FUSION_GRAPH_WEIGHT) - 1.0) > 0.01:
            warnings.append(
                "FUSION_VECTOR_WEIGHT + FUSION_GRAPH_WEIGHT should sum to 1.0"
            )

        # Validate dimensional consistency
        if cls.EMBEDDING_DIM <= 0:
            errors.append("EMBEDDING_DIM must be a positive integer")

        # Validate timeout settings
        if cls.QUERY_TIMEOUT_SECONDS <= 0:
            errors.append("QUERY_TIMEOUT_SECONDS must be positive")

        # Check for production readiness
        if cls.ENVIRONMENT == "production":
            if not cls.ENABLE_TLS:
                warnings.append("TLS should be enabled in production")

            if not cls.ENABLE_AUTH:
                warnings.append("Authentication should be enabled in production")

            if cls.DEBUG_MODE:
                warnings.append("Debug mode should be disabled in production")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "settings_count": len(
                [attr for attr in dir(cls) if not attr.startswith("_")]
            ),
        }

    @classmethod
    def get_summary(cls) -> dict:
        """
        Get a summary of current configuration settings.

        Returns:
            Dictionary containing configuration summary
        """
        return {
            "environment": cls.ENVIRONMENT,
            "milvus": {
                "host": cls.MILVUS_HOST,
                "port": cls.MILVUS_PORT,
                "collection": cls.COLLECTION_NAME,
                "embedding_dim": cls.EMBEDDING_DIM,
                "index_type": cls.MILVUS_INDEX_TYPE,
            },
            "neo4j": {
                "uri": cls.NEO4J_URI,
                "user": cls.NEO4J_USER,
                "min_relationship_strength": cls.NEO4J_MIN_RELATIONSHIP_STRENGTH,
                "max_traversal_depth": cls.NEO4J_MAX_TRAVERSAL_DEPTH,
            },
            "nlp": {
                "model": cls.NLP_MODEL,
                "confidence_threshold": cls.ENTITY_CONFIDENCE_THRESHOLD,
            },
            "fusion": {
                "confidence_threshold": cls.CONFIDENCE_THRESHOLD,
                "vector_weight": cls.FUSION_VECTOR_WEIGHT,
                "graph_weight": cls.FUSION_GRAPH_WEIGHT,
            },
            "performance_targets": {
                "latency_95p": cls.TARGET_LATENCY_95_PERCENTILE,
                "accuracy_f1": cls.TARGET_ACCURACY_F1_SCORE,
                "uptime": cls.TARGET_UPTIME,
            },
            "security": {
                "tls_enabled": cls.ENABLE_TLS,
                "auth_enabled": cls.ENABLE_AUTH,
                "rbac_roles": cls.RBAC_ROLES,
            },
        }
