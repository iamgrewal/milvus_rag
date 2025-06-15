"""
Comprehensive tests for the Enhanced Configuration Settings

Tests all Phase 2 configuration parameters, validation methods,
and helper functions for the hybrid RAG system.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag.config.settings import Config


class TestPhase1Configuration:
    """Test Phase 1 configuration parameters."""

    def test_milvus_configuration(self):
        """Test Milvus database configuration."""
        assert hasattr(Config, "MILVUS_HOST")
        assert hasattr(Config, "MILVUS_PORT")
        assert hasattr(Config, "COLLECTION_NAME")
        assert hasattr(Config, "EMBEDDING_DIM")
        
        assert Config.MILVUS_HOST == "localhost"
        assert Config.MILVUS_PORT == "19530"
        assert Config.COLLECTION_NAME == "graph_rag"
        assert Config.EMBEDDING_DIM == 384

    def test_neo4j_configuration(self):
        """Test Neo4j database configuration."""
        assert hasattr(Config, "NEO4J_URI")
        assert hasattr(Config, "NEO4J_USER")
        assert hasattr(Config, "NEO4J_PASSWORD")
        assert hasattr(Config, "NEO4J_MAX_CONNECTION_LIFETIME")
        assert hasattr(Config, "NEO4J_MAX_CONNECTION_POOL_SIZE")
        
        assert Config.NEO4J_URI == "bolt://localhost:7687"
        assert Config.NEO4J_USER == "neo4j"
        assert Config.NEO4J_MAX_CONNECTION_LIFETIME == 3600
        assert Config.NEO4J_MAX_CONNECTION_POOL_SIZE == 50

    def test_redis_configuration(self):
        """Test Redis configuration."""
        assert hasattr(Config, "REDIS_HOST")
        assert hasattr(Config, "REDIS_PORT")
        assert hasattr(Config, "REDIS_DB")
        assert hasattr(Config, "REDIS_PASSWORD")
        
        assert Config.REDIS_HOST == "localhost"
        assert Config.REDIS_PORT == 6379
        assert Config.REDIS_DB == 0

    def test_model_configuration(self):
        """Test model and NLP configuration."""
        assert hasattr(Config, "NLP_MODEL")
        assert hasattr(Config, "EMBEDDING_MODEL")
        assert hasattr(Config, "QA_MODEL")
        
        assert Config.NLP_MODEL == "en_core_web_sm"
        assert Config.EMBEDDING_MODEL == "all-MiniLM-L6-v2"
        assert Config.QA_MODEL == "deepset/bert-base-cased-squad2"


class TestPhase2ContextEnhancement:
    """Test Phase 2 context enhancement configuration."""

    def test_context_enhancement_settings(self):
        """Test context enhancement engine settings."""
        assert hasattr(Config, "CONTEXT_ENHANCEMENT_ENABLED")
        assert hasattr(Config, "CONTEXT_CACHE_TTL")
        assert hasattr(Config, "CONTEXT_MAX_RELATIONSHIPS")
        assert hasattr(Config, "CONTEXT_EXPANSION_DEPTH")
        
        assert Config.CONTEXT_ENHANCEMENT_ENABLED is True
        assert Config.CONTEXT_CACHE_TTL == 300
        assert Config.CONTEXT_MAX_RELATIONSHIPS == 100
        assert Config.CONTEXT_EXPANSION_DEPTH == 2

    def test_relationship_discovery_flags(self):
        """Test relationship discovery strategy flags."""
        assert hasattr(Config, "NEO4J_RELATIONSHIP_DISCOVERY_ENABLED")
        assert hasattr(Config, "SEMANTIC_RELATIONSHIP_DISCOVERY_ENABLED")
        assert hasattr(Config, "COOCCURRENCE_RELATIONSHIP_DISCOVERY_ENABLED")
        assert hasattr(Config, "TEMPORAL_RELATIONSHIP_DISCOVERY_ENABLED")
        
        assert Config.NEO4J_RELATIONSHIP_DISCOVERY_ENABLED is True
        assert Config.SEMANTIC_RELATIONSHIP_DISCOVERY_ENABLED is True
        assert Config.COOCCURRENCE_RELATIONSHIP_DISCOVERY_ENABLED is True
        assert Config.TEMPORAL_RELATIONSHIP_DISCOVERY_ENABLED is True

    def test_relationship_discovery_thresholds(self):
        """Test relationship discovery thresholds."""
        assert hasattr(Config, "NEO4J_MIN_CONFIDENCE")
        assert hasattr(Config, "SEMANTIC_SIMILARITY_THRESHOLD")
        assert hasattr(Config, "COOCCURRENCE_MIN_FREQUENCY")
        assert hasattr(Config, "TEMPORAL_RELATIONSHIP_WINDOW_DAYS")
        
        assert Config.NEO4J_MIN_CONFIDENCE == 0.3
        assert Config.SEMANTIC_SIMILARITY_THRESHOLD == 0.7
        assert Config.COOCCURRENCE_MIN_FREQUENCY == 2
        assert Config.TEMPORAL_RELATIONSHIP_WINDOW_DAYS == 30


class TestPhase2GraphExpansion:
    """Test Phase 2 graph expansion configuration."""

    def test_graph_expansion_settings(self):
        """Test graph expansion settings."""
        assert hasattr(Config, "GRAPH_EXPANSION_ENABLED")
        assert hasattr(Config, "GRAPH_EXPANSION_FACTOR")
        assert hasattr(Config, "GRAPH_MAX_HOPS")
        assert hasattr(Config, "GRAPH_MIN_RELATIONSHIP_STRENGTH")
        
        assert Config.GRAPH_EXPANSION_ENABLED is True
        assert Config.GRAPH_EXPANSION_FACTOR == 2.0
        assert Config.GRAPH_MAX_HOPS == 3
        assert Config.GRAPH_MIN_RELATIONSHIP_STRENGTH == 0.1

    def test_pattern_discovery_settings(self):
        """Test pattern discovery settings."""
        assert hasattr(Config, "PATTERN_DISCOVERY_ENABLED")
        assert hasattr(Config, "PATTERN_MIN_FREQUENCY")
        assert hasattr(Config, "PATTERN_MAX_LENGTH")
        assert hasattr(Config, "PATTERN_CACHE_DURATION")
        
        assert Config.PATTERN_DISCOVERY_ENABLED is True
        assert Config.PATTERN_MIN_FREQUENCY == 2
        assert Config.PATTERN_MAX_LENGTH == 3
        assert Config.PATTERN_CACHE_DURATION == 1800

    def test_multihop_traversal_settings(self):
        """Test multi-hop traversal settings."""
        assert hasattr(Config, "MULTIHOP_MAX_PATHS")
        assert hasattr(Config, "MULTIHOP_MIN_CONFIDENCE")
        assert hasattr(Config, "MULTIHOP_TIMEOUT_SECONDS")
        
        assert Config.MULTIHOP_MAX_PATHS == 20
        assert Config.MULTIHOP_MIN_CONFIDENCE == 0.1
        assert Config.MULTIHOP_TIMEOUT_SECONDS == 30


class TestPhase2MemorySystem:
    """Test Phase 2 memory system configuration."""

    def test_memory_system_settings(self):
        """Test memory system configuration."""
        assert hasattr(Config, "MEMORY_SYSTEM_ENABLED")
        assert hasattr(Config, "MEMORY_CACHE_TTL")
        assert hasattr(Config, "MEMORY_MAX_CACHE_SIZE")
        assert hasattr(Config, "MEMORY_SIMILARITY_THRESHOLD")
        
        assert Config.MEMORY_SYSTEM_ENABLED is True
        assert Config.MEMORY_CACHE_TTL == 3600
        assert Config.MEMORY_MAX_CACHE_SIZE == 10000
        assert Config.MEMORY_SIMILARITY_THRESHOLD == 0.8

    def test_memory_cleanup_settings(self):
        """Test memory cleanup settings."""
        assert hasattr(Config, "MEMORY_CLEANUP_INTERVAL")
        assert hasattr(Config, "MEMORY_LRU_CLEANUP_RATIO")
        assert hasattr(Config, "MEMORY_MAX_MEMORY_MB")
        
        assert Config.MEMORY_CLEANUP_INTERVAL == 300
        assert Config.MEMORY_LRU_CLEANUP_RATIO == 0.1
        assert Config.MEMORY_MAX_MEMORY_MB == 1024


class TestPhase2SimilarityThresholds:
    """Test Phase 2 similarity threshold configuration."""

    def test_vector_similarity_settings(self):
        """Test vector similarity settings."""
        assert hasattr(Config, "VECTOR_SIMILARITY_THRESHOLD")
        assert hasattr(Config, "VECTOR_MAX_RESULTS")
        assert hasattr(Config, "VECTOR_SIMILARITY_METRIC")
        
        assert Config.VECTOR_SIMILARITY_THRESHOLD == 0.7
        assert Config.VECTOR_MAX_RESULTS == 20
        assert Config.VECTOR_SIMILARITY_METRIC == "cosine"

    def test_graph_similarity_settings(self):
        """Test graph similarity settings."""
        assert hasattr(Config, "GRAPH_SIMILARITY_THRESHOLD")
        assert hasattr(Config, "GRAPH_MAX_RESULTS")
        assert hasattr(Config, "GRAPH_RELATIONSHIP_WEIGHT_DECAY")
        
        assert Config.GRAPH_SIMILARITY_THRESHOLD == 0.5
        assert Config.GRAPH_MAX_RESULTS == 15
        assert Config.GRAPH_RELATIONSHIP_WEIGHT_DECAY == 0.8

    def test_hybrid_fusion_settings(self):
        """Test hybrid fusion settings."""
        assert hasattr(Config, "HYBRID_VECTOR_WEIGHT")
        assert hasattr(Config, "HYBRID_GRAPH_WEIGHT")
        assert hasattr(Config, "HYBRID_SIMILARITY_THRESHOLD")
        assert hasattr(Config, "HYBRID_MAX_RESULTS")
        
        assert Config.HYBRID_VECTOR_WEIGHT == 0.6
        assert Config.HYBRID_GRAPH_WEIGHT == 0.4
        assert Config.HYBRID_SIMILARITY_THRESHOLD == 0.6
        assert Config.HYBRID_MAX_RESULTS == 25
        
        # Test that weights sum to 1.0
        assert abs((Config.HYBRID_VECTOR_WEIGHT + Config.HYBRID_GRAPH_WEIGHT) - 1.0) < 0.01


class TestPhase2ValidationSettings:
    """Test Phase 2 validation and self-correction configuration."""

    def test_self_correction_framework(self):
        """Test self-correction framework settings."""
        assert hasattr(Config, "SELF_CORRECTION_ENABLED")
        assert hasattr(Config, "VALIDATION_MIN_CONFIDENCE")
        assert hasattr(Config, "VALIDATION_MAX_ITERATIONS")
        assert hasattr(Config, "VALIDATION_TIMEOUT_SECONDS")
        
        assert Config.SELF_CORRECTION_ENABLED is True
        assert Config.VALIDATION_MIN_CONFIDENCE == 0.7
        assert Config.VALIDATION_MAX_ITERATIONS == 3
        assert Config.VALIDATION_TIMEOUT_SECONDS == 60

    def test_confidence_scoring_settings(self):
        """Test confidence scoring settings."""
        assert hasattr(Config, "CONFIDENCE_SCORING_ENABLED")
        assert hasattr(Config, "CONFIDENCE_SOURCE_WEIGHT")
        assert hasattr(Config, "CONFIDENCE_CONSISTENCY_WEIGHT")
        assert hasattr(Config, "CONFIDENCE_COMPLETENESS_WEIGHT")
        
        assert Config.CONFIDENCE_SCORING_ENABLED is True
        assert Config.CONFIDENCE_SOURCE_WEIGHT == 0.4
        assert Config.CONFIDENCE_CONSISTENCY_WEIGHT == 0.3
        assert Config.CONFIDENCE_COMPLETENESS_WEIGHT == 0.3
        
        # Test that confidence weights sum to 1.0
        weight_sum = (
            Config.CONFIDENCE_SOURCE_WEIGHT +
            Config.CONFIDENCE_CONSISTENCY_WEIGHT +
            Config.CONFIDENCE_COMPLETENESS_WEIGHT
        )
        assert abs(weight_sum - 1.0) < 0.01

    def test_hallucination_detection(self):
        """Test hallucination detection settings."""
        assert hasattr(Config, "HALLUCINATION_DETECTION_ENABLED")
        assert hasattr(Config, "HALLUCINATION_THRESHOLD")
        assert hasattr(Config, "HALLUCINATION_CHECK_TIMEOUT")
        
        assert Config.HALLUCINATION_DETECTION_ENABLED is True
        assert Config.HALLUCINATION_THRESHOLD == 0.3
        assert Config.HALLUCINATION_CHECK_TIMEOUT == 30

    def test_validation_rules(self):
        """Test validation rule settings."""
        assert hasattr(Config, "VALIDATION_ENTITY_CONSISTENCY")
        assert hasattr(Config, "VALIDATION_RELATIONSHIP_CONSISTENCY")
        assert hasattr(Config, "VALIDATION_TEMPORAL_CONSISTENCY")
        assert hasattr(Config, "VALIDATION_FACTUAL_CONSISTENCY")
        
        assert Config.VALIDATION_ENTITY_CONSISTENCY is True
        assert Config.VALIDATION_RELATIONSHIP_CONSISTENCY is True
        assert Config.VALIDATION_TEMPORAL_CONSISTENCY is True
        assert Config.VALIDATION_FACTUAL_CONSISTENCY is True


class TestPhase2PerformanceSettings:
    """Test Phase 2 performance and optimization configuration."""

    def test_performance_settings(self):
        """Test performance monitoring settings."""
        assert hasattr(Config, "PERFORMANCE_MONITORING_ENABLED")
        assert hasattr(Config, "PERFORMANCE_STATS_TTL")
        assert hasattr(Config, "PERFORMANCE_SLOW_QUERY_THRESHOLD")
        
        assert Config.PERFORMANCE_MONITORING_ENABLED is True
        assert Config.PERFORMANCE_STATS_TTL == 3600
        assert Config.PERFORMANCE_SLOW_QUERY_THRESHOLD == 1.0

    def test_batch_processing_settings(self):
        """Test batch processing settings."""
        assert hasattr(Config, "BATCH_SIZE_ENTITIES")
        assert hasattr(Config, "BATCH_SIZE_RELATIONSHIPS")
        assert hasattr(Config, "BATCH_PROCESSING_TIMEOUT")
        
        assert Config.BATCH_SIZE_ENTITIES == 1000
        assert Config.BATCH_SIZE_RELATIONSHIPS == 500
        assert Config.BATCH_PROCESSING_TIMEOUT == 300

    def test_connection_pooling_settings(self):
        """Test connection pooling settings."""
        assert hasattr(Config, "CONNECTION_POOL_SIZE")
        assert hasattr(Config, "CONNECTION_TIMEOUT")
        assert hasattr(Config, "CONNECTION_RETRY_ATTEMPTS")
        
        assert Config.CONNECTION_POOL_SIZE == 20
        assert Config.CONNECTION_TIMEOUT == 30
        assert Config.CONNECTION_RETRY_ATTEMPTS == 3


class TestPhase2CachingSettings:
    """Test Phase 2 caching configuration."""

    def test_multi_level_caching(self):
        """Test multi-level caching settings."""
        # Query cache
        assert hasattr(Config, "QUERY_CACHE_ENABLED")
        assert hasattr(Config, "QUERY_CACHE_TTL")
        assert hasattr(Config, "QUERY_CACHE_MAX_SIZE")
        
        assert Config.QUERY_CACHE_ENABLED is True
        assert Config.QUERY_CACHE_TTL == 1800
        assert Config.QUERY_CACHE_MAX_SIZE == 5000
        
        # Embedding cache
        assert hasattr(Config, "EMBEDDING_CACHE_ENABLED")
        assert hasattr(Config, "EMBEDDING_CACHE_TTL")
        assert hasattr(Config, "EMBEDDING_CACHE_MAX_SIZE")
        
        assert Config.EMBEDDING_CACHE_ENABLED is True
        assert Config.EMBEDDING_CACHE_TTL == 7200
        assert Config.EMBEDDING_CACHE_MAX_SIZE == 50000
        
        # Entity cache
        assert hasattr(Config, "ENTITY_CACHE_ENABLED")
        assert hasattr(Config, "ENTITY_CACHE_TTL")
        assert hasattr(Config, "ENTITY_CACHE_MAX_SIZE")
        
        assert Config.ENTITY_CACHE_ENABLED is True
        assert Config.ENTITY_CACHE_TTL == 3600
        assert Config.ENTITY_CACHE_MAX_SIZE == 20000

    def test_cache_eviction_policies(self):
        """Test cache eviction policy settings."""
        assert hasattr(Config, "CACHE_EVICTION_POLICY")
        assert hasattr(Config, "CACHE_MEMORY_PRESSURE_THRESHOLD")
        
        assert Config.CACHE_EVICTION_POLICY == "lru"
        assert Config.CACHE_MEMORY_PRESSURE_THRESHOLD == 0.8


class TestQualityAndSystemSettings:
    """Test quality and system configuration."""

    def test_result_quality_settings(self):
        """Test result quality settings."""
        assert hasattr(Config, "RESULT_DEDUPLICATION_ENABLED")
        assert hasattr(Config, "RESULT_DEDUPLICATION_THRESHOLD")
        assert hasattr(Config, "RESULT_RANKING_ENABLED")
        
        assert Config.RESULT_DEDUPLICATION_ENABLED is True
        assert Config.RESULT_DEDUPLICATION_THRESHOLD == 0.95
        assert Config.RESULT_RANKING_ENABLED is True

    def test_context_quality_settings(self):
        """Test context quality settings."""
        assert hasattr(Config, "CONTEXT_QUALITY_THRESHOLD")
        assert hasattr(Config, "CONTEXT_RELEVANCE_THRESHOLD")
        assert hasattr(Config, "CONTEXT_COMPLETENESS_THRESHOLD")
        
        assert Config.CONTEXT_QUALITY_THRESHOLD == 0.6
        assert Config.CONTEXT_RELEVANCE_THRESHOLD == 0.5
        assert Config.CONTEXT_COMPLETENESS_THRESHOLD == 0.7

    def test_logging_configuration(self):
        """Test logging configuration."""
        assert hasattr(Config, "LOG_LEVEL")
        assert hasattr(Config, "LOG_FORMAT")
        assert hasattr(Config, "LOG_FILE_ENABLED")
        assert hasattr(Config, "LOG_FILE_PATH")
        
        assert Config.LOG_LEVEL == "INFO"
        assert "%(asctime)s" in Config.LOG_FORMAT
        assert Config.LOG_FILE_ENABLED is False
        assert Config.LOG_FILE_PATH == "logs/graphrag.log"

    def test_health_check_configuration(self):
        """Test health check configuration."""
        assert hasattr(Config, "HEALTH_CHECK_ENABLED")
        assert hasattr(Config, "HEALTH_CHECK_INTERVAL")
        assert hasattr(Config, "HEALTH_CHECK_TIMEOUT")
        
        assert Config.HEALTH_CHECK_ENABLED is True
        assert Config.HEALTH_CHECK_INTERVAL == 60
        assert Config.HEALTH_CHECK_TIMEOUT == 10

    def test_debug_settings(self):
        """Test debug and development settings."""
        assert hasattr(Config, "DEBUG_MODE")
        assert hasattr(Config, "VERBOSE_LOGGING")
        assert hasattr(Config, "PROFILE_PERFORMANCE")
        
        assert Config.DEBUG_MODE is False
        assert Config.VERBOSE_LOGGING is False
        assert Config.PROFILE_PERFORMANCE is False


class TestConfigurationValidation:
    """Test configuration validation methods."""

    def test_validate_configuration_success(self):
        """Test successful configuration validation."""
        errors = Config.validate_configuration()
        assert isinstance(errors, list)
        assert len(errors) == 0  # Should be no errors with default config

    def test_validate_threshold_ranges(self):
        """Test threshold range validation."""
        # Test with invalid threshold values
        with patch.object(Config, 'SEMANTIC_SIMILARITY_THRESHOLD', 1.5):
            errors = Config.validate_configuration()
            assert any("SEMANTIC_SIMILARITY_THRESHOLD" in error for error in errors)
        
        with patch.object(Config, 'VECTOR_SIMILARITY_THRESHOLD', -0.1):
            errors = Config.validate_configuration()
            assert any("VECTOR_SIMILARITY_THRESHOLD" in error for error in errors)

    def test_validate_weight_sums(self):
        """Test weight sum validation."""
        # Test invalid hybrid weights
        with patch.object(Config, 'HYBRID_VECTOR_WEIGHT', 0.7):
            with patch.object(Config, 'HYBRID_GRAPH_WEIGHT', 0.4):
                errors = Config.validate_configuration()
                assert any("HYBRID_VECTOR_WEIGHT + HYBRID_GRAPH_WEIGHT" in error for error in errors)
        
        # Test invalid confidence weights
        with patch.object(Config, 'CONFIDENCE_SOURCE_WEIGHT', 0.5):
            errors = Config.validate_configuration()
            assert any("Confidence scoring weights" in error for error in errors)

    def test_validate_positive_integers(self):
        """Test positive integer validation."""
        with patch.object(Config, 'GRAPH_MAX_HOPS', 0):
            errors = Config.validate_configuration()
            assert any("GRAPH_MAX_HOPS must be positive" in error for error in errors)
        
        with patch.object(Config, 'MEMORY_MAX_CACHE_SIZE', -100):
            errors = Config.validate_configuration()
            assert any("MEMORY_MAX_CACHE_SIZE must be positive" in error for error in errors)

    def test_get_phase2_enabled_features(self):
        """Test getting enabled Phase 2 features."""
        features = Config.get_phase2_enabled_features()
        assert isinstance(features, list)
        
        expected_features = [
            "Context Enhancement",
            "Memory System",
            "Self-Correction",
            "Performance Monitoring",
            "Pattern Discovery",
            "Graph Expansion"
        ]
        
        for feature in expected_features:
            assert feature in features

    def test_get_discovery_strategies(self):
        """Test getting enabled discovery strategies."""
        strategies = Config.get_discovery_strategies()
        assert isinstance(strategies, list)
        
        expected_strategies = [
            "Neo4j Graph",
            "Semantic Similarity",
            "Co-occurrence",
            "Temporal"
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategies

    def test_get_discovery_strategies_with_disabled_features(self):
        """Test discovery strategies with some features disabled."""
        with patch.object(Config, 'SEMANTIC_RELATIONSHIP_DISCOVERY_ENABLED', False):
            with patch.object(Config, 'TEMPORAL_RELATIONSHIP_DISCOVERY_ENABLED', False):
                strategies = Config.get_discovery_strategies()
                
                assert "Neo4j Graph" in strategies
                assert "Co-occurrence" in strategies
                assert "Semantic Similarity" not in strategies
                assert "Temporal" not in strategies


class TestEnvironmentVariableOverrides:
    """Test environment variable overrides."""

    def test_environment_variable_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {"MILVUS_HOST": "test-host"}):
            # Need to reload the class to pick up new environment variable
            from importlib import reload
            import graphrag.config.settings as settings_module
            reload(settings_module)
            
            assert settings_module.Config.MILVUS_HOST == "test-host"

    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing."""
        with patch.dict(os.environ, {"CONTEXT_ENHANCEMENT_ENABLED": "false"}):
            from importlib import reload
            import graphrag.config.settings as settings_module
            reload(settings_module)
            
            assert settings_module.Config.CONTEXT_ENHANCEMENT_ENABLED is False

    def test_numeric_environment_variables(self):
        """Test numeric environment variable parsing."""
        with patch.dict(os.environ, {
            "MEMORY_MAX_CACHE_SIZE": "5000",
            "VECTOR_SIMILARITY_THRESHOLD": "0.85"
        }):
            from importlib import reload
            import graphrag.config.settings as settings_module
            reload(settings_module)
            
            assert settings_module.Config.MEMORY_MAX_CACHE_SIZE == 5000
            assert settings_module.Config.VECTOR_SIMILARITY_THRESHOLD == 0.85


class TestConfigurationGrouping:
    """Test configuration grouping and organization."""

    def test_phase1_parameters_exist(self):
        """Test that all Phase 1 parameters are preserved."""
        phase1_params = [
            "MILVUS_HOST", "MILVUS_PORT", "COLLECTION_NAME", "EMBEDDING_DIM",
            "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD",
            "REDIS_HOST", "REDIS_PORT", "REDIS_DB",
            "NLP_MODEL", "EMBEDDING_MODEL", "QA_MODEL"
        ]
        
        for param in phase1_params:
            assert hasattr(Config, param), f"Phase 1 parameter {param} is missing"

    def test_phase2_parameters_exist(self):
        """Test that all major Phase 2 parameters are present."""
        phase2_sections = [
            # Context Enhancement
            "CONTEXT_ENHANCEMENT_ENABLED", "SEMANTIC_SIMILARITY_THRESHOLD",
            # Graph Expansion
            "GRAPH_EXPANSION_ENABLED", "PATTERN_DISCOVERY_ENABLED",
            # Memory System
            "MEMORY_SYSTEM_ENABLED", "MEMORY_SIMILARITY_THRESHOLD",
            # Validation
            "SELF_CORRECTION_ENABLED", "VALIDATION_MIN_CONFIDENCE",
            # Performance
            "PERFORMANCE_MONITORING_ENABLED", "BATCH_SIZE_ENTITIES",
            # Caching
            "QUERY_CACHE_ENABLED", "EMBEDDING_CACHE_ENABLED"
        ]
        
        for param in phase2_sections:
            assert hasattr(Config, param), f"Phase 2 parameter {param} is missing"

    def test_configuration_completeness(self):
        """Test that configuration covers all major Phase 2 areas."""
        # Get all class attributes that are configuration parameters
        config_attrs = [attr for attr in dir(Config) 
                       if not attr.startswith('_') and not callable(getattr(Config, attr))]
        
        # Should have substantial number of configuration parameters
        assert len(config_attrs) >= 80, f"Expected at least 80 config parameters, found {len(config_attrs)}"
        
        # Check for presence of major configuration categories
        categories = [
            "CONTEXT_", "GRAPH_", "MEMORY_", "VALIDATION_", 
            "PERFORMANCE_", "CACHE_", "HYBRID_", "VECTOR_"
        ]
        
        for category in categories:
            category_params = [attr for attr in config_attrs if attr.startswith(category)]
            assert len(category_params) > 0, f"No parameters found for category {category}"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])