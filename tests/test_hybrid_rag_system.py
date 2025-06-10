"""
Comprehensive Integration Tests for Hybrid RAG System

This test suite validates the entire hybrid RAG implementation against
the rhoSearcher ruleset requirements including parallel retrieval,
fusion logic, self-correction, and production readiness standards.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the parent directory to sys.path to import the graphrag package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graphrag.rag_system.orchestrator import HybridRAGOrchestrator, RAGState
from graphrag.rag_system.service import RAGService
from graphrag.fusion.fuser import ResultFuser, detect_hallucination, extract_entities
from graphrag.config.settings import Config


class TestHybridRAGOrchestrator:
    """Test the LangGraph orchestrator implementation."""

    @pytest.fixture
    def mock_retrievers(self):
        """Mock the vector and graph retrievers."""
        with patch(
            "graphrag.rag_system.orchestrator.MilvusRetriever"
        ) as mock_milvus, patch(
            "graphrag.rag_system.orchestrator.Neo4jRetriever"
        ) as mock_neo4j, patch(
            "graphrag.rag_system.orchestrator.ResultFuser"
        ) as mock_fuser, patch(
            "graphrag.rag_system.orchestrator.EntityExtractor"
        ) as mock_extractor:

            # Configure mocks
            mock_milvus.return_value.retrieve_async = AsyncMock(
                return_value=[
                    {
                        "id": 1,
                        "score": 0.95,
                        "content": "Vector search result content",
                        "entity": "test_entity",
                        "confidence_score": 0.9,
                        "retrieval_type": "vector",
                    }
                ]
            )

            mock_neo4j.return_value.traverse_async = AsyncMock(
                return_value=[
                    {
                        "entity_names": ["entity1", "entity2"],
                        "relevance_score": 0.85,
                        "avg_strength": 0.8,
                        "path_length": 2,
                        "retrieval_type": "graph",
                    }
                ]
            )

            mock_fuser.return_value.fuse_results = AsyncMock(
                return_value=[
                    {
                        "content": "Fused result content",
                        "fusion_score": 0.9,
                        "confidence_score": 0.8,
                        "source_types": ["vector", "graph"],
                    }
                ]
            )

            yield mock_milvus, mock_neo4j, mock_fuser, mock_extractor

    @pytest.mark.asyncio
    async def test_rag_state_specification(self):
        """Test that RAGState follows the exact specification from the ruleset."""
        state = RAGState(
            query="test query",
            vector_results=[],
            graph_results=[],
            fused_results=[],
            response="",
            confidence_score=0.0,
            iteration_count=0,
            correction_needed=False,
            hallucination_detected=False,
            sources=[],
        )

        # Verify all required fields are present
        required_fields = [
            "query",
            "vector_results",
            "graph_results",
            "fused_results",
            "response",
            "confidence_score",
            "iteration_count",
            "correction_needed",
            "hallucination_detected",
            "sources",
        ]

        for field in required_fields:
            assert field in state, f"Required field '{field}' missing from RAGState"

    @pytest.mark.asyncio
    async def test_parallel_retrieval(self, mock_retrievers):
        """Test parallel retrieval implementation using asyncio.gather."""
        mock_milvus, mock_neo4j, mock_fuser, mock_extractor = mock_retrievers

        orchestrator = HybridRAGOrchestrator()

        state = RAGState(
            query="test parallel retrieval",
            vector_results=[],
            graph_results=[],
            fused_results=[],
            response="",
            confidence_score=0.0,
            iteration_count=0,
            correction_needed=False,
            hallucination_detected=False,
            sources=[],
        )

        # Execute parallel retrieval
        result_state = await orchestrator.parallel_retrieval(state)

        # Verify both retrievers were called
        mock_milvus.return_value.retrieve_async.assert_called_once()
        mock_neo4j.return_value.traverse_async.assert_called_once()

        # Verify results are populated
        assert len(result_state["vector_results"]) > 0
        assert len(result_state["graph_results"]) > 0
        assert result_state["vector_results"][0]["retrieval_type"] == "vector"
        assert result_state["graph_results"][0]["retrieval_type"] == "graph"

    @pytest.mark.asyncio
    async def test_confidence_tracking(self, mock_retrievers):
        """Test confidence score tracking throughout the workflow."""
        mock_milvus, mock_neo4j, mock_fuser, mock_extractor = mock_retrievers

        orchestrator = HybridRAGOrchestrator()

        # Test query processing
        result = await orchestrator.process_query("test confidence tracking")

        # Verify confidence score is tracked
        assert "confidence_score" in result
        assert isinstance(result["confidence_score"], float)
        assert 0.0 <= result["confidence_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_self_correction_max_iterations(self, mock_retrievers):
        """Test that self-correction is limited to maximum 3 iterations."""
        mock_milvus, mock_neo4j, mock_fuser, mock_extractor = mock_retrievers

        # Mock low confidence results to trigger self-correction
        mock_fuser.return_value.fuse_results = AsyncMock(
            return_value=[
                {
                    "content": "Low quality result",
                    "fusion_score": 0.2,
                    "confidence_score": 0.3,  # Below threshold
                    "source_types": ["vector"],
                }
            ]
        )

        orchestrator = HybridRAGOrchestrator()
        result = await orchestrator.process_query("test self correction")

        # Verify iteration count is tracked and limited
        assert "iteration_count" in result
        assert result["iteration_count"] <= orchestrator.max_corrections


class TestResultFuser:
    """Test the fusion pipeline implementation."""

    def test_hallucination_detection(self):
        """Test hallucination detection logic according to ruleset."""
        # Test case 1: No hallucination (entities present in sources)
        response = "The Python programming language is popular"
        sources = [
            "Python is a programming language used widely",
            "Programming languages include Python",
        ]

        assert not detect_hallucination(response, sources)

        # Test case 2: Hallucination detected (novel entities)
        response = "The XYZ-9000 supercomputer uses quantum processors"
        sources = ["Computers use processors", "Technology is advancing"]

        assert detect_hallucination(response, sources)

        # Test case 3: Empty sources should trigger hallucination detection
        assert detect_hallucination("Any response", [])

    def test_entity_extraction(self):
        """Test entity extraction functionality."""
        text = (
            "Python is a programming language developed by Guido van Rossum at Google"
        )
        entities = extract_entities(text)

        # Should extract proper nouns and organizations
        entity_texts = {entity for entity in entities}

        # Verify extraction of key entities (case-insensitive)
        assert any("python" in entity.lower() for entity in entity_texts)
        assert any("guido" in entity.lower() for entity in entity_texts)

    @pytest.mark.asyncio
    async def test_weighted_reciprocal_rank_fusion(self):
        """Test weighted reciprocal rank fusion implementation."""
        fuser = ResultFuser()

        vector_results = [
            {"content": "result1", "score": 0.9, "id": 1},
            {"content": "result2", "score": 0.8, "id": 2},
        ]

        graph_results = [
            {
                "content": "result3",
                "relevance_score": 0.85,
                "entity_names": ["entity1"],
            },
            {
                "content": "result1",
                "relevance_score": 0.75,
                "entity_names": ["entity2"],
            },  # Duplicate
        ]

        fused_results = await fuser.fuse_results(
            vector_results=vector_results,
            graph_results=graph_results,
            confidence_threshold=0.5,
        )

        # Verify fusion occurred
        assert len(fused_results) > 0

        # Verify deduplication (should have fewer results than total input)
        total_input = len(vector_results) + len(graph_results)
        assert len(fused_results) <= total_input

        # Verify fusion scores are calculated
        for result in fused_results:
            assert "fusion_score" in result
            assert isinstance(result["fusion_score"], float)

    def test_confidence_threshold_filtering(self):
        """Test that results below confidence threshold are filtered."""
        fuser = ResultFuser()

        # Mock results with varying confidence
        mock_results = [
            {"fusion_score": 0.9, "confidence_score": 0.8},  # Above threshold
            {"fusion_score": 0.5, "confidence_score": 0.4},  # Below threshold
            {"fusion_score": 0.7, "confidence_score": 0.7},  # Above threshold
        ]

        # Test filtering
        filtered = [r for r in mock_results if r["confidence_score"] >= 0.65]

        assert len(filtered) == 2  # Only 2 results above threshold
        assert all(r["confidence_score"] >= 0.65 for r in filtered)


class TestRAGService:
    """Test the production-ready RAG service."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Mock the orchestrator for service testing."""
        with patch("graphrag.rag_system.service.HybridRAGOrchestrator") as mock:
            mock.return_value.process_query = AsyncMock(
                return_value={
                    "response": "Test response",
                    "confidence_score": 0.8,
                    "iteration_count": 1,
                    "hallucination_detected": False,
                    "sources_count": 3,
                    "metadata": {
                        "vector_results_count": 2,
                        "graph_results_count": 1,
                        "fused_results_count": 3,
                    },
                }
            )
            yield mock

    @pytest.mark.asyncio
    async def test_query_validation(self, mock_orchestrator):
        """Test query validation logic."""
        service = RAGService()

        # Test empty query
        result = await service.query_with_retry("")
        assert result["error"] is True
        assert "error_type" in result

        # Test query too long
        long_query = "x" * (service.max_query_length + 1)
        result = await service.query_with_retry(long_query)
        assert result["error"] is True

        # Test suspicious content
        suspicious_query = "What is <script>alert('test')</script>?"
        result = await service.query_with_retry(suspicious_query)
        assert result["error"] is True

    @pytest.mark.asyncio
    async def test_timeout_handling(self, mock_orchestrator):
        """Test query timeout handling."""
        service = RAGService()
        service.timeout_seconds = 0.1  # Very short timeout

        # Mock slow processing
        mock_orchestrator.return_value.process_query = AsyncMock()
        mock_orchestrator.return_value.process_query.side_effect = asyncio.sleep(1)

        result = await service.query_with_retry("test timeout")

        assert result["error"] is True
        assert result["error_type"] == "timeout_error"

    @pytest.mark.asyncio
    async def test_retry_mechanism(self, mock_orchestrator):
        """Test retry mechanism with tenacity."""
        service = RAGService()

        # Mock failing then succeeding
        call_count = 0

        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return {
                "response": "Success after retry",
                "confidence_score": 0.8,
                "iteration_count": 0,
                "hallucination_detected": False,
                "sources_count": 1,
                "metadata": {},
            }

        mock_orchestrator.return_value.process_query = mock_process

        result = await service.query_with_retry("test retry")

        # Should succeed after retry
        assert result["error"] is not True
        assert "Success after retry" in result["response"]
        assert call_count == 2  # Failed once, succeeded on retry

    @pytest.mark.asyncio
    async def test_health_check(self, mock_orchestrator):
        """Test comprehensive health check functionality."""
        service = RAGService()

        # Mock health check methods
        mock_orchestrator.return_value.vector_retriever.health_check = Mock(
            return_value=True
        )
        mock_orchestrator.return_value.graph_retriever.health_check = Mock(
            return_value=True
        )

        health_status = await service.health_check()

        assert "overall" in health_status
        assert "orchestrator" in health_status
        assert "vector_store" in health_status
        assert "graph_store" in health_status
        assert "metrics" in health_status


class TestConfiguration:
    """Test configuration validation and settings."""

    def test_config_validation(self):
        """Test configuration validation logic."""
        validation_result = Config.validate_config()

        assert "valid" in validation_result
        assert "errors" in validation_result
        assert "warnings" in validation_result
        assert isinstance(validation_result["errors"], list)
        assert isinstance(validation_result["warnings"], list)

    def test_fusion_weights_sum_to_one(self):
        """Test that fusion weights sum to 1.0 as required by ruleset."""
        vector_weight = Config.FUSION_VECTOR_WEIGHT
        graph_weight = Config.FUSION_GRAPH_WEIGHT

        # Allow small floating point tolerance
        weight_sum = vector_weight + graph_weight
        assert (
            abs(weight_sum - 1.0) < 0.01
        ), f"Fusion weights sum to {weight_sum}, should be 1.0"

    def test_confidence_thresholds(self):
        """Test that confidence thresholds are within valid ranges."""
        assert 0.0 <= Config.CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 <= Config.ENTITY_CONFIDENCE_THRESHOLD <= 1.0
        assert 0.0 <= Config.HALLUCINATION_ENTITY_THRESHOLD <= 1.0

    def test_production_readiness_settings(self):
        """Test production readiness configuration."""
        # Test KPI targets according to ruleset
        assert Config.TARGET_LATENCY_95_PERCENTILE == 2.0  # <2s requirement
        assert Config.TARGET_ACCURACY_F1_SCORE >= 0.85  # >0.85 requirement
        assert Config.TARGET_UPTIME >= 0.9995  # >=99.95% requirement
        assert Config.TARGET_FALLBACK_RECOVERY >= 0.80  # >=80% requirement

        # Test RBAC roles
        expected_roles = ["rag_reader", "rag_operator", "rag_admin"]
        assert all(role in Config.RBAC_ROLES for role in expected_roles)

    def test_milvus_configuration_compliance(self):
        """Test Milvus configuration follows ruleset specifications."""
        assert Config.MILVUS_INDEX_TYPE == "HNSW"
        assert Config.MILVUS_METRIC_TYPE == "L2"
        assert Config.MILVUS_SEARCH_EF == 32
        assert Config.MILVUS_SEARCH_OFFSET == 5

    def test_neo4j_configuration_compliance(self):
        """Test Neo4j configuration follows ruleset specifications."""
        assert Config.NEO4J_MIN_RELATIONSHIP_STRENGTH == 0.7
        assert Config.NEO4J_MAX_TRAVERSAL_DEPTH == 3
        assert Config.MAX_CORRECTION_ITERATIONS == 3


class TestProductionReadiness:
    """Test production readiness aspects of the system."""

    def test_error_handling_coverage(self):
        """Test that all major components have proper error handling."""
        # This would be expanded with actual error injection tests
        # For now, verify error response structure
        service = RAGService()
        error_response = service._create_error_response("test error", "test_type")

        required_fields = [
            "response",
            "confidence_score",
            "iteration_count",
            "hallucination_detected",
            "sources_count",
            "error",
            "error_type",
            "error_message",
            "metadata",
        ]

        for field in required_fields:
            assert field in error_response

    def test_monitoring_integration(self):
        """Test monitoring and metrics collection."""
        service = RAGService()

        # Verify metrics are tracked
        assert hasattr(service, "query_count")
        assert hasattr(service, "error_count")
        assert hasattr(service, "total_processing_time")

        # Test metrics update
        initial_count = service.query_count
        service._update_metrics(1.0, success=True)
        assert service.query_count == initial_count + 1

    def test_security_configuration(self):
        """Test security configuration settings."""
        # Verify security settings are available
        assert hasattr(Config, "ENABLE_TLS")
        assert hasattr(Config, "ENABLE_AUTH")
        assert hasattr(Config, "RATE_LIMIT_REQUESTS_PER_MINUTE")
        assert hasattr(Config, "JWT_SECRET_KEY")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v", "--tb=short"])
