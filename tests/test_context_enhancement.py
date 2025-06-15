"""
Comprehensive tests for the Context Enhancement Engine

Tests all four discovery strategies:
1. Neo4j graph traversal
2. Semantic similarity analysis
3. Co-occurrence pattern detection
4. Temporal relationship analysis
"""

import asyncio
import os

# Import the modules to test
import sys
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import networkx as nx
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag.graph_enrichment.context_engine import (
    ContextEnhancementEngine,
    ContextGraph,
    EntityRelationship,
)


class TestContextEnhancementEngine:
    """Test suite for ContextEnhancementEngine."""

    @pytest.fixture
    def mock_neo4j_manager(self):
        """Mock Neo4j manager for testing."""
        mock_manager = Mock()
        mock_session = Mock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__.return_value = mock_session
        mock_context_manager.__exit__.return_value = None
        mock_manager.driver.session.return_value = mock_context_manager
        return mock_manager

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing."""
        mock_service = Mock()
        return mock_service

    @pytest.fixture
    def mock_similarity_model(self):
        """Mock sentence transformer model."""
        mock_model = Mock()
        # Return fixed embeddings for testing
        mock_model.encode.return_value = np.array(
            [
                [0.1, 0.2, 0.3],  # Entity 1
                [0.4, 0.5, 0.6],  # Entity 2
                [0.7, 0.8, 0.9],  # Entity 3
            ]
        )
        return mock_model

    @pytest.fixture
    async def context_engine(self, mock_neo4j_manager, mock_embedding_service):
        """Create context enhancement engine with mocked dependencies."""
        with patch("graphrag.graph_enrichment.context_engine.ModelCache") as mock_cache:
            mock_cache.return_value.get_embedding_model.return_value = Mock()

            engine = ContextEnhancementEngine(
                neo4j_manager=mock_neo4j_manager,
                embedding_service=mock_embedding_service,
                similarity_threshold=0.7,
                cooccurrence_window=3,
                temporal_window=3600,
            )
            return engine

    @pytest.mark.asyncio
    async def test_enhance_query_context_basic(self, context_engine):
        """Test basic context enhancement functionality."""
        query = "What is machine learning?"
        entities = ["machine learning", "AI", "algorithm"]

        # Mock the internal methods
        context_engine._find_neo4j_relationships = AsyncMock(
            return_value=[
                EntityRelationship(
                    "machine learning", "AI", "RELATED", 0.9, "neo4j_traversal"
                )
            ]
        )
        context_engine._find_semantic_relationships = AsyncMock(
            return_value=[
                EntityRelationship(
                    "AI", "algorithm", "SEMANTIC_SIMILAR", 0.8, "semantic_similarity"
                )
            ]
        )
        context_engine._find_cooccurrence_relationships = AsyncMock(return_value=[])
        context_engine._find_temporal_relationships = AsyncMock(return_value=[])

        result = await context_engine.enhance_query_context(query, entities)

        assert isinstance(result, ContextGraph)
        assert len(result.entities) >= 3
        assert len(result.relationships) == 2
        assert result.confidence_score > 0
        assert isinstance(result.graph, nx.Graph)

    @pytest.mark.asyncio
    async def test_neo4j_relationship_discovery(
        self, context_engine, mock_neo4j_manager
    ):
        """Test Neo4j graph traversal relationship discovery."""
        entities = ["Python", "programming"]

        # Mock Neo4j query results
        mock_records = [
            {
                "source": "Python",
                "target": "programming",
                "rels": [Mock(type="LANGUAGE_OF")],
                "distance": 1,
            },
            {
                "source": "Python",
                "target": "software",
                "rels": [Mock(type="USED_FOR")],
                "distance": 2,
            },
        ]

        mock_session = (
            mock_neo4j_manager.driver.session.return_value.__enter__.return_value
        )
        mock_session.run.return_value = mock_records

        relationships = await context_engine._find_neo4j_relationships(
            entities, max_depth=2
        )

        assert len(relationships) >= 2
        assert relationships[0].source == "Python"
        assert relationships[0].target == "programming"
        assert relationships[0].discovery_method == "neo4j_traversal"
        assert (
            relationships[0].confidence > relationships[1].confidence
        )  # Closer relationships have higher confidence

    @pytest.mark.asyncio
    async def test_semantic_relationship_discovery(
        self, context_engine, mock_similarity_model
    ):
        """Test semantic similarity relationship discovery."""
        entities = ["cat", "dog", "fish"]
        query = "What pets are similar?"

        # Mock the similarity model
        context_engine.similarity_model = mock_similarity_model

        # Mock embeddings that show cat and dog are similar
        mock_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # cat
                [0.9, 0.1, 0.0],  # dog (similar to cat)
                [0.0, 0.0, 1.0],  # fish (different)
            ]
        )
        mock_similarity_model.encode.return_value = mock_embeddings

        relationships = await context_engine._find_semantic_relationships(
            entities, query
        )

        # Should find cat-dog similarity (dot product = 0.9)
        assert len(relationships) >= 1
        semantic_rel = next(
            (r for r in relationships if r.relationship_type == "SEMANTIC_SIMILAR"),
            None,
        )
        assert semantic_rel is not None
        assert semantic_rel.confidence >= context_engine.similarity_threshold
        assert semantic_rel.discovery_method == "semantic_similarity"

    @pytest.mark.asyncio
    async def test_cooccurrence_relationship_discovery(self, context_engine):
        """Test co-occurrence pattern relationship discovery."""
        entities = ["apple", "fruit"]
        query = "Apple is a type of fruit that grows on trees"

        relationships = await context_engine._find_cooccurrence_relationships(
            entities, query
        )

        # Should find apple-fruit co-occurrence
        assert len(relationships) >= 1
        cooccur_rel = relationships[0]
        assert cooccur_rel.relationship_type == "CO_OCCURRENCE"
        assert cooccur_rel.discovery_method == "cooccurrence_analysis"
        assert "apple" in [cooccur_rel.source, cooccur_rel.target]
        assert "fruit" in [cooccur_rel.source, cooccur_rel.target]

    @pytest.mark.asyncio
    async def test_temporal_relationship_discovery(
        self, context_engine, mock_neo4j_manager
    ):
        """Test temporal relationship discovery."""
        entities = ["event1", "event2"]
        current_time = time.time()

        # Mock temporal Neo4j query results
        mock_records = [
            {
                "source": "event1",
                "target": "event2",
                "timestamp": current_time - 1800,
                "rel_type": "FOLLOWS",
            },
            {
                "source": "event1",
                "target": "event2",
                "timestamp": current_time - 1900,
                "rel_type": "PRECEDES",
            },
        ]

        mock_session = (
            mock_neo4j_manager.driver.session.return_value.__enter__.return_value
        )
        mock_session.run.return_value = mock_records

        relationships = await context_engine._find_temporal_relationships(entities)

        assert len(relationships) == 2
        for rel in relationships:
            assert rel.discovery_method == "temporal_analysis"
            assert rel.temporal_info is not None
            assert "timestamp" in rel.temporal_info
            assert rel.relationship_type.startswith("TEMPORAL_")

    def test_deduplicate_and_rank_relationships(self, context_engine):
        """Test relationship deduplication and ranking."""
        relationships = [
            EntityRelationship("A", "B", "TYPE1", 0.9, "neo4j_traversal"),
            EntityRelationship(
                "B", "A", "TYPE2", 0.8, "semantic_similarity"
            ),  # Same pair, different order
            EntityRelationship("A", "C", "TYPE3", 0.7, "cooccurrence_analysis"),
            EntityRelationship(
                "A", "B", "TYPE1", 0.6, "temporal_analysis"
            ),  # Duplicate pair
        ]

        unique_relationships = context_engine._deduplicate_and_rank_relationships(
            relationships
        )

        # Should have 2 unique relationships (A-B and A-C)
        assert len(unique_relationships) == 2

        # Should be sorted by confidence descending
        assert unique_relationships[0].confidence >= unique_relationships[1].confidence

        # A-B relationship should combine methods and boost confidence
        ab_rel = next(
            r for r in unique_relationships if {r.source, r.target} == {"A", "B"}
        )
        assert "neo4j_traversal" in ab_rel.discovery_method
        assert ab_rel.confidence > 0.9  # Boosted for multi-method discovery

    def test_build_context_graph(self, context_engine):
        """Test NetworkX context graph construction."""
        entities = ["A", "B", "C"]
        relationships = [
            EntityRelationship("A", "B", "RELATED", 0.9, "neo4j_traversal"),
            EntityRelationship("B", "C", "SIMILAR", 0.8, "semantic_similarity"),
        ]
        discovery_methods = {"neo4j_traversal", "semantic_similarity"}

        context_graph = context_engine._build_context_graph(
            entities, relationships, discovery_methods
        )

        assert isinstance(context_graph, ContextGraph)
        assert len(context_graph.entities) == 3
        assert len(context_graph.relationships) == 2
        assert context_graph.confidence_score == pytest.approx(
            0.85, rel=1e-10
        )  # Average of 0.9 and 0.8
        assert context_graph.discovery_methods == discovery_methods

        # Test NetworkX graph
        graph = context_graph.graph
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")

    @pytest.mark.asyncio
    async def test_extract_entities_from_query(self, context_engine):
        """Test entity extraction from query text."""
        query = 'What is "machine learning" and how does Python relate to Data Science?'

        entities = await context_engine._extract_entities_from_query(query)

        assert "machine learning" in entities
        assert "Python" in entities
        assert "Data Science" in entities
        assert len(entities) >= 3

    def test_cache_functionality(self, context_engine):
        """Test relationship caching mechanism."""
        cache_key = "test_key"
        test_relationships = [EntityRelationship("A", "B", "TEST", 0.9, "test_method")]

        # Test cache miss
        assert not context_engine._is_cache_valid(cache_key)

        # Add to cache
        context_engine._relationship_cache[cache_key] = test_relationships
        context_engine._cache_timestamps[cache_key] = time.time()

        # Test cache hit
        assert context_engine._is_cache_valid(cache_key)

        # Test cache expiry
        context_engine._cache_timestamps[cache_key] = time.time() - 400  # Expired
        assert not context_engine._is_cache_valid(cache_key)

    def test_get_context_summary(self, context_engine):
        """Test context graph summary generation."""
        # Create a test context graph
        entities = ["A", "B", "C", "D"]
        relationships = [
            EntityRelationship("A", "B", "RELATED", 0.9, "neo4j_traversal"),
            EntityRelationship("B", "C", "SIMILAR", 0.8, "semantic_similarity"),
            EntityRelationship("C", "D", "CO_OCCUR", 0.7, "cooccurrence_analysis"),
        ]
        discovery_methods = {
            "neo4j_traversal",
            "semantic_similarity",
            "cooccurrence_analysis",
        }

        context_graph = context_engine._build_context_graph(
            entities, relationships, discovery_methods
        )
        summary = context_engine.get_context_summary(context_graph)

        assert summary["total_entities"] == 4
        assert summary["total_relationships"] == 3
        assert summary["confidence_score"] == pytest.approx(0.8, rel=1e-2)
        assert len(summary["discovery_methods"]) == 3
        assert "graph_density" in summary
        assert "connected_components" in summary
        assert "relationship_types" in summary

    @pytest.mark.asyncio
    async def test_error_handling(self, context_engine):
        """Test error handling and graceful degradation."""
        query = "test query"
        entities = ["test"]

        # Mock methods to raise exceptions
        context_engine._find_neo4j_relationships = AsyncMock(
            side_effect=Exception("Neo4j error")
        )
        context_engine._find_semantic_relationships = AsyncMock(
            side_effect=Exception("Semantic error")
        )
        context_engine._find_cooccurrence_relationships = AsyncMock(return_value=[])
        context_engine._find_temporal_relationships = AsyncMock(return_value=[])

        # Should not raise exception and return minimal context graph
        result = await context_engine.enhance_query_context(query, entities)

        assert isinstance(result, ContextGraph)
        assert result.confidence_score >= 0.0  # Should be non-negative

    @pytest.mark.asyncio
    async def test_parallel_processing(self, context_engine):
        """Test that discovery methods run in parallel."""
        query = "test query"
        entities = ["test"]

        # Mock methods with delays to test parallelism
        async def slow_neo4j(*args, **kwargs):
            await asyncio.sleep(0.1)
            return []

        async def slow_semantic(*args, **kwargs):
            await asyncio.sleep(0.1)
            return []

        context_engine._find_neo4j_relationships = slow_neo4j
        context_engine._find_semantic_relationships = slow_semantic
        context_engine._find_cooccurrence_relationships = AsyncMock(return_value=[])
        context_engine._find_temporal_relationships = AsyncMock(return_value=[])

        start_time = time.time()
        await context_engine.enhance_query_context(query, entities)
        elapsed_time = time.time() - start_time

        # Should complete in less than 0.15 seconds if running in parallel
        # (vs 0.2+ seconds if running sequentially)
        assert elapsed_time < 0.15


@pytest.mark.asyncio
async def test_integration_with_real_networkx():
    """Integration test with real NetworkX operations."""
    # Create a context enhancement engine with minimal dependencies
    with patch("graphrag.graph_enrichment.context_engine.Neo4jManager"):
        with patch("graphrag.graph_enrichment.context_engine.EmbeddingService"):
            with patch("graphrag.graph_enrichment.context_engine.ModelCache"):
                engine = ContextEnhancementEngine()

                # Test real NetworkX graph operations
                entities = ["A", "B", "C", "D"]
                relationships = [
                    EntityRelationship("A", "B", "CONNECTED", 0.9, "test"),
                    EntityRelationship("B", "C", "CONNECTED", 0.8, "test"),
                    EntityRelationship("C", "D", "CONNECTED", 0.7, "test"),
                    EntityRelationship("A", "D", "CONNECTED", 0.6, "test"),
                ]

                context_graph = engine._build_context_graph(
                    entities, relationships, {"test"}
                )

                # Test NetworkX analysis
                assert nx.is_connected(context_graph.graph)
                assert nx.number_of_nodes(context_graph.graph) == 4
                assert nx.number_of_edges(context_graph.graph) == 4

                # Test graph algorithms
                shortest_path = nx.shortest_path(context_graph.graph, "A", "D")
                assert len(shortest_path) >= 2  # Direct or indirect path exists

                centrality = nx.degree_centrality(context_graph.graph)
                assert all(0 <= score <= 1 for score in centrality.values())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
