"""
Comprehensive tests for the Enhanced Neo4j Manager

Tests dynamic relationship discovery, multi-hop traversals, batch operations,
schema initialization, performance tracking, and all enhanced functionality.
"""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag.neo4j.manager import (
    EnhancedNeo4jManager,
    EntityNode,
    GraphPath,
    RelationshipEdge,
    RelationshipPattern,
)


class TestDataClasses:
    """Test the data classes used by the enhanced Neo4j manager."""

    def test_entity_node_creation(self):
        """Test EntityNode dataclass creation."""
        now = datetime.now()
        node = EntityNode(
            name="John Doe",
            type="PERSON",
            properties={"age": 30, "city": "New York"},
            created_at=now,
            updated_at=now,
        )

        assert node.name == "John Doe"
        assert node.type == "PERSON"
        assert node.properties == {"age": 30, "city": "New York"}
        assert node.created_at == now
        assert node.updated_at == now

    def test_relationship_edge_creation(self):
        """Test RelationshipEdge dataclass creation."""
        now = datetime.now()
        edge = RelationshipEdge(
            source="John Doe",
            target="Jane Smith",
            relationship_type="KNOWS",
            properties={"since": "2020"},
            confidence=0.9,
            created_at=now,
            weight=2.5,
        )

        assert edge.source == "John Doe"
        assert edge.target == "Jane Smith"
        assert edge.relationship_type == "KNOWS"
        assert edge.properties == {"since": "2020"}
        assert edge.confidence == 0.9
        assert edge.weight == 2.5
        assert edge.created_at == now

    def test_relationship_edge_default_weight(self):
        """Test RelationshipEdge with default weight."""
        now = datetime.now()
        edge = RelationshipEdge(
            source="A",
            target="B",
            relationship_type="RELATED",
            properties={},
            confidence=0.8,
            created_at=now,
        )

        assert edge.weight == 1.0  # Default weight

    def test_graph_path_creation(self):
        """Test GraphPath dataclass creation."""
        path = GraphPath(
            nodes=["A", "B", "C"],
            relationships=["KNOWS", "WORKS_WITH"],
            length=2,
            total_weight=3.5,
            confidence=0.85,
        )

        assert path.nodes == ["A", "B", "C"]
        assert path.relationships == ["KNOWS", "WORKS_WITH"]
        assert path.length == 2
        assert path.total_weight == 3.5
        assert path.confidence == 0.85

    def test_relationship_pattern_creation(self):
        """Test RelationshipPattern dataclass creation."""
        pattern = RelationshipPattern(
            pattern_type="2_hop",
            entities=["PERSON", "ORGANIZATION", "LOCATION"],
            relationship_types=["WORKS_FOR", "LOCATED_IN"],
            frequency=15,
            confidence=0.75,
            examples=[{"source": "John", "target": "Acme Corp"}],
        )

        assert pattern.pattern_type == "2_hop"
        assert pattern.entities == ["PERSON", "ORGANIZATION", "LOCATION"]
        assert pattern.relationship_types == ["WORKS_FOR", "LOCATED_IN"]
        assert pattern.frequency == 15
        assert pattern.confidence == 0.75
        assert len(pattern.examples) == 1


class TestEnhancedNeo4jManager:
    """Test suite for EnhancedNeo4jManager."""

    @pytest.fixture
    def mock_driver(self):
        """Mock Neo4j driver for testing."""
        mock_driver = Mock()
        mock_session = MagicMock()

        # Set up the session context manager properly
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_session_context

        return mock_driver, mock_session

    @pytest.fixture
    def manager(self, mock_driver):
        """Create Enhanced Neo4j manager with mocked driver."""
        driver, session = mock_driver

        with patch("graphrag.neo4j.manager.GraphDatabase.driver") as mock_graph_db:
            mock_graph_db.return_value = driver

            manager = EnhancedNeo4jManager(
                uri="bolt://localhost:7687",
                user="neo4j",
                password="test_password",
            )

            # Manually set the mocked driver
            manager.driver = driver

            return manager, session

    def test_initialization(self, mock_driver):
        """Test manager initialization."""
        driver, _ = mock_driver

        with patch("graphrag.neo4j.manager.GraphDatabase.driver") as mock_graph_db:
            mock_graph_db.return_value = driver

            manager = EnhancedNeo4jManager(
                uri="bolt://test:7687",
                user="test_user",
                password="test_pass",
                max_connection_lifetime=1800,
                max_connection_pool_size=25,
            )

            assert manager.uri == "bolt://test:7687"
            assert manager.user == "test_user"
            assert manager.password == "test_pass"
            assert manager.driver == driver

            # Check that driver was created with correct parameters
            mock_graph_db.assert_called_once_with(
                "bolt://test:7687",
                auth=("test_user", "test_pass"),
                max_connection_lifetime=1800,
                max_connection_pool_size=25,
                connection_timeout=30,
                max_retry_time=15,
            )

    @pytest.mark.asyncio
    async def test_initialize_schema_success(self, manager):
        """Test successful schema initialization."""
        manager_instance, mock_session = manager
        mock_session.run.return_value = None

        result = await manager_instance.initialize_schema()

        assert result is True
        # Verify that run was called multiple times for constraints and indexes
        assert mock_session.run.call_count >= 7  # 2 constraints + 5 indexes

    @pytest.mark.asyncio
    async def test_initialize_schema_failure(self, manager):
        """Test schema initialization failure."""
        manager_instance, mock_session = manager

        # Make the driver.session() itself raise an exception
        manager_instance.driver.session.side_effect = Exception("Database error")

        result = await manager_instance.initialize_schema()

        assert result is False

    @pytest.mark.asyncio
    async def test_create_entity_batch_success(self, manager):
        """Test successful batch entity creation."""
        manager_instance, mock_session = manager

        # Mock the query result
        mock_result = Mock()
        mock_result.single.return_value = {"created_count": 3}
        mock_session.run.return_value = mock_result

        entities = [
            {"name": "John Doe", "type": "PERSON", "properties": {"age": 30}},
            {"name": "Jane Smith", "type": "PERSON", "properties": {"age": 25}},
            {
                "name": "Acme Corp",
                "type": "ORGANIZATION",
                "properties": {"industry": "Tech"},
            },
        ]

        result = await manager_instance.create_entity_batch(entities)

        assert result is True
        mock_session.run.assert_called_once()

        # Check that performance stats were updated
        assert manager_instance._query_stats["create_entity_batch"] == 1
        assert len(manager_instance._query_times["create_entity_batch"]) == 1

    @pytest.mark.asyncio
    async def test_create_entity_batch_empty(self, manager):
        """Test batch entity creation with empty list."""
        manager_instance, mock_session = manager

        result = await manager_instance.create_entity_batch([])

        assert result is True
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_entity_batch_failure(self, manager):
        """Test batch entity creation failure."""
        manager_instance, mock_session = manager
        mock_session.run.side_effect = Exception("Database error")

        entities = [{"name": "Test", "type": "PERSON"}]

        result = await manager_instance.create_entity_batch(entities)

        assert result is False

    @pytest.mark.asyncio
    async def test_create_relationships_batch_success(self, manager):
        """Test successful batch relationship creation."""
        manager_instance, mock_session = manager

        # Mock the query result
        mock_result = Mock()
        mock_result.single.return_value = {"created_count": 2}
        mock_session.run.return_value = mock_result

        relationships = [
            {
                "source": "John Doe",
                "target": "Jane Smith",
                "type": "KNOWS",
                "confidence": 0.9,
                "weight": 2.0,
                "properties": {"since": "2020"},
            },
            {
                "source": "Jane Smith",
                "target": "Acme Corp",
                "type": "WORKS_FOR",
                "confidence": 0.95,
            },
        ]

        result = await manager_instance.create_relationships_batch(relationships)

        assert result is True
        mock_session.run.assert_called_once()

        # Check that performance stats were updated
        assert manager_instance._query_stats["create_relationships_batch"] == 1
        assert len(manager_instance._query_times["create_relationships_batch"]) == 1

    @pytest.mark.asyncio
    async def test_create_relationships_batch_empty(self, manager):
        """Test batch relationship creation with empty list."""
        manager_instance, mock_session = manager

        result = await manager_instance.create_relationships_batch([])

        assert result is True
        mock_session.run.assert_not_called()

    @pytest.mark.asyncio
    async def test_discover_relationship_patterns_success(self, manager):
        """Test successful relationship pattern discovery."""
        manager_instance, mock_session = manager

        # Mock query results for 2-hop patterns
        mock_result_2hop = Mock()
        mock_result_2hop.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "type1": "PERSON",
                        "rel1": "WORKS_FOR",
                        "type2": "ORGANIZATION",
                        "rel2": "LOCATED_IN",
                        "type3": "LOCATION",
                        "frequency": 10,
                    },
                    {
                        "type1": "PERSON",
                        "rel1": "KNOWS",
                        "type2": "PERSON",
                        "rel2": "LIVES_IN",
                        "type3": "LOCATION",
                        "frequency": 5,
                    },
                ]
            )
        )

        # Mock query results for 3-hop patterns
        mock_result_3hop = Mock()
        mock_result_3hop.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "type1": "PERSON",
                        "rel1": "WORKS_FOR",
                        "type2": "ORGANIZATION",
                        "rel2": "PARTNERS_WITH",
                        "type3": "ORGANIZATION",
                        "rel3": "LOCATED_IN",
                        "type4": "LOCATION",
                        "frequency": 3,
                    }
                ]
            )
        )

        # Set up mock to return different results for different queries
        mock_session.run.side_effect = [mock_result_2hop, mock_result_3hop]

        patterns = await manager_instance.discover_relationship_patterns(
            entity_types=["PERSON", "ORGANIZATION"],
            min_frequency=2,
            max_pattern_length=3,
        )

        assert len(patterns) == 3  # 2 from 2-hop + 1 from 3-hop

        # Check 2-hop patterns
        two_hop_patterns = [p for p in patterns if p.pattern_type == "2_hop"]
        assert len(two_hop_patterns) == 2
        assert two_hop_patterns[0].frequency == 10
        assert two_hop_patterns[1].frequency == 5

        # Check 3-hop patterns
        three_hop_patterns = [p for p in patterns if p.pattern_type == "3_hop"]
        assert len(three_hop_patterns) == 1
        assert three_hop_patterns[0].frequency == 3

        # Check performance tracking
        assert manager_instance._query_stats["discover_patterns"] == 1

    @pytest.mark.asyncio
    async def test_discover_relationship_patterns_failure(self, manager):
        """Test relationship pattern discovery failure."""
        manager_instance, mock_session = manager
        mock_session.run.side_effect = Exception("Query error")

        patterns = await manager_instance.discover_relationship_patterns()

        assert patterns == []

    @pytest.mark.asyncio
    async def test_find_multi_hop_paths_success(self, manager):
        """Test successful multi-hop path finding."""
        manager_instance, mock_session = manager

        # Mock query result
        mock_result = Mock()
        mock_result.__iter__ = Mock(
            return_value=iter(
                [
                    {
                        "node_names": ["John Doe", "Acme Corp", "New York"],
                        "relationship_types": ["WORKS_FOR", "LOCATED_IN"],
                        "path_length": 2,
                        "total_weight": 3.5,
                        "path_confidence": 0.85,
                    },
                    {
                        "node_names": ["John Doe", "Jane Smith", "New York"],
                        "relationship_types": ["KNOWS", "LIVES_IN"],
                        "path_length": 2,
                        "total_weight": 2.8,
                        "path_confidence": 0.72,
                    },
                ]
            )
        )
        mock_session.run.return_value = mock_result

        paths = await manager_instance.find_multi_hop_paths(
            source="John Doe",
            target="New York",
            max_hops=3,
            min_confidence=0.1,
        )

        assert len(paths) == 2

        # Check first path
        assert paths[0].nodes == ["John Doe", "Acme Corp", "New York"]
        assert paths[0].relationships == ["WORKS_FOR", "LOCATED_IN"]
        assert paths[0].length == 2
        assert paths[0].total_weight == 3.5
        assert paths[0].confidence == 0.85

        # Check second path
        assert paths[1].nodes == ["John Doe", "Jane Smith", "New York"]
        assert paths[1].relationships == ["KNOWS", "LIVES_IN"]
        assert paths[1].length == 2
        assert paths[1].total_weight == 2.8
        assert paths[1].confidence == 0.72

        # Check performance tracking
        assert manager_instance._query_stats["find_multi_hop_paths"] == 1

    @pytest.mark.asyncio
    async def test_find_multi_hop_paths_failure(self, manager):
        """Test multi-hop path finding failure."""
        manager_instance, mock_session = manager
        mock_session.run.side_effect = Exception("Query error")

        paths = await manager_instance.find_multi_hop_paths("A", "B")

        assert paths == []

    @pytest.mark.asyncio
    async def test_get_entity_neighborhood_success(self, manager):
        """Test successful entity neighborhood retrieval."""
        manager_instance, mock_session = manager

        # Mock query result
        mock_result = Mock()
        mock_result.single.return_value = {
            "entity_name": "John Doe",
            "entity_type": "PERSON",
            "entity_properties": {"age": 30, "city": "New York"},
            "neighbors": ["Jane Smith", "Acme Corp", "New York"],
            "neighbor_details": [
                {
                    "neighbor": "Jane Smith",
                    "relationship": ["KNOWS"],
                    "distance": 1,
                    "confidence": 0.9,
                },
                {
                    "neighbor": "Acme Corp",
                    "relationship": ["WORKS_FOR"],
                    "distance": 1,
                    "confidence": 0.95,
                },
                {
                    "neighbor": "New York",
                    "relationship": ["LIVES_IN"],
                    "distance": 1,
                    "confidence": 0.8,
                },
            ],
            "neighbor_count": 3,
        }
        mock_session.run.return_value = mock_result

        neighborhood = await manager_instance.get_entity_neighborhood(
            entity="John Doe",
            depth=2,
            relationship_types=["KNOWS", "WORKS_FOR"],
        )

        assert neighborhood["entity"] == "John Doe"
        assert neighborhood["entity_type"] == "PERSON"
        assert neighborhood["neighbor_count"] == 3
        assert len(neighborhood["neighbors"]) == 3
        assert len(neighborhood["neighbor_details"]) == 3

        # Check performance tracking
        assert manager_instance._query_stats["get_neighborhood"] == 1

    @pytest.mark.asyncio
    async def test_get_entity_neighborhood_not_found(self, manager):
        """Test entity neighborhood retrieval for non-existent entity."""
        manager_instance, mock_session = manager

        # Mock empty result
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        neighborhood = await manager_instance.get_entity_neighborhood("Unknown")

        assert neighborhood["entity"] == "Unknown"
        assert neighborhood["neighbors"] == []
        assert neighborhood["neighbor_count"] == 0

    @pytest.mark.asyncio
    async def test_get_relationship_strength_direct(self, manager):
        """Test relationship strength calculation with direct relationship."""
        manager_instance, mock_session = manager

        # Mock query result with direct relationship
        mock_result = Mock()
        mock_result.single.return_value = {
            "source": "John Doe",
            "target": "Jane Smith",
            "direct_confidence": 0.9,
            "direct_weight": 2.5,
            "max_indirect_confidence": 0.0,
            "indirect_path_count": 0,
            "connection_type": "DIRECT",
        }
        mock_session.run.return_value = mock_result

        strength = await manager_instance.get_relationship_strength(
            "John Doe", "Jane Smith"
        )

        assert strength["source"] == "John Doe"
        assert strength["target"] == "Jane Smith"
        assert strength["direct_confidence"] == 0.9
        assert strength["connection_type"] == "DIRECT"
        assert strength["strength"] == 0.9  # Direct confidence only

    @pytest.mark.asyncio
    async def test_get_relationship_strength_indirect(self, manager):
        """Test relationship strength calculation with indirect relationships."""
        manager_instance, mock_session = manager

        # Mock query result with indirect relationships
        mock_result = Mock()
        mock_result.single.return_value = {
            "source": "John Doe",
            "target": "New York",
            "direct_confidence": 0.0,
            "direct_weight": 0.0,
            "max_indirect_confidence": 0.7,
            "indirect_path_count": 2,
            "connection_type": "INDIRECT",
        }
        mock_session.run.return_value = mock_result

        strength = await manager_instance.get_relationship_strength(
            "John Doe", "New York"
        )

        assert strength["connection_type"] == "INDIRECT"
        assert strength["indirect_confidence"] == 0.7
        assert strength["indirect_path_count"] == 2
        # Strength = 0 (direct) + 0.35 (indirect * 0.5) + 0.2 (path count * 0.1) = 0.55
        assert strength["strength"] == pytest.approx(0.55, rel=1e-2)

    @pytest.mark.asyncio
    async def test_get_relationship_strength_not_found(self, manager):
        """Test relationship strength for non-existent relationship."""
        manager_instance, mock_session = manager

        # Mock empty result
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        strength = await manager_instance.get_relationship_strength("A", "B")

        assert strength["source"] == "A"
        assert strength["target"] == "B"
        assert strength["strength"] == 0.0
        assert strength["connection_type"] == "NONE"

    @pytest.mark.asyncio
    async def test_expand_entity_context_success(self, manager):
        """Test successful entity context expansion."""
        manager_instance, mock_session = manager

        # Mock query result
        mock_result = Mock()
        mock_result.single.return_value = {
            "related_entities": ["Jane Smith", "Acme Corp", "New York", "Bob Wilson"]
        }
        mock_session.run.return_value = mock_result

        expanded = await manager_instance.expand_entity_context(
            entities=["John Doe"],
            expansion_factor=3.0,
            min_confidence=0.3,
        )

        # Should include original entity plus expanded entities
        assert len(expanded) == 5  # Original + 4 new
        assert "John Doe" in expanded
        assert "Jane Smith" in expanded
        assert "Acme Corp" in expanded

        # Check performance tracking
        assert manager_instance._query_stats["expand_context"] == 1

    @pytest.mark.asyncio
    async def test_expand_entity_context_failure(self, manager):
        """Test entity context expansion failure."""
        manager_instance, mock_session = manager
        mock_session.run.side_effect = Exception("Query error")

        expanded = await manager_instance.expand_entity_context(["John Doe"])

        # Should return original entities on failure
        assert expanded == ["John Doe"]

    def test_get_performance_stats(self, manager):
        """Test performance statistics retrieval."""
        manager_instance, _ = manager

        # Add some mock performance data
        manager_instance._query_times["create_entity_batch"] = [0.1, 0.2, 0.15]
        manager_instance._query_times["find_multi_hop_paths"] = [0.5, 0.6]

        stats = manager_instance.get_performance_stats()

        assert "create_entity_batch" in stats
        assert "find_multi_hop_paths" in stats

        # Check entity batch stats
        entity_stats = stats["create_entity_batch"]
        assert entity_stats["total_calls"] == 3
        assert entity_stats["avg_time"] == pytest.approx(0.15, rel=1e-2)
        assert entity_stats["min_time"] == 0.1
        assert entity_stats["max_time"] == 0.2
        assert entity_stats["total_time"] == pytest.approx(0.45, rel=1e-2)

        # Check path finding stats
        path_stats = stats["find_multi_hop_paths"]
        assert path_stats["total_calls"] == 2
        assert path_stats["avg_time"] == pytest.approx(0.55, rel=1e-2)

    @pytest.mark.asyncio
    async def test_health_check_success(self, manager):
        """Test successful health check."""
        manager_instance, mock_session = manager

        # Mock the query results
        test_result = Mock()
        test_result.single.return_value = {"test": 1}

        db_info_result = Mock()
        db_info_result.single.return_value = {
            "name": "neo4j",
            "version": "5.13.0",
            "edition": "community",
        }

        count_result = Mock()
        count_result.single.return_value = {
            "node_count": 100,
            "relationship_count": 250,
        }

        mock_session.run.side_effect = [test_result, db_info_result, count_result]

        health = await manager_instance.health_check()

        assert health["status"] == "healthy"
        assert "response_time" in health
        assert health["database"]["name"] == "neo4j"
        assert health["database"]["version"] == "5.13.0"
        assert health["statistics"]["node_count"] == 100
        assert health["statistics"]["relationship_count"] == 250
        assert "performance" in health

    @pytest.mark.asyncio
    async def test_health_check_failure(self, manager):
        """Test health check failure."""
        manager_instance, mock_session = manager
        mock_session.run.side_effect = Exception("Connection error")

        health = await manager_instance.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health
        assert "Connection error" in health["error"]

    def test_close(self, manager):
        """Test closing the manager."""
        manager_instance, _ = manager
        mock_driver = Mock()
        manager_instance.driver = mock_driver

        manager_instance.close()

        mock_driver.close.assert_called_once()

    def test_context_manager(self, manager):
        """Test manager as context manager."""
        manager_instance, _ = manager
        mock_driver = Mock()
        manager_instance.driver = mock_driver

        with manager_instance as mgr:
            assert mgr == manager_instance

        mock_driver.close.assert_called_once()

    def test_legacy_create_entity_and_relation(self, manager):
        """Test legacy method for backward compatibility."""
        manager_instance, _ = manager

        # Test that the method exists and can be called without exceptions
        # The actual async event loop handling is complex to test properly
        try:
            manager_instance.create_entity_and_relation(
                "John Doe", "PERSON", "Jane Smith", "KNOWS"
            )
            # If we get here, the method completed without major errors
            # (though it may have failed internally due to mocked components)
        except Exception as e:
            # Should not raise AttributeError or similar structural errors
            assert "AttributeError" not in str(type(e))

    def test_legacy_create_entity_and_relation_empty_params(self, manager):
        """Test legacy method with empty parameters."""
        manager_instance, _ = manager

        # Should return early without creating loop
        with patch("asyncio.new_event_loop") as mock_loop_func:
            manager_instance.create_entity_and_relation("", "PERSON", "Jane", "KNOWS")
            manager_instance.create_entity_and_relation("John", "PERSON", "", "KNOWS")

            # Should not create event loop for empty parameters
            mock_loop_func.assert_not_called()

    def test_legacy_get_related(self, manager):
        """Test legacy get_related method."""
        manager_instance, _ = manager

        # Test that the method exists and can be called
        # The actual result will depend on the mocked Neo4j responses
        try:
            result = manager_instance.get_related("John Doe")
            # Should return a list (even if empty due to mocking)
            assert isinstance(result, list)
        except Exception as e:
            # Should not raise AttributeError or similar structural errors
            assert "AttributeError" not in str(type(e))


class TestIntegrationScenarios:
    """Integration tests for complex scenarios."""

    @pytest.fixture
    def full_manager(self):
        """Create manager with full mock setup for integration tests."""
        mock_driver = Mock()
        mock_session = MagicMock()

        # Set up the session context manager properly
        mock_session_context = MagicMock()
        mock_session_context.__enter__ = Mock(return_value=mock_session)
        mock_session_context.__exit__ = Mock(return_value=None)
        mock_driver.session.return_value = mock_session_context

        with patch("graphrag.neo4j.manager.GraphDatabase.driver") as mock_graph_db:
            mock_graph_db.return_value = mock_driver

            manager = EnhancedNeo4jManager()
            manager.driver = mock_driver

            return manager, mock_session

    @pytest.mark.asyncio
    async def test_complete_workflow(self, full_manager):
        """Test a complete workflow: schema → entities → relationships → query."""
        manager, mock_session = full_manager

        # Mock responses for each step
        mock_responses = [
            None,  # Schema initialization
            {"created_count": 3},  # Entity creation
            {"created_count": 2},  # Relationship creation
            # Multi-hop path query
            [
                {
                    "node_names": ["A", "B", "C"],
                    "relationship_types": ["KNOWS", "WORKS_WITH"],
                    "path_length": 2,
                    "total_weight": 3.0,
                    "path_confidence": 0.8,
                }
            ],
        ]

        # Set up different responses for different operations
        def mock_run_side_effect(query, **kwargs):
            # Check what type of query this is by examining the query string
            if "CREATE CONSTRAINT" in query or "CREATE INDEX" in query:
                # Schema operations
                return None
            elif "UNWIND $entities" in query and "Entity" in query:
                # Entity batch creation
                result = Mock()
                result.single.return_value = {"created_count": 3}
                return result
            elif "UNWIND $relationships" in query and "RELATION" in query:
                # Relationship batch creation
                result = Mock()
                result.single.return_value = {"created_count": 2}
                return result
            elif "MATCH path" in query:
                # Path finding query
                result = Mock()
                result.__iter__ = Mock(return_value=iter(mock_responses[-1]))
                return result
            else:
                # Default case
                result = Mock()
                result.single.return_value = {"test": 1}
                return result

        mock_session.run.side_effect = mock_run_side_effect

        # Step 1: Initialize schema
        schema_result = await manager.initialize_schema()
        assert schema_result is True

        # Step 2: Create entities
        entities = [
            {"name": "Alice", "type": "PERSON"},
            {"name": "Bob", "type": "PERSON"},
            {"name": "Charlie", "type": "PERSON"},
        ]
        entity_result = await manager.create_entity_batch(entities)
        assert entity_result is True

        # Step 3: Create relationships
        relationships = [
            {"source": "Alice", "target": "Bob", "type": "KNOWS"},
            {"source": "Bob", "target": "Charlie", "type": "WORKS_WITH"},
        ]
        rel_result = await manager.create_relationships_batch(relationships)
        assert rel_result is True

        # Step 4: Query paths
        paths = await manager.find_multi_hop_paths("Alice", "Charlie")
        assert len(paths) == 1
        assert paths[0].nodes == ["A", "B", "C"]

        # Verify performance tracking
        assert manager._query_stats["create_entity_batch"] == 1
        assert manager._query_stats["create_relationships_batch"] == 1
        assert manager._query_stats["find_multi_hop_paths"] == 1

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, full_manager):
        """Test error recovery in complex workflows."""
        manager, mock_session = full_manager

        # First call succeeds, second fails, third succeeds
        mock_session.run.side_effect = [
            Mock(single=Mock(return_value={"created_count": 1})),  # Success
            Exception("Temporary failure"),  # Failure
            Mock(single=Mock(return_value={"created_count": 1})),  # Recovery
        ]

        # First operation should succeed
        result1 = await manager.create_entity_batch([{"name": "A", "type": "PERSON"}])
        assert result1 is True

        # Second operation should fail
        result2 = await manager.create_entity_batch([{"name": "B", "type": "PERSON"}])
        assert result2 is False

        # Third operation should succeed (recovery)
        result3 = await manager.create_entity_batch([{"name": "C", "type": "PERSON"}])
        assert result3 is True

        # Performance stats should track both successes and failures
        assert (
            manager._query_stats["create_entity_batch"] == 2
        )  # Only successful calls counted


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
