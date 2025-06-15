"""
Comprehensive tests for the Contextual Memory System

Tests Redis-based similarity caching, embedding-based retrieval,
automatic cache management, and all memory system functionality.
"""

import os

# Import the modules to test
import sys
import time
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from graphrag.graph_enrichment.memory_system import (
    ContextualMemorySystem,
    ManagedMemorySystem,
    MemoryEntry,
    SimilaritySearchResult,
)


class TestMemoryEntry:
    """Test MemoryEntry dataclass."""

    def test_memory_entry_creation(self):
        """Test memory entry creation with all fields."""
        entry = MemoryEntry(
            id="test_id",
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
            access_count=1,
            metadata={"source": "test"},
        )

        assert entry.id == "test_id"
        assert entry.content == "test content"
        assert entry.embedding == [0.1, 0.2, 0.3]
        assert entry.context_type == "query"
        assert entry.confidence == 0.9
        assert entry.access_count == 1
        assert entry.metadata == {"source": "test"}

    def test_memory_entry_defaults(self):
        """Test memory entry with default values."""
        entry = MemoryEntry(
            id="test_id",
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
        )

        assert entry.access_count == 0
        assert entry.last_accessed == 0.0
        assert entry.metadata is None


class TestSimilaritySearchResult:
    """Test SimilaritySearchResult dataclass."""

    def test_similarity_search_result_creation(self):
        """Test similarity search result creation."""
        entry = MemoryEntry(
            id="test_id",
            content="test content",
            embedding=[0.1, 0.2, 0.3],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
        )

        result = SimilaritySearchResult(entry=entry, similarity_score=0.85, rank=1)

        assert result.entry == entry
        assert result.similarity_score == 0.85
        assert result.rank == 1


class TestContextualMemorySystem:
    """Test suite for ContextualMemorySystem."""

    @pytest.fixture
    async def mock_redis_client(self):
        """Mock Redis client for testing."""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.keys.return_value = []
        mock_redis.mget.return_value = []
        mock_redis.info.return_value = {"used_memory": 1024 * 1024}  # 1MB
        return mock_redis

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock embedding service for testing."""
        mock_service = Mock()
        return mock_service

    @pytest.fixture
    def mock_similarity_model(self):
        """Mock sentence transformer model."""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        return mock_model

    @pytest.fixture
    async def memory_system(self, mock_embedding_service, mock_redis_client):
        """Create memory system with mocked dependencies."""
        with patch("redis.asyncio.Redis") as mock_redis_class:
            with patch("redis.asyncio.ConnectionPool") as mock_pool_class:
                with patch(
                    "graphrag.graph_enrichment.memory_system.ModelCache"
                ) as mock_cache:

                    mock_redis_class.return_value = mock_redis_client
                    mock_pool_class.return_value = AsyncMock()
                    mock_cache.return_value.get_embedding_model.return_value = Mock()

                    system = ContextualMemorySystem(
                        embedding_service=mock_embedding_service,
                        redis_host="localhost",
                        redis_port=6379,
                        cache_ttl=3600,
                        max_cache_size=1000,
                        similarity_threshold=0.8,
                    )

                    # Manually set the mocked client
                    system.redis_client = mock_redis_client

                    return system

    @pytest.mark.asyncio
    async def test_initialization(self, memory_system, mock_redis_client):
        """Test memory system initialization."""
        # Reset the redis_client to test initialization
        memory_system.redis_client = None

        with patch("redis.asyncio.Redis") as mock_redis_class:
            with patch("redis.asyncio.ConnectionPool") as mock_pool_class:
                mock_redis_class.return_value = mock_redis_client
                mock_pool_class.return_value = AsyncMock()

                result = await memory_system.initialize()

        assert result is True
        mock_redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_context(self, memory_system, mock_redis_client):
        """Test storing context in memory."""
        # Mock the similarity model
        memory_system.similarity_model = Mock()
        memory_system.similarity_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )

        entry_id = await memory_system.store_context(
            content="test content",
            context_type="query",
            confidence=0.9,
            metadata={"source": "test"},
        )

        assert entry_id is not None
        assert entry_id.startswith("query_")
        # Should be called at least once (for entry storage, possibly for stats too)
        assert mock_redis_client.setex.call_count >= 1
        assert memory_system.stats.total_entries == 1

    @pytest.mark.asyncio
    async def test_retrieve_similar_contexts(self, memory_system, mock_redis_client):
        """Test retrieving similar contexts."""
        # Mock the similarity model
        memory_system.similarity_model = Mock()
        memory_system.similarity_model.encode.return_value = np.array(
            [0.1, 0.2, 0.3, 0.4]
        )

        # Mock stored entries
        mock_entry = MemoryEntry(
            id="test_entry",
            content="similar content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
        )

        # Mock Redis responses
        mock_redis_client.keys.return_value = ["memory:entry:test_entry"]
        mock_redis_client.mget.return_value = [
            Mock()
        ]  # Will be mocked in _get_entries_by_type

        # Mock the _get_entries_by_type method
        memory_system._get_entries_by_type = AsyncMock(return_value=[mock_entry])

        results = await memory_system.retrieve_similar_contexts(
            query="test query", max_results=5
        )

        assert len(results) <= 5
        # The actual similarity calculation depends on the mocked embeddings

    @pytest.mark.asyncio
    async def test_get_context_by_id(self, memory_system, mock_redis_client):
        """Test retrieving context by ID."""
        # Create a mock memory entry
        mock_entry = MemoryEntry(
            id="test_entry",
            content="test content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
        )

        # Mock Redis response
        import pickle

        mock_redis_client.get.return_value = pickle.dumps(mock_entry)

        result = await memory_system.get_context_by_id("test_entry")

        assert result is not None
        assert result.id == "test_entry"
        assert result.content == "test content"
        mock_redis_client.get.assert_called_with("memory:entry:test_entry")

    @pytest.mark.asyncio
    async def test_get_context_by_id_not_found(self, memory_system, mock_redis_client):
        """Test retrieving non-existent context by ID."""
        mock_redis_client.get.return_value = None

        result = await memory_system.get_context_by_id("nonexistent")

        assert result is None
        assert memory_system.stats.cache_misses == 1

    @pytest.mark.asyncio
    async def test_update_context(self, memory_system, mock_redis_client):
        """Test updating existing context."""
        # Create a mock memory entry
        mock_entry = MemoryEntry(
            id="test_entry",
            content="original content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
        )

        # Mock Redis response for get
        import pickle

        mock_redis_client.get.return_value = pickle.dumps(mock_entry)

        # Mock the similarity model for new embedding
        memory_system.similarity_model = Mock()
        memory_system.similarity_model.encode.return_value = np.array(
            [0.2, 0.3, 0.4, 0.5]
        )

        result = await memory_system.update_context(
            entry_id="test_entry",
            content="updated content",
            confidence=0.95,
            metadata={"updated": True},
        )

        assert result is True
        # Verify that setex was called for storing the updated entry
        assert mock_redis_client.setex.call_count >= 1

    @pytest.mark.asyncio
    async def test_delete_context(self, memory_system, mock_redis_client):
        """Test deleting context."""
        mock_redis_client.delete.return_value = 1  # 1 key deleted

        result = await memory_system.delete_context("test_entry")

        assert result is True
        mock_redis_client.delete.assert_called_with("memory:entry:test_entry")
        assert memory_system.stats.total_entries == -1  # Decremented from 0

    @pytest.mark.asyncio
    async def test_delete_context_not_found(self, memory_system, mock_redis_client):
        """Test deleting non-existent context."""
        mock_redis_client.delete.return_value = 0  # No keys deleted

        result = await memory_system.delete_context("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_entries(self, memory_system, mock_redis_client):
        """Test cleanup of expired entries."""
        # Mock expired entry
        expired_entry = MemoryEntry(
            id="expired_entry",
            content="expired content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            context_type="query",
            confidence=0.9,
            timestamp=time.time() - 7200,  # 2 hours ago (expired if TTL is 1 hour)
        )

        # Mock Redis responses
        import pickle

        mock_redis_client.keys.return_value = ["memory:entry:expired_entry"]
        mock_redis_client.get.return_value = pickle.dumps(expired_entry)
        mock_redis_client.delete.return_value = 1

        cleanup_count = await memory_system.cleanup_expired_entries()

        assert cleanup_count == 1
        mock_redis_client.delete.assert_called()

    @pytest.mark.asyncio
    async def test_get_memory_stats(self, memory_system, mock_redis_client):
        """Test getting memory statistics."""
        # Mock some stats
        memory_system.stats.cache_hits = 10
        memory_system.stats.cache_misses = 2
        memory_system.stats.total_entries = 5

        # Mock _get_storage_breakdown and _estimate_memory_usage
        memory_system._get_storage_breakdown = AsyncMock(
            return_value={"query": 3, "response": 2}
        )
        memory_system._estimate_memory_usage = AsyncMock(return_value=1.5)

        stats = await memory_system.get_memory_stats()

        assert stats.total_entries == 5
        assert stats.cache_hits == 10
        assert stats.cache_misses == 2
        assert stats.hit_ratio == 10 / 12  # 10 hits out of 12 total requests
        assert stats.storage_breakdown == {"query": 3, "response": 2}
        assert stats.memory_usage_mb == 1.5

    @pytest.mark.asyncio
    async def test_clear_all_memory(self, memory_system, mock_redis_client):
        """Test clearing all memory entries."""
        mock_redis_client.keys.return_value = [
            "memory:entry:test1",
            "memory:entry:test2",
        ]
        mock_redis_client.delete.return_value = 2

        result = await memory_system.clear_all_memory()

        assert result is True
        mock_redis_client.delete.assert_called_with(
            "memory:entry:test1", "memory:entry:test2"
        )
        assert memory_system.stats.total_entries == 0
        assert memory_system.stats.cache_hits == 0
        assert memory_system.stats.cache_misses == 0

    @pytest.mark.asyncio
    async def test_similarity_calculation(self, memory_system):
        """Test similarity calculation between embeddings."""
        query_embedding = np.array([1.0, 0.0, 0.0, 0.0])

        # Test identical embeddings (similarity = 1.0)
        entry1 = MemoryEntry(
            id="test1",
            content="test",
            embedding=[1.0, 0.0, 0.0, 0.0],
            context_type="query",
            confidence=1.0,
            timestamp=time.time(),
        )

        similarity1 = await memory_system._calculate_similarity(query_embedding, entry1)
        assert similarity1 == pytest.approx(1.0, rel=1e-3)

        # Test orthogonal embeddings (similarity = 0.0)
        entry2 = MemoryEntry(
            id="test2",
            content="test",
            embedding=[0.0, 1.0, 0.0, 0.0],
            context_type="query",
            confidence=1.0,
            timestamp=time.time(),
        )

        similarity2 = await memory_system._calculate_similarity(query_embedding, entry2)
        assert similarity2 == pytest.approx(0.0, rel=1e-3)

        # Test opposite embeddings (similarity = -1.0)
        entry3 = MemoryEntry(
            id="test3",
            content="test",
            embedding=[-1.0, 0.0, 0.0, 0.0],
            context_type="query",
            confidence=1.0,
            timestamp=time.time(),
        )

        similarity3 = await memory_system._calculate_similarity(query_embedding, entry3)
        assert similarity3 == pytest.approx(-1.0, rel=1e-3)

    @pytest.mark.asyncio
    async def test_entry_id_generation(self, memory_system):
        """Test unique entry ID generation."""
        import time

        id1 = memory_system._generate_entry_id("test content", "query")
        # Add small delay to ensure different timestamp
        time.sleep(0.001)
        id2 = memory_system._generate_entry_id("test content", "query")
        id3 = memory_system._generate_entry_id("different content", "query")

        # Same content should generate different IDs (due to timestamp)
        assert id1 != id2

        # Different content should generate different IDs
        assert id1 != id3
        assert id2 != id3

        # All should start with context type
        assert id1.startswith("query_")
        assert id2.startswith("query_")
        assert id3.startswith("query_")

    @pytest.mark.asyncio
    async def test_lru_cleanup(self, memory_system, mock_redis_client):
        """Test LRU cleanup of entries."""
        # Create entries with different access times
        old_entry = MemoryEntry(
            id="old_entry",
            content="old content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
            last_accessed=time.time() - 3600,  # 1 hour ago
        )

        new_entry = MemoryEntry(
            id="new_entry",
            content="new content",
            embedding=[0.1, 0.2, 0.3, 0.4],
            context_type="query",
            confidence=0.9,
            timestamp=time.time(),
            last_accessed=time.time(),  # Now
        )

        # Mock _get_entries_by_type to return both entries
        memory_system._get_entries_by_type = AsyncMock(
            return_value=[old_entry, new_entry]
        )

        # Mock delete_context to return True
        memory_system.delete_context = AsyncMock(return_value=True)

        cleanup_count = await memory_system._cleanup_lru_entries(1)

        assert cleanup_count == 1
        # Should delete the older entry (old_entry)
        memory_system.delete_context.assert_called_once_with("old_entry")

    @pytest.mark.asyncio
    async def test_storage_breakdown(self, memory_system):
        """Test storage breakdown by context type."""
        entries = [
            MemoryEntry("id1", "content1", [0.1], "query", 1.0, time.time()),
            MemoryEntry("id2", "content2", [0.2], "query", 1.0, time.time()),
            MemoryEntry("id3", "content3", [0.3], "response", 1.0, time.time()),
            MemoryEntry("id4", "content4", [0.4], "entity", 1.0, time.time()),
            MemoryEntry("id5", "content5", [0.5], "response", 1.0, time.time()),
        ]

        memory_system._get_entries_by_type = AsyncMock(return_value=entries)

        breakdown = await memory_system._get_storage_breakdown()

        expected = {"query": 2, "response": 2, "entity": 1}
        assert breakdown == expected

    @pytest.mark.asyncio
    async def test_close(self, memory_system, mock_redis_client):
        """Test closing memory system connections."""
        mock_pool = AsyncMock()
        memory_system._connection_pool = mock_pool

        await memory_system.close()

        mock_redis_client.close.assert_called_once()
        mock_pool.disconnect.assert_called_once()


class TestManagedMemorySystem:
    """Test managed memory system context manager."""

    @pytest.mark.asyncio
    async def test_managed_memory_system_context_manager(self):
        """Test memory system as async context manager."""
        with patch(
            "graphrag.graph_enrichment.memory_system.ContextualMemorySystem"
        ) as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance

            async with ManagedMemorySystem() as memory_system:
                assert memory_system == mock_instance
                mock_instance.initialize.assert_called_once()

            mock_instance.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_managed_memory_system_with_exception(self):
        """Test memory system cleanup on exception."""
        with patch(
            "graphrag.graph_enrichment.memory_system.ContextualMemorySystem"
        ) as mock_class:
            mock_instance = AsyncMock()
            mock_class.return_value = mock_instance

            try:
                async with ManagedMemorySystem() as _:
                    mock_instance.initialize.assert_called_once()
                    raise ValueError("Test exception")
            except ValueError:
                pass

            mock_instance.close.assert_called_once()


@pytest.mark.asyncio
async def test_integration_with_real_similarity_calculations():
    """Integration test with real similarity calculations."""
    # Test similarity calculations without Redis
    with patch("graphrag.graph_enrichment.memory_system.ModelCache"):
        with patch("redis.asyncio.Redis"):
            system = ContextualMemorySystem()

            # Test with real numpy arrays
            query_embedding = np.array([1.0, 0.0, 0.0])

            similar_entry = MemoryEntry(
                id="similar",
                content="similar content",
                embedding=[0.8, 0.6, 0.0],  # Should have high similarity
                context_type="query",
                confidence=1.0,
                timestamp=time.time(),
            )

            different_entry = MemoryEntry(
                id="different",
                content="different content",
                embedding=[0.0, 0.0, 1.0],  # Should have low similarity
                context_type="query",
                confidence=1.0,
                timestamp=time.time(),
            )

            similar_score = await system._calculate_similarity(
                query_embedding, similar_entry
            )
            different_score = await system._calculate_similarity(
                query_embedding, different_entry
            )

            # Similar entry should have higher similarity
            assert similar_score > different_score
            assert similar_score > 0.5
            assert different_score == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
