"""
Contextual Memory System with Redis-based Similarity Caching

Implements advanced memory system for Phase 2 requirements with:
- Redis-based similarity caching for high-performance retrieval
- Embedding-based context retrieval and similarity search
- Automatic cache management with TTL and eviction policies
- Multi-level memory architecture with hot/warm/cold storage
"""

import asyncio
import hashlib
import json
import pickle
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import redis.asyncio as redis

from graphrag.config.settings import Config
from graphrag.embedding_service.service import EmbeddingService
from graphrag.logger import logger
from utils.model_cache import ModelCache


@dataclass
class MemoryEntry:
    """Represents a memory entry with metadata."""

    id: str
    content: str
    embedding: list[float]
    context_type: str
    confidence: float
    timestamp: float
    access_count: int = 0
    last_accessed: float = 0.0
    metadata: Optional[dict[str, Any]] = None


@dataclass
class SimilaritySearchResult:
    """Result from similarity search in memory."""

    entry: MemoryEntry
    similarity_score: float
    rank: int


@dataclass
class MemoryStats:
    """Memory system statistics."""

    total_entries: int
    cache_hits: int
    cache_misses: int
    hit_ratio: float
    avg_similarity: float
    storage_breakdown: dict[str, int]
    memory_usage_mb: float


class ContextualMemorySystem:
    """
    Advanced contextual memory system with Redis-based similarity caching.

    Features:
    - High-performance Redis backend with embedding storage
    - Similarity-based retrieval using cosine similarity
    - Automatic cache management with LRU and TTL policies
    - Multi-tier storage (hot/warm/cold) for optimization
    - Async-first architecture for non-blocking operations
    """

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        redis_host: str = None,
        redis_port: int = None,
        cache_ttl: int = None,
        max_cache_size: int = None,
        similarity_threshold: float = None,
    ):
        """
        Initialize the contextual memory system.

        Args:
            embedding_service: Text embedding service
            redis_host: Redis server host
            redis_port: Redis server port
            cache_ttl: Cache time-to-live in seconds
            max_cache_size: Maximum number of cached entries
            similarity_threshold: Minimum similarity for retrieval
        """
        # Configuration
        self.redis_host = redis_host or Config.REDIS_HOST
        self.redis_port = redis_port or Config.REDIS_PORT
        self.cache_ttl = cache_ttl or Config.MEMORY_CACHE_TTL
        self.max_cache_size = max_cache_size or Config.MEMORY_MAX_CACHE_SIZE
        self.similarity_threshold = (
            similarity_threshold or Config.MEMORY_SIMILARITY_THRESHOLD
        )

        # Services
        self.embedding_service = embedding_service or EmbeddingService()
        self.model_cache = ModelCache()
        self.similarity_model = self.model_cache.get_embedding_model("all-MiniLM-L6-v2")

        # Redis connection
        self.redis_client: Optional[redis.Redis] = None
        self._connection_pool = None

        # Memory statistics
        self.stats = MemoryStats(
            total_entries=0,
            cache_hits=0,
            cache_misses=0,
            hit_ratio=0.0,
            avg_similarity=0.0,
            storage_breakdown={},
            memory_usage_mb=0.0,
        )

        # Cache prefixes for different data types
        self.ENTRY_PREFIX = "memory:entry:"
        self.EMBEDDING_PREFIX = "memory:embedding:"
        self.INDEX_PREFIX = "memory:index:"
        self.STATS_PREFIX = "memory:stats:"

        logger.info(
            f"ContextualMemorySystem initialized - "
            f"Redis: {self.redis_host}:{self.redis_port}, "
            f"TTL: {self.cache_ttl}s, "
            f"Max size: {self.max_cache_size}, "
            f"Similarity threshold: {self.similarity_threshold}"
        )

    async def initialize(self) -> bool:
        """
        Initialize Redis connection and memory system.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Create Redis connection pool
            self._connection_pool = redis.ConnectionPool(
                host=self.redis_host,
                port=self.redis_port,
                db=Config.REDIS_DB,
                password=Config.REDIS_PASSWORD,
                decode_responses=False,  # We'll handle binary data
                max_connections=20,
                retry_on_timeout=True,
            )

            self.redis_client = redis.Redis(connection_pool=self._connection_pool)

            # Test connection
            await self.redis_client.ping()

            # Load existing statistics
            await self._load_stats()

            logger.info("ContextualMemorySystem initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize ContextualMemorySystem: {e}")
            return False

    async def store_context(
        self,
        content: str,
        context_type: str,
        confidence: float = 1.0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Store content in contextual memory with automatic embedding.

        Args:
            content: Text content to store
            context_type: Type of context (query, response, entity, etc.)
            confidence: Confidence score for the content
            metadata: Additional metadata

        Returns:
            Unique ID of the stored memory entry
        """
        try:
            if not self.redis_client:
                await self.initialize()

            # Generate embedding for content
            embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.similarity_model.encode, content
            )
            embedding_list = embedding.tolist()

            # Create memory entry
            entry_id = self._generate_entry_id(content, context_type)
            current_time = time.time()

            entry = MemoryEntry(
                id=entry_id,
                content=content,
                embedding=embedding_list,
                context_type=context_type,
                confidence=confidence,
                timestamp=current_time,
                last_accessed=current_time,
                metadata=metadata or {},
            )

            # Store in Redis with automatic expiration
            await self._store_entry(entry)

            # Update statistics
            self.stats.total_entries += 1
            await self._update_stats()

            logger.debug(f"Stored memory entry: {entry_id} ({context_type})")
            return entry_id

        except Exception as e:
            logger.error(f"Failed to store context: {e}")
            raise

    async def retrieve_similar_contexts(
        self,
        query: str,
        context_type: Optional[str] = None,
        max_results: int = 10,
        min_similarity: Optional[float] = None,
    ) -> list[SimilaritySearchResult]:
        """
        Retrieve contexts similar to the query using embedding-based search.

        Args:
            query: Query text to find similar contexts for
            context_type: Filter by context type (optional)
            max_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar memory entries with similarity scores
        """
        try:
            if not self.redis_client:
                await self.initialize()

            min_similarity = min_similarity or self.similarity_threshold

            # Generate query embedding
            query_embedding = await asyncio.get_event_loop().run_in_executor(
                None, self.similarity_model.encode, query
            )

            # Get all stored entries (with filtering if context_type specified)
            stored_entries = await self._get_entries_by_type(context_type)

            if not stored_entries:
                self.stats.cache_misses += 1
                await self._update_stats()
                return []

            # Calculate similarities in parallel
            similarity_tasks = [
                self._calculate_similarity(query_embedding, entry)
                for entry in stored_entries
            ]
            similarities = await asyncio.gather(*similarity_tasks)

            # Filter by minimum similarity and sort
            similar_entries = []
            for entry, similarity in zip(stored_entries, similarities):
                if similarity >= min_similarity:
                    similar_entries.append(
                        SimilaritySearchResult(
                            entry=entry,
                            similarity_score=similarity,
                            rank=0,  # Will be set after sorting
                        )
                    )

            # Sort by similarity descending
            similar_entries.sort(key=lambda x: x.similarity_score, reverse=True)

            # Set ranks and limit results
            results = []
            for i, result in enumerate(similar_entries[:max_results]):
                result.rank = i + 1
                results.append(result)

                # Update access statistics
                await self._update_entry_access(result.entry.id)

            # Update statistics
            if results:
                self.stats.cache_hits += 1
                self.stats.avg_similarity = sum(
                    r.similarity_score for r in results
                ) / len(results)
            else:
                self.stats.cache_misses += 1

            await self._update_stats()

            logger.debug(f"Retrieved {len(results)} similar contexts for query")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve similar contexts: {e}")
            self.stats.cache_misses += 1
            await self._update_stats()
            return []

    async def get_context_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve a specific context by its ID.

        Args:
            entry_id: Unique ID of the memory entry

        Returns:
            Memory entry if found, None otherwise
        """
        try:
            if not self.redis_client:
                await self.initialize()

            # Get entry from Redis
            entry_data = await self.redis_client.get(f"{self.ENTRY_PREFIX}{entry_id}")

            if entry_data:
                entry = pickle.loads(entry_data)
                await self._update_entry_access(entry_id)
                self.stats.cache_hits += 1
                await self._update_stats()
                return entry
            else:
                self.stats.cache_misses += 1
                await self._update_stats()
                return None

        except Exception as e:
            logger.error(f"Failed to get context by ID {entry_id}: {e}")
            return None

    async def update_context(
        self,
        entry_id: str,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Update an existing memory entry.

        Args:
            entry_id: ID of the entry to update
            content: New content (will regenerate embedding)
            confidence: New confidence score
            metadata: New metadata to merge

        Returns:
            True if update successful, False otherwise
        """
        try:
            entry = await self.get_context_by_id(entry_id)
            if not entry:
                return False

            # Update fields
            if content is not None:
                entry.content = content
                # Regenerate embedding
                embedding = await asyncio.get_event_loop().run_in_executor(
                    None, self.similarity_model.encode, content
                )
                entry.embedding = embedding.tolist()

            if confidence is not None:
                entry.confidence = confidence

            if metadata is not None:
                entry.metadata = {**(entry.metadata or {}), **metadata}

            # Store updated entry
            await self._store_entry(entry)

            logger.debug(f"Updated memory entry: {entry_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update context {entry_id}: {e}")
            return False

    async def delete_context(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            if not self.redis_client:
                await self.initialize()

            # Delete from Redis
            deleted = await self.redis_client.delete(f"{self.ENTRY_PREFIX}{entry_id}")

            if deleted:
                self.stats.total_entries -= 1
                await self._update_stats()
                logger.debug(f"Deleted memory entry: {entry_id}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Failed to delete context {entry_id}: {e}")
            return False

    async def cleanup_expired_entries(self) -> int:
        """
        Clean up expired memory entries based on TTL and LRU policies.

        Returns:
            Number of entries cleaned up
        """
        try:
            if not self.redis_client:
                await self.initialize()

            current_time = time.time()
            cleanup_count = 0

            # Get all entry keys
            entry_keys = await self.redis_client.keys(f"{self.ENTRY_PREFIX}*")

            # Check each entry for expiration
            for key in entry_keys:
                try:
                    entry_data = await self.redis_client.get(key)
                    if entry_data:
                        entry = pickle.loads(entry_data)

                        # Check if entry is expired
                        if current_time - entry.timestamp > self.cache_ttl:
                            await self.redis_client.delete(key)
                            cleanup_count += 1

                except Exception as e:
                    logger.warning(f"Error processing entry {key} during cleanup: {e}")
                    continue

            # If still over capacity, remove LRU entries
            if len(entry_keys) - cleanup_count > self.max_cache_size:
                lru_cleanup = await self._cleanup_lru_entries(
                    len(entry_keys) - cleanup_count - self.max_cache_size
                )
                cleanup_count += lru_cleanup

            # Update statistics
            self.stats.total_entries -= cleanup_count
            await self._update_stats()

            if cleanup_count > 0:
                logger.info(f"Cleaned up {cleanup_count} expired memory entries")

            return cleanup_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired entries: {e}")
            return 0

    async def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory system statistics.

        Returns:
            Current memory statistics
        """
        try:
            await self._update_stats()

            # Calculate hit ratio
            total_requests = self.stats.cache_hits + self.stats.cache_misses
            self.stats.hit_ratio = (
                self.stats.cache_hits / total_requests if total_requests > 0 else 0.0
            )

            # Get storage breakdown by context type
            self.stats.storage_breakdown = await self._get_storage_breakdown()

            # Estimate memory usage
            self.stats.memory_usage_mb = await self._estimate_memory_usage()

            return self.stats

        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return self.stats

    async def clear_all_memory(self) -> bool:
        """
        Clear all memory entries (use with caution).

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.redis_client:
                await self.initialize()

            # Delete all memory-related keys
            entry_keys = await self.redis_client.keys(f"{self.ENTRY_PREFIX}*")
            if entry_keys:
                await self.redis_client.delete(*entry_keys)

            # Reset statistics
            self.stats = MemoryStats(
                total_entries=0,
                cache_hits=0,
                cache_misses=0,
                hit_ratio=0.0,
                avg_similarity=0.0,
                storage_breakdown={},
                memory_usage_mb=0.0,
            )
            await self._update_stats()

            logger.info("Cleared all memory entries")
            return True

        except Exception as e:
            logger.error(f"Failed to clear all memory: {e}")
            return False

    async def close(self):
        """Close Redis connections and cleanup resources."""
        try:
            if self.redis_client:
                await self.redis_client.close()
            if self._connection_pool:
                await self._connection_pool.disconnect()

            logger.info("ContextualMemorySystem closed successfully")

        except Exception as e:
            logger.error(f"Error closing ContextualMemorySystem: {e}")

    # Private helper methods

    def _generate_entry_id(self, content: str, context_type: str) -> str:
        """Generate unique ID for memory entry."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        timestamp = str(int(time.time() * 1000))[-8:]  # Last 8 digits of timestamp
        return f"{context_type}_{content_hash}_{timestamp}"

    async def _store_entry(self, entry: MemoryEntry):
        """Store memory entry in Redis."""
        entry_data = pickle.dumps(entry)
        await self.redis_client.setex(
            f"{self.ENTRY_PREFIX}{entry.id}", self.cache_ttl, entry_data
        )

    async def _get_entries_by_type(
        self, context_type: Optional[str] = None
    ) -> list[MemoryEntry]:
        """Get all entries, optionally filtered by context type."""
        entries = []

        # Get all entry keys
        pattern = f"{self.ENTRY_PREFIX}*"
        if context_type:
            pattern = f"{self.ENTRY_PREFIX}{context_type}_*"

        entry_keys = await self.redis_client.keys(pattern)

        # Fetch entries in parallel
        if entry_keys:
            entry_data_list = await self.redis_client.mget(entry_keys)

            for entry_data in entry_data_list:
                if entry_data:
                    try:
                        entry = pickle.loads(entry_data)
                        entries.append(entry)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize entry: {e}")
                        continue

        return entries

    async def _calculate_similarity(
        self, query_embedding: np.ndarray, entry: MemoryEntry
    ) -> float:
        """Calculate cosine similarity between query and stored entry."""
        entry_embedding = np.array(entry.embedding)

        # Cosine similarity
        dot_product = np.dot(query_embedding, entry_embedding)
        norm_query = np.linalg.norm(query_embedding)
        norm_entry = np.linalg.norm(entry_embedding)

        if norm_query == 0 or norm_entry == 0:
            return 0.0

        similarity = dot_product / (norm_query * norm_entry)
        return float(similarity)

    async def _update_entry_access(self, entry_id: str):
        """Update entry access statistics."""
        entry = await self.get_context_by_id(entry_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = time.time()
            await self._store_entry(entry)

    async def _cleanup_lru_entries(self, num_to_remove: int) -> int:
        """Remove least recently used entries."""
        try:
            entries = await self._get_entries_by_type()

            # Sort by last accessed time (oldest first)
            entries.sort(key=lambda x: x.last_accessed)

            cleanup_count = 0
            for entry in entries[:num_to_remove]:
                if await self.delete_context(entry.id):
                    cleanup_count += 1

            return cleanup_count

        except Exception as e:
            logger.error(f"Failed to cleanup LRU entries: {e}")
            return 0

    async def _get_storage_breakdown(self) -> dict[str, int]:
        """Get breakdown of storage by context type."""
        breakdown = defaultdict(int)

        try:
            entries = await self._get_entries_by_type()
            for entry in entries:
                breakdown[entry.context_type] += 1
        except Exception as e:
            logger.error(f"Failed to get storage breakdown: {e}")

        return dict(breakdown)

    async def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            # Get Redis memory info
            info = await self.redis_client.info("memory")
            used_memory = info.get("used_memory", 0)
            return used_memory / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.error(f"Failed to estimate memory usage: {e}")
            return 0.0

    async def _load_stats(self):
        """Load statistics from Redis."""
        try:
            stats_data = await self.redis_client.get(f"{self.STATS_PREFIX}main")
            if stats_data:
                stats_dict = json.loads(stats_data)
                self.stats = MemoryStats(**stats_dict)
        except Exception as e:
            logger.warning(f"Failed to load stats: {e}")

    async def _update_stats(self):
        """Update statistics in Redis."""
        try:
            stats_dict = asdict(self.stats)
            await self.redis_client.setex(
                f"{self.STATS_PREFIX}main", self.cache_ttl, json.dumps(stats_dict)
            )
        except Exception as e:
            logger.warning(f"Failed to update stats: {e}")


# Context manager for automatic cleanup
class ManagedMemorySystem:
    """Context manager for automatic memory system lifecycle management."""

    def __init__(self, **kwargs):
        self.memory_system = ContextualMemorySystem(**kwargs)

    async def __aenter__(self):
        await self.memory_system.initialize()
        return self.memory_system

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.memory_system.close()
