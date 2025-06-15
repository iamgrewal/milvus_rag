"""
Milvus Retriever for Hybrid RAG System

This module implements production-ready vector retrieval using Milvus with HNSW indexing,
partitioning strategies, and optimized search parameters according to the rhoSearcher ruleset.
"""

import asyncio
from typing import Any, Dict, List, Optional

import structlog
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    SearchParams,
    connections,
    utility,
)

from ..config.settings import Config

logger = structlog.get_logger(__name__)


class MilvusRetriever:
    """
    Production-ready Milvus vector retriever implementing HNSW indexing
    with dynamic partitioning and optimized search parameters.
    """

    def __init__(self):
        self.collection_name = Config.COLLECTION_NAME
        self.embedding_dim = Config.EMBEDDING_DIM
        self.connection_alias = "default"
        self.collection: Optional[Collection] = None

        # Search parameters according to ruleset
        self.search_params = {
            "metric_type": "L2",
            "offset": 5,
            "ignore_growing": False,
            "params": {"ef": 32},
        }

        self._initialize_connection()
        self._setup_collection()

    def _initialize_connection(self):
        """Initialize connection to Milvus server."""
        try:
            connections.connect(
                alias=self.connection_alias,
                host=Config.MILVUS_HOST,
                port=Config.MILVUS_PORT,
            )
            logger.info(
                "Milvus connection established",
                host=Config.MILVUS_HOST,
                port=Config.MILVUS_PORT,
            )
        except Exception as e:
            logger.error("Failed to connect to Milvus", error=str(e))
            raise

    def _setup_collection(self):
        """Set up collection with proper schema and HNSW indexing."""
        try:
            # Drop existing collection if it exists (for clean setup)
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                logger.info("Dropped existing collection", name=self.collection_name)

            # Define collection schema
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False,
                    description="Primary key for vector records",
                ),
                FieldSchema(
                    name="entity",
                    dtype=DataType.VARCHAR,
                    max_length=500,
                    description="Entity or document identifier",
                ),
                FieldSchema(
                    name="entity_type",
                    dtype=DataType.VARCHAR,
                    max_length=100,
                    description="Type classification of the entity",
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=8192,
                    description="Text content for the vector",
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dim,
                    description="Dense vector embedding",
                ),
                FieldSchema(
                    name="confidence_score",
                    dtype=DataType.FLOAT,
                    description="Confidence score for the embedding",
                ),
                FieldSchema(
                    name="source",
                    dtype=DataType.VARCHAR,
                    max_length=256,
                    description="Source document or system identifier",
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description="Hybrid RAG Collection with HNSW indexing",
                enable_dynamic_field=True,
            )

            # Create collection
            self.collection = Collection(
                name=self.collection_name, schema=schema, using=self.connection_alias
            )

            logger.info("Collection created", name=self.collection_name)

            # Setup partitioning based on data size expectations
            self._setup_partitioning()

            # Create HNSW index according to ruleset
            self._create_hnsw_index()

            # Load collection for searching
            self.collection.load()
            logger.info("Collection loaded and ready for search")

        except Exception as e:
            logger.error("Failed to setup collection", error=str(e))
            raise

    def _setup_partitioning(self):
        """Setup partitioning strategy based on expected data volume."""
        try:
            # Determine partition count based on expected data size
            # <1M docs → 2 partitions, 1–10M docs → 8 partitions, >10M docs → 16 partitions
            expected_doc_count = getattr(Config, "EXPECTED_DOC_COUNT", 1000000)

            if expected_doc_count < 1000000:
                partition_count = 2
            elif expected_doc_count <= 10000000:
                partition_count = 8
            else:
                partition_count = 16

            # Create partitions
            for i in range(partition_count):
                partition_name = f"partition_{i}"
                if not self.collection.has_partition(partition_name):
                    self.collection.create_partition(partition_name)

            logger.info(
                "Partitioning setup completed",
                partition_count=partition_count,
                expected_docs=expected_doc_count,
            )

        except Exception as e:
            logger.error("Failed to setup partitioning", error=str(e))
            raise

    def _create_hnsw_index(self):
        """Create HNSW index with optimized parameters."""
        try:
            # HNSW index parameters optimized for production
            index_params = {
                "index_type": "HNSW",
                "metric_type": "L2",
                "params": {
                    "M": 16,  # Number of bi-directional links for every node
                    "efConstruction": 256,  # Size of dynamic candidate list
                },
            }

            # Create index on embedding field
            self.collection.create_index(
                field_name="embedding",
                index_params=index_params,
                index_name="embedding_hnsw_index",
            )

            # Create scalar indexes for better filtering performance
            self.collection.create_index(
                field_name="entity_type", index_name="entity_type_index"
            )

            self.collection.create_index(
                field_name="confidence_score", index_name="confidence_index"
            )

            logger.info("HNSW index created successfully")

        except Exception as e:
            logger.error("Failed to create HNSW index", error=str(e))
            raise

    async def retrieve_async(
        self, query: str, top_k: int = 10, min_confidence: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Asynchronous vector retrieval with confidence filtering.

        Args:
            query: Query string (will be embedded)
            top_k: Number of top results to return
            min_confidence: Minimum confidence threshold for results

        Returns:
            List of retrieved documents with metadata
        """
        try:
            # In production, this would call an embedding service
            # For now, using placeholder embedding
            query_embedding = await self._get_query_embedding(query)

            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Execute search with optimized parameters
            results = await self._execute_search(
                embedding=query_embedding, top_k=top_k, min_confidence=min_confidence
            )

            logger.info(
                "Vector retrieval completed",
                query_length=len(query),
                results_count=len(results),
            )

            return results

        except Exception as e:
            logger.error("Vector retrieval failed", error=str(e), query=query[:100])
            return []

    async def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for query string."""
        try:
            # Placeholder - in production, this would call embedding service
            # Using simple hash-based embedding for demonstration
            import hashlib
            import struct

            # Generate deterministic "embedding" from query hash
            hash_obj = hashlib.md5(query.encode())
            hash_bytes = hash_obj.digest()

            # Convert to float vector of specified dimension
            embedding = []
            for i in range(0, len(hash_bytes), 4):
                if len(embedding) >= self.embedding_dim:
                    break
                chunk = hash_bytes[i : i + 4]
                if len(chunk) == 4:
                    float_val = struct.unpack("f", chunk)[0]
                    embedding.append(float_val)

            # Pad to required dimension
            while len(embedding) < self.embedding_dim:
                embedding.append(0.0)

            return embedding[: self.embedding_dim]

        except Exception as e:
            logger.error("Failed to generate query embedding", error=str(e))
            return None

    async def _execute_search(
        self, embedding: List[float], top_k: int, min_confidence: float
    ) -> List[Dict[str, Any]]:
        """Execute vector search with filtering."""
        try:
            # Search parameters with confidence filtering
            search_params = SearchParams(self.search_params)

            # Build filter expression for confidence
            filter_expr = f"confidence_score >= {min_confidence}"

            # Execute search
            search_results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=filter_expr,
                output_fields=[
                    "entity",
                    "entity_type",
                    "content",
                    "confidence_score",
                    "source",
                ],
                consistency_level="Strong",
            )

            # Process results
            processed_results = []
            for hits in search_results:
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "score": 1.0
                        / (1.0 + hit.distance),  # Convert distance to similarity
                        "entity": hit.entity.get("entity", ""),
                        "entity_type": hit.entity.get("entity_type", ""),
                        "content": hit.entity.get("content", ""),
                        "confidence_score": hit.entity.get("confidence_score", 0.0),
                        "source": hit.entity.get("source", ""),
                        "retrieval_type": "vector",
                    }
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error("Search execution failed", error=str(e))
            return []

    def insert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Insert documents into the collection with embeddings.

        Args:
            documents: List of document dictionaries with content and metadata

        Returns:
            True if insertion successful, False otherwise
        """
        try:
            if not documents:
                return True

            # Prepare data for insertion
            entities = []
            for i, doc in enumerate(documents):
                # Generate embedding for document content
                embedding = asyncio.run(
                    self._get_query_embedding(doc.get("content", ""))
                )
                if not embedding:
                    continue

                entity = [
                    doc.get("id", i),  # id
                    doc.get("entity", f"entity_{i}"),  # entity
                    doc.get("entity_type", "document"),  # entity_type
                    doc.get("content", ""),  # content
                    embedding,  # embedding
                    doc.get("confidence_score", 1.0),  # confidence_score
                    doc.get("source", "unknown"),  # source
                ]
                entities.append(entity)

            if entities:
                # Insert data
                mr = self.collection.insert(entities)
                self.collection.flush()

                logger.info(
                    "Documents inserted successfully",
                    count=len(entities),
                    primary_keys=mr.primary_keys[:5],
                )  # Log first 5 keys
                return True

            return False

        except Exception as e:
            logger.error("Document insertion failed", error=str(e))
            return False

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics for monitoring."""
        try:
            stats = {
                "name": self.collection_name,
                "num_entities": self.collection.num_entities,
                "schema": str(self.collection.schema),
                "indexes": [index.field_name for index in self.collection.indexes],
                "partitions": [p.name for p in self.collection.partitions],
                "is_loaded": utility.load_state(self.collection_name),
            }

            return stats

        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {}

    def health_check(self) -> bool:
        """Perform health check on the Milvus connection and collection."""
        try:
            # Check connection
            if not connections.has_connection(self.connection_alias):
                return False

            # Check collection exists and is loaded
            if not utility.has_collection(self.collection_name):
                return False

            # Simple search test
            test_embedding = [0.1] * self.embedding_dim
            search_results = self.collection.search(
                data=[test_embedding],
                anns_field="embedding",
                param=self.search_params,
                limit=1,
            )

            return True

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
