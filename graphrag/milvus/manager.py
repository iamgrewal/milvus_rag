from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from graphrag.config.settings import Config
from graphrag.logger import logger


class MilvusManager:
    def __init__(self):
        connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)

        # Check if collection exists, create only if needed (preserve data and indexes)
        if not utility.has_collection(Config.COLLECTION_NAME):
            logger.info(f"Creating new collection: {Config.COLLECTION_NAME}")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
            ]
            schema = CollectionSchema(fields, description="Graph RAG Collection")
            self.collection = Collection(Config.COLLECTION_NAME, schema)

            # HNSW index configuration for 2-3x performance improvement
            hnsw_index_params = {
                "index_type": "HNSW",
                "metric_type": "L2",  # L2 distance for better accuracy
                "params": {
                    "M": 16,             # Max connections per node (higher = better recall, more memory)
                    "efConstruction": 500 # Size of candidate list during construction (higher = better quality)
                }
            }

            logger.info("Creating HNSW index for optimal performance")
            self.collection.create_index("embedding", hnsw_index_params)
        else:
            logger.info(f"Using existing collection: {Config.COLLECTION_NAME}")
            self.collection = Collection(Config.COLLECTION_NAME)

        # Load collection for search operations
        if not self.collection.is_loaded:
            self.collection.load()
            logger.info("Collection loaded for search operations")

    def insert(self, records):
        """Insert records with automatic flushing for persistence."""
        self.collection.insert(records)
        # Flush to ensure data persistence and index updates
        self.collection.flush()
        logger.info(f"Inserted {len(records)} records and flushed to storage")

    def get_collection_stats(self):
        """Get collection statistics for monitoring."""
        stats = utility.get_query_segment_info(Config.COLLECTION_NAME)
        return {
            "total_entities": self.collection.num_entities,
            "is_loaded": self.collection.is_loaded,
            "segments": len(stats) if stats else 0
        }

    def search(self, embedding, top_k=3):
        # HNSW search parameters optimized for performance
        search_params = {
            "metric_type": "L2",
            "params": {
                "ef": max(top_k * 4, 64)  # Search width (higher = better recall, slower search)
            }
        }

        return self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["entity", "entity_type"]
        )
