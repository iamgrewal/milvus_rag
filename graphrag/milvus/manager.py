from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from graphrag.config.settings import Config
from graphrag.logger import logger

class MilvusManager:
    def __init__(self):
        connections.connect(host=Config.MILVUS_HOST, port=Config.MILVUS_PORT)
        if utility.has_collection(Config.COLLECTION_NAME):
            utility.drop_collection(Config.COLLECTION_NAME)
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="entity", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="entity_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
        ]
        schema = CollectionSchema(fields, description="Graph RAG Collection")
        self.collection = Collection(Config.COLLECTION_NAME, schema)
        self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
        self.collection.load()

    def insert(self, records):
        self.collection.insert(records)

    def search(self, embedding, top_k=3):
        return self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["entity"]
        )
