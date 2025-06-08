from sentence_transformers import SentenceTransformer
from graphrag.logger import logger

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded")

    def embed(self, text):
        return self.model.encode(text).tolist()
