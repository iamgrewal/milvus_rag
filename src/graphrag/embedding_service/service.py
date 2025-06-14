import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from graphrag.logger import logger
from utils.model_cache import ModelCache

class EmbeddingService:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_cache = ModelCache()
        self.model = self.model_cache.get_embedding_model(model_name)
        self.model_name = model_name
        logger.info(f"Embedding service initialized with cached model: {model_name}")

    def embed(self, text):
        return self.model.encode(text).tolist()
