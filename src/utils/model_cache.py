import threading
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from graphrag.logger import logger


class ModelCache:
    """Singleton cache for ML models to eliminate redundant loading overhead."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelCache, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._models: Dict[str, Any] = {}
            self._model_lock = threading.Lock()
            self._initialized = True
            logger.info("ModelCache initialized")
    
    def get_embedding_model(self, model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
        """Get or load embedding model with caching."""
        cache_key = f"embedding_{model_name}"
        
        if cache_key not in self._models:
            with self._model_lock:
                if cache_key not in self._models:
                    logger.info(f"Loading embedding model: {model_name}")
                    self._models[cache_key] = SentenceTransformer(model_name)
                    logger.info(f"Embedding model cached: {model_name}")
        
        return self._models[cache_key]
    
    def clear_cache(self):
        """Clear all cached models."""
        with self._model_lock:
            self._models.clear()
            logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "cached_models": len(self._models),
            "model_keys": list(self._models.keys())
        }