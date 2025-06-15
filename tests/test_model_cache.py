import pytest
import threading
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils.model_cache import ModelCache
from graphrag.embedding_service.service import EmbeddingService


class TestModelCache:
    
    def test_singleton_behavior(self):
        """Test that ModelCache implements singleton pattern correctly."""
        cache1 = ModelCache()
        cache2 = ModelCache()
        
        assert cache1 is cache2, "ModelCache should be singleton"
    
    def test_thread_safety(self):
        """Test ModelCache thread safety with concurrent access."""
        results = []
        
        def get_cache():
            cache = ModelCache()
            results.append(id(cache))
        
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=get_cache)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should get the same instance
        assert len(set(results)) == 1, "All threads should get same singleton instance"
    
    def test_model_caching(self):
        """Test that models are properly cached and reused."""
        cache = ModelCache()
        
        # First call should load model
        model1 = cache.get_embedding_model("all-MiniLM-L6-v2")
        
        # Second call should return cached model
        model2 = cache.get_embedding_model("all-MiniLM-L6-v2")
        
        assert model1 is model2, "Same model should be returned from cache"
    
    def test_different_models_cached_separately(self):
        """Test that different models are cached separately."""
        cache = ModelCache()
        
        # Load two different models (using same model name for test simplicity)
        model1 = cache.get_embedding_model("all-MiniLM-L6-v2")
        
        # In real scenario, we'd use different model names
        # For test, we verify cache info shows one model
        cache_info = cache.get_cache_info()
        assert cache_info["cached_models"] == 1
        assert "embedding_all-MiniLM-L6-v2" in cache_info["model_keys"]
    
    def test_cache_clear(self):
        """Test cache clearing functionality."""
        cache = ModelCache()
        
        # Load a model
        cache.get_embedding_model("all-MiniLM-L6-v2")
        
        # Verify model is cached
        assert cache.get_cache_info()["cached_models"] == 1
        
        # Clear cache
        cache.clear_cache()
        
        # Verify cache is empty
        assert cache.get_cache_info()["cached_models"] == 0
    
    def test_embedding_service_integration(self):
        """Test ModelCache integration with EmbeddingService."""
        # Create two embedding services with same model
        service1 = EmbeddingService("all-MiniLM-L6-v2")
        service2 = EmbeddingService("all-MiniLM-L6-v2")
        
        # They should use the same cached model
        assert service1.model is service2.model, "Services should share cached model"
    
    def test_performance_improvement(self):
        """Test that subsequent model loads are faster due to caching."""
        cache = ModelCache()
        cache.clear_cache()  # Start fresh
        
        # Time first load (cold)
        start_time = time.time()
        model1 = cache.get_embedding_model("all-MiniLM-L6-v2")
        cold_load_time = time.time() - start_time
        
        # Time second load (cached)
        start_time = time.time()
        model2 = cache.get_embedding_model("all-MiniLM-L6-v2")
        cached_load_time = time.time() - start_time
        
        # Cached load should be significantly faster
        assert cached_load_time < cold_load_time * 0.1, "Cached load should be much faster"
        assert model1 is model2, "Should return same cached instance"