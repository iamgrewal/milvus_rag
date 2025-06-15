import pytest
import sys
import os
import time
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from graphrag.milvus.manager import MilvusManager
from graphrag.config.settings import Config
from pymilvus import utility


class TestMilvusHNSW:
    """Test HNSW index configuration and performance."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test."""
        # Clean up any existing collection before test
        if utility.has_collection(Config.COLLECTION_NAME):
            utility.drop_collection(Config.COLLECTION_NAME)
        yield
        # Clean up after test
        if utility.has_collection(Config.COLLECTION_NAME):
            utility.drop_collection(Config.COLLECTION_NAME)
    
    def test_hnsw_index_creation(self):
        """Test that HNSW index is created correctly."""
        manager = MilvusManager()
        
        # Check collection exists and is loaded
        assert utility.has_collection(Config.COLLECTION_NAME)
        assert manager.collection.is_loaded
        
        # Check index information
        indexes = manager.collection.indexes
        assert len(indexes) > 0
        
        # Get the embedding field index
        embedding_index = None
        for index in indexes:
            if index.field_name == "embedding":
                embedding_index = index
                break
        
        assert embedding_index is not None
        assert embedding_index.params["index_type"] == "HNSW"
        assert embedding_index.params["metric_type"] == "L2"
        assert embedding_index.params["params"]["M"] == 16
        assert embedding_index.params["params"]["efConstruction"] == 500
    
    def test_collection_persistence(self):
        """Test that collection is not recreated on subsequent initializations."""
        # First initialization
        manager1 = MilvusManager()
        collection_name = Config.COLLECTION_NAME
        
        # Insert test data
        test_data = [
            {"id": 1, "entity": "test_entity", "entity_type": "PERSON", 
             "embedding": np.random.random(Config.EMBEDDING_DIM).tolist()}
        ]
        manager1.insert(test_data)
        
        # Get initial stats
        stats1 = manager1.get_collection_stats()
        assert stats1["total_entities"] > 0
        
        # Second initialization (should not drop collection)
        manager2 = MilvusManager()
        stats2 = manager2.get_collection_stats()
        
        # Data should be preserved
        assert stats2["total_entities"] == stats1["total_entities"]
        assert stats2["is_loaded"] == True
    
    def test_search_performance(self):
        """Test search performance with HNSW index."""
        manager = MilvusManager()
        
        # Insert test data
        test_data = []
        for i in range(100):
            test_data.append({
                "id": i,
                "entity": f"entity_{i}",
                "entity_type": "TEST",
                "embedding": np.random.random(Config.EMBEDDING_DIM).tolist()
            })
        
        # Measure insert time
        start_time = time.time()
        manager.insert(test_data)
        insert_time = time.time() - start_time
        
        # Measure search time
        query_embedding = np.random.random(Config.EMBEDDING_DIM).tolist()
        start_time = time.time()
        results = manager.search(query_embedding, top_k=5)
        search_time = time.time() - start_time
        
        # Verify results
        assert len(results) > 0
        assert len(results[0]) <= 5
        
        # Performance should be reasonable
        assert search_time < 0.1, f"Search too slow: {search_time}s"
        
        print(f"Insert time: {insert_time:.3f}s, Search time: {search_time:.3f}s")
    
    def test_search_accuracy(self):
        """Test search accuracy with HNSW index."""
        manager = MilvusManager()
        
        # Create test data with known similar vectors
        base_vector = np.random.random(Config.EMBEDDING_DIM).tolist()
        
        test_data = [
            {"id": 0, "entity": "target", "entity_type": "TARGET", "embedding": base_vector},
            {"id": 1, "entity": "similar1", "entity_type": "SIMILAR", 
             "embedding": (np.array(base_vector) + np.random.random(Config.EMBEDDING_DIM) * 0.1).tolist()},
            {"id": 2, "entity": "similar2", "entity_type": "SIMILAR", 
             "embedding": (np.array(base_vector) + np.random.random(Config.EMBEDDING_DIM) * 0.1).tolist()},
        ]
        
        # Add some dissimilar vectors
        for i in range(3, 20):
            test_data.append({
                "id": i,
                "entity": f"different_{i}",
                "entity_type": "DIFFERENT",
                "embedding": np.random.random(Config.EMBEDDING_DIM).tolist()
            })
        
        manager.insert(test_data)
        
        # Search for similar vectors
        results = manager.search(base_vector, top_k=3)
        
        # Should find the exact match and similar vectors
        assert len(results) > 0
        assert len(results[0]) >= 3
        
        # First result should be exact match or very close
        top_result = results[0][0]
        assert top_result.distance < 0.01  # Very small distance for L2
    
    def test_collection_stats(self):
        """Test collection statistics functionality."""
        manager = MilvusManager()
        
        # Initial stats
        stats = manager.get_collection_stats()
        assert "total_entities" in stats
        assert "is_loaded" in stats
        assert "segments" in stats
        assert stats["is_loaded"] == True
        
        # Insert data and check stats update
        test_data = [
            {"id": 1, "entity": "test", "entity_type": "TEST", 
             "embedding": np.random.random(Config.EMBEDDING_DIM).tolist()}
        ]
        manager.insert(test_data)
        
        updated_stats = manager.get_collection_stats()
        assert updated_stats["total_entities"] > stats["total_entities"]
    
    def test_batch_operations(self):
        """Test batch insert and search operations."""
        manager = MilvusManager()
        
        # Large batch insert
        batch_size = 500
        test_data = []
        for i in range(batch_size):
            test_data.append({
                "id": i,
                "entity": f"batch_entity_{i}",
                "entity_type": "BATCH",
                "embedding": np.random.random(Config.EMBEDDING_DIM).tolist()
            })
        
        start_time = time.time()
        manager.insert(test_data)
        batch_insert_time = time.time() - start_time
        
        # Verify all data was inserted
        stats = manager.get_collection_stats()
        assert stats["total_entities"] >= batch_size
        
        # Multiple search operations
        search_times = []
        for _ in range(10):
            query_embedding = np.random.random(Config.EMBEDDING_DIM).tolist()
            start_time = time.time()
            results = manager.search(query_embedding, top_k=10)
            search_time = time.time() - start_time
            search_times.append(search_time)
            
            assert len(results) > 0
            assert len(results[0]) <= 10
        
        avg_search_time = sum(search_times) / len(search_times)
        
        print(f"Batch insert ({batch_size} records): {batch_insert_time:.3f}s")
        print(f"Average search time: {avg_search_time:.3f}s")
        
        # Performance targets
        assert avg_search_time < 0.05, f"Average search too slow: {avg_search_time}s"