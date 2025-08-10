#!/usr/bin/env python3
"""
Tests for intelligent caching system.

This module tests the LRU cache, batch-aware caching, cache warming,
and overall intelligent caching system functionality.
"""

import unittest
import numpy as np
import tempfile
import os
import time
import threading
from unittest.mock import Mock, patch

from src.lsm.data.intelligent_caching import (
    CacheConfig, CacheMetrics, LRUCache, BatchCache, 
    CacheWarmer, IntelligentCachingSystem
)


class TestCacheMetrics(unittest.TestCase):
    """Test cache metrics tracking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = CacheMetrics(window_size=10)
    
    def test_initialization(self):
        """Test metrics initialization."""
        self.assertEqual(self.metrics.hits, 0)
        self.assertEqual(self.metrics.misses, 0)
        self.assertEqual(self.metrics.total_requests, 0)
        self.assertEqual(self.metrics.get_hit_rate(), 0.0)
    
    def test_hit_recording(self):
        """Test recording cache hits."""
        self.metrics.record_hit(0.001)
        self.assertEqual(self.metrics.hits, 1)
        self.assertEqual(self.metrics.total_requests, 1)
        self.assertEqual(self.metrics.get_hit_rate(), 1.0)
    
    def test_miss_recording(self):
        """Test recording cache misses."""
        self.metrics.record_miss(0.002)
        self.assertEqual(self.metrics.misses, 1)
        self.assertEqual(self.metrics.total_requests, 1)
        self.assertEqual(self.metrics.get_hit_rate(), 0.0)
    
    def test_mixed_hit_miss_recording(self):
        """Test recording mixed hits and misses."""
        self.metrics.record_hit()
        self.metrics.record_hit()
        self.metrics.record_miss()
        
        self.assertEqual(self.metrics.hits, 2)
        self.assertEqual(self.metrics.misses, 1)
        self.assertEqual(self.metrics.total_requests, 3)
        self.assertAlmostEqual(self.metrics.get_hit_rate(), 2/3, places=2)
    
    def test_batch_metrics(self):
        """Test batch metrics recording."""
        self.metrics.record_batch_hit(5, 0.001)
        self.metrics.record_batch_miss(3, 0.002)
        
        self.assertEqual(self.metrics.batch_hits, 5)
        self.assertEqual(self.metrics.batch_misses, 3)
        self.assertAlmostEqual(self.metrics.get_batch_hit_rate(), 5/8, places=2)
    
    def test_token_frequency_tracking(self):
        """Test token frequency tracking."""
        self.metrics.record_token_access(1)
        self.metrics.record_token_access(2)
        self.metrics.record_token_access(1)
        
        self.assertEqual(self.metrics.token_frequencies[1], 2)
        self.assertEqual(self.metrics.token_frequencies[2], 1)
        
        most_frequent = self.metrics.get_most_frequent_tokens(5)
        self.assertEqual(most_frequent[0], (1, 2))
        self.assertEqual(most_frequent[1], (2, 1))
    
    def test_recent_hit_rate(self):
        """Test recent hit rate calculation."""
        # Fill window with hits and misses
        for _ in range(5):
            self.metrics.record_hit()
        for _ in range(5):
            self.metrics.record_miss()
        
        self.assertEqual(self.metrics.get_recent_hit_rate(), 0.5)
    
    def test_metrics_summary(self):
        """Test comprehensive metrics summary."""
        self.metrics.record_hit(0.001)
        self.metrics.record_miss(0.002)
        self.metrics.record_batch_hit(3, 0.001)
        self.metrics.record_token_access(1)
        
        summary = self.metrics.get_summary()
        
        self.assertIn('cache_performance', summary)
        self.assertIn('performance', summary)
        self.assertIn('frequency_analysis', summary)
        self.assertEqual(summary['cache_performance']['hits'], 1)
        self.assertEqual(summary['cache_performance']['misses'], 1)


class TestLRUCache(unittest.TestCase):
    """Test LRU cache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = LRUCache(max_size=3)
        self.embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.embedding2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.embedding3 = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        self.embedding4 = np.array([10.0, 11.0, 12.0], dtype=np.float32)
    
    def test_initialization(self):
        """Test cache initialization."""
        self.assertEqual(self.cache.max_size, 3)
        self.assertEqual(self.cache.size(), 0)
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        self.cache.put(1, self.embedding1)
        retrieved = self.cache.get(1)
        
        self.assertIsNotNone(retrieved)
        np.testing.assert_array_equal(retrieved, self.embedding1)
        self.assertEqual(self.cache.size(), 1)
    
    def test_get_nonexistent(self):
        """Test getting non-existent key."""
        result = self.cache.get(999)
        self.assertIsNone(result)
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        # Fill cache to capacity
        self.cache.put(1, self.embedding1)
        self.cache.put(2, self.embedding2)
        self.cache.put(3, self.embedding3)
        
        # Access key 1 to make it most recently used
        self.cache.get(1)
        
        # Add new item - should evict key 2 (least recently used)
        self.cache.put(4, self.embedding4)
        
        self.assertIsNone(self.cache.get(2))  # Should be evicted
        self.assertIsNotNone(self.cache.get(1))  # Should still exist
        self.assertIsNotNone(self.cache.get(3))  # Should still exist
        self.assertIsNotNone(self.cache.get(4))  # Should exist
    
    def test_update_existing_key(self):
        """Test updating existing key."""
        self.cache.put(1, self.embedding1)
        self.cache.put(1, self.embedding2)  # Update
        
        retrieved = self.cache.get(1)
        np.testing.assert_array_equal(retrieved, self.embedding2)
        self.assertEqual(self.cache.size(), 1)
    
    def test_batch_operations(self):
        """Test batch get and put operations."""
        # Put batch
        batch_items = {1: self.embedding1, 2: self.embedding2}
        self.cache.put_batch(batch_items)
        
        # Get batch
        found_items, missing_keys = self.cache.get_batch([1, 2, 3])
        
        self.assertEqual(len(found_items), 2)
        self.assertEqual(missing_keys, [3])
        np.testing.assert_array_equal(found_items[1], self.embedding1)
        np.testing.assert_array_equal(found_items[2], self.embedding2)
    
    def test_contains(self):
        """Test contains method."""
        self.cache.put(1, self.embedding1)
        
        self.assertTrue(self.cache.contains(1))
        self.assertFalse(self.cache.contains(2))
    
    def test_clear(self):
        """Test cache clearing."""
        self.cache.put(1, self.embedding1)
        self.cache.put(2, self.embedding2)
        
        self.cache.clear()
        
        self.assertEqual(self.cache.size(), 0)
        self.assertIsNone(self.cache.get(1))
        self.assertIsNone(self.cache.get(2))
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        self.cache.put(1, self.embedding1)
        memory_usage = self.cache.get_memory_usage()
        
        self.assertGreater(memory_usage, 0)
    
    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        def worker(start_key):
            for i in range(10):
                key = start_key + i
                embedding = np.array([float(key)] * 3, dtype=np.float32)
                self.cache.put(key, embedding)
                retrieved = self.cache.get(key)
                if retrieved is not None:
                    np.testing.assert_array_equal(retrieved, embedding)
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=worker, args=(i * 100,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Cache should have some items (exact number depends on eviction)
        self.assertGreater(self.cache.size(), 0)


class TestBatchCache(unittest.TestCase):
    """Test batch-aware cache functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cache = BatchCache(max_size=5)
        self.embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.embedding2 = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        self.embedding3 = np.array([7.0, 8.0, 9.0], dtype=np.float32)
    
    def test_initialization(self):
        """Test batch cache initialization."""
        self.assertEqual(self.cache.max_size, 5)
        self.assertEqual(len(self.cache.cache), 0)
    
    def test_batch_deduplication(self):
        """Test batch deduplication functionality."""
        token_ids = [1, 2, 1, 3, 2, 1]  # Duplicates: 1 appears 3 times, 2 appears 2 times
        
        cached_embeddings, missing_tokens, token_positions = self.cache.get_batch_embeddings(
            token_ids, deduplication=True
        )
        
        # Should have unique tokens only
        self.assertEqual(set(missing_tokens), {1, 2, 3})
        
        # Check position mapping
        self.assertEqual(token_positions[1], [0, 2, 5])  # Positions of token 1
        self.assertEqual(token_positions[2], [1, 4])     # Positions of token 2
        self.assertEqual(token_positions[3], [3])        # Position of token 3
    
    def test_batch_without_deduplication(self):
        """Test batch processing without deduplication."""
        token_ids = [1, 2, 1, 3]
        
        cached_embeddings, missing_tokens, token_positions = self.cache.get_batch_embeddings(
            token_ids, deduplication=False
        )
        
        # Should treat each occurrence separately
        self.assertEqual(missing_tokens, token_ids)
        # Without deduplication, we still group by unique tokens but track all positions
        self.assertEqual(len(token_positions), 3)  # 3 unique tokens: 1, 2, 3
        self.assertEqual(token_positions[1], [0, 2])  # Token 1 at positions 0 and 2
        self.assertEqual(token_positions[2], [1])     # Token 2 at position 1
        self.assertEqual(token_positions[3], [3])     # Token 3 at position 3
    
    def test_batch_reconstruction(self):
        """Test batch reconstruction from cached embeddings."""
        # Cache some embeddings
        embeddings = {1: self.embedding1, 2: self.embedding2, 3: self.embedding3}
        self.cache.put_batch_embeddings(embeddings)
        
        # Define token positions
        token_positions = {1: [0, 2], 2: [1], 3: [3]}
        batch_size = 4
        embedding_dim = 3
        
        # Reconstruct batch
        batch_embeddings = self.cache.reconstruct_batch(
            embeddings, token_positions, batch_size, embedding_dim
        )
        
        self.assertEqual(batch_embeddings.shape, (4, 3))
        np.testing.assert_array_equal(batch_embeddings[0], self.embedding1)  # Position 0: token 1
        np.testing.assert_array_equal(batch_embeddings[1], self.embedding2)  # Position 1: token 2
        np.testing.assert_array_equal(batch_embeddings[2], self.embedding1)  # Position 2: token 1
        np.testing.assert_array_equal(batch_embeddings[3], self.embedding3)  # Position 3: token 3
    
    def test_frequency_based_eviction(self):
        """Test eviction based on access frequency."""
        # Fill cache to capacity first
        for i in range(5):  # Fill to max_size=5
            embedding = np.array([float(i)] * 3, dtype=np.float32)
            self.cache.put_batch_embeddings({i: embedding})
        
        # Cache should be at max capacity
        self.assertEqual(len(self.cache.cache), 5)
        
        # Access some tokens multiple times to increase their frequency
        for _ in range(5):
            self.cache.get_batch_embeddings([0, 1], deduplication=True)
        
        # Add more embeddings - should evict least frequent
        new_embeddings = {10: np.array([10.0] * 3, dtype=np.float32)}
        self.cache.put_batch_embeddings(new_embeddings)
        
        # Cache should still be at max capacity
        self.assertEqual(len(self.cache.cache), 5)
        
        # Tokens 0 and 1 should still be in cache due to high frequency
        self.assertIn(0, self.cache.cache)
        self.assertIn(1, self.cache.cache)
        self.assertIn(10, self.cache.cache)
    
    def test_clear(self):
        """Test cache clearing."""
        embeddings = {1: self.embedding1, 2: self.embedding2}
        self.cache.put_batch_embeddings(embeddings)
        
        self.cache.clear()
        
        self.assertEqual(len(self.cache.cache), 0)
        self.assertEqual(len(self.cache.access_counts), 0)
    
    def test_stats(self):
        """Test cache statistics."""
        embeddings = {1: self.embedding1, 2: self.embedding2}
        self.cache.put_batch_embeddings(embeddings)
        
        # Access tokens to generate statistics
        self.cache.get_batch_embeddings([1, 2, 1], deduplication=True)
        
        stats = self.cache.get_stats()
        
        self.assertEqual(stats['size'], 2)
        self.assertEqual(stats['max_size'], 5)
        self.assertGreater(stats['total_accesses'], 0)
        self.assertGreater(stats['avg_access_count'], 0)


class TestCacheWarmer(unittest.TestCase):
    """Test cache warming functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lru_cache = LRUCache(max_size=100)
        self.batch_cache = BatchCache(max_size=100)
        self.config = CacheConfig(warmup_size=10, warmup_threads=2)
        self.warmer = CacheWarmer(self.lru_cache, self.batch_cache, self.config)
        
        # Mock embedding getter
        self.embedding_getter = Mock()
        self.embedding_getter.return_value = np.random.rand(10, 3).astype(np.float32)
    
    def tearDown(self):
        """Clean up after tests."""
        self.warmer.shutdown()
    
    def test_frequency_based_warming(self):
        """Test cache warming based on token frequencies."""
        token_frequencies = {1: 100, 2: 50, 3: 25, 4: 10, 5: 5}
        
        self.warmer.warm_cache_from_frequencies(
            token_frequencies, self.embedding_getter, warmup_size=3
        )
        
        # Should have called embedding_getter
        self.assertTrue(self.embedding_getter.called)
        
        # Most frequent tokens should be in cache
        self.assertTrue(self.lru_cache.contains(1))
        self.assertTrue(self.lru_cache.contains(2))
        self.assertTrue(self.lru_cache.contains(3))
    
    def test_random_warming(self):
        """Test random cache warming."""
        vocab_size = 100
        
        self.warmer.warm_cache_random(vocab_size, self.embedding_getter, warmup_size=5)
        
        # Should have called embedding_getter
        self.assertTrue(self.embedding_getter.called)
        
        # Should have some items in cache
        self.assertGreater(self.lru_cache.size(), 0)
    
    def test_sequential_warming(self):
        """Test sequential cache warming."""
        vocab_size = 100
        
        self.warmer.warm_cache_sequential(vocab_size, self.embedding_getter, warmup_size=5)
        
        # Should have called embedding_getter
        self.assertTrue(self.embedding_getter.called)
        
        # Should have tokens 0-4 in cache
        for i in range(5):
            self.assertTrue(self.lru_cache.contains(i))


class TestIntelligentCachingSystem(unittest.TestCase):
    """Test the main intelligent caching system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = CacheConfig(
            max_cache_size=10,
            enable_batch_caching=True,
            batch_cache_size=10,
            enable_cache_warming=True,
            enable_metrics=True
        )
        self.caching_system = IntelligentCachingSystem(self.config)
        
        # Mock embedding getter
        self.embedding_getter = Mock()
        self.embedding_getter.side_effect = self._mock_embedding_getter
    
    def tearDown(self):
        """Clean up after tests."""
        self.caching_system.shutdown()
    
    def _mock_embedding_getter(self, token_ids):
        """Mock embedding getter function."""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        
        embeddings = []
        for token_id in token_ids:
            # Create deterministic embeddings based on token_id
            embedding = np.array([float(token_id), float(token_id) * 2, float(token_id) * 3], dtype=np.float32)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def test_initialization(self):
        """Test caching system initialization."""
        self.assertIsNotNone(self.caching_system.lru_cache)
        self.assertIsNotNone(self.caching_system.batch_cache)
        self.assertIsNotNone(self.caching_system.metrics)
        self.assertIsNotNone(self.caching_system.cache_warmer)
    
    def test_single_embedding_caching(self):
        """Test caching of single embeddings."""
        # First access - should be cache miss
        embedding1 = self.caching_system.get_embedding(1, self.embedding_getter)
        expected_embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(embedding1, expected_embedding)
        
        # Second access - should be cache hit
        embedding2 = self.caching_system.get_embedding(1, self.embedding_getter)
        np.testing.assert_array_equal(embedding2, expected_embedding)
        
        # Check metrics
        stats = self.caching_system.get_cache_stats()
        self.assertEqual(stats['metrics']['cache_performance']['total_requests'], 2)
        self.assertEqual(stats['metrics']['cache_performance']['hits'], 1)
        self.assertEqual(stats['metrics']['cache_performance']['misses'], 1)
    
    def test_batch_embedding_caching(self):
        """Test batch embedding caching."""
        token_ids = [1, 2, 3, 1, 2]  # With duplicates
        
        # First batch access
        embeddings1 = self.caching_system.get_batch_embeddings(token_ids, self.embedding_getter)
        
        self.assertEqual(embeddings1.shape, (5, 3))
        
        # Verify embeddings are correct
        expected_1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expected_2 = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        expected_3 = np.array([3.0, 6.0, 9.0], dtype=np.float32)
        
        np.testing.assert_array_equal(embeddings1[0], expected_1)
        np.testing.assert_array_equal(embeddings1[1], expected_2)
        np.testing.assert_array_equal(embeddings1[2], expected_3)
        np.testing.assert_array_equal(embeddings1[3], expected_1)  # Duplicate
        np.testing.assert_array_equal(embeddings1[4], expected_2)  # Duplicate
        
        # Second batch access - should have cache hits
        embeddings2 = self.caching_system.get_batch_embeddings([1, 2], self.embedding_getter)
        
        np.testing.assert_array_equal(embeddings2[0], expected_1)
        np.testing.assert_array_equal(embeddings2[1], expected_2)
    
    def test_cache_warming(self):
        """Test cache warming functionality."""
        token_frequencies = {1: 100, 2: 50, 3: 25}
        
        # Warm cache
        self.caching_system.warm_cache(
            self.embedding_getter, 
            token_frequencies=token_frequencies
        )
        
        # Check that frequent tokens are cached
        self.assertTrue(self.caching_system.lru_cache.contains(1))
        self.assertTrue(self.caching_system.lru_cache.contains(2))
        self.assertTrue(self.caching_system.lru_cache.contains(3))
    
    def test_cache_persistence(self):
        """Test cache persistence functionality."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            cache_file = tmp_file.name
        
        try:
            # Configure persistence
            self.caching_system.config.enable_persistence = True
            self.caching_system.config.cache_file_path = cache_file
            
            # Add some items to cache
            self.caching_system.get_embedding(1, self.embedding_getter)
            self.caching_system.get_embedding(2, self.embedding_getter)
            
            # Save cache state
            self.caching_system._save_cache_state()
            
            # Create new caching system and load state
            new_system = IntelligentCachingSystem(self.config)
            new_system.load_cache_state(cache_file)
            
            # Check that items are loaded
            self.assertTrue(new_system.lru_cache.contains(1))
            self.assertTrue(new_system.lru_cache.contains(2))
            
            new_system.shutdown()
        
        finally:
            if os.path.exists(cache_file):
                os.unlink(cache_file)
    
    def test_memory_management(self):
        """Test memory management and cleanup."""
        # Fill cache beyond memory threshold (simulate)
        for i in range(20):  # More than cache size
            self.caching_system.get_embedding(i, self.embedding_getter)
        
        # Trigger memory check
        self.caching_system._check_memory_usage()
        
        # Cache should still be functional
        embedding = self.caching_system.get_embedding(100, self.embedding_getter)
        self.assertIsNotNone(embedding)
    
    def test_cache_stats(self):
        """Test comprehensive cache statistics."""
        # Generate some cache activity
        self.caching_system.get_embedding(1, self.embedding_getter)
        self.caching_system.get_batch_embeddings([2, 3, 4], self.embedding_getter)
        
        stats = self.caching_system.get_cache_stats()
        
        # Check structure
        self.assertIn('lru_cache', stats)
        self.assertIn('batch_cache', stats)
        self.assertIn('metrics', stats)
        self.assertIn('config', stats)
        
        # Check values
        self.assertGreater(stats['lru_cache']['size'], 0)
        self.assertGreater(stats['metrics']['cache_performance']['total_requests'], 0)
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        # Add items to caches
        self.caching_system.get_embedding(1, self.embedding_getter)
        self.caching_system.get_batch_embeddings([2, 3], self.embedding_getter)
        
        # Clear all caches
        self.caching_system.clear_all_caches()
        
        # Check that caches are empty
        self.assertEqual(self.caching_system.lru_cache.size(), 0)
        self.assertEqual(len(self.caching_system.batch_cache.cache), 0)
    
    def test_config_variations(self):
        """Test different configuration options."""
        # Test with batch caching disabled
        config_no_batch = CacheConfig(enable_batch_caching=False)
        system_no_batch = IntelligentCachingSystem(config_no_batch)
        
        self.assertIsNone(system_no_batch.batch_cache)
        self.assertIsNone(system_no_batch.cache_warmer)
        
        system_no_batch.shutdown()
        
        # Test with metrics disabled
        config_no_metrics = CacheConfig(enable_metrics=False)
        system_no_metrics = IntelligentCachingSystem(config_no_metrics)
        
        self.assertIsNone(system_no_metrics.metrics)
        
        system_no_metrics.shutdown()


if __name__ == '__main__':
    unittest.main()