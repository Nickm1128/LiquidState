#!/usr/bin/env python3
"""
Intelligent Caching System Demo.

This demo shows how to use the intelligent caching system for enhanced
tokenizer embeddings with LRU cache, batch-aware caching, and cache warming.
"""

import numpy as np
import time
import random
from typing import List, Dict

# Add src to path for imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.lsm.data.intelligent_caching import (
    CacheConfig, IntelligentCachingSystem
)


class MockEmbeddingProvider:
    """
    Mock embedding provider to simulate expensive embedding computations.
    
    This simulates a real embedding layer that takes time to compute embeddings.
    """
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 128, 
                 computation_delay: float = 0.001):
        """
        Initialize mock embedding provider.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            computation_delay: Simulated computation delay per embedding
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.computation_delay = computation_delay
        self.computation_count = 0
        
        # Pre-generate embeddings for consistency
        np.random.seed(42)
        self.embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
        
        print(f"Initialized MockEmbeddingProvider: vocab_size={vocab_size}, "
              f"embedding_dim={embedding_dim}")
    
    def get_embeddings(self, token_ids: List[int]) -> np.ndarray:
        """
        Get embeddings for token IDs with simulated computation delay.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Embeddings array
        """
        # Simulate computation delay
        time.sleep(self.computation_delay * len(token_ids))
        self.computation_count += len(token_ids)
        
        # Return embeddings
        embeddings = self.embeddings[token_ids]
        return embeddings
    
    def get_computation_stats(self) -> Dict[str, int]:
        """Get computation statistics."""
        return {
            'total_computations': self.computation_count,
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim
        }


def demonstrate_basic_caching():
    """Demonstrate basic LRU caching functionality."""
    print("\n" + "="*60)
    print("BASIC LRU CACHING DEMONSTRATION")
    print("="*60)
    
    # Create caching system with basic configuration
    config = CacheConfig(
        max_cache_size=100,
        enable_batch_caching=False,  # Disable for this demo
        enable_cache_warming=False,
        enable_metrics=True
    )
    
    caching_system = IntelligentCachingSystem(config)
    embedding_provider = MockEmbeddingProvider(vocab_size=1000, embedding_dim=64)
    
    print(f"Created caching system with max_cache_size={config.max_cache_size}")
    
    # Test single embedding access
    print("\n1. Testing single embedding access:")
    
    token_id = 42
    
    # First access (cache miss)
    start_time = time.time()
    embedding1 = caching_system.get_embedding(token_id, embedding_provider.get_embeddings)
    first_access_time = time.time() - start_time
    
    # Second access (cache hit)
    start_time = time.time()
    embedding2 = caching_system.get_embedding(token_id, embedding_provider.get_embeddings)
    second_access_time = time.time() - start_time
    
    print(f"   First access (miss): {first_access_time*1000:.2f}ms")
    print(f"   Second access (hit): {second_access_time*1000:.2f}ms")
    if second_access_time > 0:
        print(f"   Speedup: {first_access_time/second_access_time:.1f}x")
    else:
        print(f"   Speedup: >1000x (cache hit was instantaneous)")
    
    # Verify embeddings are identical
    np.testing.assert_array_equal(embedding1, embedding2)
    print("   ✓ Embeddings are identical")
    
    # Test multiple accesses
    print("\n2. Testing multiple token access patterns:")
    
    # Access pattern with repeated tokens
    access_pattern = [1, 2, 3, 1, 4, 2, 5, 1, 3, 6]
    
    start_time = time.time()
    for token_id in access_pattern:
        embedding = caching_system.get_embedding(token_id, embedding_provider.get_embeddings)
    total_time = time.time() - start_time
    
    print(f"   Accessed {len(access_pattern)} tokens in {total_time*1000:.2f}ms")
    
    # Show cache statistics
    stats = caching_system.get_cache_stats()
    cache_perf = stats['metrics']['cache_performance']
    
    print(f"   Cache hit rate: {cache_perf['hit_rate']:.2%}")
    print(f"   Total requests: {cache_perf['total_requests']}")
    print(f"   Cache hits: {cache_perf['hits']}")
    print(f"   Cache misses: {cache_perf['misses']}")
    
    caching_system.shutdown()


def demonstrate_batch_caching():
    """Demonstrate batch-aware caching with deduplication."""
    print("\n" + "="*60)
    print("BATCH-AWARE CACHING DEMONSTRATION")
    print("="*60)
    
    # Create caching system with batch caching enabled
    config = CacheConfig(
        max_cache_size=100,
        enable_batch_caching=True,
        batch_cache_size=50,
        batch_deduplication=True,
        enable_metrics=True
    )
    
    caching_system = IntelligentCachingSystem(config)
    embedding_provider = MockEmbeddingProvider(vocab_size=1000, embedding_dim=64)
    
    print(f"Created batch-aware caching system")
    print(f"   LRU cache size: {config.max_cache_size}")
    print(f"   Batch cache size: {config.batch_cache_size}")
    print(f"   Deduplication enabled: {config.batch_deduplication}")
    
    # Test batch with duplicates
    print("\n1. Testing batch processing with duplicates:")
    
    # Create batch with many duplicates
    batch_tokens = [1, 2, 3, 1, 2, 4, 1, 5, 2, 3, 1, 6, 2]
    unique_tokens = len(set(batch_tokens))
    
    print(f"   Batch size: {len(batch_tokens)} tokens")
    print(f"   Unique tokens: {unique_tokens}")
    print(f"   Duplication ratio: {len(batch_tokens)/unique_tokens:.1f}x")
    
    # First batch access
    start_time = time.time()
    embeddings1 = caching_system.get_batch_embeddings(batch_tokens, embedding_provider.get_embeddings)
    first_batch_time = time.time() - start_time
    
    # Second batch access (should be faster due to caching)
    start_time = time.time()
    embeddings2 = caching_system.get_batch_embeddings(batch_tokens, embedding_provider.get_embeddings)
    second_batch_time = time.time() - start_time
    
    print(f"   First batch: {first_batch_time*1000:.2f}ms")
    print(f"   Second batch: {second_batch_time*1000:.2f}ms")
    if second_batch_time > 0:
        print(f"   Speedup: {first_batch_time/second_batch_time:.1f}x")
    else:
        print(f"   Speedup: >1000x (cached batch was instantaneous)")
    
    # Verify embeddings are identical
    np.testing.assert_array_equal(embeddings1, embeddings2)
    print("   ✓ Batch embeddings are identical")
    
    # Test overlapping batches
    print("\n2. Testing overlapping batch patterns:")
    
    batch1 = [1, 2, 3, 4, 5]
    batch2 = [3, 4, 5, 6, 7]  # Overlaps with batch1
    batch3 = [1, 6, 8, 9, 10]  # Partially overlaps
    
    # Process batches
    start_time = time.time()
    emb1 = caching_system.get_batch_embeddings(batch1, embedding_provider.get_embeddings)
    emb2 = caching_system.get_batch_embeddings(batch2, embedding_provider.get_embeddings)
    emb3 = caching_system.get_batch_embeddings(batch3, embedding_provider.get_embeddings)
    total_time = time.time() - start_time
    
    print(f"   Processed 3 overlapping batches in {total_time*1000:.2f}ms")
    
    # Show batch cache statistics
    stats = caching_system.get_cache_stats()
    batch_perf = stats['metrics']['cache_performance']
    
    print(f"   Batch hit rate: {batch_perf['batch_hit_rate']:.2%}")
    print(f"   Batch hits: {batch_perf['batch_hits']}")
    print(f"   Batch misses: {batch_perf['batch_misses']}")
    
    caching_system.shutdown()


def demonstrate_cache_warming():
    """Demonstrate cache warming strategies."""
    print("\n" + "="*60)
    print("CACHE WARMING DEMONSTRATION")
    print("="*60)
    
    # Create caching system with cache warming enabled
    config = CacheConfig(
        max_cache_size=200,
        enable_batch_caching=True,
        enable_cache_warming=True,
        warmup_strategy="frequency",
        warmup_size=50,
        warmup_threads=2,
        enable_metrics=True
    )
    
    caching_system = IntelligentCachingSystem(config)
    embedding_provider = MockEmbeddingProvider(vocab_size=1000, embedding_dim=64)
    
    print(f"Created caching system with cache warming")
    print(f"   Warmup strategy: {config.warmup_strategy}")
    print(f"   Warmup size: {config.warmup_size}")
    print(f"   Warmup threads: {config.warmup_threads}")
    
    # Simulate token frequency data (Zipfian distribution)
    print("\n1. Generating token frequency data (Zipfian distribution):")
    
    vocab_size = 1000
    token_frequencies = {}
    
    for i in range(vocab_size):
        # Zipfian distribution: frequency inversely proportional to rank
        frequency = int(1000 / (i + 1))
        if frequency > 0:
            token_frequencies[i] = frequency
    
    top_tokens = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"   Top 10 most frequent tokens: {top_tokens}")
    
    # Warm cache based on frequencies
    print("\n2. Warming cache with most frequent tokens:")
    
    start_time = time.time()
    caching_system.warm_cache(
        embedding_provider.get_embeddings,
        vocab_size=vocab_size,
        token_frequencies=token_frequencies
    )
    warmup_time = time.time() - start_time
    
    print(f"   Cache warming completed in {warmup_time*1000:.2f}ms")
    print(f"   Cache size after warming: {caching_system.lru_cache.size()}")
    
    # Test performance with warmed cache
    print("\n3. Testing performance with warmed cache:")
    
    # Create test batches with frequent tokens
    frequent_tokens = [token_id for token_id, _ in top_tokens[:20]]
    test_batch = []
    
    for _ in range(100):
        # 80% chance of frequent token, 20% chance of random token
        if random.random() < 0.8:
            test_batch.append(random.choice(frequent_tokens))
        else:
            test_batch.append(random.randint(100, 999))
    
    # Process test batch
    start_time = time.time()
    embeddings = caching_system.get_batch_embeddings(test_batch, embedding_provider.get_embeddings)
    processing_time = time.time() - start_time
    
    print(f"   Processed {len(test_batch)} tokens in {processing_time*1000:.2f}ms")
    
    # Show final statistics
    stats = caching_system.get_cache_stats()
    cache_perf = stats['metrics']['cache_performance']
    
    print(f"   Final cache hit rate: {cache_perf['hit_rate']:.2%}")
    print(f"   Total cache requests: {cache_perf['total_requests']}")
    
    caching_system.shutdown()


def demonstrate_performance_comparison():
    """Compare performance with and without caching."""
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    embedding_provider = MockEmbeddingProvider(vocab_size=1000, embedding_dim=128, 
                                             computation_delay=0.002)  # Slower for demo
    
    # Test without caching
    print("1. Testing WITHOUT caching:")
    
    test_tokens = []
    for _ in range(200):
        # Create realistic access pattern with some repetition
        if random.random() < 0.3:  # 30% chance of repeat
            test_tokens.append(random.choice(test_tokens) if test_tokens else random.randint(0, 99))
        else:
            test_tokens.append(random.randint(0, 99))
    
    start_time = time.time()
    computation_count_before = embedding_provider.computation_count
    
    embeddings_no_cache = []
    for token_id in test_tokens:
        embedding = embedding_provider.get_embeddings([token_id])
        embeddings_no_cache.append(embedding[0])
    
    no_cache_time = time.time() - start_time
    computations_no_cache = embedding_provider.computation_count - computation_count_before
    
    print(f"   Time: {no_cache_time*1000:.2f}ms")
    print(f"   Computations: {computations_no_cache}")
    print(f"   Unique tokens: {len(set(test_tokens))}")
    
    # Test with caching
    print("\n2. Testing WITH intelligent caching:")
    
    config = CacheConfig(
        max_cache_size=100,
        enable_batch_caching=True,
        enable_cache_warming=True,
        warmup_strategy="frequency",
        enable_metrics=True
    )
    
    caching_system = IntelligentCachingSystem(config)
    
    # Warm cache with frequent tokens
    token_frequencies = {}
    for token_id in test_tokens:
        token_frequencies[token_id] = token_frequencies.get(token_id, 0) + 1
    
    caching_system.warm_cache(
        embedding_provider.get_embeddings,
        token_frequencies=token_frequencies
    )
    
    start_time = time.time()
    computation_count_before = embedding_provider.computation_count
    
    embeddings_with_cache = []
    for token_id in test_tokens:
        embedding = caching_system.get_embedding(token_id, embedding_provider.get_embeddings)
        embeddings_with_cache.append(embedding)
    
    with_cache_time = time.time() - start_time
    computations_with_cache = embedding_provider.computation_count - computation_count_before
    
    print(f"   Time: {with_cache_time*1000:.2f}ms")
    print(f"   Computations: {computations_with_cache}")
    
    # Show comparison
    print("\n3. Performance comparison:")
    if with_cache_time > 0:
        speedup = no_cache_time / with_cache_time
        print(f"   Speedup: {speedup:.1f}x")
    else:
        print(f"   Speedup: >1000x (cached processing was instantaneous)")
    
    computation_reduction = (computations_no_cache - computations_with_cache) / computations_no_cache
    print(f"   Computation reduction: {computation_reduction:.1%}")
    
    # Verify results are identical
    for i, (emb1, emb2) in enumerate(zip(embeddings_no_cache, embeddings_with_cache)):
        np.testing.assert_array_almost_equal(emb1, emb2, decimal=5)
    
    print("   ✓ Results are identical")
    
    # Show final cache statistics
    stats = caching_system.get_cache_stats()
    print(f"\n4. Final cache statistics:")
    print(f"   Cache hit rate: {stats['metrics']['cache_performance']['hit_rate']:.2%}")
    print(f"   LRU cache utilization: {stats['lru_cache']['utilization']:.1%}")
    print(f"   Memory usage: {stats['lru_cache']['memory_usage_mb']:.2f} MB")
    
    caching_system.shutdown()


def demonstrate_advanced_features():
    """Demonstrate advanced caching features."""
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    # Create caching system with all features enabled
    config = CacheConfig(
        max_cache_size=100,
        enable_batch_caching=True,
        batch_cache_size=50,
        enable_cache_warming=True,
        enable_metrics=True,
        enable_persistence=False,  # Disabled for demo
        memory_threshold_mb=10.0,
        cleanup_ratio=0.3
    )
    
    caching_system = IntelligentCachingSystem(config)
    embedding_provider = MockEmbeddingProvider(vocab_size=500, embedding_dim=64)
    
    print("1. Testing memory management:")
    
    # Fill cache beyond capacity to trigger eviction
    for i in range(150):  # More than max_cache_size
        embedding = caching_system.get_embedding(i, embedding_provider.get_embeddings)
    
    print(f"   Added 150 embeddings to cache with max_size=100")
    print(f"   Current cache size: {caching_system.lru_cache.size()}")
    print(f"   Memory usage: {caching_system.lru_cache.get_memory_usage():.2f} MB")
    
    # Test batch patterns
    print("\n2. Testing batch pattern recognition:")
    
    # Create repeating batch patterns
    pattern1 = [1, 2, 3, 4, 5]
    pattern2 = [10, 11, 12, 13]
    
    for _ in range(5):  # Repeat patterns
        caching_system.get_batch_embeddings(pattern1, embedding_provider.get_embeddings)
        caching_system.get_batch_embeddings(pattern2, embedding_provider.get_embeddings)
    
    # Show metrics
    stats = caching_system.get_cache_stats()
    metrics = stats['metrics']
    
    print(f"   Batch hit rate: {metrics['cache_performance']['batch_hit_rate']:.2%}")
    print(f"   Unique batch patterns: {metrics['frequency_analysis']['unique_batch_patterns']}")
    
    # Test cache warming strategies
    print("\n3. Testing different warming strategies:")
    
    strategies = ["frequency", "random", "sequential"]
    
    for strategy in strategies:
        # Clear cache
        caching_system.clear_all_caches()
        
        # Update config
        caching_system.config.warmup_strategy = strategy
        caching_system.config.warmup_size = 20
        
        start_time = time.time()
        
        if strategy == "frequency":
            token_frequencies = {i: 100 - i for i in range(50)}  # Decreasing frequency
            caching_system.warm_cache(
                embedding_provider.get_embeddings,
                token_frequencies=token_frequencies
            )
        else:
            caching_system.warm_cache(
                embedding_provider.get_embeddings,
                vocab_size=500
            )
        
        warmup_time = time.time() - start_time
        cache_size = caching_system.lru_cache.size()
        
        print(f"   {strategy.capitalize()} warming: {warmup_time*1000:.2f}ms, "
              f"cached {cache_size} embeddings")
    
    # Show comprehensive statistics
    print("\n4. Comprehensive statistics:")
    final_stats = caching_system.get_cache_stats()
    
    print(f"   Total operations: {final_stats['operation_count']}")
    print(f"   LRU cache stats: {final_stats['lru_cache']}")
    print(f"   Batch cache stats: {final_stats['batch_cache']}")
    
    # Show embedding provider statistics
    provider_stats = embedding_provider.get_computation_stats()
    print(f"   Total computations: {provider_stats['total_computations']}")
    
    caching_system.shutdown()


def main():
    """Run all demonstrations."""
    print("INTELLIGENT CACHING SYSTEM DEMONSTRATION")
    print("This demo shows the capabilities of the intelligent caching system")
    print("for enhanced tokenizer embeddings.")
    
    try:
        demonstrate_basic_caching()
        demonstrate_batch_caching()
        demonstrate_cache_warming()
        demonstrate_performance_comparison()
        demonstrate_advanced_features()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nKey takeaways:")
        print("• LRU caching provides significant speedup for repeated token access")
        print("• Batch-aware caching optimizes duplicate token handling within batches")
        print("• Cache warming preloads frequent tokens for better initial performance")
        print("• Intelligent caching reduces both computation time and redundant operations")
        print("• Memory management ensures stable performance under varying loads")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()