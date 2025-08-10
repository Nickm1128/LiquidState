#!/usr/bin/env python3
"""
Intelligent caching system for enhanced tokenizer embeddings.

This module provides LRU cache for frequently accessed token embeddings,
batch-aware caching to avoid redundant computations, and cache warming
and preloading strategies for optimal performance.
"""

import os
import json
import time
import threading
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Union, List, Tuple, Set
from dataclasses import dataclass
from collections import OrderedDict
from abc import ABC, abstractmethod
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils.lsm_exceptions import (
    TokenizerError, TokenizerNotFittedError, 
    TokenizerLoadError, TokenizerSaveError, InvalidInputError
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheConfig:
    """Configuration for intelligent caching system."""
    
    # LRU Cache settings
    max_cache_size: int = 10000  # Maximum number of cached embeddings
    cache_hit_threshold: float = 0.7  # Hit rate threshold for cache effectiveness
    
    # Batch-aware caching
    enable_batch_caching: bool = True
    batch_cache_size: int = 5000  # Size of batch-specific cache
    batch_deduplication: bool = True  # Remove duplicates within batches
    
    # Cache warming and preloading
    enable_cache_warming: bool = True
    warmup_strategy: str = "frequency"  # "frequency", "random", "sequential"
    warmup_size: int = 1000  # Number of embeddings to preload
    warmup_threads: int = 2  # Number of threads for parallel warming
    
    # Performance monitoring
    enable_metrics: bool = True
    metrics_window_size: int = 1000  # Window size for performance metrics
    
    # Cache persistence
    enable_persistence: bool = False
    cache_file_path: Optional[str] = None
    save_interval: int = 100  # Save cache every N operations
    
    # Memory management
    memory_threshold_mb: float = 500.0  # Memory threshold for cache eviction
    cleanup_ratio: float = 0.2  # Ratio of cache to clean when threshold exceeded


class CacheMetrics:
    """Metrics tracking for cache performance."""
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize cache metrics.
        
        Args:
            window_size: Size of sliding window for metrics
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.batch_hits = 0
        self.batch_misses = 0
        self.recent_requests = []
        self.start_time = time.time()
        
        # Performance metrics
        self.avg_lookup_time = 0.0
        self.avg_batch_lookup_time = 0.0
        self.memory_usage_mb = 0.0
        
        # Frequency tracking
        self.token_frequencies = {}
        self.batch_patterns = {}
    
    def record_hit(self, lookup_time: float = 0.0):
        """Record a cache hit."""
        self.hits += 1
        self.total_requests += 1
        self._update_lookup_time(lookup_time)
        self._update_recent_requests(True)
    
    def record_miss(self, lookup_time: float = 0.0):
        """Record a cache miss."""
        self.misses += 1
        self.total_requests += 1
        self._update_lookup_time(lookup_time)
        self._update_recent_requests(False)
    
    def record_batch_hit(self, batch_size: int, lookup_time: float = 0.0):
        """Record a batch cache hit."""
        self.batch_hits += batch_size
        self._update_batch_lookup_time(lookup_time)
    
    def record_batch_miss(self, batch_size: int, lookup_time: float = 0.0):
        """Record a batch cache miss."""
        self.batch_misses += batch_size
        self._update_batch_lookup_time(lookup_time)
    
    def record_token_access(self, token_id: int):
        """Record access to a specific token."""
        self.token_frequencies[token_id] = self.token_frequencies.get(token_id, 0) + 1
    
    def record_batch_pattern(self, batch_hash: str, batch_size: int):
        """Record a batch access pattern."""
        if batch_hash not in self.batch_patterns:
            self.batch_patterns[batch_hash] = {'count': 0, 'size': batch_size}
        self.batch_patterns[batch_hash]['count'] += 1
    
    def _update_lookup_time(self, lookup_time: float):
        """Update average lookup time."""
        if self.total_requests == 1:
            self.avg_lookup_time = lookup_time
        else:
            alpha = 0.1  # Exponential moving average factor
            self.avg_lookup_time = (1 - alpha) * self.avg_lookup_time + alpha * lookup_time
    
    def _update_batch_lookup_time(self, lookup_time: float):
        """Update average batch lookup time."""
        total_batch_requests = self.batch_hits + self.batch_misses
        if total_batch_requests == 1:
            self.avg_batch_lookup_time = lookup_time
        else:
            alpha = 0.1
            self.avg_batch_lookup_time = (1 - alpha) * self.avg_batch_lookup_time + alpha * lookup_time
    
    def _update_recent_requests(self, is_hit: bool):
        """Update recent requests sliding window."""
        self.recent_requests.append(is_hit)
        if len(self.recent_requests) > self.window_size:
            self.recent_requests.pop(0)
    
    def get_hit_rate(self) -> float:
        """Get overall cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    def get_recent_hit_rate(self) -> float:
        """Get recent cache hit rate."""
        if not self.recent_requests:
            return 0.0
        return sum(self.recent_requests) / len(self.recent_requests)
    
    def get_batch_hit_rate(self) -> float:
        """Get batch cache hit rate."""
        total_batch_requests = self.batch_hits + self.batch_misses
        if total_batch_requests == 0:
            return 0.0
        return self.batch_hits / total_batch_requests
    
    def get_most_frequent_tokens(self, top_k: int = 100) -> List[Tuple[int, int]]:
        """Get most frequently accessed tokens."""
        return sorted(self.token_frequencies.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    def get_most_frequent_batch_patterns(self, top_k: int = 50) -> List[Tuple[str, Dict]]:
        """Get most frequent batch patterns."""
        return sorted(self.batch_patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:top_k]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        runtime = time.time() - self.start_time
        
        return {
            'cache_performance': {
                'hit_rate': self.get_hit_rate(),
                'recent_hit_rate': self.get_recent_hit_rate(),
                'batch_hit_rate': self.get_batch_hit_rate(),
                'total_requests': self.total_requests,
                'hits': self.hits,
                'misses': self.misses,
                'batch_hits': self.batch_hits,
                'batch_misses': self.batch_misses
            },
            'performance': {
                'avg_lookup_time_ms': self.avg_lookup_time * 1000,
                'avg_batch_lookup_time_ms': self.avg_batch_lookup_time * 1000,
                'requests_per_second': self.total_requests / runtime if runtime > 0 else 0,
                'memory_usage_mb': self.memory_usage_mb
            },
            'frequency_analysis': {
                'unique_tokens_accessed': len(self.token_frequencies),
                'unique_batch_patterns': len(self.batch_patterns),
                'most_frequent_tokens': self.get_most_frequent_tokens(10),
                'most_frequent_batch_patterns': len(self.get_most_frequent_batch_patterns(5))
            },
            'runtime_seconds': runtime
        }


class LRUCache:
    """
    Least Recently Used (LRU) cache for token embeddings.
    
    This cache maintains frequently accessed embeddings in memory
    with automatic eviction of least recently used items.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to cache
        """
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        
        logger.info(f"Initialized LRU cache with max_size={max_size}")
    
    def get(self, key: int) -> Optional[np.ndarray]:
        """
        Get item from cache.
        
        Args:
            key: Cache key (token ID)
            
        Returns:
            Cached embedding or None if not found
        """
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                return value.copy()  # Return copy to prevent modification
            return None
    
    def put(self, key: int, value: np.ndarray):
        """
        Put item in cache.
        
        Args:
            key: Cache key (token ID)
            value: Embedding to cache
        """
        with self.lock:
            if key in self.cache:
                # Update existing item and move to end
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used item
                self.cache.popitem(last=False)
            
            # Add new item
            self.cache[key] = value.copy()
    
    def get_batch(self, keys: List[int]) -> Tuple[Dict[int, np.ndarray], List[int]]:
        """
        Get multiple items from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Tuple of (found_items, missing_keys)
        """
        found_items = {}
        missing_keys = []
        
        with self.lock:
            for key in keys:
                if key in self.cache:
                    # Move to end and get value
                    value = self.cache.pop(key)
                    self.cache[key] = value
                    found_items[key] = value.copy()
                else:
                    missing_keys.append(key)
        
        return found_items, missing_keys
    
    def put_batch(self, items: Dict[int, np.ndarray]):
        """
        Put multiple items in cache.
        
        Args:
            items: Dictionary of key-value pairs to cache
        """
        with self.lock:
            for key, value in items.items():
                if key in self.cache:
                    self.cache.pop(key)
                elif len(self.cache) >= self.max_size:
                    # Remove least recently used items
                    num_to_remove = len(self.cache) + len(items) - self.max_size
                    for _ in range(max(1, num_to_remove)):
                        if self.cache:
                            self.cache.popitem(last=False)
                
                self.cache[key] = value.copy()
    
    def contains(self, key: int) -> bool:
        """Check if key is in cache."""
        with self.lock:
            return key in self.cache
    
    def size(self) -> int:
        """Get current cache size."""
        with self.lock:
            return len(self.cache)
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
    
    def get_memory_usage(self) -> float:
        """
        Get approximate memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        with self.lock:
            if not self.cache:
                return 0.0
            
            # Estimate memory usage
            sample_value = next(iter(self.cache.values()))
            bytes_per_embedding = sample_value.nbytes
            total_bytes = len(self.cache) * bytes_per_embedding
            return total_bytes / (1024 * 1024)  # Convert to MB
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_mb': self.get_memory_usage(),
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
            }


class BatchCache:
    """
    Batch-aware cache that handles batch patterns and deduplication.
    
    This cache is optimized for batch processing scenarios where
    the same tokens appear multiple times within a batch.
    """
    
    def __init__(self, max_size: int = 5000):
        """
        Initialize batch cache.
        
        Args:
            max_size: Maximum number of unique embeddings to cache
        """
        self.max_size = max_size
        self.cache = {}
        self.access_counts = {}
        self.lock = threading.RLock()
        
        logger.info(f"Initialized batch cache with max_size={max_size}")
    
    def get_batch_embeddings(self, token_ids: List[int], 
                           deduplication: bool = True) -> Tuple[np.ndarray, List[int], Dict[int, int]]:
        """
        Get embeddings for a batch of tokens with deduplication.
        
        Args:
            token_ids: List of token IDs
            deduplication: Whether to deduplicate tokens within batch
            
        Returns:
            Tuple of (cached_embeddings, missing_token_ids, token_positions)
        """
        if deduplication:
            # Create mapping of unique tokens to their positions
            unique_tokens = []
            token_positions = {}
            
            for i, token_id in enumerate(token_ids):
                if token_id not in token_positions:
                    token_positions[token_id] = []
                    unique_tokens.append(token_id)
                token_positions[token_id].append(i)
        else:
            # Without deduplication, treat each occurrence separately
            unique_tokens = token_ids
            token_positions = {}
            for i, token_id in enumerate(token_ids):
                if token_id not in token_positions:
                    token_positions[token_id] = []
                token_positions[token_id].append(i)
        
        # Get cached embeddings
        cached_embeddings = {}
        missing_tokens = []
        
        with self.lock:
            for token_id in unique_tokens:
                if token_id in self.cache:
                    cached_embeddings[token_id] = self.cache[token_id].copy()
                    self.access_counts[token_id] = self.access_counts.get(token_id, 0) + 1
                else:
                    missing_tokens.append(token_id)
        
        return cached_embeddings, missing_tokens, token_positions
    
    def put_batch_embeddings(self, embeddings: Dict[int, np.ndarray]):
        """
        Cache embeddings for multiple tokens.
        
        Args:
            embeddings: Dictionary mapping token IDs to embeddings
        """
        with self.lock:
            # Check if we need to evict items
            new_items = [token_id for token_id in embeddings.keys() if token_id not in self.cache]
            if len(self.cache) + len(new_items) > self.max_size:
                num_to_evict = len(self.cache) + len(new_items) - self.max_size
                self._evict_least_frequent(num_to_evict)
            
            # Add new embeddings
            for token_id, embedding in embeddings.items():
                self.cache[token_id] = embedding.copy()
                self.access_counts[token_id] = self.access_counts.get(token_id, 0) + 1
    
    def _evict_least_frequent(self, num_to_evict: int):
        """
        Evict least frequently used items to make space.
        
        Args:
            num_to_evict: Number of items to evict
        """
        if num_to_evict <= 0:
            return
        
        # Sort by access count (ascending) and evict least frequent
        sorted_items = sorted(self.access_counts.items(), key=lambda x: x[1])
        
        evicted_count = 0
        for token_id, _ in sorted_items:
            if evicted_count >= num_to_evict:
                break
                
            if token_id in self.cache:
                del self.cache[token_id]
                evicted_count += 1
            if token_id in self.access_counts:
                del self.access_counts[token_id]
    
    def reconstruct_batch(self, cached_embeddings: Dict[int, np.ndarray],
                         token_positions: Dict[int, List[int]],
                         batch_size: int, embedding_dim: int) -> np.ndarray:
        """
        Reconstruct full batch embeddings from cached unique embeddings.
        
        Args:
            cached_embeddings: Dictionary of cached embeddings
            token_positions: Mapping of token IDs to their positions in batch
            batch_size: Size of the original batch
            embedding_dim: Embedding dimension
            
        Returns:
            Reconstructed batch embeddings
        """
        batch_embeddings = np.zeros((batch_size, embedding_dim), dtype=np.float32)
        
        for token_id, positions in token_positions.items():
            if token_id in cached_embeddings:
                embedding = cached_embeddings[token_id]
                for pos in positions:
                    batch_embeddings[pos] = embedding
        
        return batch_embeddings
    
    def clear(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch cache statistics."""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            avg_access_count = total_accesses / len(self.access_counts) if self.access_counts else 0
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'total_accesses': total_accesses,
                'avg_access_count': avg_access_count,
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
            }


class CacheWarmer:
    """
    Cache warming and preloading system.
    
    This class implements strategies to preload frequently used embeddings
    into cache to improve performance.
    """
    
    def __init__(self, cache: LRUCache, batch_cache: BatchCache, 
                 config: CacheConfig):
        """
        Initialize cache warmer.
        
        Args:
            cache: LRU cache to warm
            batch_cache: Batch cache to warm
            config: Cache configuration
        """
        self.cache = cache
        self.batch_cache = batch_cache
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.warmup_threads)
        
        logger.info(f"Initialized cache warmer with strategy={config.warmup_strategy}")
    
    def warm_cache_from_frequencies(self, token_frequencies: Dict[int, int],
                                  embedding_getter: callable,
                                  warmup_size: Optional[int] = None):
        """
        Warm cache based on token frequencies.
        
        Args:
            token_frequencies: Dictionary mapping token IDs to frequencies
            embedding_getter: Function to get embeddings for token IDs
            warmup_size: Number of embeddings to preload (uses config default if None)
        """
        warmup_size = warmup_size or self.config.warmup_size
        
        # Get most frequent tokens
        most_frequent = sorted(token_frequencies.items(), key=lambda x: x[1], reverse=True)
        tokens_to_warm = [token_id for token_id, _ in most_frequent[:warmup_size]]
        
        logger.info(f"Warming cache with {len(tokens_to_warm)} most frequent tokens")
        
        # Preload embeddings in parallel
        self._preload_embeddings_parallel(tokens_to_warm, embedding_getter)
    
    def warm_cache_random(self, vocab_size: int, embedding_getter: callable,
                         warmup_size: Optional[int] = None):
        """
        Warm cache with random tokens.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_getter: Function to get embeddings for token IDs
            warmup_size: Number of embeddings to preload
        """
        warmup_size = warmup_size or self.config.warmup_size
        
        # Generate random token IDs
        import random
        tokens_to_warm = random.sample(range(vocab_size), min(warmup_size, vocab_size))
        
        logger.info(f"Warming cache with {len(tokens_to_warm)} random tokens")
        
        # Preload embeddings in parallel
        self._preload_embeddings_parallel(tokens_to_warm, embedding_getter)
    
    def warm_cache_sequential(self, vocab_size: int, embedding_getter: callable,
                            warmup_size: Optional[int] = None):
        """
        Warm cache with sequential tokens (0, 1, 2, ...).
        
        Args:
            vocab_size: Size of vocabulary
            embedding_getter: Function to get embeddings for token IDs
            warmup_size: Number of embeddings to preload
        """
        warmup_size = warmup_size or self.config.warmup_size
        
        # Generate sequential token IDs
        tokens_to_warm = list(range(min(warmup_size, vocab_size)))
        
        logger.info(f"Warming cache with {len(tokens_to_warm)} sequential tokens")
        
        # Preload embeddings in parallel
        self._preload_embeddings_parallel(tokens_to_warm, embedding_getter)
    
    def _preload_embeddings_parallel(self, token_ids: List[int], embedding_getter: callable):
        """
        Preload embeddings in parallel using thread pool.
        
        Args:
            token_ids: List of token IDs to preload
            embedding_getter: Function to get embeddings
        """
        # Split token IDs into chunks for parallel processing
        chunk_size = max(1, len(token_ids) // self.config.warmup_threads)
        chunks = [token_ids[i:i + chunk_size] for i in range(0, len(token_ids), chunk_size)]
        
        # Submit preloading tasks
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._preload_chunk, chunk, embedding_getter)
            futures.append(future)
        
        # Wait for completion
        completed = 0
        for future in as_completed(futures):
            try:
                chunk_size = future.result()
                completed += chunk_size
                logger.debug(f"Preloaded {completed}/{len(token_ids)} embeddings")
            except Exception as e:
                logger.warning(f"Error preloading embeddings chunk: {e}")
        
        logger.info(f"Cache warming completed: {completed}/{len(token_ids)} embeddings preloaded")
    
    def _preload_chunk(self, token_ids: List[int], embedding_getter: callable) -> int:
        """
        Preload a chunk of embeddings.
        
        Args:
            token_ids: List of token IDs to preload
            embedding_getter: Function to get embeddings
            
        Returns:
            Number of embeddings successfully preloaded
        """
        try:
            # Get embeddings for this chunk
            embeddings = embedding_getter(token_ids)
            
            # Add to both caches
            if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
                embedding_dict = {token_id: embeddings[i] for i, token_id in enumerate(token_ids)}
                self.cache.put_batch(embedding_dict)
                self.batch_cache.put_batch_embeddings(embedding_dict)
            
            return len(token_ids)
        except Exception as e:
            logger.warning(f"Error preloading chunk: {e}")
            return 0
    
    def shutdown(self):
        """Shutdown the thread pool executor."""
        self.executor.shutdown(wait=True)


class IntelligentCachingSystem:
    """
    Main intelligent caching system that coordinates all caching strategies.
    
    This system combines LRU caching, batch-aware caching, and cache warming
    to provide optimal performance for token embedding access patterns.
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        Initialize intelligent caching system.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        
        # Initialize caches
        self.lru_cache = LRUCache(self.config.max_cache_size)
        self.batch_cache = BatchCache(self.config.batch_cache_size) if self.config.enable_batch_caching else None
        
        # Initialize metrics
        self.metrics = CacheMetrics(self.config.metrics_window_size) if self.config.enable_metrics else None
        
        # Initialize cache warmer
        self.cache_warmer = CacheWarmer(self.lru_cache, self.batch_cache, self.config) if self.batch_cache else None
        
        # Persistence
        self.operation_count = 0
        
        logger.info(f"Initialized intelligent caching system with config: {self.config}")
    
    def get_embedding(self, token_id: int, embedding_getter: callable) -> np.ndarray:
        """
        Get embedding for a single token with caching.
        
        Args:
            token_id: Token ID to get embedding for
            embedding_getter: Function to get embedding if not cached
            
        Returns:
            Token embedding
        """
        start_time = time.time()
        
        # Try LRU cache first
        embedding = self.lru_cache.get(token_id)
        
        if embedding is not None:
            # Cache hit
            if self.metrics:
                self.metrics.record_hit(time.time() - start_time)
                self.metrics.record_token_access(token_id)
            return embedding
        
        # Cache miss - get embedding and cache it
        embedding = embedding_getter([token_id])
        if isinstance(embedding, np.ndarray):
            if embedding.ndim == 2:
                embedding = embedding[0]  # Extract single embedding
            self.lru_cache.put(token_id, embedding)
        
        if self.metrics:
            self.metrics.record_miss(time.time() - start_time)
            self.metrics.record_token_access(token_id)
        
        self._update_operation_count()
        return embedding
    
    def get_batch_embeddings(self, token_ids: List[int], 
                           embedding_getter: callable) -> np.ndarray:
        """
        Get embeddings for a batch of tokens with intelligent caching.
        
        Args:
            token_ids: List of token IDs
            embedding_getter: Function to get embeddings for missing tokens
            
        Returns:
            Batch of embeddings
        """
        start_time = time.time()
        batch_size = len(token_ids)
        
        if not token_ids:
            return np.array([])
        
        # Create batch hash for pattern tracking
        batch_hash = self._create_batch_hash(token_ids) if self.metrics else None
        
        if self.batch_cache and self.config.enable_batch_caching:
            # Use batch-aware caching
            cached_embeddings, missing_tokens, token_positions = self.batch_cache.get_batch_embeddings(
                token_ids, self.config.batch_deduplication
            )
            
            # Get missing embeddings
            if missing_tokens:
                missing_embeddings_array = embedding_getter(missing_tokens)
                if isinstance(missing_embeddings_array, np.ndarray) and missing_embeddings_array.ndim == 2:
                    missing_embeddings = {
                        token_id: missing_embeddings_array[i] 
                        for i, token_id in enumerate(missing_tokens)
                    }
                    
                    # Cache the missing embeddings
                    self.batch_cache.put_batch_embeddings(missing_embeddings)
                    self.lru_cache.put_batch(missing_embeddings)
                    
                    # Merge with cached embeddings
                    cached_embeddings.update(missing_embeddings)
            
            # Reconstruct full batch
            if cached_embeddings:
                sample_embedding = next(iter(cached_embeddings.values()))
                embedding_dim = sample_embedding.shape[0]
                result = self.batch_cache.reconstruct_batch(
                    cached_embeddings, token_positions, batch_size, embedding_dim
                )
            else:
                # Fallback to direct embedding getter
                result = embedding_getter(token_ids)
            
            # Record metrics
            if self.metrics:
                hit_count = batch_size - len(missing_tokens)
                miss_count = len(missing_tokens)
                
                if hit_count > 0:
                    self.metrics.record_batch_hit(hit_count, time.time() - start_time)
                if miss_count > 0:
                    self.metrics.record_batch_miss(miss_count, time.time() - start_time)
                
                if batch_hash:
                    self.metrics.record_batch_pattern(batch_hash, batch_size)
                
                for token_id in token_ids:
                    self.metrics.record_token_access(token_id)
        
        else:
            # Use individual LRU caching
            cached_items, missing_keys = self.lru_cache.get_batch(token_ids)
            
            if missing_keys:
                # Get missing embeddings
                missing_embeddings_array = embedding_getter(missing_keys)
                if isinstance(missing_embeddings_array, np.ndarray) and missing_embeddings_array.ndim == 2:
                    missing_embeddings = {
                        token_id: missing_embeddings_array[i] 
                        for i, token_id in enumerate(missing_keys)
                    }
                    
                    # Cache missing embeddings
                    self.lru_cache.put_batch(missing_embeddings)
                    cached_items.update(missing_embeddings)
            
            # Reconstruct batch in original order
            if cached_items:
                sample_embedding = next(iter(cached_items.values()))
                embedding_dim = sample_embedding.shape[0]
                result = np.zeros((batch_size, embedding_dim), dtype=np.float32)
                
                for i, token_id in enumerate(token_ids):
                    if token_id in cached_items:
                        result[i] = cached_items[token_id]
            else:
                result = embedding_getter(token_ids)
            
            # Record metrics
            if self.metrics:
                hit_count = len(cached_items)
                miss_count = len(missing_keys)
                
                if hit_count > 0:
                    self.metrics.record_batch_hit(hit_count, time.time() - start_time)
                if miss_count > 0:
                    self.metrics.record_batch_miss(miss_count, time.time() - start_time)
                
                for token_id in token_ids:
                    self.metrics.record_token_access(token_id)
        
        self._update_operation_count()
        return result
    
    def _create_batch_hash(self, token_ids: List[int]) -> str:
        """Create hash for batch pattern tracking."""
        # Sort token IDs to create consistent hash for same set of tokens
        sorted_tokens = sorted(set(token_ids))
        token_str = ','.join(map(str, sorted_tokens))
        return hashlib.md5(token_str.encode()).hexdigest()[:16]
    
    def warm_cache(self, embedding_getter: callable, vocab_size: Optional[int] = None,
                  token_frequencies: Optional[Dict[int, int]] = None):
        """
        Warm the cache using the configured strategy.
        
        Args:
            embedding_getter: Function to get embeddings
            vocab_size: Size of vocabulary (required for random/sequential strategies)
            token_frequencies: Token frequency data (required for frequency strategy)
        """
        if not self.config.enable_cache_warming or not self.cache_warmer:
            return
        
        logger.info(f"Starting cache warming with strategy: {self.config.warmup_strategy}")
        
        try:
            if self.config.warmup_strategy == "frequency" and token_frequencies:
                self.cache_warmer.warm_cache_from_frequencies(
                    token_frequencies, embedding_getter, self.config.warmup_size
                )
            elif self.config.warmup_strategy == "random" and vocab_size:
                self.cache_warmer.warm_cache_random(
                    vocab_size, embedding_getter, self.config.warmup_size
                )
            elif self.config.warmup_strategy == "sequential" and vocab_size:
                self.cache_warmer.warm_cache_sequential(
                    vocab_size, embedding_getter, self.config.warmup_size
                )
            else:
                logger.warning(f"Cannot warm cache: missing required parameters for strategy {self.config.warmup_strategy}")
        
        except Exception as e:
            logger.error(f"Error during cache warming: {e}")
    
    def _update_operation_count(self):
        """Update operation count and handle persistence."""
        self.operation_count += 1
        
        # Check memory usage and cleanup if needed
        if self.operation_count % 100 == 0:  # Check every 100 operations
            self._check_memory_usage()
        
        # Handle persistence
        if (self.config.enable_persistence and 
            self.operation_count % self.config.save_interval == 0):
            self._save_cache_state()
    
    def _check_memory_usage(self):
        """Check memory usage and cleanup if threshold exceeded."""
        try:
            lru_memory = self.lru_cache.get_memory_usage()
            total_memory = lru_memory
            
            if self.batch_cache:
                # Estimate batch cache memory (simplified)
                batch_stats = self.batch_cache.get_stats()
                if batch_stats['size'] > 0:
                    # Rough estimate based on LRU cache
                    total_memory += lru_memory * (batch_stats['size'] / self.lru_cache.size()) if self.lru_cache.size() > 0 else 0
            
            if self.metrics:
                self.metrics.memory_usage_mb = total_memory
            
            # Cleanup if threshold exceeded
            if total_memory > self.config.memory_threshold_mb:
                logger.info(f"Memory threshold exceeded ({total_memory:.1f}MB > {self.config.memory_threshold_mb}MB), cleaning up caches")
                self._cleanup_caches()
        
        except Exception as e:
            logger.warning(f"Error checking memory usage: {e}")
    
    def _cleanup_caches(self):
        """Clean up caches to reduce memory usage."""
        # Calculate how much to clean
        cleanup_size = int(self.lru_cache.max_size * self.config.cleanup_ratio)
        
        # Clear portion of LRU cache (oldest items)
        with self.lru_cache.lock:
            for _ in range(min(cleanup_size, len(self.lru_cache.cache))):
                if self.lru_cache.cache:
                    self.lru_cache.cache.popitem(last=False)
        
        # Clear batch cache if enabled
        if self.batch_cache:
            self.batch_cache.clear()
        
        logger.info(f"Cache cleanup completed: removed ~{cleanup_size} items")
    
    def _save_cache_state(self):
        """Save cache state to disk for persistence."""
        if not self.config.cache_file_path:
            return
        
        try:
            cache_state = {
                'lru_cache': dict(self.lru_cache.cache),
                'operation_count': self.operation_count,
                'timestamp': time.time()
            }
            
            with open(self.config.cache_file_path, 'wb') as f:
                pickle.dump(cache_state, f)
            
            logger.debug(f"Saved cache state to {self.config.cache_file_path}")
        
        except Exception as e:
            logger.warning(f"Error saving cache state: {e}")
    
    def load_cache_state(self, cache_file_path: Optional[str] = None):
        """
        Load cache state from disk.
        
        Args:
            cache_file_path: Path to cache file (uses config default if None)
        """
        file_path = cache_file_path or self.config.cache_file_path
        if not file_path or not os.path.exists(file_path):
            return
        
        try:
            with open(file_path, 'rb') as f:
                cache_state = pickle.load(f)
            
            # Restore LRU cache
            if 'lru_cache' in cache_state:
                self.lru_cache.cache = OrderedDict(cache_state['lru_cache'])
            
            # Restore operation count
            if 'operation_count' in cache_state:
                self.operation_count = cache_state['operation_count']
            
            logger.info(f"Loaded cache state from {file_path}")
        
        except Exception as e:
            logger.warning(f"Error loading cache state: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'lru_cache': self.lru_cache.get_stats(),
            'operation_count': self.operation_count,
            'config': {
                'max_cache_size': self.config.max_cache_size,
                'enable_batch_caching': self.config.enable_batch_caching,
                'enable_cache_warming': self.config.enable_cache_warming,
                'warmup_strategy': self.config.warmup_strategy
            }
        }
        
        if self.batch_cache:
            stats['batch_cache'] = self.batch_cache.get_stats()
        
        if self.metrics:
            stats['metrics'] = self.metrics.get_summary()
        
        return stats
    
    def clear_all_caches(self):
        """Clear all caches."""
        self.lru_cache.clear()
        if self.batch_cache:
            self.batch_cache.clear()
        
        if self.metrics:
            self.metrics.reset()
        
        logger.info("Cleared all caches")
    
    def shutdown(self):
        """Shutdown the caching system."""
        if self.cache_warmer:
            self.cache_warmer.shutdown()
        
        if self.config.enable_persistence:
            self._save_cache_state()
        
        logger.info("Intelligent caching system shutdown complete")