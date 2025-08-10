#!/usr/bin/env python3
"""
Memory-efficient embedding storage for large vocabularies.

This module provides memory-mapped embedding matrices, compressed storage,
and gradient checkpointing support for memory-constrained training.
"""

import os
import json
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import pickle
import gzip
import mmap
from pathlib import Path

from ..utils.lsm_exceptions import (
    TokenizerError, TokenizerNotFittedError, 
    TokenizerLoadError, TokenizerSaveError, InvalidInputError
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryStorageConfig:
    """Configuration for memory-efficient embedding storage."""
    
    # Memory mapping options
    use_memory_mapping: bool = True
    memory_map_threshold: int = 100000  # Vocab size threshold for memory mapping
    memory_map_dir: Optional[str] = None  # Directory for memory-mapped files
    
    # Compression options
    use_compression: bool = True
    compression_level: int = 6  # gzip compression level (1-9)
    compression_threshold: int = 50000  # Vocab size threshold for compression
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = False
    checkpoint_segments: int = 4  # Number of segments for gradient checkpointing
    
    # Cache options
    enable_embedding_cache: bool = True
    cache_size: int = 10000  # Number of embeddings to cache
    cache_strategy: str = "lru"  # "lru", "lfu", or "random"
    
    # Performance options
    prefetch_size: int = 1000  # Number of embeddings to prefetch
    batch_load_size: int = 5000  # Batch size for loading embeddings
    
    # GPU acceleration options
    enable_gpu_acceleration: bool = True
    use_gpu_memory_mapping: bool = True  # Use GPU memory for frequently accessed embeddings


class MemoryMappedEmbedding:
    """
    Memory-mapped embedding matrix for large vocabularies.
    
    This class provides efficient storage and access to large embedding matrices
    using memory mapping to avoid loading the entire matrix into RAM.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 storage_dir: Optional[str] = None, dtype: np.dtype = np.float32):
        """
        Initialize memory-mapped embedding.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            storage_dir: Directory to store memory-mapped files
            dtype: Data type for embeddings
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        
        # Create storage directory
        if storage_dir is None:
            self.storage_dir = tempfile.mkdtemp(prefix="lsm_embeddings_")
        else:
            self.storage_dir = storage_dir
            os.makedirs(self.storage_dir, exist_ok=True)
        
        self.embedding_file = os.path.join(self.storage_dir, "embeddings.dat")
        self.metadata_file = os.path.join(self.storage_dir, "metadata.json")
        
        # Initialize memory-mapped array
        self._initialize_memory_map()
        
        logger.info(f"Initialized memory-mapped embedding: {vocab_size}x{embedding_dim} "
                   f"in {self.storage_dir}")
    
    def _initialize_memory_map(self):
        """Initialize the memory-mapped embedding array."""
        shape = (self.vocab_size, self.embedding_dim)
        
        # Create or open memory-mapped file
        if not os.path.exists(self.embedding_file):
            # Create new memory-mapped file
            self.embeddings = np.memmap(
                self.embedding_file, 
                dtype=self.dtype, 
                mode='w+', 
                shape=shape
            )
            
            # Initialize with small random values
            self.embeddings[:] = np.random.normal(
                0, 0.02, shape
            ).astype(self.dtype)
            
            # Save metadata
            metadata = {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'dtype': str(self.dtype),
                'shape': shape
            }
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f)
        else:
            # Load existing memory-mapped file
            self.embeddings = np.memmap(
                self.embedding_file, 
                dtype=self.dtype, 
                mode='r+', 
                shape=shape
            )
        
        # Flush to ensure data is written
        self.embeddings.flush()
    
    def get_embeddings(self, indices: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get embeddings for given indices.
        
        Args:
            indices: Token indices to retrieve embeddings for
            
        Returns:
            Embedding vectors for the given indices
        """
        if isinstance(indices, int):
            return self.embeddings[indices].copy()
        else:
            return self.embeddings[indices].copy()
    
    def set_embeddings(self, indices: Union[int, List[int], np.ndarray], 
                      values: np.ndarray):
        """
        Set embeddings for given indices.
        
        Args:
            indices: Token indices to set embeddings for
            values: Embedding values to set
        """
        self.embeddings[indices] = values
        self.embeddings.flush()
    
    def update_embeddings(self, indices: Union[int, List[int], np.ndarray], 
                         updates: np.ndarray):
        """
        Update embeddings with gradients.
        
        Args:
            indices: Token indices to update
            updates: Gradient updates to apply
        """
        self.embeddings[indices] += updates
        self.embeddings.flush()
    
    def get_batch(self, start_idx: int, end_idx: int) -> np.ndarray:
        """
        Get a batch of embeddings.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            
        Returns:
            Batch of embeddings
        """
        return self.embeddings[start_idx:end_idx].copy()
    
    def set_batch(self, start_idx: int, end_idx: int, values: np.ndarray):
        """
        Set a batch of embeddings.
        
        Args:
            start_idx: Starting index
            end_idx: Ending index
            values: Embedding values to set
        """
        self.embeddings[start_idx:end_idx] = values
        self.embeddings.flush()
    
    def save_to_file(self, filepath: str):
        """
        Save embeddings to a regular file.
        
        Args:
            filepath: Path to save embeddings
        """
        np.save(filepath, self.embeddings)
        logger.info(f"Saved memory-mapped embeddings to {filepath}")
    
    def load_from_file(self, filepath: str):
        """
        Load embeddings from a regular file.
        
        Args:
            filepath: Path to load embeddings from
        """
        embeddings = np.load(filepath)
        if embeddings.shape != (self.vocab_size, self.embedding_dim):
            raise InvalidInputError(
                "embedding_shape", 
                f"{(self.vocab_size, self.embedding_dim)}", 
                str(embeddings.shape)
            )
        
        self.embeddings[:] = embeddings
        self.embeddings.flush()
        logger.info(f"Loaded embeddings from {filepath}")
    
    def cleanup(self):
        """Clean up memory-mapped files."""
        try:
            if hasattr(self, 'embeddings'):
                del self.embeddings
            
            if os.path.exists(self.embedding_file):
                os.remove(self.embedding_file)
            if os.path.exists(self.metadata_file):
                os.remove(self.metadata_file)
            
            # Remove directory if empty
            try:
                os.rmdir(self.storage_dir)
            except OSError:
                pass  # Directory not empty
                
            logger.info(f"Cleaned up memory-mapped embedding files")
        except Exception as e:
            logger.warning(f"Error cleaning up memory-mapped files: {e}")


class CompressedEmbeddingStorage:
    """
    Compressed embedding storage with on-demand decompression.
    
    This class provides compressed storage for embedding matrices to save
    disk space and memory, with on-demand decompression for access.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 compression_level: int = 6, chunk_size: int = 1000):
        """
        Initialize compressed embedding storage.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            compression_level: gzip compression level (1-9)
            chunk_size: Size of chunks for compression
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.compression_level = compression_level
        self.chunk_size = chunk_size
        
        # Calculate number of chunks
        self.num_chunks = (vocab_size + chunk_size - 1) // chunk_size
        
        # Storage for compressed chunks
        self.compressed_chunks: Dict[int, bytes] = {}
        self.chunk_cache: Dict[int, np.ndarray] = {}
        self.cache_size = min(10, self.num_chunks)  # Cache up to 10 chunks
        
        # Initialize with random embeddings
        self._initialize_embeddings()
        
        logger.info(f"Initialized compressed embedding storage: {vocab_size}x{embedding_dim} "
                   f"with {self.num_chunks} chunks of size {chunk_size}")
    
    def _initialize_embeddings(self):
        """Initialize embeddings with random values and compress them."""
        for chunk_idx in range(self.num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.vocab_size)
            chunk_size = end_idx - start_idx
            
            # Create random embeddings for this chunk
            chunk_embeddings = np.random.normal(
                0, 0.02, (chunk_size, self.embedding_dim)
            ).astype(np.float32)
            
            # Compress and store
            self._compress_chunk(chunk_idx, chunk_embeddings)
    
    def _compress_chunk(self, chunk_idx: int, embeddings: np.ndarray):
        """
        Compress and store a chunk of embeddings.
        
        Args:
            chunk_idx: Index of the chunk
            embeddings: Embedding array to compress
        """
        # Serialize embeddings
        serialized = pickle.dumps(embeddings)
        
        # Compress with gzip
        compressed = gzip.compress(serialized, compresslevel=self.compression_level)
        
        # Store compressed data
        self.compressed_chunks[chunk_idx] = compressed
        
        # Remove from cache if present
        if chunk_idx in self.chunk_cache:
            del self.chunk_cache[chunk_idx]
    
    def _decompress_chunk(self, chunk_idx: int) -> np.ndarray:
        """
        Decompress and return a chunk of embeddings.
        
        Args:
            chunk_idx: Index of the chunk to decompress
            
        Returns:
            Decompressed embedding array
        """
        # Check cache first
        if chunk_idx in self.chunk_cache:
            return self.chunk_cache[chunk_idx]
        
        # Decompress from storage
        if chunk_idx not in self.compressed_chunks:
            raise InvalidInputError("chunk_idx", f"valid chunk index (0-{self.num_chunks-1})", str(chunk_idx))
        
        compressed_data = self.compressed_chunks[chunk_idx]
        
        # Decompress
        decompressed = gzip.decompress(compressed_data)
        embeddings = pickle.loads(decompressed)
        
        # Add to cache (with LRU eviction)
        if len(self.chunk_cache) >= self.cache_size:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(self.chunk_cache))
            del self.chunk_cache[oldest_key]
        
        self.chunk_cache[chunk_idx] = embeddings
        return embeddings
    
    def get_embeddings(self, indices: Union[int, List[int], np.ndarray]) -> np.ndarray:
        """
        Get embeddings for given indices.
        
        Args:
            indices: Token indices to retrieve embeddings for
            
        Returns:
            Embedding vectors for the given indices
        """
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        results = []
        
        # Group indices by chunk
        chunk_indices = {}
        for i, idx in enumerate(indices):
            chunk_idx = idx // self.chunk_size
            local_idx = idx % self.chunk_size
            
            if chunk_idx not in chunk_indices:
                chunk_indices[chunk_idx] = []
            chunk_indices[chunk_idx].append((i, local_idx))
        
        # Initialize result array
        result_embeddings = np.zeros((len(indices), self.embedding_dim), dtype=np.float32)
        
        # Retrieve embeddings from each chunk
        for chunk_idx, idx_pairs in chunk_indices.items():
            chunk_embeddings = self._decompress_chunk(chunk_idx)
            
            for result_idx, local_idx in idx_pairs:
                result_embeddings[result_idx] = chunk_embeddings[local_idx]
        
        return result_embeddings
    
    def set_embeddings(self, indices: Union[int, List[int], np.ndarray], 
                      values: np.ndarray):
        """
        Set embeddings for given indices.
        
        Args:
            indices: Token indices to set embeddings for
            values: Embedding values to set
        """
        if isinstance(indices, int):
            indices = [indices]
            values = values.reshape(1, -1)
        elif isinstance(indices, np.ndarray):
            indices = indices.tolist()
        
        # Group indices by chunk
        chunk_updates = {}
        for i, idx in enumerate(indices):
            chunk_idx = idx // self.chunk_size
            local_idx = idx % self.chunk_size
            
            if chunk_idx not in chunk_updates:
                chunk_updates[chunk_idx] = []
            chunk_updates[chunk_idx].append((local_idx, values[i]))
        
        # Update each affected chunk
        for chunk_idx, updates in chunk_updates.items():
            # Decompress chunk
            chunk_embeddings = self._decompress_chunk(chunk_idx)
            
            # Apply updates
            for local_idx, new_value in updates:
                chunk_embeddings[local_idx] = new_value
            
            # Recompress and store
            self._compress_chunk(chunk_idx, chunk_embeddings)
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        total_compressed_size = sum(len(data) for data in self.compressed_chunks.values())
        uncompressed_size = self.vocab_size * self.embedding_dim * 4  # float32
        compression_ratio = uncompressed_size / total_compressed_size if total_compressed_size > 0 else 0
        
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'num_chunks': self.num_chunks,
            'chunk_size': self.chunk_size,
            'uncompressed_size_bytes': uncompressed_size,
            'compressed_size_bytes': total_compressed_size,
            'compression_ratio': compression_ratio,
            'compression_level': self.compression_level,
            'cached_chunks': len(self.chunk_cache)
        }
    
    def save_to_file(self, filepath: str):
        """
        Save compressed embeddings to file.
        
        Args:
            filepath: Path to save compressed embeddings
        """
        save_data = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'chunk_size': self.chunk_size,
            'compression_level': self.compression_level,
            'num_chunks': self.num_chunks,
            'compressed_chunks': self.compressed_chunks
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved compressed embeddings to {filepath}")
    
    def load_from_file(self, filepath: str):
        """
        Load compressed embeddings from file.
        
        Args:
            filepath: Path to load compressed embeddings from
        """
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        # Validate compatibility
        if (save_data['vocab_size'] != self.vocab_size or 
            save_data['embedding_dim'] != self.embedding_dim):
            raise InvalidInputError(
                "embedding_dimensions",
                f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}",
                f"vocab_size={save_data['vocab_size']}, embedding_dim={save_data['embedding_dim']}"
            )
        
        # Load data
        self.chunk_size = save_data['chunk_size']
        self.compression_level = save_data['compression_level']
        self.num_chunks = save_data['num_chunks']
        self.compressed_chunks = save_data['compressed_chunks']
        
        # Clear cache
        self.chunk_cache.clear()
        
        logger.info(f"Loaded compressed embeddings from {filepath}")


class GradientCheckpointedEmbedding(keras.layers.Layer):
    """
    Embedding layer with gradient checkpointing support.
    
    This layer implements gradient checkpointing to reduce memory usage
    during training by recomputing activations instead of storing them.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 checkpoint_segments: int = 4, **kwargs):
        """
        Initialize gradient checkpointed embedding.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            checkpoint_segments: Number of segments for gradient checkpointing
            **kwargs: Additional keras layer arguments
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.checkpoint_segments = checkpoint_segments
        
        # Create embedding layer
        self.embedding = keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            name="checkpointed_embedding"
        )
        
        logger.info(f"Initialized gradient checkpointed embedding: {vocab_size}x{embedding_dim} "
                   f"with {checkpoint_segments} segments")
    
    def build(self, input_shape):
        """Build the layer."""
        super().build(input_shape)
        self.embedding.build(input_shape)
    
    @tf.recompute_grad
    def _checkpointed_embedding(self, inputs):
        """Embedding computation with gradient checkpointing."""
        return self.embedding(inputs)
    
    def call(self, inputs, training=None):
        """
        Forward pass with gradient checkpointing.
        
        Args:
            inputs: Input token IDs
            training: Whether in training mode
            
        Returns:
            Embedded tokens
        """
        if training and self.checkpoint_segments > 1:
            # Use gradient checkpointing during training
            return self._checkpointed_embedding(inputs)
        else:
            # Regular forward pass during inference
            return self.embedding(inputs)
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'checkpoint_segments': self.checkpoint_segments
        })
        return config


class MemoryEfficientEmbeddingLayer(keras.layers.Layer):
    """
    Memory-efficient embedding layer that combines multiple storage strategies.
    
    This layer automatically selects the best storage strategy based on
    vocabulary size and memory constraints.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 config: Optional[MemoryStorageConfig] = None, **kwargs):
        """
        Initialize memory-efficient embedding layer.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            config: Memory storage configuration
            **kwargs: Additional keras layer arguments
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.config = config or MemoryStorageConfig()
        
        # Select storage strategy
        self._select_storage_strategy()
        
        logger.info(f"Initialized memory-efficient embedding: {vocab_size}x{embedding_dim} "
                   f"using {self.storage_strategy}")
    
    def _select_storage_strategy(self):
        """Select the best storage strategy based on configuration and constraints."""
        if (self.config.use_memory_mapping and 
            self.vocab_size >= self.config.memory_map_threshold):
            self.storage_strategy = "memory_mapped"
            self._init_memory_mapped_storage()
        elif (self.config.use_compression and 
              self.vocab_size >= self.config.compression_threshold):
            self.storage_strategy = "compressed"
            self._init_compressed_storage()
        elif self.config.use_gradient_checkpointing:
            self.storage_strategy = "gradient_checkpointed"
            self._init_gradient_checkpointed_storage()
        else:
            self.storage_strategy = "standard"
            self._init_standard_storage()
    
    def _init_memory_mapped_storage(self):
        """Initialize memory-mapped storage."""
        self.memory_mapped_embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim,
            self.config.memory_map_dir
        )
    
    def _init_compressed_storage(self):
        """Initialize compressed storage."""
        self.compressed_embedding = CompressedEmbeddingStorage(
            self.vocab_size,
            self.embedding_dim,
            self.config.compression_level
        )
    
    def _init_gradient_checkpointed_storage(self):
        """Initialize gradient checkpointed storage."""
        self.checkpointed_embedding = GradientCheckpointedEmbedding(
            self.vocab_size,
            self.embedding_dim,
            self.config.checkpoint_segments
        )
    
    def _init_standard_storage(self):
        """Initialize standard embedding storage."""
        self.standard_embedding = keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            name="standard_embedding"
        )
    
    def build(self, input_shape):
        """Build the layer."""
        super().build(input_shape)
        
        if self.storage_strategy == "gradient_checkpointed":
            self.checkpointed_embedding.build(input_shape)
        elif self.storage_strategy == "standard":
            self.standard_embedding.build(input_shape)
    
    def call(self, inputs, training=None):
        """
        Forward pass using the selected storage strategy.
        
        Args:
            inputs: Input token IDs
            training: Whether in training mode
            
        Returns:
            Embedded tokens
        """
        if self.storage_strategy == "memory_mapped":
            return self._call_memory_mapped(inputs)
        elif self.storage_strategy == "compressed":
            return self._call_compressed(inputs)
        elif self.storage_strategy == "gradient_checkpointed":
            return self.checkpointed_embedding(inputs, training=training)
        else:
            return self.standard_embedding(inputs)
    
    def _call_memory_mapped(self, inputs):
        """Call using memory-mapped storage."""
        # Convert inputs to numpy for indexing
        input_shape = tf.shape(inputs)
        flat_inputs = tf.reshape(inputs, [-1])
        
        # Create a wrapper function that handles numpy conversion
        def get_embeddings_wrapper(indices):
            indices_np = indices.numpy()
            embeddings = self.memory_mapped_embedding.get_embeddings(indices_np)
            return embeddings.astype(np.float32)
        
        # Get embeddings from memory-mapped storage
        embeddings = tf.py_function(
            func=get_embeddings_wrapper,
            inp=[flat_inputs],
            Tout=tf.float32
        )
        
        # Set shape for embeddings
        flat_size = tf.reduce_prod(input_shape)
        embeddings.set_shape([None, self.embedding_dim])
        
        # Reshape to original input shape + embedding dimension
        output_shape = tf.concat([input_shape, [self.embedding_dim]], axis=0)
        embeddings = tf.reshape(embeddings, output_shape)
        
        return embeddings
    
    def _call_compressed(self, inputs):
        """Call using compressed storage."""
        # Convert inputs to numpy for indexing
        input_shape = tf.shape(inputs)
        flat_inputs = tf.reshape(inputs, [-1])
        
        # Create a wrapper function that handles numpy conversion
        def get_embeddings_wrapper(indices):
            indices_np = indices.numpy()
            embeddings = self.compressed_embedding.get_embeddings(indices_np)
            return embeddings.astype(np.float32)
        
        # Get embeddings from compressed storage
        embeddings = tf.py_function(
            func=get_embeddings_wrapper,
            inp=[flat_inputs],
            Tout=tf.float32
        )
        
        # Set shape for embeddings
        embeddings.set_shape([None, self.embedding_dim])
        
        # Reshape to original input shape + embedding dimension
        output_shape = tf.concat([input_shape, [self.embedding_dim]], axis=0)
        embeddings = tf.reshape(embeddings, output_shape)
        
        return embeddings
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary with storage statistics
        """
        stats = {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'storage_strategy': self.storage_strategy
        }
        
        if self.storage_strategy == "compressed":
            stats.update(self.compressed_embedding.get_compression_stats())
        
        return stats
    
    def get_config(self):
        """Get layer configuration."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'storage_config': {
                'use_memory_mapping': self.config.use_memory_mapping,
                'memory_map_threshold': self.config.memory_map_threshold,
                'use_compression': self.config.use_compression,
                'compression_level': self.config.compression_level,
                'compression_threshold': self.config.compression_threshold,
                'use_gradient_checkpointing': self.config.use_gradient_checkpointing,
                'checkpoint_segments': self.config.checkpoint_segments
            }
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        storage_config = MemoryStorageConfig(**config.pop('storage_config', {}))
        vocab_size = config.pop('vocab_size')
        embedding_dim = config.pop('embedding_dim')
        return cls(vocab_size, embedding_dim, storage_config, **config)
    
    def cleanup(self):
        """Clean up storage resources."""
        if self.storage_strategy == "memory_mapped" and hasattr(self, 'memory_mapped_embedding'):
            self.memory_mapped_embedding.cleanup()