#!/usr/bin/env python3
"""
Tests for memory-efficient embedding storage.

This module tests memory-mapped embeddings, compressed storage,
gradient checkpointing, and the integrated memory-efficient embedding layer.
"""

import os
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import pytest
from unittest.mock import patch, MagicMock

from src.lsm.data.memory_efficient_storage import (
    MemoryStorageConfig,
    MemoryMappedEmbedding,
    CompressedEmbeddingStorage,
    GradientCheckpointedEmbedding,
    MemoryEfficientEmbeddingLayer
)
from src.lsm.data.configurable_sinusoidal_embedder import (
    SinusoidalConfig,
    ConfigurableSinusoidalEmbedder
)
from src.lsm.utils.lsm_exceptions import InvalidInputError


class TestMemoryStorageConfig:
    """Test memory storage configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MemoryStorageConfig()
        
        assert config.use_memory_mapping is True
        assert config.memory_map_threshold == 100000
        assert config.use_compression is True
        assert config.compression_level == 6
        assert config.use_gradient_checkpointing is False
        assert config.enable_embedding_cache is True
        assert config.cache_size == 10000
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MemoryStorageConfig(
            use_memory_mapping=False,
            memory_map_threshold=50000,
            compression_level=9,
            use_gradient_checkpointing=True,
            cache_size=5000
        )
        
        assert config.use_memory_mapping is False
        assert config.memory_map_threshold == 50000
        assert config.compression_level == 9
        assert config.use_gradient_checkpointing is True
        assert config.cache_size == 5000


class TestMemoryMappedEmbedding:
    """Test memory-mapped embedding functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.vocab_size = 1000
        self.embedding_dim = 128
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test memory-mapped embedding initialization."""
        embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir
        )
        
        assert embedding.vocab_size == self.vocab_size
        assert embedding.embedding_dim == self.embedding_dim
        assert embedding.embeddings.shape == (self.vocab_size, self.embedding_dim)
        assert os.path.exists(embedding.embedding_file)
        assert os.path.exists(embedding.metadata_file)
        
        embedding.cleanup()
    
    def test_get_single_embedding(self):
        """Test getting a single embedding."""
        embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir
        )
        
        # Get single embedding
        result = embedding.get_embeddings(0)
        assert result.shape == (self.embedding_dim,)
        assert isinstance(result, np.ndarray)
        
        embedding.cleanup()
    
    def test_get_multiple_embeddings(self):
        """Test getting multiple embeddings."""
        embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir
        )
        
        # Get multiple embeddings
        indices = [0, 1, 2, 10, 100]
        result = embedding.get_embeddings(indices)
        assert result.shape == (len(indices), self.embedding_dim)
        
        embedding.cleanup()
    
    def test_set_embeddings(self):
        """Test setting embeddings."""
        embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir
        )
        
        # Set single embedding
        new_value = np.ones(self.embedding_dim, dtype=np.float32)
        embedding.set_embeddings(0, new_value)
        
        # Verify it was set
        result = embedding.get_embeddings(0)
        np.testing.assert_array_equal(result, new_value)
        
        embedding.cleanup()
    
    def test_batch_operations(self):
        """Test batch get and set operations."""
        embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir
        )
        
        # Test batch get
        batch = embedding.get_batch(0, 10)
        assert batch.shape == (10, self.embedding_dim)
        
        # Test batch set
        new_batch = np.ones((10, self.embedding_dim), dtype=np.float32)
        embedding.set_batch(0, 10, new_batch)
        
        # Verify batch was set
        result_batch = embedding.get_batch(0, 10)
        np.testing.assert_array_equal(result_batch, new_batch)
        
        embedding.cleanup()
    
    def test_save_and_load(self):
        """Test saving and loading embeddings."""
        embedding = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir
        )
        
        # Set some values
        test_values = np.random.random((self.vocab_size, self.embedding_dim)).astype(np.float32)
        embedding.embeddings[:] = test_values
        
        # Save to file
        save_path = os.path.join(self.temp_dir, "test_embeddings.npy")
        embedding.save_to_file(save_path)
        assert os.path.exists(save_path)
        
        # Create new embedding and load
        embedding2 = MemoryMappedEmbedding(
            self.vocab_size, 
            self.embedding_dim, 
            self.temp_dir + "_2"
        )
        embedding2.load_from_file(save_path)
        
        # Verify loaded values match
        np.testing.assert_array_equal(embedding.embeddings, embedding2.embeddings)
        
        embedding.cleanup()
        embedding2.cleanup()


class TestCompressedEmbeddingStorage:
    """Test compressed embedding storage functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.embedding_dim = 128
        self.chunk_size = 100
    
    def test_initialization(self):
        """Test compressed storage initialization."""
        storage = CompressedEmbeddingStorage(
            self.vocab_size, 
            self.embedding_dim,
            chunk_size=self.chunk_size
        )
        
        assert storage.vocab_size == self.vocab_size
        assert storage.embedding_dim == self.embedding_dim
        assert storage.chunk_size == self.chunk_size
        assert storage.num_chunks == (self.vocab_size + self.chunk_size - 1) // self.chunk_size
        assert len(storage.compressed_chunks) == storage.num_chunks
    
    def test_get_single_embedding(self):
        """Test getting a single embedding from compressed storage."""
        storage = CompressedEmbeddingStorage(
            self.vocab_size, 
            self.embedding_dim,
            chunk_size=self.chunk_size
        )
        
        # Get single embedding
        result = storage.get_embeddings(0)
        assert result.shape == (1, self.embedding_dim)
        
        # Get embedding by index
        result_single = storage.get_embeddings([0])
        assert result_single.shape == (1, self.embedding_dim)
    
    def test_get_multiple_embeddings(self):
        """Test getting multiple embeddings from compressed storage."""
        storage = CompressedEmbeddingStorage(
            self.vocab_size, 
            self.embedding_dim,
            chunk_size=self.chunk_size
        )
        
        # Get multiple embeddings across chunks
        indices = [0, 50, 150, 250, 500]
        result = storage.get_embeddings(indices)
        assert result.shape == (len(indices), self.embedding_dim)
    
    def test_set_embeddings(self):
        """Test setting embeddings in compressed storage."""
        storage = CompressedEmbeddingStorage(
            self.vocab_size, 
            self.embedding_dim,
            chunk_size=self.chunk_size
        )
        
        # Set single embedding
        new_value = np.ones((1, self.embedding_dim), dtype=np.float32)
        storage.set_embeddings(0, new_value)
        
        # Verify it was set
        result = storage.get_embeddings(0)
        np.testing.assert_array_equal(result, new_value)
    
    def test_compression_stats(self):
        """Test compression statistics."""
        storage = CompressedEmbeddingStorage(
            self.vocab_size, 
            self.embedding_dim,
            chunk_size=self.chunk_size
        )
        
        stats = storage.get_compression_stats()
        
        assert stats['vocab_size'] == self.vocab_size
        assert stats['embedding_dim'] == self.embedding_dim
        assert stats['num_chunks'] == storage.num_chunks
        assert stats['compression_ratio'] > 0
        assert 'compressed_size_bytes' in stats
        assert 'uncompressed_size_bytes' in stats
    
    def test_save_and_load(self):
        """Test saving and loading compressed storage."""
        storage = CompressedEmbeddingStorage(
            self.vocab_size, 
            self.embedding_dim,
            chunk_size=self.chunk_size
        )
        
        # Set some test values
        test_indices = [0, 50, 150]
        test_values = np.random.random((len(test_indices), self.embedding_dim)).astype(np.float32)
        storage.set_embeddings(test_indices, test_values)
        
        # Save to file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            save_path = f.name
        
        try:
            storage.save_to_file(save_path)
            
            # Create new storage and load
            storage2 = CompressedEmbeddingStorage(
                self.vocab_size, 
                self.embedding_dim,
                chunk_size=self.chunk_size
            )
            storage2.load_from_file(save_path)
            
            # Verify loaded values match
            result = storage2.get_embeddings(test_indices)
            np.testing.assert_array_almost_equal(result, test_values, decimal=5)
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)


class TestGradientCheckpointedEmbedding:
    """Test gradient checkpointed embedding layer."""
    
    def test_initialization(self):
        """Test gradient checkpointed embedding initialization."""
        vocab_size = 1000
        embedding_dim = 128
        checkpoint_segments = 4
        
        layer = GradientCheckpointedEmbedding(
            vocab_size, 
            embedding_dim, 
            checkpoint_segments
        )
        
        assert layer.vocab_size == vocab_size
        assert layer.embedding_dim == embedding_dim
        assert layer.checkpoint_segments == checkpoint_segments
    
    def test_forward_pass(self):
        """Test forward pass with gradient checkpointing."""
        vocab_size = 100
        embedding_dim = 64
        batch_size = 8
        seq_length = 10
        
        layer = GradientCheckpointedEmbedding(vocab_size, embedding_dim)
        
        # Create test input
        inputs = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
        
        # Forward pass
        outputs = layer(inputs, training=True)
        
        assert outputs.shape == (batch_size, seq_length, embedding_dim)
        assert outputs.dtype == tf.float32
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        vocab_size = 1000
        embedding_dim = 128
        checkpoint_segments = 4
        
        layer = GradientCheckpointedEmbedding(
            vocab_size, 
            embedding_dim, 
            checkpoint_segments
        )
        
        config = layer.get_config()
        
        assert config['vocab_size'] == vocab_size
        assert config['embedding_dim'] == embedding_dim
        assert config['checkpoint_segments'] == checkpoint_segments


class TestMemoryEfficientEmbeddingLayer:
    """Test the integrated memory-efficient embedding layer."""
    
    def test_strategy_selection_memory_mapped(self):
        """Test memory-mapped strategy selection."""
        vocab_size = 200000  # Above memory map threshold
        embedding_dim = 128
        
        config = MemoryStorageConfig(
            use_memory_mapping=True,
            memory_map_threshold=100000
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config.memory_map_dir = temp_dir
            layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
            
            assert layer.storage_strategy == "memory_mapped"
            assert hasattr(layer, 'memory_mapped_embedding')
            
            # Clean up
            layer.cleanup()
    
    def test_strategy_selection_compressed(self):
        """Test compressed strategy selection."""
        vocab_size = 75000  # Above compression threshold, below memory map threshold
        embedding_dim = 128
        
        config = MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=True,
            compression_threshold=50000
        )
        
        layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
        
        assert layer.storage_strategy == "compressed"
        assert hasattr(layer, 'compressed_embedding')
    
    def test_strategy_selection_gradient_checkpointed(self):
        """Test gradient checkpointed strategy selection."""
        vocab_size = 1000
        embedding_dim = 128
        
        config = MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=False,
            use_gradient_checkpointing=True
        )
        
        layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
        
        assert layer.storage_strategy == "gradient_checkpointed"
        assert hasattr(layer, 'checkpointed_embedding')
    
    def test_strategy_selection_standard(self):
        """Test standard strategy selection."""
        vocab_size = 1000
        embedding_dim = 128
        
        config = MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=False,
            use_gradient_checkpointing=False
        )
        
        layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
        
        assert layer.storage_strategy == "standard"
        assert hasattr(layer, 'standard_embedding')
    
    def test_forward_pass_standard(self):
        """Test forward pass with standard storage."""
        vocab_size = 100
        embedding_dim = 64
        batch_size = 8
        seq_length = 10
        
        config = MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=False,
            use_gradient_checkpointing=False
        )
        
        layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
        
        # Create test input
        inputs = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
        
        # Forward pass
        outputs = layer(inputs)
        
        assert outputs.shape == (batch_size, seq_length, embedding_dim)
        assert outputs.dtype == tf.float32
    
    def test_storage_stats(self):
        """Test storage statistics."""
        vocab_size = 1000
        embedding_dim = 128
        
        config = MemoryStorageConfig(
            use_compression=True,
            compression_threshold=500
        )
        
        layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
        
        stats = layer.get_storage_stats()
        
        assert stats['vocab_size'] == vocab_size
        assert stats['embedding_dim'] == embedding_dim
        assert 'storage_strategy' in stats
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        vocab_size = 1000
        embedding_dim = 128
        
        config = MemoryStorageConfig(
            use_memory_mapping=True,
            compression_level=9,
            use_gradient_checkpointing=True
        )
        
        layer = MemoryEfficientEmbeddingLayer(vocab_size, embedding_dim, config)
        
        layer_config = layer.get_config()
        
        assert layer_config['vocab_size'] == vocab_size
        assert layer_config['embedding_dim'] == embedding_dim
        assert 'storage_config' in layer_config
        
        # Test reconstruction
        reconstructed = MemoryEfficientEmbeddingLayer.from_config(layer_config)
        assert reconstructed.vocab_size == vocab_size
        assert reconstructed.embedding_dim == embedding_dim


class TestConfigurableSinusoidalEmbedderIntegration:
    """Test integration with ConfigurableSinusoidalEmbedder."""
    
    def test_memory_efficient_storage_integration(self):
        """Test memory-efficient storage integration with sinusoidal embedder."""
        vocab_size = 1000
        embedding_dim = 128
        
        # Create memory storage config
        memory_config = MemoryStorageConfig(
            use_compression=True,
            compression_threshold=500,
            use_gradient_checkpointing=True
        )
        
        # Create sinusoidal config with memory-efficient storage
        sinusoidal_config = SinusoidalConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            use_memory_efficient_storage=True,
            memory_storage_config=memory_config
        )
        
        # Create embedder
        embedder = ConfigurableSinusoidalEmbedder(sinusoidal_config)
        
        # Test that memory-efficient storage is used
        assert embedder.config.use_memory_efficient_storage is True
        assert hasattr(embedder, 'token_embedding')
        assert isinstance(embedder.token_embedding, MemoryEfficientEmbeddingLayer)
    
    def test_memory_storage_stats(self):
        """Test memory storage statistics from sinusoidal embedder."""
        vocab_size = 1000
        embedding_dim = 128
        
        memory_config = MemoryStorageConfig(use_compression=True)
        sinusoidal_config = SinusoidalConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            use_memory_efficient_storage=True,
            memory_storage_config=memory_config
        )
        
        embedder = ConfigurableSinusoidalEmbedder(sinusoidal_config)
        
        # Get memory storage stats
        stats = embedder.get_memory_storage_stats()
        
        assert stats['use_memory_efficient_storage'] is True
        assert stats['vocab_size'] == vocab_size
        assert stats['embedding_dim'] == embedding_dim
    
    def test_config_serialization_with_memory_storage(self):
        """Test configuration serialization with memory storage."""
        vocab_size = 1000
        embedding_dim = 128
        
        memory_config = MemoryStorageConfig(
            use_memory_mapping=True,
            compression_level=9
        )
        
        sinusoidal_config = SinusoidalConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            use_memory_efficient_storage=True,
            memory_storage_config=memory_config
        )
        
        embedder = ConfigurableSinusoidalEmbedder(sinusoidal_config)
        
        # Test config serialization
        config = embedder.get_config()
        
        assert config['sinusoidal_config']['use_memory_efficient_storage'] is True
        assert config['sinusoidal_config']['memory_storage_config'] is not None
        
        # Test saving and loading config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_path = f.name
        
        try:
            embedder.save_config(config_path)
            loaded_embedder = ConfigurableSinusoidalEmbedder.load_config(config_path)
            
            assert loaded_embedder.config.use_memory_efficient_storage is True
            assert loaded_embedder.config.memory_storage_config is not None
            assert loaded_embedder.config.memory_storage_config.use_memory_mapping is True
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_cleanup_memory_storage(self):
        """Test memory storage cleanup."""
        vocab_size = 1000
        embedding_dim = 128
        
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_config = MemoryStorageConfig(
                use_memory_mapping=True,
                memory_map_threshold=500,
                memory_map_dir=temp_dir
            )
            
            sinusoidal_config = SinusoidalConfig(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                use_memory_efficient_storage=True,
                memory_storage_config=memory_config
            )
            
            embedder = ConfigurableSinusoidalEmbedder(sinusoidal_config)
            
            # Test cleanup
            embedder.cleanup_memory_storage()
            
            # Should not raise any errors


if __name__ == "__main__":
    pytest.main([__file__])