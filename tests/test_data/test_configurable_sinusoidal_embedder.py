#!/usr/bin/env python3
"""
Tests for ConfigurableSinusoidalEmbedder.

This module tests the configurable sinusoidal embedding layer with learnable
frequency parameters, supporting both absolute and relative positional encodings.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import tempfile
import os
import json

from src.lsm.data.configurable_sinusoidal_embedder import (
    SinusoidalConfig,
    ConfigurableSinusoidalEmbedder,
    SinusoidalEmbedderFactory,
    EmbeddingDimensionOptimizer
)
from src.lsm.utils.lsm_exceptions import InvalidInputError, TokenizerSaveError, TokenizerLoadError


class TestSinusoidalConfig:
    """Test SinusoidalConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SinusoidalConfig()
        
        assert config.embedding_dim == 128
        assert config.vocab_size == 10000
        assert config.max_sequence_length == 512
        assert config.base_frequency == 10000.0
        assert config.frequency_scaling == 1.0
        assert config.learnable_frequencies is True
        assert config.use_absolute_position is True
        assert config.use_relative_position is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SinusoidalConfig(
            embedding_dim=256,
            vocab_size=50000,
            base_frequency=5000.0,
            learnable_frequencies=False,
            use_relative_position=True
        )
        
        assert config.embedding_dim == 256
        assert config.vocab_size == 50000
        assert config.base_frequency == 5000.0
        assert config.learnable_frequencies is False
        assert config.use_relative_position is True


class TestConfigurableSinusoidalEmbedder:
    """Test ConfigurableSinusoidalEmbedder class."""
    
    @pytest.fixture
    def default_config(self):
        """Create default configuration for testing."""
        return SinusoidalConfig(
            embedding_dim=64,
            vocab_size=1000,
            max_sequence_length=128
        )
    
    @pytest.fixture
    def embedder(self, default_config):
        """Create embedder instance for testing."""
        return ConfigurableSinusoidalEmbedder(default_config)
    
    def test_initialization(self, default_config):
        """Test embedder initialization."""
        embedder = ConfigurableSinusoidalEmbedder(default_config)
        
        assert embedder.config == default_config
        assert embedder._fitted is False
        assert hasattr(embedder, 'token_embedding')
        assert hasattr(embedder, 'frequency_weights')
    
    def test_invalid_config_validation(self):
        """Test validation of invalid configuration parameters."""
        # Test negative embedding_dim
        with pytest.raises(InvalidInputError, match="embedding_dim must be positive"):
            config = SinusoidalConfig(embedding_dim=-1)
            ConfigurableSinusoidalEmbedder(config)
        
        # Test negative vocab_size
        with pytest.raises(InvalidInputError, match="vocab_size must be positive"):
            config = SinusoidalConfig(vocab_size=-1)
            ConfigurableSinusoidalEmbedder(config)
        
        # Test negative base_frequency
        with pytest.raises(InvalidInputError, match="base_frequency must be positive"):
            config = SinusoidalConfig(base_frequency=-1.0)
            ConfigurableSinusoidalEmbedder(config)
    
    def test_build_layer(self, embedder):
        """Test layer building."""
        input_shape = (None, 32)  # (batch_size, sequence_length)
        embedder.build(input_shape)
        
        assert embedder.built is True
        assert embedder.token_embedding.built is True
    
    def test_call_basic(self, embedder):
        """Test basic forward pass."""
        # Create sample input
        batch_size, seq_length = 2, 10
        inputs = tf.random.uniform((batch_size, seq_length), maxval=1000, dtype=tf.int32)
        
        # Build and call
        embedder.build((None, seq_length))
        outputs = embedder(inputs)
        
        # Check output shape
        expected_shape = (batch_size, seq_length, embedder.config.embedding_dim)
        assert outputs.shape == expected_shape
        
        # Check output is not all zeros
        assert not tf.reduce_all(tf.equal(outputs, 0.0))
    
    def test_learnable_frequencies(self):
        """Test learnable frequency parameters."""
        config = SinusoidalConfig(
            embedding_dim=64,
            vocab_size=1000,
            learnable_frequencies=True
        )
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Check that frequency weights are trainable
        assert embedder.frequency_weights.trainable is True
        assert embedder.phase_shift.trainable is True
    
    def test_fixed_frequencies(self):
        """Test fixed frequency parameters."""
        config = SinusoidalConfig(
            embedding_dim=64,
            vocab_size=1000,
            learnable_frequencies=False
        )
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Check that frequency weights are not trainable
        assert not hasattr(embedder.frequency_weights, 'trainable') or not embedder.frequency_weights.trainable
    
    def test_relative_position_encoding(self):
        """Test relative positional encoding."""
        config = SinusoidalConfig(
            embedding_dim=64,
            vocab_size=1000,
            use_relative_position=True,
            relative_position_window=32
        )
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Check relative position embedding exists
        assert hasattr(embedder, 'relative_position_embedding')
        
        # Test forward pass
        inputs = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32)
        embedder.build((None, 10))
        outputs = embedder(inputs)
        
        assert outputs.shape == (2, 10, 64)
    
    def test_adapt_to_vocabulary(self, embedder):
        """Test vocabulary adaptation."""
        original_vocab_size = embedder.config.vocab_size
        new_vocab_size = 2000
        
        # Build the embedder first
        embedder.build((None, 10))
        
        # Adapt to new vocabulary size
        embedder.adapt_to_vocabulary(new_vocab_size)
        
        assert embedder.config.vocab_size == new_vocab_size
        assert embedder._fitted is True
        
        # Test with invalid vocab size
        with pytest.raises(InvalidInputError, match="vocab_size must be positive"):
            embedder.adapt_to_vocabulary(-1)
    
    def test_get_embedding_patterns(self, embedder):
        """Test embedding pattern extraction."""
        embedder.build((None, 10))
        patterns = embedder.get_embedding_patterns(max_positions=50)
        
        assert 'positional_encoding' in patterns
        assert 'frequencies' in patterns
        assert 'positions' in patterns
        assert 'embedding_dim' in patterns
        
        # Check shapes
        assert patterns['positional_encoding'].shape == (50, embedder.config.embedding_dim)
        assert patterns['frequencies'].shape == (embedder.config.embedding_dim // 2,)
        assert len(patterns['positions']) == 50
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_embeddings(self, mock_savefig, mock_show, embedder):
        """Test embedding visualization."""
        embedder.build((None, 10))
        
        # Test visualization without saving
        embedder.visualize_embeddings(max_positions=20)
        mock_show.assert_called_once()
        
        # Test visualization with saving
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            embedder.visualize_embeddings(max_positions=20, save_path=tmp.name)
            mock_savefig.assert_called_once()
        
        os.unlink(tmp.name)
    
    def test_visualize_embeddings_no_matplotlib(self, embedder):
        """Test visualization when matplotlib is not available."""
        embedder.build((None, 10))
        
        with patch('src.lsm.data.configurable_sinusoidal_embedder.plt', None):
            with patch('builtins.__import__', side_effect=ImportError):
                # Should not raise an error, just log a warning
                embedder.visualize_embeddings()
    
    def test_get_config(self, embedder):
        """Test configuration serialization."""
        config = embedder.get_config()
        
        assert 'sinusoidal_config' in config
        sinusoidal_config = config['sinusoidal_config']
        
        assert sinusoidal_config['embedding_dim'] == embedder.config.embedding_dim
        assert sinusoidal_config['vocab_size'] == embedder.config.vocab_size
        assert sinusoidal_config['learnable_frequencies'] == embedder.config.learnable_frequencies
    
    def test_from_config(self, embedder):
        """Test configuration deserialization."""
        config = embedder.get_config()
        new_embedder = ConfigurableSinusoidalEmbedder.from_config(config)
        
        assert new_embedder.config.embedding_dim == embedder.config.embedding_dim
        assert new_embedder.config.vocab_size == embedder.config.vocab_size
        assert new_embedder.config.learnable_frequencies == embedder.config.learnable_frequencies
    
    def test_save_load_config(self, embedder):
        """Test saving and loading configuration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            # Save configuration
            embedder.save_config(tmp.name)
            
            # Load configuration
            loaded_embedder = ConfigurableSinusoidalEmbedder.load_config(tmp.name)
            
            assert loaded_embedder.config.embedding_dim == embedder.config.embedding_dim
            assert loaded_embedder.config.vocab_size == embedder.config.vocab_size
            assert loaded_embedder.config.learnable_frequencies == embedder.config.learnable_frequencies
        
        os.unlink(tmp.name)
    
    def test_save_config_error(self, embedder):
        """Test save configuration error handling."""
        with pytest.raises(TokenizerSaveError):
            embedder.save_config('/invalid/path/config.json')
    
    def test_load_config_error(self):
        """Test load configuration error handling."""
        with pytest.raises(TokenizerLoadError):
            ConfigurableSinusoidalEmbedder.load_config('/nonexistent/config.json')
    
    def test_mathematical_properties(self, embedder):
        """Test mathematical properties of sinusoidal embeddings."""
        embedder.build((None, 10))
        
        # Test that embeddings are different for different positions
        inputs1 = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int32)
        inputs2 = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
        
        outputs1 = embedder(inputs1)
        outputs2 = embedder(inputs2)
        
        # Embeddings should be different
        assert not tf.reduce_all(tf.equal(outputs1, outputs2))
        
        # Test that same tokens at same positions produce same embeddings
        inputs_same = tf.constant([[0, 1, 2], [0, 1, 2]], dtype=tf.int32)
        outputs_same = embedder(inputs_same)
        
        assert tf.reduce_all(tf.equal(outputs_same[0], outputs_same[1]))


class TestSinusoidalEmbedderFactory:
    """Test SinusoidalEmbedderFactory class."""
    
    def test_create_default(self):
        """Test default embedder creation."""
        embedder = SinusoidalEmbedderFactory.create_default(vocab_size=5000, embedding_dim=256)
        
        assert embedder.config.vocab_size == 5000
        assert embedder.config.embedding_dim == 256
        assert embedder.config.learnable_frequencies is True
        assert embedder.config.use_absolute_position is True
        assert embedder.config.use_relative_position is False
    
    def test_create_relative_position(self):
        """Test relative position embedder creation."""
        embedder = SinusoidalEmbedderFactory.create_relative_position(
            vocab_size=5000, 
            embedding_dim=256,
            relative_window=128
        )
        
        assert embedder.config.vocab_size == 5000
        assert embedder.config.embedding_dim == 256
        assert embedder.config.use_relative_position is True
        assert embedder.config.relative_position_window == 128
    
    def test_create_fixed_frequency(self):
        """Test fixed frequency embedder creation."""
        embedder = SinusoidalEmbedderFactory.create_fixed_frequency(
            vocab_size=5000,
            embedding_dim=256,
            base_frequency=5000.0
        )
        
        assert embedder.config.vocab_size == 5000
        assert embedder.config.embedding_dim == 256
        assert embedder.config.learnable_frequencies is False
        assert embedder.config.base_frequency == 5000.0
    
    def test_create_high_performance(self):
        """Test high performance embedder creation."""
        embedder = SinusoidalEmbedderFactory.create_high_performance(
            vocab_size=5000,
            embedding_dim=256
        )
        
        assert embedder.config.vocab_size == 5000
        assert embedder.config.embedding_dim == 256
        assert embedder.config.use_mixed_precision is True
        assert embedder.config.gradient_checkpointing is True


class TestIntegration:
    """Integration tests for ConfigurableSinusoidalEmbedder."""
    
    def test_training_integration(self):
        """Test integration with TensorFlow training."""
        config = SinusoidalConfig(
            embedding_dim=32,
            vocab_size=100,
            max_sequence_length=20
        )
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Create a simple model
        inputs = keras.layers.Input(shape=(10,), dtype=tf.int32)
        embeddings = embedder(inputs)
        outputs = keras.layers.GlobalAveragePooling1D()(embeddings)
        outputs = keras.layers.Dense(1, activation='sigmoid')(outputs)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Create dummy data
        x_train = tf.random.uniform((32, 10), maxval=100, dtype=tf.int32)
        y_train = tf.random.uniform((32, 1))
        
        # Test training
        history = model.fit(x_train, y_train, epochs=1, verbose=0)
        
        assert len(history.history['loss']) == 1
        assert not np.isnan(history.history['loss'][0])
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large vocabulary."""
        config = SinusoidalConfig(
            embedding_dim=128,
            vocab_size=50000,  # Large vocabulary
            max_sequence_length=512
        )
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Test that we can create and use the embedder without memory issues
        inputs = tf.random.uniform((4, 100), maxval=50000, dtype=tf.int32)
        embedder.build((None, 100))
        outputs = embedder(inputs)
        
        assert outputs.shape == (4, 100, 128)
    
    def test_different_sequence_lengths(self):
        """Test handling of different sequence lengths."""
        config = SinusoidalConfig(embedding_dim=64, vocab_size=1000)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Test different sequence lengths
        for seq_len in [5, 10, 50, 100]:
            inputs = tf.random.uniform((2, seq_len), maxval=1000, dtype=tf.int32)
            embedder.build((None, seq_len))
            outputs = embedder(inputs)
            
            assert outputs.shape == (2, seq_len, 64)


class TestAutomaticEmbeddingAdaptation:
    """Test automatic embedding layer adaptation functionality."""
    
    def test_adapt_to_vocabulary(self):
        """Test adapting embedder to new vocabulary size."""
        config = SinusoidalConfig(embedding_dim=128, vocab_size=1000)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build with initial vocab size
        inputs = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32)
        embedder.build((None, 10))
        initial_outputs = embedder(inputs)
        
        # Adapt to larger vocabulary
        embedder.adapt_to_vocabulary(5000)
        
        assert embedder.config.vocab_size == 5000
        assert embedder._fitted is True
        
        # Test with new vocabulary range
        new_inputs = tf.random.uniform((2, 10), maxval=5000, dtype=tf.int32)
        new_outputs = embedder(new_inputs)
        
        assert new_outputs.shape == (2, 10, 128)
    
    def test_adapt_to_tokenizer(self):
        """Test adapting embedder to tokenizer adapter."""
        from unittest.mock import MagicMock
        from src.lsm.data.enhanced_tokenization import TokenizerAdapter, TokenizerConfig
        
        # Create a proper mock that inherits from TokenizerAdapter
        class MockTokenizerAdapter(TokenizerAdapter):
            def __init__(self):
                config = TokenizerConfig(backend='test', model_name='test-model')
                super().__init__(config)
                self._is_initialized = True
                self._vocab_size = 3000
            
            def initialize(self):
                pass
            
            def tokenize(self, texts, add_special_tokens=True, padding=True, truncation=True):
                return [[1, 2, 3]]
            
            def decode(self, token_ids, skip_special_tokens=True):
                return "test"
            
            def get_vocab_size(self):
                return 3000
            
            def get_vocab(self):
                return {'test': 1, 'token': 2}
            
            def get_special_tokens(self):
                return {'<pad>': 0, '<unk>': 1}
            
            @classmethod
            def load_adapter_config(cls, load_path):
                return cls()
        
        mock_adapter = MockTokenizerAdapter()
        
        config = SinusoidalConfig(embedding_dim=128, vocab_size=1000)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Adapt to tokenizer
        embedder.adapt_to_tokenizer(mock_adapter)
        
        assert embedder.config.vocab_size == 3000
        assert embedder._fitted is True
        assert hasattr(embedder, '_tokenizer_info')
        assert embedder._tokenizer_info['backend'] == 'test'
        assert embedder._tokenizer_info['vocab_size'] == 3000
    
    def test_adapt_embedding_dimension(self):
        """Test adapting embedder to new embedding dimension."""
        config = SinusoidalConfig(embedding_dim=128, vocab_size=1000)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build with initial dimension
        inputs = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32)
        embedder.build((None, 10))
        initial_outputs = embedder(inputs)
        
        # Adapt to new dimension
        embedder.adapt_embedding_dimension(256, preserve_properties=True)
        
        assert embedder.config.embedding_dim == 256
        
        # Test with new dimension
        embedder.build((None, 10))
        new_outputs = embedder(inputs)
        
        assert new_outputs.shape == (2, 10, 256)
    
    def test_adapt_embedding_dimension_invalid(self):
        """Test error handling for invalid embedding dimensions."""
        config = SinusoidalConfig(embedding_dim=128, vocab_size=1000)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Test negative dimension
        with pytest.raises(InvalidInputError):
            embedder.adapt_embedding_dimension(-1)
        
        # Test odd dimension
        with pytest.raises(InvalidInputError):
            embedder.adapt_embedding_dimension(127)
    
    def test_scale_embedding_weights(self):
        """Test scaling embedding weights to new dimensions."""
        config = SinusoidalConfig(embedding_dim=128, vocab_size=100)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Create test weights
        old_weights = np.random.normal(0, 0.02, (100, 128)).astype(np.float32)
        
        # Test expanding dimension
        new_weights = embedder._scale_embedding_weights(old_weights, 128, 256)
        assert new_weights.shape == (100, 256)
        assert np.allclose(new_weights[:, :128], old_weights)
        
        # Test reducing dimension
        new_weights = embedder._scale_embedding_weights(old_weights, 128, 64)
        assert new_weights.shape == (100, 64)
    
    def test_scale_frequency_weights(self):
        """Test scaling frequency weights to new dimensions."""
        config = SinusoidalConfig(embedding_dim=128, vocab_size=100)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Create test frequency weights
        old_weights = np.random.normal(0, 0.02, 64).astype(np.float32)
        
        # Test expanding
        new_weights = embedder._scale_frequency_weights(old_weights, 64, 128)
        assert new_weights.shape == (128,)
        assert np.allclose(new_weights[:64], old_weights)
        
        # Test reducing
        new_weights = embedder._scale_frequency_weights(old_weights, 64, 32)
        assert new_weights.shape == (32,)
    
    def test_get_adaptation_info(self):
        """Test getting adaptation information."""
        config = SinusoidalConfig(embedding_dim=128, vocab_size=1000)
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        info = embedder.get_adaptation_info()
        
        assert info['vocab_size'] == 1000
        assert info['embedding_dim'] == 128
        assert info['is_fitted'] is False
        assert info['learnable_frequencies'] is True
        assert 'base_frequency' in info
        assert 'frequency_scaling' in info


class TestEmbeddingDimensionOptimizer:
    """Test embedding dimension optimization utilities."""
    
    def test_calculate_optimal_dimension(self):
        """Test calculating optimal embedding dimensions."""
        # Test different model sizes
        for size in ['small', 'medium', 'large', 'xlarge']:
            dim = EmbeddingDimensionOptimizer.calculate_optimal_dimension(
                vocab_size=10000, target_model_size=size
            )
            assert dim >= 64
            assert dim <= 2048
            assert dim % 2 == 0  # Should be even
    
    def test_calculate_optimal_dimension_invalid_size(self):
        """Test error handling for invalid model size."""
        with pytest.raises(InvalidInputError):
            EmbeddingDimensionOptimizer.calculate_optimal_dimension(
                vocab_size=10000, target_model_size='invalid'
            )
    
    def test_has_nice_factorization(self):
        """Test nice factorization detection."""
        # Powers of 2 should have nice factorization
        assert EmbeddingDimensionOptimizer._has_nice_factorization(64)
        assert EmbeddingDimensionOptimizer._has_nice_factorization(128)
        assert EmbeddingDimensionOptimizer._has_nice_factorization(256)
        
        # Numbers with small prime factors should have nice factorization
        assert EmbeddingDimensionOptimizer._has_nice_factorization(96)  # 2^5 * 3
        assert EmbeddingDimensionOptimizer._has_nice_factorization(120)  # 2^3 * 3 * 5
    
    def test_suggest_dimension_scaling(self):
        """Test dimension scaling suggestions."""
        # Test scaling up
        new_dim = EmbeddingDimensionOptimizer.suggest_dimension_scaling(
            old_vocab_size=1000, new_vocab_size=4000, old_embedding_dim=128
        )
        assert new_dim > 128
        assert new_dim % 2 == 0
        
        # Test scaling down
        new_dim = EmbeddingDimensionOptimizer.suggest_dimension_scaling(
            old_vocab_size=4000, new_vocab_size=1000, old_embedding_dim=256
        )
        assert new_dim < 256
        assert new_dim % 2 == 0
    
    def test_suggest_dimension_scaling_invalid(self):
        """Test error handling for invalid scaling parameters."""
        with pytest.raises(InvalidInputError):
            EmbeddingDimensionOptimizer.suggest_dimension_scaling(
                old_vocab_size=0, new_vocab_size=1000, old_embedding_dim=128
            )


class TestSinusoidalEmbedderFactoryAutoAdapted:
    """Test auto-adapted factory methods."""
    
    def test_create_auto_adapted(self):
        """Test creating auto-adapted embedder."""
        from unittest.mock import MagicMock
        
        # Create mock tokenizer adapter
        mock_adapter = MagicMock()
        mock_adapter._is_initialized = True
        mock_adapter.get_vocab_size.return_value = 5000
        mock_adapter.get_special_tokens.return_value = {'<pad>': 0}
        mock_adapter.config.backend = 'test'
        mock_adapter.config.model_name = 'test-model'
        
        # Create auto-adapted embedder
        embedder = SinusoidalEmbedderFactory.create_auto_adapted(
            mock_adapter, target_model_size='medium'
        )
        
        assert isinstance(embedder, ConfigurableSinusoidalEmbedder)
        assert embedder.config.vocab_size == 5000
        assert embedder._fitted is True
        assert hasattr(embedder, '_tokenizer_info')
    
    def test_create_auto_adapted_invalid_adapter(self):
        """Test error handling for invalid adapter."""
        with pytest.raises(InvalidInputError):
            SinusoidalEmbedderFactory.create_auto_adapted("not_an_adapter")


if __name__ == '__main__':
    pytest.main([__file__])