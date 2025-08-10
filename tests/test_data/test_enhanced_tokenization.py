#!/usr/bin/env python3
"""
Tests for enhanced tokenization system.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.lsm.data.enhanced_tokenization import (
    TokenizerAdapter, TokenizerConfig, TokenizerRegistry, 
    EnhancedTokenizerWrapper
)
from src.lsm.utils.lsm_exceptions import TokenizerError, TokenizerSaveError, TokenizerLoadError


class MockTokenizerAdapter(TokenizerAdapter):
    """Mock tokenizer adapter for testing."""
    
    def __init__(self, config: TokenizerConfig):
        super().__init__(config)
        self._vocab = {'hello': 1, 'world': 2, 'test': 3, '[PAD]': 0, '[EOS]': 4}
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
    
    def initialize(self) -> None:
        self._vocab_size = len(self._vocab)
        self._is_initialized = True
    
    def tokenize(self, texts, add_special_tokens=True, padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        
        result = []
        for text in texts:
            tokens = []
            words = text.lower().split()
            for word in words:
                tokens.append(self._vocab.get(word, 3))  # 3 is 'test' as UNK
            
            if add_special_tokens:
                tokens.append(4)  # EOS token
            
            if padding and len(tokens) < 5:
                tokens.extend([0] * (5 - len(tokens)))  # PAD to length 5
            
            result.append(tokens)
        
        return result
    
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids[0], int):
            # Single sequence
            words = []
            for token_id in token_ids:
                if skip_special_tokens and token_id in [0, 4]:  # PAD, EOS
                    continue
                words.append(self._reverse_vocab.get(token_id, '[UNK]'))
            return ' '.join(words)
        else:
            # Batch
            return [self.decode(seq, skip_special_tokens) for seq in token_ids]
    
    def get_vocab_size(self):
        return self._vocab_size
    
    def get_vocab(self):
        return self._vocab.copy()
    
    def get_special_tokens(self):
        return {'pad_token_id': 0, 'eos_token_id': 4}
    
    @classmethod
    def load_adapter_config(cls, load_path):
        config = TokenizerConfig(backend='mock', model_name='mock-model')
        adapter = cls(config)
        adapter.initialize()
        return adapter


class TestTokenizerConfig:
    """Test TokenizerConfig dataclass."""
    
    def test_init_minimal(self):
        """Test minimal initialization."""
        config = TokenizerConfig(backend='test', model_name='test-model')
        
        assert config.backend == 'test'
        assert config.model_name == 'test-model'
        assert config.max_length == 512
        assert config.special_tokens is None
        assert config.backend_specific_config is None
    
    def test_init_full(self):
        """Test full initialization."""
        special_tokens = {'pad': '[PAD]'}
        backend_config = {'param1': 'value1'}
        
        config = TokenizerConfig(
            backend='test',
            model_name='test-model',
            max_length=256,
            special_tokens=special_tokens,
            backend_specific_config=backend_config
        )
        
        assert config.backend == 'test'
        assert config.model_name == 'test-model'
        assert config.max_length == 256
        assert config.special_tokens == special_tokens
        assert config.backend_specific_config == backend_config


class TestTokenizerAdapter:
    """Test TokenizerAdapter abstract base class."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter for testing."""
        config = TokenizerConfig(backend='mock', model_name='mock-model')
        adapter = MockTokenizerAdapter(config)
        adapter.initialize()
        return adapter
    
    def test_init(self):
        """Test adapter initialization."""
        config = TokenizerConfig(backend='test', model_name='test-model')
        adapter = MockTokenizerAdapter(config)
        
        assert adapter.config == config
        assert adapter._tokenizer is None
        assert adapter._vocab_size is None
        assert not adapter._is_initialized
    
    def test_initialize(self, mock_adapter):
        """Test adapter initialization."""
        assert mock_adapter._is_initialized
        assert mock_adapter._vocab_size == 5
    
    def test_tokenize(self, mock_adapter):
        """Test tokenization."""
        result = mock_adapter.tokenize("hello world")
        assert result == [[1, 2, 4, 0, 0]]  # hello, world, EOS, PAD, PAD
        
        result = mock_adapter.tokenize(["hello", "world test"])
        assert len(result) == 2
        assert result[0] == [1, 4, 0, 0, 0]  # hello, EOS, PAD, PAD, PAD
        assert result[1] == [2, 3, 4, 0, 0]  # world, test, EOS, PAD, PAD
    
    def test_decode(self, mock_adapter):
        """Test decoding."""
        result = mock_adapter.decode([1, 2, 4, 0, 0])
        assert result == "hello world"
        
        result = mock_adapter.decode([[1, 2, 4], [2, 3, 4]])
        assert result == ["hello world", "world test"]
    
    def test_encode_single(self, mock_adapter):
        """Test single encoding."""
        result = mock_adapter.encode_single("hello world")
        assert result == [1, 2, 4]  # No padding for single encoding
    
    def test_decode_single(self, mock_adapter):
        """Test single decoding."""
        result = mock_adapter.decode_single([1, 2, 4, 0, 0])
        assert result == "hello world"
    
    def test_get_vocab_size(self, mock_adapter):
        """Test vocabulary size."""
        assert mock_adapter.get_vocab_size() == 5
    
    def test_get_vocab(self, mock_adapter):
        """Test vocabulary retrieval."""
        vocab = mock_adapter.get_vocab()
        expected = {'hello': 1, 'world': 2, 'test': 3, '[PAD]': 0, '[EOS]': 4}
        assert vocab == expected
    
    def test_get_special_tokens(self, mock_adapter):
        """Test special tokens retrieval."""
        special_tokens = mock_adapter.get_special_tokens()
        assert special_tokens == {'pad_token_id': 0, 'eos_token_id': 4}
    
    def test_get_token_embeddings_shape(self, mock_adapter):
        """Test token embeddings shape."""
        shape = mock_adapter.get_token_embeddings_shape(128)
        assert shape == (5, 128)
    
    def test_save_adapter_config(self, mock_adapter):
        """Test saving adapter configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            mock_adapter.save_adapter_config(temp_dir)
            
            config_path = os.path.join(temp_dir, 'mock_adapter_config.json')
            assert os.path.exists(config_path)
    
    def test_repr(self, mock_adapter):
        """Test string representation."""
        repr_str = repr(mock_adapter)
        assert "MockTokenizerAdapter" in repr_str
        assert "backend=mock" in repr_str
        assert "model=mock-model" in repr_str


class TestTokenizerRegistry:
    """Test TokenizerRegistry class."""
    
    def setUp(self):
        """Clear registry before each test."""
        TokenizerRegistry._adapters.clear()
        TokenizerRegistry._model_mappings.clear()
    
    def test_register_adapter(self):
        """Test adapter registration."""
        self.setUp()
        
        TokenizerRegistry.register_adapter(
            'mock', MockTokenizerAdapter, ['mock-model', 'test-model']
        )
        
        assert 'mock' in TokenizerRegistry._adapters
        assert TokenizerRegistry._adapters['mock'] == MockTokenizerAdapter
        assert TokenizerRegistry._model_mappings['mock-model'] == 'mock'
        assert TokenizerRegistry._model_mappings['test-model'] == 'mock'
    
    def test_get_adapter_class(self):
        """Test getting adapter class."""
        self.setUp()
        
        TokenizerRegistry.register_adapter(
            'mock', MockTokenizerAdapter, ['mock-model']
        )
        
        # Test direct backend lookup
        adapter_class = TokenizerRegistry.get_adapter_class('mock')
        assert adapter_class == MockTokenizerAdapter
        
        # Test model pattern matching
        adapter_class = TokenizerRegistry.get_adapter_class('mock-model')
        assert adapter_class == MockTokenizerAdapter
    
    def test_get_adapter_class_not_found(self):
        """Test error when adapter not found."""
        self.setUp()
        
        with pytest.raises(TokenizerError, match="No adapter found"):
            TokenizerRegistry.get_adapter_class('nonexistent')
    
    def test_create_adapter(self):
        """Test creating adapter instance."""
        self.setUp()
        
        TokenizerRegistry.register_adapter(
            'mock', MockTokenizerAdapter, ['mock-model']
        )
        
        adapter = TokenizerRegistry.create_adapter('mock-model')
        assert isinstance(adapter, MockTokenizerAdapter)
        assert adapter._is_initialized
    
    def test_list_available_backends(self):
        """Test listing available backends."""
        self.setUp()
        
        TokenizerRegistry.register_adapter('mock1', MockTokenizerAdapter)
        TokenizerRegistry.register_adapter('mock2', MockTokenizerAdapter)
        
        backends = TokenizerRegistry.list_available_backends()
        assert set(backends) == {'mock1', 'mock2'}
    
    def test_list_supported_models(self):
        """Test listing supported models."""
        self.setUp()
        
        TokenizerRegistry.register_adapter(
            'mock', MockTokenizerAdapter, ['model1', 'model2']
        )
        
        models = TokenizerRegistry.list_supported_models()
        assert models == {'mock': ['model1', 'model2']}


class TestEnhancedTokenizerWrapper:
    """Test EnhancedTokenizerWrapper class."""
    
    def setUp(self):
        """Set up test environment."""
        TokenizerRegistry._adapters.clear()
        TokenizerRegistry._model_mappings.clear()
        TokenizerRegistry.register_adapter(
            'mock', MockTokenizerAdapter, ['mock-model']
        )
    
    @pytest.fixture
    def wrapper(self):
        """Create wrapper for testing."""
        self.setUp()
        return EnhancedTokenizerWrapper('mock-model', embedding_dim=64)
    
    def test_init_with_string(self):
        """Test initialization with string tokenizer name."""
        self.setUp()
        
        wrapper = EnhancedTokenizerWrapper('mock-model', embedding_dim=64)
        
        assert wrapper.embedding_dim == 64
        assert wrapper.max_length == 512
        assert isinstance(wrapper._adapter, MockTokenizerAdapter)
        assert wrapper._adapter._is_initialized
    
    def test_init_with_adapter(self):
        """Test initialization with adapter instance."""
        config = TokenizerConfig(backend='mock', model_name='mock-model')
        adapter = MockTokenizerAdapter(config)
        adapter.initialize()
        
        wrapper = EnhancedTokenizerWrapper(adapter, embedding_dim=64)
        
        assert wrapper.embedding_dim == 64
        assert wrapper._adapter == adapter
    
    def test_init_invalid_tokenizer(self):
        """Test initialization with invalid tokenizer."""
        with pytest.raises(TokenizerError, match="Invalid tokenizer type"):
            EnhancedTokenizerWrapper(123, embedding_dim=64)
    
    def test_get_adapter(self, wrapper):
        """Test getting adapter."""
        adapter = wrapper.get_adapter()
        assert isinstance(adapter, MockTokenizerAdapter)
    
    def test_tokenize(self, wrapper):
        """Test tokenization."""
        result = wrapper.tokenize("hello world")
        assert result == [[1, 2, 4, 0, 0]]
    
    def test_decode(self, wrapper):
        """Test decoding."""
        result = wrapper.decode([1, 2, 4, 0, 0])
        assert result == "hello world"
    
    def test_encode_single(self, wrapper):
        """Test single encoding."""
        result = wrapper.encode_single("hello world")
        assert result == [1, 2, 4]  # No padding for single encoding
    
    def test_decode_single(self, wrapper):
        """Test single decoding."""
        result = wrapper.decode_single([1, 2, 4, 0, 0])
        assert result == "hello world"
    
    def test_get_vocab_size(self, wrapper):
        """Test vocabulary size."""
        assert wrapper.get_vocab_size() == 5
    
    def test_get_vocab(self, wrapper):
        """Test vocabulary retrieval."""
        vocab = wrapper.get_vocab()
        assert len(vocab) == 5
        assert 'hello' in vocab
    
    def test_get_special_tokens(self, wrapper):
        """Test special tokens."""
        tokens = wrapper.get_special_tokens()
        assert 'pad_token_id' in tokens
        assert 'eos_token_id' in tokens
    
    def test_get_token_embeddings_shape(self, wrapper):
        """Test token embeddings shape."""
        shape = wrapper.get_token_embeddings_shape()
        assert shape == (5, 64)
        
        shape = wrapper.get_token_embeddings_shape(128)
        assert shape == (5, 128)
    
    def test_create_sinusoidal_embedder(self, wrapper):
        """Test creating sinusoidal embedder."""
        embedder = wrapper.create_sinusoidal_embedder()
        
        assert embedder.vocab_size == 5
        assert embedder.embedding_dim == 64
    
    def test_fit_sinusoidal_embedder_with_texts(self, wrapper):
        """Test fitting sinusoidal embedder with text data."""
        training_texts = ["hello world", "world test", "hello test"]
        
        embedder = wrapper.fit_sinusoidal_embedder(training_texts, epochs=2)
        
        assert embedder._is_fitted
        assert wrapper._is_fitted
        assert wrapper.get_sinusoidal_embedder() == embedder
    
    def test_fit_sinusoidal_embedder_with_tokens(self, wrapper):
        """Test fitting sinusoidal embedder with token data."""
        training_tokens = np.array([[1, 2, 4], [2, 3, 4], [1, 3, 4]])
        
        embedder = wrapper.fit_sinusoidal_embedder(training_tokens, epochs=2)
        
        assert embedder._is_fitted
        assert wrapper._is_fitted
    
    def test_fit_sinusoidal_embedder_invalid_data(self, wrapper):
        """Test fitting with invalid data."""
        with pytest.raises(Exception):  # Should raise InvalidInputError
            wrapper.fit_sinusoidal_embedder(123, epochs=2)
    
    def test_embed_not_fitted(self, wrapper):
        """Test embedding without fitting."""
        with pytest.raises(Exception):  # Should raise TokenizerNotFittedError
            wrapper.embed([1, 2, 3])
    
    def test_embed_after_fitting(self, wrapper):
        """Test embedding after fitting."""
        training_tokens = np.array([[1, 2, 4], [2, 3, 4]])
        wrapper.fit_sinusoidal_embedder(training_tokens, epochs=2)
        
        embeddings = wrapper.embed([1, 2, 4])
        
        assert embeddings.shape == (3, 64)
        assert embeddings.dtype == np.float32
    
    def test_save_and_load(self, wrapper):
        """Test saving and loading wrapper."""
        # Fit embedder first
        training_tokens = np.array([[1, 2, 4], [2, 3, 4]])
        wrapper.fit_sinusoidal_embedder(training_tokens, epochs=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save wrapper
            wrapper.save(temp_dir)
            
            # Check files exist
            assert os.path.exists(os.path.join(temp_dir, 'enhanced_tokenizer_config.json'))
            assert os.path.exists(os.path.join(temp_dir, 'mock_adapter_config.json'))
            assert os.path.exists(os.path.join(temp_dir, 'sinusoidal_embedder'))
            
            # Load wrapper
            loaded_wrapper = EnhancedTokenizerWrapper.load(temp_dir)
            
            assert loaded_wrapper.embedding_dim == wrapper.embedding_dim
            assert loaded_wrapper.get_vocab_size() == wrapper.get_vocab_size()
            assert loaded_wrapper._is_fitted == wrapper._is_fitted
    
    def test_save_without_embedder(self, wrapper):
        """Test saving without fitted embedder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            wrapper.save(temp_dir)
            
            # Should save successfully without embedder
            assert os.path.exists(os.path.join(temp_dir, 'enhanced_tokenizer_config.json'))
            assert not os.path.exists(os.path.join(temp_dir, 'sinusoidal_embedder'))
    
    def test_load_error(self):
        """Test load error handling."""
        with pytest.raises(Exception):  # Should raise TokenizerLoadError
            EnhancedTokenizerWrapper.load("/invalid/path")
    
    def test_repr(self, wrapper):
        """Test string representation."""
        repr_str = repr(wrapper)
        
        assert "EnhancedTokenizerWrapper" in repr_str
        assert "embedding_dim=64" in repr_str
        assert "fitted=False" in repr_str


class TestAutomaticEmbeddingAdaptation:
    """Test automatic embedding adaptation functionality in EnhancedTokenizerWrapper."""
    
    @pytest.fixture
    def wrapper(self):
        """Create wrapper with mock adapter for testing."""
        from src.lsm.data.enhanced_tokenization import TokenizerAdapter, TokenizerConfig
        
        # Create a proper mock that inherits from TokenizerAdapter
        class MockTokenizerAdapter(TokenizerAdapter):
            def __init__(self):
                config = TokenizerConfig(backend='mock', model_name='mock-model')
                super().__init__(config)
                self._is_initialized = True
                self._vocab_size = 1000
            
            def initialize(self):
                pass
            
            def tokenize(self, texts, add_special_tokens=True, padding=True, truncation=True):
                return [[1, 2, 3]]
            
            def decode(self, token_ids, skip_special_tokens=True):
                return "test"
            
            def get_vocab_size(self):
                return 1000
            
            def get_vocab(self):
                return {'test': 1, 'token': 2}
            
            def get_special_tokens(self):
                return {'<pad>': 0, '<unk>': 1}
            
            @classmethod
            def load_adapter_config(cls, load_path):
                return cls()
        
        mock_adapter = MockTokenizerAdapter()
        
        return EnhancedTokenizerWrapper(
            tokenizer=mock_adapter,
            embedding_dim=128,
            max_length=512
        )
    
    def test_create_configurable_sinusoidal_embedder(self, wrapper):
        """Test creating configurable sinusoidal embedder."""
        embedder = wrapper.create_configurable_sinusoidal_embedder(
            learnable_frequencies=True,
            use_relative_position=False
        )
        
        from src.lsm.data.configurable_sinusoidal_embedder import ConfigurableSinusoidalEmbedder
        assert isinstance(embedder, ConfigurableSinusoidalEmbedder)
        assert embedder.config.vocab_size == 1000
        assert embedder.config.embedding_dim == 128
        assert embedder._fitted is True
        assert hasattr(embedder, '_tokenizer_info')
    
    def test_auto_adapt_embedding_dimension(self, wrapper):
        """Test automatic embedding dimension adaptation."""
        embedder = wrapper.auto_adapt_embedding_dimension(
            target_dim=256, preserve_properties=True
        )
        
        from src.lsm.data.configurable_sinusoidal_embedder import ConfigurableSinusoidalEmbedder
        assert isinstance(embedder, ConfigurableSinusoidalEmbedder)
        assert embedder.config.embedding_dim == 256
        assert embedder.config.vocab_size == 1000
        assert embedder._fitted is True
    
    def test_auto_adapt_embedding_dimension_invalid(self, wrapper):
        """Test error handling for invalid dimension adaptation."""
        with pytest.raises(Exception):  # Should raise InvalidInputError
            wrapper.auto_adapt_embedding_dimension(target_dim=-1)
        
        with pytest.raises(Exception):  # Should raise InvalidInputError
            wrapper.auto_adapt_embedding_dimension(target_dim=127)  # Odd dimension
    
    def test_create_optimized_embedder(self, wrapper):
        """Test creating optimized embedder."""
        embedder = wrapper.create_optimized_embedder(
            target_model_size='medium',
            learnable_frequencies=True,
            preserve_properties=True
        )
        
        from src.lsm.data.configurable_sinusoidal_embedder import ConfigurableSinusoidalEmbedder
        assert isinstance(embedder, ConfigurableSinusoidalEmbedder)
        assert embedder.config.vocab_size == 1000
        assert embedder._fitted is True
        assert hasattr(embedder, '_tokenizer_info')
    
    def test_get_embedding_dimension_suggestions(self, wrapper):
        """Test getting embedding dimension suggestions."""
        suggestions = wrapper.get_embedding_dimension_suggestions()
        
        assert isinstance(suggestions, dict)
        assert 'small' in suggestions
        assert 'medium' in suggestions
        assert 'large' in suggestions
        assert 'xlarge' in suggestions
        
        # All suggestions should be positive and even
        for size, dim in suggestions.items():
            assert dim > 0
            assert dim % 2 == 0
        
        # Larger model sizes should generally have larger dimensions
        assert suggestions['small'] <= suggestions['medium']
        assert suggestions['medium'] <= suggestions['large']
        assert suggestions['large'] <= suggestions['xlarge']


if __name__ == "__main__":
    pytest.main([__file__])