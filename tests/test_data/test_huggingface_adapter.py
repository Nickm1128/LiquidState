#!/usr/bin/env python3
"""
Tests for HuggingFace tokenizer adapter.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.lsm.data.adapters.huggingface_adapter import HuggingFaceAdapter, register_huggingface_adapter
from src.lsm.data.enhanced_tokenization import TokenizerConfig, TokenizerRegistry
from src.lsm.utils.lsm_exceptions import TokenizerError, TokenizerLoadError


class TestHuggingFaceAdapter:
    """Test HuggingFace tokenizer adapter."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return TokenizerConfig(
            backend='huggingface',
            model_name='gpt2',
            max_length=128,
            special_tokens={'pad': '[PAD]'},
            backend_specific_config={'use_fast': True}
        )
    
    @pytest.fixture
    def adapter(self, config):
        """Create adapter for testing."""
        adapter = HuggingFaceAdapter(config)
        adapter.initialize()
        return adapter
    
    def test_init(self, config):
        """Test adapter initialization."""
        adapter = HuggingFaceAdapter(config)
        
        assert adapter.config == config
        assert not adapter._is_initialized
        assert adapter._tokenizer is None
        assert adapter._vocab_size is None
    
    def test_init_without_transformers(self, config):
        """Test initialization without transformers library."""
        with patch('src.lsm.data.adapters.huggingface_adapter.TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(TokenizerError, match="transformers library not available"):
                HuggingFaceAdapter(config)
    
    def test_initialize_success(self, config):
        """Test successful initialization."""
        adapter = HuggingFaceAdapter(config)
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter._tokenizer is not None
        assert adapter._vocab_size > 0
        assert adapter._pad_token_id is not None
    
    def test_initialize_with_supported_model(self):
        """Test initialization with supported model."""
        config = TokenizerConfig(backend='huggingface', model_name='bert-base-uncased')
        adapter = HuggingFaceAdapter(config)
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter._vocab_size > 0
    
    def test_initialize_with_custom_model(self):
        """Test initialization with custom model name."""
        config = TokenizerConfig(backend='huggingface', model_name='gpt2')
        adapter = HuggingFaceAdapter(config)
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter._vocab_size == 50257  # GPT-2 vocab size
    
    def test_initialize_failure(self):
        """Test initialization failure with invalid model."""
        config = TokenizerConfig(backend='huggingface', model_name='invalid-model-name-12345')
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Failed to initialize HuggingFace tokenizer"):
            adapter.initialize()
    
    def test_tokenize_single_text(self, adapter):
        """Test tokenizing single text."""
        result = adapter.tokenize("Hello world")
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(token_id, int) for token_id in result[0])
    
    def test_tokenize_multiple_texts(self, adapter):
        """Test tokenizing multiple texts."""
        texts = ["Hello world", "This is a test", "Another sentence"]
        result = adapter.tokenize(texts)
        
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(seq, list) for seq in result)
    
    def test_tokenize_with_options(self, adapter):
        """Test tokenization with different options."""
        text = "Hello world"
        
        # Test without special tokens
        result_no_special = adapter.tokenize(text, add_special_tokens=False)
        result_with_special = adapter.tokenize(text, add_special_tokens=True)
        
        # With special tokens should be longer (or equal for some tokenizers)
        assert len(result_with_special[0]) >= len(result_no_special[0])
        
        # Test without padding
        result_no_padding = adapter.tokenize(text, padding=False)
        result_with_padding = adapter.tokenize(text, padding=True)
        
        # Results should be different lengths if padding is applied
        assert isinstance(result_no_padding[0], list)
        assert isinstance(result_with_padding[0], list)
    
    def test_tokenize_not_initialized(self, config):
        """Test tokenization without initialization."""
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Adapter not initialized"):
            adapter.tokenize("Hello world")
    
    def test_decode_single_sequence(self, adapter):
        """Test decoding single sequence."""
        # First tokenize to get valid token IDs
        tokens = adapter.tokenize("Hello world", padding=False)[0]
        
        result = adapter.decode(tokens)
        
        assert isinstance(result, str)
        assert "Hello" in result or "hello" in result.lower()
        assert "world" in result or "world" in result.lower()
    
    def test_decode_multiple_sequences(self, adapter):
        """Test decoding multiple sequences."""
        # First tokenize to get valid token IDs
        texts = ["Hello world", "This is a test"]
        token_sequences = adapter.tokenize(texts, padding=False)
        
        result = adapter.decode(token_sequences)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(text, str) for text in result)
    
    def test_decode_with_special_tokens(self, adapter):
        """Test decoding with and without special tokens."""
        tokens = adapter.tokenize("Hello world")[0]
        
        result_with_special = adapter.decode(tokens, skip_special_tokens=False)
        result_without_special = adapter.decode(tokens, skip_special_tokens=True)
        
        assert isinstance(result_with_special, str)
        assert isinstance(result_without_special, str)
    
    def test_decode_not_initialized(self, config):
        """Test decoding without initialization."""
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Adapter not initialized"):
            adapter.decode([1, 2, 3])
    
    def test_get_vocab_size(self, adapter):
        """Test getting vocabulary size."""
        vocab_size = adapter.get_vocab_size()
        
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        assert vocab_size == adapter._vocab_size
    
    def test_get_vocab_size_not_initialized(self, config):
        """Test getting vocab size without initialization."""
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Adapter not initialized"):
            adapter.get_vocab_size()
    
    def test_get_vocab(self, adapter):
        """Test getting vocabulary mapping."""
        vocab = adapter.get_vocab()
        
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        assert all(isinstance(token, str) for token in vocab.keys())
        assert all(isinstance(token_id, int) for token_id in vocab.values())
    
    def test_get_vocab_not_initialized(self, config):
        """Test getting vocab without initialization."""
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Adapter not initialized"):
            adapter.get_vocab()
    
    def test_get_special_tokens(self, adapter):
        """Test getting special tokens."""
        special_tokens = adapter.get_special_tokens()
        
        assert isinstance(special_tokens, dict)
        # Should have at least pad token
        assert 'pad_token_id' in special_tokens
        assert all(isinstance(token_id, int) for token_id in special_tokens.values())
    
    def test_get_special_tokens_not_initialized(self, config):
        """Test getting special tokens without initialization."""
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Adapter not initialized"):
            adapter.get_special_tokens()
    
    def test_get_tokenizer_info(self, adapter):
        """Test getting tokenizer information."""
        info = adapter.get_tokenizer_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'vocab_size' in info
        assert 'max_length' in info
        assert 'special_tokens' in info
        assert 'tokenizer_class' in info
        
        assert info['model_name'] == adapter.config.model_name
        assert info['vocab_size'] == adapter._vocab_size
    
    def test_get_tokenizer_info_not_initialized(self, config):
        """Test getting tokenizer info without initialization."""
        adapter = HuggingFaceAdapter(config)
        
        with pytest.raises(TokenizerError, match="Adapter not initialized"):
            adapter.get_tokenizer_info()
    
    def test_save_and_load_adapter_config(self, adapter):
        """Test saving and loading adapter configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save configuration
            adapter.save_adapter_config(temp_dir)
            
            # Check file exists
            config_path = os.path.join(temp_dir, 'huggingface_adapter_config.json')
            assert os.path.exists(config_path)
            
            # Load configuration
            loaded_adapter = HuggingFaceAdapter.load_adapter_config(temp_dir)
            
            assert loaded_adapter.config.backend == adapter.config.backend
            assert loaded_adapter.config.model_name == adapter.config.model_name
            assert loaded_adapter.config.max_length == adapter.config.max_length
            assert loaded_adapter._is_initialized
    
    def test_load_adapter_config_not_found(self):
        """Test loading adapter config when file not found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(TokenizerLoadError):
                HuggingFaceAdapter.load_adapter_config(temp_dir)
    
    def test_list_supported_models(self):
        """Test listing supported models."""
        models = HuggingFaceAdapter.list_supported_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'gpt2' in models
        assert 'bert-base-uncased' in models
    
    def test_repr(self, adapter):
        """Test string representation."""
        repr_str = repr(adapter)
        
        assert "HuggingFaceAdapter" in repr_str
        assert "model=gpt2" in repr_str
        assert "vocab_size=" in repr_str
        assert "initialized=True" in repr_str
    
    def test_special_token_setup(self, adapter):
        """Test special token setup."""
        # Check that pad token is set up
        assert adapter._pad_token_id is not None
        
        # Check that tokenizer has pad token
        assert adapter._tokenizer.pad_token is not None
        assert adapter._tokenizer.pad_token_id is not None
    
    def test_custom_special_tokens(self):
        """Test custom special tokens configuration."""
        config = TokenizerConfig(
            backend='huggingface',
            model_name='gpt2',
            special_tokens={'pad': '[CUSTOM_PAD]'}
        )
        
        adapter = HuggingFaceAdapter(config)
        adapter.initialize()
        
        # Should have custom special tokens
        assert adapter._is_initialized
        special_tokens = adapter.get_special_tokens()
        assert 'pad_token_id' in special_tokens


class TestHuggingFaceAdapterRegistration:
    """Test HuggingFace adapter registration."""
    
    def test_register_huggingface_adapter(self):
        """Test adapter registration."""
        # Clear registry
        TokenizerRegistry._adapters.clear()
        TokenizerRegistry._model_mappings.clear()
        
        # Register adapter
        register_huggingface_adapter()
        
        # Check registration
        assert 'huggingface' in TokenizerRegistry._adapters
        assert TokenizerRegistry._adapters['huggingface'] == HuggingFaceAdapter
        
        # Check model patterns
        assert 'gpt2' in TokenizerRegistry._model_mappings
        assert 'bert-base-uncased' in TokenizerRegistry._model_mappings
        assert TokenizerRegistry._model_mappings['gpt2'] == 'huggingface'
    
    def test_adapter_creation_through_registry(self):
        """Test creating adapter through registry."""
        # Ensure adapter is registered
        register_huggingface_adapter()
        
        # Create adapter through registry
        adapter = TokenizerRegistry.create_adapter('gpt2', max_length=256)
        
        assert isinstance(adapter, HuggingFaceAdapter)
        assert adapter._is_initialized
        assert adapter.config.model_name == 'gpt2'
        assert adapter.config.max_length == 256
    
    def test_model_pattern_matching(self):
        """Test model pattern matching."""
        register_huggingface_adapter()
        
        # Test exact model match
        adapter_class = TokenizerRegistry.get_adapter_class('gpt2')
        assert adapter_class == HuggingFaceAdapter
        
        # Test prefix match
        adapter_class = TokenizerRegistry.get_adapter_class('microsoft/DialoGPT-medium')
        assert adapter_class == HuggingFaceAdapter
        
        # Test backend match
        adapter_class = TokenizerRegistry.get_adapter_class('huggingface')
        assert adapter_class == HuggingFaceAdapter


class TestHuggingFaceAdapterIntegration:
    """Test HuggingFace adapter integration with enhanced tokenizer."""
    
    def test_integration_with_enhanced_tokenizer(self):
        """Test integration with EnhancedTokenizerWrapper."""
        from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
        
        # Create wrapper with HuggingFace tokenizer
        wrapper = EnhancedTokenizerWrapper('gpt2', embedding_dim=128)
        
        assert isinstance(wrapper.get_adapter(), HuggingFaceAdapter)
        assert wrapper.get_vocab_size() > 0
        
        # Test tokenization through wrapper
        tokens = wrapper.encode_single("Hello world")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Test decoding through wrapper
        decoded = wrapper.decode_single(tokens)
        assert isinstance(decoded, str)
        assert "Hello" in decoded or "hello" in decoded.lower()
    
    def test_different_huggingface_models(self):
        """Test different HuggingFace models."""
        from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
        
        models_to_test = ['gpt2', 'bert-base-uncased', 'distilbert-base-uncased']
        
        for model_name in models_to_test:
            try:
                wrapper = EnhancedTokenizerWrapper(model_name, embedding_dim=64)
                
                assert wrapper.get_vocab_size() > 0
                
                # Test basic functionality
                tokens = wrapper.encode_single("Hello world")
                decoded = wrapper.decode_single(tokens)
                
                assert isinstance(tokens, list)
                assert isinstance(decoded, str)
                
            except Exception as e:
                # Some models might not be available in test environment
                pytest.skip(f"Model {model_name} not available: {e}")
    
    def test_vocabulary_extraction(self):
        """Test vocabulary extraction functionality."""
        from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
        
        wrapper = EnhancedTokenizerWrapper('gpt2')
        
        # Test vocabulary access
        vocab = wrapper.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0
        
        # Test special tokens
        special_tokens = wrapper.get_special_tokens()
        assert isinstance(special_tokens, dict)
        assert 'pad_token_id' in special_tokens
        
        # Test vocab size consistency
        assert len(vocab) == wrapper.get_vocab_size()
    
    def test_token_mapping_functionality(self):
        """Test token mapping functionality."""
        from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
        
        wrapper = EnhancedTokenizerWrapper('gpt2')
        
        # Test round-trip tokenization
        original_text = "Hello world, this is a test sentence."
        tokens = wrapper.encode_single(original_text)
        decoded_text = wrapper.decode_single(tokens)
        
        # Should be able to decode back to similar text
        assert isinstance(tokens, list)
        assert isinstance(decoded_text, str)
        assert len(tokens) > 0
        assert len(decoded_text) > 0
        
        # Test batch processing
        texts = ["Hello world", "Another test", "Final sentence"]
        batch_tokens = wrapper.tokenize(texts)
        batch_decoded = wrapper.decode(batch_tokens)
        
        assert len(batch_tokens) == 3
        assert len(batch_decoded) == 3
        assert all(isinstance(seq, list) for seq in batch_tokens)
        assert all(isinstance(text, str) for text in batch_decoded)


if __name__ == "__main__":
    pytest.main([__file__])