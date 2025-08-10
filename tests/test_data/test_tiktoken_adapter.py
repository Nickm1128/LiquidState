#!/usr/bin/env python3
"""
Tests for TiktokenAdapter.

This module contains comprehensive tests for the OpenAI tiktoken adapter,
including tokenization, decoding, special token handling, and model support.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch

# Import the adapter and related classes
from src.lsm.data.adapters.tiktoken_adapter import TiktokenAdapter
from src.lsm.data.enhanced_tokenization import TokenizerConfig
from src.lsm.utils.lsm_exceptions import TokenizerError, TokenizerLoadError


class TestTiktokenAdapter:
    """Test cases for TiktokenAdapter."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic tokenizer configuration for testing."""
        return TokenizerConfig(
            backend='tiktoken',
            model_name='gpt-3.5-turbo',
            max_length=512
        )
    
    @pytest.fixture
    def custom_config(self):
        """Custom tokenizer configuration with special tokens."""
        return TokenizerConfig(
            backend='tiktoken',
            model_name='gpt-4',
            max_length=1024,
            special_tokens={
                'pad_token': '<PAD>',
                'eos_token': '<END>'
            }
        )
    
    def test_initialization_success(self, basic_config):
        """Test successful adapter initialization."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            assert adapter._is_initialized
            assert adapter.get_vocab_size() > 0
            assert adapter._encoding_name == 'cl100k_base'
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_initialization_unsupported_model(self):
        """Test initialization with unsupported model."""
        config = TokenizerConfig(
            backend='tiktoken',
            model_name='unsupported-model',
            max_length=512
        )
        
        try:
            adapter = TiktokenAdapter(config)
            with pytest.raises(TokenizerError):
                adapter.initialize()
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_supported_models_list(self):
        """Test that supported models list is comprehensive."""
        supported = TiktokenAdapter.list_supported_models()
        
        # Check for key OpenAI models
        assert 'gpt-3.5-turbo' in supported
        assert 'gpt-4' in supported
        assert 'text-davinci-003' in supported
        assert 'cl100k_base' in supported
        assert 'p50k_base' in supported
        assert 'r50k_base' in supported
    
    def test_tokenize_single_text(self, basic_config):
        """Test tokenizing a single text."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            text = "Hello, world!"
            result = adapter.tokenize(text)
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], list)
            assert all(isinstance(token_id, int) for token_id in result[0])
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_tokenize_multiple_texts(self, basic_config):
        """Test tokenizing multiple texts."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            texts = ["Hello, world!", "How are you?", "This is a test."]
            result = adapter.tokenize(texts)
            
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(seq, list) for seq in result)
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_tokenize_with_padding(self, basic_config):
        """Test tokenization with padding."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            texts = ["Short", "This is a much longer text that should be longer"]
            result = adapter.tokenize(texts, padding=True)
            
            # All sequences should have the same length when padded
            lengths = [len(seq) for seq in result]
            assert len(set(lengths)) == 1  # All lengths should be the same
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_tokenize_with_truncation(self, basic_config):
        """Test tokenization with truncation."""
        try:
            # Use a very small max_length to force truncation
            config = TokenizerConfig(
                backend='tiktoken',
                model_name='gpt-3.5-turbo',
                max_length=5
            )
            adapter = TiktokenAdapter(config)
            adapter.initialize()
            
            long_text = "This is a very long text that should definitely be truncated because it exceeds the maximum length"
            result = adapter.tokenize(long_text, truncation=True)
            
            assert len(result[0]) <= 5
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_decode_single_sequence(self, basic_config):
        """Test decoding a single sequence."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            text = "Hello, world!"
            token_ids = adapter.tokenize(text, add_special_tokens=False, padding=False)[0]
            decoded = adapter.decode(token_ids)
            
            assert isinstance(decoded, str)
            # The decoded text should be similar to the original
            assert "Hello" in decoded
            assert "world" in decoded
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_decode_multiple_sequences(self, basic_config):
        """Test decoding multiple sequences."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            texts = ["Hello", "World"]
            token_ids = adapter.tokenize(texts, add_special_tokens=False, padding=False)
            decoded = adapter.decode(token_ids)
            
            assert isinstance(decoded, list)
            assert len(decoded) == 2
            assert all(isinstance(text, str) for text in decoded)
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_special_tokens_setup(self, basic_config):
        """Test special tokens are properly set up."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            special_tokens = adapter.get_special_tokens()
            
            assert isinstance(special_tokens, dict)
            assert 'eos_token_id' in special_tokens
            assert 'pad_token_id' in special_tokens
            assert isinstance(special_tokens['eos_token_id'], int)
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_vocab_size(self, basic_config):
        """Test vocabulary size retrieval."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            vocab_size = adapter.get_vocab_size()
            
            assert isinstance(vocab_size, int)
            assert vocab_size > 0
            # cl100k_base should have around 100k tokens
            assert vocab_size > 50000
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_get_vocab_partial(self, basic_config):
        """Test vocabulary retrieval (partial)."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            vocab = adapter.get_vocab()
            
            assert isinstance(vocab, dict)
            # Should have some tokens, but not the full vocabulary
            assert len(vocab) > 0
            assert len(vocab) < adapter.get_vocab_size()
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_tokenizer_info(self, basic_config):
        """Test tokenizer information retrieval."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            info = adapter.get_tokenizer_info()
            
            assert isinstance(info, dict)
            assert 'model_name' in info
            assert 'encoding_name' in info
            assert 'vocab_size' in info
            assert 'backend' in info
            assert info['backend'] == 'tiktoken'
            assert info['model_name'] == 'gpt-3.5-turbo'
            assert info['encoding_name'] == 'cl100k_base'
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_different_model_encodings(self):
        """Test different models use correct encodings."""
        test_cases = [
            ('gpt-3.5-turbo', 'cl100k_base'),
            ('gpt-4', 'cl100k_base'),
            ('text-davinci-003', 'p50k_base'),
            ('text-davinci-001', 'r50k_base'),
            ('cl100k_base', 'cl100k_base'),
        ]
        
        try:
            for model_name, expected_encoding in test_cases:
                config = TokenizerConfig(
                    backend='tiktoken',
                    model_name=model_name,
                    max_length=512
                )
                adapter = TiktokenAdapter(config)
                adapter.initialize()
                
                assert adapter._encoding_name == expected_encoding
                
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_save_and_load_config(self, basic_config):
        """Test saving and loading adapter configuration."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save configuration
                adapter.save_adapter_config(temp_dir)
                
                # Check config file was created
                config_path = os.path.join(temp_dir, 'tiktoken_adapter_config.json')
                assert os.path.exists(config_path)
                
                # Load configuration
                loaded_adapter = TiktokenAdapter.load_adapter_config(temp_dir)
                
                assert loaded_adapter._is_initialized
                assert loaded_adapter.config.model_name == basic_config.model_name
                assert loaded_adapter.get_vocab_size() == adapter.get_vocab_size()
                
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_load_nonexistent_config(self):
        """Test loading from nonexistent config path."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                nonexistent_path = os.path.join(temp_dir, 'nonexistent')
                
                with pytest.raises(TokenizerLoadError):
                    TiktokenAdapter.load_adapter_config(nonexistent_path)
                    
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_not_initialized_error(self, basic_config):
        """Test that methods raise error when not initialized."""
        try:
            adapter = TiktokenAdapter(basic_config)
            # Don't call initialize()
            
            with pytest.raises(TokenizerError):
                adapter.tokenize("test")
            
            with pytest.raises(TokenizerError):
                adapter.decode([1, 2, 3])
            
            with pytest.raises(TokenizerError):
                adapter.get_vocab_size()
            
            with pytest.raises(TokenizerError):
                adapter.get_vocab()
            
            with pytest.raises(TokenizerError):
                adapter.get_special_tokens()
                
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_custom_special_tokens(self, custom_config):
        """Test custom special tokens configuration."""
        try:
            adapter = TiktokenAdapter(custom_config)
            adapter.initialize()
            
            special_tokens = adapter.get_special_tokens()
            
            # Should have the default tokens plus any custom ones that were successfully encoded
            assert 'eos_token_id' in special_tokens
            assert 'pad_token_id' in special_tokens
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    def test_repr(self, basic_config):
        """Test string representation."""
        try:
            adapter = TiktokenAdapter(basic_config)
            adapter.initialize()
            
            repr_str = repr(adapter)
            
            assert 'TiktokenAdapter' in repr_str
            assert 'gpt-3.5-turbo' in repr_str
            assert 'cl100k_base' in repr_str
            assert 'initialized=True' in repr_str
            
        except ImportError:
            pytest.skip("tiktoken not available")
    
    @patch('src.lsm.data.adapters.tiktoken_adapter.TIKTOKEN_AVAILABLE', False)
    def test_tiktoken_not_available(self, basic_config):
        """Test behavior when tiktoken is not available."""
        with pytest.raises(TokenizerError, match="tiktoken library not available"):
            TiktokenAdapter(basic_config)


if __name__ == '__main__':
    pytest.main([__file__])