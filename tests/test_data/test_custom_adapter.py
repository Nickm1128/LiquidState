#!/usr/bin/env python3
"""
Tests for custom tokenizer adapter.

This module tests the custom tokenizer adapter functionality including
interface validation, automatic vocabulary size detection, and error handling.
"""

import pytest
import tempfile
import os
from typing import List, Dict

from src.lsm.data.adapters.custom_adapter import (
    CustomAdapter, CustomTokenizerWrapper, CustomTokenizerProtocol,
    create_custom_tokenizer, create_custom_tokenizer_from_object
)
from src.lsm.data.enhanced_tokenization import TokenizerConfig, TokenizerRegistry
from src.lsm.utils.lsm_exceptions import TokenizerError, TokenizerLoadError


class SimpleCharTokenizer:
    """Simple character-based tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 256):
        self._vocab_size = vocab_size
    
    def encode(self, text: str) -> List[int]:
        """Encode text as character codes."""
        return [min(ord(c), self._vocab_size - 1) for c in text]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode character codes to text."""
        return ''.join(chr(tid) for tid in token_ids if 0 <= tid < self._vocab_size)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return {chr(i): i for i in range(min(128, self._vocab_size))}  # ASCII subset
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special tokens."""
        return {
            'pad_token_id': 0,
            'unk_token_id': 1,
            'bos_token_id': 2,
            'eos_token_id': 3
        }


class IncompleteTokenizer:
    """Tokenizer missing required methods for testing validation."""
    
    def encode(self, text: str) -> List[int]:
        return [1, 2, 3]
    
    # Missing decode and get_vocab_size methods


class BrokenTokenizer:
    """Tokenizer with broken functionality for testing error handling."""
    
    def encode(self, text: str) -> List[int]:
        raise ValueError("Broken encode method")
    
    def decode(self, token_ids: List[int]) -> str:
        raise ValueError("Broken decode method")
    
    def get_vocab_size(self) -> int:
        return -1  # Invalid vocab size


class TestCustomTokenizerWrapper:
    """Test CustomTokenizerWrapper functionality."""
    
    def test_wrapper_creation(self):
        """Test creating wrapper with functions."""
        def encode_fn(text):
            return [ord(c) for c in text]
        
        def decode_fn(token_ids):
            return ''.join(chr(tid) for tid in token_ids)
        
        wrapper = CustomTokenizerWrapper(encode_fn, decode_fn, vocab_size=256)
        
        assert wrapper.get_vocab_size() == 256
        assert wrapper.encode("hello") == [104, 101, 108, 108, 111]
        assert wrapper.decode([104, 101, 108, 108, 111]) == "hello"
    
    def test_wrapper_auto_vocab_detection(self):
        """Test automatic vocabulary size detection."""
        def encode_fn(text):
            return [ord(c) for c in text]
        
        def decode_fn(token_ids):
            return ''.join(chr(tid) for tid in token_ids)
        
        wrapper = CustomTokenizerWrapper(encode_fn, decode_fn)  # No vocab_size provided
        
        # Should auto-detect based on test texts
        assert wrapper.get_vocab_size() > 0
        assert wrapper.encode("A") == [65]
    
    def test_wrapper_with_vocab(self):
        """Test wrapper with vocabulary mapping."""
        vocab = {'a': 0, 'b': 1, 'c': 2}
        
        def encode_fn(text):
            return [vocab.get(c, len(vocab)) for c in text]
        
        def decode_fn(token_ids):
            reverse_vocab = {v: k for k, v in vocab.items()}
            return ''.join(reverse_vocab.get(tid, '?') for tid in token_ids)
        
        wrapper = CustomTokenizerWrapper(encode_fn, decode_fn, vocab=vocab)
        
        assert wrapper.get_vocab_size() == 3
        assert wrapper.get_vocab() == vocab
        assert wrapper.encode("abc") == [0, 1, 2]
        assert wrapper.decode([0, 1, 2]) == "abc"


class TestCustomAdapter:
    """Test CustomAdapter functionality."""
    
    def test_adapter_with_tokenizer_object(self):
        """Test adapter with custom tokenizer object."""
        tokenizer = SimpleCharTokenizer(vocab_size=128)
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_char',
            max_length=10,
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        assert adapter.get_vocab_size() == 128
        assert adapter._is_initialized
        
        # Test tokenization
        result = adapter.tokenize(["hello"])
        assert len(result) == 1
        assert len(result[0]) <= 10  # Respects max_length
        
        # Test decoding
        decoded = adapter.decode(result)
        assert isinstance(decoded, list)
        assert len(decoded) == 1
    
    def test_adapter_with_functions(self):
        """Test adapter with encode/decode functions."""
        def encode_fn(text):
            return [ord(c) for c in text[:5]]  # Limit to 5 chars
        
        def decode_fn(token_ids):
            return ''.join(chr(tid) for tid in token_ids)
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_fn',
            max_length=10,
            backend_specific_config={
                'encode_fn': encode_fn,
                'decode_fn': decode_fn,
                'vocab_size': 256
            }
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        assert adapter.get_vocab_size() == 256
        
        # Test tokenization
        result = adapter.tokenize(["hello world"])
        assert len(result) == 1
        assert len(result[0]) <= 10
    
    def test_adapter_validation(self):
        """Test adapter validation functionality."""
        tokenizer = SimpleCharTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_validation',
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        # Validation should pass
        assert adapter._is_initialized
        
        # Test disabling validation
        adapter.set_validation_enabled(False)
        assert not adapter._validation_enabled
    
    def test_adapter_interface_validation_failure(self):
        """Test adapter validation with incomplete tokenizer."""
        incomplete_tokenizer = IncompleteTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_incomplete',
            backend_specific_config={'tokenizer': incomplete_tokenizer}
        )
        
        adapter = CustomAdapter(config)
        
        with pytest.raises(TokenizerError, match="must implement 'decode' method"):
            adapter.initialize()
    
    def test_adapter_functionality_validation_failure(self):
        """Test adapter validation with broken tokenizer."""
        broken_tokenizer = BrokenTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_broken',
            backend_specific_config={'tokenizer': broken_tokenizer}
        )
        
        adapter = CustomAdapter(config)
        
        with pytest.raises(TokenizerError, match="validation failed"):
            adapter.initialize()
    
    def test_adapter_special_tokens(self):
        """Test special token handling."""
        tokenizer = SimpleCharTokenizer()
        special_tokens = {'pad_token_id': 0, 'unk_token_id': 1}
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_special',
            special_tokens=special_tokens,
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        special = adapter.get_special_tokens()
        assert 'pad_token_id' in special
        assert 'unk_token_id' in special
        assert special['pad_token_id'] == 0
    
    def test_adapter_padding_and_truncation(self):
        """Test padding and truncation functionality."""
        tokenizer = SimpleCharTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_padding',
            max_length=5,
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        # Test truncation
        result = adapter.tokenize(["hello world"], truncation=True, padding=False)
        assert len(result[0]) <= 5
        
        # Test padding
        result = adapter.tokenize(["hi", "hello"], padding=True)
        assert len(result[0]) == len(result[1])  # Same length after padding
    
    def test_adapter_batch_processing(self):
        """Test batch processing functionality."""
        tokenizer = SimpleCharTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_batch',
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        texts = ["hello", "world", "test"]
        result = adapter.tokenize(texts)
        
        assert len(result) == 3
        assert all(isinstance(seq, list) for seq in result)
        
        # Test decoding
        decoded = adapter.decode(result)
        assert len(decoded) == 3
        assert all(isinstance(text, str) for text in decoded)
    
    def test_adapter_info(self):
        """Test getting adapter information."""
        tokenizer = SimpleCharTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_info',
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        info = adapter.get_tokenizer_info()
        
        assert 'model_name' in info
        assert 'vocab_size' in info
        assert 'tokenizer_class' in info
        assert info['model_name'] == 'test_info'
        assert info['vocab_size'] == 256
    
    def test_adapter_save_config(self):
        """Test saving adapter configuration."""
        tokenizer = SimpleCharTokenizer()
        
        config = TokenizerConfig(
            backend='custom',
            model_name='test_save',
            backend_specific_config={'tokenizer': tokenizer}
        )
        
        adapter = CustomAdapter(config)
        adapter.initialize()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter.save_adapter_config(temp_dir)
            
            config_path = os.path.join(temp_dir, 'custom_adapter_config.json')
            assert os.path.exists(config_path)
    
    def test_adapter_load_config_error(self):
        """Test that loading custom adapter raises error."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(TokenizerLoadError, match="cannot be loaded"):
                CustomAdapter.load_adapter_config(temp_dir)
    
    def test_adapter_missing_config(self):
        """Test adapter with missing backend config."""
        config = TokenizerConfig(
            backend='custom',
            model_name='test_missing',
            backend_specific_config=None
        )
        
        adapter = CustomAdapter(config)
        
        with pytest.raises(TokenizerError, match="requires backend_specific_config"):
            adapter.initialize()
    
    def test_adapter_invalid_config(self):
        """Test adapter with invalid backend config."""
        config = TokenizerConfig(
            backend='custom',
            model_name='test_invalid',
            backend_specific_config={'invalid': 'config'}
        )
        
        adapter = CustomAdapter(config)
        
        with pytest.raises(TokenizerError, match="must provide either"):
            adapter.initialize()


class TestConvenienceFunctions:
    """Test convenience functions for creating custom adapters."""
    
    def test_create_custom_tokenizer(self):
        """Test create_custom_tokenizer function."""
        def encode_fn(text):
            return [ord(c) for c in text]
        
        def decode_fn(token_ids):
            return ''.join(chr(tid) for tid in token_ids)
        
        adapter = create_custom_tokenizer(
            encode_fn, decode_fn, 
            vocab_size=256, 
            model_name="test_convenience"
        )
        
        assert adapter._is_initialized
        assert adapter.get_vocab_size() == 256
        assert adapter.config.model_name == "test_convenience"
        
        # Test functionality
        result = adapter.tokenize(["hello"])
        assert len(result) == 1
        
        decoded = adapter.decode(result)
        assert decoded[0] == "hello"
    
    def test_create_custom_tokenizer_from_object(self):
        """Test create_custom_tokenizer_from_object function."""
        tokenizer = SimpleCharTokenizer(vocab_size=128)
        
        adapter = create_custom_tokenizer_from_object(
            tokenizer, 
            model_name="test_object"
        )
        
        assert adapter._is_initialized
        assert adapter.get_vocab_size() == 128
        assert adapter.config.model_name == "test_object"
        
        # Test functionality
        result = adapter.tokenize(["test"])
        assert len(result) == 1
    
    def test_create_with_special_tokens(self):
        """Test creating custom tokenizer with special tokens."""
        def encode_fn(text):
            return [ord(c) for c in text]
        
        def decode_fn(token_ids):
            return ''.join(chr(tid) for tid in token_ids)
        
        special_tokens = {'pad_token_id': 0, 'unk_token_id': 1}
        
        adapter = create_custom_tokenizer(
            encode_fn, decode_fn,
            special_tokens=special_tokens
        )
        
        assert adapter.get_special_tokens()['pad_token_id'] == 0
        assert adapter.get_special_tokens()['unk_token_id'] == 1


class TestTokenizerRegistry:
    """Test TokenizerRegistry integration with custom adapter."""
    
    def test_registry_custom_adapter(self):
        """Test that custom adapter is registered."""
        backends = TokenizerRegistry.list_available_backends()
        assert 'custom' in backends
    
    def test_create_custom_adapter_via_registry(self):
        """Test creating custom adapter through registry."""
        def encode_fn(text):
            return [ord(c) for c in text]
        
        def decode_fn(token_ids):
            return ''.join(chr(tid) for tid in token_ids)
        
        adapter = TokenizerRegistry.create_adapter(
            'custom',
            max_length=10,
            backend_specific_config={
                'encode_fn': encode_fn,
                'decode_fn': decode_fn,
                'vocab_size': 256
            }
        )
        
        assert isinstance(adapter, CustomAdapter)
        assert adapter._is_initialized
        assert adapter.get_vocab_size() == 256


class TestProtocolCompliance:
    """Test CustomTokenizerProtocol compliance."""
    
    def test_protocol_compliance(self):
        """Test that SimpleCharTokenizer implements the protocol."""
        tokenizer = SimpleCharTokenizer()
        
        # Should be recognized as implementing the protocol
        assert isinstance(tokenizer, CustomTokenizerProtocol)
        
        # Test protocol methods
        token_ids = tokenizer.encode("test")
        assert isinstance(token_ids, list)
        assert all(isinstance(tid, int) for tid in token_ids)
        
        decoded = tokenizer.decode(token_ids)
        assert isinstance(decoded, str)
        
        vocab_size = tokenizer.get_vocab_size()
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
    
    def test_protocol_non_compliance(self):
        """Test that incomplete tokenizer doesn't implement protocol."""
        incomplete = IncompleteTokenizer()
        
        # Should not be recognized as implementing the protocol
        assert not isinstance(incomplete, CustomTokenizerProtocol)


if __name__ == '__main__':
    pytest.main([__file__])