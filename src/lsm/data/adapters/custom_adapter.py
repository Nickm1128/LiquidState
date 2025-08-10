#!/usr/bin/env python3
"""
Custom tokenizer adapter for enhanced tokenization system.

This module provides an adapter interface for user-provided custom tokenizers,
allowing them to work with the enhanced tokenizer system through automatic
vocabulary size detection and validation.
"""

import os
import json
import inspect
from typing import List, Dict, Any, Optional, Union, Callable, Protocol, runtime_checkable

from ..enhanced_tokenization import TokenizerAdapter, TokenizerConfig, TokenizerRegistry
from ...utils.lsm_exceptions import TokenizerError, TokenizerLoadError, InvalidInputError
from ...utils.lsm_logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class CustomTokenizerProtocol(Protocol):
    """
    Protocol defining the interface that custom tokenizers must implement.
    
    This protocol ensures that custom tokenizers provide the minimum required
    methods to work with the enhanced tokenization system.
    """
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        ...
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            
        Returns:
            Decoded text
        """
        ...
    
    def get_vocab_size(self) -> int:
        """
        Get vocabulary size.
        
        Returns:
            Size of the vocabulary
        """
        ...


class CustomTokenizerWrapper:
    """
    Wrapper for custom tokenizers that don't implement the full protocol.
    
    This class provides a bridge between simple tokenizer functions and
    the CustomTokenizerProtocol interface.
    """
    
    def __init__(self, encode_fn: Callable[[str], List[int]], 
                 decode_fn: Callable[[List[int]], str],
                 vocab_size: Optional[int] = None,
                 vocab: Optional[Dict[str, int]] = None):
        """
        Initialize custom tokenizer wrapper.
        
        Args:
            encode_fn: Function to encode text to token IDs
            decode_fn: Function to decode token IDs to text
            vocab_size: Vocabulary size (will be auto-detected if None)
            vocab: Vocabulary mapping (optional)
        """
        self._encode_fn = encode_fn
        self._decode_fn = decode_fn
        self._vocab = vocab or {}
        self._vocab_size = vocab_size
        
        # Auto-detect vocabulary size if not provided
        if self._vocab_size is None:
            self._vocab_size = self._detect_vocab_size()
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return self._encode_fn(text)
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self._decode_fn(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self._vocab
    
    def _detect_vocab_size(self) -> int:
        """
        Auto-detect vocabulary size by testing the tokenizer.
        
        Returns:
            Detected vocabulary size
        """
        if self._vocab:
            return len(self._vocab)
        
        # Try to detect vocab size by encoding various texts and finding max token ID
        test_texts = [
            "Hello world!",
            "The quick brown fox jumps over the lazy dog.",
            "1234567890",
            "!@#$%^&*()",
            "αβγδε",  # Greek letters
            "こんにちは",  # Japanese
            "🚀🌟💡",  # Emojis
        ]
        
        max_token_id = 0
        try:
            for text in test_texts:
                token_ids = self._encode_fn(text)
                if token_ids:
                    max_token_id = max(max_token_id, max(token_ids))
            
            # Estimate vocab size as max_token_id + 1 (assuming 0-based indexing)
            estimated_size = max_token_id + 1
            
            logger.info(f"Auto-detected vocabulary size: {estimated_size} (max token ID: {max_token_id})")
            return estimated_size
            
        except Exception as e:
            logger.warning(f"Could not auto-detect vocabulary size: {e}")
            # Return a reasonable default
            return 10000


class CustomAdapter(TokenizerAdapter):
    """
    Adapter for user-provided custom tokenizers.
    
    This adapter wraps custom tokenizers to work with the enhanced tokenizer
    system, providing automatic vocabulary size detection, validation, and
    error handling.
    """
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialize custom adapter.
        
        Args:
            config: Tokenizer configuration
        """
        super().__init__(config)
        
        self._custom_tokenizer = None
        self._vocab = {}
        self._special_tokens = {}
        
        # Validation settings
        self._validation_enabled = True
        self._max_validation_samples = 100
    
    def initialize(self) -> None:
        """Initialize the custom tokenizer."""
        try:
            # Get custom tokenizer from backend-specific config
            if not self.config.backend_specific_config:
                raise TokenizerError("Custom tokenizer requires backend_specific_config")
            
            tokenizer_config = self.config.backend_specific_config
            
            # Handle different ways of providing custom tokenizer
            if 'tokenizer' in tokenizer_config:
                # Direct tokenizer object
                self._custom_tokenizer = tokenizer_config['tokenizer']
            elif 'encode_fn' in tokenizer_config and 'decode_fn' in tokenizer_config:
                # Separate encode/decode functions
                self._custom_tokenizer = CustomTokenizerWrapper(
                    encode_fn=tokenizer_config['encode_fn'],
                    decode_fn=tokenizer_config['decode_fn'],
                    vocab_size=tokenizer_config.get('vocab_size'),
                    vocab=tokenizer_config.get('vocab', {})
                )
            else:
                raise TokenizerError(
                    "Custom tokenizer must provide either 'tokenizer' object or "
                    "'encode_fn' and 'decode_fn' functions"
                )
            
            # Validate tokenizer interface
            self._validate_tokenizer_interface()
            
            # Get vocabulary information
            self._vocab_size = self._custom_tokenizer.get_vocab_size()
            
            if hasattr(self._custom_tokenizer, 'get_vocab'):
                self._vocab = self._custom_tokenizer.get_vocab()
            
            # Set up special tokens
            self._setup_special_tokens()
            
            # Validate tokenizer functionality
            if self._validation_enabled:
                self._validate_tokenizer_functionality()
            
            self._is_initialized = True
            
            logger.info(f"Initialized custom tokenizer with vocab size {self._vocab_size}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to initialize custom tokenizer: {str(e)}")
    
    def _validate_tokenizer_interface(self) -> None:
        """
        Validate that the custom tokenizer implements required interface.
        
        Raises:
            TokenizerError: If tokenizer doesn't implement required methods
        """
        required_methods = ['encode', 'decode', 'get_vocab_size']
        
        for method_name in required_methods:
            if not hasattr(self._custom_tokenizer, method_name):
                raise TokenizerError(
                    f"Custom tokenizer must implement '{method_name}' method"
                )
            
            method = getattr(self._custom_tokenizer, method_name)
            if not callable(method):
                raise TokenizerError(
                    f"Custom tokenizer '{method_name}' must be callable"
                )
        
        # Check if tokenizer implements the protocol
        if not isinstance(self._custom_tokenizer, CustomTokenizerProtocol):
            logger.warning(
                "Custom tokenizer does not fully implement CustomTokenizerProtocol. "
                "Some features may not work correctly."
            )
    
    def _validate_tokenizer_functionality(self) -> None:
        """
        Validate that the custom tokenizer works correctly.
        
        Raises:
            TokenizerError: If tokenizer functionality is invalid
        """
        test_cases = [
            "Hello world!",
            "The quick brown fox.",
            "123",
            "",  # Empty string
            " ",  # Whitespace
        ]
        
        for i, text in enumerate(test_cases):
            if i >= self._max_validation_samples:
                break
            
            try:
                # Test encode
                token_ids = self._custom_tokenizer.encode(text)
                
                if not isinstance(token_ids, list):
                    raise TokenizerError(
                        f"encode() must return a list, got {type(token_ids)}"
                    )
                
                if not all(isinstance(tid, int) for tid in token_ids):
                    raise TokenizerError(
                        "encode() must return a list of integers"
                    )
                
                # Test decode
                decoded_text = self._custom_tokenizer.decode(token_ids)
                
                if not isinstance(decoded_text, str):
                    raise TokenizerError(
                        f"decode() must return a string, got {type(decoded_text)}"
                    )
                
                # Check vocab size consistency
                vocab_size = self._custom_tokenizer.get_vocab_size()
                if not isinstance(vocab_size, int) or vocab_size <= 0:
                    raise TokenizerError(
                        f"get_vocab_size() must return a positive integer, got {vocab_size}"
                    )
                
                # Check token IDs are within vocab range
                if token_ids and max(token_ids) >= vocab_size:
                    logger.warning(
                        f"Token ID {max(token_ids)} exceeds vocab size {vocab_size}. "
                        "This may indicate incorrect vocabulary size."
                    )
                
            except Exception as e:
                raise TokenizerError(
                    f"Custom tokenizer validation failed on text '{text}': {str(e)}"
                )
        
        logger.info(f"Custom tokenizer validation passed for {len(test_cases)} test cases")
    
    def _setup_special_tokens(self) -> None:
        """Set up special tokens for the custom tokenizer."""
        # Get special tokens from config
        if self.config.special_tokens:
            self._special_tokens.update(self.config.special_tokens)
        
        # Try to get special tokens from tokenizer if it supports them
        if hasattr(self._custom_tokenizer, 'get_special_tokens'):
            try:
                tokenizer_special_tokens = self._custom_tokenizer.get_special_tokens()
                if isinstance(tokenizer_special_tokens, dict):
                    self._special_tokens.update(tokenizer_special_tokens)
            except Exception as e:
                logger.warning(f"Could not get special tokens from custom tokenizer: {e}")
        
        # Set default special tokens if not provided
        if 'pad_token_id' not in self._special_tokens:
            self._special_tokens['pad_token_id'] = 0
        if 'unk_token_id' not in self._special_tokens:
            self._special_tokens['unk_token_id'] = 1
    
    def tokenize(self, texts: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: bool = True, 
                 truncation: bool = True) -> List[List[int]]:
        """
        Tokenize texts to token IDs.
        
        Args:
            texts: Single text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            List of token ID sequences
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # Tokenize each text
            token_sequences = []
            for text in texts:
                token_ids = self._custom_tokenizer.encode(text)
                
                # Add special tokens if requested (before truncation to account for them)
                if add_special_tokens:
                    token_ids = self._add_special_tokens(token_ids)
                
                # Apply truncation after adding special tokens
                if truncation and len(token_ids) > self.config.max_length:
                    token_ids = token_ids[:self.config.max_length]
                
                token_sequences.append(token_ids)
            
            # Apply padding
            if padding:
                token_sequences = self._pad_sequences(token_sequences)
            
            return token_sequences
            
        except Exception as e:
            raise TokenizerError(f"Custom tokenization failed: {str(e)}")
    
    def _add_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Add special tokens to token sequence.
        
        Args:
            token_ids: Original token IDs
            
        Returns:
            Token IDs with special tokens added
        """
        # Add BOS token if available
        if 'bos_token_id' in self._special_tokens:
            token_ids = [self._special_tokens['bos_token_id']] + token_ids
        
        # Add EOS token if available
        if 'eos_token_id' in self._special_tokens:
            token_ids = token_ids + [self._special_tokens['eos_token_id']]
        
        return token_ids
    
    def _pad_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """
        Pad sequences to the same length.
        
        Args:
            sequences: List of token sequences
            
        Returns:
            Padded sequences
        """
        if not sequences:
            return sequences
        
        # Find maximum length
        max_length = max(len(seq) for seq in sequences)
        max_length = min(max_length, self.config.max_length)
        
        # Pad sequences
        pad_token_id = self._special_tokens.get('pad_token_id', 0)
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) > max_length:
                seq = seq[:max_length]
            elif len(seq) < max_length:
                seq = seq + [pad_token_id] * (max_length - len(seq))
            padded_sequences.append(seq)
        
        return padded_sequences
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], 
               skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text(s)
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        try:
            # Handle single sequence
            if isinstance(token_ids[0], int):
                if skip_special_tokens:
                    token_ids = self._remove_special_tokens(token_ids)
                return self._custom_tokenizer.decode(token_ids)
            
            # Handle batch of sequences
            decoded_texts = []
            for seq in token_ids:
                if skip_special_tokens:
                    seq = self._remove_special_tokens(seq)
                decoded_texts.append(self._custom_tokenizer.decode(seq))
            
            return decoded_texts
            
        except Exception as e:
            raise TokenizerError(f"Custom decoding failed: {str(e)}")
    
    def _remove_special_tokens(self, token_ids: List[int]) -> List[int]:
        """
        Remove special tokens from token sequence.
        
        Args:
            token_ids: Token IDs with special tokens
            
        Returns:
            Token IDs without special tokens
        """
        special_token_ids = set(self._special_tokens.values())
        return [tid for tid in token_ids if tid not in special_token_ids]
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._vocab
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._special_tokens.copy()
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the custom tokenizer.
        
        Returns:
            Dictionary with tokenizer information
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        info = {
            'model_name': self.config.model_name,
            'vocab_size': self._vocab_size,
            'max_length': self.config.max_length,
            'special_tokens': self._special_tokens,
            'tokenizer_class': self._custom_tokenizer.__class__.__name__,
            'has_vocab_mapping': bool(self._vocab),
            'validation_enabled': self._validation_enabled
        }
        
        # Add custom tokenizer specific info if available
        if hasattr(self._custom_tokenizer, 'get_info'):
            try:
                custom_info = self._custom_tokenizer.get_info()
                if isinstance(custom_info, dict):
                    info.update(custom_info)
            except Exception as e:
                logger.warning(f"Could not get custom tokenizer info: {e}")
        
        return info
    
    def set_validation_enabled(self, enabled: bool) -> None:
        """
        Enable or disable tokenizer validation.
        
        Args:
            enabled: Whether to enable validation
        """
        self._validation_enabled = enabled
        logger.info(f"Custom tokenizer validation {'enabled' if enabled else 'disabled'}")
    
    @classmethod
    def load_adapter_config(cls, load_path: str) -> 'CustomAdapter':
        """
        Load adapter from saved configuration.
        
        Args:
            load_path: Directory path to load config from
            
        Returns:
            Loaded CustomAdapter instance
            
        Note:
            Custom tokenizers cannot be fully serialized, so this method
            will raise an error. Users must recreate custom adapters manually.
        """
        raise TokenizerLoadError(
            load_path, 
            "Custom tokenizers cannot be loaded from saved configuration. "
            "Please recreate the custom adapter with your tokenizer implementation."
        )
    
    def save_adapter_config(self, save_path: str) -> None:
        """
        Save adapter-specific configuration.
        
        Args:
            save_path: Directory path to save config
            
        Note:
            Custom tokenizers cannot be fully serialized. Only metadata is saved.
        """
        config_dict = {
            'backend': self.config.backend,
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'special_tokens': self.config.special_tokens,
            'vocab_size': self.get_vocab_size() if self._is_initialized else None,
            'tokenizer_info': self.get_tokenizer_info() if self._is_initialized else None,
            'note': 'Custom tokenizers cannot be fully serialized. This is metadata only.'
        }
        
        config_path = os.path.join(save_path, 'custom_adapter_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.warning(
            "Custom tokenizer metadata saved, but tokenizer implementation cannot be serialized. "
            "You will need to recreate the custom adapter manually when loading."
        )
    
    def __repr__(self) -> str:
        return (f"CustomAdapter(model={self.config.model_name}, "
                f"vocab_size={self._vocab_size}, initialized={self._is_initialized})")


def create_custom_tokenizer(encode_fn: Callable[[str], List[int]], 
                           decode_fn: Callable[[List[int]], str],
                           vocab_size: Optional[int] = None,
                           vocab: Optional[Dict[str, int]] = None,
                           special_tokens: Optional[Dict[str, int]] = None,
                           max_length: int = 512,
                           model_name: str = "custom") -> CustomAdapter:
    """
    Convenience function to create a custom tokenizer adapter.
    
    Args:
        encode_fn: Function to encode text to token IDs
        decode_fn: Function to decode token IDs to text
        vocab_size: Vocabulary size (will be auto-detected if None)
        vocab: Vocabulary mapping (optional)
        special_tokens: Special token IDs (optional)
        max_length: Maximum sequence length
        model_name: Name for the custom tokenizer
        
    Returns:
        Initialized CustomAdapter instance
        
    Example:
        >>> def my_encode(text):
        ...     return [ord(c) for c in text]
        >>> 
        >>> def my_decode(token_ids):
        ...     return ''.join(chr(tid) for tid in token_ids)
        >>> 
        >>> adapter = create_custom_tokenizer(my_encode, my_decode, vocab_size=256)
    """
    config = TokenizerConfig(
        backend='custom',
        model_name=model_name,
        max_length=max_length,
        special_tokens=special_tokens,
        backend_specific_config={
            'encode_fn': encode_fn,
            'decode_fn': decode_fn,
            'vocab_size': vocab_size,
            'vocab': vocab
        }
    )
    
    adapter = CustomAdapter(config)
    adapter.initialize()
    
    return adapter


def create_custom_tokenizer_from_object(tokenizer_obj: Any,
                                       special_tokens: Optional[Dict[str, int]] = None,
                                       max_length: int = 512,
                                       model_name: str = "custom") -> CustomAdapter:
    """
    Convenience function to create a custom tokenizer adapter from an object.
    
    Args:
        tokenizer_obj: Custom tokenizer object implementing the required interface
        special_tokens: Special token IDs (optional)
        max_length: Maximum sequence length
        model_name: Name for the custom tokenizer
        
    Returns:
        Initialized CustomAdapter instance
        
    Example:
        >>> class MyTokenizer:
        ...     def encode(self, text):
        ...         return [ord(c) for c in text]
        ...     def decode(self, token_ids):
        ...         return ''.join(chr(tid) for tid in token_ids)
        ...     def get_vocab_size(self):
        ...         return 256
        >>> 
        >>> my_tokenizer = MyTokenizer()
        >>> adapter = create_custom_tokenizer_from_object(my_tokenizer)
    """
    config = TokenizerConfig(
        backend='custom',
        model_name=model_name,
        max_length=max_length,
        special_tokens=special_tokens,
        backend_specific_config={
            'tokenizer': tokenizer_obj
        }
    )
    
    adapter = CustomAdapter(config)
    adapter.initialize()
    
    return adapter


# Register the custom adapter
def register_custom_adapter():
    """Register the custom adapter with the tokenizer registry."""
    TokenizerRegistry.register_adapter(
        'custom', 
        CustomAdapter, 
        ['custom']
    )
    
    logger.info("Registered custom tokenizer adapter")


# Auto-register when module is imported
register_custom_adapter()