#!/usr/bin/env python3
"""
Enhanced tokenization system for LSM with flexible backend support.

This module provides an enhanced tokenizer wrapper that can adapt any tokenizer
backend (HuggingFace, OpenAI, spaCy, custom) with sinusoidal embeddings and
streaming data processing capabilities.

Classes:
    TokenizerConfig: Configuration dataclass for tokenizer adapters
    TokenizerAdapter: Abstract base class for tokenizer backend adapters
    TokenizerRegistry: Registry system for automatic backend detection and loading
    EnhancedTokenizerWrapper: Main wrapper class for enhanced tokenization

Key Features:
    - Support for multiple tokenizer backends (HuggingFace, OpenAI, spaCy, custom)
    - Automatic vocabulary size detection and adaptation
    - Sinusoidal embedding integration with configurable parameters
    - Streaming data processing for large datasets
    - Intelligent caching system for performance optimization
    - Memory-efficient storage and GPU acceleration support

Example:
    Basic usage with automatic backend detection:
    
    >>> from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
    >>> tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=256)
    >>> tokens = tokenizer.tokenize(['Hello world', 'How are you?'])
    >>> embedder = tokenizer.create_configurable_sinusoidal_embedder()
    
    Advanced usage with streaming data:
    
    >>> tokenizer = EnhancedTokenizerWrapper(
    ...     'bert-base-uncased',
    ...     embedding_dim=512,
    ...     enable_caching=True
    ... )
    >>> embedder = tokenizer.fit_streaming(
    ...     'large_dataset.txt',
    ...     batch_size=1000,
    ...     auto_adjust_batch_size=True
    ... )

See Also:
    - ConfigurableSinusoidalEmbedder: For advanced sinusoidal embedding configuration
    - StreamingDataIterator: For memory-efficient data processing
    - IntelligentCachingSystem: For performance optimization
"""

import os
import json
import pickle
import numpy as np
import hashlib
import random
from typing import List, Dict, Any, Optional, Union, Tuple, Type, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import logging

from ..utils.lsm_exceptions import (
    TokenizerError, TokenizerNotFittedError, 
    TokenizerLoadError, TokenizerSaveError, InvalidInputError
)
from ..utils.lsm_logging import get_logger
from .intelligent_caching import CacheConfig, IntelligentCachingSystem

logger = get_logger(__name__)


@dataclass
class TokenizerConfig:
    """Configuration for tokenizer adapters."""
    backend: str
    model_name: str
    max_length: int = 512
    special_tokens: Optional[Dict[str, str]] = None
    backend_specific_config: Optional[Dict[str, Any]] = None


class TokenizerAdapter(ABC):
    """
    Abstract base class for tokenizer adapters.
    
    This class defines the standardized interface that all tokenizer backends
    must implement to work with the enhanced tokenizer system. It provides
    a unified API for tokenization, vocabulary access, and special token handling
    across different tokenizer implementations.
    
    Attributes:
        config (TokenizerConfig): Configuration object for the adapter
        _tokenizer: The underlying tokenizer instance (backend-specific)
        _vocab_size (int): Size of the tokenizer's vocabulary
        _is_initialized (bool): Whether the adapter has been initialized
    
    Abstract Methods:
        initialize(): Initialize the underlying tokenizer
        tokenize(): Convert texts to token ID sequences
        decode(): Convert token IDs back to text
        get_vocab_size(): Get the vocabulary size
        get_vocab(): Get the vocabulary mapping
        get_special_tokens(): Get special token IDs
        load_adapter_config(): Load adapter from saved configuration
    
    Example:
        Creating a custom tokenizer adapter:
        
        >>> class MyTokenizerAdapter(TokenizerAdapter):
        ...     def initialize(self):
        ...         # Initialize your tokenizer here
        ...         pass
        ...     
        ...     def tokenize(self, texts, **kwargs):
        ...         # Implement tokenization logic
        ...         return token_sequences
        ...     
        ...     # Implement other abstract methods...
        
        >>> config = TokenizerConfig(backend='custom', model_name='my_model')
        >>> adapter = MyTokenizerAdapter(config)
        >>> adapter.initialize()
    
    Note:
        All concrete implementations must call super().__init__(config) and
        set self._is_initialized = True after successful initialization.
    """
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialize tokenizer adapter.
        
        Args:
            config: Tokenizer configuration
        """
        self.config = config
        self._tokenizer = None
        self._vocab_size = None
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the underlying tokenizer."""
        pass
    
    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: bool = True, 
                 truncation: bool = True) -> List[List[int]]:
        """
        Tokenize texts to token IDs.
        
        This method converts input text(s) into sequences of token IDs that can
        be used for embedding lookup and model processing. The method handles
        both single strings and lists of strings uniformly.
        
        Args:
            texts (Union[str, List[str]]): Single text string or list of text strings
                to tokenize. Empty strings are handled gracefully.
            add_special_tokens (bool, optional): Whether to add model-specific special
                tokens (e.g., [CLS], [SEP] for BERT, <|endoftext|> for GPT).
                Defaults to True.
            padding (bool, optional): Whether to pad sequences to the same length
                within the batch. Padding token is determined by the tokenizer.
                Defaults to True.
            truncation (bool, optional): Whether to truncate sequences that exceed
                the maximum length specified in the tokenizer configuration.
                Defaults to True.
        
        Returns:
            List[List[int]]: List of token ID sequences, where each inner list
                represents the token IDs for one input text. All sequences will
                have the same length if padding=True.
        
        Raises:
            TokenizerError: If tokenization fails or tokenizer is not initialized
            InvalidInputError: If input texts are in an invalid format
        
        Example:
            >>> adapter = SomeTokenizerAdapter(config)
            >>> adapter.initialize()
            >>> tokens = adapter.tokenize(['Hello world', 'How are you?'])
            >>> print(tokens)  # [[101, 7592, 2088, 102], [101, 2129, 2024, 2017, 1029, 102]]
        
        Note:
            The exact token IDs returned depend on the specific tokenizer backend
            and model. Special tokens and their positions may vary between models.
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        pass
    
    @abstractmethod
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        pass
    
    def encode_single(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        return self.tokenize([text], add_special_tokens=add_special_tokens, 
                           padding=False, truncation=True)[0]
    
    def decode_single(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a single sequence of token IDs.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_token_embeddings_shape(self, embedding_dim: int) -> Tuple[int, int]:
        """
        Get the shape for token embeddings matrix.
        
        Args:
            embedding_dim: Desired embedding dimension
            
        Returns:
            Tuple of (vocab_size, embedding_dim)
        """
        return (self.get_vocab_size(), embedding_dim)
    
    def save_adapter_config(self, save_path: str) -> None:
        """
        Save adapter-specific configuration.
        
        Args:
            save_path: Directory path to save config
        """
        config_dict = {
            'backend': self.config.backend,
            'model_name': self.config.model_name,
            'max_length': self.config.max_length,
            'special_tokens': self.config.special_tokens,
            'backend_specific_config': self.config.backend_specific_config,
            'vocab_size': self.get_vocab_size() if self._is_initialized else None
        }
        
        config_path = os.path.join(save_path, f'{self.config.backend}_adapter_config.json')
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    @abstractmethod
    def load_adapter_config(cls, load_path: str) -> 'TokenizerAdapter':
        """
        Load adapter from saved configuration.
        
        Args:
            load_path: Directory path to load config from
            
        Returns:
            Loaded adapter instance
        """
        pass
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(backend={self.config.backend}, "
                f"model={self.config.model_name}, vocab_size={self._vocab_size})")


class TokenizerRegistry:
    """
    Registry system for automatic tokenizer backend detection and loading.
    
    This class manages the registration and instantiation of different
    tokenizer adapters based on backend names or model names.
    """
    
    _adapters: Dict[str, Type[TokenizerAdapter]] = {}
    _model_mappings: Dict[str, str] = {}
    
    @classmethod
    def register_adapter(cls, backend_name: str, adapter_class: Type[TokenizerAdapter],
                        model_patterns: Optional[List[str]] = None) -> None:
        """
        Register a tokenizer adapter.
        
        Args:
            backend_name: Name of the backend (e.g., 'huggingface', 'openai')
            adapter_class: Adapter class to register
            model_patterns: List of model name patterns this adapter handles
        """
        cls._adapters[backend_name] = adapter_class
        
        if model_patterns:
            for pattern in model_patterns:
                cls._model_mappings[pattern] = backend_name
        
        logger.info(f"Registered tokenizer adapter: {backend_name} -> {adapter_class.__name__}")
    
    @classmethod
    def get_adapter_class(cls, backend_or_model: str) -> Type[TokenizerAdapter]:
        """
        Get adapter class for a backend or model name.
        
        Args:
            backend_or_model: Backend name or model name
            
        Returns:
            Adapter class
            
        Raises:
            TokenizerError: If no adapter found
        """
        # First try direct backend lookup
        if backend_or_model in cls._adapters:
            return cls._adapters[backend_or_model]
        
        # Then try model pattern matching
        for pattern, backend in cls._model_mappings.items():
            if pattern in backend_or_model or backend_or_model.startswith(pattern):
                return cls._adapters[backend]
        
        # If no exact match, try partial matching
        for pattern, backend in cls._model_mappings.items():
            if pattern.lower() in backend_or_model.lower():
                return cls._adapters[backend]
        
        raise TokenizerError(
            f"No adapter found for backend/model: {backend_or_model}. "
            f"Available backends: {list(cls._adapters.keys())}"
        )
    
    @classmethod
    def create_adapter(cls, backend_or_model: str, max_length: int = 512,
                      special_tokens: Optional[Dict[str, str]] = None,
                      backend_specific_config: Optional[Dict[str, Any]] = None) -> TokenizerAdapter:
        """
        Create a tokenizer adapter instance.
        
        Args:
            backend_or_model: Backend name or model name
            max_length: Maximum sequence length
            special_tokens: Special token configuration
            backend_specific_config: Backend-specific configuration
            
        Returns:
            Initialized tokenizer adapter
        """
        adapter_class = cls.get_adapter_class(backend_or_model)
        
        # Determine backend and model name
        if backend_or_model in cls._adapters:
            backend = backend_or_model
            model_name = backend_specific_config.get('model_name', 'default') if backend_specific_config else 'default'
        else:
            # Find backend from model patterns
            backend = None
            for pattern, backend_name in cls._model_mappings.items():
                if pattern in backend_or_model or backend_or_model.startswith(pattern):
                    backend = backend_name
                    break
            
            if backend is None:
                # Try partial matching
                for pattern, backend_name in cls._model_mappings.items():
                    if pattern.lower() in backend_or_model.lower():
                        backend = backend_name
                        break
            
            if backend is None:
                raise TokenizerError(f"Could not determine backend for model: {backend_or_model}")
            
            model_name = backend_or_model
        
        config = TokenizerConfig(
            backend=backend,
            model_name=model_name,
            max_length=max_length,
            special_tokens=special_tokens,
            backend_specific_config=backend_specific_config
        )
        
        adapter = adapter_class(config)
        adapter.initialize()
        
        return adapter
    
    @classmethod
    def list_available_backends(cls) -> List[str]:
        """Get list of available backends."""
        return list(cls._adapters.keys())
    
    @classmethod
    def list_supported_models(cls) -> Dict[str, List[str]]:
        """Get mapping of backends to supported model patterns."""
        backend_models = {}
        for model, backend in cls._model_mappings.items():
            if backend not in backend_models:
                backend_models[backend] = []
            backend_models[backend].append(model)
        return backend_models


class EnhancedTokenizerWrapper:
    """
    Enhanced tokenizer wrapper that can adapt any tokenizer backend.
    
    This class provides a unified interface for different tokenizer backends
    while maintaining compatibility with the existing LSM tokenization system.
    It supports automatic backend detection, sinusoidal embedding adaptation,
    and streaming data processing for large datasets.
    
    The wrapper automatically detects the appropriate tokenizer backend based on
    the model name or explicit backend specification, handles vocabulary size
    adaptation, and provides seamless integration with sinusoidal embeddings.
    
    Attributes:
        embedding_dim (int): Dimension for sinusoidal embeddings
        max_length (int): Maximum sequence length for tokenization
        special_tokens (Dict[str, str]): Special token configuration
        backend_specific_config (Dict[str, Any]): Backend-specific configuration
        enable_caching (bool): Whether intelligent caching is enabled
        _adapter (TokenizerAdapter): The underlying tokenizer adapter
        _caching_system (IntelligentCachingSystem): Caching system instance
        _sinusoidal_embedder: The fitted sinusoidal embedder (if any)
        _is_fitted (bool): Whether the tokenizer has been fitted
    
    Example:
        Basic usage with automatic backend detection:
        
        >>> tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=256)
        >>> tokens = tokenizer.tokenize(['Hello world'])
        >>> embedder = tokenizer.create_configurable_sinusoidal_embedder()
        
        Advanced usage with custom configuration:
        
        >>> tokenizer = EnhancedTokenizerWrapper(
        ...     tokenizer='bert-base-uncased',
        ...     embedding_dim=512,
        ...     max_length=256,
        ...     special_tokens={'pad_token': '[PAD]'},
        ...     backend_specific_config={'do_lower_case': True},
        ...     enable_caching=True
        ... )
        
        Streaming data processing:
        
        >>> embedder = tokenizer.fit_streaming(
        ...     data_source='large_dataset.txt',
        ...     batch_size=1000,
        ...     auto_adjust_batch_size=True,
        ...     memory_threshold_mb=500.0
        ... )
    
    See Also:
        - TokenizerAdapter: Base class for tokenizer backends
        - ConfigurableSinusoidalEmbedder: For advanced embedding configuration
        - StreamingDataIterator: For memory-efficient data processing
    """
    
    def __init__(self, tokenizer: Union[str, TokenizerAdapter], 
                 embedding_dim: int = 128,
                 max_length: int = 512,
                 special_tokens: Optional[Dict[str, str]] = None,
                 backend_specific_config: Optional[Dict[str, Any]] = None,
                 enable_caching: bool = True,
                 cache_config: Optional[CacheConfig] = None):
        """
        Initialize EnhancedTokenizerWrapper.
        
        Args:
            tokenizer: Tokenizer backend name, model name, or adapter instance
            embedding_dim: Dimension for sinusoidal embeddings
            max_length: Maximum sequence length
            special_tokens: Special token configuration
            backend_specific_config: Backend-specific configuration
            enable_caching: Whether to enable intelligent caching
            cache_config: Configuration for intelligent caching system
        """
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.special_tokens = special_tokens or {}
        self.backend_specific_config = backend_specific_config or {}
        self.enable_caching = enable_caching
        
        # Initialize tokenizer adapter
        if isinstance(tokenizer, str):
            self._adapter = TokenizerRegistry.create_adapter(
                tokenizer, max_length, special_tokens, backend_specific_config
            )
        elif isinstance(tokenizer, TokenizerAdapter):
            self._adapter = tokenizer
            if not self._adapter._is_initialized:
                self._adapter.initialize()
        else:
            raise TokenizerError(f"Invalid tokenizer type: {type(tokenizer)}")
        
        # Initialize intelligent caching system
        if self.enable_caching:
            if cache_config is None:
                # Create default cache configuration optimized for tokenizer usage
                cache_config = CacheConfig(
                    max_cache_size=min(10000, max(1000, self._adapter.get_vocab_size() // 10)),
                    enable_batch_caching=True,
                    batch_cache_size=min(5000, max(500, self._adapter.get_vocab_size() // 20)),
                    enable_cache_warming=True,
                    warmup_strategy="frequency",
                    warmup_size=min(1000, max(100, self._adapter.get_vocab_size() // 50)),
                    enable_metrics=True
                )
            self._caching_system = IntelligentCachingSystem(cache_config)
            logger.info(f"Enabled intelligent caching with config: {cache_config}")
        else:
            self._caching_system = None
        
        # Initialize sinusoidal embedder (will be created when needed)
        self._sinusoidal_embedder = None
        self._is_fitted = False
        
        logger.info(f"Initialized EnhancedTokenizerWrapper with {self._adapter}")
    
    def get_adapter(self) -> TokenizerAdapter:
        """Get the underlying tokenizer adapter."""
        return self._adapter
    
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
        return self._adapter.tokenize(texts, add_special_tokens, padding, truncation)
    
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
        return self._adapter.decode(token_ids, skip_special_tokens)
    
    def encode_single(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        return self._adapter.encode_single(text, add_special_tokens)
    
    def decode_single(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a single sequence of token IDs.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self._adapter.decode_single(token_ids, skip_special_tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._adapter.get_vocab_size()
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self._adapter.get_vocab()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return self._adapter.get_special_tokens()
    
    def get_token_embeddings_shape(self, embedding_dim: Optional[int] = None) -> Tuple[int, int]:
        """
        Get the shape for token embeddings matrix.
        
        Args:
            embedding_dim: Desired embedding dimension (uses instance default if None)
            
        Returns:
            Tuple of (vocab_size, embedding_dim)
        """
        dim = embedding_dim or self.embedding_dim
        return self._adapter.get_token_embeddings_shape(dim)
    
    def create_sinusoidal_embedder(self, max_position: int = 10000, 
                                  temperature: float = 1.0) -> 'SinusoidalEmbedder':
        """
        Create a sinusoidal embedder adapted to this tokenizer.
        
        Args:
            max_position: Maximum position for positional encoding
            temperature: Temperature parameter for sinusoidal patterns
            
        Returns:
            SinusoidalEmbedder instance
        """
        from .tokenization import SinusoidalEmbedder
        
        vocab_size = self.get_vocab_size()
        embedder = SinusoidalEmbedder(
            vocab_size=vocab_size,
            embedding_dim=self.embedding_dim,
            max_position=max_position,
            temperature=temperature
        )
        
        return embedder
    
    def create_configurable_sinusoidal_embedder(self, 
                                               learnable_frequencies: bool = True,
                                               use_relative_position: bool = False,
                                               base_frequency: float = 10000.0,
                                               frequency_scaling: float = 1.0,
                                               **kwargs) -> 'ConfigurableSinusoidalEmbedder':
        """
        Create a configurable sinusoidal embedder automatically adapted to this tokenizer.
        
        Args:
            learnable_frequencies: Whether to use learnable frequency parameters
            use_relative_position: Whether to use relative positional encoding
            base_frequency: Base frequency for sinusoidal patterns
            frequency_scaling: Scaling factor for frequencies
            **kwargs: Additional configuration parameters
            
        Returns:
            ConfigurableSinusoidalEmbedder instance adapted to this tokenizer
        """
        from .configurable_sinusoidal_embedder import ConfigurableSinusoidalEmbedder, SinusoidalConfig
        
        # Get vocabulary size from tokenizer
        vocab_size = self.get_vocab_size()
        
        # Create configuration with automatic adaptation
        config = SinusoidalConfig(
            embedding_dim=self.embedding_dim,
            vocab_size=vocab_size,
            max_sequence_length=self.max_length,
            base_frequency=base_frequency,
            frequency_scaling=frequency_scaling,
            learnable_frequencies=learnable_frequencies,
            use_absolute_position=True,
            use_relative_position=use_relative_position,
            **kwargs
        )
        
        # Create embedder
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Automatically adapt to this tokenizer
        embedder.adapt_to_tokenizer(self._adapter)
        
        logger.info(f"Created ConfigurableSinusoidalEmbedder adapted to {self._adapter}")
        
        return embedder
    
    def auto_adapt_embedding_dimension(self, target_dim: int, 
                                     preserve_properties: bool = True) -> 'ConfigurableSinusoidalEmbedder':
        """
        Automatically create and adapt a sinusoidal embedder with optimal dimension matching.
        
        Args:
            target_dim: Target embedding dimension
            preserve_properties: Whether to preserve mathematical properties
            
        Returns:
            ConfigurableSinusoidalEmbedder with adapted dimensions
        """
        from .configurable_sinusoidal_embedder import ConfigurableSinusoidalEmbedder, SinusoidalConfig
        
        if target_dim <= 0:
            raise InvalidInputError("target_dim", "positive integer", str(target_dim))
        
        if target_dim % 2 != 0:
            raise InvalidInputError("target_dim", "even integer", str(target_dim))
        
        # Get vocabulary size from tokenizer
        vocab_size = self.get_vocab_size()
        
        # Create configuration with target dimension
        config = SinusoidalConfig(
            embedding_dim=target_dim,
            vocab_size=vocab_size,
            max_sequence_length=self.max_length,
            learnable_frequencies=True,
            use_absolute_position=True,
            use_relative_position=False
        )
        
        # Create embedder
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Adapt to tokenizer
        embedder.adapt_to_tokenizer(self._adapter)
        
        # If we had a different original dimension, adapt with property preservation
        if self.embedding_dim != target_dim and preserve_properties:
            embedder.adapt_embedding_dimension(target_dim, preserve_properties)
        
        logger.info(f"Auto-adapted embedding dimension to {target_dim} for {self._adapter}")
        
        return embedder
    
    def create_optimized_embedder(self, target_model_size: str = 'medium',
                                 learnable_frequencies: bool = True,
                                 preserve_properties: bool = True) -> 'ConfigurableSinusoidalEmbedder':
        """
        Create a sinusoidal embedder with automatically optimized dimensions.
        
        Args:
            target_model_size: Target model size ('small', 'medium', 'large', 'xlarge')
            learnable_frequencies: Whether to use learnable frequencies
            preserve_properties: Whether to preserve mathematical properties
            
        Returns:
            ConfigurableSinusoidalEmbedder with optimized dimensions
        """
        from .configurable_sinusoidal_embedder import SinusoidalEmbedderFactory
        
        # Create auto-adapted embedder using factory
        embedder = SinusoidalEmbedderFactory.create_auto_adapted(
            self._adapter, target_model_size, learnable_frequencies, preserve_properties
        )
        
        logger.info(f"Created optimized embedder for {self._adapter} with model size {target_model_size}")
        
        return embedder
    
    def get_embedding_dimension_suggestions(self) -> Dict[str, int]:
        """
        Get embedding dimension suggestions for different model sizes.
        
        This method provides recommended embedding dimensions for different model
        sizes based on the tokenizer's vocabulary size and common architectural
        patterns. The suggestions balance model capacity with computational efficiency.
        
        Returns:
            Dict[str, int]: Dictionary mapping model size names to suggested dimensions.
                Keys include 'small', 'medium', 'large', 'xlarge' with corresponding
                optimal embedding dimensions.
        
        Example:
            >>> tokenizer = EnhancedTokenizerWrapper('gpt2')
            >>> suggestions = tokenizer.get_embedding_dimension_suggestions()
            >>> print(suggestions)
            {'small': 128, 'medium': 256, 'large': 512, 'xlarge': 1024}
            >>> 
            >>> # Use suggestion for medium model
            >>> embedder = tokenizer.create_configurable_sinusoidal_embedder(
            ...     embedding_dim=suggestions['medium']
            ... )
        
        Note:
            The suggestions are based on the tokenizer's vocabulary size and
            established best practices for embedding dimensions. Larger vocabularies
            typically benefit from higher-dimensional embeddings.
        """
        from .configurable_sinusoidal_embedder import EmbeddingDimensionOptimizer
        
        vocab_size = self.get_vocab_size()
        suggestions = {}
        
        for size in ['small', 'medium', 'large', 'xlarge']:
            suggestions[size] = EmbeddingDimensionOptimizer.calculate_optimal_dimension(
                vocab_size, size, preserve_mathematical_properties=True
            )
        
        return suggestions
    
    def fit_sinusoidal_embedder(self, training_data: Union[List[str], np.ndarray],
                               epochs: int = 100,
                               max_position: int = 10000,
                               temperature: float = 1.0) -> 'SinusoidalEmbedder':
        """
        Fit a sinusoidal embedder on training data.
        
        Args:
            training_data: Training texts or token sequences
            epochs: Number of training epochs
            max_position: Maximum position for positional encoding
            temperature: Temperature parameter for sinusoidal patterns
            
        Returns:
            Fitted SinusoidalEmbedder instance
        """
        # Convert text data to token sequences if needed
        if isinstance(training_data, list) and isinstance(training_data[0], str):
            token_sequences = self.tokenize(training_data, padding=True, truncation=True)
            token_sequences = np.array(token_sequences)
        elif isinstance(training_data, np.ndarray):
            token_sequences = training_data
        else:
            raise InvalidInputError("training_data", "list of strings or numpy array", str(type(training_data)))
        
        # Create and fit embedder
        self._sinusoidal_embedder = self.create_sinusoidal_embedder(max_position, temperature)
        self._sinusoidal_embedder.fit(token_sequences, epochs)
        self._is_fitted = True
        
        logger.info(f"Fitted sinusoidal embedder on {len(token_sequences)} sequences")
        return self._sinusoidal_embedder
    
    def fit_streaming(self, data_source: Union[str, List[str], 'StreamingDataIterator'],
                     batch_size: int = 1000,
                     epochs: int = 100,
                     max_position: int = 10000,
                     temperature: float = 1.0,
                     memory_threshold_mb: float = 1000.0,
                     progress_callback: Optional[callable] = None,
                     auto_adjust_batch_size: bool = True,
                     min_batch_size: int = 100,
                     max_batch_size: int = 10000) -> 'SinusoidalEmbedder':
        """
        Fit a sinusoidal embedder on streaming data for memory-efficient training.
        
        This method processes large datasets that don't fit in memory by using
        streaming data processing with configurable batch sizes, progress tracking,
        and memory usage monitoring.
        
        Args:
            data_source: Data source (file path, directory, file list, or StreamingDataIterator)
            batch_size: Initial batch size for processing
            epochs: Number of training epochs
            max_position: Maximum position for positional encoding
            temperature: Temperature parameter for sinusoidal patterns
            memory_threshold_mb: Memory threshold in MB for automatic batch size adjustment
            progress_callback: Optional callback function for progress updates
            auto_adjust_batch_size: Whether to automatically adjust batch size based on memory
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            
        Returns:
            Fitted SinusoidalEmbedder instance
            
        Raises:
            InvalidInputError: If invalid parameters provided
            TokenizerError: If fitting fails
        """
        import psutil
        import time
        from .streaming_data_iterator import StreamingDataIterator
        
        # Validate parameters
        if batch_size <= 0:
            raise InvalidInputError("batch_size", "positive integer", str(batch_size))
        if epochs <= 0:
            raise InvalidInputError("epochs", "positive integer", str(epochs))
        if memory_threshold_mb <= 0:
            raise InvalidInputError("memory_threshold_mb", "positive float", str(memory_threshold_mb))
        if min_batch_size <= 0 or min_batch_size > max_batch_size:
            raise InvalidInputError("min_batch_size", f"positive integer <= {max_batch_size}", str(min_batch_size))
        
        logger.info(f"Starting streaming tokenizer fitting with batch_size={batch_size}, epochs={epochs}")
        
        # Create streaming iterator if not provided
        if isinstance(data_source, StreamingDataIterator):
            streaming_iterator = data_source
        else:
            # Don't pass progress_callback to StreamingDataIterator as it has different signature
            streaming_iterator = StreamingDataIterator(
                data_source=data_source,
                batch_size=batch_size,
                memory_threshold_mb=memory_threshold_mb,
                auto_adjust_batch_size=auto_adjust_batch_size,
                progress_callback=None  # We'll handle progress in our own loop
            )
        
        # Initialize statistics collection
        total_sequences = 0
        total_tokens = 0
        vocab_stats = {}
        sequence_lengths = []
        current_batch_size = batch_size
        
        # Memory monitoring setup
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create sinusoidal embedder
        self._sinusoidal_embedder = self.create_sinusoidal_embedder(max_position, temperature)
        
        # Progress tracking variables
        start_time = time.time()
        batches_processed = 0
        
        try:
            # Multi-epoch training with streaming
            for epoch in range(epochs):
                logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                epoch_start_time = time.time()
                epoch_sequences = 0
                epoch_tokens = 0
                
                # Reset iterator for new epoch
                streaming_iterator.reset()
                
                # Process batches in current epoch
                for batch_idx, batch_data in enumerate(streaming_iterator):
                    batch_start_time = time.time()
                    
                    # Monitor memory usage
                    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_usage = current_memory - initial_memory
                    
                    # Auto-adjust batch size if memory threshold exceeded
                    if auto_adjust_batch_size and memory_usage > memory_threshold_mb:
                        new_batch_size = max(min_batch_size, int(current_batch_size * 0.8))
                        if new_batch_size != current_batch_size:
                            current_batch_size = new_batch_size
                            streaming_iterator.batch_size = current_batch_size
                            logger.warning(f"Memory threshold exceeded ({memory_usage:.1f}MB), "
                                         f"reducing batch size to {current_batch_size}")
                    
                    # Process batch data
                    if len(batch_data) == 0:
                        continue  # Skip empty batches
                    
                    if isinstance(batch_data[0], str):
                        # Text data - tokenize
                        token_sequences = self.tokenize(batch_data, padding=True, truncation=True)
                    elif isinstance(batch_data[0], dict):
                        # Dictionary data - extract text field and tokenize
                        text_data = []
                        for item in batch_data:
                            if isinstance(item, dict):
                                # Try to extract text from common fields
                                text = item.get('text', item.get('content', item.get('message', str(item))))
                            else:
                                text = str(item)
                            text_data.append(text)
                        token_sequences = self.tokenize(text_data, padding=True, truncation=True)
                    else:
                        # Already tokenized data
                        token_sequences = batch_data
                    
                    # Convert to numpy array for processing
                    token_sequences = np.array(token_sequences)
                    
                    # Update statistics
                    batch_sequences = len(token_sequences)
                    batch_tokens = np.sum(token_sequences != 0)  # Count non-padding tokens
                    
                    total_sequences += batch_sequences
                    total_tokens += batch_tokens
                    epoch_sequences += batch_sequences
                    epoch_tokens += batch_tokens
                    
                    # Collect vocabulary statistics
                    unique_tokens, counts = np.unique(token_sequences, return_counts=True)
                    for token, count in zip(unique_tokens, counts):
                        if token != 0:  # Skip padding tokens
                            vocab_stats[int(token)] = vocab_stats.get(int(token), 0) + int(count)
                    
                    # Collect sequence length statistics
                    for seq in token_sequences:
                        seq_len = np.sum(seq != 0)  # Count non-padding tokens
                        sequence_lengths.append(seq_len)
                    
                    # Fit embedder on current batch
                    if epoch == 0:  # Only fit on first epoch to build embeddings
                        self._sinusoidal_embedder.fit_batch(token_sequences)
                    else:  # Update embeddings on subsequent epochs
                        self._sinusoidal_embedder.update_batch(token_sequences)
                    
                    batches_processed += 1
                    batch_time = time.time() - batch_start_time
                    
                    # Progress callback
                    if progress_callback:
                        progress_info = {
                            'epoch': epoch + 1,
                            'total_epochs': epochs,
                            'batch': batch_idx + 1,
                            'sequences_processed': total_sequences,
                            'tokens_processed': total_tokens,
                            'current_batch_size': current_batch_size,
                            'memory_usage_mb': memory_usage,
                            'batch_time_seconds': batch_time,
                            'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0
                        }
                        progress_callback(progress_info)
                    
                    # Log progress periodically
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                                  f"processed {batch_sequences} sequences, "
                                  f"memory usage: {memory_usage:.1f}MB, "
                                  f"batch time: {batch_time:.2f}s")
                
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Completed epoch {epoch + 1}: "
                          f"processed {epoch_sequences} sequences, "
                          f"epoch time: {epoch_time:.2f}s")
            
            # Finalize embedder training
            if total_sequences > 0:
                self._sinusoidal_embedder.finalize_training()
                self._is_fitted = True
            else:
                # Handle empty data case - create minimal embedder
                logger.warning("No sequences processed - creating minimal embedder")
                self._sinusoidal_embedder._embedding_matrix = self._sinusoidal_embedder._initialize_embeddings()
                self._sinusoidal_embedder._positional_encodings = self._sinusoidal_embedder._create_positional_encodings()
                self._sinusoidal_embedder._is_fitted = True
                self._is_fitted = True
            
            # Calculate final statistics
            total_time = time.time() - start_time
            avg_sequence_length = np.mean(sequence_lengths) if sequence_lengths else 0
            vocab_coverage = len(vocab_stats) / self.get_vocab_size() * 100 if total_sequences > 0 else 0
            
            # Log final statistics
            logger.info(f"Streaming tokenizer fitting completed:")
            logger.info(f"  Total sequences processed: {total_sequences:,}")
            logger.info(f"  Total tokens processed: {total_tokens:,}")
            logger.info(f"  Average sequence length: {avg_sequence_length:.1f}")
            logger.info(f"  Vocabulary coverage: {vocab_coverage:.1f}%")
            logger.info(f"  Total training time: {total_time:.2f}s")
            logger.info(f"  Batches processed: {batches_processed}")
            logger.info(f"  Final batch size: {current_batch_size}")
            
            # Store training statistics
            self._training_stats = {
                'total_sequences': total_sequences,
                'total_tokens': total_tokens,
                'avg_sequence_length': avg_sequence_length,
                'vocab_stats': vocab_stats,
                'vocab_coverage': vocab_coverage,
                'training_time': total_time,
                'batches_processed': batches_processed,
                'final_batch_size': current_batch_size,
                'epochs': epochs,
                'memory_threshold_mb': memory_threshold_mb
            }
            
            return self._sinusoidal_embedder
            
        except Exception as e:
            logger.error(f"Error during streaming tokenizer fitting: {str(e)}")
            raise TokenizerError(f"Streaming fitting failed: {str(e)}")
    
    def get_training_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get training statistics from the last streaming fit operation.
        
        Returns:
            Dictionary containing training statistics, or None if not fitted
        """
        return getattr(self, '_training_stats', None)
    
    def get_sinusoidal_embedder(self) -> Optional['SinusoidalEmbedder']:
        """Get the fitted sinusoidal embedder."""
        return self._sinusoidal_embedder
    
    def embed(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Convert token IDs to sinusoidal embeddings.
        
        Args:
            token_ids: Token IDs to embed
            
        Returns:
            Embeddings array
            
        Raises:
            TokenizerNotFittedError: If sinusoidal embedder not fitted
        """
        if self._sinusoidal_embedder is None:
            raise TokenizerNotFittedError("embed - call fit_sinusoidal_embedder first")
        
        return self._sinusoidal_embedder.embed(token_ids)
    
    def _create_deterministic_seed(self, data_source: Union[str, List[str]], 
                                  batch_size: int, epochs: int) -> int:
        """
        Create a deterministic seed based on data source and parameters.
        
        This ensures consistent results across streaming batches by using
        a hash of the data source and training parameters.
        
        Args:
            data_source: Data source specification
            batch_size: Batch size for processing
            epochs: Number of training epochs
            
        Returns:
            Deterministic seed value
        """
        # Create a string representation of the configuration
        if isinstance(data_source, list):
            source_str = '|'.join(sorted(str(s) for s in data_source))
        else:
            source_str = str(data_source)
        
        config_str = f"{source_str}|{batch_size}|{epochs}|{self.embedding_dim}|{self.max_length}"
        
        # Create hash and convert to seed
        hash_obj = hashlib.md5(config_str.encode('utf-8'))
        seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)
        
        return seed
    
    def _save_checkpoint(self, checkpoint_path: str, epoch: int, batch_idx: int,
                        total_sequences: int, total_tokens: int, 
                        vocab_stats: Dict[int, int], sequence_lengths: List[int],
                        embedder_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Save training checkpoint for resumable streaming.
        
        Args:
            checkpoint_path: Path to save checkpoint
            epoch: Current epoch number
            batch_idx: Current batch index within epoch
            total_sequences: Total sequences processed so far
            total_tokens: Total tokens processed so far
            vocab_stats: Vocabulary statistics
            sequence_lengths: List of sequence lengths
            embedder_state: Optional embedder state to save
        """
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            
            checkpoint_data = {
                'epoch': epoch,
                'batch_idx': batch_idx,
                'total_sequences': total_sequences,
                'total_tokens': total_tokens,
                'vocab_stats': vocab_stats,
                'sequence_lengths': sequence_lengths,
                'embedder_state': embedder_state,
                'tokenizer_config': {
                    'embedding_dim': self.embedding_dim,
                    'max_length': self.max_length,
                    'vocab_size': self.get_vocab_size()
                }
            }
            
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            logger.debug(f"Checkpoint saved: epoch {epoch}, batch {batch_idx}")
            
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {str(e)}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Load training checkpoint for resumable streaming.
        
        Args:
            checkpoint_path: Path to load checkpoint from
            
        Returns:
            Checkpoint data dictionary, or None if loading fails
        """
        try:
            if not os.path.exists(checkpoint_path):
                return None
            
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Validate checkpoint compatibility
            tokenizer_config = checkpoint_data.get('tokenizer_config', {})
            if (tokenizer_config.get('embedding_dim') != self.embedding_dim or
                tokenizer_config.get('max_length') != self.max_length or
                tokenizer_config.get('vocab_size') != self.get_vocab_size()):
                logger.warning("Checkpoint incompatible with current tokenizer configuration")
                return None
            
            logger.info(f"Checkpoint loaded: epoch {checkpoint_data['epoch']}, "
                       f"batch {checkpoint_data['batch_idx']}")
            return checkpoint_data
            
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {str(e)}")
            return None
    
    def _validate_streaming_consistency(self, streaming_embedder: 'SinusoidalEmbedder',
                                      validation_data: List[str], 
                                      batch_size: int = 100) -> Dict[str, float]:
        """
        Validate that streaming results match batch processing results.
        
        This method compares embeddings generated by streaming training
        with those from equivalent batch processing to ensure consistency.
        
        Args:
            streaming_embedder: Embedder trained via streaming
            validation_data: Sample data for validation
            batch_size: Batch size for validation processing
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("Validating streaming consistency against batch processing")
        
        try:
            # Create a batch-trained embedder for comparison
            batch_embedder = self.create_sinusoidal_embedder()
            
            # Tokenize validation data
            validation_tokens = self.tokenize(validation_data, padding=True, truncation=True)
            validation_tokens = np.array(validation_tokens)
            
            # Train batch embedder on same data
            batch_embedder.fit(validation_tokens, epochs=10)  # Reduced epochs for validation
            
            # Generate embeddings from both methods
            streaming_embeddings = streaming_embedder.embed(validation_tokens)
            batch_embeddings = batch_embedder.embed(validation_tokens)
            
            # Calculate consistency metrics
            mse = np.mean((streaming_embeddings - batch_embeddings) ** 2)
            mae = np.mean(np.abs(streaming_embeddings - batch_embeddings))
            
            # Calculate cosine similarity
            streaming_flat = streaming_embeddings.reshape(-1, streaming_embeddings.shape[-1])
            batch_flat = batch_embeddings.reshape(-1, batch_embeddings.shape[-1])
            
            cosine_similarities = []
            for i in range(len(streaming_flat)):
                s_vec = streaming_flat[i]
                b_vec = batch_flat[i]
                
                # Avoid division by zero
                s_norm = np.linalg.norm(s_vec)
                b_norm = np.linalg.norm(b_vec)
                
                if s_norm > 1e-8 and b_norm > 1e-8:
                    cosine_sim = np.dot(s_vec, b_vec) / (s_norm * b_norm)
                    cosine_similarities.append(cosine_sim)
            
            avg_cosine_similarity = np.mean(cosine_similarities) if cosine_similarities else 0.0
            
            # Calculate embedding matrix similarity
            streaming_matrix = streaming_embedder.get_embedding_matrix()
            batch_matrix = batch_embedder.get_embedding_matrix()
            
            matrix_mse = np.mean((streaming_matrix - batch_matrix) ** 2)
            matrix_correlation = np.corrcoef(streaming_matrix.flatten(), 
                                           batch_matrix.flatten())[0, 1]
            
            validation_metrics = {
                'embedding_mse': float(mse),
                'embedding_mae': float(mae),
                'avg_cosine_similarity': float(avg_cosine_similarity),
                'matrix_mse': float(matrix_mse),
                'matrix_correlation': float(matrix_correlation),
                'validation_samples': len(validation_data)
            }
            
            logger.info(f"Streaming consistency validation completed:")
            logger.info(f"  Embedding MSE: {mse:.6f}")
            logger.info(f"  Embedding MAE: {mae:.6f}")
            logger.info(f"  Avg Cosine Similarity: {avg_cosine_similarity:.4f}")
            logger.info(f"  Matrix Correlation: {matrix_correlation:.4f}")
            
            return validation_metrics
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'embedding_mse': float('inf'),
                'embedding_mae': float('inf'),
                'avg_cosine_similarity': 0.0,
                'matrix_mse': float('inf'),
                'matrix_correlation': 0.0,
                'validation_samples': len(validation_data),
                'error': str(e)
            }
    
    def fit_streaming_with_consistency(self, 
                                     data_source: Union[str, List[str], 'StreamingDataIterator'],
                                     batch_size: int = 1000,
                                     epochs: int = 100,
                                     max_position: int = 10000,
                                     temperature: float = 1.0,
                                     memory_threshold_mb: float = 1000.0,
                                     progress_callback: Optional[callable] = None,
                                     auto_adjust_batch_size: bool = True,
                                     min_batch_size: int = 100,
                                     max_batch_size: int = 10000,
                                     enable_checkpointing: bool = True,
                                     checkpoint_dir: Optional[str] = None,
                                     checkpoint_frequency: int = 10,
                                     validate_consistency: bool = True,
                                     validation_sample_size: int = 100,
                                     deterministic_seed: Optional[int] = None) -> 'SinusoidalEmbedder':
        """
        Fit a sinusoidal embedder with streaming consistency features.
        
        This enhanced version of fit_streaming includes:
        - Deterministic processing across streaming batches
        - Checkpointing and resumable streaming functionality
        - Validation to ensure streaming results match batch processing
        
        Args:
            data_source: Data source (file path, directory, file list, or StreamingDataIterator)
            batch_size: Initial batch size for processing
            epochs: Number of training epochs
            max_position: Maximum position for positional encoding
            temperature: Temperature parameter for sinusoidal patterns
            memory_threshold_mb: Memory threshold in MB for automatic batch size adjustment
            progress_callback: Optional callback function for progress updates
            auto_adjust_batch_size: Whether to automatically adjust batch size based on memory
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            enable_checkpointing: Whether to enable checkpointing for resumable training
            checkpoint_dir: Directory to save checkpoints (uses temp dir if None)
            checkpoint_frequency: Save checkpoint every N batches
            validate_consistency: Whether to validate streaming vs batch consistency
            validation_sample_size: Number of samples to use for validation
            deterministic_seed: Fixed seed for deterministic processing (auto-generated if None)
            
        Returns:
            Fitted SinusoidalEmbedder instance with consistency validation results
            
        Raises:
            InvalidInputError: If invalid parameters provided
            TokenizerError: If fitting fails
        """
        import psutil
        import time
        import tempfile
        from .streaming_data_iterator import StreamingDataIterator
        
        # Validate parameters
        if batch_size <= 0:
            raise InvalidInputError("batch_size", "positive integer", str(batch_size))
        if epochs <= 0:
            raise InvalidInputError("epochs", "positive integer", str(epochs))
        if memory_threshold_mb <= 0:
            raise InvalidInputError("memory_threshold_mb", "positive float", str(memory_threshold_mb))
        if min_batch_size <= 0 or min_batch_size > max_batch_size:
            raise InvalidInputError("min_batch_size", f"positive integer <= {max_batch_size}", str(min_batch_size))
        if checkpoint_frequency <= 0:
            raise InvalidInputError("checkpoint_frequency", "positive integer", str(checkpoint_frequency))
        if validation_sample_size <= 0:
            raise InvalidInputError("validation_sample_size", "positive integer", str(validation_sample_size))
        
        # Set up deterministic processing
        if deterministic_seed is None:
            deterministic_seed = self._create_deterministic_seed(data_source, batch_size, epochs)
        
        # Set random seeds for deterministic processing
        np.random.seed(deterministic_seed)
        random.seed(deterministic_seed)
        
        logger.info(f"Starting streaming tokenizer fitting with consistency features:")
        logger.info(f"  Batch size: {batch_size}, Epochs: {epochs}")
        logger.info(f"  Deterministic seed: {deterministic_seed}")
        logger.info(f"  Checkpointing: {enable_checkpointing}")
        logger.info(f"  Validation: {validate_consistency}")
        
        # Set up checkpointing
        checkpoint_path = None
        if enable_checkpointing:
            if checkpoint_dir is None:
                checkpoint_dir = tempfile.mkdtemp(prefix="lsm_streaming_checkpoint_")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "streaming_checkpoint.pkl")
        
        # Try to load existing checkpoint
        checkpoint_data = None
        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint_data = self._load_checkpoint(checkpoint_path)
        
        # Create streaming iterator if not provided
        if isinstance(data_source, StreamingDataIterator):
            streaming_iterator = data_source
        else:
            # Don't pass progress_callback to StreamingDataIterator as it has different signature
            streaming_iterator = StreamingDataIterator(
                data_source=data_source,
                batch_size=batch_size,
                memory_threshold_mb=memory_threshold_mb,
                auto_adjust_batch_size=auto_adjust_batch_size,
                progress_callback=None  # We'll handle progress in our own loop
            )
        
        # Initialize or restore state from checkpoint
        if checkpoint_data:
            logger.info("Resuming from checkpoint")
            total_sequences = checkpoint_data['total_sequences']
            total_tokens = checkpoint_data['total_tokens']
            vocab_stats = checkpoint_data['vocab_stats']
            sequence_lengths = checkpoint_data['sequence_lengths']
            start_epoch = checkpoint_data['epoch']
            start_batch_idx = checkpoint_data['batch_idx']
            
            # Restore embedder state if available
            if checkpoint_data.get('embedder_state'):
                # TODO: Implement embedder state restoration
                pass
        else:
            # Initialize fresh state
            total_sequences = 0
            total_tokens = 0
            vocab_stats = {}
            sequence_lengths = []
            start_epoch = 0
            start_batch_idx = 0
        
        current_batch_size = batch_size
        
        # Memory monitoring setup
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Create sinusoidal embedder
        self._sinusoidal_embedder = self.create_sinusoidal_embedder(max_position, temperature)
        
        # Progress tracking variables
        start_time = time.time()
        batches_processed = 0
        validation_data_collected = []
        
        try:
            # Multi-epoch training with streaming
            for epoch in range(start_epoch, epochs):
                logger.info(f"Starting epoch {epoch + 1}/{epochs}")
                epoch_start_time = time.time()
                epoch_sequences = 0
                epoch_tokens = 0
                
                # Reset iterator for new epoch (skip if resuming mid-epoch)
                if epoch > start_epoch or start_batch_idx == 0:
                    streaming_iterator.reset()
                
                # Process batches in current epoch
                batch_start_idx = start_batch_idx if epoch == start_epoch else 0
                for batch_idx, batch_data in enumerate(streaming_iterator):
                    # Skip batches if resuming from checkpoint
                    if batch_idx < batch_start_idx:
                        continue
                    
                    batch_start_time = time.time()
                    
                    # Monitor memory usage
                    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_usage = current_memory - initial_memory
                    
                    # Auto-adjust batch size if memory threshold exceeded
                    if auto_adjust_batch_size and memory_usage > memory_threshold_mb:
                        new_batch_size = max(min_batch_size, int(current_batch_size * 0.8))
                        if new_batch_size != current_batch_size:
                            current_batch_size = new_batch_size
                            streaming_iterator.batch_size = current_batch_size
                            logger.warning(f"Memory threshold exceeded ({memory_usage:.1f}MB), "
                                         f"reducing batch size to {current_batch_size}")
                    
                    # Process batch data
                    if len(batch_data) == 0:
                        continue  # Skip empty batches
                    
                    if isinstance(batch_data[0], str):
                        # Text data - tokenize
                        token_sequences = self.tokenize(batch_data, padding=True, truncation=True)
                        text_data = batch_data
                    elif isinstance(batch_data[0], dict):
                        # Dictionary data - extract text field and tokenize
                        text_data = []
                        for item in batch_data:
                            if isinstance(item, dict):
                                # Try to extract text from common fields
                                text = item.get('text', item.get('content', item.get('message', str(item))))
                            else:
                                text = str(item)
                            text_data.append(text)
                        token_sequences = self.tokenize(text_data, padding=True, truncation=True)
                    else:
                        # Already tokenized data
                        token_sequences = batch_data
                        text_data = None
                    
                    # Convert to numpy array for processing
                    token_sequences = np.array(token_sequences)
                    
                    # Collect validation data (deterministic sampling)
                    if (validate_consistency and 
                        len(validation_data_collected) < validation_sample_size and
                        text_data is not None):
                        # Use deterministic sampling based on batch index
                        sample_indices = np.random.RandomState(deterministic_seed + batch_idx).choice(
                            len(text_data), 
                            min(10, len(text_data)), 
                            replace=False
                        )
                        for idx in sample_indices:
                            if len(validation_data_collected) < validation_sample_size:
                                validation_data_collected.append(text_data[idx])
                    
                    # Update statistics
                    batch_sequences = len(token_sequences)
                    batch_tokens = np.sum(token_sequences != 0)  # Count non-padding tokens
                    
                    total_sequences += batch_sequences
                    total_tokens += batch_tokens
                    epoch_sequences += batch_sequences
                    epoch_tokens += batch_tokens
                    
                    # Collect vocabulary statistics
                    unique_tokens, counts = np.unique(token_sequences, return_counts=True)
                    for token, count in zip(unique_tokens, counts):
                        if token != 0:  # Skip padding tokens
                            vocab_stats[int(token)] = vocab_stats.get(int(token), 0) + int(count)
                    
                    # Collect sequence length statistics
                    for seq in token_sequences:
                        seq_len = np.sum(seq != 0)  # Count non-padding tokens
                        sequence_lengths.append(seq_len)
                    
                    # Fit embedder on current batch
                    if epoch == 0:  # Only fit on first epoch to build embeddings
                        self._sinusoidal_embedder.fit_batch(token_sequences)
                    else:  # Update embeddings on subsequent epochs
                        self._sinusoidal_embedder.update_batch(token_sequences)
                    
                    batches_processed += 1
                    batch_time = time.time() - batch_start_time
                    
                    # Save checkpoint periodically
                    if (enable_checkpointing and 
                        checkpoint_path and 
                        batch_idx % checkpoint_frequency == 0):
                        self._save_checkpoint(
                            checkpoint_path, epoch, batch_idx,
                            total_sequences, total_tokens, vocab_stats, sequence_lengths
                        )
                    
                    # Progress callback
                    if progress_callback:
                        progress_info = {
                            'epoch': epoch + 1,
                            'total_epochs': epochs,
                            'batch': batch_idx + 1,
                            'sequences_processed': total_sequences,
                            'tokens_processed': total_tokens,
                            'current_batch_size': current_batch_size,
                            'memory_usage_mb': memory_usage,
                            'batch_time_seconds': batch_time,
                            'avg_sequence_length': np.mean(sequence_lengths) if sequence_lengths else 0,
                            'deterministic_seed': deterministic_seed,
                            'checkpoint_enabled': enable_checkpointing
                        }
                        progress_callback(progress_info)
                    
                    # Log progress periodically
                    if batch_idx % 10 == 0:
                        logger.info(f"Epoch {epoch + 1}, Batch {batch_idx + 1}: "
                                  f"processed {batch_sequences} sequences, "
                                  f"memory usage: {memory_usage:.1f}MB, "
                                  f"batch time: {batch_time:.2f}s")
                
                # Reset start_batch_idx for subsequent epochs
                start_batch_idx = 0
                
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Completed epoch {epoch + 1}: "
                          f"processed {epoch_sequences} sequences, "
                          f"epoch time: {epoch_time:.2f}s")
            
            # Finalize embedder training
            if total_sequences > 0:
                self._sinusoidal_embedder.finalize_training()
                self._is_fitted = True
            else:
                # Handle empty data case - create minimal embedder
                logger.warning("No sequences processed - creating minimal embedder")
                self._sinusoidal_embedder._embedding_matrix = self._sinusoidal_embedder._initialize_embeddings()
                self._sinusoidal_embedder._positional_encodings = self._sinusoidal_embedder._create_positional_encodings()
                self._sinusoidal_embedder._is_fitted = True
                self._is_fitted = True
            
            # Calculate final statistics
            total_time = time.time() - start_time
            avg_sequence_length = np.mean(sequence_lengths) if sequence_lengths else 0
            vocab_coverage = len(vocab_stats) / self.get_vocab_size() * 100
            
            # Validate streaming consistency
            validation_metrics = {}
            if validate_consistency and validation_data_collected:
                validation_metrics = self._validate_streaming_consistency(
                    self._sinusoidal_embedder, validation_data_collected
                )
            
            # Log final statistics
            logger.info(f"Streaming tokenizer fitting completed:")
            logger.info(f"  Total sequences processed: {total_sequences:,}")
            logger.info(f"  Total tokens processed: {total_tokens:,}")
            logger.info(f"  Average sequence length: {avg_sequence_length:.1f}")
            logger.info(f"  Vocabulary coverage: {vocab_coverage:.1f}%")
            logger.info(f"  Total training time: {total_time:.2f}s")
            logger.info(f"  Batches processed: {batches_processed}")
            logger.info(f"  Final batch size: {current_batch_size}")
            logger.info(f"  Deterministic seed used: {deterministic_seed}")
            
            # Store training statistics
            self._training_stats = {
                'total_sequences': total_sequences,
                'total_tokens': total_tokens,
                'avg_sequence_length': avg_sequence_length,
                'vocab_stats': vocab_stats,
                'vocab_coverage': vocab_coverage,
                'training_time': total_time,
                'batches_processed': batches_processed,
                'final_batch_size': current_batch_size,
                'epochs': epochs,
                'memory_threshold_mb': memory_threshold_mb,
                'deterministic_seed': deterministic_seed,
                'checkpointing_enabled': enable_checkpointing,
                'validation_metrics': validation_metrics
            }
            
            # Clean up checkpoint if successful
            if enable_checkpointing and checkpoint_path and os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    logger.info("Training checkpoint cleaned up")
                except Exception as e:
                    logger.warning(f"Failed to clean up checkpoint: {str(e)}")
            
            return self._sinusoidal_embedder
            
        except Exception as e:
            logger.error(f"Error during streaming tokenizer fitting: {str(e)}")
            
            # Save emergency checkpoint
            if enable_checkpointing and checkpoint_path:
                try:
                    self._save_checkpoint(
                        checkpoint_path + ".emergency", epoch, batch_idx,
                        total_sequences, total_tokens, vocab_stats, sequence_lengths
                    )
                    logger.info(f"Emergency checkpoint saved to {checkpoint_path}.emergency")
                except Exception as checkpoint_error:
                    logger.error(f"Failed to save emergency checkpoint: {str(checkpoint_error)}")
            
            raise TokenizerError(f"Streaming fitting failed: {str(e)}")
    
    def get_cached_embedding(self, token_id: int, embedding_getter: callable) -> np.ndarray:
        """
        Get embedding for a single token with intelligent caching.
        
        Args:
            token_id: Token ID to get embedding for
            embedding_getter: Function to get embedding if not cached
            
        Returns:
            Token embedding
        """
        if self._caching_system is None:
            # No caching - call embedding getter directly
            return embedding_getter([token_id])[0]
        
        return self._caching_system.get_embedding(token_id, embedding_getter)
    
    def get_cached_batch_embeddings(self, token_ids: List[int], 
                                   embedding_getter: callable) -> np.ndarray:
        """
        Get embeddings for a batch of tokens with intelligent caching.
        
        Args:
            token_ids: List of token IDs
            embedding_getter: Function to get embeddings for missing tokens
            
        Returns:
            Batch of embeddings
        """
        if self._caching_system is None:
            # No caching - call embedding getter directly
            return embedding_getter(token_ids)
        
        return self._caching_system.get_batch_embeddings(token_ids, embedding_getter)
    
    def warm_embedding_cache(self, embedding_getter: callable,
                           token_frequencies: Optional[Dict[int, int]] = None):
        """
        Warm the embedding cache using the configured strategy.
        
        Args:
            embedding_getter: Function to get embeddings
            token_frequencies: Token frequency data (for frequency-based warming)
        """
        if self._caching_system is None:
            logger.warning("Caching is disabled - cannot warm cache")
            return
        
        vocab_size = self.get_vocab_size()
        self._caching_system.warm_cache(
            embedding_getter, 
            vocab_size=vocab_size,
            token_frequencies=token_frequencies
        )
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary with cache statistics or None if caching disabled
        """
        if self._caching_system is None:
            return None
        
        return self._caching_system.get_cache_stats()
    
    def clear_cache(self):
        """Clear all caches."""
        if self._caching_system is not None:
            self._caching_system.clear_all_caches()
            logger.info("Cleared embedding caches")
    
    def configure_cache(self, cache_config: CacheConfig):
        """
        Reconfigure the caching system.
        
        Args:
            cache_config: New cache configuration
        """
        if self._caching_system is not None:
            # Shutdown existing system
            self._caching_system.shutdown()
        
        # Create new caching system
        self._caching_system = IntelligentCachingSystem(cache_config)
        logger.info(f"Reconfigured caching system with new config: {cache_config}")
    
    def enable_cache_persistence(self, cache_file_path: str):
        """
        Enable cache persistence to disk.
        
        Args:
            cache_file_path: Path to save/load cache state
        """
        if self._caching_system is None:
            logger.warning("Caching is disabled - cannot enable persistence")
            return
        
        self._caching_system.config.enable_persistence = True
        self._caching_system.config.cache_file_path = cache_file_path
        
        # Try to load existing cache state
        self._caching_system.load_cache_state(cache_file_path)
        
        logger.info(f"Enabled cache persistence: {cache_file_path}")
    
    def shutdown_caching(self):
        """Shutdown the caching system and clean up resources."""
        if self._caching_system is not None:
            self._caching_system.shutdown()
            logger.info("Shutdown caching system")
    
    def save(self, save_path: str) -> None:
        """
        Save enhanced tokenizer wrapper.
        
        Args:
            save_path: Directory path to save wrapper
        """
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save adapter configuration
            self._adapter.save_adapter_config(save_path)
            
            # Save wrapper configuration
            wrapper_config = {
                'embedding_dim': self.embedding_dim,
                'max_length': self.max_length,
                'special_tokens': self.special_tokens,
                'backend_specific_config': self.backend_specific_config,
                'is_fitted': self._is_fitted,
                'adapter_backend': self._adapter.config.backend,
                'enable_caching': self.enable_caching
            }
            
            # Save cache configuration if caching is enabled
            if self._caching_system is not None:
                cache_config_dict = {
                    'max_cache_size': self._caching_system.config.max_cache_size,
                    'enable_batch_caching': self._caching_system.config.enable_batch_caching,
                    'batch_cache_size': self._caching_system.config.batch_cache_size,
                    'enable_cache_warming': self._caching_system.config.enable_cache_warming,
                    'warmup_strategy': self._caching_system.config.warmup_strategy,
                    'warmup_size': self._caching_system.config.warmup_size,
                    'enable_metrics': self._caching_system.config.enable_metrics,
                    'enable_persistence': self._caching_system.config.enable_persistence,
                    'memory_threshold_mb': self._caching_system.config.memory_threshold_mb
                }
                wrapper_config['cache_config'] = cache_config_dict
            
            config_path = os.path.join(save_path, 'enhanced_tokenizer_config.json')
            with open(config_path, 'w') as f:
                json.dump(wrapper_config, f, indent=2)
            
            # Save sinusoidal embedder if fitted
            if self._sinusoidal_embedder is not None:
                embedder_path = os.path.join(save_path, 'sinusoidal_embedder')
                self._sinusoidal_embedder.save(embedder_path)
            
            logger.info(f"EnhancedTokenizerWrapper saved to {save_path}")
            
        except Exception as e:
            raise TokenizerSaveError(save_path, str(e))
    
    @classmethod
    def load(cls, load_path: str) -> 'EnhancedTokenizerWrapper':
        """
        Load enhanced tokenizer wrapper.
        
        Args:
            load_path: Directory path to load wrapper from
            
        Returns:
            Loaded EnhancedTokenizerWrapper instance
        """
        try:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Load path does not exist: {load_path}")
            
            # Load wrapper configuration
            config_path = os.path.join(load_path, 'enhanced_tokenizer_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                wrapper_config = json.load(f)
            
            # Load adapter
            backend = wrapper_config['adapter_backend']
            adapter_class = TokenizerRegistry.get_adapter_class(backend)
            adapter = adapter_class.load_adapter_config(load_path)
            
            # Restore cache configuration if available
            cache_config = None
            if 'cache_config' in wrapper_config:
                cache_config = CacheConfig(**wrapper_config['cache_config'])
            
            # Create wrapper instance
            wrapper = cls(
                tokenizer=adapter,
                embedding_dim=wrapper_config['embedding_dim'],
                max_length=wrapper_config['max_length'],
                special_tokens=wrapper_config['special_tokens'],
                backend_specific_config=wrapper_config['backend_specific_config'],
                enable_caching=wrapper_config.get('enable_caching', True),
                cache_config=cache_config
            )
            
            wrapper._is_fitted = wrapper_config['is_fitted']
            
            # Load sinusoidal embedder if exists
            embedder_path = os.path.join(load_path, 'sinusoidal_embedder')
            if os.path.exists(embedder_path):
                from .tokenization import SinusoidalEmbedder
                wrapper._sinusoidal_embedder = SinusoidalEmbedder.load(embedder_path)
            
            logger.info(f"EnhancedTokenizerWrapper loaded from {load_path}")
            return wrapper
            
        except Exception as e:
            raise TokenizerLoadError(load_path, str(e))
    
    def __repr__(self) -> str:
        return (f"EnhancedTokenizerWrapper(adapter={self._adapter}, "
                f"embedding_dim={self.embedding_dim}, fitted={self._is_fitted})")


# Auto-discovery and registration of adapters
def _discover_and_register_adapters():
    """Discover and register available tokenizer adapters."""
    try:
        # Import and register HuggingFace adapter
        from .adapters.huggingface_adapter import register_huggingface_adapter
        register_huggingface_adapter()
        logger.info("Registered HuggingFace adapter")
    except ImportError as e:
        logger.warning(f"Could not register HuggingFace adapter: {e}")
    
    try:
        # Import and register tiktoken adapter
        from .adapters.tiktoken_adapter import register_tiktoken_adapter
        register_tiktoken_adapter()
        logger.info("Registered tiktoken adapter")
    except ImportError as e:
        logger.warning(f"Could not register tiktoken adapter: {e}")
    
    try:
        # Import and register spaCy adapter
        from .adapters.spacy_adapter import register_spacy_adapter
        register_spacy_adapter()
        logger.info("Registered spaCy adapter")
    except ImportError as e:
        logger.warning(f"Could not register spaCy adapter: {e}")
    
    try:
        # Import and register custom tokenizer adapter
        from .adapters.custom_adapter import register_custom_adapter
        register_custom_adapter()
        logger.info("Registered custom tokenizer adapter")
    except ImportError as e:
        logger.warning(f"Could not register custom tokenizer adapter: {e}")
    
    logger.info(f"Enhanced tokenizer system initialized with {len(TokenizerRegistry.list_available_backends())} backends")


# Initialize the system
_discover_and_register_adapters()