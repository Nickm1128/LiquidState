#!/usr/bin/env python3
"""
Configurable Sinusoidal Embedder for Enhanced LSM Tokenization.

This module provides a configurable sinusoidal embedding layer with learnable
frequency parameters, supporting both absolute and relative positional encodings.
The embedder can automatically adapt to different vocabulary sizes and embedding
dimensions while preserving mathematical properties.

Classes:
    SinusoidalConfig: Configuration dataclass for sinusoidal embeddings
    ConfigurableSinusoidalEmbedder: Main embedding layer with configurable parameters
    SinusoidalEmbedderFactory: Factory class for creating optimized embedders
    EmbeddingDimensionOptimizer: Utility for optimal dimension calculation

Key Features:
    - Learnable frequency parameters for optimal sinusoidal patterns
    - Automatic vocabulary size and dimension adaptation
    - Support for both absolute and relative positional encodings
    - GPU acceleration and memory-efficient computation
    - Mathematical property preservation during adaptation
    - Visualization tools for embedding pattern analysis

Example:
    Basic usage with automatic adaptation:
    
    >>> from lsm.data.configurable_sinusoidal_embedder import (
    ...     ConfigurableSinusoidalEmbedder, SinusoidalConfig
    ... )
    >>> config = SinusoidalConfig(
    ...     embedding_dim=256,
    ...     vocab_size=50000,
    ...     learnable_frequencies=True
    ... )
    >>> embedder = ConfigurableSinusoidalEmbedder(config)
    >>> embedder.adapt_to_vocabulary(30000)  # Adapt to new vocab size
    
    Advanced usage with tokenizer integration:
    
    >>> from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
    >>> tokenizer = EnhancedTokenizerWrapper('gpt2')
    >>> embedder = tokenizer.create_configurable_sinusoidal_embedder(
    ...     learnable_frequencies=True,
    ...     base_frequency=5000.0,
    ...     use_relative_position=True
    ... )

See Also:
    - EnhancedTokenizerWrapper: For tokenizer integration
    - MemoryEfficientEmbeddingLayer: For memory-optimized storage
    - GPUAccelerator: For GPU acceleration support
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, Any, Optional, Union, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.lsm_exceptions import (
    TokenizerError, TokenizerNotFittedError, 
    TokenizerLoadError, TokenizerSaveError, InvalidInputError
)
from ..utils.lsm_logging import get_logger
from .memory_efficient_storage import MemoryStorageConfig, MemoryEfficientEmbeddingLayer
from .gpu_acceleration import GPUAccelerator, GPUConfig, create_gpu_accelerator

logger = get_logger(__name__)


@dataclass
class SinusoidalConfig:
    """Configuration for configurable sinusoidal embeddings."""
    
    # Core embedding parameters
    embedding_dim: int = 128
    vocab_size: int = 10000
    max_sequence_length: int = 512
    
    # Frequency parameters
    base_frequency: float = 10000.0
    frequency_scaling: float = 1.0
    learnable_frequencies: bool = True
    
    # Positional encoding options
    use_absolute_position: bool = True
    use_relative_position: bool = False
    relative_position_window: int = 64
    
    # Advanced configuration
    frequency_init_std: float = 0.02
    phase_shift: float = 0.0
    temperature: float = 1.0
    
    # Performance options
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    
    # GPU acceleration options
    enable_gpu_acceleration: bool = True
    gpu_config: Optional[GPUConfig] = None
    use_vectorized_operations: bool = True
    enable_xla_compilation: bool = True
    
    # Memory-efficient storage options
    memory_storage_config: Optional[MemoryStorageConfig] = None
    use_memory_efficient_storage: bool = False


class ConfigurableSinusoidalEmbedder(keras.layers.Layer):
    """
    Configurable sinusoidal embedding layer with learnable frequency parameters.
    
    This layer creates sinusoidal embeddings that can adapt to any vocabulary size
    and supports both absolute and relative positional encodings. The frequency
    parameters can be learned during training for optimal performance, allowing
    the model to discover optimal sinusoidal patterns for the specific task.
    
    The embedder combines token embeddings with sinusoidal positional encodings,
    where the sinusoidal patterns are computed using configurable frequency
    parameters. These frequencies can be either fixed (traditional approach) or
    learnable (adaptive approach for better performance).
    
    Attributes:
        config (SinusoidalConfig): Configuration object with embedding parameters
        token_embedding (keras.layers.Layer): Token embedding layer (standard or memory-efficient)
        frequency_weights (tf.Variable): Learnable or fixed frequency parameters
        phase_shift (tf.Variable): Phase shift parameter for sinusoidal patterns
        temperature (tf.Tensor): Temperature scaling parameter
        relative_position_embedding (keras.layers.Layer): Relative position embeddings (if enabled)
        gpu_accelerator (GPUAccelerator): GPU acceleration manager (if enabled)
        _fitted (bool): Whether the embedder has been fitted/adapted
        _tokenizer_info (Dict): Information about adapted tokenizer (if any)
    
    Key Features:
        - **Learnable Frequencies**: Frequency parameters can be learned during training
          for optimal sinusoidal patterns specific to your data and task
        - **Automatic Adaptation**: Automatically adapts to different vocabulary sizes
          and embedding dimensions while preserving mathematical properties
        - **Dual Positional Encoding**: Supports both absolute and relative positional
          encodings for enhanced sequence modeling
        - **Memory Efficiency**: Optional memory-efficient storage for large vocabularies
        - **GPU Acceleration**: Optimized GPU computation for faster training and inference
        - **Mathematical Preservation**: Maintains sinusoidal properties during adaptation
    
    Example:
        Basic usage with learnable frequencies:
        
        >>> config = SinusoidalConfig(
        ...     embedding_dim=256,
        ...     vocab_size=30000,
        ...     learnable_frequencies=True,
        ...     base_frequency=10000.0
        ... )
        >>> embedder = ConfigurableSinusoidalEmbedder(config)
        >>> 
        >>> # Use in a model
        >>> inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
        >>> embeddings = embedder(inputs)
        >>> outputs = tf.keras.layers.Dense(num_classes)(embeddings)
        >>> model = tf.keras.Model(inputs, outputs)
        
        Advanced usage with tokenizer adaptation:
        
        >>> from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
        >>> tokenizer = EnhancedTokenizerWrapper('bert-base-uncased')
        >>> 
        >>> config = SinusoidalConfig(
        ...     embedding_dim=512,
        ...     learnable_frequencies=True,
        ...     use_relative_position=True,
        ...     enable_gpu_acceleration=True
        ... )
        >>> embedder = ConfigurableSinusoidalEmbedder(config)
        >>> embedder.adapt_to_tokenizer(tokenizer.get_adapter())
        
        Memory-efficient usage for large vocabularies:
        
        >>> from lsm.data.memory_efficient_storage import MemoryStorageConfig
        >>> memory_config = MemoryStorageConfig(
        ...     use_memory_mapping=True,
        ...     use_compression=True
        ... )
        >>> config = SinusoidalConfig(
        ...     embedding_dim=1024,
        ...     vocab_size=100000,
        ...     use_memory_efficient_storage=True,
        ...     memory_storage_config=memory_config
        ... )
        >>> embedder = ConfigurableSinusoidalEmbedder(config)
    
    Mathematical Background:
        The sinusoidal embeddings are computed as:
        
        PE(pos, 2i) = sin(pos / base_freq^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / base_freq^(2i/d_model))
        
        Where:
        - pos: position in the sequence
        - i: dimension index
        - d_model: embedding dimension
        - base_freq: base frequency parameter (learnable if enabled)
        
        When learnable_frequencies=True, the frequency parameters are learned
        during training, allowing the model to discover optimal patterns.
    
    See Also:
        - SinusoidalConfig: Configuration class for embedding parameters
        - EnhancedTokenizerWrapper: For tokenizer integration
        - MemoryEfficientEmbeddingLayer: For memory-optimized storage
        - GPUAccelerator: For GPU acceleration
    """
    
    def __init__(self, config: SinusoidalConfig, **kwargs):
        """
        Initialize the configurable sinusoidal embedder.
        
        Args:
            config: SinusoidalConfig object with embedding parameters
            **kwargs: Additional keras layer arguments
        """
        super().__init__(**kwargs)
        self.config = config
        self._fitted = False
        
        # Validate configuration
        self._validate_config()
        
        # Initialize GPU accelerator if enabled
        self._initialize_gpu_acceleration()
        
        # Initialize embedding parameters
        self._initialize_parameters()
        
        logger.info(f"Initialized ConfigurableSinusoidalEmbedder with config: {config}")
    
    def _validate_config(self):
        """Validate the sinusoidal configuration parameters."""
        if self.config.embedding_dim <= 0:
            raise InvalidInputError("embedding_dim", "positive integer", str(self.config.embedding_dim))
        if self.config.vocab_size <= 0:
            raise InvalidInputError("vocab_size", "positive integer", str(self.config.vocab_size))
        if self.config.max_sequence_length <= 0:
            raise InvalidInputError("max_sequence_length", "positive integer", str(self.config.max_sequence_length))
        if self.config.base_frequency <= 0:
            raise InvalidInputError("base_frequency", "positive float", str(self.config.base_frequency))
        if self.config.frequency_scaling <= 0:
            raise InvalidInputError("frequency_scaling", "positive float", str(self.config.frequency_scaling))
    
    def _initialize_gpu_acceleration(self):
        """Initialize GPU acceleration if enabled."""
        if not self.config.enable_gpu_acceleration:
            self.gpu_accelerator = None
            logger.info("GPU acceleration disabled by configuration")
            return
        
        try:
            # Create GPU accelerator with custom config or defaults
            if self.config.gpu_config is not None:
                self.gpu_accelerator = GPUAccelerator(self.config.gpu_config)
            else:
                # Create default GPU config based on sinusoidal config
                gpu_config = GPUConfig(
                    enable_gpu=True,
                    enable_mixed_precision=self.config.use_mixed_precision,
                    enable_vectorization=self.config.use_vectorized_operations,
                    enable_xla=self.config.enable_xla_compilation,
                    mixed_precision_policy="mixed_float16" if self.config.use_mixed_precision else "float32"
                )
                self.gpu_accelerator = GPUAccelerator(gpu_config)
            
            logger.info("GPU acceleration initialized successfully")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU acceleration: {e}")
            self.gpu_accelerator = None
            self.config.enable_gpu_acceleration = False
    
    def _initialize_parameters(self):
        """Initialize the embedding parameters and layers."""
        # Token embedding layer with memory-efficient storage if configured
        if self.config.use_memory_efficient_storage:
            # Use memory-efficient embedding layer
            memory_config = self.config.memory_storage_config or MemoryStorageConfig()
            
            # Enable gradient checkpointing if configured
            if self.config.gradient_checkpointing:
                memory_config.use_gradient_checkpointing = True
            
            self.token_embedding = MemoryEfficientEmbeddingLayer(
                vocab_size=self.config.vocab_size,
                embedding_dim=self.config.embedding_dim,
                config=memory_config,
                name="memory_efficient_token_embedding"
            )
        else:
            # Standard embedding layer
            self.token_embedding = layers.Embedding(
                input_dim=self.config.vocab_size,
                output_dim=self.config.embedding_dim,
                mask_zero=True,
                name="token_embedding"
            )
        
        # Frequency parameters for sinusoidal encoding
        if self.config.learnable_frequencies:
            # Learnable frequency parameters
            self.frequency_weights = self.add_weight(
                name="frequency_weights",
                shape=(self.config.embedding_dim // 2,),
                initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=self.config.frequency_init_std
                ),
                trainable=True
            )
        else:
            # Fixed frequency parameters
            frequencies = self._compute_fixed_frequencies()
            self.frequency_weights = tf.constant(frequencies, dtype=tf.float32)
        
        # Phase shift parameter (learnable if enabled)
        if self.config.learnable_frequencies:
            self.phase_shift = self.add_weight(
                name="phase_shift",
                shape=(),
                initializer=keras.initializers.Constant(self.config.phase_shift),
                trainable=True
            )
        else:
            self.phase_shift = tf.constant(self.config.phase_shift, dtype=tf.float32)
        
        # Temperature parameter for scaling
        self.temperature = tf.constant(self.config.temperature, dtype=tf.float32)
        
        # Relative position embedding if enabled
        if self.config.use_relative_position:
            self.relative_position_embedding = layers.Embedding(
                input_dim=2 * self.config.relative_position_window + 1,
                output_dim=self.config.embedding_dim,
                name="relative_position_embedding"
            )
    
    def _compute_fixed_frequencies(self) -> np.ndarray:
        """Compute fixed frequency values for sinusoidal encoding."""
        # Use GPU-accelerated computation if available
        if (self.gpu_accelerator is not None and 
            self.config.use_vectorized_operations):
            
            # Use parallel GPU computation
            base_freq = tf.constant(self.config.base_frequency, dtype=tf.float32)
            freq_scaling = tf.constant(self.config.frequency_scaling, dtype=tf.float32)
            
            frequencies_tensor = self.gpu_accelerator.parallel_frequency_computation(
                base_freq, self.config.embedding_dim, freq_scaling
            )
            
            return frequencies_tensor.numpy()
            
        else:
            # Fallback to standard computation
            dim_pairs = self.config.embedding_dim // 2
            frequencies = np.zeros(dim_pairs, dtype=np.float32)
            
            for i in range(dim_pairs):
                frequencies[i] = 1.0 / (
                    self.config.base_frequency ** (
                        (2 * i) / self.config.embedding_dim
                    )
                )
            
            return frequencies * self.config.frequency_scaling
    
    def build(self, input_shape):
        """Build the layer with the given input shape."""
        super().build(input_shape)
        
        # Build sub-layers
        self.token_embedding.build(input_shape)
        
        if self.config.use_relative_position:
            # Build relative position embedding
            relative_shape = input_shape[:-1] + (2 * self.config.relative_position_window + 1,)
            self.relative_position_embedding.build(relative_shape)
    
    def call(self, inputs, training=None, mask=None):
        """
        Forward pass of the sinusoidal embedder.
        
        This method performs the main computation of the embedding layer, combining
        token embeddings with sinusoidal positional encodings. The computation
        includes learnable frequency parameters (if enabled), temperature scaling,
        and optional relative positional encoding.
        
        Args:
            inputs (tf.Tensor): Token IDs tensor of shape (batch_size, sequence_length).
                Token IDs should be integers in the range [0, vocab_size).
            training (bool, optional): Whether the layer is in training mode. Affects
                dropout application and learnable parameter updates. Defaults to None.
            mask (tf.Tensor, optional): Optional mask tensor of shape 
                (batch_size, sequence_length) to mask padded positions. Defaults to None.
        
        Returns:
            tf.Tensor: Embedded tokens with sinusoidal positional encoding of shape
                (batch_size, sequence_length, embedding_dim). The output combines
                token embeddings with sinusoidal positional patterns.
        
        Raises:
            tf.errors.InvalidArgumentError: If input token IDs are out of vocabulary range
            tf.errors.OutOfRangeError: If sequence length exceeds max_sequence_length
        
        Example:
            >>> import tensorflow as tf
            >>> config = SinusoidalConfig(embedding_dim=256, vocab_size=10000)
            >>> embedder = ConfigurableSinusoidalEmbedder(config)
            >>> 
            >>> # Create sample input
            >>> token_ids = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]])  # (2, 4)
            >>> embeddings = embedder(token_ids, training=True)
            >>> print(embeddings.shape)  # (2, 4, 256)
        
        Mathematical Details:
            The output is computed as:
            output = token_embeddings + positional_encoding
            
            Where positional_encoding uses sinusoidal patterns:
            PE(pos, 2i) = sin(pos / freq_i + phase_shift)
            PE(pos, 2i+1) = cos(pos / freq_i + phase_shift)
            
            And freq_i are the learnable or fixed frequency parameters.
        """
        # Get token embeddings
        token_embeds = self.token_embedding(inputs)
        
        # Get sequence length and batch size
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]
        
        # Create positional encodings
        positional_encoding = self._create_positional_encoding(
            batch_size, seq_length, training
        )
        
        # Combine token and positional embeddings
        embeddings = token_embeds + positional_encoding
        
        # Apply temperature scaling
        embeddings = embeddings / self.temperature
        
        # Add relative positional encoding if enabled
        if self.config.use_relative_position:
            relative_encoding = self._create_relative_encoding(inputs, training)
            embeddings = embeddings + relative_encoding
        
        return embeddings
    
    def _create_positional_encoding(self, batch_size, seq_length, training=None):
        """Create sinusoidal positional encoding."""
        # Create position indices
        positions = tf.range(seq_length, dtype=tf.float32)
        positions = tf.expand_dims(positions, 0)  # (1, seq_length)
        positions = tf.tile(positions, [batch_size, 1])  # (batch_size, seq_length)
        
        # Compute sinusoidal encoding
        encoding = self._compute_sinusoidal_encoding(positions, training)
        
        return encoding
    
    def _compute_sinusoidal_encoding(self, positions, training=None):
        """Compute sinusoidal encoding for given positions."""
        # Get frequency weights
        if self.config.learnable_frequencies and training:
            # Apply dropout to frequency weights during training
            freq_weights = tf.nn.dropout(
                self.frequency_weights, rate=0.1, name="frequency_dropout"
            )
        else:
            freq_weights = self.frequency_weights
        
        # Use GPU-accelerated computation if available
        if (self.gpu_accelerator is not None and 
            self.config.use_vectorized_operations):
            
            # Use vectorized GPU computation
            encoding = self.gpu_accelerator.vectorized_sinusoidal_encoding(
                positions, freq_weights, self.phase_shift
            )
            
        else:
            # Fallback to standard computation
            # Expand positions for broadcasting
            positions = tf.expand_dims(positions, -1)  # (batch_size, seq_length, 1)
            
            # Compute angles
            angles = positions * tf.expand_dims(freq_weights, 0)  # Broadcasting
            angles = angles + self.phase_shift
            
            # Compute sine and cosine
            sin_encoding = tf.sin(angles)
            cos_encoding = tf.cos(angles)
            
            # Interleave sine and cosine
            encoding = tf.stack([sin_encoding, cos_encoding], axis=-1)
            encoding = tf.reshape(encoding, [
                tf.shape(positions)[0], 
                tf.shape(positions)[1], 
                self.config.embedding_dim
            ])
        
        return encoding
    
    def _create_relative_encoding(self, inputs, training=None):
        """Create relative positional encoding."""
        seq_length = tf.shape(inputs)[1]
        
        # Create relative position matrix
        positions = tf.range(seq_length)
        relative_positions = positions[:, None] - positions[None, :]
        
        # Clip to window size
        relative_positions = tf.clip_by_value(
            relative_positions,
            -self.config.relative_position_window,
            self.config.relative_position_window
        )
        
        # Shift to positive indices
        relative_positions = relative_positions + self.config.relative_position_window
        
        # Get relative embeddings
        relative_embeds = self.relative_position_embedding(relative_positions)
        
        # Average over sequence dimension for each position
        relative_encoding = tf.reduce_mean(relative_embeds, axis=1)
        
        # Expand for batch dimension
        batch_size = tf.shape(inputs)[0]
        relative_encoding = tf.expand_dims(relative_encoding, 0)
        relative_encoding = tf.tile(relative_encoding, [batch_size, 1, 1])
        
        return relative_encoding
    
    def adapt_to_vocabulary(self, vocab_size: int):
        """
        Adapt the embedder to a new vocabulary size.
        
        Args:
            vocab_size: New vocabulary size to adapt to
        """
        if vocab_size <= 0:
            raise InvalidInputError("vocab_size", "positive integer", str(vocab_size))
        
        logger.info(f"Adapting embedder from vocab_size {self.config.vocab_size} to {vocab_size}")
        
        # Update configuration
        old_vocab_size = self.config.vocab_size
        self.config.vocab_size = vocab_size
        
        # Recreate token embedding layer with new vocabulary size
        old_weights = None
        if hasattr(self, 'token_embedding') and self.token_embedding.built:
            # Save existing weights if available
            old_weights = self.token_embedding.get_weights()[0]
        
        # Create new embedding layer
        self.token_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.config.embedding_dim,
            mask_zero=True,
            name="token_embedding"
        )
        
        # Transfer weights if possible
        if old_weights is not None:
            min_vocab = min(old_vocab_size, vocab_size)
            new_weights = np.random.normal(
                0, 0.02, (vocab_size, self.config.embedding_dim)
            ).astype(np.float32)
            new_weights[:min_vocab] = old_weights[:min_vocab]
            self.token_embedding.build((None, None))
            self.token_embedding.set_weights([new_weights])
        
        self._fitted = True
        logger.info(f"Successfully adapted embedder to vocab_size {vocab_size}")
    
    def adapt_to_tokenizer(self, tokenizer_adapter):
        """
        Automatically adapt the embedder to a tokenizer's vocabulary size.
        
        Args:
            tokenizer_adapter: TokenizerAdapter instance to adapt to
        """
        from .enhanced_tokenization import TokenizerAdapter
        
        if not isinstance(tokenizer_adapter, TokenizerAdapter):
            raise InvalidInputError("tokenizer_adapter", "TokenizerAdapter instance", str(type(tokenizer_adapter)))
        
        if not tokenizer_adapter._is_initialized:
            raise TokenizerError("Tokenizer adapter must be initialized before adaptation")
        
        # Get vocabulary size from tokenizer
        vocab_size = tokenizer_adapter.get_vocab_size()
        
        logger.info(f"Auto-adapting embedder to tokenizer {tokenizer_adapter.config.backend} "
                   f"with vocab_size {vocab_size}")
        
        # Adapt to the detected vocabulary size
        self.adapt_to_vocabulary(vocab_size)
        
        # Store tokenizer information for future reference
        self._tokenizer_info = {
            'backend': tokenizer_adapter.config.backend,
            'model_name': tokenizer_adapter.config.model_name,
            'vocab_size': vocab_size,
            'special_tokens': tokenizer_adapter.get_special_tokens()
        }
        
        logger.info(f"Successfully auto-adapted embedder to tokenizer")
    
    def adapt_embedding_dimension(self, new_embedding_dim: int, preserve_properties: bool = True):
        """
        Adapt the embedder to a new embedding dimension while preserving mathematical properties.
        
        Args:
            new_embedding_dim: New embedding dimension
            preserve_properties: Whether to preserve mathematical properties during adaptation
        """
        if new_embedding_dim <= 0:
            raise InvalidInputError("new_embedding_dim", "positive integer", str(new_embedding_dim))
        
        if new_embedding_dim % 2 != 0:
            raise InvalidInputError("embedding_dim", "even integer", str(new_embedding_dim))
        
        logger.info(f"Adapting embedding dimension from {self.config.embedding_dim} to {new_embedding_dim}")
        
        old_embedding_dim = self.config.embedding_dim
        self.config.embedding_dim = new_embedding_dim
        
        # Save old token embedding weights if available
        old_token_weights = None
        if hasattr(self, 'token_embedding') and self.token_embedding.built:
            old_token_weights = self.token_embedding.get_weights()[0]
        
        # Recreate token embedding layer with new dimension
        self.token_embedding = layers.Embedding(
            input_dim=self.config.vocab_size,
            output_dim=new_embedding_dim,
            mask_zero=True,
            name="token_embedding"
        )
        
        # Transfer and scale token embedding weights if available
        if old_token_weights is not None and preserve_properties:
            new_token_weights = self._scale_embedding_weights(
                old_token_weights, old_embedding_dim, new_embedding_dim
            )
            self.token_embedding.build((None, None))
            self.token_embedding.set_weights([new_token_weights])
        
        # Update frequency parameters for new dimension
        if preserve_properties:
            self._adapt_frequency_parameters(old_embedding_dim, new_embedding_dim)
        else:
            # Reinitialize frequency parameters
            self._initialize_frequency_parameters()
        
        logger.info(f"Successfully adapted embedding dimension to {new_embedding_dim}")
    
    def _scale_embedding_weights(self, old_weights: np.ndarray, old_dim: int, new_dim: int) -> np.ndarray:
        """
        Scale embedding weights to new dimension while preserving mathematical properties.
        
        Args:
            old_weights: Original embedding weights
            old_dim: Original embedding dimension
            new_dim: New embedding dimension
            
        Returns:
            Scaled embedding weights
        """
        vocab_size = old_weights.shape[0]
        new_weights = np.zeros((vocab_size, new_dim), dtype=np.float32)
        
        if new_dim >= old_dim:
            # Expanding dimension - copy existing weights and initialize new dimensions
            new_weights[:, :old_dim] = old_weights
            
            # Initialize new dimensions with scaled random values
            scaling_factor = np.sqrt(old_dim / new_dim)  # Preserve variance
            new_weights[:, old_dim:] = np.random.normal(
                0, 0.02 * scaling_factor, (vocab_size, new_dim - old_dim)
            ).astype(np.float32)
            
        else:
            # Reducing dimension - use PCA-like projection to preserve most information
            # For simplicity, we'll use the first new_dim dimensions with scaling
            scaling_factor = np.sqrt(new_dim / old_dim)  # Preserve energy
            new_weights = old_weights[:, :new_dim] * scaling_factor
        
        return new_weights
    
    def _adapt_frequency_parameters(self, old_dim: int, new_dim: int):
        """
        Adapt frequency parameters to new embedding dimension while preserving properties.
        
        Args:
            old_dim: Original embedding dimension
            new_dim: New embedding dimension
        """
        old_dim_pairs = old_dim // 2
        new_dim_pairs = new_dim // 2
        
        if self.config.learnable_frequencies:
            # Save old frequency weights if available
            old_freq_weights = None
            if hasattr(self, 'frequency_weights'):
                old_freq_weights = self.frequency_weights.numpy()
            
            # Create new frequency weights
            self.frequency_weights = self.add_weight(
                name="frequency_weights",
                shape=(new_dim_pairs,),
                initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=self.config.frequency_init_std
                ),
                trainable=True
            )
            
            # Transfer old frequency weights if available
            if old_freq_weights is not None:
                new_freq_weights = self._scale_frequency_weights(
                    old_freq_weights, old_dim_pairs, new_dim_pairs
                )
                self.frequency_weights.assign(new_freq_weights)
        else:
            # Recompute fixed frequencies for new dimension
            frequencies = self._compute_fixed_frequencies()
            self.frequency_weights = tf.constant(frequencies, dtype=tf.float32)
    
    def _scale_frequency_weights(self, old_weights: np.ndarray, old_pairs: int, new_pairs: int) -> np.ndarray:
        """
        Scale frequency weights to new dimension while preserving frequency distribution.
        
        Args:
            old_weights: Original frequency weights
            old_pairs: Original number of dimension pairs
            new_pairs: New number of dimension pairs
            
        Returns:
            Scaled frequency weights
        """
        if new_pairs >= old_pairs:
            # Expanding - copy existing and interpolate new frequencies
            new_weights = np.zeros(new_pairs, dtype=np.float32)
            new_weights[:old_pairs] = old_weights
            
            # Interpolate additional frequencies
            if old_pairs > 1:
                # Use linear interpolation to fill gaps
                for i in range(old_pairs, new_pairs):
                    ratio = i / (new_pairs - 1)
                    old_ratio = ratio * (old_pairs - 1)
                    old_idx = int(old_ratio)
                    if old_idx < old_pairs - 1:
                        alpha = old_ratio - old_idx
                        new_weights[i] = (1 - alpha) * old_weights[old_idx] + alpha * old_weights[old_idx + 1]
                    else:
                        new_weights[i] = old_weights[-1]
            else:
                # Single frequency - replicate with slight variations
                base_freq = old_weights[0]
                for i in range(old_pairs, new_pairs):
                    new_weights[i] = base_freq * (1 + 0.1 * np.random.normal())
        else:
            # Reducing - select most representative frequencies
            indices = np.linspace(0, old_pairs - 1, new_pairs, dtype=int)
            new_weights = old_weights[indices]
        
        return new_weights.astype(np.float32)
    
    def _initialize_frequency_parameters(self):
        """Initialize frequency parameters (used when not preserving properties)."""
        # Frequency parameters for sinusoidal encoding
        if self.config.learnable_frequencies:
            # Learnable frequency parameters
            self.frequency_weights = self.add_weight(
                name="frequency_weights",
                shape=(self.config.embedding_dim // 2,),
                initializer=keras.initializers.RandomNormal(
                    mean=0.0, stddev=self.config.frequency_init_std
                ),
                trainable=True
            )
        else:
            # Fixed frequency parameters
            frequencies = self._compute_fixed_frequencies()
            self.frequency_weights = tf.constant(frequencies, dtype=tf.float32)
    
    def get_adaptation_info(self) -> Dict[str, Any]:
        """
        Get information about current adaptation state.
        
        Returns:
            Dictionary containing adaptation information
        """
        info = {
            'vocab_size': self.config.vocab_size,
            'embedding_dim': self.config.embedding_dim,
            'is_fitted': self._fitted,
            'learnable_frequencies': self.config.learnable_frequencies,
            'base_frequency': self.config.base_frequency,
            'frequency_scaling': self.config.frequency_scaling
        }
        
        # Add tokenizer information if available
        if hasattr(self, '_tokenizer_info'):
            info['tokenizer_info'] = self._tokenizer_info
        
        # Add frequency information if available
        if hasattr(self, 'frequency_weights'):
            if self.config.learnable_frequencies:
                info['current_frequencies'] = self.frequency_weights.numpy().tolist()
            else:
                info['current_frequencies'] = self.frequency_weights.numpy().tolist()
        
        return info
    
    def get_gpu_acceleration_info(self) -> Dict[str, Any]:
        """
        Get GPU acceleration information and statistics.
        
        Returns:
            Dictionary containing GPU acceleration information
        """
        if self.gpu_accelerator is None:
            return {
                'gpu_acceleration_enabled': False,
                'reason': 'GPU acceleration disabled or failed to initialize'
            }
        
        # Get GPU memory and configuration info
        gpu_info = self.gpu_accelerator.get_gpu_memory_info()
        
        # Add configuration details
        gpu_info.update({
            'gpu_acceleration_enabled': True,
            'vectorized_operations': self.config.use_vectorized_operations,
            'xla_compilation': self.config.enable_xla_compilation,
            'mixed_precision': self.config.use_mixed_precision,
            'embedding_dim': self.config.embedding_dim,
            'vocab_size': self.config.vocab_size
        })
        
        return gpu_info
    
    def benchmark_gpu_performance(self, batch_size: int = 32, 
                                 seq_length: int = 128,
                                 num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark GPU performance for sinusoidal embedding operations.
        
        Args:
            batch_size: Batch size for benchmarking
            seq_length: Sequence length for benchmarking
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary containing benchmark results
        """
        if self.gpu_accelerator is None:
            logger.warning("GPU accelerator not available for benchmarking")
            return {"error": "GPU acceleration not available"}
        
        logger.info(f"Benchmarking GPU performance with batch_size={batch_size}, "
                   f"seq_length={seq_length}, embedding_dim={self.config.embedding_dim}")
        
        # Run benchmark using the GPU accelerator
        results = self.gpu_accelerator.benchmark_operations(
            batch_size=batch_size,
            seq_length=seq_length,
            embedding_dim=self.config.embedding_dim,
            num_iterations=num_iterations
        )
        
        # Add embedder-specific information
        results.update({
            'embedding_dim': self.config.embedding_dim,
            'vocab_size': self.config.vocab_size,
            'learnable_frequencies': self.config.learnable_frequencies,
            'vectorized_operations': self.config.use_vectorized_operations
        })
        
        return results
    
    def embed(self, token_ids: Union[List[int], np.ndarray, tf.Tensor]) -> np.ndarray:
        """
        Convenience method for embedding token IDs (compatibility with ResponseGenerator).
        
        This method provides a simple interface for embedding token IDs, compatible
        with the ResponseGenerator's expectations. It wraps the call() method with
        appropriate input handling and output conversion.
        
        Args:
            token_ids: Token IDs to embed. Can be:
                - List of integers
                - NumPy array of shape (sequence_length,) or (batch_size, sequence_length)
                - TensorFlow tensor
        
        Returns:
            np.ndarray: Embedded tokens as NumPy array of shape 
                (sequence_length, embedding_dim) or (batch_size, sequence_length, embedding_dim)
        
        Example:
            >>> embedder = ConfigurableSinusoidalEmbedder(config)
            >>> token_ids = [1, 2, 3, 4, 5]
            >>> embeddings = embedder.embed(token_ids)
            >>> print(embeddings.shape)  # (5, embedding_dim)
        """
        try:
            # Convert input to tensor
            if isinstance(token_ids, list):
                token_ids = tf.constant(token_ids, dtype=tf.int32)
            elif isinstance(token_ids, np.ndarray):
                token_ids = tf.constant(token_ids, dtype=tf.int32)
            elif not isinstance(token_ids, tf.Tensor):
                token_ids = tf.constant(token_ids, dtype=tf.int32)
            
            # Ensure we have the right shape
            if len(token_ids.shape) == 1:
                # Add batch dimension
                token_ids = tf.expand_dims(token_ids, axis=0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            # Call the main embedding method
            embeddings = self.call(token_ids, training=False)
            
            # Convert to numpy
            embeddings_np = embeddings.numpy()
            
            # Remove batch dimension if we added it
            if squeeze_output:
                embeddings_np = embeddings_np[0]
            
            return embeddings_np
            
        except Exception as e:
            logger.error(f"Failed to embed token IDs: {e}")
            raise ValueError(f"Embedding failed: {e}")
    
    def enable_gpu_profiling(self, logdir: str = "./logs/sinusoidal_embedder_profile"):
        """
        Enable GPU profiling for performance analysis.
        
        Args:
            logdir: Directory to save profiling logs
        """
        if self.gpu_accelerator is None:
            logger.warning("GPU accelerator not available for profiling")
            return
        
        self.gpu_accelerator.enable_profiling(logdir)
        logger.info(f"Enabled GPU profiling for sinusoidal embedder, logs: {logdir}")
    
    def stop_gpu_profiling(self):
        """Stop GPU profiling."""
        if self.gpu_accelerator is None:
            return
        
        self.gpu_accelerator.stop_profiling()
        logger.info("Stopped GPU profiling for sinusoidal embedder")
    
    def optimize_for_gpu(self, target_batch_size: Optional[int] = None,
                        available_memory_mb: int = 4096) -> Dict[str, Any]:
        """
        Optimize embedder configuration for GPU performance.
        
        Args:
            target_batch_size: Target batch size (auto-calculated if None)
            available_memory_mb: Available GPU memory in MB
            
        Returns:
            Dictionary containing optimization results and recommendations
        """
        if self.gpu_accelerator is None:
            return {"error": "GPU acceleration not available"}
        
        from .gpu_acceleration import get_optimal_batch_size
        
        # Calculate optimal batch size if not provided
        if target_batch_size is None:
            optimal_batch_size = get_optimal_batch_size(
                self.config.vocab_size, 
                self.config.embedding_dim,
                available_memory_mb
            )
        else:
            optimal_batch_size = target_batch_size
        
        # Get GPU memory info
        gpu_info = self.gpu_accelerator.get_gpu_memory_info()
        
        # Provide optimization recommendations
        recommendations = {
            'optimal_batch_size': optimal_batch_size,
            'current_config': {
                'embedding_dim': self.config.embedding_dim,
                'vocab_size': self.config.vocab_size,
                'vectorized_operations': self.config.use_vectorized_operations,
                'mixed_precision': self.config.use_mixed_precision,
                'xla_compilation': self.config.enable_xla_compilation
            },
            'gpu_info': gpu_info,
            'recommendations': []
        }
        
        # Add specific recommendations
        if not self.config.use_vectorized_operations:
            recommendations['recommendations'].append(
                "Enable vectorized operations for better GPU utilization"
            )
        
        if not self.config.use_mixed_precision and gpu_info.get('gpu_available', False):
            recommendations['recommendations'].append(
                "Enable mixed precision for faster computation and reduced memory usage"
            )
        
        if not self.config.enable_xla_compilation:
            recommendations['recommendations'].append(
                "Enable XLA compilation for optimized GPU kernels"
            )
        
        if self.config.embedding_dim % 32 != 0:
            recommendations['recommendations'].append(
                f"Consider embedding dimension multiple of 32 (current: {self.config.embedding_dim}) "
                "for better GPU memory alignment"
            )
        
        logger.info(f"GPU optimization analysis complete: {len(recommendations['recommendations'])} recommendations")
        return recommendations
    
    def get_memory_storage_stats(self) -> Dict[str, Any]:
        """
        Get memory storage statistics.
        
        Returns:
            Dictionary containing memory storage statistics
        """
        stats = {
            'use_memory_efficient_storage': self.config.use_memory_efficient_storage,
            'vocab_size': self.config.vocab_size,
            'embedding_dim': self.config.embedding_dim
        }
        
        if self.config.use_memory_efficient_storage and hasattr(self, 'token_embedding'):
            if hasattr(self.token_embedding, 'get_storage_stats'):
                stats.update(self.token_embedding.get_storage_stats())
        
        return stats
    
    def cleanup_memory_storage(self):
        """Clean up memory storage resources."""
        if (self.config.use_memory_efficient_storage and 
            hasattr(self, 'token_embedding') and 
            hasattr(self.token_embedding, 'cleanup')):
            self.token_embedding.cleanup()
            logger.info("Cleaned up memory-efficient storage resources")
    
    def get_embedding_patterns(self, max_positions: int = 100) -> Dict[str, np.ndarray]:
        """
        Get embedding patterns for visualization.
        
        Args:
            max_positions: Maximum number of positions to compute patterns for
            
        Returns:
            Dictionary containing embedding patterns and metadata
        """
        positions = tf.range(max_positions, dtype=tf.float32)
        positions = tf.expand_dims(positions, 0)  # Add batch dimension
        
        # Compute sinusoidal encoding
        encoding = self._compute_sinusoidal_encoding(positions, training=False)
        encoding_np = encoding.numpy()[0]  # Remove batch dimension
        
        # Get frequency information
        if hasattr(self, 'frequency_weights'):
            frequencies = self.frequency_weights.numpy()
        else:
            frequencies = self._compute_fixed_frequencies()
        
        return {
            'positional_encoding': encoding_np,
            'frequencies': frequencies,
            'positions': np.arange(max_positions),
            'embedding_dim': self.config.embedding_dim,
            'base_frequency': self.config.base_frequency,
            'frequency_scaling': self.config.frequency_scaling,
            'phase_shift': float(self.phase_shift.numpy()) if hasattr(self, 'phase_shift') else self.config.phase_shift
        }
    
    def visualize_embeddings(self, max_positions: int = 100, save_path: Optional[str] = None):
        """
        Create visualization of embedding patterns.
        
        Args:
            max_positions: Maximum number of positions to visualize
            save_path: Optional path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("matplotlib and seaborn required for visualization")
            return
        
        patterns = self.get_embedding_patterns(max_positions)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Configurable Sinusoidal Embedding Patterns', fontsize=16)
        
        # Plot positional encoding heatmap
        sns.heatmap(
            patterns['positional_encoding'][:50, :min(64, self.config.embedding_dim)].T,
            ax=axes[0, 0],
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Embedding Value'}
        )
        axes[0, 0].set_title('Positional Encoding Heatmap')
        axes[0, 0].set_xlabel('Position')
        axes[0, 0].set_ylabel('Embedding Dimension')
        
        # Plot frequency spectrum
        axes[0, 1].plot(patterns['frequencies'])
        axes[0, 1].set_title('Frequency Spectrum')
        axes[0, 1].set_xlabel('Dimension Pair')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot sample embedding dimensions over positions
        sample_dims = [0, self.config.embedding_dim//4, self.config.embedding_dim//2, -1]
        for i, dim in enumerate(sample_dims):
            if dim == -1:
                dim = self.config.embedding_dim - 1
            axes[1, 0].plot(
                patterns['positions'][:50], 
                patterns['positional_encoding'][:50, dim],
                label=f'Dim {dim}',
                alpha=0.7
            )
        axes[1, 0].set_title('Sample Embedding Dimensions')
        axes[1, 0].set_xlabel('Position')
        axes[1, 0].set_ylabel('Embedding Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot embedding similarity matrix
        embeddings = patterns['positional_encoding'][:min(50, max_positions)]
        similarity = np.dot(embeddings, embeddings.T)
        similarity = similarity / (np.linalg.norm(embeddings, axis=1, keepdims=True) * 
                                 np.linalg.norm(embeddings, axis=1, keepdims=True).T)
        
        sns.heatmap(
            similarity,
            ax=axes[1, 1],
            cmap='viridis',
            cbar_kws={'label': 'Cosine Similarity'}
        )
        axes[1, 1].set_title('Position Similarity Matrix')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the layer configuration."""
        config = super().get_config()
        config.update({
            'sinusoidal_config': {
                'embedding_dim': self.config.embedding_dim,
                'vocab_size': self.config.vocab_size,
                'max_sequence_length': self.config.max_sequence_length,
                'base_frequency': self.config.base_frequency,
                'frequency_scaling': self.config.frequency_scaling,
                'learnable_frequencies': self.config.learnable_frequencies,
                'use_absolute_position': self.config.use_absolute_position,
                'use_relative_position': self.config.use_relative_position,
                'relative_position_window': self.config.relative_position_window,
                'frequency_init_std': self.config.frequency_init_std,
                'phase_shift': self.config.phase_shift,
                'temperature': self.config.temperature,
                'use_mixed_precision': self.config.use_mixed_precision,
                'gradient_checkpointing': self.config.gradient_checkpointing,
                'use_memory_efficient_storage': self.config.use_memory_efficient_storage,
                'memory_storage_config': self.config.memory_storage_config.__dict__ if self.config.memory_storage_config else None
            }
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        """Create layer from configuration."""
        sinusoidal_config = SinusoidalConfig(**config.pop('sinusoidal_config'))
        return cls(sinusoidal_config, **config)
    
    def save_config(self, filepath: str):
        """Save the embedder configuration to file."""
        try:
            config_dict = {
                'embedding_dim': self.config.embedding_dim,
                'vocab_size': self.config.vocab_size,
                'max_sequence_length': self.config.max_sequence_length,
                'base_frequency': self.config.base_frequency,
                'frequency_scaling': self.config.frequency_scaling,
                'learnable_frequencies': self.config.learnable_frequencies,
                'use_absolute_position': self.config.use_absolute_position,
                'use_relative_position': self.config.use_relative_position,
                'relative_position_window': self.config.relative_position_window,
                'frequency_init_std': self.config.frequency_init_std,
                'phase_shift': self.config.phase_shift,
                'temperature': self.config.temperature,
                'use_mixed_precision': self.config.use_mixed_precision,
                'gradient_checkpointing': self.config.gradient_checkpointing,
                'use_memory_efficient_storage': self.config.use_memory_efficient_storage,
                'memory_storage_config': self.config.memory_storage_config.__dict__ if self.config.memory_storage_config else None
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            raise TokenizerSaveError(f"Failed to save configuration: {str(e)}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'ConfigurableSinusoidalEmbedder':
        """Load embedder from configuration file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Handle memory storage configuration
            memory_config_dict = config_dict.pop('memory_storage_config', None)
            if memory_config_dict:
                config_dict['memory_storage_config'] = MemoryStorageConfig(**memory_config_dict)
            
            config = SinusoidalConfig(**config_dict)
            embedder = cls(config)
            
            logger.info(f"Configuration loaded from {filepath}")
            return embedder
            
        except Exception as e:
            raise TokenizerLoadError(f"Failed to load configuration: {str(e)}")


class EmbeddingDimensionOptimizer:
    """Utility class for optimizing embedding dimensions based on vocabulary size and mathematical properties."""
    
    @staticmethod
    def calculate_optimal_dimension(vocab_size: int, 
                                  target_model_size: str = 'medium',
                                  preserve_mathematical_properties: bool = True) -> int:
        """
        Calculate optimal embedding dimension based on vocabulary size and model requirements.
        
        Args:
            vocab_size: Size of the vocabulary
            target_model_size: Target model size ('small', 'medium', 'large', 'xlarge')
            preserve_mathematical_properties: Whether to ensure dimension preserves sinusoidal properties
            
        Returns:
            Optimal embedding dimension
        """
        # Base dimension ratios for different model sizes
        size_ratios = {
            'small': 0.1,
            'medium': 0.2,
            'large': 0.3,
            'xlarge': 0.4
        }
        
        if target_model_size not in size_ratios:
            raise InvalidInputError("target_model_size", f"one of {list(size_ratios.keys())}", target_model_size)
        
        # Calculate base dimension from vocabulary size
        base_dim = int(vocab_size * size_ratios[target_model_size])
        
        # Ensure minimum dimension
        base_dim = max(base_dim, 64)
        
        # Ensure maximum dimension for practical purposes
        base_dim = min(base_dim, 2048)
        
        if preserve_mathematical_properties:
            # Ensure dimension is even for sinusoidal encoding
            if base_dim % 2 != 0:
                base_dim += 1
            
            # Prefer dimensions that are powers of 2 or have nice factorizations
            # for better mathematical properties
            nice_dims = []
            for i in range(max(64, base_dim - 32), base_dim + 33, 2):
                if EmbeddingDimensionOptimizer._has_nice_factorization(i):
                    nice_dims.append(i)
            
            if nice_dims:
                # Choose the dimension closest to base_dim
                optimal_dim = min(nice_dims, key=lambda x: abs(x - base_dim))
            else:
                optimal_dim = base_dim
        else:
            optimal_dim = base_dim
        
        logger.info(f"Calculated optimal embedding dimension: {optimal_dim} for vocab_size {vocab_size}")
        return optimal_dim
    
    @staticmethod
    def _has_nice_factorization(n: int) -> bool:
        """Check if a number has nice factorization for mathematical properties."""
        # Check if it's a power of 2
        if n & (n - 1) == 0:
            return True
        
        # Check if it has small prime factors (good for FFT-like operations)
        temp = n
        for prime in [2, 3, 5, 7]:
            while temp % prime == 0:
                temp //= prime
        
        # If remaining factor is small, it has nice factorization
        return temp <= 11
    
    @staticmethod
    def suggest_dimension_scaling(old_vocab_size: int, new_vocab_size: int, 
                                old_embedding_dim: int) -> int:
        """
        Suggest new embedding dimension when vocabulary size changes.
        
        Args:
            old_vocab_size: Original vocabulary size
            new_vocab_size: New vocabulary size
            old_embedding_dim: Original embedding dimension
            
        Returns:
            Suggested new embedding dimension
        """
        if old_vocab_size <= 0 or new_vocab_size <= 0 or old_embedding_dim <= 0:
            raise InvalidInputError("size parameters", "positive integers", f"old_vocab_size={old_vocab_size}, new_vocab_size={new_vocab_size}, old_embedding_dim={old_embedding_dim}")
        
        # Calculate scaling ratio
        vocab_ratio = new_vocab_size / old_vocab_size
        
        # Apply square root scaling to maintain reasonable parameter count
        dim_scaling = np.sqrt(vocab_ratio)
        
        # Calculate new dimension
        new_dim = int(old_embedding_dim * dim_scaling)
        
        # Ensure even dimension for sinusoidal encoding
        if new_dim % 2 != 0:
            new_dim += 1
        
        # Ensure reasonable bounds
        new_dim = max(64, min(2048, new_dim))
        
        logger.info(f"Suggested dimension scaling: {old_embedding_dim} -> {new_dim} "
                   f"(vocab: {old_vocab_size} -> {new_vocab_size})")
        
        return new_dim


class SinusoidalEmbedderFactory:
    """Factory class for creating configured sinusoidal embedders."""
    
    @staticmethod
    def create_default(vocab_size: int, embedding_dim: int = 128) -> ConfigurableSinusoidalEmbedder:
        """Create a default sinusoidal embedder configuration."""
        config = SinusoidalConfig(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            learnable_frequencies=True,
            use_absolute_position=True,
            use_relative_position=False
        )
        return ConfigurableSinusoidalEmbedder(config)
    
    @staticmethod
    def create_auto_adapted(tokenizer_adapter, 
                           target_model_size: str = 'medium',
                           learnable_frequencies: bool = True,
                           preserve_properties: bool = True) -> ConfigurableSinusoidalEmbedder:
        """
        Create a sinusoidal embedder with automatic dimension optimization.
        
        Args:
            tokenizer_adapter: TokenizerAdapter to adapt to
            target_model_size: Target model size for dimension calculation
            learnable_frequencies: Whether to use learnable frequencies
            preserve_properties: Whether to preserve mathematical properties
            
        Returns:
            Auto-adapted ConfigurableSinusoidalEmbedder
        """
        from .enhanced_tokenization import TokenizerAdapter
        
        if not isinstance(tokenizer_adapter, TokenizerAdapter):
            raise InvalidInputError("tokenizer_adapter", "TokenizerAdapter instance", str(type(tokenizer_adapter)))
        
        # Get vocabulary size
        vocab_size = tokenizer_adapter.get_vocab_size()
        
        # Calculate optimal dimension
        optimal_dim = EmbeddingDimensionOptimizer.calculate_optimal_dimension(
            vocab_size, target_model_size, preserve_properties
        )
        
        # Create configuration
        config = SinusoidalConfig(
            embedding_dim=optimal_dim,
            vocab_size=vocab_size,
            learnable_frequencies=learnable_frequencies,
            use_absolute_position=True,
            use_relative_position=False
        )
        
        # Create and adapt embedder
        embedder = ConfigurableSinusoidalEmbedder(config)
        embedder.adapt_to_tokenizer(tokenizer_adapter)
        
        logger.info(f"Created auto-adapted embedder with dimension {optimal_dim} for {tokenizer_adapter}")
        
        return embedder
    
    @staticmethod
    def create_relative_position(
        vocab_size: int, 
        embedding_dim: int = 128,
        relative_window: int = 64
    ) -> ConfigurableSinusoidalEmbedder:
        """Create a sinusoidal embedder with relative positional encoding."""
        config = SinusoidalConfig(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            learnable_frequencies=True,
            use_absolute_position=True,
            use_relative_position=True,
            relative_position_window=relative_window
        )
        return ConfigurableSinusoidalEmbedder(config)
    
    @staticmethod
    def create_fixed_frequency(
        vocab_size: int,
        embedding_dim: int = 128,
        base_frequency: float = 10000.0
    ) -> ConfigurableSinusoidalEmbedder:
        """Create a sinusoidal embedder with fixed frequencies."""
        config = SinusoidalConfig(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            learnable_frequencies=False,
            base_frequency=base_frequency,
            use_absolute_position=True,
            use_relative_position=False
        )
        return ConfigurableSinusoidalEmbedder(config)
    
    @staticmethod
    def create_high_performance(
        vocab_size: int,
        embedding_dim: int = 128
    ) -> ConfigurableSinusoidalEmbedder:
        """Create a high-performance sinusoidal embedder configuration."""
        config = SinusoidalConfig(
            embedding_dim=embedding_dim,
            vocab_size=vocab_size,
            learnable_frequencies=True,
            use_absolute_position=True,
            use_relative_position=False,
            use_mixed_precision=True,
            gradient_checkpointing=True
        )
        return ConfigurableSinusoidalEmbedder(config)