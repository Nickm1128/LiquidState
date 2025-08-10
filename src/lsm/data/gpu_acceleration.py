#!/usr/bin/env python3
"""
GPU acceleration utilities for LSM tokenizer and embedder components.

This module provides GPU optimization utilities including CUDA/GPU processing,
vectorized operations, and mixed precision support for faster computation.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..utils.lsm_exceptions import (
    TokenizerError, InvalidInputError
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUConfig:
    """Configuration for GPU acceleration."""
    
    # GPU device configuration
    enable_gpu: bool = True
    gpu_device: Optional[str] = None  # e.g., '/GPU:0', auto-detect if None
    allow_memory_growth: bool = True
    memory_limit: Optional[int] = None  # Memory limit in MB
    
    # Mixed precision configuration
    enable_mixed_precision: bool = True
    mixed_precision_policy: str = "mixed_float16"  # "mixed_float16" or "mixed_bfloat16"
    loss_scale: Optional[float] = None  # Auto if None
    
    # Vectorization configuration
    enable_vectorization: bool = True
    batch_vectorization_threshold: int = 32  # Minimum batch size for vectorization
    parallel_iterations: int = 10  # For tf.map_fn
    
    # XLA compilation
    enable_xla: bool = True
    xla_auto_jit: bool = True
    
    # Performance monitoring
    enable_profiling: bool = False
    profile_batch_range: Tuple[int, int] = (100, 200)


class GPUAccelerator:
    """
    GPU acceleration manager for LSM components.
    
    This class handles GPU device configuration, mixed precision setup,
    and provides optimized computation functions for sinusoidal embeddings.
    """
    
    def __init__(self, config: GPUConfig):
        """
        Initialize GPU accelerator.
        
        Args:
            config: GPUConfig object with acceleration settings
        """
        self.config = config
        self._gpu_available = False
        self._mixed_precision_enabled = False
        self._xla_enabled = False
        
        # Initialize GPU configuration
        self._setup_gpu()
        self._setup_mixed_precision()
        self._setup_xla()
        
        logger.info(f"Initialized GPU accelerator: GPU={self._gpu_available}, "
                   f"Mixed Precision={self._mixed_precision_enabled}, XLA={self._xla_enabled}")
    
    def _setup_gpu(self):
        """Setup GPU device configuration."""
        try:
            # Check GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if not gpus:
                logger.warning("No GPUs found, falling back to CPU")
                self.config.enable_gpu = False
                return
            
            if not self.config.enable_gpu:
                logger.info("GPU acceleration disabled by configuration")
                return
            
            # Configure GPU memory growth
            for gpu in gpus:
                try:
                    if self.config.allow_memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Set memory limit if specified
                    if self.config.memory_limit is not None:
                        tf.config.experimental.set_memory_limit(
                            gpu, self.config.memory_limit
                        )
                    
                    logger.info(f"Configured GPU: {gpu.name}")
                    
                except RuntimeError as e:
                    logger.warning(f"Failed to configure GPU {gpu.name}: {e}")
            
            self._gpu_available = True
            
            # Set default device if specified
            if self.config.gpu_device:
                self._default_device = self.config.gpu_device
            else:
                self._default_device = '/GPU:0' if gpus else '/CPU:0'
            
            logger.info(f"Using device: {self._default_device}")
            
        except Exception as e:
            logger.error(f"Failed to setup GPU: {e}")
            self.config.enable_gpu = False
            self._gpu_available = False
    
    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if not self.config.enable_mixed_precision:
            logger.info("Mixed precision disabled by configuration")
            return
        
        if not self._gpu_available:
            logger.warning("Mixed precision requires GPU, disabling")
            self.config.enable_mixed_precision = False
            return
        
        try:
            # Set mixed precision policy
            policy = keras.mixed_precision.Policy(self.config.mixed_precision_policy)
            keras.mixed_precision.set_global_policy(policy)
            
            # Configure loss scaling if specified
            if self.config.loss_scale is not None:
                self._loss_scale = keras.mixed_precision.LossScaleOptimizer(
                    keras.optimizers.Adam(), dynamic=False, 
                    initial_loss_scale=self.config.loss_scale
                )
            else:
                self._loss_scale = keras.mixed_precision.LossScaleOptimizer(
                    keras.optimizers.Adam(), dynamic=True
                )
            
            self._mixed_precision_enabled = True
            logger.info(f"Enabled mixed precision: {self.config.mixed_precision_policy}")
            
        except Exception as e:
            logger.error(f"Failed to setup mixed precision: {e}")
            self.config.enable_mixed_precision = False
            self._mixed_precision_enabled = False
    
    def _setup_xla(self):
        """Setup XLA compilation."""
        if not self.config.enable_xla:
            logger.info("XLA compilation disabled by configuration")
            return
        
        try:
            # Enable XLA auto-jit if configured
            if self.config.xla_auto_jit:
                tf.config.optimizer.set_jit(True)
            
            self._xla_enabled = True
            logger.info("Enabled XLA compilation")
            
        except Exception as e:
            logger.error(f"Failed to setup XLA: {e}")
            self.config.enable_xla = False
            self._xla_enabled = False
    
    def get_device_context(self):
        """Get device context for GPU operations."""
        if self._gpu_available and hasattr(self, '_default_device'):
            return tf.device(self._default_device)
        else:
            return tf.device('/CPU:0')
    
    def get_optimizer(self, base_optimizer: keras.optimizers.Optimizer = None):
        """
        Get optimizer with mixed precision support.
        
        Args:
            base_optimizer: Base optimizer to wrap
            
        Returns:
            Optimizer with mixed precision support if enabled
        """
        if base_optimizer is None:
            base_optimizer = keras.optimizers.Adam(learning_rate=0.001)
        
        if self._mixed_precision_enabled:
            return keras.mixed_precision.LossScaleOptimizer(base_optimizer)
        else:
            return base_optimizer
    
    @tf.function(experimental_relax_shapes=True)
    def vectorized_sinusoidal_encoding(self, positions: tf.Tensor, 
                                     frequencies: tf.Tensor, 
                                     phase_shift: tf.Tensor = 0.0) -> tf.Tensor:
        """
        Vectorized sinusoidal encoding computation optimized for GPU.
        
        Args:
            positions: Position tensor of shape (batch_size, seq_length)
            frequencies: Frequency tensor of shape (embedding_dim // 2,)
            phase_shift: Phase shift scalar
            
        Returns:
            Sinusoidal encoding tensor
        """
        with self.get_device_context():
            # Expand dimensions for broadcasting
            positions = tf.expand_dims(positions, -1)  # (batch_size, seq_length, 1)
            frequencies = tf.expand_dims(frequencies, 0)  # (1, embedding_dim // 2)
            frequencies = tf.expand_dims(frequencies, 0)  # (1, 1, embedding_dim // 2)
            
            # Compute angles with vectorized operations
            angles = positions * frequencies + phase_shift
            
            # Compute sine and cosine in parallel
            sin_encoding = tf.sin(angles)
            cos_encoding = tf.cos(angles)
            
            # Interleave sine and cosine efficiently
            encoding = tf.stack([sin_encoding, cos_encoding], axis=-1)
            
            # Reshape to final embedding dimension
            batch_size = tf.shape(positions)[0]
            seq_length = tf.shape(positions)[1]
            embedding_dim = tf.shape(frequencies)[-1] * 2
            
            encoding = tf.reshape(encoding, [batch_size, seq_length, embedding_dim])
            
            return encoding
    
    @tf.function(experimental_relax_shapes=True)
    def batch_embedding_lookup(self, embedding_matrix: tf.Tensor, 
                              indices: tf.Tensor) -> tf.Tensor:
        """
        Optimized batch embedding lookup for GPU.
        
        Args:
            embedding_matrix: Embedding matrix of shape (vocab_size, embedding_dim)
            indices: Token indices of shape (batch_size, seq_length)
            
        Returns:
            Embedded tokens
        """
        with self.get_device_context():
            # Use tf.nn.embedding_lookup for optimized GPU performance
            embeddings = tf.nn.embedding_lookup(embedding_matrix, indices)
            return embeddings
    
    @tf.function(experimental_relax_shapes=True)
    def parallel_frequency_computation(self, base_frequency: tf.Tensor,
                                     embedding_dim: int,
                                     frequency_scaling: tf.Tensor = 1.0) -> tf.Tensor:
        """
        Parallel computation of frequency values for sinusoidal encoding.
        
        Args:
            base_frequency: Base frequency value
            embedding_dim: Embedding dimension
            frequency_scaling: Frequency scaling factor
            
        Returns:
            Computed frequency values
        """
        with self.get_device_context():
            dim_pairs = embedding_dim // 2
            
            # Vectorized frequency computation
            i_values = tf.range(dim_pairs, dtype=tf.float32)
            exponents = (2.0 * i_values) / tf.cast(embedding_dim, tf.float32)
            
            frequencies = 1.0 / tf.pow(base_frequency, exponents)
            frequencies = frequencies * frequency_scaling
            
            return frequencies
    
    @tf.function(experimental_relax_shapes=True)
    def optimized_attention_weights(self, query: tf.Tensor, 
                                   key: tf.Tensor,
                                   temperature: tf.Tensor = 1.0) -> tf.Tensor:
        """
        Optimized attention weight computation for sinusoidal embeddings.
        
        Args:
            query: Query tensor
            key: Key tensor  
            temperature: Temperature scaling
            
        Returns:
            Attention weights
        """
        with self.get_device_context():
            # Compute attention scores with optimized matrix multiplication
            scores = tf.linalg.matmul(query, key, transpose_b=True)
            scores = scores / temperature
            
            # Apply softmax with numerical stability
            attention_weights = tf.nn.softmax(scores, axis=-1)
            
            return attention_weights
    
    def enable_profiling(self, logdir: str = "./logs/gpu_profile"):
        """
        Enable GPU profiling for performance analysis.
        
        Args:
            logdir: Directory to save profiling logs
        """
        if not self.config.enable_profiling:
            return
        
        try:
            # Start profiler
            tf.profiler.experimental.start(logdir)
            logger.info(f"Started GPU profiling, logs will be saved to {logdir}")
            
        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
    
    def stop_profiling(self):
        """Stop GPU profiling."""
        if not self.config.enable_profiling:
            return
        
        try:
            tf.profiler.experimental.stop()
            logger.info("Stopped GPU profiling")
            
        except Exception as e:
            logger.error(f"Failed to stop profiling: {e}")
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get GPU memory usage information.
        
        Returns:
            Dictionary containing GPU memory statistics
        """
        if not self._gpu_available:
            return {"gpu_available": False}
        
        try:
            # Get GPU memory info
            gpus = tf.config.experimental.list_physical_devices('GPU')
            memory_info = {}
            
            for i, gpu in enumerate(gpus):
                try:
                    memory_info[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "memory_growth_enabled": self.config.allow_memory_growth,
                        "memory_limit": self.config.memory_limit
                    }
                except Exception as e:
                    logger.warning(f"Failed to get memory info for GPU {i}: {e}")
            
            return {
                "gpu_available": True,
                "num_gpus": len(gpus),
                "mixed_precision_enabled": self._mixed_precision_enabled,
                "xla_enabled": self._xla_enabled,
                "memory_info": memory_info
            }
            
        except Exception as e:
            logger.error(f"Failed to get GPU memory info: {e}")
            return {"gpu_available": False, "error": str(e)}
    
    def benchmark_operations(self, batch_size: int = 32, 
                           seq_length: int = 128,
                           embedding_dim: int = 256,
                           num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark GPU operations for performance analysis.
        
        Args:
            batch_size: Batch size for benchmarking
            seq_length: Sequence length for benchmarking
            embedding_dim: Embedding dimension for benchmarking
            num_iterations: Number of iterations to run
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Running GPU benchmark: batch_size={batch_size}, "
                   f"seq_length={seq_length}, embedding_dim={embedding_dim}")
        
        # Create test data
        positions = tf.random.uniform((batch_size, seq_length), 0, seq_length, dtype=tf.float32)
        frequencies = tf.random.uniform((embedding_dim // 2,), 0.001, 0.1, dtype=tf.float32)
        embedding_matrix = tf.random.normal((10000, embedding_dim), dtype=tf.float32)
        indices = tf.random.uniform((batch_size, seq_length), 0, 10000, dtype=tf.int32)
        
        results = {}
        
        # Benchmark sinusoidal encoding
        start_time = tf.timestamp()
        for _ in range(num_iterations):
            _ = self.vectorized_sinusoidal_encoding(positions, frequencies)
        end_time = tf.timestamp()
        results["sinusoidal_encoding_ms"] = float((end_time - start_time) * 1000 / num_iterations)
        
        # Benchmark embedding lookup
        start_time = tf.timestamp()
        for _ in range(num_iterations):
            _ = self.batch_embedding_lookup(embedding_matrix, indices)
        end_time = tf.timestamp()
        results["embedding_lookup_ms"] = float((end_time - start_time) * 1000 / num_iterations)
        
        # Benchmark frequency computation
        base_freq = tf.constant(10000.0)
        start_time = tf.timestamp()
        for _ in range(num_iterations):
            _ = self.parallel_frequency_computation(base_freq, embedding_dim)
        end_time = tf.timestamp()
        results["frequency_computation_ms"] = float((end_time - start_time) * 1000 / num_iterations)
        
        logger.info(f"Benchmark results: {results}")
        return results


def create_gpu_accelerator(enable_gpu: bool = True,
                          enable_mixed_precision: bool = True,
                          enable_xla: bool = True) -> GPUAccelerator:
    """
    Create a GPU accelerator with default configuration.
    
    Args:
        enable_gpu: Whether to enable GPU acceleration
        enable_mixed_precision: Whether to enable mixed precision
        enable_xla: Whether to enable XLA compilation
        
    Returns:
        Configured GPUAccelerator instance
    """
    config = GPUConfig(
        enable_gpu=enable_gpu,
        enable_mixed_precision=enable_mixed_precision,
        enable_xla=enable_xla
    )
    
    return GPUAccelerator(config)


def get_optimal_batch_size(vocab_size: int, embedding_dim: int, 
                          available_memory_mb: int = 4096) -> int:
    """
    Calculate optimal batch size for GPU processing based on memory constraints.
    
    Args:
        vocab_size: Vocabulary size
        embedding_dim: Embedding dimension
        available_memory_mb: Available GPU memory in MB
        
    Returns:
        Optimal batch size
    """
    # Validate inputs
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    if embedding_dim <= 0:
        raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
    if available_memory_mb <= 0:
        raise ValueError(f"available_memory_mb must be positive, got {available_memory_mb}")
    
    # Estimate memory usage per sample (in bytes)
    # Token embeddings + positional encodings + gradients + overhead
    memory_per_sample = (
        embedding_dim * 4 +  # Token embedding (float32)
        embedding_dim * 4 +  # Positional encoding (float32)
        embedding_dim * 4 * 2 +  # Gradients (float32, forward + backward)
        1024  # Overhead
    )
    
    # Convert available memory to bytes
    available_memory_bytes = available_memory_mb * 1024 * 1024
    
    # Calculate optimal batch size with safety margin
    optimal_batch_size = int(available_memory_bytes * 0.8 / memory_per_sample)
    
    # Ensure minimum and maximum bounds
    optimal_batch_size = max(1, min(optimal_batch_size, 1024))
    
    logger.info(f"Calculated optimal batch size: {optimal_batch_size} "
               f"for vocab_size={vocab_size}, embedding_dim={embedding_dim}")
    
    return optimal_batch_size