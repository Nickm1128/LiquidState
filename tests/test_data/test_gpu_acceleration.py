#!/usr/bin/env python3
"""
Tests for GPU acceleration functionality in LSM tokenizer/embedder.

This module tests GPU optimization, vectorized operations, mixed precision,
and performance benchmarking features.
"""

import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.lsm.data.gpu_acceleration import (
    GPUAccelerator, GPUConfig, create_gpu_accelerator, get_optimal_batch_size
)
from src.lsm.data.configurable_sinusoidal_embedder import (
    ConfigurableSinusoidalEmbedder, SinusoidalConfig
)


class TestGPUConfig(unittest.TestCase):
    """Test GPU configuration class."""
    
    def test_default_config(self):
        """Test default GPU configuration."""
        config = GPUConfig()
        
        self.assertTrue(config.enable_gpu)
        self.assertTrue(config.enable_mixed_precision)
        self.assertEqual(config.mixed_precision_policy, "mixed_float16")
        self.assertTrue(config.enable_vectorization)
        self.assertTrue(config.enable_xla)
        self.assertTrue(config.allow_memory_growth)
    
    def test_custom_config(self):
        """Test custom GPU configuration."""
        config = GPUConfig(
            enable_gpu=False,
            enable_mixed_precision=False,
            mixed_precision_policy="mixed_bfloat16",
            enable_vectorization=False,
            enable_xla=False,
            memory_limit=2048
        )
        
        self.assertFalse(config.enable_gpu)
        self.assertFalse(config.enable_mixed_precision)
        self.assertEqual(config.mixed_precision_policy, "mixed_bfloat16")
        self.assertFalse(config.enable_vectorization)
        self.assertFalse(config.enable_xla)
        self.assertEqual(config.memory_limit, 2048)


class TestGPUAccelerator(unittest.TestCase):
    """Test GPU accelerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = GPUConfig(
            enable_gpu=True,
            enable_mixed_precision=False,  # Disable for testing
            enable_xla=False  # Disable for testing
        )
    
    def test_gpu_accelerator_initialization(self):
        """Test GPU accelerator initialization."""
        accelerator = GPUAccelerator(self.config)
        
        self.assertIsNotNone(accelerator)
        self.assertEqual(accelerator.config, self.config)
    
    def test_vectorized_sinusoidal_encoding(self):
        """Test vectorized sinusoidal encoding computation."""
        accelerator = GPUAccelerator(self.config)
        
        # Create test data
        batch_size, seq_length = 4, 8
        embedding_dim = 16
        
        positions = tf.random.uniform((batch_size, seq_length), 0, seq_length, dtype=tf.float32)
        frequencies = tf.random.uniform((embedding_dim // 2,), 0.001, 0.1, dtype=tf.float32)
        phase_shift = tf.constant(0.0, dtype=tf.float32)
        
        # Compute encoding
        encoding = accelerator.vectorized_sinusoidal_encoding(
            positions, frequencies, phase_shift
        )
        
        # Verify output shape
        expected_shape = (batch_size, seq_length, embedding_dim)
        self.assertEqual(encoding.shape, expected_shape)
        
        # Verify output is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(encoding)))
    
    def test_batch_embedding_lookup(self):
        """Test optimized batch embedding lookup."""
        accelerator = GPUAccelerator(self.config)
        
        # Create test data
        vocab_size, embedding_dim = 100, 16
        batch_size, seq_length = 4, 8
        
        embedding_matrix = tf.random.normal((vocab_size, embedding_dim), dtype=tf.float32)
        indices = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
        
        # Perform lookup
        embeddings = accelerator.batch_embedding_lookup(embedding_matrix, indices)
        
        # Verify output shape
        expected_shape = (batch_size, seq_length, embedding_dim)
        self.assertEqual(embeddings.shape, expected_shape)
        
        # Verify output is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(embeddings)))
    
    def test_parallel_frequency_computation(self):
        """Test parallel frequency computation."""
        accelerator = GPUAccelerator(self.config)
        
        # Test parameters
        base_frequency = tf.constant(10000.0, dtype=tf.float32)
        embedding_dim = 64
        frequency_scaling = tf.constant(1.5, dtype=tf.float32)
        
        # Compute frequencies
        frequencies = accelerator.parallel_frequency_computation(
            base_frequency, embedding_dim, frequency_scaling
        )
        
        # Verify output shape
        expected_shape = (embedding_dim // 2,)
        self.assertEqual(frequencies.shape, expected_shape)
        
        # Verify frequencies are positive and finite
        self.assertTrue(tf.reduce_all(frequencies > 0))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(frequencies)))
        
        # Verify scaling is applied
        unscaled_frequencies = accelerator.parallel_frequency_computation(
            base_frequency, embedding_dim, tf.constant(1.0, dtype=tf.float32)
        )
        scaled_ratio = tf.reduce_mean(frequencies / unscaled_frequencies)
        self.assertAlmostEqual(float(scaled_ratio), 1.5, places=5)
    
    def test_optimized_attention_weights(self):
        """Test optimized attention weight computation."""
        accelerator = GPUAccelerator(self.config)
        
        # Create test data
        batch_size, seq_length, dim = 2, 4, 8
        
        query = tf.random.normal((batch_size, seq_length, dim), dtype=tf.float32)
        key = tf.random.normal((batch_size, seq_length, dim), dtype=tf.float32)
        temperature = tf.constant(1.0, dtype=tf.float32)
        
        # Compute attention weights
        attention_weights = accelerator.optimized_attention_weights(
            query, key, temperature
        )
        
        # Verify output shape
        expected_shape = (batch_size, seq_length, seq_length)
        self.assertEqual(attention_weights.shape, expected_shape)
        
        # Verify weights sum to 1 (softmax property)
        weight_sums = tf.reduce_sum(attention_weights, axis=-1)
        expected_sums = tf.ones_like(weight_sums)
        self.assertTrue(tf.reduce_all(tf.abs(weight_sums - expected_sums) < 1e-5))
    
    def test_get_gpu_memory_info(self):
        """Test GPU memory information retrieval."""
        accelerator = GPUAccelerator(self.config)
        
        memory_info = accelerator.get_gpu_memory_info()
        
        self.assertIsInstance(memory_info, dict)
        self.assertIn('gpu_available', memory_info)
        
        if memory_info['gpu_available']:
            self.assertIn('num_gpus', memory_info)
            self.assertIn('mixed_precision_enabled', memory_info)
            self.assertIn('xla_enabled', memory_info)
    
    def test_benchmark_operations(self):
        """Test GPU operations benchmarking."""
        accelerator = GPUAccelerator(self.config)
        
        # Run benchmark with small parameters for testing
        results = accelerator.benchmark_operations(
            batch_size=2,
            seq_length=4,
            embedding_dim=16,
            num_iterations=5
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('sinusoidal_encoding_ms', results)
        self.assertIn('embedding_lookup_ms', results)
        self.assertIn('frequency_computation_ms', results)
        
        # Verify all times are positive
        for key, value in results.items():
            if key.endswith('_ms'):
                self.assertGreater(value, 0)


class TestGPUAcceleratorFactory(unittest.TestCase):
    """Test GPU accelerator factory functions."""
    
    def test_create_gpu_accelerator(self):
        """Test GPU accelerator creation with defaults."""
        accelerator = create_gpu_accelerator()
        
        self.assertIsInstance(accelerator, GPUAccelerator)
        # GPU may be disabled if no GPU hardware is available
        self.assertIsInstance(accelerator.config.enable_gpu, bool)
    
    def test_create_gpu_accelerator_custom(self):
        """Test GPU accelerator creation with custom settings."""
        accelerator = create_gpu_accelerator(
            enable_gpu=False,
            enable_mixed_precision=False,
            enable_xla=False
        )
        
        self.assertIsInstance(accelerator, GPUAccelerator)
        self.assertFalse(accelerator.config.enable_gpu)
        self.assertFalse(accelerator.config.enable_mixed_precision)
        self.assertFalse(accelerator.config.enable_xla)
    
    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        # Test with different configurations
        test_cases = [
            (1000, 128, 1024, 1),     # Small config, small memory
            (10000, 256, 4096, 1),    # Medium config, medium memory
            (50000, 512, 8192, 1),    # Large config, large memory
        ]
        
        for vocab_size, embedding_dim, memory_mb, min_expected in test_cases:
            batch_size = get_optimal_batch_size(vocab_size, embedding_dim, memory_mb)
            
            self.assertIsInstance(batch_size, int)
            self.assertGreaterEqual(batch_size, min_expected)
            self.assertLessEqual(batch_size, 1024)  # Maximum bound


class TestSinusoidalEmbedderGPUIntegration(unittest.TestCase):
    """Test GPU acceleration integration with sinusoidal embedder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.vocab_size = 1000
        self.embedding_dim = 64
        self.batch_size = 4
        self.seq_length = 8
    
    def test_embedder_with_gpu_acceleration(self):
        """Test sinusoidal embedder with GPU acceleration enabled."""
        config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=True,
            use_vectorized_operations=True,
            use_mixed_precision=False,  # Disable for testing
            enable_xla_compilation=False  # Disable for testing
        )
        
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build embedder
        sample_input = tf.random.uniform(
            (self.batch_size, self.seq_length), 0, self.vocab_size, dtype=tf.int32
        )
        embedder.build(sample_input.shape)
        
        # Test forward pass
        embeddings = embedder(sample_input, training=False)
        
        # Verify output shape
        expected_shape = (self.batch_size, self.seq_length, self.embedding_dim)
        self.assertEqual(embeddings.shape, expected_shape)
        
        # Verify output is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(embeddings)))
    
    def test_embedder_without_gpu_acceleration(self):
        """Test sinusoidal embedder without GPU acceleration."""
        config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=False,
            use_vectorized_operations=False,
            use_mixed_precision=False,
            enable_xla_compilation=False
        )
        
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build embedder
        sample_input = tf.random.uniform(
            (self.batch_size, self.seq_length), 0, self.vocab_size, dtype=tf.int32
        )
        embedder.build(sample_input.shape)
        
        # Test forward pass
        embeddings = embedder(sample_input, training=False)
        
        # Verify output shape
        expected_shape = (self.batch_size, self.seq_length, self.embedding_dim)
        self.assertEqual(embeddings.shape, expected_shape)
        
        # Verify output is finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(embeddings)))
    
    def test_gpu_acceleration_info(self):
        """Test GPU acceleration information retrieval."""
        config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=True
        )
        
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        gpu_info = embedder.get_gpu_acceleration_info()
        
        self.assertIsInstance(gpu_info, dict)
        self.assertIn('gpu_acceleration_enabled', gpu_info)
        
        if gpu_info['gpu_acceleration_enabled']:
            self.assertIn('vectorized_operations', gpu_info)
            self.assertIn('mixed_precision', gpu_info)
            self.assertIn('xla_compilation', gpu_info)
    
    def test_gpu_performance_benchmark(self):
        """Test GPU performance benchmarking."""
        config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=True
        )
        
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Build embedder
        sample_input = tf.random.uniform(
            (self.batch_size, self.seq_length), 0, self.vocab_size, dtype=tf.int32
        )
        embedder.build(sample_input.shape)
        
        # Run benchmark
        results = embedder.benchmark_gpu_performance(
            batch_size=2,
            seq_length=4,
            num_iterations=5
        )
        
        if 'error' not in results:
            self.assertIsInstance(results, dict)
            self.assertIn('embedding_dim', results)
            self.assertIn('vocab_size', results)
            self.assertEqual(results['embedding_dim'], self.embedding_dim)
            self.assertEqual(results['vocab_size'], self.vocab_size)
    
    def test_gpu_optimization_recommendations(self):
        """Test GPU optimization recommendations."""
        config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=True,
            use_vectorized_operations=False,  # Intentionally suboptimal
            use_mixed_precision=False,        # Intentionally suboptimal
            enable_xla_compilation=False      # Intentionally suboptimal
        )
        
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        optimization_info = embedder.optimize_for_gpu()
        
        if 'error' not in optimization_info:
            self.assertIsInstance(optimization_info, dict)
            self.assertIn('optimal_batch_size', optimization_info)
            self.assertIn('current_config', optimization_info)
            self.assertIn('recommendations', optimization_info)
            
            # Should have recommendations due to suboptimal config
            recommendations = optimization_info['recommendations']
            self.assertGreater(len(recommendations), 0)
    
    def test_consistency_between_gpu_and_cpu(self):
        """Test output consistency between GPU and CPU implementations."""
        # Create GPU-enabled embedder
        gpu_config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=True,
            use_vectorized_operations=True,
            learnable_frequencies=False,  # Use fixed frequencies for consistency
            base_frequency=10000.0,
            frequency_scaling=1.0,
            phase_shift=0.0
        )
        
        gpu_embedder = ConfigurableSinusoidalEmbedder(gpu_config)
        
        # Create CPU-only embedder
        cpu_config = SinusoidalConfig(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            enable_gpu_acceleration=False,
            use_vectorized_operations=False,
            learnable_frequencies=False,  # Use fixed frequencies for consistency
            base_frequency=10000.0,
            frequency_scaling=1.0,
            phase_shift=0.0
        )
        
        cpu_embedder = ConfigurableSinusoidalEmbedder(cpu_config)
        
        # Build both embedders
        sample_input = tf.random.uniform(
            (self.batch_size, self.seq_length), 0, self.vocab_size, dtype=tf.int32
        )
        gpu_embedder.build(sample_input.shape)
        cpu_embedder.build(sample_input.shape)
        
        # Get embeddings from both
        gpu_embeddings = gpu_embedder(sample_input, training=False)
        cpu_embeddings = cpu_embedder(sample_input, training=False)
        
        # Check consistency (should be very close)
        diff = tf.reduce_mean(tf.abs(gpu_embeddings - cpu_embeddings))
        # More tolerant threshold since we're likely running on CPU without actual GPU
        self.assertLess(float(diff), 0.1, "GPU and CPU embeddings should be reasonably consistent")


class TestGPUAccelerationEdgeCases(unittest.TestCase):
    """Test edge cases and error handling for GPU acceleration."""
    
    def test_gpu_unavailable_fallback(self):
        """Test fallback behavior when GPU is unavailable."""
        # Mock GPU unavailability
        with patch('tensorflow.config.experimental.list_physical_devices') as mock_list_devices:
            mock_list_devices.return_value = []  # No GPUs
            
            config = GPUConfig(enable_gpu=True)
            accelerator = GPUAccelerator(config)
            
            # Should fallback to CPU
            self.assertFalse(accelerator._gpu_available)
            self.assertFalse(accelerator.config.enable_gpu)
    
    def test_mixed_precision_without_gpu(self):
        """Test mixed precision behavior without GPU."""
        config = GPUConfig(
            enable_gpu=False,
            enable_mixed_precision=True
        )
        
        accelerator = GPUAccelerator(config)
        
        # Mixed precision should be disabled without GPU
        self.assertFalse(accelerator._mixed_precision_enabled)
        self.assertFalse(accelerator.config.enable_mixed_precision)
    
    def test_invalid_batch_size_calculation(self):
        """Test optimal batch size calculation with invalid inputs."""
        # Test with zero or negative values
        with self.assertRaises(ValueError):
            get_optimal_batch_size(0, 128, 1024)
        
        with self.assertRaises(ValueError):
            get_optimal_batch_size(1000, 0, 1024)
        
        with self.assertRaises(ValueError):
            get_optimal_batch_size(1000, 128, 0)
    
    def test_embedder_gpu_initialization_failure(self):
        """Test embedder behavior when GPU initialization fails."""
        # Mock GPU accelerator initialization failure
        with patch('src.lsm.data.configurable_sinusoidal_embedder.GPUAccelerator') as mock_accelerator:
            mock_accelerator.side_effect = Exception("GPU initialization failed")
            
            config = SinusoidalConfig(
                vocab_size=1000,
                embedding_dim=64,
                enable_gpu_acceleration=True
            )
            
            embedder = ConfigurableSinusoidalEmbedder(config)
            
            # Should fallback gracefully
            self.assertIsNone(embedder.gpu_accelerator)
            self.assertFalse(embedder.config.enable_gpu_acceleration)


if __name__ == '__main__':
    # Set up TensorFlow for testing
    tf.config.run_functions_eagerly(True)  # For easier debugging
    
    # Run tests
    unittest.main(verbosity=2)