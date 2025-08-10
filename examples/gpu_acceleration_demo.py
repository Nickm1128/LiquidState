#!/usr/bin/env python3
"""
GPU Acceleration Demo for Sinusoidal Embedder.

This demo showcases GPU acceleration features including:
- CUDA/GPU processing optimization
- Vectorized operations for batch processing
- Mixed precision support for faster computation
- Performance benchmarking and profiling
"""

import os
import sys
import numpy as np
import tensorflow as tf
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lsm.data.configurable_sinusoidal_embedder import (
    ConfigurableSinusoidalEmbedder, SinusoidalConfig
)
from lsm.data.gpu_acceleration import (
    GPUAccelerator, GPUConfig, create_gpu_accelerator, get_optimal_batch_size
)
from lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)


def demonstrate_gpu_configuration():
    """Demonstrate GPU configuration and setup."""
    print("\n" + "="*60)
    print("GPU CONFIGURATION DEMONSTRATION")
    print("="*60)
    
    # Create GPU accelerator with different configurations
    print("\n1. Creating GPU accelerator with default settings...")
    gpu_accelerator = create_gpu_accelerator()
    
    # Get GPU information
    gpu_info = gpu_accelerator.get_gpu_memory_info()
    print(f"GPU Available: {gpu_info.get('gpu_available', False)}")
    print(f"Number of GPUs: {gpu_info.get('num_gpus', 0)}")
    print(f"Mixed Precision: {gpu_info.get('mixed_precision_enabled', False)}")
    print(f"XLA Enabled: {gpu_info.get('xla_enabled', False)}")
    
    # Create custom GPU configuration
    print("\n2. Creating custom GPU configuration...")
    custom_gpu_config = GPUConfig(
        enable_gpu=True,
        enable_mixed_precision=True,
        mixed_precision_policy="mixed_float16",
        enable_vectorization=True,
        enable_xla=True,
        allow_memory_growth=True
    )
    
    custom_accelerator = GPUAccelerator(custom_gpu_config)
    print(f"Custom GPU accelerator created successfully")
    
    return gpu_accelerator, custom_accelerator


def demonstrate_sinusoidal_embedder_gpu():
    """Demonstrate sinusoidal embedder with GPU acceleration."""
    print("\n" + "="*60)
    print("SINUSOIDAL EMBEDDER GPU ACCELERATION")
    print("="*60)
    
    # Configuration parameters
    vocab_size = 10000
    embedding_dim = 256
    batch_size = 32
    seq_length = 128
    
    print(f"\nConfiguration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Embedding Dimension: {embedding_dim}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_length}")
    
    # Create embedder with GPU acceleration enabled
    print("\n1. Creating embedder with GPU acceleration...")
    gpu_config = SinusoidalConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        learnable_frequencies=True,
        enable_gpu_acceleration=True,
        use_vectorized_operations=True,
        use_mixed_precision=True,
        enable_xla_compilation=True
    )
    
    gpu_embedder = ConfigurableSinusoidalEmbedder(gpu_config)
    
    # Build the embedder
    sample_input = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
    gpu_embedder.build(sample_input.shape)
    
    # Get GPU acceleration info
    gpu_info = gpu_embedder.get_gpu_acceleration_info()
    print(f"GPU Acceleration Enabled: {gpu_info.get('gpu_acceleration_enabled', False)}")
    print(f"Vectorized Operations: {gpu_info.get('vectorized_operations', False)}")
    print(f"Mixed Precision: {gpu_info.get('mixed_precision', False)}")
    print(f"XLA Compilation: {gpu_info.get('xla_compilation', False)}")
    
    # Create embedder without GPU acceleration for comparison
    print("\n2. Creating embedder without GPU acceleration...")
    cpu_config = SinusoidalConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        learnable_frequencies=True,
        enable_gpu_acceleration=False,
        use_vectorized_operations=False,
        use_mixed_precision=False,
        enable_xla_compilation=False
    )
    
    cpu_embedder = ConfigurableSinusoidalEmbedder(cpu_config)
    cpu_embedder.build(sample_input.shape)
    
    return gpu_embedder, cpu_embedder, sample_input


def benchmark_performance(gpu_embedder, cpu_embedder, sample_input):
    """Benchmark performance comparison between GPU and CPU embedders."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARKING")
    print("="*60)
    
    num_iterations = 100
    
    print(f"\nRunning {num_iterations} iterations for performance comparison...")
    
    # Warm up both embedders
    print("Warming up embedders...")
    for _ in range(10):
        _ = gpu_embedder(sample_input, training=False)
        _ = cpu_embedder(sample_input, training=False)
    
    # Benchmark GPU embedder
    print("\n1. Benchmarking GPU-accelerated embedder...")
    start_time = time.time()
    for i in range(num_iterations):
        embeddings_gpu = gpu_embedder(sample_input, training=False)
        if i % 20 == 0:
            print(f"  Iteration {i}/{num_iterations}")
    gpu_time = time.time() - start_time
    
    # Benchmark CPU embedder
    print("\n2. Benchmarking CPU embedder...")
    start_time = time.time()
    for i in range(num_iterations):
        embeddings_cpu = cpu_embedder(sample_input, training=False)
        if i % 20 == 0:
            print(f"  Iteration {i}/{num_iterations}")
    cpu_time = time.time() - start_time
    
    # Calculate performance metrics
    gpu_avg_time = (gpu_time / num_iterations) * 1000  # ms
    cpu_avg_time = (cpu_time / num_iterations) * 1000  # ms
    speedup = cpu_time / gpu_time if gpu_time > 0 else 1.0
    
    print(f"\nPerformance Results:")
    print(f"  GPU Average Time: {gpu_avg_time:.2f} ms per iteration")
    print(f"  CPU Average Time: {cpu_avg_time:.2f} ms per iteration")
    print(f"  Speedup: {speedup:.2f}x")
    
    # Verify output consistency
    print(f"\n3. Verifying output consistency...")
    diff = tf.reduce_mean(tf.abs(embeddings_gpu - embeddings_cpu))
    print(f"  Mean absolute difference: {diff:.6f}")
    print(f"  Outputs are {'consistent' if diff < 1e-4 else 'different'}")
    
    return {
        'gpu_time_ms': gpu_avg_time,
        'cpu_time_ms': cpu_avg_time,
        'speedup': speedup,
        'output_difference': float(diff)
    }


def demonstrate_gpu_benchmarking(gpu_embedder):
    """Demonstrate built-in GPU benchmarking features."""
    print("\n" + "="*60)
    print("BUILT-IN GPU BENCHMARKING")
    print("="*60)
    
    # Run built-in benchmark
    print("\n1. Running built-in GPU benchmark...")
    benchmark_results = gpu_embedder.benchmark_gpu_performance(
        batch_size=32,
        seq_length=128,
        num_iterations=50
    )
    
    if 'error' not in benchmark_results:
        print(f"Benchmark Results:")
        for key, value in benchmark_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"Benchmark failed: {benchmark_results['error']}")
    
    # Get optimization recommendations
    print("\n2. Getting GPU optimization recommendations...")
    optimization_info = gpu_embedder.optimize_for_gpu(available_memory_mb=4096)
    
    if 'error' not in optimization_info:
        print(f"Optimization Recommendations:")
        print(f"  Optimal Batch Size: {optimization_info.get('optimal_batch_size', 'N/A')}")
        
        recommendations = optimization_info.get('recommendations', [])
        if recommendations:
            print(f"  Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"    {i}. {rec}")
        else:
            print(f"  No additional recommendations - configuration is optimal!")
    else:
        print(f"Optimization analysis failed: {optimization_info['error']}")


def demonstrate_mixed_precision():
    """Demonstrate mixed precision training benefits."""
    print("\n" + "="*60)
    print("MIXED PRECISION DEMONSTRATION")
    print("="*60)
    
    vocab_size = 20000
    embedding_dim = 512
    batch_size = 64
    seq_length = 256
    
    print(f"\nLarge model configuration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Embedding Dimension: {embedding_dim}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_length}")
    
    # Create embedder with mixed precision
    print("\n1. Creating embedder with mixed precision...")
    mixed_precision_config = SinusoidalConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        enable_gpu_acceleration=True,
        use_mixed_precision=True,
        use_vectorized_operations=True,
        enable_xla_compilation=True
    )
    
    mixed_precision_embedder = ConfigurableSinusoidalEmbedder(mixed_precision_config)
    
    # Create embedder without mixed precision
    print("2. Creating embedder without mixed precision...")
    full_precision_config = SinusoidalConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        enable_gpu_acceleration=True,
        use_mixed_precision=False,
        use_vectorized_operations=True,
        enable_xla_compilation=True
    )
    
    full_precision_embedder = ConfigurableSinusoidalEmbedder(full_precision_config)
    
    # Build embedders
    sample_input = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
    mixed_precision_embedder.build(sample_input.shape)
    full_precision_embedder.build(sample_input.shape)
    
    # Compare performance
    print("\n3. Comparing mixed precision vs full precision...")
    
    # Warm up
    for _ in range(5):
        _ = mixed_precision_embedder(sample_input)
        _ = full_precision_embedder(sample_input)
    
    # Benchmark mixed precision
    start_time = time.time()
    for _ in range(20):
        _ = mixed_precision_embedder(sample_input)
    mixed_precision_time = time.time() - start_time
    
    # Benchmark full precision
    start_time = time.time()
    for _ in range(20):
        _ = full_precision_embedder(sample_input)
    full_precision_time = time.time() - start_time
    
    speedup = full_precision_time / mixed_precision_time if mixed_precision_time > 0 else 1.0
    
    print(f"Mixed Precision Time: {mixed_precision_time:.3f}s")
    print(f"Full Precision Time: {full_precision_time:.3f}s")
    print(f"Mixed Precision Speedup: {speedup:.2f}x")


def demonstrate_optimal_batch_size():
    """Demonstrate optimal batch size calculation."""
    print("\n" + "="*60)
    print("OPTIMAL BATCH SIZE CALCULATION")
    print("="*60)
    
    # Test different configurations
    configurations = [
        (5000, 128, 2048),   # Small vocab, small dim
        (10000, 256, 4096),  # Medium vocab, medium dim
        (50000, 512, 8192),  # Large vocab, large dim
        (100000, 768, 16384) # Very large vocab, large dim
    ]
    
    print(f"\nCalculating optimal batch sizes for different configurations:")
    print(f"{'Vocab Size':<12} {'Embed Dim':<12} {'Memory (MB)':<12} {'Optimal Batch':<15}")
    print("-" * 60)
    
    for vocab_size, embedding_dim, memory_mb in configurations:
        optimal_batch = get_optimal_batch_size(vocab_size, embedding_dim, memory_mb)
        print(f"{vocab_size:<12} {embedding_dim:<12} {memory_mb:<12} {optimal_batch:<15}")


def main():
    """Main demonstration function."""
    print("GPU Acceleration Demo for Sinusoidal Embedder")
    print("=" * 60)
    
    try:
        # Check TensorFlow GPU availability
        print(f"TensorFlow version: {tf.__version__}")
        print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
        
        # Demonstrate GPU configuration
        gpu_accelerator, custom_accelerator = demonstrate_gpu_configuration()
        
        # Demonstrate sinusoidal embedder with GPU acceleration
        gpu_embedder, cpu_embedder, sample_input = demonstrate_sinusoidal_embedder_gpu()
        
        # Benchmark performance
        benchmark_results = benchmark_performance(gpu_embedder, cpu_embedder, sample_input)
        
        # Demonstrate built-in benchmarking
        demonstrate_gpu_benchmarking(gpu_embedder)
        
        # Demonstrate mixed precision
        demonstrate_mixed_precision()
        
        # Demonstrate optimal batch size calculation
        demonstrate_optimal_batch_size()
        
        print("\n" + "="*60)
        print("GPU ACCELERATION DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        
        # Summary
        print(f"\nSummary:")
        print(f"  GPU acceleration provides {benchmark_results['speedup']:.2f}x speedup")
        print(f"  Mixed precision and XLA compilation further improve performance")
        print(f"  Vectorized operations optimize batch processing")
        print(f"  Built-in benchmarking helps optimize configurations")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()