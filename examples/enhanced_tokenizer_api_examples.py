#!/usr/bin/env python3
"""
Comprehensive examples for the Enhanced Tokenizer API.

This file demonstrates all major features and usage patterns of the enhanced
tokenizer system, including basic tokenization, advanced sinusoidal embeddings,
streaming data processing, and performance optimization.
"""

import os
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any, Optional

# Import enhanced tokenizer components
from lsm.data.enhanced_tokenization import (
    EnhancedTokenizerWrapper, TokenizerConfig, TokenizerRegistry
)
from lsm.data.configurable_sinusoidal_embedder import (
    ConfigurableSinusoidalEmbedder, SinusoidalConfig
)
from lsm.data.streaming_data_iterator import StreamingDataIterator
from lsm.data.intelligent_caching import CacheConfig
from lsm.data.gpu_acceleration import GPUConfig
from lsm.data.memory_efficient_storage import MemoryStorageConfig


def example_basic_tokenization():
    """
    Example 1: Basic tokenization with different backends.
    
    Demonstrates how to use the enhanced tokenizer with various backends
    including automatic backend detection and basic tokenization operations.
    """
    print("=" * 60)
    print("Example 1: Basic Tokenization")
    print("=" * 60)
    
    # Test different tokenizer backends
    backends = [
        'gpt2',                    # OpenAI GPT-2
        'bert-base-uncased',       # HuggingFace BERT
        'en_core_web_sm'           # spaCy English model (if available)
    ]
    
    sample_texts = [
        "Hello, world! How are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence."
    ]
    
    for backend in backends:
        try:
            print(f"\n--- Testing {backend} ---")
            
            # Create tokenizer with automatic backend detection
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=backend,
                embedding_dim=256,
                max_length=128
            )
            
            # Basic tokenization
            tokens = tokenizer.tokenize(sample_texts)
            print(f"Vocabulary size: {tokenizer.get_vocab_size():,}")
            print(f"Sample tokens: {tokens[0][:10]}...")  # First 10 tokens
            
            # Decode back to text
            decoded = tokenizer.decode(tokens[0])
            print(f"Decoded text: {decoded[:50]}...")
            
            # Get special tokens
            special_tokens = tokenizer.get_special_tokens()
            print(f"Special tokens: {list(special_tokens.keys())}")
            
        except Exception as e:
            print(f"Backend {backend} failed: {e}")
    
    print("\n" + "=" * 60)


def example_advanced_sinusoidal_embeddings():
    """
    Example 2: Advanced sinusoidal embeddings with learnable parameters.
    
    Demonstrates how to create and configure sinusoidal embedders with
    learnable frequencies, relative positioning, and automatic adaptation.
    """
    print("=" * 60)
    print("Example 2: Advanced Sinusoidal Embeddings")
    print("=" * 60)
    
    # Create tokenizer
    tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=512)
    
    # Example 2a: Basic configurable embedder
    print("\n--- Basic Configurable Embedder ---")
    embedder_basic = tokenizer.create_configurable_sinusoidal_embedder(
        learnable_frequencies=True,
        base_frequency=10000.0,
        frequency_scaling=1.0
    )
    
    print(f"Embedder vocab size: {embedder_basic.config.vocab_size}")
    print(f"Embedding dimension: {embedder_basic.config.embedding_dim}")
    print(f"Learnable frequencies: {embedder_basic.config.learnable_frequencies}")
    
    # Example 2b: Advanced embedder with relative positioning
    print("\n--- Advanced Embedder with Relative Positioning ---")
    embedder_advanced = tokenizer.create_configurable_sinusoidal_embedder(
        learnable_frequencies=True,
        use_relative_position=True,
        base_frequency=5000.0,
        frequency_scaling=1.5,
        relative_position_window=32,
        temperature=0.8
    )
    
    # Example 2c: Demonstrate embedding dimension adaptation
    print("\n--- Embedding Dimension Adaptation ---")
    suggestions = tokenizer.get_embedding_dimension_suggestions()
    print(f"Dimension suggestions: {suggestions}")
    
    # Create embedder with optimal dimension for 'large' model
    embedder_optimized = tokenizer.auto_adapt_embedding_dimension(
        target_dim=suggestions['large'],
        preserve_properties=True
    )
    
    print(f"Optimized embedding dimension: {embedder_optimized.config.embedding_dim}")
    
    # Example 2d: Use embedder in a simple model
    print("\n--- Using Embedder in Keras Model ---")
    
    # Create a simple classification model
    inputs = tf.keras.Input(shape=(None,), dtype=tf.int32, name='token_ids')
    embeddings = embedder_basic(inputs)
    
    # Add some processing layers
    pooled = tf.keras.layers.GlobalAveragePooling1D()(embeddings)
    dense = tf.keras.layers.Dense(128, activation='relu')(pooled)
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='predictions')(dense)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    
    print("Model created successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Test with sample data
    sample_tokens = tokenizer.tokenize(["This is a test sentence."])
    sample_input = tf.constant(sample_tokens)
    sample_output = model(sample_input)
    print(f"Sample output shape: {sample_output.shape}")
    
    print("\n" + "=" * 60)


def example_streaming_data_processing():
    """
    Example 3: Streaming data processing for large datasets.
    
    Demonstrates memory-efficient processing of large datasets using
    streaming iterators with adaptive batch size management.
    """
    print("=" * 60)
    print("Example 3: Streaming Data Processing")
    print("=" * 60)
    
    # Create sample data files for demonstration
    sample_data_dir = "temp_streaming_demo"
    os.makedirs(sample_data_dir, exist_ok=True)
    
    # Create sample text file
    text_file = os.path.join(sample_data_dir, "sample.txt")
    with open(text_file, 'w') as f:
        for i in range(1000):
            f.write(f"This is sample text line {i} for streaming demonstration.\n")
    
    # Create sample JSONL file
    jsonl_file = os.path.join(sample_data_dir, "sample.jsonl")
    with open(jsonl_file, 'w') as f:
        for i in range(500):
            f.write(f'{{"text": "JSON sample {i}", "label": {i % 5}}}\n')
    
    try:
        # Example 3a: Basic streaming iterator
        print("\n--- Basic Streaming Iterator ---")
        
        def progress_callback(processed, total_estimate):
            if processed % 100 == 0:  # Print every 100 items
                percent = (processed / total_estimate) * 100 if total_estimate > 0 else 0
                print(f"  Processed: {processed:,} items ({percent:.1f}%)")
        
        iterator = StreamingDataIterator(
            data_source=[text_file, jsonl_file],
            batch_size=50,
            auto_adjust_batch_size=False,
            progress_callback=progress_callback,
            extract_text=True,
            text_field='text'
        )
        
        batch_count = 0
        total_items = 0
        
        for batch in iterator:
            batch_count += 1
            total_items += len(batch)
            
            if batch_count <= 3:  # Show first few batches
                print(f"  Batch {batch_count}: {len(batch)} items")
                print(f"    Sample: {batch[0][:50]}...")
        
        print(f"Total batches: {batch_count}")
        print(f"Total items: {total_items}")
        
        # Example 3b: Adaptive streaming with memory monitoring
        print("\n--- Adaptive Streaming with Memory Monitoring ---")
        
        iterator_adaptive = StreamingDataIterator(
            data_source=sample_data_dir,
            batch_size=100,
            memory_threshold_mb=50.0,  # Low threshold for demo
            auto_adjust_batch_size=True,
            min_batch_size=10,
            max_batch_size=200,
            memory_check_interval=5,
            extract_text=True
        )
        
        batch_count = 0
        for batch in iterator_adaptive:
            batch_count += 1
            
            if batch_count <= 5:
                current_batch_size = len(batch)
                print(f"  Batch {batch_count}: {current_batch_size} items "
                      f"(configured: {iterator_adaptive.batch_size})")
            
            if batch_count >= 10:  # Limit for demo
                break
        
        # Show adaptive statistics
        stats = iterator_adaptive.get_adaptive_stats()
        print(f"\nAdaptive Statistics:")
        print(f"  Final batch size: {stats['current_config']['batch_size']}")
        print(f"  Memory threshold: {stats['current_config']['memory_threshold_mb']}MB")
        print(f"  Adjustments made: {len(stats['adjustment_history'])}")
        
        # Example 3c: Streaming tokenizer fitting
        print("\n--- Streaming Tokenizer Fitting ---")
        
        tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=128)
        
        def fitting_progress(processed, total_estimate):
            if processed % 200 == 0:
                print(f"  Fitting progress: {processed:,} sequences processed")
        
        # Fit embedder on streaming data
        embedder = tokenizer.fit_streaming(
            data_source=sample_data_dir,
            batch_size=50,
            epochs=5,  # Small number for demo
            memory_threshold_mb=100.0,
            auto_adjust_batch_size=True,
            progress_callback=fitting_progress
        )
        
        print("Streaming fitting completed!")
        print(f"Embedder vocab size: {embedder.vocab_size}")
        print(f"Embedding dimension: {embedder.embedding_dim}")
        
    finally:
        # Clean up sample files
        import shutil
        if os.path.exists(sample_data_dir):
            shutil.rmtree(sample_data_dir)
    
    print("\n" + "=" * 60)


def example_performance_optimization():
    """
    Example 4: Performance optimization with caching, GPU acceleration, and memory efficiency.
    
    Demonstrates how to configure and use performance optimization features
    including intelligent caching, GPU acceleration, and memory-efficient storage.
    """
    print("=" * 60)
    print("Example 4: Performance Optimization")
    print("=" * 60)
    
    # Example 4a: Intelligent caching configuration
    print("\n--- Intelligent Caching Configuration ---")
    
    cache_config = CacheConfig(
        max_cache_size=20000,
        enable_batch_caching=True,
        batch_cache_size=10000,
        enable_cache_warming=True,
        warmup_strategy="frequency",
        warmup_size=2000,
        enable_metrics=True
    )
    
    tokenizer_cached = EnhancedTokenizerWrapper(
        'gpt2',
        embedding_dim=256,
        enable_caching=True,
        cache_config=cache_config
    )
    
    print(f"Cache configuration:")
    print(f"  Max cache size: {cache_config.max_cache_size:,}")
    print(f"  Batch caching: {cache_config.enable_batch_caching}")
    print(f"  Cache warming: {cache_config.enable_cache_warming}")
    print(f"  Warmup strategy: {cache_config.warmup_strategy}")
    
    # Test caching performance
    sample_texts = [
        "This is a test sentence for caching.",
        "Another test sentence with different content.",
        "This is a test sentence for caching.",  # Repeated for cache hit
        "Yet another unique sentence for testing."
    ]
    
    # First pass - populate cache
    print("\nFirst tokenization pass (populating cache):")
    tokens1 = tokenizer_cached.tokenize(sample_texts)
    
    # Second pass - should hit cache
    print("Second tokenization pass (using cache):")
    tokens2 = tokenizer_cached.tokenize(sample_texts)
    
    # Verify results are identical
    print(f"Results identical: {tokens1 == tokens2}")
    
    # Example 4b: GPU acceleration configuration
    print("\n--- GPU Acceleration Configuration ---")
    
    # Check GPU availability
    gpu_available = len(tf.config.experimental.list_physical_devices('GPU')) > 0
    print(f"GPU available: {gpu_available}")
    
    if gpu_available:
        gpu_config = GPUConfig(
            enable_gpu=True,
            enable_mixed_precision=True,
            mixed_precision_policy="mixed_float16",
            enable_vectorization=True,
            enable_xla=True
        )
        
        sinusoidal_config = SinusoidalConfig(
            embedding_dim=512,
            vocab_size=50000,
            learnable_frequencies=True,
            enable_gpu_acceleration=True,
            gpu_config=gpu_config,
            use_mixed_precision=True,
            use_vectorized_operations=True,
            enable_xla_compilation=True
        )
        
        embedder_gpu = ConfigurableSinusoidalEmbedder(sinusoidal_config)
        
        print("GPU-accelerated embedder created!")
        print(f"  Mixed precision: {sinusoidal_config.use_mixed_precision}")
        print(f"  Vectorized operations: {sinusoidal_config.use_vectorized_operations}")
        print(f"  XLA compilation: {sinusoidal_config.enable_xla_compilation}")
        
        # Get GPU acceleration info
        gpu_info = embedder_gpu.get_gpu_acceleration_info()
        print(f"  GPU acceleration enabled: {gpu_info['gpu_acceleration_enabled']}")
        
        # Benchmark GPU performance
        print("\nBenchmarking GPU performance...")
        benchmark_results = embedder_gpu.benchmark_gpu_performance(
            batch_size=32,
            seq_length=128,
            num_iterations=50
        )
        
        if 'error' not in benchmark_results:
            print(f"  Average forward pass time: {benchmark_results.get('avg_forward_time', 'N/A'):.4f}s")
            print(f"  Throughput: {benchmark_results.get('throughput', 'N/A'):.1f} samples/sec")
    else:
        print("GPU not available, skipping GPU acceleration example")
    
    # Example 4c: Memory-efficient storage for large vocabularies
    print("\n--- Memory-Efficient Storage ---")
    
    memory_config = MemoryStorageConfig(
        use_memory_mapping=True,
        memory_map_threshold=10000,
        use_compression=True,
        compression_level=6,
        use_gradient_checkpointing=True,
        enable_embedding_cache=True,
        cache_size=5000
    )
    
    large_vocab_config = SinusoidalConfig(
        embedding_dim=1024,
        vocab_size=100000,  # Large vocabulary
        learnable_frequencies=True,
        use_memory_efficient_storage=True,
        memory_storage_config=memory_config
    )
    
    print(f"Memory-efficient configuration:")
    print(f"  Vocabulary size: {large_vocab_config.vocab_size:,}")
    print(f"  Embedding dimension: {large_vocab_config.embedding_dim}")
    print(f"  Memory mapping: {memory_config.use_memory_mapping}")
    print(f"  Compression: {memory_config.use_compression}")
    print(f"  Gradient checkpointing: {memory_config.use_gradient_checkpointing}")
    
    try:
        embedder_memory_efficient = ConfigurableSinusoidalEmbedder(large_vocab_config)
        print("Memory-efficient embedder created successfully!")
        
        # Test with sample input
        sample_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
        output = embedder_memory_efficient(sample_input)
        print(f"Sample output shape: {output.shape}")
        
    except Exception as e:
        print(f"Memory-efficient embedder creation failed: {e}")
    
    print("\n" + "=" * 60)


def example_multi_backend_comparison():
    """
    Example 5: Multi-backend comparison and benchmarking.
    
    Demonstrates how to compare different tokenizer backends and
    benchmark their performance characteristics.
    """
    print("=" * 60)
    print("Example 5: Multi-Backend Comparison")
    print("=" * 60)
    
    # Test sentences with different characteristics
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning and artificial intelligence are transforming technology.",
        "Hello, world! How are you doing today? 😊",
        "This is a longer sentence with more complex vocabulary and punctuation marks.",
        "Short text.",
        "Numbers: 123, 456.789, and symbols: @#$%^&*()",
    ]
    
    # Backends to compare
    backends_to_test = [
        ('gpt2', 'GPT-2'),
        ('bert-base-uncased', 'BERT Base Uncased'),
        ('distilbert-base-uncased', 'DistilBERT Base'),
    ]
    
    results = {}
    
    for backend_name, display_name in backends_to_test:
        try:
            print(f"\n--- Testing {display_name} ---")
            
            # Create tokenizer
            tokenizer = EnhancedTokenizerWrapper(
                backend_name,
                embedding_dim=256,
                max_length=128
            )
            
            # Basic statistics
            vocab_size = tokenizer.get_vocab_size()
            special_tokens = tokenizer.get_special_tokens()
            
            print(f"Vocabulary size: {vocab_size:,}")
            print(f"Special tokens: {len(special_tokens)}")
            
            # Tokenize test sentences
            tokens = tokenizer.tokenize(test_sentences)
            
            # Calculate statistics
            token_counts = [len(seq) for seq in tokens]
            avg_tokens = np.mean(token_counts)
            max_tokens = max(token_counts)
            min_tokens = min(token_counts)
            
            print(f"Token count statistics:")
            print(f"  Average: {avg_tokens:.1f}")
            print(f"  Range: {min_tokens} - {max_tokens}")
            
            # Test decoding
            decoded = tokenizer.decode(tokens[0])
            original = test_sentences[0]
            
            print(f"Decoding test:")
            print(f"  Original: {original}")
            print(f"  Decoded:  {decoded}")
            print(f"  Match: {original.lower().strip() == decoded.lower().strip()}")
            
            # Store results
            results[backend_name] = {
                'vocab_size': vocab_size,
                'avg_tokens': avg_tokens,
                'token_range': (min_tokens, max_tokens),
                'special_tokens_count': len(special_tokens)
            }
            
            # Test embedding dimension suggestions
            suggestions = tokenizer.get_embedding_dimension_suggestions()
            print(f"Embedding suggestions: {suggestions}")
            
        except Exception as e:
            print(f"Failed to test {display_name}: {e}")
            results[backend_name] = {'error': str(e)}
    
    # Summary comparison
    print(f"\n--- Comparison Summary ---")
    print(f"{'Backend':<20} {'Vocab Size':<12} {'Avg Tokens':<12} {'Token Range':<15}")
    print("-" * 65)
    
    for backend_name, display_name in backends_to_test:
        if backend_name in results and 'error' not in results[backend_name]:
            data = results[backend_name]
            vocab_size = f"{data['vocab_size']:,}"
            avg_tokens = f"{data['avg_tokens']:.1f}"
            token_range = f"{data['token_range'][0]}-{data['token_range'][1]}"
            
            print(f"{display_name:<20} {vocab_size:<12} {avg_tokens:<12} {token_range:<15}")
        else:
            print(f"{display_name:<20} {'ERROR':<12} {'N/A':<12} {'N/A':<15}")
    
    print("\n" + "=" * 60)


def example_integration_with_convenience_api():
    """
    Example 6: Integration with LSM convenience API.
    
    Demonstrates how to use the enhanced tokenizer with existing
    LSM convenience classes like LSMGenerator.
    """
    print("=" * 60)
    print("Example 6: Integration with Convenience API")
    print("=" * 60)
    
    try:
        # This would normally import from lsm.convenience
        # For this example, we'll simulate the integration
        print("\n--- Enhanced Tokenizer with LSMGenerator ---")
        
        # Create enhanced tokenizer
        tokenizer = EnhancedTokenizerWrapper(
            'gpt2',
            embedding_dim=256,
            max_length=128,
            enable_caching=True
        )
        
        # Create configurable embedder
        embedder = tokenizer.create_configurable_sinusoidal_embedder(
            learnable_frequencies=True,
            base_frequency=10000.0
        )
        
        print("Enhanced tokenizer and embedder created for convenience API integration")
        print(f"Tokenizer vocab size: {tokenizer.get_vocab_size():,}")
        print(f"Embedder dimension: {embedder.config.embedding_dim}")
        
        # Simulate convenience API usage
        sample_data = [
            "This is training data for the LSM model.",
            "Another example sentence for training.",
            "More training data with different content."
        ]
        
        # Tokenize data as would be done in convenience API
        tokenized_data = tokenizer.tokenize(sample_data)
        print(f"Tokenized {len(sample_data)} training samples")
        print(f"Sample tokens: {tokenized_data[0][:10]}...")
        
        # Create embeddings as would be done in convenience API
        token_tensor = tf.constant(tokenized_data)
        embeddings = embedder(token_tensor)
        print(f"Generated embeddings shape: {embeddings.shape}")
        
        print("\nIntegration successful! Enhanced tokenizer ready for convenience API.")
        
    except Exception as e:
        print(f"Integration example failed: {e}")
    
    print("\n" + "=" * 60)


def main():
    """
    Run all examples to demonstrate the enhanced tokenizer API.
    """
    print("Enhanced Tokenizer API Examples")
    print("=" * 60)
    print("This script demonstrates all major features of the enhanced tokenizer system.")
    print("Examples may take a few minutes to complete...")
    print()
    
    try:
        # Run all examples
        example_basic_tokenization()
        example_advanced_sinusoidal_embeddings()
        example_streaming_data_processing()
        example_performance_optimization()
        example_multi_backend_comparison()
        example_integration_with_convenience_api()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nExample execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()