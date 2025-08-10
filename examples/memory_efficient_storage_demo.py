#!/usr/bin/env python3
"""
Memory-Efficient Embedding Storage Demo.

This example demonstrates the memory-efficient embedding storage capabilities
including memory-mapped embeddings, compressed storage, and gradient checkpointing.
"""

import os
import sys
import tempfile
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lsm.data.memory_efficient_storage import (
    MemoryStorageConfig,
    MemoryMappedEmbedding,
    CompressedEmbeddingStorage,
    MemoryEfficientEmbeddingLayer
)
from lsm.data.configurable_sinusoidal_embedder import (
    SinusoidalConfig,
    ConfigurableSinusoidalEmbedder
)


def demo_memory_mapped_embeddings():
    """Demonstrate memory-mapped embedding functionality."""
    print("=" * 60)
    print("Memory-Mapped Embeddings Demo")
    print("=" * 60)
    
    # Configuration
    vocab_size = 50000
    embedding_dim = 256
    
    print(f"Creating memory-mapped embedding: {vocab_size}x{embedding_dim}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create memory-mapped embedding
        embedding = MemoryMappedEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            storage_dir=temp_dir
        )
        
        print(f"Storage directory: {temp_dir}")
        print(f"Embedding file size: {os.path.getsize(embedding.embedding_file) / (1024*1024):.2f} MB")
        
        # Test single embedding access
        print("\nTesting single embedding access...")
        single_embedding = embedding.get_embeddings(0)
        print(f"Single embedding shape: {single_embedding.shape}")
        
        # Test batch access
        print("\nTesting batch access...")
        batch_indices = list(range(0, 1000, 100))  # [0, 100, 200, ..., 900]
        batch_embeddings = embedding.get_embeddings(batch_indices)
        print(f"Batch embeddings shape: {batch_embeddings.shape}")
        
        # Test setting embeddings
        print("\nTesting embedding updates...")
        new_values = np.ones((len(batch_indices), embedding_dim), dtype=np.float32)
        embedding.set_embeddings(batch_indices, new_values)
        
        # Verify updates
        updated_embeddings = embedding.get_embeddings(batch_indices)
        print(f"All embeddings updated correctly: {np.allclose(updated_embeddings, new_values)}")
        
        # Test batch operations
        print("\nTesting batch operations...")
        batch_data = embedding.get_batch(1000, 1100)
        print(f"Batch data shape: {batch_data.shape}")
        
        # Clean up
        embedding.cleanup()
        print("Memory-mapped embedding cleaned up successfully")


def demo_compressed_storage():
    """Demonstrate compressed embedding storage."""
    print("\n" + "=" * 60)
    print("Compressed Embedding Storage Demo")
    print("=" * 60)
    
    # Configuration
    vocab_size = 25000
    embedding_dim = 128
    chunk_size = 1000
    
    print(f"Creating compressed storage: {vocab_size}x{embedding_dim}")
    print(f"Chunk size: {chunk_size}")
    
    # Create compressed storage
    storage = CompressedEmbeddingStorage(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        compression_level=6,
        chunk_size=chunk_size
    )
    
    # Get compression statistics
    stats = storage.get_compression_stats()
    print(f"\nCompression Statistics:")
    print(f"  Uncompressed size: {stats['uncompressed_size_bytes'] / (1024*1024):.2f} MB")
    print(f"  Compressed size: {stats['compressed_size_bytes'] / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"  Number of chunks: {stats['num_chunks']}")
    
    # Test embedding access across chunks
    print("\nTesting cross-chunk access...")
    test_indices = [0, 500, 1500, 5000, 10000, 20000]  # Across multiple chunks
    embeddings = storage.get_embeddings(test_indices)
    print(f"Retrieved embeddings shape: {embeddings.shape}")
    
    # Test setting embeddings
    print("\nTesting embedding updates...")
    new_values = np.random.random((len(test_indices), embedding_dim)).astype(np.float32)
    storage.set_embeddings(test_indices, new_values)
    
    # Verify updates
    updated_embeddings = storage.get_embeddings(test_indices)
    print(f"Updates successful: {np.allclose(updated_embeddings, new_values, atol=1e-5)}")
    
    # Test save and load
    print("\nTesting save and load...")
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
        save_path = f.name
    
    try:
        storage.save_to_file(save_path)
        print(f"Saved compressed storage to: {save_path}")
        print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
        
        # Load into new storage
        storage2 = CompressedEmbeddingStorage(vocab_size, embedding_dim, chunk_size=chunk_size)
        storage2.load_from_file(save_path)
        
        # Verify loaded data
        loaded_embeddings = storage2.get_embeddings(test_indices)
        print(f"Load successful: {np.allclose(loaded_embeddings, new_values, atol=1e-5)}")
        
    finally:
        if os.path.exists(save_path):
            os.unlink(save_path)


def demo_memory_efficient_layer():
    """Demonstrate the integrated memory-efficient embedding layer."""
    print("\n" + "=" * 60)
    print("Memory-Efficient Embedding Layer Demo")
    print("=" * 60)
    
    # Test different storage strategies
    strategies = [
        ("Standard", MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=False,
            use_gradient_checkpointing=False
        )),
        ("Compressed", MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=True,
            compression_threshold=1000
        )),
        ("Gradient Checkpointed", MemoryStorageConfig(
            use_memory_mapping=False,
            use_compression=False,
            use_gradient_checkpointing=True,
            checkpoint_segments=4
        ))
    ]
    
    vocab_size = 5000
    embedding_dim = 128
    batch_size = 16
    seq_length = 32
    
    # Create test input
    test_input = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
    
    for strategy_name, config in strategies:
        print(f"\nTesting {strategy_name} strategy...")
        
        # Create layer
        layer = MemoryEfficientEmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            config=config
        )
        
        print(f"  Storage strategy: {layer.storage_strategy}")
        
        # Forward pass
        outputs = layer(test_input, training=True)
        print(f"  Output shape: {outputs.shape}")
        print(f"  Output dtype: {outputs.dtype}")
        
        # Get storage stats
        stats = layer.get_storage_stats()
        print(f"  Storage stats: {stats}")
        
        # Test configuration serialization
        layer_config = layer.get_config()
        reconstructed = MemoryEfficientEmbeddingLayer.from_config(layer_config)
        print(f"  Config serialization successful: {reconstructed.storage_strategy == layer.storage_strategy}")


def demo_sinusoidal_embedder_integration():
    """Demonstrate integration with ConfigurableSinusoidalEmbedder."""
    print("\n" + "=" * 60)
    print("Sinusoidal Embedder Integration Demo")
    print("=" * 60)
    
    # Configuration for large vocabulary
    vocab_size = 30000
    embedding_dim = 256
    
    print(f"Creating sinusoidal embedder with memory-efficient storage")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")
    
    # Create memory storage configuration
    memory_config = MemoryStorageConfig(
        use_compression=True,
        compression_level=6,
        compression_threshold=10000,
        use_gradient_checkpointing=True,
        checkpoint_segments=4
    )
    
    # Create sinusoidal configuration
    sinusoidal_config = SinusoidalConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        base_frequency=10000.0,
        learnable_frequencies=True,
        use_memory_efficient_storage=True,
        memory_storage_config=memory_config
    )
    
    # Create embedder
    embedder = ConfigurableSinusoidalEmbedder(sinusoidal_config)
    
    print(f"Memory-efficient storage enabled: {embedder.config.use_memory_efficient_storage}")
    print(f"Token embedding type: {type(embedder.token_embedding).__name__}")
    
    # Test forward pass
    batch_size = 8
    seq_length = 64
    test_input = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
    
    print(f"\nTesting forward pass...")
    print(f"Input shape: {test_input.shape}")
    
    outputs = embedder(test_input, training=True)
    print(f"Output shape: {outputs.shape}")
    print(f"Output dtype: {outputs.dtype}")
    
    # Get memory storage statistics
    memory_stats = embedder.get_memory_storage_stats()
    print(f"\nMemory Storage Statistics:")
    for key, value in memory_stats.items():
        print(f"  {key}: {value}")
    
    # Test configuration save/load with memory storage
    print(f"\nTesting configuration save/load...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        embedder.save_config(config_path)
        print(f"Configuration saved to: {config_path}")
        
        # Load configuration
        loaded_embedder = ConfigurableSinusoidalEmbedder.load_config(config_path)
        print(f"Configuration loaded successfully")
        print(f"Memory-efficient storage preserved: {loaded_embedder.config.use_memory_efficient_storage}")
        
        # Test that loaded embedder works
        loaded_outputs = loaded_embedder(test_input, training=True)
        print(f"Loaded embedder output shape: {loaded_outputs.shape}")
        
    finally:
        if os.path.exists(config_path):
            os.unlink(config_path)
    
    # Clean up memory storage
    embedder.cleanup_memory_storage()
    print("Memory storage cleaned up successfully")


def demo_performance_comparison():
    """Demonstrate performance comparison between storage strategies."""
    print("\n" + "=" * 60)
    print("Performance Comparison Demo")
    print("=" * 60)
    
    import time
    
    # Test configuration
    vocab_size = 10000
    embedding_dim = 128
    batch_size = 32
    seq_length = 128
    num_iterations = 10
    
    # Create test input
    test_input = tf.random.uniform((batch_size, seq_length), 0, vocab_size, dtype=tf.int32)
    
    # Test different configurations
    configs = [
        ("Standard", SinusoidalConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            use_memory_efficient_storage=False
        )),
        ("Memory-Efficient (Compressed)", SinusoidalConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            use_memory_efficient_storage=True,
            memory_storage_config=MemoryStorageConfig(
                use_compression=True,
                compression_threshold=5000
            )
        )),
        ("Memory-Efficient (Gradient Checkpointed)", SinusoidalConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            use_memory_efficient_storage=True,
            memory_storage_config=MemoryStorageConfig(
                use_gradient_checkpointing=True,
                checkpoint_segments=4
            )
        ))
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\nTesting {config_name}...")
        
        # Create embedder
        embedder = ConfigurableSinusoidalEmbedder(config)
        
        # Warm up
        _ = embedder(test_input, training=True)
        
        # Time forward passes
        start_time = time.time()
        for _ in range(num_iterations):
            _ = embedder(test_input, training=True)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_iterations
        results.append((config_name, avg_time))
        
        print(f"  Average forward pass time: {avg_time*1000:.2f} ms")
        
        # Get memory stats if available
        if config.use_memory_efficient_storage:
            memory_stats = embedder.get_memory_storage_stats()
            if 'storage_strategy' in memory_stats:
                print(f"  Storage strategy: {memory_stats['storage_strategy']}")
        
        # Clean up
        if hasattr(embedder, 'cleanup_memory_storage'):
            embedder.cleanup_memory_storage()
    
    # Summary
    print(f"\nPerformance Summary:")
    print(f"{'Configuration':<40} {'Avg Time (ms)':<15}")
    print("-" * 55)
    for config_name, avg_time in results:
        print(f"{config_name:<40} {avg_time*1000:<15.2f}")


def main():
    """Run all memory-efficient storage demos."""
    print("Memory-Efficient Embedding Storage Demonstration")
    print("This demo showcases various memory-efficient storage strategies")
    print("for large vocabulary embeddings in the LSM system.")
    
    try:
        # Run individual demos
        demo_memory_mapped_embeddings()
        demo_compressed_storage()
        demo_memory_efficient_layer()
        demo_sinusoidal_embedder_integration()
        demo_performance_comparison()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
        print("\nKey Features Demonstrated:")
        print("• Memory-mapped embeddings for large vocabularies")
        print("• Compressed storage with on-demand decompression")
        print("• Gradient checkpointing for memory-constrained training")
        print("• Automatic storage strategy selection")
        print("• Integration with sinusoidal embeddings")
        print("• Configuration serialization and loading")
        print("• Performance comparison between strategies")
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()