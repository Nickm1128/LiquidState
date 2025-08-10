#!/usr/bin/env python3
"""
Demo script for ConfigurableSinusoidalEmbedder.

This script demonstrates the usage of the configurable sinusoidal embedding layer
with different configurations and features.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib and seaborn not available - visualization demos will be skipped")

from src.lsm.data.configurable_sinusoidal_embedder import (
    SinusoidalConfig,
    ConfigurableSinusoidalEmbedder,
    SinusoidalEmbedderFactory
)


def demo_basic_usage():
    """Demonstrate basic usage of ConfigurableSinusoidalEmbedder."""
    print("=== Basic Usage Demo ===")
    
    # Create a basic configuration
    config = SinusoidalConfig(
        embedding_dim=128,
        vocab_size=5000,
        max_sequence_length=256,
        learnable_frequencies=True
    )
    
    # Create embedder
    embedder = ConfigurableSinusoidalEmbedder(config)
    
    # Create sample input
    batch_size, seq_length = 4, 20
    inputs = tf.random.uniform((batch_size, seq_length), maxval=5000, dtype=tf.int32)
    
    # Build and apply embedder
    embedder.build((None, seq_length))
    embeddings = embedder(inputs)
    
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {embeddings.shape}")
    print(f"Embedding range: [{tf.reduce_min(embeddings):.3f}, {tf.reduce_max(embeddings):.3f}]")
    print()


def demo_factory_methods():
    """Demonstrate factory methods for creating embedders."""
    print("=== Factory Methods Demo ===")
    
    vocab_size = 10000
    embedding_dim = 256
    
    # Default embedder
    default_embedder = SinusoidalEmbedderFactory.create_default(vocab_size, embedding_dim)
    print(f"Default embedder - Learnable frequencies: {default_embedder.config.learnable_frequencies}")
    
    # Relative position embedder
    relative_embedder = SinusoidalEmbedderFactory.create_relative_position(
        vocab_size, embedding_dim, relative_window=64
    )
    print(f"Relative embedder - Use relative position: {relative_embedder.config.use_relative_position}")
    
    # Fixed frequency embedder
    fixed_embedder = SinusoidalEmbedderFactory.create_fixed_frequency(
        vocab_size, embedding_dim, base_frequency=5000.0
    )
    print(f"Fixed embedder - Learnable frequencies: {fixed_embedder.config.learnable_frequencies}")
    print(f"Fixed embedder - Base frequency: {fixed_embedder.config.base_frequency}")
    
    # High performance embedder
    hp_embedder = SinusoidalEmbedderFactory.create_high_performance(vocab_size, embedding_dim)
    print(f"HP embedder - Mixed precision: {hp_embedder.config.use_mixed_precision}")
    print(f"HP embedder - Gradient checkpointing: {hp_embedder.config.gradient_checkpointing}")
    print()


def demo_vocabulary_adaptation():
    """Demonstrate vocabulary size adaptation."""
    print("=== Vocabulary Adaptation Demo ===")
    
    # Create embedder with initial vocabulary
    embedder = SinusoidalEmbedderFactory.create_default(vocab_size=1000, embedding_dim=64)
    print(f"Initial vocab size: {embedder.config.vocab_size}")
    
    # Build with sample input
    inputs = tf.random.uniform((2, 10), maxval=1000, dtype=tf.int32)
    embedder.build((None, 10))
    initial_output = embedder(inputs)
    print(f"Initial output shape: {initial_output.shape}")
    
    # Adapt to larger vocabulary
    new_vocab_size = 5000
    embedder.adapt_to_vocabulary(new_vocab_size)
    print(f"Adapted vocab size: {embedder.config.vocab_size}")
    
    # Test with new vocabulary range
    new_inputs = tf.random.uniform((2, 10), maxval=5000, dtype=tf.int32)
    adapted_output = embedder(new_inputs)
    print(f"Adapted output shape: {adapted_output.shape}")
    print()


def demo_positional_encoding_comparison():
    """Compare absolute and relative positional encodings."""
    print("=== Positional Encoding Comparison ===")
    
    vocab_size = 1000
    embedding_dim = 64
    seq_length = 20
    
    # Absolute position only
    abs_config = SinusoidalConfig(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        use_absolute_position=True,
        use_relative_position=False
    )
    abs_embedder = ConfigurableSinusoidalEmbedder(abs_config)
    
    # Both absolute and relative
    both_config = SinusoidalConfig(
        embedding_dim=embedding_dim,
        vocab_size=vocab_size,
        use_absolute_position=True,
        use_relative_position=True,
        relative_position_window=32
    )
    both_embedder = ConfigurableSinusoidalEmbedder(both_config)
    
    # Test with same input
    inputs = tf.constant([[1, 2, 3, 4, 5] * 4], dtype=tf.int32)  # Repeated pattern
    
    abs_embedder.build((None, seq_length))
    both_embedder.build((None, seq_length))
    
    abs_output = abs_embedder(inputs)
    both_output = both_embedder(inputs)
    
    print(f"Absolute only output shape: {abs_output.shape}")
    print(f"Both encodings output shape: {both_output.shape}")
    
    # Compare similarity between positions
    abs_similarity = tf.linalg.matmul(abs_output[0], abs_output[0], transpose_b=True)
    both_similarity = tf.linalg.matmul(both_output[0], both_output[0], transpose_b=True)
    
    print(f"Absolute encoding - avg similarity: {tf.reduce_mean(abs_similarity):.3f}")
    print(f"Both encodings - avg similarity: {tf.reduce_mean(both_similarity):.3f}")
    print()


def demo_embedding_patterns():
    """Demonstrate embedding pattern analysis."""
    print("=== Embedding Patterns Demo ===")
    
    # Create embedder with specific configuration
    config = SinusoidalConfig(
        embedding_dim=64,
        vocab_size=1000,
        base_frequency=1000.0,
        frequency_scaling=2.0,
        learnable_frequencies=False  # Use fixed for consistent patterns
    )
    embedder = ConfigurableSinusoidalEmbedder(config)
    embedder.build((None, 50))
    
    # Get embedding patterns
    patterns = embedder.get_embedding_patterns(max_positions=50)
    
    print(f"Positional encoding shape: {patterns['positional_encoding'].shape}")
    print(f"Number of frequencies: {len(patterns['frequencies'])}")
    print(f"Frequency range: [{patterns['frequencies'].min():.6f}, {patterns['frequencies'].max():.6f}]")
    print(f"Base frequency: {patterns['base_frequency']}")
    print(f"Frequency scaling: {patterns['frequency_scaling']}")
    
    # Analyze pattern properties
    encoding = patterns['positional_encoding']
    
    # Check orthogonality between positions
    pos_0 = encoding[0]
    pos_1 = encoding[1]
    pos_10 = encoding[10]
    
    dot_01 = np.dot(pos_0, pos_1)
    dot_010 = np.dot(pos_0, pos_10)
    
    print(f"Dot product (pos 0, pos 1): {dot_01:.3f}")
    print(f"Dot product (pos 0, pos 10): {dot_010:.3f}")
    print()


def demo_training_integration():
    """Demonstrate integration with model training."""
    print("=== Training Integration Demo ===")
    
    # Create embedder
    config = SinusoidalConfig(
        embedding_dim=32,
        vocab_size=1000,
        learnable_frequencies=True
    )
    embedder = ConfigurableSinusoidalEmbedder(config)
    
    # Create a simple classification model
    inputs = keras.layers.Input(shape=(20,), dtype=tf.int32)
    embeddings = embedder(inputs)
    pooled = keras.layers.GlobalAveragePooling1D()(embeddings)
    outputs = keras.layers.Dense(2, activation='softmax')(pooled)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Generate dummy data
    x_train = tf.random.uniform((100, 20), maxval=1000, dtype=tf.int32)
    y_train = tf.random.uniform((100,), maxval=2, dtype=tf.int32)
    
    # Train for a few epochs
    print("Training model...")
    history = model.fit(x_train, y_train, epochs=3, verbose=0)
    
    print(f"Final loss: {history.history['loss'][-1]:.4f}")
    print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
    
    # Check if frequencies were learned
    initial_frequencies = embedder._compute_fixed_frequencies()
    learned_frequencies = embedder.frequency_weights.numpy()
    
    freq_change = np.mean(np.abs(learned_frequencies - initial_frequencies))
    print(f"Average frequency change: {freq_change:.6f}")
    print()


def demo_save_load_config():
    """Demonstrate saving and loading configurations."""
    print("=== Save/Load Configuration Demo ===")
    
    # Create embedder with custom configuration
    config = SinusoidalConfig(
        embedding_dim=128,
        vocab_size=5000,
        base_frequency=8000.0,
        frequency_scaling=1.5,
        learnable_frequencies=True,
        use_relative_position=True,
        relative_position_window=64
    )
    original_embedder = ConfigurableSinusoidalEmbedder(config)
    
    # Save configuration
    config_path = "temp_embedder_config.json"
    original_embedder.save_config(config_path)
    print(f"Configuration saved to {config_path}")
    
    # Load configuration
    loaded_embedder = ConfigurableSinusoidalEmbedder.load_config(config_path)
    print(f"Configuration loaded from {config_path}")
    
    # Verify configurations match
    print(f"Original embedding_dim: {original_embedder.config.embedding_dim}")
    print(f"Loaded embedding_dim: {loaded_embedder.config.embedding_dim}")
    print(f"Original base_frequency: {original_embedder.config.base_frequency}")
    print(f"Loaded base_frequency: {loaded_embedder.config.base_frequency}")
    
    # Clean up
    import os
    os.remove(config_path)
    print(f"Cleaned up {config_path}")
    print()


def demo_visualization():
    """Demonstrate embedding visualization (if matplotlib available)."""
    print("=== Visualization Demo ===")
    
    if not VISUALIZATION_AVAILABLE:
        print("Matplotlib and seaborn not available - skipping visualization demo")
        print()
        return
    
    try:
        # Create embedder
        config = SinusoidalConfig(
            embedding_dim=64,
            vocab_size=1000,
            learnable_frequencies=False,  # Fixed for consistent visualization
            base_frequency=100.0
        )
        embedder = ConfigurableSinusoidalEmbedder(config)
        embedder.build((None, 50))
        
        print("Creating embedding visualization...")
        embedder.visualize_embeddings(max_positions=50, save_path="embedder_patterns.png")
        print("Visualization saved as 'embedder_patterns.png'")
        
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print()


def main():
    """Run all demos."""
    print("ConfigurableSinusoidalEmbedder Demo")
    print("=" * 50)
    print()
    
    demo_basic_usage()
    demo_factory_methods()
    demo_vocabulary_adaptation()
    demo_positional_encoding_comparison()
    demo_embedding_patterns()
    demo_training_integration()
    demo_save_load_config()
    demo_visualization()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()