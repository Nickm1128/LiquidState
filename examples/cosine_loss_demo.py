#!/usr/bin/env python3
"""
Demonstration of cosine similarity loss function for LSM training pipeline.

This script shows how to use the enhanced cosine similarity loss functions
with both 2D and 3D CNN architectures for response-level training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from src.lsm.core.loss_functions import (
    CosineSimilarityLoss,
    ResponseLevelCosineLoss,
    CNNLossCalculator,
    create_cosine_similarity_loss,
    create_response_level_loss,
    get_loss_for_architecture
)

def demonstrate_basic_cosine_loss():
    """Demonstrate basic cosine similarity loss functionality."""
    print("=== Basic Cosine Similarity Loss Demo ===\n")
    
    # Create sample embeddings
    batch_size = 4
    embedding_dim = 8
    
    np.random.seed(42)
    y_true = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    y_pred = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    
    print(f"Input shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")
    
    # Standard cosine similarity loss
    loss_fn = CosineSimilarityLoss()
    loss = loss_fn(y_true, y_pred)
    print(f"Standard cosine loss: {loss.numpy():.4f}")
    
    # With temperature scaling
    loss_fn_temp = CosineSimilarityLoss(temperature=0.5)
    loss_temp = loss_fn_temp(y_true, y_pred)
    print(f"Temperature-scaled loss (T=0.5): {loss_temp.numpy():.4f}")
    
    # With margin
    loss_fn_margin = CosineSimilarityLoss(margin=0.1)
    margin_loss = loss_fn_margin.compute_margin_loss(y_true, y_pred)
    print(f"Margin-based loss (margin=0.1): {margin_loss.numpy():.4f}")
    
    # Perfect similarity case
    perfect_loss = loss_fn(y_true, y_true)
    print(f"Perfect similarity loss: {perfect_loss.numpy():.6f}")
    
    # Opposite vectors case
    opposite_loss = loss_fn(y_true, -y_true)
    print(f"Opposite vectors loss: {opposite_loss.numpy():.4f}")
    
    print()

def demonstrate_response_level_loss():
    """Demonstrate response-level cosine similarity loss."""
    print("=== Response-Level Cosine Loss Demo ===\n")
    
    # Create sample response embeddings
    batch_size = 6
    embedding_dim = 16
    
    np.random.seed(123)
    y_true = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    y_pred = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    
    print(f"Response embeddings shape: {y_true.shape}")
    
    # Standard response-level loss
    response_loss_fn = ResponseLevelCosineLoss()
    response_loss = response_loss_fn(y_true, y_pred)
    print(f"Response-level loss: {response_loss.numpy():.4f}")
    
    # With different weights
    response_loss_fn_weighted = ResponseLevelCosineLoss(
        sequence_weight=2.0,
        coherence_weight=0.2,
        diversity_weight=0.1
    )
    weighted_loss = response_loss_fn_weighted(y_true, y_pred)
    print(f"Weighted response-level loss: {weighted_loss.numpy():.4f}")
    
    # Show individual components
    sequence_loss = response_loss_fn._compute_sequence_loss(y_true, y_pred)
    coherence_penalty = response_loss_fn._compute_coherence_penalty(y_pred)
    diversity_bonus = response_loss_fn._compute_diversity_bonus(y_pred)
    
    print(f"  - Sequence loss: {sequence_loss.numpy():.4f}")
    print(f"  - Coherence penalty: {coherence_penalty.numpy():.4f}")
    print(f"  - Diversity bonus: {diversity_bonus.numpy():.4f}")
    
    print()

def demonstrate_cnn_loss_calculator():
    """Demonstrate CNN loss calculator for 2D and 3D architectures."""
    print("=== CNN Loss Calculator Demo ===\n")
    
    calculator = CNNLossCalculator()
    
    # Create test data
    batch_size = 4
    embedding_dim = 12
    system_dim = 8
    
    np.random.seed(456)
    y_true = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    y_pred = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    system_context = tf.constant(np.random.randn(batch_size, system_dim).astype(np.float32))
    
    print(f"Main embeddings shape: {y_true.shape}")
    print(f"System context shape: {system_context.shape}")
    
    # 2D CNN loss calculation
    print("\n--- 2D CNN Architecture ---")
    loss_2d = calculator.calculate_loss_2d(y_true, y_pred, loss_type="cosine_similarity")
    print(f"2D CNN cosine loss: {loss_2d.numpy():.4f}")
    
    loss_2d_mse = calculator.calculate_loss_2d(y_true, y_pred, loss_type="mse")
    print(f"2D CNN MSE loss: {loss_2d_mse.numpy():.4f}")
    
    # 3D CNN loss calculation
    print("\n--- 3D CNN Architecture ---")
    loss_3d = calculator.calculate_loss_3d(y_true, y_pred, loss_type="cosine_similarity")
    print(f"3D CNN cosine loss (no system): {loss_3d.numpy():.4f}")
    
    loss_3d_system = calculator.calculate_loss_3d(
        y_true, y_pred, system_context, 
        loss_type="cosine_similarity", system_weight=0.1
    )
    print(f"3D CNN cosine loss (with system): {loss_3d_system.numpy():.4f}")
    
    # Show system context effect
    system_effect = abs(loss_3d_system.numpy() - loss_3d.numpy())
    print(f"System context effect: {system_effect:.4f}")
    
    # Different loss types for 3D
    print("\n--- Different Loss Types for 3D CNN ---")
    loss_types = ["cosine_similarity", "response_level_cosine", "huber"]
    for loss_type in loss_types:
        loss = calculator.calculate_loss_3d(
            y_true, y_pred, system_context, 
            loss_type=loss_type, system_weight=0.05
        )
        print(f"{loss_type}: {loss.numpy():.4f}")
    
    print()

def demonstrate_keras_integration():
    """Demonstrate integration with Keras models."""
    print("=== Keras Integration Demo ===\n")
    
    # Create a simple model for response-level training
    embedding_dim = 16
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(embedding_dim,)),
        tf.keras.layers.Dense(embedding_dim, activation='linear')
    ])
    
    print("Model architecture:")
    model.summary()
    
    # Use cosine similarity loss
    cosine_loss = CosineSimilarityLoss()
    model.compile(
        optimizer='adam',
        loss=cosine_loss,
        metrics=['mae']
    )
    
    # Create training data
    batch_size = 32
    np.random.seed(789)
    x_train = tf.random.normal((batch_size, embedding_dim))
    y_train = tf.random.normal((batch_size, embedding_dim))
    
    print(f"\nTraining data shape: {x_train.shape}")
    print("Training with cosine similarity loss...")
    
    # Train for a few epochs
    history = model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=8,
        verbose=1
    )
    
    print(f"\nFinal training loss: {history.history['loss'][-1]:.4f}")
    
    # Test prediction
    test_input = tf.random.normal((1, embedding_dim))
    prediction = model.predict(test_input, verbose=0)
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Prediction shape: {prediction.shape}")
    
    # Calculate loss on prediction
    test_loss = cosine_loss(test_input, prediction)
    print(f"Test loss: {test_loss.numpy():.4f}")
    
    print()

def demonstrate_convenience_functions():
    """Demonstrate convenience functions for easy usage."""
    print("=== Convenience Functions Demo ===\n")
    
    # Create loss functions using convenience functions
    print("Creating loss functions with convenience functions:")
    
    # Standard cosine loss
    cosine_loss = create_cosine_similarity_loss()
    print("✓ Standard cosine similarity loss created")
    
    # Temperature-scaled cosine loss
    temp_cosine_loss = create_cosine_similarity_loss(temperature=0.7, weight_factor=1.5)
    print("✓ Temperature-scaled cosine similarity loss created")
    
    # Response-level loss
    response_loss = create_response_level_loss(
        sequence_weight=1.5,
        coherence_weight=0.15,
        diversity_weight=0.08
    )
    print("✓ Response-level cosine loss created")
    
    # Architecture-specific loss functions
    loss_2d = get_loss_for_architecture("2d", "cosine_similarity")
    loss_3d = get_loss_for_architecture("3d", "cosine_similarity")
    print("✓ Architecture-specific loss functions created")
    
    # Test them with sample data
    batch_size = 4
    embedding_dim = 8
    
    np.random.seed(999)
    y_true = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    y_pred = tf.constant(np.random.randn(batch_size, embedding_dim).astype(np.float32))
    
    print(f"\nTesting with sample data (shape: {y_true.shape}):")
    print(f"Standard cosine loss: {cosine_loss(y_true, y_pred).numpy():.4f}")
    print(f"Temperature-scaled loss: {temp_cosine_loss(y_true, y_pred).numpy():.4f}")
    print(f"Response-level loss: {response_loss(y_true, y_pred).numpy():.4f}")
    print(f"2D architecture loss: {loss_2d(y_true, y_pred).numpy():.4f}")
    print(f"3D architecture loss: {loss_3d(y_true, y_pred).numpy():.4f}")
    
    print()

def main():
    """Run all demonstrations."""
    print("Cosine Similarity Loss Function Demonstration")
    print("=" * 50)
    print()
    
    demonstrate_basic_cosine_loss()
    demonstrate_response_level_loss()
    demonstrate_cnn_loss_calculator()
    demonstrate_keras_integration()
    demonstrate_convenience_functions()
    
    print("🎉 All demonstrations completed successfully!")
    print("\nKey Features Demonstrated:")
    print("✓ Basic cosine similarity loss with temperature scaling and margin")
    print("✓ Response-level loss with coherence and diversity components")
    print("✓ 2D and 3D CNN loss calculation with system context support")
    print("✓ Keras integration for training neural networks")
    print("✓ Convenience functions for easy loss function creation")
    print("✓ Support for different loss types (MSE, Huber, etc.)")

if __name__ == "__main__":
    main()