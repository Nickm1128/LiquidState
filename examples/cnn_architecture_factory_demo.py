#!/usr/bin/env python3
"""
CNN Architecture Factory Demo

This script demonstrates how to use the CNNArchitectureFactory to create
different types of CNN models for the LSM training pipeline enhancement.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf
from src.lsm.core.cnn_architecture_factory import (
    CNNArchitectureFactory,
    create_standard_2d_cnn,
    create_system_aware_3d_cnn,
    create_residual_cnn_model
)


def main():
    """Demonstrate CNN Architecture Factory usage."""
    print("CNN Architecture Factory Demo")
    print("=" * 50)
    
    # Initialize factory
    factory = CNNArchitectureFactory()
    
    # Configuration
    window_size = 16
    embedding_dim = 128
    system_dim = 64
    batch_size = 4
    
    print(f"Configuration:")
    print(f"  Window size: {window_size}")
    print(f"  Embedding dimension: {embedding_dim}")
    print(f"  System dimension: {system_dim}")
    print(f"  Batch size: {batch_size}")
    print()
    
    # Show supported types
    print("Supported architectures:", factory.get_supported_architectures())
    print("Supported attention types:", factory.get_supported_attention_types())
    print("Supported loss types:", factory.get_supported_loss_types())
    print()
    
    # 1. Create and test 2D CNN
    print("1. Creating 2D CNN with spatial attention...")
    model_2d = factory.create_2d_cnn(
        input_shape=(window_size, window_size, 1),
        output_dim=embedding_dim,
        use_attention=True,
        attention_type="spatial"
    )
    model_2d = factory.compile_model(model_2d, loss_type="cosine_similarity")
    
    print(f"   Model created: {model_2d.name}")
    print(f"   Parameters: {model_2d.count_params():,}")
    print(f"   Input shape: {model_2d.input_shape}")
    print(f"   Output shape: {model_2d.output_shape}")
    
    # Test with dummy data
    dummy_2d = np.random.random((batch_size, window_size, window_size, 1)).astype(np.float32)
    output_2d = model_2d(dummy_2d)
    print(f"   Test output shape: {output_2d.shape}")
    print()
    
    # 2. Create and test 3D CNN for system messages
    print("2. Creating 3D CNN for system message integration...")
    model_3d = factory.create_3d_cnn(
        input_shape=(window_size, window_size, window_size, 1),
        output_dim=embedding_dim,
        system_dim=system_dim
    )
    model_3d = factory.compile_model(model_3d, loss_type="cosine_similarity")
    
    print(f"   Model created: {model_3d.name}")
    print(f"   Parameters: {model_3d.count_params():,}")
    print(f"   Number of inputs: {len(model_3d.inputs)}")
    print(f"   Reservoir input shape: {model_3d.inputs[0].shape}")
    print(f"   System input shape: {model_3d.inputs[1].shape}")
    print(f"   Output shape: {model_3d.output_shape}")
    
    # Test with dummy data
    dummy_reservoir = np.random.random((batch_size, window_size, window_size, window_size, 1)).astype(np.float32)
    dummy_system = np.random.random((batch_size, system_dim)).astype(np.float32)
    output_3d = model_3d([dummy_reservoir, dummy_system])
    print(f"   Test output shape: {output_3d.shape}")
    print()
    
    # 3. Create and test residual CNN
    print("3. Creating Residual CNN with skip connections...")
    model_residual = factory.create_residual_cnn(
        input_shape=(window_size, window_size, 1),
        output_dim=embedding_dim,
        depth=3,
        use_attention=True
    )
    model_residual = factory.compile_model(model_residual, loss_type="cosine_similarity")
    
    print(f"   Model created: {model_residual.name}")
    print(f"   Parameters: {model_residual.count_params():,}")
    print(f"   Input shape: {model_residual.input_shape}")
    print(f"   Output shape: {model_residual.output_shape}")
    
    # Test with dummy data
    output_residual = model_residual(dummy_2d)
    print(f"   Test output shape: {output_residual.shape}")
    print()
    
    # 4. Create multi-scale CNN
    print("4. Creating Multi-scale CNN...")
    model_multiscale = factory.create_multi_scale_cnn(
        input_shape=(window_size, window_size, 1),
        output_dim=embedding_dim,
        scales=[3, 5, 7],
        use_attention=True
    )
    model_multiscale = factory.compile_model(model_multiscale, loss_type="cosine_similarity")
    
    print(f"   Model created: {model_multiscale.name}")
    print(f"   Parameters: {model_multiscale.count_params():,}")
    print(f"   Input shape: {model_multiscale.input_shape}")
    print(f"   Output shape: {model_multiscale.output_shape}")
    
    # Test with dummy data
    output_multiscale = model_multiscale(dummy_2d)
    print(f"   Test output shape: {output_multiscale.shape}")
    print()
    
    # 5. Demonstrate convenience functions
    print("5. Testing convenience functions...")
    
    # Standard 2D CNN
    standard_model = create_standard_2d_cnn(window_size, embedding_dim)
    print(f"   Standard 2D CNN: {standard_model.count_params():,} parameters")
    
    # System-aware 3D CNN
    system_model = create_system_aware_3d_cnn(window_size, embedding_dim, system_dim)
    print(f"   System-aware 3D CNN: {system_model.count_params():,} parameters")
    
    # Residual CNN
    residual_model = create_residual_cnn_model(window_size, embedding_dim, depth=2)
    print(f"   Residual CNN: {residual_model.count_params():,} parameters")
    print()
    
    # 6. Demonstrate different loss functions
    print("6. Testing different loss functions...")
    
    # Create a simple model for testing
    test_model = factory.create_2d_cnn(
        input_shape=(window_size, window_size, 1),
        output_dim=embedding_dim,
        use_attention=False
    )
    
    # Test MSE loss
    mse_model = factory.compile_model(test_model, loss_type="mse")
    print(f"   MSE loss model compiled successfully")
    
    # Test Huber loss
    huber_model = factory.compile_model(test_model, loss_type="huber")
    print(f"   Huber loss model compiled successfully")
    
    # Test cosine similarity loss
    cosine_model = factory.compile_model(test_model, loss_type="cosine_similarity")
    print(f"   Cosine similarity loss model compiled successfully")
    print()
    
    # 7. Demonstrate cosine similarity loss function
    print("7. Testing cosine similarity loss function...")
    
    # Create test vectors
    y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    y_pred_perfect = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
    y_pred_partial = tf.constant([[0.8, 0.2, 0.0], [0.1, 0.9, 0.0]], dtype=tf.float32)
    
    # Calculate losses
    loss_perfect = factory._cosine_similarity_loss(y_true, y_pred_perfect)
    loss_partial = factory._cosine_similarity_loss(y_true, y_pred_partial)
    
    print(f"   Perfect prediction loss: {loss_perfect.numpy()}")
    print(f"   Partial prediction loss: {loss_partial.numpy()}")
    print(f"   Loss difference: {(loss_partial - loss_perfect).numpy()}")
    
    # Calculate similarities
    sim_perfect = factory._cosine_similarity_metric(y_true, y_pred_perfect)
    sim_partial = factory._cosine_similarity_metric(y_true, y_pred_partial)
    
    print(f"   Perfect prediction similarity: {sim_perfect.numpy()}")
    print(f"   Partial prediction similarity: {sim_partial.numpy()}")
    print()
    
    print("Demo completed successfully!")
    print("The CNNArchitectureFactory provides flexible CNN creation for:")
    print("- 2D CNNs with various attention mechanisms")
    print("- 3D CNNs for system message integration")
    print("- Residual CNNs with skip connections")
    print("- Multi-scale CNNs for different feature scales")
    print("- Multiple loss functions including cosine similarity")
    print("- Easy model compilation and configuration")


if __name__ == "__main__":
    # Suppress TensorFlow warnings for cleaner output
    tf.get_logger().setLevel('ERROR')
    
    main()