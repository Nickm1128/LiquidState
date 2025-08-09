#!/usr/bin/env python3
"""
Tests for CNN Architecture Factory.

This module tests the CNNArchitectureFactory class and its various
CNN creation methods including 2D, 3D, and residual architectures.
"""

import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from src.lsm.core.cnn_architecture_factory import (
    CNNArchitectureFactory,
    CNNArchitectureError,
    CNNType,
    AttentionType,
    LossType,
    create_standard_2d_cnn,
    create_system_aware_3d_cnn,
    create_residual_cnn_model
)


class TestCNNArchitectureFactory(unittest.TestCase):
    """Test cases for CNN Architecture Factory."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.factory = CNNArchitectureFactory()
        self.window_size = 8  # Small size for testing
        self.embedding_dim = 64
        self.system_dim = 32
        
    def test_factory_initialization(self):
        """Test factory initialization and supported types."""
        # Test supported architectures
        architectures = self.factory.get_supported_architectures()
        self.assertIn("2d", architectures)
        self.assertIn("3d", architectures)
        self.assertIn("residual_2d", architectures)
        
        # Test supported attention types
        attention_types = self.factory.get_supported_attention_types()
        self.assertIn("spatial", attention_types)
        self.assertIn("channel", attention_types)
        self.assertIn("none", attention_types)
        
        # Test supported loss types
        loss_types = self.factory.get_supported_loss_types()
        self.assertIn("mse", loss_types)
        self.assertIn("cosine_similarity", loss_types)
        self.assertIn("huber", loss_types)
    
    def test_create_2d_cnn_basic(self):
        """Test basic 2D CNN creation."""
        input_shape = (self.window_size, self.window_size, 1)
        
        model = self.factory.create_2d_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            use_attention=False
        )
        
        # Test model structure
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, *input_shape))
        self.assertEqual(model.output_shape, (None, self.embedding_dim))
        
        # Test with dummy data
        dummy_input = np.random.random((2, *input_shape)).astype(np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, self.embedding_dim))
    
    def test_create_2d_cnn_with_attention(self):
        """Test 2D CNN creation with attention mechanisms."""
        input_shape = (self.window_size, self.window_size, 1)
        
        # Test spatial attention
        model_spatial = self.factory.create_2d_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            use_attention=True,
            attention_type="spatial"
        )
        
        self.assertIsInstance(model_spatial, tf.keras.Model)
        
        # Test channel attention
        model_channel = self.factory.create_2d_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            use_attention=True,
            attention_type="channel"
        )
        
        self.assertIsInstance(model_channel, tf.keras.Model)
        
        # Test combined attention
        model_combined = self.factory.create_2d_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            use_attention=True,
            attention_type="spatial_channel"
        )
        
        self.assertIsInstance(model_combined, tf.keras.Model)
    
    def test_create_3d_cnn(self):
        """Test 3D CNN creation for system message integration."""
        input_shape = (self.window_size, self.window_size, self.window_size, 1)
        
        model = self.factory.create_3d_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            system_dim=self.system_dim
        )
        
        # Test model structure
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(len(model.inputs), 2)  # Reservoir + system inputs
        self.assertEqual(model.output_shape, (None, self.embedding_dim))
        
        # Test with dummy data
        dummy_reservoir = np.random.random((2, *input_shape)).astype(np.float32)
        dummy_system = np.random.random((2, self.system_dim)).astype(np.float32)
        output = model([dummy_reservoir, dummy_system])
        self.assertEqual(output.shape, (2, self.embedding_dim))
    
    def test_create_residual_cnn(self):
        """Test residual CNN creation."""
        input_shape = (self.window_size, self.window_size, 1)
        
        model = self.factory.create_residual_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            depth=2
        )
        
        # Test model structure
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, *input_shape))
        self.assertEqual(model.output_shape, (None, self.embedding_dim))
        
        # Test with dummy data
        dummy_input = np.random.random((2, *input_shape)).astype(np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, self.embedding_dim))
    
    def test_create_multi_scale_cnn(self):
        """Test multi-scale CNN creation."""
        input_shape = (self.window_size, self.window_size, 1)
        
        model = self.factory.create_multi_scale_cnn(
            input_shape=input_shape,
            output_dim=self.embedding_dim,
            scales=[3, 5],
            use_attention=True
        )
        
        # Test model structure
        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape, (None, *input_shape))
        self.assertEqual(model.output_shape, (None, self.embedding_dim))
        
        # Test with dummy data
        dummy_input = np.random.random((2, *input_shape)).astype(np.float32)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, self.embedding_dim))
    
    def test_compile_model_mse(self):
        """Test model compilation with MSE loss."""
        input_shape = (self.window_size, self.window_size, 1)
        model = self.factory.create_2d_cnn(input_shape, self.embedding_dim)
        
        compiled_model = self.factory.compile_model(
            model, 
            loss_type="mse",
            learning_rate=0.001
        )
        
        self.assertIsInstance(compiled_model, tf.keras.Model)
        self.assertAlmostEqual(compiled_model.optimizer.learning_rate.numpy(), 0.001, places=6)
    
    def test_compile_model_cosine_similarity(self):
        """Test model compilation with cosine similarity loss."""
        input_shape = (self.window_size, self.window_size, 1)
        model = self.factory.create_2d_cnn(input_shape, self.embedding_dim)
        
        compiled_model = self.factory.compile_model(
            model, 
            loss_type="cosine_similarity",
            learning_rate=0.002
        )
        
        self.assertIsInstance(compiled_model, tf.keras.Model)
        self.assertAlmostEqual(compiled_model.optimizer.learning_rate.numpy(), 0.002, places=6)
    
    def test_input_validation_2d(self):
        """Test input validation for 2D CNN creation."""
        # Test invalid input shape
        with self.assertRaises(CNNArchitectureError):
            self.factory.create_2d_cnn(
                input_shape=(10, 10),  # Missing channel dimension
                output_dim=self.embedding_dim
            )
        
        # Test invalid output dimension
        with self.assertRaises(CNNArchitectureError):
            self.factory.create_2d_cnn(
                input_shape=(10, 10, 1),
                output_dim=0  # Invalid output dimension
            )
        
        # Test invalid attention type
        with self.assertRaises(CNNArchitectureError):
            self.factory.create_2d_cnn(
                input_shape=(10, 10, 1),
                output_dim=self.embedding_dim,
                attention_type="invalid_attention"
            )
    
    def test_input_validation_3d(self):
        """Test input validation for 3D CNN creation."""
        # Test invalid input shape
        with self.assertRaises(CNNArchitectureError):
            self.factory.create_3d_cnn(
                input_shape=(10, 10, 10),  # Missing channel dimension
                output_dim=self.embedding_dim,
                system_dim=self.system_dim
            )
        
        # Test invalid system dimension
        with self.assertRaises(CNNArchitectureError):
            self.factory.create_3d_cnn(
                input_shape=(10, 10, 10, 1),
                output_dim=self.embedding_dim,
                system_dim=-1  # Invalid system dimension
            )
    
    def test_convenience_functions(self):
        """Test convenience functions for model creation."""
        # Test standard 2D CNN
        model_2d = create_standard_2d_cnn(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim
        )
        self.assertIsInstance(model_2d, tf.keras.Model)
        
        # Test system-aware 3D CNN
        model_3d = create_system_aware_3d_cnn(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            system_dim=self.system_dim
        )
        self.assertIsInstance(model_3d, tf.keras.Model)
        
        # Test residual CNN
        model_residual = create_residual_cnn_model(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim
        )
        self.assertIsInstance(model_residual, tf.keras.Model)
    
    def test_cosine_similarity_loss_function(self):
        """Test cosine similarity loss function."""
        # Create test vectors
        y_true = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]], dtype=tf.float32)
        
        # Test loss function
        loss = self.factory._cosine_similarity_loss(y_true, y_pred)
        
        # First prediction should have lower loss (perfect match)
        # Second prediction should have higher loss (partial match)
        self.assertLess(loss[0].numpy(), loss[1].numpy())
        
        # Test metric function
        similarity = self.factory._cosine_similarity_metric(y_true, y_pred)
        
        # First prediction should have higher similarity
        self.assertGreater(similarity[0].numpy(), similarity[1].numpy())
    
    def test_error_handling(self):
        """Test error handling in factory methods."""
        # Test with invalid configuration that should raise CNNArchitectureError
        with patch('tensorflow.keras.Model') as mock_model:
            mock_model.side_effect = Exception("TensorFlow error")
            
            with self.assertRaises(CNNArchitectureError) as context:
                self.factory.create_2d_cnn(
                    input_shape=(10, 10, 1),
                    output_dim=self.embedding_dim
                )
            
            self.assertIn("2D CNN", str(context.exception))
            self.assertIn("Model creation failed", str(context.exception))


if __name__ == '__main__':
    # Suppress TensorFlow warnings for cleaner test output
    tf.get_logger().setLevel('ERROR')
    
    unittest.main()