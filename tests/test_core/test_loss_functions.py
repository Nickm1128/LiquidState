#!/usr/bin/env python3
"""
Tests for enhanced loss functions module.

This module tests the cosine similarity loss functions and CNN loss calculators
for both 2D and 3D architectures.
"""

import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock

from src.lsm.core.loss_functions import (
    CosineSimilarityLoss,
    ResponseLevelCosineLoss,
    CNNLossCalculator,
    LossFunctionError,
    LossType,
    create_cosine_similarity_loss,
    create_response_level_loss,
    get_loss_for_architecture
)


class TestCosineSimilarityLoss(unittest.TestCase):
    """Test cases for CosineSimilarityLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = CosineSimilarityLoss()
        self.batch_size = 4
        self.embedding_dim = 8
        
        # Create test data
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
    
    def test_basic_cosine_similarity_loss(self):
        """Test basic cosine similarity loss computation."""
        loss = self.loss_fn(self.y_true, self.y_pred)
        
        # Loss should be a scalar tensor
        self.assertEqual(loss.shape, ())
        
        # Loss should be between 0 and 2 (1 - cosine_similarity range)
        self.assertGreaterEqual(loss.numpy(), 0.0)
        self.assertLessEqual(loss.numpy(), 2.0)
    
    def test_perfect_similarity(self):
        """Test loss when predictions perfectly match targets."""
        # Identical vectors should have cosine similarity of 1, loss of 0
        identical_pred = self.y_true
        loss = self.loss_fn(self.y_true, identical_pred)
        
        # Should be very close to 0
        self.assertLess(loss.numpy(), 1e-6)
    
    def test_opposite_vectors(self):
        """Test loss when predictions are opposite to targets."""
        # Opposite vectors should have cosine similarity of -1, loss of 2
        opposite_pred = -self.y_true
        loss = self.loss_fn(self.y_true, opposite_pred)
        
        # Should be close to 2
        self.assertGreater(loss.numpy(), 1.9)
        self.assertLess(loss.numpy(), 2.1)
    
    def test_temperature_scaling(self):
        """Test temperature scaling effect on loss."""
        loss_fn_temp = CosineSimilarityLoss(temperature=0.5)
        
        loss_normal = self.loss_fn(self.y_true, self.y_pred)
        loss_temp = loss_fn_temp(self.y_true, self.y_pred)
        
        # Temperature scaling should affect the loss value
        self.assertNotAlmostEqual(loss_normal.numpy(), loss_temp.numpy(), places=4)
    
    def test_margin_loss(self):
        """Test margin-based cosine similarity loss."""
        loss_fn_margin = CosineSimilarityLoss(margin=0.1)
        
        # Test with vectors that have high similarity
        similar_pred = self.y_true + 0.01 * tf.random.normal(self.y_true.shape)
        margin_loss = loss_fn_margin.compute_margin_loss(self.y_true, similar_pred)
        
        # Margin loss should be non-negative
        self.assertGreaterEqual(margin_loss.numpy(), 0.0)
    
    def test_weighted_loss(self):
        """Test weighted cosine similarity loss."""
        # Create sample weights
        sample_weights = tf.constant([1.0, 2.0, 0.5, 1.5])
        
        weighted_loss = self.loss_fn.compute_weighted_loss(
            self.y_true, self.y_pred, sample_weights
        )
        unweighted_loss = self.loss_fn(self.y_true, self.y_pred)
        
        # Weighted loss should be different from unweighted
        self.assertNotAlmostEqual(weighted_loss.numpy(), unweighted_loss.numpy(), places=4)
    
    def test_reduction_methods(self):
        """Test different reduction methods."""
        loss_fn_none = CosineSimilarityLoss(reduction='none')
        loss_fn_sum = CosineSimilarityLoss(reduction='sum')
        loss_fn_mean = CosineSimilarityLoss(reduction='mean')
        
        loss_none = loss_fn_none(self.y_true, self.y_pred)
        loss_sum = loss_fn_sum(self.y_true, self.y_pred)
        loss_mean = loss_fn_mean(self.y_true, self.y_pred)
        
        # Check shapes
        self.assertEqual(loss_none.shape, (self.batch_size,))
        self.assertEqual(loss_sum.shape, ())
        self.assertEqual(loss_mean.shape, ())
        
        # Check relationships
        self.assertAlmostEqual(
            loss_sum.numpy(), 
            tf.reduce_sum(loss_none).numpy(), 
            places=5
        )
        self.assertAlmostEqual(
            loss_mean.numpy(), 
            tf.reduce_mean(loss_none).numpy(), 
            places=5
        )
    
    def test_invalid_parameters(self):
        """Test error handling for invalid parameters."""
        with self.assertRaises(ValueError):
            CosineSimilarityLoss(temperature=0.0)
        
        with self.assertRaises(ValueError):
            CosineSimilarityLoss(epsilon=0.0)
        
        with self.assertRaises(ValueError):
            CosineSimilarityLoss(reduction='invalid')


class TestResponseLevelCosineLoss(unittest.TestCase):
    """Test cases for ResponseLevelCosineLoss class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss_fn = ResponseLevelCosineLoss()
        self.batch_size = 4
        self.embedding_dim = 8
        
        # Create test data
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
    
    def test_response_level_loss_computation(self):
        """Test response-level loss computation."""
        loss = self.loss_fn(self.y_true, self.y_pred)
        
        # Loss should be a scalar tensor
        self.assertEqual(loss.shape, ())
        
        # Loss should be finite
        self.assertTrue(tf.math.is_finite(loss))
    
    def test_coherence_penalty(self):
        """Test coherence penalty computation."""
        coherence_penalty = self.loss_fn._compute_coherence_penalty(self.y_pred)
        
        # Should be a scalar and non-negative
        self.assertEqual(coherence_penalty.shape, ())
        self.assertGreaterEqual(coherence_penalty.numpy(), 0.0)
    
    def test_diversity_bonus(self):
        """Test diversity bonus computation."""
        diversity_bonus = self.loss_fn._compute_diversity_bonus(self.y_pred)
        
        # Should be a scalar and non-negative
        self.assertEqual(diversity_bonus.shape, ())
        self.assertGreaterEqual(diversity_bonus.numpy(), 0.0)
    
    def test_weight_effects(self):
        """Test effect of different weight parameters."""
        loss_fn_high_coherence = ResponseLevelCosineLoss(coherence_weight=1.0)
        loss_fn_high_diversity = ResponseLevelCosineLoss(diversity_weight=1.0)
        
        loss_normal = self.loss_fn(self.y_true, self.y_pred)
        loss_coherence = loss_fn_high_coherence(self.y_true, self.y_pred)
        loss_diversity = loss_fn_high_diversity(self.y_true, self.y_pred)
        
        # Different weights should produce different losses
        self.assertNotAlmostEqual(loss_normal.numpy(), loss_coherence.numpy(), places=3)
        self.assertNotAlmostEqual(loss_normal.numpy(), loss_diversity.numpy(), places=3)


class TestCNNLossCalculator(unittest.TestCase):
    """Test cases for CNNLossCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.calculator = CNNLossCalculator()
        self.batch_size = 4
        self.embedding_dim = 8
        self.system_dim = 6
        
        # Create test data
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
        self.system_context = tf.constant(
            np.random.randn(self.batch_size, self.system_dim).astype(np.float32)
        )
    
    def test_2d_cnn_loss_calculation(self):
        """Test 2D CNN loss calculation."""
        loss = self.calculator.calculate_loss_2d(
            self.y_true, self.y_pred, loss_type="cosine_similarity"
        )
        
        # Should return a scalar loss
        self.assertEqual(loss.shape, ())
        self.assertTrue(tf.math.is_finite(loss))
    
    def test_3d_cnn_loss_calculation(self):
        """Test 3D CNN loss calculation."""
        loss = self.calculator.calculate_loss_3d(
            self.y_true, self.y_pred, loss_type="cosine_similarity"
        )
        
        # Should return a scalar loss
        self.assertEqual(loss.shape, ())
        self.assertTrue(tf.math.is_finite(loss))
    
    def test_3d_cnn_with_system_context(self):
        """Test 3D CNN loss calculation with system context."""
        loss_without_system = self.calculator.calculate_loss_3d(
            self.y_true, self.y_pred, loss_type="cosine_similarity"
        )
        
        loss_with_system = self.calculator.calculate_loss_3d(
            self.y_true, self.y_pred, self.system_context, 
            loss_type="cosine_similarity", system_weight=0.1
        )
        
        # System context should affect the loss
        self.assertNotAlmostEqual(
            loss_without_system.numpy(), 
            loss_with_system.numpy(), 
            places=4
        )
    
    def test_different_loss_types(self):
        """Test different loss function types."""
        loss_types = ["cosine_similarity", "mse", "huber", "response_level_cosine"]
        
        for loss_type in loss_types:
            with self.subTest(loss_type=loss_type):
                loss = self.calculator.calculate_loss_2d(
                    self.y_true, self.y_pred, loss_type=loss_type
                )
                
                # All losses should be finite scalars
                self.assertEqual(loss.shape, ())
                self.assertTrue(tf.math.is_finite(loss))
    
    def test_get_loss_function(self):
        """Test getting loss function by type."""
        loss_fn = self.calculator.get_loss_function("cosine_similarity")
        
        # Should be callable
        self.assertTrue(callable(loss_fn))
        
        # Should work when called
        loss = loss_fn(self.y_true, self.y_pred)
        self.assertTrue(tf.math.is_finite(loss))
    
    def test_supported_loss_types(self):
        """Test getting supported loss types."""
        supported_types = self.calculator.get_supported_loss_types()
        
        # Should include expected types
        expected_types = [
            "mse", "cosine_similarity", "cosine_similarity_weighted",
            "cosine_similarity_temperature", "cosine_similarity_margin",
            "response_level_cosine", "huber"
        ]
        
        for expected_type in expected_types:
            self.assertIn(expected_type, supported_types)
    
    def test_input_validation_2d(self):
        """Test input validation for 2D CNN loss."""
        # Test None inputs
        with self.assertRaises(LossFunctionError):
            self.calculator.calculate_loss_2d(None, self.y_pred)
        
        # Test shape mismatch
        wrong_shape_pred = tf.constant(np.random.randn(2, 4).astype(np.float32))
        with self.assertRaises(LossFunctionError):
            self.calculator.calculate_loss_2d(self.y_true, wrong_shape_pred)
        
        # Test wrong dimensions
        wrong_dim_true = tf.constant(np.random.randn(4, 8, 2).astype(np.float32))
        with self.assertRaises(LossFunctionError):
            self.calculator.calculate_loss_2d(wrong_dim_true, self.y_pred)
    
    def test_input_validation_3d(self):
        """Test input validation for 3D CNN loss."""
        # Test system context batch size mismatch
        wrong_batch_system = tf.constant(np.random.randn(2, 6).astype(np.float32))
        with self.assertRaises(LossFunctionError):
            self.calculator.calculate_loss_3d(
                self.y_true, self.y_pred, wrong_batch_system
            )
        
        # Test wrong system context dimensions
        wrong_dim_system = tf.constant(np.random.randn(4, 6, 2).astype(np.float32))
        with self.assertRaises(LossFunctionError):
            self.calculator.calculate_loss_3d(
                self.y_true, self.y_pred, wrong_dim_system
            )
    
    def test_unsupported_loss_type(self):
        """Test error handling for unsupported loss types."""
        with self.assertRaises(LossFunctionError):
            self.calculator.calculate_loss_2d(
                self.y_true, self.y_pred, loss_type="unsupported_loss"
            )
    
    def test_system_context_penalty(self):
        """Test system context penalty computation."""
        penalty = self.calculator._compute_system_context_penalty(
            self.y_pred, self.system_context
        )
        
        # Should be a scalar and non-negative
        self.assertEqual(penalty.shape, ())
        self.assertGreaterEqual(penalty.numpy(), 0.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions."""
    
    def test_create_cosine_similarity_loss(self):
        """Test cosine similarity loss creation function."""
        loss_fn = create_cosine_similarity_loss(temperature=0.5, margin=0.1)
        
        # Should be a CosineSimilarityLoss instance
        self.assertIsInstance(loss_fn, CosineSimilarityLoss)
        
        # Should have correct parameters
        self.assertEqual(loss_fn.temperature, 0.5)
        self.assertEqual(loss_fn.margin, 0.1)
    
    def test_create_response_level_loss(self):
        """Test response-level loss creation function."""
        loss_fn = create_response_level_loss(
            sequence_weight=2.0, coherence_weight=0.2
        )
        
        # Should be a ResponseLevelCosineLoss instance
        self.assertIsInstance(loss_fn, ResponseLevelCosineLoss)
        
        # Should have correct parameters
        self.assertEqual(loss_fn.sequence_weight, 2.0)
        self.assertEqual(loss_fn.coherence_weight, 0.2)
    
    def test_get_loss_for_architecture(self):
        """Test getting loss function for architecture type."""
        # Test 2D architecture
        loss_fn_2d = get_loss_for_architecture("2d", "cosine_similarity")
        self.assertTrue(callable(loss_fn_2d))
        
        # Test 3D architecture
        loss_fn_3d = get_loss_for_architecture("3d", "cosine_similarity")
        self.assertTrue(callable(loss_fn_3d))
        
        # Test unsupported architecture
        with self.assertRaises(ValueError):
            get_loss_for_architecture("4d", "cosine_similarity")


class TestLossIntegration(unittest.TestCase):
    """Integration tests for loss functions with TensorFlow/Keras."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 8
        self.embedding_dim = 16
        
        # Create test data
        np.random.seed(42)
        self.y_true = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
        self.y_pred = tf.constant(
            np.random.randn(self.batch_size, self.embedding_dim).astype(np.float32)
        )
    
    def test_keras_compatibility(self):
        """Test compatibility with Keras training loop."""
        # Create a simple model
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.embedding_dim, input_shape=(self.embedding_dim,))
        ])
        
        # Use cosine similarity loss
        loss_fn = CosineSimilarityLoss()
        
        # Compile model with custom loss
        model.compile(optimizer='adam', loss=loss_fn)
        
        # Test that it works with fit (single step)
        try:
            model.fit(
                self.y_true, self.y_true, 
                epochs=1, batch_size=4, verbose=0
            )
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success, "Model should train with cosine similarity loss")
    
    def test_gradient_computation(self):
        """Test that gradients can be computed through the loss function."""
        loss_fn = CosineSimilarityLoss()
        
        # Create trainable variables
        pred_var = tf.Variable(self.y_pred)
        
        with tf.GradientTape() as tape:
            loss = loss_fn(self.y_true, pred_var)
        
        # Compute gradients
        gradients = tape.gradient(loss, pred_var)
        
        # Gradients should not be None and should have correct shape
        self.assertIsNotNone(gradients)
        self.assertEqual(gradients.shape, pred_var.shape)
        
        # Gradients should not be all zeros (unless perfect match)
        self.assertGreater(tf.reduce_sum(tf.abs(gradients)).numpy(), 1e-6)


if __name__ == '__main__':
    unittest.main()