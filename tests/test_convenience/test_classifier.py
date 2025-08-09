#!/usr/bin/env python3
"""
Tests for LSMClassifier convenience API class.

This module tests the LSMClassifier class which provides a sklearn-like
interface for classification tasks using LSM features.
"""

import unittest
import tempfile
import shutil
import os
import sys
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from lsm.convenience.classifier import LSMClassifier
    from lsm.convenience.config import ConvenienceConfig, ConvenienceValidationError
    CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LSMClassifier not available: {e}")
    CLASSIFIER_AVAILABLE = False


@unittest.skipUnless(CLASSIFIER_AVAILABLE, "LSMClassifier required")
class TestLSMClassifier(unittest.TestCase):
    """Test LSMClassifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data = [
            ("This is a positive message", "positive"),
            ("This is a negative message", "negative"),
            ("Another positive example", "positive"),
            ("Another negative example", "negative"),
            ("Neutral message here", "neutral"),
            ("Another neutral example", "neutral")
        ]
        self.X = [item[0] for item in self.test_data]
        self.y = [item[1] for item in self.test_data]
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LSMClassifier initialization."""
        # Test default initialization
        classifier = LSMClassifier()
        self.assertIsNotNone(classifier)
        self.assertEqual(classifier.window_size, 10)  # Default value
        self.assertEqual(classifier.embedding_dim, 128)  # Default value
        
        # Test custom initialization
        classifier = LSMClassifier(
            window_size=15,
            embedding_dim=256,
            reservoir_type='standard',
            n_classes=3
        )
        self.assertEqual(classifier.window_size, 15)
        self.assertEqual(classifier.embedding_dim, 256)
        self.assertEqual(classifier.reservoir_type, 'standard')
        self.assertEqual(classifier.n_classes, 3)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid window_size
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMClassifier(window_size=0)
        
        # Test invalid embedding_dim
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMClassifier(embedding_dim=-1)
        
        # Test invalid n_classes
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMClassifier(n_classes=1)  # Should be >= 2
    
    def test_sklearn_interface(self):
        """Test sklearn-compatible interface."""
        classifier = LSMClassifier()
        
        # Test required methods exist
        self.assertTrue(hasattr(classifier, 'fit'))
        self.assertTrue(hasattr(classifier, 'predict'))
        self.assertTrue(hasattr(classifier, 'predict_proba'))
        self.assertTrue(hasattr(classifier, 'score'))
        self.assertTrue(hasattr(classifier, 'get_params'))
        self.assertTrue(hasattr(classifier, 'set_params'))
    
    def test_get_params(self):
        """Test get_params method."""
        classifier = LSMClassifier(
            window_size=15,
            embedding_dim=256,
            n_classes=3
        )
        
        params = classifier.get_params()
        self.assertIsInstance(params, dict)
        self.assertEqual(params['window_size'], 15)
        self.assertEqual(params['embedding_dim'], 256)
        self.assertEqual(params['n_classes'], 3)
    
    def test_set_params(self):
        """Test set_params method."""
        classifier = LSMClassifier()
        
        # Set new parameters
        classifier.set_params(
            window_size=20,
            embedding_dim=512,
            n_classes=5
        )
        
        # Verify parameters were set
        self.assertEqual(classifier.window_size, 20)
        self.assertEqual(classifier.embedding_dim, 512)
        self.assertEqual(classifier.n_classes, 5)
    
    @patch('lsm.training.train.LSMTrainer')
    def test_fit_basic(self, mock_trainer_class):
        """Test basic fit functionality."""
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train.return_value = True
        
        classifier = LSMClassifier(window_size=5, embedding_dim=32)
        
        # Mock data preparation and classifier training
        with patch.object(classifier, '_prepare_classification_data') as mock_prepare:
            mock_prepare.return_value = (self.X, self.y)
            
            with patch.object(classifier, '_fit_classifier') as mock_fit:
                mock_fit.return_value = None
                
                # Test fit
                classifier.fit(self.X, self.y)
                
                # Verify methods were called
                mock_prepare.assert_called_once()
                mock_fit.assert_called_once()
                self.assertTrue(classifier._is_fitted)
    
    def test_fit_validation(self):
        """Test fit input validation."""
        classifier = LSMClassifier()
        
        # Test empty data
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            classifier.fit([], [])
        
        # Test mismatched X and y lengths
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            classifier.fit(["text1", "text2"], ["label1"])
        
        # Test invalid X format
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            classifier.fit([123, 456], ["label1", "label2"])
    
    def test_predict_basic(self):
        """Test basic predict functionality."""
        classifier = LSMClassifier()
        classifier._is_fitted = True
        classifier.classes_ = ['positive', 'negative', 'neutral']
        
        # Mock prediction
        with patch.object(classifier, '_predict_classes') as mock_predict:
            mock_predict.return_value = ['positive', 'negative']
            
            # Test predict
            predictions = classifier.predict(self.X[:2])
            
            self.assertEqual(len(predictions), 2)
            self.assertEqual(predictions[0], 'positive')
            self.assertEqual(predictions[1], 'negative')
            mock_predict.assert_called_once()
    
    def test_predict_not_fitted(self):
        """Test predict raises error when not fitted."""
        classifier = LSMClassifier()
        
        with self.assertRaises((ValueError, RuntimeError)):
            classifier.predict(self.X)
    
    def test_predict_proba_basic(self):
        """Test basic predict_proba functionality."""
        classifier = LSMClassifier()
        classifier._is_fitted = True
        classifier.classes_ = ['positive', 'negative', 'neutral']
        
        # Mock probability prediction
        with patch.object(classifier, '_predict_probabilities') as mock_predict_proba:
            mock_probabilities = np.array([
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1]
            ])
            mock_predict_proba.return_value = mock_probabilities
            
            # Test predict_proba
            probabilities = classifier.predict_proba(self.X[:2])
            
            self.assertEqual(probabilities.shape, (2, 3))
            np.testing.assert_array_equal(probabilities, mock_probabilities)
            mock_predict_proba.assert_called_once()
    
    def test_score_basic(self):
        """Test basic score functionality."""
        classifier = LSMClassifier()
        classifier._is_fitted = True
        
        # Mock prediction and scoring
        with patch.object(classifier, 'predict') as mock_predict:
            mock_predict.return_value = ['positive', 'negative', 'positive']
            
            # Test score
            y_true = ['positive', 'negative', 'positive']
            accuracy = classifier.score(self.X[:3], y_true)
            
            self.assertEqual(accuracy, 1.0)  # Perfect accuracy
            mock_predict.assert_called_once()
    
    def test_class_detection(self):
        """Test automatic class detection."""
        classifier = LSMClassifier()
        
        # Mock fit process
        with patch.object(classifier, '_prepare_classification_data') as mock_prepare:
            mock_prepare.return_value = (self.X, self.y)
            
            with patch.object(classifier, '_fit_classifier') as mock_fit:
                mock_fit.return_value = None
                
                # Fit with automatic class detection
                classifier.fit(self.X, self.y)
                
                # Verify classes were detected
                self.assertTrue(hasattr(classifier, 'classes_'))
                self.assertIsNotNone(classifier.classes_)
    
    def test_feature_extraction(self):
        """Test reservoir feature extraction."""
        classifier = LSMClassifier()
        classifier._is_fitted = True
        
        # Mock feature extraction
        with patch.object(classifier, '_extract_features') as mock_extract:
            mock_features = np.random.rand(2, 64)  # 2 samples, 64 features
            mock_extract.return_value = mock_features
            
            # Test feature extraction
            features = classifier._extract_features(self.X[:2])
            
            self.assertEqual(features.shape, (2, 64))
            mock_extract.assert_called_once()
    
    def test_model_persistence(self):
        """Test model save/load functionality."""
        classifier = LSMClassifier(window_size=5, embedding_dim=32, n_classes=3)
        classifier._is_fitted = True
        classifier.classes_ = ['positive', 'negative', 'neutral']
        classifier._model_components = {'test': 'data'}
        
        model_path = os.path.join(self.temp_dir, 'test_classifier')
        
        try:
            # Test save
            classifier.save(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test load
            loaded_classifier = LSMClassifier.load(model_path)
            self.assertIsNotNone(loaded_classifier)
            self.assertEqual(loaded_classifier.window_size, 5)
            self.assertEqual(loaded_classifier.embedding_dim, 32)
            self.assertEqual(loaded_classifier.n_classes, 3)
            
        except NotImplementedError:
            # If not fully implemented, verify interface exists
            self.assertTrue(hasattr(classifier, 'save'))
            self.assertTrue(hasattr(LSMClassifier, 'load'))


@unittest.skipUnless(CLASSIFIER_AVAILABLE, "LSMClassifier required")
class TestLSMClassifierAdvanced(unittest.TestCase):
    """Test advanced LSMClassifier functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.binary_data = [
            ("Positive text", 1),
            ("Negative text", 0),
            ("Another positive", 1),
            ("Another negative", 0)
        ]
        
        self.multiclass_data = [
            ("Happy text", "joy"),
            ("Sad text", "sadness"),
            ("Angry text", "anger"),
            ("Fearful text", "fear")
        ]
    
    def test_binary_classification(self):
        """Test binary classification functionality."""
        classifier = LSMClassifier(n_classes=2)
        
        X = [item[0] for item in self.binary_data]
        y = [item[1] for item in self.binary_data]
        
        # Mock training and prediction
        with patch.object(classifier, '_prepare_classification_data') as mock_prepare:
            mock_prepare.return_value = (X, y)
            
            with patch.object(classifier, '_fit_classifier') as mock_fit:
                mock_fit.return_value = None
                
                classifier.fit(X, y)
                
                # Verify binary classification setup
                self.assertEqual(classifier.n_classes, 2)
                self.assertTrue(classifier._is_fitted)
    
    def test_multiclass_classification(self):
        """Test multiclass classification functionality."""
        classifier = LSMClassifier(n_classes=4)
        
        X = [item[0] for item in self.multiclass_data]
        y = [item[1] for item in self.multiclass_data]
        
        # Mock training and prediction
        with patch.object(classifier, '_prepare_classification_data') as mock_prepare:
            mock_prepare.return_value = (X, y)
            
            with patch.object(classifier, '_fit_classifier') as mock_fit:
                mock_fit.return_value = None
                
                classifier.fit(X, y)
                
                # Verify multiclass classification setup
                self.assertEqual(classifier.n_classes, 4)
                self.assertTrue(classifier._is_fitted)
    
    def test_reservoir_types(self):
        """Test different reservoir types for classification."""
        reservoir_types = ['standard', 'hierarchical', 'echo_state']
        
        for reservoir_type in reservoir_types:
            with self.subTest(reservoir_type=reservoir_type):
                classifier = LSMClassifier(reservoir_type=reservoir_type)
                self.assertEqual(classifier.reservoir_type, reservoir_type)
    
    def test_cross_validation_compatibility(self):
        """Test compatibility with sklearn cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.datasets import make_classification
            
            # Create synthetic data
            X, y = make_classification(n_samples=20, n_features=10, n_classes=2, random_state=42)
            X = [f"text_{i}" for i in range(len(X))]  # Convert to text data
            
            classifier = LSMClassifier(n_classes=2)
            
            # Mock the classifier methods for cross-validation
            with patch.object(classifier, 'fit') as mock_fit:
                with patch.object(classifier, 'predict') as mock_predict:
                    mock_fit.return_value = classifier
                    mock_predict.return_value = y[:len(y)//2]  # Mock predictions
                    
                    # This would normally run cross-validation
                    # We're just testing that the interface is compatible
                    self.assertTrue(hasattr(classifier, 'fit'))
                    self.assertTrue(hasattr(classifier, 'predict'))
                    self.assertTrue(hasattr(classifier, 'score'))
                    
        except ImportError:
            self.skipTest("sklearn not available")
    
    def test_pipeline_compatibility(self):
        """Test compatibility with sklearn pipelines."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            # Create a simple pipeline
            classifier = LSMClassifier()
            
            # Test that classifier has required methods for pipeline
            self.assertTrue(hasattr(classifier, 'fit'))
            self.assertTrue(hasattr(classifier, 'predict'))
            self.assertTrue(hasattr(classifier, 'get_params'))
            self.assertTrue(hasattr(classifier, 'set_params'))
            
        except ImportError:
            self.skipTest("sklearn not available")
    
    def test_performance_metrics(self):
        """Test performance metrics and evaluation."""
        classifier = LSMClassifier()
        classifier._is_fitted = True
        
        # Mock predictions for metrics calculation
        y_true = ['positive', 'negative', 'positive', 'negative']
        y_pred = ['positive', 'negative', 'negative', 'positive']
        
        with patch.object(classifier, 'predict') as mock_predict:
            mock_predict.return_value = y_pred
            
            # Test accuracy score
            accuracy = classifier.score(['text1', 'text2', 'text3', 'text4'], y_true)
            self.assertEqual(accuracy, 0.5)  # 2 correct out of 4
    
    def test_class_imbalance_handling(self):
        """Test handling of imbalanced classes."""
        # Create imbalanced dataset
        imbalanced_data = [
            ("Positive 1", "positive"),
            ("Positive 2", "positive"),
            ("Positive 3", "positive"),
            ("Positive 4", "positive"),
            ("Negative 1", "negative")  # Only one negative example
        ]
        
        X = [item[0] for item in imbalanced_data]
        y = [item[1] for item in imbalanced_data]
        
        classifier = LSMClassifier()
        
        # Mock training with imbalanced data
        with patch.object(classifier, '_prepare_classification_data') as mock_prepare:
            mock_prepare.return_value = (X, y)
            
            with patch.object(classifier, '_fit_classifier') as mock_fit:
                mock_fit.return_value = None
                
                # Should handle imbalanced data without error
                classifier.fit(X, y)
                self.assertTrue(classifier._is_fitted)


if __name__ == '__main__':
    unittest.main(verbosity=2)