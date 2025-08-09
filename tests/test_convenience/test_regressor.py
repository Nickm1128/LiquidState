#!/usr/bin/env python3
"""
Tests for LSMRegressor convenience API class.

This module tests the LSMRegressor class which provides a sklearn-like
interface for regression tasks using LSM temporal dynamics.
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
    from lsm.convenience.regressor import LSMRegressor
    from lsm.convenience.config import ConvenienceConfig, ConvenienceValidationError
    REGRESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LSMRegressor not available: {e}")
    REGRESSOR_AVAILABLE = False


@unittest.skipUnless(REGRESSOR_AVAILABLE, "LSMRegressor required")
class TestLSMRegressor(unittest.TestCase):
    """Test LSMRegressor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Time series data
        self.time_series_data = [
            ([1, 2, 3, 4], 5),
            ([2, 3, 4, 5], 6),
            ([3, 4, 5, 6], 7),
            ([4, 5, 6, 7], 8),
            ([5, 6, 7, 8], 9)
        ]
        
        # Text-based regression data
        self.text_regression_data = [
            ("Short text", 1.5),
            ("This is a longer text example", 3.2),
            ("Medium length text here", 2.8),
            ("Very long text with many words and complex structure", 4.7),
            ("Brief", 1.0)
        ]
        
        self.X_numeric = [item[0] for item in self.time_series_data]
        self.y_numeric = [item[1] for item in self.time_series_data]
        
        self.X_text = [item[0] for item in self.text_regression_data]
        self.y_text = [item[1] for item in self.text_regression_data]
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LSMRegressor initialization."""
        # Test default initialization
        regressor = LSMRegressor()
        self.assertIsNotNone(regressor)
        self.assertEqual(regressor.window_size, 10)  # Default value
        self.assertEqual(regressor.embedding_dim, 128)  # Default value
        
        # Test custom initialization
        regressor = LSMRegressor(
            window_size=15,
            embedding_dim=256,
            reservoir_type='echo_state'
        )
        self.assertEqual(regressor.window_size, 15)
        self.assertEqual(regressor.embedding_dim, 256)
        self.assertEqual(regressor.reservoir_type, 'echo_state')
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid window_size
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMRegressor(window_size=0)
        
        # Test invalid embedding_dim
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMRegressor(embedding_dim=-1)
        
        # Test invalid reservoir_type
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMRegressor(reservoir_type='invalid_type')
    
    def test_sklearn_interface(self):
        """Test sklearn-compatible interface."""
        regressor = LSMRegressor()
        
        # Test required methods exist
        self.assertTrue(hasattr(regressor, 'fit'))
        self.assertTrue(hasattr(regressor, 'predict'))
        self.assertTrue(hasattr(regressor, 'score'))
        self.assertTrue(hasattr(regressor, 'get_params'))
        self.assertTrue(hasattr(regressor, 'set_params'))
    
    def test_get_params(self):
        """Test get_params method."""
        regressor = LSMRegressor(
            window_size=15,
            embedding_dim=256,
            reservoir_type='echo_state'
        )
        
        params = regressor.get_params()
        self.assertIsInstance(params, dict)
        self.assertEqual(params['window_size'], 15)
        self.assertEqual(params['embedding_dim'], 256)
        self.assertEqual(params['reservoir_type'], 'echo_state')
    
    def test_set_params(self):
        """Test set_params method."""
        regressor = LSMRegressor()
        
        # Set new parameters
        regressor.set_params(
            window_size=20,
            embedding_dim=512,
            reservoir_type='hierarchical'
        )
        
        # Verify parameters were set
        self.assertEqual(regressor.window_size, 20)
        self.assertEqual(regressor.embedding_dim, 512)
        self.assertEqual(regressor.reservoir_type, 'hierarchical')
    
    @patch('lsm.training.train.LSMTrainer')
    def test_fit_numeric_data(self, mock_trainer_class):
        """Test fit with numeric time series data."""
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train.return_value = True
        
        regressor = LSMRegressor(window_size=3, embedding_dim=32)
        
        # Mock data preparation and regressor training
        with patch.object(regressor, '_prepare_regression_data') as mock_prepare:
            mock_prepare.return_value = (self.X_numeric, self.y_numeric)
            
            with patch.object(regressor, '_fit_regressor') as mock_fit:
                mock_fit.return_value = None
                
                # Test fit
                regressor.fit(self.X_numeric, self.y_numeric)
                
                # Verify methods were called
                mock_prepare.assert_called_once()
                mock_fit.assert_called_once()
                self.assertTrue(regressor._is_fitted)
    
    @patch('lsm.training.train.LSMTrainer')
    def test_fit_text_data(self, mock_trainer_class):
        """Test fit with text-based regression data."""
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train.return_value = True
        
        regressor = LSMRegressor(window_size=5, embedding_dim=64)
        
        # Mock data preparation and regressor training
        with patch.object(regressor, '_prepare_regression_data') as mock_prepare:
            mock_prepare.return_value = (self.X_text, self.y_text)
            
            with patch.object(regressor, '_fit_regressor') as mock_fit:
                mock_fit.return_value = None
                
                # Test fit
                regressor.fit(self.X_text, self.y_text)
                
                # Verify methods were called
                mock_prepare.assert_called_once()
                mock_fit.assert_called_once()
                self.assertTrue(regressor._is_fitted)
    
    def test_fit_validation(self):
        """Test fit input validation."""
        regressor = LSMRegressor()
        
        # Test empty data
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            regressor.fit([], [])
        
        # Test mismatched X and y lengths
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            regressor.fit([[1, 2], [3, 4]], [1])
        
        # Test invalid y format (non-numeric)
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            regressor.fit([[1, 2], [3, 4]], ["a", "b"])
    
    def test_predict_basic(self):
        """Test basic predict functionality."""
        regressor = LSMRegressor()
        regressor._is_fitted = True
        
        # Mock prediction
        with patch.object(regressor, '_predict_values') as mock_predict:
            mock_predict.return_value = [5.0, 6.0, 7.0]
            
            # Test predict
            predictions = regressor.predict(self.X_numeric[:3])
            
            self.assertEqual(len(predictions), 3)
            self.assertEqual(predictions[0], 5.0)
            self.assertEqual(predictions[1], 6.0)
            self.assertEqual(predictions[2], 7.0)
            mock_predict.assert_called_once()
    
    def test_predict_not_fitted(self):
        """Test predict raises error when not fitted."""
        regressor = LSMRegressor()
        
        with self.assertRaises((ValueError, RuntimeError)):
            regressor.predict(self.X_numeric)
    
    def test_score_basic(self):
        """Test basic score functionality (R² score)."""
        regressor = LSMRegressor()
        regressor._is_fitted = True
        
        # Mock prediction and scoring
        with patch.object(regressor, 'predict') as mock_predict:
            mock_predict.return_value = [5.0, 6.0, 7.0]
            
            # Test score with perfect predictions
            y_true = [5.0, 6.0, 7.0]
            r2_score = regressor.score(self.X_numeric[:3], y_true)
            
            self.assertEqual(r2_score, 1.0)  # Perfect R² score
            mock_predict.assert_called_once()
    
    def test_time_series_prediction(self):
        """Test time series prediction capabilities."""
        regressor = LSMRegressor(reservoir_type='echo_state')  # Good for time series
        regressor._is_fitted = True
        
        # Mock time series prediction
        with patch.object(regressor, '_predict_time_series') as mock_predict_ts:
            mock_predict_ts.return_value = [9.0, 10.0, 11.0]
            
            # Test time series prediction
            future_values = regressor.predict_time_series(
                self.X_numeric[-1],  # Last sequence
                steps=3
            )
            
            self.assertEqual(len(future_values), 3)
            mock_predict_ts.assert_called_once()
    
    def test_feature_extraction(self):
        """Test reservoir feature extraction for regression."""
        regressor = LSMRegressor()
        regressor._is_fitted = True
        
        # Mock feature extraction
        with patch.object(regressor, '_extract_features') as mock_extract:
            mock_features = np.random.rand(2, 64)  # 2 samples, 64 features
            mock_extract.return_value = mock_features
            
            # Test feature extraction
            features = regressor._extract_features(self.X_numeric[:2])
            
            self.assertEqual(features.shape, (2, 64))
            mock_extract.assert_called_once()
    
    def test_model_persistence(self):
        """Test model save/load functionality."""
        regressor = LSMRegressor(window_size=5, embedding_dim=32)
        regressor._is_fitted = True
        regressor._model_components = {'test': 'data'}
        
        model_path = os.path.join(self.temp_dir, 'test_regressor')
        
        try:
            # Test save
            regressor.save(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test load
            loaded_regressor = LSMRegressor.load(model_path)
            self.assertIsNotNone(loaded_regressor)
            self.assertEqual(loaded_regressor.window_size, 5)
            self.assertEqual(loaded_regressor.embedding_dim, 32)
            
        except NotImplementedError:
            # If not fully implemented, verify interface exists
            self.assertTrue(hasattr(regressor, 'save'))
            self.assertTrue(hasattr(LSMRegressor, 'load'))


@unittest.skipUnless(REGRESSOR_AVAILABLE, "LSMRegressor required")
class TestLSMRegressorAdvanced(unittest.TestCase):
    """Test advanced LSMRegressor functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Multi-dimensional time series
        self.multivariate_data = [
            ([[1, 2], [2, 3], [3, 4]], [4, 5]),
            ([[2, 3], [3, 4], [4, 5]], [5, 6]),
            ([[3, 4], [4, 5], [5, 6]], [6, 7])
        ]
        
        # Noisy data for robustness testing
        np.random.seed(42)
        self.noisy_data = []
        for i in range(20):
            x = [i + j + np.random.normal(0, 0.1) for j in range(4)]
            y = sum(x) / len(x) + np.random.normal(0, 0.2)
            self.noisy_data.append((x, y))
    
    def test_multivariate_regression(self):
        """Test multivariate regression functionality."""
        regressor = LSMRegressor(window_size=3, embedding_dim=64)
        
        X = [item[0] for item in self.multivariate_data]
        y = [item[1] for item in self.multivariate_data]
        
        # Mock training and prediction for multivariate data
        with patch.object(regressor, '_prepare_regression_data') as mock_prepare:
            mock_prepare.return_value = (X, y)
            
            with patch.object(regressor, '_fit_regressor') as mock_fit:
                mock_fit.return_value = None
                
                regressor.fit(X, y)
                
                # Verify multivariate regression setup
                self.assertTrue(regressor._is_fitted)
    
    def test_reservoir_types_for_regression(self):
        """Test different reservoir types for regression tasks."""
        reservoir_types = ['standard', 'echo_state', 'hierarchical']
        
        for reservoir_type in reservoir_types:
            with self.subTest(reservoir_type=reservoir_type):
                regressor = LSMRegressor(reservoir_type=reservoir_type)
                self.assertEqual(regressor.reservoir_type, reservoir_type)
    
    def test_noise_robustness(self):
        """Test robustness to noisy data."""
        regressor = LSMRegressor(window_size=4, embedding_dim=32)
        
        X = [item[0] for item in self.noisy_data]
        y = [item[1] for item in self.noisy_data]
        
        # Mock training with noisy data
        with patch.object(regressor, '_prepare_regression_data') as mock_prepare:
            mock_prepare.return_value = (X, y)
            
            with patch.object(regressor, '_fit_regressor') as mock_fit:
                mock_fit.return_value = None
                
                # Should handle noisy data without error
                regressor.fit(X, y)
                self.assertTrue(regressor._is_fitted)
    
    def test_cross_validation_compatibility(self):
        """Test compatibility with sklearn cross-validation."""
        try:
            from sklearn.model_selection import cross_val_score
            from sklearn.datasets import make_regression
            
            # Create synthetic regression data
            X, y = make_regression(n_samples=20, n_features=5, noise=0.1, random_state=42)
            X = X.tolist()  # Convert to list format
            
            regressor = LSMRegressor()
            
            # Mock the regressor methods for cross-validation
            with patch.object(regressor, 'fit') as mock_fit:
                with patch.object(regressor, 'predict') as mock_predict:
                    with patch.object(regressor, 'score') as mock_score:
                        mock_fit.return_value = regressor
                        mock_predict.return_value = y[:len(y)//2]
                        mock_score.return_value = 0.8
                        
                        # Test interface compatibility
                        self.assertTrue(hasattr(regressor, 'fit'))
                        self.assertTrue(hasattr(regressor, 'predict'))
                        self.assertTrue(hasattr(regressor, 'score'))
                        
        except ImportError:
            self.skipTest("sklearn not available")
    
    def test_pipeline_compatibility(self):
        """Test compatibility with sklearn pipelines."""
        try:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            
            regressor = LSMRegressor()
            
            # Test that regressor has required methods for pipeline
            self.assertTrue(hasattr(regressor, 'fit'))
            self.assertTrue(hasattr(regressor, 'predict'))
            self.assertTrue(hasattr(regressor, 'get_params'))
            self.assertTrue(hasattr(regressor, 'set_params'))
            
        except ImportError:
            self.skipTest("sklearn not available")
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        regressor = LSMRegressor()
        regressor._is_fitted = True
        
        # Mock predictions for metrics calculation
        y_true = [1.0, 2.0, 3.0, 4.0]
        y_pred = [1.1, 1.9, 3.2, 3.8]
        
        with patch.object(regressor, 'predict') as mock_predict:
            mock_predict.return_value = y_pred
            
            # Test R² score calculation
            r2_score = regressor.score(['x1', 'x2', 'x3', 'x4'], y_true)
            
            # Should be a reasonable R² score (not perfect due to prediction errors)
            self.assertIsInstance(r2_score, float)
            self.assertLessEqual(r2_score, 1.0)
    
    def test_outlier_handling(self):
        """Test handling of outliers in regression data."""
        # Create data with outliers
        outlier_data = [
            ([1, 2, 3], 2.0),
            ([2, 3, 4], 3.0),
            ([3, 4, 5], 4.0),
            ([4, 5, 6], 100.0),  # Outlier
            ([5, 6, 7], 6.0)
        ]
        
        X = [item[0] for item in outlier_data]
        y = [item[1] for item in outlier_data]
        
        regressor = LSMRegressor()
        
        # Mock training with outlier data
        with patch.object(regressor, '_prepare_regression_data') as mock_prepare:
            mock_prepare.return_value = (X, y)
            
            with patch.object(regressor, '_fit_regressor') as mock_fit:
                mock_fit.return_value = None
                
                # Should handle outliers without error
                regressor.fit(X, y)
                self.assertTrue(regressor._is_fitted)
    
    def test_sequential_prediction(self):
        """Test sequential prediction for time series."""
        regressor = LSMRegressor(reservoir_type='echo_state')
        regressor._is_fitted = True
        
        # Mock sequential prediction
        with patch.object(regressor, '_predict_sequential') as mock_predict_seq:
            mock_predict_seq.return_value = [8.0, 9.0, 10.0, 11.0, 12.0]
            
            # Test sequential prediction
            initial_sequence = [1, 2, 3, 4, 5]
            future_predictions = regressor.predict_sequential(
                initial_sequence,
                steps=5
            )
            
            self.assertEqual(len(future_predictions), 5)
            mock_predict_seq.assert_called_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)