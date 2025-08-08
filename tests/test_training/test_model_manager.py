#!/usr/bin/env python3
"""
Test suite for ModelManager class.

This module provides comprehensive tests for the ModelManager functionality
including model discovery, validation, and cleanup operations.
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.lsm.management.model_manager import ModelManager
from src.lsm.training.model_config import ModelConfiguration, TrainingMetadata

class TestModelManager(unittest.TestCase):
    """Test cases for ModelManager class."""
    
    def setUp(self):
        """Set up test environment with temporary directories."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelManager(self.test_dir)
        
        # Create test model directories
        self.valid_model_dir = os.path.join(self.test_dir, "models_20250108_123456")
        self.invalid_model_dir = os.path.join(self.test_dir, "models_20250108_654321")
        self.incomplete_model_dir = os.path.join(self.test_dir, "models_20250108_111111")
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_valid_model(self, model_dir: str):
        """Create a valid model directory structure for testing."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Create reservoir model directory with required files
        reservoir_dir = os.path.join(model_dir, "reservoir_model")
        os.makedirs(reservoir_dir, exist_ok=True)
        
        # Create minimal Keras model files
        with open(os.path.join(reservoir_dir, "saved_model.pb"), 'w') as f:
            f.write("dummy model data")
        
        variables_dir = os.path.join(reservoir_dir, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
            f.write("dummy variables")
        with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
            f.write("dummy index")
        
        # Create CNN model directory
        cnn_dir = os.path.join(model_dir, "cnn_model")
        os.makedirs(cnn_dir, exist_ok=True)
        
        with open(os.path.join(cnn_dir, "saved_model.pb"), 'w') as f:
            f.write("dummy cnn model")
        
        cnn_variables_dir = os.path.join(cnn_dir, "variables")
        os.makedirs(cnn_variables_dir, exist_ok=True)
        with open(os.path.join(cnn_variables_dir, "variables.data-00000-of-00001"), 'w') as f:
            f.write("dummy cnn variables")
        with open(os.path.join(cnn_variables_dir, "variables.index"), 'w') as f:
            f.write("dummy cnn index")
        
        # Create tokenizer directory
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        tokenizer_config = {
            "max_features": 10000,
            "embedding_dim": 128,
            "is_fitted": True,
            "vocabulary_size": 5000
        }
        with open(os.path.join(tokenizer_dir, "config.json"), 'w') as f:
            json.dump(tokenizer_config, f)
        
        with open(os.path.join(tokenizer_dir, "vectorizer.pkl"), 'wb') as f:
            f.write(b"dummy vectorizer data")
        
        with open(os.path.join(tokenizer_dir, "vocabulary.json"), 'w') as f:
            json.dump(["hello", "world", "test"], f)
        
        # Create configuration file
        config = ModelConfiguration(
            window_size=10,
            embedding_dim=128,
            reservoir_type='standard',
            created_at=datetime.now().isoformat()
        )
        config.save(os.path.join(model_dir, "config.json"))
        
        # Create training metadata
        metadata = TrainingMetadata(
            training_completed_at=datetime.now().isoformat(),
            training_duration_seconds=3600.0,
            dataset_info={
                'source': 'test_dataset',
                'num_sequences': 1000,
                'train_samples': 800,
                'test_samples': 200
            },
            performance_metrics={
                'final_test_mse': 0.0234,
                'final_test_mae': 0.1123
            },
            system_info={
                'python_version': '3.11.9',
                'platform': 'Test Platform'
            }
        )
        metadata.save(os.path.join(model_dir, "metadata.json"))
        
        # Create training history
        with open(os.path.join(model_dir, "training_history.csv"), 'w') as f:
            f.write("epoch,train_mse,test_mse\n1,0.1,0.12\n2,0.08,0.09\n")
    
    def _create_incomplete_model(self, model_dir: str):
        """Create an incomplete model directory for testing cleanup."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Only create config file, missing other components
        config = ModelConfiguration()
        config.save(os.path.join(model_dir, "config.json"))
    
    def test_list_available_models_empty(self):
        """Test listing models when no models exist."""
        models = self.manager.list_available_models()
        self.assertEqual(len(models), 0)
    
    def test_list_available_models_with_valid_model(self):
        """Test listing models with a valid model present."""
        self._create_valid_model(self.valid_model_dir)
        
        models = self.manager.list_available_models()
        self.assertEqual(len(models), 1)
        self.assertTrue(models[0]['is_valid'])
        self.assertEqual(models[0]['name'], "models_20250108_123456")
    
    def test_get_model_info_nonexistent(self):
        """Test getting info for non-existent model."""
        info = self.manager.get_model_info("/nonexistent/path")
        self.assertFalse(info['is_valid'])
        self.assertIn('does not exist', info['error'])
    
    def test_get_model_info_valid_model(self):
        """Test getting comprehensive info for a valid model."""
        self._create_valid_model(self.valid_model_dir)
        
        info = self.manager.get_model_info(self.valid_model_dir)
        
        # Check basic info
        self.assertTrue(info['is_valid'])
        self.assertEqual(info['name'], "models_20250108_123456")
        self.assertIn('configuration', info)
        self.assertIn('training_metadata', info)
        self.assertIn('tokenizer', info)
        
        # Check components
        components = info['components']
        self.assertTrue(components['reservoir_model'])
        self.assertTrue(components['cnn_model'])
        self.assertTrue(components['tokenizer'])
        self.assertTrue(components['config'])
        self.assertTrue(components['metadata'])
        
        # Check architecture info
        arch = info['architecture']
        self.assertEqual(arch['window_size'], 10)
        self.assertEqual(arch['embedding_dim'], 128)
        self.assertEqual(arch['reservoir_type'], 'standard')
        
        # Check tokenizer info
        tokenizer = info['tokenizer']
        self.assertEqual(tokenizer['vocabulary_size'], 5000)
        self.assertEqual(tokenizer['max_features'], 10000)
    
    def test_validate_model_valid(self):
        """Test validation of a valid model."""
        self._create_valid_model(self.valid_model_dir)
        
        is_valid, errors = self.manager.validate_model(self.valid_model_dir)
        self.assertTrue(is_valid)
        
        # Should only have non-critical errors if any
        critical_errors = [err for err in errors if not err.endswith("(non-critical)")]
        self.assertEqual(len(critical_errors), 0)
    
    def test_validate_model_missing_components(self):
        """Test validation of model with missing components."""
        os.makedirs(self.invalid_model_dir, exist_ok=True)
        
        is_valid, errors = self.manager.validate_model(self.invalid_model_dir)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Should have errors about missing components
        error_text = ' '.join(errors)
        self.assertIn('Missing', error_text)
    
    def test_validate_model_nonexistent(self):
        """Test validation of non-existent model."""
        is_valid, errors = self.manager.validate_model("/nonexistent/path")
        self.assertFalse(is_valid)
        self.assertIn('does not exist', errors[0])
    
    def test_cleanup_incomplete_models_dry_run(self):
        """Test cleanup in dry run mode."""
        # Create valid and incomplete models
        self._create_valid_model(self.valid_model_dir)
        self._create_incomplete_model(self.incomplete_model_dir)
        
        # Run cleanup in dry run mode
        cleanup_candidates = self.manager.cleanup_incomplete_models(dry_run=True)
        
        # Should find the incomplete model but not delete it
        self.assertEqual(len(cleanup_candidates), 1)
        self.assertIn("models_20250108_111111", cleanup_candidates[0])
        self.assertTrue(os.path.exists(self.incomplete_model_dir))
        self.assertTrue(os.path.exists(self.valid_model_dir))
    
    def test_cleanup_incomplete_models_actual(self):
        """Test actual cleanup of incomplete models."""
        # Create valid and incomplete models
        self._create_valid_model(self.valid_model_dir)
        self._create_incomplete_model(self.incomplete_model_dir)
        
        # Run actual cleanup
        cleanup_candidates = self.manager.cleanup_incomplete_models(dry_run=False)
        
        # Should find and delete the incomplete model
        self.assertEqual(len(cleanup_candidates), 1)
        self.assertFalse(os.path.exists(self.incomplete_model_dir))
        self.assertTrue(os.path.exists(self.valid_model_dir))
    
    def test_get_model_summary(self):
        """Test getting a human-readable model summary."""
        self._create_valid_model(self.valid_model_dir)
        
        summary = self.manager.get_model_summary(self.valid_model_dir)
        
        self.assertIn("✅", summary)  # Valid model indicator
        self.assertIn("models_20250108_123456", summary)
        self.assertIn("standard reservoir", summary)
        self.assertIn("Window Size: 10", summary)
        self.assertIn("Embedding Dim: 128", summary)
        self.assertIn("Test MSE:", summary)
        self.assertIn("Vocabulary:", summary)
    
    def test_get_model_summary_invalid(self):
        """Test getting summary for invalid model."""
        os.makedirs(self.invalid_model_dir, exist_ok=True)
        
        summary = self.manager.get_model_summary(self.invalid_model_dir)
        
        self.assertIn("❌", summary)  # Invalid model indicator
        self.assertIn("Invalid Model", summary)
        self.assertIn("Errors:", summary)
    
    def test_list_models_summary(self):
        """Test getting formatted summary of all models."""
        # Create multiple models
        self._create_valid_model(self.valid_model_dir)
        self._create_incomplete_model(self.incomplete_model_dir)
        
        summary = self.manager.list_models_summary()
        
        # Should only include valid models
        self.assertIn("Found 1 valid models", summary)
        self.assertIn("models_20250108_123456", summary)
        self.assertNotIn("models_20250108_111111", summary)  # Incomplete model excluded
    
    def test_list_models_summary_empty(self):
        """Test summary when no models exist."""
        summary = self.manager.list_models_summary()
        self.assertIn("No valid models found", summary)
    
    def test_check_model_components(self):
        """Test checking which model components are present."""
        self._create_valid_model(self.valid_model_dir)
        
        components = self.manager._check_model_components(self.valid_model_dir)
        
        self.assertTrue(components['reservoir_model'])
        self.assertTrue(components['cnn_model'])
        self.assertTrue(components['tokenizer'])
        self.assertTrue(components['config'])
        self.assertTrue(components['metadata'])
        self.assertTrue(components['training_history'])
    
    def test_is_valid_keras_model(self):
        """Test Keras model validation."""
        # Create valid Keras model structure
        model_dir = os.path.join(self.test_dir, "test_keras_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Missing files - should be invalid
        self.assertFalse(self.manager._is_valid_keras_model(model_dir))
        
        # Add required files
        with open(os.path.join(model_dir, "saved_model.pb"), 'w') as f:
            f.write("dummy")
        
        variables_dir = os.path.join(model_dir, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
            f.write("dummy")
        with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
            f.write("dummy")
        
        # Now should be valid
        self.assertTrue(self.manager._is_valid_keras_model(model_dir))
    
    def test_validate_tokenizer(self):
        """Test tokenizer validation."""
        tokenizer_dir = os.path.join(self.test_dir, "test_tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # Empty directory - should have errors
        errors = self.manager._validate_tokenizer(tokenizer_dir)
        self.assertGreater(len(errors), 0)
        
        # Add required files
        with open(os.path.join(tokenizer_dir, "config.json"), 'w') as f:
            json.dump({"test": "config"}, f)
        with open(os.path.join(tokenizer_dir, "vectorizer.pkl"), 'wb') as f:
            f.write(b"dummy")
        with open(os.path.join(tokenizer_dir, "vocabulary.json"), 'w') as f:
            json.dump(["test"], f)
        
        # Should have no errors now
        errors = self.manager._validate_tokenizer(tokenizer_dir)
        self.assertEqual(len(errors), 0)
    
    def test_calculate_directory_size(self):
        """Test directory size calculation."""
        test_dir = os.path.join(self.test_dir, "size_test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Create some test files
        with open(os.path.join(test_dir, "file1.txt"), 'w') as f:
            f.write("x" * 1024)  # 1KB
        
        subdir = os.path.join(test_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)
        with open(os.path.join(subdir, "file2.txt"), 'w') as f:
            f.write("y" * 2048)  # 2KB
        
        size_mb = self.manager._calculate_directory_size(test_dir)
        
        # Should be approximately 3KB = 0.003MB
        self.assertGreater(size_mb, 0.002)
        self.assertLess(size_mb, 0.01)

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)