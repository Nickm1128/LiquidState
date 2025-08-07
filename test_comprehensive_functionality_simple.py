#!/usr/bin/env python3
"""
Simplified comprehensive test suite that doesn't require TensorFlow.

This module provides comprehensive tests for:
- DialogueTokenizer save/load and decoding methods
- ModelConfiguration persistence
- Model management functionality
- Error handling
"""

import os
import json
import time
import tempfile
import shutil
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Tuple

# Import components that don't require TensorFlow
from data_loader import DialogueTokenizer
from model_config import ModelConfiguration, TrainingMetadata
from src.lsm.management.model_manager import ModelManager
from lsm_exceptions import *
from lsm_logging import get_logger

logger = get_logger(__name__)

class TestDialogueTokenizerIntegration(unittest.TestCase):
    """Test DialogueTokenizer integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.tokenizer = DialogueTokenizer(max_features=500, embedding_dim=64)
        
        self.sample_texts = [
            "hello world how are you",
            "good morning everyone",
            "what is your name today",
            "i am fine thank you",
            "the weather is beautiful",
            "let's go for a walk",
            "have you seen the movie",
            "i love reading books",
            "technology is advancing rapidly",
            "artificial intelligence is amazing"
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_complete_tokenizer_workflow(self):
        """Test complete tokenizer fit-save-load-predict workflow."""
        # Step 1: Fit tokenizer
        self.tokenizer.fit(self.sample_texts)
        self.assertTrue(self.tokenizer.is_fitted)
        
        # Step 2: Test encoding
        test_texts = ["hello world", "good morning"]
        original_encoded = self.tokenizer.encode(test_texts)
        self.assertEqual(original_encoded.shape, (2, 64))
        
        # Step 3: Test decoding
        decoded_text = self.tokenizer.decode_embedding(original_encoded[0])
        self.assertIsInstance(decoded_text, str)
        self.assertNotEqual(decoded_text, "[ERROR]")
        
        # Step 4: Save tokenizer
        save_path = os.path.join(self.test_dir, "workflow_tokenizer")
        self.tokenizer.save(save_path)
        
        # Step 5: Load tokenizer in new instance
        new_tokenizer = DialogueTokenizer(max_features=500, embedding_dim=64)
        new_tokenizer.load(save_path)
        
        # Step 6: Verify loaded tokenizer works identically
        loaded_encoded = new_tokenizer.encode(test_texts)
        np.testing.assert_array_almost_equal(original_encoded, loaded_encoded, decimal=5)
        
        loaded_decoded = new_tokenizer.decode_embedding(loaded_encoded[0])
        self.assertEqual(decoded_text, loaded_decoded)
        
        # Step 7: Test batch operations
        batch_decoded = new_tokenizer.decode_embeddings_batch(loaded_encoded)
        self.assertEqual(len(batch_decoded), 2)
        
        # Step 8: Test closest texts functionality
        closest_texts = new_tokenizer.get_closest_texts(loaded_encoded[0], top_k=3)
        self.assertEqual(len(closest_texts), 3)
        
        print("Complete tokenizer workflow test passed")


class TestModelConfigurationIntegration(unittest.TestCase):
    """Test ModelConfiguration integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_configuration_workflow(self):
        """Test complete configuration creation-save-load-validate workflow."""
        # Step 1: Create configuration
        config = ModelConfiguration(
            window_size=12,
            embedding_dim=256,
            reservoir_type='hierarchical',
            reservoir_units=[512, 256, 128],
            sparsity=0.05,
            epochs=30,
            batch_size=64,
            tokenizer_max_features=5000
        )
        
        # Step 2: Validate configuration
        errors = config.validate()
        self.assertEqual(len(errors), 0, f"Configuration validation failed: {errors}")
        
        # Step 3: Save configuration
        config_path = os.path.join(self.test_dir, "workflow_config.json")
        config.save(config_path)
        self.assertTrue(os.path.exists(config_path))
        
        # Step 4: Load configuration
        loaded_config = ModelConfiguration.load(config_path)
        
        # Step 5: Verify loaded configuration
        self.assertEqual(loaded_config.window_size, 12)
        self.assertEqual(loaded_config.embedding_dim, 256)
        self.assertEqual(loaded_config.reservoir_type, 'hierarchical')
        self.assertEqual(loaded_config.reservoir_units, [512, 256, 128])
        self.assertEqual(loaded_config.sparsity, 0.05)
        self.assertEqual(loaded_config.epochs, 30)
        self.assertEqual(loaded_config.batch_size, 64)
        self.assertEqual(loaded_config.tokenizer_max_features, 5000)
        
        # Step 6: Test configuration summary
        summary = loaded_config.get_summary()
        self.assertIn("Window Size: 12", summary)
        self.assertIn("Embedding Dimension: 256", summary)
        self.assertIn("Reservoir Type: hierarchical", summary)
        
        # Step 7: Test dictionary conversion
        config_dict = loaded_config.to_dict()
        reconstructed_config = ModelConfiguration.from_dict(config_dict)
        self.assertEqual(reconstructed_config.window_size, 12)
        
        print("Complete configuration workflow test passed")


class TestModelManagerIntegration(unittest.TestCase):
    """Test ModelManager integration functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelManager(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_complete_mock_model(self, model_name: str) -> str:
        """Create a complete mock model for testing."""
        model_dir = os.path.join(self.test_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create model directories
        for model_type in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(model_dir, model_type)
            os.makedirs(model_path, exist_ok=True)
            
            # Create Keras model files
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write(f"mock {model_type} data")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("mock variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("mock index")
        
        # Create tokenizer
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        tokenizer_config = {
            "max_features": 1000,
            "embedding_dim": 128,
            "is_fitted": True,
            "vocabulary_size": 500
        }
        with open(os.path.join(tokenizer_dir, "config.json"), 'w') as f:
            json.dump(tokenizer_config, f)
        
        with open(os.path.join(tokenizer_dir, "vectorizer.pkl"), 'wb') as f:
            f.write(b"mock vectorizer data")
        
        with open(os.path.join(tokenizer_dir, "vocabulary.json"), 'w') as f:
            json.dump(["hello", "world", "test", "sample"], f)
        
        # Create configuration
        config = ModelConfiguration(
            window_size=10,
            embedding_dim=128,
            reservoir_type='standard'
        )
        config.save(os.path.join(model_dir, "config.json"))
        
        # Create metadata
        metadata = TrainingMetadata(
            training_completed_at="2025-01-08T12:00:00Z",
            training_duration_seconds=1800.0,
            dataset_info={
                'source': 'test_dataset',
                'num_sequences': 1000
            },
            performance_metrics={
                'final_test_mse': 0.025,
                'final_test_mae': 0.15
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
        
        return model_dir
    
    def test_model_management_workflow(self):
        """Test complete model management workflow."""
        # Step 1: Create multiple models
        model1_dir = self._create_complete_mock_model("models_20250108_120000")
        model2_dir = self._create_complete_mock_model("models_20250108_130000")
        
        # Step 2: List available models
        models = self.manager.list_available_models()
        self.assertEqual(len(models), 2)
        
        # Step 3: Validate models
        for model in models:
            self.assertTrue(model['is_valid'])
            self.assertIn('configuration', model)
            self.assertIn('training_metadata', model)
            self.assertIn('tokenizer', model)
        
        # Step 4: Get detailed model info
        model_info = self.manager.get_model_info(model1_dir)
        self.assertTrue(model_info['is_valid'])
        self.assertEqual(model_info['architecture']['window_size'], 10)
        self.assertEqual(model_info['architecture']['embedding_dim'], 128)
        self.assertEqual(model_info['tokenizer']['vocabulary_size'], 500)
        
        # Step 5: Test model summary
        summary = self.manager.get_model_summary(model1_dir)
        self.assertIn("models_20250108_120000", summary)
        self.assertIn("standard reservoir", summary)
        self.assertIn("Window Size: 10", summary)
        
        # Step 6: Test models list summary
        list_summary = self.manager.list_models_summary()
        self.assertIn("Found 2 valid models", list_summary)
        
        # Step 7: Test cleanup (dry run)
        cleanup_candidates = self.manager.cleanup_incomplete_models(dry_run=True)
        self.assertEqual(len(cleanup_candidates), 0)  # All models are complete
        
        print("Complete model management workflow test passed")


class TestErrorHandlingIntegration(unittest.TestCase):
    """Test error handling integration."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_error_handling_workflow(self):
        """Test comprehensive error handling workflow."""
        # Test 1: Tokenizer errors
        tokenizer = DialogueTokenizer()
        
        # Should raise error when not fitted
        with self.assertRaises(TokenizerNotFittedError):
            tokenizer.encode(["test"])
        
        # Test 2: Configuration errors
        invalid_config = ModelConfiguration(
            window_size=-1,
            embedding_dim=0,
            reservoir_type='invalid'
        )
        
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)
        
        # Test 3: Model manager errors
        manager = ModelManager()
        
        # Should handle non-existent model gracefully
        is_valid, errors = manager.validate_model("/nonexistent/path")
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Test 4: File operation errors
        try:
            config = ModelConfiguration()
            config.save("/invalid/path/config.json")
            self.fail("Should have raised an error")
        except Exception as e:
            self.assertIsInstance(e, (ConfigurationError, OSError))
        
        print("Error handling workflow test passed")


def run_simplified_comprehensive_tests():
    """Run all simplified comprehensive functionality tests."""
    print("Running Simplified Comprehensive Functionality Tests")
    print("=" * 60)
    
    test_suites = [
        ('DialogueTokenizer Integration', TestDialogueTokenizerIntegration),
        ('ModelConfiguration Integration', TestModelConfigurationIntegration),
        ('ModelManager Integration', TestModelManagerIntegration),
        ('Error Handling Integration', TestErrorHandlingIntegration),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_name, test_class in test_suites:
        print(f"\n{suite_name}")
        print("-" * 40)
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=1, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        # Count results
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        
        total_tests += tests_run
        passed_tests += tests_run - failures - errors
        
        if failures > 0 or errors > 0:
            failed_tests.append(suite_name)
            print(f"  FAILED: {failures + errors}/{tests_run} tests failed")
            
            # Print failure details
            for test, traceback in result.failures + result.errors:
                print(f"    FAILED: {test}")
                if 'AssertionError:' in traceback:
                    error_msg = traceback.split('AssertionError:')[-1].strip()
                    print(f"    {error_msg}")
        else:
            print(f"  PASSED: All {tests_run} tests passed")
    
    # Summary
    print(f"\nSimplified Comprehensive Test Summary")
    print("=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    
    if failed_tests:
        print(f"Failed test suites: {', '.join(failed_tests)}")
        return False
    else:
        print("All simplified comprehensive functionality tests passed!")
        return True


if __name__ == "__main__":
    success = run_simplified_comprehensive_tests()
    exit(0 if success else 1)