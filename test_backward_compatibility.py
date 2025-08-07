#!/usr/bin/env python3
"""
Backward compatibility tests for LSM models.

This module tests backward compatibility with old model formats,
migration utilities, and graceful fallbacks for legacy models.
"""

import os
import json
import tempfile
import shutil
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.lsm.management.model_manager import ModelManager
from model_config import ModelConfiguration
from inference import OptimizedLSMInference
from lsm_exceptions import ModelLoadError, BackwardCompatibilityError
from data_loader import DialogueTokenizer

class TestBackwardCompatibilityDetection(unittest.TestCase):
    """Test detection and handling of old model formats."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.manager = ModelManager(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_legacy_model_v1(self) -> str:
        """Create a legacy model format (version 1) - minimal structure."""
        model_dir = os.path.join(self.test_dir, "legacy_model_v1")
        os.makedirs(model_dir, exist_ok=True)
        
        # Legacy v1: Only has basic model files, no config or tokenizer
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            # Create minimal Keras model structure
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("legacy v1 model data")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("legacy variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("legacy index")
        
        # Legacy v1: Simple parameters file instead of proper config
        legacy_params = {
            "window_size": 10,
            "embedding_dim": 128,
            "model_type": "lsm_legacy_v1",
            "created_date": "2024-01-01"
        }
        with open(os.path.join(model_dir, "model_params.json"), 'w') as f:
            json.dump(legacy_params, f)
        
        return model_dir
    
    def _create_legacy_model_v2(self) -> str:
        """Create a legacy model format (version 2) - has config but no tokenizer."""
        model_dir = os.path.join(self.test_dir, "legacy_model_v2")
        os.makedirs(model_dir, exist_ok=True)
        
        # Legacy v2: Has model files and config, but missing tokenizer
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("legacy v2 model data")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("legacy v2 variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("legacy v2 index")
        
        # Legacy v2: Has proper ModelConfiguration but missing tokenizer
        config = ModelConfiguration(
            window_size=8,
            embedding_dim=96,
            reservoir_type='standard',
            model_version="2.0"
        )
        config.save(os.path.join(model_dir, "config.json"))
        
        # Legacy v2: Has training history but no metadata
        with open(os.path.join(model_dir, "training_history.csv"), 'w') as f:
            f.write("epoch,train_loss,val_loss\n1,0.5,0.6\n2,0.4,0.5\n")
        
        return model_dir
    
    def _create_corrupted_model(self) -> str:
        """Create a corrupted model for testing error handling."""
        model_dir = os.path.join(self.test_dir, "corrupted_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Corrupted: Has some files but they're invalid
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            f.write("invalid json content {")  # Malformed JSON
        
        # Missing model directories entirely
        return model_dir
    
    def test_legacy_v1_detection(self):
        """Test detection of legacy v1 model format."""
        legacy_v1_dir = self._create_legacy_model_v1()
        
        # Test model validation
        is_valid, errors = self.manager.validate_model(legacy_v1_dir)
        
        # Should be invalid due to missing components
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Should detect missing configuration
        error_text = ' '.join(errors)
        self.assertIn('Configuration', error_text)
        
        # Get model info
        model_info = self.manager.get_model_info(legacy_v1_dir)
        self.assertFalse(model_info['is_valid'])
        self.assertIn('validation_errors', model_info)
        
        print(f"Legacy v1 detected with {len(errors)} validation errors")
    
    def test_legacy_v2_detection(self):
        """Test detection of legacy v2 model format."""
        legacy_v2_dir = self._create_legacy_model_v2()
        
        # Test model validation
        is_valid, errors = self.manager.validate_model(legacy_v2_dir)
        
        # Should be invalid due to missing tokenizer
        self.assertFalse(is_valid)
        
        # Should have configuration info available
        model_info = self.manager.get_model_info(legacy_v2_dir)
        self.assertIn('configuration', model_info)
        self.assertEqual(model_info['configuration']['window_size'], 8)
        self.assertEqual(model_info['configuration']['embedding_dim'], 96)
        
        # Should detect missing tokenizer
        error_text = ' '.join(errors)
        self.assertIn('tokenizer', error_text.lower())
        
        print(f"Legacy v2 detected with configuration but missing tokenizer")
    
    def test_corrupted_model_detection(self):
        """Test detection of corrupted model files."""
        corrupted_dir = self._create_corrupted_model()
        
        # Test model validation
        is_valid, errors = self.manager.validate_model(corrupted_dir)
        
        # Should be invalid
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Get model info should handle corruption gracefully
        model_info = self.manager.get_model_info(corrupted_dir)
        self.assertFalse(model_info['is_valid'])
        self.assertIn('validation_errors', model_info)
        
        print(f"Corrupted model detected with {len(errors)} errors")


class TestBackwardCompatibilityErrorMessages(unittest.TestCase):
    """Test that backward compatibility errors provide helpful messages."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_minimal_legacy_model(self) -> str:
        """Create minimal legacy model for error testing."""
        model_dir = os.path.join(self.test_dir, "minimal_legacy")
        os.makedirs(model_dir, exist_ok=True)
        
        # Only create reservoir model, missing everything else
        reservoir_dir = os.path.join(model_dir, "reservoir_model")
        os.makedirs(reservoir_dir, exist_ok=True)
        
        with open(os.path.join(reservoir_dir, "saved_model.pb"), 'w') as f:
            f.write("minimal model")
        
        variables_dir = os.path.join(reservoir_dir, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
            f.write("minimal")
        with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
            f.write("minimal")
        
        return model_dir
    
    def test_inference_error_messages(self):
        """Test that inference with legacy models provides helpful error messages."""
        legacy_dir = self._create_minimal_legacy_model()
        
        # Try to create inference with legacy model
        try:
            inference = OptimizedLSMInference(legacy_dir)
            self.fail("Expected error when loading legacy model")
        except Exception as e:
            # Error message should be informative
            error_msg = str(e)
            self.assertGreater(len(error_msg), 20)  # Should have meaningful message
            
            # Should mention what's missing or wrong
            error_lower = error_msg.lower()
            self.assertTrue(
                any(keyword in error_lower for keyword in 
                    ['missing', 'not found', 'invalid', 'config', 'tokenizer']),
                f"Error message not informative enough: {error_msg}"
            )
            
            print(f"Helpful error message: {error_msg[:100]}...")
    
    def test_model_manager_error_messages(self):
        """Test that ModelManager provides helpful error messages for legacy models."""
        legacy_dir = self._create_minimal_legacy_model()
        
        manager = ModelManager()
        model_info = manager.get_model_info(legacy_dir)
        
        # Should have validation errors with helpful messages
        self.assertFalse(model_info['is_valid'])
        self.assertIn('validation_errors', model_info)
        self.assertGreater(len(model_info['validation_errors']), 0)
        
        # Error messages should be specific
        for error in model_info['validation_errors']:
            self.assertGreater(len(error), 10)  # Should be descriptive
            self.assertTrue(
                any(keyword in error.lower() for keyword in 
                    ['missing', 'not found', 'invalid', 'required']),
                f"Error message not specific enough: {error}"
            )
        
        print(f"Validation errors: {model_info['validation_errors']}")


class TestMigrationUtilities(unittest.TestCase):
    """Test migration utilities for old model formats."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_migration_detection(self):
        """Test detection of models that need migration."""
        # Create a model that could potentially be migrated
        model_dir = os.path.join(self.test_dir, "migration_candidate")
        os.makedirs(model_dir, exist_ok=True)
        
        # Has models and some config, but in old format
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("migration candidate model")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("migration variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("migration index")
        
        # Old-style config that could be converted
        old_config = {
            "window_size": 12,
            "embedding_dim": 256,
            "model_version": "1.5",
            "architecture": "standard_reservoir"
        }
        with open(os.path.join(model_dir, "old_config.json"), 'w') as f:
            json.dump(old_config, f)
        
        manager = ModelManager()
        model_info = manager.get_model_info(model_dir)
        
        # Should detect as invalid but potentially migratable
        self.assertFalse(model_info['is_valid'])
        
        # Should have some model components
        components = model_info.get('components', {})
        self.assertTrue(components.get('reservoir_model', False))
        self.assertTrue(components.get('cnn_model', False))
        
        print(f"Migration candidate detected: {len(model_info['validation_errors'])} issues")
    
    def test_migration_instructions(self):
        """Test that migration instructions are provided for old models."""
        # This test verifies that when old models are detected,
        # helpful migration instructions are provided
        
        model_dir = os.path.join(self.test_dir, "needs_migration")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create partial model structure
        reservoir_dir = os.path.join(model_dir, "reservoir_model")
        os.makedirs(reservoir_dir, exist_ok=True)
        with open(os.path.join(reservoir_dir, "saved_model.pb"), 'w') as f:
            f.write("old model")
        
        manager = ModelManager()
        
        # Get model summary which should include migration guidance
        summary = manager.get_model_summary(model_dir)
        
        # Summary should indicate invalid model
        self.assertIn("❌", summary)
        self.assertIn("Invalid Model", summary)
        
        # Should provide some indication of what's wrong
        self.assertTrue(len(summary) > 50)  # Should be informative
        
        print(f"Migration guidance in summary: {len(summary)} characters")


class TestGracefulFallbacks(unittest.TestCase):
    """Test graceful fallback mechanisms for legacy model handling."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_partial_model_loading(self):
        """Test handling of partially complete models."""
        partial_dir = os.path.join(self.test_dir, "partial_model")
        os.makedirs(partial_dir, exist_ok=True)
        
        # Create only reservoir model, missing CNN
        reservoir_dir = os.path.join(partial_dir, "reservoir_model")
        os.makedirs(reservoir_dir, exist_ok=True)
        
        with open(os.path.join(reservoir_dir, "saved_model.pb"), 'w') as f:
            f.write("partial model")
        
        variables_dir = os.path.join(reservoir_dir, "variables")
        os.makedirs(variables_dir, exist_ok=True)
        with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
            f.write("partial")
        with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
            f.write("partial")
        
        # Add minimal config
        config = ModelConfiguration(window_size=5, embedding_dim=32)
        config.save(os.path.join(partial_dir, "config.json"))
        
        manager = ModelManager()
        is_valid, errors = manager.validate_model(partial_dir)
        
        # Should gracefully handle partial model
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Should identify specific missing components
        error_text = ' '.join(errors)
        self.assertIn('CNN', error_text)
        
        print(f"Partial model handled gracefully: {len(errors)} specific errors")
    
    def test_cleanup_of_invalid_models(self):
        """Test cleanup functionality for invalid/legacy models."""
        # Create several invalid models
        invalid_models = []
        
        for i in range(3):
            invalid_dir = os.path.join(self.test_dir, f"invalid_model_{i}")
            os.makedirs(invalid_dir, exist_ok=True)
            
            # Create incomplete model structure
            with open(os.path.join(invalid_dir, "incomplete.txt"), 'w') as f:
                f.write("incomplete model")
            
            invalid_models.append(invalid_dir)
        
        manager = ModelManager(self.test_dir)
        
        # Test cleanup in dry run mode
        cleanup_candidates = manager.cleanup_incomplete_models(dry_run=True)
        
        # Should find the invalid models
        self.assertGreater(len(cleanup_candidates), 0)
        
        # Models should still exist after dry run
        for model_dir in invalid_models:
            self.assertTrue(os.path.exists(model_dir))
        
        print(f"Cleanup identified {len(cleanup_candidates)} invalid models")
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms for various failure scenarios."""
        # Test with completely empty directory
        empty_dir = os.path.join(self.test_dir, "empty_model")
        os.makedirs(empty_dir, exist_ok=True)
        
        manager = ModelManager()
        
        # Should handle empty directory gracefully
        model_info = manager.get_model_info(empty_dir)
        self.assertFalse(model_info['is_valid'])
        self.assertIn('validation_errors', model_info)
        
        # Test with directory containing only random files
        random_dir = os.path.join(self.test_dir, "random_files")
        os.makedirs(random_dir, exist_ok=True)
        
        with open(os.path.join(random_dir, "random.txt"), 'w') as f:
            f.write("random content")
        with open(os.path.join(random_dir, "another.log"), 'w') as f:
            f.write("log content")
        
        # Should handle random files gracefully
        random_info = manager.get_model_info(random_dir)
        self.assertFalse(random_info['is_valid'])
        
        print("Error recovery mechanisms working correctly")


def run_backward_compatibility_tests():
    """Run all backward compatibility tests."""
    print("🔄 Running Backward Compatibility Tests")
    print("=" * 50)
    
    test_suites = [
        ('Legacy Model Detection', TestBackwardCompatibilityDetection),
        ('Error Messages', TestBackwardCompatibilityErrorMessages),
        ('Migration Utilities', TestMigrationUtilities),
        ('Graceful Fallbacks', TestGracefulFallbacks),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_name, test_class in test_suites:
        print(f"\n📋 {suite_name}")
        print("-" * 30)
        
        # Create and run test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
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
            print(f"  ❌ {failures + errors}/{tests_run} tests failed")
            
            # Print failure details
            for test, traceback in result.failures + result.errors:
                print(f"    FAILED: {test}")
                if 'AssertionError:' in traceback:
                    error_msg = traceback.split('AssertionError:')[-1].strip()
                    print(f"    {error_msg}")
        else:
            print(f"  ✅ All {tests_run} tests passed")
    
    # Summary
    print(f"\n📊 Backward Compatibility Test Summary")
    print("=" * 50)
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    
    if failed_tests:
        print(f"Failed test suites: {', '.join(failed_tests)}")
        return False
    else:
        print("🎉 All backward compatibility tests passed!")
        return True


if __name__ == "__main__":
    success = run_backward_compatibility_tests()
    exit(0 if success else 1)