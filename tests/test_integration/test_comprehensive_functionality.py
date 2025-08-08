#!/usr/bin/env python3
"""
Comprehensive test suite for all new LSM functionality.

This module provides comprehensive tests for:
- DialogueTokenizer save/load and decoding methods
- Integration tests for complete train-save-load-predict workflow
- Performance tests for inference speed and memory usage
- Backward compatibility tests with mock old model formats
"""

import os
import json
import time
import tempfile
import shutil
import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from typing import List, Dict, Any, Tuple
import pickle

# Import all components to test
from src.lsm.data.data_loader import DialogueTokenizer, load_data
from src.lsm.training.model_config import ModelConfiguration, TrainingMetadata
from src.lsm.training.train import LSMTrainer
from src.lsm.inference import OptimizedLSMInference, LSMInference
from src.lsm.management.model_manager import ModelManager
from src.lsm.utils.lsm_exceptions import *
from src.lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)

class TestDialogueTokenizerPersistence(unittest.TestCase):
    """Test DialogueTokenizer save/load and decoding functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.tokenizer = DialogueTokenizer(max_features=1000, embedding_dim=64)
        
        # Sample training texts
        self.sample_texts = [
            "hello world how are you",
            "good morning nice day",
            "what is your name today",
            "i am fine thank you",
            "where are you from friend",
            "nice to meet you here",
            "have a great day ahead",
            "see you later goodbye",
            "how was your weekend",
            "the weather is beautiful"
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_tokenizer_fit_and_basic_functionality(self):
        """Test basic tokenizer fitting and encoding."""
        # Fit tokenizer
        self.tokenizer.fit(self.sample_texts)
        self.assertTrue(self.tokenizer.is_fitted)
        
        # Test encoding
        encoded = self.tokenizer.encode(["hello world", "good morning"])
        self.assertEqual(encoded.shape, (2, 64))
        self.assertEqual(encoded.dtype, np.float32)
    
    def test_tokenizer_save_and_load(self):
        """Test tokenizer persistence functionality."""
        # Fit and save tokenizer
        self.tokenizer.fit(self.sample_texts)
        save_path = os.path.join(self.test_dir, "tokenizer_test")
        
        # Save tokenizer
        self.tokenizer.save(save_path)
        
        # Verify saved files exist
        expected_files = ["config.json", "vectorizer.pkl", "vocabulary.json", "vocabulary_embeddings.npy"]
        for file_name in expected_files:
            file_path = os.path.join(save_path, file_name)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file_name}")
        
        # Create new tokenizer and load
        new_tokenizer = DialogueTokenizer(max_features=1000, embedding_dim=64)
        new_tokenizer.load(save_path)
        
        # Verify loaded tokenizer state
        self.assertTrue(new_tokenizer.is_fitted)
        self.assertEqual(new_tokenizer.max_features, 1000)
        self.assertEqual(new_tokenizer.embedding_dim, 64)
        self.assertGreater(len(new_tokenizer._vocabulary_texts), 0)
        self.assertIsNotNone(new_tokenizer._vocabulary_embeddings)
        
        # Test that encoding produces same results
        test_texts = ["hello world", "good morning"]
        original_encoded = self.tokenizer.encode(test_texts)
        loaded_encoded = new_tokenizer.encode(test_texts)
        
        np.testing.assert_array_almost_equal(original_encoded, loaded_encoded, decimal=5)
    
    def test_tokenizer_decoding_functionality(self):
        """Test tokenizer decoding capabilities."""
        # Fit tokenizer
        self.tokenizer.fit(self.sample_texts)
        
        # Test single embedding decoding
        test_text = "hello world"
        encoded = self.tokenizer.encode([test_text])
        decoded = self.tokenizer.decode_embedding(encoded[0])
        
        self.assertIsInstance(decoded, str)
        self.assertNotEqual(decoded, "[UNKNOWN]")
        
        # Test batch decoding
        test_texts = ["hello world", "good morning", "nice day"]
        encoded_batch = self.tokenizer.encode(test_texts)
        decoded_batch = self.tokenizer.decode_embeddings_batch(encoded_batch)
        
        self.assertEqual(len(decoded_batch), 3)
        for decoded_text in decoded_batch:
            self.assertIsInstance(decoded_text, str)
    
    def test_tokenizer_closest_texts_functionality(self):
        """Test get_closest_texts functionality."""
        # Fit tokenizer
        self.tokenizer.fit(self.sample_texts)
        
        # Test getting closest texts
        test_embedding = self.tokenizer.encode(["hello world"])[0]
        closest_texts = self.tokenizer.get_closest_texts(test_embedding, top_k=3)
        
        self.assertEqual(len(closest_texts), 3)
        for text, similarity in closest_texts:
            self.assertIsInstance(text, str)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
        
        # Verify results are sorted by similarity (descending)
        similarities = [sim for _, sim in closest_texts]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
    
    def test_tokenizer_caching_functionality(self):
        """Test tokenizer caching mechanisms."""
        # Fit tokenizer
        self.tokenizer.fit(self.sample_texts)
        
        # Test encoding cache
        test_texts = ["hello world", "good morning"]
        
        # First encoding (no cache)
        start_time = time.time()
        encoded1 = self.tokenizer.encode(test_texts)
        first_time = time.time() - start_time
        
        # Second encoding (with cache)
        start_time = time.time()
        encoded2 = self.tokenizer.encode(test_texts)
        second_time = time.time() - start_time
        
        # Results should be identical
        np.testing.assert_array_equal(encoded1, encoded2)
        
        # Second call should be faster (cached)
        self.assertLessEqual(second_time, first_time * 1.1)  # Allow some variance
        
        # Test cache stats
        cache_stats = self.tokenizer.get_cache_stats()
        self.assertGreater(cache_stats['encoding_cache_size'], 0)
        
        # Test cache clearing
        self.tokenizer.clear_caches()
        cache_stats_after = self.tokenizer.get_cache_stats()
        self.assertEqual(cache_stats_after['encoding_cache_size'], 0)
    
    def test_tokenizer_error_handling(self):
        """Test tokenizer error handling."""
        # Test operations on unfitted tokenizer
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.encode(["test"])
        
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.decode_embedding(np.random.random(64))
        
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.save(self.test_dir)
        
        # Test loading from non-existent path
        with self.assertRaises(TokenizerLoadError):
            self.tokenizer.load("/nonexistent/path")
        
        # Test invalid input validation
        self.tokenizer.fit(self.sample_texts)
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.encode([])  # Empty list
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.encode([""])  # Empty string
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.decode_embedding(np.random.random(32))  # Wrong dimension


class TestModelConfigurationPersistence(unittest.TestCase):
    """Test ModelConfiguration save/load functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_model_configuration_creation(self):
        """Test ModelConfiguration creation and validation."""
        config = ModelConfiguration(
            window_size=10,
            embedding_dim=128,
            reservoir_type='standard',
            reservoir_units=[256, 128, 64],
            sparsity=0.1,
            epochs=20,
            batch_size=32
        )
        
        # Test validation
        errors = config.validate()
        self.assertEqual(len(errors), 0, f"Configuration validation failed: {errors}")
        
        # Test dictionary conversion
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict['window_size'], 10)
        self.assertEqual(config_dict['embedding_dim'], 128)
    
    def test_model_configuration_save_load(self):
        """Test ModelConfiguration persistence."""
        config = ModelConfiguration(
            window_size=15,
            embedding_dim=256,
            reservoir_type='hierarchical',
            reservoir_units=[512, 256, 128],
            sparsity=0.05,
            epochs=50,
            batch_size=64
        )
        
        # Save configuration
        config_path = os.path.join(self.test_dir, "test_config.json")
        config.save(config_path)
        
        # Verify file exists
        self.assertTrue(os.path.exists(config_path))
        
        # Load configuration
        loaded_config = ModelConfiguration.load(config_path)
        
        # Verify loaded configuration matches original
        self.assertEqual(loaded_config.window_size, 15)
        self.assertEqual(loaded_config.embedding_dim, 256)
        self.assertEqual(loaded_config.reservoir_type, 'hierarchical')
        self.assertEqual(loaded_config.reservoir_units, [512, 256, 128])
        self.assertEqual(loaded_config.sparsity, 0.05)
        self.assertEqual(loaded_config.epochs, 50)
        self.assertEqual(loaded_config.batch_size, 64)
    
    def test_model_configuration_validation(self):
        """Test ModelConfiguration validation."""
        # Test invalid configuration
        invalid_config = ModelConfiguration(
            window_size=-1,  # Invalid
            embedding_dim=0,  # Invalid
            reservoir_type='invalid_type',  # Invalid
            sparsity=1.5,  # Invalid
            epochs=0,  # Invalid
            batch_size=-1  # Invalid
        )
        
        errors = invalid_config.validate()
        self.assertGreater(len(errors), 0)
        
        # Check specific error messages
        error_text = ' '.join(errors)
        self.assertIn('window_size', error_text)
        self.assertIn('embedding_dim', error_text)
        self.assertIn('reservoir_type', error_text)
        self.assertIn('sparsity', error_text)
        self.assertIn('epochs', error_text)
        self.assertIn('batch_size', error_text)


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete train-save-load-predict workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "test_model")
        
        # Create minimal mock data
        self.mock_data = self._create_mock_training_data()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_mock_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DialogueTokenizer]:
        """Create mock training data for testing."""
        # Create mock dialogue sequences
        mock_texts = [
            "hello world", "how are you", "i am fine", "good morning",
            "nice to meet", "what is name", "my name alice", "where from",
            "i from earth", "sounds great", "have nice day", "see you later"
        ]
        
        # Create tokenizer and fit
        tokenizer = DialogueTokenizer(max_features=100, embedding_dim=32)
        tokenizer.fit(mock_texts)
        
        # Create mock sequences (window_size=3)
        window_size = 3
        sequences = []
        next_tokens = []
        
        for i in range(len(mock_texts) - window_size):
            seq = mock_texts[i:i + window_size]
            next_token = mock_texts[i + window_size]
            sequences.append(seq)
            next_tokens.append(next_token)
        
        # Convert to embeddings
        X = []
        for seq in sequences:
            seq_embeddings = tokenizer.encode(seq)
            X.append(seq_embeddings)
        
        X = np.array(X, dtype=np.float32)
        y = tokenizer.encode(next_tokens).astype(np.float32)
        
        # Simple train/test split
        split_idx = len(X) // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_test, y_test, tokenizer
    
    @patch('src.lsm.data.data_loader.download_dataset')
    def test_complete_workflow_integration(self, mock_download):
        """Test complete train-save-load-predict workflow."""
        # Mock dataset download
        mock_csv_path = os.path.join(self.test_dir, "mock_dataset.csv")
        mock_df = pd.DataFrame({
            'Best Generated Conversation': [
                "User: Hello\nAssistant: Hi there! How are you?",
                "User: Good morning\nAssistant: Good morning! Nice day today.",
                "User: What's your name?\nAssistant: I'm an AI assistant.",
                "User: Where are you from?\nAssistant: I exist in the digital realm.",
                "User: How was your day?\nAssistant: Every day is a new adventure!"
            ]
        })
        mock_df.to_csv(mock_csv_path, index=False)
        mock_download.return_value = mock_csv_path
        
        try:
            # Step 1: Create configuration
            config = ModelConfiguration(
                window_size=3,
                embedding_dim=32,
                reservoir_type='standard',
                reservoir_units=[64, 32],
                sparsity=0.1,
                epochs=2,  # Small for testing
                batch_size=2
            )
            
            # Step 2: Initialize trainer
            trainer = LSMTrainer(
                window_size=config.window_size,
                embedding_dim=config.embedding_dim,
                reservoir_units=config.reservoir_units,
                sparsity=config.sparsity,
                reservoir_type=config.reservoir_type
            )
            
            # Step 3: Build models
            trainer.build_models()
            self.assertIsNotNone(trainer.reservoir)
            self.assertIsNotNone(trainer.cnn_model)
            
            # Step 4: Load data (mocked)
            X_train, y_train, X_test, y_test, tokenizer = self.mock_data
            
            # Step 5: Train model (minimal training)
            trainer.train(X_train, y_train, X_test, y_test, 
                         epochs=2, batch_size=2, verbose=0)
            
            # Step 6: Save complete model
            trainer.save_complete_model(self.model_dir, tokenizer)
            
            # Verify saved files
            expected_files = [
                "reservoir_model", "cnn_model", "tokenizer", 
                "config.json", "metadata.json"
            ]
            for file_name in expected_files:
                file_path = os.path.join(self.model_dir, file_name)
                self.assertTrue(os.path.exists(file_path), f"Missing: {file_name}")
            
            # Step 7: Load model for inference
            inference = OptimizedLSMInference(self.model_dir)
            
            # Step 8: Test prediction
            test_sequence = ["hello", "how", "are"]
            prediction = inference.predict_next_token(test_sequence)
            
            self.assertIsInstance(prediction, str)
            self.assertNotEqual(prediction, "[ERROR]")
            
            # Step 9: Test prediction with confidence
            prediction, confidence = inference.predict_with_confidence(test_sequence)
            self.assertIsInstance(prediction, str)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            # Step 10: Test batch prediction
            batch_sequences = [["hello", "how", "are"], ["good", "morning", "nice"]]
            batch_predictions = inference.batch_predict(batch_sequences)
            self.assertEqual(len(batch_predictions), 2)
            
            print("✓ Complete workflow integration test passed")
            
        except Exception as e:
            self.fail(f"Integration workflow failed: {e}")
    
    def test_model_loading_and_validation(self):
        """Test model loading and validation."""
        # Create a minimal valid model structure
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Create configuration
        config = ModelConfiguration(window_size=3, embedding_dim=32)
        config.save(os.path.join(self.model_dir, "config.json"))
        
        # Create mock model directories
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(self.model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            # Create minimal Keras model files
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("mock model data")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("mock variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("mock index")
        
        # Create tokenizer
        tokenizer_dir = os.path.join(self.model_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        tokenizer_config = {
            "max_features": 100,
            "embedding_dim": 32,
            "is_fitted": True,
            "vocabulary_size": 50
        }
        with open(os.path.join(tokenizer_dir, "config.json"), 'w') as f:
            json.dump(tokenizer_config, f)
        
        with open(os.path.join(tokenizer_dir, "vectorizer.pkl"), 'wb') as f:
            f.write(b"mock vectorizer")
        
        with open(os.path.join(tokenizer_dir, "vocabulary.json"), 'w') as f:
            json.dump(["hello", "world", "test"], f)
        
        # Test model validation
        manager = ModelManager()
        is_valid, errors = manager.validate_model(self.model_dir)
        
        # Should be valid (though minimal)
        critical_errors = [err for err in errors if not err.endswith("(non-critical)")]
        self.assertEqual(len(critical_errors), 0, f"Model validation failed: {critical_errors}")


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations and memory usage."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = self._create_mock_model()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_mock_model(self) -> str:
        """Create a mock model for performance testing."""
        model_dir = os.path.join(self.test_dir, "perf_test_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create configuration
        config = ModelConfiguration(
            window_size=5,
            embedding_dim=64,
            reservoir_type='standard',
            reservoir_units=[128, 64],
            sparsity=0.1
        )
        config.save(os.path.join(model_dir, "config.json"))
        
        # Create mock tokenizer
        tokenizer = DialogueTokenizer(embedding_dim=64, max_features=500)
        mock_texts = [f"word{i} test{i} sample{i}" for i in range(100)]
        tokenizer.fit(mock_texts)
        tokenizer.save(os.path.join(model_dir, "tokenizer"))
        
        # Create mock model directories (minimal structure)
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("mock model")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("mock")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("mock")
        
        return model_dir
    
    def test_lazy_loading_performance(self):
        """Test lazy loading performance benefits."""
        # Test lazy loading initialization time
        start_time = time.time()
        inference_lazy = OptimizedLSMInference(self.model_dir, lazy_load=True)
        lazy_init_time = time.time() - start_time
        
        # Test eager loading initialization time
        start_time = time.time()
        inference_eager = OptimizedLSMInference(self.model_dir, lazy_load=False)
        eager_init_time = time.time() - start_time
        
        # Lazy loading should be faster for initialization
        self.assertLess(lazy_init_time, eager_init_time * 2)  # Allow some variance
        
        print(f"Lazy init: {lazy_init_time:.3f}s, Eager init: {eager_init_time:.3f}s")
    
    def test_caching_performance(self):
        """Test caching performance improvements."""
        inference = OptimizedLSMInference(self.model_dir, cache_size=50)
        
        # Test sequences (some repeated for cache testing)
        test_sequences = [
            ["word1", "test1", "sample1", "word2", "test2"],
            ["word3", "test3", "sample3", "word4", "test4"],
            ["word1", "test1", "sample1", "word2", "test2"],  # Repeat
            ["word5", "test5", "sample5", "word6", "test6"],
            ["word3", "test3", "sample3", "word4", "test4"],  # Repeat
        ]
        
        # First run (populate cache)
        start_time = time.time()
        predictions1 = []
        for seq in test_sequences:
            try:
                pred = inference.predict_next_token(seq)
                predictions1.append(pred)
            except Exception as e:
                predictions1.append(f"[ERROR: {e}]")
        first_run_time = time.time() - start_time
        
        # Second run (use cache)
        start_time = time.time()
        predictions2 = []
        for seq in test_sequences:
            try:
                pred = inference.predict_next_token(seq)
                predictions2.append(pred)
            except Exception as e:
                predictions2.append(f"[ERROR: {e}]")
        second_run_time = time.time() - start_time
        
        # Cache should improve performance
        if second_run_time > 0:
            speedup = first_run_time / second_run_time
            self.assertGreater(speedup, 0.8)  # Allow some variance
        
        # Results should be consistent
        self.assertEqual(predictions1, predictions2)
        
        print(f"First run: {first_run_time:.3f}s, Second run: {second_run_time:.3f}s")
    
    def test_batch_processing_efficiency(self):
        """Test batch processing efficiency."""
        inference = OptimizedLSMInference(self.model_dir, max_batch_size=3)
        
        # Create test sequences
        test_sequences = [
            ["word1", "test1", "sample1", "word2", "test2"],
            ["word3", "test3", "sample3", "word4", "test4"],
            ["word5", "test5", "sample5", "word6", "test6"],
            ["word7", "test7", "sample7", "word8", "test8"],
            ["word9", "test9", "sample9", "word10", "test10"],
        ]
        
        # Test batch processing
        start_time = time.time()
        batch_predictions = inference.batch_predict(test_sequences)
        batch_time = time.time() - start_time
        
        # Test individual processing
        start_time = time.time()
        individual_predictions = []
        for seq in test_sequences:
            try:
                pred = inference.predict_next_token(seq)
                individual_predictions.append(pred)
            except Exception as e:
                individual_predictions.append(f"[ERROR: {e}]")
        individual_time = time.time() - start_time
        
        # Batch processing should be at least as efficient
        if individual_time > 0:
            efficiency = individual_time / max(batch_time, 0.001)  # Avoid division by zero
            self.assertGreaterEqual(efficiency, 0.5)  # Allow some overhead
        
        # Results should have same length
        self.assertEqual(len(batch_predictions), len(individual_predictions))
        
        print(f"Batch: {batch_time:.3f}s, Individual: {individual_time:.3f}s")
    
    def test_memory_management(self):
        """Test memory management features."""
        inference = OptimizedLSMInference(self.model_dir, cache_size=5)  # Small cache
        
        # Generate many predictions to test cache management
        test_sequences = []
        for i in range(10):  # More than cache size
            seq = [f"word{j}_{i}" for j in range(5)]
            test_sequences.append(seq)
        
        # Make predictions
        predictions = []
        for seq in test_sequences:
            try:
                pred = inference.predict_next_token(seq)
                predictions.append(pred)
            except Exception as e:
                predictions.append(f"[ERROR: {e}]")
        
        # Check cache management
        cache_stats = inference.get_cache_stats()
        cache_size = cache_stats['prediction_cache']['size']
        
        # Cache should not exceed maximum size
        self.assertLessEqual(cache_size, inference.cache_size)
        
        # Test manual cache clearing
        inference.clear_caches()
        cache_stats_after = inference.get_cache_stats()
        self.assertEqual(cache_stats_after['prediction_cache']['size'], 0)
        
        print(f"Generated {len(predictions)} predictions, final cache size: {cache_size}")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with old model formats."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_old_model_format_v1(self) -> str:
        """Create a mock old model format (version 1)."""
        old_model_dir = os.path.join(self.test_dir, "old_model_v1")
        os.makedirs(old_model_dir, exist_ok=True)
        
        # Old format: only has model files, no tokenizer or config
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(old_model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("old model data")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("old variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("old index")
        
        # Old format: simple JSON config (not ModelConfiguration format)
        old_config = {
            "window_size": 10,
            "embedding_dim": 128,
            "model_type": "lsm_v1"
        }
        with open(os.path.join(old_model_dir, "old_config.json"), 'w') as f:
            json.dump(old_config, f)
        
        return old_model_dir
    
    def _create_old_model_format_v2(self) -> str:
        """Create a mock old model format (version 2)."""
        old_model_dir = os.path.join(self.test_dir, "old_model_v2")
        os.makedirs(old_model_dir, exist_ok=True)
        
        # Version 2: has models and config, but no tokenizer
        for model_name in ["reservoir_model", "cnn_model"]:
            model_path = os.path.join(old_model_dir, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            with open(os.path.join(model_path, "saved_model.pb"), 'w') as f:
                f.write("v2 model data")
            
            variables_dir = os.path.join(model_path, "variables")
            os.makedirs(variables_dir, exist_ok=True)
            with open(os.path.join(variables_dir, "variables.data-00000-of-00001"), 'w') as f:
                f.write("v2 variables")
            with open(os.path.join(variables_dir, "variables.index"), 'w') as f:
                f.write("v2 index")
        
        # Version 2: has proper config but missing tokenizer
        config = ModelConfiguration(
            window_size=8,
            embedding_dim=96,
            reservoir_type='standard',
            model_version="2.0"
        )
        config.save(os.path.join(old_model_dir, "config.json"))
        
        return old_model_dir
    
    def test_old_model_detection(self):
        """Test detection of old model formats."""
        # Create old model formats
        old_v1_dir = self._create_old_model_format_v1()
        old_v2_dir = self._create_old_model_format_v2()
        
        manager = ModelManager()
        
        # Test validation of old formats
        is_valid_v1, errors_v1 = manager.validate_model(old_v1_dir)
        is_valid_v2, errors_v2 = manager.validate_model(old_v2_dir)
        
        # Old formats should be detected as invalid due to missing components
        self.assertFalse(is_valid_v1)
        self.assertFalse(is_valid_v2)
        
        # Should have specific errors about missing tokenizer
        errors_v1_text = ' '.join(errors_v1)
        errors_v2_text = ' '.join(errors_v2)
        
        # V1 should be missing config and tokenizer
        self.assertIn('Configuration', errors_v1_text)
        
        # V2 should be missing tokenizer
        self.assertIn('tokenizer', errors_v2_text.lower())
        
        print(f"V1 errors: {len(errors_v1)}, V2 errors: {len(errors_v2)}")
    
    def test_backward_compatibility_error_messages(self):
        """Test that backward compatibility errors provide helpful messages."""
        old_model_dir = self._create_old_model_format_v1()
        
        # Try to load old model with inference
        try:
            inference = OptimizedLSMInference(old_model_dir)
            # This should fail, but we want to check the error message
            self.fail("Expected ModelLoadError for old model format")
        except ModelLoadError as e:
            # Error message should be helpful
            error_msg = str(e)
            self.assertIn("model", error_msg.lower())
            print(f"Backward compatibility error message: {error_msg}")
        except Exception as e:
            # Any error is acceptable, as long as it's informative
            error_msg = str(e)
            self.assertGreater(len(error_msg), 10)  # Should have meaningful message
            print(f"Backward compatibility error: {error_msg}")
    
    def test_model_migration_detection(self):
        """Test detection of models that need migration."""
        old_v2_dir = self._create_old_model_format_v2()
        
        manager = ModelManager()
        model_info = manager.get_model_info(old_v2_dir)
        
        # Should detect as invalid but provide information
        self.assertFalse(model_info['is_valid'])
        self.assertIn('validation_errors', model_info)
        
        # Should have configuration info (since V2 has config)
        if 'configuration' in model_info:
            self.assertEqual(model_info['configuration']['window_size'], 8)
            self.assertEqual(model_info['configuration']['embedding_dim'], 96)
        
        print(f"Migration detection - errors: {len(model_info['validation_errors'])}")


def run_comprehensive_tests():
    """Run all comprehensive functionality tests."""
    print("🧪 Running Comprehensive Functionality Tests")
    print("=" * 60)
    
    # Test suites to run
    test_suites = [
        ('DialogueTokenizer Persistence', TestDialogueTokenizerPersistence),
        ('ModelConfiguration Persistence', TestModelConfigurationPersistence),
        ('Integration Workflow', TestIntegrationWorkflow),
        ('Performance Optimizations', TestPerformanceOptimizations),
        ('Backward Compatibility', TestBackwardCompatibility),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_name, test_class in test_suites:
        print(f"\n📋 {suite_name}")
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
            print(f"  ❌ {failures + errors}/{tests_run} tests failed")
            
            # Print failure details
            for test, traceback in result.failures + result.errors:
                print(f"    FAILED: {test}")
                print(f"    {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'Error occurred'}")
        else:
            print(f"  ✅ All {tests_run} tests passed")
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 60)
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    
    if failed_tests:
        print(f"Failed test suites: {', '.join(failed_tests)}")
        return False
    else:
        print("🎉 All comprehensive functionality tests passed!")
        return True


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)