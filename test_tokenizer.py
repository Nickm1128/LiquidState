#!/usr/bin/env python3
"""
Focused unit tests for DialogueTokenizer save/load and decoding methods.

This module provides detailed tests specifically for the enhanced DialogueTokenizer
functionality including persistence, decoding, and caching mechanisms.
"""

import os
import json
import time
import tempfile
import shutil
import numpy as np
import unittest
from unittest.mock import patch, MagicMock
import pickle

from data_loader import DialogueTokenizer
from lsm_exceptions import (
    TokenizerError, TokenizerNotFittedError, TokenizerLoadError, 
    TokenizerSaveError, InvalidInputError
)

class TestDialogueTokenizerCore(unittest.TestCase):
    """Test core DialogueTokenizer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.tokenizer = DialogueTokenizer(max_features=1000, embedding_dim=128)
        
        # Comprehensive sample texts for testing
        self.sample_texts = [
            "hello world how are you doing today",
            "good morning everyone nice to see you",
            "what is your name and where are you from",
            "i am fine thank you for asking me",
            "the weather is beautiful today isn't it",
            "let's go for a walk in the park",
            "have you seen the latest movie release",
            "i love reading books in my free time",
            "technology is advancing very rapidly these days",
            "artificial intelligence is changing the world",
            "machine learning algorithms are quite fascinating",
            "natural language processing is a complex field",
            "deep learning models require lots of data",
            "neural networks can learn complex patterns",
            "computer vision applications are everywhere now"
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization with different parameters."""
        # Test default initialization
        tokenizer1 = DialogueTokenizer()
        self.assertEqual(tokenizer1.max_features, 10000)
        self.assertEqual(tokenizer1.embedding_dim, 128)
        self.assertFalse(tokenizer1.is_fitted)
        
        # Test custom initialization
        tokenizer2 = DialogueTokenizer(max_features=5000, embedding_dim=256)
        self.assertEqual(tokenizer2.max_features, 5000)
        self.assertEqual(tokenizer2.embedding_dim, 256)
        self.assertFalse(tokenizer2.is_fitted)
    
    def test_tokenizer_fitting(self):
        """Test tokenizer fitting process."""
        # Test fitting with valid texts
        self.tokenizer.fit(self.sample_texts)
        
        self.assertTrue(self.tokenizer.is_fitted)
        self.assertGreater(len(self.tokenizer._vocabulary_texts), 0)
        self.assertIsNotNone(self.tokenizer._vocabulary_embeddings)
        self.assertEqual(len(self.tokenizer._vocabulary_texts), 
                        len(self.tokenizer._vocabulary_embeddings))
        
        # Test that text-to-embedding mapping is created
        self.assertGreater(len(self.tokenizer._text_to_embedding), 0)
    
    def test_tokenizer_encoding(self):
        """Test tokenizer encoding functionality."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test single text encoding
        test_texts = ["hello world", "good morning"]
        encoded = self.tokenizer.encode(test_texts)
        
        self.assertEqual(encoded.shape, (2, 128))
        self.assertEqual(encoded.dtype, np.float32)
        
        # Test that encoding is deterministic
        encoded2 = self.tokenizer.encode(test_texts)
        np.testing.assert_array_equal(encoded, encoded2)
        
        # Test encoding with different text lengths
        varied_texts = ["hi", "hello world how are you", "a"]
        encoded_varied = self.tokenizer.encode(varied_texts)
        self.assertEqual(encoded_varied.shape, (3, 128))
    
    def test_tokenizer_decoding_single(self):
        """Test single embedding decoding."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test decoding of known embeddings
        test_text = "hello world"
        encoded = self.tokenizer.encode([test_text])
        decoded = self.tokenizer.decode_embedding(encoded[0])
        
        self.assertIsInstance(decoded, str)
        self.assertNotEqual(decoded, "[UNKNOWN]")
        self.assertNotEqual(decoded, "[ERROR]")
        
        # Test decoding with random embedding (should return fallback)
        random_embedding = np.random.random(128).astype(np.float32)
        decoded_random = self.tokenizer.decode_embedding(random_embedding)
        self.assertIsInstance(decoded_random, str)
    
    def test_tokenizer_decoding_batch(self):
        """Test batch embedding decoding."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test batch decoding
        test_texts = ["hello world", "good morning", "nice day"]
        encoded_batch = self.tokenizer.encode(test_texts)
        decoded_batch = self.tokenizer.decode_embeddings_batch(encoded_batch)
        
        self.assertEqual(len(decoded_batch), 3)
        for decoded_text in decoded_batch:
            self.assertIsInstance(decoded_text, str)
            self.assertNotEqual(decoded_text, "[ERROR]")
        
        # Test empty batch
        empty_batch = np.array([]).reshape(0, 128)
        decoded_empty = self.tokenizer.decode_embeddings_batch(empty_batch)
        self.assertEqual(len(decoded_empty), 0)
    
    def test_tokenizer_closest_texts(self):
        """Test get_closest_texts functionality."""
        self.tokenizer.fit(self.sample_texts)
        
        # Test getting closest texts
        test_embedding = self.tokenizer.encode(["hello world"])[0]
        closest_texts = self.tokenizer.get_closest_texts(test_embedding, top_k=5)
        
        self.assertEqual(len(closest_texts), 5)
        
        # Verify structure of results
        for text, similarity in closest_texts:
            self.assertIsInstance(text, str)
            self.assertIsInstance(similarity, float)
            self.assertGreaterEqual(similarity, 0.0)
            self.assertLessEqual(similarity, 1.0)
        
        # Verify results are sorted by similarity (descending)
        similarities = [sim for _, sim in closest_texts]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
        
        # Test with different k values
        for k in [1, 3, 10]:
            closest_k = self.tokenizer.get_closest_texts(test_embedding, top_k=k)
            self.assertEqual(len(closest_k), min(k, len(self.tokenizer._vocabulary_texts)))


class TestDialogueTokenizerPersistence(unittest.TestCase):
    """Test DialogueTokenizer save/load functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.tokenizer = DialogueTokenizer(max_features=500, embedding_dim=64)
        
        self.sample_texts = [
            "hello world", "good morning", "nice day", "how are you",
            "i am fine", "thank you", "see you later", "goodbye friend",
            "what is new", "nothing much", "just working", "having fun"
        ]
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_save_fitted_tokenizer(self):
        """Test saving a fitted tokenizer."""
        # Fit tokenizer
        self.tokenizer.fit(self.sample_texts)
        
        # Save tokenizer
        save_path = os.path.join(self.test_dir, "tokenizer_save_test")
        self.tokenizer.save(save_path)
        
        # Verify all required files are created
        expected_files = [
            "config.json",
            "vectorizer.pkl", 
            "vocabulary.json",
            "vocabulary_embeddings.npy"
        ]
        
        for file_name in expected_files:
            file_path = os.path.join(save_path, file_name)
            self.assertTrue(os.path.exists(file_path), f"Missing file: {file_name}")
        
        # Verify config file content
        with open(os.path.join(save_path, "config.json"), 'r') as f:
            config = json.load(f)
        
        self.assertEqual(config["max_features"], 500)
        self.assertEqual(config["embedding_dim"], 64)
        self.assertTrue(config["is_fitted"])
        self.assertGreater(config["vocabulary_size"], 0)
        
        # Verify vocabulary file content
        with open(os.path.join(save_path, "vocabulary.json"), 'r') as f:
            vocabulary = json.load(f)
        
        self.assertIsInstance(vocabulary, list)
        self.assertGreater(len(vocabulary), 0)
        
        # Verify embeddings file
        embeddings = np.load(os.path.join(save_path, "vocabulary_embeddings.npy"))
        self.assertEqual(embeddings.shape[1], 64)
        self.assertEqual(embeddings.shape[0], len(vocabulary))
    
    def test_load_tokenizer(self):
        """Test loading a saved tokenizer."""
        # Fit and save tokenizer
        self.tokenizer.fit(self.sample_texts)
        save_path = os.path.join(self.test_dir, "tokenizer_load_test")
        self.tokenizer.save(save_path)
        
        # Create new tokenizer and load
        new_tokenizer = DialogueTokenizer(max_features=500, embedding_dim=64)
        new_tokenizer.load(save_path)
        
        # Verify loaded state
        self.assertTrue(new_tokenizer.is_fitted)
        self.assertEqual(new_tokenizer.max_features, 500)
        self.assertEqual(new_tokenizer.embedding_dim, 64)
        self.assertGreater(len(new_tokenizer._vocabulary_texts), 0)
        self.assertIsNotNone(new_tokenizer._vocabulary_embeddings)
        
        # Test that loaded tokenizer produces same results
        test_texts = ["hello world", "good morning"]
        original_encoded = self.tokenizer.encode(test_texts)
        loaded_encoded = new_tokenizer.encode(test_texts)
        
        np.testing.assert_array_almost_equal(original_encoded, loaded_encoded, decimal=5)
        
        # Test decoding consistency
        original_decoded = self.tokenizer.decode_embedding(original_encoded[0])
        loaded_decoded = new_tokenizer.decode_embedding(loaded_encoded[0])
        self.assertEqual(original_decoded, loaded_decoded)
    
    def test_save_unfitted_tokenizer_error(self):
        """Test that saving unfitted tokenizer raises error."""
        save_path = os.path.join(self.test_dir, "unfitted_save_test")
        
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.save(save_path)
    
    def test_load_nonexistent_path_error(self):
        """Test loading from non-existent path raises error."""
        nonexistent_path = os.path.join(self.test_dir, "nonexistent")
        
        with self.assertRaises(TokenizerLoadError):
            self.tokenizer.load(nonexistent_path)
    
    def test_load_incomplete_tokenizer_error(self):
        """Test loading incomplete tokenizer data raises error."""
        incomplete_path = os.path.join(self.test_dir, "incomplete_tokenizer")
        os.makedirs(incomplete_path, exist_ok=True)
        
        # Create only config file, missing other required files
        config = {
            "max_features": 500,
            "embedding_dim": 64,
            "is_fitted": True,
            "vocabulary_size": 10
        }
        with open(os.path.join(incomplete_path, "config.json"), 'w') as f:
            json.dump(config, f)
        
        with self.assertRaises(TokenizerLoadError):
            self.tokenizer.load(incomplete_path)
    
    def test_load_dimension_mismatch_error(self):
        """Test loading tokenizer with dimension mismatch raises error."""
        # Create tokenizer with different embedding dimension
        different_tokenizer = DialogueTokenizer(max_features=500, embedding_dim=32)
        different_tokenizer.fit(self.sample_texts)
        
        save_path = os.path.join(self.test_dir, "dimension_mismatch_test")
        different_tokenizer.save(save_path)
        
        # Try to load with tokenizer having different embedding dimension
        with self.assertRaises(TokenizerLoadError):
            self.tokenizer.load(save_path)  # self.tokenizer has embedding_dim=64


class TestDialogueTokenizerCaching(unittest.TestCase):
    """Test DialogueTokenizer caching mechanisms."""
    
    def setUp(self):
        """Set up test environment."""
        self.tokenizer = DialogueTokenizer(max_features=200, embedding_dim=32)
        
        self.sample_texts = [
            "cache test one", "cache test two", "cache test three",
            "performance test", "speed test", "memory test",
            "hello world", "good morning", "nice day"
        ]
        
        self.tokenizer.fit(self.sample_texts)
    
    def test_encoding_cache(self):
        """Test encoding cache functionality."""
        test_texts = ["cache test one", "cache test two"]
        
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
        
        # Second call should be faster or similar (cached)
        self.assertLessEqual(second_time, first_time * 1.5)  # Allow some variance
        
        # Check cache stats
        cache_stats = self.tokenizer.get_cache_stats()
        self.assertGreater(cache_stats['encoding_cache_size'], 0)
    
    def test_decoding_cache(self):
        """Test decoding cache functionality."""
        test_embedding = self.tokenizer.encode(["cache test one"])[0]
        
        # First decoding (no cache)
        start_time = time.time()
        decoded1 = self.tokenizer.decode_embedding(test_embedding)
        first_time = time.time() - start_time
        
        # Second decoding (with cache)
        start_time = time.time()
        decoded2 = self.tokenizer.decode_embedding(test_embedding)
        second_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(decoded1, decoded2)
        
        # Second call should be faster (cached)
        self.assertLessEqual(second_time, first_time * 1.5)  # Allow some variance
        
        # Check cache stats
        cache_stats = self.tokenizer.get_cache_stats()
        self.assertGreater(cache_stats['decoding_cache_size'], 0)
    
    def test_similarity_cache(self):
        """Test similarity cache functionality."""
        test_embedding = self.tokenizer.encode(["cache test one"])[0]
        
        # First similarity search (no cache)
        start_time = time.time()
        closest1 = self.tokenizer.get_closest_texts(test_embedding, top_k=3)
        first_time = time.time() - start_time
        
        # Second similarity search (with cache)
        start_time = time.time()
        closest2 = self.tokenizer.get_closest_texts(test_embedding, top_k=3)
        second_time = time.time() - start_time
        
        # Results should be identical
        self.assertEqual(closest1, closest2)
        
        # Second call should be faster (cached)
        self.assertLessEqual(second_time, first_time * 1.5)  # Allow some variance
        
        # Check cache stats
        cache_stats = self.tokenizer.get_cache_stats()
        self.assertGreater(cache_stats['similarity_cache_size'], 0)
    
    def test_cache_size_management(self):
        """Test cache size management and cleanup."""
        # Set small cache size for testing
        self.tokenizer._cache_size = 3
        
        # Generate more items than cache size
        test_texts_list = [
            ["test1"], ["test2"], ["test3"], ["test4"], ["test5"]
        ]
        
        # Encode multiple times to fill cache beyond limit
        for texts in test_texts_list:
            self.tokenizer.encode(texts)
        
        # Cache should not exceed maximum size
        cache_stats = self.tokenizer.get_cache_stats()
        self.assertLessEqual(cache_stats['encoding_cache_size'], self.tokenizer._cache_size)
    
    def test_cache_clearing(self):
        """Test manual cache clearing."""
        # Generate some cached items
        test_texts = ["clear test one", "clear test two"]
        self.tokenizer.encode(test_texts)
        
        test_embedding = self.tokenizer.encode(["clear test one"])[0]
        self.tokenizer.decode_embedding(test_embedding)
        self.tokenizer.get_closest_texts(test_embedding, top_k=2)
        
        # Verify caches have items
        cache_stats_before = self.tokenizer.get_cache_stats()
        self.assertGreater(cache_stats_before['total_cache_items'], 0)
        
        # Clear caches
        self.tokenizer.clear_caches()
        
        # Verify caches are empty
        cache_stats_after = self.tokenizer.get_cache_stats()
        self.assertEqual(cache_stats_after['total_cache_items'], 0)
    
    def test_cache_configuration(self):
        """Test cache configuration and stats."""
        config = self.tokenizer.get_config()
        
        # Verify cache configuration is included
        self.assertIn('cache_stats', config)
        cache_stats = config['cache_stats']
        
        self.assertIn('encoding_cache_size', cache_stats)
        self.assertIn('decoding_cache_size', cache_stats)
        self.assertIn('similarity_cache_size', cache_stats)
        self.assertIn('max_cache_size', cache_stats)
        
        # Verify cache size setting
        self.assertEqual(cache_stats['max_cache_size'], self.tokenizer._cache_size)


class TestDialogueTokenizerErrorHandling(unittest.TestCase):
    """Test DialogueTokenizer error handling and edge cases."""
    
    def setUp(self):
        """Set up test environment."""
        self.tokenizer = DialogueTokenizer(max_features=100, embedding_dim=32)
    
    def test_unfitted_tokenizer_errors(self):
        """Test errors when using unfitted tokenizer."""
        # Test encode on unfitted tokenizer
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.encode(["test"])
        
        # Test decode on unfitted tokenizer
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.decode_embedding(np.random.random(32))
        
        # Test get_closest_texts on unfitted tokenizer
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.get_closest_texts(np.random.random(32))
        
        # Test save on unfitted tokenizer
        with self.assertRaises(TokenizerNotFittedError):
            self.tokenizer.save("/tmp/test")
    
    def test_invalid_input_validation(self):
        """Test input validation errors."""
        # Fit tokenizer first
        self.tokenizer.fit(["hello world", "good morning"])
        
        # Test encode with invalid inputs
        with self.assertRaises(InvalidInputError):
            self.tokenizer.encode([])  # Empty list
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.encode([""])  # Empty string
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.encode("not a list")  # Not a list
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.encode([123])  # Not strings
        
        # Test decode with invalid embedding dimensions
        with self.assertRaises(InvalidInputError):
            self.tokenizer.decode_embedding(np.random.random(16))  # Wrong dimension
        
        with self.assertRaises(InvalidInputError):
            self.tokenizer.decode_embedding("not an array")  # Not an array
    
    def test_empty_vocabulary_handling(self):
        """Test handling of empty vocabulary scenarios."""
        # Create tokenizer with empty vocabulary
        empty_tokenizer = DialogueTokenizer(max_features=100, embedding_dim=32)
        empty_tokenizer.is_fitted = True  # Simulate fitted but empty
        empty_tokenizer._vocabulary_texts = []
        empty_tokenizer._vocabulary_embeddings = None
        
        # Test decoding with empty vocabulary
        test_embedding = np.random.random(32).astype(np.float32)
        decoded = empty_tokenizer.decode_embedding(test_embedding)
        self.assertEqual(decoded, "[UNKNOWN]")
        
        # Test get_closest_texts with empty vocabulary
        closest = empty_tokenizer.get_closest_texts(test_embedding)
        self.assertEqual(closest, [("[UNKNOWN]", 0.0)])
    
    def test_corrupted_data_handling(self):
        """Test handling of corrupted or invalid data."""
        # Test fitting with invalid texts
        with self.assertRaises(DataValidationError):
            self.tokenizer.fit([])  # Empty list
        
        # Test fitting with all empty strings
        with self.assertRaises(DataValidationError):
            self.tokenizer.fit(["", "   ", "\n"])  # All empty/whitespace
    
    def test_memory_error_simulation(self):
        """Test handling of memory-related errors."""
        # This test simulates memory errors during operations
        self.tokenizer.fit(["test text for memory simulation"])
        
        # Test with very large embedding (should handle gracefully)
        try:
            large_embedding = np.random.random(self.tokenizer.embedding_dim * 1000).astype(np.float32)
            # This should either work or raise a clear error
            result = self.tokenizer.decode_embedding(large_embedding[:self.tokenizer.embedding_dim])
            self.assertIsInstance(result, str)
        except Exception as e:
            # Should be a clear, informative error
            self.assertIsInstance(e, (InvalidInputError, TokenizerError))


def run_tokenizer_tests():
    """Run all tokenizer tests."""
    print("Running DialogueTokenizer Tests")
    print("=" * 50)
    
    test_suites = [
        ('Core Functionality', TestDialogueTokenizerCore),
        ('Persistence', TestDialogueTokenizerPersistence),
        ('Caching', TestDialogueTokenizerCaching),
        ('Error Handling', TestDialogueTokenizerErrorHandling),
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for suite_name, test_class in test_suites:
        print(f"\n{suite_name}")
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
            print(f"  FAIL: {failures + errors}/{tests_run} tests failed")
            
            # Print failure details
            for test, traceback in result.failures + result.errors:
                print(f"    FAILED: {test}")
                # Extract just the assertion error message
                if 'AssertionError:' in traceback:
                    error_msg = traceback.split('AssertionError:')[-1].strip()
                    print(f"    {error_msg}")
        else:
            print(f"  PASS: All {tests_run} tests passed")
    
    # Summary
    print(f"\nTokenizer Test Summary")
    print("=" * 50)
    print(f"Total tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    
    if failed_tests:
        print(f"Failed test suites: {', '.join(failed_tests)}")
        return False
    else:
        print("All DialogueTokenizer tests passed!")
        return True


if __name__ == "__main__":
    success = run_tokenizer_tests()
    exit(0 if success else 1)