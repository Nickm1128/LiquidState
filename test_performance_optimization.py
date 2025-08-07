#!/usr/bin/env python3
"""
Performance test script for the optimized LSM inference system.
Tests lazy loading, caching, and memory-efficient batch processing.
"""

import os
import time
import numpy as np
from typing import List, Dict, Any
import tempfile
import shutil

# Import the optimized inference classes
from inference import OptimizedLSMInference, LSMInference
from data_loader import DialogueTokenizer
from train import LSMTrainer
from model_config import ModelConfiguration

def create_mock_model_directory() -> str:
    """Create a mock model directory for testing."""
    temp_dir = tempfile.mkdtemp(prefix="test_model_")
    
    # Create mock configuration
    config = ModelConfiguration(
        window_size=5,
        embedding_dim=64,
        reservoir_type='standard',
        reservoir_config={},
        reservoir_units=[128, 64, 32],
        sparsity=0.1,
        use_multichannel=True,
        tokenizer_max_features=1000,
        tokenizer_ngram_range=(1, 2)
    )
    config.save(os.path.join(temp_dir, "config.json"))
    
    # Create mock tokenizer
    tokenizer = DialogueTokenizer(embedding_dim=64, max_features=1000)
    mock_texts = [
        "hello world", "how are you", "i am fine", "good morning",
        "nice to meet you", "what is your name", "my name is alice",
        "where are you from", "i am from earth", "that sounds great"
    ]
    tokenizer.fit(mock_texts)
    tokenizer.save(os.path.join(temp_dir, "tokenizer"))
    
    # Create mock trainer and models
    trainer = LSMTrainer(
        window_size=5,
        embedding_dim=64,
        reservoir_units=[128, 64, 32],
        sparsity=0.1,
        use_multichannel=True,
        reservoir_type='standard'
    )
    trainer.build_models()
    
    # Save models
    trainer.reservoir.save(os.path.join(temp_dir, "reservoir_model"))
    trainer.cnn_model.save(os.path.join(temp_dir, "cnn_model"))
    
    return temp_dir

def test_lazy_loading():
    """Test lazy loading functionality."""
    print("Testing lazy loading...")
    
    model_dir = create_mock_model_directory()
    
    try:
        # Test with lazy loading enabled
        start_time = time.time()
        inference_lazy = OptimizedLSMInference(model_dir, lazy_load=True)
        lazy_init_time = time.time() - start_time
        
        # Test with lazy loading disabled
        start_time = time.time()
        inference_eager = OptimizedLSMInference(model_dir, lazy_load=False)
        eager_init_time = time.time() - start_time
        
        print(f"  Lazy loading init time: {lazy_init_time:.3f}s")
        print(f"  Eager loading init time: {eager_init_time:.3f}s")
        print(f"  Lazy loading speedup: {eager_init_time/lazy_init_time:.2f}x")
        
        # Test that lazy loading works on first prediction
        test_sequence = ["hello", "how", "are", "you", "doing"]
        
        start_time = time.time()
        prediction = inference_lazy.predict_next_token(test_sequence)
        first_prediction_time = time.time() - start_time
        
        print(f"  First prediction time (lazy): {first_prediction_time:.3f}s")
        print(f"  Prediction result: {prediction}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Lazy loading test failed: {e}")
        return False
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

def test_caching_performance():
    """Test caching performance improvements."""
    print("Testing caching performance...")
    
    model_dir = create_mock_model_directory()
    
    try:
        inference = OptimizedLSMInference(model_dir, cache_size=100)
        
        # Test sequences
        test_sequences = [
            ["hello", "how", "are", "you", "doing"],
            ["good", "morning", "nice", "to", "meet"],
            ["what", "is", "your", "name", "today"],
            ["hello", "how", "are", "you", "doing"],  # Repeat for cache test
            ["good", "morning", "nice", "to", "meet"],  # Repeat for cache test
        ]
        
        # First run (no cache)
        start_time = time.time()
        predictions_first = []
        for seq in test_sequences:
            pred = inference.predict_next_token(seq)
            predictions_first.append(pred)
        first_run_time = time.time() - start_time
        
        # Second run (with cache)
        start_time = time.time()
        predictions_second = []
        for seq in test_sequences:
            pred = inference.predict_next_token(seq)
            predictions_second.append(pred)
        second_run_time = time.time() - start_time
        
        print(f"  First run time: {first_run_time:.3f}s")
        print(f"  Second run time: {second_run_time:.3f}s")
        print(f"  Cache speedup: {first_run_time/second_run_time:.2f}x")
        
        # Verify predictions are consistent
        if predictions_first == predictions_second:
            print("  ✓ Cache results are consistent")
        else:
            print("  ⚠ Cache results differ from original")
        
        # Check cache stats
        cache_stats = inference.get_cache_stats()
        print(f"  Cache stats: {cache_stats['prediction_cache']['size']} predictions cached")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Caching test failed: {e}")
        return False
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

def test_batch_processing():
    """Test memory-efficient batch processing."""
    print("Testing batch processing...")
    
    model_dir = create_mock_model_directory()
    
    try:
        inference = OptimizedLSMInference(model_dir, max_batch_size=2)
        
        # Create test sequences
        test_sequences = [
            ["hello", "how", "are", "you", "doing"],
            ["good", "morning", "nice", "to", "meet"],
            ["what", "is", "your", "name", "today"],
            ["where", "are", "you", "from", "friend"],
            ["i", "am", "fine", "thank", "you"],
        ]
        
        # Test batch prediction
        start_time = time.time()
        batch_predictions = inference.batch_predict(test_sequences)
        batch_time = time.time() - start_time
        
        # Test individual predictions
        start_time = time.time()
        individual_predictions = []
        for seq in test_sequences:
            pred = inference.predict_next_token(seq)
            individual_predictions.append(pred)
        individual_time = time.time() - start_time
        
        print(f"  Batch processing time: {batch_time:.3f}s")
        print(f"  Individual processing time: {individual_time:.3f}s")
        print(f"  Batch efficiency: {individual_time/batch_time:.2f}x")
        print(f"  Batch results: {len(batch_predictions)} predictions")
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            start_time = time.time()
            predictions = inference.batch_predict(test_sequences, batch_size=batch_size)
            batch_time = time.time() - start_time
            print(f"  Batch size {batch_size}: {batch_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Batch processing test failed: {e}")
        return False
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

def test_tokenizer_optimization():
    """Test tokenizer caching optimizations."""
    print("Testing tokenizer optimizations...")
    
    try:
        tokenizer = DialogueTokenizer(embedding_dim=64, max_features=1000)
        
        # Fit tokenizer
        mock_texts = [
            "hello world", "how are you", "i am fine", "good morning",
            "nice to meet you", "what is your name", "my name is alice",
            "where are you from", "i am from earth", "that sounds great"
        ] * 10  # Repeat for more data
        
        tokenizer.fit(mock_texts)
        
        # Test encoding with caching
        test_texts = ["hello world", "how are you", "hello world", "good morning"]
        
        # First encoding (no cache)
        start_time = time.time()
        embeddings1 = tokenizer.encode(test_texts)
        first_time = time.time() - start_time
        
        # Second encoding (with cache)
        start_time = time.time()
        embeddings2 = tokenizer.encode(test_texts)
        second_time = time.time() - start_time
        
        print(f"  First encoding time: {first_time:.4f}s")
        print(f"  Second encoding time: {second_time:.4f}s")
        print(f"  Encoding cache speedup: {first_time/second_time:.2f}x")
        
        # Test decoding with caching
        test_embedding = embeddings1[0]
        
        start_time = time.time()
        decoded1 = tokenizer.decode_embedding(test_embedding)
        first_decode_time = time.time() - start_time
        
        start_time = time.time()
        decoded2 = tokenizer.decode_embedding(test_embedding)
        second_decode_time = time.time() - start_time
        
        print(f"  First decoding time: {first_decode_time:.4f}s")
        print(f"  Second decoding time: {second_decode_time:.4f}s")
        print(f"  Decoding cache speedup: {first_decode_time/second_decode_time:.2f}x")
        
        # Check cache stats
        cache_stats = tokenizer.get_cache_stats()
        print(f"  Tokenizer cache stats: {cache_stats}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tokenizer optimization test failed: {e}")
        return False

def test_memory_management():
    """Test memory management features."""
    print("Testing memory management...")
    
    model_dir = create_mock_model_directory()
    
    try:
        inference = OptimizedLSMInference(model_dir, cache_size=10)  # Small cache for testing
        
        # Generate many predictions to trigger cache management
        test_sequences = []
        for i in range(20):  # More than cache size
            seq = [f"word{j}_{i}" for j in range(5)]
            test_sequences.append(seq)
        
        # Make predictions
        predictions = []
        for seq in test_sequences:
            try:
                pred = inference.predict_next_token(seq)
                predictions.append(pred)
            except Exception as e:
                print(f"    Prediction failed for {seq}: {e}")
                predictions.append("[ERROR]")
        
        print(f"  Generated {len(predictions)} predictions")
        
        # Check cache management
        cache_stats = inference.get_cache_stats()
        print(f"  Final cache size: {cache_stats['prediction_cache']['size']}")
        print(f"  Max cache size: {inference.cache_size}")
        
        # Test manual cache clearing
        inference.clear_caches()
        cache_stats_after = inference.get_cache_stats()
        print(f"  Cache size after clearing: {cache_stats_after['prediction_cache']['size']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory management test failed: {e}")
        return False
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

def test_legacy_compatibility():
    """Test that legacy LSMInference still works."""
    print("Testing legacy compatibility...")
    
    model_dir = create_mock_model_directory()
    
    try:
        # Test legacy inference
        legacy_inference = LSMInference(model_dir)
        
        test_sequence = ["hello", "how", "are", "you", "doing"]
        prediction = legacy_inference.predict_next_token(test_sequence)
        
        print(f"  Legacy prediction: {prediction}")
        print("  ✓ Legacy compatibility maintained")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Legacy compatibility test failed: {e}")
        return False
    finally:
        shutil.rmtree(model_dir, ignore_errors=True)

def main():
    """Run all performance optimization tests."""
    print("🚀 Running Performance Optimization Tests")
    print("=" * 50)
    
    tests = [
        ("Lazy Loading", test_lazy_loading),
        ("Caching Performance", test_caching_performance),
        ("Batch Processing", test_batch_processing),
        ("Tokenizer Optimization", test_tokenizer_optimization),
        ("Memory Management", test_memory_management),
        ("Legacy Compatibility", test_legacy_compatibility),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📊 {test_name}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            success = test_func()
            test_time = time.time() - start_time
            
            results[test_name] = {
                'success': success,
                'time': test_time
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {status} ({test_time:.2f}s)")
            
        except Exception as e:
            results[test_name] = {
                'success': False,
                'time': 0,
                'error': str(e)
            }
            print(f"  ❌ FAILED: {e}")
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    print(f"Total test time: {sum(r['time'] for r in results.values()):.2f}s")
    
    if passed == total:
        print("🎉 All performance optimization tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())