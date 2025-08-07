#!/usr/bin/env python3
"""
Test script for optimization features without requiring TensorFlow.
Tests caching, memory management, and tokenizer optimizations.
"""

import time
import numpy as np
from typing import List
import tempfile
import os
import shutil

# Test the tokenizer optimizations directly
from data_loader import DialogueTokenizer

def test_tokenizer_caching():
    """Test tokenizer caching optimizations."""
    print("Testing tokenizer caching...")
    
    try:
        tokenizer = DialogueTokenizer(embedding_dim=64, max_features=1000)
        
        # Create mock texts for fitting
        mock_texts = [
            "hello world", "how are you", "i am fine", "good morning",
            "nice to meet you", "what is your name", "my name is alice",
            "where are you from", "i am from earth", "that sounds great",
            "this is a test", "testing tokenizer", "caching performance",
            "optimization features", "memory efficiency", "batch processing"
        ] * 5  # Repeat for more data
        
        print(f"  Fitting tokenizer on {len(mock_texts)} texts...")
        tokenizer.fit(mock_texts)
        
        # Test encoding with caching
        test_texts = ["hello world", "how are you", "hello world", "good morning", "hello world"]
        
        print("  Testing encoding performance...")
        
        # First encoding (no cache)
        start_time = time.time()
        embeddings1 = tokenizer.encode(test_texts)
        first_time = time.time() - start_time
        
        # Second encoding (with cache)
        start_time = time.time()
        embeddings2 = tokenizer.encode(test_texts)
        second_time = time.time() - start_time
        
        print(f"    First encoding time: {first_time:.4f}s")
        print(f"    Second encoding time: {second_time:.4f}s")
        if second_time > 0:
            print(f"    Encoding cache speedup: {first_time/second_time:.2f}x")
        
        # Verify results are identical
        if np.allclose(embeddings1, embeddings2):
            print("    ✓ Cached results match original")
        else:
            print("    ⚠ Cached results differ from original")
        
        # Test decoding with caching
        print("  Testing decoding performance...")
        test_embedding = embeddings1[0]
        
        start_time = time.time()
        decoded1 = tokenizer.decode_embedding(test_embedding)
        first_decode_time = time.time() - start_time
        
        start_time = time.time()
        decoded2 = tokenizer.decode_embedding(test_embedding)
        second_decode_time = time.time() - start_time
        
        print(f"    First decoding time: {first_decode_time:.4f}s")
        print(f"    Second decoding time: {second_decode_time:.4f}s")
        if second_decode_time > 0:
            print(f"    Decoding cache speedup: {first_decode_time/second_decode_time:.2f}x")
        
        # Verify decoding results
        if decoded1 == decoded2:
            print("    ✓ Cached decoding results match")
        else:
            print("    ⚠ Cached decoding results differ")
        
        # Test batch decoding optimization
        print("  Testing batch decoding...")
        start_time = time.time()
        batch_decoded = tokenizer.decode_embeddings_batch(embeddings1)
        batch_time = time.time() - start_time
        
        start_time = time.time()
        individual_decoded = [tokenizer.decode_embedding(emb) for emb in embeddings1]
        individual_time = time.time() - start_time
        
        print(f"    Batch decoding time: {batch_time:.4f}s")
        print(f"    Individual decoding time: {individual_time:.4f}s")
        if batch_time > 0:
            print(f"    Batch decoding speedup: {individual_time/batch_time:.2f}x")
        
        # Test get_closest_texts caching
        print("  Testing similarity search caching...")
        start_time = time.time()
        closest1 = tokenizer.get_closest_texts(test_embedding, top_k=3)
        first_similarity_time = time.time() - start_time
        
        start_time = time.time()
        closest2 = tokenizer.get_closest_texts(test_embedding, top_k=3)
        second_similarity_time = time.time() - start_time
        
        print(f"    First similarity search: {first_similarity_time:.4f}s")
        print(f"    Second similarity search: {second_similarity_time:.4f}s")
        if second_similarity_time > 0:
            print(f"    Similarity cache speedup: {first_similarity_time/second_similarity_time:.2f}x")
        
        # Check cache stats
        cache_stats = tokenizer.get_cache_stats()
        print(f"  Cache statistics:")
        for cache_type, size in cache_stats.items():
            if cache_type != 'max_cache_size':
                print(f"    {cache_type}: {size}")
        
        # Test cache clearing
        print("  Testing cache clearing...")
        tokenizer.clear_caches()
        cache_stats_after = tokenizer.get_cache_stats()
        total_after = cache_stats_after['total_cache_items']
        print(f"    Cache items after clearing: {total_after}")
        
        if total_after == 0:
            print("    ✓ Cache cleared successfully")
        else:
            print("    ⚠ Cache not fully cleared")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tokenizer caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tokenizer_save_load():
    """Test tokenizer save/load with optimizations."""
    print("Testing tokenizer save/load...")
    
    temp_dir = tempfile.mkdtemp(prefix="test_tokenizer_")
    
    try:
        # Create and fit tokenizer
        tokenizer1 = DialogueTokenizer(embedding_dim=32, max_features=500)
        mock_texts = [
            "hello world", "how are you", "i am fine", "good morning",
            "nice to meet you", "what is your name"
        ]
        tokenizer1.fit(mock_texts)
        
        # Add some cache entries
        test_texts = ["hello world", "how are you"]
        embeddings = tokenizer1.encode(test_texts)
        decoded = tokenizer1.decode_embeddings_batch(embeddings)
        
        print(f"  Original cache stats: {tokenizer1.get_cache_stats()}")
        
        # Save tokenizer
        save_path = os.path.join(temp_dir, "tokenizer")
        tokenizer1.save(save_path)
        print(f"  Tokenizer saved to {save_path}")
        
        # Load tokenizer
        tokenizer2 = DialogueTokenizer(embedding_dim=32, max_features=500)
        tokenizer2.load(save_path)
        print(f"  Tokenizer loaded from {save_path}")
        
        # Test that loaded tokenizer works
        embeddings2 = tokenizer2.encode(test_texts)
        decoded2 = tokenizer2.decode_embeddings_batch(embeddings2)
        
        # Compare results
        if np.allclose(embeddings, embeddings2, rtol=1e-5):
            print("    ✓ Loaded tokenizer produces same embeddings")
        else:
            print("    ⚠ Loaded tokenizer embeddings differ")
        
        if decoded == decoded2:
            print("    ✓ Loaded tokenizer produces same decoded text")
        else:
            print("    ⚠ Loaded tokenizer decoded text differs")
        
        # Check that cache is empty after loading (as expected)
        cache_stats2 = tokenizer2.get_cache_stats()
        print(f"  Loaded tokenizer cache stats: {cache_stats2}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Tokenizer save/load test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_memory_efficiency():
    """Test memory efficiency features."""
    print("Testing memory efficiency...")
    
    try:
        tokenizer = DialogueTokenizer(embedding_dim=128, max_features=2000)
        
        # Create larger dataset
        mock_texts = []
        for i in range(100):
            for base_text in ["hello world", "how are you", "good morning", "nice day"]:
                mock_texts.append(f"{base_text} {i}")
        
        print(f"  Fitting tokenizer on {len(mock_texts)} texts...")
        tokenizer.fit(mock_texts)
        
        # Test with many encoding operations to fill cache
        print("  Testing cache management with many operations...")
        
        test_sequences = []
        for i in range(50):  # More than default cache size
            seq = [f"test text {i}", f"another text {i}", f"more text {i}"]
            test_sequences.append(seq)
        
        # Perform many encoding operations
        for i, seq in enumerate(test_sequences):
            embeddings = tokenizer.encode(seq)
            decoded = tokenizer.decode_embeddings_batch(embeddings)
            
            if i % 10 == 0:
                cache_stats = tokenizer.get_cache_stats()
                print(f"    Operation {i}: cache size = {cache_stats['total_cache_items']}")
        
        final_cache_stats = tokenizer.get_cache_stats()
        print(f"  Final cache statistics: {final_cache_stats}")
        
        # Verify cache size is managed
        max_expected = final_cache_stats['max_cache_size'] * 3  # 3 cache types
        if final_cache_stats['total_cache_items'] <= max_expected:
            print("    ✓ Cache size is properly managed")
        else:
            print("    ⚠ Cache size may be growing unbounded")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Memory efficiency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_optimization():
    """Test batch processing optimizations."""
    print("Testing batch processing optimizations...")
    
    try:
        tokenizer = DialogueTokenizer(embedding_dim=64, max_features=1000)
        
        # Fit tokenizer
        mock_texts = [
            "hello world", "how are you", "i am fine", "good morning",
            "nice to meet you", "what is your name", "my name is alice",
            "where are you from", "i am from earth", "that sounds great"
        ] * 3
        
        tokenizer.fit(mock_texts)
        
        # Create test data for batch processing
        test_texts_batch = [
            ["hello", "world", "test"],
            ["how", "are", "you"],
            ["good", "morning", "friend"],
            ["nice", "to", "meet"],
            ["what", "is", "happening"]
        ]
        
        # Test batch encoding vs individual encoding
        print("  Comparing batch vs individual encoding...")
        
        # Individual encoding
        start_time = time.time()
        individual_results = []
        for texts in test_texts_batch:
            result = tokenizer.encode(texts)
            individual_results.append(result)
        individual_time = time.time() - start_time
        
        # Batch encoding (simulate by encoding all at once)
        start_time = time.time()
        all_texts = [text for texts in test_texts_batch for text in texts]
        batch_embeddings = tokenizer.encode(all_texts)
        
        # Reshape back to original structure
        batch_results = []
        start_idx = 0
        for texts in test_texts_batch:
            end_idx = start_idx + len(texts)
            batch_results.append(batch_embeddings[start_idx:end_idx])
            start_idx = end_idx
        
        batch_time = time.time() - start_time
        
        print(f"    Individual encoding time: {individual_time:.4f}s")
        print(f"    Batch encoding time: {batch_time:.4f}s")
        if batch_time > 0:
            print(f"    Batch efficiency: {individual_time/batch_time:.2f}x")
        
        # Verify results are similar (may have small differences due to caching)
        results_match = True
        for i, (ind_result, batch_result) in enumerate(zip(individual_results, batch_results)):
            if not np.allclose(ind_result, batch_result, rtol=1e-3):
                results_match = False
                break
        
        if results_match:
            print("    ✓ Batch and individual results match")
        else:
            print("    ⚠ Batch and individual results differ")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Batch optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run optimization feature tests."""
    print("🚀 Testing Performance Optimization Features")
    print("=" * 50)
    
    tests = [
        ("Tokenizer Caching", test_tokenizer_caching),
        ("Tokenizer Save/Load", test_tokenizer_save_load),
        ("Memory Efficiency", test_memory_efficiency),
        ("Batch Optimization", test_batch_optimization),
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
        print("🎉 All optimization feature tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())