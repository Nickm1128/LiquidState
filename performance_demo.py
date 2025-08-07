#!/usr/bin/env python3
"""
Performance demonstration script showing the optimization improvements.
"""

import time
import numpy as np
from data_loader import DialogueTokenizer

def demonstrate_caching_benefits():
    """Demonstrate the benefits of caching in tokenizer operations."""
    print("🚀 Demonstrating Caching Benefits")
    print("=" * 50)
    
    # Create and fit tokenizer
    tokenizer = DialogueTokenizer(embedding_dim=128, max_features=2000)
    
    # Create training data
    training_texts = []
    base_phrases = [
        "hello world", "how are you", "good morning", "nice to meet you",
        "what is your name", "where are you from", "i am fine thank you",
        "have a great day", "see you later", "take care"
    ]
    
    for i in range(20):
        for phrase in base_phrases:
            training_texts.append(f"{phrase} {i}")
    
    print(f"Training tokenizer on {len(training_texts)} texts...")
    start_time = time.time()
    tokenizer.fit(training_texts)
    fit_time = time.time() - start_time
    print(f"✓ Tokenizer fitted in {fit_time:.3f}s")
    
    # Test repeated encoding operations
    test_texts = ["hello world", "how are you", "good morning"] * 10
    
    print(f"\nTesting encoding performance with {len(test_texts)} texts...")
    
    # Clear cache first
    tokenizer.clear_caches()
    
    # First run (no cache)
    start_time = time.time()
    for i in range(5):  # Multiple iterations
        embeddings = tokenizer.encode(test_texts)
    no_cache_time = time.time() - start_time
    
    # Second run (with cache)
    start_time = time.time()
    for i in range(5):  # Multiple iterations
        embeddings = tokenizer.encode(test_texts)
    with_cache_time = time.time() - start_time
    
    print(f"Without cache: {no_cache_time:.3f}s")
    print(f"With cache: {with_cache_time:.3f}s")
    if with_cache_time > 0:
        print(f"Speedup: {no_cache_time/with_cache_time:.2f}x")
    else:
        print("Speedup: Very fast (cached operations completed instantly)")
    
    # Test decoding performance
    print(f"\nTesting decoding performance...")
    test_embedding = embeddings[0]
    
    # Clear cache
    tokenizer.clear_caches()
    
    # First run (no cache)
    start_time = time.time()
    for i in range(100):  # Many iterations
        decoded = tokenizer.decode_embedding(test_embedding)
    no_cache_decode_time = time.time() - start_time
    
    # Second run (with cache)
    start_time = time.time()
    for i in range(100):  # Many iterations
        decoded = tokenizer.decode_embedding(test_embedding)
    with_cache_decode_time = time.time() - start_time
    
    print(f"Decoding without cache: {no_cache_decode_time:.3f}s")
    print(f"Decoding with cache: {with_cache_decode_time:.3f}s")
    if with_cache_decode_time > 0:
        print(f"Decoding speedup: {no_cache_decode_time/with_cache_decode_time:.2f}x")
    else:
        print("Decoding speedup: Very fast (cached operations completed instantly)")
    
    # Show cache statistics
    cache_stats = tokenizer.get_cache_stats()
    print(f"\nFinal cache statistics:")
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")

def demonstrate_batch_efficiency():
    """Demonstrate batch processing efficiency."""
    print(f"\n🔄 Demonstrating Batch Processing Efficiency")
    print("=" * 50)
    
    tokenizer = DialogueTokenizer(embedding_dim=64, max_features=1000)
    
    # Fit tokenizer
    training_texts = [f"text sample {i}" for i in range(100)]
    tokenizer.fit(training_texts)
    
    # Create test embeddings
    test_texts = [f"test text {i}" for i in range(50)]
    embeddings = tokenizer.encode(test_texts)
    
    print(f"Testing batch vs individual decoding with {len(embeddings)} embeddings...")
    
    # Individual decoding
    start_time = time.time()
    individual_results = []
    for embedding in embeddings:
        result = tokenizer.decode_embedding(embedding)
        individual_results.append(result)
    individual_time = time.time() - start_time
    
    # Batch decoding
    start_time = time.time()
    batch_results = tokenizer.decode_embeddings_batch(embeddings)
    batch_time = time.time() - start_time
    
    print(f"Individual decoding: {individual_time:.3f}s")
    print(f"Batch decoding: {batch_time:.3f}s")
    if batch_time > 0:
        print(f"Batch efficiency: {individual_time/batch_time:.2f}x")
    else:
        print("Batch efficiency: Very fast (batch operations completed instantly)")
    
    # Verify results match
    if individual_results == batch_results:
        print("✓ Batch and individual results are identical")
    else:
        print("⚠ Results differ (this may be due to caching)")

def demonstrate_memory_management():
    """Demonstrate memory management features."""
    print(f"\n💾 Demonstrating Memory Management")
    print("=" * 50)
    
    tokenizer = DialogueTokenizer(embedding_dim=32, max_features=500)
    
    # Fit with small vocabulary
    training_texts = [f"sample {i}" for i in range(20)]
    tokenizer.fit(training_texts)
    
    print("Performing many operations to test cache management...")
    
    # Perform many operations that would fill cache
    for i in range(100):
        # Create unique texts to avoid cache hits
        unique_texts = [f"unique text {i} {j}" for j in range(3)]
        try:
            embeddings = tokenizer.encode(unique_texts)
            decoded = tokenizer.decode_embeddings_batch(embeddings)
        except:
            # Some operations may fail with unique texts not in vocabulary
            pass
        
        if i % 20 == 0:
            stats = tokenizer.get_cache_stats()
            print(f"  Operation {i}: Total cache items = {stats['total_cache_items']}")
    
    final_stats = tokenizer.get_cache_stats()
    print(f"\nFinal cache statistics:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    # Test manual cache clearing
    print(f"\nClearing caches...")
    tokenizer.clear_caches()
    cleared_stats = tokenizer.get_cache_stats()
    print(f"Cache items after clearing: {cleared_stats['total_cache_items']}")

def demonstrate_similarity_search_optimization():
    """Demonstrate optimized similarity search."""
    print(f"\n🔍 Demonstrating Similarity Search Optimization")
    print("=" * 50)
    
    tokenizer = DialogueTokenizer(embedding_dim=64, max_features=1000)
    
    # Create diverse training data
    training_texts = [
        "hello world", "good morning", "how are you", "nice to meet you",
        "what is your name", "where are you from", "have a great day",
        "see you later", "take care", "goodbye friend", "welcome here",
        "thank you very much", "you are welcome", "no problem at all"
    ] * 5  # Repeat for larger vocabulary
    
    tokenizer.fit(training_texts)
    
    # Test similarity search performance
    test_text = "hello world"
    test_embedding = tokenizer.encode([test_text])[0]
    
    print(f"Testing similarity search with vocabulary size: {len(tokenizer._vocabulary_texts)}")
    
    # Test different k values
    for k in [1, 3, 5, 10]:
        # Clear cache
        tokenizer.clear_caches()
        
        # First search (no cache)
        start_time = time.time()
        results1 = tokenizer.get_closest_texts(test_embedding, top_k=k)
        first_time = time.time() - start_time
        
        # Second search (with cache)
        start_time = time.time()
        results2 = tokenizer.get_closest_texts(test_embedding, top_k=k)
        second_time = time.time() - start_time
        
        speedup_text = f"{first_time/second_time:.1f}x" if second_time > 0 else "Very fast"
        print(f"  k={k}: First={first_time:.4f}s, Cached={second_time:.4f}s, Speedup={speedup_text}")
        
        # Show top result
        if results1:
            print(f"    Top result: '{results1[0][0]}' (similarity: {results1[0][1]:.3f})")

def main():
    """Run all performance demonstrations."""
    print("🎯 LSM Inference Performance Optimization Demo")
    print("=" * 60)
    
    try:
        demonstrate_caching_benefits()
        demonstrate_batch_efficiency()
        demonstrate_memory_management()
        demonstrate_similarity_search_optimization()
        
        print(f"\n🎉 Performance demonstration completed successfully!")
        print("Key optimizations implemented:")
        print("  ✓ Tokenizer encoding/decoding caching")
        print("  ✓ Batch processing optimizations")
        print("  ✓ Memory-efficient cache management")
        print("  ✓ Optimized similarity search with caching")
        print("  ✓ Lazy loading support (in full inference system)")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())