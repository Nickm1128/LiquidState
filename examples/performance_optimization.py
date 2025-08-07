#!/usr/bin/env python3
"""
Performance Optimization Examples
==================================

This script demonstrates various performance optimization techniques for the LSM inference system.
Shows caching strategies, memory management, and batch processing optimizations.
"""

import sys
import os
import time
import gc
from typing import List, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import OptimizedLSMInference
from lsm_exceptions import ModelLoadError
from src.lsm.management.model_manager import ModelManager

# Optional psutil for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def find_example_model():
    """Find an available model for demonstration."""
    manager = ModelManager()
    models = manager.list_available_models()
    
    if not models:
        print("❌ No trained models found!")
        print("Please train a model first using: python main.py train")
        return None
    
    model = models[0]
    print(f"✅ Using model: {model['path']}")
    print()
    return model['path']

def get_memory_usage():
    """Get current memory usage in MB."""
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return None

def cache_optimization_example():
    """Demonstrate cache optimization strategies."""
    print("🚀 Cache Optimization Example")
    print("=" * 50)
    
    model_path = find_example_model()
    if not model_path:
        return
    
    # Test different cache sizes
    cache_sizes = [100, 500, 1000, 2000]
    test_dialogues = [
        ["Hello", "How are you?"],
        ["Good morning", "Nice weather"],
        ["What's your name?", "I'm Alice"],
        ["How was your day?", "It was great"],
        ["I love programming", "Me too"]
    ] * 20  # 100 total dialogues with repetition
    
    print("Testing different cache sizes with repeated dialogues...")
    print()
    
    results = []
    
    for cache_size in cache_sizes:
        print(f"Testing cache size: {cache_size}")
        
        try:
            # Initialize with specific cache size
            inference = OptimizedLSMInference(
                model_path=model_path,
                cache_size=cache_size,
                lazy_load=True
            )
            
            # Warm up
            inference.predict_next_token(["Hello"])
            
            # Time the predictions
            start_time = time.time()
            for dialogue in test_dialogues:
                inference.predict_next_token(dialogue)
            end_time = time.time()
            
            # Get cache statistics
            stats = inference.get_cache_stats()
            
            result = {
                'cache_size': cache_size,
                'total_time': end_time - start_time,
                'hit_rate': stats.get('hit_rate', 0),
                'avg_time_per_prediction': (end_time - start_time) / len(test_dialogues) * 1000
            }
            results.append(result)
            
            print(f"  Time: {result['total_time']:.3f}s")
            print(f"  Hit rate: {result['hit_rate']:.2%}")
            print(f"  Avg per prediction: {result['avg_time_per_prediction']:.1f}ms")
            print()
            
        except Exception as e:
            print(f"  ❌ Error with cache size {cache_size}: {e}")
            print()
    
    # Show comparison
    if results:
        print("Cache Size Comparison:")
        print("-" * 70)
        print(f"{'Cache Size':<12} {'Total Time':<12} {'Hit Rate':<12} {'Avg Time (ms)':<15}")
        print("-" * 70)
        
        for result in results:
            print(f"{result['cache_size']:<12} "
                  f"{result['total_time']:<12.3f} "
                  f"{result['hit_rate']:<12.2%} "
                  f"{result['avg_time_per_prediction']:<15.1f}")
        
        # Find optimal cache size
        best_result = min(results, key=lambda x: x['total_time'])
        print()
        print(f"🏆 Best performance: Cache size {best_result['cache_size']} "
              f"({best_result['total_time']:.3f}s total, {best_result['hit_rate']:.1%} hit rate)")
    
    print()

def memory_management_example():
    """Demonstrate memory management techniques."""
    print("💾 Memory Management Example")
    print("=" * 50)
    
    model_path = find_example_model()
    if not model_path:
        return
    
    if not PSUTIL_AVAILABLE:
        print("⚠️  psutil not available - memory monitoring limited")
        print("Install with: pip install psutil")
        print()
    
    try:
        # Initialize inference
        inference = OptimizedLSMInference(
            model_path=model_path,
            cache_size=1000,
            lazy_load=True
        )
        
        # Create a large dataset to stress memory
        base_dialogues = [
            ["Hello", "How are you?"],
            ["Good morning", "Nice weather"],
            ["What's your name?", "I'm Alice"],
            ["How was your day?", "It was great"],
            ["I love programming", "Me too"]
        ]
        large_dataset = base_dialogues * 200  # 1000 dialogues
        
        print(f"Processing {len(large_dataset)} dialogues with memory monitoring...")
        print()
        
        initial_memory = get_memory_usage()
        if initial_memory:
            print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Process without memory management
        print("1. Processing without memory management:")
        start_time = time.time()
        
        for i, dialogue in enumerate(large_dataset):
            inference.predict_next_token(dialogue)
            
            # Monitor memory every 200 predictions
            if i % 200 == 0 and i > 0:
                current_memory = get_memory_usage()
                if current_memory and initial_memory:
                    growth = current_memory - initial_memory
                    print(f"   After {i} predictions: {current_memory:.1f} MB (+{growth:.1f} MB)")
        
        no_management_time = time.time() - start_time
        final_memory_no_mgmt = get_memory_usage()
        
        print(f"   Total time: {no_management_time:.3f}s")
        if final_memory_no_mgmt and initial_memory:
            print(f"   Memory growth: {final_memory_no_mgmt - initial_memory:.1f} MB")
        print()
        
        # Clear caches and reset
        inference.clear_caches()
        gc.collect()
        
        # Process with memory management
        print("2. Processing with memory management:")
        start_time = time.time()
        
        for i, dialogue in enumerate(large_dataset):
            inference.predict_next_token(dialogue)
            
            # Clear caches every 100 predictions
            if i % 100 == 0 and i > 0:
                inference.clear_caches()
                gc.collect()
            
            # Monitor memory every 200 predictions
            if i % 200 == 0 and i > 0:
                current_memory = get_memory_usage()
                if current_memory and initial_memory:
                    growth = current_memory - initial_memory
                    print(f"   After {i} predictions: {current_memory:.1f} MB (+{growth:.1f} MB)")
        
        management_time = time.time() - start_time
        final_memory_mgmt = get_memory_usage()
        
        print(f"   Total time: {management_time:.3f}s")
        if final_memory_mgmt and initial_memory:
            print(f"   Memory growth: {final_memory_mgmt - initial_memory:.1f} MB")
        
        # Compare results
        print()
        print("Memory Management Comparison:")
        print("-" * 40)
        print(f"Without management: {no_management_time:.3f}s")
        print(f"With management:    {management_time:.3f}s")
        
        if final_memory_no_mgmt and final_memory_mgmt and initial_memory:
            no_mgmt_growth = final_memory_no_mgmt - initial_memory
            mgmt_growth = final_memory_mgmt - initial_memory
            print(f"Memory growth (no mgmt): {no_mgmt_growth:.1f} MB")
            print(f"Memory growth (with mgmt): {mgmt_growth:.1f} MB")
            print(f"Memory saved: {no_mgmt_growth - mgmt_growth:.1f} MB")
        
        print()
        
    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        print()

def batch_size_optimization_example():
    """Demonstrate batch size optimization."""
    print("📦 Batch Size Optimization Example")
    print("=" * 50)
    
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        inference = OptimizedLSMInference(
            model_path=model_path,
            cache_size=500,
            max_batch_size=128
        )
        
        # Create test dataset
        dialogues = [
            ["Hello", "How are you?"],
            ["Good morning", "Nice weather"],
            ["What's your name?", "I'm Alice"],
            ["How was your day?", "It was great"]
        ] * 50  # 200 dialogues
        
        batch_sizes = [1, 4, 8, 16, 32, 64]
        results = []
        
        print(f"Testing batch processing with {len(dialogues)} dialogues...")
        print()
        
        for batch_size in batch_sizes:
            try:
                print(f"Testing batch size: {batch_size}")
                
                # Clear caches for fair comparison
                inference.clear_caches()
                
                start_time = time.time()
                predictions = inference.batch_predict(dialogues, batch_size=batch_size)
                end_time = time.time()
                
                total_time = end_time - start_time
                throughput = len(dialogues) / total_time
                
                result = {
                    'batch_size': batch_size,
                    'total_time': total_time,
                    'throughput': throughput,
                    'avg_time_per_item': total_time / len(dialogues) * 1000
                }
                results.append(result)
                
                print(f"  Time: {total_time:.3f}s")
                print(f"  Throughput: {throughput:.1f} dialogues/second")
                print(f"  Avg per item: {result['avg_time_per_item']:.1f}ms")
                print()
                
            except Exception as e:
                print(f"  ❌ Error with batch size {batch_size}: {e}")
                print()
        
        # Show comparison
        if results:
            print("Batch Size Performance Comparison:")
            print("-" * 75)
            print(f"{'Batch Size':<12} {'Total Time':<12} {'Throughput':<15} {'Avg Time (ms)':<15}")
            print("-" * 75)
            
            for result in results:
                print(f"{result['batch_size']:<12} "
                      f"{result['total_time']:<12.3f} "
                      f"{result['throughput']:<15.1f} "
                      f"{result['avg_time_per_item']:<15.1f}")
            
            # Find optimal batch size
            best_result = max(results, key=lambda x: x['throughput'])
            print()
            print(f"🏆 Optimal batch size: {best_result['batch_size']} "
                  f"({best_result['throughput']:.1f} dialogues/second)")
        
        print()
        
    except Exception as e:
        print(f"❌ Batch size optimization failed: {e}")
        print()

def lazy_loading_comparison():
    """Compare lazy loading vs eager loading performance."""
    print("⚡ Lazy Loading vs Eager Loading Comparison")
    print("=" * 50)
    
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        # Test eager loading
        print("1. Testing eager loading (lazy_load=False):")
        start_time = time.time()
        eager_inference = OptimizedLSMInference(
            model_path=model_path,
            lazy_load=False,
            cache_size=500
        )
        eager_load_time = time.time() - start_time
        print(f"   Load time: {eager_load_time:.3f}s")
        
        # Test first prediction with eager loading
        start_time = time.time()
        prediction1 = eager_inference.predict_next_token(["Hello", "World"])
        eager_first_prediction = time.time() - start_time
        print(f"   First prediction: {eager_first_prediction*1000:.1f}ms")
        print()
        
        # Test lazy loading
        print("2. Testing lazy loading (lazy_load=True):")
        start_time = time.time()
        lazy_inference = OptimizedLSMInference(
            model_path=model_path,
            lazy_load=True,
            cache_size=500
        )
        lazy_load_time = time.time() - start_time
        print(f"   Load time: {lazy_load_time:.3f}s")
        
        # Test first prediction with lazy loading
        start_time = time.time()
        prediction2 = lazy_inference.predict_next_token(["Hello", "World"])
        lazy_first_prediction = time.time() - start_time
        print(f"   First prediction: {lazy_first_prediction*1000:.1f}ms")
        print()
        
        # Compare subsequent predictions
        print("3. Comparing subsequent predictions:")
        test_dialogues = [
            ["Good morning"],
            ["How are you?", "Fine"],
            ["What's up?", "Not much", "You?"]
        ]
        
        # Eager loading subsequent predictions
        start_time = time.time()
        for dialogue in test_dialogues:
            eager_inference.predict_next_token(dialogue)
        eager_subsequent = time.time() - start_time
        
        # Lazy loading subsequent predictions
        start_time = time.time()
        for dialogue in test_dialogues:
            lazy_inference.predict_next_token(dialogue)
        lazy_subsequent = time.time() - start_time
        
        print(f"   Eager loading: {eager_subsequent*1000:.1f}ms total")
        print(f"   Lazy loading: {lazy_subsequent*1000:.1f}ms total")
        print()
        
        # Summary
        print("Loading Strategy Comparison:")
        print("-" * 50)
        print(f"{'Strategy':<15} {'Init Time':<12} {'First Pred':<12} {'Subsequent':<12}")
        print("-" * 50)
        print(f"{'Eager':<15} {eager_load_time:<12.3f} {eager_first_prediction*1000:<12.1f} {eager_subsequent*1000:<12.1f}")
        print(f"{'Lazy':<15} {lazy_load_time:<12.3f} {lazy_first_prediction*1000:<12.1f} {lazy_subsequent*1000:<12.1f}")
        print()
        
        print("Recommendations:")
        if lazy_load_time < eager_load_time:
            print("✅ Use lazy loading for faster startup")
        if eager_first_prediction < lazy_first_prediction:
            print("✅ Use eager loading for consistent prediction latency")
        else:
            print("✅ Lazy loading provides good overall performance")
        
        print()
        
    except Exception as e:
        print(f"❌ Loading comparison failed: {e}")
        print()

def concurrent_inference_example():
    """Demonstrate concurrent inference processing."""
    print("🔄 Concurrent Inference Example")
    print("=" * 50)
    
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        inference = OptimizedLSMInference(
            model_path=model_path,
            cache_size=1000,
            lazy_load=True
        )
        
        # Create test data
        test_dialogues = [
            ["Hello", "How are you?"],
            ["Good morning", "Nice weather"],
            ["What's your name?", "I'm Alice"],
            ["How was your day?", "It was great"],
            ["I love programming", "Me too"],
            ["What's the weather?", "It's sunny"],
            ["Can you help me?", "Of course"],
            ["I'm feeling happy", "That's wonderful"]
        ] * 10  # 80 dialogues
        
        print(f"Processing {len(test_dialogues)} dialogues...")
        print()
        
        # Sequential processing
        print("1. Sequential processing:")
        start_time = time.time()
        sequential_results = []
        for dialogue in test_dialogues:
            result = inference.predict_next_token(dialogue)
            sequential_results.append(result)
        sequential_time = time.time() - start_time
        
        print(f"   Time: {sequential_time:.3f}s")
        print(f"   Throughput: {len(test_dialogues)/sequential_time:.1f} dialogues/second")
        print()
        
        # Concurrent processing (note: this may not improve performance due to GIL)
        print("2. Concurrent processing (ThreadPoolExecutor):")
        
        def predict_dialogue(dialogue):
            return inference.predict_next_token(dialogue)
        
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            concurrent_results = list(executor.map(predict_dialogue, test_dialogues))
        concurrent_time = time.time() - start_time
        
        print(f"   Time: {concurrent_time:.3f}s")
        print(f"   Throughput: {len(test_dialogues)/concurrent_time:.1f} dialogues/second")
        print()
        
        # Batch processing comparison
        print("3. Batch processing:")
        start_time = time.time()
        batch_results = inference.batch_predict(test_dialogues, batch_size=16)
        batch_time = time.time() - start_time
        
        print(f"   Time: {batch_time:.3f}s")
        print(f"   Throughput: {len(test_dialogues)/batch_time:.1f} dialogues/second")
        print()
        
        # Compare results
        print("Processing Method Comparison:")
        print("-" * 60)
        print(f"{'Method':<15} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10}")
        print("-" * 60)
        print(f"{'Sequential':<15} {sequential_time:<12.3f} {len(test_dialogues)/sequential_time:<15.1f} {'1.0x':<10}")
        print(f"{'Concurrent':<15} {concurrent_time:<12.3f} {len(test_dialogues)/concurrent_time:<15.1f} {sequential_time/concurrent_time:<10.1f}x")
        print(f"{'Batch':<15} {batch_time:<12.3f} {len(test_dialogues)/batch_time:<15.1f} {sequential_time/batch_time:<10.1f}x")
        print()
        
        # Verify results are consistent
        if len(set([len(sequential_results), len(concurrent_results), len(batch_results)])) == 1:
            print("✅ All methods produced the same number of results")
        else:
            print("⚠️  Different methods produced different numbers of results")
        
        print()
        
    except Exception as e:
        print(f"❌ Concurrent inference test failed: {e}")
        print()

def performance_monitoring_example():
    """Demonstrate comprehensive performance monitoring."""
    print("📊 Performance Monitoring Example")
    print("=" * 50)
    
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        inference = OptimizedLSMInference(
            model_path=model_path,
            cache_size=1000,
            lazy_load=True
        )
        
        # Create monitoring data
        dialogues = [
            ["Hello", "How are you?"],
            ["Good morning", "Nice weather"],
            ["What's your name?", "I'm Alice"]
        ] * 100  # 300 dialogues with repetition for cache testing
        
        print(f"Running performance monitoring with {len(dialogues)} predictions...")
        print()
        
        # Track metrics
        prediction_times = []
        cache_stats_history = []
        memory_history = []
        
        start_time = time.time()
        
        for i, dialogue in enumerate(dialogues):
            # Time individual prediction
            pred_start = time.time()
            prediction = inference.predict_next_token(dialogue)
            pred_time = time.time() - pred_start
            prediction_times.append(pred_time)
            
            # Collect stats every 50 predictions
            if i % 50 == 0:
                stats = inference.get_cache_stats()
                cache_stats_history.append({
                    'iteration': i,
                    'hit_rate': stats.get('hit_rate', 0),
                    'cache_size': stats.get('prediction_cache_size', 0),
                    'total_requests': stats.get('total_requests', 0)
                })
                
                memory = get_memory_usage()
                if memory:
                    memory_history.append({'iteration': i, 'memory_mb': memory})
        
        total_time = time.time() - start_time
        
        # Analyze results
        print("Performance Analysis:")
        print("-" * 40)
        print(f"Total predictions: {len(prediction_times)}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average time per prediction: {sum(prediction_times)/len(prediction_times)*1000:.1f}ms")
        print(f"Fastest prediction: {min(prediction_times)*1000:.1f}ms")
        print(f"Slowest prediction: {max(prediction_times)*1000:.1f}ms")
        print(f"Throughput: {len(prediction_times)/total_time:.1f} predictions/second")
        print()
        
        # Cache performance over time
        if cache_stats_history:
            print("Cache Performance Over Time:")
            print("-" * 50)
            print(f"{'Iteration':<12} {'Hit Rate':<12} {'Cache Size':<12} {'Total Req':<12}")
            print("-" * 50)
            
            for stats in cache_stats_history:
                print(f"{stats['iteration']:<12} "
                      f"{stats['hit_rate']:<12.2%} "
                      f"{stats['cache_size']:<12} "
                      f"{stats['total_requests']:<12}")
            
            final_stats = cache_stats_history[-1]
            print()
            print(f"Final cache hit rate: {final_stats['hit_rate']:.2%}")
            print()
        
        # Memory usage over time
        if memory_history:
            print("Memory Usage Over Time:")
            print("-" * 30)
            print(f"{'Iteration':<12} {'Memory (MB)':<12}")
            print("-" * 30)
            
            for mem_stat in memory_history:
                print(f"{mem_stat['iteration']:<12} {mem_stat['memory_mb']:<12.1f}")
            
            if len(memory_history) > 1:
                memory_growth = memory_history[-1]['memory_mb'] - memory_history[0]['memory_mb']
                print(f"\nMemory growth: {memory_growth:.1f} MB")
            print()
        
        # Performance recommendations
        print("Performance Recommendations:")
        print("-" * 40)
        
        avg_time = sum(prediction_times) / len(prediction_times) * 1000
        if avg_time > 100:
            print("⚠️  High average prediction time. Consider:")
            print("   - Increasing cache size")
            print("   - Using batch processing")
            print("   - Enabling GPU acceleration")
        elif avg_time < 10:
            print("🚀 Excellent prediction performance!")
        else:
            print("✅ Good prediction performance")
        
        if cache_stats_history:
            final_hit_rate = cache_stats_history[-1]['hit_rate']
            if final_hit_rate > 0.8:
                print("🎯 Excellent cache performance!")
            elif final_hit_rate > 0.5:
                print("✅ Good cache performance")
            else:
                print("⚠️  Low cache hit rate. Consider:")
                print("   - Increasing cache size")
                print("   - Processing similar inputs together")
        
        print()
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        print()

def main():
    """Run all performance optimization examples."""
    print("🚀 LSM Performance Optimization Examples")
    print("=" * 60)
    print()
    
    try:
        # Run all optimization examples
        cache_optimization_example()
        memory_management_example()
        batch_size_optimization_example()
        lazy_loading_comparison()
        concurrent_inference_example()
        performance_monitoring_example()
        
        print("🎉 All performance optimization examples completed!")
        print()
        print("Key Performance Tips:")
        print("- Use appropriate cache sizes for your workload")
        print("- Implement memory management for long-running processes")
        print("- Choose optimal batch sizes through testing")
        print("- Use lazy loading for faster startup")
        print("- Prefer batch processing over concurrent threading")
        print("- Monitor cache hit rates and memory usage")
        print("- Clear caches periodically in long-running applications")
        
    except Exception as e:
        print(f"❌ Performance optimization examples failed: {e}")
        print("Check the troubleshooting guide for help.")

if __name__ == "__main__":
    main()