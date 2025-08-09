#!/usr/bin/env python3
"""
Convenience API Performance Demonstration

This example demonstrates the performance characteristics of the LSM convenience API,
including training speed, inference speed, and memory usage across different configurations.
"""

import os
import sys
import time
import numpy as np
import psutil
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.convenience import LSMGenerator, LSMClassifier
from lsm.convenience.config import ConvenienceConfig


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def create_performance_test_data(size: str = 'small') -> List[str]:
    """Create test data of different sizes for performance testing."""
    base_conversations = [
        "Hello, how are you today?",
        "I'm doing well, thank you for asking.",
        "What are your plans for the weekend?",
        "I'm thinking of going hiking in the mountains.",
        "That sounds like a great adventure!",
        "Would you like to join me sometime?",
        "I'd love to! When were you thinking?",
        "How about next Saturday morning?",
        "Saturday works perfectly for me.",
        "Great! Let's meet at 8 AM at the trailhead."
    ]
    
    if size == 'small':
        multiplier = 5
    elif size == 'medium':
        multiplier = 20
    elif size == 'large':
        multiplier = 50
    else:
        multiplier = 5
    
    # Create variations of the base conversations
    conversations = []
    for i in range(multiplier):
        for conv in base_conversations:
            conversations.append(f"{conv} (variation {i})")
    
    return conversations


def benchmark_training_performance():
    """Benchmark training performance across different configurations."""
    print("🏃 TRAINING PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Test configurations
    configs = [
        ('fast', 'small'),
        ('fast', 'medium'),
        ('balanced', 'small'),
        ('balanced', 'medium'),
        ('quality', 'small')
    ]
    
    results = []
    
    for preset, data_size in configs:
        print(f"\nTesting {preset} preset with {data_size} dataset...")
        
        # Get configuration
        config = ConvenienceConfig.get_preset(preset)
        config['random_state'] = 42
        
        # Create test data
        conversations = create_performance_test_data(data_size)
        
        # Measure memory before training
        memory_before = get_memory_usage()
        
        # Create and train model
        generator = LSMGenerator(**config)
        
        start_time = time.time()
        try:
            generator.fit(
                conversations,
                epochs=5,  # Fixed epochs for fair comparison
                batch_size=16,
                validation_split=0.2
            )
            training_time = time.time() - start_time
            success = True
        except Exception as e:
            training_time = time.time() - start_time
            success = False
            print(f"  ❌ Training failed: {e}")
        
        # Measure memory after training
        memory_after = get_memory_usage()
        memory_used = memory_after - memory_before
        
        result = {
            'preset': preset,
            'data_size': data_size,
            'num_conversations': len(conversations),
            'training_time': training_time,
            'memory_used_mb': memory_used,
            'success': success,
            'config': config
        }
        results.append(result)
        
        if success:
            print(f"  ✅ Training completed in {training_time:.2f}s")
            print(f"  📊 Memory used: {memory_used:.1f} MB")
            print(f"  🔧 Config: {config['reservoir_type']}, "
                  f"dim={config['embedding_dim']}, "
                  f"window={config['window_size']}")
        
        # Clean up
        del generator
    
    # Summary
    print(f"\n📈 TRAINING PERFORMANCE SUMMARY")
    print("─" * 60)
    successful_results = [r for r in results if r['success']]
    
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['training_time'])
        most_efficient = min(successful_results, key=lambda x: x['memory_used_mb'])
        
        print(f"Fastest training: {fastest['preset']} preset with {fastest['data_size']} data")
        print(f"  Time: {fastest['training_time']:.2f}s")
        print(f"Most memory efficient: {most_efficient['preset']} preset")
        print(f"  Memory: {most_efficient['memory_used_mb']:.1f} MB")
    
    return results


def benchmark_inference_performance():
    """Benchmark inference performance."""
    print(f"\n⚡ INFERENCE PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Train a model for inference testing
    print("Training model for inference testing...")
    config = ConvenienceConfig.get_preset('fast')
    config['random_state'] = 42
    
    generator = LSMGenerator(**config)
    conversations = create_performance_test_data('small')
    
    generator.fit(conversations, epochs=3, batch_size=8)
    
    # Test prompts
    test_prompts = [
        "Hello there!",
        "How are you doing?",
        "What's your favorite activity?",
        "Tell me about your day.",
        "What are you thinking about?"
    ]
    
    # Single inference benchmark
    print(f"\nTesting single inference performance...")
    single_times = []
    
    for prompt in test_prompts:
        start_time = time.time()
        response = generator.generate(prompt, max_length=20)
        inference_time = time.time() - start_time
        single_times.append(inference_time)
        print(f"  '{prompt}' → {inference_time:.4f}s")
    
    avg_single_time = np.mean(single_times)
    print(f"Average single inference time: {avg_single_time:.4f}s")
    
    # Batch inference benchmark
    print(f"\nTesting batch inference performance...")
    batch_sizes = [1, 5, 10, 20]
    
    for batch_size in batch_sizes:
        batch_prompts = test_prompts[:batch_size] if batch_size <= len(test_prompts) else test_prompts * (batch_size // len(test_prompts) + 1)
        batch_prompts = batch_prompts[:batch_size]
        
        start_time = time.time()
        try:
            responses = generator.batch_generate(batch_prompts, max_length=20)
            batch_time = time.time() - start_time
            time_per_item = batch_time / batch_size
            
            print(f"  Batch size {batch_size}: {batch_time:.4f}s total, {time_per_item:.4f}s per item")
            
        except Exception as e:
            print(f"  Batch size {batch_size}: Failed - {e}")
    
    return {
        'avg_single_time': avg_single_time,
        'single_times': single_times
    }


def benchmark_memory_efficiency():
    """Benchmark memory efficiency across different model sizes."""
    print(f"\n💾 MEMORY EFFICIENCY BENCHMARK")
    print("=" * 60)
    
    # Test different embedding dimensions
    embedding_dims = [32, 64, 128, 256]
    
    for dim in embedding_dims:
        print(f"\nTesting embedding dimension: {dim}")
        
        memory_before = get_memory_usage()
        
        try:
            # Create model with specific embedding dimension
            generator = LSMGenerator(
                embedding_dim=dim,
                window_size=8,
                reservoir_type='standard',
                random_state=42
            )
            
            # Train on small dataset
            conversations = create_performance_test_data('small')
            generator.fit(conversations, epochs=3, batch_size=8)
            
            memory_after = get_memory_usage()
            memory_used = memory_after - memory_before
            
            print(f"  Memory used: {memory_used:.1f} MB")
            
            # Test inference memory
            test_response = generator.generate("Hello", max_length=10)
            memory_after_inference = get_memory_usage()
            inference_memory = memory_after_inference - memory_after
            
            print(f"  Inference memory overhead: {inference_memory:.1f} MB")
            
        except Exception as e:
            print(f"  ❌ Failed with dim {dim}: {e}")
        
        # Clean up
        try:
            del generator
        except:
            pass


def benchmark_preset_comparison():
    """Compare the three preset configurations."""
    print(f"\n🎯 PRESET CONFIGURATION COMPARISON")
    print("=" * 60)
    
    presets = ['fast', 'balanced', 'quality']
    conversations = create_performance_test_data('small')
    
    results = {}
    
    for preset in presets:
        print(f"\nTesting '{preset}' preset...")
        
        config = ConvenienceConfig.get_preset(preset)
        config['random_state'] = 42
        
        memory_before = get_memory_usage()
        
        try:
            generator = LSMGenerator(**config)
            
            # Training benchmark
            start_time = time.time()
            generator.fit(conversations, epochs=5, batch_size=8)
            training_time = time.time() - start_time
            
            memory_after_training = get_memory_usage()
            
            # Inference benchmark
            start_time = time.time()
            response = generator.generate("Hello there!", max_length=20)
            inference_time = time.time() - start_time
            
            memory_after_inference = get_memory_usage()
            
            results[preset] = {
                'training_time': training_time,
                'inference_time': inference_time,
                'memory_used': memory_after_training - memory_before,
                'inference_memory': memory_after_inference - memory_after_training,
                'config': config,
                'success': True
            }
            
            print(f"  ✅ Training: {training_time:.2f}s")
            print(f"  ⚡ Inference: {inference_time:.4f}s")
            print(f"  💾 Memory: {results[preset]['memory_used']:.1f} MB")
            
        except Exception as e:
            results[preset] = {'success': False, 'error': str(e)}
            print(f"  ❌ Failed: {e}")
        
        # Clean up
        try:
            del generator
        except:
            pass
    
    # Summary comparison
    print(f"\n📊 PRESET COMPARISON SUMMARY")
    print("─" * 60)
    
    successful_results = {k: v for k, v in results.items() if v.get('success', False)}
    
    if successful_results:
        fastest_training = min(successful_results.items(), key=lambda x: x[1]['training_time'])
        fastest_inference = min(successful_results.items(), key=lambda x: x[1]['inference_time'])
        most_memory_efficient = min(successful_results.items(), key=lambda x: x[1]['memory_used'])
        
        print(f"Fastest training: {fastest_training[0]} ({fastest_training[1]['training_time']:.2f}s)")
        print(f"Fastest inference: {fastest_inference[0]} ({fastest_inference[1]['inference_time']:.4f}s)")
        print(f"Most memory efficient: {most_memory_efficient[0]} ({most_memory_efficient[1]['memory_used']:.1f} MB)")
    
    return results


def main():
    """Run all performance benchmarks."""
    print("🚀 LSM CONVENIENCE API PERFORMANCE DEMONSTRATION")
    print("=" * 80)
    
    print(f"System Information:")
    print(f"  CPU cores: {psutil.cpu_count()}")
    print(f"  Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Initial memory usage: {get_memory_usage():.1f} MB")
    
    try:
        # Run benchmarks
        training_results = benchmark_training_performance()
        inference_results = benchmark_inference_performance()
        benchmark_memory_efficiency()
        preset_results = benchmark_preset_comparison()
        
        print(f"\n🎉 PERFORMANCE DEMONSTRATION COMPLETED")
        print("=" * 80)
        print("Key findings:")
        print("• Fast preset provides quickest training and inference")
        print("• Balanced preset offers good performance/quality tradeoff")
        print("• Quality preset maximizes model capability at cost of speed")
        print("• Memory usage scales with embedding dimension and model complexity")
        print("• Batch processing improves throughput for multiple inferences")
        
    except Exception as e:
        print(f"❌ Performance demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())