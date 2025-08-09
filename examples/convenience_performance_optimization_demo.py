#!/usr/bin/env python3
"""
LSM Convenience API Performance Optimization Demo

This example demonstrates the performance monitoring and optimization features
of the LSM convenience API, including automatic memory management, performance
profiling, and benchmarking capabilities.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the path so we can import the LSM modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lsm.convenience import (
    LSMGenerator, PerformanceProfiler, MemoryMonitor, 
    AutoMemoryManager, ConvenienceAPIBenchmark,
    run_quick_benchmark
)
from lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)


def demonstrate_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMONSTRATION")
    print("="*60)
    
    # Create a performance profiler
    profiler = PerformanceProfiler()
    
    # Create sample conversation data
    conversations = [
        "Hello, how are you today?",
        "I need help with my account.",
        "Can you explain how this works?",
        "What are the available options?",
        "Thank you for your assistance.",
    ] * 20  # 100 conversations
    
    print(f"Training LSM model with {len(conversations)} conversations...")
    
    # Profile the training operation
    with profiler.profile_operation("lsm_training_demo", data_size=len(conversations)):
        generator = LSMGenerator(
            window_size=5,
            embedding_dim=64,
            epochs=3,
            auto_optimize_memory=True
        )
        generator.fit(conversations, verbose=False)
    
    # Profile generation operations
    test_prompts = [
        "Hello, how can I help you?",
        "What would you like to know?",
        "I'm here to assist you."
    ]
    
    print(f"Generating responses for {len(test_prompts)} prompts...")
    
    with profiler.profile_operation("lsm_generation_demo", num_prompts=len(test_prompts)):
        responses = generator.batch_generate(test_prompts, auto_optimize_batch=True)
    
    print("\nGenerated responses:")
    for prompt, response in zip(test_prompts, responses):
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")
        print()
    
    # Display performance metrics
    print("\nPerformance Metrics Summary:")
    print("-" * 40)
    
    summary = profiler.get_metrics_summary()
    print(f"Total operations: {summary.get('operation_count', 0)}")
    print(f"Total duration: {summary.get('total_duration', 0):.3f}s")
    print(f"Average duration: {summary.get('avg_duration', 0):.3f}s")
    print(f"Max memory peak: {summary.get('max_memory_peak', 0):.1f}MB")
    
    # Export detailed metrics
    output_path = Path("performance_metrics.json")
    profiler.export_metrics(output_path)
    print(f"\nDetailed metrics exported to: {output_path}")
    
    return generator


def demonstrate_memory_management():
    """Demonstrate automatic memory management."""
    print("\n" + "="*60)
    print("MEMORY MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    # Create memory manager
    memory_manager = AutoMemoryManager(
        memory_threshold=0.8,  # Trigger optimization at 80% memory usage
        cleanup_threshold=0.9   # Aggressive cleanup at 90%
    )
    
    # Create a larger dataset to demonstrate memory optimization
    large_conversations = [
        f"This is conversation number {i} with some content to make it longer. "
        f"We want to demonstrate how the memory manager handles larger datasets "
        f"and optimizes memory usage automatically during training and inference."
        for i in range(500)
    ]
    
    print(f"Creating LSM model for {len(large_conversations)} conversations...")
    
    # Create generator with memory optimization
    generator = LSMGenerator(
        window_size=10,
        embedding_dim=128,
        epochs=5
    )
    
    # Check memory before training
    initial_memory = MemoryMonitor.get_current_memory_mb()
    available_memory = MemoryMonitor.get_available_memory_mb()
    
    print(f"Initial memory usage: {initial_memory:.1f}MB")
    print(f"Available memory: {available_memory:.1f}MB")
    
    # Optimize model configuration for memory
    optimization_info = generator.optimize_for_memory()
    
    print("\nMemory Optimization Results:")
    print("-" * 30)
    if optimization_info.get('optimizations_applied'):
        for optimization in optimization_info['optimizations_applied']:
            print(f"  - {optimization}")
        
        memory_saved = optimization_info.get('memory_saved_mb', 0)
        print(f"  Estimated memory saved: {memory_saved:.1f}MB")
    else:
        print("  No optimizations needed")
    
    # Train with automatic memory management
    print("\nTraining with automatic memory management...")
    
    # Register cleanup callback
    def cleanup_callback():
        print("  Memory cleanup callback triggered")
    
    memory_manager.register_cleanup_callback(cleanup_callback)
    
    # Check memory and optimize before training
    memory_info = memory_manager.check_memory_and_optimize("training_start")
    print(f"Memory check before training: {memory_info.get('memory_percent', 0):.1%} used")
    
    # Train the model
    generator.fit(large_conversations, verbose=False, auto_optimize_memory=True)
    
    # Check memory after training
    final_memory = MemoryMonitor.get_current_memory_mb()
    memory_delta = final_memory - initial_memory
    
    print(f"\nMemory usage after training: {final_memory:.1f}MB")
    print(f"Memory delta: {memory_delta:+.1f}MB")
    
    # Demonstrate batch size optimization
    test_prompts = ["Test prompt"] * 100
    
    print(f"\nOptimizing batch size for {len(test_prompts)} prompts...")
    
    optimal_batch_size = memory_manager.optimize_batch_size(
        data_size=len(test_prompts),
        model_config=generator.get_params()
    )
    
    print(f"Optimal batch size: {optimal_batch_size}")
    
    # Generate with optimized batch size
    responses = generator.batch_generate(
        test_prompts[:10],  # Just first 10 for demo
        auto_optimize_batch=True
    )
    
    print(f"Generated {len(responses)} responses with optimized batching")
    
    return generator


def demonstrate_benchmarking():
    """Demonstrate benchmarking capabilities."""
    print("\n" + "="*60)
    print("BENCHMARKING DEMONSTRATION")
    print("="*60)
    
    print("Running quick benchmark to compare convenience vs direct API...")
    
    # Run quick benchmark
    try:
        benchmark_results = run_quick_benchmark()
        
        print("\nQuick Benchmark Results:")
        print("-" * 30)
        
        if 'summary' in benchmark_results:
            summary = benchmark_results['summary']
            print(f"Total benchmarks: {summary.get('total_benchmarks', 0)}")
            
            if 'average_performance_overhead' in summary:
                print(f"Average performance overhead: {summary['average_performance_overhead']}")
            
            if 'recommendations' in summary:
                print("\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
        
        # Show detailed results for training if available
        if 'training' in benchmark_results:
            print("\nTraining Performance Details:")
            for dataset_size, result in benchmark_results['training'].items():
                if 'comparison' in result:
                    comp = result['comparison']
                    if 'convenience_overhead_percent' in comp:
                        overhead = comp['convenience_overhead_percent']
                        print(f"  {dataset_size} dataset: {overhead:+.1f}% overhead")
        
        # Show memory usage if available
        if 'memory' in benchmark_results:
            print("\nMemory Usage Comparison:")
            for dataset_size, operations in benchmark_results['memory'].items():
                print(f"  {dataset_size} dataset:")
                for operation, memory_data in operations.items():
                    conv_peak = memory_data.get('convenience_api', {}).get('peak_memory_mb', 0)
                    direct_peak = memory_data.get('direct_api', {}).get('peak_memory_mb', 0)
                    
                    if conv_peak > 0 and direct_peak > 0:
                        ratio = conv_peak / direct_peak
                        print(f"    {operation}: {ratio:.2f}x memory usage")
    
    except Exception as e:
        print(f"Benchmark failed: {e}")
        print("This is expected if training components are not available")


def demonstrate_performance_optimization_features():
    """Demonstrate advanced performance optimization features."""
    print("\n" + "="*60)
    print("ADVANCED PERFORMANCE OPTIMIZATION")
    print("="*60)
    
    # Create a generator with performance monitoring enabled
    generator = LSMGenerator(
        window_size=8,
        embedding_dim=96,
        epochs=3
    )
    
    # Enable performance monitoring
    generator.enable_performance_monitoring(True)
    
    # Create test data
    conversations = [
        f"Conversation {i}: This is a test conversation to demonstrate "
        f"performance optimization features in the LSM convenience API."
        for i in range(200)
    ]
    
    print(f"Training model with performance monitoring enabled...")
    
    # Train with performance monitoring
    generator.fit(conversations, verbose=False)
    
    # Get performance metrics
    metrics = generator.get_performance_metrics()
    
    print("\nModel Performance Metrics:")
    print("-" * 30)
    
    model_info = metrics.get('model_info', {})
    print(f"Model class: {model_info.get('class', 'Unknown')}")
    print(f"Is fitted: {model_info.get('is_fitted', False)}")
    print(f"Estimated size: {model_info.get('estimated_size_mb', 0):.1f}MB")
    
    training_metadata = metrics.get('training_metadata', {})
    if training_metadata:
        print(f"Training time: {training_metadata.get('training_time', 0):.2f}s")
        print(f"Data size: {training_metadata.get('data_size', 0)} samples")
        print(f"Epochs: {training_metadata.get('epochs', 0)}")
        print(f"Batch size: {training_metadata.get('batch_size', 0)}")
    
    # Export performance report
    report_path = Path("performance_report.json")
    generator.export_performance_report(report_path)
    print(f"\nPerformance report exported to: {report_path}")
    
    # Demonstrate performance cache clearing
    print("\nClearing performance cache to free memory...")
    generator.clear_performance_cache()
    
    # Test generation with performance monitoring
    test_prompts = [
        "How does performance monitoring work?",
        "What are the optimization features?",
        "Can you explain memory management?"
    ]
    
    print(f"\nGenerating responses with performance monitoring...")
    responses = generator.batch_generate(test_prompts)
    
    for i, (prompt, response) in enumerate(zip(test_prompts, responses)):
        print(f"\n{i+1}. Prompt: {prompt}")
        print(f"   Response: {response}")
    
    return generator


def main():
    """Main demonstration function."""
    print("LSM Convenience API Performance Optimization Demo")
    print("This demo showcases performance monitoring, memory management,")
    print("and benchmarking capabilities of the LSM convenience API.")
    
    try:
        # Demonstrate performance monitoring
        generator1 = demonstrate_performance_monitoring()
        
        # Demonstrate memory management
        generator2 = demonstrate_memory_management()
        
        # Demonstrate benchmarking
        demonstrate_benchmarking()
        
        # Demonstrate advanced optimization features
        generator3 = demonstrate_performance_optimization_features()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        print("\nKey Features Demonstrated:")
        print("- Automatic performance monitoring and profiling")
        print("- Memory usage tracking and optimization")
        print("- Automatic memory management and cleanup")
        print("- Batch size optimization for memory efficiency")
        print("- Model configuration optimization")
        print("- Benchmarking convenience vs direct API")
        print("- Performance metrics export and reporting")
        
        print("\nGenerated Files:")
        print("- performance_metrics.json: Detailed performance metrics")
        print("- performance_report.json: Model performance report")
        
        print("\nThe LSM convenience API provides comprehensive performance")
        print("monitoring and optimization features while maintaining ease of use!")
        
    except ImportError as e:
        print(f"\nDemo requires additional dependencies: {e}")
        print("Please ensure all LSM components are properly installed.")
        
    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        print("This may be due to missing dependencies or system limitations.")


if __name__ == "__main__":
    main()