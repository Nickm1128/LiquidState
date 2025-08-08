#!/usr/bin/env python3
"""
Batch Processing Examples
=========================

This script demonstrates efficient batch processing capabilities of the LSM inference system.
Shows how to process multiple dialogue sequences efficiently with memory management.
"""

import sys
import os
import time
import csv
from typing import List, Dict, Any

# Import from the reorganized package structure
try:
    from src.lsm import OptimizedLSMInference, ModelManager
    from src.lsm.utils.lsm_exceptions import ModelLoadError, InferenceError
    # Check if OptimizedLSMInference is actually available (not None)
    if OptimizedLSMInference is None:
        raise ImportError("OptimizedLSMInference is None")
except ImportError as e:
    # Handle TensorFlow import issues gracefully
    if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
        print("❌ TensorFlow import error detected.")
        print("This example requires TensorFlow to be properly installed.")
        print("Please check your TensorFlow installation and try again.")
        print("\nFor installation help, see: https://www.tensorflow.org/install")
        sys.exit(1)
    
    # Fallback to direct imports if package structure isn't complete
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from inference import OptimizedLSMInference
        from src.lsm.utils.lsm_exceptions import ModelLoadError, InferenceError
        from src.lsm.management.model_manager import ModelManager
    except ImportError as fallback_error:
        if "tensorflow" in str(fallback_error).lower() or "dll" in str(fallback_error).lower():
            print("❌ TensorFlow import error detected.")
            print("This example requires TensorFlow to be properly installed.")
            print("Please check your TensorFlow installation and try again.")
            print("\nFor installation help, see: https://www.tensorflow.org/install")
            sys.exit(1)
        else:
            raise fallback_error

def find_example_model():
    """Find an available model for demonstration."""
    manager = ModelManager()
    models = manager.list_available_models()
    
    if not models:
        print("❌ No trained models found!")
        print("Please train a model first using: python main.py train")
        return None
    
    # Use the most recent model
    model = models[0]
    print(f"✅ Using model: {model['path']}")
    print()
    
    return model['path']

def create_sample_dialogues() -> List[List[str]]:
    """Create sample dialogue sequences for batch processing."""
    return [
        ["Hello", "How are you?"],
        ["Good morning", "Nice weather today"],
        ["What's your name?", "I'm Alice"],
        ["How was your day?", "It was great"],
        ["I love programming", "Me too"],
        ["What's the weather like?", "It's sunny"],
        ["Can you help me?", "Of course"],
        ["I'm feeling happy", "That's wonderful"],
        ["What time is it?", "It's 3 PM"],
        ["Where are you from?", "I'm from California"],
        ["Do you like music?", "Yes, I love it"],
        ["What's your favorite food?", "I like pizza"],
        ["How old are you?", "I'm 25"],
        ["What do you do?", "I'm a developer"],
        ["Nice to meet you", "Nice to meet you too"],
        ["Have a great day", "You too"],
        ["See you later", "Goodbye"],
        ["Thanks for your help", "You're welcome"],
        ["I need some advice", "I'm here to help"],
        ["This is interesting", "I agree"]
    ]

def basic_batch_processing_example(inference):
    """Demonstrate basic batch processing."""
    print("📦 Basic Batch Processing Example")
    print("=" * 50)
    
    # Create sample data
    dialogues = create_sample_dialogues()
    
    print(f"Processing {len(dialogues)} dialogue sequences...")
    print()
    
    try:
        # Measure processing time
        start_time = time.time()
        
        # Process all dialogues at once
        predictions = inference.batch_predict(dialogues)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Display results
        print("Results:")
        print("-" * 60)
        for i, (dialogue, prediction) in enumerate(zip(dialogues, predictions), 1):
            dialogue_str = " → ".join(dialogue)
            print(f"{i:2d}. {dialogue_str:<35} → {prediction}")
        
        print()
        print(f"⏱️  Processing Time: {processing_time:.3f} seconds")
        print(f"📊 Average per sequence: {processing_time/len(dialogues)*1000:.1f} ms")
        print()
        
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        print()

def batch_size_comparison_example(inference):
    """Compare different batch sizes for performance."""
    print("⚡ Batch Size Performance Comparison")
    print("=" * 50)
    
    # Create larger dataset for meaningful comparison
    base_dialogues = create_sample_dialogues()
    large_dialogues = base_dialogues * 5  # 100 sequences
    
    batch_sizes = [1, 8, 16, 32, 64]
    results = []
    
    print(f"Testing with {len(large_dialogues)} dialogue sequences...")
    print()
    
    for batch_size in batch_sizes:
        try:
            print(f"Testing batch size: {batch_size}")
            
            start_time = time.time()
            predictions = inference.batch_predict(large_dialogues, batch_size=batch_size)
            end_time = time.time()
            
            processing_time = end_time - start_time
            sequences_per_second = len(large_dialogues) / processing_time
            
            results.append({
                'batch_size': batch_size,
                'total_time': processing_time,
                'sequences_per_second': sequences_per_second,
                'avg_time_per_sequence': processing_time / len(large_dialogues) * 1000
            })
            
            print(f"  Time: {processing_time:.3f}s, Speed: {sequences_per_second:.1f} seq/s")
            
        except Exception as e:
            print(f"  ❌ Failed with batch size {batch_size}: {e}")
    
    print()
    print("Performance Summary:")
    print("-" * 60)
    print(f"{'Batch Size':<12} {'Total Time':<12} {'Speed (seq/s)':<15} {'Avg (ms/seq)':<12}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['batch_size']:<12} "
              f"{result['total_time']:<12.3f} "
              f"{result['sequences_per_second']:<15.1f} "
              f"{result['avg_time_per_sequence']:<12.1f}")
    
    # Find optimal batch size
    if results:
        best_result = max(results, key=lambda x: x['sequences_per_second'])
        print()
        print(f"🏆 Optimal batch size: {best_result['batch_size']} "
              f"({best_result['sequences_per_second']:.1f} sequences/second)")
    
    print()

def memory_efficient_processing_example(inference):
    """Demonstrate memory-efficient processing of large datasets."""
    print("💾 Memory-Efficient Processing Example")
    print("=" * 50)
    
    # Create a very large dataset
    base_dialogues = create_sample_dialogues()
    huge_dialogues = base_dialogues * 20  # 400 sequences
    
    print(f"Processing {len(huge_dialogues)} dialogue sequences with memory management...")
    print()
    
    try:
        # Get initial cache stats
        initial_stats = inference.get_cache_stats()
        
        # Process in chunks with memory management
        chunk_size = 50
        all_predictions = []
        
        for i in range(0, len(huge_dialogues), chunk_size):
            chunk = huge_dialogues[i:i + chunk_size]
            chunk_num = i // chunk_size + 1
            total_chunks = (len(huge_dialogues) + chunk_size - 1) // chunk_size
            
            print(f"Processing chunk {chunk_num}/{total_chunks} ({len(chunk)} sequences)...")
            
            # Process chunk
            chunk_predictions = inference.batch_predict(chunk, batch_size=16)
            all_predictions.extend(chunk_predictions)
            
            # Get memory stats
            stats = inference.get_cache_stats()
            if 'memory_mb' in stats:
                print(f"  Memory usage: {stats['memory_mb']:.1f} MB")
            
            # Clear cache periodically to manage memory
            if chunk_num % 4 == 0:  # Every 4 chunks
                print("  🧹 Clearing caches to manage memory...")
                inference.clear_caches()
        
        # Final statistics
        final_stats = inference.get_cache_stats()
        
        print()
        print("Memory Management Results:")
        print("-" * 40)
        print(f"Total sequences processed: {len(all_predictions)}")
        print(f"Initial cache size: {initial_stats.get('prediction_cache_size', 0)}")
        print(f"Final cache size: {final_stats.get('prediction_cache_size', 0)}")
        
        if 'memory_mb' in final_stats:
            print(f"Final memory usage: {final_stats['memory_mb']:.1f} MB")
        
        print("✅ Large dataset processed successfully with memory management!")
        print()
        
    except Exception as e:
        print(f"❌ Memory-efficient processing failed: {e}")
        print()

def file_based_batch_processing_example(inference):
    """Demonstrate processing dialogues from a file."""
    print("📄 File-Based Batch Processing Example")
    print("=" * 50)
    
    # Create sample input file
    input_file = "sample_dialogues.csv"
    output_file = "predictions_output.csv"
    
    try:
        # Create sample CSV file
        dialogues = create_sample_dialogues()
        
        print(f"Creating sample input file: {input_file}")
        with open(input_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dialogue_id', 'sequence'])  # Header
            
            for i, dialogue in enumerate(dialogues, 1):
                # Join dialogue with a separator
                sequence_str = " | ".join(dialogue)
                writer.writerow([f"dialogue_{i:03d}", sequence_str])
        
        print(f"✅ Created {input_file} with {len(dialogues)} dialogues")
        print()
        
        # Read and process file
        print("Reading dialogues from file...")
        file_dialogues = []
        dialogue_ids = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogue_id = row['dialogue_id']
                sequence_str = row['sequence']
                
                # Split sequence back into list
                dialogue = sequence_str.split(' | ')
                
                file_dialogues.append(dialogue)
                dialogue_ids.append(dialogue_id)
        
        print(f"📖 Read {len(file_dialogues)} dialogues from file")
        
        # Process dialogues
        print("Processing dialogues...")
        start_time = time.time()
        predictions = inference.batch_predict(file_dialogues)
        processing_time = time.time() - start_time
        
        # Save results to output file
        print(f"Saving results to: {output_file}")
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dialogue_id', 'input_sequence', 'predicted_next_token'])
            
            for dialogue_id, dialogue, prediction in zip(dialogue_ids, file_dialogues, predictions):
                input_sequence = " | ".join(dialogue)
                writer.writerow([dialogue_id, input_sequence, prediction])
        
        print(f"✅ Results saved to {output_file}")
        print(f"⏱️  Processing time: {processing_time:.3f} seconds")
        print(f"📊 Average per sequence: {processing_time/len(file_dialogues)*1000:.1f} ms")
        print()
        
        # Show sample results
        print("Sample Results:")
        print("-" * 80)
        for i in range(min(5, len(predictions))):
            dialogue_str = " → ".join(file_dialogues[i])
            print(f"{dialogue_ids[i]}: {dialogue_str:<40} → {predictions[i]}")
        
        if len(predictions) > 5:
            print(f"... and {len(predictions) - 5} more results in {output_file}")
        
        print()
        
    except Exception as e:
        print(f"❌ File-based processing failed: {e}")
        print()
    
    finally:
        # Clean up temporary files
        for temp_file in [input_file, output_file]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🧹 Cleaned up temporary file: {temp_file}")
                except:
                    pass

def performance_monitoring_example(inference):
    """Demonstrate performance monitoring during batch processing."""
    print("📊 Performance Monitoring Example")
    print("=" * 50)
    
    dialogues = create_sample_dialogues() * 3  # 60 sequences
    
    print(f"Processing {len(dialogues)} sequences with performance monitoring...")
    print()
    
    try:
        # Get initial stats
        initial_stats = inference.get_cache_stats()
        
        # Process with timing
        start_time = time.time()
        predictions = inference.batch_predict(dialogues, batch_size=16)
        end_time = time.time()
        
        # Get final stats
        final_stats = inference.get_cache_stats()
        
        # Calculate metrics
        total_time = end_time - start_time
        sequences_per_second = len(dialogues) / total_time
        avg_time_per_sequence = total_time / len(dialogues) * 1000
        
        # Display performance metrics
        print("Performance Metrics:")
        print("-" * 40)
        print(f"Total sequences: {len(dialogues)}")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Sequences per second: {sequences_per_second:.1f}")
        print(f"Average time per sequence: {avg_time_per_sequence:.1f} ms")
        print()
        
        print("Cache Performance:")
        print("-" * 40)
        print(f"Initial cache size: {initial_stats.get('prediction_cache_size', 0)}")
        print(f"Final cache size: {final_stats.get('prediction_cache_size', 0)}")
        print(f"Cache hit rate: {final_stats.get('hit_rate', 0):.2%}")
        print(f"Total requests: {final_stats.get('total_requests', 0)}")
        print(f"Cache hits: {final_stats.get('cache_hits', 0)}")
        
        if 'memory_mb' in final_stats:
            print(f"Memory usage: {final_stats['memory_mb']:.1f} MB")
        
        print()
        
        # Performance recommendations
        print("Performance Recommendations:")
        print("-" * 40)
        
        if sequences_per_second < 10:
            print("⚠️  Low throughput detected. Consider:")
            print("   - Increasing batch size")
            print("   - Enabling GPU acceleration")
            print("   - Reducing cache size if memory-limited")
        elif sequences_per_second > 50:
            print("🚀 Excellent throughput!")
        else:
            print("✅ Good throughput performance")
        
        hit_rate = final_stats.get('hit_rate', 0)
        if hit_rate < 0.1:
            print("⚠️  Low cache hit rate. Consider:")
            print("   - Increasing cache size")
            print("   - Processing similar sequences together")
        elif hit_rate > 0.5:
            print("🎯 Excellent cache performance!")
        
        print()
        
    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        print()

def main():
    """Run all batch processing examples."""
    print("🚀 LSM Batch Processing Examples")
    print("=" * 60)
    print()
    
    # Find a model to use
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        # Initialize inference with optimizations for batch processing
        print("🔧 Initializing inference system for batch processing...")
        inference = OptimizedLSMInference(
            model_path=model_path,
            lazy_load=True,
            cache_size=2000,  # Larger cache for batch processing
            max_batch_size=64  # Larger batch size
        )
        print("✅ Inference system ready!")
        print()
        
        # Run examples
        basic_batch_processing_example(inference)
        batch_size_comparison_example(inference)
        memory_efficient_processing_example(inference)
        file_based_batch_processing_example(inference)
        performance_monitoring_example(inference)
        
        print("🎉 All batch processing examples completed successfully!")
        print()
        print("Key Takeaways:")
        print("- Use appropriate batch sizes for your hardware")
        print("- Monitor memory usage for large datasets")
        print("- Clear caches periodically for long-running processes")
        print("- Use file-based processing for very large datasets")
        print("- Monitor cache hit rates for performance optimization")
        
    except ModelLoadError as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure the model directory contains all required files.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check the troubleshooting guide in README.md for help.")

if __name__ == "__main__":
    main()