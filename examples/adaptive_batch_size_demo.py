#!/usr/bin/env python3
"""
Demonstration of adaptive batch size management in StreamingDataIterator.

This example shows how the iterator automatically adjusts batch sizes based on
memory usage and system constraints, with fallback mechanisms for memory-constrained
environments.
"""

import tempfile
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.data.streaming_data_iterator import StreamingDataIterator


def create_demo_data():
    """Create demo data files for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create a large text file
    text_file = temp_dir / "large_text.txt"
    with open(text_file, 'w') as f:
        for i in range(1000):
            f.write(f"This is line {i} with some sample text content for testing.\n")
    
    # Create a JSON file
    json_file = temp_dir / "data.json"
    data = [{"id": i, "text": f"Sample text {i}", "value": i * 2} for i in range(500)]
    with open(json_file, 'w') as f:
        json.dump(data, f)
    
    return [text_file, json_file]


def demo_basic_adaptive_management():
    """Demonstrate basic adaptive batch size management."""
    print("=== Basic Adaptive Batch Size Management ===")
    
    files = create_demo_data()
    
    # Create iterator with adaptive management enabled
    iterator = StreamingDataIterator(
        data_source=[str(f) for f in files],
        batch_size=100,  # Start with moderate batch size
        auto_adjust_batch_size=True,
        memory_threshold_mb=50.0,  # Low threshold for demo
        memory_check_interval=2,  # Check every 2 batches
        extract_text=True  # Extract text for processing
    )
    
    print(f"Initial configuration:")
    print(f"  Batch size: {iterator.batch_size}")
    print(f"  Memory threshold: {iterator.memory_threshold_mb}MB")
    print(f"  Auto-adjust enabled: {iterator.auto_adjust_batch_size}")
    
    batch_count = 0
    for batch in iterator:
        batch_count += 1
        progress = iterator.get_progress()
        
        print(f"\nBatch {batch_count}:")
        print(f"  Items in batch: {len(batch)}")
        print(f"  Current batch size: {progress['current_batch_size']}")
        print(f"  Memory usage: {progress['memory_usage_mb']:.1f}MB")
        print(f"  Emergency mode: {progress['emergency_mode']}")
        print(f"  Total adjustments: {progress['total_adjustments']}")
        
        # Stop after a few batches for demo
        if batch_count >= 5:
            break
    
    # Show adaptive statistics
    stats = iterator.get_adaptive_stats()
    print(f"\n=== Adaptive Statistics ===")
    print(f"Total batches processed: {stats['current_state']['batches_processed']}")
    print(f"Adjustment history: {len(stats['adjustment_history'])} adjustments")
    
    for i, adj in enumerate(stats['adjustment_history']):
        print(f"  Adjustment {i+1}: {adj['old_size']} -> {adj['new_size']} ({adj['reason']})")


def demo_memory_constrained_environment():
    """Demonstrate fallback mechanisms for memory-constrained environments."""
    print("\n\n=== Memory-Constrained Environment Demo ===")
    
    files = create_demo_data()
    
    # Create iterator with very low memory thresholds
    iterator = StreamingDataIterator(
        data_source=[str(f) for f in files],
        batch_size=200,  # Start with large batch size
        auto_adjust_batch_size=True,
        memory_threshold_mb=10.0,  # Very low threshold
        emergency_threshold_mb=15.0,  # Low emergency threshold
        min_batch_size=5,  # Small minimum
        memory_check_interval=1,  # Check every batch
        extract_text=True
    )
    
    print(f"Memory-constrained configuration:")
    print(f"  Initial batch size: {iterator.batch_size}")
    print(f"  Memory threshold: {iterator.memory_threshold_mb}MB")
    print(f"  Emergency threshold: {iterator.emergency_threshold_mb}MB")
    print(f"  Minimum batch size: {iterator.min_batch_size}")
    
    batch_count = 0
    for batch in iterator:
        batch_count += 1
        progress = iterator.get_progress()
        
        print(f"\nBatch {batch_count}:")
        print(f"  Items: {len(batch)}, Batch size: {progress['current_batch_size']}")
        print(f"  Memory: {progress['memory_usage_mb']:.1f}MB, Emergency: {progress['emergency_mode']}")
        print(f"  Consecutive adjustments: {progress['consecutive_adjustments']}")
        
        if batch_count >= 3:
            break


def demo_configuration_updates():
    """Demonstrate runtime configuration updates."""
    print("\n\n=== Runtime Configuration Updates ===")
    
    files = create_demo_data()
    
    iterator = StreamingDataIterator(
        data_source=[str(f) for f in files],
        batch_size=50,
        auto_adjust_batch_size=True,
        memory_threshold_mb=100.0,
        extract_text=True
    )
    
    print("Initial settings:")
    config = iterator.get_adaptive_stats()['current_config']
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Update configuration during runtime
    print("\nUpdating configuration...")
    iterator.configure_adaptive_settings(
        memory_threshold_mb=50.0,
        min_batch_size=10,
        max_batch_size=200,
        memory_check_interval=3
    )
    
    print("Updated settings:")
    config = iterator.get_adaptive_stats()['current_config']
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Process some data
    batch_count = 0
    for batch in iterator:
        batch_count += 1
        if batch_count >= 2:
            break
    
    # Show memory trend analysis
    trend = iterator.get_memory_trend()
    print(f"\nMemory trend analysis:")
    print(f"  Trend: {trend['trend']}")
    print(f"  Slope: {trend['slope']:.2f}MB per batch")
    print(f"  R-squared: {trend['r_squared']:.3f}")


def demo_text_extraction_modes():
    """Demonstrate different text extraction modes."""
    print("\n\n=== Text Extraction Modes ===")
    
    files = create_demo_data()
    
    # Mode 1: Raw data (no text extraction)
    print("Mode 1: Raw data (extract_text=False)")
    iterator1 = StreamingDataIterator(
        data_source=[str(files[1])],  # JSON file
        batch_size=3,
        extract_text=False
    )
    
    batch = next(iterator1)
    print(f"  Sample item: {batch[0]}")
    print(f"  Item type: {type(batch[0])}")
    
    # Mode 2: Text extraction enabled
    print("\nMode 2: Text extraction (extract_text=True)")
    iterator2 = StreamingDataIterator(
        data_source=[str(files[1])],  # JSON file
        batch_size=3,
        extract_text=True,
        text_field='text'  # Extract from 'text' field
    )
    
    batch = next(iterator2)
    print(f"  Sample item: {batch[0]}")
    print(f"  Item type: {type(batch[0])}")


if __name__ == "__main__":
    print("Adaptive Batch Size Management Demo")
    print("=" * 50)
    
    try:
        demo_basic_adaptive_management()
        demo_memory_constrained_environment()
        demo_configuration_updates()
        demo_text_extraction_modes()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()