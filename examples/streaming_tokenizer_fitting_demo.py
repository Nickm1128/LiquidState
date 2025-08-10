#!/usr/bin/env python3
"""
Streaming Tokenizer Fitting Demo

This example demonstrates how to use the streaming tokenizer fitting functionality
to train sinusoidal embeddings on large datasets that don't fit in memory.

The streaming approach provides:
- Memory-efficient processing with configurable batch sizes
- Progress tracking and memory usage monitoring
- Automatic batch size adjustment based on memory constraints
- Comprehensive training statistics collection
"""

import os
import tempfile
import json
from pathlib import Path

# Import LSM components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from src.lsm.data.streaming_data_iterator import StreamingDataIterator


def create_sample_dataset(data_dir: Path, num_files: int = 3, texts_per_file: int = 1000):
    """Create sample dataset files for demonstration."""
    print(f"Creating sample dataset with {num_files} files, {texts_per_file} texts per file...")
    
    # Sample texts for different domains
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require large amounts of training data",
        "Natural language processing enables computers to understand human language",
        "Deep learning networks can learn complex patterns from data",
        "Tokenization is the process of breaking text into smaller units",
        "Embeddings represent words as dense vectors in high-dimensional space",
        "Streaming data processing allows handling datasets larger than memory",
        "Sinusoidal patterns can enhance neural network learning capabilities",
        "Liquid state machines are a type of recurrent neural network",
        "Reservoir computing uses fixed random networks for temporal processing"
    ]
    
    for file_idx in range(num_files):
        # Create text file
        text_file = data_dir / f"dataset_{file_idx}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            for i in range(texts_per_file):
                # Vary the text content
                text = sample_texts[i % len(sample_texts)]
                if i % 3 == 0:
                    text = f"Document {i}: {text}"
                elif i % 3 == 1:
                    text = f"{text} This is sample text number {i}."
                f.write(f"{text}\n")
        
        # Create JSON file
        json_file = data_dir / f"dataset_{file_idx}.json"
        json_data = []
        for i in range(texts_per_file // 2):  # Fewer JSON entries
            json_data.append({
                "id": i,
                "text": f"JSON entry {i}: {sample_texts[i % len(sample_texts)]}",
                "metadata": {"file": file_idx, "index": i}
            })
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    
    print(f"Created {num_files * 2} files in {data_dir}")


def progress_callback(progress_info):
    """Progress callback function to track training progress."""
    print(f"Epoch {progress_info['epoch']}/{progress_info['total_epochs']}, "
          f"Batch {progress_info['batch']}, "
          f"Sequences: {progress_info['sequences_processed']}, "
          f"Memory: {progress_info['memory_usage_mb']:.1f}MB, "
          f"Avg seq len: {progress_info['avg_sequence_length']:.1f}")


def demonstrate_basic_streaming_fitting():
    """Demonstrate basic streaming tokenizer fitting."""
    print("\n" + "="*60)
    print("BASIC STREAMING TOKENIZER FITTING DEMO")
    print("="*60)
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        create_sample_dataset(data_dir, num_files=2, texts_per_file=100)
        
        # Create enhanced tokenizer wrapper
        # Using 'gpt2' as an example - in practice you'd use any supported tokenizer
        try:
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer='gpt2',  # This would use HuggingFace GPT-2 tokenizer
                embedding_dim=128,
                max_length=64
            )
        except Exception:
            # Fallback to mock tokenizer for demo if HuggingFace not available
            print("Note: Using mock tokenizer for demo (HuggingFace not available)")
            from tests.test_data.test_streaming_tokenizer_fitting import MockTokenizerAdapter
            from src.lsm.data.enhanced_tokenization import TokenizerConfig
            
            config = TokenizerConfig(backend='mock', model_name='demo-model')
            mock_adapter = MockTokenizerAdapter(config)
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=mock_adapter,
                embedding_dim=128,
                max_length=64
            )
        
        # Fit streaming tokenizer
        print("\nStarting streaming tokenizer fitting...")
        embedder = tokenizer.fit_streaming(
            data_source=str(data_dir),  # Process entire directory
            batch_size=50,
            epochs=2,
            memory_threshold_mb=500.0,
            progress_callback=progress_callback,
            auto_adjust_batch_size=True
        )
        
        # Display training statistics
        stats = tokenizer.get_training_stats()
        print(f"\nTraining completed successfully!")
        print(f"Total sequences processed: {stats['total_sequences']:,}")
        print(f"Total tokens processed: {stats['total_tokens']:,}")
        print(f"Average sequence length: {stats['avg_sequence_length']:.1f}")
        print(f"Vocabulary coverage: {stats['vocab_coverage']:.1f}%")
        print(f"Training time: {stats['training_time']:.2f} seconds")
        print(f"Batches processed: {stats['batches_processed']}")
        print(f"Final batch size: {stats['final_batch_size']}")


def demonstrate_streaming_iterator_usage():
    """Demonstrate using StreamingDataIterator directly."""
    print("\n" + "="*60)
    print("STREAMING DATA ITERATOR DEMO")
    print("="*60)
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        create_sample_dataset(data_dir, num_files=1, texts_per_file=50)
        
        # Create streaming iterator
        iterator = StreamingDataIterator(
            data_source=str(data_dir),
            batch_size=10,
            memory_threshold_mb=100.0,
            auto_adjust_batch_size=True
        )
        
        print(f"\nProcessing data with StreamingDataIterator...")
        batch_count = 0
        total_items = 0
        
        for batch in iterator:
            batch_count += 1
            total_items += len(batch)
            print(f"Batch {batch_count}: {len(batch)} items")
            
            # Show first few items from first batch
            if batch_count == 1:
                print("Sample items from first batch:")
                for i, item in enumerate(batch[:3]):
                    print(f"  {i+1}: {item[:50]}...")
        
        # Get final progress
        progress = iterator.get_progress()
        print(f"\nProcessing completed:")
        print(f"Total batches: {batch_count}")
        print(f"Total items: {total_items}")
        print(f"Memory usage: {progress['memory_usage_mb']:.1f}MB")


def demonstrate_memory_monitoring():
    """Demonstrate memory monitoring and batch size adjustment."""
    print("\n" + "="*60)
    print("MEMORY MONITORING DEMO")
    print("="*60)
    
    # Create temporary dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        create_sample_dataset(data_dir, num_files=1, texts_per_file=200)
        
        # Create tokenizer with memory monitoring
        try:
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer='gpt2',
                embedding_dim=64,
                max_length=32
            )
        except Exception:
            # Fallback to mock tokenizer
            from tests.test_data.test_streaming_tokenizer_fitting import MockTokenizerAdapter
            from src.lsm.data.enhanced_tokenization import TokenizerConfig
            
            config = TokenizerConfig(backend='mock', model_name='demo-model')
            mock_adapter = MockTokenizerAdapter(config)
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=mock_adapter,
                embedding_dim=64,
                max_length=32
            )
        
        # Track memory usage
        memory_history = []
        def memory_progress_callback(progress_info):
            memory_history.append(progress_info['memory_usage_mb'])
            if len(memory_history) % 5 == 0:  # Print every 5th update
                print(f"Memory usage: {progress_info['memory_usage_mb']:.1f}MB, "
                      f"Batch size: {progress_info['current_batch_size']}")
        
        print("\nFitting with memory monitoring (low threshold to trigger adjustment)...")
        embedder = tokenizer.fit_streaming(
            data_source=str(data_dir),
            batch_size=100,  # Start with large batch size
            epochs=1,
            memory_threshold_mb=50.0,  # Low threshold to trigger adjustment
            progress_callback=memory_progress_callback,
            auto_adjust_batch_size=True,
            min_batch_size=10,
            max_batch_size=200
        )
        
        stats = tokenizer.get_training_stats()
        print(f"\nMemory monitoring results:")
        print(f"Initial batch size: 100")
        print(f"Final batch size: {stats['final_batch_size']}")
        print(f"Max memory usage: {max(memory_history):.1f}MB")
        print(f"Training completed successfully with memory management!")


def demonstrate_comprehensive_statistics():
    """Demonstrate comprehensive statistics collection."""
    print("\n" + "="*60)
    print("COMPREHENSIVE STATISTICS DEMO")
    print("="*60)
    
    # Create temporary dataset with varied content
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = Path(temp_dir)
        
        # Create files with different text lengths and vocabulary
        texts_short = ["Short text.", "Brief message.", "Quick note."] * 20
        texts_medium = ["This is a medium length sentence with more words."] * 15
        texts_long = ["This is a much longer sentence that contains many more words and provides more complex vocabulary for the tokenizer to process and learn from."] * 10
        
        # Write different file types
        with open(data_dir / "short_texts.txt", 'w') as f:
            for text in texts_short:
                f.write(f"{text}\n")
        
        with open(data_dir / "medium_texts.txt", 'w') as f:
            for text in texts_medium:
                f.write(f"{text}\n")
        
        with open(data_dir / "long_texts.txt", 'w') as f:
            for text in texts_long:
                f.write(f"{text}\n")
        
        # Create tokenizer
        try:
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer='gpt2',
                embedding_dim=96,
                max_length=128
            )
        except Exception:
            # Fallback to mock tokenizer
            from tests.test_data.test_streaming_tokenizer_fitting import MockTokenizerAdapter
            from src.lsm.data.enhanced_tokenization import TokenizerConfig
            
            config = TokenizerConfig(backend='mock', model_name='demo-model')
            mock_adapter = MockTokenizerAdapter(config)
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=mock_adapter,
                embedding_dim=96,
                max_length=128
            )
        
        # Fit with detailed statistics
        print("\nFitting tokenizer with comprehensive statistics collection...")
        embedder = tokenizer.fit_streaming(
            data_source=str(data_dir),
            batch_size=25,
            epochs=1
        )
        
        # Display comprehensive statistics
        stats = tokenizer.get_training_stats()
        print(f"\nComprehensive Training Statistics:")
        print(f"{'='*40}")
        print(f"Dataset Statistics:")
        print(f"  Total sequences: {stats['total_sequences']:,}")
        print(f"  Total tokens: {stats['total_tokens']:,}")
        print(f"  Average sequence length: {stats['avg_sequence_length']:.2f}")
        print(f"  Vocabulary coverage: {stats['vocab_coverage']:.1f}%")
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {stats['epochs']}")
        print(f"  Batches processed: {stats['batches_processed']}")
        print(f"  Final batch size: {stats['final_batch_size']}")
        print(f"  Memory threshold: {stats['memory_threshold_mb']:.1f}MB")
        
        print(f"\nPerformance Metrics:")
        print(f"  Total training time: {stats['training_time']:.2f} seconds")
        print(f"  Sequences per second: {stats['total_sequences'] / stats['training_time']:.1f}")
        print(f"  Tokens per second: {stats['total_tokens'] / stats['training_time']:.1f}")
        
        print(f"\nVocabulary Statistics (top 5 tokens):")
        vocab_stats = stats['vocab_stats']
        sorted_vocab = sorted(vocab_stats.items(), key=lambda x: x[1], reverse=True)
        for token_id, count in sorted_vocab[:5]:
            print(f"  Token {token_id}: {count:,} occurrences")


def main():
    """Run all streaming tokenizer fitting demonstrations."""
    print("Streaming Tokenizer Fitting Demonstrations")
    print("This demo shows how to use streaming data processing for training")
    print("sinusoidal embeddings on large datasets that don't fit in memory.")
    
    try:
        # Run demonstrations
        demonstrate_basic_streaming_fitting()
        demonstrate_streaming_iterator_usage()
        demonstrate_memory_monitoring()
        demonstrate_comprehensive_statistics()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey Features Demonstrated:")
        print("✓ Memory-efficient streaming data processing")
        print("✓ Configurable batch sizes with automatic adjustment")
        print("✓ Progress tracking and memory usage monitoring")
        print("✓ Comprehensive training statistics collection")
        print("✓ Support for multiple data formats (text, JSON)")
        print("✓ Multi-epoch training with incremental learning")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        print("This may be due to missing dependencies or system constraints.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())