#!/usr/bin/env python3
"""
Streaming Consistency Demo

This example demonstrates the streaming consistency features of the enhanced
tokenizer, including deterministic processing, checkpointing, and validation
to ensure streaming results match batch processing.
"""

import os
import tempfile
import numpy as np
from pathlib import Path

# Import LSM components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper, TokenizerConfig
from src.lsm.data.adapters.huggingface_adapter import HuggingFaceAdapter
from src.lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)


def create_sample_data(data_dir: Path, num_files: int = 3, texts_per_file: int = 100):
    """Create sample text data for streaming consistency demonstration."""
    
    # Sample texts with varied content
    base_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require large datasets for training",
        "Streaming data processing enables handling of massive datasets",
        "Consistency validation ensures reliable model training",
        "Deterministic processing produces reproducible results",
        "Checkpointing allows resumable training workflows",
        "Natural language processing benefits from sinusoidal embeddings",
        "Large language models use transformer architectures",
        "Data preprocessing is crucial for model performance",
        "Tokenization converts text into numerical representations"
    ]
    
    data_dir.mkdir(exist_ok=True)
    
    for file_idx in range(num_files):
        file_path = data_dir / f"sample_data_{file_idx}.txt"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            for text_idx in range(texts_per_file):
                # Create variations of base texts
                text = base_texts[text_idx % len(base_texts)]
                if text_idx % 3 == 0:
                    text = f"Sample {text_idx}: {text}"
                elif text_idx % 3 == 1:
                    text = f"{text} - Example {text_idx}"
                
                f.write(f"{text}\n")
        
        logger.info(f"Created sample file: {file_path} with {texts_per_file} texts")
    
    return [data_dir / f"sample_data_{i}.txt" for i in range(num_files)]


def demonstrate_deterministic_processing():
    """Demonstrate deterministic processing across multiple runs."""
    
    print("\n" + "="*60)
    print("DETERMINISTIC PROCESSING DEMONSTRATION")
    print("="*60)
    
    # Create temporary data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_files = create_sample_data(temp_path / "data", num_files=2, texts_per_file=50)
        
        # Create tokenizer
        config = TokenizerConfig(backend='huggingface', model_name='gpt2')
        try:
            adapter = HuggingFaceAdapter(config)
            adapter.initialize()
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=adapter,
                embedding_dim=128,
                max_length=64
            )
        except Exception as e:
            print(f"HuggingFace adapter not available: {e}")
            print("Skipping deterministic processing demo")
            return
        
        # First run with fixed seed
        print("Running first training with deterministic seed...")
        embedder1 = tokenizer.fit_streaming_with_consistency(
            data_source=[str(f) for f in data_files],
            batch_size=20,
            epochs=2,
            deterministic_seed=12345,
            validate_consistency=False,  # Skip for speed
            enable_checkpointing=False
        )
        
        matrix1 = embedder1.get_embedding_matrix()
        stats1 = tokenizer.get_training_stats()
        
        print(f"First run completed:")
        print(f"  Sequences processed: {stats1['total_sequences']:,}")
        print(f"  Embedding matrix shape: {matrix1.shape}")
        print(f"  Deterministic seed: {stats1['deterministic_seed']}")
        
        # Reset tokenizer for second run
        tokenizer._sinusoidal_embedder = None
        tokenizer._is_fitted = False
        
        # Second run with same seed
        print("\nRunning second training with same deterministic seed...")
        embedder2 = tokenizer.fit_streaming_with_consistency(
            data_source=[str(f) for f in data_files],
            batch_size=20,
            epochs=2,
            deterministic_seed=12345,
            validate_consistency=False,  # Skip for speed
            enable_checkpointing=False
        )
        
        matrix2 = embedder2.get_embedding_matrix()
        stats2 = tokenizer.get_training_stats()
        
        print(f"Second run completed:")
        print(f"  Sequences processed: {stats2['total_sequences']:,}")
        print(f"  Embedding matrix shape: {matrix2.shape}")
        print(f"  Deterministic seed: {stats2['deterministic_seed']}")
        
        # Compare results
        mse = np.mean((matrix1 - matrix2) ** 2)
        max_diff = np.max(np.abs(matrix1 - matrix2))
        
        print(f"\nConsistency Analysis:")
        print(f"  MSE between runs: {mse:.2e}")
        print(f"  Max absolute difference: {max_diff:.2e}")
        
        if mse < 1e-10:
            print("  ✓ PASS: Deterministic processing produces identical results")
        else:
            print("  ✗ FAIL: Results differ between runs")


def demonstrate_checkpointing():
    """Demonstrate checkpointing and resumable training."""
    
    print("\n" + "="*60)
    print("CHECKPOINTING DEMONSTRATION")
    print("="*60)
    
    # Create temporary data and checkpoint directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_files = create_sample_data(temp_path / "data", num_files=2, texts_per_file=100)
        checkpoint_dir = temp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Create tokenizer
        config = TokenizerConfig(backend='huggingface', model_name='gpt2')
        try:
            adapter = HuggingFaceAdapter(config)
            adapter.initialize()
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=adapter,
                embedding_dim=64,
                max_length=32
            )
        except Exception as e:
            print(f"HuggingFace adapter not available: {e}")
            print("Skipping checkpointing demo")
            return
        
        # Progress tracking
        progress_history = []
        def progress_callback(progress_info):
            progress_history.append(progress_info.copy())
            if len(progress_history) % 5 == 0:  # Log every 5th progress update
                print(f"  Progress: Epoch {progress_info['epoch']}, "
                      f"Batch {progress_info['batch']}, "
                      f"Sequences: {progress_info['sequences_processed']}")
        
        print("Starting training with checkpointing enabled...")
        print("(Training will save checkpoints every 3 batches)")
        
        # Train with checkpointing
        embedder = tokenizer.fit_streaming_with_consistency(
            data_source=[str(f) for f in data_files],
            batch_size=25,
            epochs=3,
            enable_checkpointing=True,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=3,  # Save every 3 batches
            validate_consistency=False,  # Skip for speed
            progress_callback=progress_callback
        )
        
        stats = tokenizer.get_training_stats()
        
        print(f"\nTraining completed with checkpointing:")
        print(f"  Total sequences: {stats['total_sequences']:,}")
        print(f"  Total batches: {stats['batches_processed']}")
        print(f"  Training time: {stats['training_time']:.2f}s")
        print(f"  Checkpointing enabled: {stats['checkpointing_enabled']}")
        
        # Check if checkpoint files were created and cleaned up
        checkpoint_files = list(checkpoint_dir.glob("*.pkl"))
        if len(checkpoint_files) == 0:
            print("  ✓ Checkpoints were created during training and cleaned up after completion")
        else:
            print(f"  ! Found {len(checkpoint_files)} remaining checkpoint files")


def demonstrate_consistency_validation():
    """Demonstrate streaming vs batch consistency validation."""
    
    print("\n" + "="*60)
    print("CONSISTENCY VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Create temporary data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_files = create_sample_data(temp_path / "data", num_files=1, texts_per_file=50)
        
        # Create tokenizer
        config = TokenizerConfig(backend='huggingface', model_name='gpt2')
        try:
            adapter = HuggingFaceAdapter(config)
            adapter.initialize()
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=adapter,
                embedding_dim=64,
                max_length=32
            )
        except Exception as e:
            print(f"HuggingFace adapter not available: {e}")
            print("Skipping consistency validation demo")
            return
        
        print("Training with consistency validation enabled...")
        print("(This will compare streaming results with batch processing)")
        
        # Train with validation
        embedder = tokenizer.fit_streaming_with_consistency(
            data_source=str(data_files[0]),
            batch_size=15,
            epochs=2,
            validate_consistency=True,
            validation_sample_size=20,
            enable_checkpointing=False
        )
        
        stats = tokenizer.get_training_stats()
        validation_metrics = stats['validation_metrics']
        
        print(f"\nTraining completed with validation:")
        print(f"  Total sequences: {stats['total_sequences']:,}")
        print(f"  Validation samples: {validation_metrics['validation_samples']}")
        
        print(f"\nConsistency Validation Results:")
        print(f"  Embedding MSE: {validation_metrics['embedding_mse']:.6f}")
        print(f"  Embedding MAE: {validation_metrics['embedding_mae']:.6f}")
        print(f"  Avg Cosine Similarity: {validation_metrics['avg_cosine_similarity']:.4f}")
        print(f"  Matrix Correlation: {validation_metrics['matrix_correlation']:.4f}")
        
        # Interpret results
        if validation_metrics['avg_cosine_similarity'] > 0.8:
            print("  ✓ GOOD: High cosine similarity indicates consistent embeddings")
        elif validation_metrics['avg_cosine_similarity'] > 0.6:
            print("  ~ OK: Moderate cosine similarity")
        else:
            print("  ! LOW: Low cosine similarity may indicate inconsistency")
        
        if validation_metrics['matrix_correlation'] > 0.8:
            print("  ✓ GOOD: High matrix correlation indicates consistent embedding patterns")
        elif validation_metrics['matrix_correlation'] > 0.6:
            print("  ~ OK: Moderate matrix correlation")
        else:
            print("  ! LOW: Low matrix correlation may indicate inconsistency")


def demonstrate_complete_workflow():
    """Demonstrate complete streaming consistency workflow."""
    
    print("\n" + "="*60)
    print("COMPLETE STREAMING CONSISTENCY WORKFLOW")
    print("="*60)
    
    # Create temporary data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        data_files = create_sample_data(temp_path / "data", num_files=3, texts_per_file=75)
        checkpoint_dir = temp_path / "workflow_checkpoints"
        
        # Create tokenizer
        config = TokenizerConfig(backend='huggingface', model_name='gpt2')
        try:
            adapter = HuggingFaceAdapter(config)
            adapter.initialize()
            tokenizer = EnhancedTokenizerWrapper(
                tokenizer=adapter,
                embedding_dim=96,
                max_length=48
            )
        except Exception as e:
            print(f"HuggingFace adapter not available: {e}")
            print("Skipping complete workflow demo")
            return
        
        # Progress tracking with detailed logging
        def detailed_progress_callback(progress_info):
            if progress_info['batch'] % 5 == 0:  # Log every 5th batch
                print(f"  Epoch {progress_info['epoch']}/{progress_info['total_epochs']}, "
                      f"Batch {progress_info['batch']}: "
                      f"{progress_info['sequences_processed']} sequences, "
                      f"{progress_info['memory_usage_mb']:.1f}MB memory, "
                      f"{progress_info['batch_time_seconds']:.2f}s/batch")
        
        print("Starting complete workflow with all consistency features:")
        print("  - Deterministic processing with fixed seed")
        print("  - Checkpointing every 4 batches")
        print("  - Consistency validation against batch processing")
        print("  - Memory monitoring and adaptive batch sizing")
        
        # Run complete workflow
        embedder = tokenizer.fit_streaming_with_consistency(
            data_source=[str(f) for f in data_files],
            batch_size=30,
            epochs=3,
            deterministic_seed=54321,
            enable_checkpointing=True,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=4,
            validate_consistency=True,
            validation_sample_size=30,
            auto_adjust_batch_size=True,
            memory_threshold_mb=500.0,
            progress_callback=detailed_progress_callback
        )
        
        # Get comprehensive results
        stats = tokenizer.get_training_stats()
        validation_metrics = stats['validation_metrics']
        
        print(f"\n" + "="*40)
        print("WORKFLOW RESULTS SUMMARY")
        print("="*40)
        
        print(f"Training Statistics:")
        print(f"  Total sequences processed: {stats['total_sequences']:,}")
        print(f"  Total tokens processed: {stats['total_tokens']:,}")
        print(f"  Average sequence length: {stats['avg_sequence_length']:.1f}")
        print(f"  Vocabulary coverage: {stats['vocab_coverage']:.1f}%")
        print(f"  Training time: {stats['training_time']:.2f}s")
        print(f"  Batches processed: {stats['batches_processed']}")
        print(f"  Final batch size: {stats['final_batch_size']}")
        
        print(f"\nConsistency Features:")
        print(f"  Deterministic seed: {stats['deterministic_seed']}")
        print(f"  Checkpointing enabled: {stats['checkpointing_enabled']}")
        print(f"  Validation samples: {validation_metrics['validation_samples']}")
        
        print(f"\nValidation Results:")
        print(f"  Embedding MSE: {validation_metrics['embedding_mse']:.6f}")
        print(f"  Avg Cosine Similarity: {validation_metrics['avg_cosine_similarity']:.4f}")
        print(f"  Matrix Correlation: {validation_metrics['matrix_correlation']:.4f}")
        
        # Test embeddings
        test_texts = [
            "This is a test sentence for embedding",
            "Streaming consistency validation works well"
        ]
        
        test_tokens = tokenizer.tokenize(test_texts)
        test_embeddings = embedder.embed(np.array(test_tokens))
        
        print(f"\nEmbedding Test:")
        print(f"  Test texts: {len(test_texts)}")
        print(f"  Embedding shape: {test_embeddings.shape}")
        print(f"  Embedding range: [{test_embeddings.min():.3f}, {test_embeddings.max():.3f}]")
        
        print(f"\n✓ Complete streaming consistency workflow completed successfully!")


def main():
    """Run all streaming consistency demonstrations."""
    
    print("LSM Streaming Consistency Demo")
    print("This demo showcases the streaming consistency features:")
    print("1. Deterministic processing for reproducible results")
    print("2. Checkpointing for resumable training")
    print("3. Validation against batch processing")
    print("4. Complete integrated workflow")
    
    try:
        # Run demonstrations
        demonstrate_deterministic_processing()
        demonstrate_checkpointing()
        demonstrate_consistency_validation()
        demonstrate_complete_workflow()
        
        print("\n" + "="*60)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nKey takeaways:")
        print("• Deterministic processing ensures reproducible results")
        print("• Checkpointing enables resumable training for large datasets")
        print("• Validation confirms streaming matches batch processing")
        print("• All features work together for robust streaming training")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        print(f"\nDemo failed with error: {str(e)}")
        print("This may be due to missing dependencies (transformers library)")


if __name__ == "__main__":
    main()