#!/usr/bin/env python3
"""
Demonstration of the new standard tokenizer integration and sinusoidal embeddings.

This script shows how to use the StandardTokenizerWrapper, SinusoidalEmbedder,
and EmbeddingOptimizer for enhanced LSM training.
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.data.tokenization import (
    StandardTokenizerWrapper, 
    SinusoidalEmbedder, 
    EmbeddingOptimizer
)

def demonstrate_standard_tokenizer():
    """Demonstrate StandardTokenizerWrapper usage."""
    print("🔤 Standard Tokenizer Demonstration")
    print("=" * 50)
    
    try:
        # Initialize tokenizer
        tokenizer = StandardTokenizerWrapper('gpt2', max_length=128)
        print(f"✅ Initialized {tokenizer}")
        
        # Sample texts
        texts = [
            "Hello, how are you today?",
            "I'm doing well, thank you!",
            "What's your favorite hobby?",
            "I enjoy reading and learning."
        ]
        
        # Tokenize texts
        print("\n📝 Tokenizing texts:")
        for i, text in enumerate(texts):
            tokens = tokenizer.encode_single(text)
            decoded = tokenizer.decode_single(tokens)
            print(f"  {i+1}. '{text}'")
            print(f"     Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"     Decoded: '{decoded}'")
        
        # Batch tokenization
        print("\n📦 Batch tokenization:")
        batch_tokens = tokenizer.tokenize(texts)
        batch_decoded = tokenizer.decode(batch_tokens)
        
        print(f"  Batch shape: {len(batch_tokens)} sequences")
        print(f"  Vocab size: {tokenizer.get_vocab_size()}")
        
        # Special tokens
        special_tokens = tokenizer.get_special_tokens()
        print(f"  Special tokens: {special_tokens}")
        
    except Exception as e:
        print(f"❌ Error with standard tokenizer: {e}")
        print("   Note: This requires the 'transformers' library")

def demonstrate_sinusoidal_embedder():
    """Demonstrate SinusoidalEmbedder usage."""
    print("\n🌊 Sinusoidal Embedder Demonstration")
    print("=" * 50)
    
    # Create embedder
    vocab_size = 1000
    embedding_dim = 128
    embedder = SinusoidalEmbedder(vocab_size=vocab_size, embedding_dim=embedding_dim)
    print(f"✅ Created {embedder}")
    
    # Create sample training data
    np.random.seed(42)
    training_data = np.random.randint(0, vocab_size, (50, 20))
    print(f"📊 Training data shape: {training_data.shape}")
    
    # Fit embedder
    print("🏋️ Fitting embedder...")
    embedder.fit(training_data, epochs=3)
    print("✅ Embedder fitted successfully")
    
    # Test embedding
    test_tokens = [1, 5, 10, 50, 100]
    embeddings = embedder.embed(test_tokens)
    print(f"🔢 Embedded {len(test_tokens)} tokens to shape: {embeddings.shape}")
    
    # Analyze embedding properties
    embedding_matrix = embedder.get_embedding_matrix()
    norms = np.linalg.norm(embedding_matrix, axis=1)
    print(f"📈 Embedding norms - mean: {np.mean(norms):.4f}, std: {np.std(norms):.4f}")
    
    # Test batch embedding
    batch_tokens = np.array([[1, 5, 10], [2, 6, 11], [3, 7, 12]])
    batch_embeddings = embedder.embed(batch_tokens)
    print(f"📦 Batch embeddings shape: {batch_embeddings.shape}")

def demonstrate_embedding_optimizer():
    """Demonstrate EmbeddingOptimizer usage."""
    print("\n⚡ Embedding Optimizer Demonstration")
    print("=" * 50)
    
    # Create optimizer
    optimizer = EmbeddingOptimizer(learning_rate=0.05, max_iterations=5)
    print(f"✅ Created {optimizer}")
    
    # Create sample embeddings
    np.random.seed(42)
    vocab_size, embedding_dim = 100, 64
    sample_embeddings = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
    
    # Normalize embeddings
    norms = np.linalg.norm(sample_embeddings, axis=1, keepdims=True)
    sample_embeddings = sample_embeddings / (norms + 1e-8)
    
    print(f"📊 Sample embeddings shape: {sample_embeddings.shape}")
    
    # Analyze initial sinusoidality
    print("🔍 Analyzing initial sinusoidality...")
    initial_metrics = optimizer.analyze_sinusoidality(sample_embeddings)
    print(f"  Frequency score: {initial_metrics['frequency_score']:.4f}")
    print(f"  Autocorr score: {initial_metrics['autocorr_score']:.4f}")
    print(f"  Smoothness score: {initial_metrics['smoothness_score']:.4f}")
    print(f"  Overall score: {initial_metrics['overall_score']:.4f}")
    
    # Analyze reservoir compatibility
    print("🔍 Analyzing reservoir compatibility...")
    compat_metrics = optimizer.evaluate_reservoir_compatibility(sample_embeddings)
    print(f"  Diversity score: {compat_metrics['diversity_score']:.4f}")
    print(f"  Norm consistency: {compat_metrics['norm_consistency']:.4f}")
    print(f"  Compatibility score: {compat_metrics['compatibility_score']:.4f}")
    
    # Optimize embeddings
    print("⚡ Optimizing embeddings...")
    optimized_embeddings, opt_info = optimizer.optimize_embeddings(sample_embeddings)
    
    print(f"✅ Optimization completed in {opt_info['iterations']} iterations")
    print(f"  Final score: {opt_info['final_score']:.4f}")
    print(f"  Converged: {opt_info['converged']}")
    
    # Analyze optimized embeddings
    print("🔍 Analyzing optimized embeddings...")
    final_metrics = optimizer.analyze_sinusoidality(optimized_embeddings)
    print(f"  Frequency score: {final_metrics['frequency_score']:.4f} "
          f"(Δ: {final_metrics['frequency_score'] - initial_metrics['frequency_score']:+.4f})")
    print(f"  Overall score: {final_metrics['overall_score']:.4f} "
          f"(Δ: {final_metrics['overall_score'] - initial_metrics['overall_score']:+.4f})")

def demonstrate_integration():
    """Demonstrate integration of all components."""
    print("\n🔗 Integration Demonstration")
    print("=" * 50)
    
    try:
        # Create tokenizer
        tokenizer = StandardTokenizerWrapper('gpt2', max_length=64)
        vocab_size = tokenizer.get_vocab_size()
        
        # Create embedder with tokenizer vocab size
        embedder = SinusoidalEmbedder(vocab_size=vocab_size, embedding_dim=128)
        
        # Sample texts
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is fascinating and powerful.",
            "Natural language processing enables amazing applications."
        ]
        
        # Tokenize texts
        tokenized = tokenizer.tokenize(texts, padding=True, truncation=True)
        print(f"📝 Tokenized {len(texts)} texts")
        
        # Create training data from tokens
        training_data = np.array(tokenized)
        print(f"📊 Training data shape: {training_data.shape}")
        
        # Fit embedder
        print("🏋️ Fitting embedder on tokenized data...")
        embedder.fit(training_data, epochs=2)
        
        # Create optimizer and optimize embeddings
        optimizer = EmbeddingOptimizer(learning_rate=0.02, max_iterations=3)
        
        initial_embeddings = embedder.get_embedding_matrix()
        optimized_embeddings, _ = optimizer.optimize_embeddings(initial_embeddings)
        
        # Update embedder with optimized embeddings
        embedder._embedding_matrix = optimized_embeddings
        
        # Test final embedding
        test_text = "This is a test sentence."
        test_tokens = tokenizer.encode_single(test_text)
        final_embeddings = embedder.embed(test_tokens)
        
        print(f"✅ Final embeddings shape: {final_embeddings.shape}")
        print(f"🎯 Successfully integrated tokenizer, embedder, and optimizer!")
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        print("   Note: This requires the 'transformers' library")

def main():
    """Run all demonstrations."""
    print("🚀 LSM Enhanced Tokenization System Demo")
    print("=" * 60)
    
    demonstrate_standard_tokenizer()
    demonstrate_sinusoidal_embedder()
    demonstrate_embedding_optimizer()
    demonstrate_integration()
    
    print("\n🎉 Demo completed!")
    print("\nKey Features Demonstrated:")
    print("  ✅ Standard tokenizer integration (GPT-2, BERT, etc.)")
    print("  ✅ Sinusoidal embedding optimization")
    print("  ✅ Embedding quality analysis and optimization")
    print("  ✅ Full pipeline integration")

if __name__ == "__main__":
    main()