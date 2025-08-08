#!/usr/bin/env python3
"""
Show examples of dialogue sequences and their embeddings in natural language.
This script demonstrates how the LSM processes dialogue data by showing:
1. Original dialogue texts
2. Input sequences (embeddings converted back to text)  
3. Target outputs (next tokens)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple

# Add current directory to path for imports
sys.path.append('.')

from data_loader import load_data

def find_closest_vocabulary_match(embedding: np.ndarray, tokenizer) -> Tuple[str, float]:
    """
    Find the closest vocabulary item to an embedding.
    
    Args:
        embedding: The embedding vector to match
        tokenizer: The trained tokenizer with vocabulary
        
    Returns:
        Tuple of (closest_text, similarity_score)
    """
    try:
        vocab_embeddings = tokenizer._vocabulary_embeddings
        vocab_texts = tokenizer._vocabulary_texts
        
        # Calculate cosine similarity
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        vocab_norms = vocab_embeddings / (np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(vocab_norms, embedding_norm)
        best_idx = np.argmax(similarities)
        
        return vocab_texts[best_idx], similarities[best_idx]
        
    except Exception as e:
        return f"<error: {str(e)}>", 0.0

def show_dialogue_examples(window_size: int = 8, embedding_dim: int = 64, 
                          num_examples: int = 5) -> None:
    """
    Show examples of how dialogue data is processed into embeddings and sequences.
    """
    
    print("=" * 80)
    print("LSM DIALOGUE PROCESSING EXAMPLES")
    print("=" * 80)
    
    # Load data
    print("Loading dialogue data...")
    try:
        X_train, y_train, X_test, y_test, tokenizer = load_data(
            window_size=window_size,
            test_size=0.2,
            embedding_dim=embedding_dim
        )
        
        print(f"✓ Data loaded successfully")
        print(f"  Training sequences: {X_train.shape[0]}")
        print(f"  Test sequences: {X_test.shape[0]}")
        print(f"  Window size: {window_size}")
        print(f"  Embedding dimension: {embedding_dim}")
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # Show vocabulary information
    print(f"\n📚 VOCABULARY INFORMATION:")
    print(f"  Vocabulary size: {len(tokenizer._vocabulary_texts)}")
    print(f"  Vocabulary items:")
    
    for i, text in enumerate(tokenizer._vocabulary_texts):
        embedding_norm = np.linalg.norm(tokenizer._vocabulary_embeddings[i])
        print(f"    {i+1:2d}: '{text}' (embedding norm: {embedding_norm:.3f})")
    
    # Show how raw dialogue gets processed
    print(f"\n🔄 DATA PROCESSING PIPELINE:")
    print(f"  Raw dialogue text → TF-IDF vectorization → Embedding reduction → Sequence windows")
    
    # Show examples from test set
    print(f"\n🎭 DIALOGUE SEQUENCE EXAMPLES:")
    print(f"These show how the model sees dialogue as sequences of embeddings...")
    
    # Select random examples
    np.random.seed(42)
    example_indices = np.random.choice(len(X_test), min(num_examples, len(X_test)), replace=False)
    
    for i, idx in enumerate(example_indices):
        print(f"\n" + "─" * 80)
        print(f"EXAMPLE {i+1}/{len(example_indices)} (Test Sample #{idx})")
        print("─" * 80)
        
        input_sequence = X_test[idx]  # Shape: (window_size, embedding_dim)
        target_output = y_test[idx]   # Shape: (embedding_dim,)
        
        print(f"\n📝 INPUT SEQUENCE (Dialogue History - {window_size} turns):")
        
        for t in range(window_size):
            embedding = input_sequence[t]
            closest_text, similarity = find_closest_vocabulary_match(embedding, tokenizer)
            
            # Show embedding statistics
            embedding_norm = np.linalg.norm(embedding)
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            
            print(f"  Turn {t+1}: '{closest_text}' (similarity: {similarity:.3f})")
            print(f"          Embedding stats: norm={embedding_norm:.3f}, mean={embedding_mean:.3f}, std={embedding_std:.3f}")
        
        print(f"\n🎯 TARGET OUTPUT (Expected Next Token):")
        target_text, target_similarity = find_closest_vocabulary_match(target_output, tokenizer)
        target_norm = np.linalg.norm(target_output)
        target_mean = np.mean(target_output)
        
        print(f"  Next token: '{target_text}' (similarity: {target_similarity:.3f})")
        print(f"  Target embedding stats: norm={target_norm:.3f}, mean={target_mean:.3f}")
        
        print(f"\n📊 SEQUENCE ANALYSIS:")
        
        # Calculate sequence diversity
        sequence_norms = [np.linalg.norm(input_sequence[t]) for t in range(window_size)]
        sequence_diversity = np.std(sequence_norms)
        
        # Calculate how different each turn is from the others
        similarities_within_sequence = []
        for t1 in range(window_size):
            for t2 in range(t1+1, window_size):
                emb1 = input_sequence[t1] / (np.linalg.norm(input_sequence[t1]) + 1e-8)
                emb2 = input_sequence[t2] / (np.linalg.norm(input_sequence[t2]) + 1e-8)
                sim = np.dot(emb1, emb2)
                similarities_within_sequence.append(sim)
        
        avg_internal_similarity = np.mean(similarities_within_sequence) if similarities_within_sequence else 0.0
        
        print(f"  Sequence diversity: {sequence_diversity:.4f}")
        print(f"  Average internal similarity: {avg_internal_similarity:.4f}")
        print(f"  Sequence complexity: {'High' if sequence_diversity > 0.1 else 'Medium' if sequence_diversity > 0.05 else 'Low'}")
    
    # Show overall dataset statistics
    print(f"\n📈 DATASET STATISTICS:")
    
    # Calculate embedding statistics across all data
    all_embeddings = np.concatenate([X_train.reshape(-1, embedding_dim), X_test.reshape(-1, embedding_dim)])
    embedding_norms = np.linalg.norm(all_embeddings, axis=1)
    
    print(f"  Total embeddings: {len(all_embeddings)}")
    print(f"  Embedding norm statistics:")
    print(f"    Mean: {np.mean(embedding_norms):.4f}")
    print(f"    Std:  {np.std(embedding_norms):.4f}")
    print(f"    Min:  {np.min(embedding_norms):.4f}")
    print(f"    Max:  {np.max(embedding_norms):.4f}")
    
    # Calculate target statistics
    all_targets = np.concatenate([y_train, y_test])
    target_norms = np.linalg.norm(all_targets, axis=1)
    
    print(f"  Target embedding norm statistics:")
    print(f"    Mean: {np.mean(target_norms):.4f}")
    print(f"    Std:  {np.std(target_norms):.4f}")
    print(f"    Min:  {np.min(target_norms):.4f}")
    print(f"    Max:  {np.max(target_norms):.4f}")

def main():
    """Main function to show dialogue examples."""
    parser = argparse.ArgumentParser(
        description="Show examples of LSM dialogue processing in natural language"
    )
    
    parser.add_argument('--window-size', type=int, default=8,
                       help='Size of sequence window')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Dimension of token embeddings')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of examples to show')
    
    args = parser.parse_args()
    
    # Show examples
    show_dialogue_examples(
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        num_examples=args.num_examples
    )
    
    print("\n" + "=" * 80)
    print("DIALOGUE EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nThis shows how the LSM processes dialogue:")
    print("1. Text is converted to embeddings using TF-IDF + dimensionality reduction")
    print("2. Embeddings are organized into sequences of fixed window size")
    print("3. Each sequence represents a dialogue history → next token prediction task")
    print("4. The model learns to map dialogue patterns to appropriate responses")
    
    return 0

if __name__ == "__main__":
    exit(main())