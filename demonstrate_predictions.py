#!/usr/bin/env python3
"""
Demonstrate LSM predictions by training a mini model and showing examples.
This shows real input/target/prediction comparisons in natural language.
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple

# Add current directory to path for imports
sys.path.append('.')

from data_loader import load_data
from train import LSMTrainer

def decode_embedding_to_text(embedding: np.ndarray, tokenizer, top_k: int = 3) -> str:
    """
    Convert an embedding back to the most likely text representation.
    """
    try:
        vocab_embeddings = tokenizer._vocabulary_embeddings
        vocab_texts = tokenizer._vocabulary_texts
        
        # Calculate cosine similarity
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        vocab_norms = vocab_embeddings / (np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(vocab_norms, embedding_norm)
        
        # Get top-k most similar
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            similarity = similarities[idx]
            text = vocab_texts[idx]
            results.append(f"{text} (sim: {similarity:.3f})")
        
        return " | ".join(results)
        
    except Exception as e:
        return f"<decode_error: {str(e)}>"

def format_sequence_dialogue(sequence: np.ndarray, tokenizer, window_size: int) -> List[str]:
    """Format a sequence as dialogue turns."""
    dialogue = []
    for t in range(window_size):
        embedding = sequence[t]
        text = decode_embedding_to_text(embedding, tokenizer, top_k=1)
        # Extract just the most likely match
        text = text.split(" (sim:")[0] if " (sim:" in text else text
        dialogue.append(f"Turn {t+1}: \"{text}\"")
    
    return dialogue

def train_and_demonstrate(window_size: int = 5, embedding_dim: int = 32, 
                         epochs: int = 3, num_examples: int = 3) -> None:
    """
    Train a quick LSM model and show prediction examples.
    """
    
    print("=" * 80)
    print("LSM PREDICTION DEMONSTRATION")
    print("=" * 80)
    
    # Load data with smaller dimensions for faster training
    print("Loading dialogue data...")
    try:
        X_train, y_train, X_test, y_test, tokenizer = load_data(
            window_size=window_size,
            test_size=0.3,  # Larger test set for more examples
            embedding_dim=embedding_dim
        )
        
        print(f"Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test sequences")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Show vocabulary
    print(f"\nVocabulary ({len(tokenizer._vocabulary_texts)} items):")
    for i, text in enumerate(tokenizer._vocabulary_texts):
        print(f"  {i+1}: \"{text}\"")
    
    # Initialize and train a small model
    print(f"\nTraining LSM model (window_size={window_size}, embedding_dim={embedding_dim})...")
    
    trainer = LSMTrainer(
        window_size=window_size,
        embedding_dim=embedding_dim,
        reservoir_units=[64, 32],  # Smaller model for faster training
        sparsity=0.1,
        use_multichannel=False  # Simpler single-channel approach
    )
    
    # Train the model
    try:
        results = trainer.train(
            X_train, y_train, X_test, y_test,
            epochs=epochs,
            batch_size=8,  # Small batch size
            validation_split=0.1
        )
        
        final_mse = results.get('final_test_mse', 0.0)
        final_mae = results.get('final_test_mae', 0.0)
        
        print(f"\nTraining completed!")
        print(f"Final Test MSE: {final_mse:.6f}")
        print(f"Final Test MAE: {final_mae:.6f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Generate predictions and show examples
    print(f"\n" + "=" * 80)
    print("PREDICTION EXAMPLES")
    print("=" * 80)
    
    # Select random test examples
    np.random.seed(42)
    example_indices = np.random.choice(len(X_test), min(num_examples, len(X_test)), replace=False)
    
    for i, idx in enumerate(example_indices):
        print(f"\n" + "─" * 70)
        print(f"EXAMPLE {i+1}/{len(example_indices)}")
        print("─" * 70)
        
        input_sequence = X_test[idx]  # Shape: (window_size, embedding_dim)
        target_output = y_test[idx]   # Shape: (embedding_dim,)
        
        # Get model prediction
        try:
            predicted_output = trainer.predict(input_sequence.reshape(1, window_size, embedding_dim))[0]
        except Exception as e:
            print(f"Prediction error: {e}")
            predicted_output = np.zeros_like(target_output)
        
        # Format as dialogue
        print("\nDialogue History:")
        dialogue_turns = format_sequence_dialogue(input_sequence, tokenizer, window_size)
        for turn in dialogue_turns:
            print(f"  {turn}")
        
        print(f"\nExpected Next Response:")
        target_text = decode_embedding_to_text(target_output, tokenizer, top_k=3)
        print(f"  {target_text}")
        
        print(f"\nModel Predicts:")
        predicted_text = decode_embedding_to_text(predicted_output, tokenizer, top_k=3)
        print(f"  {predicted_text}")
        
        # Calculate prediction quality
        target_norm = target_output / (np.linalg.norm(target_output) + 1e-8)
        pred_norm = predicted_output / (np.linalg.norm(predicted_output) + 1e-8)
        similarity = np.dot(target_norm, pred_norm)
        mse = np.mean((target_output - predicted_output) ** 2)
        
        print(f"\nPrediction Quality:")
        print(f"  Cosine Similarity: {similarity:.4f}")
        print(f"  MSE: {mse:.6f}")
        
        # Qualitative assessment
        if similarity > 0.7:
            quality = "Excellent - Very close match"
        elif similarity > 0.5:
            quality = "Good - Reasonable similarity"
        elif similarity > 0.3:
            quality = "Fair - Some similarity"
        else:
            quality = "Poor - Low similarity"
        
        print(f"  Assessment: {quality}")
        
        # Show embedding comparison
        print(f"\nEmbedding Analysis:")
        print(f"  Target norm: {np.linalg.norm(target_output):.3f}")
        print(f"  Prediction norm: {np.linalg.norm(predicted_output):.3f}")
        print(f"  Target mean: {np.mean(target_output):.3f}")
        print(f"  Prediction mean: {np.mean(predicted_output):.3f}")
    
    print(f"\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nWhat this shows:")
    print(f"• The LSM learns to map dialogue patterns to appropriate responses")
    print(f"• Higher similarity scores indicate better predictions")
    print(f"• The model processes dialogue history as temporal sequences")
    print(f"• Reservoir dynamics capture conversation flow and context")

def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Train and demonstrate LSM predictions with natural language examples"
    )
    
    parser.add_argument('--window-size', type=int, default=5,
                       help='Size of sequence window (smaller = faster training)')
    parser.add_argument('--embedding-dim', type=int, default=32,
                       help='Embedding dimension (smaller = faster training)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of prediction examples to show')
    
    args = parser.parse_args()
    
    print(f"Starting demonstration with:")
    print(f"  Window size: {args.window_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Examples: {args.num_examples}")
    print()
    
    train_and_demonstrate(
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        num_examples=args.num_examples
    )
    
    return 0

if __name__ == "__main__":
    exit(main())