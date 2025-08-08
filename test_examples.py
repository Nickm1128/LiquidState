#!/usr/bin/env python3
"""
Test script that shows examples of model predictions in natural language.
Loads a trained LSM model and displays input sequences, target outputs, and predictions.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Add src directory to path for imports
sys.path.append('src')

from src.lsm.data.data_loader import load_data, DialogueTokenizer
from src.lsm.training.train import LSMTrainer
from lsm_logging import get_logger

logger = get_logger(__name__)

def load_trained_model(model_path: str, window_size: int, embedding_dim: int) -> LSMTrainer:
    """Load a trained LSM model from the specified path."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    print(f"Loading trained model from {model_path}")
    
    # Initialize trainer
    trainer = LSMTrainer(
        window_size=window_size,
        embedding_dim=embedding_dim
    )
    
    # Build models first (needed for custom layers to be registered)
    trainer.build_models()
    
    # Then load the weights manually to avoid deserialization issues
    import tensorflow as tf
    from src.lsm.core.reservoir import SparseDense
    from src.lsm.core.cnn_model import ParametricSineActivation, SpatialAttentionBlock
    
    # Register custom layers to avoid deserialization errors
    tf.keras.utils.get_custom_objects()['SparseDense'] = SparseDense
    tf.keras.utils.get_custom_objects()['ParametricSineActivation'] = ParametricSineActivation
    tf.keras.utils.get_custom_objects()['SpatialAttentionBlock'] = SpatialAttentionBlock
    
    # Load models
    reservoir_path = os.path.join(model_path, "reservoir_model.keras")
    cnn_path = os.path.join(model_path, "cnn_model.keras")
    
    if os.path.exists(reservoir_path):
        trainer.reservoir = tf.keras.models.load_model(reservoir_path)
        print("Reservoir model loaded successfully")
    
    if os.path.exists(cnn_path):
        trainer.cnn_model = tf.keras.models.load_model(cnn_path)
        print("CNN model loaded successfully")
    
    return trainer

def decode_embedding_to_text(embedding: np.ndarray, tokenizer: DialogueTokenizer, top_k: int = 3) -> str:
    """
    Convert an embedding back to the most likely text representation.
    
    Args:
        embedding: The embedding vector to decode
        tokenizer: The trained tokenizer with vocabulary
        top_k: Number of top matches to show
        
    Returns:
        String representation of the most likely text
    """
    try:
        # Get vocabulary embeddings from tokenizer
        vocab_embeddings = tokenizer.vocabulary_embeddings  # Shape: (vocab_size, embedding_dim)
        vocab_texts = tokenizer.vocabulary_texts
        
        # Calculate cosine similarity between embedding and all vocabulary embeddings
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        vocab_norms = vocab_embeddings / (np.linalg.norm(vocab_embeddings, axis=1, keepdims=True) + 1e-8)
        
        similarities = np.dot(vocab_norms, embedding_norm)
        
        # Get top-k most similar vocabulary items
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            similarity = similarities[idx]
            text = vocab_texts[idx]
            results.append(f"{text} ({similarity:.3f})")
        
        return " | ".join(results)
        
    except Exception as e:
        return f"<decode_error: {str(e)}>"

def format_sequence_as_text(sequence: np.ndarray, tokenizer: DialogueTokenizer, window_size: int) -> List[str]:
    """
    Convert a sequence of embeddings to a list of text representations.
    
    Args:
        sequence: Input sequence of shape (window_size, embedding_dim)
        tokenizer: The trained tokenizer
        window_size: Size of the sequence window
        
    Returns:
        List of text representations for each timestep
    """
    texts = []
    for t in range(window_size):
        embedding = sequence[t]
        text = decode_embedding_to_text(embedding, tokenizer, top_k=1)
        # Extract just the first (most likely) match
        text = text.split(" (")[0]  # Remove similarity score
        texts.append(f"T{t+1}: {text}")
    
    return texts

def show_prediction_examples(trainer: LSMTrainer, X_test: np.ndarray, y_test: np.ndarray, 
                           tokenizer: DialogueTokenizer, num_examples: int = 5) -> None:
    """
    Show examples of model predictions compared to targets.
    
    Args:
        trainer: Trained LSM model
        X_test: Test input sequences
        y_test: Test target outputs  
        tokenizer: Trained tokenizer for decoding
        num_examples: Number of examples to show
    """
    print("=" * 80)
    print("MODEL PREDICTION EXAMPLES")
    print("=" * 80)
    
    # Get predictions for all test data
    print("Generating predictions...")
    y_pred = trainer.predict(X_test)
    
    # Calculate overall metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"\nOverall Test Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    
    # Show individual examples
    num_examples = min(num_examples, len(X_test))
    example_indices = np.random.choice(len(X_test), num_examples, replace=False)
    
    for i, idx in enumerate(example_indices):
        print(f"\n" + "─" * 80)
        print(f"EXAMPLE {i+1}/{num_examples} (Test Sample #{idx})")
        print("─" * 80)
        
        # Input sequence
        input_sequence = X_test[idx]  # Shape: (window_size, embedding_dim)
        target_output = y_test[idx]   # Shape: (embedding_dim,)
        predicted_output = y_pred[idx]  # Shape: (embedding_dim,)
        
        print("\n🔤 INPUT SEQUENCE (Dialogue History):")
        input_texts = format_sequence_as_text(input_sequence, tokenizer, trainer.window_size)
        for text in input_texts:
            print(f"  {text}")
        
        print("\n🎯 TARGET (Expected Next Token):")
        target_text = decode_embedding_to_text(target_output, tokenizer, top_k=3)
        print(f"  {target_text}")
        
        print("\n🤖 PREDICTED (Model Output):")
        predicted_text = decode_embedding_to_text(predicted_output, tokenizer, top_k=3)
        print(f"  {predicted_text}")
        
        # Calculate similarity between target and prediction
        target_norm = target_output / (np.linalg.norm(target_output) + 1e-8)
        pred_norm = predicted_output / (np.linalg.norm(predicted_output) + 1e-8)
        similarity = np.dot(target_norm, pred_norm)
        
        print(f"\n📊 PREDICTION QUALITY:")
        print(f"  Cosine Similarity: {similarity:.4f}")
        print(f"  MSE: {np.mean((target_output - predicted_output) ** 2):.6f}")
        print(f"  MAE: {np.mean(np.abs(target_output - predicted_output)):.6f}")

def show_vocabulary_info(tokenizer: DialogueTokenizer) -> None:
    """Display information about the tokenizer vocabulary."""
    print("=" * 80)
    print("TOKENIZER VOCABULARY")
    print("=" * 80)
    
    vocab_texts = tokenizer.vocabulary_texts
    vocab_embeddings = tokenizer.vocabulary_embeddings
    
    print(f"Vocabulary Size: {len(vocab_texts)}")
    print(f"Embedding Dimension: {vocab_embeddings.shape[1]}")
    
    print(f"\nVocabulary Items:")
    for i, text in enumerate(vocab_texts):
        embedding_norm = np.linalg.norm(vocab_embeddings[i])
        print(f"  {i+1:2d}: '{text}' (norm: {embedding_norm:.3f})")

def main():
    """Main function to run prediction examples."""
    parser = argparse.ArgumentParser(
        description="Show examples of LSM model predictions in natural language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Number of prediction examples to show')
    parser.add_argument('--window-size', type=int, default=10,
                       help='Window size used during training')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension used during training')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (should match training)')
    parser.add_argument('--show-vocab', action='store_true',
                       help='Show vocabulary information')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible examples')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    try:
        # Load the trained model
        trainer = load_trained_model(args.model_path, args.window_size, args.embedding_dim)
        
        # Load test data (matching the training configuration)
        print("Loading test data...")
        _, _, X_test, y_test, tokenizer = load_data(
            window_size=args.window_size,
            test_size=args.test_size,
            embedding_dim=args.embedding_dim
        )
        
        print(f"Test data loaded: {X_test.shape[0]} sequences")
        
        # Show vocabulary if requested
        if args.show_vocab:
            show_vocabulary_info(tokenizer)
        
        # Show prediction examples
        show_prediction_examples(
            trainer=trainer,
            X_test=X_test,
            y_test=y_test,
            tokenizer=tokenizer,
            num_examples=args.num_examples
        )
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        logger.exception("Error in prediction examples")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())