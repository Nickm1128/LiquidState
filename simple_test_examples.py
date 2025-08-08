#!/usr/bin/env python3
"""
Simple test script that shows examples of model predictions in natural language.
Works with the current LSM setup by using the inference module.
"""

import os
import sys
import argparse
import numpy as np
from typing import List

# Add current directory to path for imports
sys.path.append('.')

from src.lsm.inference import LSMInference
from data_loader import load_data

def decode_embedding_to_text(embedding: np.ndarray, tokenizer, top_k: int = 3) -> str:
    """
    Convert an embedding back to the most likely text representation.
    """
    try:
        # Get vocabulary information from tokenizer
        vocab_embeddings = tokenizer.vocabulary_embeddings  
        vocab_texts = tokenizer.vocabulary_texts
        
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

def format_sequence_as_dialogue(sequence: np.ndarray, tokenizer, window_size: int) -> List[str]:
    """
    Convert a sequence of embeddings to dialogue turns.
    """
    dialogue_turns = []
    for t in range(window_size):
        embedding = sequence[t]
        text = decode_embedding_to_text(embedding, tokenizer, top_k=1)
        # Extract just the most likely match (remove similarity score)
        text = text.split(" (sim:")[0] if " (sim:" in text else text
        dialogue_turns.append(f"Turn {t+1}: {text}")
    
    return dialogue_turns

def show_prediction_examples(model_path: str, num_examples: int = 3, 
                           window_size: int = 8, embedding_dim: int = 64) -> None:
    """Show examples of model predictions."""
    
    print("=" * 80)
    print("LSM MODEL PREDICTION EXAMPLES")
    print("=" * 80)
    
    # Initialize inference engine
    print(f"Loading model from {model_path}...")
    try:
        inference_engine = LSMInference(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model path exists and contains trained models.")
        return
    
    # Load test data
    print("Loading test data...")
    try:
        _, _, X_test, y_test, tokenizer = load_data(
            window_size=window_size,
            test_size=0.2,
            embedding_dim=embedding_dim
        )
        print(f"Test data loaded: {X_test.shape[0]} sequences")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Show vocabulary info
    vocab_size = len(tokenizer.vocabulary_texts)
    print(f"\nVocabulary size: {vocab_size}")
    print("Sample vocabulary items:")
    for i, text in enumerate(tokenizer.vocabulary_texts[:5]):
        print(f"  {i+1}: '{text}'")
    
    # Get some predictions
    print("\nGenerating predictions...")
    
    # Select random examples
    np.random.seed(42)  # For reproducible examples
    example_indices = np.random.choice(len(X_test), min(num_examples, len(X_test)), replace=False)
    
    for i, idx in enumerate(example_indices):
        print(f"\n" + "─" * 80)
        print(f"EXAMPLE {i+1}/{len(example_indices)} (Test Sample #{idx})")
        print("─" * 80)
        
        # Get input sequence and target
        input_sequence = X_test[idx]  # Shape: (window_size, embedding_dim)
        target_output = y_test[idx]   # Shape: (embedding_dim,)
        
        # Get model prediction
        try:
            predicted_output = inference_engine.predict_next_token(input_sequence)
            print("✓ Prediction generated successfully")
        except Exception as e:
            print(f"✗ Prediction failed: {e}")
            predicted_output = np.zeros_like(target_output)
        
        print("\n🗣️ DIALOGUE HISTORY:")
        dialogue_turns = format_sequence_as_dialogue(input_sequence, tokenizer, window_size)
        for turn in dialogue_turns:
            print(f"   {turn}")
        
        print("\n🎯 EXPECTED NEXT:")
        target_text = decode_embedding_to_text(target_output, tokenizer, top_k=3)
        print(f"   {target_text}")
        
        print("\n🤖 MODEL PREDICTS:")
        predicted_text = decode_embedding_to_text(predicted_output, tokenizer, top_k=3)
        print(f"   {predicted_text}")
        
        # Calculate prediction quality
        target_norm = target_output / (np.linalg.norm(target_output) + 1e-8)
        pred_norm = predicted_output / (np.linalg.norm(predicted_output) + 1e-8)
        similarity = np.dot(target_norm, pred_norm)
        mse = np.mean((target_output - predicted_output) ** 2)
        
        print(f"\n📊 PREDICTION QUALITY:")
        print(f"   Cosine Similarity: {similarity:.4f}")
        print(f"   Mean Squared Error: {mse:.6f}")
        
        # Simple interpretation
        if similarity > 0.8:
            quality = "Excellent ✨"
        elif similarity > 0.6:
            quality = "Good ✅"
        elif similarity > 0.4:
            quality = "Fair ⚡"
        else:
            quality = "Poor ❌"
        print(f"   Overall Quality: {quality}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Show LSM model prediction examples in natural language"
    )
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of examples to show')
    parser.add_argument('--window-size', type=int, default=8,
                       help='Window size used during training')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension used during training')
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        print("Available model directories:")
        for item in os.listdir('.'):
            if item.startswith('models_'):
                print(f"  - {item}")
        return 1
    
    # Run examples
    show_prediction_examples(
        model_path=args.model_path,
        num_examples=args.num_examples,
        window_size=args.window_size,
        embedding_dim=args.embedding_dim
    )
    
    print("\n" + "=" * 80)
    print("EXAMPLES COMPLETE")
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit(main())