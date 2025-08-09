#!/usr/bin/env python3
"""
Convenience API Prediction Demonstration

This example shows how to use the LSM convenience API to train a model
and demonstrate predictions with natural language examples, similar to
the legacy demonstrate_predictions.py but using the new convenience API.
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.convenience import LSMGenerator
from lsm.convenience.config import ConvenienceConfig


def create_sample_conversations() -> List[str]:
    """Create sample conversation data for demonstration."""
    conversations = [
        "Hello there! How are you doing today?",
        "I'm doing well, thank you for asking. How about you?",
        "That's great to hear! I'm having a wonderful day.",
        "What are your plans for the weekend?",
        "I'm thinking of going hiking. Do you enjoy outdoor activities?",
        "Yes, I love being in nature. Hiking sounds like a great idea!",
        "Would you like to join me sometime?",
        "That sounds like fun! I'd love to go hiking with you.",
        "Perfect! Let's plan something for next Saturday.",
        "Saturday works great for me. What time should we meet?",
        "How about 8 AM at the trailhead?",
        "8 AM sounds perfect. I'll see you there!",
        "Great! Don't forget to bring water and snacks.",
        "Good reminder! I'll pack everything we need.",
        "Looking forward to our hiking adventure!",
        "Me too! It's going to be a fantastic day.",
        "What's your favorite type of music?",
        "I really enjoy classical music. How about you?",
        "I'm more into rock and pop music myself.",
        "That's cool! Music taste is so personal.",
        "Do you play any instruments?",
        "I play the piano. It's very relaxing.",
        "That's amazing! I've always wanted to learn piano.",
        "You should give it a try. It's never too late to start!",
        "Maybe I will. Do you have any beginner tips?",
        "Start with simple songs and practice regularly.",
        "Thanks for the advice! I'll look into piano lessons.",
        "You're welcome! I'm sure you'll do great.",
        "What's your favorite book?",
        "I love reading science fiction novels.",
        "Any particular author you'd recommend?",
        "Isaac Asimov is fantastic. His robot series is amazing.",
        "I'll definitely check that out. Thanks for the suggestion!",
        "You're welcome! I think you'll really enjoy his work."
    ]
    return conversations


def demonstrate_convenience_training_and_prediction(
    window_size: int = 5,
    embedding_dim: int = 64,
    epochs: int = 10,
    num_examples: int = 3,
    preset: str = 'fast'
) -> None:
    """
    Demonstrate training and prediction using the convenience API.
    """
    
    print("=" * 80)
    print("LSM CONVENIENCE API PREDICTION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    conversations = create_sample_conversations()
    print(f"Created {len(conversations)} sample conversations")
    
    # Get preset configuration
    config = ConvenienceConfig.get_preset(preset)
    config.update({
        'window_size': window_size,
        'embedding_dim': embedding_dim,
        'random_state': 42
    })
    
    print(f"\nUsing configuration preset '{preset}':")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create and train the generator
    print(f"\nCreating LSM Generator with convenience API...")
    generator = LSMGenerator(**config)
    
    print(f"Training model on conversation data...")
    print(f"  Epochs: {epochs}")
    print(f"  Training samples: {len(conversations)}")
    
    try:
        # Train the model
        generator.fit(
            conversations,
            epochs=epochs,
            batch_size=8,
            validation_split=0.2
        )
        
        print(f"✓ Training completed successfully!")
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    # Demonstrate predictions
    print(f"\n" + "=" * 80)
    print("PREDICTION EXAMPLES")
    print("=" * 80)
    
    # Test prompts for generation
    test_prompts = [
        "Hello there!",
        "What are your plans for the weekend?",
        "Do you play any instruments?",
        "What's your favorite book?",
        "I love being in nature."
    ]
    
    for i, prompt in enumerate(test_prompts[:num_examples]):
        print(f"\n" + "─" * 70)
        print(f"EXAMPLE {i+1}/{num_examples}")
        print("─" * 70)
        
        print(f"\nInput Prompt:")
        print(f"  \"{prompt}\"")
        
        try:
            # Generate response
            response = generator.generate(
                prompt,
                max_length=30,
                temperature=0.8
            )
            
            print(f"\nGenerated Response:")
            print(f"  \"{response}\"")
            
            # Try with different temperature
            creative_response = generator.generate(
                prompt,
                max_length=30,
                temperature=1.2
            )
            
            print(f"\nMore Creative Response (higher temperature):")
            print(f"  \"{creative_response}\"")
            
        except Exception as e:
            print(f"  Error generating response: {e}")
    
    # Demonstrate batch generation
    print(f"\n" + "─" * 70)
    print("BATCH GENERATION EXAMPLE")
    print("─" * 70)
    
    batch_prompts = [
        "How are you today?",
        "What's the weather like?",
        "Tell me something interesting."
    ]
    
    print(f"\nBatch input prompts:")
    for i, prompt in enumerate(batch_prompts):
        print(f"  {i+1}: \"{prompt}\"")
    
    try:
        batch_responses = generator.batch_generate(
            batch_prompts,
            max_length=25,
            temperature=1.0
        )
        
        print(f"\nBatch responses:")
        for i, response in enumerate(batch_responses):
            print(f"  {i+1}: \"{response}\"")
            
    except Exception as e:
        print(f"  Error in batch generation: {e}")
    
    # Show model information
    print(f"\n" + "=" * 80)
    print("MODEL INFORMATION")
    print("=" * 80)
    
    try:
        params = generator.get_params()
        print(f"\nModel parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
            
    except Exception as e:
        print(f"Error getting model parameters: {e}")
    
    print(f"\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nKey features demonstrated:")
    print(f"• Simple model creation with preset configurations")
    print(f"• Easy training with fit() method")
    print(f"• Text generation with generate() method")
    print(f"• Batch processing capabilities")
    print(f"• Temperature control for creativity")
    print(f"• sklearn-compatible parameter interface")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate LSM convenience API for text generation"
    )
    
    parser.add_argument('--window-size', type=int, default=5,
                       help='Size of sequence window')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of prediction examples to show')
    parser.add_argument('--preset', type=str, default='fast',
                       choices=['fast', 'balanced', 'quality'],
                       help='Configuration preset to use')
    
    args = parser.parse_args()
    
    print(f"Starting convenience API demonstration with:")
    print(f"  Window size: {args.window_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Examples: {args.num_examples}")
    print(f"  Preset: {args.preset}")
    print()
    
    demonstrate_convenience_training_and_prediction(
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        num_examples=args.num_examples,
        preset=args.preset
    )
    
    return 0


if __name__ == "__main__":
    exit(main())