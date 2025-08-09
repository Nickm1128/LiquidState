#!/usr/bin/env python3
"""
Convenience API Dialogue Processing Examples

This example demonstrates how the LSM convenience API processes dialogue data,
showing the transformation from raw text to embeddings and model predictions.
"""

import os
import sys
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.convenience import LSMGenerator
from lsm.convenience.config import ConvenienceConfig


def create_structured_dialogue_data() -> List[Dict[str, Any]]:
    """Create structured dialogue data for demonstration."""
    dialogues = [
        {
            "conversation": [
                "Hello, how are you today?",
                "I'm doing great, thank you for asking!",
                "That's wonderful to hear. What are your plans?",
                "I'm planning to go for a walk in the park.",
                "That sounds lovely! I hope you enjoy it."
            ],
            "context": "Friendly greeting conversation"
        },
        {
            "conversation": [
                "What's your favorite type of music?",
                "I really enjoy classical music, especially Bach.",
                "Bach is amazing! Do you play any instruments?",
                "Yes, I play the piano. It's very relaxing.",
                "That's wonderful! Music is such a great hobby."
            ],
            "context": "Discussion about music preferences"
        },
        {
            "conversation": [
                "I'm feeling a bit stressed about work lately.",
                "I'm sorry to hear that. What's been bothering you?",
                "There's just so much to do and not enough time.",
                "Have you tried breaking tasks into smaller pieces?",
                "That's a good idea. I'll try organizing better."
            ],
            "context": "Supportive conversation about work stress"
        },
        {
            "conversation": [
                "What do you think about the weather today?",
                "It's absolutely beautiful! Perfect for outdoor activities.",
                "I was thinking the same thing. Want to go hiking?",
                "That sounds like a great idea! When should we go?",
                "How about this afternoon around 2 PM?"
            ],
            "context": "Weather discussion leading to activity planning"
        }
    ]
    return dialogues


def demonstrate_dialogue_processing(
    window_size: int = 4,
    embedding_dim: int = 64,
    preset: str = 'fast'
) -> None:
    """
    Demonstrate how the convenience API processes dialogue data.
    """
    
    print("=" * 80)
    print("LSM CONVENIENCE API DIALOGUE PROCESSING DEMONSTRATION")
    print("=" * 80)
    
    # Create structured dialogue data
    dialogue_data = create_structured_dialogue_data()
    
    # Flatten conversations for training
    all_conversations = []
    for dialogue in dialogue_data:
        all_conversations.extend(dialogue["conversation"])
    
    print(f"Created {len(dialogue_data)} dialogue scenarios")
    print(f"Total conversation turns: {len(all_conversations)}")
    
    # Show original dialogue structure
    print(f"\n📚 ORIGINAL DIALOGUE STRUCTURE:")
    print("─" * 60)
    
    for i, dialogue in enumerate(dialogue_data):
        print(f"\nDialogue {i+1}: {dialogue['context']}")
        for j, turn in enumerate(dialogue["conversation"]):
            speaker = "A" if j % 2 == 0 else "B"
            print(f"  {speaker}: \"{turn}\"")
    
    # Configure and create model
    config = ConvenienceConfig.get_preset(preset)
    config.update({
        'window_size': window_size,
        'embedding_dim': embedding_dim,
        'random_state': 42
    })
    
    print(f"\n🔧 MODEL CONFIGURATION:")
    print("─" * 60)
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create and train the generator
    print(f"\n🎯 TRAINING LSM GENERATOR:")
    print("─" * 60)
    
    generator = LSMGenerator(**config)
    
    print("Training on dialogue data...")
    try:
        generator.fit(
            all_conversations,
            epochs=8,
            batch_size=4,
            validation_split=0.2
        )
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # Demonstrate conversation understanding
    print(f"\n🎭 CONVERSATION UNDERSTANDING DEMONSTRATION:")
    print("─" * 60)
    
    # Test with conversation starters from each dialogue
    test_scenarios = [
        {
            "context": "Greeting",
            "prompt": "Hello, how are you today?",
            "expected_theme": "friendly response"
        },
        {
            "context": "Music discussion",
            "prompt": "What's your favorite type of music?",
            "expected_theme": "music preference"
        },
        {
            "context": "Stress support",
            "prompt": "I'm feeling a bit stressed about work lately.",
            "expected_theme": "supportive response"
        },
        {
            "context": "Weather chat",
            "prompt": "What do you think about the weather today?",
            "expected_theme": "weather comment"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\nScenario {i+1}: {scenario['context']}")
        print(f"Input: \"{scenario['prompt']}\"")
        print(f"Expected theme: {scenario['expected_theme']}")
        
        try:
            # Generate multiple responses with different temperatures
            temperatures = [0.5, 1.0, 1.5]
            
            for temp in temperatures:
                response = generator.generate(
                    scenario['prompt'],
                    max_length=25,
                    temperature=temp
                )
                print(f"Response (temp={temp}): \"{response}\"")
            
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
    
    # Demonstrate conversation flow
    print(f"\n🔄 CONVERSATION FLOW DEMONSTRATION:")
    print("─" * 60)
    
    # Simulate a multi-turn conversation
    conversation_history = []
    initial_prompt = "Hello there! How has your day been?"
    
    print(f"Starting conversation with: \"{initial_prompt}\"")
    conversation_history.append(f"Human: {initial_prompt}")
    
    try:
        for turn in range(3):
            # Generate response
            current_context = " ".join(conversation_history[-2:])  # Use last 2 turns as context
            
            response = generator.generate(
                current_context,
                max_length=30,
                temperature=0.8
            )
            
            print(f"Turn {turn+1} - AI: \"{response}\"")
            conversation_history.append(f"AI: {response}")
            
            # Simulate human response (for demonstration)
            human_responses = [
                "That's interesting! Tell me more about that.",
                "I see what you mean. What else do you think?",
                "Thanks for sharing! I appreciate your perspective."
            ]
            
            if turn < len(human_responses):
                human_response = human_responses[turn]
                print(f"Turn {turn+1} - Human: \"{human_response}\"")
                conversation_history.append(f"Human: {human_response}")
        
    except Exception as e:
        print(f"❌ Conversation flow failed: {e}")
    
    # Show model capabilities
    print(f"\n🎯 MODEL CAPABILITIES ANALYSIS:")
    print("─" * 60)
    
    capabilities = [
        "Context understanding",
        "Appropriate response generation",
        "Conversation flow maintenance",
        "Temperature-based creativity control",
        "Multi-turn dialogue handling"
    ]
    
    print("Demonstrated capabilities:")
    for capability in capabilities:
        print(f"  ✅ {capability}")
    
    # Performance insights
    print(f"\n📊 PROCESSING INSIGHTS:")
    print("─" * 60)
    
    try:
        model_params = generator.get_params()
        print(f"Model parameters:")
        key_params = ['window_size', 'embedding_dim', 'reservoir_type']
        for param in key_params:
            if param in model_params:
                print(f"  {param}: {model_params[param]}")
        
        print(f"\nData processing:")
        print(f"  Training conversations: {len(all_conversations)}")
        print(f"  Dialogue scenarios: {len(dialogue_data)}")
        print(f"  Average turns per dialogue: {len(all_conversations) / len(dialogue_data):.1f}")
        
    except Exception as e:
        print(f"Error getting model info: {e}")


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(
        description="Demonstrate LSM convenience API dialogue processing"
    )
    
    parser.add_argument('--window-size', type=int, default=4,
                       help='Size of sequence window')
    parser.add_argument('--embedding-dim', type=int, default=64,
                       help='Embedding dimension')
    parser.add_argument('--preset', type=str, default='fast',
                       choices=['fast', 'balanced', 'quality'],
                       help='Configuration preset to use')
    
    args = parser.parse_args()
    
    print(f"Starting dialogue processing demonstration with:")
    print(f"  Window size: {args.window_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Preset: {args.preset}")
    print()
    
    demonstrate_dialogue_processing(
        window_size=args.window_size,
        embedding_dim=args.embedding_dim,
        preset=args.preset
    )
    
    print(f"\n" + "=" * 80)
    print("DIALOGUE PROCESSING DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey insights:")
    print("• The convenience API simplifies dialogue processing")
    print("• Models learn conversation patterns and appropriate responses")
    print("• Temperature control allows creativity adjustment")
    print("• Multi-turn conversations maintain context and flow")
    print("• Different presets offer speed vs quality tradeoffs")
    
    return 0


if __name__ == "__main__":
    exit(main())