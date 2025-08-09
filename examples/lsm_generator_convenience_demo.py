#!/usr/bin/env python3
"""
Demonstration of the LSMGenerator convenience API.

This example shows how to use the simplified, scikit-learn-like interface
for training LSM models and generating text responses.
"""

import sys
import os
sys.path.append('.')

from src.lsm.convenience.generator import LSMGenerator

def main():
    print("🚀 LSMGenerator Convenience API Demo")
    print("=" * 50)
    
    # Create sample conversation data
    print("\n📝 Creating sample conversation data...")
    conversations = [
        "Hello there!",
        "Hi, how are you doing today?",
        "I'm doing great, thanks for asking!",
        "That's wonderful to hear.",
        "How has your day been?",
        "It's been quite productive, thank you.",
        "I'm glad to hear that.",
        "What are your plans for the weekend?",
        "I'm thinking of going hiking.",
        "That sounds like a great idea!",
        "Would you like to join me?",
        "I'd love to, but I have other commitments.",
        "No worries, maybe next time!",
        "Definitely, let's plan something soon.",
        "Sounds perfect!",
        "Have a great rest of your day!",
        "You too, take care!",
        "Thanks, goodbye!",
        "Goodbye!"
    ]
    print(f"✅ Created {len(conversations)} conversation examples")
    
    # Create LSMGenerator with fast preset for demo
    print("\n🤖 Creating LSMGenerator with 'fast' preset...")
    generator = LSMGenerator.from_preset('fast')
    print(f"✅ Generator created with parameters:")
    params = generator.get_params()
    for key, value in list(params.items())[:8]:  # Show first 8 parameters
        print(f"   {key}: {value}")
    print("   ...")
    
    # Train the model
    print("\n🎯 Training the LSM model...")
    print("This may take a few minutes...")
    
    try:
        generator.fit(
            X=conversations,
            epochs=3,  # Small number for demo
            batch_size=8,
            validation_split=0.2,
            verbose=True
        )
        print("✅ Training completed successfully!")
        
        # Display training metadata
        metadata = generator._training_metadata
        print(f"   Training time: {metadata['training_time']:.2f} seconds")
        print(f"   Data size: {metadata['data_size']} conversations")
        print(f"   Epochs: {metadata['epochs']}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("This is expected in some environments due to model complexity.")
        return
    
    # Test text generation
    print("\n💬 Testing text generation...")
    test_prompts = [
        "Hello!",
        "How are you?",
        "What are your plans?",
        "Have a great day!"
    ]
    
    for prompt in test_prompts:
        try:
            response = generator.generate(prompt, max_length=20)
            print(f"   Prompt: '{prompt}'")
            print(f"   Response: '{response}'")
            print()
        except Exception as e:
            print(f"   Prompt: '{prompt}' -> Generation failed: {e}")
    
    # Test batch generation
    print("\n📦 Testing batch generation...")
    try:
        batch_responses = generator.batch_generate(
            prompts=["Hi there!", "How's it going?"],
            max_length=15
        )
        for i, (prompt, response) in enumerate(zip(["Hi there!", "How's it going?"], batch_responses)):
            print(f"   Batch {i+1}: '{prompt}' -> '{response}'")
    except Exception as e:
        print(f"   Batch generation failed: {e}")
    
    # Test sklearn compatibility
    print("\n🔬 Testing sklearn compatibility...")
    try:
        # Test predict method
        predictions = generator.predict(["Hello", "Goodbye"])
        print(f"   Predictions: {len(predictions)} responses generated")
        
        # Test score method
        score = generator.score(["Hello", "Goodbye"])
        print(f"   Score: {score}")
        
        # Test parameter manipulation
        original_temp = generator.temperature
        generator.set_params(temperature=0.8)
        new_temp = generator.get_params()['temperature']
        print(f"   Temperature changed: {original_temp} -> {new_temp}")
        
    except Exception as e:
        print(f"   Sklearn compatibility test failed: {e}")
    
    # Show model information
    print("\n📊 Model Information:")
    print(f"   Model type: {type(generator).__name__}")
    print(f"   Is fitted: {generator._is_fitted}")
    print(f"   Reservoir type: {generator.reservoir_type}")
    print(f"   Window size: {generator.window_size}")
    print(f"   Embedding dim: {generator.embedding_dim}")
    
    print("\n🎉 Demo completed!")
    print("\nThe LSMGenerator provides a simple, sklearn-like interface for:")
    print("• Training LSM models on conversation data")
    print("• Generating text responses")
    print("• Batch processing")
    print("• Integration with sklearn pipelines")
    print("• Model persistence (save/load)")

if __name__ == "__main__":
    main()