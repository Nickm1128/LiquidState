#!/usr/bin/env python3
"""
LSMGenerator Demo - Demonstrating the convenience API for text generation.

This example shows how to use the LSMGenerator class for training LSM models
and generating text responses with a simple, scikit-learn-like interface.
"""

import sys
import os

# Add src to path for demo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_basic_usage():
    """Demonstrate basic LSMGenerator usage."""
    print("🚀 LSMGenerator Basic Usage Demo")
    print("=" * 50)
    
    try:
        from lsm.convenience import LSMGenerator
        
        # Create generator with default settings
        print("Creating LSMGenerator...")
        generator = LSMGenerator()
        print(f"✅ Generator created: {generator}")
        
        # Show parameters
        params = generator.get_params()
        print(f"📋 Default parameters: {list(params.keys())}")
        print(f"   - Window size: {params['window_size']}")
        print(f"   - Embedding dim: {params['embedding_dim']}")
        print(f"   - Reservoir type: {params['reservoir_type']}")
        
        # Prepare sample conversation data
        conversations = [
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "What's the weather like?",
            "It's sunny and warm today.",
            "That sounds nice!",
            "Yes, perfect for a walk.",
            "Can you help me with something?",
            "Of course! What do you need help with?",
            "I'm looking for a good restaurant.",
            "I'd recommend the Italian place downtown."
        ]
        
        print(f"\n📚 Sample data: {len(conversations)} conversation turns")
        
        # Note: We can't actually train without proper TensorFlow setup and data,
        # but we can show the interface
        print("\n🔧 Training interface (would train with proper setup):")
        print("generator.fit(conversations, epochs=10, batch_size=16)")
        print("# This would train the model on the conversation data")
        
        print("\n💬 Generation interface (after training):")
        print("response = generator.generate('Hello, how are you?')")
        print("# This would generate a response to the prompt")
        
        print("\n🎯 Batch generation interface:")
        print("responses = generator.batch_generate(['Hello', 'How are you?'])")
        print("# This would generate responses for multiple prompts")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_preset_usage():
    """Demonstrate preset configurations."""
    print("\n🎨 LSMGenerator Preset Configurations")
    print("=" * 50)
    
    try:
        from lsm.convenience import LSMGenerator, ConvenienceConfig
        
        # Show available presets
        presets = ConvenienceConfig.list_presets()
        print("📋 Available presets:")
        for name, description in presets.items():
            print(f"   - {name}: {description}")
        
        # Create generators with different presets
        print("\n🚀 Creating generators with presets:")
        
        # Fast preset for experimentation
        fast_gen = LSMGenerator.from_preset('fast')
        print(f"⚡ Fast generator: window_size={fast_gen.window_size}, embedding_dim={fast_gen.embedding_dim}")
        
        # Quality preset for production
        quality_gen = LSMGenerator.from_preset('quality')
        print(f"💎 Quality generator: window_size={quality_gen.window_size}, embedding_dim={quality_gen.embedding_dim}")
        
        # Text generation optimized preset
        text_gen = LSMGenerator.from_preset('text_generation')
        print(f"📝 Text generation generator: reservoir_type={text_gen.reservoir_type}")
        
        # Custom preset with overrides
        custom_gen = LSMGenerator.from_preset('balanced', temperature=0.8, max_length=100)
        print(f"🎛️  Custom generator: temperature={custom_gen.temperature}, max_length={custom_gen.max_length}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_data_formats():
    """Demonstrate different data input formats."""
    print("\n📊 LSMGenerator Data Format Support")
    print("=" * 50)
    
    try:
        from lsm.convenience import LSMGenerator
        
        generator = LSMGenerator()
        
        # Format 1: Simple string list
        print("📝 Format 1: Simple string list")
        simple_data = ["Hello", "Hi there", "How are you?", "I'm good"]
        processed = generator._preprocess_training_data(simple_data, None)
        print(f"   Input: {len(simple_data)} strings")
        print(f"   Processed: {len(processed['conversations'])} conversations")
        
        # Format 2: Single conversation string
        print("\n📝 Format 2: Single conversation string")
        single_string = """User: Hello there!
Assistant: Hi! How can I help you today?
User: I'm looking for information about Python.
Assistant: Python is a great programming language! What specifically would you like to know?"""
        processed = generator._preprocess_training_data(single_string, None)
        print(f"   Input: Single string with {len(single_string.split())} words")
        print(f"   Processed: {len(processed['conversations'])} conversation turns")
        
        # Format 3: Structured data with system messages
        print("\n📝 Format 3: Structured data with system messages")
        structured_data = [
            {
                "messages": ["Hello", "Hi! How can I help?"],
                "system": "Be friendly and helpful"
            },
            {
                "messages": ["What's 2+2?", "2+2 equals 4"],
                "system": "Be accurate with math"
            }
        ]
        processed = generator._preprocess_training_data(structured_data, None)
        print(f"   Input: {len(structured_data)} structured conversations")
        print(f"   Processed: {len(processed['conversations'])} messages")
        print(f"   System messages: {len(processed['system_messages']) if processed['system_messages'] else 0}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_sklearn_compatibility():
    """Demonstrate sklearn compatibility features."""
    print("\n🔬 LSMGenerator Sklearn Compatibility")
    print("=" * 50)
    
    try:
        from lsm.convenience import LSMGenerator
        
        generator = LSMGenerator()
        
        # Parameter management
        print("🔧 Parameter management:")
        params = generator.get_params()
        print(f"   get_params(): {len(params)} parameters")
        
        # Modify parameters
        generator.set_params(window_size=15, temperature=0.7)
        new_params = generator.get_params()
        print(f"   After set_params(): window_size={new_params['window_size']}, temperature={new_params['temperature']}")
        
        # Sklearn tags
        tags = generator.__sklearn_tags__()
        print(f"   sklearn tags: {list(tags.keys())[:5]}...")
        
        # Model representation
        print(f"   Model repr: {repr(generator)}")
        
        # Predict interface (sklearn-compatible)
        print("\n🎯 Sklearn-compatible prediction interface:")
        print("   predictions = generator.predict(['Hello', 'How are you?'])")
        print("   # This would return generated responses as a list")
        
        # Score interface (dummy for text generation)
        score = generator.score(["test"])
        print(f"   score(['test']): {score} (dummy score for compatibility)")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_error_handling():
    """Demonstrate error handling and validation."""
    print("\n⚠️  LSMGenerator Error Handling")
    print("=" * 50)
    
    try:
        from lsm.convenience import LSMGenerator, ConvenienceValidationError
        
        # Invalid parameters
        print("🚫 Testing parameter validation:")
        try:
            generator = LSMGenerator(window_size=-1)
        except ConvenienceValidationError as e:
            print(f"   ✅ Caught invalid window_size: {str(e).split(chr(10))[0]}")
        
        try:
            generator = LSMGenerator(reservoir_type='invalid')
        except ConvenienceValidationError as e:
            print(f"   ✅ Caught invalid reservoir_type: {str(e).split(chr(10))[0]}")
        
        # Invalid training data
        print("\n🚫 Testing data validation:")
        generator = LSMGenerator()
        
        try:
            generator.fit([])
        except ConvenienceValidationError as e:
            print(f"   ✅ Caught empty data: {str(e).split(chr(10))[0]}")
        
        try:
            generator.fit(123)
        except ConvenienceValidationError as e:
            print(f"   ✅ Caught invalid data type: {str(e).split(chr(10))[0]}")
        
        # Unfitted model operations
        print("\n🚫 Testing unfitted model validation:")
        from lsm.utils.lsm_exceptions import InvalidInputError
        
        try:
            generator.generate("Hello")
        except InvalidInputError as e:
            print(f"   ✅ Caught unfitted model: {str(e).split('(')[0]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Run all demos."""
    print("🎭 LSMGenerator Comprehensive Demo")
    print("=" * 60)
    print("This demo showcases the LSMGenerator convenience API features.")
    print("Note: Actual training requires proper TensorFlow setup and data.")
    print("=" * 60)
    
    demos = [
        demo_basic_usage,
        demo_preset_usage,
        demo_data_formats,
        demo_sklearn_compatibility,
        demo_error_handling
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ Demo {demo.__name__} failed: {e}")
        print()
    
    print("🎉 Demo completed!")
    print("\nNext steps:")
    print("1. Install all dependencies (TensorFlow, etc.)")
    print("2. Prepare your conversation data")
    print("3. Train your first LSM model:")
    print("   generator = LSMGenerator.from_preset('fast')")
    print("   generator.fit(your_conversations)")
    print("   response = generator.generate('Hello!')")

if __name__ == "__main__":
    main()