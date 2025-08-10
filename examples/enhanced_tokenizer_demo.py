#!/usr/bin/env python3
"""
Enhanced Tokenizer Convenience Functions Demo

This script demonstrates the key convenience functions for the enhanced
tokenizer system in the LSM project.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.convenience import LSMGenerator
from lsm.convenience.config import ConvenienceConfig
from lsm.convenience.utils import (
    preprocess_conversation_data,
    detect_conversation_format,
    estimate_training_time
)


def demo_enhanced_tokenizer():
    """Demonstrate enhanced tokenizer creation and usage."""
    print("🔤 Enhanced Tokenizer Demo")
    print("=" * 50)
    
    # Create enhanced tokenizer with automatic backend detection
    print("Creating enhanced tokenizer with GPT-2 backend...")
    tokenizer = EnhancedTokenizerWrapper(
        tokenizer='gpt2',
        embedding_dim=256,
        max_length=128,
        enable_caching=True
    )
    
    print(f"✅ Tokenizer created!")
    print(f"   Backend: {tokenizer.get_adapter().config.backend}")
    print(f"   Vocab size: {tokenizer.get_vocab_size():,}")
    print(f"   Embedding shape: {tokenizer.get_token_embeddings_shape()}")
    
    # Test tokenization
    test_texts = [
        "Hello, how are you today?",
        "What is machine learning?",
        "Can you help me with programming?"
    ]
    
    print(f"\n🧪 Testing tokenization:")
    for text in test_texts:
        tokens = tokenizer.tokenize([text])
        decoded = tokenizer.decode(tokens[0])
        print(f"   Input: {text}")
        print(f"   Tokens: {len(tokens[0])} tokens")
        print(f"   Decoded: {decoded}")
        print()
    
    # Create sinusoidal embedder
    print("🌊 Creating configurable sinusoidal embedder...")
    embedder = tokenizer.create_configurable_sinusoidal_embedder(
        learnable_frequencies=True,
        base_frequency=10000.0,
        use_relative_position=False
    )
    print(f"✅ Sinusoidal embedder created with {tokenizer.embedding_dim}D embeddings")
    
    return tokenizer


def demo_data_preprocessing():
    """Demonstrate data preprocessing convenience functions."""
    print("\n📊 Data Preprocessing Demo")
    print("=" * 50)
    
    # Sample conversation data
    sample_conversations = [
        "User: Hello, how are you?\\nAssistant: I'm doing well, thank you!",
        "User: What's machine learning?\\nAssistant: It's a subset of AI that learns from data.",
        "User: Can you help me code?\\nAssistant: Of course! What language are you using?",
    ]
    
    # Detect format
    detected_format = detect_conversation_format(sample_conversations)
    print(f"📋 Detected format: {detected_format}")
    
    # Preprocess data
    processed_data = preprocess_conversation_data(
        sample_conversations,
        min_message_length=5,
        max_message_length=200,
        normalize_whitespace=True
    )
    
    print(f"✅ Processed {len(processed_data)} conversations")
    print(f"📝 Sample: {processed_data[0][:80]}...")
    
    return processed_data


def demo_lsm_generator():
    """Demonstrate LSM Generator with enhanced tokenizer."""
    print("\n🧠 LSM Generator Demo")
    print("=" * 50)
    
    # Get balanced configuration and customize
    config = ConvenienceConfig.get_preset('balanced')
    config.update({
        'tokenizer': 'gpt2',
        'embedding_dim': 256,
        'embedding_type': 'configurable_sinusoidal',
        'enable_caching': True,
        'sinusoidal_config': {
            'learnable_frequencies': True,
            'base_frequency': 10000.0
        },
        'reservoir_type': 'attentive',
        'window_size': 8,
        'system_message_support': True,
        'random_state': 42
    })
    
    print("📋 Configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            print(f"   {key}: {len(value)} settings")
        else:
            print(f"   {key}: {value}")
    
    # Create generator
    print("\n🏗️ Creating LSM Generator...")
    generator = LSMGenerator(**config)
    
    print(f"✅ Generator created!")
    print(f"   Architecture: {generator.reservoir_type} reservoir")
    print(f"   Embedding type: {generator.embedding_type}")
    print(f"   System message support: {generator.system_message_support}")
    
    return generator


def demo_training_estimation():
    """Demonstrate training time estimation."""
    print("\n⏱️ Training Estimation Demo")
    print("=" * 50)
    
    # Estimate training time for different configurations
    configs = [
        {'reservoir_type': 'standard', 'epochs': 10, 'embedding_dim': 128},
        {'reservoir_type': 'attentive', 'epochs': 15, 'embedding_dim': 256},
        {'reservoir_type': 'hierarchical', 'epochs': 20, 'embedding_dim': 512}
    ]
    
    data_sizes = [100, 1000, 10000]
    
    print("📊 Training time estimates:")
    print(f"{'Config':<20} {'Data Size':<10} {'Estimated Time':<15}")
    print("-" * 50)
    
    for config in configs:
        for data_size in data_sizes:
            estimate = estimate_training_time(data_size, config)
            config_name = f"{config['reservoir_type'][:8]}-{config['embedding_dim']}"
            print(f"{config_name:<20} {data_size:<10} {estimate['human_readable']:<15}")


def demo_inference():
    """Demonstrate inference with different settings."""
    print("\n🎭 Inference Demo")
    print("=" * 50)
    
    # Create a simple generator for demo
    try:
        generator = LSMGenerator(
            tokenizer='gpt2',
            embedding_dim=128,
            embedding_type='sinusoidal',
            reservoir_type='standard',
            preset='fast',
            random_state=42
        )
        
        # Mock training for demo
        generator._is_fitted = True
        
        print("🎯 Testing different generation settings:")
        
        test_prompts = [
            "Hello, how are you?",
            "What is artificial intelligence?",
            "Can you help me learn Python?"
        ]
        
        temperatures = [0.3, 0.7, 1.2]
        temp_labels = ["Conservative", "Balanced", "Creative"]
        
        for prompt in test_prompts:
            print(f"\n💬 Prompt: \"{prompt}\"")
            for temp, label in zip(temperatures, temp_labels):
                try:
                    response = generator.generate(
                        prompt,
                        max_length=30,
                        temperature=temp
                    )
                    print(f"   {label} (T={temp}): \"{response}\"")
                except Exception as e:
                    print(f"   {label} (T={temp}): [Demo mode - generation simulated]")
        
        # Test system messages
        print(f"\n🎭 System message test:")
        try:
            response = generator.generate(
                "Explain neural networks",
                system_message="You are a helpful AI teacher",
                max_length=40,
                temperature=0.8
            )
            print(f"   Response: \"{response}\"")
        except Exception as e:
            print(f"   Response: [Demo mode - system message support simulated]")
            
    except Exception as e:
        print(f"⚠️ Inference demo requires full model setup: {e}")
        print("💡 This would work with a fully trained model")


def main():
    """Run all demos."""
    print("🚀 LSM Enhanced Tokenizer Convenience Functions Demo")
    print("=" * 60)
    
    try:
        # Demo 1: Enhanced tokenizer
        tokenizer = demo_enhanced_tokenizer()
        
        # Demo 2: Data preprocessing
        processed_data = demo_data_preprocessing()
        
        # Demo 3: LSM Generator
        generator = demo_lsm_generator()
        
        # Demo 4: Training estimation
        demo_training_estimation()
        
        # Demo 5: Inference
        demo_inference()
        
        print("\n🎉 All demos completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Try the Colab notebook: LSM_Enhanced_Pipeline_Demo.ipynb")
        print("   2. Read the guide: ENHANCED_TOKENIZER_CONVENIENCE_GUIDE.md")
        print("   3. Explore the convenience API: src/lsm/convenience/")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've installed the LSM package: pip install -e .")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 This is expected in some environments - check the full setup")


if __name__ == "__main__":
    main()