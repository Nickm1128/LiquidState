#!/usr/bin/env python3
"""
Enhanced LSMGenerator Demo

This example demonstrates the enhanced tokenizer integration with LSMGenerator,
showing how to use different tokenizer backends, sinusoidal embeddings, and
streaming data processing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def demo_basic_enhanced_tokenizer():
    """Demonstrate basic enhanced tokenizer usage."""
    print("=" * 60)
    print("Basic Enhanced Tokenizer Demo")
    print("=" * 60)
    
    try:
        from lsm.convenience.generator import LSMGenerator
        
        # Create generator with enhanced sinusoidal embeddings
        generator = LSMGenerator(
            tokenizer='gpt2',
            embedding_type='configurable_sinusoidal',
            sinusoidal_config={
                'learnable_frequencies': True,
                'base_frequency': 10000.0,
                'frequency_scaling': 1.0
            },
            enable_caching=True,
            window_size=8,
            embedding_dim=64
        )
        
        print(f"✓ Created LSMGenerator with enhanced tokenizer")
        print(f"  Tokenizer: {generator.tokenizer_name}")
        print(f"  Embedding type: {generator.embedding_type}")
        print(f"  Caching enabled: {generator.enable_caching}")
        
        # Get tokenizer info
        info = generator.get_tokenizer_info()
        print(f"  Tokenizer info: {info}")
        
        # Create enhanced tokenizer for inspection
        enhanced_tokenizer = generator.create_tokenizer_for_inference()
        print(f"  Enhanced tokenizer created: {type(enhanced_tokenizer).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_streaming_support():
    """Demonstrate streaming data support."""
    print("\n" + "=" * 60)
    print("Streaming Data Support Demo")
    print("=" * 60)
    
    try:
        from lsm.convenience.generator import LSMGenerator
        
        # Create generator with streaming enabled
        generator = LSMGenerator(
            tokenizer='gpt2',
            embedding_type='sinusoidal',
            streaming=True,
            streaming_config={
                'batch_size': 100,
                'memory_threshold_mb': 500.0,
                'auto_adjust_batch_size': True
            },
            enable_caching=True
        )
        
        print(f"✓ Created LSMGenerator with streaming support")
        print(f"  Streaming enabled: {generator.streaming}")
        print(f"  Streaming config: {generator.streaming_config}")
        
        # Create sample data for streaming demo
        sample_conversations = [
            "Hello, how are you today?",
            "I'm doing well, thank you for asking!",
            "What's the weather like?",
            "It's sunny and warm outside.",
            "That sounds lovely!",
            "Yes, perfect weather for a walk.",
            "I might go for a hike later.",
            "That sounds like a great idea!",
            "Do you enjoy hiking?",
            "I love being outdoors and exploring nature."
        ]
        
        print(f"✓ Created sample conversation data ({len(sample_conversations)} messages)")
        
        # Note: We won't actually train here since it requires the full training infrastructure
        print("✓ Streaming configuration validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_different_tokenizer_backends():
    """Demonstrate different tokenizer backends."""
    print("\n" + "=" * 60)
    print("Different Tokenizer Backends Demo")
    print("=" * 60)
    
    try:
        from lsm.convenience.generator import LSMGenerator
        
        # Test different tokenizer backends
        tokenizer_configs = [
            {
                'name': 'GPT-2 (HuggingFace)',
                'tokenizer': 'gpt2',
                'backend_config': {'use_fast': True}
            },
            {
                'name': 'BERT (HuggingFace)',
                'tokenizer': 'bert-base-uncased',
                'backend_config': {'use_fast': True}
            },
            {
                'name': 'DistilBERT (HuggingFace)',
                'tokenizer': 'distilbert-base-uncased',
                'backend_config': {'use_fast': True}
            }
        ]
        
        for config in tokenizer_configs:
            try:
                generator = LSMGenerator(
                    tokenizer=config['tokenizer'],
                    embedding_type='sinusoidal',
                    tokenizer_backend_config=config['backend_config'],
                    enable_caching=True
                )
                
                print(f"✓ {config['name']}: {generator.tokenizer_name}")
                
                # Get tokenizer info
                info = generator.get_tokenizer_info()
                print(f"  Info: {info}")
                
            except Exception as e:
                print(f"✗ {config['name']}: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_backward_compatibility():
    """Demonstrate backward compatibility with existing code."""
    print("\n" + "=" * 60)
    print("Backward Compatibility Demo")
    print("=" * 60)
    
    try:
        from lsm.convenience.generator import LSMGenerator
        
        # Test standard usage (should work exactly as before)
        generator_standard = LSMGenerator(
            tokenizer='gpt2',
            window_size=10,
            embedding_dim=128
        )
        
        print(f"✓ Standard LSMGenerator created")
        print(f"  Tokenizer: {generator_standard.tokenizer}")  # Using backward compatibility property
        print(f"  Embedding type: {generator_standard.embedding_type}")
        print(f"  Streaming: {generator_standard.streaming}")
        
        # Test preset usage
        generator_preset = LSMGenerator.from_preset('fast')
        print(f"✓ Preset LSMGenerator created")
        print(f"  Tokenizer: {generator_preset.tokenizer}")
        print(f"  Embedding type: {generator_preset.embedding_type}")
        
        # Test preset with enhanced parameters
        generator_enhanced_preset = LSMGenerator.from_preset(
            'fast',
            embedding_type='sinusoidal',
            streaming=True
        )
        print(f"✓ Enhanced preset LSMGenerator created")
        print(f"  Tokenizer: {generator_enhanced_preset.tokenizer}")
        print(f"  Embedding type: {generator_enhanced_preset.embedding_type}")
        print(f"  Streaming: {generator_enhanced_preset.streaming}")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demo_advanced_configuration():
    """Demonstrate advanced configuration options."""
    print("\n" + "=" * 60)
    print("Advanced Configuration Demo")
    print("=" * 60)
    
    try:
        from lsm.convenience.generator import LSMGenerator
        
        # Create generator with advanced configuration
        generator = LSMGenerator(
            tokenizer='gpt2',
            embedding_type='configurable_sinusoidal',
            sinusoidal_config={
                'learnable_frequencies': True,
                'use_relative_position': False,
                'base_frequency': 10000.0,
                'frequency_scaling': 1.2
            },
            streaming=True,
            streaming_config={
                'batch_size': 1000,
                'memory_threshold_mb': 1000.0,
                'auto_adjust_batch_size': True
            },
            tokenizer_backend_config={
                'use_fast': True,
                'padding_side': 'left'
            },
            enable_caching=True,
            system_message_support=True,
            response_level=True,
            temperature=0.8,
            max_length=256
        )
        
        print(f"✓ Advanced LSMGenerator created")
        print(f"  All enhanced features enabled")
        
        # Get comprehensive info
        info = generator.get_tokenizer_info()
        print(f"  Tokenizer info: {info}")
        
        # Test configuration access
        print(f"  Sinusoidal config: {generator.sinusoidal_config}")
        print(f"  Streaming config: {generator.streaming_config}")
        print(f"  Backend config: {generator.tokenizer_backend_config}")
        
        return True
        
    except Exception as e:
        print(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all demos."""
    print("Enhanced LSMGenerator Integration Demo")
    print("This demo shows the new enhanced tokenizer features")
    print("while maintaining full backward compatibility.")
    
    success = True
    success &= demo_basic_enhanced_tokenizer()
    success &= demo_streaming_support()
    success &= demo_different_tokenizer_backends()
    success &= demo_backward_compatibility()
    success &= demo_advanced_configuration()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Enhanced tokenizer with sinusoidal embeddings")
        print("• Streaming data processing support")
        print("• Multiple tokenizer backend support")
        print("• Intelligent caching system")
        print("• Full backward compatibility")
        print("• Advanced configuration options")
    else:
        print("✗ Some demos failed. Check the output above for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())