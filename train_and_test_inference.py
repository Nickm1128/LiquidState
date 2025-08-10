#!/usr/bin/env python3
"""
Train and Test Inference Script

This script trains a fresh model and immediately tests inference without 
saving/loading to avoid serialization issues.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def setup_environment():
    """Setup the environment for training."""
    print("[SETUP] Setting up environment...")
    
    # Import warnings and suppress them for cleaner output
    import warnings
    warnings.filterwarnings('ignore')
    
    # TensorFlow setup
    try:
        import tensorflow as tf
        print(f"[INFO] TensorFlow version: {tf.__version__}")
        
        # Configure GPU if available
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"[SUCCESS] GPU configured: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(f"[WARNING] GPU configuration failed: {e}")
        else:
            print("[INFO] Using CPU (no GPU detected)")
            
    except ImportError:
        print("[ERROR] TensorFlow not available")
        return False
    
    return True


def create_sample_data():
    """Create sample conversation data."""
    print("[DATA] Creating sample conversation data...")
    
    try:
        from lsm.convenience.utils import (
            preprocess_conversation_data,
            detect_conversation_format
        )
        
        # Create sample conversation data
        sample_conversations = [
            "User: Hello, how are you today?\nAssistant: I'm doing well, thank you for asking! How can I help you?",
            "User: What's machine learning?\nAssistant: Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "User: Can you explain neural networks?\nAssistant: Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information.",
            "User: What's the weather like?\nAssistant: I don't have access to current weather data, but I'd recommend checking a weather app or website for accurate information.",
            "User: Tell me a joke\nAssistant: Why don't scientists trust atoms? Because they make up everything!",
            "User: How do I learn programming?\nAssistant: Start with a beginner-friendly language like Python, practice regularly, and work on small projects to build your skills.",
            "User: What's your favorite book?\nAssistant: I don't have personal preferences, but I can recommend popular books in various genres if you'd like!",
            "User: Explain quantum computing\nAssistant: Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in fundamentally different ways than classical computers.",
            "User: How does photosynthesis work?\nAssistant: Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen using chlorophyll.",
            "User: What is artificial intelligence?\nAssistant: Artificial intelligence is a branch of computer science that aims to create machines capable of intelligent behavior and decision-making.",
        ] * 5  # Repeat for more training data
        
        print(f"[INFO] Created {len(sample_conversations)} sample conversations")
        
        # Detect format
        detected_format = detect_conversation_format(sample_conversations)
        print(f"[INFO] Detected format: {detected_format}")
        
        # Preprocess conversations
        processed_conversations = preprocess_conversation_data(
            sample_conversations,
            min_message_length=5,
            max_message_length=500,
            normalize_whitespace=True
        )
        
        print(f"[SUCCESS] Processed {len(processed_conversations)} conversations")
        return processed_conversations
        
    except Exception as e:
        print(f"[ERROR] Data creation failed: {e}")
        traceback.print_exc()
        return None


def create_and_train_generator():
    """Create and train LSM generator."""
    print("[TRAIN] Creating and training LSM generator...")
    
    try:
        from lsm.convenience import LSMGenerator
        from lsm.convenience.config import ConvenienceConfig
        
        # Get configuration and customize for quick training
        config = ConvenienceConfig.get_preset('fast')
        config.update({
            # Enhanced tokenizer settings
            'tokenizer': 'gpt2',
            'embedding_dim': 64,  # Smaller for faster training
            'max_length': 32,     # Shorter sequences
            'embedding_type': 'configurable_sinusoidal',
            'enable_caching': False,
            
            # Sinusoidal embedding configuration
            'sinusoidal_config': {
                'learnable_frequencies': True,
                'base_frequency': 10000.0,
                'use_relative_position': False
            },
            
            # Model architecture - simplified
            'reservoir_type': 'standard',  # Simpler than attentive
            'window_size': 4,              # Smaller window
            'system_message_support': False,  # Disable for simplicity
            'response_level': True,
            'random_state': 42,
            
            # Training settings - very fast
            'epochs': 5,
            'batch_size': 16
        })
        
        print("[INFO] Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"   {key}: {len(value)} settings")
            else:
                print(f"   {key}: {value}")
        
        # Create generator
        generator = LSMGenerator(**config)
        print("[SUCCESS] LSM Generator created!")
        
        # Create sample data
        processed_conversations = create_sample_data()
        if processed_conversations is None:
            return None
        
        # Train the model
        print("[TRAIN] Starting training...")
        generator.fit(
            processed_conversations,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            verbose=1
        )
        
        print("[SUCCESS] Training completed!")
        return generator
        
    except Exception as e:
        print(f"[ERROR] Generator creation/training failed: {e}")
        traceback.print_exc()
        return None


def test_inference_immediately(generator):
    """Test inference immediately after training."""
    print("[INFERENCE] Testing inference immediately after training...")
    
    try:
        # Test prompts
        test_prompts = [
            "Hello",
            "How are you?",
            "What is AI?",
            "Tell me about Python",
            "Explain machine learning"
        ]
        
        print("\n" + "="*60)
        print("[TEST] IMMEDIATE INFERENCE TEST")
        print("="*60)
        
        successful_tests = 0
        total_tests = len(test_prompts)
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n[TEST] Test {i+1}/{total_tests}: \"{prompt}\"")
            print("-" * 40)
            
            try:
                # Test with different parameters
                temperatures = [0.5, 0.8, 1.0]
                
                for temp in temperatures:
                    try:
                        print(f"   [CONFIG] Testing T={temp}...")
                        
                        # Generate response
                        response = generator.generate(
                            prompt,
                            max_length=30,
                            temperature=temp,
                            return_confidence=True
                        )
                        
                        # Handle different return types
                        if isinstance(response, tuple):
                            response_text, confidence = response
                            print(f"   [SUCCESS] T={temp}: \"{response_text}\" (conf: {confidence:.3f})")
                        else:
                            print(f"   [SUCCESS] T={temp}: \"{response}\"")
                        
                        successful_tests += 1
                        break  # Success, move to next prompt
                        
                    except Exception as param_error:
                        print(f"   [ERROR] T={temp} failed: {param_error}")
                        continue
                else:
                    print(f"   [ERROR] All temperatures failed for this prompt")
                    
            except Exception as e:
                print(f"   [ERROR] Test failed: {e}")
                print(f"   [DEBUG] Error type: {type(e).__name__}")
        
        print(f"\n[RESULTS] Successful tests: {successful_tests}/{total_tests}")
        
        # Test model info
        print(f"\n[INFO] Model Information:")
        try:
            model_info = generator.get_model_info()
            for key, value in model_info.items():
                if isinstance(value, dict):
                    print(f"   {key}: {len(value)} components")
                else:
                    print(f"   {key}: {value}")
        except Exception as info_error:
            print(f"   [ERROR] Could not get model info: {info_error}")
        
        # Test enhanced tokenizer
        print(f"\n[TOKENIZER] Enhanced Tokenizer Test:")
        try:
            enhanced_tokenizer = generator.get_enhanced_tokenizer()
            if enhanced_tokenizer:
                print(f"   [SUCCESS] Tokenizer: {type(enhanced_tokenizer).__name__}")
                print(f"   Backend: {enhanced_tokenizer.get_adapter().config.backend}")
                print(f"   Vocab size: {enhanced_tokenizer.get_vocab_size():,}")
                
                # Test tokenization
                test_text = "Hello world"
                tokens = enhanced_tokenizer.tokenize([test_text])
                decoded = enhanced_tokenizer.decode(tokens[0])
                print(f"   [TEST] '{test_text}' -> {len(tokens[0])} tokens -> '{decoded}'")
            else:
                print("   [ERROR] No enhanced tokenizer found")
        except Exception as tok_error:
            print(f"   [ERROR] Tokenizer test failed: {tok_error}")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"[ERROR] Inference testing failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("[START] Train and Test Inference Script")
    print("=" * 60)
    print("Training a fresh model and testing inference immediately")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Step 1: Environment setup
    print("\n" + "="*60)
    print("STEP 1: ENVIRONMENT SETUP")
    print("="*60)
    results['environment'] = setup_environment()
    
    if not results['environment']:
        print("[ERROR] Environment setup failed")
        return False
    
    # Step 2: Create and train generator
    print("\n" + "="*60)
    print("STEP 2: CREATE AND TRAIN GENERATOR")
    print("="*60)
    generator = create_and_train_generator()
    results['training'] = generator is not None
    
    if not results['training']:
        print("[ERROR] Training failed")
        return False
    
    # Step 3: Test inference immediately
    print("\n" + "="*60)
    print("STEP 3: TEST INFERENCE IMMEDIATELY")
    print("="*60)
    results['inference'] = test_inference_immediately(generator)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name.replace('_', ' ').title():<20} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Training and inference work correctly.")
        print("[INFO] The issue is likely with model serialization/deserialization.")
    else:
        print("[WARNING] Some tests failed.")
        failures = [name for name, success in results.items() if not success]
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. {failure.replace('_', ' ').title()}")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)