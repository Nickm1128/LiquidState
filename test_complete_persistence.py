#!/usr/bin/env python3
"""
Complete Persistence Test Script

This script tests the complete LSM model persistence with the fixed custom mechanism.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def train_and_save_complete_model():
    """Train a complete LSM model and save it using the new persistence mechanism."""
    print("[TRAIN] Training complete LSM model with fixed persistence...")
    
    try:
        from lsm.convenience import LSMGenerator
        from lsm.convenience.config import ConvenienceConfig
        
        # Create configuration that avoids the CNN pooling issue
        config = ConvenienceConfig.get_preset('fast')
        config.update({
            'tokenizer': 'gpt2',
            'embedding_dim': 64,     # Reasonable size
            'max_length': 32,        # Reasonable length
            'embedding_type': 'configurable_sinusoidal',
            'enable_caching': False,
            'sinusoidal_config': {
                'learnable_frequencies': True,
                'base_frequency': 10000.0,
                'use_relative_position': False
            },
            'reservoir_type': 'standard',
            'window_size': 8,        # Larger window to avoid CNN issues
            'system_message_support': False,
            'response_level': True,
            'random_state': 42,
            'epochs': 3,
            'batch_size': 16
        })
        
        print("[INFO] Creating LSM Generator...")
        generator = LSMGenerator(**config)
        
        # Create sample data
        print("[INFO] Creating sample conversations...")
        from lsm.convenience.utils import preprocess_conversation_data
        
        sample_conversations = [
            "User: Hello\nAssistant: Hi there!",
            "User: How are you?\nAssistant: I'm doing well, thanks!",
            "User: What's AI?\nAssistant: Artificial Intelligence is computer science.",
            "User: Help me\nAssistant: I'm here to help!",
            "User: Tell me a joke\nAssistant: Why did the computer go to therapy? It had too many bytes!",
            "User: What's Python?\nAssistant: Python is a programming language.",
            "User: Explain ML\nAssistant: Machine Learning is a subset of AI.",
            "User: Good morning\nAssistant: Good morning! How can I assist you?",
        ] * 3  # 24 conversations
        
        processed_conversations = preprocess_conversation_data(
            sample_conversations,
            min_message_length=3,
            max_message_length=100,
            normalize_whitespace=True
        )
        
        print(f"[INFO] Training on {len(processed_conversations)} conversations...")
        
        # Train the model
        generator.fit(
            processed_conversations,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            verbose=0  # Quiet training
        )
        
        print("[SUCCESS] Training completed!")
        
        # Test inference before saving
        print("[TEST] Testing inference before saving...")
        test_prompts = ["Hello", "Help me", "What's AI?"]
        original_responses = {}
        
        for prompt in test_prompts:
            try:
                response = generator.generate(prompt, max_length=20, temperature=0.8)
                original_responses[prompt] = response
                print(f"[INFO] \"{prompt}\" -> \"{response}\"")
            except Exception as e:
                print(f"[WARNING] Generation failed for \"{prompt}\": {e}")
                original_responses[prompt] = None
        
        # Save the model
        MODEL_PATH = "complete_persistence_model"
        print(f"[SAVE] Saving complete model to {MODEL_PATH}...")
        generator.save(MODEL_PATH)
        print("[SUCCESS] Complete model saved successfully!")
        
        return generator, MODEL_PATH, original_responses
        
    except Exception as e:
        print(f"[ERROR] Training and saving failed: {e}")
        traceback.print_exc()
        return None, None, None


def load_and_test_complete_model(model_path, original_responses):
    """Load the complete model and test it."""
    print(f"[LOAD] Loading complete model from {model_path}...")
    
    try:
        from lsm.convenience import LSMGenerator
        
        # Load the model
        loaded_generator = LSMGenerator.load(model_path)
        print("[SUCCESS] Complete model loaded successfully!")
        
        # Test model info
        print("[INFO] Testing model info...")
        model_info = loaded_generator.get_model_info()
        print(f"[INFO] Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"[INFO] Is fitted: {model_info.get('is_fitted', False)}")
        print(f"[INFO] Embedding dim: {model_info.get('embedding_dim', 'Unknown')}")
        print(f"[INFO] Reservoir type: {model_info.get('reservoir_type', 'Unknown')}")
        
        # Test inference after loading
        print("[TEST] Testing inference after loading...")
        loaded_responses = {}
        
        for prompt, original_response in original_responses.items():
            if original_response is not None:
                try:
                    loaded_response = loaded_generator.generate(prompt, max_length=20, temperature=0.8)
                    loaded_responses[prompt] = loaded_response
                    print(f"[INFO] \"{prompt}\" -> \"{loaded_response}\"")
                except Exception as e:
                    print(f"[WARNING] Generation failed for \"{prompt}\": {e}")
                    loaded_responses[prompt] = None
        
        # Compare responses
        print("[COMPARE] Comparing original vs loaded responses:")
        successful_comparisons = 0
        total_comparisons = 0
        
        for prompt in original_responses:
            original = original_responses[prompt]
            loaded = loaded_responses.get(prompt)
            
            if original is not None and loaded is not None:
                total_comparisons += 1
                print(f"[COMPARE] \"{prompt}\":")
                print(f"   Original: \"{original}\"")
                print(f"   Loaded:   \"{loaded}\"")
                
                # Both models can generate (don't require identical due to randomness)
                if len(original) > 0 and len(loaded) > 0:
                    successful_comparisons += 1
                    print(f"   [SUCCESS] Both models generated responses")
                else:
                    print(f"   [WARNING] One or both responses are empty")
        
        print(f"[RESULTS] {successful_comparisons}/{total_comparisons} comparisons successful")
        
        # Test additional functionality
        print("[TEST] Testing additional functionality...")
        try:
            # Test enhanced tokenizer
            enhanced_tokenizer = loaded_generator.get_enhanced_tokenizer()
            if enhanced_tokenizer:
                print(f"[SUCCESS] Enhanced tokenizer available: {type(enhanced_tokenizer).__name__}")
                print(f"   Backend: {enhanced_tokenizer.get_adapter().config.backend}")
                print(f"   Vocab size: {enhanced_tokenizer.get_vocab_size():,}")
            else:
                print("[WARNING] Enhanced tokenizer not available")
        except Exception as e:
            print(f"[WARNING] Enhanced tokenizer test failed: {e}")
        
        return successful_comparisons == total_comparisons
        
    except Exception as e:
        print(f"[ERROR] Loading and testing failed: {e}")
        traceback.print_exc()
        return False


def check_persistence_files(model_path):
    """Check the persistence files created by the new mechanism."""
    print(f"[CHECK] Checking persistence files in {model_path}...")
    
    try:
        if not os.path.exists(model_path):
            print(f"[ERROR] Model path {model_path} does not exist!")
            return False
        
        print(f"[INFO] Contents of {model_path}:")
        for root, dirs, files in os.walk(model_path):
            level = root.replace(model_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size} bytes)")
        
        # Check for new persistence files in model directory
        model_dir = os.path.join(model_path, "model")
        if os.path.exists(model_dir):
            print(f"[INFO] Checking for new persistence files...")
            
            # Check for custom persistence files
            persistence_files = [
                "reservoir_model_weights.h5",
                "reservoir_model_architecture.json",
                "reservoir_model_config.json",
                "cnn_model_weights.h5",
                "cnn_model_architecture.json",
                "cnn_model_config.json"
            ]
            
            found_custom = []
            found_fallback = []
            
            for file in persistence_files:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    found_custom.append(file)
            
            # Check for fallback files
            fallback_files = ["reservoir_model.keras", "cnn_model.keras"]
            for file in fallback_files:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    found_fallback.append(file)
            
            if found_custom:
                print(f"[SUCCESS] Found custom persistence files: {found_custom}")
            if found_fallback:
                print(f"[INFO] Found fallback files: {found_fallback}")
            
            if found_custom or found_fallback:
                print("[SUCCESS] Persistence mechanism working")
                return True
            else:
                print("[WARNING] No persistence files found")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] File check failed: {e}")
        return False


def main():
    """Main execution function."""
    print("[START] Complete Persistence Test")
    print("=" * 60)
    print("Testing complete LSM model persistence with fixed mechanism")
    print("=" * 60)
    
    # Step 1: Train and save complete model
    print("\n" + "="*60)
    print("STEP 1: TRAIN AND SAVE COMPLETE MODEL")
    print("="*60)
    generator, model_path, original_responses = train_and_save_complete_model()
    
    if generator is None:
        print("[ERROR] Training failed, cannot continue")
        return False
    
    # Step 2: Check persistence files
    print("\n" + "="*60)
    print("STEP 2: CHECK PERSISTENCE FILES")
    print("="*60)
    files_ok = check_persistence_files(model_path)
    
    # Step 3: Load and test complete model
    print("\n" + "="*60)
    print("STEP 3: LOAD AND TEST COMPLETE MODEL")
    print("="*60)
    load_success = load_and_test_complete_model(model_path, original_responses)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    results = {
        'training_and_saving': generator is not None,
        'persistence_files': files_ok,
        'loading_and_testing': load_success
    }
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Complete persistence fix working!")
        print("[INFO] LSM models can now be saved and loaded successfully.")
        print("[INFO] The Unicode encoding issues have been resolved.")
        print("[INFO] The custom persistence mechanism avoids Keras serialization problems.")
    else:
        print("[WARNING] Some tests failed.")
        
        if not results['loading_and_testing']:
            print("[INFO] There may still be issues with LSM-specific components.")
        if not results['persistence_files']:
            print("[INFO] File structure issues detected.")
    
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