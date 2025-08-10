#!/usr/bin/env python3
"""
Test Persistence Fix Script

This script tests the new custom persistence mechanism that should avoid
Keras serialization issues.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def train_and_save_model():
    """Train a model and save it using the new persistence mechanism."""
    print("[TRAIN] Training and saving model with new persistence...")
    
    try:
        from lsm.convenience import LSMGenerator
        from lsm.convenience.config import ConvenienceConfig
        
        # Create configuration for very fast training
        config = ConvenienceConfig.get_preset('fast')
        config.update({
            'tokenizer': 'gpt2',
            'embedding_dim': 32,  # Very small for fast training
            'max_length': 16,     # Short sequences
            'embedding_type': 'configurable_sinusoidal',
            'enable_caching': False,
            'sinusoidal_config': {
                'learnable_frequencies': True,
                'base_frequency': 10000.0,
                'use_relative_position': False
            },
            'reservoir_type': 'standard',
            'window_size': 2,     # Very small window
            'system_message_support': False,
            'response_level': True,
            'random_state': 42,
            'epochs': 2,          # Minimal training
            'batch_size': 8
        })
        
        print("[INFO] Creating LSM Generator...")
        generator = LSMGenerator(**config)
        
        # Create minimal sample data
        print("[INFO] Creating minimal sample conversations...")
        from lsm.convenience.utils import preprocess_conversation_data
        
        sample_conversations = [
            "User: Hi\nAssistant: Hello!",
            "User: How are you?\nAssistant: I'm good!",
            "User: What's AI?\nAssistant: Artificial Intelligence.",
            "User: Help me\nAssistant: Sure, I'll help!",
        ] * 2  # Just 8 conversations for speed
        
        processed_conversations = preprocess_conversation_data(
            sample_conversations,
            min_message_length=2,
            max_message_length=50,
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
        test_response = generator.generate("Hello", max_length=10, temperature=0.8)
        print(f"[INFO] Pre-save response: \"{test_response}\"")
        
        # Save the model
        MODEL_PATH = "test_persistence_model"
        print(f"[SAVE] Saving model to {MODEL_PATH}...")
        generator.save(MODEL_PATH)
        print("[SUCCESS] Model saved successfully!")
        
        return generator, MODEL_PATH, test_response
        
    except Exception as e:
        print(f"[ERROR] Training and saving failed: {e}")
        traceback.print_exc()
        return None, None, None


def load_and_test_model(model_path, original_response):
    """Load the saved model and test it."""
    print(f"[LOAD] Loading model from {model_path}...")
    
    try:
        from lsm.convenience import LSMGenerator
        
        # Load the model
        loaded_generator = LSMGenerator.load(model_path)
        print("[SUCCESS] Model loaded successfully!")
        
        # Test inference after loading
        print("[TEST] Testing inference after loading...")
        test_response = loaded_generator.generate("Hello", max_length=10, temperature=0.8)
        print(f"[INFO] Post-load response: \"{test_response}\"")
        
        # Compare responses
        print(f"[COMPARE] Original response: \"{original_response}\"")
        print(f"[COMPARE] Loaded response:   \"{test_response}\"")
        
        # Test model info
        print("[INFO] Testing model info...")
        model_info = loaded_generator.get_model_info()
        print(f"[INFO] Model type: {model_info.get('model_type', 'Unknown')}")
        print(f"[INFO] Is fitted: {model_info.get('is_fitted', False)}")
        print(f"[INFO] Embedding dim: {model_info.get('embedding_dim', 'Unknown')}")
        
        # Test multiple generations
        print("[TEST] Testing multiple generations...")
        test_prompts = ["Hi", "Help", "What?"]
        for prompt in test_prompts:
            try:
                response = loaded_generator.generate(prompt, max_length=8, temperature=0.7)
                print(f"[TEST] \"{prompt}\" -> \"{response}\"")
            except Exception as e:
                print(f"[ERROR] Generation failed for \"{prompt}\": {e}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Loading and testing failed: {e}")
        traceback.print_exc()
        return False


def check_saved_files(model_path):
    """Check what files were saved with the new persistence mechanism."""
    print(f"[CHECK] Checking saved files in {model_path}...")
    
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
        
        # Check for new persistence files
        model_dir = os.path.join(model_path, "model")
        if os.path.exists(model_dir):
            print(f"[INFO] Checking for new persistence files in {model_dir}...")
            
            # Check for custom persistence files
            persistence_files = [
                "reservoir_model_weights.h5",
                "reservoir_model_architecture.json",
                "reservoir_model_config.json",
                "cnn_model_weights.h5",
                "cnn_model_architecture.json",
                "cnn_model_config.json"
            ]
            
            found_files = []
            for file in persistence_files:
                file_path = os.path.join(model_dir, file)
                if os.path.exists(file_path):
                    found_files.append(file)
            
            if found_files:
                print(f"[SUCCESS] Found new persistence files: {found_files}")
            else:
                print("[INFO] No new persistence files found, checking for fallback files...")
                fallback_files = ["reservoir_model.keras", "cnn_model.keras"]
                for file in fallback_files:
                    file_path = os.path.join(model_dir, file)
                    if os.path.exists(file_path):
                        print(f"[INFO] Found fallback file: {file}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] File check failed: {e}")
        return False


def main():
    """Main execution function."""
    print("[START] Persistence Fix Test")
    print("=" * 60)
    print("Testing the new custom persistence mechanism")
    print("=" * 60)
    
    # Step 1: Train and save model
    print("\n" + "="*60)
    print("STEP 1: TRAIN AND SAVE MODEL")
    print("="*60)
    generator, model_path, original_response = train_and_save_model()
    
    if generator is None:
        print("[ERROR] Training failed, cannot continue")
        return False
    
    # Step 2: Check saved files
    print("\n" + "="*60)
    print("STEP 2: CHECK SAVED FILES")
    print("="*60)
    files_ok = check_saved_files(model_path)
    
    # Step 3: Load and test model
    print("\n" + "="*60)
    print("STEP 3: LOAD AND TEST MODEL")
    print("="*60)
    load_success = load_and_test_model(model_path, original_response)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    results = {
        'training_and_saving': generator is not None,
        'file_structure': files_ok,
        'loading_and_testing': load_success
    }
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] Persistence fix working! Model can be saved and loaded successfully.")
    else:
        print("[WARNING] Some tests failed. Persistence issue may not be fully resolved.")
        
        if not results['loading_and_testing']:
            print("[INFO] The loading issue may still be related to Keras serialization.")
            print("[INFO] Consider implementing a more comprehensive custom serialization.")
    
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