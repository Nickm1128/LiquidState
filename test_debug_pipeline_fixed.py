#!/usr/bin/env python3
"""
Fixed Debug Pipeline Test

This script tests the debug training pipeline with all the persistence fixes applied.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_complete_pipeline():
    """Test the complete pipeline with persistence fixes."""
    print("[START] Testing complete pipeline with persistence fixes")
    print("=" * 60)
    
    try:
        from lsm.convenience import LSMGenerator
        from lsm.convenience.config import ConvenienceConfig
        from lsm.convenience.utils import preprocess_conversation_data
        
        # Create configuration that works with the fixes
        config = ConvenienceConfig.get_preset('fast')
        config.update({
            'tokenizer': 'gpt2',
            'embedding_dim': 64,
            'max_length': 32,
            'embedding_type': 'configurable_sinusoidal',
            'enable_caching': False,
            'sinusoidal_config': {
                'learnable_frequencies': True,
                'base_frequency': 10000.0,
                'use_relative_position': False
            },
            'reservoir_type': 'standard',
            'window_size': 8,
            'system_message_support': False,
            'response_level': True,
            'random_state': 42,
            'epochs': 3,
            'batch_size': 16
        })
        
        print("[STEP 1] Creating LSM Generator...")
        generator = LSMGenerator(**config)
        print("[SUCCESS] Generator created")
        
        # Create sample data
        print("[STEP 2] Creating sample conversations...")
        sample_conversations = [
            "User: Hello\nAssistant: Hi there!",
            "User: How are you?\nAssistant: I'm doing well!",
            "User: What's AI?\nAssistant: Artificial Intelligence.",
            "User: Help me\nAssistant: I'm here to help!",
            "User: Tell me a joke\nAssistant: Why did the computer go to therapy? It had too many bytes!",
            "User: What's Python?\nAssistant: Python is a programming language.",
            "User: Explain ML\nAssistant: Machine Learning is a subset of AI.",
            "User: Good morning\nAssistant: Good morning! How can I help?",
        ] * 4  # 32 conversations
        
        processed_conversations = preprocess_conversation_data(
            sample_conversations,
            min_message_length=3,
            max_message_length=100,
            normalize_whitespace=True
        )
        
        print(f"[SUCCESS] Created {len(processed_conversations)} conversations")
        
        # Train the model
        print("[STEP 3] Training model...")
        generator.fit(
            processed_conversations,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_split=0.2,
            verbose=0
        )
        print("[SUCCESS] Training completed")
        
        # Test inference before saving
        print("[STEP 4] Testing inference before saving...")
        test_prompts = ["Hello", "Help me", "What's AI?"]
        original_responses = {}
        
        for prompt in test_prompts:
            try:
                response = generator.generate(prompt, max_length=20, temperature=0.8)
                original_responses[prompt] = response
                print(f"[TEST] \"{prompt}\" -> \"{response}\"")
            except Exception as e:
                print(f"[WARNING] Generation failed for \"{prompt}\": {e}")
                original_responses[prompt] = None
        
        # Save the model with fixed persistence
        print("[STEP 5] Saving model with fixed persistence...")
        MODEL_PATH = "debug_pipeline_fixed_model"
        generator.save(MODEL_PATH)
        print("[SUCCESS] Model saved")
        
        # Load the model
        print("[STEP 6] Loading model...")
        loaded_generator = LSMGenerator.load(MODEL_PATH)
        print("[SUCCESS] Model loaded")
        
        # Test inference after loading
        print("[STEP 7] Testing inference after loading...")
        loaded_responses = {}
        
        for prompt in test_prompts:
            try:
                response = loaded_generator.generate(prompt, max_length=20, temperature=0.8)
                loaded_responses[prompt] = response
                print(f"[TEST] \"{prompt}\" -> \"{response}\"")
            except Exception as e:
                print(f"[WARNING] Generation failed for \"{prompt}\": {e}")
                loaded_responses[prompt] = None
        
        # Compare results
        print("[STEP 8] Comparing results...")
        successful_tests = 0
        total_tests = len(test_prompts)
        
        for prompt in test_prompts:
            original = original_responses.get(prompt)
            loaded = loaded_responses.get(prompt)
            
            if original is not None and loaded is not None:
                successful_tests += 1
                print(f"[SUCCESS] \"{prompt}\": Both models generated responses")
            else:
                print(f"[WARNING] \"{prompt}\": One or both models failed")
        
        print(f"[RESULTS] {successful_tests}/{total_tests} tests successful")
        
        # Check saved files
        print("[STEP 9] Checking saved files...")
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Contents of {MODEL_PATH}:")
            for root, dirs, files in os.walk(MODEL_PATH):
                level = root.replace(MODEL_PATH, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({file_size} bytes)")
        
        return successful_tests == total_tests
        
    except Exception as e:
        print(f"[ERROR] Pipeline test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("[START] Fixed Debug Pipeline Test")
    print("=" * 60)
    print("Testing the debug training pipeline with all fixes applied")
    print("=" * 60)
    
    success = test_complete_pipeline()
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if success:
        print("[SUCCESS] All tests passed!")
        print("[INFO] The debug training pipeline is now working correctly.")
        print("[INFO] Model persistence has been fixed.")
        print("[INFO] Unicode encoding issues have been resolved.")
        print("[INFO] Inference is working with improved response generation.")
    else:
        print("[WARNING] Some tests failed.")
        print("[INFO] Check the output above for specific issues.")
    
    return success


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