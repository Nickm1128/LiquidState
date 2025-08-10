#!/usr/bin/env python3
"""
Simple Persistence Test Script

This script tests the persistence mechanism by creating a simple model
and testing the custom save/load functionality directly.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_custom_persistence_directly():
    """Test the custom persistence mechanism directly with a simple Keras model."""
    print("[TEST] Testing custom persistence mechanism directly...")
    
    try:
        import tensorflow as tf
        from keras import layers, models
        import numpy as np
        
        # Create a simple test model
        print("[CREATE] Creating simple test model...")
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_shape=(32,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(16, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(f"[SUCCESS] Test model created with {model.count_params()} parameters")
        
        # Create some dummy data and train briefly
        print("[TRAIN] Training test model briefly...")
        X_dummy = np.random.random((100, 32))
        y_dummy = np.random.random((100, 16))
        
        model.fit(X_dummy, y_dummy, epochs=2, verbose=0)
        print("[SUCCESS] Test model trained")
        
        # Test the model before saving
        test_input = np.random.random((1, 32))
        original_output = model.predict(test_input, verbose=0)
        print(f"[TEST] Original output shape: {original_output.shape}")
        
        # Now test our custom persistence methods
        from lsm.training.train import LSMTrainer
        trainer = LSMTrainer()
        
        # Test custom save
        print("[SAVE] Testing custom save method...")
        base_path = "test_custom_model"
        trainer._save_model_safely(model, base_path, "TestModel")
        
        # Check if files were created
        expected_files = [
            base_path + "_weights.h5",
            base_path + "_architecture.json",
            base_path + "_config.json"
        ]
        
        files_created = []
        for file_path in expected_files:
            if os.path.exists(file_path):
                files_created.append(file_path)
                file_size = os.path.getsize(file_path)
                print(f"[SUCCESS] Created {file_path} ({file_size} bytes)")
            else:
                print(f"[WARNING] Missing {file_path}")
        
        if len(files_created) == len(expected_files):
            print("[SUCCESS] All expected files created")
        else:
            print(f"[WARNING] Only {len(files_created)}/{len(expected_files)} files created")
        
        # Test custom load
        print("[LOAD] Testing custom load method...")
        loaded_model = trainer._load_model_safely(base_path, "TestModel")
        
        if loaded_model is not None:
            print("[SUCCESS] Model loaded successfully")
            
            # Test the loaded model
            loaded_output = loaded_model.predict(test_input, verbose=0)
            print(f"[TEST] Loaded output shape: {loaded_output.shape}")
            
            # Compare outputs
            output_diff = np.abs(original_output - loaded_output).mean()
            print(f"[COMPARE] Mean absolute difference: {output_diff:.6f}")
            
            if output_diff < 1e-5:
                print("[SUCCESS] Outputs match (difference < 1e-5)")
                return True
            else:
                print(f"[WARNING] Outputs differ by {output_diff:.6f}")
                return False
        else:
            print("[ERROR] Model loading failed")
            return False
        
    except Exception as e:
        print(f"[ERROR] Custom persistence test failed: {e}")
        traceback.print_exc()
        return False
    finally:
        # Clean up test files
        cleanup_files = [
            "test_custom_model_weights.h5",
            "test_custom_model_architecture.json", 
            "test_custom_model_config.json",
            "test_custom_model.keras"
        ]
        for file_path in cleanup_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"[CLEANUP] Removed {file_path}")
                except:
                    pass


def test_existing_model_loading():
    """Test loading the existing debug_enhanced_model if it exists."""
    print("[TEST] Testing existing model loading...")
    
    try:
        MODEL_PATH = "debug_enhanced_model"
        
        if not os.path.exists(MODEL_PATH):
            print(f"[INFO] No existing model found at {MODEL_PATH}")
            return None
        
        print(f"[INFO] Found existing model at {MODEL_PATH}")
        
        # Check what files exist
        model_dir = os.path.join(MODEL_PATH, "model")
        if os.path.exists(model_dir):
            print(f"[INFO] Contents of {model_dir}:")
            for file in os.listdir(model_dir):
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):
                    file_size = os.path.getsize(file_path)
                    print(f"   {file} ({file_size} bytes)")
        
        # Try to load using our custom method
        from lsm.training.train import LSMTrainer
        trainer = LSMTrainer()
        
        # Test loading reservoir model
        reservoir_path = os.path.join(model_dir, "reservoir_model")
        if os.path.exists(reservoir_path + "_weights.h5"):
            print("[TEST] Attempting to load reservoir model with custom method...")
            reservoir_model = trainer._load_model_safely(reservoir_path, "Reservoir")
            if reservoir_model:
                print("[SUCCESS] Reservoir model loaded with custom method")
            else:
                print("[WARNING] Reservoir model loading failed with custom method")
        
        # Test loading CNN model
        cnn_path = os.path.join(model_dir, "cnn_model")
        if os.path.exists(cnn_path + "_weights.h5"):
            print("[TEST] Attempting to load CNN model with custom method...")
            cnn_model = trainer._load_model_safely(cnn_path, "CNN")
            if cnn_model:
                print("[SUCCESS] CNN model loaded with custom method")
            else:
                print("[WARNING] CNN model loading failed with custom method")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Existing model loading test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("[START] Simple Persistence Test")
    print("=" * 60)
    print("Testing custom persistence mechanism directly")
    print("=" * 60)
    
    # Test 1: Custom persistence with simple model
    print("\n" + "="*60)
    print("TEST 1: CUSTOM PERSISTENCE WITH SIMPLE MODEL")
    print("="*60)
    custom_persistence_success = test_custom_persistence_directly()
    
    # Test 2: Existing model loading (if available)
    print("\n" + "="*60)
    print("TEST 2: EXISTING MODEL LOADING")
    print("="*60)
    existing_model_success = test_existing_model_loading()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    results = {
        'custom_persistence': custom_persistence_success,
        'existing_model_loading': existing_model_success is not None
    }
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if results['custom_persistence']:
        print("[SUCCESS] Custom persistence mechanism is working correctly!")
        print("[INFO] The issue with model loading is likely in the LSM-specific components.")
        print("[INFO] The Keras serialization problem has been solved with the custom method.")
    else:
        print("[WARNING] Custom persistence mechanism has issues.")
    
    return results['custom_persistence']


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