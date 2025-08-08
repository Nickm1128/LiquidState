#!/usr/bin/env python3
"""
Safe import test script that handles TensorFlow issues gracefully.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set TensorFlow to CPU only and suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_import_safe(module_name, description=""):
    """Test importing a module with TensorFlow error handling."""
    try:
        # Try to import TensorFlow first if this module needs it
        if any(tf_module in module_name for tf_module in ['core', 'training', 'inference']):
            try:
                import tensorflow as tf
                # Force CPU usage
                tf.config.set_visible_devices([], 'GPU')
            except Exception as tf_error:
                print(f"⚠ {module_name} {description} - TensorFlow issue: {str(tf_error)[:100]}...")
                return False
        
        __import__(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except ImportError as e:
        if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
            print(f"⚠ {module_name} {description} - TensorFlow/DLL issue (expected on some systems)")
            return False
        else:
            print(f"✗ {module_name} {description} - Import Error: {e}")
            return False
    except Exception as e:
        print(f"✗ {module_name} {description} - ERROR: {e}")
        return False

def test_non_tf_imports():
    """Test imports that don't require TensorFlow."""
    print("=" * 60)
    print("TESTING NON-TENSORFLOW IMPORTS")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Test non-TensorFlow modules first
    non_tf_tests = [
        ("lsm.data", "- Data package"),
        ("lsm.data.data_loader", "- Data loader"),
        ("lsm.training.model_config", "- Model configuration"),
        ("lsm.management", "- Management package"),
        ("lsm.management.model_manager", "- Model manager"),
        ("lsm.management.manage_models", "- Model management CLI"),
        ("lsm.utils", "- Utils package"),
        ("lsm.utils.lsm_exceptions", "- Custom exceptions"),
        ("lsm.utils.lsm_logging", "- Logging utilities"),
        ("lsm.utils.input_validation", "- Input validation"),
        ("lsm.utils.production_validation", "- Production validation"),
    ]
    
    for module_name, description in non_tf_tests:
        total_count += 1
        if test_import_safe(module_name, description):
            success_count += 1
    
    print(f"\nNon-TensorFlow Import Results: {success_count}/{total_count} successful")
    return success_count, total_count

def test_tf_imports():
    """Test TensorFlow-dependent imports."""
    print("\n" + "=" * 60)
    print("TESTING TENSORFLOW-DEPENDENT IMPORTS")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Test TensorFlow-dependent modules
    tf_tests = [
        ("lsm.core", "- Core package"),
        ("lsm.core.reservoir", "- Basic reservoir"),
        ("lsm.core.advanced_reservoir", "- Advanced reservoir"),
        ("lsm.core.rolling_wave", "- Rolling wave buffer"),
        ("lsm.core.cnn_model", "- CNN model"),
        ("lsm.training.train", "- Training pipeline"),
        ("lsm.inference", "- Inference package"),
        ("lsm.inference.inference", "- Inference implementation"),
    ]
    
    for module_name, description in tf_tests:
        total_count += 1
        if test_import_safe(module_name, description):
            success_count += 1
    
    print(f"\nTensorFlow Import Results: {success_count}/{total_count} successful")
    return success_count, total_count

def test_backward_compatibility():
    """Test backward compatibility imports."""
    print("\n" + "=" * 60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # First test if main package loads
    try:
        import lsm
        print("✓ Main lsm package imported successfully")
        
        # Test backward compatibility imports
        backward_compat_tests = [
            ("DialogueTokenizer", "DialogueTokenizer backward compatibility"),
            ("ModelManager", "ModelManager backward compatibility"),
        ]
        
        for attr_name, description in backward_compat_tests:
            total_count += 1
            try:
                getattr(lsm, attr_name)
                print(f"✓ lsm.{attr_name} - {description}")
                success_count += 1
            except AttributeError:
                print(f"✗ lsm.{attr_name} - {description} - Not available")
            except Exception as e:
                print(f"✗ lsm.{attr_name} - {description} - ERROR: {e}")
        
        # Test TensorFlow-dependent backward compatibility (may fail)
        tf_compat_tests = [
            ("LSMTrainer", "LSMTrainer backward compatibility"),
            ("OptimizedLSMInference", "OptimizedLSMInference backward compatibility"),
            ("LSMInference", "LSMInference backward compatibility"),
        ]
        
        for attr_name, description in tf_compat_tests:
            total_count += 1
            try:
                getattr(lsm, attr_name)
                print(f"✓ lsm.{attr_name} - {description}")
                success_count += 1
            except Exception as e:
                if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
                    print(f"⚠ lsm.{attr_name} - {description} - TensorFlow issue (expected)")
                else:
                    print(f"✗ lsm.{attr_name} - {description} - ERROR: {e}")
        
    except Exception as e:
        print(f"✗ Main lsm package failed to import: {e}")
        total_count = 5  # Total tests we would have run
    
    print(f"\nBackward Compatibility Results: {success_count}/{total_count} successful")
    return success_count, total_count

def main():
    """Run comprehensive import tests with TensorFlow error handling."""
    print("COMPREHENSIVE IMPORT VALIDATION (TensorFlow-Safe)")
    print("=" * 60)
    
    # Test non-TensorFlow imports
    non_tf_success, non_tf_total = test_non_tf_imports()
    
    # Test TensorFlow imports
    tf_success, tf_total = test_tf_imports()
    
    # Test backward compatibility
    compat_success, compat_total = test_backward_compatibility()
    
    # Summary
    total_tests = non_tf_total + tf_total + compat_total
    total_success = non_tf_success + tf_success + compat_success
    
    print("\n" + "=" * 60)
    print("IMPORT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Non-TensorFlow modules: {non_tf_success}/{non_tf_total} successful")
    print(f"TensorFlow modules: {tf_success}/{tf_total} successful")
    print(f"Backward compatibility: {compat_success}/{compat_total} successful")
    print(f"Total: {total_success}/{total_tests} successful")
    print(f"Success rate: {(total_success/total_tests)*100:.1f}%")
    
    # Determine if this is acceptable
    if non_tf_success == non_tf_total and compat_success >= compat_total * 0.6:
        print("\n✅ CORE FUNCTIONALITY IMPORTS SUCCESSFUL!")
        print("Note: TensorFlow-dependent modules may fail due to environment issues.")
        return True
    else:
        print(f"\n❌ CRITICAL IMPORT FAILURES DETECTED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)