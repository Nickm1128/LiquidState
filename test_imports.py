#!/usr/bin/env python3
"""
Comprehensive import test script to verify all modules can be imported correctly.
"""

import sys
import traceback
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import(module_name, description=""):
    """Test importing a module and return success status."""
    try:
        __import__(module_name)
        print(f"✓ {module_name} {description}")
        return True
    except Exception as e:
        print(f"✗ {module_name} {description} - ERROR: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive import tests."""
    print("=" * 60)
    print("COMPREHENSIVE IMPORT VALIDATION")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Test main package
    tests = [
        ("lsm", "- Main package"),
        
        # Core modules
        ("lsm.core", "- Core package"),
        ("lsm.core.reservoir", "- Basic reservoir"),
        ("lsm.core.advanced_reservoir", "- Advanced reservoir"),
        ("lsm.core.rolling_wave", "- Rolling wave buffer"),
        ("lsm.core.cnn_model", "- CNN model"),
        
        # Data modules
        ("lsm.data", "- Data package"),
        ("lsm.data.data_loader", "- Data loader"),
        
        # Training modules
        ("lsm.training", "- Training package"),
        ("lsm.training.train", "- Training pipeline"),
        ("lsm.training.model_config", "- Model configuration"),
        
        # Inference modules
        ("lsm.inference", "- Inference package"),
        ("lsm.inference.inference", "- Inference implementation"),
        
        # Management modules
        ("lsm.management", "- Management package"),
        ("lsm.management.model_manager", "- Model manager"),
        ("lsm.management.manage_models", "- Model management CLI"),
        
        # Utility modules
        ("lsm.utils", "- Utils package"),
        ("lsm.utils.lsm_exceptions", "- Custom exceptions"),
        ("lsm.utils.lsm_logging", "- Logging utilities"),
        ("lsm.utils.input_validation", "- Input validation"),
        ("lsm.utils.production_validation", "- Production validation"),
    ]
    
    print("\n1. Testing individual module imports:")
    print("-" * 40)
    
    for module_name, description in tests:
        total_count += 1
        if test_import(module_name, description):
            success_count += 1
    
    print(f"\nModule Import Results: {success_count}/{total_count} successful")
    
    # Test backward compatibility imports
    print("\n2. Testing backward compatibility imports:")
    print("-" * 40)
    
    backward_compat_tests = [
        ("from lsm import LSMTrainer", "LSMTrainer backward compatibility"),
        ("from lsm import OptimizedLSMInference", "OptimizedLSMInference backward compatibility"),
        ("from lsm import LSMInference", "LSMInference backward compatibility"),
        ("from lsm import DialogueTokenizer", "DialogueTokenizer backward compatibility"),
    ]
    
    compat_success = 0
    compat_total = len(backward_compat_tests)
    
    for import_statement, description in backward_compat_tests:
        try:
            exec(import_statement)
            print(f"✓ {import_statement} - {description}")
            compat_success += 1
        except Exception as e:
            print(f"✗ {import_statement} - {description} - ERROR: {e}")
    
    print(f"\nBackward Compatibility Results: {compat_success}/{compat_total} successful")
    
    # Test specific class imports
    print("\n3. Testing specific class imports:")
    print("-" * 40)
    
    class_tests = [
        ("from lsm.core.reservoir import ReservoirLayer", "ReservoirLayer class"),
        ("from lsm.core.advanced_reservoir import HierarchicalReservoir", "HierarchicalReservoir class"),
        ("from lsm.data.data_loader import DialogueTokenizer", "DialogueTokenizer class"),
        ("from lsm.training.train import LSMTrainer", "LSMTrainer class"),
        ("from lsm.inference.inference import OptimizedLSMInference", "OptimizedLSMInference class"),
        ("from lsm.management.model_manager import ModelManager", "ModelManager class"),
    ]
    
    class_success = 0
    class_total = len(class_tests)
    
    for import_statement, description in class_tests:
        try:
            exec(import_statement)
            print(f"✓ {import_statement} - {description}")
            class_success += 1
        except Exception as e:
            print(f"✗ {import_statement} - {description} - ERROR: {e}")
    
    print(f"\nClass Import Results: {class_success}/{class_total} successful")
    
    # Summary
    total_tests = total_count + compat_total + class_total
    total_success = success_count + compat_success + class_success
    
    print("\n" + "=" * 60)
    print("IMPORT VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Successful: {total_success}")
    print(f"Failed: {total_tests - total_success}")
    print(f"Success rate: {(total_success/total_tests)*100:.1f}%")
    
    if total_success == total_tests:
        print("\n🎉 ALL IMPORTS SUCCESSFUL!")
        return True
    else:
        print(f"\n❌ {total_tests - total_success} IMPORT(S) FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)