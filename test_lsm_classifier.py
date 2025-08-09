#!/usr/bin/env python3
"""
Simple test script for LSMClassifier to verify the implementation works.
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lsm_classifier():
    """Test basic LSMClassifier functionality."""
    try:
        print("Testing LSMClassifier...")
        
        # Import the classifier
        from lsm.convenience import LSMClassifier
        print("✓ LSMClassifier imported successfully")
        
        # Create sample data
        X = [
            "This is a positive example",
            "This is another positive example", 
            "This is a negative example",
            "This is another negative example",
            "This is a neutral example",
            "This is another neutral example"
        ]
        y = [1, 1, 0, 0, 2, 2]  # 3 classes
        
        print(f"✓ Created sample data: {len(X)} samples, {len(set(y))} classes")
        
        # Create classifier
        classifier = LSMClassifier(
            window_size=5,
            embedding_dim=32,
            reservoir_type='standard',
            classifier_type='logistic',
            feature_extraction='mean',
            random_state=42
        )
        print("✓ LSMClassifier created successfully")
        
        # Test parameter validation
        params = classifier.get_params()
        print(f"✓ Parameters retrieved: {len(params)} parameters")
        
        # Test sklearn compatibility
        tags = classifier.__sklearn_tags__()
        print(f"✓ sklearn tags: {tags['requires_y']}, multiclass: {tags['multiclass']}")
        
        print("\nClassifier created successfully!")
        print("Note: Full training test requires TensorFlow and may take time.")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_classifier_validation():
    """Test parameter validation."""
    try:
        print("\nTesting parameter validation...")
        
        from lsm.convenience import LSMClassifier
        from lsm.convenience.config import ConvenienceValidationError
        
        # Test invalid classifier type
        try:
            classifier = LSMClassifier(classifier_type='invalid')
            print("✗ Should have failed with invalid classifier type")
            return False
        except ConvenienceValidationError:
            print("✓ Invalid classifier type properly rejected")
        
        # Test invalid feature extraction
        try:
            classifier = LSMClassifier(feature_extraction='invalid')
            print("✗ Should have failed with invalid feature extraction")
            return False
        except ConvenienceValidationError:
            print("✓ Invalid feature extraction properly rejected")
        
        # Test valid parameters
        classifier = LSMClassifier(
            classifier_type='random_forest',
            feature_extraction='last',
            n_classes=5
        )
        print("✓ Valid parameters accepted")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        return False

def test_preset_creation():
    """Test creating classifier from presets."""
    try:
        print("\nTesting preset creation...")
        
        from lsm.convenience import LSMClassifier
        
        # Test fast preset
        classifier = LSMClassifier.from_preset('fast')
        print("✓ Fast preset created")
        
        # Test with overrides
        classifier = LSMClassifier.from_preset(
            'balanced', 
            classifier_type='random_forest'
        )
        print("✓ Preset with overrides created")
        
        return True
        
    except Exception as e:
        print(f"✗ Preset test failed: {e}")
        return False

if __name__ == "__main__":
    print("LSMClassifier Implementation Test")
    print("=" * 40)
    
    success = True
    
    success &= test_lsm_classifier()
    success &= test_classifier_validation()
    success &= test_preset_creation()
    
    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed!")
        sys.exit(1)