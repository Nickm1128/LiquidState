#!/usr/bin/env python3
"""
Test the constructor step by step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing constructor...")

# Import all required components
from lsm.convenience.base import LSMBase
from sklearn.base import ClassifierMixin
from lsm.convenience.config import ConvenienceValidationError
from lsm.utils.input_validation import validate_positive_integer
from lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)

print("✓ All imports successful")

# Test the constructor logic step by step
try:
    print("Testing parameter validation...")
    
    # Test validate_positive_integer
    result = validate_positive_integer(10, 'test', min_value=1, max_value=100)
    print(f"✓ validate_positive_integer works: {result}")
    
    # Test ConvenienceValidationError
    try:
        raise ConvenienceValidationError("test error", suggestion="test suggestion")
    except ConvenienceValidationError as e:
        print("✓ ConvenienceValidationError works")
    
    print("✓ Parameter validation components work")
    
except Exception as e:
    print("✗ Parameter validation failed:", e)
    import traceback
    traceback.print_exc()

# Test class definition with minimal constructor
try:
    print("Testing minimal class with constructor...")
    
    class TestLSMClassifier(LSMBase, ClassifierMixin):
        def __init__(self, 
                     window_size: int = 10,
                     embedding_dim: int = 128,
                     classifier_type: str = 'logistic',
                     **kwargs):
            
            # Store parameters
            self.classifier_type = classifier_type
            
            # Initialize base class
            super().__init__(
                window_size=window_size,
                embedding_dim=embedding_dim,
                **kwargs
            )
            
            # sklearn-compatible attributes
            self.classes_ = None
            self.n_classes_ = None
            
            # Validate parameters
            valid_types = ['logistic', 'random_forest']
            if self.classifier_type not in valid_types:
                raise ConvenienceValidationError(
                    f"Invalid classifier_type: {self.classifier_type}",
                    valid_options=valid_types
                )
        
        def fit(self, X, y=None, **fit_params):
            return self
        
        def predict(self, X):
            return []
    
    print("✓ Class definition successful")
    
    # Test instantiation
    classifier = TestLSMClassifier()
    print("✓ Default instantiation successful")
    
    classifier = TestLSMClassifier(classifier_type='random_forest')
    print("✓ Custom instantiation successful")
    
    # Test invalid parameter
    try:
        classifier = TestLSMClassifier(classifier_type='invalid')
        print("✗ Should have failed with invalid parameter")
    except ConvenienceValidationError:
        print("✓ Invalid parameter properly rejected")
    
except Exception as e:
    print("✗ Class definition failed:", e)
    import traceback
    traceback.print_exc()

print("Constructor tests completed!")