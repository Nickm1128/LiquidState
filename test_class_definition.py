#!/usr/bin/env python3
"""
Test class definition step by step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing class definition...")

# Import required components
from lsm.convenience.base import LSMBase
from sklearn.base import ClassifierMixin
from lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)

print("✓ All imports successful")

# Test minimal class definition
try:
    class TestClassifier(LSMBase, ClassifierMixin):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
        
        def fit(self, X, y=None, **fit_params):
            return self
        
        def predict(self, X):
            return []
    
    print("✓ Minimal class definition works")
    
    # Test instantiation
    classifier = TestClassifier()
    print("✓ Class instantiation works")
    
except Exception as e:
    print("✗ Class definition failed:", e)
    import traceback
    traceback.print_exc()

# Test more complex class definition
try:
    class ComplexTestClassifier(LSMBase, ClassifierMixin):
        def __init__(self, 
                     window_size: int = 10,
                     embedding_dim: int = 128,
                     **kwargs):
            super().__init__(
                window_size=window_size,
                embedding_dim=embedding_dim,
                **kwargs
            )
            self.classes_ = None
        
        def fit(self, X, y=None, **fit_params):
            self._is_fitted = True
            return self
        
        def predict(self, X):
            self._check_is_fitted()
            return []
        
        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.update({'requires_y': True})
            return tags
    
    print("✓ Complex class definition works")
    
    # Test instantiation
    classifier = ComplexTestClassifier()
    print("✓ Complex class instantiation works")
    
except Exception as e:
    print("✗ Complex class definition failed:", e)
    import traceback
    traceback.print_exc()

print("Class definition tests completed!")