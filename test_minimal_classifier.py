#!/usr/bin/env python3
"""
Test minimal classifier to debug the issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test step by step
print("Testing imports...")

try:
    from lsm.convenience.base import LSMBase
    print("✓ LSMBase imported")
except Exception as e:
    print("✗ LSMBase failed:", e)
    exit(1)

try:
    from sklearn.base import ClassifierMixin
    print("✓ ClassifierMixin imported")
except Exception as e:
    print("✗ ClassifierMixin failed:", e)
    exit(1)

# Test class definition
print("\nTesting class definition...")

try:
    class TestClassifier(LSMBase, ClassifierMixin):
        def __init__(self):
            super().__init__()
        
        def fit(self, X, y=None, **fit_params):
            return self
        
        def predict(self, X):
            return []
    
    print("✓ Test class defined successfully")
    
    # Test instantiation
    classifier = TestClassifier()
    print("✓ Test class instantiated successfully")
    
except Exception as e:
    print("✗ Test class failed:", e)
    import traceback
    traceback.print_exc()
    exit(1)

print("\nAll basic tests passed!")