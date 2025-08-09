#!/usr/bin/env python3
"""Test the LSMTrainer constructor."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from lsm.training.train import LSMTrainer
    import inspect
    
    # Check constructor signature
    sig = inspect.signature(LSMTrainer.__init__)
    print("Constructor parameters:")
    for name, param in sig.parameters.items():
        print(f"  {name}: {param}")
    
    # Try to create instance
    print("\nTrying to create LSMTrainer with use_huggingface_data=False...")
    trainer = LSMTrainer(use_huggingface_data=False)
    print("✓ Success!")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()