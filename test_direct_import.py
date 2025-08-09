#!/usr/bin/env python3
"""Test direct import of LSMTrainer."""

import importlib.util
import sys

# Load the module directly from file
spec = importlib.util.spec_from_file_location("train", "src/lsm/training/train.py")
train_module = importlib.util.module_from_spec(spec)
sys.modules["train"] = train_module
spec.loader.exec_module(train_module)

# Get the LSMTrainer class
LSMTrainer = train_module.LSMTrainer

# Check constructor
import inspect
sig = inspect.signature(LSMTrainer.__init__)
print("Constructor parameters:")
for name, param in sig.parameters.items():
    print(f"  {name}: {param}")

# Try to create instance
print("\nTrying to create LSMTrainer with use_huggingface_data=False...")
try:
    trainer = LSMTrainer(use_huggingface_data=False)
    print("✓ Success!")
except Exception as e:
    print(f"✗ Error: {e}")