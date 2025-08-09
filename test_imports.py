#!/usr/bin/env python3
"""
Test script to verify LSM imports work correctly.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

print("Testing LSM imports...")

try:
    from lsm.core import ReservoirLayer, build_reservoir
    print("✅ Core imports successful")
    print(f"   ReservoirLayer: {ReservoirLayer}")
    print(f"   build_reservoir: {build_reservoir}")
except ImportError as e:
    print(f"❌ Core import failed: {e}")

try:
    from lsm.training import LSMTrainer, ModelConfiguration
    print("✅ Training imports successful")
    print(f"   LSMTrainer: {LSMTrainer}")
    print(f"   ModelConfiguration: {ModelConfiguration}")
except ImportError as e:
    print(f"❌ Training import failed: {e}")

try:
    from lsm.inference import OptimizedLSMInference
    print("✅ Inference imports successful")
    print(f"   OptimizedLSMInference: {OptimizedLSMInference}")
except ImportError as e:
    print(f"❌ Inference import failed: {e}")

try:
    from lsm.data import DialogueTokenizer, load_data
    print("✅ Data imports successful")
    print(f"   DialogueTokenizer: {DialogueTokenizer}")
    print(f"   load_data: {load_data}")
except ImportError as e:
    print(f"❌ Data import failed: {e}")

print("\nAll import tests completed!")