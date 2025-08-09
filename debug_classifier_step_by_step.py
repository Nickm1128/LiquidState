#!/usr/bin/env python3
"""
Debug classifier step by step to find the issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing classifier imports step by step...")

# Test each import from the classifier file
imports_to_test = [
    "import time",
    "import warnings", 
    "from typing import Any, Dict, List, Optional, Union, Tuple",
    "import numpy as np",
    "from pathlib import Path",
    "import json",
    "import pickle",
    "from sklearn.base import ClassifierMixin",
    "from sklearn.metrics import accuracy_score",
    "from sklearn.preprocessing import LabelEncoder",
    "from sklearn.linear_model import LogisticRegression",
    "from sklearn.ensemble import RandomForestClassifier",
    "from sklearn.model_selection import train_test_split",
    "from lsm.convenience.base import LSMBase",
    "from lsm.convenience.config import ConvenienceConfig, ConvenienceValidationError",
    "from lsm.utils.lsm_exceptions import LSMError, TrainingSetupError, TrainingExecutionError, InvalidInputError, ModelLoadError",
    "from lsm.utils.input_validation import validate_positive_integer, validate_positive_float, validate_string_list, create_helpful_error_message",
    "from lsm.utils.lsm_logging import get_logger"
]

namespace = {}

for i, import_stmt in enumerate(imports_to_test):
    try:
        exec(import_stmt, namespace)
        print(f"✓ {i+1:2d}. {import_stmt}")
    except Exception as e:
        print(f"✗ {i+1:2d}. {import_stmt}")
        print(f"    Error: {e}")
        break

print(f"\nSuccessfully imported {len([x for x in imports_to_test if True])} statements")

# Test the training components import
print("\nTesting training components import...")
try:
    exec("from lsm.training.train import LSMTrainer", namespace)
    print("✓ LSMTrainer imported")
except Exception as e:
    print("✗ LSMTrainer failed:", e)

try:
    exec("from lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder", namespace)
    print("✓ Tokenization components imported")
except Exception as e:
    print("✗ Tokenization components failed:", e)

# Test logger creation
print("\nTesting logger creation...")
try:
    exec("logger = get_logger(__name__)", namespace)
    print("✓ Logger created")
except Exception as e:
    print("✗ Logger creation failed:", e)

print("\nAll import tests completed!")