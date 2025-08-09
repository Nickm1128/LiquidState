#!/usr/bin/env python3
"""
Debug classifier imports step by step.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports step by step...")

try:
    import time
    print("✓ time")
except Exception as e:
    print("✗ time:", e)

try:
    import warnings
    print("✓ warnings")
except Exception as e:
    print("✗ warnings:", e)

try:
    from typing import Any, Dict, List, Optional, Union, Tuple
    print("✓ typing")
except Exception as e:
    print("✗ typing:", e)

try:
    import numpy as np
    print("✓ numpy")
except Exception as e:
    print("✗ numpy:", e)

try:
    from pathlib import Path
    print("✓ pathlib")
except Exception as e:
    print("✗ pathlib:", e)

try:
    import json
    print("✓ json")
except Exception as e:
    print("✗ json:", e)

try:
    import pickle
    print("✓ pickle")
except Exception as e:
    print("✗ pickle:", e)

try:
    from sklearn.base import ClassifierMixin
    print("✓ sklearn.base.ClassifierMixin")
except Exception as e:
    print("✗ sklearn.base.ClassifierMixin:", e)

try:
    from sklearn.metrics import accuracy_score
    print("✓ sklearn.metrics.accuracy_score")
except Exception as e:
    print("✗ sklearn.metrics.accuracy_score:", e)

try:
    from sklearn.preprocessing import LabelEncoder
    print("✓ sklearn.preprocessing.LabelEncoder")
except Exception as e:
    print("✗ sklearn.preprocessing.LabelEncoder:", e)

try:
    from sklearn.linear_model import LogisticRegression
    print("✓ sklearn.linear_model.LogisticRegression")
except Exception as e:
    print("✗ sklearn.linear_model.LogisticRegression:", e)

try:
    from sklearn.ensemble import RandomForestClassifier
    print("✓ sklearn.ensemble.RandomForestClassifier")
except Exception as e:
    print("✗ sklearn.ensemble.RandomForestClassifier:", e)

try:
    from sklearn.model_selection import train_test_split
    print("✓ sklearn.model_selection.train_test_split")
except Exception as e:
    print("✗ sklearn.model_selection.train_test_split:", e)

try:
    from lsm.convenience.base import LSMBase
    print("✓ lsm.convenience.base.LSMBase")
except Exception as e:
    print("✗ lsm.convenience.base.LSMBase:", e)

try:
    from lsm.convenience.config import ConvenienceConfig, ConvenienceValidationError
    print("✓ lsm.convenience.config")
except Exception as e:
    print("✗ lsm.convenience.config:", e)

try:
    from lsm.utils.lsm_exceptions import (
        LSMError, TrainingSetupError, TrainingExecutionError, 
        InvalidInputError, ModelLoadError
    )
    print("✓ lsm.utils.lsm_exceptions")
except Exception as e:
    print("✗ lsm.utils.lsm_exceptions:", e)

try:
    from lsm.utils.input_validation import (
        validate_positive_integer, validate_positive_float,
        validate_string_list, create_helpful_error_message
    )
    print("✓ lsm.utils.input_validation")
except Exception as e:
    print("✗ lsm.utils.input_validation:", e)

try:
    from lsm.utils.lsm_logging import get_logger
    print("✓ lsm.utils.lsm_logging")
except Exception as e:
    print("✗ lsm.utils.lsm_logging:", e)

print("\nAll imports tested!")