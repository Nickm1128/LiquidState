#!/usr/bin/env python3
"""
Backward compatibility shim for legacy train module.

This module provides backward compatibility for code that imports from the legacy
'train' module. All functionality has been moved to the src/lsm structure.

DEPRECATED: This module is deprecated. Please update your imports to use:
- from src.lsm.training.train import LSMTrainer, run_training
- from src.lsm.utils.random_utils import set_random_seeds
"""

import warnings
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Issue deprecation warning
warnings.warn(
    "Importing from 'train' module is deprecated. "
    "Please update your imports to use 'from src.lsm.training.train import LSMTrainer, run_training'",
    DeprecationWarning,
    stacklevel=2
)

try:
    # Import from new locations
    from lsm.training.train import LSMTrainer, run_training
    from lsm.utils.random_utils import set_random_seeds
    
    # Make available at module level for backward compatibility
    __all__ = ['LSMTrainer', 'run_training', 'set_random_seeds']
    
except ImportError as e:
    # Fallback error message
    raise ImportError(
        f"Failed to import from new module structure: {e}\n"
        "Please ensure the src/lsm package is properly installed and update your imports to:\n"
        "- from src.lsm.training.train import LSMTrainer, run_training\n"
        "- from src.lsm.utils.random_utils import set_random_seeds"
    ) from e