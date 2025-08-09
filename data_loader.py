#!/usr/bin/env python3
"""
Backward compatibility shim for legacy data_loader module.

This module provides backward compatibility for code that imports from the legacy
'data_loader' module. All functionality has been moved to the src/lsm structure.

DEPRECATED: This module is deprecated. Please update your imports to use:
- from src.lsm.data.data_loader import load_data, DialogueTokenizer
"""

import warnings
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Issue deprecation warning
warnings.warn(
    "Importing from 'data_loader' module is deprecated. "
    "Please update your imports to use 'from src.lsm.data.data_loader import load_data, DialogueTokenizer'",
    DeprecationWarning,
    stacklevel=2
)

try:
    # Import from new locations
    from lsm.data.data_loader import load_data, DialogueTokenizer
    
    # Make available at module level for backward compatibility
    __all__ = ['load_data', 'DialogueTokenizer']
    
except ImportError as e:
    # Fallback error message
    raise ImportError(
        f"Failed to import from new module structure: {e}\n"
        "Please ensure the src/lsm package is properly installed and update your imports to:\n"
        "- from src.lsm.data.data_loader import load_data, DialogueTokenizer"
    ) from e