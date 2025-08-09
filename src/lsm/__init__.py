"""
Sparse Sine-Activated Liquid State Machine (LSM) Package

This package provides a complete implementation of a Liquid State Machine
with sparse connectivity and sine activation functions for neural computation.

The package is organized into the following modules:
- core: Basic reservoir components and advanced architectures
- data: Data loading and preprocessing utilities
- training: Training pipeline and configuration management
- inference: Inference engines for trained models
- management: Model management and discovery utilities
- utils: Utility functions, exceptions, logging, and validation

For backward compatibility, commonly used classes are available at the package level.
"""

import sys
import os

# Import all subpackage contents with error handling
# Some modules may require TensorFlow which might not be available

# Always available modules (no TensorFlow dependency)
from .data import *
from .management import *
from .utils import *
from .pipeline import *

# Training module (handles TensorFlow gracefully)
from .training import *

# Convenience API (handles TensorFlow gracefully)
try:
    from .convenience import *
    _CONVENIENCE_AVAILABLE = True
except ImportError as e:
    _CONVENIENCE_AVAILABLE = False
    # Create placeholder for LSMGenerator if not available
    class _ConveniencePlaceholder:
        """Placeholder class for when convenience API cannot be imported."""
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Convenience API is not available. This may be due to missing TensorFlow "
                "or other dependencies. Please ensure all required dependencies are installed."
            )
    
    LSMGenerator = _ConveniencePlaceholder
    LSMClassifier = _ConveniencePlaceholder
    LSMRegressor = _ConveniencePlaceholder

# Core module (requires TensorFlow)
try:
    from .core import *
    _CORE_AVAILABLE = True
except ImportError as e:
    _CORE_AVAILABLE = False
    # Define placeholder variables for missing core components
    SparseDense = None
    ParametricSineActivation = None
    ReservoirLayer = None

# Handle inference imports (may not exist yet if task 5 isn't complete)
_INFERENCE_AVAILABLE = False
OptimizedLSMInference = None
LSMInference = None

try:
    from .inference import *
    _INFERENCE_AVAILABLE = True
except ImportError:
    # If inference package doesn't exist yet, try importing from root level for compatibility
    # Add root directory to path temporarily
    root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    if root_path not in sys.path:
        sys.path.insert(0, root_path)
    
    try:
        from inference import OptimizedLSMInference, LSMInference
        _INFERENCE_AVAILABLE = True
    except ImportError:
        # If inference.py can't be imported (e.g., TensorFlow issues), create placeholder classes
        # This maintains backward compatibility even when TensorFlow is not available
        class _InferencePlaceholder:
            """Placeholder class for when inference modules cannot be imported."""
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Inference classes are not available. This may be due to missing TensorFlow "
                    "or other dependencies. Please ensure all required dependencies are installed."
                )
        
        OptimizedLSMInference = _InferencePlaceholder
        LSMInference = _InferencePlaceholder
    finally:
        # Remove the temporary path addition
        if root_path in sys.path:
            sys.path.remove(root_path)

# Backward compatibility aliases for commonly used classes
# These maintain the old import patterns that users might be using

# Core components - most commonly used (may be None if TensorFlow not available)
if _CORE_AVAILABLE:
    ReservoirLayer = ReservoirLayer  # Already imported from core
    SparseDense = SparseDense  # Already imported from core
    ParametricSineActivation = ParametricSineActivation  # Already imported from core

# Data processing
DialogueTokenizer = DialogueTokenizer  # Already imported from data
load_data = load_data  # Already imported from data

# Training system
LSMTrainer = LSMTrainer if 'LSMTrainer' in globals() else None  # May not be available without TensorFlow
ModelConfiguration = ModelConfiguration  # Already imported from training

# Model management
ModelManager = ModelManager  # Already imported from management

# Common exceptions (already imported from utils)
# LSMError, ModelError, etc. are already available

# Inference classes (handled above with try/except)

# Create a comprehensive __all__ list
__all__ = [
    # Re-export everything from available subpackages
    *getattr(sys.modules.get(f'{__name__}.data', type('', (), {})()), '__all__', []),
    *getattr(sys.modules.get(f'{__name__}.training', type('', (), {})()), '__all__', []),
    *getattr(sys.modules.get(f'{__name__}.management', type('', (), {})()), '__all__', []),
    *getattr(sys.modules.get(f'{__name__}.utils', type('', (), {})()), '__all__', []),
    *getattr(sys.modules.get(f'{__name__}.pipeline', type('', (), {})()), '__all__', []),
]

# Add core exports if available
if _CORE_AVAILABLE:
    __all__.extend(getattr(sys.modules.get(f'{__name__}.core', type('', (), {})()), '__all__', []))

# Add convenience API exports if available
if _CONVENIENCE_AVAILABLE:
    __all__.extend(getattr(sys.modules.get(f'{__name__}.convenience', type('', (), {})()), '__all__', []))
else:
    # Always include convenience class names for backward compatibility
    __all__.extend(['LSMGenerator', 'LSMClassifier', 'LSMRegressor'])

# Add inference exports (always include for backward compatibility)
try:
    inference_module = sys.modules.get(f'{__name__}.inference')
    if inference_module and hasattr(inference_module, '__all__'):
        __all__.extend(inference_module.__all__)
    else:
        # Always include inference class names for backward compatibility
        __all__.extend(['OptimizedLSMInference', 'LSMInference'])
except:
    __all__.extend(['OptimizedLSMInference', 'LSMInference'])

# Remove duplicates while preserving order
seen = set()
__all__ = [x for x in __all__ if not (x in seen or seen.add(x))]

# Package metadata
__version__ = "1.0.0"
__author__ = "LSM Development Team"
__description__ = "Sparse Sine-Activated Liquid State Machine Implementation"