#!/usr/bin/env python3
"""
Model configuration wrapper that imports from the proper source structure.
This maintains backward compatibility with the root-level imports.
"""

# Import everything from the actual implementation
from src.lsm.training.model_config import (
    ModelConfiguration,
    TrainingMetadata
)

# Re-export for backward compatibility
__all__ = ['ModelConfiguration', 'TrainingMetadata']