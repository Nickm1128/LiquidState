#!/usr/bin/env python3
"""
Training wrapper that imports from the proper source structure.
This maintains backward compatibility with the root-level imports.
"""

# Import everything from the actual implementation
from src.lsm.training.train import (
    LSMTrainer, 
    set_random_seeds,
    run_training
)

# Re-export for backward compatibility
__all__ = ['LSMTrainer', 'set_random_seeds', 'run_training']