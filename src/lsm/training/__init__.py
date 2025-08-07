"""
Training system components for the Sparse Sine-Activated LSM.

This module contains the training pipeline, model configuration management,
and related utilities for training LSM models.
"""

# Import configuration classes first (no TensorFlow dependency)
from .model_config import ModelConfiguration, TrainingMetadata

# Import training classes (requires TensorFlow)
try:
    from .train import LSMTrainer, run_training
    _TRAINING_AVAILABLE = True
except ImportError as e:
    # Handle cases where TensorFlow is not available
    _TRAINING_AVAILABLE = False
    LSMTrainer = None
    run_training = None

__all__ = [
    'ModelConfiguration', 
    'TrainingMetadata'
]

if _TRAINING_AVAILABLE:
    __all__.extend(['LSMTrainer', 'run_training'])