"""
Model management utilities for the LSM project.

This package provides tools for discovering, validating, and managing
trained LSM models in the workspace.
"""

from .model_manager import ModelManager
from .manage_models import main as manage_models_main

__all__ = [
    'ModelManager',
    'manage_models_main'
]