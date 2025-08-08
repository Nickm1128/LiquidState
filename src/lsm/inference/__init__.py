"""
Inference module for the Sparse Sine-Activated LSM.

This module provides optimized inference engines for trained LSM models,
including performance optimizations like caching, lazy loading, and batch processing.

Classes:
    OptimizedLSMInference: Performance-optimized inference with caching and memory management
    LSMInference: Legacy inference class for backward compatibility

The module handles model loading, prediction, and provides both interactive and
programmatic interfaces for inference operations.
"""

from .inference import OptimizedLSMInference, LSMInference

__all__ = [
    'OptimizedLSMInference',
    'LSMInference'
]