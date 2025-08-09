"""
Inference module for the Sparse Sine-Activated LSM.

This module provides optimized inference engines for trained LSM models,
including performance optimizations like caching, lazy loading, and batch processing.
It also includes response-level generation capabilities for complete response inference
and comprehensive reservoir management.

Classes:
    OptimizedLSMInference: Performance-optimized inference with caching and memory management
    LSMInference: Legacy inference class for backward compatibility
    ResponseGenerator: Main orchestrator for response-level inference
    ResponseInferenceModel: Secondary model for complete response prediction from embeddings
    ReservoirManager: Manages reservoir strategy decisions and multiple reservoir instances
    TokenEmbeddingSequence: Container for token embedding sequences
    ResponseGenerationResult: Result of response generation
    ResponsePredictionResult: Result of response prediction from inference model
    ReservoirInstance: Container for reservoir instance with metadata
    ReservoirOutput: Container for reservoir processing output
    ReservoirCoordinationResult: Result of coordinating multiple reservoir outputs
    TrainingConfig: Configuration for response-level training
    ModelArchitecture: Enumeration of supported model architectures

The module handles model loading, prediction, and provides both interactive and
programmatic interfaces for inference operations, including complete response generation,
response-level learning from token embedding sequences, and intelligent reservoir management.
"""

from .inference import OptimizedLSMInference, LSMInference
from .response_generator import (
    ResponseGenerator, 
    TokenEmbeddingSequence, 
    ResponseGenerationResult,
    ReservoirStrategy,
    create_response_generator,
    create_system_aware_response_generator
)
from .response_inference_model import (
    ResponseInferenceModel,
    ResponsePredictionResult,
    TrainingConfig,
    ModelArchitecture,
    create_response_inference_model,
    create_transformer_response_model,
    create_lstm_response_model
)
from .reservoir_manager import (
    ReservoirManager,
    ReservoirInstance,
    ReservoirOutput,
    ReservoirCoordinationResult,
    ReservoirType,
    ReservoirManagerError,
    create_reservoir_manager,
    create_high_performance_reservoir_manager
)

__all__ = [
    'OptimizedLSMInference',
    'LSMInference',
    'ResponseGenerator',
    'TokenEmbeddingSequence',
    'ResponseGenerationResult',
    'ReservoirStrategy',
    'create_response_generator',
    'create_system_aware_response_generator',
    'ResponseInferenceModel',
    'ResponsePredictionResult',
    'TrainingConfig',
    'ModelArchitecture',
    'create_response_inference_model',
    'create_transformer_response_model',
    'create_lstm_response_model',
    'ReservoirManager',
    'ReservoirInstance',
    'ReservoirOutput',
    'ReservoirCoordinationResult',
    'ReservoirType',
    'ReservoirManagerError',
    'create_reservoir_manager',
    'create_high_performance_reservoir_manager'
]