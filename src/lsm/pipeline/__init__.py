"""
Pipeline orchestration module for the LSM project.

This module provides the main pipeline coordinator that manages all components,
supports component swapping and experimentation, and handles configuration
management for different architectures.
"""

from .pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfiguration,
    ComponentRegistry,
    ComponentType,
    ArchitectureType,
    ComponentSpec,
    PipelineError,
    ComponentSwapError,
    ConfigurationError,
    create_pipeline_orchestrator,
    create_experimental_pipeline
)

from .colab_compatibility import (
    ColabCompatibilityManager,
    ColabEnvironmentInfo,
    ColabSetupConfig,
    ColabCompatibilityError,
    EnvironmentSetupError,
    DependencyInstallError,
    setup_colab_environment,
    quick_start_colab
)

__all__ = [
    'PipelineOrchestrator',
    'PipelineConfiguration',
    'ComponentRegistry',
    'ComponentType',
    'ArchitectureType',
    'ComponentSpec',
    'PipelineError',
    'ComponentSwapError',
    'ConfigurationError',
    'create_pipeline_orchestrator',
    'create_experimental_pipeline',
    'ColabCompatibilityManager',
    'ColabEnvironmentInfo',
    'ColabSetupConfig',
    'ColabCompatibilityError',
    'EnvironmentSetupError',
    'DependencyInstallError',
    'setup_colab_environment',
    'quick_start_colab'
]