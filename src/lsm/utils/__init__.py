"""
Utility modules for the LSM project.

This package contains utility functions and classes for:
- Exception handling and custom exceptions
- Logging infrastructure
- Input validation
- Production validation and testing
"""

from .lsm_exceptions import *
from .lsm_logging import get_logger, setup_logging, log_performance, log_function_call
from .input_validation import (
    validate_file_path,
    validate_directory_path,
    validate_positive_integer,
    validate_positive_float,
    validate_string_list,
    validate_numpy_array,
    validate_dialogue_sequence,
    validate_model_configuration,
    validate_training_parameters,
    validate_json_file,
    create_helpful_error_message,
    validate_memory_requirements,
    validate_disk_space,
    validate_inputs
)
from .production_validation import ProductionValidator

__all__ = [
    # Exceptions
    'LSMError', 'ModelError', 'ModelLoadError', 'ModelSaveError', 'ModelValidationError',
    'ConfigurationError', 'InvalidConfigurationError', 'MissingConfigurationError',
    'TokenizerError', 'TokenizerNotFittedError', 'TokenizerLoadError', 'TokenizerSaveError',
    'InferenceError', 'InvalidInputError', 'PredictionError',
    'DataError', 'DataLoadError', 'DataValidationError',
    'TrainingError', 'TrainingSetupError', 'TrainingExecutionError',
    'ResourceError', 'InsufficientMemoryError', 'DiskSpaceError',
    'BackwardCompatibilityError', 'UnsupportedModelVersionError', 'MigrationError',
    
    # Logging
    'get_logger', 'setup_logging', 'log_performance', 'log_function_call',
    
    # Validation
    'validate_file_path', 'validate_directory_path', 'validate_positive_integer',
    'validate_positive_float', 'validate_string_list', 'validate_numpy_array',
    'validate_dialogue_sequence', 'validate_model_configuration',
    'validate_training_parameters', 'validate_json_file', 'create_helpful_error_message',
    'validate_memory_requirements', 'validate_disk_space', 'validate_inputs',
    
    # Production validation
    'ProductionValidator'
]