#!/usr/bin/env python3
"""
Input validation utilities for the Sparse Sine-Activated LSM project.

This module provides comprehensive input validation functions with helpful
error messages for all public methods in the LSM system.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import re

from lsm_exceptions import (
    InvalidInputError, DataValidationError, ConfigurationError,
    InvalidConfigurationError, MissingConfigurationError
)
from lsm_logging import get_logger

logger = get_logger(__name__)


def validate_file_path(file_path: str, must_exist: bool = True, 
                      must_be_file: bool = True, description: str = "file") -> None:
    """
    Validate file path with helpful error messages.
    
    Args:
        file_path: Path to validate
        must_exist: Whether the file must exist
        must_be_file: Whether the path must be a file (not directory)
        description: Description of the file for error messages
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(file_path, str):
        raise InvalidInputError(
            f"{description} path",
            "string path",
            f"{type(file_path).__name__}"
        )
    
    if not file_path.strip():
        raise InvalidInputError(
            f"{description} path",
            "non-empty string",
            "empty string"
        )
    
    if must_exist and not os.path.exists(file_path):
        raise InvalidInputError(
            f"{description} path",
            f"existing path",
            f"non-existent path '{file_path}'"
        )
    
    if must_exist and must_be_file and not os.path.isfile(file_path):
        if os.path.isdir(file_path):
            raise InvalidInputError(
                f"{description} path",
                "file path",
                f"directory path '{file_path}'"
            )
        else:
            raise InvalidInputError(
                f"{description} path",
                "file path",
                f"invalid file path '{file_path}'"
            )


def validate_directory_path(dir_path: str, must_exist: bool = True, 
                           create_if_missing: bool = False, description: str = "directory") -> None:
    """
    Validate directory path with optional creation.
    
    Args:
        dir_path: Directory path to validate
        must_exist: Whether the directory must exist
        create_if_missing: Whether to create the directory if it doesn't exist
        description: Description of the directory for error messages
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(dir_path, str):
        raise InvalidInputError(
            f"{description} path",
            "string path",
            f"{type(dir_path).__name__}"
        )
    
    if not dir_path.strip():
        raise InvalidInputError(
            f"{description} path",
            "non-empty string",
            "empty string"
        )
    
    if not os.path.exists(dir_path):
        if must_exist and not create_if_missing:
            raise InvalidInputError(
                f"{description} path",
                "existing directory",
                f"non-existent directory '{dir_path}'"
            )
        elif create_if_missing:
            try:
                os.makedirs(dir_path, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            except OSError as e:
                raise InvalidInputError(
                    f"{description} path",
                    "writable directory path",
                    f"path that cannot be created: {e}"
                )
    elif not os.path.isdir(dir_path):
        raise InvalidInputError(
            f"{description} path",
            "directory path",
            f"file path '{dir_path}'"
        )


def validate_positive_integer(value: Any, name: str, min_value: int = 1, 
                             max_value: Optional[int] = None) -> int:
    """
    Validate positive integer with bounds checking.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value (if specified)
        
    Returns:
        Validated integer value
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(value, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        else:
            raise InvalidInputError(
                name,
                "integer",
                f"{type(value).__name__} ({value})"
            )
    
    if value < min_value:
        raise InvalidInputError(
            name,
            f"integer >= {min_value}",
            f"{value}"
        )
    
    if max_value is not None and value > max_value:
        raise InvalidInputError(
            name,
            f"integer <= {max_value}",
            f"{value}"
        )
    
    return value


def validate_positive_float(value: Any, name: str, min_value: float = 0.0, 
                           max_value: Optional[float] = None, 
                           exclude_zero: bool = False) -> float:
    """
    Validate positive float with bounds checking.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value (if specified)
        exclude_zero: Whether to exclude zero from valid values
        
    Returns:
        Validated float value
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise InvalidInputError(
            name,
            "number",
            f"{type(value).__name__} ({value})"
        )
    
    value = float(value)
    
    if exclude_zero and value == 0.0:
        raise InvalidInputError(
            name,
            f"non-zero number",
            "0.0"
        )
    
    if value < min_value:
        raise InvalidInputError(
            name,
            f"number >= {min_value}",
            f"{value}"
        )
    
    if max_value is not None and value > max_value:
        raise InvalidInputError(
            name,
            f"number <= {max_value}",
            f"{value}"
        )
    
    return value


def validate_string_list(value: Any, name: str, min_length: int = 0, 
                        max_length: Optional[int] = None, 
                        allow_empty_strings: bool = False) -> List[str]:
    """
    Validate list of strings.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        min_length: Minimum list length
        max_length: Maximum list length (if specified)
        allow_empty_strings: Whether to allow empty strings in the list
        
    Returns:
        Validated list of strings
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(value, list):
        raise InvalidInputError(
            name,
            "list of strings",
            f"{type(value).__name__}"
        )
    
    if len(value) < min_length:
        raise InvalidInputError(
            name,
            f"list with at least {min_length} items",
            f"list with {len(value)} items"
        )
    
    if max_length is not None and len(value) > max_length:
        raise InvalidInputError(
            name,
            f"list with at most {max_length} items",
            f"list with {len(value)} items"
        )
    
    for i, item in enumerate(value):
        if not isinstance(item, str):
            raise InvalidInputError(
                f"{name}[{i}]",
                "string",
                f"{type(item).__name__}"
            )
        
        if not allow_empty_strings and not item.strip():
            raise InvalidInputError(
                f"{name}[{i}]",
                "non-empty string",
                "empty or whitespace-only string"
            )
    
    return value


def validate_numpy_array(value: Any, name: str, expected_shape: Optional[Tuple] = None,
                        expected_dtype: Optional[np.dtype] = None,
                        min_dimensions: int = 1, max_dimensions: Optional[int] = None) -> np.ndarray:
    """
    Validate numpy array with shape and dtype checking.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error messages
        expected_shape: Expected array shape (None for any shape)
        expected_dtype: Expected data type
        min_dimensions: Minimum number of dimensions
        max_dimensions: Maximum number of dimensions
        
    Returns:
        Validated numpy array
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(value, np.ndarray):
        try:
            value = np.array(value)
        except Exception as e:
            raise InvalidInputError(
                name,
                "numpy array or array-like",
                f"{type(value).__name__} (conversion failed: {e})"
            )
    
    if value.ndim < min_dimensions:
        raise InvalidInputError(
            name,
            f"array with at least {min_dimensions} dimensions",
            f"array with {value.ndim} dimensions"
        )
    
    if max_dimensions is not None and value.ndim > max_dimensions:
        raise InvalidInputError(
            name,
            f"array with at most {max_dimensions} dimensions",
            f"array with {value.ndim} dimensions"
        )
    
    if expected_shape is not None:
        # Allow None in expected_shape to mean "any size for this dimension"
        if len(expected_shape) != value.ndim:
            raise InvalidInputError(
                name,
                f"array with {len(expected_shape)} dimensions",
                f"array with {value.ndim} dimensions"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, value.shape)):
            if expected is not None and expected != actual:
                raise InvalidInputError(
                    name,
                    f"array with shape {expected_shape}",
                    f"array with shape {value.shape}"
                )
    
    if expected_dtype is not None and value.dtype != expected_dtype:
        try:
            value = value.astype(expected_dtype)
        except Exception as e:
            raise InvalidInputError(
                name,
                f"array with dtype {expected_dtype}",
                f"array with dtype {value.dtype} (conversion failed: {e})"
            )
    
    return value


def validate_dialogue_sequence(sequence: Any, expected_length: int, name: str = "dialogue sequence") -> List[str]:
    """
    Validate dialogue sequence for inference.
    
    Args:
        sequence: Sequence to validate
        expected_length: Expected sequence length
        name: Name for error messages
        
    Returns:
        Validated dialogue sequence
        
    Raises:
        InvalidInputError: If validation fails
    """
    if not isinstance(sequence, list):
        raise InvalidInputError(
            name,
            "list of strings",
            f"{type(sequence).__name__}"
        )
    
    if len(sequence) != expected_length:
        raise InvalidInputError(
            name,
            f"list with exactly {expected_length} dialogue turns",
            f"list with {len(sequence)} turns"
        )
    
    for i, turn in enumerate(sequence):
        if not isinstance(turn, str):
            raise InvalidInputError(
                f"{name}[{i}]",
                "string",
                f"{type(turn).__name__}"
            )
        
        if not turn.strip():
            raise InvalidInputError(
                f"{name}[{i}]",
                "non-empty dialogue turn",
                "empty or whitespace-only string"
            )
        
        # Check for reasonable length (not too short or too long)
        if len(turn.strip()) < 2:
            raise InvalidInputError(
                f"{name}[{i}]",
                "dialogue turn with at least 2 characters",
                f"turn with {len(turn.strip())} characters: '{turn.strip()}'"
            )
        
        if len(turn) > 1000:  # Reasonable upper limit
            raise InvalidInputError(
                f"{name}[{i}]",
                "dialogue turn with at most 1000 characters",
                f"turn with {len(turn)} characters"
            )
    
    return sequence


def validate_model_configuration(config: Dict[str, Any]) -> List[str]:
    """
    Validate model configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required fields
    required_fields = {
        'window_size': int,
        'embedding_dim': int,
        'reservoir_type': str
    }
    
    for field, expected_type in required_fields.items():
        if field not in config:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(config[field], expected_type):
            errors.append(f"Field '{field}' must be {expected_type.__name__}, got {type(config[field]).__name__}")
    
    # Validate specific fields
    if 'window_size' in config:
        try:
            validate_positive_integer(config['window_size'], 'window_size', min_value=1, max_value=100)
        except InvalidInputError as e:
            errors.append(f"Invalid window_size: {e.message}")
    
    if 'embedding_dim' in config:
        try:
            validate_positive_integer(config['embedding_dim'], 'embedding_dim', min_value=1, max_value=2048)
        except InvalidInputError as e:
            errors.append(f"Invalid embedding_dim: {e.message}")
    
    if 'reservoir_type' in config:
        valid_types = ['standard', 'hierarchical', 'attentive', 'echo_state', 'deep']
        if config['reservoir_type'] not in valid_types:
            errors.append(f"Invalid reservoir_type: must be one of {valid_types}")
    
    if 'sparsity' in config:
        try:
            validate_positive_float(config['sparsity'], 'sparsity', min_value=0.0, max_value=1.0)
        except InvalidInputError as e:
            errors.append(f"Invalid sparsity: {e.message}")
    
    return errors


def validate_training_parameters(params: Dict[str, Any]) -> List[str]:
    """
    Validate training parameters.
    
    Args:
        params: Training parameters dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate epochs
    if 'epochs' in params:
        try:
            validate_positive_integer(params['epochs'], 'epochs', min_value=1, max_value=1000)
        except InvalidInputError as e:
            errors.append(f"Invalid epochs: {e.message}")
    
    # Validate batch_size
    if 'batch_size' in params:
        try:
            validate_positive_integer(params['batch_size'], 'batch_size', min_value=1, max_value=1024)
        except InvalidInputError as e:
            errors.append(f"Invalid batch_size: {e.message}")
    
    # Validate test_size
    if 'test_size' in params:
        try:
            validate_positive_float(params['test_size'], 'test_size', min_value=0.01, max_value=0.9)
        except InvalidInputError as e:
            errors.append(f"Invalid test_size: {e.message}")
    
    # Validate validation_split
    if 'validation_split' in params:
        try:
            validate_positive_float(params['validation_split'], 'validation_split', min_value=0.0, max_value=0.5)
        except InvalidInputError as e:
            errors.append(f"Invalid validation_split: {e.message}")
    
    return errors


def validate_json_file(file_path: str, required_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Validate and load JSON file.
    
    Args:
        file_path: Path to JSON file
        required_keys: List of required keys in the JSON
        
    Returns:
        Loaded JSON data
        
    Raises:
        InvalidInputError: If validation fails
        InvalidConfigurationError: If JSON is invalid
    """
    validate_file_path(file_path, must_exist=True, description="JSON file")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise InvalidConfigurationError(file_path, [f"Invalid JSON format: {e}"])
    except Exception as e:
        raise InvalidInputError(
            "JSON file",
            "readable JSON file",
            f"file that cannot be read: {e}"
        )
    
    if not isinstance(data, dict):
        raise InvalidConfigurationError(file_path, ["JSON must contain an object, not array or primitive"])
    
    if required_keys:
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise MissingConfigurationError(file_path, missing_keys)
    
    return data


def create_helpful_error_message(operation: str, error: Exception, suggestions: Optional[List[str]] = None) -> str:
    """
    Create a helpful error message with suggestions.
    
    Args:
        operation: Description of the operation that failed
        error: The original error
        suggestions: List of suggestions to fix the error
        
    Returns:
        Formatted error message with suggestions
    """
    message_parts = [
        f"❌ {operation} failed:",
        f"   Error: {error}",
    ]
    
    if suggestions:
        message_parts.append("   Suggestions:")
        for suggestion in suggestions:
            message_parts.append(f"   • {suggestion}")
    
    return "\n".join(message_parts)


def validate_memory_requirements(operation: str, estimated_mb: float, safety_factor: float = 1.5) -> None:
    """
    Validate that sufficient memory is available for an operation.
    
    Args:
        operation: Description of the operation
        estimated_mb: Estimated memory requirement in MB
        safety_factor: Safety factor to apply to the estimate
        
    Raises:
        InsufficientMemoryError: If insufficient memory is available
    """
    try:
        import psutil
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        required_mb = estimated_mb * safety_factor
        
        if available_mb < required_mb:
            from lsm_exceptions import InsufficientMemoryError
            raise InsufficientMemoryError(operation, required_mb, available_mb)
        
        logger.debug(f"Memory check passed for {operation}", 
                    required_mb=required_mb, 
                    available_mb=available_mb)
        
    except ImportError:
        logger.warning("psutil not available for memory checking")
    except Exception as e:
        logger.warning(f"Memory check failed: {e}")


def validate_disk_space(path: str, estimated_mb: float, safety_factor: float = 1.2) -> None:
    """
    Validate that sufficient disk space is available.
    
    Args:
        path: Path where space is needed
        estimated_mb: Estimated space requirement in MB
        safety_factor: Safety factor to apply to the estimate
        
    Raises:
        DiskSpaceError: If insufficient disk space is available
    """
    try:
        import shutil
        available_mb = shutil.disk_usage(path).free / (1024 * 1024)
        required_mb = estimated_mb * safety_factor
        
        if available_mb < required_mb:
            from lsm_exceptions import DiskSpaceError
            raise DiskSpaceError(path, required_mb, available_mb)
        
        logger.debug(f"Disk space check passed for {path}", 
                    required_mb=required_mb, 
                    available_mb=available_mb)
        
    except Exception as e:
        logger.warning(f"Disk space check failed: {e}")


# Validation decorators

def validate_inputs(**validators):
    """
    Decorator to validate function inputs.
    
    Args:
        **validators: Mapping of parameter names to validation functions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get function signature
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    try:
                        validated_value = validator(value)
                        bound_args.arguments[param_name] = validated_value
                    except Exception as e:
                        logger.error(f"Input validation failed for {func.__name__}.{param_name}: {e}")
                        raise
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator