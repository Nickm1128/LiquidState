#!/usr/bin/env python3
"""
Custom exception classes for the Sparse Sine-Activated LSM project.

This module defines a hierarchy of custom exceptions for different error categories
in the LSM system, providing clear error messages and context for debugging.
"""

from typing import Optional, List, Dict, Any


class LSMError(Exception):
    """Base exception class for all LSM-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize LSM error.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Return formatted error message with details."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class ModelError(LSMError):
    """Base class for model-related errors."""
    pass


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, model_path: str, reason: str, missing_components: Optional[List[str]] = None):
        details = {"model_path": model_path, "reason": reason}
        if missing_components:
            details["missing_components"] = missing_components
        
        message = f"Failed to load model from '{model_path}': {reason}"
        if missing_components:
            message += f". Missing components: {', '.join(missing_components)}"
        
        super().__init__(message, details)
        self.model_path = model_path
        self.missing_components = missing_components or []


class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    
    def __init__(self, save_path: str, reason: str, component: Optional[str] = None):
        details = {"save_path": save_path, "reason": reason}
        if component:
            details["component"] = component
        
        message = f"Failed to save model to '{save_path}': {reason}"
        if component:
            message += f" (component: {component})"
        
        super().__init__(message, details)
        self.save_path = save_path
        self.component = component


class ModelValidationError(ModelError):
    """Raised when model validation fails."""
    
    def __init__(self, model_path: str, validation_errors: List[str]):
        details = {"model_path": model_path, "validation_errors": validation_errors}
        message = f"Model validation failed for '{model_path}': {'; '.join(validation_errors)}"
        
        super().__init__(message, details)
        self.model_path = model_path
        self.validation_errors = validation_errors


class ConfigurationError(LSMError):
    """Base class for configuration-related errors."""
    pass


class InvalidConfigurationError(ConfigurationError):
    """Raised when configuration is invalid or malformed."""
    
    def __init__(self, config_path: str, validation_errors: List[str]):
        details = {"config_path": config_path, "validation_errors": validation_errors}
        message = f"Invalid configuration in '{config_path}': {'; '.join(validation_errors)}"
        
        super().__init__(message, details)
        self.config_path = config_path
        self.validation_errors = validation_errors


class MissingConfigurationError(ConfigurationError):
    """Raised when required configuration is missing."""
    
    def __init__(self, config_path: str, missing_keys: List[str]):
        details = {"config_path": config_path, "missing_keys": missing_keys}
        message = f"Missing required configuration keys in '{config_path}': {', '.join(missing_keys)}"
        
        super().__init__(message, details)
        self.config_path = config_path
        self.missing_keys = missing_keys


class TokenizerError(LSMError):
    """Base class for tokenizer-related errors."""
    pass


class TokenizerNotFittedError(TokenizerError):
    """Raised when attempting to use an unfitted tokenizer."""
    
    def __init__(self, operation: str):
        message = f"Tokenizer must be fitted before performing '{operation}'. Call fit() first."
        details = {"operation": operation}
        super().__init__(message, details)
        self.operation = operation


class TokenizerLoadError(TokenizerError):
    """Raised when tokenizer loading fails."""
    
    def __init__(self, tokenizer_path: str, reason: str, missing_files: Optional[List[str]] = None):
        details = {"tokenizer_path": tokenizer_path, "reason": reason}
        if missing_files:
            details["missing_files"] = missing_files
        
        message = f"Failed to load tokenizer from '{tokenizer_path}': {reason}"
        if missing_files:
            message += f". Missing files: {', '.join(missing_files)}"
        
        super().__init__(message, details)
        self.tokenizer_path = tokenizer_path
        self.missing_files = missing_files or []


class TokenizerSaveError(TokenizerError):
    """Raised when tokenizer saving fails."""
    
    def __init__(self, save_path: str, reason: str):
        details = {"save_path": save_path, "reason": reason}
        message = f"Failed to save tokenizer to '{save_path}': {reason}"
        
        super().__init__(message, details)
        self.save_path = save_path


class InferenceError(LSMError):
    """Base class for inference-related errors."""
    pass


class InvalidInputError(InferenceError):
    """Raised when input validation fails."""
    
    def __init__(self, input_description: str, expected_format: str, actual_format: str):
        details = {
            "input_description": input_description,
            "expected_format": expected_format,
            "actual_format": actual_format
        }
        message = f"Invalid {input_description}: expected {expected_format}, got {actual_format}"
        
        super().__init__(message, details)
        self.input_description = input_description
        self.expected_format = expected_format
        self.actual_format = actual_format


class PredictionError(InferenceError):
    """Raised when prediction fails."""
    
    def __init__(self, reason: str, input_shape: Optional[tuple] = None):
        details = {"reason": reason}
        if input_shape:
            details["input_shape"] = input_shape
        
        message = f"Prediction failed: {reason}"
        if input_shape:
            message += f" (input shape: {input_shape})"
        
        super().__init__(message, details)
        self.input_shape = input_shape


class DataError(LSMError):
    """Base class for data-related errors."""
    pass


class DataLoadError(DataError):
    """Raised when data loading fails."""
    
    def __init__(self, data_source: str, reason: str):
        details = {"data_source": data_source, "reason": reason}
        message = f"Failed to load data from '{data_source}': {reason}"
        
        super().__init__(message, details)
        self.data_source = data_source


class DataValidationError(DataError):
    """Raised when data validation fails."""
    
    def __init__(self, data_description: str, validation_errors: List[str]):
        details = {"data_description": data_description, "validation_errors": validation_errors}
        message = f"Data validation failed for {data_description}: {'; '.join(validation_errors)}"
        
        super().__init__(message, details)
        self.data_description = data_description
        self.validation_errors = validation_errors


class TrainingError(LSMError):
    """Base class for training-related errors."""
    pass


class TrainingSetupError(TrainingError):
    """Raised when training setup fails."""
    
    def __init__(self, reason: str, component: Optional[str] = None):
        details = {"reason": reason}
        if component:
            details["component"] = component
        
        message = f"Training setup failed: {reason}"
        if component:
            message += f" (component: {component})"
        
        super().__init__(message, details)
        self.component = component


class TrainingExecutionError(TrainingError):
    """Raised when training execution fails."""
    
    def __init__(self, epoch: Optional[int], reason: str):
        details = {"reason": reason}
        if epoch is not None:
            details["epoch"] = epoch
        
        message = f"Training execution failed: {reason}"
        if epoch is not None:
            message += f" (at epoch {epoch})"
        
        super().__init__(message, details)
        self.epoch = epoch


class ResourceError(LSMError):
    """Base class for resource-related errors."""
    pass


class InsufficientMemoryError(ResourceError):
    """Raised when there's insufficient memory for an operation."""
    
    def __init__(self, operation: str, required_mb: Optional[float] = None, available_mb: Optional[float] = None):
        details = {"operation": operation}
        if required_mb is not None:
            details["required_mb"] = required_mb
        if available_mb is not None:
            details["available_mb"] = available_mb
        
        message = f"Insufficient memory for {operation}"
        if required_mb and available_mb:
            message += f": requires {required_mb:.1f}MB, available {available_mb:.1f}MB"
        
        super().__init__(message, details)
        self.operation = operation
        self.required_mb = required_mb
        self.available_mb = available_mb


class DiskSpaceError(ResourceError):
    """Raised when there's insufficient disk space."""
    
    def __init__(self, path: str, required_mb: Optional[float] = None, available_mb: Optional[float] = None):
        details = {"path": path}
        if required_mb is not None:
            details["required_mb"] = required_mb
        if available_mb is not None:
            details["available_mb"] = available_mb
        
        message = f"Insufficient disk space at '{path}'"
        if required_mb and available_mb:
            message += f": requires {required_mb:.1f}MB, available {available_mb:.1f}MB"
        
        super().__init__(message, details)
        self.path = path
        self.required_mb = required_mb
        self.available_mb = available_mb


class DatasetIntegrationError(LSMError):
    """Base class for dataset integration errors."""
    pass


class HuggingFaceDatasetError(DatasetIntegrationError):
    """Raised when HuggingFace dataset operations fail."""
    
    def __init__(self, dataset_name: str, operation: str, reason: str):
        details = {"dataset_name": dataset_name, "operation": operation, "reason": reason}
        message = f"HuggingFace dataset operation '{operation}' failed for '{dataset_name}': {reason}"
        
        super().__init__(message, details)
        self.dataset_name = dataset_name
        self.operation = operation


class ConversationSplitError(DatasetIntegrationError):
    """Raised when conversation-aware data splitting fails."""
    
    def __init__(self, reason: str, total_conversations: Optional[int] = None):
        details = {"reason": reason}
        if total_conversations is not None:
            details["total_conversations"] = total_conversations
        
        message = f"Conversation splitting failed: {reason}"
        if total_conversations is not None:
            message += f" (total conversations: {total_conversations})"
        
        super().__init__(message, details)
        self.total_conversations = total_conversations


class DatasetValidationError(DatasetIntegrationError):
    """Raised when dataset validation fails."""
    
    def __init__(self, dataset_name: str, validation_errors: List[str]):
        details = {"dataset_name": dataset_name, "validation_errors": validation_errors}
        message = f"Dataset validation failed for '{dataset_name}': {'; '.join(validation_errors)}"
        
        super().__init__(message, details)
        self.dataset_name = dataset_name
        self.validation_errors = validation_errors


class BackwardCompatibilityError(LSMError):
    """Base class for backward compatibility errors."""
    pass


class UnsupportedModelVersionError(BackwardCompatibilityError):
    """Raised when trying to load an unsupported model version."""
    
    def __init__(self, model_path: str, model_version: str, supported_versions: List[str]):
        details = {
            "model_path": model_path,
            "model_version": model_version,
            "supported_versions": supported_versions
        }
        message = (f"Unsupported model version '{model_version}' in '{model_path}'. "
                  f"Supported versions: {', '.join(supported_versions)}")
        
        super().__init__(message, details)
        self.model_path = model_path
        self.model_version = model_version
        self.supported_versions = supported_versions


class MigrationError(BackwardCompatibilityError):
    """Raised when model migration fails."""
    
    def __init__(self, old_path: str, new_path: str, reason: str):
        details = {"old_path": old_path, "new_path": new_path, "reason": reason}
        message = f"Failed to migrate model from '{old_path}' to '{new_path}': {reason}"
        
        super().__init__(message, details)
        self.old_path = old_path
        self.new_path = new_path


# Utility functions for error handling

def format_validation_errors(errors: List[str], max_errors: int = 5) -> str:
    """Format validation errors for display."""
    if not errors:
        return "No errors"
    
    if len(errors) <= max_errors:
        return "; ".join(errors)
    else:
        displayed_errors = errors[:max_errors]
        remaining = len(errors) - max_errors
        return "; ".join(displayed_errors) + f"; ... and {remaining} more errors"


def create_error_context(operation: str, **kwargs) -> Dict[str, Any]:
    """Create error context dictionary for debugging."""
    context = {
        "operation": operation,
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    context.update(kwargs)
    return context


def handle_file_operation_error(operation: str, file_path: str, error: Exception) -> LSMError:
    """Convert file operation exceptions to appropriate LSM exceptions."""
    if isinstance(error, FileNotFoundError):
        return ModelLoadError(file_path, f"File not found during {operation}")
    elif isinstance(error, PermissionError):
        return ModelSaveError(file_path, f"Permission denied during {operation}")
    elif isinstance(error, OSError) and "No space left on device" in str(error):
        return DiskSpaceError(file_path)
    else:
        return LSMError(f"File operation '{operation}' failed for '{file_path}': {error}")