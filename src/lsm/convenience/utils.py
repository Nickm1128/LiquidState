"""
Utility functions for the LSM convenience API.

This module provides helper functions for parameter validation, error handling,
and common operations used across the convenience API classes.
"""

import os
import sys
import time
import psutil
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np

from ..utils.lsm_exceptions import (
    LSMError, InsufficientMemoryError, DiskSpaceError,
    InvalidInputError, DataValidationError
)
from ..utils.lsm_logging import get_logger
from .config import ConvenienceValidationError

logger = get_logger(__name__)


def validate_conversation_data(data: Any, name: str = "conversation data") -> List[str]:
    """
    Validate conversation data format with helpful error messages.
    
    This function provides backward compatibility while leveraging the new
    data format handling system for comprehensive validation and conversion.
    
    Parameters
    ----------
    data : any
        Data to validate
    name : str, default="conversation data"
        Name for error messages
        
    Returns
    -------
    validated_data : list
        Validated conversation data as list of strings
        
    Raises
    ------
    ConvenienceValidationError
        If data format is invalid
    """
    try:
        # Import here to avoid circular imports
        from .data_formats import DataFormatHandler
        
        # Create handler with default settings
        handler = DataFormatHandler()
        
        # Process the data and return as simple list
        validated_data = handler.process_conversation_data(
            data=data,
            preprocess=True,
            return_format="simple_list"
        )
        
        if not validated_data:
            raise ConvenienceValidationError(
                f"{name} contains no valid conversations after processing",
                suggestion="Check data format and ensure it contains meaningful conversation content"
            )
        
        logger.debug(f"Validated {len(validated_data)} conversation messages")
        return validated_data
        
    except ConvenienceValidationError:
        # Re-raise convenience validation errors as-is
        raise
    except Exception as e:
        # Wrap other exceptions in ConvenienceValidationError
        raise ConvenienceValidationError(
            f"Failed to validate {name}: {e}",
            suggestion="Check data format and ensure it matches supported conversation formats"
        )


def validate_classification_data(X: Any, y: Any) -> Tuple[List[str], np.ndarray]:
    """
    Validate classification data format.
    
    Parameters
    ----------
    X : any
        Input features (text data)
    y : any
        Target labels
        
    Returns
    -------
    X_validated : list
        Validated input data as list of strings
    y_validated : np.ndarray
        Validated labels as numpy array
        
    Raises
    ------
    ConvenienceValidationError
        If data format is invalid
    """
    # Validate X
    if not isinstance(X, (list, np.ndarray)):
        raise ConvenienceValidationError(
            "X must be a list or array of text samples",
            suggestion="Provide input as: ['text1', 'text2', ...] or numpy array"
        )
    
    if len(X) == 0:
        raise ConvenienceValidationError(
            "X cannot be empty",
            suggestion="Provide at least one text sample for training"
        )
    
    # Convert to list of strings
    X_validated = []
    for i, sample in enumerate(X):
        if not isinstance(sample, str):
            raise ConvenienceValidationError(
                f"X[{i}] must be a string, got {type(sample).__name__}",
                suggestion="Ensure all input samples are text strings"
            )
        
        if len(sample.strip()) == 0:
            raise ConvenienceValidationError(
                f"X[{i}] cannot be empty",
                suggestion="Remove empty samples or provide valid text"
            )
        
        X_validated.append(sample.strip())
    
    # Validate y
    if y is None:
        raise ConvenienceValidationError(
            "y cannot be None for classification",
            suggestion="Provide target labels for supervised learning"
        )
    
    try:
        y_validated = np.array(y)
    except Exception as e:
        raise ConvenienceValidationError(
            f"Cannot convert y to numpy array: {e}",
            suggestion="Provide labels as a list or array of integers/strings"
        )
    
    if len(y_validated) != len(X_validated):
        raise ConvenienceValidationError(
            f"X and y must have same length: X has {len(X_validated)}, y has {len(y_validated)}",
            suggestion="Ensure each input sample has a corresponding label"
        )
    
    if len(y_validated.shape) > 1 and y_validated.shape[1] > 1:
        raise ConvenienceValidationError(
            "Multi-label classification not yet supported",
            suggestion="Use single labels for each sample"
        )
    
    return X_validated, y_validated.flatten()


def validate_regression_data(X: Any, y: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate regression data format.
    
    Parameters
    ----------
    X : any
        Input features
    y : any
        Target values
        
    Returns
    -------
    X_validated : np.ndarray
        Validated input data
    y_validated : np.ndarray
        Validated target values
        
    Raises
    ------
    ConvenienceValidationError
        If data format is invalid
    """
    # Validate X
    try:
        X_validated = np.array(X)
    except Exception as e:
        raise ConvenienceValidationError(
            f"Cannot convert X to numpy array: {e}",
            suggestion="Provide input as a list or array of numerical sequences"
        )
    
    if X_validated.size == 0:
        raise ConvenienceValidationError(
            "X cannot be empty",
            suggestion="Provide at least one input sequence for training"
        )
    
    # Ensure X is 2D
    if X_validated.ndim == 1:
        X_validated = X_validated.reshape(-1, 1)
    elif X_validated.ndim > 2:
        raise ConvenienceValidationError(
            f"X must be 1D or 2D array, got {X_validated.ndim}D",
            suggestion="Reshape input to (n_samples, n_features) format"
        )
    
    # Validate y
    if y is None:
        raise ConvenienceValidationError(
            "y cannot be None for regression",
            suggestion="Provide target values for supervised learning"
        )
    
    try:
        y_validated = np.array(y, dtype=float)
    except Exception as e:
        raise ConvenienceValidationError(
            f"Cannot convert y to numerical array: {e}",
            suggestion="Provide target values as numbers (int or float)"
        )
    
    if len(y_validated) != len(X_validated):
        raise ConvenienceValidationError(
            f"X and y must have same length: X has {len(X_validated)}, y has {len(y_validated)}",
            suggestion="Ensure each input sample has a corresponding target value"
        )
    
    return X_validated, y_validated.flatten()


def check_system_resources(operation: str, estimated_memory_mb: float = 0,
                          estimated_disk_mb: float = 0, 
                          path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check system resources and provide warnings if insufficient.
    
    Parameters
    ----------
    operation : str
        Description of the operation
    estimated_memory_mb : float, default=0
        Estimated memory requirement in MB
    estimated_disk_mb : float, default=0
        Estimated disk space requirement in MB
    path : str, optional
        Path for disk space checking
        
    Returns
    -------
    resource_info : dict
        Information about system resources
        
    Raises
    ------
    InsufficientMemoryError
        If insufficient memory is available
    DiskSpaceError
        If insufficient disk space is available
    """
    resource_info = {}
    
    try:
        # Check memory
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / (1024 * 1024)
        resource_info['available_memory_mb'] = available_memory_mb
        resource_info['total_memory_mb'] = memory.total / (1024 * 1024)
        
        if estimated_memory_mb > 0:
            if available_memory_mb < estimated_memory_mb * 1.5:  # 50% safety margin
                if available_memory_mb < estimated_memory_mb:
                    raise InsufficientMemoryError(
                        operation, estimated_memory_mb, available_memory_mb
                    )
                else:
                    logger.warning(
                        f"Low memory for {operation}: "
                        f"estimated {estimated_memory_mb:.1f}MB, "
                        f"available {available_memory_mb:.1f}MB"
                    )
        
        # Check disk space
        if estimated_disk_mb > 0 and path:
            import shutil
            disk_usage = shutil.disk_usage(path)
            available_disk_mb = disk_usage.free / (1024 * 1024)
            resource_info['available_disk_mb'] = available_disk_mb
            
            if available_disk_mb < estimated_disk_mb * 1.2:  # 20% safety margin
                if available_disk_mb < estimated_disk_mb:
                    raise DiskSpaceError(path, estimated_disk_mb, available_disk_mb)
                else:
                    logger.warning(
                        f"Low disk space for {operation}: "
                        f"estimated {estimated_disk_mb:.1f}MB, "
                        f"available {available_disk_mb:.1f}MB"
                    )
        
        # Check CPU
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        resource_info['cpu_count'] = cpu_count
        resource_info['cpu_percent'] = cpu_percent
        
        if cpu_percent > 90:
            logger.warning(f"High CPU usage ({cpu_percent:.1f}%) may slow down {operation}")
        
    except ImportError:
        logger.warning("psutil not available for resource checking")
    except Exception as e:
        logger.warning(f"Resource check failed: {e}")
    
    return resource_info


def estimate_training_time(data_size: int, config: Dict[str, Any]) -> Dict[str, float]:
    """
    Estimate training time based on data size and configuration.
    
    Parameters
    ----------
    data_size : int
        Number of training samples
    config : dict
        Model configuration
        
    Returns
    -------
    estimates : dict
        Time estimates in different units
    """
    # Base time per sample (rough estimates in seconds)
    base_time_per_sample = {
        'standard': 0.001,
        'hierarchical': 0.002,
        'attentive': 0.005,
        'echo_state': 0.0015,
        'deep': 0.008
    }
    
    reservoir_type = config.get('reservoir_type', 'standard')
    epochs = config.get('epochs', 50)
    embedding_dim = config.get('embedding_dim', 128)
    
    # Calculate base time
    base_time = base_time_per_sample.get(reservoir_type, 0.002)
    
    # Adjust for model complexity
    complexity_factor = (embedding_dim / 128) ** 0.5
    
    # Total estimated time
    total_seconds = data_size * epochs * base_time * complexity_factor
    
    return {
        'seconds': total_seconds,
        'minutes': total_seconds / 60,
        'hours': total_seconds / 3600,
        'human_readable': format_duration(total_seconds)
    }


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Parameters
    ----------
    seconds : float
        Duration in seconds
        
    Returns
    -------
    formatted : str
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def create_progress_callback(description: str = "Training") -> Callable:
    """
    Create a progress callback function for training.
    
    Parameters
    ----------
    description : str, default="Training"
        Description of the operation
        
    Returns
    -------
    callback : callable
        Progress callback function
    """
    start_time = time.time()
    
    def progress_callback(epoch: int, total_epochs: int, 
                         loss: Optional[float] = None,
                         metrics: Optional[Dict[str, float]] = None):
        """Progress callback function."""
        elapsed = time.time() - start_time
        progress = epoch / total_epochs
        
        # Estimate remaining time
        if progress > 0:
            eta = elapsed * (1 - progress) / progress
            eta_str = format_duration(eta)
        else:
            eta_str = "unknown"
        
        # Format progress message
        progress_bar = "█" * int(progress * 20) + "░" * (20 - int(progress * 20))
        
        message = f"\r{description}: [{progress_bar}] {epoch}/{total_epochs} "
        message += f"({progress*100:.1f}%) - ETA: {eta_str}"
        
        if loss is not None:
            message += f" - Loss: {loss:.4f}"
        
        if metrics:
            for name, value in metrics.items():
                message += f" - {name}: {value:.4f}"
        
        print(message, end="", flush=True)
        
        if epoch == total_epochs:
            print()  # New line when complete
    
    return progress_callback


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module with error handling.
    
    Parameters
    ----------
    module_name : str
        Name of the module to import
    package : str, optional
        Package name for relative imports
        
    Returns
    -------
    module : module or None
        Imported module or None if import failed
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError as e:
        logger.warning(f"Failed to import {module_name}: {e}")
        return None


def cleanup_temp_files(temp_dir: Union[str, Path]) -> None:
    """
    Clean up temporary files and directories.
    
    Parameters
    ----------
    temp_dir : str or Path
        Directory containing temporary files
    """
    try:
        temp_path = Path(temp_dir)
        if temp_path.exists() and temp_path.is_dir():
            import shutil
            shutil.rmtree(temp_path)
            logger.debug(f"Cleaned up temporary directory: {temp_path}")
    except Exception as e:
        logger.warning(f"Failed to clean up temporary files: {e}")


def get_optimal_batch_size(data_size: int, available_memory_mb: float,
                          model_memory_mb: float) -> int:
    """
    Calculate optimal batch size based on available memory.
    
    Parameters
    ----------
    data_size : int
        Total number of samples
    available_memory_mb : float
        Available memory in MB
    model_memory_mb : float
        Memory required by the model in MB
        
    Returns
    -------
    batch_size : int
        Optimal batch size
    """
    # Reserve memory for model and system
    usable_memory_mb = available_memory_mb - model_memory_mb - 500  # 500MB buffer
    
    if usable_memory_mb <= 0:
        return 1  # Minimum batch size
    
    # Estimate memory per sample (rough estimate)
    memory_per_sample_mb = 0.1  # Adjust based on actual usage
    
    max_batch_size = int(usable_memory_mb / memory_per_sample_mb)
    
    # Reasonable bounds
    batch_size = max(1, min(max_batch_size, data_size, 256))
    
    logger.debug(f"Calculated optimal batch size: {batch_size}")
    return batch_size


def validate_sklearn_compatibility(estimator: Any) -> List[str]:
    """
    Validate sklearn compatibility of an estimator.
    
    Parameters
    ----------
    estimator : object
        Estimator to validate
        
    Returns
    -------
    issues : list
        List of compatibility issues (empty if compatible)
    """
    issues = []
    
    # Check required methods
    required_methods = ['fit', 'get_params', 'set_params']
    for method in required_methods:
        if not hasattr(estimator, method):
            issues.append(f"Missing required method: {method}")
    
    # Check if fit returns self
    if hasattr(estimator, 'fit'):
        try:
            # This is a basic check - in practice, we'd need actual data
            pass
        except Exception:
            pass
    
    # Check parameter consistency
    if hasattr(estimator, 'get_params') and hasattr(estimator, 'set_params'):
        try:
            params = estimator.get_params()
            estimator.set_params(**params)
        except Exception as e:
            issues.append(f"Parameter get/set inconsistency: {e}")
    
    return issues


def validate_structured_conversation_data(data: Any, 
                                        name: str = "structured conversation data",
                                        require_system_messages: bool = False,
                                        require_roles: bool = False) -> List[Dict[str, Any]]:
    """
    Validate structured conversation data with system message support.
    
    This function validates and processes conversation data that includes
    system messages, roles, and other structured information.
    
    Parameters
    ----------
    data : any
        Structured conversation data to validate
    name : str, default="structured conversation data"
        Name for error messages
    require_system_messages : bool, default=False
        Whether to require system messages in conversations
    require_roles : bool, default=False
        Whether to require role information
        
    Returns
    -------
    validated_data : list
        List of validated conversation dictionaries
        
    Raises
    ------
    ConvenienceValidationError
        If data format is invalid
    """
    try:
        from .data_formats import DataFormatHandler
        
        # Create handler with default settings
        handler = DataFormatHandler()
        
        # Process the data and return as structured format
        conversations = handler.process_conversation_data(
            data=data,
            preprocess=True,
            return_format="structured"
        )
        
        if not conversations:
            raise ConvenienceValidationError(
                f"{name} contains no valid conversations after processing",
                suggestion="Check data format and ensure it contains meaningful conversation content"
            )
        
        # Convert to dictionary format for backward compatibility
        validated_conversations = []
        for conv in conversations:
            conv_dict = {
                "messages": conv.messages,
                "system_message": conv.system_message,
                "roles": conv.roles,
                "conversation_id": conv.conversation_id,
                "metadata": conv.metadata
            }
            
            # Apply additional validation requirements
            if require_system_messages and not conv.system_message:
                raise ConvenienceValidationError(
                    f"System message required but missing in conversation {conv.conversation_id}",
                    suggestion="Provide system messages for all conversations or set require_system_messages=False"
                )
            
            if require_roles and not conv.roles:
                raise ConvenienceValidationError(
                    f"Roles required but missing in conversation {conv.conversation_id}",
                    suggestion="Provide role information for all messages or set require_roles=False"
                )
            
            validated_conversations.append(conv_dict)
        
        logger.debug(f"Validated {len(validated_conversations)} structured conversations")
        return validated_conversations
        
    except ConvenienceValidationError:
        raise
    except Exception as e:
        raise ConvenienceValidationError(
            f"Failed to validate {name}: {e}",
            suggestion="Check data format and ensure it matches supported structured conversation formats"
        )


def detect_conversation_format(data: Any) -> str:
    """
    Automatically detect the format of conversation data.
    
    Parameters
    ----------
    data : any
        Conversation data to analyze
        
    Returns
    -------
    format_name : str
        Detected format name
    """
    try:
        from .data_formats import ConversationFormatDetector
        
        detector = ConversationFormatDetector()
        detected_format = detector.detect_format(data)
        
        return detected_format.value
        
    except Exception as e:
        logger.warning(f"Failed to detect conversation format: {e}")
        return "unknown"


def convert_conversation_format(data: Any, 
                              target_format: str = "simple_list",
                              source_format: Optional[str] = None) -> Any:
    """
    Convert conversation data between different formats.
    
    Parameters
    ----------
    data : any
        Input conversation data
    target_format : str, default="simple_list"
        Target format name
    source_format : str, optional
        Source format name (auto-detected if None)
        
    Returns
    -------
    converted_data : any
        Converted conversation data
        
    Raises
    ------
    ConvenienceValidationError
        If conversion fails
    """
    try:
        from .data_formats import DataFormatHandler, ConversationFormat
        
        # Convert source format string to enum if provided
        source_format_enum = None
        if source_format:
            try:
                source_format_enum = ConversationFormat(source_format)
            except ValueError:
                raise ConvenienceValidationError(
                    f"Unsupported source format: {source_format}",
                    suggestion="Use one of the supported format names"
                )
        
        # Create handler and convert
        handler = DataFormatHandler()
        converted_data = handler.convert_format(
            data=data,
            source_format=source_format_enum,
            target_format=target_format
        )
        
        logger.debug(f"Converted conversation data to {target_format} format")
        return converted_data
        
    except ConvenienceValidationError:
        raise
    except Exception as e:
        raise ConvenienceValidationError(
            f"Failed to convert conversation format: {e}",
            suggestion="Check data format and target format specification"
        )


def get_conversation_statistics(data: Any) -> Dict[str, Any]:
    """
    Get detailed statistics about conversation data.
    
    Parameters
    ----------
    data : any
        Conversation data to analyze
        
    Returns
    -------
    statistics : dict
        Dictionary with conversation statistics
    """
    try:
        from .data_formats import DataFormatHandler
        
        handler = DataFormatHandler()
        validation_results = handler.validate_conversation_data(data)
        
        return validation_results.get("statistics", {})
        
    except Exception as e:
        logger.warning(f"Failed to get conversation statistics: {e}")
        return {"error": str(e)}


def preprocess_conversation_data(data: Any,
                               min_message_length: int = 1,
                               max_message_length: int = 1000,
                               min_conversation_length: int = 1,
                               max_conversation_length: int = 100,
                               normalize_whitespace: bool = True,
                               return_format: str = "simple_list") -> Any:
    """
    Preprocess conversation data with customizable options.
    
    Parameters
    ----------
    data : any
        Input conversation data
    min_message_length : int, default=1
        Minimum message length in characters
    max_message_length : int, default=1000
        Maximum message length in characters
    min_conversation_length : int, default=1
        Minimum number of messages per conversation
    max_conversation_length : int, default=100
        Maximum number of messages per conversation
    normalize_whitespace : bool, default=True
        Whether to normalize whitespace
    return_format : str, default="simple_list"
        Format to return data in
        
    Returns
    -------
    processed_data : any
        Preprocessed conversation data
        
    Raises
    ------
    ConvenienceValidationError
        If preprocessing fails
    """
    try:
        from .data_formats import DataFormatHandler
        
        # Create handler with custom preprocessing config
        preprocessor_config = {
            "min_message_length": min_message_length,
            "max_message_length": max_message_length,
            "min_conversation_length": min_conversation_length,
            "max_conversation_length": max_conversation_length,
            "normalize_whitespace": normalize_whitespace,
            "remove_empty_messages": True
        }
        
        handler = DataFormatHandler(preprocessor_config=preprocessor_config)
        
        processed_data = handler.process_conversation_data(
            data=data,
            preprocess=True,
            return_format=return_format
        )
        
        logger.debug(f"Preprocessed conversation data with {len(processed_data) if isinstance(processed_data, list) else 'N/A'} items")
        return processed_data
        
    except ConvenienceValidationError:
        raise
    except Exception as e:
        raise ConvenienceValidationError(
            f"Failed to preprocess conversation data: {e}",
            suggestion="Check preprocessing parameters and data format"
        )