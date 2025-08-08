#!/usr/bin/env python3
"""
Logging infrastructure for the Sparse Sine-Activated LSM project.

This module provides centralized logging configuration and utilities
for debugging, monitoring, and error tracking throughout the LSM system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
import traceback


class LSMFormatter(logging.Formatter):
    """Custom formatter for LSM logging with structured output."""
    
    def __init__(self, include_context: bool = True):
        """
        Initialize LSM formatter.
        
        Args:
            include_context: Whether to include additional context in log messages
        """
        self.include_context = include_context
        
        # Color codes for console output
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
        
        # Base format
        base_format = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        super().__init__(base_format, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with optional context and colors."""
        # Add context information if available
        if self.include_context and hasattr(record, 'context'):
            context_str = json.dumps(record.context, separators=(',', ':'))
            record.message = f"{record.getMessage()} | Context: {context_str}"
        else:
            record.message = record.getMessage()
        
        # Format the base message
        formatted = super().format(record)
        
        # Add colors for console output
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            color = self.colors.get(record.levelname, '')
            reset = self.colors['RESET']
            formatted = f"{color}{formatted}{reset}"
        
        return formatted


class LSMLogger:
    """Enhanced logger for LSM operations with context tracking."""
    
    def __init__(self, name: str, level: int = logging.INFO):
        """
        Initialize LSM logger.
        
        Args:
            name: Logger name (usually module name)
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.context = {}
    
    def set_context(self, **kwargs):
        """Set context information for subsequent log messages."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context information."""
        self.context.clear()
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context information."""
        # Merge context with additional kwargs
        full_context = {**self.context, **kwargs}
        
        # Create log record with context
        extra = {'context': full_context} if full_context else {}
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with full traceback and context."""
        # Add exception information to context
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            kwargs.update({
                'exception_type': exc_info[0].__name__,
                'exception_message': str(exc_info[1]),
                'traceback': traceback.format_exc()
            })
        
        self._log_with_context(logging.ERROR, message, **kwargs)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    max_file_size_mb: int = 10,
    backup_count: int = 5,
    include_context: bool = True
) -> None:
    """
    Set up centralized logging configuration for the LSM system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, no file logging)
        console_output: Whether to output logs to console
        max_file_size_mb: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        include_context: Whether to include context in log messages
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = LSMFormatter(include_context=include_context)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log the setup
    logger = LSMLogger(__name__)
    logger.info("Logging system initialized", 
                log_level=log_level, 
                log_file=log_file, 
                console_output=console_output)


def get_logger(name: str) -> LSMLogger:
    """
    Get an LSM logger instance for the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        LSMLogger instance
    """
    return LSMLogger(name)


def log_function_call(func_name: str, args: tuple = (), kwargs: Dict[str, Any] = None):
    """
    Decorator to log function calls with arguments.
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"Calling {func_name}", 
                        function=func_name,
                        args_count=len(args),
                        kwargs_keys=list(kwargs.keys()) if kwargs else [])
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Completed {func_name} successfully")
                return result
            except Exception as e:
                logger.exception(f"Error in {func_name}", 
                               function=func_name,
                               error_type=type(e).__name__)
                raise
        
        return wrapper
    return decorator


def log_performance(operation: str):
    """
    Decorator to log performance metrics for operations.
    
    Args:
        operation: Description of the operation being timed
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            logger = get_logger(func.__module__)
            
            start_time = time.time()
            logger.debug(f"Starting {operation}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(f"Completed {operation}", 
                           operation=operation,
                           duration_seconds=round(duration, 3))
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"Failed {operation} after {duration:.3f}s", 
                           operation=operation,
                           duration_seconds=round(duration, 3),
                           error_type=type(e).__name__)
                raise
        
        return wrapper
    return decorator


def create_operation_logger(operation_name: str, **context) -> LSMLogger:
    """
    Create a logger with pre-set context for a specific operation.
    
    Args:
        operation_name: Name of the operation
        **context: Additional context to include in all log messages
        
    Returns:
        LSMLogger with pre-set context
    """
    logger = get_logger(f"lsm.{operation_name}")
    logger.set_context(operation=operation_name, **context)
    return logger


def log_system_info():
    """Log system information for debugging purposes."""
    import platform
    import psutil
    
    logger = get_logger(__name__)
    
    try:
        # System information
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'memory_available_gb': round(psutil.virtual_memory().available / (1024**3), 2)
        }
        
        logger.info("System information", **system_info)
        
        # GPU information if available
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                gpu_info = {
                    'gpu_count': len(gpus),
                    'gpu_names': [gpu.name for gpu in gpus]
                }
                logger.info("GPU information", **gpu_info)
            else:
                logger.info("No GPUs detected")
        except ImportError:
            logger.debug("TensorFlow not available for GPU detection")
        
    except Exception as e:
        logger.warning(f"Failed to log system information: {e}")


# Default logging setup for the LSM system
def setup_default_logging():
    """Set up default logging configuration for LSM."""
    # Create logs directory
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(logs_dir, f"lsm_{timestamp}.log")
    
    # Set up logging
    setup_logging(
        log_level="INFO",
        log_file=log_file,
        console_output=True,
        include_context=True
    )
    
    # Log system information
    log_system_info()


# Initialize default logging when module is imported
if not logging.getLogger().handlers:
    setup_default_logging()