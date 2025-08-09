"""
LSM Convenience API - Scikit-learn compatible interface for LSM models.

This module provides a simplified, sklearn-like interface for creating and using
Liquid State Machine models without dealing with the complexity of the underlying
multi-component architecture.

Main Classes:
    LSMGenerator: For text generation and conversational AI
    LSMClassifier: For classification tasks using LSM features
    LSMRegressor: For regression tasks using LSM temporal dynamics
    
Example Usage:
    >>> from lsm.convenience import LSMGenerator
    >>> generator = LSMGenerator()
    >>> generator.fit(conversations)
    >>> response = generator.generate("Hello, how are you?")
"""

from .base import LSMBase
from .config import ConvenienceConfig, ConvenienceValidationError
from .data_formats import (
    ConversationFormat, ConversationData, DataFormatHandler,
    ConversationParser, ConversationPreprocessor
)
from .utils import (
    validate_conversation_data, validate_classification_data, 
    validate_regression_data, check_system_resources,
    estimate_training_time, create_progress_callback,
    validate_structured_conversation_data, detect_conversation_format,
    convert_conversation_format, get_conversation_statistics,
    preprocess_conversation_data
)

# Performance monitoring and optimization
try:
    from .performance import (
        PerformanceProfiler, MemoryMonitor, AutoMemoryManager,
        BenchmarkSuite, get_global_profiler, get_global_memory_manager,
        monitor_performance, manage_memory
    )
    from .benchmarks import (
        ConvenienceAPIBenchmark, run_quick_benchmark, run_full_benchmark
    )
    _PERFORMANCE_AVAILABLE = True
except ImportError as e:
    PerformanceProfiler = None
    MemoryMonitor = None
    AutoMemoryManager = None
    BenchmarkSuite = None
    ConvenienceAPIBenchmark = None
    get_global_profiler = None
    get_global_memory_manager = None
    monitor_performance = None
    manage_memory = None
    run_quick_benchmark = None
    run_full_benchmark = None
    _PERFORMANCE_AVAILABLE = False

# Import convenience classes with error handling
try:
    from .generator import LSMGenerator
    _GENERATOR_AVAILABLE = True
except ImportError as e:
    LSMGenerator = None
    _GENERATOR_AVAILABLE = False

try:
    from .classifier import LSMClassifier
    _CLASSIFIER_AVAILABLE = True
except ImportError as e:
    LSMClassifier = None
    _CLASSIFIER_AVAILABLE = False

try:
    from .regressor import LSMRegressor
    _REGRESSOR_AVAILABLE = True
except ImportError as e:
    LSMRegressor = None
    _REGRESSOR_AVAILABLE = False

# CLI functionality (optional import)
try:
    from . import cli
    _CLI_AVAILABLE = True
except ImportError as e:
    cli = None
    _CLI_AVAILABLE = False

__all__ = [
    'LSMBase',
    'ConvenienceConfig', 
    'ConvenienceValidationError',
    'ConversationFormat',
    'ConversationData',
    'DataFormatHandler',
    'ConversationParser',
    'ConversationPreprocessor',
    'validate_conversation_data',
    'validate_classification_data',
    'validate_regression_data',
    'check_system_resources',
    'estimate_training_time',
    'create_progress_callback',
    'validate_structured_conversation_data',
    'detect_conversation_format',
    'convert_conversation_format',
    'get_conversation_statistics',
    'preprocess_conversation_data'
]

# Add performance monitoring components to __all__ if available
if _PERFORMANCE_AVAILABLE:
    __all__.extend([
        'PerformanceProfiler',
        'MemoryMonitor', 
        'AutoMemoryManager',
        'BenchmarkSuite',
        'ConvenienceAPIBenchmark',
        'get_global_profiler',
        'get_global_memory_manager',
        'monitor_performance',
        'manage_memory',
        'run_quick_benchmark',
        'run_full_benchmark'
    ])

# Add available classes to __all__
if _GENERATOR_AVAILABLE:
    __all__.append('LSMGenerator')

if _CLASSIFIER_AVAILABLE:
    __all__.append('LSMClassifier')

if _REGRESSOR_AVAILABLE:
    __all__.append('LSMRegressor')

if _CLI_AVAILABLE:
    __all__.append('cli')

__version__ = '1.0.0'