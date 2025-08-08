"""
TensorFlow compatibility layer for handling import issues gracefully.

This module provides a compatibility layer that handles TensorFlow import failures
and provides fallback mechanisms for when TensorFlow is not available or has DLL issues.
"""

import sys
import os
import warnings
from typing import Any, Optional, Dict, Callable
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)

# Global state for TensorFlow availability
_TF_AVAILABLE = None
_TF_MODULE = None
_TF_ERROR = None

def _try_import_tensorflow():
    """Attempt to import TensorFlow with various fallback strategies."""
    global _TF_AVAILABLE, _TF_MODULE, _TF_ERROR
    
    if _TF_AVAILABLE is not None:
        return _TF_AVAILABLE, _TF_MODULE, _TF_ERROR
    
    # Set environment variables to help with TensorFlow issues
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')  # Suppress TF warnings
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')  # Force CPU mode
    
    strategies = [
        ("Standard import", lambda: __import__('tensorflow')),
        ("CPU-only import", lambda: _import_tf_cpu_only()),
        ("Minimal import", lambda: _import_tf_minimal()),
    ]
    
    for strategy_name, import_func in strategies:
        try:
            logger.info(f"Attempting TensorFlow import: {strategy_name}")
            tf_module = import_func()
            
            # Test basic functionality
            _ = tf_module.constant([1, 2, 3])
            
            _TF_AVAILABLE = True
            _TF_MODULE = tf_module
            _TF_ERROR = None
            logger.info(f"TensorFlow successfully imported using: {strategy_name}")
            return True, tf_module, None
            
        except Exception as e:
            logger.warning(f"TensorFlow import failed ({strategy_name}): {str(e)[:100]}...")
            _TF_ERROR = e
            continue
    
    # All strategies failed
    _TF_AVAILABLE = False
    _TF_MODULE = None
    logger.error("All TensorFlow import strategies failed")
    return False, None, _TF_ERROR

def _import_tf_cpu_only():
    """Import TensorFlow with CPU-only configuration."""
    import tensorflow as tf
    
    # Configure for CPU only
    tf.config.set_visible_devices([], 'GPU')
    
    # Disable GPU memory growth warnings
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    
    return tf

def _import_tf_minimal():
    """Import TensorFlow with minimal configuration."""
    # Try importing with minimal setup
    import tensorflow.compat.v2 as tf
    tf.enable_v2_behavior()
    return tf

class TensorFlowProxy:
    """
    Proxy object that provides TensorFlow functionality when available,
    and raises informative errors when not available.
    """
    
    def __init__(self):
        self._available, self._module, self._error = _try_import_tensorflow()
    
    def __getattr__(self, name: str) -> Any:
        if not self._available:
            raise ImportError(
                f"TensorFlow is not available. Cannot access '{name}'. "
                f"Original error: {self._error}. "
                "Please check your TensorFlow installation or use CPU-only mode."
            )
        return getattr(self._module, name)
    
    def __bool__(self) -> bool:
        return self._available
    
    @property
    def available(self) -> bool:
        """Check if TensorFlow is available."""
        return self._available
    
    @property
    def error(self) -> Optional[Exception]:
        """Get the TensorFlow import error if any."""
        return self._error

class KerasProxy:
    """Proxy for Keras functionality."""
    
    def __init__(self, tf_proxy: TensorFlowProxy):
        self._tf_proxy = tf_proxy
    
    def __getattr__(self, name: str) -> Any:
        if not self._tf_proxy.available:
            raise ImportError(
                f"Keras is not available because TensorFlow is not available. "
                f"Cannot access 'keras.{name}'. "
                f"Original error: {self._tf_proxy.error}"
            )
        return getattr(self._tf_proxy._module.keras, name)

# Create global proxies
tf = TensorFlowProxy()
keras = KerasProxy(tf)

def require_tensorflow(func: Callable) -> Callable:
    """
    Decorator that ensures TensorFlow is available before calling a function.
    
    Args:
        func: Function that requires TensorFlow
        
    Returns:
        Decorated function that checks TensorFlow availability
    """
    def wrapper(*args, **kwargs):
        if not tf.available:
            raise ImportError(
                f"Function '{func.__name__}' requires TensorFlow, but it's not available. "
                f"Error: {tf.error}. "
                "Please install TensorFlow or fix the installation issues."
            )
        return func(*args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper

def get_tensorflow_info() -> Dict[str, Any]:
    """
    Get information about TensorFlow availability and configuration.
    
    Returns:
        Dictionary with TensorFlow status information
    """
    info = {
        'available': tf.available,
        'error': str(tf.error) if tf.error else None,
        'version': None,
        'gpu_available': False,
        'devices': []
    }
    
    if tf.available:
        try:
            info['version'] = tf.__version__
            info['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
            info['devices'] = [device.name for device in tf.config.list_physical_devices()]
        except Exception as e:
            info['device_query_error'] = str(e)
    
    return info

def create_fallback_layer(layer_name: str, *args, **kwargs):
    """
    Create a fallback layer when TensorFlow is not available.
    
    Args:
        layer_name: Name of the layer type
        *args, **kwargs: Layer arguments
        
    Returns:
        Fallback layer object
    """
    class FallbackLayer:
        def __init__(self, name, *args, **kwargs):
            self.name = name
            self.args = args
            self.kwargs = kwargs
        
        def __call__(self, *args, **kwargs):
            raise ImportError(
                f"Cannot use {self.name} layer because TensorFlow is not available. "
                f"Original error: {tf.error}"
            )
        
        def build(self, *args, **kwargs):
            raise ImportError(f"Cannot build {self.name} layer - TensorFlow not available")
        
        def call(self, *args, **kwargs):
            raise ImportError(f"Cannot call {self.name} layer - TensorFlow not available")
    
    return FallbackLayer(layer_name, *args, **kwargs)

# Convenience functions for common TensorFlow operations
def safe_constant(value, dtype=None, shape=None, name=None):
    """Safely create a TensorFlow constant."""
    if not tf.available:
        raise ImportError("Cannot create TensorFlow constant - TensorFlow not available")
    return tf.constant(value, dtype=dtype, shape=shape, name=name)

def safe_random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None, name=None):
    """Safely create random normal tensor."""
    if not tf.available:
        raise ImportError("Cannot create random tensor - TensorFlow not available")
    return tf.random.normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed, name=name)

def check_tensorflow_installation():
    """
    Check TensorFlow installation and provide diagnostic information.
    
    Returns:
        Tuple of (success: bool, info: dict)
    """
    info = get_tensorflow_info()
    
    if info['available']:
        logger.info(f"TensorFlow {info['version']} is available")
        if info['gpu_available']:
            logger.info("GPU support is available")
        else:
            logger.info("Running in CPU-only mode")
        return True, info
    else:
        logger.error(f"TensorFlow is not available: {info['error']}")
        return False, info

if __name__ == "__main__":
    # Test the compatibility layer
    success, info = check_tensorflow_installation()
    print("TensorFlow Compatibility Check:")
    print(f"Available: {info['available']}")
    if info['error']:
        print(f"Error: {info['error']}")
    if info['version']:
        print(f"Version: {info['version']}")
        print(f"GPU Available: {info['gpu_available']}")
        print(f"Devices: {info['devices']}")