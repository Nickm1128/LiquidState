"""
Configuration management and validation for the LSM convenience API.

This module provides intelligent defaults, parameter validation, and preset
configurations for common use cases.
"""

import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from ..utils.lsm_exceptions import LSMError, ConfigurationError
from ..utils.input_validation import (
    validate_positive_integer, validate_positive_float, 
    create_helpful_error_message
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class ConvenienceValidationError(LSMError):
    """
    Validation error with helpful suggestions for the convenience API.
    
    This exception provides clear error messages with actionable suggestions
    to help users fix common configuration issues.
    """
    
    def __init__(self, message: str, suggestion: Optional[str] = None, 
                 valid_options: Optional[List[str]] = None):
        """
        Initialize validation error with suggestions.
        
        Parameters
        ----------
        message : str
            The error message
        suggestion : str, optional
            A helpful suggestion to fix the error
        valid_options : list, optional
            List of valid options for the parameter
        """
        super().__init__(message)
        self.suggestion = suggestion
        self.valid_options = valid_options
    
    def __str__(self) -> str:
        """Return formatted error message with suggestions."""
        msg = self.message
        
        if self.suggestion:
            msg += f"\n💡 Suggestion: {self.suggestion}"
        
        if self.valid_options:
            options_str = ", ".join(f"'{opt}'" for opt in self.valid_options)
            msg += f"\n✅ Valid options: {options_str}"
        
        return msg


class ConvenienceConfig:
    """
    Configuration management for the LSM convenience API.
    
    This class provides preset configurations, parameter validation,
    and intelligent defaults for different use cases.
    """
    
    # Preset configurations for common use cases
    PRESETS = {
        'fast': {
            'window_size': 5,
            'embedding_dim': 64,
            'reservoir_type': 'standard',
            'reservoir_config': {
                'reservoir_units': [50, 25],
                'sparsity': 0.2,
                'spectral_radius': 0.8
            },
            'epochs': 10,
            'batch_size': 64,
            'description': 'Fast training with reduced accuracy - good for experimentation'
        },
        
        'balanced': {
            'window_size': 10,
            'embedding_dim': 128,
            'reservoir_type': 'hierarchical',
            'reservoir_config': {
                'reservoir_units': [100, 50],
                'sparsity': 0.1,
                'spectral_radius': 0.9,
                'hierarchy_levels': 2
            },
            'epochs': 50,
            'batch_size': 32,
            'description': 'Balanced performance and training time - recommended default'
        },
        
        'quality': {
            'window_size': 20,
            'embedding_dim': 256,
            'reservoir_type': 'attentive',
            'reservoir_config': {
                'reservoir_units': [200, 100, 50],
                'sparsity': 0.05,
                'spectral_radius': 0.95,
                'attention_heads': 4
            },
            'epochs': 100,
            'batch_size': 16,
            'description': 'High quality results with longer training time'
        },
        
        'text_generation': {
            'window_size': 15,
            'embedding_dim': 128,
            'reservoir_type': 'hierarchical',
            'reservoir_config': {
                'reservoir_units': [150, 75],
                'sparsity': 0.1,
                'spectral_radius': 0.9,
                'hierarchy_levels': 2
            },
            'system_message_support': True,
            'response_level': True,
            'epochs': 75,
            'batch_size': 24,
            'description': 'Optimized for conversational AI and text generation'
        },
        
        'classification': {
            'window_size': 8,
            'embedding_dim': 96,
            'reservoir_type': 'standard',
            'reservoir_config': {
                'reservoir_units': [80, 40],
                'sparsity': 0.15,
                'spectral_radius': 0.85
            },
            'epochs': 40,
            'batch_size': 48,
            'description': 'Optimized for text classification tasks'
        },
        
        'time_series': {
            'window_size': 12,
            'embedding_dim': 64,
            'reservoir_type': 'echo_state',
            'reservoir_config': {
                'reservoir_units': [120],
                'sparsity': 0.1,
                'spectral_radius': 0.9,
                'leaking_rate': 0.3
            },
            'epochs': 60,
            'batch_size': 32,
            'description': 'Optimized for time series prediction and regression'
        }
    }
    
    # Valid parameter ranges and options
    PARAMETER_CONSTRAINTS = {
        'window_size': {'min': 1, 'max': 100, 'type': int},
        'embedding_dim': {'min': 8, 'max': 2048, 'type': int},
        'reservoir_type': {
            'options': ['standard', 'hierarchical', 'attentive', 'echo_state', 'deep'],
            'type': str
        },
        'epochs': {'min': 1, 'max': 1000, 'type': int},
        'batch_size': {'min': 1, 'max': 1024, 'type': int},
        'sparsity': {'min': 0.01, 'max': 0.9, 'type': float},
        'spectral_radius': {'min': 0.1, 'max': 1.5, 'type': float},
        'leaking_rate': {'min': 0.01, 'max': 1.0, 'type': float},
        'temperature': {'min': 0.1, 'max': 2.0, 'type': float},
        'max_length': {'min': 1, 'max': 1000, 'type': int}
    }
    
    # Recommended parameter combinations
    PARAMETER_RECOMMENDATIONS = {
        'text_generation': {
            'reservoir_type': 'hierarchical',
            'window_size': 10,
            'embedding_dim': 128,
            'reason': 'Hierarchical reservoirs work well for sequential text generation'
        },
        'classification': {
            'reservoir_type': 'standard',
            'window_size': 8,
            'embedding_dim': 96,
            'reason': 'Standard reservoirs are efficient for classification tasks'
        },
        'regression': {
            'reservoir_type': 'echo_state',
            'window_size': 12,
            'embedding_dim': 64,
            'reason': 'Echo state networks excel at temporal pattern learning'
        }
    }
    
    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any]:
        """
        Get a preset configuration by name.
        
        Parameters
        ----------
        name : str
            Name of the preset configuration
            
        Returns
        -------
        config : dict
            Preset configuration dictionary
            
        Raises
        ------
        ConvenienceValidationError
            If the preset name is invalid
        """
        if name not in cls.PRESETS:
            raise ConvenienceValidationError(
                f"Unknown preset: '{name}'",
                suggestion="Use list_presets() to see available presets",
                valid_options=list(cls.PRESETS.keys())
            )
        
        return cls.PRESETS[name].copy()
    
    @classmethod
    def list_presets(cls) -> Dict[str, str]:
        """
        List all available preset configurations with descriptions.
        
        Returns
        -------
        presets : dict
            Mapping of preset names to descriptions
        """
        return {name: config['description'] 
                for name, config in cls.PRESETS.items()}
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any], 
                       context: str = "general") -> List[str]:
        """
        Validate parameter dictionary and return list of errors.
        
        Parameters
        ----------
        params : dict
            Parameters to validate
        context : str, default='general'
            Context for validation (affects recommendations)
            
        Returns
        -------
        errors : list
            List of validation error messages (empty if valid)
        """
        errors = []
        
        for param_name, value in params.items():
            if param_name in cls.PARAMETER_CONSTRAINTS:
                constraint = cls.PARAMETER_CONSTRAINTS[param_name]
                
                try:
                    # Type validation
                    expected_type = constraint['type']
                    if not isinstance(value, expected_type):
                        if expected_type == int and isinstance(value, float) and value.is_integer():
                            value = int(value)
                        else:
                            errors.append(
                                f"{param_name} must be {expected_type.__name__}, got {type(value).__name__}"
                            )
                            continue
                    
                    # Range validation for numeric types
                    if expected_type in (int, float):
                        if 'min' in constraint and value < constraint['min']:
                            errors.append(
                                f"{param_name} must be >= {constraint['min']}, got {value}"
                            )
                        if 'max' in constraint and value > constraint['max']:
                            errors.append(
                                f"{param_name} must be <= {constraint['max']}, got {value}"
                            )
                    
                    # Options validation for string types
                    elif 'options' in constraint:
                        if value not in constraint['options']:
                            errors.append(
                                f"{param_name} must be one of {constraint['options']}, got '{value}'"
                            )
                
                except Exception as e:
                    errors.append(f"Error validating {param_name}: {e}")
        
        # Context-specific validation
        if context in cls.PARAMETER_RECOMMENDATIONS:
            recommendations = cls.PARAMETER_RECOMMENDATIONS[context]
            for param, recommended_value in recommendations.items():
                if param in params and params[param] != recommended_value:
                    logger.info(
                        f"For {context}, consider using {param}='{recommended_value}': "
                        f"{recommendations.get('reason', 'Better performance expected')}"
                    )
        
        return errors
    
    @classmethod
    def suggest_preset(cls, **params) -> Optional[str]:
        """
        Suggest the best preset based on provided parameters.
        
        Parameters
        ----------
        **params : dict
            Parameters to match against presets
            
        Returns
        -------
        preset_name : str or None
            Name of the best matching preset, or None if no good match
        """
        if not params:
            return 'balanced'  # Default recommendation
        
        # Score each preset based on parameter matches
        scores = {}
        for preset_name, preset_config in cls.PRESETS.items():
            score = 0
            total_params = 0
            
            for param, value in params.items():
                if param in preset_config:
                    total_params += 1
                    if preset_config[param] == value:
                        score += 1
                    elif isinstance(value, (int, float)) and isinstance(preset_config[param], (int, float)):
                        # Partial score for numeric values based on closeness
                        diff = abs(value - preset_config[param])
                        max_val = max(value, preset_config[param])
                        if max_val > 0:
                            similarity = 1 - (diff / max_val)
                            score += max(0, similarity)
            
            if total_params > 0:
                scores[preset_name] = score / total_params
        
        # Return the preset with the highest score
        if scores:
            best_preset = max(scores, key=scores.get)
            if scores[best_preset] > 0.5:  # Only suggest if reasonably good match
                return best_preset
        
        return None
    
    @classmethod
    def create_config(cls, preset: Optional[str] = None, 
                     task_type: Optional[str] = None,
                     **overrides) -> Dict[str, Any]:
        """
        Create a configuration with intelligent defaults.
        
        Parameters
        ----------
        preset : str, optional
            Name of preset to use as base
        task_type : str, optional
            Type of task ('text_generation', 'classification', 'regression')
        **overrides : dict
            Parameters to override in the configuration
            
        Returns
        -------
        config : dict
            Complete configuration dictionary
            
        Raises
        ------
        ConvenienceValidationError
            If invalid parameters are provided
        """
        # Start with base configuration
        if preset:
            config = cls.get_preset(preset)
        elif task_type and task_type in cls.PRESETS:
            config = cls.get_preset(task_type)
        else:
            config = cls.get_preset('balanced')
        
        # Apply overrides
        config.update(overrides)
        
        # Validate the final configuration
        errors = cls.validate_params(config, context=task_type or 'general')
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"• {error}" for error in errors)
            raise ConvenienceValidationError(
                error_msg,
                suggestion="Check parameter values and ranges"
            )
        
        return config
    
    @classmethod
    def auto_configure(cls, data_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically configure parameters based on data characteristics.
        
        Parameters
        ----------
        data_info : dict
            Information about the data (size, type, etc.)
            
        Returns
        -------
        config : dict
            Automatically configured parameters
        """
        config = cls.get_preset('balanced').copy()
        
        # Adjust based on data size
        data_size = data_info.get('size', 1000)
        if data_size < 100:
            # Small dataset - use fast preset
            config.update(cls.get_preset('fast'))
            config['epochs'] = min(config['epochs'], 20)
        elif data_size > 10000:
            # Large dataset - can afford more complex model
            config['embedding_dim'] = min(config['embedding_dim'] * 2, 256)
            config['epochs'] = max(config['epochs'], 75)
        
        # Adjust based on sequence length
        avg_length = data_info.get('avg_sequence_length', 10)
        if avg_length < 5:
            config['window_size'] = max(3, avg_length)
        elif avg_length > 50:
            config['window_size'] = min(25, avg_length // 2)
        
        # Adjust based on task type
        task_type = data_info.get('task_type')
        if task_type == 'classification':
            config.update(cls.get_preset('classification'))
        elif task_type == 'regression':
            config.update(cls.get_preset('time_series'))
        elif task_type == 'generation':
            config.update(cls.get_preset('text_generation'))
        
        # Memory-based adjustments
        available_memory_mb = data_info.get('available_memory_mb', 4000)
        if available_memory_mb < 2000:
            # Low memory - reduce model size
            config['embedding_dim'] = min(config['embedding_dim'], 64)
            config['batch_size'] = min(config['batch_size'], 16)
            if 'reservoir_units' in config['reservoir_config']:
                config['reservoir_config']['reservoir_units'] = [
                    max(25, units // 2) for units in config['reservoir_config']['reservoir_units']
                ]
        
        logger.info(f"Auto-configured parameters based on data: {data_info}")
        return config
    
    @classmethod
    def save_config(cls, config: Dict[str, Any], path: Union[str, Path]) -> None:
        """
        Save configuration to a JSON file.
        
        Parameters
        ----------
        config : dict
            Configuration to save
        path : str or Path
            Path to save the configuration file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        logger.info(f"Configuration saved to {path}")
    
    @classmethod
    def load_config(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from a JSON file.
        
        Parameters
        ----------
        path : str or Path
            Path to the configuration file
            
        Returns
        -------
        config : dict
            Loaded configuration
            
        Raises
        ------
        ConvenienceValidationError
            If the configuration file is invalid
        """
        path = Path(path)
        
        if not path.exists():
            raise ConvenienceValidationError(
                f"Configuration file not found: {path}",
                suggestion="Check the file path or create a new configuration"
            )
        
        try:
            with open(path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError as e:
            raise ConvenienceValidationError(
                f"Invalid JSON in configuration file: {e}",
                suggestion="Check the JSON syntax in the configuration file"
            )
        
        # Validate loaded configuration
        errors = cls.validate_params(config)
        if errors:
            error_msg = "Loaded configuration is invalid:\n" + "\n".join(f"• {error}" for error in errors)
            raise ConvenienceValidationError(
                error_msg,
                suggestion="Fix the configuration file or create a new one"
            )
        
        logger.info(f"Configuration loaded from {path}")
        return config