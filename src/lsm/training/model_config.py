#!/usr/bin/env python3
"""
Model configuration management for the Sparse Sine-Activated LSM.

This module provides centralized configuration management for all model
parameters, training settings, and system metadata.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import platform
import sys

from lsm_exceptions import (
    ConfigurationError, InvalidConfigurationError, MissingConfigurationError,
    handle_file_operation_error
)
from lsm_logging import get_logger, log_performance

logger = get_logger(__name__)

@dataclass
class ModelConfiguration:
    """
    Centralized configuration for LSM model parameters and training settings.
    """
    
    # Model architecture parameters
    window_size: int = 10
    embedding_dim: int = 128
    reservoir_type: str = 'standard'
    reservoir_config: Dict[str, Any] = field(default_factory=dict)
    reservoir_units: List[int] = field(default_factory=lambda: [256, 128, 64])
    sparsity: float = 0.1
    use_multichannel: bool = True
    
    # Training parameters
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.1
    test_size: float = 0.2
    
    # Tokenizer configuration
    tokenizer_max_features: int = 10000
    tokenizer_ngram_range: tuple = field(default_factory=lambda: (1, 2))
    
    # System metadata (auto-populated)
    model_version: str = "1.0"
    created_at: Optional[str] = None
    python_version: Optional[str] = None
    platform_info: Optional[str] = None
    
    def __post_init__(self):
        """Auto-populate system metadata if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        
        if self.python_version is None:
            self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if self.platform_info is None:
            self.platform_info = f"{platform.system()} {platform.release()}"
    
    def save(self, path: str) -> None:
        """
        Save configuration to JSON file.
        
        Args:
            path: File path to save configuration
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert to dictionary and handle special types
        config_dict = self._to_serializable_dict()
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfiguration':
        """
        Load configuration from JSON file.
        
        Args:
            path: File path to load configuration from
            
        Returns:
            ModelConfiguration instance
        """
        if not os.path.exists(path):
            raise InvalidConfigurationError(path, ["Configuration file not found"])
        
        try:
            with open(path, 'r') as f:
                config_dict = json.load(f)
            return cls.from_dict(config_dict)
        except json.JSONDecodeError as e:
            raise InvalidConfigurationError(path, [f"Invalid JSON format: {e}"])
        except Exception as e:
            raise handle_file_operation_error("load configuration", path, e)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelConfiguration':
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            ModelConfiguration instance
        """
        # Handle special conversions
        if 'tokenizer_ngram_range' in data and isinstance(data['tokenizer_ngram_range'], list):
            data['tokenizer_ngram_range'] = tuple(data['tokenizer_ngram_range'])
        
        # Filter out unknown fields to handle version compatibility
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        
        return cls(**filtered_data)
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with JSON-serializable types."""
        config_dict = self.to_dict()
        
        # Convert tuple to list for JSON serialization
        if isinstance(config_dict.get('tokenizer_ngram_range'), tuple):
            config_dict['tokenizer_ngram_range'] = list(config_dict['tokenizer_ngram_range'])
        
        return config_dict
    
    def validate(self) -> List[str]:
        """
        Validate configuration parameters.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate basic parameters
        if self.window_size <= 0:
            errors.append("window_size must be positive")
        
        if self.embedding_dim <= 0:
            errors.append("embedding_dim must be positive")
        
        if self.reservoir_type not in ['standard', 'hierarchical', 'attentive', 'echo_state', 'deep']:
            errors.append(f"Invalid reservoir_type: {self.reservoir_type}")
        
        if not (0.0 < self.sparsity <= 1.0):
            errors.append("sparsity must be between 0 and 1")
        
        if self.epochs <= 0:
            errors.append("epochs must be positive")
        
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if not (0.0 < self.learning_rate < 1.0):
            errors.append("learning_rate must be between 0 and 1")
        
        if not (0.0 <= self.validation_split < 1.0):
            errors.append("validation_split must be between 0 and 1")
        
        if not (0.0 < self.test_size < 1.0):
            errors.append("test_size must be between 0 and 1")
        
        # Validate reservoir units
        if not self.reservoir_units or not all(isinstance(u, int) and u > 0 for u in self.reservoir_units):
            errors.append("reservoir_units must be a list of positive integers")
        
        # Validate tokenizer parameters
        if self.tokenizer_max_features <= 0:
            errors.append("tokenizer_max_features must be positive")
        
        if (not isinstance(self.tokenizer_ngram_range, (tuple, list)) or 
            len(self.tokenizer_ngram_range) != 2 or
            not all(isinstance(n, int) and n > 0 for n in self.tokenizer_ngram_range) or
            self.tokenizer_ngram_range[0] > self.tokenizer_ngram_range[1]):
            errors.append("tokenizer_ngram_range must be a tuple/list of two positive integers (min, max)")
        
        return errors
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        summary = f"""
Model Configuration Summary:
===========================
Architecture:
  - Window Size: {self.window_size}
  - Embedding Dimension: {self.embedding_dim}
  - Reservoir Type: {self.reservoir_type}
  - Reservoir Units: {self.reservoir_units}
  - Sparsity: {self.sparsity}
  - Multi-channel: {self.use_multichannel}

Training:
  - Epochs: {self.epochs}
  - Batch Size: {self.batch_size}
  - Learning Rate: {self.learning_rate}
  - Validation Split: {self.validation_split}
  - Test Size: {self.test_size}

Tokenizer:
  - Max Features: {self.tokenizer_max_features}
  - N-gram Range: {self.tokenizer_ngram_range}

System:
  - Model Version: {self.model_version}
  - Created: {self.created_at}
  - Python: {self.python_version}
  - Platform: {self.platform_info}
        """.strip()
        
        return summary
    
    def update_from_args(self, args) -> None:
        """
        Update configuration from command-line arguments.
        
        Args:
            args: Parsed command-line arguments
        """
        # Map common argument names to configuration fields
        arg_mapping = {
            'window_size': 'window_size',
            'embedding_dim': 'embedding_dim',
            'reservoir_type': 'reservoir_type',
            'sparsity': 'sparsity',
            'epochs': 'epochs',
            'batch_size': 'batch_size',
            'test_size': 'test_size'
        }
        
        for arg_name, config_field in arg_mapping.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                setattr(self, config_field, getattr(args, arg_name))
        
        # Handle reservoir config if provided
        if hasattr(args, 'reservoir_config') and args.reservoir_config:
            try:
                import json
                self.reservoir_config = json.loads(args.reservoir_config)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse reservoir_config: {args.reservoir_config}")

@dataclass
class TrainingMetadata:
    """
    Metadata about a completed training session.
    """
    
    training_completed_at: str
    training_duration_seconds: float
    dataset_info: Dict[str, Any]
    performance_metrics: Dict[str, float]
    system_info: Dict[str, str]
    
    def save(self, path: str) -> None:
        """Save metadata to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)
        
        print(f"Training metadata saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingMetadata':
        """Load metadata from JSON file."""
        if not os.path.exists(path):
            raise InvalidConfigurationError(path, ["Metadata file not found"])
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        except json.JSONDecodeError as e:
            raise InvalidConfigurationError(path, [f"Invalid JSON format: {e}"])
        except Exception as e:
            raise handle_file_operation_error("load metadata", path, e)
    
    @classmethod
    def create_from_training(cls, training_results: Dict[str, Any], 
                           dataset_info: Dict[str, Any]) -> 'TrainingMetadata':
        """
        Create metadata from training results.
        
        Args:
            training_results: Results from training session
            dataset_info: Information about the dataset used
            
        Returns:
            TrainingMetadata instance
        """
        return cls(
            training_completed_at=datetime.now().isoformat(),
            training_duration_seconds=training_results.get('training_time', 0.0),
            dataset_info=dataset_info,
            performance_metrics={
                'final_test_mse': training_results.get('test_mse', 0.0),
                'final_test_mae': training_results.get('test_mae', 0.0),
                'best_val_loss': min(training_results.get('history', {}).get('val_loss', [float('inf')]))
            },
            system_info={
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'platform': f"{platform.system()} {platform.release()}"
            }
        )

if __name__ == "__main__":
    # Test configuration functionality
    print("Testing ModelConfiguration...")
    
    # Create default configuration
    config = ModelConfiguration()
    print("✓ Created default configuration")
    
    # Test validation
    errors = config.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
    else:
        print("✓ Configuration is valid")
    
    # Test serialization
    config_dict = config.to_dict()
    print(f"✓ Converted to dictionary with {len(config_dict)} fields")
    
    # Test deserialization
    new_config = ModelConfiguration.from_dict(config_dict)
    print("✓ Created from dictionary")
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config_path = f.name
    
    try:
        config.save(config_path)
        loaded_config = ModelConfiguration.load(config_path)
        print("✓ Save/load successful")
        
        # Verify they're the same
        if config.to_dict() == loaded_config.to_dict():
            print("✓ Loaded configuration matches original")
        else:
            print("❌ Loaded configuration differs from original")
    
    finally:
        os.unlink(config_path)
    
    # Test summary
    summary = config.get_summary()
    print(f"✓ Generated summary ({len(summary)} characters)")
    
    print("\n🎉 ModelConfiguration tests completed!")