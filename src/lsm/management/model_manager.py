#!/usr/bin/env python3
"""
Model Manager for the Sparse Sine-Activated LSM.

This module provides model discovery, validation, and management utilities
for trained LSM models. It handles model directory scanning, metadata extraction,
integrity checking, and cleanup operations.
"""

import os
import json
import glob
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from ..training.model_config import ModelConfiguration, TrainingMetadata
from ..data.data_loader import DialogueTokenizer
from ..utils.lsm_exceptions import (
    ModelValidationError, ModelLoadError, InvalidInputError,
    handle_file_operation_error
)
from ..utils.lsm_logging import get_logger, log_performance

logger = get_logger(__name__)

class ModelManager:
    """
    High-level model management and discovery for LSM models.
    
    This class provides utilities for:
    - Discovering available models in the workspace
    - Extracting model metadata and configuration
    - Validating model integrity and completeness
    - Cleaning up incomplete or corrupted models
    """
    
    def __init__(self, models_root_dir: str = "."):
        """
        Initialize ModelManager.
        
        Args:
            models_root_dir: Root directory to search for model directories
        """
        self.models_root_dir = os.path.abspath(models_root_dir)
        logger.info(f"ModelManager initialized with root directory: {self.models_root_dir}")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Scan for valid model directories and return their information.
        
        Returns:
            List of dictionaries containing model information
        """
        logger.info("Scanning for available models...")
        
        models = []
        
        # Look for directories matching the pattern "models_YYYYMMDD_HHMMSS"
        model_pattern = os.path.join(self.models_root_dir, "models_*")
        potential_dirs = glob.glob(model_pattern)
        
        for model_dir in potential_dirs:
            if os.path.isdir(model_dir):
                try:
                    model_info = self.get_model_info(model_dir)
                    if model_info.get('is_valid', False):
                        models.append(model_info)
                        logger.debug(f"Found valid model: {model_dir}")
                    else:
                        logger.warning(f"Found invalid model directory: {model_dir}")
                except Exception as e:
                    logger.error(f"Error processing model directory {model_dir}: {e}")
        
        logger.info(f"Found {len(models)} valid models")
        return sorted(models, key=lambda x: x.get('created_at', ''), reverse=True)
    
    def get_model_info(self, model_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive metadata and configuration details from a model directory.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Dictionary containing detailed model information
        """
        if not os.path.exists(model_path):
            return {
                'path': model_path,
                'is_valid': False,
                'error': f"Model directory does not exist: {model_path}"
            }
        
        model_info = {
            'path': os.path.abspath(model_path),
            'name': os.path.basename(model_path),
            'is_valid': False,
            'components': {},
            'validation_errors': []
        }
        
        try:
            # Check for required components
            components = self._check_model_components(model_path)
            model_info['components'] = components
            
            # Load configuration if available
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                try:
                    config = ModelConfiguration.load(config_path)
                    model_info['configuration'] = config.to_dict()
                    model_info['created_at'] = config.created_at
                    model_info['model_version'] = config.model_version
                    model_info['architecture'] = {
                        'window_size': config.window_size,
                        'embedding_dim': config.embedding_dim,
                        'reservoir_type': config.reservoir_type,
                        'use_multichannel': config.use_multichannel
                    }
                except Exception as e:
                    model_info['validation_errors'].append(f"Failed to load configuration: {e}")
            else:
                model_info['validation_errors'].append("Configuration file missing")
            
            # Load training metadata if available
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    metadata = TrainingMetadata.load(metadata_path)
                    model_info['training_metadata'] = {
                        'completed_at': metadata.training_completed_at,
                        'duration_seconds': metadata.training_duration_seconds,
                        'dataset_info': metadata.dataset_info,
                        'performance_metrics': metadata.performance_metrics,
                        'system_info': metadata.system_info
                    }
                except Exception as e:
                    model_info['validation_errors'].append(f"Failed to load training metadata: {e}")
            
            # Load tokenizer info if available
            tokenizer_path = os.path.join(model_path, "tokenizer")
            if os.path.exists(tokenizer_path):
                try:
                    tokenizer_config_path = os.path.join(tokenizer_path, "config.json")
                    if os.path.exists(tokenizer_config_path):
                        with open(tokenizer_config_path, 'r') as f:
                            tokenizer_config = json.load(f)
                        model_info['tokenizer'] = tokenizer_config
                except Exception as e:
                    model_info['validation_errors'].append(f"Failed to load tokenizer info: {e}")
            
            # Get file sizes and modification times
            model_info['file_info'] = self._get_file_info(model_path)
            
            # Validate model integrity
            is_valid, validation_errors = self.validate_model(model_path)
            model_info['is_valid'] = is_valid
            model_info['validation_errors'].extend(validation_errors)
            
            # Calculate total size
            model_info['total_size_mb'] = self._calculate_directory_size(model_path)
            
        except Exception as e:
            model_info['validation_errors'].append(f"Unexpected error during info extraction: {e}")
            logger.error(f"Error extracting model info from {model_path}: {e}")
        
        return model_info
    
    @log_performance("model validation")
    def validate_model(self, model_path: str) -> Tuple[bool, List[str]]:
        """
        Check model integrity and completeness.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        logger.debug("Starting model validation", model_path=model_path)
        errors = []
        
        try:
            if not os.path.exists(model_path):
                error = f"Model directory does not exist: {model_path}"
                logger.warning(error)
                return False, [error]
            
            if not os.path.isdir(model_path):
                error = f"Model path is not a directory: {model_path}"
                logger.warning(error)
                return False, [error]
        except Exception as e:
            error = f"Failed to access model path: {e}"
            logger.error(error, model_path=model_path)
            return False, [error]
        
        # Check for essential components
        required_components = {
            'reservoir_model': 'Reservoir model directory',
            'cnn_model': 'CNN model directory',
            'config.json': 'Configuration file'
        }
        
        for component, description in required_components.items():
            component_path = os.path.join(model_path, component)
            if not os.path.exists(component_path):
                errors.append(f"Missing {description}: {component}")
        
        # Check if model directories contain actual model files
        reservoir_path = os.path.join(model_path, "reservoir_model")
        if os.path.exists(reservoir_path):
            if not self._is_valid_keras_model(reservoir_path):
                errors.append("Reservoir model directory is not a valid Keras model")
        
        cnn_path = os.path.join(model_path, "cnn_model")
        if os.path.exists(cnn_path):
            if not self._is_valid_keras_model(cnn_path):
                errors.append("CNN model directory is not a valid Keras model")
        
        # Validate configuration file
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                config = ModelConfiguration.load(config_path)
                config_errors = config.validate()
                if config_errors:
                    errors.extend([f"Configuration error: {err}" for err in config_errors])
            except Exception as e:
                errors.append(f"Failed to load or validate configuration: {e}")
        
        # Check tokenizer if present
        tokenizer_path = os.path.join(model_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            tokenizer_errors = self._validate_tokenizer(tokenizer_path)
            errors.extend(tokenizer_errors)
        
        # Check for training history
        history_path = os.path.join(model_path, "training_history.csv")
        if not os.path.exists(history_path):
            errors.append("Training history file missing (non-critical)")
        
        is_valid = len([err for err in errors if not err.endswith("(non-critical)")]) == 0
        
        return is_valid, errors
    
    def cleanup_incomplete_models(self, dry_run: bool = True) -> List[str]:
        """
        Find and optionally remove incomplete or corrupted model directories.
        
        Args:
            dry_run: If True, only report what would be cleaned up without actually deleting
            
        Returns:
            List of directories that were (or would be) cleaned up
        """
        logger.info(f"Scanning for incomplete models (dry_run={dry_run})...")
        
        cleanup_candidates = []
        
        # Find all potential model directories
        model_pattern = os.path.join(self.models_root_dir, "models_*")
        potential_dirs = glob.glob(model_pattern)
        
        for model_dir in potential_dirs:
            if os.path.isdir(model_dir):
                is_valid, errors = self.validate_model(model_dir)
                
                # Consider for cleanup if it has critical errors
                critical_errors = [err for err in errors if not err.endswith("(non-critical)")]
                if critical_errors:
                    cleanup_info = {
                        'path': model_dir,
                        'errors': critical_errors,
                        'size_mb': self._calculate_directory_size(model_dir)
                    }
                    cleanup_candidates.append(cleanup_info)
                    
                    if not dry_run:
                        try:
                            shutil.rmtree(model_dir)
                            logger.info(f"Removed incomplete model: {model_dir}")
                        except Exception as e:
                            logger.error(f"Failed to remove {model_dir}: {e}")
                    else:
                        logger.info(f"Would remove incomplete model: {model_dir} "
                                  f"({len(critical_errors)} errors, {cleanup_info['size_mb']:.1f} MB)")
        
        if dry_run:
            logger.info(f"Found {len(cleanup_candidates)} incomplete models "
                       f"(total size: {sum(c['size_mb'] for c in cleanup_candidates):.1f} MB)")
        else:
            logger.info(f"Cleaned up {len(cleanup_candidates)} incomplete models")
        
        return [c['path'] for c in cleanup_candidates]
    
    def _check_model_components(self, model_path: str) -> Dict[str, bool]:
        """Check which model components are present."""
        components = {
            'reservoir_model': os.path.exists(os.path.join(model_path, "reservoir_model")),
            'cnn_model': os.path.exists(os.path.join(model_path, "cnn_model")),
            'tokenizer': os.path.exists(os.path.join(model_path, "tokenizer")),
            'config': os.path.exists(os.path.join(model_path, "config.json")),
            'metadata': os.path.exists(os.path.join(model_path, "metadata.json")),
            'training_history': os.path.exists(os.path.join(model_path, "training_history.csv"))
        }
        return components
    
    def _is_valid_keras_model(self, model_path: str) -> bool:
        """Check if a directory contains a valid Keras model."""
        # Check for essential Keras model files
        required_files = ['saved_model.pb']
        variables_dir = os.path.join(model_path, 'variables')
        
        # Check for saved_model.pb
        if not os.path.exists(os.path.join(model_path, 'saved_model.pb')):
            return False
        
        # Check for variables directory
        if not os.path.exists(variables_dir):
            return False
        
        # Check for variables files
        variables_files = os.listdir(variables_dir)
        if not any(f.startswith('variables.data') for f in variables_files):
            return False
        
        if 'variables.index' not in variables_files:
            return False
        
        return True
    
    def _validate_tokenizer(self, tokenizer_path: str) -> List[str]:
        """Validate tokenizer directory contents."""
        errors = []
        
        required_files = ['config.json', 'vectorizer.pkl']
        for file_name in required_files:
            file_path = os.path.join(tokenizer_path, file_name)
            if not os.path.exists(file_path):
                errors.append(f"Missing tokenizer file: {file_name}")
        
        # Check if vocabulary files exist
        vocab_files = ['vocabulary.json', 'vocabulary_embeddings.npy']
        vocab_exists = any(os.path.exists(os.path.join(tokenizer_path, f)) for f in vocab_files)
        if not vocab_exists:
            errors.append("Missing tokenizer vocabulary files")
        
        return errors
    
    def _get_file_info(self, model_path: str) -> Dict[str, Dict[str, Any]]:
        """Get file information for model components."""
        file_info = {}
        
        components = ['reservoir_model', 'cnn_model', 'tokenizer', 'config.json', 'metadata.json', 'training_history.csv']
        
        for component in components:
            component_path = os.path.join(model_path, component)
            if os.path.exists(component_path):
                try:
                    stat = os.stat(component_path)
                    file_info[component] = {
                        'size_bytes': stat.st_size if os.path.isfile(component_path) else self._calculate_directory_size(component_path) * 1024 * 1024,
                        'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'is_directory': os.path.isdir(component_path)
                    }
                except Exception as e:
                    file_info[component] = {'error': str(e)}
        
        return file_info
    
    def _calculate_directory_size(self, directory_path: str) -> float:
        """Calculate total size of directory in MB."""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        pass  # Skip files that can't be accessed
        except Exception:
            pass
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_model_summary(self, model_path: str) -> str:
        """
        Get a human-readable summary of a model.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            Formatted string summary of the model
        """
        info = self.get_model_info(model_path)
        
        if not info.get('is_valid', False):
            return f"❌ Invalid Model: {info.get('name', 'Unknown')}\n" + \
                   f"   Errors: {', '.join(info.get('validation_errors', []))}"
        
        summary_lines = [
            f"✅ {info.get('name', 'Unknown Model')}",
            f"   Path: {info['path']}",
            f"   Size: {info.get('total_size_mb', 0):.1f} MB"
        ]
        
        if 'architecture' in info:
            arch = info['architecture']
            summary_lines.extend([
                f"   Architecture: {arch.get('reservoir_type', 'unknown')} reservoir",
                f"   Window Size: {arch.get('window_size', 'unknown')}",
                f"   Embedding Dim: {arch.get('embedding_dim', 'unknown')}"
            ])
        
        if 'training_metadata' in info:
            meta = info['training_metadata']
            perf = meta.get('performance_metrics', {})
            summary_lines.extend([
                f"   Training: {meta.get('completed_at', 'unknown')}",
                f"   Test MSE: {perf.get('final_test_mse', 'unknown'):.6f}" if isinstance(perf.get('final_test_mse'), (int, float)) else f"   Test MSE: {perf.get('final_test_mse', 'unknown')}"
            ])
        
        if 'tokenizer' in info:
            tok = info['tokenizer']
            summary_lines.append(f"   Vocabulary: {tok.get('vocabulary_size', 'unknown')} words")
        
        return '\n'.join(summary_lines)
    
    def list_models_summary(self) -> str:
        """Get a formatted summary of all available models."""
        models = self.list_available_models()
        
        if not models:
            return "No valid models found in the workspace."
        
        summary_lines = [f"Found {len(models)} valid models:\n"]
        
        for i, model in enumerate(models, 1):
            model_summary = self.get_model_summary(model['path'])
            summary_lines.append(f"{i}. {model_summary}")
            summary_lines.append("")  # Empty line between models
        
        return '\n'.join(summary_lines)

if __name__ == "__main__":
    # Test ModelManager functionality
    print("Testing ModelManager...")
    
    manager = ModelManager()
    
    # List available models
    print("\n📋 Available Models:")
    print("=" * 50)
    models = manager.list_available_models()
    
    if models:
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['name']}")
            print(f"   Valid: {'✅' if model['is_valid'] else '❌'}")
            print(f"   Size: {model.get('total_size_mb', 0):.1f} MB")
            
            if model.get('validation_errors'):
                print(f"   Errors: {', '.join(model['validation_errors'][:3])}")
    else:
        print("No models found.")
    
    # Test cleanup (dry run)
    print(f"\n🧹 Cleanup Analysis:")
    print("=" * 50)
    cleanup_candidates = manager.cleanup_incomplete_models(dry_run=True)
    
    if cleanup_candidates:
        print(f"Found {len(cleanup_candidates)} incomplete models that could be cleaned up.")
    else:
        print("No incomplete models found.")
    
    print("\n🎉 ModelManager tests completed!")