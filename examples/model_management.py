#!/usr/bin/env python3
"""
Model Management Examples
=========================

This script demonstrates the model management capabilities of the LSM system.
Shows how to discover, validate, and manage trained models.
"""

import sys
import os
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.management.model_manager import ModelManager
from lsm_exceptions import ModelLoadError

def model_discovery_example():
    """Demonstrate model discovery and listing."""
    print("🔍 Model Discovery Example")
    print("=" * 40)
    
    try:
        manager = ModelManager()
        models = manager.list_available_models()
        
        if not models:
            print("❌ No trained models found in the current directory.")
            print("Train a model first using: python main.py train")
            print()
            return []
        
        print(f"✅ Found {len(models)} trained model(s):")
        print()
        
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['path']}")
            print(f"   Created: {model.get('created_at', 'Unknown')}")
            print(f"   Status: {model.get('status', 'Unknown')}")
            
            # Show performance metrics if available
            if 'test_mse' in model:
                print(f"   Test MSE: {model['test_mse']:.6f}")
            if 'test_mae' in model:
                print(f"   Test MAE: {model['test_mae']:.6f}")
            
            # Show configuration highlights
            config = model.get('configuration', {})
            if config:
                print(f"   Window Size: {config.get('window_size', 'N/A')}")
                print(f"   Reservoir Type: {config.get('reservoir_type', 'N/A')}")
            
            print()
        
        return models
        
    except Exception as e:
        print(f"❌ Model discovery failed: {e}")
        print()
        return []

def model_validation_example(models: List[Dict[str, Any]]):
    """Demonstrate model validation."""
    print("✅ Model Validation Example")
    print("=" * 40)
    
    if not models:
        print("No models available for validation.")
        print()
        return
    
    manager = ModelManager()
    
    for model in models[:3]:  # Validate first 3 models
        model_path = model['path']
        print(f"Validating model: {model_path}")
        
        try:
            is_valid, errors = manager.validate_model(model_path)
            
            if is_valid:
                print("  ✅ Model is valid and complete")
            else:
                print("  ❌ Model validation failed:")
                for error in errors:
                    print(f"    - {error}")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Validation error: {e}")
            print()

def detailed_model_info_example(models: List[Dict[str, Any]]):
    """Demonstrate detailed model information retrieval."""
    print("📋 Detailed Model Information Example")
    print("=" * 40)
    
    if not models:
        print("No models available for detailed inspection.")
        print()
        return
    
    manager = ModelManager()
    
    # Show detailed info for the first model
    model = models[0]
    model_path = model['path']
    
    print(f"Detailed information for: {model_path}")
    print("-" * 60)
    
    try:
        info = manager.get_model_info(model_path)
        
        # Configuration information
        print("📊 Configuration:")
        config = info.get('configuration', {})
        if config:
            print(f"  Model Version: {config.get('model_version', 'N/A')}")
            print(f"  Window Size: {config.get('window_size', 'N/A')}")
            print(f"  Embedding Dimension: {config.get('embedding_dim', 'N/A')}")
            print(f"  Reservoir Type: {config.get('reservoir_type', 'N/A')}")
            print(f"  Reservoir Units: {config.get('reservoir_units', 'N/A')}")
            print(f"  Sparsity: {config.get('sparsity', 'N/A')}")
            print(f"  Use Multichannel: {config.get('use_multichannel', 'N/A')}")
        else:
            print("  Configuration not available")
        print()
        
        # Training metadata
        print("🏋️ Training Information:")
        metadata = info.get('metadata', {})
        if metadata:
            print(f"  Training Completed: {metadata.get('training_completed_at', 'N/A')}")
            print(f"  Training Duration: {metadata.get('training_duration_seconds', 'N/A')} seconds")
            
            # Dataset information
            dataset_info = metadata.get('dataset_info', {})
            if dataset_info:
                print(f"  Dataset Source: {dataset_info.get('source', 'N/A')}")
                print(f"  Total Sequences: {dataset_info.get('num_sequences', 'N/A')}")
                print(f"  Training Samples: {dataset_info.get('train_samples', 'N/A')}")
                print(f"  Test Samples: {dataset_info.get('test_samples', 'N/A')}")
            
            # Performance metrics
            performance = metadata.get('performance_metrics', {})
            if performance:
                print(f"  Final Test MSE: {performance.get('final_test_mse', 'N/A')}")
                print(f"  Final Test MAE: {performance.get('final_test_mae', 'N/A')}")
                print(f"  Best Validation Loss: {performance.get('best_val_loss', 'N/A')}")
        else:
            print("  Training metadata not available")
        print()
        
        # System information
        print("💻 System Information:")
        system_info = metadata.get('system_info', {}) if metadata else {}
        if system_info:
            print(f"  Python Version: {system_info.get('python_version', 'N/A')}")
            print(f"  TensorFlow Version: {system_info.get('tensorflow_version', 'N/A')}")
            print(f"  Platform: {system_info.get('platform', 'N/A')}")
        else:
            print("  System information not available")
        print()
        
        # File information
        print("📁 File Information:")
        file_info = info.get('file_info', {})
        if file_info:
            total_size = 0
            for component, files in file_info.items():
                if isinstance(files, dict) and 'size_mb' in files:
                    size_mb = files['size_mb']
                    total_size += size_mb
                    print(f"  {component}: {size_mb:.2f} MB")
            print(f"  Total Size: {total_size:.2f} MB")
        else:
            print("  File information not available")
        print()
        
    except Exception as e:
        print(f"❌ Failed to get detailed model info: {e}")
        print()

def model_summary_example(models: List[Dict[str, Any]]):
    """Demonstrate human-readable model summaries."""
    print("📄 Model Summary Example")
    print("=" * 40)
    
    if not models:
        print("No models available for summary.")
        print()
        return
    
    manager = ModelManager()
    
    # Show summary for first few models
    for model in models[:2]:
        model_path = model['path']
        
        try:
            summary = manager.get_model_summary(model_path)
            print(summary)
            print("-" * 60)
            
        except Exception as e:
            print(f"❌ Failed to get summary for {model_path}: {e}")
            print()

def model_comparison_example(models: List[Dict[str, Any]]):
    """Demonstrate model comparison."""
    print("⚖️  Model Comparison Example")
    print("=" * 40)
    
    if len(models) < 2:
        print("Need at least 2 models for comparison.")
        print()
        return
    
    print("Comparing available models:")
    print()
    
    # Create comparison table
    print(f"{'Model':<25} {'Created':<20} {'Test MSE':<12} {'Window':<8} {'Type':<12}")
    print("-" * 85)
    
    for model in models:
        model_name = os.path.basename(model['path'])
        created = model.get('created_at', 'Unknown')[:19] if model.get('created_at') else 'Unknown'
        test_mse = f"{model.get('test_mse', 0):.6f}" if 'test_mse' in model else 'N/A'
        
        config = model.get('configuration', {})
        window_size = str(config.get('window_size', 'N/A'))
        reservoir_type = config.get('reservoir_type', 'N/A')
        
        print(f"{model_name:<25} {created:<20} {test_mse:<12} {window_size:<8} {reservoir_type:<12}")
    
    print()
    
    # Find best performing model
    valid_models = [m for m in models if 'test_mse' in m and m['test_mse'] is not None]
    if valid_models:
        best_model = min(valid_models, key=lambda x: x['test_mse'])
        print(f"🏆 Best performing model: {os.path.basename(best_model['path'])}")
        print(f"   Test MSE: {best_model['test_mse']:.6f}")
        print()

def cleanup_example():
    """Demonstrate model cleanup functionality."""
    print("🧹 Model Cleanup Example")
    print("=" * 40)
    
    try:
        manager = ModelManager()
        
        # Find cleanup candidates (dry run)
        print("Scanning for incomplete or corrupted models...")
        cleanup_candidates = manager.cleanup_incomplete_models(dry_run=True)
        
        if not cleanup_candidates:
            print("✅ No incomplete models found. All models appear to be complete.")
        else:
            print(f"⚠️  Found {len(cleanup_candidates)} incomplete model(s):")
            for candidate in cleanup_candidates:
                print(f"  - {candidate}")
            
            print()
            print("To actually remove these models, run:")
            print("python manage_models.py cleanup --no-dry-run")
        
        print()
        
    except Exception as e:
        print(f"❌ Cleanup scan failed: {e}")
        print()

def model_management_tips():
    """Provide tips for effective model management."""
    print("💡 Model Management Tips")
    print("=" * 40)
    
    tips = [
        "Use descriptive model names with timestamps for easy identification",
        "Regularly validate models to ensure integrity",
        "Monitor disk space usage - models can be large",
        "Keep training logs and metadata for reproducibility",
        "Clean up incomplete models to save disk space",
        "Back up your best performing models",
        "Use model comparison to track improvements over time",
        "Document model configurations for future reference"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")
    
    print()

def main():
    """Run all model management examples."""
    print("🚀 LSM Model Management Examples")
    print("=" * 50)
    print()
    
    try:
        # Discover models
        models = model_discovery_example()
        
        if models:
            # Run management examples
            model_validation_example(models)
            detailed_model_info_example(models)
            model_summary_example(models)
            model_comparison_example(models)
        
        # Cleanup example (always run)
        cleanup_example()
        
        # Tips
        model_management_tips()
        
        print("🎉 All model management examples completed!")
        print()
        
        if models:
            print("Next steps:")
            print("- Use the best performing model for inference")
            print("- Try the interactive mode with your chosen model")
            print("- Explore batch processing capabilities")
        else:
            print("Next steps:")
            print("- Train your first model: python main.py train")
            print("- Come back to explore model management features")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check the troubleshooting guide in README.md for help.")

if __name__ == "__main__":
    main()