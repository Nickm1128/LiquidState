#!/usr/bin/env python3
"""
CLI utility for managing LSM models using ModelManager.

This script provides a command-line interface for discovering, validating,
and managing trained LSM models in the workspace.
"""

import argparse
import sys
from .model_manager import ModelManager

def main():
    """Main CLI interface for model management."""
    parser = argparse.ArgumentParser(
        description="LSM Model Management Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--models-dir', type=str, default='.',
                       help='Root directory to search for models')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available models')
    list_parser.add_argument('--detailed', action='store_true',
                           help='Show detailed information for each model')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed info about a specific model')
    info_parser.add_argument('model_path', help='Path to the model directory')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a specific model')
    validate_parser.add_argument('model_path', help='Path to the model directory')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up incomplete models')
    cleanup_parser.add_argument('--dry-run', action='store_true', default=True,
                               help='Show what would be cleaned up without actually deleting')
    cleanup_parser.add_argument('--force', action='store_true',
                               help='Actually perform cleanup (overrides --dry-run)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Initialize ModelManager
    try:
        manager = ModelManager(args.models_dir)
    except Exception as e:
        print(f"❌ Failed to initialize ModelManager: {e}")
        return 1
    
    # Execute commands
    try:
        if args.command == 'list':
            return handle_list_command(manager, args)
        elif args.command == 'info':
            return handle_info_command(manager, args)
        elif args.command == 'validate':
            return handle_validate_command(manager, args)
        elif args.command == 'cleanup':
            return handle_cleanup_command(manager, args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return 1
    
    except Exception as e:
        print(f"❌ Error executing command: {e}")
        return 1

def handle_list_command(manager: ModelManager, args) -> int:
    """Handle the list command."""
    print("🔍 Scanning for available models...")
    
    if args.detailed:
        # Show detailed summary
        summary = manager.list_models_summary()
        print(summary)
    else:
        # Show simple list
        models = manager.list_available_models()
        
        if not models:
            print("No valid models found in the workspace.")
            return 0
        
        print(f"\nFound {len(models)} valid models:\n")
        
        for i, model in enumerate(models, 1):
            status = "✅" if model['is_valid'] else "❌"
            size = model.get('total_size_mb', 0)
            created = model.get('created_at', 'Unknown')
            
            print(f"{i}. {status} {model['name']}")
            print(f"   Size: {size:.1f} MB")
            print(f"   Created: {created}")
            
            if 'architecture' in model:
                arch = model['architecture']
                print(f"   Type: {arch.get('reservoir_type', 'unknown')} reservoir")
            
            print()
    
    return 0

def handle_info_command(manager: ModelManager, args) -> int:
    """Handle the info command."""
    print(f"📊 Getting detailed information for: {args.model_path}")
    
    info = manager.get_model_info(args.model_path)
    
    if not info.get('is_valid', False):
        print(f"\n❌ Invalid Model: {info.get('name', 'Unknown')}")
        if 'error' in info:
            print(f"Error: {info['error']}")
        if info.get('validation_errors'):
            print("Validation Errors:")
            for error in info['validation_errors']:
                print(f"  - {error}")
        return 1
    
    # Display comprehensive information
    print(f"\n✅ {info['name']}")
    print("=" * 50)
    
    # Basic info
    print(f"Path: {info['path']}")
    print(f"Size: {info.get('total_size_mb', 0):.1f} MB")
    print(f"Valid: {'Yes' if info['is_valid'] else 'No'}")
    
    # Architecture
    if 'architecture' in info:
        arch = info['architecture']
        print(f"\nArchitecture:")
        print(f"  Reservoir Type: {arch.get('reservoir_type', 'unknown')}")
        print(f"  Window Size: {arch.get('window_size', 'unknown')}")
        print(f"  Embedding Dim: {arch.get('embedding_dim', 'unknown')}")
        print(f"  Multi-channel: {arch.get('use_multichannel', 'unknown')}")
    
    # Training info
    if 'training_metadata' in info:
        meta = info['training_metadata']
        print(f"\nTraining:")
        print(f"  Completed: {meta.get('completed_at', 'unknown')}")
        print(f"  Duration: {meta.get('duration_seconds', 0):.1f} seconds")
        
        if 'performance_metrics' in meta:
            perf = meta['performance_metrics']
            print(f"  Test MSE: {perf.get('final_test_mse', 'unknown')}")
            print(f"  Test MAE: {perf.get('final_test_mae', 'unknown')}")
        
        if 'dataset_info' in meta:
            dataset = meta['dataset_info']
            print(f"  Dataset: {dataset.get('source', 'unknown')}")
            print(f"  Samples: {dataset.get('num_sequences', 'unknown')}")
    
    # Tokenizer info
    if 'tokenizer' in info:
        tok = info['tokenizer']
        print(f"\nTokenizer:")
        print(f"  Vocabulary Size: {tok.get('vocabulary_size', 'unknown')}")
        print(f"  Max Features: {tok.get('max_features', 'unknown')}")
        print(f"  Fitted: {'Yes' if tok.get('is_fitted', False) else 'No'}")
    
    # Components
    if 'components' in info:
        comp = info['components']
        print(f"\nComponents:")
        for component, present in comp.items():
            status = "✅" if present else "❌"
            print(f"  {status} {component}")
    
    return 0

def handle_validate_command(manager: ModelManager, args) -> int:
    """Handle the validate command."""
    print(f"🔍 Validating model: {args.model_path}")
    
    is_valid, errors = manager.validate_model(args.model_path)
    
    if is_valid:
        print(f"✅ Model is valid!")
        
        # Show non-critical warnings if any
        warnings = [err for err in errors if err.endswith("(non-critical)")]
        if warnings:
            print(f"\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
    else:
        print(f"❌ Model validation failed!")
        print(f"\nErrors found:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    return 0

def handle_cleanup_command(manager: ModelManager, args) -> int:
    """Handle the cleanup command."""
    dry_run = args.dry_run and not args.force
    
    if dry_run:
        print("🔍 Analyzing incomplete models (dry run)...")
    else:
        print("🧹 Cleaning up incomplete models...")
        confirm = input("Are you sure you want to delete incomplete models? (y/N): ")
        if confirm.lower() != 'y':
            print("Cleanup cancelled.")
            return 0
    
    cleanup_candidates = manager.cleanup_incomplete_models(dry_run=dry_run)
    
    if not cleanup_candidates:
        print("✅ No incomplete models found.")
        return 0
    
    if dry_run:
        print(f"\nFound {len(cleanup_candidates)} incomplete models:")
        for path in cleanup_candidates:
            print(f"  - {path}")
        print(f"\nTo actually clean up, run: {sys.argv[0]} cleanup --force")
    else:
        print(f"✅ Cleaned up {len(cleanup_candidates)} incomplete models.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())