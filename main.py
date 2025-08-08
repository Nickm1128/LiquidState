#!/usr/bin/env python3
"""
Main CLI interface for the Sparse Sine-Activated Liquid State Machine.

This script provides a command-line interface to train and evaluate
the LSM model for next-token prediction on dialogue data.
"""

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

from lsm_exceptions import ModelLoadError, InvalidInputError
from lsm_logging import get_logger, setup_default_logging

logger = get_logger(__name__)

import numpy as np
import tensorflow as tf

from train import run_training, LSMTrainer, set_random_seeds
from data_loader import load_data

def setup_gpu():
    """Configure GPU if available."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Memory growth enabled.")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPUs found. Using CPU.")

def train_command(args) -> Dict[str, Any]:
    """Execute training command."""
    print("Starting LSM training with parameters:")
    print(f"  Window size: {args.window_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Test size: {args.test_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Reservoir type: {args.reservoir_type}")
    print(f"  Use CNN attention: {args.use_attention}")
    print(f"  Random seed: {args.seed}")
    
    # Set random seed
    set_random_seeds(args.seed)
    
    # Parse reservoir configuration if provided
    reservoir_config = {}
    if hasattr(args, 'reservoir_config') and args.reservoir_config:
        try:
            reservoir_config = json.loads(args.reservoir_config)
            print(f"  Reservoir config: {reservoir_config}")
        except json.JSONDecodeError as e:
            print(f"  Error parsing reservoir config: {e}")
            print("  Using default configuration for reservoir type")
    
    # Run training
    results = run_training(
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        test_size=args.test_size,
        embedding_dim=args.embedding_dim,
        reservoir_type=args.reservoir_type,
        reservoir_config=reservoir_config,
        use_attention=args.use_attention
    )
    
    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, "training_results.json")
        
        # Convert numpy types to regular Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                serializable_results[key] = float(value)
            elif isinstance(value, dict):
                serializable_results[key] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    return results

def evaluate_command(args) -> Dict[str, Any]:
    """Execute evaluation command."""
    if not os.path.exists(args.model_path):
        raise ModelLoadError(args.model_path, "Model path does not exist")
    
    print(f"Loading model from {args.model_path}")
    
    # Initialize trainer and load models
    trainer = LSMTrainer(
        window_size=args.window_size,
        embedding_dim=args.embedding_dim
    )
    trainer.load_models(args.model_path)
    
    # Load test data
    print("Loading test data...")
    _, _, X_test, y_test, _ = load_data(
        window_size=args.window_size,
        test_size=args.test_size,
        embedding_dim=args.embedding_dim
    )
    
    # Make predictions
    print("Making predictions...")
    y_pred = trainer.predict(X_test)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(mse)
    
    results = {
        'test_mse': float(mse),
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'num_samples': len(y_test)
    }
    
    print(f"Evaluation results:")
    print(f"  Test MSE: {results['test_mse']:.6f}")
    print(f"  Test MAE: {results['test_mae']:.6f}")
    print(f"  Test RMSE: {results['test_rmse']:.6f}")
    print(f"  Number of samples: {results['num_samples']}")
    
    return results

def data_info_command(args) -> Dict[str, Any]:
    """Display information about the dataset."""
    print("Loading dataset information...")
    
    try:
        X_train, y_train, X_test, y_test, _ = load_data(
            window_size=args.window_size,
            test_size=args.test_size,
            embedding_dim=args.embedding_dim
        )
        
        info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'window_size': args.window_size,
            'embedding_dim': args.embedding_dim,
            'input_shape': list(X_train.shape),
            'output_shape': list(y_train.shape),
            'train_embedding_stats': {
                'mean': float(np.mean(X_train)),
                'std': float(np.std(X_train)),
                'min': float(np.min(X_train)),
                'max': float(np.max(X_train))
            },
            'test_embedding_stats': {
                'mean': float(np.mean(X_test)),
                'std': float(np.std(X_test)),
                'min': float(np.min(X_test)),
                'max': float(np.max(X_test))
            }
        }
        
        print("Dataset Information:")
        print(f"  Training samples: {info['train_samples']}")
        print(f"  Test samples: {info['test_samples']}")
        print(f"  Window size: {info['window_size']}")
        print(f"  Embedding dimension: {info['embedding_dim']}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Output shape: {info['output_shape']}")
        print(f"  Training data stats: mean={info['train_embedding_stats']['mean']:.4f}, "
              f"std={info['train_embedding_stats']['std']:.4f}")
        
        return info
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {'error': str(e)}

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Sparse Sine-Activated Liquid State Machine for Next-Token Prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save outputs')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the LSM model')
    train_parser.add_argument('--window-size', type=int, default=10,
                             help='Size of sequence window')
    train_parser.add_argument('--embedding-dim', type=int, default=128,
                             help='Dimension of token embeddings')
    train_parser.add_argument('--test-size', type=float, default=0.2,
                             help='Fraction of data to use for testing')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Training batch size')
    train_parser.add_argument('--epochs', type=int, default=20,
                             help='Number of training epochs')
    train_parser.add_argument('--reservoir-type', type=str, default='standard',
                             choices=['standard', 'hierarchical', 'attentive', 'echo_state', 'deep'],
                             help='Type of reservoir architecture to use')
    train_parser.add_argument('--reservoir-config', type=str, default=None,
                             help='JSON string with reservoir configuration parameters')
    train_parser.add_argument('--use-attention', action='store_true', default=True,
                             help='Enable spatial attention in CNN (enabled by default)')
    train_parser.add_argument('--no-attention', dest='use_attention', action='store_false',
                             help='Disable spatial attention in CNN')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate trained model')
    eval_parser.add_argument('--model-path', type=str, required=True,
                           help='Path to saved model directory')
    eval_parser.add_argument('--window-size', type=int, default=10,
                           help='Size of sequence window')
    eval_parser.add_argument('--embedding-dim', type=int, default=128,
                           help='Dimension of token embeddings')
    eval_parser.add_argument('--test-size', type=float, default=0.2,
                           help='Fraction of data to use for testing')
    
    # Data info command
    info_parser = subparsers.add_parser('data-info', help='Display dataset information')
    info_parser.add_argument('--window-size', type=int, default=10,
                           help='Size of sequence window')
    info_parser.add_argument('--embedding-dim', type=int, default=128,
                           help='Dimension of token embeddings')
    info_parser.add_argument('--test-size', type=float, default=0.2,
                           help='Fraction of data to use for testing')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup GPU
    setup_gpu()
    
    # Execute command
    try:
        if args.command == 'train':
            results = train_command(args)
            
        elif args.command == 'evaluate':
            results = evaluate_command(args)
            
        elif args.command == 'data-info':
            results = data_info_command(args)
            
        print(f"\nCommand '{args.command}' completed successfully!")
        
    except Exception as e:
        print(f"Error executing command '{args.command}': {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
