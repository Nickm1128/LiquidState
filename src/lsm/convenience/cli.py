#!/usr/bin/env python3
"""
Command-line interface for the LSM convenience API.

This module provides a user-friendly CLI that leverages the convenience API
to make LSM training and inference accessible from the command line.
"""

import argparse
import os
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Optional

from ..utils.lsm_exceptions import ModelLoadError, InvalidInputError
from ..utils.lsm_logging import get_logger, setup_default_logging
from .generator import LSMGenerator
from .classifier import LSMClassifier
from .regressor import LSMRegressor
from .config import ConvenienceConfig
from pathlib import Path

logger = get_logger(__name__)


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


def train_generator_command(args) -> Dict[str, Any]:
    """Execute text generation training command."""
    print("Starting LSM text generation training with parameters:")
    print(f"  Data path: {args.data_path}")
    print(f"  Window size: {args.window_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Preset: {args.preset}")
    
    # Load preset configuration if specified
    config_params = {}
    if args.preset:
        config_params = ConvenienceConfig.get_preset(args.preset)
        print(f"  Using preset config: {config_params}")
    
    # Override with command line arguments
    config_params.update({
        'window_size': args.window_size,
        'embedding_dim': args.embedding_dim,
        'reservoir_type': args.reservoir_type,
        'system_message_support': args.system_messages,
        'random_state': args.seed
    })
    
    # Create generator
    generator = LSMGenerator(**config_params)
    
    # Load training data
    print(f"Loading training data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        raise InvalidInputError(f"Data path does not exist: {args.data_path}")
    
    # Simple text file loading for now
    with open(args.data_path, 'r', encoding='utf-8') as f:
        conversations = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(conversations)} conversation examples")
    
    # Train the model
    print("Training LSM generator...")
    generator.fit(
        conversations,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=args.validation_split
    )
    
    # Save the model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "lsm_generator")
        generator.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Save training metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'parameters': config_params,
            'data_path': args.data_path,
            'num_conversations': len(conversations)
        }
        
        metadata_path = os.path.join(args.output_dir, "training_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training metadata saved to {metadata_path}")
    
    return {'status': 'completed', 'model_path': model_path if args.output_dir else None}


def generate_command(args) -> Dict[str, Any]:
    """Execute text generation command."""
    print(f"Loading LSM generator from {args.model_path}")
    
    if not os.path.exists(args.model_path):
        raise ModelLoadError(args.model_path, "Model path does not exist")
    
    # Load the generator
    generator = LSMGenerator.load(args.model_path)
    
    if args.interactive:
        # Interactive chat mode
        print("Starting interactive chat mode. Type 'quit' to exit.")
        system_message = args.system_message if args.system_message else None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_input:
                    continue
                
                response = generator.generate(
                    user_input,
                    system_message=system_message,
                    max_length=args.max_length,
                    temperature=args.temperature
                )
                
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting chat mode...")
                break
            except Exception as e:
                print(f"Error generating response: {e}")
    
    else:
        # Single generation
        if not args.prompt:
            raise InvalidInputError("Prompt is required for non-interactive mode")
        
        print(f"Generating response to: '{args.prompt}'")
        
        response = generator.generate(
            args.prompt,
            system_message=args.system_message,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        print(f"Generated response: {response}")
        
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            output_file = os.path.join(args.output_dir, "generation_output.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Response: {response}\n")
            print(f"Output saved to {output_file}")
    
    return {'status': 'completed'}


def train_classifier_command(args) -> Dict[str, Any]:
    """Execute classification training command."""
    print("Starting LSM classification training with parameters:")
    print(f"  Data path: {args.data_path}")
    print(f"  Window size: {args.window_size}")
    print(f"  Embedding dim: {args.embedding_dim}")
    print(f"  Epochs: {args.epochs}")
    
    # Load preset configuration if specified
    config_params = {}
    if args.preset:
        config_params = ConvenienceConfig.get_preset(args.preset)
    
    config_params.update({
        'window_size': args.window_size,
        'embedding_dim': args.embedding_dim,
        'reservoir_type': args.reservoir_type,
        'random_state': args.seed
    })
    
    # Create classifier
    classifier = LSMClassifier(**config_params)
    
    # Load training data (expecting CSV format with text,label columns)
    print(f"Loading training data from {args.data_path}...")
    import pandas as pd
    
    data = pd.read_csv(args.data_path)
    if 'text' not in data.columns or 'label' not in data.columns:
        raise InvalidInputError("Data file must contain 'text' and 'label' columns")
    
    X = data['text'].tolist()
    y = data['label'].tolist()
    
    print(f"Loaded {len(X)} training examples with {len(set(y))} classes")
    
    # Train the classifier
    classifier.fit(X, y, epochs=args.epochs, batch_size=args.batch_size)
    
    # Save the model
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, "lsm_classifier")
        classifier.save(model_path)
        print(f"Model saved to {model_path}")
    
    return {'status': 'completed', 'model_path': model_path if args.output_dir else None}


def benchmark_command(args):
    """Run performance benchmarks."""
    try:
        from .benchmarks import run_quick_benchmark, run_full_benchmark
        
        print("LSM Convenience API Performance Benchmark")
        print("=" * 50)
        
        if args.quick:
            print("Running quick benchmark...")
            results = run_quick_benchmark()
        else:
            print("Running comprehensive benchmark...")
            output_dir = Path(args.output) if args.output else None
            results = run_full_benchmark(output_dir=output_dir)
        
        # Display summary
        if 'summary' in results:
            summary = results['summary']
            print(f"\nBenchmark Summary:")
            print(f"Total benchmarks: {summary.get('total_benchmarks', 0)}")
            
            if 'average_performance_overhead' in summary:
                print(f"Average performance overhead: {summary['average_performance_overhead']}")
            
            if 'recommendations' in summary:
                print("\nRecommendations:")
                for rec in summary['recommendations']:
                    print(f"  - {rec}")
        
        print("\nBenchmark completed successfully!")
        
    except ImportError:
        print("Error: Performance benchmarking components not available.")
        print("Please ensure all dependencies are installed.")
        return 1
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return 1
    
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LSM Convenience API Command Line Interface",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Global arguments
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Directory to save outputs')
    parser.add_argument('--preset', type=str, choices=['fast', 'balanced', 'quality'],
                       help='Use preset configuration')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train generator command
    train_gen_parser = subparsers.add_parser('train-generator', 
                                           help='Train text generation model')
    train_gen_parser.add_argument('--data-path', type=str, required=True,
                                help='Path to training data file')
    train_gen_parser.add_argument('--window-size', type=int, default=10,
                                help='Size of sequence window')
    train_gen_parser.add_argument('--embedding-dim', type=int, default=128,
                                help='Dimension of token embeddings')
    train_gen_parser.add_argument('--epochs', type=int, default=50,
                                help='Number of training epochs')
    train_gen_parser.add_argument('--batch-size', type=int, default=32,
                                help='Training batch size')
    train_gen_parser.add_argument('--validation-split', type=float, default=0.2,
                                help='Fraction of data for validation')
    train_gen_parser.add_argument('--reservoir-type', type=str, default='hierarchical',
                                choices=['standard', 'hierarchical', 'attentive', 'echo_state', 'deep'],
                                help='Type of reservoir architecture')
    train_gen_parser.add_argument('--system-messages', action='store_true',
                                help='Enable system message support')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate text responses')
    gen_parser.add_argument('--model-path', type=str, required=True,
                          help='Path to trained generator model')
    gen_parser.add_argument('--prompt', type=str,
                          help='Input prompt for generation')
    gen_parser.add_argument('--system-message', type=str,
                          help='System message for context')
    gen_parser.add_argument('--max-length', type=int, default=50,
                          help='Maximum response length')
    gen_parser.add_argument('--temperature', type=float, default=1.0,
                          help='Generation temperature')
    gen_parser.add_argument('--interactive', action='store_true',
                          help='Start interactive chat mode')
    
    # Train classifier command
    train_cls_parser = subparsers.add_parser('train-classifier',
                                           help='Train classification model')
    train_cls_parser.add_argument('--data-path', type=str, required=True,
                                help='Path to training data CSV file')
    train_cls_parser.add_argument('--window-size', type=int, default=10,
                                help='Size of sequence window')
    train_cls_parser.add_argument('--embedding-dim', type=int, default=128,
                                help='Dimension of token embeddings')
    train_cls_parser.add_argument('--epochs', type=int, default=50,
                                help='Number of training epochs')
    train_cls_parser.add_argument('--batch-size', type=int, default=32,
                                help='Training batch size')
    train_cls_parser.add_argument('--reservoir-type', type=str, default='standard',
                                choices=['standard', 'hierarchical', 'attentive', 'echo_state', 'deep'],
                                help='Type of reservoir architecture')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark',
                                           help='Run performance benchmarks')
    benchmark_parser.add_argument('--quick', action='store_true',
                                help='Run quick benchmark (faster, less comprehensive)')
    benchmark_parser.add_argument('--output', type=str,
                                help='Directory to save benchmark results')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Setup logging
    setup_default_logging()
    
    # Setup GPU
    setup_gpu()
    
    # Execute command
    try:
        if args.command == 'train-generator':
            results = train_generator_command(args)
            
        elif args.command == 'generate':
            results = generate_command(args)
            
        elif args.command == 'train-classifier':
            results = train_classifier_command(args)
            
        elif args.command == 'benchmark':
            results = benchmark_command(args)
            
        print(f"\nCommand '{args.command}' completed successfully!")
        
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        print(f"Error executing command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()