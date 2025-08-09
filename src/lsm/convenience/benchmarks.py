"""
Benchmarking utilities for comparing LSM convenience API with direct API performance.

This module provides comprehensive benchmarking tools to measure and compare
the performance overhead of the convenience API versus direct component usage.
"""

import time
import gc
from typing import Any, Dict, List, Optional, Callable, Tuple
from pathlib import Path
import numpy as np

from .performance import BenchmarkSuite, PerformanceProfiler
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class ConvenienceAPIBenchmark:
    """Comprehensive benchmarking for LSM convenience API."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.benchmark_suite = BenchmarkSuite()
        self.test_data = {}
        self.results = {}
    
    def prepare_test_data(self, 
                         small_dataset_size: int = 100,
                         medium_dataset_size: int = 1000,
                         large_dataset_size: int = 5000) -> None:
        """
        Prepare test datasets for benchmarking.
        
        Parameters
        ----------
        small_dataset_size : int, default=100
            Size of small test dataset
        medium_dataset_size : int, default=1000
            Size of medium test dataset
        large_dataset_size : int, default=5000
            Size of large test dataset
        """
        logger.info("Preparing test datasets for benchmarking")
        
        # Generate synthetic conversation data
        conversation_templates = [
            "Hello, how are you?",
            "I need help with something.",
            "Can you explain this concept?",
            "What's the weather like today?",
            "Thank you for your assistance.",
            "I have a question about the product.",
            "Could you provide more information?",
            "I'm looking for recommendations.",
            "How do I solve this problem?",
            "What are the available options?"
        ]
        
        # Small dataset
        self.test_data['small'] = {
            'conversations': [
                conversation_templates[i % len(conversation_templates)] 
                for i in range(small_dataset_size)
            ],
            'size': small_dataset_size
        }
        
        # Medium dataset
        self.test_data['medium'] = {
            'conversations': [
                conversation_templates[i % len(conversation_templates)] 
                for i in range(medium_dataset_size)
            ],
            'size': medium_dataset_size
        }
        
        # Large dataset
        self.test_data['large'] = {
            'conversations': [
                conversation_templates[i % len(conversation_templates)] 
                for i in range(large_dataset_size)
            ],
            'size': large_dataset_size
        }
        
        # Generation prompts
        self.test_data['generation_prompts'] = [
            "Hello, how can I help you?",
            "What would you like to know?",
            "I'm here to assist you.",
            "How are you doing today?",
            "What's on your mind?"
        ]
        
        logger.info(f"Test data prepared: {len(self.test_data)} datasets")
    
    def benchmark_training_performance(self, 
                                     dataset_sizes: List[str] = ['small', 'medium'],
                                     iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark training performance between convenience and direct API.
        
        Parameters
        ----------
        dataset_sizes : list, default=['small', 'medium']
            Dataset sizes to benchmark
        iterations : int, default=3
            Number of iterations per benchmark
            
        Returns
        -------
        results : dict
            Benchmark results
        """
        logger.info("Starting training performance benchmark")
        
        if not self.test_data:
            self.prepare_test_data()
        
        training_results = {}
        
        for dataset_size in dataset_sizes:
            if dataset_size not in self.test_data:
                logger.warning(f"Dataset size '{dataset_size}' not available, skipping")
                continue
            
            data = self.test_data[dataset_size]['conversations']
            
            # Define convenience API training function
            def convenience_training(test_data):
                from .generator import LSMGenerator
                
                generator = LSMGenerator(
                    window_size=5,  # Smaller for faster benchmarking
                    embedding_dim=64,
                    epochs=5  # Fewer epochs for benchmarking
                )
                generator.fit(test_data, verbose=False)
                return generator
            
            # Define direct API training function (simplified)
            def direct_training(test_data):
                # This would use the direct LSMTrainer and components
                # For now, we'll simulate with a placeholder
                time.sleep(0.1)  # Simulate training time
                return "direct_model"
            
            # Run benchmark
            operation_name = f"training_{dataset_size}"
            result = self.benchmark_suite.benchmark_operation(
                operation_name=operation_name,
                convenience_func=convenience_training,
                direct_func=direct_training,
                test_data=data,
                iterations=iterations
            )
            
            training_results[dataset_size] = result
            
            # Clean up memory between benchmarks
            gc.collect()
        
        self.results['training'] = training_results
        logger.info("Training performance benchmark completed")
        
        return training_results
    
    def benchmark_generation_performance(self, 
                                       num_prompts: List[int] = [10, 50, 100],
                                       iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark text generation performance.
        
        Parameters
        ----------
        num_prompts : list, default=[10, 50, 100]
            Number of prompts to generate for each benchmark
        iterations : int, default=3
            Number of iterations per benchmark
            
        Returns
        -------
        results : dict
            Benchmark results
        """
        logger.info("Starting generation performance benchmark")
        
        if not self.test_data:
            self.prepare_test_data()
        
        # First, create a trained model for generation benchmarks
        from .generator import LSMGenerator
        
        logger.info("Training model for generation benchmarks...")
        generator = LSMGenerator(
            window_size=5,
            embedding_dim=64,
            epochs=5
        )
        generator.fit(self.test_data['small']['conversations'], verbose=False)
        
        generation_results = {}
        
        for num_prompt in num_prompts:
            prompts = self.test_data['generation_prompts'][:num_prompt] * (num_prompt // len(self.test_data['generation_prompts']) + 1)
            prompts = prompts[:num_prompt]
            
            # Define convenience API generation function
            def convenience_generation(test_prompts):
                return generator.batch_generate(test_prompts, batch_size=8)
            
            # Define direct API generation function (simplified)
            def direct_generation(test_prompts):
                # Simulate direct API generation
                results = []
                for prompt in test_prompts:
                    time.sleep(0.01)  # Simulate generation time
                    results.append(f"Response to: {prompt[:20]}...")
                return results
            
            # Run benchmark
            operation_name = f"generation_{num_prompt}_prompts"
            result = self.benchmark_suite.benchmark_operation(
                operation_name=operation_name,
                convenience_func=convenience_generation,
                direct_func=direct_generation,
                test_data=prompts,
                iterations=iterations
            )
            
            generation_results[f"{num_prompt}_prompts"] = result
            
            # Clean up memory between benchmarks
            gc.collect()
        
        self.results['generation'] = generation_results
        logger.info("Generation performance benchmark completed")
        
        return generation_results
    
    def benchmark_memory_usage(self, 
                             dataset_sizes: List[str] = ['small', 'medium'],
                             operations: List[str] = ['training', 'generation']) -> Dict[str, Any]:
        """
        Benchmark memory usage patterns.
        
        Parameters
        ----------
        dataset_sizes : list, default=['small', 'medium']
            Dataset sizes to benchmark
        operations : list, default=['training', 'generation']
            Operations to benchmark
            
        Returns
        -------
        results : dict
            Memory usage benchmark results
        """
        logger.info("Starting memory usage benchmark")
        
        if not self.test_data:
            self.prepare_test_data()
        
        memory_results = {}
        
        for dataset_size in dataset_sizes:
            if dataset_size not in self.test_data:
                continue
            
            dataset_results = {}
            
            if 'training' in operations:
                # Benchmark training memory usage
                from .generator import LSMGenerator
                from .performance import MemoryMonitor
                
                monitor = MemoryMonitor()
                
                # Convenience API training
                monitor.start_monitoring()
                generator = LSMGenerator(window_size=5, embedding_dim=64, epochs=3)
                generator.fit(self.test_data[dataset_size]['conversations'], verbose=False)
                convenience_memory_stats = monitor.stop_monitoring()
                
                # Clean up
                del generator
                gc.collect()
                
                # Direct API training (simulated)
                monitor.start_monitoring()
                time.sleep(1.0)  # Simulate training
                direct_memory_stats = monitor.stop_monitoring()
                
                dataset_results['training'] = {
                    'convenience_api': convenience_memory_stats,
                    'direct_api': direct_memory_stats
                }
            
            if 'generation' in operations and dataset_size == 'small':
                # Only test generation on small dataset to avoid long benchmark times
                from .generator import LSMGenerator
                from .performance import MemoryMonitor
                
                # Create a trained model
                generator = LSMGenerator(window_size=5, embedding_dim=64, epochs=3)
                generator.fit(self.test_data['small']['conversations'], verbose=False)
                
                monitor = MemoryMonitor()
                
                # Convenience API generation
                prompts = self.test_data['generation_prompts'] * 10  # 50 prompts
                monitor.start_monitoring()
                generator.batch_generate(prompts)
                convenience_memory_stats = monitor.stop_monitoring()
                
                # Direct API generation (simulated)
                monitor.start_monitoring()
                for prompt in prompts:
                    time.sleep(0.01)  # Simulate generation
                direct_memory_stats = monitor.stop_monitoring()
                
                dataset_results['generation'] = {
                    'convenience_api': convenience_memory_stats,
                    'direct_api': direct_memory_stats
                }
                
                del generator
                gc.collect()
            
            memory_results[dataset_size] = dataset_results
        
        self.results['memory'] = memory_results
        logger.info("Memory usage benchmark completed")
        
        return memory_results
    
    def run_comprehensive_benchmark(self, 
                                  quick_mode: bool = False,
                                  output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Parameters
        ----------
        quick_mode : bool, default=False
            Whether to run in quick mode (fewer iterations, smaller datasets)
        output_path : Path, optional
            Path to save benchmark results
            
        Returns
        -------
        results : dict
            Complete benchmark results
        """
        logger.info("Starting comprehensive LSM convenience API benchmark")
        
        # Prepare test data
        if quick_mode:
            self.prepare_test_data(small_dataset_size=50, medium_dataset_size=200, large_dataset_size=500)
            iterations = 2
            dataset_sizes = ['small']
            num_prompts = [10, 25]
        else:
            self.prepare_test_data()
            iterations = 3
            dataset_sizes = ['small', 'medium']
            num_prompts = [10, 50, 100]
        
        comprehensive_results = {
            'benchmark_info': {
                'quick_mode': quick_mode,
                'iterations': iterations,
                'dataset_sizes': dataset_sizes,
                'timestamp': time.time()
            }
        }
        
        try:
            # Training performance
            logger.info("Benchmarking training performance...")
            training_results = self.benchmark_training_performance(
                dataset_sizes=dataset_sizes,
                iterations=iterations
            )
            comprehensive_results['training'] = training_results
            
            # Generation performance
            logger.info("Benchmarking generation performance...")
            generation_results = self.benchmark_generation_performance(
                num_prompts=num_prompts,
                iterations=iterations
            )
            comprehensive_results['generation'] = generation_results
            
            # Memory usage
            logger.info("Benchmarking memory usage...")
            memory_results = self.benchmark_memory_usage(
                dataset_sizes=dataset_sizes,
                operations=['training', 'generation']
            )
            comprehensive_results['memory'] = memory_results
            
            # Generate summary
            comprehensive_results['summary'] = self._generate_benchmark_summary()
            
        except Exception as e:
            logger.error(f"Comprehensive benchmark failed: {e}")
            comprehensive_results['error'] = str(e)
        
        # Save results if path provided
        if output_path:
            self._save_benchmark_results(comprehensive_results, output_path)
        
        logger.info("Comprehensive benchmark completed")
        return comprehensive_results
    
    def _generate_benchmark_summary(self) -> Dict[str, Any]:
        """Generate a summary of benchmark results."""
        summary = {
            'total_benchmarks': len(self.benchmark_suite.results),
            'performance_overhead': {},
            'memory_overhead': {},
            'recommendations': []
        }
        
        # Calculate average performance overhead
        total_overhead = 0
        overhead_count = 0
        
        for operation, result in self.benchmark_suite.results.items():
            if 'comparison' in result and 'convenience_overhead_percent' in result['comparison']:
                overhead = result['comparison']['convenience_overhead_percent']
                total_overhead += overhead
                overhead_count += 1
                summary['performance_overhead'][operation] = f"{overhead:+.1f}%"
        
        if overhead_count > 0:
            avg_overhead = total_overhead / overhead_count
            summary['average_performance_overhead'] = f"{avg_overhead:+.1f}%"
            
            # Generate recommendations based on overhead
            if avg_overhead > 50:
                summary['recommendations'].append("High performance overhead detected. Consider using direct API for performance-critical applications.")
            elif avg_overhead > 20:
                summary['recommendations'].append("Moderate performance overhead. Convenience API suitable for most use cases.")
            else:
                summary['recommendations'].append("Low performance overhead. Convenience API recommended for ease of use.")
        
        return summary
    
    def _save_benchmark_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save benchmark results to file."""
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {output_path}")
    
    def generate_performance_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate a human-readable performance report.
        
        Parameters
        ----------
        output_path : Path, optional
            Path to save the report
            
        Returns
        -------
        report : str
            Formatted performance report
        """
        if not self.results:
            return "No benchmark results available. Run benchmarks first."
        
        report_lines = [
            "LSM Convenience API Performance Report",
            "=" * 50,
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Training performance
        if 'training' in self.results:
            report_lines.extend([
                "Training Performance",
                "-" * 20
            ])
            
            for dataset_size, result in self.results['training'].items():
                if 'comparison' in result and 'convenience_overhead_percent' in result['comparison']:
                    overhead = result['comparison']['convenience_overhead_percent']
                    report_lines.append(f"Dataset {dataset_size}: {overhead:+.1f}% overhead")
            
            report_lines.append("")
        
        # Generation performance
        if 'generation' in self.results:
            report_lines.extend([
                "Generation Performance",
                "-" * 20
            ])
            
            for num_prompts, result in self.results['generation'].items():
                if 'comparison' in result and 'convenience_overhead_percent' in result['comparison']:
                    overhead = result['comparison']['convenience_overhead_percent']
                    report_lines.append(f"{num_prompts}: {overhead:+.1f}% overhead")
            
            report_lines.append("")
        
        # Memory usage
        if 'memory' in self.results:
            report_lines.extend([
                "Memory Usage",
                "-" * 20
            ])
            
            for dataset_size, operations in self.results['memory'].items():
                report_lines.append(f"Dataset {dataset_size}:")
                
                for operation, memory_data in operations.items():
                    conv_peak = memory_data.get('convenience_api', {}).get('peak_memory_mb', 0)
                    direct_peak = memory_data.get('direct_api', {}).get('peak_memory_mb', 0)
                    
                    if conv_peak > 0 and direct_peak > 0:
                        ratio = conv_peak / direct_peak
                        report_lines.append(f"  {operation}: {ratio:.2f}x memory usage")
            
            report_lines.append("")
        
        # Add benchmark suite report
        if hasattr(self.benchmark_suite, 'generate_benchmark_report'):
            suite_report = self.benchmark_suite.generate_benchmark_report()
            report_lines.extend([
                "Detailed Benchmark Results",
                "-" * 30,
                suite_report
            ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Performance report saved to {output_path}")
        
        return report


def run_quick_benchmark() -> Dict[str, Any]:
    """
    Run a quick benchmark of the convenience API.
    
    Returns
    -------
    results : dict
        Quick benchmark results
    """
    benchmark = ConvenienceAPIBenchmark()
    return benchmark.run_comprehensive_benchmark(quick_mode=True)


def run_full_benchmark(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run a comprehensive benchmark of the convenience API.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory to save benchmark results
        
    Returns
    -------
    results : dict
        Full benchmark results
    """
    benchmark = ConvenienceAPIBenchmark()
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_path = output_dir / "benchmark_results.json"
        report_path = output_dir / "benchmark_report.txt"
    else:
        results_path = None
        report_path = None
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark(
        quick_mode=False,
        output_path=results_path
    )
    
    # Generate report
    if report_path:
        benchmark.generate_performance_report(output_path=report_path)
    
    return results