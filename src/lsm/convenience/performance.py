"""
Performance monitoring and optimization utilities for the LSM convenience API.

This module provides automatic memory management, performance logging, benchmarking
utilities, and optimization features to ensure the convenience API performs well
while maintaining ease of use.
"""

import gc
import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import numpy as np
import json
from datetime import datetime, timedelta

from ..utils.lsm_logging import get_logger, log_performance
from ..utils.lsm_exceptions import LSMError, InsufficientMemoryError

logger = get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_before: Optional[float] = None
    memory_after: Optional[float] = None
    memory_peak: Optional[float] = None
    memory_delta: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_memory: Optional[float] = None
    data_size: Optional[int] = None
    batch_size: Optional[int] = None
    model_params: Optional[Dict[str, Any]] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
        
        if self.memory_before and self.memory_after:
            self.memory_delta = self.memory_after - self.memory_before
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'operation': self.operation,
            'duration': self.duration,
            'memory_before_mb': self.memory_before,
            'memory_after_mb': self.memory_after,
            'memory_peak_mb': self.memory_peak,
            'memory_delta_mb': self.memory_delta,
            'cpu_percent': self.cpu_percent,
            'gpu_memory_mb': self.gpu_memory,
            'data_size': self.data_size,
            'batch_size': self.batch_size,
            'model_params': self.model_params,
            'custom_metrics': self.custom_metrics,
            'timestamp': datetime.now().isoformat()
        }


class MemoryMonitor:
    """Monitor and manage memory usage during operations."""
    
    def __init__(self, check_interval: float = 1.0, memory_threshold: float = 0.9):
        """
        Initialize memory monitor.
        
        Parameters
        ----------
        check_interval : float, default=1.0
            Interval in seconds between memory checks
        memory_threshold : float, default=0.9
            Memory usage threshold (0-1) to trigger warnings
        """
        self.check_interval = check_interval
        self.memory_threshold = memory_threshold
        self.monitoring = False
        self.peak_memory = 0.0
        self.memory_history = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.peak_memory = 0.0
        self.memory_history.clear()
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.debug("Memory monitoring started")
    
    def stop_monitoring(self) -> Dict[str, float]:
        """
        Stop memory monitoring and return statistics.
        
        Returns
        -------
        stats : dict
            Memory usage statistics
        """
        if not self.monitoring:
            return {}
        
        self.monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        
        stats = {
            'peak_memory_mb': self.peak_memory,
            'avg_memory_mb': np.mean(self.memory_history) if self.memory_history else 0.0,
            'memory_samples': len(self.memory_history)
        }
        
        logger.debug("Memory monitoring stopped", **stats)
        return stats
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_event.wait(self.check_interval):
            try:
                current_memory = self.get_current_memory_mb()
                self.memory_history.append(current_memory)
                
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                
                # Check for memory threshold
                memory_percent = psutil.virtual_memory().percent / 100.0
                if memory_percent > self.memory_threshold:
                    logger.warning(
                        f"High memory usage detected: {memory_percent:.1%}",
                        memory_percent=memory_percent,
                        current_memory_mb=current_memory
                    )
                    
                    # Trigger garbage collection
                    gc.collect()
                
            except Exception as e:
                logger.warning(f"Memory monitoring error: {e}")
    
    @staticmethod
    def get_current_memory_mb() -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    @staticmethod
    def get_available_memory_mb() -> float:
        """Get available system memory in MB."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return 0.0


class PerformanceProfiler:
    """Profile performance of LSM operations."""
    
    def __init__(self, enable_memory_monitoring: bool = True):
        """
        Initialize performance profiler.
        
        Parameters
        ----------
        enable_memory_monitoring : bool, default=True
            Whether to enable detailed memory monitoring
        """
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_monitor = MemoryMonitor() if enable_memory_monitoring else None
        self.metrics_history = []
        self.current_metrics = None
    
    @contextmanager
    def profile_operation(self, operation: str, **context):
        """
        Context manager for profiling operations.
        
        Parameters
        ----------
        operation : str
            Name of the operation being profiled
        **context : dict
            Additional context information
        """
        # Initialize metrics
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            memory_before=MemoryMonitor.get_current_memory_mb(),
            cpu_percent=psutil.cpu_percent(interval=None),
            custom_metrics=context
        )
        
        # Start memory monitoring
        if self.memory_monitor:
            self.memory_monitor.start_monitoring()
        
        self.current_metrics = metrics
        
        try:
            logger.debug(f"Starting profiling: {operation}", **context)
            yield metrics
            
        except Exception as e:
            logger.error(f"Operation failed during profiling: {operation}", error=str(e))
            raise
            
        finally:
            # Finalize metrics
            metrics.end_time = time.time()
            metrics.memory_after = MemoryMonitor.get_current_memory_mb()
            
            # Stop memory monitoring and get stats
            if self.memory_monitor:
                memory_stats = self.memory_monitor.stop_monitoring()
                metrics.memory_peak = memory_stats.get('peak_memory_mb', 0.0)
            
            # Calculate derived metrics
            metrics.finalize()
            
            # Store metrics
            self.metrics_history.append(metrics)
            self.current_metrics = None
            
            # Log performance summary
            logger.info(
                f"Operation completed: {operation}",
                duration=f"{metrics.duration:.3f}s" if metrics.duration else "N/A",
                memory_delta=f"{metrics.memory_delta:.1f}MB" if metrics.memory_delta else "N/A",
                memory_peak=f"{metrics.memory_peak:.1f}MB" if metrics.memory_peak else "N/A"
            )
    
    def get_metrics_summary(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of performance metrics.
        
        Parameters
        ----------
        operation : str, optional
            Filter metrics by operation name
            
        Returns
        -------
        summary : dict
            Performance metrics summary
        """
        if not self.metrics_history:
            return {}
        
        # Filter metrics if operation specified
        metrics = self.metrics_history
        if operation:
            metrics = [m for m in metrics if m.operation == operation]
        
        if not metrics:
            return {}
        
        # Calculate summary statistics
        durations = [m.duration for m in metrics if m.duration is not None]
        memory_deltas = [m.memory_delta for m in metrics if m.memory_delta is not None]
        memory_peaks = [m.memory_peak for m in metrics if m.memory_peak is not None]
        
        summary = {
            'operation_count': len(metrics),
            'total_duration': sum(durations) if durations else 0.0,
            'avg_duration': np.mean(durations) if durations else 0.0,
            'min_duration': np.min(durations) if durations else 0.0,
            'max_duration': np.max(durations) if durations else 0.0,
            'avg_memory_delta': np.mean(memory_deltas) if memory_deltas else 0.0,
            'max_memory_peak': np.max(memory_peaks) if memory_peaks else 0.0,
            'operations': list(set(m.operation for m in metrics))
        }
        
        return summary
    
    def export_metrics(self, filepath: Union[str, Path]) -> None:
        """
        Export metrics to JSON file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to export file
        """
        filepath = Path(filepath)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics_count': len(self.metrics_history),
            'summary': self.get_metrics_summary(),
            'detailed_metrics': [m.to_dict() for m in self.metrics_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Performance metrics exported to {filepath}")


class AutoMemoryManager:
    """Automatic memory management for LSM operations."""
    
    def __init__(self, 
                 memory_threshold: float = 0.85,
                 cleanup_threshold: float = 0.9,
                 enable_auto_gc: bool = True):
        """
        Initialize automatic memory manager.
        
        Parameters
        ----------
        memory_threshold : float, default=0.85
            Memory usage threshold to trigger optimization
        cleanup_threshold : float, default=0.9
            Memory usage threshold to trigger aggressive cleanup
        enable_auto_gc : bool, default=True
            Whether to enable automatic garbage collection
        """
        self.memory_threshold = memory_threshold
        self.cleanup_threshold = cleanup_threshold
        self.enable_auto_gc = enable_auto_gc
        self._cleanup_callbacks = []
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """
        Register a callback function for memory cleanup.
        
        Parameters
        ----------
        callback : callable
            Function to call during memory cleanup
        """
        self._cleanup_callbacks.append(callback)
    
    def check_memory_and_optimize(self, operation: str = "unknown") -> Dict[str, Any]:
        """
        Check memory usage and perform optimization if needed.
        
        Parameters
        ----------
        operation : str, default="unknown"
            Name of the current operation
            
        Returns
        -------
        optimization_info : dict
            Information about optimization actions taken
        """
        try:
            memory_info = psutil.virtual_memory()
            memory_percent = memory_info.percent / 100.0
            available_mb = memory_info.available / (1024 * 1024)
            
            optimization_info = {
                'memory_percent': memory_percent,
                'available_mb': available_mb,
                'actions_taken': []
            }
            
            # Check if optimization is needed
            if memory_percent > self.memory_threshold:
                logger.info(
                    f"Memory optimization triggered for {operation}",
                    memory_percent=f"{memory_percent:.1%}",
                    available_mb=f"{available_mb:.1f}MB"
                )
                
                # Perform garbage collection
                if self.enable_auto_gc:
                    collected = gc.collect()
                    optimization_info['actions_taken'].append(f"garbage_collection_{collected}_objects")
                    logger.debug(f"Garbage collection freed {collected} objects")
                
                # Call cleanup callbacks
                for callback in self._cleanup_callbacks:
                    try:
                        callback()
                        optimization_info['actions_taken'].append("cleanup_callback")
                    except Exception as e:
                        logger.warning(f"Cleanup callback failed: {e}")
                
                # Aggressive cleanup if memory is critically low
                if memory_percent > self.cleanup_threshold:
                    self._aggressive_cleanup()
                    optimization_info['actions_taken'].append("aggressive_cleanup")
                
                # Check memory again after optimization
                new_memory_info = psutil.virtual_memory()
                new_memory_percent = new_memory_info.percent / 100.0
                memory_freed = (memory_percent - new_memory_percent) * 100
                
                optimization_info['memory_freed_percent'] = memory_freed
                
                if memory_freed > 0.01:  # More than 1% freed
                    logger.info(f"Memory optimization freed {memory_freed:.1f}% memory")
                else:
                    logger.warning("Memory optimization had minimal effect")
            
            return optimization_info
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")
            return {'error': str(e)}
    
    def _aggressive_cleanup(self):
        """Perform aggressive memory cleanup."""
        logger.warning("Performing aggressive memory cleanup")
        
        # Multiple garbage collection passes
        for i in range(3):
            collected = gc.collect()
            if collected == 0:
                break
            logger.debug(f"Aggressive GC pass {i+1}: freed {collected} objects")
        
        # Clear any module-level caches if available
        try:
            # Clear numpy cache
            if hasattr(np, '_NoValue'):
                np._NoValue.clear()
        except:
            pass
    
    def estimate_memory_requirement(self, 
                                  data_size: int,
                                  model_config: Dict[str, Any]) -> float:
        """
        Estimate memory requirement for an operation.
        
        Parameters
        ----------
        data_size : int
            Size of the data to process
        model_config : dict
            Model configuration parameters
            
        Returns
        -------
        estimated_mb : float
            Estimated memory requirement in MB
        """
        # Base memory estimation
        window_size = model_config.get('window_size', 10)
        embedding_dim = model_config.get('embedding_dim', 128)
        batch_size = model_config.get('batch_size', 32)
        
        # Estimate memory components
        embedding_memory = data_size * embedding_dim * 4 / (1024 * 1024)  # 4 bytes per float32
        reservoir_memory = window_size * embedding_dim * 4 / (1024 * 1024)
        batch_memory = batch_size * window_size * embedding_dim * 4 / (1024 * 1024)
        
        # Add overhead (50% safety margin)
        total_estimated = (embedding_memory + reservoir_memory + batch_memory) * 1.5
        
        logger.debug(f"Estimated memory requirement: {total_estimated:.1f}MB")
        return total_estimated
    
    def optimize_batch_size(self, 
                           data_size: int,
                           model_config: Dict[str, Any],
                           target_memory_mb: Optional[float] = None) -> int:
        """
        Optimize batch size based on available memory.
        
        Parameters
        ----------
        data_size : int
            Total data size
        model_config : dict
            Model configuration
        target_memory_mb : float, optional
            Target memory usage (defaults to 70% of available)
            
        Returns
        -------
        optimal_batch_size : int
            Optimized batch size
        """
        if target_memory_mb is None:
            available_mb = MemoryMonitor.get_available_memory_mb()
            target_memory_mb = available_mb * 0.7  # Use 70% of available memory
        
        # Start with current batch size
        current_batch_size = model_config.get('batch_size', 32)
        
        # Binary search for optimal batch size
        min_batch = 1
        max_batch = min(data_size, current_batch_size * 4)
        optimal_batch = current_batch_size
        
        while min_batch <= max_batch:
            test_batch = (min_batch + max_batch) // 2
            test_config = {**model_config, 'batch_size': test_batch}
            
            estimated_memory = self.estimate_memory_requirement(data_size, test_config)
            
            if estimated_memory <= target_memory_mb:
                optimal_batch = test_batch
                min_batch = test_batch + 1
            else:
                max_batch = test_batch - 1
        
        # Ensure minimum batch size
        optimal_batch = max(1, optimal_batch)
        
        if optimal_batch != current_batch_size:
            logger.info(
                f"Optimized batch size: {current_batch_size} -> {optimal_batch}",
                estimated_memory_mb=self.estimate_memory_requirement(data_size, {**model_config, 'batch_size': optimal_batch}),
                target_memory_mb=target_memory_mb
            )
        
        return optimal_batch


class BenchmarkSuite:
    """Benchmarking utilities for comparing convenience vs direct API performance."""
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.profiler = PerformanceProfiler()
        self.results = {}
    
    def benchmark_operation(self, 
                          operation_name: str,
                          convenience_func: Callable,
                          direct_func: Callable,
                          test_data: Any,
                          iterations: int = 3) -> Dict[str, Any]:
        """
        Benchmark convenience API vs direct API for an operation.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation being benchmarked
        convenience_func : callable
            Function using convenience API
        direct_func : callable
            Function using direct API
        test_data : any
            Test data for the operation
        iterations : int, default=3
            Number of iterations to run
            
        Returns
        -------
        benchmark_results : dict
            Comparison results
        """
        logger.info(f"Starting benchmark: {operation_name}")
        
        convenience_metrics = []
        direct_metrics = []
        
        # Benchmark convenience API
        for i in range(iterations):
            with self.profiler.profile_operation(f"{operation_name}_convenience_iter_{i}"):
                try:
                    convenience_func(test_data)
                except Exception as e:
                    logger.error(f"Convenience API failed in iteration {i}: {e}")
                    continue
            
            if self.profiler.metrics_history:
                convenience_metrics.append(self.profiler.metrics_history[-1])
        
        # Benchmark direct API
        for i in range(iterations):
            with self.profiler.profile_operation(f"{operation_name}_direct_iter_{i}"):
                try:
                    direct_func(test_data)
                except Exception as e:
                    logger.error(f"Direct API failed in iteration {i}: {e}")
                    continue
            
            if self.profiler.metrics_history:
                direct_metrics.append(self.profiler.metrics_history[-1])
        
        # Calculate comparison results
        results = self._calculate_comparison(
            operation_name, convenience_metrics, direct_metrics
        )
        
        self.results[operation_name] = results
        
        logger.info(f"Benchmark completed: {operation_name}")
        return results
    
    def _calculate_comparison(self, 
                            operation_name: str,
                            convenience_metrics: List[PerformanceMetrics],
                            direct_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate comparison statistics between convenience and direct API."""
        
        def extract_stats(metrics_list):
            if not metrics_list:
                return {}
            
            durations = [m.duration for m in metrics_list if m.duration is not None]
            memory_deltas = [m.memory_delta for m in metrics_list if m.memory_delta is not None]
            memory_peaks = [m.memory_peak for m in metrics_list if m.memory_peak is not None]
            
            return {
                'avg_duration': np.mean(durations) if durations else 0.0,
                'std_duration': np.std(durations) if durations else 0.0,
                'avg_memory_delta': np.mean(memory_deltas) if memory_deltas else 0.0,
                'avg_memory_peak': np.mean(memory_peaks) if memory_peaks else 0.0,
                'iterations': len(metrics_list)
            }
        
        convenience_stats = extract_stats(convenience_metrics)
        direct_stats = extract_stats(direct_metrics)
        
        # Calculate relative performance
        comparison = {
            'operation': operation_name,
            'convenience_api': convenience_stats,
            'direct_api': direct_stats,
            'comparison': {}
        }
        
        if convenience_stats.get('avg_duration', 0) > 0 and direct_stats.get('avg_duration', 0) > 0:
            duration_ratio = convenience_stats['avg_duration'] / direct_stats['avg_duration']
            comparison['comparison']['duration_ratio'] = duration_ratio
            comparison['comparison']['convenience_overhead_percent'] = (duration_ratio - 1) * 100
        
        if convenience_stats.get('avg_memory_delta', 0) != 0 and direct_stats.get('avg_memory_delta', 0) != 0:
            memory_ratio = convenience_stats['avg_memory_delta'] / direct_stats['avg_memory_delta']
            comparison['comparison']['memory_ratio'] = memory_ratio
        
        return comparison
    
    def generate_benchmark_report(self, output_path: Optional[Union[str, Path]] = None) -> str:
        """
        Generate a comprehensive benchmark report.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Path to save the report
            
        Returns
        -------
        report : str
            Formatted benchmark report
        """
        if not self.results:
            return "No benchmark results available."
        
        report_lines = [
            "LSM Convenience API Benchmark Report",
            "=" * 40,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for operation, results in self.results.items():
            report_lines.extend([
                f"Operation: {operation}",
                "-" * 20,
                f"Convenience API:",
                f"  Average Duration: {results['convenience_api'].get('avg_duration', 0):.4f}s",
                f"  Memory Delta: {results['convenience_api'].get('avg_memory_delta', 0):.1f}MB",
                f"  Memory Peak: {results['convenience_api'].get('avg_memory_peak', 0):.1f}MB",
                f"Direct API:",
                f"  Average Duration: {results['direct_api'].get('avg_duration', 0):.4f}s",
                f"  Memory Delta: {results['direct_api'].get('avg_memory_delta', 0):.1f}MB",
                f"  Memory Peak: {results['direct_api'].get('avg_memory_peak', 0):.1f}MB",
                ""
            ])
            
            if 'comparison' in results and results['comparison']:
                comp = results['comparison']
                if 'convenience_overhead_percent' in comp:
                    overhead = comp['convenience_overhead_percent']
                    report_lines.append(f"Convenience API Overhead: {overhead:+.1f}%")
                
                if 'memory_ratio' in comp:
                    memory_ratio = comp['memory_ratio']
                    report_lines.append(f"Memory Usage Ratio: {memory_ratio:.2f}x")
                
                report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            output_path = Path(output_path)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Benchmark report saved to {output_path}")
        
        return report


# Global instances for convenience
_global_profiler = None
_global_memory_manager = None


def get_global_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def get_global_memory_manager() -> AutoMemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = AutoMemoryManager()
    return _global_memory_manager


# Decorator for automatic performance monitoring
def monitor_performance(operation_name: Optional[str] = None):
    """
    Decorator to automatically monitor performance of functions.
    
    Parameters
    ----------
    operation_name : str, optional
        Name of the operation (defaults to function name)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            profiler = get_global_profiler()
            
            with profiler.profile_operation(op_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Decorator for automatic memory management
def manage_memory(memory_threshold: float = 0.85):
    """
    Decorator to automatically manage memory during function execution.
    
    Parameters
    ----------
    memory_threshold : float, default=0.85
        Memory threshold to trigger optimization
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            memory_manager = get_global_memory_manager()
            memory_manager.memory_threshold = memory_threshold
            
            # Check memory before operation
            memory_manager.check_memory_and_optimize(f"before_{func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Check memory after operation
                memory_manager.check_memory_and_optimize(f"after_{func.__name__}")
        
        return wrapper
    return decorator