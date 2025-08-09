#!/usr/bin/env python3
"""
Tests for LSM convenience API performance monitoring and optimization.

This module tests the performance monitoring, memory management, and
benchmarking capabilities of the convenience API.
"""

import unittest
import tempfile
import time
from pathlib import Path
import sys

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from lsm.convenience.performance import (
        PerformanceProfiler, MemoryMonitor, AutoMemoryManager,
        BenchmarkSuite, PerformanceMetrics
    )
    from lsm.convenience.benchmarks import ConvenienceAPIBenchmark
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False


@unittest.skipUnless(PERFORMANCE_AVAILABLE, "Performance monitoring not available")
class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring capabilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.profiler = PerformanceProfiler()
        self.memory_monitor = MemoryMonitor()
        self.memory_manager = AutoMemoryManager()
    
    def test_performance_metrics_creation(self):
        """Test creation and manipulation of performance metrics."""
        metrics = PerformanceMetrics(
            operation="test_operation",
            start_time=time.time(),
            data_size=100
        )
        
        self.assertEqual(metrics.operation, "test_operation")
        self.assertIsNotNone(metrics.start_time)
        self.assertEqual(metrics.data_size, 100)
        
        # Test finalization
        metrics.end_time = metrics.start_time + 1.0
        metrics.memory_before = 100.0
        metrics.memory_after = 120.0
        metrics.finalize()
        
        self.assertAlmostEqual(metrics.duration, 1.0, places=1)
        self.assertAlmostEqual(metrics.memory_delta, 20.0, places=1)
        
        # Test conversion to dict
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertIn('operation', metrics_dict)
        self.assertIn('duration', metrics_dict)
    
    def test_memory_monitor(self):
        """Test memory monitoring functionality."""
        # Test static methods
        current_memory = MemoryMonitor.get_current_memory_mb()
        self.assertIsInstance(current_memory, float)
        self.assertGreater(current_memory, 0)
        
        available_memory = MemoryMonitor.get_available_memory_mb()
        self.assertIsInstance(available_memory, float)
        self.assertGreater(available_memory, 0)
        
        # Test monitoring start/stop
        self.memory_monitor.start_monitoring()
        self.assertTrue(self.memory_monitor.monitoring)
        
        # Let it monitor for a short time
        time.sleep(0.1)
        
        stats = self.memory_monitor.stop_monitoring()
        self.assertFalse(self.memory_monitor.monitoring)
        self.assertIsInstance(stats, dict)
        self.assertIn('peak_memory_mb', stats)
    
    def test_performance_profiler(self):
        """Test performance profiling functionality."""
        # Test profiling context manager
        with self.profiler.profile_operation("test_operation", test_param="value"):
            time.sleep(0.1)  # Simulate some work
        
        # Check that metrics were recorded
        self.assertEqual(len(self.profiler.metrics_history), 1)
        
        metrics = self.profiler.metrics_history[0]
        self.assertEqual(metrics.operation, "test_operation")
        self.assertIsNotNone(metrics.duration)
        self.assertGreater(metrics.duration, 0.05)  # Should be at least 0.05s
        
        # Test metrics summary
        summary = self.profiler.get_metrics_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary.get('operation_count'), 1)
        self.assertGreater(summary.get('total_duration', 0), 0)
    
    def test_auto_memory_manager(self):
        """Test automatic memory management."""
        # Test memory estimation
        model_config = {
            'window_size': 10,
            'embedding_dim': 128,
            'batch_size': 32
        }
        
        estimated_memory = self.memory_manager.estimate_memory_requirement(
            data_size=1000,
            model_config=model_config
        )
        
        self.assertIsInstance(estimated_memory, float)
        self.assertGreater(estimated_memory, 0)
        
        # Test batch size optimization
        optimal_batch_size = self.memory_manager.optimize_batch_size(
            data_size=1000,
            model_config=model_config
        )
        
        self.assertIsInstance(optimal_batch_size, int)
        self.assertGreater(optimal_batch_size, 0)
        self.assertLessEqual(optimal_batch_size, 1000)
        
        # Test memory check and optimization
        optimization_info = self.memory_manager.check_memory_and_optimize("test_operation")
        self.assertIsInstance(optimization_info, dict)
        self.assertIn('memory_percent', optimization_info)
        self.assertIn('available_mb', optimization_info)
    
    def test_benchmark_suite(self):
        """Test benchmarking functionality."""
        benchmark_suite = BenchmarkSuite()
        
        # Define simple test functions
        def convenience_func(data):
            time.sleep(0.01)  # Simulate convenience API overhead
            return f"convenience_result_{len(data)}"
        
        def direct_func(data):
            time.sleep(0.005)  # Simulate faster direct API
            return f"direct_result_{len(data)}"
        
        test_data = list(range(10))
        
        # Run benchmark
        result = benchmark_suite.benchmark_operation(
            operation_name="test_operation",
            convenience_func=convenience_func,
            direct_func=direct_func,
            test_data=test_data,
            iterations=2
        )
        
        # Verify results
        self.assertIsInstance(result, dict)
        self.assertIn('operation', result)
        self.assertIn('convenience_api', result)
        self.assertIn('direct_api', result)
        self.assertEqual(result['operation'], "test_operation")
        
        # Check that both APIs were benchmarked
        self.assertGreater(result['convenience_api'].get('iterations', 0), 0)
        self.assertGreater(result['direct_api'].get('iterations', 0), 0)
    
    def test_export_functionality(self):
        """Test exporting performance data."""
        # Profile an operation
        with self.profiler.profile_operation("export_test"):
            time.sleep(0.05)
        
        # Test export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            self.profiler.export_metrics(temp_path)
            
            # Verify file was created and contains data
            self.assertTrue(temp_path.exists())
            
            import json
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            self.assertIsInstance(exported_data, dict)
            self.assertIn('export_timestamp', exported_data)
            self.assertIn('metrics_count', exported_data)
            self.assertIn('detailed_metrics', exported_data)
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


@unittest.skipUnless(PERFORMANCE_AVAILABLE, "Performance monitoring not available")
class TestConvenienceAPIBenchmark(unittest.TestCase):
    """Test convenience API benchmarking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.benchmark = ConvenienceAPIBenchmark()
    
    def test_test_data_preparation(self):
        """Test preparation of test datasets."""
        self.benchmark.prepare_test_data(
            small_dataset_size=10,
            medium_dataset_size=20,
            large_dataset_size=30
        )
        
        # Verify test data was created
        self.assertIn('small', self.benchmark.test_data)
        self.assertIn('medium', self.benchmark.test_data)
        self.assertIn('large', self.benchmark.test_data)
        
        # Check data structure
        small_data = self.benchmark.test_data['small']
        self.assertIn('conversations', small_data)
        self.assertIn('size', small_data)
        self.assertEqual(len(small_data['conversations']), 10)
        self.assertEqual(small_data['size'], 10)
    
    def test_benchmark_report_generation(self):
        """Test benchmark report generation."""
        # Add some mock results
        self.benchmark.results = {
            'training': {
                'small': {
                    'operation': 'training_small',
                    'comparison': {
                        'convenience_overhead_percent': 25.0
                    }
                }
            }
        }
        
        report = self.benchmark.generate_performance_report()
        
        self.assertIsInstance(report, str)
        self.assertIn("Performance Report", report)
        self.assertIn("small:", report)  # The report shows "Dataset small:" not "training_small"
        self.assertIn("25.0%", report)


class TestPerformanceDecorators(unittest.TestCase):
    """Test performance monitoring decorators."""
    
    @unittest.skipUnless(PERFORMANCE_AVAILABLE, "Performance monitoring not available")
    def test_monitor_performance_decorator(self):
        """Test the monitor_performance decorator."""
        from lsm.convenience.performance import monitor_performance, get_global_profiler
        
        @monitor_performance("test_decorated_function")
        def test_function(duration=0.01):
            time.sleep(duration)
            return "test_result"
        
        # Clear any existing metrics
        profiler = get_global_profiler()
        profiler.metrics_history.clear()
        
        # Call decorated function
        result = test_function(0.05)
        
        # Verify result
        self.assertEqual(result, "test_result")
        
        # Verify performance was monitored
        self.assertGreater(len(profiler.metrics_history), 0)
        
        # Find our operation in the metrics
        our_metrics = [m for m in profiler.metrics_history if m.operation == "test_decorated_function"]
        self.assertGreater(len(our_metrics), 0)
        
        metrics = our_metrics[0]
        self.assertIsNotNone(metrics.duration)
        self.assertGreater(metrics.duration, 0.04)  # Should be at least 0.04s
    
    @unittest.skipUnless(PERFORMANCE_AVAILABLE, "Performance monitoring not available")
    def test_manage_memory_decorator(self):
        """Test the manage_memory decorator."""
        from lsm.convenience.performance import manage_memory
        
        @manage_memory(memory_threshold=0.9)
        def memory_intensive_function():
            # Simulate memory-intensive operation
            data = list(range(1000))
            return len(data)
        
        # This should run without errors
        result = memory_intensive_function()
        self.assertEqual(result, 1000)


if __name__ == '__main__':
    unittest.main()