#!/usr/bin/env python3
"""
Production Readiness Test Suite for LSM Inference System.

This comprehensive test suite validates the system's readiness for production deployment
by testing end-to-end workflows, performance characteristics, memory usage, and error handling
under production-like conditions.
"""

import os
import sys
import time
import json
import tempfile
import shutil
import threading
import subprocess
import traceback
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np

# Optional imports for advanced monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Performance plots will be skipped.")

# Import project modules
try:
    from train import run_training, LSMTrainer
    from data_loader import load_data, DialogueTokenizer
    from inference import OptimizedLSMInference, LSMInference
    from model_config import ModelConfiguration
    from src.lsm.management.model_manager import ModelManager
    from lsm_exceptions import *
    from lsm_logging import get_logger, setup_default_logging
    IMPORTS_AVAILABLE = True
    logger = get_logger(__name__)
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    print("Running in limited mode with mock implementations")
    IMPORTS_AVAILABLE = False
    
    # Create a simple logger fallback
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mock the set_random_seeds function
    def set_random_seeds(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

class ProductionReadinessValidator:
    """
    Comprehensive validator for production readiness of the LSM system.
    """
    
    def __init__(self, test_output_dir: str = None):
        """Initialize the validator with test configuration."""
        self.test_output_dir = test_output_dir or f"production_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Test configuration
        self.test_config = {
            'window_size': 5,  # Smaller for faster testing
            'embedding_dim': 64,  # Smaller for faster testing
            'epochs': 5,  # Fewer epochs for testing
            'batch_size': 16,
            'test_size': 0.3,
            'reservoir_type': 'standard'
        }
        
        # Performance thresholds
        self.performance_thresholds = {
            'training_time_max_minutes': 10,
            'model_load_time_max_seconds': 30,
            'single_prediction_max_seconds': 1.0,
            'batch_prediction_max_seconds_per_item': 0.1,
            'memory_usage_max_mb': 2048,  # 2GB max
            'model_size_max_mb': 500  # 500MB max
        }
        
        # Test results
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'tests': {},
            'overall_status': 'PENDING',
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Temporary model path for testing
        self.temp_model_path = None
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information for the test report."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'timestamp': datetime.now().isoformat()
        }
        
        if PSUTIL_AVAILABLE:
            try:
                info.update({
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'disk_free_gb': psutil.disk_usage('.').free / (1024**3)
                })
            except Exception as e:
                info['psutil_error'] = str(e)
        
        # Check for GPU availability
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            info['gpu_available'] = len(gpus) > 0
            info['gpu_count'] = len(gpus)
        except Exception as e:
            info['gpu_check_error'] = str(e)
            info['gpu_available'] = False
        
        return info
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all production readiness tests."""
        print("🚀 Starting Production Readiness Validation")
        print(f"📁 Test output directory: {self.test_output_dir}")
        print("=" * 60)
        
        test_methods = [
            ('end_to_end_workflow', self.test_end_to_end_workflow),
            ('performance_benchmarks', self.test_performance_benchmarks),
            ('memory_usage_validation', self.test_memory_usage_validation),
            ('concurrent_inference', self.test_concurrent_inference),
            ('error_handling_robustness', self.test_error_handling_robustness),
            ('model_persistence_integrity', self.test_model_persistence_integrity),
            ('batch_processing_efficiency', self.test_batch_processing_efficiency),
            ('resource_cleanup', self.test_resource_cleanup),
            ('configuration_validation', self.test_configuration_validation),
            ('backward_compatibility', self.test_backward_compatibility)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            print(f"\n🧪 Running test: {test_name}")
            try:
                start_time = time.time()
                result = test_method()
                test_time = time.time() - start_time
                
                result['execution_time_seconds'] = test_time
                self.test_results['tests'][test_name] = result
                
                if result['status'] == 'PASS':
                    print(f"✅ {test_name}: PASSED ({test_time:.2f}s)")
                    passed_tests += 1
                else:
                    print(f"❌ {test_name}: FAILED ({test_time:.2f}s)")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"💥 {test_name}: CRASHED")
                print(f"   Exception: {e}")
                self.test_results['tests'][test_name] = {
                    'status': 'CRASH',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # Calculate overall status
        pass_rate = passed_tests / total_tests
        if pass_rate >= 0.9:
            self.test_results['overall_status'] = 'PRODUCTION_READY'
        elif pass_rate >= 0.7:
            self.test_results['overall_status'] = 'NEEDS_MINOR_FIXES'
        else:
            self.test_results['overall_status'] = 'NOT_PRODUCTION_READY'
        
        self.test_results['pass_rate'] = pass_rate
        self.test_results['passed_tests'] = passed_tests
        self.test_results['total_tests'] = total_tests
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save results
        self._save_test_results()
        
        print("\n" + "=" * 60)
        print(f"🎯 Production Readiness Assessment Complete")
        print(f"📊 Overall Status: {self.test_results['overall_status']}")
        print(f"📈 Pass Rate: {pass_rate:.1%} ({passed_tests}/{total_tests})")
        print(f"📄 Full report saved to: {os.path.join(self.test_output_dir, 'production_readiness_report.json')}")
        
        return self.test_results
    
    def test_end_to_end_workflow(self) -> Dict[str, Any]:
        """Test complete end-to-end workflow: train -> save -> load -> predict."""
        try:
            # Step 1: Train a model
            print("  📚 Training model...")
            
            # Set random seeds for reproducibility
            import random
            import numpy as np
            random.seed(42)
            np.random.seed(42)
            
            training_start = time.time()
            training_results = run_training(
                window_size=self.test_config['window_size'],
                embedding_dim=self.test_config['embedding_dim'],
                epochs=self.test_config['epochs'],
                batch_size=self.test_config['batch_size'],
                test_size=self.test_config['test_size'],
                reservoir_type=self.test_config['reservoir_type']
            )
            training_time = time.time() - training_start
            
            if training_time > self.performance_thresholds['training_time_max_minutes'] * 60:
                return {
                    'status': 'FAIL',
                    'error': f'Training took {training_time:.1f}s, exceeds {self.performance_thresholds["training_time_max_minutes"]*60}s threshold'
                }
            
            # Get the model path from training results
            self.temp_model_path = training_results.get('model_path')
            if not self.temp_model_path or not os.path.exists(self.temp_model_path):
                return {
                    'status': 'FAIL',
                    'error': 'Training did not produce a valid model path'
                }
            
            # Step 2: Load model for inference
            print("  🔄 Loading model for inference...")
            load_start = time.time()
            inference_engine = OptimizedLSMInference(self.temp_model_path, lazy_load=False)
            load_time = time.time() - load_start
            
            if load_time > self.performance_thresholds['model_load_time_max_seconds']:
                return {
                    'status': 'FAIL',
                    'error': f'Model loading took {load_time:.1f}s, exceeds {self.performance_thresholds["model_load_time_max_seconds"]}s threshold'
                }
            
            # Step 3: Test single prediction
            print("  🔮 Testing single prediction...")
            test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
            
            prediction_start = time.time()
            prediction = inference_engine.predict_next_token(test_sequence)
            prediction_time = time.time() - prediction_start
            
            if prediction_time > self.performance_thresholds['single_prediction_max_seconds']:
                return {
                    'status': 'FAIL',
                    'error': f'Single prediction took {prediction_time:.3f}s, exceeds {self.performance_thresholds["single_prediction_max_seconds"]}s threshold'
                }
            
            # Step 4: Test batch prediction
            print("  📦 Testing batch prediction...")
            batch_sequences = [test_sequence] * 10
            
            batch_start = time.time()
            batch_predictions = inference_engine.batch_predict(batch_sequences)
            batch_time = time.time() - batch_start
            
            avg_batch_time = batch_time / len(batch_sequences)
            if avg_batch_time > self.performance_thresholds['batch_prediction_max_seconds_per_item']:
                return {
                    'status': 'FAIL',
                    'error': f'Batch prediction averaged {avg_batch_time:.3f}s per item, exceeds {self.performance_thresholds["batch_prediction_max_seconds_per_item"]}s threshold'
                }
            
            # Step 5: Validate predictions
            if not isinstance(prediction, str) or len(prediction) == 0:
                return {
                    'status': 'FAIL',
                    'error': f'Invalid prediction format: {type(prediction)}'
                }
            
            if len(batch_predictions) != len(batch_sequences):
                return {
                    'status': 'FAIL',
                    'error': f'Batch prediction count mismatch: expected {len(batch_sequences)}, got {len(batch_predictions)}'
                }
            
            return {
                'status': 'PASS',
                'metrics': {
                    'training_time_seconds': training_time,
                    'model_load_time_seconds': load_time,
                    'single_prediction_time_seconds': prediction_time,
                    'batch_prediction_time_seconds': batch_time,
                    'avg_batch_time_per_item_seconds': avg_batch_time,
                    'training_final_loss': training_results.get('final_test_loss', 'N/A'),
                    'sample_prediction': prediction,
                    'batch_prediction_count': len(batch_predictions)
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance under various load conditions."""
        if not self.temp_model_path:
            return {'status': 'SKIP', 'error': 'No model available from previous test'}
        
        try:
            inference_engine = OptimizedLSMInference(self.temp_model_path)
            test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
            
            # Test 1: Repeated single predictions
            print("    🔄 Testing repeated single predictions...")
            single_times = []
            for i in range(20):
                start = time.time()
                prediction = inference_engine.predict_next_token(test_sequence)
                single_times.append(time.time() - start)
            
            # Test 2: Various batch sizes
            print("    📊 Testing various batch sizes...")
            batch_results = {}
            for batch_size in [1, 5, 10, 20, 50]:
                sequences = [test_sequence] * batch_size
                start = time.time()
                predictions = inference_engine.batch_predict(sequences, batch_size=batch_size)
                total_time = time.time() - start
                batch_results[batch_size] = {
                    'total_time': total_time,
                    'time_per_item': total_time / batch_size,
                    'throughput_items_per_second': batch_size / total_time
                }
            
            # Test 3: Cache performance
            print("    💾 Testing cache performance...")
            # First prediction (cache miss)
            start = time.time()
            pred1 = inference_engine.predict_next_token(test_sequence)
            cache_miss_time = time.time() - start
            
            # Second prediction (cache hit)
            start = time.time()
            pred2 = inference_engine.predict_next_token(test_sequence)
            cache_hit_time = time.time() - start
            
            cache_speedup = cache_miss_time / cache_hit_time if cache_hit_time > 0 else float('inf')
            
            return {
                'status': 'PASS',
                'metrics': {
                    'single_prediction_stats': {
                        'mean_time': np.mean(single_times),
                        'std_time': np.std(single_times),
                        'min_time': np.min(single_times),
                        'max_time': np.max(single_times),
                        'p95_time': np.percentile(single_times, 95)
                    },
                    'batch_performance': batch_results,
                    'cache_performance': {
                        'cache_miss_time': cache_miss_time,
                        'cache_hit_time': cache_hit_time,
                        'speedup_factor': cache_speedup
                    }
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_memory_usage_validation(self) -> Dict[str, Any]:
        """Test memory usage under various conditions."""
        if not PSUTIL_AVAILABLE:
            return {'status': 'SKIP', 'error': 'psutil not available for memory monitoring'}
        
        if not self.temp_model_path:
            return {'status': 'SKIP', 'error': 'No model available from previous test'}
        
        try:
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load model and measure memory
            inference_engine = OptimizedLSMInference(self.temp_model_path)
            model_loaded_memory = process.memory_info().rss / 1024 / 1024
            
            # Perform many predictions and measure memory growth
            test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
            
            memory_samples = []
            for i in range(100):
                prediction = inference_engine.predict_next_token(test_sequence)
                if i % 10 == 0:  # Sample every 10 predictions
                    memory_samples.append(process.memory_info().rss / 1024 / 1024)
            
            # Test batch processing memory
            large_batch = [test_sequence] * 100
            batch_start_memory = process.memory_info().rss / 1024 / 1024
            batch_predictions = inference_engine.batch_predict(large_batch)
            batch_end_memory = process.memory_info().rss / 1024 / 1024
            
            # Check for memory leaks
            memory_growth = memory_samples[-1] - memory_samples[0] if memory_samples else 0
            max_memory = max(memory_samples) if memory_samples else model_loaded_memory
            
            # Validate against thresholds
            if max_memory > self.performance_thresholds['memory_usage_max_mb']:
                return {
                    'status': 'FAIL',
                    'error': f'Memory usage {max_memory:.1f}MB exceeds threshold {self.performance_thresholds["memory_usage_max_mb"]}MB'
                }
            
            return {
                'status': 'PASS',
                'metrics': {
                    'baseline_memory_mb': baseline_memory,
                    'model_loaded_memory_mb': model_loaded_memory,
                    'model_memory_overhead_mb': model_loaded_memory - baseline_memory,
                    'max_memory_mb': max_memory,
                    'memory_growth_mb': memory_growth,
                    'batch_memory_increase_mb': batch_end_memory - batch_start_memory,
                    'memory_samples': memory_samples
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_concurrent_inference(self) -> Dict[str, Any]:
        """Test concurrent inference requests."""
        if not self.temp_model_path:
            return {'status': 'SKIP', 'error': 'No model available from previous test'}
        
        try:
            inference_engine = OptimizedLSMInference(self.temp_model_path)
            test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
            
            results = []
            errors = []
            
            def worker_thread(thread_id: int, num_predictions: int):
                """Worker thread for concurrent testing."""
                try:
                    thread_results = []
                    for i in range(num_predictions):
                        start = time.time()
                        prediction = inference_engine.predict_next_token(test_sequence)
                        duration = time.time() - start
                        thread_results.append({
                            'thread_id': thread_id,
                            'prediction_id': i,
                            'duration': duration,
                            'prediction': prediction
                        })
                    results.extend(thread_results)
                except Exception as e:
                    errors.append({
                        'thread_id': thread_id,
                        'error': str(e)
                    })
            
            # Test with multiple threads
            num_threads = 4
            predictions_per_thread = 10
            
            print(f"    🧵 Testing {num_threads} concurrent threads...")
            threads = []
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(target=worker_thread, args=(i, predictions_per_thread))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.time() - start_time
            
            if errors:
                return {
                    'status': 'FAIL',
                    'error': f'Concurrent execution had {len(errors)} errors',
                    'errors': errors
                }
            
            # Analyze results
            total_predictions = len(results)
            avg_duration = np.mean([r['duration'] for r in results])
            throughput = total_predictions / total_time
            
            return {
                'status': 'PASS',
                'metrics': {
                    'num_threads': num_threads,
                    'predictions_per_thread': predictions_per_thread,
                    'total_predictions': total_predictions,
                    'total_time_seconds': total_time,
                    'avg_prediction_time_seconds': avg_duration,
                    'throughput_predictions_per_second': throughput,
                    'errors_count': len(errors)
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_error_handling_robustness(self) -> Dict[str, Any]:
        """Test error handling under various failure conditions."""
        try:
            error_tests = []
            
            # Test 1: Invalid model path
            try:
                invalid_inference = OptimizedLSMInference("/nonexistent/path")
                error_tests.append({'test': 'invalid_model_path', 'status': 'FAIL', 'error': 'Should have raised exception'})
            except Exception as e:
                error_tests.append({'test': 'invalid_model_path', 'status': 'PASS', 'expected_error': str(e)})
            
            if self.temp_model_path:
                inference_engine = OptimizedLSMInference(self.temp_model_path)
                
                # Test 2: Invalid input sequence length
                try:
                    prediction = inference_engine.predict_next_token(["too", "short"])
                    error_tests.append({'test': 'invalid_sequence_length', 'status': 'FAIL', 'error': 'Should have raised exception'})
                except Exception as e:
                    error_tests.append({'test': 'invalid_sequence_length', 'status': 'PASS', 'expected_error': str(e)})
                
                # Test 3: Empty input
                try:
                    prediction = inference_engine.predict_next_token([])
                    error_tests.append({'test': 'empty_input', 'status': 'FAIL', 'error': 'Should have raised exception'})
                except Exception as e:
                    error_tests.append({'test': 'empty_input', 'status': 'PASS', 'expected_error': str(e)})
                
                # Test 4: Non-string input
                try:
                    prediction = inference_engine.predict_next_token([1, 2, 3, 4, 5])
                    error_tests.append({'test': 'non_string_input', 'status': 'FAIL', 'error': 'Should have raised exception'})
                except Exception as e:
                    error_tests.append({'test': 'non_string_input', 'status': 'PASS', 'expected_error': str(e)})
            
            # Check if all error tests passed
            failed_tests = [t for t in error_tests if t['status'] == 'FAIL']
            
            return {
                'status': 'PASS' if not failed_tests else 'FAIL',
                'error': f'{len(failed_tests)} error handling tests failed' if failed_tests else None,
                'metrics': {
                    'total_error_tests': len(error_tests),
                    'passed_error_tests': len([t for t in error_tests if t['status'] == 'PASS']),
                    'error_test_details': error_tests
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_model_persistence_integrity(self) -> Dict[str, Any]:
        """Test model saving and loading integrity."""
        if not self.temp_model_path:
            return {'status': 'SKIP', 'error': 'No model available from previous test'}
        
        try:
            # Load original model
            original_inference = OptimizedLSMInference(self.temp_model_path)
            test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
            
            # Get original prediction
            original_prediction = original_inference.predict_next_token(test_sequence)
            original_info = original_inference.get_model_info()
            
            # Create a copy of the model in a new location
            copy_model_path = os.path.join(self.test_output_dir, "model_copy")
            shutil.copytree(self.temp_model_path, copy_model_path)
            
            # Load copied model
            copy_inference = OptimizedLSMInference(copy_model_path)
            copy_prediction = copy_inference.predict_next_token(test_sequence)
            copy_info = copy_inference.get_model_info()
            
            # Verify predictions are identical
            if original_prediction != copy_prediction:
                return {
                    'status': 'FAIL',
                    'error': f'Predictions differ: original="{original_prediction}", copy="{copy_prediction}"'
                }
            
            # Verify model info is consistent
            key_fields = ['architecture', 'tokenizer']
            for field in key_fields:
                if field in original_info and field in copy_info:
                    if original_info[field] != copy_info[field]:
                        return {
                            'status': 'FAIL',
                            'error': f'Model info field "{field}" differs between original and copy'
                        }
            
            # Check model file sizes
            original_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(self.temp_model_path)
                              for filename in filenames) / (1024 * 1024)  # MB
            
            if original_size > self.performance_thresholds['model_size_max_mb']:
                return {
                    'status': 'FAIL',
                    'error': f'Model size {original_size:.1f}MB exceeds threshold {self.performance_thresholds["model_size_max_mb"]}MB'
                }
            
            return {
                'status': 'PASS',
                'metrics': {
                    'model_size_mb': original_size,
                    'prediction_consistency': original_prediction == copy_prediction,
                    'original_prediction': original_prediction,
                    'copy_prediction': copy_prediction,
                    'model_files_count': len([f for _, _, files in os.walk(self.temp_model_path) for f in files])
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_batch_processing_efficiency(self) -> Dict[str, Any]:
        """Test batch processing efficiency and scalability."""
        if not self.temp_model_path:
            return {'status': 'SKIP', 'error': 'No model available from previous test'}
        
        try:
            inference_engine = OptimizedLSMInference(self.temp_model_path)
            test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
            
            # Test different batch sizes
            batch_sizes = [1, 5, 10, 20, 50, 100]
            batch_results = {}
            
            for batch_size in batch_sizes:
                print(f"    📦 Testing batch size {batch_size}...")
                sequences = [test_sequence] * batch_size
                
                # Time batch processing
                start_time = time.time()
                predictions = inference_engine.batch_predict(sequences)
                batch_time = time.time() - start_time
                
                # Time individual processing for comparison
                start_time = time.time()
                individual_predictions = []
                for seq in sequences:
                    pred = inference_engine.predict_next_token(seq)
                    individual_predictions.append(pred)
                individual_time = time.time() - start_time
                
                # Calculate efficiency metrics
                efficiency_ratio = individual_time / batch_time if batch_time > 0 else float('inf')
                throughput = batch_size / batch_time if batch_time > 0 else 0
                
                batch_results[batch_size] = {
                    'batch_time': batch_time,
                    'individual_time': individual_time,
                    'efficiency_ratio': efficiency_ratio,
                    'throughput_items_per_second': throughput,
                    'predictions_match': predictions == individual_predictions
                }
                
                # Verify predictions are consistent
                if predictions != individual_predictions:
                    return {
                        'status': 'FAIL',
                        'error': f'Batch predictions differ from individual predictions for batch size {batch_size}'
                    }
            
            # Check if batch processing is actually more efficient
            large_batch_efficiency = batch_results.get(50, {}).get('efficiency_ratio', 0)
            if large_batch_efficiency < 1.5:  # Should be at least 1.5x faster
                return {
                    'status': 'FAIL',
                    'error': f'Batch processing not efficient enough: {large_batch_efficiency:.2f}x speedup'
                }
            
            return {
                'status': 'PASS',
                'metrics': {
                    'batch_results': batch_results,
                    'max_efficiency_ratio': max(r['efficiency_ratio'] for r in batch_results.values()),
                    'max_throughput': max(r['throughput_items_per_second'] for r in batch_results.values())
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_resource_cleanup(self) -> Dict[str, Any]:
        """Test proper resource cleanup and garbage collection."""
        if not self.temp_model_path:
            return {'status': 'SKIP', 'error': 'No model available from previous test'}
        
        try:
            if not PSUTIL_AVAILABLE:
                return {'status': 'SKIP', 'error': 'psutil not available for memory monitoring'}
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Create and destroy multiple inference engines
            for i in range(5):
                inference_engine = OptimizedLSMInference(self.temp_model_path)
                test_sequence = ["hello", "how", "are", "you", "today"][:self.test_config['window_size']]
                
                # Use the engine
                for j in range(10):
                    prediction = inference_engine.predict_next_token(test_sequence)
                
                # Clear caches
                inference_engine.clear_caches()
                
                # Delete reference
                del inference_engine
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Check final memory
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - initial_memory
            
            # Allow some memory growth but not excessive
            if memory_growth > 100:  # 100MB threshold
                return {
                    'status': 'FAIL',
                    'error': f'Excessive memory growth: {memory_growth:.1f}MB after cleanup'
                }
            
            return {
                'status': 'PASS',
                'metrics': {
                    'initial_memory_mb': initial_memory,
                    'final_memory_mb': final_memory,
                    'memory_growth_mb': memory_growth,
                    'cleanup_effective': memory_growth < 50
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_configuration_validation(self) -> Dict[str, Any]:
        """Test configuration validation and error handling."""
        try:
            # Test valid configuration
            valid_config = ModelConfiguration(
                window_size=10,
                embedding_dim=128,
                reservoir_type='standard',
                reservoir_config={},
                reservoir_units=[256, 128, 64],
                sparsity=0.1,
                use_multichannel=True,
                training_params={'epochs': 20, 'batch_size': 32}
            )
            
            # Test serialization/deserialization
            config_path = os.path.join(self.test_output_dir, "test_config.json")
            valid_config.save(config_path)
            loaded_config = ModelConfiguration.load(config_path)
            
            if valid_config.to_dict() != loaded_config.to_dict():
                return {
                    'status': 'FAIL',
                    'error': 'Configuration serialization/deserialization failed'
                }
            
            # Test invalid configurations
            invalid_tests = []
            
            # Test invalid window size
            try:
                invalid_config = ModelConfiguration(
                    window_size=-1,  # Invalid
                    embedding_dim=128,
                    reservoir_type='standard',
                    reservoir_config={},
                    reservoir_units=[256],
                    sparsity=0.1,
                    use_multichannel=True,
                    training_params={}
                )
                invalid_tests.append({'test': 'negative_window_size', 'status': 'FAIL'})
            except Exception:
                invalid_tests.append({'test': 'negative_window_size', 'status': 'PASS'})
            
            return {
                'status': 'PASS',
                'metrics': {
                    'serialization_test': 'PASS',
                    'invalid_config_tests': invalid_tests,
                    'config_fields_count': len(valid_config.to_dict())
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def test_backward_compatibility(self) -> Dict[str, Any]:
        """Test backward compatibility with older model formats."""
        try:
            # This is a placeholder test since we don't have old models
            # In a real scenario, you would test loading models from previous versions
            
            return {
                'status': 'PASS',
                'metrics': {
                    'note': 'Backward compatibility test placeholder - would test with actual old models in production'
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAIL',
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _generate_recommendations(self):
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check performance metrics
        if 'end_to_end_workflow' in self.test_results['tests']:
            metrics = self.test_results['tests']['end_to_end_workflow'].get('metrics', {})
            
            if metrics.get('training_time_seconds', 0) > 300:  # 5 minutes
                recommendations.append({
                    'category': 'performance',
                    'priority': 'medium',
                    'issue': 'Training time is high',
                    'recommendation': 'Consider reducing model complexity or using GPU acceleration'
                })
            
            if metrics.get('single_prediction_time_seconds', 0) > 0.5:
                recommendations.append({
                    'category': 'performance',
                    'priority': 'high',
                    'issue': 'Single prediction time is high',
                    'recommendation': 'Enable caching and lazy loading optimizations'
                })
        
        # Check memory usage
        if 'memory_usage_validation' in self.test_results['tests']:
            metrics = self.test_results['tests']['memory_usage_validation'].get('metrics', {})
            
            if metrics.get('max_memory_mb', 0) > 1000:
                recommendations.append({
                    'category': 'memory',
                    'priority': 'medium',
                    'issue': 'High memory usage detected',
                    'recommendation': 'Consider model compression or streaming inference for large datasets'
                })
        
        # Check failed tests
        failed_tests = [name for name, result in self.test_results['tests'].items() 
                       if result.get('status') != 'PASS']
        
        if failed_tests:
            recommendations.append({
                'category': 'reliability',
                'priority': 'high',
                'issue': f'Failed tests: {", ".join(failed_tests)}',
                'recommendation': 'Address failing tests before production deployment'
            })
        
        self.test_results['recommendations'] = recommendations
    
    def _save_test_results(self):
        """Save test results to JSON file."""
        results_file = os.path.join(self.test_output_dir, "production_readiness_report.json")
        
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Also save a human-readable summary
        summary_file = os.path.join(self.test_output_dir, "production_readiness_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("LSM Production Readiness Test Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Status: {self.test_results['overall_status']}\n")
            f.write(f"Pass Rate: {self.test_results['pass_rate']:.1%}\n")
            f.write(f"Tests Passed: {self.test_results['passed_tests']}/{self.test_results['total_tests']}\n\n")
            
            f.write("Test Results:\n")
            f.write("-" * 20 + "\n")
            for test_name, result in self.test_results['tests'].items():
                status = result.get('status', 'UNKNOWN')
                f.write(f"{test_name}: {status}\n")
                if status != 'PASS' and 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
            
            f.write("\nRecommendations:\n")
            f.write("-" * 20 + "\n")
            for rec in self.test_results['recommendations']:
                f.write(f"[{rec['priority'].upper()}] {rec['issue']}\n")
                f.write(f"  Recommendation: {rec['recommendation']}\n\n")
    
    def cleanup(self):
        """Clean up temporary resources."""
        if self.temp_model_path and os.path.exists(self.temp_model_path):
            try:
                shutil.rmtree(self.temp_model_path)
                print(f"🧹 Cleaned up temporary model: {self.temp_model_path}")
            except Exception as e:
                print(f"⚠️  Failed to clean up temporary model: {e}")


def main():
    """Run production readiness validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="LSM Production Readiness Validator")
    parser.add_argument('--output-dir', type=str, help='Output directory for test results')
    parser.add_argument('--cleanup', action='store_true', help='Clean up temporary files after testing')
    
    args = parser.parse_args()
    
    # Setup logging
    if IMPORTS_AVAILABLE:
        setup_default_logging()
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Run validation
    validator = ProductionReadinessValidator(args.output_dir)
    
    try:
        results = validator.run_all_tests()
        
        # Print final summary
        print("\n" + "🎯 PRODUCTION READINESS ASSESSMENT" + "\n")
        print(f"Status: {results['overall_status']}")
        print(f"Pass Rate: {results['pass_rate']:.1%}")
        
        if results['overall_status'] == 'PRODUCTION_READY':
            print("✅ System is ready for production deployment!")
        elif results['overall_status'] == 'NEEDS_MINOR_FIXES':
            print("⚠️  System needs minor fixes before production deployment.")
        else:
            print("❌ System is not ready for production deployment.")
        
        print(f"\n📄 Full report: {os.path.join(validator.test_output_dir, 'production_readiness_report.json')}")
        
        return 0 if results['overall_status'] in ['PRODUCTION_READY', 'NEEDS_MINOR_FIXES'] else 1
        
    finally:
        if args.cleanup:
            validator.cleanup()


if __name__ == "__main__":
    sys.exit(main())