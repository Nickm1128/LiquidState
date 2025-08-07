#!/usr/bin/env python3
"""
Enhanced Production Readiness Validation for LSM System.

This script performs comprehensive validation of the LSM system for production deployment,
including end-to-end testing, performance benchmarking, and deployment readiness assessment.
"""

import os
import sys
import time
import json
import tempfile
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Any, Tuple
import numpy as np

# Optional imports for monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

class ProductionValidator:
    """Enhanced production readiness validator."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the validator."""
        self.output_dir = output_dir or f"production_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'validation_results': {},
            'overall_status': 'PENDING',
            'recommendations': []
        }
        
        print(f"🚀 Production Validation Started")
        print(f"📁 Output directory: {self.output_dir}")
        print("=" * 60)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
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
        
        return info
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate the Python environment and dependencies."""
        print("🔍 Validating Environment...")
        
        validation = {
            'status': 'PASS',
            'checks': {},
            'errors': []
        }
        
        # Check Python version
        python_version = sys.version_info
        if python_version >= (3, 11):
            validation['checks']['python_version'] = 'PASS'
        else:
            validation['checks']['python_version'] = 'FAIL'
            validation['errors'].append(f"Python {python_version.major}.{python_version.minor} < 3.11")
        
        # Check required modules
        required_modules = [
            'numpy', 'pandas', 'scikit-learn', 'tensorflow'
        ]
        
        for module in required_modules:
            try:
                __import__(module)
                validation['checks'][f'module_{module}'] = 'PASS'
            except ImportError as e:
                validation['checks'][f'module_{module}'] = 'FAIL'
                validation['errors'].append(f"Missing module: {module}")
        
        # Check project modules
        project_modules = [
            'train', 'inference', 'data_loader', 'model_config', 'model_manager'
        ]
        
        for module in project_modules:
            try:
                __import__(module)
                validation['checks'][f'project_{module}'] = 'PASS'
            except ImportError as e:
                validation['checks'][f'project_{module}'] = 'FAIL'
                validation['errors'].append(f"Missing project module: {module}")
        
        # Overall status
        if validation['errors']:
            validation['status'] = 'FAIL'
        
        return validation
    
    def validate_training_pipeline(self) -> Dict[str, Any]:
        """Validate the training pipeline by training a small model."""
        print("🎯 Validating Training Pipeline...")
        
        validation = {
            'status': 'PENDING',
            'metrics': {},
            'errors': []
        }
        
        try:
            # Run training with minimal parameters
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, 'main.py', 'train',
                '--window-size', '5',
                '--embedding-dim', '32',
                '--epochs', '3',
                '--batch-size', '16',
                '--reservoir-type', 'standard'
            ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                validation['status'] = 'PASS'
                validation['metrics']['training_time_seconds'] = training_time
                
                # Extract model path from output
                output_lines = result.stdout.split('\n')
                model_path = None
                for line in output_lines:
                    if 'Model saved to:' in line:
                        model_path = line.split('Model saved to:')[-1].strip()
                        break
                
                validation['metrics']['model_path'] = model_path
                validation['metrics']['stdout'] = result.stdout[-500:]  # Last 500 chars
                
            else:
                validation['status'] = 'FAIL'
                validation['errors'].append(f"Training failed with return code {result.returncode}")
                validation['errors'].append(f"stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            validation['status'] = 'FAIL'
            validation['errors'].append("Training timed out after 5 minutes")
        except Exception as e:
            validation['status'] = 'FAIL'
            validation['errors'].append(f"Training error: {str(e)}")
        
        return validation
    
    def validate_inference_pipeline(self, model_path: str = None) -> Dict[str, Any]:
        """Validate the inference pipeline."""
        print("🔮 Validating Inference Pipeline...")
        
        validation = {
            'status': 'PENDING',
            'metrics': {},
            'errors': []
        }
        
        if not model_path:
            # Try to find a model
            import glob
            model_dirs = glob.glob("models_*")
            if model_dirs:
                model_path = sorted(model_dirs)[-1]  # Use the latest
            else:
                validation['status'] = 'SKIP'
                validation['errors'].append("No trained model available")
                return validation
        
        try:
            # Test basic inference
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, 'inference.py',
                '--model-path', model_path,
                '--input-text', 'hello', 'how', 'are', 'you', 'today'
            ], capture_output=True, text=True, timeout=60)
            
            inference_time = time.time() - start_time
            
            if result.returncode == 0:
                validation['status'] = 'PASS'
                validation['metrics']['inference_time_seconds'] = inference_time
                validation['metrics']['model_path'] = model_path
                validation['metrics']['sample_output'] = result.stdout.strip()[-200:]  # Last 200 chars
            else:
                validation['status'] = 'FAIL'
                validation['errors'].append(f"Inference failed with return code {result.returncode}")
                validation['errors'].append(f"stderr: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            validation['status'] = 'FAIL'
            validation['errors'].append("Inference timed out after 60 seconds")
        except Exception as e:
            validation['status'] = 'FAIL'
            validation['errors'].append(f"Inference error: {str(e)}")
        
        return validation
    
    def validate_model_management(self) -> Dict[str, Any]:
        """Validate model management functionality."""
        print("📋 Validating Model Management...")
        
        validation = {
            'status': 'PENDING',
            'metrics': {},
            'errors': []
        }
        
        try:
            # Test model listing
            result = subprocess.run([
                sys.executable, 'manage_models.py', 'list'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                validation['status'] = 'PASS'
                validation['metrics']['list_output'] = result.stdout.strip()
                
                # Count models found
                output_lines = result.stdout.split('\n')
                model_count = 0
                for line in output_lines:
                    if 'models_' in line and '✅' in line:
                        model_count += 1
                
                validation['metrics']['models_found'] = model_count
            else:
                validation['status'] = 'FAIL'
                validation['errors'].append(f"Model listing failed: {result.stderr}")
                
        except Exception as e:
            validation['status'] = 'FAIL'
            validation['errors'].append(f"Model management error: {str(e)}")
        
        return validation
    
    def validate_performance(self, model_path: str = None) -> Dict[str, Any]:
        """Validate performance characteristics."""
        print("⚡ Validating Performance...")
        
        validation = {
            'status': 'PENDING',
            'metrics': {},
            'errors': []
        }
        
        if not model_path:
            # Try to find a model
            import glob
            model_dirs = glob.glob("models_*")
            if model_dirs:
                model_path = sorted(model_dirs)[-1]
            else:
                validation['status'] = 'SKIP'
                validation['errors'].append("No trained model available")
                return validation
        
        try:
            # Test performance demo
            start_time = time.time()
            
            result = subprocess.run([
                sys.executable, 'performance_demo.py',
                '--model-path', model_path,
                '--num-predictions', '10'
            ], capture_output=True, text=True, timeout=120)
            
            demo_time = time.time() - start_time
            
            if result.returncode == 0:
                validation['status'] = 'PASS'
                validation['metrics']['demo_time_seconds'] = demo_time
                validation['metrics']['performance_output'] = result.stdout.strip()[-300:]
                
                # Extract performance metrics from output
                output = result.stdout
                if 'Average prediction time:' in output:
                    for line in output.split('\n'):
                        if 'Average prediction time:' in line:
                            try:
                                time_str = line.split(':')[-1].strip().replace('s', '')
                                validation['metrics']['avg_prediction_time'] = float(time_str)
                            except:
                                pass
                
            else:
                validation['status'] = 'FAIL'
                validation['errors'].append(f"Performance demo failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            validation['status'] = 'FAIL'
            validation['errors'].append("Performance demo timed out")
        except Exception as e:
            validation['status'] = 'FAIL'
            validation['errors'].append(f"Performance validation error: {str(e)}")
        
        return validation
    
    def validate_system_resources(self) -> Dict[str, Any]:
        """Validate system resource usage."""
        print("💾 Validating System Resources...")
        
        validation = {
            'status': 'PASS',
            'metrics': {},
            'warnings': []
        }
        
        if PSUTIL_AVAILABLE:
            try:
                # Memory check
                memory = psutil.virtual_memory()
                validation['metrics']['memory_total_gb'] = memory.total / (1024**3)
                validation['metrics']['memory_available_gb'] = memory.available / (1024**3)
                validation['metrics']['memory_percent_used'] = memory.percent
                
                if memory.available < 2 * (1024**3):  # Less than 2GB available
                    validation['warnings'].append("Low available memory (< 2GB)")
                
                # Disk check
                disk = psutil.disk_usage('.')
                validation['metrics']['disk_total_gb'] = disk.total / (1024**3)
                validation['metrics']['disk_free_gb'] = disk.free / (1024**3)
                validation['metrics']['disk_percent_used'] = (disk.used / disk.total) * 100
                
                if disk.free < 5 * (1024**3):  # Less than 5GB free
                    validation['warnings'].append("Low disk space (< 5GB)")
                
                # CPU check
                cpu_percent = psutil.cpu_percent(interval=1)
                validation['metrics']['cpu_percent'] = cpu_percent
                validation['metrics']['cpu_count'] = psutil.cpu_count()
                
                if cpu_percent > 90:
                    validation['warnings'].append("High CPU usage")
                
            except Exception as e:
                validation['warnings'].append(f"Resource monitoring error: {str(e)}")
        else:
            validation['warnings'].append("psutil not available - limited resource monitoring")
        
        return validation
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests."""
        
        # Run validation steps
        validation_steps = [
            ('environment', self.validate_environment),
            ('training_pipeline', self.validate_training_pipeline),
            ('inference_pipeline', lambda: self.validate_inference_pipeline()),
            ('model_management', self.validate_model_management),
            ('performance', lambda: self.validate_performance()),
            ('system_resources', self.validate_system_resources)
        ]
        
        model_path = None
        passed_validations = 0
        total_validations = len(validation_steps)
        
        for step_name, step_func in validation_steps:
            print(f"\n🧪 Running validation: {step_name}")
            try:
                start_time = time.time()
                result = step_func()
                execution_time = time.time() - start_time
                
                result['execution_time_seconds'] = execution_time
                self.results['validation_results'][step_name] = result
                
                # Extract model path for subsequent tests
                if step_name == 'training_pipeline' and result.get('metrics', {}).get('model_path'):
                    model_path = result['metrics']['model_path']
                
                # Update validation functions with model path
                if model_path and step_name in ['inference_pipeline', 'performance']:
                    if step_name == 'inference_pipeline':
                        self.validate_inference_pipeline = lambda: self.validate_inference_pipeline(model_path)
                    elif step_name == 'performance':
                        self.validate_performance = lambda: self.validate_performance(model_path)
                
                if result['status'] == 'PASS':
                    print(f"✅ {step_name}: PASSED ({execution_time:.2f}s)")
                    passed_validations += 1
                elif result['status'] == 'SKIP':
                    print(f"⏭️  {step_name}: SKIPPED ({execution_time:.2f}s)")
                    print(f"   Reason: {result.get('errors', ['Unknown'])[0]}")
                else:
                    print(f"❌ {step_name}: FAILED ({execution_time:.2f}s)")
                    if result.get('errors'):
                        print(f"   Errors: {result['errors'][0]}")
                        
            except Exception as e:
                print(f"💥 {step_name}: CRASHED")
                print(f"   Exception: {e}")
                self.results['validation_results'][step_name] = {
                    'status': 'CRASH',
                    'error': str(e)
                }
        
        # Calculate overall status
        pass_rate = passed_validations / total_validations
        if pass_rate >= 0.8:
            self.results['overall_status'] = 'PRODUCTION_READY'
        elif pass_rate >= 0.6:
            self.results['overall_status'] = 'NEEDS_MINOR_FIXES'
        else:
            self.results['overall_status'] = 'NOT_PRODUCTION_READY'
        
        self.results['pass_rate'] = pass_rate
        self.results['passed_validations'] = passed_validations
        self.results['total_validations'] = total_validations
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Save results
        self._save_results()
        
        print("\n" + "=" * 60)
        print(f"🎯 Production Validation Complete")
        print(f"📊 Overall Status: {self.results['overall_status']}")
        print(f"📈 Pass Rate: {pass_rate:.1%} ({passed_validations}/{total_validations})")
        
        return self.results
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for step_name, result in self.results['validation_results'].items():
            if result['status'] == 'FAIL':
                recommendations.append({
                    'category': 'critical',
                    'step': step_name,
                    'issue': f"Validation failed: {step_name}",
                    'recommendation': f"Fix errors in {step_name}: {result.get('errors', ['Unknown error'])[0]}"
                })
            elif result['status'] == 'SKIP':
                recommendations.append({
                    'category': 'warning',
                    'step': step_name,
                    'issue': f"Validation skipped: {step_name}",
                    'recommendation': f"Address prerequisites for {step_name}"
                })
        
        # System resource recommendations
        if 'system_resources' in self.results['validation_results']:
            resource_result = self.results['validation_results']['system_resources']
            for warning in resource_result.get('warnings', []):
                recommendations.append({
                    'category': 'performance',
                    'step': 'system_resources',
                    'issue': warning,
                    'recommendation': f"Address system resource issue: {warning}"
                })
        
        self.results['recommendations'] = recommendations
    
    def _save_results(self):
        """Save validation results to files."""
        # Save JSON report
        json_path = os.path.join(self.output_dir, 'production_validation_report.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save text summary
        summary_path = os.path.join(self.output_dir, 'production_validation_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("LSM Production Validation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Status: {self.results['overall_status']}\n")
            f.write(f"Pass Rate: {self.results['pass_rate']:.1%}\n")
            f.write(f"Validations Passed: {self.results['passed_validations']}/{self.results['total_validations']}\n\n")
            
            f.write("Validation Results:\n")
            f.write("-" * 20 + "\n")
            for step_name, result in self.results['validation_results'].items():
                f.write(f"{step_name}: {result['status']}\n")
                if result.get('errors'):
                    f.write(f"  Error: {result['errors'][0]}\n")
            
            f.write("\nRecommendations:\n")
            f.write("-" * 20 + "\n")
            for rec in self.results['recommendations']:
                f.write(f"[{rec['category'].upper()}] {rec['issue']}\n")
                f.write(f"  Recommendation: {rec['recommendation']}\n\n")
        
        print(f"📄 Full report saved to: {json_path}")
        print(f"📄 Summary saved to: {summary_path}")


def main():
    """Main function to run production validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LSM Production Validation')
    parser.add_argument('--output-dir', help='Output directory for validation results')
    
    args = parser.parse_args()
    
    # Run validation
    validator = ProductionValidator(args.output_dir)
    results = validator.run_comprehensive_validation()
    
    # Exit with appropriate code
    if results['overall_status'] == 'PRODUCTION_READY':
        print("\n✅ System is PRODUCTION READY!")
        return 0
    elif results['overall_status'] == 'NEEDS_MINOR_FIXES':
        print("\n⚠️  System needs minor fixes before production deployment.")
        return 1
    else:
        print("\n❌ System is NOT PRODUCTION READY.")
        return 2


if __name__ == "__main__":
    sys.exit(main())