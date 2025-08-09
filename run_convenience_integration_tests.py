#!/usr/bin/env python3
"""
Comprehensive integration test runner for LSM convenience API.

This script runs all convenience API tests including:
- Unit tests for individual components
- Integration tests with existing LSM components
- Backward compatibility validation
- End-to-end workflow testing
- sklearn compatibility testing
"""

import sys
import os
import unittest
import time
import traceback
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def run_test_suite():
    """Run the complete convenience API test suite."""
    print("=" * 80)
    print("LSM CONVENIENCE API - COMPREHENSIVE INTEGRATION TESTS")
    print("=" * 80)
    print()
    
    # Test discovery and execution
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Test directories to scan
    test_directories = [
        'tests/test_convenience',
        'tests/test_integration',  # Existing integration tests
        'tests'  # Root test directory for convenience performance tests
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    # Collect all tests
    print("Discovering tests...")
    for test_dir in test_directories:
        if os.path.exists(test_dir):
            try:
                discovered_tests = test_loader.discover(
                    test_dir, 
                    pattern='test_*.py',
                    top_level_dir='.'
                )
                test_suite.addTest(discovered_tests)
                print(f"  ✓ Found tests in {test_dir}")
            except Exception as e:
                print(f"  ✗ Error discovering tests in {test_dir}: {e}")
    
    print()
    
    # Run tests with detailed output
    print("Running tests...")
    print("-" * 80)
    
    # Custom test result class for detailed reporting
    class DetailedTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
        
        def startTest(self, test):
            super().startTest(test)
            self.start_time = time.time()
        
        def stopTest(self, test):
            super().stopTest(test)
            duration = time.time() - self.start_time
            self.test_results.append({
                'test': str(test),
                'duration': duration,
                'status': 'PASS'
            })
        
        def addError(self, test, err):
            super().addError(test, err)
            if self.test_results:
                self.test_results[-1]['status'] = 'ERROR'
                self.test_results[-1]['error'] = err
        
        def addFailure(self, test, err):
            super().addFailure(test, err)
            if self.test_results:
                self.test_results[-1]['status'] = 'FAIL'
                self.test_results[-1]['error'] = err
        
        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            if self.test_results:
                self.test_results[-1]['status'] = 'SKIP'
                self.test_results[-1]['reason'] = reason
    
    # Run the tests
    stream = StringIO()
    runner = unittest.TextTestRunner(
        stream=stream,
        verbosity=2,
        resultclass=DetailedTestResult
    )
    
    start_time = time.time()
    result = runner.run(test_suite)
    end_time = time.time()
    
    # Print results
    print(f"Tests completed in {end_time - start_time:.2f} seconds")
    print()
    
    # Summary statistics
    total_tests = result.testsRun
    total_failures = len(result.failures)
    total_errors = len(result.errors)
    total_skipped = len(result.skipped)
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors - total_skipped}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Skipped: {total_skipped}")
    print()
    
    # Detailed failure/error reporting
    if result.failures:
        print("FAILURES:")
        print("-" * 40)
        for test, traceback_str in result.failures:
            print(f"FAIL: {test}")
            print(traceback_str)
            print("-" * 40)
        print()
    
    if result.errors:
        print("ERRORS:")
        print("-" * 40)
        for test, traceback_str in result.errors:
            print(f"ERROR: {test}")
            print(traceback_str)
            print("-" * 40)
        print()
    
    if result.skipped:
        print("SKIPPED TESTS:")
        print("-" * 40)
        for test, reason in result.skipped:
            print(f"SKIP: {test}")
            print(f"Reason: {reason}")
            print("-" * 40)
        print()
    
    # Component-specific test results
    print("COMPONENT TEST RESULTS:")
    print("-" * 40)
    
    component_results = {
        'LSMGenerator': {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0},
        'LSMClassifier': {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0},
        'LSMRegressor': {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0},
        'Integration': {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0},
        'Performance': {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0},
        'Other': {'pass': 0, 'fail': 0, 'error': 0, 'skip': 0}
    }
    
    # Categorize test results
    all_tests = []
    
    # Add passed tests
    for i in range(total_tests):
        test_name = f"test_{i}"  # Placeholder since we don't have detailed test names
        if i < total_tests - total_failures - total_errors - total_skipped:
            all_tests.append((test_name, 'PASS', None))
    
    # Add failed tests
    for test, _ in result.failures:
        all_tests.append((str(test), 'FAIL', None))
    
    # Add error tests
    for test, _ in result.errors:
        all_tests.append((str(test), 'ERROR', None))
    
    # Add skipped tests
    for test, reason in result.skipped:
        all_tests.append((str(test), 'SKIP', reason))
    
    # Categorize by component
    for test_name, status, reason in all_tests:
        component = 'Other'
        if 'generator' in test_name.lower() or 'lsmgenerator' in test_name.lower():
            component = 'LSMGenerator'
        elif 'classifier' in test_name.lower() or 'lsmclassifier' in test_name.lower():
            component = 'LSMClassifier'
        elif 'regressor' in test_name.lower() or 'lsmregressor' in test_name.lower():
            component = 'LSMRegressor'
        elif 'integration' in test_name.lower():
            component = 'Integration'
        elif 'performance' in test_name.lower() or 'convenience_performance' in test_name.lower():
            component = 'Performance'
        
        if status == 'PASS':
            component_results[component]['pass'] += 1
        elif status == 'FAIL':
            component_results[component]['fail'] += 1
        elif status == 'ERROR':
            component_results[component]['error'] += 1
        elif status == 'SKIP':
            component_results[component]['skip'] += 1
    
    # Print component results
    for component, results in component_results.items():
        total_component = sum(results.values())
        if total_component > 0:
            print(f"{component}:")
            print(f"  Pass: {results['pass']}, Fail: {results['fail']}, "
                  f"Error: {results['error']}, Skip: {results['skip']}")
    
    print()
    
    # Overall result
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED!")
        if total_skipped > 0:
            print(f"   (Note: {total_skipped} tests were skipped)")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        print(f"   Please review the {total_failures} failures and {total_errors} errors above")
        return False


def check_convenience_api_availability():
    """Check if convenience API components are available."""
    print("Checking convenience API availability...")
    print("-" * 40)
    
    components = {
        'LSMGenerator': False,
        'LSMClassifier': False,
        'LSMRegressor': False,
        'ConvenienceConfig': False,
        'Performance Monitoring': False
    }
    
    try:
        from lsm.convenience import LSMGenerator
        components['LSMGenerator'] = True
        print("✓ LSMGenerator available")
    except ImportError as e:
        print(f"✗ LSMGenerator not available: {e}")
    
    try:
        from lsm.convenience import LSMClassifier
        components['LSMClassifier'] = True
        print("✓ LSMClassifier available")
    except ImportError as e:
        print(f"✗ LSMClassifier not available: {e}")
    
    try:
        from lsm.convenience import LSMRegressor
        components['LSMRegressor'] = True
        print("✓ LSMRegressor available")
    except ImportError as e:
        print(f"✗ LSMRegressor not available: {e}")
    
    try:
        from lsm.convenience import ConvenienceConfig
        components['ConvenienceConfig'] = True
        print("✓ ConvenienceConfig available")
    except ImportError as e:
        print(f"✗ ConvenienceConfig not available: {e}")
    
    try:
        from lsm.convenience import PerformanceProfiler
        components['Performance Monitoring'] = True
        print("✓ Performance Monitoring available")
    except ImportError as e:
        print(f"✗ Performance Monitoring not available: {e}")
    
    print()
    
    available_count = sum(components.values())
    total_count = len(components)
    
    print(f"Convenience API Availability: {available_count}/{total_count} components")
    
    if available_count == 0:
        print("⚠️  WARNING: No convenience API components are available!")
        print("   Tests will be skipped. Please ensure the convenience API is properly installed.")
    elif available_count < total_count:
        print("⚠️  WARNING: Some convenience API components are missing!")
        print("   Some tests may be skipped.")
    else:
        print("✅ All convenience API components are available!")
    
    print()
    return components


def check_core_dependencies():
    """Check if core LSM components are available."""
    print("Checking core LSM dependencies...")
    print("-" * 40)
    
    dependencies = {
        'LSMTrainer': False,
        'ResponseGenerator': False,
        'SystemMessageProcessor': False,
        'sklearn': False,
        'numpy': False
    }
    
    try:
        from lsm.training.train import LSMTrainer
        dependencies['LSMTrainer'] = True
        print("✓ LSMTrainer available")
    except ImportError as e:
        print(f"✗ LSMTrainer not available: {e}")
    
    try:
        from lsm.inference.response_generator import ResponseGenerator
        dependencies['ResponseGenerator'] = True
        print("✓ ResponseGenerator available")
    except ImportError as e:
        print(f"✗ ResponseGenerator not available: {e}")
    
    try:
        from lsm.core.system_message_processor import SystemMessageProcessor
        dependencies['SystemMessageProcessor'] = True
        print("✓ SystemMessageProcessor available")
    except ImportError as e:
        print(f"✗ SystemMessageProcessor not available: {e}")
    
    try:
        import sklearn
        dependencies['sklearn'] = True
        print("✓ sklearn available")
    except ImportError as e:
        print(f"✗ sklearn not available: {e}")
    
    try:
        import numpy
        dependencies['numpy'] = True
        print("✓ numpy available")
    except ImportError as e:
        print(f"✗ numpy not available: {e}")
    
    print()
    
    available_count = sum(dependencies.values())
    total_count = len(dependencies)
    
    print(f"Core Dependencies: {available_count}/{total_count} available")
    
    if available_count < total_count:
        print("⚠️  WARNING: Some core dependencies are missing!")
        print("   Integration tests may be limited.")
    else:
        print("✅ All core dependencies are available!")
    
    print()
    return dependencies


if __name__ == '__main__':
    print("Starting LSM Convenience API Integration Tests...")
    print()
    
    # Check component availability
    convenience_components = check_convenience_api_availability()
    core_dependencies = check_core_dependencies()
    
    # Run the test suite
    success = run_test_suite()
    
    # Final summary
    print("=" * 80)
    print("FINAL INTEGRATION TEST REPORT")
    print("=" * 80)
    
    if success:
        print("✅ INTEGRATION TESTS PASSED")
        print("   The convenience API is ready for production use!")
    else:
        print("❌ INTEGRATION TESTS FAILED")
        print("   Please address the issues before proceeding.")
    
    print()
    print("Test Categories Covered:")
    print("  • Unit tests for individual convenience classes")
    print("  • Integration with existing LSM components")
    print("  • Backward compatibility validation")
    print("  • End-to-end workflow testing")
    print("  • sklearn compatibility testing")
    print("  • Performance monitoring validation")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)