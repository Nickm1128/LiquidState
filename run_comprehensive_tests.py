#!/usr/bin/env python3
"""
Master test runner for comprehensive LSM functionality tests.

This script runs all comprehensive tests for the new LSM functionality including:
- DialogueTokenizer save/load and decoding methods
- Integration tests for complete train-save-load-predict workflow
- Performance tests for inference speed and memory usage
- Backward compatibility tests with mock old model formats
"""

import sys
import time
import subprocess
from typing import List, Tuple, Dict, Any

def run_test_module(module_name: str, description: str) -> Tuple[bool, float, str]:
    """
    Run a test module and return results.
    
    Args:
        module_name: Name of the test module to run
        description: Human-readable description of the test
        
    Returns:
        Tuple of (success, duration, output)
    """
    print(f"\nRunning {description}")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the test module
        result = subprocess.run(
            [sys.executable, module_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        
        success = result.returncode == 0
        
        if success:
            print(f"PASS: {description} completed successfully in {duration:.2f}s")
        else:
            print(f"FAIL: {description} failed after {duration:.2f}s")
        
        return success, duration, result.stdout + result.stderr
        
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"TIMEOUT: {description} timed out after {duration:.2f}s")
        return False, duration, "Test timed out"
    
    except Exception as e:
        duration = time.time() - start_time
        print(f"CRASH: {description} crashed: {e}")
        return False, duration, str(e)

def main():
    """Run all comprehensive tests."""
    print("LSM Comprehensive Test Suite")
    print("=" * 60)
    print("Running comprehensive tests for all new LSM functionality...")
    
    # Define test modules to run
    test_modules = [
        ("test_tokenizer.py", "DialogueTokenizer Tests"),
        ("test_comprehensive_functionality_simple.py", "Integration & Core Functionality Tests"),
        ("test_backward_compatibility.py", "Backward Compatibility Tests"),
        ("test_performance_optimization.py", "Performance Optimization Tests"),
        ("test_error_handling.py", "Error Handling Tests"),
        ("test_model_manager.py", "Model Manager Tests"),
    ]
    
    # Track results
    results: List[Dict[str, Any]] = []
    total_start_time = time.time()
    
    # Run each test module
    for module_name, description in test_modules:
        success, duration, output = run_test_module(module_name, description)
        
        results.append({
            'module': module_name,
            'description': description,
            'success': success,
            'duration': duration,
            'output': output
        })
    
    total_duration = time.time() - total_start_time
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    passed_tests = sum(1 for r in results if r['success'])
    total_tests = len(results)
    
    print(f"Total test modules: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Total execution time: {total_duration:.2f}s")
    
    print(f"\nDetailed Results:")
    print("-" * 80)
    
    for result in results:
        status = "PASS" if result['success'] else "FAIL"
        print(f"{status} | {result['description']:<40} | {result['duration']:>6.2f}s")
    
    # Show failed tests in detail
    failed_tests = [r for r in results if not r['success']]
    if failed_tests:
        print(f"\nFailed Test Details:")
        print("-" * 80)
        
        for result in failed_tests:
            print(f"\n{result['description']} ({result['module']}):")
            print(f"Duration: {result['duration']:.2f}s")
            
            # Show last few lines of output for context
            output_lines = result['output'].split('\n')
            relevant_lines = [line for line in output_lines[-20:] if line.strip()]
            if relevant_lines:
                print("Last output:")
                for line in relevant_lines[-10:]:  # Show last 10 relevant lines
                    print(f"  {line}")
    
    # Performance summary
    print(f"\nPerformance Summary:")
    print("-" * 80)
    
    fastest = min(results, key=lambda x: x['duration'])
    slowest = max(results, key=lambda x: x['duration'])
    
    print(f"Fastest test: {fastest['description']} ({fastest['duration']:.2f}s)")
    print(f"Slowest test: {slowest['description']} ({slowest['duration']:.2f}s)")
    print(f"Average test time: {sum(r['duration'] for r in results) / len(results):.2f}s")
    
    # Final verdict
    print(f"\nFinal Verdict:")
    print("=" * 80)
    
    if passed_tests == total_tests:
        print("ALL TESTS PASSED! The LSM system is ready for production.")
        print("\nKey achievements:")
        print("  - DialogueTokenizer persistence and decoding working correctly")
        print("  - Complete train-save-load-predict workflow functional")
        print("  - Performance optimizations validated")
        print("  - Backward compatibility handled gracefully")
        print("  - Error handling comprehensive and robust")
        return 0
    else:
        print(f"WARNING: {total_tests - passed_tests} test module(s) failed.")
        print("Please review the failed tests above and fix the issues.")
        print("\nNext steps:")
        print("  1. Review failed test output above")
        print("  2. Fix identified issues")
        print("  3. Re-run tests to verify fixes")
        return 1

if __name__ == "__main__":
    exit(main())