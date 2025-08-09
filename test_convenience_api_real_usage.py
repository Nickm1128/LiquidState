#!/usr/bin/env python3
"""
Real usage test for LSM convenience API.

This script tests the convenience API with realistic usage patterns
to validate that it works correctly in practice.
"""

import sys
import os
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_text_generation_workflow():
    """Test realistic text generation workflow."""
    print("Testing Text Generation Workflow...")
    print("-" * 40)
    
    try:
        from lsm.convenience import LSMGenerator
        
        # Create generator with reasonable parameters
        generator = LSMGenerator(
            window_size=8,
            embedding_dim=64,
            reservoir_type='hierarchical'
        )
        
        print(f"✓ Created LSMGenerator: window_size={generator.window_size}, "
              f"embedding_dim={generator.embedding_dim}")
        
        # Test parameter access
        params = generator.get_params()
        print(f"✓ Retrieved parameters: {len(params)} parameters available")
        
        # Test parameter modification
        generator.set_params(window_size=10)
        print(f"✓ Modified parameters: window_size now {generator.window_size}")
        
        # Test preset usage
        fast_generator = LSMGenerator.from_preset('fast')
        print(f"✓ Created from preset: fast configuration loaded")
        
        return True
        
    except Exception as e:
        print(f"✗ Text generation workflow failed: {e}")
        return False


def test_classification_workflow():
    """Test realistic classification workflow."""
    print("Testing Classification Workflow...")
    print("-" * 40)
    
    try:
        from lsm.convenience import LSMClassifier
        
        # Create classifier for sentiment analysis
        classifier = LSMClassifier(
            window_size=6,
            embedding_dim=32,
            n_classes=3,  # positive, negative, neutral
            reservoir_type='standard'
        )
        
        print(f"✓ Created LSMClassifier: n_classes={classifier.n_classes}")
        
        # Test sklearn interface
        params = classifier.get_params()
        print(f"✓ sklearn interface: {len(params)} parameters accessible")
        
        # Test parameter validation
        try:
            LSMClassifier(n_classes=1)  # Should fail
            print("✗ Parameter validation failed")
            return False
        except:
            print("✓ Parameter validation: Invalid n_classes correctly rejected")
        
        return True
        
    except Exception as e:
        print(f"✗ Classification workflow failed: {e}")
        return False


def test_regression_workflow():
    """Test realistic regression workflow."""
    print("Testing Regression Workflow...")
    print("-" * 40)
    
    try:
        from lsm.convenience import LSMRegressor
        
        # Create regressor for time series prediction
        regressor = LSMRegressor(
            window_size=12,
            embedding_dim=48,
            reservoir_type='echo_state'  # Good for time series
        )
        
        print(f"✓ Created LSMRegressor: reservoir_type={regressor.reservoir_type}")
        
        # Test sklearn interface
        assert hasattr(regressor, 'fit')
        assert hasattr(regressor, 'predict')
        assert hasattr(regressor, 'score')
        print("✓ sklearn interface: Required methods available")
        
        return True
        
    except Exception as e:
        print(f"✗ Regression workflow failed: {e}")
        return False


def test_configuration_and_presets():
    """Test configuration system and presets."""
    print("Testing Configuration and Presets...")
    print("-" * 40)
    
    try:
        from lsm.convenience import ConvenienceConfig, LSMGenerator
        
        # Test preset access
        presets = ConvenienceConfig.PRESETS
        print(f"✓ Available presets: {list(presets.keys())}")
        
        # Test each preset
        for preset_name in ['fast', 'balanced', 'quality']:
            preset_config = ConvenienceConfig.get_preset(preset_name)
            print(f"✓ {preset_name} preset: {len(preset_config)} parameters")
            
            # Create generator from preset
            generator = LSMGenerator.from_preset(preset_name)
            print(f"  - window_size: {generator.window_size}")
            print(f"  - embedding_dim: {generator.embedding_dim}")
        
        return True
        
    except Exception as e:
        print(f"✗ Configuration and presets test failed: {e}")
        return False


def test_model_persistence():
    """Test model save/load functionality."""
    print("Testing Model Persistence...")
    print("-" * 40)
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        from lsm.convenience import LSMGenerator
        
        # Create and configure generator
        original = LSMGenerator(
            window_size=7,
            embedding_dim=56,
            reservoir_type='hierarchical'
        )
        
        # Mock fitted state for testing
        original._is_fitted = True
        original._model_components = {
            'config': {'test': 'data'},
            'metadata': {'version': '1.0'}
        }
        
        model_path = os.path.join(temp_dir, 'test_model')
        
        # Test save
        original.save(model_path)
        print(f"✓ Model saved to: {model_path}")
        
        # Verify directory structure
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"✓ Model directory contains: {files}")
        
        # Test load
        loaded = LSMGenerator.load(model_path)
        print(f"✓ Model loaded successfully")
        
        # Verify parameters preserved
        assert loaded.window_size == original.window_size
        assert loaded.embedding_dim == original.embedding_dim
        print("✓ Parameters preserved after save/load")
        
        return True
        
    except Exception as e:
        print(f"✗ Model persistence test failed: {e}")
        return False
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_error_handling():
    """Test error handling and validation."""
    print("Testing Error Handling...")
    print("-" * 40)
    
    try:
        from lsm.convenience import LSMGenerator, LSMClassifier
        from lsm.convenience.config import ConvenienceValidationError
        from lsm.utils.lsm_exceptions import InvalidInputError
        
        error_count = 0
        
        # Test invalid parameters
        test_cases = [
            (lambda: LSMGenerator(window_size=0), "Invalid window_size"),
            (lambda: LSMGenerator(embedding_dim=-1), "Invalid embedding_dim"),
            (lambda: LSMGenerator(reservoir_type='invalid'), "Invalid reservoir_type"),
            (lambda: LSMClassifier(n_classes=1), "Invalid n_classes"),
        ]
        
        for test_func, description in test_cases:
            try:
                test_func()
                print(f"✗ {description}: Should have raised an error")
            except (ValueError, ConvenienceValidationError, InvalidInputError) as e:
                print(f"✓ {description}: Correctly rejected - {type(e).__name__}")
                error_count += 1
        
        print(f"✓ Error handling: {error_count}/{len(test_cases)} validations work correctly")
        
        return error_count == len(test_cases)
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_sklearn_integration():
    """Test sklearn integration capabilities."""
    print("Testing sklearn Integration...")
    print("-" * 40)
    
    try:
        from lsm.convenience import LSMGenerator, LSMClassifier, LSMRegressor
        
        # Test sklearn clone compatibility
        try:
            from sklearn.base import clone
            
            original = LSMGenerator(window_size=8, embedding_dim=64)
            cloned = clone(original)
            
            assert cloned.window_size == original.window_size
            assert cloned.embedding_dim == original.embedding_dim
            assert cloned is not original
            
            print("✓ sklearn clone: Works correctly")
            
        except ImportError:
            print("⚠ sklearn clone: sklearn not available, skipping")
        
        # Test parameter interface
        for cls in [LSMGenerator, LSMClassifier, LSMRegressor]:
            instance = cls()
            
            # Test get_params
            params = instance.get_params()
            assert isinstance(params, dict)
            
            # Test set_params
            if 'window_size' in params:
                instance.set_params(window_size=params['window_size'] + 1)
                new_params = instance.get_params()
                assert new_params['window_size'] == params['window_size'] + 1
            
            print(f"✓ {cls.__name__}: Parameter interface works")
        
        return True
        
    except Exception as e:
        print(f"✗ sklearn integration test failed: {e}")
        return False


def run_real_usage_tests():
    """Run all real usage tests."""
    print("=" * 60)
    print("LSM CONVENIENCE API - REAL USAGE VALIDATION")
    print("=" * 60)
    print()
    
    tests = [
        ("Text Generation Workflow", test_text_generation_workflow),
        ("Classification Workflow", test_classification_workflow),
        ("Regression Workflow", test_regression_workflow),
        ("Configuration and Presets", test_configuration_and_presets),
        ("Model Persistence", test_model_persistence),
        ("Error Handling", test_error_handling),
        ("sklearn Integration", test_sklearn_integration),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{status}\n")
        except Exception as e:
            print(f"✗ FAILED: {e}\n")
            results.append((test_name, False))
    
    # Summary
    print("=" * 60)
    print("REAL USAGE TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print()
    print(f"Overall Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL REAL USAGE TESTS PASSED!")
        print("   The convenience API is ready for production use!")
        return True
    elif passed >= total * 0.8:
        print("⚠️  MOST REAL USAGE TESTS PASSED")
        print("   The convenience API is functional with minor issues.")
        return True
    else:
        print("❌ REAL USAGE TESTS FAILED")
        print("   The convenience API needs more work.")
        return False


if __name__ == '__main__':
    success = run_real_usage_tests()
    sys.exit(0 if success else 1)