#!/usr/bin/env python3
"""Quick test of validation functionality."""

from input_validation import validate_positive_integer, InvalidInputError
from lsm_exceptions import LSMError

def test_validation():
    print("Testing input validation...")
    
    # Test valid input
    try:
        result = validate_positive_integer(5, "test_param")
        print(f"✓ Valid input accepted: {result}")
    except Exception as e:
        print(f"❌ Valid input rejected: {e}")
    
    # Test invalid input
    try:
        validate_positive_integer(-1, "test_param")
        print("❌ Invalid input accepted (should have failed)")
    except InvalidInputError as e:
        print(f"✓ Invalid input correctly rejected: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def test_exceptions():
    print("\nTesting custom exceptions...")
    
    try:
        raise LSMError("Test error", {"context": "test"})
    except LSMError as e:
        print(f"✓ Custom exception working: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_validation()
    test_exceptions()
    print("\n🎉 Quick validation tests completed!")