#!/usr/bin/env python3
"""Check if the methods were added to the file."""

# Read the file and check for the methods
with open('src/lsm/training/train.py', 'r') as f:
    content = f.read()

methods_to_check = [
    'def initialize_response_level_training',
    'def prepare_response_level_data',
    'def _calculate_system_influence',
    'def validate_system_aware_generation',
    'def update_model_configuration_for_3d_cnn',
    '_add_missing_methods_to_lsm_trainer'
]

print("Checking for methods in the file:")
for method in methods_to_check:
    if method in content:
        print(f"✓ Found: {method}")
    else:
        print(f"✗ Missing: {method}")

# Check if the function call is there
if '_add_missing_methods_to_lsm_trainer()' in content:
    print("✓ Function call found")
else:
    print("✗ Function call missing")

# Check line count
lines = content.split('\n')
print(f"\nFile has {len(lines)} lines")
print(f"Last 5 lines:")
for i, line in enumerate(lines[-5:], len(lines)-4):
    print(f"{i}: {line}")