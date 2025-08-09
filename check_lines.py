#!/usr/bin/env python3
"""
Check the classifier file line by line to find the issue.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Checking classifier file line by line...")

# Read the file and execute it line by line
with open('src/lsm/convenience/classifier.py', 'r') as f:
    lines = f.readlines()

print(f"File has {len(lines)} lines")

# Execute the file in chunks to find where it fails
namespace = {}
current_chunk = ""
line_num = 0

try:
    for i, line in enumerate(lines):
        line_num = i + 1
        current_chunk += line
        
        # Try to execute every 10 lines or at class definition
        if (i + 1) % 10 == 0 or 'class LSMClassifier' in line or i == len(lines) - 1:
            try:
                exec(current_chunk, namespace)
                print(f"✓ Lines 1-{line_num} executed successfully")
                current_chunk = ""
            except Exception as e:
                print(f"✗ Error at line {line_num}: {e}")
                print(f"Line content: {line.strip()}")
                break

    # Check if class was defined
    if 'LSMClassifier' in namespace:
        print("✓ LSMClassifier found in namespace")
    else:
        print("✗ LSMClassifier not found in namespace")
        print("Available names:", [x for x in namespace.keys() if not x.startswith('_')])

except Exception as e:
    print(f"✗ Execution failed at line {line_num}: {e}")
    print(f"Current chunk:\n{current_chunk}")

print("Line-by-line check completed!")