#!/usr/bin/env python3
"""
Fix TrainingExecutionError constructor calls
"""

import re

def fix_training_errors():
    """Fix all TrainingExecutionError constructor calls in train.py"""
    
    file_path = "src/lsm/training/train.py"
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Pattern to match TrainingExecutionError calls with single string argument
    pattern = r'raise TrainingExecutionError\(f?"([^"]+)"\)'
    
    # Replace with correct constructor call
    def replacement(match):
        message = match.group(1)
        return f'raise TrainingExecutionError(None, f"{message}")'
    
    # Apply the replacement
    new_content = re.sub(pattern, replacement, content)
    
    # Also fix the case without f-string
    pattern2 = r'raise TrainingExecutionError\("([^"]+)"\)'
    def replacement2(match):
        message = match.group(1)
        return f'raise TrainingExecutionError(None, "{message}")'
    
    new_content = re.sub(pattern2, replacement2, new_content)
    
    # Write back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("Fixed TrainingExecutionError constructor calls")

if __name__ == "__main__":
    fix_training_errors()