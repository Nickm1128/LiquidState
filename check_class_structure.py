import ast

with open('src/lsm/training/train.py', 'r') as f:
    lines = f.readlines()

# Find the LSMTrainer class
in_class = False
class_indent = 0
method_count = 0

for i, line in enumerate(lines):
    if 'class LSMTrainer:' in line:
        in_class = True
        class_indent = len(line) - len(line.lstrip())
        print(f"Line {i+1}: Found LSMTrainer class, indent={class_indent}")
        continue
    
    if in_class:
        current_indent = len(line) - len(line.lstrip())
        
        # If we hit a line with same or less indentation than class, class has ended
        if line.strip() and current_indent <= class_indent:
            print(f"Line {i+1}: Class ended, indent={current_indent}, line: {line.strip()[:50]}")
            break
        
        # Check for method definitions
        if line.strip().startswith('def '):
            method_name = line.strip().split('(')[0].replace('def ', '')
            print(f"Line {i+1}: Method '{method_name}', indent={current_indent}")
            method_count += 1
            
            # Check for response-level methods
            if 'response' in method_name.lower():
                print(f"  -> Found response-level method: {method_name}")

print(f"Total methods found: {method_count}")