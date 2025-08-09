import ast

try:
    with open('src/lsm/training/train.py', 'r') as f:
        content = f.read()
    
    ast.parse(content)
    print("File parses correctly")
    
    # Check class structure
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == 'LSMTrainer':
            print(f"Found LSMTrainer class with {len(node.body)} members")
            methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
            print(f"Methods: {methods}")
            
            # Check for response-level methods
            response_methods = [m for m in methods if 'response' in m.lower()]
            print(f"Response-level methods: {response_methods}")
            
except SyntaxError as e:
    print(f"Syntax error: {e}")
except Exception as e:
    print(f"Error: {e}")