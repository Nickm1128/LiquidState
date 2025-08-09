#!/usr/bin/env python3
"""
LSM Command Line Interface Entry Point

This script provides a convenient entry point for the LSM convenience API CLI.
It can be used directly from the command line or as a module.

Usage:
    python lsm_cli.py train-generator --data-path conversations.txt
    python lsm_cli.py generate --model-path ./model --prompt "Hello there!"
    python -m lsm_cli train-classifier --data-path data.csv
"""

import sys
import os

# Add src to path to ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from lsm.convenience.cli import main
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error importing LSM convenience CLI: {e}")
    print("Please ensure the src/lsm package is properly installed.")
    print("You may need to install required dependencies:")
    print("  pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error running LSM CLI: {e}")
    sys.exit(1)