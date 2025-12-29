#!/usr/bin/env python3
"""Integration test for main.py with minimal setup."""

import os
import sys
import subprocess
import tempfile
import shutil

print("Testing main.py integration...")

# Create a temporary directory for results
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

try:
    # Test if main.py can be called with --help
    result = subprocess.run(
        [sys.executable, "tensorflow_impl/main.py", "--help"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ main.py --help executed successfully")
    else:
        print(f"✗ main.py --help failed with return code {result.returncode}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    
    # Check if the help output contains expected arguments
    expected_args = ['--defence', '--attack', '--eps', '--bs', '--datapath']
    help_output = result.stdout + result.stderr
    
    for arg in expected_args:
        if arg in help_output:
            print(f"✓ Found expected argument: {arg}")
        else:
            print(f"✗ Missing expected argument: {arg}")
            sys.exit(1)
    
    print("\n✓ main.py integration test passed!")
    print("The main program is properly configured and can be executed.")
    
finally:
    # Clean up
    shutil.rmtree(temp_dir)
    print(f"\nCleaned up temporary directory")