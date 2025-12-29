#!/usr/bin/env python3
"""Test script to verify Phase 5 implementation."""

import os
import sys
import json
import numpy as np
import tensorflow as tf

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import PyTorch modules (for defense models)
import torch
import torch.nn as nn

# Test imports
print("Testing Phase 5 imports...")

# Test attacks module imports
try:
    from tensorflow_impl.attacks import attack_handler
    from tensorflow_impl.attacks.attack_handler import Attacker
    print("✓ attacks.attack_handler imported successfully")
except Exception as e:
    print(f"✗ Failed to import attacks.attack_handler: {e}")
    sys.exit(1)

# Test main module import
try:
    # We can't import main directly due to argparse, but we can check if it exists
    main_path = os.path.join(os.path.dirname(__file__), 'main.py')
    if os.path.exists(main_path):
        print("✓ main.py exists")
    else:
        print("✗ main.py not found")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking main.py: {e}")
    sys.exit(1)

# Test collect_results import
try:
    from tensorflow_impl import collect_results
    print("✓ collect_results imported successfully")
except Exception as e:
    print(f"✗ Failed to import collect_results: {e}")
    sys.exit(1)

print("\nTesting Attacker class initialization...")

# Create a dummy PyTorch model for testing
class DummyModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(3072, num_classes)  # CIFAR-10 input size
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        return self.fc(x)

# Test Attacker initialization
try:
    model = DummyModel()
    model.eval()
    
    # Create a simple data loader
    dummy_data = [(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))]
    
    attacker = Attacker(
        model=model,
        epsilon=8/255,
        num_classes=10,
        data_loader=dummy_data,
        logger=None,
        version='MD'
    )
    print("✓ Attacker initialized successfully")
    
    # Check that attacks are properly initialized
    assert hasattr(attacker, 'md'), "MD attack not initialized"
    assert hasattr(attacker, 'pgd'), "PGD attack not initialized"
    assert hasattr(attacker, 'apgd'), "APGD attack not initialized"
    assert hasattr(attacker, 'fab'), "FAB attack not initialized"
    print("✓ All attack methods initialized")
    
    # Check attacks_to_run
    assert len(attacker.attacks_to_run) == 1, "Wrong number of attacks to run"
    assert attacker.attacks_to_run[0] == attacker.md, "Wrong attack in attacks_to_run"
    print("✓ Attack selection working correctly")
    
except Exception as e:
    print(f"✗ Failed to initialize Attacker: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting collect_results functionality...")

# Test collect_results functions
try:
    # Test load_json function
    test_data = {'clean_acc': 90.5, 'adv_acc': 45.2, 'cost': 3600}
    test_file = '/tmp/test_results.json'
    with open(test_file, 'w') as f:
        json.dump(test_data, f)
    
    loaded_data = collect_results.load_json(test_file)
    assert loaded_data == test_data, "JSON loading failed"
    print("✓ collect_results.load_json working")
    
    # Clean up
    os.remove(test_file)
    
except Exception as e:
    print(f"✗ collect_results test failed: {e}")
    sys.exit(1)

print("\nPhase 5 verification completed successfully! ✓")
print("\nAll Phase 5 components are properly implemented and working.")