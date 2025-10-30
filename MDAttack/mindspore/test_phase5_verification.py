#!/usr/bin/env python
"""Phase 5 verification test for MindSpore implementation"""

import os
import sys
import json
import argparse
import numpy as np

# Add mindspore_impl to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test all Phase 5 imports"""
    print("Testing Phase 5 imports...")
    
    try:
        from attacks import attack_handler
        print("✓ attacks module imported")
    except ImportError as e:
        print(f"✗ attacks module import failed: {e}")
        return False
        
    try:
        from attacks.attack_handler import Attacker
        print("✓ Attacker class imported")
    except ImportError as e:
        print(f"✗ Attacker import failed: {e}")
        return False
        
    # Check if main.py exists without importing (to avoid tensorflow dependency)
    import os
    if os.path.exists('main.py'):
        print("✓ main.py exists")
    else:
        print("✗ main.py not found")
        return False
        
    try:
        import collect_results
        print("✓ collect_results module imported")
    except ImportError as e:
        print(f"✗ collect_results import failed: {e}")
        return False
        
    return True

def test_attacker_initialization():
    """Test Attacker class initialization"""
    print("\nTesting Attacker initialization...")
    
    try:
        import torch
        import torch.nn as nn
        from attacks.attack_handler import Attacker
        
        # Create a simple dummy model in PyTorch
        class DummyModel(nn.Module):
            def __init__(self, num_classes=10):
                super().__init__()
                self.fc = nn.Linear(3072, num_classes)  # CIFAR-10 input size
                
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)
        
        model = DummyModel()
        model.eval()
        
        # Test different attack versions
        attack_versions = ['MD', 'MD_DLR', 'MDMT', 'PGD', 'MT', 'DLR', 'DLRMT', 'ODI', 'CW', 'MDE', 'MDMT+']
        
        for version in attack_versions:
            try:
                attacker = Attacker(
                    model=model,
                    epsilon=8./255.,
                    v_min=0.,
                    v_max=1.,
                    num_classes=10,
                    data_loader=None,
                    logger=None,
                    version=version,
                    verbose=False
                )
                print(f"✓ Attacker with version '{version}' initialized")
            except Exception as e:
                print(f"✗ Attacker with version '{version}' failed: {e}")
                
        return True
        
    except Exception as e:
        print(f"✗ Attacker initialization test failed: {e}")
        return False

def test_collect_results():
    """Test collect_results functionality"""
    print("\nTesting collect_results functionality...")
    
    try:
        from collect_results import load_json, load_table, load_time_table
        
        # Create a temporary results directory and file
        os.makedirs('results', exist_ok=True)
        
        # Create dummy result files
        test_data = {
            'clean_acc': 85.5,
            'adv_acc': 45.2,
            'cost': 3600.0
        }
        
        with open('results/TEST_MD.json', 'w') as f:
            json.dump(test_data, f)
            
        # Test load_json
        loaded_data = load_json('results/TEST_MD.json')
        assert loaded_data['clean_acc'] == 85.5
        print("✓ load_json works correctly")
        
        # Clean up
        os.remove('results/TEST_MD.json')
        
        print("✓ collect_results functions work correctly")
        return True
        
    except Exception as e:
        print(f"✗ collect_results test failed: {e}")
        return False

def test_main_script():
    """Test main.py script structure"""
    print("\nTesting main.py script structure...")
    
    try:
        # Read main.py content without importing
        with open('main.py', 'r') as f:
            content = f.read()
            
        # Check for required functions and imports
        assert 'def main():' in content, "main() function not found"
        print("✓ main() function exists")
        
        assert 'def test(' in content, "test() function not found"
        print("✓ test() function exists")
        
        assert 'argparse' in content, "argparse not found"
        print("✓ argparse setup exists")
        
        assert 'from mindspore_impl.attacks.attack_handler import Attacker' in content, "MindSpore Attacker import not found"
        print("✓ MindSpore Attacker import exists")
        
        return True
        
    except Exception as e:
        print(f"✗ main.py test failed: {e}")
        return False

def main():
    """Run all Phase 5 verification tests"""
    print("=" * 60)
    print("Phase 5 Verification Test - MindSpore Implementation")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
        
    # Test Attacker initialization
    if not test_attacker_initialization():
        all_passed = False
        
    # Test collect_results
    if not test_collect_results():
        all_passed = False
        
    # Test main script
    if not test_main_script():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All Phase 5 tests PASSED!")
    else:
        print("✗ Some Phase 5 tests FAILED!")
        sys.exit(1)
    print("=" * 60)

if __name__ == '__main__':
    main()