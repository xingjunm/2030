#!/usr/bin/env python3
"""
Test script to verify Phase 4 implementation (Advanced Attack Algorithms)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import nn

# Test imports
print("Testing imports...")
try:
    from attacks.autopgd_pt import APGDAttack, APGDAttack_targeted
    print("✓ autopgd_pt imports successful")
except Exception as e:
    print(f"✗ autopgd_pt import error: {e}")

try:
    from attacks.fab_base import FABAttack
    print("✓ fab_base imports successful")
except Exception as e:
    print(f"✗ fab_base import error: {e}")

try:
    from attacks.fab_projections import projection_linf, projection_l2, projection_l1
    print("✓ fab_projections imports successful")
except Exception as e:
    print(f"✗ fab_projections import error: {e}")

try:
    from attacks.fab_pt import FABAttack_PT
    print("✓ fab_pt imports successful")
except Exception as e:
    print(f"✗ fab_pt import error: {e}")

# Create a simple model for testing
class SimpleModel(nn.Cell):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc = nn.Dense(784, num_classes)
    
    def construct(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)

# Test APGDAttack
print("\nTesting APGDAttack...")
try:
    model = SimpleModel()
    attack = APGDAttack(model, n_iter=10, eps=0.3, n_restarts=1, verbose=False)
    
    # Create dummy data
    x = ops.rand((4, 1, 28, 28), dtype=ms.float32)
    y = ops.randint(0, 10, (4,), dtype=ms.int32)
    
    # Run attack
    adv_x = attack.perturb(x, y)
    print(f"✓ APGDAttack test passed - input shape: {x.shape}, output shape: {adv_x.shape}")
except Exception as e:
    print(f"✗ APGDAttack test failed: {e}")

# Test APGDAttack_targeted
print("\nTesting APGDAttack_targeted...")
try:
    attack_targeted = APGDAttack_targeted(model, n_iter=10, eps=0.3, n_restarts=1, verbose=False)
    adv_x_targeted = attack_targeted.perturb(x, y)
    print(f"✓ APGDAttack_targeted test passed - input shape: {x.shape}, output shape: {adv_x_targeted.shape}")
except Exception as e:
    print(f"✗ APGDAttack_targeted test failed: {e}")

# Test FABAttack_PT
print("\nTesting FABAttack_PT...")
try:
    fab_attack = FABAttack_PT(model.construct, norm='Linf', n_iter=10, eps=0.3, n_restarts=1, verbose=False)
    adv_x_fab = fab_attack.perturb(x, y)
    print(f"✓ FABAttack_PT test passed - input shape: {x.shape}, output shape: {adv_x_fab.shape}")
except Exception as e:
    print(f"✗ FABAttack_PT test failed: {e}")

# Test projection functions
print("\nTesting projection functions...")
try:
    # Test data
    points = ops.rand((10, 20), dtype=ms.float32)
    w = ops.rand((10, 20), dtype=ms.float32)
    b = ops.rand((10,), dtype=ms.float32)
    
    # Test Linf projection
    proj_linf = projection_linf(points, w, b)
    print(f"✓ projection_linf test passed - output shape: {proj_linf.shape}")
    
    # Test L2 projection
    proj_l2 = projection_l2(points, w, b)
    print(f"✓ projection_l2 test passed - output shape: {proj_l2.shape}")
    
    # Test L1 projection
    proj_l1 = projection_l1(points, w, b)
    print(f"✓ projection_l1 test passed - output shape: {proj_l1.shape}")
except Exception as e:
    print(f"✗ Projection functions test failed: {e}")

print("\nPhase 4 verification complete!")