#!/usr/bin/env python3
"""
Simple test for Phase 4 with better error handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor

# Simple model
class SimpleModel(nn.Cell):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc = nn.Dense(784, num_classes)
    
    def construct(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)

# Test basic functionality
print("Testing basic imports and attack instantiation...")
try:
    from attacks.autopgd_pt import APGDAttack
    from attacks.fab_pt import FABAttack_PT
    
    model = SimpleModel()
    
    # Test with simple data
    x = Tensor([[[[0.5] * 28] * 28]], ms.float32)  # Single image
    y = Tensor([5], ms.int32)  # Single label
    
    print("Input shape:", x.shape)
    print("Label shape:", y.shape)
    
    # Test APGDAttack
    print("\nTesting APGDAttack...")
    attack = APGDAttack(model, n_iter=2, eps=0.1, verbose=False)
    adv_x = attack.perturb(x, y)
    print("✓ APGDAttack works - output shape:", adv_x.shape)
    
    # Test FABAttack_PT with debugging
    print("\nTesting FABAttack_PT...")
    fab_attack = FABAttack_PT(model.construct, norm='Linf', n_iter=2, eps=0.1, verbose=True)
    
    # Get model prediction first
    output = model(x)
    print("Model output shape:", output.shape)
    print("Model prediction:", output.argmax(1))
    
    try:
        adv_x_fab = fab_attack.perturb(x, y)
        print("✓ FABAttack_PT works - output shape:", adv_x_fab.shape)
    except Exception as e:
        print(f"✗ FABAttack_PT failed with error: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nSimple test complete!")