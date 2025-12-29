#!/usr/bin/env python3
"""
Debug test for Phase 4 gradient issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mindspore as ms
import mindspore.ops as ops
from mindspore import nn, Tensor
from mindspore.ops import GradOperation

# Simple model
class SimpleModel(nn.Cell):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc = nn.Dense(784, num_classes)
    
    def construct(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        return self.fc(x)

# Test gradient computation
print("Testing gradient computation...")

model = SimpleModel()
x = Tensor([[[[0.5] * 28] * 28]], ms.float32)
y = Tensor([5], ms.int32)

# Test 1: Simple gradient
print("\nTest 1: Simple gradient")
def forward_simple(x):
    output = model(x)
    loss = output.sum()
    return loss

grad_fn_simple = GradOperation()(forward_simple)
try:
    grad = grad_fn_simple(x)
    print("✓ Simple gradient works")
except Exception as e:
    print(f"✗ Simple gradient failed: {e}")

# Test 2: Gradient with auxiliary outputs
print("\nTest 2: Gradient with auxiliary outputs")
def forward_with_aux(x):
    output = model(x)
    loss = output.sum()
    return loss, output

grad_fn_aux = GradOperation(get_all=False, get_by_list=True, sens_param=True)
try:
    # Correct way to use grad_fn with auxiliary outputs
    grad_compute = grad_fn_aux(forward_with_aux, (x,))
    grad = grad_compute(Tensor(1.0, ms.float32))
    print("✓ Gradient with aux works")
except Exception as e:
    print(f"✗ Gradient with aux failed: {e}")

# Test 3: How APGDAttack should do it
print("\nTest 3: APGDAttack style gradient")
criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')

def forward_apgd(x_adv):
    logits = model(x_adv)
    loss_indiv = criterion(logits, y)
    loss = loss_indiv.sum()
    return loss

grad_fn_apgd = GradOperation(get_all=True)
try:
    grad = grad_fn_apgd(forward_apgd)(x)
    print(f"✓ APGDAttack style gradient works, grad shape: {grad[0].shape}")
except Exception as e:
    print(f"✗ APGDAttack style gradient failed: {e}")

print("\nDebug test complete!")