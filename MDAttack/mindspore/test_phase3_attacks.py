import sys
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

# Add parent directory to path
sys.path.append('/root/MDAttack/mindspore_impl')

# Import the migrated modules
from attacks.utils import adv_check_and_update, one_hot_tensor
from attacks.PGD import PGDAttack, MTPGDAttack, cw_loss, margin_loss
from attacks.MD import MDAttack, MDMTAttack

# Simple test model
class SimpleModel(nn.Cell):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Dense(784, 128)
        self.fc2 = nn.Dense(128, num_classes)
        self.relu = nn.ReLU()
        
    def construct(self, x):
        # Flatten if needed
        if len(x.shape) == 4:
            x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def test_utils():
    print("Testing utils functions...")
    
    # Test one_hot_tensor
    y = Tensor([0, 1, 2, 3], ms.int32)
    one_hot = one_hot_tensor(y, 10)
    print(f"One-hot shape: {one_hot.shape}")
    assert one_hot.shape == (4, 10), "One-hot shape mismatch"
    
    # Test adv_check_and_update
    batch_size = 4
    num_classes = 10
    X_cur = Tensor(np.random.randn(batch_size, 784), ms.float32)
    logits = Tensor(np.random.randn(batch_size, num_classes), ms.float32)
    y = Tensor([0, 1, 2, 3], ms.int32)
    not_correct = ops.zeros_like(y).astype(ms.int64)
    X_adv = ops.zeros_like(X_cur)
    
    X_adv_new, nc_new = adv_check_and_update(X_cur, logits, y, not_correct, X_adv)
    print(f"Updated adversarial examples shape: {X_adv_new.shape}")
    print("✓ Utils functions test passed")

def test_loss_functions():
    print("\nTesting loss functions...")
    
    batch_size = 4
    num_classes = 10
    logits = Tensor(np.random.randn(batch_size, num_classes), ms.float32)
    y = Tensor([0, 1, 2, 3], ms.int32)
    
    # Test CW loss
    cw = cw_loss(logits, y)
    print(f"CW loss: {cw}")
    
    # Test margin loss
    margin = margin_loss(logits, y)
    print(f"Margin loss: {margin}")
    print("✓ Loss functions test passed")

def test_pgd_attack():
    print("\nTesting PGD Attack...")
    
    model = SimpleModel()
    pgd_attack = PGDAttack(model, epsilon=0.3, num_steps=10, step_size=0.01, 
                          num_restarts=1, random_start=True)
    
    # Test input - ensure it's in [0, 1] range
    x = Tensor(np.random.rand(4, 784), ms.float32)  # rand generates [0, 1)
    y = Tensor([0, 1, 2, 3], ms.int32)
    
    # Run attack
    x_adv = pgd_attack.perturb(x, y)
    print(f"Adversarial examples shape: {x_adv.shape}")
    
    # Check epsilon constraint
    diff = ops.abs(x_adv - x)
    max_diff = ops.reduce_max(diff)
    print(f"Max perturbation: {max_diff.asnumpy():.4f} (epsilon: 0.3)")
    assert max_diff <= 0.3 + 1e-6, "Epsilon constraint violated"
    print("✓ PGD Attack test passed")

def test_mtpgd_attack():
    print("\nTesting MTPGD Attack...")
    
    model = SimpleModel()
    mtpgd_attack = MTPGDAttack(model, epsilon=0.3, num_steps=10, step_size=0.01,
                              num_restarts=1, random_start=True)
    
    # Test input - ensure it's in [0, 1] range
    x = Tensor(np.random.rand(4, 784), ms.float32)  # rand generates [0, 1)
    y = Tensor([0, 1, 2, 3], ms.int32)
    
    # Run attack
    x_adv = mtpgd_attack.perturb(x, y)
    print(f"Adversarial examples shape: {x_adv.shape}")
    
    # Check epsilon constraint
    diff = ops.abs(x_adv - x)
    max_diff = ops.reduce_max(diff)
    print(f"Max perturbation: {max_diff.asnumpy():.4f} (epsilon: 0.3)")
    assert max_diff <= 0.3 + 1e-6, "Epsilon constraint violated"
    print("✓ MTPGD Attack test passed")

def test_md_attack():
    print("\nTesting MD Attack...")
    
    model = SimpleModel()
    md_attack = MDAttack(model, epsilon=0.3, num_steps=10, step_size=0.01,
                        num_random_starts=1, change_point=5)
    
    # Test input - ensure it's in [0, 1] range
    x = Tensor(np.random.rand(4, 784), ms.float32)  # rand generates [0, 1)
    y = Tensor([0, 1, 2, 3], ms.int32)
    
    # Run attack
    x_adv = md_attack.perturb(x, y)
    print(f"Adversarial examples shape: {x_adv.shape}")
    
    # Check epsilon constraint
    diff = ops.abs(x_adv - x)
    max_diff = ops.reduce_max(diff)
    print(f"Max perturbation: {max_diff.asnumpy():.4f} (epsilon: 0.3)")
    assert max_diff <= 0.3 + 1e-6, "Epsilon constraint violated"
    print("✓ MD Attack test passed")

def test_mdmt_attack():
    print("\nTesting MDMT Attack...")
    
    model = SimpleModel()
    mdmt_attack = MDMTAttack(model, epsilon=0.3, num_steps=10, step_size=0.01,
                            num_random_starts=1, change_point=5)
    
    # Test input - ensure it's in [0, 1] range
    x = Tensor(np.random.rand(4, 784), ms.float32)  # rand generates [0, 1)
    y = Tensor([0, 1, 2, 3], ms.int32)
    
    # Run attack
    x_adv = mdmt_attack.perturb(x, y)
    print(f"Adversarial examples shape: {x_adv.shape}")
    
    # Check epsilon constraint
    diff = ops.abs(x_adv - x)
    max_diff = ops.reduce_max(diff)
    print(f"Max perturbation: {max_diff.asnumpy():.4f} (epsilon: 0.3)")
    assert max_diff <= 0.3 + 1e-6, "Epsilon constraint violated"
    print("✓ MDMT Attack test passed")

def test_image_inputs():
    print("\nTesting with image-like inputs...")
    
    model = SimpleModel()
    
    # Test with 4D input (batch, channels, height, width)
    x = Tensor(np.random.rand(4, 1, 28, 28), ms.float32)  # rand generates [0, 1)
    y = Tensor([0, 1, 2, 3], ms.int32)
    
    # Test PGD with image input
    pgd_attack = PGDAttack(model, epsilon=0.3, num_steps=5)
    x_adv = pgd_attack.perturb(x, y)
    assert x_adv.shape == x.shape, "Shape mismatch for image input"
    print("✓ PGD with image input passed")
    
    # Test MD with image input
    md_attack = MDAttack(model, epsilon=0.3, num_steps=5)
    x_adv = md_attack.perturb(x, y)
    assert x_adv.shape == x.shape, "Shape mismatch for image input"
    print("✓ MD with image input passed")

if __name__ == "__main__":
    print("Starting Phase 3 Attack Implementation Tests...")
    print("=" * 50)
    
    # Set context
    ms.set_context(mode=ms.PYNATIVE_MODE)
    
    try:
        test_utils()
        test_loss_functions()
        test_pgd_attack()
        test_mtpgd_attack()
        test_md_attack()
        test_mdmt_attack()
        test_image_inputs()
        
        print("\n" + "=" * 50)
        print("All Phase 3 tests passed successfully! ✓")
        print("The attack implementations are working correctly.")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)