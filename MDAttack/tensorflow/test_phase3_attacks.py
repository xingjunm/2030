#!/usr/bin/env python3
"""Test script for Phase 3: Core Attack Algorithms"""

import os
import sys
import tensorflow as tf
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import TensorFlow implementations
from tensorflow_impl.attacks.utils import adv_check_and_update, one_hot_tensor
from tensorflow_impl.attacks.MD import MDAttack, MDMTAttack
from tensorflow_impl.attacks.PGD import PGDAttack, MTPGDAttack, cw_loss, margin_loss
from tensorflow_impl.models.wideresnet import WideResNet

def test_utils():
    """Test utility functions"""
    print("\n=== Testing utility functions ===")
    
    # Test one_hot_tensor
    y = tf.constant([0, 1, 2, 3])
    one_hot = one_hot_tensor(y, num_classes=10)
    assert one_hot.shape == (4, 10)
    assert tf.reduce_all(tf.equal(tf.argmax(one_hot, axis=1), tf.cast(y, tf.int64)))
    print("✓ one_hot_tensor working correctly")
    
    # Test adv_check_and_update
    batch_size = 4
    X_cur = tf.random.normal((batch_size, 32, 32, 3))
    X_adv = tf.zeros_like(X_cur)
    logits = tf.random.normal((batch_size, 10))
    y = tf.constant([0, 1, 2, 3])
    not_correct = tf.zeros_like(y, dtype=tf.int64)
    
    X_adv_new, not_correct_new = adv_check_and_update(X_cur, logits, y, not_correct, X_adv)
    assert X_adv_new.shape == X_cur.shape
    assert not_correct_new.shape == y.shape
    print("✓ adv_check_and_update working correctly")

def test_loss_functions():
    """Test loss functions"""
    print("\n=== Testing loss functions ===")
    
    # Test data
    batch_size = 4
    num_classes = 10
    logits = tf.random.normal((batch_size, num_classes))
    y = tf.constant([0, 1, 2, 3])
    
    # Test CW loss
    cw_loss_val = cw_loss(logits, y)
    assert cw_loss_val.shape == ()  # Scalar
    print(f"✓ CW loss: {cw_loss_val.numpy():.4f}")
    
    # Test margin loss
    margin_loss_val = margin_loss(logits, y)
    assert margin_loss_val.shape == ()  # Scalar
    print(f"✓ Margin loss: {margin_loss_val.numpy():.4f}")

def create_simple_model():
    """Create a simple model for testing"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10)
    ])
    return model

def test_pgd_attack():
    """Test PGD Attack"""
    print("\n=== Testing PGD Attack ===")
    
    model = create_simple_model()
    
    # Test standard PGD
    attack = PGDAttack(model, epsilon=8./255., num_steps=10, step_size=2./255.)
    
    # Create test data
    x = tf.random.uniform((4, 32, 32, 3), 0, 1)
    y = tf.constant([0, 1, 2, 3])
    
    # Run attack
    x_adv = attack.perturb(x, y)
    
    # Check output
    assert x_adv.shape == x.shape
    assert tf.reduce_all(tf.abs(x_adv - x) <= attack.epsilon + 1e-6)
    print("✓ PGD Attack working correctly")
    
    # Test with ODI
    attack_odi = PGDAttack(model, epsilon=8./255., num_steps=10, use_odi=True)
    x_adv_odi = attack_odi.perturb(x, y)
    assert x_adv_odi.shape == x.shape
    print("✓ PGD Attack with ODI working correctly")
    
    # Test with CW loss
    attack_cw = PGDAttack(model, epsilon=8./255., num_steps=10, type='CW')
    x_adv_cw = attack_cw.perturb(x, y)
    assert x_adv_cw.shape == x.shape
    print("✓ PGD Attack with CW loss working correctly")

def test_mt_pgd_attack():
    """Test Multi-Targeted PGD Attack"""
    print("\n=== Testing MT-PGD Attack ===")
    
    model = create_simple_model()
    
    # Test MT-PGD
    attack = MTPGDAttack(model, epsilon=8./255., num_steps=5, step_size=2./255., num_classes=10)
    
    # Create test data
    x = tf.random.uniform((2, 32, 32, 3), 0, 1)
    y = tf.constant([0, 1])
    
    # Run attack
    x_adv = attack.perturb(x, y)
    
    # Check output
    assert x_adv.shape == x.shape
    assert tf.reduce_all(tf.abs(x_adv - x) <= attack.epsilon + 1e-6)
    print("✓ MT-PGD Attack working correctly")

def test_md_attack():
    """Test MD Attack"""
    print("\n=== Testing MD Attack ===")
    
    model = create_simple_model()
    
    # Test standard MD
    attack = MDAttack(model, epsilon=8./255., num_steps=10, change_point=5)
    
    # Create test data
    x = tf.random.uniform((4, 32, 32, 3), 0, 1)
    y = tf.constant([0, 1, 2, 3])
    
    # Run attack
    x_adv = attack.perturb(x, y)
    
    # Check output
    assert x_adv.shape == x.shape
    assert tf.reduce_all(tf.abs(x_adv - x) <= attack.epsilon + 1e-6)
    print("✓ MD Attack working correctly")
    
    # Test with ODI
    attack_odi = MDAttack(model, epsilon=8./255., num_steps=10, use_odi=True)
    x_adv_odi = attack_odi.perturb(x, y)
    assert x_adv_odi.shape == x.shape
    print("✓ MD Attack with ODI working correctly")
    
    # Test with DLR loss
    attack_dlr = MDAttack(model, epsilon=8./255., num_steps=10, use_dlr=True)
    x_adv_dlr = attack_dlr.perturb(x, y)
    assert x_adv_dlr.shape == x.shape
    print("✓ MD Attack with DLR loss working correctly")

def test_mdmt_attack():
    """Test MD Multi-Targeted Attack"""
    print("\n=== Testing MDMT Attack ===")
    
    model = create_simple_model()
    
    # Test MDMT
    attack = MDMTAttack(model, epsilon=8./255., num_steps=5, change_point=2, num_classes=10)
    
    # Create test data
    x = tf.random.uniform((2, 32, 32, 3), 0, 1)
    y = tf.constant([0, 1])
    
    # Run attack
    x_adv = attack.perturb(x, y)
    
    # Check output
    assert x_adv.shape == x.shape
    assert tf.reduce_all(tf.abs(x_adv - x) <= attack.epsilon + 1e-6)
    print("✓ MDMT Attack working correctly")

def test_attack_integration():
    """Test attacks with real model"""
    print("\n=== Testing Attack Integration with Wide ResNet ===")
    
    # Create Wide ResNet model
    model = WideResNet(depth=28, widen_factor=10, num_classes=10)
    
    # Build model
    dummy_input = tf.zeros((1, 32, 32, 3))
    _ = model(dummy_input)
    
    # Create test data
    x = tf.random.uniform((2, 32, 32, 3), 0, 1)
    y = tf.constant([0, 1])
    
    # Test different attacks
    attacks = [
        PGDAttack(model, epsilon=8./255., num_steps=5),
        MDAttack(model, epsilon=8./255., num_steps=5, change_point=2),
        MTPGDAttack(model, epsilon=8./255., num_steps=3, num_classes=10),
        MDMTAttack(model, epsilon=8./255., num_steps=3, change_point=1, num_classes=10)
    ]
    
    for i, attack in enumerate(attacks):
        x_adv = attack.perturb(x, y)
        assert x_adv.shape == x.shape
        assert tf.reduce_all(tf.abs(x_adv - x) <= attack.epsilon + 1e-6)
        print(f"✓ Attack {i+1} integrated successfully with Wide ResNet")

def main():
    """Run all tests"""
    print("Starting Phase 3 tests: Core Attack Algorithms")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        test_utils()
        test_loss_functions()
        test_pgd_attack()
        test_mt_pgd_attack()
        test_md_attack()
        test_mdmt_attack()
        test_attack_integration()
        
        print("\n" + "=" * 60)
        print("✅ All Phase 3 tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()