#!/usr/bin/env python3
"""Test script for Phase 3: Core Attack Algorithms"""

import paddle
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the translated modules
from attacks.utils import adv_check_and_update, one_hot_tensor
from attacks.MD import MDAttack, MDMTAttack
from attacks.PGD import PGDAttack, MTPGDAttack

def test_utils():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test adv_check_and_update
    batch_size = 4
    num_classes = 10
    
    # Create test data
    logits = paddle.randn([batch_size, num_classes])
    y = paddle.randint(0, num_classes, [batch_size])
    X_cur = paddle.randn([batch_size, 3, 32, 32])
    not_correct = paddle.zeros([batch_size], dtype='int64')
    X_adv = paddle.zeros_like(X_cur)
    
    # Test function
    X_adv_new, not_correct_new = adv_check_and_update(X_cur, logits, y, not_correct, X_adv)
    print(f"  ✓ adv_check_and_update: X_adv shape = {X_adv_new.shape}")
    
    # Test one_hot_tensor
    y_one_hot = one_hot_tensor(y, num_classes)
    print(f"  ✓ one_hot_tensor: shape = {y_one_hot.shape}, sum per row = {y_one_hot.sum(axis=1)[:2]}")
    
    print("Utility functions test passed!\n")

def test_md_attack():
    """Test MD attack implementations"""
    print("Testing MD Attack...")
    
    # Create a simple model for testing
    class SimpleModel(paddle.nn.Layer):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = paddle.nn.Conv2D(3, 16, 3, padding=1)
            self.pool = paddle.nn.AdaptiveAvgPool2D(1)
            self.fc = paddle.nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = paddle.flatten(x, 1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Test MDAttack
    print("  Testing MDAttack...")
    md_attack = MDAttack(
        model=model,
        epsilon=8/255,
        num_steps=10,
        use_odi=False
    )
    
    # Create test data with values in [0, 1]
    images = paddle.rand([2, 3, 32, 32])  # Use rand to get [0, 1] values
    labels = paddle.randint(0, 10, [2])
    
    # Run attack
    adv_images = md_attack.perturb(images, labels)
    perturbation = (adv_images - images).abs()
    max_pert = perturbation.max()
    
    print(f"    ✓ MDAttack output shape: {adv_images.shape}")
    print(f"    ✓ Max perturbation: {max_pert:.4f} (eps={8/255:.4f})")
    assert max_pert <= 8/255 + 1e-4, "Perturbation exceeds epsilon"  # Allow small tolerance for floating point
    
    # Test MDMTAttack
    print("  Testing MDMTAttack...")
    mdmt_attack = MDMTAttack(
        model=model,
        epsilon=8/255,
        num_steps=10,
        use_odi=False
    )
    
    adv_images_mt = mdmt_attack.perturb(images, labels)
    perturbation_mt = (adv_images_mt - images).abs()
    max_pert_mt = perturbation_mt.max()
    
    print(f"    ✓ MDMTAttack output shape: {adv_images_mt.shape}")
    print(f"    ✓ Max perturbation: {max_pert_mt:.4f} (eps={8/255:.4f})")
    assert max_pert_mt <= 8/255 + 1e-4, "Perturbation exceeds epsilon"  # Allow small tolerance
    
    print("MD Attack test passed!\n")

def test_pgd_attack():
    """Test PGD attack implementations"""
    print("Testing PGD Attack...")
    
    # Create a simple model for testing
    class SimpleModel(paddle.nn.Layer):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = paddle.nn.Conv2D(3, 16, 3, padding=1)
            self.pool = paddle.nn.AdaptiveAvgPool2D(1)
            self.fc = paddle.nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = paddle.flatten(x, 1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    # Test PGDAttack
    print("  Testing PGDAttack...")
    pgd_attack = PGDAttack(
        model=model,
        epsilon=8/255,
        num_steps=10,
        use_odi=False
    )
    
    # Create test data with values in [0, 1]
    images = paddle.rand([2, 3, 32, 32])  # Use rand to get [0, 1] values
    labels = paddle.randint(0, 10, [2])
    
    # Run attack
    adv_images = pgd_attack.perturb(images, labels)
    perturbation = (adv_images - images).abs()
    max_pert = perturbation.max()
    
    print(f"    ✓ PGDAttack output shape: {adv_images.shape}")
    print(f"    ✓ Max perturbation: {max_pert:.4f} (eps={8/255:.4f})")
    assert max_pert <= 8/255 + 1e-4, "Perturbation exceeds epsilon"  # Allow small tolerance
    
    # Test MTPGDAttack
    print("  Testing MTPGDAttack...")
    mtpgd_attack = MTPGDAttack(
        model=model,
        epsilon=8/255,
        num_steps=10
    )
    
    adv_images_mt = mtpgd_attack.perturb(images, labels)
    perturbation_mt = (adv_images_mt - images).abs()
    max_pert_mt = perturbation_mt.max()
    
    print(f"    ✓ MTPGDAttack output shape: {adv_images_mt.shape}")
    print(f"    ✓ Max perturbation: {max_pert_mt:.4f} (eps={8/255:.4f})")
    assert max_pert_mt <= 8/255 + 1e-4, "Perturbation exceeds epsilon"  # Allow small tolerance
    
    print("PGD Attack test passed!\n")

def test_attack_variants():
    """Test different attack variants and loss functions"""
    print("Testing attack variants...")
    
    class SimpleModel(paddle.nn.Layer):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv = paddle.nn.Conv2D(3, 16, 3, padding=1)
            self.pool = paddle.nn.AdaptiveAvgPool2D(1)
            self.fc = paddle.nn.Linear(16, num_classes)
        
        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = paddle.flatten(x, 1)
            x = self.fc(x)
            return x
    
    model = SimpleModel()
    model.eval()
    
    images = paddle.rand([2, 3, 32, 32])  # Use rand to get [0, 1] values
    labels = paddle.randint(0, 10, [2])
    
    # Test MD with ODI
    print("  Testing MD with ODI...")
    md_odi = MDAttack(model, epsilon=8/255, num_steps=5, use_odi=True)
    adv_odi = md_odi.perturb(images, labels)
    print(f"    ✓ MD-ODI output shape: {adv_odi.shape}")
    
    # Test MD with DLR loss
    print("  Testing MD with DLR loss...")
    md_dlr = MDAttack(model, epsilon=8/255, num_steps=5, use_dlr=True)
    adv_dlr = md_dlr.perturb(images, labels)
    print(f"    ✓ MD-DLR output shape: {adv_dlr.shape}")
    
    # Test MDMT with ODI
    print("  Testing MDMT with ODI...")
    mdmt_odi = MDMTAttack(model, epsilon=8/255, num_steps=5, use_odi=True)
    adv_mdmt_odi = mdmt_odi.perturb(images, labels)
    print(f"    ✓ MDMT-ODI output shape: {adv_mdmt_odi.shape}")
    
    # Test PGD with ODI
    print("  Testing PGD with ODI...")
    pgd_odi = PGDAttack(model, epsilon=8/255, num_steps=5, use_odi=True)
    adv_odi_pgd = pgd_odi.perturb(images, labels)
    print(f"    ✓ PGD-ODI output shape: {adv_odi_pgd.shape}")
    
    print("Attack variants test passed!\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 3 Testing: Core Attack Algorithms")
    print("=" * 60)
    print()
    
    try:
        test_utils()
        test_md_attack()
        test_pgd_attack()
        test_attack_variants()
        
        print("=" * 60)
        print("✅ All Phase 3 tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())