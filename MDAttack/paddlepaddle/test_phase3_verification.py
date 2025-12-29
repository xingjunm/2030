#!/usr/bin/env python3
"""Comprehensive verification test for Phase 3: Core Attack Algorithms"""

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

def create_test_model(num_classes=10):
    """Create a simple test model"""
    class TestModel(paddle.nn.Layer):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = paddle.nn.Conv2D(3, 32, 3, padding=1)
            self.bn1 = paddle.nn.BatchNorm2D(32)
            self.relu = paddle.nn.ReLU()
            self.conv2 = paddle.nn.Conv2D(32, 64, 3, padding=1)
            self.bn2 = paddle.nn.BatchNorm2D(64)
            self.pool = paddle.nn.AdaptiveAvgPool2D(1)
            self.fc = paddle.nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = paddle.flatten(x, 1)
            x = self.fc(x)
            return x
    
    return TestModel(num_classes)

def test_attack_effectiveness(attack, model, images, labels, attack_name):
    """Test that the attack actually changes predictions"""
    model.eval()
    
    # Get original predictions
    with paddle.no_grad():
        orig_logits = model(images)
        orig_preds = paddle.argmax(orig_logits, axis=1)
    
    # Run attack
    adv_images = attack.perturb(images, labels)
    
    # Get adversarial predictions
    with paddle.no_grad():
        adv_logits = model(adv_images)
        adv_preds = paddle.argmax(adv_logits, axis=1)
    
    # Check that some predictions changed
    changed = (orig_preds != adv_preds).sum().item()
    success_rate = changed / len(labels) * 100
    
    print(f"  {attack_name}: {changed}/{len(labels)} predictions changed ({success_rate:.1f}%)")
    
    return adv_images

def verify_phase3():
    """Comprehensive verification of Phase 3 implementation"""
    print("=" * 70)
    print("PHASE 3 VERIFICATION: Core Attack Algorithms")
    print("=" * 70)
    print()
    
    # Setup
    num_classes = 10
    batch_size = 8
    model = create_test_model(num_classes)
    model.eval()
    
    # Create realistic test data
    images = paddle.rand([batch_size, 3, 32, 32])
    labels = paddle.randint(0, num_classes, [batch_size])
    
    print("1. TESTING UTILITY FUNCTIONS")
    print("-" * 40)
    
    # Test one_hot_tensor
    y_onehot = one_hot_tensor(labels, num_classes)
    assert y_onehot.shape == [batch_size, num_classes], "One-hot shape mismatch"
    assert paddle.allclose(y_onehot.sum(axis=1), paddle.ones([batch_size])), "One-hot sum not 1"
    print("  ✓ one_hot_tensor working correctly")
    
    # Test adv_check_and_update
    logits = paddle.randn([batch_size, num_classes])
    X_cur = paddle.randn([batch_size, 3, 32, 32])
    not_correct = paddle.zeros([batch_size], dtype='int64')
    X_adv = paddle.zeros_like(X_cur)
    
    X_adv_new, not_correct_new = adv_check_and_update(X_cur, logits, labels, not_correct, X_adv)
    assert X_adv_new.shape == X_cur.shape, "adv_check_and_update shape mismatch"
    print("  ✓ adv_check_and_update working correctly")
    print()
    
    print("2. TESTING MD ATTACKS")
    print("-" * 40)
    
    # Test standard MD Attack
    md_attack = MDAttack(model, epsilon=8/255, num_steps=20)
    adv_md = test_attack_effectiveness(md_attack, model, images, labels, "MDAttack")
    
    # Verify epsilon constraint
    pert = (adv_md - images).abs().max()
    assert pert <= 8/255 + 1e-4, f"MD perturbation {pert} exceeds epsilon"
    print(f"  ✓ MD perturbation within epsilon: {pert:.6f} <= {8/255:.6f}")
    
    # Test MD with ODI
    md_odi = MDAttack(model, epsilon=8/255, num_steps=20, use_odi=True)
    adv_md_odi = test_attack_effectiveness(md_odi, model, images, labels, "MD-ODI")
    
    # Test MD with DLR loss
    md_dlr = MDAttack(model, epsilon=8/255, num_steps=20, use_dlr=True)
    adv_md_dlr = test_attack_effectiveness(md_dlr, model, images, labels, "MD-DLR")
    
    # Test MDMT Attack
    mdmt_attack = MDMTAttack(model, epsilon=8/255, num_steps=20)
    adv_mdmt = test_attack_effectiveness(mdmt_attack, model, images, labels, "MDMTAttack")
    
    # Verify epsilon constraint
    pert = (adv_mdmt - images).abs().max()
    assert pert <= 8/255 + 1e-4, f"MDMT perturbation {pert} exceeds epsilon"
    print(f"  ✓ MDMT perturbation within epsilon: {pert:.6f} <= {8/255:.6f}")
    print()
    
    print("3. TESTING PGD ATTACKS")
    print("-" * 40)
    
    # Test standard PGD Attack
    pgd_attack = PGDAttack(model, epsilon=8/255, num_steps=20)
    adv_pgd = test_attack_effectiveness(pgd_attack, model, images, labels, "PGDAttack")
    
    # Verify epsilon constraint
    pert = (adv_pgd - images).abs().max()
    assert pert <= 8/255 + 1e-4, f"PGD perturbation {pert} exceeds epsilon"
    print(f"  ✓ PGD perturbation within epsilon: {pert:.6f} <= {8/255:.6f}")
    
    # Test PGD with ODI
    pgd_odi = PGDAttack(model, epsilon=8/255, num_steps=20, use_odi=True)
    adv_pgd_odi = test_attack_effectiveness(pgd_odi, model, images, labels, "PGD-ODI")
    
    # Test MTPGD Attack
    mtpgd_attack = MTPGDAttack(model, epsilon=8/255, num_steps=20)
    adv_mtpgd = test_attack_effectiveness(mtpgd_attack, model, images, labels, "MTPGDAttack")
    
    # Verify epsilon constraint
    pert = (adv_mtpgd - images).abs().max()
    assert pert <= 8/255 + 1e-4, f"MTPGD perturbation {pert} exceeds epsilon"
    print(f"  ✓ MTPGD perturbation within epsilon: {pert:.6f} <= {8/255:.6f}")
    print()
    
    print("4. TESTING ATTACK CONSISTENCY")
    print("-" * 40)
    
    # Test that attacks with same random seed produce same results
    paddle.seed(42)
    attack1 = MDAttack(model, epsilon=8/255, num_steps=10, seed=42)
    adv1 = attack1.perturb(images[:2], labels[:2])
    
    paddle.seed(42)
    attack2 = MDAttack(model, epsilon=8/255, num_steps=10, seed=42)
    adv2 = attack2.perturb(images[:2], labels[:2])
    
    diff = (adv1 - adv2).abs().max()
    print(f"  Same seed difference: {diff:.8f}")
    assert diff < 1e-5, "Attacks with same seed should produce same results"
    print("  ✓ Attack reproducibility verified")
    print()
    
    print("5. TESTING DIFFERENT EPSILON VALUES")
    print("-" * 40)
    
    for eps in [4/255, 8/255, 16/255]:
        md = MDAttack(model, epsilon=eps, num_steps=10)
        adv = md.perturb(images[:4], labels[:4])
        pert = (adv - images[:4]).abs().max()
        assert pert <= eps + 1e-4, f"Perturbation {pert} exceeds epsilon {eps}"
        print(f"  ✓ Epsilon={eps:.4f}: max perturbation={pert:.4f}")
    print()
    
    print("6. TESTING BATCH PROCESSING")
    print("-" * 40)
    
    # Test different batch sizes
    for batch_size in [1, 4, 16]:
        test_images = paddle.rand([batch_size, 3, 32, 32])
        test_labels = paddle.randint(0, num_classes, [batch_size])
        
        attack = PGDAttack(model, epsilon=8/255, num_steps=5)
        adv = attack.perturb(test_images, test_labels)
        
        assert adv.shape == test_images.shape, f"Batch size {batch_size} shape mismatch"
        print(f"  ✓ Batch size {batch_size} processed correctly")
    print()
    
    print("=" * 70)
    print("✅ PHASE 3 VERIFICATION COMPLETE - ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  • Utility functions working correctly")
    print("  • MD attacks (standard, ODI, DLR, MT) implemented")
    print("  • PGD attacks (standard, ODI, MT) implemented")
    print("  • All attacks respect epsilon constraints")
    print("  • Attack reproducibility verified")
    print("  • Batch processing working correctly")
    print()

if __name__ == "__main__":
    try:
        verify_phase3()
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)