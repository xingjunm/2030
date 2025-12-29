#!/usr/bin/env python3
"""Test script for Phase 2: Model Architecture migration to TensorFlow"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.wideresnet import WideResNet, BasicBlock, NetworkBlock


def test_basic_block():
    """Test BasicBlock functionality"""
    print("Testing BasicBlock...")
    
    # Create a BasicBlock
    block = BasicBlock(in_planes=16, out_planes=32, stride=2, dropRate=0.3)
    
    # Create dummy input
    x = tf.random.normal((2, 32, 32, 16))  # [batch, height, width, channels]
    
    # Forward pass
    output = block(x, training=True)
    
    # Check output shape
    expected_shape = (2, 16, 16, 32)  # stride=2 reduces spatial dimensions
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"✓ BasicBlock output shape: {output.shape}")
    
    # Test with equal in/out planes
    block_equal = BasicBlock(in_planes=32, out_planes=32, stride=1)
    x_equal = tf.random.normal((2, 32, 32, 32))
    output_equal = block_equal(x_equal, training=False)
    assert output_equal.shape == (2, 32, 32, 32)
    print("✓ BasicBlock with equal in/out planes works correctly")


def test_network_block():
    """Test NetworkBlock functionality"""
    print("\nTesting NetworkBlock...")
    
    # Create a NetworkBlock
    net_block = NetworkBlock(
        nb_layers=3,
        in_planes=16,
        out_planes=32,
        block=BasicBlock,
        stride=2,
        dropRate=0.0
    )
    
    # Create dummy input
    x = tf.random.normal((2, 32, 32, 16))
    
    # Forward pass
    output = net_block(x, training=True)
    
    # Check output shape
    expected_shape = (2, 16, 16, 32)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    
    print(f"✓ NetworkBlock output shape: {output.shape}")


def test_wideresnet():
    """Test WideResNet functionality"""
    print("\nTesting WideResNet...")
    
    # Test different configurations
    configs = [
        (16, 10, 1),   # depth=16, num_classes=10, widen_factor=1
        (28, 100, 10), # depth=28, num_classes=100, widen_factor=10
        (40, 10, 4),   # depth=40, num_classes=10, widen_factor=4
    ]
    
    for depth, num_classes, widen_factor in configs:
        print(f"\n  Testing WRN-{depth}-{widen_factor} with {num_classes} classes...")
        
        # Create model
        model = WideResNet(
            depth=depth,
            num_classes=num_classes,
            widen_factor=widen_factor,
            dropRate=0.3
        )
        
        # Create dummy input (32x32 images, 3 channels)
        x = tf.random.normal((4, 32, 32, 3))
        
        # Forward pass
        output = model(x, training=True)
        
        # Check output shape
        expected_shape = (4, num_classes)
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        
        print(f"  ✓ Output shape: {output.shape}")
        
        # Test with latent mode
        model.latent = True
        features = model(x, training=False)
        assert isinstance(features, list), "Expected list of features in latent mode"
        assert len(features) > 0, "Expected non-empty features list"
        print(f"  ✓ Latent mode returns {len(features)} features")
        
        # Reset latent mode
        model.latent = False


def test_model_building():
    """Test that model builds correctly"""
    print("\nTesting model building and weight initialization...")
    
    # Create model
    model = WideResNet(depth=16, num_classes=10, widen_factor=1)
    
    # Build the model by calling it once
    x = tf.random.normal((1, 32, 32, 3))
    _ = model(x, training=False)
    
    # Check that all layers are built
    total_params = model.count_params()
    print(f"✓ Model built successfully with {total_params:,} parameters")
    
    # Check specific layer types
    conv_count = 0
    bn_count = 0
    dense_count = 0
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_count += 1
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            bn_count += 1
        elif isinstance(layer, tf.keras.layers.Dense):
            dense_count += 1
    
    print(f"✓ Found {conv_count} Conv2D layers")
    print(f"✓ Found {bn_count} BatchNormalization layers")
    print(f"✓ Found {dense_count} Dense layers")


def test_gradient_flow():
    """Test that gradients flow correctly through the model"""
    print("\nTesting gradient flow...")
    
    # Create model
    model = WideResNet(depth=16, num_classes=10, widen_factor=1)
    
    # Create dummy data
    x = tf.random.normal((2, 32, 32, 3))
    y_true = tf.one_hot([0, 5], depth=10)
    
    # Compute loss and gradients
    with tf.GradientTape() as tape:
        y_pred = model(x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        loss = tf.reduce_mean(loss)
    
    # Get gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check that gradients exist and are not None
    assert all(g is not None for g in gradients), "Some gradients are None"
    
    # Check that gradients have reasonable magnitudes
    grad_norms = [tf.norm(g).numpy() for g in gradients]
    
    # Print actual gradient norm range for debugging
    print(f"  Gradient norm range: [{min(grad_norms):.4f}, {max(grad_norms):.4f}]")
    
    # Filter out zero gradients (can happen with batch norm parameters)
    non_zero_norms = [norm for norm in grad_norms if norm > 0]
    
    # Check that we have at least some non-zero gradients
    assert len(non_zero_norms) > 0, "All gradients are zero"
    
    # Check that non-zero gradients have reasonable magnitudes
    assert all(0 < norm < 10000 for norm in non_zero_norms), f"Gradient magnitudes out of reasonable range: [{min(non_zero_norms)}, {max(non_zero_norms)}]"
    
    print(f"✓ Gradients flow correctly (loss: {loss.numpy():.4f})")
    print(f"✓ {len(non_zero_norms)}/{len(grad_norms)} gradients are non-zero with reasonable magnitudes")


def test_defense_compatibility():
    """Test that defense models can be loaded (they remain PyTorch)"""
    print("\nTesting defense module compatibility...")
    
    # Check that defense directory exists
    defense_path = os.path.join(os.path.dirname(__file__), 'defense')
    assert os.path.exists(defense_path), f"Defense directory not found at {defense_path}"
    
    # Check key defense files
    expected_files = ['__init__.py', 'utils.py', 'TRADES.py', 'MART.py', 'AWP.py']
    for file in expected_files:
        file_path = os.path.join(defense_path, file)
        assert os.path.exists(file_path), f"Defense file {file} not found"
    
    print("✓ All defense files present (keeping PyTorch implementation)")


def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 2: Model Architecture Tests")
    print("=" * 60)
    
    # Disable GPU if not available
    if not tf.config.list_physical_devices('GPU'):
        print("Note: Running on CPU")
    
    try:
        test_basic_block()
        test_network_block()
        test_wideresnet()
        test_model_building()
        test_gradient_flow()
        test_defense_compatibility()
        
        print("\n" + "=" * 60)
        print("✅ All Phase 2 tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()