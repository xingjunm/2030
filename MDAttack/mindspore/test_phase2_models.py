import mindspore as ms
import mindspore.numpy as mnp
from mindspore import context

# Set context
context.set_context(mode=context.PYNATIVE_MODE)

# Import the converted model
from models.wideresnet import WideResNet

def test_wideresnet():
    print("Testing WideResNet model...")
    
    # Test model instantiation
    model = WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3)
    print("✓ Model created successfully")
    
    # Test forward pass
    batch_size = 4
    dummy_input = ms.Tensor(mnp.randn(batch_size, 3, 32, 32), dtype=ms.float32)
    
    # Set model to eval mode
    model.set_train(False)
    
    # Forward pass
    output = model(dummy_input)
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    
    # Check output shape
    assert output.shape == (batch_size, 10), f"Expected shape ({batch_size}, 10), got {output.shape}"
    print("✓ Output shape correct")
    
    # Test with latent features
    model.latent = True
    features = model(dummy_input)
    print(f"✓ Latent features mode successful, returned {len(features)} features")
    
    # Reset latent mode
    model.latent = False
    
    # Test training mode
    model.set_train(True)
    output_train = model(dummy_input)
    print(f"✓ Training mode forward pass successful")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_wideresnet()