#!/usr/bin/env python
"""Test script to verify the integration between PyTorch defense models and PaddlePaddle attacks."""

import sys
import paddle
import torch
import numpy as np

# Test the PyTorchToPaddleModel wrapper
sys.path.insert(0, '/root/MDAttack/paddlepaddle_impl')
from main import PyTorchToPaddleModel

def test_model_wrapper():
    """Test the PyTorchToPaddleModel wrapper functionality."""
    print("Testing PyTorchToPaddleModel wrapper...")
    
    # Create a simple PyTorch model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 5)
            
        def forward(self, x):
            return self.linear(x)
    
    # Create PyTorch model
    pytorch_model = SimpleModel()
    pytorch_model.eval()
    
    # Wrap with PyTorchToPaddleModel
    wrapped_model = PyTorchToPaddleModel(pytorch_model)
    
    # Test with PaddlePaddle tensor input
    paddle_input = paddle.randn([4, 10])
    print(f"Input shape (PaddlePaddle): {paddle_input.shape}")
    
    # Get output
    output = wrapped_model(paddle_input)
    print(f"Output shape (PaddlePaddle): {output.shape}")
    print(f"Output type: {type(output)}")
    
    assert isinstance(output, paddle.Tensor), "Output should be a PaddlePaddle tensor"
    assert output.shape == [4, 5], f"Expected shape [4, 5], got {output.shape}"
    
    print("✓ Model wrapper test passed!\n")
    
def test_model_with_list_output():
    """Test wrapper with models that return list of outputs."""
    print("Testing model with list outputs...")
    
    # Create a PyTorch model that returns multiple outputs
    class MultiOutputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(10, 5)
            self.linear2 = torch.nn.Linear(10, 3)
            
        def forward(self, x):
            return [self.linear1(x), self.linear2(x)]
    
    # Create PyTorch model
    pytorch_model = MultiOutputModel()
    pytorch_model.eval()
    
    # Wrap with PyTorchToPaddleModel
    wrapped_model = PyTorchToPaddleModel(pytorch_model)
    
    # Test with PaddlePaddle tensor input
    paddle_input = paddle.randn([4, 10])
    
    # Get output
    outputs = wrapped_model(paddle_input)
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output 1 shape: {outputs[0].shape}")
    print(f"Output 2 shape: {outputs[1].shape}")
    
    assert isinstance(outputs, list), "Output should be a list"
    assert len(outputs) == 2, f"Expected 2 outputs, got {len(outputs)}"
    assert all(isinstance(o, paddle.Tensor) for o in outputs), "All outputs should be PaddlePaddle tensors"
    assert outputs[0].shape == [4, 5], f"Expected shape [4, 5] for output 1, got {outputs[0].shape}"
    assert outputs[1].shape == [4, 3], f"Expected shape [4, 3] for output 2, got {outputs[1].shape}"
    
    print("✓ Multi-output model test passed!\n")

def test_data_conversion():
    """Test data conversion between frameworks."""
    print("Testing data conversion...")
    
    # Test numpy to paddle
    np_array = np.random.randn(2, 3).astype(np.float32)
    paddle_tensor = paddle.to_tensor(np_array)
    print(f"Numpy to Paddle: {np_array.shape} -> {paddle_tensor.shape}")
    
    # Test paddle to numpy
    np_from_paddle = paddle_tensor.numpy()
    print(f"Paddle to Numpy: {paddle_tensor.shape} -> {np_from_paddle.shape}")
    
    # Test paddle to torch (via numpy)
    torch_tensor = torch.from_numpy(paddle_tensor.numpy())
    print(f"Paddle to Torch: {paddle_tensor.shape} -> {torch_tensor.shape}")
    
    # Verify values are preserved
    assert np.allclose(np_array, np_from_paddle), "Values should be preserved in conversion"
    assert np.allclose(np_array, torch_tensor.numpy()), "Values should be preserved in conversion"
    
    print("✓ Data conversion test passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing PyTorch-PaddlePaddle Integration")
    print("=" * 60 + "\n")
    
    test_data_conversion()
    test_model_wrapper()
    test_model_with_list_output()
    
    print("=" * 60)
    print("All integration tests passed successfully!")
    print("=" * 60)