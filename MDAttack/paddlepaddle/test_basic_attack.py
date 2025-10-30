"""Basic test to verify PaddlePaddle attack implementation works"""
import paddle
import numpy as np
from attacks.MD import MDAttack
from models.wideresnet import WideResNet

# Create a simple dummy model for testing
class DummyModel:
    def __init__(self):
        self.num_classes = 10
        
    def __call__(self, x):
        # Simple dummy output
        batch_size = x.shape[0]
        # Return a single tensor for attacks, but defense models return list
        return paddle.randn([batch_size, self.num_classes])
    
    def eval(self):
        return self

def test_md_attack():
    print("Testing MD Attack implementation...")
    
    # Create dummy model
    model = DummyModel()
    
    # Create some dummy data
    batch_size = 10
    x = paddle.randn([batch_size, 3, 32, 32])
    y = paddle.randint(0, 10, [batch_size])
    
    # Create MDAttack instance
    attacker = MDAttack(model, epsilon=8/255, num_steps=20, step_size=2/255, 
                       num_random_starts=1)
    
    # Run attack
    print("Running MD attack...")
    x_adv = attacker.perturb(x, y)
    
    # Check output
    print(f"Original shape: {x.shape}")
    print(f"Adversarial shape: {x_adv.shape}")
    print(f"Perturbation L_inf: {paddle.max(paddle.abs(x_adv - x)).item():.4f}")
    print(f"Expected max perturbation: {8/255:.4f}")
    
    # Verify perturbation is within epsilon
    assert paddle.max(paddle.abs(x_adv - x)) <= 8/255 + 1e-6
    print("✓ Perturbation within epsilon bound")
    
    # Test that the shapes match
    assert x.shape == x_adv.shape
    print("✓ Output shape matches input shape")
    
    print("\nMD Attack test passed!")

if __name__ == "__main__":
    test_md_attack()