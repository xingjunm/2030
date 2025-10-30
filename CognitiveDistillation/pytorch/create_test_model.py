#!/usr/bin/env python3
"""Create a test model for TensorFlow implementation testing"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tensorflow_impl'))

import tensorflow as tf
from models.resnet import ResNet18
import numpy as np

# Create and save a test model
model = ResNet18(num_classes=10)

# Build the model with dummy input
dummy_input = tf.random.normal([1, 3, 32, 32])
_ = model(dummy_input)

# Create checkpoint directory
os.makedirs('tensorflow_impl/experiments/readme_test/test_quick/checkpoints', exist_ok=True)

# Save in TensorFlow format (.weights.h5 format for Keras)
model.save_weights('tensorflow_impl/experiments/readme_test/test_quick/checkpoints/model_state_dict.weights.h5')
print("✓ Test model saved")

# Also save dummy optimizer and scheduler states
optimizer = tf.optimizers.SGD(learning_rate=0.1)
optimizer.iterations.assign(1000)

# Save optimizer state
checkpoint = tf.train.Checkpoint(optimizer=optimizer)
checkpoint.save('tensorflow_impl/experiments/readme_test/test_quick/checkpoints/optimizer_state_dict')
print("✓ Optimizer state saved")

# Create a dummy scheduler state
scheduler_state = {'milestones': [45, 60], 'gamma': 0.1, 'last_epoch': 1}
np.save('tensorflow_impl/experiments/readme_test/test_quick/checkpoints/scheduler_state_dict.npy', scheduler_state)
print("✓ Scheduler state saved")

# Copy config file
import shutil
os.makedirs('tensorflow_impl/experiments/readme_test/test_quick', exist_ok=True)
shutil.copy('configs/test_tf/test_quick.yaml', 
            'tensorflow_impl/experiments/readme_test/test_quick/test_quick.yaml')
print("✓ Config copied")

print("\n✅ Test model and experiment setup complete!")