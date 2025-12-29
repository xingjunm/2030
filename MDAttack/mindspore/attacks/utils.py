import mindspore as ms
import mindspore.ops as ops
import numpy as np


def adv_check_and_update(X_cur, logits, y, not_correct, X_adv):
    """
    Update adversarial examples based on predictions.
    
    Args:
        X_cur: Current adversarial examples
        logits: Model predictions
        y: True labels
        not_correct: Counter for incorrect predictions
        X_adv: Best adversarial examples found so far
        
    Returns:
        Updated X_adv and not_correct counter
    """
    # Get predicted classes (argmax along dimension 1)
    adv_pred = ops.Argmax(axis=1)(logits)
    
    # Check which predictions are incorrect
    nc = (adv_pred != y)
    
    # Update the not_correct counter
    # Cast boolean to int64 (long in PyTorch)
    # In the original code, += modifies in-place, but we'll return the updated value
    not_correct = not_correct + ops.cast(nc, ms.int64)
    
    # Update adversarial examples where predictions are incorrect
    # Use select operation for conditional assignment
    # Expand nc to match X_cur dimensions for broadcasting
    nc_expanded = ops.expand_dims(nc, axis=1)
    # Repeat the mask for all dimensions of X_cur
    for _ in range(len(X_cur.shape) - 2):
        nc_expanded = ops.expand_dims(nc_expanded, axis=-1)
    
    # Use select operation: where nc is True, take X_cur, otherwise take X_adv
    X_adv = ops.select(nc_expanded, X_cur, X_adv)
    
    return X_adv, not_correct


def one_hot_tensor(y_batch_tensor, num_classes):
    """
    Create one-hot encoded tensor from label tensor.
    
    Args:
        y_batch_tensor: Tensor of class indices
        num_classes: Total number of classes
        
    Returns:
        One-hot encoded tensor
    """
    # Create a tensor filled with zeros
    # Note: MindSpore doesn't require explicit device specification (exemption #3)
    batch_size = y_batch_tensor.shape[0]
    
    # Try using numpy for more robust one-hot encoding
    # to avoid potential issues with OneHot op in certain contexts
    import numpy as np
    y_numpy = y_batch_tensor.asnumpy()
    y_onehot_numpy = np.zeros((batch_size, num_classes), dtype=np.float32)
    y_onehot_numpy[np.arange(batch_size), y_numpy] = 1.0
    
    # Convert back to MindSpore tensor
    y_tensor = ms.Tensor(y_onehot_numpy, ms.float32)
    
    return y_tensor