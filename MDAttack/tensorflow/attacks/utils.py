import tensorflow as tf
import numpy as np


def adv_check_and_update(X_cur, logits, y, not_correct, X_adv):
    """
    Updates adversarial examples based on predictions.
    
    Args:
        X_cur: Current adversarial examples
        logits: Model predictions (logits)
        y: True labels
        not_correct: Counter for incorrect predictions
        X_adv: Best adversarial examples found so far
    
    Returns:
        X_adv: Updated adversarial examples
        not_correct: Updated counter
    """
    # Get predicted classes (argmax along last dimension)
    adv_pred = tf.argmax(logits, axis=1)
    
    # Check which predictions are incorrect
    # Cast to same dtype to avoid mismatch
    nc = tf.not_equal(adv_pred, tf.cast(y, adv_pred.dtype))
    
    # Update the counter (convert boolean to int)
    # Cast to the same dtype as not_correct
    not_correct = not_correct + tf.cast(nc, not_correct.dtype)
    
    # Update X_adv where predictions are incorrect
    # In TensorFlow, we use tf.where for conditional assignment
    # Expand nc to match X_cur dimensions for broadcasting
    nc_expanded = tf.expand_dims(nc, axis=-1)
    if len(X_cur.shape) > 2:
        for _ in range(len(X_cur.shape) - 2):
            nc_expanded = tf.expand_dims(nc_expanded, axis=-1)
    
    # Use tf.where to conditionally update X_adv
    X_adv = tf.where(nc_expanded, X_cur, X_adv)
    
    return X_adv, not_correct


def one_hot_tensor(y_batch_tensor, num_classes):
    """
    Creates one-hot encoded tensors.
    
    Args:
        y_batch_tensor: Tensor of class indices
        num_classes: Number of classes
    
    Returns:
        One-hot encoded tensor
    """
    # TensorFlow has built-in one_hot function
    # Note: PyTorch version used FloatTensor on CUDA, but per exemption #3,
    # we don't need to explicitly specify device in TensorFlow
    y_tensor = tf.one_hot(y_batch_tensor, num_classes, dtype=tf.float32)
    
    return y_tensor