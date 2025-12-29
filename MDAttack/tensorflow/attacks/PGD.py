import time
import sys
import tensorflow as tf
import numpy as np
from .utils import adv_check_and_update, one_hot_tensor


def cw_loss(logits, y):
    """
    Carlini-Wagner loss function.
    
    Args:
        logits: Model predictions (logits)
        y: True labels (indices)
    
    Returns:
        CW loss value
    """
    # Get the correct logit for each sample
    batch_indices = tf.range(tf.shape(logits)[0])
    indices = tf.stack([batch_indices, tf.cast(y, tf.int32)], axis=1)
    correct_logit = tf.reduce_sum(tf.gather_nd(logits, indices))
    
    # Get the second largest logit (or largest if it's not the correct one)
    # Sort logits in descending order and get top 2
    sorted_indices = tf.argsort(logits, axis=1, direction='DESCENDING')[:, :2]
    
    # If the largest is the correct class, use the second largest; otherwise use the largest
    top_class = sorted_indices[:, 0]
    second_class = sorted_indices[:, 1]
    new_y = tf.where(tf.equal(top_class, tf.cast(y, tf.int32)), second_class, top_class)
    
    # Get the wrong logit
    indices = tf.stack([batch_indices, new_y], axis=1)
    wrong_logit = tf.reduce_sum(tf.gather_nd(logits, indices))
    
    # CW loss: -ReLU(correct - wrong)
    loss = -tf.nn.relu(correct_logit - wrong_logit)
    return loss


def margin_loss(logits, y):
    """
    Margin loss function.
    
    Args:
        logits: Model predictions (logits)
        y: True labels (indices)
    
    Returns:
        Margin loss value
    """
    batch_size = tf.shape(logits)[0]
    num_classes = tf.shape(logits)[1]
    
    # Get logit for true class
    batch_indices = tf.range(batch_size)
    indices = tf.stack([batch_indices, tf.cast(y, tf.int32)], axis=1)
    logit_org = tf.gather_nd(logits, indices)
    
    # Create a mask to exclude the true class when finding max
    # Note: The original code assumes 10 classes, but we'll make it more general
    y_one_hot = tf.one_hot(y, num_classes, dtype=tf.float32)
    masked_logits = logits - y_one_hot * 9999
    
    # Get the largest logit excluding the true class
    target_class = tf.argmax(masked_logits, axis=1, output_type=tf.int32)
    indices = tf.stack([batch_indices, target_class], axis=1)
    logit_target = tf.gather_nd(logits, indices)
    
    # Margin loss: -logit_org + logit_target
    loss = -logit_org + logit_target
    loss = tf.reduce_sum(loss)
    return loss


class PGDAttack:
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_restarts=1, v_min=0., v_max=1., num_classes=10,
                 random_start=False, type='PGD', use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.type = type
        self.use_odi = use_odi

    def _get_rand_noise(self, X):
        """Generate random noise for initialization."""
        eps = self.epsilon
        # Generate uniform random noise in [-eps, eps]
        return tf.random.uniform(tf.shape(X), minval=-eps, maxval=eps, dtype=X.dtype)

    def perturb(self, x_in, y_in):
        """
        Perform PGD attack.
        
        Args:
            x_in: Input images
            y_in: True labels
            
        Returns:
            X_adv: Adversarial examples
        """
        model = self.model
        epsilon = self.epsilon
        
        # Initialize adversarial examples
        X_adv = tf.identity(x_in)
        X_pgd = tf.Variable(x_in)
        nc = tf.zeros_like(y_in)

        for r in range(self.num_restarts):
            if self.random_start:
                random_noise = self._get_rand_noise(x_in)
                X_pgd.assign(X_pgd + random_noise)

            if self.use_odi:
                out = model(x_in)
                rv = tf.random.uniform(tf.shape(out), minval=-1., maxval=1., dtype=out.dtype)

            for i in range(self.num_steps):
                # Use GradientTape for gradient computation (exemption #13)
                with tf.GradientTape() as tape:
                    tape.watch(X_pgd)
                    
                    if self.use_odi and i < 2:
                        loss = tf.reduce_sum(model(X_pgd) * rv)
                    elif self.use_odi:
                        loss = margin_loss(model(X_pgd), y_in)
                    elif self.type == 'CW':
                        loss = cw_loss(model(X_pgd), y_in)
                    else:
                        # CrossEntropy loss
                        loss = tf.keras.losses.sparse_categorical_crossentropy(
                            y_in, model(X_pgd), from_logits=True
                        )
                        loss = tf.reduce_mean(loss)
                
                # Get gradients
                gradients = tape.gradient(loss, X_pgd)
                
                # Update step
                if self.use_odi and i < 2:
                    eta = epsilon * tf.sign(gradients)
                else:
                    eta = self.step_size * tf.sign(gradients)
                
                # Update X_pgd
                X_pgd.assign_add(eta)
                
                # Project back to epsilon ball
                eta = tf.clip_by_value(X_pgd - x_in, -epsilon, epsilon)
                X_pgd.assign(x_in + eta)
                
                # Clamp to valid range [0, 1]
                X_pgd.assign(tf.clip_by_value(X_pgd, 0, 1.0))
                
                # Check and update adversarial examples
                logits = self.model(X_pgd)
                X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv


class MTPGDAttack:
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_restarts=1, v_min=0., v_max=1., num_classes=10,
                 random_start=False, use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.random_start = random_start
        self.num_restarts = num_restarts
        self.v_min = v_min
        self.v_max = v_max
        self.num_classes = num_classes
        self.use_odi = use_odi

    def _get_rand_noise(self, X):
        """Generate random noise for initialization."""
        eps = self.epsilon
        # Generate uniform random noise in [-eps, eps]
        return tf.random.uniform(tf.shape(X), minval=-eps, maxval=eps, dtype=X.dtype)

    def perturb(self, x_in, y_in):
        """
        Perform Multi-Targeted PGD attack.
        
        Args:
            x_in: Input images
            y_in: True labels
            
        Returns:
            X_adv: Adversarial examples
        """
        model = self.model
        epsilon = self.epsilon
        
        # Initialize adversarial examples
        X_adv = tf.identity(x_in)
        X_pgd = tf.Variable(x_in)
        nc = tf.zeros_like(y_in)

        # Attack each target class
        for t in range(self.num_classes):
            # One-hot encoding for ground truth and target
            y_gt = one_hot_tensor(y_in, self.num_classes)
            y_tg_indices = tf.zeros_like(y_in) + t
            y_tg = one_hot_tensor(y_tg_indices, self.num_classes)
            
            for r in range(self.num_restarts):
                if self.random_start:
                    random_noise = self._get_rand_noise(x_in)
                    X_pgd.assign(X_pgd + random_noise)

                if self.use_odi:
                    out = model(x_in)
                    rv = tf.random.uniform(tf.shape(out), minval=-1., maxval=1., dtype=out.dtype)

                for i in range(self.num_steps):
                    # Use GradientTape for gradient computation (exemption #13)
                    with tf.GradientTape() as tape:
                        tape.watch(X_pgd)
                        
                        if self.use_odi and i < 2:
                            loss = tf.reduce_sum(model(X_pgd) * rv)
                        else:
                            z = model(X_pgd)
                            z_y = y_gt * z
                            z_t = y_tg * z
                            # Multi-targeted loss: minimize correct class, maximize target class
                            loss = tf.reduce_mean(-tf.reduce_sum(z_y, axis=1) + 
                                                  tf.reduce_sum(z_t, axis=1))
                    
                    # Get gradients
                    gradients = tape.gradient(loss, X_pgd)
                    
                    # Update step
                    if self.use_odi and i < 2:
                        eta = epsilon * tf.sign(gradients)
                    else:
                        eta = self.step_size * tf.sign(gradients)
                    
                    # Update X_pgd
                    X_pgd.assign_add(eta)
                    
                    # Project back to epsilon ball
                    eta = tf.clip_by_value(X_pgd - x_in, -epsilon, epsilon)
                    X_pgd.assign(x_in + eta)
                    
                    # Clamp to valid range [0, 1]
                    X_pgd.assign(tf.clip_by_value(X_pgd, 0, 1.0))
                    
                    # Check and update adversarial examples
                    logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv