import time
import sys
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.ops import GradOperation
from .utils import adv_check_and_update, one_hot_tensor


def cw_loss(logits, y):
    """
    Carlini-Wagner loss function.
    
    Args:
        logits: Model output logits
        y: True labels
        
    Returns:
        CW loss value
    """
    # Get the correct logit for each sample
    y_unsqueeze = ops.expand_dims(y, axis=1).astype(ms.int64)
    correct_logit = ops.reduce_sum(ops.gather_elements(logits, 1, y_unsqueeze).squeeze(axis=1))
    
    # Get the indices of top-2 logits
    _, tmp1 = ops.TopK(sorted=True)(logits, 2)
    
    # Get the second-best class if the best is the true class, otherwise get the best
    # Check if the highest logit corresponds to the true label
    mask = (tmp1[:, 0] == y)
    new_y = ops.select(mask, tmp1[:, 1], tmp1[:, 0])
    
    # Get the wrong logit
    new_y_unsqueeze = ops.expand_dims(new_y, axis=1).astype(ms.int64)
    wrong_logit = ops.reduce_sum(ops.gather_elements(logits, 1, new_y_unsqueeze).squeeze(axis=1))
    
    # Compute loss with ReLU
    loss = -ops.relu(correct_logit - wrong_logit)
    return loss


def margin_loss(logits, y):
    """
    Margin-based loss function.
    
    Args:
        logits: Model output logits
        y: True labels
        
    Returns:
        Margin loss value
    """
    # Get the logit for the true class
    y_view = y.view(-1, 1).astype(ms.int64)
    logit_org = ops.gather_elements(logits, 1, y_view)
    
    # Create a mask to exclude the true class when finding max
    # Create one-hot encoding and multiply by large value to mask
    one_hot_op = ops.OneHot(axis=-1)
    on_value = ms.Tensor(9999.0, ms.float32)
    off_value = ms.Tensor(0.0, ms.float32)
    mask = one_hot_op(y, 10, on_value, off_value)
    
    # Get the highest logit excluding the true class
    masked_logits = logits - mask
    argmax_op = ops.Argmax(axis=1)
    argmax_indices = argmax_op(masked_logits)
    argmax_indices = ops.expand_dims(argmax_indices, 1)
    logit_target = ops.gather_elements(logits, 1, argmax_indices)
    
    # Compute loss
    loss = -logit_org + logit_target
    loss = ops.reduce_sum(loss)
    return loss


class PGDAttack():
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
        
        # Create gradient operation for computing gradients
        self.grad_op = GradOperation(get_all=False, get_by_list=False, sens_param=True)

    def _get_rand_noise(self, X):
        """Generate random noise for initialization."""
        eps = self.epsilon
        # Create uniform random noise in [-eps, eps]
        # MindSpore doesn't need device specification (exemption #3)
        return ops.uniform(X.shape, Tensor(-eps, ms.float32), Tensor(eps, ms.float32), dtype=ms.float32)

    def _compute_loss(self, X_pgd, y_in, rv=None, step=0):
        """Compute the loss for gradient computation."""
        output = self.model(X_pgd)
        
        if self.use_odi and step < 2 and rv is not None:
            # ODI loss
            loss = ops.reduce_sum(output * rv)
        elif self.use_odi and step >= 2:
            # Margin loss for ODI after initial steps
            loss = margin_loss(output, y_in)
        elif self.type == 'CW':
            # CW loss
            loss = cw_loss(output, y_in)
        else:
            # Cross-entropy loss
            ce_loss = nn.CrossEntropyLoss()
            loss = ce_loss(output, y_in)
            
        return loss

    def perturb(self, x_in, y_in):
        """
        Perform PGD attack on input images.
        
        Args:
            x_in: Input images
            y_in: True labels
            
        Returns:
            Adversarial examples
        """
        model = self.model
        epsilon = self.epsilon
        
        # Initialize adversarial examples
        X_adv = x_in.copy()  # MindSpore tensors don't have detach().clone()
        X_pgd = x_in.copy()
        nc = ops.zeros_like(y_in)

        for r in range(self.num_restarts):
            if self.random_start:
                random_noise = self._get_rand_noise(x_in)
                X_pgd = X_pgd + random_noise

            # For ODI, compute random vector
            rv = None
            if self.use_odi:
                out = model(x_in)
                rv = ops.uniform(out.shape, Tensor(-1.0, ms.float32), Tensor(1.0, ms.float32), dtype=ms.float32)

            for i in range(self.num_steps):
                # Define forward function for gradient computation
                def forward_fn(X):
                    return self._compute_loss(X, y_in, rv, i)
                
                # Compute gradient with respect to X_pgd
                grad_fn = self.grad_op(forward_fn)
                grad = grad_fn(X_pgd, Tensor(1.0, ms.float32))
                
                # Update X_pgd based on gradient sign
                if self.use_odi and i < 2:
                    eta = epsilon * ops.sign(grad)
                else:
                    eta = self.step_size * ops.sign(grad)
                    
                X_pgd = X_pgd + eta
                
                # Project back to epsilon ball
                eta = ops.clip_by_value(X_pgd - x_in, -epsilon, epsilon)
                X_pgd = x_in + eta
                
                # Clamp to valid range [v_min, v_max]
                X_pgd = ops.clip_by_value(X_pgd, self.v_min, self.v_max)
                
                # Check if attack is successful and update
                logits = self.model(X_pgd)
                X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv


class MTPGDAttack():
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
        
        # Create gradient operation for computing gradients
        self.grad_op = GradOperation(get_all=False, get_by_list=False, sens_param=True)

    def _get_rand_noise(self, X):
        """Generate random noise for initialization."""
        eps = self.epsilon
        # Create uniform random noise in [-eps, eps]
        return ops.uniform(X.shape, Tensor(-eps, ms.float32), Tensor(eps, ms.float32), dtype=ms.float32)

    def _compute_loss(self, X_pgd, y_gt, y_tg, rv=None, step=0):
        """Compute the loss for gradient computation in multi-targeted attack."""
        z = self.model(X_pgd)
        
        if self.use_odi and step < 2 and rv is not None:
            # ODI loss
            loss = ops.reduce_sum(z * rv)
        else:
            # Multi-targeted loss
            z_y = y_gt * z
            z_t = y_tg * z
            loss = ops.reduce_mean(-z_y + z_t)
            
        return loss

    def perturb(self, x_in, y_in):
        """
        Perform Multi-Targeted PGD attack on input images.
        
        Args:
            x_in: Input images
            y_in: True labels
            
        Returns:
            Adversarial examples
        """
        model = self.model
        epsilon = self.epsilon
        
        # Initialize adversarial examples
        X_adv = x_in.copy()
        X_pgd = x_in.copy()
        nc = ops.zeros_like(y_in)

        # Attack for each target class
        for t in range(self.num_classes):
            # Create one-hot encodings for ground truth and target
            y_gt = one_hot_tensor(y_in, self.num_classes)
            y_tg_indices = ops.zeros_like(y_in) + t
            y_tg = one_hot_tensor(y_tg_indices, self.num_classes)
            
            for r in range(self.num_restarts):
                if self.random_start:
                    random_noise = self._get_rand_noise(x_in)
                    X_pgd = X_pgd + random_noise

                # For ODI, compute random vector
                rv = None
                if self.use_odi:
                    out = model(x_in)
                    rv = ops.uniform(out.shape, Tensor(-1.0, ms.float32), Tensor(1.0, ms.float32), dtype=ms.float32)

                for i in range(self.num_steps):
                    # Define forward function for gradient computation
                    def forward_fn(X):
                        return self._compute_loss(X, y_gt, y_tg, rv, i)
                    
                    # Compute gradient with respect to X_pgd
                    grad_fn = self.grad_op(forward_fn)
                    grad = grad_fn(X_pgd, Tensor(1.0, ms.float32))
                    
                    # Update X_pgd based on gradient sign
                    if self.use_odi and i < 2:
                        eta = epsilon * ops.sign(grad)
                    else:
                        eta = self.step_size * ops.sign(grad)
                        
                    X_pgd = X_pgd + eta
                    
                    # Project back to epsilon ball
                    eta = ops.clip_by_value(X_pgd - x_in, -epsilon, epsilon)
                    X_pgd = x_in + eta
                    
                    # Clamp to valid range [v_min, v_max]
                    X_pgd = ops.clip_by_value(X_pgd, self.v_min, self.v_max)
                    
                    # Check if attack is successful and update
                    logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv