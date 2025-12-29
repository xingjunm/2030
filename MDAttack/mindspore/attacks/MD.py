import time
import sys
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from .utils import adv_check_and_update, one_hot_tensor


class MDAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_random_starts=1, v_min=0., v_max=1., change_point=25,
                 first_step_size=16./255., seed=0, norm='Linf', num_classes=10,
                 use_odi=False, use_dlr=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_random_starts = num_random_starts
        self.v_min = v_min
        self.v_max = v_max
        self.change_point = change_point
        self.first_step_size = first_step_size
        self.seed = seed
        self.norm = norm
        self.num_classes = num_classes
        self.use_odi = use_odi
        self.use_dlr = use_dlr
        self.initial_step_size = 2.0 * epsilon

    def dlr_loss(self, x, y):
        # Sort along dimension 1
        x_sorted = ops.Sort(axis=1)(x)[0]
        ind_sorted = ops.Sort(axis=1)(x)[1]
        
        # Check if the highest prediction is the true label
        ind = ops.cast(ind_sorted[:, -1] == y, ms.float32)
        
        # Gather values for true labels
        y_expanded = ops.expand_dims(y, axis=1)
        x_y = ops.GatherD()(x, 1, y_expanded).squeeze(axis=1)
        
        # Calculate DLR loss
        loss = -(x_y - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (ops.reduce_sum(x_sorted, axis=1) + 1e-12)
        return loss

    def perturb(self, x_in, y_in):
        change_point = self.change_point
        # Handle input shapes - keep original batch dimension
        x = x_in
        y = y_in if len(y_in.shape) == 1 else y_in[0]  # If y is 2D, take first dimension

        # Clone y to avoid any reference issues
        y = y.copy()
        
        nat_logits = self.model(x)
        nat_pred = ops.Argmax(axis=1)(nat_logits)
        nat_correct = (nat_pred == y)

        # One-hot encoding - compute outside gradient context
        y_gt = one_hot_tensor(y, self.num_classes)

        nc = ops.zeros_like(y)
        X_adv = x
        gt_prev = 0

        # Define gradient function for the attack
        grad_fn = ops.GradOperation(get_all=False, get_by_list=False, sens_param=False)

        for _ in range(max(self.num_random_starts, 1)):
            for r in range(2):
                # Generate random noise
                r_noise = Tensor(np.random.uniform(-self.epsilon, self.epsilon, x.shape), ms.float32)
                X_pgd = x + r_noise
                
                if self.use_odi:
                    rv = Tensor(np.random.uniform(-1., 1., nat_logits.shape), ms.float32)

                for i in range(self.num_steps):
                    # Define loss function for gradient computation
                    def loss_fn(X_pgd_inner):
                        logits = self.model(X_pgd_inner)
                        
                        # Calculate z_max and z_y
                        z_max = ops.reduce_max(logits * (1 - y_gt) - y_gt * 1.e8, axis=1)
                        z_y = ops.reduce_max(logits * y_gt - (1 - y_gt) * 1.e8, axis=1)
                        
                        if self.use_odi and i < 2:
                            loss = ops.reduce_sum(logits * rv)
                        elif i < 1:
                            loss_per_sample = z_y
                            loss = ops.reduce_mean(loss_per_sample)
                        elif i < change_point:
                            loss_per_sample = z_max if r else -z_y
                            loss = ops.reduce_mean(loss_per_sample)
                        elif self.use_dlr:
                            loss = ops.reduce_mean(self.dlr_loss(logits, y))
                        else:
                            loss_per_sample = z_max - z_y
                            loss = ops.reduce_mean(loss_per_sample)
                        
                        return loss
                    
                    # Compute gradient
                    X_pgd_param = Parameter(X_pgd, requires_grad=True)
                    grad = grad_fn(loss_fn)(X_pgd_param)
                    
                    # Update adversarial examples
                    logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, logits, y, nc, X_adv)
                    
                    # Calculate step size
                    if self.use_odi and i < 2:
                        alpha = self.epsilon
                    elif i > change_point:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i-change_point - 1) / (self.num_steps-change_point) * np.pi))
                    else:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i - 1) / (self.num_steps-change_point) * np.pi))
                    
                    # Update X_pgd
                    # Ensure grad is a tensor
                    if not isinstance(grad, ms.Tensor):
                        grad = ms.Tensor(grad, ms.float32)
                    # Ensure alpha is a float scalar
                    alpha = float(alpha)
                    eta = alpha * ops.sign(grad)
                    # Ensure eta is a tensor
                    if not isinstance(eta, ms.Tensor):
                        eta = ms.Tensor(eta, ms.float32)
                    X_pgd = X_pgd + eta
                    
                    # Clip to epsilon ball
                    X_pgd = ops.minimum(ops.maximum(X_pgd, x - self.epsilon), x + self.epsilon)
                    X_pgd = ops.clip_by_value(X_pgd, self.v_min, self.v_max)
                
                # Final update check
                final_logits = self.model(X_pgd)
                X_adv, nc = adv_check_and_update(X_pgd, final_logits, y, nc, X_adv)

        return X_adv


class MDMTAttack():
    def __init__(self, model, epsilon=8./255., num_steps=50, step_size=2./255.,
                 num_random_starts=1, v_min=0., v_max=1., change_point=25,
                 first_step_size=16./255., seed=0, norm='Linf', num_classes=10,
                 use_odi=False):
        self.model = model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.num_random_starts = num_random_starts
        self.v_min = v_min
        self.v_max = v_max
        self.change_point = change_point
        self.first_step_size = first_step_size
        self.seed = seed
        self.norm = norm
        self.num_classes = num_classes
        self.use_odi = use_odi

    def perturb(self, x_in, y_in):
        change_point = self.change_point
        # Handle input shapes - keep original batch dimension
        x = x_in
        y = y_in if len(y_in.shape) == 1 else y_in[0]  # If y is 2D, take first dimension

        # Clone y to avoid any reference issues
        y = y.copy()
        
        nat_logits = self.model(x)
        nat_pred = ops.Argmax(axis=1)(nat_logits)
        nat_correct = (nat_pred == y)
        nc = ops.zeros_like(y)
        X_adv = x

        # Define gradient function for the attack
        grad_fn = ops.GradOperation(get_all=False, get_by_list=False, sens_param=False)

        for t in range(9, -1, -1):
            # Create one-hot encodings
            y_gt = one_hot_tensor(y, self.num_classes)
            
            # Get the t-th highest prediction for each sample
            sorted_indices = ops.Sort(axis=1)(nat_logits)[1]
            y_tg_indices = sorted_indices[:, -t]
            y_tg = one_hot_tensor(y_tg_indices, self.num_classes)

            for _ in range(max(self.num_random_starts, 1)):
                for r in range(2):
                    # Generate random noise
                    r_noise = Tensor(np.random.uniform(-self.epsilon, self.epsilon, x.shape), ms.float32)
                    X_pgd = x + r_noise

                    if self.use_odi:
                        rv = Tensor(np.random.uniform(-1., 1., nat_logits.shape), ms.float32)

                    for i in range(self.num_steps):
                        # Define loss function for gradient computation
                        def loss_fn(X_pgd_inner):
                            logits = self.model(X_pgd_inner)
                            
                            # Calculate z_t and z_y
                            z_t = ops.reduce_max(y_tg * logits - (1 - y_tg) * 1.e8, axis=1)
                            z_y = ops.reduce_max(y_gt * logits - (1 - y_gt) * 1.e8, axis=1)
                            
                            if self.use_odi and i < 2:
                                loss = ops.reduce_sum(logits * rv)
                            elif i < 1:
                                loss = ops.reduce_mean(z_y)
                            elif i < change_point:
                                loss = ops.reduce_mean(z_t) if r else ops.reduce_mean(-z_y)
                            else:
                                loss = ops.reduce_mean(z_t - z_y)
                            
                            return loss
                        
                        # Compute gradient
                        X_pgd_param = Parameter(X_pgd, requires_grad=True)
                        grad = grad_fn(loss_fn)(X_pgd_param)
                        
                        # Update adversarial examples
                        logits = self.model(X_pgd)
                        X_adv, nc = adv_check_and_update(X_pgd, logits, y, nc, X_adv)
                        
                        # Calculate step size
                        if self.use_odi and i < 2:
                            alpha = self.epsilon
                        elif i > change_point:
                            alpha = self.epsilon * 2.0 * 0.5 * (1 + np.cos((i-change_point-1) / (self.num_steps-change_point) * np.pi))
                        else:
                            alpha = self.epsilon * 2.0 * 0.5 * (1 + np.cos((i-1) / (self.num_steps-change_point) * np.pi))
                        
                        # Update X_pgd
                        # Ensure grad is a tensor
                        if not isinstance(grad, ms.Tensor):
                            grad = ms.Tensor(grad, ms.float32)
                        # Ensure alpha is a float scalar
                        alpha = float(alpha)
                        eta = alpha * ops.sign(grad)
                        # Ensure eta is a tensor
                        if not isinstance(eta, ms.Tensor):
                            eta = ms.Tensor(eta, ms.float32)
                        X_pgd = X_pgd + eta
                        
                        # Clip to epsilon ball
                        X_pgd = ops.minimum(ops.maximum(X_pgd, x - self.epsilon), x + self.epsilon)
                        X_pgd = ops.clip_by_value(X_pgd, self.v_min, self.v_max)
                    
                    # Final update check
                    final_logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, final_logits, y, nc, X_adv)

        return X_adv