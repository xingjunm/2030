import time
import sys
import tensorflow as tf
import numpy as np
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
        # Sort logits in descending order
        x_sorted = tf.sort(x, axis=1, direction='DESCENDING')
        ind_sorted = tf.argsort(x, axis=1, direction='DESCENDING')
        
        # Check if the highest logit corresponds to the true class
        ind = tf.cast(tf.equal(ind_sorted[:, 0], y), tf.float32)
        
        # Get logits for true class
        batch_indices = tf.range(tf.shape(x)[0])
        y_logits = tf.gather_nd(x, tf.stack([batch_indices, y], axis=1))
        
        # Compute DLR loss
        # When highest logit is true class, use second highest; otherwise use highest
        dlr = -(y_logits - x_sorted[:, 1] * ind - x_sorted[:, 0] * (1. - ind)) / (tf.reduce_sum(x_sorted, axis=1) + 1e-12)
        
        return dlr

    def perturb(self, x_in, y_in):
        change_point = self.change_point
        
        # Ensure inputs have correct dimensions
        if len(x_in.shape) == 3:
            x = tf.expand_dims(x_in, axis=0)
        else:
            x = tf.identity(x_in)
            
        if len(y_in.shape) == 0:
            y = tf.expand_dims(y_in, axis=0)
        else:
            y = tf.identity(y_in)

        nat_logits = self.model(x)
        nat_pred = tf.argmax(nat_logits, axis=1)
        nat_correct = tf.equal(nat_pred, tf.cast(y, nat_pred.dtype))

        y_gt = tf.one_hot(y, self.num_classes)

        nc = tf.zeros_like(y)
        X_adv = tf.identity(x)
        gt_prev = 0

        for _ in range(max(self.num_random_starts, 1)):
            for r in range(2):
                # Generate random noise
                r_noise = tf.random.uniform(shape=tf.shape(x), minval=-self.epsilon, maxval=self.epsilon)
                X_pgd = x + r_noise
                
                if self.use_odi:
                    rv = tf.random.uniform(shape=tf.shape(nat_logits), minval=-1., maxval=1.)

                for i in range(self.num_steps):
                    with tf.GradientTape() as tape:
                        tape.watch(X_pgd)
                        logits = self.model(X_pgd)
                        
                        # Compute z_max and z_y
                        masked_logits_max = logits * (1 - y_gt) - y_gt * 1.e8
                        z_max = tf.reduce_max(masked_logits_max, axis=1)
                        max_idx = tf.argmax(masked_logits_max, axis=1)
                        
                        masked_logits_y = logits * y_gt - (1 - y_gt) * 1.e8
                        z_y = tf.reduce_max(masked_logits_y, axis=1)

                        if self.use_odi and i < 2:
                            loss = tf.reduce_sum(logits * rv)
                        elif i < 1:
                            loss_per_sample = z_y
                            loss = tf.reduce_mean(loss_per_sample)
                        elif i < change_point:
                            loss_per_sample = z_max if r else -z_y
                            loss = tf.reduce_mean(loss_per_sample)
                        elif self.use_dlr:
                            loss = tf.reduce_mean(self.dlr_loss(logits, y))
                        else:
                            loss_per_sample = z_max - z_y
                            loss = tf.reduce_mean(loss_per_sample)
                            
                        X_adv, nc = adv_check_and_update(X_pgd, logits, y, nc, X_adv)
                    
                    # Compute gradients
                    grad = tape.gradient(loss, X_pgd)
                    
                    # Compute step size (alpha) with cosine annealing
                    if self.use_odi and i < 2:
                        alpha = self.epsilon
                    elif i > change_point:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i-change_point - 1) / (self.num_steps-change_point) * np.pi))
                    else:
                        alpha = self.initial_step_size * 0.5 * (1 + np.cos((i - 1) / (self.num_steps-change_point) * np.pi))
                    
                    # Update adversarial example
                    eta = alpha * tf.sign(grad)
                    X_pgd = X_pgd + eta
                    
                    # Project back to epsilon ball
                    X_pgd = tf.minimum(tf.maximum(X_pgd, x - self.epsilon), x + self.epsilon)
                    X_pgd = tf.clip_by_value(X_pgd, self.v_min, self.v_max)
                    
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
        
        # Ensure inputs have correct dimensions
        if len(x_in.shape) == 3:
            x = tf.expand_dims(x_in, axis=0)
        else:
            x = tf.identity(x_in)
            
        if len(y_in.shape) == 0:
            y = tf.expand_dims(y_in, axis=0)
        else:
            y = tf.identity(y_in)

        nat_logits = self.model(x)
        nat_pred = tf.argmax(nat_logits, axis=1)
        nat_correct = tf.equal(nat_pred, tf.cast(y, nat_pred.dtype))
        nc = tf.zeros_like(y)
        X_adv = tf.identity(x)

        for t in range(9, -1, -1):
            y_gt = one_hot_tensor(y, self.num_classes)
            
            # Get the t-th largest logit index for each sample
            sorted_indices = tf.argsort(nat_logits, axis=1, direction='DESCENDING')
            if t == 0:
                target_indices = sorted_indices[:, 0]
            else:
                target_indices = sorted_indices[:, t]
            y_tg = one_hot_tensor(target_indices, self.num_classes)

            for _ in range(max(self.num_random_starts, 1)):
                for r in range(2):
                    # Generate random noise
                    r_noise = tf.random.uniform(shape=tf.shape(x), minval=-self.epsilon, maxval=self.epsilon)
                    X_pgd = x + r_noise

                    if self.use_odi:
                        rv = tf.random.uniform(shape=tf.shape(nat_logits), minval=-1., maxval=1.)

                    for i in range(self.num_steps):
                        with tf.GradientTape() as tape:
                            tape.watch(X_pgd)
                            logits = self.model(X_pgd)
                            
                            # Compute z_t and z_y
                            masked_logits_t = y_tg * logits - (1 - y_tg) * 1.e8
                            z_t = tf.reduce_max(masked_logits_t, axis=1)
                            
                            masked_logits_y = y_gt * logits - (1 - y_gt) * 1.e8
                            z_y = tf.reduce_max(masked_logits_y, axis=1)
                            
                            if self.use_odi and i < 2:
                                loss = tf.reduce_sum(logits * rv)
                            elif i < 1:
                                loss = tf.reduce_mean(z_y)
                            elif i < change_point:
                                loss = tf.reduce_mean(z_t) if r else tf.reduce_mean(-z_y)
                            else:
                                loss = tf.reduce_mean(z_t - z_y)
                                
                            X_adv, nc = adv_check_and_update(X_pgd, logits, y, nc, X_adv)
                        
                        # Compute gradients
                        grad = tape.gradient(loss, X_pgd)
                        
                        # Compute step size (alpha) with cosine annealing
                        if self.use_odi and i < 2:
                            alpha = self.epsilon
                        elif i > change_point:
                            alpha = self.epsilon * 2.0 * 0.5 * (1 + np.cos((i-change_point-1) / (self.num_steps-change_point) * np.pi))
                        else:
                            alpha = self.epsilon * 2.0 * 0.5 * (1 + np.cos((i-1) / (self.num_steps-change_point) * np.pi))
                        
                        # Update adversarial example
                        eta = alpha * tf.sign(grad)
                        X_pgd = X_pgd + eta
                        
                        # Project back to epsilon ball
                        X_pgd = tf.minimum(tf.maximum(X_pgd, x - self.epsilon), x + self.epsilon)
                        X_pgd = tf.clip_by_value(X_pgd, self.v_min, self.v_max)
                    
                    # Final update check
                    final_logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, final_logits, y, nc, X_adv)

        return X_adv