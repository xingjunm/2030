import time
import sys
import paddle
import paddle.optimizer as optim
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .utils import adv_check_and_update, one_hot_tensor


def cw_loss(logits, y):
    correct_logit = paddle.sum(paddle.gather(logits, y.unsqueeze(1).astype('int64'), axis=1).squeeze())
    tmp1 = paddle.argsort(logits, axis=1)[:, -2:]
    new_y = paddle.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])
    wrong_logit = paddle.sum(paddle.gather(logits, new_y.unsqueeze(1).astype('int64'), axis=1).squeeze())
    loss = - F.relu(correct_logit-wrong_logit)
    return loss


def margin_loss(logits, y):
    logit_org = paddle.gather(logits, y.reshape([-1, 1]), axis=1)
    # Create eye matrix and compute logit_target
    # Note: PaddlePaddle doesn't need explicit device specification (exemption #3)
    eye_matrix = paddle.eye(10, dtype='float32')
    logit_target = paddle.gather(logits, 
                                  paddle.argmax(logits - eye_matrix[y] * 9999, axis=1, keepdim=True),
                                  axis=1)
    loss = -logit_org + logit_target
    loss = paddle.sum(loss)
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

    def _get_rand_noise(self, X):
        eps = self.epsilon
        # PaddlePaddle doesn't need explicit device specification (exemption #3)
        # Use float32 by default (exemption #2)
        return paddle.uniform(shape=X.shape, min=-eps, max=eps, dtype='float32')

    def perturb(self, x_in, y_in):
        model = self.model
        # No device management needed in PaddlePaddle (exemption #3)
        epsilon = self.epsilon
        X_adv = x_in.detach().clone()
        X_pgd = x_in.clone()
        X_pgd.stop_gradient = False
        nc = paddle.zeros_like(y_in)

        for r in range(self.num_restarts):
            if self.random_start:
                random_noise = self._get_rand_noise(x_in)
                with paddle.no_grad():
                    X_pgd = X_pgd + random_noise
                X_pgd = X_pgd.clone()
                X_pgd.stop_gradient = False

            if self.use_odi:
                out = model(x_in)
                # Use float32 by default (exemption #2)
                rv = paddle.uniform(shape=out.shape, min=-1., max=1., dtype='float32')

            for i in range(self.num_steps):
                # Ensure X_pgd requires gradient
                X_pgd.stop_gradient = False
                
                if self.use_odi and i < 2:
                    loss = paddle.sum(model(X_pgd) * rv)
                elif self.use_odi:
                    loss = margin_loss(model(X_pgd), y_in)
                elif self.type == 'CW':
                    loss = cw_loss(model(X_pgd), y_in)
                else:
                    loss = paddle.nn.CrossEntropyLoss()(model(X_pgd), y_in)
                
                loss.backward()
                
                # Now X_pgd.grad should exist
                if self.use_odi and i < 2:
                    eta = epsilon * paddle.sign(X_pgd.grad)
                else:
                    eta = self.step_size * paddle.sign(X_pgd.grad)
                
                # Update X_pgd with gradient (using no_grad context to avoid tracking this operation)
                with paddle.no_grad():
                    X_pgd = X_pgd + eta
                    # Clamp perturbation
                    eta = paddle.clip(X_pgd - x_in, -epsilon, epsilon)
                    X_pgd = x_in + eta
                    # Clamp to valid range
                    X_pgd = paddle.clip(X_pgd, 0, 1.0)
                
                # Make X_pgd differentiable again for next iteration
                X_pgd = X_pgd.clone()
                X_pgd.stop_gradient = False
                
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

    def _get_rand_noise(self, X):
        eps = self.epsilon
        # PaddlePaddle doesn't need explicit device specification (exemption #3)
        # Use float32 by default (exemption #2)
        return paddle.uniform(shape=X.shape, min=-eps, max=eps, dtype='float32')

    def perturb(self, x_in, y_in):
        model = self.model
        # No device management needed in PaddlePaddle (exemption #3)
        epsilon = self.epsilon
        X_adv = x_in.detach().clone()
        X_pgd = x_in.clone()
        X_pgd.stop_gradient = False
        nc = paddle.zeros_like(y_in)

        for t in range(self.num_classes):
            y_gt = one_hot_tensor(y_in, self.num_classes)
            y_tg = paddle.zeros_like(y_in)
            y_tg = y_tg + t
            y_tg = one_hot_tensor(y_tg, self.num_classes)
            # y_tg = one_hot_tensor(targets, self.num_classes)
            
            for r in range(self.num_restarts):
                if self.random_start:
                    random_noise = self._get_rand_noise(x_in)
                    with paddle.no_grad():
                        X_pgd = X_pgd + random_noise
                    X_pgd = X_pgd.clone()
                    X_pgd.stop_gradient = False

                if self.use_odi:
                    out = model(x_in)
                    # Use float32 by default (exemption #2)
                    rv = paddle.uniform(shape=out.shape, min=-1., max=1., dtype='float32')

                for i in range(self.num_steps):
                    # Ensure X_pgd requires gradient
                    X_pgd.stop_gradient = False
                    
                    if self.use_odi and i < 2:
                        loss = paddle.sum(model(X_pgd) * rv)
                    else:
                        z = model(X_pgd)
                        z_y = y_gt * z
                        z_t = y_tg * z
                        loss = paddle.mean(-z_y + z_t)
                    
                    loss.backward()
                    
                    # Now X_pgd.grad should exist
                    if self.use_odi and i < 2:
                        eta = epsilon * paddle.sign(X_pgd.grad)
                    else:
                        eta = self.step_size * paddle.sign(X_pgd.grad)
                    
                    # Update X_pgd with gradient (using no_grad context to avoid tracking this operation)
                    with paddle.no_grad():
                        X_pgd = X_pgd + eta
                        # Clamp perturbation
                        eta = paddle.clip(X_pgd - x_in, -epsilon, epsilon)
                        X_pgd = x_in + eta
                        # Clamp to valid range
                        X_pgd = paddle.clip(X_pgd, 0, 1.0)
                    
                    # Make X_pgd differentiable again for next iteration
                    X_pgd = X_pgd.clone()
                    X_pgd.stop_gradient = False
                    
                    logits = self.model(X_pgd)
                    X_adv, nc = adv_check_and_update(X_pgd, logits, y_in, nc, X_adv)

        return X_adv