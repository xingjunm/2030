import numpy as np
import time
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import GradOperation

import os
import sys


class APGDAttack():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        # Device is not explicitly used in MindSpore (exemption #3)
        self.device = device
        
        # Pre-define operations
        self.sort = ops.Sort(axis=1)
        self.sign = ops.Sign()
        self.minimum = ops.Minimum()
        self.maximum = ops.Maximum()
        self.sqrt = ops.Sqrt()
        self.abs = ops.Abs()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()
        self.concat = ops.Concat()
        self.cast = ops.Cast()
        self.nonzero = ops.NonZero()
        self.gather = ops.Gather()
        
        # Initialize gradient function for the model
        self.grad_fn = GradOperation(get_all=True)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = self.sort(x)
        batch_size = x.shape[0]
        
        # Create indices for gathering
        batch_indices = ms.numpy.arange(batch_size)
        
        # Check if the highest confidence prediction is correct
        ind = self.cast(ind_sorted[:, -1] == y, ms.float32)
        
        # Gather the logits for true labels
        x_y = x[batch_indices, y]
        
        return -(x_y - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def ce_loss_fn(self, logits, labels):
        """Cross entropy loss function for MindSpore"""
        loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='none')
        return loss_fn(logits, labels)

    def attack_single_run(self, x_in, y_in):
        x = x_in if len(x_in.shape) == 4 else self.expand_dims(x_in, 0)
        y = y_in if len(y_in.shape) == 1 else self.expand_dims(y_in, 0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        # Initialize adversarial examples
        if self.norm == 'Linf':
            t = 2 * ops.uniform(x.shape, Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) - 1
            t_reshape = t.reshape((t.shape[0], -1))
            t_max = self.abs(t_reshape).max(axis=1, keepdims=True)[0].reshape((-1, 1, 1, 1))
            x_adv = x + self.eps * ops.ones((x.shape[0], 1, 1, 1), ms.float32) * t / t_max
        elif self.norm == 'L2':
            t = ops.standard_normal(x.shape).astype(ms.float32)
            t_norm = self.sqrt((t ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
            x_adv = x + self.eps * ops.ones((x.shape[0], 1, 1, 1), ms.float32) * t / t_norm
        
        x_adv = ops.clip_by_value(x_adv, Tensor(0., ms.float32), Tensor(1., ms.float32))
        x_best = x_adv.copy()
        x_best_adv = x_adv.copy()
        loss_steps = ops.zeros((self.n_iter, x.shape[0]), ms.float32)
        loss_best_steps = ops.zeros((self.n_iter + 1, x.shape[0]), ms.float32)
        acc_steps = ops.zeros_like(loss_best_steps)

        # Select loss function
        if self.loss == 'ce':
            criterion_indiv = self.ce_loss_fn
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss')

        # Compute gradient for the first iteration
        grad = ops.zeros_like(x)
        for _ in range(self.eot_iter):
            def forward_fn(x_adv):
                logits = self.model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
                return loss
            
            grad_result = self.grad_fn(forward_fn)(x_adv)
            grad = grad + grad_result[0]
            
        grad = grad / float(self.eot_iter)
        grad_best = grad.copy()
        
        # Compute logits and loss for accuracy check
        logits = self.model(x_adv)
        loss_indiv = criterion_indiv(logits, y)
        
        acc = logits.argmax(axis=1) == y
        acc_steps[0] = self.cast(acc, ms.float32)
        loss_best = loss_indiv.copy()

        step_size = self.eps * ops.ones((x.shape[0], 1, 1, 1), ms.float32) * Tensor([2.0], ms.float32).reshape((1, 1, 1, 1))
        x_adv_old = x_adv.copy()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.copy()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.copy()

            a = 0.75 if i > 0 else 1.0

            if self.norm == 'Linf':
                x_adv_1 = x_adv + step_size * self.sign(grad)
                x_adv_1 = ops.clip_by_value(self.minimum(self.maximum(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                x_adv_1 = ops.clip_by_value(
                    self.minimum(self.maximum(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 
                    0.0, 1.0
                )

            elif self.norm == 'L2':
                grad_norm = self.sqrt((grad ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
                x_adv_1 = x_adv + step_size * grad / grad_norm
                
                diff = x_adv_1 - x
                diff_norm = self.sqrt((diff ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
                factor = self.minimum(
                    self.eps * ops.ones(x.shape, ms.float32), 
                    diff_norm
                ) / diff_norm
                x_adv_1 = ops.clip_by_value(x + diff * factor, 0.0, 1.0)
                
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                diff = x_adv_1 - x
                diff_norm = self.sqrt((diff ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
                factor = self.minimum(
                    self.eps * ops.ones(x.shape, ms.float32), 
                    diff_norm
                ) / (diff_norm + 1e-12)
                x_adv_1 = ops.clip_by_value(x + diff * factor, 0.0, 1.0)

            x_adv = x_adv_1

            ### get gradient
            grad = ops.zeros_like(x)
            for _ in range(self.eot_iter):
                def forward_fn_iter(x_adv):
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                    return loss
                
                grad_result = self.grad_fn(forward_fn_iter)(x_adv)
                grad = grad + grad_result[0]
                
            grad = grad / float(self.eot_iter)
            
            # Compute logits and loss for this iteration
            logits = self.model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            
            # Get predictions and loss
            logits = self.model(x_adv)
            loss_indiv = criterion_indiv(logits, y)

            pred = logits.argmax(axis=1) == y
            acc = self.minimum(self.cast(acc, ms.float32), self.cast(pred, ms.float32))
            acc_steps[i + 1] = acc
            
            # Update best adversarial examples for misclassified samples
            pred_false = self.cast(pred == False, ms.int32)
            if pred_false.sum() > 0:
                # Get indices where prediction is false
                false_indices = self.nonzero(pred_false).squeeze()
                if false_indices.ndim == 0:
                    false_indices = self.expand_dims(false_indices, 0)
                for idx in false_indices.asnumpy():
                    x_best_adv[idx] = x_adv[idx]
            
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum().asnumpy()))

            ### check step size
            y1 = loss_indiv.copy()
            loss_steps[i] = y1
            
            # Update best values where loss improved
            improved = y1 > loss_best
            improved_indices = self.nonzero(self.cast(improved, ms.int32)).squeeze()
            if improved_indices.size > 0:
                if improved_indices.ndim == 0:
                    improved_indices = self.expand_dims(improved_indices, 0)
                for idx in improved_indices.asnumpy():
                    x_best[idx] = x_adv[idx]
                    grad_best[idx] = grad[idx]
                    loss_best[idx] = y1[idx]
                    
            loss_best_steps[i + 1] = loss_best

            counter3 += 1

            if counter3 == k:
                fl_oscillation = self.check_oscillation(
                    loss_steps.asnumpy(), i, k, 
                    loss_best.asnumpy(), k3=self.thr_decr
                )
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.asnumpy() >= loss_best.asnumpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.copy()

                if np.sum(fl_oscillation) > 0:
                    # Reduce step size for oscillating indices
                    fl_indices = np.where(fl_oscillation)[0]
                    for idx in fl_indices:
                        step_size[idx] = step_size[idx] / 2.0
                    n_reduced = fl_oscillation.astype(float).sum()

                    # Reset to best values for oscillating indices
                    for idx in fl_indices:
                        x_adv[idx] = x_best[idx]
                        grad[idx] = grad_best[idx]

                counter3 = 0
                k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in if len(x_in.shape) == 4 else self.expand_dims(x_in, 0)
        y = y_in if len(y_in.shape) == 1 else self.expand_dims(y_in, 0)

        adv = x.copy()
        acc = self.cast(self.model(x).argmax(axis=1) == y, ms.float32)
        loss = -1e10 * ops.ones_like(acc)
        
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.mean().asnumpy()))
        startt = time.time()

        if not best_loss:
            # Set random seed
            ms.set_seed(self.seed)
            np.random.seed(self.seed)

            if not cheap:
                raise ValueError('not implemented yet')

            else:
                for counter in range(self.n_restarts):
                    ind_to_fool = self.nonzero(self.cast(acc, ms.int32)).squeeze()
                    if ind_to_fool.ndim == 0:
                        ind_to_fool = self.expand_dims(ind_to_fool, 0)
                    if ind_to_fool.size != 0:
                        ind_to_fool_np = ind_to_fool.asnumpy()
                        x_to_fool = ops.gather(x, Tensor(ind_to_fool_np, ms.int32), 0)
                        y_to_fool = ops.gather(y, Tensor(ind_to_fool_np, ms.int32), 0)
                        
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        
                        # Find indices where attack succeeded
                        ind_curr = self.nonzero(self.cast(acc_curr == 0, ms.int32)).squeeze()
                        if ind_curr.size > 0:
                            if ind_curr.ndim == 0:
                                ind_curr = self.expand_dims(ind_curr, 0)
                            ind_curr_np = ind_curr.asnumpy()
                            
                            # Update results for successful attacks
                            for i, curr_idx in enumerate(ind_curr_np):
                                fool_idx = ind_to_fool_np[curr_idx]
                                acc[fool_idx] = 0
                                adv[fool_idx] = adv_curr[curr_idx]
                                
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, acc.mean().asnumpy(), time.time() - startt))

            return adv

        else:
            adv_best = x.copy()
            loss_best = ops.ones((x.shape[0],), ms.float32) * (-float('inf'))
            
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = self.nonzero(self.cast(loss_curr > loss_best, ms.int32)).squeeze()
                
                if ind_curr.size > 0:
                    if ind_curr.ndim == 0:
                        ind_curr = self.expand_dims(ind_curr, 0)
                    ind_curr_np = ind_curr.asnumpy()
                    
                    for idx in ind_curr_np:
                        adv_best[idx] = best_curr[idx]
                        loss_best[idx] = loss_curr[idx]

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, loss_best.sum().asnumpy()))

            return adv_best


class APGDAttack_targeted():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, eot_iter=1, rho=.75, verbose=False, device='cuda',
                 n_target_classes=9):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        # Device is not explicitly used in MindSpore (exemption #3)
        self.device = device
        self.n_target_classes = n_target_classes
        
        # Pre-define operations
        self.sort = ops.Sort(axis=1)
        self.sign = ops.Sign()
        self.minimum = ops.Minimum()
        self.maximum = ops.Maximum()
        self.sqrt = ops.Sqrt()
        self.abs = ops.Abs()
        self.expand_dims = ops.ExpandDims()
        self.squeeze = ops.Squeeze()
        self.concat = ops.Concat()
        self.cast = ops.Cast()
        self.nonzero = ops.NonZero()
        self.gather = ops.Gather()
        
        # Initialize gradient function for the model
        self.grad_fn = GradOperation(get_all=True)

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = self.sort(x)
        batch_size = x.shape[0]
        batch_indices = ms.numpy.arange(batch_size)
        
        x_y = x[batch_indices, y]
        x_target = x[batch_indices, y_target]
        
        return -(x_y - x_target) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = x_in if len(x_in.shape) == 4 else self.expand_dims(x_in, 0)
        y = y_in if len(y_in.shape) == 1 else self.expand_dims(y_in, 0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        # Initialize adversarial examples
        if self.norm == 'Linf':
            t = 2 * ops.uniform(x.shape, Tensor(0.0, ms.float32), Tensor(1.0, ms.float32)) - 1
            t_reshape = t.reshape((t.shape[0], -1))
            t_max = self.abs(t_reshape).max(axis=1, keepdims=True)[0].reshape((-1, 1, 1, 1))
            x_adv = x + self.eps * ops.ones((x.shape[0], 1, 1, 1), ms.float32) * t / t_max
        elif self.norm == 'L2':
            t = ops.standard_normal(x.shape).astype(ms.float32)
            t_norm = self.sqrt((t ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
            x_adv = x + self.eps * ops.ones((x.shape[0], 1, 1, 1), ms.float32) * t / t_norm
            
        x_adv = ops.clip_by_value(x_adv, Tensor(0., ms.float32), Tensor(1., ms.float32))
        x_best = x_adv.copy()
        x_best_adv = x_adv.copy()
        loss_steps = ops.zeros((self.n_iter, x.shape[0]), ms.float32)
        loss_best_steps = ops.zeros((self.n_iter + 1, x.shape[0]), ms.float32)
        acc_steps = ops.zeros_like(loss_best_steps)

        # Get target class
        output = self.model(x)
        sorted_output, sorted_indices = self.sort(output)
        y_target = sorted_indices[:, -self.target_class]

        # Define forward function for gradient computation
        def forward_fn(x_adv):
            logits = self.model(x_adv)
            loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
            loss = loss_indiv.sum()
            return loss

        # First forward-backward pass
        grad = ops.zeros_like(x)
        for _ in range(self.eot_iter):
            grad_result = self.grad_fn(forward_fn)(x_adv)
            grad = grad + grad_result[0]
            
        grad = grad / float(self.eot_iter)
        grad_best = grad.copy()

        # Get initial accuracy and loss
        logits = self.model(x_adv)
        loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
        
        acc = logits.argmax(axis=1) == y
        acc_steps[0] = self.cast(acc, ms.float32)
        loss_best = loss_indiv.copy()

        step_size = self.eps * ops.ones((x.shape[0], 1, 1, 1), ms.float32) * Tensor([2.0], ms.float32).reshape((1, 1, 1, 1))
        x_adv_old = x_adv.copy()
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = loss_best.copy()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.copy()

            a = 0.75 if i > 0 else 1.0

            if self.norm == 'Linf':
                x_adv_1 = x_adv + step_size * self.sign(grad)
                x_adv_1 = ops.clip_by_value(self.minimum(self.maximum(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                x_adv_1 = ops.clip_by_value(
                    self.minimum(self.maximum(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 
                    0.0, 1.0
                )

            elif self.norm == 'L2':
                grad_norm = self.sqrt((grad ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
                x_adv_1 = x_adv + step_size[0] * grad / grad_norm
                
                diff = x_adv_1 - x
                diff_norm = self.sqrt((diff ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
                factor = self.minimum(
                    self.eps * ops.ones(x.shape, ms.float32), 
                    diff_norm
                ) / diff_norm
                x_adv_1 = ops.clip_by_value(x + diff * factor, 0.0, 1.0)
                
                x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                diff = x_adv_1 - x
                diff_norm = self.sqrt((diff ** 2).sum(axis=(1, 2, 3), keepdims=True)) + 1e-12
                factor = self.minimum(
                    self.eps * ops.ones(x.shape, ms.float32), 
                    diff_norm
                ) / (diff_norm + 1e-12)
                x_adv_1 = ops.clip_by_value(x + diff * factor, 0.0, 1.0)

            x_adv = x_adv_1

            ### get gradient
            grad = ops.zeros_like(x)
            for _ in range(self.eot_iter):
                def forward_fn_iter(x_adv):
                    logits = self.model(x_adv)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = loss_indiv.sum()
                    return loss
                
                grad_result = self.grad_fn(forward_fn_iter)(x_adv)
                grad = grad + grad_result[0]
                
            grad = grad / float(self.eot_iter)
            
            # Compute logits and loss for this iteration
            logits = self.model(x_adv)
            loss_indiv = criterion_indiv(logits, y)

            # Get predictions and loss
            logits = self.model(x_adv)
            loss_indiv = self.dlr_loss_targeted(logits, y, y_target)

            pred = logits.argmax(axis=1) == y
            acc = self.minimum(self.cast(acc, ms.float32), self.cast(pred, ms.float32))
            acc_steps[i + 1] = acc
            
            # Update best adversarial examples for misclassified samples
            pred_false = self.cast(pred == False, ms.int32)
            if pred_false.sum() > 0:
                false_indices = self.nonzero(pred_false).squeeze()
                if false_indices.ndim == 0:
                    false_indices = self.expand_dims(false_indices, 0)
                for idx in false_indices.asnumpy():
                    x_best_adv[idx] = x_adv[idx]
                    
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, loss_best.sum().asnumpy()))

            ### check step size
            y1 = loss_indiv.copy()
            loss_steps[i] = y1
            
            # Update best values where loss improved
            improved = y1 > loss_best
            improved_indices = self.nonzero(self.cast(improved, ms.int32)).squeeze()
            if improved_indices.size > 0:
                if improved_indices.ndim == 0:
                    improved_indices = self.expand_dims(improved_indices, 0)
                for idx in improved_indices.asnumpy():
                    x_best[idx] = x_adv[idx]
                    grad_best[idx] = grad[idx]
                    loss_best[idx] = y1[idx]
                    
            loss_best_steps[i + 1] = loss_best

            counter3 += 1

            if counter3 == k:
                fl_oscillation = self.check_oscillation(
                    loss_steps.asnumpy(), i, k, 
                    loss_best.asnumpy(), k3=self.thr_decr
                )
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.asnumpy() >= loss_best.asnumpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.copy()

                if np.sum(fl_oscillation) > 0:
                    # Reduce step size for oscillating indices
                    fl_indices = np.where(fl_oscillation)[0]
                    for idx in fl_indices:
                        step_size[idx] = step_size[idx] / 2.0
                    n_reduced = fl_oscillation.astype(float).sum()

                    # Reset to best values for oscillating indices
                    for idx in fl_indices:
                        x_adv[idx] = x_best[idx]
                        grad[idx] = grad_best[idx]

                counter3 = 0
                k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, acc, loss_best, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = x_in if len(x_in.shape) == 4 else self.expand_dims(x_in, 0)
        y = y_in if len(y_in.shape) == 1 else self.expand_dims(y_in, 0)

        adv = x.copy()
        acc = self.cast(self.model(x).argmax(axis=1) == y, ms.float32)
        loss = -1e10 * ops.ones_like(acc)
        
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.mean().asnumpy()))
        startt = time.time()

        # Set random seed
        ms.set_seed(self.seed)
        np.random.seed(self.seed)

        if not cheap:
            raise ValueError('not implemented yet')

        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = self.nonzero(self.cast(acc, ms.int32)).squeeze()
                    if ind_to_fool.ndim == 0:
                        ind_to_fool = self.expand_dims(ind_to_fool, 0)
                    if ind_to_fool.size != 0:
                        ind_to_fool_np = ind_to_fool.asnumpy()
                        x_to_fool = ops.gather(x, Tensor(ind_to_fool_np, ms.int32), 0)
                        y_to_fool = ops.gather(y, Tensor(ind_to_fool_np, ms.int32), 0)
                        
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        
                        # Find indices where attack succeeded
                        ind_curr = self.nonzero(self.cast(acc_curr == 0, ms.int32)).squeeze()
                        if ind_curr.size > 0:
                            if ind_curr.ndim == 0:
                                ind_curr = self.expand_dims(ind_curr, 0)
                            ind_curr_np = ind_curr.asnumpy()
                            
                            # Update results for successful attacks
                            for i, curr_idx in enumerate(ind_curr_np):
                                fool_idx = ind_to_fool_np[curr_idx]
                                acc[fool_idx] = 0
                                adv[fool_idx] = adv_curr[curr_idx]
                                
                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, acc.mean().asnumpy(), self.eps, time.time() - startt))

        return adv