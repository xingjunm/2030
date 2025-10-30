import numpy as np
import time
import tensorflow as tf

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
        # Device operations use TensorFlow's default device allocation (exemption #3)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss(self, x, y):
        x_sorted = tf.sort(x, axis=1, direction='DESCENDING')
        ind_sorted = tf.argsort(x, axis=1, direction='DESCENDING')
        ind = tf.cast(ind_sorted[:, 0] == y, tf.float32)
        
        batch_indices = tf.range(tf.shape(x)[0])
        y_true = tf.gather_nd(x, tf.stack([batch_indices, y], axis=1))
        
        return -(y_true - x_sorted[:, 1] * ind - x_sorted[:, 0] * (1. - ind)) / (x_sorted[:, 0] - x_sorted[:, 2] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = tf.identity(x_in) if len(x_in.shape) == 4 else tf.expand_dims(x_in, 0)
        y = tf.identity(y_in) if len(y_in.shape) == 1 else tf.expand_dims(y_in, 0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * tf.random.uniform(tf.shape(x), dtype=tf.float32) - 1
            t_reshape = tf.reshape(t, [tf.shape(t)[0], -1])
            t_max = tf.reduce_max(tf.abs(t_reshape), axis=1, keepdims=True)
            t_max = tf.reshape(t_max, [-1, 1, 1, 1])
            x_adv = x + self.eps * tf.ones([tf.shape(x)[0], 1, 1, 1], dtype=tf.float32) * t / t_max
        elif self.norm == 'L2':
            t = tf.random.normal(tf.shape(x), dtype=tf.float32)
            t_norm = tf.sqrt(tf.reduce_sum(t ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
            x_adv = x + self.eps * tf.ones([tf.shape(x)[0], 1, 1, 1], dtype=tf.float32) * t / t_norm
        x_adv = tf.clip_by_value(x_adv, 0., 1.)
        x_best = tf.identity(x_adv)
        x_best_adv = tf.identity(x_adv)
        loss_steps = np.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = np.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = np.zeros_like(loss_best_steps)

        if self.loss == 'ce':
            def criterion_indiv(logits, y):
                return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        else:
            raise ValueError('unknown loss')

        # Initial gradient computation
        grad = tf.zeros_like(x)
        for _ in range(self.eot_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = criterion_indiv(logits, y)
                loss = tf.reduce_sum(loss_indiv)
            
            grad += tape.gradient(loss, x_adv)  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)
        grad_best = tf.identity(grad)

        acc = tf.cast(tf.argmax(logits, axis=1) == y, tf.float32)
        acc_steps[0] = acc.numpy()
        loss_best = loss_indiv.numpy()

        step_size = self.eps * tf.ones([x.shape[0], 1, 1, 1], dtype=tf.float32) * 2.0
        x_adv_old = tf.identity(x_adv)
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = np.copy(loss_best)
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            grad2 = x_adv - x_adv_old
            x_adv_old = tf.identity(x_adv)

            a = 0.75 if i > 0 else 1.0

            if self.norm == 'Linf':
                x_adv_1 = x_adv + step_size * tf.sign(grad)
                x_adv_1 = tf.clip_by_value(tf.minimum(tf.maximum(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                x_adv_1 = tf.clip_by_value(tf.minimum(tf.maximum(x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a), x - self.eps), x + self.eps), 0.0, 1.0)

            elif self.norm == 'L2':
                grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
                x_adv_1 = x_adv + step_size * grad / grad_norm
                delta = x_adv_1 - x
                delta_norm = tf.sqrt(tf.reduce_sum(delta ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
                factor = tf.minimum(self.eps * tf.ones(tf.shape(x), dtype=tf.float32), delta_norm)
                x_adv_1 = tf.clip_by_value(x + delta / delta_norm * factor, 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                delta = x_adv_1 - x
                delta_norm = tf.sqrt(tf.reduce_sum(delta ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
                factor = tf.minimum(self.eps * tf.ones(tf.shape(x), dtype=tf.float32), delta_norm)
                x_adv_1 = tf.clip_by_value(x + delta / delta_norm * factor, 0.0, 1.0)

            x_adv = x_adv_1

            ### get gradient
            grad = tf.zeros_like(x)
            for _ in range(self.eot_iter):
                with tf.GradientTape() as tape:
                    tape.watch(x_adv)
                    logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = criterion_indiv(logits, y)
                    loss = tf.reduce_sum(loss_indiv)
                
                grad += tape.gradient(loss, x_adv)  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = tf.cast(tf.argmax(logits, axis=1) == y, tf.float32)
            acc = tf.minimum(acc, pred)
            acc_steps[i + 1] = acc.numpy()
            
            # Update x_best_adv for samples where prediction is wrong
            pred_wrong = tf.cast(pred == 0, tf.bool)
            pred_wrong_indices = tf.where(pred_wrong)
            if tf.size(pred_wrong_indices) > 0:
                # Convert to numpy for indexing
                x_best_adv_np = x_best_adv.numpy()
                x_adv_np = x_adv.numpy()
                pred_wrong_np = pred_wrong.numpy()
                x_best_adv_np[pred_wrong_np] = x_adv_np[pred_wrong_np]
                x_best_adv = tf.constant(x_best_adv_np)
            
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, np.sum(loss_best)))

            ### check step size
            y1 = loss_indiv.numpy()
            loss_steps[i] = y1
            ind = y1 > loss_best
            
            if np.any(ind):
                x_best_np = x_best.numpy()
                x_adv_np = x_adv.numpy()
                grad_best_np = grad_best.numpy()
                grad_np = grad.numpy()
                
                x_best_np[ind] = x_adv_np[ind]
                grad_best_np[ind] = grad_np[ind]
                loss_best[ind] = y1[ind]
                
                x_best = tf.constant(x_best_np)
                grad_best = tf.constant(grad_best_np)
            
            loss_best_steps[i + 1] = loss_best

            counter3 += 1

            if counter3 == k:
                fl_oscillation = self.check_oscillation(loss_steps, i, k, loss_best, k3=self.thr_decr)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check >= loss_best)
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = np.copy(loss_best)

                if np.sum(fl_oscillation) > 0:
                    step_size_np = step_size.numpy()
                    step_size_np[u[fl_oscillation]] /= 2.0
                    step_size = tf.constant(step_size_np)
                    n_reduced = fl_oscillation.astype(float).sum()

                    # Update x_adv and grad for oscillating samples
                    x_adv_np = x_adv.numpy()
                    x_best_np = x_best.numpy()
                    grad_np = grad.numpy()
                    grad_best_np = grad_best.numpy()
                    
                    x_adv_np[fl_oscillation] = x_best_np[fl_oscillation]
                    grad_np[fl_oscillation] = grad_best_np[fl_oscillation]
                    
                    x_adv = tf.constant(x_adv_np)
                    grad = tf.constant(grad_np)

                counter3 = 0
                k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, acc, tf.constant(loss_best), x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = tf.identity(x_in) if len(x_in.shape) == 4 else tf.expand_dims(x_in, 0)
        y = tf.identity(y_in) if len(y_in.shape) == 1 else tf.expand_dims(y_in, 0)

        adv = tf.identity(x)
        acc = tf.cast(tf.argmax(self.model(x), axis=1) == y, tf.float32)
        loss = -1e10 * tf.ones_like(acc)
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(tf.reduce_mean(acc).numpy()))
        startt = time.time()

        if not best_loss:
            tf.random.set_seed(self.seed)

            if not cheap:
                raise ValueError('not implemented yet')

            else:
                for counter in range(self.n_restarts):
                    acc_np = acc.numpy()
                    ind_to_fool = np.where(acc_np != 0)[0]
                    if ind_to_fool.size != 0:
                        x_to_fool = tf.gather(x, ind_to_fool)
                        y_to_fool = tf.gather(y, ind_to_fool)
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        
                        acc_curr_np = acc_curr.numpy()
                        ind_curr = np.where(acc_curr_np == 0)[0]
                        
                        if ind_curr.size > 0:
                            # Update acc and adv
                            acc_np = acc.numpy()
                            adv_np = adv.numpy()
                            adv_curr_np = adv_curr.numpy()
                            
                            for idx, orig_idx in enumerate(ind_to_fool[ind_curr]):
                                acc_np[orig_idx] = 0
                                adv_np[orig_idx] = adv_curr_np[ind_curr[idx]]
                            
                            acc = tf.constant(acc_np)
                            adv = tf.constant(adv_np)
                        
                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                                counter, tf.reduce_mean(acc).numpy(), time.time() - startt))

            return adv

        else:
            adv_best = tf.identity(x)
            loss_best = tf.ones([x.shape[0]]) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y)
                ind_curr = loss_curr > loss_best
                
                if tf.reduce_any(ind_curr):
                    ind_curr_np = ind_curr.numpy()
                    adv_best_np = adv_best.numpy()
                    best_curr_np = best_curr.numpy()
                    loss_best_np = loss_best.numpy()
                    loss_curr_np = loss_curr.numpy()
                    
                    adv_best_np[ind_curr_np] = best_curr_np[ind_curr_np]
                    loss_best_np[ind_curr_np] = loss_curr_np[ind_curr_np]
                    
                    adv_best = tf.constant(adv_best_np)
                    loss_best = tf.constant(loss_best_np)

                if self.verbose:
                    print('restart {} - loss: {:.5f}'.format(counter, tf.reduce_sum(loss_best).numpy()))

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
        # Device operations use TensorFlow's default device allocation (exemption #3)
        self.n_target_classes = n_target_classes

    def check_oscillation(self, x, j, k, y5, k3=0.5):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
          t += x[j - counter5] > x[j - counter5 - 1]

        return t <= k*k3*np.ones(t.shape)

    def check_shape(self, x):
        return x if len(x.shape) > 0 else np.expand_dims(x, 0)

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted = tf.sort(x, axis=1, direction='DESCENDING')
        
        batch_indices = tf.range(tf.shape(x)[0])
        y_values = tf.gather_nd(x, tf.stack([batch_indices, y], axis=1))
        y_target_values = tf.gather_nd(x, tf.stack([batch_indices, y_target], axis=1))
        
        return -(y_values - y_target_values) / (x_sorted[:, 0] - .5 * x_sorted[:, 2] - .5 * x_sorted[:, 3] + 1e-12)

    def attack_single_run(self, x_in, y_in):
        x = tf.identity(x_in) if len(x_in.shape) == 4 else tf.expand_dims(x_in, 0)
        y = tf.identity(y_in) if len(y_in.shape) == 1 else tf.expand_dims(y_in, 0)

        self.n_iter_2, self.n_iter_min, self.size_decr = max(int(0.22 * self.n_iter), 1), max(int(0.06 * self.n_iter), 1), max(int(0.03 * self.n_iter), 1)
        if self.verbose:
            print('parameters: ', self.n_iter, self.n_iter_2, self.n_iter_min, self.size_decr)

        if self.norm == 'Linf':
            t = 2 * tf.random.uniform(tf.shape(x), dtype=tf.float32) - 1
            t_reshape = tf.reshape(t, [tf.shape(t)[0], -1])
            t_max = tf.reduce_max(tf.abs(t_reshape), axis=1, keepdims=True)
            t_max = tf.reshape(t_max, [-1, 1, 1, 1])
            x_adv = x + self.eps * tf.ones([tf.shape(x)[0], 1, 1, 1], dtype=tf.float32) * t / t_max
        elif self.norm == 'L2':
            t = tf.random.normal(tf.shape(x), dtype=tf.float32)
            t_norm = tf.sqrt(tf.reduce_sum(t ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
            x_adv = x + self.eps * tf.ones([tf.shape(x)[0], 1, 1, 1], dtype=tf.float32) * t / t_norm
        x_adv = tf.clip_by_value(x_adv, 0., 1.)
        x_best = tf.identity(x_adv)
        x_best_adv = tf.identity(x_adv)
        loss_steps = np.zeros([self.n_iter, x.shape[0]])
        loss_best_steps = np.zeros([self.n_iter + 1, x.shape[0]])
        acc_steps = np.zeros_like(loss_best_steps)

        output = self.model(x)
        y_target = tf.argsort(output, axis=1, direction='DESCENDING')[:, self.target_class - 1]

        # Initial gradient computation
        grad = tf.zeros_like(x)
        for _ in range(self.eot_iter):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                loss = tf.reduce_sum(loss_indiv)
            
            grad += tape.gradient(loss, x_adv)  # 1 backward pass (eot_iter = 1)

        grad /= float(self.eot_iter)
        grad_best = tf.identity(grad)

        acc = tf.cast(tf.argmax(logits, axis=1) == y, tf.float32)
        acc_steps[0] = acc.numpy()
        loss_best = loss_indiv.numpy()

        step_size = self.eps * tf.ones([x.shape[0], 1, 1, 1], dtype=tf.float32) * 2.0
        x_adv_old = tf.identity(x_adv)
        counter = 0
        k = self.n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0

        loss_best_last_check = np.copy(loss_best)
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(self.n_iter):
            ### gradient step
            grad2 = x_adv - x_adv_old
            x_adv_old = tf.identity(x_adv)

            a = 0.75 if i > 0 else 1.0

            if self.norm == 'Linf':
                x_adv_1 = x_adv + step_size * tf.sign(grad)
                x_adv_1 = tf.clip_by_value(tf.minimum(tf.maximum(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)
                x_adv_1 = tf.clip_by_value(tf.minimum(tf.maximum(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - self.eps), x + self.eps), 0.0, 1.0)

            elif self.norm == 'L2':
                grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
                x_adv_1 = x_adv + step_size * grad / grad_norm
                delta = x_adv_1 - x
                delta_norm = tf.sqrt(tf.reduce_sum(delta ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
                factor = tf.minimum(self.eps * tf.ones(tf.shape(x), dtype=tf.float32), delta_norm)
                x_adv_1 = tf.clip_by_value(x + delta / delta_norm * factor, 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                delta = x_adv_1 - x
                delta_norm = tf.sqrt(tf.reduce_sum(delta ** 2, axis=[1, 2, 3], keepdims=True) + 1e-12)
                factor = tf.minimum(self.eps * tf.ones(tf.shape(x), dtype=tf.float32), delta_norm)
                x_adv_1 = tf.clip_by_value(x + delta / delta_norm * factor, 0.0, 1.0)

            x_adv = x_adv_1

            ### get gradient
            grad = tf.zeros_like(x)
            for _ in range(self.eot_iter):
                with tf.GradientTape() as tape:
                    tape.watch(x_adv)
                    logits = self.model(x_adv)  # 1 forward pass (eot_iter = 1)
                    loss_indiv = self.dlr_loss_targeted(logits, y, y_target)
                    loss = tf.reduce_sum(loss_indiv)
                
                grad += tape.gradient(loss, x_adv)  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)

            pred = tf.cast(tf.argmax(logits, axis=1) == y, tf.float32)
            acc = tf.minimum(acc, pred)
            acc_steps[i + 1] = acc.numpy()
            
            # Update x_best_adv for samples where prediction is wrong
            pred_wrong = tf.cast(pred == 0, tf.bool)
            pred_wrong_indices = tf.where(pred_wrong)
            if tf.size(pred_wrong_indices) > 0:
                # Convert to numpy for indexing
                x_best_adv_np = x_best_adv.numpy()
                x_adv_np = x_adv.numpy()
                pred_wrong_np = pred_wrong.numpy()
                x_best_adv_np[pred_wrong_np] = x_adv_np[pred_wrong_np]
                x_best_adv = tf.constant(x_best_adv_np)
            
            if self.verbose:
                print('iteration: {} - Best loss: {:.6f}'.format(i, np.sum(loss_best)))

            ### check step size
            y1 = loss_indiv.numpy()
            loss_steps[i] = y1
            ind = y1 > loss_best
            
            if np.any(ind):
                x_best_np = x_best.numpy()
                x_adv_np = x_adv.numpy()
                grad_best_np = grad_best.numpy()
                grad_np = grad.numpy()
                
                x_best_np[ind] = x_adv_np[ind]
                grad_best_np[ind] = grad_np[ind]
                loss_best[ind] = y1[ind]
                
                x_best = tf.constant(x_best_np)
                grad_best = tf.constant(grad_best_np)
            
            loss_best_steps[i + 1] = loss_best

            counter3 += 1

            if counter3 == k:
                fl_oscillation = self.check_oscillation(loss_steps, i, k, loss_best, k3=self.thr_decr)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check >= loss_best)
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = np.copy(loss_best)

                if np.sum(fl_oscillation) > 0:
                    step_size_np = step_size.numpy()
                    step_size_np[u[fl_oscillation]] /= 2.0
                    step_size = tf.constant(step_size_np)
                    n_reduced = fl_oscillation.astype(float).sum()

                    # Update x_adv and grad for oscillating samples
                    x_adv_np = x_adv.numpy()
                    x_best_np = x_best.numpy()
                    grad_np = grad.numpy()
                    grad_best_np = grad_best.numpy()
                    
                    x_adv_np[fl_oscillation] = x_best_np[fl_oscillation]
                    grad_np[fl_oscillation] = grad_best_np[fl_oscillation]
                    
                    x_adv = tf.constant(x_adv_np)
                    grad = tf.constant(grad_np)

                counter3 = 0
                k = np.maximum(k - self.size_decr, self.n_iter_min)

        return x_best, acc, tf.constant(loss_best), x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True):
        assert self.norm in ['Linf', 'L2']
        x = tf.identity(x_in) if len(x_in.shape) == 4 else tf.expand_dims(x_in, 0)
        y = tf.identity(y_in) if len(y_in.shape) == 1 else tf.expand_dims(y_in, 0)

        adv = tf.identity(x)
        acc = tf.cast(tf.argmax(self.model(x), axis=1) == y, tf.float32)
        loss = -1e10 * tf.ones_like(acc)
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(tf.reduce_mean(acc).numpy()))
        startt = time.time()

        tf.random.set_seed(self.seed)

        if not cheap:
            raise ValueError('not implemented yet')

        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    acc_np = acc.numpy()
                    ind_to_fool = np.where(acc_np != 0)[0]
                    if ind_to_fool.size != 0:
                        x_to_fool = tf.gather(x, ind_to_fool)
                        y_to_fool = tf.gather(y, ind_to_fool)
                        best_curr, acc_curr, loss_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                        
                        acc_curr_np = acc_curr.numpy()
                        ind_curr = np.where(acc_curr_np == 0)[0]
                        
                        if ind_curr.size > 0:
                            # Update acc and adv
                            acc_np = acc.numpy()
                            adv_np = adv.numpy()
                            adv_curr_np = adv_curr.numpy()
                            
                            for idx, orig_idx in enumerate(ind_to_fool[ind_curr]):
                                acc_np[orig_idx] = 0
                                adv_np[orig_idx] = adv_curr_np[ind_curr[idx]]
                            
                            acc = tf.constant(acc_np)
                            adv = tf.constant(adv_np)
                        
                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, tf.reduce_mean(acc).numpy(), self.eps, time.time() - startt))

        return adv