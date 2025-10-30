# Copyright (c) 2019-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import time
import numpy as np
import tensorflow as tf

from attacks.fab_projections import projection_linf, projection_l2, projection_l1

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}


class FABAttack():
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
    :param norm:          Lp-norm to minimize ('Linf', 'L2', 'L1' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           epsilon for the random restarts
    :param alpha_max:     alpha_max
    :param eta:           overshooting
    :param beta:          backward step
    """

    def __init__(
            self,
            norm='Linf',
            n_restarts=1,
            n_iter=100,
            eps=None,
            alpha_max=0.1,
            eta=1.05,
            beta=0.9,
            loss_fn=None,
            verbose=False,
            seed=0,
            targeted=False,
            device=None,
            n_target_classes=9):
        """ FAB-attack implementation in tensorflow """

        self.norm = norm
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.eps = eps if eps is not None else DEFAULT_EPS_DICT_BY_NORM[norm]
        self.alpha_max = alpha_max
        self.eta = eta
        self.beta = beta
        self.targeted = False
        self.verbose = verbose
        self.seed = seed
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes

    def check_shape(self, x):
        return x if len(x.shape) > 0 else tf.expand_dims(x, 0)

    def _predict_fn(self, x):
        raise NotImplementedError("Virtual function.")

    def _get_predicted_label(self, x):
        raise NotImplementedError("Virtual function.")

    def get_diff_logits_grads_batch(self, imgs, la):
        raise NotImplementedError("Virtual function.")

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
       raise NotImplementedError("Virtual function.")

    def attack_single_run(self, x, y=None, use_rand_start=False, is_targeted=False):
        """
        :param x:             clean images
        :param y:             clean labels, if None we use the predicted labels
        :param is_targeted    True if we ise targeted version. Targeted class is assigned by `self.target_class`
        """

        # Device handling - TensorFlow automatically handles device placement
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = tf.identity(tf.cast(x, tf.float32))

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = tf.identity(tf.cast(y_pred, tf.int64))
        else:
            y = tf.identity(tf.cast(y, tf.int64))
        pred = tf.equal(y_pred, y)
        corr_classified = tf.reduce_sum(tf.cast(pred, tf.float32))
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(tf.reduce_mean(tf.cast(pred, tf.float32)).numpy()))
        if tf.reduce_sum(tf.cast(pred, tf.int32)) == 0:
            return x
        
        # Get indices of correctly classified samples
        pred_indices = tf.where(pred)
        pred_indices = tf.squeeze(pred_indices)
        pred_indices = self.check_shape(pred_indices)

        if is_targeted:
            output = self._predict_fn(x)
            sorted_output = tf.nn.top_k(output, k=output.shape[-1], sorted=True)[1]
            la_target = sorted_output[:, -self.target_class]
            la_target2 = tf.gather(la_target, pred_indices)

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = tf.gather(x, pred_indices)
        la2 = tf.gather(y, pred_indices)
        if len(im2.shape) == self.ndims:
            im2 = tf.expand_dims(im2, 0)
        bs = im2.shape[0]
        u1 = tf.range(bs, dtype=tf.int32)
        adv = tf.identity(im2)
        adv_c = tf.identity(x)
        res2 = 1e10 * tf.ones([bs], dtype=tf.float32)
        res_c = tf.zeros([x.shape[0]], dtype=tf.float32)
        x1 = tf.identity(im2)
        x0 = tf.reshape(im2, [bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * tf.random.uniform(x1.shape, dtype=tf.float32) - 1
                    min_res = tf.minimum(res2, self.eps * tf.ones_like(res2))
                    min_res = tf.reshape(min_res, [-1] + [1]*self.ndims)
                    t_norm = tf.reduce_max(tf.abs(tf.reshape(t, [t.shape[0], -1])), axis=1, keepdims=True)
                    t_norm = tf.reshape(t_norm, [-1] + [1]*self.ndims)
                    x1 = im2 + min_res * t / t_norm * 0.5
                elif self.norm == 'L2':
                    t = tf.random.normal(x1.shape, dtype=tf.float32)
                    min_res = tf.minimum(res2, self.eps * tf.ones_like(res2))
                    min_res = tf.reshape(min_res, [-1] + [1]*self.ndims)
                    t_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(t, [t.shape[0], -1])), axis=-1))
                    t_norm = tf.reshape(t_norm, [t.shape[0]] + [1]*self.ndims)
                    x1 = im2 + min_res * t / t_norm * 0.5
                elif self.norm == 'L1':
                    t = tf.random.normal(x1.shape, dtype=tf.float32)
                    min_res = tf.minimum(res2, self.eps * tf.ones_like(res2))
                    min_res = tf.reshape(min_res, [-1] + [1]*self.ndims)
                    t_norm = tf.reduce_sum(tf.abs(tf.reshape(t, [t.shape[0], -1])), axis=-1)
                    t_norm = tf.reshape(t_norm, [t.shape[0]] + [1]*self.ndims)
                    x1 = im2 + min_res * t / t_norm / 2

                x1 = tf.clip_by_value(x1, 0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                if is_targeted:
                    df, dg = self.get_diff_logits_grads_batch_targeted(x1, la2, la_target2)
                else:
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)
                
                if self.norm == 'Linf':
                    dg_sum = tf.reduce_sum(tf.abs(tf.reshape(dg, [dg.shape[0], dg.shape[1], -1])), axis=-1)
                    dist1 = tf.abs(df) / (1e-12 + dg_sum)
                elif self.norm == 'L2':
                    dg_sum = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(dg, [dg.shape[0], dg.shape[1], -1])), axis=-1))
                    dist1 = tf.abs(df) / (1e-12 + dg_sum)
                elif self.norm == 'L1':
                    dg_max = tf.reduce_max(tf.abs(tf.reshape(dg, [df.shape[0], df.shape[1], -1])), axis=2)
                    dist1 = tf.abs(df) / (1e-12 + dg_max)
                else:
                    raise ValueError('norm not supported')
                
                ind = tf.argmin(dist1, axis=1)
                ind = tf.cast(ind, tf.int32)  # Ensure ind is int32
                dg2 = tf.gather_nd(dg, tf.stack([u1, ind], axis=1))
                df_selected = tf.gather_nd(df, tf.stack([u1, ind], axis=1))
                # Reshape both to same dimension for multiplication
                dg2_flat = tf.reshape(dg2, [bs, -1])
                x1_flat = tf.reshape(x1, [bs, -1])
                b = -df_selected + tf.reduce_sum(dg2_flat * x1_flat, axis=-1)
                w = dg2_flat
                if self.norm == 'Linf':
                    d3 = projection_linf(
                        tf.concat([x1_flat, x0], 0),
                        tf.concat([w, w], 0),
                        tf.concat([b, b], 0))
                elif self.norm == 'L2':
                    d3 = projection_l2(
                        tf.concat([x1_flat, x0], 0),
                        tf.concat([w, w], 0),
                        tf.concat([b, b], 0))
                elif self.norm == 'L1':
                    d3 = projection_l1(
                        tf.concat([x1_flat, x0], 0),
                        tf.concat([w, w], 0),
                        tf.concat([b, b], 0))
                
                d1 = tf.reshape(d3[:bs], x1.shape)
                d2 = tf.reshape(d3[-bs:], x1.shape)
                
                if self.norm == 'Linf':
                    a0 = tf.reduce_max(tf.abs(d3), axis=1, keepdims=True)
                    a0 = tf.reshape(a0, [-1] + [1]*self.ndims)
                elif self.norm == 'L2':
                    a0 = tf.sqrt(tf.reduce_sum(tf.square(d3), axis=1, keepdims=True))
                    a0 = tf.reshape(a0, [-1] + [1]*self.ndims)
                elif self.norm == 'L1':
                    a0 = tf.reduce_sum(tf.abs(d3), axis=1, keepdims=True)
                    a0 = tf.reshape(a0, [-1] + [1]*self.ndims)
                
                a0 = tf.maximum(a0, 1e-8 * tf.ones_like(a0))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                
                alpha = tf.minimum(
                    tf.maximum(a1 / (a1 + a2), tf.zeros_like(a1)),
                    self.alpha_max * tf.ones_like(a1))
                
                x1 = tf.clip_by_value(
                    (x1 + self.eta * d1) * (1 - alpha) + (im2 + d2 * self.eta) * alpha,
                    0.0, 1.0)

                is_adv = tf.not_equal(self._get_predicted_label(x1), la2)

                if tf.reduce_sum(tf.cast(is_adv, tf.int32)) > 0:
                    ind_adv = tf.where(is_adv)
                    ind_adv = tf.squeeze(ind_adv)
                    ind_adv = self.check_shape(ind_adv)
                    
                    x1_adv = tf.gather(x1, ind_adv)
                    im2_adv = tf.gather(im2, ind_adv)
                    
                    if self.norm == 'Linf':
                        t = tf.reduce_max(tf.abs(tf.reshape(x1_adv - im2_adv, [ind_adv.shape[0], -1])), axis=1)
                    elif self.norm == 'L2':
                        t = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(x1_adv - im2_adv, [ind_adv.shape[0], -1])), axis=-1))
                    elif self.norm == 'L1':
                        t = tf.reduce_sum(tf.abs(tf.reshape(x1_adv - im2_adv, [ind_adv.shape[0], -1])), axis=-1)
                    
                    # Update adv and res2 using numpy for complex indexing
                    adv_np = adv.numpy()
                    res2_np = res2.numpy()
                    x1_np = x1.numpy()
                    im2_np = im2.numpy()
                    ind_adv_np = ind_adv.numpy()
                    t_np = t.numpy()
                    
                    mask = t_np < res2_np[ind_adv_np]
                    for i, idx in enumerate(ind_adv_np):
                        if mask[i]:
                            adv_np[idx] = x1_np[idx]
                            res2_np[idx] = t_np[i]
                        x1_np[idx] = im2_np[idx] + (x1_np[idx] - im2_np[idx]) * self.beta
                    
                    adv = tf.constant(adv_np, dtype=tf.float32)
                    res2 = tf.constant(res2_np, dtype=tf.float32)
                    x1 = tf.constant(x1_np, dtype=tf.float32)

                counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(tf.reduce_sum(tf.cast(ind_succ, tf.float32)).numpy(), corr_classified.numpy()) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        # Update res_c and adv_c using numpy for complex indexing
        res_c_np = res_c.numpy()
        adv_c_np = adv_c.numpy()
        pred_indices_np = pred_indices.numpy()
        res2_np = res2.numpy()
        ind_succ_np = ind_succ.numpy()
        adv_np = adv.numpy()
        
        res_c_np[pred_indices_np] = res2_np * ind_succ_np.astype(float) + 1e10 * (1 - ind_succ_np.astype(float))
        
        ind_succ_indices = np.where(ind_succ_np)[0]
        for idx in ind_succ_indices:
            adv_c_np[pred_indices_np[idx]] = adv_np[idx]
        
        adv_c = tf.constant(adv_c_np, dtype=tf.float32)

        return adv_c

    def perturb(self, x, y):
        # Device handling - TensorFlow automatically handles device placement
        adv = tf.identity(x)
        
        output = self._predict_fn(x)
        acc = tf.equal(tf.argmax(output, axis=1), y)

        startt = time.time()

        tf.random.set_seed(self.seed)

        if not self.targeted:
            for counter in range(self.n_restarts):
                ind_to_fool = tf.where(acc)
                ind_to_fool = tf.squeeze(ind_to_fool)
                if len(ind_to_fool.shape) == 0: 
                    ind_to_fool = tf.expand_dims(ind_to_fool, 0)
                
                if tf.size(ind_to_fool) != 0:
                    x_to_fool = tf.gather(x, ind_to_fool)
                    y_to_fool = tf.gather(y, ind_to_fool)
                    adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0), is_targeted=False)

                    output_curr = self._predict_fn(adv_curr)
                    acc_curr = tf.equal(tf.argmax(output_curr, axis=1), y_to_fool)
                    
                    if self.norm == 'Linf':
                        res = tf.reduce_max(tf.abs(tf.reshape(x_to_fool - adv_curr, [x_to_fool.shape[0], -1])), axis=1)
                    elif self.norm == 'L2':
                        res = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(x_to_fool - adv_curr, [x_to_fool.shape[0], -1])), axis=-1))
                    elif self.norm == 'L1':
                        res = tf.reduce_sum(tf.abs(tf.reshape(x_to_fool - adv_curr, [x_to_fool.shape[0], -1])), axis=-1)
                    
                    acc_curr = tf.logical_or(acc_curr, res > self.eps)

                    ind_curr = tf.where(tf.logical_not(acc_curr))
                    ind_curr = tf.squeeze(ind_curr)
                    
                    # Update acc and adv using numpy for complex indexing
                    if tf.size(ind_curr) > 0:
                        acc_np = acc.numpy()
                        adv_np = adv.numpy()
                        ind_to_fool_np = ind_to_fool.numpy()
                        ind_curr_np = ind_curr.numpy()
                        adv_curr_np = adv_curr.numpy()
                        
                        for idx in ind_curr_np:
                            acc_np[ind_to_fool_np[idx]] = False
                            adv_np[ind_to_fool_np[idx]] = adv_curr_np[idx]
                        
                        acc = tf.constant(acc_np, dtype=tf.bool)
                        adv = tf.constant(adv_np, dtype=tf.float32)

                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                            counter, tf.reduce_mean(tf.cast(acc, tf.float32)).numpy(), self.eps, time.time() - startt))

        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    ind_to_fool = tf.where(acc)
                    ind_to_fool = tf.squeeze(ind_to_fool)
                    if len(ind_to_fool.shape) == 0: 
                        ind_to_fool = tf.expand_dims(ind_to_fool, 0)
                    
                    if tf.size(ind_to_fool) != 0:
                        x_to_fool = tf.gather(x, ind_to_fool)
                        y_to_fool = tf.gather(y, ind_to_fool)
                        adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0), is_targeted=True)

                        output_curr = self._predict_fn(adv_curr)
                        acc_curr = tf.equal(tf.argmax(output_curr, axis=1), y_to_fool)
                        
                        if self.norm == 'Linf':
                            res = tf.reduce_max(tf.abs(tf.reshape(x_to_fool - adv_curr, [x_to_fool.shape[0], -1])), axis=1)
                        elif self.norm == 'L2':
                            res = tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(x_to_fool - adv_curr, [x_to_fool.shape[0], -1])), axis=-1))
                        elif self.norm == 'L1':
                            res = tf.reduce_sum(tf.abs(tf.reshape(x_to_fool - adv_curr, [x_to_fool.shape[0], -1])), axis=-1)
                        
                        acc_curr = tf.logical_or(acc_curr, res > self.eps)

                        ind_curr = tf.where(tf.logical_not(acc_curr))
                        ind_curr = tf.squeeze(ind_curr)
                        
                        # Update acc and adv using numpy for complex indexing
                        if tf.size(ind_curr) > 0:
                            acc_np = acc.numpy()
                            adv_np = adv.numpy()
                            ind_to_fool_np = ind_to_fool.numpy()
                            ind_curr_np = ind_curr.numpy()
                            adv_curr_np = adv_curr.numpy()
                            
                            for idx in ind_curr_np:
                                acc_np[ind_to_fool_np[idx]] = False
                                adv_np[ind_to_fool_np[idx]] = adv_curr_np[idx]
                            
                            acc = tf.constant(acc_np, dtype=tf.bool)
                            adv = tf.constant(adv_np, dtype=tf.float32)

                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, tf.reduce_mean(tf.cast(acc, tf.float32)).numpy(), self.eps, time.time() - startt))

        return adv