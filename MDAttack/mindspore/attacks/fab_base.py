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

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor

# Local imports - assume fab_projections will be available
from .fab_projections import projection_linf, projection_l2, projection_l1

DEFAULT_EPS_DICT_BY_NORM = {'Linf': .3, 'L2': 1., 'L1': 5.0}


def zero_gradients(x):
    """
    Simple implementation of zero_gradients function for MindSpore
    """
    if isinstance(x, Tensor):
        if x.grad is not None:
            x.grad = ops.zeros_like(x.grad)


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
        """ FAB-attack implementation in mindspore """

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
        self.device = device  # Not used in MindSpore (device setting handled differently)
        self.n_target_classes = n_target_classes

    def check_shape(self, x):
        return x if len(x.shape) > 0 else ops.expand_dims(x, 0)

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
        :param is_targeted    True if we use targeted version. Targeted class is assigned by `self.target_class`
        """

        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.astype(ms.float32)

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.astype(ms.int32)
        else:
            y = y.astype(ms.int32)
        
        pred = ops.equal(y_pred, y)
        corr_classified = ops.cast(pred, ms.float32).sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(ops.cast(pred, ms.float32).mean()))
        if pred.sum() == 0:
            return x
        
        # Get indices where pred is True
        pred_indices = ops.nonzero(pred)
        if pred_indices.size > 0:
            pred = self.check_shape(pred_indices.squeeze())
        else:
            pred = Tensor([], ms.int32)

        if is_targeted:
            output = self._predict_fn(x)
            la_target = ops.sort(output, axis=-1)[1][:, -self.target_class]
            la_target2 = la_target[pred]

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred]
        la2 = y[pred]
        if len(im2.shape) == self.ndims:
            im2 = ops.expand_dims(im2, 0)
        bs = im2.shape[0]
        u1 = mnp.arange(bs)
        adv = im2.copy()
        adv_c = x.copy()
        res2 = 1e10 * ops.ones((bs,), ms.float32)
        res_c = ops.zeros((x.shape[0],), ms.float32)
        x1 = im2.copy()
        x0 = im2.copy().reshape((bs, -1))
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * ops.rand(x1.shape, ms.float32) - 1
                    res_eps = ops.minimum(res2, self.eps * ops.ones(res2.shape, ms.float32))
                    res_eps = res_eps.reshape((-1, *[1]*self.ndims))
                    t_max = t.reshape((t.shape[0], -1)).abs().max(axis=1, keepdims=True)[0]
                    t_max = t_max.reshape((-1, *[1]*self.ndims))
                    x1 = im2 + res_eps * t / t_max * .5
                elif self.norm == 'L2':
                    t = ops.randn(x1.shape, ms.float32)
                    res_eps = ops.minimum(res2, self.eps * ops.ones(res2.shape, ms.float32))
                    res_eps = res_eps.reshape((-1, *[1]*self.ndims))
                    t_norm = (t ** 2).reshape(t.shape[0], -1).sum(axis=-1).sqrt()
                    t_norm = t_norm.reshape(t.shape[0], *[1]*self.ndims)
                    x1 = im2 + res_eps * t / t_norm * .5
                elif self.norm == 'L1':
                    t = ops.randn(x1.shape, ms.float32)
                    res_eps = ops.minimum(res2, self.eps * ops.ones(res2.shape, ms.float32))
                    res_eps = res_eps.reshape((-1, *[1]*self.ndims))
                    t_norm = t.abs().reshape(t.shape[0], -1).sum(axis=-1)
                    t_norm = t_norm.reshape(t.shape[0], *[1]*self.ndims)
                    x1 = im2 + res_eps * t / t_norm / 2

                x1 = ops.clip_by_value(x1, 0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                # Get gradients
                if is_targeted:
                    df, dg = self.get_diff_logits_grads_batch_targeted(x1, la2, la_target2)
                else:
                    df, dg = self.get_diff_logits_grads_batch(x1, la2)
                
                if self.norm == 'Linf':
                    dist1 = df.abs() / (1e-12 + 
                                       dg.abs()
                                       .reshape(dg.shape[0], dg.shape[1], -1)
                                       .sum(axis=-1))
                elif self.norm == 'L2':
                    dist1 = df.abs() / (1e-12 + (dg ** 2)
                                       .reshape(dg.shape[0], dg.shape[1], -1)
                                       .sum(axis=-1).sqrt())
                elif self.norm == 'L1':
                    dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                        (df.shape[0], df.shape[1], -1)).max(axis=2))
                else:
                    raise ValueError('norm not supported')
                
                ind = dist1.min(axis=1)
                dg2 = dg[u1, ind]
                b = (- df[u1, ind] + (dg2 * x1).reshape(x1.shape[0], -1)
                                    .sum(axis=-1))
                w = dg2.reshape((bs, -1))

                if self.norm == 'Linf':
                    d3 = projection_linf(
                        ops.concat((x1.reshape((bs, -1)), x0), 0),
                        ops.concat((w, w), 0),
                        ops.concat((b, b), 0))
                elif self.norm == 'L2':
                    d3 = projection_l2(
                        ops.concat((x1.reshape((bs, -1)), x0), 0),
                        ops.concat((w, w), 0),
                        ops.concat((b, b), 0))
                elif self.norm == 'L1':
                    d3 = projection_l1(
                        ops.concat((x1.reshape((bs, -1)), x0), 0),
                        ops.concat((w, w), 0),
                        ops.concat((b, b), 0))
                
                d1 = d3[:bs].reshape(x1.shape)
                d2 = d3[-bs:].reshape(x1.shape)
                
                if self.norm == 'Linf':
                    a0 = d3.abs().max(axis=1, keepdims=True).reshape(-1, *[1]*self.ndims)
                elif self.norm == 'L2':
                    a0 = (d3 ** 2).sum(axis=1, keepdims=True).sqrt().reshape(-1, *[1]*self.ndims)
                elif self.norm == 'L1':
                    a0 = d3.abs().sum(axis=1, keepdims=True).reshape(-1, *[1]*self.ndims)
                
                a0 = ops.maximum(a0, 1e-8 * ops.ones(a0.shape, ms.float32))
                a1 = a0[:bs]
                a2 = a0[-bs:]
                
                alpha = ops.minimum(ops.maximum(a1 / (a1 + a2),
                                              ops.zeros(a1.shape, ms.float32)),
                                  self.alpha_max * ops.ones(a1.shape, ms.float32))
                x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                      (im2 + d2 * self.eta) * alpha)
                x1 = ops.clip_by_value(x1, 0.0, 1.0)

                is_adv = ops.not_equal(self._get_predicted_label(x1), la2)

                if is_adv.sum() > 0:
                    nonzero_result = ops.nonzero(is_adv)
                    if nonzero_result.size > 0:
                        ind_adv = self.check_shape(nonzero_result.squeeze())
                    else:
                        ind_adv = Tensor([], ms.int32)
                    
                    if self.norm == 'Linf':
                        t = (x1[ind_adv] - im2[ind_adv]).reshape(
                            (ind_adv.shape[0], -1)).abs().max(axis=1)
                    elif self.norm == 'L2':
                        t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                            .reshape(ind_adv.shape[0], -1).sum(axis=-1).sqrt()
                    elif self.norm == 'L1':
                        t = (x1[ind_adv] - im2[ind_adv])\
                            .abs().reshape(ind_adv.shape[0], -1).sum(axis=-1)
                    
                    t_cond = ops.cast(t < res2[ind_adv], ms.float32).reshape((-1, *[1]*self.ndims))
                    t_cond_inv = ops.cast(t >= res2[ind_adv], ms.float32).reshape((-1, *[1]*self.ndims))
                    
                    adv[ind_adv] = x1[ind_adv] * t_cond + adv[ind_adv] * t_cond_inv
                    res2[ind_adv] = t * ops.cast(t < res2[ind_adv], ms.float32) + \
                                   res2[ind_adv] * ops.cast(t >= res2[ind_adv], ms.float32)
                    x1[ind_adv] = im2[ind_adv] + (x1[ind_adv] - im2[ind_adv]) * self.beta

                counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ops.cast(ind_succ, ms.float32).sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ops.cast(ind_succ, ms.float32) + 1e10 * (1 - ops.cast(ind_succ, ms.float32))
        nonzero_result = ops.nonzero(ind_succ)
        if nonzero_result.size > 0:
            ind_succ = self.check_shape(nonzero_result.squeeze())
        else:
            ind_succ = Tensor([], ms.int32)
        adv_c[pred[ind_succ]] = adv[ind_succ]

        return adv_c

    def perturb(self, x, y):
        adv = x.copy()
        
        acc = ops.equal(self._predict_fn(x).argmax(1), y)

        startt = time.time()

        # Set random seed
        ms.set_seed(self.seed)

        if not self.targeted:
            for counter in range(self.n_restarts):
                nonzero_result = ops.nonzero(acc)
                if nonzero_result.size > 0:
                    ind_to_fool = nonzero_result.squeeze()
                else:
                    ind_to_fool = Tensor([], ms.int32)
                
                if ind_to_fool.size != 0:
                    x_to_fool, y_to_fool = x[ind_to_fool], y[ind_to_fool]
                    adv_curr = self.attack_single_run(x_to_fool, y_to_fool, 
                                                    use_rand_start=(counter > 0), 
                                                    is_targeted=False)

                    acc_curr = ops.equal(self._predict_fn(adv_curr).max(1), y_to_fool)
                    if self.norm == 'Linf':
                        res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).max(1)
                    elif self.norm == 'L2':
                        res = ((x_to_fool - adv_curr) ** 2).reshape(x_to_fool.shape[0], -1).sum(axis=-1).sqrt()
                    elif self.norm == 'L1':
                        res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).sum(-1)
                    acc_curr = ops.maximum(acc_curr, res > self.eps)

                    nonzero_result = ops.nonzero(ops.equal(acc_curr, 0))
                    if nonzero_result.size > 0:
                        ind_curr = nonzero_result.squeeze()
                    else:
                        ind_curr = Tensor([], ms.int32)
                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr]

                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                            counter, ops.cast(acc, ms.float32).mean(), self.eps, time.time() - startt))

        else:
            for target_class in range(2, self.n_target_classes + 2):
                self.target_class = target_class
                for counter in range(self.n_restarts):
                    nonzero_result = ops.nonzero(acc)
                    if nonzero_result[0].size > 0:
                        ind_to_fool = nonzero_result[0]
                    else:
                        ind_to_fool = Tensor([], ms.int32)
                    
                    if ind_to_fool.size != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool], y[ind_to_fool]
                        adv_curr = self.attack_single_run(x_to_fool, y_to_fool, 
                                                        use_rand_start=(counter > 0), 
                                                        is_targeted=True)

                        acc_curr = ops.equal(self._predict_fn(adv_curr).max(1), y_to_fool)
                        if self.norm == 'Linf':
                            res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).max(1)
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2).reshape(x_to_fool.shape[0], -1).sum(axis=-1).sqrt()
                        elif self.norm == 'L1':
                            res = (x_to_fool - adv_curr).abs().reshape(x_to_fool.shape[0], -1).sum(-1)
                        acc_curr = ops.maximum(acc_curr, res > self.eps)

                        nonzero_result = ops.nonzero(ops.equal(acc_curr, 0))
                        if nonzero_result[0].size > 0:
                            ind_curr = nonzero_result[0]
                        else:
                            ind_curr = Tensor([], ms.int32)
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr]

                        if self.verbose:
                            print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, self.target_class, ops.cast(acc, ms.float32).mean(), self.eps, time.time() - startt))

        return adv