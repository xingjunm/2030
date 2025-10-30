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

import paddle

from .fab_projections import projection_linf, projection_l2, projection_l1

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
        """ FAB-attack implementation in paddlepaddle """

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
        self.device = device  # Not used in PaddlePaddle (exemption #3)
        self.n_target_classes = n_target_classes

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

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

        # No device setting needed in PaddlePaddle (exemption #3)
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)

        x = x.detach().clone().astype('float32')

        y_pred = self._get_predicted_label(x)
        if y is None:
            y = y_pred.detach().clone().astype('int64')
        else:
            y = y.detach().clone().astype('int64')
        pred = y_pred == y
        corr_classified = pred.astype('float32').sum()
        if self.verbose:
            print('Clean accuracy: {:.2%}'.format(pred.astype('float32').mean()))
        if pred.sum() == 0:
            return x
        pred = self.check_shape(paddle.nonzero(pred).squeeze())

        if is_targeted:
            output = self._predict_fn(x)
            la_target = paddle.sort(output, axis=-1)[1][:, -self.target_class]
            la_target2 = la_target[pred].detach().clone()

        startt = time.time()
        # runs the attack only on correctly classified points
        im2 = x[pred].detach().clone()
        la2 = y[pred].detach().clone()
        if len(im2.shape) == self.ndims:
            im2 = im2.unsqueeze(0)
        bs = im2.shape[0]
        u1 = paddle.arange(bs)
        adv = im2.clone()
        adv_c = x.clone()
        res2 = 1e10 * paddle.ones([bs], dtype='float32')
        res_c = paddle.zeros([x.shape[0]], dtype='float32')
        x1 = im2.clone()
        x0 = im2.clone().reshape([bs, -1])
        counter_restarts = 0

        while counter_restarts < 1:
            if use_rand_start:
                if self.norm == 'Linf':
                    t = 2 * paddle.rand(x1.shape, dtype='float32') - 1
                    x1 = im2 + (paddle.minimum(res2,
                                          self.eps * paddle.ones(res2.shape, dtype='float32')
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.reshape([t.shape[0], -1]).abs()
                                         .max(axis=1, keepdim=True)[0]
                                         .reshape([-1, *[1]*self.ndims])) * .5
                elif self.norm == 'L2':
                    t = paddle.randn(x1.shape, dtype='float32')
                    x1 = im2 + (paddle.minimum(res2,
                                          self.eps * paddle.ones(res2.shape, dtype='float32')
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / ((t ** 2)
                                         .reshape([t.shape[0], -1])
                                         .sum(axis=-1)
                                         .sqrt()
                                         .reshape([t.shape[0], *[1]*self.ndims])) * .5
                elif self.norm == 'L1':
                    t = paddle.randn(x1.shape, dtype='float32')
                    x1 = im2 + (paddle.minimum(res2,
                                          self.eps * paddle.ones(res2.shape, dtype='float32')
                                          ).reshape([-1, *[1]*self.ndims])
                                ) * t / (t.abs().reshape([t.shape[0], -1])
                                         .sum(axis=-1)
                                         .reshape([t.shape[0], *[1]*self.ndims])) / 2

                x1 = paddle.clip(x1, 0.0, 1.0)

            counter_iter = 0
            while counter_iter < self.n_iter:
                with paddle.no_grad():
                    if is_targeted:
                        df, dg = self.get_diff_logits_grads_batch_targeted(x1, la2, la_target2)
                    else:
                        df, dg = self.get_diff_logits_grads_batch(x1, la2)
                    if self.norm == 'Linf':
                        dist1 = df.abs() / (1e-12 +
                                            dg.abs()
                                            .reshape([dg.shape[0], dg.shape[1], -1])
                                            .sum(axis=-1))
                    elif self.norm == 'L2':
                        dist1 = df.abs() / (1e-12 + (dg ** 2)
                                            .reshape([dg.shape[0], dg.shape[1], -1])
                                            .sum(axis=-1).sqrt())
                    elif self.norm == 'L1':
                        dist1 = df.abs() / (1e-12 + dg.abs().reshape(
                            [df.shape[0], df.shape[1], -1]).max(axis=2)[0])
                    else:
                        raise ValueError('norm not supported')
                    ind = dist1.min(axis=1)[1]
                    dg2 = dg[u1, ind]
                    b = (- df[u1, ind] + (dg2 * x1).reshape([x1.shape[0], -1])
                                         .sum(axis=-1))
                    w = dg2.reshape([bs, -1])

                    if self.norm == 'Linf':
                        d3 = projection_linf(
                            paddle.concat([x1.reshape([bs, -1]), x0], axis=0),
                            paddle.concat([w, w], axis=0),
                            paddle.concat([b, b], axis=0))
                    elif self.norm == 'L2':
                        d3 = projection_l2(
                            paddle.concat([x1.reshape([bs, -1]), x0], axis=0),
                            paddle.concat([w, w], axis=0),
                            paddle.concat([b, b], axis=0))
                    elif self.norm == 'L1':
                        d3 = projection_l1(
                            paddle.concat([x1.reshape([bs, -1]), x0], axis=0),
                            paddle.concat([w, w], axis=0),
                            paddle.concat([b, b], axis=0))
                    d1 = paddle.reshape(d3[:bs], x1.shape)
                    d2 = paddle.reshape(d3[-bs:], x1.shape)
                    if self.norm == 'Linf':
                        a0 = d3.abs().max(axis=1, keepdim=True)[0]\
                            .reshape([-1, *[1]*self.ndims])
                    elif self.norm == 'L2':
                        a0 = (d3 ** 2).sum(axis=1, keepdim=True).sqrt()\
                            .reshape([-1, *[1]*self.ndims])
                    elif self.norm == 'L1':
                        a0 = d3.abs().sum(axis=1, keepdim=True)\
                            .reshape([-1, *[1]*self.ndims])
                    a0 = paddle.maximum(a0, 1e-8 * paddle.ones(
                        a0.shape, dtype='float32'))
                    a1 = a0[:bs]
                    a2 = a0[-bs:]
                    alpha = paddle.minimum(paddle.maximum(a1 / (a1 + a2),
                                                paddle.zeros(a1.shape, dtype='float32')),
                                      self.alpha_max * paddle.ones(a1.shape, dtype='float32'))
                    x1 = ((x1 + self.eta * d1) * (1 - alpha) +
                          (im2 + d2 * self.eta) * alpha)
                    x1 = paddle.clip(x1, 0.0, 1.0)

                    is_adv = self._get_predicted_label(x1) != la2

                    if is_adv.sum() > 0:
                        ind_adv = paddle.nonzero(is_adv).squeeze()
                        ind_adv = self.check_shape(ind_adv)
                        if self.norm == 'Linf':
                            t = (x1[ind_adv] - im2[ind_adv]).reshape(
                                [ind_adv.shape[0], -1]).abs().max(axis=1)[0]
                        elif self.norm == 'L2':
                            t = ((x1[ind_adv] - im2[ind_adv]) ** 2)\
                                .reshape([ind_adv.shape[0], -1]).sum(axis=-1).sqrt()
                        elif self.norm == 'L1':
                            t = (x1[ind_adv] - im2[ind_adv])\
                                .abs().reshape([ind_adv.shape[0], -1]).sum(axis=-1)
                        adv[ind_adv] = x1[ind_adv] * (t < res2[ind_adv])\
                            .astype('float32').reshape([-1, *[1]*self.ndims]) + adv[ind_adv]\
                            * (t >= res2[ind_adv]).astype('float32').reshape(
                            [-1, *[1]*self.ndims])
                        res2[ind_adv] = t * (t < res2[ind_adv]).astype('float32')\
                            + res2[ind_adv] * (t >= res2[ind_adv]).astype('float32')
                        x1[ind_adv] = im2[ind_adv] + (
                            x1[ind_adv] - im2[ind_adv]) * self.beta

                counter_iter += 1

            counter_restarts += 1

        ind_succ = res2 < 1e10
        if self.verbose:
            print('success rate: {:.0f}/{:.0f}'
                  .format(ind_succ.astype('float32').sum(), corr_classified) +
                  ' (on correctly classified points) in {:.1f} s'
                  .format(time.time() - startt))

        res_c[pred] = res2 * ind_succ.astype('float32') + 1e10 * (1 - ind_succ.astype('float32'))
        ind_succ = self.check_shape(paddle.nonzero(ind_succ).squeeze())
        adv_c[pred[ind_succ]] = adv[ind_succ].clone()

        return adv_c

    def perturb(self, x, y):
        # No device setting needed in PaddlePaddle (exemption #3)
        adv = x.clone()
        with paddle.no_grad():
            acc = self._predict_fn(x).max(1)[1] == y

            startt = time.time()

            paddle.seed(self.seed)

            if not self.targeted:
                for counter in range(self.n_restarts):
                    ind_to_fool = paddle.nonzero(acc).squeeze()
                    if len(ind_to_fool.shape) == 0: 
                        ind_to_fool = ind_to_fool.unsqueeze(0)
                    if ind_to_fool.numel() != 0:
                        x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                        adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0), is_targeted=False)

                        acc_curr = self._predict_fn(adv_curr).max(1)[1] == y_to_fool
                        if self.norm == 'Linf':
                            res = (x_to_fool - adv_curr).abs().reshape([x_to_fool.shape[0], -1]).max(1)[0]
                        elif self.norm == 'L2':
                            res = ((x_to_fool - adv_curr) ** 2).reshape([x_to_fool.shape[0], -1]).sum(axis=-1).sqrt()
                        elif self.norm == 'L1':
                            res = (x_to_fool - adv_curr).abs().reshape([x_to_fool.shape[0], -1]).sum(-1)
                        acc_curr = paddle.maximum(acc_curr, res > self.eps)

                        ind_curr = paddle.nonzero(acc_curr == 0).squeeze()
                        acc[ind_to_fool[ind_curr]] = 0
                        adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                        if self.verbose:
                            print('restart {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                counter, acc.astype('float32').mean(), self.eps, time.time() - startt))

            else:
                for target_class in range(2, self.n_target_classes + 2):
                    self.target_class = target_class
                    for counter in range(self.n_restarts):
                        ind_to_fool = paddle.nonzero(acc).squeeze()
                        if len(ind_to_fool.shape) == 0: 
                            ind_to_fool = ind_to_fool.unsqueeze(0)
                        if ind_to_fool.numel() != 0:
                            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                            adv_curr = self.attack_single_run(x_to_fool, y_to_fool, use_rand_start=(counter > 0), is_targeted=True)

                            acc_curr = self._predict_fn(adv_curr).max(1)[1] == y_to_fool
                            if self.norm == 'Linf':
                                res = (x_to_fool - adv_curr).abs().reshape([x_to_fool.shape[0], -1]).max(1)[0]
                            elif self.norm == 'L2':
                                res = ((x_to_fool - adv_curr) ** 2).reshape([x_to_fool.shape[0], -1]).sum(axis=-1).sqrt()
                            elif self.norm == 'L1':
                                res = (x_to_fool - adv_curr).abs().reshape([x_to_fool.shape[0], -1]).sum(-1)
                            acc_curr = paddle.maximum(acc_curr, res > self.eps)

                            ind_curr = paddle.nonzero(acc_curr == 0).squeeze()
                            acc[ind_to_fool[ind_curr]] = 0
                            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()

                            if self.verbose:
                                print('restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                                    counter, self.target_class, acc.astype('float32').mean(), self.eps, time.time() - startt))

        return adv