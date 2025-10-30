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


def zero_gradients(x):
    """Zero out the gradients of a tensor."""
    if x.grad is not None:
        x.clear_gradient()


from .fab_base import FABAttack


class FABAttack_PT(FABAttack):
    """
    Fast Adaptive Boundary Attack (Linf, L2, L1)
    https://arxiv.org/abs/1907.02044
    
    :param predict:       forward pass function
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
            predict,
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

        self.predict = predict
        super().__init__(norm,
                         n_restarts,
                         n_iter,
                         eps,
                         alpha_max,
                         eta,
                         beta,
                         loss_fn,
                         verbose,
                         seed,
                         targeted,
                         device,
                         n_target_classes)

    def _predict_fn(self, x):
        return self.predict(x)

    def _get_predicted_label(self, x):
        with paddle.no_grad():
            outputs = self._predict_fn(x)
        _, y = paddle.max(outputs, axis=1), paddle.argmax(outputs, axis=1)
        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        im = imgs.clone()
        im.stop_gradient = False
        
        y = self.predict(im)

        g2 = paddle.zeros([y.shape[-1], *imgs.shape], dtype='float32')
        
        for counter in range(y.shape[-1]):
            if im.grad is not None:
                im.clear_gradient()
            
            # Create a one-hot gradient mask for the current class
            grad_outputs = paddle.zeros_like(y, dtype='float32')
            grad_outputs[:, counter] = 1.0
            
            # Compute gradient for this class
            grads = paddle.grad(
                outputs=[y],
                inputs=[im],
                grad_outputs=[grad_outputs],
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            
            if grads is not None:
                g2[counter] = grads
            else:
                g2[counter] = paddle.zeros_like(imgs, dtype='float32')

        g2 = paddle.transpose(g2, [1, 0, 2, 3, 4]).detach()
        y2 = y.detach()
        df = y2 - y2[paddle.arange(imgs.shape[0]), la].unsqueeze(1)
        dg = g2 - g2[paddle.arange(imgs.shape[0]), la].unsqueeze(1)
        df[paddle.arange(imgs.shape[0]), la] = 1e10

        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = paddle.arange(imgs.shape[0])
        im = imgs.clone()
        im.stop_gradient = False
        
        y = self.predict(im)
        diffy = -(y[u, la] - y[u, la_target])
        sumdiffy = diffy.sum()

        # Compute gradient
        grads = paddle.grad(
            outputs=[sumdiffy],
            inputs=[im],
            retain_graph=False,
            create_graph=False
        )[0]
        
        if grads is not None:
            graddiffy = grads
        else:
            graddiffy = paddle.zeros_like(imgs, dtype='float32')
            
        df = diffy.detach().unsqueeze(1)
        dg = graddiffy.unsqueeze(1)

        return df, dg