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

import mindspore as ms
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import grad as ms_grad
from mindspore.ops import composite as C

from .fab_base import zero_gradients, FABAttack


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
        """ FAB-attack implementation in mindspore """

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
        outputs = self._predict_fn(x)
        y = outputs.argmax(1)
        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        # Clone the input tensor
        im = imgs.copy()
        
        # Define gradient function for each logit
        def logit_grad_fn(x, idx):
            """Compute gradient of a specific logit with respect to input"""
            y = self.predict(x)
            # Create a one-hot vector for the specific logit
            grad_mask = ops.zeros_like(y)
            grad_mask[:, idx] = 1.0
            # Return the weighted sum of logits
            return ops.reduce_sum(y * grad_mask)
        
        # Get the logits
        y = self.predict(im)
        n_classes = y.shape[-1]
        batch_size = imgs.shape[0]
        
        # Initialize gradient tensor
        g2 = ops.zeros((n_classes, *imgs.shape), dtype=imgs.dtype)
        
        # Compute gradients for each class
        for counter in range(n_classes):
            # Create gradient function for this specific logit
            grad_fn = ms_grad(lambda x: logit_grad_fn(x, counter))
            # Compute gradient
            grad = grad_fn(im)
            g2[counter] = grad
        
        # Transpose to match PyTorch output shape
        g2 = ops.transpose(g2, (1, 0, 2, 3, 4))
        
        # Get the predictions
        y2 = y
        
        # Compute differences from the true label logits
        arange_batch = mnp.arange(batch_size)
        df = y2 - ops.expand_dims(y2[arange_batch, la], 1)
        dg = g2 - ops.expand_dims(g2[arange_batch, la], 1)
        
        # Set the diagonal to a large value to avoid selecting the true label
        df[arange_batch, la] = 1e10
        
        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = mnp.arange(imgs.shape[0])
        im = imgs.copy()
        
        # Define the targeted loss function
        def targeted_loss_fn(x):
            y = self.predict(x)
            # Compute the difference between true label and target label logits
            diffy = -(y[u, la] - y[u, la_target])
            return ops.reduce_sum(diffy)
        
        # Compute gradient of the targeted loss
        grad_fn = ms_grad(targeted_loss_fn)
        graddiffy = grad_fn(im)
        
        # Compute the logit differences
        y = self.predict(im)
        diffy = -(y[u, la] - y[u, la_target])
        
        # Prepare output tensors
        df = ops.expand_dims(diffy, 1)
        dg = ops.expand_dims(graddiffy, 1)
        
        return df, dg