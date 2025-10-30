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

import tensorflow as tf

from attacks.fab_base import FABAttack


def zero_gradients(x):
    """Zero out the gradients of a tensor in TensorFlow.
    In TensorFlow with GradientTape, gradients are computed explicitly,
    so this function is not needed in the same way as PyTorch."""
    pass  # No-op in TensorFlow as gradients are managed by GradientTape


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
        """ FAB-attack implementation in tensorflow """

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
        y = tf.argmax(outputs, axis=1)
        return y

    def get_diff_logits_grads_batch(self, imgs, la):
        # Clone the images and ensure they require gradients
        im = tf.Variable(imgs, trainable=True)
        
        # Initialize gradient storage
        batch_size = imgs.shape[0]
        num_classes = None
        
        # Ensure la is int32
        la = tf.cast(la, tf.int32)
        
        # First forward pass to get number of classes
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(im)
            y = self.predict(im)
            num_classes = y.shape[-1]
        
        # Storage for gradients
        g2 = tf.zeros([num_classes, *imgs.shape], dtype=tf.float32)
        
        # Compute gradients for each class
        for counter in range(num_classes):
            with tf.GradientTape() as tape:
                tape.watch(im)
                y = self.predict(im)
                # Select the specific class output
                y_class = y[:, counter]
                
            # Compute gradients
            grad = tape.gradient(y_class, im)
            
            # Store gradients - need to handle the indexing properly
            # Convert to numpy for assignment then back to tensor
            g2_np = g2.numpy()
            g2_np[counter] = grad.numpy()
            g2 = tf.constant(g2_np, dtype=tf.float32)
        
        # Transpose to match expected shape
        g2 = tf.transpose(g2, [1, 0, 2, 3, 4])
        
        # Compute logits
        y2 = y
        
        # Compute differences from the true class
        la_indices = tf.stack([tf.range(batch_size, dtype=tf.int32), la], axis=1)
        y_la = tf.gather_nd(y2, la_indices)
        df = y2 - tf.expand_dims(y_la, 1)
        
        # Get gradients differences
        g_la = tf.gather_nd(g2, la_indices)
        dg = g2 - tf.expand_dims(g_la, 1)
        
        # Set diagonal to large value
        df_np = df.numpy()
        for i in range(batch_size):
            df_np[i, la[i].numpy()] = 1e10
        df = tf.constant(df_np, dtype=tf.float32)
        
        return df, dg

    def get_diff_logits_grads_batch_targeted(self, imgs, la, la_target):
        u = tf.range(imgs.shape[0], dtype=tf.int32)
        im = tf.Variable(imgs, trainable=True)
        
        # Ensure la and la_target are int32
        la = tf.cast(la, tf.int32)
        la_target = tf.cast(la_target, tf.int32)
        
        with tf.GradientTape() as tape:
            tape.watch(im)
            y = self.predict(im)
            
            # Get the difference between true class and target class
            y_la = tf.gather_nd(y, tf.stack([u, la], axis=1))
            y_la_target = tf.gather_nd(y, tf.stack([u, la_target], axis=1))
            diffy = -(y_la - y_la_target)
            sumdiffy = tf.reduce_sum(diffy)
        
        # Compute gradient
        graddiffy = tape.gradient(sumdiffy, im)
        
        df = tf.expand_dims(diffy, 1)
        dg = tf.expand_dims(graddiffy, 1)
        
        return df, dg