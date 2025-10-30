import paddle
import numpy as np


def adv_check_and_update(X_cur, logits, y, not_correct, X_adv):
    adv_pred = paddle.argmax(logits, axis=1)
    nc = (adv_pred != y)
    not_correct += nc.astype('int64')
    X_adv[nc] = X_cur[nc].detach()
    return X_adv, not_correct


def one_hot_tensor(y_batch_tensor, num_classes):
    # Create a float32 tensor (following exemption #2 for default float32)
    y_tensor = paddle.zeros([y_batch_tensor.shape[0], num_classes], dtype='float32')
    # Use paddle's indexing to set the one-hot values
    indices = paddle.arange(len(y_batch_tensor))
    y_tensor[indices, y_batch_tensor] = 1.0
    return y_tensor