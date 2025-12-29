from __future__ import division, absolute_import, print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale, StandardScaler
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
STDEVS = {
    'mnist': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,\
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,\
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,\
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,\
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,\
            'cwi': 0.25, 'df': 0.25,\
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,\
            'sa': 0.3, 'sta': 0.3, 'hop': 0.3, 'zoo': 0.3
            },
    'cifar': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,\
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,\
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,\
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,\
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,\
            'cwi': 0.125, 'df': 0.125,\
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,\
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125, 'ap': 0.125
            },
    'imagenet': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,\
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,\
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,\
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,\
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,\
            'cwi': 0.125, 'df': 0.125,\
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,\
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125, 'ap': 0.125
            },
    'svhn': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,\
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,\
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,\
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,\
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,\
            'cwi': 0.125, 'df': 0.125,\
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,\
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125, 'ap': 0.125
            }
}

def flip(x, nb_diff):
    """
    Helper function for get_noisy_samples
    :param x:
    :param nb_diff:
    :return:
    """
    original_shape = x.shape
    x = np.copy(np.reshape(x, (-1,)))
    candidate_inds = np.where(x < 0.99)[0]
    #assert candidate_inds.shape[0] >= nb_diff
    inds = np.random.choice(candidate_inds, nb_diff)
    x[inds] = 1.

    return np.reshape(x, original_shape)


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    TODO
    :param X_test:
    :param X_test_adv:
    :param dataset:
    :param attack:
    :return:
    """
    X_test_noisy = np.minimum(
            np.maximum(
                X_test + np.random.normal(loc=0, scale=STDEVS[dataset][attack],
                                          size=X_test.shape),
                0
            ),
            1
        )
    X_test_noisy = X_test_noisy.astype(np.float32)

    return X_test_noisy

def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    model.set_train(True)
    # Get all modules in a flattened list, similar to PyTorch's model.modules()
    def get_all_modules(module):
        modules = []
        for name, cell in module.cells_and_names():
            modules.append(cell)
        return modules
    
    list_modules = get_all_modules(model)
    # MindSpore uses out_channels instead of out_features for Dense layers
    output_dim = list_modules[-1].out_channels
    
    softmax = ops.Softmax(axis=1)
    get_output = lambda data : softmax(model(data))
    X = Tensor(X, mindspore.float32)
    
    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            output[i * batch_size:(i + 1) * batch_size] = get_output(X[i * batch_size:(i + 1) * batch_size]).asnumpy()
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)

def get_deep_representations(model, X, batch_size=256, dataset='mnist'):
    model.set_train(False)
    X = Tensor(X, mindspore.float32)

    # RISK_INFO: [实现差异的潜在风险] - 自定义的 get_all_modules 函数用于模拟 PyTorch 的 model.modules()。对于复杂的非顺序模型，其模块遍历顺序可能与 PyTorch 不同，这可能导致错误的层被挂载（hooked）。
    # Get all modules in a flattened list, similar to PyTorch's model.modules()
    def get_all_modules(module):
        modules = [module]
        for cell in module.cells():
            if isinstance(cell, nn.SequentialCell):
                # Add the sequential itself
                modules.append(cell)
                # Add all its sub-modules
                for sub_cell in cell.cells():
                    modules.extend(get_all_modules(sub_cell))
            else:
                modules.append(cell)
        return modules
    
    list_modules = get_all_modules(model)
    
    # Find the last hidden layer (Dense or Conv2d) before the final layer
    last_hidden_idx = None
    for idx, module in enumerate(list_modules[-2::-1]):
        if isinstance(module, (nn.Dense, nn.Conv2d)):
            last_hidden_idx = len(list_modules) - idx - 2
            break
    
    if last_hidden_idx is None:
        raise ValueError("No suitable hidden layer found")
    
    # print(list_modules[last_hidden_idx])
    
    last_hidden_output = None
    def last_hidden_hook(cell, inputs, outputs):
        nonlocal last_hidden_output
        last_hidden_output = outputs
        return outputs
    
    # Register hook on the target layer
    handle = list_modules[last_hidden_idx].register_forward_hook(last_hidden_hook)
    
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))    
    output = []     # output = np.zeros(shape=(len(X), output_dim))
    for i in tqdm(range(n_batches)):
        model(X[i * batch_size:(i + 1) * batch_size])
        flatten_hidden_output = ops.Reshape()(last_hidden_output, (last_hidden_output.shape[0], -1))
        output.append(flatten_hidden_output.asnumpy())
    
    # Remove the hook after use
    handle.remove()
    
    output = np.vstack(output)
    return output


def score_point(tup):
    """
    TODO
    :param tup:
    :return:
    """
    x, kde = tup

    return kde.score_samples(np.reshape(x, (1, -1)))[0]


def score_samples(kdes, samples, preds, n_jobs=None):
    """
    TODO
    :param kdes:
    :param samples:
    :param preds:
    :param n_jobs:
    :return:
    """
    if n_jobs is not None:
        p = mp.Pool(n_jobs)
    else:
        p = mp.Pool()
    results = np.asarray(
        p.map(
            score_point,
            [(x, kdes[i]) for x, i in zip(samples, preds)]
        )
    )
    p.close()
    p.join()

    return results


def normalize(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]

def normalize_std(normal, adv, noisy):
    """
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    scaler = StandardScaler()
    total = scaler.fit_transform(np.concatenate((normal, adv, noisy)).reshape((-1,1)))
    total = total.reshape((-1,))
    # total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:], scaler


def train_lr(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    TODO
    :param densities_pos:
    :param densities_neg:
    :param uncerts_pos:
    :param uncerts_neg:
    :return:
    """
    values_neg = np.concatenate(
        (densities_neg.reshape((1, -1)),
         uncerts_neg.reshape((1, -1))),
        axis=0).transpose([1, 0])
    values_pos = np.concatenate(
        (densities_pos.reshape((1, -1)),
         uncerts_pos.reshape((1, -1))),
        axis=0).transpose([1, 0])

    values = np.concatenate((values_neg, values_pos))
    labels = np.concatenate(
        (np.zeros_like(densities_neg), np.ones_like(densities_pos)))

    lr = LogisticRegressionCV(n_jobs=-1, max_iter=1000).fit(values, labels)

    return values, labels, lr


def compute_roc(probs_neg, probs_pos, plot=False):
    """
    TODO
    :param probs_neg:
    :param probs_pos:
    :param plot:
    :return:
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                 label='ROC (AUC = %0.4f)' % auc_score)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()

    return fpr, tpr, auc_score