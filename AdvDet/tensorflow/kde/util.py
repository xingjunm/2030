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
import tensorflow as tf
from todo_placeholder import TODO

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

# RISK_INFO: [API 行为不等价] - PyTorch的Dropout实现及其底层随机数生成器可能与TensorFlow不同，这可能影响每次迭代中被丢弃的神经元的精确集合。
# RISK_INFO: [实现差异的潜在风险] - 原代码显式地将张量移动到指定的'device'（CPU/GPU）。此实现依赖于TensorFlow的默认设备放置策略，其行为可能因环境配置而异。
def get_mc_predictions(model, X, nb_iter=50, batch_size=256):
    """
    Get Monte Carlo predictions using dropout at inference time
    """
    # Enable training mode to use dropout
    training_mode = True
    
    # Get output dimension from the last layer
    last_layer = model.layers[-1]
    output_dim = last_layer.units
    
    # Convert to TensorFlow tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    def predict():
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        output = np.zeros(shape=(len(X), output_dim))
        for i in range(n_batches):
            batch = X[i * batch_size:(i + 1) * batch_size]
            # Get predictions with dropout enabled
            batch_pred = model(batch, training=training_mode)
            output[i * batch_size:(i + 1) * batch_size] = tf.nn.softmax(batch_pred).numpy()
        return output

    preds_mc = []
    for i in tqdm(range(nb_iter)):
        preds_mc.append(predict())

    return np.asarray(preds_mc)

# RISK_INFO: [API 行为不等价] - PyTorch的`model.modules()`和TensorFlow的`model.layers`对模型结构的表述可能不同，尤其是在复杂或嵌套模型中。这可能导致选择了不同的“倒数第二层”。
# RISK_INFO: [实现差异的潜在风险] - 提取中间层特征的方法有根本性差异（PyTorch hooks vs. 新建Keras子模型）。虽然目标相同，但这种实现上的巨大差异可能引入细微的行为偏差。
def get_deep_representations(model, X, batch_size=256, dataset='mnist'):
    """
    Get deep representations from the second-to-last layer
    """
    # Create a new model that outputs the second-to-last layer
    # Find the last hidden layer (before the final classification layer)
    last_hidden_layer = None
    for i in range(len(model.layers) - 1, -1, -1):
        layer = model.layers[i]
        if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
            if i < len(model.layers) - 1:  # Not the last layer
                last_hidden_layer = model.layers[i]
                break
    
    if last_hidden_layer is None:
        raise ValueError("Could not find a suitable hidden layer")
    
    # Create a model that outputs the hidden layer
    # Handle Sequential models differently
    if isinstance(model, tf.keras.Sequential):
        # Build the input shape if not already built
        if not model.built:
            # Infer input shape from the first layer
            if hasattr(model.layers[0], 'input_shape'):
                model.build(model.layers[0].input_shape)
            else:
                # Default input shapes based on dataset
                if dataset == 'mnist':
                    model.build((None, 28, 28, 1))
                elif dataset == 'cifar-10':
                    model.build((None, 32, 32, 3))
                elif dataset == 'svhn':
                    model.build((None, 32, 32, 3))
                else:
                    raise ValueError(f"Unknown dataset: {dataset}")
        
        # For Sequential models, create a new model up to the desired layer
        layer_outputs = []
        for layer in model.layers[:model.layers.index(last_hidden_layer) + 1]:
            layer_outputs.append(layer)
        hidden_model = tf.keras.Sequential(layer_outputs)
    else:
        # For functional models
        hidden_model = tf.keras.Model(inputs=model.input, outputs=last_hidden_layer.output)
    
    # Convert to TensorFlow tensor
    X = tf.convert_to_tensor(X, dtype=tf.float32)
    
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))    
    output = []
    for i in tqdm(range(n_batches)):
        batch = X[i * batch_size:(i + 1) * batch_size]
        hidden_output = hidden_model(batch, training=False)
        # Flatten the output
        flatten_output = tf.reshape(hidden_output, (hidden_output.shape[0], -1))
        output.extend(flatten_output.numpy())
    
    return np.array(output)


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