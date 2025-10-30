from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA
import tensorflow as tf

# Gaussian noise scale sizes that were determined so that the average
# L-2 perturbation size is equal to that of the adversarial samples
# mnist roughly L2_difference/20
# cifar roughly L2_difference/54
# svhn roughly L2_difference/60
# be very carefully with these settings, tune to have noisy/adv have the same L2-norm
# otherwise artifact will lose its accuracy

# fined tuned again when retrained all models with X in [-0.5, 0.5]
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
    'svhn': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,\
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,\
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,\
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,\
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,\
            'cwi': 0.125, 'df': 0.125,\
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,\
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125
            },
    'tiny': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,\
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,\
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,\
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,\
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,\
            'cwi': 0.125, 'df': 0.125,\
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,\
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125
            },
}

CLIP_MIN = 0.0
CLIP_MAX = 1.0
# CLIP_MIN = -0.5
# CLIP_MAX = 0.5
PATH_DATA = "data/"

# Set random seed
np.random.seed(0)

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

def get_layer_wise_activations(model, dataset):
    """
    Get the deep activation outputs.
    :param model:
    :param dataset: 'mnist', 'cifar', 'svhn', has different submanifolds architectures  
    :return: 
    """
    assert dataset in ['mnist', 'cifar', 'svhn', 'tiny'], \
        "dataset parameter must be either 'mnist' 'cifar' or 'svhn'"
    if dataset == 'mnist':
        # mnist model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    elif dataset == 'cifar':
        # cifar-10 model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    elif dataset == 'svhn':
        # svhn model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers])
    else:
        # tiny model
        acts = [model.layers[0].input]
        acts.extend([layer.output for layer in model.layers[-50:]])
    return acts

# lid of a single query point x
def mle_single(data, x, k=20):
    # data = np.asarray(data, dtype=np.float32)
    # x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

def mle_batch(data, batch, k):
    data  = np.asarray(data,  dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    
    # select number of neighbors
    k_actual = min(k, len(data)-1)
    # f  = lambda v: - k / np.sum( np.log(v/v[-1]) ) # original LID
    f2 = lambda v: - np.log( v / v[-1] ) # multiLID
    dist = cdist(batch, data)
    dist = np.apply_along_axis(np.sort, axis=1, arr=dist)[:,1:k_actual+1]
    multi_lid = np.apply_along_axis(f2, axis=1, arr=dist)
    
    # Pad the result to match the expected k size if needed
    if k_actual < k:
        # Pad with zeros to match expected shape
        pad_width = [(0, 0), (0, k - k_actual)]
        multi_lid = np.pad(multi_lid, pad_width, mode='constant', constant_values=0)
    
    return multi_lid

# mean distance of x to its k nearest neighbours
def kmean_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: np.mean(v)
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# mean distance of x to its k nearest neighbours
def kmean_pca_batch(data, batch, k=10):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)
    a = np.zeros(batch.shape[0])
    for i in np.arange(batch.shape[0]):
        tmp = np.concatenate((data, [batch[i]]))
        tmp_pca = PCA(n_components=2).fit_transform(tmp)
        a[i] = kmean_batch(tmp_pca[:-1], tmp_pca[-1], k=k)
    return a

def get_normalization(dataset):

    mean = None
    std  = None
    if dataset == 'mnist':
        mean = [0.1307]
        std  = [0.3081]
    elif dataset == 'cifar':
        mean = [0.4914, 0.4822, 0.4465]
        std  = [0.2023, 0.1994, 0.2010]
    elif dataset == 'imagenet32' or dataset == 'imagenet64' or dataset == 'imagenet128':
        mean = [0.4810, 0.4574, 0.4078]
        std  = [0.2146, 0.2104, 0.2138]
    elif dataset == 'imagenet':        
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        raise("Err: normalization not found!")

    return mean, std

def normalize_images(images, dataset):
    mean, std = get_normalization(dataset)
    if dataset == 'mnist':
        images[:,0,:,:] = (images[:,0,:,:] - mean[0]) / std[0]
    else:   
        images[:,0,:,:] = (images[:,0,:,:] - mean[0]) / std[0]
        images[:,1,:,:] = (images[:,1,:,:] - mean[1]) / std[1]
        images[:,2,:,:] = (images[:,2,:,:] - mean[2]) / std[2]
    return images

def get_layer_feature_maps(activation_dict, act_layer_list):
    act_val_list = []
    for it in act_layer_list:
        act_val = activation_dict[it]
        act_val_list.append(act_val)
    return act_val_list

def get_logical_block_indices(model):
    """Identify indices of layers that represent end of logical blocks"""
    block_indices = []
    
    for i, layer in enumerate(model.layers):
        # Add pooling layers as they typically end conv blocks
        if isinstance(layer, (tf.keras.layers.MaxPooling2D, 
                            tf.keras.layers.AveragePooling2D,
                            tf.keras.layers.GlobalAveragePooling2D,
                            tf.keras.layers.GlobalMaxPooling2D)):
            block_indices.append(i)
        # Add Dense layers as they represent fully connected blocks
        elif isinstance(layer, tf.keras.layers.Dense):
            block_indices.append(i)
        # Add Flatten layer as it marks transition from conv to dense
        elif isinstance(layer, tf.keras.layers.Flatten):
            if i > 0:  # Add the layer before Flatten as end of conv block
                block_indices.append(i-1)
    
    # Always include the final layer
    if len(model.layers) - 1 not in block_indices:
        block_indices.append(len(model.layers) - 1)
    
    # Remove duplicates and sort
    block_indices = sorted(list(set(block_indices)))
    
    return block_indices

def multiLID(model, X, X_noisy, X_adv, dataset, k=10, batch_size=100):
    # get references to wanted activation layers of different networks
    # the number of activation layers is the number of featues for the LID
    # TensorFlow doesn't need explicit eval mode, training=False will be used during inference
    
    # Get logical block indices to mimic PyTorch's children()
    block_indices = get_logical_block_indices(model)
    
    # Create intermediate models for each block (including the last one)
    funcs = []
    
    # Build model if not already built
    if not model.built:
        # Get input shape from X
        input_shape = X.shape[1:]
        model.build((None,) + input_shape)
    
    for idx in block_indices:
        # For Sequential models, we need to create a new model up to the target layer
        if isinstance(model, tf.keras.Sequential):
            # Create a new Sequential model with layers up to idx
            intermediate_model = tf.keras.Sequential(model.layers[:idx+1])
            # Build the intermediate model
            intermediate_model.build((None,) + X.shape[1:])
        else:
            # For functional models, use the standard approach
            intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.layers[idx].output)
        funcs.append(intermediate_model)
    
    lid_dim = len(funcs)
    print("Number of layers to estimate: ", lid_dim)

    shape = np.shape(X[0])
    
    def estimate(i_batch):
        # estimation of the MLE batch
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start

        # prepare data structure 
        lid_batch = np.zeros(shape=(n_feed, k, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, k, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, k, lid_dim))
        
        # Initialize tensors for batch data
        batch = tf.zeros([n_feed, shape[0], shape[1], shape[2]], dtype=tf.float32)
        batch_adv = tf.zeros([n_feed, shape[0], shape[1], shape[2]], dtype=tf.float32)
        batch_noisy = tf.zeros([n_feed, shape[0], shape[1], shape[2]], dtype=tf.float32)
        
        # RISK_INFO: [张量操作差异] - 使用 tf.Variable 和 .assign() 来构建批处理张量，与原架构中对 paddle 张量的直接切片赋值不同。这引入了状态性 (statefulness)，可能影响性能或在 tf.function 装饰的函数中产生意外行为。
        batch = tf.Variable(batch)
        batch_adv = tf.Variable(batch_adv)
        batch_noisy = tf.Variable(batch_noisy)
        
        for j in range(n_feed):
            batch[j,:,:,:].assign(tf.convert_to_tensor(X[start + j], dtype=tf.float32))
            batch_adv[j,:,:,:].assign(tf.convert_to_tensor(X_adv[start + j], dtype=tf.float32))
            batch_noisy[j,:,:,:].assign(tf.convert_to_tensor(X_noisy[start + j], dtype=tf.float32))

        # batch = normalize_images(batch, dataset)
        # batch_adv = normalize_images(batch_adv, dataset)
        # batch_noisy = normalize_images(batch_noisy, dataset)

        for i, func in enumerate(funcs):
            if i+1 == lid_dim:
                func = model
            X_act = func(batch, training=False).numpy()
            X_act = X_act.reshape((n_feed, -1))

            X_adv_act = func(batch_adv, training=False).numpy()
            X_adv_act = X_adv_act.reshape((n_feed, -1))

            X_noisy_act = func(batch_noisy, training=False).numpy()
            X_noisy_act = X_noisy_act.reshape((n_feed, -1))

            # random clean samples
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, :, i] = mle_batch(X_act, X_act, k=k)
            lid_batch_adv[:, :, i] = mle_batch(X_act, X_adv_act, k=k)
            lid_batch_noisy[:, :, i] = mle_batch(X_act, X_noisy_act, k=k)

        return lid_batch, lid_batch_noisy, lid_batch_adv

    lids = []
    lids_adv = []
    lids_noisy = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))

    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_noisy, lid_batch_adv = estimate(i_batch)
        lids.extend(lid_batch)
        lids_adv.extend(lid_batch_adv)
        lids_noisy.extend(lid_batch_noisy)
        # print("lids: ", lids.shape)
        # print("lids_adv: ", lids_noisy.shape)
        # print("lids_noisy: ", lids_noisy.shape)

    lids = np.asarray(lids, dtype=np.float32)
    lids_noisy = np.asarray(lids_noisy, dtype=np.float32)
    lids_adv = np.asarray(lids_adv, dtype=np.float32)

    return lids, lids_noisy, lids_adv


def normalize(normal, adv, noisy):
    """Z-score normalisation
    TODO
    :param normal:
    :param adv:
    :param noisy:
    :return:
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))

    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(X, y):
    """
    TODO
    :param X: the data samples
    :param y: the labels
    :return:
    """
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=5000).fit(X, y.ravel())
    return lr


def train_lr_rfeinman(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
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

    lr = LogisticRegressionCV(n_jobs=-1).fit(values, labels)

    return values, labels, lr

def compute_roc(y_true, y_pred, plot=False):
    """
    TODO
    :param y_true: ground truth
    :param y_pred: predictions
    :param plot:
    :return:
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
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

def compute_roc_rfeinman(probs_neg, probs_pos, plot=False):
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

def random_split(X, Y):
    """
    Random split the data into 80% for training and 20% for testing
    :param X: 
    :param Y: 
    :return: 
    """
    print("random split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    num_train = int(num_samples * 0.8)
    rand_pert = np.random.permutation(num_samples)
    X = X[rand_pert]
    Y = Y[rand_pert]
    X_train, X_test = X[:num_train], X[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    return X_train, Y_train, X_test, Y_test

def block_split(X, Y):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 100.
    :param X: 
    :param Y: 
    :return: 
    """
    print("Isolated split 80%, 20% for training and testing")
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.007) * 100

    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test