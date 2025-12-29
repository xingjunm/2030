"""
LID (Local Intrinsic Dimensionality) utilities for PaddlePaddle
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import paddle
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
from tqdm import tqdm

# Gaussian noise scale sizes for different datasets and attacks
STDEVS = {
    'mnist': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,
            'cwi': 0.25, 'df': 0.25,
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,
            'sa': 0.3, 'sta': 0.3, 'hop': 0.3, 'zoo': 0.3
            },
    'cifar': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,
            'cwi': 0.125, 'df': 0.125,
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125, 'ap': 0.125
            },
    'svhn': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,
            'cwi': 0.125, 'df': 0.125,
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125, 'ap': 0.125
            },
    'imagenet': {'fgsm_0.03125': 0.03125, 'fgsm_0.0625': 0.0625, 'fgsm_0.125': 0.125, 'fgsm_0.25': 0.25, 'fgsm_0.3125': 0.3125, 'fgsm_0.5': 0.5,
            'bim_0.03125': 0.03125, 'bim_0.0625': 0.0625, 'bim_0.125': 0.125, 'bim_0.25': 0.25, 'bim_0.3125': 0.3125, 'bim_0.5': 0.5,
            'pgd1_5': 0.03125, 'pgd1_10': 0.0625, 'pgd1_15': 0.125, 'pgd1_20': 0.125, 'pgd1_25': 0.125, 'pgd1_30': 0.25, 'pgd1_40': 0.3125,
            'pgd2_0.25': 0.03125, 'pgd2_0.3125': 0.0625, 'pgd2_0.5': 0.125, 'pgd2_1': 0.125, 'pgd2_1.5': 0.25, 'pgd2_2': 0.3125,
            'pgdi_0.03125': 0.03125, 'pgdi_0.0625': 0.0625, 'pgdi_0.125': 0.125, 'pgdi_0.25': 0.25, 'pgdi_0.3125': 0.3125, 'pgdi_0.5': 0.5,
            'cwi': 0.125, 'df': 0.125,
            'hca_0.03125': 0.03125, 'hca_0.0625': 0.0625, 'hca_0.125': 0.125, 'hca_0.25': 0.25, 'hca_0.3125': 0.3125, 'hca_0.5': 0.5,
            'sa': 0.125, 'sta': 0.125, 'hop': 0.125, 'zoo': 0.125, 'ap': 0.125
            },
}

CLIP_MIN = 0.0
CLIP_MAX = 1.0
PATH_DATA = "data/"


def get_noisy_samples(X_test, X_test_adv, dataset, attack):
    """
    Generate noisy samples by adding Gaussian noise to clean samples
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
    # Convert to torch for cdist computation
    import torch
    a = torch.cdist(torch.from_numpy(x), torch.from_numpy(data)).numpy()
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

def mle_batch(data, batch, k):
    """
    Maximum likelihood estimation of local intrinsic dimensionality
    """
    k = min(k, len(data)-1)
    
    # Convert to PaddlePaddle tensors if needed
    if not isinstance(data, paddle.Tensor):
        data = paddle.to_tensor(data)
    if not isinstance(batch, paddle.Tensor):
        batch = paddle.to_tensor(batch)
    
    # Compute pairwise distances
    # RISK_INFO: [API 行为不等价] - 新架构手动实现了 cdist，而原架构使用 torch.cdist。这可能导致数值精度或性能上的差异。
    # paddle.cdist doesn't exist, so we compute manually
    # Using broadcasting: ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a,b>
    data_sqnorms = paddle.sum(data**2, axis=1, keepdim=True)
    batch_sqnorms = paddle.sum(batch**2, axis=1, keepdim=True)
    distances = batch_sqnorms + data_sqnorms.T - 2 * paddle.mm(batch, data.T)
    # RISK_INFO: [API 行为不等价] - 新增的 paddle.clip 操作是为了处理数值误差，这在原架构中不存在，可能导致在边缘情况下的行为差异。
    distances = paddle.sqrt(paddle.clip(distances, min=0))  # Clip to handle numerical errors
    
    # Sort distances
    sorted_distances = paddle.sort(distances, axis=1)
    # Get k nearest neighbors (excluding the first one which is the point itself)
    knn_distances = sorted_distances[:, 1:k+1]
    
    # RISK_INFO: [张量操作差异] - 新架构使用 Python for 循环，而原架构使用 torch.unbind 和列表推导式，可能存在性能差异。
    # Compute LID for each sample
    lid_values = []
    for i in range(knn_distances.shape[0]):
        v = knn_distances[i]
        # Compute MLE: -k / sum(log(v_i / v_k))
        lid = -k / paddle.sum(paddle.log(v / v[-1]))
        lid_values.append(lid)
    
    lid = paddle.stack(lid_values).cpu().numpy()
    return lid

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

def get_lids_random_batch(model, X, X_noisy, X_adv, dataset, k=10, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model:
    :param X: normal images
    :param X_noisy: noisy images
    :param X_adv: advserial images    
    :param dataset: 'mnist', 'cifar', has different DNN architectures  
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """

    model.eval()
    device = 'gpu' if paddle.is_compiled_with_cuda() else 'cpu'
    paddle.set_device(device)

    # get deep representations
    funcs = [paddle.nn.Sequential(*list(model.children())[:i+1]) for i in range(len(list(model.children())))]
    lid_dim = len(funcs)
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        # estimation of the MLE batch
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start

        # prepare data structure 
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_adv = np.zeros(shape=(n_feed, lid_dim))
        lid_batch_noisy = np.zeros(shape=(n_feed, lid_dim))

        for i, func in enumerate(funcs):
            if i+1 == lid_dim:
                func = model
            X_act = func(paddle.to_tensor(X[start:end]))
            X_act = X_act.reshape((n_feed, -1))

            X_adv_act = func(paddle.to_tensor(X_adv[start:end]))
            X_adv_act = X_adv_act.reshape((n_feed, -1))

            X_noisy_act = func(paddle.to_tensor(X_noisy[start:end]))
            X_noisy_act = X_noisy_act.reshape((n_feed, -1))

            # random clean samples
            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i] = mle_batch(X_act, X_act, k=k)
            lid_batch_adv[:, i] = mle_batch(X_act, X_adv_act, k=k)
            lid_batch_noisy[:, i] = mle_batch(X_act, X_noisy_act, k=k)

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
        # print("lids: ", np.array(lids).shape)
        # print("lids_adv: ", np.array(lids_noisy).shape)
        # print("lids_noisy: ", np.array(lids_noisy).shape)

    lids = np.asarray(lids)
    lids_noisy = np.asarray(lids_noisy)
    lids_adv = np.asarray(lids_adv)

    return lids, lids_noisy, lids_adv




def normalize(normal, adv, noisy):
    """
    Normalize the data using StandardScaler
    """
    n_samples = len(normal)
    total = scale(np.concatenate((normal, adv, noisy)))
    
    return total[:n_samples], total[n_samples:2*n_samples], total[2*n_samples:]


def train_lr(X, y):
    """
    Train logistic regression classifier
    """
    lr = LogisticRegressionCV(n_jobs=-1, max_iter=1000).fit(X, y.ravel())
    return lr


def train_lr_rfeinman(densities_pos, densities_neg, uncerts_pos, uncerts_neg):
    """
    Train logistic regression for Rfeinman's method
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
    Compute ROC curve and AUC score
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred)
    
    if plot:
        import matplotlib.pyplot as plt
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
    Compute ROC for Rfeinman's method
    """
    probs = np.concatenate((probs_neg, probs_pos))
    labels = np.concatenate((np.zeros_like(probs_neg), np.ones_like(probs_pos)))
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_score = auc(fpr, tpr)
    
    if plot:
        import matplotlib.pyplot as plt
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
    Split the data into training and testing with isolation
    This is the specific split used in the LID paper to ensure
    no information leakage between train and test sets.
    """
    print("Isolated split 70%, 30% for training and testing")
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


