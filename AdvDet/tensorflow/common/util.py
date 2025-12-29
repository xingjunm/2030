from __future__ import division, absolute_import, print_function

import os
import sys
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import random
import json
import pickle
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import load_mnist, load_cifar10

def set_seed(args):
    """
    :param args:
    :return:
    """
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # TensorFlow's equivalent of PyTorch's deterministic behavior
    # RISK_INFO: [API 行为不等价] - PyTorch's `torch.backends.cudnn.deterministic` is a runtime flag, whereas `os.environ` must be set before TensorFlow initialization. This can lead to different behavior if TF is imported before this function is called.
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def normalize_mean(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def normalize_linear(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test

def normalize_meanstd(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2))
    std = np.std(X_train, axis=(0, 1, 2))
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test, mean, std

def get_data(dataset='mnist', return_channel=True, scale=True):
    """
    """
    assert dataset in ['mnist', 'cifar', 'svhn'], "Dataset parameter must be either 'mnist', 'cifar' or 'svhn'"
    assert scale in [True, False], "scale parameter must be either True or False"
    if dataset=='mnist':
        (X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_mnist()
    elif dataset=='cifar':
        (X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    elif dataset=='svhn':
        (X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value = load_svhn()
    else:
        raise Exception('No dataset selected')
    
    if scale:
        X_train = X_train / 255.0
        X_test = X_test / 255.0
    
    if not return_channel:
        X_train = X_train.squeeze(axis=3)
        X_test = X_test.squeeze(axis=3)

    return (X_train, y_train), (X_test, y_test), min_pixel_value, max_pixel_value

def load_svhn(raw: bool = False):
    if not os.path.isfile("./Datasets/SVHN/cropped/train_32x32.mat"):
        print('Downloading SVHN train set...')
        call(
            "curl -o ./Datasets/SVHN/cropped/train_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
            shell=True
        )

    if not os.path.isfile("./Datasets/SVHN/cropped/test_32x32.mat"):
        print('Downloading SVHN test set...')
        call(
            "curl -o ./Datasets/SVHN/cropped/test_32x32.mat "
            "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
            shell=True
        )

    train = sio.loadmat('./Datasets/SVHN/cropped/train_32x32.mat')
    test = sio.loadmat('./Datasets/SVHN/cropped/test_32x32.mat')
    x_train = np.transpose(train['X'], axes=[3, 0, 1, 2])
    x_test = np.transpose(test['X'], axes=[3, 0, 1, 2])
    # reshape (n_samples, 1) to (n_samples,) and change 1-index to 0-index
    y_train = train['y'] - 1
    y_test = test['y'] - 1

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = np.reshape(x_train, (73257, 32, 32, 3))
    x_test = np.reshape(x_test, (26032, 32, 32, 3))

    min_, max_ = 0.0, 255.0
    if not raw:
        min_, max_ = 0.0, 1.0
        x_train, y_train = preprocess(x_train, y_train, clip_values=(0, 255))
        x_test, y_test = preprocess(x_test, y_test, clip_values=(0, 255))

    return (x_train, y_train), (x_test, y_test), min_, max_

def save_results(dst, results_all):
    """
    """
    with open(dst, 'w') as f:
        json.dump(results_all, f, indent=2)

def load_characteristics(X, Y, file_name):
    """
    """
    for i in range(10):
        if i==0:
            file_name_str = os.path.join('data/', file_name)
            print("Loading sample characteristics from [{}]".format(file_name_str))
            X = np.load(file_name_str)
        else:
            pass
    return X, Y

def test(model, X, Y, batch_size=128):
    """
    """
    output = model.predict(X, batch_size=batch_size)
    pred = np.argmax(output, axis=1)
    label = np.argmax(Y, axis=1)
    
    num_correct = np.sum(pred == label)
    acc = float(num_correct) / float(len(X))
    
    return acc, pred

def block_split(X, Y, out):
    """
    Split the data into 80% for training and 20% for testing
    in a block size of 10000
    :param X: 
    :param Y: 
    :param out: 
    :return: 
    """
    num_samples = X.shape[0]
    partition = int(num_samples / 10)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition: 3*partition], Y[2*partition: 3*partition]
    
    if out == 'train':
        X_train = np.concatenate((X_adv[:8000], X_norm[:8000], X_noisy[:8000]))
        Y_train = np.concatenate((Y_adv[:8000], Y_norm[:8000], Y_noisy[:8000]))
        return X_train, Y_train
    
    elif out == 'test':
        X_test = np.concatenate((X_adv[8000:], X_norm[8000:], X_noisy[8000:]))
        Y_test = np.concatenate((Y_adv[8000:], Y_norm[8000:], Y_noisy[8000:]))
        return X_test, Y_test

def block_split_adv(X, Y, size=100):
    """
    """
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition: 3*partition], Y[2*partition: 3*partition]
    
    if size==100:
        selected = random.sample(range(partition), size)
        
        X_test = np.concatenate((X_norm[selected], X_noisy[selected], X_adv))
        Y_test = np.concatenate((Y_norm[selected], Y_noisy[selected], Y_adv))
    
    elif size==1000:
        selected = random.sample(range(partition), size)
        X_test = np.concatenate((X_norm[selected], X_noisy[selected], X_adv[:size]))
        Y_test = np.concatenate((Y_norm[selected], Y_noisy[selected], Y_adv[:size]))
    else:
        raise Exception("Size is not supported")
    
    return X_test, Y_test

def get_noisy_samples(X_test, X_test_adv, dataset='mnist', attack='fgsm'):
    """
    """
    if dataset=='mnist':
        X_test_noisy = np.minimum(
            np.maximum(
                X_test_adv + np.random.normal(loc=0, scale=0.2, 
                                            size=X_test_adv.shape),
                0
            ),
            1
        )
    elif dataset=='cifar':
        X_test_noisy = np.minimum(
            np.maximum(
                X_test_adv + np.random.normal(loc=0, scale=8/255., 
                                            size=X_test_adv.shape),
                0
            ),
            1
        )
    elif dataset=='svhn':
        X_test_noisy = np.minimum(
            np.maximum(
                X_test_adv + np.random.normal(loc=0, scale=8/255., 
                                            size=X_test_adv.shape),
                0
            ),
            1
        )
    else:
        raise Exception("Dataset is not supported")
    
    return X_test_noisy

def random_split(X, Y, model, X_train, Y_train, X_test, Y_test, num_classes=10, split=0.8):
    """
    """
    _, pred = test(model, X, Y)
    
    size = int(split * X.shape[0])
    X_adv, Y_adv = X[:size], Y[:size]
    X_test = np.concatenate((X_test, X[size:]))
    Y_test = np.concatenate((Y_test, Y[size:]))
    
    # Correctly classified samples
    inds_correct = np.where(pred == Y_adv.argmax(axis=1))[0]
    X_adv = X_adv[inds_correct]
    Y_adv = Y_adv[inds_correct]
    
    X_train = np.concatenate((X_train, X_adv))
    Y_train = np.concatenate((Y_train, Y_adv))
    
    return X_train, Y_train, X_test, Y_test

def scalar_metrics(Y_true, Y_pred, labels):
    """
    """
    acc = accuracy_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred, pos_label=labels[0])
    recall = recall_score(Y_true, Y_pred, pos_label=labels[0])
    return acc, precision, recall

def compute_roc(Y_true, Y_pred_score, plot=False):
    """
    """
    fpr, tpr, _ = roc_curve(Y_true, Y_pred_score)
    roc_auc = auc(fpr, tpr)
    if plot:
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, color='blue',
                label='ROC (AUC = %0.4f)' % roc_auc)
        plt.legend(loc='lower right')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.show()
    
    return fpr, tpr, roc_auc

def random_split_to_two(X, Y, split=0.8):
    """
    """
    size = int(split * X.shape[0])
    X_1, Y_1 = X[:size], Y[:size]
    X_2, Y_2 = X[size:], Y[size:]
    return X_1, Y_1, X_2, Y_2

def transform_test_data_for_model(model_path, X_test):
    raise NotImplementedError("Skip")

# RISK_INFO: [API 行为不等价] - The base class is changed from `torch.utils.data.Dataset` to `tf.keras.utils.Sequence`. While the implemented methods are the same, the behavior of the data loading pipeline that uses this class (e.g., `torch.utils.data.DataLoader` vs. `model.fit`) will differ, especially regarding multiprocessing and performance.
class GetLoader(tf.keras.utils.Sequence):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    
    def __len__(self):
        return len(self.data)

def get_tpr_fpr(true_labels, pred_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))

    AP = np.sum(true_labels)
    AN = np.sum(1-true_labels)

    tpr = TP/AP if AP>0 else np.nan
    fpr = FP/AN if AN>0 else np.nan

    return tpr, fpr, TP, AP, FP, AN

def evalulate_detection_test(Y_detect_test, Y_detect_pred):
    accuracy = accuracy_score(Y_detect_test, Y_detect_pred, normalize=True, sample_weight=None)
    tpr, fpr, tp, ap, fp, an = get_tpr_fpr(Y_detect_test, Y_detect_pred)
    return accuracy, tpr, fpr, tp, ap, fp, an

def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    :param X_pos: positive samples
    :param X_neg: negative samples
    :return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    print("X_pos: ", X_pos.shape)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    print("X_neg: ", X_neg.shape)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]

def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]

def preprocess(x: np.ndarray, y: np.ndarray, nb_classes: int = 10, clip_values: tuple = (0, 255)):
    """
    Scales `x` to [0, 1] and converts `y` to class categorical confidences.

    :param x: Data instances.
    :param clip_values: Original data range allowed value for features, either one respective scalar or one value per
           feature.
    :return: Rescaled values of `x`.
    """
    if clip_values is None:
        min_, max_ = np.amin(x), np.amax(x)
    else:
        min_, max_ = clip_values
    normalized_x = (x - min_) / (max_ - min_)
    categorical_y = to_categorical(y, nb_classes)

    return normalized_x, categorical_y

def to_categorical(labels, nb_classes) -> np.ndarray:
    """
    Convert an array of labels to binary class matrix.

    :param labels: An array of integer labels of shape `(nb_samples,)`.
    :param nb_classes: The number of classes (possible labels).
    :return: A binary matrix representation of `y` in the shape `(nb_samples, nb_classes)`.
    """
    labels = np.array(labels, dtype=int)
    if nb_classes is None:
        nb_classes = np.max(labels) + 1
    categorical = np.zeros((labels.shape[0], nb_classes), dtype=np.float32)
    categorical[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return categorical

class Average_Saliency(object):
    def __init__(self, model, output_index=0):
        pass

    def get_grad(self, input_image):
        pass

    def get_average_grad(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples