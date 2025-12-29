# AUDIT_SKIP
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

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype
from mindspore.common.initializer import Normal
from mindspore.nn import Accuracy

from art.estimators.classification.mindspore import MindSporeClassifier
from art.utils import load_mnist, load_cifar10

def set_seed(args):
    """
    :param args:
    :return:
    """
    mindspore.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

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

class GetLoader:
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

def construct_model(model_name='cnn', dataset='mnist', training_mode='load', epochs=50, batch_size=128, verbose=True):
    """
    Construct model setup.
    """
    if model_name =='cnn':
        if dataset == 'mnist':
            from baseline.cnn.cnn_mnist import MNISTCNN
            model = MNISTCNN(mode=training_mode, filename='cnn_mnist.ckpt', epochs=epochs, batch_size=batch_size)
            classifier = model.classifier
        elif dataset == 'cifar':
            from baseline.cnn.cnn_cifar10 import CIFAR10CNN
            model = CIFAR10CNN(mode=training_mode, filename='cnn_cifar.ckpt', epochs=epochs, batch_size=batch_size)
            classifier = model.classifier
        elif dataset == 'svhn':
            from baseline.cnn.cnn_svhn import SVHNCNN
            model = SVHNCNN(mode=training_mode, filename='cnn_svhn.ckpt', epochs=epochs, batch_size=batch_size)
            classifier = model.classifier
        else:
            raise Exception("Sorry, the dataset is unknown.")
    elif model_name =='resnet':
        if dataset == 'cifar':
            from baseline.models.resnet import CIFARResNet50
            model = CIFARResNet50(mode=training_mode, filename='resnet50_cifar.ckpt', epochs=epochs, batch_size=batch_size)
            classifier = model.classifier
        elif dataset == 'svhn':
            from baseline.models.resnet import SVHNResNet50  
            model = SVHNResNet50(mode=training_mode, filename='resnet50_svhn.ckpt', epochs=epochs, batch_size=batch_size)
            classifier = model.classifier
        elif dataset == 'imagenet':
            from baseline.models.resnet import IMAGENETResNet50  
            model = IMAGENETResNet50(mode=training_mode, filename='resnet50_imagenet.ckpt', epochs=epochs, batch_size=batch_size)
            classifier = model.classifier
        else:
            raise Exception("Sorry, the resnet model on {} dataset is unknown.".format(dataset))
    else:
        raise Exception("Sorry, the model is unknown.")

    return classifier

def create_mindspore_dataset(x_data, y_data, batch_size=32, is_train=True):
    """
    Create MindSpore dataset from numpy arrays
    """
    dataset = ds.NumpySlicesDataset({"data": x_data, "label": y_data}, shuffle=is_train)
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset

class CrossEntropyLossWrapper(nn.Cell):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=False)
        
    def construct(self, logits, labels):
        return self.loss_fn(logits, labels)

def get_next_class(Y_test):
    num_classes = Y_test.shape[1]
    Y_test_labels = np.argmax(Y_test, axis=1)
    Y_test_labels = (Y_test_labels + 1) % num_classes
    return np.eye(num_classes)[Y_test_labels]

def get_least_likely_class(Y_pred):
    num_classes = Y_pred.shape[1]
    Y_target_labels = np.argmin(Y_pred, axis=1)
    return np.eye(num_classes)[Y_target_labels]

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

# Placeholder classes/functions that need to be implemented based on specific requirements
class _TodoPlaceholder:
    """
    一个内部类，用于创建我们唯一的 TODO 占位符。
    它会重写所有常见的"魔术方法"，使任何使用它的尝试都失败。
    """

    def _explode(self, *args, **kwargs):
        """一个统一的"爆炸"方法。"""
        raise NotImplementedError(
            "操作失败：您正在尝试使用一个'TODO'占位符。"
            "请先完成该功能的具体实现再使用它。"
        )

    # --- 核心交互 ---
    def __call__(self, *args, **kwargs):
        self._explode()

    def __getattribute__(self, name):
        # __repr__ 是唯一允许的属性，用于显示友好信息
        if name == "__repr__":
            return super().__getattribute__(name)
        self._explode()

    def __setattr__(self, name, value):
        self._explode()

    def __delattr__(self, name):
        self._explode()

    # --- 类型转换与布尔判断 (非常重要！) ---
    def __bool__(self):
        # 这个方法阻止了 `if my_variable:` 这样的代码意外通过
        self._explode()

    def __str__(self):
        self._explode()

    def __repr__(self):
        self._explode()

    # --- 容器与迭代 ---
    def __getitem__(self, key):
        self._explode()

    def __setitem__(self, key, value):
        self._explode()
        
    def __iter__(self):
        self._explode()

    def __len__(self):
        self._explode()
    
    # --- 运算操作 (可以根据需要添加更多) ---
    __add__ = __sub__ = __mul__ = __div__ = _explode
    __eq__ = __ne__ = __lt__ = __gt__ = _explode

TODO = _TodoPlaceholder()

def save_image(tensor, filename):
    """
    Save a tensor as an image file.
    This function mimics torchvision.utils.save_image for MindSpore.
    
    Args:
        tensor: numpy array or MindSpore tensor with shape (C, H, W)
        filename: output filename
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to numpy if it's a MindSpore tensor
    if hasattr(tensor, 'asnumpy'):
        img = tensor.asnumpy()
    else:
        img = np.array(tensor)
    
    # Ensure the image is in the right shape
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        # Convert from (C, H, W) to (H, W, C)
        img = np.transpose(img, (1, 2, 0))
    
    # Clip values to [0, 1] range
    img = np.clip(img, 0, 1)
    
    # If grayscale, remove the channel dimension
    if img.shape[2] == 1:
        img = img.squeeze(axis=2)
    
    # Save the image
    plt.imsave(filename, img)