import tensorflow as tf
import numpy as np
from .utils import transform_options, dataset_options
from torchvision import transforms


class TorchDatasetWrapper:
    """Wrapper to make PyTorch datasets work with TensorFlow's tf.data API.
    
    This wrapper handles the conversion between PyTorch datasets and TensorFlow,
    including proper handling of PIL images and torch tensors.
    """
    def __init__(self, torch_dataset):
        self.dataset = torch_dataset
        self._len = len(torch_dataset)
    
    def __len__(self):
        return self._len
    
    def __getitem__(self, index):
        """Get item from PyTorch dataset and convert to numpy/TF format.
        
        Handles the conversion of PIL images and torch tensors to numpy arrays
        that can be used with TensorFlow.
        """
        # Get data from PyTorch dataset
        data, label = self.dataset[index]
        
        # Convert torch tensor to numpy array
        # The torchvision transforms already convert PIL to tensor and apply ToTensor()
        # which transposes from HWC to CHW and scales to [0,1]
        if hasattr(data, 'numpy'):
            # It's a torch tensor
            data = data.numpy()
        elif hasattr(data, 'convert'):
            # It's a PIL image (shouldn't happen with our transforms, but just in case)
            data = np.array(data)
            # Convert HWC to CHW and scale to [0,1] like ToTensor() does
            data = data.transpose(2, 0, 1).astype(np.float32) / 255.0
        
        # Ensure label is numpy scalar
        if hasattr(label, 'item'):
            label = label.item()
        elif hasattr(label, 'numpy'):
            label = label.numpy()
        else:
            label = np.array(label)
        
        return data, label


class DatasetGenerator():
    def __init__(self, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 **kwargs):

        if train_d_type not in transform_options:
            raise TypeError('Unknown Dataset')
        elif test_d_type not in transform_options:
            raise TypeError('Unknown Dataset')

        self.train_bs = train_bs
        self.eval_bs = eval_bs
        self.n_workers = n_workers
        self.train_path = train_path
        self.test_path = test_path

        train_tf = transform_options[train_d_type]["train_transform"]
        test_tf = transform_options[test_d_type]["test_transform"]
        train_tf = transforms.Compose(train_tf)
        test_tf = transforms.Compose(test_tf)

        # Create PyTorch datasets
        torch_train_set = dataset_options[train_d_type](train_path, train_tf,
                                                        False, kwargs)
        torch_test_set = dataset_options[test_d_type](test_path, test_tf,
                                                      True, kwargs)
        
        # Wrap them for TensorFlow
        self.train_set = TorchDatasetWrapper(torch_train_set)
        self.test_set = TorchDatasetWrapper(torch_test_set)
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)

    def get_loader(self, train_shuffle=True):
        """Create TensorFlow data loaders from the wrapped datasets.
        
        Returns tf.data.Dataset objects that are equivalent to PyTorch DataLoaders.
        """
        # Create TensorFlow datasets from the wrapped PyTorch datasets
        def train_generator():
            indices = np.arange(len(self.train_set))
            if train_shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                yield self.train_set[idx]
        
        def test_generator():
            for idx in range(len(self.test_set)):
                yield self.test_set[idx]
        
        # Get output signature from first item
        sample_data, sample_label = self.train_set[0]
        output_signature = (
            tf.TensorSpec(shape=sample_data.shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
        
        # Create tf.data.Dataset objects
        train_dataset = tf.data.Dataset.from_generator(
            train_generator,
            output_signature=output_signature
        )
        
        test_dataset = tf.data.Dataset.from_generator(
            test_generator,
            output_signature=output_signature
        )
        
        # Apply batching and prefetching for performance
        # Note: TensorFlow doesn't have direct equivalents for pin_memory and num_workers
        # prefetch with AUTOTUNE provides similar performance benefits
        train_loader = train_dataset.batch(self.train_bs, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        test_loader = test_dataset.batch(self.eval_bs, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        
        return train_loader, test_loader