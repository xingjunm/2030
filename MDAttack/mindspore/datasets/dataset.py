from .utils import transform_options, dataset_options
import mindspore as ms
import mindspore.dataset as ds
from torchvision import transforms
import numpy as np


class TorchDatasetWrapper:
    """Wrapper for PyTorch datasets to work with MindSpore.
    
    This wrapper handles the conversion between PyTorch datasets and MindSpore,
    ensuring that torch tensor operations are limited to within the dataset.
    """
    def __init__(self, torch_dataset):
        self.torch_dataset = torch_dataset
        
    def __getitem__(self, index):
        """Get item and convert from PyTorch format to numpy arrays.
        
        According to mindspore-note.md and mindspore-exemptions.md:
        - We need to define __getitem__ to handle data format conversion
        - Convert torch tensors to numpy arrays or MindSpore tensors
        - This keeps torch dependencies limited to dataset internals
        """
        # Get item from PyTorch dataset
        data, label = self.torch_dataset[index]
        
        # Convert to numpy if it's a torch tensor
        if hasattr(data, 'numpy'):  # It's a torch tensor
            data = data.numpy()
        elif hasattr(data, '__array__'):  # PIL Image or other array-like
            data = np.array(data)
            
        # Ensure label is numpy type
        if hasattr(label, 'numpy'):
            label = label.numpy()
        else:
            label = np.array(label)
            
        return data, label
    
    def __len__(self):
        return len(self.torch_dataset)


class DatasetGenerator():
    def __init__(self, train_bs=128, eval_bs=256, seed=0, n_workers=4,
                 train_d_type='CIFAR10', test_d_type='CIFAR10',
                 train_path='../../datasets/', test_path='../../datasets/',
                 **kwargs):

        if train_d_type not in transform_options:
            raise ValueError('Unknown Dataset')  # Fixed from raise('Unknown Dataset')
        elif test_d_type not in transform_options:
            raise ValueError('Unknown Dataset')  # Fixed from raise('Unknown Dataset')

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
        
        # Wrap them for MindSpore compatibility
        self.train_set = TorchDatasetWrapper(torch_train_set)
        self.test_set = TorchDatasetWrapper(torch_test_set)
        
        self.train_set_length = len(self.train_set)
        self.test_set_length = len(self.test_set)

    def get_loader(self, train_shuffle=True):
        """Create MindSpore data loaders.
        
        According to mindspore-note.md:
        - MindSpore uses mindspore.dataset API instead of torch.utils.data.DataLoader
        """
        # Create MindSpore dataset from the wrapped datasets
        train_dataset = ds.GeneratorDataset(
            self.train_set,
            column_names=["data", "label"],
            num_parallel_workers=self.n_workers,
            shuffle=train_shuffle
        )
        
        test_dataset = ds.GeneratorDataset(
            self.test_set,
            column_names=["data", "label"],
            num_parallel_workers=self.n_workers,
            shuffle=False
        )
        
        # Set batch size
        train_loader = train_dataset.batch(
            batch_size=self.train_bs,
            drop_remainder=False  # equivalent to drop_last=False
        )
        
        test_loader = test_dataset.batch(
            batch_size=self.eval_bs,
            drop_remainder=False  # equivalent to drop_last=False
        )
        
        return train_loader, test_loader