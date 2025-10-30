import numpy as np
import paddle
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet
from PIL import Image

transform_options = {
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(15),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "MNIST": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]}
    }

transform_options["CIFAR10Noisy"] = transform_options["CIFAR10"]


class DatasetWrapper:
    """Wrapper class for torchvision datasets to convert torch tensors to numpy arrays.
    
    According to paddlepaddle-note.md #1 and #6, and paddlepaddle-exemptions.md #4,
    we need to wrap torchvision datasets and add __getitem__ to handle format conversion.
    This keeps torch dependencies limited to the dataset boundary.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __getitem__(self, index):
        # Get the original data from torchvision dataset
        data, label = self.dataset[index]
        
        # Convert torch tensor to numpy array
        # torchvision's ToTensor() returns a torch.Tensor
        if hasattr(data, 'numpy'):  # Check if it's a torch tensor
            data = data.numpy()
        
        # PaddlePaddle's DataLoader will auto-convert numpy to paddle.Tensor
        # as noted in paddlepaddle-note.md #4
        return data, label
    
    def __len__(self):
        return len(self.dataset)


dataset_options = {
    "CIFAR10": lambda path, transform, is_test, kwargs:
        DatasetWrapper(CIFAR10(root=path, train=not is_test, download=True,
                               transform=transform)),
    "CIFAR100": lambda path, transform, is_test, kwargs:
        DatasetWrapper(CIFAR100(root=path, train=not is_test, download=True,
                                transform=transform)),
    "SVHN": lambda path, transform, is_test, kwargs:
        DatasetWrapper(SVHN(root=path, split='test' if is_test else 'train', download=True,
                           transform=transform)),
    "MNIST": lambda path, transform, is_test, kwargs:
        DatasetWrapper(MNIST(root=path, train=not is_test, download=True,
                            transform=transform)),
    "ImageNet": lambda path, transform, is_test, kwargs:
        DatasetWrapper(ImageNet(root=path, split='val' if is_test else 'train',
                               transform=transform))
}