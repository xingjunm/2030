# TensorFlow Detection Module

This is the TensorFlow implementation of the adversarial detection methods, migrated from the PyTorch version.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train a CNN model for a specific dataset:

```bash
python train_model.py -d mnist -e 50 -b 128
python train_model.py -d cifar -e 100 -b 512
python train_model.py -d svhn -e 100 -b 256
```

### Generating Adversarial Examples

To generate adversarial examples:

```bash
python generate_adv.py -d mnist
python generate_adv.py -d cifar
python generate_adv.py -d svhn
```

### Running Detectors

Detection methods are being migrated. Currently implemented:
- KDE (Kernel Density Estimation) - utility functions
- LID (Local Intrinsic Dimensionality) - utility functions

## Implementation Status

### Completed:
- ✅ Basic directory structure
- ✅ Common utilities
- ✅ CNN models for MNIST, CIFAR-10
- ✅ Training script
- ✅ Adversarial generation script
- ✅ KDE utility functions
- ✅ LID utility functions

### In Progress:
- 🔄 Detection scripts (detect_kde.py, detect_lid.py, etc.)
- 🔄 MagNet detection method
- 🔄 Other detection methods (FS, NSS, NIC, multiLID)

### TODO:
- ❌ SVHN CNN model (requires ResNet18 implementation)
- ❌ ImageNet support
- ❌ Complete detector implementations
- ❌ Testing and validation

## Notes

- Models are saved in HDF5 format (.h5) instead of PyTorch format (.pt)
- TensorFlow uses NHWC format by default (no need to transpose for CNNs)
- Some advanced attacks may require additional implementation