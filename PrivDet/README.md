# PrivDet: Private Dataset Origin Detection

**Supports:** [PyTorch](./pytorch) · [TensorFlow](./tensorflow) · [PaddlePaddle](./paddlepaddle) · [MindSpore](./mindspore)

---

PrivDet ships cross-framework training scripts for building a binary detector that differentiates images originating from distinct data distributions (e.g., COCO versus CIFAR-10). Each subdirectory offers a feature-matched implementation, making it easy to prototype and compare experiments across deep-learning ecosystems.

## Repository Layout
- `pytorch/`: Baseline implementation built on the `torchvision` ResNet-50.
- `tensorflow/`: Equivalent pipeline implemented with Keras/TensorFlow.
- `paddlepaddle/`: Paddle-based data loading and training scripts.
- `mindspore/`: MindSpore training scripts and inference examples.

## Quick Start
1. Prepare data by placing the Hugging Face parquet files under `data/coco` and `data/cifar10`.
2. Pick your preferred framework and switch into the corresponding folder, e.g., `cd pytorch`.
3. Launch the training script:
   ```bash
   python detect_classifier_training.py
   ```
   The script will automatically:
   - Parse the parquet files and build the combined dataset.
   - Split the data into 70/30 train/test sets.
   - Train a binary classifier and export the confusion matrix, training curves, and weights.

## Results & Outputs
- `models/`: Stores the fine-tuned detector weights.
- `confusion_matrix.png`: Automatically generated after training to visualize performance.
- Console metrics: accuracy, precision, recall, F1 score, and more.

## Notes
- Adjust constants such as `NUM_SAMPLES_PER_CLASS` or `BATCH_SIZE` to suit your dataset size.
- Set `USE_GPU=0` when running on CPU-only machines.
