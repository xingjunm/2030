import os
import io
import math
import random
import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# 控制 MindSpore 的设备在导入前设置
USE_GPU_ENV = os.environ.get('USE_GPU', '1').lower()
if USE_GPU_ENV in {'0', 'false', 'no'} and 'DEVICE_TARGET' not in os.environ:
    os.environ['DEVICE_TARGET'] = 'CPU'

import mindspore as ms
from mindspore import context, nn, ops, Tensor
import mindspore.dataset as ds
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

IMAGE_SIZE = (224, 224)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def set_random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class HFParquetDataset:
    """处理 Hugging Face 下载的 parquet 数据集，存储图像字节"""

    def __init__(
        self,
        data_path: str,
        dataset_type: str = 'coco',
        max_samples: Optional[int] = None,
    ):
        self.dataset_type = dataset_type
        self.data: List[bytes] = []

        print(f"正在加载 {dataset_type} 数据集从路径: {data_path}")

        parquet_files = []
        if os.path.exists(data_path):
            for root, _, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))

        if not parquet_files:
            raise FileNotFoundError(f"在路径 {data_path} 中没有找到parquet文件")

        print(f"找到 {len(parquet_files)} 个parquet文件")

        loaded_samples = 0
        for parquet_file in parquet_files:
            if max_samples and loaded_samples >= max_samples:
                break

            print(f"正在处理: {parquet_file}")
            try:
                df = pd.read_parquet(parquet_file)
                print(f"数据列: {df.columns.tolist()}")
                print(f"数据形状: {df.shape}")

                if dataset_type == 'coco':
                    if 'image' in df.columns:
                        image_col = 'image'
                    elif 'img' in df.columns:
                        image_col = 'img'
                    else:
                        image_col = next(
                            (col for col in df.columns if 'image' in col.lower() or 'img' in col.lower()),
                            None,
                        )
                        if image_col is None:
                            raise ValueError(
                                f"在COCO数据中找不到图像列，可用列: {df.columns.tolist()}"
                            )
                elif dataset_type == 'cifar10':
                    if 'img' in df.columns:
                        image_col = 'img'
                    elif 'image' in df.columns:
                        image_col = 'image'
                    else:
                        image_col = next(
                            (col for col in df.columns if 'image' in col.lower() or 'img' in col.lower()),
                            None,
                        )
                        if image_col is None:
                            raise ValueError(
                                f"在CIFAR-10数据中找不到图像列，可用列: {df.columns.tolist()}"
                            )
                else:
                    raise ValueError(f"不支持的数据集类型: {dataset_type}")

                for _, row in df.iterrows():
                    if max_samples and loaded_samples >= max_samples:
                        break

                    try:
                        image_data = row[image_col]

                        if isinstance(image_data, dict):
                            if 'bytes' in image_data:
                                image_bytes = image_data['bytes']
                            elif 'path' in image_data and isinstance(image_data['path'], str):
                                with open(image_data['path'], 'rb') as f:
                                    image_bytes = f.read()
                            else:
                                image_bytes = next(iter(image_data.values()))
                        elif isinstance(image_data, bytes):
                            image_bytes = image_data
                        else:
                            image_bytes = bytes(image_data)

                        # 验证图像
                        Image.open(io.BytesIO(image_bytes)).convert('RGB')

                        self.data.append(image_bytes)
                        loaded_samples += 1

                        if loaded_samples % 100 == 0:
                            print(f"已加载 {loaded_samples} 个样本...")

                    except Exception as err:  # noqa: BLE001
                        print(f"处理样本时出错: {err}")
                        continue

            except Exception as err:  # noqa: BLE001
                print(f"读取文件 {parquet_file} 时出错: {err}")
                continue

        print(f"{dataset_type} 数据集加载完成，共 {loaded_samples} 个样本")

        if loaded_samples == 0:
            raise ValueError(f"没有成功加载任何 {dataset_type} 样本")

    def __len__(self) -> int:
        return len(self.data)


class COCOCIFARDataset:
    """合并 COCO 与 CIFAR 数据集，分别赋予 0/1 标签"""

    def __init__(self, coco_path: str, cifar_path: str, num_samples_per_class: int):
        self.data: List[bytes] = []
        self.labels: List[int] = []

        print(f"开始构建合并数据集，每个类别 {num_samples_per_class} 个样本")

        print("正在加载COCO数据...")
        coco_dataset = HFParquetDataset(coco_path, 'coco', num_samples_per_class)
        for img_bytes in coco_dataset.data:
            self.data.append(img_bytes)
            self.labels.append(0)

        print("正在加载CIFAR-10数据...")
        cifar_dataset = HFParquetDataset(cifar_path, 'cifar10', num_samples_per_class)
        for img_bytes in cifar_dataset.data:
            self.data.append(img_bytes)
            self.labels.append(1)

        print(
            f"数据集构建完成: COCO样本 {len(coco_dataset.data)} 个, CIFAR样本 {len(cifar_dataset.data)} 个"
        )
        print(f"总样本数: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)


def load_datasets(
    coco_path: str,
    cifar_path: str,
    num_samples_per_class: int = 1000,
) -> Tuple[List[bytes], List[int], List[bytes], List[int]]:
    dataset = COCOCIFARDataset(
        coco_path=coco_path,
        cifar_path=cifar_path,
        num_samples_per_class=num_samples_per_class,
    )

    total_size = len(dataset)
    train_size = int(0.7 * total_size)

    indices = np.arange(total_size)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = [dataset.data[i] for i in train_indices]
    train_labels = [dataset.labels[i] for i in train_indices]
    test_images = [dataset.data[i] for i in test_indices]
    test_labels = [dataset.labels[i] for i in test_indices]

    return train_images, train_labels, test_images, test_labels


def random_rotate(image: Image.Image, max_degrees: float = 10.0) -> Image.Image:
    angle = random.uniform(-max_degrees, max_degrees)
    return image.rotate(angle)


def random_color_jitter(image: Image.Image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1) -> Image.Image:
    if brightness > 0:
        enhancer = ImageEnhance.Brightness(image)
        factor = 1 + random.uniform(-brightness, brightness)
        image = enhancer.enhance(max(factor, 0))
    if contrast > 0:
        enhancer = ImageEnhance.Contrast(image)
        factor = 1 + random.uniform(-contrast, contrast)
        image = enhancer.enhance(max(factor, 0))
    if saturation > 0:
        enhancer = ImageEnhance.Color(image)
        factor = 1 + random.uniform(-saturation, saturation)
        image = enhancer.enhance(max(factor, 0))
    if hue > 0:
        hsv = np.array(image.convert('HSV'))
        hue_shift = int(random.uniform(-hue, hue) * 255)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 255
        image = Image.fromarray(hsv, mode='HSV').convert('RGB')
    return image


def preprocess_image(image_bytes: bytes, training: bool) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMAGE_SIZE)

    if training:
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = random_rotate(image)
        image = random_color_jitter(image)

    image_np = np.asarray(image).astype(np.float32) / 255.0
    image_np = (image_np - MEAN) / STD
    image_np = np.transpose(image_np, (2, 0, 1))  # CHW
    return image_np


class MindSporeGenerator:
    def __init__(self, images: List[bytes], labels: List[int], training: bool):
        self.images = images
        self.labels = labels
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = preprocess_image(self.images[idx], self.training)
        label = np.int32(self.labels[idx])
        return image, label

    def to_generator(self):
        for idx in range(len(self.images)):
            yield self.__getitem__(idx)


def create_dataset(images: List[bytes], labels: List[int], batch_size: int, training: bool) -> ds.GeneratorDataset:
    generator = MindSporeGenerator(images, labels, training)
    dataset = ds.GeneratorDataset(source=generator, column_names=['image', 'label'], shuffle=training)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset


def create_model(num_classes: int = 2) -> nn.Cell:
    errors = []

    try:
        from mindspore.vision.models import resnet50
        network = resnet50(pretrained=True)
        classifier_attr = 'classifier' if hasattr(network, 'classifier') else 'fc'
    except Exception as err:  # noqa: BLE001
        errors.append(f"mindspore.vision.models.resnet50 预训练加载失败: {err}")
        network = None
        classifier_attr = None

    if network is None:
        try:
            from mindcv.models import resnet50  # type: ignore
            network = resnet50(pretrained=True)
            classifier_attr = 'classifier' if hasattr(network, 'classifier') else 'fc'
        except Exception as err:  # noqa: BLE001
            errors.append(f"mindcv.models.resnet50 预训练加载失败: {err}")
            raise RuntimeError(
                "无法加载预训练 ResNet50 模型，请确认已安装 mindspore.vision 或 mindcv 并预先准备好权重。\n"
                + "\n".join(errors)
            ) from err

    if classifier_attr == 'classifier':
        in_channels = network.classifier.in_channels
        network.classifier = nn.Dense(in_channels, num_classes)
    elif classifier_attr == 'fc':
        in_channels = network.fc.in_channels
        network.fc = nn.Dense(in_channels, num_classes)
    elif classifier_attr is None:
        pass
    else:
        raise AttributeError('ResNet50 模型缺少可识别的分类层 (classifier/fc)。')

    return network


def train_model(
    network: nn.Cell,
    train_dataset: ds.GeneratorDataset,
    val_dataset: ds.GeneratorDataset,
    num_epochs: int,
    learning_rate: float,
    steps_per_epoch: int,
):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    if steps_per_epoch > 0:
        total_steps = num_epochs * steps_per_epoch
        lr_array = np.full(total_steps, learning_rate, dtype=np.float32)
        if num_epochs > 7:
            lr_array[7 * steps_per_epoch:] = learning_rate * 0.1
        if num_epochs > 14:
            lr_array[14 * steps_per_epoch:] = learning_rate * 0.01
        lr = Tensor(lr_array)
    else:
        lr = Tensor([learning_rate], ms.float32)

    optimizer = nn.Adam(params=network.trainable_params(), learning_rate=lr, weight_decay=1e-4)

    loss_net = nn.WithLossCell(network, loss_fn)
    train_step = nn.TrainOneStepCell(loss_net, optimizer)
    train_step.set_train()

    best_val_acc = 0.0
    best_ckpt_path = os.path.join('models', 'best_mindspore_tmp.ckpt')
    history = {
        'train_losses': [],
        'train_accuracies': [],
        'val_losses': [],
        'val_accuracies': [],
    }

    argmax = ops.Argmax(axis=1)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        network.set_train(True)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in train_dataset.create_dict_iterator():
            images = Tensor(batch['image'])
            labels = Tensor(batch['label'])

            loss = train_step(images, labels)
            logits = network(images)
            preds = argmax(logits)
            correct = ops.equal(preds, labels).astype(ms.float32).sum().asnumpy()

            total_loss += float(loss.asnumpy())
            total_correct += correct
            total_samples += labels.shape[0]

        epoch_train_loss = total_loss / max(steps_per_epoch, 1)
        epoch_train_acc = 100.0 * total_correct / max(total_samples, 1)

        network.set_train(False)
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        ops_softmax = ops.Softmax(axis=1)
        for batch in val_dataset.create_dict_iterator():
            images = Tensor(batch['image'])
            labels = Tensor(batch['label'])
            logits = network(images)
            loss = loss_fn(logits, labels)
            preds = argmax(logits)

            val_loss += float(loss.asnumpy())
            val_correct += ops.equal(preds, labels).astype(ms.float32).sum().asnumpy()
            val_total += labels.shape[0]

        epoch_val_loss = val_loss / max(len(val_dataset), 1)
        epoch_val_acc = 100.0 * val_correct / max(val_total, 1)

        history['train_losses'].append(epoch_train_loss)
        history['train_accuracies'].append(epoch_train_acc)
        history['val_losses'].append(epoch_val_loss)
        history['val_accuracies'].append(epoch_val_acc)

        print(f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.2f}%")
        print(f"验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_acc:.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            os.makedirs('models', exist_ok=True)
            save_checkpoint(network, best_ckpt_path)

        train_dataset.reset()
        val_dataset.reset()

    if os.path.exists(best_ckpt_path):
        param_dict = load_checkpoint(best_ckpt_path)
        load_param_into_net(network, param_dict)
        os.remove(best_ckpt_path)

    return network, history, best_val_acc


def evaluate_model(network: nn.Cell, dataset: ds.GeneratorDataset):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    network.set_train(False)
    all_labels = []
    all_predictions = []
    all_probs = []
    total_loss = 0.0
    total_batches = 0

    ops_softmax = ops.Softmax(axis=1)
    argmax = ops.Argmax(axis=1)

    for batch in dataset.create_dict_iterator():
        images = Tensor(batch['image'])
        labels = Tensor(batch['label'])

        logits = network(images)
        loss = loss_fn(logits, labels)
        probs = ops_softmax(logits)
        preds = argmax(logits)

        total_loss += float(loss.asnumpy())
        total_batches += 1

        all_labels.extend(labels.asnumpy().tolist())
        all_predictions.extend(preds.asnumpy().tolist())
        all_probs.extend(probs.asnumpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')

    print("\n=== 测试结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print("\n=== 详细分类报告 ===")
    class_names = ['COCO', 'CIFAR-10']
    print(classification_report(all_labels, all_predictions, target_names=class_names))

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': np.array(all_probs),
        'loss': total_loss / max(total_batches, 1),
    }

    dataset.reset()
    return results


def plot_results(history, test_results):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(history['train_losses'], label='训练损失', marker='o')
    axes[0, 0].plot(history['val_losses'], label='验证损失', marker='s')
    axes[0, 0].set_title('训练过程 - 损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(history['train_accuracies'], label='训练准确率', marker='o')
    axes[0, 1].plot(history['val_accuracies'], label='验证准确率', marker='s')
    axes[0, 1].set_title('训练过程 - 准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['COCO', 'CIFAR-10'],
        yticklabels=['COCO', 'CIFAR-10'],
        ax=axes[1, 0],
    )
    axes[1, 0].set_title('混淆矩阵')
    axes[1, 0].set_xlabel('预测标签')
    axes[1, 0].set_ylabel('真实标签')

    probs = np.array(test_results['probabilities'])
    axes[1, 1].hist(probs[:, 0], bins=50, alpha=0.7, label='COCO类概率', color='blue')
    axes[1, 1].hist(probs[:, 1], bins=50, alpha=0.7, label='CIFAR-10类概率', color='orange')
    axes[1, 1].set_title('预测概率分布')
    axes[1, 1].set_xlabel('概率')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('training_results_mindspore.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['COCO', 'CIFAR-10'],
        yticklabels=['COCO', 'CIFAR-10'],
    )
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix_mindspore.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    set_random_seed(42)

    device_target = os.environ.get('DEVICE_TARGET', None)
    use_gpu = USE_GPU_ENV not in {'0', 'false', 'no'}

    if device_target:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)
        print(f"使用显式设置的设备: {device_target}")
    else:
        target = 'GPU' if use_gpu else 'CPU'
        context.set_context(mode=context.PYNATIVE_MODE, device_target=target)
        print(f"设备自动设置为: {target}")

    COCO_PATH = os.path.join('data', 'coco')
    CIFAR_PATH = os.path.join('data', 'cifar10')

    NUM_SAMPLES_PER_CLASS = int(os.environ.get('NUM_SAMPLES_PER_CLASS', 1000))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 15))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.001))

    print("=== COCO-CIFAR二分类训练程序 (MindSpore) ===")
    print(f"COCO数据路径: {COCO_PATH}")
    print(f"CIFAR数据路径: {CIFAR_PATH}")
    print(f"每个类别使用 {NUM_SAMPLES_PER_CLASS} 个样本")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")

    if not os.path.exists(COCO_PATH):
        raise FileNotFoundError(f"COCO数据路径不存在: {COCO_PATH}")
    if not os.path.exists(CIFAR_PATH):
        raise FileNotFoundError(f"CIFAR数据路径不存在: {CIFAR_PATH}")

    print("\n1. 加载和预处理数据集...")
    train_images, train_labels, test_images, test_labels = load_datasets(
        coco_path=COCO_PATH,
        cifar_path=CIFAR_PATH,
        num_samples_per_class=NUM_SAMPLES_PER_CLASS,
    )

    train_dataset = create_dataset(train_images, train_labels, BATCH_SIZE, training=True)
    test_dataset = create_dataset(test_images, test_labels, BATCH_SIZE, training=False)

    print(f"训练集大小: {len(train_labels)}")
    print(f"测试集大小: {len(test_labels)}")

    steps_per_epoch = math.ceil(len(train_labels) / BATCH_SIZE)

    print("\n2. 创建ResNet50模型...")
    network = create_model(num_classes=2)

    print("\n3. 开始训练模型...")
    network, history, best_val_acc = train_model(
        network,
        train_dataset,
        test_dataset,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
    )

    print("\n4. 评估模型性能...")
    test_results = evaluate_model(network, test_dataset)

    print("\n5. 生成结果图表...")
    plot_results(history, test_results)

    print("\n6. 保存模型...")
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'coco_cifar_resnet50_mindspore.ckpt')
    save_checkpoint(network, model_path)
    print(f"模型已保存为 '{model_path}'")

    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最终测试准确率: {test_results['accuracy']:.4f}")


if __name__ == '__main__':
    main()
