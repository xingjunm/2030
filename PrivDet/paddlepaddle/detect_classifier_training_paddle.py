import os
import io
import copy
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import Dataset, DataLoader, random_split
from paddle.vision import transforms
from paddle.vision.models import resnet50
from PIL import Image
from tqdm import tqdm

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class HFParquetDataset(Dataset):
    """处理 Hugging Face 下载的 parquet 格式数据集"""

    def __init__(
        self,
        data_path: str,
        dataset_type: str = 'coco',
        max_samples: Optional[int] = None,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.dataset_type = dataset_type
        self.data = []

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

                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        self.data.append(image)
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

    def __getitem__(self, idx: int) -> paddle.Tensor:
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        return image


class COCOCIFARDataset(Dataset):
    """合并 COCO 与 CIFAR 数据集并生成标签。COCO 为 0，CIFAR 为 1。"""

    def __init__(
        self,
        coco_path: str,
        cifar_path: str,
        num_samples_per_class: int,
        transform=None,
    ):
        super().__init__()
        self.transform = transform
        self.data = []
        self.labels = []

        print(f"开始构建合并数据集，每个类别 {num_samples_per_class} 个样本")

        print("正在加载COCO数据...")
        coco_dataset = HFParquetDataset(coco_path, 'coco', num_samples_per_class)
        for img in coco_dataset.data:
            self.data.append(img)
            self.labels.append(0)

        print("正在加载CIFAR-10数据...")
        cifar_dataset = HFParquetDataset(cifar_path, 'cifar10', num_samples_per_class)
        for img in cifar_dataset.data:
            self.data.append(img)
            self.labels.append(1)

        print(
            f"数据集构建完成: COCO样本 {len(coco_dataset.data)} 个, CIFAR样本 {len(cifar_dataset.data)} 个"
        )
        print(f"总样本数: {len(self.data)}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[paddle.Tensor, int]:
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


def load_datasets(coco_path: str, cifar_path: str, num_samples_per_class: int = 1000):
    """加载数据集并按 7:3 划分训练 / 测试集"""

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = COCOCIFARDataset(
        coco_path=coco_path,
        cifar_path=cifar_path,
        num_samples_per_class=num_samples_per_class,
        transform=train_transform,
    )

    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    paddle.seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    class TestDatasetWrapper(Dataset):
        def __init__(self, subset, transform):
            super().__init__()
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            real_idx = self.subset.indices[idx]
            image = self.subset.dataset.data[real_idx]
            label = self.subset.dataset.labels[real_idx]

            if self.transform:
                image = self.transform(image)
            else:
                image = transforms.ToTensor()(image)

            return image, label

    test_dataset_wrapped = TestDatasetWrapper(test_dataset, test_transform)

    return train_dataset, test_dataset_wrapped


def create_model(num_classes: int = 2) -> nn.Layer:
    """创建 PaddlePaddle 版 ResNet50 并替换分类头"""

    model = resnet50(pretrained=True)
    in_features = model.fc.weight.shape[0]
    model.fc = nn.Linear(in_features=in_features, out_features=num_classes)
    return model


def set_device() -> str:
    if paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        device = 'gpu'
    else:
        device = 'cpu'
    paddle.device.set_device(device)
    print(f"使用设备: {device}")
    return device


def train_model(model, train_loader, val_loader, num_epochs: int = 15, learning_rate: float = 0.001):
    """训练模型，返回最佳模型以及训练记录"""

    device = set_device()

    if device == 'gpu' and paddle.distributed.ParallelEnv().nranks > 1:
        model = paddle.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    lr_scheduler = optim.lr.StepDecay(learning_rate=learning_rate, step_size=7, gamma=0.1)
    optimizer = optim.Adam(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=1e-4)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        train_pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_pbar:
            inputs = inputs.astype('float32')
            labels = paddle.to_tensor(labels, dtype='int64')

            optimizer.clear_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = paddle.argmax(outputs, axis=1)
            total_samples += labels.shape[0]
            correct_predictions += paddle.sum(preds == labels).item()

            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct_predictions / total_samples:.2f}%'
            })

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100.0 * correct_predictions / total_samples

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with paddle.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs = inputs.astype('float32')
                labels = paddle.to_tensor(labels, dtype='int64')

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = paddle.argmax(outputs, axis=1)
                val_total += labels.shape[0]
                val_correct += paddle.sum(preds == labels).item()

                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.0 * val_correct / val_total:.2f}%'
                })

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100.0 * val_correct / val_total

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = copy.deepcopy(model.state_dict())

        print(f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.2f}%")
        print(f"验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_acc:.2f}%")

        lr_scheduler.step()

    if best_model_state is not None:
        model.set_state_dict(best_model_state)

    history = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc,
    }

    return model, history


def evaluate_model(model, test_loader):
    """在测试集上评估模型并返回指标"""

    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with paddle.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for inputs, labels in test_pbar:
            inputs = inputs.astype('float32')
            labels_tensor = paddle.to_tensor(labels, dtype='int64')

            outputs = model(inputs)
            probs = paddle.nn.functional.softmax(outputs, axis=1)
            preds = paddle.argmax(outputs, axis=1)

            all_predictions.extend(preds.numpy())
            all_labels.extend(labels_tensor.numpy())
            all_probs.extend(probs.numpy())

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

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
    }


def plot_results(history, test_results):
    """绘制训练与评估结果图表"""

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
    plt.savefig('training_results_paddle.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('confusion_matrix_paddle.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """主训练入口"""

    COCO_PATH = os.path.join('data', 'coco')
    CIFAR_PATH = os.path.join('data', 'cifar10')

    NUM_SAMPLES_PER_CLASS = int(os.environ.get('NUM_SAMPLES_PER_CLASS', 1000))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 15))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.001))

    print("=== COCO-CIFAR二分类训练程序 (PaddlePaddle) ===")
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
    train_dataset, test_dataset = load_datasets(
        coco_path=COCO_PATH,
        cifar_path=CIFAR_PATH,
        num_samples_per_class=NUM_SAMPLES_PER_CLASS,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    print("\n2. 创建ResNet50模型...")
    model = create_model(num_classes=2)

    print("\n3. 开始训练模型...")
    trained_model, history = train_model(
        model,
        train_loader,
        test_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
    )

    print("\n4. 评估模型性能...")
    test_results = evaluate_model(trained_model, test_loader)

    print("\n5. 生成结果图表...")
    plot_results(history, test_results)

    print("\n6. 保存模型...")
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'coco_cifar_resnet50_paddle.pdparams')
    paddle.save(trained_model.state_dict(), model_path)
    print(f"模型已保存为 '{model_path}'")

    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
    print(f"最终测试准确率: {test_results['accuracy']:.4f}")


if __name__ == '__main__':
    main()
