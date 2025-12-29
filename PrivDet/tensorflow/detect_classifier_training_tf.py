import os
import io
import math
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

USE_GPU_ENV = os.environ.get('USE_GPU', '1').lower()
if USE_GPU_ENV in {'0', 'false', 'no'} and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from PIL import Image
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


IMAGE_SIZE = (224, 224)
MEAN = tf.constant([0.485, 0.456, 0.406], dtype=tf.float32)
STD = tf.constant([0.229, 0.224, 0.225], dtype=tf.float32)
MEAN = tf.reshape(MEAN, (1, 1, 3))
STD = tf.reshape(STD, (1, 1, 3))


class HFParquetDataset:
    """处理 Hugging Face 下载的 parquet 数据集，保存图像字节。"""

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

                        # 验证图像可读
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
    """合并 COCO 与 CIFAR 数据集并生成标签。"""

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
    """加载数据并按 7:3 划分训练与测试集"""

    full_dataset = COCOCIFARDataset(
        coco_path=coco_path,
        cifar_path=cifar_path,
        num_samples_per_class=num_samples_per_class,
    )

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)

    indices = np.arange(total_size)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = [full_dataset.data[i] for i in train_indices]
    train_labels = [full_dataset.labels[i] for i in train_indices]
    test_images = [full_dataset.data[i] for i in test_indices]
    test_labels = [full_dataset.labels[i] for i in test_indices]

    return train_images, train_labels, test_images, test_labels


def build_data_augmentation_layer() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip('horizontal'),
            tf.keras.layers.RandomRotation(0.03),
            tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0.0, 1.0)),
            tf.keras.layers.RandomContrast(factor=0.2),
            tf.keras.layers.RandomSaturation(factor=(0.8, 1.2), value_range=(0.0, 1.0)),
        ],
        name='data_augmentation',
    )


def decode_image(image_bytes: tf.Tensor) -> tf.Tensor:
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_dataset(
    images: List[bytes],
    labels: List[int],
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    data_augmentation = build_data_augmentation_layer()
    AUTOTUNE = tf.data.AUTOTUNE

    def _map_fn(image_bytes, label):
        image = decode_image(image_bytes)
        if training:
            image = data_augmentation(image, training=True)
        image = tf.keras.applications.resnet50.preprocess_input(image * 255.0)
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(buffer_size=len(images), seed=42, reshuffle_each_iteration=True)

    dataset = dataset.map(_map_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(AUTOTUNE)

    return dataset


def create_model(num_classes: int, learning_rate: float, steps_per_epoch: int) -> tf.keras.Model:
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3),
    )
    base_model.trainable = True

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    if steps_per_epoch and steps_per_epoch > 0:
        boundaries = [steps_per_epoch * 7, steps_per_epoch * 14]
        values = [learning_rate, learning_rate * 0.1, learning_rate * 0.01]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    else:
        lr_schedule = learning_rate

    optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-4)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    return model


class BestModelCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0.0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_acc = logs.get('val_accuracy')
        if val_acc is not None and val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc
            self.best_weights = self.model.get_weights()


def evaluate_model(model: tf.keras.Model, test_dataset: tf.data.Dataset, true_labels: List[int]):
    probs = model.predict(test_dataset, verbose=0)
    predictions = np.argmax(probs, axis=1)
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='weighted')
    recall = recall_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')

    print("\n=== 测试结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")

    print("\n=== 详细分类报告 ===")
    class_names = ['COCO', 'CIFAR-10']
    print(classification_report(true_labels, predictions, target_names=class_names))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': predictions,
        'labels': np.array(true_labels),
        'probabilities': probs,
    }


def plot_results(history, test_results):
    train_losses = history['loss']
    val_losses = history['val_loss']
    train_accuracies = [acc * 100 for acc in history['accuracy']]
    val_accuracies = [acc * 100 for acc in history['val_accuracy']]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(train_losses, label='训练损失', marker='o')
    axes[0, 0].plot(val_losses, label='验证损失', marker='s')
    axes[0, 0].set_title('训练过程 - 损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(train_accuracies, label='训练准确率', marker='o')
    axes[0, 1].plot(val_accuracies, label='验证准确率', marker='s')
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
    plt.savefig('training_results_tf.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('confusion_matrix_tf.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    COCO_PATH = os.path.join('data', 'coco')
    CIFAR_PATH = os.path.join('data', 'cifar10')

    NUM_SAMPLES_PER_CLASS = int(os.environ.get('NUM_SAMPLES_PER_CLASS', 1000))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 32))
    NUM_EPOCHS = int(os.environ.get('NUM_EPOCHS', 15))
    LEARNING_RATE = float(os.environ.get('LEARNING_RATE', 0.001))

    print("=== COCO-CIFAR二分类训练程序 (TensorFlow) ===")
    print(f"COCO数据路径: {COCO_PATH}")
    print(f"CIFAR数据路径: {CIFAR_PATH}")
    print(f"每个类别使用 {NUM_SAMPLES_PER_CLASS} 个样本")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")

    use_gpu = os.environ.get('USE_GPU', '1').lower() not in {'0', 'false', 'no'}
    if not use_gpu:
        print("已禁用 GPU，使用 CPU 训练")
    else:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"检测到 {len(gpus)} 个 GPU")
        except Exception as gpu_err:  # noqa: BLE001
            print(f"GPU 配置失败，将回退到 CPU: {gpu_err}")
            tf.config.set_visible_devices([], 'GPU')

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

    train_dataset = preprocess_dataset(train_images, train_labels, BATCH_SIZE, training=True)
    test_dataset = preprocess_dataset(test_images, test_labels, BATCH_SIZE, training=False)

    print(f"训练集大小: {len(train_labels)}")
    print(f"测试集大小: {len(test_labels)}")

    steps_per_epoch = math.ceil(len(train_labels) / BATCH_SIZE)

    print("\n2. 创建ResNet50模型...")
    model = create_model(num_classes=2, learning_rate=LEARNING_RATE, steps_per_epoch=steps_per_epoch)

    print("\n3. 开始训练模型...")
    best_callback = BestModelCallback()
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=NUM_EPOCHS,
        callbacks=[best_callback],
        verbose=1,
    )

    if best_callback.best_weights is not None:
        model.set_weights(best_callback.best_weights)

    best_val_acc = best_callback.best_val_accuracy * 100

    print("\n4. 评估模型性能...")
    test_results = evaluate_model(model, test_dataset, test_labels)

    print("\n5. 生成结果图表...")
    plot_results(history.history, test_results)

    print("\n6. 保存模型...")
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'coco_cifar_resnet50_tf.keras')
    model.save(model_path)
    print(f"模型已保存为 '{model_path}'")

    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"最终测试准确率: {test_results['accuracy']:.4f}")


if __name__ == '__main__':
    main()
