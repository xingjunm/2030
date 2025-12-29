import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from PIL import Image
import pandas as pd
import os
import io
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HFParquetDataset(Dataset):
    """
    处理Hugging Face下载的parquet格式数据集
    """
    def __init__(self, data_path, dataset_type='coco', max_samples=None, transform=None):
        self.transform = transform
        self.data = []
        self.dataset_type = dataset_type
        
        print(f"正在加载 {dataset_type} 数据集从路径: {data_path}")
        
        # 获取所有parquet文件
        parquet_files = []
        if os.path.exists(data_path):
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    if file.endswith('.parquet'):
                        parquet_files.append(os.path.join(root, file))
        
        if not parquet_files:
            raise FileNotFoundError(f"在路径 {data_path} 中没有找到parquet文件")
        
        print(f"找到 {len(parquet_files)} 个parquet文件")
        
        # 读取parquet文件
        loaded_samples = 0
        for parquet_file in parquet_files:
            if max_samples and loaded_samples >= max_samples:
                break
                
            print(f"正在处理: {parquet_file}")
            try:
                df = pd.read_parquet(parquet_file)
                
                # 检查数据结构
                print(f"数据列: {df.columns.tolist()}")
                print(f"数据形状: {df.shape}")
                
                # 根据数据集类型处理不同的列名
                if dataset_type == 'coco':
                    # COCO数据集通常有'image'列
                    if 'image' in df.columns:
                        image_col = 'image'
                    elif 'img' in df.columns:
                        image_col = 'img'
                    else:
                        # 查找可能包含图像数据的列
                        for col in df.columns:
                            if 'image' in col.lower() or 'img' in col.lower():
                                image_col = col
                                break
                        else:
                            raise ValueError(f"在COCO数据中找不到图像列，可用列: {df.columns.tolist()}")
                
                elif dataset_type == 'cifar10':
                    # CIFAR-10数据集通常有'img'或'image'列
                    if 'img' in df.columns:
                        image_col = 'img'
                    elif 'image' in df.columns:
                        image_col = 'image'
                    else:
                        # 查找可能包含图像数据的列
                        for col in df.columns:
                            if 'image' in col.lower() or 'img' in col.lower():
                                image_col = col
                                break
                        else:
                            raise ValueError(f"在CIFAR-10数据中找不到图像列，可用列: {df.columns.tolist()}")
                
                # 处理图像数据
                for idx, row in df.iterrows():
                    if max_samples and loaded_samples >= max_samples:
                        break
                    
                    try:
                        # 获取图像数据
                        image_data = row[image_col]
                        
                        # 处理不同格式的图像数据
                        if isinstance(image_data, dict):
                            # 如果是字典格式，通常包含'bytes'键
                            if 'bytes' in image_data:
                                image_bytes = image_data['bytes']
                            elif 'path' in image_data:
                                # 如果有路径信息，跳过（因为我们处理的是嵌入的图像）
                                continue
                            else:
                                # 尝试直接使用字典中的数据
                                image_bytes = list(image_data.values())[0]
                        elif isinstance(image_data, bytes):
                            image_bytes = image_data
                        else:
                            # 尝试转换为bytes
                            image_bytes = bytes(image_data)
                        
                        # 将bytes转换为PIL Image
                        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                        self.data.append(image)
                        loaded_samples += 1
                        
                        if loaded_samples % 100 == 0:
                            print(f"已加载 {loaded_samples} 个样本...")
                    
                    except Exception as e:
                        print(f"处理样本时出错: {e}")
                        continue
                
            except Exception as e:
                print(f"读取文件 {parquet_file} 时出错: {e}")
                continue
        
        print(f"{dataset_type} 数据集加载完成，共 {loaded_samples} 个样本")
        
        if loaded_samples == 0:
            raise ValueError(f"没有成功加载任何 {dataset_type} 样本")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        if self.transform:
            image = self.transform(image)
        return image

class COCOCIFARDataset(Dataset):
    """
    合并COCO和CIFAR数据集的自定义数据集
    COCO样本标记为0，CIFAR样本标记为1
    """
    def __init__(self, coco_path, cifar_path, num_samples_per_class, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        
        print(f"开始构建合并数据集，每个类别 {num_samples_per_class} 个样本")
        
        # 加载COCO数据
        print("正在加载COCO数据...")
        coco_dataset = HFParquetDataset(coco_path, 'coco', num_samples_per_class)
        for img in coco_dataset.data:
            self.data.append(img)
            self.labels.append(0)  # COCO标记为0
        
        # 加载CIFAR数据
        print("正在加载CIFAR-10数据...")
        cifar_dataset = HFParquetDataset(cifar_path, 'cifar10', num_samples_per_class)
        for img in cifar_dataset.data:
            self.data.append(img)
            self.labels.append(1)  # CIFAR标记为1
        
        print(f"数据集构建完成: COCO样本 {len(coco_dataset.data)} 个, CIFAR样本 {len(cifar_dataset.data)} 个")
        print(f"总样本数: {len(self.data)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_datasets(coco_path, cifar_path, num_samples_per_class=1000):
    """加载和预处理数据集"""
    
    # 创建训练和测试的变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # 创建合并数据集
    full_dataset = COCOCIFARDataset(
        coco_path=coco_path,
        cifar_path=cifar_path,
        num_samples_per_class=num_samples_per_class,
        transform=train_transform
    )
    
    # 7:3分割训练集和测试集
    train_size = int(0.7 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # 设置随机种子以确保可重现的分割
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # 为测试集创建新的数据集实例以应用不同的变换
    class TestDatasetWrapper:
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            # 获取原始数据
            real_idx = self.dataset.indices[idx]
            image = self.dataset.dataset.data[real_idx]
            label = self.dataset.dataset.labels[real_idx]
            
            # 应用测试变换
            if self.transform:
                image = self.transform(image)
            
            return image, label
    
    test_dataset_wrapped = TestDatasetWrapper(test_dataset, test_transform)
    
    return train_dataset, test_dataset_wrapped



def create_model(num_classes=2):
    """创建ResNet50模型"""
    from transformers import ResNetForImageClassification

    # 从本地缓存的 Hugging Face 权重加载 ResNet50
    model = ResNetForImageClassification.from_pretrained(
        'microsoft/resnet-50',
        cache_dir='models/resnet50',
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    return model


def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    """训练模型"""
    
    # 检查GPU可用性并使用所有GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # 记录训练过程
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits  # 获取实际的预测logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct_predictions / total_samples:.2f}%'
            })
        
        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100. * correct_predictions / total_samples
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation")
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits  # 获取实际的预测logits
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * val_correct / val_total:.2f}%'
                })
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100. * val_correct / val_total
        
        # 记录指标
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        # 保存最佳模型
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"训练损失: {epoch_train_loss:.4f}, 训练准确率: {epoch_train_acc:.2f}%")
        print(f"验证损失: {epoch_val_loss:.4f}, 验证准确率: {epoch_val_acc:.2f}%")
        
        scheduler.step()
    
    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    return model, {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_val_acc': best_val_acc
    }




def evaluate_model(model, test_loader):
    """评估模型并生成详细指标"""
    device = next(model.parameters()).device
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing")
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits
            
            # 获取预测概率
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算各种指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"\n=== 测试结果 ===")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall): {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 详细分类报告
    print(f"\n=== 详细分类报告 ===")
    class_names = ['COCO', 'CIFAR-10']
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }

def plot_results(history, test_results):
    """绘制训练过程和结果"""
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练和验证损失
    axes[0, 0].plot(history['train_losses'], label='训练损失', marker='o')
    axes[0, 0].plot(history['val_losses'], label='验证损失', marker='s')
    axes[0, 0].set_title('训练过程 - 损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练和验证准确率
    axes[0, 1].plot(history['train_accuracies'], label='训练准确率', marker='o')
    axes[0, 1].plot(history['val_accuracies'], label='验证准确率', marker='s')
    axes[0, 1].set_title('训练过程 - 准确率')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 混淆矩阵
    cm = confusion_matrix(test_results['labels'], test_results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['COCO', 'CIFAR-10'],
                yticklabels=['COCO', 'CIFAR-10'],
                ax=axes[1, 0])
    axes[1, 0].set_title('混淆矩阵')
    axes[1, 0].set_xlabel('预测标签')
    axes[1, 0].set_ylabel('真实标签')
    
    # 预测概率分布
    probs = np.array(test_results['probabilities'])
    axes[1, 1].hist(probs[:, 0], bins=50, alpha=0.7, label='COCO类概率', color='blue')
    axes[1, 1].hist(probs[:, 1], bins=50, alpha=0.7, label='CIFAR-10类概率', color='orange')
    axes[1, 1].set_title('预测概率分布')
    axes[1, 1].set_xlabel('概率')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存混淆矩阵为单独图片
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['COCO', 'CIFAR-10'],
                yticklabels=['COCO', 'CIFAR-10'])
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    
    # 数据集路径 - 使用当前目录下准备好的数据
    COCO_PATH = os.path.join('data', 'coco')
    CIFAR_PATH = os.path.join('data', 'cifar10')

    # 设置参数（保持与原配置一致，样本数根据当前数据集体量适当缩减）
    NUM_SAMPLES_PER_CLASS = 1000  # 每个类别的样本数
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LEARNING_RATE = 0.001
    
    print("=== COCO-CIFAR二分类训练程序 ===")
    print(f"COCO数据路径: {COCO_PATH}")
    print(f"CIFAR数据路径: {CIFAR_PATH}")
    print(f"每个类别使用 {NUM_SAMPLES_PER_CLASS} 个样本")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"学习率: {LEARNING_RATE}")
    
    # 检查路径是否存在
    if not os.path.exists(COCO_PATH):
        raise FileNotFoundError(f"COCO数据路径不存在: {COCO_PATH}")
    if not os.path.exists(CIFAR_PATH):
        raise FileNotFoundError(f"CIFAR数据路径不存在: {CIFAR_PATH}")
    
    # 加载数据集
    print("\n1. 加载和预处理数据集...")
    train_dataset, test_dataset = load_datasets(
        coco_path=COCO_PATH,
        cifar_path=CIFAR_PATH,
        num_samples_per_class=NUM_SAMPLES_PER_CLASS
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    print("\n2. 创建ResNet50模型...")
    model = create_model(num_classes=2)
    
    # 训练模型
    print("\n3. 开始训练模型...")
    trained_model, history = train_model(
        model, train_loader, test_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE
    )
    
    # 评估模型
    print("\n4. 评估模型性能...")
    test_results = evaluate_model(trained_model, test_loader)
    
    # 绘制结果
    print("\n5. 生成结果图表...")
    plot_results(history, test_results)
    
    # 保存模型
    print("\n6. 保存模型...")
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'coco_cifar_resnet50_0915.pth')
    torch.save(trained_model.state_dict(), model_path)
    print(f"模型已保存为 '{model_path}'")
    
    print("\n=== 训练完成 ===")
    print(f"最佳验证准确率: {history['best_val_acc']:.2f}%")
    print(f"最终测试准确率: {test_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
