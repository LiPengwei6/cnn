import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter
from tqdm import tqdm
from typing import List, Tuple, Dict

# 设置中文字体支持 - 使用更健壮的方法
try:
    # 尝试多种常见中文字体
    font_names = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Source Han Sans CN', 'PingFang SC', 'Noto Sans CJK SC']
    
    chinese_font = None
    for font_name in font_names:
        # 检查系统中是否有这个字体
        font_path = fm.findfont(fm.FontProperties(family=font_name))
        if font_path and not 'DejaVuSans.ttf' in font_path:  # 避免降级到默认字体
            chinese_font = font_name
            break
    
    if chinese_font:
        plt.rcParams['font.sans-serif'] = [chinese_font]
        plt.rcParams['axes.unicode_minus'] = False
        print(f"成功设置中文字体: {chinese_font}")
    else:
        print("警告: 未找到中文字体，可能无法正确显示中文")
except Exception as e:
    print(f"设置中文字体时出错: {str(e)}")
    print("继续使用默认字体")

# --------------- 全局配置参数 ---------------
CONFIG = {
    "img_size": (250, 250),
    "batch_size": 64,
    "learning_rate": 1e-4,
    "epochs": 10,
    "patience": 3,
    "reduction_ratio": 16,
    "kernel_size": 7,
    "random_seed": 42,
    "num_classes": 3
}

# 设置随机种子，确保可重复性
torch.manual_seed(CONFIG["random_seed"])
np.random.seed(CONFIG["random_seed"])

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------- 技术路线与改进点 ---------------
# - 本代码示例涵盖：
#   requirements.txt) 数据集准备与预处理（类别平衡、增强等）
#   2) 改进CNN网络结构（残差+CBAM注意力）
#   3) 数据增强策略
#   4) 训练过程与指标评估

# ============== 数据集介绍与预处理 ===============
class InvalidDatasetException(Exception):
    """当图像路径数量和标签数量不符时，引发此异常"""
    def __init__(self, len_of_paths: int, len_of_labels: int):
        super().__init__(
            f"Number of paths ({len_of_paths}) is not compatible with number of labels ({len_of_labels})"
        )

class FileNotFoundInDatasetException(Exception):
    """当数据集中的图像文件不存在时，引发此异常"""
    def __init__(self, file_path: str):
        super().__init__(f"File not found in dataset: {file_path}")

class AnimalDataset(Dataset):
    """自定义数据集：持有图像路径、对应标签以及图像的统一尺寸"""
    def __init__(self, 
                 img_paths: List[str], 
                 img_labels: List[int], 
                 size_of_images: Tuple[int, int], 
                 transform=None,
                 verify_files: bool = True):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.size_of_images = size_of_images
        self.transform = transform
        
        # 验证输入
        if len(self.img_paths) != len(self.img_labels):
            raise InvalidDatasetException(len(self.img_paths), len(self.img_labels))
        
        # 验证文件存在性（可选）
        if verify_files:
            for path in self.img_paths:
                if not os.path.exists(path):
                    raise FileNotFoundInDatasetException(path)

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        try:
            # 使用上下文管理器优化资源使用
            with Image.open(self.img_paths[index]) as img:
                image = img.convert('RGB')
                image = image.resize(self.size_of_images)
                
                if self.transform is not None:
                    image = self.transform(image)
            return image, self.img_labels[index]
        except Exception as e:
            print(f"Error loading image at {self.img_paths[index]}: {str(e)}")
            # 返回一个伪造的替代图像和标签（防止训练中断）
            dummy_img = torch.zeros((3, *self.size_of_images))
            return dummy_img, self.img_labels[index]

# ============== 注意力机制(CBAM)的集成 ===============
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        mid_channels = max(8, in_channels // reduction_ratio)  # 确保通道数不会太小
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(),
            nn.Linear(mid_channels, in_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        b, c, h, w = x.size()
        # 平均池化 & 最大池化
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        scale = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pool = torch.cat([avg_pool, max_pool], dim=1)
        scale = torch.sigmoid(self.conv(pool))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

# ============== 网络结构优化（含残差+CBAM） ===============
class BasicBlock(nn.Module):
    """简易残差块示例：conv+BN+ReLU x2，加一个shortcut"""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.cbam = CBAM(out_channels)  # 在残差块结尾加入CBAM
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(identity)
        out = F.relu(out)
        # CBAM在残差叠加、ReLU之后再做
        out = self.cbam(out)
        return out

class ImprovedCNN(nn.Module):
    """结合残差结构和CBAM的改进CNN示例"""
    def __init__(self, num_classes: int = 3):
        super(ImprovedCNN, self).__init__()
        
        # 使用nn.Sequential简化网络定义
        self.features = nn.Sequential(
            # 初始层
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 残差层
            BasicBlock(in_channels=32, out_channels=64, stride=2),
            BasicBlock(in_channels=64, out_channels=128, stride=2),
            BasicBlock(in_channels=128, out_channels=256, stride=2),
            
            # 全局池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ============== 数据处理与加载函数 ===============
def load_image_paths(base_path: str, category: str) -> List[str]:
    """安全加载图像路径，附带错误检查"""
    path = os.path.join(base_path, category)
    if not os.path.exists(path):
        raise FileNotFoundError(f"路径不存在: {path}")
    
    # 获取所有图像文件路径
    image_paths = glob.glob(f"{path}/*")
    
    if not image_paths:
        raise ValueError(f"在{path}路径下未找到图像文件")
    
    return image_paths

def prepare_data(data_dir: Dict[str, str]) -> Tuple:
    """
    准备数据集，包括读取文件、处理类别不平衡、构建数据加载器
    
    Args:
        data_dir: 包含各类别数据路径的字典
        
    Returns:
        元组（train_loader, test_loader, label_map）
    """
    # 验证数据目录
    for dir_type, dir_path in data_dir.items():
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"{dir_type}数据目录不存在: {dir_path}")
    
    # 标签映射
    label_map = {0: "Cat", 1: "Dog", 2: "Wild"}
    
    # 读取文件并划分类别
    paths = []
    labels = []
    
    try:
        # 读取猫图像
        cat_paths = (load_image_paths(data_dir["train"], "cat") + 
                    load_image_paths(data_dir["val"], "cat"))
        for p in cat_paths:
            paths.append(p)
            labels.append(0)
            
        # 读取狗图像
        dog_paths = (load_image_paths(data_dir["train"], "dog") + 
                    load_image_paths(data_dir["val"], "dog"))
        for p in dog_paths:
            paths.append(p)
            labels.append(1)
            
        # 读取野生动物图像
        wild_paths = (load_image_paths(data_dir["train"], "wild") + 
                    load_image_paths(data_dir["val"], "wild"))
        for p in wild_paths:
            paths.append(p)
            labels.append(2)
    except Exception as e:
        print(f"加载图像路径时出错: {str(e)}")
        raise
    
    # 类别分布可视化
    data = pd.DataFrame({'classes': labels})
    for class_label, class_name in label_map.items():
        count_ = data[data['classes'] == class_label].shape[0]
        print(f"类别 {class_name}: {count_} 张照片")

    # 可视化原始分布
    plt.figure(figsize=(5, 4))
    sns.countplot(x=data['classes'], color='#2596be')
    plt.title('原始类别分布')
    plt.show()
    
    # 处理类别不平衡
    labels = np.array(labels)
    paths = np.array(paths)
    counter = Counter(labels)
    print("原始样本数量:", counter)
    
    # 欠采样，使三类平衡
    cat_indices = np.where(labels == 0)[0]
    dog_indices = np.where(labels == 1)[0]
    wild_indices = np.where(labels == 2)[0]
    min_samples = min(len(cat_indices), len(dog_indices), len(wild_indices))
    
    undersampled_cat = resample(cat_indices, replace=False, 
                               n_samples=min_samples, 
                               random_state=CONFIG["random_seed"])
    undersampled_dog = resample(dog_indices, replace=False, 
                               n_samples=min_samples, 
                               random_state=CONFIG["random_seed"])
    undersampled_wild = resample(wild_indices, replace=False, 
                                n_samples=min_samples, 
                                random_state=CONFIG["random_seed"])
    
    undersampled_indices = np.concatenate((undersampled_cat, undersampled_dog, undersampled_wild))
    undersampled_paths = paths[undersampled_indices]
    undersampled_labels = labels[undersampled_indices]
    counter_undersampled = Counter(undersampled_labels)
    print("欠采样后的样本数量:", counter_undersampled)
    
    # 欠采样后的分布可视化
    plt.figure(figsize=(5, 4))
    sns.countplot(x=undersampled_labels, color='#2596be')
    plt.title('欠采样后的类别分布')
    plt.show()
    
    # ============== 数据增强策略应用 ===============
    import torchvision.transforms as T
    
    # 训练集数据增强
    train_transform = T.Compose([
        T.RandomResizedCrop(CONFIG["img_size"], scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.ToTensor()
    ])
    
    # 测试集只做基本处理
    test_transform = T.Compose([
        T.Resize(CONFIG["img_size"]),
        T.ToTensor()
    ])
    
    # 数据集划分
    train_indices, test_indices = train_test_split(
        list(range(len(undersampled_paths))), 
        test_size=0.2, 
        random_state=CONFIG["random_seed"],
        stratify=undersampled_labels  # 确保训练集和测试集中类别分布一致
    )
    
    # 创建训练集和测试集
    train_paths = undersampled_paths[train_indices]
    train_labels = undersampled_labels[train_indices]
    test_paths = undersampled_paths[test_indices]
    test_labels = undersampled_labels[test_indices]
    
    # 构建独立的训练和测试Dataset
    train_dataset = AnimalDataset(
        train_paths, 
        train_labels, 
        CONFIG["img_size"], 
        transform=train_transform,
        verify_files=False  # 性能考虑，在大型数据集上可以跳过完整的文件验证
    )
    
    test_dataset = AnimalDataset(
        test_paths, 
        test_labels, 
        CONFIG["img_size"], 
        transform=test_transform,
        verify_files=False
    )
    
    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=CONFIG["batch_size"]
    )
    
    return train_loader, test_loader, label_map

# ============== 训练与评估函数 ===============
def validate_model(model: nn.Module, data_loader: DataLoader, criterion=None) -> Tuple[float, float]:
    """
    在给定数据加载器上验证模型性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        criterion: 损失函数（如果为None则只计算准确率）
        
    Returns:
        元组（val_loss, val_accuracy）如果没有提供criterion，则val_loss为0
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            
            if criterion is not None:
                loss = criterion(outputs, labels.long())
                val_loss += loss.item() * images.size(0)
            
            _, preds = torch.max(outputs, dim=1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)
    
    val_accuracy = 100.0 * correct / total
    
    if criterion is not None:
        val_loss = val_loss / total
        return val_loss, val_accuracy
    else:
        return 0.0, val_accuracy

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                test_loader: DataLoader, 
                epochs: int = 10) -> Tuple:
    """
    训练模型函数
    
    Args:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        
    Returns:
        训练完成的模型和训练历史
    """
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=CONFIG["learning_rate"])
    
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    best_acc = 0.0
    patience = CONFIG["patience"]
    stopping_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm添加进度条
        with tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}") as pbar:
            for images, labels in pbar:
                try:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels.long())
                    loss.backward()
                    optimizer.step()
        
                    # 更新统计信息
                    running_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, dim=1)
                    correct += torch.sum(preds == labels).item()
                    total += labels.size(0)
                    
                    # 实时更新进度条信息
                    pbar.set_postfix(loss=loss.item(), acc=f"{100.0*correct/total:.2f}%")
                except Exception as e:
                    print(f"训练批次处理出错: {str(e)}")
                    continue
        
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # 验证阶段 - 使用分离出的验证函数
        val_loss, val_acc = validate_model(model, test_loader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        # 余弦退火调度
        scheduler.step()
    
        print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
        # Early Stopping 判断和模型保存
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_state = model.state_dict()
            # 保存最佳模型
            torch.save(best_model_state, "checkpoints/best_model.pth")
            print(f"Checkpoint saved: val_acc={val_acc:.2f}%")
            stopping_counter = 0
        else:
            stopping_counter += 1
            if stopping_counter >= patience:
                print("Validation Acc not improved for consecutive epochs, early stopping!")
                break
    
    # 恢复最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    return model, train_losses, train_accuracies, val_losses, val_accuracies

def evaluate_model(model: nn.Module, 
                   test_loader: DataLoader, 
                   label_map: Dict[int, str]) -> Tuple:
    """
    评估模型性能
    
    Args:
        model: 待评估的模型
        test_loader: 测试数据加载器
        label_map: 标签映射字典
        
    Returns:
        性能指标和混淆矩阵
    """
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in test_loader:
            try:
                images = images.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
            except Exception as e:
                print(f"评估批次处理出错: {str(e)}")
                continue
    
    # 计算各指标
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    # 打印指标
    print(f"测试集准确率: {acc:.4f}")
    print(f"精确率(宏平均): {prec:.4f}")
    print(f"召回率(宏平均): {rec:.4f}")
    print(f"F1分数(宏平均): {f1:.4f}")
    
    # 混淆矩阵可视化
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[label_map[i] for i in range(len(label_map))],
                yticklabels=[label_map[i] for i in range(len(label_map))])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    
    return acc, prec, rec, f1, cm

def visualize_training_history(train_losses: List[float], 
                              train_accuracies: List[float],
                              val_losses: List[float] = None,
                              val_accuracies: List[float] = None) -> None:
    """可视化训练过程损失和准确率"""
    plt.figure(figsize=(10,8))
    
    plt.subplot(2,1,1)
    plt.plot(range(1, len(train_losses)+1), train_losses, '-o', label='Train')
    if val_losses:
        plt.plot(range(1, len(val_losses)+1), val_losses, '-o', label='Validation')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, '-o', label='Train')
    if val_accuracies:
        plt.plot(range(1, len(val_accuracies)+1), val_accuracies, '-o', label='Validation')
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ============== 主函数 ===============
def main():
    """主函数，包含完整的训练和评估流程"""
    print("开始：动物分类改进CNN模型训练与评估流程")
    
    try:
        # 数据路径设置
        data_dir = {
            "train": "archive/afhq/train",
            "val": "archive/afhq/val"
        }
        
        # 检查路径是否存在
        for key, path in data_dir.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据路径不存在: {path}")
        
        # 准备数据
        train_loader, test_loader, label_map = prepare_data(data_dir)
        
        # 创建模型
        model = ImprovedCNN(num_classes=CONFIG["num_classes"]).to(DEVICE)
        print(f"模型创建完成，使用设备: {DEVICE}")
        
        # 训练模型
        print("开始训练模型...")
        model, train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, 
            train_loader, 
            test_loader, 
            epochs=CONFIG["epochs"]
        )
        
        # 评估模型
        print("开始评估模型...")
        acc, prec, rec, f1, cm = evaluate_model(model, test_loader, label_map)
        
        # 可视化训练历史
        visualize_training_history(train_losses, train_accuracies, val_losses, val_accuracies)
        
        print(f"模型训练与评估完成。最终测试集准确率: {acc:.4f}")
        
    except Exception as e:
        print(f"运行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

# 运行主函数
if __name__ == "__main__":
    main()
