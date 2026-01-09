"""
PyTorch实现的MLP模型用于MNIST衣物识别

直接运行即可训练MLP模型：
    python train_mlp.py
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class Config:
    """配置类：存储所有默认超参数和路径配置"""
    # 数据路径
    DATA_DIR = "../../data"
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    
    # 训练超参数
    BATCH_SIZE = 100
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    DROPOUT = 0.2
    
    # 模型超参数
    IMAGE_SIZE = 28
    NUM_CLASSES = 10
    
    # 数据加载设置
    NUM_WORKERS = 0  # Windows上建议设为0
    PIN_MEMORY = True
    
    # 数据预处理参数（MNIST标准化参数）
    NORMALIZE_MEAN = (0.1307,)
    NORMALIZE_STD = (0.3081,)
    
    # 训练设置
    SEED = 42
    MODEL_SAVE_PATH = "../../models/mlp_pytorch.pt"
    
    # 设备设置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True  # 是否使用自动混合精度（仅在GPU上有效）


class MLP(nn.Module):
    """
    MLP模型：用于Fashion-MNIST数据集
    
    结构说明：
        全连接层1：28*28 -> 512 -> 128 -> 64 -> 10（输出层）
    
    参数说明：
        dropout: Dropout概率，用于缓解过拟合
    """
    
    def __init__(self, dropout: float = Config.DROPOUT):
        """
        初始化MLP模型
        
        Args:
            dropout: Dropout概率，默认0.2
        """
        super().__init__()
        # 全连接层1：28*28 -> 512
        self.fc1 = nn.Linear(28 * 28, 512)
        self.relu1 = nn.ReLU(inplace=True)
        # Dropout层：减少过拟合
        self.dropout1 = nn.Dropout(p=dropout)
        
        # 全连接层2：512 -> 128
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU(inplace=True)
        # Dropout层：减少过拟合
        self.dropout2 = nn.Dropout(p=dropout)
        
        # 全连接层3：128 -> 64
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU(inplace=True)
        # Dropout层：减少过拟合
        self.dropout3 = nn.Dropout(p=dropout)
        
        # 全连接层4：64 -> 10（输出层）
        self.fc4 = nn.Linear(64, Config.NUM_CLASSES)
        
        # 自定义参数初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        对网络中的线性层使用合适的初始化方法，
        使得训练更加稳定。
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        
        Args:
            x: 输入张量，形状为 (batch_size, 1, 28, 28)
            
        Returns:
            输出logits，形状为 (batch_size, 10)
        """
        # 展平输入
        x = x.view(-1, 28 * 28)  # [batch, 1, 28, 28] -> [batch, 784]
        
        # 全连接层1
        x = self.fc1(x)  # [batch, 784] -> [batch, 512]
        x = self.relu1(x)
        x = self.dropout1(x)  # Dropout训练时使用
        
        # 全连接层2
        x = self.fc2(x)  # [batch, 512] -> [batch, 256]
        x = self.relu2(x)
        x = self.dropout2(x)  # Dropout训练时使用
        
        # 全连接层3
        x = self.fc3(x)  # [batch, 256] -> [batch, 128]
        x = self.relu3(x)
        x = self.dropout3(x)  # Dropout训练时使用
        
        # 全连接层4（输出层）
        x = self.fc4(x)  # [batch, 128] -> [batch, 10]
        return x


def get_data_transforms() -> transforms.Compose:
    """
    获取数据预处理变换
    
    Returns:
        数据变换组合
    """
    return transforms.Compose([
        transforms.Grayscale(),  # 确保图片是灰度图
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),  # 调整大小为28x28
        transforms.ToTensor(),  # 转换为张量并归一化到[0,1]
        transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD),  # 标准化
    ])


def load_data() -> Tuple[DataLoader, DataLoader]:
    """
    构建训练集和测试集的 DataLoader
        
    Returns:
        (train_loader, test_loader): 训练集和测试集 DataLoader
        
    Raises:
        FileNotFoundError: 如果数据目录不存在
    """
    data_path = Path(Config.DATA_DIR)
    train_dir = data_path / Config.TRAIN_DIR
    test_dir = data_path / Config.TEST_DIR
    
    # 检查数据目录是否存在
    if not train_dir.exists():
        raise FileNotFoundError(f'训练数据目录不存在: {train_dir}')
    if not test_dir.exists():
        raise FileNotFoundError(f'测试数据目录不存在: {test_dir}')
    
    transform = get_data_transforms()
    
    # 构建训练集
    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=transform,
    )
    
    # 构建测试集
    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=transform,
    )
    
    # 训练集 DataLoader：shuffle=True 打乱数据，有利于模型训练
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    
    # 测试集 DataLoader：不打乱，评估时顺序无关紧要
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    
    return train_loader, test_loader


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    计算分类准确率
    
    Args:
        logits: 模型输出的预测分数，形状为 (batch_size, num_classes)
        targets: 真实标签，形状为 (batch_size,) 的长整型张量
        
    Returns:
        当前 batch 的平均准确率（float 标量）
    """
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None
) -> Tuple[float, float]:
    """
    训练模型一个完整的 epoch
    
    Args:
        model: 要训练的神经网络模型
        dataloader: 训练集 DataLoader
        criterion: 损失函数（这里使用交叉熵）
        optimizer: 优化器
        device: 训练设备（CPU 或 GPU）
        scaler: GradScaler 对象，用于在 GPU 上做自动混合精度训练
                
    Returns:
        (epoch_loss, epoch_acc): 当前 epoch 在训练集上的平均损失和平均准确率
    """
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        if not use_amp:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        running_loss += loss.item() * images.size(0)
        running_acc += calculate_accuracy(logits.detach(), labels) * images.size(0)
    
    size = len(dataloader.dataset)
    return running_loss / size, running_acc / size


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    在验证集或测试集上评估模型性能
    
    Args:
        model: 要评估的神经网络模型
        dataloader: 测试集 DataLoader
        criterion: 损失函数
        device: 评估设备（CPU 或 GPU）
        
    Returns:
        (avg_loss, avg_acc): 平均损失和平均准确率
    """
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        running_acc += calculate_accuracy(logits, labels) * images.size(0)
    
    size = len(dataloader.dataset)
    return running_loss / size, running_acc / size


def save_model(model: nn.Module, save_path: str = Config.MODEL_SAVE_PATH) -> None:
    """
    保存模型权重
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存到: {save_path}")


def main():
    """
    程序入口：完成以下步骤
    1. 设置随机种子与运行设备（CPU / GPU）；
    2. 构建数据加载器、模型、损失函数和优化器；
    3. 进行多轮训练，并在每轮后进行测试集评估；
    4. 保存在测试集上表现最好的模型权重。
    """
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
        torch.cuda.manual_seed_all(Config.SEED)
    
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印配置信息
    print("=" * 60)
    print("MLP 训练配置信息")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"数据集目录: {Config.DATA_DIR}")
    print(f"批次大小: {Config.BATCH_SIZE}")
    print(f"训练轮数: {Config.EPOCHS}")
    print(f"学习率: {Config.LEARNING_RATE}")
    print(f"Dropout: {Config.DROPOUT}")
    print(f"AMP: {'启用' if Config.USE_AMP and device.type == 'cuda' else '禁用'}")
    print("=" * 60)
    
    # 构建训练集和测试集的 DataLoader
    print("\n正在加载数据...")
    try:
        train_loader, test_loader = load_data()
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"测试样本数: {len(test_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 创建模型
    print("\n正在创建MLP模型...")
    model = MLP(dropout=Config.DROPOUT).to(device)
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # 创建GradScaler（用于AMP）
    use_amp = Config.USE_AMP and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=True) if use_amp else None
    
    # 用于记录在测试集上取得的最佳准确率
    best_acc = 0.0
    
    # 训练循环
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, Config.MODEL_SAVE_PATH)
        
        print(
            f"Epoch {epoch:02d}/{Config.EPOCHS} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Test Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )
    
    print("-" * 60)
    print(f"最佳测试准确率: {best_acc:.4f}")
    print(f"模型权重已保存到: {Config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
