"""
PyTorch实现的CNN模型用于MNIST手写数字识别

特性：
1. 使用 nn.Module 封装网络结构，便于复用与保存模型；
2. 使用 torchvision.datasets.ImageFolder 从本地文件夹（mnist_train 和 mnist_test）读取数据集；
3. 将训练、测试循环封装为函数，代码结构更清晰；
4. 使用命令行参数 argparse 方便调整超参数（学习率、batch_size、epoch 数等）；
5. 支持 GPU / CPU 自动选择，并在 GPU 上可选使用自动混合精度（AMP）加速。
"""

import argparse
import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
    MODEL_SAVE_PATH = "../../models/cnn_pytorch.pt"
    
    # 设备设置
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    USE_AMP = True  # 是否使用自动混合精度（仅在GPU上有效）


class CNN(nn.Module):
    """
    CNN模型：用于MNIST手写数字识别
    
    结构说明：
        第一层卷积：5x5卷积核，1个输入通道，32个输出通道
        第一层池化：2x2最大池化
        第二层卷积：5x5卷积核，32个输入通道，64个输出通道
        第二层池化：2x2最大池化
        全连接层1：7*7*64 -> 1024 + ReLU + Dropout
        全连接层2：1024 -> 10（输出层）
    
    参数说明：
        dropout: Dropout概率，用于缓解过拟合
    """
    
    def __init__(self, dropout: float = Config.DROPOUT):
        """
        初始化CNN模型
        
        Args:
            dropout: Dropout概率，默认0.2
        """
        super().__init__()
        # 第一层卷积：5x5卷积核，1个输入通道，32个输出通道，padding='SAME'等价于padding=2
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        # 最大池化：2x2窗口，步长为2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二层卷积：5x5卷积核，32个输入通道，64个输出通道，padding='SAME'等价于padding=2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        # 最大池化：2x2窗口，步长为2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层1：7*7*64 -> 1024
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.relu3 = nn.ReLU(inplace=True)
        # Dropout层：减少过拟合
        self.dropout = nn.Dropout(p=dropout)
        
        # 全连接层2：1024 -> 10
        self.fc2 = nn.Linear(1024, Config.NUM_CLASSES)
        
        # 自定义参数初始化
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        对网络中的卷积层和线性层使用合适的初始化方法，
        使得训练更加稳定。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
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
        # 第一层卷积+池化
        x = self.conv1(x)  # [batch, 1, 28, 28] -> [batch, 32, 28, 28]
        x = self.relu1(x)
        x = self.pool1(x)  # [batch, 32, 28, 28] -> [batch, 32, 14, 14]
        
        # 第二层卷积+池化
        x = self.conv2(x)  # [batch, 32, 14, 14] -> [batch, 64, 14, 14]
        x = self.relu2(x)
        x = self.pool2(x)  # [batch, 64, 14, 14] -> [batch, 64, 7, 7]
        
        # 展平
        x = x.view(-1, 7 * 7 * 64)  # [batch, 64, 7, 7] -> [batch, 3136]
        
        # 全连接层1
        x = self.fc1(x)  # [batch, 3136] -> [batch, 1024]
        x = self.relu3(x)
        x = self.dropout(x)  # Dropout训练时使用
        
        # 全连接层2（输出层）
        x = self.fc2(x)  # [batch, 1024] -> [batch, 10]
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


def load_data(
    data_dir: str = Config.DATA_DIR,
    batch_size: int = Config.BATCH_SIZE,
    num_workers: int = Config.NUM_WORKERS,
    pin_memory: bool = Config.PIN_MEMORY
) -> Tuple[DataLoader, DataLoader]:
    """
    构建训练集和测试集的 DataLoader
    
    Args:
        data_dir: 数据集根目录（包含 mnist_train 和 mnist_test 文件夹）
        batch_size: 每个 mini-batch 中的样本数量
        num_workers: DataLoader 使用多少个子进程来加载数据
        pin_memory: 是否使用pin_memory（GPU训练时建议开启）
        
    Returns:
        (train_loader, test_loader): 训练集和测试集 DataLoader
        
    Raises:
        FileNotFoundError: 如果数据目录不存在
    """
    data_path = Path(data_dir)
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # 测试集 DataLoader：不打乱，评估时顺序无关紧要
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
    # 在类别维度上取最大值下标，即为预测类别
    preds = logits.argmax(dim=1)
    # 与真实标签比较，相等为 1，不等为 0，再取平均即可得到准确率
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
        scaler: GradScaler 对象，用于在 GPU 上做自动混合精度训练，
                若在 CPU 上训练或禁用 AMP，则为 None
                
    Returns:
        (epoch_loss, epoch_acc): 当前 epoch 在训练集上的平均损失和平均准确率
    """
    # 将模型置于训练模式（启用 Dropout / BatchNorm 等）
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    
    for images, labels in dataloader:
        # 把数据移动到指定设备（CPU 或 GPU）
        images, labels = images.to(device), labels.to(device)
        
        # 每个 mini-batch 开始前先清空梯度
        optimizer.zero_grad()
        
        # 在 GPU + AMP 场景下，使用 autocast 在混合精度下进行前向与反向；
        # 在 CPU 或关闭 AMP 时，scaler 为 None，autocast 不会启用。
        use_amp = scaler is not None
        with torch.amp.autocast('cuda', enabled=use_amp):
            # 前向传播，得到 logits（未经过 softmax）
            logits = model(images)
            # 计算交叉熵损失，criterion 内部会自动做 log-softmax
            loss = criterion(logits, labels)
        
        if not use_amp:
            # 普通 FP32 训练流程
            loss.backward()
            optimizer.step()
        else:
            # AMP 训练流程：先缩放 loss 再反向传播，最后更新缩放因子
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 累加当前 batch 的损失和准确率（乘以 batch_size 是为了后面做总平均）
        running_loss += loss.item() * images.size(0)
        running_acc += calculate_accuracy(logits.detach(), labels) * images.size(0)
    
    # 数据集总样本数
    size = len(dataloader.dataset)
    # 返回平均损失和平均准确率
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
    
    使用 @torch.no_grad() 装饰器可以关闭梯度计算，
    减少内存占用和加快推理速度。
    
    Args:
        model: 要评估的神经网络模型
        dataloader: 测试集 DataLoader
        criterion: 损失函数
        device: 评估设备（CPU 或 GPU）
        
    Returns:
        (avg_loss, avg_acc): 平均损失和平均准确率
    """
    # 将模型置于评估模式（关闭 Dropout / 固定 BatchNorm 统计量）
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


def create_model(
    dropout: float = Config.DROPOUT,
    device: torch.device = Config.DEVICE
) -> nn.Module:
    """
    创建并初始化模型
    
    Args:
        dropout: Dropout概率
        device: 计算设备
        
    Returns:
        初始化后的模型
    """
    model = CNN(dropout=dropout).to(device)
    return model


def create_optimizer(
    model: nn.Module,
    learning_rate: float = Config.LEARNING_RATE
) -> optim.Optimizer:
    """
    创建优化器
    
    Args:
        model: 模型
        learning_rate: 学习率
        
    Returns:
        优化器（Adam）
    """
    return optim.Adam(model.parameters(), lr=learning_rate)


def create_scaler(use_amp: bool, device: torch.device) -> Optional[torch.amp.GradScaler]:
    """
    创建GradScaler用于自动混合精度训练
    
    Args:
        use_amp: 是否使用AMP
        device: 计算设备
        
    Returns:
        GradScaler对象，如果不需要AMP则返回None
    """
    if use_amp and device.type == 'cuda':
        return torch.amp.GradScaler('cuda', enabled=True)
    return None


def save_model(model: nn.Module, save_path: str = Config.MODEL_SAVE_PATH) -> None:
    """
    保存模型权重
    
    Args:
        model: 要保存的模型
        save_path: 保存路径
    """
    torch.save(model.state_dict(), save_path)
    print(f"模型权重已保存到: {save_path}")


def parse_args() -> argparse.Namespace:
    """
    使用 argparse 从命令行解析超参数，方便在终端中灵活调整
    
    常用参数示例：
        --batch-size 100
        --epochs 10
        --lr 0.0001
        --dropout 0.2
        --num-workers 0   （在 Windows 或 CPU 机器上建议设为 0）
        --no-amp          （在只用 CPU 训练时建议加上，关闭 AMP）
        
    Returns:
        解析后的命令行参数
    """
    parser = argparse.ArgumentParser(description="PyTorch版本MNIST卷积神经网络")
    
    # 数据集目录
    parser.add_argument(
        "--data-dir",
        type=str,
        default=Config.DATA_DIR,
        help="数据集根目录（包含 mnist_train 和 mnist_test 文件夹）"
    )
    
    # 训练超参数
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE,
                       help="每个 batch 中的样本数")
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS,
                       help="训练轮数")
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE,
                       help="学习率")
    parser.add_argument("--dropout", type=float, default=Config.DROPOUT,
                       help="Dropout 概率")
    
    # 数据加载设置
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS,
                       help="DataLoader 的工作进程数（Windows建议设为0）")
    
    # 训练设置
    parser.add_argument("--seed", type=int, default=Config.SEED,
                       help="随机数种子，保证可复现性")
    parser.add_argument("--model-save-path", type=str, default=Config.MODEL_SAVE_PATH,
                       help="模型保存路径")
    
    # AMP设置
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="禁用自动混合精度（默认在CUDA上启用）",
    )
    
    return parser.parse_args()


def setup_environment(seed: int = Config.SEED) -> None:
    """
    设置训练环境（随机种子等）
    
    Args:
        seed: 随机数种子
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_config(args: argparse.Namespace, device: torch.device) -> None:
    """
    打印配置信息
    
    Args:
        args: 命令行参数
        device: 计算设备
    """
    print("=" * 60)
    print("训练配置信息")
    print("=" * 60)
    print(f"使用设备: {device}")
    print(f"数据集目录: {args.data_dir}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print(f"学习率: {args.lr}")
    print(f"Dropout: {args.dropout}")
    print(f"随机种子: {args.seed}")
    print(f"AMP: {'启用' if not args.no_amp and device.type == 'cuda' else '禁用'}")
    print("=" * 60)


def main():
    """
    程序入口：完成以下步骤
    1. 解析命令行参数；
    2. 设置随机种子与运行设备（CPU / GPU）；
    3. 构建数据加载器、模型、损失函数和优化器；
    4. 进行多轮训练，并在每轮后进行测试集评估；
    5. 保存在测试集上表现最好的模型权重。
    """
    args = parse_args()
    
    # 设置训练环境
    setup_environment(args.seed)
    
    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 打印配置信息
    print_config(args, device)
    
    # 构建训练集和测试集的 DataLoader
    print("\n正在加载数据...")
    try:
        train_loader, test_loader = load_data(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"训练样本数: {len(train_loader.dataset)}")
        print(f"测试样本数: {len(test_loader.dataset)}")
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 创建模型
    print("\n正在创建模型...")
    model = create_model(dropout=args.dropout, device=device)
    
    # 创建损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, args.lr)
    
    # 创建GradScaler（用于AMP）
    use_amp = not args.no_amp and device.type == 'cuda'
    scaler = create_scaler(use_amp, device)
    
    # 用于记录在测试集上取得的最佳准确率
    best_acc = 0.0
    
    # 训练循环
    print("\n开始训练...")
    print("-" * 60)
    
    for epoch in range(1, args.epochs + 1):
        # 在训练集上训练一个 epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # 在测试集上评估当前模型
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # 若本轮测试集准确率更高，则更新 best_acc 并保存当前模型参数
        if val_acc > best_acc:
            best_acc = val_acc
            save_model(model, args.model_save_path)
        
        # 打印当前 epoch 的训练损失 / 准确率以及测试损失 / 准确率
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
            f"Test Loss {val_loss:.4f} Acc {val_acc:.4f}"
        )
    
    print("-" * 60)
    print(f"最佳测试准确率: {best_acc:.4f}")
    print(f"模型权重已保存到: {args.model_save_path}")


if __name__ == "__main__":
    # 只有当本脚本作为主程序运行时，才会执行 main()。
    # 若在其他 Python 文件中 import 本脚本，则不会自动开始训练。
    main()
