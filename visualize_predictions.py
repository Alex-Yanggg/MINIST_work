"""
MLP模型预测可视化脚本

从测试集中随机选择9张图片，使用MLP模型进行预测，并以9宫格形式可视化结果。
"""

import os
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 导入模型定义
import sys
sys.path.append(str(Path(__file__).parent / "scripts" / "training"))
from train_mlp import MLP, Config


# 可视化配置常量
class VisualizationConfig:
    """可视化配置类"""
    # 布局设置
    GRID_SIZE = (2, 2)  # 2x2网格
    FIGURE_SIZE = (10, 10)  # 图像尺寸
    DPI = 200  # 分辨率
    
    # 颜色配置
    BACKGROUND_COLOR = '#ffffff'  # 白色背景
    CORRECT_COLOR = '#2ecc71'  # 正确预测颜色（绿色）
    ERROR_COLOR = '#e74c3c'  # 错误预测颜色（红色）
    TEXT_COLOR = '#2c3e50'  # 主要文字颜色
    SUBTEXT_COLOR = '#7f8c8d'  # 次要文字颜色
    
    # 字体设置
    TITLE_FONT_SIZE = 22
    SUBTITLE_FONT_SIZE = 16
    CONFIDENCE_FONT_SIZE = 13
    
    # 边框设置
    BORDER_WIDTH = 2
    BORDER_PADDING = 0.02


def load_mlp_model(model_path: str = "models/mlp_pytorch.pt", 
                   device: Optional[torch.device] = None) -> nn.Module:
    """
    加载训练好的MLP模型权重
    
    该函数会创建MLP模型实例，并从指定路径加载训练好的权重参数。
    如果模型文件不存在，将抛出FileNotFoundError异常。
    
    Args:
        model_path: 模型权重文件的路径，默认为 "models/mlp_pytorch.pt"
        device: 计算设备（CPU或GPU），如果为None则自动检测并使用可用设备
        
    Returns:
        加载好权重的MLP模型，已设置为评估模式（eval mode）
        
    Raises:
        FileNotFoundError: 当模型文件不存在时抛出
    """
    # 自动检测设备
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例并移动到指定设备
    model = MLP(dropout=Config.DROPOUT).to(device)
    
    # 检查并加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"✓ 成功加载模型权重: {model_path}")
    
    # 设置为评估模式（关闭dropout等训练特性）
    model.eval()
    return model


def get_data_transforms() -> transforms.Compose:
    """
    获取与训练时相同的数据预处理变换
    
    Returns:
        数据变换组合
    """
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((Config.IMAGE_SIZE, Config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(Config.NORMALIZE_MEAN, Config.NORMALIZE_STD),
    ])


def load_test_images(test_dir: str = "data/test", num_samples: int = 4) -> List[Tuple[str, int, Image.Image]]:
    """
    从测试集中随机加载图片
    
    Args:
        test_dir: 测试集目录路径
        num_samples: 需要加载的图片数量
        
    Returns:
        [(图片路径, 真实标签, PIL图片对象), ...]
    """
    test_path = Path(test_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
    
    # 先收集所有图片文件路径和标签（不加载图片）
    image_paths_with_labels = []
    
    # 遍历0-9文件夹
    for label in range(10):
        label_dir = test_path / str(label)
        if label_dir.exists():
            # 获取该类别下的所有图片文件路径
            image_files = list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))
            for img_file in image_files:
                image_paths_with_labels.append((img_file, label))
    
    if len(image_paths_with_labels) < num_samples:
        print(f"警告: 测试集中只有 {len(image_paths_with_labels)} 张图片，少于请求的 {num_samples} 张")
        num_samples = len(image_paths_with_labels)
    
    # 先随机选择文件路径
    selected_paths = random.sample(image_paths_with_labels, num_samples)
    
    # 只加载被选中的图片
    images_with_labels = []
    for img_file, label in selected_paths:
        try:
            img = Image.open(img_file)
            images_with_labels.append((str(img_file), label, img))
        except Exception as e:
            print(f"加载图片失败 {img_file}: {e}")
    
    return images_with_labels


def predict_images(model: nn.Module, 
                   images: List[Tuple[str, int, Image.Image]], 
                   transform: transforms.Compose, 
                   device: torch.device) -> List[Tuple[int, int, float]]:
    """
    对图片进行批量预测
    
    对输入的图片列表进行预处理，然后使用MLP模型进行预测，
    返回预测标签、真实标签和对应的置信度。
    
    Args:
        model: 已加载的MLP模型（应处于eval模式）
        images: 图片列表，格式为 [(图片路径, 真实标签, PIL图片对象), ...]
        transform: 数据预处理变换（应与训练时保持一致）
        device: 计算设备（CPU或GPU）
        
    Returns:
        预测结果列表，格式为 [(真实标签, 预测标签, 置信度), ...]
        置信度为预测类别的softmax概率值
    """
    results = []
    
    # 禁用梯度计算以节省内存和加速推理
    with torch.no_grad():
        for img_path, true_label, pil_img in images:
            # 应用预处理变换：转换为张量、标准化等
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # 模型前向传播，获取logits
            output = model(img_tensor)
            
            # 计算softmax概率分布
            probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # 获取最大概率值和对应的类别索引
            confidence, predicted = torch.max(probabilities, 1)
            
            # 转换为Python标量
            pred_label = predicted.item()
            conf = confidence.item()
            
            results.append((true_label, pred_label, conf))
    
    return results


def visualize_predictions(images: List[Tuple[str, int, Image.Image]], 
                         results: List[Tuple[int, int, float]],
                         save_path: str = "predictions_visualization.png") -> None:
    """
    以简洁大气的2x2网格形式可视化预测结果
    
    采用极简设计风格，白色背景，清晰的层次结构和配色方案。
    
    Args:
        images: 图片列表，格式为 [(图片路径, 真实标签, PIL图片对象), ...]
        results: 预测结果列表，格式为 [(真实标签, 预测标签, 置信度), ...]
        save_path: 保存路径，默认为 "predictions_visualization.png"
    """
    config = VisualizationConfig
    
    # 创建图形和子图
    fig, axes = plt.subplots(
        *config.GRID_SIZE, 
        figsize=config.FIGURE_SIZE, 
        facecolor=config.BACKGROUND_COLOR
    )
    axes = axes.flatten() if config.GRID_SIZE[0] * config.GRID_SIZE[1] > 1 else [axes]
    
    # 设置主标题
    fig.suptitle(
        'MLP Model Predictions on Fashion-MNIST', 
        fontsize=config.TITLE_FONT_SIZE, 
        fontweight='bold',
        color=config.TEXT_COLOR,
        y=0.98
    )
    
    # 遍历每张图片和对应的预测结果
    for idx, ((img_path, true_label, pil_img), (true, pred, conf)) in enumerate(zip(images, results)):
        ax = axes[idx]
        
        # 判断预测是否正确
        is_correct = (true == pred)
        
        # 根据预测结果选择颜色
        border_color = config.CORRECT_COLOR if is_correct else config.ERROR_COLOR
        status_symbol = '✓' if is_correct else '✗'
        
        # 设置子图背景为白色
        ax.set_facecolor(config.BACKGROUND_COLOR)
        
        # 显示图片
        ax.imshow(pil_img, cmap='gray', aspect='equal', interpolation='bilinear')
        ax.axis('off')
        
        # 添加简洁的边框（仅在底部和右侧）
        for spine_name, spine in ax.spines.items():
            if spine_name in ['bottom', 'right']:
                spine.set_color(border_color)
                spine.set_linewidth(config.BORDER_WIDTH)
                spine.set_visible(True)
            else:
                spine.set_visible(False)
        
        # 添加预测信息（简洁的文本标签）
        # 上方显示：Pred | Target
        info_text = f'Pred: {pred}  |  Target: {true}  {status_symbol}'
        ax.text(
            0.5, 1.05, 
            info_text, 
            transform=ax.transAxes,
            fontsize=config.SUBTITLE_FONT_SIZE,
            fontweight='600',
            color=config.TEXT_COLOR,
            ha='center',
            va='bottom'
        )
        
        # 下方显示：置信度
        conf_text = f'{conf:.1%}'
        ax.text(
            0.5, -0.08,
            conf_text,
            transform=ax.transAxes,
            fontsize=config.CONFIDENCE_FONT_SIZE,
            color=config.SUBTEXT_COLOR,
            ha='center',
            va='top'
        )
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    plt.savefig(
        save_path, 
        dpi=config.DPI, 
        bbox_inches='tight', 
        facecolor=config.BACKGROUND_COLOR,
        edgecolor='none'
    )
    plt.close()
    print(f"可视化结果已保存到: {save_path}")


def main() -> None:
    """
    主函数：执行完整的预测和可视化流程
    
    流程包括：
    1. 设置随机种子（基于系统时间）
    2. 初始化计算设备
    3. 加载MLP模型权重
    4. 从测试集中随机选择图片
    5. 进行预测
    6. 统计并打印准确率
    7. 生成可视化结果
    """
    print("=" * 60)
    print("MLP Fashion-MNIST 预测可视化")
    print("=" * 60)
    
    # 使用系统时间作为随机种子，确保每次运行结果不同
    seed = int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"随机种子: {seed}")
    
    # 自动检测并使用可用的计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"计算设备: {device}")
    
    # 步骤1：加载预训练模型
    print("\n[步骤 1/4] 加载MLP模型...")
    model_path = "models/mlp_pytorch.pt"
    model = load_mlp_model(model_path, device)
    
    # 步骤2：准备数据预处理管道
    print("\n[步骤 2/4] 准备数据预处理...")
    transform = get_data_transforms()
    
    # 步骤3：从测试集中随机采样图片
    print("\n[步骤 3/4] 从测试集加载图片...")
    num_samples = VisualizationConfig.GRID_SIZE[0] * VisualizationConfig.GRID_SIZE[1]
    test_images = load_test_images("data/test", num_samples=num_samples)
    print(f"✓ 成功加载 {len(test_images)} 张图片")
    
    # 步骤4：进行预测
    print("\n[步骤 4/4] 执行预测...")
    predictions = predict_images(model, test_images, transform, device)
    
    # 统计并打印预测准确率
    correct = sum(1 for true, pred, _ in predictions if true == pred)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n预测统计:")
    print(f"  正确: {correct}/{total}")
    print(f"  准确率: {accuracy:.2%}")
    
    # 生成并保存可视化结果
    print("\n生成可视化结果...")
    visualize_predictions(test_images, predictions, save_path="predictions_visualization.png")
    
    print("\n完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

