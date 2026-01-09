"""
MLP模型预测可视化脚本

从测试集中随机选择9张图片，使用MLP模型进行预测，并以9宫格形式可视化结果。
"""

import os
import random
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 导入模型定义
import sys
sys.path.append(str(Path(__file__).parent / "scripts" / "training"))
from train_mlp import MLP, Config


def load_mlp_model(model_path: str = "models/mlp_pytorch.pt", device: torch.device = None) -> nn.Module:
    """
    加载训练好的MLP模型
    
    Args:
        model_path: 模型权重文件路径
        device: 计算设备，如果为None则自动选择
        
    Returns:
        加载好权重的MLP模型
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型
    model = MLP(dropout=Config.DROPOUT).to(device)
    
    # 加载权重
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"成功加载模型权重: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
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


def predict_images(model: nn.Module, images: List[Tuple[str, int, Image.Image]], 
                   transform: transforms.Compose, device: torch.device) -> List[Tuple[int, int, float]]:
    """
    对图片进行预测
    
    Args:
        model: MLP模型
        images: [(图片路径, 真实标签, PIL图片对象), ...]
        transform: 数据预处理变换
        device: 计算设备
        
    Returns:
        [(真实标签, 预测标签, 置信度), ...]
    """
    results = []
    
    with torch.no_grad():
        for img_path, true_label, pil_img in images:
            # 预处理图片
            img_tensor = transform(pil_img).unsqueeze(0).to(device)
            
            # 预测
            output = model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            pred_label = predicted.item()
            conf = confidence.item()
            
            results.append((true_label, pred_label, conf))
    
    return results


def visualize_predictions(images: List[Tuple[str, int, Image.Image]], 
                         results: List[Tuple[int, int, float]],
                         save_path: str = "predictions_visualization.png"):
    """
    以2x2形式可视化预测结果（新风格）
    
    Args:
        images: [(图片路径, 真实标签, PIL图片对象), ...]
        results: [(真实标签, 预测标签, 置信度), ...]
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), facecolor='#808080')
    fig.suptitle('MLP-Fashion-MNIST', fontsize=24, fontweight='bold', 
                 color='white', y=0.97)
    
    for idx, ((img_path, true_label, pil_img), (true, pred, conf)) in enumerate(zip(images, results)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        is_correct = (true == pred)
        # 蓝色表示正确，红色表示错误
        border_color = '#3498db' if is_correct else '#e74c3c'
        text_color = '#3498db' if is_correct else '#e74c3c'
        status_text = '✓' if is_correct else '✗'
        
        ax.set_facecolor('#ffffff')
        ax.imshow(pil_img, cmap='gray', aspect='auto')
        ax.axis('off')
        
        # 更粗的边框
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(5)
        
        # 信息显示在图片上方，字体更大
        title_text = f'Pred: {pred}  Target: {true}  {status_text}'
        ax.set_title(title_text, fontsize=18, fontweight='bold', 
                    color=text_color, pad=10)
        
        # 置信度显示在图片下方，字体更大
        conf_text = f'Confidence: {conf:.1%}'
        ax.text(0.5, -0.08, conf_text, transform=ax.transAxes, 
               fontsize=14, color='#ffffff', ha='center', va='top', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#808080')
    plt.close()
    print(f"可视化结果已保存到: {save_path}")


def main():
    """主函数"""
    # 使用系统时间作为随机种子
    seed = int(time.time())
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"随机种子: {seed}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print("\n正在加载MLP模型...")
    model_path = "models/mlp_pytorch.pt"
    model = load_mlp_model(model_path, device)
    
    # 准备数据变换
    transform = get_data_transforms()
    
    # 从测试集中随机选择4张图片
    print("\n正在从测试集中加载图片...")
    test_images = load_test_images("data/test", num_samples=4)
    print(f"成功加载 {len(test_images)} 张图片")
    
    # 进行预测
    print("\n正在进行预测...")
    predictions = predict_images(model, test_images, transform, device)
    
    # 统计准确率
    correct = sum(1 for true, pred, _ in predictions if true == pred)
    accuracy = correct / len(predictions)
    print(f"\n本次预测准确率: {accuracy:.2%} ({correct}/{len(predictions)})")
    
    # 可视化结果
    print("\n正在生成可视化...")
    visualize_predictions(test_images, predictions, save_path="predictions_visualization.png")


if __name__ == "__main__":
    main()

