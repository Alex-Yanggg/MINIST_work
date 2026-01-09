# Fashion-MNIST 衣着识别系统

基于 PyTorch 实现的 Fashion-MNIST 图像分类项目，提供从数据预处理、模型训练到 Web 服务部署的完整解决方案。

---

## 📌 项目概述

### 关于 Fashion-MNIST

[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) 是由 Zalando Research 发布的图像数据集，作为经典 MNIST 手写数字数据集的替代品。它包含 **70,000 张 28×28 灰度图像**，涵盖 10 个衣着类别，具有以下特点：

- **训练集**: 60,000 张图像
- **测试集**: 10,000 张图像
- **图像规格**: 28×28 像素，单通道灰度图
- **类别数量**: 10 类（T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴）

相比传统 MNIST，Fashion-MNIST 具有更高的分类难度，更适合作为机器学习算法的基准测试数据集。

### 项目特性

| 特性 | 说明 |
|------|------|
| **多模型支持** | 提供 MLP（多层感知机）和 CNN（卷积神经网络）两种架构 |
| **完整流水线** | 数据转换 → 模型训练 → Web 推理 → 容器化部署 |
| **高性能推理** | 基于 FastAPI 的异步 Web 服务，支持 GPU 加速 |
| **开箱即用** | 预置训练好的模型权重，可直接启动服务 |

### 技术栈

- **深度学习框架**: PyTorch 1.9+
- **Web 框架**: FastAPI + Uvicorn
- **图像处理**: OpenCV、Pillow
- **容器化**: Docker

---

## 🚀 快速启动

### 环境准备

```bash
# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 数据准备 & 训练 & 启动服务

```bash
# 1. 转换数据（将 idx-ubyte 转为 ImageFolder 格式）
python scripts/data_preparation/convert_mnist_train.py
python scripts/data_preparation/convert_mnist_test.py

# 2. 训练模型（CNN 约 3 分钟，准确率 ~91%）
python scripts/training/train_cnn_pytorch.py --model-type cnn --epochs 10

# 3. 启动 Web 服务
python scripts/deployment/app.py
```

### 测试 API

```bash
# 预处理用户图片（可选）
python scripts/data_preparation/preprocess_user_images.py

# 调用预测接口
curl -X POST "http://localhost:8000/predict" \
  -F "model=cnn" \
  -F "file=@data/user_samples/processed/0_001_processed.png"
```

访问 http://localhost:8000 可使用 Web 界面上传图片进行识别。

---

## 🧠 模型架构

### CNN（卷积神经网络）— 推荐

```
输入层     [batch, 1, 28, 28]
    ↓
Conv2D     32 filters, 5×5, padding=2, ReLU
MaxPool2D  2×2, stride=2                        → [batch, 32, 14, 14]
    ↓
Conv2D     64 filters, 5×5, padding=2, ReLU
MaxPool2D  2×2, stride=2                        → [batch, 64, 7, 7]
    ↓
Flatten                                         → [batch, 3136]
    ↓
Linear     3136 → 1024, ReLU, Dropout(0.2)
Linear     1024 → 10                            → [batch, 10]
```

**参数量**: ~3.3M | **测试准确率**: ~91%

### MLP（多层感知机）

```
输入层     [batch, 784]  (28×28 展平)
    ↓
Linear     784 → 512, ReLU, Dropout(0.2)
Linear     512 → 256, ReLU, Dropout(0.2)
Linear     256 → 128, ReLU, Dropout(0.2)
Linear     128 → 10                             → [batch, 10]
```

**参数量**: ~0.5M | **测试准确率**: ~89%

---

## 🏷️ 类别标签

| 标签 | 英文名称 | 中文名称 |
|:----:|----------|----------|
| 0 | T-shirt/top | T恤/上衣 |
| 1 | Trouser | 裤子 |
| 2 | Pullover | 套头衫 |
| 3 | Dress | 连衣裙 |
| 4 | Coat | 外套 |
| 5 | Sandal | 凉鞋 |
| 6 | Shirt | 衬衫 |
| 7 | Sneaker | 运动鞋 |
| 8 | Bag | 包 |
| 9 | Ankle boot | 短靴 |

---

## 📂 项目结构

```
MINIST_work/
├── scripts/
│   ├── data_preparation/       # 数据预处理脚本
│   │   ├── convert_mnist_train.py
│   │   ├── convert_mnist_test.py
│   │   └── preprocess_user_images.py
│   ├── training/               # 模型训练
│   │   └── train_cnn_pytorch.py
│   └── deployment/             # 服务部署
│       └── app.py
├── service/                    # FastAPI 服务模块
│   ├── app.py                  # 应用工厂
│   ├── inference.py            # 推理逻辑
│   ├── models.py               # 模型加载
│   └── preprocessing.py        # 图像预处理
├── data/
│   ├── raw/                    # 原始 idx-ubyte 文件
│   ├── train/                  # ImageFolder 训练集 (0~9/)
│   ├── test/                   # ImageFolder 测试集 (0~9/)
│   └── user_samples/           # 用户测试图片
├── models/                     # 模型权重 (.pt)
├── docs/                       # 文档
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🔧 进阶使用

### 训练参数

```bash
# 完整参数列表
python scripts/training/train_cnn_pytorch.py --help

# 常用示例
--model-type cnn|mlp    # 模型类型
--epochs 20             # 训练轮数
--batch-size 64         # 批次大小
--lr 0.001              # 学习率
--dropout 0.3           # Dropout 比例
--no-amp                # 禁用混合精度（CPU 训练时建议）
--num-workers 0         # DataLoader 工作进程数（Windows 建议设为 0）
```

### 可视化预测

```bash
python visualize_predictions.py
```

生成 `predictions_visualization.png`，展示 9 张随机测试图片的预测结果。

### Docker 部署

```bash
docker build -t fashion-mnist-service .
docker run -p 8000:8000 fashion-mnist-service
```

---

## 📊 API 说明

### `GET /`

返回 Web 界面，支持上传图片进行可视化预测。

### `POST /predict`

**请求参数**:
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| model | string | 否 | `cnn`（默认）或 `mlp` |
| file | file | 是 | 图片文件（PNG/JPG） |

**响应示例**:
```json
{
  "success": true,
  "model": "cnn",
  "result": {
    "prediction": 7,
    "confidence": "95.32%",
    "top_predictions": [7, 9, 5]
  },
  "message": "预测结果: 7 (置信度: 95.32%)"
}
```

---

## ❓ 常见问题

| 问题 | 解决方案 |
|------|----------|
| 找不到训练数据 | 确认 `data/raw/` 下有 4 个 `*-idx*-ubyte` 文件，然后运行数据转换脚本 |
| 模型文件不存在 | 先运行训练脚本生成 `models/*.pt`，或使用预训练权重 |
| CUDA 不可用 | 系统会自动降级为 CPU 模式，训练时建议加 `--no-amp` |
| Windows 虚拟环境激活失败 | 使用 `.venv\Scripts\activate`，PowerShell 需调整执行策略 |

---

## 📚 参考资料

- [Fashion-MNIST 官方仓库](https://github.com/zalandoresearch/fashion-mnist)
- [PyTorch 官方文档](https://pytorch.org/docs/)
- [FastAPI 官方文档](https://fastapi.tiangolo.com/)

---

## 📖 扩展文档

- **数据格式详解**: [docs/data_format_guide.md](docs/data_format_guide.md)
- **实验指导书**: [docs/experiment_guide.md](docs/experiment_guide.md)

> **备注**: 仓库中部分脚本/文件名沿用历史命名（如 `convert_mnist_*.py`），实际处理的是 Fashion-MNIST 数据集。
