# MNIST手写数字识别实验项目指导书

## 项目概述

本实验项目实现了一个完整的MNIST手写数字识别系统，包含数据预处理、模型训练、服务部署和Docker容器化等完整流程。项目支持两种模型架构（MLP多层感知机和CNN卷积神经网络），并提供Web服务接口供用户上传手写数字图片进行识别。

### 项目特色
- **多模型支持**：提供MLP和CNN两种模型架构
- **完整工作流**：从数据转换到模型部署的完整流程
- **鲁棒预处理**：支持多种用户手写图片预处理方法
- **Web服务**：基于FastAPI的RESTful API服务
- **Docker化**：支持容器化部署

## 实验环境要求

### 硬件要求
- CPU: 支持x86_64架构
- 内存: 至少4GB RAM
- 存储: 至少5GB可用空间

### 软件要求
- Python 3.8+
- PyTorch 1.9+
- CUDA (可选，用于GPU加速)
- Docker (用于容器化部署)

### Python依赖包
```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.0.0
fastapi>=0.68.0
uvicorn>=0.15.0
python-multipart>=0.0.5
```

## 第一部分：MNIST数据集转换

### 1.1 数据集格式说明

MNIST数据集包含4个二进制文件：
- `train-images.idx3-ubyte`: 训练集图片（60,000个样本）
- `train-labels.idx1-ubyte`: 训练集标签
- `t10k-images.idx3-ubyte`: 测试集图片（10,000个样本）
- `t10k-labels.idx1-ubyte`: 测试集标签

每个图片为28×28像素的灰度图像，像素值范围0-255。

### 1.2 数据转换脚本

运行训练集转换：
```bash
cd scripts/data_preparation
python convert_mnist_train.py
```

运行测试集转换：
```bash
python convert_mnist_test.py
```

转换后的数据将保存在`data/train`和`data/test`目录中，每个数字（0-9）对应一个子目录。

### 1.3 验证转换结果

检查转换是否成功：
```bash
ls -la data/train/  # 应看到0-9共10个子目录
ls -la data/test/   # 应看到0-9共10个子目录
ls data/train/0/ | wc -l  # 检查训练集中数字0的样本数量
```

## 第二部分：模型训练

### 2.1 MLP模型训练

#### 2.1.1 NumPy实现的MLP
```bash
cd scripts/training
python train_mlp_numpy.py
```

该脚本使用纯NumPy实现三层MLP网络：
- 输入层：784节点（28×28）
- 隐藏层：128节点
- 输出层：10节点（数字0-9）

#### 2.1.2 PyTorch实现的MLP
```bash
python train_mlp_pytorch.py
```

PyTorch版本支持更多特性：
- 可配置的隐藏层结构（默认512-256-128）
- Dropout正则化
- GPU加速支持
- 自动混合精度训练

训练参数可以通过命令行调整：
```bash
# 使用不同的学习率和批次大小
python train_mlp_pytorch.py --lr 0.001 --batch-size 64 --epochs 15

# 在CPU上训练（默认自动选择GPU）
python train_mlp_pytorch.py --num-workers 0 --no-amp
```

### 2.2 CNN模型训练

```bash
python train_cnn_pytorch.py
```

CNN模型架构：
- 卷积层1：32个5×5卷积核
- 最大池化：2×2
- 卷积层2：64个5×5卷积核
- 最大池化：2×2
- 全连接层：1024→10

CNN模型同样支持命令行参数配置。

### 2.3 训练结果

训练完成后会在`models/`目录下生成：
- `mlp_pytorch.pt`: PyTorch MLP模型权重
- `mlp_numpy.pkl`: NumPy MLP模型权重
- `cnn_pytorch.pt`: CNN模型权重

训练日志会显示每个epoch的训练损失、准确率和测试准确率。

## 第三部分：用户手写图片预处理

### 3.1 预处理流程

用户手写图片预处理脚本`userimgTomnist.py`实现了以下步骤：
1. 灰度化转换
2. 高斯滤波去噪
3. 自适应二值化
4. 轮廓提取和裁剪
5. 尺寸调整和居中对齐

### 3.2 运行预处理

```bash
cd scripts/data_preparation
python preprocess_user_images.py
```

该脚本会处理`data/user_samples`目录中的所有图片，并在`processed`子目录中生成处理后的图片。

### 3.3 预处理效果验证

查看预处理效果：
```bash
ls userhandwritten_digits/processed/
```

处理后的图片将统一为28×28像素，数字居中对齐，背景为黑色，数字为白色，符合MNIST格式要求。

## 第四部分：模型服务部署

### 4.1 服务架构

项目提供基于FastAPI的Web服务，支持：
- 模型选择（MLP/CNN）
- 图片上传和识别
- 批量预测结果
- 置信度分析

### 4.2 启动Web服务

```bash
cd scripts/deployment
python app.py
```

服务将在http://localhost:8000启动，提供Web界面和REST API。

### 4.3 API接口说明

#### 预测接口
- **URL**: `POST /predict`
- **参数**:
  - `model`: 模型选择（"mlp" 或 "cnn"）
  - `file`: 上传的图片文件
- **返回**: JSON格式预测结果

#### Web界面
访问http://localhost:8000可使用浏览器上传图片进行识别。

### 4.4 服务测试

使用curl测试API：
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "model=cnn" \
     -F "file=@userhandwritten_digits/processed/5_001_processed.png"
```

## 第五部分：Docker容器化部署

### 5.1 创建Dockerfile

创建`Dockerfile`：
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "web_service.py"]
```

### 5.2 创建requirements.txt

```txt
fastapi>=0.68.0
uvicorn[standard]>=0.15.0
python-multipart>=0.0.5
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.21.0
torch>=1.9.0
torchvision>=0.10.0
```

### 5.3 构建和运行Docker容器

```bash
# 构建镜像
docker build -t mnist-service .

# 运行容器
docker run -p 8000:8000 mnist-service
```

### 5.4 验证Docker部署

```bash
# 检查容器状态
docker ps

# 测试服务
curl http://localhost:8000/
```

## 第六部分：实验报告和分析

### 6.1 性能对比

记录不同模型的训练时间、准确率等指标：

| 模型 | 训练时间 | 测试准确率 | 模型大小 |
|------|----------|------------|----------|
| NumPy MLP | ~30秒 | ~92% | ~1MB |
| PyTorch MLP | ~2分钟 | ~97% | ~2MB |
| CNN | ~3分钟 | ~98% | ~3MB |

### 6.2 用户图片识别效果

测试用户手写图片的识别效果，分析：
- 预处理效果对识别准确率的影响
- 不同书写风格的识别难度
- 模型的鲁棒性分析

### 6.3 部署效果评估

- 服务响应时间测试
- 并发处理能力评估
- Docker容器资源占用分析

## 第七部分：实验总结和扩展

### 7.1 实验收获

通过本实验，掌握：
1. MNIST数据集处理流程
2. 深度学习模型训练调参
3. PyTorch框架使用
4. Web服务开发部署
5. Docker容器化技术

### 7.2 可能的扩展方向

1. **模型优化**：
   - 尝试更深的网络结构
   - 集成学习方法
   - 注意力机制

2. **数据增强**：
   - 旋转、缩放、扭曲等数据增强
   - 生成对抗网络(GAN)数据合成

3. **服务增强**：
   - 批量预测接口
   - 模型热更新
   - 监控和日志系统

4. **部署优化**：
   - Kubernetes集群部署
   - GPU加速推理
   - 边缘计算部署

## 附录：常见问题解决

### Q1: 训练过程中内存不足
A: 减小batch_size参数，或使用CPU训练。

### Q2: 图片预处理失败
A: 检查图片格式，确保为常见格式（PNG/JPG），或调整阈值参数。

### Q3: Docker构建失败
A: 确保网络连接正常，或使用国内镜像源。

### Q4: 模型识别准确率低
A: 检查预处理效果，尝试不同的模型，或增加训练数据。

---

**实验完成标志**：
- [ ] 数据集转换成功
- [ ] 至少一种模型训练完成，准确率>95%
- [ ] 用户图片预处理脚本运行成功
- [ ] Web服务正常启动并能识别图片
- [ ] Docker容器化部署成功
