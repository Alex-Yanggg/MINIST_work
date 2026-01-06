# MNIST手写数字识别项目

基于PyTorch实现的完整MNIST手写数字识别系统，支持多模型架构、Web服务部署和Docker容器化。

## 快速开始

### 1. 环境准备
####  创建虚拟环境
```
python -m venv .venv
```
#### 激活虚拟环境
```cmd
.\.venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

### 2. 数据转换
```bash
cd scripts/data_preparation
python convert_mnist_train.py
python convert_mnist_test.py
```

### 3. 模型训练
```bash
cd scripts/training
# 训练MLP模型
python train_mlp_pytorch.py

# 训练CNN模型
python train_cnn_pytorch.py
```

### 4. 启动服务
```bash
cd scripts/deployment
python app.py
```

访问 http://localhost:8000 使用Web界面进行数字识别。

## 项目结构

```
├── scripts/                    # 所有脚本文件
│   ├── data_preparation/       # 数据准备脚本
│   │   ├── convert_mnist_train.py
│   │   ├── convert_mnist_test.py
│   │   └── preprocess_user_images.py
│   ├── training/               # 模型训练脚本
│   │   ├── train_mlp_numpy.py
│   │   ├── train_mlp_pytorch.py
│   │   └── train_cnn_pytorch.py
│   └── deployment/             # 部署脚本
│       └── app.py
├── service/                    # Web服务模块
│   ├── __init__.py
│   ├── app.py
│   ├── html.py
│   ├── inference.py
│   ├── models.py
│   └── preprocessing.py
├── models/                     # 训练好的模型文件
│   ├── mlp_numpy.pkl
│   ├── mlp_pytorch.pt
│   └── cnn_pytorch.pt
├── data/                       # 数据目录
│   ├── raw/                    # 原始MNIST数据
│   ├── train/                  # 训练集图片
│   ├── test/                   # 测试集图片
│   └── user_samples/           # 用户手写样本
├── docs/                       # 文档
│   ├── experiment_guide.md     # 实验指导书
│   └── data_format_guide.md    # 数据格式说明
├── requirements.txt            # Python依赖
├── Dockerfile                  # Docker配置
└── README.md                   # 项目说明
```

## Docker部署

```bash
# 构建镜像
docker build -t mnist-service .

# 运行容器
docker run -p 8000:8000 mnist-service
```

## API使用

```bash
# 使用curl测试API
curl -X POST "http://localhost:8000/predict" \
     -F "model=cnn" \
     -F "file=@path/to/image.png"
```

## 实验报告

详细的实验指导和分析请参考[docs/experiment_guide.md](./docs/experiment_guide.md)。
