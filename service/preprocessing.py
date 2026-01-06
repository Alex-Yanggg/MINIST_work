from __future__ import annotations

import io
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF


def build_transforms() -> transforms.Compose:
    """与训练保持一致的数据预处理。"""
    return transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def _apply_mnist_normalize(tensor: torch.Tensor) -> torch.Tensor:
    """复用训练时的 Normalize 设置，保持分布一致。"""
    return TF.normalize(tensor, [0.1307], [0.3081])


def basic_mnist_transform(image_input: Any, normalize: bool = True, reverse_color: bool = True) -> torch.Tensor:
    """
    简化路径：假定输入较干净，直接缩放到28x28并标准化，不做轮廓提取。
    支持文件路径 / 字节 / PIL.Image。
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("L")
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise ValueError(f"不支持的输入类型: {type(image_input)}，应为 bytes、str 或 PIL.Image")

    img = img.resize((28, 28), Image.LANCZOS)
    mnist_array = np.array(img, dtype=np.float32)

    if reverse_color:
        mnist_array = 255.0 - mnist_array
    if normalize:
        mnist_array = mnist_array / 255.0

    tensor = torch.from_numpy(mnist_array).unsqueeze(0)
    tensor = _apply_mnist_normalize(tensor)
    return tensor.unsqueeze(0)


def robust_mnist_preprocess(
    image_input: Any,
    normalize: bool = True,
    reverse_color: bool = True,
    use_basic_fallback: bool = True,
) -> torch.Tensor:
    """
    鲁棒预处理：去噪、阈值、轮廓裁剪、居中，对真实拍摄/噪声图像更稳健。
    失败时可回退到 basic_mnist_transform，保证服务可用。
    """
    if isinstance(image_input, bytes):
        img = Image.open(io.BytesIO(image_input)).convert("L")
    elif isinstance(image_input, str):
        img = Image.open(image_input).convert("L")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("L")
    else:
        raise ValueError(f"不支持的输入类型: {type(image_input)}，应为 bytes、str 或 PIL.Image")

    gray = np.array(img, dtype=np.uint8)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    try:
        _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("未找到数字轮廓")
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = binary[y : y + h, x : x + w]

        aspect_ratio = w / h if h != 0 else 1.0
        if aspect_ratio > 1:
            new_w = 28
            new_h = max(1, int(new_w / aspect_ratio))
        else:
            new_h = 28
            new_w = max(1, int(new_h * aspect_ratio))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        padded = np.zeros((28, 28), dtype=np.uint8)
        offset_x = (28 - new_w) // 2
        offset_y = (28 - new_h) // 2
        padded[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized

        mnist_array = np.array(padded, dtype=np.float32)
    except Exception:
        if not use_basic_fallback:
            raise
        return basic_mnist_transform(image_input, normalize=normalize, reverse_color=reverse_color)

    if reverse_color:
        mnist_array = 255.0 - mnist_array
    if normalize:
        mnist_array = mnist_array / 255.0

    tensor = torch.from_numpy(mnist_array).unsqueeze(0)
    tensor = _apply_mnist_normalize(tensor)
    return tensor.unsqueeze(0)


def load_image(file_bytes: bytes, reverse_color: bool = False) -> torch.Tensor:
    """将上传的图片字节转为网络输入张量，自动判断清洁度选择预处理路径。"""
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    gray = np.array(img, dtype=np.uint8)

    if _should_use_basic(gray):
        return basic_mnist_transform(img, normalize=True, reverse_color=reverse_color)

    return robust_mnist_preprocess(img, normalize=True, reverse_color=reverse_color)


def _should_use_basic(gray: np.ndarray) -> bool:
    """
    简单启发式判断图片是否已近似“干净 28x28”：
    - 尺寸正好 28x28
    - 轮廓面积占比在合理范围且轮廓数量少
    - 边缘密度低，噪点少
    """
    h, w = gray.shape[:2]

    # 直接符合目标尺寸
    if h == 28 and w == 28:
        return True

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= 4]

    if not valid_contours:
        return False

    largest = max(valid_contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / float(h * w)
    num_contours = len(valid_contours)

    edges = cv2.Canny(blurred, 50, 150)
    edge_ratio = edges.sum() / 255.0 / (h * w)

    # 合理的面积占比与轮廓数量，且边缘密度不高时认为是“干净”图
    if 0.05 <= area_ratio <= 0.65 and num_contours <= 2 and edge_ratio < 0.08:
        return True

    return False

