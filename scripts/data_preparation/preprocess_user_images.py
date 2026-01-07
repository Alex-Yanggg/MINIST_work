import cv2
import numpy as np
import os
from PIL import Image

# 配置参数
INPUT_DIR = "../../data/user_samples"  # 原始衣物图片存放目录
OUTPUT_DIR = "../../data/user_samples/processed"  # 处理后数据集输出目录
TARGET_SIZE = (28, 28)  # 目标图像尺寸（MNIST标准）
THRESHOLD_VALUE = 127  # 二值化阈值



def preprocess_single_image(img_path):
    """预处理单张图片：灰度化、二值化、归一化、居中对齐"""
    # 1. 读取图片并转为灰度图
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 去噪（高斯滤波）
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # 3. 二值化（反色，因为MNIST背景为黑、数字为白）
    _, binary = cv2.threshold(blurred, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 4. 提取数字轮廓并裁剪
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    cropped = binary[y:y+h, x:x+w]
    
    # 5. 调整尺寸并居中对齐
    # 保持长宽比，先缩放至目标尺寸的短边
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = TARGET_SIZE[0]
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = TARGET_SIZE[1]
        new_w = int(new_h * aspect_ratio)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 创建空白画布并居中放置数字
    padded = np.zeros(TARGET_SIZE, dtype=np.uint8)
    offset_x = (TARGET_SIZE[0] - new_w) // 2
    offset_y = (TARGET_SIZE[1] - new_h) // 2
    padded[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = resized
    
    return padded


def main():
    # 遍历输入目录的图片（假设文件名格式为 数字_序号.jpg，如 5_001.jpg）
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith((".jpg", ".png", ".jpeg")):
            continue
        # 从文件名提取标签
        img_path = os.path.join(INPUT_DIR, filename)
        
        # 预处理图片
        processed_img = preprocess_single_image(img_path)
        if processed_img is not None:
            # 保存处理后的图片到output/processed文件夹
            processed_dir = os.path.join(INPUT_DIR, "processed")
            os.makedirs(processed_dir, exist_ok=True)
            save_name = f"{os.path.splitext(filename)[0]}_processed.png"
            save_path = os.path.join(processed_dir, save_name)
            Image.fromarray(processed_img).save(save_path)

if __name__ == "__main__":
    main()