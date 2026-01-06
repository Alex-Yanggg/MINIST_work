from __future__ import annotations


import os
from typing import Dict, Tuple

import torch


from train_cnn_pytorch import Config as CNNConfig, create_model as create_cnn_model
# from train_mlp_pytorch import Config as MLPConfig, create_model as create_mlp_model

def load_models(device: torch.device) -> Dict[str, Tuple[torch.nn.Module, str]]:
    """加载 CNN 与 MLP 模型权重，返回模型字典。"""
    models: Dict[str, Tuple[torch.nn.Module, str]] = {}

    cnn = create_cnn_model(dropout=CNNConfig.DROPOUT, device=device)
    if os.path.exists(CNNConfig.MODEL_SAVE_PATH):
        state = torch.load(CNNConfig.MODEL_SAVE_PATH, map_location=device)
        cnn.load_state_dict(state)
    else:
        print(f"[警告] 未找到权重文件: {CNNConfig.MODEL_SAVE_PATH}，将使用随机初始化权重。")
    cnn.eval()
    models["cnn"] = (cnn, CNNConfig.MODEL_SAVE_PATH)

    # mlp = create_mlp_model(dropout=MLPConfig.DROPOUT, device=device)
    # if os.path.exists(MLPConfig.MODEL_SAVE_PATH):
    #     state = torch.load(MLPConfig.MODEL_SAVE_PATH, map_location=device)
    #     mlp.load_state_dict(state)
    # else:
    #     print(f"[警告] 未找到权重文件: {MLPConfig.MODEL_SAVE_PATH}，将使用随机初始化权重。")
    # mlp.eval()
    # models["mlp"] = (mlp, MLPConfig.MODEL_SAVE_PATH)

    return models

