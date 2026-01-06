from __future__ import annotations

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .html import HTML
from .inference import predict
from .models import load_models
from .preprocessing import load_image


def create_app() -> FastAPI:
    """构建 FastAPI 实例并装配路由。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(device)

    app = FastAPI(title="MNIST Web Service", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return HTML

    @app.post("/predict", response_class=JSONResponse)
    async def handle_predict(
        model: str = Form("cnn"),
        file: UploadFile = File(...),
    ):
        if model not in models:
            raise HTTPException(status_code=400, detail=f"不支持的模型: {model}")
        if not file.filename:
            raise HTTPException(status_code=400, detail="未选择文件")

        try:
            file_bytes = await file.read()
            tensor = load_image(file_bytes)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"图片解析失败: {e}")

        model_obj, weight_path = models[model]
        result = predict(model_obj, tensor, device)

        return {
            "success": True,
            "model": model,
            "weights": weight_path,
            "device": str(device),
            "result": {
                "prediction": result["prediction"],
                "confidence": result["confidence_percent"],
                "top_predictions": result["top_predictions"],
                "all_probabilities": result["all_probabilities"],
            },
            "message": f"预测结果: 数字 {result['prediction']} (置信度: {result['confidence_percent']})",
        }

    return app

